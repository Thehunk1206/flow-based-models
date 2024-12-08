import copy
import math
import os
import argparse

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm, trange

from torchvision.utils import save_image

from utils.data_loader import UnlabeledImageDataset
from utils.sampler import euler_solver, heun_solver
from conditional_flow_matcher import ConditionalFlowMatcher, OptimalTransportConditionalFlowMatcher
from models.unet_model import UNetModelWrapper

from loguru import logger
from typing import Optional


# Generate Validation Samples
def generate_samples(
    model, 
    accelerator,
    savedir: str, 
    step: int, 
    image_size: tuple = (128, 128), 
    net_: str = "normal", 
    grid_size: tuple = (5, 5), 
    method: str = 'euler'
):
    """
    Generate samples using specified integration method with Accelerate support.

    Parameters
    ----------
    model: torch.nn.Module
        Neural network for generating samples.
    accelerator: Accelerator
        Hugging Face Accelerate wrapper.
    savedir: str
        Path to save generated images.
    step: int
        Current training step.
    image_size: tuple
        Size of generated images.
    net_: str
        Network type identifier.
    grid_size: tuple
        Grid size for arranging generated images.
    method: str
        Integration method ('euler' or 'heun').
    """
    _supported_methods = ["euler", "heun"]
    
    # Ensure the model is in evaluation mode
    model.eval()

    # Prepare integration parameters
    with torch.no_grad():
        # Adjust the tensor size based on grid_size
        num_images = grid_size[0] * grid_size[1]
        
        # Generate initial random noise
        x0 = torch.randn(num_images, 3, image_size[0], image_size[1])
        x0 = x0.to(accelerator.device)
        
        # Create time span for integration
        t_span = torch.linspace(0, 1, 100, device=accelerator.device)
        
        # Choose integration method
        if method == 'euler':
            traj = euler_solver(model, x0, t_span, device=accelerator.device)
        elif method == 'heun':
            traj = heun_solver(model, x0, t_span, device=accelerator.device)
        else:
            raise ValueError(f"Unsupported integration method: {method}, supported methods: {_supported_methods}")
        
        # Post-process the trajectory
        traj = traj.view([-1, 3, image_size[0], image_size[1]]).clip(-1, 1)
        traj = traj / 2 + 0.5
    
    # Prepare for saving (move to CPU if needed)
    if accelerator.is_main_process:
        # Ensure traj is on CPU for saving
        traj = traj.cpu()
        
        # Save the generated images
        save_path = os.path.join(savedir, f"{net_}_generated_FM_images_step_{step}.png")
        save_image(traj, save_path, nrow=grid_size[0])

    model.train()

def ema(source: torch.nn.Module, target: torch.nn.Module, decay: float):
    """Exponential Moving Average update."""
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )

def infiniteloop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x

def warmup_lr(current_step: int, warmup_steps: int) -> float:
    """
    Linear learning rate warmup.
    
    Parameters:
    -----------
    current_step : int
        Current training step.
    warmup_steps : int
        Number of warmup steps.
    
    Returns:
    --------
    float
        Learning rate scaling factor.
    """
    return min(current_step, warmup_steps) / warmup_steps

def train(
    *,
    output_dir: str = "./outputs/results_flickr_exp3/",
    train_data_dir: str = "/disk1/BharatDiffusion/kohya_ss/experimental_sricpts/real_faces_128",
    model_type: str = "otcfm",
    ot_method: str = "exact",
    sigma: float = 0.0,
    num_channel: int = 128,
    learning_rate: float = 1e-4,
    grad_clip: float = 1.0,
    total_steps: int = 1000001,
    warmup_steps: int = 10000,
    batch_size: int = 32,
    ema_decay: float = 0.9999,
    save_step: int = 10000,
    seed: Optional[int] = None,
    mixed_precision: str = "bf16",
    gradient_accumulation_steps: int = 4,
    report_with:str = "tensorboard",
    project_name: str = "conditional_flow_matching"
):
    """
    Main training loop.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save outputs.
    train_data_dir : str
        path to training dataset directory
    model_type : str
        Type of flow matching model.
    ot_method: str
        Optimal Transport Method. one of ["exact", "sinkhorn", "unbalanced", "partial"]
    sigma : float
        Standard deviation for flow matching.
    num_channel : int
        Base channel count for UNet.
    learning_rate : float
        Initial learning rate.
    grad_clip : float
        Gradient clipping norm.
    total_steps : int
        Total training steps.
    warmup_steps : int
        Learning rate warmup steps.
    batch_size : int
        Training batch size.
    ema_decay : float
        EMA decay rate.
    save_step : int
        Frequency of saving checkpoints.
    seed : Optional[int]
        Random seed for reproducibility.
    mixed_precision : str
        Mixed precision training type.
    gradient_accumulation_steps : int
        Number of gradient accumulation steps.
    report_with : str
        Logging framework to use.
    project_name: str
        Project Name
    """

    if not os.path.exists(train_data_dir):
        raise ValueError(f"train_data_dir {train_data_dir} does not exist.")
    
    # set weight dtype
    if mixed_precision == "bf16":
        weigth_dtype = torch.bfloat16
    elif mixed_precision == "fp16":
        weigth_dtype = torch.float16
    else:
        weigth_dtype = torch.float32
    
    # Initialize Accelerate object
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with=report_with,
        project_dir=output_dir,
    )

    if seed is not None:
        set_seed(seed)

    # Logging
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        accelerator.init_trackers(project_name=project_name)
    
    # Prepare data transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset
    dataset = UnlabeledImageDataset(
        image_dir=train_data_dir,
        transform=transform
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )

    # calc total_epochs and steps per epoch
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    num_epochs = math.ceil(total_steps / steps_per_epoch)

    # Initialize model
    net_model = UNetModelWrapper(
        dim=(3, 128, 128),
        num_res_blocks=2,
        num_channels=num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1
    ).to(device=accelerator.device, dtype=weigth_dtype)

    # Create EMA model
    ema_model = copy.deepcopy(net_model)

    # Initialize the Flow matcher type
    if model_type == "otcfm":
        FM = OptimalTransportConditionalFlowMatcher(sigma=sigma, ot_method=ot_method)
    elif model_type == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {model_type}, must be one of ['otcfm', 'icfm']"
        )
    
        # Optimizer
    optim = torch.optim.AdamW(net_model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, 
        lr_lambda=lambda step: warmup_lr(step, warmup_steps)
    )

    # Prepare with Accelerator
    net_model, ema_model, optim, lr_scheduler, dataloader = accelerator.prepare(
        net_model, ema_model, optim, lr_scheduler, dataloader
    )

    # Model size logging
    model_size = sum(p.numel() for p in net_model.parameters())
    accelerator.print(f"Model params: {model_size / 1024 / 1024:.2f} M")

    # Training loop
    global_step = 0

    for epoch in trange(num_epochs, desc="Epochs", disable=not accelerator.is_main_process):
        net_model.train()
        
        # Create a progress bar with loss and lr information
        batch_iterator = tqdm(
            dataloader, 
            desc="Batches", 
            disable=not accelerator.is_main_process,
            bar_format='{l_bar}{bar:10}{r_bar}'
        )
        
        for batch in batch_iterator:
            with accelerator.accumulate(net_model):
                # Prepare batch
                x1 = batch
                x0 = torch.randn_like(x1)

                # Flow Matching sampling
                t, xt, ut = FM.get_sample_location_and_conditional_flow(x0, x1)

                # cast to weight dtype
                x0 = x0.to(dtype=weigth_dtype, device=accelerator.device)
                x1 = x1.to(dtype=weigth_dtype, device=accelerator.device)
                t  = t.to(dtype=weigth_dtype, device=accelerator.device)
                xt = xt.to(dtype=weigth_dtype, device=accelerator.device)
                ut = ut.to(dtype=weigth_dtype, device=accelerator.device)

                # Forward pass
                vt = net_model(t, xt)
                loss = F.mse_loss(vt.float(), ut.float())

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_model.parameters(), grad_clip)

                # Optimizer step
                optim.step()
                lr_scheduler.step()
                optim.zero_grad()

                # EMA update
                # TODO: Check if the ema's are correct
                if accelerator.sync_gradients and accelerator.is_main_process: 
                    ema(net_model, ema_model, ema_decay)

                # Update progress bar
                current_lr = lr_scheduler.get_last_lr()[0]
                batch_iterator.set_postfix({
                    'Loss': f'{loss.item():.4f}', 
                    'LR': f'{current_lr:.6f}'
                })

                # Logging
                if global_step % 100 == 0:
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/lr": current_lr
                    }, step=global_step)

                # Sampling and checkpoint
                if save_step > 0 and global_step % save_step == 0:
                    # Generate samples
                    generate_samples(
                        net_model, 
                        accelerator, 
                        output_dir, 
                        global_step, 
                        image_size=(128,128), 
                        net_="normal", 
                        grid_size=(5,5)
                    )
                    generate_samples(
                        ema_model, 
                        accelerator, 
                        output_dir, 
                        global_step, 
                        image_size=(128,128), 
                        net_="ema", 
                        grid_size=(5,5)
                    )

                    # Save checkpoint
                    if accelerator.is_main_process:
                        checkpoint = {
                            "net_model": accelerator.unwrap_model(net_model).state_dict(),
                            "ema_model": accelerator.unwrap_model(ema_model).state_dict(),
                            "optimizer": optim.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "step": global_step
                        }
                        torch.save(
                            checkpoint, 
                            os.path.join(output_dir, f"{model_type}_weights_step_{global_step}.pt")
                        )

                global_step += 1

                # Break if total steps reached
                if global_step >= total_steps:
                    break

            if global_step >= total_steps:
                break
    # End training
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Conditional Flow Matching Training Script")
    
    # Training configuration
    parser.add_argument("--output_dir", type=str, default="./outputs/results_flickr_exp3/", 
                        help="Directory to save outputs")
    parser.add_argument("--train_data_dir", type=str, 
                        default="/disk1/BharatDiffusion/kohya_ss/experimental_sricpts/real_faces_128", 
                        help="Path to training dataset directory")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, choices=["otcfm", "icfm"], default="otcfm", 
                        help="Type of flow matching model")
    parser.add_argument("--ot_method", type=str, default="exact", 
                        choices=["exact", "sinkhorn", "unbalanced", "partial"],
                        help="Optimal Transport Method")
    parser.add_argument("--sigma", type=float, default=0.0, 
                        help="Standard deviation for flow matching")
    parser.add_argument("--num_channel", type=int, default=128, 
                        help="Base channel count for UNet")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Initial learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, 
                        help="Gradient clipping norm")
    parser.add_argument("--total_steps", type=int, default=1000001, 
                        help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=10000, 
                        help="Learning rate warmup steps")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Training batch size")
    parser.add_argument("--ema_decay", type=float, default=0.9999, 
                        help="EMA decay rate")
    parser.add_argument("--save_step", type=int, default=10000, 
                        help="Frequency of saving checkpoints")
    
    # Training environment
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--mixed_precision", type=str, default="bf16", 
                        choices=["no", "fp16", "bf16"], 
                        help="Mixed precision training type")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Number of gradient accumulation steps")
    parser.add_argument("--report_with", type=str, default="tensorboard", 
                        choices=["tensorboard", "wandb"], 
                        help="Logging framework to use")
    parser.add_argument("--project_name", type=str, default="conditional_flow_matching", 
                        help="Project Name")
    
    # Parse arguments
    args = parser.parse_args()
    
    train(
        output_dir=args.output_dir,
        train_data_dir=args.train_data_dir,
        model_type=args.model_type,
        ot_method=args.ot_method,
        sigma=args.sigma,
        num_channel=args.num_channel,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        ema_decay=args.ema_decay,
        save_step=args.save_step,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_with=args.report_with,
        project_name=args.project_name
    )

if __name__ == "__main__":
    main()
