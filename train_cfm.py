import argparse
import copy
import math
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from utils.data_loader import UnlabeledImageDataset
from utils.train_utils import (generate_samples, 
                                find_latest_checkpoint, 
                                cleanup_old_checkpoints, 
                                ema, infiniteloop, 
                                warmup_lr
                            )

from conditional_flow_matcher import ConditionalFlowMatcher, OptimalTransportConditionalFlowMatcher
from models.unet_model import UNetModelWrapper


# ### Set Environment Variables ###
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WORLD_SIZE"] = "1"

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Flow Matching Training Script")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="otcfm", choices=["otcfm", "icfm"], 
                        help="Flow matching model type")
    parser.add_argument("--output_dir", type=str, default="./outputs/results_flickr_exp3/", 
                        help="Output directory")
    
    # UNet configuration
    parser.add_argument("--num_channel", type=int, default=128, 
                        help="Base channel of UNet")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Target learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, 
                        help="Gradient norm clipping")
    parser.add_argument("--total_steps", type=int, default=1000001, 
                        help="Total training steps")
    parser.add_argument("--warmup", type=int, default=10000, 
                        help="Learning rate warmup steps")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of workers for DataLoader")
    parser.add_argument("--ema_decay", type=float, default=0.9999, 
                        help="EMA decay rate")
    
    # Evaluation parameters
    parser.add_argument("--save_step", type=int, default=10000, 
                        help="Frequency of saving checkpoints (0 to disable)")
    
    # Image dataset parameters
    parser.add_argument("--image_dir", type=str, 
                        default="/disk1/BharatDiffusion/kohya_ss/experimental_sricpts/real_faces_128", 
                        help="Directory containing training images")
    
    # Logging parameters
    parser.add_argument("--log_dir", type=str, default="./logs", 
                        help="TensorBoard log directory")
    
    # last n checkpoints to save, delete the rest checkpoints for saving the disk space
    parser.add_argument("--keep_n_checkpoints", type=int, default=10,
                        help="Number of previous checkpoints to keep")
    
    return parser.parse_args()


def train(args):
    """Main training function."""
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output and log directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset and DataLoader
    dataset = UnlabeledImageDataset(
        image_dir=args.image_dir,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    # Calculate number of epochs
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    num_epochs = math.ceil(args.total_steps / steps_per_epoch)

    # Model initialization
    net_model = UNetModelWrapper(
        dim=(3, 128, 128),
        num_res_blocks=2,
        num_channels=args.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    # Model size logging
    model_size = sum(p.data.nelement() for p in net_model.parameters())
    print(f"Model params: {model_size / 1024 / 1024:.2f} M")

    # EMA Model
    ema_model = copy.deepcopy(net_model)

    # Optimizer and Scheduler
    optim = torch.optim.AdamW(net_model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: warmup_lr(step, args.warmup))

    # Flow Matcher initialization
    sigma = 0.0
    if args.model == "otcfm":
        FM = OptimalTransportConditionalFlowMatcher(sigma=sigma, ot_method='exact')
    elif args.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {args.model}, must be one of ['otcfm', 'icfm']"
        )

    # Directories
    savedir = os.path.join(args.output_dir, args.model)
    os.makedirs(savedir, exist_ok=True)

     # Load checkpoint if exists
    start_step = 1    
    latest_model = find_latest_checkpoint(savedir)
    if latest_model:
        checkpoint = torch.load(latest_model, map_location=device, weights_only=True)
        net_model.load_state_dict(checkpoint['net_model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        optim.load_state_dict(checkpoint['optim'])
        sched.load_state_dict(checkpoint['sched'])
        start_step = checkpoint['step']
        print(f"Resuming from step {start_step}")

    # Ptach work for now. TODO: Remove the Global steps later
    global_step = start_step
    
    # Training Loop
    # with trange(num_epochs, dynamic_ncols=True) as epoch_pbar:
    #     for epoch in epoch_pbar:
    #         epoch_pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")

    with trange(start_step, args.total_steps, initial=start_step, total=args.total_steps, dynamic_ncols=True) as step_pbar:
        for step in step_pbar:
            global_step += 1

            optim.zero_grad()
            
            # Get batch
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)
            
            # Flow matching core
            t, xt, ut = FM.get_sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
            optim.step()
            sched.step()
            
            # EMA update
            ema(net_model, ema_model, args.ema_decay)

            # Logging
            writer.add_scalar('Training/Loss', loss.item(), global_step)
            writer.add_scalar('Learning Rate', optim.param_groups[0]['lr'], global_step)

            # Sample and save
            if args.save_step > 0 and global_step % args.save_step == 0:
                # Generate and save samples
                normal_net_sample = generate_samples(
                    net_model, savedir, global_step, 
                    image_size=(128,128), net_="normal", grid_size=(5,5)
                )
                ema_net_sample = generate_samples(
                    ema_model, savedir, global_step, 
                    image_size=(128,128), net_="ema", grid_size=(5,5)
                )

                # Log generated Image.
                writer.add_images(
                    "Validation/NormalNet", normal_net_sample, global_step
                )
                writer.add_images(
                    "Validation/EmaNet", ema_net_sample, global_step)

                # Save model checkpoints
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": global_step,
                    },
                    os.path.join(savedir, f"{args.model}_weights_step_{global_step}.pt"),
                )

                cleanup_old_checkpoints(savedir, args.keep_n_checkpoints)
            
            step_pbar.set_description(f"loss: {loss.item():.4f}")

    # Close TensorBoard writer
    writer.close()


def main():
    """Main entry point."""
    args = parse_arguments()
    train(args)


if __name__ == "__main__":
    main()