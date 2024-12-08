import copy
import math
import os

import torch
from absl import app, flags
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from torch.utils.data import DistributedSampler
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from tqdm import trange
# from utils_cifar import ema, generate_samples, infiniteloop, setup

from torchdyn.core import NeuralODE

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

from utils.data_loader import UnlabeledImageDataset
from utils.sampler import euler_solver, heun_solver

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# from torchcfm.conditional_flow_matching import (
#     ConditionalFlowMatcher,
#     ExactOptimalTransportConditionalFlowMatcher,
#     TargetConditionalFlowMatcher,
#     VariancePreservingConditionalFlowMatcher,
# )
# from torchcfm.models.unet.unet import UNetModelWrapper

from conditional_flow_matcher import ConditionalFlowMatcher, OptimalTransportConditionalFlowMatcher
from models.unet_model import UNetModelWrapper


### Set Environment Variables ###
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["WORLD_SIZE"] = "1"

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./outputs/results_flickr_exp3/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 1e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 1000001, help="total training steps"
) 
flags.DEFINE_integer("warmup", 10000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 32, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_string(
    "master_addr", "localhost", help="master address for Distributed Data Parallel"
)
flags.DEFINE_string("master_port", "12355", help="master port for Distributed Data Parallel")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    10000,
    help="frequency of saving checkpoints, 0 to disable during training",
)


def setup(
    rank: int,
    total_num_gpus: int,
    master_addr: str = "0.0.0.0",
    master_port: str = "9669",
    backend: str = "nccl",
):
    """Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        total_num_gpus: Number of GPUs used in the job.
        master_addr: IP address of the master node.
        master_port: Port number of the master node.
        backend: Backend to use.
    """

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )


# def generate_samples(model, parallel, savedir, step, image_size:tuple=(32,32), net_="normal", grid_size=(8, 8)):
#     """Save generated images for sanity check along training.

#     Parameters
#     ----------
#     model:
#         represents the neural network that we want to generate samples from.
#     parallel: bool
#         represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
#     savedir: str
#         represents the path where we want to save the generated images.
#     step: int
#         represents the current step of training.
#     grid_size: tuple
#         represents the grid size for arranging the generated images.
#     """
#     model.eval()

#     model_ = copy.deepcopy(model)
#     if parallel:
#         # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
#         model_ = model_.module.to(device)

#     node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
#     with torch.no_grad():
#         # Adjust the tensor size based on grid_size
#         num_images = grid_size[0] * grid_size[1]
#         traj = node_.trajectory(
#             torch.randn(num_images, 3, image_size[0], image_size[1], device=device),
#             t_span=torch.linspace(0, 1, 100, device=device),
#         )
#         traj = traj[-1, :].view([-1, 3, image_size[0], image_size[1]]).clip(-1, 1)
#         traj = traj / 2 + 0.5
#     save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=grid_size[0])

#     model.train()

def generate_samples(
    model, 
    parallel, 
    savedir, 
    step, 
    image_size:tuple=(32,32), 
    net_="normal", 
    grid_size=(8, 8), 
    method='euler'
):
    """
    Generate samples using Euler or Euler-Heun method.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from.
    parallel: bool
        represents the parallel training flag.
    savedir: str
        represents the path where we want to save the generated images.
    step: int
        represents the current step of training.
    image_size: tuple
        size of the generated images.
    net_: str
        network type identifier.
    grid_size: tuple
        represents the grid size for arranging the generated images.
    method: str
        integration method to use ('euler' or 'euler_heun').
    """
    _supported_method = ["euler", "heun"]
    # Ensure the model is in evaluation mode
    model.eval()

    # Handle parallel training scenario
    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference
        model_ = model_.module.to(device)

    # Prepare integration parameters
    with torch.no_grad():
        # Adjust the tensor size based on grid_size
        num_images = grid_size[0] * grid_size[1]
        
        # Generate initial random noise
        x0 = torch.randn(num_images, 3, image_size[0], image_size[1], device=device)
        
        # Create time span for integration
        t_span = torch.linspace(0, 1, 100, device=device)
        
        # Choose integration method
        if method == 'euler':
            traj = euler_solver(model_, x0, t_span, device=device)
        elif method == 'heun':
            traj = heun_solver(model_, x0, t_span, device=device)
        else:
            raise ValueError(f"Unsupported integration method: {method}, supported methods: {_supported_method}")
        
        # Post-process the trajectory
        traj = traj.view([-1, 3, image_size[0], image_size[1]]).clip(-1, 1)
        traj = traj / 2 + 0.5
    
    # Save the generated images
    save_image(
        traj, 
        f"{savedir}{net_}_generated_FM_images_step_{step}.png", 
        nrow=grid_size[0]
    )

    model.train()


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        # for x, y in iter(dataloader):
        for x in iter(dataloader):
            yield x

def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(rank, total_num_gpus, argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    if FLAGS.parallel and total_num_gpus > 1:
        # When using `DistributedDataParallel`, we need to divide the batch
        # size ourselves based on the total number of GPUs of the current node.
        batch_size_per_gpu = FLAGS.batch_size // total_num_gpus
        setup(rank, total_num_gpus, FLAGS.master_addr, FLAGS.master_port)
    else:
        batch_size_per_gpu = FLAGS.batch_size

    # DATASETS/DATALOADER
    # dataset = datasets.CIFAR10(
    #     root="./data",
    #     train=True,
    #     download=True,
    #     transform=transforms.Compose(
    #         [
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ]
    #     ),
    # )
    transform = transforms.Compose([
        transforms.Resize((128, 128)),   # Resize images to 256x256
        # transforms.CenterCrop((64, 128)),  # Crop the center 256x256
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),            # Convert image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
    ])

    dataset = UnlabeledImageDataset(
        image_dir="/disk1/BharatDiffusion/kohya_ss/experimental_sricpts/real_faces_128",
        transform=transform
    )

    sampler = DistributedSampler(dataset) if FLAGS.parallel else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=False if FLAGS.parallel else True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    # Calculate number of epochs
    steps_per_epoch = math.ceil(len(dataset) / FLAGS.batch_size)
    num_epochs = math.ceil(FLAGS.total_steps / steps_per_epoch)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 128, 128),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(
        rank
    )  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.AdamW(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        net_model = DistributedDataParallel(net_model, device_ids=[rank])
        ema_model = DistributedDataParallel(ema_model, device_ids=[rank])

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = OptimalTransportConditionalFlowMatcher(sigma=sigma, ot_method='exact')
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm']"
        )

    savedir = FLAGS.output_dir + FLAGS.model + "/"
    os.makedirs(savedir, exist_ok=True)

    global_step = 0  # to keep track of the global step in training loop
    
    #TODO: Add a perceptual loss
    with trange(num_epochs, dynamic_ncols=True) as epoch_pbar:
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            if sampler is not None:
                sampler.set_epoch(epoch)

            with trange(steps_per_epoch, dynamic_ncols=True) as step_pbar:
                for step in step_pbar:
                    global_step += step

                    optim.zero_grad()
                    x1 = next(datalooper).to(rank)
                    x0 = torch.randn_like(x1)
                    t, xt, ut = FM.get_sample_location_and_conditional_flow(x0, x1)
                    vt = net_model(t, xt)
                    loss = torch.mean((vt - ut) ** 2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
                    optim.step()
                    sched.step()
                    ema(net_model, ema_model, FLAGS.ema_decay)  # new

                    # sample and Saving the weights
                    if FLAGS.save_step > 0 and global_step % FLAGS.save_step == 0:
                        generate_samples(
                            net_model, FLAGS.parallel, savedir, global_step, image_size=(128,128), net_="normal", grid_size=(5,5)
                        )
                        generate_samples(
                            ema_model, FLAGS.parallel, savedir, global_step, image_size=(128,128), net_="ema", grid_size=(5,5)
                        )
                        torch.save(
                            {
                                "net_model": net_model.state_dict(),
                                "ema_model": ema_model.state_dict(),
                                "sched": sched.state_dict(),
                                "optim": optim.state_dict(),
                                "step": global_step,
                            },
                            savedir + f"{FLAGS.model}_weights_step_{global_step}.pt",
                        )
                    step_pbar.set_description(f"loss: {loss.item():.4f}")


def main(argv):
    # get world size (number of GPUs)
    total_num_gpus = int(os.getenv("WORLD_SIZE", 1))

    if FLAGS.parallel and total_num_gpus > 1:
        train(rank=int(os.getenv("RANK", 0)), total_num_gpus=total_num_gpus, argv=argv)
    else:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        train(rank=device, total_num_gpus=total_num_gpus, argv=argv)


if __name__ == "__main__":
    app.run(main)