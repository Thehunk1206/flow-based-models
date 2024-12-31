import os

import torch
from torchvision.utils import save_image

from .sampler import euler_solver, heun_solver
import copy

from loguru import logger
def generate_samples(
    model, 
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
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure the model is in evaluation mode
    model.eval()

    model_ = copy.deepcopy(model)

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
    return traj

def find_latest_checkpoint(dir_path:str):
    """Find the latest checkpoint in the directory."""
    checkpoints = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
    if not checkpoints:
        return None
    
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(dir_path, latest)

def cleanup_old_checkpoints(dir_path:str, keep_n:int):
    """Keep only the latest n checkpoints."""
    checkpoints = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
    if len(checkpoints) <= keep_n:
        return
    
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for ckpt in checkpoints[:-keep_n]:
        logger.info(f"Removing old checkpoint: {ckpt}")
        os.remove(os.path.join(dir_path, ckpt))

def ema(source, target, decay):
    """Exponential Moving Average update."""
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )

def infiniteloop(dataloader):
    """Create an infinite loop over the dataloader."""
    while True:
        for x in iter(dataloader):
            yield x


def warmup_lr(step, warmup_steps):
    """Linear warmup learning rate scheduler."""
    return min(step, warmup_steps) / warmup_steps