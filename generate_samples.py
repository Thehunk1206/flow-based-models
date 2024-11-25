import os
import torch
import argparse
from tqdm import tqdm, trange
from torchvision.utils import save_image
from torchcfm.models.unet.unet import UNetModelWrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Sampling script for Conditional Flow Matching model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the checkpoint file')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Total number of images to generate')
    parser.add_argument('--batch_size', type=int, default=25,
                       help='Batch size for generation')
    parser.add_argument('--output_dir', type=str, default='generated_samples',
                       help='Directory to save generated images')
    parser.add_argument('--image_size', type=int, nargs=2, default=[128, 128],
                       help='Image size (height, width)')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Number of steps in the ODE solver')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[5, 5],
                       help='Grid size for the output image')
    parser.add_argument('--num_channels', type=int, default=128,
                       help='Number of base channels in UNet')
    parser.add_argument('--use_ema', action='store_true',
                       help='Use EMA model for inference')
    parser.add_argument('--solver', type=str, default='euler',
                       choices=['euler', 'heun'],
                       help='ODE solver type')
    parser.add_argument('--save_grid', action='store_true',
                       help='Save grid of samples for each batch')
    return parser.parse_args()

def euler_solver(model, x0, t_span, device, pbar=None):
    """Euler solver for ODE integration with progress bar"""
    x = x0
    dt = t_span[1] - t_span[0]
    
    for t_idx in range(len(t_span) - 1):
        t = t_span[t_idx] * torch.ones(x0.shape[0], device=device)
        dx = model(t, x)
        x = x + dx * dt
        if pbar is not None:
            pbar.update(1)
    
    return x

def heun_solver(model, x0, t_span, device, pbar=None):
    """Heun's solver (improved Euler) for ODE integration with progress bar"""
    x = x0
    dt = t_span[1] - t_span[0]
    
    for t_idx in range(len(t_span) - 1):
        t = t_span[t_idx] * torch.ones(x0.shape[0], device=device)
        t_next = t_span[t_idx + 1] * torch.ones(x0.shape[0], device=device)
        
        # First step: Euler
        dx1 = model(t, x)
        x_euler = x + dx1 * dt
        
        # Second step: Correction
        dx2 = model(t_next, x_euler)
        x = x + (dx1 + dx2) * dt / 2
        
        if pbar is not None:
            pbar.update(1)
    
    return x

def load_model(args, device):
    # Initialize model
    model = UNetModelWrapper(
        dim=(3, args.image_size[0], args.image_size[1]),
        num_res_blocks=2,
        num_channels=args.num_channels,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.0,  # No dropout during inference
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if args.use_ema:
        model.load_state_dict(checkpoint['ema_model'])
        print("Loaded EMA model weights")
    else:
        model.load_state_dict(checkpoint['net_model'])
        print("Loaded standard model weights")
    
    model.eval()
    return model

@torch.no_grad()
def generate_batch(model, args, device, batch_size, batch_idx, pbar=None):
    """Generate a batch of samples using the trained model"""
    # Generate random initial noise
    x0 = torch.randn(batch_size, 3, args.image_size[0], args.image_size[1], device=device)
    
    # Create time steps
    t_span = torch.linspace(0, 1, args.num_steps, device=device)
    
    # Select solver
    solver = euler_solver if args.solver == 'euler' else heun_solver
    
    # Generate samples
    samples = solver(model, x0, t_span, device, pbar)
    
    # Normalize to [0, 1] range
    samples = samples.clip(-1, 1)
    samples = samples / 2 + 0.5
    
    return samples

def save_batch(samples, args, batch_idx, start_idx):
    """Save a batch of generated samples"""
    # Save individual samples
    for i, sample in enumerate(samples):
        sample_idx = start_idx + i
        individual_path = os.path.join(args.output_dir, f'sample_{sample_idx:05d}.png')
        save_image(sample, individual_path)
    
    # Optionally save grid for this batch
    if args.save_grid:
        grid_path = os.path.join(args.output_dir, f'grid_batch_{batch_idx:03d}.png')
        save_image(samples, grid_path, nrow=min(8, int(samples.shape[0]**0.5)))

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args, device)
    
    # Calculate number of batches
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    # Initialize progress bars
    total_steps = num_batches * (args.num_steps - 1)
    with tqdm(total=total_steps, desc="Generating samples") as pbar:
        for batch_idx in range(num_batches):
            # Calculate batch size for last batch
            current_batch_size = min(args.batch_size, 
                                   args.num_samples - batch_idx * args.batch_size)
            
            # Generate batch
            samples = generate_batch(model, args, device, current_batch_size, 
                                  batch_idx, pbar)
            
            # Save batch
            start_idx = batch_idx * args.batch_size
            save_batch(samples, args, batch_idx, start_idx)
            
            # Memory cleanup
            torch.cuda.empty_cache()
    
    print(f"Generated {args.num_samples} samples in {num_batches} batches")
    print(f"Samples saved to {args.output_dir}")

if __name__ == "__main__":
    main()

'''
python generate_samples.py     --checkpoint outputs/results_flickr_exp2/otcfm/otcfm_weights_step_954730000.pt     --num_samples 200     --batch_size 4     --output_dir generated_samples     --image_size 128 128     --num_steps 5     --use_ema     --solver heun     --save_grid
'''