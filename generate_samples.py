import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchcfm.models.unet.unet import UNetModelWrapper

class ImageGenerator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self._setup_directories()
        self.intermediate_images = []  # Store intermediate images
        
    def _setup_directories(self):
        """Create necessary directories for outputs"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        if self.args.save_intermediates:
            self.intermediates_dir = os.path.join(self.args.output_dir, 'intermediates')
            os.makedirs(self.intermediates_dir, exist_ok=True)
    
    def _load_model(self):
        """Initialize and load the model"""
        model = UNetModelWrapper(
            dim=(3, self.args.image_size[0], self.args.image_size[1]),
            num_res_blocks=2,
            num_channels=self.args.num_channels,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.0,
        ).to(self.device)

        checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
        if self.args.use_ema:
            model.load_state_dict(checkpoint['ema_model'])
            print("Loaded EMA model weights")
        else:
            model.load_state_dict(checkpoint['net_model'])
            print("Loaded standard model weights")
        
        model.eval()
        return model
    
    @staticmethod
    def normalize_samples(x):
        """Normalize samples to [0, 1] range"""
        x = x.clip(-1, 1)
        return x / 2 + 0.5
    
    def store_intermediate(self, x, step_idx):
        """Store intermediate generation steps"""
        samples = self.normalize_samples(x)
        grid = make_grid(samples, nrow=samples.shape[0])
        grid_np = grid.cpu().numpy().transpose(1, 2, 0)
        self.intermediate_images.append((step_idx, grid_np))
    
    def save_intermediate_grid(self, batch_idx):
        """Save all intermediate steps in a single matplotlib grid"""
        if not self.intermediate_images:
            return
            
        num_steps = len(self.intermediate_images)
        fig_width = 15
        fig_height = (fig_width * num_steps) / 4  # Adjust aspect ratio
        
        fig, axes = plt.subplots(num_steps, 1, figsize=(fig_width, fig_height))
        if num_steps == 1:
            axes = [axes]
            
        fig.suptitle(f'Generation Progress (Batch {batch_idx})', fontsize=16)
        
        for (step_idx, img), ax in zip(self.intermediate_images, axes):
            ax.imshow(img)
            ax.set_title(f'Step {step_idx}')
            ax.axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(self.intermediates_dir, f'progress_batch_{batch_idx:03d}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # Clear stored intermediates
        self.intermediate_images = []
    
    def euler_solver(self, x0, t_span, batch_idx, pbar=None):
        """Euler solver with intermediate saves"""
        x = x0
        dt = t_span[1] - t_span[0]
        
        # Save initial state
        if self.args.save_intermediates:
            self.store_intermediate(x, 0)
        
        for t_idx in range(len(t_span) - 1):
            t = t_span[t_idx] * torch.ones(x0.shape[0], device=self.device)
            dx = self.model(t, x)
            x = x + dx * dt
            
            if self.args.save_intermediates and (t_idx + 1) % self.args.intermediate_freq == 0:
                self.store_intermediate(x, t_idx + 1)
            
            if pbar is not None:
                pbar.update(1)
        
        return x
    
    def heun_solver(self, x0, t_span, batch_idx, pbar=None):
        """Heun's solver with intermediate saves"""
        x = x0
        dt = t_span[1] - t_span[0]
        
        # Save initial state
        if self.args.save_intermediates:
            self.store_intermediate(x, 0)
        
        for t_idx in range(len(t_span) - 1):
            t = t_span[t_idx] * torch.ones(x0.shape[0], device=self.device)
            t_next = t_span[t_idx + 1] * torch.ones(x0.shape[0], device=self.device)
            
            # First step: Euler
            dx1 = self.model(t, x)
            x_euler = x + dx1 * dt
            
            # Second step: Correction
            dx2 = self.model(t_next, x_euler)
            x = x + (dx1 + dx2) * dt / 2
            
            if self.args.save_intermediates and (t_idx + 1) % self.args.intermediate_freq == 0:
                self.store_intermediate(x, t_idx + 1)
            
            if pbar is not None:
                pbar.update(1)
        
        return x
    
    def save_batch(self, samples, batch_idx, start_idx):
        """Save a batch of generated samples"""
        # Save individual samples
        for i, sample in enumerate(samples):
            sample_idx = start_idx + i
            individual_path = os.path.join(
                self.args.output_dir, 
                f'sample_{sample_idx:05d}.png'
            )
            save_image(sample, individual_path)
        
        # Save grid
        if self.args.save_grid:
            grid_path = os.path.join(
                self.args.output_dir, 
                f'grid_batch_{batch_idx:03d}.png'
            )
            save_image(
                samples, 
                grid_path, 
                nrow=min(8, int(samples.shape[0]**0.5))
            )
            
        # Save intermediate progress grid
        if self.args.save_intermediates:
            self.save_intermediate_grid(batch_idx)
    
    @torch.no_grad()
    def generate_batch(self, batch_size, batch_idx, pbar=None):
        """Generate a batch of samples"""
        # Generate random initial noise
        x0 = torch.randn(
            batch_size, 
            3, 
            self.args.image_size[0], 
            self.args.image_size[1], 
            device=self.device
        )
        
        # Create time steps
        t_span = torch.linspace(0, 1, self.args.num_steps, device=self.device)
        
        # Select solver and generate samples
        solver = self.euler_solver if self.args.solver == 'euler' else self.heun_solver
        samples = solver(x0, t_span, batch_idx, pbar)
        
        # Normalize samples
        samples = self.normalize_samples(samples)
        
        return samples
    
    def generate(self):
        """Main generation loop"""
        num_batches = (self.args.num_samples + self.args.batch_size - 1) // self.args.batch_size
        total_steps = num_batches * (self.args.num_steps - 1)
        
        with tqdm(total=total_steps, desc="Generating samples") as pbar:
            for batch_idx in range(num_batches):
                # Calculate batch size for last batch
                current_batch_size = min(
                    self.args.batch_size,
                    self.args.num_samples - batch_idx * self.args.batch_size
                )
                
                # Generate batch
                samples = self.generate_batch(current_batch_size, batch_idx, pbar)
                
                # Save batch
                start_idx = batch_idx * self.args.batch_size
                self.save_batch(samples, batch_idx, start_idx)
                
                # Memory cleanup
                torch.cuda.empty_cache()
        
        print(f"Generated {self.args.num_samples} samples in {num_batches} batches")
        print(f"Samples saved to {self.args.output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Sampling script for CFM model')
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
    parser.add_argument('--num_channels', type=int, default=128,
                       help='Number of base channels in UNet')
    parser.add_argument('--use_ema', action='store_true',
                       help='Use EMA model for inference')
    parser.add_argument('--solver', type=str, default='euler',
                       choices=['euler', 'heun'],
                       help='ODE solver type')
    parser.add_argument('--save_grid', action='store_true',
                       help='Save grid of samples for each batch')
    parser.add_argument('--save_intermediates', action='store_true',
                       help='Save intermediate steps during generation')
    parser.add_argument('--intermediate_freq', type=int, default=10,
                       help='Frequency of saving intermediate steps')
    return parser.parse_args()

def main():
    args = parse_args()
    generator = ImageGenerator(args)
    generator.generate()

if __name__ == "__main__":
    main()


"""
python generate_samples_grid.py --checkpoint outputs/results_otcfm_32_otcfm-large-batch_exp/otcfm/otcfm_weights_step_2000000.pt  --num_samples 4 --batch_size 4 --output_dir sample_ot-cfm_large_batch --image_size 128 128 --num_steps 8 --use_ema --solver heun --save_grid --save_intermediates --intermediate_freq 
"""