import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
from PIL import Image
import cv2
import numpy as np
import random
from typing import List, Tuple, Dict
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class MultiVideoDataset(Dataset):
    def __init__(self, video_dir: str, num_frames: int = 16, image_size: int = 128, 
                 stride: int = 1, augment: bool = True):
        """
        Args:
            video_dir (str): Directory containing video clips
            num_frames (int): Number of frames in each sequence
            image_size (int): Size to resize frames to
            stride (int): Number of frames to slide window by
            augment (bool): Whether to apply augmentations
        """
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.stride = stride
        self.augment = augment
        
        # Store video paths and their frame counts
        self.video_files = [f for f in os.listdir(video_dir) 
                           if f.endswith(('.mp4', '.avi', '.mov'))]
        
        # Dictionary to store frame sequences for each video
        self.video_sequences: Dict[str, List[List[int]]] = defaultdict(list)
        # List to store all available sequences for random sampling
        self.all_sequences: List[Tuple[str, List[int]]] = []
        
        self._compute_sequences()
        
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                               std=[0.5, 0.5, 0.5])
        ])
        
        # Augmentations
        if self.augment:
            self.spatial_aug = ConsistentSpatialAugmentation(image_size)
    
    def _compute_sequences(self):
        """Pre-compute all valid frame sequences for each video"""
        print(f"Processing {len(self.video_files)} videos...")
        
        for video_file in self.video_files:
            video_path = os.path.join(self.video_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if total_frames >= self.num_frames:
                # Calculate all possible sequences with stride
                start_indices = range(0, total_frames - self.num_frames + 1, self.stride)
                for start_idx in start_indices:
                    sequence_indices = list(range(start_idx, start_idx + self.num_frames))
                    self.video_sequences[video_file].append(sequence_indices)
                    self.all_sequences.append((video_file, sequence_indices))
        
        print(f"Total sequences available: {len(self.all_sequences)}")
        
    def __len__(self):
        return len(self.all_sequences)
    
    def _load_frame_sequence(self, video_path: str, frame_indices: List[int]) -> torch.Tensor:
        """Load a sequence of frames from a video"""
        frames = torch.zeros((self.num_frames, 3, self.image_size, self.image_size))
        
        cap = cv2.VideoCapture(video_path)
        current_frame = 0
        frame_idx = 0
        
        while cap.isOpened() and frame_idx < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame in frame_indices:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply transforms
                frame_tensor = self.base_transform(frame)
                frames[frame_idx] = frame_tensor
                frame_idx += 1
                
            current_frame += 1
            
            if frame_idx == self.num_frames:
                break
                
        cap.release()
        return frames

    def __getitem__(self, idx):
        video_file, frame_indices = self.all_sequences[idx]
        video_path = os.path.join(self.video_dir, video_file)
        
        # Load frame sequence
        frames = self._load_frame_sequence(video_path, frame_indices)
        
        # Apply augmentations if in training mode
        if self.augment:
            frames = self.spatial_aug(frames)
        
        # Return frames along with metadata
        metadata = {
            'video_file': video_file,
            'start_frame': frame_indices[0],
            'end_frame': frame_indices[-1]
        }
        
        return frames, metadata

class ConsistentSpatialAugmentation:
    """Applies consistent spatial augmentations across all frames"""
    def __init__(self, image_size=128):
        self.image_size = image_size
        
    def __call__(self, frames):
        """
        Args:
            frames: torch.Tensor of shape (T, C, H, W)
        Returns:
            augmented frames
        """
        # Random horizontal flip
        if random.random() < 0.5:
            frames = torch.flip(frames, [3])
            
        # Random color jitter (consistent across frames)
        if random.random() < 0.8:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            frames = TF.adjust_brightness(frames, brightness)
            frames = TF.adjust_contrast(frames, contrast)
            frames = TF.adjust_saturation(frames, saturation)
            
        return frames

class VideoCache:
    """Simple cache for video frames to improve loading efficiency"""
    def __init__(self, max_videos=5):
        self.max_videos = max_videos
        self.cache = {}
    
    def get_frames(self, video_path):
        if video_path not in self.cache:
            if len(self.cache) >= self.max_videos:
                # Remove oldest video from cache
                self.cache.pop(next(iter(self.cache)))
            # Load video frames
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            self.cache[video_path] = frames
        return self.cache[video_path]

def get_video_dataloader(
    video_dir: str,
    batch_size: int = 8,
    num_frames: int = 16,
    image_size: int = 128,
    stride: int = 8,
    num_workers: int = 4,
    augment: bool = True,
    shuffle: bool = True
):
    """
    Creates a DataLoader that randomly samples sequences from multiple videos.
    
    Args:
        video_dir (str): Directory containing video clips
        batch_size (int): Batch size
        num_frames (int): Number of frames in each sequence
        image_size (int): Size to resize frames to
        stride (int): Number of frames to slide window by
        num_workers (int): Number of worker processes
        augment (bool): Whether to apply augmentations
        shuffle (bool): Whether to shuffle the dataset
    
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = MultiVideoDataset(
        video_dir=video_dir,
        num_frames=num_frames,
        image_size=image_size,
        stride=stride,
        augment=augment
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # Shuffle only during training
        num_workers=num_workers,
        pin_memory=True
    )




# Assuming the previous code is in the same file or imported
# def driver_function():
#     # Set the path to your video directory
#     video_dir = '/disk1/flow_based_research/CelebV-HQ/sample_celeb_V-HQ'
    
#     # Ensure the directory exists
#     if not os.path.exists(video_dir):
#         print(f"Error: Directory {video_dir} does not exist.")
#         return
    
#     # Create dataloader with some sample parameters
#     dataloader = get_video_dataloader(
#         video_dir=video_dir,
#         batch_size=4,  # Process 4 video sequences at a time
#         num_frames=32,  # Extract 16 frames per sequence
#         image_size=128,  # Resize frames to 128x128
#         stride=8,  # Slide window by 8 frames
#         num_workers=2,  # Use 2 worker processes
#         augment=True  # Apply data augmentation
#     )
    
#     # print(len(dataloader))
#     # Iterate through the dataloader
#     for batch_idx, (frames, metadata) in enumerate(dataloader):
#         print(f"Batch {batch_idx}:")
#         print("Frames shape:", frames.shape)
        
#         # Print metadata for each video in the batch
#         # for i, meta in enumerate(metadata):
#         #     print(f"  Video {i}:")
#         #     print(f"    Filename: {meta['video_file']}")
#         #     print(f"    Start frame: {meta['start_frame']}")
#         #     print(f"    End frame: {meta['end_frame']}")
#         print(metadata)
#         # break
        
#         # Optional: Visualize the first batch
#         if batch_idx == 2:

            
#             for k in range(frames.shape[0]):
#                 for i in range(min(32, frames.shape[1])):
#                     # Create a grid of frames from the first video sequence
#                     fig, axes = plt.subplots(4, 8, figsize=(20, 5))
#                     axes = axes.flatten()
#                     # Convert tensor to numpy and denormalize
#                     frame = frames[k, i].numpy()
#                     # frame = (frame * 0.5) + 0.5  # Denormalize
#                     frame = frame.transpose(1, 2, 0)  # CHW to HWC
                    
#                     axes[i].imshow(frame)
#                     axes[i].axis('off')
                
#                 plt.tight_layout()
#                 plt.suptitle('Sample Frames from First Video Sequence')
#                 plt.savefig(f'sample_frames_{k}.png')
        
#         # Break after first batch for demonstration
#         break



def visualize_batch_frames(frames, metadata, output_dir='batch_frames'):
    """
    Visualize all frames for each sequence in a batch
    
    Args:
    - frames: Tensor of shape [batch_size, num_frames, channels, height, width]
    - metadata: List of dictionaries with video metadata
    - output_dir: Directory to save visualization images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through each sequence in the batch
    for batch_idx, (batch_frames, batch_meta) in enumerate(zip(frames, metadata)):
        # Create a large figure for this sequence
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(4, 8, figure=fig)
        # fig.suptitle(f'Sequence {batch_idx}: {batch_meta["video_file"]} (Frames {batch_meta["start_frame"]}-{batch_meta["end_frame"]})', fontsize=16)
        
        # Visualize each frame in the sequence
        for frame_idx in range(batch_frames.shape[0]):
            ax = fig.add_subplot(gs[frame_idx // 8, frame_idx % 8])
            
            # Convert tensor to numpy and denormalize
            frame = batch_frames[frame_idx].numpy()
            
            # Transpose from CxHxW to HxWxC
            frame = frame.transpose(1, 2, 0)
            
            # Denormalize (assuming normalization was done with mean=0.5, std=0.5)
            frame = (frame * 0.5) + 0.5
            
            # Clip values to ensure they're in [0,1] range
            frame = np.clip(frame, 0, 1)
            
            ax.imshow(frame)
            ax.set_title(f'Frame {frame_idx}')
            ax.axis('off')
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'sequence_{batch_idx}_frames.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def driver_function():
    # Set the path to your video directory
    video_dir = '/disk1/flow_based_research/CelebV-HQ/downloaded_celebvhq/processed'
    
    # Ensure the directory exists
    if not os.path.exists(video_dir):
        print(f"Error: Directory {video_dir} does not exist.")
        return
    
    # Create dataloader with some sample parameters
    dataloader = get_video_dataloader(
        video_dir=video_dir,
        batch_size=4,  # Process 4 video sequences at a time
        num_frames=32,  # Extract 32 frames per sequence
        image_size=128,  # Resize frames to 128x128
        stride=8,  # Slide window by 8 frames
        num_workers=2,  # Use 2 worker processes
        augment=False,  # Apply data augmentation
        shuffle=True # Shuffle the dataset
    )
    
    # Iterate through the dataloader
    for batch_idx, (frames, metadata) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print("Frames shape:", frames.shape)
        print(metadata)
        
        # Visualize the entire batch
        visualize_batch_frames(frames, metadata)
        
        # Optionally, break after first batch if you want to see just one
        break

if __name__ == "__main__":
    driver_function()

