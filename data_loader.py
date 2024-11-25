import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loguru import logger

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.image_filenames = os.listdir(image_dir)  # List of image filenames
        self.transform = transform

        logger.info(f"Total number of images found: {len(self.image_filenames)}")


    def __len__(self):
        # Return the number of images
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load an image
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")  # Convert image to RGB

        if self.transform:
            image = self.transform(image)  # Apply transformations if any

        return image  # Return the image only (no labels)

if __name__ == "__main__":

    # Example image transformations
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),   # Resize images to 256x256
        transforms.CenterCrop((128, 256)),  # Crop the center 256x256
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),            # Convert image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
    ])

    # Initialize the dataset
    image_dir = '/disk1/BharatDiffusion/kohya_ss/experimental_sricpts/anime_images'  # Specify your image directory
    dataset = UnlabeledImageDataset(image_dir=image_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=32,      # Adjust the batch size as needed
        shuffle=True,       # Shuffle the data at the beginning of each epoch
        num_workers=4       # Number of subprocesses to use for data loading
    )

    for images in dataloader:
        logger.info(f"Shape of images: {images.shape}")
