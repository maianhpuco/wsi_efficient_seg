import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class WSIDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        """
        Args:
            data_dir (str): Directory containing the WSI data (e.g., '/input_slide/')
            transform: PyTorch transforms for the images
            mask_transform: PyTorch transforms for the masks (if different)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mask_transform = mask_transform or transform  # Use same transform if mask_transform not provided
        
        # Collect image and mask paths (similar to your main function)
        image = []
        seg = []
        types = glob(os.path.join(data_dir, '*'))
        for type in types:
            now_imgs = glob(os.path.join(type, 'img', '*.tiff'))  # Assuming input is TIFF as in your code
            image.extend(now_imgs)
            now_lbls = glob(os.path.join(type, 'mask', '*mask.tiff'))
            seg.extend(now_lbls)

        self.images = sorted(image)
        self.segs = sorted(seg)
        
        if len(self.images) != len(self.segs):
            raise ValueError(f"Number of images ({len(self.images)}) and masks ({len(self.segs)}) do not match!")
        
        print(f"Loaded {len(self.images)} WSI images from {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # Convert TIFF to RGB
        
        # Load mask
        mask_path = self.segs[idx]
        mask = Image.open(mask_path).convert('L')  # 'L' mode for grayscale mask
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask

def get_wsi_dataloader(data_dir, batch_size=1, shuffle=False, num_workers=0):
    """
    Create a DataLoader for WSI-level data
    
    Args:
        data_dir (str): Directory containing the WSI data
        batch_size (int): Batch size for the DataLoader (default 1 due to large WSI size)
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for loading data
    
    Returns:
        DataLoader object
    """
    # Define transforms for images
    image_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Define transforms for masks (no normalization needed)
    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create dataset
    dataset = WSIDataset(
        data_dir=data_dir,
        transform=image_transform,
        mask_transform=mask_transform
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Faster data transfer to GPU if available
    )
    
    return dataloader

# Example usage
if __name__ == "__main__":
    data_dir = "/project/hnguyen2/mvu9/datasets/kidney_pathology_image/train/Task2_WSI_level"  
    
    wsi_loader = get_wsi_dataloader(data_dir, batch_size=1, shuffle=False, num_workers=0)
    
    # Test the DataLoader
    for images, masks in wsi_loader:
        print(f"Image batch shape: {images.shape}")  # [batch_size, channels, height, width]
        print(f"Mask batch shape: {masks.shape}")    # [batch_size, 1, height, width]
        
        # Optionally visualize the first image and mask
        img = images[0].permute(1, 2, 0).numpy()  # Convert to HWC for visualization
        mask = masks[0][0].numpy()  # Remove channel dim for mask
        print("Check the shape of img and mask")
        print(img.shape, mask.shape)
        
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(img)
        # plt.title("WSI Image")
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask, cmap='gray')
        # plt.title("Mask")
        # plt.show()
        
        break  # Only process one batch for testing