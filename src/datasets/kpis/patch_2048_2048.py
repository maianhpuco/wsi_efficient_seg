import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class Patch_2048(Dataset):
    """
    A custom dataset for semantic segmentation.
    
    Expects two directories:
      - images_dir: directory containing RGB images.
      - masks_dir: directory containing corresponding segmentation masks.
      
    Assumes that image and mask filenames match.
    """
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        # List image files and sort them so that they match with masks.
        self.image_files = sorted([
            f for f in os.listdir(images_dir) 
            if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.mask_files = sorted([
            f for f in os.listdir(masks_dir)
            if os.path.isfile(os.path.join(masks_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # Optionally, verify that the number and names match.
        assert len(self.image_files) == len(self.mask_files), "Number of images and masks must be equal."
        # Optionally check that each image has a corresponding mask.
        for img_file, mask_file in zip(self.image_files, self.mask_files):
            if os.path.splitext(img_file)[0] != os.path.splitext(mask_file)[0]:
                raise ValueError(f"Image {img_file} and mask {mask_file} do not match.")

        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Build full paths for image and mask.
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        # Open image and mask.
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Optionally, apply transforms.
        if self.transform is not None:
            image = self.transform(image)
        else:
            # For example, convert to tensor and normalize.
            image = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                # If EfficientViT requires a particular normalization, add T.Normalize(mean, std)
            ])(image)
            
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            # Convert mask to a tensor of type long (each pixel as class index)
            mask = np.array(mask, dtype=np.int64)
            mask = torch.from_numpy(mask)
            
        return image, mask

# # Example usage:
# if __name__ == "__main__":
#     images_dir = "/path/to/your/images"
#     masks_dir = "/path/to/your/masks"
    
#     dataset = Patch_256_256(images_dir, masks_dir)
    
#     # Example: Create a DataLoader for training.
#     from torch.utils.data import DataLoader
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
#     # Iterate through one batch to test
#     for images, masks in dataloader:
#         print("Image batch shape:", images.shape)  # Expected: [B, 3, H, W]
#         print("Mask batch shape:", masks.shape)      # Expected: [B, H, W]
#         break
