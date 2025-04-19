import os
from glob import glob
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torch 
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF 
# Dataset for WSI patches

class WSIPatch2048Dataset(Dataset):
    def __init__(self, patch_dir, target_size=2048, img_transform=None, mask_transform=None):  # Default to 2048 unless resizing
        self.patch_dir = patch_dir
        self.target_size = target_size
        self.img_transform = img_transform
        self.mask_transform = mask_transform 
        
        self.image_paths = sorted(glob(os.path.join(patch_dir, "**/*_img.png"), recursive=True))
        if not self.image_paths:
            raise ValueError(f"No image patches found in {patch_dir}")
        print(f"Loaded {len(self.image_paths)} patches from {patch_dir}")

    def __len__(self):
        return len(self.image_paths)
    
    # def __getitem__(self, idx):
    #     img_path = self.image_paths[idx]
    #     mask_path = img_path.replace("_img.png", "_mask.png")

    #     # Load image and mask (PIL)
    #     img = Image.open(img_path).convert("RGB")
    #     mask = Image.open(mask_path).convert("L")

    #     # Convert mask to NumPy and remap 255 → 1
    #     mask_np = np.array(mask, dtype=np.uint8)
    #     mask_np[mask_np == 255] = 1

    #     # Convert back to PIL to apply optional transforms (resize, etc.)
    #     mask = Image.fromarray(mask_np)

    #     # Apply image transform
    #     if self.img_transform:
    #         img = self.img_transform(img)
    #     else:
    #         img = TF.to_tensor(img)
    #         # img = TF.resize(img, [64, 64], interpolation=TF.InterpolationMode.BILINEAR) #TESTING THIS stride_list=[32, 16, 8], 

    #     # Resize mask to match model output size (64x64)
    #     if self.mask_transform:
    #         mask = self.mask_transform(mask)
            
    #     # mask = TF.resize(mask, [64, 64], interpolation=TF.InterpolationMode.NEAREST)
    #     mask = torch.as_tensor(np.array(mask), dtype=torch.long)

    #     filename = os.path.basename(img_path)
    #     return img, mask, filename
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = img_path.replace("_img.png", "_mask.png")

        # Load image and mask (PIL)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
    
        # Convert mask to NumPy and remap 255 → 1
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np[mask_np == 255] = 1
        mask = Image.fromarray(mask_np)
        print("mask shape:", mask.size)
        print("img shape:", img.size)
        
        # Conditional resizing to target_size if needed
        if img.size != (self.target_size, self.target_size):
            print("yessss")
            img = TF.resize(img, [self.target_size, self.target_size], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.target_size, self.target_size], interpolation=TF.InterpolationMode.NEAREST)

        # Apply additional image transforms if provided
        if self.img_transform:
            img = self.img_transform(img)
        else:
            img = TF.to_tensor(img)

        # Apply additional mask transforms if provided
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Resize mask to model output size (e.g., 64x64) if needed
            # mask = TF.resize(mask, [64, 64], interpolation=TF.InterpolationMode.NEAREST)
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        filename = os.path.basename(img_path)
        return img, mask, filename 

