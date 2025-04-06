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
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = img_path.replace("_img.png", "_mask.png")

        # Load PIL Images
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask

        # Apply transforms if provided
        if self.img_transform:
            img = self.img_transform(img)  # should return a tensor [3, H, W]
        else:
            img = TF.to_tensor(img)

        if self.mask_transform:
            mask = self.mask_transform(mask)  # usually resize or augment as PIL
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        else:
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        # Resize manually if needed
        if self.target_size != 2048:
            img = TF.resize(img, [self.target_size, self.target_size], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask.unsqueeze(0).float(), [self.target_size, self.target_size], interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()

        filename = os.path.basename(img_path)
        
        return img, mask, filename
        
    # def __getitem__(self, idx):
    #     img_path = self.image_paths[idx]

    #     # Load image and mask
    #     img = Image.open(img_path).convert("RGB")
    #     mask_path = img_path.replace("_img.png", "_mask.png")
    #     mask = Image.open(mask_path).convert("L")  # grayscale mask with class indices

    #     # Apply transforms (ToTensor, Resize, Normalize, etc.)
    #     if self.img_transform:
    #         img = self.img_transform(img)  # Expected to output tensor [3, H, W]
    #     else:
    #         img = T.ToTensor()(img)

    #     if self.mask_transform:
    #         mask = self.mask_transform(mask)
    #         mask = torch.as_tensor(np.array(mask), dtype=torch.long)
    #     else:
    #         mask = torch.as_tensor(np.array(mask), dtype=torch.long)

    #     # Optional: Resize if not already resized by transforms
    #     if self.target_size != 2048:
    #         img = TF.resize(img, [self.target_size, self.target_size], interpolation=T.InterpolationMode.BILINEAR)
    #         mask = TF.resize(mask.unsqueeze(0).float(), [self.target_size, self.target_size], interpolation=T.InterpolationMode.NEAREST).squeeze(0).long()

    #     filename = os.path.basename(img_path)
    #     return img, mask, filename
 
    # def __getitem__(self, idx):
    #     img_path = self.image_paths[idx]
    #     # Load image (2048x2048x3 RGB)
    #     img = Image.open(img_path).convert("RGB")
    #     # Load mask (2048x2048 grayscale)
    #     mask_path = img_path.replace("_img.png", "_mask.png")
    #     mask = Image.open(mask_path).convert("L")  # Grayscale mask
        
    #     if self.img_transform:
    #         img = self.img_transform(img)
    #     if self.mask_transform: 
    #         mask = self.mask_transform(mask) 
    #         mask = torch.as_tensor(np.array(mask), dtype=torch.long) 
    #     else: 
    #         mask = torch.as_tensor(np.array(mask), dtype=torch.long)
    #     mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)
    #     # Convert to tensors
    #     img_tensor = T.ToTensor()(img)  # Shape: [3, 2048, 2048]
    #     mask_tensor = T.ToTensor()(mask)  # Shape: [1, 2048, 2048]

    #     # Optional: Resize if target_size != 2048
    #     if self.target_size != 2048:
    #         img_tensor = TF.resize(img_tensor, [self.target_size, self.target_size], interpolation=T.InterpolationMode.BILINEAR)
    #         mask_tensor = TF.resize(mask_tensor, [self.target_size, self.target_size], interpolation=T.InterpolationMode.NEAREST)
        
    #     filename = os.path.basename(img_path)
    #     return img_tensor, mask_tensor, filename 
    
        # return img_tensor, mask_tensor  # Both are tensors now
