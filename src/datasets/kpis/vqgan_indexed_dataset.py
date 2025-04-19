import os
from glob import glob
from PIL import Image
import numpy as np
import torch

import imageio 

import os
from glob import glob

import numpy as np
import imageio
import scipy.ndimage as ndi
from tqdm import tqdm

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class VQGANIndexedDataset(Dataset):
    def __init__(
        self, 
        patch_dir, 
        vqgan_model=None, 
        target_size=2048, 
        patch_size=256, 
        stride=256, 
        img_transform=None, 
        mask_transform=None
    ):
        self.patch_dir = patch_dir
        self.vqgan = vqgan_model
        self.target_size = target_size
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.patch_size = patch_size
        self.stride = stride  

        self.image_paths = sorted(glob(os.path.join(patch_dir, "**/*_img.png"), recursive=True))
        if not self.image_paths:
            raise ValueError(f"No image patches found in {patch_dir}")
        print(f"Loaded {len(self.image_paths)} patches from {patch_dir}")

    def __len__(self):
        return len(self.image_paths)
    
    def split_patches(self, img_np, mask_np):
        '''
        img_np: [H, W, C] numpy array
        mask_np: [H, W] numpy array
        Yields: [3, P, P] tensor, [P, P] tensor
        '''
        H, W = img_np.shape[:2]
        x_slide = (H - self.patch_size) // self.stride + 1
        y_slide = (W - self.patch_size) // self.stride + 1

        with tqdm(total=x_slide * y_slide, desc=f"Extracting patches") as pbar:
            for xi in range(x_slide):
                for yi in range(y_slide):
                    x = xi * self.stride if xi != x_slide - 1 else H - self.patch_size
                    y = yi * self.stride if yi != y_slide - 1 else W - self.patch_size

                    img_patch = img_np[x:x + self.patch_size, y:y + self.patch_size, :]
                    mask_patch = mask_np[x:x + self.patch_size, y:y + self.patch_size]

                    assert img_patch.shape == (self.patch_size, self.patch_size, 3)
                    assert mask_patch.shape == (self.patch_size, self.patch_size)

                    # Convert to tensor
                    img_tensor = TF.to_tensor(Image.fromarray(img_patch))
                    mask_tensor = torch.from_numpy(mask_patch).long()

                    if self.img_transform:
                        img_tensor = self.img_transform(img_tensor)
                    if self.mask_transform:
                        mask_tensor = self.mask_transform(mask_tensor)

                    pbar.update(1)
                    yield img_tensor, mask_tensor

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = img_path.replace("_img.png", "_mask.png")

        # Load as PIL and resize
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if img.size != (self.target_size, self.target_size):
            img = TF.resize(img, [self.target_size, self.target_size], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.target_size, self.target_size], interpolation=TF.InterpolationMode.NEAREST)

        # Convert to NumPy
        img_np = np.array(img)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np[mask_np == 255] = 1

        # Yield all patches as a list (or use custom collate for DataLoader)
        patch_list = []
        for patch_img, patch_mask in self.split_patches(img_np, mask_np):
            patch_list.append((patch_img, patch_mask))
            # if you want all, don't break
            # break  # Uncomment if you want only one patch for now

        return patch_list
 