import os
from glob import glob
from PIL import Image
import numpy as np
import torch


import os
from glob import glob

import numpy as np
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
        self.vqgan = vqgan_model.eval() if vqgan_model is not None else None
        self.target_size = target_size
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.patch_size = patch_size
        self.stride = stride  

        self.device = next(self.vqgan.parameters()).device if self.vqgan is not None else torch.device("cpu")
        self.image_paths = sorted(glob(os.path.join(patch_dir, "**/*_img.png"), recursive=True))
        if not self.image_paths:
            raise ValueError(f"No image patches found in {patch_dir}")
        print(f"Loaded {len(self.image_paths)} patches from {patch_dir}")

    def __len__(self):
        return len(self.image_paths)

    def preprocess_vqgan(self, x):
        return 2. * x - 1.  # Normalize to [-1, 1]

    def split_patches(self, img_np, mask_np):
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

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if img.size != (self.target_size, self.target_size):
            img = TF.resize(img, [self.target_size, self.target_size], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.target_size, self.target_size], interpolation=TF.InterpolationMode.NEAREST)

        img_np = np.array(img)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np[mask_np == 255] = 1

        vq_patches = []

        for patch_img, patch_mask in self.split_patches(img_np, mask_np):
            if self.vqgan is not None:
                patch_img = patch_img.unsqueeze(0).to(self.device)  # [1, 3, 256, 256]
                patch_img = self.preprocess_vqgan(patch_img)

                with torch.no_grad():
                    _, _, [_, _, indices] = self.vqgan.encode(patch_img)  # [1, H', W']
                    indices = indices.squeeze(0).cpu()  # [H', W']
            else:
                indices = torch.zeros((16, 16), dtype=torch.long)  # fallback placeholder

            vq_patches.append((indices, patch_mask))

        return vq_patches
