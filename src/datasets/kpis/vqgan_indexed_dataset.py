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

        # Gather image paths
        self.image_paths = sorted(glob(os.path.join(patch_dir, "**/*_img.png"), recursive=True))
        if not self.image_paths:
            raise ValueError(f"No image patches found in {patch_dir}")
        print(f"Loaded {len(self.image_paths)} patches from {patch_dir}")
        self.patch_size = self.patch_size
        self.stride = stride 
        
    def __len__(self):
        return len(self.image_paths)
    
    def split_patches(self, img, mask):
        '''
        input: img: [C, H, W]
        output: patches: [N, C, P, P]
        '''
         # Calculate expected patches
        h, w = img.shape[:2]
        x_slide = (h - self.patch_size) // self.stride + 1
        y_slide = (w - self.patch_size) // self.stride + 1
        expected_patches = x_slide * y_slide

        # Subfolder and WSI ID for saving patches
        # subfolder = os.path.dirname(img_path).split("/")[-1]
        # wsi_id = os.path.basename(img_path).replace("_wsi.tiff", "")
        # save_path = os.path.join(save_dir, subfolder, wsi_id)
        # os.makedirs(save_path, exist_ok=True)

        # Extract patches and count actual patches written
        actual_patches = 0
        with tqdm(total=expected_patches, desc=f"Extracting patches") as pbar:
            for xi in range(x_slide):
                for yi in range(y_slide):
                    now_x = xi * self.stride if xi != x_slide - 1 else h - self.patch_size
                    now_y = yi * self.stride if yi != y_slide - 1 else w - self.patch_size

                    # Extract image patch
                    img_patch = img[now_x:now_x + self.patch_size, now_y:now_y + self.patch_size, :]
                    assert img_patch.shape == (self.patch_size, self.patch_size, 3)

                    # Extract mask patch
                    mask_patch = mask[now_x:now_x + self.patch_size, now_y:now_y + self.patch_size]
                    assert mask_patch.shape == (self.patch_size, self.patch_size)
                    yield img_patch, mask_patch 
                    actual_patches += 1
                    pbar.update(1) 
                    
                    
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
        
        if img.size != (self.target_size, self.target_size):
            # print("yessss")
            img = TF.resize(img, [self.target_size, self.target_size], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.target_size, self.target_size], interpolation=TF.InterpolationMode.NEAREST) 
        
        for patch_img, patch_mask in self.split_patches(img, mask):
            print(patch_img.shape, patch_mask.shape) 
            break     
        
        
        
    
        
        # # Encode image using VQGAN
        # with torch.no_grad():
        #     z = self.vqgan.encoder(img_tensor)
        #     z_q, indices, _ = self.vqgan.codebook(z)

        # # Reshape indices and embeddings
        # indices = indices.view(z_q.shape[2], z_q.shape[3])  # [H, W]
        # embeddings = self.vqgan.codebook.embedding(indices)  # [H, W, C]

        # return indices.cpu(), embeddings.cpu(), os.path.basename(img_path)

if __name__=='__main__':
    