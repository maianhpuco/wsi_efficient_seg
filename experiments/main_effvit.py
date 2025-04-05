import os
from glob import glob
import yaml
import torch
import argparse
from tqdm import tqdm 
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
import io
import PIL
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

# Device setup
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

# Model loading functions
def load_model(path, device):
    if path.startswith('http'):
        from urllib.request import urlopen
        with urlopen(path) as f:
            buf = io.BytesIO(f.read())
    else:
        with open(path, 'rb') as f:
            buf = io.BytesIO(f.read())
    return torch.load(buf, map_location=device)

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

# Dataset for WSI patches
class WSIPatch2048Dataset(Dataset):
    def __init__(self, patch_dir, target_size=2048):  # Default to 2048 unless resizing
        self.patch_dir = patch_dir
        self.target_size = target_size
        self.image_paths = sorted(glob(os.path.join(patch_dir, "**/*_img.png"), recursive=True))
        if not self.image_paths:
            raise ValueError(f"No image patches found in {patch_dir}")
        print(f"Loaded {len(self.image_paths)} patches from {patch_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load image (2048x2048x3 RGB)
        img = Image.open(img_path).convert("RGB")
        # Load mask (2048x2048 grayscale)
        mask_path = img_path.replace("_img.png", "_mask.png")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask

        # Convert to tensors
        img_tensor = T.ToTensor()(img)  # Shape: [3, 2048, 2048]
        mask_tensor = T.ToTensor()(mask)  # Shape: [1, 2048, 2048]

        # Optional: Resize if target_size != 2048
        if self.target_size != 2048:
            img_tensor = TF.resize(img_tensor, [self.target_size, self.target_size], interpolation=T.InterpolationMode.BILINEAR)
            mask_tensor = TF.resize(mask_tensor, [self.target_size, self.target_size], interpolation=T.InterpolationMode.NEAREST)

        return img_tensor, mask_tensor  # Both are tensors now

def main(args):
    # Validate train_test_val
    if args.train_test_val not in ["train", "test", "val"]:
        raise ValueError("train_test_val must be one of 'train', 'test', or 'val'")

    # Dataset and DataLoader
    patch_dir = config[f"{args.train_test_val}_wsi_processed_patch_save_dir"]
    dataset = WSIPatch2048Dataset(patch_dir, target_size=2048)  # Keep 2048x2048
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    # Iterate over batches
    for batch_idx, (img, mask) in enumerate(tqdm(dataloader, desc="Encoding patches")):
        img = img.to(DEVICE)  # Shape: [batch_size, 3, 2048, 2048]
        mask = mask.to(DEVICE)  # Shape: [batch_size, 1, 2048, 2048]
        print("img shape: ", img.shape)
        print("mask shape: ", mask.shape)
        break  # Remove this if you want to process all batches
    
    print("Encoding complete")

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # Argument parser
    parser = argparse.ArgumentParser(description="Process WSI patches")
    parser.add_argument("--config", type=str, default="configs/main_vqqan_effvit_kpi_slide.yaml", help="Path to YAML config file")
    parser.add_argument("--data_config", type=str, default="configs/kpis.yaml", help="Path to data YAML config file")
    parser.add_argument("--train_test_val", type=str, default="train", help="Specify train/test/val")
    
    args = parser.parse_args()

    # Load config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.data_config, 'r') as f:
        config.update(yaml.safe_load(f))
    
    # Validate config keys
    required_keys = ["data_dir", f"{args.train_test_val}_wsi_processed_patch_save_dir"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required config keys: {missing_keys}")

    main(args)