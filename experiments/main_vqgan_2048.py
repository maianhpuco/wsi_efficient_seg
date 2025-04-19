import os
import sys 
from glob import glob
import yaml
import torch
import argparse
from tqdm import tqdm 

from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Union, Any

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
print(f"Project root added to sys.path: {PROJECT_ROOT}") 
from src.datasets.kpis.vqgan_indexed_dataset import  VQGANIndexedDataset


# Device setup
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

def main(args):
    # Validate train_test_val
    if args.train_test_val not in ["train", "test", "val"]:
        raise ValueError("train_test_val must be one of 'train', 'test', or 'val'")
        # Define transformations
    img_transform = transforms.Compose([
        # transforms.Resize((512, 512)),  # Adjust size as needed 
        transforms.ToTensor(),  
    ])
    mask_transform = transforms.Compose([
        # transforms.Resize((512, 512), interpolation=Image.NEAREST),  # Preserve label values
        # No ToTensor
    ]) 
    # Dataset and DataLoader
    patch_dir = config[f"{args.train_test_val}_wsi_processed_patch_save_dir"]
    dataset = VQGANIndexedDataset(
        patch_dir, 
        target_size=512, # target_size=2048, 
        img_transform=img_transform, 
        mask_transform=None
        )  # Keep 2048x2048
    # dataset = WSIPatch2048Dataset(
    dataset = VQGANIndexedDataset(  
        patch_dir, 
        vqgan_model=None, 
        target_size=2048, 
        patch_size=256, 
        stride=256, 
        img_transform=None, 
        mask_transform=None
    )
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=torch.cuda.is_available()
    # )

    
    # # Iterate over batches
    # for batch_idx, (img, mask, filename) in enumerate(tqdm(dataloader, desc="Reading patches")):
    #     # print("Image file name:", filename)
    #     img = img.to(DEVICE)  # Shape: [batch_size, 3, 2048, 2048]
    #     mask = mask.to(DEVICE)  # Shape: [batch_size, 1, 2048, 2048]
        
    #     print("Check shape of image and mask")
    #     print("img shape: ", img.shape)
    #     print("mask shape: ", mask.shape)
        
    #     break  # Remove this if you want to process all batches


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Process WSI patches")
    parser.add_argument("--config", type=str, default="configs/main_effvit.yaml", help="Path to YAML config file")
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
 
    args.checkpoint_dir = config.get('checkpoint_dir')
 
    main(args)
