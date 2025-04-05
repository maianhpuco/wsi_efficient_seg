import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import yaml
import argparse

class WSIPatchDataset(Dataset):
    def __init__(self, patch_dir, transform=None, mask_transform=None):
        """
        Args:
            patch_dir (str): Directory containing pre-extracted WSI patches.
            transform (callable, optional): Transforms for images.
            mask_transform (callable, optional): Transforms for masks.
        """
        self.patch_dir = patch_dir
        self.transform = transform
        self.mask_transform = mask_transform or transform  # Default to image transform if not provided

        # Find all image patches
        self.image_paths = sorted(glob(os.path.join(patch_dir, "**/*_img.png"), recursive=True))
        self.mask_paths = [p.replace("_img.png", "_mask.png") for p in self.image_paths]

        # Validate that each image has a corresponding mask
        missing_masks = [p for p, m in zip(self.image_paths, self.mask_paths) if not os.path.exists(m)]
        if missing_masks:
            raise ValueError(f"Missing mask files for: {missing_masks[:5]} (and possibly more)")

        print(f"Loaded {len(self.image_paths)} patch pairs from {patch_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Load as RGB

        # Load mask
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert("L")  # Load as grayscale

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

def get_wsi_patch_dataloader(patch_dir, batch_size=4, shuffle=True, num_workers=4, transform=None, mask_transform=None):
    """
    Create a DataLoader for WSI patches.
    
    Args:
        patch_dir (str): Directory with pre-extracted patches.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes for data loading.
        transform (callable, optional): Transforms for images.
        mask_transform (callable, optional): Transforms for masks.
    
    Returns:
        DataLoader: PyTorch DataLoader for the patch dataset.
    """
    # Default transforms if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor (0-1 range)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    if mask_transform is None:
        mask_transform = transforms.Compose([
            transforms.ToTensor()  # Convert to tensor (0-1 range)
        ])

    # Create dataset
    dataset = WSIPatchDataset(patch_dir, transform=transform, mask_transform=mask_transform)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Faster data transfer to GPU if available
    )
    return dataloader

if __name__ == "__main__":
    # Argument parser for config and train/test/val split
    parser = argparse.ArgumentParser(description="Load WSI patch dataset")
    parser.add_argument("--config", type=str, default="configs/kpis.yaml", help="Path to YAML config file")
    parser.add_argument("--train_test_val", type=str, default="train", help="Specify train/test/val")
    args = parser.parse_args()

    # Load config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required config keys
    required_keys = ["data_dir", f"{args.train_test_val}_wsi_processed_patch_save_dir"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required config keys: {missing_keys}")

    # Validate train_test_val argument
    if args.train_test_val not in ["train", "test", "val"]:
        raise ValueError("train_test_val must be one of 'train', 'test', or 'val'")

    # Get patch directory
    patch_dir = config[f"{args.train_test_val}_wsi_processed_patch_save_dir"]

    # Create DataLoader
    dataloader = get_wsi_patch_dataloader(
        patch_dir,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )

    # Test the DataLoader
    for i, (images, masks) in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print(f"  Image shape: {images.shape}")  # e.g., [4, 3, 2048, 2048]
        print(f"  Mask shape: {masks.shape}")    # e.g., [4, 1, 2048, 2048]
        break  # Just test one batch