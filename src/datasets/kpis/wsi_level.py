import os
from glob import glob
from monai.transforms import (
    LoadImage,
    Compose,
    ScaleIntensity,
    Resize,
    EnsureChannelFirst,
    ToTensor
)
from monai.data import DataLoader, ArrayDataset
import torch


def get_monai_wsi_dataloader(data_dir, batch_size=1, shuffle=False, num_workers=0):
    """
    Create a MONAI DataLoader for WSI-level data.

    Args:
        data_dir (str): Directory containing the WSI data
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for loading data

    Returns:
        DataLoader: MONAI DataLoader for WSI images and masks
    """
    image_paths = []
    mask_paths = []

    types = glob(os.path.join(data_dir, '*'))
    for folder in types:
        image_paths.extend(glob(os.path.join(folder, '*_wsi.tiff')))
        mask_paths.extend(glob(os.path.join(folder, '*_mask.tiff')))
        
    print("Len of image and mask: ", len(image_paths), len(mask_paths)) 
    
    # image_paths = sorted(image_paths)
    # mask_paths = sorted(mask_paths)

    # assert len(image_paths) == len(mask_paths), f"Found {len(image_paths)} images and {len(mask_paths)} masks"

    # # Define MONAI transforms
    # image_transforms = Compose([
    #     LoadImage(image_only=True),
    #     EnsureChannelFirst(),
    #     Resize((512, 512)),
    #     ScaleIntensity(),
    #     ToTensor()
    # ])

    # mask_transforms = Compose([
    #     LoadImage(image_only=True),
    #     EnsureChannelFirst(),
    #     Resize((512, 512)),
    #     ToTensor()
    # ])

    # # Create MONAI dataset
    # dataset = ArrayDataset(image_paths, image_transforms, mask_paths, mask_transforms)

    # # Create MONAI DataLoader
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=shuffle,
    #     num_workers=num_workers,
    #     pin_memory=torch.cuda.is_available()
    # )

    # return dataloader


# Test run
if __name__ == "__main__":
    data_dir = "/project/hnguyen2/mvu9/datasets/kidney_pathology_image/train/Task2_WSI_level"
    dataloader = get_monai_wsi_dataloader(data_dir, batch_size=1)

    # for images, masks in dataloader:
    #     print(f"Image batch shape: {images.shape}")
    #     print(f"Mask batch shape: {masks.shape}")
    #     break
