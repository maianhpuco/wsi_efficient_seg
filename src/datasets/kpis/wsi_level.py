import os
from glob import glob
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, Resize, ScaleIntensity, EnsureChannelFirst, ToTensor
import tifffile
import scipy.ndimage as ndi
import numpy as np

class WSITIFFDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None, resize_factor=0.5):
        """
        Args:
            data_dir (str): Root directory containing folders with 'img/*.tiff' and 'mask/*mask.tiff'
            transform: MONAI transforms for image
            mask_transform: MONAI transforms for mask
            resize_factor: Resizing factor for TIFF image downsampling
        """
        self.image_paths = []
        self.mask_paths = []
        self.transform = transform
        self.mask_transform = mask_transform if mask_transform else transform
        self.resize_factor = resize_factor

        types = glob(os.path.join(data_dir, '*'))
        
        for folder in types:
            self.image_paths.extend(glob(os.path.join(folder, '*_wsi.tiff')))
            self.mask_paths.extend(glob(os.path.join(folder, '*_mask.tiff')))

        self.image_paths = sorted(self.image_paths)
        self.mask_paths = sorted(self.mask_paths)

        assert len(self.image_paths) == len(self.mask_paths), f"Mismatch: {len(self.image_paths)} images, {len(self.mask_paths)} masks"

    def __len__(self):
        return len(self.image_paths)

    def load_and_resize_tiff(self, path, level=1, is_mask=False):
        # Load the first page from multi-page TIFF
        image = tifffile.imread(path, key=0)

        # Apply zoom (scipy.ndimage.zoom) for resizing
        zoom_factor = self.resize_factor
        print(f"Loading {path} with zoom factor {zoom_factor}")
        if is_mask:
            image = ndi.zoom(image, (zoom_factor, zoom_factor), order=0)  # nearest neighbor for masks
        else:
            image = ndi.zoom(image, (zoom_factor, zoom_factor, 1), order=1)  # bilinear for RGB
        return image.astype(np.float32)

    def __getitem__(self, idx):
        image = self.load_and_resize_tiff(self.image_paths[idx], is_mask=False)
        mask = self.load_and_resize_tiff(self.mask_paths[idx], is_mask=True)
        
        print("----shape of the image and mask----", image.shape, mask.shape)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


def get_monai_tiff_dataloader(data_dir, batch_size=1, shuffle=False, num_workers=0):
    # Define MONAI-compatible transforms
    image_transforms = Compose([
        EnsureChannelFirst(),
        Resize((512, 512)),
        ScaleIntensity(),
        ToTensor()
    ])

    mask_transforms = Compose([
        EnsureChannelFirst(),
        Resize((512, 512)),
        ToTensor()
    ])

    dataset = WSITIFFDataset(data_dir, transform=image_transforms, mask_transform=mask_transforms)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return loader

#  Example usage
if __name__ == "__main__": 
    print("Loading dataset...") 
    data_dir = "/project/hnguyen2/mvu9/datasets/kidney_pathology_image/train/Task2_WSI_level"
    img_path_example = '/project/hnguyen2/mvu9/datasets/kidney_pathology_image/train/Task2_WSI_level/normal/normal_F3_wsi.tiff'
    image = tifffile.imread(img_path_example, key=0) 
    print("Image shape:", image.shape)
     
    # dataloader = get_monai_tiff_dataloader(data_dir, batch_size=1)

    # for img, mask in dataloader:
    #     print("Image shape:", img.shape)
    #     print("Mask shape:", mask.shape)
    #     break