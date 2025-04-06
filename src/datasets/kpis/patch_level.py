import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class PatchDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, mask_transform=None, return_masks=True):
        """
        Args:
            root_dir (str): Root directory of the dataset (kidney_pathology_image)
            split (str): 'train', 'test', or 'validation'
            transform: PyTorch transforms for the images
            mask_transform: PyTorch transforms for the masks (if different)
            return_masks (bool): Whether to return masks along with images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform or transform  # Use same transform if mask_transform not provided
        self.return_masks = return_masks
        
        # Path to Task1_patch_level for the given split
        self.patch_dir = os.path.join(root_dir, split, 'Task1_patch_level', split)
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.png', '.tiff', '.jpeg'}
        
        # Categories (56Nx, DN, NEP25, normal)
        self.categories = ['56Nx', 'DN', 'NEP25', 'normal']
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # Collect all image paths and their labels
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        
        for category in self.categories:
            category_dir = os.path.join(self.patch_dir, category)
            if not os.path.exists(category_dir):
                continue
                
            # Each category has subdirectories (e.g., 12-299, 11-362)
            for subdir in os.listdir(category_dir):
                img_dir = os.path.join(category_dir, subdir, 'img')
                mask_dir = os.path.join(category_dir, subdir, 'mask')

                print(f"Processing category: {subdir}")
                print(f"Image directory: {img_dir}")
                print(f"Mask directory: {mask_dir}")
                
                if not os.path.exists(img_dir) or (self.return_masks and not os.path.exists(mask_dir)):
                    continue
                    
                # Get image files
                img_files = sorted([f for f in os.listdir(img_dir) 
                                  if os.path.splitext(f)[1].lower() in self.image_extensions])
                
                for img_file in img_files:
                    print(f"Processing image: {img_file}")
                    self.image_paths.append(os.path.join(img_dir, img_file))
                    if self.return_masks:
                        mask_file = os.path.join(mask_dir, img_file)  # Assumes mask has same filename
                        self.mask_paths.append(mask_file)
                    self.labels.append(self.category_to_idx[category])
        
        print(f"Loaded {len(self.image_paths)} patch images from {split} split")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms to image
        if self.transform:
            image = self.transform(image)
            
        # Get label
        label = self.labels[idx]
        
        if self.return_masks:
            # Load mask
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert('L')  # 'L' mode for grayscale
            
            # Apply transforms to mask
            if self.mask_transform:
                mask = self.mask_transform(mask)
                
            return image, mask, label
        else:
            return image, label

def get_patch_dataloader(root_dir, split='train', batch_size=32, shuffle=True, num_workers=4, return_masks=True):
    """
    Create a DataLoader for patch-level data
    
    Args:
        root_dir (str): Root directory of the dataset
        split (str): 'train', 'test', or 'validation'
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for loading data
        return_masks (bool): Whether to include masks in the data
    
    Returns:
        DataLoader object
    """
    # Define transforms for images
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Define transforms for masks (no normalization needed)
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create dataset
    dataset = PatchDataset(
        root_dir=root_dir,
        split=split,
        transform=image_transform,
        mask_transform=mask_transform if return_masks else None,
        return_masks=return_masks
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU if available
    )
    
    return dataloader

# Example usage
if __name__ == "__main__":
    dataset_path = "/project/hnguyen2/mvu9/datasets/kidney_pathology_image/"
    
    # Create dataloaders for all splits
    train_loader = get_patch_dataloader(dataset_path, split='train', batch_size=32)
    test_loader = get_patch_dataloader(dataset_path, split='test', batch_size=32)
    val_loader = get_patch_dataloader(dataset_path, split='validation', batch_size=32)
    
    # Example: Iterate through one batch
    for batch in train_loader:
        if len(batch) == 3:  # With masks
            images, masks, labels = batch
            print(f"Images shape: {images.shape}")  # [batch_size, channels, height, width]
            print(f"Masks shape: {masks.shape}")    # [batch_size, 1, height, width]
            print(f"Labels: {labels}")
        else:  # Without masks
            images, labels = batch
            print(f"Images shape: {images.shape}")
            print(f"Labels: {labels}")
        break