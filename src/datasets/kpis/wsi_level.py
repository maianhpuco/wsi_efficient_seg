import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class WSIDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset (kidney_pathology_image)
            split (str): 'train', 'test', or 'validation'
            transform: PyTorch transforms for the images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Path to Task2_WSI_level for the given split
        self.wsi_dir = os.path.join(root_dir, split, 'Task2_WSI_level')
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.png', '.tiff', '.jpeg'}
        
        # Categories (56Nx, DN, NEP25, normal)
        self.categories = ['56Nx', 'DN', 'NEP25', 'normal']
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # Collect all image paths and their labels
        self.image_paths = []
        self.labels = []
        
        for category in self.categories:
            category_dir = os.path.join(self.wsi_dir, category)
            if not os.path.exists(category_dir):
                continue
                
            for file_name in os.listdir(category_dir):
                if os.path.splitext(file_name)[1].lower() in self.image_extensions:
                    self.image_paths.append(os.path.join(category_dir, file_name))
                    self.labels.append(self.category_to_idx[category])
        
        print(f"Loaded {len(self.image_paths)} WSI images from {split} split")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
            
        # Get label
        label = self.labels[idx]
        
        return image, label

def get_wsi_dataloader(root_dir, split='train', batch_size=4, shuffle=True, num_workers=2):
    """
    Create a DataLoader for WSI-level data
    
    Args:
        root_dir (str): Root directory of the dataset
        split (str): 'train', 'test', or 'validation'
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for loading data
    
    Returns:
        DataLoader object
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Create dataset
    dataset = WSIDataset(root_dir=root_dir, split=split, transform=transform)
    
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
    train_loader = get_wsi_dataloader(dataset_path, split='train', batch_size=4)
    test_loader = get_wsi_dataloader(dataset_path, split='test', batch_size=4)
    val_loader = get_wsi_dataloader(dataset_path, split='validation', batch_size=4)
    
    # Example: Iterate through one batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")  # [batch_size, channels, height, width]
        print(f"Labels: {labels}")
        break