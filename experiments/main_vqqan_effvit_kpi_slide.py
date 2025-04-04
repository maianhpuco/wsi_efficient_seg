import os 
from torch.utils.data import DataLoader 
from src.datasets.kpis.wsi_level import WSIDataset, get_wsi_dataloader 
# from src .datasets.kpis.patch_level import PatchDataset 
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