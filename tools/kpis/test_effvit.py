import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from glob import glob
import argparse

from src.datasets.kpis.wsi_level import WSIPatchDataset
from tools.kpis import efficientvit_seg_b2

def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Load a trained EfficientViT segmentation model from a checkpoint.
    
    Parameters:
        checkpoint_path (str): Path to the model checkpoint file
        device (str): Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        model: The loaded model
    """
    # Initialize the model
    model = efficientvit_seg_b2(pretrained=False)
    
    # Load the checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Model loaded from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Move model to the specified device
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Test EfficientViT segmentation model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run inference on')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.device)
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    
    # Create dataset and dataloader    
    test_dataset = WSIPatchDataset(args.data_dir, transform=transform, mask_transform=mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Run inference
    with torch.no_grad():
        for i, (images, masks, filenames) in enumerate(test_loader):
            images = images.to(args.device)
            outputs = model(images)
            
            # Process outputs (assuming binary segmentation)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            # Save results
            for j, pred in enumerate(preds):
                pred_img = Image.fromarray((pred * 255).astype(np.uint8))
                save_path = os.path.join(args.output_dir, f"{os.path.basename(filenames[j])}_pred.png")
                pred_img.save(save_path)
            
            if i % 10 == 0:
                print(f"Processed {i}/{len(test_loader)} images")
    
    print("Testing complete.")

if __name__ == "__main__":
    main()
