import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Union, Any
import sys
sys.path.append("src/includes/efficientvit")
# from efficientvit.models.efficientvit.seg import efficientvit_seg_b2  # Adjust based on your model choice

from typing import Optional

import torch
import torch.nn as nn


def efficientvit_seg_b2(**kwargs):
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2
    from efficientvit.models.efficientvit.seg import EfficientViTSeg
    from efficientvit.models.efficientvit.seg import SegHead 
    from efficientvit.models.utils import build_kwargs_from_config
    
    backbone = efficientvit_backbone_b2(**kwargs)

    head = SegHead(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[384, 192, 96],
        stride_list=[32, 16, 8],
        head_stride=8,
        head_width=96,
        head_depth=3,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
        n_classes=19,
        **build_kwargs_from_config(kwargs, SegHead),
    )
    model = EfficientViTSeg(backbone, head)
    
    return model 

def train_efficientvit_segmentation(
    dataloader,
    model_name: str = 'b2',
    dataset_name: str = 'cityscapes',
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir: str = './checkpoints',
    log_interval: int = 10
):
    """
    Trains the EfficientViT segmentation model using the specified dataset.

    Parameters:
        data_dir (str): Path to the directory containing WSI patches.
        model_name (str): Name of the EfficientViT model variant to use. Default is 'b2'.
        dataset_name (str): Name of the dataset. Default is 'cityscapes'.
        target_size (int): Target size for resizing images and masks. Default is 2048.
        batch_size (int): Number of samples per batch. Default is 1.
        num_epochs (int): Number of training epochs. Default is 10.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        device (str): Device to run the training on ('cuda' or 'cpu'). Default is 'cuda' if available.
        checkpoint_dir (str): Directory to save model checkpoints. Default is './checkpoints'.
        log_interval (int): Number of batches to wait before logging training status. Default is 10.
    """

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)


    model = efficientvit_seg_b2(pretrained=False)


    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, masks, filenames) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % log_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'efficientvit_{model_name}_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model checkpoint saved to {checkpoint_path}')

    print('Training complete.')
