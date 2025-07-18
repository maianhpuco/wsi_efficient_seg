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

def efficientvit_seg_custom(**kwargs):
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0
    from efficientvit.models.efficientvit.seg import EfficientViTSeg, SegHead
    from efficientvit.models.utils import build_kwargs_from_config


    class Index1DToSegmentation(nn.Module):
        def __init__(self, num_codes: int, num_classes: int, embed_dim: int = 256, **kwargs):
            super().__init__()
            self.embedding = nn.Embedding(num_codes, embed_dim)  # [B, 64] → [B, 64, 256]

            # Backbone expecting input shape: [B, 256, 64, 64]
            backbone = efficientvit_backbone_b0(in_channels=embed_dim, **kwargs)

            head = SegHead(
                fid_list=["stage4", "stage3", "stage2"],
                in_channel_list=[128, 64, 32],
                stride_list=[32, 16, 8],
                head_stride=8,
                head_width=64,
                head_depth=2,
                expand_ratio=4,
                middle_op="mbconv",
                final_expand=4,
                n_classes=num_classes,
                **build_kwargs_from_config(kwargs, SegHead),
            )

            self.seg_model = EfficientViTSeg(backbone, head)
            self.upsample = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)  # 64 → 2048

        def forward(self, x):  # x: [B, 64]
            x = self.embedding(x)                     # [B, 64, 256]
            x = x.view(x.size(0), 8, 8, -1)           # [B, 8, 8, 256]
            x = x.permute(0, 3, 1, 2).contiguous()    # [B, 256, 8, 8]
            x = nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)  # [B, 256, 64, 64]
            x = self.seg_model(x)                     # [B, num_classes, 64, 64]
            return self.upsample(x)                   # [B, num_classes, 2048, 2048]


    # Instantiate model
    model = Index1DToSegmentation(
        num_codes=1024,
        num_classes=2,
        embed_dim=256
    ) 
    return model

def efficientvit_seg_l2(**kwargs):
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l2
    from efficientvit.models.efficientvit.seg import EfficientViTSeg
    from efficientvit.models.efficientvit.seg import SegHead 
    from efficientvit.models.utils import build_kwargs_from_config
    
    backbone = efficientvit_backbone_l2(**kwargs)
    # backbone = EfficientViTLargeBackbone(
    #     width_list=[32, 64, 128, 256, 512],
    #     depth_list=[1, 2, 2, 8, 8],
    #     **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    # )
    head = SegHead(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],   # Match backbone channel dims
        stride_list=[32, 16, 8],
        head_stride=1,
        # head_stride=8,
        head_width=128,                   # Wider than b2's 96
        head_depth=3,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
        n_classes=2,
        **build_kwargs_from_config(kwargs, SegHead),
    ) 
    model = EfficientViTSeg(backbone, head)
    
    return model 
 

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
        n_classes=2,
        **build_kwargs_from_config(kwargs, SegHead),
    )
    model = EfficientViTSeg(backbone, head)
    
    return model 


def train_efficientvit_segmentation(
    dataloader,
    model_name: str = 'b2',
    dataset_name: str = 'kpis',
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

    if model_name == 'b2': 
        model = efficientvit_seg_b2(pretrained=False)
    elif model_name == 'l2': 
        model = efficientvit_seg_l2(pretrained=False) 
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Choose 'b2' or 'l2'.") 

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
            print("- input shape: ", images.shape)
            print("- outputs shape: ", outputs.shape)
            print("- masks shape: ", masks.shape)
            
            # - input shape:  torch.Size([1, 3, 2048, 2048])
            # - outputs shape:  torch.Size([1, 2, 256, 256])
            # - masks shape:  torch.Size([1, 2048, 2048]) 
            # Reshape masks to match model output size        
            loss = criterion(outputs, masks)
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
