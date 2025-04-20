import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Union, Any

# Add EfficientViT to import path
sys.path.append("src/includes/efficientvit")

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



if __name__ == "__main__":
    
    model = Index1DToSegmentation( # this is just untrained model, can be replace by trained model 
                num_codes=1024,
                num_classes=2,
                embed_dim=256
            ) 
    indices = torch.randint(0, 1024, (1, 64))  # 1D vector input: [B, 64]
    print("Input indices shape:", indices.shape)
    output_mask = model(indices)              # [1, 2, 2048, 2048]
    print("Output mask shape:", output_mask.shape)
