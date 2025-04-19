import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Union, Any
import sys
sys.path.append("src/includes/efficientvit")

from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0
from efficientvit.models.efficientvit.seg import EfficientViTSeg, SegHead
from efficientvit.models.utils import build_kwargs_from_config


class Index1DToSegmentation(nn.Module):
    def __init__(self, num_codes: int, num_classes: int, embed_dim: int = 256, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, embed_dim)  # [B, 64] → [B, 64, 256]
        self.project = nn.Linear(embed_dim, 64 * 64)         # → [B, 64, 4096]
        self.reshape = lambda x: x.view(-1, 1, 64, 64)        # → [B, 1, 64, 64]

        # Backbone expecting 1 input channel
        backbone = efficientvit_backbone_b0(in_channels=1, **kwargs)

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
        self.upsample = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)

    def forward(self, x):  # x: [B, 64]
        x = self.embedding(x)                   # [B, 64, 256]
        x = self.project(x)                     # [B, 64, 4096]
        x = x.view(x.size(0), 1, 64, 64)        # [B, 1, 64, 64]
        x = self.seg_model(x)                   # [B, num_classes, 64, 64]
        return self.upsample(x)                 # [B, num_classes, 2048, 2048]


# Instantiate model
model = Index1DToSegmentation(
    num_codes=1024,
    num_classes=2,
    embed_dim=256
)

if __name__ == "__main__":
    indices = torch.randint(0, 1024, (1, 64))  # 1D vector input: [B, 64]
    output_mask = model(indices)              # [1, 2, 2048, 2048]
    print("Output mask shape:", output_mask.shape)
