import torch
import torch.nn as nn
from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0
from efficientvit.models.efficientvit.seg import EfficientViTSeg, SegHead
from efficientvit.models.utils import build_kwargs_from_config


def efficientvit_seg_from_index_b0(num_codes: int, num_classes: int, embed_dim: int = 128, **kwargs):
    # 1. Index embedding layer
    embedding_layer = nn.Embedding(num_codes, embed_dim)

    # 2. EfficientViT backbone that accepts embedded input
    backbone = efficientvit_backbone_b0(in_channels=embed_dim, **kwargs)

    # 3. Segmentation head
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

    model = EfficientViTSeg(backbone, head)

    return nn.Sequential(
        embedding_layer,  # [B, 64, 64] → [B, embed_dim, 64, 64]
        model,            # → [B, num_classes, 64, 64]
        nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)  # → [B, num_classes, 2048, 2048]
    )


# Example usage
model = efficientvit_seg_from_index_b0(
    num_codes=1024,
    num_classes=2,
    embed_dim=128
)

if __name__ == "__main__":
        # Example input: random indices 
    indices = torch.randint(0, 1024, (1, 64, 64))  # example index map
    output_mask = model(indices)  # [1, 2, 2048, 2048]
    print("Output mask shape:", output_mask.shape)
 