import torch
import torch.nn as nn
import torch.nn.functional as F

class Index2Segmentation(nn.Module):
    def __init__(self, num_codes, embed_dim=256, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, embed_dim)  # e.g. 1024 codes

        self.backbone = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 64 → 128
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 → 256
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 256 → 512
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=4, padding=0),  # 512 → 2048
        )

    def forward(self, x_idx):  # [B, 64, 64]
        x = self.embedding(x_idx)  # [B, 64, 64, C]
        x = x.permute(0, 3, 1, 2)  # [B, C, 64, 64]
        x = self.backbone(x)       # [B, 512, 64, 64]
        out = self.decoder(x)      # [B, num_classes, 2048, 2048]
        return out
