import torch
import torch.nn as nn
import torch.nn.functional as F


class ObsEncoder(nn.Module):
    def __init__(self, emb_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))   # collapse H×W → 1×1
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, emb_dim)
        )


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Tensor of shape (B, T, V, 224, 224, 3), dtype torch.ByteTensor
        Returns:
            emb: Tensor of shape (B, 1, emb_dim)
        """
        x = obs[:, -1, -1]  # (B, 224, 224, 3)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 3, 224, 224)
        # Convert to float and normalize to [0,1]
        x = x.float() / 255.0

        # CNN encoding
        x = self.encoder(x)  # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 64)
        
        # FC layers
        emb = self.fc(x)  # (B, emb_dim)
        emb = emb.unsqueeze(1)  # (B, 1, emb_dim)

        return emb

