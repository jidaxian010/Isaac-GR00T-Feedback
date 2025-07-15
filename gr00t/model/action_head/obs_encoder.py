import torch
import torch.nn as nn
import torchvision.models as models


# Fast image perception
class ObsEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # Assume input is (B, T, V, 224, 224, 3)
        self.input_dim = 224 * 224 * 3
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        # x: (B, T, V, 224, 224, 3)
        # Extract the last frame (T-1) for each batch and view
        x = x[:, -1]  # (B, V, 224, 224, 3)
        B, V, H, W, C = x.shape
        x = x.view(B * V, H * W * C)  # Flatten each image
        features = self.mlp(x)  # (B*V, emb_dim)
        features = features.view(B, V, -1)  # (B, V, emb_dim)
        features = features.mean(dim=1)     # (B, emb_dim)
        return features
        
        

