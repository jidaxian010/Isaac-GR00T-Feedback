import torch
import torch.nn as nn
from torchvision.models import resnet18

class ObsEncoder(nn.Module):
    def __init__(self, emb_dim: int = 512, pretrained: bool = True):
        super(ObsEncoder, self).__init__()
        # Initialize ResNet18 backbone
        backbone = resnet18(pretrained=pretrained)
        # Extract feature dimension
        self.feature_dim = backbone.fc.in_features
        # Remove the original classifier
        backbone.fc = nn.Identity()
        self.backbone = backbone
        # Projection layer to embedding dimension
        self.projector = nn.Linear(self.feature_dim, emb_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Tensor of shape (B, T, V, 224, 224, 3), dtype torch.ByteTensor
        Returns:
            emb: Tensor of shape (B, emb_dim)
        """
        x = obs[:, -1, -1]  # (B, 224, 224, 3)
        print("ObsEncoder input NaN?", torch.isnan(x).any().item())
        # # Reorder to (B, 3, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 3, 224, 224)
        # Convert to float and normalize to [0,1]
        x = x.float() / 255.0
        print("ObsEncoder normalized NaN?", torch.isnan(x).any().item())
        print("ObsEncoder normalized min/max/mean:", x.min().item(), x.max().item(), x.mean().item())
        # Match backbone parameter dtype (e.g., float32 or float16)
        target_dtype = next(self.backbone.parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)
        # Extract features
        self.backbone.eval()
        features = self.backbone(x)
        print("ObsEncoder backbone features NaN?", torch.isnan(features).any().item())
        # Project to embedding
        emb = self.projector(features)
        print("ObsEncoder projector emb NaN?", torch.isnan(emb).any().item())
        emb = emb.unsqueeze(1)
        return emb

