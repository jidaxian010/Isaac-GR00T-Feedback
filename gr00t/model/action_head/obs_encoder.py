import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights


class ObsEncoder(nn.Module):
    def __init__(self, emb_dim: int = 512, use_pretrained: bool = True, resnet_type: str = "resnet18"):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_pretrained = use_pretrained
        self.resnet_type = resnet_type

        # Load pre-trained ResNet
        if resnet_type == "resnet18":
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
            feature_dim = 512  # ResNet18 feature dimension
        elif resnet_type == "resnet50":
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
            feature_dim = 2048  # ResNet50 feature dimension
        else:
            raise ValueError(f"Unknown resnet_type: {resnet_type}")

        # Remove the final fully connected layer and average pooling
        # Keep everything up to avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        self.feature_dim = feature_dim

        # Add adaptive pooling and projection to desired embedding dimension
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(feature_dim, emb_dim)

        # Initialize projection layer
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        # Track if ResNet has been reloaded
        self._resnet_reloaded = False
        self._projection_reloaded = False

    def _reload_resnet(self):
        """Reload ResNet backbone if weights became NaN (e.g., from meta tensor loading)"""
        # Get current device
        device = next(self.backbone.parameters()).device

        # Reload pre-trained ResNet
        if self.resnet_type == "resnet18":
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if self.use_pretrained else None)
        elif self.resnet_type == "resnet50":
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if self.use_pretrained else None)
        else:
            raise ValueError(f"Unknown resnet_type: {self.resnet_type}")

        # Create new backbone
        new_backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Copy weights to the same device
        new_backbone = new_backbone.to(device)

        # Replace the backbone
        self.backbone = new_backbone
        self._resnet_reloaded = True
        print(f"DEBUG: ResNet backbone reloaded successfully on device {device}")

    def _reload_projection(self, device):
        """Reload projection layer if weights became NaN (e.g., from meta tensor loading)"""
        # Create new projection layer
        new_projection = nn.Linear(self.feature_dim, self.emb_dim)

        # Initialize properly
        nn.init.xavier_uniform_(new_projection.weight)
        nn.init.zeros_(new_projection.bias)

        # Move to correct device
        new_projection = new_projection.to(device)

        # Replace the projection
        self.projection = new_projection
        self._projection_reloaded = True
        print(f"DEBUG: Projection layer reloaded successfully on device {device}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Tensor of shape (B, T, V, 224, 224, 3), dtype torch.ByteTensor
        Returns:
            emb: Tensor of shape (B, 1, emb_dim)
        """
        x = obs[:, -1, -1]  # (B, 224, 224, 3), eye-in-hand

        # Fix the permute to ensure correct channel dimension
        if x.shape[-1] == 3:  # If channels are in the last dimension
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, 3, 224, 224)
        else:  # If channels are already in the second dimension
            x = x.contiguous()  # (B, 3, 224, 224)

        # Convert to float and normalize for ImageNet pre-trained models
        # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        x = x.float() / 255.0

        # Check for NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"DEBUG: NaN/Inf in x after float conversion!")
            x = torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Check after normalization
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"DEBUG: NaN/Inf in x after normalization! Range: [{x.min().item()}, {x.max().item()}]")
            x = torch.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)

        # ResNet backbone
        features = self.backbone(x)  # (B, feature_dim, H, W)

        # Check after backbone
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(
                f"DEBUG: NaN/Inf in features after backbone! Range: [{features.min().item()}, {features.max().item()}]"
            )
            # Check ResNet weights and reload if needed
            has_nan_weights = False
            for name, param in self.backbone.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"DEBUG: NaN/Inf in ResNet parameter: {name}")
                    has_nan_weights = True

            # Reload ResNet if weights are NaN (happens when model is loaded from checkpoint with meta tensors)
            if has_nan_weights and not self._resnet_reloaded:
                print(f"DEBUG: Reloading ResNet backbone due to NaN weights...")
                self._reload_resnet()
                # Retry forward pass
                features = self.backbone(x)
                if torch.isnan(features).any() or torch.isinf(features).any():
                    print(f"DEBUG: Still NaN after reload, using nan_to_num")
                    features = torch.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
            else:
                features = torch.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)

        # Global average pooling
        features = self.pool(features)  # (B, feature_dim, 1, 1)
        features = features.view(features.size(0), -1)  # (B, feature_dim)

        # Check after pooling
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(
                f"DEBUG: NaN/Inf in features after pooling! Range: [{features.min().item()}, {features.max().item()}]"
            )
            features = torch.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)

        # Check projection weights before forward pass
        if torch.isnan(self.projection.weight).any() or torch.isinf(self.projection.weight).any():
            if not self._projection_reloaded:
                print(f"DEBUG: Projection weight has NaN/Inf! Reloading projection layer...")
                device = next(self.projection.parameters()).device
                self._reload_projection(device)

        # Project to desired embedding dimension
        emb = self.projection(features)  # (B, emb_dim)

        # Check after projection
        if torch.isnan(emb).any() or torch.isinf(emb).any():
            print(f"DEBUG: NaN/Inf in emb after projection!")
            print(f"DEBUG: Features range: [{features.min().item()}, {features.max().item()}]")
            # Check if projection weights are still NaN
            if torch.isnan(self.projection.weight).any() or torch.isinf(self.projection.weight).any():
                print(f"DEBUG: Projection weight still has NaN/Inf after reload! Trying reinitialization...")
                device = next(self.projection.parameters()).device
                with torch.no_grad():
                    # Try to reinitialize in-place
                    self.projection.weight.data = torch.randn_like(self.projection.weight.data)
                    nn.init.xavier_uniform_(self.projection.weight)
                    nn.init.zeros_(self.projection.bias)
                emb = self.projection(features)  # Recompute
                if torch.isnan(emb).any() or torch.isinf(emb).any():
                    print(f"DEBUG: Still NaN after reinit, using nan_to_num")
                    emb = torch.nan_to_num(emb, nan=0.0, posinf=10.0, neginf=-10.0)
            else:
                # Features might have NaN
                if torch.isnan(features).any() or torch.isinf(features).any():
                    print(f"DEBUG: Features have NaN/Inf! Cleaning...")
                    features = torch.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
                    emb = self.projection(features)
                emb = torch.nan_to_num(emb, nan=0.0, posinf=10.0, neginf=-10.0)

        # Scale and bound output to [-10, 10] range
        # ResNet features are typically well-scaled, so this should work well
        emb = torch.tanh(emb) * 10.0  # Bound to [-10, 10]

        # Final check
        if torch.isnan(emb).any() or torch.isinf(emb).any():
            print(f"DEBUG: NaN/Inf in final emb!")
            emb = torch.nan_to_num(emb, nan=0.0, posinf=10.0, neginf=-10.0)

        emb = emb.unsqueeze(1)  # (B, 1, emb_dim)

        return emb
