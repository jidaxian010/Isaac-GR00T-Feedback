import torch
import torch.nn as nn
import torch.nn.functional as F


class ObsEncoder(nn.Module):
    def __init__(self, emb_dim: int = 512):
        super().__init__()
        # Very simple CNN structure to avoid NaN issues
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
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, emb_dim)
        )
        
        # Proper weight initialization
        self._init_weights()
    
    def _print_cnn_weights(self):
        """Print CNN weights to check initialization"""
        pass
    

    
    def _init_weights(self):
        """Initialize weights properly to avoid NaN values"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Use a more conservative initialization
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        



    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Tensor of shape (B, T, V, 224, 224, 3), dtype torch.ByteTensor
        Returns:
            emb: Tensor of shape (B, 1, emb_dim)
        """
        x = obs[:, -1, -1]  # (B, 224, 224, 3)
        
        # Fix the permute to ensure correct channel dimension
        if x.shape[-1] == 3:  # If channels are in the last dimension
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, 3, 224, 224)
        else:  # If channels are already in the second dimension
            x = x.contiguous()  # (B, 3, 224, 224)
        
        # Convert to float and normalize to [0,1]
        x = x.float() / 255.0

        # CNN encoding
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            
            # Clip values to prevent explosion
            if isinstance(layer, nn.Conv2d):
                x = torch.clamp(x, -10.0, 10.0)
        
        x = x.view(x.size(0), -1)  # (B, 256)
        
        # FC layers
        emb = self.fc(x)  # (B, emb_dim)
        
        # Clip values to prevent explosion
        emb = torch.clamp(emb, -10.0, 10.0)
        
        emb = emb.unsqueeze(1)  # (B, 1, emb_dim)

        return emb

###END OF FILE###