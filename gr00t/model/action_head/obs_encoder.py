import torch
import torch.nn as nn
import torch.nn.functional as F


class ObsEncoder(nn.Module):
    def __init__(self, emb_dim: int = 512):
        super().__init__()
        # Very simple CNN structure to avoid NaN issues
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
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
        
        # Force materialization using to_empty()
        self.to_empty(device='cpu')
        if torch.cuda.is_available():
            self.to('cuda')
        
        # Re-initialize weights after materialization
        self._init_weights()
        
        # Print network structure and weights for debugging
        self._print_cnn_weights()
    
    def _print_cnn_weights(self):
        """Print CNN weights to check initialization"""
        print("[DEBUG] === CNN Structure and Weights ===")
        for i, layer in enumerate(self.encoder):
            print(f"[DEBUG] Layer {i}: {layer}")
            if hasattr(layer, 'weight'):
                weight = layer.weight.data
                print(f"[DEBUG]   Weight shape: {weight.shape}")
                print(f"[DEBUG]   Weight range: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
                print(f"[DEBUG]   Weight mean: {weight.mean().item():.6f}")
                print(f"[DEBUG]   Weight std: {weight.std().item():.6f}")
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias = layer.bias.data
                print(f"[DEBUG]   Bias shape: {bias.shape}")
                print(f"[DEBUG]   Bias range: [{bias.min().item():.6f}, {bias.max().item():.6f}]")
        print("[DEBUG] === FC Layers ===")
        for i, layer in enumerate(self.fc):
            print(f"[DEBUG] Layer {i}: {layer}")
            if hasattr(layer, 'weight'):
                weight = layer.weight.data
                print(f"[DEBUG]   Weight shape: {weight.shape}")
                print(f"[DEBUG]   Weight range: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
                print(f"[DEBUG]   Weight mean: {weight.mean().item():.6f}")
                print(f"[DEBUG]   Weight std: {weight.std().item():.6f}")
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias = layer.bias.data
                print(f"[DEBUG]   Bias shape: {bias.shape}")
                print(f"[DEBUG]   Bias range: [{bias.min().item():.6f}, {bias.max().item():.6f}]")
        print("[DEBUG] =================================")
    
    def _materialize_meta_tensors(self):
        """Force materialization of meta tensors"""
        print("[DEBUG] Materializing meta tensors...")
        # Move to CPU first to force materialization
        self.to('cpu')
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Force materialization by creating a copy
                module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    module.bias.data = module.bias.data.clone()
                print(f"[DEBUG] Materialized Conv2d layer: weight range [{module.weight.min().item():.6f}, {module.weight.max().item():.6f}]")
            elif isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    module.bias.data = module.bias.data.clone()
                print(f"[DEBUG] Materialized Linear layer: weight range [{module.weight.min().item():.6f}, {module.weight.max().item():.6f}]")
        print("[DEBUG] Meta tensors materialized.")
    
    def _init_weights(self):
        """Initialize weights properly to avoid NaN values"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Keep your safe initialization - 0.02 that works
                with torch.no_grad():
                    module.weight.data = torch.randn_like(module.weight.data) * 0.02
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight.data = torch.randn_like(module.weight.data) * 0.02
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                # Safe batch norm initialization
                nn.init.ones_(module.weight)
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
        print(f"[DEBUG] Final output: shape={emb.shape}, range=[{emb.min().item():.6f}, {emb.max().item():.6f}]")

        return emb

###END OF FILE###