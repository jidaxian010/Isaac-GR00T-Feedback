import torch
import torch.nn as nn
import torchvision.models as models


# Fast image perception
class ObsEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, emb_dim)

    def forward(self, x):
        # x: (B, T, V, 224, 224, 3)
        # Extract the last frame (T-1) for each batch and view
        x = x[:, -1]  # (B, V, 224, 224, 3)
        B, V, H, W, C = x.shape
        x = x.view(B * V, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B*V, 3, 224, 224)
        features = self.cnn(x)      # (B*V, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B*V, 512)
        features = self.fc(features)  # (B*V, emb_dim)
        # Pool across views to get (B, emb_dim) if needed
        features = features.view(B, V, -1)  # (B, V, emb_dim)
        features = features.mean(dim=1)     # (B, emb_dim)
        return features
        
        

