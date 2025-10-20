import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from gr00t.model.action_head.obs_encoder import ObsEncoder


class ActionUpdater(nn.Module):
    def __init__(self, action_dim: int, obs_emb_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.obs_emb_dim = obs_emb_dim
        
        # MLP for updating actions based on observation feedback
        self.action_updater = nn.Sequential(
            nn.Linear(obs_emb_dim + action_dim, 128),  # obs + action input
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # Bound output to [-1, 1]
        )
        
        # Initialize with small weights to prevent gradient explosion
        for module in self.action_updater.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.001)
                nn.init.zeros_(module.bias)
                # Additional safety: clamp weights to prevent explosion
                with torch.no_grad():
                    module.weight.clamp_(-0.1, 0.1)

    def forward(self, action_chunk, obs_features):
        """
        Args:
            action_chunk: [B, 4, action_dim] - 4 actions to update
            obs_features: [B, obs_emb_dim] - observation features for feedback
        Returns:
            action_update: [B, 4, action_dim] - updates to apply to actions
        """
        B = action_chunk.shape[0]
        
        # Reshape actions to [B*4, action_dim] and expand obs features to [B*4, obs_emb_dim]
        action_flat = action_chunk.contiguous().view(-1, action_chunk.shape[-1])  # [B*4, action_dim]
        obs_expanded = obs_features.unsqueeze(1).expand(-1, 4, -1).contiguous().view(-1, obs_features.shape[-1])  # [B*4, obs_emb_dim]
        
        # Concatenate action and obs features
        action_obs = torch.cat([action_flat, obs_expanded], dim=-1)  # [B*4, action_dim + obs_emb_dim]
        
        # Clamp inputs to prevent extreme values
        action_obs = torch.clamp(action_obs, min=-10.0, max=10.0)
        
        # Apply action updater MLP with stability checks
        try:
            with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision for this MLP
                x = action_obs.float()
                
                # Check if input has nan/inf
                if torch.isnan(x).any() or torch.isinf(x).any():
                    x = torch.zeros_like(x)
                
                # Apply action updater MLP
                x = self.action_updater(x)
                
                # Check for nan/inf in output
                if torch.isnan(x).any() or torch.isinf(x).any():
                    x = torch.zeros_like(x)
                
                action_update = x.view(B, 4, -1)  # [B, 4, action_dim]
                
        except Exception:
            action_update = torch.zeros(B, 4, action_chunk.shape[-1], device=action_chunk.device)
        
        # Check for nan/inf values
        if torch.isnan(action_update).any() or torch.isinf(action_update).any():
            action_update = torch.zeros_like(action_update)
        
        # Scale down the feedback to prevent instability
        feedback_scale = 0.1  # Small scaling factor
        action_update = action_update * feedback_scale
        
        return action_update


class FeedbackAction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        
        # Obs encoder for processing observation images
        self.obs_encoder_alone = ObsEncoder(emb_dim=256)
        
        # Action updater for applying feedback
        self.action_updater = ActionUpdater(
            action_dim=self.action_dim,
            obs_emb_dim=256
        )

    def forward(self, action_head_output: BatchFeature, window_idx: int, action_input: BatchFeature) -> BatchFeature:

        # ground truth action
        velocity = action_head_output.gt_actions
        # predicted raw action
        pred_actions = action_head_output.pred_actions

        # compute loss
        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    def get_action(self, action_head_output: BatchFeature, window_idx: int, action_input: BatchFeature) -> BatchFeature:
        pred_actions = action_head_output.pred_actions
        return BatchFeature(data={"action_pred": pred_actions})