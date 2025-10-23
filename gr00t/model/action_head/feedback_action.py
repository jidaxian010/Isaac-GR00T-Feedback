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

        self.obs_encoder = ObsEncoder(emb_dim=obs_emb_dim)
        self.action_updater = nn.Sequential(
            nn.Linear(obs_emb_dim + action_dim, 128),  # obs + action input: 256 + 32 = 288
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # Bound output to [-1, 1]
        )

    def forward(self, pred_action_chunk: torch.Tensor, obs_frame: torch.Tensor, window_idx: int) -> torch.Tensor:
        """
        Input: pred_action_chunk: [B, 4, action_dim] (4 actions), obs_frame: [B, V, 1, H, W, C] (1 frame)
        Output: action_update: [B, 4, action_dim] (4 actions)
        """

        # self.obs_encoder.eval()
        obs_features = self.obs_encoder(obs_frame)  # [B, 1, 256]
        # self.obs_encoder.train()  # Set back to train mode

        # Clamp obs_features to prevent extreme values
        obs_features = torch.clamp(obs_features, min=-10.0, max=10.0)

        obs_features = obs_features.expand(-1, 4, -1).contiguous()  # Expanded to [B, 4, 256]
        action_obs_concat = torch.cat([pred_action_chunk, obs_features], dim=-1)  # [B, 4, 288]
        B = action_obs_concat.shape[0]
        action_obs_flat = action_obs_concat.view(-1, action_obs_concat.shape[-1])  # [B*4, 288]
        delta_action_flat = self.action_updater(action_obs_flat)  # [B*4, action_dim]
        delta_action = delta_action_flat.view(B, 4, -1)  # [B, 4, action_dim]
        print(f"window_idx: {window_idx}")
        print(
            f"pred_action_chunk shape: {pred_action_chunk.shape}, {pred_action_chunk.min().item()}, {pred_action_chunk.max().item()}"
        )
        print(
            f"delta_action shape: {delta_action.shape}, {delta_action.min().item()}, {delta_action.max().item()}"
        )

        action_update = pred_action_chunk + 0.5 * delta_action  # [B, 4, action_dim]

        return action_update  # [B, 4, action_dim]


class FeedbackAction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon

        # Action updater for applying feedback
        self.action_updater = ActionUpdater(action_dim=self.action_dim, obs_emb_dim=256)

    def forward(
        self, action_head_output: BatchFeature, time_step: int, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Input: action_head_output: BatchFeature, time_step: int, action_input: BatchFeature
        Output: updated_actions: [B, 16, action_dim]
        """
        # ground truth action
        velocity = action_head_output.gt_actions
        # predicted raw action
        pred_actions = action_head_output.pred_actions

        # update action
        action_updates = []  # Collect all action updates
        for window_idx in range(0, 4):  # window_idx: 0, 1, 2, 3
            
            obs_frame = action_input.simple_img[
                :, :, window_idx : window_idx + 1, :, :, :
            ]  # sliced obs frame: [B, V, 1, H, W, C]
            pred_action_chunk = pred_actions[:, window_idx * 4 : (window_idx + 1) * 4, :]  # sliced action: [B, 4, action_dim]
            action_update = self.action_updater(pred_action_chunk, obs_frame, window_idx)

            # # test
            # obs_features = action_head_output.obs_features
            # action_update = self.action_updater(pred_action_chunk, obs_features, window_idx)

            action_updates.append(action_update)  # Collect update: [B, 4, action_dim]

        updated_actions = torch.cat(action_updates, dim=1)  # [B, 16, action_dim]

        # compute loss
        action_mask = action_input.action_mask
        loss = F.mse_loss(updated_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    def get_action( 
        self, action_head_output: BatchFeature, time_step: int, action_input: BatchFeature
    ) -> BatchFeature:
        window_idx = time_step %4
        print(f"[FEEDBACK ACTION] at timestep {time_step}, window_idx: {window_idx}")
        obs_frame = action_input.simple_img # sliced obs frame: [B, V, 1, H, W, C], only one fresh frame
        pred_actions = action_head_output.action_pred  # [B, 16, action_dim] - use action_pred during inference
        pred_action_chunk = pred_actions[:, window_idx * 4 : (window_idx + 1) * 4, :]
        action_update = self.action_updater(pred_action_chunk, obs_frame, window_idx)
        return BatchFeature(data={"action_pred": action_update})
