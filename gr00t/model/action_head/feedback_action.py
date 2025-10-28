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
        )

    def forward(self, pred_action_chunk: torch.Tensor, obs_frame: torch.Tensor, window_idx: int) -> torch.Tensor:
        """
        Input: pred_action_chunk: [B, 4, action_dim] (4 actions), obs_frame: [B, V, 1, H, W, C] (1 frame)
        Output: action_update: [B, 4, action_dim] (4 actions)
        """

        obs_features = self.obs_encoder(obs_frame)  # [B, 1, 256] - now bounded to [-1, 1]

        # Normalize both tensors before concatenation for stable training
        obs_features_norm = obs_features / (
            obs_features.norm(dim=-1, keepdim=True) + 1e-8
        )  # L2 normalize obs_features
        pred_action_chunk_norm = pred_action_chunk / (
            pred_action_chunk.norm(dim=-1, keepdim=True) + 1e-8
        )  # L2 normalize pred_action_chunk

        obs_features_norm = obs_features_norm.expand(-1, 4, -1).contiguous()  # Expanded to [B, 4, 256]
        action_obs_concat = torch.cat([pred_action_chunk_norm, obs_features_norm], dim=-1)  # [B, 4, 288]
        B = action_obs_concat.shape[0]
        action_obs_flat = action_obs_concat.view(-1, action_obs_concat.shape[-1])  # [B*4, 288]
        delta_action_flat = self.action_updater(action_obs_flat)  # [B*4, action_dim]
        delta_action = delta_action_flat.view(B, 4, -1)  # [B, 4, action_dim]

        # Clamp delta_action to reasonable range
        delta_action = torch.clamp(delta_action, min=-1.0, max=1.0)
        print(f"window_idx: {window_idx}")
        print(f"obs_frame shape: {obs_frame.shape}, range: {obs_frame.min().item()}, {obs_frame.max().item()}")
        print(
            f"obs_features_norm shape: {obs_features_norm.shape}, range: {obs_features_norm.min().item()}, {obs_features_norm.max().item()}"
        )
        print(
            f"pred_action_chunk_norm shape: {pred_action_chunk_norm.shape}, range: {pred_action_chunk_norm.min().item()}, {pred_action_chunk_norm.max().item()}"
        )
        print(
            f"delta_action shape: {delta_action.shape}, range: {delta_action.min().item()}, {delta_action.max().item()}"
        )

        action_update = (
            pred_action_chunk + 0.2 * delta_action
        )  # [B, 4, action_dim] - clamped delta_action allows larger scale

        print(
            f"action_update shape: {action_update.shape}, range: {action_update.min().item()}, {action_update.max().item()}"
        )
        return action_update  # [B, 4, action_dim]


class FeedbackAction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon

        # Action updater for applying feedback
        self.action_updater = ActionUpdater(action_dim=self.action_dim, obs_emb_dim=256)

        # Initialize action_updater weights more conservatively
        for module in self.action_updater.action_updater.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.01)  # Very small gain
                nn.init.zeros_(module.bias)

        # Track whether feedback_action is trainable
        self.tune_feedback = True

    def set_trainable(self, trainable: bool):
        """Set whether this module is trainable."""
        self.tune_feedback = trainable
        # Freeze/unfreeze all parameters
        self.requires_grad_(trainable)
        print(f"FeedbackAction trainable: {trainable}")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_feedback:
                self.action_updater.eval()

    def forward(
        self, action_head_output: BatchFeature, time_step: int, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Input: action_head_output: BatchFeature, time_step: int, action_input: BatchFeature
        Output: updated_actions: [B, 16, action_dim]
        """
        velocity = action_head_output.gt_actions  # ground truth action
        pred_actions = action_head_output.pred_actions

        action_updates = []  # Collect all action updates
        for window_idx in range(0, 4):  # window_idx: 0, 1, 2, 3
            # prepare obs_frame
            obs_frame = action_input.simple_img[
                :, :, window_idx : window_idx + 1, :, :, :
            ]  # sliced obs frame: [B, V, 1, H, W, C]
            # prepare pred_action_chunk
            pred_action_chunk = pred_actions[
                :, window_idx * 4 : (window_idx + 1) * 4, :
            ]  # sliced action: [B, 4, action_dim]

            action_update = self.action_updater(pred_action_chunk, obs_frame, window_idx)

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
        window_idx = time_step % 4
        # prepare obs_frame
        obs_frame = action_input.simple_img  # obs frame: [B, V, 1, H, W, C], only one fresh frame at a time
        # prepare pred_actions
        pred_actions = action_head_output.action_pred  # [B, 16, action_dim] - use action_pred during inference
        pred_action_chunk = pred_actions[:, window_idx * 4 : (window_idx + 1) * 4, :]

        action_update = self.action_updater(pred_action_chunk, obs_frame, window_idx)
        return BatchFeature(data={"action_pred": action_update})
