import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from gr00t.model.action_head.obs_encoder import ObsEncoder


class ActionUpdater(nn.Module):
    def __init__(self, hidden_size: int, obs_emb_dim: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.obs_emb_dim = obs_emb_dim

        self.obs_encoder = ObsEncoder(emb_dim=obs_emb_dim)
        # Update latent features instead of decoded actions
        # Use smaller network with proper initialization
        self.latent_updater = nn.Sequential(
            nn.Linear(obs_emb_dim + hidden_size, 512),  # obs + latent: 256 + 1024 = 1280
            nn.LayerNorm(512),  # Add layer norm for stability
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # Add layer norm for stability
            nn.ReLU(),
            nn.Linear(256, hidden_size),  # Output: hidden_size
        )

        # Initialize weights carefully to prevent NaN
        for module in self.latent_updater.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.01)  # Very small gain
                nn.init.zeros_(module.bias)

    def forward(self, latent_features: torch.Tensor, obs_frame: torch.Tensor, window_idx: int) -> torch.Tensor:
        """
        Input:
            latent_features: [B, 4, hidden_size] (4 latent action features)
            obs_frame: [B, V, 1, H, W, C] (1 frame)
        Output:
            updated_latent: [B, 4, hidden_size]
        """
        obs_features = self.obs_encoder(obs_frame)  # [B, 1, 256]

        # Normalize both tensors before concatenation for stable training
        obs_features_norm = obs_features / (
            obs_features.norm(dim=-1, keepdim=True) + 1e-8
        )  # L2 normalize obs_features
        latent_features_norm = latent_features / (
            latent_features.norm(dim=-1, keepdim=True) + 1e-8
        )  # L2 normalize latent_features

        obs_features_norm = obs_features_norm.expand(-1, 4, -1).contiguous()  # Expanded to [B, 4, 256]
        latent_obs_concat = torch.cat([latent_features_norm, obs_features_norm], dim=-1)  # [B, 4, hidden_size+256]
        B = latent_obs_concat.shape[0]
        latent_obs_flat = latent_obs_concat.view(-1, latent_obs_concat.shape[-1])  # [B*4, hidden_size+256]
        delta_latent_flat = self.latent_updater(latent_obs_flat)  # [B*4, hidden_size]
        delta_latent = delta_latent_flat.view(B, 4, -1)  # [B, 4, hidden_size]

        # Clamp delta_latent to reasonable range
        delta_latent = torch.clamp(delta_latent, min=-1.0, max=1.0)

        print(f"window_idx: {window_idx}")
        print(f"obs_frame shape: {obs_frame.shape}, range: {obs_frame.min().item()}, {obs_frame.max().item()}")
        print(
            f"obs_features_norm shape: {obs_features_norm.shape}, range: {obs_features_norm.min().item()}, {obs_features_norm.max().item()}"
        )
        print(
            f"latent_features_norm shape: {latent_features_norm.shape}, range: {latent_features_norm.min().item()}, {latent_features_norm.max().item()}"
        )
        print(
            f"delta_latent shape: {delta_latent.shape}, range: {delta_latent.min().item()}, {delta_latent.max().item()}"
        )

        updated_latent = latent_features + 0.2 * delta_latent  # [B, 4, hidden_size] - update latent features

        print(
            f"updated_latent shape: {updated_latent.shape}, range: {updated_latent.min().item()}, {updated_latent.max().item()}"
        )
        return updated_latent  # [B, 4, hidden_size]


class FeedbackAction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.hidden_size = config.hidden_size

        # Don't store action_decoder reference here - will be accessed from parent model
        # This avoids duplicate parameter issues when saving checkpoints

        # Action updater for latent features (already initialized in ActionUpdater.__init__)
        self.action_updater = ActionUpdater(hidden_size=self.hidden_size, obs_emb_dim=256)

        # Track whether feedback_action is trainable
        self.tune_feedback = True

    def set_trainable(self, trainable: bool, action_decoder=None):
        """Set whether this module is trainable.

        Args:
            trainable: Whether to make this module trainable
            action_decoder: Reference to shared action_decoder (needed to set its trainability)
        """
        self.tune_feedback = trainable
        # Freeze/unfreeze action_updater parameters
        self.action_updater.requires_grad_(trainable)

        # IMPORTANT: Ensure action_decoder is trainable when FeedbackAction is trainable
        # action_decoder is shared with action_head, so we need to explicitly set it
        if trainable and action_decoder is not None:
            action_decoder.requires_grad_(True)
            print(f"FeedbackAction trainable: {trainable} (action_decoder also enabled)")
        else:
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
        self, action_head_output: BatchFeature, time_step: int, action_input: BatchFeature, action_decoder=None
    ) -> BatchFeature:
        """
        Input: action_head_output: BatchFeature containing model_output (latent) and gt_actions
        Output: loss after updating latent features and decoding
        """
        velocity = action_head_output.gt_actions  # ground truth action
        model_output = action_head_output.model_output  # [B, T, hidden_size] - latent features
        embodiment_id = action_input.embodiment_id

        # Extract action-related latent features (last 16 tokens)
        action_latents = model_output[:, -self.action_horizon :, :]  # [B, 16, hidden_size]

        latent_updates = []  # Collect all latent updates
        for window_idx in range(0, 4):  # window_idx: 0, 1, 2, 3
            # prepare obs_frame
            obs_frame = action_input.simple_img[
                :, :, window_idx : window_idx + 1, :, :, :
            ]  # sliced obs frame: [B, V, 1, H, W, C]
            # prepare latent_chunk
            latent_chunk = action_latents[
                :, window_idx * 4 : (window_idx + 1) * 4, :
            ]  # sliced latent: [B, 4, hidden_size]

            updated_latent = self.action_updater(latent_chunk, obs_frame, window_idx)

            latent_updates.append(updated_latent)  # Collect update: [B, 4, hidden_size]

        updated_latents = torch.cat(latent_updates, dim=1)  # [B, 16, hidden_size]

        # Decode the updated latent features to get actions
        if action_decoder is None:
            raise ValueError("action_decoder must be provided to FeedbackAction.forward()")
        updated_actions = action_decoder(updated_latents, embodiment_id)  # [B, 16, action_dim]

        # compute loss
        action_mask = action_input.action_mask
        loss = F.mse_loss(updated_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    def get_action(
        self, action_head_output: BatchFeature, time_step: int, action_input: BatchFeature, action_decoder=None
    ) -> BatchFeature:
        window_idx = time_step % 4
        # prepare obs_frame
        obs_frame = action_input.simple_img  # obs frame: [B, V, 1, H, W, C], only one fresh frame at a time
        # prepare model_output (latent features)
        model_output = action_head_output.model_output  # [B, T, hidden_size]
        embodiment_id = action_input.embodiment_id

        # Extract action-related latent features (last 16 tokens)
        action_latents = model_output[:, -self.action_horizon :, :]  # [B, 16, hidden_size]
        latent_chunk = action_latents[:, window_idx * 4 : (window_idx + 1) * 4, :]  # [B, 4, hidden_size]

        # Update latent features
        updated_latent = self.action_updater(latent_chunk, obs_frame, window_idx)

        # Decode to get actions
        if action_decoder is None:
            raise ValueError("action_decoder must be provided to FeedbackAction.get_action()")
        action_update = action_decoder(updated_latent, embodiment_id)  # [B, 4, action_dim]

        return BatchFeature(data={"action_pred": action_update})
