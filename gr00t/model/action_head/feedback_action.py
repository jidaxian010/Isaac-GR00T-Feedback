import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from gr00t.model.action_head.obs_encoder import ObsEncoder
from gr00t.model.action_head.flow_matching_action_head import CategorySpecificMLP


class ActionUpdater(nn.Module):
    def __init__(self, hidden_size: int, num_embodiments: int, action_dim: int, obs_emb_dim: int = 256):
        super().__init__()
        self.hidden_size = hidden_size  # 1024 - latent embedding dimension
        self.action_dim = action_dim  # 32 - final action dimension
        self.obs_emb_dim = obs_emb_dim  # 256 - observation embedding dimension
        self.num_embodiments = num_embodiments

        self.obs_encoder = ObsEncoder(emb_dim=obs_emb_dim)

        # Update latent features: input is obs (256) + latent (1024) = 1280
        self.latent_updater = nn.Sequential(
            nn.Linear(obs_emb_dim + hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, hidden_size),  # Output: 1024 (same as latent dim)
        )

        # Decoder: latent (1024) → action (32)
        self.action_decoder = CategorySpecificMLP(
            num_categories=num_embodiments,
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=action_dim,
        )

        # Initialize weights carefully to prevent NaN
        for module in self.latent_updater.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        latent_chunk: torch.Tensor,
        obs_frame: torch.Tensor,
        window_idx: int,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input:
            latent_chunk: [B, 4, hidden_size=1024] (4 latent action features)
            obs_frame: [B, V, 1, H, W, C] (1 frame)
            embodiment_id: [B] (embodiment IDs)
        Output:
            action_update: [B, 4, action_dim=32] (4 real actions)
        """
        obs_features = self.obs_encoder(obs_frame)  # [B, 1, 256]

        # Normalize both tensors before concatenation for stable training
        obs_features_norm = obs_features / (obs_features.norm(dim=-1, keepdim=True) + 1e-8)
        latent_chunk_norm = latent_chunk / (latent_chunk.norm(dim=-1, keepdim=True) + 1e-8)

        obs_features_norm = obs_features_norm.expand(-1, 4, -1).contiguous()  # [B, 4, 256]
        latent_obs_concat = torch.cat([latent_chunk_norm, obs_features_norm], dim=-1)  # [B, 4, 1280]

        B = latent_obs_concat.shape[0]
        latent_obs_flat = latent_obs_concat.view(-1, latent_obs_concat.shape[-1])  # [B*4, 1280]
        delta_latent_flat = self.latent_updater(latent_obs_flat)  # [B*4, 1024]
        delta_latent = delta_latent_flat.view(B, 4, -1)  # [B, 4, 1024]

        # Clamp delta to reasonable range
        delta_latent = torch.clamp(delta_latent, min=-1.0, max=1.0)

        # Update latent features
        updated_latent = latent_chunk + 0.2 * delta_latent  # [B, 4, 1024]

        # Decode to real actions
        action_update = self.action_decoder(updated_latent, embodiment_id)  # [B, 4, 32]

        if window_idx == 0:
            print(f"obs_features: {obs_features.shape}, range: {obs_features.min()}, {obs_features.max()}")
            print(
                f"obs_features_norm: {obs_features_norm.shape}, range: {obs_features_norm.min()}, {obs_features_norm.max()}"
            )
            print(f"latent_chunk: {latent_chunk.shape}, range: {latent_chunk.min()}, {latent_chunk.max()}")
            print(
                f"latent_chunk_norm: {latent_chunk_norm.shape}, range: {latent_chunk_norm.min()}, {latent_chunk_norm.max()}"
            )
            print(f"delta_latent: {delta_latent.shape}, range: {delta_latent.min()}, {delta_latent.max()}")
            print(f"updated_latent: {updated_latent.shape}, range: {updated_latent.min()}, {updated_latent.max()}")
            print(f"action_update: {action_update.shape}, range: {action_update.min()}, {action_update.max()}")

        return action_update  # [B, 4, action_dim]


class FeedbackAction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.hidden_size = config.hidden_size

        # Action updater for applying feedback (includes its own decoder)
        self.action_updater = ActionUpdater(
            hidden_size=self.hidden_size,
            num_embodiments=config.max_num_embodiments,
            action_dim=self.action_dim,
            obs_emb_dim=256,
        )

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
        embodiment_id = action_input.embodiment_id
        velocity = action_head_output.gt_actions  # ground truth action [B, 16, action_dim]
        model_output = action_head_output.model_output  # [B, 49, hidden_size=1024]

        # Slice out only the action latents (last 16 tokens)
        latent_actions = model_output[:, -self.action_horizon :, :]  # [B, 16, 1024]

        action_updates = []  # Collect all action updates
        for window_idx in range(0, 4):  # window_idx: 0, 1, 2, 3
            # Prepare obs_frame
            obs_frame = action_input.simple_img[:, :, window_idx : window_idx + 1, :, :, :]  # [B, V, 1, H, W, C]

            # Prepare latent_chunk (4 latent actions for this window)
            latent_chunk = latent_actions[:, window_idx * 4 : (window_idx + 1) * 4, :]  # [B, 4, 1024]

            # Update latent and decode to actions
            action_update = self.action_updater(
                latent_chunk, obs_frame, window_idx, embodiment_id
            )  # [B, 4, action_dim=32]

            action_updates.append(action_update)

        # Concatenate 4 windows: 4 × [B, 4, 32] → [B, 16, 32]
        updated_actions = torch.cat(action_updates, dim=1)  # [B, 16, action_dim]

        # Compute loss
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
        """
        Input: action_head_output with model_output, time_step, action_input
        Output: [B, 4, action_dim] actions for the current window
        """
        window_idx = time_step % 4
        embodiment_id = action_input.embodiment_id

        # Get model output (latent features)
        model_output = action_head_output.model_output  # [B, 49, hidden_size=1024]

        # Slice out only the action latents (last 16 tokens)
        latent_actions = model_output[:, -self.action_horizon :, :]  # [B, 16, 1024]

        # Get the latent chunk for this specific window
        latent_chunk = latent_actions[:, window_idx * 4 : (window_idx + 1) * 4, :]  # [B, 4, 1024]

        # Prepare obs_frame (single fresh observation)
        obs_frame = action_input.simple_img  # [B, V, 1, H, W, C]

        # Update latent and decode to actions
        action_update = self.action_updater(
            latent_chunk, obs_frame, window_idx, embodiment_id
        )  # [B, 4, action_dim=32]

        return BatchFeature(data={"action_pred": action_update})
