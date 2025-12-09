import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from gr00t.model.action_head.obs_encoder import ObsEncoder
from gr00t.model.action_head.flow_matching_action_head import CategorySpecificMLP


class ObserverMLP(nn.Module):
    """
    mlp(obs_encoder(obs), x)
    Combines observation features with model output action features.
    """

    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        # Two layer MLP: map from concatenated features to output through hidden layer
        self.layer = CategorySpecificMLP(num_categories, 2 * input_dim, hidden_dim, output_dim)
        self.obs_encoder = ObsEncoder(emb_dim=input_dim)

    def forward(self, model_output_action, obs, cat_ids, window_idx):
        """
        Args:
            x: model_output_action of shape (B, action_horizon, hidden_size)
            obs: observation image of shape (B, T, V, H, W, C) or similar
            cat_ids: embodiment_id of shape (B,)
        Returns:
            output of shape (B, action_horizon, action_dim)
        """
        # Encode observation: (B, 1, hidden_size)
        obs_feature = self.obs_encoder(obs)  # (B, 1, hidden_size)

        # Expand obs_feature to match x's sequence length: (B, action_horizon, hidden_size)
        B, action_horizon, hidden_size = model_output_action.shape
        obs_feature = obs_feature.expand(B, action_horizon, hidden_size)

        # Combine obs_feature with model_output_action (concatenate them)
        combined = torch.cat([obs_feature, model_output_action], dim=-1)  # (B, action_horizon, 2*hidden_size)

        # Two layer MLP: map to output through hidden layer
        output = self.layer(combined, cat_ids)  # (B, action_horizon, action_dim)

        # Bound output to [-5, 5] range using tanh (smooth, differentiable)
        # This preserves gradient flow unlike hard clamping
        output = torch.tanh(output) * 5.0

        if window_idx == 0:
            print(f"window_idx: {window_idx}")
            print(
                f"obs_feature: {obs_feature.shape}, range {obs_feature.min().item()}, {obs_feature.max().item()}"
            )
            print(
                f"model_output_action: {model_output_action.shape}, range {model_output_action.min().item()}, {model_output_action.max().item()}"
            )
            print(f"feedback output range: {output.shape}, range {output.min().item()}, {output.max().item()}")
        return output


class FeedbackAction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.hidden_size = config.hidden_size

        self.action_decoder_observe = ObserverMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

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
                self.action_decoder_observe.eval()

    def forward(
        self, action_head_output: BatchFeature, time_step: int, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Input: action_head_output: BatchFeature, time_step: int, action_input: BatchFeature
        Output: updated_actions: [B, 16, action_dim]
        """
        gt_actions = action_input.action  # ground truth action
        model_output_action = action_head_output.final_model
        final_raw_action = action_head_output.final_raw_action
        dt = action_head_output.dt
        observation = action_input.simple_img
        embodiment_id = action_input.embodiment_id

        print(
            f"observation shape: {observation.shape}, range {observation.min().item()}, {observation.max().item()}"
        )

        pred_actions_frame_list = []
        # observation shape: [B, V, T, H, W, C] = [16, 1, 4, 224, 224, 3]
        num_frames = observation.shape[2]  # T = 4
        for window_idx in range(num_frames):  # for 4 frames
            # Slice observation: [B, V, 1, H, W, C] = [16, 1, 1, 224, 224, 3]
            observation_frame = observation[:, :, window_idx : window_idx + 1, :, :, :]
            # Permute to match ObsEncoder expected format: [B, T, V, H, W, C]
            observation_frame = observation_frame.permute(0, 2, 1, 3, 4, 5)  # [B, 1, V, H, W, C]

            # Slice model output action: [B, 4, hidden_size] = [16, 4, 1024]
            model_output_action_frame = model_output_action[:, window_idx * 4 : (window_idx + 1) * 4, :]
            # Slice final raw action: [B, 4, action_dim] = [16, 4, 32]
            final_raw_action_frame = final_raw_action[:, window_idx * 4 : (window_idx + 1) * 4, :]

            # Process with ObserverMLP: [B, 4, 32]
            pred_velocity_frame = self.action_decoder_observe(
                model_output_action_frame, observation_frame, embodiment_id, window_idx
            )

            pred_action_frame = final_raw_action_frame + dt * pred_velocity_frame
            pred_actions_frame_list.append(pred_action_frame)

        pred_actions = torch.cat(pred_actions_frame_list, dim=1)  # [B, 16, 32]
        pred_actions_normalized = torch.tanh(pred_actions)  # normalize to match gt_actions

        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions_normalized, gt_actions, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    def get_action(
        self,
        action_head_output: BatchFeature,
        time_step: int,
        action_input: BatchFeature,
    ) -> BatchFeature:
        """
        Input: action_head_output with model_output, time_step, action_input
        Output: [B, 4, action_dim] actions for the current window
        """
        window_idx = time_step % 4
        embodiment_id = action_input.embodiment_id
        model_output_action = action_head_output.final_model
        final_raw_action = action_head_output.final_raw_action
        dt = action_head_output.dt
        observation = action_input.simple_img

        observation_frame = observation
        # Permute to match ObsEncoder expected format: [B, T, V, H, W, C]
        observation_frame = observation_frame.permute(0, 2, 1, 3, 4, 5)  # [B, 1, V, H, W, C]

        # Slice model output action: [B, 4, hidden_size] = [16, 4, 1024]
        model_output_action_frame = model_output_action[:, window_idx * 4 : (window_idx + 1) * 4, :]
        # Slice final raw action: [B, 4, action_dim] = [16, 4, 32]
        final_raw_action_frame = final_raw_action[:, window_idx * 4 : (window_idx + 1) * 4, :]

        # Process with ObserverMLP: [B, 4, 32]
        pred_velocity_frame = self.action_decoder_observe(
            model_output_action_frame, observation_frame, embodiment_id, window_idx
        )

        pred_action_frame = final_raw_action_frame + dt * pred_velocity_frame  # [B, 4, 32]

        return BatchFeature(data=pred_action_frame)
