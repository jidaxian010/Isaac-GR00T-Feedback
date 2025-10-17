# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from gr00t.model.action_head.obs_encoder import ObsEncoder

from .cross_attention_dit import DiT, SelfAttentionTransformer


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories

        # Use Xavier/Glorot initialization for better gradient flow
        init_scale = (2.0 / (input_dim + hidden_dim)) ** 0.5
        self.W = nn.Parameter(init_scale * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]

        result = torch.bmm(x, selected_W) + selected_b.unsqueeze(1)

        return result


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})
    backbone_embedding_dim: int = field(default=1536, metadata={"help": "Backbone embedding channel dimension."})

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(default=True, metadata={"help": "Whether to tune the diffusion model."})
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )

        # Obs encoder for processing observation images
        self.obs_encoder_alone = ObsEncoder(emb_dim=256)  # Much smaller output dimension
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        # Action updater MLP for feedback mechanism - much more stable with smaller input
        self.action_updater = nn.Sequential(
            nn.Linear(256 + config.action_dim, 128),  # 256 (obs) + 32 (action) = 288 input
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config.action_dim),
            nn.Tanh(),  # Bound output to [-1, 1]
        )
        # Initialize with extremely small weights to prevent gradient explosion
        for module in self.action_updater.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.001)  # Even smaller gain
                nn.init.zeros_(module.bias)
                # Additional safety: clamp weights to prevent explosion
                with torch.no_grad():
                    module.weight.clamp_(-0.1, 0.1)
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)

        # Always ensure the obs_encoder is trainable
        self.obs_encoder_alone.requires_grad_(True)
        self.action_updater.requires_grad_(True)

        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print("Obs encoder is always trainable")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)

        # Get the minimum batch size among all tensors to be concatenated
        min_len = min(
            state_features.shape[0],
            future_tokens.shape[0],
            action_features.shape[0],
        )

        # Slice all tensors to the minimum batch size
        state_features = state_features[:min_len]
        future_tokens = future_tokens[:min_len]
        action_features = action_features[:min_len]

        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # TEMPORARY: Disable feedback mechanism to test stability
        use_feedback = True  # Set to False to disable feedback, True to enable

        if not use_feedback:
            # Skip feedback mechanism - use original predictions
            pass
        else:
            # Slice pred_actions from 16 action chunks into 4 chunks, 4 action each
            action_1 = pred_actions[:, 0:4, :]  # Actions 0-3: no feedback
            action_2 = pred_actions[:, 4:8, :]  # Actions 4-7: feedback from frame 0
            action_3 = pred_actions[:, 8:12, :]  # Actions 8-11: feedback from frame 1
            action_4 = pred_actions[:, 12:16, :]  # Actions 12-15: feedback from frame 2

            # Get observation frames - simple_img shape is [B, V, T, H, W, C]
            # We need to create proper input for obs_encoder_alone which expects [B, T, V, H, W, C]

            # Extract individual frames and reshape for obs_encoder_alone
            # simple_img is [B, V, T, H, W, C], we want [B, T, V, H, W, C] for obs_encoder_alone
            obs_frame_0 = action_input.simple_img[:, :, 0:1, :, :, :]  # [B, V, 1, H, W, C] - first frame
            obs_frame_1 = action_input.simple_img[:, :, 1:2, :, :, :]  # [B, V, 1, H, W, C] - second frame
            obs_frame_2 = action_input.simple_img[:, :, 2:3, :, :, :]  # [B, V, 1, H, W, C] - third frame

            # Permute to match obs_encoder_alone expected input: [B, T, V, H, W, C]
            obs_frame_0 = obs_frame_0.permute(0, 2, 1, 3, 4, 5)  # [B, 1, V, H, W, C]
            obs_frame_1 = obs_frame_1.permute(0, 2, 1, 3, 4, 5)  # [B, 1, V, H, W, C]
            obs_frame_2 = obs_frame_2.permute(0, 2, 1, 3, 4, 5)  # [B, 1, V, H, W, C]

            # Encode observation frames
            obs_features_0 = self.obs_encoder_alone(obs_frame_0).squeeze(1)  # [B, 1, emb_dim] -> [B, emb_dim]
            obs_features_1 = self.obs_encoder_alone(obs_frame_1).squeeze(1)  # [B, 1, emb_dim] -> [B, emb_dim]
            obs_features_2 = self.obs_encoder_alone(obs_frame_2).squeeze(1)  # [B, 1, emb_dim] -> [B, emb_dim]

            print(f"obs_features_0_range: {obs_features_0.min():.6f}, {obs_features_0.max():.6f}")

            # Update actions with feedback
            updated_action_1 = action_1

            # For action chunks 2, 3, 4: apply feedback mechanism
            # Reshape actions to [B*4, action_dim] and obs_features to [B*4, emb_dim]
            B = action_2.shape[0]
            action_2_flat = action_2.contiguous().view(-1, action_2.shape[-1])  # [B*4, action_dim]
            action_3_flat = action_3.contiguous().view(-1, action_3.shape[-1])  # [B*4, action_dim]
            action_4_flat = action_4.contiguous().view(-1, action_4.shape[-1])  # [B*4, action_dim]

            obs_features_0_expanded = (
                obs_features_0.unsqueeze(1).expand(-1, 4, -1).contiguous().view(-1, obs_features_0.shape[-1])
            )  # [B*4, emb_dim]
            obs_features_1_expanded = (
                obs_features_1.unsqueeze(1).expand(-1, 4, -1).contiguous().view(-1, obs_features_1.shape[-1])
            )  # [B*4, emb_dim]
            obs_features_2_expanded = (
                obs_features_2.unsqueeze(1).expand(-1, 4, -1).contiguous().view(-1, obs_features_2.shape[-1])
            )  # [B*4, emb_dim]

            # Concatenate action and obs features for MLP input
            action_obs_2 = torch.cat(
                [action_2_flat, obs_features_0_expanded], dim=-1
            )  # [B*4, action_dim + emb_dim]
            action_obs_3 = torch.cat(
                [action_3_flat, obs_features_1_expanded], dim=-1
            )  # [B*4, action_dim + emb_dim]
            action_obs_4 = torch.cat(
                [action_4_flat, obs_features_2_expanded], dim=-1
            )  # [B*4, action_dim + emb_dim]

            # Clamp inputs to prevent extreme values
            action_obs_2 = torch.clamp(action_obs_2, min=-10.0, max=10.0)
            action_obs_3 = torch.clamp(action_obs_3, min=-10.0, max=10.0)
            action_obs_4 = torch.clamp(action_obs_4, min=-10.0, max=10.0)

            # Apply action updater MLP with stability checks
            try:
                # Apply gradient clipping to the action updater inputs
                with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision for this MLP
                    x = action_obs_2.float()

                    # Check if input has nan/inf
                    if torch.isnan(x).any() or torch.isinf(x).any():
                        x = torch.zeros_like(x)

                    # Apply action updater MLP
                    x = self.action_updater(x)

                    # Check for nan/inf in output
                    if torch.isnan(x).any() or torch.isinf(x).any():
                        x = torch.zeros_like(x)

                    action_update_2 = x.view(B, 4, -1)  # [B, 4, action_dim]

                    # For now, just use zeros for the other updates to test
                    action_update_3 = torch.zeros_like(action_update_2)
                    action_update_4 = torch.zeros_like(action_update_2)

            except Exception:
                action_update_2 = torch.zeros(B, 4, action_2.shape[-1], device=action_2.device)
                action_update_3 = torch.zeros(B, 4, action_3.shape[-1], device=action_3.device)
                action_update_4 = torch.zeros(B, 4, action_4.shape[-1], device=action_4.device)

            # Check for nan/inf values
            if torch.isnan(action_update_2).any() or torch.isinf(action_update_2).any():
                action_update_2 = torch.zeros_like(action_update_2)
            if torch.isnan(action_update_3).any() or torch.isinf(action_update_3).any():
                action_update_3 = torch.zeros_like(action_update_3)
            if torch.isnan(action_update_4).any() or torch.isinf(action_update_4).any():
                action_update_4 = torch.zeros_like(action_update_4)

            # Scale down the feedback to prevent instability
            feedback_scale = 0.1  # Small scaling factor
            action_update_2 = action_update_2 * feedback_scale
            action_update_3 = action_update_3 * feedback_scale
            action_update_4 = action_update_4 * feedback_scale

            print(f"action_update2_range: {action_update_2.min():.6f}, {action_update_2.max():.6f}")

            # Add feedback to original actions
            updated_action_2 = action_2 + action_update_2
            updated_action_3 = action_3 + action_update_3
            updated_action_4 = action_4 + action_update_4

            # Concatenate all updated actions
            pred_actions = torch.cat(
                (updated_action_1, updated_action_2, updated_action_3, updated_action_4), dim=1
            )

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask

        # print("pred_actions:", pred_actions)
        # print("velocity:", velocity)
        # print("action_mask:", action_mask)
        # print("loss (before reduction):", F.mse_loss(pred_actions, velocity, reduction="none"))

        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        state_features = self.state_encoder(action_input.state, embodiment_id)  # old encoder

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
