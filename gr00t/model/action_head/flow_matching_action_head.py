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
from .vl_updater import VLM_Updater

from .cross_attention_dit import DiT, SelfAttentionTransformer


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
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

# # Encode state and obs together
# class StateObsMLP(nn.Module):
#     def __init__(self, num_categories, state_input_dim, emb_dim, output_dim):
#         super().__init__()
#         self.num_categories = num_categories
#         self.obs_layer = ObsEncoder(emb_dim) # defined in obs_encoder.py
#         self.state_layer = CategorySpecificLinear(num_categories, state_input_dim, emb_dim) # 
#         self.layer2 = CategorySpecificLinear(num_categories, emb_dim*2, output_dim)

#     def forward(self, state, obs, cat_ids):
#         obs_emb = self.obs_layer(obs)
#         state_emb = F.relu(self.state_layer(state))
#         x = torch.cat((state_emb, obs_emb), dim=1) # 

#         return self.layer2(x, cat_ids) # 1536

# Encode state and obs together
class StateObsMLP(nn.Module):
    def __init__(self, num_categories, state_input_dim, emb_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.obs_layer = ObsEncoder(emb_dim) # defined in obs_encoder.py
        self.obs_layer_mlp = CategorySpecificLinear(num_categories, emb_dim, emb_dim)
        self.state_layer = CategorySpecificLinear(num_categories, state_input_dim, emb_dim) # 
        self.layer2 = CategorySpecificLinear(num_categories, emb_dim*2, output_dim)
        
        # Initialize weights properly for better training stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state, obs, cat_ids):

        obs_linear = (obs[:, -1, -1].float().view(obs.shape[0], -1)) / 255.0  # (B, 224*224*3)
        
        obs_linear_512 = obs_linear[:, :512].unsqueeze(1)  # (B, 1, 512)
        
        # obs_emb = self.obs_layer_mlp(obs_linear_512, cat_ids)  # (B, 1, emb_dim)
        obs_emb = self.obs_layer(obs) # shape: (B, 1, emb_dim) using cnn

        # Ensure state has the right shape for CategorySpecificLinear: (B, 1, state_dim)
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (B, state_dim) -> (B, 1, state_dim)
        
        state_emb = F.relu(self.state_layer(state, cat_ids))  # shape: (B, 1, emb_dim)
        
        x = torch.cat((state_emb, obs_emb), dim=2) # shape: (B, 1, 2*emb_dim)

        output = self.layer2(x, cat_ids) # 1536
        
        return output




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
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

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

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

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
        self.state_obs_encoder = StateObsMLP(
            num_categories=config.max_num_embodiments,
            state_input_dim=config.max_state_dim, # 64
            emb_dim=self.hidden_size // 2, # 1024 / 2 = 512
            output_dim=self.input_embedding_dim, # fixed, 1536
        )
        self.vlm_updater = VLM_Updater(
            num_categories=config.max_num_embodiments,
            vlm_dim=config.backbone_embedding_dim,
            emb_dim=self.hidden_size // 2,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
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
        
        # Always ensure the new state_obs_encoder is trainable (not affected by tune_projector)
        self.state_obs_encoder.requires_grad_(True)
        
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print("New state_obs_encoder is always trainable (not through LoRA)")
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
    
    def process_ground_truth_vlm_emb(self, ground_truth_vlm_emb: torch.Tensor) -> torch.Tensor:
        """Process ground truth VLM embeddings the same way as backbone features"""
        processed = self.vlln(ground_truth_vlm_emb)
        processed = self.vl_self_attention(processed)
        return processed

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        1. ADD OBS
        """
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

        ## Embed state.
        # state_features = self.state_encoder(action_input.state, embodiment_id) # old encoder
        state_features = self.state_obs_encoder(action_input.state, action_input.simple_img, embodiment_id)

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
        min_len = min(state_features.shape[0], future_tokens.shape[0], action_features.shape[0])

        # Slice all tensors to the minimum batch size
        state_features = state_features[:min_len]
        future_tokens = future_tokens[:min_len]
        action_features = action_features[:min_len]

        # Debug printout
        if not (state_features.shape[0] == future_tokens.shape[0] == action_features.shape[0]):
            print(
                f"[DEBUG] Batch size mismatch after slicing! "
                f"state_features: {state_features.shape}, "
                f"future_tokens: {future_tokens.shape}, "
                f"action_features: {action_features.shape}"
            )
        else:
            pass

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



    # def forward(self, backbone_output: BatchFeature, action_input: BatchFeature, init: bool = False, ground_truth_vlm_emb: torch.Tensor = None) -> BatchFeature:
    #     """
    #     2. VLM UPDATER
    #     """
    #     # Set frozen modules to eval
    #     self.set_frozen_modules_to_eval_mode()

    #     backbone_output = self.process_backbone_output(backbone_output)

    #     if self.config.expand_batch is not None:
    #         for k, v in backbone_output.items():
    #             ndim = len(v.shape)
    #             factors = [self.config.expand_batch]
    #             while len(factors) < ndim:
    #                 factors.append(1)
    #             factors = tuple(factors)
    #             expanded = v.repeat(*factors)
    #             backbone_output[k] = expanded

    #         for k, v in action_input.items():
    #             ndim = len(v.shape)
    #             factors = [self.config.expand_batch]
    #             while len(factors) < ndim:
    #                 factors.append(1)
    #             factors = tuple(factors)
    #             expanded = v.repeat(*factors)
    #             action_input[k] = expanded

        
    #     # Get embodiment ID.
    #     embodiment_id = action_input.embodiment_id

    #     # Get vision and language embeddings.
    #     vl_embs = backbone_output.backbone_features

    #     predicted_vlm_emb = None
    #     if init:
    #         # First step: use original VLM embeddings
    #         predicted_vlm_emb = vl_embs
    #     else:
    #         # Other steps: update VLM embeddings with current image
    #         predicted_vlm_emb = self.vlm_updater(action_input.simple_img, vl_embs, embodiment_id)
    #         # Store for next step:
    #         backbone_output["backbone_features"] = predicted_vlm_emb
    #     # Get device AFTER vl_embs might have been updated
    #     device = predicted_vlm_emb.device

    #     state_features = self.state_encoder(action_input.state, embodiment_id) # old encoder

    #     # Embed noised action trajectory.
    #     actions = action_input.action
    #     noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
    #     t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
    #     t = t[:, None, None]  # shape (B,1,1) for broadcast

    #     noisy_trajectory = (1 - t) * noise + t * actions
    #     velocity = actions - noise

    #     # Convert (continuous) t -> discrete if needed
    #     t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
    #     action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

    #     # Maybe add position embedding.
    #     if self.config.add_pos_embed:
    #         pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
    #         pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
    #         action_features = action_features + pos_embs

    #     # Join vision, language, state and action embedding along sequence dimension.
    #     future_tokens = self.future_tokens.weight.unsqueeze(0).expand(predicted_vlm_emb.shape[0], -1, -1)

    #     # Get the minimum batch size among all tensors to be concatenated
    #     min_len = min(state_features.shape[0], future_tokens.shape[0], action_features.shape[0])

    #     # Slice all tensors to the minimum batch size
    #     state_features = state_features[:min_len]
    #     future_tokens = future_tokens[:min_len]
    #     action_features = action_features[:min_len]

    #     # Debug printout
    #     if not (state_features.shape[0] == future_tokens.shape[0] == action_features.shape[0]):
    #         print(
    #             f"[DEBUG] Batch size mismatch after slicing! "
    #             f"state_features: {state_features.shape}, "
    #             f"future_tokens: {future_tokens.shape}, "
    #             f"action_features: {action_features.shape}"
    #         )
    #     else:
    #         pass

    #     sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

    #     vl_attn_mask = backbone_output.backbone_attention_mask

    #     model_output = self.model(
    #         hidden_states=sa_embs,
    #         encoder_hidden_states=predicted_vlm_emb,
    #         encoder_attention_mask=vl_attn_mask,
    #         timestep=t_discretized,
    #         return_all_hidden_states=False,  # NOTE (YL): not using flare now
    #     )
    #     pred = self.action_decoder(model_output, embodiment_id)
    #     pred_actions = pred[:, -actions.shape[1] :]

    #     # Slice out only the action portion of pred and target.
    #     action_mask = action_input.action_mask



    #     # Compute action loss with numerical stability
    #     action_loss_raw = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
    #     action_loss = action_loss_raw.sum() / (action_mask.sum() + 1e-8)  # Add epsilon for stability
        
    #     # Debug action loss
    #     print(f"pred_actions range: [{pred_actions.min().item():.6f}, {pred_actions.max().item():.6f}]")
    #     print(f"velocity range: [{velocity.min().item():.6f}, {velocity.max().item():.6f}]")
        
    #     # Compute VLM prediction loss (if ground truth is provided and not first step)
    #     vlm_loss = torch.tensor(0.0, device=action_loss.device, dtype=action_loss.dtype)
    #     if ground_truth_vlm_emb is not None and not init:
    #         # Process ground truth embeddings the same way as backbone features
    #         ground_truth_vlm_emb_processed = self.process_ground_truth_vlm_emb(ground_truth_vlm_emb)
            
    #         # Ensure both tensors have the same sequence length
    #         min_seq_len = min(predicted_vlm_emb.shape[1], ground_truth_vlm_emb_processed.shape[1])
    #         predicted_vlm_emb_sliced = predicted_vlm_emb[:, :min_seq_len, :]
    #         ground_truth_vlm_emb_sliced = ground_truth_vlm_emb_processed[:, :min_seq_len, :]
            
    #         # Add numerical stability checks
    #         print(f"predicted_vlm_emb_sliced range: [{predicted_vlm_emb_sliced.min().item():.6f}, {predicted_vlm_emb_sliced.max().item():.6f}]")
    #         print(f"ground_truth_vlm_emb_sliced range: [{ground_truth_vlm_emb_sliced.min().item():.6f}, {ground_truth_vlm_emb_sliced.max().item():.6f}]")
            
    #         # Check for NaN or inf values
    #         if torch.isnan(predicted_vlm_emb_sliced).any() or torch.isinf(predicted_vlm_emb_sliced).any():
    #             print("WARNING: predicted_vlm_emb_sliced contains NaN or inf values!")
    #         if torch.isnan(ground_truth_vlm_emb_sliced).any() or torch.isinf(ground_truth_vlm_emb_sliced).any():
    #             print("WARNING: ground_truth_vlm_emb_sliced contains NaN or inf values!")
    #         if torch.isnan(ground_truth_vlm_emb_processed).any() or torch.isinf(ground_truth_vlm_emb_processed).any():
    #             print("WARNING: ground_truth_vlm_emb_processed contains NaN or inf values!")
            
    #         vlm_loss = F.mse_loss(predicted_vlm_emb_sliced, ground_truth_vlm_emb_sliced)
    #         print(f"vlm_loss: {vlm_loss.item()}")
    #     else:
    #         print(f"VLM loss not computed - init: {init}, ground_truth_vlm_emb is None: {ground_truth_vlm_emb is None}")
        
    #     # Combine losses with numerical stability
    #     total_loss = action_loss + 0.5 * vlm_loss  # Increase VLM loss weight
        
    #     # Check for NaN or inf in total loss
    #     if torch.isnan(total_loss) or torch.isinf(total_loss):
    #         print("WARNING: total_loss is NaN or inf! Using fallback loss.")
    #         total_loss = torch.tensor(1.0, device=action_loss.device, dtype=action_loss.dtype, requires_grad=True)
        
        
    #     output_dict = {
    #         "loss": total_loss,
    #         "action_loss": action_loss,
    #         "vlm_loss": vlm_loss,
    #         "loss_breakdown": {
    #             "action_loss": action_loss.item(),
    #             "vlm_loss": vlm_loss.item(),
    #             "total_loss": total_loss.item()
    #         }
    #     }
        
    #     # Add the updated VLM embeddings to the output for reuse
    #     if not init:
    #         output_dict["updated_vlm_emb"] = predicted_vlm_emb
            
    #     return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        # state_features = self.state_encoder(action_input.state, embodiment_id)
        state_features = self.state_obs_encoder(action_input.state, action_input.simple_img, embodiment_id)

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
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
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
