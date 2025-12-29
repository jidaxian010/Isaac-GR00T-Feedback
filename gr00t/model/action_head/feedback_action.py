import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from gr00t.model.action_head.obs_encoder import ObsEncoder


class FeedbackDecoder(nn.Module):
    """
    Transformer-based feedback decoder with DINOv2 visual encoder.

    Design Philosophy:
    - model_output_action is PRIMARY signal (action features dominate)
    - Visual features provide CONTEXT/REFINEMENT (small contribution via learnable gate)
    - Transformer allows action tokens to selectively attend to relevant visual regions
    - Strong residual connection ensures action features remain dominant

    Architecture:
      1. DINOv2 encoder: obs [B,T,V,H,W,C] -> feature map [B, 768, 7, 7]
      2. Flatten to spatial tokens: [B, 49, 768]
      3. Project latent_action and obs_tokens to d_model
      4. TransformerDecoderLayer: cross-attention (action queries attend to visual keys/values)
      5. Residual: tgt_final = tgt + visual_gate * (tgt2 - tgt)  [95% action, 5% visual]
      6. Output head: predict velocity [B, action_horizon, action_dim]

    Why Transformer over MLP?
    - Spatial awareness: Can attend to relevant image regions per action token
    - Flexible: Different action tokens can focus on different visual areas
    - Capacity: Better at learning complex action-visual relationships
    - Already have MLP baseline, transformer adds complementary capabilities
    """

    def __init__(
        self,
        num_categories,
        input_dim,
        hidden_dim,
        output_dim,
        d_model=256,
        nhead=4,
        dim_ff=512,
        visual_gate_initial=0.30,
    ):
        """
        Args:
            num_categories: Number of embodiment categories (not used)
            input_dim: Latent action dimension (e.g., 1024)
            hidden_dim: Not used (kept for compatibility)
            output_dim: Action dimension (e.g., 7 for 7-DOF robot)
            d_model: Transformer hidden dimension (default 256)
            nhead: Number of attention heads (default 4)
            dim_ff: Feedforward dimension (default 512)
            visual_gate_initial: Initial value for visual gate (default 0.30 = 30% visual influence)
        """
        super().__init__()
        self.num_categories = num_categories
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.gate_initial = visual_gate_initial  # Store initial value for reinitialization

        # DINOv2 encoder: outputs [B, 768, 7, 7]
        self.obs_encoder = ObsEncoder(model_name="facebook/dinov2-base", freeze=True)
        C_obs = 768  # DINOv2-base output channels

        # Simple projection layers with Xavier initialization
        self.act_proj = nn.Linear(input_dim, d_model)
        self.obs_proj = nn.Linear(C_obs, d_model)
        self.vel_head = nn.Linear(d_model, output_dim)

        # Initialize with Xavier/Glorot uniform
        nn.init.xavier_uniform_(self.act_proj.weight)
        nn.init.zeros_(self.act_proj.bias)
        nn.init.xavier_uniform_(self.obs_proj.weight)
        nn.init.zeros_(self.obs_proj.bias)
        nn.init.xavier_uniform_(self.vel_head.weight)
        nn.init.zeros_(self.vel_head.bias)

        # Transformer decoder layer (self-attn + cross-attn + FFN)
        self.dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,  # Shapes are [B, T, D]
            dropout=0.0,
            activation="gelu",
            norm_first=True,  # Pre-norm for stability
        )

        # Learnable gate to control visual feature influence
        # This ensures model_output_action is PRIMARY, visual features provide refinement
        # Initial value is configurable via visual_gate_initial parameter
        # Lower value = more action-focused, less visual-dependent
        self.visual_gate = nn.Parameter(torch.tensor(self.gate_initial))

        # Track first forward pass to check weights after checkpoint loads
        self._first_forward_done = False

    def forward(self, model_output_action, obs, cat_ids):
        """
        Args:
            model_output_action: [B, action_horizon, input_dim] - latent action sequence
            obs: [B, T, V, H, W, C] - observation images
            cat_ids: [B,] - embodiment IDs (not used)
        Returns:
            output: [B, action_horizon, action_dim] - predicted velocity
        """
        # Check weights on first forward pass (checkpoint may have corrupted them)
        if not self._first_forward_done:
            self._first_forward_done = True

            # Check if weights have NaN/Inf (corrupted by checkpoint loading)
            weights_corrupted = (
                torch.isnan(self.act_proj.weight).any()
                or torch.isinf(self.act_proj.weight).any()
                or torch.isnan(self.obs_proj.weight).any()
                or torch.isinf(self.obs_proj.weight).any()
                or torch.isnan(self.vel_head.weight).any()
                or torch.isinf(self.vel_head.weight).any()
            )

            # Check visual_gate: should be ~gate_initial, if it's 0 or corrupted, reinitialize
            gate_val = self.visual_gate.item()
            gate_corrupted = (
                torch.isnan(self.visual_gate).any()
                or torch.isinf(self.visual_gate).any()
                or abs(gate_val) < 1e-6  # Effectively 0 (checkpoint might have set it to 0)
                # or abs(gate_val - self.gate_initial) > 0.1  # Way off from expected value
            )

            # Also check transformer decoder layer
            decoder_corrupted = False
            for param in self.dec_layer.parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    decoder_corrupted = True
                    break

            if weights_corrupted or decoder_corrupted or gate_corrupted:
                print("⚠ WARNING: FeedbackDecoder weights corrupted by checkpoint!")
                if gate_corrupted:
                    print(
                        f"  visual_gate is corrupted/reset: {self.visual_gate.item():.6f} (expected ~{self.gate_initial})"
                    )
                print("  Reinitializing with fresh Xavier weights...")

                device = self.act_proj.weight.device
                dtype = self.act_proj.weight.dtype

                # Reinitialize with fresh tensors
                with torch.no_grad():
                    # act_proj
                    fan_in, fan_out = self.act_proj.weight.shape[1], self.act_proj.weight.shape[0]
                    std = (2.0 / (fan_in + fan_out)) ** 0.5
                    self.act_proj.weight.copy_(torch.randn(fan_out, fan_in, device=device, dtype=dtype) * std)
                    self.act_proj.bias.zero_()

                    # obs_proj
                    fan_in, fan_out = self.obs_proj.weight.shape[1], self.obs_proj.weight.shape[0]
                    std = (2.0 / (fan_in + fan_out)) ** 0.5
                    self.obs_proj.weight.copy_(torch.randn(fan_out, fan_in, device=device, dtype=dtype) * std)
                    self.obs_proj.bias.zero_()

                    # vel_head
                    fan_in, fan_out = self.vel_head.weight.shape[1], self.vel_head.weight.shape[0]
                    std = (2.0 / (fan_in + fan_out)) ** 0.5
                    self.vel_head.weight.copy_(torch.randn(fan_out, fan_in, device=device, dtype=dtype) * std)
                    self.vel_head.bias.zero_()

                    # Reinitialize visual_gate to gate_initial if corrupted
                    if gate_corrupted:
                        print(
                            f"  Reinitializing visual_gate from {self.visual_gate.item():.6f} to {self.gate_initial}"
                        )
                        self.visual_gate.copy_(torch.tensor(self.gate_initial, device=device, dtype=dtype))

                    # Reinitialize transformer decoder layer if corrupted
                    if decoder_corrupted:
                        print("  ⚠ Transformer decoder also corrupted - reinitializing...")
                        for name, param in self.dec_layer.named_parameters():
                            if "weight" in name and param.ndim >= 2:
                                # Xavier for weight matrices
                                if param.ndim == 2:
                                    fan_in_dec, fan_out_dec = param.shape[1], param.shape[0]
                                    std_dec = (2.0 / (fan_in_dec + fan_out_dec)) ** 0.5
                                    param.copy_(torch.randn_like(param) * std_dec)
                            elif "bias" in name:
                                param.zero_()

                        # Verify reinitialization worked
                        still_bad = False
                        for name, param in self.dec_layer.named_parameters():
                            if torch.isnan(param).any() or torch.isinf(param).any():
                                print(f"  ✗ WARNING: {name} still has NaN/Inf after reinit!")
                                still_bad = True

                        if still_bad:
                            print("  ⚠ Recreating transformer decoder from scratch...")
                            self.dec_layer = nn.TransformerDecoderLayer(
                                d_model=self.d_model,
                                nhead=4,
                                dim_feedforward=512,
                                batch_first=True,
                                dropout=0.0,
                                activation="gelu",
                                norm_first=True,
                            ).to(device=device, dtype=dtype)

                print(f"✓ All FeedbackDecoder layers reinitialized on {device}")

            # Always check and fix visual_gate if needed (even if other weights are fine)
            if gate_corrupted and not (weights_corrupted or decoder_corrupted):
                # Gate is corrupted but other weights are fine - just fix the gate
                device = self.visual_gate.device
                dtype = self.visual_gate.dtype
                with torch.no_grad():
                    print(f"  Fixing visual_gate: {self.visual_gate.item():.6f} -> {self.gate_initial}")
                    self.visual_gate.copy_(torch.tensor(self.gate_initial, device=device, dtype=dtype))

            # Always print visual_gate value on first forward pass for debugging
            gate_val = self.visual_gate.item()
            print(f"✓ visual_gate initialized: {gate_val:.6f} (visual influence: {gate_val * 100:.2f}%)")

        # 1) Extract last timestep, last view and prepare for DINOv2
        # obs: [B, T, V, H, W, C] -> extract last frame: [B, 224, 224, 3]
        x = obs[:, -1, -1]  # [B, 224, 224, 3]

        # Convert to [B, 3, 224, 224] for DINOv2
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()  # [B, 3, 224, 224]

        # DINOv2 encoder -> feature map [B, 768, 7, 7]
        feat = self.obs_encoder(x)  # [B, 768, 7, 7]

        B, C_obs, H, W = feat.shape
        N = H * W  # Number of spatial tokens (49 for 7x7)

        # 2) Flatten to spatial tokens [B, N, C_obs]
        obs_tokens_raw = feat.flatten(2).transpose(1, 2)  # [B, 49, 768]

        # 3) Project to d_model with simple Linear layers
        memory = self.obs_proj(obs_tokens_raw)  # [B, 49, d_model]
        tgt = self.act_proj(model_output_action)  # [B, 16, d_model]

        # Runtime check for NaN/extreme values (indicates bad weights)
        memory_bad = torch.isnan(memory).any() or torch.isinf(memory).any() or memory.abs().max() > 1e10
        tgt_bad = torch.isnan(tgt).any() or torch.isinf(tgt).any() or tgt.abs().max() > 1e10

        if memory_bad or tgt_bad:
            print("\n⚠ CRITICAL: Projection outputs have extreme/NaN values!")
            print(f"  obs_tokens range: [{obs_tokens_raw.min().item():.2f}, {obs_tokens_raw.max().item():.2f}]")
            print(
                f"  model_output_action range: [{model_output_action.min().item():.2f}, {model_output_action.max().item():.2f}]"
            )
            print(f"  memory range: [{memory.min().item():.2e}, {memory.max().item():.2e}]")
            print(f"  tgt range: [{tgt.min().item():.2e}, {tgt.max().item():.2e}]")
            print(
                f"  obs_proj weight range: [{self.obs_proj.weight.min().item():.2e}, {self.obs_proj.weight.max().item():.2e}]"
            )
            print(
                f"  act_proj weight range: [{self.act_proj.weight.min().item():.2e}, {self.act_proj.weight.max().item():.2e}]"
            )

            print("\n  Attempting emergency reinitialization with fresh tensors...")
            device = self.obs_proj.weight.device
            dtype = self.obs_proj.weight.dtype

            # Create completely fresh tensors using torch.randn
            with torch.no_grad():
                # obs_proj
                fan_in, fan_out = self.obs_proj.weight.shape[1], self.obs_proj.weight.shape[0]
                std = (2.0 / (fan_in + fan_out)) ** 0.5
                self.obs_proj.weight.copy_(torch.randn(fan_out, fan_in, device=device, dtype=dtype) * std)
                self.obs_proj.bias.zero_()

                # act_proj
                fan_in, fan_out = self.act_proj.weight.shape[1], self.act_proj.weight.shape[0]
                std = (2.0 / (fan_in + fan_out)) ** 0.5
                self.act_proj.weight.copy_(torch.randn(fan_out, fan_in, device=device, dtype=dtype) * std)
                self.act_proj.bias.zero_()

            print(f"  Weights reinitialized on {device}, dtype={dtype}")
            print(
                f"  New obs_proj weight range: [{self.obs_proj.weight.min().item():.3f}, {self.obs_proj.weight.max().item():.3f}]"
            )
            print(
                f"  New act_proj weight range: [{self.act_proj.weight.min().item():.3f}, {self.act_proj.weight.max().item():.3f}]"
            )

            # Retry
            memory = self.obs_proj(obs_tokens_raw)
            tgt = self.act_proj(model_output_action)
            print(f"  After reinit - memory range: [{memory.min().item():.2e}, {memory.max().item():.2e}]")
            print(f"  After reinit - tgt range: [{tgt.min().item():.2e}, {tgt.max().item():.2e}]")

            # Final check
            memory_bad = torch.isnan(memory).any() or torch.isinf(memory).any() or memory.abs().max() > 1e10
            tgt_bad = torch.isnan(tgt).any() or torch.isinf(tgt).any() or tgt.abs().max() > 1e10
            if memory_bad or tgt_bad:
                raise RuntimeError(
                    "Projection layers still bad after reinitialization!\n"
                    "The checkpoint is incompatible with this architecture."
                )

        # 4) Transformer decoder: (self-attn on tgt) + (cross-attn tgt<-memory) + FFN
        # This allows action tokens to attend to visual tokens, but we'll heavily weight
        # the original action features to ensure they remain primary
        tgt2 = self.dec_layer(tgt, memory)  # [B, 16, d_model]

        # 4.25) Stabilize transformer output to prevent gradient explosion
        # Clip extreme values and scale to match input magnitude
        tgt2_max = tgt2.abs().max().item()
        if tgt2_max > 5.0:
            # Scale down if too large (prevents gradient explosion)
            scale_factor = 5.0 / tgt2_max
            tgt2 = tgt2 * scale_factor
            if not hasattr(self, "_warned_large_output"):
                print(f"⚠ WARNING: Large transformer output (max={tgt2_max:.2f}), scaling by {scale_factor:.3f}")
                self._warned_large_output = True

        # 4.5) Strong residual connection: action features are PRIMARY, visual features are refinement
        # Formula: tgt_final = tgt + visual_gate * (tgt2 - tgt)
        # visual_gate controls the balance: (1-gate)% action features, gate% visual refinement
        # This ensures model_output_action dominates, visual features provide refinement
        # The transformer learns HOW to refine, but action features are the base
        tgt_final = tgt + self.visual_gate * (tgt2 - tgt)  # [B, 16, d_model]

        # Check if transformer decoder produced NaN
        if torch.isnan(tgt2).any() or torch.isinf(tgt2).any():
            print("\n⚠ CRITICAL: Transformer decoder produced NaN!")
            print(f"  Input tgt range: [{tgt.min().item():.2f}, {tgt.max().item():.2f}]")
            print(f"  Input memory range: [{memory.min().item():.2f}, {memory.max().item():.2f}]")
            print(f"  Output tgt2 has NaN: {torch.isnan(tgt2).any()}")
            print(f"  Output tgt2 has Inf: {torch.isinf(tgt2).any()}")
            print(f"  Visual gate value: {self.visual_gate.item():.3f}")

            # Check decoder layer weights in detail
            print("\n  Checking decoder layer weights:")
            weights_bad = False
            for name, param in self.dec_layer.named_parameters():
                has_nan = torch.isnan(param).any().item()
                has_inf = torch.isinf(param).any().item()
                if has_nan or has_inf:
                    print(f"    ✗ {name}: NaN={has_nan}, Inf={has_inf}")
                    weights_bad = True
                else:
                    pmin, pmax = param.min().item(), param.max().item()
                    print(f"    ✓ {name}: range=[{pmin:.3f}, {pmax:.3f}]")

            if not weights_bad:
                print("  ⚠ Weights look OK but output is NaN - possible numerical instability!")
                print("  Attempting to recreate transformer decoder with better stability...")

                device = tgt.device
                dtype = tgt.dtype

                # Recreate with layer norm epsilon for stability
                self.dec_layer = nn.TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=4,
                    dim_feedforward=512,
                    batch_first=True,
                    dropout=0.0,
                    activation="gelu",
                    norm_first=True,
                    layer_norm_eps=1e-5,  # Default, but explicit
                ).to(device=device, dtype=dtype)

                print("  Retrying forward pass with fresh decoder...")
                tgt2 = self.dec_layer(tgt, memory)
                tgt_final = tgt + self.visual_gate * (tgt2 - tgt)

                if torch.isnan(tgt2).any() or torch.isinf(tgt2).any():
                    print("  ✗ Still NaN after recreation!")
                    raise RuntimeError(
                        "TransformerDecoderLayer producing NaN even after recreation!\n"
                        "This suggests a fundamental incompatibility.\n"
                        "Try training from scratch WITHOUT loading any checkpoint."
                    )
                else:
                    print(f"  ✓ SUCCESS! New tgt2 range: [{tgt2.min().item():.2f}, {tgt2.max().item():.2f}]")
                    print(f"  tgt_final range: [{tgt_final.min().item():.2f}, {tgt_final.max().item():.2f}]")
            else:
                raise RuntimeError(
                    "TransformerDecoderLayer weights still corrupted!\n"
                    "The checkpoint loading is overriding our reinitialization.\n"
                    "Solution: Train from scratch WITHOUT loading checkpoint."
                )

        # 5) Output velocity per action token
        pred_vel = self.vel_head(tgt_final)  # [B, 16, action_dim]

        # Debug prints
        print(f"obs: {obs.shape}, range [{obs.min().item()}, {obs.max().item()}]")
        print(f"x (prepared): {x.shape}")
        print(f"feat (DINOv2): {feat.shape}, range [{feat.min().item():.2f}, {feat.max().item():.2f}]")
        print(f"obs_tokens: {obs_tokens_raw.shape}, spatial_tokens={N}")
        print(
            f"model_output_action: {model_output_action.shape}, range [{model_output_action.min().item():.2f}, {model_output_action.max().item():.2f}]"
        )
        print(f"memory: {memory.shape}, range [{memory.min().item():.2f}, {memory.max().item():.2f}]")
        print(f"tgt (action features): {tgt.shape}, range [{tgt.min().item():.2f}, {tgt.max().item():.2f}]")
        print(f"tgt2 (transformer output): {tgt2.shape}, range [{tgt2.min().item():.2f}, {tgt2.max().item():.2f}]")
        print(
            f"visual_gate: {self.visual_gate.item():.3f} (visual influence: {self.visual_gate.item() * 100:.1f}%)"
        )
        print(
            f"tgt_final (tgt + gate*(tgt2-tgt)): {tgt_final.shape}, range [{tgt_final.min().item():.2f}, {tgt_final.max().item():.2f}]"
        )
        print(f"pred_vel: {pred_vel.shape}, range [{pred_vel.min().item():.2f}, {pred_vel.max().item():.2f}]")

        return pred_vel


class FeedbackAction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.hidden_size = config.hidden_size

        self.action_decoder_observe = FeedbackDecoder(
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
        final_model_output_action = action_head_output.final_model
        final_raw_action = action_head_output.final_raw_action
        dt = action_head_output.dt

        pred_velocity = self.action_decoder_observe(
            final_model_output_action, action_input.simple_img, action_input.embodiment_id
        )
        pred_actions = final_raw_action + dt * pred_velocity

        pred_actions_normalized = torch.tanh(pred_actions)  # normalize to match gt_actions

        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions_normalized, gt_actions, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    def get_action(
        self, action_head_output: BatchFeature, time_step: int, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Process action prediction during inference, applying action_decoder_observe
        at the final step similar to training.
        """
        # Check if we have the final step information (from flow_matching_action_head)

        print(f"@ feedback_action time_step: {time_step}")
        window_idx = time_step % 4

        final_model_output_action = action_head_output.final_model
        final_raw_action = action_head_output.final_raw_action
        dt = action_head_output.dt
    
        # Apply action_decoder_observe similar to training forward
        pred_velocity = self.action_decoder_observe(
            final_model_output_action, action_input.simple_img, action_input.embodiment_id
        )
        pred_actions = final_raw_action + dt * pred_velocity
        pred_actions_window = pred_actions[:, window_idx * 4 : (window_idx + 1) * 4, :]
        return BatchFeature(data={"action_pred": pred_actions_window})
