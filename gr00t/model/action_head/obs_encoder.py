import torch
import torch.nn as nn
import torch.nn.functional as F


class ObsEncoder(nn.Module):
    """
    DINOv2-based ObsEncoder that returns features with shape (B, 768, 7, 7).

    Input:
      obs: (B, 3, 224, 224)  float in [0,1] or [0,255]

    Output:
      feat_7: (B, 768, 7, 7)

    Notes:
    - Uses HuggingFace `transformers` AutoModel for DINOv2.
    - Drops CLS token and reshapes patch tokens into a 2D grid.
    - Pools to 7x7 with adaptive average pooling.
    - Freezes backbone by default (fast + stable).
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, model_name: str = "facebook/dinov2-base", freeze: bool = True):
        super().__init__()
        try:
            from transformers import AutoModel
        except Exception as e:
            raise ImportError(
                "This encoder requires HuggingFace transformers.\nInstall with: pip install transformers"
            ) from e

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        hidden = getattr(self.model.config, "hidden_size", None)
        if hidden != 768:
            raise ValueError(
                f"Expected a DINOv2 model with hidden_size=768, but got hidden_size={hidden}. "
                "Use a base-sized DINOv2 model, e.g. facebook/dinov2-base."
            )

        self.freeze = freeze
        self._weights_reloaded = False

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        # Check if weights are valid (skip for meta tensors)
        self._check_weights()

    def _check_weights(self):
        """Check if weights are valid (not NaN/Inf)."""
        sample_param = next(self.model.parameters())

        # Skip if on meta device (loading from checkpoint)
        if sample_param.device.type == "meta":
            print("⚠ DINOv2 on meta device - will be loaded from checkpoint")
            return

        # Check for NaN
        if torch.isnan(sample_param).any():
            print("⚠ WARNING: DINOv2 weights contain NaN!")
            print("  This likely means checkpoint has corrupted DINOv2 weights")
            print("  Will auto-reload fresh weights on first forward pass")

    def _reload_weights(self):
        """Reload fresh DINOv2 weights."""
        print("\n" + "=" * 60)
        print("Reloading DINOv2 with fresh pretrained weights...")
        print("=" * 60)

        device = next(self.model.parameters()).device

        # Skip if on meta device
        if device.type == "meta":
            print("⚠ Cannot reload: on meta device")
            return False

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(self.model_name).to(device)

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        # Verify
        sample_param = next(self.model.parameters())
        if not torch.isnan(sample_param).any():
            print(f"✓ DINOv2 reloaded successfully on {device}")
            print(f"  Sample weight range: [{sample_param.min().item():.3f}, {sample_param.max().item():.3f}]")
            print("=" * 60 + "\n")
            return True
        else:
            print("✗ FAILED: Reloaded weights still contain NaN!")
            print("=" * 60 + "\n")
            return False

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            x = x.float()
        # If input looks like [0,255], rescale to [0,1]
        if x.max() > 1.5:
            x = x / 255.0
        mean = self.IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
        std = self.IMAGENET_STD.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, 3, 224, 224)
        returns: (B, 768, 7, 7)
        """
        x = self._normalize(obs)

        if self.freeze:
            with torch.no_grad():
                out = self.model(pixel_values=x)
        else:
            out = self.model(pixel_values=x)

        # last_hidden_state: (B, 1 + P, 768) where P is num patch tokens
        tok = out.last_hidden_state  # (B, 1+P, 768)
        tok = tok[:, 1:, :]  # drop CLS -> (B, P, 768)

        B, P, C = tok.shape
        g = int(P**0.5)
        if g * g != P:
            raise ValueError(f"Patch token count P={P} is not a perfect square; can't reshape to grid.")

        # reshape into (B, 768, g, g)
        feat = tok.transpose(1, 2).contiguous().view(B, C, g, g)

        # pool to (B, 768, 7, 7)
        features = F.adaptive_avg_pool2d(feat, (7, 7))

        # Check for NaN and auto-reload if needed
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("\n⚠ WARNING: DINOv2 output contains NaN/Inf!")
            print(f"  Input x range: [{x.min().item():.3f}, {x.max().item():.3f}]")
            print(f"  Features shape: {features.shape}")
            print(f"  NaN count: {torch.isnan(features).sum().item()} / {features.numel()}")

            # Try to reload weights once
            if not self._weights_reloaded:
                success = self._reload_weights()
                self._weights_reloaded = True

                if success:
                    # Retry forward pass
                    print("Retrying forward pass with fresh weights...")
                    if self.freeze:
                        with torch.no_grad():
                            out = self.model(pixel_values=x)
                    else:
                        out = self.model(pixel_values=x)

                    tok = out.last_hidden_state[:, 1:, :]
                    B, P, C = tok.shape
                    g = int(P**0.5)
                    feat = tok.transpose(1, 2).contiguous().view(B, C, g, g)
                    features = F.adaptive_avg_pool2d(feat, (7, 7))

                    if not torch.isnan(features).any() and not torch.isinf(features).any():
                        print(
                            f"✓ SUCCESS! Features range: [{features.min().item():.3f}, {features.max().item():.3f}]\n"
                        )
                        return features
                    else:
                        print("✗ Still NaN after reload\n")

            raise RuntimeError(
                "DINOv2 producing NaN values!\n"
                "This means the checkpoint has corrupted DINOv2 weights.\n"
                "Solution: Either train from scratch or use a checkpoint without DINOv2 weights."
            )

        return features
