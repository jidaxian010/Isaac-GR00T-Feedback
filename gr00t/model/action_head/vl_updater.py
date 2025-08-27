import torch
import torch.nn.functional as F
from torch import nn

from .obs_encoder import ObsEncoder


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


class VLM_Updater(nn.Module):
    def __init__(self, num_categories, vlm_dim, emb_dim, num_heads=8):
        super().__init__()
        self.vlm_dim = vlm_dim
        self.attention = nn.MultiheadAttention(embed_dim=vlm_dim, num_heads=num_heads, dropout=0.0, batch_first=True)

        self.obs_encoder = ObsEncoder(emb_dim)
        # Project image features to VLM dimension
        self.img_projection = nn.Linear(emb_dim, vlm_dim)
        # Fusion layer to combine VLM and image features
        self.fusion = CategorySpecificLinear(num_categories, vlm_dim * 2, vlm_dim)
        
        # Add layer normalization for stability
        self.img_norm = nn.LayerNorm(vlm_dim)
        self.attention_norm = nn.LayerNorm(vlm_dim)
        self.fusion_norm = nn.LayerNorm(vlm_dim)
        
        # Initialize weights with smaller values for stability
        self._init_weights()
        
        # Force materialization of meta tensors in obs_encoder
        self.obs_encoder._materialize_meta_tensors()
        
        # Ensure attention module is in float32 for stability
        self.attention = self.attention.float()
    
    def _init_weights(self):
        """Initialize weights with smaller values to prevent NaN"""
        # Initialize attention weights
        for name, param in self.attention.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize projection and fusion weights
        for module in [self.img_projection, self.fusion]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=0.01)  # Smaller gain
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, obs, pre_vlm_emb, cat_ids):
        # obs: (B, T, V, H, W, C) -> ObsEncoder -> (B, 1, emb_dim)  
        img_features = self.obs_encoder(obs)  # (B, 1, emb_dim)
        
        # Remove the extra dimension to get (B, emb_dim)
        img_features = img_features.squeeze(1)  # (B, emb_dim)
        
        # Debug: Check img_features
        if torch.isnan(img_features).any() or torch.isinf(img_features).any():
            print("WARNING: img_features contains NaN or inf values!")
            print(f"img_features range: [{img_features.min().item():.6f}, {img_features.max().item():.6f}]")
        
        # Project image features to VLM dimension
        img_features_vlm = self.img_projection(img_features)  # (B, vlm_dim)
        
        # Normalize projected features
        img_features_vlm = self.img_norm(img_features_vlm)
        
        # Debug: Check img_features_vlm
        if torch.isnan(img_features_vlm).any() or torch.isinf(img_features_vlm).any():
            print("WARNING: img_features_vlm contains NaN or inf values!")
            print(f"img_features_vlm range: [{img_features_vlm.min().item():.6f}, {img_features_vlm.max().item():.6f}]")
        
        # Expand image features to match VLM sequence length
        B, N, D = pre_vlm_emb.shape  # (B, N, vlm_dim)
        img_features_expanded = img_features_vlm.unsqueeze(1).expand(B, N, D)  # (B, N, vlm_dim)
        
        # Debug: Check pre_vlm_emb
        if torch.isnan(pre_vlm_emb).any() or torch.isinf(pre_vlm_emb).any():
            print("WARNING: pre_vlm_emb contains NaN or inf values!")
            print(f"pre_vlm_emb range: [{pre_vlm_emb.min().item():.6f}, {pre_vlm_emb.max().item():.6f}]")
        
        # Use attention to update VLM embeddings with image information
        # Ensure consistent dtype for attention
        query = pre_vlm_emb.to(torch.float32)
        key = img_features_expanded.to(torch.float32)
        value = img_features_expanded.to(torch.float32)
        
        # Clip ALL inputs to prevent explosion
        query = torch.clamp(query, -10.0, 10.0)
        key = torch.clamp(key, -10.0, 10.0)
        value = torch.clamp(value, -10.0, 10.0)
        
        # Debug: Check attention inputs
        if torch.isnan(query).any() or torch.isinf(query).any():
            print("WARNING: query contains NaN or inf values!")
            print(f"query range: [{query.min().item():.6f}, {query.max().item():.6f}]")
        if torch.isnan(key).any() or torch.isinf(key).any():
            print("WARNING: key contains NaN or inf values!")
            print(f"key range: [{key.min().item():.6f}, {key.max().item():.6f}]")
        if torch.isnan(value).any() or torch.isinf(value).any():
            print("WARNING: value contains NaN or inf values!")
            print(f"value range: [{value.min().item():.6f}, {value.max().item():.6f}]")
        
        # Let's manually compute attention to debug the issue
        # Compute attention scores: (B, N, N) = (B, N, D) @ (B, D, N)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # (B, N, N)
        
        # Scale by sqrt(d_k)
        d_k = query.size(-1)
        attention_scores = attention_scores / d_k**0.5  
        
        # Clip attention scores to prevent extreme values
        attention_scores = torch.clamp(attention_scores, -10.0, 10.0)
        
        # Debug: Check attention scores before softmax
        print(f"attention_scores range: [{attention_scores.min().item():.6f}, {attention_scores.max().item():.6f}]")
        print(f"attention_scores std: {attention_scores.std().item():.6f}")
        if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
            print("WARNING: attention_scores contains NaN or inf values!")
            print(f"attention_scores shape: {attention_scores.shape}")
            print(f"d_k: {d_k}")
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Debug: Check attention weights after softmax
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            print("WARNING: attention_weights contains NaN or inf values!")
            print(f"attention_weights range: [{attention_weights.min().item():.6f}, {attention_weights.max().item():.6f}]")
            print(f"attention_weights shape: {attention_weights.shape}")
        
        # Debug: Check value tensor
        print(f"value range: [{value.min().item():.6f}, {value.max().item():.6f}]")
        print(f"attention_weights range: [{attention_weights.min().item():.6f}, {attention_weights.max().item():.6f}]")
        
        # Clip value tensor to prevent explosion
        value = torch.clamp(value, -10.0, 10.0)
        
        # Apply attention weights to values
        vlm_updated = torch.matmul(attention_weights, value)  # (B, N, D)
        
        # Clip the output immediately
        vlm_updated = torch.clamp(vlm_updated, -10.0, 10.0)
        
        # Scale the output to match the original embedding scale
        vlm_updated = vlm_updated * 100.0  # More reasonable scaling
        
        # Clip again after scaling
        vlm_updated = torch.clamp(vlm_updated, -1000.0, 1000.0)
        
        # Convert back to original dtype
        vlm_updated = vlm_updated.to(pre_vlm_emb.dtype)
        
        # Debug: Check vlm_updated before normalization
        if torch.isnan(vlm_updated).any() or torch.isinf(vlm_updated).any():
            print("WARNING: vlm_updated contains NaN or inf values before normalization!")
            print(f"vlm_updated range: [{vlm_updated.min().item():.6f}, {vlm_updated.max().item():.6f}]")
        else:
            # Check for extreme values that might cause issues
            vlm_max = vlm_updated.abs().max().item()
            if vlm_max > 100.0:
                print(f"WARNING: vlm_updated has large values: max_abs={vlm_max:.2f}")
                print(f"vlm_updated range: [{vlm_updated.min().item():.6f}, {vlm_updated.max().item():.6f}]")
        
        # Normalize attention output
        vlm_updated = self.attention_norm(vlm_updated)
        
        # Debug: Check vlm_updated
        if torch.isnan(vlm_updated).any() or torch.isinf(vlm_updated).any():
            print("WARNING: vlm_updated contains NaN or inf values!")
            return pre_vlm_emb  # Return original embeddings if NaN detected
        
        # Combine original and updated VLM embeddings
        combined = torch.cat((pre_vlm_emb, vlm_updated), dim=-1)  # (B, N, vlm_dim * 2)
        
        # Debug: Check combined
        if torch.isnan(combined).any() or torch.isinf(combined).any():
            print("WARNING: combined contains NaN or inf values!")
            print(f"combined range: [{combined.min().item():.6f}, {combined.max().item():.6f}]")
        
        # Apply fusion to get final VLM embeddings
        current_vlm = self.fusion(combined, cat_ids)  # (B, N, vlm_dim)
        
        # Normalize fusion output
        current_vlm = self.fusion_norm(current_vlm)
        
        # Debug: Check current_vlm
        if torch.isnan(current_vlm).any() or torch.isinf(current_vlm).any():
            print("WARNING: current_vlm contains NaN or inf values!")
            print(f"current_vlm range: [{current_vlm.min().item():.6f}, {current_vlm.max().item():.6f}]")
        
        return current_vlm

############################################################################
## 2. VLM updater, by adding
class VLMUpdateModule(nn.Module):
    """VLM update module using RMS normalization and learnable blending"""
    
    def __init__(self, obs_dim, vlm_dim, num_categories):
        super().__init__()
        self.obs_dim = obs_dim
        self.vlm_dim = vlm_dim
        self.num_categories = num_categories
        
        # 1) Project obs to VLM space
        self.obs_projection = CategorySpecificLinear(num_categories, obs_dim, vlm_dim)
        
        # 2) Learnable blending weight α (sigmoid to bound [0,1])
        self.alpha_logit = nn.Parameter(torch.empty(num_categories))
        
        # 3) Learnable target RMS for final scaling
        self.target_rms = nn.Parameter(torch.empty(num_categories))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        # Initialize alpha to start around 0.85 (sigmoid(1.7) ≈ 0.85)
        nn.init.uniform_(self.alpha_logit, 1.5, 2.0)  # sigmoid gives 0.82-0.88
        
        # Initialize target RMS based on typical VLM magnitude
        nn.init.constant_(self.target_rms, 100.0)
    
    def rms(self, x, eps=1e-6):
        """Compute RMS along feature dimension"""
        return (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    
    def forward(self, past_vlm, obs_features, cat_ids):
        """
        Args:
            past_vlm: (B, T, d) - past VLM embeddings
            obs_features: (B, T, d_obs) - observation features
            cat_ids: (B,) - embodiment category IDs
        Returns:
            updated_vlm: (B, T, d) - updated VLM embeddings
        """
        # NaN detection and prevention
        if torch.isnan(past_vlm).any() or torch.isinf(past_vlm).any():
            print("WARNING: past_vlm contains NaN/inf values! Returning original.")
            return past_vlm
            
        if torch.isnan(obs_features).any() or torch.isinf(obs_features).any():
            print("WARNING: obs_features contains NaN/inf values! Returning original past_vlm.")
            return past_vlm
        
        B, T, _ = past_vlm.shape
        
        # 1) Project obs to VLM space FIRST
        obs_projected = self.obs_projection(obs_features, cat_ids)  # (B, 1, vlm_dim)
        
        # 2) Expand obs to match past_vlm sequence length if needed
        if obs_projected.shape[1] == 1:
            obs_projected = obs_projected.expand(-1, T, -1)  # (B, T, vlm_dim)
        
        # 3) RMS normalization: past_u = past / rms(past)
        past_u = past_vlm / self.rms(past_vlm)
        
        # 4) RMS normalization: obs_u = obs / rms(obs)
        obs_u = obs_projected / self.rms(obs_projected)
        
        # 5) Get learnable blending weight α
        alpha = torch.sigmoid(self.alpha_logit[cat_ids])  # (B,) in [0,1]
        alpha = alpha.view(-1, 1, 1)  # (B, 1, 1) for broadcasting
        
        # 6) Blend: x = α*past_u + (1-α)*obs_u
        x = alpha * past_u + (1 - alpha) * obs_u
        
        # 7) Final normalization and scaling: vlm_t = x / rms(x) * target_rms
        x_normalized = x / self.rms(x)
        
        # Get target RMS for each category
        target_rms = self.target_rms[cat_ids].view(-1, 1, 1)  # (B, 1, 1)
        target_rms = torch.clamp(target_rms, 4.0, 200.0)
        
        # Final output
        vlm_t = x_normalized * target_rms
        
        # Debug prints (remove in production)
        print(f"[DEBUG] alpha: {alpha.squeeze().tolist()}")
        print(f"[DEBUG] target_rms: {target_rms.squeeze().tolist()}")
        print(f"[DEBUG] vlm_t range: [{vlm_t.min().item():.6f}, {vlm_t.max().item():.6f}]")
        
        return vlm_t