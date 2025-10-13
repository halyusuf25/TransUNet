import torch
import torch.nn as nn
import math

class SHSAttention(nn.Module):
    def __init__(self, config, vis, alternate_partial_attn=False):
        super(SHSAttention, self).__init__()
        self.vis = vis
        self.hidden_size = config.hidden_size
        self.alternate_partial_attn = alternate_partial_attn

        # Calculate partial dimension (r = 1/4.67 from SHViT paper)
        self.pdim = int(self.hidden_size / 4.67) #as pare the SHViT paper default 
        self.qk_dim = 16  # Fixed per SHViT design
        
        # Normalization for partial channels (LayerNorm for 1D sequence)
        self.pre_norm = nn.LayerNorm(self.pdim)
        
        # Combined QKV projection for partial channels
        self.qkv = nn.Linear(self.pdim, 2*self.qk_dim + self.pdim)
        
        self.scale = self.qk_dim ** -0.5  # Scaling factor
        
        # Final projection (maintains dimension)
        self.out = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states):
        B, seq_len, _ = hidden_states.shape
        
        # Split into attended (pdim) and residual channels
        if self.alternate_partial_attn:
            x1 = hidden_states[..., -self.pdim:]  # (B, seq_len, pdim)
            x2 = hidden_states[..., :-self.pdim]   # (B, seq_len, hidden_size-pdim)
        else:
            x1 = hidden_states[..., :self.pdim]  # (B, seq_len, pdim)
            x2 = hidden_states[..., self.pdim:]   # (B, seq_len, hidden_size-pdim)

        # Normalize partial channels
        x1 = self.pre_norm(x1)  # (B, seq_len, pdim)
        
        # Generate Q, K, V from partial channels
        qkv = self.qkv(x1)  # (B, seq_len, 2*qk_dim + pdim)
        q, k, v = torch.split(qkv, [self.qk_dim, self.qk_dim, self.pdim], dim=-1)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, seq_len, seq_len)
        attn_probs = self.softmax(attn_scores)
        weights = attn_probs if self.vis else None
        attn_probs = self.attn_dropout(attn_probs)
        
        # Attend to values
        attended_x1 = torch.matmul(attn_probs, v)  # (B, seq_len, pdim)
        
        # Concatenate with residual channels
        if self.alternate_partial_attn:
            output = torch.cat([x2, attended_x1], dim=-1)
        else:
            output = torch.cat([attended_x1, x2], dim=-1)  # (B, seq_len, hidden_size)
        
        # Final projection
        output = self.out(output)
        output = self.proj_dropout(output)
        
        return output, weights
    
    
    

class TopkAttention(nn.Module):
    """Top-k token selection attention without assuming a CLS token.

    - Computes standard multi-head attention to get attended features `x_attn`.
    - Scores tokens by the average attention they receive (key centrality):
      mean over heads and queries of attn_prob to each key.
    - Selects top-k tokens globally from the full sequence (no CLS special case).
    - Returns both raw indices `idx` (B, K) and broadcasted `index` (B, K, C)
      for convenient `torch.gather` on the token dimension.
    """
    def __init__(self, config, vis, dim, qkv_bias=False, keep_rate=0.5):
        super().__init__()
        self.vis = vis
        self.num_heads = config.transformer["num_heads"]
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1.0, f"keep_rate must > 0 and <= 1.0, got {keep_rate}"

    def forward(self, x, keep_rate=None, tokens=None):
        # if keep_rate is None:
        #     keep_rate = self.keep_rate

        B, N, C = x.shape
        # QKV: [B, N, 3, H, C/H] -> [3, B, H, N, C/H]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention probabilities
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale            # [B, H, N, N]
        attn_prob = attn_logits.softmax(dim=-1)                          # [B, H, N, N]
        weights = attn_prob if self.vis else None
        attn = self.attn_drop(attn_prob)

        # Attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Default: keep all tokens
        remain_tokens = N

        # Prune only if keep_rate < 1 or an explicit token count is provided
        if (self.keep_rate < 1.0) or (tokens is not None):
            remain_tokens = math.ceil(self.keep_rate * N) if tokens is None else tokens
            # Clamp to valid range
            remain_tokens = max(1, min(remain_tokens, N))
            if remain_tokens == N:
                return x, None, None, None, remain_tokens, weights

            # CLS-free scoring: average attention received by each key across all queries and heads
            # attn_prob: [B, H, N_q, N_k] = [B, H, N, N]
            # 1) mean over heads -> [B, N, N]; 2) mean over queries -> [B, N]
            key_scores = attn_prob.mean(dim=1).mean(dim=1)  # [B, N]

            # Top-k token indices per batch (absolute in the current sequence)
            _, idx = torch.topk(key_scores, remain_tokens, dim=1, largest=True, sorted=True)  # [B, K]
            # Broadcast indices for torch.gather on x (B, N, C) along dim=1
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, K, C]
            return x, index, idx, key_scores, remain_tokens, weights

        return x, None, None, None, remain_tokens, weights
