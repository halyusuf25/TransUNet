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
        # weights = attn_prob if self.vis else None
        weights = attn_prob
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


class AdaptiveSpatialAttention(nn.Module):
    """Adaptive spatial attention that only attends to high-scoring tokens.

    The module compresses tokens with channel-wise statistics, predicts an
    attention mask with a light convolutional head, selects tokens whose scores
    exceed a threshold, applies self-attention on the selected subset, and
    leaves the remaining tokens untouched so the output shape stays (B, N, C).
    """

    def __init__(
        self,
        config,
        alpha=0.5,
        conv_kernel_size=1,
        use_bn=True,
        activation=nn.Sigmoid(),
        min_tokens=1, #minimum number of tokens to keep per sample
        vis=True,
    ):
        super().__init__()
        self.args = config
        # self.dim = dim if dim is not None else getattr(config, "hidden_size", None)
        self.dim = self.args.hidden_size
        if conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd to preserve length.")
        self.alpha = alpha
        self.min_tokens = max(1, min_tokens)
        self.vis = vis

        padding = conv_kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=1,
            kernel_size=conv_kernel_size,
            padding=padding,
            bias=not use_bn,
        )
        self.bn = nn.BatchNorm1d(1) if use_bn else None
        # self.activation = activation if activation is not None else nn.Sigmoid()
        self.activation = activation
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.args.transformer["num_heads"],
            dropout=self.args.transformer["attention_dropout_rate"],
            batch_first=True,
        )

    def _compute_scores(self, x):
        if self.args.verbose:
            print(f"[AdaptiveSpatialAttention] input x shape to _compute_scores: {x.shape}")
        x_tokens_first = x.transpose(1, 2)  # (B, C, N)
        avg_map = torch.mean(x_tokens_first, dim=1, keepdim=True)
        max_map, _ = torch.max(x_tokens_first, dim=1, keepdim=True)
        pooled = torch.cat([avg_map, max_map], dim=1)
        mask = self.conv(pooled)
        if self.bn is not None:
            mask = self.bn(mask)
        mask = self.activation(mask)
        if self.args.verbose:
            print(f"[AdaptiveSpatialAttention] mask shape: {mask.shape}")
        squeezed = mask.squeeze(1)  # (B, N)
        if self.args.verbose:
            print(f"[AdaptiveSpatialAttention] scores shape after squeeze: {squeezed.shape}")
        return squeezed

    def _select_indices(self, scores, threshold):
        B, N = scores.shape
        idx_list = []
        if self.args.verbose:
            print(f"[AdaptiveSpatialAttention] selecting indices from scores shape: {scores.shape}")
        for b in range(B):
            idx = torch.nonzero(scores[b] >= threshold, as_tuple=False).squeeze(-1)
            if idx.numel() < self.min_tokens:
                k = min(max(self.min_tokens, 1), N)
                idx = torch.topk(scores[b], k=k, dim=0).indices
            idx, _ = torch.sort(idx)
            idx_list.append(idx)
            if self.args.verbose:
                print(f"[AdaptiveSpatialAttention] batch {b} selected tokens: {idx.shape} (count={idx.numel()})")
        return idx_list

    def forward(self, x, alpha=None, return_indices=False):
        B, _, _ = x.shape
        if self.args.verbose:
            print(f"[AdaptiveSpatialAttention] forward input shape: {x.shape}")
        scores = self._compute_scores(x)
        threshold = self.alpha if alpha is None else alpha
        if isinstance(threshold, torch.Tensor):
            if threshold.numel() != 1:
                raise ValueError("alpha must be a float or 0-d tensor.")
            threshold = threshold.item()
        idx_list = self._select_indices(scores, threshold)
        if self.args.verbose:
            print(f"[AdaptiveSpatialAttention] idx_list length: {len(idx_list)}")

        updated = x.clone()
        attn_weights = []
        for b, idx in enumerate(idx_list):
            tokens = x[b : b + 1, idx, :]
            if self.args.verbose:
                print(f"[AdaptiveSpatialAttention] batch {b} token subset shape: {tokens.shape}")
            attn_out, weights = self.self_attn(tokens, tokens, tokens, need_weights=True)
            if self.args.verbose:
                print(f"[AdaptiveSpatialAttention] batch {b} attn_out shape: {attn_out.shape}")
            updated[b, idx, :] = attn_out.squeeze(0)
            if self.args.verbose:
                print(f"[AdaptiveSpatialAttention] batch {b} updated slice idx: {idx}")
                print(f"[AdaptiveSpatialAttention] batch {b} updated slice shape: {updated[b, idx, :].shape}")
            attn_weights.append(weights)
            if self.args.verbose:
                print(f"[AdaptiveSpatialAttention] batch {b} attention weights shape: {weights.shape}")

        idx_return = idx_list if return_indices else None
        if self.args.verbose:
            print(f"[AdaptiveSpatialAttention] forward output shape: {updated.shape}")
        # return updated, idx_return, attn_weights
        return updated, weights

