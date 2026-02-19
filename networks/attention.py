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

    def __init__(
        self,
        config,
        embed_dim: int,
        keep_rate: float = 0.5,
        min_tokens: int = 10,
        qkv_bias: bool = True,
    ):
        super().__init__()
        
        self.args = config
        self.num_heads = int(config.transformer["num_heads"])
        if embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({self.num_heads}).")

        if keep_rate is None:
            keep_rate = float(getattr(config, "topk_attn", 1.0))
        if not (0.0 < keep_rate <= 1.0):
            raise ValueError(f"keep_rate must be in (0, 1], got {keep_rate}.")

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.keep_rate = float(keep_rate)
        self.min_tokens = max(1, int(min_tokens))

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_drop = nn.Dropout(config.transformer["attention_dropout_rate"])

    def _compute_significance_score(self,
        A: torch.Tensor,  # (B, H, N, N)  attention AFTER softmax
        V: torch.Tensor,  # (B, H, N, Dh)
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Step B only (no CLS): compute normalized significance scores S over N tokens.

        Returns:
        S: (B, N) where S[b].sum() == 1
        """
        if A.dim() != 4 or V.dim() != 4:
            raise ValueError(f"Expected A and V to be 4D. Got A={A.shape}, V={V.shape}")
        if A.shape[:3] != V.shape[:3] or A.shape[2] != A.shape[3]:
            raise ValueError(f"Shape mismatch. A={A.shape} must be (B,H,N,N) and V={V.shape} must be (B,H,N,Dh).")

        # a_{b,h,j} = mean_q A_{b,h,q,j}  -> (B,H,N)
        a = A.mean(dim=2)

        # n_{b,h,j} = ||V_{b,h,j}||_2 -> (B,H,N)
        n = torch.linalg.vector_norm(V, ord=2, dim=-1)

        # w_{b,h,j} = a * n  -> (B,H,N)
        w = (a * n).clamp(min=0.0)

        # aggregate heads -> (B,N)
        w = w.mean(dim=1)

        # normalize over tokens -> (B,N)
        Z = w.sum(dim=-1, keepdim=True).clamp(min=eps)
        S = w / Z
        return S



    def _num_tokens_to_keep(self, N: int) -> int:
        # keep at least one token and enforce a configurable lower bound.
        k_from_rate = int(math.ceil(self.keep_rate * N))
        return min(N, max(self.min_tokens, k_from_rate))

    @staticmethod
    def _gather_topk_queries(attn: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        # attn: [B, H, N, N], topk_idx: [B, k] -> gathered_attn: [B, H, k, N]
        B, H, _, N = attn.shape
        k = topk_idx.shape[1]
        gather_index = topk_idx[:, None, :, None].expand(B, H, k, N)
        return torch.gather(attn, dim=2, index=gather_index)

    def _gumbel_topk(self, x: torch.Tensor, K: int = 8) -> torch.Tensor:
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}.")
        if K > x.shape[-1]:
            raise ValueError(f"K ({K}) cannot exceed x dimension ({x.shape[-1]}).")

        loc =torch.zeros_like(x, dtype=torch.float32)
        scale = torch.ones_like(x, dtype=torch.float32)
        gumbel = torch.distributions.Gumbel(loc, scale)
        scores = torch.log(x) + gumbel.sample().to(dtype=x.dtype)
        return scores.topk(K, dim=-1, largest=True, sorted=True).indices

    def forward(self, x: torch.Tensor, return_indices: bool = False):

        if x.dim() != 3:
            raise ValueError(f"Expected x to be [B, N, D], got {tuple(x.shape)}")

        B, N, D = x.shape
        if D != self.embed_dim:
            raise ValueError(f"Expected embedding dim D={self.embed_dim}, got {D}.")

        # qkv: [B, N, 3D] -> [3, B, H, N, Dh], where Dh = D / H.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: [B, H, N, Dh]

        # Full attention map over all query-key token pairs: [B, H, N, N].
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if attn.shape != (B, self.num_heads, N, N):
            raise RuntimeError(f"Expected attn shape {(B, self.num_heads, N, N)}, got {tuple(attn.shape)}.")

        # Token importance score from attention, aggregated over heads and queries: [B, N].
        token_score = attn.mean(dim=1).mean(dim=1)
        if token_score.shape != (B, N):
            raise RuntimeError(f"Expected token_score shape {(B, N)}, got {tuple(token_score.shape)}.")

        # Select top-k query tokens per sample.
        k_keep = self._num_tokens_to_keep(N)
        if self.args.use_gumbel_topk:
            normalized_token_scores = self._compute_significance_score(attn, v)  # (B, N)
            topk_idx = self._gumbel_topk(normalized_token_scores, K=k_keep)
        else:
            topk_idx = token_score.topk(k_keep, dim=-1, largest=True, sorted=True).indices  # [B, k]

        # Reduce attention map on query dimension only: [B, H, N, N] -> [B, H, k, N].
        topk_attn = self._gather_topk_queries(attn, topk_idx)
        if topk_attn.shape != (B, self.num_heads, k_keep, N):
            raise RuntimeError(
                f"Expected reduced attn shape {(B, self.num_heads, k_keep, N)}, got {tuple(topk_attn.shape)}."
            )

        # Apply reduced attention to all value tokens: [B, H, k, N] @ [B, H, N, Dh] -> [B, H, k, Dh].
        y = topk_attn @ v

        # Merge heads back: [B, H, k, Dh] -> [B, k, D], then output projection.
        y = y.transpose(1, 2).contiguous().reshape(B, k_keep, D)
        y = self.proj_drop(self.proj(y))

        if return_indices:
            return y, topk_attn, topk_idx
        return y, topk_attn


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
