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
    
    def _compute_significance_score(
        self,
        A: torch.Tensor,  # (B, H, N, N) attention AFTER softmax
        V: torch.Tensor,  # (B, H, N, Dh)
        eps: float = 1e-12,
        v_norm_mode: str = "concat",  # "mean" or "concat"
    ) -> torch.Tensor:
        """
        Step B only (no CLS): compute normalized significance scores S over N tokens.

        v_norm_mode:
        - "mean":  n_{b,h,j} = ||V_{b,h,j}||_2, then w = mean_h(a*n)
        - "concat": n_{b,j}  = ||concat_h(V_{b,h,j})||_2, then w = a_bar * n

        Returns:
        S: (B, N) where S[b].sum() == 1
        """
        if A.dim() != 4 or V.dim() != 4:
            raise ValueError(f"Expected A and V to be 4D. Got A={A.shape}, V={V.shape}")
        if A.shape[:3] != V.shape[:3] or A.shape[2] != A.shape[3]:
            raise ValueError(
                f"Shape mismatch. A={A.shape} must be (B,H,N,N) and V={V.shape} must be (B,H,N,Dh)."
            )

        B, H, N, _ = A.shape
        Dh = V.shape[-1]

        # a_{b,h,j} = mean_q A_{b,h,q,j}  -> (B,H,N)
        a = A.mean(dim=2)

        if v_norm_mode == "mean":
            # n_{b,h,j} = ||V_{b,h,j}||_2 -> (B,H,N)
            n = torch.linalg.vector_norm(V, ord=2, dim=-1)

            # w_{b,h,j} = a * n -> (B,H,N)
            w = (a * n).clamp(min=0.0)

            # aggregate heads -> (B,N)
            w = w.mean(dim=1)

        elif v_norm_mode == "concat":
            # aggregate heads for attention weights -> (B,N)
            a_bar = a.mean(dim=1)

            # concat heads for values: (B,H,N,Dh) -> (B,N,H,Dh) -> (B,N,H*Dh)
            V_cat = V.permute(0, 2, 1, 3).reshape(B, N, H * Dh)

            # n_{b,j} = ||concat_h(V_{b,h,j})||_2 -> (B,N)
            n_cat = torch.linalg.vector_norm(V_cat, ord=2, dim=-1)

            # w_{b,j} = a_bar * n_cat -> (B,N)
            w = (a_bar * n_cat).clamp(min=0.0)

        else:
            raise ValueError("v_norm_mode must be either 'mean' or 'concat'")

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


class ATSAttention(nn.Module):

    def __init__(
        self,
        config,
        embed_dim: int,
        keep_rate: float = None,
        K: int = None,
        min_tokens: int = 10,
        qkv_bias: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()

        self.args = config
        self.num_heads = int(config.transformer["num_heads"])
        if embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({self.num_heads}).")

        if keep_rate is None and K is None:
            raise ValueError("At least one of keep_rate or K must be specified.")
        if keep_rate is not None and not (0.0 < float(keep_rate) <= 1.0):
            raise ValueError(f"keep_rate must be in (0, 1], got {keep_rate}.")
        if K is not None and int(K) <= 0:
            raise ValueError(f"K must be a positive integer, got {K}.")

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.keep_rate = float(keep_rate) if keep_rate is not None else None
        self.K = int(K) if K is not None else None
        self.min_tokens = max(10, int(min_tokens))
        self.eps = float(eps)

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_drop = nn.Dropout(config.transformer["attention_dropout_rate"])

    def _num_tokens_to_keep(self, N: int) -> int:
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}.")

        K = self.K
        keep_rate = self.keep_rate

        if K is not None:
            k_keep = int(K)
        elif keep_rate is not None:
            if not (0.0 < float(keep_rate) <= 1.0):
                raise ValueError(f"keep_rate must be in (0, 1], got {keep_rate}.")
            k_keep = int(math.floor(float(keep_rate) * N + 0.5))
        else:
            raise ValueError("Either K or keep_rate must be specified.")
        
        k_keep = max(self.min_tokens, k_keep)
        k_keep = min(N, k_keep)
        return k_keep

    def _normalize_scores(self, r: torch.Tensor) -> torch.Tensor:
        if r.dim() != 2:
            raise ValueError(f"Expected r to be [B, N], got {tuple(r.shape)}.")

        B, N = r.shape
        if N <= 0:
            raise ValueError("Token dimension N must be > 0.")

        r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        sum_r = r.sum(dim=-1, keepdim=True)
        S = r / sum_r.clamp(min=self.eps)
        S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

        sum_s = S.sum(dim=-1, keepdim=True)
        degenerate = (~torch.isfinite(sum_r)) | (sum_r <= self.eps) | (~torch.isfinite(sum_s)) | (sum_s <= self.eps)
        if degenerate.any():
            uniform = torch.full_like(S, 1.0 / float(N))
            S = torch.where(degenerate.expand_as(S), uniform, S)

        S = S / S.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        return S

    def score_assignment_step(self, attn: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        No-CLS ATS adaptation.
        A: [B, H, N, N], V: [B, H, N, Dh] -> S: [B, N]
        """
        if attn.dim() != 4 or v.dim() != 4:
            raise ValueError(f"Expected attn and v to be 4D. Got attn={attn.shape}, v={v.shape}")
        if attn.shape[:3] != v.shape[:3] or attn.shape[2] != attn.shape[3]:
            raise ValueError(f"Shape mismatch. attn={attn.shape}, v={v.shape}")

        a_bar = attn.mean(dim=2)  # [B, H, N]
        m = torch.linalg.vector_norm(v, ord=2, dim=-1)  # [B, H, N]
        r = (a_bar * m).sum(dim=1)  # [B, N]
        return self._normalize_scores(r)

    def inverse_transform_sampling(self, scores: torch.Tensor, K: int):
        """
        Deterministic inverse transform sampling on the CDF of scores.
        Returns padded unique indices and validity mask:
        - sampled_idx: [B, K_max] (torch.long)
        - sampled_mask: [B, K_max] (bool), True for valid sampled entries
        """
        if scores.dim() != 2:
            raise ValueError(f"Expected scores to be [B, N], got {tuple(scores.shape)}.")
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}.")

        B, N = scores.shape
        if N <= 0:
            raise ValueError("Token dimension N must be > 0.")

        if K >= N:
            idx = torch.arange(N, device=scores.device, dtype=torch.long).unsqueeze(0).expand(B, N)
            mask = torch.ones(B, N, device=scores.device, dtype=torch.bool)
            return idx, mask

        scores = self._normalize_scores(scores)
        cdf = torch.cumsum(scores, dim=-1).clamp_(0.0, 1.0)
        cdf[:, -1] = 1.0

        steps = torch.arange(1, K + 1, device=scores.device, dtype=scores.dtype)
        u = ((2.0 * steps) - 1.0) / (2.0 * K)
        u = u.unsqueeze(0).expand(B, K).contiguous()

        sampled = torch.searchsorted(cdf, u, right=False).clamp_(min=0, max=N - 1).long()

        unique_indices = [torch.unique(sampled[b], sorted=True) for b in range(B)]
        max_k = max(idx.numel() for idx in unique_indices)
        max_k = max(1, max_k)

        sampled_idx = torch.zeros(B, max_k, device=scores.device, dtype=torch.long)
        sampled_mask = torch.zeros(B, max_k, device=scores.device, dtype=torch.bool)
        for b, idx in enumerate(unique_indices):
            if idx.numel() == 0:
                sampled_idx[b, 0] = 0
                sampled_mask[b, 0] = True
                continue
            sampled_idx[b, : idx.numel()] = idx
            sampled_mask[b, : idx.numel()] = True

        return sampled_idx, sampled_mask

    @staticmethod
    def _gather_queries(attn: torch.Tensor, sampled_idx: torch.Tensor) -> torch.Tensor:
        # attn: [B, H, N, N], sampled_idx: [B, K_max] -> [B, H, K_max, N]
        B, H, _, N = attn.shape
        k_max = sampled_idx.shape[1]
        gather_index = sampled_idx[:, None, :, None].expand(B, H, k_max, N)
        return torch.gather(attn, dim=2, index=gather_index)

    def forward(
        self,
        x: torch.Tensor,
        return_indices: bool = False,
    ):
        if x.dim() != 3:
            raise ValueError(f"Expected x to be [B, N, D], got {tuple(x.shape)}.")

        B, N, D = x.shape
        if D != self.embed_dim:
            raise ValueError(f"Expected embedding dim D={self.embed_dim}, got {D}.")

        k_keep = self._num_tokens_to_keep(N)
        # qkv: [B, N, 3D] -> [3, B, H, N, Dh]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.args.verbose:
            print(f"[ATSAttention] Input x shape: {x.shape}, keeping top {k_keep} tokens per sample.")
            print(f"[ATSAttention] q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # [B, H, N, N]

        if k_keep >= N:
            if self.args.verbose:
                print(f"[ATSAttention] keep_rate results in keeping all tokens (k_keep={k_keep} >= N={N}). Using full attention.")
            full_attn = self.attn_drop(attn)
            y = full_attn @ v
            y = y.transpose(1, 2).contiguous().reshape(B, N, D)
            y = self.proj_drop(self.proj(y))

            idx = torch.arange(N, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, N)
            mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
            if return_indices:
                # return y, full_attn, idx, mask
                return y, full_attn, idx
            return y, full_attn

        scores = self.score_assignment_step(attn, v)  # [B, N]
        sampled_idx, sampled_mask = self.inverse_transform_sampling(scores, k_keep)

        attn = self.attn_drop(attn)
        attn_s = self._gather_queries(attn, sampled_idx)  # [B, H, K_max, N]
        attn_s = attn_s * sampled_mask[:, None, :, None].to(attn_s.dtype)

        y = attn_s @ v  # [B, H, K_max, Dh]
        y = y.transpose(1, 2).contiguous().reshape(B, sampled_idx.shape[1], D)
        y = self.proj_drop(self.proj(y))
        y = y * sampled_mask.unsqueeze(-1).to(y.dtype)

        if return_indices:
            # return y, attn_s, sampled_idx, sampled_mask
            return y, attn_s, sampled_idx
        return y, attn_s


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
