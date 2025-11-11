import torch

def topk_indices(x: torch.Tensor, attention_scores: torch.Tensor, keep_rate: float):
    """
    Select top-k token indices per batch based on keep_rate and attention_scores.

    Args:
        x: Tensor of shape [B, N, C] input features
        keep_rate: Fraction of tokens to keep (0 < keep_rate <= 1)
        attention_scores: Tensor of shape [B, H, N_q, N_k] = [B, H, N, N]

    Returns:
        idx: Tensor [B, K] of top-k indices where K = keep_rate * N
        index: Tensor [B, K, C] ready for torch.gather on dim=1 of x
    """
    B , N, C = x.shape
    attention_scores = attention_scores.mean(dim=1).mean(dim=1)  # [B, N]
    k = max(1, int(keep_rate * N))
    _ , idx = torch.topk(attention_scores, k, dim=1, largest=True, sorted=True)  # [B, K]
    index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, K, C]
    return idx, index