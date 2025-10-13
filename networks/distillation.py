from __future__ import annotations
from typing import Dict, Iterable, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional, List, Union, Literal
import warnings as Warnings

# ---------- Config ----------
class KDTarget(str, Enum):
    LOGITS = "logits"
    INTERMEDIATE = "intermediate"          # pair-wise (SKD)
    BACKBONE = "backbone"                  # MGD-style
    LOGITS_INTERMEDIATE = "logits+intermediate"
    ALL = "all"

@dataclass
class KDWeights:
    logits: float = 0.5
    intermediate: float = 0.5
    backbone: float = 1.0

# ---------- Individual loss implementations ----------

# -------------------------------------------------------------------------
# 1) Pixel-wise distillation (PX) — KD with temperature at the pixel level
#     (Hinton KD, applied per-pixel as in SKD pixel-wise baseline)
# -------------------------------------------------------------------------

def pixel_wise_distillation_loss(
    student_logits: torch.Tensor,    # [N, C, H, W]
    teacher_logits: torch.Tensor,    # [N, C, H, W]
    temperature: float = 1.0,
    mask: Optional[torch.Tensor] = None,   # [N, H, W] or [N,1,H,W]
    reduction: str = "mean",
) -> torch.Tensor:
    T = temperature

    # 1) Temperature-scaled probabilities
    s_logp = F.log_softmax(student_logits / T, dim=1)  # [N,C,H,W]
    t_prob = F.softmax(teacher_logits / T, dim=1)      # [N,C,H,W]

    # 2) Elementwise KL(q_t || p_s) per class
    kl_per_class = F.kl_div(s_logp, t_prob, reduction="none")  # [N,C,H,W]

    # 3) Sum over classes → per-pixel KL, then multiply by T^2
    kl_per_pixel = kl_per_class.sum(dim=1) * (T * T)   # [N,H,W]

    # 4) Optional spatial mask; otherwise mean/sum over all pixels & batch
    if mask is not None:
        # broadcast-safe mask: [N,H,W] or [N,1,H,W] → [N,H,W]
        if mask.dim() == 4 and mask.size(1) == 1:
            mask = mask.squeeze(1)
        mask = mask.float()
        if reduction == "mean":
            # mean over masked pixels only
            loss = (kl_per_pixel * mask).sum() / mask.sum().clamp_min(1.)
        else: # sum
            # sum over masked pixels only
            loss = (kl_per_pixel * mask).sum()
    else:
        loss = kl_per_pixel.mean() if reduction == "mean" else kl_per_pixel.sum()

    return loss


# -----------------------------------------------------------------------------------
# 2) Pair-wise distillation (PR) — match pairwise cosine similarities across pixels
#     Following SKD Sec. 3.1: a_ij = <fi, fj> / (||fi|| ||fj||), L2 between matrices
# -----------------------------------------------------------------------------------

def _pairwise_cosine_matrix(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    feat: [N, C, H, W] -> returns [N, HW, HW] cosine similarity matrices.
    """
    n, c, h, w = feat.shape
    x = feat.flatten(2)                                   # [N, C, HW]
    x = F.normalize(x, p=2, dim=1, eps=eps)              # L2 norm across channels
    sim = torch.bmm(x.transpose(1, 2), x)                # [N, HW, HW]
    return sim

def pair_wise_distillation_loss(
    student_features: Union[torch.Tensor, List[torch.Tensor]],
    teacher_features: Union[torch.Tensor, List[torch.Tensor]],
    target_hw_max: int = 4096,      # safety: cap HW for similarity matrix (e.g., <= 64x64)
    spatial_downsample: Optional[int] = None,  # e.g., 2/4 to reduce H,W before sim
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Computes the SKD pair-wise similarity loss. If lists are provided,
    it averages the loss across layers.
    - If H*W is large, we adaptively pool to keep HW <= target_hw_max.
    - spatial_downsample optionally avg-pools before computing similarities.
    """
    if isinstance(student_features, torch.Tensor):
        student_features = [student_features]
    if isinstance(teacher_features, torch.Tensor):
        teacher_features = [teacher_features]
    assert len(student_features) == len(teacher_features), "Mismatch in number of feature maps."

    losses = []
    for s_feat, t_feat in zip(student_features, teacher_features):
        # Optional spatial pooling
        if spatial_downsample and spatial_downsample > 1:
            Warnings.warn(f"Pair-wise distillation: downsampling spatially by {spatial_downsample}")
            s_feat = F.avg_pool2d(s_feat, kernel_size=spatial_downsample, stride=spatial_downsample)
            t_feat = F.avg_pool2d(t_feat, kernel_size=spatial_downsample, stride=spatial_downsample)

        # Ensure HW small enough by adaptive pooling
        n, _, h_s, w_s = s_feat.shape
        n2, _, h_t, w_t = t_feat.shape
        
        # Bring both to the same spatial size (use student’s by default)
        if (h_s != h_t) or (w_s != w_t):
            t_feat = F.adaptive_avg_pool2d(t_feat, (h_s, w_s))

        hw = h_s * w_s
        if hw > target_hw_max:
            # Try reduce both to (floor(sqrt(target)), floor(sqrt(target)))
            Warnings.warn(f"Pair-wise distillation: reducing HW={hw} to <= {target_hw_max}")
            side = int((target_hw_max) ** 0.5)
            s_feat = F.adaptive_avg_pool2d(s_feat, (side, side))
            t_feat = F.adaptive_avg_pool2d(t_feat, (side, side))

        As = _pairwise_cosine_matrix(s_feat)   # [N, HW, HW]
        At = _pairwise_cosine_matrix(t_feat)
        # L2 between matrices, normalized by (HW^2), averaged over batch
        diff = (As - At).pow(2)
        # normalize by matrix size (matches Eq. (2) normalization spirit)
        denom = diff.shape[1] * diff.shape[2]
        loss = diff.sum(dim=(1,2)) / float(denom)
        losses.append(loss.mean())

    total = torch.stack(losses).mean()
    return total if reduction == "mean" else total * len(losses)

# ---------------------------------------------------------------------------------------
# 3) Masked Generative Distillation (MGD) — adapters + projector trained with the student
#     Eq. (2)-(5): mask student feat spatially, adapt (1x1), then G=Conv3x3-ReLU-Conv3x3
# ---------------------------------------------------------------------------------------
class MGD(nn.Module):
    """
    One-stage MGD head for the last backbone feature.
    - adapter: 1x1 conv to align student channels to teacher channels
    - projector: 3x3 - ReLU - 3x3 to generate teacher-like features from masked student
    
    """
    def __init__(self, c_s: int, c_t: int):
        super().__init__()
        self.c_s = c_s # student channels
        self.c_t = c_t # teacher channels
        # 1) Adapter: align channels if needed
        self.adapter = nn.Conv2d(c_s, c_t, kernel_size=1, bias=True)
        # 2) Projector: conv layers to generate teacher-like features
        self.projector = nn.Sequential(
            nn.Conv2d(c_t, c_t, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_t, c_t, kernel_size=3, padding=1, bias=True),
        )

    @torch.no_grad()
    def _make_mask(self, shape, mask_ratio: float, device):
        N, Ct, H, W = shape
        keep_prob = 1.0 - mask_ratio
        # Binary spatial mask shared across channels: [N,1,H,W] → broadcast to Ct
        M = torch.bernoulli(torch.full((N, 1, H, W), keep_prob, device=device))
        return M

    def forward(self, student_feat_last: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        student_feat_last: [N, C_s, H, W]  (last backbone feature from student)
        returns generated feature \hat{T}: [N, C_t, H, W]
        """
        N, C_s, H, W = student_feat_last.shape
        # 1) Align channels to teacher space
        #TODO: check if the size of student and teacher features are same before applying adapter.
        if self.c_s == self.c_t:
            f_align = student_feat_last
        else:
            f_align = self.adapter(student_feat_last)                   # [N, C_t, H, W]
        # 2) Random spatial mask
        M = self._make_mask(f_align.shape, mask_ratio, f_align.device)  # [N,1,H,W]
        f_masked = f_align * M                                      # zero out masked positions
        # 3) Generate teacher-like feature
        f_gen = self.projector(f_masked)                            # [N, C_t, H, W]
        return f_gen

def masked_generation_distillation_loss(
    student_feat_last: torch.Tensor,     # [N, C_s, H_s, W_s]
    teacher_feat_last: torch.Tensor,     # [N, C_t, H_t, W_t]
    mgd: MGD,
    mask_ratio: float = 0.5,
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    """
    MGD loss on the LAST backbone feature only:
      L = mean_{n,c,h,w} ( G( A(S_last) ⊙ M ) - T_last )^2
    """
    assert reduction in ("mean", "sum")
    # Generate teacher-like features from masked, aligned student feature
    gen = mgd(student_feat_last, mask_ratio=mask_ratio)   # [N, C_t, H*, W*]

    # Match teacher spatial size if needed
    if gen.shape[-2:] != teacher_feat_last.shape[-2:]:
        teacher_feat_last = F.adaptive_avg_pool2d(teacher_feat_last, gen.shape[-2:])

    # Per-sample MSE over C,H,W
    per_sample = torch.mean((gen - teacher_feat_last) ** 2, dim=(1, 2, 3))  # [N]
    # per_sample = F.mse_loss(gen, teacher_feat_last, reduction="none").mean(dim=(1, 2, 3))  # [N]

    return per_sample.mean() if reduction == "mean" else per_sample.sum()


# ---------- Dispatcher ----------
LossFn = Callable[..., torch.Tensor]

def _targets_to_set(kd_points: str) -> set:
    kd_points = kd_points.lower()
    if kd_points in {KDTarget.ALL, "all"}:
        return {"logits", "intermediate", "backbone"}
    if kd_points in {KDTarget.LOGITS_INTERMEDIATE, "logits+intermediate", "logits_intermediate"}:
        return {"logits", "intermediate"}
    if kd_points in {KDTarget.LOGITS, "logits"}:
        return {"logits"}
    if kd_points in {KDTarget.INTERMEDIATE, "intermediate"}:
        return {"intermediate"}
    if kd_points in {KDTarget.BACKBONE, "backbone"}:
        return {"backbone"}
    raise ValueError(f"Unknown kd_points: {kd_points}")

def compute_kd_loss(
    *,
    kd_points: str,
    weights: KDWeights,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_features: Iterable[torch.Tensor],
    teacher_features: Iterable[torch.Tensor],
    backbone_pair: Tuple[torch.Tensor, torch.Tensor] | None = None,
    mgd_predictor: nn.Module | None = None,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    selected = _targets_to_set(kd_points)

    components: Dict[str, torch.Tensor] = {}
    if "logits" in selected:
        components["logits"] = pixel_wise_distillation_loss(
            student_logits, 
            teacher_logits,
            temperature=temperature,
            mask=None,
            reduction="mean",
        )
    
    if "intermediate" in selected:
        # components["intermediate"] = kd_pairwise_loss(student_features, teacher_features)
        components["intermediate"] = pair_wise_distillation_loss(
            student_features,
            teacher_features,
            target_hw_max=16384, #default value is 4096 (i.e., 64x64)
            spatial_downsample=None,
            reduction="mean",
        )

    if "backbone" in selected:
        assert backbone_pair is not None and mgd_predictor is not None, \
            "Provide backbone_pair=(student_feat, teacher_feat) and mgd_predictor for MGD"
        s_back, t_back = backbone_pair
        components["backbone"] = masked_generation_distillation_loss(
            s_back,
            t_back,
            mgd=mgd_predictor,
            mask_ratio=0.5,
            reduction="mean",
        )

    # weighted sum
    total = (
        weights.logits       * components.get("logits",       torch.tensor(0., device=student_logits.device)) +
        weights.intermediate * components.get("intermediate", torch.tensor(0., device=student_logits.device)) +
        weights.backbone     * components.get("backbone",     torch.tensor(0., device=student_logits.device))
    )
    
    # Report scalars for logging
    scalars = {k: float(v.detach().item()) for k, v in components.items()}
    scalars["total_kd"] = float(total.detach().item())
    return total, scalars
