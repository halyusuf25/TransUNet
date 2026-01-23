import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.ndimage import distance_transform_edt as _scipy_distance_transform_edt
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss


class BULoss(nn.Module):
    """Boundary/Uncertainty-Aware loss (loss_function.tex, Sec. 1.1-1.3)."""

    def __init__(
        self,
        loss_option: str = "C",
        #OPTION A : \mathcal{L}_{total} = \mathcal{L}_{Dice}^w+ \mathcal{L}_{CE}^w
        #OPTION B : \mathcal{L}_{total} = \mathcal{L}_{Dice}+ \mathcal{L}_{CE}^w
        #OPTION C : \mathcal{L}_{total} = \mathcal{L}_{Dice}+ \mathcal{L}_{CE}
        distance_type: str = "unsigned",
        tau: float = 1.0,
        bm_min: float = 0.2,
        bm_max: float = 3.0,
        alpha: float = 10.0,
        eps: float = 1e-6,
        background_class: int = 0,
        cdist_chunk_size: int = 4096,
        mu_dice: float = 0.5,
        mu_ce: float = 0.5,
        args = None,
    ) -> None:
        super().__init__()
        loss_option = args.buloss_option
        loss_option = loss_option.upper()
        if loss_option not in {"A", "B", "C"}:
            raise ValueError(
                f"loss_option must be 'A', 'B', or 'C', got {loss_option!r}"
            )
        distance_type = args.distance_map_type
        distance_type = distance_type.lower()
        if distance_type not in {"dtm", "unsigned", "signed"}:
            raise ValueError(
                "distance_type must be one of {'dtm','unsigned','signed'}, "
                f"got {distance_type!r}"
            )
        if tau <= 0:
            raise ValueError("tau must be > 0.")

        self.args = args
        self.loss_option = loss_option
        self.distance_type = distance_type 
        print(f"Using BULoss with option {self.loss_option}, distance type {self.distance_type}")
        self.tau = float(self.args.tau) if self.args.tau is not None else float(tau)
        self.bm_min = float(self.args.bm_min) if self.args.bm_min is not None else float(bm_min)
        self.bm_max = float(self.args.bm_max) if self.args.bm_max is not None else float(bm_max)
        self.alpha = float(self.args.alpha) if self.args.alpha is not None else float(alpha)
        self.eps = float(eps)
        self.background_class = int(background_class)
        self.cdist_chunk_size = int(cdist_chunk_size)
        self.mu_dice = float(mu_dice)
        self.mu_ce = float(mu_ce)
        
        self._ce_loss_function = CrossEntropyLoss()
        self._dice_loss_function = DiceLoss(args.num_classes)
        

        # 3x3 kernel for binary erosion when estimating the boundary (Sec. 1.2).
        self.register_buffer("_boundary_kernel", torch.ones(1, 1, 3, 3))

    def compute_uncertainty_map(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute UM(x)=U(x) from entropy (loss_function.tex, Sec. 1.1.1-1.1.2)."""
        if probs.dim() != 4:
            raise ValueError("probs must be [B, C, H, W]")
        num_classes = probs.size(1)
        log_base = math.log(float(max(num_classes, 2)))
        # eps = 1e-8
        # probs = probs.clamp(min=eps)
        # entropy = -(probs * torch.log(probs)).sum(dim=1, keepdim=True)
        # probs: [B,C,H,W], already softmaxed
        entropy = -torch.xlogy(probs, probs).sum(dim=1, keepdim=True)  # safe 0*log(0)=0
        return entropy / log_base

    def compute_distance_map(self, target: torch.Tensor) -> torch.Tensor:
        """Compute distance map D (loss_function.tex, Sec. 1.2)."""
        if target.dim() != 3:
            raise ValueError("target must be [B, H, W]")
        target = target.long()
        with torch.no_grad():
            foreground = target != self.background_class
            boundary = self._compute_boundary_mask(foreground)
            # if _scipy_distance_transform_edt is not None and np is not None:
            #     distance = self._distance_transform_scipy(boundary)
            # else:
            #     distance = self._distance_transform_torch(boundary)
            distance = self._distance_transform_scipy(boundary)

            if self.distance_type == "dtm":
                distance = distance * foreground.float()
            elif self.distance_type == "signed":
                distance = torch.where(foreground, -distance, distance)

        return distance.unsqueeze(1)

    def compute_boundary_map(self, distance: torch.Tensor) -> torch.Tensor:
        """Compute BM(x) from D(x) (loss_function.tex, Eq. (bm_long))."""
        bm = torch.exp(-distance / self.tau)
        bm_mid = torch.where(bm < self.bm_min, torch.zeros_like(bm), bm)
        bm_mid = torch.where(bm >= self.bm_max, torch.full_like(bm, self.alpha), bm_mid)
        return bm_mid

    def compute_weights(
        self,
        boundary_map: torch.Tensor,
        uncertainty_map: torch.Tensor,
        # option_specific_masking: bool = False,
    ) -> torch.Tensor:
        """Compute w(x)=exp(BM(x)*UM(x))"""
        # if option_specific_masking:
        #     # Reserved hook; no extra masking in the current spec.
        #     pass
        return torch.exp(boundary_map * uncertainty_map)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        return_details: bool = False,
    ):
        """Return total loss and (optionally) diagnostics."""
        if logits.dim() != 4:
            raise ValueError("logits must be [B, C, H, W]")
        if target.dim() != 3:
            raise ValueError("target must be [B, H, W]")

        
        if self.loss_option == "C":
            # Standard Dice + CE loss without boundary/uncertainty weighting.
            loss_ce = self._ce_loss_function(logits, target)
            loss_dice = self._dice_loss_function(logits, target, softmax=True)
            total = self.mu_dice * loss_dice + self.mu_ce * loss_ce
            if not return_details:
                return total
            details: Dict[str, torch.Tensor] = {
                "L_CE": loss_ce.detach(),
                "L_Dice": loss_dice.detach(),
            }
            return total, details
        elif self.loss_option == "B":
            loss_dice = self._dice_loss_function(logits, target, softmax=True)
            
        
        probs = F.softmax(logits, dim=1)
        uncertainty_map = self.compute_uncertainty_map(probs)
        distance_map = self.compute_distance_map(target)
        boundary_map = self.compute_boundary_map(distance_map)
        weights = self.compute_weights(boundary_map, uncertainty_map)

        # log_probs = F.log_softmax(logits, dim=1)
        # ce_per_pixel = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        

        loss_wce = self._weighted_ce(logits, target, weights)

        loss_wdice = self._dice_loss(probs, target, weights=weights)

        if self.loss_option == "A":
            total = self.mu_dice * loss_wdice + self.mu_ce * loss_wce
        elif self.loss_option == "B":
            total = self.mu_dice * loss_dice + self.mu_ce * loss_wce

        if not return_details:
            return total

        details: Dict[str, torch.Tensor] = {
            "UM_mean": uncertainty_map.mean().detach(),
            "BM_mean": boundary_map.mean().detach(),
            "w_mean": weights.mean().detach(),
            "L_wCE": loss_wce.detach(),
            "L_Dice": loss_dice.detach(),
            "L_wDice": loss_wdice.detach(),
        }
        return total, details

    def _weighted_ce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted CE loss (loss_function.tex, Sec. 1.3.2)."""
        ce_per_pixel = F.cross_entropy(logits, target, reduction="none")
        weights_spatial = weights.squeeze(1)
        weight_sum = weights_spatial.sum(dim=(1, 2)).clamp_min(self.eps)
        loss_wce = (weights_spatial * ce_per_pixel).sum(dim=(1, 2)) / weight_sum
        return loss_wce.mean()

    def _dice_loss(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Dice loss, weighted if weights are provided (Sec. 1.3.3)."""
        if probs.dim() != 4:
            raise ValueError("probs must be [B, C, H, W]")
        target = target.long()
        num_classes = probs.size(1)
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)
        target_onehot = target_onehot.to(dtype=probs.dtype, device=probs.device)

        if weights is None:
            weights = torch.ones(
                (probs.size(0), 1, probs.size(2), probs.size(3)),
                device=probs.device,
                dtype=probs.dtype,
            )
        else:
            if weights.dim() == 3:
                weights = weights.unsqueeze(1)
            weights = weights.to(dtype=probs.dtype, device=probs.device)

        smooth = 1e-5
        dims = (0, 2, 3)
        intersection = (weights * probs * target_onehot).sum(dim=dims)
        y_sum = (weights * target_onehot * target_onehot).sum(dim=dims)
        z_sum = (weights * probs * probs).sum(dim=dims)
        dice = (2.0 * intersection + smooth) / (z_sum + y_sum + smooth)
        return (1.0 - dice).mean()

    def _compute_boundary_mask(self, foreground: torch.Tensor) -> torch.Tensor:
        """Approximate boundary B from the foreground mask (loss_function.tex, Sec. 1.2)."""
        if foreground.dim() == 3:
            foreground = foreground.unsqueeze(1)
        foreground = foreground.to(dtype=self._boundary_kernel.dtype)
        kernel = self._boundary_kernel.to(device=foreground.device)
        neighbor_count = F.conv2d(foreground, kernel, padding=1)
        eroded = neighbor_count == kernel.numel()
        boundary = foreground.bool() & ~eroded
        return boundary.squeeze(1)

    def _distance_transform_scipy(self, boundary: torch.Tensor) -> torch.Tensor:
        """Distance transform via SciPy (preferred when available)."""
        distance_maps = []
        for b in range(boundary.size(0)):
            bmask = boundary[b].detach().cpu().numpy().astype(np.uint8)
            if bmask.sum() == 0:
                dist = np.zeros_like(bmask, dtype=np.float32)
            else:
                dist = _scipy_distance_transform_edt(bmask == 0).astype(np.float32)
            distance_maps.append(torch.from_numpy(dist))
        return torch.stack(distance_maps, dim=0).to(device=boundary.device)

    def _distance_transform_torch(self, boundary: torch.Tensor) -> torch.Tensor:
        """Torch fallback distance transform (exact but slower; Sec. 1.2)."""
        batch, height, width = boundary.shape
        device = boundary.device
        dtype = torch.float32

        ys = torch.arange(height, device=device, dtype=dtype)
        xs = torch.arange(width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).view(-1, 2)

        distance_maps = []
        for b in range(batch):
            bmask = boundary[b].view(-1)
            if not bmask.any():
                distance_maps.append(torch.zeros((height, width), device=device, dtype=dtype))
                continue
            boundary_coords = coords[bmask]
            min_dist = self._min_cdist(coords, boundary_coords)
            distance_maps.append(min_dist.view(height, width))

        return torch.stack(distance_maps, dim=0)

    def _min_cdist(self, coords: torch.Tensor, boundary_coords: torch.Tensor) -> torch.Tensor:
        """Compute min Euclidean distance from coords to boundary_coords (chunked)."""
        if boundary_coords.numel() == 0:
            return torch.zeros(coords.size(0), device=coords.device, dtype=coords.dtype)
        if boundary_coords.size(0) == 1:
            return torch.norm(coords - boundary_coords[0], dim=1)

        chunk = max(1, self.cdist_chunk_size)
        min_dist = torch.full(
            (coords.size(0),), float("inf"), device=coords.device, dtype=coords.dtype
        )
        for start in range(0, coords.size(0), chunk):
            chunk_coords = coords[start : start + chunk]
            dist = torch.cdist(chunk_coords, boundary_coords)
            min_dist[start : start + chunk] = dist.min(dim=1).values
        return min_dist
