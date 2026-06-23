import math
from typing import Dict

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
        loss_option: str = "B",
        #OPTION A : \mathcal{L}_{total} = (1-lambda) * \mathcal{L}_{Dice}^w + lambda * \mathcal{L}_{CE}^w
        #OPTION B : \mathcal{L}_{total} = (1-lambda) * \mathcal{L}_{Dice} + lambda * \mathcal{L}_{CE}^w
        #OPTION C : \mathcal{L}_{total} = (1-lambda) * \mathcal{L}_{Dice} + lambda * \mathcal{L}_{CE}
        distance_type: str = "unsigned",
        tau_min: float = 1e-3,
        alpha: float = 1.0,
        eps: float = 1e-8,
        background_class: int = 0,
        cdist_chunk_size: int = 4096,
        args = None,
    ) -> None:
        super().__init__()
        self.args = args

        self.loss_option = self.args.buloss_option
        self.distance_type = self.args.distance_map_type
        print(f"Using BULoss with option {self.loss_option}, distance type {self.distance_type}")

        self.learn_tau = self.args.learn_tau
        self.tau_min = self.args.tau_min

        if self.learn_tau:
            self.tau_init = self.args.tau_init
            tau_delta = torch.tensor(self.tau_init - self.tau_min, dtype=torch.float32)
            rho0 = tau_delta + torch.log(-torch.expm1(-tau_delta))
            self.rho = nn.Parameter(rho0.clone().detach())
            if getattr(self.args, "verbose", False):
                self.rho.register_hook(
                    lambda grad: self._verbose_tensor_stats("rho.grad", grad)
                )
        else:
            if self.args.tau <= 0:
                raise ValueError("tau must be > 0 when learn_tau=False.")
            self.tau_init = None
            # Use provided fixed tau from args
            self.register_buffer("_tau_fixed", torch.tensor(self.args.tau, dtype=torch.float32))

        if getattr(self.args, "verbose", False) and self.learn_tau and self.loss_option in {"A", "B"}:
            print(f"[BULoss] Initialized with tau={self.tau_init}, learn_tau={self.learn_tau}, tau_min={self.tau_min}")
            print(f"rho initial value: {rho0.item()} (corresponding to tau={self.tau_init})")

        self.alpha = self.args.alpha

        self.boundary_radius = self.args.boundary_radius
        self.lambda_ = float(getattr(self.args, "lambda_", 0.5))
        self.eps = float(eps)

        self._ce_loss_function = CrossEntropyLoss()
        num_classes = self.args.num_classes
        self._dice_loss_function = DiceLoss(num_classes)

    def get_tau(self) -> torch.Tensor:
        if self.learn_tau:
            return self.tau_min + F.softplus(self.rho)
        return self._tau_fixed

    def _is_verbose(self) -> bool:
        return bool(getattr(self.args, "verbose", False))

    def _verbose_tensor_stats(self, name: str, tensor: torch.Tensor) -> None:
        tensor_detached = tensor.detach()
        total_count = tensor_detached.numel()
        if tensor_detached.is_floating_point() or tensor_detached.is_complex():
            finite_mask = torch.isfinite(tensor_detached)
            nan_count = int(torch.isnan(tensor_detached).sum().item())
            posinf_count = int(torch.isposinf(tensor_detached).sum().item())
            neginf_count = int(torch.isneginf(tensor_detached).sum().item())
        else:
            finite_mask = torch.ones_like(tensor_detached, dtype=torch.bool)
            nan_count = 0
            posinf_count = 0
            neginf_count = 0
        finite_count = int(finite_mask.sum().item())

        if finite_count > 0:
            finite_values = tensor_detached[finite_mask].float()
            min_value = float(finite_values.min().item())
            max_value = float(finite_values.max().item())
            mean_value = float(finite_values.mean().item())
        else:
            min_value = float("nan")
            max_value = float("nan")
            mean_value = float("nan")

        print(
            f"[BULoss][debug] {name}: shape={tuple(tensor_detached.shape)} "
            f"finite={finite_count}/{total_count} nan={nan_count} "
            f"+inf={posinf_count} -inf={neginf_count} "
            f"min={min_value:.6g} max={max_value:.6g} mean={mean_value:.6g}"
        )

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
            boundary = self._compute_boundary_mask(target)
            distance = self._distance_transform_scipy(boundary)
            if self._is_verbose():
                boundary_counts = boundary.flatten(1).sum(dim=1)
                no_boundary_count = int((boundary_counts == 0).sum().item())
                print(
                    f"[BULoss][debug] boundary pixels per sample: "
                    f"{boundary_counts.detach().cpu().tolist()} | "
                    f"no_boundary_samples={no_boundary_count}/{boundary.size(0)}"
                )
                self._verbose_tensor_stats("boundary_mask", boundary)
                self._verbose_tensor_stats("distance_map_raw", distance)

        return distance.unsqueeze(1)

    def compute_boundary_map(self, distance: torch.Tensor) -> torch.Tensor:
        """Convert the distance map D(x) into a boundary map BM(x).

        Uses an exponential decay based on the distance to the nearest boundary, scaled by
        tau. Invalid or infinite distances are kept masked out and remain zero in the output.
        """
        tau = self.get_tau().to(device=distance.device, dtype=distance.dtype)
        finite_distance = torch.isfinite(distance)
        safe_distance = distance.masked_fill(~finite_distance, 0.0)
        boundary_map = torch.exp(-safe_distance / tau)
        return boundary_map.masked_fill(~finite_distance, 0.0)

    def compute_weights(
        self,
        boundary_map: torch.Tensor,
        uncertainty_map: torch.Tensor,
        # option_specific_masking: bool = False,
    ) -> torch.Tensor:
        """Compute w(x)=exp(alpha*BM(x)*UM(x))."""
        return torch.exp(self.alpha * boundary_map * uncertainty_map)

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
        if getattr(self.args, "verbose", False):
            print(f"[BULoss] tau={float(self.get_tau().detach())}")
            self._verbose_tensor_stats("logits", logits)
            self._verbose_tensor_stats("target", target)
            self._verbose_tensor_stats("tau", self.get_tau())
            if self.learn_tau:
                self._verbose_tensor_stats("rho", self.rho)
        
        target = target.to(device=logits.device, dtype=torch.long)
        
        if self.loss_option == "C":
            # Standard Dice + CE loss without boundary/uncertainty weighting.
            loss_ce = self._ce_loss_function(logits, target)
            loss_dice = self._dice_loss_function(logits, target, softmax=True)
            total = (
                (1.0 - self.lambda_) * loss_dice + self.lambda_ * loss_ce
            )
            if self._is_verbose():
                self._verbose_tensor_stats("loss_ce", loss_ce)
                self._verbose_tensor_stats("loss_dice", loss_dice)
                self._verbose_tensor_stats("loss_total", total)
            if not return_details:
                return total
            details: Dict[str, torch.Tensor] = {
                "L_CE": loss_ce.detach(),
                "L_Dice": loss_dice.detach(),
            }
            return total, details

        probs = F.softmax(logits, dim=1)
        uncertainty_map = self.compute_uncertainty_map(probs)
        distance_map = self.compute_distance_map(target)
        boundary_map = self.compute_boundary_map(distance_map)
        weights = self.compute_weights(boundary_map, uncertainty_map)
        if self._is_verbose():
            self._verbose_tensor_stats("probs", probs)
            self._verbose_tensor_stats("uncertainty_map", uncertainty_map)
            self._verbose_tensor_stats("distance_map", distance_map)
            self._verbose_tensor_stats("boundary_map", boundary_map)
            self._verbose_tensor_stats("weights", weights)

        loss_wce = self._weighted_ce(logits, target, weights)

        if self.loss_option == "A":
            loss_wdice = self._weighted_dice_loss(probs, target, weights)
            total = (
                (1.0 - self.lambda_) * loss_wdice + self.lambda_ * loss_wce
            )
        elif self.loss_option == "B":
            loss_dice = self._dice_loss_function(logits, target, softmax=True)
            total = (
                (1.0 - self.lambda_) * loss_dice + self.lambda_ * loss_wce
            )

        if self._is_verbose():
            self._verbose_tensor_stats("loss_wce", loss_wce)
            if self.loss_option == "A":
                self._verbose_tensor_stats("loss_wdice", loss_wdice)
            if self.loss_option == "B":
                self._verbose_tensor_stats("loss_dice", loss_dice)
            self._verbose_tensor_stats("loss_total", total)

        if not return_details:
            return total

        details: Dict[str, torch.Tensor] = {
            "UM_mean": uncertainty_map.mean().detach(),
            "UM_max": uncertainty_map.max().detach(),
            "BM_mean": boundary_map.mean().detach(),
            "BM_max": boundary_map.max().detach(),
            "weights": weights.detach(),
            "w_mean": weights.mean().detach(),
            "w_max": weights.max().detach(),
            "L_wCE": loss_wce.detach(),
        }
        if self.loss_option == "A":
            details["L_wDice"] = loss_wdice.detach()
        if self.loss_option == "B":
            details["L_Dice"] = loss_dice.detach()
            
        return total, details

    def _weighted_ce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted CE loss (loss_function.tex, Sec. 1.3.2)."""
        ce_per_pixel = F.cross_entropy(logits, target, reduction="none")
        weights_spatial = self._normalize_weights(weights, logits).squeeze(1)
        weight_sum = weights_spatial.sum().clamp_min(self.eps)
        return (weights_spatial * ce_per_pixel).sum() / weight_sum

    def _weighted_dice_loss(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted multiclass Dice loss (loss_function.tex, Sec. 1.3.3)."""
        if probs.dim() != 4:
            raise ValueError("probs must be [B, C, H, W]")
        target = target.long()
        num_classes = probs.size(1)
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)
        target_onehot = target_onehot.to(dtype=probs.dtype, device=probs.device)
        weights = self._normalize_weights(weights, probs)

        eps = 1e-8
        dims = (0, 2, 3)
        intersection = (weights * probs * target_onehot).sum(dim=dims)
        pred_sum = (weights * probs).sum(dim=dims)
        target_sum = (weights * target_onehot).sum(dim=dims)
        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        return (1.0 - dice).mean()

    def _normalize_weights(self, weights: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if weights.dim() == 3:
            weights = weights.unsqueeze(1)
        if weights.dim() != 4 or weights.size(1) != 1:
            raise ValueError("weights must be [B, 1, H, W] or [B, H, W]")
        if weights.size(0) != reference.size(0) or weights.shape[-2:] != reference.shape[-2:]:
            raise ValueError("weights must match the batch and spatial dimensions of the reference tensor")
        return weights.to(device=reference.device, dtype=reference.dtype)

    def _compute_boundary_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Detect multiclass label boundaries using local shifted comparisons."""
        if target.dim() != 3:
            raise ValueError("target must be [B, H, W]")

        batch, height, width = target.shape
        boundary = torch.zeros((batch, height, width), device=target.device, dtype=torch.bool)
        radius = self.boundary_radius

        for dy in range(-radius, radius + 1):
            y_start = max(0, -dy)
            y_end = min(height, height - dy)
            if y_start >= y_end:
                continue
            y_neighbor = slice(y_start + dy, y_end + dy)
            y_current = slice(y_start, y_end)

            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                x_start = max(0, -dx)
                x_end = min(width, width - dx)
                if x_start >= x_end:
                    continue
                x_neighbor = slice(x_start + dx, x_end + dx)
                x_current = slice(x_start, x_end)
                differs = target[:, y_current, x_current] != target[:, y_neighbor, x_neighbor]
                boundary[:, y_current, x_current] |= differs

        return boundary

    def _distance_transform_scipy(self, boundary: torch.Tensor) -> torch.Tensor:
        """Distance transform via SciPy (preferred when available)."""
        if boundary.dim() != 3:
            raise ValueError("boundary must be [B, H, W]")
        boundary = boundary.bool()
        distance_maps = []
        for b in range(boundary.size(0)):
            bmask = boundary[b].detach().cpu().numpy().astype(bool)
            if bmask.any():
                dist = _scipy_distance_transform_edt(~bmask).astype(np.float32)
            else:
                dist = np.full(bmask.shape, np.inf, dtype=np.float32)
            distance_maps.append(torch.from_numpy(dist))
        return torch.stack(distance_maps, dim=0).to(device=boundary.device)
