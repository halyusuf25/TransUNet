"""AWQ-style quantization helper for the ViT-based segmentation model in vit_seg_modeling.py.

What changed vs the earlier draft:
- Removed any dependency on importing `_search_module_scale`.
- Imports ONLY the *public* AWQ scaling API from auto_scale.py:
    `auto_scale_block`.

Important practical detail:
- Upstream `auto_scale_block()` is written for specific HuggingFace LLM decoder blocks and
  raises NotImplementedError for unknown module types. Your ViT `Block` is not one of
  those supported types (see auto_scale.py), so we provide a ViT-specific scaling routine
  that reuses the same math from the nested `_search_module_scale`.
- We avoid calling AWQ's `apply_scale()` because it moves modules to CPU after scaling.

This file focuses on W4 (grouped) weight-only quantization using WQLinear.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom

# -----------------------------------------------------------------------------
# Imports from the AWQ code you attached.
# We prefer importing from the installed package path; otherwise we fall back
# to local files next to this script.
# -----------------------------------------------------------------------------

# try:
#     from awq.quantize.auto_scale import auto_scale_block, apply_scale
#     from awq.quantize.quantizer import pseudo_quantize_tensor
#     from awq.quantize.qmodule import WQLinear
# except Exception:
#     # Fallback: assume the files live in the same folder as this script.
#     # (You can also adjust sys.path to point at your local awq/quantize folder.)
#     from auto_scale import auto_scale_block, apply_scale
#     from quantizer import pseudo_quantize_tensor
#     from qmodule import WQLinear
from awq.quantize.auto_scale import auto_scale_block
from awq.quantize.quantizer import pseudo_quantize_tensor
from awq.quantize.qmodule import WQLinear

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

@torch.no_grad()
def _get_act_scale(x: torch.Tensor) -> torch.Tensor:
    """Per-channel average magnitude (mean absolute activation).

    Matches AWQ's `get_act_scale` implementation:
        x.abs().view(-1, x.shape[-1]).mean(0)
    """
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def _scale_ln_fcs_cpu(ln: nn.LayerNorm, fcs: Sequence[nn.Linear], scales: torch.Tensor) -> None:
    """CPU-safe version of scale_ln_fcs.

    apply_scale() in AWQ hard-calls `.cuda()`. If you are running without CUDA,
    this helper keeps everything on the current device.

    Equivalent to auto_scale.scale_ln_fcs:
      ln.weight /= s
      ln.bias   /= s
      fc.weight *= s (on input channels)
    """
    if not isinstance(fcs, (list, tuple)):
        fcs = [fcs]

    scales = scales.to(ln.weight.device).to(ln.weight.dtype)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1).to(fc.weight.device).to(fc.weight.dtype))


@dataclass
class BlockCalibInputs:
    """Cached inputs needed to do ViT-block scaling search."""

    attn_in: torch.Tensor  # input to q/k/v (post attention_norm)
    ffn_in: torch.Tensor   # input to fc1 (post ffn_norm)


class AWQViTSegQuantizer:
    """Quantize a ViT-based segmentation model (vit_seg_modeling.py) with AWQ-style scaling.

    Usage:
        quantizer = AWQViTSegQuantizer(model, calib_loader, w_bit=4, q_group_size=128)
        qmodel = quantizer.quantize()

    Notes:
    - `calib_loader` should yield the same kind of inputs as your model expects.
    - Scaling search is done on the Transformer encoder blocks only.
    - Then we replace selected nn.Linear layers by WQLinear (4-bit group quant).
    """

    def __init__(
        self,
        model: nn.Module,
        calib_loader,
        w_bit: int = 4,
        q_group_size: int = 128,
        n_calib_batches: int = 8,
        device: Optional[torch.device] = None,
        args: Any = None,
    ):
        self.model = model
        self.calib_loader = calib_loader
        self.w_bit = int(w_bit)
        self.q_group_size = int(q_group_size)
        self.n_calib_batches = int(n_calib_batches)
        self.args = args

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        # AWQ's quantizer expects zero_point True (see quantizer.py / real_quantize_model_weight).
        self.q_config = {"zero_point": True, "q_group_size": self.q_group_size}
        self._se_scales_cache: Optional[List[torch.Tensor]] = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def quantize(self) -> nn.Module:
        """Return a quantized model (in-place modification of self.model)."""

        self.model.eval()

        # 1) Find encoder blocks (tries to match common TransUNet/ViT naming)
        blocks = self._get_vit_blocks(self.model)
        if not blocks:
            raise RuntimeError(
                "Could not find ViT transformer blocks. "
                "Please adapt _get_vit_blocks() to your model structure."
            )

        # 2) Collect calibration inputs for each block.
        #    For AWQ-style scaling we need the inputs to:
        #      - q/k/v linears (post attention_norm)
        #      - fc1 (post ffn_norm)
        block_inputs = self._collect_block_inputs(blocks)
                
        # 3) For each block:
        #    (a) auto-scale (AWQ)
        #    (b) replace selected nn.Linear with WQLinear
        for block_idx, (blk, inputs) in enumerate(zip(blocks, block_inputs)):
            # self._auto_scale_block_dispatch(blk, inputs)
            if getattr(self.args, "use_se_block", False):
                scales_list = self._auto_se_scale(blk, inputs, block_idx)
            else:
                scales_list = self._auto_scale_block_vit(blk, inputs)
                                
            self._apply_scales(blk, scales_list)
            
            self._convert_selected_linears_in_block(blk)

            # Help release memory between blocks.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.model

    # -------------------------------------------------------------------------
    # Locate blocks
    # -------------------------------------------------------------------------

    def _get_vit_blocks(self, model: nn.Module) -> List[nn.Module]:
        """Try to locate ViT/Transformer encoder blocks inside the segmentation model."""
        # Common patterns in TransUNet-like code:
        # model.transformer.encoder.layer or model.transformer.encoder.blocks
        candidates = []

        # brute-force: collect modules named 'Block' or that look like ViT blocks
        for name, m in model.named_modules():
            cls = m.__class__.__name__.lower()
            # Your vit_seg_modeling.py defines class Block(nn.Module)
            # with attrs attention_norm, ffn_norm, attn, ffn
            if cls == "block":
                if hasattr(m, "attention_norm") and hasattr(m, "ffn_norm") and hasattr(m, "attn") and hasattr(m, "ffn"):
                    candidates.append(m)

        # Fallback: try common nested containers
        if candidates:
            return candidates

        # Example: model.transformer.encoder.layer is a ModuleList
        for path in [
            "transformer.encoder.layer",
            "transformer.encoder.blocks",
            "encoder.layer",
            "encoder.blocks",
        ]:
            try:
                obj = self._get_op_by_name(model, path)
                if isinstance(obj, (nn.ModuleList, list, tuple)):
                    candidates = [b for b in obj if isinstance(b, nn.Module)]
                    if candidates:
                        return candidates
            except Exception:
                pass

        return []

    # -------------------------------------------------------------------------
    # Calibration input collection
    # -------------------------------------------------------------------------

    def _extract_se_scales_from_output(
        self,
        output: Any,
        expected_len: int,
    ) -> Optional[List[torch.Tensor]]:
        if not isinstance(output, (list, tuple)) or not output:
            raise ValueError(f"Unable to extract SE scales from output. Expected output to be a non-empty list/tuple, got {type(output)}")
        candidate = output[-1]
        if isinstance(candidate, (list, tuple)) and len(candidate) == expected_len:
            if all(torch.is_tensor(t) for t in candidate):
                return [t.detach() for t in candidate]
        raise ValueError("Unable to extract SE scales from output. Expected output to be a non-empty list/tuple.")

    @torch.no_grad()
    def _collect_block_inputs(self, blocks: List[nn.Module]) -> List[BlockCalibInputs]:
        """Cache per-block inputs needed for scaling search using forward pre-hooks."""
        # We'll record ONLY the first time each block sees data during calibration.
        captured: List[Optional[BlockCalibInputs]] = [None] * len(blocks)

        handles = []

        # For each block, hook q_proj input and fc1 input.
        for i, blk in enumerate(blocks):
            # Hook input to query projection (post attention_norm)
            def _make_q_hook(idx):
                def q_hook(module, inp):
                    # inp is a tuple (x,)
                    x = inp[0].detach()
                    if captured[idx] is None:
                        captured[idx] = BlockCalibInputs(attn_in=x, ffn_in=None)  # type: ignore[arg-type]
                return q_hook

            # Hook input to fc1 (post ffn_norm)
            def _make_fc1_hook(idx):
                def fc1_hook(module, inp):
                    x = inp[0].detach()
                    if captured[idx] is None:
                        captured[idx] = BlockCalibInputs(attn_in=None, ffn_in=x)  # type: ignore[arg-type]
                    elif captured[idx].ffn_in is None:
                        captured[idx].ffn_in = x
                return fc1_hook

            # Resolve modules inside the block
            if hasattr(blk, "attn") and hasattr(blk.attn, "query"):
                handles.append(blk.attn.query.register_forward_pre_hook(lambda m, inp, idx=i: _make_q_hook(idx)(m, inp)))
            else:
                raise AttributeError("Expected blk.attn.query to exist for ViT block.")

            if hasattr(blk, "ffn") and hasattr(blk.ffn, "fc1"):
                handles.append(blk.ffn.fc1.register_forward_pre_hook(lambda m, inp, idx=i: _make_fc1_hook(idx)(m, inp)))
            else:
                raise AttributeError("Expected blk.ffn.fc1 to exist for ViT block.")

        # Run a few batches through the model to trigger hooks.
        n_seen = 0
        # for batch in self.calib_loader:
        for i_batch, batch in enumerate(self.calib_loader):
            image = batch["image"]     
            input = self._extract_input_tensor(image).to(self.device)

            output = self.model(input)
            if getattr(self.args, "use_se_block", False) and self._se_scales_cache is None:
                se_scales = self._extract_se_scales_from_output(output, expected_len=len(blocks))
                if se_scales is not None:
                    self._se_scales_cache = se_scales
                else:
                    raise RuntimeError("SE scales not found in model output during calibration.")
            # for i in range(input.shape[0]):
            #     _ = self.model(input[i])
                
            n_seen += 1
            if n_seen >= self.n_calib_batches:
                break

        for h in handles:
            h.remove()

        # Post-process: fill missing fields by running again if needed.
        # (Usually both hooks should fire in a single forward.)
        out: List[BlockCalibInputs] = []
        for i, cap in enumerate(captured):
            if cap is None or cap.attn_in is None or cap.ffn_in is None:
                raise RuntimeError(
                    f"Failed to capture required inputs for block index {i}. "
                    "Make sure the calibration forward actually executes attention and MLP."
                )
            out.append(cap)

        return out

    def _resolve_patch_size(self, image: Any) -> Tuple[int, int]:
        """Resolve the resize target to match the model's expected img_size."""
        if self.args is not None and hasattr(self.args, "img_size") and self.args.img_size:
            size = int(self.args.img_size)
            return (size, size)
        if torch.is_tensor(image):
            if image.dim() >= 3 and image.shape[-1] <= 4:
                return (int(image.shape[-3]), int(image.shape[-2]))
            return (int(image.shape[-2]), int(image.shape[-1]))
        if isinstance(image, np.ndarray):
            if image.ndim >= 3 and image.shape[-1] <= 4:
                return (int(image.shape[-3]), int(image.shape[-2]))
            return (int(image.shape[-2]), int(image.shape[-1]))
        return (224, 224)

    def _extract_input_tensor(self, image: Any) -> torch.Tensor:
        patch_size = self._resolve_patch_size(image)
        input_images = []
        image = image.squeeze(0).cpu().detach().numpy()
        if self.args.dataset == 'Synapse':
            for ind in range(image.shape[0]):
                slice = image[ind, :, :]    
                x, y = slice.shape[0], slice.shape[1]
                if x != patch_size[0] or y != patch_size[1]:
                    slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
                input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float()
                input_images.append(input)
            input_tensor = torch.cat(input_images, dim=0)
            return input_tensor
        elif self.args.dataset == 'Cataract1k':
            # Resize each channel to patch_size and stack
            resized_channels = []
            H, W, C = image.shape
            for ch in range(C):
                slice = image[:, :, ch]
                # Resize to patch_size
                resized_slice = zoom(slice, (patch_size[0]/H, patch_size[1]/W), order=3)
                resized_channels.append(resized_slice)
            # Convert to tensor (1, 3, H, W)
            input = torch.from_numpy(np.stack(resized_channels, axis=0)).unsqueeze(0).float()
            input_images.append(input)
            input_tensor = torch.cat(input_images, dim=0)
            return input_tensor
        else:
            raise NotImplementedError("Dataset not supported for input extraction.")
            

    # -------------------------------------------------------------------------
    # Scaling search / application
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _auto_scale_block_dispatch(self, blk: nn.Module, inputs: BlockCalibInputs) -> None:
        """Try upstream auto_scale_block(); if unsupported, use a ViT-specific implementation."""

        # Upstream `auto_scale_block()` only supports several HF decoder layer types and
        # raises NotImplementedError for others.
        # We still try it first to satisfy the "use auto_scale_block" requirement.
        try:
            # Construct an input_feat dict (names are module-local paths) in case
            # upstream supports it in a future extension.
            input_feat = {
                "attn.query": inputs.attn_in,
                "ffn.fc1": inputs.ffn_in,
            }
            scales_list = auto_scale_block(
                blk,
                module_kwargs={},
                w_bit=self.w_bit,
                q_config=self.q_config,
                input_feat=input_feat,
            )
            
            print(f"auto_scale_block succeeded for block {blk} with scales_list: {scales_list}")
            # If it didn't raise, we can apply it.
            self._apply_scales(blk, scales_list)
            return
        except NotImplementedError:
            # Expected for ViT blocks.
            pass
            # raise RuntimeError("auto_scale_block raised NotImplementedError unexpectedly.")  # for debugging

        # ViT-specific: search scales for LN->(q,k,v) and LN->fc1.
        scales_list = self._auto_scale_block_vit(blk, inputs)
        print(f"ViT-specific auto-scaling for block {blk} with scales shape: {scales_list[0][2].shape}")
        self._apply_scales(blk, scales_list)

    def _se_scale_to_vec(self, se_scale: torch.Tensor) -> torch.Tensor:
        if se_scale.dim() == 3:  # (B, 1, C)
            return se_scale.mean(dim=0).squeeze(0)
        if se_scale.dim() == 4:  # (B, C, 1, 1)
            return se_scale.mean(dim=0).squeeze(-1).squeeze(-1)
        if se_scale.dim() == 2:  # (B, C)
            return se_scale.mean(dim=0)
        return se_scale.view(-1)

    @torch.no_grad()
    def _apply_scales(self, blk: nn.Module, scales_list) -> None:
        """Apply scales without moving modules across devices."""

        for prev_op_name, layer_names, scales in scales_list:
            prev_op = self._get_op_by_name(blk, prev_op_name)
            layers = [self._get_op_by_name(blk, n) for n in layer_names]
            if isinstance(prev_op, nn.LayerNorm):
                _scale_ln_fcs_cpu(prev_op, layers, scales)
            else:
                raise NotImplementedError(
                    "Scale application only implemented for LayerNorm -> Linear in ViT blocks."
                )

    def _get_op_by_name(self, module: nn.Module, name: str) -> nn.Module:
        """Minimal get_op_by_name equivalent for dotted paths (to avoid extra imports)."""
        cur = module
        for part in name.split("."):
            if not hasattr(cur, part):
                raise AttributeError(f"Module {type(cur)} has no attribute '{part}' (while resolving '{name}').")
            cur = getattr(cur, part)
        return cur

    @torch.no_grad()
    def _auto_scale_block_vit(self, blk: nn.Module, inputs: BlockCalibInputs):
        """Build the `scales_list` for a ViT `Block`.

        Returns a list of tuples in the same format as upstream auto_scale_block():
            (prev_op_name, (layer_name_1, layer_name_2, ...), scales)

        We implement two scaling points:
          1) attention_norm -> (attn.query, attn.key, attn.value)
          2) ffn_norm       -> (ffn.fc1)

        These correspond to AWQ's LN->QKV and LN->fc1 scaling patterns in LLM blocks.
        """

        # Weight quantize function used *inside* the grid search.
        def w_quantize_func(p: torch.Tensor) -> torch.Tensor:
            return pseudo_quantize_tensor(p, n_bit=self.w_bit, **self.q_config).detach()

        scales_list = []

        # 1) attention_norm -> q/k/v
        if not (hasattr(blk, "attention_norm") and hasattr(blk, "attn")):
            raise AttributeError("Expected ViT Block to have attention_norm and attn.")
        q = blk.attn.query
        k = blk.attn.key
        v = blk.attn.value

        scales_attn = self._search_best_scales(
            module2inspect=blk.attn,
            linears2scale=[q, k, v],
            x=inputs.attn_in,
            w_quantize_func=w_quantize_func,
            kwargs={},
        )
        scales_list.append(("attention_norm", ("attn.query", "attn.key", "attn.value"), scales_attn.cpu()))

        # 2) ffn_norm -> fc1
        if not (hasattr(blk, "ffn_norm") and hasattr(blk, "ffn") and hasattr(blk.ffn, "fc1")):
            raise AttributeError("Expected ViT Block to have ffn_norm and ffn.fc1.")

        scales_fc1 = self._search_best_scales(
            module2inspect=blk.ffn,
            linears2scale=[blk.ffn.fc1],
            x=inputs.ffn_in,
            w_quantize_func=w_quantize_func,
            kwargs={},
        )
        scales_list.append(("ffn_norm", ("ffn.fc1",), scales_fc1.cpu()))

        return scales_list

    @torch.no_grad()
    def _auto_se_scale(self, blk: nn.Module, inputs: BlockCalibInputs, block_idx: int):
        """Build the `scales_list` for a ViT `Block` using SE-derived scales."""
        if self._se_scales_cache is None:
            raise RuntimeError("SE scales were not captured; ensure the model outputs SE scales during calibration.")
        if block_idx >= len(self._se_scales_cache):
            raise RuntimeError(
                f"SE scales missing for block index {block_idx} (cache has {len(self._se_scales_cache)} entries)."
            )

        se_scale = self._se_scales_cache[block_idx]
        se_vec = self._se_scale_to_vec(se_scale).detach()

        scales_list = []

        # 1) attention_norm -> q/k/v
        if not (hasattr(blk, "attention_norm") and hasattr(blk, "attn")):
            raise AttributeError("Expected ViT Block to have attention_norm and attn.")
        scales_list.append(("attention_norm", ("attn.query", "attn.key", "attn.value"), se_vec.cpu()))

        # 2) ffn_norm -> fc1
        if not (hasattr(blk, "ffn_norm") and hasattr(blk, "ffn") and hasattr(blk.ffn, "fc1")):
            raise AttributeError("Expected ViT Block to have ffn_norm and ffn.fc1.")
        scales_list.append(("ffn_norm", ("ffn.fc1",), se_vec.cpu()))

        return scales_list

    @torch.no_grad()
    def _print_se_vs_awq(self, block_idx: int, se_scale: torch.Tensor, scales_list) -> None:
        """Verbose comparison between SE scales and AWQ per-channel scaling."""
        se_vec = self._se_scale_to_vec(se_scale)

        attn_scales = scales_list[0][2].to(se_vec.device, dtype=se_vec.dtype)
        ffn_scales = scales_list[1][2].to(se_vec.device, dtype=se_vec.dtype)

        if se_vec.numel() != attn_scales.numel() or se_vec.numel() != ffn_scales.numel():
            print(
                f"[SE vs AWQ] block {block_idx}: shape mismatch "
                f"se_vec={tuple(se_vec.shape)}, attn={tuple(attn_scales.shape)}, ffn={tuple(ffn_scales.shape)}"
            )
            return

        def _stats(t: torch.Tensor):
            return (
                t.mean().item(),
                t.min().item(),
                t.max().item(),
                t.std(unbiased=False).item(),
            )

        def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            denom = (a.norm() * b.norm()).clamp_min(1e-12)
            return (a * b).sum() / denom

        se_mean, se_min, se_max, se_std = _stats(se_vec)
        attn_mean, attn_min, attn_max, attn_std = _stats(attn_scales)
        ffn_mean, ffn_min, ffn_max, ffn_std = _stats(ffn_scales)
        cos_attn = _cos_sim(se_vec, attn_scales).item()
        cos_ffn = _cos_sim(se_vec, ffn_scales).item()

        print(
            f"[SE vs AWQ] block {block_idx}: "
            f"se_scale{tuple(se_scale.shape)} -> vec{tuple(se_vec.shape)} | "
            f"se(mean/min/max/std)={se_mean:.4g}/{se_min:.4g}/{se_max:.4g}/{se_std:.4g} | "
            f"attn(mean/min/max/std)={attn_mean:.4g}/{attn_min:.4g}/{attn_max:.4g}/{attn_std:.4g} | "
            f"ffn(mean/min/max/std)={ffn_mean:.4g}/{ffn_min:.4g}/{ffn_max:.4g}/{ffn_std:.4g} | "
            f"cos(se,attn)={cos_attn:.4f}, cos(se,ffn)={cos_ffn:.4f}"
        )

    @torch.no_grad()
    def _search_best_scales(
        self,
        module2inspect: nn.Module,
        linears2scale: Sequence[nn.Linear],
        x: torch.Tensor,
        w_quantize_func,
        kwargs: Optional[Dict[str, Any]] = None,
        n_grid: int = 20,
    ) -> torch.Tensor:
        """ViT-adapted copy of AWQ's nested `_search_module_scale` logic.

        This is the same idea as in auto_scale.py:
        - compute per-channel act magnitude
        - sweep `ratio` in [0,1) on a grid
        - build scales = x_max**ratio (normalized)
        - temporarily reparameterize weights with that scales, quantize, restore scale
        - measure output MSE vs original output
        - pick best scales

        See the original nested definition in auto_scale.py.
        """

        if kwargs is None:
            kwargs = {}

        # Put x on the same device as module2inspect.
        dev = next(module2inspect.parameters()).device
        x = x.to(dev)

        # Baseline output.
        with torch.no_grad():
            org_out= module2inspect(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]
        

        # Get per-channel activation scale.
        x_max = _get_act_scale(x) # AWQ's per-channel mean abs activation
        

        best_error = float("inf")
        best_scales: Optional[torch.Tensor] = None

        # Save weights once, restore each iteration.
        org_sd = {k: v.detach().cpu() for k, v in module2inspect.state_dict().items()}

        for gi in range(n_grid):
            ratio = gi / float(n_grid)
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()

            # Apply temporary scaling + quantization to selected linears.
            for fc in linears2scale:
                s = scales.view(1, -1).to(fc.weight.device).to(fc.weight.dtype)
                fc.weight.mul_(s)
                fc.weight.data = w_quantize_func(fc.weight.data) / s

            out = module2inspect(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (org_out - out).float().pow(2).mean().item()
            if loss < best_error:
                best_error = loss
                best_scales = scales.detach().clone()

            # Restore parameters.
            module2inspect.load_state_dict(org_sd)

        if best_scales is None:
            raise RuntimeError("Failed to find any valid scales (best_scales is None).")

        # Safety
        if torch.isnan(best_scales).any():
            raise RuntimeError("NaNs encountered in best_scales.")

        return best_scales.detach()

    # -------------------------------------------------------------------------
    # Replace linears by WQLinear
    # -------------------------------------------------------------------------

    def _convert_selected_linears_in_block(self, blk: nn.Module) -> None:
        """Replace relevant nn.Linear layers with WQLinear (weight-only 4-bit).

        We quantize:
          - attention q/k/v/out projections
          - MLP fc1/fc2

        We do NOT quantize LayerNorm, Dropout, Softmax, matmuls, etc.
        """
        # Collect targets (module-local dotted names)
        to_replace = []
        quantized = []
        verbose = getattr(self.args, "verbose", False)

        # Attention projections
        if hasattr(blk, "attn"):
            for name in ["query", "key", "value", "out"]:
                if hasattr(blk.attn, name) and isinstance(getattr(blk.attn, name), nn.Linear):
                    to_replace.append(f"attn.{name}")

        # MLP
        if hasattr(blk, "ffn"):
            for name in ["fc1", "fc2"]:
                if hasattr(blk.ffn, name) and isinstance(getattr(blk.ffn, name), nn.Linear):
                    to_replace.append(f"ffn.{name}")

        # Replace
        for dotted in to_replace:
            parent, leaf = dotted.rsplit(".", 1)
            parent_mod = self._get_op_by_name(blk, parent)
            lin = getattr(parent_mod, leaf)
            assert isinstance(lin, nn.Linear)

            # Sanity checks for WQLinear packing constraints:
            if self.q_group_size > 0 and (lin.in_features % self.q_group_size != 0):
                # Skip if group size doesn't divide in_features.
                continue
            if lin.out_features % (32 // self.w_bit) != 0:
                # Skip if output channels can't be packed as required by qmodule.WQLinear
                continue

            qlin = self._linear_to_wqlinear(lin)
            setattr(parent_mod, leaf, qlin)
            quantized.append(dotted)

        if verbose and quantized:
            print(
                f"[quantize] Replaced linears in {type(blk).__name__}: {', '.join(quantized)}"
            )
        else: 
            if verbose:
                print(f"[quantize] No linears replaced in {type(blk).__name__}.")

    def _linear_to_wqlinear(self, lin: nn.Linear) -> nn.Module:
        """Convert a torch.nn.Linear to AWQ's WQLinear with real (scale/zero) quantization."""
        # AWQ CUDA kernels expect half inputs; ensure weights (and bias) are half before packing.
        lin.weight.data = lin.weight.data.to(dtype=torch.float16, device=lin.weight.device)
        if lin.bias is not None:
            lin.bias.data = lin.bias.data.to(dtype=torch.float16, device=lin.bias.device)

        # We perform real quantization by obtaining per-group scales and zeros via pseudo_quantize_tensor.
        w = lin.weight.data
        w_q, scales, zeros = pseudo_quantize_tensor(w, n_bit=self.w_bit, get_scale_zp=True, **self.q_config)
        lin.weight.data = w_q  # not strictly necessary, but keeps weights consistent

        qlin = WQLinear.from_linear(
            lin,
            w_bit=self.w_bit,
            group_size=self.q_group_size,
            init_only=False,
            scales=scales,
            zeros=zeros,
        )
        return self._wrap_wqlinear_with_input_cast(qlin)

    def _wrap_wqlinear_with_input_cast(self, qlin: WQLinear) -> nn.Module:
        """Wrap WQLinear so inputs are cast to the expected dtype/device for the AWQ kernel."""

        class _InputCastWQLinear(nn.Module):
            def __init__(self, inner: WQLinear):
                super().__init__()
                self.inner = inner

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.to(device=self.inner.qweight.device, dtype=self.inner.scales.dtype)
                # AWQ CUDA kernels assume contiguous input
                if not x.is_contiguous():
                    x = x.contiguous()
                return self.inner(x)

        return _InputCastWQLinear(qlin)
