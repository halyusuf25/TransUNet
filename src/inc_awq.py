"""Optional INC 3.10 activation-aware AWQ with INC-recovered floating-point QDQ."""

import logging
import re
from itertools import islice

import numpy as np
import torch
from scipy.ndimage import zoom
from torch import nn


_ENCODER_PREFIX = "transformer.encoder."


def select_inc_awq_layers(model):
    """Validate the actual encoder layout before INC can mutate any weights."""
    from networks.attention import ATSAttention, TopkAttention
    from networks.vit_seg_modeling import Attention, Block

    encoder = getattr(getattr(model, "transformer", None), "encoder", None)
    blocks = getattr(encoder, "layer", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise ValueError("inc_awq requires net.transformer.encoder.layer to be a nonempty nn.ModuleList")
    if getattr(model.args, "use_swin", False):
        raise ValueError("inc_awq does not support the Swin configuration; use the ResNet/ViT encoder")
    if not model.transformer.embeddings.hybrid:
        raise ValueError("inc_awq requires hybrid embeddings; this repository's nonhybrid Embeddings.forward is unsupported")

    selected = []
    for index, block in enumerate(blocks):
        prefix = f"{_ENCODER_PREFIX}layer.{index}."
        if type(block) is not Block:
            raise ValueError(f"inc_awq requires the repository Block at {prefix[:-1]}")
        if type(block.attn) is Attention:
            projections = ("query", "key", "value", "out")
        elif type(block.attn) in (TopkAttention, ATSAttention):
            projections = ("qkv", "proj")
        else:
            raise ValueError(
                f"inc_awq does not support {type(block.attn).__name__} at {prefix}attn "
                "(including partial-channel SHSA); supported layouts are Attention, TopkAttention and ATSAttention"
            )
        names = [f"attn.{name}" for name in projections] + ["ffn.fc1", "ffn.fc2"]
        actual = {name for name, module in block.named_modules() if isinstance(module, nn.Linear)}
        if actual != set(names):
            raise ValueError(f"inc_awq found an unsupported linear layout at {prefix}: {sorted(actual)}")
        for name in names:
            module = block.get_submodule(name)
            expected_in = block.ffn.fc1.out_features if name == "ffn.fc2" else block.hidden_size
            expected_out = (
                3 * block.hidden_size if name == "attn.qkv" else
                block.ffn.fc1.out_features if name == "ffn.fc1" else block.hidden_size
            )
            if (type(module) is not nn.Linear or module.in_features != expected_in
                    or module.out_features != expected_out or module.in_features % 128):
                raise ValueError(
                    f"inc_awq requires full-channel floating-point Linear projections and input widths "
                    f"divisible by group_size=128; unsupported layer: {prefix}{name}"
                )
            selected.append(prefix + name)
    return selected


class _EncoderAWQAdapter(nn.Module):
    """Expose only encoder blocks to INC's first-ModuleList discovery and tracing.

    These are the original blocks, including token selection and residual gathers.
    INC replays tuple output[0]; the original Encoder still composes local indices
    and VisionTransformer scatters tokens back into the image grid at inference.
    """

    def __init__(self, blocks):
        super().__init__()
        self.layer = blocks

    def forward(self, hidden_states):
        for block in self.layer:
            hidden_states = block(hidden_states)[0]
        return hidden_states


def _retain_inc_qdq(model, selected_dtypes):
    """Materialize INC's own dequantized weights, preserving input-scale wrappers.

    INC 3.10's final RTN conversion packs even though AWQ requests return_int=False.
    recover() is the same QDQ reconstruction its native floating forward uses.
    """
    from neural_compressor.torch.algorithms.weight_only.modules import INCWeightOnlyLinear, MulLinear

    for name, dtype in selected_dtypes.items():
        module = model.get_submodule(name)
        if isinstance(module, MulLinear):
            name += ".linear"
            module = module.linear
        if isinstance(module, INCWeightOnlyLinear):
            weight = module.recover().to(dtype=dtype)
            linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None,
                               device=weight.device, dtype=dtype).eval()
            linear.weight.copy_(weight)
            if module.bias is not None:
                linear.bias.copy_(module.bias)
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, linear)


def _calibration_images(image, args):
    """Match existing order-3 volume/frame resizing without importing legacy AWQ.

    Each loader batch is one volume (all slices) or one HWC frame, as in test.py.
    Frame normalization follows test_single_frame_present_classes in utils.py.
    """
    image = image.squeeze(0).detach().cpu().numpy()
    size = int(args.img_size)
    if args.dataset in ("Synapse", "ACDC"):
        if image.ndim != 3:
            raise ValueError("inc_awq volume calibration expects [1, slices, H, W]")
        images = [zoom(s, (size / s.shape[0], size / s.shape[1]), order=3) for s in image]
        result = torch.from_numpy(np.stack(images)).unsqueeze(1).float()
    elif args.dataset in ("Cataract1k", "EndoVis2018"):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("inc_awq frame calibration expects [1, H, W, 3]")
        images = [zoom(image[..., c], (size / image.shape[0], size / image.shape[1]), order=3)
                  for c in range(3)]
        result = torch.from_numpy(np.stack(images)).unsqueeze(0).float()
        if getattr(args, "normalize_present_class_eval", getattr(args, "normalize_endovis_eval", False)):
            mean = result.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            std = result.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            result = (result / 255.0 - mean) / std
    else:
        raise ValueError(f"inc_awq calibration does not support dataset {args.dataset!r}")
    return result


@torch.no_grad()
def quantize_inc_awq(model, calib_loader, n_calib_batches, args):
    """Apply standard activation-aware AWQ in place, independently of SE gates."""
    selected = select_inc_awq_layers(model)
    if n_calib_batches < 1:
        raise ValueError("--quantize_calibrate_batch_size must be a positive number of loader batches")
    try:
        from neural_compressor.torch.quantization import AWQConfig, quantize
        from neural_compressor.torch.utils import get_accelerator
    except ImportError as exc:
        raise ImportError(
            "inc_awq requires Intel Neural Compressor's PyTorch API. In a Python >=3.11 "
            "environment with the repository's PyTorch dependencies, install: "
            "python -m pip install neural-compressor==3.10 prettytable psutil py-cpuinfo pydantic packaging"
        ) from exc

    logging.info("INC AWQ selected %d encoder layers:\n%s", len(selected), "\n".join(selected))
    logging.info("INC AWQ uses activation calibration independent of SE gates; recovered W4 QDQ "
                 "retains floating-point weights/compute, not packed INT4 storage or accelerated INT4 inference.")
    model.eval()
    device = next(model.parameters()).device
    # Cache only embeddings. INC will capture/replay every batch through the real
    # blocks; SE predictors, backbone, decoder and head never enter its model.
    tokens = []
    for batch in islice(calib_loader, n_calib_batches):
        images = _calibration_images(batch["image"], args).to(device)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        hidden_states, _ = model.transformer.embeddings(images)
        tokens.append(hidden_states.detach().cpu())
    if len(tokens) != n_calib_batches:
        raise ValueError(f"inc_awq requested {n_calib_batches} calibration loader batches, "
                         f"but only {len(tokens)} were available")

    # INC automatically moves blocks to its selected accelerator during convert.
    # Capture calibration inputs there too, then restore the inference device.
    inc_device = get_accelerator().current_device_name()
    adapter = _EncoderAWQAdapter(model.transformer.encoder.layer).eval()
    selected_dtypes = {name.removeprefix(_ENCODER_PREFIX): model.get_submodule(name).weight.dtype
                       for name in selected}
    config = AWQConfig(dtype="fp32")
    awq = AWQConfig(dtype="int", bits=4, group_size=128, group_dim=1, use_sym=False,
                    use_auto_scale=True, use_auto_clip=True, folding=False)
    for name in selected:
        # INC local names are regexes; anchor them to exclude every other layer.
        config.set_local("^" + re.escape(name.removeprefix(_ENCODER_PREFIX)) + "$", awq)

    def calibrate(prepared):
        prepared.eval()
        for hidden_states in tokens:
            prepared(hidden_states.to(inc_device))

    logging.info("INC AWQ calibrating on all %d requested loader batches", len(tokens))
    first_block = adapter.layer[0]
    forwards = {key: first_block.__dict__[key] for key in ("forward", "forward_orig")
                if key in first_block.__dict__}
    try:
        adapter.to(inc_device)
        quantize(adapter, config, run_fn=calibrate, example_inputs=tokens[0][:1].to(inc_device), inplace=True)
        _retain_inc_qdq(adapter, selected_dtypes)
    finally:
        # INC temporarily replaces the first block's forward while capturing.
        # Remove its interception even if calibration/conversion raises.
        for key in ("forward", "forward_orig"):
            first_block.__dict__.pop(key, None)
        first_block.__dict__.update(forwards)
        model.to(device)
    # Keep INC's compensating scales and MulLinear wrappers on the shared blocks.
    return model
