import torch
import torch.nn as nn

from src.benchmark import _fallback_hook_macs_vit, _model_size_metrics
from utils import model_size_mb_benchmark


def test_compression_ratio_uses_persistent_model_sizes():
    fp32_bytes = model_size_mb_benchmark(nn.Linear(4, 4))["total_bytes"]

    same_size = _model_size_metrics(fp32_bytes, fp32_bytes)
    assert same_size["compression_ratio"] == 1.0

    compressed = _model_size_metrics(fp32_bytes, fp32_bytes // 2)
    assert compressed["compression_ratio"] > 1.0


def test_hook_flop_counter_does_not_deepcopy_non_leaf_tensors():
    model = nn.Linear(4, 4)
    model.cached_quantizer_tensor = model.weight * 2
    model.train()

    macs = _fallback_hook_macs_vit(model, torch.ones(1, 4))

    assert macs == 16
    assert model.training
    assert not model._forward_hooks
