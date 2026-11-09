"""Projection-only FP16 evaluation checks using synthetic weights and inputs.

Run: .venv/bin/python -m unittest discover -s tests -p test_fp16_projections.py -v
No dataset, trained checkpoint, AWQ, or INC is needed. CPU forward checks do not
establish segmentation accuracy or GPU performance.
"""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import torch
from torch import nn

from networks.vit_seg_modeling import VisionTransformer
from src.benchmark import count_flops_gflops, count_parameters
from src.fp16_projections import FP16Projection, convert_fp16_projections
from src.test_helpers import parse_test_args
from test_se_auxiliary import small_config
from utils import model_size_mb_benchmark


def projection_model(fused=False):
    model = nn.Module()
    model.transformer = nn.Module()
    model.transformer.encoder = nn.Module()
    block = nn.Module()
    block.attn = nn.Module()
    if fused:
        block.attn.qkv = nn.Linear(128, 384)
        block.attn.proj = nn.Linear(128, 128)
    else:
        for name in ("query", "key", "value", "out"):
            setattr(block.attn, name, nn.Linear(128, 128))
    block.ffn = nn.Module()
    block.ffn.fc1 = nn.Linear(128, 256)
    block.ffn.fc2 = nn.Linear(256, 128, bias=False)
    block.norm = nn.LayerNorm(128)
    model.transformer.encoder.layer = nn.ModuleList([block])
    model.unrelated = nn.Linear(128, 128).double()
    model.register_buffer("untouched_buffer", torch.ones(3, dtype=torch.float64))
    return model


class FP16ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_cli_default_opt_in_and_incompatible_arguments(self):
        for flags, enabled in (([], False), (["--fp16_projections"], True)):
            with self.subTest(flags=flags), mock.patch.object(sys, "argv", ["test.py", *flags]):
                args = parse_test_args()
                self.assertEqual(args.fp16_projections, enabled)
                self.assertFalse(args.quantize)
        error = io.StringIO()
        with mock.patch.object(sys, "argv", ["test.py", "--fp16_projections", "--quantize"]):
            with redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                parse_test_args()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--fp16_projections", error.getvalue())
        self.assertIn("--quantize", error.getvalue())

    def test_exact_scope_storage_replacement_and_accounted_bytes(self):
        for fused in (False, True):
            with self.subTest(fused=fused):
                model = projection_model(fused).eval()
                originals = dict(model.named_parameters())
                before = {name: value.detach().clone() for name, value in originals.items()}
                original_modules = dict(model.named_modules())
                old_size = model_size_mb_benchmark(model)
                old_counts = count_parameters(model)
                report = convert_fp16_projections(model)
                json.dumps(report)
                self.assertEqual(report["label"], "FP16 Transformer projections")
                self.assertEqual(report["skipped_targets"], {})
                names = ("attn.qkv", "attn.proj") if fused else (
                    "attn.query", "attn.key", "attn.value", "attn.out")
                expected = {"transformer.encoder.layer.0." + name
                            for name in (*names, "ffn.fc1", "ffn.fc2")}
                self.assertEqual(set(report["converted_modules"]), expected)
                converted_elements = 0
                for name in expected:
                    wrapper = model.get_submodule(name)
                    self.assertIsInstance(wrapper, FP16Projection)
                    self.assertIs(wrapper.linear, original_modules[name])
                    self.assertFalse(wrapper.training)
                    for leaf, parameter in wrapper.linear.named_parameters():
                        converted_elements += parameter.numel()
                        self.assertEqual(parameter.dtype, torch.float16)
                        torch.testing.assert_close(parameter, before[name + "." + leaf].half(),
                                                   rtol=0, atol=0)
                # In-place conversion retains only one parameter set; no FP32 masters.
                self.assertEqual({id(p) for p in model.parameters()},
                                 {id(p) for p in originals.values()})
                for name, parameter in originals.items():
                    if name.rsplit(".", 1)[0] not in expected:
                        self.assertIs(model.get_parameter(name), parameter)
                        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
                self.assertEqual(model.untouched_buffer.dtype, torch.float64)
                self.assertEqual(count_parameters(model), old_counts)
                new_size = model_size_mb_benchmark(model)
                self.assertEqual(old_size["total_bytes"] - new_size["total_bytes"],
                                 2 * converted_elements)
                self.assertEqual(new_size["dtype_bytes"]["torch.float16"],
                                 2 * converted_elements)

    def test_input_dtype_device_contiguity_and_fp16_output(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                projection = FP16Projection(nn.Linear(128, 16).to(device)).eval()
                inputs = torch.randn(2, 128, 5).transpose(1, 2)
                self.assertFalse(inputs.is_contiguous())
                captured = []
                handle = projection.linear.register_forward_pre_hook(
                    lambda module, args: captured.append(args[0]))
                try:
                    with torch.no_grad():
                        output = projection(inputs)
                finally:
                    handle.remove()
                self.assertEqual(captured[0].dtype, torch.float16)
                self.assertEqual(captured[0].device, projection.linear.weight.device)
                self.assertTrue(captured[0].is_contiguous())
                self.assertEqual(output.dtype, torch.float16)
                self.assertEqual(output.shape, (2, 5, 16))
                self.assertTrue(torch.isfinite(output).all())

    def test_ineligible_and_missing_targets_are_reported(self):
        model = projection_model()
        attention = model.transformer.encoder.layer[0].attn
        attention.query = nn.Linear(127, 128)
        attention.key = nn.Linear(128, 127)
        attention.value = nn.Identity()
        with self.assertLogs(level="WARNING"):
            report = convert_fp16_projections(model)
        expected = {"transformer.encoder.layer.0.attn." + name
                    for name in ("query", "key", "value")}
        self.assertEqual(set(report["skipped_targets"]), expected)
        self.assertEqual(len(report["converted_modules"]), 3)
        self.assertEqual(attention.query.weight.dtype, torch.float32)
        self.assertEqual(attention.key.weight.dtype, torch.float32)
        with self.assertRaises(ValueError):
            convert_fp16_projections(nn.Sequential(nn.Linear(128, 128)))

    def test_mixed_ffn_eligibility_is_rejected_before_mutation(self):
        model = projection_model()
        ffn = model.transformer.encoder.layer[0].ffn
        ffn.fc1 = nn.Linear(128, 136)
        ffn.fc2 = nn.Linear(136, 128)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        with self.assertRaises(ValueError):
            convert_fp16_projections(model)
        self.assertFalse(any(isinstance(module, FP16Projection) for module in model.modules()))
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_real_full_token_and_gumbel_forwards_and_profiling(self):
        for attention, mode in (("standard", "manual"), ("gumbel", "manual"),
                                ("gumbel", "dist")):
            for drop_se in (False, True):
                with self.subTest(attention=attention, mode=mode, drop_se=drop_se):
                    torch.manual_seed(83)
                    config = small_config(attention=attention)
                    config.gumbel_sampling_mode = mode
                    config.patches.grid = (4, 4)
                    source = VisionTransformer(config, img_size=64, num_classes=3).eval()
                    checkpoint = source.state_dict()
                    self.assertTrue(any("SELayer" in name for name in checkpoint))
                    model = VisionTransformer(config, img_size=64, num_classes=3).eval()
                    model.load_state_dict(checkpoint, strict=True)
                    del checkpoint, source
                    if drop_se:
                        del model.transformer.encoder.SELayer
                        model.transformer.encoder.args.drop_se_block = True
                        self.assertFalse(any("SELayer" in name for name in model.state_dict()))
                    sample = torch.randn(1, 3, 64, 64)
                    before_flops, before_counts = count_flops_gflops(model, sample)
                    report = convert_fp16_projections(model)
                    seen = []
                    handles = []

                    def check_input(module, inputs):
                        self.assertEqual(inputs[0].dtype, torch.float16)
                        self.assertEqual(inputs[0].device, module.weight.device)
                        self.assertTrue(inputs[0].is_contiguous())
                        seen.append(module)

                    for name in report["converted_modules"]:
                        handles.append(model.get_submodule(name).linear.register_forward_pre_hook(check_input))
                    try:
                        with torch.no_grad():
                            logits, attention_maps, _, se_scales = model(sample)
                    finally:
                        for handle in handles:
                            handle.remove()
                    self.assertEqual(len(seen), len(report["converted_modules"]))
                    self.assertEqual(logits.shape, (1, 3, 64, 64))
                    self.assertEqual(logits.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(logits).all())
                    self.assertEqual(len(se_scales), 0 if drop_se else 2)
                    if attention == "gumbel":
                        self.assertLess(attention_maps[0].shape[-2], attention_maps[0].shape[-1])
                    after_flops, after_counts = count_flops_gflops(model, sample)
                    self.assertEqual(after_flops, before_flops)
                    self.assertEqual(after_counts, before_counts)
                    for name in report["converted_modules"]:
                        self.assertEqual(model.get_submodule(name).linear.weight.dtype,
                                         torch.float16)

    def test_evaluation_import_and_floating_benchmark_do_not_import_backends(self):
        script = """
import builtins
attempts = []
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in ('awq', 'neural_compressor'):
        attempts.append(name)
        raise ImportError('optional quantization backend intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import test
import torch
from torch import nn
from src.fp16_projections import FP16Projection
from src.benchmark import count_flops_gflops
model = nn.Sequential(FP16Projection(nn.Linear(128, 128))).eval()
count_flops_gflops(model, torch.randn(1, 2, 128))
assert not attempts, attempts
"""
        result = subprocess.run([sys.executable, "-c", script],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
