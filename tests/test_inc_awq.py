"""Focused INC AWQ checks; real calibration uses synthetic repository-model inputs.

Run from the repository root::

    .venv/bin/python -m unittest discover -s tests -p test_inc_awq.py -v

Checks invoking INC are skipped only when the optional package is absent.
"""

import argparse
import builtins
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from src import inc_awq
from src.inc_awq import quantize_inc_awq, select_inc_awq_layers
from networks.vit_seg_modeling import Block, VisionTransformer
from src.test_helpers import add_quantization_args
from test_se_auxiliary import small_config


def _model(attention="standard"):
    config = small_config(use_se=True, aux=False, attention=attention)
    # Top-K retains at least ten tokens: sixteen patches exercise real pruning.
    config.patches.grid = (4, 4)
    return VisionTransformer(config, img_size=64, num_classes=3)


class _CalibrationCases:
    def __init__(self, frame=False):
        self.consumed = 0
        self.frame = frame

    def __iter__(self):
        for slices in (1, 2, 3):
            self.consumed += 1
            shape = (1, 48, 56, 3) if self.frame else (1, slices, 48, 56)
            image = torch.rand(shape)
            if self.frame:
                image = image.mul(255).to(torch.uint8)
            yield {"image": image}


class INCAWQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(73)

    def test_parser_preserves_legacy_default_and_requires_quantize_opt_in(self):
        parser = add_quantization_args(argparse.ArgumentParser())
        default = parser.parse_args([])
        self.assertFalse(default.quantize)
        self.assertEqual(default.quantize_backend, "legacy")
        self.assertEqual(default.quantize_calibrate_batch_size, 8)
        self.assertEqual(parser.parse_args(["--quantize"]).quantize_backend, "legacy")
        backend_only = parser.parse_args(["--quantize_backend", "inc_awq"])
        self.assertFalse(backend_only.quantize)
        enabled = parser.parse_args([
            "--quantize", "--quantize_backend", "inc_awq",
            "--quantize_calibrate_batch_size", "3",
        ])
        self.assertTrue(enabled.quantize)
        self.assertEqual(enabled.quantize_backend, "inc_awq")
        self.assertEqual(enabled.quantize_calibrate_batch_size, 3)

    def test_imports_do_not_require_either_optional_quantizer(self):
        # A fresh interpreter also catches accidental eager imports in test.py.
        script = """
import importlib.abc
import sys
class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'awq', 'neural_compressor'}:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, RejectOptional())
import src.inc_awq
import test
assert 'networks.quantizer' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_inc_has_actionable_versioned_install_error(self):
        model = _model()
        loader = _CalibrationCases()
        original_import = builtins.__import__

        def without_inc(name, *args, **kwargs):
            if name.split(".")[0] == "neural_compressor":
                raise ModuleNotFoundError("simulated missing neural_compressor")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=without_inc):
            with self.assertRaisesRegex(ImportError, r"neural-compressor==3\.10"):
                quantize_inc_awq(model, loader, 2, SimpleNamespace(dataset="Synapse", img_size=64))
        self.assertEqual(loader.consumed, 0)

    def test_exact_encoder_projection_allowlist_excludes_se_backbone_and_decoder(self):
        for attention in ("standard", "topk", "gumbel", "ats"):
            with self.subTest(attention=attention):
                model = _model(attention)
                suffixes = (["attn.query", "attn.key", "attn.value", "attn.out"]
                            if attention == "standard" else ["attn.qkv", "attn.proj"])
                expected = {
                    f"transformer.encoder.layer.{i}.{suffix}"
                    for i in range(2) for suffix in suffixes + ["ffn.fc1", "ffn.fc2"]
                }
                actual = select_inc_awq_layers(model)
                self.assertEqual(set(actual), expected)
                self.assertEqual(len(actual), len(expected))
                for name in actual:
                    self.assertIsInstance(model.get_submodule(name), nn.Linear)
                excluded = [name for name, layer in model.named_modules()
                            if isinstance(layer, (nn.Linear, nn.Conv2d)) and name not in expected]
                self.assertTrue(any("SELayer" in name for name in excluded))
                self.assertTrue(any("hybrid_model" in name for name in excluded))
                self.assertTrue(any("decoder" in name for name in excluded))
                self.assertTrue(any("segmentation_head" in name for name in excluded))

    def test_unsupported_layouts_reject_before_calibration_or_weight_mutation(self):
        for invalid in ("shsa", "swin", "nonhybrid", "unknown", "nonlinear", "group_width"):
            with self.subTest(invalid=invalid):
                config = small_config(use_se=True, aux=False)
                if invalid == "shsa":
                    config.use_shsa = True
                model = VisionTransformer(config, img_size=32, num_classes=3)
                block = model.transformer.encoder.layer[0]
                if invalid == "swin":
                    model.config.use_swin = True
                elif invalid == "nonhybrid":
                    model.transformer.embeddings.hybrid = False
                elif invalid == "unknown":
                    block.attn = nn.Identity()
                elif invalid == "nonlinear":
                    block.ffn.fc1 = nn.Identity()
                elif invalid == "group_width":
                    block.ffn.fc2 = nn.Linear(255, 128)
                before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
                loader = _CalibrationCases()
                with self.assertRaisesRegex(ValueError, "SHSA|Swin|swin|hybrid|attention|Linear|linear|128|group"):
                    quantize_inc_awq(model, loader, 2, SimpleNamespace(dataset="Synapse", img_size=32))
                self.assertEqual(loader.consumed, 0)
                self.assertEqual(set(model.state_dict()), set(before))
                for name, expected in before.items():
                    torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)

    @unittest.skipUnless(importlib.util.find_spec("neural_compressor"), "optional neural-compressor is not installed")
    def test_invalid_calibration_counts_reject_before_conversion(self):
        from neural_compressor.torch import quantization

        for available, requested in ((0, 2), (1, 2), (1, 0), (1, -1)):
            with self.subTest(available=available, requested=requested):
                model = _model()
                before = {name: value.clone() for name, value in model.state_dict().items()}
                loader = [{"image": torch.rand(1, 1, 64, 64)} for _ in range(available)]
                with mock.patch.object(quantization, "quantize") as convert:
                    with self.assertRaisesRegex(ValueError, "calibration|loader batches"):
                        quantize_inc_awq(model, loader, requested, SimpleNamespace(dataset="Synapse", img_size=64))
                convert.assert_not_called()
                for name, expected in before.items():
                    torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)

    @unittest.skipUnless(importlib.util.find_spec("neural_compressor"), "optional neural-compressor is not installed")
    def test_inc_failure_restores_original_block_forward(self):
        from neural_compressor.torch import quantization

        model = _model()
        first = model.transformer.encoder.layer[0]
        before = {name: value.clone() for name, value in model.state_dict().items()}

        def fail_after_interception(adapter, *args, **kwargs):
            self.assertIs(adapter.layer[0], first)
            first.forward_orig = first.forward
            first.forward = lambda *args, **kwargs: None
            raise RuntimeError("injected INC capture failure")

        with mock.patch.object(quantization, "quantize", side_effect=fail_after_interception):
            with self.assertRaisesRegex(RuntimeError, "injected INC capture failure"):
                quantize_inc_awq(model, _CalibrationCases(), 1,
                                SimpleNamespace(dataset="Synapse", img_size=64))
        self.assertNotIn("forward", first.__dict__)
        self.assertNotIn("forward_orig", first.__dict__)
        for name, expected in before.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 64, 64))[0]
        self.assertEqual(logits.shape, (1, 3, 64, 64))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    @unittest.skipUnless(importlib.util.find_spec("neural_compressor"), "optional neural-compressor is not installed")
    def test_real_inc_calibration_qdq_and_pruned_segmentation_forward(self):
        # No INC algorithm is mocked: exercise the public API and actual tuple-
        # returning Blocks for separate Q/K/V and fused QKV attention layouts.
        for attention, dataset in (("standard", "Synapse"), ("topk", "ACDC"),
                                   ("gumbel", "EndoVis2018"), ("ats", "Cataract1k")):
            with self.subTest(attention=attention, dataset=dataset):
                model = _model(attention).train()
                selected = select_inc_awq_layers(model)
                blocks = list(model.transformer.encoder.layer)
                excluded = {
                    name: tensor.clone() for name, tensor in model.state_dict().items()
                    if not name.startswith("transformer.encoder.layer.")
                }
                original_weights = {name: model.get_submodule(name).weight.detach().clone()
                                    for name in selected}
                loader = _CalibrationCases(frame=dataset in {"EndoVis2018", "Cataract1k"})
                args = SimpleNamespace(dataset=dataset, img_size=64, use_se_block=True,
                                       normalize_present_class_eval=False)
                modes = []
                se_calls = []
                image_batches = []

                def capture_mode(module, inputs):
                    modes.append((module.training, torch.is_grad_enabled()))

                handles = [block.register_forward_pre_hook(capture_mode) for block in blocks]
                handles.append(model.transformer.embeddings.register_forward_pre_hook(capture_mode))
                handles.append(model.transformer.embeddings.register_forward_pre_hook(
                    lambda m, i: image_batches.append(i[0].shape)))
                handles.extend(layer.register_forward_pre_hook(lambda m, i: se_calls.append(m))
                               for layer in model.transformer.encoder.SELayer)
                retain_qdq = inc_awq._retain_inc_qdq
                qdq_checked = []

                def check_native_qdq(adapter, selected_dtypes):
                    # Compare real INC floating forwards with the final QDQ
                    # modules, including each compensating input multiplier.
                    references = {}
                    for name in selected_dtypes:
                        module = adapter.get_submodule(name)
                        linear = getattr(module, "linear", module)
                        device = next(linear.buffers()).device
                        probe = torch.linspace(-1, 1, linear.in_features, device=device).reshape(1, 1, -1)
                        scale = getattr(module, "input_scale", None)
                        references[name] = (probe, module(probe).clone(), scale,
                                            None if scale is None else scale.clone())
                    retain_qdq(adapter, selected_dtypes)
                    for name, (probe, expected, scale, scale_copy) in references.items():
                        module = adapter.get_submodule(name)
                        actual = module(probe)
                        tolerance = 0 if probe.device.type == "cpu" else 1e-3
                        torch.testing.assert_close(actual.float(), expected.float(),
                                                   rtol=tolerance, atol=tolerance)
                        if scale is not None:
                            self.assertIs(module.input_scale, scale)
                            torch.testing.assert_close(scale, scale_copy, rtol=0, atol=0)
                        qdq_checked.append(name)

                try:
                    with mock.patch.object(inc_awq, "_retain_inc_qdq", side_effect=check_native_qdq):
                        result = quantize_inc_awq(model, loader, 2, args)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertIs(result, model)
                self.assertEqual(loader.consumed, 2)
                self.assertTrue(modes)
                self.assertTrue(all(mode == (False, False) for mode in modes), modes)
                self.assertEqual(se_calls, [])
                expected_batches = [1, 1] if loader.frame else [1, 2]
                self.assertEqual(image_batches, [(batch, 3, 64, 64) for batch in expected_batches])
                self.assertFalse(model.training)
                self.assertEqual(list(model.transformer.encoder.layer), blocks)
                self.assertTrue(all(isinstance(block, Block) for block in blocks))
                for name, expected in excluded.items():
                    torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
                self.assertTrue(any(module.__class__.__name__ == "MulLinear"
                                    for module in model.modules()))
                self.assertFalse(any(module.__class__.__name__ == "INCWeightOnlyLinear"
                                     for module in model.modules()))
                self.assertEqual(len(qdq_checked), len(selected))
                changed = []
                for name in selected:
                    weight = model.get_submodule(name).weight
                    self.assertTrue(weight.is_floating_point())
                    self.assertTrue(bool(torch.isfinite(weight).all()))
                    changed.append(not torch.equal(weight, original_weights[name]))
                self.assertTrue(all(changed))

                # Mirror test.py's removal of checkpoint SE predictors before
                # inference; quantization itself must preserve excluded weights.
                model.transformer.encoder.SELayer = None
                outputs = []
                encoder_outputs = []
                handles = [block.register_forward_hook(lambda m, i, o: outputs.append(o))
                           for block in blocks]
                handles.append(model.transformer.encoder.register_forward_hook(
                    lambda m, i, o: encoder_outputs.append(o)))
                try:
                    with torch.no_grad():
                        logits = model(torch.randn(1, 3, 64, 64))[0]
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(logits.shape, (1, 3, 64, 64))
                self.assertTrue(bool(torch.isfinite(logits).all()))
                self.assertEqual(len(outputs), 2)
                if attention == "standard":
                    self.assertTrue(all(isinstance(output, tuple) and len(output) == 2 for output in outputs))
                    self.assertIsNone(encoder_outputs[0][2])
                else:
                    indices = torch.arange(16).unsqueeze(0)
                    for output in outputs:
                        self.assertIsInstance(output, tuple)
                        self.assertEqual(len(output), 3)
                        self.assertEqual(output[0].shape[1], output[2].shape[1])
                        indices = indices.gather(1, output[2])
                    self.assertLess(indices.shape[1], 16)
                    torch.testing.assert_close(encoder_outputs[0][2], indices, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
