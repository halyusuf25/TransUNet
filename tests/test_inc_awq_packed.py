"""INC-to-WQLinear packing checks using real integer packing implementations.

Run with an INC 3.10 environment and the repository's legacy AWQ installation::

    python -m unittest discover -s tests -p test_inc_awq_packed.py -v

AWQ_QMODULE_SOURCE may point to an installed awq/quantize/qmodule.py when the
INC test environment does not expose that installation. CPU checks mock only
the unavailable CUDA extension; the stub raises if a kernel is invoked. CUDA
numerical and segmentation checks explicitly skip without working CUDA kernels.
"""

import argparse
import builtins
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from src import inc_awq
from src.test_helpers import add_quantization_args
from networks.vit_seg_modeling import VisionTransformer
from test_inc_awq import _CalibrationCases
from test_se_auxiliary import small_config


def _model(attention="standard"):
    config = small_config(use_se=True, aux=False, attention=attention)
    config.hidden_size = 256
    config.transformer.mlp_dim = 512
    config.patches.grid = (4, 4)
    return VisionTransformer(config, img_size=64, num_classes=3)


def _qmodule_source():
    candidates = []
    if os.environ.get("AWQ_QMODULE_SOURCE"):
        candidates.append(Path(os.environ["AWQ_QMODULE_SOURCE"]))
    spec = importlib.util.find_spec("awq")
    if spec is not None and spec.submodule_search_locations:
        candidates.extend(Path(root) / "quantize" / "qmodule.py"
                          for root in spec.submodule_search_locations)
    path = next((path for path in candidates if path.is_file()), None)
    if path is None:
        raise unittest.SkipTest("Installed legacy AWQ qmodule source unavailable; set AWQ_QMODULE_SOURCE")
    return path


def _load_qmodule(cuda=False):
    name = "_inc_packed_test_qmodule_cuda" if cuda else "_inc_packed_test_qmodule_cpu"
    spec = importlib.util.spec_from_file_location(name, _qmodule_source())
    module = importlib.util.module_from_spec(spec)
    if cuda:
        try:
            import awq_inference_engine
        except (ImportError, OSError) as exc:
            raise unittest.SkipTest(f"AWQ CUDA extension unavailable: {exc}") from exc
        for name in ("gemv_forward_cuda_new", "gemm_forward_cuda_new"):
            if not callable(getattr(awq_inference_engine, name, None)):
                raise unittest.SkipTest(f"AWQ CUDA extension lacks {name}")
        spec.loader.exec_module(module)
    else:
        engine = ModuleType("awq_inference_engine")
        engine.gemv_forward_cuda_new = mock.Mock(side_effect=AssertionError("CPU test cannot execute a CUDA kernel"))
        engine.gemm_forward_cuda_new = mock.Mock(side_effect=AssertionError("CPU test cannot execute a CUDA kernel"))
        with mock.patch.dict(sys.modules, {"awq_inference_engine": engine}):
            spec.loader.exec_module(module)
    return module


def _unpack_wq_int16(qweight, out_features, in_features):
    """Reverse the CUDA kernel's row-interleaved nibble layout independently."""
    words = qweight.to(torch.int32) & 0xFFFF
    nibbles = torch.stack([(words >> shift) & 15 for shift in (0, 4, 8, 12)], dim=-1)
    values = nibbles.reshape(out_features // 4, in_features // 64, 4, 64)
    values = values.transpose(1, 2).reshape(out_features, in_features)
    values = values.reshape(out_features, in_features // 32, 4, 2, 4)
    values = values.transpose(3, 4).reshape(out_features, in_features)
    values = values.reshape(out_features, in_features // 32, 4, 4, 2)
    return values.transpose(2, 3).reshape(out_features, in_features)


def _inc_linear(in_features=256, out_features=128, device="cpu"):
    from neural_compressor.torch.algorithms.weight_only.modules import INCWeightOnlyLinear

    rows = torch.arange(out_features, device=device)[:, None]
    columns = torch.arange(in_features, device=device)[None, :]
    codes = (rows * 3 + columns * 5 + (columns // 16) * 7).remainder(16).to(torch.int32)
    groups = torch.arange(in_features // 128, device=device)[None, :]
    zeros = (rows + groups * 5).remainder(16).to(torch.int32)
    scales = (0.01 + (rows + groups * 3).remainder(17) / 100).half()
    bias = torch.linspace(-0.2, 0.2, out_features, device=device).half()
    linear = INCWeightOnlyLinear(in_features, out_features, bits=4, group_size=128,
                                 dtype="int", zp=True, bias=True,
                                 use_optimum_format=True, device=device)
    linear.pack(codes.clone(), scales.clone(), zeros.clone(), bias.clone())
    return linear, codes, scales, zeros, bias


class INCPackedArgumentTests(unittest.TestCase):
    def test_new_backend_is_opt_in_without_changing_existing_defaults(self):
        parser = add_quantization_args(argparse.ArgumentParser())
        self.assertEqual(parser.parse_args([]).quantize_backend, "legacy")
        args = parser.parse_args(["--quantize_backend", "inc_awq_packed"])
        self.assertFalse(args.quantize)
        args = parser.parse_args(["--quantize", "--quantize_backend", "inc_awq_packed"])
        self.assertTrue(args.quantize)
        self.assertEqual(args.quantize_backend, "inc_awq_packed")
        self.assertEqual(args.quantize_calibrate_batch_size, 8)

    @unittest.skipUnless(importlib.util.find_spec("neural_compressor"), "optional neural-compressor is not installed")
    def test_cpu_rejected_before_model_or_calibration_mutation(self):
        model = _model().train()
        loader = _CalibrationCases()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        args = SimpleNamespace(quantize_backend="inc_awq_packed", dataset="Synapse", img_size=64)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex((RuntimeError, ValueError), "CUDA|cuda"):
                inc_awq.quantize_inc_awq(model, loader, 2, args)
        self.assertEqual(loader.consumed, 0)
        self.assertTrue(model.training)
        for name, expected in before.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)

    def test_missing_awq_dependency_has_actionable_error(self):
        model = _model()
        selected = inc_awq.select_inc_awq_layers(model)
        original_import = builtins.__import__

        def without_awq(name, *args, **kwargs):
            if name.split(".")[0] == "awq":
                raise ImportError("simulated missing legacy AWQ")
            return original_import(name, *args, **kwargs)

        # Reach dependency validation without allocating any CUDA tensors.
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)), \
             mock.patch.object(torch.cuda, "current_stream", return_value=None), \
             mock.patch.object(torch.cuda, "default_stream", return_value=None), \
             mock.patch.object(model, "parameters", return_value=iter([SimpleNamespace(device=torch.device("cuda:0"))])), \
             mock.patch.object(inc_awq, "version", return_value="3.10"), \
             mock.patch("builtins.__import__", side_effect=without_awq):
            with self.assertRaisesRegex(ImportError, "awq/kernels|awq_inference_engine"):
                inc_awq._load_packed_backend(model, selected)

    def test_unsupported_version_and_kernel_dimensions_reject_before_loading_awq(self):
        for invalid in ("version", "input_width", "output_width"):
            with self.subTest(invalid=invalid):
                model = _model()
                selected = inc_awq.select_inc_awq_layers(model)
                projection = model.get_submodule(selected[0])
                if invalid == "input_width":
                    projection.in_features = 128
                elif invalid == "output_width":
                    projection.out_features = 136
                with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                     mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)), \
                     mock.patch.object(torch.cuda, "current_stream", return_value=None), \
                     mock.patch.object(torch.cuda, "default_stream", return_value=None), \
                     mock.patch.object(model, "parameters", return_value=iter([SimpleNamespace(device=torch.device("cuda:0"))])), \
                     mock.patch.object(inc_awq, "version", return_value="3.7.1" if invalid == "version" else "3.10"):
                    with self.assertRaisesRegex((ValueError, RuntimeError), "3.10|256|128"):
                        inc_awq._load_packed_backend(model, selected)


@unittest.skipUnless(importlib.util.find_spec("neural_compressor"), "optional neural-compressor is not installed")
class INCPackedConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(73)

    def assert_no_float_weight(self, packed):
        weight_shape = (packed.inner.out_features, packed.inner.in_features)
        for module in packed.modules():
            self.assertFalse(hasattr(module, "weight"))
            self.assertFalse(any(isinstance(value, torch.Tensor)
                                 and value.is_floating_point()
                                 and value.shape == weight_shape
                                 for value in module.__dict__.values()))
        self.assertEqual(list(packed.parameters()), [])
        self.assertFalse(any(value.is_floating_point() and value.shape == weight_shape
                             for value in packed.state_dict().values()))

    def test_exact_integer_codes_scales_zero_padding_and_bias_without_requantizing(self):
        qmodule = _load_qmodule()
        for in_features in (256, 768, 1024, 1280):
            with self.subTest(in_features=in_features):
                source, codes, scales, zeros, bias = _inc_linear(in_features=in_features)
                before = {name: value.clone() for name, value in source.state_dict().items()}
                with mock.patch.object(source, "recover", side_effect=AssertionError("must preserve integers")), \
                     mock.patch.object(qmodule.WQLinear, "from_linear", side_effect=AssertionError("must not requantize")):
                    packed = inc_awq._pack_inc_linear(source, qmodule)
                inner = packed.inner
                self.assertIsInstance(inner, qmodule.WQLinear)
                self.assertEqual((inner.w_bit, inner.group_size), (4, 128))
                self.assertEqual(inner.qweight.dtype, torch.int16)
                self.assertEqual(inner.qweight.shape, (128 // 4, in_features))
                recovered_codes = _unpack_wq_int16(inner.qweight, 128, in_features)
                torch.testing.assert_close(recovered_codes, codes, rtol=0, atol=0)
                groups = in_features // 128
                self.assertEqual(inner.scales.dtype, torch.float16)
                self.assertEqual(inner.scales.shape,
                                 (qmodule.calculate_zeros_width(in_features, 128) * 8, 128))
                torch.testing.assert_close(inner.scales[:groups], scales.T, rtol=0, atol=0)
                torch.testing.assert_close(inner.scaled_zeros[:groups], -(scales.float() * zeros).half().T,
                                           rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(inner.scales[groups:]).item(), 0)
                self.assertEqual(torch.count_nonzero(inner.scaled_zeros[groups:]).item(), 0)
                torch.testing.assert_close(inner.bias, bias, rtol=0, atol=0)
                self.assertTrue(all(value.is_contiguous() for value in inner.buffers()))
                self.assert_no_float_weight(packed)
                for name, expected in before.items():
                    torch.testing.assert_close(source.state_dict()[name], expected, rtol=0, atol=0)

    def test_mullinear_compensation_and_unselected_modules_are_preserved(self):
        from neural_compressor.torch.algorithms.weight_only.modules import MulLinear

        qmodule = _load_qmodule()
        source, _, _, _, _ = _inc_linear()
        scale = torch.linspace(0.5, 1.5, 256)
        wrapper = MulLinear(source, scale.clone())
        adapter = nn.Module()
        adapter.layer = nn.ModuleList([nn.Module()])
        adapter.layer[0].projection = wrapper
        adapter.excluded = nn.Linear(256, 128)
        excluded = adapter.excluded
        excluded_weight = excluded.weight.detach().clone()
        input_scale = wrapper.input_scale
        with torch.no_grad():
            inc_awq._retain_inc_packed(adapter, {"layer.0.projection": torch.float32}, qmodule)
        self.assertIs(adapter.layer[0].projection, wrapper)
        self.assertIs(wrapper.input_scale, input_scale)
        torch.testing.assert_close(wrapper.input_scale, scale, rtol=0, atol=0)
        self.assertIsInstance(wrapper.linear.inner, qmodule.WQLinear)
        self.assert_no_float_weight(wrapper.linear)
        self.assertIs(adapter.excluded, excluded)
        torch.testing.assert_close(excluded.weight, excluded_weight, rtol=0, atol=0)

    def test_unsupported_inc_metadata_rejects_without_partial_replacement(self):
        qmodule = _load_qmodule()
        for invalid in ("bits", "group_size", "scale_dtype", "zero_scale", "reordered_groups", "missing_zeros"):
            with self.subTest(invalid=invalid):
                first, _, _, _, _ = _inc_linear()
                second, _, _, _, _ = _inc_linear()
                if invalid == "bits":
                    second.bits = 8
                elif invalid == "group_size":
                    second.group_size = 64
                elif invalid == "scale_dtype":
                    second.scales = second.scales.float()
                elif invalid == "zero_scale":
                    second.scales.zero_()
                elif invalid == "reordered_groups":
                    second.g_idx = (torch.arange(256) // 128).roll(128)
                else:
                    del second.qzeros
                adapter = nn.Module()
                adapter.layer = nn.ModuleList([first, second])
                before = {name: value.clone() for name, value in adapter.state_dict().items()}
                with self.assertRaisesRegex(ValueError, "INT4|group|scales|shapes"):
                    inc_awq._retain_inc_packed(adapter, {"layer.0": torch.float32, "layer.1": torch.float32}, qmodule)
                self.assertIs(adapter.layer[0], first)
                self.assertIs(adapter.layer[1], second)
                for name, expected in before.items():
                    torch.testing.assert_close(adapter.state_dict()[name], expected, rtol=0, atol=0)

    def test_real_inc_calibration_to_packed_preserves_pruning_and_exclusions(self):
        qmodule = _load_qmodule()
        for attention in ("topk", "gumbel", "ats"):
            with self.subTest(attention=attention):
                model = _model(attention).train()
                selected = inc_awq.select_inc_awq_layers(model)
                blocks = list(model.transformer.encoder.layer)
                excluded = {name: value.clone() for name, value in model.state_dict().items()
                            if not name.startswith("transformer.encoder.layer.")}
                loader = _CalibrationCases()
                modes = []
                token_counts = []
                def capture_block(module, inputs, output):
                    modes.append((module.training, torch.is_grad_enabled()))
                    token_counts.append((inputs[0].shape[1], output[0].shape[1]))

                handles = [block.register_forward_hook(capture_block) for block in blocks]
                args = SimpleNamespace(quantize_backend="inc_awq_packed", dataset="Synapse", img_size=64)
                try:
                    # CPU validates real calibration and packing; only the CUDA
                    # deployment preflight is bypassed, never INC or arithmetic.
                    with mock.patch.object(inc_awq, "_load_packed_backend", return_value=qmodule):
                        result = inc_awq.quantize_inc_awq(model, loader, 2, args)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertIs(result, model)
                self.assertEqual(loader.consumed, 2)
                self.assertTrue(modes)
                self.assertTrue(all(mode == (False, False) for mode in modes))
                self.assertTrue(any(after < before for before, after in token_counts))
                self.assertEqual(list(model.transformer.encoder.layer), blocks)
                for name in selected:
                    module = model.get_submodule(name)
                    packed = getattr(module, "linear", module)
                    self.assertIsInstance(packed.inner, qmodule.WQLinear)
                    self.assert_no_float_weight(packed)
                for name, expected in excluded.items():
                    torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: real packed GEMV/GEMM not executed")
    def test_cuda_gemv_and_gemm_match_packed_dequantized_reference(self):
        from neural_compressor.torch.algorithms.weight_only.modules import MulLinear

        qmodule = _load_qmodule(cuda=True)
        source, codes, scales, _, bias = _inc_linear(device="cuda")
        inc_weight = source.recover().half()
        packed = inc_awq._pack_inc_linear(source, qmodule).cuda()
        # WQLinear reconstructs with a rounded FP16 zero-point offset. Match
        # that representation while reporting its separate delta from INC QDQ.
        offsets = packed.inner.scaled_zeros[:2].T
        weight = (codes.float() * scales.float().repeat_interleave(128, dim=1)
                  + offsets.float().repeat_interleave(128, dim=1)).half()
        input_scale = torch.linspace(0.5, 1.5, 256, device="cuda")
        compensated = MulLinear(packed, input_scale)
        for rows in (1, 7, 8, 19, 32, 33, 128, 196):
            with self.subTest(rows=rows):
                inputs = torch.randn(1, rows, 256, device="cuda", dtype=torch.float32)
                with torch.no_grad():
                    expected = F.linear((inputs * input_scale).half(), weight, bias)
                    inc_expected = F.linear((inputs * input_scale).half(), inc_weight, bias)
                    actual = compensated(inputs)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, torch.float16)
                self.assertTrue(bool(torch.isfinite(actual).all()))
                error = actual.float() - expected.float()
                relative_rms = (error.square().mean() / expected.float().square().mean()).sqrt().item()
                inc_delta = (actual.float() - inc_expected.float()).abs().max().item()
                print(f"packed CUDA rows={rows}: max_abs={error.abs().max().item():.6g}, "
                      f"rms={error.square().mean().sqrt().item():.6g}, relative_rms={relative_rms:.6g}, "
                      f"INC_QDQ_delta={inc_delta:.6g}")
                # These CUDA kernels accumulate in FP16 (unlike F.linear's
                # FP32 accumulator), so cancellation near zero needs an absolute
                # bound as well as a small relative error across the output.
                self.assertLess(relative_rms, 0.002)
                torch.testing.assert_close(actual.float(), expected.float(), rtol=0.005, atol=0.05)
                self.assert_no_float_weight(packed)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: packed segmentation forward not executed")
    def test_cuda_real_inc_packed_segmentation_forward_and_indices(self):
        _load_qmodule(cuda=True)
        for attention in ("standard", "topk", "gumbel", "ats"):
            with self.subTest(attention=attention):
                model = _model(attention).cuda()
                selected = inc_awq.select_inc_awq_layers(model)
                args = SimpleNamespace(quantize_backend="inc_awq_packed", dataset="Synapse", img_size=64)
                inc_awq.quantize_inc_awq(model, _CalibrationCases(), 2, args)
                model.transformer.encoder.SELayer = None
                outputs = []
                handle = model.transformer.encoder.register_forward_hook(lambda m, i, o: outputs.append(o))
                try:
                    with torch.no_grad():
                        logits = model(torch.randn(1, 3, 64, 64, device="cuda"))[0]
                finally:
                    handle.remove()
                self.assertEqual(logits.shape, (1, 3, 64, 64))
                self.assertTrue(bool(torch.isfinite(logits).all()))
                for name in selected:
                    module = model.get_submodule(name)
                    self.assert_no_float_weight(getattr(module, "linear", module))
                if attention == "standard":
                    self.assertIsNone(outputs[0][2])
                else:
                    indices = outputs[0][2]
                    self.assertLess(indices.shape[1], 16)
                    self.assertTrue(bool(((indices >= 0) & (indices < 16)).all()))


if __name__ == "__main__":
    unittest.main()
