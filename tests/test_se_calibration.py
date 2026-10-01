"""CPU checks of production SE candidate selection with synthetic segmentation cases.

Run with::

    .venv/bin/python -m unittest discover -s tests -p test_se_calibration.py -v

AWQ imports are mocked only while loading a private copy of the production
module; these tests do not require AWQ's deployment dependencies or CUDA.
"""

import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn


def _load_quantizer():
    name = "_se_calibration_test_quantizer"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "networks" / "quantizer.py"
    )
    module = importlib.util.module_from_spec(spec)
    dependencies = {
        path: ModuleType(path) for path in (
            "awq", "awq.quantize", "awq.quantize.auto_scale",
            "awq.quantize.quantizer", "awq.quantize.qmodule",
        )
    }
    dependencies["awq.quantize.auto_scale"].auto_scale_block = mock.Mock()
    dependencies["awq.quantize.quantizer"].pseudo_quantize_tensor = mock.Mock()
    dependencies["awq.quantize.qmodule"].WQLinear = mock.Mock()
    with mock.patch.dict(sys.modules, {name: module, **dependencies}):
        spec.loader.exec_module(module)
    return module


quantizer_module = _load_quantizer()


def _gates(case, slices):
    # Different widths and original slice dimensions expose block mixing or
    # premature reduction to a channel vector.
    return [
        torch.full((slices, 1, 2), float(case + 1), requires_grad=True),
        torch.full((slices, 1, 3), float(case + 11), requires_grad=True),
    ]


class _Cases:
    def __init__(self, sizes):
        self.sizes = sizes
        self.iterations = 0
        self.consumed = 0

    def __iter__(self):
        self.iterations += 1
        for case, slices in enumerate(self.sizes):
            self.consumed += 1
            # Synapse loader format: [batch=1, slices, height, width].
            yield {"image": torch.full((1, slices, 2, 2), float(case))}


class _SegmentationModel(nn.Module):
    def __init__(self, mse, events):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0, 3.0]))
        self.register_buffer("candidate", torch.tensor(-1))
        self.register_buffer("offset", torch.tensor(7.0))
        self.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.mse = mse
        self.events = events
        self.fp_gates = []
        self.fail_quantized_forward = None
        self.fail_reference_forward = None

    def forward(self, image):
        assert not self.training, "calibration must preserve evaluation mode"
        assert not torch.is_grad_enabled(), "calibration must disable gradients"
        case = int(image.flatten()[0].item())
        candidate = int(self.candidate.item())
        if candidate < 0:
            self.events.append(("fp", case))
            if self.fail_reference_forward == case:
                self.weight.add_(50)
                self.offset.add_(50)
                raise RuntimeError("injected reference failure")
            # This aliases a CPU parameter: cached references must be cloned
            # before temporary quantization mutates the original storage.
            logits = self.weight[0].expand_as(image)
            gates = _gates(case, image.shape[0])
            self.fp_gates.append(gates)
        else:
            self.events.append(("q", candidate, case))
            if self.fail_quantized_forward == (candidate, case):
                self.weight.add_(50)
                self.offset.add_(50)
                raise RuntimeError("injected forward failure")
            logits = torch.full_like(image, 2.0 + math.sqrt(self.mse[candidate][case]))
            # Evaluation gates deliberately differ from the installed candidate.
            gates = [torch.full_like(gate, -999) for gate in _gates(case, image.shape[0])]
        return {"logits": logits, "se_scale": gates}


class SECalibrationTests(unittest.TestCase):
    def fixture(self, mse, sizes=None, limit=None):
        sizes = tuple(sizes if sizes is not None else [1] * len(mse))
        events = []
        model = _SegmentationModel(mse, events).eval()
        loader = _Cases(sizes)
        quantizer = quantizer_module.AWQViTSegQuantizer(
            model, loader, n_calib_batches=limit if limit is not None else len(sizes),
            device=torch.device("cpu"), args=SimpleNamespace(dataset="Synapse", img_size=2),
        )
        original = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        fixture = SimpleNamespace(
            model=model, loader=loader, quantizer=quantizer, events=events,
            original=original, sizes=sizes, fail_quantization=None,
        )

        def install(blocks, scales, w_quantize_func):
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(list(blocks), list(model.blocks))
            self.assert_restored(fixture)
            candidate = int(scales[0].flatten()[0].item()) - 1
            self.assert_gates(scales, candidate, sizes[candidate])
            fixture.events.append(("install", candidate))
            # Exercise the actual w_quantize_func wrapper and its detachment.
            quantized = w_quantize_func(model.weight)
            self.assertFalse(quantized.requires_grad)
            model.weight.add_(100)
            model.offset.add_(100)
            model.candidate.fill_(candidate)
            if fixture.fail_quantization == candidate:
                raise RuntimeError("injected quantization failure")

        fixture.install = mock.Mock(side_effect=install)
        quantizer._quantize_full_model_with_se_scales = fixture.install
        return fixture

    def search(self, fixture):
        with mock.patch.object(
            quantizer_module, "pseudo_quantize_tensor", return_value=fixture.model.weight
        ) as pseudo_quantize:
            result = fixture.quantizer._search_best_se_scales_full_model(list(fixture.model.blocks))
        self.assertEqual(pseudo_quantize.call_count, fixture.install.call_count)
        for call in pseudo_quantize.call_args_list:
            self.assertEqual(call.kwargs, {"n_bit": 4, "zero_point": True, "q_group_size": 128})
        return result

    def assert_restored(self, fixture):
        for name, expected in fixture.original.items():
            torch.testing.assert_close(fixture.model.state_dict()[name], expected, rtol=0, atol=0)
        self.assertFalse(fixture.model.training)
        self.assertIsNone(fixture.model.weight.grad)

    def assert_gates(self, actual, case, slices):
        self.assertIsInstance(actual, list)
        self.assertEqual(len(actual), 2)
        for tensor, expected in zip(actual, _gates(case, slices)):
            self.assertEqual(tensor.device.type, "cpu")
            self.assertFalse(tensor.requires_grad)
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)

    def test_every_candidate_scores_every_case_with_equal_case_weight(self):
        mse = [[0.010, 0.300, 0.500], [0.015, 0.080, 0.100], [0.020, 0.120, 0.090]]
        fixture = self.fixture(mse, sizes=(1, 2, 100))
        result = self.search(fixture)
        self.assert_gates(result, case=1, slices=2)
        self.assertEqual(fixture.install.call_count, 3)
        expected_events = [("fp", case) for case in range(3)]
        for candidate in range(3):
            expected_events.append(("install", candidate))
            expected_events.extend(("q", candidate, case) for case in range(3))
        self.assertEqual(fixture.events, expected_events)
        self.assertEqual((fixture.loader.iterations, fixture.loader.consumed), (1, 3))
        self.assert_restored(fixture)
        # The old diagonal comparison chooses A; element weighting chooses C.
        self.assertEqual(min(range(3), key=lambda j: mse[j][j]), 0)
        self.assertEqual(min(range(3), key=lambda j: sum(
            error * slices for error, slices in zip(mse[j], fixture.sizes)
        )), 2)
        for original_gate in fixture.model.fp_gates[1]:
            with torch.no_grad():
                original_gate.fill_(-50)
        self.assert_gates(result, case=1, slices=2)

    def test_limit_does_not_consume_an_extra_case(self):
        fixture = self.fixture([[0.1, 0.2], [0.3, 0.4]], sizes=(1, 2, 3), limit=2)
        self.assert_gates(self.search(fixture), case=0, slices=1)
        self.assertEqual((fixture.loader.iterations, fixture.loader.consumed), (1, 2))
        self.assertEqual(fixture.install.call_count, 2)
        self.assertEqual(sum(event[0] == "q" for event in fixture.events), 4)
        self.assert_restored(fixture)

    def test_single_case_and_loader_shorter_than_limit(self):
        fixture = self.fixture([[0.3]], sizes=(4,), limit=8)
        self.assert_gates(self.search(fixture), case=0, slices=4)
        self.assertEqual(fixture.events, [("fp", 0), ("install", 0), ("q", 0, 0)])
        self.assert_restored(fixture)

    def test_first_candidate_wins_an_exact_tie(self):
        fixture = self.fixture([[0.25, 0.25], [0.25, 0.25]], sizes=(1, 3))
        self.assert_gates(self.search(fixture), case=0, slices=1)
        self.assert_restored(fixture)

    def test_logits_are_cast_to_float32_before_subtraction(self):
        fixture = self.fixture([[0.0]])
        original_forward = fixture.model.forward

        def half_logits(image):
            output = original_forward(image)
            value = 60000 if fixture.model.candidate.item() < 0 else -60000
            output["logits"] = torch.full_like(output["logits"], value, dtype=torch.float16)
            return output

        # Subtracting in float16 would overflow and wrongly reject this candidate.
        with mock.patch.object(fixture.model, "forward", side_effect=half_logits):
            self.assert_gates(self.search(fixture), case=0, slices=1)
        self.assert_restored(fixture)

    def test_nonpositive_limits_are_rejected_without_consuming_cases(self):
        for limit in (0, -1):
            with self.subTest(limit=limit):
                fixture = self.fixture([[0.1]], limit=limit)
                with self.assertRaisesRegex(ValueError, "positive|greater than|at least"):
                    self.search(fixture)
                self.assertEqual(fixture.loader.consumed, 0)
                fixture.install.assert_not_called()
                self.assert_restored(fixture)

    def test_empty_subset_has_clear_error(self):
        fixture = self.fixture([], sizes=(), limit=3)
        with self.assertRaisesRegex(RuntimeError, "empty|[Nn]o.*calibration"):
            self.search(fixture)
        fixture.install.assert_not_called()
        self.assert_restored(fixture)

    def test_nonfinite_case_errors_are_not_omitted(self):
        for invalid in (float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                fixture = self.fixture([[0.0, invalid], [0.25, 0.25]])
                self.assert_gates(self.search(fixture), case=1, slices=1)
                self.assertEqual(sum(event[0] == "q" for event in fixture.events), 4)
                self.assert_restored(fixture)

    def test_no_finite_candidate_has_clear_error_and_restores_state(self):
        fixture = self.fixture([[float("nan"), 0.0], [0.0, float("inf")]])
        with self.assertRaisesRegex(RuntimeError, "finite"):
            self.search(fixture)
        self.assertEqual(sum(event[0] == "q" for event in fixture.events), 4)
        self.assert_restored(fixture)

    def test_state_restored_on_quantization_and_forward_failures(self):
        for failure in ("quantization", "forward", "reference"):
            with self.subTest(failure=failure):
                fixture = self.fixture([[0.1, 0.2], [0.3, 0.4]])
                if failure == "quantization":
                    fixture.fail_quantization = 0
                elif failure == "forward":
                    fixture.model.fail_quantized_forward = (0, 1)
                else:
                    fixture.model.fail_reference_forward = 1
                with self.assertRaisesRegex(RuntimeError, "injected .*failure"):
                    self.search(fixture)
                self.assert_restored(fixture)


if __name__ == "__main__":
    unittest.main()
