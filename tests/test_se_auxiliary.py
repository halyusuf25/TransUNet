"""Focused auxiliary SE checks using real model modules and synthetic inputs.

Run from the repository root with its training environment::

    .venv/bin/python -m unittest discover -s tests -p test_se_auxiliary.py -v

The deployment-reference test reads only AWQ's standalone quantization function
from its installed source; it does not import AWQ's deployment dependencies.
CUDA checks are explicitly skipped when the required hardware is unavailable.
"""

import ast
import copy
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from networks.se_block import SELayer
from networks.vit_seg_configs import get_r50_b16_config
from networks.vit_seg_modeling import Block, Encoder, VisionTransformer
from src.se_auxiliary import (
    build_quantization_targets,
    load_se_metadata,
    save_se_metadata,
    validate_se_calibration_config,
)
from src.se_aux_training import (
    compute_se_aux_loss,
    validate_se_aux_args,
    verify_se_optimizer,
)


def small_config(use_se=True, aux=True, attention="standard"):
    config = get_r50_b16_config()
    config.hidden_size = 128
    config.transformer.mlp_dim = 256
    config.transformer.num_heads = 4
    config.transformer.num_layers = 2
    config.transformer.dropout_rate = 0.0
    config.transformer.attention_dropout_rate = 0.0
    config.resnet.num_layers = (1, 1, 1)
    del config.resnet["width_factor"]
    config.resnet.width_factor = 0.5
    config.patches.grid = (2, 2)
    config.decoder_channels = (16, 8, 8, 4)
    config.n_classes = 3
    config.n_skip = 0
    for flag in ("use_swin", "use_efficientnet", "use_shsa",
                 "use_alternate_shsa", "use_ats", "use_gumbel_topk",
                 "verbose", "drop_se_block"):
        setattr(config, flag, False)
    config.topk_attn = 0.0 if attention == "standard" else 0.5
    config.use_ats = attention == "ats"
    config.use_gumbel_topk = attention == "gumbel"
    config.gumbel_sampling_mode = "manual"
    config.use_se_block = use_se
    config.se_aux_loss = aux
    config.se_aux_weight = 0.1
    config.se_pooling_mode = "rms" if aux else "mean"
    config.se_aux_bits = 4
    config.se_aux_group_size = 128
    config.se_aux_zero_point = True
    config.se_aux_eps = 1e-12
    return config


def small_model(aux=True):
    model = VisionTransformer(small_config(aux=aux), img_size=32, num_classes=3)
    # Ensure active ReLUs and a nondegenerate aggregate SE gradient. Individual
    # scalar gradients need not all be nonzero in normal training.
    with torch.no_grad():
        for layer in model.transformer.encoder.SELayer:
            layer.fc[0].weight.fill_(0.003)
            layer.fc[2].weight.fill_(0.02)
    return model


def se_parameters(model):
    return list(model.transformer.encoder.SELayer.parameters())


def base_loss(logits, labels):
    """Real segmentation CE plus multiclass soft Dice on synthetic masks."""
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(labels, logits.shape[1]).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    dice = (2 * (probabilities * one_hot).sum(dims) + 1e-5) / (
        probabilities.square().sum(dims) + one_hot.square().sum(dims) + 1e-5
    )
    return 0.5 * F.cross_entropy(logits, labels) + 0.5 * (1 - dice.mean())


class AuxiliarySETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(73)

    def test_all_four_flag_combinations(self):
        for use_se, aux in ((False, False), (True, False), (True, True)):
            with self.subTest(use_se=use_se, aux=aux):
                args = SimpleNamespace(use_se_block=use_se, se_aux_loss=aux,
                                       se_aux_weight=0.1, use_kd=False)
                validate_se_aux_args(args)
                model = Encoder(small_config(use_se, aux), vis=False).train()
                result = model(torch.randn(2, 5, 128), return_se_aux=aux)
                self.assertEqual(len(result), 5 if aux else 4)
                if use_se:
                    self.assertTrue(all(layer.pooling_mode == ("rms" if aux else "mean")
                                        for layer in model.SELayer))
                else:
                    self.assertEqual(result[3], [])
        with self.assertRaisesRegex(ValueError, "use_se_block"):
            validate_se_aux_args(SimpleNamespace(use_se_block=False, se_aux_loss=True,
                                                 se_aux_weight=0.1))

    def test_auxiliary_weight_validation(self):
        for weight in (0, -0.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                validate_se_aux_args(SimpleNamespace(use_se_block=True, se_aux_loss=True,
                                                     se_aux_weight=weight))
        validate_se_aux_args(SimpleNamespace(use_se_block=False, se_aux_loss=False,
                                             se_aux_weight=float("nan")))

    def test_rms_distinguishes_opposite_sign_amplitudes_and_keeps_input(self):
        rms = SELayer(16, pooling_mode="rms")
        mean = SELayer(16, pooling_mode="mean")
        with torch.no_grad():
            rms.fc[0].weight.fill_(0.05)
            rms.fc[2].weight.fill_(0.2)
        mean.load_state_dict(rms.state_dict())
        x = torch.tensor([1.0, -1.0]).reshape(1, 2, 1).expand(1, 2, 16).clone()
        y, gates = rms(x)
        y2, gates2 = rms(x * 2)
        self.assertIs(y, x)
        torch.testing.assert_close(y2, x * 2, rtol=0, atol=0)
        self.assertTrue(bool((gates2 > gates).all()))
        torch.testing.assert_close(mean(x)[1], mean(x * 2)[1], rtol=0, atol=0)
        torch.testing.assert_close(mean(x)[0], x * mean(x)[1], rtol=0, atol=0)

    def test_rms_float32_accumulation_and_mlp_dtype(self):
        layer = SELayer(16, pooling_mode="rms").half()
        x = torch.full((2, 8, 16), 400.0, dtype=torch.float16, requires_grad=True)
        captured = []
        hook = layer.fc.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
        try:
            _, gates = layer(x)
        finally:
            hook.remove()
        self.assertEqual(captured[0].dtype, torch.float16)
        torch.testing.assert_close(captured[0], torch.full((2, 16), 400.0, dtype=torch.float16))
        self.assertTrue(bool(torch.isfinite(gates).all()))
        gates.float().sum().backward()
        self.assertIsNone(x.grad)

    def test_targets_shape_detachment_finiteness_and_weight_immutability(self):
        for batch in (1, 3):
            with self.subTest(batch=batch):
                x = torch.randn(batch, 11, 128, requires_grad=True)
                w = torch.randn(96, 128, requires_grad=True)
                before_x, before_w = x.detach().clone(), w.detach().clone()
                target = build_quantization_targets(x, w)
                self.assertEqual(target.shape, (batch, 1, 128))
                self.assertEqual(target.dtype, torch.float32)
                self.assertFalse(target.requires_grad)
                self.assertIsNone(target.grad_fn)
                self.assertTrue(bool(torch.isfinite(target).all()))
                self.assertTrue(bool(((target > 0) & (target < 1)).all()))
                torch.testing.assert_close(x, before_x, rtol=0, atol=0)
                torch.testing.assert_close(w, before_w, rtol=0, atol=0)

    def test_zero_activations_or_zero_weight_error_produce_half(self):
        random_x, random_w = torch.randn(3, 7, 128), torch.randn(64, 128)
        for x, w in ((torch.zeros_like(random_x), random_w),
                     (random_x, torch.zeros_like(random_w))):
            torch.testing.assert_close(build_quantization_targets(x, w),
                                       torch.full((3, 1, 128), 0.5), rtol=0, atol=0)

    def test_targets_match_installed_deployment_quantizer(self):
        spec = importlib.util.find_spec("awq")
        if spec is None or not spec.submodule_search_locations:
            self.skipTest("Installed AWQ source unavailable for independent numerical comparison")
        paths = [Path(root) / "quantize" / "quantizer.py"
                 for root in spec.submodule_search_locations]
        path = next((candidate for candidate in paths if candidate.is_file()), None)
        if path is None:
            self.skipTest("Installed AWQ pseudo_quantize_tensor source unavailable")
        syntax = ast.parse(path.read_text())
        function = next(node for node in syntax.body
                        if isinstance(node, ast.FunctionDef) and node.name == "pseudo_quantize_tensor")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        reference = namespace["pseudo_quantize_tensor"]
        x = torch.randn(3, 7, 256)
        w = torch.randn(8, 256)
        w[0].zero_()
        w[1].fill_(0.3)
        w[2].fill_(-0.3)
        w[3].fill_(1e-8)
        for bits, group in ((4, 128), (4, 64), (8, 128)):
            with self.subTest(bits=bits, group=group):
                w_hat = reference(w.clone(), n_bit=bits, zero_point=True,
                                  q_group_size=group, inplace=False, get_scale_zp=False)
                error = (x.square().mean(1) * (w - w_hat).square().mean(0)).sqrt()
                expected = ((error + 1e-12) / (error + error.mean(-1, keepdim=True) + 2e-12)).unsqueeze(1)
                torch.testing.assert_close(build_quantization_targets(x, w, bits=bits, group_size=group),
                                           expected, rtol=0, atol=0)

    def test_mse_exact_image_channel_pairing_for_single_and_multiple_images(self):
        for batch in (1, 3):
            with self.subTest(batch=batch):
                result = Encoder(small_config(), vis=False).train()(
                    torch.randn(batch, 9, 128), return_se_aux=True)
                gates, targets = result[3:]
                self.assertEqual(len(gates), 2)
                for gate, target in zip(gates, targets):
                    self.assertEqual(gate.shape, (batch, 1, 128))
                    self.assertEqual(gate.shape, target.shape)
                explicit = sum((gate[b, 0, c] - target[b, 0, c]).square()
                               for gate, target in zip(gates, targets)
                               for b in range(batch) for c in range(128)) / (2 * batch * 128)
                torch.testing.assert_close(compute_se_aux_loss(gates, targets, expected_blocks=2), explicit)

    def test_auxiliary_loss_rejects_broadcasting_correspondence_and_nonfinite_values(self):
        gate = torch.full((2, 1, 128), 0.4, requires_grad=True)
        target = torch.full_like(gate, 0.5, requires_grad=False)
        invalid = (([gate], [target.squeeze(1)]), ([gate, gate], [target]),
                   ([gate], [target * float("nan")]), ([gate * float("inf")], [target]),
                   ([gate], [target.requires_grad_()]))
        for gates, targets in invalid:
            with self.subTest(shapes=[tuple(t.shape) for t in targets]), self.assertRaises((ValueError, RuntimeError)):
                compute_se_aux_loss(gates, targets)
        with self.assertRaises((ValueError, RuntimeError)):
            compute_se_aux_loss([gate], [target.detach()], expected_blocks=2)

    def test_supported_attention_targets_use_actual_projection_inputs_and_weights(self):
        for attention in ("standard", "topk", "gumbel", "ats"):
            with self.subTest(attention=attention):
                block = Block(small_config(attention=attention), vis=False).train()
                with torch.no_grad():
                    block.attention_norm.weight.copy_(torch.linspace(0.5, 1.5, 128))
                    block.ffn_norm.bias.copy_(torch.linspace(-0.2, 0.2, 128))
                captured = {}
                hooks = []
                names = ("query", "key", "value") if attention == "standard" else ("qkv",)
                def capture(name):
                    def hook(module, inputs):
                        captured[name] = inputs[0].detach().clone()
                    return hook
                for name in names:
                    hooks.append(getattr(block.attn, name).register_forward_pre_hook(capture(name)))
                hooks.append(block.ffn.fc1.register_forward_pre_hook(capture("fc1")))
                x = torch.randn(2, 24, 128)
                try:
                    result = block(x, return_se_aux=True)
                finally:
                    for hook in hooks:
                        hook.remove()
                qkv_input = captured[names[0]]
                for name in names:
                    torch.testing.assert_close(captured[name], block.attention_norm(x))
                qkv_weight = torch.cat([getattr(block.attn, name).weight for name in names], 0)
                target_qkv = build_quantization_targets(qkv_input, qkv_weight)
                target_fc1 = build_quantization_targets(captured["fc1"], block.ffn.fc1.weight)
                expected = 0.5 * (target_qkv + target_fc1)
                self.assertEqual(result[-1].shape, (2, 1, 128))
                torch.testing.assert_close(result[-1], expected, rtol=0, atol=0)
                self.assertEqual(qkv_input.shape[1], 24)
                self.assertEqual(captured["fc1"].shape[1], 24 if attention == "standard" else 12)

    def test_ats_variable_replica_token_counts_return_gatherable_training_outputs(self):
        config = small_config(attention="ats")
        config.patches.grid = (6, 6)
        uniform_model = VisionTransformer(config, img_size=96, num_classes=3).train()
        concentrated_model = copy.deepcopy(uniform_model)
        token_counts = []
        outputs = []
        for model, batch, concentrated in ((uniform_model, 2, False),
                                           (concentrated_model, 1, True)):
            counts = []
            def scores(attention, values):
                batch_size, _, tokens, _ = attention.shape
                result = attention.new_full((batch_size, tokens), 1.0 / tokens)
                if concentrated:
                    result.zero_()
                    result[:, 0] = 1
                return result
            def capture(module, inputs, output):
                counts.append(output[1].shape[-2])
            with ExitStack() as context:
                for block in model.transformer.encoder.layer:
                    context.enter_context(mock.patch.object(block.attn, "score_assignment_step", side_effect=scores))
                    handle = block.attn.register_forward_hook(capture)
                    context.callback(handle.remove)
                output = model(torch.randn(batch, 3, 96, 96), return_se_aux=True)
            token_counts.append(counts)
            outputs.append(output)
            self.assertEqual(output[1], [])
            self.assertEqual(output[0].shape, (batch, 3, 96, 96))
            self.assertTrue(all(g.shape == t.shape == (batch, 1, 128)
                                for g, t in zip(output[3], output[4])))
        self.assertEqual(token_counts, [[18, 10], [1, 1]])
        self.assertEqual(torch.cat([output[0] for output in outputs]).shape, (3, 3, 96, 96))
        gates = [torch.cat([output[3][block] for output in outputs]) for block in range(2)]
        targets = [torch.cat([output[4][block] for output in outputs]) for block in range(2)]
        expected = (2 * compute_se_aux_loss(outputs[0][3], outputs[0][4])
                    + compute_se_aux_loss(outputs[1][3], outputs[1][4])) / 3
        torch.testing.assert_close(compute_se_aux_loss(gates, targets), expected)
        with torch.no_grad():
            self.assertEqual(len(uniform_model.eval()(torch.randn(1, 3, 96, 96))[1]), 2)

    def test_partial_channel_shsa_is_rejected(self):
        config = small_config()
        config.use_shsa = True
        with self.assertRaisesRegex(ValueError, "SHSA|partial|channel"):
            Encoder(config, vis=False)

    def test_auxiliary_only_backward_reaches_se_and_no_main_parameters(self):
        model = small_model().train()
        result = model(torch.randn(2, 3, 32, 32), return_se_aux=True)
        loss = compute_se_aux_loss(result[3], result[4], expected_blocks=2)
        loss.backward()
        auxiliary_ids = {id(p) for p in se_parameters(model)}
        gradients = [p.grad for p in se_parameters(model)]
        self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters() if id(p) not in auxiliary_ids))

    def test_segmentation_only_backward_has_no_se_gradients(self):
        model = small_model().train()
        result = model(torch.randn(2, 3, 32, 32), return_se_aux=True)
        labels = torch.randint(3, (2, 32, 32))
        base_loss(result[0], labels).backward()
        self.assertTrue(all(p.grad is None for p in se_parameters(model)))
        self.assertGreater(model.segmentation_head[0].weight.grad.norm().item(), 0)

    def test_combined_step_updates_se_once_and_preserves_main_gradients_and_logits(self):
        model = small_model().train()
        reference = copy.deepcopy(model)
        images, labels = torch.randn(2, 3, 32, 32), torch.randint(3, (2, 32, 32))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        verify_se_optimizer(model, optimizer)
        before = [p.detach().clone() for p in se_parameters(model)]
        logits, _, _, gates, targets = model(images, return_se_aux=True)
        reference_logits = reference(images)[0]
        torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
        total = base_loss(logits, labels) + 0.1 * compute_se_aux_loss(gates, targets)
        total.backward()
        base_loss(reference_logits, labels).backward()
        for (name, parameter), (reference_name, reference_parameter) in zip(
                model.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, reference_name)
            if ".SELayer." in name:
                self.assertIsNone(reference_parameter.grad)
            elif parameter.grad is not None or reference_parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, se_parameters(model))))

    def test_optimizer_validation_detects_missing_and_duplicate_se_parameters(self):
        model = small_model()
        auxiliary_ids = {id(p) for p in se_parameters(model)}
        missing = torch.optim.SGD([p for p in model.parameters() if id(p) not in auxiliary_ids], lr=0.1)
        with self.assertRaises((ValueError, RuntimeError)):
            verify_se_optimizer(model, missing)
        duplicate = torch.optim.SGD(model.parameters(), lr=0.1)
        duplicate.param_groups[0]["params"].append(se_parameters(model)[0])
        with self.assertRaises((ValueError, RuntimeError)):
            verify_se_optimizer(model, duplicate)

    def test_unequal_replica_chunks_reduce_by_image_count(self):
        # DataParallel concatenates returned tensors before the trainer's loss.
        # Unequal chunks must not receive equal weight as replica-local means.
        gates = torch.tensor([0.1, 0.2, 0.9]).view(3, 1, 1).expand(3, 1, 128)
        targets = torch.zeros_like(gates)
        gathered_gates = torch.cat([gates[:2], gates[2:]], dim=0)
        gathered_targets = torch.cat([targets[:2], targets[2:]], dim=0)
        actual = compute_se_aux_loss([gathered_gates], [gathered_targets])
        expected = (0.1 ** 2 + 0.2 ** 2 + 0.9 ** 2) / 3
        self.assertAlmostEqual(actual.item(), expected, places=6)
        incorrect_equal_replica_mean = 0.5 * (gates[:2].square().mean() + gates[2:].square().mean())
        self.assertGreater(abs(actual.item() - incorrect_equal_replica_mean.item()), 0.1)

    def test_targets_recomputed_after_current_weight_change(self):
        encoder = Encoder(small_config(), vis=False).train()
        x = torch.randn(2, 8, 128)
        before = encoder(x, return_se_aux=True)[4][0]
        with torch.no_grad():
            encoder.layer[0].attn.query.weight[:, :64].mul_(9)
        after = encoder(x, return_se_aux=True)[4][0]
        self.assertFalse(torch.equal(before, after))

    def test_checkpoint_metadata_restores_rms_predictions_without_enabling_loss(self):
        model = small_model().eval()
        images = torch.randn(2, 3, 32, 32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save(model.state_dict(), path)
            save_se_metadata(model, path)
            restored = small_model(aux=False).eval()
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            load_se_metadata(restored, path)
            self.assertFalse(restored.transformer.encoder.se_aux_loss)
            self.assertTrue(all(layer.pooling_mode == "rms" for layer in restored.transformer.encoder.SELayer))
            with torch.no_grad():
                expected, actual = model(images), restored(images, return_se_aux=True)
            self.assertEqual(len(actual), 4)
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            for left, right in zip(actual[3], expected[3]):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            validate_se_calibration_config(restored, bits=4, group_size=128, zero_point=True)
            with self.assertRaises(ValueError):
                validate_se_calibration_config(restored, bits=8, group_size=128, zero_point=True)
            restored.train()
            self.assertEqual(len(restored(images)), 4)
            with self.assertRaisesRegex(ValueError, "auxiliary"):
                restored(images, return_se_aux=True)

    def test_legacy_checkpoint_without_metadata_uses_mean(self):
        model = small_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            torch.save(model.state_dict(), path)
            load_se_metadata(model, path)
            self.assertFalse(model.transformer.encoder.se_aux_loss)
            self.assertTrue(all(layer.pooling_mode == "mean" for layer in model.transformer.encoder.SELayer))

    def test_evaluation_skips_targets_and_se_removal_preserves_logits(self):
        model = small_model().eval()
        images = torch.randn(2, 3, 32, 32)
        with mock.patch("networks.vit_seg_modeling.build_quantization_targets",
                        side_effect=AssertionError("evaluation generated targets")):
            with torch.no_grad():
                before = model(images, return_se_aux=True)
                self.assertEqual(len(before), 4)
                # This is the production post-quantization removal used by test.py.
                del model.transformer.encoder.SELayer
                after = model(images, return_se_aux=True)
        torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
        self.assertEqual(after[3], [])
        self.assertFalse(any(isinstance(module, SELayer) for module in model.modules()))
        self.assertFalse(any("SELayer" in name for name, _ in model.named_parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable for autocast check")
    def test_cuda_autocast_preserves_finite_rms_gates_and_float32_targets(self):
        model = small_model().cuda().train()
        images = torch.randn(2, 3, 32, 32, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(images, return_se_aux=True)
            loss = compute_se_aux_loss(output[3], output[4])
        self.assertTrue(all(t.dtype == torch.float32 and not t.requires_grad for t in output[4]))
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in output[3]))
        loss.backward()
        gradients = [p.grad for p in se_parameters(model)]
        self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2,
                         "Two CUDA devices unavailable for uneven DataParallel batch")
    def test_dataparallel_uneven_batch_matches_global_image_mean(self):
        # Encoder avoids batch-dependent decoder BatchNorm; all outputs are real
        # encoder tensors. B=3 on two devices creates chunks of sizes 2 and 1.
        # Distinguish host NCCL/driver failures from model regressions. This
        # independent standard PyTorch module must work before testing our code.
        preflight = nn.DataParallel(nn.Linear(2, 2).cuda(0), device_ids=[0, 1])
        try:
            preflight(torch.ones(3, 2, device="cuda:0")).sum().backward()
        except RuntimeError as error:
            if "NCCL" in str(error):
                self.skipTest("Host NCCL unavailable in vanilla DataParallel preflight: " + str(error))
            raise
        del preflight
        encoder = Encoder(small_config(), vis=False).cuda(0).train()
        reference = copy.deepcopy(encoder)
        parallel = nn.DataParallel(encoder, device_ids=[0, 1])
        x = torch.randn(3, 9, 128, device="cuda:0")
        output = parallel(x, return_se_aux=True)
        expected = reference(x, return_se_aux=True)
        self.assertTrue(all(g.shape == t.shape == (3, 1, 128)
                            for g, t in zip(output[3], output[4])))
        loss = compute_se_aux_loss(output[3], output[4])
        reference_loss = compute_se_aux_loss(expected[3], expected[4])
        torch.testing.assert_close(loss, reference_loss)
        paired = torch.stack([(g - t).square().mean(dim=(1, 2))
                              for g, t in zip(output[3], output[4])]).mean()
        torch.testing.assert_close(loss, paired, rtol=0, atol=0)
        loss.backward()
        reference_loss.backward()
        for actual, wanted in zip(encoder.SELayer.parameters(), reference.SELayer.parameters()):
            torch.testing.assert_close(actual.grad, wanted.grad, rtol=2e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
