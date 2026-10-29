"""SE quantization layout and Gumbel replay regressions, without AWQ kernels.

Run with ``.venv/bin/python -m unittest discover -s tests -p
test_se_gumbel_quantization.py -v``. CUDA replay is checked when available.
"""

from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from networks.attention import TopkAttention
from test_se_calibration import quantizer_module


def _attention(mode="manual"):
    config = SimpleNamespace(
        transformer={"num_heads": 2, "attention_dropout_rate": 0.0},
        use_gumbel_topk=True, gumbel_sampling_mode=mode,
    )
    return TopkAttention(config, embed_dim=16, keep_rate=0.5, min_tokens=1)


def _block(fused):
    block = nn.Module()
    block.attention_norm = nn.LayerNorm(16)
    block.ffn_norm = nn.LayerNorm(16)
    block.attn = _attention() if fused else nn.Module()
    if not fused:
        for name in ("query", "key", "value", "out"):
            setattr(block.attn, name, nn.Linear(16, 16))
    block.ffn = nn.Module()
    block.ffn.fc1 = nn.Linear(16, 32)
    block.ffn.fc2 = nn.Linear(32, 16)
    return block


def _quantizer(model, loader=(), device="cpu"):
    return quantizer_module.AWQViTSegQuantizer(
        model, loader, q_group_size=8, n_calib_batches=2,
        device=torch.device(device), args=SimpleNamespace(use_se_block=True),
    )


class SEAttentionLayoutTests(unittest.TestCase):
    def test_candidate_quantizes_every_projection_with_only_input_scales(self):
        for fused in (True, False):
            with self.subTest(fused=fused):
                block = _block(fused)
                quantizer = _quantizer(block)
                scale = torch.linspace(0.5, 2.0, 16)
                before = {name: value.clone() for name, value in block.state_dict().items()}
                inputs = ("attn.qkv",) if fused else ("attn.query", "attn.key", "attn.value")
                output = "attn.proj" if fused else "attn.out"
                names = (*inputs, output, "ffn.fc1", "ffn.fc2")
                seen = []

                def quantize(weight):
                    seen.append(weight.clone())
                    return weight + 0.125

                quantizer._quantize_full_model_with_se_scales([block], [scale], quantize)
                self.assertEqual(len(seen), len(names))
                for name, received in zip(names, seen):
                    original = before[name + ".weight"]
                    scaled = name in (*inputs, "ffn.fc1")
                    torch.testing.assert_close(received, original * scale if scaled else original)
                    expected = original + (0.125 / scale if scaled else 0.125)
                    torch.testing.assert_close(quantizer._get_op_by_name(block, name).weight, expected)
                # Candidate evaluation cancels the input scale locally; final
                # conversion is responsible for folding it into LayerNorm.
                for name in ("attention_norm", "ffn_norm"):
                    torch.testing.assert_close(getattr(block, name).weight, before[name + ".weight"])

    def test_final_scale_folding_and_packing_cover_fused_and_separate_layouts(self):
        for fused in (True, False):
            with self.subTest(fused=fused):
                block = _block(fused)
                quantizer = _quantizer(block)
                scale = torch.linspace(0.5, 2.0, 16)
                with torch.no_grad():
                    block.attention_norm.bias.copy_(torch.linspace(-1, 1, 16))
                    block.ffn_norm.bias.copy_(torch.linspace(1, 2, 16))
                before = {name: value.clone() for name, value in block.state_dict().items()}
                inputs = ("attn.qkv",) if fused else ("attn.query", "attn.key", "attn.value")
                output = "attn.proj" if fused else "attn.out"
                names = (*inputs, output, "ffn.fc1", "ffn.fc2")
                sample = torch.randn(2, 6, 16)
                original_outputs = {
                    name: quantizer._get_op_by_name(block, name)(
                        block.ffn_norm(sample) if name == "ffn.fc1" else block.attention_norm(sample)
                    ).detach()
                    for name in (*inputs, "ffn.fc1")
                }
                with mock.patch.object(quantizer, "_search_best_se_scales_full_model", return_value=[scale]):
                    scales = quantizer._auto_se_scale([block])[0]
                quantizer._apply_scales(block, scales)
                for name in ("attention_norm", "ffn_norm"):
                    for affine in ("weight", "bias"):
                        torch.testing.assert_close(
                            getattr(getattr(block, name), affine), before[name + "." + affine] / scale,
                        )
                for name in names:
                    scaled = name in (*inputs, "ffn.fc1")
                    torch.testing.assert_close(
                        quantizer._get_op_by_name(block, name).weight,
                        before[name + ".weight"] * scale if scaled else before[name + ".weight"],
                    )
                for name, expected in original_outputs.items():
                    norm = block.ffn_norm if name == "ffn.fc1" else block.attention_norm
                    torch.testing.assert_close(quantizer._get_op_by_name(block, name)(norm(sample)), expected)
                originals = [quantizer._get_op_by_name(block, name) for name in names]
                replacements = [nn.Identity() for _ in names]
                with mock.patch.object(quantizer, "_linear_to_wqlinear", side_effect=replacements) as pack:
                    quantizer._convert_selected_linears_in_block(block)
                self.assertEqual([call.args[0] for call in pack.call_args_list], originals)
                for name, replacement in zip(names, replacements):
                    self.assertIs(quantizer._get_op_by_name(block, name), replacement)


def _rng_state():
    return [torch.get_rng_state().clone(), *(
        state.clone() for state in torch.cuda.get_rng_state_all()
    )] if torch.cuda.is_available() else [torch.get_rng_state().clone()]


class _RandomModel(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.blocks = nn.ModuleList([_attention(mode), _attention(mode)])
        self.register_buffer("candidate", torch.tensor(-1))
        self.records = []
        self.reference_rng = None
        self.fail_forward = False
        # Uniform attention with values concentrated on the first input channel
        # makes reference tokens predictably differ from candidate tokens.
        with torch.no_grad():
            self.blocks[0].qkv.weight.zero_()
            self.blocks[0].qkv.bias.zero_()
            self.blocks[0].qkv.weight[32:, 0] = 1
        for index, attention in enumerate(self.blocks):
            original = attention._gumbel_topk

            def record(scores, K, mode, original=original, index=index):
                draws = []
                if mode == "dist":
                    sampler = torch.distributions.Gumbel.sample

                    def sample(distribution, *args, **kwargs):
                        result = sampler(distribution, *args, **kwargs)
                        draws.append(result.detach().clone())
                        return result

                    patch = mock.patch.object(torch.distributions.Gumbel, "sample", sample)
                else:
                    sampler = torch.rand

                    def sample(*args, **kwargs):
                        result = sampler(*args, **kwargs)
                        draws.append(result.detach().clone())
                        return result

                    patch = mock.patch.object(torch, "rand", sample)
                with patch:
                    selected = original(scores, K=K, mode=mode)
                self.records.append((
                    int(self.candidate), self.case, index, draws[0],
                    scores.detach().clone(), selected.detach().clone(),
                ))
                return selected

            attention._gumbel_topk = record

    def forward(self, inputs):
        self.case = int(inputs[0, 0, 2])
        # Exercise CPU state even when attention samples on CUDA.
        torch.rand(1)
        x = inputs
        for attention in self.blocks:
            x = attention(x)[0]
        if self.fail_forward and self.candidate >= 0 and self.case == 1:
            raise RuntimeError("injected candidate forward failure")
        if self.candidate < 0:
            self.reference_rng = _rng_state()
        gates = [torch.full((1, 1, 16), self.case + 1.0, device=inputs.device) for _ in self.blocks]
        return {"logits": x, "se_scale": gates}


class SEGumbelReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def run_search(self, mode, device="cpu", failure=None):
        torch.manual_seed(57)
        model = _RandomModel(mode).eval().to(device)
        batches = []
        for case in range(2):
            inputs = torch.ones(1, 16, 16)
            inputs[:, :8, 1] = 1e-20
            inputs[:, 8:, 0] = 1e-20
            inputs[:, :, 2] = case
            batches.append({"image": inputs})
        quantizer = _quantizer(model, batches, device)
        quantizer._extract_input_tensor = lambda image: image
        original_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}

        @torch.no_grad()
        def install(blocks, scales, quantize):
            # Candidate setup also consumes randomness: its state must not leak
            # into the caller, or into the following replayed model forward.
            torch.rand(3)
            torch.rand(3, device=device)
            if failure == "install":
                raise RuntimeError("injected candidate installation failure")
            model.candidate.fill_(int(scales[0].flatten()[0]) - 1)
            blocks[0].qkv.weight[32:, 0] = 0
            blocks[0].qkv.weight[32:, 1] = 1

        model.fail_forward = failure == "forward"
        with mock.patch.object(quantizer, "_quantize_full_model_with_se_scales", side_effect=install):
            if failure:
                with self.assertRaisesRegex(RuntimeError, "injected candidate"):
                    quantizer._search_best_se_scales_full_model(list(model.blocks))
            else:
                quantizer._search_best_se_scales_full_model(list(model.blocks))
        for actual, expected in zip(_rng_state(), model.reference_rng):
            self.assertTrue(torch.equal(actual, expected), "candidate calibration leaked RNG state")
        for name, expected in original_state.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
        return model.records

    def check_replay(self, mode, device):
        records = self.run_search(mode, device)
        references = {(case, block): (noise, scores, selected)
                      for candidate, case, block, noise, scores, selected in records if candidate < 0}
        candidates = [record for record in records if record[0] >= 0]
        self.assertEqual(len(references), 4)
        self.assertEqual(len(candidates), 8)
        self.assertFalse(torch.equal(references[0, 0][0], references[1, 0][0]))
        for candidate, case, block, noise, scores, selected in candidates:
            fp_noise, fp_scores, fp_selected = references[case, block]
            torch.testing.assert_close(noise, fp_noise, rtol=0, atol=0)
            if block == 0:
                self.assertFalse(torch.equal(scores, fp_scores))
                self.assertFalse(torch.equal(selected.sort().values, fp_selected.sort().values))

    def test_cpu_replays_each_case_and_recomputes_gumbel_choices(self):
        for mode in ("dist", "manual"):
            with self.subTest(mode=mode):
                self.check_replay(mode, "cpu")

    def test_cpu_restores_rng_after_candidate_failures(self):
        for mode in ("dist", "manual"):
            for failure in ("install", "forward"):
                with self.subTest(mode=mode, failure=failure):
                    self.run_search(mode, failure=failure)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_replay_and_exception_restoration(self):
        for mode in ("dist", "manual"):
            with self.subTest(mode=mode):
                self.check_replay(mode, "cuda")
                self.run_search(mode, "cuda", failure="forward")


if __name__ == "__main__":
    unittest.main()
