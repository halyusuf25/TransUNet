"""Exercise sampling, shared input reuse, and reporting through the comparison CLI."""

import contextlib
import io
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.se_auxiliary import DEFAULT_SE_AUX_CONFIG
from tools import compare_se_proxy


class CompareSEProxyTest(unittest.TestCase):
    def test_multiple_seed_cli_repeats_winner_search_with_matched_draws(self):
        seeds = (31, 17)
        slices = ["a", "b", "c", "d", "e"]
        metrics = {
            (31, "se"): (0.90, 0.60, 4), (31, "direct_proxy"): (0.78, 0.65, 5),
            (17, "se"): (0.80, 0.50, 8), (17, "direct_proxy"): (0.82, 0.45, 5),
        }
        calls = []

        def winner_search(args, source, inputs, model):
            calls.append((args.calibration_seed, source, inputs, model))
            self.assertEqual((args.checkpoint, args.dataset, args.volume_path, args.list_dir),
                             (checkpoint, "Synapse", str(test_root), str(root)))
            self.assertFalse(model.converted)
            model.converted = True
            dice, iou, hd95 = metrics[args.calibration_seed, source]
            winner = 0 if source == "se" else 1
            return winner, 0.2, dict(mean_dice=dice, mean_iou=iou, mean_hd95=hd95), DEFAULT_SE_AUX_CONFIG

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as mocks:
            root = Path(tmp)
            checkpoint, test_root, output_dir = root / "weights.pth", root / "test", root / "runs"
            (root / "train.txt").write_text("e\nb\na\nc\nd\nb\n")
            for i, name in enumerate(slices):
                image = np.full((16, 16), i, dtype=np.float32)
                np.savez(root / (name + ".npz"), image=image, label=np.zeros_like(image))
            mocks.enter_context(patch.object(compare_se_proxy, "run_method", side_effect=winner_search))
            mocks.enter_context(patch.object(compare_se_proxy, "load_rms_model",
                                             side_effect=lambda args: SimpleNamespace(converted=False)))
            mocks.enter_context(patch("torch.cuda.is_available", return_value=True))
            stdout = mocks.enter_context(contextlib.redirect_stdout(io.StringIO()))
            compare_se_proxy.main([
                "--dataset", "Synapse", "--checkpoint", str(checkpoint),
                "--multiple_seeds", "--calibration-seeds", "31", "17", "--calibration-size", "2",
                "--train-root", tmp, "--list-dir", tmp, "--volume-path", str(test_root),
                "--img-size", "16", "--output-dir", str(output_dir),
            ])

            self.assertEqual([(seed, method) for seed, method, _, _ in calls],
                             [(seed, method) for seed in seeds for method in ("se", "direct_proxy")])
            self.assertEqual(len({id(model) for _, _, _, model in calls}), 4)
            for seed, se, proxy in zip(seeds, calls[::2], calls[1::2]):
                ids = random.Random(seed).sample(slices, 2)
                self.assertIs(se[2], proxy[2])
                self.assertEqual([tensor[0, 0, 0, 0].item() for tensor in se[2]], [slices.index(i) for i in ids])
                report = (output_dir / f"seed_{seed}" / "results.md").read_text()
                self.assertIn(f"Ordered calibration IDs: `{ids[0]}`, `{ids[1]}`", report)
                self.assertIn(f"| SE-AUX | `{ids[0]}` |", report)
                self.assertIn(f"| Direct proxy | `{ids[1]}` |", report)

            summary = (output_dir / "summary.md").read_text()
            self.assertIn(summary, stdout.getvalue())
            lines = [line.strip("|").split("|") for line in summary.splitlines() if line.startswith("| ")]
            rows = {cells[0].strip(): [cell.strip() for cell in cells[1:]] for cells in lines}
            expected_rows = {
                "[31](seed_31/results.md)": [90, 78, 12, 60, 65, -5, 4, 5, -1],
                "[17](seed_17/results.md)": [80, 82, -2, 50, 45, 5, 8, 5, 3],
                "**Mean**": [85, 80, 5, 55, 55, 0, 6, 5, 1],
                "**Sample SD**": [7.0711, 2.8284, 9.8995, 7.0711, 14.1421, 7.0711, 2.8284, 0, 2.8284],
            }
            for label, expected in expected_rows.items():
                with self.subTest(row=label):
                    np.testing.assert_allclose([float(cell) for cell in rows[label]], expected, rtol=0, atol=0.0001)
            self.assertTrue(rows["[31](seed_31/results.md)"][2].startswith("+"))

    def test_multiple_seed_defaults_and_sample_sd_for_one_draw(self):
        required = ["--dataset", "Synapse", "--checkpoint", "weights.pth"]
        options = compare_se_proxy.parse_args(required + ["--multiple_seeds"])
        self.assertEqual(list(options.calibration_seeds), [1234, 1235, 1236, 1237, 1238])
        for extra in (["--case_by_case"], ["--calibration-seeds", "7", "7"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                compare_se_proxy.parse_args(required + ["--multiple_seeds"] + extra)
        args = compare_se_proxy.experiment_args(options)
        result = (0, 0.2, dict(mean_dice=0.75, mean_iou=0.5, mean_hd95=1), DEFAULT_SE_AUX_CONFIG)
        summary = compare_se_proxy.format_seed_summary(args, [(7, result, result)])
        sd = next(line for line in summary.splitlines() if line.startswith("| **Sample SD**"))
        self.assertEqual([cell.strip() for cell in sd.strip("|").split("|")[1:]], ["N/A"] * 9)

    def test_paired_report_matches_ids_and_averages_test_metrics(self):
        args = compare_se_proxy.experiment_args(compare_se_proxy.parse_args([
            "--dataset", "Synapse", "--checkpoint", "weights.pth",
        ]))
        def result(index, mse, dice, iou, hd95):
            metrics = dict(mean_dice=dice, mean_iou=iou, mean_hd95=hd95)
            return index, mse, metrics, DEFAULT_SE_AUX_CONFIG

        ids = ("slice_b", "slice_a")
        se = [result(0, 0.25, 0.75, 0.55, 3), result(1, 0.5, 0.65, 0.45, 5)]
        proxy = [result(1, 0.125, 0.70, 0.40, 7), result(0, 0.375, 0.70, 0.60, 2)]
        report = compare_se_proxy.format_paired_report(args, ids, se, proxy)
        self.assertIn("| `slice_b` | 0.25 | 0.375 | 75.0000 | 70.0000 | +5.0000 | "
                      "55.0000 | 60.0000 | -5.0000 | 3.0000 | 2.0000 | +1.0000 |", report)
        self.assertIn("| `slice_a` | 0.5 | 0.125 | 65.0000 | 70.0000 | -5.0000 | "
                      "45.0000 | 40.0000 | +5.0000 | 5.0000 | 7.0000 | -2.0000 |", report)
        self.assertIn("| **Paired mean** | 0.375 | 0.25 | 70.0000 | 70.0000 | +0.0000 | "
                      "50.0000 | 50.0000 | +0.0000 | 4.0000 | 4.5000 | -0.5000 |", report)
        self.assertIn("SE-AUX with `slice_b` (0.25) and direct proxy with `slice_a` (0.125)", report)
        with self.assertRaisesRegex(ValueError, "one result per calibration image"):
            compare_se_proxy.format_paired_report(args, ids, se, [proxy[0], proxy[0]])

    def test_acdc_lists_select_only_members_of_each_patient_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = compare_se_proxy.experiment_args(compare_se_proxy.parse_args([
                "--dataset", "ACDC", "--checkpoint", "unused.pth", "--list-dir", tmp,
            ]))
            for split, folder, suffix, patients in (
                ("train", "ACDC_training_slices", "_frame01_slice_0", ("021", "022", "001")),
                ("test_vol", "ACDC_training_volumes", "_frame01", ("001", "002", "021")),
            ):
                with self.subTest(split=split):
                    (root / folder).mkdir()
                    names = [f"patient{patient}{suffix}" for patient in patients]
                    for name in names:
                        (root / folder / (name + ".h5")).touch()
                    manifest = root / (split + ".txt")
                    manifest.write_text(f"{names[1]}.h5\n{names[0]}\n{names[1]}\n")
                    dataset = args.Dataset(base_dir=tmp, split=split, list_dir=args.list_dir)
                    self.assertEqual(dataset.sample_list, [names[1] + ".h5", names[0] + ".h5"])
                    manifest.write_text(names[2])
                    with self.assertRaisesRegex(ValueError, "outside the available ACDC " + split):
                        args.Dataset(base_dir=tmp, split=split, list_dir=args.list_dir)

    def test_one_invocation_shares_sampled_tensors_and_reports_signed_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = ["slice_c", "slice_a", "slice_b", "slice_a"]
            (root / "train.txt").write_text("\n".join(ids))
            for index, name in enumerate(sorted(set(ids))):
                image = np.full((16, 16), index + 1, dtype=np.float32)
                np.savez(root / (name + ".npz"), image=image, label=np.zeros_like(image))

            expected = random.Random(11).sample(sorted(set(ids)), 3)
            se_metrics = dict(mean_dice=0.875, mean_iou=0.65, mean_hd95=3)
            proxy_metrics = dict(mean_dice=0.75, mean_iou=0.7, mean_hd95=4)
            scores = [(0, 0.5, se_metrics, DEFAULT_SE_AUX_CONFIG),
                      (1, 0.125, proxy_metrics, DEFAULT_SE_AUX_CONFIG)]
            output_dir = root / "comparison report"
            stdout = io.StringIO()
            with contextlib.ExitStack() as mocks:
                run = mocks.enter_context(patch.object(compare_se_proxy, "run_method", side_effect=scores))
                load = mocks.enter_context(patch.object(compare_se_proxy, "load_rms_model", side_effect=[object(), object()]))
                mocks.enter_context(patch("torch.cuda.is_available", return_value=True))
                mocks.enter_context(contextlib.redirect_stdout(stdout))
                compare_se_proxy.main([
                    "--dataset", "Synapse", "--checkpoint", str(root / "weights.pth"),
                    "--train-root", tmp, "--list-dir", tmp, "--volume-path", str(root / "test"),
                    "--calibration-size", "3", "--calibration-seed", "11",
                    "--img_size", "16", "--output-dir", str(output_dir),
                ])

            se, proxy = run.call_args_list
            self.assertEqual(load.call_count, 2)
            self.assertIsNot(se.args[3], proxy.args[3])
            self.assertEqual((se.args[1], proxy.args[1]), ("se", "direct_proxy"))
            self.assertIs(se.args[2], proxy.args[2])
            for name, tensor in zip(expected, se.args[2]):
                self.assertEqual(tuple(tensor.shape), (1, 1, 16, 16))
                self.assertTrue(tensor.eq(sorted(set(ids)).index(name) + 1).all())

            report = (output_dir / "results.md").read_text()
            self.assertTrue(stdout.getvalue().endswith(report))
            self.assertIn(f"| SE-AUX | `{expected[0]}` |", report)
            self.assertIn(f"| Direct proxy | `{expected[1]}` |", report)
            self.assertIn("| SE-AUX minus direct proxy | — | +0.375 | +12.5000 pp | -5.0000 pp | -1.0000 |", report)

    def test_paired_cli_scores_once_then_packs_fresh_models_for_every_image(self):
        import torch
        import test as evaluator
        from src.se_auxiliary import save_se_metadata
        from test_se_auxiliary import small_config

        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        collections = []
        packed_models = []
        scored_models = []
        evaluated = []

        def make_quantizer(args, source, inputs, model):
            collections.append(inputs)
            marker = model.transformer.encoder.layer[0].ffn_norm.weight.data
            self.assertEqual(marker[0].item(), 0.125)
            self.assertTrue(hasattr(model.transformer.encoder, "SELayer"))
            self.assertFalse(model.training)

            def score():
                scored_models.append((source, model))
                candidates = []
                for image in inputs:
                    code = int(image[0, 0, 0, 0]) + (2 if source == "direct_proxy" else 0)
                    candidates.append(SimpleNamespace(scales=code, mean_mse=code / 100))
                return candidates

            def pack(*, candidate_scales):
                packed_models.append(model)
                marker.fill_(candidate_scales)
                return model

            return SimpleNamespace(score_candidates=score, quantize=pack)

        def inference(args, model):
            self.assertFalse(hasattr(model.transformer.encoder, "SELayer"))
            code = model.transformer.encoder.layer[0].ffn_norm.weight[0].item()
            evaluated.append(code)
            return dict(mean_dice=code / 10, mean_iou=code / 20, mean_hd95=code)

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as mocks:
            root = Path(tmp)
            ids = ["slice_a", "slice_b"]
            (root / "train.txt").write_text("\n".join(ids))
            for index, name in enumerate(ids, start=1):
                image = np.full((32, 32), index, dtype=np.float32)
                np.savez(root / (name + ".npz"), image=image, label=np.zeros_like(image))
            checkpoint = root / "weights.pth"
            argv = ["--dataset", "Synapse", "--case_by_case", "--checkpoint", str(checkpoint),
                    "--list-dir", tmp, "--train-root", tmp, "--volume-path", str(root / "test"),
                    "--calibration-size", "2", "--calibration-seed", "7", "--img-size", "32"]
            mocks.enter_context(patch.object(compare_se_proxy, "ROOT", root))
            mocks.enter_context(patch.dict(evaluator.CONFIGS_ViT_seg, {"R50-ViT-B_16": small_config()}))
            fp_model = evaluator.build_test_model(compare_se_proxy.experiment_args(compare_se_proxy.parse_args(argv)))
            fp_model.transformer.encoder.layer[0].ffn_norm.weight.data.fill_(0.125)
            torch.save(fp_model.state_dict(), checkpoint)
            save_se_metadata(fp_model, checkpoint)
            mocks.enter_context(patch("torch.cuda.is_available", return_value=True))
            mocks.enter_context(patch.object(compare_se_proxy, "make_quantizer", side_effect=make_quantizer))
            mocks.enter_context(patch.object(evaluator, "inference", side_effect=inference))
            stdout = mocks.enter_context(contextlib.redirect_stdout(io.StringIO()))
            compare_se_proxy.main(argv)
            report = (root / "benchmark/Synapse/compare_se_proxy/case_by_case/results.md").read_text()
            self.assertTrue(stdout.getvalue().endswith(report))

        self.assertEqual([source for source, _ in scored_models], ["se", "direct_proxy"])
        all_models = packed_models + [model for _, model in scored_models]
        self.assertEqual(len({id(model) for model in all_models}), 6)
        self.assertTrue(all(inputs is collections[0] for inputs in collections))
        ordered_ids = random.Random(7).sample(ids, 2)
        codes = [ids.index(name) + 1 for name in ordered_ids]
        self.assertEqual(evaluated, codes + [code + 2 for code in codes])
        rows = [line for line in report.splitlines() if line.startswith("| `")]
        self.assertEqual(len(rows), len(ordered_ids))
        for row, name, code in zip(rows, ordered_ids, codes):
            self.assertTrue(row.startswith(f"| `{name}` | {code / 100} | {(code + 2) / 100} | "
                                           f"{10 * code:.4f} | {10 * (code + 2):.4f} | -20.0000 |"))


if __name__ == "__main__":
    unittest.main()
