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


def markdown_rows(report):
    rows = {}
    for line in report.splitlines():
        if line.startswith("| "):
            label, *cells = (cell.strip() for cell in line.strip("|").split("|"))
            rows[label] = cells
    return rows


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
            rows = markdown_rows(summary)
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
        options = compare_se_proxy.parse_args(required + ["--multiple_seeds", "--case_by_case"])
        self.assertEqual(list(options.calibration_seeds), [1234, 1235, 1236, 1237, 1238])
        self.assertTrue(options.case_by_case)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            compare_se_proxy.parse_args(required + ["--multiple_seeds", "--calibration-seeds", "7", "7"])
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
        rows = markdown_rows(report)
        np.testing.assert_allclose([float(v) for v in rows["`slice_b`"]],
                                   [.25, .375, -.125, 75, 70, 5, 55, 60, -5, 3, 2, 1])
        np.testing.assert_allclose([float(v) for v in rows["`slice_a`"]],
                                   [.5, .125, .375, 65, 70, -5, 45, 40, 5, 5, 7, -2])
        mean = rows["**Paired mean**"]
        np.testing.assert_allclose([float(v) for v in mean], [.375, .25, .125, 70, 70, 0, 50, 50, 0, 4, 4.5, -.5])
        for metric in ("Calibration logit MSE", "Test Dice", "Test IoU", "HD95"):
            self.assertEqual(rows[metric], ["1", "1", "0"])
        with self.assertRaisesRegex(ValueError, "one result per calibration image"):
            compare_se_proxy.format_paired_report(args, ids, se, [proxy[0], proxy[0]])

    def test_matched_summary_averages_within_seeds_before_sample_sd_and_win_counts(self):
        args = compare_se_proxy.experiment_args(compare_se_proxy.parse_args([
            "--dataset", "Synapse", "--checkpoint", "weights.pth", "--calibration-size", "2",
        ]))

        def result(index, mse, dice, iou, hd95):
            metrics = dict(mean_dice=dice, mean_iou=iou, mean_hd95=hd95)
            return index, mse, metrics, DEFAULT_SE_AUX_CONFIG

        proxy = [result(i, .125, .5, .5, 4) for i in range(2)]
        se_first = [result(0, .0625, .625, .5, 3), result(1, .125, .375, .625, 5)]
        se_second = [result(i, .25, .75, .375, 2) for i in range(2)]
        runs = [(31, se_first, proxy), (17, se_second, proxy)]
        rows = markdown_rows(compare_se_proxy.format_matched_seed_summary(args, runs))
        seed_means = [[-.03125, 0, 6.25, 0], [.125, 25, -12.5, -2]]
        for seed, expected in zip((31, 17), seed_means):
            np.testing.assert_allclose([float(v) for v in rows[str(seed)]], expected)
        stats = [[float(v) for v in cell.split("±")] for cell in rows["**Mean ± sample SD**"]]
        np.testing.assert_allclose(np.asarray(stats)[:, 0], np.mean(seed_means, axis=0), atol=.0001)
        np.testing.assert_allclose(np.asarray(stats)[:, 1], np.std(seed_means, axis=0, ddof=1), atol=.0001)
        self.assertEqual(rows["Matched images"], ["1 / 2 / 1", "3 / 1 / 0", "1 / 2 / 1", "3 / 1 / 0"])
        self.assertEqual(rows["Seed means"], ["1 / 1 / 0", "1 / 0 / 1", "1 / 1 / 0", "1 / 0 / 1"])
        single = markdown_rows(compare_se_proxy.format_matched_seed_summary(args, runs[:1]))
        self.assertTrue(all(cell.endswith("± N/A") for cell in single["**Mean ± sample SD**"]))

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

    def test_case_by_case_reuses_each_winner_and_supports_one_or_multiple_draws(self):
        import torch
        import test as evaluator
        from src.se_auxiliary import save_se_metadata
        from test_se_auxiliary import small_config

        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        scored = []
        packed = []
        evaluated = []

        def make_quantizer(args, source, inputs, model):
            marker = model.transformer.encoder.layer[0].ffn_norm.weight.data
            self.assertEqual(marker[0].item(), 0.125)
            self.assertTrue(hasattr(model.transformer.encoder, "SELayer"))
            self.assertFalse(model.training)

            def score():
                scored.append((args.calibration_seed, source, inputs))
                candidates = []
                for image in inputs:
                    value = int(image[0, 0, 0, 0])
                    code = value if source == "se" else 9 - value
                    candidates.append(SimpleNamespace(scales=code, mean_mse=code / 100))
                return candidates

            def pack(*, candidate_scales=None):
                if candidate_scales is None:
                    candidates = score()
                    index = min(range(len(candidates)), key=lambda i: candidates[i].mean_mse)
                    winner = candidates[index]
                    quantizer.winning_candidate_index = index
                    quantizer.winning_calibration_mse = winner.mean_mse
                    candidate_scales = winner.scales
                packed.append((args.calibration_seed, source, candidate_scales))
                marker.fill_(candidate_scales)
                return model

            quantizer = SimpleNamespace(score_candidates=score, quantize=pack)
            return quantizer

        def inference(args, model):
            self.assertFalse(hasattr(model.transformer.encoder, "SELayer"))
            code = model.transformer.encoder.layer[0].ffn_norm.weight[0].item()
            evaluated.append((args.calibration_seed, code))
            return dict(mean_dice=code / 10, mean_iou=code / 20, mean_hd95=code)

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as mocks:
            root = Path(tmp)
            ids = ["slice_a", "slice_b", "slice_c"]
            (root / "train.txt").write_text("\n".join(ids))
            for index, name in enumerate(ids, start=1):
                image = np.full((32, 32), index, dtype=np.float32)
                np.savez(root / (name + ".npz"), image=image, label=np.zeros_like(image))
            checkpoint = root / "weights.pth"
            argv = ["--dataset", "Synapse", "--checkpoint", str(checkpoint),
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

            compare_se_proxy.main(argv + ["--output-dir", str(root / "selected")])
            baseline = (root / "selected/results.md").read_text()
            self.assertEqual(len(evaluated), 2)
            for seeds in ([7], [7, 3]):
                with self.subTest(seeds=seeds):
                    scored.clear()
                    packed.clear()
                    evaluated.clear()
                    stdout.seek(0)
                    stdout.truncate()
                    output = root / ("single" if len(seeds) == 1 else "multi")
                    flags = ["--case_by_case", "--output-dir", str(output)]
                    if len(seeds) > 1:
                        flags += ["--multiple_seeds", "--calibration-seeds", *map(str, seeds)]
                    compare_se_proxy.main(argv + flags)
                    self.assertEqual([(seed, method) for seed, method, _ in scored],
                                     [(seed, method) for seed in seeds for method in ("se", "direct_proxy")])
                    expected_packs = []
                    for i, seed in enumerate(seeds):
                        self.assertIs(scored[2 * i][2], scored[2 * i + 1][2])
                        ordered_ids = random.Random(seed).sample(ids, 2)
                        draw_dir = output if len(seeds) == 1 else output / f"seed_{seed}"
                        primary = (draw_dir / "results.md").read_text()
                        self.assertIn(primary, stdout.getvalue())
                        if seed == 7:
                            self.assertEqual(primary, baseline)
                        rows = markdown_rows(primary)
                        self.assertEqual(rows["SE-AUX"][0], f"`{min(ordered_ids)}`")
                        self.assertEqual(rows["Direct proxy"][0], f"`{max(ordered_ids)}`")
                        matched = (draw_dir / "case_by_case_results.md").read_text()
                        self.assertIn(matched, stdout.getvalue())
                        rows = markdown_rows(matched)
                        self.assertEqual([key for key in rows if key.startswith("`")], [f"`{s}`" for s in ordered_ids])
                        codes = [ids.index(name) + 1 for name in ordered_ids]
                        for name, code in zip(ordered_ids, codes):
                            proxy = 9 - code
                            delta = code - proxy
                            expected = [code / 100, proxy / 100, delta / 100,
                                        10 * code, 10 * proxy, 10 * delta, 5 * code, 5 * proxy, 5 * delta,
                                        code, proxy, delta]
                            np.testing.assert_allclose([float(cell) for cell in rows[f"`{name}`"]], expected)
                        expected_packs.extend((seed, "se", code) for code in codes)
                        expected_packs.extend((seed, "direct_proxy", 9 - code) for code in codes)
                    self.assertCountEqual(packed, expected_packs)
                    self.assertCountEqual(evaluated, [(seed, code) for seed, _, code in expected_packs])
                    if len(seeds) > 1:
                        for filename in ("summary.md", "case_by_case_summary.md"):
                            self.assertIn((output / filename).read_text(), stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
