"""Exercise the standalone table generator without model or third-party dependencies."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "generate_quantization_table.py"


def result(method):
    """Use the result writer's metadata, including INC's false quantized_model note."""
    quantized = method != "baseline"
    return {
        "arguments": {
            "quantize": quantized,
            "quantize_backend": "inc_awq" if method == "awq" else "legacy",
            "use_se_block": quantized,
            "drop_se_block": False,
            "vit_name": "R50-ViT-B_16",
            "img_size": 224,
            "n_skip": 3,
            "vit_patches_size": 16,
            "ckpt": "the_same_checkpoint.pth",
        },
        "notes": {
            "quantized_model": method == "se",
            "checkpoint": "the_same_checkpoint.pth",
        },
        "accuracy": {
            "mean_dice": 0.8958534598350525,
            "mean_iou": 0.8181998133659363,
            "mean_hd95": 3.173283576965332,
        },
        "metrics": {
            "throughput_img_s_mean": 367.9544441480523,
            "throughput_img_s_std": 1.2877859012796309,
            "latency_ms_mean_single": 10.717229668426514,
            "latency_ms_std_single": 0.4946141721649705,
            "latency_ms_mean_batch36": 2.7177609460830685,
            "latency_ms_mean_mean": 2.7177609460830685,
            "latency_ms_mean_std": 0.009487857141927871,
        },
        "model_size": {
            "total_bytes": 1000 if method == "baseline" else 250,
            "total_mb": 999,
        },
        "model_runtime_memory": {
            "peak_allocated_bytes": 1861219840,
            "peak_allocated_mb": 1774.99755859375,
            "peak_reserved_bytes": 2917138432,
            "unit": "MiB",
            "input_size": [3, 224, 224],
        },
    }


class GenerateQuantizationTableTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.source = self.work / "results"
        self.source.mkdir()
        self.output = self.work / "table.tex"

    def write(self, dataset, name, document):
        folder = self.source / dataset
        folder.mkdir(exist_ok=True)
        path = folder / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def dataset(self, name="ACDC"):
        # Deliberately misleading, unsorted names test metadata-only classification.
        documents = {
            "baseline": result("baseline"),
            "awq": result("awq"),
            "se": result("se"),
        }
        names = {
            "baseline": "z_se_aux.json",
            "awq": "m_baseline.json",
            "se": "a_awq.json",
        }
        paths = {
            method: self.write(name, names[method], document)
            for method, document in documents.items()
        }
        return documents, paths

    @staticmethod
    def rewrite(path, document):
        path.write_text(json.dumps(document), encoding="utf-8")

    def invoke(self, source=None, default_output=False):
        command = [sys.executable, str(SCRIPT), str(source or self.source)]
        if not default_output:
            command.extend(["--output", str(self.output)])
        return subprocess.run(
            command,
            cwd=self.work,
            text=True,
            capture_output=True,
            check=False,
        )

    def generate(self):
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return self.output.read_text(encoding="utf-8"), completed

    def row(self, table, method):
        rows = [line for line in table.splitlines() if line.startswith("& " + method + " &")]
        self.assertEqual(len(rows), 1, rows)
        return [cell.strip() for cell in rows[0].rstrip().removesuffix(r"\\").split("&")[1:]]

    def assert_failure(self, completed):
        self.assertNotEqual(completed.returncode, 0)
        self.assertTrue(completed.stderr.strip())
        self.assertNotIn("Traceback", completed.stderr)

    def test_known_acdc_values_same_checkpoint_and_source_preservation(self):
        _, paths = self.dataset()
        original = {path: path.read_bytes() for path in paths.values()}
        table, completed = self.generate()
        self.assertEqual(
            self.row(table, "SE-AUX (ours)"),
            [
                "SE-AUX (ours)", "89.59", "81.82", "3.173",
                "367.954 (1.288)", "10.717 (0.495)", r"4.000$\times$", "1775.0",
            ],
        )
        self.assertEqual(self.row(table, "Non-quantized")[6], "--")
        self.assertEqual(self.row(table, "AWQ")[6], r"4.000$\times$")
        self.assertEqual(original, {path: path.read_bytes() for path in paths.values()})
        self.assertIn(str(self.output), completed.stdout + completed.stderr)
        for package in ("graphicx", "array", "makecell", "multirow"):
            self.assertIn(package, completed.stdout + completed.stderr)

    def test_default_output_and_exact_table_structure(self):
        self.dataset()
        completed = self.invoke(default_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        table = (self.work / "performance_quantization.tex").read_text(encoding="utf-8")
        prefix = r"""\begin{table*}[!t]
\caption{QUANTIZATION PERFORMANCE COMPARISON ACROSS MEDICAL SEGMENTATION DATASETS}
\label{tab:quantization_performance}
\centering
\setlength{\extrarowheight}{2pt}
\renewcommand{\arraystretch}{1.15}
\resizebox{\textwidth}{!}{%
\begin{tabular}{clccccccc}
\hline
\multicolumn{1}{c}{\makecell[c]{Dataset}} &
\multicolumn{1}{c}{\makecell[c]{Method}} &
\makecell[c]{Dice (\%) $\uparrow$} &
\makecell[c]{IoU (\%) $\uparrow$} &
\makecell[c]{HD95 $\downarrow$} &
\makecell[c]{Throughput\\(img/s) $\uparrow$} &
\makecell[c]{Latency\\(ms) $\downarrow$} &
\makecell[c]{Compression\\Ratio $\uparrow$} &
\makecell[c]{Memory\\(MiB)$^{*}$ $\downarrow$} \\[2pt]
\hline"""
        suffix = r"""\multicolumn{9}{l}{\scriptsize
$^{*}$ Peak allocated runtime memory. Values in parentheses denote standard deviation.}
\end{tabular}%
}
\end{table*}"""
        self.assertTrue(table.startswith(prefix), table)
        self.assertTrue(table.rstrip().endswith(suffix), table)
        self.assertIn("\\multirow{3}{*}{ACDC}\n& Non-quantized", table)
        self.assertNotIn(r"\documentclass", table)
        self.assertNotIn("Insert generated", table)

    def test_dataset_and_method_order(self):
        for dataset in ("Zebra", "ACDC", "Cataract1k", "Alpha", "Synapse"):
            self.dataset(dataset)
        table, _ = self.generate()
        groups = [line for line in table.splitlines() if line.startswith(r"\multirow")]
        self.assertEqual(
            groups,
            [r"\multirow{3}{*}{" + name + "}" for name in (
                "Cataract-1K", "Synapse", "ACDC", "Alpha", "Zebra",
            )],
        )
        methods = [line.split("&")[1].strip() for line in table.splitlines() if line.startswith("& ")]
        self.assertEqual(methods, ["Non-quantized", "AWQ", "SE-AUX (ours)"] * 5)

    def test_single_image_latency_does_not_fall_back_to_batch_latency(self):
        documents, paths = self.dataset()
        metrics = documents["se"]["metrics"]
        del metrics["latency_ms_mean_single"]
        self.rewrite(paths["se"], documents["se"])
        table, completed = self.generate()
        self.assertEqual(self.row(table, "SE-AUX (ours)")[5], "--")
        self.assertTrue(completed.stderr.strip())

    def test_unique_arbitrary_batch_throughput_fallback(self):
        documents, paths = self.dataset()
        metrics = documents["se"]["metrics"]
        del metrics["throughput_img_s_mean"]
        metrics["throughput_img_s_batch17"] = 123.4567
        self.rewrite(paths["se"], documents["se"])
        table, _ = self.generate()
        self.assertEqual(self.row(table, "SE-AUX (ours)")[4], "123.457 (1.288)")

    def test_ambiguous_batch_throughput_is_missing(self):
        documents, paths = self.dataset()
        metrics = documents["se"]["metrics"]
        del metrics["throughput_img_s_mean"]
        metrics.update(throughput_img_s_batch17=100, throughput_img_s_batch29=200)
        self.rewrite(paths["se"], documents["se"])
        table, completed = self.generate()
        self.assertEqual(self.row(table, "SE-AUX (ours)")[4], "--")
        self.assertTrue(completed.stderr.strip())

    def test_memory_fallback_requires_explicit_mib_unit(self):
        documents, paths = self.dataset()
        memory = documents["se"]["model_runtime_memory"]
        del memory["peak_allocated_bytes"]
        for unit, expected in (("MiB", "1775.0"), ("MB", "--"), (None, "--")):
            with self.subTest(unit=unit):
                memory["unit"] = unit
                self.rewrite(paths["se"], documents["se"])
                table, completed = self.generate()
                self.assertEqual(self.row(table, "SE-AUX (ours)")[7], expected)
                if expected == "--":
                    self.assertTrue(completed.stderr.strip())

    def test_missing_standard_deviations_keep_means_without_recomputation(self):
        documents, paths = self.dataset()
        metrics = documents["se"]["metrics"]
        del metrics["throughput_img_s_std"]
        metrics["throughput_img_s_runs"] = [1, 10000]
        metrics["latency_ms_std_single"] = None
        self.rewrite(paths["se"], documents["se"])
        table, completed = self.generate()
        row = self.row(table, "SE-AUX (ours)")
        self.assertEqual(row[4:6], ["367.954", "10.717"])
        self.assertTrue(completed.stderr.strip())

    def test_null_missing_and_nonfinite_metrics_are_missing(self):
        documents, paths = self.dataset()
        documents["se"]["accuracy"] = {
            "mean_dice": None, "mean_hd95": float("inf"),
        }
        documents["se"]["metrics"]["throughput_img_s_mean"] = float("nan")
        documents["se"]["metrics"]["latency_ms_mean_single"] = float("-inf")
        documents["se"]["model_runtime_memory"] = {"input_size": [3, 224, 224]}
        self.rewrite(paths["se"], documents["se"])
        table, completed = self.generate()
        row = self.row(table, "SE-AUX (ours)")
        self.assertEqual(row[1:6], ["--"] * 5)
        self.assertEqual(row[7], "--")
        self.assertTrue(completed.stderr.strip())

    def test_compression_uses_total_bytes_and_allows_different_checkpoints(self):
        documents, paths = self.dataset()
        documents["baseline"]["model_size"]["total_bytes"] = 996
        documents["se"]["model_size"]["total_bytes"] = 300
        documents["se"]["arguments"]["ckpt"] = "different.pth"
        documents["se"]["notes"]["checkpoint"] = "different.pth"
        for method in ("baseline", "se"):
            self.rewrite(paths[method], documents[method])
        table, _ = self.generate()
        self.assertEqual(self.row(table, "SE-AUX (ours)")[6], r"3.320$\times$")

    def test_incompatible_architecture_or_input_suppresses_compression(self):
        documents, paths = self.dataset()
        for section, key, value in (
            ("arguments", "vit_name", "ViT-L_16"),
            ("arguments", "img_size", 384),
            ("model_runtime_memory", "input_size", [1, 224, 224]),
        ):
            with self.subTest(section=section, key=key):
                document = copy.deepcopy(documents["se"])
                document[section][key] = value
                self.rewrite(paths["se"], document)
                table, completed = self.generate()
                self.assertEqual(self.row(table, "SE-AUX (ours)")[6], "--")
                self.assertTrue(completed.stderr.strip())

    def test_invalid_model_sizes_suppress_compression(self):
        documents, paths = self.dataset()
        for size in (None, 0, -1, float("inf")):
            with self.subTest(size=size):
                documents["se"]["model_size"]["total_bytes"] = size
                self.rewrite(paths["se"], documents["se"])
                table, completed = self.generate()
                self.assertEqual(self.row(table, "SE-AUX (ours)")[6], "--")
                self.assertTrue(completed.stderr.strip())

    def test_invalid_directory_and_malformed_json_name_the_source(self):
        missing = self.work / "not-a-directory"
        completed = self.invoke(source=missing)
        self.assert_failure(completed)
        self.assertIn(str(missing), completed.stderr)
        _, paths = self.dataset()
        paths["se"].write_text("{broken JSON", encoding="utf-8")
        completed = self.invoke()
        self.assert_failure(completed)
        self.assertIn(paths["se"].name, completed.stderr)

    def test_unknown_method_metadata_is_not_guessed(self):
        documents, paths = self.dataset()
        documents["awq"]["arguments"]["quantize_backend"] = "unrecognized_backend"
        del documents["awq"]["arguments"]["use_se_block"]
        self.rewrite(paths["awq"], documents["awq"])
        completed = self.invoke()
        self.assert_failure(completed)
        self.assertIn(paths["awq"].name, completed.stderr)
        self.assertIn("quantize", completed.stderr)
        self.assertIn("unrecognized_backend", completed.stderr)

    def test_duplicate_json_keys_are_rejected_with_filename(self):
        documents, paths = self.dataset()
        encoded = json.dumps(documents["se"])
        duplicated = encoded.replace(
            '"quantize": true', '"quantize": false, "quantize": true', 1,
        )
        self.assertNotEqual(encoded, duplicated)
        paths["se"].write_text(duplicated, encoding="utf-8")
        completed = self.invoke()
        self.assert_failure(completed)
        self.assertIn(paths["se"].name, completed.stderr)
        self.assertIn("quantize", completed.stderr)

    def test_conflicting_legacy_se_flags_are_not_guessed(self):
        documents, paths = self.dataset()
        documents["se"]["arguments"]["drop_se_block"] = True
        self.rewrite(paths["se"], documents["se"])
        completed = self.invoke()
        self.assert_failure(completed)
        self.assertIn(paths["se"].name, completed.stderr)
        self.assertIn("use_se_block", completed.stderr)
        self.assertIn("drop_se_block", completed.stderr)

    def test_output_cannot_overwrite_a_source_json(self):
        _, paths = self.dataset()
        original = {path: path.read_bytes() for path in paths.values()}
        self.output = paths["se"]
        completed = self.invoke()
        self.assert_failure(completed)
        self.assertIn(paths["se"].name, completed.stderr)
        self.assertEqual(original, {path: path.read_bytes() for path in paths.values()})

    def test_legacy_without_se_and_packed_inc_are_awq(self):
        documents, paths = self.dataset()
        for backend, use_se in (("legacy", False), ("inc_awq_packed", True)):
            with self.subTest(backend=backend):
                document = copy.deepcopy(documents["awq"])
                document["arguments"]["quantize_backend"] = backend
                document["arguments"]["use_se_block"] = use_se
                document["notes"]["quantized_model"] = True
                self.rewrite(paths["awq"], document)
                table, _ = self.generate()
                self.assertEqual(self.row(table, "AWQ")[6], r"4.000$\times$")

    def test_duplicate_and_missing_methods_report_files_and_metadata(self):
        documents, paths = self.dataset()
        duplicate = self.write("ACDC", "another.json", documents["baseline"])
        completed = self.invoke()
        self.assert_failure(completed)
        for path in (paths["baseline"], duplicate):
            self.assertIn(path.name, completed.stderr)
        self.assertIn("quantize", completed.stderr)
        duplicate.unlink()
        paths["se"].unlink()
        completed = self.invoke()
        self.assert_failure(completed)
        self.assertIn("ACDC", completed.stderr)
        self.assertIn("SE", completed.stderr)
        self.assertIn(paths["baseline"].name, completed.stderr)
        self.assertIn(paths["awq"].name, completed.stderr)
        self.assertIn("quantize", completed.stderr)

    def test_dataset_names_escape_latex_special_characters(self):
        self.dataset("A&B_%$#{}~^\\")
        table, _ = self.generate()
        escaped = r"A\&B\_\%\$\#\{\}\textasciitilde{}\textasciicircum{}\textbackslash{}"
        self.assertIn(r"\multirow{3}{*}{" + escaped + "}", table)


if __name__ == "__main__":
    unittest.main()
