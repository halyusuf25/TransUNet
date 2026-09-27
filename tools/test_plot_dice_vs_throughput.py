"""Numerical and artist-level checks for the manuscript plotting options.

Run from the repository root with::

    python -m unittest tools.test_plot_dice_vs_throughput
"""

import re
import unittest
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox

from tools import plot_dice_vs_throughput as plotting


class MemoryAreaTests(unittest.TestCase):
    def test_memory_ratios_are_preserved(self):
        memory = pd.Series([10.0, 100.0, 1000.0, 10000.0])
        areas = plotting.scale_memory_to_marker_area(memory)
        np.testing.assert_allclose(areas / memory.to_numpy(), 0.09)
        self.assertEqual(float(areas.max()), 900.0)

    def test_explicit_scale_and_equal_values(self):
        memory = pd.Series([200.0, 200.0, 200.0])
        np.testing.assert_allclose(
            plotting.scale_memory_to_marker_area(memory), [900.0] * 3
        )
        np.testing.assert_allclose(
            plotting.scale_memory_to_marker_area(memory, scale_factor=0.25),
            [50.0] * 3,
        )

    def test_invalid_memory_and_scale_are_rejected(self):
        for values in ([], [0.0], [-1.0], [np.inf], [np.nan], ["invalid"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                plotting.scale_memory_to_marker_area(pd.Series(values, dtype=object))
        for value in (0.0, -1.0, np.inf, np.nan):
            for keyword in ("max_area", "scale_factor"):
                with self.subTest(keyword=keyword, value=value):
                    with self.assertRaises(ValueError):
                        plotting.scale_memory_to_marker_area(
                            pd.Series([100.0]), **{keyword: value}
                        )

    def test_legend_uses_the_same_area_scale(self):
        memory = pd.Series([100.0, 200.0, 400.0, 900.0])
        handles = plotting._memory_legend_handles(memory, scale_factor=0.5)
        self.assertEqual(len(handles), 3)
        np.testing.assert_allclose(
            [handle.get_markersize() ** 2 for handle in handles],
            plotting.scale_memory_to_marker_area(
                pd.Series([100.0, 300.0, 900.0]), scale_factor=0.5
            ),
        )
        self.assertTrue(all("MiB" in handle.get_label() for handle in handles))

    def test_repeated_legend_reference_values_are_removed(self):
        equal = plotting._memory_legend_handles(pd.Series([120.0, 120.0]))
        repeated = plotting._memory_legend_handles(pd.Series([120.0, 120.0, 900.0]))
        self.assertEqual(len(equal), 1)
        self.assertEqual(len(repeated), 2)
        self.assertAlmostEqual(equal[0].get_markersize() ** 2, 900.0)

    def test_drawn_legend_radii_share_calibration_without_mutating_source_handles(self):
        memory = pd.Series([100.0, 400.0, 900.0])
        for radius in (0.5, 1.0, 2.5):
            with self.subTest(radius=radius):
                factor = plotting._reference_memory_scale(radius, 100.0)
                handles = plotting._memory_legend_handles(memory, scale_factor=factor)
                original_diameters = np.array([handle.get_markersize() for handle in handles])
                figure = plotting.plt.figure()
                try:
                    legend = plotting._create_memory_legend(figure, handles, font_size=5.0)
                    rendered_diameters = [glyph.get_markersize() for glyph in legend.legend_handles]
                    np.testing.assert_array_equal(rendered_diameters, original_diameters)
                    np.testing.assert_allclose(
                        np.array(rendered_diameters) * 25.4 / 144.0,
                        radius * np.array([1.0, 2.0, 3.0]),
                    )
                    np.testing.assert_array_equal(
                        [handle.get_markersize() for handle in handles], original_diameters
                    )
                finally:
                    plotting.plt.close(figure)

    def test_physical_memory_scale_uses_disk_area_and_reports_rounding(self):
        for scale_factor in (0.025, 0.7, 123.0):
            expected = 1.0 / (np.pi / 4.0 * scale_factor * (25.4 / 72.0) ** 2)
            for vertical in (False, True):
                with self.subTest(scale_factor=scale_factor, vertical=vertical):
                    label = plotting._memory_scale_label(scale_factor, vertical=vertical)
                    self.assertIn("1 mm²", label)
                    self.assertIn("outline excluded", label)
                    match = re.search(r"≈\s*([0-9.eE+-]+)\s*MiB", label)
                    self.assertIsNotNone(match)
                    self.assertAlmostEqual(float(match.group(1)) / expected, 1.0, delta=5e-4)


class PlotArtistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_data = plotting.load_and_validate_csv(plotting.resolve_csv_path())

    def setUp(self):
        self.data = self.source_data.copy(deep=True)
        self.figures = []
        self.rc_context = matplotlib.rc_context()
        self.rc_context.__enter__()

    def tearDown(self):
        for figure in self.figures:
            plotting.plt.close(figure)
        self.rc_context.__exit__(None, None, None)

    def plot(self, datasets=("Synapse",), **options):
        figure = plotting.plot_dice_vs_throughput(self.data, datasets, **options)
        self.figures.append(figure)
        figure.canvas.draw()
        return figure

    @staticmethod
    def scatter_artists(figure, zorder):
        return [
            artist for artist in figure.axes[0].collections
            if isinstance(artist, PathCollection) and artist.get_zorder() == zorder
        ]

    @staticmethod
    def legends(figure):
        return figure.findobj(match=Legend)

    def assert_legend_glyphs_fit(self, figure):
        """Measure drawn markers only; Line2D extents include undrawn endpoints."""
        renderer = figure.canvas.get_renderer()
        for legend in self.legends(figure):
            legend_box = legend.get_window_extent(renderer)
            for glyph in legend.findobj(match=Line2D):
                if glyph.get_marker() in (None, "none", "None", "", " "):
                    continue
                offsets = glyph.get_xydata()
                if glyph.get_markevery() is not None:
                    offsets = offsets[glyph.get_markevery()]
                points = glyph.get_transform().transform(offsets)
                marker = MarkerStyle(glyph.get_marker())
                bounds = marker.get_path().transformed(marker.get_transform()).get_extents()
                scale = glyph.get_markersize() * figure.dpi / 72.0
                stroke = glyph.get_markeredgewidth() * figure.dpi / 144.0
                for x_value, y_value in points:
                    box = Bbox.from_extents(
                        x_value + bounds.x0 * scale - stroke,
                        y_value + bounds.y0 * scale - stroke,
                        x_value + bounds.x1 * scale + stroke,
                        y_value + bounds.y1 * scale + stroke,
                    )
                    self.assertGreaterEqual(box.x0, legend_box.x0 - 0.01)
                    self.assertGreaterEqual(box.y0, legend_box.y0 - 0.01)
                    self.assertLessEqual(box.x1, legend_box.x1 + 0.01)
                    self.assertLessEqual(box.y1, legend_box.y1 + 0.01)
                    for label in legend.texts:
                        self.assertFalse(box.overlaps(label.get_window_extent(renderer)))

    def assert_scatter_inside_axes(self, figure):
        """Measure rendered marker paths and their stroke, not just centers."""
        axes = figure.axes[0]
        axes_box = axes.get_window_extent(figure.canvas.get_renderer())
        pixels_per_point = figure.dpi / 72.0
        for artist in axes.collections:
            if not isinstance(artist, PathCollection):
                continue
            points = artist.get_offset_transform().transform(artist.get_offsets())
            sizes = artist.get_sizes()
            widths = artist.get_linewidths()
            paths = artist.get_paths()
            for index, (x_value, y_value) in enumerate(points):
                scale = np.sqrt(sizes[index % len(sizes)]) * pixels_per_point
                half_stroke = widths[index % len(widths)] * pixels_per_point / 2.0
                path_box = paths[index % len(paths)].get_extents()
                bounds = (
                    x_value + path_box.x0 * scale - half_stroke,
                    y_value + path_box.y0 * scale - half_stroke,
                    x_value + path_box.x1 * scale + half_stroke,
                    y_value + path_box.y1 * scale + half_stroke,
                )
                with self.subTest(point=index, zorder=artist.get_zorder()):
                    self.assertGreaterEqual(bounds[0], axes_box.x0 - 0.01)
                    self.assertGreaterEqual(bounds[1], axes_box.y0 - 0.01)
                    self.assertLessEqual(bounds[2], axes_box.x1 + 0.01)
                    self.assertLessEqual(bounds[3], axes_box.y1 + 0.01)

    def test_dice_guides_are_opt_in_and_values_require_lines(self):
        parser = plotting.build_argument_parser()
        defaults = parser.parse_args([])
        self.assertFalse(defaults.dice_guide_lines)
        self.assertFalse(defaults.dice_guide_labels)
        selected = parser.parse_args(["--dice_guide_lines", "--dice_guide_labels"])
        self.assertTrue(selected.dice_guide_lines)
        self.assertTrue(selected.dice_guide_labels)
        ordinary = self.plot(paper_style=True)
        labels_only = self.plot(paper_style=True, dice_guide_labels=True)
        np.testing.assert_array_equal(
            np.asarray(ordinary.canvas.buffer_rgba()), np.asarray(labels_only.canvas.buffer_rgba())
        )

    def test_dice_guides_reach_final_y_axis_and_preserve_bubbles(self):
        for datasets in (("Synapse",), tuple(plotting.DATASET_COLUMNS)):
            options = dict(
                datasets=datasets, paper_style=True, add_memory_legende=True,
                model_center_markers=True, model_center_colors=True,
                show_model_marker_legend=True, radius=2, runtime_memory=2048,
                memory_legend_font_size=4, axis_label_font_size=6, remove_title=True,
                show_throughput_std=True,
            )
            ordinary = self.plot(**options)
            for show_labels in (False, True):
                with self.subTest(datasets=datasets, labels=show_labels):
                    figure = self.plot(
                        dice_guide_lines=True, dice_guide_labels=show_labels, **options,
                    )
                    axes = figure.axes[0]
                    lines = [line for line in axes.lines if (line.get_gid() or "").startswith(
                        "dice-guide-line:"
                    )]
                    labels = [label for label in axes.texts if (label.get_gid() or "").startswith(
                        "dice-guide-value:"
                    )]
                    self.assertEqual(len(lines), 2 * len(datasets))
                    self.assertEqual(len(labels), 2 * len(datasets) if show_labels else 0)
                    for zorder in (3, 4):
                        before = self.scatter_artists(ordinary, zorder)
                        after = self.scatter_artists(figure, zorder)
                        self.assertEqual(len(before), len(after))
                        for first, second in zip(before, after):
                            for attribute in (
                                "get_offsets", "get_sizes", "get_facecolors",
                                "get_edgecolors", "get_linewidths",
                            ):
                                np.testing.assert_array_equal(
                                    getattr(first, attribute)(), getattr(second, attribute)()
                                )
                    for dpi in (100, 200):
                        figure.set_dpi(dpi)
                        figure.canvas.draw()
                        renderer = figure.canvas.get_renderer()
                        for line in lines:
                            _, dataset, model = line.get_gid().split(":", 2)
                            row = self.data.loc[self.data["Model"].eq(model)].iloc[0]
                            np.testing.assert_allclose(line.get_xdata(), [
                                axes.get_xlim()[0], row[plotting.THROUGHPUT_COLUMN],
                            ])
                            np.testing.assert_allclose(
                                line.get_ydata(), [row[plotting.DATASET_COLUMNS[dataset]]] * 2
                            )
                            endpoints = line.get_transform().transform(line.get_xydata())
                            self.assertAlmostEqual(endpoints[0, 0], axes.bbox.x0)
                            self.assertAlmostEqual(endpoints[0, 1], endpoints[1, 1])
                            self.assertTrue(line.is_dashed())
                            self.assertEqual(to_rgba(line.get_color()),
                                             to_rgba(plotting.OURS_OUTLINE_COLOR))
                        boxes = [label.get_window_extent(renderer) for label in labels]
                        visible_ticks = [tick.label1 for tick in axes.yaxis.get_major_ticks()
                                         if tick.label1.get_visible()]
                        for index, (label, box) in enumerate(zip(labels, boxes)):
                            self.assertEqual(label.get_fontsize(), 6)
                            expected_y = axes.transData.transform((0, float(label.get_text())))[1]
                            self.assertAlmostEqual((box.y0 + box.y1) / 2, expected_y)
                            self.assertLess(box.x1, axes.bbox.x0)
                            self.assertGreaterEqual(box.x0, figure.bbox.x0)
                            self.assertGreaterEqual(box.y0, figure.bbox.y0)
                            self.assertLessEqual(box.y1, figure.bbox.y1)
                            other_boxes = boxes[index + 1:] + [
                                artist.get_window_extent(renderer) for artist in
                                [axes.yaxis.label] + visible_ticks + self.legends(figure)
                            ]
                            self.assertTrue(all(not box.overlaps(other) for other in other_boxes))
                        self.assert_scatter_inside_axes(figure)
                    self.assertEqual(figure.get_figwidth(), ordinary.get_figwidth())
        pd.testing.assert_frame_equal(self.data, self.source_data)

    def test_dice_guides_skip_missing_models_and_deduplicate_equal_value_labels(self):
        self.data = self.data.loc[~self.data["Model"].isin(("AD2Former", plotting.OURS_MODEL))]
        figure = self.plot(dice_guide_lines=True, dice_guide_labels=True)
        self.assertFalse(any((artist.get_gid() or "").startswith("dice-guide-")
                             for artist in figure.findobj()))
        self.data = self.source_data.copy(deep=True)
        selected = self.data["Model"].isin(("AD2Former", plotting.OURS_MODEL))
        self.data.loc[selected, plotting.DATASET_COLUMNS["Synapse"]] = 80.25
        figure = self.plot(dice_guide_lines=True, dice_guide_labels=True)
        labels = [label for label in figure.axes[0].texts
                  if (label.get_gid() or "").startswith("dice-guide-value:")]
        self.assertEqual([label.get_text() for label in labels], ["80.25"])

    def test_dice_guides_avoid_direct_model_labels(self):
        figure = self.plot(
            datasets=tuple(plotting.DATASET_COLUMNS), paper_style=True,
            add_memory_legende=True, radius=2, runtime_memory=2048,
            dice_guide_lines=True, dice_guide_labels=True, display_model_names=True,
            model_center_markers=True, axis_label_font_size=6,
        )
        renderer = figure.canvas.get_renderer()
        annotations = [label for label in figure.axes[0].texts
                       if not (label.get_gid() or "").startswith("dice-guide-")]
        for line in figure.axes[0].lines:
            if not (line.get_gid() or "").startswith("dice-guide-line:"):
                continue
            path = line.get_path().transformed(line.get_transform())
            for label in annotations:
                self.assertFalse(path.intersects_bbox(
                    Text.get_window_extent(label, renderer), filled=False
                ))

    def test_optional_flags_and_memory_legend_dependency(self):
        parser = plotting.build_argument_parser()
        defaults = parser.parse_args([])
        self.assertFalse(defaults.buble_size_legened)
        self.assertFalse(defaults.highlight_ours)
        self.assertFalse(defaults.remove_title)
        self.assertFalse(defaults.right_memory_legend)
        self.assertFalse(defaults.scaled_memorey_legened)
        self.assertEqual(parser.parse_args(["--model_marker_legend"]).model_marker_legend, 7)
        with self.assertRaisesRegex(ValueError, "requires --memorey"):
            self.plot(show_bubble_legend=True)
        figure = self.plot()
        self.assertFalse(self.legends(figure))
        self.assertTrue(figure.axes[0].get_title())
        self.assertEqual(figure.axes[0].get_xlabel(), "Throughput (images/s)")
        self.assertEqual(figure.axes[0].get_ylabel(), "Dice (%)")

    def test_memory_legend_modifiers_are_inert_without_the_legend(self):
        ordinary = self.plot(paper_style=True)
        for right, scaled in ((True, False), (False, True), (True, True)):
            with self.subTest(right=right, scaled=scaled):
                figure = self.plot(
                    paper_style=True, right_memory_legend=right,
                    scaled_memorey_legened=scaled,
                )
                self.assertFalse(self.legends(figure))
                np.testing.assert_array_equal(
                    figure.get_size_inches(), ordinary.get_size_inches()
                )
                np.testing.assert_array_equal(
                    figure.axes[0].get_position().bounds,
                    ordinary.axes[0].get_position().bounds,
                )

    def test_memory_legend_options_preserve_plot_and_fit_the_canvas(self):
        options = dict(
            paper_style=True, memory_scaling=True, show_bubble_legend=True,
            model_center_markers=True, model_center_colors=True,
            show_model_marker_legend=True,
        )
        baseline = self.plot(**options)
        memory = self.data[plotting.MEMORY_COLUMN]
        reference_values = np.unique([memory.min(), memory.median(), memory.max()])
        baseline_bubbles = self.scatter_artists(baseline, 3)[0]
        baseline_centers = self.scatter_artists(baseline, 4)
        expected_sizes = reference_values * plotting.MAX_MEMORY_MARKER_AREA / memory.max()
        for right, scaled in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(right=right, scaled=scaled):
                figure = baseline if not (right or scaled) else self.plot(
                    right_memory_legend=right, scaled_memorey_legened=scaled,
                    **options,
                )
                axes = figure.axes[0]
                bubbles = self.scatter_artists(figure, 3)[0]
                for attribute in ("get_offsets", "get_sizes", "get_facecolors", "get_linewidths"):
                    np.testing.assert_array_equal(
                        getattr(baseline_bubbles, attribute)(), getattr(bubbles, attribute)()
                    )
                centers = self.scatter_artists(figure, 4)
                self.assertEqual(len(centers), len(baseline_centers))
                for before, after in zip(baseline_centers, centers):
                    for attribute in ("get_offsets", "get_sizes", "get_facecolors"):
                        np.testing.assert_array_equal(
                            getattr(before, attribute)(), getattr(after, attribute)()
                        )
                    np.testing.assert_array_equal(
                        before.get_paths()[0].vertices, after.get_paths()[0].vertices
                    )
                memory_legend = next(
                    legend for legend in self.legends(figure)
                    if len(legend.legend_handles) == len(reference_values)
                )
                reference_glyphs = memory_legend.legend_handles
                np.testing.assert_allclose(
                    sorted(glyph.get_markersize() ** 2 for glyph in reference_glyphs),
                    expected_sizes,
                )
                if scaled:
                    self.assertTrue(all(not label.get_text() for label in memory_legend.texts))
                    scale_note = memory_legend.findobj(
                        match=lambda artist: artist.get_gid() == "memory-legend-scale-label"
                    )[0]
                    self.assertIn("MiB", scale_note.get_text())
                else:
                    self.assertTrue(all("MiB" in label.get_text() for label in memory_legend.texts))
                renderer = figure.canvas.get_renderer()
                legend_boxes = [
                    legend.get_window_extent(renderer) for legend in self.legends(figure)
                ]
                for index, box in enumerate(legend_boxes):
                    self.assertGreaterEqual(box.x0, figure.bbox.x0 - 0.01)
                    self.assertGreaterEqual(box.y0, figure.bbox.y0 - 0.01)
                    self.assertLessEqual(box.x1, figure.bbox.x1 + 0.01)
                    self.assertLessEqual(box.y1, figure.bbox.y1 + 0.01)
                    self.assertFalse(box.overlaps(axes.bbox))
                    self.assertFalse(box.overlaps(axes.xaxis.label.get_window_extent(renderer)))
                    for other in legend_boxes[index + 1:]:
                        self.assertFalse(box.overlaps(other))
                model_legend = next(
                    legend for legend in self.legends(figure)
                    if len(legend.legend_handles) == len(self.data)
                )
                self.assertLess(
                    model_legend.get_window_extent(renderer).y1,
                    axes.xaxis.label.get_window_extent(renderer).y0,
                )
                if right:
                    self.assertGreater(memory_legend.get_window_extent(renderer).x0, axes.bbox.x1)
                    self.assertGreater(figure.get_figwidth(), baseline.get_figwidth())
                    marker_centers = [
                        glyph.get_transform().transform(glyph.get_xydata())[0]
                        for glyph in reference_glyphs
                    ]
                    np.testing.assert_allclose(
                        [point[0] for point in marker_centers], marker_centers[0][0]
                    )
                    self.assertEqual(
                        len({round(point[1], 3) for point in marker_centers}),
                        len(reference_values),
                    )
                else:
                    self.assertLess(memory_legend.get_window_extent(renderer).y1, axes.bbox.y0)
                    self.assertAlmostEqual(figure.get_figwidth(), baseline.get_figwidth())
                self.assert_legend_glyphs_fit(figure)
                self.assert_scatter_inside_axes(figure)
        pd.testing.assert_frame_equal(self.data, self.source_data)

    def test_scale_key_matches_rendered_disk_dimensions_at_different_dpi(self):
        figure = self.plot(
            paper_style=True, memory_scaling=True, show_bubble_legend=True,
            scaled_memorey_legened=True, right_memory_legend=True,
        )
        memory_legend = self.legends(figure)[0]
        scale_note = memory_legend.findobj(
            match=lambda artist: artist.get_gid() == "memory-legend-scale-label"
        )[0]
        match = re.search(r"≈\s*([0-9.eE+-]+)\s*MiB", scale_note.get_text())
        self.assertIsNotNone(match)
        coefficient = float(match.group(1))
        bubbles = self.scatter_artists(figure, 3)[0]
        memory = self.data[plotting.MEMORY_COLUMN].to_numpy()
        reference_values = np.unique([memory.min(), np.median(memory), memory.max()])
        reference_glyphs = sorted(
            memory_legend.legend_handles, key=lambda glyph: glyph.get_markersize()
        )
        for dpi in (100, 200):
            figure.set_dpi(dpi)
            figure.canvas.draw()
            with self.subTest(dpi=dpi):
                path_diameter = bubbles.get_paths()[0].get_extents().width
                diameter_mm = path_diameter * bubbles.get_transforms()[:, 0, 0] / dpi * 25.4
                disk_area_mm2 = np.pi / 4.0 * diameter_mm ** 2
                np.testing.assert_allclose(memory / disk_area_mm2, coefficient, rtol=5e-4)
                for value, glyph in zip(reference_values, reference_glyphs):
                    marker = MarkerStyle(glyph.get_marker())
                    path = marker.get_path().transformed(marker.get_transform())
                    diameter_mm = path.get_extents().width * glyph.get_markersize() / 72 * 25.4
                    reference_disk_area_mm2 = np.pi / 4.0 * diameter_mm ** 2
                    self.assertAlmostEqual(value / reference_disk_area_mm2 / coefficient,
                                           1.0, delta=5e-4)

    @staticmethod
    def radius_legend(figure):
        return next(
            legend for legend in figure.findobj(match=Legend)
            if any(
                "mm radius =" in label.get_text() or label.get_text().startswith("r =")
                for label in legend.texts
            )
        )

    def test_compact_memory_label_is_opt_in_and_uses_plot_calibration(self):
        parser = plotting.build_argument_parser()
        self.assertFalse(parser.parse_args([]).memory_legend_compact_label)
        self.assertTrue(parser.parse_args([
            "--memory_legend_compact_label",
        ]).memory_legend_compact_label)
        for radius, memory in ((1.0, 1024.0), (2.0, 2048.0), (1.25, 512.5)):
            with self.subTest(radius=radius, memory=memory):
                options = dict(
                    paper_style=True, add_memory_legende=True, radius=radius,
                    runtime_memory=memory, memory_legend_radius_annotation=2.5,
                    model_center_markers=True, model_center_colors=True,
                    show_model_marker_legend=True,
                )
                ordinary = self.plot(**options)
                compact = self.plot(memory_legend_compact_label=True, **options)
                old_legend = self.radius_legend(ordinary)
                new_legend = self.radius_legend(compact)
                self.assertEqual(old_legend.texts[0].get_text(),
                                 f"{radius:g} mm radius = {memory:g} MiB")
                self.assertEqual(new_legend.texts[0].get_text(),
                                 f"r = {radius:g}mm ↔ {memory:g} MiB")
                self.assertEqual(new_legend.get_title().get_text(),
                                 old_legend.get_title().get_text())
                self.assertEqual(new_legend.texts[0].get_fontsize(),
                                 old_legend.texts[0].get_fontsize())
                for before, after in zip(old_legend.legend_handles, new_legend.legend_handles):
                    self.assertEqual(before.get_markersize(), after.get_markersize())
                    self.assertEqual(before.get_markeredgewidth(), after.get_markeredgewidth())
                for zorder in (3, 4):
                    before = self.scatter_artists(ordinary, zorder)
                    after = self.scatter_artists(compact, zorder)
                    self.assertEqual(len(before), len(after))
                    for first, second in zip(before, after):
                        for attribute in (
                            "get_offsets", "get_sizes", "get_facecolors",
                            "get_edgecolors", "get_linewidths",
                        ):
                            np.testing.assert_array_equal(
                                getattr(first, attribute)(), getattr(second, attribute)()
                            )
                self.assert_legend_glyphs_fit(compact)
                self.assert_scatter_inside_axes(compact)

    def test_compact_memory_label_is_inert_without_reference_legend(self):
        for show_bubble_legend in (False, True):
            options = dict(
                paper_style=True, memory_scaling=True, show_bubble_legend=show_bubble_legend,
            )
            ordinary = self.plot(**options)
            compact = self.plot(memory_legend_compact_label=True, **options)
            np.testing.assert_array_equal(
                np.asarray(ordinary.canvas.buffer_rgba()), np.asarray(compact.canvas.buffer_rgba())
            )

    def test_reference_radius_options_are_opt_in_and_inactive_values_change_nothing(self):
        parser = plotting.build_argument_parser()
        defaults = parser.parse_args([])
        self.assertFalse(defaults.add_memory_legende)
        self.assertEqual(defaults.radius, 1.0)
        self.assertEqual(defaults.runtime_memory, 1024.0)
        selected = parser.parse_args([
            "--add_memory_legende", "--radius", "1.5", "--runtime_memory", "512",
        ])
        self.assertTrue(selected.add_memory_legende)
        self.assertEqual(selected.radius, 1.5)
        self.assertEqual(selected.runtime_memory, 512.0)
        for memory_scaling in (False, True):
            with self.subTest(memory_scaling=memory_scaling):
                options = dict(paper_style=True, memory_scaling=memory_scaling)
                ordinary = self.plot(**options)
                changed = self.plot(radius=2.0, runtime_memory=256.0, **options)
                np.testing.assert_array_equal(
                    np.asarray(ordinary.canvas.buffer_rgba()),
                    np.asarray(changed.canvas.buffer_rgba()),
                )

    def test_memory_legend_annotation_and_title_options(self):
        parser = plotting.build_argument_parser()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.memory_legend_radius_annotation, 1.0)
        self.assertIsInstance(defaults.memory_legend_radius_annotation, float)
        self.assertEqual(defaults.memory_legend_title, "Peak Runtime Memory")
        self.assertEqual(defaults.memory_legend_font_size, 5)
        selected = parser.parse_args([
            "--memory_legend_radius_annotation", "2.5",
            "--memory_legend_title", "Runtime memory",
            "--memory_legend_font_size", "7",
        ])
        self.assertEqual(selected.memory_legend_radius_annotation, 2.5)
        self.assertEqual(selected.memory_legend_title, "Runtime memory")
        self.assertEqual(selected.memory_legend_font_size, 7)
        ordinary = self.plot(memory_scaling=True, paper_style=True)
        modified = self.plot(
            memory_scaling=True, paper_style=True,
            memory_legend_radius_annotation=2.5, memory_legend_title="Runtime memory",
        )
        self.assertFalse(self.legends(modified))
        np.testing.assert_array_equal(
            np.asarray(ordinary.canvas.buffer_rgba()), np.asarray(modified.canvas.buffer_rgba())
        )
        for font_size in (5, 7):
            for title in ("Peak Runtime Memory", "Runtime memory\n(Synapse)", ""):
                with self.subTest(font_size=font_size, title=title):
                    figure = self.plot(
                        paper_style=True, add_memory_legende=True,
                        memory_scaling=True, show_bubble_legend=True,
                        scaled_memorey_legened=(font_size == 7),
                        memory_legend_font_size=font_size, memory_legend_title=title,
                    )
                    for legend in self.legends(figure):
                        self.assertEqual(legend.get_title().get_text(), title)
                        self.assertEqual(legend.get_title().get_fontsize(), font_size)
                        expected_radius = (
                            1.0 if legend is self.radius_legend(figure)
                            else np.sqrt(self.data[plotting.MEMORY_COLUMN].min() / 1024.0)
                        )
                        self.assertAlmostEqual(
                            min(glyph.get_markersize() for glyph in legend.legend_handles)
                            * 25.4 / 144, expected_radius,
                        )
                        self.assertTrue(all(
                            label.get_fontsize() == font_size for label in legend.texts
                        ))
                        renderer = figure.canvas.get_renderer()
                        title_box = legend.get_title().get_window_extent(renderer)
                        legend_box = legend.get_window_extent(renderer)
                        if title:
                            self.assertGreaterEqual(title_box.y0, legend_box.y0)
                            self.assertLessEqual(title_box.y1, legend_box.y1)
                            self.assertTrue(all(
                                title_box.y0 > label.get_window_extent(renderer).y1
                                for label in legend.texts
                            ))

    def test_legacy_annotation_option_keeps_shared_scale_and_fits_at_multiple_dpi(self):
        cases = (
            dict(add_memory_legende=True),
            dict(show_bubble_legend=True),
            dict(show_bubble_legend=True, right_memory_legend=True),
            dict(show_bubble_legend=True, scaled_memorey_legened=True),
            dict(add_memory_legende=True, show_bubble_legend=True,
                 right_memory_legend=True, scaled_memorey_legened=True),
        )
        for case in cases:
            with self.subTest(options=case):
                options = dict(
                    paper_style=True, memory_scaling=True, model_center_markers=True,
                    model_center_colors=True, show_model_marker_legend=True, **case,
                )
                ordinary = self.plot(**options)
                annotated = self.plot(memory_legend_radius_annotation=2.5, **options)
                np.testing.assert_array_equal(
                    np.asarray(ordinary.canvas.buffer_rgba()),
                    np.asarray(annotated.canvas.buffer_rgba()),
                )
                for zorder in (3, 4):
                    before = self.scatter_artists(ordinary, zorder)
                    after = self.scatter_artists(annotated, zorder)
                    self.assertEqual(len(before), len(after))
                    for first, second in zip(before, after):
                        for attribute in (
                            "get_offsets", "get_sizes", "get_facecolors",
                            "get_edgecolors", "get_linewidths",
                        ):
                            np.testing.assert_array_equal(
                                getattr(first, attribute)(), getattr(second, attribute)()
                            )
                        np.testing.assert_array_equal(
                            first.get_paths()[0].vertices, second.get_paths()[0].vertices
                        )
                for first, second in zip(self.legends(ordinary), self.legends(annotated)):
                    if not first.get_title().get_text():
                        continue  # Model-symbol legend is unchanged.
                    self.assertEqual(second.get_title().get_text(), first.get_title().get_text())
                    diameters_before = [glyph.get_markersize() for glyph in first.legend_handles]
                    diameters_after = [glyph.get_markersize() for glyph in second.legend_handles]
                    np.testing.assert_array_equal(diameters_after, diameters_before)
                    lines = second.findobj(
                        match=lambda artist: artist.get_gid() == "memory-legend-radius-line"
                    )
                    labels = second.findobj(
                        match=lambda artist: artist.get_gid() == "memory-legend-radius-label"
                    )
                    self.assertEqual(len(lines), len(second.legend_handles))
                    self.assertEqual(len(labels), len(lines))
                    for dpi in (100, 200):
                        annotated.set_dpi(dpi)
                        annotated.canvas.draw()
                        renderer = annotated.canvas.get_renderer()
                        for circle, line, label in zip(second.legend_handles, lines, labels):
                            center = circle.get_transform().transform(circle.get_xydata())[0]
                            radius = circle.get_markersize() / 2 * dpi / 72
                            endpoints = line.get_transform().transform(line.get_xydata())
                            np.testing.assert_allclose(endpoints[0], center)
                            np.testing.assert_allclose(endpoints[1], center + [radius, 0])
                            self.assertEqual(circle.get_markeredgewidth(), 0.4)
                            label_box = label.get_window_extent(renderer)
                            self.assertGreater(
                                label_box.y0, endpoints[0, 1] + line.get_linewidth() * dpi / 144
                            )
                            self.assertAlmostEqual(
                                (label_box.x0 + label_box.x1) / 2, center[0] + radius * 0.45
                            )
                            corners = np.array([
                                [label_box.x0, label_box.y0], [label_box.x0, label_box.y1],
                                [label_box.x1, label_box.y0], [label_box.x1, label_box.y1],
                            ])
                            distances = np.linalg.norm(corners - center, axis=1)
                            self.assertTrue(np.all(distances < radius))
                        self.assert_legend_glyphs_fit(annotated)
                        self.assert_scatter_inside_axes(annotated)
                renderer = annotated.canvas.get_renderer()
                boxes = [legend.get_window_extent(renderer) for legend in self.legends(annotated)]
                for index, box in enumerate(boxes):
                    self.assertGreaterEqual(box.x0, annotated.bbox.x0)
                    self.assertGreaterEqual(box.y0, annotated.bbox.y0)
                    self.assertLessEqual(box.x1, annotated.bbox.x1)
                    self.assertLessEqual(box.y1, annotated.bbox.y1)
                    for other in boxes[index + 1:]:
                        self.assertFalse(box.overlaps(other))
        pd.testing.assert_frame_equal(self.data, self.source_data)

    def test_radius_label_is_above_line_inside_small_legend_circles(self):
        for radius_mm in (0.5, 1.0):
            with self.subTest(radius_mm=radius_mm):
                figure = self.plot(
                    paper_style=True, add_memory_legende=True,
                    radius=radius_mm,
                    memory_legend_radius_annotation=radius_mm,
                )
                legend = self.radius_legend(figure)
                circle = legend.legend_handles[0]
                label = legend.findobj(
                    match=lambda artist: artist.get_gid() == "memory-legend-radius-label"
                )[0]
                line = legend.findobj(
                    match=lambda artist: artist.get_gid() == "memory-legend-radius-line"
                )[0]
                for dpi in (100, 200, 300):
                    figure.set_dpi(dpi)
                    figure.canvas.draw()
                    center = circle.get_transform().transform(circle.get_xydata())[0]
                    radius = radius_mm * dpi / 25.4
                    box = label.get_window_extent(figure.canvas.get_renderer())
                    self.assertGreater(box.y0, center[1] + line.get_linewidth() * dpi / 144)
                    self.assertAlmostEqual((box.x0 + box.x1) / 2, center[0] + radius * 0.45)
                    corners = np.array([
                        [box.x0, box.y0], [box.x0, box.y1],
                        [box.x1, box.y0], [box.x1, box.y1],
                    ])
                    self.assertTrue(np.all(np.linalg.norm(corners - center, axis=1) < radius))
                    self.assertEqual(circle.get_markeredgewidth(), 0.4)

    def test_reference_radius_matches_requested_examples_and_activates_bubbles(self):
        self.data[plotting.MEMORY_COLUMN] = np.resize([256.0, 1024.0, 4096.0], len(self.data))
        figure = self.plot(add_memory_legende=True, paper_style=True)
        bubbles = self.scatter_artists(figure, 3)[0]
        self.assertTrue(self.scatter_artists(figure, 4))
        diameter_path = bubbles.get_paths()[0].get_extents().width
        radii_mm = (
            diameter_path * bubbles.get_transforms()[:, 0, 0] / figure.dpi * 25.4 / 2.0
        )
        np.testing.assert_allclose(radii_mm[:3], [0.5, 1.0, 2.0])
        legend = self.radius_legend(figure)
        self.assertEqual([label.get_text() for label in legend.texts], [
            "1 mm radius = 1024 MiB",
        ])
        glyph = legend.findobj(match=Line2D)[0]
        marker = MarkerStyle(glyph.get_marker())
        path = marker.get_path().transformed(marker.get_transform())
        radius_mm = path.get_extents().width * glyph.get_markersize() / 72.0 * 25.4 / 2.0
        self.assertAlmostEqual(radius_mm, 1.0)

    def test_synapse_user_command_has_one_scale_even_with_conflicting_annotation_radius(self):
        options = dict(
            paper_style=True, add_memory_legende=True,
            model_center_markers=True, model_center_colors=True,
            show_model_marker_legend=True, memory_legend_title="Peak Runtime Memory",
            memory_legend_font_size=4, radius=2, runtime_memory=2048,
            axis_label_font_size=6, remove_title=True, memory_legend_compact_label=True,
        )
        memory = self.data[plotting.MEMORY_COLUMN].to_numpy()
        expected_radii_mm = 2.0 * np.sqrt(memory / 2048.0)
        baseline = None
        for annotation_radius in (2.0, 1.0, 3.0):
            with self.subTest(annotation_radius=annotation_radius):
                figure = self.plot(
                    memory_legend_radius_annotation=annotation_radius, **options,
                )
                if baseline is None:
                    baseline = figure
                else:
                    np.testing.assert_array_equal(
                        np.asarray(figure.canvas.buffer_rgba()),
                        np.asarray(baseline.canvas.buffer_rgba()),
                    )
                bubbles = self.scatter_artists(figure, 3)[0]
                np.testing.assert_array_equal(bubbles.get_offsets(), self.data[[
                    plotting.THROUGHPUT_COLUMN, plotting.DATASET_COLUMNS["Synapse"],
                ]].to_numpy())
                diameter_px = (
                    bubbles.get_paths()[0].get_extents().width
                    * bubbles.get_transforms()[:, 0, 0]
                )
                radii_mm = diameter_px / figure.dpi * 25.4 / 2.0
                np.testing.assert_allclose(radii_mm, expected_radii_mm)
                legend = self.radius_legend(figure)
                self.assertEqual(legend.texts[0].get_text(), "r = 2mm ↔ 2048 MiB")
                self.assertEqual(legend.get_title().get_text(), "Peak Runtime Memory")
                reference = legend.legend_handles[0]
                marker = MarkerStyle(reference.get_marker())
                path = marker.get_path().transformed(marker.get_transform())
                reference_radius_mm = (
                    path.get_extents().width * reference.get_markersize() * 25.4 / 144.0
                )
                self.assertAlmostEqual(reference_radius_mm, 2.0)
                np.testing.assert_allclose(
                    (radii_mm / reference_radius_mm) ** 2, memory / 2048.0,
                )
                self.assert_legend_glyphs_fit(figure)
                self.assert_scatter_inside_axes(figure)
        pd.testing.assert_frame_equal(self.data, self.source_data)

    def test_reference_radius_preserves_geometry_colors_and_center_markers(self):
        options = dict(
            paper_style=True, memory_scaling=True, model_center_markers=True,
            model_center_colors=True, show_model_marker_legend=True,
        )
        ordinary = self.plot(**options)
        calibrated = self.plot(add_memory_legende=True, radius=1.5, runtime_memory=512, **options)
        before = self.scatter_artists(ordinary, 3)[0]
        after = self.scatter_artists(calibrated, 3)[0]
        for attribute in ("get_offsets", "get_facecolors", "get_edgecolors", "get_linewidths"):
            np.testing.assert_array_equal(getattr(before, attribute)(), getattr(after, attribute)())
        expected_radii_mm = 1.5 * np.sqrt(self.data[plotting.MEMORY_COLUMN] / 512.0)
        np.testing.assert_allclose(np.sqrt(after.get_sizes()) * 25.4 / 144.0, expected_radii_mm)
        ordinary_centers = self.scatter_artists(ordinary, 4)
        calibrated_centers = self.scatter_artists(calibrated, 4)
        self.assertEqual(len(ordinary_centers), len(calibrated_centers))
        for before, after in zip(ordinary_centers, calibrated_centers):
            for attribute in ("get_offsets", "get_sizes", "get_facecolors", "get_linewidths"):
                np.testing.assert_array_equal(
                    getattr(before, attribute)(), getattr(after, attribute)()
                )
            np.testing.assert_array_equal(
                before.get_paths()[0].vertices, after.get_paths()[0].vertices
            )
        self.assertEqual(self.radius_legend(calibrated).texts[0].get_text(),
                         "1.5 mm radius = 512 MiB")
        self.assert_scatter_inside_axes(calibrated)
        pd.testing.assert_frame_equal(self.data, self.source_data)

    def test_reference_radius_is_physical_at_multiple_dpi_and_figure_sizes(self):
        figure = self.plot(add_memory_legende=True, radius=1.25, runtime_memory=2048)
        bubbles = self.scatter_artists(figure, 3)[0]
        reference = self.radius_legend(figure).findobj(match=Line2D)[0]
        expected_radii_mm = 1.25 * np.sqrt(self.data[plotting.MEMORY_COLUMN] / 2048.0)
        for size_scale in (1.0, 1.2):
            figure.set_size_inches(6.4 * size_scale, 4.8 * size_scale)
            for dpi in (100, 200, 300):
                with self.subTest(size_scale=size_scale, dpi=dpi):
                    figure.set_dpi(dpi)
                    figure.canvas.draw()
                    path = bubbles.get_paths()[0].get_extents()
                    diameter_px = path.width * bubbles.get_transforms()[:, 0, 0]
                    np.testing.assert_allclose(diameter_px / dpi * 25.4 / 2, expected_radii_mm)
                    marker = MarkerStyle(reference.get_marker())
                    reference_path = marker.get_path().transformed(marker.get_transform())
                    diameter_px = (
                        reference_path.get_extents().width * reference.get_markersize()
                        * figure.canvas.get_renderer().points_to_pixels(1)
                    )
                    self.assertAlmostEqual(diameter_px / dpi * 25.4 / 2, 1.25)

    def test_calibrated_legend_fits_lower_right_and_avoids_synapse_data(self):
        figure = self.plot(
            paper_style=True, add_memory_legende=True,
            model_center_markers=True, model_center_colors=True,
            show_model_marker_legend=True, display_model_names=True,
        )
        axes = figure.axes[0]
        renderer = figure.canvas.get_renderer()
        legend = self.radius_legend(figure)
        box = legend.get_window_extent(renderer)
        self.assertGreaterEqual(box.x0, axes.bbox.x0)
        self.assertGreaterEqual(box.y0, axes.bbox.y0)
        self.assertLessEqual(box.x1, axes.bbox.x1)
        self.assertLessEqual(box.y1, axes.bbox.y1)
        self.assertGreater((box.x0 + box.x1) / 2, (axes.bbox.x0 + axes.bbox.x1) / 2)
        self.assertLess((box.y0 + box.y1) / 2, (axes.bbox.y0 + axes.bbox.y1) / 2)
        bubbles = self.scatter_artists(figure, 3)[0]
        centers = bubbles.get_offset_transform().transform(bubbles.get_offsets())
        radii = (np.sqrt(bubbles.get_sizes()) + bubbles.get_linewidths()) * figure.dpi / 144
        for (x_value, y_value), radius in zip(centers, radii):
            bubble_box = Bbox.from_extents(
                x_value - radius, y_value - radius, x_value + radius, y_value + radius,
            )
            self.assertFalse(box.overlaps(bubble_box))
        for annotation in figure.findobj(match=Annotation):
            self.assertFalse(box.overlaps(Text.get_window_extent(annotation, renderer)))
        for other in self.legends(figure):
            if other is not legend:
                self.assertLess(other.get_window_extent(renderer).y1, axes.bbox.y0)
        self.assert_legend_glyphs_fit(figure)
        self.assert_scatter_inside_axes(figure)

    def test_all_datasets_and_memory_legends_share_the_same_physical_calibration(self):
        memory = self.data[plotting.MEMORY_COLUMN]
        references = np.unique([memory.min(), memory.median(), memory.max()])
        factor = (2.0 * 1.25 * 72.0 / 25.4) ** 2 / 2048.0
        for right, scaled in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(right=right, scaled=scaled):
                figure = self.plot(
                    datasets=tuple(plotting.DATASET_COLUMNS), paper_style=True,
                    add_memory_legende=True, radius=1.25, runtime_memory=2048,
                    show_bubble_legend=True, right_memory_legend=right,
                    scaled_memorey_legened=scaled,
                )
                bubbles = self.scatter_artists(figure, 3)
                self.assertEqual(len(bubbles), len(plotting.DATASET_COLUMNS))
                for collection in bubbles:
                    np.testing.assert_allclose(collection.get_sizes(), memory * factor)
                legacy = next(
                    legend for legend in figure.legends
                    if legend.get_title().get_text().startswith(
                        plotting.DEFAULT_MEMORY_LEGEND_TITLE
                    )
                )
                reference_sizes = sorted(
                    glyph.get_markersize() ** 2 for glyph in legacy.legend_handles
                )
                expected_sizes = references * factor
                np.testing.assert_allclose(reference_sizes, expected_sizes)
                reference = self.radius_legend(figure).legend_handles[0]
                self.assertAlmostEqual(reference.get_markersize() * 25.4 / 144, 1.25)
                self.assert_legend_glyphs_fit(figure)
                self.assert_scatter_inside_axes(figure)

    def test_crowded_reference_legend_reserves_space_below_bubbles_and_std_caps(self):
        # Occupy the lower-right region continuously so a local move cannot fit
        # the key. Synthetic coordinates stay in this test's private dataframe.
        self.data[plotting.THROUGHPUT_COLUMN] = [0, 100, 100, 100, 100, 100, 100, 100]
        self.data[plotting.THROUGHPUT_STD_COLUMN] = [0, 10, 10, 10, 10, 10, 10, 10]
        self.data[plotting.DATASET_COLUMNS["Synapse"]] = [95, 50, 53, 56, 59, 62, 65, 68]
        self.data[plotting.MEMORY_COLUMN] = 1024.0
        original_data = self.data.copy(deep=True)
        original_padding = plotting._pad_for_marker_extents
        reserved_limits = []

        def record_padding(axes, bottom_padding_points=0):
            before = axes.get_ylim()
            original_padding(axes, bottom_padding_points=bottom_padding_points)
            if bottom_padding_points > 0:
                reserved_limits.append((before, axes.get_ylim()))

        with patch.object(plotting, "_pad_for_marker_extents", side_effect=record_padding):
            figure = self.plot(
                paper_style=True, add_memory_legende=True, show_throughput_std=True,
            )
        self.assertEqual(len(reserved_limits), 1)
        before, after = reserved_limits[0]
        self.assertLess(after[0], before[0])
        self.assertGreaterEqual(after[1], before[1])
        axes = figure.axes[0]
        renderer = figure.canvas.get_renderer()
        legend = self.radius_legend(figure)
        box = legend.get_window_extent(renderer)
        bubbles = self.scatter_artists(figure, 3)[0]
        np.testing.assert_array_equal(bubbles.get_offsets(), original_data[[
            plotting.THROUGHPUT_COLUMN, plotting.DATASET_COLUMNS["Synapse"],
        ]].to_numpy())
        reference_area = legend.findobj(match=Line2D)[0].get_markersize() ** 2
        np.testing.assert_allclose(bubbles.get_sizes(), reference_area)
        centers = bubbles.get_offset_transform().transform(bubbles.get_offsets())
        radii = (np.sqrt(bubbles.get_sizes()) + bubbles.get_linewidths()) * figure.dpi / 144
        self.assertTrue(np.all(centers[:, 1] - radii > box.y1))
        for line in axes.lines:
            if line.get_marker() != "|":
                continue
            points = line.get_transform().transform(line.get_xydata())
            half_height = (line.get_markersize() + line.get_markeredgewidth()) * figure.dpi / 144
            # The horizontal error segments share each cap's y-coordinate.
            self.assertTrue(np.all(points[:, 1] - half_height > box.y1))
            self.assertTrue(np.all(points[:, 0] >= axes.bbox.x0))
            self.assertTrue(np.all(points[:, 0] <= axes.bbox.x1))
        self.assert_legend_glyphs_fit(figure)
        self.assert_scatter_inside_axes(figure)
        pd.testing.assert_frame_equal(self.data, original_data)

    def test_reference_radius_rejects_invalid_physical_inputs(self):
        for keyword in ("radius", "runtime_memory", "memory_legend_radius_annotation"):
            for value in (0.0, -1.0, np.nan, np.inf):
                with self.subTest(keyword=keyword, value=value):
                    with self.assertRaisesRegex(ValueError, keyword):
                        self.plot(add_memory_legende=True, **{keyword: value})

    def test_highlight_preserves_coordinates_areas_and_center_markers(self):
        options = dict(memory_scaling=True, model_center_markers=True, model_center_colors=True)
        ordinary = self.plot(**options)
        highlighted = self.plot(highlight_ours=True, **options)
        normal_bubbles = self.scatter_artists(ordinary, 3)[0]
        highlighted_bubbles = self.scatter_artists(highlighted, 3)[0]
        np.testing.assert_array_equal(
            normal_bubbles.get_offsets(), highlighted_bubbles.get_offsets()
        )
        np.testing.assert_array_equal(normal_bubbles.get_sizes(), highlighted_bubbles.get_sizes())
        ours_index = self.data["Model"].tolist().index(plotting.OURS_MODEL)
        normal_faces = np.broadcast_to(normal_bubbles.get_facecolors(), (len(self.data), 4))
        highlighted_faces = np.broadcast_to(
            highlighted_bubbles.get_facecolors(), (len(self.data), 4)
        )
        self.assertFalse(np.array_equal(normal_faces[ours_index], highlighted_faces[ours_index]))
        np.testing.assert_array_equal(
            np.delete(normal_faces, ours_index, axis=0),
            np.delete(highlighted_faces, ours_index, axis=0),
        )
        np.testing.assert_allclose(
            normal_bubbles.get_edgecolors()[ours_index, :3],
            to_rgba(plotting.OURS_OUTLINE_COLOR)[:3],
        )
        self.assertEqual(normal_bubbles.get_linewidths()[ours_index], plotting.OURS_OUTLINE_WIDTH)
        ordinary_centers = self.scatter_artists(ordinary, 4)
        highlighted_centers = self.scatter_artists(highlighted, 4)
        self.assertEqual(len(ordinary_centers), len(highlighted_centers))
        for before, after in zip(ordinary_centers, highlighted_centers):
            for attribute in ("get_offsets", "get_sizes", "get_facecolors"):
                np.testing.assert_array_equal(
                    getattr(before, attribute)(), getattr(after, attribute)()
                )
            np.testing.assert_array_equal(
                before.get_paths()[0].vertices, after.get_paths()[0].vertices
            )
        pd.testing.assert_frame_equal(self.data, self.source_data)

    def test_multiple_dataset_highlight_keeps_dataset_fills_and_one_direct_label(self):
        options = dict(datasets=tuple(plotting.DATASET_COLUMNS), memory_scaling=True)
        ordinary = self.plot(**options)
        highlighted = self.plot(highlight_ours=True, display_model_names=True, **options)
        for before, after in zip(
            self.scatter_artists(ordinary, 3), self.scatter_artists(highlighted, 3)
        ):
            before_faces = np.broadcast_to(before.get_facecolors(), (len(self.data), 4))
            after_faces = np.broadcast_to(after.get_facecolors(), (len(self.data), 4))
            np.testing.assert_array_equal(before_faces, after_faces)
        ours_labels = [
            " ".join(label.get_text().split()) for label in highlighted.axes[0].texts
            if "Lightweight" in label.get_text()
        ]
        self.assertEqual(ours_labels, ["Lightweight TransUNet (ours)"])

    def test_remove_title_skips_fitting_and_reclaims_axes_space(self):
        titled = self.plot(paper_style=True)
        untitled = self.plot(
            paper_style=True, remove_title=True, title="Very long title " * 100,
            title_font_size=200,
        )
        self.assertEqual(untitled.axes[0].get_title(), "")
        self.assertGreater(untitled.axes[0].bbox.height, titled.axes[0].bbox.height)

    def test_explicit_typography_and_labels_are_honored(self):
        figure = self.plot(
            paper_style=True, title="First line\\nSecond line", title_font_size=12,
            xlabel="Custom throughput", ylabel="Custom Dice", axis_label_font_size=11,
            memory_scaling=True, model_center_markers=True, center_icon_size=13,
            show_model_marker_legend=True, model_marker_legend_font_size=9,
        )
        axes = figure.axes[0]
        self.assertEqual(axes.title.get_fontsize(), 12)
        self.assertEqual(axes.title.get_ha(), "center")
        self.assertEqual(axes.get_xlabel(), "Custom throughput")
        self.assertEqual(axes.get_ylabel(), "Custom Dice")
        labels = (
            [axes.xaxis.label, axes.yaxis.label]
            + axes.get_xticklabels() + axes.get_yticklabels()
        )
        self.assertTrue(all(label.get_fontsize() == 11 for label in labels))
        self.assertTrue(all(text.get_fontsize() == 9 for text in self.legends(figure)[0].texts))
        for marker in self.scatter_artists(figure, 4):
            np.testing.assert_array_equal(marker.get_sizes(), [13])

    def test_final_layout_fits_bubbles_legends_and_standard_deviations(self):
        self.data.loc[0, plotting.THROUGHPUT_STD_COLUMN] = 400.0
        figure = self.plot(
            paper_style=True, memory_scaling=True, show_throughput_std=True,
            model_center_markers=True, show_model_marker_legend=True,
            model_marker_legend_font_size=10, show_bubble_legend=True, highlight_ours=True,
        )
        self.assertAlmostEqual(figure.get_figwidth(), 3.5)
        self.assert_scatter_inside_axes(figure)
        self.assert_legend_glyphs_fit(figure)
        axes = figure.axes[0]
        renderer = figure.canvas.get_renderer()
        memory_legend = next(
            legend for legend in self.legends(figure)
            if legend.get_title().get_text() == plotting.DEFAULT_MEMORY_LEGEND_TITLE
        )
        reference_areas = sorted(
            glyph.get_markersize() ** 2 for glyph in memory_legend.legend_handles
        )
        memory = self.data[plotting.MEMORY_COLUMN]
        np.testing.assert_allclose(
            reference_areas,
            np.array([memory.min(), memory.median(), memory.max()])
            * plotting.MAX_MEMORY_MARKER_AREA / memory.max(),
        )
        legend_boxes = [legend.get_window_extent(renderer) for legend in self.legends(figure)]
        for index, box in enumerate(legend_boxes):
            self.assertGreaterEqual(box.x0, figure.bbox.x0 - 0.01)
            self.assertGreaterEqual(box.y0, figure.bbox.y0 - 0.01)
            self.assertLessEqual(box.x1, figure.bbox.x1 + 0.01)
            self.assertLessEqual(box.y1, figure.bbox.y1 + 0.01)
            self.assertFalse(box.overlaps(axes.xaxis.label.get_window_extent(renderer)))
            for other in legend_boxes[index + 1:]:
                self.assertFalse(box.overlaps(other))
        for line in axes.lines:
            if line.get_marker() != "|":
                continue
            points = line.get_transform().transform(line.get_xydata())
            half_height = (line.get_markersize() + line.get_markeredgewidth()) * figure.dpi / 144
            self.assertTrue(np.all(points[:, 0] >= axes.bbox.x0))
            self.assertTrue(np.all(points[:, 0] <= axes.bbox.x1))
            self.assertTrue(np.all(points[:, 1] - half_height >= axes.bbox.y0))
            self.assertTrue(np.all(points[:, 1] + half_height <= axes.bbox.y1))
        model_legend = next(
            legend for legend in self.legends(figure) if len(legend.texts) == len(self.data)
        )
        model_rows = {
            round(label.get_window_extent(renderer).y0, 3) for label in model_legend.texts
        }
        self.assertGreater(len(model_rows), 2)

    def test_all_datasets_legends_and_annotations_have_separate_space(self):
        figure = self.plot(
            datasets=tuple(plotting.DATASET_COLUMNS), paper_style=True,
            memory_scaling=True, model_center_markers=True, model_center_colors=True,
            show_model_marker_legend=True, show_bubble_legend=True,
            display_model_names=True, highlight_ours=True,
        )
        self.assertAlmostEqual(figure.get_figwidth(), 7.16)
        self.assert_scatter_inside_axes(figure)
        self.assert_legend_glyphs_fit(figure)
        renderer = figure.canvas.get_renderer()
        annotation_boxes = [
            Text.get_window_extent(annotation, renderer)
            for annotation in figure.findobj(match=Annotation)
        ]
        other_boxes = [legend.get_window_extent(renderer) for legend in self.legends(figure)]
        axes = figure.axes[0]
        other_boxes += [
            axes.xaxis.label.get_window_extent(renderer),
            axes.yaxis.label.get_window_extent(renderer),
        ]
        for index, box in enumerate(annotation_boxes):
            for other in annotation_boxes[index + 1:] + other_boxes:
                self.assertFalse(box.overlaps(other))

    def test_annotation_connectors_do_not_cross_other_labels(self):
        figure = self.plot(
            paper_style=True, memory_scaling=True, model_center_markers=True,
            model_center_colors=True, show_model_marker_legend=True,
            show_bubble_legend=True, display_model_names=True, highlight_ours=True,
            # This readable override requires connectors for the crowded models.
            model_name_font_size=9,
        )
        # Exercise output-resolution transforms after layout, as raster export does.
        figure.set_dpi(200)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        annotations = figure.findobj(match=Annotation)
        connectors = [
            annotation for annotation in annotations if annotation.arrow_patch is not None
        ]
        self.assertTrue(connectors)
        for annotation in connectors:
            arrow = annotation.arrow_patch
            path = arrow.get_path().transformed(arrow.get_transform())
            for other in annotations:
                if other is annotation:
                    continue
                with self.subTest(connector=annotation.get_text(), label=other.get_text()):
                    self.assertFalse(
                        path.intersects_bbox(Text.get_window_extent(other, renderer), filled=False)
                    )


if __name__ == "__main__":
    unittest.main()
