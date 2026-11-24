import unittest
from unittest.mock import patch

import numpy as np
from medpy.metric import binary

from utils import calculate_metric_percase


class VolumeMetricsTests(unittest.TestCase):
    def test_foreground_crop_matches_full_grid_metrics(self):
        label = np.zeros((14, 20, 30), dtype=np.uint8)
        pred = np.zeros_like(label, dtype=np.float32)
        # Include a component touching the image edge and a disconnected false
        # positive beyond the ground-truth bounds. MedPy treats nonzero as true.
        label[0:5, 6:12, 8:19] = 3
        pred[1:6, 7:14, 11:21] = 0.5
        pred[8, 3, 22] = 1
        for spacing in (None, (5.0, 1.2, 0.75)):
            with self.subTest(spacing=spacing):
                expected = (
                    binary.dc(pred, label),
                    binary.hd95(pred, label, voxelspacing=spacing),
                    binary.jc(pred, label),
                )
                with patch.object(binary, "hd95", wraps=binary.hd95) as hd95:
                    actual = calculate_metric_percase(pred, label, spacing)
                hd95.assert_called_once()
                crop_pred, crop_label = hd95.call_args.args
                np.testing.assert_array_equal(crop_pred, pred[0:9, 3:14, 8:23])
                np.testing.assert_array_equal(crop_label, label[0:9, 3:14, 8:23])
                np.testing.assert_equal(actual, expected)

    def test_empty_masks_keep_full_volume_penalty(self):
        empty = np.zeros((8, 5, 11), dtype=bool)
        organ = empty.copy()
        organ[1, 2, 3] = True
        spacing = (2.0, 0.5, 1.0)
        full_diagonal = np.linalg.norm([14.0, 2.0, 10.0])
        for pred, label in ((empty, organ), (organ, empty)):
            np.testing.assert_equal(
                calculate_metric_percase(pred, label, spacing), (0.0, full_diagonal, 0.0)
            )
        np.testing.assert_equal(calculate_metric_percase(empty, empty, spacing), (1.0, np.nan, 1.0))


if __name__ == "__main__":
    unittest.main()
