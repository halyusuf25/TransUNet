#!/usr/bin/env python3
"""Print sample ACDC z-spacing values used for 3D volume evaluation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_ROOT = "/data/halyusuf/data/ACDC"
SPACING_KEYS = (
    "voxelspacing_zyx",
    "spacing_zyx",
    "voxelspacing",
    "spacing",
    "spacing_mm",
    "pixdim",
    "zooms",
)
SPACING_ZYX_KEYS = {"voxelspacing_zyx", "spacing_zyx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print sample z-spacing values for ACDC volume cases."
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="ACDC dataset root. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--volume-dir",
        default=None,
        help="Optional volume H5 directory. Defaults to ROOT/ACDC_training_volumes.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Number of cases to print. Use 0 to scan all discovered cases.",
    )
    parser.add_argument(
        "--acdc_zspacing",
        type=float,
        default=5.0,
        help="Fallback z-spacing in mm to report when spacing metadata is unavailable.",
    )
    return parser.parse_args()


def case_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def normalize_spacing_zyx(spacing: object, key: str) -> Tuple[float, float, float]:
    values = np.asarray(spacing, dtype=np.float32).reshape(-1)
    if key == "pixdim" and values.size >= 4:
        values = values[1:4]
    else:
        values = values[:3]

    if values.size != 3:
        raise ValueError("spacing key {} does not contain 3 spatial values".format(key))

    if key not in SPACING_ZYX_KEYS:
        values = values[::-1]

    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("spacing key {} has invalid values {}".format(key, values.tolist()))

    return tuple(float(v) for v in values)


def format_spacing(spacing_zyx: Sequence[float]) -> str:
    return "({:.6g}, {:.6g}, {:.6g})".format(*spacing_zyx)


def iter_h5_spacing_values(h5f) -> Iterable[Tuple[str, object, str]]:
    for key in SPACING_KEYS:
        if key in h5f.attrs:
            yield key, h5f.attrs[key], "h5 attr {}".format(key)
        if key in h5f:
            yield key, h5f[key][()], "h5 dataset {}".format(key)
        for dataset_key in ("image", "label"):
            if dataset_key in h5f and key in h5f[dataset_key].attrs:
                yield key, h5f[dataset_key].attrs[key], "h5 {} attr {}".format(dataset_key, key)


def spacing_from_h5(path: Path) -> Tuple[Optional[Tuple[float, float, float]], str]:
    try:
        import h5py
    except ImportError:
        return None, "h5py is not installed"

    try:
        with h5py.File(path, "r") as h5f:
            errors: List[str] = []
            for key, raw_spacing, source in iter_h5_spacing_values(h5f):
                try:
                    return normalize_spacing_zyx(raw_spacing, key), source
                except ValueError as exc:
                    errors.append(str(exc))
    except OSError as exc:
        return None, "could not read H5: {}".format(exc)

    if errors:
        return None, "; ".join(errors)
    return None, "no spacing metadata in H5"


def nifti_candidates(root: Path, volume_path: Path) -> List[Path]:
    stem = case_stem(volume_path)
    patient_id = stem.split("_")[0]
    directories = [
        root,
        root / "ACDC_training_volumes",
        root / patient_id,
        root / "training" / patient_id,
        root / "database" / "training" / patient_id,
        root / "ACDC_training" / patient_id,
    ]

    candidates: List[Path] = []
    for directory in directories:
        candidates.append(directory / "{}.nii.gz".format(stem))
        candidates.append(directory / "{}.nii".format(stem))

    for suffix in (".nii.gz", ".nii"):
        candidates.extend(root.rglob("{}{}".format(stem, suffix)))

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def spacing_from_nifti(path: Path) -> Tuple[Optional[Tuple[float, float, float]], str]:
    nibabel_error = None
    try:
        import nibabel as nib

        spacing = nib.load(str(path)).header.get_zooms()[:3]
        return normalize_spacing_zyx(spacing, "zooms"), "nifti header {}".format(path)
    except ImportError as exc:
        nibabel_error = "nibabel is not installed"
    except Exception as exc:
        nibabel_error = "nibabel could not read {}: {}".format(path, exc)

    try:
        import SimpleITK as sitk

        spacing = sitk.ReadImage(str(path)).GetSpacing()[:3]
        return normalize_spacing_zyx(spacing, "spacing"), "nifti header {}".format(path)
    except ImportError:
        return None, "{}; SimpleITK is not installed".format(nibabel_error)
    except Exception as exc:
        return None, "{}; SimpleITK could not read {}: {}".format(nibabel_error, path, exc)


def spacing_from_matching_nifti(
    root: Path, volume_path: Path
) -> Tuple[Optional[Tuple[float, float, float]], str]:
    checked = 0
    last_reason = "no matching original NIfTI header found"
    for candidate in nifti_candidates(root, volume_path):
        if not candidate.exists():
            continue
        checked += 1
        spacing, reason = spacing_from_nifti(candidate)
        if spacing is not None:
            return spacing, reason
        last_reason = reason

    if checked:
        return None, last_reason
    return None, "no matching original NIfTI header found"


def discover_cases(root: Path, volume_dir: Optional[str]) -> List[Path]:
    if volume_dir is None:
        resolved_volume_dir = root / "ACDC_training_volumes"
    else:
        resolved_volume_dir = Path(volume_dir)

    if resolved_volume_dir.exists():
        h5_cases = sorted(resolved_volume_dir.glob("*.h5"))
        if h5_cases:
            return h5_cases

    nifti_cases = [
        path
        for path in root.rglob("patient*_frame*.nii*")
        if "_gt" not in path.name
    ]
    return sorted(nifti_cases)


def sample_cases(cases: Sequence[Path], max_samples: int) -> Sequence[Path]:
    if max_samples == 0:
        return cases
    return cases[:max_samples]


def inspect_case(root: Path, path: Path) -> Tuple[Optional[Tuple[float, float, float]], str]:
    if path.suffix == ".h5":
        spacing, reason = spacing_from_h5(path)
        if spacing is not None:
            return spacing, reason
        nifti_spacing, nifti_reason = spacing_from_matching_nifti(root, path)
        if nifti_spacing is not None:
            return nifti_spacing, nifti_reason
        return None, "{}; {}".format(reason, nifti_reason)

    return spacing_from_nifti(path)


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.acdc_zspacing) or args.acdc_zspacing <= 0:
        raise ValueError("--acdc_zspacing must be a positive finite value")

    root = Path(args.root)
    cases = discover_cases(root, args.volume_dir)
    if not cases:
        print("WARNING: no ACDC volume H5 or NIfTI cases found under {}".format(root))
        return 1

    selected_cases = sample_cases(cases, args.max_samples)
    print("ACDC root: {}".format(root))
    print("Discovered cases: {}".format(len(cases)))
    print("Printing samples: {}".format(len(selected_cases)))

    found = 0
    missing = 0
    for path in selected_cases:
        spacing_zyx, reason = inspect_case(root, path)
        name = case_stem(path)
        if spacing_zyx is None:
            missing += 1
            fallback_zyx = (float(args.acdc_zspacing), 1.0, 1.0)
            print(
                "WARNING: {} z-spacing unavailable ({}); fallback voxelspacing_zyx={}".format(
                    name,
                    reason,
                    format_spacing(fallback_zyx),
                )
            )
            continue

        found += 1
        z_spacing = spacing_zyx[0]
        if not math.isfinite(z_spacing) or z_spacing <= 0:
            missing += 1
            found -= 1
            print("WARNING: {} invalid z-spacing {}".format(name, z_spacing))
            continue
        print(
            "{} z_spacing_mm={:.6g} voxelspacing_zyx={} source={}".format(
                name,
                z_spacing,
                format_spacing(spacing_zyx),
                reason,
            )
        )

    print("Summary: found_z_spacing={} missing_z_spacing={}".format(found, missing))
    if found == 0:
        print(
            "WARNING: no ACDC z-spacing values were available in the sampled cases; "
            "fallback --acdc_zspacing={:.6g} would be used".format(args.acdc_zspacing)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
