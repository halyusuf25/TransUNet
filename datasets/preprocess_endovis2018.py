#!/usr/bin/env python3
import argparse
import re
import shutil
from pathlib import Path

SEQ_RE = re.compile(r"^seq_(\d+)$")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def unique_dest(dest: Path) -> Path:
    """Avoid overwriting: if dest exists, append __dupN before extension."""
    if not dest.exists():
        return dest
    stem, suf = dest.stem, dest.suffix
    i = 1
    while True:
        cand = dest.with_name(f"{stem}__dup{i}{suf}")
        if not cand.exists():
            return cand
        i += 1

def copy_file(src: Path, out_dir: Path, new_name: str, dry_run: bool, preserve_metadata: bool) -> Path:
    ensure_dir(out_dir)
    dest = unique_dest(out_dir / new_name)
    if dry_run:
        return dest

    # If you don't want to preserve metadata/permissions, use copyfile().
    # If you do, use copy2().
    if preserve_metadata:
        shutil.copy2(src, dest)
    else:
        shutil.copyfile(src, dest)

    return dest

def main():
    ap = argparse.ArgumentParser(
        description="Copy EndoVis2018 left_frames to dataset/train and labels to dataset/train/labels with seq prefix."
    )
    ap.add_argument("--root", type=Path, default=Path("/data/shared/EndoVis2018/"),
                    help="Root directory that contains miccai_challenge_release_* (default: '/data/shared/EndoVis2018/')")
    ap.add_argument("--release_glob", type=str, default="miccai_challenge_release_2*",
                    help='Glob for release dirs (default: "miccai_challenge_release_2*")')
    ap.add_argument("--images_out", type=Path, default=Path("/data/shared/EndoVis_2018/train/imgs"),
                    help="Output dir for left_frames images (default: /data/shared/EndoVis_2018/train/imgs)")
    ap.add_argument("--labels_out", type=Path, default=Path("/data/shared/EndoVis_2018/train/labels"),
                    help="Output dir for labels (default: /data/shared/EndoVis_2018/train/labels)")
    ap.add_argument("--preserve_metadata", action="store_true",
                    help="Preserve metadata (uses shutil.copy2). Default is off (uses shutil.copyfile).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print what would happen without copying.")
    args = ap.parse_args()

    root = args.root.resolve()
    releases = sorted(root.glob(args.release_glob))
    if not releases:
        raise SystemExit(f"No release dirs found under {root} matching: {args.release_glob}")

    images_out = (root / args.images_out).resolve() if not args.images_out.is_absolute() else args.images_out
    labels_out = (root / args.labels_out).resolve() if not args.labels_out.is_absolute() else args.labels_out

    n_imgs = 0
    n_labs = 0

    for rel in releases:
        if not rel.is_dir():
            continue

        for seq_dir in sorted(rel.glob("seq_*")):
            if not seq_dir.is_dir():
                continue

            m = SEQ_RE.match(seq_dir.name)
            if not m:
                continue
            seq_num = m.group(1)

            # 1) left_frames -> images_out (COPY)
            left_dir = seq_dir / "left_frames"
            if left_dir.is_dir():
                for img in sorted(left_dir.glob("frame*.png")):
                    new_name = f"seq{seq_num}_{img.name}"
                    copy_file(img, images_out, new_name, args.dry_run, args.preserve_metadata)
                    n_imgs += 1

            # 2) labels -> labels_out (COPY)
            lab_dir = seq_dir / "labels"
            if lab_dir.is_dir():
                for lab in sorted(lab_dir.glob("frame*.png")):
                    new_name = f"seq{seq_num}_{lab.name}"
                    copy_file(lab, labels_out, new_name, args.dry_run, args.preserve_metadata)
                    n_labs += 1

    if args.dry_run:
        print("[DRY RUN] No files were copied.\n")

    print("Done.")
    print(f"Release dirs matched: {len(releases)}")
    print(f"Images copied (left_frames): {n_imgs} -> {images_out}")
    print(f"Labels copied: {n_labs} -> {labels_out}")

if __name__ == "__main__":
    main()
