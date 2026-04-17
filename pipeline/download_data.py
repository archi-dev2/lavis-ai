#!/usr/bin/env python3
"""
download_data.py — Download and prepare MS COCO (Karpathy split) dataset.

Creates:
    data/coco/images/train2014/
    data/coco/images/val2014/
    data/coco/annotations/
    data/coco/dataset_coco.json   (Karpathy split)

Skips downloads if data already exists.
"""

import os
import sys
import json
import glob
import zipfile
import hashlib
import argparse
import urllib.request
from pathlib import Path


# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = os.environ.get("COCO_DATA_DIR", os.path.join(os.getcwd(), "data", "coco"))

COCO_URLS = {
    "train2014": "http://images.cocodataset.org/zips/train2014.zip",
    "val2014": "http://images.cocodataset.org/zips/val2014.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}

KARPATHY_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"


class DownloadProgressBar:
    """Simple progress bar for urllib downloads."""

    def __init__(self, desc="Downloading"):
        self.desc = desc
        self.last_pct = -1

    def __call__(self, block_num, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(100, int(block_num * block_size * 100 / total_size))
        if pct != self.last_pct and pct % 5 == 0:
            downloaded_mb = block_num * block_size / 1e6
            total_mb = total_size / 1e6
            print(f"\r  {self.desc}: {pct:3d}% ({downloaded_mb:.0f}/{total_mb:.0f} MB)", end="", flush=True)
            self.last_pct = pct


def download_file(url, dest_path, desc="Downloading"):
    """Download a file with progress bar. Skip if exists."""
    if os.path.exists(dest_path):
        print(f"  Already exists: {os.path.basename(dest_path)} — skipping.")
        return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading: {url}")

    progress = DownloadProgressBar(desc=os.path.basename(dest_path))
    urllib.request.urlretrieve(url, dest_path, reporthook=progress)
    print()  # newline after progress bar

    size_mb = os.path.getsize(dest_path) / 1e6
    print(f"  Saved: {dest_path} ({size_mb:.1f} MB)")
    return dest_path


def extract_zip(zip_path, extract_to, delete_after=False):
    """Extract a zip file with progress."""
    print(f"  Extracting: {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members):
            zf.extract(member, extract_to)
            if (i + 1) % max(1, total // 20) == 0:
                pct = int((i + 1) * 100 / total)
                print(f"\r  Extracting: {pct:3d}% ({i+1}/{total})", end="", flush=True)
    print(f"\r  Extracted {total} files to {extract_to}")

    if delete_after:
        os.remove(zip_path)
        print(f"  Cleaned up: {os.path.basename(zip_path)}")


def download_coco_images(data_dir):
    """Download COCO train2014 and val2014 images."""
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for split in ["train2014", "val2014"]:
        split_dir = os.path.join(images_dir, split)
        if os.path.isdir(split_dir) and len(os.listdir(split_dir)) > 1000:
            sample_count = len(os.listdir(split_dir))
            print(f"  {split}/ already exists ({sample_count} files) — skipping.")
            continue

        zip_path = os.path.join(data_dir, f"{split}.zip")
        download_file(COCO_URLS[split], zip_path, desc=split)
        extract_zip(zip_path, images_dir, delete_after=True)


def download_annotations(data_dir):
    """Download COCO annotations."""
    ann_dir = os.path.join(data_dir, "annotations")
    if os.path.isdir(ann_dir) and glob.glob(os.path.join(ann_dir, "*.json")):
        n_files = len(glob.glob(os.path.join(ann_dir, "*.json")))
        print(f"  annotations/ already exists ({n_files} JSON files) — skipping.")
        return

    zip_path = os.path.join(data_dir, "annotations_trainval2014.zip")
    download_file(COCO_URLS["annotations"], zip_path, desc="annotations")
    extract_zip(zip_path, data_dir, delete_after=True)


def download_karpathy_split(data_dir):
    """Download Karpathy COCO split JSON."""
    karpathy_json = os.path.join(data_dir, "dataset_coco.json")
    if os.path.exists(karpathy_json):
        print(f"  Karpathy split already exists — skipping.")
        return karpathy_json

    zip_path = os.path.join(data_dir, "caption_datasets.zip")
    download_file(KARPATHY_URL, zip_path, desc="karpathy_split")
    extract_zip(zip_path, data_dir, delete_after=True)

    # The zip contains dataset_coco.json (and other dataset JSONs)
    if not os.path.exists(karpathy_json):
        # Search for it recursively
        candidates = glob.glob(os.path.join(data_dir, "**", "dataset_coco.json"), recursive=True)
        if candidates:
            import shutil
            shutil.move(candidates[0], karpathy_json)
            print(f"  Moved Karpathy JSON to: {karpathy_json}")
        else:
            print("  WARNING: dataset_coco.json not found in downloaded archive.")
            return None

    return karpathy_json


def validate_dataset(data_dir):
    """Validate the dataset structure and print summary."""
    print("\n  Validating dataset structure...")
    issues = []

    # Check images
    for split in ["train2014", "val2014"]:
        split_dir = os.path.join(data_dir, "images", split)
        if not os.path.isdir(split_dir):
            # Also check if images are directly under data_dir
            alt_dir = os.path.join(data_dir, split)
            if os.path.isdir(alt_dir):
                print(f"  NOTE: {split} found at {alt_dir} (non-standard path)")
            else:
                issues.append(f"Missing: images/{split}/")
        else:
            count = len([f for f in os.listdir(split_dir) if f.endswith(".jpg")])
            print(f"  images/{split}/: {count:,} images")

    # Check annotations
    ann_dir = os.path.join(data_dir, "annotations")
    if os.path.isdir(ann_dir):
        ann_files = glob.glob(os.path.join(ann_dir, "*.json"))
        print(f"  annotations/: {len(ann_files)} JSON files")
    else:
        issues.append("Missing: annotations/")

    # Check Karpathy split
    karpathy_json = os.path.join(data_dir, "dataset_coco.json")
    if os.path.exists(karpathy_json):
        with open(karpathy_json, "r") as f:
            data = json.load(f)
        n_images = len(data.get("images", []))
        splits = {}
        for img in data.get("images", []):
            s = img.get("split", "unknown")
            splits[s] = splits.get(s, 0) + 1
        print(f"  dataset_coco.json: {n_images:,} images")
        for s, c in sorted(splits.items()):
            print(f"    {s}: {c:,}")
    else:
        issues.append("Missing: dataset_coco.json (Karpathy split)")

    if issues:
        print(f"\n  ISSUES FOUND:")
        for issue in issues:
            print(f"    ✗ {issue}")
        return False

    print(f"\n  ✓ Dataset validation passed.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download MS COCO (Karpathy split)")
    parser.add_argument(
        "--data-dir", type=str, default=DEFAULT_DATA_DIR,
        help=f"Root directory for dataset (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--skip-images", action="store_true",
        help="Skip image download (if you already have them)"
    )
    parser.add_argument(
        "--skip-annotations", action="store_true",
        help="Skip annotation download"
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("MS COCO Dataset Download (Karpathy Split)")
    print("=" * 60)
    print(f"  Target directory: {data_dir}\n")

    # 1. Images
    if not args.skip_images:
        print("[1/3] Downloading COCO images...")
        download_coco_images(data_dir)
    else:
        print("[1/3] Skipping image download (--skip-images)")

    # 2. Annotations
    if not args.skip_annotations:
        print("\n[2/3] Downloading COCO annotations...")
        download_annotations(data_dir)
    else:
        print("\n[2/3] Skipping annotation download (--skip-annotations)")

    # 3. Karpathy split
    print("\n[3/3] Downloading Karpathy split...")
    download_karpathy_split(data_dir)

    # 4. Validate
    print("\n" + "─" * 60)
    valid = validate_dataset(data_dir)
    print("─" * 60)

    if valid:
        print("\nDataset ready for training.")
    else:
        print("\nWARNING: Dataset has issues. Training may fail.")
        sys.exit(1)


if __name__ == "__main__":
    main()
