#!/usr/bin/env python3
"""
run_all.py — Single-command pipeline runner for BLIP-2 training.

Runs everything sequentially:
  1. Check environment (GPU, packages, repo)
  2. Download dataset (skips if already present)
  3. Train model (auto-resumes if checkpoint exists)

Usage:
    python pipeline/run_all.py
    python pipeline/run_all.py --resume --batch-size 4 --epochs 10
    python pipeline/run_all.py --skip-download
    python pipeline/run_all.py --inference-only --num-samples 16
"""

import os
import sys
import json
import time
import glob
import argparse
import logging
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_all")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════════════════════
# 1. ENVIRONMENT CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_environment():
    """Check Python, GPU, and required packages."""
    logger.info("=" * 60)
    logger.info("Step 1/3: Checking environment")
    logger.info("=" * 60)

    # Python version
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        logger.error(f"Python 3.8+ required, got {major}.{minor}")
        sys.exit(1)
    logger.info(f"  Python: {sys.version.split()[0]}")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info(f"  GPU: {gpu_name} ({vram:.1f} GB)")
            logger.info(f"  CUDA: {torch.version.cuda}")
        else:
            logger.warning("  GPU: Not available — training will be very slow!")
    except ImportError:
        logger.error("  PyTorch not installed. Run setup_env.py first.")
        sys.exit(1)

    # Check LAVIS
    repo_dir = os.environ.get("LAVIS_REPO_DIR", os.path.join(SCRIPT_DIR, ".."))
    if os.path.isdir(os.path.join(SCRIPT_DIR, "..", "lavis")):
        repo_dir = os.path.join(SCRIPT_DIR, "..")

    sys.path.insert(0, os.path.abspath(repo_dir))
    try:
        import lavis  # noqa: F401
        logger.info(f"  LAVIS: imported from {repo_dir}")
    except ImportError:
        logger.error("  LAVIS not importable. Running setup_env.py...")
        subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "setup_env.py")], check=True)

    # Required packages
    required = ["torch", "transformers", "timm", "peft", "tqdm", "PIL"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.warning(f"  Missing packages: {missing}")
        logger.info("  Installing via setup_env.py...")
        subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "setup_env.py")], check=True)
    else:
        logger.info("  All required packages available.")

    logger.info("  Environment check passed.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data(data_dir, skip_download=False):
    """Download and validate COCO dataset if not present."""
    logger.info("=" * 60)
    logger.info("Step 2/3: Preparing dataset")
    logger.info("=" * 60)

    if skip_download:
        logger.info("  Skipping download (--skip-download flag set).")
        return

    karpathy_json = os.path.join(data_dir, "dataset_coco.json")
    images_exist = (
        os.path.isdir(os.path.join(data_dir, "images", "train2014"))
        or os.path.isdir(os.path.join(data_dir, "train2014"))
    )
    karpathy_exists = os.path.exists(karpathy_json)

    if images_exist and karpathy_exists:
        logger.info("  Dataset already exists, skipping download.")
        with open(karpathy_json, "r") as f:
            n_images = len(json.load(f).get("images", []))
        logger.info(f"  Karpathy split: {n_images:,} images")

        # Quick count
        for split in ["train2014", "val2014"]:
            for prefix in ["images", ""]:
                d = os.path.join(data_dir, prefix, split)
                if os.path.isdir(d):
                    count = len([f for f in os.listdir(d) if f.endswith(".jpg")])
                    logger.info(f"  {split}: {count:,} images")
                    break
        return

    logger.info("  Dataset not complete — downloading...")
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "download_data.py"), "--data-dir", data_dir]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("  Dataset download failed!")
        sys.exit(1)

    logger.info("  Dataset ready.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(args):
    """Launch training with auto-resume support."""
    logger.info("=" * 60)
    logger.info("Step 3/3: Training")
    logger.info("=" * 60)

    # Check for existing checkpoints
    latest_ckpt = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
    has_checkpoint = os.path.exists(latest_ckpt)

    if has_checkpoint:
        if args.resume:
            logger.info("  Resuming from checkpoint...")
        else:
            logger.info("  Checkpoint found. Use --resume to continue, or training starts fresh.")

    if not has_checkpoint and args.resume:
        logger.info("  --resume set but no checkpoint found. Starting fresh.")

    # Build command
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "train.py"),
        "--data-dir", args.data_dir,
        "--checkpoint-dir", args.checkpoint_dir,
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--learning-rate", str(args.learning_rate),
        "--warmup-steps", str(args.warmup_steps),
        "--save-every-n-steps", str(args.save_every_n_steps),
        "--patience", str(args.patience),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
    ]

    if args.resume and has_checkpoint:
        cmd.append("--resume")

    if args.fp16:
        cmd.append("--fp16")
    else:
        cmd.append("--no-fp16")

    if args.max_train_samples > 0:
        cmd.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_val_samples > 0:
        cmd.extend(["--max-val-samples", str(args.max_val_samples)])

    logger.info(f"  Command: {' '.join(cmd)}")
    logger.info("")

    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        logger.error(f"  Training failed after {elapsed/60:.1f} minutes!")
        sys.exit(1)

    logger.info(f"\n  Training completed in {elapsed/60:.1f} minutes.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. INFERENCE (optional)
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(args):
    """Run inference after training."""
    logger.info("\n" + "=" * 60)
    logger.info("Running inference...")
    logger.info("=" * 60)

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "inference.py"),
        "--data-dir", args.data_dir,
        "--checkpoint-dir", args.checkpoint_dir,
        "--num-samples", str(args.num_samples),
        "--save-fig", os.path.join(args.checkpoint_dir, "..", "inference_results.png"),
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.warning("  Inference failed, but training was successful.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="BLIP-2 (PVT v2 + QFormer LoRA) — Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/run_all.py                             # full pipeline
  python pipeline/run_all.py --resume                    # resume training
  python pipeline/run_all.py --resume --batch-size 4 --epochs 10
  python pipeline/run_all.py --skip-download             # skip data download
  python pipeline/run_all.py --inference-only             # only inference
        """,
    )

    # Pipeline control
    p.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint")
    p.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    p.add_argument("--inference-only", action="store_true", help="Only run inference (skip training)")

    # Paths
    p.add_argument("--data-dir", type=str, default="data/coco")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    # Training
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--save-every-n-steps", type=int, default=500)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-train-samples", type=int, default=-1)
    p.add_argument("--max-val-samples", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)

    # Inference
    p.add_argument("--num-samples", type=int, default=8)

    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "═" * 60)
    print("  BLIP-2 (PVT v2 + QFormer LoRA) — Pipeline Runner")
    print("═" * 60 + "\n")

    try:
        # 1. Environment
        check_environment()

        # 2. Data
        prepare_data(args.data_dir, skip_download=args.skip_download)

        # 3. Train or inference
        if args.inference_only:
            run_inference(args)
        else:
            train_model(args)
            run_inference(args)

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Checkpoints are safe.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  Pipeline complete!")
    print(f"  Checkpoints: {os.path.abspath(args.checkpoint_dir)}/")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
