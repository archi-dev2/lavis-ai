#!/usr/bin/env python3
"""
setup_env.py — Environment setup for BLIP-2 (PVT v2 + QFormer LoRA) training.

Handles:
  - System dependency check
  - Python package installation
  - Repository cloning
  - GPU verification
  - Environment summary
"""

import os
import sys
import subprocess
import shutil
import platform


# ── Configuration ─────────────────────────────────────────────────────────────
REPO_URL = "https://github.com/archi-dev2/lavis-ai.git"
REPO_BRANCH = "pre-training-stable"
REPO_DIR = os.environ.get("LAVIS_REPO_DIR", os.path.join(os.getcwd(), "lavis-ai"))

REQUIRED_PACKAGES = [
    "torch>=1.10.0",
    "torchvision",
    "torchaudio",
    "transformers==4.33.2",
    "timm==0.4.12",
    "accelerate",
    "datasets",
    "pillow",
    "tqdm",
    "peft",
    "sentencepiece",
    "protobuf",
    "omegaconf",
    "iopath",
    "pycocoevalcap",
    "pycocotools",
    "fairscale==0.4.4",
    "einops>=0.4.1",
]


def check_python():
    """Verify Python version >= 3.8."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"ERROR: Python 3.8+ required, got {major}.{minor}")
        sys.exit(1)
    print(f"  Python:   {platform.python_version()}")


def check_gpu():
    """Check GPU availability and print details."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  GPU:      {gpu_name} ({vram_gb:.1f} GB)")
            print(f"  CUDA:     {torch.version.cuda}")
            return True
        else:
            print("  GPU:      Not available (will use CPU — training will be slow)")
            return False
    except ImportError:
        print("  GPU:      torch not yet installed")
        return False


def install_packages():
    """Install required Python packages."""
    print("\n[2/4] Installing Python packages...")
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + REQUIRED_PACKAGES
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: Some packages may have failed:\n{result.stderr[:500]}")
    else:
        print("  All packages installed successfully.")


def clone_repo():
    """Clone the LAVIS repository if not present."""
    print(f"\n[3/4] Setting up repository...")
    if os.path.exists(REPO_DIR) and os.path.isdir(os.path.join(REPO_DIR, "lavis")):
        print(f"  Repository already exists at {REPO_DIR}")
    else:
        print(f"  Cloning {REPO_URL} (branch: {REPO_BRANCH})...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, REPO_DIR],
            check=True,
        )
        print(f"  Cloned to {REPO_DIR}")

    # Install package in development mode
    print("  Installing lavis package...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", REPO_DIR],
        check=True,
    )

    # Verify import
    sys.path.insert(0, REPO_DIR)
    try:
        import lavis  # noqa: F401
        print(f"  lavis package imported successfully from {REPO_DIR}")
    except ImportError as e:
        print(f"  WARNING: Could not import lavis: {e}")


def print_summary():
    """Print environment summary."""
    print(f"\n[4/4] Environment Summary")
    print("  " + "─" * 50)
    check_python()
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Repo:     {REPO_DIR}")
    check_gpu()

    try:
        import torch
        print(f"  PyTorch:  {torch.__version__}")
    except ImportError:
        pass
    try:
        import transformers
        print(f"  Transformers: {transformers.__version__}")
    except ImportError:
        pass
    try:
        import timm
        print(f"  timm:     {timm.__version__}")
    except ImportError:
        pass

    print("  " + "─" * 50)
    print("  Environment ready.\n")


def main():
    print("=" * 60)
    print("BLIP-2 (PVT v2 + QFormer LoRA) — Environment Setup")
    print("=" * 60)

    print("\n[1/4] Checking Python version...")
    check_python()

    install_packages()
    clone_repo()
    print_summary()


if __name__ == "__main__":
    main()
