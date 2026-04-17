#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh — Full server setup and training pipeline
#
# BLIP-2 (PVT v2 + QFormer LoRA + FlanT5-XL)
#
# Usage:
#   bash pipeline/run_pipeline.sh                    # full pipeline
#   bash pipeline/run_pipeline.sh --resume           # resume training
#   bash pipeline/run_pipeline.sh --skip-download    # skip data download
#   bash pipeline/run_pipeline.sh --inference-only   # only run inference
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"
DATA_DIR="${PROJECT_ROOT}/data/coco"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
PYTHON="${VENV_DIR}/bin/python"

# Parse flags
RESUME=""
SKIP_DOWNLOAD=""
INFERENCE_ONLY=""
BATCH_SIZE=4
EPOCHS=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)       RESUME="--resume"; shift ;;
        --skip-download) SKIP_DOWNLOAD="1"; shift ;;
        --inference-only) INFERENCE_ONLY="1"; shift ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        *)              echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "══════════════════════════════════════════════════════════════"
echo "  BLIP-2 (PVT v2 + QFormer LoRA) Training Pipeline"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: System dependencies ──────────────────────────────────────────────
echo "[1/6] Checking system dependencies..."
for cmd in python3 pip3 git wget unzip; do
    if command -v "$cmd" &>/dev/null; then
        echo "  ✓ $cmd"
    else
        echo "  ✗ $cmd not found"
        # Try to install common missing tools (non-sudo fallback)
        if command -v apt-get &>/dev/null; then
            echo "    Attempting install..."
            sudo apt-get update -qq && sudo apt-get install -y -qq "$cmd" 2>/dev/null || \
                echo "    WARNING: Could not install $cmd (no sudo?). Continuing..."
        fi
    fi
done

# ── Step 2: Virtual environment ──────────────────────────────────────────────
echo ""
echo "[2/6] Setting up Python virtual environment..."
if [ -f "${PYTHON}" ]; then
    echo "  Virtual environment already exists: ${VENV_DIR}"
else
    echo "  Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
    echo "  Created: ${VENV_DIR}"
fi
# Activate venv
source "${VENV_DIR}/bin/activate"
echo "  Python: $(python --version)"
echo "  pip:    $(pip --version | cut -d' ' -f1-2)"

# ── Step 3: Install Python dependencies ──────────────────────────────────────
echo ""
echo "[3/6] Installing Python packages..."
pip install -q --upgrade pip
python "${SCRIPT_DIR}/setup_env.py"

# ── Step 4: Download data ────────────────────────────────────────────────────
if [ -n "$INFERENCE_ONLY" ]; then
    echo ""
    echo "[4/6] Skipping data download (inference-only mode)"
elif [ -n "$SKIP_DOWNLOAD" ]; then
    echo ""
    echo "[4/6] Skipping data download (--skip-download)"
else
    echo ""
    echo "[4/6] Downloading dataset..."
    python "${SCRIPT_DIR}/download_data.py" --data-dir "${DATA_DIR}"
fi

# ── Step 5: Training ─────────────────────────────────────────────────────────
if [ -n "$INFERENCE_ONLY" ]; then
    echo ""
    echo "[5/6] Skipping training (inference-only mode)"
else
    echo ""
    echo "[5/6] Starting training..."
    python "${SCRIPT_DIR}/train.py" \
        --data-dir "${DATA_DIR}" \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --batch-size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        ${RESUME}
fi

# ── Step 6: Inference ────────────────────────────────────────────────────────
echo ""
echo "[6/6] Running inference..."
python "${SCRIPT_DIR}/inference.py" \
    --data-dir "${DATA_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --num-samples 8 \
    --save-fig "${PROJECT_ROOT}/inference_results.png"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Pipeline complete."
echo "  Checkpoints: ${CHECKPOINT_DIR}/"
echo "  Results:     ${PROJECT_ROOT}/inference_results.png"
echo "══════════════════════════════════════════════════════════════"
