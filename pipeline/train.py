#!/usr/bin/env python3
"""
train.py — Training pipeline for BLIP-2 (PVT v2 + QFormer LoRA + FlanT5).

Features:
  - Mixed precision (fp16) training
  - Gradient accumulation
  - AdamW + cosine warmup scheduler
  - Step-accurate checkpoint resume
  - Resume-safe early stopping
  - Periodic + end-of-epoch checkpointing
  - Configurable via CLI args

Usage:
    python pipeline/train.py --data-dir data/coco --epochs 10 --batch-size 4
    python pipeline/train.py --resume   # auto-resume from latest checkpoint
"""

import os
import sys
import json
import glob
import math
import time
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

# ── Ensure repo is importable ─────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.environ.get("LAVIS_REPO_DIR", os.path.join(SCRIPT_DIR, "..", "lavis-ai"))
if os.path.isdir(os.path.join(SCRIPT_DIR, "..", "lavis")):
    # Running from repo root
    REPO_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, os.path.abspath(REPO_DIR))

from lavis.models.blip2_models.blip2_t5 import Blip2T5  # noqa: E402
from lavis.models import *  # noqa: E402,F401,F403 — register all models
from lavis.processors import *  # noqa: E402,F401,F403

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

def resolve_image_path(filename, data_dir):
    """Find image across common COCO directory layouts."""
    candidates = [
        os.path.join(data_dir, "images", filename),
        os.path.join(data_dir, filename),
        os.path.join(data_dir, "images", "train2014", filename),
        os.path.join(data_dir, "images", "val2014", filename),
        os.path.join(data_dir, "train2014", filename),
        os.path.join(data_dir, "val2014", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def build_annotations(karpathy_json_path, data_dir, split_name):
    """
    Build flat annotation list from Karpathy JSON.
    Maps 'restval' → 'train' as is standard.

    Returns list of {"image": abs_path, "caption": str, "image_id": int}
    """
    with open(karpathy_json_path, "r") as f:
        data = json.load(f)

    target_splits = [split_name]
    if split_name == "train":
        target_splits.append("restval")

    annotations = []
    skipped = 0

    for img_info in data["images"]:
        if img_info["split"] not in target_splits:
            continue

        filename = img_info["filename"]
        img_path = resolve_image_path(filename, data_dir)

        if img_path is None:
            skipped += 1
            continue

        img_id = img_info.get("cocoid", img_info.get("imgid", 0))

        for sent in img_info["sentences"]:
            caption = sent["raw"].strip()
            if caption:
                annotations.append({
                    "image": img_path,
                    "caption": caption,
                    "image_id": img_id,
                })

    if skipped > 0:
        logger.warning(f"[{split_name}] Skipped {skipped} images (not found on disk)")

    return annotations


class COCOCaptionDataset(Dataset):
    """COCO Caption dataset for BLIP-2 training."""

    def __init__(self, annotations, image_size=224, is_train=True):
        self.annotations = annotations

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size, scale=(0.5, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                ann = self.annotations[idx]
                image = Image.open(ann["image"]).convert("RGB")
                image = self.transform(image)
                caption = ann["caption"]
                return {
                    "image": image,
                    "text_input": caption,
                    "text_output": caption,
                    "image_id": ann["image_id"],
                }
            except Exception:
                if attempt == max_retries - 1:
                    raise
                idx = random.randint(0, len(self.annotations) - 1)


def collate_fn(batch):
    """Stack images, keep text as lists."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "text_input": [b["text_input"] for b in batch],
        "text_output": [b["text_output"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model, optimizer, scheduler, scaler,
    epoch, global_step, batch_idx_in_epoch,
    best_val_loss, early_stopping_counter,
    checkpoint_dir, tag="epoch",
):
    """
    Save full training checkpoint for exact resume.

    Saves: model, optimizer, scheduler, scaler, epoch, global_step,
           batch_idx_in_epoch, best_val_loss, early_stopping_counter.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "batch_idx_in_epoch": batch_idx_in_epoch,
        "best_val_loss": best_val_loss,
        "early_stopping_counter": early_stopping_counter,
    }

    path = os.path.join(checkpoint_dir, f"checkpoint_{tag}_e{epoch}_s{global_step}.pt")
    torch.save(checkpoint, path)

    # Always maintain a 'latest' symlink/copy
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)

    if tag == "best":
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        torch.save(checkpoint, best_path)

    size_mb = os.path.getsize(path) / 1e6
    logger.info(
        f"Checkpoint saved: {os.path.basename(path)} ({size_mb:.1f} MB) | "
        f"epoch={epoch}, step={global_step}, best_val={best_val_loss:.4f}, "
        f"es={early_stopping_counter}"
    )

    # Cleanup old periodic checkpoints (keep last 3)
    if tag == "periodic":
        periodic_files = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint_periodic_*.pt")),
            key=os.path.getmtime,
        )
        while len(periodic_files) > 3:
            os.remove(periodic_files.pop(0))

    return path


def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_dir, device, tag=None):
    """
    Load checkpoint and restore ALL training state.
    Returns dict with state or None if no checkpoint found.
    """
    if tag:
        path = os.path.join(checkpoint_dir, f"checkpoint_{tag}.pt")
    else:
        path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")

    if not os.path.exists(path):
        logger.info("No checkpoint found. Starting from scratch.")
        return None

    logger.info(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])

    state = {
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "batch_idx_in_epoch": checkpoint.get("batch_idx_in_epoch", 0),
        "best_val_loss": checkpoint["best_val_loss"],
        "early_stopping_counter": checkpoint["early_stopping_counter"],
    }

    logger.info(
        f"Resumed: epoch={state['epoch']}, step={state['global_step']}, "
        f"batch_in_epoch={state['batch_idx_in_epoch']}, "
        f"best_val={state['best_val_loss']:.4f}, es={state['early_stopping_counter']}"
    )
    return state


def load_latest_checkpoint(model, optimizer, scheduler, scaler, checkpoint_dir, device):
    """Convenience: auto-detect and load the latest checkpoint."""
    return load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_dir, device, tag=None)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model, train_loader, optimizer, scheduler, scaler,
    epoch, global_step, args, device,
    resume_step_in_epoch=0,
    checkpoint_dir="checkpoints",
    best_val_loss=float("inf"),
    early_stopping_counter=0,
):
    """
    Train one epoch with step-accurate resume.

    Args:
        resume_step_in_epoch: batches already done this epoch (for mid-epoch resume).

    Returns:
        (avg_loss, global_step)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch}",
        dynamic_ncols=True,
    )

    for batch_idx, samples in pbar:
        # Skip already-processed batches on resume
        if batch_idx < resume_step_in_epoch:
            if batch_idx % 200 == 0:
                pbar.set_postfix_str(f"skipping to step {resume_step_in_epoch}")
            continue

        samples["image"] = samples["image"].to(device, non_blocking=True)

        # Forward (mixed precision)
        with autocast(enabled=args.fp16):
            output = model(samples)
            loss = output["loss"] / args.gradient_accumulation_steps

        # Backward
        scaler.scale(loss).backward()

        # Optimizer step every N accumulation steps
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            # Periodic checkpoint
            if args.save_every_n_steps > 0 and global_step % args.save_every_n_steps == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch=epoch, global_step=global_step,
                    batch_idx_in_epoch=batch_idx + 1,
                    best_val_loss=best_val_loss,
                    early_stopping_counter=early_stopping_counter,
                    checkpoint_dir=checkpoint_dir, tag="periodic",
                )

        batch_loss = loss.item() * args.gradient_accumulation_steps
        total_loss += batch_loss
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{batch_loss:.4f}",
            "avg": f"{total_loss / num_batches:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            "step": global_step,
        })

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def validate(model, val_loader, args, device):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Validation", dynamic_ncols=True)
    for samples in pbar:
        samples["image"] = samples["image"].to(device, non_blocking=True)

        with autocast(enabled=args.fp16):
            output = model(samples)
            loss = output["loss"]

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"val_loss": f"{total_loss / num_batches:.4f}"})

    return total_loss / max(num_batches, 1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train BLIP-2 (PVT v2 + QFormer LoRA)")

    # Paths
    p.add_argument("--data-dir", type=str, default="data/coco", help="COCO data root")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    p.add_argument("--repo-dir", type=str, default=REPO_DIR, help="LAVIS repo directory")

    # Model
    p.add_argument("--vit-model", type=str, default="pvt_v2_b2")
    p.add_argument("--t5-model", type=str, default="google/flan-t5-xl")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-query-token", type=int, default=16)
    p.add_argument("--max-txt-len", type=int, default=32)
    p.add_argument("--prompt", type=str, default="a photo of")
    p.add_argument("--freeze-vit", action="store_true", default=True)

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--batch-size-eval", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")

    # Checkpointing
    p.add_argument("--save-every-n-steps", type=int, default=500)
    p.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint")

    # Early stopping
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=0.0)

    # Data
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-train-samples", type=int, default=-1, help="-1 = all")
    p.add_argument("--max-val-samples", type=int, default=-1, help="-1 = all")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    karpathy_json = os.path.join(args.data_dir, "dataset_coco.json")
    if not os.path.exists(karpathy_json):
        logger.error(f"Karpathy JSON not found: {karpathy_json}")
        logger.error("Run download_data.py first, or set --data-dir.")
        sys.exit(1)

    logger.info("Building annotations...")
    train_anns = build_annotations(karpathy_json, args.data_dir, "train")
    val_anns = build_annotations(karpathy_json, args.data_dir, "val")

    if args.max_train_samples > 0:
        train_anns = train_anns[:args.max_train_samples]
    if args.max_val_samples > 0:
        val_anns = val_anns[:args.max_val_samples]

    logger.info(f"Train: {len(train_anns)} samples | Val: {len(val_anns)} samples")

    train_dataset = COCOCaptionDataset(train_anns, args.image_size, is_train=True)
    val_dataset = COCOCaptionDataset(val_anns, args.image_size, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size_eval, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Initializing BLIP-2 model...")
    logger.info(f"  Vision: {args.vit_model} | LLM: {args.t5_model} | Queries: {args.num_query_token}")

    model = Blip2T5(
        vit_model=args.vit_model,
        img_size=args.image_size,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=args.freeze_vit,
        num_query_token=args.num_query_token,
        t5_model=args.t5_model,
        prompt=args.prompt,
        max_txt_len=args.max_txt_len,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params:     {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # ── Optimizer, Scheduler, Scaler ──────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    scaler = GradScaler(enabled=args.fp16)

    logger.info(f"  Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")
    logger.info(f"  Effective batch: {args.batch_size * args.gradient_accumulation_steps}")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    resume_batch_idx = 0
    best_val_loss = float("inf")
    early_stopping_counter = 0

    if args.resume:
        state = load_latest_checkpoint(
            model, optimizer, scheduler, scaler, args.checkpoint_dir, device
        )
        if state is not None:
            start_epoch = state["epoch"]
            global_step = state["global_step"]
            resume_batch_idx = state["batch_idx_in_epoch"]
            best_val_loss = state["best_val_loss"]
            early_stopping_counter = state["early_stopping_counter"]

            if resume_batch_idx == 0 or resume_batch_idx >= len(train_loader):
                start_epoch += 1
                resume_batch_idx = 0

            logger.info(
                f"RESUMING: epoch={start_epoch}, step={global_step}, "
                f"batch={resume_batch_idx}, best_val={best_val_loss:.4f}, "
                f"es={early_stopping_counter}/{args.patience}"
            )

    # ── Training Loop ─────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Starting training")
    logger.info(f"{'='*60}")

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n{'━'*60}")
        logger.info(f"Epoch {epoch}/{args.epochs - 1}")
        logger.info(f"{'━'*60}")

        skip_batches = resume_batch_idx if epoch == start_epoch else 0

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            epoch=epoch, global_step=global_step, args=args, device=device,
            resume_step_in_epoch=skip_batches,
            checkpoint_dir=args.checkpoint_dir,
            best_val_loss=best_val_loss,
            early_stopping_counter=early_stopping_counter,
        )
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, args, device)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, best={best_val_loss:.4f}")

        # Early stopping (resume-safe)
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            early_stopping_counter = 0
            logger.info("New best validation loss! Saving best model.")
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch=epoch, global_step=global_step, batch_idx_in_epoch=0,
                best_val_loss=best_val_loss, early_stopping_counter=early_stopping_counter,
                checkpoint_dir=args.checkpoint_dir, tag="best",
            )
        else:
            early_stopping_counter += 1
            logger.info(f"No improvement. ES counter: {early_stopping_counter}/{args.patience}")

        # End-of-epoch checkpoint
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch=epoch, global_step=global_step, batch_idx_in_epoch=0,
            best_val_loss=best_val_loss, early_stopping_counter=early_stopping_counter,
            checkpoint_dir=args.checkpoint_dir, tag="epoch",
        )

        if early_stopping_counter >= args.patience:
            logger.info(f"\nEARLY STOPPING at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'═'*60}")
    logger.info("Training Complete")
    logger.info(f"  Final epoch:   {epoch}")
    logger.info(f"  Global steps:  {global_step}")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")
    logger.info(f"{'═'*60}")

    # Save training history
    history = {"train_losses": train_losses, "val_losses": val_losses, "best_val_loss": best_val_loss}
    history_path = os.path.join(args.checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved: {history_path}")


if __name__ == "__main__":
    main()
