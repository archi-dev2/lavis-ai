#!/usr/bin/env python3
"""
inference.py — Inference pipeline for BLIP-2 (PVT v2 + QFormer LoRA + FlanT5).

Loads best/latest checkpoint, generates captions, and displays results.

Usage:
    python pipeline/inference.py --checkpoint-dir checkpoints --data-dir data/coco
    python pipeline/inference.py --image path/to/image.jpg          # single image
    python pipeline/inference.py --num-samples 16 --save-fig out.png
"""

import os
import sys
import json
import random
import argparse
import logging

import torch
from torch.cuda.amp import autocast
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

# ── Ensure repo is importable ─────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.environ.get("LAVIS_REPO_DIR", os.path.join(SCRIPT_DIR, "..", "lavis-ai"))
if os.path.isdir(os.path.join(SCRIPT_DIR, "..", "lavis")):
    REPO_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, os.path.abspath(REPO_DIR))

from lavis.models.blip2_models.blip2_t5 import Blip2T5  # noqa: E402
from lavis.models import *  # noqa: E402,F401,F403
from lavis.processors import *  # noqa: E402,F401,F403

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("inference")

# ImageNet normalization constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def unnormalize(img_tensor):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def load_model_from_checkpoint(args, device):
    """Initialize model and load checkpoint weights."""
    logger.info("Initializing model...")
    model = Blip2T5(
        vit_model=args.vit_model,
        img_size=args.image_size,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=args.num_query_token,
        t5_model=args.t5_model,
        prompt=args.prompt,
        max_txt_len=args.max_txt_len,
    )

    # Load checkpoint
    best_path = os.path.join(args.checkpoint_dir, "checkpoint_best.pt")
    latest_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
    ckpt_path = best_path if os.path.exists(best_path) else latest_path

    if os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        logger.info(
            f"  Epoch: {ckpt['epoch']}, Step: {ckpt['global_step']}, "
            f"Val Loss: {ckpt['best_val_loss']:.4f}"
        )
    else:
        logger.warning(f"No checkpoint found in {args.checkpoint_dir} — using random weights.")

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_captions(model, images, device, prompt="a photo of", num_beams=5, max_length=30):
    """Generate captions for a batch of image tensors."""
    images = images.to(device)
    samples = {"image": images, "prompt": prompt}

    captions = model.generate(
        samples,
        use_nucleus_sampling=False,
        num_beams=num_beams,
        max_length=max_length,
        min_length=1,
    )
    return captions


def infer_single_image(model, image_path, transform, device, prompt, num_beams):
    """Generate caption for a single image file."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    captions = generate_captions(model, image_tensor, device, prompt=prompt, num_beams=num_beams)
    return captions[0]


def infer_validation_set(model, args, device):
    """Run inference on random samples from validation set."""
    # Load Karpathy annotations
    karpathy_json = os.path.join(args.data_dir, "dataset_coco.json")
    if not os.path.exists(karpathy_json):
        logger.error(f"Karpathy JSON not found: {karpathy_json}")
        return []

    with open(karpathy_json, "r") as f:
        data = json.load(f)

    # Collect val images
    val_items = []
    for img_info in data["images"]:
        if img_info["split"] != "val":
            continue
        filename = img_info["filename"]
        # Try to find image
        for prefix in ["images", "images/val2014", "val2014", ""]:
            path = os.path.join(args.data_dir, prefix, filename) if prefix else os.path.join(args.data_dir, filename)
            if os.path.exists(path):
                captions = [s["raw"].strip() for s in img_info["sentences"] if s["raw"].strip()]
                if captions:
                    val_items.append({"image": path, "captions": captions})
                break

    if not val_items:
        logger.error("No valid validation images found.")
        return []

    # Sample
    n = min(args.num_samples, len(val_items))
    samples = random.sample(val_items, n)
    transform = build_transform(args.image_size)

    results = []
    for item in tqdm(samples, desc="Generating captions"):
        image = Image.open(item["image"]).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        pred = generate_captions(model, image_tensor, device, prompt=args.prompt, num_beams=args.num_beams)

        results.append({
            "image_path": item["image"],
            "ground_truth": item["captions"][0],
            "predicted": pred[0],
        })

    return results


def display_results(results):
    """Print inference results to console."""
    print(f"\n{'='*70}")
    print("INFERENCE RESULTS")
    print(f"{'='*70}")

    for i, r in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"  Image:        {os.path.basename(r['image_path'])}")
        print(f"  Ground Truth: {r['ground_truth']}")
        print(f"  Predicted:    {r['predicted']}")

    print(f"\n{'='*70}")


def visualize_results(results, save_path=None):
    """Display results as an image grid with matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg" if save_path else "TkAgg")
        import matplotlib.pyplot as plt
        import textwrap
    except ImportError:
        logger.warning("matplotlib not available — skipping visualization.")
        return

    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5.5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        img = Image.open(r["image_path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        gt = textwrap.fill(r["ground_truth"], width=40)
        pred = textwrap.fill(r["predicted"], width=40)
        ax.set_title(f"GT: {gt}\n\nPred: {pred}", fontsize=8, ha="center")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Visualization saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="BLIP-2 Inference")

    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--data-dir", type=str, default="data/coco")
    p.add_argument("--image", type=str, default=None, help="Single image path for inference")

    p.add_argument("--vit-model", type=str, default="pvt_v2_b2")
    p.add_argument("--t5-model", type=str, default="google/flan-t5-xl")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-query-token", type=int, default=16)
    p.add_argument("--max-txt-len", type=int, default=32)
    p.add_argument("--prompt", type=str, default="a photo of")

    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--save-fig", type=str, default=None, help="Path to save visualization")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = load_model_from_checkpoint(args, device)

    if args.image:
        # Single image inference
        if not os.path.exists(args.image):
            logger.error(f"Image not found: {args.image}")
            sys.exit(1)

        transform = build_transform(args.image_size)
        caption = infer_single_image(model, args.image, transform, device, args.prompt, args.num_beams)
        print(f"\nImage:   {args.image}")
        print(f"Caption: {caption}")
    else:
        # Validation set inference
        results = infer_validation_set(model, args, device)

        if results:
            display_results(results)
            if args.save_fig:
                visualize_results(results, save_path=args.save_fig)
            else:
                visualize_results(results, save_path="inference_results.png")


if __name__ == "__main__":
    main()
