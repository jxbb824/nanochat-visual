#!/usr/bin/env python
"""
Compute FineVision test-set perplexity for one or more VLM checkpoints.
Uses the same batching/tokenization as finevision_sft but only runs eval.
"""

import argparse
import os
import math
from contextlib import nullcontext
from typing import List

import torch
from PIL import Image

from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir
from nanochat.tokenizer import get_tokenizer
from scripts.finevision_cli import load_vlm_checkpoint
from tasks.finevision import FineVision


def collate_finevision_batch(batch, pad_token_id, device):
    # batch: list of (ids, mask, images)
    nrows = len(batch)
    ncols = max(len(ids) for ids, _, _ in batch) - 1
    inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
    targets = torch.full((nrows, ncols), -1, dtype=torch.long)
    images_batch = []

    for i, (ids, mask, images) in enumerate(batch):
        n = len(ids)
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        inputs[i, : n - 1] = ids_tensor[:-1]
        row_targets = ids_tensor[1:]
        mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
        row_targets[mask_tensor == 0] = -1
        targets[i, : n - 1] = row_targets
        images_batch.append(images)

    return inputs.to(device), targets.to(device), images_batch


def build_image_batch(images_batch: List[List[Image.Image]], vision, device):
    tensors = []
    for images in images_batch:
        if images is None or len(images) == 0:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        else:
            img = images[0]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
        img = img.convert("RGB")
        t = vision.preprocess(img)
        tensors.append(t)
    return torch.stack(tensors, dim=0).to(device)


def evaluate_ppl(model, vision, tokenizer, device, autocast_ctx, max_examples, batch_size):
    ds = FineVision(split="test")
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    total_loss = 0.0
    num_batches = 0
    batch = []

    with torch.no_grad(), autocast_ctx:
        limit = len(ds) if max_examples <= 0 else min(len(ds), max_examples)
        for i in range(limit):
            doc = ds[i]
            try:
                ids, mask = tokenizer.render_conversation(doc)
            except Exception:
                continue
            if len(ids) < 2:
                continue
            images = doc.get("images", None)
            batch.append((ids, mask, images))
            if len(batch) == batch_size:
                inputs, targets, images_batch = collate_finevision_batch(batch, pad_token_id, device)
                images_tensor = build_image_batch(images_batch, vision, device)
                loss = model(inputs, targets, visual_embs=vision(images_tensor))
                total_loss += loss.item()
                num_batches += 1
                batch = []

        if batch:
            inputs, targets, images_batch = collate_finevision_batch(batch, pad_token_id, device)
            images_tensor = build_image_batch(images_batch, vision, device)
            loss = model(inputs, targets, visual_embs=vision(images_tensor))
            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        return None
    avg_loss = total_loss / num_batches
    return avg_loss, math.exp(avg_loss)


def main():
    parser = argparse.ArgumentParser(description="Compute FineVision perplexity for VLM checkpoints")
    parser.add_argument(
        "--vlm-tags",
        type=str,
        default="d10_finevision_clip,"
        "d10_finevision_clippatch_ViT-B-32_pool1,"
        "d10_finevision_sigclippatch_ViT-B-16-SigLIP_shuf2,"
        "d10_finevision_sigclippatch_ViT-B-16-SigLIP_shuf1,"
        "d20_finevision_sigclippatch_ViT-B-16-SigLIP_shuf1",
        help="Comma-separated list of VLM checkpoint tags",
    )
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device type")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--max-examples", type=int, default=1000, help="Limit test examples (-1 = full)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for loss computation")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    if ddp and ddp_rank != 0:
        compute_cleanup()
        return

    base_dir = get_base_dir()
    vlm_tags = [v.strip() for v in args.vlm_tags.split(",") if v.strip()]

    results = []
    for vlm_tag in vlm_tags:
        print(f"\n=== {vlm_tag} ===")
        tok_prefix = vlm_tag.split("_", 1)[0]
        cand_tok = os.path.join(base_dir, "tokenizer", tok_prefix)
        if os.path.isdir(cand_tok):
            os.environ["NANOCHAT_TOKENIZER_DIR"] = cand_tok
        else:
            os.environ.pop("NANOCHAT_TOKENIZER_DIR", None)
        tokenizer = get_tokenizer()

        model, vision = load_vlm_checkpoint(device, vlm_tag=vlm_tag)
        model.eval()
        vision.eval()

        res = evaluate_ppl(
            model,
            vision,
            tokenizer,
            device,
            autocast_ctx,
            max_examples=args.max_examples,
            batch_size=args.batch_size,
        )
        if res is None:
            print("No valid batches; skipping.")
            results.append((vlm_tag, None, None))
        else:
            loss, ppl = res
            print(f"avg loss: {loss:.4f} | ppl: {ppl:.2f}")
            results.append((vlm_tag, loss, ppl))

    print("\nSummary (FineVision test):")
    print("model\tloss\tppl")
    for name, loss, ppl in results:
        if loss is None:
            print(f"{name}\t-\t-")
        else:
            print(f"{name}\t{loss:.4f}\t{ppl:.2f}")

    compute_cleanup()


if __name__ == "__main__":
    main()
