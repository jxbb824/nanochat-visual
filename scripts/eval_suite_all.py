#!/usr/bin/env python
"""
Evaluate multiple FineVision checkpoints on a suite of benchmarks
without LLM judges. Benchmarks: MMStar, MME, DocVQA, OCRBench, VQAv2,
ChartQA. Prints a summary table at the end.
"""

import argparse
import os
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional

import torch

from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir
from nanochat.tokenizer import get_tokenizer
from scripts.finevision_cli import load_vlm_checkpoint
from scripts.mmstar_eval import build_visual_tokens as mmstar_build_visual_tokens
from scripts.mmstar_eval import generate_answer as mmstar_generate_answer
from scripts.mme_eval import build_visual_tokens as mme_build_visual_tokens
from scripts.mme_eval import generate_answer as mme_generate_answer
from scripts.docvqa_eval import build_visual_tokens as docvqa_build_visual_tokens
from scripts.docvqa_eval import generate_answer as docvqa_generate_answer
from scripts.ocrbench_eval import build_visual_tokens as ocr_build_visual_tokens
from scripts.ocrbench_eval import generate_answer as ocr_generate_answer
from scripts.vqav2_eval import build_visual_tokens as vqa_build_visual_tokens
from scripts.vqav2_eval import generate_answer as vqa_generate_answer
from scripts.chartqa_eval import build_visual_tokens as chart_build_visual_tokens
from scripts.chartqa_eval import generate_answer as chart_generate_answer
from tasks.mmstar import MMStar
from tasks.mme import MME
from tasks.docvqa import DocVQA
from tasks.ocrbench import OCRBench
from tasks.vqav2 import VQAv2Small
from tasks.chartqa import ChartQA


TaskConfig = Dict[str, object]


def evaluate_dataset(
    name: str,
    dataset,
    image_key: str,
    build_visual_tokens: Callable,
    generate_answer: Callable,
    model,
    vision,
    tokenizer,
    device,
    autocast_ctx,
    max_examples: int,
) -> float:
    n_total = len(dataset)
    n = n_total if max_examples <= 0 else min(n_total, max_examples)
    if n == 0:
        return 0.0

    correct = 0
    with torch.no_grad(), autocast_ctx:
        for idx in range(n):
            conversation = dataset[idx]
            image = conversation[image_key]
            question = conversation["messages"][0]["content"]

            visual_tokens = build_visual_tokens(vision, image, device)
            pred = generate_answer(
                model,
                tokenizer,
                question,
                visual_tokens,
                device,
            )
            ok = dataset.evaluate(conversation, pred)
            correct += int(ok)
    return 100.0 * correct / float(n)


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple VLM tags across common benchmarks")
    parser.add_argument(
        "--vlm-tags",
        type=str,
        default="d10_finevision_clip,"
        "d10_finevision_clippatch_ViT-B-32_pool1,"
        "d10_finevision_sigclippatch_ViT-B-16-SigLIP_shuf2,"
        "d10_finevision_sigclippatch_ViT-B-16-SigLIP_shuf1,"
        "d20_finevision_sigclippatch_ViT-B-16-SigLIP_shuf1",
        help="Comma-separated list of VLM checkpoint tags under vlm_checkpoints/",
    )
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device type")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--max-examples", type=int, default=-1, help="Limit examples per benchmark (-1 = full)")
    parser.add_argument("--step", type=int, default=None, help="Optional checkpoint step (only used when a single vlm-tag is given)")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # single-process eval only
    if ddp and ddp_rank != 0:
        compute_cleanup()
        return

    vlm_tags = [v.strip() for v in args.vlm_tags.split(",") if v.strip()]
    base_dir = get_base_dir()
    single_step = args.step if len(vlm_tags) == 1 else None

    tasks: List[TaskConfig] = [
        {
            "name": "MMStar",
            "dataset": lambda: MMStar(),
            "image_key": "mmstar_image",
            "build": mmstar_build_visual_tokens,
            "generate": mmstar_generate_answer,
        },
        {
            "name": "MME",
            "dataset": lambda: MME(),
            "image_key": "mme_image",
            "build": mme_build_visual_tokens,
            "generate": mme_generate_answer,
        },
        {
            "name": "DocVQA",
            "dataset": lambda: DocVQA(split="test"),
            "image_key": "docvqa_image",
            "build": docvqa_build_visual_tokens,
            "generate": docvqa_generate_answer,
        },
        {
            "name": "OCRBench",
            "dataset": lambda: OCRBench(split="test"),
            "image_key": "ocrbench_image",
            "build": ocr_build_visual_tokens,
            "generate": ocr_generate_answer,
        },
        {
            "name": "VQAv2",
            "dataset": lambda: VQAv2Small(split="validation"),
            "image_key": "vqav2_image",
            "build": vqa_build_visual_tokens,
            "generate": vqa_generate_answer,
        },
        {
            "name": "ChartQA",
            "dataset": lambda: ChartQA(split="test"),
            "image_key": "chartqa_image",
            "build": chart_build_visual_tokens,
            "generate": chart_generate_answer,
        },
    ]

    results: List[Dict[str, Optional[float]]] = []

    for vlm_tag in vlm_tags:
        print(f"\n=== Evaluating {vlm_tag} ===")
        # Switch tokenizer per family (d10/d16/d20...) based on tag prefix.
        tok_prefix = vlm_tag.split("_", 1)[0]
        candidate_tok = os.path.join(base_dir, "tokenizer", tok_prefix)
        if os.path.isdir(candidate_tok):
            os.environ["NANOCHAT_TOKENIZER_DIR"] = candidate_tok
        else:
            # fallback to existing env or default, will raise if missing
            os.environ.pop("NANOCHAT_TOKENIZER_DIR", None)
            candidate_tok = None
        tokenizer = get_tokenizer()

        model, vision = load_vlm_checkpoint(device, vlm_tag=vlm_tag, step=single_step)
        model.eval()
        vision.eval()

        row: Dict[str, Optional[float]] = {"model": vlm_tag}
        for task in tasks:
            name = task["name"]
            try:
                ds = task["dataset"]()
                acc = evaluate_dataset(
                    name=name,
                    dataset=ds,
                    image_key=task["image_key"],
                    build_visual_tokens=task["build"],
                    generate_answer=task["generate"],
                    model=model,
                    vision=vision,
                    tokenizer=tokenizer,
                    device=device,
                    autocast_ctx=autocast_ctx,
                    max_examples=args.max_examples,
                )
                print(f"{name}: {acc:.2f}%")
                row[name] = acc
            except Exception as e:
                print(f"{name}: FAILED ({e})")
                row[name] = None
        results.append(row)

    # summary table
    headers = ["model"] + [t["name"] for t in tasks]
    col_widths: Dict[str, int] = {}
    for h in headers:
        col_widths[h] = len(h)
    for row in results:
        for h in headers:
            val = row.get(h)
            txt = row["model"] if h == "model" else ("-" if val is None else f"{val:.2f}")
            col_widths[h] = max(col_widths[h], len(txt))

    def fmt(row, h):
        if h == "model":
            return row["model"].ljust(col_widths[h])
        val = row.get(h)
        txt = "-" if val is None else f"{val:.2f}"
        return txt.rjust(col_widths[h])

    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    sep_line = "-+-".join("-" * col_widths[h] for h in headers)
    print("\nSummary:")
    print(header_line)
    print(sep_line)
    for row in results:
        print(" | ".join(fmt(row, h) for h in headers))

    compute_cleanup()


if __name__ == "__main__":
    main()
