#!/usr/bin/env python
"""
Evaluate HuggingFace smolvlm2 checkpoints on the same vision benchmarks
(MMStar, MME, DocVQA, OCRBench, VQAv2, ChartQA). Uses plain substring/word
matching from the Task classes (no LLM judge).
"""

import argparse
from contextlib import nullcontext
from typing import Dict, List

import torch
from PIL import Image
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor

from tasks.mmstar import MMStar
from tasks.mme import MME
from tasks.docvqa import DocVQA
from tasks.ocrbench import OCRBench
from tasks.vqav2 import VQAv2Small
from tasks.chartqa import ChartQA


def _to_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return Image.fromarray(image).convert("RGB")
    if torch.is_tensor(image):
        arr = image.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return Image.fromarray(arr).convert("RGB")
    # fallback: try Pillow conversion
    try:
        return Image.fromarray(np.array(image)).convert("RGB")
    except Exception:
        return image


def build_inputs(processor, image, question: str, device):
    # Prefer chat template if available (for chat-style VLMs).
    image = _to_rgb(image)
    if hasattr(processor, "apply_chat_template"):
        messages = [{"role": "user", "content": [{"type": "text", "text": question}, {"type": "image"}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
    else:
        inputs = processor(images=image, text=question, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def generate_answer(model, processor, image, question, device, max_new_tokens=32, temperature=0.0):
    inputs = build_inputs(processor, image, question, device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return text.strip()


def evaluate_dataset(name: str, dataset, image_key: str, model, processor, device, autocast_ctx, max_examples: int) -> float:
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
            pred = generate_answer(model, processor, image, question, device)
            ok = dataset.evaluate(conversation, pred)
            correct += int(ok)
    return 100.0 * correct / float(n)


def main():
    parser = argparse.ArgumentParser(description="Evaluate smolvlm2 on vision benchmarks")
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="HuggingFace model id for AutoModelForVision2Seq/AutoProcessor",
    )
    parser.add_argument("--device", type=str, default="", help="torch device, e.g., cuda or cpu (auto if empty)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"], help="torch dtype")
    parser.add_argument("--max-examples", type=int, default=200, help="limit examples per benchmark (-1 = full)")
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (0 => greedy)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch_dtype) if device.type == "cuda" else nullcontext()

    print(f"Loading model {args.model_id} on {device} ...")
    processor = AutoProcessor.from_pretrained(
        args.model_id, torch_dtype=torch_dtype, trust_remote_code=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    tasks = [
        {"name": "MMStar", "dataset": lambda: MMStar(), "image_key": "mmstar_image"},
        {"name": "MME", "dataset": lambda: MME(), "image_key": "mme_image"},
        {"name": "DocVQA", "dataset": lambda: DocVQA(split="test"), "image_key": "docvqa_image"},
        {"name": "OCRBench", "dataset": lambda: OCRBench(split="test"), "image_key": "ocrbench_image"},
        {"name": "VQAv2", "dataset": lambda: VQAv2Small(split="validation"), "image_key": "vqav2_image"},
        {"name": "ChartQA", "dataset": lambda: ChartQA(split="test"), "image_key": "chartqa_image"},
    ]

    results: List[Dict[str, float]] = []
    for task in tasks:
        name = task["name"]
        print(f"\n{name}...")
        try:
            ds = task["dataset"]()
            acc = evaluate_dataset(
                name,
                ds,
                task["image_key"],
                model,
                processor,
                device,
                autocast_ctx,
                max_examples=args.max_examples,
            )
            print(f"{name}: {acc:.2f}%")
            results.append({"task": name, "acc": acc})
        except Exception as e:
            print(f"{name}: FAILED ({e})")
            results.append({"task": name, "acc": None})

    print("\nSummary:")
    for row in results:
        acc = "-" if row["acc"] is None else f"{row['acc']:.2f}%"
        print(f"{row['task']}: {acc}")


if __name__ == "__main__":
    main()
