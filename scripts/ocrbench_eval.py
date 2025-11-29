"""
Evaluate the vision-language model on the OCRBench benchmark:
https://huggingface.co/datasets/echo840/OCRBench

We rely on tasks/ocrbench.OCRBench to load the default subset via `datasets`,
and evaluate accuracy by checking whether any ground-truth answer string
appears in the model's generated answer.
"""

import argparse

import torch
from contextlib import nullcontext

from nanochat.common import autodetect_device_type, compute_init, compute_cleanup
from nanochat.tokenizer import get_tokenizer
from tasks.ocrbench import OCRBench
from scripts.finevision_cli import load_vlm_checkpoint


def build_visual_tokens(vision, image, device):
    img_tensor = vision.preprocess(image.convert("RGB"))
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        visual_tokens = vision(img_tensor)
    return visual_tokens


def generate_answer(model, tokenizer, question, visual_tokens, device, max_tokens=32, temperature=0.0, top_k=None):
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    tokens = [bos, user_start]
    tokens.extend(tokenizer.encode(question))
    tokens.append(user_end)
    tokens.append(assistant_start)

    generated = []
    for _ in range(max_tokens):
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model(ids, visual_embs=visual_tokens)
        logits = logits[:, -1, :]
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            vals, idx = torch.topk(logits, k, dim=-1)
            vals = vals / max(temperature, 1e-6)
            probs = torch.softmax(vals, dim=-1)
            choice_rel = torch.multinomial(probs, num_samples=1)
            next_id = idx.gather(1, choice_rel).item()
        else:
            if temperature > 0:
                logits = logits / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_id = torch.argmax(logits, dim=-1).item()
        tokens.append(next_id)
        if next_id == assistant_end:
            break
        generated.append(next_id)
    answer = tokenizer.decode(generated)
    return answer


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM on OCRBench benchmark")
    parser.add_argument(
        "--vlm-tag",
        type=str,
        default="d20_finevision",
        help="VLM checkpoint tag under vlm_checkpoints/",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=-1,
        help="Max number of examples to evaluate (-1 = all)",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="",
        choices=["cuda", "cpu", "mps"],
        help="Device type",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
    )
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # simple single-process eval: only rank 0 does work
    if ddp and ddp_rank != 0:
        compute_cleanup()
        return

    tokenizer = get_tokenizer()
    model, vision = load_vlm_checkpoint(device, vlm_tag=args.vlm_tag)

    ds = OCRBench(split="test")
    n = len(ds)
    if args.max_examples > 0:
        n = min(n, args.max_examples)

    num_correct = 0
    total = 0

    with torch.no_grad(), autocast_ctx:
        for idx in range(n):
            conversation = ds[idx]
            image = conversation["ocrbench_image"]
            question = conversation["messages"][0]["content"]

            visual_tokens = build_visual_tokens(vision, image, device)
            pred = generate_answer(
                model,
                tokenizer,
                question,
                visual_tokens,
                device,
                max_tokens=32,
                temperature=0.0,
                top_k=None,
            )

            ok = ds.evaluate(conversation, pred)
            num_correct += int(ok)
            total += 1

            if total % 50 == 0 or total == n:
                acc = 100.0 * num_correct / total
                print(f"[{total}/{n}] current accuracy: {acc:.2f}%")

    if total > 0:
        acc = 100.0 * num_correct / total
        print(f"OCRBench accuracy (test): {num_correct}/{total} = {acc:.2f}%")
    else:
        print("No examples evaluated.")

    compute_cleanup()


if __name__ == "__main__":
    main()


