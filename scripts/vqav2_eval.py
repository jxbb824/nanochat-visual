"""
Evaluate the vision-language model on the VQAv2-small benchmark:
https://huggingface.co/datasets/merve/vqav2-small

We rely on tasks/vqav2.VQAv2Small to load the split via `datasets`,
and evaluate by checking if the model output contains all words of any
ground-truth answer (order-insensitive). Optionally, a judge LLM can
be used when simple matching fails.
"""

import argparse
import os
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List

import requests
import torch

from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.tokenizer import get_tokenizer
from scripts.finevision_cli import load_vlm_checkpoint
from tasks.vqav2 import VQAv2Small


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
                logits = logits / temperature
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


def build_openrouter_judge(api_key: str, model: str, timeout: float) -> Callable:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    call_id = {"n": 0}

    def judge(problem, completion: str) -> bool:
        question = problem.get("vqav2_question", "")
        answers: List[str] = problem.get("vqav2_answers", [])
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You grade VQA answers leniently. "
                        "Return 1 if the candidate clearly matches any of the ground-truth answers, "
                        "even if wording differs; otherwise return 0. Output only 1 or 0."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Ground-truth answers: {answers}\n"
                        f"Candidate answer: {completion}\n"
                        "Reply with 1 if the candidate corresponds to any ground-truth answer, else 0."
                    ),
                },
            ],
            "temperature": 0.0,
        }
        try:
            call_id["n"] += 1
            print(f"[judge] OpenRouter call #{call_id['n']}")
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            txt = resp.json()["choices"][0]["message"]["content"]
            first = (txt or "").strip()[:1]
            return first == "1"
        except requests.HTTPError as e:
            body = ""
            try:
                body = resp.text
            except Exception:
                body = ""
            print(f"[judge] OpenRouter HTTP {resp.status_code}: {body}")
            return False
        except Exception as e:
            print(f"[judge] OpenRouter request failed: {e}")
            return False

    return judge


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM on VQAv2-small benchmark")
    parser.add_argument("--vlm-tag", type=str, default="d20_finevision", help="VLM checkpoint tag under vlm_checkpoints/")
    parser.add_argument("--max-examples", type=int, default=-1, help="Max number of examples to evaluate (-1 = all)")
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device type")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--judge-openrouter", action="store_true", help="Use OpenRouter LLM judge when simple match fails")
    parser.add_argument("--judge-model", type=str, default="openai/gpt-5-nano", help="OpenRouter model name")
    parser.add_argument("--judge-api-key-env", type=str, default="OPENROUTER_API_KEY", help="Env var for OpenRouter API key")
    parser.add_argument("--judge-timeout", type=float, default=15.0, help="OpenRouter request timeout (seconds)")
    parser.add_argument("--judge-workers", type=int, default=8, help="Max threads for judge requests")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    if ddp and ddp_rank != 0:
        compute_cleanup()
        return

    tokenizer = get_tokenizer()
    model, vision = load_vlm_checkpoint(device, vlm_tag=args.vlm_tag)

    judge_fn = None
    if args.judge_openrouter:
        api_key = os.environ.get(args.judge_api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Set {args.judge_api_key_env} for OpenRouter judging")
        judge_fn = build_openrouter_judge(api_key, args.judge_model, args.judge_timeout)

    ds = VQAv2Small(split="validation", judge_fn=judge_fn)
    n = len(ds)
    if args.max_examples > 0:
        n = min(n, args.max_examples)

    preds = []
    with torch.no_grad(), autocast_ctx:
        for idx in range(n):
            conversation = ds[idx]
            image = conversation["vqav2_image"]
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
            preds.append((conversation, pred))

            if len(preds) % 50 == 0 or len(preds) == n:
                print(f"[gen {len(preds)}/{n}]")

    def eval_one(cp):
        conv, pred = cp
        return ds.evaluate(conv, pred)

    if judge_fn and args.judge_workers > 1:
        with ThreadPoolExecutor(max_workers=args.judge_workers) as ex:
            results = list(ex.map(eval_one, preds))
    else:
        results = [eval_one(cp) for cp in preds]

    num_correct = sum(int(ok) for ok in results)
    total = len(results)
    if total > 0:
        acc = 100.0 * num_correct / total
        print(f"VQAv2-small accuracy: {num_correct}/{total} = {acc:.2f}%")
    else:
        print("No examples evaluated.")

    compute_cleanup()


if __name__ == "__main__":
    main()
