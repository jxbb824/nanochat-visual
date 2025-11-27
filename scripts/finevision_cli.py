"""
Simple CLI to query the FineVision-finetuned model with an image + question.

Example:
python -m scripts.finevision_cli -i path/to/image.png -q "What is in the picture?"
"""

import argparse
import os

import torch
from PIL import Image
from contextlib import nullcontext

from nanochat.common import compute_init, autodetect_device_type, get_base_dir
from nanochat.checkpoint_manager import load_model, load_checkpoint, find_last_step
from nanochat.vision import CLIPVisionPrefixEncoder


def build_visual_encoder(model, device, num_tokens=16, model_name="ViT-B-32", pretrained="openai"):
    vision = CLIPVisionPrefixEncoder(
        d_model=model.config.n_embd,
        num_tokens=num_tokens,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    ).to(device)

    # load vision weights if available
    base_dir = get_base_dir()
    vision_ckpt_dir = os.path.join(base_dir, "finevision_checkpoints", "d34")
    if os.path.isdir(vision_ckpt_dir):
        step = find_last_step(vision_ckpt_dir)
        state_dict, _, _ = load_checkpoint(vision_ckpt_dir, step, device, load_optimizer=False, rank=0)
        vision.load_state_dict(state_dict, strict=True)
    vision.eval()
    return vision


def build_visual_tokens(vision, image_path, device):
    img = Image.open(image_path).convert("RGB")
    img_tensor = vision.preprocess(img)  # (3, H, W)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        visual_tokens = vision(img_tensor)
    return visual_tokens


def generate_answer(model, tokenizer, question, visual_tokens, device, max_tokens=128, temperature=0.6, top_k=50):
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
        logits = logits[:, -1, :]  # last position
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            vals, idx = torch.topk(logits, k, dim=-1)
            vals = vals / temperature
            probs = torch.softmax(vals, dim=-1)
            next_id_rel = torch.multinomial(probs, num_samples=1)
            next_id = idx.gather(1, next_id_rel).item()
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_id)
        if next_id == assistant_end:
            break
        generated.append(next_id)
    answer = tokenizer.decode(generated)
    return answer


def main():
    parser = argparse.ArgumentParser(description="FineVision CLI")
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("-q", "--question", type=str, required=True, help="User question about the image")
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device type")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    assert os.path.exists(args.image), f"Image file not found: {args.image}"

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag="d34")
    vision = build_visual_encoder(model, device)

    with autocast_ctx:
        visual_tokens = build_visual_tokens(vision, args.image, device)
        answer = generate_answer(model, tokenizer, args.question, visual_tokens, device)

    print("\nQuestion:", args.question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()


