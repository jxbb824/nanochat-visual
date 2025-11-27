"""
Finetune or experiment with FineVisionMax as a vision-language dataset.

This script focuses on data loading and batching, in a style similar to chat_sft.py:
- we build a Task-style dataset (FineVision)
- we define a generator that yields (inputs, targets, images_batch)

The actual training loop and vision model integration can be added later.
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import wandb
from contextlib import nullcontext
from PIL import Image

from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    DummyWandb,
    autodetect_device_type,
    get_base_dir,
)
from nanochat.checkpoint_manager import load_model, save_checkpoint, load_checkpoint
from nanochat.vision import CLIPVisionPrefixEncoder
from tasks.finevision import FineVision


# -----------------------------------------------------------------------------
# Basic settings (keep minimal for now)

run = "dummy"  # wandb run name default ("dummy" is special - we won't log to wandb)
source = "sft"  # base|mid|sft|rl, we load the released d34 chat model
model_tag = "d34"  # default to the released d34 chat model
step = None
device_type = ""  # cuda|cpu|mps (empty => autodetect)
dtype = "bfloat16"
device_batch_size = 2  # small default for quick experiments
# only use a subset of FineVision for quicker experiments
max_examples = 20000
# vision encoder hyperparameters
vision_model_name = "ViT-B-32"
vision_pretrained = "openai"
vision_num_tokens = 16
vision_lr = 1e-4
vision_weight_decay = 0.01
# training loop
num_iterations = 10000
save_every = 1000
resume_from_step = -1

# now allow CLI to override the settings via the configurator
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join("nanochat", "configurator.py")).read())
user_config = {k: globals()[k] for k in config_keys}


# -----------------------------------------------------------------------------
# Compute / model / tokenizer init

device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-finevision", name=run, config=user_config)

model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
for p in model.parameters():
    p.requires_grad = False
model.eval()


# -----------------------------------------------------------------------------
# Dataset

train_ds = FineVision(split="train", stop=max_examples)
print0(f"FineVision train size (logical): {len(train_ds)} examples (subset)")


# -----------------------------------------------------------------------------
# DataLoader


def finevision_data_generator(dataset, batch_size):
    """
    Yields (inputs, targets, images_batch).
    - inputs, targets: 2D tensors, shape (batch, seq_len)
    - images_batch: Python list of length `batch`, each element is the `images` field
      from the underlying dataset row.
    """

    pad_token_id = tokenizer.encode_special("<|assistant_end|>")

    def collate_and_yield(batch):
        # batch: list of (ids, mask, images)
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask, _ in batch) - 1
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

        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets, images_batch

    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            images = doc.get("images", None)
            batch.append((ids, mask, images))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []


train_loader = finevision_data_generator(train_ds, batch_size=device_batch_size)


# -----------------------------------------------------------------------------
# Vision encoder and optimizer

vision = CLIPVisionPrefixEncoder(
    d_model=model.config.n_embd,
    num_tokens=vision_num_tokens,
    model_name=vision_model_name,
    pretrained=vision_pretrained,
    device=device,
).to(device)
vision.train()

optimizer = torch.optim.AdamW(
    vision.parameters(),
    lr=vision_lr,
    weight_decay=vision_weight_decay,
)


def build_image_batch(images_batch):
    tensors = []
    for images in images_batch:
        if images is None or len(images) == 0:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        else:
            img = images[0]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
        t = vision.preprocess(img)  # CLIP preprocess -> (3, H, W)
        tensors.append(t)
    return torch.stack(tensors, dim=0).to(device)


# -----------------------------------------------------------------------------
# Checkpoint helpers

base_dir = get_base_dir()
vision_ckpt_dir = os.path.join(base_dir, "finevision_checkpoints", model_tag)
os.makedirs(vision_ckpt_dir, exist_ok=True)

start_step = 0
if resume_from_step >= 0:
    vision_state, optim_state, meta_ckpt = load_checkpoint(
        vision_ckpt_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank
    )
    vision.load_state_dict(vision_state)
    optimizer.load_state_dict(optim_state)
    start_step = resume_from_step
    print0(f"Resuming FineVision training from step {resume_from_step}")


# -----------------------------------------------------------------------------
# Training loop

train_iter = iter(train_loader)

for step in range(start_step, num_iterations):
    last_step = step == num_iterations - 1

    inputs, targets, images_batch = next(train_iter)
    images_tensor = build_image_batch(images_batch)

    with autocast_ctx:
        visual_tokens = vision(images_tensor)
        loss = model(inputs, targets, visual_embs=visual_tokens)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    loss_item = loss.item()
    print0(f"Step {step:05d}/{num_iterations:05d} | loss: {loss_item:.6f}")
    wandb_run.log(
        {
            "step": step,
            "train/loss": loss_item,
        }
    )

    if last_step or (save_every > 0 and step > 0 and step % save_every == 0):
        save_checkpoint(
            vision_ckpt_dir,
            step,
            vision.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "model_tag": model_tag,
                "vision_num_tokens": vision_num_tokens,
                "user_config": user_config,
            },
            rank=ddp_rank,
        )

    if last_step:
        break


# Cleanup
wandb_run.finish()
compute_cleanup()
