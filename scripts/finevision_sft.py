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
from nanochat.vision import (
    CLIPVisionPrefixEncoder,
    CLIPPatchVisionPrefixEncoder,
    SigCLIPPatchVisionPrefixEncoder,
    PatchVisionPrefixEncoder,
)
from tasks.finevision import FineVision
from tasks.cauldron import Cauldron
from tasks.small_cauldron import SmallCauldron
from tasks.mmstar import MMStar
from tasks.mme import MME
from tasks.vqav2 import VQAv2Small
from scripts.mmstar_eval import (
    build_visual_tokens as mmstar_build_visual_tokens,
    generate_answer as mmstar_generate_answer,
)
from scripts.mme_eval import (
    build_visual_tokens as mme_build_visual_tokens,
    generate_answer as mme_generate_answer,
)
from scripts.vqav2_eval import (
    build_visual_tokens as vqav2_build_visual_tokens,
    generate_answer as vqav2_generate_answer,
)


# -----------------------------------------------------------------------------
# Basic settings (keep minimal for now)

run = "dummy"  # wandb run name default ("dummy" is special - we won't log to wandb)
source = "sft"  # base|mid|sft|rl, default to SFT checkpoint
model_tag = "d20"  # default to the released d20 chat model
step = None
device_type = ""  # cuda|cpu|mps (empty => autodetect)
dtype = "bfloat16"
device_batch_size = 4  # small default for quick experiments
# training dataset: "finevision" (local parquet subset) or "cauldron" (HuggingFaceM4/the_cauldron)
train_dataset = "finevision"
# only use a subset of the training data for quicker experiments (applies to both datasets)
max_examples = 900000
# shuffle training data each epoch to reduce distribution drift
shuffle_data = True
# Cauldron-specific hyperparameters
cauldron_subset = "ai2d"
cauldron_split = "train"
cauldron_cache_root = "/data"
# vision encoder hyperparameters
vision_model_name = "ViT-B-32"
vision_pretrained = "openai"
vision_num_tokens = 64
# vision encoder type:
# - "clip_global"  -> CLIPVisionPrefixEncoder (global image embedding)
# - "clip_patch"   -> CLIPPatchVisionPrefixEncoder (CLIP ViT patch tokens)
# - "patch"        -> PatchVisionPrefixEncoder (learned Conv2d patch embedding)
# - "sigclip_patch"-> SigCLIPPatchVisionPrefixEncoder (SigCLIP ViT patch tokens with pixel unshuffle)
vision_encoder_type = "clip_patch"
# patch-based encoder hyperparameters
# - for "clip_patch": only `vision_pool` is used (spatial pooling over CLIP patches)
# - for "patch": all three are used (image_size, patch_size, pool)
# - for "sigclip_patch": `vision_shuffle_factor` controls pixel-unshuffle spatial reduction
vision_image_size = 224
vision_patch_size = 16
vision_pool = 2
vision_shuffle_factor = 2
vision_lr = 3e-3
vision_weight_decay = 0.01
# LLM optimizer hyperparameters (reuse GPT.setup_optimizers style)
llm_unembedding_lr = 0.004
llm_embedding_lr = 0.2 #0.2
llm_matrix_lr = 0.02
llm_weight_decay = 0.0
llm_init_lr_frac = 0.02
# training loop
num_iterations = 50000
save_every = 20000
resume_from_step = -1
# vision eval
vis_eval_every = 500
mmstar_eval_examples = 100
mme_eval_examples = 100
vqav2_eval_examples = 100
#
# checkpoint naming: final directory is f"{model_tag}_{vlm_tag_suffix}"
vlm_tag_suffix = "finevision"

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
model.train()


# -----------------------------------------------------------------------------
# Dataset

if train_dataset == "finevision":
    train_ds = FineVision(split="train", stop=max_examples)
    print0(f"FineVision train size (logical): {len(train_ds)} examples (subset)")
    # use the last parquet shard as a held-out test set
    test_ds = FineVision(split="test")
    print0(f"FineVision test size (logical): {len(test_ds)} examples (last shard)")
elif train_dataset == "cauldron":
    train_ds = Cauldron(
        subset=cauldron_subset,
        split=cauldron_split,
        cache_root=cauldron_cache_root,
        stop=max_examples,
    )
    print0(
        f"The Cauldron subset='{cauldron_subset}' split='{cauldron_split}' "
        f"train size (logical): {len(train_ds)} examples"
    )
elif train_dataset == "small_cauldron":
    train_ds = SmallCauldron(
        split=cauldron_split,
        cache_root=cauldron_cache_root,
        stop=max_examples,
    )
    print0(f"small-cauldron train size (logical): {len(train_ds)} examples")
else:
    raise ValueError(f"Unsupported train_dataset: {train_dataset}")

if train_dataset != "finevision":
    test_ds = None


# -----------------------------------------------------------------------------
# DataLoader


def collate_finevision_batch(batch, pad_token_id):
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


def finevision_data_generator(dataset, batch_size, shuffle=True):
    """
    Yields (inputs, targets, images_batch).
    - inputs, targets: 2D tensors, shape (batch, seq_len)
    - images_batch: Python list of length `batch`, each element is the `images` field
      from the underlying dataset row.
    """

    pad_token_id = tokenizer.encode_special("<|assistant_end|>")

    batch = []
    rng = torch.Generator()
    rng.manual_seed(42)
    while True:
        if shuffle:
            indices = torch.randperm(len(dataset), generator=rng)
        else:
            indices = torch.arange(len(dataset))
        for i in indices.tolist():
            if i % ddp_world_size != ddp_rank:
                continue
            doc = dataset[i]
            try:
                ids, mask = tokenizer.render_conversation(doc)
            except Exception as e:
                # Be robust to any bad conversations: skip and continue training
                if master_process:
                    print0(f"Skipping FineVision example {i} due to tokenization error: {e}")
                continue
            if len(ids) < 2:
                # too short to form inputs/targets
                continue
            images = doc.get("images", None)
            batch.append((ids, mask, images))
            if len(batch) == batch_size:
                yield collate_finevision_batch(batch, pad_token_id)
                batch = []


train_loader = finevision_data_generator(train_ds, batch_size=device_batch_size, shuffle=shuffle_data)


# -----------------------------------------------------------------------------
# Vision encoder and optimizers

if vision_encoder_type == "clip_global":
    vision = CLIPVisionPrefixEncoder(
        d_model=model.config.n_embd,
        num_tokens=vision_num_tokens,
        model_name=vision_model_name,
        pretrained=vision_pretrained,
        device=device,
    ).to(device)
elif vision_encoder_type == "clip_patch":
    vision = CLIPPatchVisionPrefixEncoder(
        d_model=model.config.n_embd,
        model_name=vision_model_name,
        pretrained=vision_pretrained,
        pool=vision_pool,
        device=device,
    ).to(device)
    vision_num_tokens = getattr(vision, "num_tokens", vision_num_tokens)
    user_config["vision_model_name"] = vision_model_name
    user_config["vision_pretrained"] = vision_pretrained
    user_config["vision_pool"] = vision_pool
elif vision_encoder_type == "sigclip_patch":
    vision = SigCLIPPatchVisionPrefixEncoder(
        d_model=model.config.n_embd,
        model_name=vision_model_name,
        pretrained=vision_pretrained,
        shuffle_factor=vision_shuffle_factor,
        device=device,
    ).to(device)
    vision_num_tokens = getattr(vision, "num_tokens", vision_num_tokens)
    user_config["vision_model_name"] = vision_model_name
    user_config["vision_pretrained"] = vision_pretrained
    user_config["vision_shuffle_factor"] = vision_shuffle_factor
elif vision_encoder_type == "patch":
    vision = PatchVisionPrefixEncoder(
        d_model=model.config.n_embd,
        image_size=vision_image_size,
        patch_size=vision_patch_size,
        pool=vision_pool,
    ).to(device)
    # align logical num_tokens with the encoder for logging/checkpointing
    vision_num_tokens = getattr(vision, "num_tokens", vision_num_tokens)
    user_config["vision_image_size"] = vision_image_size
    user_config["vision_patch_size"] = vision_patch_size
    user_config["vision_pool"] = vision_pool
else:
    raise ValueError(f"Unsupported vision_encoder_type: {vision_encoder_type}")

# make sure user_config reflects the actual encoder settings
user_config["vision_encoder_type"] = vision_encoder_type
user_config["vision_num_tokens"] = vision_num_tokens

# If the user did not override the suffix explicitly, derive a more descriptive one
if vlm_tag_suffix == "finevision":
    if vision_encoder_type == "clip_global":
        vlm_tag_suffix = "finevision_clip"
    elif vision_encoder_type == "clip_patch":
        safe_name = str(vision_model_name).replace("/", "-")
        vlm_tag_suffix = f"finevision_clippatch_{safe_name}_pool{vision_pool}"
    elif vision_encoder_type == "sigclip_patch":
        safe_name = str(vision_model_name).replace("/", "-")
        vlm_tag_suffix = f"finevision_sigclippatch_{safe_name}_shuf{vision_shuffle_factor}"
    elif vision_encoder_type == "patch":
        vlm_tag_suffix = f"finevision_patch_i{vision_image_size}_p{vision_patch_size}_pool{vision_pool}"
user_config["vlm_tag_suffix"] = vlm_tag_suffix

vision.train()

# separate optimizers for LLM and vision encoder
llm_optimizers = model.setup_optimizers(
    unembedding_lr=llm_unembedding_lr,
    embedding_lr=llm_embedding_lr,
    matrix_lr=llm_matrix_lr,
    weight_decay=llm_weight_decay,
)
for opt in llm_optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * llm_init_lr_frac
        group["initial_lr"] = group["lr"]

vision_optimizer = torch.optim.AdamW(
    vision.parameters(),
    lr=vision_lr,
    weight_decay=vision_weight_decay,
)
for group in vision_optimizer.param_groups:
    group["initial_lr"] = group["lr"]


def get_lr_multiplier(step: int) -> float:
    if num_iterations <= 0:
        return 1.0
    return 1.0 - float(step) / float(num_iterations)


def build_image_batch(images_batch):
    tensors = []
    for images in images_batch:
        if images is None or len(images) == 0:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        else:
            img = images[0]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
        # ensure 3-channel RGB before preprocessing (some images may be RGBA or grayscale)
        img = img.convert("RGB")
        t = vision.preprocess(img)  # preprocess -> (3, H, W)
        tensors.append(t)
    return torch.stack(tensors, dim=0).to(device)


def evaluate_finevision_loss(model, vision, tokenizer, dataset, batch_size, device, autocast_ctx, max_examples=128):
    if dataset is None:
        return None

    was_training_model = model.training
    was_training_vision = vision.training
    model.eval()
    vision.eval()

    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    total_loss = 0.0
    num_batches = 0

    batch = []
    with torch.no_grad(), autocast_ctx:
        dataset_size = min(len(dataset), max_examples)
        for i in range(dataset_size):
            doc = dataset[i]
            try:
                ids, mask = tokenizer.render_conversation(doc)
            except Exception as e:
                if master_process:
                    print0(f"Skipping FineVision test example {i} due to tokenization error: {e}")
                continue
            if len(ids) < 2:
                continue
            images = doc.get("images", None)
            batch.append((ids, mask, images))
            if len(batch) == batch_size:
                inputs, targets, images_batch = collate_finevision_batch(batch, pad_token_id)
                images_tensor = build_image_batch(images_batch)
                visual_tokens = vision(images_tensor)
                loss = model(inputs, targets, visual_embs=visual_tokens)
                total_loss += loss.item()
                num_batches += 1
                batch = []

        if batch:
            inputs, targets, images_batch = collate_finevision_batch(batch, pad_token_id)
            images_tensor = build_image_batch(images_batch)
            visual_tokens = vision(images_tensor)
            loss = model(inputs, targets, visual_embs=visual_tokens)
            total_loss += loss.item()
            num_batches += 1

    if was_training_model:
        model.train()
    if was_training_vision:
        vision.train()

    if num_batches == 0:
        return None
    return total_loss / num_batches


# -----------------------------------------------------------------------------
# Vision benchmark evaluation helpers (MMStar + MME)

def evaluate_mmstar(model, vision, tokenizer, device, autocast_ctx, max_examples):
    ds = MMStar()
    n = len(ds) if max_examples <= 0 else min(len(ds), max_examples)
    was_training_model = model.training
    was_training_vision = vision.training
    model.eval()
    vision.eval()
    num_correct, total = 0, 0
    with torch.no_grad(), autocast_ctx:
        for idx in range(n):
            conversation = ds[idx]
            image = conversation["mmstar_image"]
            question = conversation["messages"][0]["content"]
            visual_tokens = mmstar_build_visual_tokens(vision, image, device)
            pred = mmstar_generate_answer(
                model,
                tokenizer,
                question,
                visual_tokens,
                device,
                max_tokens=64,
                temperature=0.0,
                top_k=None,
            )
            ok = ds.evaluate(conversation, pred)
            num_correct += int(ok)
            total += 1
    acc = float(num_correct) / total if total > 0 else 0.0
    if was_training_model:
        model.train()
    if was_training_vision:
        vision.train()
    return acc


def evaluate_mme(model, vision, tokenizer, device, autocast_ctx, max_examples):
    ds = MME()
    n = len(ds) if max_examples <= 0 else min(len(ds), max_examples)
    was_training_model = model.training
    was_training_vision = vision.training
    model.eval()
    vision.eval()
    num_correct, total = 0, 0
    with torch.no_grad(), autocast_ctx:
        for idx in range(n):
            conversation = ds[idx]
            image = conversation["mme_image"]
            question = conversation["messages"][0]["content"]
            visual_tokens = mme_build_visual_tokens(vision, image, device)
            pred = mme_generate_answer(
                model,
                tokenizer,
                question,
                visual_tokens,
                device,
                max_tokens=8,
                temperature=0.0,
                top_k=None,
            )
            ok = ds.evaluate(conversation, pred)
            num_correct += int(ok)
            total += 1
    acc = float(num_correct) / total if total > 0 else 0.0
    if was_training_model:
        model.train()
    if was_training_vision:
        vision.train()
    return acc


def evaluate_vqav2(model, vision, tokenizer, device, autocast_ctx, max_examples):
    ds = VQAv2Small(split="validation")
    n = len(ds) if max_examples <= 0 else min(len(ds), max_examples)
    was_training_model = model.training
    was_training_vision = vision.training
    model.eval()
    vision.eval()
    num_correct, total = 0, 0
    with torch.no_grad(), autocast_ctx:
        for idx in range(n):
            conversation = ds[idx]
            image = conversation["vqav2_image"]
            question = conversation["messages"][0]["content"]
            visual_tokens = vqav2_build_visual_tokens(vision, image, device)
            pred = vqav2_generate_answer(
                model,
                tokenizer,
                question,
                visual_tokens,
                device,
                max_tokens=64,
                temperature=0.0,
                top_k=None,
            )
            ok = ds.evaluate(conversation, pred)
            num_correct += int(ok)
            total += 1
    acc = float(num_correct) / total if total > 0 else 0.0
    if was_training_model:
        model.train()
    if was_training_vision:
        vision.train()
    return acc


# -----------------------------------------------------------------------------
# Checkpoint helpers (joint GPT + vision)

base_dir = get_base_dir()
vlm_tag = f"{model_tag}_{vlm_tag_suffix}"
vlm_ckpt_dir = os.path.join(base_dir, "vlm_checkpoints", vlm_tag)
os.makedirs(vlm_ckpt_dir, exist_ok=True)

start_step = 0
if resume_from_step >= 0:
    ckpt_path = os.path.join(vlm_ckpt_dir, f"vlm_{resume_from_step:06d}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        vision.load_state_dict(ckpt["vision_state"])
        llm_state = ckpt["optim_state"].get("llm")
        if isinstance(llm_state, list):
            for opt, state in zip(llm_optimizers, llm_state):
                opt.load_state_dict(state)
        elif llm_state is not None and len(llm_optimizers) > 0:
            # backward compatibility: older checkpoints with a single LLM optimizer
            llm_optimizers[0].load_state_dict(llm_state)
        vision_optimizer.load_state_dict(ckpt["optim_state"]["vision"])
        start_step = ckpt["step"] + 1
        print0(f"Resuming FineVision training from step {ckpt['step']}")


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

    for opt in llm_optimizers:
        opt.zero_grad(set_to_none=True)
    vision_optimizer.zero_grad(set_to_none=True)

    loss.backward()

    lrm = get_lr_multiplier(step)
    for opt in llm_optimizers:
        for group in opt.param_groups:
            base_lr = group.get("initial_lr", group["lr"])
            group["lr"] = base_lr * lrm
    for group in vision_optimizer.param_groups:
        base_lr = group.get("initial_lr", group["lr"])
        group["lr"] = base_lr * lrm

    for opt in llm_optimizers:
        opt.step()
    vision_optimizer.step()

    loss_item = loss.item()
    print0(f"Step {step:05d}/{num_iterations:05d} | loss: {loss_item:.6f}")
    wandb_run.log(
        {
            "step": step,
            "train/loss": loss_item,
        }
    )

    # periodic visual benchmark evaluation (FineVision test loss + MMStar + MME)
    if master_process and (last_step or (vis_eval_every > 0 and step > 0 and step % vis_eval_every == 0)):
        finevision_test_loss = evaluate_finevision_loss(
            model, vision, tokenizer, test_ds, device_batch_size, device, autocast_ctx
        )
        mmstar_acc = evaluate_mmstar(model, vision, tokenizer, device, autocast_ctx, mmstar_eval_examples)
        mme_acc = evaluate_mme(model, vision, tokenizer, device, autocast_ctx, mme_eval_examples)
        vqav2_acc = evaluate_vqav2(model, vision, tokenizer, device, autocast_ctx, vqav2_eval_examples)
        log_payload = {
            "step": step,
            "mmstar/acc": mmstar_acc,
            "mme/acc": mme_acc,
            "vqav2/acc": vqav2_acc,
        }
        if finevision_test_loss is not None:
            log_payload["finevision/test_loss"] = finevision_test_loss
        wandb_run.log(log_payload)

    if last_step or (save_every > 0 and step > 0 and step % save_every == 0):
        ckpt = {
            "step": step,
            "model_state": model.state_dict(),
            "vision_state": vision.state_dict(),
            "model_config": meta["model_config"],
            "user_config": user_config,
            "optim_state": {
                "llm": [opt.state_dict() for opt in llm_optimizers],
                "vision": vision_optimizer.state_dict(),
            },
        }
        ckpt_path = os.path.join(vlm_ckpt_dir, f"vlm_{step:06d}.pt")
        torch.save(ckpt, ckpt_path)

    if last_step:
        break


# Cleanup
wandb_run.finish()
compute_cleanup()
