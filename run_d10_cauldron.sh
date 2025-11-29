#!/bin/bash

# Train a d10-based vision-language model on The Cauldron (HuggingFaceM4/the_cauldron)
# using the same FineVision-style framework.
#
# Usage (with wandb):
#   WANDB_RUN=d10_cauldron \
#   NANOCHAT_BASE_DIR="$HOME/.cache/nanochat" \
#     screen -L -Logfile d10_cauldron.log -S d10_cauldron bash run_d10_cauldron.sh

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=d10_cauldron
fi

# More descriptive VLM tag for the CLIP-based patch encoder on Cauldron
# Note: ViT-B-32 has 7x7=49 patches, so pool must be 1 or 7.
VLM_TAG_SUFFIX=cauldron_clippatch_ViT-B-32_pool1_ai2d
VLM_TAG="d10_${VLM_TAG_SUFFIX}"

# -----------------------------------------------------------------------------
# Download the nanochat-d10 base run from Hugging Face (if not already present)

BASE_D10_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d10"
if [ ! -d "$BASE_D10_DIR" ]; then
    python -m scripts.download_nanochat_d10
else
    echo "Found existing d10 base checkpoints in $BASE_D10_DIR, skipping download."
fi

# Use tokenizer specific to d10 (can be overridden by user)
export NANOCHAT_TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer/d10"

# -----------------------------------------------------------------------------
# Run Cauldron finetuning with d10 base as backbone

NPROC_PER_NODE=2
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
  -m scripts.finevision_sft -- \
  --run="$WANDB_RUN" \
  --source=base \
  --model_tag=d10 \
  --vlm_tag_suffix="$VLM_TAG_SUFFIX" \
  --train_dataset=cauldron \
  --cauldron_subset=ai2d \
  --cauldron_cache_root=/data \
  --vision_encoder_type=clip_patch \
  --vision_pool=1 \
  --device_batch_size=16 \
  --num_iterations=20000

# -----------------------------------------------------------------------------

python -m scripts.mmstar_eval --vlm-tag="$VLM_TAG" --device-type=cuda
python -m scripts.mme_eval    --vlm-tag="$VLM_TAG" --device-type=cuda


