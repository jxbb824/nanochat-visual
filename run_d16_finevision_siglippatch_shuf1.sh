#!/bin/bash

# Train a d16-based vision-language model on FineVision with SigLIP patch encoder (shuffle=1),
# then evaluate on MMStar and MME (small subsets).
#
# Usage (with wandb):
#   WANDB_RUN=d16_finevision_siglippatch_shuf1 \
#   NANOCHAT_BASE_DIR="$HOME/.cache/nanochat" \
#     screen -L -Logfile d16_finevision_siglippatch_shuf1.log -S d16_finevision_siglippatch_shuf1 bash run_d16_finevision_siglippatch_shuf1.sh

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
# uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=d16_finevision_siglippatch_shuf1
fi

# SigLIP patch encoder setup (pixel-unshuffle x1)
SIGCLIP_MODEL="ViT-B-16-SigLIP"
SIGCLIP_PRETRAINED="webli"
SIGCLIP_SHUFFLE_FACTOR=1
VLM_TAG_SUFFIX=finevision_sigclippatch_${SIGCLIP_MODEL}_shuf${SIGCLIP_SHUFFLE_FACTOR}
VLM_TAG="d16_${VLM_TAG_SUFFIX}"

# -----------------------------------------------------------------------------
# Download the nanochat-d16 base run from Hugging Face (if not already present)

BASE_D16_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d16"
if [ ! -d "$BASE_D16_DIR" ]; then
    python -m scripts.download_nanochat_d16
else
    echo "Found existing d16 base checkpoints in $BASE_D16_DIR, skipping download."
fi

# Use tokenizer specific to d16 (can be overridden by user)
export NANOCHAT_TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer/d16"

# -----------------------------------------------------------------------------
# Prepare FineVision local parquet subset (only once)

FINEVISION_PARQUET_DIR="$NANOCHAT_BASE_DIR/finevision_parquet"
if [ ! -d "$FINEVISION_PARQUET_DIR" ] || [ -z "$(ls -1 "$FINEVISION_PARQUET_DIR"/*.parquet 2>/dev/null)" ]; then
    echo "FineVision parquet subset not found at $FINEVISION_PARQUET_DIR, preparing a new subset..."
    python -m scripts.prepare_finevision_subset --num_shards=400
else
    echo "Found existing FineVision parquet subset at $FINEVISION_PARQUET_DIR, reusing it."
fi

# -----------------------------------------------------------------------------
# Run FineVision finetuning with d16 base as backbone (SigLIP, shuffle=1)

NPROC_PER_NODE=2
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
  -m scripts.finevision_sft -- \
  --run="$WANDB_RUN" \
  --source=base \
  --model_tag=d16 \
  --vlm_tag_suffix="$VLM_TAG_SUFFIX" \
  --vision_encoder_type=sigclip_patch \
  --vision_model_name="$SIGCLIP_MODEL" \
  --vision_pretrained="$SIGCLIP_PRETRAINED" \
  --vision_shuffle_factor="$SIGCLIP_SHUFFLE_FACTOR" \
  --device_batch_size=8 \
  --num_iterations=40000

# -----------------------------------------------------------------------------

python -m scripts.mmstar_eval --vlm-tag="$VLM_TAG" --device-type=cuda
python -m scripts.mme_eval    --vlm-tag="$VLM_TAG" --device-type=cuda
