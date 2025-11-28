#!/bin/bash

# Simple script to finetune the nanochat d34 chat model on FineVision with a CLIP encoder.

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
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
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Download the nanochat-d20 run from Hugging Face
# Only if it isn't already present.

CHAT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d20"

if [ ! -d "$CHAT_DIR" ]; then
    python -m scripts.download_nanochat_d20
else
    echo "Found existing d20 checkpoints in $CHAT_DIR, skipping download."
fi

# Use tokenizer specific to d20 (can be overridden by user)
export NANOCHAT_TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer/d20"

# -----------------------------------------------------------------------------
# Prepare FineVision local parquet subset (only once)

FINEVISION_PARQUET_DIR="$NANOCHAT_BASE_DIR/finevision_parquet"
if [ ! -d "$FINEVISION_PARQUET_DIR" ] || [ -z "$(ls -1 "$FINEVISION_PARQUET_DIR"/*.parquet 2>/dev/null)" ]; then
    echo "FineVision parquet subset not found at $FINEVISION_PARQUET_DIR, preparing a new subset..."
    python -m scripts.prepare_finevision_subset --num_shards=100
else
    echo "Found existing FineVision parquet subset at $FINEVISION_PARQUET_DIR, reusing it."
fi

# -----------------------------------------------------------------------------
# Run FineVision finetuning

NPROC_PER_NODE=2
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.finevision_sft -- --run=$WANDB_RUN --device_batch_size=6


