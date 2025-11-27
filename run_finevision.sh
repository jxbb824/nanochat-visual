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
# Download the released d34 chat model + tokenizer from Hugging Face
# Only if it isn't already present, to avoid overwriting any existing d34.

CHAT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d34"
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"

if [ ! -f "$CHAT_DIR/model_169150.pt" ] || [ ! -f "$CHAT_DIR/meta_169150.json" ]; then
    python -m scripts.download_nanochat_d34
else
    echo "Found existing d34 checkpoint in $CHAT_DIR, skipping download."
fi

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
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.finevision_sft -- --run=$WANDB_RUN --device_batch_size=2


