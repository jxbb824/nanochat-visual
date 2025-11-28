#!/bin/bash

# Train a d10-based vision-language model on FineVision and then evaluate
# it on MMStar and MME (small subsets).
#
# Usage (with wandb):
#   WANDB_RUN=d10_finevision \
#   NANOCHAT_BASE_DIR="$HOME/.cache/nanochat" \
#     screen -L -Logfile d10_finevision.log -S d10_finevision bash run_d10_finevision.sh

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
    WANDB_RUN=d10_finevision
fi

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
# Prepare FineVision local parquet subset (only once)

FINEVISION_PARQUET_DIR="$NANOCHAT_BASE_DIR/finevision_parquet"
if [ ! -d "$FINEVISION_PARQUET_DIR" ] || [ -z "$(ls -1 "$FINEVISION_PARQUET_DIR"/*.parquet 2>/dev/null)" ]; then
    echo "FineVision parquet subset not found at $FINEVISION_PARQUET_DIR, preparing a new subset..."
    python -m scripts.prepare_finevision_subset --num_shards=100
else
    echo "Found existing FineVision parquet subset at $FINEVISION_PARQUET_DIR, reusing it."
fi

# -----------------------------------------------------------------------------
# Run FineVision finetuning with d10 base as backbone

NPROC_PER_NODE=2
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
  -m scripts.finevision_sft -- \
  --run="$WANDB_RUN" \
  --source=base \
  --model_tag=d10 \
  --device_batch_size=12 \
  --num_iterations=20000

# -----------------------------------------------------------------------------

python -m scripts.mmstar_eval --vlm-tag=d10_finevision --device-type=cuda
python -m scripts.mme_eval    --vlm-tag=d10_finevision --device-type=cuda


