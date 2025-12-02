#!/bin/bash

# Train a d20-based vision-language model on FineVision with SigLIP patch encoder (shuffle=1),
# then evaluate on MMStar and MME (small subsets).
#
# Usage (with wandb):
#   WANDB_RUN=d20_finevision_siglippatch_shuf1 \
#   NANOCHAT_BASE_DIR="$HOME/.cache/nanochat" \
#     screen -L -Logfile d20_finevision_siglippatch_shuf1.log -S d20_finevision_siglippatch_shuf1 bash run_d20_finevision_siglippatch_shuf1.sh

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
    WANDB_RUN=d20_finevision_siglippatch_shuf1
fi

# SigLIP patch encoder setup (pixel-unshuffle x1)
SIGCLIP_MODEL="ViT-B-16-SigLIP"
SIGCLIP_PRETRAINED="webli"
SIGCLIP_SHUFFLE_FACTOR=1
VLM_TAG_SUFFIX=finevision_sigclippatch_${SIGCLIP_MODEL}_shuf${SIGCLIP_SHUFFLE_FACTOR}
VLM_TAG="d20_${VLM_TAG_SUFFIX}"

# -----------------------------------------------------------------------------
# Download the nanochat-d20 SFT run from Hugging Face (if not already present)

SFT_D20_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d20"
if [ ! -d "$SFT_D20_DIR" ]; then
    python -m scripts.download_nanochat_d20
else
    echo "Found existing d20 SFT checkpoints in $SFT_D20_DIR, skipping download."
fi

# Use tokenizer specific to d20 (can be overridden by user)
export NANOCHAT_TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer/d20"

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
# Run FineVision finetuning with d20 SFT as backbone (SigLIP, shuffle=1)

NPROC_PER_NODE=2
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
  -m scripts.finevision_sft -- \
  --run="$WANDB_RUN" \
  --source=sft \
  --model_tag=d20 \
  --vlm_tag_suffix="$VLM_TAG_SUFFIX" \
  --vision_encoder_type=sigclip_patch \
  --vision_model_name="$SIGCLIP_MODEL" \
  --vision_pretrained="$SIGCLIP_PRETRAINED" \
  --vision_shuffle_factor="$SIGCLIP_SHUFFLE_FACTOR" \
  --device_batch_size=6 \
  --num_iterations=80000

# -----------------------------------------------------------------------------

python -m scripts.mmstar_eval --vlm-tag="$VLM_TAG" --device-type=cuda
python -m scripts.mme_eval    --vlm-tag="$VLM_TAG" --device-type=cuda
