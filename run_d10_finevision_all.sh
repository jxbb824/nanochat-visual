#!/bin/bash

# Run four FineVision experiments back-to-back with distinct wandb runs.

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

BASE_D10_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d10"
if [ ! -d "$BASE_D10_DIR" ]; then
    python -m scripts.download_nanochat_d10
else
    echo "Found existing d10 base checkpoints in $BASE_D10_DIR, skipping download."
fi

export NANOCHAT_TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer/d10"

FINEVISION_PARQUET_DIR="$NANOCHAT_BASE_DIR/finevision_parquet"
if [ ! -d "$FINEVISION_PARQUET_DIR" ] || [ -z "$(ls -1 "$FINEVISION_PARQUET_DIR"/*.parquet 2>/dev/null)" ]; then
    echo "FineVision parquet subset not found at $FINEVISION_PARQUET_DIR, preparing a new subset..."
    python -m scripts.prepare_finevision_subset --num_shards=400
else
    echo "Found existing FineVision parquet subset at $FINEVISION_PARQUET_DIR, reusing it."
fi

NPROC_PER_NODE=${NPROC_PER_NODE:-2}
ITERATIONS=${NUM_ITERATIONS:-10000}
BATCH=${DEVICE_BATCH_SIZE:-16}

configs=(
  # "d10_finevision_clip|finevision_clip|clip_global|ViT-B-32|openai|"
  # "d10_finevision_clippatch|finevision_clippatch_ViT-B-32_pool1|clip_patch|ViT-B-32|openai|--vision_pool=1"
  # "d10_finevision_siglippatch_shuf2|finevision_sigclippatch_ViT-B-16-SigLIP_shuf2|sigclip_patch|ViT-B-16-SigLIP|webli|--vision_shuffle_factor=2"
  "d10_finevision_siglippatch_shuf1|finevision_sigclippatch_ViT-B-16-SigLIP_shuf1|sigclip_patch|ViT-B-16-SigLIP|webli|--vision_shuffle_factor=1"
)

for idx in "${!configs[@]}"; do
  cfg="${configs[$idx]}"
  IFS='|' read -r RUN_NAME VLM_SUFFIX ENC MODEL PRETRAIN EXTRA <<< "$cfg"
  export WANDB_RUN="$RUN_NAME"
  torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.finevision_sft -- \
    --run="$RUN_NAME" \
    --source=base \
    --model_tag=d10 \
    --vlm_tag_suffix="$VLM_SUFFIX" \
    --vision_encoder_type="$ENC" \
    --vision_model_name="$MODEL" \
    --vision_pretrained="$PRETRAIN" \
    --device_batch_size=$BATCH \
    --num_iterations=$ITERATIONS \
    $EXTRA

  if [ "$idx" -lt $((${#configs[@]} - 1)) ]; then
    echo "Sleeping 10s before next run..."
    sleep 10
  fi
done
