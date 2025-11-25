"""
FineVision dataset adapter for NanoChat / NanoGPT training.

Replaces:
- Karpathy FineWeb text-only parquet
With:
- FineVision (image+text) parquet
And returns:
- text batches identical in format to original NanoGPT dataset.py
"""

import os
from nanochat.common import get_base_dir

from finevision_loader import (
    download_finevision_parquets,
    finevision_text_batches,
)

# =====================================================================
# Dataset configuration
# =====================================================================

FINEVISION_SUBSET = "CoSyn_400k_chart"
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "finevision_data")
os.makedirs(DATA_DIR, exist_ok=True)


def list_parquet_files(*args, **kwargs):
    """
    Dummy API for compatibility.

    The real logic is inside finevision_loader.
    We return a list of local FineVision shard paths
    because tokenizing loader expects this to exist.
    """
    return download_finevision_parquets(
        FINEVISION_SUBSET,
        root=DATA_DIR,
    )

def parquets_iter_batched(split, start=0, step=1, max_shards=None):
    """
    Replacement of original NanoGPT parquets_iter_batched.
    Now backed by FineVision text batches.

    split: "train" or "val"
    start / step: for DDP (rank, world_size)
    max_shards: use only first K FineVision shards
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    for image_batch, text_batch in finevision_text_batches(
        subset=FINEVISION_SUBSET,
        split=split,
        root=DATA_DIR,
        start=start,
        step=step,
    ):
        yield image_batch, text_batch