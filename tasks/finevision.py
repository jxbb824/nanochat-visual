"""
FineVision dataset.
We use a small local subset of FineVisionMax prepared by scripts/prepare_finevision_subset.py.
Each row has:
- images: list of images
- texts: list of {user, assistant} turns
"""

import os
from glob import glob

from datasets import load_dataset

from nanochat.common import get_base_dir
from tasks.common import Task


class FineVision(Task):
    """FineVisionMax local subset (backed by local parquet shards)."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split == "train", "FineVision currently only supports split='train'"
        base_dir = get_base_dir()
        parquet_dir = os.path.join(base_dir, "finevision_parquet")
        if not os.path.isdir(parquet_dir):
            raise FileNotFoundError(
                f"FineVision parquet subset not found at {parquet_dir}. "
                "Run `python -m scripts.prepare_finevision_subset` first."
            )
        # parquet files may be nested in subdirectories (e.g. full/data-00000.parquet)
        files = sorted(glob(os.path.join(parquet_dir, "**", "*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(
                f"No parquet files found in {parquet_dir}. "
                "Run `python -m scripts.prepare_finevision_subset` first."
            )
        self.ds = load_dataset("parquet", data_files={"train": files}, split="train")
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        images = row["images"]
        texts = row["texts"]

        messages = []
        for turn in texts:
            user = turn.get("user", None)
            assistant = turn.get("assistant", None)
            if user is not None and len(user) > 0:
                messages.append({"role": "user", "content": user})
            if assistant is not None and len(assistant) > 0:
                messages.append({"role": "assistant", "content": assistant})

        assert len(messages) >= 2, "FineVision conversation must have at least one user/assistant pair"

        conversation = {
            "messages": messages,
            # Keep the raw images attached for downstream vision models.
            "images": images,
        }
        return conversation


