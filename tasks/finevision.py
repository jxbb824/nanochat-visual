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

    def __init__(self, split: str = "train", **kwargs):
        super().__init__(**kwargs)
        assert split in ("train", "test"), "FineVision supports split='train' or 'test'"

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

        if len(files) == 1:
            # Degenerate case: only one shard available, use it for both splits.
            selected_files = files
        elif split == "train":
            # Use all but the last shard for training.
            selected_files = files[:-1]
        else:
            # Use only the last shard for testing.
            selected_files = files[-1:]

        self.ds = load_dataset("parquet", data_files={"train": selected_files}, split="train")
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        images = row["images"]
        texts = row["texts"]

        messages = []
        for turn in texts:
            # we require strict user/assistant pairs for render_conversation
            user = (turn.get("user") or "").strip()
            assistant = (turn.get("assistant") or "").strip()
            if not user or not assistant:
                # skip incomplete turns that would break user/assistant alternation
                continue
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})

        assert len(messages) >= 2, "FineVision conversation must have at least one user/assistant pair"

        conversation = {
            "messages": messages,
            # Keep the raw images attached for downstream vision models.
            "images": images,
        }
        return conversation


