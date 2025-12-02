"""
The Cauldron multimodal dataset:
https://huggingface.co/datasets/HuggingFaceM4/the_cauldron

This is a collection of ~50 vision-language datasets used in Idefics/SmolVLM.
Each subset shares the same schema:
- images: list of images
- texts: list of {user, assistant, source} turns

We mirror the FineVision task so that training code (e.g. finevision_sft.py)
can reuse the same data flow:
- Task returns a dict with "messages" (user/assistant pairs) and "images".
"""

from __future__ import annotations

import os
from typing import Any, List

from datasets import get_dataset_config_names, load_dataset

from tasks.common import Task


class Cauldron(Task):
    """
    Wrapper around HuggingFaceM4/the_cauldron.

    By default we load the "ai2d" subset and the "train" split.
    Use subset="all" to iterate over every available config sequentially
    (without concatenating into memory).
    """

    def __init__(
        self,
        subset: str = "ai2d",
        split: str = "train",
        cache_root: str = "/data",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if cache_root is None or cache_root == "":
            cache_dir = None
        else:
            # store under /data/the_cauldron by default
            cache_dir = os.path.join(cache_root, "the_cauldron")
            os.makedirs(cache_dir, exist_ok=True)

        self.subset = subset
        self.split = split

        if subset in ("all", "*"):
            subset_names = get_dataset_config_names("HuggingFaceM4/the_cauldron")
        else:
            subset_names = [subset]
        subset_names = sorted(subset_names)

        self.datasets: List[tuple[str, Any]] = []
        total_len = 0
        for name in subset_names:
            ds = load_dataset(
                "HuggingFaceM4/the_cauldron",
                name,
                split=split,
                cache_dir=cache_dir,
            )
            if len(ds) == 0:
                continue
            self.datasets.append((name, ds))
            total_len += len(ds)

        self.length = total_len
        assert self.length > 0, f"The Cauldron subset(s) {subset_names} ({split}) are empty?"

        # Keep backward compatibility: expose single dataset as self.ds
        self.ds = self.datasets[0][1] if len(self.datasets) == 1 else None
        self.subset_names = [name for name, _ in self.datasets]

    def num_examples(self) -> int:
        return self.length

    def get_example(self, index: int) -> dict[str, Any]:
        assert 0 <= index < self.length, f"Index {index} out of range for Cauldron with {self.length} rows"

        # Map global index into the correct subset without materializing all rows.
        row = None
        for _, ds in self.datasets:
            if index < len(ds):
                row = ds[index]
                break
            index -= len(ds)
        assert row is not None, "Failed to locate row in Cauldron datasets"

        images = row.get("images") or []
        texts = row.get("texts") or []

        messages: List[dict[str, str]] = []
        sources: List[str] = []

        for turn in texts:
            user = str(turn.get("user") or "").strip()
            assistant = str(turn.get("assistant") or "").strip()
            src = str(turn.get("source") or "").strip()
            if src:
                sources.append(src)
            if not user or not assistant:
                # skip incomplete turns that would break user/assistant alternation
                continue
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})

        assert messages, "Cauldron conversation must have at least one user/assistant pair"

        conversation: dict[str, Any] = {
            "messages": messages,
            "images": images,
        }
        if sources:
            conversation["cauldron_sources"] = sources
        return conversation
