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

from datasets import load_dataset

from tasks.common import Task


class Cauldron(Task):
    """
    Wrapper around HuggingFaceM4/the_cauldron.

    By default we load the "ai2d" subset and the "train" split.
    The full dataset is ~170 GB; we allow downloading all data to /data,
    but callers can override `subset` if they want other configs.
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

        self.ds = load_dataset(
            "HuggingFaceM4/the_cauldron",
            subset,
            split=split,
            cache_dir=cache_dir,
        )
        self.length = len(self.ds)
        assert self.length > 0, f"The Cauldron subset '{subset}' ({split}) is empty?"

    def num_examples(self) -> int:
        return self.length

    def get_example(self, index: int) -> dict[str, Any]:
        row = self.ds[index]
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

