"""
Small Cauldron dataset:
https://huggingface.co/datasets/thomasgauthier/small-cauldron

Schema matches the full Cauldron:
- images: list of images
- texts: list of {user, assistant, source} turns
"""

import os
from typing import Any, List

from datasets import load_dataset

from tasks.common import Task


class SmallCauldron(Task):
    """
    Minimal wrapper around thomasgauthier/small-cauldron (single split).
    """

    def __init__(
        self,
        split: str = "train",
        cache_root: str = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if cache_root is None or cache_root == "":
            cache_dir = None
        else:
            cache_dir = os.path.join(cache_root, "small_cauldron")
            os.makedirs(cache_dir, exist_ok=True)

        self.split = split
        self.ds = load_dataset("thomasgauthier/small-cauldron", split=split, cache_dir=cache_dir)
        self.length = len(self.ds)
        assert self.length > 0, f"small-cauldron split '{split}' is empty?"

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
                continue
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})

        assert messages, "small-cauldron conversation must have at least one user/assistant pair"

        conversation: dict[str, Any] = {
            "messages": messages,
            "images": images,
        }
        if sources:
            conversation["cauldron_sources"] = sources
        return conversation
