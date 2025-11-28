"""
MMStar evaluation dataset, backed by the official HuggingFace dataset:
https://huggingface.co/datasets/Lin-Chen/MMStar

We use the `val` split, which has 1.5K examples with columns:
- question: string, already including "Options: A: ..., B: ..., ..."
- image: image feature (PIL.Image)
- answer: string, letter like "A"/"B"/"C"/"D"
- category, l2_category, meta_info: extra metadata (ignored here)
"""

from typing import Any

from datasets import load_dataset

from tasks.common import Task


class MMStar(Task):
    """
    Minimal MMStar benchmark wrapper over Lin-Chen/MMStar (val split).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # small dataset (~1.5K rows), ok to load fully in memory
        self.ds = load_dataset("Lin-Chen/MMStar", split="val")
        self.length = len(self.ds)
        assert self.length > 0, "Lin-Chen/MMStar(val) is empty?"

    @property
    def eval_type(self):
        # we'll do generative evaluation: model generates an answer string
        return "generative"

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row: dict[str, Any] = self.ds[index]
        image = row["image"]
        # question already contains "Options: A: ..., B: ..., ..."
        q = row["question"]
        answer = row["answer"]  # letter like "A"

        prompt = q.strip() + "\n\nPlease answer with the option letter (A, B, C, or D) only."

        conversation = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            # keep raw metadata for evaluation
            "mmstar_image": image,
            "mmstar_answer": answer,
        }
        return conversation

    def evaluate(self, problem, completion: str) -> bool:
        """
        Evaluate a completion against the ground truth.
        For multiple choice, we accept either the correct letter (A/B/...)
        or the exact option text (case-insensitive, stripped).
        """
        gt = str(problem["mmstar_answer"]).strip()
        completion = (completion or "").strip()
        if not completion:
            return False

        # normalize to lower case for comparison
        gt_norm = gt.lower()
        comp_norm = completion.lower()

        # if ground truth is a single letter, only check the first non-space char
        if len(gt_norm) == 1 and gt_norm.isalpha():
            first_char = None
            for ch in comp_norm:
                if ch.isalpha():
                    first_char = ch
                    break
            return first_char == gt_norm

        # otherwise, check if gt text appears in completion
        return gt_norm in comp_norm


