"""
MME evaluation dataset, backed by the official HuggingFace dataset:
https://huggingface.co/datasets/darkyarding/MME

Columns:
- question_id: string
- image: image (PIL.Image)
- question: string
- answer: string ("Yes" or "No")
- category: string (ignored here, could be used for per-category scores)
"""

from typing import Any

from datasets import load_dataset

from tasks.common import Task


class MME(Task):
    """
    Minimal MME benchmark wrapper over darkyarding/MME (test split).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("darkyarding/MME", split="test")
        self.length = len(self.ds)
        assert self.length > 0, "darkyarding/MME(test) is empty?"

    @property
    def eval_type(self):
        # generative: model produces a free-form "yes"/"no" answer
        return "generative"

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row: dict[str, Any] = self.ds[index]
        image = row["image"]
        question = row["question"]
        answer = row["answer"]  # "Yes" or "No"
        category = row.get("category", "")

        prompt = question.strip() + "\n\nAnswer with a single word: yes or no."

        conversation = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "mme_image": image,
            "mme_answer": answer,
            "mme_category": category,
        }
        return conversation

    def evaluate(self, problem, completion: str) -> bool:
        """
        Evaluate a completion against the ground truth yes/no answer.
        We scan the completion for the first occurrence of 'yes' or 'no'
        (case-insensitive) and compare to the ground truth.
        """
        gt = str(problem["mme_answer"]).strip().lower()
        completion = (completion or "").strip().lower()
        if gt not in {"yes", "no"}:
            return False
        if not completion:
            return False

        idx_yes = completion.find("yes")
        idx_no = completion.find("no")

        # normalize "no" vs "none"/"not" by requiring whole word-ish match
        def valid_match(text: str, idx: int, word: str) -> bool:
            if idx < 0:
                return False
            before = text[idx - 1] if idx - 1 >= 0 else " "
            after = text[idx + len(word)] if idx + len(word) < len(text) else " "
            return not before.isalpha() and not after.isalpha()

        if not valid_match(completion, idx_yes, "yes"):
            idx_yes = -1
        if not valid_match(completion, idx_no, "no"):
            idx_no = -1

        if idx_yes == -1 and idx_no == -1:
            return False

        # choose the earliest valid occurrence
        if idx_yes != -1 and (idx_no == -1 or idx_yes < idx_no):
            pred = "yes"
        else:
            pred = "no"

        return pred == gt


