"""
DocVQA evaluation dataset, backed by the official HuggingFace dataset:
https://huggingface.co/datasets/lmms-lab/DocVQA

We use the DocVQA subset and, in practice, only the `test` split.

Columns (DocVQA subset):
- image: image (PIL.Image)
- question: string
- answers: list of strings (one or more acceptable answers)
- questionId, question_types, docId, ...: extra metadata (ignored here)
"""

from typing import Any, List

from datasets import load_dataset

from tasks.common import Task


class DocVQA(Task):
    """
    Minimal DocVQA benchmark wrapper over lmms-lab/DocVQA (DocVQA subset).
    """

    def __init__(self, split: str = "test", **kwargs):
        super().__init__(**kwargs)
        assert split in {"validation", "test"}, f"Unsupported split: {split}"
        # Explicitly select the DocVQA subset to avoid confusion with InfographicVQA
        self.ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split)
        self.length = len(self.ds)
        assert self.length > 0, "lmms-lab/DocVQA split is empty?"
        self.split = split

    @property
    def eval_type(self) -> str:
        return "generative"

    def num_examples(self) -> int:
        return self.length

    def get_example(self, index: int) -> dict[str, Any]:
        row: dict[str, Any] = self.ds[index]
        image = row["image"]
        question = str(row["question"])
        # Some rows may have answers=None; treat that as no ground-truth answers.
        answers = row.get("answers")
        if answers is None:
            answers_list: List[str] = []
        elif isinstance(answers, (str, int, float)):
            answers_list = [str(answers)]
        else:
            answers_list = [str(x) for x in answers if x is not None]

        prompt = question.strip()

        conversation = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "docvqa_image": image,
            "docvqa_answers": answers_list,
        }
        return conversation

    def evaluate(self, problem: dict[str, Any], completion: str) -> bool:
        """
        Evaluate a completion by checking if any ground-truth answer string
        appears in the model answer (case-insensitive substring match).
        """
        answers: List[str] = problem.get("docvqa_answers", [])
        pred = (completion or "").strip().lower()
        if not pred or not answers:
            return False

        for ans in answers:
            s = str(ans).strip().lower()
            if not s:
                continue
            if s in pred:
                return True
        return False


