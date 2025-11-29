"""
OCRBench evaluation dataset, backed by the official HuggingFace dataset:
https://huggingface.co/datasets/echo840/OCRBench

We use the default subset and the `test` split.

Columns:
- image: image (PIL.Image)
- question: string
- answer: list of strings (one or more acceptable transcripts)
- dataset, question_type: extra metadata (ignored here)
"""

from typing import Any, List

from datasets import load_dataset

from tasks.common import Task


class OCRBench(Task):
    """
    Minimal OCRBench wrapper over echo840/OCRBench (default subset).
    """

    def __init__(self, split: str = "test", **kwargs):
        super().__init__(**kwargs)
        assert split == "test", f"OCRBench only provides split='test', got {split}"
        self.ds = load_dataset("echo840/OCRBench", "default", split=split)
        self.length = len(self.ds)
        assert self.length > 0, "echo840/OCRBench(test) is empty?"
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
        answers = row.get("answer")

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
            "ocrbench_image": image,
            "ocrbench_answers": answers_list,
        }
        return conversation

    def evaluate(self, problem: dict[str, Any], completion: str) -> bool:
        """
        Evaluate a completion by checking if any ground-truth answer string
        appears in the model answer (case-insensitive substring match).
        """
        answers: List[str] = problem.get("ocrbench_answers", [])
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


