"""
ChartQA evaluation dataset, backed by the official HuggingFace dataset:
https://huggingface.co/datasets/HuggingFaceM4/ChartQA

Columns:
- image: image (PIL.Image)
- query: string
- label: list of strings (usually length 1)
- human_or_machine: string label (ignored here)
"""

from typing import Any, List

from datasets import load_dataset

from tasks.common import Task


class ChartQA(Task):
    """
    Minimal ChartQA benchmark wrapper over HuggingFaceM4/ChartQA.
    """

    def __init__(self, split: str = "test", **kwargs):
        super().__init__(**kwargs)
        assert split in {"train", "val", "test"}, f"Unsupported split: {split}"
        self.ds = load_dataset("HuggingFaceM4/ChartQA", split=split)
        self.length = len(self.ds)
        assert self.length > 0, "HuggingFaceM4/ChartQA split is empty?"
        self.split = split

    @property
    def eval_type(self) -> str:
        return "generative"

    def num_examples(self) -> int:
        return self.length

    def get_example(self, index: int) -> dict[str, Any]:
        row: dict[str, Any] = self.ds[index]
        image = row["image"]
        question = str(row["query"])
        labels = row.get("label", [])

        if isinstance(labels, (str, int, float)):
            labels_list: List[str] = [str(labels)]
        else:
            labels_list = [str(x) for x in labels]

        prompt = question.strip()

        conversation = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "chartqa_image": image,
            "chartqa_labels": labels_list,
        }
        return conversation

    def evaluate(self, problem: dict[str, Any], completion: str) -> bool:
        """
        Evaluate a completion by checking if any ground-truth label string
        appears in the model answer (case-insensitive substring match).
        """
        labels: List[str] = problem.get("chartqa_labels", [])
        answer = (completion or "").strip().lower()
        if not answer or not labels:
            return False

        for label in labels:
            s = str(label).strip().lower()
            if not s:
                continue
            if s in answer:
                return True
        return False


