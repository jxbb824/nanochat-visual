"""
VQAv2-small evaluation dataset, backed by:
https://huggingface.co/datasets/merve/vqav2-small

We use a simple generative evaluation:
- model sees the question and associated image
- answer is considered correct if all words of any ground-truth answer appear
  in the model's completion (order-insensitive), or if a judge_fn says so.
"""

import re
import string
from typing import Any, Iterable, List, Optional

from datasets import load_dataset

from tasks.common import Task


def _to_str_list(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, (int, float)):
        return [str(val)]
    if isinstance(val, (list, tuple)):
        out: List[str] = []
        for x in val:
            out.extend(_to_str_list(x))
        return out
    if isinstance(val, dict):
        # common VQAv2 format: {"answer": "...", ...}
        if "answer" in val:
            return _to_str_list(val["answer"])
    return []


def _dedup_preserve(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _tokenize(s: str) -> List[str]:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return [t for t in s.split() if t]


class VQAv2Small(Task):
    """
    Minimal wrapper over merve/vqav2-small.
    """

    def __init__(self, split: str = "validation", judge_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("merve/vqav2-small", split=split)
        self.length = len(self.ds)
        assert self.length > 0, f"merve/vqav2-small({split}) is empty?"
        self.split = split
        self.judge_fn = judge_fn

    @property
    def eval_type(self) -> str:
        return "generative"

    def num_examples(self) -> int:
        return self.length

    def _extract_answers(self, row: dict[str, Any]) -> List[str]:
        answers: List[str] = []
        # Common keys in VQAv2-style datasets
        for key in ["answers", "answer", "multiple_choice_answer"]:
            if key in row:
                answers.extend(_to_str_list(row[key]))
        return _dedup_preserve([a.strip() for a in answers if str(a).strip()])

    def get_example(self, index: int) -> dict[str, Any]:
        row: dict[str, Any] = self.ds[index]
        image = row["image"]
        question = str(row.get("question", "")).strip()
        answers = self._extract_answers(row)

        conversation = {
            "messages": [
                {"role": "user", "content": question},
            ],
            "vqav2_image": image,
            "vqav2_answers": answers,
            "vqav2_question": question,
        }
        return conversation

    def _matches(self, pred: str, answer: str) -> bool:
        if not answer:
            return False
        pred_norm = pred.lower()
        ans_norm = answer.lower()
        # exact substring match (lenient)
        if ans_norm in pred_norm:
            return True
        pred_tokens = set(_tokenize(pred))
        ans_tokens = _tokenize(answer)
        return bool(ans_tokens) and all(tok in pred_tokens for tok in ans_tokens)

    def evaluate(self, problem: dict[str, Any], completion: str, use_judge: bool = True) -> bool:
        answers: List[str] = problem.get("vqav2_answers", [])
        pred = (completion or "").strip()
        if not pred or not answers:
            return False

        for ans in answers:
            if self._matches(pred, ans):
                return True

        if self.judge_fn and use_judge:
            return bool(self.judge_fn(problem, completion))
        return False
