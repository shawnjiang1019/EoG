"""Answer metrics (Hit@1 / F1), shared by explorer and reasoner.

The functions mirror reward_func.py (evaluate_f1_score, evaluate_hits_at_1,
extract_answers_from_text, parse_model_output) so there is a single source of
truth. reward_func.py can be pointed here later to drop the duplication.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

from ..base.schema import ReasonerExample, ReasonerOutput


def parse_model_output(solution_str: str) -> dict[str, Any]:
    """Split a raw `<think>...</think><answer>...</answer>` string."""
    m = re.search(r"^(.*?)<answer>", solution_str, re.DOTALL | re.IGNORECASE)
    reasoning = m.group(1).strip() if m else ""
    am = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL | re.IGNORECASE)
    answer_text = am.group(1).strip() if am else ""
    try:
        final_answer = ast.literal_eval(answer_text)
    except (ValueError, SyntaxError):
        final_answer = answer_text
    return {"reasoning": reasoning, "final_answer": final_answer}


def extract_answers_from_text(answer_input: Any) -> list[str]:
    """Normalize a model answer (list / json string / tagged / plain) to list[str]."""
    if not answer_input:
        return []
    if isinstance(answer_input, list):
        return [str(a).strip() for a in answer_input if str(a).strip()]
    text = str(answer_input).strip().replace("<answer>", "").replace("</answer>", "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(a).strip() for a in parsed if str(a).strip()]
        except json.JSONDecodeError:
            inner = text[1:-1].strip()
            if inner:
                return [p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()]
    if "," in text:
        return [p.strip() for p in text.split(",") if p.strip()]
    return [text]


def hits_at_1(predicted: Any, correct: list[str]) -> float:
    preds = extract_answers_from_text(predicted)
    if not preds or not correct:
        return 0.0
    for p in preds:
        pl = p.lower().strip()
        for c in correct:
            cl = c.lower().strip()
            if pl == cl or cl in pl or pl in cl:
                return 1.0
    return 0.0


def f1_score(predicted: Any, correct: list[str]) -> float:
    preds = {p.lower().strip() for p in extract_answers_from_text(predicted)}
    gold = {c.lower().strip() for c in correct}
    if not preds and not gold:
        return 1.0
    if not preds or not gold:
        return 0.0
    inter = preds & gold
    if not inter:
        return 0.0
    precision = len(inter) / len(preds)
    recall = len(inter) / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_predictions(
    examples: list[ReasonerExample], outputs: list[ReasonerOutput]
) -> dict[str, float]:
    """Mean Hit@1 / F1 over aligned (example, output) pairs."""
    assert len(examples) == len(outputs), "examples/outputs length mismatch"
    n = len(examples) or 1
    h = sum(hits_at_1(o.pred_answers, ex.answers) for ex, o in zip(examples, outputs))
    f = sum(f1_score(o.pred_answers, ex.answers) for ex, o in zip(examples, outputs))
    return {"hit@1": h / n, "f1": f / n, "n": len(examples)}
