"""Explorer<->reasoner data contract.

Every reasoner variant consumes the same `ReasonerExample` and returns a
`ReasonerOutput`, so the explorer and the reasoner can evolve independently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

Triple = list  # [subject, relation, object]


@dataclass
class Path:
    """One candidate reasoning path produced by the explorer."""

    triples: list[Triple]                 # [[s, r, o], ...]
    score: Optional[float] = None         # explorer confidence/rank, if any
    embedding: Optional[list[float]] = None  # precomputed path embedding (soft_prompt etc.)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"triples": self.triples}
        if self.score is not None:
            d["score"] = self.score
        if self.embedding is not None:
            d["embedding"] = self.embedding
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Path":
        return cls(
            triples=[list(t) for t in d.get("triples", [])],
            score=d.get("score"),
            embedding=d.get("embedding"),
        )


@dataclass
class ReasonerExample:
    """A single question + the explorer's candidate paths + gold supervision."""

    id: str
    question: str
    q_entity: list[str]
    paths: list[Path] = field(default_factory=list)
    ground_truth: dict[str, Any] = field(default_factory=dict)
    # ground_truth keys mirror data/EoG_process.py: correct_answers, answer_entities,
    # question_entities, graph_info, reasoning_path.

    @property
    def answers(self) -> list[str]:
        a = self.ground_truth.get("correct_answers", [])
        return [a] if isinstance(a, str) else list(a)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "q_entity": self.q_entity,
            "paths": [p.to_dict() for p in self.paths],
            "ground_truth": self.ground_truth,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReasonerExample":
        return cls(
            id=str(d.get("id", "")),
            question=d["question"],
            q_entity=list(d.get("q_entity", [])),
            paths=[Path.from_dict(p) for p in d.get("paths", [])],
            ground_truth=d.get("ground_truth", {}),
        )


@dataclass
class ReasonerOutput:
    """What a reasoner returns at inference time."""

    pred_answers: list[str]
    reasoning: Optional[str] = None
    raw: Optional[str] = None


def read_jsonl(path: str) -> Iterable[ReasonerExample]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield ReasonerExample.from_dict(json.loads(line))


def write_jsonl(path: str, examples: Iterable[ReasonerExample]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
