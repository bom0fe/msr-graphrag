"""Normalized data schema for multi-hop QA datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Passage:
    """One passage in the retrieval corpus."""

    title: str
    text: str
    is_supporting: bool = False
    source_qid: Optional[str] = None

    def to_document(self) -> str:
        """Return a document string for KG indexing."""
        title = self.title.strip()
        body = self.text.strip()
        if title and not body.lower().startswith(title.lower()):
            return f"{title}. {body}"
        return body

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "text": self.text,
            "is_supporting": self.is_supporting,
            "source_qid": self.source_qid,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Passage":
        return cls(
            title=d["title"],
            text=d["text"],
            is_supporting=d.get("is_supporting", False),
            source_qid=d.get("source_qid"),
        )


@dataclass
class QAExample:
    """One normalized multi-hop QA example."""

    qid: str
    question: str
    answer: str
    answer_aliases: List[str] = field(default_factory=list)
    gold_titles: List[str] = field(default_factory=list)
    passages: List[Passage] = field(default_factory=list)
    num_hops: Optional[int] = None
    qtype: Optional[str] = None
    level: Optional[str] = None
    dataset: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_answers(self) -> List[str]:
        """Return answer aliases for max-over-gold EM/F1."""
        ans = [self.answer] + list(self.answer_aliases)
        seen, out = set(), []
        for a in ans:
            a = (a or "").strip()
            if a and a.lower() not in seen:
                seen.add(a.lower())
                out.append(a)
        return out

    def difficulty_bucket(self) -> str:
        """Return easy/medium/hard from hop count first, then dataset level."""
        if self.num_hops is not None:
            if self.num_hops <= 2:
                return "easy"
            if self.num_hops == 3:
                return "medium"
            return "hard"
        if self.level:
            return {"easy": "easy", "medium": "medium", "hard": "hard"}.get(
                self.level.lower(), "medium"
            )
        return "medium"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "answer": self.answer,
            "answer_aliases": list(self.answer_aliases),
            "gold_titles": list(self.gold_titles),
            "passages": [p.to_dict() for p in self.passages],
            "num_hops": self.num_hops,
            "qtype": self.qtype,
            "level": self.level,
            "dataset": self.dataset,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QAExample":
        return cls(
            qid=d["qid"],
            question=d["question"],
            answer=d["answer"],
            answer_aliases=d.get("answer_aliases", []),
            gold_titles=d.get("gold_titles", []),
            passages=[Passage.from_dict(p) for p in d.get("passages", [])],
            num_hops=d.get("num_hops"),
            qtype=d.get("qtype"),
            level=d.get("level"),
            dataset=d.get("dataset"),
            meta=d.get("meta", {}),
        )


@dataclass
class Corpus:
    """Deduplicated passage set used for KG indexing."""

    name: str
    passages: List[Passage] = field(default_factory=list)

    def documents(self) -> List[str]:
        return [p.to_document() for p in self.passages]

    def __len__(self) -> int:
        return len(self.passages)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passages": [p.to_dict() for p in self.passages]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Corpus":
        return cls(
            name=d.get("name", "corpus"),
            passages=[Passage.from_dict(p) for p in d.get("passages", [])],
        )
