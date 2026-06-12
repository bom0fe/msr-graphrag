"""Question analysis for entity-seeded graph retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..backends.base import LLMBackend


@dataclass
class QueryAnalysis:
    query: str
    entities: List[str] = field(default_factory=list)
    qtype: str = "unknown"


class QueryAnalyzer:
    def __init__(self, backend: LLMBackend, use_llm: bool = True):
        self.backend = backend
        self.use_llm = use_llm

    def analyze(self, query: str) -> QueryAnalysis:
        ents = self.backend.extract_entities(query) if self.use_llm else []
        if not ents:
            ents = self._heuristic_entities(query)
        ents = self._dedup(ents)
        return QueryAnalysis(query=query, entities=ents, qtype=self._qtype(query))

    @staticmethod
    def _heuristic_entities(query: str) -> List[str]:
        ents = re.findall(r'"([^"]{2,80})"', query)
        ents += re.findall(r"\b[A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+){0,5}", query)
        ents += re.findall(
            r"\b[A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+){0,4}\s+\([^)]+\)",
            query,
        )
        anchors = re.findall(
            r"\b(?:film|book|album|city|country|director|mother|father|spouse|company|"
            r"organization|university|team|award|performer|author|founder)\s+"
            r"([A-Z][^?.,;]{1,80})",
            query,
            flags=re.IGNORECASE,
        )
        ents.extend(anchors)
        return ents

    @staticmethod
    def _dedup(ents: List[str]) -> List[str]:
        stop = {
            "who", "what", "where", "when", "which", "whose", "how", "the", "a", "an",
            "film", "book", "album", "city", "country", "director", "mother", "father",
            "spouse", "company", "organization", "university", "team", "award",
        }
        seen, out = set(), []
        for e in ents:
            e = re.sub(r"\s+", " ", e.strip().strip('"').strip(" ?.,;:"))
            k = e.lower()
            if e and k not in seen and len(e) >= 2 and k not in stop:
                seen.add(k)
                out.append(e)
        return out[:12]

    @staticmethod
    def _qtype(query: str) -> str:
        q = query.lower()
        if any(w in q for w in [
            " or ", "which is", "compare", "older", "younger",
            "more", "larger", "first", "earlier", "later",
        ]):
            return "comparison"
        if q.startswith(("who", "what", "where", "when")):
            return "bridge"
        return "unknown"
