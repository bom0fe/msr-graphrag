"""Final answer generation from the selected evidence subgraph."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..kg.graph_store import GraphStore
from ..retrieval.state import RetrievalState

if TYPE_CHECKING:
    from ..backends.base import LLMBackend


_ANSWER_PROMPT = """You are a precise multi-hop QA system. Using the knowledge-graph
evidence below, answer the question with a SHORT span (an entity, name, number, date,
or yes/no). Output only the answer text, nothing else. If the evidence supports a
yes/no comparison, answer only yes or no.

Question: {query}

Evidence:
{evidence}

Answer:"""


@dataclass
class AnswerResult:
    answer: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class AnswerGenerator:
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def generate(self, state: RetrievalState, store: GraphStore) -> AnswerResult:
        evidence = state.evidence_text(store, max_nodes=80, max_edges=100)
        prompt = _ANSWER_PROMPT.format(query=state.query, evidence=evidence)
        res = self.backend.generate(
            prompt, max_new_tokens=64, temperature=0.0, role="answer_gen"
        )
        return AnswerResult(
            answer=_clean_answer(res.text),
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
        )


def _clean_answer(text: str) -> str:
    """Convert chatty model output into an eval-friendly short span."""
    out = (text or "").strip()
    out = re.sub(r"^```[a-zA-Z]*", "", out).strip().rstrip("`").strip()
    out = re.sub(r"^(answer|final answer)\s*:\s*", "", out, flags=re.I).strip()
    out = out.splitlines()[0].strip()
    m = re.match(r"^(yes|no)\b", out, flags=re.I)
    if m:
        return m.group(1).lower()
    out = re.split(
        r"\s+(?:because|since|as shown|according to)\b",
        out,
        maxsplit=1,
        flags=re.I,
    )[0]
    return out.strip(" \t\r\n\"'`.,;:")
