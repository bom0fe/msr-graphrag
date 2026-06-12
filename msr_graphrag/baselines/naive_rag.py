"""NaiveRAG 베이스라인 — "항상 최대 검색".

MSR-GraphRAG 의 적응적 중단과 대비되는 비교군. 패시지(청크) 임베딩 검색을
고정 step 수만큼 반복 누적하며, 조기 중단 없이 항상 ``max_steps`` 를 소모한다.
효율성 지표(step·토큰)의 상한 레퍼런스 역할.

같은 코퍼스를 사용하되 그래프가 아닌 평면 청크 검색을 쓴다 → 그래프 구조 +
메타인지 게이트의 기여를 분리 측정.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np

from ..backends.base import TokenAccount
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # 순환 import 방지
    from ..backends.base import LLMBackend
from ..data.schema import Passage


def _l2(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


class PassageIndex:
    """코퍼스 패시지 임베딩 인덱스 (NaiveRAG 검색기)."""

    def __init__(self, passages: List[Passage], embed_fn):
        self.passages = passages
        self.embed_fn = embed_fn
        texts = [p.to_document()[:512] for p in passages]
        self.emb = _l2(np.asarray(embed_fn(texts), dtype=np.float32)) if passages else \
            np.zeros((0, 1), dtype=np.float32)

    def search(self, query: str, top_k: int, exclude: set) -> List[int]:
        if len(self.passages) == 0:
            return []
        q = _l2(np.asarray(self.embed_fn([query]), dtype=np.float32))
        sims = (self.emb @ q.T).ravel()
        order = np.argsort(-sims)
        out = []
        for i in order:
            if int(i) in exclude:
                continue
            out.append(int(i))
            if len(out) >= top_k:
                break
        return out


@dataclass
class NaiveConfig:
    top_k_per_step: int = 3
    max_steps: int = 7  # 항상 이만큼 (early stop 없음)
    max_context_passages: int = 21


@dataclass
class NaiveOutput:
    query: str
    answer: str
    n_steps: int
    token_usage: Dict[str, Any]
    n_passages: int = 0
    traces: List[Dict[str, Any]] = field(default_factory=list)

    def to_record(self) -> Dict[str, Any]:
        return {"query": self.query, "answer": self.answer, "n_steps": self.n_steps,
                "token_usage": self.token_usage, "n_passages": self.n_passages,
                "traces": self.traces, "stopped": True,
                "stop_reason": "fixed_max_steps", "final_nodes": self.n_passages}


_ANS_PROMPT = """Answer the question with a SHORT span (entity/name/number/date/yes-no)
using only the passages. Output only the answer.

Question: {query}

Passages:
{ctx}

Answer:"""


class NaiveRAG:
    def __init__(self, backend: LLMBackend, passage_index: PassageIndex,
                 config: Optional[NaiveConfig] = None):
        self.backend = backend
        self.index = passage_index
        self.cfg = config or NaiveConfig()

    def answer(self, query: str, reset_account: bool = True) -> NaiveOutput:
        if reset_account:
            self.backend.account = TokenAccount()
        selected: List[int] = []
        sel_set: set = set()
        traces = []
        for step in range(1, self.cfg.max_steps + 1):
            ids = self.index.search(query, self.cfg.top_k_per_step, sel_set)
            for i in ids:
                if len(selected) >= self.cfg.max_context_passages:
                    break
                selected.append(i)
                sel_set.add(i)
            traces.append({"step": step, "n_passages": len(selected),
                           "decision": "EXPAND" if step < self.cfg.max_steps else "STOP"})
            if len(selected) >= self.cfg.max_context_passages:
                # 컨텍스트 상한 도달해도 step 수는 명목상 max 까지 카운트 (항상 최대)
                pass
        ctx = "\n".join(f"- {self.index.passages[i].to_document()[:300]}"
                        for i in selected)
        res = self.backend.generate(_ANS_PROMPT.format(query=query, ctx=ctx),
                                    max_new_tokens=64, role="answer_gen")
        return NaiveOutput(query=query, answer=res.text.strip(),
                           n_steps=self.cfg.max_steps,
                           token_usage=self.backend.account.snapshot(),
                           n_passages=len(selected), traces=traces)
