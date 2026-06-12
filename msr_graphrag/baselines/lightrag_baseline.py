"""LightRAGBaseline — 표준 LightRAG mix-mode 비교군.

LightRAG 의 기본 검색(mix: local+global 혼합)을 단일샷으로 호출하는 비교군.
MSR-GraphRAG 와 동일한 KG/backend 를 쓰되, stride-wise 적응 중단 없이 한 번에
컨텍스트를 끌어와 답한다 → "그래프 RAG 자체" 대비 "메타인지 게이트"의 순효과를 분리.

n_steps 는 개념상 1(단일 검색)로 보고하되, only_need_context 로 받은 컨텍스트를
같은 AnswerGenerator 로 답하게 하여 생성 경로를 통일한다(공정 비교).
LightRAG 미설치 시 ImportError.
"""
from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from ..backends.base import TokenAccount

if TYPE_CHECKING:
    from ..backends.base import LLMBackend


@dataclass
class LightRAGOutput:
    query: str
    answer: str
    n_steps: int
    token_usage: Dict[str, Any]
    mode: str = "mix"
    context_chars: int = 0
    traces: List[Dict[str, Any]] = field(default_factory=list)

    def to_record(self) -> Dict[str, Any]:
        return {"query": self.query, "answer": self.answer, "n_steps": self.n_steps,
                "token_usage": self.token_usage, "mode": self.mode,
                "context_chars": self.context_chars, "traces": self.traces,
                "stopped": True, "stop_reason": "single_shot",
                "final_nodes": 0}


class LightRAGBaseline:
    def __init__(self, backend: "LLMBackend", working_dir: str,
                 mode: str = "mix", answer_max_tokens: int = 48):
        self.backend = backend
        self.working_dir = working_dir
        self.mode = mode
        self.answer_max_tokens = answer_max_tokens
        self._rag = None

    def _ensure_rag(self):
        if self._rag is not None:
            return
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc
        embedding_func = EmbeddingFunc(
            embedding_dim=self.backend.embed_dim, max_token_size=8192,
            func=self.backend.embed_func_for_lightrag(),
        )
        self._rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self.backend.llm_func_for_lightrag(),
            embedding_func=embedding_func,
        )

    def answer(self, query: str, reset_account: bool = True) -> LightRAGOutput:
        self._ensure_rag()
        from lightrag import QueryParam
        if reset_account:
            self.backend.account = TokenAccount()

        async def _q():
            rag = self._rag
            if hasattr(rag, "initialize_storages"):
                await rag.initialize_storages()
            # only_need_context=True → 검색 컨텍스트만 회수 (생성은 우리 경로로 통일)
            param = QueryParam(mode=self.mode, only_need_context=True)
            fn = rag.aquery if hasattr(rag, "aquery") else None
            if fn is not None:
                return await fn(query, param=param)
            return rag.query(query, param=param)

        context = asyncio.run(_q())
        context = context if isinstance(context, str) else str(context)

        prompt = (
            "Answer the question with a short factual span using ONLY the context.\n\n"
            f"Context:\n{context[:4000]}\n\nQuestion: {query}\nAnswer:"
        )
        res = self.backend.generate(
            prompt, max_new_tokens=self.answer_max_tokens, temperature=0.0,
            role="answer",
        )
        return LightRAGOutput(
            query=query, answer=res.text.strip(), n_steps=1,
            token_usage=self.backend.account.snapshot(),
            mode=self.mode, context_chars=len(context),
        )
