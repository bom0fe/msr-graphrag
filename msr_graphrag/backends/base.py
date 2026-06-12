"""LLM 백엔드 추상 인터페이스.

MSR-GraphRAG 의 모든 LLM 호출은 이 인터페이스를 거친다. 구현체:
- ``MockBackend``        : 결정론적 스텁. GPU/HF 없이 파이프라인·데모·테스트 실행.
- ``HFTransformersBackend`` : transformers. output_scores 로 로짓 직접 접근(AGS).
- ``VLLMBackend``        : vLLM. logprobs 로 고속 추론(AGS), 대규모 실험 권장.

토큰 사용량은 모든 호출에서 누적 집계된다(효율성 지표의 핵심).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

from ..scorer.uncertainty import LogitTrace


@dataclass
class GenerationResult:
    """LLM 생성 1회 결과 + 토큰 회계 + 로짓 트레이스."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    logit_trace: LogitTrace = field(default_factory=LogitTrace)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TokenAccount:
    """누적 토큰/호출 집계기."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_calls: int = 0
    by_role: Dict[str, int] = field(default_factory=dict)  # 호출 주체별 토큰

    def add(self, res: GenerationResult, role: str = "generic") -> None:
        self.prompt_tokens += res.prompt_tokens
        self.completion_tokens += res.completion_tokens
        self.n_calls += 1
        self.by_role[role] = self.by_role.get(role, 0) + res.total_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def snapshot(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "n_calls": self.n_calls,
            "by_role": dict(self.by_role),
        }


class LLMBackend(ABC):
    """LLM 백엔드 공통 인터페이스."""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        self.account = TokenAccount()

    # --- 생성 ---------------------------------------------------------------
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        with_logits: bool = False,
        role: str = "generic",
        system: Optional[str] = None,
    ) -> GenerationResult:
        """프롬프트 → 텍스트(+선택적 LogitTrace). 토큰은 self.account 에 누적."""

    # --- 임베딩 -------------------------------------------------------------
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """문자열 리스트 → (N, d) 임베딩 행렬 (L2 정규화 권장)."""

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    # --- 편의: 엔티티 추출 (LLM 기반, 백엔드 공통 기본 구현) ----------------
    def extract_entities(self, query: str, role: str = "query_analyzer") -> List[str]:
        """질문에서 핵심 엔티티 추출. JSON 리스트만 출력하도록 강제.

        구현체가 override 가능. 기본은 generate + 파싱.
        """
        prompt = (
            "Extract the key named entities and noun phrases from the question that "
            "are needed to find the answer. Return ONLY a JSON array of strings, "
            "no prose, no markdown.\n\n"
            f"Question: {query}\n\nEntities:"
        )
        res = self.generate(prompt, max_new_tokens=128, temperature=0.0, role=role)
        return _parse_json_string_list(res.text)

    def llm_func_for_lightrag(self):
        """LightRAG 의 llm_model_func 시그니처에 맞는 async 콜러블 반환.

        LightRAG 는 `async def f(prompt, system_prompt=None, history_messages=[], **kw)`
        형태를 기대한다.
        """
        backend = self

        async def _f(prompt, system_prompt=None, history_messages=None, **kwargs):
            res = backend.generate(
                prompt,
                max_new_tokens=kwargs.get("max_tokens", 1024),
                temperature=0.0,
                role="kg_index",
                system=system_prompt,
            )
            return res.text

        return _f

    def embed_func_for_lightrag(self):
        """LightRAG 의 embedding_func 에 맞는 async 콜러블 반환."""
        backend = self

        async def _f(texts):
            return backend.embed(list(texts))

        return _f


# ---------------------------------------------------------------------------
# 공용 파서 / 유틸
# ---------------------------------------------------------------------------
def _parse_json_string_list(text: str) -> List[str]:
    """LLM 출력에서 JSON 문자열 배열을 관대하게 파싱."""
    import json
    import re

    text = text.strip()
    # 코드펜스 제거
    text = re.sub(r"^```[a-zA-Z]*", "", text).strip().rstrip("`").strip()
    # 첫 대괄호 블록 추출
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    # 줄 단위 fallback
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789.) ").strip().strip('"').strip("'")
        if line and len(line) < 100:
            out.append(line)
    return out[:20]


def simple_token_count(text: str) -> int:
    """공백/구두점 근사 토큰 카운트 (백엔드 토크나이저 미가용 시 fallback)."""
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))
