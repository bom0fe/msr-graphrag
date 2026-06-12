"""MockBackend — 결정론적 스텁 LLM.

목적
----
GPU/HF 다운로드 없이도 (1) 전체 파이프라인 스모크 테스트, (2) 데모 UI 구동,
(3) CI 검증을 가능케 한다. 실제 추론 품질은 없으나, 인터페이스·로짓 트레이스·
토큰 회계·엔티티 추출이 실제 백엔드와 동일하게 동작한다.

설계 원칙
--------
- 임베딩: 문자 n-gram 해시 → 고정 차원 벡터(L2 정규화). 동일 문자열은 동일 벡터.
- 엔티티 추출: 대문자 시작 토큰/따옴표 구절을 규칙 기반 추출.
- 생성: 컨텍스트에 정답 후보가 포함되면 그 후보를 답으로 내고 높은 margin,
        아니면 빈 답 + 낮은 margin → AGS 가 evidence 충분성에 반응하도록.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Optional

import numpy as np

from .base import LLMBackend, GenerationResult, simple_token_count
from ..scorer.uncertainty import LogitTrace

_DIM = 256
_STOP = set(
    "the a an of to in on at for and or but with from by as is are was were be been "
    "being this that these those it its their his her our your my we you they i he she "
    "what which who whom whose when where why how did do does done has have had will "
    "would can could should may might must".split()
)


def _hash_vec(text: str, dim: int = _DIM) -> np.ndarray:
    """문자 3-gram 해시 임베딩 (결정론적)."""
    v = np.zeros(dim, dtype=np.float32)
    s = f"  {text.lower()}  "
    for i in range(len(s) - 2):
        g = s[i : i + 3]
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class MockBackend(LLMBackend):
    def __init__(self, model_name: str = "mock", answer_oracle: Optional[dict] = None):
        super().__init__(model_name=model_name)
        # answer_oracle: {qid_or_question: gold_answer} — 데모에서 정답 노출 시뮬레이션용.
        self.answer_oracle = answer_oracle or {}

    @property
    def embed_dim(self) -> int:
        return _DIM

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, _DIM), dtype=np.float32)
        return np.stack([_hash_vec(t) for t in texts], axis=0)

    def extract_entities(self, query: str, role: str = "query_analyzer") -> List[str]:
        # 토큰 회계 반영 (실제 백엔드와 동일하게 호출 1회로 카운트)
        res = GenerationResult(
            text="", prompt_tokens=simple_token_count(query), completion_tokens=8
        )
        self.account.add(res, role=role)
        ents = []
        # 따옴표 구절
        ents += re.findall(r'"([^"]{2,40})"', query)
        # 대문자 시작 연속 토큰 (간이 고유명사)
        for m in re.finditer(r"\b([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)*)", query):
            ents.append(m.group(1))
        # 일반 명사 후보 (소문자, 불용어 제외, 길이>=4)
        for w in re.findall(r"\b[a-z][a-z'-]{3,}\b", query.lower()):
            if w not in _STOP:
                ents.append(w)
        # 중복 제거 (대소문자 보존, 최초 등장 우선)
        seen, out = set(), []
        for e in ents:
            k = e.lower()
            if k not in seen:
                seen.add(k)
                out.append(e.strip())
        return out[:12]

    def _gold_for_prompt(self, prompt: str) -> Optional[str]:
        """프롬프트에 포함된 질문을 oracle 키와 매칭 → 해당 질문의 gold 반환."""
        low = prompt.lower()
        for q, gold in self.answer_oracle.items():
            key = (q or "").lower().strip()
            if len(key) >= 8 and key[:48] in low:
                return gold
        return None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        with_logits: bool = False,
        role: str = "generic",
        system: Optional[str] = None,
    ) -> GenerationResult:
        ptoks = simple_token_count((system or "") + " " + prompt)
        low_prompt = prompt.lower()

        # 증거 근거 판단: 현재 질문의 gold 가 컨텍스트(증거)에 실제 등장할 때만 확신.
        # → 초기 step(브리지 엔티티 미검색)엔 미확신 → EXPAND, 도달 후 STOP (메커니즘 시연).
        answer = ""
        confident = False
        gold = self._gold_for_prompt(prompt)
        if gold is not None:
            if gold.lower() in low_prompt:
                answer, confident = gold, True
            else:
                answer, confident = "unknown", False          # 아직 증거 부족
        else:
            # oracle 미지정(실데이터/일반 질의): 증거 내 최빈 고유명사를 약한 추정으로.
            caps = re.findall(r"\b[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,3}", prompt)
            caps = [c for c in caps if c.lower() not in _STOP and len(c) > 2]
            if caps:
                from collections import Counter
                answer = Counter(caps).most_common(1)[0][0]
                confident = len(prompt) > 600
            else:
                answer = "unknown"

        ctoks = max(1, simple_token_count(answer))
        trace = self._mock_trace(answer, confident, with_logits)
        res = GenerationResult(
            text=answer,
            prompt_tokens=ptoks,
            completion_tokens=ctoks,
            logit_trace=trace,
        )
        self.account.add(res, role=role)
        return res

    @staticmethod
    def _mock_trace(answer: str, confident: bool, with_logits: bool) -> LogitTrace:
        if not with_logits:
            return LogitTrace(available=False)
        n = max(1, len(answer.split()))
        rng = np.random.default_rng(abs(hash(answer)) % (2**32))
        if confident:
            top1 = list(-0.05 - 0.10 * rng.random(n))  # 높은 확률(작은 음수)
            top2 = [t - (2.5 + rng.random()) for t in top1]  # 큰 갭
            ent = list(0.2 + 0.3 * rng.random(n))  # 낮은 엔트로피
        else:
            top1 = list(-1.2 - 0.8 * rng.random(n))
            top2 = [t - (0.2 + 0.3 * rng.random()) for t in top1]  # 작은 갭
            ent = list(1.8 + 1.2 * rng.random(n))  # 높은 엔트로피
        chosen = top1
        return LogitTrace(
            top1_logprobs=[float(x) for x in top1],
            top2_logprobs=[float(x) for x in top2],
            token_entropies=[float(x) for x in ent],
            chosen_logprobs=[float(x) for x in chosen],
            available=True,
        )
