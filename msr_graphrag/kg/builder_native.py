"""내장 KG 빌더 (LightRAG 미사용 경로).

코퍼스 → 엔티티/관계 추출 → GraphStore. 두 가지 전략:
- ``llm``       : LLM 에 구조화 프롬프트로 (entity, type, desc)·(src, rel, dst) 추출.
                  실제 백엔드(Qwen/Llama)에서 사용. LightRAG 의 추출 프롬프트를 단순화.
- ``heuristic`` : 고유명사 co-occurrence 기반. MockBackend/데모/빠른 검증용.

산출 GraphStore 는 LightRAG GraphML 과 동일 스키마 → GraphRetriever 가 두 경로를
동일하게 순회.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # 순환 import 방지: 타입 힌트 전용
    from ..backends.base import LLMBackend
from .graph_store import GraphStore

_STOP = set(
    "the a an of to in on at for and or but with from by as is are was were be been the "
    "this that these those it its their his her our your we you they i he she what which "
    "who whom whose when where why how".split()
)

_EXTRACT_PROMPT = """You are an information extraction engine for building a knowledge graph.
From the passage, extract entities and the relations between them.

Return ONLY valid JSON (no markdown) with this schema:
{{"entities": [{{"name": "...", "type": "...", "description": "..."}}],
  "relations": [{{"source": "...", "target": "...", "relation": "...", "description": "..."}}]}}

Rules:
- Entities: named people, organizations, locations, works, dates, events, key concepts.
- Use canonical surface forms. Keep descriptions under 25 words.
- Only include relations where both endpoints are in entities.
- Prefer high-precision relations useful for multi-hop QA over generic co-occurrence.
- Do not invent facts that are not stated in the passage.
- Keep at most 12 entities and 20 relations.

Passage:
{passage}

JSON:"""


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _candidate_entities(text: str) -> List[str]:
    """대문자 시작 고유명사 구 추출 (휴리스틱)."""
    cands = re.findall(r"\b[A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+){0,4}", text)
    out = []
    for c in cands:
        c = c.strip().strip(".")
        if len(c) < 2:
            continue
        toks = c.split()
        if all(t.lower() in _STOP for t in toks):
            continue
        out.append(c)
    return out


class NativeKGBuilder:
    def __init__(self, backend: LLMBackend, strategy: str = "llm",
                 max_chars_per_doc: int = 4000):
        self.backend = backend
        self.strategy = strategy
        self.max_chars = max_chars_per_doc

    def build(self, documents: List[str], verbose: bool = False) -> GraphStore:
        store = GraphStore()
        for i, doc in enumerate(documents):
            doc = doc[: self.max_chars]
            if self.strategy == "llm":
                ok = self._extract_llm(doc, store, source_id=f"doc-{i}")
                if not ok:
                    self._extract_heuristic(doc, store, source_id=f"doc-{i}")
            else:
                self._extract_heuristic(doc, store, source_id=f"doc-{i}")
            if verbose and (i + 1) % 50 == 0:
                print(f"[KG] indexed {i+1}/{len(documents)} docs; {store.stats()}")
        store._rebuild_index()
        return store

    # --- LLM 전략 -----------------------------------------------------------
    def _extract_llm(self, doc: str, store: GraphStore, source_id: str) -> bool:
        prompt = _EXTRACT_PROMPT.format(passage=doc)
        res = self.backend.generate(prompt, max_new_tokens=1024, temperature=0.0,
                                    role="kg_index")
        data = _parse_extraction(res.text)
        if data is None:
            return False
        ents = data.get("entities", [])
        rels = data.get("relations", [])
        if not ents:
            return False
        for e in ents:
            name = str(e.get("name", "")).strip()
            if name:
                store.add_entity(name, entity_type=str(e.get("type", "UNKNOWN")),
                                 description=str(e.get("description", "")),
                                 source_id=source_id)
        for r in rels:
            s = str(r.get("source", "")).strip()
            t = str(r.get("target", "")).strip()
            if s and t and s != t:
                store.add_relation(s, t, description=str(r.get("description", "")),
                                   keywords=str(r.get("relation", "")), source_id=source_id)
        return True

    # --- 휴리스틱 전략 ------------------------------------------------------
    def _extract_heuristic(self, doc: str, store: GraphStore, source_id: str) -> None:
        """문장 단위 co-occurrence: 같은 문장에 등장한 고유명사 쌍을 연결."""
        for sent in _sentences(doc):
            ents = _candidate_entities(sent)
            # 빈도 상위만 노드화 (잡음 억제)
            uniq = list(dict.fromkeys(ents))
            for e in uniq:
                store.add_entity(e, entity_type="ENTITY",
                                 description=sent[:160], source_id=source_id)
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    store.add_relation(uniq[i], uniq[j],
                                       description=sent[:160],
                                       keywords="co-occurrence", source_id=source_id)


def _parse_extraction(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text).strip().rstrip("`").strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        if not isinstance(data, dict):
            return None
        data.setdefault("entities", [])
        data.setdefault("relations", [])
        if not isinstance(data["entities"], list) or not isinstance(data["relations"], list):
            return None
        return data
    except Exception:
        return None
