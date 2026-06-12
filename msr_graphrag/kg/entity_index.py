"""EntityIndex — 엔티티 임베딩 인덱스.

질문 엔티티(텍스트) → KG 노드 매칭(시드 선정), 그리고 breadth 확장 시
'질문과 유사하나 미커버된 노드' 탐색에 사용한다. LightRAG 도 entity vdb 를
유지하지만, 순회 제어를 직접 하기 위해 경량 인덱스를 따로 둔다.

저장: 노드 id 리스트 + (N,d) 임베딩 행렬(L2 정규화). 검색: 코사인 = 내적.
"""
from __future__ import annotations

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np

from .graph_store import GraphStore, normalize_entity


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return x / n


class EntityIndex:
    def __init__(self, node_ids: List[str], embeddings: np.ndarray, embed_fn):
        self.node_ids = node_ids
        self.emb = _l2norm(embeddings.astype(np.float32)) if len(node_ids) else embeddings
        self.embed_fn = embed_fn  # Callable[[List[str]], np.ndarray]
        self._pos = {n: i for i, n in enumerate(node_ids)}

    @classmethod
    def build(cls, store: GraphStore, embed_fn) -> "EntityIndex":
        """노드 텍스트(이름+설명)를 임베딩해 인덱스 구성."""
        node_ids = list(store.G.nodes)
        if not node_ids:
            return cls([], np.zeros((0, 1), dtype=np.float32), embed_fn)
        texts = [store.node_text(n)[:512] for n in node_ids]
        emb = embed_fn(texts)
        return cls(node_ids, np.asarray(emb, dtype=np.float32), embed_fn)

    def search(self, query_texts: List[str], top_k: int = 5,
               exclude: Optional[set] = None) -> List[Tuple[str, float]]:
        """질의 텍스트들에 대해 max-pool 유사도 상위 노드 반환."""
        if not query_texts or len(self.node_ids) == 0:
            return []
        q = _l2norm(np.asarray(self.embed_fn(query_texts), dtype=np.float32))
        sims = self.emb @ q.T  # (N, Q)
        score = sims.max(axis=1)  # 각 노드의 최대 유사도
        order = np.argsort(-score)
        exclude = exclude or set()
        out: List[Tuple[str, float]] = []
        for i in order:
            nid = self.node_ids[i]
            if nid in exclude:
                continue
            out.append((nid, float(score[i])))
            if len(out) >= top_k:
                break
        return out

    def match_entities(self, entity_texts: List[str], threshold: float = 0.45,
                       max_per_entity: int = 2) -> List[str]:
        """질문 엔티티 각각을 가장 유사한 노드로 매핑(시드 선정).

        1) 정확/정규화 일치 우선, 2) 임베딩 유사도 threshold 이상 fallback.
        """
        seeds: List[str] = []
        seen = set()
        for et in entity_texts:
            key = normalize_entity(et)
            # 정규화 일치
            exact = None
            for nid in self.node_ids:
                if normalize_entity(nid) == key:
                    exact = nid
                    break
            if exact is not None:
                if exact not in seen:
                    seeds.append(exact); seen.add(exact)
                continue
            # 임베딩 유사도
            cands = self.search([et], top_k=max_per_entity)
            for nid, sc in cands:
                if sc >= threshold and nid not in seen:
                    seeds.append(nid); seen.add(nid)
        return seeds

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"node_ids": self.node_ids, "emb": self.emb}, f)

    @classmethod
    def load(cls, path: str, embed_fn) -> "EntityIndex":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["node_ids"], d["emb"], embed_fn)
