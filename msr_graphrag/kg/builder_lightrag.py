"""builder_lightrag — LightRAG 로 KG 구축.

LightRAG(HKUDS/LightRAG, EMNLP2025)를 backend(임의 HF/vLLM 모델)로 구동하여
엔티티/관계 그래프를 만들고, 그 산출물(``graph_chunk_entity_relation.graphml``)을
GraphStore 로 로드한다. 이렇게 하면 LightRAG 의 고품질 추출 그래프 위에서
MSR-GraphRAG 의 stride-wise 순회를 그대로 수행할 수 있다.

LightRAG 의 query 는 단일샷이므로, 본 프로젝트는 LightRAG 를 '그래프 빌더'로만
사용하고(KG 인덱싱), 검색 루프는 GraphRetriever/MSRController 가 담당한다.
(논문 포지셔닝: "LightRAG 위에 metacognitive gate 를 얹는다".)

LightRAG 미설치/실행 불가 환경에서는 NativeKGBuilder 를 쓰면 된다.
"""
from __future__ import annotations

import os
import asyncio
from typing import List, TYPE_CHECKING

from .graph_store import GraphStore

if TYPE_CHECKING:
    from ..backends.base import LLMBackend


def build_with_lightrag(
    backend: "LLMBackend",
    documents: List[str],
    working_dir: str = "./lightrag_workdir",
    verbose: bool = False,
) -> GraphStore:
    """documents → LightRAG 인덱싱 → GraphStore 반환.

    Parameters
    ----------
    backend : LLMBackend (llm_func_for_lightrag / embed_func_for_lightrag 제공)
    documents : 인덱싱할 원문 청크 리스트
    working_dir : LightRAG 산출물 디렉토리 (GraphML/벡터DB 저장)
    """
    try:
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc
        try:                                   # 신버전 초기화 헬퍼
            from lightrag.kg.shared_storage import initialize_pipeline_status
        except Exception:
            initialize_pipeline_status = None
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "LightRAG 가 필요합니다: pip install lightrag-hku\n"
            f"(원인: {e})  대안: kg_backend='native' 사용."
        )

    os.makedirs(working_dir, exist_ok=True)

    embedding_func = EmbeddingFunc(
        embedding_dim=backend.embed_dim,
        max_token_size=8192,
        func=backend.embed_func_for_lightrag(),
    )

    async def _run() -> None:
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=backend.llm_func_for_lightrag(),
            embedding_func=embedding_func,
        )
        # 신버전: storages/pipeline 초기화 필요
        if hasattr(rag, "initialize_storages"):
            await rag.initialize_storages()
        if initialize_pipeline_status is not None:
            await initialize_pipeline_status()

        for i, doc in enumerate(documents):
            if verbose:
                print(f"[lightrag] insert {i + 1}/{len(documents)}")
            res = rag.ainsert(doc) if hasattr(rag, "ainsert") else None
            if res is not None:
                await res
            else:
                rag.insert(doc)  # 동기 fallback

        if hasattr(rag, "finalize_storages"):
            await rag.finalize_storages()

    asyncio.run(_run())

    store = GraphStore.from_lightrag_workdir(working_dir)
    if verbose:
        print(f"[lightrag] KG loaded: {store.stats()}")
    return store
