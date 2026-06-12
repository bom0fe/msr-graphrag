"""베이스라인 패키지.

- NaiveRAG          : 평면 청크 검색 + 항상 최대 step (early stop 없음).
- LightRAGBaseline  : 표준 LightRAG mix-mode 단일샷 (LightRAG 설치 시).
"""
from .naive_rag import NaiveRAG, PassageIndex, NaiveConfig, NaiveOutput

__all__ = ["NaiveRAG", "PassageIndex", "NaiveConfig", "NaiveOutput",
           "LightRAGBaseline", "LightRAGOutput"]


def __getattr__(name):  # 지연 로드 (LightRAG 의존성 회피)
    if name in ("LightRAGBaseline", "LightRAGOutput"):
        from .lightrag_baseline import LightRAGBaseline, LightRAGOutput
        return {"LightRAGBaseline": LightRAGBaseline,
                "LightRAGOutput": LightRAGOutput}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
