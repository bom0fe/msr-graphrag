"""GraphStore — KG 통합 표현.

LightRAG 가 생성하는 GraphML(``graph_chunk_entity_relation.graphml``)과 동일한
스키마를 사용한다. 따라서 (1) LightRAG 산출 그래프를 그대로 로드하거나,
(2) 내장 빌더로 만든 그래프를 동일 인터페이스로 순회할 수 있다.

스키마
------
node attr : entity_id(=name), entity_type, description, source_id
edge attr : description, keywords, weight, source_id

stride-wise 순회(깊이/폭 제어)는 ``GraphRetriever`` 가 이 클래스의 이웃/서브그래프
연산을 호출해 수행한다.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Iterable, Tuple

import networkx as nx


def normalize_entity(name: str) -> str:
    """엔티티 정규화 키 (대소문자/따옴표/공백 무시). LightRAG 도 대문자화 경향."""
    return " ".join(str(name).strip().strip('"').strip().split()).lower()


class GraphStore:
    def __init__(self, graph: Optional[nx.Graph] = None):
        # 무방향 그래프로 통일 (관계 방향은 edge attr 에 보존 가능, 순회는 무방향).
        self.G: nx.Graph = graph if graph is not None else nx.Graph()
        self._norm_index: Dict[str, str] = {}  # 정규화키 → 실제 node id
        self._rebuild_index()

    # --- 구성 ---------------------------------------------------------------
    def _rebuild_index(self) -> None:
        self._norm_index = {normalize_entity(n): n for n in self.G.nodes}

    def add_entity(self, name: str, entity_type: str = "UNKNOWN",
                   description: str = "", source_id: str = "") -> str:
        key = normalize_entity(name)
        if key in self._norm_index:
            nid = self._norm_index[key]
            # description 병합 (LightRAG 도 SEP 로 누적)
            if description:
                prev = self.G.nodes[nid].get("description", "")
                if description not in prev:
                    self.G.nodes[nid]["description"] = (
                        (prev + " | " + description).strip(" |") if prev else description
                    )
            return nid
        self.G.add_node(
            name, entity_id=name, entity_type=entity_type,
            description=description, source_id=source_id,
        )
        self._norm_index[key] = name
        return name

    def add_relation(self, src: str, dst: str, description: str = "",
                     keywords: str = "", weight: float = 1.0, source_id: str = "") -> None:
        s = self.add_entity(src)
        d = self.add_entity(dst)
        if self.G.has_edge(s, d):
            e = self.G[s][d]
            e["weight"] = float(e.get("weight", 1.0)) + float(weight)
            if description and description not in e.get("description", ""):
                e["description"] = (e.get("description", "") + " | " + description).strip(" |")
            if keywords and keywords not in e.get("keywords", ""):
                e["keywords"] = (e.get("keywords", "") + ", " + keywords).strip(", ")
        else:
            self.G.add_edge(s, d, description=description, keywords=keywords,
                            weight=float(weight), source_id=source_id)

    # --- 조회 ---------------------------------------------------------------
    def has(self, name: str) -> bool:
        return normalize_entity(name) in self._norm_index

    def resolve(self, name: str) -> Optional[str]:
        return self._norm_index.get(normalize_entity(name))

    def node_text(self, nid: str) -> str:
        d = self.G.nodes[nid]
        desc = d.get("description", "")
        et = d.get("entity_type", "")
        return f"{nid} ({et}): {desc}" if et and et != "UNKNOWN" else f"{nid}: {desc}"

    def edge_text(self, u: str, v: str) -> str:
        e = self.G[u][v]
        return f"{u} — {v}: {e.get('description','')}".strip()

    def neighbors(self, nid: str) -> List[str]:
        return list(self.G.neighbors(nid)) if nid in self.G else []

    def k_hop_nodes(self, seeds: Iterable[str], k: int = 1) -> Set[str]:
        """seeds 로부터 k-hop 이내 도달 노드 집합 (BFS)."""
        seeds = [s for s in seeds if s in self.G]
        frontier: Set[str] = set(seeds)
        visited: Set[str] = set(seeds)
        for _ in range(max(k, 0)):
            nxt: Set[str] = set()
            for n in frontier:
                for m in self.G.neighbors(n):
                    if m not in visited:
                        nxt.add(m)
            visited |= nxt
            frontier = nxt
            if not frontier:
                break
        return visited

    def induced_subgraph(self, nodes: Iterable[str]) -> nx.Graph:
        ns = [n for n in nodes if n in self.G]
        return self.G.subgraph(ns).copy()

    def degree(self, nid: str) -> int:
        return self.G.degree(nid) if nid in self.G else 0

    # --- 통계 ---------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        n = self.G.number_of_nodes()
        m = self.G.number_of_edges()
        deg = (2 * m / n) if n else 0.0
        return {"n_nodes": n, "n_edges": m, "avg_degree": deg,
                "n_components": nx.number_connected_components(self.G) if n else 0}

    # --- 입출력 -------------------------------------------------------------
    def save_graphml(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # GraphML 은 None 속성을 허용하지 않음 → 빈 문자열로 보정
        for _, d in self.G.nodes(data=True):
            for k, v in list(d.items()):
                if v is None:
                    d[k] = ""
        for _, _, d in self.G.edges(data=True):
            for k, v in list(d.items()):
                if v is None:
                    d[k] = ""
        nx.write_graphml(self.G, path)

    @classmethod
    def load_graphml(cls, path: str) -> "GraphStore":
        G = nx.read_graphml(path)
        # 무방향으로 통일
        if G.is_directed():
            G = G.to_undirected()
        return cls(G)

    @classmethod
    def from_lightrag_workdir(cls, working_dir: str) -> "GraphStore":
        """LightRAG working_dir 의 표준 GraphML 파일을 로드."""
        cand = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
        if not os.path.exists(cand):
            # 디렉토리 내 임의 graphml 탐색
            for f in os.listdir(working_dir):
                if f.endswith(".graphml"):
                    cand = os.path.join(working_dir, f)
                    break
        return cls.load_graphml(cand)
