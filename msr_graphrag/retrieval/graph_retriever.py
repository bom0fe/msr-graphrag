"""Stride-wise graph retriever and expansion operators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import networkx as nx
import numpy as np

from ..kg.entity_index import EntityIndex
from ..kg.graph_store import GraphStore, normalize_entity
from .state import RetrievalState


@dataclass
class RetrieverConfig:
    seed_top_k: int = 4
    seed_threshold: float = 0.40
    breadth_add: int = 2
    depth_expand_per_comp: int = 3
    pivot_neighbors: int = 4
    seed_ego_neighbors: int = 2
    max_subgraph_nodes: int = 200


class GraphRetriever:
    def __init__(
        self,
        store: GraphStore,
        entity_index: EntityIndex,
        config: Optional[RetrieverConfig] = None,
    ):
        self.store = store
        self.index = entity_index
        self.cfg = config or RetrieverConfig()

    def initialize(self, state: RetrievalState) -> List[str]:
        """Match query entities to seeds and include a small seed ego."""
        seeds = self.index.match_entities(
            state.query_entities, threshold=self.cfg.seed_threshold, max_per_entity=2
        )
        if len(seeds) < 1:
            extra = self.index.search([state.query], top_k=self.cfg.seed_top_k)
            seeds += [n for n, _ in extra]
        seeds = list(dict.fromkeys(seeds))[: self.cfg.seed_top_k]

        state.seeds = seeds
        state.query_entity_nodes = list(seeds)
        added: Set[str] = set(seeds)
        for s in seeds:
            nbrs = self._ranked_neighbors(s, exclude=added)
            for nb in nbrs[: self.cfg.seed_ego_neighbors]:
                added.add(nb)
        state.selected_nodes |= added
        return list(added)

    def expand(self, state: RetrievalState, mode: str) -> List[str]:
        if len(state.selected_nodes) >= self.cfg.max_subgraph_nodes:
            return []
        if mode == "breadth":
            new = self._expand_breadth(state)
        elif mode == "depth":
            new = self._expand_depth(state)
        elif mode in ("depth_pivot", "pivot"):
            new = self._expand_pivot(state)
        else:
            new = self._expand_depth(state)
        state.selected_nodes |= set(new)
        return new

    def _expand_breadth(self, state: RetrievalState) -> List[str]:
        sel_norm = {normalize_entity(n) for n in state.selected_nodes}
        missing = [
            e for e in state.query_entities if normalize_entity(e) not in sel_norm
        ]
        queries = missing if missing else [state.query]
        cands = self.index.search(
            queries, top_k=self.cfg.breadth_add * 3, exclude=state.selected_nodes
        )
        new: List[str] = []
        for nid, _ in cands:
            new.append(nid)
            nbrs = self._ranked_neighbors(
                nid, exclude=set(state.selected_nodes) | set(new)
            )
            if nbrs:
                new.append(nbrs[0])
            if len([x for x in new if x not in state.selected_nodes]) >= self.cfg.breadth_add:
                break
        return [n for n in new if n not in state.selected_nodes]

    def _expand_depth(self, state: RetrievalState) -> List[str]:
        sub = state.subgraph(self.store)
        comps = list(nx.connected_components(sub)) if sub.number_of_nodes() else []
        new: List[str] = []
        if len(comps) <= 1:
            boundary_src = list(sub.nodes) if sub.number_of_nodes() else list(state.selected_nodes)
            new += self._frontier_expand(
                boundary_src, state.selected_nodes, limit=self.cfg.depth_expand_per_comp
            )
        else:
            for comp in comps:
                got = self._frontier_expand(
                    list(comp),
                    set(state.selected_nodes) | set(new),
                    limit=self.cfg.depth_expand_per_comp,
                )
                new += got
        return [n for n in new if n not in state.selected_nodes]

    def _expand_pivot(self, state: RetrievalState) -> List[str]:
        sub = state.subgraph(self.store)
        if sub.number_of_nodes() == 0:
            return self._expand_depth(state)
        nodes = list(sub.nodes)
        sims = self._query_sims(state.query, nodes)
        degs = np.array([sub.degree(n) for n in nodes], dtype=np.float32)
        degs = degs / (degs.max() + 1e-6)
        score = 0.6 * sims + 0.4 * degs
        order = list(np.argsort(-score))
        new: List[str] = []
        for idx in order:
            pivot = nodes[int(idx)]
            nbrs = self._ranked_neighbors(
                pivot, exclude=set(state.selected_nodes) | set(new)
            )
            for nb in nbrs:
                new.append(nb)
                if len(new) >= self.cfg.pivot_neighbors:
                    return new
            if new:
                break
        return new

    def _frontier_expand(
        self, src_nodes: List[str], exclude: Set[str], limit: int
    ) -> List[str]:
        """Return highest-weight unseen neighbors from source nodes."""
        cand_scores = {}
        for s in src_nodes:
            if s not in self.store.G:
                continue
            for nb in self.store.G.neighbors(s):
                if nb in exclude:
                    continue
                w = float(self.store.G[s][nb].get("weight", 1.0))
                cand_scores[nb] = max(cand_scores.get(nb, 0.0), w)
        ranked = sorted(cand_scores.items(), key=lambda x: -x[1])
        return [n for n, _ in ranked[:limit]]

    def _ranked_neighbors(self, nid: str, exclude: Set[str]) -> List[str]:
        if nid not in self.store.G:
            return []
        nbrs = [
            (nb, float(self.store.G[nid][nb].get("weight", 1.0)))
            for nb in self.store.G.neighbors(nid)
            if nb not in exclude
        ]
        nbrs.sort(key=lambda x: -x[1])
        return [n for n, _ in nbrs]

    def _query_sims(self, query: str, nodes: List[str]) -> np.ndarray:
        if not nodes:
            return np.zeros(0, dtype=np.float32)
        pos = {n: i for i, n in enumerate(self.index.node_ids)}
        qv = self.index.embed_fn([query])
        qv = qv / (np.linalg.norm(qv, axis=-1, keepdims=True) + 1e-9)
        out = np.zeros(len(nodes), dtype=np.float32)
        for i, n in enumerate(nodes):
            if n in pos:
                out[i] = float(self.index.emb[pos[n]] @ qv[0])
        return out
