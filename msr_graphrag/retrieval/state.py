"""Accumulated state for stride-wise graph retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from ..kg.graph_store import GraphStore


@dataclass
class ScoreBreakdown:
    """One retrieval-stride score record."""

    step: int
    ecs: float
    pcs: float
    ags: float
    conf_gap: float
    decision: str
    expand_mode: Optional[str] = None
    weakest: Optional[str] = None
    draft_answer: str = ""
    rationale: str = ""
    n_nodes: int = 0
    n_edges: int = 0
    n_components: int = 0
    covered_entities: List[str] = field(default_factory=list)
    missing_entities: List[str] = field(default_factory=list)
    new_nodes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "ecs": round(self.ecs, 4),
            "pcs": round(self.pcs, 4),
            "ags": round(self.ags, 4),
            "conf_gap": round(self.conf_gap, 4),
            "decision": self.decision,
            "expand_mode": self.expand_mode,
            "weakest": self.weakest,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_components": self.n_components,
            "draft_answer": self.draft_answer,
            "covered_entities": self.covered_entities,
            "missing_entities": self.missing_entities,
            "new_nodes": self.new_nodes,
            "rationale": self.rationale,
        }


@dataclass
class RetrievalState:
    query: str
    query_entities: List[str] = field(default_factory=list)
    query_entity_nodes: List[str] = field(default_factory=list)
    seeds: List[str] = field(default_factory=list)
    selected_nodes: Set[str] = field(default_factory=set)
    step: int = 0
    traces: List[ScoreBreakdown] = field(default_factory=list)
    stopped: bool = False
    stop_reason: str = ""

    def subgraph(self, store: GraphStore) -> nx.Graph:
        return store.induced_subgraph(self.selected_nodes)

    def evidence_text(
        self, store: GraphStore, max_nodes: int = 60, max_edges: int = 80
    ) -> str:
        """Serialize the current subgraph as compact LLM evidence text."""
        sub = self.subgraph(store)
        nodes = sorted(sub.nodes, key=lambda n: -sub.degree(n))[:max_nodes]
        lines = ["### Entities"]
        for n in nodes:
            lines.append(f"- {store.node_text(n)[:300]}")
        lines.append("\n### Relations")
        cnt = 0
        for u, v in sub.edges:
            if cnt >= max_edges:
                break
            lines.append(f"- {store.edge_text(u, v)[:300]}")
            cnt += 1
        return "\n".join(lines)

    def add_trace(self, t: ScoreBreakdown) -> None:
        self.traces.append(t)

    def summary(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "n_steps": self.step,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "final_nodes": len(self.selected_nodes),
            "traces": [t.to_dict() for t in self.traces],
        }
