"""Metacognitive evidence sufficiency scorer.

Each retrieval stride asks whether the current evidence is enough to answer.

    ConfGap = 1 - (alpha * ECS + beta * PCS + gamma * AGS)

ECS measures question-entity coverage in the selected subgraph.
PCS measures subgraph path coherence through connected components.
AGS measures answer generability from a short model draft and logit signal.

STOP when ConfGap < tau. Otherwise EXPAND toward the weakest active signal:
ECS -> breadth, PCS -> depth, AGS -> depth_pivot.

If logits are unavailable, AGS receives zero weight and ECS/PCS are renormalized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple

import networkx as nx

from ..kg.graph_store import GraphStore, normalize_entity
from ..retrieval.state import RetrievalState, ScoreBreakdown
from .uncertainty import answer_generability as compute_ags

if TYPE_CHECKING:
    from ..backends.base import LLMBackend


@dataclass
class ScorerConfig:
    alpha: float = 0.45  # ECS weight.
    beta: float = 0.4  # PCS weight.
    gamma: float = 0.15  # AGS weight.
    tau: float = 0.30  # STOP threshold.
    ags_signal: str = "margin"  # margin | entropy | nll | margin_entropy
    ags_lambda: float = 0.5
    use_ags: bool = True  # False for AGS ablations.
    use_ecs: bool = True
    use_pcs: bool = True
    enable_routing: bool = True  # False fixes expansion to depth.
    draft_max_tokens: int = 48  # Short draft length for AGS.
    abstain_penalty: float = 0.35  # Downweight UNKNOWN-style drafts.


_DRAFT_PROMPT = """You are checking whether the evidence is enough to answer a
multi-hop question. Produce the most likely SHORT answer span using ONLY the evidence.
If no supported answer can be inferred, output exactly UNKNOWN. Do not explain.

Question: {query}

Evidence:
{evidence}

Answer:"""


class MetacognitiveScorer:
    def __init__(self, backend: LLMBackend, config: Optional[ScorerConfig] = None):
        self.backend = backend
        self.cfg = config or ScorerConfig()

    def compute_ecs(
        self, state: RetrievalState, store: GraphStore
    ) -> Tuple[float, List[str], List[str]]:
        """Return entity coverage plus covered and missing query entities."""
        q_ents = state.query_entities
        if not q_ents:
            return 1.0, [], []
        sel_norm = {normalize_entity(n) for n in state.selected_nodes}
        covered, missing = [], []
        for e in q_ents:
            ek = normalize_entity(e)
            hit = ek in sel_norm or any(ek in s or s in ek for s in sel_norm)
            (covered if hit else missing).append(e)
        ecs = len(covered) / max(len(q_ents), 1)
        return ecs, covered, missing

    def compute_pcs(self, state: RetrievalState, store: GraphStore) -> Tuple[float, int]:
        """Return subgraph connectivity score and component count."""
        sub = state.subgraph(store)
        n = sub.number_of_nodes()
        if n <= 1:
            return (1.0 if n == 1 else 0.0), max(n, 0)
        n_comp = nx.number_connected_components(sub)
        pcs = 1.0 - (n_comp - 1) / max(n, 1)
        return max(pcs, 0.0), n_comp

    def compute_ags(
        self, state: RetrievalState, store: GraphStore
    ) -> Tuple[float, str, bool]:
        """Return draft-answer confidence from the configured logit signal."""
        evidence = state.evidence_text(store)
        prompt = _DRAFT_PROMPT.format(query=state.query, evidence=evidence)
        res = self.backend.generate(
            prompt,
            max_new_tokens=self.cfg.draft_max_tokens,
            temperature=0.0,
            with_logits=True,
            role="ags_draft",
        )
        ags = compute_ags(
            res.logit_trace, signal=self.cfg.ags_signal, lam=self.cfg.ags_lambda
        )
        if _looks_abstained(res.text):
            ags *= self.cfg.abstain_penalty
        return ags, res.text.strip(), res.logit_trace.available

    def score(
        self, state: RetrievalState, store: GraphStore, max_steps: int
    ) -> ScoreBreakdown:
        ecs, covered, missing = self.compute_ecs(state, store)
        pcs, n_comp = self.compute_pcs(state, store)
        ags, draft, logit_ok = (
            self.compute_ags(state, store) if self.cfg.use_ags else (0.0, "", False)
        )

        # Zero inactive or unavailable signals, then renormalize remaining weights.
        w = {
            "ecs": self.cfg.alpha if self.cfg.use_ecs else 0.0,
            "pcs": self.cfg.beta if self.cfg.use_pcs else 0.0,
            "ags": self.cfg.gamma if (self.cfg.use_ags and logit_ok) else 0.0,
        }
        wsum = sum(w.values())
        if wsum <= 0:
            w = {"ecs": 1.0, "pcs": 0.0, "ags": 0.0}
            wsum = 1.0
        conf = (w["ecs"] * ecs + w["pcs"] * pcs + w["ags"] * ags) / wsum
        conf_gap = 1.0 - conf

        decision = "STOP" if conf_gap < self.cfg.tau else "EXPAND"
        sub = state.subgraph(store)

        weakest, expand_mode = None, None
        if decision == "EXPAND":
            weakest, expand_mode = self._route_expansion(ecs, pcs, ags, logit_ok)

        rationale = self._rationale(
            ecs, pcs, ags, conf_gap, decision, expand_mode, missing, logit_ok
        )
        return ScoreBreakdown(
            step=state.step,
            ecs=ecs,
            pcs=pcs,
            ags=ags,
            conf_gap=conf_gap,
            decision=decision,
            expand_mode=expand_mode,
            weakest=weakest,
            draft_answer=draft,
            rationale=rationale,
            n_nodes=sub.number_of_nodes(),
            n_edges=sub.number_of_edges(),
            n_components=n_comp,
            covered_entities=covered,
            missing_entities=missing,
        )

    def _route_expansion(
        self, ecs: float, pcs: float, ags: float, logit_ok: bool
    ) -> Tuple[str, str]:
        """Map the weakest active score to an expansion mode."""
        if not self.cfg.enable_routing:
            return "FIXED", "depth"
        cands: List[Tuple[str, float]] = []
        if self.cfg.use_ecs:
            cands.append(("ECS", ecs))
        if self.cfg.use_pcs:
            cands.append(("PCS", pcs))
        if self.cfg.use_ags and logit_ok:
            cands.append(("AGS", ags))
        if not cands:
            return "ECS", "breadth"
        weakest = min(cands, key=lambda x: x[1])[0]
        mode = {"ECS": "breadth", "PCS": "depth", "AGS": "depth_pivot"}[weakest]
        return weakest, mode

    @staticmethod
    def _rationale(ecs, pcs, ags, gap, decision, mode, missing, logit_ok) -> str:
        parts = [
            f"ECS={ecs:.2f}",
            f"PCS={pcs:.2f}",
            (f"AGS={ags:.2f}" if logit_ok else "AGS=n/a(no-logits)"),
            f"ConfGap={gap:.2f}",
        ]
        head = " | ".join(parts)
        if decision == "STOP":
            return f"{head} -> ConfGap<tau -> STOP (evidence sufficient)"
        miss = f" missing={missing[:3]}" if missing else ""
        return f"{head} -> EXPAND[{mode}]{miss}"


def _looks_abstained(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(
        x in t
        for x in [
            "unknown",
            "insufficient",
            "not enough",
            "cannot determine",
            "can't determine",
            "not provided",
            "no answer",
        ]
    )
