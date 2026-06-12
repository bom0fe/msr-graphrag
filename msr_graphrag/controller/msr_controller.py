"""Stride-wise metacognitive retrieval controller."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from ..kg.entity_index import EntityIndex
from ..kg.graph_store import GraphStore
from ..retrieval.graph_retriever import GraphRetriever, RetrieverConfig
from ..retrieval.query_analyzer import QueryAnalyzer
from ..retrieval.state import RetrievalState
from ..scorer.metacognitive_scorer import MetacognitiveScorer, ScorerConfig

if TYPE_CHECKING:
    from ..backends.base import LLMBackend


@dataclass
class ControllerConfig:
    max_steps: int = 7
    min_steps: int = 1
    verbose: bool = False


class MSRController:
    def __init__(
        self,
        backend: "LLMBackend",
        store: GraphStore,
        entity_index: EntityIndex,
        scorer_config: Optional[ScorerConfig] = None,
        retriever_config: Optional[RetrieverConfig] = None,
        controller_config: Optional[ControllerConfig] = None,
        use_llm_entities: bool = True,
    ):
        self.backend = backend
        self.store = store
        self.analyzer = QueryAnalyzer(backend, use_llm=use_llm_entities)
        self.retriever = GraphRetriever(store, entity_index, retriever_config)
        self.scorer = MetacognitiveScorer(backend, scorer_config)
        self.cfg = controller_config or ControllerConfig()

    def run(self, query: str) -> RetrievalState:
        analysis = self.analyzer.analyze(query)
        state = RetrievalState(query=query, query_entities=analysis.entities)

        added = self.retriever.initialize(state)
        if self.cfg.verbose:
            print(f"[init] entities={analysis.entities} seeds={state.seeds} "
                  f"+{len(added)} nodes")

        for step in range(1, self.cfg.max_steps + 1):
            state.step = step
            sb = self.scorer.score(state, self.store, self.cfg.max_steps)
            sb.new_nodes = list(added)
            state.add_trace(sb)
            if self.cfg.verbose:
                print(f"[step {step}] {sb.rationale}")

            if sb.decision == "STOP" and step >= self.cfg.min_steps:
                state.stopped = True
                state.stop_reason = "confidence_sufficient"
                break

            added = self.retriever.expand(state, sb.expand_mode or "depth")
            if not added and step >= self.cfg.min_steps:
                state.stopped = True
                state.stop_reason = "no_expansion_possible"
                break
        else:
            state.stopped = True
            state.stop_reason = "max_steps_reached"

        return state
