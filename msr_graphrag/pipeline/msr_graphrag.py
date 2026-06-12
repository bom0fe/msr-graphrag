"""End-to-end MSR-GraphRAG pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .answer_generator import AnswerGenerator

if TYPE_CHECKING:
    from ..backends.base import LLMBackend
    from ..controller.msr_controller import ControllerConfig, MSRController
    from ..kg.builder_native import NativeKGBuilder
    from ..kg.entity_index import EntityIndex
    from ..kg.graph_store import GraphStore
    from ..retrieval.graph_retriever import RetrieverConfig
    from ..retrieval.state import RetrievalState
    from ..scorer.metacognitive_scorer import ScorerConfig


@dataclass
class MSROutput:
    query: str
    answer: str
    n_steps: int
    stopped: bool
    stop_reason: str
    token_usage: Dict[str, Any]
    traces: List[Dict[str, Any]] = field(default_factory=list)
    final_nodes: int = 0
    state: Optional["RetrievalState"] = None

    def to_record(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "n_steps": self.n_steps,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "token_usage": self.token_usage,
            "final_nodes": self.final_nodes,
            "traces": self.traces,
        }


class MSRGraphRAG:
    def __init__(
        self,
        backend: "LLMBackend",
        kg_backend: str = "native",
        kg_strategy: str = "llm",
        scorer_config: Optional["ScorerConfig"] = None,
        retriever_config: Optional["RetrieverConfig"] = None,
        controller_config: Optional["ControllerConfig"] = None,
        use_llm_entities: bool = True,
        working_dir: str = "./msr_workdir",
    ):
        from ..controller.msr_controller import ControllerConfig
        from ..retrieval.graph_retriever import RetrieverConfig
        from ..scorer.metacognitive_scorer import ScorerConfig

        self.backend = backend
        self.kg_backend = kg_backend
        self.kg_strategy = kg_strategy
        self.scorer_config = scorer_config or ScorerConfig()
        self.retriever_config = retriever_config or RetrieverConfig()
        self.controller_config = controller_config or ControllerConfig()
        self.use_llm_entities = use_llm_entities
        self.working_dir = working_dir

        self.store: Optional["GraphStore"] = None
        self.entity_index: Optional["EntityIndex"] = None
        self.answer_gen = AnswerGenerator(backend)

    def index(self, documents: List[str], verbose: bool = False) -> "GraphStore":
        """Build the KG and entity index for the provided documents."""
        from ..kg.builder_native import NativeKGBuilder
        from ..kg.entity_index import EntityIndex

        if self.kg_backend == "lightrag":
            self.store = self._index_lightrag(documents, verbose)
        else:
            builder = NativeKGBuilder(self.backend, strategy=self.kg_strategy)
            self.store = builder.build(documents, verbose=verbose)
        self.entity_index = EntityIndex.build(self.store, self.backend.embed)
        return self.store

    def _index_lightrag(self, documents: List[str], verbose: bool) -> "GraphStore":
        from ..kg.builder_lightrag import build_with_lightrag

        return build_with_lightrag(self.backend, documents, self.working_dir, verbose)

    def set_kg(self, store: "GraphStore", index: Optional["EntityIndex"] = None) -> None:
        """Inject a prebuilt KG for experiment reuse."""
        from ..kg.entity_index import EntityIndex

        self.store = store
        self.entity_index = index or EntityIndex.build(store, self.backend.embed)

    def save_kg(self, path_dir: str) -> None:
        assert self.store is not None and self.entity_index is not None
        os.makedirs(path_dir, exist_ok=True)
        self.store.save_graphml(os.path.join(path_dir, "kg.graphml"))
        self.entity_index.save(os.path.join(path_dir, "entity_index.pkl"))

    def load_kg(self, path_dir: str) -> None:
        from ..kg.entity_index import EntityIndex
        from ..kg.graph_store import GraphStore

        self.store = GraphStore.load_graphml(os.path.join(path_dir, "kg.graphml"))
        self.entity_index = EntityIndex.load(
            os.path.join(path_dir, "entity_index.pkl"), self.backend.embed
        )

    def answer(self, query: str, reset_account: bool = True) -> MSROutput:
        """Run metacognitive retrieval and final answer generation."""
        assert self.store is not None and self.entity_index is not None, "call index() first"
        if reset_account:
            from ..backends.base import TokenAccount

            self.backend.account = TokenAccount()

        from ..controller.msr_controller import MSRController

        controller = MSRController(
            self.backend,
            self.store,
            self.entity_index,
            scorer_config=self.scorer_config,
            retriever_config=self.retriever_config,
            controller_config=self.controller_config,
            use_llm_entities=self.use_llm_entities,
        )
        state = controller.run(query)
        ans = self.answer_gen.generate(state, self.store)

        return MSROutput(
            query=query,
            answer=ans.answer,
            n_steps=state.step,
            stopped=state.stopped,
            stop_reason=state.stop_reason,
            token_usage=self.backend.account.snapshot(),
            traces=[t.to_dict() for t in state.traces],
            final_nodes=len(state.selected_nodes),
            state=state,
        )
