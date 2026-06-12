"""Experiment matrix runner.

Each (dataset, model) cell builds one KG and shares it across all systems in the
cell. Records are stored per example and then aggregated into JSON files that
the analysis scripts can plot directly.

System groups:
- msr_full: full MSR-GraphRAG with ECS, PCS, AGS, and routing.
- graph_fixed: fixed graph traversal baseline.
- naive: non-graph passage retrieval baseline.
- lightrag: optional LightRAG baseline.
- ablations: ecs_only, pcs_only, ags_only, no_routing, tau sweep, weight grid.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..backends.base import LLMBackend
from ..baselines.naive_rag import NaiveConfig, NaiveRAG, PassageIndex
from ..controller.msr_controller import ControllerConfig
from ..data.schema import Corpus, QAExample
from ..pipeline.msr_graphrag import MSRGraphRAG
from ..retrieval.graph_retriever import RetrieverConfig
from ..scorer.metacognitive_scorer import ScorerConfig
from .metrics import aggregate, score_prediction


@dataclass
class SystemSpec:
    """Evaluation system or ablation configuration."""

    name: str
    kind: str = "msr"  # msr | naive | lightrag
    scorer: Optional[ScorerConfig] = None
    controller: Optional[ControllerConfig] = None
    retriever: Optional[RetrieverConfig] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def default_systems(
    tau: float = 0.30, max_steps: int = 7, ags_signal: str = "margin"
) -> List[SystemSpec]:
    """Return the standard main-system comparison set."""
    base_ctrl = ControllerConfig(max_steps=max_steps, min_steps=1)
    full = ScorerConfig(tau=tau, ags_signal=ags_signal)
    return [
        SystemSpec("msr_full", "msr", full, base_ctrl),
        SystemSpec(
            "graph_fixed",
            "msr",
            ScorerConfig(tau=-1.0, ags_signal=ags_signal, enable_routing=False),
            base_ctrl,
        ),
        SystemSpec("naive", "naive", extra={"max_steps": max_steps}),
        SystemSpec(
            "ecs_only",
            "msr",
            ScorerConfig(tau=tau, use_ecs=True, use_pcs=False, use_ags=False),
            base_ctrl,
        ),
        SystemSpec(
            "pcs_only",
            "msr",
            ScorerConfig(tau=tau, use_ecs=False, use_pcs=True, use_ags=False),
            base_ctrl,
        ),
        SystemSpec(
            "ags_only",
            "msr",
            ScorerConfig(
                tau=tau, use_ecs=False, use_pcs=False, use_ags=True,
                ags_signal=ags_signal
            ),
            base_ctrl,
        ),
        SystemSpec(
            "no_routing",
            "msr",
            ScorerConfig(tau=tau, ags_signal=ags_signal, enable_routing=False),
            base_ctrl,
        ),
    ]


def tau_sweep_systems(
    taus=(0.10, 0.20, 0.30, 0.40, 0.50),
    max_steps: int = 7,
    ags_signal: str = "margin",
) -> List[SystemSpec]:
    """Return full MSR systems that vary only the stop threshold."""
    ctrl = ControllerConfig(max_steps=max_steps, min_steps=1)
    return [
        SystemSpec(
            f"tau_{t:.2f}",
            "msr",
            ScorerConfig(tau=t, ags_signal=ags_signal),
            ctrl,
        )
        for t in taus
    ]


def weight_grid_systems(
    grid: Optional[List[Tuple[float, float, float]]] = None,
    tau: float = 0.30,
    max_steps: int = 7,
    ags_signal: str = "margin",
) -> List[SystemSpec]:
    """Return full MSR systems with alternative ECS/PCS/AGS weights."""
    grid = grid or [
        (0.45, 0.4, 0.15),
        (0.3, 0.3, 0.4),
        (0.5, 0.3, 0.2),
        (0.2, 0.2, 0.6),
        (0.4, 0.4, 0.2),
        (0.33, 0.33, 0.34),
    ]
    ctrl = ControllerConfig(max_steps=max_steps, min_steps=1)
    out = []
    for a, b, g in grid:
        out.append(
            SystemSpec(
                f"abg_{a}_{b}_{g}",
                "msr",
                ScorerConfig(
                    alpha=a, beta=b, gamma=g, tau=tau, ags_signal=ags_signal
                ),
                ctrl,
            )
        )
    return out


def _build_msr(
    backend: LLMBackend, spec: SystemSpec, store, entity_index, kg_backend: str
) -> MSRGraphRAG:
    rag = MSRGraphRAG(
        backend,
        kg_backend=kg_backend,
        scorer_config=spec.scorer or ScorerConfig(),
        controller_config=spec.controller or ControllerConfig(),
        retriever_config=spec.retriever or RetrieverConfig(),
    )
    rag.set_kg(store, entity_index)
    return rag


def make_answer_fn(
    backend: LLMBackend,
    spec: SystemSpec,
    *,
    store,
    entity_index,
    corpus: Corpus,
    kg_backend: str,
    working_dir: str,
) -> Callable[[str], Any]:
    """Build the answer callable for one system specification."""
    if spec.kind == "msr":
        rag = _build_msr(backend, spec, store, entity_index, kg_backend)
        return lambda q: rag.answer(q)
    if spec.kind == "naive":
        pidx = PassageIndex(corpus.passages, backend.embed)
        cfg = NaiveConfig(max_steps=spec.extra.get("max_steps", 7))
        naive = NaiveRAG(backend, pidx, cfg)
        return lambda q: naive.answer(q)
    if spec.kind == "lightrag":
        from ..baselines.lightrag_baseline import LightRAGBaseline

        lr = LightRAGBaseline(
            backend, working_dir, mode=spec.extra.get("mode", "mix")
        )
        return lambda q: lr.answer(q)
    raise ValueError(f"unknown system kind: {spec.kind}")


def evaluate_system(
    answer_fn: Callable[[str], Any],
    examples: List[QAExample],
    keep_traces: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate one system over all examples and return per-example records."""
    records: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples):
        t0 = time.time()
        out = answer_fn(ex.question)
        sc = score_prediction(out.answer, ex.all_answers)
        rec = {
            "qid": ex.qid,
            "em": sc["em"],
            "f1": sc["f1"],
            "n_steps": getattr(out, "n_steps", 1),
            "total_tokens": out.token_usage["total_tokens"],
            "prompt_tokens": out.token_usage.get("prompt_tokens", 0),
            "completion_tokens": out.token_usage.get("completion_tokens", 0),
            "n_calls": out.token_usage.get("n_calls", 0),
            "difficulty": ex.difficulty_bucket(),
            "num_hops": ex.num_hops,
            "pred": out.answer,
            "gold": ex.answer,
            "stop_reason": getattr(out, "stop_reason", None),
            "latency_s": round(time.time() - t0, 3),
        }
        if keep_traces:
            rec["traces"] = getattr(out, "traces", [])
        records.append(rec)
        if verbose and (i + 1) % 25 == 0:
            print(f"    [{i + 1}/{len(examples)}] em={sc['em']:.0f} "
                  f"steps={rec['n_steps']}")
    return records


class ExperimentRunner:
    """Build one KG per (dataset, model) cell and evaluate every system."""

    def __init__(
        self,
        out_dir: str = "./results",
        kg_backend: str = "native",
        kg_strategy: str = "llm",
        keep_traces: bool = False,
    ):
        self.out_dir = out_dir
        self.kg_backend = kg_backend
        self.kg_strategy = kg_strategy
        self.keep_traces = keep_traces
        os.makedirs(out_dir, exist_ok=True)

    def run_one_cell(
        self,
        backend: LLMBackend,
        model_tag: str,
        dataset_tag: str,
        examples: List[QAExample],
        corpus: Corpus,
        systems: List[SystemSpec],
        working_dir: str,
        prebuilt: Optional[Tuple[Any, Any]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run one (model, dataset) cell with a shared KG.

        prebuilt=(store, entity_index) skips graph rebuilding for every system.
        """
        if verbose:
            print(f"\n=== CELL model={model_tag} dataset={dataset_tag} "
                  f"({len(examples)} q, {len(corpus.passages)} passages) ===")

        if prebuilt is not None:
            store, entity_index = prebuilt
            if verbose:
                print(f"  [KG] reuse prebuilt: {store.stats()}")
        else:
            builder = MSRGraphRAG(
                backend,
                kg_backend=self.kg_backend,
                kg_strategy=self.kg_strategy,
                working_dir=working_dir,
            )
            if verbose:
                print("  [KG] building ...")
            builder.index(corpus.documents(), verbose=False)
            store, entity_index = builder.store, builder.entity_index
            if verbose:
                print(f"  [KG] {store.stats()}")

        cell: Dict[str, Any] = {
            "model": model_tag,
            "dataset": dataset_tag,
            "kg_stats": store.stats(),
            "n_examples": len(examples),
            "systems": {},
        }

        for spec in systems:
            if verbose:
                print(f"  [SYS] {spec.name} ...")
            try:
                fn = make_answer_fn(
                    backend,
                    spec,
                    store=store,
                    entity_index=entity_index,
                    corpus=corpus,
                    kg_backend=self.kg_backend,
                    working_dir=working_dir,
                )
                recs = evaluate_system(
                    fn, examples, keep_traces=self.keep_traces, verbose=verbose
                )
                cell["systems"][spec.name] = {
                    "kind": spec.kind,
                    "config": _spec_config_dict(spec),
                    "metrics": aggregate(recs),
                    "records": recs,
                }
            except Exception as e:
                cell["systems"][spec.name] = {"error": repr(e)}
                if verbose:
                    print(f"    [SKIP] {spec.name}: {e}")

        path = os.path.join(self.out_dir, f"{model_tag}__{dataset_tag}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cell, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"  [SAVE] {path}")
        return cell


def _spec_config_dict(spec: SystemSpec) -> Dict[str, Any]:
    d: Dict[str, Any] = {"kind": spec.kind}
    if spec.scorer is not None:
        d["scorer"] = {
            "alpha": spec.scorer.alpha,
            "beta": spec.scorer.beta,
            "gamma": spec.scorer.gamma,
            "tau": spec.scorer.tau,
            "ags_signal": spec.scorer.ags_signal,
            "use_ecs": spec.scorer.use_ecs,
            "use_pcs": spec.scorer.use_pcs,
            "use_ags": spec.scorer.use_ags,
            "enable_routing": spec.scorer.enable_routing,
            "abstain_penalty": spec.scorer.abstain_penalty,
        }
    if spec.controller is not None:
        d["controller"] = {
            "max_steps": spec.controller.max_steps,
            "min_steps": spec.controller.min_steps,
        }
    d.update(spec.extra)
    return d
