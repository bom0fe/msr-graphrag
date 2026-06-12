#!/usr/bin/env python
"""Run ablation experiments.

Supported ablation groups:
- component: ECS-only, PCS-only, AGS-only, ECS+PCS, no-routing.
- tau: stop-threshold sensitivity over {0.1, 0.2, 0.3, 0.4, 0.5}.
- weights: ECS/PCS/AGS weight grid.
- all: every group above.

Results are stored as results_ablation/<model>__<dataset>__<ablation>.json.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from msr_graphrag.backends import build_backend
from msr_graphrag.controller.msr_controller import ControllerConfig
from msr_graphrag.data import load_corpus, load_examples, load_toy_dataset
from msr_graphrag.eval.runner import (
    ExperimentRunner,
    SystemSpec,
    tau_sweep_systems,
    weight_grid_systems,
)
from msr_graphrag.scorer.metacognitive_scorer import ScorerConfig


def _tag(model: str) -> str:
    return re.sub(r"[^\w.-]+", "_", model).strip("_")


def _component_systems(tau, max_steps, ags_signal):
    """Return component ablations around the current default scorer weights."""
    ctrl = ControllerConfig(max_steps=max_steps, min_steps=1)
    return [
        SystemSpec(
            "msr_full",
            "msr",
            ScorerConfig(tau=tau, ags_signal=ags_signal),
            ctrl,
        ),
        SystemSpec(
            "graph_fixed",
            "msr",
            ScorerConfig(tau=-1.0, ags_signal=ags_signal, enable_routing=False),
            ctrl,
        ),
        SystemSpec(
            "ecs_only",
            "msr",
            ScorerConfig(tau=tau, use_ecs=True, use_pcs=False, use_ags=False),
            ctrl,
        ),
        SystemSpec(
            "pcs_only",
            "msr",
            ScorerConfig(tau=tau, use_ecs=False, use_pcs=True, use_ags=False),
            ctrl,
        ),
        SystemSpec(
            "ags_only",
            "msr",
            ScorerConfig(
                tau=tau, use_ecs=False, use_pcs=False, use_ags=True,
                ags_signal=ags_signal
            ),
            ctrl,
        ),
        SystemSpec(
            "ecs_pcs",
            "msr",
            ScorerConfig(tau=tau, use_ecs=True, use_pcs=True, use_ags=False),
            ctrl,
        ),
        SystemSpec(
            "no_routing",
            "msr",
            ScorerConfig(tau=tau, ags_signal=ags_signal, enable_routing=False),
            ctrl,
        ),
    ]


def _systems_for(ablation, tau, max_steps, ags_signal):
    if ablation == "component":
        return {"component": _component_systems(tau, max_steps, ags_signal)}
    if ablation == "tau":
        return {"tau": tau_sweep_systems(max_steps=max_steps, ags_signal=ags_signal)}
    if ablation == "weights":
        return {
            "weights": weight_grid_systems(
                tau=tau, max_steps=max_steps, ags_signal=ags_signal
            )
        }
    if ablation == "all":
        return {
            "component": _component_systems(tau, max_steps, ags_signal),
            "tau": tau_sweep_systems(max_steps=max_steps, ags_signal=ags_signal),
            "weights": weight_grid_systems(
                tau=tau, max_steps=max_steps, ags_signal=ags_signal
            ),
        }
    raise ValueError(ablation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--ablation", default="all", choices=["component", "tau", "weights", "all"]
    )
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa"])
    ap.add_argument("--backend", default="mock", choices=["mock", "hf", "vllm"])
    ap.add_argument("--models", nargs="+", default=["mock"])
    ap.add_argument("--kg-backend", default="native", choices=["native", "lightrag"])
    ap.add_argument("--kg-strategy", default="llm", choices=["llm", "heuristic"])
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--max-steps", type=int, default=7)
    ap.add_argument("--ags-signal", default="margin")
    ap.add_argument("--out", default="results_ablation")
    args = ap.parse_args()

    groups = _systems_for(args.ablation, args.tau, args.max_steps, args.ags_signal)

    if args.smoke:
        examples, corpus = load_toy_dataset()
        backend = build_backend(
            "mock", answer_oracle={e.question: e.answer for e in examples}
        )
        runner = ExperimentRunner(
            out_dir=args.out, kg_backend="native", kg_strategy="heuristic"
        )
        for gname, systems in groups.items():
            runner.run_one_cell(
                backend,
                "mock",
                f"toy__{gname}",
                examples,
                corpus,
                systems,
                working_dir=os.path.join(args.out, "wd"),
            )
        print(f"\n[smoke] ablation done -> {args.out}")
        return

    runner = ExperimentRunner(out_dir=args.out, kg_backend=args.kg_backend,
                              kg_strategy=args.kg_strategy)
    for model in args.models:
        backend = build_backend(args.backend, model_name=model)
        mtag = _tag(model)
        for ds in args.datasets:
            try:
                examples = load_examples(os.path.join(args.data, ds, "examples.json"))
                corpus = load_corpus(os.path.join(args.data, ds, "corpus.json"))
            except Exception as e:
                print(f"[skip] {ds}: {e}")
                continue
            for gname, systems in groups.items():
                wd = os.path.join(args.out, "wd", ds, mtag, gname)
                runner.run_one_cell(
                    backend, mtag, f"{ds}__{gname}", examples, corpus,
                    systems, working_dir=wd
                )
    print(f"\n[done] ablation -> {args.out}")


if __name__ == "__main__":
    main()
