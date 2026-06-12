#!/usr/bin/env python
"""Run the main system comparison matrix.

For each (model, dataset) cell, the runner builds or reuses one KG and evaluates
the standard system set. Results are written to <out>/<model>__<dataset>.json.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from msr_graphrag.backends import build_backend
from msr_graphrag.data import load_corpus, load_examples, load_toy_dataset
from msr_graphrag.eval.runner import ExperimentRunner, default_systems
from msr_graphrag.kg.entity_index import EntityIndex
from msr_graphrag.kg.graph_store import GraphStore


def _tag(model: str) -> str:
    return re.sub(r"[^\w.-]+", "_", model).strip("_")


def _maybe_reuse_kg(reuse_root, dataset, model_tag, backend):
    if not reuse_root:
        return None
    d = os.path.join(reuse_root, dataset, model_tag)
    g = os.path.join(d, "kg.graphml")
    p = os.path.join(d, "entity_index.pkl")
    if os.path.exists(g) and os.path.exists(p):
        store = GraphStore.load_graphml(g)
        idx = EntityIndex.load(p, backend.embed)
        return store, idx
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Run a fast MockBackend toy-data validation.")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki", "musique"])
    ap.add_argument("--backend", default="mock", choices=["mock", "hf", "vllm"])
    ap.add_argument("--models", nargs="+", default=["mock"])
    ap.add_argument("--kg-backend", default="native", choices=["native", "lightrag"])
    ap.add_argument("--kg-strategy", default="llm", choices=["llm", "heuristic"])
    ap.add_argument("--reuse-kg", default=None, help="Root directory from script 02.")
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--max-steps", type=int, default=7)
    ap.add_argument("--ags-signal", default="margin",
                    choices=["margin", "entropy", "nll", "margin_entropy"])
    ap.add_argument("--keep-traces", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    runner = ExperimentRunner(
        out_dir=args.out,
        kg_backend=args.kg_backend,
        kg_strategy=args.kg_strategy,
        keep_traces=args.keep_traces,
    )
    systems = default_systems(
        tau=args.tau, max_steps=args.max_steps, ags_signal=args.ags_signal
    )

    if args.smoke:
        examples, corpus = load_toy_dataset()
        backend = build_backend(
            "mock", answer_oracle={e.question: e.answer for e in examples}
        )
        runner.kg_strategy = "heuristic"
        runner.run_one_cell(
            backend,
            "mock",
            "toy",
            examples,
            corpus,
            systems,
            working_dir=os.path.join(args.out, "wd_toy"),
        )
        print(f"\n[smoke] done -> {args.out}")
        return

    for model in args.models:
        backend = build_backend(args.backend, model_name=model)
        mtag = _tag(model)
        for ds in args.datasets:
            try:
                examples = load_examples(os.path.join(args.data, ds, "examples.json"))
                corpus = load_corpus(os.path.join(args.data, ds, "corpus.json"))
            except Exception as e:
                print(f"[skip] {ds}: {e} (run scripts/01_build_corpus.py first)")
                continue
            prebuilt = _maybe_reuse_kg(args.reuse_kg, ds, mtag, backend)
            wd = os.path.join(args.out, "wd", ds, mtag)
            runner.run_one_cell(
                backend, mtag, ds, examples, corpus, systems,
                working_dir=wd, prebuilt=prebuilt
            )
    print(f"\n[done] results -> {args.out}")


if __name__ == "__main__":
    main()
