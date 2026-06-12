#!/usr/bin/env python
"""Build and save a KG plus entity index for one dataset/model cell."""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from msr_graphrag.backends import build_backend
from msr_graphrag.data import load_corpus
from msr_graphrag.pipeline.msr_graphrag import MSRGraphRAG


def _tag(model: str) -> str:
    return re.sub(r"[^\w.-]+", "_", model).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backend", default="mock", choices=["mock", "hf", "vllm"])
    ap.add_argument("--model", default="mock")
    ap.add_argument("--kg-backend", default="native", choices=["native", "lightrag"])
    ap.add_argument("--kg-strategy", default="llm", choices=["llm", "heuristic"])
    ap.add_argument("--out", default="artifacts/kg")
    args = ap.parse_args()

    corpus = load_corpus(os.path.join(args.data, args.dataset, "corpus.json"))
    print(f"[corpus] {args.dataset}: {len(corpus.passages)} passages")

    backend = (
        build_backend(args.backend, model_name=args.model)
        if args.backend != "mock"
        else build_backend("mock")
    )

    working_dir = os.path.join(args.out, args.dataset, _tag(args.model), "lightrag_wd")
    rag = MSRGraphRAG(
        backend,
        kg_backend=args.kg_backend,
        kg_strategy=args.kg_strategy,
        working_dir=working_dir,
    )
    print(f"[index] backend={args.backend} kg={args.kg_backend}/{args.kg_strategy}")
    rag.index(corpus.documents(), verbose=True)
    print(f"[index] KG stats: {rag.store.stats()}")

    out_dir = os.path.join(args.out, args.dataset, _tag(args.model))
    rag.save_kg(out_dir)
    print(f"[save] -> {out_dir}/(kg.graphml, entity_index.pkl)")


if __name__ == "__main__":
    main()
