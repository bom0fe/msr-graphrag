#!/usr/bin/env python
"""Sample multi-hop QA datasets and build pooled corpora.

Outputs:
  <out>/<dataset>/examples.json
  <out>/<dataset>/corpus.json
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from msr_graphrag.data import (
    build_corpus,
    load_dataset_examples,
    save_corpus,
    save_examples,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki", "musique"])
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    for name in args.datasets:
        print(f"\n=== {name} (n={args.n_samples}) ===")
        try:
            examples = load_dataset_examples(
                name, n_samples=args.n_samples, seed=args.seed
            )
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue
        corpus = build_corpus(examples, name=name)
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        save_examples(examples, os.path.join(out_dir, "examples.json"))
        save_corpus(corpus, os.path.join(out_dir, "corpus.json"))
        dist = Counter(e.difficulty_bucket() for e in examples)
        print(f"  examples={len(examples)} passages={len(corpus.passages)} "
              f"difficulty={dict(dist)}")
        print(f"  saved -> {out_dir}")


if __name__ == "__main__":
    main()
