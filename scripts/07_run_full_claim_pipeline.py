#!/usr/bin/env python
"""Run the full claim-oriented pipeline into a separate result directory.

This script is intentionally conservative: it keeps each stage explicit and
stores logs next to outputs so a long remote run can be audited after the fact.
It uses the improved code paths: direct HF dataset loaders, graph_fixed
baseline, answer cleanup, claim summaries, and plots.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        code = proc.wait()
    if code:
        raise SystemExit(f"command failed ({code}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="result_claim_full")
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki", "musique"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--backend", default="hf", choices=["mock", "hf", "vllm"])
    ap.add_argument("--kg-strategy", default="heuristic", choices=["heuristic", "llm"])
    ap.add_argument("--max-steps", type=int, default=7)
    ap.add_argument("--ags-signal", default="margin_entropy")
    args = ap.parse_args()

    out = ROOT / args.out
    data = out / "data_processed"
    logs = out / "logs"
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")

    run([
        sys.executable, "scripts/01_build_corpus.py",
        "--datasets", *args.datasets,
        "--n-samples", str(args.n_samples),
        "--out", str(data),
    ], logs / "01_build_corpus.log")

    for ds in args.datasets:
        run([
            sys.executable, "scripts/03_run_experiments.py",
            "--data", str(data),
            "--datasets", ds,
            "--backend", args.backend,
            "--models", args.model,
            "--kg-strategy", args.kg_strategy,
            "--max-steps", str(args.max_steps),
            "--ags-signal", args.ags_signal,
            "--out", str(out / "main"),
            "--keep-traces",
        ], logs / f"03_main_{ds}.log")

    run([
        sys.executable, "scripts/04_run_ablation.py",
        "--ablation", "all",
        "--data", str(data),
        "--datasets", *args.datasets,
        "--backend", "mock",
        "--models", "mock",
        "--kg-strategy", "heuristic",
        "--out", str(out / "ablation_mock"),
    ], logs / "04_ablation_mock.log")

    run([
        sys.executable, "scripts/05_make_plots.py",
        "--results", str(out / "main"),
        "--ablation", str(out / "ablation_mock"),
        "--out", str(out / "figures"),
    ], logs / "05_plots.log")

    run([
        sys.executable, "scripts/06_summarize_claims.py",
        "--results", str(out / "main"),
        "--out", str(out / "claim_summary.json"),
    ], logs / "06_claim_summary.log")

    run([sys.executable, "-m", "pytest", "tests", "-q"], logs / "pytest.log")


if __name__ == "__main__":
    main()
