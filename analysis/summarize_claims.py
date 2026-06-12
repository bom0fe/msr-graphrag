"""Create a compact claim-oriented summary from experiment JSON files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Iterable


def _iter_cells(results_dir: str) -> Iterable[Dict[str, Any]]:
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                yield json.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] skip {path}: {exc}")


def _pct_delta(new: float, base: float) -> float | None:
    if not base:
        return None
    return round(100.0 * (new - base) / base, 2)


def _pct_savings(new_cost: float, base_cost: float) -> float | None:
    if not base_cost:
        return None
    return round(100.0 * (base_cost - new_cost) / base_cost, 2)


def summarize(results_dir: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"cells": {}, "macro": {}}
    macro = {}
    for cell in _iter_cells(results_dir):
        model = cell.get("model")
        dataset = cell.get("dataset")
        key = f"{model}::{dataset}"
        systems = cell.get("systems", {})
        msr = systems.get("msr_full", {}).get("metrics", {})
        base = (
            systems.get("graph_fixed", {}).get("metrics")
            or systems.get("naive", {}).get("metrics", {})
        )
        row = {
            "model": model,
            "dataset": dataset,
            "n_examples": cell.get("n_examples"),
            "msr": msr,
            "baseline": base,
            "delta_f1_pct": _pct_delta(float(msr.get("f1", 0)), float(base.get("f1", 0))),
            "step_savings_pct": _pct_savings(float(msr.get("avg_steps", 0)),
                                             float(base.get("avg_steps", 0))),
            "token_savings_pct": _pct_savings(float(msr.get("avg_tokens", 0)),
                                              float(base.get("avg_tokens", 0))),
        }
        out["cells"][key] = row
        for k in ("f1", "avg_steps", "avg_tokens"):
            macro.setdefault(f"msr_{k}", []).append(float(msr.get(k, 0)))
            macro.setdefault(f"baseline_{k}", []).append(float(base.get(k, 0)))
    for k, vals in macro.items():
        out["macro"][k] = round(sum(vals) / len(vals), 4) if vals else 0
    if out["macro"]:
        out["macro"]["step_savings_pct"] = _pct_savings(
            out["macro"].get("msr_avg_steps", 0),
            out["macro"].get("baseline_avg_steps", 0),
        )
        out["macro"]["token_savings_pct"] = _pct_savings(
            out["macro"].get("msr_avg_tokens", 0),
            out["macro"].get("baseline_avg_tokens", 0),
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="claim_summary.json")
    args = ap.parse_args()
    summary = summarize(args.results)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[summary] {args.out}")


if __name__ == "__main__":
    main()
