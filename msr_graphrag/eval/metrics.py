"""Evaluation metrics: EM, F1, and efficiency aggregates."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List


def normalize_answer(s: str) -> str:
    """Apply standard SQuAD-style answer normalization."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


def exact_match(pred: str, golds: List[str]) -> float:
    p = normalize_answer(pred)
    return 1.0 if any(p == normalize_answer(g) for g in golds) else 0.0


def f1_score(pred: str, golds: List[str]) -> float:
    best = 0.0
    p_toks = normalize_answer(pred).split()
    for g in golds:
        g_toks = normalize_answer(g).split()
        if not p_toks or not g_toks:
            best = max(best, 1.0 if p_toks == g_toks else 0.0)
            continue
        common = Counter(p_toks) & Counter(g_toks)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        precision = n_same / len(p_toks)
        recall = n_same / len(g_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def score_prediction(pred: str, golds: List[str]) -> Dict[str, float]:
    return {"em": exact_match(pred, golds), "f1": f1_score(pred, golds)}


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-example records into dataset-level metrics."""
    if not records:
        return {}
    n = len(records)
    em = sum(r["em"] for r in records) / n
    f1 = sum(r["f1"] for r in records) / n
    steps = sum(r["n_steps"] for r in records) / n
    toks = sum(r["total_tokens"] for r in records) / n
    latency = sum(r.get("latency_s", 0.0) for r in records) / n
    calls = sum(r.get("n_calls", 0.0) for r in records) / n

    by_diff: Dict[str, Dict[str, float]] = {}
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        b = r.get("difficulty", "all")
        buckets.setdefault(b, []).append(r)
    for b, rs in buckets.items():
        m = len(rs)
        by_diff[b] = {
            "n": m,
            "em": sum(x["em"] for x in rs) / m,
            "f1": sum(x["f1"] for x in rs) / m,
            "avg_steps": sum(x["n_steps"] for x in rs) / m,
            "avg_tokens": sum(x["total_tokens"] for x in rs) / m,
            "avg_latency_s": sum(x.get("latency_s", 0.0) for x in rs) / m,
            "avg_calls": sum(x.get("n_calls", 0.0) for x in rs) / m,
        }

    return {
        "n": n,
        "em": round(em, 4),
        "f1": round(f1, 4),
        "avg_steps": round(steps, 4),
        "avg_tokens": round(toks, 2),
        "avg_latency_s": round(latency, 4),
        "avg_calls": round(calls, 4),
        "by_difficulty": by_diff,
    }
