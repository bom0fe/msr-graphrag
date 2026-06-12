"""Unit tests for EM/F1 normalization and metric aggregation."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from msr_graphrag.eval.metrics import (
    aggregate,
    exact_match,
    f1_score,
    normalize_answer,
    score_prediction,
)


def test_normalize():
    assert normalize_answer("The  Cat.") == "cat"
    assert normalize_answer("a An THE") == ""
    assert normalize_answer("Bong Joon-ho!") == "bong joonho"


def test_exact_match():
    assert exact_match("Warsaw", ["warsaw"]) == 1.0
    assert exact_match("the Warsaw", ["Warsaw"]) == 1.0
    assert exact_match("Krakow", ["Warsaw"]) == 0.0
    assert exact_match("Warsaw, Poland", ["Warsaw", "Warsaw, Poland"]) == 1.0


def test_f1():
    f = f1_score("Robert Zemeckis Jr", ["Robert Zemeckis"])
    assert 0.0 < f < 1.0
    assert f1_score("Bong Joon-ho", ["Bong Joon-ho"]) == 1.0
    assert f1_score("apple", ["banana"]) == 0.0


def test_score_prediction_keys():
    sc = score_prediction("Warsaw", ["Warsaw"])
    assert set(sc.keys()) == {"em", "f1"} and sc["em"] == 1.0 and sc["f1"] == 1.0


def test_aggregate_by_difficulty():
    recs = [
        {"em": 1.0, "f1": 1.0, "n_steps": 1, "total_tokens": 100, "difficulty": "easy"},
        {"em": 0.0, "f1": 0.5, "n_steps": 3, "total_tokens": 300, "difficulty": "hard"},
        {"em": 1.0, "f1": 1.0, "n_steps": 4, "total_tokens": 400, "difficulty": "hard"},
    ]
    agg = aggregate(recs)
    assert agg["n"] == 3
    assert abs(agg["em"] - (2 / 3)) < 1e-3
    assert agg["by_difficulty"]["hard"]["n"] == 2
    assert abs(agg["by_difficulty"]["hard"]["avg_steps"] - 3.5) < 1e-6
    assert agg["by_difficulty"]["easy"]["avg_steps"] == 1


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[ok] {fn.__name__}")
    print(f"\n[PASS] metrics: {len(fns)} tests")


if __name__ == "__main__":
    _run_all()
