import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.summarize_claims import summarize


def test_summarize_claims_relative_savings(tmp_path):
    cell = {
        "model": "m",
        "dataset": "d",
        "n_examples": 2,
        "systems": {
            "msr_full": {"metrics": {"f1": 0.8, "avg_steps": 2.0, "avg_tokens": 100.0}},
            "graph_fixed": {"metrics": {"f1": 0.75, "avg_steps": 4.0, "avg_tokens": 200.0}},
        },
    }
    path = tmp_path / "m__d.json"
    path.write_text(json.dumps(cell), encoding="utf-8")
    s = summarize(str(tmp_path))
    row = s["cells"]["m::d"]
    assert row["step_savings_pct"] == 50.0
    assert row["token_savings_pct"] == 50.0
    assert s["macro"]["msr_f1"] == 0.8
