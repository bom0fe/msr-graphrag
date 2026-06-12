"""Shared plotting utilities for experiment results."""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTE = {
    "msr_full": "#2563eb",
    "naive": "#9ca3af",
    "lightrag": "#f59e0b",
    "ecs_only": "#10b981",
    "pcs_only": "#8b5cf6",
    "ags_only": "#ef4444",
    "ecs_pcs": "#14b8a6",
    "no_routing": "#db2777",
}
_FALLBACK = ["#0ea5e9", "#a3e635", "#fb7185", "#facc15", "#34d399", "#c084fc"]


def color_for(name: str, i: int = 0) -> str:
    return PALETTE.get(name, _FALLBACK[i % len(_FALLBACK)])


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "axes.unicode_minus": False,
    })


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all cell JSON files from a result directory."""
    cells = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(p, "r", encoding="utf-8") as f:
                c = json.load(f)
            c["_path"] = p
            cells.append(c)
        except Exception as e:
            print(f"[warn] load fail {p}: {e}")
    return cells


def iter_system_metrics(cells: List[Dict[str, Any]]):
    """Yield (model, dataset, system_name, metrics, config) tuples."""
    for c in cells:
        for name, s in c.get("systems", {}).items():
            if "error" in s or "metrics" not in s or not s["metrics"]:
                continue
            yield c.get("model"), c.get("dataset"), name, s["metrics"], s.get("config", {})


def savefig(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
