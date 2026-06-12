"""Generate figures and summary CSV files from experiment results."""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import (
    color_for,
    iter_system_metrics,
    load_results,
    savefig,
    setup_style,
)


def plot_frontier(
    cells, out_dir, xkey="avg_tokens", fname="frontier.png",
    xlabel="Avg tokens / question",
):
    setup_style()
    agg = defaultdict(lambda: {"x": [], "f1": []})
    for _, _, name, m, _ in iter_system_metrics(cells):
        agg[name]["x"].append(m.get(xkey, 0))
        agg[name]["f1"].append(m.get("f1", 0))
    if not agg:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for i, (name, d) in enumerate(sorted(agg.items())):
        x, y = float(np.mean(d["x"])), float(np.mean(d["f1"]))
        ax.scatter(
            x, y, s=140, color=color_for(name, i), edgecolor="white",
            linewidth=1.5, zorder=3, marker="*" if name == "msr_full" else "o"
        )
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 4),
                    fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("F1")
    ax.set_title(
        f"Efficiency frontier: F1 vs {xlabel}\n"
        "(upper-left = better: lower cost, higher accuracy)"
    )
    return savefig(fig, out_dir, fname)


def plot_steps_by_difficulty(cells, out_dir, fname="steps_by_difficulty.png"):
    setup_style()
    buckets = ["easy", "medium", "hard"]
    sys_vals = defaultdict(lambda: {b: [] for b in buckets})
    for _, _, name, m, _ in iter_system_metrics(cells):
        bd = m.get("by_difficulty", {})
        for b in buckets:
            if b in bd and bd[b].get("n", 0) > 0:
                sys_vals[name][b].append(bd[b]["avg_steps"])
    show = [s for s in ["msr_full", "graph_fixed", "naive", "lightrag"] if s in sys_vals]
    show += [s for s in sys_vals if s not in show][:2]
    if not show:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(len(buckets))
    w = 0.8 / max(len(show), 1)
    for i, name in enumerate(show):
        ys = [
            float(np.mean(sys_vals[name][b])) if sys_vals[name][b] else 0
            for b in buckets
        ]
        ax.bar(x + i * w, ys, width=w, label=name, color=color_for(name, i))
    ax.set_xticks(x + w * (len(show) - 1) / 2)
    ax.set_xticklabels([b.capitalize() for b in buckets])
    ax.set_ylabel("Avg retrieval steps")
    ax.set_title("Avg retrieval steps by difficulty")
    ax.legend()
    return savefig(fig, out_dir, fname)


def plot_component_ablation(cells, out_dir, fname="ablation_component.png"):
    setup_style()
    order = [
        "msr_full", "graph_fixed", "ecs_only", "pcs_only",
        "ags_only", "ecs_pcs", "no_routing",
    ]
    f1 = defaultdict(list)
    steps = defaultdict(list)
    for _, ds, name, m, _ in iter_system_metrics(cells):
        if name in order:
            f1[name].append(m.get("f1", 0))
            steps[name].append(m.get("avg_steps", 0))
    names = [n for n in order if n in f1]
    if len(names) < 2:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
    xs = np.arange(len(names))
    colors = [color_for(n, i) for i, n in enumerate(names)]
    a1.bar(xs, [np.mean(f1[n]) for n in names], color=colors)
    a1.set_xticks(xs)
    a1.set_xticklabels(names, rotation=30, ha="right")
    a1.set_ylabel("F1")
    a1.set_title("F1 by component")
    a2.bar(xs, [np.mean(steps[n]) for n in names], color=colors)
    a2.set_xticks(xs)
    a2.set_xticklabels(names, rotation=30, ha="right")
    a2.set_ylabel("Avg steps")
    a2.set_title("Avg steps by component")
    fig.suptitle("Ablation: contribution of ECS / PCS / AGS / routing",
                 fontweight="bold")
    return savefig(fig, out_dir, fname)


def plot_tau_sweep(cells, out_dir, fname="tau_sweep.png"):
    setup_style()
    pts = []
    for _, _, name, m, cfg in iter_system_metrics(cells):
        if name.startswith("tau_"):
            tau = cfg.get("scorer", {}).get("tau")
            if tau is None:
                try:
                    tau = float(name.split("_")[1])
                except Exception:
                    continue
            pts.append((tau, m.get("f1", 0), m.get("avg_steps", 0)))
    if len(pts) < 2:
        return None
    pts.sort()
    taus = [p[0] for p in pts]
    f1s = [p[1] for p in pts]
    st = [p[2] for p in pts]
    fig, ax1 = plt.subplots(figsize=(6.6, 4.6))
    ax2 = ax1.twinx()
    ax1.plot(taus, f1s, "o-", color="#2563eb", label="F1")
    ax2.plot(taus, st, "s--", color="#f59e0b", label="Avg steps")
    ax1.set_xlabel("tau (STOP threshold)")
    ax1.set_ylabel("F1", color="#2563eb")
    ax2.set_ylabel("Avg steps", color="#f59e0b")
    ax1.set_title("tau sensitivity: accuracy vs retrieval cost")
    return savefig(fig, out_dir, fname)


def plot_confgap_timeline(
    cells, out_dir, fname="confgap_timeline.png", max_lines: int = 6
):
    setup_style()
    series = []
    selected_model = None
    selected_dataset = None
    for c in cells:
        s = c.get("systems", {}).get("msr_full")
        if not s or "records" not in s:
            continue
        selected_model = str(c.get("model") or "unknown_model")
        selected_dataset = str(c.get("dataset") or "unknown_dataset")
        tau = s.get("config", {}).get("scorer", {}).get("tau", 0.30)
        for r in s["records"]:
            tr = r.get("traces") or []
            gaps = [t.get("conf_gap") for t in tr if t.get("conf_gap") is not None]
            if gaps:
                diff = str(r.get("difficulty", "?"))[:1]
                series.append((
                    f"{selected_dataset} | {r.get('qid')}({diff})",
                    gaps,
                    tau,
                    selected_model,
                    selected_dataset,
                ))
        if series:
            break
    if not series:
        return None
    series = series[:max_lines]
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    tau = series[0][2]
    model_label = (selected_model or series[0][3]).replace(
        "Qwen_Qwen2.5", "Qwen/Qwen2.5"
    )
    for i, (label, gaps, _, _, _) in enumerate(series):
        ax.plot(range(1, len(gaps) + 1), gaps, "o-", label=label,
                color=color_for("", i))
    ax.axhline(tau, color="#ef4444", linestyle="--", linewidth=1.5,
               label=f"tau={tau}")
    ax.set_xlabel("retrieval step")
    ax.set_ylabel("Confidence Gap")
    ax.set_title(f"ConfGap trajectory - {model_label}")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return savefig(fig, out_dir, fname)


def write_summary_table(cells, out_dir, fname="summary_table.csv"):
    rows = []
    baseline_by_cell = {}
    for c in cells:
        model, ds = c.get("model"), c.get("dataset")
        systems = c.get("systems", {})
        base = systems.get("graph_fixed") or systems.get("naive")
        if base and "metrics" in base:
            baseline_by_cell[(model, ds)] = base["metrics"]
    for model, ds, name, m, _ in iter_system_metrics(cells):
        base = baseline_by_cell.get((model, ds), {})
        base_steps = float(base.get("avg_steps") or 0)
        base_tokens = float(base.get("avg_tokens") or 0)
        step_savings = None
        token_savings = None
        if base_steps > 0:
            step_savings = round(
                100.0 * (base_steps - float(m.get("avg_steps", 0))) / base_steps, 2
            )
        if base_tokens > 0:
            token_savings = round(
                100.0 * (base_tokens - float(m.get("avg_tokens", 0))) / base_tokens, 2
            )
        rows.append({
            "model": model,
            "dataset": ds,
            "system": name,
            "n": m.get("n"),
            "EM": m.get("em"),
            "F1": m.get("f1"),
            "avg_steps": m.get("avg_steps"),
            "avg_tokens": m.get("avg_tokens"),
            "avg_latency_s": m.get("avg_latency_s"),
            "avg_calls": m.get("avg_calls"),
            "step_savings_vs_baseline_pct": step_savings,
            "token_savings_vs_baseline_pct": token_savings,
        })
    if not rows:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def make_all_plots(
    results_dir: str, out_dir: str, ablation_dir: Optional[str] = None
) -> list[str]:
    cells = load_results(results_dir)
    made = []
    for fn, kw in [
        (plot_frontier, dict(
            xkey="avg_tokens",
            fname="frontier_accuracy_vs_tokens.png",
            xlabel="Avg tokens / question",
        )),
        (plot_frontier, dict(
            xkey="avg_steps",
            fname="frontier_accuracy_vs_steps.png",
            xlabel="Avg retrieval steps",
        )),
        (plot_steps_by_difficulty, {}),
        (plot_confgap_timeline, {}),
        (write_summary_table, {}),
    ]:
        try:
            p = fn(cells, out_dir, **kw)
            if p:
                made.append(p)
        except Exception as e:
            print(f"[warn] {fn.__name__}: {e}")

    abl_cells = load_results(ablation_dir) if ablation_dir else cells
    for fn in (plot_component_ablation, plot_tau_sweep):
        try:
            p = fn(abl_cells, out_dir)
            if p:
                made.append(p)
        except Exception as e:
            print(f"[warn] {fn.__name__}: {e}")

    print(f"[plots] {len(made)} figure(s) -> {out_dir}")
    for p in made:
        print("  -", os.path.basename(p))
    return made
