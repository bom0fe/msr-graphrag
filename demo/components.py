"""demo/components — Streamlit 데모용 렌더링 컴포넌트.

제안서 Demo Scenes 1–4 를 구성하는 시각 요소:
  - confidence_dashboard : ECS/PCS/AGS 막대 + ConfGap 게이지 ("확신 미터")
  - confgap_timeline     : step 별 ConfGap 추이 + τ 임계선 + STOP 지점
  - subgraph_html        : PyVis 인터랙티브 서브그래프 (seed/신규 노드 강조)
  - trace_panel          : step 별 메타인지 결정(EXPAND/STOP)·근거 텍스트
  - comparison_table     : Naive vs MSR (정확도/step/토큰) 나란히

plotly/pyvis 의존. Streamlit 앱(app.py)에서 호출.
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional

import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# 확신 미터 (게이지 + 컴포넌트 막대)
# ---------------------------------------------------------------------------
def confgap_gauge(conf_gap: float, tau: float) -> go.Figure:
    """ConfGap 게이지. τ 미만(초록)=STOP 가능, 이상(빨강)=확장 필요."""
    confidence = max(0.0, 1.0 - conf_gap)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence, 3),
        number={"font": {"size": 40}},
        title={"text": f"Answer Confidence (1 - ConfGap)<br>"
                       f"<span style='font-size:12px'>STOP if ConfGap &lt; τ={tau}</span>"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#2563eb"},
            "steps": [
                {"range": [0, 1 - tau], "color": "#fee2e2"},   # 확장 영역
                {"range": [1 - tau, 1], "color": "#dcfce7"},   # 중단 영역
            ],
            "threshold": {
                "line": {"color": "#16a34a", "width": 4},
                "thickness": 0.85, "value": 1 - tau,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=70, b=10))
    return fig


def component_bars(ecs: float, pcs: float, ags: float) -> go.Figure:
    """ECS/PCS/AGS 컴포넌트 막대 (어떤 신호가 약한지 = 확장 방향 단서)."""
    names = ["ECS<br>(entity)", "PCS<br>(connectivity)", "AGS<br>(generability)"]
    vals = [ecs, pcs, ags]
    colors = ["#10b981", "#8b5cf6", "#ef4444"]
    fig = go.Figure(go.Bar(
        x=names, y=vals, marker_color=colors,
        text=[f"{v:.2f}" for v in vals], textposition="outside",
    ))
    fig.update_yaxes(range=[0, 1.05], title="score")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                      title="Confidence components (weakest → expansion target)")
    return fig


# ---------------------------------------------------------------------------
# ConfGap 타임라인
# ---------------------------------------------------------------------------
def confgap_timeline(traces: List[Dict[str, Any]], tau: float) -> go.Figure:
    steps = [t["step"] for t in traces]
    gaps = [t["conf_gap"] for t in traces]
    decisions = [t.get("decision", "") for t in traces]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=gaps, mode="lines+markers+text",
        line=dict(color="#2563eb", width=3),
        marker=dict(size=12,
                    color=["#16a34a" if d == "STOP" else "#2563eb"
                           for d in decisions]),
        text=[("STOP" if d == "STOP" else "") for d in decisions],
        textposition="top center", name="ConfGap",
    ))
    fig.add_hline(y=tau, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"τ = {tau}", annotation_position="top right")
    fig.update_layout(height=300, title="Confidence Gap per step",
                      xaxis_title="retrieval step", yaxis_title="ConfGap",
                      margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(dtick=1)
    return fig


# ---------------------------------------------------------------------------
# PyVis 서브그래프
# ---------------------------------------------------------------------------
def subgraph_html(store, selected_nodes: List[str], seeds: List[str],
                  new_nodes: Optional[List[str]] = None,
                  height: str = "460px") -> str:
    """선택 서브그래프를 PyVis 인터랙티브 HTML 로. seed/신규 노드 강조."""
    from pyvis.network import Network
    new_nodes = set(new_nodes or [])
    seeds = set(seeds or [])
    sub = store.induced_subgraph(list(selected_nodes)) \
        if hasattr(store, "induced_subgraph") else store.subgraph(selected_nodes)

    net = Network(height=height, width="100%", bgcolor="#ffffff",
                  font_color="#111827", directed=False)
    net.barnes_hut(gravity=-8000, spring_length=110)
    for n in sub.nodes():
        if n in seeds:
            color, size = "#2563eb", 28           # 시드
        elif n in new_nodes:
            color, size = "#f59e0b", 24           # 최신 확장
        else:
            color, size = "#9ca3af", 18
        title = (sub.nodes[n].get("description", "") or n)[:160]
        net.add_node(n, label=str(n)[:24], title=title, color=color, size=size)
    for u, v, d in sub.edges(data=True):
        net.add_edge(u, v, title=(d.get("description", "") or "")[:120],
                     value=float(d.get("weight", 1.0)))
    try:
        return net.generate_html(notebook=False)
    except TypeError:
        return net.generate_html()


# ---------------------------------------------------------------------------
# 트레이스 패널 / 비교표 (Streamlit 위젯에 넣을 행 데이터)
# ---------------------------------------------------------------------------
def trace_rows(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for t in traces:
        rows.append({
            "step": t["step"],
            "ECS": round(t["ecs"], 3), "PCS": round(t["pcs"], 3),
            "AGS": round(t["ags"], 3), "ConfGap": round(t["conf_gap"], 3),
            "decision": t.get("decision", ""),
            "mode": t.get("expand_mode", "") or "",
            "weakest": t.get("weakest", "") or "",
            "draft": (t.get("draft_answer") or "")[:40],
            "rationale": t.get("rationale", ""),
        })
    return rows


def comparison_rows(msr_out, naive_out) -> List[Dict[str, Any]]:
    def row(tag, out):
        tu = out.token_usage
        return {"system": tag, "answer": out.answer,
                "steps": getattr(out, "n_steps", "-"),
                "total_tokens": tu.get("total_tokens", "-"),
                "llm_calls": tu.get("n_calls", "-"),
                "stop_reason": getattr(out, "stop_reason", "-")}
    rows = [row("MSR-GraphRAG", msr_out)]
    if naive_out is not None:
        rows.append(row("NaiveRAG (always max)", naive_out))
    return rows
