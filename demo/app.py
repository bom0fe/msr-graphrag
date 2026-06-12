"""Interactive Streamlit demo for MSR-GraphRAG."""
from __future__ import annotations

import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from demo import components as C
from msr_graphrag.backends import build_backend
from msr_graphrag.baselines.naive_rag import NaiveConfig, NaiveRAG, PassageIndex
from msr_graphrag.controller.msr_controller import ControllerConfig
from msr_graphrag.data import load_corpus, load_examples, load_toy_dataset
from msr_graphrag.eval.metrics import score_prediction
from msr_graphrag.pipeline.msr_graphrag import MSRGraphRAG
from msr_graphrag.scorer.metacognitive_scorer import ScorerConfig


st.set_page_config(
    page_title="MSR-GraphRAG Demo",
    layout="wide",
    page_icon="MSR",
)


@st.cache_resource(show_spinner=True)
def load_pipeline(
    backend_kind: str,
    model: str,
    dataset: str,
    data_dir: str,
    kg_strategy: str,
):
    """Build or load the backend, corpus, graph, and passage index."""
    if dataset == "toy":
        examples, corpus = load_toy_dataset()
        oracle = {e.question: e.answer for e in examples}
        backend = (
            build_backend("mock", answer_oracle=oracle)
            if backend_kind == "mock"
            else build_backend(backend_kind, model_name=model)
        )
        strategy = "heuristic" if backend_kind == "mock" else kg_strategy
    else:
        examples = load_examples(os.path.join(data_dir, dataset, "examples.json"))
        corpus = load_corpus(os.path.join(data_dir, dataset, "corpus.json"))
        backend = build_backend(backend_kind, model_name=model)
        strategy = kg_strategy

    rag = MSRGraphRAG(backend, kg_backend="native", kg_strategy=strategy)
    rag.index(corpus.documents(), verbose=False)
    pidx = PassageIndex(corpus.passages, backend.embed)
    return backend, examples, corpus, rag, pidx


def make_rag_with_cfg(
    backend,
    store,
    index,
    tau,
    max_steps,
    ags_signal,
    alpha,
    beta,
    gamma,
    use_ecs,
    use_pcs,
    use_ags,
    routing,
):
    rag = MSRGraphRAG(
        backend,
        kg_backend="native",
        scorer_config=ScorerConfig(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            tau=tau,
            ags_signal=ags_signal,
            use_ecs=use_ecs,
            use_pcs=use_pcs,
            use_ags=use_ags,
            enable_routing=routing,
        ),
        controller_config=ControllerConfig(max_steps=max_steps, min_steps=1),
    )
    rag.set_kg(store, index)
    return rag


st.sidebar.title("MSR-GraphRAG")
st.sidebar.caption("Adaptive graph retrieval with a Confidence Gap controller")

backend_kind = st.sidebar.selectbox("Backend", ["mock", "hf", "vllm"], index=0)
model = st.sidebar.text_input(
    "Model",
    value="mock" if backend_kind == "mock" else "Qwen/Qwen2.5-7B-Instruct",
)
dataset = st.sidebar.selectbox(
    "Dataset", ["toy", "hotpotqa", "2wiki", "musique"], index=0
)
data_dir = st.sidebar.text_input("Processed data directory", value="data/processed")
kg_strategy = st.sidebar.selectbox("KG construction", ["llm", "heuristic"], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Confidence Gap Gate")
tau = st.sidebar.slider("Tau: stop threshold", 0.05, 0.60, 0.30, 0.05)
max_steps = st.sidebar.slider("Maximum retrieval steps", 1, 12, 7, 1)
ags_signal = st.sidebar.selectbox(
    "AGS confidence signal", ["margin", "entropy", "nll", "margin_entropy"], index=3
)

with st.sidebar.expander("Weights and ablations"):
    alpha = st.slider("Alpha: ECS weight", 0.0, 1.0, 0.45, 0.05)
    beta = st.slider("Beta: PCS weight", 0.0, 1.0, 0.40, 0.05)
    gamma = st.slider("Gamma: AGS weight", 0.0, 1.0, 0.15, 0.05)
    use_ecs = st.checkbox("Enable ECS", True)
    use_pcs = st.checkbox("Enable PCS", True)
    use_ags = st.checkbox("Enable AGS", True)
    routing = st.checkbox("Route expansion by weakest signal", True)

animate = st.sidebar.checkbox("Animate retrieval steps", True)
step_delay = st.sidebar.slider("Animation delay in seconds", 0.0, 1.5, 0.6, 0.1)


st.title("Metacognitive Self-Regulating GraphRAG")
st.markdown(
    """
This demo shows how MSR-GraphRAG controls graph retrieval with a Confidence Gap.
At each step, the system scores the current evidence using entity coverage
(ECS), graph coherence (PCS), and answerability (AGS). If the Confidence Gap is
below the threshold, retrieval stops. Otherwise, the graph expands toward the
weakest signal.
"""
)

st.markdown(
    """
**Recording guide:** choose a question, run MSR-GraphRAG, then show the
confidence gauge, component bars, ConfGap timeline, evidence graph, trace table,
and Naive RAG comparison.
"""
)

try:
    backend, examples, corpus, base_rag, pidx = load_pipeline(
        backend_kind, model, dataset, data_dir, kg_strategy
    )
except Exception as e:
    st.error(
        f"Pipeline load failed: {e}\n\n"
        "For non-toy datasets, prepare data first with `scripts/01_build_corpus.py`."
    )
    st.stop()

st.success(
    f"Knowledge graph ready. {base_rag.store.stats()} | "
    f"Corpus: {len(corpus.passages)} passages"
)

ex_map = {f"[{e.difficulty_bucket()}] {e.question}": e for e in examples}
col_q1, col_q2 = st.columns([3, 2])
with col_q1:
    picked = st.selectbox("Example question", list(ex_map.keys()))
with col_q2:
    custom = st.text_input("Custom question", "")

question = custom.strip() or ex_map[picked].question
gold = None if custom.strip() else ex_map[picked].all_answers

run = st.button("Run MSR-GraphRAG", type="primary")

if run:
    rag = make_rag_with_cfg(
        backend,
        base_rag.store,
        base_rag.entity_index,
        tau,
        max_steps,
        ags_signal,
        alpha,
        beta,
        gamma,
        use_ecs,
        use_pcs,
        use_ags,
        routing,
    )
    out = rag.answer(question)
    traces = out.traces

    st.markdown("## Adaptive Retrieval Trace")
    ph_gauge, ph_bars = st.columns(2)
    ph_time = st.empty()
    ph_status = st.empty()
    ph_graph = st.empty()

    shown = range(1, len(traces) + 1) if animate else [len(traces)]
    for k in shown:
        t = traces[k - 1]
        with ph_gauge:
            st.plotly_chart(
                C.confgap_gauge(t["conf_gap"], tau),
                use_container_width=True,
                key=f"g{k}",
            )
        with ph_bars:
            st.plotly_chart(
                C.component_bars(t["ecs"], t["pcs"], t["ags"]),
                use_container_width=True,
                key=f"b{k}",
            )
        ph_time.plotly_chart(
            C.confgap_timeline(traces[:k], tau),
            use_container_width=True,
            key=f"t{k}",
        )

        if t.get("decision") == "STOP":
            ph_status.success(
                f"Enough evidence at step {t['step']}: "
                f"ConfGap={t['conf_gap']:.3f} < tau={tau}. Retrieval stops."
            )
        else:
            ph_status.info(
                f"Step {t['step']}: ConfGap={t['conf_gap']:.3f} >= tau={tau}. "
                f"Expand with mode `{t.get('expand_mode')}` "
                f"because the weakest signal is `{t.get('weakest')}`."
            )

        selected = []
        for tt in traces[:k]:
            selected += tt.get("new_nodes", [])
        try:
            html = C.subgraph_html(
                base_rag.store,
                out.state.selected_nodes if hasattr(out, "state") else selected,
                seeds=out.state.seeds if hasattr(out, "state") else [],
                new_nodes=t.get("new_nodes", []),
            )
            with ph_graph:
                st.components.v1.html(html, height=480, scrolling=True)
        except Exception as e:
            ph_graph.caption(f"Evidence graph rendering skipped: {e}")

        if animate and step_delay:
            time.sleep(step_delay)

    st.markdown("### Metacognitive Trace Table")
    st.dataframe(C.trace_rows(traces), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Final answer", out.answer)
    c2.metric("Retrieval steps", out.n_steps)
    c3.metric("Total tokens", out.token_usage["total_tokens"])
    if gold:
        sc = score_prediction(out.answer, gold)
        st.caption(f"Gold answers: {gold} | EM={sc['em']:.0f} | F1={sc['f1']:.2f}")

    st.markdown("## Comparison with Fixed-Budget Naive RAG")
    naive = NaiveRAG(backend, pidx, NaiveConfig(max_steps=max_steps))
    nout = naive.answer(question)
    st.dataframe(C.comparison_rows(out, nout), use_container_width=True)
    d_steps = nout.n_steps - out.n_steps
    st.info(
        f"MSR stopped after {out.n_steps} step(s), while Naive RAG used "
        f"{nout.n_steps} step(s). Saved retrieval steps: {d_steps}. "
        "The main behavior to highlight is adaptive stopping for easy cases "
        "and targeted expansion when evidence is incomplete."
    )
else:
    st.info(
        "Select a backend, dataset, threshold, and question from the sidebar, "
        "then click Run MSR-GraphRAG. The default mock + toy setting runs "
        "immediately without a GPU."
    )
