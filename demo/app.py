"""demo/app.py — MSR-GraphRAG 인터랙티브 데모 (Streamlit).

실행:
  streamlit run demo/app.py

제안서 Demo Scenes 매핑
----------------------
  Scene 1: 질문 입력 → KG 위 stride-wise 순회 시작
  Scene 2: 매 step "확신 미터"(ConfGap 게이지 + ECS/PCS/AGS 막대) 실시간 표시
  Scene 3: ConfGap < τ 도달 → "Enough evidence!" STOP, 서브그래프 확정
  Scene 4: NaiveRAG(항상 최대 검색) 와 정확도/step/토큰 나란히 비교

기본은 MockBackend + 토이 데이터라 GPU/HF 없이 데모 영상 촬영 가능.
사이드바에서 HF/vLLM 백엔드와 HotpotQA/2Wiki/MuSiQue 로 전환 가능.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import streamlit as st

from msr_graphrag.backends import build_backend
from msr_graphrag.data import load_toy_dataset, load_examples, load_corpus
from msr_graphrag.pipeline.msr_graphrag import MSRGraphRAG
from msr_graphrag.scorer.metacognitive_scorer import ScorerConfig
from msr_graphrag.controller.msr_controller import ControllerConfig
from msr_graphrag.baselines.naive_rag import NaiveRAG, PassageIndex, NaiveConfig
from msr_graphrag.eval.metrics import score_prediction
from demo import components as C


st.set_page_config(page_title="MSR-GraphRAG Demo", layout="wide",
                   page_icon="🧭")


# ---------------------------------------------------------------------------
# 리소스 (백엔드 + KG) 캐시
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline(backend_kind: str, model: str, dataset: str, data_dir: str,
                  kg_strategy: str):
    """백엔드/데이터/KG 구축 (캐시). 토이는 heuristic, 그 외 llm 추출."""
    if dataset == "toy":
        examples, corpus = load_toy_dataset()
        oracle = {e.question: e.answer for e in examples}
        backend = build_backend("mock", answer_oracle=oracle) \
            if backend_kind == "mock" else build_backend(backend_kind, model_name=model)
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


def make_rag_with_cfg(_backend, _store, _index, tau, max_steps, ags_signal,
                      alpha, beta, gamma, use_ecs, use_pcs, use_ags, routing):
    rag = MSRGraphRAG(
        _backend, kg_backend="native",
        scorer_config=ScorerConfig(
            alpha=alpha, beta=beta, gamma=gamma, tau=tau, ags_signal=ags_signal,
            use_ecs=use_ecs, use_pcs=use_pcs, use_ags=use_ags,
            enable_routing=routing),
        controller_config=ControllerConfig(max_steps=max_steps, min_steps=1),
    )
    rag.set_kg(_store, _index)
    return rag


# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
st.sidebar.title("🧭 MSR-GraphRAG")
st.sidebar.caption("Metacognitive Self-Regulating Agentic GraphRAG")

backend_kind = st.sidebar.selectbox("Backend", ["mock", "hf", "vllm"], index=0)
model = st.sidebar.text_input("Model", value="mock" if backend_kind == "mock"
                              else "Qwen/Qwen2.5-7B-Instruct")
dataset = st.sidebar.selectbox("Dataset", ["toy", "hotpotqa", "2wiki", "musique"],
                               index=0)
data_dir = st.sidebar.text_input("Data dir", value="data/processed")
kg_strategy = st.sidebar.selectbox("KG strategy", ["llm", "heuristic"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Metacognitive gate")
tau = st.sidebar.slider("τ (STOP threshold)", 0.05, 0.60, 0.30, 0.05)
max_steps = st.sidebar.slider("Max steps", 1, 12, 7, 1)
ags_signal = st.sidebar.selectbox("AGS signal",
                                  ["margin", "entropy", "nll", "margin_entropy"])
with st.sidebar.expander("Weights / ablation"):
    alpha = st.slider("α (ECS)", 0.0, 1.0, 0.3, 0.05)
    beta = st.slider("β (PCS)", 0.0, 1.0, 0.3, 0.05)
    gamma = st.slider("γ (AGS)", 0.0, 1.0, 0.4, 0.05)
    use_ecs = st.checkbox("use ECS", True)
    use_pcs = st.checkbox("use PCS", True)
    use_ags = st.checkbox("use AGS", True)
    routing = st.checkbox("expansion routing", True)
animate = st.sidebar.checkbox("Animate steps", True)
step_delay = st.sidebar.slider("Step delay (s)", 0.0, 1.5, 0.6, 0.1)


# ---------------------------------------------------------------------------
# 본문
# ---------------------------------------------------------------------------
st.title("Metacognitive Self-Regulating Agentic GraphRAG")
st.markdown(
    "기존 Agentic GraphRAG 는 외부 그래프를 **탐색**하지만 *무엇을 모르는지* 스스로 "
    "진단하지 못한다. 본 데모는 매 검색 step 마다 **Confidence Gap** "
    "(= 1 − α·ECS − β·PCS − γ·AGS)을 계산해, 근거가 충분하면 **STOP**, 아니면 "
    "약한 신호 방향으로 **EXPAND** 하는 메타인지 게이트를 시각화한다."
)

try:
    backend, examples, corpus, base_rag, pidx = load_pipeline(
        backend_kind, model, dataset, data_dir, kg_strategy)
except Exception as e:
    st.error(f"파이프라인 로드 실패: {e}\n\n토이가 아니면 먼저 "
             "`scripts/01_build_corpus.py` 로 데이터를 준비하세요.")
    st.stop()

st.success(f"KG 준비 완료 · {base_rag.store.stats()} · 코퍼스 {len(corpus.passages)} passages")

# 질문 선택/입력 (Scene 1)
ex_map = {f"[{e.difficulty_bucket()}] {e.question}": e for e in examples}
col_q1, col_q2 = st.columns([3, 2])
with col_q1:
    picked = st.selectbox("예제 질문", list(ex_map.keys()))
with col_q2:
    custom = st.text_input("또는 직접 입력", "")
question = custom.strip() or ex_map[picked].question
gold = None if custom.strip() else ex_map[picked].all_answers

run = st.button("▶ Run MSR-GraphRAG", type="primary")

if run:
    rag = make_rag_with_cfg(backend, base_rag.store, base_rag.entity_index,
                            tau, max_steps, ags_signal, alpha, beta, gamma,
                            use_ecs, use_pcs, use_ags, routing)
    out = rag.answer(question)
    traces = out.traces

    st.markdown("## Scene 2–3 · 메타인지 순회 (확신 미터)")
    ph_gauge, ph_bars = st.columns(2)
    ph_time = st.empty()
    ph_status = st.empty()
    ph_graph = st.empty()

    # 단계별 애니메이션 (Scene 2) → STOP (Scene 3)
    shown = range(1, len(traces) + 1) if animate else [len(traces)]
    for k in shown:
        t = traces[k - 1]
        with ph_gauge:
            st.plotly_chart(C.confgap_gauge(t["conf_gap"], tau),
                            use_container_width=True, key=f"g{k}")
        with ph_bars:
            st.plotly_chart(C.component_bars(t["ecs"], t["pcs"], t["ags"]),
                            use_container_width=True, key=f"b{k}")
        ph_time.plotly_chart(C.confgap_timeline(traces[:k], tau),
                             use_container_width=True, key=f"t{k}")
        if t.get("decision") == "STOP":
            ph_status.success(f"✅ **Enough evidence!** step {t['step']} 에서 "
                              f"ConfGap={t['conf_gap']:.3f} < τ={tau} → STOP")
        else:
            ph_status.info(f"🔎 step {t['step']}: ConfGap={t['conf_gap']:.3f} ≥ τ "
                           f"→ EXPAND[{t.get('expand_mode')}] "
                           f"(weakest={t.get('weakest')})")
        # 누적 서브그래프 (해당 step 까지 신규 노드 강조)
        sel = []
        for tt in traces[:k]:
            sel += tt.get("new_nodes", [])
        try:
            html = C.subgraph_html(base_rag.store, out.state.selected_nodes
                                   if hasattr(out, "state") else sel,
                                   seeds=out.state.seeds if hasattr(out, "state") else [],
                                   new_nodes=t.get("new_nodes", []))
            with ph_graph:
                st.components.v1.html(html, height=480, scrolling=True)
        except Exception as e:
            ph_graph.caption(f"(subgraph 렌더 생략: {e})")
        if animate and step_delay:
            time.sleep(step_delay)

    # 답변 + 트레이스 패널
    st.markdown("### 🧠 메타인지 트레이스")
    st.dataframe(C.trace_rows(traces), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Answer", out.answer)
    c2.metric("Retrieval steps", out.n_steps)
    c3.metric("Total tokens", out.token_usage["total_tokens"])
    if gold:
        sc = score_prediction(out.answer, gold)
        st.caption(f"gold={gold} · EM={sc['em']:.0f} · F1={sc['f1']:.2f}")

    # Scene 4 · Naive 비교
    st.markdown("## Scene 4 · NaiveRAG(항상 최대 검색) 대비")
    naive = NaiveRAG(backend, pidx, NaiveConfig(max_steps=max_steps))
    nout = naive.answer(question)
    st.dataframe(C.comparison_rows(out, nout), use_container_width=True)
    d_steps = (nout.n_steps - out.n_steps)
    st.info(f"동일/유사 정확도에서 MSR 은 **{out.n_steps} step** 으로 종료 "
            f"(Naive {nout.n_steps} step). 절감 step = {d_steps}. "
            "어려운 질문엔 확장, 쉬운 질문엔 조기 중단하는 적응적 동작이 핵심.")
else:
    st.info("좌측에서 백엔드·데이터·τ 를 설정하고, 질문을 골라 **Run** 을 누르세요. "
            "기본값(mock+toy)은 GPU 없이 즉시 동작합니다.")
