"""Unit tests for the metacognitive scorer."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from msr_graphrag.backends.mock import MockBackend
from msr_graphrag.kg.builder_native import NativeKGBuilder
from msr_graphrag.kg.entity_index import EntityIndex
from msr_graphrag.retrieval.state import RetrievalState
from msr_graphrag.scorer.metacognitive_scorer import MetacognitiveScorer, ScorerConfig


def _setup():
    docs = [
        "Marie Curie. Marie Curie was a physicist who won two Nobel Prizes. "
        "Marie Curie was raised in Poland.",
        "Poland. Poland is a country in Central Europe. The capital of Poland is Warsaw.",
        "Warsaw. Warsaw is the largest city of Poland.",
    ]
    be = MockBackend()
    store = NativeKGBuilder(be, strategy="heuristic").build(docs)
    idx = EntityIndex.build(store, be.embed)
    return be, store, idx


def test_default_weights_updated():
    cfg = ScorerConfig()
    assert cfg.alpha == 0.45
    assert cfg.beta == 0.4
    assert cfg.gamma == 0.15


def test_confgap_formula():
    be, store, idx = _setup()
    sc = MetacognitiveScorer(be, ScorerConfig(alpha=0.45, beta=0.4, gamma=0.15))
    st = RetrievalState(
        query="Where was Marie Curie born?",
        query_entities=["Marie Curie", "Poland"],
    )
    st.selected_nodes |= {"Marie Curie", "Poland", "Nobel Prizes"}
    b = sc.score(st, store, max_steps=7)
    expect_conf = (0.45 * b.ecs + 0.4 * b.pcs + 0.15 * b.ags) / 1.0
    assert abs((1.0 - expect_conf) - b.conf_gap) < 1e-6
    assert 0.0 <= b.conf_gap <= 1.0


def test_weight_renormalization_when_disabled():
    be, store, idx = _setup()
    sc = MetacognitiveScorer(
        be, ScorerConfig(use_ecs=True, use_pcs=False, use_ags=False)
    )
    st = RetrievalState(query="q", query_entities=["Marie Curie", "Poland", "Warsaw"])
    st.selected_nodes |= {"Marie Curie"}
    b = sc.score(st, store, max_steps=7)
    assert abs(b.ecs - (1 / 3)) < 1e-6
    assert abs(b.conf_gap - (1 - b.ecs)) < 1e-6


def test_ags_fallback_when_logits_unavailable():
    be, store, idx = _setup()

    class NoLogitBackend(MockBackend):
        def generate(self, *a, **k):
            k["with_logits"] = False
            return super().generate(*a, **k)

    nb = NoLogitBackend()
    sc = MetacognitiveScorer(nb, ScorerConfig(alpha=0.45, beta=0.4, gamma=0.15))
    st = RetrievalState(query="q", query_entities=["Marie Curie", "Poland"])
    st.selected_nodes |= {"Marie Curie", "Poland"}
    b = sc.score(st, store, max_steps=7)
    conf = (0.45 * b.ecs + 0.4 * b.pcs) / 0.85
    assert abs((1 - conf) - b.conf_gap) < 1e-6


def test_routing_targets_weakest():
    be, store, idx = _setup()
    sc = MetacognitiveScorer(be, ScorerConfig(enable_routing=True))
    st = RetrievalState(
        query="q", query_entities=["Marie Curie", "Zeus", "Atlantis", "Mars"]
    )
    st.selected_nodes |= {"Marie Curie"}
    b = sc.score(st, store, max_steps=7)
    if b.decision == "EXPAND":
        assert b.expand_mode in {"breadth", "depth", "depth_pivot"}
        assert b.weakest in {"ECS", "PCS", "AGS"}


def test_tau_boundary_decision():
    be, store, idx = _setup()
    st = RetrievalState(query="q", query_entities=["Marie Curie"])
    st.selected_nodes |= {"Marie Curie", "Poland", "Warsaw"}
    b_hi = MetacognitiveScorer(be, ScorerConfig(tau=0.99)).score(st, store, 7)
    assert b_hi.decision == "STOP"

    st2 = RetrievalState(query="q", query_entities=["Marie Curie", "Zeus", "Atlantis"])
    st2.selected_nodes |= {"Marie Curie"}
    b_lo = MetacognitiveScorer(be, ScorerConfig(tau=0.01)).score(st2, store, 7)
    assert b_lo.decision == "EXPAND"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[ok] {fn.__name__}")
    print(f"\n[PASS] scorer: {len(fns)} tests")


if __name__ == "__main__":
    _run_all()
