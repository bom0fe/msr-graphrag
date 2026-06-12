"""End-to-end smoke test with MockBackend and toy data."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from msr_graphrag.backends.mock import MockBackend
from msr_graphrag.baselines.naive_rag import NaiveConfig, NaiveRAG, PassageIndex
from msr_graphrag.controller.msr_controller import ControllerConfig
from msr_graphrag.data.loaders import load_toy_dataset
from msr_graphrag.eval.metrics import score_prediction
from msr_graphrag.pipeline.msr_graphrag import MSRGraphRAG
from msr_graphrag.scorer.metacognitive_scorer import ScorerConfig


def main():
    examples, corpus = load_toy_dataset()
    oracle = {ex.question: ex.answer for ex in examples}
    backend = MockBackend(answer_oracle=oracle)

    print("=== 1) KG indexing (heuristic, mock) ===")
    rag = MSRGraphRAG(
        backend,
        kg_backend="native",
        kg_strategy="heuristic",
        scorer_config=ScorerConfig(tau=0.30, use_ags=True),
        controller_config=ControllerConfig(max_steps=6, min_steps=1, verbose=True),
        use_llm_entities=True,
    )
    rag.index(corpus.documents(), verbose=False)
    print("KG stats:", rag.store.stats())
    assert rag.store.stats()["n_nodes"] > 0, "KG is empty"

    print("\n=== 2) MSR-GraphRAG query ===")
    msr_records = []
    for ex in examples:
        out = rag.answer(ex.question)
        sc = score_prediction(out.answer, ex.all_answers)
        print(f"\nQ: {ex.question}\n  gold={ex.answer!r} pred={out.answer!r} "
              f"steps={out.n_steps} EM={sc['em']} F1={sc['f1']:.2f} "
              f"tokens={out.token_usage['total_tokens']} stop={out.stop_reason}")
        msr_records.append({
            **sc,
            "n_steps": out.n_steps,
            "total_tokens": out.token_usage["total_tokens"],
        })

    print("\n=== 3) NaiveRAG baseline ===")
    pidx = PassageIndex(corpus.passages, backend.embed)
    naive = NaiveRAG(backend, pidx, NaiveConfig(top_k_per_step=2, max_steps=6))
    naive_records = []
    for ex in examples:
        out = naive.answer(ex.question)
        sc = score_prediction(out.answer, ex.all_answers)
        naive_records.append({
            **sc,
            "n_steps": out.n_steps,
            "total_tokens": out.token_usage["total_tokens"],
        })
        print(f"Q: {ex.question[:40]}... pred={out.answer!r} steps={out.n_steps} "
              f"tokens={out.token_usage['total_tokens']}")

    print("\n=== 4) Summary ===")

    def avg(rs, k):
        return sum(r[k] for r in rs) / len(rs)

    print(f"MSR   : EM={avg(msr_records,'em'):.2f} F1={avg(msr_records,'f1'):.2f} "
          f"steps={avg(msr_records,'n_steps'):.1f} tokens={avg(msr_records,'total_tokens'):.0f}")
    print(f"Naive : EM={avg(naive_records,'em'):.2f} F1={avg(naive_records,'f1'):.2f} "
          f"steps={avg(naive_records,'n_steps'):.1f} tokens={avg(naive_records,'total_tokens'):.0f}")

    out = rag.answer(examples[0].question)
    assert len(out.traces) >= 1
    t0 = out.traces[0]
    for k in ["ecs", "pcs", "ags", "conf_gap", "decision"]:
        assert k in t0, f"trace missing {k}"
    print("\n[OK] smoke test passed. trace keys:", list(t0.keys()))


if __name__ == "__main__":
    main()
