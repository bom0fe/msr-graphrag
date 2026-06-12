#!/usr/bin/env python
"""quickstart — MSR-GraphRAG 최소 사용 예제 (오프라인, GPU 불필요).

  python examples/quickstart.py

핵심 API:
  1) backend 생성 → 2) MSRGraphRAG.index(documents) → 3) .answer(question)
  out.traces 에 step 별 ECS/PCS/AGS/ConfGap/결정이 기록된다.

실제 모델은 backend 만 교체:
  from msr_graphrag.backends import build_backend
  backend = build_backend("vllm", model_name="Qwen/Qwen2.5-7B-Instruct")
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from msr_graphrag.backends import build_backend
from msr_graphrag.data import load_toy_dataset
from msr_graphrag.pipeline.msr_graphrag import MSRGraphRAG
from msr_graphrag.scorer.metacognitive_scorer import ScorerConfig
from msr_graphrag.controller.msr_controller import ControllerConfig
from msr_graphrag.eval.metrics import score_prediction


def main():
    examples, corpus = load_toy_dataset()

    # 1) backend (오프라인 데모용 mock; 실제론 build_backend("vllm", model_name=...))
    backend = build_backend(
        "mock", answer_oracle={e.question: e.answer for e in examples})

    # 2) 인덱싱 (KG 구축)
    rag = MSRGraphRAG(
        backend, kg_backend="native", kg_strategy="heuristic",
        scorer_config=ScorerConfig(tau=0.30, ags_signal="margin"),
        controller_config=ControllerConfig(max_steps=6, min_steps=1),
    )
    rag.index(corpus.documents())
    print("KG:", rag.store.stats(), "\n")

    # 3) 질의 + step 별 메타인지 트레이스 출력
    for ex in examples:
        out = rag.answer(ex.question)
        sc = score_prediction(out.answer, ex.all_answers)
        print(f"Q[{ex.difficulty_bucket()}]: {ex.question}")
        print(f"  → answer={out.answer!r}  steps={out.n_steps}  "
              f"EM={sc['em']:.0f} F1={sc['f1']:.2f}  "
              f"tokens={out.token_usage['total_tokens']}")
        for t in out.traces:
            print(f"    step{t['step']}: ECS={t['ecs']:.2f} PCS={t['pcs']:.2f} "
                  f"AGS={t['ags']:.2f} ConfGap={t['conf_gap']:.2f} "
                  f"→ {t['decision']}"
                  + (f"[{t['expand_mode']}]" if t['decision'] == 'EXPAND' else ""))
        print()


if __name__ == "__main__":
    main()
