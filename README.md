# MSR-GraphRAG

Metacognitive Self-Regulating Graph Retrieval-Augmented Generation

MSR-GraphRAG is a research prototype for adaptive graph-based retrieval in
multi-hop question answering. The project studies whether an LLM retrieval
system can decide, at each graph traversal stride, whether the current evidence
is sufficient or whether the graph should be expanded further.

The central idea is a training-free metacognitive controller. Instead of using a
fixed graph depth for every question, the controller computes a Confidence Gap
from three evidence-sufficiency signals:

- ECS: Evidence Coverage Score
- PCS: Path Coherence Score
- AGS: Answer Generability Score

The controller stops when the evidence is judged sufficient and expands the
graph when the evidence is incomplete.

```text
Confidence(S) = (w_ecs * ECS + w_pcs * PCS + w_ags * AGS)
                / (w_ecs + w_pcs + w_ags)
ConfGap(S)   = 1 - Confidence(S)

STOP   if ConfGap < tau
EXPAND otherwise
```

The final default scorer configuration is:

```text
alpha = 0.45  # ECS weight
beta  = 0.40  # PCS weight
gamma = 0.15  # AGS weight
tau   = 0.30  # stopping threshold
```

## Research Motivation

GraphRAG systems are useful for multi-hop QA because they can traverse
entities, passages, and relations rather than retrieving independent passages
only. However, graph traversal introduces a control problem:

- Too little expansion misses required evidence.
- Too much expansion wastes tokens and can introduce distracting context.
- Fixed-depth traversal gives easy and hard questions the same retrieval budget.

MSR-GraphRAG frames this as an adaptive evidence-sufficiency problem. At each
retrieval step, the system asks whether the current evidence state is enough to
answer. If not, it expands in the direction suggested by the weakest signal.

## Method

### Evidence Coverage Score

ECS measures whether entities extracted from the question are covered by the
currently selected graph nodes.

```text
ECS(S) = |covered question entities| / max(|question entities|, 1)
```

If no question entities are extracted, ECS is set to 1.0. When ECS is the
weakest signal, the retriever favors breadth-oriented expansion to search for
missing question entities.

### Path Coherence Score

PCS measures whether the selected evidence subgraph is structurally connected.
Let `n` be the number of selected nodes and `c` the number of connected
components:

```text
PCS(S) = 1 - (c - 1) / max(n, 1)
```

PCS is 1.0 for a single-node subgraph and 0.0 for an empty subgraph. When PCS is
low, the system favors depth-oriented expansion to connect evidence fragments.

### Answer Generability Score

AGS estimates whether the backend model can generate a short answer from the
current evidence. The model is prompted to draft a short answer using only the
retrieved evidence and to output `UNKNOWN` when the evidence is insufficient.
Token-level logit traces are converted into confidence values.

For the `margin_entropy` setting used in the final experiments:

```text
C_margin  = 1 - exp(-lambda * avg_top1_top2_logprob_margin)
C_entropy = 1 - min(avg_token_entropy / h_max, 1)
AGS(S)    = 0.5 * (C_margin + C_entropy)
```

If the draft answer looks like an abstention, AGS is multiplied by an abstain
penalty. If logits are unavailable, AGS is disabled and the remaining ECS/PCS
weights are renormalized.

## Repository Layout

```text
msr_graphrag/
  backends/       # Mock, Hugging Face, and vLLM backend adapters
  baselines/      # Naive RAG and LightRAG-compatible baseline wrappers
  controller/     # Stride-wise self-regulating retrieval loop
  data/           # Dataset loading and schemas
  eval/           # Metrics and experiment runner
  kg/             # Graph store, entity index, and KG builders
  pipeline/       # End-to-end MSR-GraphRAG pipeline
  retrieval/      # Query analysis, retrieval state, and graph expansion
  scorer/         # ECS/PCS/AGS scoring and Confidence Gap logic

analysis/         # Plotting and claim-summary utilities
configs/          # Experiment configurations
demo/             # Streamlit demo
examples/         # Minimal usage examples
scripts/          # Reproducible experiment commands
tests/            # Unit and smoke tests
```

Generated result directories are intentionally excluded from the repository.

## Installation

Create an environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[all]"
```

For lightweight local checks without GPU dependencies:

```bash
pip install -e .
```

## Quick Checks

Run the smoke experiments with the mock backend:

```bash
python scripts/03_run_experiments.py --smoke
python scripts/04_run_ablation.py --smoke --ablation all
python scripts/05_make_plots.py --results result_smoke/main --ablation result_smoke/ablation --out result_smoke/figures
```

Run tests:

```bash
python -m pytest tests -q
```

If pytest is unavailable, the core test files can be run directly:

```bash
python tests/test_scorer.py
python tests/test_metrics.py
python tests/test_answer_generator.py
python tests/test_summarize_claims.py
```

## Running Main Experiments

Build processed corpora:

```bash
python scripts/01_build_corpus.py \
  --datasets hotpotqa 2wiki musique \
  --n-samples 300 \
  --out result_final/data_processed
```

Run Hugging Face experiments with Qwen2.5-7B-Instruct:

```bash
python scripts/03_run_experiments.py \
  --data result_final/data_processed \
  --datasets hotpotqa 2wiki musique \
  --backend hf \
  --models Qwen/Qwen2.5-7B-Instruct \
  --kg-strategy heuristic \
  --max-steps 7 \
  --ags-signal margin_entropy \
  --out result_final/main \
  --keep-traces
```

Run ablations:

```bash
python scripts/04_run_ablation.py \
  --ablation all \
  --data result_final/data_processed \
  --datasets hotpotqa 2wiki musique \
  --backend mock \
  --models mock \
  --kg-strategy heuristic \
  --out result_final/ablation_mock
```

Generate plots and a claim summary:

```bash
python scripts/05_make_plots.py \
  --results result_final/main \
  --ablation result_final/ablation_mock \
  --out result_final/figures

python scripts/06_summarize_claims.py \
  --results result_final/main \
  --out result_final/claim_summary.json
```

## Demo

Run the Streamlit demo:

```bash
streamlit run demo/app.py
```

The demo supports inspecting retrieval traces, Confidence Gap trajectories,
single-signal ablations, and baseline comparisons.

## Final Experimental Interpretation

The final GPU run used Qwen/Qwen2.5-7B-Instruct on HotpotQA,
2WikiMultihopQA, and MuSiQue with 300 examples per dataset. The final
configuration used `alpha=0.45`, `beta=0.40`, `gamma=0.15`, `tau=0.30`, and
`max_steps=7`.

The strongest supported claim is:

> MSR-GraphRAG substantially improves fixed graph traversal for graph-based
> multi-hop retrieval. Across three datasets and 900 total examples, it improved
> macro F1 from 0.0210 to 0.2249 while reducing retrieval steps by 65.25% and
> token consumption by 80.50%.

The result should be interpreted carefully. The current heuristic graph
pipeline did not outperform the naive passage-retrieval baseline in answer F1.
Therefore, the contribution is best framed as an adaptive control mechanism
that makes graph retrieval much more efficient and less brittle than fixed graph
expansion, rather than as a state-of-the-art QA accuracy result.

See `final_summary.txt` for the full research-style report.

## Notes on Reproducibility

- Large generated outputs are excluded by `.gitignore`.
- Hugging Face gated models may require `huggingface-cli login` or an
  authenticated `HF_TOKEN`.
- GPU experiments were validated with an RTX 3090 and PyTorch CUDA builds.
- The mock backend is useful for fast control-flow and ablation diagnostics but
  should not be treated as final answer-quality evidence.

## License

This project is released under the MIT License.
