"""VLLMBackend — vLLM 백엔드 (대규모 실험 권장).

vLLM 의 ``SamplingParams(logprobs=k)`` 로 생성 토큰별 상위-k logprob 를 받아
top-1/top-2 갭(=AGS margin 신호)과 선택토큰 NLL 을 계산한다. transformers 대비
처리량이 높아 3개 벤치마크 × 2모델 전수 실험에 적합.

엔트로피는 상위-k logprob 만으로는 부정확하므로(전체 분포 미관측), margin/nll 신호를
우선 사용한다. (scorer 기본 ags_signal='margin' 과 일치.)

임베딩은 sentence-transformers 로 분리(기본 BAAI/bge-small-en-v1.5).
GPU 환경 전용(본 컨테이너 미실행).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import LLMBackend, GenerationResult, _parse_json_string_list
from ..scorer.uncertainty import LogitTrace


class VLLMBackend(LLMBackend):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        logprobs_k: int = 5,
        device: str = "cuda",
        trust_remote_code: bool = True,
    ):
        super().__init__(model_name=model_name)
        from vllm import LLM
        from transformers import AutoTokenizer

        self.logprobs_k = logprobs_k
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
        self.llm = LLM(
            model=model_name, dtype=dtype, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
        )

        self._embed_model_name = embed_model_name
        self._device = device
        self._embedder = None
        self._embed_dim: Optional[int] = None

    # --- 임베딩 -------------------------------------------------------------
    def _ensure_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                self._embed_model_name, device=self._device)
            self._embed_dim = int(
                self._embedder.get_sentence_embedding_dimension())

    @property
    def embed_dim(self) -> int:
        self._ensure_embedder()
        return int(self._embed_dim)

    def embed(self, texts: List[str]) -> np.ndarray:
        self._ensure_embedder()
        if not texts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        emb = self._embedder.encode(
            list(texts), normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return emb.astype(np.float32)

    # --- 프롬프트 구성 ------------------------------------------------------
    def _render(self, prompt: str, system: Optional[str]) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return (f"{system}\n\n" if system else "") + prompt

    # --- 생성 ---------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        with_logits: bool = False,
        role: str = "generic",
        system: Optional[str] = None,
    ) -> GenerationResult:
        from vllm import SamplingParams

        text_in = self._render(prompt, system)
        sp = SamplingParams(
            temperature=float(temperature),
            max_tokens=max_new_tokens,
            logprobs=self.logprobs_k if with_logits else None,
        )
        outs = self.llm.generate([text_in], sp, use_tqdm=False)
        comp = outs[0].outputs[0]
        text = comp.text.strip()

        prompt_tokens = len(outs[0].prompt_token_ids)
        completion_tokens = len(comp.token_ids)

        if with_logits and comp.logprobs is not None:
            trace = self._trace_from_logprobs(comp)
        else:
            trace = LogitTrace(available=False)

        res = GenerationResult(
            text=text, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, logit_trace=trace,
        )
        self.account.add(res, role=role)
        return res

    def _trace_from_logprobs(self, comp) -> LogitTrace:
        """vLLM CompletionOutput.logprobs → LogitTrace.

        comp.logprobs: List[Dict[token_id, Logprob]] (스텝별 상위-k).
        comp.token_ids: 실제 선택 토큰 id 리스트.
        """
        top1, top2, chosen = [], [], []
        token_ids = list(comp.token_ids)
        for i, step in enumerate(comp.logprobs):
            if not step:
                continue
            # logprob 내림차순 정렬
            items = sorted(step.items(), key=lambda kv: -kv[1].logprob)
            lp1 = items[0][1].logprob
            lp2 = items[1][1].logprob if len(items) > 1 else (lp1 - 10.0)
            top1.append(float(lp1))
            top2.append(float(lp2))
            cid = token_ids[i] if i < len(token_ids) else items[0][0]
            if cid in step:
                chosen.append(float(step[cid].logprob))
            else:
                chosen.append(float(lp1))
        return LogitTrace(
            top1_logprobs=top1, top2_logprobs=top2,
            token_entropies=[],  # 상위-k 만으론 전체 엔트로피 부정확 → 미사용
            chosen_logprobs=chosen, available=True,
        )

    def extract_entities(self, query: str, role: str = "query_analyzer") -> List[str]:
        prompt = (
            "Extract the key named entities and noun phrases from the question "
            "needed to find the answer. Return ONLY a JSON array of strings.\n\n"
            f"Question: {query}\n\nEntities:"
        )
        res = self.generate(prompt, max_new_tokens=96, temperature=0.0, role=role)
        return _parse_json_string_list(res.text)
