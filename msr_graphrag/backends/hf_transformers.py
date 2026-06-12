"""HFTransformersBackend — HuggingFace transformers 백엔드.

핵심: ``model.generate(output_scores=True, return_dict_in_generate=True)`` 로
스텝별 로짓을 직접 받아 top-1/top-2 갭·엔트로피·선택토큰 logprob 를 산출한다.
→ AGS(Answer Generability Score)의 로짓 신호를 정확히 제공.

모델 예시 (HF Hub):
- Qwen/Qwen2.5-7B-Instruct
- meta-llama/Llama-3.1-8B-Instruct
임베딩은 sentence-transformers (기본 BAAI/bge-small-en-v1.5) 로 분리.

GPU 환경 전용(본 컨테이너에선 미실행). vLLM 대비 느리나 로짓 접근이 단순/정확.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import LLMBackend, GenerationResult, _parse_json_string_list
from ..scorer.uncertainty import LogitTrace


class HFTransformersBackend(LLMBackend):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        trust_remote_code: bool = True,
    ):
        super().__init__(model_name=model_name)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self._dtype = getattr(torch, dtype, torch.bfloat16)
        self.max_model_len = max_model_len

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self._dtype, device_map=device,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        # 임베딩 모델 (지연 로드)
        self._embed_model_name = embed_model_name
        self._embedder = None
        self._embed_dim: Optional[int] = None

    # --- 임베딩 -------------------------------------------------------------
    def _ensure_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                self._embed_model_name, device=self.device)
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
    def _build_inputs(self, prompt: str, system: Optional[str]):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        try:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # chat template 미지원 모델 fallback
            text = (f"{system}\n\n" if system else "") + prompt
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_model_len,
        ).to(self.model.device)
        return enc

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
        torch = self.torch
        enc = self._build_inputs(prompt, system)
        input_len = int(enc["input_ids"].shape[1])
        do_sample = temperature and temperature > 0.0

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=bool(do_sample),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
        if with_logits:
            gen_kwargs.update(output_scores=True, return_dict_in_generate=True)

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        if with_logits:
            seq = out.sequences[0]
            gen_ids = seq[input_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            trace = self._trace_from_scores(out.scores, gen_ids)
        else:
            gen_ids = out[0][input_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            trace = LogitTrace(available=False)

        res = GenerationResult(
            text=text.strip(),
            prompt_tokens=input_len,
            completion_tokens=int(len(gen_ids)),
            logit_trace=trace,
        )
        self.account.add(res, role=role)
        return res

    def _trace_from_scores(self, scores, gen_ids) -> LogitTrace:
        """generate(output_scores=True) 의 스텝별 로짓 → LogitTrace."""
        torch = self.torch
        top1, top2, ents, chosen = [], [], [], []
        n = min(len(scores), int(gen_ids.shape[0]))
        for i in range(n):
            logits = scores[i][0].float()
            logp = torch.log_softmax(logits, dim=-1)
            topv, _ = torch.topk(logp, k=2)
            top1.append(float(topv[0]))
            top2.append(float(topv[1]))
            p = logp.exp()
            ents.append(float(-(p * logp).sum()))
            cid = int(gen_ids[i])
            chosen.append(float(logp[cid]))
        return LogitTrace(
            top1_logprobs=top1, top2_logprobs=top2,
            token_entropies=ents, chosen_logprobs=chosen, available=True,
        )

    # --- 엔티티 추출 (JSON 강제) -------------------------------------------
    def extract_entities(self, query: str, role: str = "query_analyzer") -> List[str]:
        prompt = (
            "Extract the key named entities and noun phrases from the question "
            "needed to find the answer. Return ONLY a JSON array of strings.\n\n"
            f"Question: {query}\n\nEntities:"
        )
        res = self.generate(prompt, max_new_tokens=96, temperature=0.0, role=role)
        ents = _parse_json_string_list(res.text)
        return ents
