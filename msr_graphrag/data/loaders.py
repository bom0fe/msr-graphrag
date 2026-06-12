"""데이터 로더 — HuggingFace 멀티홉 QA 데이터셋 → 정규화 스키마.

지원: HotpotQA, 2WikiMultihopQA, MuSiQue. 각 데이터셋의 서로 다른 원본 포맷을
``QAExample`` 로 통일하고, 샘플링 후 패시지를 풀링해 KG 인덱싱용 ``Corpus`` 를 만든다
(PIKE-RAG / KET-RAG 방식: 질문별 gold+distractor 문단을 단일 코퍼스로 합침).

HuggingFace 미설치/오프라인 환경을 위해 ``load_toy_dataset`` 도 제공(스모크 테스트).

기본 HF 식별자(필요 시 override):
  hotpotqa : "hotpot_qa"            config="distractor" split="validation"
  2wiki    : "xanhho/2WikiMultihopQA"            split="validation"
  musique  : "dgslibisey/MuSiQue"                split="validation"
"""
from __future__ import annotations

import json
import random
from typing import List, Dict, Any, Optional, Tuple

from .schema import QAExample, Passage, Corpus

# 데이터셋별 기본 HF 설정 (override 가능). 식별자는 환경에 따라 다를 수 있으므로
# CLI/config 에서 바꿀 수 있게 둔다.
DEFAULT_HF = {
    "hotpotqa": {"path": "hotpotqa/hotpot_qa", "config": "distractor", "split": "validation"},
    "2wiki": {"path": "xanhho/2WikiMultihopQA", "config": None, "split": "validation"},
    "musique": {"path": "dgslibisey/MuSiQue", "config": None, "split": "validation"},
}


# ---------------------------------------------------------------------------
# 어댑터: 원본 row → QAExample
# ---------------------------------------------------------------------------
def _adapt_hotpotqa(row: Dict[str, Any], idx: int) -> QAExample:
    ctx = row.get("context", {})
    titles = ctx.get("title", []) if isinstance(ctx, dict) else [c[0] for c in ctx]
    sents = ctx.get("sentences", []) if isinstance(ctx, dict) else [c[1] for c in ctx]
    sf = row.get("supporting_facts", {})
    gold_titles = list(set(sf.get("title", []))) if isinstance(sf, dict) else \
        list({x[0] for x in sf})
    passages = []
    for t, ss in zip(titles, sents):
        text = " ".join(ss) if isinstance(ss, list) else str(ss)
        passages.append(Passage(title=str(t), text=text,
                                 is_supporting=str(t) in set(gold_titles),
                                 source_qid=str(row.get("id", idx))))
    return QAExample(
        qid=str(row.get("id", idx)), question=row["question"], answer=row.get("answer", ""),
        gold_titles=gold_titles, passages=passages,
        num_hops=2, qtype=row.get("type"), level=row.get("level"),
        dataset="hotpotqa",
    )


def _adapt_2wiki(row: Dict[str, Any], idx: int) -> QAExample:
    ctx = _maybe_json(row.get("context", []))
    # 포맷 변형 처리: list[[title,[sents]]] 또는 {title:[], content:[]}
    titles, sents = [], []
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sents = ctx.get("content", ctx.get("sentences", []))
    else:
        for c in ctx:
            titles.append(c[0]); sents.append(c[1])
    sf = _maybe_json(row.get("supporting_facts", []))
    if isinstance(sf, dict):
        gold_titles = list(set(sf.get("title", [])))
    else:
        gold_titles = list({x[0] for x in sf})
    passages = []
    for t, ss in zip(titles, sents):
        text = " ".join(ss) if isinstance(ss, list) else str(ss)
        passages.append(Passage(title=str(t), text=text,
                                 is_supporting=str(t) in set(gold_titles),
                                 source_qid=str(row.get("_id", row.get("id", idx)))))
    # 2Wiki 는 evidences(triple) 로 hop 수 추정 가능
    ev = _maybe_json(row.get("evidences", row.get("evidences_id", [])))
    num_hops = len(ev) if isinstance(ev, list) and ev else None
    return QAExample(
        qid=str(row.get("_id", row.get("id", idx))), question=row["question"],
        answer=row.get("answer", ""), gold_titles=gold_titles, passages=passages,
        num_hops=num_hops, qtype=row.get("type"), dataset="2wiki",
    )


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _adapt_musique(row: Dict[str, Any], idx: int) -> QAExample:
    paras = row.get("paragraphs", [])
    passages, gold_titles = [], []
    for p in paras:
        t = p.get("title", "")
        sup = bool(p.get("is_supporting", False))
        if sup:
            gold_titles.append(t)
        passages.append(Passage(title=str(t), text=p.get("paragraph_text", ""),
                                 is_supporting=sup, source_qid=str(row.get("id", idx))))
    decomp = row.get("question_decomposition", [])
    num_hops = len(decomp) if decomp else None
    aliases = row.get("answer_aliases", []) or []
    return QAExample(
        qid=str(row.get("id", idx)), question=row["question"],
        answer=row.get("answer", ""), answer_aliases=list(aliases),
        gold_titles=gold_titles, passages=passages, num_hops=num_hops,
        dataset="musique",
    )


_ADAPTERS = {"hotpotqa": _adapt_hotpotqa, "2wiki": _adapt_2wiki, "musique": _adapt_musique}


# ---------------------------------------------------------------------------
# 로드 API
# ---------------------------------------------------------------------------
def load_dataset_examples(name: str, n_samples: int = 200, seed: int = 42,
                          hf_path: Optional[str] = None, hf_config: Optional[str] = None,
                          split: Optional[str] = None) -> List[QAExample]:
    """HF 데이터셋 로드 → 샘플링 → QAExample 리스트.

    name: 'hotpotqa' | '2wiki' | 'musique'.
    실패 시 명확한 에러를 던진다(식별자/네트워크 점검 유도).
    """
    name = name.lower()
    if name not in _ADAPTERS:
        raise ValueError(f"unknown dataset: {name}")
    cfg = DEFAULT_HF[name]
    path = hf_path or cfg["path"]
    config = hf_config if hf_config is not None else cfg["config"]
    split = split or cfg["split"]

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets 필요 (HF 데이터셋 로드)") from e

    load_kwargs = {"split": split, "trust_remote_code": True}
    if config:
        ds = load_dataset(path, config, **load_kwargs)
    elif name == "2wiki" and path == DEFAULT_HF["2wiki"]["path"]:
        ds = load_dataset(
            "parquet",
            data_files={split: f"hf://datasets/{path}/dev.parquet"},
            split=split,
        )
    elif name == "musique" and path == DEFAULT_HF["musique"]["path"]:
        rows = _load_musique_jsonl(split)
        n = min(n_samples, len(rows))
        rng = random.Random(seed)
        idxs = rng.sample(range(len(rows)), n)
        out = []
        for i in idxs:
            try:
                out.append(_adapt_musique(rows[i], i))
            except Exception as e:
                print(f"[warn] skip {name}[{i}]: {e}")
        return out
    else:
        ds = load_dataset(path, **load_kwargs)

    n = min(n_samples, len(ds))
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), n)
    adapt = _ADAPTERS[name]
    out = []
    for i in idxs:
        try:
            out.append(adapt(ds[i], i))
        except Exception as e:  # noqa: BLE001 - 단일 row 오류는 건너뜀
            print(f"[warn] skip {name}[{i}]: {e}")
    return out


def _load_musique_jsonl(split: str) -> List[Dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    filename = "musique_ans_v1.0_dev.jsonl"
    if split not in ("validation", "dev"):
        filename = "musique_ans_v1.0_train.jsonl"
    path = hf_hub_download("dgslibisey/MuSiQue", filename, repo_type="dataset")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_corpus(examples: List[QAExample], name: str = "corpus") -> Corpus:
    """예제들의 패시지를 (title,text) 기준 중복 제거하여 단일 코퍼스로 풀링."""
    seen = set()
    passages: List[Passage] = []
    for ex in examples:
        for p in ex.passages:
            key = (p.title.strip().lower(), p.text.strip()[:120].lower())
            if key in seen:
                continue
            seen.add(key)
            passages.append(p)
    return Corpus(name=name, passages=passages)


# ---------------------------------------------------------------------------
# 오프라인 토이 데이터 (스모크 테스트/데모)
# ---------------------------------------------------------------------------
def load_toy_dataset() -> Tuple[List[QAExample], Corpus]:
    """HF 없이 구동 가능한 소규모 멀티홉 예제 + 코퍼스."""
    passages = [
        Passage("Parasite (film)",
                "Parasite is a 2019 South Korean black comedy thriller film directed by "
                "Bong Joon-ho. It won the Academy Award for Best Picture.", True, "q1"),
        Passage("Bong Joon-ho",
                "Bong Joon-ho is a South Korean film director. He directed Parasite and "
                "won the Academy Award for Best Director in 2020.", True, "q1"),
        Passage("Academy Award for Best Director",
                "The Academy Award for Best Director is presented by the Academy of Motion "
                "Picture Arts and Sciences. Bong Joon-ho won it for Parasite.", False, "q1"),
        Passage("Memories of Murder",
                "Memories of Murder is a 2003 South Korean crime film directed by Bong "
                "Joon-ho, based on a true story.", False, "q1"),
        Passage("Tom Hanks",
                "Tom Hanks is an American actor. He won the Academy Award for Best Actor "
                "for Philadelphia and Forrest Gump.", True, "q2"),
        Passage("Forrest Gump",
                "Forrest Gump is a 1994 American film directed by Robert Zemeckis, starring "
                "Tom Hanks. It won the Academy Award for Best Picture.", True, "q2"),
        Passage("Robert Zemeckis",
                "Robert Zemeckis is an American film director known for Forrest Gump and "
                "Back to the Future.", False, "q2"),
        Passage("Marie Curie",
                "Marie Curie was a physicist who won two Nobel Prizes in Physics and "
                "Chemistry. Marie Curie was raised in Poland.", True, "q3"),
        Passage("Poland",
                "Poland is a country in Central Europe. The capital of Poland is Warsaw.",
                True, "q3"),
        Passage("Warsaw",
                "Warsaw is the largest city of Poland and a center of science and culture.",
                False, "q3"),
    ]
    examples = [
        QAExample("q1", "Who directed the film that won Best Picture, Parasite?",
                  "Bong Joon-ho", gold_titles=["Parasite (film)", "Bong Joon-ho"],
                  passages=passages[:4], num_hops=2, qtype="bridge", dataset="toy"),
        QAExample("q2", "Which director made the Best Picture film starring Tom Hanks "
                        "that he won Best Actor for?",
                  "Robert Zemeckis", gold_titles=["Tom Hanks", "Forrest Gump"],
                  passages=passages[4:7], num_hops=3, qtype="bridge", dataset="toy"),
        QAExample("q3", "In which city was the physicist who won two Nobel Prizes born?",
                  "Warsaw", answer_aliases=["Warsaw, Poland"],
                  gold_titles=["Marie Curie", "Warsaw"],
                  passages=passages[7:], num_hops=2, qtype="bridge", dataset="toy"),
    ]
    return examples, build_corpus(examples, name="toy")


# ---------------------------------------------------------------------------
# 디스크 직렬화 (스크립트 단계 간 공유: 01_build_corpus → 02_index_kg → 03_run)
# ---------------------------------------------------------------------------
def save_examples(examples: List[QAExample], path: str) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in examples], f, ensure_ascii=False, indent=2)


def load_examples(path: str) -> List[QAExample]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return [QAExample.from_dict(d) for d in json.load(f)]


def save_corpus(corpus: Corpus, path: str) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corpus.to_dict(), f, ensure_ascii=False, indent=2)


def load_corpus(path: str) -> Corpus:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return Corpus.from_dict(json.load(f))
