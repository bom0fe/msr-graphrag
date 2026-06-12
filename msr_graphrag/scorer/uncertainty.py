"""Training-free uncertainty and confidence utilities for AGS.

The scorer consumes LogitTrace values returned by a backend generation call.
When logits are unavailable, LogitTrace.available is False and AGS returns 0.0;
the metacognitive scorer then renormalizes ECS/PCS weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class LogitTrace:
    """Token-level logit metadata from one generation call."""

    top1_logprobs: List[float] = field(default_factory=list)
    top2_logprobs: List[float] = field(default_factory=list)
    token_entropies: List[float] = field(default_factory=list)
    chosen_logprobs: List[float] = field(default_factory=list)
    available: bool = True

    def __len__(self) -> int:
        return max(
            len(self.top1_logprobs),
            len(self.token_entropies),
            len(self.chosen_logprobs),
        )


def _safe_mean(xs: List[float], default: float = 0.0) -> float:
    return float(np.mean(xs)) if len(xs) else default


def margin_uncertainty(trace: LogitTrace, prefix_k: Optional[int] = None) -> float:
    """Return the average top-1/top-2 logprob margin."""
    if not trace.available or not trace.top1_logprobs:
        return 0.0
    n = len(trace.top1_logprobs)
    k = n if prefix_k is None else min(prefix_k, n)
    gaps = []
    for i in range(k):
        t1 = trace.top1_logprobs[i]
        t2 = trace.top2_logprobs[i] if i < len(trace.top2_logprobs) else (t1 - 10.0)
        gaps.append(max(t1 - t2, 0.0))
    return _safe_mean(gaps)


def entropy_uncertainty(trace: LogitTrace, prefix_k: Optional[int] = None) -> float:
    """Return average token entropy over the draft prefix."""
    if not trace.available or not trace.token_entropies:
        return 0.0
    n = len(trace.token_entropies)
    k = n if prefix_k is None else min(prefix_k, n)
    return _safe_mean(trace.token_entropies[:k])


def nll_uncertainty(trace: LogitTrace) -> float:
    """Return normalized negative log likelihood for the chosen draft tokens."""
    if not trace.available or not trace.chosen_logprobs:
        return 0.0
    return -_safe_mean(trace.chosen_logprobs)


def confidence_from_margin(margin: float, lam: float = 0.5) -> float:
    """Map a nonnegative margin to confidence in [0, 1]."""
    return float(1.0 - math.exp(-max(lam, 1e-6) * max(margin, 0.0)))


def confidence_from_entropy(entropy: float, h_max: float = 4.0) -> float:
    """Map entropy to confidence in [0, 1]."""
    return float(1.0 - min(max(entropy, 0.0) / max(h_max, 1e-6), 1.0))


def confidence_from_nll(nll: float, scale: float = 1.5) -> float:
    """Map normalized NLL to confidence in [0, 1]."""
    return float(math.exp(-max(nll, 0.0) / max(scale, 1e-6)))


def answer_generability(
    trace: LogitTrace,
    signal: str = "margin",
    lam: float = 0.5,
    h_max: float = 4.0,
    nll_scale: float = 1.5,
    prefix_k: Optional[int] = None,
) -> float:
    """Return AGS confidence in [0, 1] from the selected logit signal."""
    if not trace.available:
        return 0.0
    if signal == "margin":
        return confidence_from_margin(margin_uncertainty(trace, prefix_k), lam)
    if signal == "entropy":
        return confidence_from_entropy(entropy_uncertainty(trace, prefix_k), h_max)
    if signal == "nll":
        return confidence_from_nll(nll_uncertainty(trace), nll_scale)
    if signal == "margin_entropy":
        c_m = confidence_from_margin(margin_uncertainty(trace, prefix_k), lam)
        c_h = confidence_from_entropy(entropy_uncertainty(trace, prefix_k), h_max)
        return 0.5 * (c_m + c_h)
    raise ValueError(f"unknown AGS signal: {signal}")
