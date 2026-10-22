"""
OmniCognitive V5 — Adaptive Information-Theoretic Logits Processor.

Novel mechanisms (patent-pending):
  [1] Prompt-Conditioned Threshold Calibration
  [2] Vocabulary-Partitioned Min-P
  [3] Chaos Memory Bank (Retrospective Novelty Injection)
  [4] Smooth EMA Chaos Feedback

Author : [YOUR_NAME]
License: Apache 2.0 (code) / Patent Pending (methods)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import torch
from transformers import LogitsProcessor

logger = logging.getLogger("omnicognitive")

# ============================================================================
# Presets
# ============================================================================

PRESETS: Dict[str, Dict[str, Any]] = {
    "calm": {
        "base_threshold": 1.6, "alpha": 0.10, "varentropy_weight": 0.10,
        "k_min": 2, "k_max": 24, "min_p": 0.05,
        "temp_min": 0.20, "cooling_rate": 0.45,
        "chaos_streak_limit": 2, "chaos_entropy_margin": 0.4,
    },
    "balanced": {
        "base_threshold": 1.2, "alpha": 0.08, "varentropy_weight": 0.15,
        "k_min": 4, "k_max": 40, "min_p": 0.08,
        "temp_min": 0.35, "cooling_rate": 0.30,
        "chaos_streak_limit": 3, "chaos_entropy_margin": 0.5,
    },
    "creative": {
        "base_threshold": 0.9, "alpha": 0.05, "varentropy_weight": 0.25,
        "k_min": 8, "k_max": 64, "min_p": 0.03,
        "temp_min": 0.50, "cooling_rate": 0.18,
        "chaos_streak_limit": 5, "chaos_entropy_margin": 0.8,
    },
}

# ============================================================================
# Telemetry
# ============================================================================

def _pct(sorted_vals: List[float], p: float) -> float:
    n = len(sorted_vals)
    return float(sorted_vals[min(int(n * p), n - 1)]) if n else 0.0


@dataclass(slots=True)
class Telemetry:
    """Per-step diagnostics. All lists stay in lockstep (batch-aggregated)."""
    entropies: List[float] = field(default_factory=list)
    varentropies: List[float] = field(default_factory=list)
    effective_temps: List[float] = field(default_factory=list)
    surviving_tokens: List[float] = field(default_factory=list)
    chaos_emas: List[float] = field(default_factory=list)
    novelty_injections: List[int] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        n = len(self.entropies)
        if n == 0:
            return {}
        toks = sorted(self.surviving_tokens)
        mh, mv = sum(self.entropies) / n, sum(self.varentropies) / n
        mt, mc = sum(self.effective_temps) / n, sum(self.chaos_emas) / n
        q = ("High-Chaos-Exploratory" if mh > 2.2 and mv > 1.2
             else "High-Entropy-Uniform" if mh > 2.2
             else "High-Variance-Bimodal" if mv > 1.2
             else "Stable-Convergent")
        return {
            "total_steps": n,
            "mean_entropy_nats": round(mh, 4),
            "mean_varentropy": round(mv, 4),
            "mean_effective_temp": round(mt, 4),
            "mean_chaos_ema": round(mc, 4),
            "novelty_injections": sum(self.novelty_injections),
            "structural_quadrant": q,
            "tokens_p50": _pct(toks, 0.50),
            "tokens_p95": _pct(toks, 0.95),
            "tokens_p99": _pct(toks, 0.99),
        }

    def reset(self) -> None:
        for lst in (self.entropies, self.varentropies, self.effective_temps,
                    self.surviving_tokens, self.chaos_emas, self.novelty_injections):
            lst.clear()

# ============================================================================
# Main processor
# ============================================================================

class OmniCognitiveV5(LogitsProcessor):
    """
    Adaptive logits processor with four novel mechanisms.

    See README for full documentation.
    """

    def __init__(
        self,
        prompt_len: int,
        preset: Optional[str] = None,
        prompt_entropy_ref: float = 2.5,
        prompt_gain: float = 0.4,
        function_token_mask: Optional[torch.Tensor] = None,
        function_min_p_scale: float = 0.4,
        bank_size: int = 8,
        novelty_lambda: float = 0.25,
        repetition_entropy_floor: float = 0.8,
        repetition_patience: int = 4,
        chaos_rho: float = 0.85,
        chaos_gain: float = 0.6,
        chaos_cap: float = 4.0,
        base_threshold: Optional[float] = None,
        alpha: Optional[float] = None,
        varentropy_weight: Optional[float] = None,
        k_min: Optional[int] = None,
        k_max: Optional[int] = None,
        min_p: Optional[float] = None,
        temp_min: Optional[float] = None,
        cooling_rate: Optional[float] = None,
        decay_lambda: float = 0.45,
        repetition_penalty: float = 1.15,
        frequency_penalty: float = 0.15,
        presence_penalty: float = 0.20,
        dry_n: float = 4.0,
        chaos_streak_limit: Optional[int] = None,
        chaos_entropy_margin: Optional[float] = None,
        reuse_buffers: bool = True,
        record_telemetry: bool = True,
    ) -> None:
        cfg = dict(PRESETS["balanced"]) if preset is None else dict(
            PRESETS.get(preset, PRESETS["balanced"]))
        if preset is not None and preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Choose from {list(PRESETS)}")

        for k, v in {
            "base_threshold": base_threshold, "alpha": alpha,
            "varentropy_weight": varentropy_weight, "k_min": k_min, "k_max": k_max,
            "min_p": min_p, "temp_min": temp_min, "cooling_rate": cooling_rate,
            "chaos_streak_limit": chaos_streak_limit,
            "chaos_entropy_margin": chaos_entropy_margin,
        }.items():
            if v is not None:
                cfg[k] = v

        # Validation
        if not isinstance(prompt_len, int) or prompt_len < 0:
            raise ValueError(f"prompt_len must be non-negative int, got {prompt_len!r}")
        if cfg["k_min"] < 1 or cfg["k_max"] < cfg["k_min"]:
            raise ValueError(f"Require 1 <= k_min <= k_max")
        if not (0.0 < cfg["min_p"] <= 1.0):
            raise ValueError(f"min_p must be in (0, 1]")
        if not (0.0 < cfg["temp_min"] <= 1.0):
            raise ValueError(f"temp_min must be in (0, 1]")
        if not (0.0 < chaos_rho < 1.0):
            raise ValueError(f"chaos_rho must be in (0, 1)")
        if novelty_lambda < 0 or function_min_p_scale <= 0 or repetition_penalty <= 0 or bank_size < 0:
            raise ValueError("Invalid novel-mechanism hyperparameter")

        self.prompt_len = prompt_len
        self.base_threshold = float(cfg["base_threshold"])
        self.alpha = float(cfg["alpha"])
        self.gamma = float(cfg["varentropy_weight"])
        self.k_min, self.k_max = int(cfg["k_min"]), int(cfg["k_max"])
        self.min_p = float(cfg["min_p"])
        self.temp_min = float(cfg["temp_min"])
        self.beta = float(cfg["cooling_rate"])
        self.decay_lambda = float(decay_lambda)
        self.rep_penalty = float(repetition_penalty)
        self.freq_penalty = float(frequency_penalty)
        self.pres_penalty = float(presence_penalty)
        self.dry_n = float(dry_n)
        self.chaos_limit = int(cfg["chaos_streak_limit"])
        self.chaos_entropy_margin = float(cfg["chaos_entropy_margin"])

        self.prompt_entropy_ref = float(prompt_entropy_ref)
        self.prompt_gain = float(prompt_gain)
        self._calibrated_threshold: Optional[float] = None

        self.function_token_mask = function_token_mask
        self.function_min_p_scale = float(function_min_p_scale)

        self.bank_size = int(bank_size)
        self.novelty_lambda = float(novelty_lambda)
        self.repetition_entropy_floor = float(repetition_entropy_floor)
        self.repetition_patience = int(repetition_patience)
        self._chaos_bank: Deque[torch.Tensor] = deque(maxlen=bank_size)
        self._repetition_streak: Optional[torch.Tensor] = None

        self.chaos_rho = float(chaos_rho)
        self.chaos_gain = float(chaos_gain)
        self.chaos_cap = float(chaos_cap)
        self._chaos_ema: Optional[torch.Tensor] = None

        self.reuse_buffers = bool(reuse_buffers)
        self.record_telemetry = bool(record_telemetry)
        self.telemetry = Telemetry()
        self._freq_buf: Optional[torch.Tensor] = None
        self._freq_buf_key: Optional[tuple] = None
        self._novelty_injections_this_step = 0

    def calibrate_from_prompt(self, prompt_logits: torch.Tensor) -> float:
        """[NOVEL #1] Call once after prefill. Returns calibrated threshold."""
        lp = torch.log_softmax(prompt_logits.float(), dim=-1)
        H = float(-(lp.exp() * lp).sum().item())
        offset = self.prompt_gain * math.tanh((H - self.prompt_entropy_ref) / 2.0)
        self._calibrated_threshold = self.base_threshold + offset
        logger.debug("Calibrated threshold: %.4f (H_prompt=%.3f)",
                     self._calibrated_threshold, H)
        return self._calibrated_threshold

    def reset_state(self) -> None:
        self._chaos_bank.clear()
        self._chaos_ema = None
        self._repetition_streak = None
        self._freq_buf = None
        self._freq_buf_key = None
        self._novelty_injections_this_step = 0
        self.telemetry.reset()

    def get_telemetry_summary(self) -> Dict[str, Any]:
        return self.telemetry.summary()

    def _ensure_buffers(self, B: int, device: torch.device) -> None:
        if self._chaos_ema is None or self._chaos_ema.shape[0] != B or self._chaos_ema.device != device:
            self._chaos_ema = torch.zeros(B, device=device, dtype=torch.float32)
        if self._repetition_streak is None or self._repetition_streak.shape[0] != B or self._repetition_streak.device != device:
            self._repetition_streak = torch.zeros(B, device=device, dtype=torch.long)

    def _get_freq_buf(self, B: int, V: int, device: torch.device) -> torch.Tensor:
        key = (B, V, str(device))
        if self._freq_buf is None or self._freq_buf_key != key:
            self._freq_buf = torch.zeros((B, V), dtype=torch.float32, device=device)
            self._freq_buf_key = key
        else:
            self._freq_buf.zero_()
        return self._freq_buf

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device, orig_dtype = scores.device, scores.dtype
        B, V = scores.shape
        self._novelty_injections_this_step = 0

        if self.prompt_len > input_ids.shape[-1]:
            raise ValueError(f"prompt_len={self.prompt_len} > input length {input_ids.shape[-1]}")

        self._ensure_buffers(B, device)
        history = input_ids[:, self.prompt_len:]

        if history.shape[1] == 0:
            self._chaos_ema.zero_()
            self._repetition_streak.zero_()
            self._chaos_bank.clear()

        s = scores.float()

        # 1) Classic penalties
        if history.shape[1] > 0:
            freq = (self._get_freq_buf(B, V, device) if self.reuse_buffers
                    else torch.zeros((B, V), dtype=torch.float32, device=device))
            freq.scatter_add_(1, history, torch.ones_like(history, dtype=torch.float32))
            appeared = freq > 0.0
            s = torch.where(appeared,
                            torch.where(s > 0, s / self.rep_penalty, s * self.rep_penalty),
                            s)
            s = s - freq * self.freq_penalty - appeared.to(torch.float32) * self.pres_penalty

        # 2) Entropy + varentropy
        log_probs = torch.log_softmax(s, dim=-1)
        if not torch.isfinite(log_probs).all():
            logger.warning("Non-finite log-probs; passthrough.")
            return scores
        probs = log_probs.exp()
        valid = probs > 0.0
        surprisal = torch.where(valid, -log_probs, torch.zeros_like(log_probs))
        entropy = (probs * surprisal).sum(dim=-1)
        centered = torch.where(valid, surprisal - entropy.unsqueeze(-1), torch.zeros_like(surprisal))
        varentropy = (probs * centered * centered).sum(dim=-1)

        # 3) Dynamic threshold
        gen_steps = max(1, input_ids.shape[-1] - self.prompt_len)
        base = self._calibrated_threshold if self._calibrated_threshold is not None else self.base_threshold
        dynamic_threshold = base + self.alpha * math.log1p(gen_steps)
        excess = torch.clamp(entropy - dynamic_threshold, min=0.0)
        instability = excess * (1.0 + self.gamma * varentropy)

        # 4) Smooth EMA chaos feedback [NOVEL #4]
        self._chaos_ema = self.chaos_rho * self._chaos_ema + (1.0 - self.chaos_rho) * excess
        chaos_modifier = torch.exp(self.chaos_gain * self._chaos_ema.clamp(max=self.chaos_cap))
        in_hard_chaos = excess > self.chaos_entropy_margin
        self._repetition_streak = torch.where(
            in_hard_chaos,
            torch.zeros_like(self._repetition_streak),
            self._repetition_streak + (entropy < self.repetition_entropy_floor).long(),
        )

        # 5) Adaptive temperature
        effective_temp = torch.clamp(
            1.0 / (1.0 + self.beta * instability * chaos_modifier),
            min=self.temp_min, max=1.0)
        scaled = s / effective_temp.unsqueeze(-1)

        # 6) Chaos memory bank [NOVEL #3]
        chaos_detected = excess > self.chaos_entropy_margin
        if chaos_detected.any() and self.bank_size > 0:
            for b in range(B):
                if chaos_detected[b].item():
                    self._chaos_bank.append(scaled[b].detach().clone())

        if self._chaos_bank and self.repetition_patience > 0:
            stuck = self._repetition_streak >= self.repetition_patience
            if stuck.any():
                anchor = self._chaos_bank[0]
                for b in range(B):
                    if stuck[b].item():
                        scaled[b] = (1.0 - self.novelty_lambda) * scaled[b] + self.novelty_lambda * anchor
                        self._novelty_injections_this_step += 1
                self._repetition_streak[stuck] = 0

        # 7) Adaptive Top-K + partitioned Min-P [NOVEL #2]
        scaled_lp = torch.log_softmax(scaled, dim=-1)
        max_lp = scaled_lp.amax(dim=-1, keepdim=True)

        if self.function_token_mask is not None and self.function_token_mask.numel() == V:
            fn = self.function_token_mask.to(device)
            thr_c = max_lp + math.log(self.min_p)
            thr_f = max_lp + math.log(self.min_p * self.function_min_p_scale)
            min_p_mask = torch.where(fn.unsqueeze(0), scaled_lp >= thr_f, scaled_lp >= thr_c)
        else:
            min_p_mask = scaled_lp >= (max_lp + math.log(self.min_p))

        uncertain = excess > 0.0
        k_adapt = torch.clamp(
            torch.round(self.k_min + (self.k_max - self.k_min) * torch.exp(-self.decay_lambda * instability)).long(),
            min=self.k_min, max=min(self.k_max, V))
        row_k = torch.where(uncertain, k_adapt, torch.ones_like(k_adapt))
        max_k = min(int(row_k.max().item()), V)

        _, top_idx = torch.topk(scaled, k=max_k, dim=-1)
        ranks = torch.arange(max_k, device=device).unsqueeze(0)
        topk_mask = torch.zeros_like(scaled, dtype=torch.bool)
        topk_mask.scatter_(-1, top_idx, ranks < row_k.unsqueeze(1))

        keep = torch.where(uncertain.unsqueeze(-1), topk_mask & min_p_mask, min_p_mask)
        keep.scatter_(-1, torch.argmax(scaled, dim=-1, keepdim=True), True)
        output = scaled.masked_fill(~keep, float("-inf"))

        # 8) Telemetry
        if self.record_telemetry:
            self.telemetry.entropies.append(float(entropy.mean().item()))
            self.telemetry.varentropies.append(float(varentropy.mean().item()))
            self.telemetry.effective_temps.append(float(effective_temp.mean().item()))
            self.telemetry.surviving_tokens.append(float(keep.sum(dim=-1).float().mean().item()))
            self.telemetry.chaos_emas.append(float(self._chaos_ema.mean().item()))
            self.telemetry.novelty_injections.append(self._novelty_injections_this_step)

        return output.to(orig_dtype)
