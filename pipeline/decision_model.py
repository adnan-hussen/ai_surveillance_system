"""System-One Multimodal Decision Model Wrapper for LiquidAI/d1-3B."""

from __future__ import annotations
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from PIL import Image

from config import PipelineConfig

logger = logging.getLogger(__name__)


def _select_d1_dtype(torch_mod, device: str):
    if device == "cpu":
        return torch_mod.float32
    if device == "mps":
        # Apple Silicon (M1/M2/M3/M4)
        return torch_mod.float16
    if device == "cuda" and torch_mod.cuda.get_device_capability()[0] >= 8:
        return torch_mod.bfloat16
    return torch_mod.float16


class DecisionModelEngine:
    """Thread-safe wrapper around LiquidAI/d1-3B System-One zero-token decision model.

    Supports both:
      1. `model.engine.probabilities(state, rules, [image])`
      2. `model.system_one(state=state, questions={"q": rule}, images=[image])`
    All GPU calls are serialized on a single worker thread so CUDA graphs remain thread-bound.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.engine = None
        self.device = "cpu"
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="d1_3b_worker")

    def load(self) -> None:
        """Loads LiquidAI/d1-3B on the single worker thread and runs warm-up passes."""
        self._worker.submit(self._load_sync).result()

    def _load_sync(self) -> None:
        import torch
        from transformers import AutoModel

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
        dtype = _select_d1_dtype(torch, self.device)
        logger.info(
            "Loading System-One model %s on %s (dtype=%s)...",
            self.config.decision_model_id,
            self.device,
            dtype,
        )
        self.model = AutoModel.from_pretrained(
            self.config.decision_model_id,
            trust_remote_code=True,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()

        if self.config.compile_decision_model and self.device == "cuda":
            logger.info("Compiling d1-3B with CUDA graphs (mode='reduce-overhead')...")
            self.model.compile(mode="reduce-overhead")

        self.engine = getattr(self.model, "engine", None)
        logger.info("LiquidAI/d1-3B System-One engine ready.")


    def _infer_sync(self, image: Image.Image, rule_spec: dict, state: dict | None = None) -> tuple[float, float, float]:
        """Executes a single forward pass without token generation.

        Returns:
            (match_probability, negative_probability, latency_ms)
        """
        t0 = time.perf_counter()
        img = image.convert("RGB")
        img.thumbnail((self.config.decision_input_side, self.config.decision_input_side))

        import torch

        with torch.no_grad():
            if self.engine is not None and hasattr(self.engine, "probabilities"):
                probs_list = self.engine.probabilities(state, [rule_spec], [img])
                p_match = float(probs_list[0][0])
                p_neg = float(probs_list[0][1]) if len(probs_list[0]) > 1 else 1.0 - p_match
            elif hasattr(self.model, "system_one"):
                out = self.model.system_one(
                    state=state or {},
                    questions={"q": rule_spec},
                    images=[img],
                )
                ans = out["answers"]["q"]
                probs_map = ans.get("probabilities", {})
                if "Yes" in probs_map:
                    p_match = float(probs_map["Yes"])
                    p_neg = float(probs_map.get("No", 1.0 - p_match))
                elif "noul" in ans:
                    p_match = float(ans["noul"])
                    p_neg = 1.0 - p_match
                else:
                    p_match = float(ans.get("confidence", 0.0))
                    p_neg = 1.0 - p_match
            else:
                raise RuntimeError("Loaded model exposes neither .engine.probabilities nor .system_one")

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        return p_match, p_neg, ms

    def evaluate_subject_crop(
        self,
        padded_crop: Image.Image,
        detection_rule: str,
        camera_id: str,
        track_id: int,
    ) -> dict[str, Any]:
        """Evaluates a context-padded subject crop against the active detection rule in one pass."""
        rule_spec = {"type": "noul", "instructions": detection_rule.strip()}
        state = {"camera_id": camera_id, "track_id": track_id}
        p_match, p_neg, latency_ms = self._worker.submit(
            self._infer_sync, padded_crop, rule_spec, state
        ).result()
        is_triggered = p_match >= self.config.decision_threshold
        return {
            "detection_rule": detection_rule,
            "type": "noul",
            "match_probability": round(p_match, 4),
            "negative_probability": round(p_neg, 4),
            "confidence": round(max(p_match, p_neg), 4),
            "is_triggered": is_triggered,
            "threshold": self.config.decision_threshold,
            "latency_ms": round(latency_ms, 2),
        }
