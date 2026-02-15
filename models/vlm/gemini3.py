from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from PIL import Image

from core.base import VLM
from core.registry import register
from core.types import VLMResult


def _parse_json(raw_text: str) -> dict | None:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        if "```" in raw_text:
            json_str = raw_text.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            try:
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                pass
    return None


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H,W,3), got {frame.shape}"
    return Image.fromarray(frame[:, :, ::-1])


def _normalize_generated_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, dict):
        for key in ("text", "output", "response", "generated_text"):
            value = output.get(key)
            if isinstance(value, str):
                return value.strip()
    text = getattr(output, "text", None)
    if isinstance(text, str):
        return text.strip()
    candidates = getattr(output, "candidates", None)
    if candidates:
        parts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    parts.append(part_text.strip())
        if parts:
            return "\n".join(parts).strip()
    return str(output).strip()


@register("vlm", "gemini3")
class Gemini3VLM(VLM):
    """Gemini-backed VLM endpoint.

    Environment:
    - GEMINI_API_KEY or GOOGLE_API_KEY
    - GEMINI_MODEL (optional, default: gemini-2.5-flash)
    """

    def __init__(self, max_new_tokens: int = 512):
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._sdk = None

    def load(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini API key not set. Export GEMINI_API_KEY (or GOOGLE_API_KEY) before loading VLM."
            )

        import google.generativeai as genai

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)
        self._sdk = genai

    def predict(self, frame: np.ndarray, prompt: str) -> VLMResult:
        assert self._model is not None, "Call load() first"
        pil_image = _frame_to_pil(frame).convert("RGB")
        response = self._model.generate_content(
            [prompt, pil_image],
            generation_config=self._sdk.types.GenerationConfig(
                max_output_tokens=self.max_new_tokens,
                temperature=0.1,
            ),
        )
        raw_text = _normalize_generated_text(response)
        return VLMResult(raw_text=raw_text, parsed=_parse_json(raw_text), frame_idx=-1)

    def unload(self) -> None:
        self._model = None
        self._sdk = None

