from __future__ import annotations

import base64
import json
import os
from typing import Any
from urllib import error, request

import cv2
import numpy as np

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


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _frame_to_data_url(frame: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame for LM Studio request.")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@register("vlm", "lmstudio-cosmos-reason2-8b")
class LMStudioCosmosReason2VLM(VLM):
    """LM Studio OpenAI-compatible backend for NVIDIA Cosmos-Reason2-8B.

    Env vars:
    - LMSTUDIO_BASE_URL (default: http://127.0.0.1:1234/v1)
    - LMSTUDIO_API_KEY (default: lm-studio)
    - LMSTUDIO_MODEL (default: nvidia/Cosmos-Reason2-8B)
    """

    def __init__(self, max_new_tokens: int = 512, timeout_s: int = 120):
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s
        self._base_url = ""
        self._api_key = ""
        self._model_name = ""

    def load(self) -> None:
        self._base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
        self._api_key = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
        self._model_name = os.getenv("LMSTUDIO_MODEL", "nvidia/Cosmos-Reason2-8B")

    def predict(self, frame: np.ndarray, prompt: str) -> VLMResult:
        if not self._base_url:
            raise RuntimeError("Call load() before predict().")

        image_url = _frame_to_data_url(frame)
        payload = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": self.max_new_tokens,
        }

        req = request.Request(
            url=f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LM Studio HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(
                "LM Studio request failed. Ensure LM Studio server is running and a vision-capable model is loaded."
            ) from exc

        data = json.loads(body)
        choices = data.get("choices", [])
        if not choices:
            raw_text = ""
        else:
            message = choices[0].get("message", {})
            raw_text = _normalize_content(message.get("content", ""))

        return VLMResult(raw_text=raw_text, parsed=_parse_json(raw_text), frame_idx=-1)

    def unload(self) -> None:
        self._base_url = ""
        self._api_key = ""
        self._model_name = ""


@register("vlm", "lmstudio-gemma-3n-e4b")
class LMStudioGemma3nE4BVLM(LMStudioCosmosReason2VLM):
    """LM Studio OpenAI-compatible backend for Google Gemma 3n e4b."""

    def load(self) -> None:
        self._base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
        self._api_key = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
        self._model_name = os.getenv("LMSTUDIO_MODEL", "google/gemma-3n-e4b")
