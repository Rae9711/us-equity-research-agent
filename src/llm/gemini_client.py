from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

import httpx


def _vault_surrogate_key() -> str:
    """Surrogate for the Secure Vault `custom.gemini` credential.

    The real API key never lands on disk: authd hands out an ``hsurr:``
    surrogate and the egress layer swaps in the real key on requests to
    generativelanguage.googleapis.com. Returns "" when the vault entry
    is unavailable.
    """
    try:
        bin_dir = "/opt/muse/skills/custom-connector-setup/bin"
        if bin_dir not in sys.path:
            sys.path.insert(0, bin_dir)
        from dynamic_credentials import dynamic_credential_entry

        entry = dynamic_credential_entry("custom.gemini")
        return str(entry.get("surrogate", "")).strip()
    except Exception:
        return ""


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _extract_text(result: dict[str, Any]) -> str:
    chunks: list[str] = []
    for cand in result.get("candidates", []) or []:
        content = cand.get("content") or {}
        for part in content.get("parts", []) or []:
            if part.get("text"):
                chunks.append(part["text"])
    return "".join(chunks)


class GeminiClient:
    """Gemini GenerateContent client with the same complete_json interface
    as AnthropicClient. Shadow-mode second opinion for morning research:
    it scores the same prompt but never drives sizing or entries."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = (
            api_key or os.environ.get("GEMINI_API_KEY", "") or _vault_surrogate_key()
        )
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

    def _generate(self, prompt: str, max_tokens: int, json_mode: bool) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent"
        )
        body: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=120) as client:
                    resp = client.post(
                        url,
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": self.api_key,
                        },
                        json=body,
                    )
                    resp.raise_for_status()
                    return _extract_text(resp.json())
            except httpx.HTTPStatusError as exc:
                last_error = exc
                # Gemini 503s on demand spikes; shadow mode can afford one
                # short retry before giving up for the day.
                if exc.response.status_code == 503 and attempt < 2:
                    time.sleep(10)
                    continue
                raise
        raise last_error  # pragma: no cover - loop always returns or raises

    def complete_json(self, system: str, user: str, max_tokens: int = 16384) -> dict[str, Any]:
        prompt = f"{system}\n\n{user}"
        text = self._generate(prompt, max_tokens, json_mode=True)
        try:
            return _parse_json(text)
        except json.JSONDecodeError:
            repair = self._generate(
                "Fix the JSON below. Output ONLY valid JSON, no commentary.\n\n" + text,
                16384,
                json_mode=True,
            )
            return _parse_json(repair)
