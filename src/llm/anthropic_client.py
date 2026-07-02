from __future__ import annotations

import json
import os
import re
from typing import Any

import anthropic


class AnthropicClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get(
            "ANTHROPIC_MODEL", "claude-sonnet-5"
        )
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete_json(self, system: str, user: str, max_tokens: int = 16384) -> dict[str, Any]:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        try:
            return _parse_json(text)
        except json.JSONDecodeError:
            repair = self.client.messages.create(
                model=self.model,
                max_tokens=16384,
                system="Fix the JSON below. Output ONLY valid JSON, no commentary.",
                messages=[{"role": "user", "content": text}],
            )
            repair_text = "".join(
                block.text for block in repair.content if block.type == "text"
            )
            return _parse_json(repair_text)


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
