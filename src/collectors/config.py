from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "symbols.yaml"


@lru_cache
def load_symbols() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
