"""L4 Bayesian Driver Engine — updates driver posterior weights each day.

Uses Bayesian update rule:
    posterior(driver) ∝ prior(driver) * likelihood(attribution | driver)

Persists weights to data/bayesian_drivers.json.
No LLM involved.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS: dict[str, float] = {
    "AI": 0.35,
    "Bond": 0.20,
    "Macro": 0.20,
    "Oil": 0.10,
    "Risk": 0.15,
}

_DRIVER_MAP = {
    "AI/Semiconductor": "AI",
    "Bond/Rates": "Bond",
    "Oil/Geo": "Oil",
    "Macro": "Macro",
    "Employment": "Macro",
    "Fed": "Macro",
    "Risk Appetite": "Risk",
    "Unknown": None,
}

_PERSISTENCE_PATH = Path(
    os.environ.get("BAYESIAN_PATH", "/data/bayesian_drivers.json")
)


def _data_path() -> Path:
    return _PERSISTENCE_PATH


def load_weights() -> dict[str, float]:
    p = _data_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load bayesian weights: %s", exc)
    return dict(_DEFAULT_WEIGHTS)


def save_weights(weights: dict[str, float]) -> None:
    p = _data_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(weights, indent=2, ensure_ascii=False), encoding="utf-8")


def update_weights(
    attribution: dict[str, float],
    actual_driver: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Update Bayesian weights given today's attribution and actual driver.

    Returns (prior_weights, posterior_weights).
    """
    prior = load_weights()

    canonical_driver = _DRIVER_MAP.get(actual_driver)
    if canonical_driver is None:
        logger.info("Unknown driver '%s', skipping Bayesian update", actual_driver)
        return prior, prior

    # Likelihood: drivers that contributed heavily get boosted
    attr_by_canonical = {
        "AI": attribution.get("ai", 0.0),
        "Bond": attribution.get("bond", 0.0),
        "Oil": attribution.get("oil", 0.0),
        "Macro": attribution.get("macro", 0.0),
        "Risk": attribution.get("other", 0.0),
    }

    # Posterior ∝ prior * (1 + attribution_weight)
    posterior = {}
    for driver, prior_w in prior.items():
        likelihood_factor = 1.0 + attr_by_canonical.get(driver, 0.0) * 2.0
        if driver == canonical_driver:
            likelihood_factor *= 1.5  # extra boost for confirmed driver
        posterior[driver] = prior_w * likelihood_factor

    # Normalize
    total = sum(posterior.values())
    posterior = {k: round(v / total, 4) for k, v in posterior.items()}

    save_weights(posterior)
    logger.info(
        "Bayesian update: prior=%s posterior=%s (driver=%s)",
        {k: f"{v:.3f}" for k, v in prior.items()},
        {k: f"{v:.3f}" for k, v in posterior.items()},
        actual_driver,
    )
    return prior, posterior


def max_delta(prior: dict[str, float], posterior: dict[str, float]) -> tuple[str, float]:
    """Return (driver, delta) with largest absolute change."""
    deltas = {k: abs(posterior.get(k, 0) - prior.get(k, 0)) for k in prior}
    driver = max(deltas, key=lambda k: deltas[k])
    return driver, deltas[driver]
