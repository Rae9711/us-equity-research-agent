"""Job: refresh the background news cache (manual / future cron use).

ADVISORY ONLY — 模拟交易 · 不构成投资建议.

Usage:
  python -m src.jobs.news_cache_refresh [--force]

Without --force this is a no-op unless the cache exists and is stale
(older than NEWS_CACHE_REFRESH_MIN_MINUTES); with --force it always
collects. Never raises on collection failure: the previous cache stays
in place and the error is recorded in the cache file.
"""

from __future__ import annotations

import argparse
import json
import logging

from src.paper.news_cache import (
    NEWS_CACHE_REFRESH_MIN_MINUTES,
    refresh_news_cache,
    refresh_news_cache_if_stale,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the background news cache")
    parser.add_argument("--force", action="store_true", help="Collect even when the cache is fresh")
    args = parser.parse_args()
    if args.force:
        cache = refresh_news_cache()
        refreshed = True
    else:
        cache, refreshed = refresh_news_cache_if_stale()
    print(json.dumps(
        {
            "refreshed": refreshed,
            "updated_at": cache.get("updated_at"),
            "last_refresh_error": cache.get("last_refresh_error"),
            "tickers": sorted((cache.get("tickers") or {}).keys()),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
