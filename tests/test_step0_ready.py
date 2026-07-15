"""Step0 data_ready: hard misses vs soft checklist/warnings."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.collectors.step0 import _flatten_missing, _soft_checklist_warnings


class SoftNewsChecklistTests(unittest.TestCase):
    def test_bloomberg_wsj_are_skipped_from_hard_missing(self) -> None:
        checklist = {
            "polygon_news": True,
            "bloomberg": False,
            "wsj": False,
        }
        self.assertEqual(_flatten_missing(checklist, "news"), [])

    def test_bloomberg_emits_soft_warning_only(self) -> None:
        checklist = {
            "news": {
                "polygon_news": True,
                "bloomberg": False,
                "wsj": True,
            }
        }
        soft = _soft_checklist_warnings(checklist)
        self.assertEqual(len(soft), 1)
        self.assertTrue(soft[0].startswith("⚠ news.bloomberg"))

    def test_polygon_still_hard_missing(self) -> None:
        checklist = {
            "polygon_news": False,
            "bloomberg": False,
            "wsj": False,
        }
        self.assertEqual(_flatten_missing(checklist, "news"), ["news.polygon_news"])


class DataReadyPartitionTests(unittest.TestCase):
    def test_soft_prefix_does_not_block(self) -> None:
        missing = [
            "⚠ news.bloomberg: RSS 不可用（可选源，不阻断就绪）",
            "⚠ macro.DGS10: FRED 最新 2026-07-13，上一交易日 2026-07-14（FRED 正常滞后）",
        ]
        hard = [m for m in missing if not str(m).startswith("⚠")]
        self.assertTrue(len(hard) == 0)


if __name__ == "__main__":
    unittest.main()
