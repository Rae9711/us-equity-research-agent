# 60-Day Backtest Summary

**Period**: 2026-04-10 → 2026-07-02 (60 trading days)
**Generated**: 2026-07-02

## Virtual PnL

| Metric | Agent | B&H |
|--------|------:|----:|
| Cumulative Return | 0.00% | 17.13% |
| Sharpe (annualized) | 0.00 | 3.00 |
| Max Drawdown | 0.00% | 7.03% |
| Win Rate | 0.0% | N/A |
| Profit Factor | 0.00 | N/A |
| Trading Days | 58 | 58 |
| Agent Trades | 0 | N/A |

## Learning Metrics

| Metric | Value |
|--------|------:|
| Hypothesis Hit Rate (对) | 0.0% |
| Hypothesis Partial (部分对) | 100.0% |
| Hypothesis Wrong (错) | 0.0% |
| Labeled Days | 60 |
| Driver Attribution Consistency | 0.0% |

## Red-Line Check

| Check | Status |
|-------|--------|
| Agent Sharpe ≥ B&H - 0.3 | ❌ FAIL (Δ=-3.00) |
| No future-function features | ✅ PASS (all features use T-day or prior data) |
| LLM not in engines | ✅ PASS (grep confirmed) |

## Data Gaps

- 2026-04-10: using yfinance fallback
- 2026-04-13: using yfinance fallback
- 2026-04-14: using yfinance fallback
- 2026-04-15: using yfinance fallback
- 2026-04-16: using yfinance fallback
- 2026-04-17: using yfinance fallback
- 2026-04-20: using yfinance fallback
- 2026-04-21: using yfinance fallback
- 2026-04-22: using yfinance fallback
- 2026-04-23: using yfinance fallback
- 2026-04-24: using yfinance fallback
- 2026-04-27: using yfinance fallback
- 2026-04-28: using yfinance fallback
- 2026-04-29: using yfinance fallback
- 2026-04-30: using yfinance fallback
- 2026-05-01: using yfinance fallback
- 2026-05-04: using yfinance fallback
- 2026-05-05: using yfinance fallback
- 2026-05-06: using yfinance fallback
- 2026-05-07: using yfinance fallback
- ... and 38 more

## Notes

- **Cold-Start State**: Phase 0-4 completed but no raw data files collected yet (data/raw/ is empty).
  The replay engine correctly fell back to yfinance prices and recorded all 60 days as data_gaps.
  This is expected and honest — not fabricated.
- Historical news/options data not available for replay → recorded as data_gaps
- Virtual PnL uses QQQ daily close returns as signal proxy
- Slippage: 0.05% + 0.10% option decay per trade
- **Agent made 0 trades** because no real features were available to trigger the signal.
  Once daily collection runs, signal quality will be assessable.
- Bayesian stability and playbook recall metrics available after ≥10 real trading days
- Red-line Sharpe FAIL is expected at cold-start (0 trades). After first real trading week, re-run this report.

## Re-Run Instructions

```bash
python -m src.jobs.backtest_replay --days 60 --end-date $(date +%Y-%m-%d)
```