# NYSE Holiday Incident — 2026-07-04

## Summary

Independence Day 2026 falls on **Saturday July 4**. NYSE observes the closure on **Friday July 3** (early close / full closure per exchange calendar). The scheduler previously ran Mon–Fri cron jobs without checking NYSE holidays, so jobs could fire on observed holiday sessions.

## Root cause (pre-fix)

- `prior_trading_day()` skipped weekends only, not NYSE holidays
- `trading_day_number()` counted weekdays including holidays → false "Day N" on closed days
- APScheduler cron `mon-fri` fired on 2026-07-03 (observed holiday) and would show progress on 2026-07-04 in UI if artifacts existed

## Fix (this release)

- `src/utils/trading_calendar.py`: NYSE holiday set 2025–2027, `is_trading_day()`, `employment_situation_date()` for BLS NFP moves
- `src/runner/main.py` + `catchup.py`: skip all Step jobs when `not is_trading_day()`
- Web UI: `NYSE 休市` badge, zero step progress on holidays
- Day counter excludes non-session days

## VPS investigation (manual)

After deploy, SSH to VPS and check:

```bash
docker compose -p trading-os -f /opt/trading-os/repo/deploy/docker-compose.yml logs runner | grep -E "2026-07-0[34]|skipped|NYSE"
ls -la /data/raw/2026-07-0*.json /data/reports/2026-07-0*/ 2>/dev/null
```

Expected post-fix: **no new Step 0–8 artifacts** on 2026-07-03/04 unless manually backfilled.

## Related: 2026-07-02 NFP

NFP moved to **Thursday 2026-07-02** because Friday 2026-07-03 is NYSE holiday. Pre-fix system mislabeled FOMC due to undated catalyst extraction from FRED calendar.
