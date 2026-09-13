# Event-LS point-in-time data contract

The event-enhanced long/short strategy is research-only. A 10% average monthly
return is a stretch KPI, not a forecast or guarantee.

## Required history

Every observation must include `observed_at` (what it describes), `known_at`
(when it became knowable), and the universe membership interval
`[member_from, member_to)`. The immutable partition contract is defined in
`config/pit_contract.yaml`.

Required categories:

- `daily`: at least five years of adjusted OHLCV and corporate actions.
- `intraday`: 1- or 5-minute OHLCV plus a spread proxy at decision boundaries.
- `news`: exact publication and vendor-arrival timestamps.
- `earnings`: report time, actual, contemporaneous consensus and estimate
  revision history.
- `borrow`: locate availability, annualized fee, SSR and forced-cover flags.
- `universe`: historical S&P 500 / Nasdaq 100 membership including removals,
  delistings and IPO dates.
- `macro`: SPY, QQQ, VIX, sector ETFs, rates and release timestamps.

## Validation workflow

1. Load source records into `PITPartitionStore`; published partitions cannot be
   overwritten.
2. Run `audit_rows(rows, decision_at)` before feature construction.
3. Refuse rows with `known_at > decision_at`, invalid membership, duplicate
   identifiers, or missing short-borrow fields.
4. Run `data_gap_report`; any required missing category prevents research
   acceptance.
5. Backtests must set `pit_quality_passed=true`; otherwise the acceptance gate
   remains failed regardless of returns.

The backtest JSON must provide date-keyed `borrow_available`, `borrow_rates`,
`ssr_restricted`, and `forced_cover` maps. Static symbol maps may be used for
diagnostics, but they deliberately fail PIT acceptance because they cannot
prove what was known on each historical date. Missing fees block the affected
short; SSR blocks new or increased shorts; forced-cover flags liquidate an
existing short even during its minimum holding period.

Opening fills use `open_volume` only; completed daily volume is never used for
auction participation. The input must also include a successful
`pit_audit_report` with positive counts for every required category. Sector and
beta metadata are resolved only from prior known bars. Missing held-symbol
marks for more than one session fail PIT and portfolio-control acceptance.

Qualification evidence is imported only by an explicit
`--record-qualification <immutable-id>` CLI option. Reusing an evidence ID with
different metrics is rejected. Passing OOS evidence merely opens the paper
observation stage; it never enables execution.

The OOS report includes monthly returns, drawdown, Sharpe, Calmar, profit
factor, turnover, capacity, long/short contribution, and PIT-classified bull,
bear, high-volatility, and sideways contribution. Missing any required regime
fails candidate acceptance; it is not repaired by retuning the test period.

The strategy never backfills missing values with future revisions, current
index members, or current borrow availability.
