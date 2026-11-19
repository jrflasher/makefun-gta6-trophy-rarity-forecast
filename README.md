# GTA VI Trophy Rarity Forecast — make.fun `gta6-trophy-rarity-forecast-01`

Entry for the [make.fun](https://make.fun) bounty
[`gta6-trophy-rarity-forecast-01`](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its
Exophase earn rate at 2026-11-26T00:00:00Z, as five quantiles per trophy
(5%, 25%, 50%, 75%, 95%), scored by mean pinball loss on the logit
scale.

## Method — refresh inversion in logit space

Exophase does not update its counts continuously: the page sits
unchanged for hours, then refreshes in one jump (launch night: 638
players tracked for ~4.5 h, then 3,868 in a single refresh, every rate
but the 100% one falling). A rate is a ratio, `rate = earners /
players-tracked`, so the forecast is built from the two counts
separately:

1. The case history is collapsed into distinct snapshots. Between two
   snapshots the newly tracked cohort's earn fraction per trophy is
   recovered exactly: `c = (E_new − E_old) / (N_new − N_old)`.
2. Per trophy, the logit of the rate is fitted against
   `ln(1 + hours since 2026-11-19T00:00Z)` over the snapshots
   (weighted least squares, 12-hour recency half-life), anchored at the
   last read and read at the target. Slopes are shrunk toward their
   tier group and the extrapolation is damped (progression saturates).
3. Flat history (one snapshot, as in the practice cases): no trend can
   be fitted, so the documented first refresh is the prior — one
   refresh before the target multiplies players ~6× and the new cohort
   earns each trophy at ~0.15× the incumbent rate (the refresh added
   3,230 players and 3,333 trophies, almost all of them the
   first-mission trophy). Earners and players are projected through
   the refresh separately, then divided.
4. Spread: a logit fan per trophy combining the random-walk term
   (0.5·√days), the slope's standard error, and refresh-jump risk from
   the case's own history; asymmetric (wider downward when the trend
   is flat/falling, wider upward when it is rising). The 100% trophy
   (first mission) is forecast at 100.
5. Story-funnel monotonicity: Chapter 1 ≥ Chapter 2 ≥ Chapter 3 ≥
   Chapter 4 ≥ finale ≥ platinum, enforced on every quantile.

## Backtests (local, practice answers reconstructed from the rules)

The practice answers are not in the kit, so the refresh was
reconstructed from the totals the rules document (638 → 3,868
players, 943 → 4,276 trophies earned). Two reconstructions bracket
the truth; both reproduce the documented persistence-baseline
practice loss (~1.51) and typical miss (~0.68 pp):

| proxy answer | this entry | persistence baseline | ratio |
|---|---|---|---|
| dilution-only | 0.318 | 1.688 | 0.19 |
| dilution + incumbent progress | 0.426 | 1.518 | 0.28 |

Synthetic rolling-origin backtest (4 checkpoints, horizons 12–72 h,
refreshes + progression): mean loss 0.234 vs persistence 0.301
(ratio 0.78), 90% coverage 1.00.

Full notes and the backtest script are in the build notes; see
`backtest.py` there.

## How it runs

```sh
python3 main.py   # reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR (default /output)
```

Python 3.13, standard library only, no network, deterministic —
two runs on the same case write identical bytes, in ~0.06 s per
case (limit 8 s).

## Licence

MIT — see `LICENSE`.
