"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its Exophase earn rate at the case's target, as percentages.

Method -- refresh inversion in logit space
------------------------------------------
Exophase does not update its counts continuously: the page sits unchanged
for hours, then refreshes in one jump (launch night: 638 players tracked
for ~4.5 h, then 3,868 in a single refresh, every rate but the 100% one
falling). A rate is a ratio, rate = earners / players-tracked, so the
forecast is built from the two counts separately:

1.  The case history is collapsed into distinct snapshots (a new snapshot
    starts whenever players-tracked or any rate changes). Between two
    snapshots the newly tracked cohort's earn fraction for each trophy is
    recovered exactly: c = (E_new - E_old) / (N_new - N_old), where
    E = rate * N / 100. That cohort rate is the fundamental quantity --
    it is what the next refresh will dilute the page with.

2.  Per trophy, the logit of the rate is fitted against ln(1 + hours
    since the release day began, 2026-11-19T00:00Z) over the snapshots,
    weighted least squares with a 12-hour recency half-life, anchored at
    the last observed read and read at the target. The fitted slope is
    shrunk toward its tier group (platinum/gold/silver/bronze-story/
    bronze-other), because trophies of a kind move together, and clamped
    to a plausible band.

3.  Flat history (only one snapshot, as in the practice cases): no trend
    can be fitted, so the documented first refresh is used as the prior --
    one refresh before the target multiplies players-tracked ~6x and the
    new cohort earns each trophy at ~0.70x the incumbent rate (launch
    night: 943 trophies earned by 638 players, 1.48 each; the refresh
    added 3,230 players and 3,333 trophies, 1.03 each). Earners and
    players are projected through that refresh separately, then divided.

4.  Spread: a logit fan per trophy, combining the random-walk term
    (0.5 * sqrt(days), the house baseline's), the slope's standard error
    times the extrapolation distance, and refresh-jump risk (the median
    absolute logit jump per refresh in the case's own history, times the
    square root of the expected number of refreshes before the target).
    The fan is asymmetric: when the fitted trend is flat or falling the
    lower tail is wider (dilution risk), when it is rising the upper tail
    is wider (progression risk). The 100% trophy (first mission, every
    tracked player holds it) is forecast at 100 with a hair-wide band.

5.  Story-funnel monotonicity: the forecast medians (and every quantile)
    are forced to respect Chapter 1 >= Chapter 2 >= Chapter 3 >=
    Chapter 4 >= finale >= platinum, by pooling any violating run to its
    logit mean.

Standard library only, no network, no randomness: two runs on the same
case write the same bytes. Runtime is a few milliseconds per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Story-funnel chain: chapter trophies in Exophase id order, platinum last.
FUNNEL = [3, 4, 5, 6, 7, 1]
# Tier groups used for slope shrinkage. Bronze is split: story chapters
# move with the story, the rest move with the side content.
STORY_BRONZE = {3, 4, 5, 6}


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(ts):
    return (ts - RELEASE).total_seconds() / 3600.0


def group_of(trophy):
    tier = trophy["tier"]
    if tier == "bronze" and trophy["id"] in STORY_BRONZE:
        return "story"
    return tier


def snapshots(case):
    """Collapse the history into distinct snapshots.

    Returns a list of (time, players_tracked, rates) with consecutive
    identical states merged, keeping the first read's time for a state.
    """
    out = []
    prev_key = None
    for h in case["history"]:
        rates = tuple(h["rates"])
        key = (h["players_tracked"], rates)
        if key == prev_key:
            continue
        prev_key = key
        out.append((parse_ts(h["at"]), h["players_tracked"], rates))
    return out


def wls_line(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, resid_std, se_b)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0, 0.0, None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, len(xs) - 2)
    var = sum(w * r * r for w, r in zip(ws, resid)) / (sw * dof / max(1, len(xs)))
    resid_std = math.sqrt(max(0.0, var))
    se_b = resid_std / math.sqrt(sxx) if sxx > 0 else None
    return a, b, resid_std, se_b


def refresh_projection(last_rate, last_n, horizon_hours):
    """Flat-history prior: project earners and players through refreshes.

    One refresh is expected before the target when the horizon is at
    least an hour. The first refresh multiplies players ~6x with the new
    cohort earning at 0.70x the incumbent rate (launch night, documented
    in the rules); later refreshes are smaller (x2.5) with a more
    progressed cohort (0.85x). Returns the projected rate in percent.
    """
    if horizon_hours < 1.0:
        return last_rate
    n = float(last_n)
    e = last_rate / 100.0 * n
    rate = last_rate / 100.0
    # Refresh 1: the documented first refresh. The new cohort earns
    # ~only the first-mission trophy: its non-100 trophies per player
    # (0.126) are a fraction of the incumbents' (0.478), and part of
    # that is incumbent progress, so the per-trophy cohort factor is
    # ~0.15. Later refreshes capture more incumbent progress.
    for mult, cohort in ((6.06, 0.15), (2.5, 0.35), (2.5, 0.50), (2.0, 0.60)):
        if horizon_hours < 1.0:
            break
        new_n = n * mult
        e += cohort * rate * (new_n - n)
        n = new_n
        rate = e / n
        horizon_hours -= 5.0  # refreshes run hours apart
    return rate * 100.0


def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    target_ts = parse_ts(case["target"])
    x_target = math.log1p(max(0.0, hours_since_release(target_ts)))

    snaps = snapshots(case)
    last_t, last_n, last_rates = snaps[-1]
    x_last = math.log1p(max(0.0, hours_since_release(last_t)))

    # Per-trophy fitted slope / spread inputs.
    med_logit = [0.0] * n_tr
    spread = [0.0] * n_tr
    slopes = [0.0] * n_tr

    # Median absolute logit jump per refresh, pooled across trophies:
    # the refresh-jump risk for the fan.
    jumps = []
    for k in range(1, len(snaps)):
        _, _, r_prev = snaps[k - 1]
        _, _, r_cur = snaps[k]
        for i in range(n_tr):
            if r_prev[i] is not None and r_cur[i] is not None:
                jumps.append(abs(logit(r_cur[i]) - logit(r_prev[i])))
    jumps.sort()
    med_jump = jumps[len(jumps) // 2] if jumps else 0.35
    # Expected refreshes before the target: from the snapshot spacing,
    # else the documented hours-apart cadence.
    if len(snaps) >= 3:
        gaps = [(snaps[k][0] - snaps[k - 1][0]).total_seconds() / 3600.0
                for k in range(1, len(snaps))]
        gaps.sort()
        interval = max(1.0, gaps[len(gaps) // 2])
    else:
        interval = 5.0
    n_refresh = max(0.0, horizon_hours / interval)
    refresh_risk = math.sqrt(n_refresh) * med_jump

    # Fit each trophy's logit trend.
    raw_slopes = []
    for i, trophy in enumerate(trophies):
        xs, ys, ws = [], [], []
        for t, _, rates in snaps:
            r = rates[i]
            if r is None:
                continue
            xs.append(math.log1p(max(0.0, hours_since_release(t))))
            ys.append(logit(r))
            age_h = max(0.0, (last_t - t).total_seconds() / 3600.0)
            ws.append(0.5 ** (age_h / 12.0))
        last_r = last_rates[i]
        if last_r is None:
            last_r = 0.01
        if len(xs) >= 3 and max(xs) - min(xs) > 0.05:
            _a, b, resid_std, se_b = wls_line(xs, ys, ws)
            b = max(-0.3, min(2.5, b))
            if se_b is None:
                se_b = 0.35
        else:
            b = 0.0
            resid_std = 0.0
            se_b = 0.35
        raw_slopes.append(b)
        # Anchor at the last read; flat history falls back to the
        # refresh prior instead of a zero slope.
        if len(snaps) == 1:
            proj = refresh_projection(last_r, last_n, horizon_hours)
            med_logit[i] = logit(proj)
            # One unseen refresh decides the answer; its allocation
            # across trophies is unknown, so the fan is wide.
            spread[i] = 1.0
            trend_se = 0.0
        else:
            dx = max(0.0, x_target - x_last)
            damp = 1.0 / (1.0 + 0.5 * dx)
            move = max(-2.0, min(2.0, b * dx * damp))
            med_logit[i] = logit(last_r) + move
            trend_se = (se_b or 0.35) * dx
            base = 0.5 * math.sqrt(max(0.0, horizon_days))
            s = math.sqrt(base * base + trend_se * trend_se
                          + refresh_risk * refresh_risk + 0.10 * 0.10)
            spread[i] = min(2.0, max(0.35, s))
        slopes.append(b)
        _ = resid_std

    # Tier shrinkage of the slopes, then re-anchor the medians.
    groups = {}
    for i, trophy in enumerate(trophies):
        groups.setdefault(group_of(trophy), []).append(i)
    overall = sum(raw_slopes) / max(1, len(raw_slopes))
    for g, idxs in groups.items():
        mean_b = sum(raw_slopes[i] for i in idxs) / len(idxs)
        anchor_b = mean_b if len(idxs) >= 3 else 0.5 * mean_b + 0.5 * overall
        for i in idxs:
            b_new = 0.8 * raw_slopes[i] + 0.2 * anchor_b
            if len(snaps) > 1:
                last_r = last_rates[i] if last_rates[i] is not None else 0.01
                dx = max(0.0, x_target - x_last)
                damp = 1.0 / (1.0 + 0.5 * dx)
                move = max(-2.0, min(2.0, b_new * dx * damp))
                med_logit[i] = logit(last_r) + move
            slopes[i] = b_new

    # With a flat history there is no fitted slope: the dominant risk is
    # dilution from the refresh, so the fan is skewed downward.
    if len(snaps) == 1:
        for i in range(n_tr):
            slopes[i] = -0.25

    # Story-funnel monotonicity on the medians (logit space).
    idx_of = {t["id"]: k for k, t in enumerate(trophies)}
    chain = [idx_of[i] for i in FUNNEL if i in idx_of]
    med = med_logit[:]
    j = 0
    while j < len(chain) - 1:
        if med[chain[j]] < med[chain[j + 1]]:
            # Pool the violating run to its mean and restart.
            k = j
            while k + 1 < len(chain) and med[chain[k]] < med[chain[k + 1]]:
                k += 1
            run = chain[j:k + 1]
            mean_l = sum(med[i] for i in run) / len(run)
            for i in run:
                med[i] = mean_l
            j = 0
        else:
            j += 1

    rows = []
    for i, trophy in enumerate(trophies):
        last_r = last_rates[i] if last_rates[i] is not None else 0.01
        if last_r >= 99.99:
            # The first-mission trophy: every tracked player holds it.
            rows.append({"id": trophy["id"], "q": [99.99, 100, 100, 100, 100]})
            continue
        m = med[i]
        s = spread[i]
        b = slopes[i]
        if b >= 0:
            s_down = s * (1.0 - 0.15 * min(1.0, b))
            s_up = s * (1.0 + 0.20 * min(1.0, b))
        else:
            s_down = s * 1.30
            s_up = s * 0.80
        qs = []
        for z in Z:
            s_z = s_down if z < 0 else (s_up if z > 0 else s)
            qs.append(round4(clamp_pct(inv_logit(m + z * s_z))))
        # Keep the fan monotone after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case["levels"] != LEVELS:
            raise ValueError(f"{path.name} asks for levels {case['levels']}")
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
