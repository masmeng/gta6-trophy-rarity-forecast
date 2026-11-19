"""Forecast GTA VI (PS5) trophy earn rates on Exophase, one case at a time.

Model: earners-over-players decomposition with class-shrunk log-time trends.

A trophy's earn rate is a ratio, rate = 100 * E / N, where E is how many tracked
players earned it and N is how many players Exophase tracks. Both move, and they
move for different reasons: E grows as the player cohort progresses through the
game, while N grows in lumps whenever Exophase refreshes its counts (on launch
night N jumped 638 -> 3,868 in one refresh while every rate fell). So this
program fits the two sides separately and combines them, instead of fitting the
ratio directly the way a plain trend line would:

  * N: least-squares fit of ln N against x = ln(1 + hours since the game's
    release), over the last 24 hours of reads (the recent regime), anchored at
    the last read and damped beyond it. When the history shows N flat at every
    read, a refresh is due: N is assumed to triple by the target. That is the
    launch-night pattern, and it only ever fires on a flat history.
  * E: least-squares fit of ln(E + 0.5) against the same x, anchored at the
    last read and damped. E never falls: the forecast is floored at the last
    observed earners and capped at the forecast N.
  * Slopes are shrunk toward their trophy class (story chapters, missions,
    grinds, early activities, platinum, other): a trophy with a noisy or short
    history borrows its class's slope; a trophy with a long clean history
    keeps its own. Classes come from the trophy's tier and description words.
  * Median: the geometric middle of two legs -- a direct logit-rate trend
    (anchored at the last rate, class-shrunk, damped) and the E/N ratio above.
    Averaging the two hedges the ways either leg can be misspecified.
  * Spread: a logit sigma that grows with the horizon and with how noisy the
    trophy's own history is, plus a fixed jump floor for refresh risk:
    sigma = sqrt((0.55*sqrt(days))^2 + 0.45^2 + (0.8*resid*gap)^2), capped
    at 1.6. Quantiles are the median's logit shifted by the normal z-scores.
  * A trophy at 100% in every read (the opening fishing trophy, earned by
    literally every tracked player through a 6x dilution) is forecast in a
    tight band just under 100 instead of through the model.

Everything is deterministic: no randomness, no wall-clock reads, no network,
standard library only. Two runs on the same case write identical bytes.

Run: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
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


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return max(0.0, (moment - RELEASE).total_seconds() / 3600.0)


def x_of(moment):
    """Log-time coordinate: ln(1 + hours since release)."""
    return math.log1p(hours_since_release(moment))


def clamp(v, lo, hi):
    return min(hi, max(lo, v))


def logit(pct):
    p = clamp(pct, 0.01, 99.99)
    return math.log(p / (100.0 - p))


def logistic(lg):
    return 100.0 / (1.0 + math.exp(-clamp(lg, -30.0, 30.0)))


def round4(x):
    """Round to 4 decimals, half up (the way JavaScript's Math.round does it)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def least_squares(xs, ys):
    """Return (slope, intercept, residual_std) of y ~ a + b*x, or None."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if n > 2:
        resid = math.sqrt(
            sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        )
    else:
        resid = 0.0
    return slope, intercept, resid


def trophy_class(trophy):
    """Rough kind of trophy, from its tier and description words."""
    tier = trophy.get("tier", "")
    desc = (trophy.get("description") or "").lower()
    if tier == "platinum":
        return "platinum"
    if desc.startswith("complete chapter") or desc == "complete the final mission.":
        return "story"
    if "complete the final mission for" in desc or "mission strands" in desc:
        return "mission"
    if desc.startswith("complete ") and any(
        word in desc
        for word in (
            "breakfast run",
            "cartel heist",
            "child's play",
            "headliners",
            "vcia takedown",
            "operation savage glory",
        )
    ):
        return "mission"
    if any(
        word in desc
        for word in (
            "100% completion",
            "250 stunts",
            "gold medal",
            "collectible card",
            "waink it",
            "50 challenges",
            "purchase all garage",
            "register 10 vehicles",
            "20 different races",
            "drone race",
        )
    ):
        return "grind"
    if any(
        word in desc
        for word in (
            "try each type of activity",
            "fishing challenge",
            "arcade game",
            "discover 100 vehicles",
            "cocktail with a view",
            "gone fishing",
        )
    ):
        return "early"
    return "other"


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target_x = x_of(parse_time(case["target"]))
    last_read = history[-1]
    last_x = x_of(parse_time(last_read["at"]))
    gap = max(0.0, target_x - last_x)
    damp = 1.0 / (1.0 + 0.35 * gap)
    damp_n = 1.0 / (1.0 + 0.5 * gap)

    # --- tracked players N: recent-regime log-time fit, anchored at last read.
    reads = []
    for read in history:
        players = read.get("players_tracked")
        if players and players > 0:
            reads.append((x_of(parse_time(read["at"])), math.log(players)))
    n_last = last_read.get("players_tracked") or 0
    if not reads or n_last <= 0:
        raise ValueError(f"{name} has no players-tracked series")
    last_moment = parse_time(last_read["at"])
    cutoff = last_moment.timestamp() - 24 * 3600
    recent = [
        (x_of(parse_time(r["at"])), math.log(r["players_tracked"]))
        for r in history
        if r.get("players_tracked") and parse_time(r["at"]).timestamp() >= cutoff
    ]
    window = recent if len(recent) >= 3 else reads
    fit_n = least_squares([p[0] for p in window], [p[1] for p in window])
    slope_n = fit_n[0] if fit_n else 0.0
    n_target = n_last * math.exp(clamp(slope_n * gap * damp_n, -1.0, 4.0))
    flat_n = all(r.get("players_tracked") == n_last for r in history)
    if flat_n and case["horizon_minutes"] >= 60:
        # No refresh yet in the history: launch night's pattern is one is due.
        n_target *= 3.0
    n_target = max(n_target, float(n_last))

    # --- per-trophy fits, then class shrinkage of both slopes.
    # Fits use the LAST 24 HOURS of reads only (the recent regime): the
    # launch-night dilution (rates collapsing as N jumped 638 -> 3,868) would
    # otherwise poison every slope with a fall that has long since reversed.
    window_cutoff = last_moment.timestamp() - 24 * 3600
    window_reads = [
        r for r in history if parse_time(r["at"]).timestamp() >= window_cutoff
    ]
    fit_reads = window_reads if len(window_reads) >= 3 else history
    direct = []  # (slope, resid, last_logit, last_rate)
    earners = []  # (slope, last_e)
    for i, trophy in enumerate(trophies):
        points = []  # (x, logit rate)
        epoints = []  # (x, ln(E + 0.5))
        last_rate = None
        last_e = 0.0
        for read in fit_reads:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            players = read.get("players_tracked") or 0
            if players <= 0:
                continue
            rate = rates[i]
            moment_x = x_of(parse_time(read["at"]))
            e_val = rate * players / 100.0
            epoints.append((moment_x, math.log(e_val + 0.5)))
            last_rate = rate
            last_e = e_val
            if 0.0 < rate < 100.0:
                points.append((moment_x, logit(rate)))
        fit_r = least_squares([p[0] for p in points], [p[1] for p in points])
        fit_e = least_squares([p[0] for p in epoints], [p[1] for p in epoints])
        direct.append(
            (
                fit_r[0] if fit_r else 0.0,
                fit_r[2] if fit_r else 0.0,
                logit(last_rate) if last_rate is not None else 0.0,
                last_rate,
            )
        )
        earners.append((fit_e[0] if fit_e else 0.0, last_e))

    classes = [trophy_class(t) for t in trophies]
    class_direct = {}
    class_earn = {}
    for cls in set(classes):
        idx = [j for j, c in enumerate(classes) if c == cls]
        class_direct[cls] = median([direct[j][0] for j in idx])
        class_earn[cls] = median([earners[j][0] for j in idx])
    global_direct = median([d[0] for d in direct])
    global_earn = median([e[0] for e in earners])

    rows = []
    for i, trophy in enumerate(trophies):
        cls = classes[i]
        slope_r, resid, last_logit, last_rate = direct[i]
        slope_e, last_e = earners[i]
        # How much history this trophy has in the fit window: more points ->
        # trust its own slope.
        n_points = sum(
            1
            for read in fit_reads
            if (read.get("rates") or [None] * len(trophies))[i] is not None
        )
        weight = n_points / (n_points + 3.0) if n_points else 0.0
        prior_r = class_direct.get(cls, global_direct)
        prior_e = class_earn.get(cls, global_earn)
        slope_r = weight * slope_r + (1.0 - weight) * prior_r
        slope_e = weight * slope_e + (1.0 - weight) * prior_e

        if last_rate is None:
            last_rate = 0.0
        # Always-100 trophy: earned by everyone, through every dilution.
        all_hundred = all(
            (read.get("rates") or [None] * len(trophies))[i] == 100
            for read in history
            if (read.get("rates") or [None] * len(trophies))[i] is not None
        ) and any(
            (read.get("rates") or [None] * len(trophies))[i] is not None
            for read in history
        )
        if all_hundred:
            rows.append({"id": trophy["id"], "q": [99.8, 99.93, 99.98, 99.995, 100.0]})
            continue

        # Leg 1: direct logit-rate trend from the last observed rate.
        leg_direct = last_logit + slope_r * gap * damp
        # Leg 2: earners over players. Earners never fall; never exceed N.
        e_target = (last_e + 0.5) * math.exp(clamp(slope_e * gap * damp, -1.0, 4.0)) - 0.5
        e_target = clamp(e_target, last_e, max(last_e, n_target * 0.999))
        rate_en = 100.0 * e_target / n_target
        leg_en = logit(rate_en)
        # Median: the middle of the two legs, in logit space.
        med_logit = 0.5 * (leg_direct + leg_en)
        med = clamp(logistic(med_logit), 0.01, 99.95)

        sigma = math.sqrt(
            (0.5 * math.sqrt(horizon_days)) ** 2
            + 0.30**2
            + (0.8 * resid * gap) ** 2
        )
        sigma = clamp(sigma, 0.15, 1.6)
        med_logit = logit(med)
        quantiles = []
        prev = -1.0
        for z in Z:
            q = clamp(logistic(med_logit + z * sigma), 0.01, 100.0)
            q = round4(q)
            if q < prev:
                q = prev
            quantiles.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
