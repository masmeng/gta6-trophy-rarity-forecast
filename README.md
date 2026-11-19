# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/masmeng/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/masmeng/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players earn each trophy in week one, from Exophase
tracking history. A trophy's earn rate is a ratio — earners over players
tracked — and the two sides move for different reasons, so the model fits
them separately instead of trending the ratio directly:

- **Players tracked (N):** log-time least-squares fit over the last 24 hours
  of reads, anchored at the last read and damped beyond it. N grows in lumps
  whenever Exophase refreshes its counts (on launch night it jumped 638 to
  3,868 in one refresh while every rate fell), so the recent regime is what
  extrapolates. When the history shows N flat at every read, a refresh is
  assumed due and N is tripled by the target.
- **Earners (E):** log-time fit of ln(E + 0.5) over the same recent window,
  anchored at the last read. Earners never fall and never exceed N.
- **Slopes** are shrunk toward their trophy class (story chapters, missions,
  grinds, early activities, platinum, other): a trophy with a noisy or short
  history borrows its class's slope; a trophy with a long clean history
  keeps its own.
- **Median:** the geometric middle of two legs — a direct logit-rate trend
  and the E/N ratio — so a misspecification in either leg is hedged.
- **Spread:** a logit sigma that grows with the horizon and with the
  trophy's own history noise, plus a fixed floor for refresh risk.
- A trophy at 100% in every read is forecast in a tight band just under 100
  instead of through the model.

Validation (local, against the kit's practice cases and a synthetic final
built from power-law N/E growth over the real 4.5-day horizon): practice
pinball loss 1.01 / 0.61 on two answer estimates vs the house baseline's
1.69 / 1.29; synthetic-final loss 0.24 vs the baseline's 0.31. Two runs on
the same case are byte-identical.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads one `*.in` case file per run from `$MAKEFUN_INPUT_DIR` and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR`: `{"trophies": [{"id": n, "q":
[q05, q25, q50, q75, q95]}]}` — five quantiles of each trophy's Exophase
earn rate (%) at the target. Python 3.13, standard library only, no network,
deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?ref=g2hsggv0&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
