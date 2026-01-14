# AV Evaluation & Safety Metrics Framework

`av-eval-framework` delivers a reproducible pipeline for autonomous-driving log evaluation. It ingests frame-level logs, detects safety-critical events, estimates rare-event rates with both frequentist and Bayesian methods, and exports tables, plots, and Markdown reports suitable for technical reviews.

## Key Features
- **Synthetic-first ingestion** with a pydantic-configurable log generator (10 Hz sampling, rich metadata, disengagement modeling) and pluggable loaders for parquet/CSV sources.
- **Event detectors** for disengagements, hard braking, lane deviations, and near-miss TTC violations, each providing precise timing, duration, and severity estimates.
- **Metrics stack** producing miles-normalized rates, categorical slice breakdowns (weather, time-of-day, traffic), temporal trends, and anomaly surfacing via simple z-score logic.
- **Rare-event analytics** via Poisson confidence intervals and Gamma-Poisson posteriors to quantify uncertainty in low-frequency safety signals.
- **Visualization + reporting** that saves publishable PNG charts and compiles a Markdown briefing linking metric definitions, CSV outputs, and plots.
- **Driver-version comparison** reporting that contrasts a baseline stack with a candidate stack across events, including rate deltas and uncertainty notes.
- **Bayesian A/B comparisons** that quantify rate deltas, posterior improvement probabilities, and slice-level lifts between baseline and candidate logs.
- **Power analysis tooling** to estimate exposure miles required to detect a target percentage change in rare-event rates.
- **Investigation helpers** that surface the highest-severity events (per configurable metrics) for quick manual review.
- **CLI workflow** powered by Typer plus a Makefile for consistent local or CI execution (generate data, run evaluation, tests).

## Repository Layout
```
av-eval-framework/
  pyproject.toml
  README.md
  Makefile
  src/av_eval/
    config.py        # global configs + thresholds
    cli.py           # Typer CLI entrypoint
    data/            # schemas, loaders, synthetic generation
    metrics/         # event detection, rates, slicing, summary logic
    viz/             # matplotlib plotting helpers
    report/          # Markdown report builder
    utils/           # logging utilities
  scripts/quickstart.py
  tests/            # pytest suite
  notebooks/        # placeholder for exploratory notebooks
```

## Data Model
Frame-level schema (pandas DataFrame):
| Column | Type | Description |
| --- | --- | --- |
| `trip_id` | str | Trip identifier |
| `timestamp_s` | float | Seconds since trip start |
| `dt_s` | float | Sample spacing (default 0.1 s) |
| `ego_speed_mps` | float | Ego speed |
| `ego_accel_mps2` | float | Ego longitudinal acceleration |
| `lane_offset_m` | float | Lateral deviation from lane center |
| `ttc_s` | float | Time-to-collision proxy |
| `engaged` | bool | Autonomy engaged status |
| `weather` | category | Per-trip weather condition |
| `time_of_day` | category | Daypart bucket |
| `traffic_density` | category | Light/medium/heavy |
| `distance_m` | float | Trip-level distance repeated per frame |
| `distance_miles` | float | Trip-level miles |
| `duration_s` | float | Trip duration |
| `log_date` | datetime64 | Synthetic log date for trend slicing |
| `driver_version` | category | "baseline" vs "candidate" stack assignment per trip |

Trip-level aggregates (distance, miles, duration) are derived from the frame stream to unlock normalized metrics.

## Metric Definitions
- **Event rate per 1,000 miles**: `(event count / exposure miles) * 1000` (column `rate_per_1k_miles`)
- **Disengagement**: transition from `engaged=True` to `False`; duration until re-engagement.
- **Hard braking**: `ego_accel_mps2` below configurable threshold for ≥0.3 s.
- **Lane deviation**: `|lane_offset_m|` exceeds 0.6 m continuously for ≥1.0 s.
- **Near miss**: `ttc_s` under 1.5 s for ≥0.2 s.
- **Slice rate**: event rate computed per weather, time-of-day, or traffic bucket.
- **Trend metric**: moving average of event rate grouped by synthetic `log_date` week.
- **Rare-event intervals**: Poisson exact CI and Gamma-Poisson posterior credible interval per event type.

## Units & Definitions
- Distances are computed in meters from speed × time and converted to miles via `1 mile = 1,609.34 m`; every exposure value in CSVs is in miles.
- Event rates are always normalized per 1,000 miles (`rate_per_1k_miles`).
- Speeds and accelerations stay in SI units (`ego_speed_mps`, `ego_accel_mps2`), lane offsets are meters, and TTC is seconds.
- Timestamps are seconds since trip start; `duration_s` repeats the trip duration on every frame for convenience.
- Trend buckets in metrics use calendar weeks based on the UTC `log_date` synthetic timestamp.

## Metrics Glossary
| Metric | Default definition | Rationale |
| --- | --- | --- |
| Disengagement | `engaged` flag transitions `True→False` and remains False until re-engagement (minimum duration = one sample `dt_s`). | Tracks any loss of autonomous control, regardless of cause. |
| Hard braking | `ego_accel_mps2 < -3.0` m/s² continuously for ≥0.3 s. | Flags aggressive decelerations that may indicate safety-critical reactions. |
| Lane deviation | `abs(lane_offset_m) > 0.6` m continuously for ≥1.0 s. | Captures sustained lateral error beyond typical lane-keeping tolerance. |
| Near-miss (TTC) | `ttc_s < 1.5` s continuously for ≥0.2 s. | Highlights short time-to-collision windows indicative of risk. |
| Exposure miles | Integral of `ego_speed_mps × dt_s` converted to miles (`meters / 1,609.34`). | Provides the denominator for all rate calculations per 1,000 miles. |

## Quickstart
```bash
git clone https://github.com/YOUR_ORG/av-eval-framework.git
cd av-eval-framework

# Create virtualenv (optional)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install project + dev extras
pip install -e .[dev]

# Generate synthetic logs and run evaluation
av-eval generate-data --out outputs/data.parquet --trips 200 --seed 123
av-eval run-eval --data outputs/data.parquet --outdir outputs

# Optional helpers
av-eval power --event near_miss --baseline-rate-per-1k 2.0 --lift -0.1 --alpha 0.05 --power 0.8
av-eval export-events --data outputs/data.parquet --outdir outputs --top 50
```
Key artifacts appear under `outputs/`:

| File | Purpose |
| --- | --- |
| `metrics_overall.csv` | Per-event counts, Poisson/Bayesian intervals |
| `metrics_slices.csv` | Weather/time/traffic slice rates |
| `metrics_trends.csv` | Weekly rate trends |
| `metrics_ab.csv` | Bayesian A/B comparison summary |
| `metrics_ab_slices.csv` | Slice-level A/B deltas |
| `metrics_driver_versions.csv` | Per-driver exposure + rates |
| `top_events.csv` | Highest-severity events for investigation |
| `plots/*.png` | Publishable figures for overall/slice/driver comparisons |
| `report.md` | Markdown briefing with tables, glossary, plots |

## Bring Your Own Logs
If you already collect frame-level telemetry, export a CSV with the canonical columns (aliases such as `speed_mps` or `ttc` are automatically normalized):

```csv
trip_id,timestamp_s,dt_s,ego_speed_mps,ego_accel_mps2,lane_offset_m,ttc_s,engaged,weather,time_of_day,traffic_density,date,driver_version
trip_001,0.0,0.5,11.2,-0.1,0.02,5.5,True,sunny,day,medium,2025-01-02,baseline
trip_001,0.5,0.5,12.0,-0.2,0.01,4.8,True,sunny,day,medium,2025-01-02,baseline
trip_010,0.0,0.5,9.4,0.0,-0.03,6.7,0,rain,night,high,2025-01-05,candidate
```

Normalize the CSV into the canonical parquet schema, then run the evaluation normally:

```bash
av-eval load-csv --csv data/raw_logs.csv --out outputs/data.parquet
av-eval run-eval --data outputs/data.parquet --outdir outputs
```

You can also skip the intermediate parquet and point `run-eval` directly at a CSV; the CLI auto-detects loaders by extension.

```

## Makefile Targets
| Command | Description |
| --- | --- |
| `make install` | Install the project in editable mode |
| `make format` | Run code formatters (ruff/black optional placeholder) |
| `make lint` | Run static checks (placeholder for ruff/mypy) |
| `make test` | Execute pytest suite |
| `make generate-data` | Wrapper around CLI `generate-data` |
| `make run-eval` | Wrapper around CLI `run-eval` |

## Testing
```bash
make test
```
Pytest exercises event detectors, rate estimators, and the end-to-end pipeline to ensure deterministic behavior using seeded synthetic data.

## Quickstart Script
`scripts/quickstart.py` demonstrates a minimal programmatic use of the API (generate → evaluate) without invoking the CLI.

## Sample Output
![Overall event rates](docs/sample_outputs/sample_plot.png)

Excerpt from [`report.md`](docs/sample_outputs/report_excerpt.md):

> Baseline vs Candidate
> 
> | event_name | baseline_rate_per_1k_miles | candidate_rate_per_1k_miles | delta_rate_per_1k_miles | pct_change | prob_candidate_better | interpretation_note |
> | --- | --- | --- | --- | --- | --- | --- |
> | hard_braking | 4.12 | 3.01 | -1.11 | -26.9 | 0.93 | Mixed evidence; gather more miles |

## Extending
- Replace `data.synthetic.generate_synthetic_logs` with a loader that ingests real logs but yields the same schema.
- Add new event detectors by following the helpers in `metrics.events`.
- Register new plots and report sections via `viz.plots` and `report.build_report`.

## Design Decisions
- **Synthetic-first inputs**: We ship a deterministic generator so CI/tests never depend on private logs yet still exercise all pipeline stages.
- **Pandas + Typer stack**: Leaned on pandas for flexible slicing/trending and Typer for a user-friendly CLI; heavier orchestration layers were intentionally avoided.
- **Bayesian comparisons**: Safety stakeholders preferred posterior probabilities over p-values, so Gamma-Poisson models drive both dashboards and power analysis.
- **Markdown reports**: Markdown keeps review artifacts diff-able in Git while still rendering cleanly in docs portals.

## License
MIT
