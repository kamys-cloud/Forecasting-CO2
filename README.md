# Reproducing the power-sector CO2 forecasting study

This package contains the code, processed input data, day-by-day forecasts of every model, selected hyperparameters, result tables and
figures for "When Does Model Complexity Pay Off? Forecast Horizon and Seasonal Representation in Power-Sector CO2 Forecasting". It
reproduces every table and figure of the article and its Supplementary Material from the supplied data.

## Start here

The package is distributed as two archives, `CO2_horizon_reproduction_1_code_data_results.zip` and
`CO2_horizon_reproduction_2_forecasts.zip`. Extract both into the same location; together they form the `CO2_horizon_reproduction`
folder.

1. Install the pinned environment below.
2. Run `python reproduce.py verify` to verify all supplied files and reconstruct the headline results from the saved forecasts without
   retraining.
3. Run `python reproduce.py smoke` to refit 12 representative configurations and compare every prediction and selected hyperparameter
   with the supplied forecasts.
4. Run `python reproduce.py tables` to rebuild every result table and figure from the supplied forecasts and compare them with the
   supplied results.
5. Run a full refit when ready. No data download, credentials or API keys are needed after installing the dependencies.

## Environment

The recorded execution environment is CPython 3.13 on Linux (x86-64). Package versions are pinned in `requirements.txt`, which lists the
complete dependency set used for all checks reported in `VALIDATION.md`. Refitting the Prophet baseline additionally needs the packages in
`requirements-refit.txt`. Linux and macOS are the intended environments; on Windows use WSL. Different operating systems or numerical
libraries may produce small numerical differences; the package records versions and checks predictions rather than assuming
cross-platform identity.

From the extracted `CO2_horizon_reproduction` folder:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py smoke
python reproduce.py tables
```

`verify` checks the SHA-256 of every distributed file against `PACKAGE_MANIFEST.json`; checks every forecast file for unique keys,
targets dated exactly h calendar days after the origin, finite values, and targets equal to the supplied observed series (daily, 7- and
14-day totals, alternative facility panels); reconstructs the national mean state-level R2 of the benchmark, Fourier and factorial models
at all seven horizons; and runs the three independent recomputation scripts (`verify_benchmark.py`, `verify_operational.py`,
`verify_scheduling.py`), which share no code with the analyses. It writes `reproduced_summaries/`.

`smoke` refits ridge, GAM and HGB for Texas at 1 and 14 days, the 14-day models without the annual calendar encodings, the 14-day Fourier
ridge, and the 7-day operational ridge with lagged inputs and with GEFS degree-day forecasts (12 configurations). It is an execution and
regression check, not a replacement for the full refit.

`tables` copies the package to `reruns/tables/root/`, runs every analysis and figure script on the supplied forecasts and compares all
regenerated result files with `results/`. The conditional attribution analysis reads the facility-level CAMPD exports, which are not
redistributed; pass `--campd DIR` to rebuild it as well.

## Refit the studies

```bash
# Benchmark, design variants, factorial, classical baselines and fitted responses: 45 states x 7 horizons
python reproduce.py run --study benchmark --workers 4

# Operational information set with GEFS forecasts, perfect-prognosis and forecast-trained variants
python reproduce.py run --study operational --workers 4

# Carbon-intensity forecasts for the four balancing authorities
python reproduce.py run --study scheduling --workers 3

# Or all three together
python reproduce.py run --study all --workers 4
```

Each job fits one state with one native numerical thread; `--workers` sets the number of parallel jobs. A full refit takes many CPU-hours;
this package does not claim a portable runtime estimate. Fresh output goes to `reruns/<study>/`; supplied files remain unchanged.
Completed per-state files are kept, so repeating the same command resumes; a resume is refused if the code, input data, environment or
study changed. If a run was force-killed, confirm no runner is active before removing its stale `RUNNING.lock`. Do not launch two runs
into the same output directory. The operational study run alone reuses the supplied benchmark Fourier selections.

Every refitted forecast file is compared with the supplied forecasts (tolerance `rtol=1e-10, atol=1e-7`) and every selected
hyperparameter with `forecasts/selected_settings.csv`; failures are reported rather than hidden. A run with `--states all` also compares the
fitted calendar responses (Fig. 6b) and the national summary of the training-design comparison (Table 8). The Prophet baseline is refitted
when the packages in `requirements-refit.txt` are installed and skipped otherwise.

## Rebuild the inputs from the public sources

The complete pipeline, including the rebuilding of every input, is listed in `code/run_all.sh` (stage A: data and models; stage B:
analyses and figures). The input steps are, from `code/`:

```bash
export CAMPD_DIR=/path/to/campd/exports      # CAMPD daily facility-aggregation CSV exports, 2019-2025
python build_state_series.py "$CAMPD_DIR" ../data/state_daily
python get_external.py ../data/external
python get_eia930.py ../data/external
python facility_validation.py
python physical_decomposition.py
python external_validation.py
python get_raw_inputs.py ../data/raw
python ba_crosswalk.py "$CAMPD_DIR" ../data/raw/eia860 ../data/derived
python parse_gridmonitor.py ../data/raw/gridmonitor ../data/external/gridmonitor_ba_daily.csv
python gefs_degree_days.py ../work/gefs ../data/weather/cenpop.txt 2021-07-10 2025-12-31 1,2,3,4,5,6,7,10,14
python gefs_calibration.py
python gefs_spread.py ../work/spread ../data/weather/cenpop.txt 2021-07-10 2025-12-31
python gefs_availability.py 2021-07-10 2025-12-31 ../data/weather/gefs_archive_audit.csv
```

Run them in a copy of the package (set `CO2_ROOT` to the copy) to compare the rebuilt files with the supplied ones. The EIA State Energy
Data System table, the annual Henry Hub prices and the state-name table in `data/external/` are small reference tables entered from the
sources listed in `SOURCE_PROVENANCE.json`.

## What is included

| Location | Purpose |
|---|---|
| `reproduce.py` | Entry point: verification, smoke refit, table reproduction and full refit with comparison against the supplied results |
| `code/` | All scripts (see `code/README.md`); `paths.py` defines the layout, `engine.py` the calendar-aligned targets, features and training filters, `models.py` the models and tuning grids |
| `code/run_all.sh` | Complete pipeline order, including data acquisition from the public sources |
| `code/build_state_series.py` | State-day series and state eligibility from the CAMPD facility exports |
| `config.yaml` | Grids, seasonal-basis options, seeds, refit and validation definitions, horizons, inference, warning, scheduling and audit settings |
| `data/state_daily/` | Daily CO2, SO2, NOx, gross load, heat input and reporting facilities for the 45 states, 2019-2025 |
| `data/panels/` | The same series for the balanced, high-coverage and gap-filled facility panels |
| `data/derived/` | Validation summaries, facility emission factors, emissions by unit class, seasonal decomposition, generation mix, facility-to-balancing-authority crosswalk and balancing-authority emissions |
| `data/external/` | NOAA CPC degree days, EIA-930 daily data, EIA Grid Monitor CO2 estimates, EIA state inventory, Henry Hub prices, state names |
| `data/weather/` | GEFS state degree-day forecasts (raw and calibrated), calibration coefficients and skill, ensemble spread, archive audit, county centres of population |
| `forecasts/` | Day-by-day forecasts of every model and design, including the perfect-prognosis and forecast-trained operational variants (one row per state or balancing authority, horizon and origin, with the observed target), and `selected_settings.csv` with every selected hyperparameter |
| `results/` | Summary tables behind every table and figure (`benchmark`, `factorial`, `inference`, `attribution`, `operational`, `warnings`, `scheduling`) |
| `figures/` | Figures 1-11, Supplementary Figures S1-S4 and the graphical abstract as produced by the scripts |
| `PACKAGE_MANIFEST.json` | SHA-256 of every distributed file except the manifest itself and the validation records |
| `SOURCE_PROVENANCE.json` | Public sources of the input data and SHA-256 of the files derived from them |
| `VALIDATION.md`, `package_validation/` | Record of the package checks and their outputs |

## Protocol to preserve

- Daily calendar reindexing without interpolation or imputation; days without reports remain missing.
- Target at exactly t + h calendar days, constructed before incomplete rows are removed; lags refer to calendar days.
- Training rows at a fitting origin T have outcome dates no later than T. The same availability restriction applies inside the
  hyperparameter-validation split.
- Hyperparameters, including the seasonal complexity, selected once per state and horizon on the last 15% of eligible observations before
  the first origin (17 July 2022) and then frozen; refits every 14 calendar days; all models within a comparison use identical forecast
  dates and targets.
- Benchmark inputs are finalized at the end of day t. Operational inputs are finalized through day t - 1, with GEFS degree-day forecasts
  from the 12 UTC run of day t.
- National statistics are equally weighted means of state-level R2, not pooled national R2. National intervals use a joint moving-block
  bootstrap of forecast-origin dates shared by all states (28-day blocks; 2,000 replicates; seed 42).

## Expected headline checks

At 14 days, mean state-level R2 is ridge 0.3352, GAM 0.4230 and HGB 0.4219 with the original calendar inputs, and 0.4320, 0.4261 and
0.4208 for the Fourier ridge, Fourier GAM and Fourier HGB. With the original inputs ridge has the highest mean accuracy at 1-2 days and
GAM at 3-14 days; HGB first exceeds ridge at five days. Under the operational information set, the Fourier ridge with GEFS degree-day
forecasts reaches 0.7624 at one day and 0.5917 at seven days, against 0.6611 and 0.4068 for ridge with the lagged inputs. Its scheduling
decisions capture 27%, 65%, 67% and 48% of the perfect-foresight reduction in average intensity in ERCOT, CAISO, NYISO and ISO-NE.

## Limits

The facility-level CAMPD exports (https://campd.epa.gov/data, "Daily emissions, facility aggregation", 2019-2025) are public but are not
redistributed; they are needed only to rebuild the state series, the facility panels, the facility-to-balancing-authority aggregation and
the conditional attribution analysis. Rebuilding the other inputs downloads public files from EIA and NOAA, whose providers may revise
historical values; a rebuild on 30 October 2026 reproduced every supplied input file (`VALIDATION.md`). Fitting times in
`results/factorial/cost.csv` and `results/benchmark/results.json` were measured during the original fits and are not recomputed by
`tables`. A complete refit takes many CPU-hours (about 16 hours with two parallel jobs on the validation machine).
