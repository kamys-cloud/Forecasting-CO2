# Code

All scripts are run from this directory; `paths.py` defines the layout (`CO2_ROOT`, `CO2_WORK` and `CAMPD_DIR` environment variables
override the defaults). `run_all.sh` lists the full order: `bash run_all.sh A` rebuilds the data and refits every model; `bash run_all.sh`
(stage B) reproduces all tables and figures from the released forecasts. With `FIG_EXPORT=DIR` set, every figure is
also written to DIR as vector PDF and 600-dpi TIFF.

| Group | Script | Purpose |
|---|---|---|
| Shared | `paths.py`, `engine.py`, `models.py` | layout; feature construction on a complete daily calendar, targets built before complete-case filtering, outcome-availability rule; model constructors and grids |
| | `jboot.py`, `panels.py`, `ba_data.py`, `gefs_names.py`, `fig_style.py` | joint date-block bootstrap; forecast panels; balancing-authority series; state names; figure style |
| Data | `build_state_series.py` | state-day series and state eligibility from the CAMPD facility exports |
| | `get_external.py`, `get_eia930.py`, `get_raw_inputs.py` | NOAA CPC degree days, EIA-923, EIA state emissions, EIA-930, EIA-860 plant files, EIA Grid Monitor workbooks |
| | `facility_validation.py`, `external_validation.py` | facility reconstruction, panels, missingness audit, EIA inventory and EIA-923/930 comparisons |
| | `physical_decomposition.py` | heat input x emission factor decomposition, degree-day and generation-mix attribution |
| | `ba_crosswalk.py`, `parse_gridmonitor.py` | facility-to-balancing-authority crosswalk, boundary error, EIA-reported BA emissions |
| | `gefs_degree_days.py`, `gefs_calibration.py`, `gefs_spread.py`, `gefs_availability.py` | GEFS degree-day forecasts, calibration and skill, ensemble spread, archive timestamps |
| Forecasts | `benchmark_forecasts.py` | ridge, GAM, HGB, persistence, expanded search |
| | `variant_forecasts.py`, `factorial_forecasts.py` | Fourier models, calendar ablations, alternative designs and panels; seasonal-representation factorial, cumulative targets, degree-day inputs |
| | `classical_baselines.py`, `classical_timing.py`, `prophet_baseline.py` | seasonal naive, Fourier-ETS, SARIMAX, Prophet |
| | `operational_forecasts.py`, `operational_variants.py` | information set at issue with GEFS forecasts; perfect-prognosis vs forecast-trained, spread, t-2 degree days |
| | `scheduling_forecasts.py` | direct daily carbon-intensity forecasts (crosswalk, EIA and proxy targets) |
| | `fitted_responses.py`, `split_forecasts.py` | fitted calendar responses; per-state working files from `forecasts/` |
| Analysis | `analysis_core.py`, `analysis_benchmark.py`, `analysis_factorial.py`, `analysis_operational.py` | DM tests, persistence, regional and GAM summaries; ablations, variants, classical, panels; factorial, equivalence, decision metrics, cost; operational |
| | `inference_sensitivity.py`, `trend_inference.py` | confirmatory family, margins, block lengths, margin translation; horizon trend with cluster and wild-cluster tests |
| | `attribution.py` | reconstruction residual and conditional out-of-sample attribution |
| | `operational_audit.py`, `operational_variants_summary.py` | calibration stability, GEFS value by season and state, spread reliability; training-design comparison |
| | `warnings_eval.py` | raw and calibrated warnings on identical held-out origins, economic value |
| | `scheduling_eval.py`, `intensity_drivers.py` | boundary agreement, ranking, marginal emission factors, electrolyser scheduling; intensity drivers |
| Checks | `verify_benchmark.py`, `verify_operational.py`, `verify_scheduling.py` | independent recomputation of reported numbers |
| Figures | `figure_workflow.py` | Fig. 1 |
| | `fig_horizon.py` | Fig. 2 |
| | `figures_main.py` | Figs. 3-7 and Supplementary Figs. S1-S3 |
| | `fig_attribution.py`, `fig_interannual.py` | Figs. 8 and 9 |
| | `figures_operational.py`, `fig_scheduling.py` | Figs. 10 and 11 |
| | `fig_warning_value.py` | Supplementary Fig. S4 |
| | `graphical_abstract.py` | graphical abstract |

Seeds: model fits 42; bootstraps 42; expanded HGB search 42. Every fit uses one thread.
