#!/usr/bin/env bash
# Full pipeline. Run from the code/ directory. Stage A rebuilds the data and refits every model (many CPU-hours, single-threaded
# fits); stage B reproduces every table and figure from the released forecasts in minutes to hours. Scripts skip outputs that exist.
set -euo pipefail
STATES=$(ls ../data/state_daily | sed 's/.csv//' | tr '\n' ' ')
H=1,2,3,5,7,10,14
W=${CO2_WORK:-../work}

if [ "${1:-B}" = "A" ]; then
  # ---- data (raw CAMPD facility-aggregation exports in ${CAMPD_DIR:-../data/raw/campd})
  python build_state_series.py "${CAMPD_DIR:-../data/raw/campd}" ../data/state_daily
  python get_external.py ../data/external
  python get_eia930.py ../data/external
  python facility_validation.py
  python physical_decomposition.py
  python external_validation.py
  python get_raw_inputs.py ../data/raw
  python ba_crosswalk.py "${CAMPD_DIR:-../data/raw/campd}" ../data/raw/eia860 ../data/derived
  python parse_gridmonitor.py ../data/raw/gridmonitor ../data/external/gridmonitor_ba_daily.csv
  python gefs_degree_days.py "$W/gefs" ../data/weather/cenpop.txt 2021-07-10 2025-12-31 1,2,3,4,5,6,7,10,14
  python -c "import glob,os,pandas as pd; pd.concat([pd.read_csv(f).assign(issue_date=os.path.basename(f)[:8]) for f in sorted(glob.glob('$W/gefs/*.csv'))]).to_csv('../data/weather/gefs_dd_raw_by_issue.csv.gz', index=False)"
  python gefs_calibration.py
  python gefs_spread.py "$W/spread" ../data/weather/cenpop.txt 2021-07-10 2025-12-31
  python -c "import glob,os,pandas as pd; pd.concat([pd.read_csv(f).assign(date=pd.to_datetime(os.path.basename(f)[:8]).strftime('%Y-%m-%d')) for f in sorted(glob.glob('$W/spread/*.csv'))]).to_csv('../data/weather/gefs_spread.csv.gz', index=False)"
  python gefs_availability.py 2021-07-10 2025-12-31 ../data/weather/gefs_archive_audit.csv
  # ---- benchmark and design variants
  python benchmark_forecasts.py "$W/pred_test" $STATES
  python variant_forecasts.py "$W/exp" ../data/state_daily fourier $H fourier_ridge $STATES
  python variant_forecasts.py "$W/exp" ../data/state_daily fourier_nl $H fourier_gam,fourier_hgb $STATES
  for v in comp_annual comp_doy comp_month comp_year comp_cal; do python variant_forecasts.py "$W/exp" ../data/state_daily $v $H ridge,gam,hgb $STATES; done
  for v in target_cal realtime resid; do python variant_forecasts.py "$W/exp" ../data/state_daily $v 1,7,14 ridge,gam,hgb $STATES; done
  for p in balanced highcov filled; do python variant_forecasts.py "$W/exp_$p" ../data/panels/$p base 1,7,14 ridge,gam,hgb $STATES; done
  for v in fa fw cs; do python factorial_forecasts.py "$W/exp" ../data/state_daily $v $H ridge,gam,hgb $STATES; done
  python factorial_forecasts.py "$W/exp" ../data/state_daily faw 1,7,14 hgbx $STATES
  python factorial_forecasts.py "$W/exp" ../data/state_daily cum7 7 ridge,gam,hgb,fourier_ridge $STATES
  python factorial_forecasts.py "$W/exp" ../data/state_daily cum14 14 ridge,gam,hgb,fourier_ridge $STATES
  for v in clim oracle; do python factorial_forecasts.py "$W/exp" ../data/state_daily $v 1,7,14 ridge,gam,hgb $STATES; done
  python classical_baselines.py "$W/pred_test" "$W/sb" $STATES
  python prophet_baseline.py "$W/pred_test" "$W/pr" $STATES
  python fitted_responses.py
  # ---- operational information set
  for v in ops ops_f ops_fw; do python operational_forecasts.py "$W/exp" ../data/state_daily $v $H ridge,gam,hgb $STATES; done
  for v in ops ops_f ops_fw; do python operational_forecasts.py "$W/exp" ../data/state_daily $v 4,6 ridge,gam,hgb TX CA NY CT MA NH RI; done
  for v in ops_fc ops_fo; do python operational_forecasts.py "$W/exp" ../data/state_daily $v 1,7,14 ridge,gam,hgb $STATES; done
  for v in pp_r ft_r ft_rs pp_l2; do python operational_variants.py "$W/opsv_ridge" $v $H ridge $STATES; done
  for v in pp_r ft_r ft_rs; do python operational_variants.py "$W/opsv_nl" $v 1,7,14 gam,hgb $STATES; done
  # ---- scheduling intensity forecasts
  for k in crosswalk eia proxy; do python scheduling_forecasts.py $k ../forecasts/scheduling_intensity_$k.csv.gz; done
else
  python split_forecasts.py
fi

# ---- analyses
python analysis_core.py
python analysis_benchmark.py core mechanism ablation variants classical fnl panels cost
python analysis_factorial.py factorial equiv yearblock application weather prophet oos cost hgbx filled
python analysis_operational.py gefs ops opsdec
python inference_sensitivity.py ../results/inference
python trend_inference.py ../results/inference/trend_cluster.csv
python warnings_eval.py ../results/warnings
python attribution.py "${CAMPD_DIR:-../data/raw/campd}" ../results/attribution
python operational_audit.py ../results/operational
if [ -d "$W/opsv_ridge" ]; then python operational_variants_summary.py "$W/opsv_ridge" "$W/opsv_nl" ../results/operational/forecast_trained.csv; fi
python scheduling_eval.py ../results/scheduling
python intensity_drivers.py ../results/scheduling/intensity_drivers.csv
# ---- independent checks and figures
python verify_benchmark.py; python verify_operational.py; python verify_scheduling.py
python figure_workflow.py                                   # Fig. 1
python fig_horizon.py ../figures/fig02_horizon.png          # Fig. 2
python figures_main.py                                      # Figs. 3-7, S1-S3
python fig_attribution.py ../figures/fig08_attribution.png  # Fig. 8
python fig_interannual.py ../figures/fig09_interannual.png  # Fig. 9
python figures_operational.py                               # Fig. 10
python fig_scheduling.py ../figures/fig11_scheduling.png    # Fig. 11
python fig_warning_value.py ../figures/figS4_warning_value.png  # Fig. S4
python graphical_abstract.py                                # graphical abstract
