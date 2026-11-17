# Package validation

Validated on 30 October 2026 on the assembled package, separate from the analysis workspace, with CPython 3.13.17 on Linux (x86-64), the
versions pinned in `requirements.txt` and, for the Prophet baseline and the GEFS files, `requirements-refit.txt`. The outputs and logs of
each check are in `package_validation/`.

- `reproduce.py verify`: every distributed file hash verified (386 files); all 33 forecast files (10,755,539 forecast rows) passed the
  key, calendar-horizon, finite-value and target checks, with targets equal to the supplied observed series; the 105 national mean
  state-level R2 values of the benchmark, Fourier and factorial models were reconstructed from the saved forecasts and agree with
  `results/factorial/factorial_mean_r2.csv`; the independent recomputation scripts reproduced 165 of 165 benchmark and factorial numbers,
  90 of 90 operational values and 16 of 16 scheduling capture shares.
- `reproduce.py smoke`: all 12 refitted configurations reproduced the supplied predictions and selected hyperparameters (maximum absolute
  difference 0.0 for the benchmark configurations and 2.3e-10 short tons for the operational models).
- `reproduce.py tables --campd DIR`: all 24 analysis and figure steps completed on a copy of the package, including the conditional
  attribution rebuilt from the CAMPD exports; all 60 regenerated result files matched the supplied results (`rtol=1e-6`) and all 16 figures
  were pixel-identical to `figures/`.
- `reproduce.py run --study all --workers 2` (complete refit): every model of the study was refitted from the supplied state series for all
  45 states: benchmark, Fourier, calendar-ablation, design-variant, panel, factorial, cumulative and degree-day models, the classical
  baselines and Prophet, the fitted calendar responses, all operational models and their perfect-prognosis and forecast-trained variants,
  and the carbon-intensity forecasts for the four balancing authorities. All 7,760 refitted forecast files matched the supplied forecasts
  and selected hyperparameters (tolerance `rtol=1e-10, atol=1e-7`; maximum absolute difference 6.7e-9 short tons), the fitted calendar
  responses matched exactly and the national summary of the training-design comparison (Table 8) matched to 1e-16. The refit took about
  16 hours with two parallel jobs and was run in resumed segments.
- Rebuild of the inputs from the public sources (`code/run_all.sh`, stage A input steps, run in a separate copy): the 45 state series and
  the state eligibility were rebuilt from the CAMPD exports, and the NOAA CPC degree days, EIA-923, EIA-930, EIA-860 and EIA Grid Monitor
  files and the GEFS archive (1,636 issue days of ensemble means and spreads, and the archive timestamps) were downloaded again and
  processed. All 205 rebuilt input files matched the supplied files (204 identical in value to double precision; the gap-filled panel
  agrees to 6e-16 relative, a difference of summation order), and the county centres of population are identical to the U.S. Census
  Bureau file. The EIA State Energy Data System table, the annual Henry Hub prices and the state-name table are small reference tables
  that are not downloaded by the scripts.

The checks establish the recorded reproduction on this platform, not an execution test on every operating system. Different operating
systems or numerical libraries may produce small numerical differences, which the comparisons report rather than hide.
