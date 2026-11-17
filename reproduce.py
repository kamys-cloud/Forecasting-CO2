"""Entry point for verifying and reproducing the power-sector CO2 forecasting study.

  python reproduce.py verify               check every supplied file and reconstruct the headline metrics from the saved forecasts
  python reproduce.py smoke                refit 12 representative configurations and compare them with the supplied forecasts
  python reproduce.py tables               rebuild every result table and figure from the supplied forecasts and compare
  python reproduce.py run --study STUDY    refit the models (benchmark | operational | scheduling | all) and compare

Supplied files are never modified: all new output goes to reruns/ and reproduced_summaries/."""
import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "1"
import argparse, glob, hashlib, json, platform, re, shutil, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
STATES = sorted(p.stem for p in (ROOT / "data" / "state_daily").glob("*.csv"))
H = [1, 2, 3, 5, 7, 10, 14]
RTOL, ATOL = 1e-10, 1e-7
KEYS = {"state", "horizon", "date", "target_date", "balancing_authority", "variant", "model"}
SKIP_HASH = {"PACKAGE_MANIFEST.json"}
SKIP_DIRS = ("reruns", "reproduced_summaries", "package_validation", ".venv", "__pycache__")


def log(msg): print(msg, flush=True)


def atomic(path, obj):
    path = Path(path); tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + "\n"); tmp.replace(path)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def payload_files():
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT).as_posix()
        if p.is_file() and not rel.startswith(SKIP_DIRS) and "__pycache__" not in rel and rel not in SKIP_HASH: yield rel


def environment():
    import sklearn, scipy, statsmodels, matplotlib
    return dict(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, scipy=scipy.__version__, statsmodels=statsmodels.__version__, matplotlib=matplotlib.__version__)


def script(args, root, work, extra_env=None, logfile=None):
    """Run one pipeline script from code/ with the package layout rooted at `root`."""
    env = dict(os.environ, CO2_ROOT=str(root), CO2_WORK=str(work)); env.update(extra_env or {})
    out = open(logfile, "a") if logfile else subprocess.DEVNULL
    r = subprocess.run([sys.executable] + [str(a) for a in args], cwd=Path(root) / "code", env=env, stdout=out, stderr=subprocess.STDOUT)
    if logfile: out.close()
    return r.returncode


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


# ============================================================================== verify
def check_manifest():
    man = json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text())
    bad = [n for n, d in man.items() if not (ROOT / n).exists() or sha256(ROOT / n) != d]
    extra = [n for n in payload_files() if n not in man]
    assert not bad, f"hash mismatch or missing: {bad[:5]}"
    if extra: log(f"Note: {len(extra)} file(s) not in the manifest were ignored, e.g. {extra[:3]}")
    return len(man)


def check_forecasts():
    """Structure of every forecast file: unique keys, exact calendar horizons, finite values, targets equal the supplied observations."""
    obs = {s: pd.read_csv(ROOT / "data" / "state_daily" / f"{s}.csv", parse_dates=["date"]).set_index("date").co2_mass for s in STATES}
    panels = {p: {s: pd.read_csv(f, parse_dates=["date"]).set_index("date").co2_mass for f in glob.glob(str(ROOT / "data" / "panels" / p / "*.csv"))
                  for s in [Path(f).stem]} for p in ("balanced", "highcov", "filled")}
    rows = []
    for f in sorted((ROOT / "forecasts").glob("*.csv.gz")):
        d = pd.read_csv(f, parse_dates=["date", "target_date"]); name = f.name[:-7]
        key = [c for c in d.columns if c in KEYS - {"target_date"}]
        assert not d.duplicated(key).any(), f"{name}: duplicated keys"
        assert ((d.target_date - d.date).dt.days == d.horizon).all(), f"{name}: target date is not origin + h calendar days"
        num = d.select_dtypes("number").to_numpy(float)
        assert not np.isinf(num).any(), f"{name}: infinite values"
        assert d[[c for c in ("horizon", "target") if c in d]].notna().all().all(), f"{name}: missing targets"
        pred = d.select_dtypes("number").drop(columns=[c for c in ("horizon", "target") if c in d])
        assert pred.notna().any(axis=1).all(), f"{name}: rows without any forecast"
        assert (pred.notna().all() | (name == "operational_training_variants")).all(), f"{name}: missing forecasts"
        target_err = np.nan
        if "state" in d:
            src = panels[name.split("_", 1)[1]] if name.startswith("panel_") else obs
            if name.startswith("cumulative_"):
                h = int(d.horizon.iloc[0])
                ref = np.concatenate([[src[s].reindex(pd.date_range(t + pd.Timedelta(days=1), t + pd.Timedelta(days=h))).sum(min_count=h)
                                       for t in g.date] for s, g in d.groupby("state", sort=False)])
                tgt = np.concatenate([g.target.to_numpy() for _, g in d.groupby("state", sort=False)])
            else:
                ref = np.concatenate([src[s].reindex(g.target_date).to_numpy() for s, g in d.groupby("state", sort=False)])
                tgt = np.concatenate([g.target.to_numpy() for _, g in d.groupby("state", sort=False)])
            target_err = float(np.nanmax(np.abs(tgt - ref) / np.maximum(np.abs(ref), 1)))
            assert np.isfinite(ref).all() and target_err < 1e-9, f"{name}: targets differ from the supplied observations"
        rows.append(dict(file=f.name, rows=len(d), horizons=",".join(map(str, sorted(d.horizon.unique()))), max_relative_target_error=target_err))
    return rows


def headline_metrics():
    """National mean of state-level R2 reconstructed from the saved forecasts and compared with the supplied summary table."""
    sources = {"baseline": {"ridge": "ridge", "gam": "gam", "hgb": "hgb"}, "fourier_ridge": {"fourier_ridge": "fourier_ridge"},
               "fourier_nonlinear": {"fourier_gam": "fourier_gam", "fourier_hgb": "fourier_hgb"},
               "factorial_annual_fourier": {"ridge": "fa_ridge", "gam": "fa_gam", "hgb": "fa_hgb"},
               "factorial_weekly_fourier": {"ridge": "fw_ridge", "gam": "fw_gam", "hgb": "fw_hgb"},
               "factorial_cyclic_spline": {"ridge": "cs_ridge", "gam": "cs_gam", "hgb": "cs_hgb"}}
    ref = pd.read_csv(ROOT / "results" / "factorial" / "factorial_mean_r2.csv").set_index("horizon")
    out = []
    for name, cols in sources.items():
        d = pd.read_csv(ROOT / "forecasts" / f"{name}.csv.gz")
        for (h,), g in d.groupby(["horizon"]):
            for c, label in cols.items():
                v = float(np.mean([r2(x.target, x[c]) for _, x in g.groupby("state")]))
                out.append(dict(forecasts=name, model=label, horizon=int(h), mean_state_r2=v, supplied=float(ref.loc[h, label])))
    m = pd.DataFrame(out); m["difference"] = m.mean_state_r2 - m.supplied
    assert (m.difference.abs() < 1e-9).all(), m[m.difference.abs() >= 1e-9]
    return m


def independent_checks(work, logfile):
    """Independent recomputation scripts (no code shared with the analyses) on per-state files rebuilt from forecasts/."""
    assert script(["split_forecasts.py"], ROOT, work, logfile=logfile) == 0, "split_forecasts failed"
    res = {}
    for s in ("verify_benchmark.py", "verify_operational.py", "verify_scheduling.py"):
        start = Path(logfile).stat().st_size if Path(logfile).exists() else 0
        rc = script([s], ROOT, work, logfile=logfile)
        text = Path(logfile).read_text()[start:]
        line = [l for l in text.splitlines() if re.search(r"mismatches|verification|agree|match", l, re.I)]
        res[s] = dict(returncode=rc, summary=line[-1] if line else "")
        assert rc == 0, f"{s} failed (see {logfile})"
    return res


def verify():
    out = ROOT / "reproduced_summaries"; out.mkdir(exist_ok=True)
    work = ROOT / "reruns" / "verify" / "work"; shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    logfile = out / "verify.log"; logfile.write_text("")
    n = check_manifest(); log(f"Package hashes verified: {n} files")
    fc = check_forecasts(); pd.DataFrame(fc).to_csv(out / "forecast_checks.csv", index=False)
    log(f"Forecast files verified: {len(fc)} (keys, calendar horizons, finite values, targets)")
    m = headline_metrics(); m.to_csv(out / "headline_metrics.csv", index=False)
    log(f"Headline metrics reconstructed: {len(m)} model-horizon means agree with results/factorial/factorial_mean_r2.csv")
    chk = independent_checks(work, logfile)
    for k, v in chk.items(): log(f"{k}: {v['summary']}")
    atomic(out / "verification.json", dict(package_hashes_checked=n, forecast_files_checked=len(fc), forecast_rows_checked=int(sum(r["rows"] for r in fc)),
                                           headline_metrics_checked=len(m), independent_checks=chk, environment=environment()))
    shutil.rmtree(ROOT / "reruns" / "verify", ignore_errors=True)
    log("PASS: all supplied files, forecasts and headline metrics verified; summaries written to reproduced_summaries/.")


# ============================================================================== comparison of refitted forecasts
def compare_csv(new, ref):
    a, b = pd.read_csv(new), pd.read_csv(ref)
    on = [c for c in ("horizon", "date", "target_date") if c in a and c in b]
    m = a.merge(b, on=on, suffixes=("", "__ref"), validate="one_to_one")
    assert len(m) == len(a) == len(b), f"different forecast rows: {len(a)} vs {len(b)}"
    err = 0.0
    for c in a.columns:
        if c in on or c == "state" or f"{c}__ref" not in m or not np.issubdtype(m[c].dtype, np.number): continue
        np.testing.assert_allclose(m[c], m[f"{c}__ref"], rtol=RTOL, atol=ATOL, err_msg=f"{new}: {c}")
        err = max(err, float(np.max(np.abs(m[c] - m[f"{c}__ref"]))))
    return err


def norm_setting(v):
    if isinstance(v, list) and len(v) == 2 and v[0] is None: return v[1]
    return v


def compare_settings(new_json, ref_json):
    a = json.loads(Path(new_json).read_text()); b = json.loads(Path(ref_json).read_text())
    a = a.get("best", a); b = b.get("best", b)
    for k in b:
        if k in a: assert norm_setting(a[k]) == norm_setting(b[k]), f"{new_json}: selected setting of {k} differs"


def compare_tree(work, ref):
    """Compare every refitted per-state file with the file rebuilt from the supplied forecasts."""
    rows = []
    for f in sorted(Path(work).rglob("*.csv")):
        rel = f.relative_to(work); r = Path(ref) / rel
        if not r.exists(): continue
        rows.append(dict(file=str(rel), max_abs_difference=compare_csv(f, r)))
        j = f.with_suffix(".json"); rj = r.with_suffix(".json")
        if rel.parts[0] == "pred_test": j, rj = f.with_name(f.stem + "_settings.json"), r.with_name(r.stem + "_settings.json")
        if j.exists() and rj.exists(): compare_settings(j, rj)
    return rows


# ============================================================================== smoke
SMOKE = [  # (label, script arguments relative to the work directory, refitted file, reference file)
    ("TX h1 ridge, GAM, HGB (benchmark)", ["variant_forecasts.py", "{W}/smoke", "../data/state_daily", "base", "1", "ridge,gam,hgb", "TX"], "smoke/base/TX_h1.csv", "pred_test/TX_h1.csv"),
    ("TX h14 ridge, GAM, HGB (benchmark)", ["variant_forecasts.py", "{W}/smoke", "../data/state_daily", "base", "14", "ridge,gam,hgb", "TX"], "smoke/base/TX_h14.csv", "pred_test/TX_h14.csv"),
    ("TX h14 ridge, GAM, HGB without annual encodings", ["variant_forecasts.py", "{W}/smoke", "../data/state_daily", "comp_annual", "14", "ridge,gam,hgb", "TX"], "smoke/comp_annual/TX_h14.csv", "exp/comp_annual/TX_h14.csv"),
    ("TX h14 Fourier ridge", ["variant_forecasts.py", "{W}/smoke", "../data/state_daily", "fourier", "14", "fourier_ridge", "TX"], "smoke/fourier/TX_h14.csv", "exp/fourier/TX_h14.csv"),
    ("TX h7 operational ridge, lagged inputs", ["operational_forecasts.py", "{W}/smoke", "../data/state_daily", "ops", "7", "ridge", "TX"], "smoke/ops/TX_h7.csv", "exp/ops/TX_h7.csv"),
    ("TX h7 operational Fourier ridge with GEFS", ["operational_forecasts.py", "{W}/smoke", "../data/state_daily", "ops_fw", "7", "ridge", "TX"], "smoke/ops_fw/TX_h7.csv", "exp/ops_fw/TX_h7.csv"),
]


def smoke():
    out = ROOT / "reruns" / "smoke"; work = out / "work"; ref = out / "reference"
    lock = acquire(out)
    try:
        logfile = out / "smoke.log"
        if not (ref / "pred_test").exists():
            assert script(["split_forecasts.py"], ROOT, ref, logfile=logfile) == 0, "split_forecasts failed"
        # the operational models reuse the annual Fourier order selected for the benchmark Fourier models
        for d in ("exp/fourier", "exp/fourier_nl"):
            (work / d).mkdir(parents=True, exist_ok=True)
            for j in (ref / d).glob("TX_h*.json"): shutil.copy(j, work / d / j.name)
        rows, n = [], 0
        for label, args, new, reference in SMOKE:
            args = [a.replace("{W}", str(work)) for a in args]
            t = time.time(); assert script(args, ROOT, work, logfile=logfile) == 0, f"{label} failed"
            err = compare_csv(work / new, ref / reference)
            j = (work / new).with_suffix(".json"); rj = (ref / reference).with_suffix(".json")
            if reference.startswith("pred_test"): rj = (ref / reference).with_name(Path(reference).stem + "_settings.json")
            if j.exists() and rj.exists(): compare_settings(j, rj)
            ncfg = len(args[5].split(","))
            n += ncfg; rows.append(dict(configuration=label, models=ncfg, max_abs_difference=err, seconds=round(time.time() - t, 1)))
            log(f"{n}/12 {label}: predictions and selected settings match the supplied forecasts (max abs difference {err:.3g})")
        res = dict(configurations=n, all_predictions_match_reference=True, max_prediction_error=max(r["max_abs_difference"] for r in rows),
                   tolerance=dict(rtol=RTOL, atol=ATOL), details=rows, environment=environment())
        atomic(out / "verification.json", res)
        log(f"PASS: {n} refitted configurations reproduce the supplied forecasts.")
    finally: release(out)


# ============================================================================== tables (stage B on a separate copy)
TABLES = [
    ["split_forecasts.py"], ["analysis_core.py"],
    ["analysis_benchmark.py", "core", "mechanism", "ablation", "variants", "classical", "fnl", "panels", "cost"],
    ["analysis_factorial.py", "factorial", "equiv", "yearblock", "application", "weather", "prophet", "oos", "cost", "hgbx", "filled"],
    ["analysis_operational.py", "gefs", "ops", "opsdec"],
    ["inference_sensitivity.py", "../results/inference"], ["trend_inference.py", "../results/inference/trend_cluster.csv"],
    ["warnings_eval.py", "../results/warnings"], ["operational_audit.py", "../results/operational"],
    ["scheduling_eval.py", "../results/scheduling"], ["intensity_drivers.py", "../results/scheduling/intensity_drivers.csv"],
    ["verify_benchmark.py"], ["verify_operational.py"], ["verify_scheduling.py"],
    ["figure_workflow.py"], ["fig_horizon.py", "../figures/fig02_horizon.png"], ["figures_main.py"],
    ["fig_attribution.py", "../figures/fig08_attribution.png"], ["fig_interannual.py", "../figures/fig09_interannual.png"],
    ["figures_operational.py"], ["fig_scheduling.py", "../figures/fig11_scheduling.png"],
    ["fig_warning_value.py", "../figures/figS4_warning_value.png"], ["graphical_abstract.py"]]


def flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from flat(v, f"{p}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from flat(v, f"{p}[{i}]")
    else: yield p, x


def compare_results(a_dir, b_dir, rtol=1e-6, atol=1e-9):
    rows = []
    for fa in sorted(Path(a_dir).rglob("*")):
        if fa.is_dir() or fa.suffix not in (".csv", ".json"): continue
        rel = fa.relative_to(a_dir); fb = Path(b_dir) / rel
        if not fb.exists(): rows.append(dict(file=str(rel), status="not regenerated")); continue
        if fa.suffix == ".json":
            a, b = dict(flat(json.loads(fa.read_text()))), dict(flat(json.loads(fb.read_text())))
            num = [k for k in a if isinstance(a[k], (int, float)) and not isinstance(a[k], bool) and isinstance(b.get(k), (int, float))]
            diff = [k for k in num if not np.isclose(a[k], b[k], rtol=rtol, atol=atol, equal_nan=True)]
            rows.append(dict(file=str(rel), values=len(num), differing=len(diff), status="match" if not diff else "differs", examples=";".join(diff[:3])))
        else:
            a, b = pd.read_csv(fa), pd.read_csv(fb)
            if a.shape != b.shape: rows.append(dict(file=str(rel), status=f"shape {a.shape} vs {b.shape}")); continue
            cols = [c for c in a.select_dtypes("number").columns if c in b]
            diff = [c for c in cols if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=rtol, atol=atol, equal_nan=True)]
            rows.append(dict(file=str(rel), values=int(a[cols].size), differing=len(diff), status="match" if not diff else "differs", examples=";".join(diff[:3])))
    return rows


def tables(campd):
    out = ROOT / "reruns" / "tables"; lock = acquire(out)
    try:
        root = out / "root"; shutil.rmtree(root, ignore_errors=True); root.mkdir(parents=True)
        for d in ("code", "data", "forecasts", "results"): shutil.copytree(ROOT / d, root / d, ignore=shutil.ignore_patterns("__pycache__"))
        (root / "figures").mkdir()
        logfile = out / "tables.log"; logfile.write_text("")
        steps = list(TABLES)
        if campd: steps.insert(10, ["attribution.py", campd, "../results/attribution"])
        for i, args in enumerate(steps, 1):
            t = time.time(); rc = script(args, root, root / "work", logfile=logfile)
            assert rc == 0, f"{args[0]} failed (see {logfile})"
            log(f"{i}/{len(steps)} {' '.join(args[:2])} ({time.time() - t:.0f}s)")
        rows = compare_results(ROOT / "results", root / "results")
        for r in rows:
            if r["file"].startswith("attribution") and not campd: r["status"] = "supplied (rebuilding requires the CAMPD exports: --campd DIR)"
        pd.DataFrame(rows).to_csv(out / "results_comparison.csv", index=False)
        bad = [r for r in rows if r["status"] != "match" and not r["status"].startswith("supplied")]
        res = dict(steps=len(steps), result_files_compared=len(rows), matching=sum(r["status"] == "match" for r in rows),
                   not_matching=bad, attribution_rebuilt=bool(campd), figures_regenerated=len(list((root / "figures").glob("*.png"))),
                   tolerance=dict(rtol=1e-6, atol=1e-9), environment=environment())
        atomic(out / "verification.json", res)
        log(f"{res['matching']}/{len(rows)} result files match the supplied results; figures in {root / 'figures'}")
        if bad: log("Files that differ are listed in reruns/tables/results_comparison.csv")
    finally: release(out)


# ============================================================================== run (refit)
def jobs_for(study, W, states):
    sd, hz = "../data/state_daily", ",".join(map(str, H))
    per_state = []          # (phase, arguments without states) -> one job per state
    if study in ("benchmark", "all"):
        per_state += [(1, ["benchmark_forecasts.py", f"{W}/pred_test"]),
                      (1, ["variant_forecasts.py", f"{W}/exp", sd, "fourier", hz, "fourier_ridge"]),
                      (1, ["variant_forecasts.py", f"{W}/exp", sd, "fourier_nl", hz, "fourier_gam,fourier_hgb"])]
        per_state += [(1, ["variant_forecasts.py", f"{W}/exp", sd, v, hz, "ridge,gam,hgb"]) for v in ("comp_annual", "comp_doy", "comp_month", "comp_year", "comp_cal")]
        per_state += [(1, ["variant_forecasts.py", f"{W}/exp", sd, v, "1,7,14", "ridge,gam,hgb"]) for v in ("target_cal", "realtime", "resid")]
        per_state += [(1, ["variant_forecasts.py", f"{W}/exp_{p}", f"../data/panels/{p}", "base", "1,7,14", "ridge,gam,hgb"]) for p in ("balanced", "highcov", "filled")]
        per_state += [(1, ["factorial_forecasts.py", f"{W}/exp", sd, v, hz, "ridge,gam,hgb"]) for v in ("fa", "fw", "cs")]
        per_state += [(1, ["factorial_forecasts.py", f"{W}/exp", sd, "faw", "1,7,14", "hgbx"]),
                      (1, ["factorial_forecasts.py", f"{W}/exp", sd, "cum7", "7", "ridge,gam,hgb,fourier_ridge"]),
                      (1, ["factorial_forecasts.py", f"{W}/exp", sd, "cum14", "14", "ridge,gam,hgb,fourier_ridge"])]
        per_state += [(1, ["factorial_forecasts.py", f"{W}/exp", sd, v, "1,7,14", "ridge,gam,hgb"]) for v in ("clim", "oracle")]
        per_state += [(2, ["classical_baselines.py", f"{W}/pred_test", f"{W}/sb"])]
        try:
            import prophet  # noqa: F401
            per_state += [(2, ["prophet_baseline.py", f"{W}/pred_test", f"{W}/pr"])]
        except ImportError: log("prophet is not installed: the Prophet baseline is skipped (see requirements-refit.txt)")
    if study in ("operational", "all"):
        per_state += [(3, ["operational_forecasts.py", f"{W}/exp", sd, v, hz, "ridge,gam,hgb"]) for v in ("ops", "ops_f", "ops_fw")]
        per_state += [(3, ["operational_forecasts.py", f"{W}/exp", sd, v, "1,7,14", "ridge,gam,hgb"]) for v in ("ops_fc", "ops_fo")]
        per_state += [(3, ["operational_variants.py", f"{W}/opsv_ridge", v, hz, "ridge"]) for v in ("pp_r", "ft_r", "ft_rs", "pp_l2")]
        per_state += [(3, ["operational_variants.py", f"{W}/opsv_nl", v, "1,7,14", "gam,hgb"]) for v in ("pp_r", "ft_r", "ft_rs")]
    jobs = [(ph, a + [s]) for ph, a in per_state for s in states]
    if study in ("operational", "all"):   # 4- and 6-day forecasts for the states of the balancing-authority proxy
        jobs += [(3, ["operational_forecasts.py", f"{W}/exp", sd, v, "4,6", "ridge,gam,hgb", s]) for v in ("ops", "ops_f", "ops_fw")
                 for s in ("TX", "CA", "NY", "CT", "MA", "NH", "RI") if s in states]
    fitted = [s for s in ["TX", "PA", "CA", "ND", "NM", "GA"] if s in states]   # six representative states of Fig. 6
    if study in ("benchmark", "all") and fitted: jobs.append((2, ["fitted_responses.py"] + fitted))
    if study in ("scheduling", "all"):
        jobs += [(4, ["scheduling_forecasts.py", k, f"{W}/scheduling_intensity_{k}.csv.gz"]) for k in ("crosswalk", "eia", "proxy")]
    return jobs


def repair(work, ref):
    """Before resuming, remove output files left incomplete by an interrupted run so that they are refitted."""
    removed = 0
    for f in sorted(Path(work).rglob("*.csv")):
        r = Path(ref) / f.relative_to(work)
        try: ok = len(pd.read_csv(f)) == (len(pd.read_csv(r)) if r.exists() else len(pd.read_csv(f)))
        except Exception: ok = False
        if not ok:
            for g in (f, f.with_suffix(".json"), f.with_name(f.stem + "_settings.json")): g.unlink(missing_ok=True)
            removed += 1
    for f in Path(work).glob("*.csv.gz"):
        try: pd.read_csv(f)
        except Exception: f.unlink(); Path(str(f).replace(".csv.gz", "_settings.json")).unlink(missing_ok=True); removed += 1
    for f in Path(work).rglob("*.json"):
        try: json.loads(f.read_text())
        except Exception: f.unlink(); removed += 1
    if removed: log(f"resume: removed {removed} incomplete file(s) from an interrupted run")


def _job(args, root, work, logdir):
    name = "_".join(Path(a).name for a in args[:4] if a)[:120].replace(",", "-")
    rc = script(args, root, work, logfile=Path(logdir) / f"{name}.log")
    return args, rc


def run(study, workers, states):
    out = ROOT / "reruns" / (study if states == STATES else f"{study}_{'-'.join(states)}"); lock = acquire(out)
    try:
        root = out / "root"; work = out / "work"; ref = out / "reference"; logs = out / "logs"
        for d in (work, logs): d.mkdir(parents=True, exist_ok=True)
        if not root.exists():
            root.mkdir(parents=True)
            for d in ("code", "data"): shutil.copytree(ROOT / d, root / d, ignore=shutil.ignore_patterns("__pycache__"))
            for d in ("forecasts", "results", "figures"): (root / d).mkdir()
        if not (ref / "pred_test").exists():
            assert script(["split_forecasts.py"], ROOT, ref, logfile=logs / "reference.log") == 0, "split_forecasts failed"
        if study == "operational":   # reuse the supplied benchmark Fourier selections
            for d in ("exp/fourier", "exp/fourier_nl"): shutil.copytree(ref / d, work / d, dirs_exist_ok=True)
        signature = dict(code={p.name: sha256(p) for p in sorted((ROOT / "code").glob("*.py"))},
                         data={str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "data" / "state_daily").glob("*.csv"))},
                         environment=environment(), study=study, states=states)
        protocol = out / "protocol.json"
        if protocol.exists(): assert json.loads(protocol.read_text()) == json.loads(json.dumps(signature)), "Resume refused: code, input data, environment or study changed."
        else: atomic(protocol, signature)
        repair(work, ref)
        jobs = jobs_for(study, work, states)
        for phase in sorted({p for p, _ in jobs}):
            batch = [a for p, a in jobs if p == phase]; done = 0
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_job, a, root, work, logs) for a in batch]
                for f in as_completed(futs):
                    a, rc = f.result(); done += 1
                    assert rc == 0, f"{' '.join(a)} failed (see {logs})"
                    log(f"phase {phase}: {done}/{len(batch)} {' '.join(Path(x).name for x in a[:-1][:4])} {a[-1]}")
        rows = compare_tree(work, ref)
        for k in ("crosswalk", "eia", "proxy"):
            new = work / f"scheduling_intensity_{k}.csv.gz"
            if new.exists():
                a = pd.read_csv(new); b = pd.read_csv(ROOT / "forecasts" / f"scheduling_intensity_{k}.csv.gz")
                m = a.merge(b, on=["balancing_authority", "variant", "model", "horizon", "date", "target_date"], suffixes=("", "__ref"), validate="one_to_one")
                assert len(m) == len(a) == len(b); np.testing.assert_allclose(m.forecast, m.forecast__ref, rtol=RTOL, atol=ATOL)
                rows.append(dict(file=new.name, max_abs_difference=float(np.max(np.abs(m.forecast - m.forecast__ref)))))
        if (work / "opsv_ridge").exists() and states == STATES:   # national summary (Table 8) against the supplied benchmark forecasts
            s = out / "forecast_trained.csv"
            assert script(["operational_variants_summary.py", work / "opsv_ridge", work / "opsv_nl", s], ROOT, work, logfile=logs / "summary.log") == 0
            a = pd.read_csv(s); b = pd.read_csv(ROOT / "results" / "operational" / "forecast_trained.csv")
            np.testing.assert_allclose(a.select_dtypes("number"), b.select_dtypes("number"), rtol=1e-8, atol=1e-10)
            rows.append(dict(file="results/operational/forecast_trained.csv", max_abs_difference=float(np.abs(a.select_dtypes("number") - b.select_dtypes("number")).max().max())))
        fr = root / "results" / "benchmark" / "fitted_responses.csv"
        if fr.exists():
            a = pd.read_csv(fr); b = pd.read_csv(ROOT / "results" / "benchmark" / "fitted_responses.csv")
            b = b[b.state.isin(a.state.unique())]
            m = a.merge(b, on=["state", "model", "date"], suffixes=("", "__ref"), validate="one_to_one")
            assert len(m) == len(a) == len(b); np.testing.assert_allclose(m.value_pct, m.value_pct__ref, rtol=1e-8, atol=1e-8)
            rows.append(dict(file="results/benchmark/fitted_responses.csv", max_abs_difference=float(np.max(np.abs(m.value_pct - m.value_pct__ref)))))
        pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
        atomic(out / "verification.json", dict(study=study, files_compared=len(rows), all_predictions_match_reference=True,
                                               max_prediction_error=max((r["max_abs_difference"] for r in rows), default=0.0),
                                               tolerance=dict(rtol=RTOL, atol=ATOL), environment=environment()))
        log(f"PASS: {len(rows)} refitted files match the supplied forecasts.")
    finally: release(out)


# ============================================================================== locking
def acquire(out):
    out.mkdir(parents=True, exist_ok=True); lock = out / "RUNNING.lock"
    try: fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except FileExistsError: sys.exit(f"{lock} exists: another run is active, or remove the stale lock after confirming none is.")
    return lock


def release(out): (out / "RUNNING.lock").unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["verify", "smoke", "tables", "run"])
    p.add_argument("--study", choices=["benchmark", "operational", "scheduling", "all"], default="all")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--states", default="all", help="run: comma-separated subset of states (default: all 45)")
    p.add_argument("--campd", default=os.environ.get("CAMPD_DIR"), help="CAMPD facility exports (tables: also rebuild the attribution analysis)")
    a = p.parse_args()
    if a.workers < 1: p.error("--workers must be positive")
    if a.command == "verify": verify()
    elif a.command == "smoke": smoke()
    elif a.command == "tables": tables(a.campd)
    else:
        states = STATES if a.states == "all" else [s.strip().upper() for s in a.states.split(",")]
        unknown = [s for s in states if s not in STATES]
        if unknown: p.error(f"unknown states: {unknown}")
        run(a.study, a.workers, states)
