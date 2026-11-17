"""Inference for the benchmark comparisons.
(1) HGB - ridge and GAM - ridge by horizon: national mean of state-level differences with joint date-block bootstrap intervals
    (28-day blocks shared by all states; 2,000 replicates) and, for comparison, a bootstrap over states.
(2) Confirmatory family. Non-inferiority of the Fourier ridge against GAM, HGB, Fourier GAM, Fourier HGB, SARIMAX and Prophet at each
    of the seven horizons (42 one-sided tests, H0: difference <= -margin). The claim 'non-inferior to every comparator at every horizon'
    is an intersection-union test, so each component is tested at alpha = 0.05 without adjustment; superiority claims (H0: difference
    <= 0) are Holm-adjusted across the same 42 tests. Bootstrap p-values are the share of replicates at or below the null boundary.
    Sensitivity: margins 0.005, 0.01, 0.02 and block lengths 28, 56, 91 and 182 days.
(3) Horizon trend of the HGB - ridge difference (least-squares slope of the national mean on horizon), by block length.
(4) Decision-relevant translation of the margin: across all models evaluated on identical rows at each horizon, the change in the
    RMSE of the national daily total and in the median state relative RMSE per 0.01 of national mean R2.
Usage: python inference_sensitivity.py OUT_DIR"""
import os
for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[_k] = "1"
import sys, json
import numpy as np, pandas as pd
from jboot import Panel, block_weights, ci
import panels as pn

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
H = pn.H; NB = 2000; BLOCKS = [28, 56, 91, 182]; MARGINS = [0.005, 0.01, 0.02]
COMP = ["gam", "hgb", "fourier_gam", "fourier_hgb", "sarimax", "prophet"]
MODELS = ["persistence", "ridge", "gam", "hgb", "fourier_ridge", "fourier_gam", "fourier_hgb", "sarimax", "prophet", "fets", "snaive7"]


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


def boot_diffs(P, pairs, L, seed=42):
    rng = np.random.default_rng(seed); n = len(P.dates); out = []
    for _ in range(NB):
        w = block_weights(n, rng, L); r = {}
        for a, b in pairs:
            for m in (a, b):
                if m not in r: r[m] = P.r2(m, w)
        out.append([np.nanmean(r[a] - r[b]) for a, b in pairs])
    return np.array(out)


rows_fig, rows_ni, rows_trend, rows_margin = [], [], [], []
trend_boot = {L: [] for L in BLOCKS}; trend_point = []
for h in H:
    df = pn.benchmark(h); fr = pn.frames(df)
    P = Panel(fr, MODELS)
    r2 = {m: P.r2(m) for m in MODELS}
    # ---------------------------------------------------------------- (1) Fig. 2b quantities
    pairs1 = [("hgb", "ridge"), ("gam", "ridge")]
    bs = boot_diffs(P, pairs1, 28)
    rng = np.random.default_rng(42)
    for k, (a, b) in enumerate(pairs1):
        d = r2[a] - r2[b]
        sb = [np.nanmean(rng.choice(d, len(d))) for _ in range(NB)]
        rows_fig.append(dict(horizon=h, a=a, b=b, mean=float(np.nanmean(d)), joint_lo=ci(bs[:, k])[0], joint_hi=ci(bs[:, k])[1],
                             stateboot_lo=ci(np.array(sb))[0], stateboot_hi=ci(np.array(sb))[1], n_a_better=int(np.sum(d > 0)),
                             **{f"state_{s}": float(v) for s, v in zip(P.states, d)} if a == "hgb" else {}))
    # ---------------------------------------------------------------- (2) confirmatory family, (3) trend
    pairs = [("fourier_ridge", c) for c in COMP] + [("hgb", "ridge")]
    trend_point.append(float(np.nanmean(r2["hgb"] - r2["ridge"])))
    for L in BLOCKS:
        bs = boot_diffs(P, pairs, L)
        trend_boot[L].append(bs[:, -1])
        for k, (a, b) in enumerate(pairs[:-1]):
            d = float(np.nanmean(r2[a] - r2[b])); x = bs[:, k]
            row = dict(horizon=h, block=L, a=a, b=b, mean=d, ci95_lo=ci(x)[0], ci95_hi=ci(x)[1], ci90_lo=ci(x, .90)[0], ci90_hi=ci(x, .90)[1],
                       p_superior=float(np.mean(x <= 0)))
            for mg in MARGINS: row[f"p_noninf_{mg}"] = float(np.mean(x <= -mg)); row[f"equiv_{mg}"] = bool(ci(x, .90)[0] > -mg and ci(x, .90)[1] < mg)
            rows_ni.append(row)
    # ---------------------------------------------------------------- (4) margin translation
    full = P.mask.all(0); Y = np.where(P.mask, P.y, 0).sum(0)[full]
    for m in MODELS:
        F = np.where(P.mask, P.p[m], 0).sum(0)[full]
        srmse = np.sqrt(np.nanmean(np.where(P.mask, (P.y - P.p[m]) ** 2, np.nan), axis=1)) / np.nanmean(np.where(P.mask, P.y, np.nan), axis=1)
        rows_margin.append(dict(horizon=h, model=m, mean_r2=float(np.nanmean(r2[m])), national_rmse_kt=float(np.sqrt(np.mean((Y - F) ** 2)) / 1e3),
                                national_mean_kt=float(Y.mean() / 1e3), median_state_rel_rmse_pct=float(100 * np.median(srmse))))
    print("h", h, "done", flush=True)

NI = pd.DataFrame(rows_ni)
for L in BLOCKS:
    for mg in MARGINS + [None]:
        sel = NI.block == L
        col = "p_superior" if mg is None else f"p_noninf_{mg}"
        NI.loc[sel, col + "_holm"] = holm(NI.loc[sel, col].values)
NI.to_csv(os.path.join(OUT, "confirmatory_family.csv"), index=False)
pd.DataFrame(rows_fig).to_csv(os.path.join(OUT, "hgb_ridge_by_horizon.csv"), index=False)
hh = np.array(H, float)
slope = float(np.polyfit(hh, trend_point, 1)[0])
for L in BLOCKS:
    B = np.array(trend_boot[L]).T                      # NB x horizons (same seeds per horizon -> independent draws across h)
    sl = np.array([np.polyfit(hh, b, 1)[0] for b in B])
    rows_trend.append(dict(block=L, slope=slope, ci95_lo=ci(sl)[0], ci95_hi=ci(sl)[1], share_positive=float(np.mean(sl > 0))))
pd.DataFrame(rows_trend).to_csv(os.path.join(OUT, "trend_by_block.csv"), index=False)
M = pd.DataFrame(rows_margin); M.to_csv(os.path.join(OUT, "margin_translation.csv"), index=False)
tr = []
for h in H:
    m = M[(M.horizon == h) & ~M.model.isin(["persistence", "snaive7"])]
    b1 = np.polyfit(m.mean_r2, m.national_rmse_kt, 1)[0]; b2 = np.polyfit(m.mean_r2, m.median_state_rel_rmse_pct, 1)[0]
    tr.append(dict(horizon=h, national_rmse_kt_per_0p01=-b1 * 0.01, pct_of_mean=-100 * b1 * 0.01 / m.national_mean_kt.mean(),
                   state_rel_rmse_pctpt_per_0p01=-b2 * 0.01, n_models=len(m)))
pd.DataFrame(tr).to_csv(os.path.join(OUT, "margin_slopes.csv"), index=False)
print(pd.DataFrame(rows_fig)[["horizon", "a", "mean", "joint_lo", "joint_hi", "stateboot_lo", "stateboot_hi"]].round(4).to_string())
print(NI[NI.block == 28][["horizon", "b", "mean", "ci90_lo", "p_noninf_0.01", "p_superior_holm"]].round(4).to_string())
print(pd.DataFrame(rows_trend).round(4)); print(pd.DataFrame(tr).round(4))
