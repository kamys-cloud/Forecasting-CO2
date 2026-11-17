"""Core multihorizon statistics: state-level R2 of the benchmark forecasts, Diebold-Mariano tests (HLN, Newey-West), Holm/BH,
persistence association, cluster-robust regression and wild-cluster test, regional and GAM summaries, expanded-search comparison and
the calendar-group ablation by horizon. Writes WORK/core/results.json and WORK/core/panel.csv.
Usage: python analysis_core.py"""
import os, sys, json, glob, itertools
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK, ensure

PRED = os.path.join(WORK, "pred_test"); OUT = ensure(WORK, "core")
H = [1, 2, 3, 5, 7, 10, 14]
REGION = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast", "RI": "Northeast",
    "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest", "WI": "Midwest",
    "IA": "Midwest", "KS": "Midwest", "MN": "Midwest", "MO": "Midwest", "NE": "Midwest",
    "ND": "Midwest", "SD": "Midwest",
    "DE": "South", "FL": "South", "GA": "South", "MD": "South", "NC": "South", "SC": "South",
    "VA": "South", "WV": "South", "DC": "South", "AL": "South", "KY": "South", "MS": "South",
    "TN": "South", "AR": "South", "LA": "South", "OK": "South", "TX": "South",
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West", "NM": "West",
    "UT": "West", "WY": "West", "AK": "West", "CA": "West", "HI": "West", "OR": "West", "WA": "West"}
res = {}

# ---------------------------------------------------------------- inputs: per-state benchmark forecasts (benchmark_forecasts.py)
pred = {}
for f in glob.glob(f"{PRED}/*_h*.csv"):
    if f.endswith("_cutoffs.csv"): continue
    p = pd.read_csv(f)
    if "target" not in p: continue
    pred[(p.state[0], int(p.horizon[0]))] = p
states = sorted({s for s, _ in pred}); assert len(states) == 45, len(states)
base = pd.DataFrame([dict(state=s, horizon=h, **{m: r2_score(p.target, p[m]) for m in ("ridge", "gam", "hgb")}) for (s, h), p in pred.items()]
                    ).set_index(["state", "horizon"])
nobs = pd.Series({k: len(p) for k, p in pred.items()})
res["missing_predictions"] = sum((s, h) not in pred for s in states for h in H)


# ---------------------------------------------------------------- helpers
def newey_west_var(d, L):
    d = np.asarray(d, float); n = len(d); c = d - d.mean(); v = np.sum(c * c) / n
    for l in range(1, L + 1):
        v += 2 * (1 - l / (L + 1)) * np.sum(c[l:] * c[:-l]) / n
    return v / n


def dm(ea, eb, h, loss="squared"):
    La, Lb = (ea ** 2, eb ** 2) if loss == "squared" else (np.abs(ea), np.abs(eb))
    d = La - Lb; n = len(d)
    L = max(h - 1, max(1, int(np.floor(4 * (n / 100) ** (2 / 9)))))
    s = d.mean() / np.sqrt(newey_west_var(d, L))
    s *= np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 0))
    return s, 2 * (1 - stats.t.cdf(abs(s), df=n - 1))


def holm(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p); a = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, min((m - i) * p[k], 1)); a[k] = run
    return a


def bh(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p); q = p[o] * m / (np.arange(m) + 1)
    q = np.minimum(np.minimum.accumulate(q[::-1])[::-1], 1); out = np.empty(m); out[o] = q; return out


def boot_ci(x, n=2000, seed=42):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ---------------------------------------------------------------- panel
rows = []
for s in states:
    for h in H:
        r = dict(state=s, horizon=h, region=REGION[s], ridge=base.loc[(s, h), "ridge"],
                 gam=base.loc[(s, h), "gam"], hgb=base.loc[(s, h), "hgb"], n=int(nobs[(s, h)]))
        if (s, h) in pred:
            p = pred[(s, h)]; y = p.target.to_numpy()
            r["persist"] = r2_score(y, p.persistence)
            r["ridge_x"] = r2_score(y, p.ridge_x); r["hgb_x"] = r2_score(y, p.hgb_x)
            r["dm"], r["p"] = dm(y - p.ridge.to_numpy(), y - p.hgb.to_numpy(), h)
            r["dm_abs"], r["p_abs"] = dm(y - p.ridge.to_numpy(), y - p.hgb.to_numpy(), h, "absolute")
        rows.append(r)
P = pd.DataFrame(rows)
P["d_hr"] = P.hgb - P.ridge; P["d_gr"] = P.gam - P.ridge; P["d_hg"] = P.hgb - P.gam
P.to_csv(f"{OUT}/panel.csv", index=False)
have = P.dropna(subset=["p"]) if "p" in P else P.iloc[0:0]
if len(have) == 315:
    P["p_holm"] = holm(P.p); P["q_bh"] = bh(P.p)
    P["p_abs_holm"] = holm(P.p_abs)
    P.to_csv(f"{OUT}/panel.csv", index=False)

# national means by horizon (Table: model R2)
g = P.groupby("horizon")
res["mean_r2"] = {h: {m: float(g.get_group(h)[m].mean()) for m in ["persist", "ridge", "gam", "hgb"] if m in P}
                  for h in H}
res["best_by_h"] = {h: max(["ridge", "gam", "hgb"], key=lambda m: res["mean_r2"][h][m]) for h in H}

# HGB - ridge: mean, CI, % favoring, DM counts
tab = {}
for h in H:
    x = g.get_group(h)
    t = dict(mean=float(x.d_hr.mean()), ci=boot_ci(x.d_hr.values), median=float(x.d_hr.median()),
             n_hgb=int((x.d_hr > 0).sum()), n_gam_r=int((x.d_gr > 0).sum()), mean_gr=float(x.d_gr.mean()),
             ci_gr=boot_ci(x.d_gr.values), mean_hg=float(x.d_hg.mean()), n_gam_over_hgb=int((x.d_hg < 0).sum()))
    if "p_holm" in P:
        x = P[P.horizon == h]
        fav_h = x.dm > 0  # positive DM: ridge has higher loss -> HGB favoured
        t.update(nominal=int((x.p < .05).sum()), nominal_r=int(((x.p < .05) & ~fav_h).sum()),
                 nominal_h=int(((x.p < .05) & fav_h).sum()),
                 holm_r=int(((x.p_holm < .05) & ~fav_h).sum()), holm_h=int(((x.p_holm < .05) & fav_h).sum()),
                 bh_r=int(((x.q_bh < .05) & ~fav_h).sum()), bh_h=int(((x.q_bh < .05) & fav_h).sum()),
                 abs_holm_r=int(((x.p_abs_holm < .05) & ~(x.dm_abs > 0)).sum()),
                 abs_holm_h=int(((x.p_abs_holm < .05) & (x.dm_abs > 0)).sum()))
    tab[h] = t
res["hgb_ridge"] = tab
if "p_holm" in P:
    res["dm_totals"] = dict(nominal=int((P.p < .05).sum()), holm=int((P.p_holm < .05).sum()),
                            bh=int((P.q_bh < .05).sum()))
    hs = P[P.p_holm < .05]
    res["holm_list"] = hs[["state", "horizon", "dm", "p_holm"]].round(4).values.tolist()

# within-state shift 1 -> 7 and 1 -> 14
w = P.pivot(index="state", columns="horizon", values="d_hr")
for a, b in [(1, 7), (1, 14)]:
    sh = w[b] - w[a]
    wt = stats.wilcoxon(w[b], w[a])
    res[f"shift_{a}_{b}"] = dict(pos=int((sh > 0).sum()), median=float(sh.median()),
                                 q1=float(sh.quantile(.25)), q3=float(sh.quantile(.75)),
                                 W=float(wt.statistic), p=float(wt.pvalue))

# crossover horizon per state (first sign change from <=0 to >0, linear interpolation)
cross = {}
for s in states:
    v = w.loc[s, H].values
    if v[0] > 0:
        cross[s] = ("left", 1.0); continue
    c = None
    for i in range(len(H) - 1):
        if v[i] <= 0 < v[i + 1]:
            c = H[i] + (0 - v[i]) * (H[i + 1] - H[i]) / (v[i + 1] - v[i]); break
    cross[s] = ("right", None) if c is None else ("cross", c)
cv = pd.Series({s: (c[1] if c[0] != "right" else np.inf) for s, c in cross.items()})
res["crossover"] = dict(left=[s for s, c in cross.items() if c[0] == "left"],
                        right=[s for s, c in cross.items() if c[0] == "right"],
                        median=float(cv.median()), q1=float(cv.quantile(.25)), q3=float(cv.quantile(.75)),
                        within2=int((cv <= 2).sum()), beyond5=sorted([(s, round(float(v), 1)) for s, v in cv.items() if 5 < v < np.inf], key=lambda z: z[1]),
                        never=[s for s, v in cv.items() if v == np.inf],
                        nonmonotone=[s for s in states if (w.loc[s, H].values[1:] > 0).any() and (np.diff(np.sign(w.loc[s, H].values)) < 0).any()])
pd.Series({s: c[1] for s, c in cross.items()}).to_csv(f"{OUT}/crossover.csv")
res["ridge_always"] = [s for s in states if (w.loc[s] <= 0).all()]
res["min_state_14"] = w[14].nsmallest(3).round(4).to_dict()

# persistence association
if "persist" in P and P.persist.notna().all():
    res["persist"] = {h: dict(median=float(g.get_group(h).persist.median()), mean=float(g.get_group(h).persist.mean())) for h in H}
    res["r_persist_dhr"] = float(np.corrcoef(P.persist, P.d_hr)[0, 1])
    res["r_persist_dgr"] = float(np.corrcoef(P.persist, P.d_gr)[0, 1])
    # within-horizon association (removes the common horizon trend)
    res["r_persist_within_h"] = {h: float(np.corrcoef(g.get_group(h).persist, g.get_group(h).d_hr)[0, 1]) for h in H}

# cluster-robust regression of d_hr on h
X = np.column_stack([np.ones(len(P)), P.horizon.values.astype(float)]); y = P.d_hr.values
XtXi = np.linalg.inv(X.T @ X); beta = XtXi @ X.T @ y


def crt(y_, cl):
    b = XtXi @ X.T @ y_; e = y_ - X @ b; meat = np.zeros((2, 2))
    for c in np.unique(cl):
        sg = X[cl == c].T @ e[cl == c]; meat += np.outer(sg, sg)
    G = len(np.unique(cl)); n, k = X.shape
    cov = (G / (G - 1)) * ((n - 1) / (n - k)) * XtXi @ meat @ XtXi
    se = np.sqrt(np.diag(cov)); return b[1] / se[1], se[1], G


reg = {}
for name, cl in [("state", P.state.values), ("region", P.region.values)]:
    t, se, G = crt(y, cl); reg[name] = dict(t=float(t), se=float(se), G=G, p=float(2 * stats.t.sf(abs(t), G - 1)))
# exact wild-cluster sign flip over the 4 regions
regions = np.unique(P.region.values); yr = np.full_like(y, y.mean()); er = y - yr; t0 = reg["region"]["t"]
ts = []
for signs in itertools.product([-1, 1], repeat=len(regions)):
    wv = P.region.map(dict(zip(regions, signs))).values
    ts.append(crt(yr + er * wv, P.region.values)[0])
reg["wild_p"] = float(np.mean(np.abs(ts) >= abs(t0) - 1e-12)); reg["wild_n_extreme"] = int(np.sum(np.abs(ts) >= abs(t0) - 1e-12))
reg["slope"] = float(beta[1]); reg["intercept"] = float(beta[0]); reg["zero_h"] = float(-beta[0] / beta[1])
res["regression"] = reg

# regional
rg = P.groupby(["region", "horizon"]).d_hr.mean().unstack()
res["regional_mean"] = rg.round(4).to_dict(orient="index")
res["regional_counts"] = {r: dict(n=int((P[(P.region == r) & (P.horizon == 1)]).shape[0]),
                                  ridge_1=int(((P.region == r) & (P.horizon == 1) & (P.d_hr < 0)).sum()),
                                  hgb_7=int(((P.region == r) & (P.horizon == 7) & (P.d_hr > 0)).sum()),
                                  hgb_14=int(((P.region == r) & (P.horizon == 14) & (P.d_hr > 0)).sum()),
                                  gam_14=int(((P.region == r) & (P.horizon == 14) & (P.d_gr > 0)).sum()))
                          for r in ["Northeast", "Midwest", "South", "West"]}

# GAM decomposition
res["r_gr_hr"] = float(np.corrcoef(P.d_gr, P.d_hr)[0, 1])
res["mean_hg_all"] = float(P.d_hg.mean())

# expanded search
if "hgb_x" in P and P.hgb_x.notna().all():
    P["d_x"] = P.hgb_x - P.ridge_x
    ex = {h: dict(mean=float(P[P.horizon == h].d_x.mean()), ci=boot_ci(P[P.horizon == h].d_x.values),
                  n_hgb=int((P[P.horizon == h].d_x > 0).sum())) for h in H}
    res["expanded"] = ex
    a = np.array([tab[h]["mean"] for h in H]); b = np.array([ex[h]["mean"] for h in H])
    res["expanded_r"] = float(np.corrcoef(a, b)[0, 1]); res["expanded_maxdiff"] = float(np.abs(a - b).max())
    flips = P[np.sign(P.d_x) != np.sign(P.d_hr)]
    res["expanded_flips"] = int(len(flips)); res["expanded_flip_max_primary_gap"] = float(flips.d_hr.abs().max()) if len(flips) else 0.0
    res["expanded_flip_median_primary_gap"] = float(flips.d_hr.abs().median()) if len(flips) else 0.0
    P.to_csv(f"{OUT}/panel.csv", index=False)

# calendar-group ablation by horizon (variant_forecasts.py, variant comp_cal) next to the full models
ab = {}
for h in H:
    fs = sorted(glob.glob(f"{WORK}/exp/comp_cal/*_h{h}.csv"))
    if len(fs) < 45: continue
    r = {m: [] for m in ("ridge", "gam", "hgb")}
    for f in fs:
        a = pd.read_csv(f)
        for m in r: r[m].append(r2_score(a.target, a[m]))
    ab.setdefault("without_calendar", {})[h] = {m: round(float(np.mean(v)), 4) for m, v in r.items()}
    ab.setdefault("baseline", {})[h] = {m: round(float(base.xs(h, level="horizon")[m].mean()), 4) for m in ("ridge", "gam", "hgb")}
res["ablation_by_h"] = ab

json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ["missing_predictions", "best_by_h"]}, indent=1))
