"""Joint moving-block bootstrap over forecast-origin dates, shared by all states (common temporal shocks are resampled together).
Statistics are national equally weighted means of state-level R2 (or differences)."""
import numpy as np, pandas as pd

L_BLOCK = 28
N_BOOT = 2000


class Panel:
    """Holds aligned arrays [state x date] of target and named predictions for one horizon."""

    def __init__(self, frames, models, dates=None):
        # frames: list of DataFrames (one per state) with columns date, target, <models>
        self.states = [f.state.iloc[0] for f in frames]
        alld = sorted(set().union(*[set(pd.to_datetime(f.date)) for f in frames])) if dates is None else dates
        self.dates = pd.DatetimeIndex(alld); idx = {d: i for i, d in enumerate(self.dates)}
        S, D = len(frames), len(self.dates)
        self.y = np.full((S, D), np.nan); self.p = {m: np.full((S, D), np.nan) for m in models}
        for i, f in enumerate(frames):
            j = np.array([idx[d] for d in pd.to_datetime(f.date)])
            self.y[i, j] = f.target.to_numpy()
            for m in models: self.p[m][i, j] = f[m].to_numpy()
        self.mask = ~np.isnan(self.y)
        for m in models: self.mask &= ~np.isnan(self.p[m])

    def r2(self, m, w=None):
        """state-level R2 with date weights w (counts); returns array over states."""
        w = np.ones(self.y.shape[1]) if w is None else w
        W = self.mask * w[None, :]
        y = np.where(self.mask, self.y, 0.0); p = np.where(self.mask, self.p[m], 0.0)
        sw = W.sum(1); sy = (W * y).sum(1); syy = (W * y * y).sum(1)
        sse = (W * (y - p) ** 2).sum(1); sst = syy - sy ** 2 / sw
        return 1 - sse / sst


def block_weights(n_dates, rng, L=L_BLOCK):
    n_blocks = int(np.ceil(n_dates / L)); w = np.zeros(n_dates)
    starts = rng.integers(0, n_dates - L + 1, n_blocks)
    for s in starts: w[s:s + L] += 1
    # trim to n_dates total draws (as in the moving-block bootstrap)
    excess = int(w.sum() - n_dates)
    if excess > 0:
        last = starts[-1]; w[last + L - excess:last + L] -= 1
    return w


def boot(panel_list, stat_fn, n_boot=N_BOOT, seed=42, L=L_BLOCK):
    """panel_list: dict name->Panel sharing (approximately) the same date index; stat_fn(r2dict)->float or array.
    Weights are drawn on the union of dates and mapped to each panel."""
    union = sorted(set().union(*[set(p.dates) for p in panel_list.values()]))
    U = pd.DatetimeIndex(union); maps = {k: U.get_indexer(p.dates) for k, p in panel_list.items()}
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_boot):
        w = block_weights(len(U), rng, L)
        out.append(stat_fn({k: (lambda m, pp=p, ww=w[maps[k]]: pp.r2(m, ww)) for k, p in panel_list.items()}))
    return np.array(out)


def ci(a, level=0.95):
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return float(np.nanpercentile(a, 100 * lo)), float(np.nanpercentile(a, 100 * hi))
