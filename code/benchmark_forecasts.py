"""Benchmark forecasts (ridge, GAM, HGB with the 27 original predictors) on the calendar-aligned rolling-origin protocol, persistence on
the identical rows, and an expanded-search (30 ridge / 60 HGB) re-tuning of ridge and HGB. One CSV per state and horizon.
Usage: python benchmark_forecasts.py OUTDIR STATE [STATE ...]      (OUTDIR is normally WORK/pred_test)"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
import sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
OUT = Path(sys.argv[1]); STATES = sys.argv[2:]
from paths import STATE_DAILY
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from models import model_for, RIDGE_ALPHA_GRID, HGB_GRID
from engine import eligible_training, prepare, FULL_FEATURES, metrics

HORIZONS = [1, 2, 3, 5, 7, 10, 14]
rng = np.random.default_rng(42)  # expanded random HGB search (seed 42)
EXP_HGB = [{"learning_rate": float(rng.uniform(0.01, 0.3)), "max_depth": int(rng.integers(2, 7)),
            "max_iter": int(rng.integers(40, 300)), "min_samples_leaf": int(rng.integers(5, 60))}
           for _ in range(60)]
EXP_RIDGE = [float(a) for a in np.logspace(-2, 3.5, 30)]


def one(state, h):
    sd = pd.read_csv(Path(STATE_DAILY) / f'{state}.csv')
    d = prepare(sd, h)
    dates = pd.date_range(pd.to_datetime(sd.date).min() + pd.Timedelta(days=30), pd.to_datetime(sd.date).max())
    start = dates[len(dates) // 2]
    dev = eligible_training(d, start); val = dev.iloc[-max(int(len(dev) * .15), 20):]
    tr = eligible_training(dev, val.date.iloc[0])
    xt, yt = tr[FULL_FEATURES].to_numpy(), tr.target.to_numpy()
    xv, yv = val[FULL_FEATURES].to_numpy(), val.target.to_numpy()
    plan = {'ridge': ('ridge', RIDGE_ALPHA_GRID), 'gam': ('gam', RIDGE_ALPHA_GRID), 'hgb': ('hgb', HGB_GRID),
            'ridge_x': ('ridge', EXP_RIDGE), 'hgb_x': ('hgb', EXP_HGB)}
    best = {}
    for key, (m, grid) in plan.items():
        sc = []
        for s in grid:
            mod = model_for(m, s); mod.fit(xt, yt); sc.append(metrics(yv, mod.predict(xv))['R2'])
        best[key] = grid[int(np.argmax(sc))]
    frames = []
    for block in pd.date_range(start, d.date.max(), freq='14D'):
        test = d[(d.date >= block) & (d.date < block + pd.Timedelta(days=14))]
        if test.empty: continue
        train = eligible_training(d, block); assert train.target_date.max() <= block
        a, b, c = train[FULL_FEATURES].to_numpy(), train.target.to_numpy(), test[FULL_FEATURES].to_numpy()
        f = test[['date', 'target_date', 'target', 'co2_mass']].rename(columns={'co2_mass': 'persistence'}).copy()
        for key, (m, _) in plan.items():
            mod = model_for(m, best[key]); mod.fit(a, b); f[key] = mod.predict(c)
        frames.append(f)
    p = pd.concat(frames, ignore_index=True); p.insert(0, 'horizon', h); p.insert(0, 'state', state)
    return p, {k: v for k, v in best.items()}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        for s in STATES:
            for h in HORIZONS:
                f = OUT / f'{s}_h{h}.csv'
                if f.exists(): continue
                t = time.time(); p, best = one(s, h)
                p.to_csv(f, index=False)
                (OUT / f'{s}_h{h}_settings.json').write_text(json.dumps(best))
                print(f'{s} h{h} {time.time()-t:.1f}s', flush=True)
    print('DONE', flush=True)
