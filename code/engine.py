"""Forecasting engine: feature construction on a complete daily calendar, target construction, outcome-availability filtering.

Targets are built by shifting CO2 on the complete daily calendar (missing days stay missing) BEFORE complete-case filtering, so a
target always refers to the calendar day t+h, never to the h-th next available row. At a fitting date T, a row (origin t, target date
t+h) is eligible for training only if t+h <= T; the same rule is applied inside the hyperparameter-validation split."""
import numpy as np, pandas as pd

CALENDAR_FEATURES = ["month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"]
LAG_FEATURES = ["co2_mass_lag0", "co2_mass_lag1", "co2_mass_lag2", "co2_mass_lag3", "co2_mass_lag7", "co2_mass_lag14", "co2_mass_lag30",
                "co2_mass_roll7_mean", "co2_mass_roll7_std", "co2_mass_roll14_mean", "co2_mass_roll14_std", "co2_mass_roll30_mean",
                "co2_mass_roll30_std"]
OPERATIONAL_FEATURES = ["gross_load_mwh", "heat_input", "so2_mass", "nox_mass", "n_facilities", "heat_per_mwh"]
FULL_FEATURES = CALENDAR_FEATURES + ["year"] + LAG_FEATURES + OPERATIONAL_FEATURES


def engineer_features(d):
    d = d.sort_values("date").copy().reset_index(drop=True)
    d["day_of_week"] = d.date.dt.dayofweek; d["month"] = d.date.dt.month; d["year"] = d.date.dt.year; d["day_of_year"] = d.date.dt.dayofyear
    d["is_weekend"] = (d.day_of_week >= 5).astype(int)
    d["month_sin"] = np.sin(2 * np.pi * d.month / 12); d["month_cos"] = np.cos(2 * np.pi * d.month / 12)
    d["dow_sin"] = np.sin(2 * np.pi * d.day_of_week / 7); d["dow_cos"] = np.cos(2 * np.pi * d.day_of_week / 7)
    d["doy_sin"] = np.sin(2 * np.pi * d.day_of_year / 365.25); d["doy_cos"] = np.cos(2 * np.pi * d.day_of_year / 365.25)
    for lag in [0, 1, 2, 3, 7, 14, 30]: d[f"co2_mass_lag{lag}"] = d.co2_mass.shift(lag)
    for w in [7, 14, 30]:
        d[f"co2_mass_roll{w}_mean"] = d.co2_mass.shift(1).rolling(w).mean(); d[f"co2_mass_roll{w}_std"] = d.co2_mass.shift(1).rolling(w).std()
    d["heat_per_mwh"] = d.heat_input / (d.gross_load_mwh + 1)
    return d


def prepare(sd, horizon):
    sd = sd.copy(); sd["date"] = pd.to_datetime(sd.date)
    if sd.date.duplicated().any(): raise ValueError("Duplicate dates")
    sd = sd.set_index("date").sort_index().reindex(pd.date_range(sd.date.min(), sd.date.max(), freq="D")); sd.index.name = "date"
    d = engineer_features(sd.reset_index())
    d["target_date"] = d.date + pd.Timedelta(days=horizon)
    d["target"] = d.co2_mass.shift(-horizon)                       # calendar shift on the complete daily index
    return d.replace([np.inf, -np.inf], np.nan).dropna(subset=FULL_FEATURES + ["target"]).reset_index(drop=True)


def eligible_training(d, cutoff):
    return d[(d.date < cutoff) & (d.target_date <= cutoff)]


def first_origin(sd):
    dates = pd.date_range(pd.to_datetime(sd.date).min() + pd.Timedelta(days=30), pd.to_datetime(sd.date).max())
    return dates[len(dates) // 2]


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float); e = y_true - y_pred; mask = y_true != 0
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "MAPE": float(np.mean(np.abs(e[mask] / y_true[mask])) * 100), "R2": r2(y_true, y_pred)}
