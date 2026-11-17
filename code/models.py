"""Model constructors and hyperparameter grids shared by all experiments."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

SEED = 42
RIDGE_ALPHA_GRID = [0.1, 0.3, 1, 3, 10, 30, 50, 100, 200, 300, 500, 1000]
HGB_GRID = [
    {"learning_rate": 0.03, "max_depth": 2, "max_iter": 100, "min_samples_leaf": 10},
    {"learning_rate": 0.03, "max_depth": 3, "max_iter": 150, "min_samples_leaf": 14},
    {"learning_rate": 0.03, "max_depth": 4, "max_iter": 150, "min_samples_leaf": 20},
    {"learning_rate": 0.06, "max_depth": 2, "max_iter": 80, "min_samples_leaf": 10},
    {"learning_rate": 0.06, "max_depth": 3, "max_iter": 120, "min_samples_leaf": 14},
    {"learning_rate": 0.06, "max_depth": 4, "max_iter": 120, "min_samples_leaf": 20},
    {"learning_rate": 0.06, "max_depth": 5, "max_iter": 100, "min_samples_leaf": 25},
    {"learning_rate": 0.1, "max_depth": 2, "max_iter": 80, "min_samples_leaf": 10},
    {"learning_rate": 0.1, "max_depth": 3, "max_iter": 100, "min_samples_leaf": 14},
    {"learning_rate": 0.1, "max_depth": 4, "max_iter": 80, "min_samples_leaf": 20},
    {"learning_rate": 0.15, "max_depth": 3, "max_iter": 80, "min_samples_leaf": 14},
    {"learning_rate": 0.15, "max_depth": 4, "max_iter": 60, "min_samples_leaf": 25},
]


def model_for(name, setting):
    """ridge: standardized inputs + ridge; gam: standardized inputs, cubic B-spline basis (5 knots) + ridge; hgb: histogram GB."""
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=setting, random_state=SEED))
    if name == "gam":
        return make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3, include_bias=False), Ridge(alpha=setting, random_state=SEED))
    return HistGradientBoostingRegressor(random_state=SEED, **setting)
