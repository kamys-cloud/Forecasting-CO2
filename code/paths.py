"""Directory layout. ROOT is the release directory (parent of code/) unless CO2_ROOT is set. Intermediate per-state forecast files and
other working outputs go to WORK (default ROOT/work; override with CO2_WORK). The raw CAMPD facility-day exports are read from CAMPD
(default ROOT/data/raw/campd; override with CAMPD_DIR)."""
import os

ROOT = os.path.abspath(os.environ.get("CO2_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
DATA = os.path.join(ROOT, "data")
STATE_DAILY = os.path.join(DATA, "state_daily")
PANELS = os.path.join(DATA, "panels")
EXTERNAL = os.path.join(DATA, "external")
DERIVED = os.path.join(DATA, "derived")
WEATHER = os.path.join(DATA, "weather")
CAMPD = os.environ.get("CAMPD_DIR", os.path.join(DATA, "raw", "campd"))
FORECASTS = os.path.join(ROOT, "forecasts")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
WORK = os.path.abspath(os.environ.get("CO2_WORK", os.path.join(ROOT, "work")))
STATES = sorted(f[:-4] for f in os.listdir(STATE_DAILY)) if os.path.isdir(STATE_DAILY) else []


def ensure(*p):
    d = os.path.join(*p); os.makedirs(d, exist_ok=True); return d


# Optional export for publication: with FIG_EXPORT=DIR every PNG figure is also written to DIR as vector PDF and 600-dpi TIFF.
FIG_EXPORT = os.environ.get("FIG_EXPORT")
if FIG_EXPORT:
    import matplotlib.figure as _mfig
    _savefig = _mfig.Figure.savefig

    def _savefig_export(self, fname, *args, **kw):
        _savefig(self, fname, *args, **kw)
        if isinstance(fname, str) and fname.lower().endswith(".png"):
            base = os.path.join(ensure(FIG_EXPORT), os.path.splitext(os.path.basename(fname))[0])
            keep = {k: v for k, v in kw.items() if k in ("bbox_inches", "pad_inches")}
            _savefig(self, base + ".pdf", **keep)
            _savefig(self, base + ".tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **keep)

    _mfig.Figure.savefig = _savefig_export
