# -*- coding: utf-8 -*-
"""
Figure 3 bridge figure for Result 3
===================================

Corrected version
-----------------
This version fixes the failure where the whole analysis table could become empty
after sample filtering. The key correction is to separate the analysis into two
layers:

1) CORE sample:
   Used for Figure 3a, 3b, and 3c.
   Requires only the variables needed for:
       - continuous spatial trend map
       - regional continuous trend bars
       - rolling continuous recovery outcome
       - definition-space robustness

2) MECHANISM sample:
   Used only for Figure 3d.
   Requires the additional event-end land / atmosphere variables.

This prevents the entire bridge figure from failing just because the mechanism
subset is sparse.

Main output
-----------
ROOT/_figure3_bridge_nw/
    figures/Figure3_result3_bridge_NW_main.png
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

try:
    from shapely.geometry import Point
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Config:
    root_dir: Path = Path(
        r"E:\temp_events_ERA5_S1S6_Nature所有数据版本\events_cc3d_postlag10_NCC_with_pr_ws_rh第三篇的数据_added_CAPE_IVT_T850_added_Z500_W500_added_Bowen_Rn_added_WIND250_WIND850"
    )
    out_dir: Optional[Path] = None

    # Main outcome definition
    post_lag_max: int = 10
    rain_threshold_mm: float = 1.0
    rain_fraction_threshold: float = 0.25
    precip_threshold_grid: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)
    area_fraction_grid: Tuple[float, ...] = (0.10, 0.15, 0.25, 0.33, 0.50)

    usable_post_coverage: float = 0.80
    usable_end_coverage: float = 0.80
    require_no_overlap: bool = True
    require_no_censor: bool = True

    # Rolling design for the continuous outcome panel
    rolling_years: int = 15
    rolling_step: int = 5
    bootstrap_n: int = 200
    random_seed: int = 42

    # Continuous-trend map settings
    # IMPORTANT: the spatial map is drawn on the native heat-footprint grid
    # rather than on coarse 1° event-centroid bins. This avoids the previous
    # artificial blank areas and keeps panel a closer to the ~25 km analysis grid.
    native_grid_deg_lon: Optional[float] = None
    native_grid_deg_lat: Optional[float] = None
    min_events_per_spatial_bin: int = 5
    min_events_per_region_trend: int = 40

    target_region: str = "Northwest"

    # Plotting / map extent
    dpi: int = 320
    font_base: int = 24
    lon_min: float = -125.0
    lon_max: float = -66.0
    lat_min: float = 25.0
    lat_max: float = 50.0

    # Cache
    progress_every: int = 250
    use_cache: bool = True

    # Sample fallback
    min_core_n_total: int = 300
    min_core_n_target: int = 60
    min_mech_n_target: int = 60

    def __post_init__(self) -> None:
        if self.out_dir is None:
            self.out_dir = self.root_dir / "_figure3_bridge_nw"
        self.cache_dir = self.out_dir / "cache"
        self.fig_dir = self.out_dir / "figures"
        self.panel_dir = self.fig_dir / "single_panels"
        self.table_dir = self.out_dir / "tables"
        for p in [self.out_dir, self.cache_dir, self.fig_dir, self.panel_dir, self.table_dir]:
            p.mkdir(parents=True, exist_ok=True)


CFG = Config()


# =============================================================================
# 7-region partition
# =============================================================================


PAPER7_STATE_TO_REGION = {
    "WA": "Northwest", "OR": "Northwest", "ID": "Northwest",
    "MT": "Northern Great Plains", "WY": "Northern Great Plains", "ND": "Northern Great Plains",
    "SD": "Northern Great Plains", "NE": "Northern Great Plains",
    "MN": "Midwest", "IA": "Midwest", "MO": "Midwest", "WI": "Midwest",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "PA": "Northeast", "NY": "Northeast", "NJ": "Northeast", "DE": "Northeast",
    "MD": "Northeast", "CT": "Northeast", "RI": "Northeast", "MA": "Northeast",
    "VT": "Northeast", "NH": "Northeast", "ME": "Northeast", "WV": "Northeast",
    "CA": "Southwest", "NV": "Southwest", "AZ": "Southwest", "UT": "Southwest",
    "CO": "Southwest", "NM": "Southwest",
    "KS": "Southern Great Plains", "OK": "Southern Great Plains", "TX": "Southern Great Plains",
    "AR": "Southern Great Plains", "LA": "Southern Great Plains",
    "VA": "Southeast", "NC": "Southeast", "SC": "Southeast", "GA": "Southeast",
    "FL": "Southeast", "AL": "Southeast", "MS": "Southeast", "TN": "Southeast", "KY": "Southeast",
}
PAPER7_ORDER = [
    "Northwest", "Northern Great Plains", "Midwest", "Northeast",
    "Southwest", "Southern Great Plains", "Southeast",
]
PAPER7_COLORS = {
    "Northwest": "#c0392b",
    "Northern Great Plains": "#7f8c8d",
    "Midwest": "#8e44ad",
    "Northeast": "#1f78b4",
    "Southwest": "#d95f02",
    "Southern Great Plains": "#1b9e77",
    "Southeast": "#e6ab02",
}

STATE_GEOM_CACHE: Dict[str, List[Dict]] = {}
COORD_REGION_CACHE: Dict[Tuple[float, float], Optional[str]] = {}


# =============================================================================
# Plot style helpers
# =============================================================================


plt.rcParams.update({
    "font.size": CFG.font_base,
    "axes.titlesize": CFG.font_base + 1,
    "axes.labelsize": CFG.font_base + 1,
    "xtick.labelsize": CFG.font_base - 1,
    "ytick.labelsize": CFG.font_base - 1,
    "legend.fontsize": CFG.font_base - 1,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
})

COL_ALL = "#636363"
COL_NW = "#b2182b"
COL_CONT_ALL = "#238b45"
COL_CONT_NW = "#005a32"
COL_EVENT = "#636363"
COL_LAND = "#8c510a"
COL_ATM = "#2166ac"
COL_BOTH = "#762a83"


def tidy_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color="0.88", lw=0.8, zorder=0)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color="0.88", lw=0.8, zorder=0)


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.12) -> None:
    """Place panel labels slightly higher to avoid crowding the plotting area."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=30, fontweight="bold",
        clip_on=False,
        zorder=20,
    )


def maybe_add_panel_label(ax, label: str, show_panel_label: bool = True,
                          x: float = -0.12, y: float = 1.12) -> None:
    if show_panel_label:
        add_panel_label(ax, label, x=x, y=y)


def savefig(fig: plt.Figure, path: Path, dpi: Optional[int] = None) -> None:
    try:
        fig.savefig(path, dpi=CFG.dpi if dpi is None else dpi)
    finally:
        plt.close(fig)


def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def to_num(s: pd.Series, fill: Optional[float] = None) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if fill is not None:
        out = out.fillna(fill)
    return out


def _zscore_from_ref(s: pd.Series, mu: float, sd: float) -> pd.Series:
    s = to_num(s)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


# =============================================================================
# Region assignment
# =============================================================================


def fallback_region(lon: float, lat: float) -> str:
    if lon <= -116:
        return "Northwest" if lat >= 42 else "Southwest"
    if lon <= -100:
        return "Northern Great Plains" if lat >= 39 else "Southern Great Plains"
    if lat >= 41:
        return "Northeast"
    if lat >= 37:
        return "Midwest"
    return "Southeast"


def load_state_geometries() -> List[Dict]:
    cache_key = "paper7_conus"
    if cache_key in STATE_GEOM_CACHE:
        return STATE_GEOM_CACHE[cache_key]
    if not (HAS_CARTOPY and HAS_SHAPELY):
        raise ImportError("cartopy + shapely are required for state-polygon region assignment")

    shp = shpreader.natural_earth(
        resolution="50m",
        category="cultural",
        name="admin_1_states_provinces_lakes",
    )
    reader = shpreader.Reader(shp)
    keep = set(PAPER7_STATE_TO_REGION.keys())
    out = []
    for rec in reader.records():
        attrs = rec.attributes
        admin = attrs.get("admin") or attrs.get("adm0_name") or attrs.get("geonunit")
        if admin not in {"United States of America", "United States"}:
            continue
        postal = attrs.get("postal")
        if not postal:
            iso2 = attrs.get("iso_3166_2")
            if iso2 and "-" in iso2:
                postal = iso2.split("-")[-1]
        if postal not in keep:
            continue
        geom = rec.geometry
        if geom is None or geom.is_empty:
            continue
        out.append({"abbr": postal, "bounds": geom.bounds, "geom": geom})
    if not out:
        raise RuntimeError("No CONUS state geometries available.")
    STATE_GEOM_CACHE[cache_key] = out
    return out


def build_region_unions() -> Dict[str, object]:
    unions = {}
    if not (HAS_CARTOPY and HAS_SHAPELY):
        return unions
    try:
        geoms = load_state_geometries()
        by_region = {r: [] for r in PAPER7_ORDER}
        for rec in geoms:
            reg = PAPER7_STATE_TO_REGION.get(rec["abbr"])
            if reg in by_region:
                by_region[reg].append(rec["geom"])
        for reg, lst in by_region.items():
            if lst:
                unions[reg] = unary_union(lst)
    except Exception:
        return {}
    return unions


def point_to_region(lon: float, lat: float) -> Optional[str]:
    key = (round(float(lon), 4), round(float(lat), 4))
    if key in COORD_REGION_CACHE:
        return COORD_REGION_CACHE[key]

    reg: Optional[str] = None
    if HAS_CARTOPY and HAS_SHAPELY:
        try:
            pt = Point(float(lon), float(lat))
            for rec in load_state_geometries():
                minx, miny, maxx, maxy = rec["bounds"]
                if not (minx <= lon <= maxx and miny <= lat <= maxy):
                    continue
                if rec["geom"].covers(pt) or rec["geom"].buffer(1e-9).covers(pt):
                    reg = PAPER7_STATE_TO_REGION.get(rec["abbr"])
                    break
        except Exception:
            reg = None
    if reg is None:
        reg = fallback_region(lon, lat)
    COORD_REGION_CACHE[key] = reg
    return reg


def assign_event_region(heat_coords: pd.DataFrame, centroid_lon: float, centroid_lat: float) -> str:
    regs: List[str] = []
    for row in heat_coords[["longitude", "latitude"]].dropna().itertuples(index=False):
        r = point_to_region(float(row.longitude), float(row.latitude))
        if r is not None:
            regs.append(r)
    if not regs:
        return fallback_region(centroid_lon, centroid_lat)
    cnt = Counter(regs).most_common()
    if len(cnt) > 1 and cnt[0][1] == cnt[1][1]:
        return point_to_region(centroid_lon, centroid_lat) or cnt[0][0]
    return cnt[0][0]


# =============================================================================
# Event discovery and summarization
# =============================================================================


FILE_RE = re.compile(r"event_(\d{4})_(\d+)_window\.csv")
DATE_COLS = ["date", "event_start", "event_end", "grid_start", "grid_end", "next_grid_start"]
NUM_COLS = [
    "temp_air", "soil_moist", "longitude", "latitude", "T90", "SM10", "dry_lag1", "heat3",
    "year", "doy", "lon_round", "lat_round", "event_id", "lag_day_event", "lag_day_grid",
    "is_heat_period_event", "is_heat_period_grid", "precipitation", "wind_speed",
    "relative_humidity", "convective_available_potential_energy_mean",
    "vertically_integrated_moisture_divergence_mean", "temperature_850hPa_mean",
    "geopotential_500hPa", "vertical_velocity_500hPa", "Bowen_ratio", "Rn", "wind250", "wind850",
    "is_post_event_0_10_nominal", "is_post_grid_0_10_nominal", "overlaps_next_local_heat",
    "is_post_event_0_10_censored", "is_post_grid_0_10_censored",
]

RAW_ALIAS = {
    "soil_moist": "soil_moist",
    "SM10": "sm10",
    "relative_humidity": "rh",
    "convective_available_potential_energy_mean": "cape",
    "vertically_integrated_moisture_divergence_mean": "ivmd",
    "temperature_850hPa_mean": "t850",
    "geopotential_500hPa": "z500",
    "vertical_velocity_500hPa": "omega500",
    "Bowen_ratio": "bowen",
    "Rn": "rn",
}

END_WINDOWS = {
    "w10": [-1, 0],
}


def discover_event_files(root: Path) -> List[Path]:
    fps = sorted(root.glob("*/*event_*_window.csv"))
    if not fps:
        fps = sorted(root.rglob("event_*_window.csv"))
    return fps


def infer_native_grid_resolution(values: pd.Series, fallback: float = 0.25) -> float:
    arr = np.sort(np.unique(np.round(to_num(values).dropna().to_numpy(dtype=float), 4)))
    if arr.size < 2:
        return float(fallback)
    diffs = np.diff(arr)
    diffs = diffs[(diffs > 1e-6) & (diffs < 2.0)]
    if diffs.size == 0:
        return float(fallback)
    res = float(np.nanmedian(diffs))
    if not np.isfinite(res) or res <= 0:
        return float(fallback)
    return float(res)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        "temperature_850hPa": "temperature_850hPa_mean",
        "vertically_integrated_moisture_divergence": "vertically_integrated_moisture_divergence_mean",
        "cape": "convective_available_potential_energy_mean",
        "rh": "relative_humidity",
        "bowen_ratio": "Bowen_ratio",
        "z500": "geopotential_500hPa",
        "w500": "vertical_velocity_500hPa",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    return df


def _build_coord_key(df: pd.DataFrame) -> pd.DataFrame:
    if "coord_key" in df.columns:
        return df
    if {"longitude", "latitude"}.issubset(df.columns):
        lon_r = to_num(df["longitude"]).round(4)
        lat_r = to_num(df["latitude"]).round(4)
        df = df.copy()
        df["coord_key"] = lon_r.astype(str) + "_" + lat_r.astype(str)
    return df


def infer_precip_scale(event_files: Sequence[Path], sample_n: int = 40) -> Tuple[float, str]:
    if not event_files:
        return 1.0, "no_files"
    idx = np.linspace(0, len(event_files) - 1, min(sample_n, len(event_files))).astype(int)
    q99s = []
    for i in idx:
        try:
            df = pd.read_csv(event_files[i], low_memory=False)
            df = _clean_columns(df)
            if "precipitation" not in df.columns:
                continue
            s = to_num(df["precipitation"]).dropna()
            if not s.empty:
                q99s.append(float(s.quantile(0.99)))
        except Exception:
            continue
    if not q99s:
        return 1.0, "precip_missing_assume_mm"
    q99 = float(np.nanmedian(q99s))
    if q99 < 0.2:
        return 1000.0, f"interpreted_as_meters_q99={q99:.6g}"
    return 1.0, f"interpreted_as_mm_q99={q99:.6g}"


def event_uid_from_path(path: Path, df: pd.DataFrame) -> str:
    m = FILE_RE.search(path.name)
    if m:
        return f"{int(m.group(1))}_{int(m.group(2)):05d}"
    year = int(to_num(df.get("year", pd.Series([np.nan]))).dropna().iloc[0])
    eid = int(to_num(df.get("event_id", pd.Series([np.nan]))).dropna().iloc[0])
    return f"{year}_{eid:05d}"


def _mean_heat_excess(heat_df: pd.DataFrame) -> float:
    if heat_df.empty or "temp_air" not in heat_df.columns or "T90" not in heat_df.columns:
        return np.nan
    exc = to_num(heat_df["temp_air"]) - to_num(heat_df["T90"])
    exc = exc.clip(lower=0)
    return float(exc.mean()) if exc.notna().any() else np.nan


def _extract_window_means(df: pd.DataFrame, footprint: set, footprint_ncells: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    lag = to_num(df.get("lag_day_event", pd.Series(np.nan, index=df.index)))
    for prefix, lags in END_WINDOWS.items():
        g = df.loc[lag.isin(lags)].copy()
        g = g.dropna(subset=["coord_key"]) if "coord_key" in g.columns else g
        if "coord_key" in g.columns:
            g = g.loc[g["coord_key"].astype(str).isin(footprint)].copy()
        if g.empty:
            out[f"{prefix}_coverage"] = np.nan
            for nm in ["sm_deficit", "bowen", "rn", "rh", "ivtconv", "cape", "z500", "t850", "ascent500"]:
                out[f"{prefix}_{nm}"] = np.nan
            continue

        out[f"{prefix}_coverage"] = float(g["coord_key"].nunique()) / footprint_ncells if footprint_ncells > 0 else np.nan
        means: Dict[str, float] = {}
        for raw_col, alias in RAW_ALIAS.items():
            if raw_col in g.columns:
                s = to_num(g[raw_col])
                means[alias] = float(s.mean()) if s.notna().any() else np.nan
            else:
                means[alias] = np.nan

        out[f"{prefix}_sm_deficit"] = means["sm10"] - means["soil_moist"] if pd.notna(means["sm10"]) and pd.notna(means["soil_moist"]) else np.nan
        out[f"{prefix}_bowen"] = means["bowen"]
        out[f"{prefix}_rn"] = means["rn"]
        out[f"{prefix}_rh"] = means["rh"]
        out[f"{prefix}_ivtconv"] = -means["ivmd"] if pd.notna(means["ivmd"]) else np.nan
        out[f"{prefix}_cape"] = means["cape"]
        out[f"{prefix}_z500"] = means["z500"]
        out[f"{prefix}_t850"] = means["t850"]
        out[f"{prefix}_ascent500"] = -means["omega500"] if pd.notna(means["omega500"]) else np.nan
    return out


def summarize_event(path: Path, precip_scale: float, cfg: Config) -> Tuple[Optional[Dict], List[Dict], List[Dict]]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        warnings.warn(f"Failed to read {path.name}: {e}")
        return None, [], []

    df = _clean_columns(df)
    df = _build_coord_key(df)
    if "date" not in df.columns or "precipitation" not in df.columns or "coord_key" not in df.columns:
        return None, [], []

    for c in DATE_COLS:
        if c in df.columns:
            df[c] = to_dt(df[c])
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = to_num(df[c])

    df["precip_mm"] = to_num(df["precipitation"]) * precip_scale
    lag = to_num(df.get("lag_day_event", pd.Series(np.nan, index=df.index)))
    is_heat = to_num(df.get("is_heat_period_event", pd.Series(np.nan, index=df.index)), fill=0).astype(int)

    heat = df.loc[is_heat == 1].copy()
    if heat.empty:
        heat = df.loc[lag <= 0].copy()
    if heat.empty:
        return None, [], []

    heat_coords = heat[["coord_key", "longitude", "latitude"]].dropna().drop_duplicates("coord_key")
    footprint = set(heat_coords["coord_key"].astype(str))
    footprint_ncells = int(len(footprint))
    if footprint_ncells == 0:
        return None, [], []

    uid = event_uid_from_path(path, df)
    year_val = int(to_num(df.get("year", pd.Series([heat["date"].dt.year.mode().iloc[0]])).dropna()).iloc[0])
    event_start = heat["date"].min()
    event_end = heat["date"].max()
    if "event_start" in df.columns and df["event_start"].notna().any():
        event_start = df["event_start"].dropna().iloc[0]
    if "event_end" in df.columns and df["event_end"].notna().any():
        event_end = df["event_end"].dropna().iloc[0]

    duration = int((event_end - event_start).days + 1) if pd.notna(event_start) and pd.notna(event_end) else int(heat["date"].nunique())
    centroid_lon = float(to_num(heat_coords["longitude"]).mean())
    centroid_lat = float(to_num(heat_coords["latitude"]).mean())
    mean_heat_excess = _mean_heat_excess(heat)
    end_doy = int(event_end.dayofyear) if pd.notna(event_end) else int(to_num(df.get("doy", pd.Series([np.nan]))).median())
    region = assign_event_region(heat_coords, centroid_lon, centroid_lat)

    # Post-event daily table
    post = df.loc[(lag >= 1) & (lag <= cfg.post_lag_max)].copy()
    post_rows: List[Dict] = []
    post_daily_cols = [
        "event_uid", "year", "lag_day", "coverage", "mean_precip_mm",
        *[f"rainfrac_p{str(pthr).replace('.', '_')}" for pthr in cfg.precip_threshold_grid],
    ]
    for lag_day, g in post.groupby(lag.loc[post.index].astype(int)):
        g2 = g.dropna(subset=["coord_key"]).drop_duplicates("coord_key")
        g2 = g2.loc[g2["coord_key"].astype(str).isin(footprint)].copy()
        n_present = int(g2["coord_key"].nunique())
        coverage = n_present / footprint_ncells if footprint_ncells > 0 else np.nan
        row = {
            "event_uid": uid,
            "year": year_val,
            "lag_day": int(lag_day),
            "coverage": coverage,
            "mean_precip_mm": float(to_num(g2["precip_mm"]).mean()) if not g2.empty else np.nan,
        }
        for pthr in cfg.precip_threshold_grid:
            rainy = int(g2.loc[to_num(g2["precip_mm"]) >= pthr, "coord_key"].nunique())
            row[f"rainfrac_p{str(pthr).replace('.', '_')}"] = rainy / footprint_ncells if footprint_ncells > 0 else np.nan
        post_rows.append(row)
    post_daily = pd.DataFrame(post_rows, columns=post_daily_cols)

    default_col = f"rainfrac_p{str(cfg.rain_threshold_mm).replace('.', '_')}"
    recovered = False
    recovery_day = np.nan
    if not post_daily.empty and default_col in post_daily.columns:
        ok = (post_daily["coverage"] >= cfg.usable_post_coverage) & (post_daily[default_col] >= cfg.rain_fraction_threshold)
        if ok.any():
            recovered = True
            recovery_day = int(post_daily.loc[ok, "lag_day"].min())

    post_min_cov = float(post_daily["coverage"].min()) if not post_daily.empty else np.nan
    observed_lags = set(post_daily["lag_day"].dropna().astype(int).tolist()) if "lag_day" in post_daily.columns else set()
    has_all_lags = int(set(range(1, cfg.post_lag_max + 1)).issubset(observed_lags))
    any_overlap = int((to_num(post.get("overlaps_next_local_heat", pd.Series(0, index=post.index)), fill=0) == 1).any())
    any_censor = int((to_num(post.get("is_post_event_0_10_censored", pd.Series(0, index=post.index)), fill=0) == 1).any())

    usable_flag = bool(has_all_lags and pd.notna(post_min_cov) and (post_min_cov >= cfg.usable_post_coverage))
    if cfg.require_no_overlap:
        usable_flag = usable_flag and (any_overlap == 0)
    if cfg.require_no_censor:
        usable_flag = usable_flag and (any_censor == 0)

    window_feats = _extract_window_means(df, footprint, footprint_ncells)
    summary = {
        "event_uid": uid,
        "source_file": str(path),
        "year": year_val,
        "event_start": event_start,
        "event_end": event_end,
        "duration": duration,
        "end_doy": end_doy,
        "centroid_lon": centroid_lon,
        "centroid_lat": centroid_lat,
        "climate_region": region,
        "footprint_ncells": footprint_ncells,
        "footprint_area_km2": footprint_ncells * 625.0,
        "mean_heat_excess": mean_heat_excess,
        "no_recovery": int(not recovered),
        "recovery_day": recovery_day,
        "recovered_by_day10": int(recovered),
        "has_all_lags": has_all_lags,
        "min_post_coverage": post_min_cov,
        "any_overlap": any_overlap,
        "any_censor": any_censor,
        "usable_flag": int(usable_flag),
        "mean_rainfrac_main": float(post_daily[default_col].mean()) if (not post_daily.empty and default_col in post_daily.columns) else np.nan,
        "auc_rainfrac_main": float(post_daily[default_col].sum()) if (not post_daily.empty and default_col in post_daily.columns) else np.nan,
        "mean_precip_mm_w10": float(post_daily["mean_precip_mm"].mean()) if not post_daily.empty else np.nan,
    }
    summary.update(window_feats)

    heat_grid_rows: List[Dict] = []
    for row in heat_coords[["coord_key", "longitude", "latitude"]].dropna().drop_duplicates("coord_key").itertuples(index=False):
        heat_grid_rows.append({
            "event_uid": uid,
            "year": year_val,
            "coord_key": str(row.coord_key),
            "longitude": float(row.longitude),
            "latitude": float(row.latitude),
        })

    return summary, post_rows, heat_grid_rows


def _summary_cache_is_valid(summary: pd.DataFrame) -> bool:
    required = {
        "event_uid", "year", "climate_region", "centroid_lon", "centroid_lat", "no_recovery",
        "duration", "mean_heat_excess", "footprint_ncells", "end_doy",
        "mean_rainfrac_main", "auc_rainfrac_main",
        "w10_coverage", "w10_sm_deficit", "w10_bowen", "w10_rn", "w10_rh", "w10_ivtconv",
        "w10_cape", "w10_z500", "w10_t850", "w10_ascent500",
    }
    return required.issubset(set(summary.columns))


def _heatgrid_cache_is_valid(heatgrid: pd.DataFrame) -> bool:
    required = {"event_uid", "year", "coord_key", "longitude", "latitude"}
    return required.issubset(set(heatgrid.columns))


def build_or_load_cache(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    fp_summary = cfg.cache_dir / "event_summary_bridge.csv"
    fp_post = cfg.cache_dir / "event_post_daily_bridge.csv"
    fp_heat = cfg.cache_dir / "event_heat_grid_bridge.csv"

    if cfg.use_cache and fp_summary.exists() and fp_post.exists() and fp_heat.exists():
        summary = pd.read_csv(fp_summary, parse_dates=["event_start", "event_end"])
        post = pd.read_csv(fp_post)
        heatgrid = pd.read_csv(fp_heat)
        if _summary_cache_is_valid(summary) and _heatgrid_cache_is_valid(heatgrid):
            return summary, post, heatgrid, "loaded_cached_tables"
        print("[WARN] Existing cache is missing required columns. Rebuilding cache...")

    event_files = discover_event_files(cfg.root_dir)
    if not event_files:
        raise FileNotFoundError(f"No event CSVs found under {cfg.root_dir}")

    precip_scale, precip_note = infer_precip_scale(event_files)
    print(f"[INFO] Discovered event files: {len(event_files):,}")
    print(f"[INFO] Precipitation interpretation: {precip_note}")

    summary_rows: List[Dict] = []
    post_rows: List[Dict] = []
    heat_rows: List[Dict] = []
    for i, fp in enumerate(event_files, start=1):
        smry, post_daily, heat_daily = summarize_event(fp, precip_scale, cfg)
        if smry is not None:
            summary_rows.append(smry)
            post_rows.extend(post_daily)
            heat_rows.extend(heat_daily)
        if i % cfg.progress_every == 0 or i == len(event_files):
            print(f"[INFO] Summarized {i:,}/{len(event_files):,} files | kept events = {len(summary_rows):,}")

    post_cols = [
        "event_uid", "year", "lag_day", "coverage", "mean_precip_mm",
        *[f"rainfrac_p{str(pthr).replace('.', '_')}" for pthr in cfg.precip_threshold_grid],
    ]
    heat_cols = ["event_uid", "year", "coord_key", "longitude", "latitude"]
    summary = pd.DataFrame(summary_rows).sort_values(["year", "event_uid"]).reset_index(drop=True)
    post = pd.DataFrame(post_rows, columns=post_cols).sort_values(["year", "event_uid", "lag_day"]).reset_index(drop=True)
    heatgrid = pd.DataFrame(heat_rows, columns=heat_cols).drop_duplicates(["event_uid", "coord_key"]).sort_values(["year", "event_uid", "coord_key"]).reset_index(drop=True)

    summary.to_csv(fp_summary, index=False, encoding="utf-8-sig")
    post.to_csv(fp_post, index=False, encoding="utf-8-sig")
    heatgrid.to_csv(fp_heat, index=False, encoding="utf-8-sig")
    return summary, post, heatgrid, precip_note



# =============================================================================
# Analysis tables
# =============================================================================


def _apply_sample_mask(summary: pd.DataFrame, cfg: Config, mode: str) -> pd.DataFrame:
    dat = summary.copy()
    cov = to_num(dat.get("w10_coverage", pd.Series(np.nan, index=dat.index)))
    usable = to_num(dat.get("usable_flag", pd.Series(0, index=dat.index)))
    has_all = to_num(dat.get("has_all_lags", pd.Series(0, index=dat.index)))

    if mode == "strict_bridge":
        mask = (usable == 1) & (cov >= cfg.usable_end_coverage)
    elif mode == "coverage_only_080":
        mask = cov >= cfg.usable_end_coverage
    elif mode == "coverage_only_070":
        mask = cov >= 0.70
    elif mode == "coverage_only_060":
        mask = cov >= 0.60
    elif mode == "has_all_lags":
        mask = has_all == 1
    else:
        mask = np.ones(len(dat), dtype=bool)
    return dat.loc[mask].copy()


def _build_base_fields(dat: pd.DataFrame) -> pd.DataFrame:
    out = dat.copy()
    out["year10"] = (to_num(out["year"]) - float(to_num(out["year"]).mean())) / 10.0
    out["logfoot"] = np.log1p(to_num(out["footprint_ncells"]))
    out["doy_sin"] = np.sin(2 * np.pi * to_num(out["end_doy"]) / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * to_num(out["end_doy"]) / 366.0)
    return out


def prepare_analysis_tables(summary: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float, float]], str]:
    sample_modes = [
        "strict_bridge",
        "coverage_only_080",
        "coverage_only_070",
        "coverage_only_060",
        "has_all_lags",
        "all_events",
    ]

    selected = None
    selected_mode = None
    for mode in sample_modes:
        dat = _apply_sample_mask(summary, cfg, mode)
        if dat.empty:
            continue
        dat = _build_base_fields(dat)

        # Core variables only
        core_var_map = {
            "duration": "duration",
            "mean_heat_excess": "mean_heat_excess",
            "logfoot": "logfoot",
            "mean_rainfrac_main": "mean_rainfrac_main",
            "auc_rainfrac_main": "auc_rainfrac_main",
        }
        for dst, src in core_var_map.items():
            out = to_num(dat[src])
            dat[dst] = out

        core_needed = [
            "event_uid", "year", "climate_region", "centroid_lon", "centroid_lat", "no_recovery",
            "duration", "mean_heat_excess", "footprint_ncells", "year10", "doy_sin", "doy_cos",
            "logfoot", "mean_rainfrac_main", "auc_rainfrac_main",
        ]
        core = dat[core_needed].copy()
        core = core.dropna(subset=["no_recovery", "duration", "mean_heat_excess", "logfoot", "mean_rainfrac_main", "auc_rainfrac_main"])
        core["climate_region"] = pd.Categorical(core["climate_region"], categories=PAPER7_ORDER, ordered=True)

        n_total = len(core)
        n_target = int((core["climate_region"].astype(str) == cfg.target_region).sum())
        if n_total >= cfg.min_core_n_total and n_target >= cfg.min_core_n_target:
            selected = dat
            selected_mode = mode
            break

    if selected is None:
        raise RuntimeError("No usable core sample could be built, even after fallback sample rules.")

    dat = selected.copy()
    dat["climate_region"] = pd.Categorical(dat["climate_region"], categories=PAPER7_ORDER, ordered=True)

    full_var_map = {
        "duration": "duration",
        "mean_heat_excess": "mean_heat_excess",
        "logfoot": "logfoot",
        "sm_deficit": "w10_sm_deficit",
        "bowen": "w10_bowen",
        "rn": "w10_rn",
        "rh": "w10_rh",
        "ivtconv": "w10_ivtconv",
        "cape": "w10_cape",
        "z500": "w10_z500",
        "t850": "w10_t850",
        "ascent500": "w10_ascent500",
        "mean_rainfrac_main": "mean_rainfrac_main",
        "auc_rainfrac_main": "auc_rainfrac_main",
    }
    ref_stats: Dict[str, Tuple[float, float]] = {}
    for dst, src in full_var_map.items():
        dat[dst] = to_num(dat[src])
        s = to_num(dat[dst])
        ref_stats[dst] = (float(s.mean()), float(s.std(ddof=0)))
        dat[f"{dst}_z"] = _zscore_from_ref(s, *ref_stats[dst])

    core_keep = [
        "event_uid", "year", "climate_region", "centroid_lon", "centroid_lat", "no_recovery",
        "duration", "mean_heat_excess", "footprint_ncells", "year10", "doy_sin", "doy_cos",
        "duration_z", "mean_heat_excess_z", "logfoot_z",
        "mean_rainfrac_main", "auc_rainfrac_main", "mean_rainfrac_main_z", "auc_rainfrac_main_z",
    ]
    core_df = dat[core_keep].dropna(subset=[
        "no_recovery", "duration_z", "mean_heat_excess_z", "logfoot_z",
        "mean_rainfrac_main", "auc_rainfrac_main",
    ]).reset_index(drop=True)
    core_df = core_df.rename(columns={
        "mean_heat_excess_z": "heat_z",
        "mean_rainfrac_main_z": "mean_rainfrac_z",
        "auc_rainfrac_main_z": "auc_rainfrac_z",
    })

    mech_keep = [
        "event_uid", "year", "climate_region", "centroid_lon", "centroid_lat", "no_recovery",
        "duration", "mean_heat_excess", "footprint_ncells", "year10", "doy_sin", "doy_cos",
        "duration_z", "mean_heat_excess_z", "logfoot_z",
        "sm_deficit_z", "bowen_z", "rn_z", "rh_z", "ivtconv_z", "cape_z", "z500_z", "t850_z", "ascent500_z",
    ]
    mech_df = dat[mech_keep].dropna(subset=[
        "no_recovery", "duration_z", "mean_heat_excess_z", "logfoot_z",
        "sm_deficit_z", "bowen_z", "rn_z", "rh_z", "ivtconv_z", "cape_z", "z500_z", "t850_z", "ascent500_z",
    ]).reset_index(drop=True)
    mech_df = mech_df.rename(columns={"mean_heat_excess_z": "heat_z"})

    return core_df, mech_df, ref_stats, selected_mode


# =============================================================================
# Model helpers
# =============================================================================


def fit_glm(formula: str, df: pd.DataFrame):
    try:
        return smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit(cov_type="HC3")
    except Exception:
        try:
            return smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()
        except Exception as e:
            warnings.warn(f"GLM failed: {formula}\n{e}")
            return None


def average_pp_per_decade(model, df: pd.DataFrame) -> float:
    if model is None or df.empty:
        return np.nan
    d0 = df.copy()
    d1 = df.copy()
    d1["year10"] = d1["year10"] + 1.0
    p0 = np.asarray(model.predict(d0), dtype=float)
    p1 = np.asarray(model.predict(d1), dtype=float)
    return float(np.nanmean(p1 - p0) * 100.0)


def _subset_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    return df.loc[df["climate_region"].astype(str) == region].copy()


# =============================================================================
# Panel a: continuous spatial trend and regional trend
# =============================================================================


def _trend_formula_local() -> str:
    return "no_recovery ~ year10 + duration_z + heat_z + logfoot_z + doy_sin + doy_cos"


def build_spatial_trend(core_df: pd.DataFrame, heatgrid_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Use the native heat-footprint cells rather than event centroids.
    # This keeps the spatial panel consistent with the underlying ~25 km grid
    # and removes the artificial empty gaps produced by coarse 1° binning.
    join_cols = [
        "event_uid", "year", "climate_region", "no_recovery", "year10",
        "duration_z", "heat_z", "logfoot_z", "doy_sin", "doy_cos",
    ]
    base = core_df[join_cols].drop_duplicates(["event_uid", "year"]).copy()
    dat = heatgrid_df.merge(base, on=["event_uid", "year"], how="inner")
    dat = dat.drop_duplicates(["event_uid", "coord_key"]).copy()
    dat = dat.loc[
        to_num(dat["longitude"]).between(cfg.lon_min, cfg.lon_max)
        & to_num(dat["latitude"]).between(cfg.lat_min, cfg.lat_max)
    ].copy()

    if dat.empty:
        return pd.DataFrame(), pd.DataFrame()

    dx = cfg.native_grid_deg_lon if cfg.native_grid_deg_lon is not None else infer_native_grid_resolution(dat["longitude"], fallback=0.25)
    dy = cfg.native_grid_deg_lat if cfg.native_grid_deg_lat is not None else infer_native_grid_resolution(dat["latitude"], fallback=0.25)
    print(f"[INFO] Spatial panel uses native footprint cells | inferred grid ≈ {dx:.4f}° × {dy:.4f}°")

    dat["lon_cell"] = np.round(to_num(dat["longitude"]) / dx) * dx
    dat["lat_cell"] = np.round(to_num(dat["latitude"]) / dy) * dy

    rows = []
    for (lonc, latc), g in dat.groupby(["lon_cell", "lat_cell"], observed=True):
        if len(g) < cfg.min_events_per_spatial_bin:
            continue
        g = g.dropna(subset=["no_recovery", "year10", "duration_z", "heat_z", "logfoot_z", "doy_sin", "doy_cos"]).copy()
        if len(g) < cfg.min_events_per_spatial_bin:
            continue
        m = fit_glm(_trend_formula_local(), g)
        trend_val = average_pp_per_decade(m, g)
        if not np.isfinite(trend_val):
            continue
        rows.append({
            "lon0": float(lonc) - dx / 2.0,
            "lon1": float(lonc) + dx / 2.0,
            "lat0": float(latc) - dy / 2.0,
            "lat1": float(latc) + dy / 2.0,
            "lonc": float(lonc),
            "latc": float(latc),
            "n": int(len(g)),
            "trend_pp_per_dec": float(trend_val),
            "mean_no_recovery_pct": float(to_num(g["no_recovery"]).mean() * 100.0),
        })
    grid_df = pd.DataFrame(rows)

    # Region bars remain event-level by design.
    reg_rows = []
    for reg in PAPER7_ORDER:
        g = core_df.loc[core_df["climate_region"].astype(str) == reg].copy()
        if len(g) < cfg.min_events_per_region_trend:
            reg_rows.append({
                "climate_region": reg,
                "n": int(len(g)),
                "trend_pp_per_dec": np.nan,
                "mean_no_recovery_pct": float(to_num(g["no_recovery"]).mean() * 100.0) if len(g) else np.nan,
            })
            continue
        m = fit_glm(_trend_formula_local(), g)
        reg_rows.append({
            "climate_region": reg,
            "n": int(len(g)),
            "trend_pp_per_dec": average_pp_per_decade(m, g),
            "mean_no_recovery_pct": float(to_num(g["no_recovery"]).mean() * 100.0),
        })
    region_df = pd.DataFrame(reg_rows)
    return grid_df, region_df



# =============================================================================
# Panel b: rolling continuous outcome
# =============================================================================


def rolling_window_centers(years: Sequence[int], width: int, step: int) -> List[int]:
    years = sorted(set(int(y) for y in years if np.isfinite(y)))
    if not years:
        return []
    ymin, ymax = min(years), max(years)
    half = width // 2
    centers = []
    c = ymin + half
    while c <= ymax - half:
        centers.append(int(c))
        c += step
    if not centers:
        centers = [int(np.median(years))]
    return centers


def window_subset(df: pd.DataFrame, center: int, width: int) -> pd.DataFrame:
    half = width // 2
    return df.loc[(df["year"] >= center - half) & (df["year"] <= center + half)].copy()


def summarize_rolling_continuous(core_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    centers = rolling_window_centers(core_df["year"].tolist(), cfg.rolling_years, cfg.rolling_step)
    for c in centers:
        win = window_subset(core_df, c, cfg.rolling_years)
        for label, g in [("All", win), (cfg.target_region, _subset_region(win, cfg.target_region))]:
            if len(g) < 30:
                continue
            rows.append({
                "window_center": c,
                "subset": label,
                "n": int(len(g)),
                "mean_rainfrac_pct": float(to_num(g["mean_rainfrac_main"]).mean() * 100.0),
                "auc_rainfrac_pct": float(to_num(g["auc_rainfrac_main"]).mean() * 100.0),
                "no_recovery_pct": float(to_num(g["no_recovery"]).mean() * 100.0),
            })
    return pd.DataFrame(rows)


def bootstrap_rolling_continuous(core_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    years = sorted(core_df["year"].dropna().astype(int).unique().tolist())
    if not years:
        return pd.DataFrame()

    boot_rows = []
    for b in range(cfg.bootstrap_n):
        sampled_years = rng.choice(years, size=len(years), replace=True)
        parts = []
        for j, yr in enumerate(sampled_years):
            g = core_df.loc[core_df["year"] == yr].copy()
            if g.empty:
                continue
            g["boot_sample"] = j
            parts.append(g)
        if not parts:
            continue
        boot = pd.concat(parts, ignore_index=True)
        rs = summarize_rolling_continuous(boot, cfg)
        rs["boot_id"] = b
        boot_rows.append(rs)

    if not boot_rows:
        return pd.DataFrame()
    boot_df = pd.concat(boot_rows, ignore_index=True)

    rows = []
    for (center, subset), g in boot_df.groupby(["window_center", "subset"], observed=True):
        vals = to_num(g["mean_rainfrac_pct"]).dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        rows.append({
            "window_center": center,
            "subset": subset,
            "mean_rainfrac_pct_lcl": float(np.nanpercentile(vals, 2.5)),
            "mean_rainfrac_pct_ucl": float(np.nanpercentile(vals, 97.5)),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Panel d: definition-space sensitivity
# =============================================================================


def outcome_from_definition(post_df: pd.DataFrame, precip_thr: float, frac_thr: float, coverage_min: float) -> pd.DataFrame:
    col = f"rainfrac_p{str(precip_thr).replace('.', '_')}"
    if col not in post_df.columns:
        raise KeyError(f"Required column missing: {col}")
    g = post_df.copy()
    g["hit"] = ((to_num(g["coverage"]) >= coverage_min) & (to_num(g[col]) >= frac_thr)).astype(int)
    event = g.groupby("event_uid", as_index=False)["hit"].max().rename(columns={"hit": "recovered_def"})
    event["no_recovery_def"] = 1 - event["recovered_def"]
    return event


def build_definition_space(core_df: pd.DataFrame, post_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    main_ids = set(core_df["event_uid"].tolist())
    sub_post = post_df.loc[post_df["event_uid"].isin(main_ids)].copy()

    base_all = "no_recovery ~ year10 + duration_z + heat_z + logfoot_z + doy_sin + doy_cos + C(climate_region)"
    base_reg = "no_recovery ~ year10 + duration_z + heat_z + logfoot_z + doy_sin + doy_cos"

    for pthr in cfg.precip_threshold_grid:
        for fthr in cfg.area_fraction_grid:
            outcome = outcome_from_definition(sub_post, pthr, fthr, cfg.usable_post_coverage)
            merged = core_df.merge(outcome[["event_uid", "no_recovery_def"]], on="event_uid", how="inner")
            if merged.empty:
                continue
            dat = merged.copy()
            dat["no_recovery"] = to_num(dat["no_recovery_def"]).fillna(0).astype(int)
            dat = dat.drop(columns=["no_recovery_def"])

            m_all = fit_glm(base_all, dat)
            rows.append({
                "subset": "All",
                "precip_thr": pthr,
                "frac_thr": fthr,
                "n": int(len(dat)),
                "trend_pp_per_dec": average_pp_per_decade(m_all, dat),
            })

            trg = _subset_region(dat, cfg.target_region)
            if len(trg) >= 40:
                m_reg = fit_glm(base_reg, trg)
                rows.append({
                    "subset": cfg.target_region,
                    "precip_thr": pthr,
                    "frac_thr": fthr,
                    "n": int(len(trg)),
                    "trend_pp_per_dec": average_pp_per_decade(m_reg, trg),
                })
    return pd.DataFrame(rows)


# =============================================================================
# Panel c: explanatory-gap panel
# =============================================================================


def build_sequential_bridge(mech_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    g = _subset_region(mech_df, cfg.target_region)
    if g.empty or len(g) < cfg.min_mech_n_target:
        return pd.DataFrame(columns=["stage", "pp_per_dec", "n"])

    stages = [
        ("Raw time trend", "no_recovery ~ year10"),
        ("+ Event composition", "no_recovery ~ year10 + duration_z + heat_z + logfoot_z + doy_sin + doy_cos"),
        ("+ Land end-state", "no_recovery ~ year10 + duration_z + heat_z + logfoot_z + doy_sin + doy_cos + sm_deficit_z + bowen_z + rn_z"),
        ("+ Atmospheric end-state", "no_recovery ~ year10 + duration_z + heat_z + logfoot_z + doy_sin + doy_cos + rh_z + ivtconv_z + cape_z + z500_z + t850_z + ascent500_z"),
        ("+ Land + atmosphere", "no_recovery ~ year10 + duration_z + heat_z + logfoot_z + doy_sin + doy_cos + sm_deficit_z + bowen_z + rn_z + rh_z + ivtconv_z + cape_z + z500_z + t850_z + ascent500_z"),
    ]
    rows = []
    for label, formula in stages:
        m = fit_glm(formula, g)
        rows.append({
            "stage": label,
            "pp_per_dec": average_pp_per_decade(m, g),
            "n": int(len(g)),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================




def draw_panel_a(fig: plt.Figure, gs_cell, grid_df: pd.DataFrame, region_df: pd.DataFrame,
                 cfg: Config, show_panel_label: bool = False) -> None:
    subgs = gs_cell.subgridspec(
        2, 2,
        height_ratios=[18, 1.45],
        width_ratios=[2.35, 1.85],
        hspace=0.42,
        wspace=0.78,
    )

    if not HAS_CARTOPY:
        raise ImportError("Panel a requires cartopy.")

    axm = fig.add_subplot(subgs[0, 0], projection=ccrs.PlateCarree())
    axm.set_extent([cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max], crs=ccrs.PlateCarree())
    axm.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=0)
    axm.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.75, zorder=3)
    axm.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.55, zorder=3)
    try:
        states = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lakes",
            scale="50m",
            facecolor="none",
        )
        axm.add_feature(states, edgecolor="0.70", linewidth=0.38, zorder=3)
    except Exception:
        pass

    smap = None
    if not grid_df.empty and "trend_pp_per_dec" in grid_df.columns:
        finite_mask = np.isfinite(to_num(grid_df["trend_pp_per_dec"]).to_numpy(dtype=float))
        finite_df = grid_df.loc[finite_mask].copy()
        if not finite_df.empty:
            trend_vals = np.abs(to_num(finite_df["trend_pp_per_dec"]).to_numpy(dtype=float))
            vmax = float(np.nanmax(trend_vals)) if trend_vals.size else np.nan
            if np.isfinite(vmax) and vmax > 0:
                vmax = max(vmax, 1.0)
                norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
                cmap = plt.get_cmap("RdBu_r")
                for _, r in finite_df.iterrows():
                    rect = plt.Rectangle(
                        (r["lon0"], r["lat0"]), r["lon1"] - r["lon0"], r["lat1"] - r["lat0"],
                        facecolor=cmap(norm(r["trend_pp_per_dec"])), edgecolor="none", alpha=0.92,
                        transform=ccrs.PlateCarree(), zorder=1,
                    )
                    axm.add_patch(rect)
                smap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    unions = build_region_unions()
    for reg, geom in unions.items():
        try:
            axm.add_geometries(
                [geom], crs=ccrs.PlateCarree(), facecolor="none",
                edgecolor=PAPER7_COLORS.get(reg, "0.3"), linewidth=2.0, zorder=4,
            )
        except Exception:
            continue

    gl = axm.gridlines(draw_labels=True, linewidth=0.35, color="0.84", alpha=0.8, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": CFG.font_base - 2}
    gl.ylabel_style = {"size": CFG.font_base - 2}
    maybe_add_panel_label(axm, "a", show_panel_label=show_panel_label, x=-0.18, y=1.24)

    cax = fig.add_subplot(subgs[1, 0])
    if smap is not None:
        cbar = fig.colorbar(smap, cax=cax, orientation="horizontal")
        cbar.set_label("Trend (pp dec$^{-1}$)", fontsize=CFG.font_base - 1)
        cbar.ax.tick_params(labelsize=CFG.font_base - 2)
    else:
        cax.set_axis_off()
        axm.text(0.5, 0.5, "No valid spatial trend estimates", transform=axm.transAxes,
                 ha="center", va="center", fontsize=CFG.font_base - 3, color="0.35")

    axr = fig.add_subplot(subgs[0, 1])
    rg = region_df.copy()
    if rg.empty:
        rg = pd.DataFrame({"climate_region": PAPER7_ORDER, "trend_pp_per_dec": [np.nan] * len(PAPER7_ORDER)})
    rg = rg.set_index("climate_region").reindex(PAPER7_ORDER).reset_index()
    y = np.arange(len(rg))
    axr.axvline(0, color="0.60", lw=1.0)
    colors = [PAPER7_COLORS.get(r, COL_ALL) for r in rg["climate_region"]]
    axr.barh(y, rg["trend_pp_per_dec"], color=colors, height=0.68, alpha=0.96)

    xmin = float(np.nanmin(to_num(rg["trend_pp_per_dec"]))) if rg["trend_pp_per_dec"].notna().any() else -1.0
    xmax = float(np.nanmax(to_num(rg["trend_pp_per_dec"]))) if rg["trend_pp_per_dec"].notna().any() else 1.0
    axr.set_xlim(min(-1.05, xmin - 0.35), max(1.65, xmax + 0.25))

    for i, row in rg.iterrows():
        val = row["trend_pp_per_dec"]
        if np.isfinite(val):
            txt = f"{val:.2f}"
            if val >= 0:
                ha = "left"
                xpos = val + 0.06
            else:
                ha = "center"
                xpos = val / 2.0
        else:
            ha = "left"
            xpos = 0.05
            txt = "NA"
        axr.text(xpos, i, txt, va="center", ha=ha, fontsize=19)

    axr.set_yticks(y)
    axr.set_yticklabels(rg["climate_region"])
    for lab in axr.get_yticklabels():
        lab.set_horizontalalignment("right")
    axr.tick_params(axis="y", pad=4)
    axr.invert_yaxis()
    axr.set_xlabel("Trend (pp dec$^{-1}$)")
    tidy_axis(axr, "x")
    axr.spines["left"].set_visible(False)

    ax_blank = fig.add_subplot(subgs[1, 1])
    ax_blank.set_axis_off()


def draw_panel_b(ax, rolling_df: pd.DataFrame, rolling_ci: pd.DataFrame, cfg: Config,
                 show_panel_label: bool = False) -> None:
    for subset, color, lw in [("All", COL_CONT_ALL, 2.4), (cfg.target_region, COL_CONT_NW, 2.9)]:
        g = rolling_df.loc[rolling_df["subset"] == subset].sort_values("window_center")
        if g.empty:
            continue
        ci = rolling_ci.loc[rolling_ci["subset"] == subset].sort_values("window_center") if rolling_ci is not None else pd.DataFrame()
        if not ci.empty and {"mean_rainfrac_pct_lcl", "mean_rainfrac_pct_ucl"}.issubset(ci.columns):
            ax.fill_between(ci["window_center"], ci["mean_rainfrac_pct_lcl"], ci["mean_rainfrac_pct_ucl"], color=color, alpha=0.14)
        ax.plot(g["window_center"], g["mean_rainfrac_pct"], color=color, lw=lw, marker="o", ms=5.5, label=subset)
    ax.set_xlabel("Window center year")
    ax.set_ylabel("Rain-return (%, d1–10)")
    ax.legend(frameon=False, loc="lower left")
    tidy_axis(ax, "y")
    maybe_add_panel_label(ax, "b", show_panel_label=show_panel_label, x=-0.12, y=1.14)


def _draw_heat(ax, src_df: pd.DataFrame, subset: str, title: str, show_ylabel: bool = True):
    g = src_df.loc[src_df["subset"] == subset].copy()
    if g.empty:
        ax.text(0.5, 0.5, "No valid estimates", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None
    piv = g.pivot(index="frac_thr", columns="precip_thr", values="trend_pp_per_dec").sort_index(ascending=False)
    arr = piv.to_numpy(dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        ax.text(0.5, 0.5, "No valid estimates", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None
    vmax = np.nanmax(np.abs(arr))
    vmax = max(float(vmax), 0.5)
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([str(c) for c in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    if show_ylabel:
        ax.set_yticklabels([str(i) for i in piv.index])
        ax.set_ylabel("Area frac.")
    else:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.set_ylabel("")
    ax.set_xlabel("Rain thr. (mm d$^{-1}$)")
    ax.set_title(title, fontsize=CFG.font_base + 1, pad=10, y=1.03)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", fontsize=18)
    return im


def draw_panel_c(fig: plt.Figure, gs_cell, def_df: pd.DataFrame, cfg: Config,
                 show_panel_label: bool = False) -> None:
    subgs = gs_cell.subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.06)
    ax1 = fig.add_subplot(subgs[0, 0])
    ax2 = fig.add_subplot(subgs[0, 1], sharey=ax1)
    cax = fig.add_subplot(subgs[0, 2])
    im1 = _draw_heat(ax1, def_df, cfg.target_region, f"{cfg.target_region}", show_ylabel=True)
    _ = _draw_heat(ax2, def_df, "All", "CONUS overall", show_ylabel=False)
    if im1 is not None:
        cbar = fig.colorbar(im1, cax=cax)
        cbar.set_label("Trend (pp dec$^{-1}$)", fontsize=CFG.font_base - 1)
        cbar.ax.tick_params(labelsize=CFG.font_base - 2)
    maybe_add_panel_label(ax1, "d", show_panel_label=show_panel_label, x=-0.12, y=1.10)


def draw_panel_d(ax, seq_df: pd.DataFrame, show_panel_label: bool = False) -> None:
    if seq_df is None or seq_df.empty:
        ax.text(0.5, 0.5, "Sequential bridge estimates unavailable\n(mechanism subset too sparse under current sample rule)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        maybe_add_panel_label(ax, "c", show_panel_label=show_panel_label, x=-0.12, y=1.14)
        return
    colors = [COL_ALL, COL_EVENT, COL_LAND, COL_ATM, COL_BOTH]
    x = np.arange(len(seq_df))
    ax.bar(x, seq_df["pp_per_dec"], color=colors[:len(seq_df)], width=0.72)
    ax.plot(x, seq_df["pp_per_dec"], color="0.15", lw=1.1, alpha=0.55)
    ax.axhline(0, color="0.55", lw=1.0)
    for i, row in seq_df.iterrows():
        if np.isfinite(row["pp_per_dec"]):
            offset = 0.12 if row["pp_per_dec"] >= 0 else -0.12
            ax.text(i, row["pp_per_dec"] + offset, f"{row['pp_per_dec']:.2f}",
                    ha="center", va="bottom" if row["pp_per_dec"] >= 0 else "top", fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(seq_df["stage"], rotation=16, ha="right")
    ax.set_ylabel("NW trend (pp dec$^{-1}$)")
    tidy_axis(ax, "y")
    maybe_add_panel_label(ax, "c", show_panel_label=show_panel_label, x=-0.12, y=1.14)


def plot_panel_a_only(grid_df: pd.DataFrame, region_df: pd.DataFrame, cfg: Config) -> None:
    fig = plt.figure(figsize=(16.0, 6.5))
    gs = fig.add_gridspec(1, 1)
    draw_panel_a(fig, gs[0, 0], grid_df, region_df, cfg, show_panel_label=False)
    fig.subplots_adjust(top=0.96, bottom=0.12, left=0.05, right=0.985)
    savefig(fig, cfg.panel_dir / "Figure3_result3_bridge_NW_panel_a.png")


def plot_panel_b_only(rolling_df: pd.DataFrame, rolling_ci: pd.DataFrame, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    draw_panel_b(ax, rolling_df, rolling_ci, cfg, show_panel_label=False)
    fig.subplots_adjust(top=0.95, bottom=0.16, left=0.13, right=0.98)
    savefig(fig, cfg.panel_dir / "Figure3_result3_bridge_NW_panel_b.png")


def plot_panel_c_only(seq_df: pd.DataFrame, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    draw_panel_d(ax, seq_df, show_panel_label=False)
    fig.subplots_adjust(top=0.95, bottom=0.22, left=0.15, right=0.98)
    savefig(fig, cfg.panel_dir / "Figure3_result3_bridge_NW_panel_c.png")


def plot_panel_d_only(def_df: pd.DataFrame, cfg: Config) -> None:
    fig = plt.figure(figsize=(13.8, 5.8))
    gs = fig.add_gridspec(1, 1)
    draw_panel_c(fig, gs[0, 0], def_df, cfg, show_panel_label=False)
    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.07, right=0.985)
    savefig(fig, cfg.panel_dir / "Figure3_result3_bridge_NW_panel_d.png")


def plot_main_figure(grid_df: pd.DataFrame, region_df: pd.DataFrame,
                     rolling_df: pd.DataFrame, rolling_ci: pd.DataFrame, def_df: pd.DataFrame, seq_df: pd.DataFrame,
                     cfg: Config) -> None:
    fig = plt.figure(figsize=(17.8, 15.0))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.10, 0.92, 0.96],
        width_ratios=[1.0, 1.0],
        wspace=0.28,
        hspace=0.38,
    )

    draw_panel_a(fig, gs[0, :], grid_df, region_df, cfg, show_panel_label=False)

    axb = fig.add_subplot(gs[1, 0])
    draw_panel_b(axb, rolling_df, rolling_ci, cfg, show_panel_label=False)
    axc = fig.add_subplot(gs[1, 1])
    draw_panel_d(axc, seq_df, show_panel_label=False)

    draw_panel_c(fig, gs[2, :], def_df, cfg, show_panel_label=False)

    fig.subplots_adjust(top=0.968, bottom=0.055, left=0.055, right=0.985)
    savefig(fig, cfg.fig_dir / "Figure3_result3_bridge_NW_main.png")
# =============================================================================
# Main
# =============================================================================


def main(cfg: Config = CFG) -> None:
    print("=" * 80)
    print("Figure 3 bridge figure from raw event-window CSVs")
    print(f"Input root : {cfg.root_dir}")
    print(f"Output dir : {cfg.out_dir}")
    print("=" * 80)

    summary, post, heatgrid, note = build_or_load_cache(cfg)
    if summary.empty:
        raise RuntimeError("No usable events available after summarization.")

    core_df, mech_df, ref_stats, sample_rule = prepare_analysis_tables(summary, cfg)
    if core_df.empty:
        raise RuntimeError("Core analysis table is empty after all fallback sample rules.")
    print(f"[INFO] Selected core sample rule: {sample_rule}")
    print(f"[INFO] Core sample size: total={len(core_df):,} | target={int((core_df['climate_region'].astype(str) == cfg.target_region).sum()):,}")
    print(f"[INFO] Mechanism sample size: total={len(mech_df):,} | target={int((mech_df['climate_region'].astype(str) == cfg.target_region).sum()):,}")

    grid_df, region_df = build_spatial_trend(core_df, heatgrid, cfg)
    rolling_df = summarize_rolling_continuous(core_df, cfg)
    rolling_ci = bootstrap_rolling_continuous(core_df, cfg)
    def_df = build_definition_space(core_df, post, cfg)
    seq_df = build_sequential_bridge(mech_df, cfg)

    summary.to_csv(cfg.table_dir / "event_summary_bridge.csv", index=False, encoding="utf-8-sig")
    post.to_csv(cfg.table_dir / "event_post_daily_bridge.csv", index=False, encoding="utf-8-sig")
    heatgrid.to_csv(cfg.table_dir / "event_heat_grid_bridge.csv", index=False, encoding="utf-8-sig")
    core_df.to_csv(cfg.table_dir / "analysis_core_bridge.csv", index=False, encoding="utf-8-sig")
    mech_df.to_csv(cfg.table_dir / "analysis_mechanism_bridge.csv", index=False, encoding="utf-8-sig")
    grid_df.to_csv(cfg.table_dir / "spatial_trend_bins_bridge.csv", index=False, encoding="utf-8-sig")
    region_df.to_csv(cfg.table_dir / "regional_trend_bridge.csv", index=False, encoding="utf-8-sig")
    rolling_df.to_csv(cfg.table_dir / "rolling_continuous_bridge.csv", index=False, encoding="utf-8-sig")
    rolling_ci.to_csv(cfg.table_dir / "rolling_continuous_ci_bridge.csv", index=False, encoding="utf-8-sig")
    def_df.to_csv(cfg.table_dir / "definition_space_bridge.csv", index=False, encoding="utf-8-sig")
    seq_df.to_csv(cfg.table_dir / "sequential_bridge_models.csv", index=False, encoding="utf-8-sig")

    plot_main_figure(grid_df, region_df, rolling_df, rolling_ci, def_df, seq_df, cfg)
    plot_panel_a_only(grid_df, region_df, cfg)
    plot_panel_b_only(rolling_df, rolling_ci, cfg)
    plot_panel_c_only(seq_df, cfg)
    plot_panel_d_only(def_df, cfg)

    print("[DONE] Figure written to:")
    print(cfg.fig_dir / "Figure3_result3_bridge_NW_main.png")
    print("[DONE] Individual panels written to:")
    print(cfg.panel_dir / "Figure3_result3_bridge_NW_panel_a.png")
    print(cfg.panel_dir / "Figure3_result3_bridge_NW_panel_b.png")
    print(cfg.panel_dir / "Figure3_result3_bridge_NW_panel_c.png")
    print(cfg.panel_dir / "Figure3_result3_bridge_NW_panel_d.png")


if __name__ == "__main__":
    main(CFG)
