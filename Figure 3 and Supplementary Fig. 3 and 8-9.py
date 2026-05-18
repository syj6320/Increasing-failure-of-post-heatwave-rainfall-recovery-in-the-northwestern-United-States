# -*- coding: utf-8 -*-
"""
Result 2 structural rebuild for an event-level drought-heatwave paper
====================================================================

7-region climate-version revision
---------------------------------
1) Replaces the old quadrant split with a 7-region climate division:
   Northwest, Northern Great Plains, Midwest, Northeast,
   Southwest, Southern Great Plains, Southeast.
2) Assigns event region by majority of heatwave-core footprint cells
   mapped to U.S. states and then to climate regions.
3) Keeps all previous figure refinements:
   - no figure / subplot titles
   - shorter axis labels
   - panel letters outside upper-left corners
   - improved legend / colorbar placement
   - native-grid spatial trend map clipped to CONUS boundary
"""

from __future__ import annotations

import math
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import ConnectionPatch
from matplotlib.collections import PatchCollection

import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

try:
    from shapely.geometry import box, Polygon, MultiPolygon, Point
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
        r"E:\temp_events_ERA5_S1S6_Nature所有数据版本\zu"
    )
    out_dir: Optional[Path] = None

    # Recovery definition
    rain_threshold_mm: float = 1.0
    rainy_footprint_fraction: float = 0.25
    post_lag_max: int = 10

    # Sample filters
    usable_min_coverage: float = 0.80
    strict_min_coverage: float = 0.95
    min_detectable_coverage: float = 0.80
    require_no_overlap: bool = True
    require_no_censor: bool = True

    # Legacy quadrant split fallback only
    split_lon: float = -100.0
    split_lat: float = 37.0

    # Region scheme
    region_scheme: str = "paper7"
    region_assign_mode: str = "majority_footprint"

    # Reviewer-facing temporal settings
    rolling_period_width: int = 15
    rolling_period_step: int = 5
    moving_average_window: int = 7
    bootstrap_reps: int = 300
    bootstrap_seed: int = 42

    # Plotting
    rolling_window: int = 7
    dpi: int = 300
    panel_label_fs: int = 27
    figure_bg: str = "white"

    # Surface settings
    duration_grid_n: int = 22
    heat_grid_n: int = 24
    surface_ref_n: int = 1200
    kernel_bw_year: float = 5.0
    kernel_bw_duration: float = 1.3
    kernel_bw_heat: float = 0.35

    # Spatial settings: native ~25 km grid
    spatial_min_events: int = 10
    spatial_min_years: int = 8
    spatial_lon_step: Optional[float] = None
    spatial_lat_step: Optional[float] = None

    # Runtime / cache
    use_cache: bool = True
    progress_every: int = 250

    def __post_init__(self) -> None:
        if self.out_dir is None:
            self.out_dir = self.root_dir / f"_rd_{self.region_scheme}"
        self.cache_dir = self.out_dir / "cache"
        self.fig_dir = self.out_dir / "figures"
        self.table_dir = self.out_dir / "tables"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)


CFG = Config()


# =============================================================================
# Style helpers
# =============================================================================

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.titlesize": 15,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.bbox": "tight",
    "figure.facecolor": CFG.figure_bg,
    "axes.facecolor": "white",
})

COL_OBS = "#1f1f1f"
COL_STD = "#b23a48"
COL_COMP = "#7a7a7a"
COL_COND = "#c76d2c"
COL_INT = "#4c72b0"


def rolling_mean(y: pd.Series, window: int) -> pd.Series:
    return y.rolling(window=window, center=True, min_periods=max(3, window // 2)).mean()


def cluster_bootstrap_ci(values: np.ndarray, n_boot: int = 300, seed: int = 42) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = rng.choice(arr, size=arr.size, replace=True)
        boots[i] = np.nanmean(samp)
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def slope_per_decade(year: np.ndarray, value: np.ndarray, weight: Optional[np.ndarray] = None) -> float:
    x = np.asarray(year, dtype=float)
    y = np.asarray(value, dtype=float)
    if weight is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = np.asarray(weight, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]
    if len(x) < 3:
        return np.nan
    x0 = x - np.average(x, weights=w)
    y0 = y - np.average(y, weights=w)
    denom = np.sum(w * x0 * x0)
    if denom <= 0:
        return np.nan
    return float(np.sum(w * x0 * y0) / denom * 10.0)


def rolling_window_pairs(years: Sequence[int], width: int, step: int) -> List[Tuple[int, int]]:
    years = sorted(set(int(y) for y in years))
    if not years:
        return []
    ymin, ymax = min(years), max(years)
    out: List[Tuple[int, int]] = []
    for start in range(ymin, ymax - width + 2, step):
        out.append((start, start + width - 1))
    return out


def savefig(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def add_panel_label(ax, label: str, x: float = -0.17, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=CFG.panel_label_fs,
        fontweight="bold",
        clip_on=False,
    )


# =============================================================================
# 7-region climate definitions
# =============================================================================

PAPER7_STATE_TO_REGION = {
    # Northwest
    "WA": "Northwest",
    "OR": "Northwest",
    "ID": "Northwest",

    # Northern Great Plains
    "MT": "Northern Great Plains",
    "WY": "Northern Great Plains",
    "ND": "Northern Great Plains",
    "SD": "Northern Great Plains",
    "NE": "Northern Great Plains",

    # Southwest
    "CA": "Southwest",
    "NV": "Southwest",
    "AZ": "Southwest",
    "UT": "Southwest",
    "CO": "Southwest",
    "NM": "Southwest",

    # Southern Great Plains
    "KS": "Southern Great Plains",
    "OK": "Southern Great Plains",
    "TX": "Southern Great Plains",
    "AR": "Southern Great Plains",
    "LA": "Southern Great Plains",

    # Midwest
    "MN": "Midwest",
    "IA": "Midwest",
    "MO": "Midwest",
    "WI": "Midwest",
    "IL": "Midwest",
    "IN": "Midwest",
    "MI": "Midwest",
    "OH": "Midwest",

    # Northeast
    "PA": "Northeast",
    "NY": "Northeast",
    "NJ": "Northeast",
    "DE": "Northeast",
    "MD": "Northeast",
    "CT": "Northeast",
    "RI": "Northeast",
    "MA": "Northeast",
    "VT": "Northeast",
    "NH": "Northeast",
    "ME": "Northeast",
    "WV": "Northeast",

    # Southeast
    "VA": "Southeast",
    "NC": "Southeast",
    "SC": "Southeast",
    "GA": "Southeast",
    "FL": "Southeast",
    "AL": "Southeast",
    "MS": "Southeast",
    "TN": "Southeast",
    "KY": "Southeast",
}

PAPER7_ORDER = [
    "Northwest",
    "Northern Great Plains",
    "Midwest",
    "Northeast",
    "Southwest",
    "Southern Great Plains",
    "Southeast",
]

PAPER7_COLORS = {
    "Northwest": "#d95f02",
    "Northern Great Plains": "#5ab4ac",
    "Midwest": "#ca0020",
    "Northeast": "#8073ac",
    "Southwest": "#66a61e",
    "Southern Great Plains": "#1b9e77",
    "Southeast": "#f1a340",
}


def get_region_meta(cfg: Config) -> Dict[str, Dict]:
    if cfg.region_scheme != "paper7":
        raise ValueError(f"This script is fixed to paper7, but got region_scheme={cfg.region_scheme}")
    return {
        "state_to_region": PAPER7_STATE_TO_REGION,
        "order": PAPER7_ORDER,
        "colors": PAPER7_COLORS,
    }


REGION_COLORS = PAPER7_COLORS

# Color used for the national/all-event aggregate in regional bar panels.
# This variable is called in plot_supp_cumulative_recovery_metrics().
ALL_COLOR = "#4d4d4d"


# =============================================================================
# Utilities
# =============================================================================

def discover_event_files(root_dir: Path) -> List[Path]:
    fps = sorted(root_dir.glob("*/*event_*_window.csv"))
    if not fps:
        fps = sorted(root_dir.rglob("event_*_window.csv"))
    return fps


_DATE_COLS = ["date", "event_start", "event_end", "grid_start", "grid_end", "next_grid_start"]
_NUMERIC_COLS = [
    "temp_air", "soil_moist", "longitude", "latitude", "T90", "SM10", "dry_lag1", "heat3",
    "year", "doy", "lon_round", "lat_round", "event_id", "lag_day_event", "lag_day_grid",
    "is_heat_period_event", "is_heat_period_grid", "is_post_event_0_10_nominal",
    "is_post_grid_0_10_nominal", "overlaps_next_local_heat", "is_post_event_0_10_censored",
    "is_post_grid_0_10_censored", "precipitation", "wind_speed", "relative_humidity",
    "convective_available_potential_energy_mean", "vertically_integrated_moisture_divergence_mean",
    "temperature_850hPa_mean", "geopotential_500hPa", "vertical_velocity_500hPa",
    "Bowen_ratio", "Rn", "wind250", "wind850",
]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        "temperature_850hPa": "temperature_850hPa_mean",
        "temperature_850hPa_mean_mean": "temperature_850hPa_mean",
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


_FILE_RE = re.compile(r"event_(\d{4})_(\d+)_window\.csv")


def _event_uid_from_path(path: Path, df: pd.DataFrame) -> str:
    m = _FILE_RE.search(path.name)
    if m:
        y = int(m.group(1))
        eid = int(m.group(2))
        return f"{y}_{eid:05d}"
    year = int(pd.to_numeric(df.get("year", pd.Series([np.nan])).dropna().iloc[0])) if "year" in df else -1
    eid = int(pd.to_numeric(df.get("event_id", pd.Series([np.nan])).dropna().iloc[0])) if "event_id" in df else -1
    return f"{year}_{eid:05d}"


def infer_precip_scale(event_files: Sequence[Path], sample_n: int = 40) -> Tuple[float, str]:
    if not event_files:
        return 1.0, "no_files_found_assume_mm"
    qs: List[float] = []
    idx = np.linspace(0, len(event_files) - 1, min(sample_n, len(event_files))).astype(int)
    for i in idx:
        fp = event_files[i]
        try:
            df = pd.read_csv(fp, low_memory=False)
            df = _clean_columns(df)
            if "precipitation" not in df.columns:
                continue
            pr = pd.to_numeric(df["precipitation"], errors="coerce").dropna()
            if not pr.empty:
                qs.append(float(pr.quantile(0.99)))
        except Exception:
            continue
    if not qs:
        return 1.0, "precipitation_column_missing_assume_mm"
    q99 = float(np.nanmedian(qs))
    if q99 < 0.2:
        return 1000.0, f"auto_meters_to_mm_q0.99={q99:.6g}"
    return 1.0, f"assume_mm_q0.99={q99:.6g}"


def assign_region4(lon: float, lat: float, split_lon: float, split_lat: float) -> str:
    ew = "West" if lon <= split_lon else "East"
    ns = "South" if lat <= split_lat else "North"
    return f"{ew}-{ns}"


def _safe_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _bool_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _mean_heat_excess(heat_df: pd.DataFrame) -> float:
    if heat_df.empty or "temp_air" not in heat_df.columns or "T90" not in heat_df.columns:
        return np.nan
    exc = pd.to_numeric(heat_df["temp_air"], errors="coerce") - pd.to_numeric(heat_df["T90"], errors="coerce")
    exc = exc.clip(lower=0)
    return float(exc.mean()) if exc.notna().any() else np.nan


# =============================================================================
# Region assignment by state polygons
# =============================================================================

_STATE_GEOM_CACHE: Dict[str, List[Dict]] = {}
_COORD_REGION_CACHE: Dict[Tuple[float, float], Optional[str]] = {}
_REGION_GEOM_CACHE: Dict[str, Dict[str, object]] = {}


def load_conus_state_geometries(region_meta: Dict) -> List[Dict]:
    cache_key = "|".join(sorted(region_meta["state_to_region"].keys()))
    if cache_key in _STATE_GEOM_CACHE:
        return _STATE_GEOM_CACHE[cache_key]

    if not (HAS_CARTOPY and HAS_SHAPELY):
        raise ImportError("cartopy + shapely are required for climate-region assignment by state polygon.")

    shp = shpreader.natural_earth(
        resolution="50m",
        category="cultural",
        name="admin_1_states_provinces_lakes",
    )
    reader = shpreader.Reader(shp)
    keep = set(region_meta["state_to_region"].keys())

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

        out.append({
            "abbr": postal,
            "bounds": geom.bounds,
            "geom": geom,
        })

    if not out:
        raise RuntimeError("No CONUS state geometries could be loaded for region assignment.")

    _STATE_GEOM_CACHE[cache_key] = out
    return out


def load_climate_region_geometries(region_meta: Dict) -> Dict[str, object]:
    cache_key = "|".join(sorted(region_meta["state_to_region"].keys()))
    if cache_key in _REGION_GEOM_CACHE:
        return _REGION_GEOM_CACHE[cache_key]
    geoms = load_conus_state_geometries(region_meta)
    by_region: Dict[str, List[object]] = {r: [] for r in region_meta["order"]}
    for rec in geoms:
        reg = region_meta["state_to_region"].get(rec["abbr"])
        if reg is not None:
            by_region.setdefault(reg, []).append(rec["geom"])
    out: Dict[str, object] = {}
    for reg, parts in by_region.items():
        if not parts:
            continue
        out[reg] = unary_union(parts)
    _REGION_GEOM_CACHE[cache_key] = out
    return out


def draw_climate_region_boundaries(ax, cfg: Config, linewidth: float = 2.2) -> None:
    if not (HAS_CARTOPY and HAS_SHAPELY):
        return
    region_meta = get_region_meta(cfg)
    reg_geoms = load_climate_region_geometries(region_meta)
    for reg in region_meta["order"]:
        geom = reg_geoms.get(reg)
        if geom is None:
            continue
        try:
            ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor="none", edgecolor="black", linewidth=linewidth, zorder=4)
        except Exception:
            continue


def apply_map_ticks(ax) -> None:
    if not HAS_CARTOPY:
        return
    try:
        xticks = np.arange(-120, -65, 10)
        yticks = np.arange(25, 51, 5)
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f', degree_symbol='°'))
        ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f', degree_symbol='°'))
        ax.tick_params(axis='x', bottom=True, labelbottom=True, top=False, labeltop=False, pad=2)
        ax.tick_params(axis='y', left=True, labelleft=True, right=False, labelright=False, pad=2)
    except Exception:
        pass


def point_to_state_abbrev(lon: float, lat: float, region_meta: Dict) -> Optional[str]:
    geoms = load_conus_state_geometries(region_meta)
    pt = Point(float(lon), float(lat))

    for rec in geoms:
        minx, miny, maxx, maxy = rec["bounds"]
        if (minx <= lon <= maxx) and (miny <= lat <= maxy):
            if rec["geom"].covers(pt):
                return rec["abbr"]

    for rec in geoms:
        if rec["geom"].buffer(1e-9).covers(pt):
            return rec["abbr"]

    return None


def coord_to_region_label(lon: float, lat: float, cfg: Config) -> Optional[str]:
    key = (round(float(lon), 4), round(float(lat), 4))
    if key in _COORD_REGION_CACHE:
        return _COORD_REGION_CACHE[key]

    region_meta = get_region_meta(cfg)
    st = point_to_state_abbrev(float(lon), float(lat), region_meta)

    if st is None:
        region = assign_region4(lon, lat, cfg.split_lon, cfg.split_lat)
    else:
        region = region_meta["state_to_region"].get(st)

    _COORD_REGION_CACHE[key] = region
    return region


def assign_event_region(
    heat_coords: pd.DataFrame,
    centroid_lon: float,
    centroid_lat: float,
    cfg: Config,
) -> str:
    regs = []
    for row in heat_coords[["longitude", "latitude"]].dropna().itertuples(index=False):
        reg = coord_to_region_label(float(row.longitude), float(row.latitude), cfg)
        if reg is not None:
            regs.append(reg)

    if not regs:
        return assign_region4(centroid_lon, centroid_lat, cfg.split_lon, cfg.split_lat)

    cnt = Counter(regs).most_common()
    if len(cnt) == 1:
        return cnt[0][0]

    if cnt[0][1] == cnt[1][1]:
        reg_cent = coord_to_region_label(centroid_lon, centroid_lat, cfg)
        if reg_cent is not None:
            return reg_cent

    return cnt[0][0]


# =============================================================================
# Event-level summary construction
# =============================================================================

def summarize_one_event(path: Path, precip_scale: float, cfg: Config) -> Optional[Dict]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        warnings.warn(f"Failed to read {path.name}: {e}")
        return None

    df = _clean_columns(df)
    if "date" not in df.columns or "precipitation" not in df.columns:
        warnings.warn(f"Required columns missing in {path.name}; skipped.")
        return None

    for col in _DATE_COLS:
        if col in df.columns:
            df[col] = _safe_dt(df[col])
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "coord_key" not in df.columns:
        if {"longitude", "latitude"}.issubset(df.columns):
            lon_r = df["longitude"].round(4)
            lat_r = df["latitude"].round(4)
            df["coord_key"] = lon_r.astype(str) + "_" + lat_r.astype(str)
        else:
            warnings.warn(f"coord_key and lon/lat missing in {path.name}; skipped.")
            return None

    df["precip_mm"] = pd.to_numeric(df["precipitation"], errors="coerce") * precip_scale
    lag = pd.to_numeric(df.get("lag_day_event", pd.Series(np.nan, index=df.index)), errors="coerce")
    is_heat = _bool_num(df.get("is_heat_period_event", pd.Series(np.nan, index=df.index)))

    heat = df.loc[is_heat == 1].copy()
    if heat.empty:
        heat = df.loc[lag <= 0].copy()
    if heat.empty:
        warnings.warn(f"No heat-period rows found in {path.name}; skipped.")
        return None

    heat_coords = heat[["coord_key", "longitude", "latitude"]].dropna().drop_duplicates("coord_key")
    footprint = set(heat_coords["coord_key"].astype(str))
    footprint_ncells = int(len(footprint))
    if footprint_ncells == 0:
        warnings.warn(f"Zero heatwave footprint in {path.name}; skipped.")
        return None

    uid = _event_uid_from_path(path, df)
    year_val = int(pd.to_numeric(df.get("year", pd.Series([heat["date"].dt.year.mode().iloc[0]])).dropna().iloc[0]))
    event_start = heat["date"].min()
    event_end = heat["date"].max()
    if "event_start" in df.columns and df["event_start"].notna().any():
        event_start = df["event_start"].dropna().iloc[0]
    if "event_end" in df.columns and df["event_end"].notna().any():
        event_end = df["event_end"].dropna().iloc[0]
    duration = int((event_end - event_start).days + 1) if pd.notna(event_start) and pd.notna(event_end) else int(heat["date"].nunique())
    centroid_lon = float(heat_coords["longitude"].mean())
    centroid_lat = float(heat_coords["latitude"].mean())
    mean_heat_excess = _mean_heat_excess(heat)
    end_doy = int(event_end.dayofyear) if pd.notna(event_end) else int(pd.to_numeric(df.get("doy", pd.Series([np.nan]))).dropna().median())
    region4 = assign_event_region(heat_coords, centroid_lon, centroid_lat, cfg)

    post = df.loc[(lag >= 1) & (lag <= cfg.post_lag_max)].copy()
    daily_rows: List[Dict] = []
    for lag_day, g in post.groupby(lag.loc[post.index].astype(int)):
        g2 = g.dropna(subset=["coord_key"]).drop_duplicates("coord_key")
        g2 = g2.loc[g2["coord_key"].astype(str).isin(footprint)].copy()
        n_present = int(g2["coord_key"].nunique())
        rainy_cells = int(g2.loc[g2["precip_mm"] >= cfg.rain_threshold_mm, "coord_key"].nunique())
        coverage = n_present / footprint_ncells if footprint_ncells > 0 else np.nan
        rain_fraction = rainy_cells / footprint_ncells if footprint_ncells > 0 else np.nan
        recovered = int((coverage >= cfg.min_detectable_coverage) and (rain_fraction >= cfg.rainy_footprint_fraction))
        daily_rows.append({
            "lag_day": int(lag_day),
            "n_present": n_present,
            "coverage": coverage,
            "rainy_cells": rainy_cells,
            "rain_fraction": rain_fraction,
            "recovered_this_day": recovered,
        })
    daily = pd.DataFrame(daily_rows)

    if not daily.empty:
        rainfrac = pd.to_numeric(daily["rain_fraction"], errors="coerce")
        coverage_ser = pd.to_numeric(daily["coverage"], errors="coerce")
        mean_rain_fraction_w10 = float(rainfrac.mean()) if rainfrac.notna().any() else np.nan
        max_rain_fraction_w10 = float(rainfrac.max()) if rainfrac.notna().any() else np.nan
        cumulative_rain_fraction_w10 = float(rainfrac.sum()) if rainfrac.notna().any() else np.nan
        mean_coverage_w10 = float(coverage_ser.mean()) if coverage_ser.notna().any() else np.nan
    else:
        mean_rain_fraction_w10 = np.nan
        max_rain_fraction_w10 = np.nan
        cumulative_rain_fraction_w10 = np.nan
        mean_coverage_w10 = np.nan

    expected_lags = set(range(1, cfg.post_lag_max + 1))
    observed_lags = set(daily["lag_day"].tolist()) if not daily.empty else set()
    has_all_lags = expected_lags.issubset(observed_lags)
    min_cov = float(daily["coverage"].min()) if not daily.empty else np.nan
    noverlap = bool((_bool_num(post.get("overlaps_next_local_heat", pd.Series(0, index=post.index))) == 1).any())
    ncensor = bool((_bool_num(post.get("is_post_event_0_10_censored", pd.Series(0, index=post.index))) == 1).any())

    any_recovery = bool((daily["recovered_this_day"] == 1).any()) if not daily.empty else False
    recovery_day = int(daily.loc[daily["recovered_this_day"] == 1, "lag_day"].min()) if any_recovery else np.nan
    no_recovery = int(not any_recovery)

    usable_flag = has_all_lags and (min_cov >= cfg.usable_min_coverage if pd.notna(min_cov) else False)
    strict_flag = has_all_lags and (min_cov >= cfg.strict_min_coverage if pd.notna(min_cov) else False)
    if cfg.require_no_overlap:
        usable_flag = usable_flag and (not noverlap)
        strict_flag = strict_flag and (not noverlap)
    if cfg.require_no_censor:
        usable_flag = usable_flag and (not ncensor)
        strict_flag = strict_flag and (not ncensor)

    out = {
        "event_uid": uid,
        "source_file": str(path),
        "year": year_val,
        "event_start": event_start,
        "event_end": event_end,
        "duration": duration,
        "end_doy": end_doy,
        "centroid_lon": centroid_lon,
        "centroid_lat": centroid_lat,
        "region4": region4,
        "footprint_ncells": footprint_ncells,
        "footprint_area_km2": footprint_ncells * 25.0 * 25.0,
        "mean_heat_excess": mean_heat_excess,
        "no_recovery": no_recovery,
        "recovery_day": recovery_day,
        "recovered_by_day10": int(any_recovery),
        "has_all_lags": int(has_all_lags),
        "min_coverage": min_cov,
        "any_overlap": int(noverlap),
        "any_censor": int(ncensor),
        "usable_flag": int(usable_flag),
        "strict_flag": int(strict_flag),
        "day1_recovered": int(recovery_day == 1) if any_recovery else 0,
        "mean_rain_fraction_w10": mean_rain_fraction_w10,
        "max_rain_fraction_w10": max_rain_fraction_w10,
        "cumulative_rain_fraction_w10": cumulative_rain_fraction_w10,
        "mean_coverage_w10": mean_coverage_w10,
    }
    return out


def build_or_load_event_table(cfg: Config) -> Tuple[pd.DataFrame, str]:
    cache_fp = cfg.cache_dir / "result2_event_summary.csv"
    required_cache_cols = {
        "mean_rain_fraction_w10", "max_rain_fraction_w10", "cumulative_rain_fraction_w10", "mean_coverage_w10"
    }
    if cfg.use_cache and cache_fp.exists():
        df = pd.read_csv(cache_fp, parse_dates=["event_start", "event_end"])
        if required_cache_cols.issubset(df.columns):
            return df, "loaded_cached_event_summary"
        warnings.warn("Cached event summary is missing continuous-outcome columns; rebuilding cache.")

    event_files = discover_event_files(cfg.root_dir)
    if not event_files:
        raise FileNotFoundError(f"No event CSV files found under: {cfg.root_dir}")

    precip_scale, precip_note = infer_precip_scale(event_files)
    print(f"[INFO] Event files discovered : {len(event_files):,}")
    print(f"[INFO] Precipitation note    : {precip_note}")

    rows: List[Dict] = []
    for i, fp in enumerate(event_files, start=1):
        out = summarize_one_event(fp, precip_scale, cfg)
        if out is not None:
            rows.append(out)
        if (i % cfg.progress_every == 0) or (i == len(event_files)):
            print(f"[INFO] Summarized {i:,}/{len(event_files):,} files | retained {len(rows):,}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Event summary table is empty.")

    df = df.sort_values(["year", "event_uid"]).reset_index(drop=True)
    df.to_csv(cache_fp, index=False, encoding="utf-8-sig")
    return df, precip_note


# =============================================================================
# Analysis table preparation
# =============================================================================

def choose_analysis_sample(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    strict = df.loc[df["strict_flag"] == 1].copy()
    usable = df.loc[df["usable_flag"] == 1].copy()
    if len(strict) >= 500:
        return strict, f"strict | n={len(strict):,}"
    if len(usable) >= 500:
        warnings.warn("Strict sample is small/empty; falling back to usable sample.")
        return usable, f"usable_fallback | n={len(usable):,}"
    warnings.warn("Usable sample is also small; falling back to all summarized events.")
    return df.copy(), f"all_events_fallback | n={len(df):,}"


def prepare_analysis_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["year", "duration", "mean_heat_excess", "footprint_ncells", "end_doy", "no_recovery", "region4"])
    for c in ["mean_rain_fraction_w10", "max_rain_fraction_w10", "cumulative_rain_fraction_w10", "mean_coverage_w10"]:
        if c not in out.columns:
            out[c] = np.nan
    out["year"] = out["year"].astype(int)
    out["duration"] = out["duration"].clip(lower=1, upper=out["duration"].quantile(0.995))
    out["mean_heat_excess"] = out["mean_heat_excess"].clip(lower=0, upper=out["mean_heat_excess"].quantile(0.995))
    out["log_footprint"] = np.log1p(out["footprint_ncells"].clip(lower=1))
    out["sin_doy"] = np.sin(2.0 * np.pi * out["end_doy"] / 366.0)
    out["cos_doy"] = np.cos(2.0 * np.pi * out["end_doy"] / 366.0)
    out["year_scaled"] = out["year"].astype(float)
    out["region4"] = pd.Categorical(out["region4"], categories=get_region_meta(cfg)["order"], ordered=True)
    return out.reset_index(drop=True)


# =============================================================================
# Standardization models
# =============================================================================

def fit_glm_standardization(df: pd.DataFrame, with_year: bool = True):
    rhs_terms = [
        "bs(duration, df=4, include_intercept=False)",
        "bs(mean_heat_excess, df=4, include_intercept=False)",
        "bs(log_footprint, df=4, include_intercept=False)",
        "sin_doy + cos_doy",
    ]
    if with_year:
        rhs_terms = ["bs(year_scaled, df=5, include_intercept=False)"] + rhs_terms
    formula = "no_recovery ~ " + " + ".join(rhs_terms)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit(maxiter=200, disp=False)
    return model


def fit_glm_standardization_full(df: pd.DataFrame):
    formula = (
        "no_recovery ~ bs(year_scaled, df=5, include_intercept=False) + "
        "bs(duration, df=4, include_intercept=False) + "
        "bs(mean_heat_excess, df=4, include_intercept=False) + "
        "bs(log_footprint, df=4, include_intercept=False) + "
        "C(region4) + sin_doy + cos_doy"
    )
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit(maxiter=250, disp=False)
    return model


def _get_model_training_bounds(model) -> Dict[str, Tuple[float, float]]:
    if hasattr(model, "_training_bounds"):
        return getattr(model, "_training_bounds")
    bounds: Dict[str, Tuple[float, float]] = {}
    try:
        train_df = model.model.data.frame.copy()
    except Exception:
        train_df = None
    if train_df is not None:
        for col in ["year_scaled", "duration", "mean_heat_excess", "log_footprint"]:
            if col in train_df.columns:
                s = pd.to_numeric(train_df[col], errors="coerce").dropna()
                if not s.empty:
                    bounds[col] = (float(s.min()), float(s.max()))
    setattr(model, "_training_bounds", bounds)
    return bounds


def _prepare_predict_df(model, df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    bounds = _get_model_training_bounds(model)
    for col, (lo, hi) in bounds.items():
        if col in tmp.columns:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce").clip(lower=lo, upper=hi)
    try:
        train_df = model.model.data.frame
        if "region4" in tmp.columns and "region4" in train_df.columns and hasattr(train_df["region4"], "cat"):
            tmp["region4"] = pd.Categorical(
                tmp["region4"],
                categories=train_df["region4"].cat.categories,
                ordered=train_df["region4"].cat.ordered,
            )
    except Exception:
        pass
    return tmp


def _safe_predict(model, df: pd.DataFrame) -> np.ndarray:
    tmp = _prepare_predict_df(model, df)
    return np.asarray(model.predict(tmp), dtype=float)


def _mean_safe_predict(model, df: pd.DataFrame) -> float:
    p = _safe_predict(model, df)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan
    return float(np.nanmean(p))


def standardized_annual_series(model, ref_df: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    rows = []
    for y in years:
        tmp = ref_df.copy()
        tmp["year_scaled"] = float(y)
        pmean = _mean_safe_predict(model, tmp)
        rows.append({"year": int(y), "std_no_recovery": pmean})
    return pd.DataFrame(rows)


def composition_only_annual_series(model, analysis_df: pd.DataFrame, years: Sequence[int], baseline_year: int) -> pd.DataFrame:
    rows = []
    for y in years:
        obs_y = analysis_df.loc[analysis_df["year"] == y].copy()
        if obs_y.empty:
            continue
        obs_y["year_scaled"] = float(baseline_year)
        pmean = _mean_safe_predict(model, obs_y)
        rows.append({"year": int(y), "composition_only": pmean})
    return pd.DataFrame(rows)


def annual_observed_metrics(df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    rows: List[Dict] = []
    for year, g in df.groupby("year", observed=True):
        rec = pd.to_numeric(g["no_recovery"], errors="coerce")
        mean_rf = pd.to_numeric(g.get("mean_rain_fraction_w10", np.nan), errors="coerce")
        max_rf = pd.to_numeric(g.get("max_rain_fraction_w10", np.nan), errors="coerce")
        cum_rf = pd.to_numeric(g.get("cumulative_rain_fraction_w10", np.nan), errors="coerce")
        d1 = pd.to_numeric(g.loc[pd.to_numeric(g["recovered_by_day10"], errors="coerce") == 1, "day1_recovered"], errors="coerce")
        lo_nr, hi_nr = cluster_bootstrap_ci(rec.to_numpy(dtype=float), n_boot=cfg.bootstrap_reps, seed=cfg.bootstrap_seed + int(year))
        lo_rf, hi_rf = cluster_bootstrap_ci(mean_rf.to_numpy(dtype=float), n_boot=cfg.bootstrap_reps, seed=cfg.bootstrap_seed + 1000 + int(year))
        rows.append({
            "year": int(year),
            "n_events": int(len(g)),
            "no_recovery_obs": float(rec.mean()) if rec.notna().any() else np.nan,
            "no_recovery_lo": lo_nr,
            "no_recovery_hi": hi_nr,
            "duration_mean": float(pd.to_numeric(g["duration"], errors="coerce").mean()),
            "heat_excess_mean": float(pd.to_numeric(g["mean_heat_excess"], errors="coerce").mean()),
            "footprint_mean": float(pd.to_numeric(g["footprint_ncells"], errors="coerce").mean()),
            "end_doy_mean": float(pd.to_numeric(g["end_doy"], errors="coerce").mean()),
            "recovered_n": int(pd.to_numeric(g["recovered_by_day10"], errors="coerce").fillna(0).sum()),
            "day1_share_among_recovered": float(d1.mean()) if d1.notna().any() else np.nan,
            "mean_rain_fraction_w10": float(mean_rf.mean()) if mean_rf.notna().any() else np.nan,
            "mean_rain_fraction_lo": lo_rf,
            "mean_rain_fraction_hi": hi_rf,
            "max_rain_fraction_w10": float(max_rf.mean()) if max_rf.notna().any() else np.nan,
            "cumulative_rain_fraction_w10": float(cum_rf.mean()) if cum_rf.notna().any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def build_regional_models(df: pd.DataFrame) -> Dict[str, object]:
    models: Dict[str, object] = {}
    for region, sub in df.groupby("region4", observed=True):
        sub = sub.copy()
        if len(sub) < 300:
            continue
        try:
            models[str(region)] = fit_glm_standardization(sub, with_year=True)
        except Exception as e:
            warnings.warn(f"Regional model failed for {region}: {e}")
    return models


def descriptive_decomposition(
    model,
    analysis_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    early_years: Sequence[int],
    recent_years: Sequence[int],
    baseline_year: int,
) -> Dict[str, float]:
    early = analysis_df.loc[analysis_df["year"].isin(early_years)].copy()
    recent = analysis_df.loc[analysis_df["year"].isin(recent_years)].copy()
    if early.empty or recent.empty:
        return {"observed": np.nan, "conditional": np.nan, "composition": np.nan, "interaction": np.nan}

    observed_change = recent["no_recovery"].mean() - early["no_recovery"].mean()

    cond_early = standardized_annual_series(model, ref_df, sorted(set(early_years)))["std_no_recovery"].mean()
    cond_recent = standardized_annual_series(model, ref_df, sorted(set(recent_years)))["std_no_recovery"].mean()
    conditional_change = cond_recent - cond_early

    early_b = early.copy()
    recent_b = recent.copy()
    early_b["year_scaled"] = float(baseline_year)
    recent_b["year_scaled"] = float(baseline_year)
    comp_early = float(np.mean(model.predict(early_b)))
    comp_recent = float(np.mean(model.predict(recent_b)))
    composition_change = comp_recent - comp_early

    interaction_change = observed_change - conditional_change - composition_change
    return {
        "observed": float(observed_change),
        "conditional": float(conditional_change),
        "composition": float(composition_change),
        "interaction": float(interaction_change),
    }


def trend_decomposition_from_annual(
    annual: pd.DataFrame,
    std_annual: pd.DataFrame,
    comp_annual: pd.DataFrame,
) -> Dict[str, float]:
    tmp = annual.merge(std_annual, on="year", how="left").merge(comp_annual, on="year", how="left")
    w = pd.to_numeric(tmp.get("n_events", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    obs = slope_per_decade(tmp["year"].to_numpy(dtype=float), tmp["no_recovery_obs"].to_numpy(dtype=float), w)
    cond = slope_per_decade(tmp["year"].to_numpy(dtype=float), tmp["std_no_recovery"].to_numpy(dtype=float), w)
    comp = slope_per_decade(tmp["year"].to_numpy(dtype=float), tmp["composition_only"].to_numpy(dtype=float), w)
    inter = obs - cond - comp if np.isfinite(obs) and np.isfinite(cond) and np.isfinite(comp) else np.nan
    return {"observed": obs, "conditional": cond, "composition": comp, "interaction": inter}


def build_trend_decomposition_table(analysis_df: pd.DataFrame, model_full, regional_models: Dict[str, object], cfg: Config) -> pd.DataFrame:
    years = sorted(analysis_df["year"].unique())
    ref_df = analysis_df.copy()
    ann_all = annual_observed_metrics(analysis_df, cfg)
    std_all = standardized_annual_series(model_full, ref_df, years)
    comp_all = composition_only_annual_series(model_full, analysis_df, years, int(round(np.mean(years))))
    rows = [{"group": "All", **trend_decomposition_from_annual(ann_all, std_all, comp_all)}]
    for region in get_region_meta(cfg)["order"]:
        sub = analysis_df.loc[analysis_df["region4"] == region].copy()
        if sub.empty or region not in regional_models:
            rows.append({"group": region, "observed": np.nan, "conditional": np.nan, "composition": np.nan, "interaction": np.nan})
            continue
        yrs = sorted(sub["year"].unique())
        ann = annual_observed_metrics(sub, cfg)
        std = standardized_annual_series(regional_models[region], sub, yrs)
        comp = composition_only_annual_series(regional_models[region], sub, yrs, int(round(np.mean(yrs))))
        rows.append({"group": region, **trend_decomposition_from_annual(ann, std, comp)})
    return pd.DataFrame(rows)


def build_rolling_period_sensitivity(analysis_df: pd.DataFrame, model_full, cfg: Config) -> pd.DataFrame:
    years = sorted(analysis_df["year"].unique())
    ref_df = analysis_df.copy()
    rows: List[Dict] = []
    for start, end in rolling_window_pairs(years, cfg.rolling_period_width, cfg.rolling_period_step):
        sub = analysis_df.loc[analysis_df["year"].between(start, end)].copy()
        if len(sub) < 100:
            continue
        ann = annual_observed_metrics(sub, cfg)
        yrs = sorted(sub["year"].unique())
        std = standardized_annual_series(model_full, ref_df, yrs)
        comp = composition_only_annual_series(model_full, sub, yrs, int(round((start + end) / 2)))
        dec = trend_decomposition_from_annual(ann, std, comp)
        rows.append({
            "window_start": int(start),
            "window_end": int(end),
            "window_mid": float((start + end) / 2),
            "n_events": int(len(sub)),
            "mean_no_recovery": float(pd.to_numeric(sub["no_recovery"], errors="coerce").mean()),
            "mean_rain_fraction_w10": float(pd.to_numeric(sub.get("mean_rain_fraction_w10", np.nan), errors="coerce").mean()),
            "observed_slope_ppd": dec["observed"] * 100.0 if np.isfinite(dec["observed"]) else np.nan,
            "conditional_slope_ppd": dec["conditional"] * 100.0 if np.isfinite(dec["conditional"]) else np.nan,
            "composition_slope_ppd": dec["composition"] * 100.0 if np.isfinite(dec["composition"]) else np.nan,
            "interaction_slope_ppd": dec["interaction"] * 100.0 if np.isfinite(dec["interaction"]) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("window_mid").reset_index(drop=True)


def build_cluster_bootstrap_summary(analysis_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    dat = analysis_df.copy()
    dat["cluster_id"] = dat["year"].astype(str) + "|" + dat["region4"].astype(str)
    clusters = dat["cluster_id"].dropna().unique().tolist()
    rng = np.random.default_rng(cfg.bootstrap_seed)
    rows: List[Dict] = []
    for b in range(cfg.bootstrap_reps):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        parts = [dat.loc[dat["cluster_id"] == cid].copy() for cid in chosen if cid is not None]
        if not parts:
            continue
        boot = pd.concat(parts, ignore_index=True)
        ann = annual_observed_metrics(boot, cfg)
        if len(ann) < 3:
            continue
        w = pd.to_numeric(ann["n_events"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        rows.append({
            "rep": int(b),
            "no_recovery_slope_ppd": slope_per_decade(ann["year"].to_numpy(dtype=float), ann["no_recovery_obs"].to_numpy(dtype=float), w) * 100.0,
            "rain_fraction_slope_ppd": slope_per_decade(ann["year"].to_numpy(dtype=float), ann["mean_rain_fraction_w10"].to_numpy(dtype=float), w) * 100.0,
            "max_rain_fraction_slope_ppd": slope_per_decade(ann["year"].to_numpy(dtype=float), ann["max_rain_fraction_w10"].to_numpy(dtype=float), w) * 100.0,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Surfaces
# =============================================================================

def _subsample_reference(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def conditional_surface_year_duration(
    model,
    ref_df: pd.DataFrame,
    year_grid: np.ndarray,
    duration_grid: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    ref = _subsample_reference(ref_df, cfg.surface_ref_n)
    z = np.full((len(duration_grid), len(year_grid)), np.nan)
    for j, y in enumerate(year_grid):
        tmp_y = ref.copy()
        tmp_y["year_scaled"] = float(y)
        for i, d in enumerate(duration_grid):
            tmp = tmp_y.copy()
            tmp["duration"] = float(d)
            z[i, j] = _mean_safe_predict(model, tmp)
    return z


def conditional_surface_year_heat(
    model,
    ref_df: pd.DataFrame,
    year_grid: np.ndarray,
    heat_grid: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    ref = _subsample_reference(ref_df, cfg.surface_ref_n)
    z = np.full((len(heat_grid), len(year_grid)), np.nan)
    for j, y in enumerate(year_grid):
        tmp_y = ref.copy()
        tmp_y["year_scaled"] = float(y)
        for i, h in enumerate(heat_grid):
            tmp = tmp_y.copy()
            tmp["mean_heat_excess"] = float(h)
            z[i, j] = _mean_safe_predict(model, tmp)
    return z


def kernel_mean_surface(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    bw_x: float,
    bw_y: float,
) -> np.ndarray:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    z = df[z_col].to_numpy(dtype=float)
    out = np.full((len(y_grid), len(x_grid)), np.nan)
    for j, xv in enumerate(x_grid):
        wx = np.exp(-0.5 * ((x - xv) / bw_x) ** 2)
        for i, yv in enumerate(y_grid):
            wy = np.exp(-0.5 * ((y - yv) / bw_y) ** 2)
            w = wx * wy
            if np.sum(w) >= 3.0:
                out[i, j] = np.sum(w * z) / np.sum(w)
    return out


# =============================================================================
# Native-grid spatial trend diagnostics
# =============================================================================

def build_or_load_native_grid_event_table(analysis_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cache_fp = cfg.cache_dir / "result2_native_grid_event_table.csv"
    required_cols = {"mean_rain_fraction_w10", "duration", "mean_heat_excess", "log_footprint", "sin_doy", "cos_doy"}
    if cfg.use_cache and cache_fp.exists():
        cached = pd.read_csv(cache_fp)
        if required_cols.issubset(cached.columns):
            return cached
        warnings.warn("Cached native-grid table is missing required columns; rebuilding cache.")

    rows: List[Dict] = []
    needed = analysis_df[[
        "event_uid", "source_file", "year", "no_recovery", "duration", "mean_heat_excess",
        "footprint_ncells", "log_footprint", "end_doy", "sin_doy", "cos_doy", "mean_rain_fraction_w10"
    ]].copy()

    for i, rec in enumerate(needed.itertuples(index=False), start=1):
        fp = Path(rec.source_file)
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception as e:
            warnings.warn(f"Failed to read native-grid source {fp.name}: {e}")
            continue
        df = _clean_columns(df)
        for col in _NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "coord_key" not in df.columns:
            if {"longitude", "latitude"}.issubset(df.columns):
                df["coord_key"] = df["longitude"].round(4).astype(str) + "_" + df["latitude"].round(4).astype(str)
            else:
                continue

        lag = pd.to_numeric(df.get("lag_day_event", pd.Series(np.nan, index=df.index)), errors="coerce")
        is_heat = _bool_num(df.get("is_heat_period_event", pd.Series(np.nan, index=df.index)))
        heat = df.loc[is_heat == 1].copy()
        if heat.empty:
            heat = df.loc[lag <= 0].copy()
        if heat.empty:
            continue

        coords = (
            heat[["coord_key", "longitude", "latitude"]]
            .dropna(subset=["coord_key", "longitude", "latitude"])
            .drop_duplicates("coord_key")
        )
        for crow in coords.itertuples(index=False):
            rows.append({
                "event_uid": rec.event_uid,
                "year": int(rec.year),
                "longitude": float(crow.longitude),
                "latitude": float(crow.latitude),
                "no_recovery": int(rec.no_recovery),
                "duration": float(rec.duration),
                "mean_heat_excess": float(rec.mean_heat_excess),
                "footprint_ncells": float(rec.footprint_ncells),
                "log_footprint": float(rec.log_footprint),
                "end_doy": float(rec.end_doy),
                "sin_doy": float(rec.sin_doy),
                "cos_doy": float(rec.cos_doy),
                "mean_rain_fraction_w10": float(rec.mean_rain_fraction_w10) if pd.notna(rec.mean_rain_fraction_w10) else np.nan,
            })
        if (i % cfg.progress_every == 0) or (i == len(needed)):
            print(f"[INFO] Native-grid table: scanned {i:,}/{len(needed):,} events")

    native_df = pd.DataFrame(rows)
    if native_df.empty:
        raise RuntimeError("Native-grid event table is empty.")
    native_df.to_csv(cache_fp, index=False, encoding="utf-8-sig")
    return native_df


def weighted_linear_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]
    if len(x) < 3:
        return np.nan
    x0 = x - np.average(x, weights=w)
    y0 = y - np.average(y, weights=w)
    denom = np.sum(w * x0 * x0)
    if denom <= 0:
        return np.nan
    return np.sum(w * x0 * y0) / denom


def build_spatial_trend_table_native(native_df: pd.DataFrame, model_noyear, cfg: Config) -> pd.DataFrame:
    tmp = native_df.copy()
    tmp["pred_noyear"] = _safe_predict(model_noyear, tmp)
    tmp["resid_noyear"] = tmp["no_recovery"] - tmp["pred_noyear"]

    annual = tmp.groupby(["longitude", "latitude", "year"], observed=True).agg(
        n_events=("event_uid", "count"),
        obs_mean=("no_recovery", "mean"),
        adj_mean=("resid_noyear", "mean"),
    ).reset_index()

    rows: List[Dict] = []
    for (lon, lat), g in annual.groupby(["longitude", "latitude"], observed=True):
        total_events = int(g["n_events"].sum())
        n_years = int(g["year"].nunique())
        if total_events < cfg.spatial_min_events or n_years < cfg.spatial_min_years:
            continue
        slope_obs = weighted_linear_slope(g["year"].to_numpy(), g["obs_mean"].to_numpy(), g["n_events"].to_numpy())
        slope_adj = weighted_linear_slope(g["year"].to_numpy(), g["adj_mean"].to_numpy(), g["n_events"].to_numpy())
        rows.append({
            "longitude": float(lon),
            "latitude": float(lat),
            "n_events": total_events,
            "n_years": n_years,
            "obs_trend_ppd": slope_obs * 1000.0,
            "adj_trend_ppd": slope_adj * 1000.0,
        })
    return pd.DataFrame(rows)


def infer_native_grid_spacing(grid_df: pd.DataFrame, cfg: Config) -> Tuple[float, float]:
    if cfg.spatial_lon_step is not None and cfg.spatial_lat_step is not None:
        return float(cfg.spatial_lon_step), float(cfg.spatial_lat_step)

    def _median_step(vals: np.ndarray, fallback: float) -> float:
        vals = np.unique(np.round(vals[np.isfinite(vals)], 4))
        if len(vals) < 2:
            return fallback
        diffs = np.diff(np.sort(vals))
        diffs = diffs[diffs > 1e-6]
        if len(diffs) == 0:
            return fallback
        step = float(np.nanmedian(diffs))
        candidates = np.array([0.1, 0.125, 0.2, 0.25, 0.3125, 0.5, 1.0])
        nearest = candidates[np.argmin(np.abs(candidates - step))]
        if abs(nearest - step) / max(step, 1e-6) < 0.2:
            return float(nearest)
        return step

    lon_step = _median_step(grid_df["longitude"].to_numpy(dtype=float), 0.25)
    lat_step = _median_step(grid_df["latitude"].to_numpy(dtype=float), 0.25)
    return lon_step, lat_step


def get_conus_geometry():
    if not (HAS_CARTOPY and HAS_SHAPELY):
        return None
    try:
        shp = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
        reader = shpreader.Reader(shp)
        geoms = []
        for rec in reader.records():
            attrs = rec.attributes
            if attrs.get("ADM0_A3") == "USA" or attrs.get("NAME_LONG") == "United States":
                geoms.append(rec.geometry)
        if not geoms:
            return None
        usa = unary_union(geoms)
        conus = usa.intersection(box(-125.0, 24.0, -66.5, 50.0))
        return conus
    except Exception:
        return None


def polygon_to_patches(geom) -> List[MplPolygon]:
    patches: List[MplPolygon] = []
    if geom is None or geom.is_empty:
        return patches
    if HAS_SHAPELY and isinstance(geom, Polygon):
        patches.append(MplPolygon(np.asarray(geom.exterior.coords), closed=True))
    elif HAS_SHAPELY and isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            patches.extend(polygon_to_patches(part))
    else:
        try:
            if geom.geom_type == "Polygon":
                patches.append(MplPolygon(np.asarray(geom.exterior.coords), closed=True))
            elif geom.geom_type == "MultiPolygon":
                for part in geom.geoms:
                    patches.extend(polygon_to_patches(part))
        except Exception:
            pass
    return patches


# =============================================================================
# Plotting
# =============================================================================


_REGION_LABEL_POS_R2 = {
    "Northwest": (-121.0, 46.2),
    "Northern Great Plains": (-103.8, 46.0),
    "Midwest": (-89.7, 43.2),
    "Northeast": (-74.5, 43.7),
    "Southwest": (-113.2, 35.0),
    "Southern Great Plains": (-97.5, 32.5),
    "Southeast": (-82.7, 32.3),
}

_REGION_CONNECT_POS_R2 = {
    "Northwest": (-118.4, 45.8),
    "Northern Great Plains": (-101.4, 46.0),
    "Midwest": (-89.8, 43.8),
    "Northeast": (-74.0, 42.8),
    "Southwest": (-112.0, 35.5),
    "Southern Great Plains": (-97.0, 32.5),
    "Southeast": (-83.0, 32.4),
}


def _lighten_color(color: str, amount: float = 0.50):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def _select_key_regions_for_panel_b(trend_df: pd.DataFrame, cfg: Config, n_keep: int = 3) -> List[str]:
    dd = trend_df.loc[trend_df["group"] != "All"].copy()
    if dd.empty:
        return list(get_region_meta(cfg)["order"][:n_keep])
    dd["atten_abs"] = np.abs((pd.to_numeric(dd["observed"], errors="coerce") - pd.to_numeric(dd["conditional"], errors="coerce")) * 100.0)
    dd["obs_abs"] = np.abs(pd.to_numeric(dd["observed"], errors="coerce") * 100.0)
    dd = dd.sort_values(["atten_abs", "obs_abs"], ascending=False)
    chosen = [g for g in dd["group"].tolist() if g in get_region_meta(cfg)["order"]]
    if "Northwest" in get_region_meta(cfg)["order"] and "Northwest" not in chosen[:n_keep]:
        chosen = ["Northwest"] + [g for g in chosen if g != "Northwest"]
    out = []
    for g in chosen:
        if g not in out:
            out.append(g)
        if len(out) >= n_keep:
            break
    if len(out) < n_keep:
        for g in get_region_meta(cfg)["order"]:
            if g not in out:
                out.append(g)
            if len(out) >= n_keep:
                break
    return out


def plot_main_figure2(
    annual: pd.DataFrame,
    std_annual: pd.DataFrame,
    trend_df: pd.DataFrame,
    year_grid: np.ndarray,
    duration_grid: np.ndarray,
    heat_grid: np.ndarray,
    cond_dur: np.ndarray,
    cond_heat: np.ndarray,
    cfg: Config,
) -> None:
    fig = plt.figure(figsize=(18.4, 10.8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    plot_df = annual.copy().sort_values("year").reset_index(drop=True)
    if "std_no_recovery" not in plot_df.columns:
        plot_df = plot_df.merge(std_annual, on="year", how="left")
    plot_df["obs_smooth"] = rolling_mean(plot_df["no_recovery_obs"], cfg.moving_average_window)
    plot_df["std_smooth"] = rolling_mean(plot_df["std_no_recovery"], cfg.moving_average_window)
    plot_df["rain_smooth"] = rolling_mean(plot_df["mean_rain_fraction_w10"], cfg.moving_average_window)

    ax.plot(plot_df["year"], plot_df["no_recovery_obs"] * 100, color="0.85", lw=0.9)
    ax.plot(plot_df["year"], plot_df["obs_smooth"] * 100, color=COL_OBS, lw=2.5, label="Observed")
    ax.plot(plot_df["year"], plot_df["std_smooth"] * 100, color=COL_STD, lw=2.3, label="Standardized")
    ax.set_xlabel("Year")
    ax.set_ylabel("No-recovery (%)")
    ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(plot_df["year"], plot_df["rain_smooth"] * 100, color="#2a9d8f", lw=2.1, ls="--", label="Mean rain frac.")
    ax2.set_ylabel("Rain frac. (%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    add_panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    dd = trend_df.set_index("group")
    selected = ["All"] + _select_key_regions_for_panel_b(trend_df, cfg, n_keep=3)
    xpos = np.arange(len(selected))
    width = 0.28
    obs_vals = []
    std_vals = []
    colors = []
    for g in selected:
        row = dd.loc[g] if g in dd.index else pd.Series(dtype=float)
        obs = float(pd.to_numeric(row.get("observed", np.nan), errors="coerce") * 100.0)
        stdv = float(pd.to_numeric(row.get("conditional", np.nan), errors="coerce") * 100.0)
        obs_vals.append(obs)
        std_vals.append(stdv)
        colors.append("#4d4d4d" if g == "All" else REGION_COLORS.get(g, "#888888"))
    for i, (g, c, obs, stdv) in enumerate(zip(selected, colors, obs_vals, std_vals)):
        ax.bar(i - width / 2, obs, width=width, color=c, alpha=0.95, label="Observed" if i == 0 else None)
        ax.bar(i + width / 2, stdv, width=width, color=_lighten_color(c, 0.45), alpha=0.95, label="Standardized" if i == 0 else None)
        if np.isfinite(obs) and np.isfinite(stdv):
            ax.plot([i - width / 2, i + width / 2], [obs, stdv], color="0.30", lw=1.0, zorder=5)
            att = obs - stdv
            ax.text(i, max(obs, stdv) + 0.08, f"Δ={att:.2f}", ha="center", va="bottom", fontsize=11)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(selected, rotation=18, ha="right")
    ax.set_ylabel("Trend (pp decade$^{-1}$)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    add_panel_label(ax, "b")

    zc = np.asarray(cond_dur, dtype=float).copy()
    ax = fig.add_subplot(gs[1, 0])
    im = ax.pcolormesh(year_grid, duration_grid, zc * 100, shading="auto", cmap="magma_r")
    if np.isfinite(zc).any():
        cs = ax.contour(year_grid, duration_grid, zc * 100, levels=6, colors="white", linewidths=0.65, alpha=0.85)
        if len(cs.levels) > 0:
            ax.clabel(cs, fmt="%.0f", fontsize=17)
    ax.set_xlabel("Year")
    ax.set_ylabel("Duration (days)")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.18, fraction=0.08)
    cbar.set_label("No-recovery (%)")
    add_panel_label(ax, "c")

    zh = np.asarray(cond_heat, dtype=float).copy()
    ax = fig.add_subplot(gs[1, 1])
    im = ax.pcolormesh(year_grid, heat_grid, zh * 100, shading="auto", cmap="magma_r")
    if np.isfinite(zh).any():
        cs = ax.contour(year_grid, heat_grid, zh * 100, levels=6, colors="white", linewidths=0.65, alpha=0.85)
        if len(cs.levels) > 0:
            ax.clabel(cs, fmt="%.0f", fontsize=17)
    ax.set_xlabel("Year")
    ax.set_ylabel("Heat excess (°C)")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.18, fraction=0.08)
    cbar.set_label("No-recovery (%)")
    add_panel_label(ax, "d")

    savefig(fig, cfg.fig_dir / "Figure2_result2_structural_rebuild_paper7.png", dpi=cfg.dpi)


def plot_supp_sample_support(annual: pd.DataFrame, cfg: Config, sample_note: str) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"hspace": 0.14})

    ax = axs[0]
    ax.bar(annual["year"], annual["n_events"], color="0.65", width=0.9)
    ax.set_ylabel("Events")
    ax.grid(axis="y", alpha=0.2)
    add_panel_label(ax, "a")

    ax = axs[1]
    ax.plot(annual["year"], annual["no_recovery_obs"] * 100, color="0.82", lw=1.2)
    ax.plot(annual["year"], rolling_mean(annual["no_recovery_obs"], cfg.rolling_window) * 100, color=COL_OBS, lw=2.2)
    ax2 = ax.twinx()
    ax2.plot(
        annual["year"],
        rolling_mean(annual["day1_share_among_recovered"], cfg.rolling_window) * 100,
        color="#5c7cfa",
        lw=2.0,
    )
    ax.set_ylabel("No-recovery (%)")
    ax2.set_ylabel("Day-1 share (%)")
    ax.set_xlabel("Year")
    ax.grid(axis="y", alpha=0.2)
    add_panel_label(ax, "b")

    savefig(fig, cfg.fig_dir / "Supp_Fig_R2_01_sample_support_paper7.png", dpi=cfg.dpi)



def plot_supp_covariate_shift(annual: pd.DataFrame, df: pd.DataFrame, cfg: Config) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(14.4, 5.3), sharex=True, gridspec_kw={"wspace": 0.28})

    panels = [
        (axs[0], "duration_mean", "Duration (days)", "a"),
        (axs[1], "heat_excess_mean", "Heat excess (°C)", "b"),
    ]
    for ax, col, ylabel, lab in panels:
        ax.plot(annual["year"], annual[col], color="0.84", lw=1.0)
        ax.plot(annual["year"], rolling_mean(annual[col], cfg.rolling_window), color=COL_OBS, lw=2.3)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        add_panel_label(ax, lab)

    savefig(fig, cfg.fig_dir / "Supp_Fig_R2_02_covariate_shift_paper7.png", dpi=cfg.dpi)



def _setup_region_overview_map(ax) -> None:
    if HAS_CARTOPY:
        ax.set_extent([-125, -66.5, 24.0, 50.0], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="white", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.60, zorder=5)
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.28, edgecolor="0.65", zorder=4)
        conus = get_conus_geometry()
        reg_geoms = load_climate_region_geometries(get_region_meta(CFG))
        patches = []
        facecolors = []
        for region in get_region_meta(CFG)["order"]:
            geom = reg_geoms.get(region)
            if geom is None:
                continue
            if conus is not None:
                try:
                    geom = geom.intersection(conus)
                except Exception:
                    pass
            cell_patches = polygon_to_patches(geom)
            patches.extend(cell_patches)
            facecolors.extend([REGION_COLORS[region]] * len(cell_patches))
        if patches:
            pc = PatchCollection(
                patches,
                facecolor=facecolors,
                edgecolor="none",
                alpha=0.92,
                zorder=2,
                transform=ccrs.PlateCarree(),
            )
            ax.add_collection(pc)
        draw_climate_region_boundaries(ax, CFG, linewidth=1.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['geo'].set_visible(False)
    else:
        ax.set_xlim(-125, -66.5)
        ax.set_ylim(24.0, 50.0)
        ax.axis("off")


def _plot_region_rawstd_panel(ax, region_ann: pd.DataFrame, region: str, cfg: Config) -> None:
    sub = region_ann.loc[region_ann["region4"] == region].copy().sort_values("year")
    if sub.empty:
        ax.text(0.5, 0.5, "No support", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="0.4")
    else:
        ax.plot(sub["year"], sub["no_recovery_obs"] * 100, color="0.88", lw=0.8)
        ax.plot(sub["year"], rolling_mean(sub["no_recovery_obs"], cfg.rolling_window) * 100, color=REGION_COLORS[region], lw=2.4, label="Observed")
        if "std_no_recovery" in sub.columns and pd.to_numeric(sub["std_no_recovery"], errors="coerce").notna().any():
            ax.plot(sub["year"], rolling_mean(sub["std_no_recovery"], cfg.rolling_window) * 100, color=COL_STD, lw=2.1, ls="--", label="Standardized")
    year_min = int(region_ann["year"].min())
    year_max = int(region_ann["year"].max())
    ax.set_xlim(year_min, year_max)
    ax.set_ylim(0.0, 100.0)
    ax.set_title(region, fontsize=16, pad=5, color=REGION_COLORS[region])
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("No-rec. (%)", fontsize=13)
    ax.grid(alpha=0.12)
    ax.tick_params(axis='both', labelsize=11, length=2.8)
    for spine in ax.spines.values():
        spine.set_color(REGION_COLORS[region])
        spine.set_linewidth(1.05)


def plot_supp_raw_vs_std_by_region(df: pd.DataFrame, std_by_region: Dict[str, pd.DataFrame], cfg: Config) -> None:
    regions = get_region_meta(cfg)["order"]
    region_rows = []
    for region in regions:
        sub = df.loc[df["region4"] == region].copy()
        ann = annual_observed_metrics(sub, cfg)
        ann["region4"] = region
        std = std_by_region.get(region)
        if std is not None:
            ann = ann.merge(std, on="year", how="left")
        region_rows.append(ann)
    regional_ann = pd.concat(region_rows, ignore_index=True) if region_rows else pd.DataFrame()
    if regional_ann.empty:
        warnings.warn("Regional annual table is empty; skipping raw-vs-standardized region figure.")
        return

    fig = plt.figure(figsize=(17.2, 10.4), facecolor="white")
    map_ax = fig.add_axes([0.33, 0.28, 0.34, 0.40], projection=ccrs.PlateCarree() if HAS_CARTOPY else None)
    _setup_region_overview_map(map_ax)

    positions = {
        "Northwest": [0.05, 0.61, 0.24, 0.18],
        "Northern Great Plains": [0.36, 0.79, 0.22, 0.16],
        "Midwest": [0.64, 0.61, 0.24, 0.18],
        "Northeast": [0.76, 0.36, 0.20, 0.18],
        "Southeast": [0.66, 0.08, 0.24, 0.18],
        "Southern Great Plains": [0.37, 0.02, 0.26, 0.18],
        "Southwest": [0.07, 0.09, 0.24, 0.18],
    }
    target_fracs = {
        "Northwest": (0.95, 0.55),
        "Northern Great Plains": (0.50, 0.02),
        "Midwest": (0.05, 0.55),
        "Northeast": (0.00, 0.40),
        "Southeast": (0.30, 1.00),
        "Southern Great Plains": (0.50, 1.00),
        "Southwest": (0.95, 0.40),
    }
    panel_labels = ["a", "b", "c", "d", "e", "f", "g"]

    first_handles = None
    first_labels = None
    for i, region in enumerate(regions):
        ax = fig.add_axes(positions[region])
        _plot_region_rawstd_panel(ax, regional_ann, region, cfg)
        add_panel_label(ax, panel_labels[i], x=-0.16, y=1.12)
        if i == 0:
            first_handles, first_labels = ax.get_legend_handles_labels()
        x0, y0 = _REGION_CONNECT_POS_R2.get(region, _REGION_LABEL_POS_R2.get(region, (-100, 35)))
        fx, fy = target_fracs[region]
        con = ConnectionPatch(
            xyA=(x0, y0), coordsA=ccrs.PlateCarree()._as_mpl_transform(map_ax) if HAS_CARTOPY else map_ax.transData,
            xyB=(fx, fy), coordsB=ax.transAxes,
            color=REGION_COLORS[region], lw=0.95, alpha=0.95,
        )
        fig.add_artist(con)
    if first_handles and first_labels:
        fig.legend(first_handles, first_labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.50, -0.01))

    savefig(fig, cfg.fig_dir / "Supp_Fig_R2_03_raw_vs_standardized_by_region_paper7.png", dpi=cfg.dpi)



def plot_supp_decomposition(decomp_df: pd.DataFrame, cfg: Config) -> None:
    # Deliberately suppressed in the revised Result 2 build.
    # The old percentage-share decomposition was unstable when observed trends were small,
    # and the absolute comparison is now moved into Figure 2b.
    return



def plot_supp_surfaces(
    year_grid: np.ndarray,
    duration_grid: np.ndarray,
    heat_grid: np.ndarray,
    raw_dur: np.ndarray,
    cond_dur: np.ndarray,
    raw_heat: np.ndarray,
    cond_heat: np.ndarray,
    cfg: Config,
) -> None:
    # Suppressed in the revised build to avoid duplicating the main structural surfaces.
    return


def plot_spatial_maps(spatial_df: pd.DataFrame, native_grid_df: pd.DataFrame, cfg: Config) -> None:
    if spatial_df.empty:
        warnings.warn("Spatial trend table is empty; skipping map figure.")
        return

    lon_step, lat_step = infer_native_grid_spacing(native_grid_df, cfg)
    conus_geom = get_conus_geometry()

    fig = plt.figure(figsize=(16, 7.4))
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        axs = [fig.add_subplot(1, 2, 1, projection=proj), fig.add_subplot(1, 2, 2, projection=proj)]
    else:
        axs = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]

    vmax = float(np.nanpercentile(np.abs(np.r_[spatial_df["obs_trend_ppd"].to_numpy(), spatial_df["adj_trend_ppd"].to_numpy()]), 95))
    vmax = max(vmax, 5.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    panels = [
        (axs[0], "obs_trend_ppd", "a", "Raw trend"),
        (axs[1], "adj_trend_ppd", "b", "Adjusted trend"),
    ]

    for ax, col, lab, sublab in panels:
        if HAS_CARTOPY:
            ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f7f7f7", edgecolor="none", zorder=0)
            ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f3efe8", edgecolor="none", zorder=0)
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55, zorder=3)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45, zorder=3)
            ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.25, edgecolor="0.55", zorder=3)
            draw_climate_region_boundaries(ax, cfg, linewidth=1.8)
            apply_map_ticks(ax)
        else:
            ax.set_xlim(-125, -66.5)
            ax.set_ylim(24, 50)
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")

        patches: List[MplPolygon] = []
        colors: List[Tuple[float, float, float, float]] = []
        for row in spatial_df[["longitude", "latitude", col]].itertuples(index=False):
            if not np.isfinite(row[2]):
                continue
            if HAS_SHAPELY:
                cell = box(
                    float(row.longitude) - lon_step / 2.0,
                    float(row.latitude) - lat_step / 2.0,
                    float(row.longitude) + lon_step / 2.0,
                    float(row.latitude) + lat_step / 2.0,
                )
                if conus_geom is not None:
                    cell = cell.intersection(conus_geom)
                if cell is None or cell.is_empty:
                    continue
                new_patches = polygon_to_patches(cell)
                if not new_patches:
                    continue
                patches.extend(new_patches)
                colors.extend([cmap(norm(float(row[2])))] * len(new_patches))
            else:
                rect = MplPolygon(
                    np.array([
                        [float(row.longitude) - lon_step / 2.0, float(row.latitude) - lat_step / 2.0],
                        [float(row.longitude) + lon_step / 2.0, float(row.latitude) - lat_step / 2.0],
                        [float(row.longitude) + lon_step / 2.0, float(row.latitude) + lat_step / 2.0],
                        [float(row.longitude) - lon_step / 2.0, float(row.latitude) + lat_step / 2.0],
                    ]),
                    closed=True,
                )
                patches.append(rect)
                colors.append(cmap(norm(float(row[2]))))

        if patches:
            pc = PatchCollection(
                patches,
                facecolor=colors,
                edgecolor="none",
                linewidths=0.0,
                alpha=0.95,
                zorder=2,
                transform=ccrs.PlateCarree() if HAS_CARTOPY else ax.transData,
            )
            ax.add_collection(pc)

        add_panel_label(ax, lab, x=-0.10, y=1.16)
        ax.text(0.02, 0.02, sublab, transform=ax.transAxes, ha="left", va="bottom", fontsize=12,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))

    smap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    smap.set_array([])
    for ax in axs:
        cbar = fig.colorbar(smap, ax=ax, orientation="horizontal", pad=0.18, fraction=0.075, shrink=0.96)
        cbar.set_label("Trend (pp decade$^{-1}$)")
    fig.subplots_adjust(bottom=0.22, wspace=0.18)
    savefig(fig, cfg.fig_dir / "Supp_Fig_R2_06_spatial_trends_raw_vs_adjusted_paper7.png", dpi=cfg.dpi)


def plot_supp_rolling_sensitivity(rolling_df: pd.DataFrame, cfg: Config) -> None:
    if rolling_df.empty:
        return

    fig, axs = plt.subplots(2, 1, figsize=(14.8, 8.8), sharex=True, gridspec_kw={"hspace": 0.18})

    ax = axs[0]
    tmp = rolling_df.sort_values("window_mid")
    ax.plot(tmp["window_mid"], tmp["mean_no_recovery"] * 100, color=COL_OBS, lw=2.2, label="Mean no-recovery")
    ax.set_ylabel("Window mean (%)")
    ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(tmp["window_mid"], tmp["mean_rain_fraction_w10"] * 100, color="#2a9d8f", lw=2.2, ls="--", label="Mean rain fraction")
    ax2.set_ylabel("Rain fraction (%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    add_panel_label(ax, "a")

    ax = axs[1]
    for col, lab, color, marker, z in [
        ("observed_slope_ppd", "Observed slope", COL_OBS, "o", 4),
        ("conditional_slope_ppd", "Conditional", COL_COND, "s", 3),
        ("composition_slope_ppd", "Composition", COL_COMP, "^", 2),
    ]:
        y = pd.to_numeric(tmp[col], errors="coerce")
        ax.plot(tmp["window_mid"], y, color=color, lw=2.2 if col=="observed_slope_ppd" else 2.0, marker=marker, ms=4.5, label=lab, zorder=z)
    ax.axhline(0, color="0.75", lw=0.8)
    ax.set_xlabel("Window midpoint year")
    ax.set_ylabel("Trend (pp decade$^{-1}$)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(alpha=0.2)
    add_panel_label(ax, "b")

    savefig(fig, cfg.fig_dir / "Supp_Fig_R2_07_rolling_window_sensitivity.png", dpi=cfg.dpi)


def plot_supp_continuous_and_bootstrap(annual: pd.DataFrame, boot_df: pd.DataFrame, cfg: Config) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(15.8, 6.3), gridspec_kw={"wspace": 0.32})

    ax = axs[0]
    tmp = annual.copy().sort_values("year")
    x = pd.to_numeric(tmp["year"], errors="coerce")
    y1 = pd.to_numeric(tmp["mean_rain_fraction_w10"], errors="coerce")
    y2 = pd.to_numeric(tmp["max_rain_fraction_w10"], errors="coerce")
    if x.notna().any() and y1.notna().any():
        ax.plot(x, y1, color="#9ad9c8", lw=1.0)
        ax.plot(x, rolling_mean(y1, cfg.moving_average_window), color="#2a9d8f", lw=2.4, label="Mean rain fraction")
    if x.notna().any() and y2.notna().any():
        ax.plot(x, rolling_mean(y2, cfg.moving_average_window), color="#e76f51", lw=2.2, label="Max rain fraction")
    ax.set_xlabel("Year")
    ax.set_ylabel("Continuous recovery")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.2)
    add_panel_label(ax, "a")

    ax = axs[1]
    cols = ["no_recovery_slope_ppd", "rain_fraction_slope_ppd", "max_rain_fraction_slope_ppd"]
    labs = ["No-recovery", "Mean rain frac", "Max rain frac"]
    pos = np.arange(len(cols))
    shown = False
    for i, col in enumerate(cols):
        vals = pd.to_numeric(boot_df.get(col, np.nan), errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        shown = True
        parts = ax.violinplot(vals, positions=[i], widths=0.78, showmeans=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_alpha(0.55)
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        ax.plot([i, i], [lo, hi], color="0.2", lw=1.4)
    if not shown:
        ax.text(0.5, 0.5, "No bootstrap distributions available", transform=ax.transAxes, ha="center", va="center")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels(labs, rotation=15)
    ax.set_ylabel("Cluster-bootstrap trend\n(pp decade$^{-1}$)")
    ax.grid(axis="y", alpha=0.2)
    add_panel_label(ax, "b")

    savefig(fig, cfg.fig_dir / "Supp_Fig_R2_08_continuous_and_cluster_bootstrap.png", dpi=cfg.dpi)




def fit_linear_standardization_continuous(df: pd.DataFrame, outcome: str):
    sub = df.copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub.dropna(subset=[outcome, "year_scaled", "duration", "mean_heat_excess", "log_footprint", "region4", "sin_doy", "cos_doy"])
    formula = (
        f"{outcome} ~ bs(year_scaled, df=5, include_intercept=False) + "
        "bs(duration, df=4, include_intercept=False) + "
        "bs(mean_heat_excess, df=4, include_intercept=False) + "
        "bs(log_footprint, df=4, include_intercept=False) + "
        "C(region4) + sin_doy + cos_doy"
    )
    model = smf.ols(formula=formula, data=sub).fit()
    return model


def standardized_annual_series_continuous(model, ref_df: pd.DataFrame, years: Sequence[int], output_col: str) -> pd.DataFrame:
    rows = []
    for y in years:
        tmp = ref_df.copy()
        tmp["year_scaled"] = float(y)
        pred = np.asarray(model.predict(_prepare_predict_df(model, tmp)), dtype=float)
        pred = pred[np.isfinite(pred)]
        rows.append({"year": int(y), output_col: float(np.nanmean(pred)) if pred.size else np.nan})
    return pd.DataFrame(rows)


def build_regional_continuous_models(df: pd.DataFrame, outcome: str, cfg: Config) -> Dict[str, object]:
    models: Dict[str, object] = {}
    for region in get_region_meta(cfg)["order"]:
        sub = df.loc[df["region4"] == region].copy()
        if len(sub) < 200:
            continue
        try:
            models[str(region)] = fit_linear_standardization_continuous(sub, outcome)
        except Exception as e:
            warnings.warn(f"Regional continuous model failed for {region} ({outcome}): {e}")
    return models


def build_continuous_adjusted_annual_series(
    analysis_df: pd.DataFrame,
    outcome: str,
    cfg: Config,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    sub = analysis_df.copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub.dropna(subset=[
        outcome, "year", "year_scaled", "duration", "mean_heat_excess",
        "log_footprint", "region4", "sin_doy", "cos_doy"
    ])
    if sub.empty:
        return pd.DataFrame(columns=["year", "observed", "adjusted"]), {}

    years = sorted(sub["year"].unique())
    ref_df = sub.copy()
    full_model = fit_linear_standardization_continuous(sub, outcome)

    annual_obs = (
        sub.groupby("year", observed=True)[outcome]
        .mean()
        .reset_index()
        .rename(columns={outcome: "observed"})
    )
    annual_adj = standardized_annual_series_continuous(full_model, ref_df, years, "adjusted")
    annual_outcome = annual_obs.merge(annual_adj, on="year", how="outer").sort_values("year").reset_index(drop=True)

    regional_models = build_regional_continuous_models(sub, outcome, cfg)
    regional_out: Dict[str, pd.DataFrame] = {"All": annual_adj.copy()}

    for region in get_region_meta(cfg)["order"]:
        reg_sub = sub.loc[sub["region4"] == region].copy()
        if reg_sub.empty or region not in regional_models:
            continue
        yrs = sorted(reg_sub["year"].unique())
        reg_adj = standardized_annual_series_continuous(
            regional_models[region], reg_sub, yrs, "adjusted"
        )
        regional_out[region] = reg_adj.copy()

    return annual_outcome, regional_out


def build_regional_continuous_trend_table(
    regional_series: Dict[str, pd.DataFrame],
    value_col: str = "adjusted",
) -> pd.DataFrame:
    rows: List[Dict] = []
    for group, df in regional_series.items():
        if df is None or df.empty or value_col not in df.columns:
            continue
        tmp = df.copy().sort_values("year").reset_index(drop=True)
        x = pd.to_numeric(tmp["year"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(tmp[value_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        slope = slope_per_decade(x, y) if len(x) >= 3 else np.nan
        rows.append({"group": group, "trend_per_decade": slope})
    return pd.DataFrame(rows)


def plot_supp_cumulative_recovery_metrics(
    mean_series: pd.DataFrame,
    max_series: pd.DataFrame,
    cumulative_series: pd.DataFrame,
    cumulative_regional_trend: pd.DataFrame,
    cfg: Config,
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(16.8, 11.4), gridspec_kw={"wspace": 0.28, "hspace": 0.30})

    ax = axs[0, 0]
    tmp = mean_series.copy().sort_values("year").reset_index(drop=True)
    x = pd.to_numeric(tmp["year"], errors="coerce")
    y_obs = pd.to_numeric(tmp["observed"], errors="coerce") * 100.0
    y_adj = pd.to_numeric(tmp["adjusted"], errors="coerce") * 100.0
    if x.notna().any() and y_obs.notna().any():
        ax.plot(x, y_obs, color="0.86", lw=0.9)
        ax.plot(x, rolling_mean(y_obs, cfg.moving_average_window), color="#2a9d8f", lw=2.4, label="Observed")
    if x.notna().any() and y_adj.notna().any():
        ax.plot(x, rolling_mean(y_adj, cfg.moving_average_window), color=COL_STD, lw=2.2, ls="--", label="Adjusted")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean rain frac. (%)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    add_panel_label(ax, "a")

    ax = axs[0, 1]
    tmp = max_series.copy().sort_values("year").reset_index(drop=True)
    x = pd.to_numeric(tmp["year"], errors="coerce")
    y_obs = pd.to_numeric(tmp["observed"], errors="coerce") * 100.0
    y_adj = pd.to_numeric(tmp["adjusted"], errors="coerce") * 100.0
    if x.notna().any() and y_obs.notna().any():
        ax.plot(x, y_obs, color="0.86", lw=0.9)
        ax.plot(x, rolling_mean(y_obs, cfg.moving_average_window), color="#e76f51", lw=2.4, label="Observed")
    if x.notna().any() and y_adj.notna().any():
        ax.plot(x, rolling_mean(y_adj, cfg.moving_average_window), color=COL_STD, lw=2.2, ls="--", label="Adjusted")
    ax.set_xlabel("Year")
    ax.set_ylabel("Max rain frac. (%)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    add_panel_label(ax, "b")

    ax = axs[1, 0]
    tmp = cumulative_series.copy().sort_values("year").reset_index(drop=True)
    x = pd.to_numeric(tmp["year"], errors="coerce")
    y_obs = pd.to_numeric(tmp["observed"], errors="coerce") * 100.0
    y_adj = pd.to_numeric(tmp["adjusted"], errors="coerce") * 100.0
    if x.notna().any() and y_obs.notna().any():
        ax.plot(x, y_obs, color="0.86", lw=0.9)
        ax.plot(x, rolling_mean(y_obs, cfg.moving_average_window), color=COL_OBS, lw=2.4, label="Observed")
    if x.notna().any() and y_adj.notna().any():
        ax.plot(x, rolling_mean(y_adj, cfg.moving_average_window), color=COL_STD, lw=2.2, ls="--", label="Adjusted")
    ax.set_xlabel("Year")
    ax.set_ylabel("10-day summed rain frac. (%·day)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    add_panel_label(ax, "c")

    ax = axs[1, 1]
    dd = cumulative_regional_trend.copy()
    order = ["All"] + get_region_meta(cfg)["order"]
    dd["group"] = pd.Categorical(dd["group"], categories=order, ordered=True)
    dd = dd.sort_values("group").reset_index(drop=True)
    xpos = np.arange(len(dd))
    group_labels = dd["group"].astype(str).tolist()
    colors = [ALL_COLOR if g == "All" else REGION_COLORS.get(g, "#888888") for g in group_labels]
    vals = pd.to_numeric(dd["trend_per_decade"], errors="coerce") * 100.0
    ax.bar(xpos, vals, color=colors, alpha=0.95)
    ax.axhline(0, color="0.6", lw=0.8)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            va = "bottom" if v >= 0 else "top"
            dy = 0.8 if v >= 0 else -0.8
            ax.text(i, v + dy, f"{v:.1f}", ha="center", va=va, fontsize=12)
    ax.set_xticks(xpos)
    ax.set_xticklabels(group_labels, rotation=20, ha="right")
    ax.set_ylabel("Trend (%·day decade$^{-1}$)")
    ax.grid(axis="y", alpha=0.2)
    add_panel_label(ax, "d")

    savefig(fig, cfg.fig_dir / "Supp_Fig_R2_02_continuous_recovery_metrics_paper7.png", dpi=cfg.dpi)

# =============================================================================
# Revisions requested on 2026-04-24
# =============================================================================

def draw_climate_region_boundaries(ax, cfg: Config, linewidth: float = 2.2) -> None:
    """Override: draw each climate-region boundary with its own region color."""
    if not (HAS_CARTOPY and HAS_SHAPELY):
        return
    region_meta = get_region_meta(cfg)
    reg_geoms = load_climate_region_geometries(region_meta)
    for reg in region_meta["order"]:
        geom = reg_geoms.get(reg)
        if geom is None:
            continue
        edge_col = REGION_COLORS.get(reg, "black")
        try:
            ax.add_geometries(
                [geom],
                crs=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor=edge_col,
                linewidth=linewidth,
                zorder=4,
            )
        except Exception:
            continue


def plot_main_figure2(
    annual: pd.DataFrame,
    std_annual: pd.DataFrame,
    trend_df: pd.DataFrame,
    year_grid: np.ndarray,
    duration_grid: np.ndarray,
    heat_grid: np.ndarray,
    cond_dur: np.ndarray,
    cond_heat: np.ndarray,
    cfg: Config,
) -> None:
    """Override: Figure 2b now shows All + all seven climate regions."""
    fig = plt.figure(figsize=(19.2, 10.8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    plot_df = annual.copy().sort_values("year").reset_index(drop=True)
    if "std_no_recovery" not in plot_df.columns:
        plot_df = plot_df.merge(std_annual, on="year", how="left")
    plot_df["obs_smooth"] = rolling_mean(plot_df["no_recovery_obs"], cfg.moving_average_window)
    plot_df["std_smooth"] = rolling_mean(plot_df["std_no_recovery"], cfg.moving_average_window)
    plot_df["rain_smooth"] = rolling_mean(plot_df["mean_rain_fraction_w10"], cfg.moving_average_window)

    ax.plot(plot_df["year"], plot_df["no_recovery_obs"] * 100, color="0.85", lw=0.9)
    ax.plot(plot_df["year"], plot_df["obs_smooth"] * 100, color=COL_OBS, lw=2.5, label="Observed")
    ax.plot(plot_df["year"], plot_df["std_smooth"] * 100, color=COL_STD, lw=2.3, label="Standardized")
    ax.set_xlabel("Year")
    ax.set_ylabel("No-recovery (%)")
    ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(plot_df["year"], plot_df["rain_smooth"] * 100, color="#2a9d8f", lw=2.1, ls="--", label="Mean rain frac.")
    ax2.set_ylabel("Rain frac. (%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    add_panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    dd = trend_df.set_index("group")
    selected = ["All"] + get_region_meta(cfg)["order"]
    xpos = np.arange(len(selected))
    width = 0.30
    obs_vals = []
    std_vals = []
    colors = []
    for g in selected:
        row = dd.loc[g] if g in dd.index else pd.Series(dtype=float)
        obs = float(pd.to_numeric(row.get("observed", np.nan), errors="coerce") * 100.0)
        stdv = float(pd.to_numeric(row.get("conditional", np.nan), errors="coerce") * 100.0)
        obs_vals.append(obs)
        std_vals.append(stdv)
        colors.append("#4d4d4d" if g == "All" else REGION_COLORS.get(g, "#888888"))

    all_vals = np.asarray(obs_vals + std_vals, dtype=float)
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size:
        ymin = float(np.nanmin(finite_vals))
        ymax = float(np.nanmax(finite_vals))
    else:
        ymin, ymax = -1.0, 1.0
    yrng = max(ymax - ymin, 0.8)
    top_pad = 0.28 * yrng
    bottom_pad = 0.12 * yrng

    for i, (g, c, obs, stdv) in enumerate(zip(selected, colors, obs_vals, std_vals)):
        ax.bar(i - width / 2, obs, width=width, color=c, alpha=0.95, label="Observed" if i == 0 else None)
        ax.bar(i + width / 2, stdv, width=width, color=_lighten_color(c, 0.45), alpha=0.95, label="Standardized" if i == 0 else None)
        if np.isfinite(obs) and np.isfinite(stdv):
            ax.plot([i - width / 2, i + width / 2], [obs, stdv], color="0.30", lw=1.0, zorder=5)
            att = obs - stdv
            ytxt = max(obs, stdv) + 0.05 * yrng
            ax.text(i, ytxt, f"Δ={att:.2f}", ha="center", va="bottom", fontsize=10.5)

    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(selected, rotation=28, ha="right")
    for lbl, g, c in zip(ax.get_xticklabels(), selected, colors):
        lbl.set_color(c)
        lbl.set_fontsize(11.5)
        if g != "All":
            lbl.set_fontweight("bold")
    ax.set_ylabel("Trend (pp decade$^{-1}$)")
    ax.set_xlim(-0.75, len(selected) - 0.25)
    ax.set_ylim(min(0.0, ymin - bottom_pad), ymax + top_pad)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    add_panel_label(ax, "b")

    zc = np.asarray(cond_dur, dtype=float).copy()
    ax = fig.add_subplot(gs[1, 0])
    im = ax.pcolormesh(year_grid, duration_grid, zc * 100, shading="auto", cmap="magma_r")
    if np.isfinite(zc).any():
        cs = ax.contour(year_grid, duration_grid, zc * 100, levels=6, colors="white", linewidths=0.65, alpha=0.85)
        if len(cs.levels) > 0:
            ax.clabel(cs, fmt="%.0f", fontsize=17)
    ax.set_xlabel("Year")
    ax.set_ylabel("Duration (days)")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.18, fraction=0.08)
    cbar.set_label("No-recovery (%)")
    add_panel_label(ax, "c")

    zh = np.asarray(cond_heat, dtype=float).copy()
    ax = fig.add_subplot(gs[1, 1])
    im = ax.pcolormesh(year_grid, heat_grid, zh * 100, shading="auto", cmap="magma_r")
    if np.isfinite(zh).any():
        cs = ax.contour(year_grid, heat_grid, zh * 100, levels=6, colors="white", linewidths=0.65, alpha=0.85)
        if len(cs.levels) > 0:
            ax.clabel(cs, fmt="%.0f", fontsize=17)
    ax.set_xlabel("Year")
    ax.set_ylabel("Heat excess (°C)")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.18, fraction=0.08)
    cbar.set_label("No-recovery (%)")
    add_panel_label(ax, "d")

    savefig(fig, cfg.fig_dir / "Figure2_result2_structural_rebuild_paper7.png", dpi=cfg.dpi)



def _build_regional_annual_for_rawstd(df: pd.DataFrame, std_by_region: Dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    regions = get_region_meta(cfg)["order"]
    region_rows = []
    for region in regions:
        sub = df.loc[df["region4"] == region].copy()
        ann = annual_observed_metrics(sub, cfg)
        ann["region4"] = region
        std = std_by_region.get(region)
        if std is not None:
            ann = ann.merge(std, on="year", how="left")
        region_rows.append(ann)
    return pd.concat(region_rows, ignore_index=True) if region_rows else pd.DataFrame()



def _save_region_overview_map_single(cfg: Config, out_dir: Path) -> None:
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(8.5, 5.8), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.8), facecolor="white")
    _setup_region_overview_map(ax)
    savefig(fig, out_dir / "Supp_Fig_R2_03_region_overview_map_paper7.png", dpi=cfg.dpi)



def _slug_region_name(region: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(region).strip().lower()).strip("_")



def _save_single_region_rawstd_panel(region_ann: pd.DataFrame, region: str, cfg: Config, out_dir: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.9), facecolor="white")
    _plot_region_rawstd_panel(ax, region_ann, region, cfg)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, loc="upper left", ncol=2)
    fname = f"Supp_Fig_R2_03_raw_vs_standardized_{_slug_region_name(region)}_paper7.png"
    savefig(fig, out_dir / fname, dpi=cfg.dpi)



def plot_supp_raw_vs_std_by_region(df: pd.DataFrame, std_by_region: Dict[str, pd.DataFrame], cfg: Config) -> None:
    """Override: keep the composite figure and standardize geometry handling."""
    regions = get_region_meta(cfg)["order"]
    regional_ann = _build_regional_annual_for_rawstd(df, std_by_region, cfg)
    if regional_ann.empty:
        warnings.warn("Regional annual table is empty; skipping raw-vs-standardized region figure.")
        return

    fig = plt.figure(figsize=(17.2, 10.4), facecolor="white")
    map_ax = fig.add_axes([0.33, 0.28, 0.34, 0.40], projection=ccrs.PlateCarree() if HAS_CARTOPY else None)
    _setup_region_overview_map(map_ax)

    positions = {
        "Northwest": [0.05, 0.61, 0.24, 0.18],
        "Northern Great Plains": [0.36, 0.79, 0.22, 0.16],
        "Midwest": [0.64, 0.61, 0.24, 0.18],
        "Northeast": [0.76, 0.36, 0.20, 0.18],
        "Southeast": [0.66, 0.08, 0.24, 0.18],
        "Southern Great Plains": [0.37, 0.02, 0.26, 0.18],
        "Southwest": [0.07, 0.09, 0.24, 0.18],
    }
    target_fracs = {
        "Northwest": (0.95, 0.55),
        "Northern Great Plains": (0.50, 0.02),
        "Midwest": (0.05, 0.55),
        "Northeast": (0.00, 0.40),
        "Southeast": (0.30, 1.00),
        "Southern Great Plains": (0.50, 1.00),
        "Southwest": (0.95, 0.40),
    }
    panel_labels = ["a", "b", "c", "d", "e", "f", "g"]

    first_handles = None
    first_labels = None
    for i, region in enumerate(regions):
        ax = fig.add_axes(positions[region])
        _plot_region_rawstd_panel(ax, regional_ann, region, cfg)
        add_panel_label(ax, panel_labels[i], x=-0.16, y=1.12)
        if i == 0:
            first_handles, first_labels = ax.get_legend_handles_labels()
        x0, y0 = _REGION_CONNECT_POS_R2.get(region, _REGION_LABEL_POS_R2.get(region, (-100, 35)))
        fx, fy = target_fracs[region]
        con = ConnectionPatch(
            xyA=(x0, y0), coordsA=ccrs.PlateCarree()._as_mpl_transform(map_ax) if HAS_CARTOPY else map_ax.transData,
            xyB=(fx, fy), coordsB=ax.transAxes,
            color=REGION_COLORS[region], lw=0.95, alpha=0.95,
        )
        fig.add_artist(con)
    if first_handles and first_labels:
        fig.legend(first_handles, first_labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.50, -0.01))

    savefig(fig, cfg.fig_dir / "Supp_Fig_R2_03_raw_vs_standardized_by_region_paper7.png", dpi=cfg.dpi)



def plot_supp_raw_vs_std_by_region_individual(df: pd.DataFrame, std_by_region: Dict[str, pd.DataFrame], cfg: Config) -> None:
    """New: export every subplot from Supp_Fig_R2_03 as standalone files."""
    regional_ann = _build_regional_annual_for_rawstd(df, std_by_region, cfg)
    if regional_ann.empty:
        warnings.warn("Regional annual table is empty; skipping standalone regional panels.")
        return
    out_dir = cfg.fig_dir / "Supp_Fig_R2_03_raw_vs_standardized_by_region_panels"
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_region_overview_map_single(cfg, out_dir)
    for region in get_region_meta(cfg)["order"]:
        _save_single_region_rawstd_panel(regional_ann, region, cfg, out_dir)


# =============================================================================
# Main workflow
# =============================================================================


def main(cfg: Config = CFG) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    cfg.table_dir.mkdir(parents=True, exist_ok=True)

    for stale in [
        "Supp_Fig_R2_02b_region_share_shift_paper7.png",
        "Supp_Fig_R2_04_decomposition_components_paper7.png",
        "Supp_Fig_R2_05_raw_vs_conditional_surfaces_paper7.png",
    ]:
        fp = cfg.fig_dir / stale
        if fp.exists():
            try:
                fp.unlink()
            except Exception:
                pass

    event_df, precip_note = build_or_load_event_table(cfg)
    print(f"[INFO] Total summarized events : {len(event_df):,}")
    print(f"[INFO] Precip note             : {precip_note}")

    analysis_df_raw, sample_note = choose_analysis_sample(event_df)
    analysis_df = prepare_analysis_table(analysis_df_raw, cfg)
    print(f"[INFO] Sample rule            : {sample_note}")

    model_full = fit_glm_standardization_full(analysis_df)
    model_noyear = fit_glm_standardization(analysis_df.drop(columns=["region4"]).assign(region4="All"), with_year=False)
    regional_models = build_regional_models(analysis_df)

    years = sorted(analysis_df["year"].unique())
    ref_df = analysis_df.copy()
    annual = annual_observed_metrics(analysis_df, cfg)
    std_annual = standardized_annual_series(model_full, ref_df, years)
    comp_annual = composition_only_annual_series(model_full, analysis_df, years, int(round(np.mean(years))))
    annual = annual.merge(std_annual, on="year", how="left").merge(comp_annual, on="year", how="left")
    annual.to_csv(cfg.table_dir / "result2_annual_metrics_paper7.csv", index=False, encoding="utf-8-sig")

    trend_df = build_trend_decomposition_table(analysis_df, model_full, regional_models, cfg)
    trend_df.to_csv(cfg.table_dir / "result2_trend_decomposition_paper7.csv", index=False, encoding="utf-8-sig")

    rolling_df = build_rolling_period_sensitivity(analysis_df, model_full, cfg)
    rolling_df.to_csv(cfg.table_dir / "result2_rolling_window_sensitivity_paper7.csv", index=False, encoding="utf-8-sig")

    std_by_region: Dict[str, pd.DataFrame] = {}
    for region, reg_model in regional_models.items():
        sub = analysis_df.loc[analysis_df["region4"] == region].copy()
        yrs = sorted(sub["year"].unique())
        std_by_region[region] = standardized_annual_series(reg_model, sub, yrs)

    mean_series, _ = build_continuous_adjusted_annual_series(analysis_df, "mean_rain_fraction_w10", cfg)
    max_series, _ = build_continuous_adjusted_annual_series(analysis_df, "max_rain_fraction_w10", cfg)
    cumulative_series, cumulative_regional_series = build_continuous_adjusted_annual_series(analysis_df, "cumulative_rain_fraction_w10", cfg)

    mean_series.to_csv(cfg.table_dir / "result2_annual_mean_rain_fraction_series.csv", index=False, encoding="utf-8-sig")
    max_series.to_csv(cfg.table_dir / "result2_annual_max_rain_fraction_series.csv", index=False, encoding="utf-8-sig")
    cumulative_series.to_csv(cfg.table_dir / "result2_annual_cumulative_rain_fraction_series.csv", index=False, encoding="utf-8-sig")

    cumulative_regional_rows: List[pd.DataFrame] = []
    for group, gdf in cumulative_regional_series.items():
        tmp_reg = gdf.copy()
        tmp_reg["group"] = group
        cumulative_regional_rows.append(tmp_reg)
    cumulative_regional_series_df = pd.concat(cumulative_regional_rows, ignore_index=True)
    cumulative_regional_series_df.to_csv(cfg.table_dir / "cumulative_regional_series.csv", index=False, encoding="utf-8-sig")

    cumulative_regional_trend = build_regional_continuous_trend_table(cumulative_regional_series, value_col="adjusted")
    cumulative_regional_trend.to_csv(cfg.table_dir / "cumulative_regional_trend_table.csv", index=False, encoding="utf-8-sig")

    year_grid = np.arange(int(analysis_df["year"].min()), int(analysis_df["year"].max()) + 1)
    duration_grid = np.linspace(max(1.0, analysis_df["duration"].quantile(0.02)), analysis_df["duration"].quantile(0.98), cfg.duration_grid_n)
    heat_grid = np.linspace(max(0.0, analysis_df["mean_heat_excess"].quantile(0.02)), analysis_df["mean_heat_excess"].quantile(0.98), cfg.heat_grid_n)

    cond_dur = conditional_surface_year_duration(model_full, ref_df, year_grid, duration_grid, cfg)
    cond_heat = conditional_surface_year_heat(model_full, ref_df, year_grid, heat_grid, cfg)

    native_grid_df = build_or_load_native_grid_event_table(analysis_df, cfg)
    spatial_df = build_spatial_trend_table_native(native_grid_df, model_noyear, cfg)
    spatial_df.to_csv(cfg.table_dir / "result2_spatial_trend_table_native_grid_paper7.csv", index=False, encoding="utf-8-sig")

    boot_df = build_cluster_bootstrap_summary(analysis_df, cfg)
    boot_df.to_csv(cfg.table_dir / "result2_cluster_bootstrap_summary_paper7.csv", index=False, encoding="utf-8-sig")

    plot_main_figure2(annual, std_annual, trend_df, year_grid, duration_grid, heat_grid, cond_dur, cond_heat, cfg)
    plot_supp_sample_support(annual, cfg, sample_note)
    plot_supp_covariate_shift(annual, analysis_df, cfg)
    plot_supp_raw_vs_std_by_region(analysis_df, std_by_region, cfg)
    plot_supp_raw_vs_std_by_region_individual(analysis_df, std_by_region, cfg)
    plot_spatial_maps(spatial_df, native_grid_df, cfg)
    plot_supp_rolling_sensitivity(rolling_df, cfg)
    plot_supp_continuous_and_bootstrap(annual, boot_df, cfg)
    plot_supp_cumulative_recovery_metrics(mean_series, max_series, cumulative_series, cumulative_regional_trend, cfg)

    print("[INFO] Revised Result 2 rebuild finished.")
    print(f"[INFO] Output directory: {cfg.out_dir}")


if __name__ == "__main__":
    main(CFG)