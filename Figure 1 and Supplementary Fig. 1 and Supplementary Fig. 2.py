# -*- coding: utf-8 -*-
"""
Result 1 final rebuild for an event-level drought-heatwave paper
===============================================================

Revised to address figure-structure comments:
1) Main Figure 1 panel b/c merged into one dual-axis continuous panel.
2) Spatial maps use clipped 7-region boundaries inside CONUS only.
3) Lat/lon labels are kept on left/bottom only; no map gridlines.
4) Supplement S1 redesigned as a central US map surrounded by 7 regional lines.
5) S4 removed; its benchmark is merged into S2.
6) Definition-space heatmaps are made more interpretable and less sparse.
7) Panel letters are placed outside the axes.
8) Event climate regions are assigned by heatwave-core footprint majority rule,
   with centroid-based assignment used only as a fallback/tie-breaker.
9) Main spatial recovery map includes stippling for grid cells whose recovery
   probability differs significantly from the CONUS cell-exposure mean.
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
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import ConnectionPatch

import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None

try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

try:
    from shapely.geometry import Point, Polygon, MultiPolygon, box
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Config:
    root_dir: Path = Path(r"E:\temp_events_ERA5_S1S6_Nature所有数据版本\zu")
    out_dir: Optional[Path] = None

    primary_rain_threshold_mm: float = 1.0
    primary_footprint_threshold: float = 0.25
    primary_window_days: int = 10

    rain_thresholds_mm: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
    footprint_thresholds: Tuple[float, ...] = (0.10, 0.15, 0.25, 0.33, 0.50)
    window_days_grid: Tuple[int, ...] = (3, 7, 10)

    min_detectable_coverage: float = 0.80
    usable_min_coverage: float = 0.80
    strict_min_coverage: float = 0.95

    use_cache: bool = True
    progress_every: int = 250
    bootstrap_n: int = 250
    bootstrap_seed: int = 42

    rolling_window: int = 7
    dpi: int = 300
    figure_bg: str = "white"
    panel_label_fs: int = 26
    map_min_events: int = 8

    # Stippling on Figure 1a / individual map panel. The test is a two-sided
    # binomial comparison of each grid-cell recovery probability against the
    # CONUS cell-exposure mean. FDR correction and a minimum effect-size filter
    # avoid treating visually trivial but statistically detectable differences
    # as manuscript-level evidence.
    map_sig_alpha: float = 0.05
    map_sig_min_abs_diff: float = 0.05   # 5 percentage points
    map_sig_point_size: float = 5.5
    map_sig_alpha_points: float = 0.72

    spatial_lon_step: Optional[float] = None
    spatial_lat_step: Optional[float] = None

    eventday_formula: str = (
        "event_occurs ~ C(lag_day) + bs(year_c, df=5, include_intercept=False) + "
        "log_duration_c + heat_excess_c + sin_doy + cos_doy + C(climate_region)"
    )

    region_order: Tuple[str, ...] = (
        "Northwest",
        "Northern Great Plains",
        "Midwest",
        "Northeast",
        "Southwest",
        "Southern Great Plains",
        "Southeast",
    )

    def __post_init__(self) -> None:
        if self.out_dir is None:
            self.out_dir = self.root_dir / "_r23"
        self.cache_dir = self.out_dir / "cache"
        self.fig_dir = self.out_dir / "figures"
        self.table_dir = self.out_dir / "tables"
        for d in (self.out_dir, self.cache_dir, self.fig_dir, self.table_dir):
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()

plt.rcParams.update({
    "font.size": 21,
    "axes.titlesize": 21,
    "axes.labelsize": 21,
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
    "legend.fontsize": 19,
    "figure.titlesize": 21,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "savefig.bbox": "tight",
    "figure.facecolor": CFG.figure_bg,
    "axes.facecolor": "white",
})

COL_ALL = "#2b2b2b"
COL_TARGET = "#b2182b"
COL_BLUE = "#2166ac"
COL_GREEN = "#1b9e77"
COL_ORANGE = "#cc6f12"
COL_PURPLE = "#7b3294"
COL_GREY = "#7a7a7a"
COL_TEAL = "#2ca25f"
COL_BROWN = "#8c510a"

REGION_COLORS = {
    "Northwest": "#c0392b",
    "Northern Great Plains": "#7f8c8d",
    "Midwest": "#8e44ad",
    "Northeast": "#1f78b4",
    "Southwest": "#d95f02",
    "Southern Great Plains": "#1b9e77",
    "Southeast": "#e6ab02",
}


# =============================================================================
# Helpers
# =============================================================================


def rolling_mean(y: pd.Series, window: int) -> pd.Series:
    return y.rolling(window=window, center=True, min_periods=max(3, window // 2)).mean()



def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=CFG.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)



def add_panel_label(ax, label: str, x: float = -0.16, y: float = 1.10) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=CFG.panel_label_fs,
        fontweight="bold",
        clip_on=False,
    )



def _safe_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")



def _bool_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)



def weighted_linear_slope(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(x, dtype=float)
    else:
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
    return float(np.sum(w * x0 * y0) / denom)



def linear_trend_pp_decade(annual_df: pd.DataFrame, value_col: str, weight_col: Optional[str] = None) -> float:
    if annual_df.empty or value_col not in annual_df.columns:
        return np.nan
    w = annual_df[weight_col].to_numpy(dtype=float) if weight_col and weight_col in annual_df.columns else None
    slope = weighted_linear_slope(annual_df["year"].to_numpy(dtype=float), annual_df[value_col].to_numpy(dtype=float), w)
    return slope * 10.0 * 100.0


# =============================================================================
# CONUS and region geometry
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

_REGION_LABEL_POS = {
    "Northwest": (-121.2, 46.2),
    "Northern Great Plains": (-104.2, 46.1),
    "Midwest": (-90.0, 43.0),
    "Northeast": (-74.8, 43.8),
    "Southwest": (-113.0, 35.0),
    "Southern Great Plains": (-98.0, 32.5),
    "Southeast": (-82.8, 32.5),
}

_REGION_CONNECT_POS = {
    "Northwest": (-118.5, 45.8),
    "Northern Great Plains": (-101.5, 46.2),
    "Midwest": (-90.0, 44.0),
    "Northeast": (-74.5, 42.7),
    "Southwest": (-112.0, 35.5),
    "Southern Great Plains": (-97.0, 32.4),
    "Southeast": (-83.0, 32.4),
}

_CONUS_GEOM = None
_STATE_RECORDS: List[Dict] = []
_REGION_GEOMS: Dict[str, object] = {}
_POINT_REGION_CACHE: Dict[Tuple[float, float], str] = {}


def get_conus_geometry():
    global _CONUS_GEOM
    if _CONUS_GEOM is not None:
        return _CONUS_GEOM
    if not (HAS_CARTOPY and HAS_SHAPELY):
        _CONUS_GEOM = None
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
            _CONUS_GEOM = None
            return None
        usa = unary_union(geoms)
        _CONUS_GEOM = usa.intersection(box(-125.0, 24.5, -66.5, 50.0))
        return _CONUS_GEOM
    except Exception:
        _CONUS_GEOM = None
        return None


def load_state_records() -> List[Dict]:
    global _STATE_RECORDS
    if _STATE_RECORDS:
        return _STATE_RECORDS
    if not (HAS_CARTOPY and HAS_SHAPELY):
        return []
    try:
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
        _STATE_RECORDS = out
        return out
    except Exception:
        _STATE_RECORDS = []
        return []


def get_region_geometries() -> Dict[str, object]:
    global _REGION_GEOMS
    if _REGION_GEOMS:
        return _REGION_GEOMS
    if HAS_SHAPELY:
        by_region = {r: [] for r in CFG.region_order}
        for rec in load_state_records():
            reg = PAPER7_STATE_TO_REGION.get(rec["abbr"])
            if reg in by_region:
                by_region[reg].append(rec["geom"])
        for reg, geoms in by_region.items():
            if geoms:
                try:
                    _REGION_GEOMS[reg] = unary_union(geoms)
                except Exception:
                    pass
    return _REGION_GEOMS


def polygon_to_patches(geom) -> List[MplPolygon]:
    patches: List[MplPolygon] = []
    if geom is None:
        return patches
    try:
        if geom.is_empty:
            return patches
    except Exception:
        pass
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


def plot_geom_boundary(ax, geom, color="black", lw=1.4, zorder=5):
    if geom is None:
        return
    try:
        if geom.is_empty:
            return
    except Exception:
        return
    if HAS_SHAPELY and isinstance(geom, Polygon):
        coords = np.asarray(geom.exterior.coords)
        ax.plot(coords[:, 0], coords[:, 1], color=color, lw=lw,
                transform=ccrs.PlateCarree() if HAS_CARTOPY else ax.transData, zorder=zorder)
    elif HAS_SHAPELY and isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            plot_geom_boundary(ax, part, color=color, lw=lw, zorder=zorder)


def _fallback_region(lon: float, lat: float) -> str:
    if lon <= -116:
        return "Northwest" if lat >= 42 else "Southwest"
    if lon <= -100:
        return "Northern Great Plains" if lat >= 39 else "Southern Great Plains"
    if lat >= 41:
        return "Northeast"
    if lat >= 37:
        return "Midwest"
    return "Southeast"


def assign_climate_region(lon: float, lat: float) -> str:
    key = (round(float(lon), 4), round(float(lat), 4))
    if key in _POINT_REGION_CACHE:
        return _POINT_REGION_CACHE[key]
    if HAS_SHAPELY:
        try:
            p = Point(float(lon), float(lat))
            for rec in load_state_records():
                minx, miny, maxx, maxy = rec["bounds"]
                if not (minx <= lon <= maxx and miny <= lat <= maxy):
                    continue
                if rec["geom"].covers(p) or rec["geom"].buffer(1e-9).covers(p):
                    reg = PAPER7_STATE_TO_REGION.get(rec["abbr"], _fallback_region(lon, lat))
                    _POINT_REGION_CACHE[key] = reg
                    return reg
        except Exception:
            pass
    reg = _fallback_region(lon, lat)
    _POINT_REGION_CACHE[key] = reg
    return reg


def assign_event_climate_region_by_footprint(
    heat_coords: pd.DataFrame,
    centroid_lon: float,
    centroid_lat: float,
) -> str:
    """Assign an event to a 7-region climate division by heatwave-core footprint majority.

    This is the event-level region rule used for consistency with Result 2:
    each unique heatwave-core footprint cell is mapped to a state polygon and then
    to the corresponding 7-region climate division. The event is assigned to the
    modal region across its heatwave-core cells. The centroid is used only when no
    footprint cell can be mapped or when the footprint vote is tied.
    """
    centroid_region = assign_climate_region(float(centroid_lon), float(centroid_lat))

    if heat_coords is None or heat_coords.empty:
        return centroid_region

    required = {"longitude", "latitude"}
    if not required.issubset(set(heat_coords.columns)):
        return centroid_region

    regs: List[str] = []
    for row in heat_coords[["longitude", "latitude"]].dropna().itertuples(index=False):
        try:
            reg = assign_climate_region(float(row.longitude), float(row.latitude))
        except Exception:
            reg = None
        if reg in CFG.region_order:
            regs.append(reg)

    if not regs:
        return centroid_region

    counts = Counter(regs).most_common()
    if len(counts) == 1:
        return counts[0][0]

    # If there is a tie for the largest footprint share, use the centroid region
    # as the deterministic tie-breaker, provided it belongs to the tied set.
    top_n = counts[0][1]
    tied = [reg for reg, n in counts if n == top_n]
    if len(tied) > 1:
        if centroid_region in tied:
            return centroid_region
        return tied[0]

    return counts[0][0]


def draw_region_boundaries(ax, lw: float = 1.8) -> None:
    geoms = get_region_geometries()
    for region in CFG.region_order:
        plot_geom_boundary(ax, geoms.get(region), color="black", lw=lw, zorder=7)


def draw_filled_region_map(ax, show_labels: bool = False, outline: bool = True, alpha: float = 0.92) -> None:
    geoms = get_region_geometries()
    patches = []
    facecolors = []
    for region in CFG.region_order:
        region_patches = polygon_to_patches(geoms.get(region))
        patches.extend(region_patches)
        facecolors.extend([REGION_COLORS[region]] * len(region_patches))
    if patches:
        pc = PatchCollection(
            patches,
            facecolor=facecolors,
            edgecolor="none",
            alpha=alpha,
            zorder=2,
            transform=ccrs.PlateCarree() if HAS_CARTOPY else ax.transData,
        )
        ax.add_collection(pc)
    if outline:
        draw_region_boundaries(ax, lw=1.6)
    if show_labels:
        for region, (x, y) in _REGION_LABEL_POS.items():
            ax.text(x, y, region, fontsize=10.5, ha="center", va="center",
                    transform=ccrs.PlateCarree() if HAS_CARTOPY else ax.transData, zorder=9)

# =============================================================================
# File discovery and cleaning
# =============================================================================


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
_FILE_RE = re.compile(r"event_(\d{4})_(\d+)_window\.csv")



def discover_event_files(root_dir: Path) -> List[Path]:
    fps = sorted(root_dir.glob("*/*event_*_window.csv"))
    if not fps:
        fps = sorted(root_dir.rglob("event_*_window.csv"))
    return fps



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



def _event_uid_from_path(path: Path, df: pd.DataFrame) -> str:
    m = _FILE_RE.search(path.name)
    if m:
        return f"{int(m.group(1))}_{int(m.group(2)):05d}"
    year = int(pd.to_numeric(df.get("year", pd.Series([np.nan])).dropna().iloc[0])) if "year" in df else -1
    eid = int(pd.to_numeric(df.get("event_id", pd.Series([np.nan])).dropna().iloc[0])) if "event_id" in df else -1
    return f"{year}_{eid:05d}"



def infer_precip_scale(event_files: Sequence[Path], sample_n: int = 40) -> Tuple[float, str]:
    if not event_files:
        return 1.0, "no_files_found_assume_mm"
    qs: List[float] = []
    idx = np.linspace(0, len(event_files) - 1, min(sample_n, len(event_files))).astype(int)
    for i in idx:
        try:
            df = pd.read_csv(event_files[i], low_memory=False)
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



def _mean_heat_excess(heat_df: pd.DataFrame) -> float:
    if heat_df.empty or "temp_air" not in heat_df.columns or "T90" not in heat_df.columns:
        return np.nan
    exc = pd.to_numeric(heat_df["temp_air"], errors="coerce") - pd.to_numeric(heat_df["T90"], errors="coerce")
    exc = exc.clip(lower=0)
    return float(exc.mean()) if exc.notna().any() else np.nan


# =============================================================================
# Event summary construction
# =============================================================================


RTHR_LABELS = {r: f"rainfrac_r{str(r).replace('.', 'p')}" for r in CFG.rain_thresholds_mm}



def summarize_one_event(path: Path, precip_scale: float, cfg: Config) -> Tuple[Optional[Dict], List[Dict]]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        warnings.warn(f"Failed to read {path.name}: {e}")
        return None, []

    df = _clean_columns(df)
    if "date" not in df.columns or "precipitation" not in df.columns:
        warnings.warn(f"Required columns missing in {path.name}; skipped.")
        return None, []

    for col in _DATE_COLS:
        if col in df.columns:
            df[col] = _safe_dt(df[col])
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "coord_key" not in df.columns:
        if {"longitude", "latitude"}.issubset(df.columns):
            df["coord_key"] = df["longitude"].round(4).astype(str) + "_" + df["latitude"].round(4).astype(str)
        else:
            warnings.warn(f"coord_key and lon/lat missing in {path.name}; skipped.")
            return None, []

    df["precip_mm"] = pd.to_numeric(df["precipitation"], errors="coerce") * precip_scale
    lag = pd.to_numeric(df.get("lag_day_event", pd.Series(np.nan, index=df.index)), errors="coerce")
    is_heat = _bool_num(df.get("is_heat_period_event", pd.Series(np.nan, index=df.index)))

    heat = df.loc[is_heat == 1].copy()
    if heat.empty:
        heat = df.loc[lag <= 0].copy()
    if heat.empty:
        warnings.warn(f"No heat-period rows found in {path.name}; skipped.")
        return None, []

    heat_coords = heat[["coord_key", "longitude", "latitude"]].dropna().drop_duplicates("coord_key")
    footprint = set(heat_coords["coord_key"].astype(str))
    footprint_ncells = int(len(footprint))
    if footprint_ncells == 0:
        warnings.warn(f"Zero heat footprint in {path.name}; skipped.")
        return None, []

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
    climate_region = assign_event_climate_region_by_footprint(heat_coords, centroid_lon, centroid_lat)
    mean_heat_excess = _mean_heat_excess(heat)
    end_doy = int(event_end.dayofyear) if pd.notna(event_end) else int(pd.to_numeric(df.get("doy", pd.Series([np.nan]))).dropna().median())

    post = df.loc[(lag >= 1) & (lag <= cfg.primary_window_days)].copy()
    daily_rows: List[Dict] = []
    for lag_day in range(1, cfg.primary_window_days + 1):
        g = post.loc[pd.to_numeric(post.get("lag_day_event"), errors="coerce") == lag_day].copy()
        g2 = g.dropna(subset=["coord_key"]).drop_duplicates("coord_key")
        g2 = g2.loc[g2["coord_key"].astype(str).isin(footprint)].copy()
        n_present = int(g2["coord_key"].nunique())
        coverage = n_present / footprint_ncells if footprint_ncells > 0 else np.nan
        row = {
            "event_uid": uid,
            "year": year_val,
            "lag_day": lag_day,
            "coverage": coverage,
            "n_present": n_present,
        }
        for rthr in cfg.rain_thresholds_mm:
            rainy_cells = int(g2.loc[g2["precip_mm"] >= rthr, "coord_key"].nunique())
            row[RTHR_LABELS[rthr]] = rainy_cells / footprint_ncells if footprint_ncells > 0 else np.nan
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)

    for rthr in cfg.rain_thresholds_mm:
        col = RTHR_LABELS[rthr]
        daily[f"recovered_{col}"] = (
            (daily["coverage"] >= cfg.min_detectable_coverage) &
            (daily[col] >= cfg.primary_footprint_threshold)
        ).astype(int)

    primary_col = RTHR_LABELS[cfg.primary_rain_threshold_mm]
    any_recovery = bool((daily[f"recovered_{primary_col}"] == 1).any())
    recovery_day = int(daily.loc[daily[f"recovered_{primary_col}"] == 1, "lag_day"].min()) if any_recovery else np.nan

    has_all_lags = bool(set(range(1, cfg.primary_window_days + 1)).issubset(set(daily["lag_day"].tolist())))
    min_cov = float(daily["coverage"].min()) if not daily.empty else np.nan
    noverlap = bool((_bool_num(post.get("overlaps_next_local_heat", pd.Series(0, index=post.index))) == 1).any())
    ncensor = bool((_bool_num(post.get("is_post_event_0_10_censored", pd.Series(0, index=post.index))) == 1).any())

    usable_flag = has_all_lags and (min_cov >= cfg.usable_min_coverage if pd.notna(min_cov) else False)
    strict_flag = has_all_lags and (min_cov >= cfg.strict_min_coverage if pd.notna(min_cov) else False)
    no_overlap_flag = usable_flag and (not noverlap)
    no_censor_flag = usable_flag and (not ncensor)

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
        "climate_region": climate_region,
        "footprint_ncells": footprint_ncells,
        "mean_heat_excess": mean_heat_excess,
        "recovered_by_day10": int(any_recovery),
        "no_recovery": int(not any_recovery),
        "first_recovery_lag": recovery_day,
        "has_all_lags": int(has_all_lags),
        "min_coverage": min_cov,
        "usable_flag": int(usable_flag),
        "strict_flag": int(strict_flag),
        "no_overlap_flag": int(no_overlap_flag),
        "no_censor_flag": int(no_censor_flag),
        "any_overlap": int(noverlap),
        "any_censor": int(ncensor),
        "max_rain_fraction_day10": float(daily[primary_col].max()),
        "mean_rain_fraction_day10": float(daily[primary_col].mean()),
        "rain_fraction_day10": float(daily.loc[daily["lag_day"] == 10, primary_col].iloc[0]),
    }
    return out, daily_rows



def build_or_load_event_tables(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    cache_summary = cfg.cache_dir / "result1_event_summary_footprint_majority.csv"
    cache_daily = cfg.cache_dir / "result1_event_lag_table_footprint_majority.csv"
    if cfg.use_cache and cache_summary.exists() and cache_daily.exists():
        summary = pd.read_csv(cache_summary, parse_dates=["event_start", "event_end"])
        daily = pd.read_csv(cache_daily)
        return summary, daily, "loaded_cached_event_summary"

    event_files = discover_event_files(cfg.root_dir)
    if not event_files:
        raise FileNotFoundError(f"No event CSV files found under: {cfg.root_dir}")

    precip_scale, precip_note = infer_precip_scale(event_files)
    print(f"[INFO] Event files discovered : {len(event_files):,}")
    print(f"[INFO] Precipitation note    : {precip_note}")

    summary_rows: List[Dict] = []
    daily_rows: List[Dict] = []
    for i, fp in enumerate(event_files, start=1):
        out, lag_rows = summarize_one_event(fp, precip_scale, cfg)
        if out is not None:
            summary_rows.append(out)
            daily_rows.extend(lag_rows)
        if (i % cfg.progress_every == 0) or (i == len(event_files)):
            print(f"[INFO] Summarized {i:,}/{len(event_files):,} files | retained {len(summary_rows):,}")

    summary = pd.DataFrame(summary_rows)
    daily = pd.DataFrame(daily_rows)
    if summary.empty or daily.empty:
        raise RuntimeError("Event summary tables are empty.")

    summary = summary.sort_values(["year", "event_uid"]).reset_index(drop=True)
    daily = daily.sort_values(["year", "event_uid", "lag_day"]).reset_index(drop=True)
    summary.to_csv(cache_summary, index=False, encoding="utf-8-sig")
    daily.to_csv(cache_daily, index=False, encoding="utf-8-sig")
    return summary, daily, precip_note


# =============================================================================
# Sample selection and derived tables
# =============================================================================



def choose_main_sample(summary: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    strict = summary.loc[summary["strict_flag"] == 1].copy()
    usable = summary.loc[summary["usable_flag"] == 1].copy()
    if len(strict) >= 500:
        return strict, f"strict | n={len(strict):,}"
    if len(usable) >= 500:
        warnings.warn("Strict sample is empty/small; falling back to usable sample.")
        return usable, f"usable | n={len(usable):,}"
    warnings.warn("Usable sample is also small; falling back to all events.")
    return summary.copy(), f"all_events | n={len(summary):,}"



def annual_primary_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    g = summary.groupby("year", observed=True)
    ann = g.agg(
        n_events=("event_uid", "count"),
        recovery_prob=("recovered_by_day10", "mean"),
        no_recovery_share=("no_recovery", "mean"),
        mean_rain_fraction_day10=("mean_rain_fraction_day10", "mean"),
        max_rain_fraction_day10=("max_rain_fraction_day10", "mean"),
    ).reset_index()
    lag = summary.loc[summary["recovered_by_day10"] == 1].groupby("year", observed=True)["first_recovery_lag"].median().rename("median_lag")
    ann = ann.merge(lag.reset_index(), on="year", how="left")
    return ann.sort_values("year").reset_index(drop=True)



def sample_support_annual(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in summary.groupby("year", observed=True):
        usable = int((g["usable_flag"] == 1).sum())
        no_overlap = int((g["no_overlap_flag"] == 1).sum())
        no_censor = int((g["no_censor_flag"] == 1).sum())
        rows.append({
            "year": int(year),
            "all_events": int(len(g)),
            "usable": usable,
            "no_overlap": no_overlap,
            "no_censor": no_censor,
            "retained_no_overlap_vs_usable": no_overlap / usable if usable > 0 else np.nan,
            "retained_no_censor_vs_usable": no_censor / usable if usable > 0 else np.nan,
            "mean_min_coverage": float(pd.to_numeric(g["min_coverage"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)



def bootstrap_rolling_metric(summary: pd.DataFrame, metric: str, years: Sequence[int], window: int, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    year_groups = {int(y): g.copy() for y, g in summary.groupby("year", observed=True)}
    out_rows = []
    for b in range(n_boot):
        vals = []
        for y in years:
            g = year_groups.get(int(y))
            if g is None or g.empty:
                vals.append(np.nan)
                continue
            g_boot = g.sample(n=len(g), replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
            if metric == "recovery_prob":
                vals.append(float(g_boot["recovered_by_day10"].mean()))
            elif metric == "median_lag":
                rec = g_boot.loc[g_boot["recovered_by_day10"] == 1, "first_recovery_lag"]
                vals.append(float(rec.median()) if len(rec) else np.nan)
            else:
                raise ValueError(metric)
        ser = pd.Series(vals, index=years)
        smooth = rolling_mean(ser, window)
        out_rows.append(pd.DataFrame({"year": years, "boot": b, "value": smooth.to_numpy(dtype=float)}))
    out = pd.concat(out_rows, ignore_index=True)
    q = out.groupby("year", observed=True)["value"].quantile([0.025, 0.5, 0.975]).unstack().reset_index()
    q.columns = ["year", "q025", "q500", "q975"]
    return q



def build_eventday_person_period(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    ybar = summary["year"].mean()
    ldbar = np.log1p(summary["duration"]).mean()
    hebar = summary["mean_heat_excess"].mean()
    for rec in summary.itertuples(index=False):
        frl = rec.first_recovery_lag if pd.notna(rec.first_recovery_lag) else np.nan
        stop = int(frl) if pd.notna(frl) else CFG.primary_window_days
        for lag_day in range(1, stop + 1):
            rows.append({
                "event_uid": rec.event_uid,
                "year": int(rec.year),
                "year_c": float(rec.year - ybar),
                "lag_day": lag_day,
                "event_occurs": int(pd.notna(frl) and int(frl) == lag_day),
                "log_duration_c": float(np.log1p(rec.duration) - ldbar),
                "heat_excess_c": float(rec.mean_heat_excess - hebar),
                "sin_doy": float(np.sin(2.0 * np.pi * rec.end_doy / 366.0)),
                "cos_doy": float(np.cos(2.0 * np.pi * rec.end_doy / 366.0)),
                "climate_region": rec.climate_region,
            })
    return pd.DataFrame(rows)



def fit_eventday_logit(summary: pd.DataFrame):
    pp = build_eventday_person_period(summary)
    pp["climate_region"] = pd.Categorical(pp["climate_region"], categories=list(CFG.region_order))
    try:
        model = smf.glm(formula=CFG.eventday_formula, data=pp, family=sm.families.Binomial()).fit(maxiter=300, disp=False)
    except Exception as e:
        warnings.warn(f"Primary event-day formula failed; using reduced model. Error: {e}")
        fallback_formula = "event_occurs ~ C(lag_day) + bs(year_c, df=5, include_intercept=False) + log_duration_c + heat_excess_c + sin_doy + cos_doy"
        model = smf.glm(formula=fallback_formula, data=pp, family=sm.families.Binomial()).fit(maxiter=300, disp=False)
    return model



def annual_model_implied_recovery(summary: pd.DataFrame, model) -> pd.DataFrame:
    pp = build_eventday_person_period(summary)
    pp["climate_region"] = pd.Categorical(pp["climate_region"], categories=list(CFG.region_order))
    pp["pred_hazard"] = model.predict(pp)
    rows = []
    for event_uid, g in pp.groupby("event_uid", observed=True):
        g = g.sort_values("lag_day")
        surv = 1.0
        rec10 = 0.0
        for hz in g["pred_hazard"].to_numpy(dtype=float):
            rec10 += surv * hz
            surv *= (1.0 - hz)
        rows.append({"event_uid": event_uid, "year": int(g["year"].iloc[0]), "model_recovery_prob": rec10})
    tmp = pd.DataFrame(rows)
    return tmp.groupby("year", observed=True)["model_recovery_prob"].mean().reset_index()


# =============================================================================
# Cell-level and definition-space tables
# =============================================================================



def build_or_load_cell_outcome_table(summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cache_fp = cfg.cache_dir / "result1_cell_outcome_table.csv"
    if cfg.use_cache and cache_fp.exists():
        return pd.read_csv(cache_fp)

    keep = summary[["event_uid", "source_file", "year", "recovered_by_day10"]].copy()
    rows: List[Dict] = []
    for i, rec in enumerate(keep.itertuples(index=False), start=1):
        fp = Path(rec.source_file)
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception:
            continue
        df = _clean_columns(df)
        for col in _NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "coord_key" not in df.columns and {"longitude", "latitude"}.issubset(df.columns):
            df["coord_key"] = df["longitude"].round(4).astype(str) + "_" + df["latitude"].round(4).astype(str)
        lag = pd.to_numeric(df.get("lag_day_event", pd.Series(np.nan, index=df.index)), errors="coerce")
        is_heat = _bool_num(df.get("is_heat_period_event", pd.Series(np.nan, index=df.index)))
        heat = df.loc[is_heat == 1].copy()
        if heat.empty:
            heat = df.loc[lag <= 0].copy()
        if heat.empty:
            continue
        coords = heat[["coord_key", "longitude", "latitude"]].dropna().drop_duplicates("coord_key")
        for crow in coords.itertuples(index=False):
            rows.append({
                "event_uid": rec.event_uid,
                "year": int(rec.year),
                "longitude": float(crow.longitude),
                "latitude": float(crow.latitude),
                "recovered_by_day10": int(rec.recovered_by_day10),
            })
        if (i % cfg.progress_every == 0) or (i == len(keep)):
            print(f"[INFO] Cell-outcome table: scanned {i:,}/{len(keep):,} events")
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Cell-outcome table is empty.")
    out.to_csv(cache_fp, index=False, encoding="utf-8-sig")
    return out



def _two_sided_binomial_pvalue(k: int, n: int, p0: float) -> float:
    """Two-sided binomial test with a normal-approximation fallback."""
    if n <= 0 or not np.isfinite(p0) or p0 <= 0.0 or p0 >= 1.0:
        return np.nan
    k = int(k)
    n = int(n)
    if binomtest is not None:
        try:
            return float(binomtest(k, n, p=p0, alternative="two-sided").pvalue)
        except Exception:
            pass
    var = n * p0 * (1.0 - p0)
    if var <= 0:
        return np.nan
    z = (k - n * p0) / math.sqrt(var)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def aggregate_cell_recovery(cell_df: pd.DataFrame, min_events: int) -> pd.DataFrame:
    """Aggregate event-footprint cells and compute map-stippling support.

    The map colour remains the empirical day-10 recovery probability for each
    heatwave-core grid cell. Stippling is based on a two-sided binomial test
    against the CONUS cell-exposure mean recovery probability. This is intended
    as a spatial-support diagnostic, not as a formal field-significance claim.
    """
    dat = cell_df.copy()
    dat["recovered_by_day10"] = pd.to_numeric(dat["recovered_by_day10"], errors="coerce")
    dat = dat.dropna(subset=["longitude", "latitude", "recovered_by_day10"])
    if dat.empty:
        return pd.DataFrame(columns=[
            "longitude", "latitude", "n_events", "n_recovered", "recovery_prob",
            "anomaly", "p_value", "p_fdr", "sig_raw_p05", "sig_fdr05"
        ])

    global_p = float(dat["recovered_by_day10"].mean())
    agg = dat.groupby(["longitude", "latitude"], observed=True).agg(
        n_events=("event_uid", "count"),
        n_recovered=("recovered_by_day10", "sum"),
        recovery_prob=("recovered_by_day10", "mean"),
    ).reset_index()
    agg = agg.loc[agg["n_events"] >= min_events].copy()
    if agg.empty:
        return agg

    agg["anomaly"] = agg["recovery_prob"] - global_p
    agg["global_recovery_prob_cell_exposure"] = global_p
    agg["p_value"] = [
        _two_sided_binomial_pvalue(int(k), int(n), global_p)
        for k, n in zip(agg["n_recovered"], agg["n_events"])
    ]
    pvals = pd.to_numeric(agg["p_value"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(pvals)
    agg["p_fdr"] = np.nan
    if valid.any():
        if multipletests is not None:
            _, qvals, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
            agg.loc[valid, "p_fdr"] = qvals
        else:
            pv = pvals[valid]
            order = np.argsort(pv)
            ranked = pv[order]
            m = len(ranked)
            q = ranked * m / np.arange(1, m + 1)
            q = np.minimum.accumulate(q[::-1])[::-1]
            q = np.clip(q, 0.0, 1.0)
            out = np.empty_like(q)
            out[order] = q
            agg.loc[valid, "p_fdr"] = out
    agg["sig_raw_p05"] = (pd.to_numeric(agg["p_value"], errors="coerce") < 0.05).astype(int)
    agg["sig_fdr05"] = (pd.to_numeric(agg["p_fdr"], errors="coerce") < 0.05).astype(int)
    return agg



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

    return _median_step(grid_df["longitude"].to_numpy(dtype=float), 0.25), _median_step(grid_df["latitude"].to_numpy(dtype=float), 0.25)



def build_definition_space_tables(summary: pd.DataFrame, lag_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annual_n = summary.groupby("year", observed=True)["event_uid"].count().rename("n_events").reset_index()
    rows_rf: List[Dict] = []
    rows_rw: List[Dict] = []
    rows_cont: List[Dict] = []
    lag_base = lag_df.copy()

    for rthr in cfg.rain_thresholds_mm:
        col = RTHR_LABELS[rthr]
        # a: mean recovery over rain x footprint, fixed window 10
        for fthr in cfg.footprint_thresholds:
            tmp = lag_base[["event_uid", "year", "lag_day", "coverage", col]].copy()
            tmp["hit"] = ((tmp["coverage"] >= cfg.min_detectable_coverage) & (tmp[col] >= fthr)).astype(int)
            ev = tmp.groupby(["event_uid", "year"], observed=True)["hit"].max().reset_index(name="recovered_def")
            ann = ev.groupby("year", observed=True)["recovered_def"].mean().reset_index(name="recovery_prob")
            mean_rec = float(ann["recovery_prob"].mean()) if not ann.empty else np.nan
            rows_rf.append({
                "rain_threshold_mm": rthr,
                "footprint_threshold": fthr,
                "mean_recovery": mean_rec,
            })
        # b: trend over rain x window, fixed footprint 0.25
        for wday in cfg.window_days_grid:
            tmp = lag_base.loc[lag_base["lag_day"] <= wday, ["event_uid", "year", "lag_day", "coverage", col]].copy()
            tmp["hit"] = ((tmp["coverage"] >= cfg.min_detectable_coverage) & (tmp[col] >= cfg.primary_footprint_threshold)).astype(int)
            ev = tmp.groupby(["event_uid", "year"], observed=True)["hit"].max().reset_index(name="recovered_def")
            ann = ev.groupby("year", observed=True)["recovered_def"].mean().reset_index(name="recovery_prob")
            ann = ann.merge(annual_n, on="year", how="left")
            trend = linear_trend_pp_decade(ann, "recovery_prob", "n_events")
            rows_rw.append({
                "rain_threshold_mm": rthr,
                "window_days": wday,
                "trend_pp_decade": trend,
                "mean_recovery": float(ann["recovery_prob"].mean()) if not ann.empty else np.nan,
            })
        tmp = lag_base.groupby(["event_uid", "year"], observed=True)[col].max().reset_index(name="max_rainfrac")
        ann = tmp.groupby("year", observed=True)["max_rainfrac"].mean().reset_index(name="mean_max_rainfrac")
        ann = ann.merge(annual_n, on="year", how="left")
        for row in ann.itertuples(index=False):
            rows_cont.append({
                "rain_threshold_mm": rthr,
                "year": int(row.year),
                "mean_max_rainfrac": float(row.mean_max_rainfrac),
                "n_events": int(row.n_events),
            })

    return pd.DataFrame(rows_rf), pd.DataFrame(rows_rw), pd.DataFrame(rows_cont)


# =============================================================================
# Plotting preparation
# =============================================================================



def regional_annual_recovery(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.groupby(["year", "climate_region"], observed=True)["recovered_by_day10"].mean().reset_index(name="recovery_prob")



def annual_lag_hazard_surface(summary: pd.DataFrame) -> pd.DataFrame:
    years = sorted(summary["year"].unique())
    rows = []
    for year, g in summary.groupby("year", observed=True):
        for lag_day in range(1, CFG.primary_window_days + 1):
            at_risk = g.loc[g["first_recovery_lag"].isna() | (g["first_recovery_lag"] >= lag_day)]
            hazard = float((g["first_recovery_lag"] == lag_day).sum() / len(at_risk)) if len(at_risk) else np.nan
            rows.append({"year": int(year), "lag_day": lag_day, "hazard": hazard})
    out = pd.DataFrame(rows)
    smooth_rows = []
    for lag_day, sub in out.groupby("lag_day", observed=True):
        ser = sub.set_index("year")["hazard"].reindex(years)
        sm = rolling_mean(ser, CFG.rolling_window)
        smooth_rows.append(pd.DataFrame({"year": years, "lag_day": lag_day, "hazard_smooth": sm.to_numpy(dtype=float)}))
    return pd.concat(smooth_rows, ignore_index=True)


# =============================================================================
# Map helpers
# =============================================================================



def _setup_conus_ax(
    ax,
    *,
    show_ticks: bool = True,
    show_state_lines: bool = True,
    show_outline: bool = True,
    land_facecolor: str = "#f8f4ea",
    ocean_facecolor: str = "#f7f7f7",
):
    if HAS_CARTOPY:
        ax.set_extent([-125, -66.5, 25, 50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=ocean_facecolor, edgecolor="none", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=land_facecolor, edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.65, zorder=6)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45, zorder=6)
        if show_state_lines:
            try:
                states = cfeature.NaturalEarthFeature(
                    category="cultural",
                    name="admin_1_states_provinces_lakes",
                    scale="50m",
                    facecolor="none",
                )
                ax.add_feature(states, linewidth=0.22, edgecolor="0.70", zorder=5)
            except Exception:
                pass
        if show_ticks:
            try:
                ax.set_xticks([-120, -110, -100, -90, -80, -70], crs=ccrs.PlateCarree())
                ax.set_yticks([25, 30, 35, 40, 45, 50], crs=ccrs.PlateCarree())
                ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f'))
                ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f'))
                ax.tick_params(axis="x", bottom=True, labelbottom=True, top=False, labeltop=False)
                ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False)
            except Exception:
                pass
        else:
            try:
                ax.set_xticks([], crs=ccrs.PlateCarree())
                ax.set_yticks([], crs=ccrs.PlateCarree())
            except Exception:
                ax.set_xticks([])
                ax.set_yticks([])
        if not show_outline:
            try:
                ax.outline_patch.set_visible(False)
            except Exception:
                pass
            for spine in ax.spines.values():
                spine.set_visible(False)
    else:
        ax.set_xlim(-125, -66.5)
        ax.set_ylim(25, 50)
        ax.set_facecolor("white")
        if show_ticks:
            ax.set_xticks([-120, -110, -100, -90, -80, -70])
            ax.set_yticks([25, 30, 35, 40, 45, 50])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        if not show_outline:
            for spine in ax.spines.values():
                spine.set_visible(False)


def _plot_native_grid_raster(ax, grid_df: pd.DataFrame, value_col: str, cmap, norm, lon_step: float, lat_step: float):
    lons = np.sort(np.unique(grid_df["longitude"].to_numpy(dtype=float)))
    lats = np.sort(np.unique(grid_df["latitude"].to_numpy(dtype=float)))
    piv = grid_df.pivot(index="latitude", columns="longitude", values=value_col).reindex(index=lats, columns=lons)
    z = piv.to_numpy(dtype=float)
    xedges = np.r_[lons - lon_step / 2.0, lons[-1] + lon_step / 2.0]
    yedges = np.r_[lats - lat_step / 2.0, lats[-1] + lat_step / 2.0]
    ax.pcolormesh(
        xedges,
        yedges,
        z,
        shading="auto",
        cmap=cmap,
        norm=norm,
        antialiased=False,
        linewidth=0.0,
        rasterized=True,
        zorder=2,
        transform=ccrs.PlateCarree() if HAS_CARTOPY else ax.transData,
    )


def _plot_map_significance_points(ax, cell_grid: pd.DataFrame, cfg: Config) -> None:
    """Overlay stippling for statistically supported spatial departures.

    Dots are drawn for cells satisfying both FDR P < cfg.map_sig_alpha and
    |recovery probability - CONUS cell-exposure mean| >= cfg.map_sig_min_abs_diff.
    """
    required = {"longitude", "latitude", "anomaly", "p_fdr", "sig_fdr05"}
    if cell_grid is None or cell_grid.empty or not required.issubset(set(cell_grid.columns)):
        return
    sig = cell_grid.loc[
        (pd.to_numeric(cell_grid["sig_fdr05"], errors="coerce") == 1) &
        (pd.to_numeric(cell_grid["p_fdr"], errors="coerce") < cfg.map_sig_alpha) &
        (pd.to_numeric(cell_grid["anomaly"], errors="coerce").abs() >= cfg.map_sig_min_abs_diff)
    ].copy()
    sig = sig.dropna(subset=["longitude", "latitude"])
    if sig.empty:
        return
    transform = ccrs.PlateCarree() if HAS_CARTOPY else ax.transData
    ax.scatter(
        sig["longitude"].to_numpy(dtype=float),
        sig["latitude"].to_numpy(dtype=float),
        s=cfg.map_sig_point_size,
        marker=".",
        color="black",
        alpha=cfg.map_sig_alpha_points,
        linewidths=0.0,
        transform=transform,
        zorder=8,
        rasterized=True,
    )


# =============================================================================
# Plotting
# =============================================================================


def make_nature_seq_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "nature_seq",
        ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"],
        N=256,
    )


def make_nature_div_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "nature_div",
        ["#3b4cc0", "#8db0fe", "#f7f7f7", "#f2a07b", "#b40426"],
        N=256,
    )


def _plot_main_map_panel(ax, cell_grid: pd.DataFrame, cfg: Config) -> None:
    lon_step, lat_step = infer_native_grid_spacing(cell_grid, cfg)
    _setup_conus_ax(ax, show_ticks=True, show_state_lines=True, show_outline=True)
    cmap = plt.get_cmap("YlOrBr")
    vmin = max(0.60, float(cell_grid["recovery_prob"].quantile(0.02))) if not cell_grid.empty else 0.60
    vmax = min(1.00, float(cell_grid["recovery_prob"].quantile(0.98))) if not cell_grid.empty else 1.00
    norm = Normalize(vmin=vmin, vmax=vmax)
    if not cell_grid.empty:
        _plot_native_grid_raster(ax, cell_grid, "recovery_prob", cmap, norm, lon_step, lat_step)
        _plot_map_significance_points(ax, cell_grid, cfg)
    draw_region_boundaries(ax, lw=1.9)
    cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal", pad=0.10, fraction=0.06)
    cbar.set_label("Day-10 recovery probability")


def _plot_main_dual_axis_panel(ax, annual: pd.DataFrame, rec_ci: pd.DataFrame, lag_ci: pd.DataFrame) -> None:
    ax2 = ax.twinx()
    ax.plot(annual["year"], annual["recovery_prob"], color="0.85", lw=1.0, zorder=1)
    ax.fill_between(rec_ci["year"], rec_ci["q025"], rec_ci["q975"], color="#c7d3e3", alpha=0.45, zorder=1)
    ln1 = ax.plot(rec_ci["year"], rec_ci["q500"], color=COL_BLUE, lw=2.8, zorder=2, label="Recovery")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Recovery prob.")
    ax.grid(alpha=0.2)

    if annual["median_lag"].notna().any():
        ax2.plot(annual["year"], annual["median_lag"], color="0.90", lw=1.0, zorder=1)
        ax2.fill_between(lag_ci["year"], lag_ci["q025"], lag_ci["q975"], color="#d7d7d7", alpha=0.35, zorder=1)
        ln2 = ax2.plot(lag_ci["year"], lag_ci["q500"], color=COL_ALL, lw=2.4, zorder=3, label="Median lag")
    else:
        ln2 = []
    ax2.set_ylabel("Lag (days)")
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, frameon=False, loc="upper right")


def _plot_main_hazard_panel(ax, hazard_surface: pd.DataFrame) -> None:
    piv = hazard_surface.pivot(index="lag_day", columns="year", values="hazard_smooth").sort_index()
    x = piv.columns.to_numpy(dtype=float)
    y = piv.index.to_numpy(dtype=float)
    z = piv.to_numpy(dtype=float)
    im = ax.pcolormesh(x, y, z, shading="auto", cmap="magma_r")
    valid = np.isfinite(z)
    if valid.any() and np.nanmax(z) > np.nanmin(z):
        levels = np.linspace(np.nanpercentile(z, 12), np.nanpercentile(z, 88), 6)
        levels = np.unique(np.round(levels, 3))
        if len(levels) >= 3:
            cs = ax.contour(x, y, z, levels=levels, colors="white", linewidths=0.9, alpha=0.9)
            ax.clabel(cs, fmt="%.02f", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Post-end lag")
    cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal", pad=0.16, fraction=0.08)
    cbar.set_label("Recovery hazard")


def _plot_s1_map_panel(ax) -> None:
    _setup_conus_ax(
        ax,
        show_ticks=False,
        show_state_lines=True,
        show_outline=False,
        land_facecolor="white",
        ocean_facecolor="white",
    )
    draw_filled_region_map(ax, show_labels=False, outline=True, alpha=0.92)


def _plot_s1_region_panel(ax, regional_ann: pd.DataFrame, region: str, cfg: Config) -> None:
    sub = regional_ann.loc[regional_ann["climate_region"] == region].copy().sort_values("year")
    if not sub.empty:
        ax.plot(sub["year"], sub["recovery_prob"], color=REGION_COLORS[region], lw=0.8, alpha=0.20)
        ax.plot(sub["year"], rolling_mean(sub["recovery_prob"], cfg.rolling_window), color=REGION_COLORS[region], lw=2.4)
    year_min = int(regional_ann["year"].min())
    year_max = int(regional_ann["year"].max())
    ax.set_title(region, fontsize=21, pad=6, color=REGION_COLORS[region])
    ax.set_xlim(year_min, year_max)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.12)
    ax.tick_params(axis='both', labelsize=18, length=3)
    ax.set_xlabel("Year", fontsize=21)
    ax.set_ylabel("Recovery", fontsize=21)
    for spine in ax.spines.values():
        spine.set_color(REGION_COLORS[region])
        spine.set_linewidth(1.1)



def plot_figure1_main(
    annual: pd.DataFrame,
    rec_ci: pd.DataFrame,
    lag_ci: pd.DataFrame,
    hazard_surface: pd.DataFrame,
    cell_grid: pd.DataFrame,
    cfg: Config,
) -> None:
    fig = plt.figure(figsize=(16.8, 9.8), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.24)

    ax = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree() if HAS_CARTOPY else None)
    _plot_main_map_panel(ax, cell_grid, cfg)

    ax = fig.add_subplot(gs[0, 1])
    _plot_main_dual_axis_panel(ax, annual, rec_ci, lag_ci)

    ax = fig.add_subplot(gs[1, 1])
    _plot_main_hazard_panel(ax, hazard_surface)

    savefig(fig, cfg.fig_dir / "Figure1_result1_main_final.png")

    # individual subpanels without a/b/c labels
    fig_a = plt.figure(figsize=(9.6, 7.6), facecolor="white")
    ax_a = fig_a.add_subplot(111, projection=ccrs.PlateCarree() if HAS_CARTOPY else None)
    _plot_main_map_panel(ax_a, cell_grid, cfg)
    savefig(fig_a, cfg.fig_dir / "Figure1_result1_main_final_panel_a_map.png")

    fig_b = plt.figure(figsize=(9.0, 5.8), facecolor="white")
    ax_b = fig_b.add_subplot(111)
    _plot_main_dual_axis_panel(ax_b, annual, rec_ci, lag_ci)
    savefig(fig_b, cfg.fig_dir / "Figure1_result1_main_final_panel_b_timeseries.png")

    fig_c = plt.figure(figsize=(9.0, 5.8), facecolor="white")
    ax_c = fig_c.add_subplot(111)
    _plot_main_hazard_panel(ax_c, hazard_surface)
    savefig(fig_c, cfg.fig_dir / "Figure1_result1_main_final_panel_c_hazard.png")


def plot_supp_s1_spatial_heterogeneity(regional_ann: pd.DataFrame, cfg: Config) -> None:
    fig = plt.figure(figsize=(17.6, 10.6), facecolor="white")

    map_ax = fig.add_axes([0.33, 0.28, 0.34, 0.42], projection=ccrs.PlateCarree() if HAS_CARTOPY else None)
    _plot_s1_map_panel(map_ax)

    # Manual panel layout matched to the revised Result 2 / S3 map-panel style.
    # Format: [left, bottom, width, height] in figure-fraction coordinates.
    positions = {
        "Northwest": [0.05, 0.61, 0.24, 0.18],
        "Northern Great Plains": [0.36, 0.79, 0.22, 0.16],
        "Midwest": [0.64, 0.61, 0.24, 0.18],
        "Northeast": [0.76, 0.36, 0.20, 0.18],
        "Southeast": [0.68, 0.08, 0.23, 0.18],
        "Southern Great Plains": [0.35, 0.07, 0.23, 0.18],
        "Southwest": [0.07, 0.295, 0.24, 0.18],
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

    for region in cfg.region_order:
        ax = fig.add_axes(positions[region])
        _plot_s1_region_panel(ax, regional_ann, region, cfg)

        x0, y0 = _REGION_CONNECT_POS.get(region, _REGION_LABEL_POS.get(region, (-100, 35)))
        fx, fy = target_fracs[region]
        con = ConnectionPatch(
            xyA=(x0, y0), coordsA=ccrs.PlateCarree()._as_mpl_transform(map_ax) if HAS_CARTOPY else map_ax.transData,
            xyB=(fx, fy), coordsB=ax.transAxes,
            color=REGION_COLORS[region], lw=1.0, alpha=0.95,
        )
        fig.add_artist(con)

    savefig(fig, cfg.fig_dir / "Supp_Fig_R1_01_spatial_heterogeneity.png")

    # individual subplots without panel letters
    fig_map = plt.figure(figsize=(8.6, 5.8), facecolor="white")
    ax_map = fig_map.add_subplot(111, projection=ccrs.PlateCarree() if HAS_CARTOPY else None)
    _plot_s1_map_panel(ax_map)
    savefig(fig_map, cfg.fig_dir / "Supp_Fig_R1_01_spatial_heterogeneity_panel_map.png")

    for region in cfg.region_order:
        fig_r = plt.figure(figsize=(8.6, 5.6), facecolor="white")
        ax_r = fig_r.add_subplot(111)
        _plot_s1_region_panel(ax_r, regional_ann, region, cfg)
        safe_region = re.sub(r"[^A-Za-z0-9]+", "_", region).strip("_")
        savefig(fig_r, cfg.fig_dir / f"Supp_Fig_R1_01_spatial_heterogeneity_{safe_region}.png")


def plot_supp_s2_sample_support(sample_ann: pd.DataFrame, annual: pd.DataFrame, model_ann: pd.DataFrame, cfg: Config) -> None:
    fig = plt.figure(figsize=(16.2, 10.2), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.24)

    # a annual counts
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(sample_ann["year"], sample_ann["all_events"], color="0.86", lw=1.0)
    ax.plot(sample_ann["year"], rolling_mean(sample_ann["all_events"], cfg.rolling_window), color=COL_GREY, lw=2.3, label="All")
    ax.plot(sample_ann["year"], rolling_mean(sample_ann["usable"], cfg.rolling_window), color=COL_BLUE, lw=2.2, label="Usable")
    ax.plot(sample_ann["year"], rolling_mean(sample_ann["no_overlap"], cfg.rolling_window), color=COL_GREEN, lw=2.2, label="No overlap")
    ax.plot(sample_ann["year"], rolling_mean(sample_ann["no_censor"], cfg.rolling_window), color=COL_TARGET, lw=2.2, label="No censor")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual count")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    add_panel_label(ax, "a")

    # b retained share versus usable
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(sample_ann["year"], sample_ann["retained_no_overlap_vs_usable"], color=COL_GREEN, lw=2.3, label="No overlap / usable")
    ax.plot(sample_ann["year"], sample_ann["retained_no_censor_vs_usable"], color=COL_TARGET, lw=2.3, label="No censor / usable")
    ax.plot(sample_ann["year"], rolling_mean(sample_ann["retained_no_overlap_vs_usable"], cfg.rolling_window), color=COL_GREEN, lw=2.6)
    ax.plot(sample_ann["year"], rolling_mean(sample_ann["retained_no_censor_vs_usable"], cfg.rolling_window), color=COL_TARGET, lw=2.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Retained share")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="lower left")
    add_panel_label(ax, "b")

    # c observed vs event-day model
    ax = fig.add_subplot(gs[1, 0])
    dd = annual.merge(model_ann, on="year", how="left")
    ax.plot(dd["year"], dd["recovery_prob"], color="0.85", lw=1.0)
    ax.plot(dd["year"], rolling_mean(dd["recovery_prob"], cfg.rolling_window), color=COL_ALL, lw=2.5, label="Observed")
    ax.plot(dd["year"], rolling_mean(dd["model_recovery_prob"], cfg.rolling_window), color=COL_TARGET, lw=2.5, label="Event-day logit")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Recovery")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="lower left")
    add_panel_label(ax, "c")

    # d mean minimum coverage
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(sample_ann["year"], sample_ann["mean_min_coverage"], color="0.82", lw=1.0)
    ax.plot(sample_ann["year"], rolling_mean(sample_ann["mean_min_coverage"], cfg.rolling_window), color=COL_BROWN, lw=2.5)
    ax.axhline(cfg.usable_min_coverage, color=COL_TARGET, lw=1.3, ls="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Minimum coverage")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    add_panel_label(ax, "d")

    savefig(fig, cfg.fig_dir / "Supp_Fig_R1_02_sample_support.png")



def plot_supp_s3_definition_space(rf_df: pd.DataFrame, rw_df: pd.DataFrame, cont_df: pd.DataFrame, cfg: Config) -> None:
    fig = plt.figure(figsize=(16.7, 10.8), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.42)

    nature_seq = make_nature_seq_cmap()
    nature_div = make_nature_div_cmap()

    # a mean recovery heatmap
    ax = fig.add_subplot(gs[0, 0])
    piv = rf_df.pivot(index="footprint_threshold", columns="rain_threshold_mm", values="mean_recovery").sort_index(ascending=False)
    piv = piv.reindex(index=sorted(cfg.footprint_thresholds, reverse=True), columns=list(cfg.rain_thresholds_mm))
    z = piv.to_numpy(dtype=float)
    try:
        nature_seq.set_bad("#d9d9d9")
    except Exception:
        pass
    im = ax.imshow(z, aspect="auto", cmap=nature_seq, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([str(c) for c in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in piv.index])
    ax.set_xlabel("Rain thr. (mm d$^{-1}$)")
    ax.set_ylabel("Area frac.")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
    cbar.set_label("Mean recovery")
    add_panel_label(ax, "a", x=-0.16, y=1.08)

    # b trend heatmap with Nature-like diverging palette
    ax = fig.add_subplot(gs[0, 1])
    piv = rw_df.pivot(index="window_days", columns="rain_threshold_mm", values="trend_pp_decade").sort_index(ascending=False)
    piv = piv.reindex(index=sorted(cfg.window_days_grid, reverse=True), columns=list(cfg.rain_thresholds_mm))
    z = piv.to_numpy(dtype=float)
    vmax = max(0.20, float(np.nanpercentile(np.abs(z), 95))) if np.isfinite(z).any() else 0.20
    try:
        nature_div.set_bad("#d9d9d9")
    except Exception:
        pass
    im = ax.imshow(z, aspect="auto", cmap=nature_div, vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([str(c) for c in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([str(v) for v in piv.index])
    ax.set_xlabel("Rain thr. (mm d$^{-1}$)")
    ax.set_ylabel("Window (days)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Trend (pp dec$^{-1}$)")
    add_panel_label(ax, "b", x=-0.16, y=1.08)

    # c continuous outcome support
    ax = fig.add_subplot(gs[1, :])
    for rthr in (1.0, 2.0, 5.0):
        sub = cont_df.loc[cont_df["rain_threshold_mm"] == rthr].copy().sort_values("year")
        if sub.empty:
            continue
        color = {1.0: COL_BLUE, 2.0: COL_GREEN, 5.0: COL_ORANGE}[rthr]
        ax.plot(sub["year"], sub["mean_max_rainfrac"], color=color, lw=1.0, alpha=0.22)
        ax.plot(sub["year"], rolling_mean(sub["mean_max_rainfrac"], cfg.rolling_window), color=color, lw=2.5, label=f"{rthr:g} mm d$^{{-1}}$")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Max rain frac.")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    add_panel_label(ax, "c", x=-0.16, y=1.08)

    savefig(fig, cfg.fig_dir / "Supp_Fig_R1_03_definition_space_and_continuous.png")


# =============================================================================
# Main workflow
# =============================================================================



def main(cfg: Config = CFG) -> None:
    summary, lag_df, precip_note = build_or_load_event_tables(cfg)
    print(f"[INFO] Total summarized events : {len(summary):,}")
    print(f"[INFO] Precip note             : {precip_note}")

    main_df, sample_note = choose_main_sample(summary)
    print(f"[INFO] Sample rule            : {sample_note}")

    annual = annual_primary_metrics(main_df)
    annual.to_csv(cfg.table_dir / "result1_annual_primary_metrics.csv", index=False, encoding="utf-8-sig")

    sample_ann = sample_support_annual(summary)
    sample_ann.to_csv(cfg.table_dir / "result1_sample_support_annual.csv", index=False, encoding="utf-8-sig")

    years = sorted(annual["year"].unique())
    rec_ci = bootstrap_rolling_metric(main_df, "recovery_prob", years, cfg.rolling_window, cfg.bootstrap_n, cfg.bootstrap_seed)
    lag_ci = bootstrap_rolling_metric(main_df, "median_lag", years, cfg.rolling_window, cfg.bootstrap_n, cfg.bootstrap_seed + 7)
    rec_ci.to_csv(cfg.table_dir / "result1_recoveryprob_rolling_ci.csv", index=False, encoding="utf-8-sig")
    lag_ci.to_csv(cfg.table_dir / "result1_lag_rolling_ci.csv", index=False, encoding="utf-8-sig")

    hazard_surface = annual_lag_hazard_surface(main_df)
    hazard_surface.to_csv(cfg.table_dir / "result1_hazard_surface.csv", index=False, encoding="utf-8-sig")

    cell_df = build_or_load_cell_outcome_table(main_df, cfg)
    cell_grid = aggregate_cell_recovery(cell_df, cfg.map_min_events)
    cell_grid.to_csv(cfg.table_dir / "result1_cell_recovery_grid.csv", index=False, encoding="utf-8-sig")

    regional_ann = regional_annual_recovery(main_df)
    regional_ann.to_csv(cfg.table_dir / "result1_regional_annual_recovery.csv", index=False, encoding="utf-8-sig")

    rf_df, rw_df, cont_df = build_definition_space_tables(
        main_df,
        lag_df.loc[lag_df["event_uid"].isin(main_df["event_uid"])].copy(),
        cfg,
    )
    rf_df.to_csv(cfg.table_dir / "result1_definition_space_rain_x_footprint.csv", index=False, encoding="utf-8-sig")
    rw_df.to_csv(cfg.table_dir / "result1_definition_space_rain_x_window.csv", index=False, encoding="utf-8-sig")
    cont_df.to_csv(cfg.table_dir / "result1_continuous_outcome_by_threshold.csv", index=False, encoding="utf-8-sig")

    model = fit_eventday_logit(main_df)
    model_ann = annual_model_implied_recovery(main_df, model)
    model_ann.to_csv(cfg.table_dir / "result1_eventday_logit_annual_recovery.csv", index=False, encoding="utf-8-sig")

    plot_figure1_main(annual, rec_ci, lag_ci, hazard_surface, cell_grid, cfg)
    plot_supp_s1_spatial_heterogeneity(regional_ann, cfg)
    plot_supp_s2_sample_support(sample_ann, annual, model_ann, cfg)
    plot_supp_s3_definition_space(rf_df, rw_df, cont_df, cfg)

    print("[INFO] Result 1 rebuild finished.")
    print(f"[INFO] Output directory: {cfg.out_dir}")


if __name__ == "__main__":
    main(CFG)
