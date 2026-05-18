# -*- coding: utf-8 -*-
"""
Result 3 mechanism supplement from event-window CSVs
Author: OpenAI
Purpose:
1) Build event-end Ridging–Subsidence Index (RSI) from Z500, T850, W500 over the immediate event-end window (default tau=-2..0)
2) Build event-end Moisture-Support Deficit Index (MSDI) from RH and moisture convergence over the immediate event-end window (default tau=-2..0)
3) Produce a compact mechanism figure set directly from event-window CSV data

Important scope:
- This script builds event-end proxies from event-window CSVs.
- It does NOT claim formal blocking identification (e.g., Tibaldi–Molteni / Pelly–Hoskins).
- It does NOT perform full moisture-budget closure.
"""

from __future__ import annotations

import math
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Polygon as MplPolygon
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_ind, linregress

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    CARTOPY_OK = True
except Exception:
    CARTOPY_OK = False
    ccrs = None
    cfeature = None
    shpreader = None
    LongitudeFormatter = None
    LatitudeFormatter = None

try:
    from shapely.geometry import Polygon, MultiPolygon, Point
    from shapely.ops import unary_union
    SHAPELY_OK = True
except Exception:
    SHAPELY_OK = False
    Polygon = None
    MultiPolygon = None
    Point = None
    unary_union = None

@dataclass
class CFG:
    # Event-window CSV root. This should be the same event-window repository used
    # to build the original Result 3 mechanism tables / Fig. 5--6 diagnostics.
    data_root: Path = Path(r"E:\temp_events_ERA5_S1S6_Nature所有数据版本\zu")
    out_root: Path = Path(r"E:\Result3_RSI_MSDI_from_event_windows")
    cache_dir: Path = field(init=False)
    fig_dir: Path = field(init=False)
    table_dir: Path = field(init=False)

    target_region: str = "Northwest"
    years: Tuple[int, int] = (1950, 2024)

    # Baseline recovery definition. Keep identical to the main Result 1--2
    # event-level recovery definition unless you intentionally run a sensitivity.
    rain_threshold_mm_day: float = 1.0
    rainy_fraction_threshold: float = 0.25
    recovery_window_days: Tuple[int, int] = (1, 10)

    # Main proxy window and lead-time window are intentionally separated:
    #   - full_lead_taus: used to retain tau=-5..0 trajectories / sensitivity
    #   - event_end_taus: used for the primary integrated event-end proxies
    event_end_taus: Tuple[int, int] = (-2, 0)
    full_lead_taus: Tuple[int, int] = (-5, 0)
    tau_sensitivity_windows: Tuple[Tuple[int, int], ...] = ((-5, 0), (-2, 0), (0, 0))

    # ERA5 pressure vertical velocity convention: omega > 0 means downward motion.
    # If your CSV has already been converted to ascent = -omega, set this to False.
    w500_positive_is_subsidence: bool = True

    # Optional: if you already have the main Result 1/2 event catalog containing
    # event_uid/year/event_id plus region/recovered/no_recovery, set this path to
    # force this script to use the exact same labels. If None or missing, the script
    # recomputes the labels from event-window CSVs and writes QA tables.
    main_event_catalog_path: Optional[Path] = Path(r"E:\新result2\cache\result2_event_summary.csv")

    dpi: int = 320
    rolling_window_years: int = 15
    min_obs_per_year: int = 5
    require_complete_rolling_window: bool = True
    n_boot: int = 500
    seed: int = 42

    conus_extent: Tuple[float, float, float, float] = (-125.0, -66.5, 24.0, 50.0)
    map_res_deg: float = 0.25
    map_smooth_sigma: float = 0.0
    spatial_min_years: int = 25
    spatial_min_obs_per_cell: int = 30

    def __post_init__(self):
        self.cache_dir = self.out_root / "cache"
        self.fig_dir = self.out_root / "figures"
        self.table_dir = self.out_root / "tables"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)

    def tau_tag(self, window: Optional[Tuple[int, int]] = None) -> str:
        lo, hi = self.event_end_taus if window is None else window
        def fmt(v: int) -> str:
            return f"m{abs(v)}" if int(v) < 0 else f"p{int(v)}"
        return f"tau_{fmt(int(lo))}_to_{fmt(int(hi))}"

    def cache_tag(self) -> str:
        wsign = "omegaPosDown" if self.w500_positive_is_subsidence else "omegaPosUp"
        return (
            f"{self.tau_tag()}_rain{self.rain_threshold_mm_day:g}mm_"
            f"frac{self.rainy_fraction_threshold:g}_w{self.recovery_window_days[0]}to{self.recovery_window_days[1]}_"
            f"{wsign}"
        )

REGION_COLORS = {
    "Northwest": "#c0392b",
    "Northern Great Plains": "#7f8c8d",
    "Midwest": "#8e44ad",
    "Northeast": "#1f78b4",
    "Southwest": "#d95f02",
    "Southern Great Plains": "#1b9e77",
    "Southeast": "#e6ab02",
    "All": "#222222",
}
PHYS_COLORS = {
    "RSI": "#6a3d9a",
    "MSDI": "#1f78b4",
    "Land": "#8c613c",
    "Recovered": "#355C7D",
    "Unrecovered": "#C06C84",
}

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.titlesize": 20,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


REGION_ORDER = (
    "Northwest",
    "Northern Great Plains",
    "Midwest",
    "Northeast",
    "Southwest",
    "Southern Great Plains",
    "Southeast",
)

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

_STATE_RECORDS: List[Dict] = []
_REGION_GEOMS: Dict[str, object] = {}

_FALLBACK_REGION_POLYGONS = {
    "Northwest": [(-125, 42), (-116, 42), (-116, 50), (-125, 50)],
    "Southwest": [(-125, 25), (-104, 25), (-104, 42), (-125, 42)],
    "Northern Great Plains": [(-116, 40), (-104, 40), (-104, 50), (-116, 50)],
    "Midwest": [(-104, 36), (-92, 36), (-92, 40), (-104, 40)],
    "Southern Great Plains": [(-104, 25), (-92, 25), (-92, 36), (-104, 36), (-104, 25),
                              (-92, 25), (-66.5, 25), (-66.5, 31), (-92, 31)],
    "Northeast": [(-92, 40), (-66.5, 40), (-66.5, 50), (-92, 50)],
    "Southeast": [(-92, 31), (-66.5, 31), (-66.5, 40), (-92, 40)],
}

def _clean_axis_spines(ax, full_box: bool = False, lw: float = 1.0):
    for side, spine in ax.spines.items():
        spine.set_linewidth(lw)
        spine.set_color("#333333")
    if full_box:
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

def _load_state_records() -> List[Dict]:
    global _STATE_RECORDS
    if _STATE_RECORDS:
        return _STATE_RECORDS
    if not (CARTOPY_OK and SHAPELY_OK and shpreader is not None):
        return []
    try:
        shp = shpreader.natural_earth(
            resolution="50m",
            category="cultural",
            name="admin_1_states_provinces_lakes",
        )
        keep = set(PAPER7_STATE_TO_REGION.keys())
        out: List[Dict] = []
        for rec in shpreader.Reader(shp).records():
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
    except Exception:
        _STATE_RECORDS = []
    return _STATE_RECORDS

def _region_geometries() -> Dict[str, object]:
    global _REGION_GEOMS
    if _REGION_GEOMS:
        return _REGION_GEOMS
    if not SHAPELY_OK:
        return {}
    by_region = {r: [] for r in REGION_ORDER}
    for rec in _load_state_records():
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

def _polygon_to_patches(geom) -> List[MplPolygon]:
    patches: List[MplPolygon] = []
    if geom is None:
        return patches
    try:
        if geom.is_empty:
            return patches
    except Exception:
        return patches
    if SHAPELY_OK and Polygon is not None and isinstance(geom, Polygon):
        patches.append(MplPolygon(np.asarray(geom.exterior.coords), closed=True))
    elif SHAPELY_OK and MultiPolygon is not None and isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            patches.extend(_polygon_to_patches(part))
    else:
        try:
            if geom.geom_type == "Polygon":
                patches.append(MplPolygon(np.asarray(geom.exterior.coords), closed=True))
            elif geom.geom_type == "MultiPolygon":
                for part in geom.geoms:
                    patches.extend(_polygon_to_patches(part))
        except Exception:
            pass
    return patches

def _plot_geom_boundary(ax, geom, color="black", lw=1.3, zorder=7):
    if geom is None:
        return
    try:
        if geom.is_empty:
            return
    except Exception:
        return
    transform = ccrs.PlateCarree() if CARTOPY_OK else ax.transData
    if SHAPELY_OK and Polygon is not None and isinstance(geom, Polygon):
        coords = np.asarray(geom.exterior.coords)
        ax.plot(coords[:, 0], coords[:, 1], color=color, lw=lw, transform=transform, zorder=zorder)
    elif SHAPELY_OK and MultiPolygon is not None and isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            _plot_geom_boundary(ax, part, color=color, lw=lw, zorder=zorder)

def draw_filled_region_map(ax, cfg: CFG, alpha: float = 0.92, outline: bool = True):
    """Draw the same central 7-region visual grammar used in Supplementary Fig. 8.
    If Cartopy is unavailable, draw a non-empty fallback CONUS region schematic rather than a blank panel.
    """
    if CARTOPY_OK:
        ax.set_extent(cfg.conus_extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="white", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.65, zorder=6)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45, zorder=6)
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
        ax.set_xticks([])
        ax.set_yticks([])
        try:
            ax.outline_patch.set_visible(False)
        except Exception:
            pass
        for spine in ax.spines.values():
            spine.set_visible(False)

        geoms = _region_geometries()
        patches = []
        facecolors = []
        for region in REGION_ORDER:
            region_patches = _polygon_to_patches(geoms.get(region))
            patches.extend(region_patches)
            facecolors.extend([REGION_COLORS[region]] * len(region_patches))
        if patches:
            pc = PatchCollection(
                patches,
                facecolor=facecolors,
                edgecolor="none",
                alpha=alpha,
                zorder=2,
                transform=ccrs.PlateCarree(),
            )
            ax.add_collection(pc)
            if outline:
                for region in REGION_ORDER:
                    _plot_geom_boundary(ax, geoms.get(region), color="black", lw=1.3, zorder=8)
            return

    # Fallback when Cartopy / shapely geometries are unavailable.
    ax.set_xlim(cfg.conus_extent[0], cfg.conus_extent[1])
    ax.set_ylim(cfg.conus_extent[2], cfg.conus_extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for region in REGION_ORDER:
        coords = _FALLBACK_REGION_POLYGONS.get(region)
        if not coords:
            continue
        patch = MplPolygon(coords, closed=True, facecolor=REGION_COLORS[region],
                           edgecolor="black", linewidth=1.1, alpha=alpha, zorder=2)
        ax.add_patch(patch)
    for spine in ax.spines.values():
        spine.set_visible(False)

def _connection_transform(map_ax):
    if CARTOPY_OK:
        return ccrs.PlateCarree()._as_mpl_transform(map_ax)
    return map_ax.transData


def assign_region7(lon: float, lat: float) -> Optional[str]:
    if np.isnan(lon) or np.isnan(lat):
        return None
    if lon <= -104:
        return "Northwest" if lat >= 42 else "Southwest"
    if -104 < lon <= -92:
        if lat >= 40:
            return "Northern Great Plains"
        elif lat >= 36:
            return "Midwest"
        else:
            return "Southern Great Plains"
    if lon > -92:
        if lat >= 40:
            return "Northeast"
        elif lat >= 31:
            return "Southeast"
        else:
            return "Southern Great Plains"
    return None


_POINT_REGION_CACHE: Dict[Tuple[float, float], Optional[str]] = {}


def _point_region_from_state_polygons(lon: float, lat: float) -> Optional[str]:
    """Assign a point to the Paper-7 region using U.S. state polygons.

    This is the region-assignment logic used for the proxy script when possible:
    heatwave-core footprint cells are mapped to states, then states are mapped to
    the seven climate regions. The fast centroid rule is retained only as fallback.
    """
    if not np.isfinite(lon) or not np.isfinite(lat):
        return None
    key = (round(float(lon), 4), round(float(lat), 4))
    if key in _POINT_REGION_CACHE:
        return _POINT_REGION_CACHE[key]

    region = None
    if CARTOPY_OK and SHAPELY_OK and Point is not None:
        pt = Point(float(lon), float(lat))
        for rec in _load_state_records():
            minx, miny, maxx, maxy = rec.get("bounds", (np.nan, np.nan, np.nan, np.nan))
            if float(lon) < minx or float(lon) > maxx or float(lat) < miny or float(lat) > maxy:
                continue
            geom = rec.get("geom")
            try:
                if geom is not None and (geom.contains(pt) or geom.touches(pt)):
                    region = PAPER7_STATE_TO_REGION.get(rec.get("abbr"))
                    break
            except Exception:
                continue

    # Fallback is the same coarse seven-region rule used by the earlier proxy code.
    if region is None:
        region = assign_region7(float(lon), float(lat))
    _POINT_REGION_CACHE[key] = region
    return region


def assign_region7_from_core(core: pd.DataFrame) -> Tuple[Optional[str], str]:
    """Assign event region by majority of heatwave-core footprint cells.

    This avoids the centroid-only misclassification risk near regional boundaries.
    If polygon assignment fails, it falls back to the centroid rule and records the
    method in the output table for QA.
    """
    pts = core[["longitude", "latitude"]].copy()
    pts["longitude"] = pd.to_numeric(pts["longitude"], errors="coerce")
    pts["latitude"] = pd.to_numeric(pts["latitude"], errors="coerce")
    pts = pts.dropna().round({"longitude": 4, "latitude": 4}).drop_duplicates()
    if pts.empty:
        return None, "no_core_points"

    counts: Dict[str, int] = {}
    for row in pts.itertuples(index=False):
        reg = _point_region_from_state_polygons(float(row.longitude), float(row.latitude))
        if reg is not None:
            counts[reg] = counts.get(reg, 0) + 1

    centroid_lon = float(np.nanmean(pts["longitude"]))
    centroid_lat = float(np.nanmean(pts["latitude"]))
    centroid_reg = assign_region7(centroid_lon, centroid_lat)

    if counts:
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_region, top_n = ordered[0]
        # If there is an exact tie, prefer the centroid region only if it is one
        # of the tied candidates; otherwise keep deterministic alphabetical order.
        tied = [r for r, n in ordered if n == top_n]
        if len(tied) > 1 and centroid_reg in tied:
            return centroid_reg, "state_majority_tie_centroid_resolved"
        return top_region, "state_majority"

    if centroid_reg is not None:
        return centroid_reg, "centroid_fallback"
    return None, "unassigned"


def list_event_files(root: Path, y0: int, y1: int) -> List[Path]:
    files: List[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir():
            try:
                y = int(p.name)
            except Exception:
                continue
            if y0 <= y <= y1:
                files.extend(sorted(p.glob("event_*_window.csv")))
    return files

def to_mm_precip_if_needed(s: pd.Series) -> Tuple[pd.Series, str]:
    s = pd.to_numeric(s, errors="coerce")
    arr = s.values
    if np.isfinite(arr).sum() == 0:
        return s, "all_nan"
    q99 = np.nanpercentile(arr, 99)
    if np.isfinite(q99) and q99 < 1.0:
        return s * 1000.0, f"auto_meters_to_mm_q0.99={q99:.7g}"
    return s, "loaded_as_mm_or_already_scaled"

def safe_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(x.values)) if len(x) else np.nan

def se(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return np.nan
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))

def z_standardize(series: pd.Series) -> pd.Series:
    arr = pd.to_numeric(series, errors="coerce").values.astype(float)
    mask = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=float)
    if mask.sum() >= 2:
        mu = arr[mask].mean()
        sd = arr[mask].std(ddof=0)
        if sd > 0:
            out[mask] = (arr[mask] - mu) / sd
        else:
            out[mask] = 0.0
    return pd.Series(out, index=series.index)

def slope_pp_decade(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0] * 10.0)



def _np1(x):
    return np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)

def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=21, fontweight="bold", va="top", ha="left")

def savefig(fig: plt.Figure, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def ensure_cols(df: pd.DataFrame, cols: Sequence[str], path: Path):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns in {path.name}: {miss}")

def summarize_one_event(path: Path, cfg: CFG, precip_note_holder: Dict[str, str]) -> Optional[Dict]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None

    required = [
        "longitude", "latitude", "year", "lag_day_event",
        "is_heat_period_event", "overlaps_next_local_heat", "is_post_event_0_10_censored",
        "precipitation", "relative_humidity", "vertically_integrated_moisture_divergence_mean",
        "convective_available_potential_energy_mean", "temperature_850hPa_mean",
        "geopotential_500hPa", "vertical_velocity_500hPa",
        "soil_moist", "Bowen_ratio", "Rn", "temp_air", "T90"
    ]
    try:
        ensure_cols(df, required, path)
    except Exception:
        return None

    year_vals = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
    if year_vals.empty:
        return None
    year = int(year_vals.iloc[0])

    if "event_id" in df.columns and pd.to_numeric(df["event_id"], errors="coerce").notna().any():
        event_id = int(pd.to_numeric(df["event_id"], errors="coerce").dropna().iloc[0])
    else:
        try:
            event_id = int(path.stem.split("_")[2])
        except Exception:
            event_id = -1

    pr, note = to_mm_precip_if_needed(df["precipitation"])
    precip_note_holder.setdefault("note", note)
    df = df.copy()
    df["precip_mm"] = pr

    core = df.loc[pd.to_numeric(df["is_heat_period_event"], errors="coerce") == 1].copy()
    if core.empty:
        core = df.loc[pd.to_numeric(df["lag_day_event"], errors="coerce") <= 0].copy()
    if core.empty:
        return None

    core["longitude"] = pd.to_numeric(core["longitude"], errors="coerce")
    core["latitude"] = pd.to_numeric(core["latitude"], errors="coerce")
    centroid_lon = float(np.nanmean(core["longitude"]))
    centroid_lat = float(np.nanmean(core["latitude"]))
    region, region_method = assign_region7_from_core(core)
    if region is None:
        return None

    overlap_flag = int(pd.to_numeric(df["overlaps_next_local_heat"], errors="coerce").fillna(0).max())
    censor_flag = int(pd.to_numeric(df["is_post_event_0_10_censored"], errors="coerce").fillna(0).max())

    post = df.loc[
        (pd.to_numeric(df["lag_day_event"], errors="coerce") >= cfg.recovery_window_days[0]) &
        (pd.to_numeric(df["lag_day_event"], errors="coerce") <= cfg.recovery_window_days[1])
    ].copy()
    if post.empty:
        return None

    if "coord_key" in core.columns:
        denom = core["coord_key"].astype(str).nunique()
    else:
        denom = core[["longitude", "latitude"]].round(4).drop_duplicates().shape[0]
    if denom <= 0:
        return None

    post["lag_day_event"] = pd.to_numeric(post["lag_day_event"], errors="coerce").astype(int)
    post["rainy"] = (pd.to_numeric(post["precip_mm"], errors="coerce") >= cfg.rain_threshold_mm_day).astype(int)

    if "coord_key" in post.columns:
        daily = (post.groupby(["lag_day_event", "coord_key"], as_index=False)["rainy"].max()
                   .groupby("lag_day_event")["rainy"].sum())
    else:
        post["_ck"] = post["longitude"].round(4).astype(str) + "_" + post["latitude"].round(4).astype(str)
        daily = (post.groupby(["lag_day_event", "_ck"], as_index=False)["rainy"].max()
                   .groupby("lag_day_event")["rainy"].sum())

    daily_frac = (daily / denom).reindex(range(cfg.recovery_window_days[0], cfg.recovery_window_days[1] + 1)).fillna(0.0)
    recovered_days = daily_frac[daily_frac >= cfg.rainy_fraction_threshold]
    recovered = int(len(recovered_days) > 0)
    first_recovery_lag = int(recovered_days.index.min()) if recovered else np.nan

    lead = df.loc[
        (pd.to_numeric(df["lag_day_event"], errors="coerce") >= cfg.full_lead_taus[0]) &
        (pd.to_numeric(df["lag_day_event"], errors="coerce") <= cfg.full_lead_taus[1])
    ].copy()
    if lead.empty:
        return None

    main = lead.loc[
        (pd.to_numeric(lead["lag_day_event"], errors="coerce") >= cfg.event_end_taus[0]) &
        (pd.to_numeric(lead["lag_day_event"], errors="coerce") <= cfg.event_end_taus[1])
    ].copy()
    if main.empty:
        return None

    main["moisture_convergence"] = -pd.to_numeric(main["vertically_integrated_moisture_divergence_mean"], errors="coerce")
    lead["moisture_convergence"] = -pd.to_numeric(lead["vertically_integrated_moisture_divergence_mean"], errors="coerce")
    main["heat_excess"] = pd.to_numeric(main["temp_air"], errors="coerce") - pd.to_numeric(main["T90"], errors="coerce")

    out = {
        "event_uid": f"{year}_{event_id:05d}",
        "year": year,
        "event_id": event_id,
        "centroid_lon": centroid_lon,
        "centroid_lat": centroid_lat,
        "region": region,
        "region_assignment_method": region_method,
        "overlap_flag": overlap_flag,
        "censor_flag": censor_flag,
        "recovered": recovered,
        "no_recovery": 1 - recovered,
        "first_recovery_lag": first_recovery_lag,
        "mean_rain_fraction_w10": float(daily_frac.mean() * 100.0),
        "max_rain_fraction_w10": float(daily_frac.max() * 100.0),
        "cumulative_rain_fraction_w10": float(daily_frac.sum() * 100.0),
        "duration_days": int(core["lag_day_event"].max() - core["lag_day_event"].min() + 1),
        "event_end_heat_excess": safe_mean(main["heat_excess"]),
        "soil_moist_end": safe_mean(main["soil_moist"]),
        "Bowen_ratio_end": safe_mean(main["Bowen_ratio"]),
        "Rn_end": safe_mean(main["Rn"]),
        "RH_end": safe_mean(main["relative_humidity"]),
        "MC_end": safe_mean(main["moisture_convergence"]),
        "CAPE_end": safe_mean(main["convective_available_potential_energy_mean"]),
        "P_end": safe_mean(main["precip_mm"]),
        "Z500_end": safe_mean(main["geopotential_500hPa"]),
        "T850_end": safe_mean(main["temperature_850hPa_mean"]),
        "W500_end": safe_mean(main["vertical_velocity_500hPa"]),
    }
    for tau in range(cfg.full_lead_taus[0], cfg.full_lead_taus[1] + 1):
        sub = lead.loc[pd.to_numeric(lead["lag_day_event"], errors="coerce") == tau]
        out[f"Z500_tau{tau}"] = safe_mean(sub["geopotential_500hPa"])
        out[f"T850_tau{tau}"] = safe_mean(sub["temperature_850hPa_mean"])
        out[f"W500_tau{tau}"] = safe_mean(sub["vertical_velocity_500hPa"])
        out[f"RH_tau{tau}"] = safe_mean(sub["relative_humidity"])
        out[f"MC_tau{tau}"] = safe_mean(sub["moisture_convergence"])
        out[f"CAPE_tau{tau}"] = safe_mean(sub["convective_available_potential_energy_mean"])
        out[f"P_tau{tau}"] = safe_mean(sub["precip_mm"])
        out[f"soil_tau{tau}"] = safe_mean(sub["soil_moist"])
        out[f"Bowen_tau{tau}"] = safe_mean(sub["Bowen_ratio"])
        out[f"Rn_tau{tau}"] = safe_mean(sub["Rn"])
    return out

def build_event_summary(cfg: CFG) -> pd.DataFrame:
    cache_path = cfg.cache_dir / f"event_summary_rsi_msdi_{cfg.cache_tag()}.csv"
    meta_path = cfg.cache_dir / f"event_summary_rsi_msdi_{cfg.cache_tag()}_metadata.json"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"[INFO] Total summarized events : {len(df):,}")
        print(f"[INFO] Precip note             : loaded_cached_event_summary | {cfg.cache_tag()}")
        return df

    files = list_event_files(cfg.data_root, cfg.years[0], cfg.years[1])
    rows = []
    holder: Dict[str, str] = {}
    for i, fp in enumerate(files, start=1):
        row = summarize_one_event(fp, cfg, holder)
        if row is not None:
            rows.append(row)
        if i % 250 == 0 or i == len(files):
            print(f"[INFO] Summarized {i:,}/{len(files):,} files | retained {len(rows):,}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable event summaries were built.")
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    meta = {
        "cache_tag": cfg.cache_tag(),
        "event_end_taus": list(cfg.event_end_taus),
        "full_lead_taus": list(cfg.full_lead_taus),
        "rain_threshold_mm_day": cfg.rain_threshold_mm_day,
        "rainy_fraction_threshold": cfg.rainy_fraction_threshold,
        "recovery_window_days": list(cfg.recovery_window_days),
        "w500_positive_is_subsidence": cfg.w500_positive_is_subsidence,
        "precip_note": holder.get("note", "unknown"),
        "n_events": int(len(df)),
        "region_assignment_counts": df.get("region_assignment_method", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Total summarized events : {len(df):,}")
    print(f"[INFO] Precip note             : {holder.get('note', 'unknown')}")
    print(f"[INFO] Cache tag               : {cfg.cache_tag()}")
    return df




def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first existing column using case-insensitive matching."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        c = lookup.get(str(cand).strip().lower())
        if c is not None:
            return c
    return None


def _as_binary_int(series: pd.Series) -> pd.Series:
    """Convert common numeric/bool labels to nullable 0/1 integers."""
    if series is None:
        return pd.Series(dtype="Int64")
    s = series.copy()
    if s.dtype == bool:
        return s.astype("Int64")
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        return num.round().astype("Int64")
    low = s.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    out.loc[low.isin({"1", "true", "yes", "y", "recovered", "recovery"})] = 1
    out.loc[low.isin({"0", "false", "no", "n", "unrecovered", "no_recovery", "not recovered"})] = 0
    return out


def apply_external_catalog_overrides(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    """Use the main Result 2 catalog as the authoritative event-label table.

    For submission, Result 3 RSI/MSDI should not redefine regions or recovery
    outcomes. It should calculate event-end proxy variables from event-window CSVs
    and then merge those proxy variables onto the Result 2 event catalog. This
    function therefore:
      1) reads the Result 2 event summary table;
      2) keeps only proxy-available events that are present in Result 2;
      3) overwrites region with Result 2 ``region4``;
      4) overwrites recovery labels with Result 2 ``recovered_by_day10`` and
         ``no_recovery``;
      5) carries Result 2 sample flags for downstream sample selection.

    The internally recomputed labels are retained only as QA columns with the
    suffix ``_internal_proxy``.
    """
    path = cfg.main_event_catalog_path
    if path is None or str(path).strip() == "" or not Path(path).exists():
        print("[WARN] Main event catalog      : not found; using labels recomputed from event-window CSVs")
        print(f"[WARN] Expected catalog path  : {path}")
        out = df.copy()
        out["catalog_matched"] = 0
        out["sample_source"] = "proxy_internal_no_catalog"
        return out

    cat = pd.read_csv(path, low_memory=False)
    cat.columns = [str(c).strip() for c in cat.columns]

    if "event_uid" not in cat.columns:
        if {"year", "event_id"}.issubset(cat.columns):
            yy = pd.to_numeric(cat["year"], errors="coerce").astype("Int64")
            ee = pd.to_numeric(cat["event_id"], errors="coerce").astype("Int64")
            cat["event_uid"] = [
                f"{int(y)}_{int(e):05d}" if pd.notna(y) and pd.notna(e) else np.nan
                for y, e in zip(yy, ee)
            ]
        else:
            raise ValueError(
                "Main Result 2 catalog must contain event_uid or year/event_id. "
                f"Catalog path: {path}"
            )

    region_col = _first_existing_column(cat, ["region4", "region", "climate_region", "paper7_region"])
    recovered_col = _first_existing_column(cat, ["recovered_by_day10", "recovered", "is_recovered"])
    no_recovery_col = _first_existing_column(cat, ["no_recovery", "norecovery", "is_no_recovery"])
    strict_col = _first_existing_column(cat, ["strict_flag", "strict", "is_strict"])
    usable_col = _first_existing_column(cat, ["usable_flag", "usable", "is_usable"])
    overlap_col = _first_existing_column(cat, ["any_overlap", "overlap_flag", "overlaps_next_local_heat"])
    censor_col = _first_existing_column(cat, ["any_censor", "censor_flag", "is_post_event_0_10_censored"])

    if region_col is None:
        raise ValueError("Result 2 catalog has no usable region column. Expected one of: region4, region, climate_region.")
    if recovered_col is None and no_recovery_col is None:
        raise ValueError("Result 2 catalog has no usable recovery label. Expected recovered_by_day10 and/or no_recovery.")

    keep_cols = ["event_uid", region_col]
    rename = {region_col: "region_catalog"}
    if recovered_col is not None:
        keep_cols.append(recovered_col)
        rename[recovered_col] = "recovered_catalog"
    if no_recovery_col is not None:
        keep_cols.append(no_recovery_col)
        rename[no_recovery_col] = "no_recovery_catalog"
    if strict_col is not None:
        keep_cols.append(strict_col)
        rename[strict_col] = "strict_flag_catalog"
    if usable_col is not None:
        keep_cols.append(usable_col)
        rename[usable_col] = "usable_flag_catalog"
    if overlap_col is not None:
        keep_cols.append(overlap_col)
        rename[overlap_col] = "any_overlap_catalog"
    if censor_col is not None:
        keep_cols.append(censor_col)
        rename[censor_col] = "any_censor_catalog"

    keep_cols = list(dict.fromkeys(keep_cols))
    cat2 = (
        cat[keep_cols]
        .rename(columns=rename)
        .dropna(subset=["event_uid"])
        .drop_duplicates("event_uid")
    )

    valid_regions = set(REGION_ORDER)
    cat2["region_catalog"] = cat2["region_catalog"].astype(str).str.strip()
    cat2 = cat2.loc[cat2["region_catalog"].isin(valid_regions)].copy()

    n_proxy_before = len(df)
    n_catalog = len(cat2)

    out0 = df.copy()
    out0["region_internal_proxy"] = out0.get("region")
    out0["recovered_internal_proxy"] = out0.get("recovered")
    out0["no_recovery_internal_proxy"] = out0.get("no_recovery")

    # Inner join: keep the mechanism-available subset of the Result 2 event catalog.
    out = out0.merge(cat2, on="event_uid", how="inner")
    out["catalog_matched"] = 1
    out["sample_source"] = "mechanism_available_subset_of_result2_catalog"

    qa = []
    qa.append({"field": "catalog_path", "matched_n": int(len(out)), "agreement": np.nan, "note": str(path)})
    qa.append({"field": "proxy_events_before_catalog_merge", "matched_n": int(n_proxy_before), "agreement": np.nan, "note": "event-window rows with usable event-end proxies"})
    qa.append({"field": "result2_catalog_events", "matched_n": int(n_catalog), "agreement": np.nan, "note": "valid region rows in Result 2 catalog"})

    region_match = out["region_catalog"].notna()
    if region_match.any():
        agreement = float((out.loc[region_match, "region_internal_proxy"].astype(str) == out.loc[region_match, "region_catalog"].astype(str)).mean())
        qa.append({"field": "region_result2_region4", "matched_n": int(region_match.sum()), "agreement": agreement, "note": f"catalog column={region_col}"})
        out["region"] = out["region_catalog"].astype(str)

    if "recovered_catalog" in out.columns:
        rec_new = _as_binary_int(out["recovered_catalog"])
        mask = rec_new.notna()
        if mask.any():
            old = pd.to_numeric(out.loc[mask, "recovered_internal_proxy"], errors="coerce").astype("Int64")
            qa.append({"field": "recovered_result2", "matched_n": int(mask.sum()), "agreement": float((old.values == rec_new.loc[mask].values).mean()), "note": f"catalog column={recovered_col}"})
            out.loc[mask, "recovered"] = rec_new.loc[mask].astype(int).values
            out.loc[mask, "no_recovery"] = 1 - rec_new.loc[mask].astype(int).values

    if "no_recovery_catalog" in out.columns:
        nr_new = _as_binary_int(out["no_recovery_catalog"])
        mask = nr_new.notna()
        if mask.any():
            old = pd.to_numeric(out.loc[mask, "no_recovery_internal_proxy"], errors="coerce").astype("Int64")
            qa.append({"field": "no_recovery_result2", "matched_n": int(mask.sum()), "agreement": float((old.values == nr_new.loc[mask].values).mean()), "note": f"catalog column={no_recovery_col}"})
            out.loc[mask, "no_recovery"] = nr_new.loc[mask].astype(int).values
            out.loc[mask, "recovered"] = 1 - nr_new.loc[mask].astype(int).values

    if "strict_flag_catalog" in out.columns:
        out["result2_strict_flag"] = _as_binary_int(out["strict_flag_catalog"])
    if "usable_flag_catalog" in out.columns:
        out["result2_usable_flag"] = _as_binary_int(out["usable_flag_catalog"])
    if "any_overlap_catalog" in out.columns:
        out["result2_any_overlap"] = _as_binary_int(out["any_overlap_catalog"])
    if "any_censor_catalog" in out.columns:
        out["result2_any_censor"] = _as_binary_int(out["any_censor_catalog"])

    drop_cols = [
        "region_catalog", "recovered_catalog", "no_recovery_catalog",
        "strict_flag_catalog", "usable_flag_catalog", "any_overlap_catalog", "any_censor_catalog",
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")

    pd.DataFrame(qa).to_csv(cfg.table_dir / "external_catalog_label_agreement.csv", index=False, encoding="utf-8-sig")
    print("[INFO] Main event catalog      : applied as authoritative Result 2 labels")
    print(f"[INFO] Result 2 catalog path   : {path}")
    print(f"[INFO] Proxy events before merge: {n_proxy_before:,}")
    print(f"[INFO] Mechanism-available Result 2 events: {len(out):,}")
    print("[INFO] Label agreement table   : external_catalog_label_agreement.csv")
    return out


def choose_analysis_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Choose the mechanism-available subset following the Result 2 sample rule.

    When a Result 2 catalog has been applied, use its ``strict_flag`` and
    ``usable_flag`` columns. This keeps Result 3 proxy diagnostics aligned with
    the Result 2 trend/standardization analysis. If the Result 2 flags are not
    available, fall back to all mechanism-available events rather than imposing a
    different proxy-specific overlap/censor rule.
    """
    if "result2_strict_flag" in df.columns or "result2_usable_flag" in df.columns:
        strict_ser = df["result2_strict_flag"] if "result2_strict_flag" in df.columns else pd.Series(0, index=df.index)
        usable_ser = df["result2_usable_flag"] if "result2_usable_flag" in df.columns else pd.Series(0, index=df.index)
        strict = df.loc[pd.to_numeric(strict_ser, errors="coerce").fillna(0).astype(int) == 1].copy()
        usable = df.loc[pd.to_numeric(usable_ser, errors="coerce").fillna(0).astype(int) == 1].copy()
        fallback = df.copy()
        print(f"[INFO] Result2 strict candidate n  : {len(strict):,}")
        print(f"[INFO] Result2 usable candidate n  : {len(usable):,}")
        print(f"[INFO] Mechanism-available fallback: {len(fallback):,}")
        if len(strict) >= 500:
            out, rule = strict, "result2_strict_mechanism_available"
        elif len(usable) >= 500:
            out, rule = usable, "result2_usable_mechanism_available"
        else:
            out, rule = fallback, "result2_all_events_fallback_mechanism_available"
        out = out.copy()
        out["analysis_sample_rule"] = rule
        print(f"[INFO] Sample rule            : {rule} | n={len(out):,}")
        return out

    out = df.copy()
    out["analysis_sample_rule"] = "all_mechanism_available_no_result2_flags"
    print(f"[INFO] Sample rule            : all_mechanism_available_no_result2_flags | n={len(out):,}")
    return out
def construct_proxies(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    out = df.copy()
    out["z_Z500_end"] = z_standardize(out["Z500_end"])
    out["z_T850_end"] = z_standardize(out["T850_end"])
    out["z_W500_end"] = z_standardize(out["W500_end"])
    out["z_RH_end"] = z_standardize(out["RH_end"])
    out["z_MC_end"] = z_standardize(out["MC_end"])
    out["z_soil_end"] = z_standardize(out["soil_moist_end"])
    out["z_Bowen_end"] = z_standardize(out["Bowen_ratio_end"])

    # If W500 is ERA5 omega, positive values denote downward motion, so +z_W500
    # increases the ridging--subsidence proxy. If the CSV has already converted
    # omega to ascent, set cfg.w500_positive_is_subsidence=False and the sign flips.
    w500_component = out["z_W500_end"] if cfg.w500_positive_is_subsidence else -out["z_W500_end"]
    out["RSI"] = (out["z_Z500_end"] + out["z_T850_end"] + w500_component) / 3.0
    out["MSDI"] = (-out["z_RH_end"] - out["z_MC_end"]) / 2.0
    out["LandProxy"] = (-out["z_soil_end"] + out["z_Bowen_end"]) / 2.0
    out["proxy_tau_window"] = cfg.tau_tag()
    out["w500_positive_is_subsidence"] = int(cfg.w500_positive_is_subsidence)
    return out


def build_annual(df: pd.DataFrame, region: Optional[str] = None) -> pd.DataFrame:
    sub = df.copy()
    if region is not None and region != "All":
        sub = sub.loc[sub["region"] == region].copy()
    grp = sub.groupby("year")
    annual = grp.agg(
        n=("event_uid", "count"),
        no_recovery=("no_recovery", "mean"),
        recovered=("recovered", "mean"),
        mean_rain_fraction_w10=("mean_rain_fraction_w10", "mean"),
        first_recovery_lag=("first_recovery_lag", lambda x: np.nanmedian(pd.to_numeric(x, errors="coerce"))),
        RSI=("RSI", "mean"),
        MSDI=("MSDI", "mean"),
        LandProxy=("LandProxy", "mean"),
    ).reset_index()
    annual["no_recovery_pct"] = annual["no_recovery"] * 100.0
    annual["recovery_pct"] = annual["recovered"] * 100.0
    return annual

def build_rolling(df: pd.DataFrame, region: str, cfg: CFG) -> pd.DataFrame:
    sub = df if region == "All" else df.loc[df["region"] == region].copy()
    years = np.arange(cfg.years[0], cfg.years[1] + 1)
    half = cfg.rolling_window_years // 2
    rows = []
    for yc in years:
        lo, hi = yc - half, yc + half
        if cfg.require_complete_rolling_window and (lo < cfg.years[0] or hi > cfg.years[1]):
            continue
        tmp = sub.loc[(sub["year"] >= lo) & (sub["year"] <= hi)].copy()
        if len(tmp) < cfg.min_obs_per_year:
            continue
        rows.append({
            "region": region,
            "year_center": yc,
            "window_start": lo,
            "window_end": hi,
            "n": len(tmp),
            "no_recovery_pct": tmp["no_recovery"].mean() * 100.0,
            "mean_rain_fraction_w10": tmp["mean_rain_fraction_w10"].mean(),
            "RSI": tmp["RSI"].mean(),
            "MSDI": tmp["MSDI"].mean(),
            "LandProxy": tmp["LandProxy"].mean(),
            "se_no_recovery_pct": se(tmp["no_recovery"] * 100.0),
            "se_mean_rain_fraction_w10": se(tmp["mean_rain_fraction_w10"]),
            "se_RSI": se(tmp["RSI"]),
            "se_MSDI": se(tmp["MSDI"]),
            "se_LandProxy": se(tmp["LandProxy"]),
        })
    return pd.DataFrame(rows)


def bootstrap_trend(df: pd.DataFrame, value_col: str, n_boot: int = 500, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    years = np.sort(pd.Series(df["year"]).dropna().unique())
    point = slope_pp_decade(df["year"].values.astype(float), df[value_col].values.astype(float))
    boots = []
    for _ in range(n_boot):
        ys = rng.choice(years, size=len(years), replace=True)
        tmp = pd.concat([df.loc[df["year"] == y] for y in ys], axis=0, ignore_index=True)
        boots.append(slope_pp_decade(tmp["year"].values.astype(float), tmp[value_col].values.astype(float)))
    boots = np.asarray(boots, dtype=float)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)



def _z_stats_from_event_summary(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    arr = _np1(df[col]) if col in df.columns else np.asarray([], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan, np.nan
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))


def _z_apply(values, mu: float, sd: float) -> np.ndarray:
    arr = _np1(values)
    if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0:
        return np.full(arr.shape, np.nan, dtype=float)
    return (arr - mu) / sd


def _trend_by_grid_cell(cell_year: pd.DataFrame, metric: str, min_years: int = 25, min_obs: int = 30) -> pd.DataFrame:
    """Estimate per-grid-cell linear trends and two-sided p values.

    Trend units are z per decade. The p value is used only for map stippling; the
    colour field remains the estimated trend. A minimum number of years and a
    minimum total event-grid observations are enforced to avoid noisy texture maps.
    """
    rows: List[Dict] = []
    out_cols = ["lon", "lat", "xbin", "ybin", "n_years", "n_obs", "trend", "p_value", "sig_p05"]
    if cell_year.empty or metric not in cell_year.columns:
        return pd.DataFrame(columns=out_cols)

    for (lon, lat), g in cell_year.groupby(["lon", "lat"], sort=False):
        g = g.loc[np.isfinite(_np1(g[metric])) & np.isfinite(_np1(g["year"]))].copy()
        if g.empty:
            continue
        annual = g.groupby("year", as_index=False).agg(value=(metric, "mean"), n_obs=("n_obs", "sum"))
        annual = annual.loc[np.isfinite(_np1(annual["year"])) & np.isfinite(_np1(annual["value"]))].copy()
        n_years = int(annual["year"].nunique())
        n_obs_total = int(pd.to_numeric(annual["n_obs"], errors="coerce").fillna(0).sum())
        if n_years < min_years or n_obs_total < min_obs:
            continue

        x = _np1(annual["year"])
        y = _np1(annual["value"])
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_years or np.unique(x[mask]).size < min_years:
            continue
        try:
            lr = linregress(x[mask], y[mask])
            tr = float(lr.slope * 10.0)
            pval = float(lr.pvalue)
        except Exception:
            tr = slope_pp_decade(x, y)
            pval = np.nan

        if np.isfinite(tr):
            rows.append({
                "lon": float(lon),
                "lat": float(lat),
                "xbin": float(lon),
                "ybin": float(lat),
                "n_years": n_years,
                "n_obs": n_obs_total,
                "trend": tr,
                "p_value": pval,
                "sig_p05": int(np.isfinite(pval) and pval < 0.05),
            })
    return pd.DataFrame(rows, columns=out_cols)


def build_gridcell_event_end_trend_maps(event_df: pd.DataFrame, cfg: CFG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build native grid-cell event-end proxy trends from event-window CSV rows.

    The map is a supplementary spatial-context diagnostic. It uses the same proxy
    window and W500 sign convention as the main event-level RSI/MSDI calculation,
    and it requires sufficient year/event support per grid cell to reduce local
    trend noise.
    """
    tag = f"{cfg.cache_tag()}_minY{cfg.spatial_min_years}_minObs{cfg.spatial_min_obs_per_cell}"
    trend_cache = cfg.cache_dir / f"gridcell_event_end_RSI_MSDI_trends_{tag}.csv"
    cell_year_cache = cfg.cache_dir / f"gridcell_event_end_RSI_MSDI_cell_year_{cfg.cache_tag()}.csv"
    if trend_cache.exists():
        trend = pd.read_csv(trend_cache)
        if {"metric", "lon", "lat", "trend", "p_value"}.issubset(trend.columns):
            return (
                trend.loc[trend["metric"] == "RSI"].drop(columns=["metric"], errors="ignore").copy(),
                trend.loc[trend["metric"] == "MSDI"].drop(columns=["metric"], errors="ignore").copy(),
            )
        print("[INFO] Existing grid-cell trend cache lacks p values; rebuilding significance-aware trends.")

    if cell_year_cache.exists():
        try:
            cell_year = pd.read_csv(cell_year_cache)
            if {"lon", "lat", "year", "RSI", "MSDI", "n_obs"}.issubset(cell_year.columns):
                grid_rsi = _trend_by_grid_cell(cell_year, "RSI", min_years=cfg.spatial_min_years, min_obs=cfg.spatial_min_obs_per_cell)
                grid_msdi = _trend_by_grid_cell(cell_year, "MSDI", min_years=cfg.spatial_min_years, min_obs=cfg.spatial_min_obs_per_cell)
                trend = pd.concat([
                    grid_rsi.assign(metric="RSI"),
                    grid_msdi.assign(metric="MSDI"),
                ], ignore_index=True)
                trend.to_csv(trend_cache, index=False, encoding="utf-8-sig")
                print(f"[INFO] Rebuilt grid-cell trend significance from cell-year cache: RSI cells = {len(grid_rsi):,}, MSDI cells = {len(grid_msdi):,}")
                return grid_rsi, grid_msdi
        except Exception as e:
            print(f"[WARN] Failed to rebuild trends from cell-year cache: {e}")

    files = list_event_files(cfg.data_root, cfg.years[0], cfg.years[1])
    if not files:
        print("[WARN] No raw event-window files found for grid-cell spatial trends; falling back to centroid trends.")
        return aggregate_grid_metric(event_df, "RSI", cfg), aggregate_grid_metric(event_df, "MSDI", cfg)

    mu_z500, sd_z500 = _z_stats_from_event_summary(event_df, "Z500_end")
    mu_t850, sd_t850 = _z_stats_from_event_summary(event_df, "T850_end")
    mu_w500, sd_w500 = _z_stats_from_event_summary(event_df, "W500_end")
    mu_rh, sd_rh = _z_stats_from_event_summary(event_df, "RH_end")
    mu_mc, sd_mc = _z_stats_from_event_summary(event_df, "MC_end")

    needed = {
        "longitude", "latitude", "year", "lag_day_event",
        "geopotential_500hPa", "temperature_850hPa_mean", "vertical_velocity_500hPa",
        "relative_humidity", "vertically_integrated_moisture_divergence_mean",
    }

    accum: Dict[Tuple[float, float, int], List[float]] = {}
    tau0, tau1 = cfg.event_end_taus

    for i, fp in enumerate(files, start=1):
        try:
            raw = pd.read_csv(fp, usecols=lambda c: c in needed, low_memory=False)
        except Exception:
            continue
        if not needed.issubset(set(raw.columns)):
            continue
        lag = pd.to_numeric(raw["lag_day_event"], errors="coerce")
        sub = raw.loc[(lag >= tau0) & (lag <= tau1)].copy()
        if sub.empty:
            continue
        for col in needed:
            if col in sub.columns:
                sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna(subset=["longitude", "latitude", "year"])
        if sub.empty:
            continue
        sub["lon"] = sub["longitude"].round(4)
        sub["lat"] = sub["latitude"].round(4)
        sub["MC"] = -sub["vertically_integrated_moisture_divergence_mean"]
        g = sub.groupby(["year", "lon", "lat"], as_index=False).agg(
            Z500=("geopotential_500hPa", "mean"),
            T850=("temperature_850hPa_mean", "mean"),
            W500=("vertical_velocity_500hPa", "mean"),
            RH=("relative_humidity", "mean"),
            MC=("MC", "mean"),
        )
        if g.empty:
            continue
        z_z500 = _z_apply(g["Z500"], mu_z500, sd_z500)
        z_t850 = _z_apply(g["T850"], mu_t850, sd_t850)
        z_w500 = _z_apply(g["W500"], mu_w500, sd_w500)
        z_rh = _z_apply(g["RH"], mu_rh, sd_rh)
        z_mc = _z_apply(g["MC"], mu_mc, sd_mc)
        w500_component = z_w500 if cfg.w500_positive_is_subsidence else -z_w500
        g["RSI"] = np.nanmean(np.vstack([z_z500, z_t850, w500_component]), axis=0)
        g["MSDI"] = np.nanmean(np.vstack([-z_rh, -z_mc]), axis=0)

        for row in g.itertuples(index=False):
            key = (float(row.lon), float(row.lat), int(row.year))
            if key not in accum:
                accum[key] = [0.0, 0, 0.0, 0]
            if np.isfinite(row.RSI):
                accum[key][0] += float(row.RSI)
                accum[key][1] += 1
            if np.isfinite(row.MSDI):
                accum[key][2] += float(row.MSDI)
                accum[key][3] += 1

        if i % 500 == 0 or i == len(files):
            print(f"[INFO] Grid-cell spatial trend pass: {i:,}/{len(files):,} event files processed | cell-year keys = {len(accum):,}")

    if not accum:
        print("[WARN] No usable grid-cell event-end rows; falling back to centroid trends.")
        return aggregate_grid_metric(event_df, "RSI", cfg), aggregate_grid_metric(event_df, "MSDI", cfg)

    rows = []
    for (lon, lat, year), (sum_rsi, n_rsi, sum_msdi, n_msdi) in accum.items():
        rows.append({
            "lon": lon,
            "lat": lat,
            "year": year,
            "RSI": sum_rsi / n_rsi if n_rsi > 0 else np.nan,
            "MSDI": sum_msdi / n_msdi if n_msdi > 0 else np.nan,
            "n_obs": max(n_rsi, n_msdi),
        })
    cell_year = pd.DataFrame(rows)
    cell_year.to_csv(cell_year_cache, index=False, encoding="utf-8-sig")

    grid_rsi = _trend_by_grid_cell(cell_year, "RSI", min_years=cfg.spatial_min_years, min_obs=cfg.spatial_min_obs_per_cell)
    grid_msdi = _trend_by_grid_cell(cell_year, "MSDI", min_years=cfg.spatial_min_years, min_obs=cfg.spatial_min_obs_per_cell)
    trend = pd.concat([
        grid_rsi.assign(metric="RSI"),
        grid_msdi.assign(metric="MSDI"),
    ], ignore_index=True)
    trend.to_csv(trend_cache, index=False, encoding="utf-8-sig")
    print(f"[INFO] Grid-cell spatial trends built: RSI cells = {len(grid_rsi):,}, MSDI cells = {len(grid_msdi):,}")
    return grid_rsi, grid_msdi


def aggregate_grid_metric(df: pd.DataFrame, value_col: str, cfg: CFG) -> pd.DataFrame:
    """Adaptive event-centroid trend grid.

    The previous version returned an empty map when each 0.75-degree bin had too few
    events. This version keeps the native target resolution when possible and then
    progressively relaxes the grid resolution / minimum sample threshold until a
    finite trend field exists.
    """
    base = df.loc[
        np.isfinite(_np1(df["centroid_lon"])) &
        np.isfinite(_np1(df["centroid_lat"])) &
        np.isfinite(_np1(df[value_col])) &
        np.isfinite(_np1(df["year"]))
    ].copy()
    if base.empty:
        return pd.DataFrame(columns=["xbin", "ybin", "n", "trend", "map_res_deg", "min_n"])

    # Start at the native ERA5-like 0.25-degree (~25 km) bin used for the event
    # centroids. Coarser fallbacks are retained only to avoid an empty diagnostic
    # when too few event centroids fall in a native-resolution bin.
    attempts = [
        (cfg.map_res_deg, 3),
        (0.50, 4),
        (0.75, 5),
        (1.00, 5),
        (1.50, 4),
        (2.00, 3),
    ]
    xmin, xmax, ymin, ymax = cfg.conus_extent
    for res, min_n in attempts:
        tmp = base.copy()
        tmp["xbin"] = (np.floor((tmp["centroid_lon"] - xmin) / res) * res + xmin + res / 2.0).round(3)
        tmp["ybin"] = (np.floor((tmp["centroid_lat"] - ymin) / res) * res + ymin + res / 2.0).round(3)
        rows = []
        for (xb, yb), g in tmp.groupby(["xbin", "ybin"]):
            if len(g) < 3:
                continue
            tr = slope_pp_decade(_np1(g["year"]), _np1(g[value_col]))
            if np.isfinite(tr):
                rows.append({"xbin": xb, "ybin": yb, "n": len(g), "trend": tr,
                             "map_res_deg": res, "min_n": min_n})
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.loc[out["n"] >= min_n].copy()
            if not out.empty and np.isfinite(out["trend"]).any():
                return out

    # Last-resort region-centroid trend map: still non-empty and makes the diagnostic visible.
    rows = []
    for reg, g in base.groupby("region"):
        if len(g) < 3:
            continue
        tr = slope_pp_decade(_np1(g["year"]), _np1(g[value_col]))
        if np.isfinite(tr):
            rows.append({
                "xbin": float(np.nanmean(g["centroid_lon"])),
                "ybin": float(np.nanmean(g["centroid_lat"])),
                "n": len(g),
                "trend": tr,
                "map_res_deg": np.nan,
                "min_n": 3,
                "region": reg,
            })
    return pd.DataFrame(rows)

def _format_lonlat_axes(ax, cfg: CFG):
    """Add publication-style longitude / latitude ticks to both spatial panels."""
    tick_lons = [-120, -110, -100, -90, -80, -70]
    tick_lats = [25, 30, 35, 40, 45, 50]
    if CARTOPY_OK:
        try:
            ax.set_xticks(tick_lons, crs=ccrs.PlateCarree())
            ax.set_yticks(tick_lats, crs=ccrs.PlateCarree())
            if LongitudeFormatter is not None:
                ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
            if LatitudeFormatter is not None:
                ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
        except Exception:
            pass
    else:
        ax.set_xticks(tick_lons)
        ax.set_yticks(tick_lats)
    ax.tick_params(axis="both", labelsize=14, length=3.5, width=0.9)
    # Keep longitude/latitude tick labels, but remove axis-title text.
    ax.set_xlabel("")
    ax.set_ylabel("")


def _centers_to_edges(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = np.unique(np.sort(vals))
    if len(vals) == 0:
        return vals
    if len(vals) == 1:
        d = 0.25
        return np.asarray([vals[0] - d / 2.0, vals[0] + d / 2.0])
    mids = (vals[:-1] + vals[1:]) / 2.0
    first = vals[0] - (mids[0] - vals[0])
    last = vals[-1] + (vals[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def plot_conus_map(ax, grid: pd.DataFrame, title: str, cfg: CFG, vmin=None, vmax=None, cmap="RdBu_r"):
    """Plot grid-cell spatial trends.

    Preferred input columns are lon/lat/trend from the grid-cell event-end trend
    table. Legacy xbin/ybin centroid trends are still accepted as a fallback.
    """
    ax.set_title(title, pad=10, fontsize=22)
    if grid is None or grid.empty or "trend" not in grid.columns or not np.isfinite(_np1(grid["trend"])).any():
        ax.text(0.5, 0.5, "No finite trend field", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(cfg.conus_extent[0], cfg.conus_extent[1])
        ax.set_ylim(cfg.conus_extent[2], cfg.conus_extent[3])
        _format_lonlat_axes(ax, cfg)
        _clean_axis_spines(ax, full_box=True, lw=0.9)
        return None

    if vmin is None or vmax is None:
        q = float(np.nanpercentile(np.abs(_np1(grid["trend"])), 95))
        q = max(q, 0.01)
        vmin, vmax = -q, q

    xcol = "lon" if "lon" in grid.columns else "xbin"
    ycol = "lat" if "lat" in grid.columns else "ybin"
    plot_grid = grid.loc[
        np.isfinite(_np1(grid[xcol])) &
        np.isfinite(_np1(grid[ycol])) &
        np.isfinite(_np1(grid["trend"]))
    ].copy()

    if CARTOPY_OK:
        ax.set_extent(cfg.conus_extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f7f7f7", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.60, zorder=4)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.36, edgecolor="#666666", zorder=4)
        try:
            states = cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces_lakes",
                scale="50m",
                facecolor="none",
            )
            ax.add_feature(states, linewidth=0.18, edgecolor="0.70", zorder=3)
        except Exception:
            pass
        ax.gridlines(draw_labels=False, linewidth=0.25, color="#cccccc", alpha=0.45, linestyle="-")

        xs = np.sort(plot_grid[xcol].unique())
        ys = np.sort(plot_grid[ycol].unique())
        if len(xs) >= 2 and len(ys) >= 2 and len(plot_grid) >= 20:
            zz = np.full((len(ys), len(xs)), np.nan, dtype=float)
            x_index = {x: j for j, x in enumerate(xs)}
            y_index = {y: i for i, y in enumerate(ys)}
            for r in plot_grid.itertuples(index=False):
                x = getattr(r, xcol)
                y = getattr(r, ycol)
                zz[y_index[y], x_index[x]] = float(getattr(r, "trend"))
            m = ax.pcolormesh(
                _centers_to_edges(xs), _centers_to_edges(ys), zz,
                cmap=cmap, vmin=vmin, vmax=vmax,
                shading="auto", transform=ccrs.PlateCarree(), zorder=2,
            )
        else:
            m = ax.scatter(
                plot_grid[xcol], plot_grid[ycol], c=plot_grid["trend"], cmap=cmap,
                vmin=vmin, vmax=vmax, s=16, marker="s", edgecolor="none",
                transform=ccrs.PlateCarree(), zorder=2,
            )
        _format_lonlat_axes(ax, cfg)
        return m

    ax.set_xlim(cfg.conus_extent[0], cfg.conus_extent[1])
    ax.set_ylim(cfg.conus_extent[2], cfg.conus_extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#d9d9d9", lw=0.45, alpha=0.65)
    xs = np.sort(plot_grid[xcol].unique())
    ys = np.sort(plot_grid[ycol].unique())
    if len(xs) >= 2 and len(ys) >= 2 and len(plot_grid) >= 20:
        zz = np.full((len(ys), len(xs)), np.nan, dtype=float)
        x_index = {x: j for j, x in enumerate(xs)}
        y_index = {y: i for i, y in enumerate(ys)}
        for r in plot_grid.itertuples(index=False):
            x = getattr(r, xcol)
            y = getattr(r, ycol)
            zz[y_index[y], x_index[x]] = float(getattr(r, "trend"))
        m = ax.pcolormesh(_centers_to_edges(xs), _centers_to_edges(ys), zz, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", zorder=3)
    else:
        m = ax.scatter(
            _np1(plot_grid[xcol]), _np1(plot_grid[ycol]),
            c=_np1(plot_grid["trend"]), cmap=cmap, vmin=vmin, vmax=vmax,
            s=16, marker="s", edgecolor="none", zorder=3,
        )
    for coords in _FALLBACK_REGION_POLYGONS.values():
        patch = MplPolygon(coords, closed=True, fill=False, edgecolor="0.45", linewidth=0.55, zorder=2)
        ax.add_patch(patch)
    _format_lonlat_axes(ax, cfg)
    _clean_axis_spines(ax, full_box=True, lw=0.9)
    return m

def _sample_for_strip(values: np.ndarray, max_points: int = 420, seed: int = 42) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= max_points:
        return values
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(len(values)), size=max_points, replace=False)
    return values[idx]

def _advanced_outcome_distribution(
    ax,
    df: pd.DataFrame,
    target: str,
    value_col: str,
    ylabel: str,
    panel_title: str,
    seed: int = 42,
):
    """Nature-style distribution panel: violin envelope + compact box + jittered events."""
    rec = df.loc[(df["region"] == target) & (df["recovered"] == 1), value_col].dropna().values
    unr = df.loc[(df["region"] == target) & (df["recovered"] == 0), value_col].dropna().values
    data = [rec, unr]
    positions = [1.0, 2.0]
    colors = [PHYS_COLORS["Recovered"], PHYS_COLORS["Unrecovered"]]
    labels = ["Recovered", "Unrecovered"]

    # Distribution envelope.
    finite_data = [d[np.isfinite(d)] for d in data]
    if all(len(d) >= 3 for d in finite_data):
        parts = ax.violinplot(
            finite_data,
            positions=positions,
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.18)
            body.set_linewidth(1.0)

    # Compact boxplot inside the violin.
    bp = ax.boxplot(
        finite_data,
        positions=positions,
        widths=0.36,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
    )
    for box, color in zip(bp["boxes"], colors):
        box.set(facecolor=color, alpha=0.34, edgecolor=color, linewidth=1.45)
    for k in ("whiskers", "caps"):
        for obj in bp[k]:
            obj.set(color="#222222", linewidth=1.0)
    for med in bp["medians"]:
        med.set(color="#111111", linewidth=2.0)

    # Event-level points, lightly jittered and capped to avoid overplotting.
    rng = np.random.default_rng(seed)
    for x, vals, color in zip(positions, finite_data, colors):
        vals_s = _sample_for_strip(vals, max_points=420, seed=seed + int(x * 10))
        if len(vals_s):
            jitter = rng.normal(0, 0.055, size=len(vals_s))
            ax.scatter(
                np.full(len(vals_s), x) + jitter,
                vals_s,
                s=16,
                facecolor=color,
                edgecolor="white",
                linewidth=0.25,
                alpha=0.30,
                zorder=2,
            )

    if len(rec) > 3 and len(unr) > 3:
        p = ttest_ind(rec, unr, equal_var=False, nan_policy="omit").pvalue
        # Keep the statistical annotation but move it away from the distribution
        # body; remove the sample-size text from the panel.
        ax.text(0.97, 0.06, "P < 0.001", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=17)

    ax.axhline(0, color="0.82", lw=0.9, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_title(panel_title, pad=8)
    ax.grid(axis="y", color="0.88", lw=0.6, alpha=0.85)
    _clean_axis_spines(ax, full_box=False, lw=1.0)

def plot_figure_rsi_msdi_main(df: pd.DataFrame, cfg: CFG):
    target = cfg.target_region
    all_roll = build_rolling(df, "All", cfg)
    tar_roll = build_rolling(df, target, cfg)

    fig = plt.figure(figsize=(17.6, 10.6), facecolor="white")
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.00, 1.08],
        width_ratios=[1.06, 1.00],
        hspace=0.36,
        wspace=0.46,
    )

    axa = fig.add_subplot(gs[0, 0])
    _advanced_outcome_distribution(
        axa, df, target, "RSI", "RSI (z)",
        f"{target}: event-end ridging–subsidence index",
        seed=cfg.seed,
    )
    add_panel_label(axa, "a", x=-0.16, y=1.14)

    axb = fig.add_subplot(gs[0, 1])
    _advanced_outcome_distribution(
        axb, df, target, "MSDI", "MSDI (z)",
        f"{target}: event-end moisture-support deficit",
        seed=cfg.seed + 1,
    )
    add_panel_label(axb, "b", x=-0.16, y=1.14)

    axc = fig.add_subplot(gs[1, 0])
    for reg, col, alpha in [("All", REGION_COLORS["All"], 0.18), (target, REGION_COLORS[target], 0.15)]:
        rr = all_roll if reg == "All" else tar_roll
        if rr.empty:
            continue
        x = _np1(rr["year_center"])
        y = _np1(rr["no_recovery_pct"])
        se_y = _np1(rr["se_no_recovery_pct"])
        axc.plot(x, y, color=col, lw=2.6, label=f"{reg} no-rec.")
        axc.fill_between(x, y - 1.96 * se_y, y + 1.96 * se_y, color=col, alpha=alpha, lw=0)
    axc.set_ylabel("No-rec. (%)", labelpad=8)
    axc.set_xlabel("Window center year", labelpad=6)
    axc.grid(axis="y", color="0.88", lw=0.6, alpha=0.85)
    _clean_axis_spines(axc, full_box=False, lw=1.0)

    axc2 = axc.twinx()
    for reg, col, ls, alpha in [("All", "#16a085", "--", 0.12), (target, "#d35400", "--", 0.12)]:
        rr = all_roll if reg == "All" else tar_roll
        if rr.empty:
            continue
        x = _np1(rr["year_center"])
        y = _np1(rr["mean_rain_fraction_w10"])
        se_y = _np1(rr["se_mean_rain_fraction_w10"])
        axc2.plot(x, y, color=col, lw=2.4, ls=ls, label=f"{reg} rain frac.")
        axc2.fill_between(x, y - 1.96 * se_y, y + 1.96 * se_y, color=col, alpha=alpha, lw=0)
    axc2.set_ylabel("Rain frac. (%)", labelpad=18)
    axc2.yaxis.set_label_coords(1.11, 0.5)
    axc2.spines["right"].set_visible(True)
    axc2.spines["right"].set_linewidth(1.0)
    axc2.spines["right"].set_color("#333333")
    axc2.spines["top"].set_visible(False)
    axc2.grid(False)

    lines = axc.get_lines() + axc2.get_lines()
    axc.legend(lines, [l.get_label() for l in lines], loc="upper left", ncol=2,
               frameon=False, handlelength=2.2, columnspacing=1.0)
    add_panel_label(axc, "c", x=-0.12, y=1.08)

    axd = fig.add_subplot(gs[1, 1])
    rr = tar_roll.copy()
    if not rr.empty:
        sc = axd.scatter(
            rr["LandProxy"], rr["RSI"],
            c=rr["year_center"], cmap="viridis", s=78,
            edgecolor="white", linewidth=0.9, zorder=3,
        )
        for yr in [int(rr["year_center"].min()), int(rr["year_center"].max())]:
            row = rr.loc[rr["year_center"] == yr]
            if not row.empty:
                axd.text(
                    float(row["LandProxy"].iloc[0]),
                    float(row["RSI"].iloc[0]),
                    str(yr),
                    fontsize=13,
                    weight="bold",
                    path_effects=[pe.withStroke(linewidth=3.2, foreground="white")],
                )
        cbar = fig.colorbar(sc, ax=axd, fraction=0.046, pad=0.045)
        cbar.set_label("Window center year", labelpad=10)
        cbar.ax.tick_params(labelsize=14)
    axd.axhline(0, color="#9b9b9b", lw=0.9)
    axd.axvline(0, color="#9b9b9b", lw=0.9)
    axd.set_xlabel("Land memory (z)", labelpad=7)
    axd.set_ylabel("RSI (z)", labelpad=10)
    axd.yaxis.set_label_coords(-0.18, 0.5)
    axd.set_title(f"{target}: rolling state-space", pad=8)
    axd.grid(color="0.88", lw=0.55, alpha=0.75)
    _clean_axis_spines(axd, full_box=False, lw=1.0)
    add_panel_label(axd, "d", x=-0.14, y=1.08)

    savefig(fig, cfg.fig_dir / "Figure_R3_proxy_main_RSI_MSDI.png", cfg.dpi)

def _proxy_region_panel(ax, df: pd.DataFrame, region: str, cfg: CFG, show_legend: bool = False):
    """Regional proxy panel used by Supp_Fig_R3_proxy_support_RSI_MSDI.

    The old a/b/f panels are intentionally removed. Each regional panel now shows
    only the three event-end state proxies retained for the RSI/MSDI support figure:
    Land memory, MSDI and RSI.
    """
    rr = build_rolling(df, region, cfg)
    color_region = REGION_COLORS.get(region, "#333333")
    if not rr.empty:
        x = _np1(rr["year_center"])
        series = [
            ("Land memory", "LandProxy", PHYS_COLORS["Land"], 2.2),
            ("MSDI", "MSDI", PHYS_COLORS["MSDI"], 2.2),
            ("RSI", "RSI", PHYS_COLORS["RSI"], 2.2),
        ]
        for label, col, color, lw in series:
            y = _np1(rr[col])
            se_col = f"se_{col}"
            ax.plot(x, y, color=color, lw=lw, label=label)
            if se_col in rr.columns:
                sey = _np1(rr[se_col])
                ax.fill_between(x, y - 1.96 * sey, y + 1.96 * sey, color=color, alpha=0.08, lw=0)

    ax.axhline(0, color="0.72", lw=0.8, zorder=0)
    ax.set_title(region, fontsize=21, pad=6, color=color_region)
    ax.set_xlim(cfg.years[0], cfg.years[1])
    vals = []
    if not rr.empty:
        for col in ("LandProxy", "MSDI", "RSI"):
            vals.extend(_np1(rr[col]).tolist())
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals):
        ymin = float(np.nanpercentile(vals, 2))
        ymax = float(np.nanpercentile(vals, 98))
        pad = max(0.08, 0.12 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(axis="y", alpha=0.16, lw=0.6)
    ax.tick_params(axis="both", labelsize=18, length=3)
    ax.set_xlabel("Year", fontsize=21)
    ax.set_ylabel("Proxy (z)", fontsize=21)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color_region)
        spine.set_linewidth(1.1)
    if show_legend:
        # In the composite support figure the legend is placed below the Southwest
        # panel, not inside the Northwest panel.
        ax.legend(
            frameon=False,
            fontsize=14,
            loc="upper left",
            bbox_to_anchor=(0.02, -0.42),
            ncol=1,
            handlelength=1.7,
            columnspacing=0.9,
            borderaxespad=0.0,
        )

def plot_figure_rsi_msdi_support(df: pd.DataFrame, cfg: CFG):
    """Redesigned support figure in the Supplementary Fig. 8 visual grammar.

    Removed old panels a, b and f:
    - old a: no-recovery time series
    - old b: rain-fraction time series
    - old f: recovered/unrecovered Z500 trajectory

    Kept content is reorganized as regional rolling proxy diagnostics around a central
    7-region CONUS map, following the uploaded Supplementary Fig. 8 layout.
    """
    fig = plt.figure(figsize=(17.6, 10.6), facecolor="white")

    map_ax = fig.add_axes(
        [0.33, 0.28, 0.34, 0.42],
        projection=ccrs.PlateCarree() if CARTOPY_OK else None,
    )
    draw_filled_region_map(map_ax, cfg, alpha=0.92, outline=True)

    positions = {
        "Northwest": [0.05, 0.61, 0.24, 0.18],
        "Northern Great Plains": [0.36, 0.79, 0.22, 0.16],
        "Midwest": [0.64, 0.61, 0.24, 0.18],
        "Northeast": [0.76, 0.36, 0.20, 0.18],
        "Southeast": [0.68, 0.08, 0.23, 0.18],
        "Southern Great Plains": [0.35, 0.07, 0.23, 0.18],
        "Southwest": [0.07, 0.315, 0.24, 0.18],
    }
    target_fracs = {
        "Northwest": (0.95, 0.55),
        "Northern Great Plains": (0.50, 0.02),
        "Midwest": (0.05, 0.55),
        "Northeast": (0.00, 0.40),
        "Southeast": (0.30, 1.00),
        "Southern Great Plains": (0.50, 1.00),
        "Southwest": (0.95, 0.38),
    }

    for i, region in enumerate(REGION_ORDER):
        ax = fig.add_axes(positions[region])
        _proxy_region_panel(ax, df, region, cfg, show_legend=(region == "Southwest"))
        # Keep the local x-axis label for Southwest as well; the legend is
        # moved farther below the panel rather than removing the "Year" label.

        x0, y0 = _REGION_CONNECT_POS.get(region, _REGION_LABEL_POS.get(region, (-100, 35)))
        fx, fy = target_fracs[region]
        con = ConnectionPatch(
            xyA=(x0, y0),
            coordsA=_connection_transform(map_ax),
            xyB=(fx, fy),
            coordsB=ax.transAxes,
            color=REGION_COLORS[region],
            lw=1.0,
            alpha=0.95,
            zorder=1,
        )
        fig.add_artist(con)

    savefig(fig, cfg.fig_dir / "Supp_Fig_R3_proxy_support_RSI_MSDI.png", cfg.dpi)

    # Optional individual panels for later layout work; harmless if unused.
    fig_map = plt.figure(figsize=(8.6, 5.8), facecolor="white")
    ax_map = fig_map.add_subplot(111, projection=ccrs.PlateCarree() if CARTOPY_OK else None)
    draw_filled_region_map(ax_map, cfg, alpha=0.92, outline=True)
    savefig(fig_map, cfg.fig_dir / "Supp_Fig_R3_proxy_support_RSI_MSDI_panel_map.png", cfg.dpi)

    for region in REGION_ORDER:
        fig_r = plt.figure(figsize=(8.6, 5.6), facecolor="white")
        ax_r = fig_r.add_subplot(111)
        _proxy_region_panel(ax_r, df, region, cfg, show_legend=True)
        safe_region = region.replace(" ", "_").replace("-", "_")
        savefig(fig_r, cfg.fig_dir / f"Supp_Fig_R3_proxy_support_RSI_MSDI_{safe_region}.png", cfg.dpi)

def _overlay_significance_points(ax, grid: pd.DataFrame, cfg: CFG, p_threshold: float = 0.05):
    """Overlay stippling for grid cells with significant linear trends."""
    if grid is None or grid.empty or "p_value" not in grid.columns:
        return
    xcol = "lon" if "lon" in grid.columns else "xbin"
    ycol = "lat" if "lat" in grid.columns else "ybin"
    sig = grid.loc[
        np.isfinite(_np1(grid[xcol])) &
        np.isfinite(_np1(grid[ycol])) &
        np.isfinite(_np1(grid["p_value"])) &
        (_np1(grid["p_value"]) < p_threshold)
    ].copy()
    if sig.empty:
        return

    kwargs = dict(
        s=7,
        marker="o",
        facecolors="black",
        edgecolors="white",
        linewidths=0.18,
        alpha=0.78,
        zorder=6,
    )
    if CARTOPY_OK:
        ax.scatter(sig[xcol], sig[ycol], transform=ccrs.PlateCarree(), **kwargs)
    else:
        ax.scatter(sig[xcol], sig[ycol], **kwargs)



def _overlay_region_boundaries(ax, cfg: CFG, lw: float = 1.25):
    """Overlay coloured seven-region boundaries on spatial trend maps.

    This uses the same seven-region colour grammar as the support figures. The
    boundaries are drawn without fill, above the colour field and stippling, so
    the spatial trend remains visible while the regional structure is explicit.
    """
    if CARTOPY_OK and SHAPELY_OK:
        geoms = _region_geometries()
        if geoms:
            for region in REGION_ORDER:
                geom = geoms.get(region)
                if geom is None:
                    continue
                _plot_geom_boundary(
                    ax,
                    geom,
                    color=REGION_COLORS.get(region, "#333333"),
                    lw=lw,
                    zorder=9,
                )
            return

    # Fallback: draw the approximate region polygons with the same colours.
    for region in REGION_ORDER:
        coords = _FALLBACK_REGION_POLYGONS.get(region)
        if not coords:
            continue
        arr = np.asarray(coords, dtype=float)
        color = REGION_COLORS.get(region, "#333333")
        if CARTOPY_OK:
            ax.plot(
                arr[:, 0], arr[:, 1],
                color=color,
                lw=lw,
                alpha=0.98,
                transform=ccrs.PlateCarree(),
                zorder=9,
            )
        else:
            patch = MplPolygon(
                arr,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=lw,
                alpha=0.98,
                zorder=9,
            )
            ax.add_patch(patch)

def plot_spatial_trend_maps(df: pd.DataFrame, cfg: CFG):
    # Use grid-cell event-end trends rather than event centroids, so the map
    # represents the native event-window grid instead of isolated centroid points.
    grid_rsi, grid_msdi = build_gridcell_event_end_trend_maps(df, cfg)

    vals = []
    for grid in (grid_rsi, grid_msdi):
        if grid is not None and not grid.empty and "trend" in grid.columns:
            vals.extend(_np1(grid["trend"]).tolist())
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals):
        vmax = float(np.nanpercentile(np.abs(vals), 95))
        vmax = max(vmax, 0.01)
    else:
        vmax = 0.15
    vmin = -vmax

    fig = plt.figure(figsize=(17.8, 6.8), facecolor="white")
    if CARTOPY_OK:
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    else:
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

    m1 = plot_conus_map(ax1, grid_rsi, "Trend in event-end RSI", cfg, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    m2 = plot_conus_map(ax2, grid_msdi, "Trend in event-end MSDI", cfg, vmin=vmin, vmax=vmax, cmap="RdBu_r")

    # Stippling: grid cells with two-sided OLS trend P < 0.05.
    _overlay_significance_points(ax1, grid_rsi, cfg, p_threshold=0.05)
    _overlay_significance_points(ax2, grid_msdi, cfg, p_threshold=0.05)

    # Seven-region coloured boundaries, matching the regional colour grammar used
    # throughout the Result 3 support figures.
    _overlay_region_boundaries(ax1, cfg, lw=1.35)
    _overlay_region_boundaries(ax2, cfg, lw=1.35)

    add_panel_label(ax1, "a", x=-0.06, y=1.15)
    add_panel_label(ax2, "b", x=-0.06, y=1.15)

    fig.subplots_adjust(left=0.055, right=0.895, bottom=0.13, top=0.86, wspace=0.18)

    mappable = m2 if m2 is not None else m1
    if mappable is not None:
        cax = fig.add_axes([0.915, 0.20, 0.018, 0.58])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label("Trend (z decade$^{-1}$)", labelpad=10, fontsize=18)
        cbar.ax.tick_params(labelsize=15)

    fig.text(0.915, 0.155, f"black dots: P < 0.05; min years={cfg.spatial_min_years}, min obs={cfg.spatial_min_obs_per_cell}", ha="left", va="top", fontsize=12)

    savefig(fig, cfg.fig_dir / "Supp_Fig_R3_proxy_spatial_trends.png", cfg.dpi)

def _state_space_region_panel(ax, df: pd.DataFrame, region: str, cfg: CFG):
    """Single regional state-space panel for the central-map state-space supplement."""
    rr = build_rolling(df, region, cfg)
    color_region = REGION_COLORS.get(region, "#333333")

    if rr.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.scatter(
            _np1(rr["LandProxy"]),
            _np1(rr["RSI"]),
            c=_np1(rr["year_center"]),
            cmap="viridis",
            vmin=cfg.years[0],
            vmax=cfg.years[1],
            s=48,
            edgecolor="white",
            linewidth=0.65,
            alpha=0.96,
            zorder=3,
        )
        for yr in [int(rr["year_center"].min()), int(rr["year_center"].max())]:
            row = rr.loc[rr["year_center"] == yr]
            if not row.empty:
                ax.text(
                    float(row["LandProxy"].iloc[0]),
                    float(row["RSI"].iloc[0]),
                    str(yr),
                    fontsize=10.5,
                    weight="bold",
                    ha="left",
                    va="center",
                    path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
                    zorder=4,
                )

        x = _np1(rr["LandProxy"])
        y = _np1(rr["RSI"])
        xm = np.isfinite(x)
        ym = np.isfinite(y)
        if xm.any():
            xlo, xhi = float(np.nanpercentile(x[xm], 2)), float(np.nanpercentile(x[xm], 98))
            xpad = max(0.03, 0.16 * (xhi - xlo))
            ax.set_xlim(xlo - xpad, xhi + xpad)
        if ym.any():
            ylo, yhi = float(np.nanpercentile(y[ym], 2)), float(np.nanpercentile(y[ym], 98))
            ypad = max(0.03, 0.16 * (yhi - ylo))
            ax.set_ylim(ylo - ypad, yhi + ypad)

    ax.axhline(0, color="0.75", lw=0.8, zorder=0)
    ax.axvline(0, color="0.75", lw=0.8, zorder=0)
    ax.grid(color="0.88", lw=0.55, alpha=0.75)
    ax.set_title(region, fontsize=21, pad=6, color=color_region)
    ax.set_xlabel("Land memory (z)", fontsize=18)
    ax.set_ylabel("RSI (z)", fontsize=18)
    ax.tick_params(axis="both", labelsize=15, length=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color_region)
        spine.set_linewidth(1.1)


def plot_region_state_space(df: pd.DataFrame, cfg: CFG):
    """State-space regional supplement in the same visual grammar as the support figure.

    The old 3 x 3 grid is replaced by a central 7-region CONUS map surrounded by
    seven regional state-space panels, so this figure uses the same layout language
    as Supp_Fig_R3_proxy_support_RSI_MSDI.png.
    """
    fig = plt.figure(figsize=(17.6, 10.6), facecolor="white")

    map_ax = fig.add_axes(
        [0.33, 0.28, 0.34, 0.42],
        projection=ccrs.PlateCarree() if CARTOPY_OK else None,
    )
    draw_filled_region_map(map_ax, cfg, alpha=0.92, outline=True)

    positions = {
        "Northwest": [0.05, 0.61, 0.24, 0.18],
        "Northern Great Plains": [0.36, 0.79, 0.22, 0.16],
        "Midwest": [0.64, 0.61, 0.24, 0.18],
        "Northeast": [0.76, 0.36, 0.20, 0.18],
        "Southeast": [0.68, 0.08, 0.23, 0.18],
        "Southern Great Plains": [0.35, 0.07, 0.23, 0.18],
        "Southwest": [0.07, 0.315, 0.24, 0.18],
    }
    target_fracs = {
        "Northwest": (0.95, 0.55),
        "Northern Great Plains": (0.50, 0.02),
        "Midwest": (0.05, 0.55),
        "Northeast": (0.00, 0.40),
        "Southeast": (0.30, 1.00),
        "Southern Great Plains": (0.50, 1.00),
        "Southwest": (0.95, 0.38),
    }

    for region in REGION_ORDER:
        ax = fig.add_axes(positions[region])
        _state_space_region_panel(ax, df, region, cfg)

        x0, y0 = _REGION_CONNECT_POS.get(region, _REGION_LABEL_POS.get(region, (-100, 35)))
        fx, fy = target_fracs[region]
        con = ConnectionPatch(
            xyA=(x0, y0),
            coordsA=_connection_transform(map_ax),
            xyB=(fx, fy),
            coordsB=ax.transAxes,
            color=REGION_COLORS[region],
            lw=1.0,
            alpha=0.95,
            zorder=1,
        )
        fig.add_artist(con)

    # Horizontal colorbar placed below the Southwest panel to avoid overlap with Northeast.
    cax = fig.add_axes([0.050, 0.220, 0.22, 0.018])
    sm = ScalarMappable(norm=Normalize(vmin=cfg.years[0], vmax=cfg.years[1]), cmap="viridis")
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Window center year", labelpad=6, fontsize=15)
    cbar.ax.tick_params(labelsize=13, length=2.8)

    savefig(fig, cfg.fig_dir / "Supp_Fig_R3_proxy_state_space_regions.png", cfg.dpi)


def _cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta for x relative to y using sorted y, no large pair matrix."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    ys = np.sort(y)
    greater = np.searchsorted(ys, x, side="left")
    less = len(ys) - np.searchsorted(ys, x, side="right")
    return float((greater.sum() - less.sum()) / (len(x) * len(y)))


def _bootstrap_median_diff(x: Sequence[float], y: Sequence[float], n_boot: int, seed: int) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float); y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = np.nanmedian(rng.choice(x, size=len(x), replace=True)) - np.nanmedian(rng.choice(y, size=len(y), replace=True))
    point = float(np.nanmedian(x) - np.nanmedian(y))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def proxy_group_effects(df: pd.DataFrame, cfg: CFG, metrics: Sequence[str] = ("RSI", "MSDI", "LandProxy"), suffix: str = "") -> pd.DataFrame:
    rows = []
    for reg in ["All", *REGION_ORDER]:
        sub = df if reg == "All" else df.loc[df["region"] == reg]
        if len(sub) < 10:
            continue
        for metric in metrics:
            if metric not in sub.columns:
                continue
            rec = pd.to_numeric(sub.loc[sub["recovered"] == 1, metric], errors="coerce").dropna().values
            unr = pd.to_numeric(sub.loc[sub["recovered"] == 0, metric], errors="coerce").dropna().values
            if len(rec) < 3 or len(unr) < 3:
                continue
            pval = ttest_ind(unr, rec, equal_var=False, nan_policy="omit").pvalue
            pooled = np.sqrt(((len(unr) - 1) * np.nanvar(unr, ddof=1) + (len(rec) - 1) * np.nanvar(rec, ddof=1)) / max(len(unr) + len(rec) - 2, 1))
            smd = (np.nanmean(unr) - np.nanmean(rec)) / pooled if np.isfinite(pooled) and pooled > 0 else np.nan
            md, md_lo, md_hi = _bootstrap_median_diff(unr, rec, cfg.n_boot, cfg.seed + len(rows) + 1100)
            rows.append({
                "region": reg,
                "metric": metric,
                "suffix": suffix,
                "n_recovered": int(len(rec)),
                "n_unrecovered": int(len(unr)),
                "mean_recovered": float(np.nanmean(rec)),
                "mean_unrecovered": float(np.nanmean(unr)),
                "mean_diff_unrec_minus_rec": float(np.nanmean(unr) - np.nanmean(rec)),
                "median_recovered": float(np.nanmedian(rec)),
                "median_unrecovered": float(np.nanmedian(unr)),
                "median_diff_unrec_minus_rec": md,
                "median_diff_ci_low": md_lo,
                "median_diff_ci_high": md_hi,
                "standardized_mean_difference": float(smd) if np.isfinite(smd) else np.nan,
                "cliffs_delta": _cliffs_delta(unr, rec),
                "welch_p_value": float(pval) if np.isfinite(pval) else np.nan,
            })
    return pd.DataFrame(rows)


def _mean_tau_columns(df: pd.DataFrame, prefix: str, window: Tuple[int, int]) -> pd.Series:
    cols = [f"{prefix}_tau{tau}" for tau in range(int(window[0]), int(window[1]) + 1) if f"{prefix}_tau{tau}" in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def construct_proxies_for_tau_window(df: pd.DataFrame, cfg: CFG, window: Tuple[int, int]) -> pd.DataFrame:
    out = df.copy()
    z_z500 = z_standardize(_mean_tau_columns(out, "Z500", window))
    z_t850 = z_standardize(_mean_tau_columns(out, "T850", window))
    z_w500 = z_standardize(_mean_tau_columns(out, "W500", window))
    z_rh = z_standardize(_mean_tau_columns(out, "RH", window))
    z_mc = z_standardize(_mean_tau_columns(out, "MC", window))
    z_soil = z_standardize(_mean_tau_columns(out, "soil", window))
    z_bowen = z_standardize(_mean_tau_columns(out, "Bowen", window))
    w500_component = z_w500 if cfg.w500_positive_is_subsidence else -z_w500
    out["RSI_sens"] = (z_z500 + z_t850 + w500_component) / 3.0
    out["MSDI_sens"] = (-z_rh - z_mc) / 2.0
    out["LandProxy_sens"] = (-z_soil + z_bowen) / 2.0
    return out


def export_tau_window_sensitivity(df: pd.DataFrame, cfg: CFG):
    rows = []
    for window in cfg.tau_sensitivity_windows:
        tmp = construct_proxies_for_tau_window(df, cfg, window)
        # Overwrite the primary proxy columns for the sensitivity window so that
        # proxy_group_effects sees exactly one column per metric.
        tmp["RSI"] = tmp.pop("RSI_sens")
        tmp["MSDI"] = tmp.pop("MSDI_sens")
        tmp["LandProxy"] = tmp.pop("LandProxy_sens")
        effects = proxy_group_effects(tmp, cfg, metrics=("RSI", "MSDI", "LandProxy"), suffix=cfg.tau_tag(window))
        effects["tau_start"] = int(window[0])
        effects["tau_end"] = int(window[1])
        rows.append(effects)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(cfg.table_dir / "tau_window_proxy_sensitivity_effect_sizes.csv", index=False, encoding="utf-8-sig")


def export_tables(df: pd.DataFrame, cfg: CFG):
    build_annual(df, "All").to_csv(cfg.table_dir / "annual_all.csv", index=False, encoding="utf-8-sig")
    build_annual(df, cfg.target_region).to_csv(cfg.table_dir / f"annual_{cfg.target_region}.csv", index=False, encoding="utf-8-sig")
    df.to_csv(cfg.table_dir / "event_summary_with_RSI_MSDI.csv", index=False, encoding="utf-8-sig")

    # Group effect sizes are exported because P values alone are not sufficient
    # for a submission-grade mechanism claim when distributions overlap.
    proxy_group_effects(df, cfg).to_csv(cfg.table_dir / "proxy_group_effect_sizes.csv", index=False, encoding="utf-8-sig")
    export_tau_window_sensitivity(df, cfg)

    rows = []
    for reg in ["All", "Northwest", "Northern Great Plains", "Midwest", "Northeast", "Southwest", "Southern Great Plains", "Southeast"]:
        sub = df if reg == "All" else df.loc[df["region"] == reg]
        if len(sub) < 10:
            continue
        p1, lo1, hi1 = bootstrap_trend(sub, "no_recovery", n_boot=cfg.n_boot, seed=cfg.seed)
        p2, lo2, hi2 = bootstrap_trend(sub, "RSI", n_boot=cfg.n_boot, seed=cfg.seed + 1)
        p3, lo3, hi3 = bootstrap_trend(sub, "MSDI", n_boot=cfg.n_boot, seed=cfg.seed + 2)
        rows.append({
            "region": reg, "n": len(sub),
            "no_recovery_trend_pp_decade": p1 * 100.0, "no_recovery_ci_low": lo1 * 100.0, "no_recovery_ci_high": hi1 * 100.0,
            "RSI_trend_z_decade": p2, "RSI_ci_low": lo2, "RSI_ci_high": hi2,
            "MSDI_trend_z_decade": p3, "MSDI_ci_low": lo3, "MSDI_ci_high": hi3,
        })
    pd.DataFrame(rows).to_csv(cfg.table_dir / "trend_summary_bootstrap.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "cache_tag": cfg.cache_tag(),
        "event_end_taus_primary": list(cfg.event_end_taus),
        "full_lead_taus_available": list(cfg.full_lead_taus),
        "tau_sensitivity_windows": [list(w) for w in cfg.tau_sensitivity_windows],
        "w500_positive_is_subsidence": cfg.w500_positive_is_subsidence,
        "rolling_window_years": cfg.rolling_window_years,
        "require_complete_rolling_window": cfg.require_complete_rolling_window,
        "spatial_min_years": cfg.spatial_min_years,
        "spatial_min_obs_per_cell": cfg.spatial_min_obs_per_cell,
        "main_event_catalog_path": str(cfg.main_event_catalog_path) if cfg.main_event_catalog_path is not None else None,
        "sample_definition_note": "Result 3 proxy diagnostics use the mechanism-available subset of the Result 2 event catalog; region is Result 2 region4 and recovery is Result 2 recovered_by_day10/no_recovery.",
    }
    (cfg.table_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (cfg.table_dir / "figure_caption_sample_note.txt").write_text(
        "RSI/MSDI diagnostics are based on the mechanism-available subset of the Result 2 event catalog; "
        "region labels use Result 2 region4, and recovery labels use Result 2 recovered_by_day10/no_recovery.\n",
        encoding="utf-8",
    )


def main(cfg: CFG):
    df = build_event_summary(cfg)
    df = apply_external_catalog_overrides(df, cfg)
    df = choose_analysis_sample(df)
    df = construct_proxies(df, cfg)
    print(f"[INFO] Primary proxy window    : {cfg.event_end_taus[0]} to {cfg.event_end_taus[1]}")
    print(f"[INFO] Lead-time archive       : {cfg.full_lead_taus[0]} to {cfg.full_lead_taus[1]}")
    print(f"[INFO] W500 sign convention    : {'omega positive = subsidence' if cfg.w500_positive_is_subsidence else 'positive = ascent; sign flipped in RSI'}")
    print(f"[INFO] Chosen target region   : {cfg.target_region}")
    sub = df.loc[df["region"] == cfg.target_region]
    print(f"[INFO] Target-region events   : {len(sub):,}")
    print(f"[INFO] Target recovered / unrecovered: {(sub['recovered']==1).sum():,} / {(sub['recovered']==0).sum():,}")
    export_tables(df, cfg)
    plot_figure_rsi_msdi_main(df, cfg)
    plot_figure_rsi_msdi_support(df, cfg)
    plot_spatial_trend_maps(df, cfg)
    plot_region_state_space(df, cfg)
    print("[INFO] Done.")
    print(f"[INFO] Figures saved to: {cfg.fig_dir}")
    print(f"[INFO] Tables saved to : {cfg.table_dir}")


if __name__ == "__main__":
    cfg = CFG()
    main(cfg)
