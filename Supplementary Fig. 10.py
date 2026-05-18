# -*- coding: utf-8 -*-
"""
Supp Fig S3 — all-region rolling outcomes in the Supp_Fig_R1_01_spatial_heterogeneity style
=============================================================================================

Purpose
-------
Build a supplementary figure for Result 3 using the SAME visual grammar as
Supp_Fig_R1_01_spatial_heterogeneity from result1_final_rebuild_revised_v2.py:

    - a central CONUS 7-region map
    - 7 surrounding regional line panels
    - connection lines from each regional mini-panel to the central map

This version removes the "All" panel and keeps only the 7 climate regions.

Outcome plotted
---------------
For each event, the continuous outcome is:

    mean rain-return fraction within post-event days 1–10
    using precipitation >= 1 mm day-1 over the original event footprint.

For each region and year, the script computes the annual mean of that event-level
continuous outcome, then plots:
    - thin raw annual line
    - thicker centered rolling mean

Inputs
------
Raw event-window CSV files under:
    ROOT/
      1950/event_1950_00001_window.csv
      1950/event_1950_00002_window.csv
      ...

Expected columns include at least:
    date, longitude, latitude, precipitation, lag_day_event,
    is_heat_period_event, year, event_id

Outputs
-------
ROOT/_figure3_bridge_nw/
    figures/Supp_Fig_S3_all_region_rolling_outcomes.png
    figures/Supp_Fig_S3_all_region_rolling_outcomes_panel_map.png
    figures/Supp_Fig_S3_all_region_rolling_outcomes_<Region>.png
    tables/supp_s3_event_summary.csv
    tables/supp_s3_regional_annual_outcome.csv
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Polygon as MplPolygon

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

try:
    from shapely.geometry import Point, Polygon, MultiPolygon
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False


@dataclass
class Config:
    root_dir: Path = Path(
        r"E:\temp_events_ERA5_S1S6_Nature所有数据版本\events_cc3d_postlag10_NCC_with_pr_ws_rh第三篇的数据_added_CAPE_IVT_T850_added_Z500_W500_added_Bowen_Rn_added_WIND250_WIND850"
    )
    out_dir: Optional[Path] = None

    primary_rain_threshold_mm: float = 1.0
    primary_window_days: int = 10

    min_detectable_coverage: float = 0.80
    usable_min_coverage: float = 0.80
    strict_min_coverage: float = 0.95

    use_cache: bool = True
    progress_every: int = 250
    rolling_window: int = 7
    dpi: int = 300
    figure_bg: str = "white"

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
            self.out_dir = self.root_dir / "_figure3_b图3中的一张"
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
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 18,
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

REGION_COLORS = {
    "Northwest": "#c0392b",
    "Northern Great Plains": "#7f8c8d",
    "Midwest": "#8e44ad",
    "Northeast": "#1f78b4",
    "Southwest": "#d95f02",
    "Southern Great Plains": "#1b9e77",
    "Southeast": "#e6ab02",
}

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
_POINT_REGION_CACHE: Dict[Tuple[float, float], str] = {}


def rolling_mean(y: pd.Series, window: int) -> pd.Series:
    return y.rolling(window=window, center=True, min_periods=max(3, window // 2)).mean()


def savefig(fig: plt.Figure, path: Path, dpi: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=CFG.dpi if dpi is None else dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _safe_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _bool_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


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


def _setup_conus_ax(ax, show_ticks: bool = False, show_outline: bool = False):
    if HAS_CARTOPY:
        ax.set_extent([-125, -66.5, 25, 50], crs=ccrs.PlateCarree())
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
        if show_ticks:
            try:
                ax.set_xticks([-120, -110, -100, -90, -80, -70], crs=ccrs.PlateCarree())
                ax.set_yticks([25, 30, 35, 40, 45, 50], crs=ccrs.PlateCarree())
                ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f'))
                ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f'))
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
        ax.set_xticks([] if not show_ticks else [-120, -110, -100, -90, -80, -70])
        ax.set_yticks([] if not show_ticks else [25, 30, 35, 40, 45, 50])
        if not show_outline:
            for spine in ax.spines.values():
                spine.set_visible(False)


_DATE_COLS = ["date", "event_start", "event_end", "grid_start", "grid_end", "next_grid_start"]
_NUMERIC_COLS = [
    "longitude", "latitude", "year", "event_id", "lag_day_event", "precipitation",
    "is_heat_period_event", "overlaps_next_local_heat", "is_post_event_0_10_censored",
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
    return df


def _event_uid_from_path(path: Path, df: pd.DataFrame) -> str:
    m = _FILE_RE.search(path.name)
    if m:
        return f"{int(m.group(1))}_{int(m.group(2)):05d}"
    year = int(to_num(df.get("year", pd.Series([np.nan]))).dropna().iloc[0]) if "year" in df else -1
    eid = int(to_num(df.get("event_id", pd.Series([np.nan]))).dropna().iloc[0]) if "event_id" in df else -1
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
            pr = to_num(df["precipitation"]).dropna()
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


def summarize_one_event(path: Path, precip_scale: float, cfg: Config) -> Optional[Dict]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        warnings.warn(f"Failed to read {path.name}: {e}")
        return None

    df = _clean_columns(df)
    if "date" not in df.columns or "precipitation" not in df.columns:
        return None

    for col in _DATE_COLS:
        if col in df.columns:
            df[col] = _safe_dt(df[col])
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = to_num(df[col])

    if "coord_key" not in df.columns:
        if {"longitude", "latitude"}.issubset(df.columns):
            df["coord_key"] = df["longitude"].round(4).astype(str) + "_" + df["latitude"].round(4).astype(str)
        else:
            return None

    df["precip_mm"] = to_num(df["precipitation"]) * precip_scale
    lag = to_num(df.get("lag_day_event", pd.Series(np.nan, index=df.index)))
    is_heat = _bool_num(df.get("is_heat_period_event", pd.Series(np.nan, index=df.index)))

    heat = df.loc[is_heat == 1].copy()
    if heat.empty:
        heat = df.loc[lag <= 0].copy()
    if heat.empty:
        return None

    heat_coords = heat[["coord_key", "longitude", "latitude"]].dropna().drop_duplicates("coord_key")
    footprint = set(heat_coords["coord_key"].astype(str))
    footprint_ncells = int(len(footprint))
    if footprint_ncells == 0:
        return None

    uid = _event_uid_from_path(path, df)
    year_val = int(to_num(df.get("year", pd.Series([heat["date"].dt.year.mode().iloc[0]])).dropna()).iloc[0])
    event_start = heat["date"].min()
    event_end = heat["date"].max()
    if "event_start" in df.columns and df["event_start"].notna().any():
        event_start = df["event_start"].dropna().iloc[0]
    if "event_end" in df.columns and df["event_end"].notna().any():
        event_end = df["event_end"].dropna().iloc[0]
    duration = int((event_end - event_start).days + 1) if pd.notna(event_start) and pd.notna(event_end) else int(heat["date"].nunique())
    centroid_lon = float(heat_coords["longitude"].mean())
    centroid_lat = float(heat_coords["latitude"].mean())
    climate_region = assign_climate_region(centroid_lon, centroid_lat)

    post = df.loc[(lag >= 1) & (lag <= cfg.primary_window_days)].copy()
    daily_rows: List[Dict] = []
    for lag_day in range(1, cfg.primary_window_days + 1):
        g = post.loc[to_num(post.get("lag_day_event", pd.Series(np.nan, index=post.index))) == lag_day].copy()
        g2 = g.dropna(subset=["coord_key"]).drop_duplicates("coord_key")
        g2 = g2.loc[g2["coord_key"].astype(str).isin(footprint)].copy()
        n_present = int(g2["coord_key"].nunique())
        coverage = n_present / footprint_ncells if footprint_ncells > 0 else np.nan
        rainy_cells = int(g2.loc[to_num(g2["precip_mm"]) >= cfg.primary_rain_threshold_mm, "coord_key"].nunique())
        rainfrac = rainy_cells / footprint_ncells if footprint_ncells > 0 else np.nan
        daily_rows.append({"lag_day": lag_day, "coverage": coverage, "rainfrac": rainfrac})

    daily = pd.DataFrame(daily_rows)
    any_recovery = bool(((daily["coverage"] >= cfg.min_detectable_coverage) & (daily["rainfrac"] >= 0.25)).any())
    recovery_day = int(daily.loc[(daily["coverage"] >= cfg.min_detectable_coverage) & (daily["rainfrac"] >= 0.25), "lag_day"].min()) if any_recovery else np.nan

    has_all_lags = bool(set(range(1, cfg.primary_window_days + 1)).issubset(set(daily["lag_day"].tolist())))
    min_cov = float(daily["coverage"].min()) if not daily.empty else np.nan

    summary = {
        "event_uid": uid,
        "source_file": str(path),
        "year": year_val,
        "event_start": event_start,
        "event_end": event_end,
        "duration": duration,
        "centroid_lon": centroid_lon,
        "centroid_lat": centroid_lat,
        "climate_region": climate_region,
        "footprint_ncells": footprint_ncells,
        "recovered_by_day10": int(any_recovery),
        "first_recovery_lag": recovery_day,
        "has_all_lags": int(has_all_lags),
        "min_coverage": min_cov,
        "usable_flag": int(has_all_lags and pd.notna(min_cov) and (min_cov >= cfg.usable_min_coverage)),
        "strict_flag": int(has_all_lags and pd.notna(min_cov) and (min_cov >= cfg.strict_min_coverage)),
        "mean_rain_fraction_day10": float(daily["rainfrac"].mean()) if not daily.empty else np.nan,
    }
    return summary


def build_or_load_event_summary(cfg: Config) -> Tuple[pd.DataFrame, str]:
    cache_summary = cfg.cache_dir / "supp_s3_event_summary.csv"
    if cfg.use_cache and cache_summary.exists():
        summary = pd.read_csv(cache_summary, parse_dates=["event_start", "event_end"])
        return summary, "loaded_cached_event_summary"

    event_files = discover_event_files(cfg.root_dir)
    if not event_files:
        raise FileNotFoundError(f"No event CSV files found under: {cfg.root_dir}")

    precip_scale, precip_note = infer_precip_scale(event_files)
    print(f"[INFO] Event files discovered : {len(event_files):,}")
    print(f"[INFO] Precipitation note    : {precip_note}")

    summary_rows: List[Dict] = []
    for i, fp in enumerate(event_files, start=1):
        out = summarize_one_event(fp, precip_scale, cfg)
        if out is not None:
            summary_rows.append(out)
        if (i % cfg.progress_every == 0) or (i == len(event_files)):
            print(f"[INFO] Summarized {i:,}/{len(event_files):,} files | retained {len(summary_rows):,}")

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise RuntimeError("Event summary table is empty.")
    summary = summary.sort_values(["year", "event_uid"]).reset_index(drop=True)
    summary.to_csv(cache_summary, index=False, encoding="utf-8-sig")
    return summary, precip_note


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


def regional_annual_outcome(summary: pd.DataFrame) -> pd.DataFrame:
    g = summary.groupby(["year", "climate_region"], observed=True)
    out = g.agg(
        n_events=("event_uid", "count"),
        mean_rain_return=("mean_rain_fraction_day10", "mean"),
    ).reset_index()
    out = out.sort_values(["climate_region", "year"]).reset_index(drop=True)
    return out


def _plot_s1_map_panel(ax) -> None:
    _setup_conus_ax(ax, show_ticks=False, show_outline=False)
    draw_filled_region_map(ax, show_labels=False, outline=True, alpha=0.92)


def _plot_region_panel(ax, regional_ann: pd.DataFrame, region: str, cfg: Config) -> None:
    sub = regional_ann.loc[regional_ann["climate_region"] == region].copy().sort_values("year")
    if not sub.empty:
        ax.plot(sub["year"], sub["mean_rain_return"] * 100.0, color=REGION_COLORS[region], lw=0.9, alpha=0.22)
        ax.plot(
            sub["year"],
            rolling_mean(sub["mean_rain_return"] * 100.0, cfg.rolling_window),
            color=REGION_COLORS[region],
            lw=2.5,
        )
    year_min = int(regional_ann["year"].min())
    year_max = int(regional_ann["year"].max())
    ax.set_title(region, fontsize=21, pad=6, color=REGION_COLORS[region])
    ax.set_xlim(year_min, year_max)
    vals = regional_ann["mean_rain_return"].to_numpy(dtype=float) * 100.0
    if np.isfinite(vals).any():
        ymin = float(np.nanpercentile(vals, 1))
        ymax = float(np.nanpercentile(vals, 99))
        pad = max(1.0, 0.08 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(alpha=0.12)
    ax.tick_params(axis='both', labelsize=18, length=3)
    ax.set_xlabel("Year", fontsize=21)
    ax.set_ylabel("Rain-return (%)", fontsize=21)
    for spine in ax.spines.values():
        spine.set_color(REGION_COLORS[region])
        spine.set_linewidth(1.1)


def plot_supp_s3_all_region_rolling_outcomes(regional_ann: pd.DataFrame, cfg: Config) -> None:
    fig = plt.figure(figsize=(17.6, 10.6), facecolor="white")

    map_ax = fig.add_axes([0.33, 0.28, 0.34, 0.42], projection=ccrs.PlateCarree() if HAS_CARTOPY else None)
    _plot_s1_map_panel(map_ax)

    # Manual panel layout tuned to avoid overlaps in the composite figure.
    # Format: [left, bottom, width, height] in figure-fraction coordinates.
    #
    # Final layout correction:
    #   1) Southwest is moved higher, but not so low that its xlabel/ylabel collide
    #      with Southern Great Plains.
    #   2) Southern Great Plains is moved lower-left and narrowed to keep a clear
    #      gap from the Southeast y-axis label.
    #   3) The central map is kept unchanged to preserve the Supp_Fig_R1_01-style grammar.
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
        "Southwest": (0.95, 0.38),
    }

    for region in cfg.region_order:
        ax = fig.add_axes(positions[region])
        _plot_region_panel(ax, regional_ann, region, cfg)

        x0, y0 = _REGION_CONNECT_POS.get(region, _REGION_LABEL_POS.get(region, (-100, 35)))
        fx, fy = target_fracs[region]
        con = ConnectionPatch(
            xyA=(x0, y0),
            coordsA=ccrs.PlateCarree()._as_mpl_transform(map_ax) if HAS_CARTOPY else map_ax.transData,
            xyB=(fx, fy),
            coordsB=ax.transAxes,
            color=REGION_COLORS[region],
            lw=1.0,
            alpha=0.95,
        )
        fig.add_artist(con)

    savefig(fig, cfg.fig_dir / "Supp_Fig_S3_all_region_rolling_outcomes.png")

    fig_map = plt.figure(figsize=(8.6, 5.8), facecolor="white")
    ax_map = fig_map.add_subplot(111, projection=ccrs.PlateCarree() if HAS_CARTOPY else None)
    _plot_s1_map_panel(ax_map)
    savefig(fig_map, cfg.fig_dir / "Supp_Fig_S3_all_region_rolling_outcomes_panel_map.png")

    for region in cfg.region_order:
        fig_r = plt.figure(figsize=(8.6, 5.6), facecolor="white")
        ax_r = fig_r.add_subplot(111)
        _plot_region_panel(ax_r, regional_ann, region, cfg)
        safe_region = re.sub(r"[^A-Za-z0-9]+", "_", region).strip("_")
        savefig(fig_r, cfg.fig_dir / f"Supp_Fig_S3_all_region_rolling_outcomes_{safe_region}.png")


def main(cfg: Config = CFG) -> None:
    summary, precip_note = build_or_load_event_summary(cfg)
    print(f"[INFO] Total summarized events : {len(summary):,}")
    print(f"[INFO] Precip note             : {precip_note}")

    main_df, sample_note = choose_main_sample(summary)
    print(f"[INFO] Sample rule            : {sample_note}")

    regional_ann = regional_annual_outcome(main_df)
    regional_ann.to_csv(cfg.table_dir / "supp_s3_regional_annual_outcome.csv", index=False, encoding="utf-8-sig")

    plot_supp_s3_all_region_rolling_outcomes(regional_ann, cfg)
    print("[INFO] Finished.")
    print(f"[INFO] Output figure: {cfg.fig_dir / 'Supp_Fig_S3_all_region_rolling_outcomes.png'}")


if __name__ == "__main__":
    main(CFG)
