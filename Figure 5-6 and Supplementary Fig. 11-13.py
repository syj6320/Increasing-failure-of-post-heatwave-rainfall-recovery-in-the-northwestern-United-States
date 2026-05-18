# -*- coding: utf-8 -*-
"""
Northwest mechanism figures — NCC-style redesign
================================================

Purpose
-------
Rebuild the Northwest blocking / moisture / land-support figures with a more
submission-grade layout for Nature Climate Change / Nature Communications style
review. This version does not change the underlying statistics. It changes the
visual grammar:

1. tighter and more hierarchical panel layout
2. consistent typography and line weights
3. dedicated legend rows (no overlap with panels)
4. slim aligned colorbars per map family
5. masked pcolormesh and contours to the valid Northwest land-domain geometry
6. reduced title clutter; panel letters outside axes; tau labels inside panels
7. balanced whitespace and final-export sizing for manuscript review

Outputs
-------
Main figures:
    Figure1_NW_blocking_main_matched_NCCredesign.png/.pdf
    Figure2_NW_moisture_land_followup_matched_NCCredesign.png/.pdf

Supplementary figures:
    Supp_Fig_NW_blocking_fulltau_matched_NCCredesign.png/.pdf
    Supp_Fig_NW_land_support_matched_NCCredesign.png/.pdf
    Supp_Fig_NW_matching_sensitivity_NCCredesign.png/.pdf
    Supp_Fig_NW_moisture_support_matched_NCCredesign.png/.pdf

Notes
-----
- Keeps your requested large on-screen typography, but organizes the canvas in a
  print-like hierarchy. For final manuscript submission, you may still need one
  more downscaling pass to reach 5–8 pt at final journal width.
- Requires: numpy, pandas, matplotlib, cartopy, shapely.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HAS_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import shapereader
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
except Exception:
    HAS_CARTOPY = False

HAS_SHAPELY = True
try:
    from shapely.ops import unary_union
    from shapely import contains_xy
except Exception:
    HAS_SHAPELY = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Style system
# -----------------------------------------------------------------------------
DISPLAY_FONT = 21
PANEL_FONT = 24
TAU_FONT = 18
LEGEND_FONT = 18

REC_COLOR = "#355C7D"      # muted blue
UNREC_COLOR = "#C05A2B"    # muted rust
T850_COLOR = "#8B1E3F"     # wine
W500_COLOR = "#2C7FB8"     # cool blue
PRECIP_COLOR = "#238443"   # dark green
BOWEN_COLOR = "#C51B8A"    # magenta
ZERO_LINE = "0.72"
SPINE = "0.25"
GRID = "0.85"

DIVERGING_CMAP = "RdBu_r"
MAP_CMAPS = {
    # Blocking maps are displayed with a one-sided sequential scale because the
    # matched Northwest Z500 differences in these panels are overwhelmingly positive.
    "z500_gpm": "viridis",
    "relative_humidity": "BrBG",
    "soil_moist": "PRGn",
    "rn": "RdBu_r",
    "moisture_convergence": "RdBu_r",
    "default": DIVERGING_CMAP,
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei"],
    "axes.labelsize": DISPLAY_FONT,
    "axes.titlesize": DISPLAY_FONT,
    "xtick.labelsize": DISPLAY_FONT,
    "ytick.labelsize": DISPLAY_FONT,
    "legend.fontsize": LEGEND_FONT,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 5.5,
    "ytick.major.size": 5.5,
    "savefig.dpi": 360,
})

CFG = {
    "root": r"E:\temp_events_ERA5_S1S6_Nature所有数据版本\events_cc3d_postlag10_NCC_with_pr_ws_rh第三篇的数据_added_CAPE_IVT_T850_added_Z500_W500_added_Bowen_Rn_added_WIND250_WIND850",
    "outdir": r"E:\temp_events_ERA5_S1S6_Nature所有数据版本\events_cc3d_postlag10_NCC_with_pr_ws_rh第三篇的数据_added_CAPE_IVT_T850_added_Z500_W500_added_Bowen_Rn_added_WIND250_WIND850\_n",
    "blocking_taus_main": [-5, -3, -1, 0],
    "followup_taus_main": [-3, 0],
    "all_taus": [-5, -4, -3, -2, -1, 0],
    "n_boot": 3000,
    "seed": 42,
    "fig1_size": (27.0, 13.4),
    "fig2_size": (27.0, 16.5),
    "supp_block_size": (24.0, 13.0),
    "supp_land_size": (22.0, 13.0),
    "supp_match_size": (24.0, 6.8),
    "supp_moisture_size": (22.0, 14.0),
}

ROOT = Path(CFG["root"])
OUTDIR = Path(CFG["outdir"])
OUTDIR.mkdir(parents=True, exist_ok=True)

PATHS = {
    "matched": {
        "ridging_grid": ROOT / "_nw_ridging_mechanism_tables" / "nw_matched_composite_grid_long.csv",
        "ridging_event": ROOT / "_nw_ridging_mechanism_tables" / "nw_matched_tau_event_box_means.csv",
        "moisture_grid": ROOT / "_nw_moisture_mechanism_tables" / "nw_moisture_matched_composite_grid_long.csv",
        "moisture_event": ROOT / "_nw_moisture_mechanism_tables" / "nw_moisture_matched_tau_event_box_means.csv",
        "land_grid": ROOT / "_nw_land_mechanism_tables" / "nw_land_matched_composite_grid_long.csv",
        "land_event": ROOT / "_nw_land_mechanism_tables" / "nw_land_matched_tau_event_box_means.csv",
    },
    "all_events": {
        "ridging_grid": ROOT / "_nw_all_events_ridging_tables_fixed_v2" / "nw_all_events_ridging_composite_grid_long.csv",
        "ridging_event": ROOT / "_nw_all_events_ridging_tables_fixed_v2" / "nw_all_events_ridging_tau_event_box_means.csv",
        "moisture_grid": ROOT / "_nw_moisture_mechanism_tables" / "nw_all_events_moisture_composite_grid_long.csv",
        "moisture_event": ROOT / "_nw_moisture_mechanism_tables" / "nw_all_events_moisture_tau_event_box_means.csv",
        "land_grid": ROOT / "_nw_all_events_land_tables" / "nw_all_events_land_composite_grid_long.csv",
        "land_event": ROOT / "_nw_all_events_land_tables" / "nw_all_events_land_tau_event_box_means.csv",
    },
}

GRID_ALIASES: Dict[str, List[str]] = {
    "z500_gpm": ["z500_gpm"],
    "t850": ["t850", "temperature_850hPa_mean"],
    "w500": ["w500", "vertical_velocity_500hPa", "omega500", "vertical_velocity_500hPa_mean"],
    "relative_humidity": ["relative_humidity", "rh"],
    "vimd": ["vimd", "vertically_integrated_moisture_divergence_mean"],
    "moisture_convergence": ["moisture_convergence"],
    "cape": ["cape", "convective_available_potential_energy_mean"],
    "precipitation_mm": ["precipitation_mm", "precip_mm"],
    "soil_moist": ["soil_moist", "soil_moisture"],
    "bowen_ratio": ["bowen_ratio", "Bowen_ratio"],
    "rn": ["rn", "Rn"],
}

EVENT_ALIASES: Dict[str, List[str]] = {
    "z500_mean_gpm": ["z500_mean_gpm"],
    "t850_mean": ["t850_mean", "temperature_850hPa_mean"],
    "w500_mean": ["w500_mean", "vertical_velocity_500hPa_mean", "omega500_mean"],
    "relative_humidity_mean": ["relative_humidity_mean", "rh_mean"],
    "vimd_mean": ["vimd_mean", "vertically_integrated_moisture_divergence_mean", "vertically_integrated_moisture_divergence_mean_mean"],
    "moisture_convergence_mean": ["moisture_convergence_mean"],
    "cape_mean": ["cape_mean", "convective_available_potential_energy_mean"],
    "precipitation_mm_mean": ["precipitation_mm_mean", "precip_mm_mean"],
    "soil_moist_mean": ["soil_moist_mean", "soil_moisture_mean"],
    "bowen_ratio_mean": ["bowen_ratio_mean", "Bowen_ratio_mean"],
    "rn_mean": ["rn_mean", "Rn_mean"],
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")


def read_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def canonical_group(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    s = s.replace({
        "recovered": "Recovered",
        "unrecovered": "Unrecovered",
        "not recovered": "Unrecovered",
        "no_recovery": "Unrecovered",
        "nonrecovered": "Unrecovered",
        "non-recovered": "Unrecovered",
    })
    return s


def maybe_convert_rn(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    med = np.nanmedian(np.abs(s.values))
    if np.isfinite(med) and med > 1e5:
        return s / 1e6
    return s


def apply_aliases(df: pd.DataFrame, alias_map: Dict[str, List[str]]) -> pd.DataFrame:
    out = df.copy()
    for canonical, candidates in alias_map.items():
        if canonical in out.columns:
            continue
        for cand in candidates:
            if cand in out.columns:
                out = out.rename(columns={cand: canonical})
                break
    return out


def clean_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "group" in out.columns:
        out["group"] = canonical_group(out["group"])
    if "tau" in out.columns:
        out["tau"] = pd.to_numeric(out["tau"], errors="coerce")
        out = out[out["tau"].notna()].copy()
        out["tau"] = out["tau"].astype(int)
    if "latitude" in out.columns:
        out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    if "longitude" in out.columns:
        out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    return out


def prepare_grid_df(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_common_columns(df)
    df = apply_aliases(df, GRID_ALIASES)
    if "rn" in df.columns:
        df["rn"] = maybe_convert_rn(df["rn"])
    return df


def prepare_event_df(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_common_columns(df)
    df = apply_aliases(df, EVENT_ALIASES)
    if "rn_mean" in df.columns:
        df["rn_mean"] = maybe_convert_rn(df["rn_mean"])
    return df


def load_block(sample_key: str, block: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if block == "ridging":
        return (
            prepare_grid_df(read_csv(PATHS[sample_key]["ridging_grid"])),
            prepare_event_df(read_csv(PATHS[sample_key]["ridging_event"])),
        )
    if block == "moisture":
        return (
            prepare_grid_df(read_csv(PATHS[sample_key]["moisture_grid"])),
            prepare_event_df(read_csv(PATHS[sample_key]["moisture_event"])),
        )
    if block == "land":
        return (
            prepare_grid_df(read_csv(PATHS[sample_key]["land_grid"])),
            prepare_event_df(read_csv(PATHS[sample_key]["land_event"])),
        )
    raise ValueError(block)


def has_usable_column(df: pd.DataFrame, col: str) -> bool:
    return (col in df.columns) and pd.to_numeric(df[col], errors="coerce").notna().any()


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int,
                      ci: Tuple[float, float] = (2.5, 97.5)) -> Tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(vals, size=vals.size, replace=True).mean()
    return float(vals.mean()), float(np.percentile(means, ci[0])), float(np.percentile(means, ci[1]))


def bootstrap_diff_ci(unr: np.ndarray, rec: np.ndarray, n_boot: int, seed: int,
                      ci: Tuple[float, float] = (2.5, 97.5)) -> Tuple[float, float, float]:
    unr = np.asarray(unr, dtype=float)
    rec = np.asarray(rec, dtype=float)
    unr = unr[np.isfinite(unr)]
    rec = rec[np.isfinite(rec)]
    if unr.size == 0 or rec.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = rng.choice(unr, size=unr.size, replace=True).mean() - rng.choice(rec, size=rec.size, replace=True).mean()
    diff = unr.mean() - rec.mean()
    return float(diff), float(np.percentile(boot, ci[0])), float(np.percentile(boot, ci[1]))


def summarize_event_trajectories(df: pd.DataFrame, var: str, taus: List[int]) -> pd.DataFrame:
    rows = []
    for tau in taus:
        sub = df[df["tau"] == tau].copy()
        if var not in sub.columns:
            rec = np.array([], dtype=float)
            unr = np.array([], dtype=float)
        else:
            vals = pd.to_numeric(sub[var], errors="coerce")
            rec = vals[sub["group"] == "Recovered"].dropna().values
            unr = vals[sub["group"] == "Unrecovered"].dropna().values
        rec_mean, rec_lo, rec_hi = bootstrap_mean_ci(rec, CFG["n_boot"], CFG["seed"] + tau + 100)
        unr_mean, unr_lo, unr_hi = bootstrap_mean_ci(unr, CFG["n_boot"], CFG["seed"] + tau + 200)
        diff_mean, diff_lo, diff_hi = bootstrap_diff_ci(unr, rec, CFG["n_boot"], CFG["seed"] + tau + 300)
        rows.append({
            "tau": tau,
            "rec_n": int(rec.size),
            "unr_n": int(unr.size),
            "rec_mean": rec_mean,
            "rec_lo": rec_lo,
            "rec_hi": rec_hi,
            "unr_mean": unr_mean,
            "unr_lo": unr_lo,
            "unr_hi": unr_hi,
            "diff_mean": diff_mean,
            "diff_lo": diff_lo,
            "diff_hi": diff_hi,
        })
    return pd.DataFrame(rows)


def build_diff_pivot(df: pd.DataFrame, tau: int, var: str, strict: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if not has_usable_column(df, var):
        if strict:
            raise ValueError(f"Column '{var}' not found or entirely empty.")
        return None, None, None

    sub = df[df["tau"] == tau].copy()
    if sub.empty:
        if strict:
            raise ValueError(f"No rows for tau={tau}")
        return None, None, None

    sub[var] = pd.to_numeric(sub[var], errors="coerce")
    sub = sub[["group", "latitude", "longitude", var]].dropna()
    if sub.empty:
        if strict:
            raise ValueError(f"No usable data for tau={tau}, var={var}")
        return None, None, None

    wide = sub.pivot_table(index=["latitude", "longitude"], columns="group", values=var, aggfunc="mean").reset_index()
    if "Recovered" not in wide.columns or "Unrecovered" not in wide.columns:
        if strict:
            raise ValueError(f"Both Recovered and Unrecovered required for tau={tau}, var={var}")
        return None, None, None

    wide["diff"] = pd.to_numeric(wide["Unrecovered"], errors="coerce") - pd.to_numeric(wide["Recovered"], errors="coerce")
    wide = wide.dropna(subset=["diff", "latitude", "longitude"])
    if wide.empty:
        if strict:
            raise ValueError(f"No valid diff field for tau={tau}, var={var}")
        return None, None, None

    lat_sorted = np.sort(wide["latitude"].unique())
    lon_sorted = np.sort(wide["longitude"].unique())
    grid = wide.pivot(index="latitude", columns="longitude", values="diff").reindex(index=lat_sorted, columns=lon_sorted)
    z = grid.values
    x, y = np.meshgrid(lon_sorted, lat_sorted)
    return x, y, z


def get_map_arrays(df: pd.DataFrame, taus: List[int], var: str) -> List[np.ndarray]:
    out = []
    for tau in taus:
        _, _, z = build_diff_pivot(df, tau, var, strict=False)
        if z is not None and np.isfinite(z).any():
            out.append(z)
    return out


def quantile_abs_limit(arrays: List[np.ndarray], q: float = 0.98, floor: Optional[float] = None) -> float:
    data = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays if a is not None]) if arrays else np.array([])
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 1.0 if floor is None else float(floor)
    lim = np.nanquantile(np.abs(data), q)
    if floor is not None:
        lim = max(lim, floor)
    if not np.isfinite(lim) or lim == 0:
        lim = 1.0 if floor is None else float(floor)
    return float(lim)


def blocking_map_limit(arrays: List[np.ndarray], q: float = 0.85, shrink: float = 0.90, floor: float = 3.0) -> float:
    """Use a tighter one-sided positive range for Northwest blocking maps.

    These matched Northwest blocking panels are dominated by positive Z500
    differences, so a symmetric negative-to-positive scale wastes most of the
    dynamic range. We therefore estimate the upper bound only from positive
    values and display the map from 0 to vmax.
    """
    data = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays if a is not None]) if arrays else np.array([])
    data = data[np.isfinite(data)]
    data = data[data > 0]
    if data.size == 0:
        return float(floor)
    vmax = np.nanquantile(data, q)
    vmax = max(float(floor), float(vmax) * float(shrink))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(floor)
    return float(vmax)


def contour_levels_from_arrays(arrays: List[np.ndarray], n_each_side: int = 3, q: float = 0.90) -> np.ndarray:
    data = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays if a is not None]) if arrays else np.array([])
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.array([])
    lim = np.nanquantile(np.abs(data), q)
    if not np.isfinite(lim) or lim <= 0:
        lim = np.nanmax(np.abs(data))
    if not np.isfinite(lim) or lim <= 0:
        return np.array([])
    pos = np.linspace(lim / n_each_side, lim, n_each_side)
    levels = np.concatenate([-pos[::-1], pos])
    return np.unique(np.round(levels, 4))


_GEOM_CACHE: Dict[Tuple[float, float, float, float], object] = {}


def compute_valid_extent(df: pd.DataFrame, vars_to_check: List[str], pad_lon: float = 0.30, pad_lat: float = 0.30) -> Tuple[float, float, float, float]:
    use = df.copy()
    mask = np.zeros(len(use), dtype=bool)
    for var in vars_to_check:
        if var in use.columns:
            mask |= pd.to_numeric(use[var], errors="coerce").notna().values
    use = use.loc[mask].copy()
    if use.empty:
        return -125.0, -100.0, 35.0, 49.0
    return (
        float(use["longitude"].min()) - pad_lon,
        float(use["longitude"].max()) + pad_lon,
        float(use["latitude"].min()) - pad_lat,
        float(use["latitude"].max()) + pad_lat,
    )


def build_nw_domain_geometry(extent: Tuple[float, float, float, float]):
    if not (HAS_CARTOPY and HAS_SHAPELY):
        return None
    if extent in _GEOM_CACHE:
        return _GEOM_CACHE[extent]
    lon_min, lon_max, lat_min, lat_max = extent
    try:
        shp = shapereader.natural_earth(resolution="50m", category="cultural", name="admin_1_states_provinces")
        rdr = shapereader.Reader(shp)
        geoms = []
        for rec in rdr.records():
            attrs = rec.attributes
            adm0 = attrs.get("adm0_a3") or attrs.get("sr_adm0_a3") or attrs.get("iso_a2")
            if adm0 not in ("USA", "US"):
                continue
            geom = rec.geometry
            if geom is None:
                continue
            minx, miny, maxx, maxy = geom.bounds
            if maxx < lon_min or minx > lon_max or maxy < lat_min or miny > lat_max:
                continue
            geoms.append(geom)
        if not geoms:
            _GEOM_CACHE[extent] = None
            return None
        union_geom = unary_union(geoms)
        _GEOM_CACHE[extent] = union_geom
        return union_geom
    except Exception:
        _GEOM_CACHE[extent] = None
        return None


def mask_to_geometry(x: np.ndarray, y: np.ndarray, z: np.ndarray, geom) -> np.ndarray:
    if geom is None or not HAS_SHAPELY:
        return z
    try:
        inside = contains_xy(geom, x, y)
        zz = np.array(z, dtype=float, copy=True)
        zz[~inside] = np.nan
        return zz
    except Exception:
        return z


def resolve_first_usable_column(df: pd.DataFrame, preferred: List[str], contains_all: Optional[List[str]] = None, fallback_contains_any: Optional[List[str]] = None) -> Optional[str]:
    for col in preferred:
        if has_usable_column(df, col):
            return col
    cols = list(df.columns)
    if contains_all:
        for col in cols:
            low = col.lower()
            if all(tok in low for tok in contains_all) and has_usable_column(df, col):
                return col
    if fallback_contains_any:
        for col in cols:
            low = col.lower()
            if any(tok in low for tok in fallback_contains_any) and has_usable_column(df, col):
                return col
    return None


# -----------------------------------------------------------------------------
# Drawing primitives
# -----------------------------------------------------------------------------
def soften_axes(ax):
    for side in ax.spines:
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis="both", which="major", pad=5, color=SPINE)


def style_timeseries(ax):
    soften_axes(ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def style_box(ax):
    soften_axes(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def panel_label(ax, text: str):
    ax.text(-0.12, 1.03, text, transform=ax.transAxes,
            fontsize=PANEL_FONT, fontweight="bold", ha="left", va="bottom")


def tau_tag(ax, tau: int):
    ax.text(0.5, 1.01, rf"$\tau$ = {tau}", transform=ax.transAxes,
            fontsize=TAU_FONT, ha="center", va="bottom")


def variable_cmap(primary_var: str) -> str:
    return MAP_CMAPS.get(primary_var, MAP_CMAPS["default"])


def _choose_geo_ticks(vmin: float, vmax: float, axis: str = "lon", sparse: bool = False) -> List[int]:
    rng = float(vmax - vmin)
    if axis == "lon":
        step = 10 if sparse else 5
    else:
        step = 2 if rng <= 10 else 4
    start = int(np.ceil(vmin / step) * step)
    end = int(np.floor(vmax / step) * step)
    ticks = list(range(start, end + 1, step))
    if sparse and len(ticks) > 4:
        ticks = ticks[::2] if len(ticks[::2]) >= 3 else ticks
    return ticks


def maybe_add_map_features(ax, extent: Tuple[float, float, float, float]):
    lon_min, lon_max, lat_min, lat_max = extent
    if HAS_CARTOPY:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.coastlines(resolution="50m", linewidth=0.75, color="0.50")
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45, edgecolor="0.55")
        try:
            states = cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces_lines",
                scale="50m",
                facecolor="none",
            )
            ax.add_feature(states, edgecolor="0.78", linewidth=0.32)
        except Exception:
            pass
        gl = ax.gridlines(draw_labels=False, linewidth=0.32, color="0.82", alpha=0.75, linestyle="--")
        gl.xlines = True
        gl.ylines = True
    else:
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.grid(True, color="0.82", linewidth=0.32, alpha=0.75, linestyle="--")
    soften_axes(ax)


def finish_map_ticks(ax, extent: Tuple[float, float, float, float], show_left: bool = False, show_bottom: bool = False, sparse_lat: bool = True, sparse_lon: bool = True):
    lon_min, lon_max, lat_min, lat_max = extent
    lon_ticks = _choose_geo_ticks(lon_min, lon_max, axis="lon", sparse=sparse_lon)
    lat_ticks = _choose_geo_ticks(lat_min, lat_max, axis="lat", sparse=sparse_lat)
    if HAS_CARTOPY:
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        try:
            ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f', degree_symbol='°'))
            ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f', degree_symbol='°'))
        except Exception:
            pass
        ax.tick_params(labelbottom=show_bottom, labelleft=show_left, labeltop=False, labelright=False)
        if not show_bottom:
            ax.set_xticklabels([])
        if not show_left:
            ax.set_yticklabels([])
    else:
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        if show_bottom:
            ax.set_xticklabels([f"{abs(int(x))}°W" if x < 0 else f"{int(x)}°E" for x in lon_ticks])
        else:
            ax.set_xticklabels([])
        if show_left:
            ax.set_yticklabels([f"{int(y)}°N" if y >= 0 else f"{abs(int(y))}°S" for y in lat_ticks])
        else:
            ax.set_yticklabels([])


def map_mesh(ax, x, y, z, cmap=None, vlim: Optional[float] = None,
             vmin: Optional[float] = None, vmax: Optional[float] = None,
             use_two_slope: bool = True):
    if x is None or y is None or z is None:
        return None
    cmap_name = DIVERGING_CMAP if cmap is None else cmap
    cmap_obj = plt.get_cmap(cmap_name).copy()
    cmap_obj.set_bad(alpha=0.0)
    if use_two_slope:
        norm = None if vlim is None else TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
    else:
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = vlim
        norm = None if vmax is None else Normalize(vmin=vmin, vmax=vmax, clip=True)
    if HAS_CARTOPY:
        return ax.pcolormesh(x, y, z, cmap=cmap_obj, norm=norm, shading="auto", transform=ccrs.PlateCarree(), rasterized=True)
    return ax.pcolormesh(x, y, z, cmap=cmap_obj, norm=norm, shading="auto", rasterized=True)


def map_contour(ax, x, y, z, levels: np.ndarray, color: str, linestyle: str = "solid", linewidths: float = 1.05):
    if x is None or y is None or z is None or levels is None or len(levels) == 0:
        return None
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z, dtype=float)
    if x.shape != y.shape or x.shape != z.shape or np.all(~np.isfinite(z)):
        return None
    try:
        if HAS_CARTOPY:
            return ax.contour(x, y, z, levels=levels, colors=color, linewidths=linewidths,
                              linestyles=linestyle, transform=ccrs.PlateCarree(), alpha=0.95)
        return ax.contour(x, y, z, levels=levels, colors=color, linewidths=linewidths, linestyles=linestyle, alpha=0.95)
    except Exception:
        return None


def add_slim_colorbar(fig, mappable, axes, label: str, pad: float = 0.025, cax=None):
    if cax is not None:
        cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    else:
        cbar = fig.colorbar(mappable, ax=axes, orientation="horizontal", fraction=0.035, pad=pad, aspect=55)
    cbar.set_label(label, fontsize=DISPLAY_FONT)
    cbar.ax.tick_params(labelsize=DISPLAY_FONT - 1)
    cbar.outline.set_linewidth(0.8)
    return cbar


def set_axis_text(ax, ylabel: Optional[str] = None, xlabel: Optional[str] = None):
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=12)
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=8)


def plot_trajectory(ax, event_df: pd.DataFrame, var: str, ylabel: str,
                    title: Optional[str] = None, legend_mode: str = "none",
                    show_xlabel: bool = True, legend_y: float = -0.24) -> pd.DataFrame:
    summ = summarize_event_trajectories(event_df, var, CFG["all_taus"])
    tau = summ["tau"].values
    style_timeseries(ax)
    ax.plot(tau, summ["rec_mean"], color=REC_COLOR, lw=2.3, marker="o", ms=4.2, label="Recovered")
    ax.fill_between(tau, summ["rec_lo"], summ["rec_hi"], color=REC_COLOR, alpha=0.16, linewidth=0)
    ax.plot(tau, summ["unr_mean"], color=UNREC_COLOR, lw=2.3, marker="o", ms=4.2, label="Unrecovered")
    ax.fill_between(tau, summ["unr_lo"], summ["unr_hi"], color=UNREC_COLOR, alpha=0.16, linewidth=0)
    ax.axvline(0, color="0.55", lw=1.0, ls="--")
    ax.set_xlim(min(CFG["all_taus"]) - 0.15, max(CFG["all_taus"]) + 0.15)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    set_axis_text(ax, ylabel=ylabel, xlabel=(r"Lead time to event end (τ)" if show_xlabel else ""))
    if title:
        ax.set_title(title, pad=6)
    if legend_mode == "inside":
        ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=2.0, columnspacing=1.4)
    elif legend_mode == "inside_ll":
        ax.legend(frameon=False, loc="lower left", ncol=2, handlelength=2.0, columnspacing=1.2)
    elif legend_mode == "below":
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=2, handlelength=2.0, columnspacing=1.2)
    return summ


def make_event_end_box(ax, event_df: pd.DataFrame, var: str, ylabel: str, title: Optional[str] = None):
    sub0 = event_df[event_df["tau"] == 0].copy()
    rec = pd.to_numeric(sub0.loc[sub0["group"] == "Recovered", var], errors="coerce").dropna().values if var in sub0.columns else np.array([])
    unr = pd.to_numeric(sub0.loc[sub0["group"] == "Unrecovered", var], errors="coerce").dropna().values if var in sub0.columns else np.array([])
    bp = ax.boxplot(
        [rec, unr], labels=["Recovered", "Unrecovered"], patch_artist=True, widths=0.56,
        medianprops=dict(color="black", linewidth=1.15),
        boxprops=dict(linewidth=0.95), whiskerprops=dict(linewidth=0.95), capprops=dict(linewidth=0.95),
        flierprops=dict(marker='o', markersize=3.2, markeredgewidth=0.75, markerfacecolor='white', markeredgecolor='0.35')
    )
    bp["boxes"][0].set_facecolor(REC_COLOR)
    bp["boxes"][0].set_alpha(0.42)
    bp["boxes"][1].set_facecolor(UNREC_COLOR)
    bp["boxes"][1].set_alpha(0.42)
    style_box(ax)
    if title:
        ax.set_title(title, pad=6)
    set_axis_text(ax, ylabel=ylabel)
    ax.text(0.03, 0.97, f"Nrec={len(rec)}\nNunrec={len(unr)}", transform=ax.transAxes,
            va="top", ha="left", fontsize=DISPLAY_FONT - 2, color="0.35")


def draw_map(ax, grid_df: pd.DataFrame, tau: int, primary_var: str, vlim: float,
             contour_specs: List[Tuple[str, np.ndarray, str, str, float]],
             extent: Tuple[float, float, float, float], geom,
             show_left: bool, show_bottom: bool, sparse_lat: bool = True, sparse_lon: bool = True,
             positive_only: bool = False, cmap_override: Optional[str] = None):
    maybe_add_map_features(ax, extent)
    x0, y0, z0 = build_diff_pivot(grid_df, tau, primary_var, strict=True)
    z0 = mask_to_geometry(x0, y0, z0, geom)
    if positive_only and z0 is not None:
        z0 = np.array(z0, dtype=float, copy=True)
        z0[z0 < 0] = 0.0
        pc = map_mesh(
            ax, x0, y0, z0,
            cmap=(cmap_override or variable_cmap(primary_var)),
            vmin=0.0, vmax=vlim, use_two_slope=False
        )
    else:
        pc = map_mesh(ax, x0, y0, z0, cmap=(cmap_override or variable_cmap(primary_var)), vlim=vlim)
    drawn = []
    for var, levels, color, ls, lw in contour_specs:
        x, y, z = build_diff_pivot(grid_df, tau, var, strict=False)
        if z is not None:
            z = mask_to_geometry(x, y, z, geom)
        cs = map_contour(ax, x, y, z, levels, color, ls, linewidths=lw)
        drawn.append(cs is not None)
    tau_tag(ax, tau)
    finish_map_ticks(ax, extent, show_left=show_left, show_bottom=show_bottom, sparse_lat=sparse_lat, sparse_lon=sparse_lon)
    return pc, drawn


def add_legend_row(ax, handles: List[Line2D], ncol: Optional[int] = None):
    ax.axis("off")
    ax.legend(handles=handles, loc="center", frameon=False, ncol=(len(handles) if ncol is None else ncol),
              columnspacing=1.7, handlelength=2.2, handletextpad=0.6)


def save_figure(fig, out_png: Path):
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 1
# -----------------------------------------------------------------------------
def plot_figure1_blocking(sample_key: str = "matched") -> Path:
    grid_df, event_df = load_block(sample_key, "ridging")
    taus = CFG["blocking_taus_main"]
    extent = compute_valid_extent(grid_df, ["z500_gpm", "t850", "w500"])
    geom = build_nw_domain_geometry(extent)

    z500_vlim = blocking_map_limit(get_map_arrays(grid_df, taus, "z500_gpm"), q=0.85, shrink=0.90, floor=3.0)
    t850_levels = contour_levels_from_arrays(get_map_arrays(grid_df, taus, "t850"), q=0.88)
    w500_levels = contour_levels_from_arrays(get_map_arrays(grid_df, taus, "w500"), q=0.88)

    fig = plt.figure(figsize=CFG["fig1_size"])
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1.08, 0.16, 0.94, 0.22], hspace=0.62, wspace=0.34)

    map_axes = []
    letters = list("abcdefgh")
    last_pc = None
    contour_t850 = False
    contour_w500 = False

    for i, tau in enumerate(taus):
        ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree()) if HAS_CARTOPY else fig.add_subplot(gs[0, i])
        pc, drawn = draw_map(
            ax, grid_df, tau, "z500_gpm", z500_vlim,
            [("t850", t850_levels, T850_COLOR, "solid", 1.0),
             ("w500", w500_levels, W500_COLOR, "dashed", 1.05)],
            extent, geom, show_left=True, show_bottom=True,
            positive_only=True, cmap_override="viridis",
        )
        panel_label(ax, letters[i])
        last_pc = pc
        contour_t850 = contour_t850 or drawn[0]
        contour_w500 = contour_w500 or drawn[1]
        map_axes.append(ax)

    cax = fig.add_subplot(gs[1, 1:3])
    add_slim_colorbar(fig, last_pc, map_axes, "Unrecovered − Recovered in Z500 (gpm)", cax=cax)
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + 0.04, pos.width, pos.height])
    ax1 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 1])
    ax3 = fig.add_subplot(gs[2, 2])
    ax4 = fig.add_subplot(gs[2, 3])

    plot_trajectory(ax1, event_df, "z500_mean_gpm", "Z500 (gpm)", legend_mode="none")
    plot_trajectory(ax2, event_df, "t850_mean", "T850 (K)", legend_mode="none")
    plot_trajectory(ax3, event_df, "w500_mean", "W500", legend_mode="none")
    make_event_end_box(ax4, event_df, "z500_mean_gpm", r"Z500 at τ = 0 (gpm)", title="Event-end separation")

    for j, ax in enumerate([ax1, ax2, ax3, ax4], start=4):
        panel_label(ax, letters[j])

    handles = [
        Line2D([0], [0], color=REC_COLOR, lw=2.3, marker="o", ms=4.2, label="Recovered"),
        Line2D([0], [0], color=UNREC_COLOR, lw=2.3, marker="o", ms=4.2, label="Unrecovered"),
    ]
    if contour_t850:
        handles.append(Line2D([0], [0], color=T850_COLOR, lw=1.7, linestyle="solid", label="ΔT850 contour"))
    if contour_w500:
        handles.append(Line2D([0], [0], color=W500_COLOR, lw=1.7, linestyle="dashed", label="ΔW500 contour"))

    leg_ax = fig.add_subplot(gs[3, :])
    add_legend_row(leg_ax, handles, ncol=min(4, len(handles)))

    out_png = OUTDIR / f"Figure1_NW_blocking_main_{sample_key}_NCCredesign.png"
    save_figure(fig, out_png)
    return out_png


# -----------------------------------------------------------------------------
# Figure 2
# -----------------------------------------------------------------------------
def plot_figure2_followup(sample_key: str = "matched") -> Path:
    moisture_grid, moisture_event = load_block(sample_key, "moisture")
    land_grid, land_event = load_block(sample_key, "land")
    taus = CFG["followup_taus_main"]

    extent_m = compute_valid_extent(moisture_grid, ["relative_humidity", "precipitation_mm"])
    extent_l = compute_valid_extent(land_grid, ["soil_moist", "bowen_ratio"])
    extent = (
        min(extent_m[0], extent_l[0]), max(extent_m[1], extent_l[1]),
        min(extent_m[2], extent_l[2]), max(extent_m[3], extent_l[3]),
    )
    geom = build_nw_domain_geometry(extent)

    rh_vlim = quantile_abs_limit(get_map_arrays(moisture_grid, taus, "relative_humidity"), q=0.98)
    sm_vlim = quantile_abs_limit(get_map_arrays(land_grid, taus, "soil_moist"), q=0.98)
    pr_levels = contour_levels_from_arrays(get_map_arrays(moisture_grid, taus, "precipitation_mm"), q=0.88)
    bowen_levels = contour_levels_from_arrays(get_map_arrays(land_grid, taus, "bowen_ratio"), q=0.88)

    fig = plt.figure(figsize=CFG["fig2_size"])
    gs = GridSpec(6, 4, figure=fig,
                  height_ratios=[1.00, 0.10, 0.13, 1.00, 0.10, 0.13],
                  hspace=0.42, wspace=0.34)

    letters = list("abcdefgh")

    # Top block
    ax_m1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()) if HAS_CARTOPY else fig.add_subplot(gs[0, 0])
    ax_m2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()) if HAS_CARTOPY else fig.add_subplot(gs[0, 1])
    ax_m3 = fig.add_subplot(gs[0, 2])
    ax_m4 = fig.add_subplot(gs[0, 3])

    pc1, _ = draw_map(ax_m1, moisture_grid, taus[0], "relative_humidity", rh_vlim,
                      [("precipitation_mm", pr_levels, PRECIP_COLOR, "solid", 1.05)],
                      extent, geom, show_left=True, show_bottom=True, sparse_lat=True)
    pc2, _ = draw_map(ax_m2, moisture_grid, taus[1], "relative_humidity", rh_vlim,
                      [("precipitation_mm", pr_levels, PRECIP_COLOR, "solid", 1.05)],
                      extent, geom, show_left=True, show_bottom=True, sparse_lat=True)
    cax1 = fig.add_subplot(gs[1, 0:2])
    add_slim_colorbar(fig, pc2, [ax_m1, ax_m2], "Unrecovered − Recovered in RH (%)", cax=cax1)
    leg_m = fig.add_subplot(gs[2, 0:2])
    add_legend_row(leg_m, [Line2D([0], [0], color=PRECIP_COLOR, lw=1.8, linestyle="solid", label="ΔPrecipitation contour")], ncol=1)

    plot_trajectory(ax_m3, moisture_event, "relative_humidity_mean", "RH (%)", legend_mode="below", legend_y=-0.24)
    plot_trajectory(ax_m4, moisture_event, "precipitation_mm_mean", r"Precip. (mm day$^{-1}$)", legend_mode="none")

    # Bottom block
    ax_l1 = fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree()) if HAS_CARTOPY else fig.add_subplot(gs[3, 0])
    ax_l2 = fig.add_subplot(gs[3, 1], projection=ccrs.PlateCarree()) if HAS_CARTOPY else fig.add_subplot(gs[3, 1])
    ax_l3 = fig.add_subplot(gs[3, 2])
    ax_l4 = fig.add_subplot(gs[3, 3])

    pc3, _ = draw_map(ax_l1, land_grid, taus[0], "soil_moist", sm_vlim,
                      [("bowen_ratio", bowen_levels, BOWEN_COLOR, "solid", 1.05)],
                      extent, geom, show_left=True, show_bottom=True, sparse_lat=True)
    pc4, _ = draw_map(ax_l2, land_grid, taus[1], "soil_moist", sm_vlim,
                      [("bowen_ratio", bowen_levels, BOWEN_COLOR, "solid", 1.05)],
                      extent, geom, show_left=True, show_bottom=True, sparse_lat=True)
    cax2 = fig.add_subplot(gs[4, 0:2])
    add_slim_colorbar(fig, pc4, [ax_l1, ax_l2], "Unrecovered − Recovered in soil moisture", cax=cax2)
    leg_l = fig.add_subplot(gs[5, 0:2])
    add_legend_row(leg_l, [Line2D([0], [0], color=BOWEN_COLOR, lw=1.8, linestyle="solid", label="ΔBowen ratio contour")], ncol=1)

    plot_trajectory(ax_l3, land_event, "soil_moist_mean", "Soil moisture", legend_mode="below", legend_y=-0.24)
    plot_trajectory(ax_l4, land_event, "bowen_ratio_mean", "Bowen ratio", legend_mode="none")

    for i, ax in enumerate([ax_m1, ax_m2, ax_m3, ax_m4, ax_l1, ax_l2, ax_l3, ax_l4]):
        panel_label(ax, letters[i])

    out_png = OUTDIR / f"Figure2_NW_moisture_land_followup_{sample_key}_NCCredesign.png"
    save_figure(fig, out_png)
    return out_png


# -----------------------------------------------------------------------------
# Supplementary figures
# -----------------------------------------------------------------------------
def plot_supp_blocking_fulltau(sample_key: str = "matched") -> Path:
    grid_df, _ = load_block(sample_key, "ridging")
    taus = CFG["all_taus"]
    extent = compute_valid_extent(grid_df, ["z500_gpm", "t850", "w500"])
    geom = build_nw_domain_geometry(extent)

    z500_vlim = blocking_map_limit(get_map_arrays(grid_df, taus, "z500_gpm"), q=0.85, shrink=0.90, floor=3.0)
    t850_levels = contour_levels_from_arrays(get_map_arrays(grid_df, taus, "t850"), q=0.88)
    w500_levels = contour_levels_from_arrays(get_map_arrays(grid_df, taus, "w500"), q=0.88)

    fig = plt.figure(figsize=CFG["supp_block_size"])
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1.0, 1.0, 0.12, 0.12], hspace=0.32, wspace=0.22)
    axes = []
    last_pc = None
    for i, tau in enumerate(taus):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree()) if HAS_CARTOPY else fig.add_subplot(gs[r, c])
        pc, _ = draw_map(ax, grid_df, tau, "z500_gpm", z500_vlim,
                         [("t850", t850_levels, T850_COLOR, "solid", 1.0),
                          ("w500", w500_levels, W500_COLOR, "dashed", 1.05)],
                         extent, geom, show_left=True, show_bottom=True, sparse_lat=True,
                         positive_only=True, cmap_override="viridis")
        panel_label(ax, chr(97 + i))
        axes.append(ax)
        last_pc = pc

    cax = fig.add_subplot(gs[2, :])
    add_slim_colorbar(fig, last_pc, axes, "Unrecovered − Recovered in Z500 (gpm)", cax=cax)
    leg_ax = fig.add_subplot(gs[3, :])
    add_legend_row(
        leg_ax,
        [
            Line2D([0], [0], color=T850_COLOR, lw=1.7, linestyle="solid", label="ΔT850 contour"),
            Line2D([0], [0], color=W500_COLOR, lw=1.7, linestyle="dashed", label="ΔW500 contour"),
        ],
        ncol=2,
    )
    out_png = OUTDIR / f"Supp_Fig_NW_blocking_fulltau_{sample_key}_NCCredesign.png"
    save_figure(fig, out_png)
    return out_png


def plot_supp_land_support(sample_key: str = "matched") -> Path:
    grid_df, event_df = load_block(sample_key, "land")
    extent = compute_valid_extent(grid_df, ["rn"])
    geom = build_nw_domain_geometry(extent)
    taus = CFG["followup_taus_main"]
    rn_vlim = quantile_abs_limit(get_map_arrays(grid_df, taus, "rn"), q=0.98, floor=1.0)

    fig = plt.figure(figsize=CFG["supp_land_size"])
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.92, 1.00, 0.12], hspace=0.50, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree()) if HAS_CARTOPY else fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree()) if HAS_CARTOPY else fig.add_subplot(gs[1, 1])

    plot_trajectory(ax1, event_df, "rn_mean", r"Rn (MJ m$^{-2}$ day$^{-1}$)", title="Rn trajectory", legend_mode="inside_ll")
    make_event_end_box(ax2, event_df, "rn_mean", r"Rn at τ = 0 (MJ m$^{-2}$ day$^{-1}$)", title="Event-end separation")
    pc3, _ = draw_map(ax3, grid_df, taus[0], "rn", rn_vlim, [], extent, geom, show_left=True, show_bottom=True, sparse_lat=True)
    pc4, _ = draw_map(ax4, grid_df, taus[1], "rn", rn_vlim, [], extent, geom, show_left=True, show_bottom=True, sparse_lat=True)
    cax = fig.add_subplot(gs[2, :])
    add_slim_colorbar(fig, pc4, [ax3, ax4], r"Unrecovered − Recovered in Rn (MJ m$^{-2}$ day$^{-1}$)", cax=cax)

    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        panel_label(ax, chr(97 + i))

    out_png = OUTDIR / f"Supp_Fig_NW_land_support_{sample_key}_NCCredesign.png"
    save_figure(fig, out_png)
    return out_png


def plot_supp_moisture_support(sample_key: str = "matched") -> Path:
    grid_df, event_df = load_block(sample_key, "moisture")
    extent = compute_valid_extent(grid_df, ["moisture_convergence", "cape"])
    geom = build_nw_domain_geometry(extent)
    taus = CFG["followup_taus_main"]
    mc_vlim = quantile_abs_limit(get_map_arrays(grid_df, taus, "moisture_convergence"), q=0.98, floor=0.06)

    fig = plt.figure(figsize=CFG["supp_moisture_size"])
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.88, 1.06, 0.12], hspace=0.58, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree()) if HAS_CARTOPY else fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree()) if HAS_CARTOPY else fig.add_subplot(gs[1, 1])

    plot_trajectory(ax1, event_df, "moisture_convergence_mean", "Moisture convergence", title="Moisture convergence trajectory", legend_mode="below", legend_y=-0.22)
    plot_trajectory(ax2, event_df, "cape_mean", "CAPE", title="CAPE trajectory", legend_mode="below", legend_y=-0.22)
    pc3, _ = draw_map(ax3, grid_df, taus[0], "moisture_convergence", mc_vlim, [], extent, geom, show_left=True, show_bottom=True, sparse_lat=True)
    pc4, _ = draw_map(ax4, grid_df, taus[1], "moisture_convergence", mc_vlim, [], extent, geom, show_left=True, show_bottom=True, sparse_lat=True)
    cax = fig.add_subplot(gs[2, :])
    add_slim_colorbar(fig, pc4, [ax3, ax4], "Unrecovered − Recovered in moisture convergence", cax=cax)

    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        panel_label(ax, chr(97 + i))

    out_png = OUTDIR / f"Supp_Fig_NW_moisture_support_{sample_key}_NCCredesign.png"
    save_figure(fig, out_png)
    return out_png


def plot_supp_matching_sensitivity() -> Optional[Path]:
    try:
        ridging_m = load_block("matched", "ridging")[1]
        ridging_a = load_block("all_events", "ridging")[1]
        moisture_m = load_block("matched", "moisture")[1]
        moisture_a = load_block("all_events", "moisture")[1]
        land_m = load_block("matched", "land")[1]
        land_a = load_block("all_events", "land")[1]
    except FileNotFoundError:
        return None

    ridging_var_m = resolve_first_usable_column(ridging_m, ["z500_mean_gpm"], contains_all=["z500"], fallback_contains_any=["z500"])
    ridging_var_a = resolve_first_usable_column(ridging_a, ["z500_mean_gpm"], contains_all=["z500"], fallback_contains_any=["z500"])
    moisture_var_m = resolve_first_usable_column(moisture_m, ["relative_humidity_mean"], contains_all=["relative", "humidity"], fallback_contains_any=["rh"])
    moisture_var_a = resolve_first_usable_column(moisture_a, ["relative_humidity_mean"], contains_all=["relative", "humidity"], fallback_contains_any=["rh"])
    land_var_m = resolve_first_usable_column(land_m, ["soil_moist_mean", "soil_moisture_mean"], contains_all=["soil", "moist"], fallback_contains_any=["soil", "moist", "land", "rn"])
    land_var_a = resolve_first_usable_column(land_a, ["soil_moist_mean", "soil_moisture_mean"], contains_all=["soil", "moist"], fallback_contains_any=["soil", "moist", "land", "rn"])

    fig = plt.figure(figsize=(24.0, 6.8))
    gs = GridSpec(1, 3, figure=fig, hspace=0.0, wspace=0.28)

    specs = [
        (fig.add_subplot(gs[0, 0]), ridging_m, ridging_var_m, ridging_a, ridging_var_a, r"ΔZ500 box mean (gpm)", "Ridging / blocking"),
        (fig.add_subplot(gs[0, 1]), moisture_m, moisture_var_m, moisture_a, moisture_var_a, r"ΔRH box mean (%)", "Moisture support"),
        (fig.add_subplot(gs[0, 2]), land_m, land_var_m, land_a, land_var_a, r"ΔLand box mean", "Land memory"),
    ]

    def _safe_summary(df, var):
        if var is None:
            return pd.DataFrame({"tau": CFG["all_taus"], "diff_mean": np.nan, "diff_lo": np.nan, "diff_hi": np.nan})
        return summarize_event_trajectories(df, var, CFG["all_taus"])

    for i, (ax, dfm, varm, dfa, vara, ylabel, title) in enumerate(specs):
        sm = _safe_summary(dfm, varm)
        sa = _safe_summary(dfa, vara)
        style_timeseries(ax)
        ax.plot(sm["tau"], sm["diff_mean"], color="#111111", lw=2.15, marker="o", ms=4.0, label="Matched")
        ax.fill_between(sm["tau"], sm["diff_lo"], sm["diff_hi"], color="#111111", alpha=0.10)
        ax.plot(sa["tau"], sa["diff_mean"], color="#8C8C8C", lw=2.0, marker="s", ms=3.8, label="All events")
        ax.fill_between(sa["tau"], sa["diff_lo"], sa["diff_hi"], color="#8C8C8C", alpha=0.10)
        ax.axhline(0, color=ZERO_LINE, lw=0.9)
        ax.axvline(0, color="0.55", lw=1.0, ls="--")
        ax.set_title(title, pad=6)
        set_axis_text(ax, ylabel=ylabel, xlabel=r"Lead time to event end (τ)")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if i == 0:
            ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=2.0, columnspacing=1.3)
        panel_label(ax, chr(97 + i))

    out_png = OUTDIR / "Supp_Fig_NW_matching_sensitivity_NCCredesign.png"
    save_figure(fig, out_png)
    return out_png


def main():
    outputs = [
        plot_figure1_blocking("matched"),
        plot_figure2_followup("matched"),
        plot_supp_blocking_fulltau("matched"),
        plot_supp_land_support("matched"),
        plot_supp_matching_sensitivity(),
        plot_supp_moisture_support("matched"),
    ]
    print("=" * 96)
    print("Finished. Output files:")
    for p in outputs:
        if p is not None:
            print(p)
    print("=" * 96)


if __name__ == "__main__":
    main()
