# -*- coding: utf-8 -*-
"""
Northwest all-events ridging / blocking table builder (fixed)
============================================================

Purpose
-------
Build CSV tables for the first hard-mechanism diagnostic using ALL Northwest
heatwave events rather than only the matched subset:

    Northwest all events grouped by recovered vs unrecovered
    tau = -5 ... 0
    Z500, T850, W500

This version fixes two problems that can appear in earlier drafts:
1) it does NOT re-filter Northwest events by usable_flag/min_post_coverage when
   the goal is explicitly "all Northwest events";
2) it robustly reduces ERA5 pressure-level fields to 2D lat-lon slices without
   using netCDF4.

Outputs
-------
All outputs are written to a NEW directory under EVENT_ROOT:
    _nw_all_events_ridging_tables_fixed_v2/
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class Config:
    event_root: Path = Path(
        r"E:\temp_events_ERA5_S1S6_Nature所有数据版本\events_cc3d_postlag10_NCC_with_pr_ws_rh第三篇的数据_added_CAPE_IVT_T850_added_Z500_W500_added_Bowen_Rn_added_WIND250_WIND850"
    )
    z500_w500_root: Path = Path(
        r"E:\第二篇的修改20251229开始修改\新增ERA5环流数据"
    )
    t850_root: Path = Path(
        r"E:\20251107\热浪后降雨\极端干旱事件后的极端降雨\ERA5的ivt数据"
    )
    out_dir: Optional[Path] = None

    prefer_existing_nw_ridging_tables: bool = True
    nw_ridging_dir_name: str = "_nw_ridging_mechanism_tables"
    prefer_existing_result3_tables: bool = True
    result3_v3_tables_dir_name: str = "_result3_scienceadv_focus_rebuild_v3/tables"

    target_region: str = "Northwest"
    tau_min: int = -5
    tau_max: int = 0

    box_lon_min: float = -125.0
    box_lon_max: float = -100.0
    box_lat_min: float = 35.0
    box_lat_max: float = 52.0

    # Only used if we must rebuild event summaries from raw event-window CSVs.
    usable_post_coverage: float = 0.80
    usable_end_coverage: float = 0.80
    rain_threshold_mm: float = 1.0
    rain_fraction_threshold: float = 0.25
    post_lag_max: int = 10
    require_no_overlap: bool = True
    require_no_censor: bool = True

    convert_geopotential_to_gpm: bool = True
    geopotential_g: float = 9.80665

    progress_every_events: int = 250
    progress_every_dates: int = 200

    def __post_init__(self) -> None:
        if self.out_dir is None:
            self.out_dir = self.event_root / "_nw_all_events_ridging_tables_fixed_v2"
        self.out_dir.mkdir(parents=True, exist_ok=True)


CFG = Config()


# -----------------------------------------------------------------------------
# Region helpers
# -----------------------------------------------------------------------------


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
COORD_REGION_CACHE: Dict[Tuple[float, float], Optional[str]] = {}

try:
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point
    HAS_STATE_SHAPES = True
except Exception:
    HAS_STATE_SHAPES = False


@lru_cache(maxsize=1)
def load_state_geometries() -> List[Dict]:
    if not HAS_STATE_SHAPES:
        raise ImportError("cartopy/shapely not available for state geometry loading")
    shp = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_1_states_provinces_lakes")
    reader = shpreader.Reader(shp)
    keep = set(PAPER7_STATE_TO_REGION.keys())
    out: List[Dict] = []
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
    return out


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


def point_to_region(lon: float, lat: float) -> Optional[str]:
    key = (round(float(lon), 4), round(float(lat), 4))
    if key in COORD_REGION_CACHE:
        return COORD_REGION_CACHE[key]
    reg: Optional[str] = None
    if HAS_STATE_SHAPES:
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


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


FILE_RE = re.compile(r"event_(\d{4})_(\d+)_window\.csv")
T850_RE = re.compile(r"(\d{4})_M(\d{1,2})", re.I)


def to_num(s: pd.Series, fill: Optional[float] = None) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if fill is not None:
        out = out.fillna(fill)
    return out


def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def normalize_lon_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(arr > 180.0, arr - 360.0, arr)


def get_event_uid_from_path(path: Path) -> str:
    m = FILE_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse event uid from file name: {path.name}")
    return f"{int(m.group(1))}_{int(m.group(2)):05d}"


def discover_event_files(root: Path) -> List[Path]:
    fps = sorted(root.glob("*/*event_*_window.csv"))
    if not fps:
        fps = sorted(root.rglob("event_*_window.csv"))
    return fps


def open_dataset_no_netcdf4(path: Path) -> xr.Dataset:
    engines = ["h5netcdf", "scipy", None]
    errs = []
    for eng in engines:
        try:
            if eng is None:
                return xr.open_dataset(path)
            return xr.open_dataset(path, engine=eng)
        except Exception as e:
            errs.append(f"engine={eng}: {e}")
    raise RuntimeError(f"Failed to open {path} without netCDF4. Details: {' | '.join(errs)}")


def infer_main_var(ds: xr.Dataset, preferred_names: Sequence[str]) -> str:
    for nm in preferred_names:
        if nm in ds.data_vars:
            return nm
    for nm, da in ds.data_vars.items():
        if da.ndim >= 2:
            return nm
    raise KeyError(f"No suitable data variable found in dataset variables={list(ds.data_vars)}")


def infer_coord_name(ds: xr.Dataset, candidates: Sequence[str]) -> str:
    for nm in candidates:
        if nm in ds.coords:
            return nm
        if nm in ds.dims:
            return nm
    raise KeyError(f"Could not infer coord from candidates={candidates}, available={list(ds.coords)} / dims={list(ds.dims)}")


def standardize_da_lon_lat(da: xr.DataArray) -> xr.DataArray:
    lon_name = infer_coord_name(da.to_dataset(name="tmp"), ["longitude", "lon", "x"])
    lat_name = infer_coord_name(da.to_dataset(name="tmp"), ["latitude", "lat", "y"])
    da = da.assign_coords({lon_name: normalize_lon_values(da[lon_name].values)}).sortby(lon_name)
    if np.any(np.diff(da[lat_name].values) < 0):
        da = da.sortby(lat_name)
    return da


def reduce_field_to_lat_lon(da: xr.DataArray, target_date: pd.Timestamp) -> xr.DataArray:
    # normalize time selection first
    if "time" in da.dims or "time" in da.coords:
        da = da.sel(time=target_date)
    elif "valid_time" in da.dims or "valid_time" in da.coords:
        da = da.sel(valid_time=target_date)
    elif "date" in da.dims or "date" in da.coords:
        da = da.sel(date=target_date)

    # select 500 hPa if any pressure-level-like dimension exists
    for lev_name in ["pressure_level", "level", "isobaricInhPa", "plev"]:
        if lev_name in da.dims or lev_name in da.coords:
            try:
                da = da.sel({lev_name: 500}, method="nearest")
            except Exception:
                pass

    # remove singleton or bookkeeping dims
    for extra in ["expver", "number", "member", "surface", "realization"]:
        if extra in da.dims:
            try:
                da = da.isel({extra: 0})
            except Exception:
                pass

    da = da.squeeze(drop=True)
    da = standardize_da_lon_lat(da)

    # ensure final field is exactly lat-lon
    keep = []
    for nm in da.dims:
        if nm in {"lat", "latitude", "y", "lon", "longitude", "x"}:
            keep.append(nm)
    if len(keep) != 2:
        # attempt final squeeze by selecting index 0 on leftover singleton dims
        for nm in list(da.dims):
            if nm not in keep:
                if da.sizes.get(nm, 0) == 1:
                    da = da.isel({nm: 0})
        da = da.squeeze(drop=True)
    return da


# -----------------------------------------------------------------------------
# Existing summary loaders / rebuild fallback
# -----------------------------------------------------------------------------


def load_existing_nw_ridging_catalog(cfg: Config) -> Optional[pd.DataFrame]:
    fp = cfg.event_root / cfg.nw_ridging_dir_name / "nw_all_event_catalog.csv"
    if not fp.exists():
        return None
    cols = pd.read_csv(fp, nrows=0).columns.tolist()
    parse_cols = [c for c in ["event_start", "event_end"] if c in cols]
    return pd.read_csv(fp, parse_dates=parse_cols)


def load_existing_result3_tables(cfg: Config) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    table_dir = cfg.event_root / cfg.result3_v3_tables_dir_name
    fp_summary = table_dir / "result3_event_summary.csv"
    fp_main = table_dir / "result3_analysis_main.csv"
    if not fp_summary.exists():
        return None, None
    cols = pd.read_csv(fp_summary, nrows=0).columns.tolist()
    parse_cols = [c for c in ["event_start", "event_end"] if c in cols]
    summary = pd.read_csv(fp_summary, parse_dates=parse_cols)
    main = pd.read_csv(fp_main) if fp_main.exists() else None
    return summary, main


def infer_precip_scale(event_files: Sequence[Path], sample_n: int = 40) -> Tuple[float, str]:
    if not event_files:
        return 1.0, "no_files"
    idx = np.linspace(0, len(event_files) - 1, min(sample_n, len(event_files))).astype(int)
    q99s = []
    for i in idx:
        try:
            df = pd.read_csv(event_files[i], low_memory=False)
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


def summarize_event_for_recovery(path: Path, precip_scale: float, cfg: Config) -> Optional[Dict]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        warnings.warn(f"Failed to read {path.name}: {e}")
        return None
    required = {"date", "longitude", "latitude", "lag_day_event", "precipitation"}
    if not required.issubset(df.columns):
        return None
    for c in ["date", "event_start", "event_end"]:
        if c in df.columns:
            df[c] = to_dt(df[c])
    for c in ["longitude", "latitude", "lag_day_event", "precipitation", "temp_air", "T90", "event_id", "year",
              "overlaps_next_local_heat", "is_post_event_0_10_censored", "is_heat_period_event"]:
        if c in df.columns:
            df[c] = to_num(df[c])

    uid = get_event_uid_from_path(path)
    year_val = int(df["year"].dropna().iloc[0]) if "year" in df.columns and df["year"].notna().any() else int(df["date"].dt.year.mode().iloc[0])
    lag = to_num(df["lag_day_event"])
    is_heat = to_num(df.get("is_heat_period_event", pd.Series(1, index=df.index)), fill=1).astype(int)
    heat = df.loc[is_heat == 1].copy()
    if heat.empty:
        heat = df.loc[lag <= 0].copy()
    if heat.empty:
        return None

    coords = heat[["longitude", "latitude"]].dropna().drop_duplicates()
    regs = [point_to_region(float(r.longitude), float(r.latitude)) for r in coords.itertuples(index=False)]
    regs = [r for r in regs if r is not None]
    region = Counter(regs).most_common(1)[0][0] if regs else fallback_region(float(coords["longitude"].mean()), float(coords["latitude"].mean()))

    event_start = heat["date"].min()
    event_end = heat["date"].max()
    if "event_start" in df.columns and df["event_start"].notna().any():
        event_start = df["event_start"].dropna().iloc[0]
    if "event_end" in df.columns and df["event_end"].notna().any():
        event_end = df["event_end"].dropna().iloc[0]

    duration = int((event_end - event_start).days + 1) if pd.notna(event_start) and pd.notna(event_end) else int(heat["date"].nunique())
    centroid_lon = float(to_num(heat["longitude"]).mean())
    centroid_lat = float(to_num(heat["latitude"]).mean())
    footprint_ncells = int(len(heat[["longitude", "latitude"]].round(4).drop_duplicates()))
    heat_excess = np.nan
    if {"temp_air", "T90"}.issubset(heat.columns):
        exc = (to_num(heat["temp_air"]) - to_num(heat["T90"])).clip(lower=0)
        heat_excess = float(exc.mean()) if exc.notna().any() else np.nan

    df = df.copy()
    df["precip_mm"] = to_num(df["precipitation"]) * precip_scale
    post = df.loc[(lag >= 1) & (lag <= cfg.post_lag_max)].copy()

    heat_keys = heat[["longitude", "latitude"]].dropna().round(4)
    heat_keys["coord_key"] = heat_keys["longitude"].astype(str) + "_" + heat_keys["latitude"].astype(str)
    footprint = set(heat_keys["coord_key"].tolist())
    if not footprint:
        return None

    recovered = False
    observed_lags: set[int] = set()
    min_post_cov = np.nan
    if not post.empty:
        post = post.copy()
        post["coord_key"] = post[["longitude", "latitude"]].round(4).astype(str).agg("_".join, axis=1)
        rows = []
        for tau, g in post.groupby(to_num(post["lag_day_event"]).astype(int)):
            g = g.dropna(subset=["coord_key"]).drop_duplicates("coord_key")
            g = g.loc[g["coord_key"].isin(footprint)].copy()
            coverage = float(g["coord_key"].nunique()) / float(len(footprint))
            rainfrac = float((to_num(g["precip_mm"]) >= cfg.rain_threshold_mm).sum()) / float(len(footprint))
            rows.append({"lag_day": int(tau), "coverage": coverage, "rainfrac": rainfrac})
        post_daily = pd.DataFrame(rows)
        if not post_daily.empty:
            observed_lags = set(post_daily["lag_day"].astype(int).tolist())
            min_post_cov = float(post_daily["coverage"].min())
            ok = (post_daily["coverage"] >= cfg.usable_post_coverage) & (post_daily["rainfrac"] >= cfg.rain_fraction_threshold)
            recovered = bool(ok.any())

    has_all_lags = int(set(range(1, cfg.post_lag_max + 1)).issubset(observed_lags))
    any_overlap = int((to_num(post.get("overlaps_next_local_heat", pd.Series(0, index=post.index)), fill=0) == 1).any()) if not post.empty else 0
    any_censor = int((to_num(post.get("is_post_event_0_10_censored", pd.Series(0, index=post.index)), fill=0) == 1).any()) if not post.empty else 0
    usable = bool(has_all_lags and pd.notna(min_post_cov) and (min_post_cov >= cfg.usable_post_coverage))
    if cfg.require_no_overlap:
        usable = usable and (any_overlap == 0)
    if cfg.require_no_censor:
        usable = usable and (any_censor == 0)

    return {
        "event_uid": uid,
        "source_file": str(path),
        "year": year_val,
        "event_start": event_start,
        "event_end": event_end,
        "duration": duration,
        "centroid_lon": centroid_lon,
        "centroid_lat": centroid_lat,
        "climate_region": region,
        "footprint_ncells": footprint_ncells,
        "mean_heat_excess": heat_excess,
        "no_recovery": int(not recovered),
        "recovered_by_day10": int(recovered),
        "has_all_lags": has_all_lags,
        "min_post_coverage": min_post_cov,
        "any_overlap": any_overlap,
        "any_censor": any_censor,
        "usable_flag": int(usable),
    }


def rebuild_nw_catalog(cfg: Config) -> pd.DataFrame:
    event_files = discover_event_files(cfg.event_root)
    if not event_files:
        raise FileNotFoundError(f"No event files found under {cfg.event_root}")
    precip_scale, note = infer_precip_scale(event_files)
    print(f"[INFO] Rebuilding NW event catalog from raw event-window files")
    print(f"[INFO] Event file count: {len(event_files):,} | precip note: {note}")
    rows: List[Dict] = []
    for i, fp in enumerate(event_files, start=1):
        row = summarize_event_for_recovery(fp, precip_scale, cfg)
        if row is not None:
            rows.append(row)
        if i % cfg.progress_every_events == 0 or i == len(event_files):
            print(f"[INFO] Summarized {i:,}/{len(event_files):,} event files | kept={len(rows):,}")
    return pd.DataFrame(rows).sort_values(["year", "event_uid"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Gridded data inventory + readers
# -----------------------------------------------------------------------------


def build_zw_inventory(root: Path) -> pd.DataFrame:
    records: List[Dict] = []
    z_files = sorted(root.rglob("geopotential_stream-oper_daily-mean.nc"))
    for zf in z_files:
        wf = zf.with_name("vertical_velocity_0_daily-mean.nc")
        if not wf.exists():
            continue
        try:
            ds = open_dataset_no_netcdf4(zf)
            time_name = infer_coord_name(ds, ["time", "valid_time", "date"])
            tvals = pd.to_datetime(ds[time_name].values)
            start = pd.Timestamp(tvals.min())
            end = pd.Timestamp(tvals.max())
            ds.close()
            records.append({"z_path": str(zf), "w_path": str(wf), "start": start, "end": end})
        except Exception as e:
            warnings.warn(f"Skipping bad Z/W file pair {zf} | {wf}: {e}")
    if not records:
        raise FileNotFoundError(f"No usable Z500/W500 files found under {root}")
    return pd.DataFrame(records).sort_values(["start", "end"]).reset_index(drop=True)


@lru_cache(maxsize=16)
def load_z_dataset(path_str: str) -> xr.DataArray:
    ds = open_dataset_no_netcdf4(Path(path_str))
    var = infer_main_var(ds, ["z", "geopotential", "zg", "geopotential_500hPa"])
    return ds[var].load()


@lru_cache(maxsize=16)
def load_w_dataset(path_str: str) -> xr.DataArray:
    ds = open_dataset_no_netcdf4(Path(path_str))
    var = infer_main_var(ds, ["w", "vertical_velocity", "omega", "vertical_velocity_0", "vertical_velocity_500hPa"])
    return ds[var].load()


def find_zw_record(inv: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    hit = inv.loc[(inv["start"] <= date) & (inv["end"] >= date)]
    if hit.empty:
        raise KeyError(f"No Z/W dataset covers date {date.date()}")
    return hit.iloc[0]


def subset_box(da: xr.DataArray, cfg: Config, date: pd.Timestamp) -> xr.DataArray:
    sel = reduce_field_to_lat_lon(da, date)
    # after reduction, coord names may still be lon/lat or longitude/latitude
    lon_name = "lon" if "lon" in sel.coords or "lon" in sel.dims else "longitude"
    lat_name = "lat" if "lat" in sel.coords or "lat" in sel.dims else "latitude"
    sel = sel.sel({lon_name: slice(cfg.box_lon_min, cfg.box_lon_max), lat_name: slice(cfg.box_lat_min, cfg.box_lat_max)})
    if lon_name != "lon" or lat_name != "lat":
        sel = sel.rename({lon_name: "lon", lat_name: "lat"})
    return sel


@lru_cache(maxsize=256)
def read_t850_month_csv(path_str: str) -> pd.DataFrame:
    fp = Path(path_str)
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    needed = {"date", "temperature_850hPa_mean", "longitude", "latitude"}
    if not needed.issubset(df.columns):
        raise KeyError(f"T850 file missing required columns: {fp}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["longitude"] = normalize_lon_values(to_num(df["longitude"]).values)
    df["latitude"] = to_num(df["latitude"]).values
    df["temperature_850hPa_mean"] = to_num(df["temperature_850hPa_mean"]).values
    return df


def build_t850_inventory(root: Path) -> Dict[Tuple[int, int], Path]:
    out: Dict[Tuple[int, int], Path] = {}
    for fp in sorted(root.rglob("*")):
        if not fp.is_file():
            continue
        m = T850_RE.search(fp.name)
        if not m:
            continue
        out[(int(m.group(1)), int(m.group(2)))] = fp
    if not out:
        raise FileNotFoundError(f"No T850 monthly CSV files found under {root}")
    return out


def get_t850_da_for_date(inv: Dict[Tuple[int, int], Path], date: pd.Timestamp, lon_target: np.ndarray, lat_target: np.ndarray, cfg: Config) -> xr.DataArray:
    key = (int(date.year), int(date.month))
    if key not in inv:
        raise KeyError(f"No T850 monthly CSV available for {key}")
    df = read_t850_month_csv(str(inv[key]))
    sub = df.loc[df["date"] == date].copy()
    if sub.empty:
        raise KeyError(f"No T850 rows for date {date.date()} in {inv[key]}")
    sub = sub.loc[(sub["longitude"] >= cfg.box_lon_min) & (sub["longitude"] <= cfg.box_lon_max) &
                  (sub["latitude"] >= cfg.box_lat_min) & (sub["latitude"] <= cfg.box_lat_max)].copy()
    if sub.empty:
        raise KeyError(f"T850 file has no data in Northwest box on {date.date()}")
    grid = sub.pivot_table(index="latitude", columns="longitude", values="temperature_850hPa_mean", aggfunc="mean")
    grid = grid.sort_index().sort_index(axis=1)
    da = xr.DataArray(grid.to_numpy(dtype=float), coords={"lat": grid.index.to_numpy(dtype=float), "lon": grid.columns.to_numpy(dtype=float)}, dims=("lat", "lon"), name="t850")
    return da.interp(lat=lat_target, lon=lon_target, method="nearest")


# -----------------------------------------------------------------------------
# All-events ridging tables
# -----------------------------------------------------------------------------


class CompositeAccumulator:
    def __init__(self) -> None:
        self.lon: Optional[np.ndarray] = None
        self.lat: Optional[np.ndarray] = None
        self.sums: Optional[np.ndarray] = None
        self.counts: Optional[np.ndarray] = None

    def add(self, z500: xr.DataArray, t850: xr.DataArray, w500: xr.DataArray) -> None:
        stack = np.stack([
            np.asarray(z500.values, dtype=float),
            np.asarray(t850.values, dtype=float),
            np.asarray(w500.values, dtype=float),
        ], axis=0)
        if self.sums is None:
            self.lon = np.asarray(z500["lon"].values, dtype=float)
            self.lat = np.asarray(z500["lat"].values, dtype=float)
            self.sums = np.zeros_like(stack, dtype=float)
            self.counts = np.zeros_like(stack, dtype=float)
        mask = np.isfinite(stack)
        self.sums[mask] += stack[mask]
        self.counts[mask] += 1.0

    def mean_stack(self) -> np.ndarray:
        if self.sums is None or self.counts is None:
            raise RuntimeError("Accumulator is empty")
        return self.sums / np.where(self.counts > 0, self.counts, np.nan)


def build_nw_all_events_ridging_tables(cfg: Config) -> None:
    summary = None

    if cfg.prefer_existing_nw_ridging_tables:
        summary = load_existing_nw_ridging_catalog(cfg)
        if summary is not None:
            print("[INFO] Reusing existing Northwest ridging event catalog")

    if summary is None:
        summary_r3, _ = (None, None)
        if cfg.prefer_existing_result3_tables:
            summary_r3, _ = load_existing_result3_tables(cfg)
            if summary_r3 is not None:
                print("[INFO] Reusing existing Result 3 v3 event summary")
        if summary_r3 is not None:
            summary = summary_r3
        else:
            summary = rebuild_nw_catalog(cfg)
            print("[INFO] Existing Northwest / Result 3 tables not found; rebuilt Northwest catalog from raw event files")

    summary = summary.copy()
    if "climate_region" not in summary.columns or "event_end" not in summary.columns:
        raise KeyError("Summary table must contain climate_region and event_end")
    summary["climate_region"] = summary["climate_region"].astype(str)
    summary["event_end"] = pd.to_datetime(summary["event_end"], errors="coerce")

    nw_catalog = summary.loc[summary["climate_region"] == cfg.target_region].copy().sort_values(["year", "event_uid"])
    if nw_catalog.empty:
        raise RuntimeError("No Northwest events found in the event catalog.")

    # IMPORTANT FIX: for the all-events version, do NOT drop events by usable_flag/min_post_coverage.
    # Keep all Northwest events and only derive the recovered/unrecovered grouping.
    if "no_recovery" not in nw_catalog.columns and "recovered_by_day10" in nw_catalog.columns:
        nw_catalog["no_recovery"] = 1 - to_num(nw_catalog["recovered_by_day10"]).fillna(0).astype(int)
    if "recovered_by_day10" not in nw_catalog.columns and "no_recovery" in nw_catalog.columns:
        nw_catalog["recovered_by_day10"] = 1 - to_num(nw_catalog["no_recovery"]).fillna(0).astype(int)

    if "no_recovery" not in nw_catalog.columns:
        raise KeyError("Northwest catalog must contain no_recovery or recovered_by_day10")

    nw_catalog["group"] = np.where(to_num(nw_catalog["no_recovery"]).fillna(0).astype(int) == 1, "Unrecovered", "Recovered")
    nw_catalog.to_csv(cfg.out_dir / "nw_all_events_ridging_event_catalog.csv", index=False, encoding="utf-8-sig")
    nw_catalog.groupby("group", as_index=False).size().rename(columns={"size": "n_events"}).to_csv(
        cfg.out_dir / "nw_all_events_ridging_group_counts.csv", index=False, encoding="utf-8-sig"
    )

    zw_inv = build_zw_inventory(cfg.z500_w500_root)
    t850_inv = build_t850_inventory(cfg.t850_root)
    print(f"[INFO] Z/W file groups discovered: {len(zw_inv):,}")
    print(f"[INFO] T850 monthly CSVs discovered: {len(t850_inv):,}")

    mean_rows: List[Dict] = []
    composite: Dict[Tuple[str, int], CompositeAccumulator] = {}
    all_dates = []
    for r in nw_catalog.itertuples(index=False):
        if pd.isna(r.event_end):
            continue
        for tau in range(cfg.tau_min, cfg.tau_max + 1):
            all_dates.append((str(r.event_uid), str(r.group), int(tau), pd.Timestamp(r.event_end) + pd.Timedelta(days=int(tau))))

    print(f"[INFO] Northwest all-events: {len(nw_catalog):,} | event-date slices: {len(all_dates):,}")
    n_fail = 0
    for i, (event_uid, group, tau, date) in enumerate(all_dates, start=1):
        try:
            rec = find_zw_record(zw_inv, date)
            z_da = subset_box(load_z_dataset(rec["z_path"]), cfg, date)
            w_da = subset_box(load_w_dataset(rec["w_path"]), cfg, date)
            t_da = get_t850_da_for_date(t850_inv, date, np.asarray(z_da["lon"].values, dtype=float), np.asarray(z_da["lat"].values, dtype=float), cfg)

            z_vals = np.asarray(z_da.values, dtype=float)
            w_vals = np.asarray(w_da.values, dtype=float)
            t_vals = np.asarray(t_da.values, dtype=float)
            z_gpm = z_vals / cfg.geopotential_g if cfg.convert_geopotential_to_gpm else z_vals

            mean_rows.append({
                "event_uid": event_uid,
                "group": group,
                "tau": tau,
                "date": date,
                "z500_mean_raw": float(np.nanmean(z_vals)),
                "z500_mean_gpm": float(np.nanmean(z_gpm)),
                "t850_mean": float(np.nanmean(t_vals)),
                "w500_mean": float(np.nanmean(w_vals)),
                "z500_std_gpm": float(np.nanstd(z_gpm)),
                "t850_std": float(np.nanstd(t_vals)),
                "w500_std": float(np.nanstd(w_vals)),
                "n_grid": int(np.isfinite(z_vals).sum()),
            })

            key = (group, tau)
            if key not in composite:
                composite[key] = CompositeAccumulator()
            composite[key].add(
                xr.DataArray(z_gpm, coords={"lat": z_da["lat"].values, "lon": z_da["lon"].values}, dims=("lat", "lon"), name="z500_gpm"),
                xr.DataArray(t_vals, coords={"lat": z_da["lat"].values, "lon": z_da["lon"].values}, dims=("lat", "lon"), name="t850"),
                xr.DataArray(w_vals, coords={"lat": z_da["lat"].values, "lon": z_da["lon"].values}, dims=("lat", "lon"), name="w500"),
            )
        except Exception as e:
            n_fail += 1
            warnings.warn(f"Failed for {event_uid} tau={tau} date={date.date()}: {e}")
        if i % cfg.progress_every_dates == 0 or i == len(all_dates):
            print(f"[INFO] Extracted {i:,}/{len(all_dates):,} all-event event-date slices")

    means = pd.DataFrame(mean_rows).sort_values(["group", "event_uid", "tau"]).reset_index(drop=True)
    means.to_csv(cfg.out_dir / "nw_all_events_ridging_tau_event_box_means.csv", index=False, encoding="utf-8-sig")

    comp_rows: List[Dict] = []
    comp_summary_rows: List[Dict] = []
    for (group, tau), acc in sorted(composite.items(), key=lambda x: (x[0][0], x[0][1])):
        try:
            mean_stack = acc.mean_stack()
        except RuntimeError:
            continue
        z500_gpm, t850, w500 = mean_stack[0], mean_stack[1], mean_stack[2]
        comp_summary_rows.append({
            "group": group,
            "tau": tau,
            "z500_mean_gpm": float(np.nanmean(z500_gpm)),
            "t850_mean": float(np.nanmean(t850)),
            "w500_mean": float(np.nanmean(w500)),
            "n_lat": int(len(acc.lat)),
            "n_lon": int(len(acc.lon)),
        })
        for ii, lat in enumerate(acc.lat):
            for jj, lon in enumerate(acc.lon):
                comp_rows.append({
                    "group": group,
                    "tau": tau,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "z500_gpm": float(z500_gpm[ii, jj]) if np.isfinite(z500_gpm[ii, jj]) else np.nan,
                    "t850": float(t850[ii, jj]) if np.isfinite(t850[ii, jj]) else np.nan,
                    "w500": float(w500[ii, jj]) if np.isfinite(w500[ii, jj]) else np.nan,
                })
    pd.DataFrame(comp_rows).to_csv(cfg.out_dir / "nw_all_events_ridging_composite_grid_long.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comp_summary_rows).to_csv(cfg.out_dir / "nw_all_events_ridging_composite_tau_summary.csv", index=False, encoding="utf-8-sig")

    info = pd.DataFrame([
        {"key": "target_region", "value": cfg.target_region},
        {"key": "tau_min", "value": cfg.tau_min},
        {"key": "tau_max", "value": cfg.tau_max},
        {"key": "box_lon_min", "value": cfg.box_lon_min},
        {"key": "box_lon_max", "value": cfg.box_lon_max},
        {"key": "box_lat_min", "value": cfg.box_lat_min},
        {"key": "box_lat_max", "value": cfg.box_lat_max},
        {"key": "n_northwest_events", "value": int(len(nw_catalog))},
        {"key": "n_recovered_events", "value": int((nw_catalog["group"] == "Recovered").sum())},
        {"key": "n_unrecovered_events", "value": int((nw_catalog["group"] == "Unrecovered").sum())},
        {"key": "n_event_tau_rows", "value": int(len(means))},
        {"key": "n_failed_event_tau_rows", "value": int(n_fail)},
    ])
    info.to_csv(cfg.out_dir / "nw_all_events_ridging_run_info.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] Northwest all-events ridging tables written to: {cfg.out_dir}")


def main(cfg: Config = CFG) -> None:
    build_nw_all_events_ridging_tables(cfg)


if __name__ == "__main__":
    main(CFG)
