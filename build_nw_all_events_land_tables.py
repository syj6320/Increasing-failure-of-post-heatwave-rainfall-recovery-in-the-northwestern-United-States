# -*- coding: utf-8 -*-
"""
Northwest all-events land-memory / surface-partition table builder
=================================================================

Purpose
-------
Build CSV tables for the third high-priority hard-mechanism diagnostic:

    Northwest all events grouped by recovered vs unrecovered
    tau = -5 ... 0
    soil_moist, Bowen_ratio, Rn

This version uses ALL Northwest events rather than only the matched subset.

Preference order for Northwest sample
-------------------------------------
1) Reuse existing Northwest ridging tables under:
       EVENT_ROOT/_nw_ridging_mechanism_tables/
   so the Northwest event catalog is consistent with the ridging diagnosis.
2) Otherwise reuse existing Result 3 v3 tables under:
       EVENT_ROOT/_result3_scienceadv_focus_rebuild_v3/tables/
3) Otherwise rebuild the Northwest event catalog from raw event-window CSV files.

Mechanism fields
----------------
- soil_moist from monthly grid CSV files
- Bowen_ratio computed as H / LE from monthly energy CSV files
- Rn from monthly energy CSV files

Outputs
-------
All outputs are written to a NEW directory:
    EVENT_ROOT/_nw_all_events_land_tables/

Main tables
-----------
- nw_all_events_land_event_catalog.csv
- nw_all_events_land_group_counts.csv
- nw_all_events_land_tau_event_box_means.csv
- nw_all_events_land_composite_grid_long.csv
- nw_all_events_land_composite_tau_summary.csv
- nw_all_events_land_run_info.csv

Notes
-----
- This script does not use netCDF4.
- Unlike the matched builder, this version uses all Northwest events and groups
  them by recovered vs unrecovered.
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


@dataclass
class Config:
    event_root: Path = Path(
        r"E:\temp_events_ERA5_S1S6_Nature所有数据版本\events_cc3d_postlag10_NCC_with_pr_ws_rh第三篇的数据_added_CAPE_IVT_T850_added_Z500_W500_added_Bowen_Rn_added_WIND250_WIND850"
    )
    energy_root: Path = Path(
        r"D:\热浪干旱nature\热浪的能量数据"
    )
    sm_root: Path = Path(
        r"E:\20251107\热浪后降雨\极端干旱事件后的极端降雨\包含风速湿度等5数据"
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

    usable_post_coverage: float = 0.80
    usable_end_coverage: float = 0.80
    rain_threshold_mm: float = 1.0
    rain_fraction_threshold: float = 0.25
    post_lag_max: int = 10
    require_no_overlap: bool = True
    require_no_censor: bool = True

    progress_every_events: int = 250
    progress_every_dates: int = 100

    def __post_init__(self) -> None:
        if self.out_dir is None:
            self.out_dir = self.event_root / "_nw_all_events_land_tables"
        self.out_dir.mkdir(parents=True, exist_ok=True)


CFG = Config()

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
        raise ImportError("cartopy/shapely not available")
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


FILE_RE = re.compile(r"event_(\d{4})_(\d+)_window\.csv")
YM_RE = re.compile(r"(\d{4})_M(\d{1,2})", re.I)


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


def load_existing_nw_ridging_tables(cfg: Config) -> Optional[pd.DataFrame]:
    table_dir = cfg.event_root / cfg.nw_ridging_dir_name
    fp_catalog = table_dir / "nw_all_event_catalog.csv"
    if not fp_catalog.exists():
        return None
    cols = pd.read_csv(fp_catalog, nrows=0).columns.tolist()
    parse_cols = [c for c in ["event_start", "event_end"] if c in cols]
    return pd.read_csv(fp_catalog, parse_dates=parse_cols)


def load_existing_result3_summary(cfg: Config) -> Optional[pd.DataFrame]:
    table_dir = cfg.event_root / cfg.result3_v3_tables_dir_name
    fp_summary = table_dir / "result3_event_summary.csv"
    if not fp_summary.exists():
        return None
    cols = pd.read_csv(fp_summary, nrows=0).columns.tolist()
    parse_cols = [c for c in ["event_start", "event_end"] if c in cols]
    return pd.read_csv(fp_summary, parse_dates=parse_cols)


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


def summarize_event(path: Path, precip_scale: float, cfg: Config) -> Optional[Dict]:
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


def rebuild_event_catalog(cfg: Config) -> pd.DataFrame:
    event_files = discover_event_files(cfg.event_root)
    if not event_files:
        raise FileNotFoundError(f"No event files found under {cfg.event_root}")
    precip_scale, note = infer_precip_scale(event_files)
    print(f"[INFO] Rebuilding NW event catalog from raw event-window files")
    print(f"[INFO] Event file count: {len(event_files):,} | precip note: {note}")
    rows: List[Dict] = []
    for i, fp in enumerate(event_files, start=1):
        row = summarize_event(fp, precip_scale, cfg)
        if row is not None:
            rows.append(row)
        if i % cfg.progress_every_events == 0 or i == len(event_files):
            print(f"[INFO] Summarized {i:,}/{len(event_files):,} event files | kept={len(rows):,}")
    return pd.DataFrame(rows).sort_values(["year", "event_uid"]).reset_index(drop=True)


def build_monthly_inventory(root: Path, required_columns: Sequence[str]) -> Dict[Tuple[int, int], Path]:
    inv: Dict[Tuple[int, int], Path] = {}
    for fp in sorted(root.rglob("*")):
        if not fp.is_file():
            continue
        m = YM_RE.search(fp.name)
        if not m:
            continue
        try:
            cols = [c.strip() for c in pd.read_csv(fp, nrows=0).columns]
        except Exception:
            continue
        if not set(required_columns).issubset(cols):
            continue
        inv[(int(m.group(1)), int(m.group(2)))] = fp
    if not inv:
        raise FileNotFoundError(f"No monthly CSV files with required columns {required_columns} found under {root}")
    return inv


@lru_cache(maxsize=256)
def read_month_csv(path_str: str) -> pd.DataFrame:
    fp = Path(path_str)
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    if "date" not in df.columns or "longitude" not in df.columns or "latitude" not in df.columns:
        raise KeyError(f"CSV missing one of date/longitude/latitude: {fp}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["longitude"] = normalize_lon_values(to_num(df["longitude"]).values)
    df["latitude"] = to_num(df["latitude"]).values
    for c in df.columns:
        if c != "date":
            try:
                df[c] = to_num(df[c]).values
            except Exception:
                pass
    return df


def make_da_from_pivot(df: pd.DataFrame, value_col: str) -> xr.DataArray:
    grid = df.pivot_table(index="latitude", columns="longitude", values=value_col, aggfunc="mean")
    grid = grid.sort_index().sort_index(axis=1)
    return xr.DataArray(grid.to_numpy(dtype=float), coords={"lat": grid.index.to_numpy(dtype=float), "lon": grid.columns.to_numpy(dtype=float)}, dims=("lat", "lon"), name=value_col)


def get_date_field_from_inventory(inv: Dict[Tuple[int, int], Path], date: pd.Timestamp, value_col: str, cfg: Config,
                                  lon_target: Optional[np.ndarray] = None, lat_target: Optional[np.ndarray] = None) -> xr.DataArray:
    key = (int(date.year), int(date.month))
    if key not in inv:
        raise KeyError(f"No monthly CSV available for {key}")
    df = read_month_csv(str(inv[key]))
    if value_col not in df.columns:
        raise KeyError(f"CSV missing value column {value_col}: {inv[key]}")
    sub = df.loc[df["date"] == date].copy()
    if sub.empty:
        raise KeyError(f"No rows for {date.date()} in {inv[key]}")
    sub = sub.loc[(sub["longitude"] >= cfg.box_lon_min) & (sub["longitude"] <= cfg.box_lon_max) &
                  (sub["latitude"] >= cfg.box_lat_min) & (sub["latitude"] <= cfg.box_lat_max)].copy()
    if sub.empty:
        raise KeyError(f"No {value_col} data in Northwest box on {date.date()} from {inv[key]}")
    da = make_da_from_pivot(sub, value_col)
    if lon_target is not None and lat_target is not None:
        da = da.interp(lat=lat_target, lon=lon_target, method="nearest")
    return da


@dataclass
class CompositeAccumulator:
    lon: Optional[np.ndarray] = None
    lat: Optional[np.ndarray] = None
    sums: Optional[np.ndarray] = None
    counts: Optional[np.ndarray] = None

    def add(self, sm: xr.DataArray, bowen: xr.DataArray, rn: xr.DataArray) -> None:
        stack = np.stack([
            np.asarray(sm.values, dtype=float),
            np.asarray(bowen.values, dtype=float),
            np.asarray(rn.values, dtype=float),
        ], axis=0)
        if self.sums is None:
            self.lon = np.asarray(sm["lon"].values, dtype=float)
            self.lat = np.asarray(sm["lat"].values, dtype=float)
            self.sums = np.zeros_like(stack, dtype=float)
            self.counts = np.zeros_like(stack, dtype=float)
        mask = np.isfinite(stack)
        self.sums[mask] += stack[mask]
        self.counts[mask] += 1.0

    def mean_stack(self) -> np.ndarray:
        if self.sums is None or self.counts is None:
            raise RuntimeError("Accumulator is empty")
        return self.sums / np.where(self.counts > 0, self.counts, np.nan)


def build_nw_all_events_land_tables(cfg: Config) -> None:
    summary = None
    if cfg.prefer_existing_nw_ridging_tables:
        summary = load_existing_nw_ridging_tables(cfg)
        if summary is not None:
            print("[INFO] Reusing existing Northwest ridging tables for Northwest event catalog")
    if summary is None and cfg.prefer_existing_result3_tables:
        summary = load_existing_result3_summary(cfg)
        if summary is not None:
            print("[INFO] Reusing existing Result 3 v3 summary for Northwest event catalog")
    if summary is None:
        summary = rebuild_event_catalog(cfg)
        print("[INFO] Existing Northwest / Result 3 tables not found; rebuilt Northwest event catalog from raw event files")

    summary = summary.copy()
    if "climate_region" not in summary.columns or "event_end" not in summary.columns:
        raise KeyError("Summary table must contain climate_region and event_end")
    summary["climate_region"] = summary["climate_region"].astype(str)
    summary["event_end"] = pd.to_datetime(summary["event_end"], errors="coerce")
    nw_catalog = summary.loc[summary["climate_region"] == cfg.target_region].copy().sort_values(["year", "event_uid"])
    if nw_catalog.empty:
        raise RuntimeError("No Northwest events found in the event catalog.")
    if "no_recovery" not in nw_catalog.columns and "recovered_by_day10" in nw_catalog.columns:
        nw_catalog["no_recovery"] = 1 - to_num(nw_catalog["recovered_by_day10"]).fillna(0).astype(int)
    if "recovered_by_day10" not in nw_catalog.columns and "no_recovery" in nw_catalog.columns:
        nw_catalog["recovered_by_day10"] = 1 - to_num(nw_catalog["no_recovery"]).fillna(0).astype(int)
    nw_catalog["group"] = np.where(to_num(nw_catalog["no_recovery"]).fillna(0).astype(int) == 1, "Unrecovered", "Recovered")
    nw_catalog.to_csv(cfg.out_dir / "nw_all_events_land_event_catalog.csv", index=False, encoding="utf-8-sig")
    nw_catalog.groupby("group", as_index=False).size().rename(columns={"size": "n_events"}).to_csv(cfg.out_dir / "nw_all_events_land_group_counts.csv", index=False, encoding="utf-8-sig")

    energy_inv = build_monthly_inventory(cfg.energy_root, ["date", "H", "LE", "Rn", "longitude", "latitude"])
    sm_inv = build_monthly_inventory(cfg.sm_root, ["date", "soil_moist", "longitude", "latitude"])
    print(f"[INFO] Energy monthly CSVs discovered: {len(energy_inv):,}")
    print(f"[INFO] Soil-moisture monthly CSVs discovered: {len(sm_inv):,}")

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
            sm_da = get_date_field_from_inventory(sm_inv, date, "soil_moist", cfg)
            lon_target = np.asarray(sm_da["lon"].values, dtype=float)
            lat_target = np.asarray(sm_da["lat"].values, dtype=float)
            h_da = get_date_field_from_inventory(energy_inv, date, "H", cfg, lon_target=lon_target, lat_target=lat_target)
            le_da = get_date_field_from_inventory(energy_inv, date, "LE", cfg, lon_target=lon_target, lat_target=lat_target)
            rn_da = get_date_field_from_inventory(energy_inv, date, "Rn", cfg, lon_target=lon_target, lat_target=lat_target)
            sm_vals = np.asarray(sm_da.values, dtype=float)
            h_vals = np.asarray(h_da.values, dtype=float)
            le_vals = np.asarray(le_da.values, dtype=float)
            rn_vals = np.asarray(rn_da.values, dtype=float)
            bowen_vals = np.full_like(h_vals, np.nan, dtype=float)
            valid = np.isfinite(h_vals) & np.isfinite(le_vals) & (np.abs(le_vals) > 1e-12)
            bowen_vals[valid] = h_vals[valid] / le_vals[valid]
            bowen_vals[~np.isfinite(bowen_vals)] = np.nan
            mean_rows.append({
                "event_uid": event_uid,
                "group": group,
                "tau": tau,
                "date": date,
                "soil_moist_mean": float(np.nanmean(sm_vals)),
                "bowen_ratio_mean": float(np.nanmean(bowen_vals)),
                "rn_mean": float(np.nanmean(rn_vals)),
                "soil_moist_std": float(np.nanstd(sm_vals)),
                "bowen_ratio_std": float(np.nanstd(bowen_vals)),
                "rn_std": float(np.nanstd(rn_vals)),
                "n_grid": int(np.isfinite(sm_vals).sum()),
            })
            key = (group, tau)
            if key not in composite:
                composite[key] = CompositeAccumulator()
            composite[key].add(
                xr.DataArray(sm_vals, coords={"lat": lat_target, "lon": lon_target}, dims=("lat", "lon"), name="soil_moist"),
                xr.DataArray(bowen_vals, coords={"lat": lat_target, "lon": lon_target}, dims=("lat", "lon"), name="bowen_ratio"),
                xr.DataArray(rn_vals, coords={"lat": lat_target, "lon": lon_target}, dims=("lat", "lon"), name="rn"),
            )
        except Exception as e:
            n_fail += 1
            warnings.warn(f"Failed for {event_uid} tau={tau} date={date.date()}: {e}")
        if i % cfg.progress_every_dates == 0 or i == len(all_dates):
            print(f"[INFO] Extracted {i:,}/{len(all_dates):,} all-event event-date slices")

    means = pd.DataFrame(mean_rows).sort_values(["group", "event_uid", "tau"]).reset_index(drop=True)
    means.to_csv(cfg.out_dir / "nw_all_events_land_tau_event_box_means.csv", index=False, encoding="utf-8-sig")

    comp_rows: List[Dict] = []
    comp_summary_rows: List[Dict] = []
    for (group, tau), acc in sorted(composite.items(), key=lambda x: (x[0][0], x[0][1])):
        try:
            mean_stack = acc.mean_stack()
        except RuntimeError:
            continue
        sm, bowen, rn = mean_stack[0], mean_stack[1], mean_stack[2]
        comp_summary_rows.append({
            "group": group,
            "tau": tau,
            "soil_moist_mean": float(np.nanmean(sm)),
            "bowen_ratio_mean": float(np.nanmean(bowen)),
            "rn_mean": float(np.nanmean(rn)),
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
                    "soil_moist": float(sm[ii, jj]) if np.isfinite(sm[ii, jj]) else np.nan,
                    "bowen_ratio": float(bowen[ii, jj]) if np.isfinite(bowen[ii, jj]) else np.nan,
                    "rn": float(rn[ii, jj]) if np.isfinite(rn[ii, jj]) else np.nan,
                })
    pd.DataFrame(comp_rows).to_csv(cfg.out_dir / "nw_all_events_land_composite_grid_long.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comp_summary_rows).to_csv(cfg.out_dir / "nw_all_events_land_composite_tau_summary.csv", index=False, encoding="utf-8-sig")

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
        {"key": "used_existing_nw_ridging_tables", "value": int(cfg.prefer_existing_nw_ridging_tables and (load_existing_nw_ridging_tables(cfg) is not None))},
    ])
    info.to_csv(cfg.out_dir / "nw_all_events_land_run_info.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] Northwest all-events land-memory mechanism tables written to: {cfg.out_dir}")


def main(cfg: Config = CFG) -> None:
    build_nw_all_events_land_tables(cfg)


if __name__ == "__main__":
    main(CFG)
