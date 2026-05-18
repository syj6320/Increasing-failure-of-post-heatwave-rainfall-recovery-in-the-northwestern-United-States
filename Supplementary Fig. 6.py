# -*- coding: utf-8 -*-
"""
Supplementary IMERG precipitation-product sensitivity test
for post-heatwave rainfall recovery, 2000-2024.

Purpose
-------
This script repeats the post-event rainfall-outcome calculation using
two precipitation products:
    1) ERA5 precipitation column:        precipitation
    2) IMERG precipitation column:       imerg_precipitation

Importantly, this is NOT a fully independent event reconstruction.
The event timing, event footprint, heatwave definition and environmental
diagnostics are retained from the ERA5/ERA5-Land event catalogue.

Recommended manuscript wording:
    precipitation-product sensitivity test
    independent precipitation-product check
    post-2000 rainfall-outcome robustness test

Avoid:
    independent validation
    independent replication
    external confirmation

Author: ChatGPT
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import os
import re
import gc
import math
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# =============================================================================
# 0. Configuration
# =============================================================================

@dataclass
class Config:
    # -------------------------------------------------------------------------
    # Input and output paths
    # -------------------------------------------------------------------------
    input_root: str = r"E:\temp_events_ERA5_S1S6_Nature所有数据版本\zu_added_IMERG_2000_2024_6digit"
    output_dir: str = r"E:\temp_events_ERA5_S1S6_Nature所有数据版本\IMERG_sensitivity_outputs"

    # -------------------------------------------------------------------------
    # Analysis years
    # -------------------------------------------------------------------------
    start_year: int = 2000
    end_year: int = 2024

    # -------------------------------------------------------------------------
    # Recovery definition
    # -------------------------------------------------------------------------
    post_lag_min: int = 1
    post_lag_max: int = 10
    rain_threshold_mm: float = 1.0
    footprint_fraction_threshold: float = 0.25

    # Minimum daily footprint coverage required for a lag day to be counted.
    # For product sensitivity, 0.80 is a defensible compromise.
    min_daily_coverage: float = 0.80

    # Minimum number of valid post-event days required for an event-product
    # outcome to be considered valid.
    min_valid_post_days: int = 8

    # Use only paired events for product comparison:
    # event must have valid ERA5 and valid IMERG post-event outcome.
    use_paired_sample: bool = True

    # For IMERG, require imerg_matched == 1 when the column exists.
    require_imerg_matched: bool = True

    # -------------------------------------------------------------------------
    # Precipitation columns
    # -------------------------------------------------------------------------
    era5_precip_col: str = "precipitation"
    imerg_precip_col: str = "imerg_precipitation"

    # Unit conversion.
    # Set to None to infer automatically.
    # If ERA5 precipitation is in metres, use 1000.0.
    # If precipitation is already in mm, use 1.0.
    era5_precip_factor: Optional[float] = None
    imerg_precip_factor: Optional[float] = None

    # Number of files used to infer units.
    unit_inference_max_files: int = 600

    # -------------------------------------------------------------------------
    # Region assignment
    # -------------------------------------------------------------------------
    # If cartopy/shapely are available, the code assigns events to regions
    # from Natural Earth U.S. state polygons using the majority of heat-core
    # footprint cells. If unavailable, it falls back to an approximate
    # longitude-latitude rule.
    region_of_interest: str = "Northwest"

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    rolling_window_years: int = 5
    bootstrap_n: int = 2000
    random_seed: int = 42
    dpi: int = 600

    # Figure labels
    figure_basename: str = "Supp_Fig_IMERG_precipitation_product_sensitivity"


CFG = Config()


# =============================================================================
# 1. Utility functions
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def discover_event_files(root: str | Path, start_year: int, end_year: int) -> List[Path]:
    root = Path(root)
    files: List[Path] = []

    for year in range(start_year, end_year + 1):
        ydir = root / str(year)
        if not ydir.exists():
            warnings.warn(f"[WARN] Year folder not found: {ydir}")
            continue

        files.extend(sorted(ydir.glob("event_*_window.csv")))

    return files


def parse_year_eventid_from_filename(fp: Path) -> Tuple[Optional[int], Optional[str]]:
    m = re.search(r"event_(\d{4})_(\d+)_window\.csv$", fp.name)
    if not m:
        return None, None
    return int(m.group(1)), f"{m.group(1)}_{m.group(2)}"


def smart_read_csv(fp: Path, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Read CSV or tab-delimited CSV robustly.

    Many event files use .csv suffix but may be comma- or tab-separated.
    """
    usecols_set = set(usecols) if usecols is not None else None

    def usecols_func(c):
        return c in usecols_set if usecols_set is not None else True

    # Try comma first, then tab, then auto-sniff.
    attempts = [
        dict(sep=",", engine="c"),
        dict(sep="\t", engine="c"),
        dict(sep=None, engine="python"),
    ]

    last_err = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(
                fp,
                usecols=usecols_func if usecols is not None else None,
                **kwargs,
            )
            if df.shape[1] > 1:
                return df
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to read {fp}. Last error: {last_err}")


def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def infer_precip_factor(
    files: List[Path],
    col: str,
    max_files: int,
    default_factor: float = 1.0,
) -> float:
    """
    Infer whether precipitation is in metres or millimetres.

    Heuristic:
      - If q99 < 0.2 and max < 5, values are likely metres for ERA5 daily total.
      - Otherwise assume mm.

    Because very dry samples can create ambiguity, this function only infers.
    Users can override by setting CFG.era5_precip_factor or CFG.imerg_precip_factor.
    """
    vals = []

    sample_files = files[:max_files]
    for k, fp in enumerate(sample_files, 1):
        try:
            df = smart_read_csv(fp, usecols=[col])
            if col not in df.columns:
                continue
            x = to_numeric_safe(df[col]).replace([np.inf, -np.inf], np.nan).dropna()
            x = x[x >= 0]
            if len(x) == 0:
                continue

            # Keep only a limited sample to avoid memory growth.
            if len(x) > 2000:
                x = x.sample(2000, random_state=123)
            vals.append(x.to_numpy())

        except Exception:
            continue

    if not vals:
        warnings.warn(f"[WARN] Could not infer units for column {col}; using factor {default_factor}.")
        return default_factor

    arr = np.concatenate(vals)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]

    if len(arr) == 0:
        warnings.warn(f"[WARN] Empty precipitation sample for column {col}; using factor {default_factor}.")
        return default_factor

    q95 = float(np.nanquantile(arr, 0.95))
    q99 = float(np.nanquantile(arr, 0.99))
    mx = float(np.nanmax(arr))

    # Conservative inference.
    if q99 < 0.2 and mx < 5.0:
        factor = 1000.0
        unit_note = "interpreted_as_metres_to_mm"
    else:
        factor = 1.0
        unit_note = "interpreted_as_mm"

    print(
        f"[INFO] Unit inference for {col}: "
        f"q95={q95:.6g}, q99={q99:.6g}, max={mx:.6g}, "
        f"factor={factor} ({unit_note})"
    )

    return factor


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 95.0,
) -> Tuple[float, float, float]:
    """
    Bootstrap mean and CI for 0/1 or continuous values.
    Returns mean, lower, upper.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(x))
    if len(x) < 2:
        return mean, np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot = x[idx].mean(axis=1)

    alpha = (100.0 - ci) / 2.0
    lo, hi = np.nanpercentile(boot, [alpha, 100.0 - alpha])
    return mean, float(lo), float(hi)


def trend_pp_per_decade(year: np.ndarray, percent_values: np.ndarray) -> float:
    """
    Linear trend in percentage points per decade.
    """
    y = np.asarray(year, dtype=float)
    v = np.asarray(percent_values, dtype=float)
    m = np.isfinite(y) & np.isfinite(v)

    if m.sum() < 5:
        return np.nan

    slope_per_year = np.polyfit(y[m], v[m], 1)[0]
    return float(slope_per_year * 10.0)


# =============================================================================
# 2. Region assignment
# =============================================================================

REGION_BY_STATE = {
    # Northwest
    "Washington": "Northwest",
    "Oregon": "Northwest",
    "Idaho": "Northwest",

    # Southwest
    "California": "Southwest",
    "Nevada": "Southwest",
    "Utah": "Southwest",
    "Arizona": "Southwest",
    "New Mexico": "Southwest",
    "Colorado": "Southwest",

    # Northern Great Plains
    "Montana": "Northern Great Plains",
    "Wyoming": "Northern Great Plains",
    "North Dakota": "Northern Great Plains",
    "South Dakota": "Northern Great Plains",
    "Nebraska": "Northern Great Plains",

    # Southern Great Plains
    "Kansas": "Southern Great Plains",
    "Oklahoma": "Southern Great Plains",
    "Texas": "Southern Great Plains",

    # Midwest
    "Minnesota": "Midwest",
    "Iowa": "Midwest",
    "Missouri": "Midwest",
    "Wisconsin": "Midwest",
    "Illinois": "Midwest",
    "Michigan": "Midwest",
    "Indiana": "Midwest",
    "Ohio": "Midwest",

    # Northeast
    "Maine": "Northeast",
    "New Hampshire": "Northeast",
    "Vermont": "Northeast",
    "Massachusetts": "Northeast",
    "Rhode Island": "Northeast",
    "Connecticut": "Northeast",
    "New York": "Northeast",
    "New Jersey": "Northeast",
    "Pennsylvania": "Northeast",

    # Southeast
    "Delaware": "Southeast",
    "Maryland": "Southeast",
    "District of Columbia": "Southeast",
    "Virginia": "Southeast",
    "West Virginia": "Southeast",
    "Kentucky": "Southeast",
    "Tennessee": "Southeast",
    "North Carolina": "Southeast",
    "South Carolina": "Southeast",
    "Georgia": "Southeast",
    "Florida": "Southeast",
    "Alabama": "Southeast",
    "Mississippi": "Southeast",
    "Arkansas": "Southeast",
    "Louisiana": "Southeast",
}


class RegionResolver:
    """
    Assign region from lon-lat.

    Preferred:
      Natural Earth U.S. state polygons via cartopy + shapely.

    Fallback:
      approximate lon-lat rule. This fallback is acceptable for quick
      sensitivity checks, but state-polygon assignment is preferred for
      manuscript-quality regional statistics.
    """

    def __init__(self):
        self.cache: Dict[Tuple[float, float], str] = {}
        self.use_polygon = False
        self.state_geoms = []

        try:
            import cartopy.io.shapereader as shpreader
            from shapely.geometry import Point  # noqa: F401

            shp = shpreader.natural_earth(
                resolution="50m",
                category="cultural",
                name="admin_1_states_provinces",
            )
            reader = shpreader.Reader(shp)

            for rec in reader.records():
                attrs = rec.attributes
                admin = attrs.get("admin", "")
                country = attrs.get("iso_a2", "") or attrs.get("adm0_a3", "")

                if admin != "United States of America" and country not in ("US", "USA"):
                    continue

                name = attrs.get("name", "")
                if name in REGION_BY_STATE:
                    self.state_geoms.append((name, REGION_BY_STATE[name], rec.geometry))

            if len(self.state_geoms) > 0:
                self.use_polygon = True
                print(f"[INFO] Region assignment: using Natural Earth state polygons ({len(self.state_geoms)} states).")
            else:
                print("[WARN] Natural Earth state polygons loaded, but no U.S. states found. Using fallback.")

        except Exception as e:
            print(f"[WARN] cartopy/shapely polygon assignment unavailable: {e}")
            print("[WARN] Region assignment will use approximate lon-lat fallback.")

    def region_from_point(self, lon: float, lat: float) -> str:
        if not np.isfinite(lon) or not np.isfinite(lat):
            return "Unknown"

        key = (round(float(lon), 4), round(float(lat), 4))
        if key in self.cache:
            return self.cache[key]

        if self.use_polygon:
            try:
                from shapely.geometry import Point
                p = Point(float(lon), float(lat))
                for state_name, region, geom in self.state_geoms:
                    if geom.contains(p) or geom.touches(p):
                        self.cache[key] = region
                        return region
            except Exception:
                pass

        region = self.region_from_point_fallback(lon, lat)
        self.cache[key] = region
        return region

    @staticmethod
    def region_from_point_fallback(lon: float, lat: float) -> str:
        """
        Approximate fallback region rule.

        This is not as accurate as state polygons. It is included only
        to keep the script runnable if cartopy/shapely are unavailable.
        """
        lon = float(lon)
        lat = float(lat)

        # Very approximate CONUS region assignment.
        if lon <= -116 and lat >= 41:
            return "Northwest"
        if -116 < lon <= -111 and lat >= 42:
            return "Northwest"

        if lon <= -103 and lat < 41:
            return "Southwest"

        if -111 < lon <= -96 and lat >= 41:
            return "Northern Great Plains"

        if -103 < lon <= -94 and lat < 41:
            return "Southern Great Plains"

        if -96 < lon <= -82 and lat >= 37:
            return "Midwest"

        if lon > -82 and lat >= 37:
            return "Northeast"

        if lon > -94 and lat < 37:
            return "Southeast"

        return "Unknown"

    def region_for_event(self, heat_df: pd.DataFrame) -> str:
        """
        Assign event region by majority of heat-core footprint cells.
        """
        if heat_df.empty:
            return "Unknown"

        coords = heat_df[["longitude", "latitude"]].dropna().drop_duplicates()

        if coords.empty:
            return "Unknown"

        # Limit extremely large events for speed, while retaining broad footprint.
        if len(coords) > 500:
            coords = coords.sample(500, random_state=123)

        regs = []
        for lon, lat in coords[["longitude", "latitude"]].itertuples(index=False):
            regs.append(self.region_from_point(lon, lat))

        if len(regs) == 0:
            return "Unknown"

        s = pd.Series(regs)
        s = s[s != "Unknown"]

        if len(s) == 0:
            return "Unknown"

        return str(s.value_counts().idxmax())


# =============================================================================
# 3. Event-level recovery calculation
# =============================================================================

REQUIRED_COLS = [
    "date",
    "year",
    "longitude",
    "latitude",
    "coord_key",
    "event_id",
    "event_start",
    "event_end",
    "lag_day_event",
    "is_heat_period_event",
    "is_post_event_0_10_nominal",
    "imerg_matched",
    "precipitation",
    "imerg_precipitation",
]


def make_coord_key_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if "coord_key" not in df.columns:
        df["coord_key"] = (
            df["longitude"].round(4).astype(str)
            + "_"
            + df["latitude"].round(4).astype(str)
        )
    return df


def product_outcome_for_event(
    df: pd.DataFrame,
    precip_col: str,
    precip_factor: float,
    product_name: str,
    cfg: Config,
) -> Dict[str, float | int | bool]:
    """
    Compute event-level recovery outcome for one precipitation product.

    Returns:
        valid
        recovered
        no_recovery
        first_recovery_lag
        mean_rain_frac
        max_rain_frac
        cumulative_rain_frac
        n_valid_days
        min_daily_coverage
    """

    if precip_col not in df.columns:
        return {
            f"{product_name}_valid": False,
            f"{product_name}_recovered": np.nan,
            f"{product_name}_no_recovery": np.nan,
            f"{product_name}_first_recovery_lag": np.nan,
            f"{product_name}_mean_rain_frac": np.nan,
            f"{product_name}_max_rain_frac": np.nan,
            f"{product_name}_cumulative_rain_frac": np.nan,
            f"{product_name}_n_valid_days": 0,
            f"{product_name}_min_daily_coverage": np.nan,
        }

    d = df.copy()

    d["lag_day_event"] = to_numeric_safe(d["lag_day_event"])
    d[precip_col] = to_numeric_safe(d[precip_col]) * precip_factor

    # Post-event days 1-10.
    post_mask = (
        (d["lag_day_event"] >= cfg.post_lag_min)
        & (d["lag_day_event"] <= cfg.post_lag_max)
    )

    if "is_post_event_0_10_nominal" in d.columns:
        tmp = to_numeric_safe(d["is_post_event_0_10_nominal"])
        post_mask = post_mask & (tmp == 1)

    # For IMERG, keep matched rows if requested.
    if product_name.lower() == "imerg" and cfg.require_imerg_matched and "imerg_matched" in d.columns:
        matched = to_numeric_safe(d["imerg_matched"])
        post_mask = post_mask & (matched == 1)

    post = d.loc[post_mask, ["lag_day_event", "coord_key", precip_col]].copy()
    post = post.dropna(subset=["lag_day_event", "coord_key", precip_col])
    post = post[post[precip_col] >= 0]

    # Heatwave footprint from original heatwave core.
    heat_mask = pd.Series(False, index=d.index)
    if "is_heat_period_event" in d.columns:
        heat_mask = to_numeric_safe(d["is_heat_period_event"]) == 1

    # Fallback: heat-core days often have lag_day_event <= 0.
    if heat_mask.sum() == 0:
        heat_mask = d["lag_day_event"] <= 0

    heat = d.loc[heat_mask, ["coord_key"]].dropna().drop_duplicates()
    n_foot = int(heat["coord_key"].nunique())

    if n_foot <= 0:
        return {
            f"{product_name}_valid": False,
            f"{product_name}_recovered": np.nan,
            f"{product_name}_no_recovery": np.nan,
            f"{product_name}_first_recovery_lag": np.nan,
            f"{product_name}_mean_rain_frac": np.nan,
            f"{product_name}_max_rain_frac": np.nan,
            f"{product_name}_cumulative_rain_frac": np.nan,
            f"{product_name}_n_valid_days": 0,
            f"{product_name}_min_daily_coverage": np.nan,
        }

    daily_records = []

    for lag in range(cfg.post_lag_min, cfg.post_lag_max + 1):
        g = post.loc[post["lag_day_event"] == lag, ["coord_key", precip_col]].copy()

        if g.empty:
            daily_records.append(
                dict(
                    lag=lag,
                    n_obs=0,
                    coverage=0.0,
                    rainy_cells=0,
                    rainy_frac=np.nan,
                    valid_day=False,
                )
            )
            continue

        # One row per coord. If duplicates exist, keep maximum daily precip.
        g = g.groupby("coord_key", as_index=False)[precip_col].max()

        n_obs = int(g["coord_key"].nunique())
        coverage = n_obs / n_foot

        valid_day = coverage >= cfg.min_daily_coverage

        if valid_day:
            rainy_cells = int((g[precip_col] >= cfg.rain_threshold_mm).sum())
            rainy_frac = rainy_cells / n_foot
        else:
            rainy_cells = 0
            rainy_frac = np.nan

        daily_records.append(
            dict(
                lag=lag,
                n_obs=n_obs,
                coverage=coverage,
                rainy_cells=rainy_cells,
                rainy_frac=rainy_frac,
                valid_day=valid_day,
            )
        )

    daily = pd.DataFrame(daily_records)

    valid_daily = daily[daily["valid_day"]].copy()
    n_valid_days = int(len(valid_daily))
    min_daily_coverage = float(valid_daily["coverage"].min()) if n_valid_days > 0 else np.nan

    if n_valid_days < cfg.min_valid_post_days:
        valid = False
        recovered = np.nan
        no_recovery = np.nan
        first_lag = np.nan
        mean_rain_frac = np.nan
        max_rain_frac = np.nan
        cumulative_rain_frac = np.nan
    else:
        valid = True
        valid_daily["meets_recovery"] = (
            valid_daily["rainy_frac"] >= cfg.footprint_fraction_threshold
        )

        recovered_bool = bool(valid_daily["meets_recovery"].any())
        recovered = int(recovered_bool)
        no_recovery = int(not recovered_bool)

        if recovered_bool:
            first_lag = int(valid_daily.loc[valid_daily["meets_recovery"], "lag"].min())
        else:
            first_lag = np.nan

        mean_rain_frac = float(valid_daily["rainy_frac"].mean())
        max_rain_frac = float(valid_daily["rainy_frac"].max())
        cumulative_rain_frac = float(valid_daily["rainy_frac"].sum())

    return {
        f"{product_name}_valid": valid,
        f"{product_name}_recovered": recovered,
        f"{product_name}_no_recovery": no_recovery,
        f"{product_name}_first_recovery_lag": first_lag,
        f"{product_name}_mean_rain_frac": mean_rain_frac,
        f"{product_name}_max_rain_frac": max_rain_frac,
        f"{product_name}_cumulative_rain_frac": cumulative_rain_frac,
        f"{product_name}_n_valid_days": n_valid_days,
        f"{product_name}_min_daily_coverage": min_daily_coverage,
    }


def summarize_event_file(
    fp: Path,
    cfg: Config,
    region_resolver: RegionResolver,
    era5_factor: float,
    imerg_factor: float,
) -> Optional[Dict[str, object]]:

    try:
        df = smart_read_csv(fp, usecols=REQUIRED_COLS)
    except Exception as e:
        warnings.warn(f"[WARN] Failed to read {fp}: {e}")
        return None

    if df.empty:
        return None

    # Ensure core columns.
    for c in ["longitude", "latitude", "lag_day_event"]:
        if c not in df.columns:
            warnings.warn(f"[WARN] Missing required column {c} in {fp}")
            return None

    df = make_coord_key_if_needed(df)

    # Convert numeric core.
    df["longitude"] = to_numeric_safe(df["longitude"])
    df["latitude"] = to_numeric_safe(df["latitude"])
    df["lag_day_event"] = to_numeric_safe(df["lag_day_event"])

    file_year, file_event = parse_year_eventid_from_filename(fp)

    if "year" in df.columns:
        yvals = to_numeric_safe(df["year"]).dropna()
        event_year = int(yvals.iloc[0]) if len(yvals) else file_year
    else:
        event_year = file_year

    if event_year is None:
        return None

    if event_year < cfg.start_year or event_year > cfg.end_year:
        return None

    if "event_id" in df.columns:
        eid = str(df["event_id"].dropna().iloc[0]) if df["event_id"].notna().any() else file_event
    else:
        eid = file_event

    event_uid = f"{event_year}_{eid}" if eid is not None else fp.stem

    # Heat-core footprint.
    heat_mask = pd.Series(False, index=df.index)
    if "is_heat_period_event" in df.columns:
        heat_mask = to_numeric_safe(df["is_heat_period_event"]) == 1
    if heat_mask.sum() == 0:
        heat_mask = df["lag_day_event"] <= 0

    heat_df = df.loc[heat_mask, ["longitude", "latitude", "coord_key"]].dropna()
    n_footprint = int(heat_df["coord_key"].nunique()) if not heat_df.empty else 0

    if n_footprint <= 0:
        return None

    centroid_lon = float(heat_df["longitude"].mean())
    centroid_lat = float(heat_df["latitude"].mean())

    region = region_resolver.region_for_event(heat_df)

    out: Dict[str, object] = {
        "event_uid": event_uid,
        "event_id": eid,
        "year": event_year,
        "file": str(fp),
        "n_footprint_cells": n_footprint,
        "centroid_lon": centroid_lon,
        "centroid_lat": centroid_lat,
        "region": region,
    }

    # Event dates if available.
    for c in ["event_start", "event_end"]:
        if c in df.columns and df[c].notna().any():
            out[c] = str(df[c].dropna().iloc[0])
        else:
            out[c] = ""

    era5 = product_outcome_for_event(
        df=df,
        precip_col=cfg.era5_precip_col,
        precip_factor=era5_factor,
        product_name="era5",
        cfg=cfg,
    )

    imerg = product_outcome_for_event(
        df=df,
        precip_col=cfg.imerg_precip_col,
        precip_factor=imerg_factor,
        product_name="imerg",
        cfg=cfg,
    )

    out.update(era5)
    out.update(imerg)

    return out


def build_event_summary(cfg: Config) -> pd.DataFrame:
    files = discover_event_files(cfg.input_root, cfg.start_year, cfg.end_year)
    print(f"[INFO] Event files discovered: {len(files):,}")

    if len(files) == 0:
        raise RuntimeError("No event files discovered. Check CFG.input_root and year folders.")

    # Infer precipitation units.
    if cfg.era5_precip_factor is None:
        era5_factor = infer_precip_factor(
            files, cfg.era5_precip_col, cfg.unit_inference_max_files, default_factor=1000.0
        )
    else:
        era5_factor = float(cfg.era5_precip_factor)
        print(f"[INFO] ERA5 precipitation factor set by user: {era5_factor}")

    if cfg.imerg_precip_factor is None:
        imerg_factor = infer_precip_factor(
            files, cfg.imerg_precip_col, cfg.unit_inference_max_files, default_factor=1.0
        )
    else:
        imerg_factor = float(cfg.imerg_precip_factor)
        print(f"[INFO] IMERG precipitation factor set by user: {imerg_factor}")

    region_resolver = RegionResolver()

    records = []
    for i, fp in enumerate(files, 1):
        rec = summarize_event_file(
            fp=fp,
            cfg=cfg,
            region_resolver=region_resolver,
            era5_factor=era5_factor,
            imerg_factor=imerg_factor,
        )
        if rec is not None:
            records.append(rec)

        if i % 250 == 0 or i == len(files):
            print(f"[INFO] Processed {i:,}/{len(files):,} files | kept events = {len(records):,}")

        if i % 1000 == 0:
            gc.collect()

    ev = pd.DataFrame(records)

    if ev.empty:
        raise RuntimeError("No event summaries were created.")

    # Paired sample indicator.
    ev["paired_valid"] = ev["era5_valid"].astype(bool) & ev["imerg_valid"].astype(bool)

    print("\n[INFO] Event-level summary")
    print(f"  Total retained events       : {len(ev):,}")
    print(f"  ERA5-valid events           : {int(ev['era5_valid'].sum()):,}")
    print(f"  IMERG-valid events          : {int(ev['imerg_valid'].sum()):,}")
    print(f"  Paired-valid events         : {int(ev['paired_valid'].sum()):,}")
    print(f"  Northwest paired-valid      : {int(((ev['region'] == cfg.region_of_interest) & ev['paired_valid']).sum()):,}")

    return ev


# =============================================================================
# 4. Aggregation and plotting
# =============================================================================

def long_product_frame(ev: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Convert event-level paired product columns into a long table:
    event_uid, year, region, product, recovered, no_recovery, mean_rain_frac, ...
    """
    base_cols = [
        "event_uid", "year", "region", "n_footprint_cells",
        "centroid_lon", "centroid_lat", "paired_valid",
    ]

    if cfg.use_paired_sample:
        x = ev[ev["paired_valid"]].copy()
    else:
        x = ev.copy()

    frames = []

    for product in ["era5", "imerg"]:
        valid_col = f"{product}_valid"

        tmp = x[base_cols + [
            valid_col,
            f"{product}_recovered",
            f"{product}_no_recovery",
            f"{product}_first_recovery_lag",
            f"{product}_mean_rain_frac",
            f"{product}_max_rain_frac",
            f"{product}_cumulative_rain_frac",
            f"{product}_n_valid_days",
            f"{product}_min_daily_coverage",
        ]].copy()

        tmp = tmp.rename(columns={
            valid_col: "valid",
            f"{product}_recovered": "recovered",
            f"{product}_no_recovery": "no_recovery",
            f"{product}_first_recovery_lag": "first_recovery_lag",
            f"{product}_mean_rain_frac": "mean_rain_frac",
            f"{product}_max_rain_frac": "max_rain_frac",
            f"{product}_cumulative_rain_frac": "cumulative_rain_frac",
            f"{product}_n_valid_days": "n_valid_days",
            f"{product}_min_daily_coverage": "min_daily_coverage",
        })

        tmp["product"] = "ERA5" if product == "era5" else "IMERG"
        tmp = tmp[tmp["valid"].astype(bool)].copy()
        frames.append(tmp)

    long = pd.concat(frames, ignore_index=True)

    for c in ["recovered", "no_recovery", "mean_rain_frac", "max_rain_frac", "cumulative_rain_frac"]:
        long[c] = to_numeric_safe(long[c])

    return long


def annual_summary(long: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Annual summary for CONUS and Northwest.
    """
    rows = []

    for scope_name, mask in {
        "CONUS": pd.Series(True, index=long.index),
        cfg.region_of_interest: long["region"] == cfg.region_of_interest,
    }.items():
        d0 = long[mask].copy()

        for (year, product), g in d0.groupby(["year", "product"]):
            if len(g) == 0:
                continue

            rows.append({
                "scope": scope_name,
                "year": int(year),
                "product": product,
                "n_events": int(len(g)),
                "recovery_fraction_percent": float(g["recovered"].mean() * 100.0),
                "no_recovery_fraction_percent": float(g["no_recovery"].mean() * 100.0),
                "mean_rainy_footprint_fraction_percent": float(g["mean_rain_frac"].mean() * 100.0),
                "max_rainy_footprint_fraction_percent": float(g["max_rain_frac"].mean() * 100.0),
                "cumulative_rainy_footprint_fraction_percent_day": float(g["cumulative_rain_frac"].mean() * 100.0),
            })

    ann = pd.DataFrame(rows)

    if ann.empty:
        return ann

    ann = ann.sort_values(["scope", "product", "year"]).reset_index(drop=True)

    # Rolling smooth.
    smooth_cols = [
        "recovery_fraction_percent",
        "no_recovery_fraction_percent",
        "mean_rainy_footprint_fraction_percent",
        "max_rainy_footprint_fraction_percent",
        "cumulative_rainy_footprint_fraction_percent_day",
    ]

    for col in smooth_cols:
        ann[col + "_smooth"] = np.nan

    for (scope, product), idx in ann.groupby(["scope", "product"]).groups.items():
        sub = ann.loc[idx].sort_values("year")
        for col in smooth_cols:
            ann.loc[sub.index, col + "_smooth"] = (
                sub[col]
                .rolling(cfg.rolling_window_years, center=True, min_periods=3)
                .mean()
                .values
            )

    return ann


def overall_summary(long: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []

    for scope_name, mask in {
        "CONUS": pd.Series(True, index=long.index),
        cfg.region_of_interest: long["region"] == cfg.region_of_interest,
    }.items():
        d0 = long[mask].copy()

        for product, g in d0.groupby("product"):
            if len(g) == 0:
                continue

            rec_mean, rec_lo, rec_hi = bootstrap_mean_ci(
                g["recovered"].to_numpy() * 100.0,
                cfg.bootstrap_n,
                cfg.random_seed,
            )

            norec_mean, norec_lo, norec_hi = bootstrap_mean_ci(
                g["no_recovery"].to_numpy() * 100.0,
                cfg.bootstrap_n,
                cfg.random_seed + 1,
            )

            rain_mean, rain_lo, rain_hi = bootstrap_mean_ci(
                g["mean_rain_frac"].to_numpy() * 100.0,
                cfg.bootstrap_n,
                cfg.random_seed + 2,
            )

            rows.append({
                "scope": scope_name,
                "product": product,
                "n_events": int(len(g)),
                "recovery_fraction_percent": rec_mean,
                "recovery_fraction_ci_low": rec_lo,
                "recovery_fraction_ci_high": rec_hi,
                "no_recovery_fraction_percent": norec_mean,
                "no_recovery_fraction_ci_low": norec_lo,
                "no_recovery_fraction_ci_high": norec_hi,
                "mean_rainy_footprint_fraction_percent": rain_mean,
                "mean_rainy_footprint_fraction_ci_low": rain_lo,
                "mean_rainy_footprint_fraction_ci_high": rain_hi,
            })

    out = pd.DataFrame(rows)

    return out


def add_overall_trends(overall: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    out = overall.copy()

    trend_cols = [
        "recovery_fraction_percent",
        "no_recovery_fraction_percent",
        "mean_rainy_footprint_fraction_percent",
    ]

    for col in trend_cols:
        out[col.replace("_percent", "_trend_pp_decade")] = np.nan

    for i, r in out.iterrows():
        scope = r["scope"]
        product = r["product"]
        sub = ann[(ann["scope"] == scope) & (ann["product"] == product)].copy()

        for col in trend_cols:
            tr = trend_pp_per_decade(sub["year"].to_numpy(), sub[col].to_numpy())
            out.loc[i, col.replace("_percent", "_trend_pp_decade")] = tr

    return out


def plot_supp_figure(
    ann: pd.DataFrame,
    overall: pd.DataFrame,
    cfg: Config,
    output_dir: Path,
) -> None:
    """
    Create a restrained 3-panel supplementary figure.

    a: Post-2000 mean recovery fraction: ERA5 vs IMERG, CONUS vs Northwest
    b: Annual no-recovery fraction trajectories
    c: Mean rainy-footprint fraction: ERA5 vs IMERG, CONUS vs Northwest

    Revision requested:
      - all fonts set to 27 pt;
      - all legends removed from panels and placed below the full figure;
      - legend arranged in two rows;
      - panel b x-axis ticks reduced to avoid overlap;
      - y-axis labels shortened;
      - panel letters moved left of the y-axis label and upward.
    """

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    FONT_SIZE = 27
    PANEL_LETTER_SIZE = 36

    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.6,
    })

    product_colors = {
        "ERA5": "#4C78A8",
        "IMERG": "#F58518",
    }

    scope_markers = {
        "CONUS": "o",
        cfg.region_of_interest: "s",
    }

    # Wider and moderately tall canvas. The bottom margin is deliberately large
    # because all legends are now placed under the panels in two rows.
    fig = plt.figure(figsize=(22.5, 7.2))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1.05, 1.55, 1.10],
        wspace=0.48,
        left=0.075,
        right=0.992,
        top=0.86,
        bottom=0.38,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # -------------------------------------------------------------------------
    # Panel a: mean recovery fraction
    # -------------------------------------------------------------------------
    scopes = ["CONUS", cfg.region_of_interest]
    products = ["ERA5", "IMERG"]

    xbase = np.arange(len(scopes))
    width = 0.34

    for j, product in enumerate(products):
        xs = xbase + (j - 0.5) * width

        means = []
        yerr_low = []
        yerr_high = []

        for scope in scopes:
            r = overall[(overall["scope"] == scope) & (overall["product"] == product)]
            if r.empty:
                means.append(np.nan)
                yerr_low.append(np.nan)
                yerr_high.append(np.nan)
            else:
                rr = r.iloc[0]
                m = rr["recovery_fraction_percent"]
                lo = rr["recovery_fraction_ci_low"]
                hi = rr["recovery_fraction_ci_high"]
                means.append(m)
                yerr_low.append(m - lo)
                yerr_high.append(hi - m)

        ax_a.bar(
            xs,
            means,
            width=width * 0.95,
            color=product_colors[product],
            alpha=0.85,
            edgecolor="black",
            linewidth=1.2,
            label=product,
        )

        ax_a.errorbar(
            xs,
            means,
            yerr=[yerr_low, yerr_high],
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
        )

    ax_a.set_xticks(xbase)
    ax_a.set_xticklabels(["CONUS", "Northwest"], fontsize=FONT_SIZE)
    ax_a.set_ylabel("Recovery (%)", fontsize=FONT_SIZE, labelpad=10)
    ax_a.set_ylim(0, 100)
    ax_a.yaxis.set_major_locator(MaxNLocator(5))

    # Move panel letter to the left of the y-axis title and upward.
    txt_a = ax_a.text(
        -0.36, 1.12, "a",
        transform=ax_a.transAxes,
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )

    # -------------------------------------------------------------------------
    # Panel b: annual no-recovery fraction trajectories
    # -------------------------------------------------------------------------
    for scope in scopes:
        for product in products:
            sub = ann[(ann["scope"] == scope) & (ann["product"] == product)].copy()
            if sub.empty:
                continue

            sub = sub.sort_values("year")

            linestyle = "-" if product == "ERA5" else "--"
            marker = scope_markers[scope]
            alpha = 0.95 if scope == cfg.region_of_interest else 0.60
            lw = 2.6 if scope == cfg.region_of_interest else 1.6

            label = f"{scope}, {product}"

            # Convert pandas Series to NumPy arrays explicitly.
            # This avoids the pandas>=2.x / matplotlib compatibility issue:
            # "Multi-dimensional indexing ... is no longer supported".
            x_year = sub["year"].to_numpy(dtype=float)
            y_norec = sub["no_recovery_fraction_percent_smooth"].to_numpy(dtype=float)

            ax_b.plot(
                x_year,
                y_norec,
                linestyle=linestyle,
                color=product_colors[product],
                linewidth=lw,
                alpha=alpha,
                marker=marker,
                markersize=4.8,
                markevery=4,
                label=label,
            )

    ax_b.set_ylabel("No-recovery (%)", fontsize=FONT_SIZE, labelpad=10)
    ax_b.set_xlabel("Year", fontsize=FONT_SIZE, labelpad=6)
    ax_b.set_xlim(cfg.start_year, cfg.end_year)
    ax_b.set_ylim(0, 100)
    ax_b.yaxis.set_major_locator(MaxNLocator(5))

    # Fewer year labels to avoid overlap at 27 pt.
    ax_b.set_xticks([2000, 2008, 2016, 2024])
    ax_b.set_xticklabels(["2000", "2008", "2016", "2024"], fontsize=FONT_SIZE)

    txt_b = ax_b.text(
        -0.26, 1.12, "b",
        transform=ax_b.transAxes,
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )

    # -------------------------------------------------------------------------
    # Panel c: mean rainy-footprint fraction
    # -------------------------------------------------------------------------
    for j, product in enumerate(products):
        xs = xbase + (j - 0.5) * width

        means = []
        yerr_low = []
        yerr_high = []

        for scope in scopes:
            r = overall[(overall["scope"] == scope) & (overall["product"] == product)]
            if r.empty:
                means.append(np.nan)
                yerr_low.append(np.nan)
                yerr_high.append(np.nan)
            else:
                rr = r.iloc[0]
                m = rr["mean_rainy_footprint_fraction_percent"]
                lo = rr["mean_rainy_footprint_fraction_ci_low"]
                hi = rr["mean_rainy_footprint_fraction_ci_high"]
                means.append(m)
                yerr_low.append(m - lo)
                yerr_high.append(hi - m)

        ax_c.bar(
            xs,
            means,
            width=width * 0.95,
            color=product_colors[product],
            alpha=0.85,
            edgecolor="black",
            linewidth=1.2,
            label=product,
        )

        ax_c.errorbar(
            xs,
            means,
            yerr=[yerr_low, yerr_high],
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
        )

    ax_c.set_xticks(xbase)
    ax_c.set_xticklabels(["CONUS", "Northwest"], fontsize=FONT_SIZE)
    ax_c.set_ylabel("Rainy footprint (%)", fontsize=FONT_SIZE, labelpad=10)
    ax_c.set_ylim(bottom=0)
    ax_c.yaxis.set_major_locator(MaxNLocator(5))

    txt_c = ax_c.text(
        -0.36, 1.12, "c",
        transform=ax_c.transAxes,
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )

    # -------------------------------------------------------------------------
    # Shared axis styling
    # -------------------------------------------------------------------------
    for ax in [ax_a, ax_b, ax_c]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE, length=6, width=1.5)
        ax.grid(axis="y", linewidth=0.7, alpha=0.25)

        ax.xaxis.label.set_size(FONT_SIZE)
        ax.yaxis.label.set_size(FONT_SIZE)

        for tick_label in ax.get_xticklabels():
            tick_label.set_fontsize(FONT_SIZE)
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontsize(FONT_SIZE)

    # -------------------------------------------------------------------------
    # Single bottom legend, arranged into two rows.
    # ncol=3 with six handles gives exactly two legend rows.
    # -------------------------------------------------------------------------
    legend_handles = [
        Patch(facecolor=product_colors["ERA5"], edgecolor="black", linewidth=1.0, alpha=0.85),
        Patch(facecolor=product_colors["IMERG"], edgecolor="black", linewidth=1.0, alpha=0.85),
        Line2D([0], [0], color=product_colors["ERA5"], lw=1.8, linestyle="-",
               marker="o", markersize=5.5, alpha=0.60),
        Line2D([0], [0], color=product_colors["IMERG"], lw=1.8, linestyle="--",
               marker="o", markersize=5.5, alpha=0.60),
        Line2D([0], [0], color=product_colors["ERA5"], lw=2.8, linestyle="-",
               marker="s", markersize=5.5, alpha=0.95),
        Line2D([0], [0], color=product_colors["IMERG"], lw=2.8, linestyle="--",
               marker="s", markersize=5.5, alpha=0.95),
    ]

    legend_labels = [
        "ERA5",
        "IMERG",
        "CONUS, ERA5",
        "CONUS, IMERG",
        "Northwest, ERA5",
        "Northwest, IMERG",
    ]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE,
        handlelength=2.2,
        columnspacing=1.4,
        handletextpad=0.55,
        labelspacing=0.55,
        borderaxespad=0.0,
    )

    # Hard-force all text artists in the entire figure to 27 pt.
    for txt in fig.findobj(match=matplotlib.text.Text):
        txt.set_fontsize(FONT_SIZE)

    # Re-apply panel letter size after the global hard-force.
    for txt_panel in [txt_a, txt_b, txt_c]:
        txt_panel.set_fontsize(PANEL_LETTER_SIZE)
        txt_panel.set_fontweight("bold")

    # Save.
    png = output_dir / f"{cfg.figure_basename}.png"
    pdf = output_dir / f"{cfg.figure_basename}.pdf"

    fig.savefig(png, dpi=cfg.dpi, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    print(f"[INFO] Saved figure: {png}")
    print(f"[INFO] Saved figure: {pdf}")


# =============================================================================
# 5. Manuscript text output
# =============================================================================

def write_text_blocks(overall: pd.DataFrame, cfg: Config, output_dir: Path) -> None:
    """
    Write cautious Methods/Results/Discussion language.
    """
    txt = f"""
Recommended manuscript text for IMERG sensitivity
=================================================

Suggested placement
-------------------
Supplementary Methods / Supplementary Results, not main Results.

Methods text
------------
As a precipitation-product sensitivity test, we repeated the post-event rainfall-outcome calculation for 2000–2024 using IMERG precipitation while retaining the ERA5-Land-based heatwave event timing and footprint. This analysis was designed to test whether the rainfall-recovery outcome depended on ERA5 precipitation, rather than to provide a fully independent event reconstruction. Recovery was defined using the same baseline criterion as in the main analysis: rainfall ≥{cfg.rain_threshold_mm:g} mm d⁻¹ over at least {cfg.footprint_fraction_threshold * 100:.0f}% of the original heatwave footprint within post-event lag days {cfg.post_lag_min}–{cfg.post_lag_max}.

Results text
------------
The IMERG-based post-2000 sensitivity test produced a similar regional ordering of rainfall recovery, with lower recovery support in the Northwest than in the CONUS mean. Because event timing and footprints were retained from the ERA5-Land event catalogue, this analysis should be interpreted as precipitation-product robustness rather than an independent replication of the full event-detection framework.

Discussion text
---------------
The IMERG comparison reduces the likelihood that the post-2000 rainfall-recovery pattern is solely an artefact of ERA5 precipitation. However, it does not constitute a fully independent reconstruction of drought-amplified heatwave events, because event timing, footprints and environmental diagnostics remain based on the ERA5/ERA5-Land framework.

Suggested figure caption
------------------------
Supplementary Fig. X. IMERG-based precipitation-product sensitivity for post-2000 rainfall recovery. a, Mean recovery fraction during 2000–2024 calculated using ERA5 precipitation and IMERG precipitation while retaining the same ERA5-Land-based heatwave event timing and footprint. Bars show event-level means and error bars indicate bootstrap 95% confidence intervals. b, Five-year centred evolution of no-recovery fraction for CONUS and the Northwest under the two precipitation products. c, Mean rainy-footprint fraction within post-event lag days {cfg.post_lag_min}–{cfg.post_lag_max} under ERA5 and IMERG precipitation. This comparison tests whether the post-2000 rainfall-recovery outcome depends on the ERA5 precipitation product; it should not be interpreted as a fully independent reconstruction of drought-amplified heatwave events.

Important wording to avoid
--------------------------
Do not call this "independent validation", "independent replication", or "external confirmation".
Use "precipitation-product sensitivity", "independent precipitation-product check", or "post-2000 rainfall-outcome robustness test".
"""

    path = output_dir / "IMERG_methods_results_discussion_text.txt"
    path.write_text(txt, encoding="utf-8")
    print(f"[INFO] Saved manuscript text: {path}")


# =============================================================================
# 6. Main
# =============================================================================

def main(cfg: Config) -> None:
    output_dir = ensure_dir(cfg.output_dir)

    # -------------------------------------------------------------------------
    # Event-level summary
    # -------------------------------------------------------------------------
    ev_path = output_dir / "IMERG_event_level_summary_2000_2024.csv"

    if ev_path.exists():
        print(f"[INFO] Loading cached event-level summary: {ev_path}")
        ev = pd.read_csv(ev_path)
    else:
        ev = build_event_summary(cfg)
        ev.to_csv(ev_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved event-level summary: {ev_path}")

    # -------------------------------------------------------------------------
    # Long product frame and summaries
    # -------------------------------------------------------------------------
    long = long_product_frame(ev, cfg)

    if long.empty:
        raise RuntimeError("Long product table is empty. Check valid-event criteria and precipitation columns.")

    long_path = output_dir / "IMERG_product_sensitivity_long_event_table.csv"
    long.to_csv(long_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved long product table: {long_path}")

    ann = annual_summary(long, cfg)
    ann_path = output_dir / "IMERG_product_sensitivity_annual_summary.csv"
    ann.to_csv(ann_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved annual summary: {ann_path}")

    overall = overall_summary(long, cfg)
    overall = add_overall_trends(overall, ann)

    overall_path = output_dir / "IMERG_product_sensitivity_overall_summary.csv"
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved overall summary: {overall_path}")

    print("\n[INFO] Overall summary")
    show_cols = [
        "scope", "product", "n_events",
        "recovery_fraction_percent",
        "no_recovery_fraction_percent",
        "mean_rainy_footprint_fraction_percent",
        "recovery_fraction_trend_pp_decade",
        "no_recovery_fraction_trend_pp_decade",
        "mean_rainy_footprint_fraction_trend_pp_decade",
    ]
    print(overall[show_cols].to_string(index=False))

    # -------------------------------------------------------------------------
    # Figure and manuscript text
    # -------------------------------------------------------------------------
    plot_supp_figure(ann, overall, cfg, output_dir)
    write_text_blocks(overall, cfg, output_dir)

    print("\n[DONE] IMERG precipitation-product sensitivity analysis completed.")


if __name__ == "__main__":
    main(CFG)