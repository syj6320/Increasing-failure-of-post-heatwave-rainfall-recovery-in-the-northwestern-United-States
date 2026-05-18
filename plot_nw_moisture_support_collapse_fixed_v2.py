# -*- coding: utf-8 -*-
"""
Northwest all-events moisture-support collapse plotting (fixed v2)
=================================================================

Robust plotting script for:
    nw_all_events_moisture_composite_grid_long.csv
    nw_all_events_moisture_tau_event_box_means.csv

Fixes relative to previous versions
-----------------------------------
1) Uses the exact Windows path provided by the user.
2) Uses a much shorter output folder name to avoid WinError 206.
3) Accepts grid-table canonical names and box-table *_mean names.
4) Keeps plotting logic unchanged: main diff maps, process-sequence, tau=0 triptych.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


@dataclass
class Config:
    root_dir: Path = Path(r"E:\temp_events_ERA5_S1S6_Nature所有数据版本\events_cc3d_postlag10_NCC_with_pr_ws_rh第三篇的数据_added_CAPE_IVT_T850_added_Z500_W500_added_Bowen_Rn_added_WIND250_WIND850\_nw_moisture_mechanism_tables")
    fp_grid: str = "nw_all_events_moisture_composite_grid_long.csv"
    fp_box: str = "nw_all_events_moisture_tau_event_box_means.csv"
    out_dir_name: str = "fig_msc"
    tau_main: Tuple[int, ...] = (-5, -2, 0)
    tau_all: Tuple[int, ...] = (-5, -4, -3, -2, -1, 0)
    dpi: int = 320


CFG = Config()

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
})


VAR_META: Dict[str, Dict[str, str]] = {
    "relative_humidity": {
        "label": "Relative humidity (%)",
        "short": "RH",
        "cmap": "RdBu_r",
    },
    "moisture_convergence": {
        "label": "Moisture convergence",
        "short": "Moisture convergence",
        "cmap": "RdBu_r",
    },
    "convective_available_potential_energy_mean": {
        "label": "CAPE (J kg$^{-1}$)",
        "short": "CAPE",
        "cmap": "RdBu_r",
    },
    "precip_mm": {
        "label": "Precipitation (mm day$^{-1}$)",
        "short": "Precipitation",
        "cmap": "RdBu_r",
    },
}

GRID_ALIASES: Dict[str, Tuple[str, ...]] = {
    "relative_humidity": ("relative_humidity", "rh"),
    "moisture_convergence": ("moisture_convergence", "moisture_convergence_mean"),
    "convective_available_potential_energy_mean": (
        "convective_available_potential_energy_mean",
        "cape",
        "cape_mean",
    ),
    "precip_mm": ("precip_mm", "precip_mm_mean", "precipitation_mm"),
}

BOX_ALIASES: Dict[str, Tuple[str, ...]] = {
    "relative_humidity": ("relative_humidity", "rh_mean", "relative_humidity_mean"),
    "moisture_convergence": ("moisture_convergence", "moisture_convergence_mean"),
    "convective_available_potential_energy_mean": (
        "convective_available_potential_energy_mean",
        "cape_mean",
        "cape",
    ),
    "precip_mm": ("precip_mm", "precip_mm_mean", "precipitation_mm"),
}

GROUP_ORDER = ["Recovered", "Unrecovered"]
GROUP_COLORS = {"Recovered": "#2b8cbe", "Unrecovered": "#cb181d"}


def add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=19, fontweight="bold", clip_on=False)


def tidy_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color="0.88", lw=0.8)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color="0.88", lw=0.8)


def ensure_out_dir(cfg: Config) -> Path:
    out = cfg.root_dir / cfg.out_dir_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _find_existing_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_table_columns(grid: pd.DataFrame, box: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grid = grid.copy()
    box = box.copy()

    rename_grid = {}
    rename_box = {}
    for canon, aliases in GRID_ALIASES.items():
        gcol = _find_existing_col(grid, aliases)
        if gcol is not None and gcol != canon:
            rename_grid[gcol] = canon
    for canon, aliases in BOX_ALIASES.items():
        bcol = _find_existing_col(box, aliases)
        if bcol is not None and bcol != canon:
            rename_box[bcol] = canon

    if rename_grid:
        grid = grid.rename(columns=rename_grid)
    if rename_box:
        box = box.rename(columns=rename_box)

    for df in (grid, box):
        for c in ["tau", "latitude", "longitude", *VAR_META.keys()]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return grid, box


def load_tables(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fp_grid = cfg.root_dir / cfg.fp_grid
    fp_box = cfg.root_dir / cfg.fp_box
    if not fp_grid.exists():
        raise FileNotFoundError(fp_grid)
    if not fp_box.exists():
        raise FileNotFoundError(fp_box)
    grid = pd.read_csv(fp_grid)
    box = pd.read_csv(fp_box)
    grid, box = normalize_table_columns(grid, box)
    return grid, box


def pivot_field(df: pd.DataFrame, var: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    piv = df.pivot(index="latitude", columns="longitude", values=var).sort_index().sort_index(axis=1)
    lat = piv.index.to_numpy(dtype=float)
    lon = piv.columns.to_numpy(dtype=float)
    arr = piv.to_numpy(dtype=float)
    return lon, lat, arr


def compute_diff_field(grid: pd.DataFrame, tau: int, var: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rec = grid.loc[(grid["group"] == "Recovered") & (grid["tau"] == tau)].copy()
    unr = grid.loc[(grid["group"] == "Unrecovered") & (grid["tau"] == tau)].copy()
    lon_r, lat_r, arr_r = pivot_field(rec, var)
    lon_u, lat_u, arr_u = pivot_field(unr, var)
    if not (np.allclose(lon_r, lon_u) and np.allclose(lat_r, lat_u)):
        raise ValueError(f"Grid mismatch for tau={tau}, var={var}")
    arr_d = arr_u - arr_r
    return lon_r, lat_r, arr_r, arr_u, arr_d


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.nanpercentile(vals, 100 * alpha / 2.0)), float(np.nanpercentile(vals, 100 * (1.0 - alpha / 2.0)))


def summarize_event_box(box: pd.DataFrame, var: str, tau_all: Tuple[int, ...]) -> pd.DataFrame:
    rows: List[Dict] = []
    for grp in GROUP_ORDER:
        for tau in tau_all:
            sub = box.loc[(box["group"] == grp) & (box["tau"] == tau)].copy()
            if sub.empty or var not in sub.columns:
                continue
            vals = pd.to_numeric(sub[var], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            lcl, ucl = percentile_ci(vals)
            rows.append({
                "group": grp,
                "tau": tau,
                "mean": float(np.nanmean(vals)),
                "lcl": lcl,
                "ucl": ucl,
                "n": int(vals.size),
            })
    return pd.DataFrame(rows)


def plot_field(ax, lon: np.ndarray, lat: np.ndarray, arr: np.ndarray, title: str,
               cmap: str = "RdBu_r", norm=None, use_cartopy: bool = False):
    if use_cartopy and HAS_CARTOPY:
        ax.set_extent([float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())], ccrs.PlateCarree())
        mesh = ax.pcolormesh(lon, lat, arr, cmap=cmap, norm=norm, shading="auto", transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3)
        try:
            states = cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces_lines",
                scale="50m",
                facecolor="none",
            )
            ax.add_feature(states, edgecolor="0.65", linewidth=0.3)
        except Exception:
            pass
    else:
        mesh = ax.pcolormesh(lon, lat, arr, cmap=cmap, norm=norm, shading="auto")
        ax.set_xlim(float(lon.min()), float(lon.max()))
        ax.set_ylim(float(lat.min()), float(lat.max()))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    ax.set_title(title)
    return mesh


def plot_main_diff_maps(grid: pd.DataFrame, cfg: Config, out_dir: Path) -> None:
    tau_list = list(cfg.tau_main)
    var_list = list(VAR_META.keys())
    use_cartopy = HAS_CARTOPY
    subplot_kw = {"projection": ccrs.PlateCarree()} if use_cartopy else {}
    fig, axes = plt.subplots(len(var_list), len(tau_list), figsize=(15.5, 16.0), subplot_kw=subplot_kw)
    if len(var_list) == 1:
        axes = np.array([axes])
    letters = list("abcdefghijklmnop")
    li = 0
    for r, var in enumerate(var_list):
        diff_arrays = []
        fields = {}
        for tau in tau_list:
            lon, lat, arr_r, arr_u, arr_d = compute_diff_field(grid, tau, var)
            fields[tau] = (lon, lat, arr_d)
            diff_arrays.append(arr_d)
        vmax = np.nanmax(np.abs(np.stack(diff_arrays)))
        if (not np.isfinite(vmax)) or vmax == 0:
            vmax = 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        for c, tau in enumerate(tau_list):
            ax = axes[r, c]
            lon, lat, arr_d = fields[tau]
            mesh = plot_field(ax, lon, lat, arr_d, title=rf"$\tau$ = {tau}", cmap=VAR_META[var]["cmap"], norm=norm, use_cartopy=use_cartopy)
            if c == 0:
                ax.text(-0.16, 0.5, VAR_META[var]["label"], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=14)
            add_panel_label(ax, letters[li], x=-0.10, y=1.03)
            li += 1
        cbar = fig.colorbar(mesh, ax=axes[r, :], fraction=0.020, pad=0.02)
        cbar.set_label(f"Unrecovered − Recovered: {VAR_META[var]['short']}")
    fig.savefig(out_dir / 'Figure_M1_moisture_support_diffmaps_main.png', dpi=cfg.dpi)
    plt.close(fig)


def plot_process_sequence(box: pd.DataFrame, cfg: Config, out_dir: Path) -> None:
    var_list = list(VAR_META.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), gridspec_kw={"wspace": 0.28, "hspace": 0.34})
    axes = axes.ravel()
    for i, var in enumerate(var_list):
        ax = axes[i]
        sm = summarize_event_box(box, var, cfg.tau_all)
        for grp in GROUP_ORDER:
            g = sm.loc[sm["group"] == grp].sort_values("tau")
            if g.empty:
                continue
            ax.fill_between(g["tau"], g["lcl"], g["ucl"], color=GROUP_COLORS[grp], alpha=0.14)
            ax.plot(g["tau"], g["mean"], color=GROUP_COLORS[grp], lw=2.3, marker="o", label=grp)
        ax.axvline(0, color="0.55", lw=1.0, ls="--")
        ax.set_xlabel(r"Days relative to event end ($\tau$)")
        ax.set_ylabel(VAR_META[var]["label"])
        ax.set_title(VAR_META[var]["short"])
        tidy_axis(ax, "y")
        add_panel_label(ax, list("abcd")[i], x=-0.12, y=1.04)
        if i == 0:
            ax.legend(frameon=False, loc="best")
    fig.savefig(out_dir / 'Figure_M2_moisture_support_sequence_main.png', dpi=cfg.dpi)
    plt.close(fig)


def plot_tau0_triptych(grid: pd.DataFrame, cfg: Config, out_dir: Path) -> None:
    tau = 0
    var_list = list(VAR_META.keys())
    use_cartopy = HAS_CARTOPY
    subplot_kw = {"projection": ccrs.PlateCarree()} if use_cartopy else {}
    fig, axes = plt.subplots(len(var_list), 3, figsize=(16.5, 16.5), subplot_kw=subplot_kw)
    letters = list("abcdefghijkl")
    li = 0
    for r, var in enumerate(var_list):
        lon, lat, arr_r, arr_u, arr_d = compute_diff_field(grid, tau, var)
        max_abs_diff = np.nanmax(np.abs(arr_d))
        if (not np.isfinite(max_abs_diff)) or max_abs_diff == 0:
            max_abs_diff = 1.0
        diff_norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs_diff, vmax=max_abs_diff)
        titles = ["Recovered", "Unrecovered", "Unrecovered − Recovered"]
        arrays = [arr_r, arr_u, arr_d]
        norms = [None, None, diff_norm]
        for c in range(3):
            ax = axes[r, c]
            mesh = plot_field(ax, lon, lat, arrays[c], title=titles[c], cmap=VAR_META[var]["cmap"], norm=norms[c], use_cartopy=use_cartopy)
            if c == 0:
                ax.text(-0.16, 0.5, VAR_META[var]["label"], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=14)
            add_panel_label(ax, letters[li], x=-0.10, y=1.03)
            li += 1
        cbar = fig.colorbar(mesh, ax=axes[r, :], fraction=0.020, pad=0.02)
        cbar.set_label(VAR_META[var]["short"])
    fig.savefig(out_dir / 'Supp_Fig_M1_moisture_support_triptych_tau0.png', dpi=cfg.dpi)
    plt.close(fig)


def main(cfg: Config = CFG) -> None:
    out_dir = ensure_out_dir(cfg)
    grid, box = load_tables(cfg)

    needed_grid = {"group", "tau", "latitude", "longitude", *VAR_META.keys()}
    missing_grid = needed_grid.difference(grid.columns)
    if missing_grid:
        raise KeyError(f"Missing required columns in grid table after alias normalization: {sorted(missing_grid)}")

    needed_box = {"group", "tau", *VAR_META.keys()}
    missing_box = needed_box.difference(box.columns)
    if missing_box:
        raise KeyError(f"Missing required columns in box-mean table after alias normalization: {sorted(missing_box)}")

    plot_main_diff_maps(grid, cfg, out_dir)
    plot_process_sequence(box, cfg, out_dir)
    plot_tau0_triptych(grid, cfg, out_dir)
    print(f"[DONE] Moisture-support collapse figures written to: {out_dir}")


if __name__ == "__main__":
    main(CFG)
