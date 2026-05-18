# -*- coding: utf-8 -*-
"""
NCC-grade | CC3D heatwave event extraction + post-event 0–10 day rainfall windows
==================================================================================

核心思想（顶刊可辩护版本）
-------------------------
1) 热浪事件识别（CC3D）只基于热浪本体：
   - 仅使用 HEAT_COL == HEAT_VALUE 的行
   - 进行 26-connected spatiotemporal connected-components 标记
   - 这样保证“事件定义”不被 post-event 研究窗口污染

2) 事件识别完成后，再回到原始逐日 CSV 中抽取分析窗口：
   - 对每个事件成员格点，抽取：
       a) 事件期内所有记录（window 文件中保留）
       b) 事件结束后 0–10 天（event-level lag）
       c) 该格点局地热浪结束后 0–10 天（grid-level lag）
   - 同时标记是否在 post-window 内与“下一次局地热浪”重叠

3) 输出 4 类文件：
   OUT_ROOT/events_cc3d_postlag10_NCC/<year>/
       event_<year>_00001_core.csv
           -> 仅热浪核心事件日（CC3D定义本体）
       event_<year>_00001_window.csv
           -> 事件期 + nominal post-window（带 lag 和 overlap 标记）
       event_<year>_00001_post0_10_event.csv
           -> 主分析推荐：event-level 0–10天，且剔除下一次局地热浪重叠
       event_<year>_00001_post0_10_grid.csv
           -> 稳健性分析：grid-level 0–10天，且剔除下一次局地热浪重叠
       events_<year>_summary.csv
           -> 年度事件汇总

推荐使用方式
-----------
A. 主分析（最适合“热浪事件后 0–10 天降雨发生”）：
   直接使用 *_post0_10_event.csv

B. 稳健性分析（局地热浪停止后的雨落响应）：
   使用 *_post0_10_grid.csv

C. 如果你还要研究“事件期间 + 事件后”的全过程演变：
   使用 *_window.csv，并根据以下字段筛选：
   - is_heat_period_event == 1
   - is_post_event_0_10_censored == 1
   - is_post_grid_0_10_censored == 1

重要说明
-------
1) 这版代码不会把 post-lag 天数加入 CC3D 连通域识别，这是故意的，也是正确的。
2) “lag0” 默认包含事件结束当天；如果你想严格“结束后的第1~10天”，把 INCLUDE_LAG0 改成 False。
3) 如果某格点在 post-window 内又进入下一次热浪，则：
   - nominal 标记仍会保留
   - censored 文件会自动剔除这些重叠行
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cc3d  # pip install connected-components-3d

# =========================
# USER SETTINGS
# =========================
OUT_ROOT = Path(r"E:\temp_events_ERA5_S1S6_Nature所有数据版本")
BUCKET_DIR = OUT_ROOT / "buckets" / "selected_columns_csv"
EVENTS_ROOT = OUT_ROOT / "events_cc3d_postlag10_NCC"

START_YEAR = 1950
END_YEAR   = 2024

HEAT_COL = "heat3"
HEAT_VALUE = 1
MIN_DURATION_DAYS = 3

POST_LAG_DAYS = 10
INCLUDE_LAG0 = True  # True: 0–10天；False: 1–10天

ROUND_COORD = 4
CHUNK_SIZE = 2_000_000
VERBOSE = True


# =========================
# Utilities
# =========================
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def clean_year_dir(year_dir: Path):
    safe_mkdir(year_dir)
    for fp in year_dir.glob("event_*.csv"):
        try:
            fp.unlink()
        except Exception:
            pass
    for fp in year_dir.glob("events_*_summary.csv"):
        try:
            fp.unlink()
        except Exception:
            pass


def append_csv(fp: Path, df: pd.DataFrame):
    """
    首次写入带 BOM，后续 append 不再写 header/BOM
    """
    if df is None or df.empty:
        return
    if fp.exists():
        df.to_csv(fp, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(fp, mode="w", header=True, index=False, encoding="utf-8-sig")


def detect_lonlat_cols_from_file(fp: Path):
    """自动识别 lon/lat 或 longitude/latitude 列名"""
    cols = pd.read_csv(fp, nrows=0).columns.tolist()
    if "lon" in cols and "lat" in cols:
        return "lon", "lat"
    if "longitude" in cols and "latitude" in cols:
        return "longitude", "latitude"
    raise ValueError(f"{fp.name} 缺少经纬度列（需要 lon/lat 或 longitude/latitude）")


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一生成：
    - date_dt : datetime64[ns]
    - date    : YYYY-MM-DD
    - year    : int
    - doy     : int
    """
    df = df.copy()

    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        mask = dt.notna()
        df = df.loc[mask].copy()
        dt = dt.loc[mask]
        df["date_dt"] = dt.dt.normalize()
        df["date"] = df["date_dt"].dt.strftime("%Y-%m-%d")
        if "year" not in df.columns:
            df["year"] = df["date_dt"].dt.year.astype(int)
        else:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df.loc[df["year"].notna()].copy()
            df["year"] = df["year"].astype(int)

        if "doy" not in df.columns:
            df["doy"] = df["date_dt"].dt.dayofyear.astype(int)
        else:
            df["doy"] = pd.to_numeric(df["doy"], errors="coerce")
            df = df.loc[df["doy"].notna()].copy()
            df["doy"] = df["doy"].astype(int)

    else:
        if ("year" not in df.columns) or ("doy" not in df.columns):
            raise ValueError("文件中既没有 'date'，也没有完整的 ('year','doy') 列。")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["doy"] = pd.to_numeric(df["doy"], errors="coerce")
        df = df.loc[df["year"].notna() & df["doy"].notna()].copy()
        df["year"] = df["year"].astype(int)
        df["doy"] = df["doy"].astype(int)
        dt = pd.to_datetime(df["year"].astype(str) + "-01-01") + pd.to_timedelta(df["doy"] - 1, unit="D")
        df["date_dt"] = dt.dt.normalize()
        df["date"] = df["date_dt"].dt.strftime("%Y-%m-%d")

    return df


def add_coord_columns(df: pd.DataFrame, lon_col: str, lat_col: str) -> pd.DataFrame:
    """
    加入：
    - lon_round
    - lat_round
    - coord_key
    """
    df = df.copy()
    df = df.dropna(subset=[lon_col, lat_col]).copy()
    df["lon_round"] = pd.to_numeric(df[lon_col], errors="coerce").round(ROUND_COORD)
    df["lat_round"] = pd.to_numeric(df[lat_col], errors="coerce").round(ROUND_COORD)
    df = df.loc[df["lon_round"].notna() & df["lat_round"].notna()].copy()
    df["coord_key"] = df["lon_round"].astype(str) + "_" + df["lat_round"].astype(str)
    return df


def read_heat_rows_only(bucket_fp: Path) -> pd.DataFrame:
    """
    分块读取，仅保留 HEAT_COL == HEAT_VALUE 的行
    """
    if not bucket_fp.exists():
        return pd.DataFrame()

    kept = []
    for chunk in pd.read_csv(bucket_fp, chunksize=CHUNK_SIZE):
        if HEAT_COL not in chunk.columns:
            raise ValueError(f"{bucket_fp.name} 缺少列 {HEAT_COL}")

        heat_num = pd.to_numeric(chunk[HEAT_COL], errors="coerce").fillna(0).astype(int)
        mask = heat_num.eq(HEAT_VALUE)
        if mask.any():
            kept.append(chunk.loc[mask].copy())

    if not kept:
        return pd.DataFrame()

    return pd.concat(kept, ignore_index=True)


def build_grid_from_year(bucket_fp: Path):
    """
    每年独立扫描该年全部经纬度，建立当年网格。
    注意：必须扫描全年文件中的所有坐标，而不是只看热浪行。
    """
    lon_col, lat_col = detect_lonlat_cols_from_file(bucket_fp)

    all_coords = set()
    usecols = [lon_col, lat_col]

    for chunk in pd.read_csv(bucket_fp, usecols=usecols, chunksize=CHUNK_SIZE):
        chunk = chunk.dropna(subset=[lon_col, lat_col])
        if chunk.empty:
            continue
        lon_r = pd.to_numeric(chunk[lon_col], errors="coerce").round(ROUND_COORD)
        lat_r = pd.to_numeric(chunk[lat_col], errors="coerce").round(ROUND_COORD)
        ok = lon_r.notna() & lat_r.notna()
        if not ok.any():
            continue
        coords = list(zip(lon_r.loc[ok].to_numpy(), lat_r.loc[ok].to_numpy()))
        all_coords.update(coords)

    if not all_coords:
        raise RuntimeError(f"{bucket_fp.name} 中没有有效坐标。")

    all_lons = sorted({lon for lon, _ in all_coords})
    all_lats = sorted({lat for _, lat in all_coords})

    lon_to_ix = {lon: i for i, lon in enumerate(all_lons)}
    lat_to_iy = {lat: i for i, lat in enumerate(all_lats)}

    nx = len(all_lons)
    ny = len(all_lats)

    if VERBOSE:
        print(f"[{bucket_fp.name}] yearly grid built: nx={nx}, ny={ny}, unique_points={len(all_coords)}")

    return lon_col, lat_col, lon_to_ix, lat_to_iy, nx, ny


# =========================
# Stage 1: heatwave core extraction
# =========================
def build_heatwave_core_for_year(year: int):
    """
    只基于热浪日做 CC3D 事件识别。
    返回：
    - df_core : 热浪核心事件行
    - stat    : 事件汇总表（含 event_start/end）
    - lon_col, lat_col
    """
    bucket_fp = BUCKET_DIR / f"{year}.csv"
    if not bucket_fp.exists():
        if VERBOSE:
            print(f"[year={year}] 文件不存在，跳过：{bucket_fp}")
        return None, None, None, None

    if VERBOSE:
        print(f"\n========== year={year} ==========")
        print(f"[year={year}] building yearly grid from {bucket_fp.name}")

    lon_col, lat_col, lon_to_ix, lat_to_iy, nx, ny = build_grid_from_year(bucket_fp)

    if VERBOSE:
        print(f"[year={year}] reading heat rows only")

    df = read_heat_rows_only(bucket_fp)
    if df.empty:
        if VERBOSE:
            print(f"[year={year}] no heat rows -> skip")
        return None, None, lon_col, lat_col

    df = add_time_columns(df)
    df = df.loc[df["year"] == year].copy()
    if df.empty:
        if VERBOSE:
            print(f"[year={year}] no rows remain after year filter -> skip")
        return None, None, lon_col, lat_col

    df = add_coord_columns(df, lon_col, lat_col)
    if df.empty:
        if VERBOSE:
            print(f"[year={year}] no valid coords after coord parsing -> skip")
        return None, None, lon_col, lat_col

    # 坐标映射到 ix / iy
    ix = df["lon_round"].map(lon_to_ix).fillna(-1).astype(np.int32).to_numpy()
    iy = df["lat_round"].map(lat_to_iy).fillna(-1).astype(np.int32).to_numpy()

    ok = (ix >= 0) & (iy >= 0)
    if not ok.all():
        n_bad = int((~ok).sum())
        print(f"[year={year}] warning: {n_bad} heat rows not found in yearly grid, dropped.")
        df = df.loc[ok].copy()
        ix = ix[ok]
        iy = iy[ok]

    if df.empty:
        if VERBOSE:
            print(f"[year={year}] all heat rows dropped after coordinate check -> skip")
        return None, None, lon_col, lat_col

    # 时间索引：只覆盖当年实际热浪出现的 DOY 范围
    doy = df["doy"].astype(int).to_numpy()
    doy0 = int(doy.min())
    doy1 = int(doy.max())
    nt = doy1 - doy0 + 1
    t = doy - doy0

    if nt <= 0:
        if VERBOSE:
            print(f"[year={year}] invalid time range -> skip")
        return None, None, lon_col, lat_col

    if VERBOSE:
        print(f"[year={year}] active DOY range: {doy0} - {doy1}, nt={nt}")

    # 3D mask
    A = np.zeros((nt, ny, nx), dtype=np.uint8)
    A[t, iy, ix] = 1

    if VERBOSE:
        est_mb = A.nbytes / 1024 / 1024
        print(f"[year={year}] voxel grid shape = (t={nt}, y={ny}, x={nx}), ones={int(A.sum()):,}, A≈{est_mb:.1f} MB")

    labels = cc3d.connected_components(A, connectivity=26)
    n_labels = int(labels.max())

    if VERBOSE:
        label_mb = labels.nbytes / 1024 / 1024
        print(f"[year={year}] cc3d components = {n_labels:,}, labels≈{label_mb:.1f} MB")

    if n_labels == 0:
        return None, None, lon_col, lat_col

    lab = labels[t, iy, ix].astype(np.int32)
    df["cc_label"] = lab
    df = df.loc[df["cc_label"] > 0].copy()
    if df.empty:
        return None, None, lon_col, lat_col

    # 最小持续天数过滤
    dur = df.groupby("cc_label")["date_dt"].nunique()
    keep_labels = dur.index[dur.values >= MIN_DURATION_DAYS].to_numpy()

    if len(keep_labels) == 0:
        if VERBOSE:
            print(f"[year={year}] no events with duration >= {MIN_DURATION_DAYS}")
        return None, None, lon_col, lat_col

    df = df.loc[df["cc_label"].isin(keep_labels)].copy()

    # 事件统计（基于 cc_label）
    stat = (
        df.groupby("cc_label")
          .agg(
              event_start_dt=("date_dt", "min"),
              event_end_dt=("date_dt", "max"),
              start_doy=("doy", "min"),
              end_doy=("doy", "max"),
              duration_days=("date_dt", "nunique"),
              n_records=("date_dt", "size"),
              n_grids=("coord_key", "nunique"),
              lon_min=(lon_col, "min"),
              lon_max=(lon_col, "max"),
              lat_min=(lat_col, "min"),
              lat_max=(lat_col, "max"),
          )
          .reset_index()
    )

    stat = stat.sort_values(
        ["event_start_dt", "event_end_dt", "n_records"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    stat["event_id"] = np.arange(1, len(stat) + 1, dtype=int)
    label_to_eid = dict(zip(stat["cc_label"].astype(int), stat["event_id"].astype(int)))
    df["event_id"] = df["cc_label"].map(label_to_eid).astype(int)

    stat["year"] = year
    stat["event_start"] = stat["event_start_dt"].dt.strftime("%Y-%m-%d")
    stat["event_end"] = stat["event_end_dt"].dt.strftime("%Y-%m-%d")
    stat["nominal_post_end_dt"] = stat["event_end_dt"] + pd.to_timedelta(POST_LAG_DAYS, unit="D")
    stat["nominal_post_end"] = stat["nominal_post_end_dt"].dt.strftime("%Y-%m-%d")

    # 把全局 event_start / event_end 回填到 core df
    event_dates = stat.loc[:, ["event_id", "event_start_dt", "event_end_dt", "event_start", "event_end"]]
    df = df.merge(event_dates, on="event_id", how="left")

    return df, stat, lon_col, lat_col


def export_core_files(year: int, df_core: pd.DataFrame, out_dir: Path, lon_col: str, lat_col: str):
    """
    输出热浪核心事件文件：
    event_<year>_00001_core.csv
    """
    if df_core is None or df_core.empty:
        return

    for eid, g in df_core.groupby("event_id", sort=True):
        eid = int(eid)
        g = g.sort_values(["date_dt", lat_col, lon_col], kind="mergesort").copy()

        # 输出时去掉纯内部标签 cc_label，但保留 event_id / coord_key / event_start / event_end
        out = g.drop(columns=["cc_label"], errors="ignore").copy()
        out = out.drop(columns=["date_dt"], errors="ignore")

        out_fp = out_dir / f"event_{year}_{eid:05d}_core.csv"
        out.to_csv(out_fp, index=False, encoding="utf-8-sig")


# =========================
# Stage 2: build post-event sampling windows
# =========================
def build_member_meta(df_core: pd.DataFrame, stat: pd.DataFrame) -> pd.DataFrame:
    """
    为每个 (event_id, coord_key) 生成成员格点元信息：
    - global event start/end
    - local grid start/end
    - next local heat start on same grid (用于重叠标记)
    """
    if df_core is None or df_core.empty:
        return pd.DataFrame()

    grid_meta = (
        df_core.groupby(["event_id", "coord_key"])
               .agg(
                   grid_start_dt=("date_dt", "min"),
                   grid_end_dt=("date_dt", "max"),
                   grid_n_heat_days=("date_dt", "nunique"),
                   lon_round=("lon_round", "first"),
                   lat_round=("lat_round", "first"),
               )
               .reset_index()
    )

    grid_meta = grid_meta.merge(
        stat.loc[:, ["event_id", "event_start_dt", "event_end_dt", "event_start", "event_end"]],
        on="event_id",
        how="left"
    )

    # 同一格点在不同事件中的下一个局地热浪开始时间
    grid_meta = grid_meta.sort_values(["coord_key", "grid_start_dt", "event_id"]).reset_index(drop=True)
    grid_meta["next_grid_start_dt"] = grid_meta.groupby("coord_key")["grid_start_dt"].shift(-1)

    grid_meta["grid_start"] = grid_meta["grid_start_dt"].dt.strftime("%Y-%m-%d")
    grid_meta["grid_end"] = grid_meta["grid_end_dt"].dt.strftime("%Y-%m-%d")
    grid_meta["next_grid_start"] = grid_meta["next_grid_start_dt"].dt.strftime("%Y-%m-%d")

    return grid_meta


def build_member_dict(member_meta: pd.DataFrame):
    """
    转为：
    { coord_key: [meta1, meta2, ...] }
    """
    if member_meta is None or member_meta.empty:
        return {}, set()

    cols = [
        "event_id",
        "event_start_dt", "event_end_dt",
        "event_start", "event_end",
        "grid_start_dt", "grid_end_dt",
        "grid_start", "grid_end",
        "next_grid_start_dt", "next_grid_start",
    ]

    member_dict = {}
    for coord_key, sub in member_meta.groupby("coord_key", sort=False):
        member_dict[coord_key] = list(sub.loc[:, cols].itertuples(index=False, name="MemberMeta"))

    return member_dict, set(member_dict.keys())


def export_window_and_post_files(
    year: int,
    bucket_fp: Path,
    out_dir: Path,
    lon_col: str,
    lat_col: str,
    member_dict: dict,
    member_keys: set,
):
    """
    再扫描全年原始逐日文件，导出：
    1) window.csv
       - 事件期 + nominal post-window（event_end + 10）
       - 含 lag_day_event / lag_day_grid
       - 含 overlap 标记

    2) post0_10_event.csv
       - 主分析推荐：event-level lag 0–10，且剔除 overlaps_next_local_heat

    3) post0_10_grid.csv
       - 稳健性分析：grid-level lag 0–10，且剔除 overlaps_next_local_heat
    """
    if not member_dict:
        return

    lag_min = 0 if INCLUDE_LAG0 else 1

    if VERBOSE:
        print(f"[year={year}] second pass: building event windows from full yearly CSV")

    for chunk in pd.read_csv(bucket_fp, chunksize=CHUNK_SIZE):
        chunk = add_time_columns(chunk)
        if chunk.empty:
            continue

        chunk = chunk.loc[chunk["year"] == year].copy()
        if chunk.empty:
            continue

        chunk = add_coord_columns(chunk, lon_col, lat_col)
        if chunk.empty:
            continue

        # 只保留属于任一事件成员格点的行
        chunk = chunk.loc[chunk["coord_key"].isin(member_keys)].copy()
        if chunk.empty:
            continue

        pieces_window = {}
        pieces_post_event = {}
        pieces_post_grid = {}

        for coord_key, cg in chunk.groupby("coord_key", sort=False):
            metas = member_dict.get(coord_key)
            if not metas:
                continue

            # 保证时间排序，lag 判断更稳
            cg = cg.sort_values("date_dt", kind="mergesort").copy()

            for meta in metas:
                nominal_end_dt = meta.event_end_dt + pd.Timedelta(days=POST_LAG_DAYS)

                # window 文件：保留整个事件期 + nominal post-window
                m = (cg["date_dt"] >= meta.event_start_dt) & (cg["date_dt"] <= nominal_end_dt)
                if not m.any():
                    continue

                tmp = cg.loc[m].copy()
                if tmp.empty:
                    continue

                tmp["event_id"] = int(meta.event_id)
                tmp["event_start"] = meta.event_start
                tmp["event_end"] = meta.event_end
                tmp["grid_start"] = meta.grid_start
                tmp["grid_end"] = meta.grid_end
                tmp["next_grid_start"] = meta.next_grid_start

                # lag 定义
                tmp["lag_day_event"] = (tmp["date_dt"] - meta.event_end_dt).dt.days
                tmp["lag_day_grid"] = (tmp["date_dt"] - meta.grid_end_dt).dt.days

                # 事件期 / 局地热浪期
                tmp["is_heat_period_event"] = (
                    (tmp["date_dt"] >= meta.event_start_dt) &
                    (tmp["date_dt"] <= meta.event_end_dt)
                ).astype(int)

                tmp["is_heat_period_grid"] = (
                    (tmp["date_dt"] >= meta.grid_start_dt) &
                    (tmp["date_dt"] <= meta.grid_end_dt)
                ).astype(int)

                # nominal post-window
                tmp["is_post_event_0_10_nominal"] = tmp["lag_day_event"].between(lag_min, POST_LAG_DAYS).astype(int)
                tmp["is_post_grid_0_10_nominal"] = tmp["lag_day_grid"].between(lag_min, POST_LAG_DAYS).astype(int)

                # 是否与“同一格点下一次热浪”重叠
                if pd.isna(meta.next_grid_start_dt):
                    overlap = pd.Series(False, index=tmp.index)
                else:
                    overlap = tmp["date_dt"] >= meta.next_grid_start_dt

                tmp["overlaps_next_local_heat"] = overlap.astype(int)

                # censored post-window（推荐用于正式分析）
                tmp["is_post_event_0_10_censored"] = (
                    (tmp["is_post_event_0_10_nominal"] == 1) &
                    (tmp["overlaps_next_local_heat"] == 0)
                ).astype(int)

                tmp["is_post_grid_0_10_censored"] = (
                    (tmp["is_post_grid_0_10_nominal"] == 1) &
                    (tmp["overlaps_next_local_heat"] == 0)
                ).astype(int)

                out_full = tmp.drop(columns=["date_dt"], errors="ignore").copy()

                eid = int(meta.event_id)
                pieces_window.setdefault(eid, []).append(out_full)

                post_event = out_full.loc[out_full["is_post_event_0_10_censored"] == 1].copy()
                if not post_event.empty:
                    pieces_post_event.setdefault(eid, []).append(post_event)

                post_grid = out_full.loc[out_full["is_post_grid_0_10_censored"] == 1].copy()
                if not post_grid.empty:
                    pieces_post_grid.setdefault(eid, []).append(post_grid)

        # 逐事件 append 写出
        for eid, dfs in pieces_window.items():
            out_fp = out_dir / f"event_{year}_{eid:05d}_window.csv"
            out_df = pd.concat(dfs, ignore_index=True)
            append_csv(out_fp, out_df)

        for eid, dfs in pieces_post_event.items():
            out_fp = out_dir / f"event_{year}_{eid:05d}_post0_10_event.csv"
            out_df = pd.concat(dfs, ignore_index=True)
            append_csv(out_fp, out_df)

        for eid, dfs in pieces_post_grid.items():
            out_fp = out_dir / f"event_{year}_{eid:05d}_post0_10_grid.csv"
            out_df = pd.concat(dfs, ignore_index=True)
            append_csv(out_fp, out_df)


def finalize_summary(year: int, stat: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    汇总 window / post 文件的记录数
    """
    if stat is None or stat.empty:
        return pd.DataFrame()

    stat = stat.copy()
    stat["n_window_records"] = 0
    stat["n_post_event_0_10_records"] = 0
    stat["n_post_grid_0_10_records"] = 0

    for i, row in stat.iterrows():
        eid = int(row["event_id"])

        fp_window = out_dir / f"event_{year}_{eid:05d}_window.csv"
        fp_post_event = out_dir / f"event_{year}_{eid:05d}_post0_10_event.csv"
        fp_post_grid = out_dir / f"event_{year}_{eid:05d}_post0_10_grid.csv"

        if fp_window.exists():
            try:
                n_window = sum(1 for _ in open(fp_window, "r", encoding="utf-8-sig")) - 1
                stat.at[i, "n_window_records"] = max(n_window, 0)
            except Exception:
                pass

        if fp_post_event.exists():
            try:
                n_post_event = sum(1 for _ in open(fp_post_event, "r", encoding="utf-8-sig")) - 1
                stat.at[i, "n_post_event_0_10_records"] = max(n_post_event, 0)
            except Exception:
                pass

        if fp_post_grid.exists():
            try:
                n_post_grid = sum(1 for _ in open(fp_post_grid, "r", encoding="utf-8-sig")) - 1
                stat.at[i, "n_post_grid_0_10_records"] = max(n_post_grid, 0)
            except Exception:
                pass

    # 输出时去掉 datetime 内部列，只保留字符串日期列
    stat_out = stat.drop(columns=["event_start_dt", "event_end_dt", "nominal_post_end_dt"], errors="ignore").copy()
    return stat_out


# =========================
# Main yearly pipeline
# =========================
def extract_year_cc3d_with_postlag(year: int):
    bucket_fp = BUCKET_DIR / f"{year}.csv"
    if not bucket_fp.exists():
        if VERBOSE:
            print(f"[year={year}] 文件不存在，跳过：{bucket_fp}")
        return

    out_dir = EVENTS_ROOT / f"{year}"
    clean_year_dir(out_dir)

    # Stage 1: 仅热浪核心事件识别
    df_core, stat, lon_col, lat_col = build_heatwave_core_for_year(year)
    if df_core is None or stat is None or df_core.empty or stat.empty:
        if VERBOSE:
            print(f"[year={year}] no valid events after core extraction")
        return

    # 导出核心事件文件
    export_core_files(year, df_core, out_dir, lon_col, lat_col)

    # Stage 2: 构造成员格点元信息，并回扫全年原始数据抽取 post-window
    member_meta = build_member_meta(df_core, stat)
    member_dict, member_keys = build_member_dict(member_meta)

    export_window_and_post_files(
        year=year,
        bucket_fp=bucket_fp,
        out_dir=out_dir,
        lon_col=lon_col,
        lat_col=lat_col,
        member_dict=member_dict,
        member_keys=member_keys,
    )

    # summary
    summary = finalize_summary(year, stat, out_dir)
    summary_fp = out_dir / f"events_{year}_summary.csv"
    summary.to_csv(summary_fp, index=False, encoding="utf-8-sig")

    if VERBOSE:
        print(f"[year={year}] kept events = {len(summary):,}")
        print(f"[year={year}] out dir      = {out_dir}")
        print(f"[year={year}] summary file = {summary_fp.name}")


def main():
    if not BUCKET_DIR.exists():
        raise FileNotFoundError(f"BUCKET_DIR 不存在: {BUCKET_DIR}")

    safe_mkdir(EVENTS_ROOT)

    for year in range(START_YEAR, END_YEAR + 1):
        try:
            extract_year_cc3d_with_postlag(year)
        except Exception as e:
            print(f"[ERROR year={year}] {repr(e)}")

    print("\nALL DONE.")
    print("events root:", EVENTS_ROOT)


if __name__ == "__main__":
    main()