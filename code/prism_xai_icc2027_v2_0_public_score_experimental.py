#!/usr/bin/env python3
"""PRISM-XAI reproducible experiment pipeline for IEEE ICC 2027.

The quick preset runs end-to-end using only common scientific Python
dependencies. The full preset uses LightGBM/TreeSHAP and a PyTorch neural
model with Integrated Gradients when those optional packages are installed.

No numerical results are hard-coded. Every table and plot is generated from
the executed runs and accompanied by the exact configuration and file hashes.

Google Colab, two real datasets, engineering verification::

    !python prism_xai_icc2027.py --real-suite verification \\
        --output-dir /content/prism_real_verification

Full ICC experiment matrix::

    !python prism_xai_icc2027.py --real-suite full \\
        --output-dir /content/prism_icc2027_results
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import urllib.request
import warnings
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/prism_xai_mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, t as student_t
from sklearn.base import RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import seaborn as sns
except ImportError:
    sns = None

SCRIPT_VERSION = "1.9.0-selective-private-release"

SCHEDULERS = [
    "periodic_fixed_dp",
    "channel_fixed_dp",
    "aox_fixed_dp",
    "repaired_mapxai",
    "joint_no_compression",
    "prism_xai",
]

SCHEDULER_LABELS = {
    "periodic_fixed_dp": "Periodic fixed-DP",
    "channel_fixed_dp": "Channel-aware fixed-DP",
    "aox_fixed_dp": "AoX-aware fixed-DP",
    "repaired_mapxai": "MAP-XAI fallback",
    "joint_no_compression": "Joint, no compression",
    "prism_xai": "PRISM-XAI",
    "prism_no_aox": "PRISM w/o AoX",
    "prism_no_privacy_price": "PRISM w/o privacy price",
    "prism_fixed_resolution": "PRISM fixed resolution",
    "prism_no_hierarchy": "PRISM w/o hierarchy calibration",
    "prism_no_fairness": "PRISM w/o fairness bonus",
    "prism_no_residual_accumulation": "PRISM w/o residual accumulation",
}


@dataclass
class ExperimentConfig:
    preset: str = "quick"
    dataset: str = "opsd"
    london_dir: Optional[str] = None
    cache_dir: str = "data_cache"
    output_dir: str = "prism_results"
    models: List[str] = field(default_factory=lambda: ["tree"])
    xai_methods: List[str] = field(default_factory=lambda: ["default"])
    seeds: List[int] = field(default_factory=lambda: [11])
    epsilon_budgets: List[float] = field(default_factory=lambda: [4.0])
    delta: float = 1e-5
    n_endpoints: int = 12
    n_zones: int = 3
    n_steps: int = 48
    calibration_steps: int = 24
    history: int = 24
    train_max_rows: int = 30000
    quick_tree_estimators: int = 100
    tree_estimators: int = 350
    neural_epochs: int = 60
    ig_steps: int = 20
    slot_duration_ms: float = 1.0
    deadline_ms: float = 5.0
    edge_delay_ms: float = 1.0
    rb_bandwidth_hz: float = 180000.0
    n_resource_blocks: int = 8
    carrier_ghz: float = 3.5
    tx_power_dbm: float = 23.0
    noise_figure_db: float = 7.0
    base_packet_bits: int = 1024
    clip_quantile: float = 0.95
    topk_eval: int = 3
    reliability_aox_threshold: float = 8.0
    hierarchy_threshold: float = 3.0
    min_critical_coverage: float = 0.05
    min_jain_fairness: float = 0.88
    residual_noise_margin: float = 1.85
    residual_upgrade_margin: float = 4.00
    residual_decay: float = 0.65
    override_gain_margin: float = 0.35
    prism_traffic_guardrail_ratio: float = 1.02
    prism_upgrade_budget_ratio: float = 1.02
    allow_prism_over_budget: bool = False
    gain_per_bit_weight: float = 2600.0
    aox_weight: float = 80.0
    energy_weight: float = 0.001
    privacy_price_weight: float = 0.18
    fairness_weight: float = 120.0
    critical_recall_weight: float = 160.0
    selective_private_release: bool = True
    max_private_releases_per_endpoint: int = 8
    public_release_score_threshold: float = 4.0
    release_cap_sigma_safety: float = 1.02
    min_budget_calibrated_sigma: float = 0.35
    adjacency: str = "add_remove"
    run_ablations: bool = False
    save_all_traces: bool = False

    @classmethod
    def full_defaults(cls) -> "ExperimentConfig":
        return cls(
            preset="full",
            dataset="uci",
            models=["tree", "neural"],
            xai_methods=["default"],
            seeds=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111],
            epsilon_budgets=[2.0, 4.0, 8.0],
            n_endpoints=32,
            n_zones=4,
            n_steps=512,
            calibration_steps=96,
            train_max_rows=80000,
            tree_estimators=500,
            neural_epochs=80,
            ig_steps=24,
            run_ablations=True,
        )


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    cal_load: np.ndarray
    eval_load: np.ndarray
    feature_names: List[str]
    feature_baseline: np.ndarray
    zones: np.ndarray
    private_attribute: np.ndarray
    eval_timestamps: pd.DatetimeIndex
    endpoint_ids: List[str]
    source_window_start: str
    source_window_end: str
    source_name: str
    source_url: str
    source_sha256: str

    @property
    def n_steps(self) -> int:
        return int(self.X_eval.shape[0])

    @property
    def n_endpoints(self) -> int:
        return int(self.X_eval.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.X_eval.shape[2])


@dataclass(frozen=True)
class PacketMode:
    name: str
    k_fraction: float
    quant_bits: int
    sigma: float


PACKET_MODES = (
    PacketMode("coarse", 0.25, 6, 12.0),
    PacketMode("balanced", 0.50, 8, 8.0),
    # Detailed packets deliberately spend much more privacy and are feasible
    # only a few times per endpoint. PRISM should reserve them for critical
    # events; fixed-DP baselines remain on the balanced mode.
    PacketMode("detailed", 1.00, 10, 2.0),
)

REPAIRED_MAP_PACKET = PacketMode("repaired_mapxai_packet", 0.25, 6, 12.0)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def safe_divide(a: np.ndarray, b: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), default, dtype=float)
    return np.divide(a, b, out=out, where=np.abs(b) > 1e-12)


def jain_index(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    denom = len(x) * np.sum(x * x)
    return float(np.sum(x) ** 2 / denom) if denom > 0 else 0.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def spearman_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    ra, rb = rankdata(a), rankdata(b)
    value = np.corrcoef(ra, rb)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = max(1, min(k, len(a)))
    ia = set(np.argpartition(np.abs(a), -k)[-k:].tolist())
    ib = set(np.argpartition(np.abs(b), -k)[-k:].tolist())
    return len(ia & ib) / k


def topk_sign_agreement(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = max(1, min(k, len(a)))
    idx = np.argpartition(np.abs(a), -k)[-k:]
    return float(np.mean(np.sign(a[idx]) == np.sign(b[idx])))


def longest_true_run(mask: Sequence[bool]) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def synthetic_load_data(config: ExperimentConfig, seed: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate non-IID, correlated hierarchical load with regime events."""
    rng = np.random.default_rng(seed)
    min_points = config.history + config.calibration_steps + config.n_steps + 420
    total_points = max(720, min_points)
    timestamps = pd.date_range("2025-01-01", periods=total_points, freq="h")
    n, z = config.n_endpoints, config.n_zones
    zones = np.arange(n) % z
    ev_flag = (rng.random(n) < 0.35).astype(int)
    heat_sensitivity = rng.uniform(0.4, 1.6, n)
    base = rng.uniform(0.4, 1.8, n)
    phase = rng.uniform(-0.7, 0.7, n)
    zone_weather = np.zeros((total_points, z))
    for zone in range(z):
        innovations = rng.normal(0, 0.12, total_points)
        for t in range(1, total_points):
            zone_weather[t, zone] = 0.92 * zone_weather[t - 1, zone] + innovations[t]

    hours = timestamps.hour.to_numpy()
    days = timestamps.dayofweek.to_numpy()
    loads = np.zeros((total_points, n), dtype=float)
    for i in range(n):
        daily = 0.35 * np.sin(2 * np.pi * (hours - 7) / 24 + phase[i])
        evening = 0.55 * ((hours >= 17) & (hours <= 22)).astype(float)
        weekend = 0.18 * (days >= 5).astype(float)
        ev = ev_flag[i] * rng.uniform(0.5, 1.2) * ((hours >= 18) & (hours <= 23))
        common = heat_sensitivity[i] * np.maximum(0, -zone_weather[:, zones[i]])
        noise = np.zeros(total_points)
        innovations = rng.normal(0, 0.09 + 0.03 * rng.random(), total_points)
        for t in range(1, total_points):
            noise[t] = 0.65 * noise[t - 1] + innovations[t]
        loads[:, i] = base[i] + daily + evening + weekend + ev + common + noise

    # A shared cold-spell/EV regime change in the evaluation region.
    event_start = total_points - max(20, config.n_steps // 3)
    event_end = min(total_points, event_start + max(8, config.n_steps // 6))
    affected_zone = 0
    affected = np.where(zones == affected_zone)[0]
    loads[event_start:event_end, affected] += np.linspace(0.3, 1.5, event_end - event_start)[:, None]
    loads = np.clip(loads, 0.01, None)
    frame = pd.DataFrame(loads, index=timestamps, columns=[f"H{i:03d}" for i in range(n)])
    return frame, ev_flag


def download_uci_eld(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    extracted = cache_dir / "ElectricityLoadDiagrams20112014.txt"
    if extracted.exists():
        return extracted
    archive = cache_dir / "electricityloaddiagrams20112014.zip"
    url = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
    if not archive.exists():
        print(f"Downloading UCI ElectricityLoadDiagrams20112014 from {url}")
        urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as handle:
        candidates = [name for name in handle.namelist() if name.lower().endswith(".txt")]
        if not candidates:
            raise RuntimeError("UCI archive contains no TXT data file")
        with handle.open(candidates[0]) as src, extracted.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return extracted


def parse_uci_datetime_index(values: pd.Index) -> pd.DatetimeIndex:
    """Parse the UCI ELD source's ISO-like timestamps without date guessing.

    ``dayfirst=True`` is unsafe here: on newer pandas versions it can infer
    ``%Y-%d-%m`` from the first ambiguous January rows and then fail at day 13.
    The official UCI file consistently stores ``YYYY-MM-DD HH:MM:SS``.
    """
    cleaned = pd.Index(values).astype(str).str.strip()
    return pd.DatetimeIndex(
        pd.to_datetime(cleaned, format="%Y-%m-%d %H:%M:%S", errors="raise")
    )


def load_uci_data(config: ExperimentConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    path = download_uci_eld(Path(config.cache_dir))
    frame = pd.read_csv(path, sep=";", decimal=",", index_col=0, low_memory=False)
    frame.index = parse_uci_datetime_index(frame.index)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    # Hourly resolution reduces cost and gives interpretable 24-hour lags.
    frame = frame.resample("h").mean().interpolate(limit_direction="both")
    active = (frame > 0).mean(axis=0)
    variance = frame.var(axis=0)
    score = active * np.log1p(variance)
    selected = score.sort_values(ascending=False).head(config.n_endpoints).index
    frame = frame.loc[:, selected].copy()
    if frame.shape[1] < config.n_endpoints:
        raise ValueError(f"Only {frame.shape[1]} usable UCI endpoints found")
    evening = frame.between_time("17:00", "23:00").mean()
    overall = frame.mean().replace(0, np.nan)
    ratio = (evening / overall).fillna(0)
    private_attribute = (ratio > ratio.median()).astype(int).to_numpy()
    return frame, private_attribute


OPSD_URL = (
    "https://data.open-power-system-data.org/household_data/2020-04-15/"
    "household_data_60min_singleindex.csv"
)


def download_opsd_household(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "household_data_60min_singleindex.csv"
    if not target.exists():
        print(f"Downloading OPSD Household Data from {OPSD_URL}")
        urllib.request.urlretrieve(OPSD_URL, target)
    return target


def load_opsd_data(config: ExperimentConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load measured OPSD household/building grid-import channels.

    The package contains 11 southern-German households/buildings. We use only
    building-level ``*_grid_import`` channels; appliance-level channels are not
    treated as independent endpoints. Residential/non-residential type is used
    only for the empirical attribute-inference sanity check.
    """
    path = download_opsd_household(Path(config.cache_dir))
    frame = pd.read_csv(path, low_memory=False)
    timestamp_col = "utc_timestamp" if "utc_timestamp" in frame.columns else frame.columns[0]
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="coerce", utc=True)
    endpoint_columns = [
        column
        for column in frame.columns
        if column.lower().endswith("_grid_import")
    ]
    if not endpoint_columns:
        raise ValueError("OPSD file has no building-level *_grid_import columns")
    cumulative = frame.set_index(timestamp_col)[endpoint_columns].apply(pd.to_numeric, errors="coerce")
    coverage = cumulative.notna().mean(axis=0)
    eligible = coverage[coverage >= 0.25].sort_values(ascending=False).index.tolist()
    if len(eligible) < max(4, config.n_zones):
        eligible = coverage.sort_values(ascending=False).index.tolist()

    needed = config.history + config.calibration_steps + config.n_steps + 120

    def longest_valid_segment(mask: np.ndarray) -> Tuple[int, int]:
        best_start = best_end = start = 0
        active = False
        for idx, value in enumerate(mask):
            if value and not active:
                start, active = idx, True
            if active and (not value or idx == len(mask) - 1):
                end = idx if not value else idx + 1
                if end - start > best_end - best_start:
                    best_start, best_end = start, end
                active = False
        return best_start, best_end

    wide = None
    selected: List[str] = []
    maximum = min(config.n_endpoints, len(eligible))
    minimum = max(4, min(config.n_zones, maximum))
    for count in range(maximum, minimum - 1, -1):
        candidate_columns = eligible[:count]
        # OPSD values are cumulative meter readings. First differences are
        # hourly imported energy; negative differences indicate meter resets.
        interval = cumulative[candidate_columns].diff()
        interval[interval < 0] = np.nan
        interval = interval.interpolate(limit=6, limit_direction="both")
        start, end = longest_valid_segment(interval.notna().all(axis=1).to_numpy())
        if end - start >= needed:
            wide = interval.iloc[start:end].copy()
            selected = candidate_columns
            break
    if wide is None:
        raise ValueError(
            "OPSD channels do not provide a sufficiently long common valid window; "
            "reduce --n-endpoints or --n-steps"
        )
    # Robustly limit documented sensor spikes after conversion to interval use.
    for column in wide.columns:
        upper = wide[column].quantile(0.999)
        if np.isfinite(upper):
            wide[column] = wide[column].clip(lower=0.0, upper=upper)
    wide = wide.interpolate(limit_direction="both").fillna(0.0)
    private_attribute = np.array([int("residential" in column.lower()) for column in wide.columns])
    return wide, private_attribute


def load_london_data(config: ExperimentConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    if not config.london_dir:
        raise ValueError("--london-dir is required for dataset=london")
    root = Path(config.london_dir)
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No London CSV files found under {root}")
    chunks: List[pd.DataFrame] = []
    required = {"LCLid", "tstp", "energy(kWh/hh)"}
    for path in files[: max(12, config.n_endpoints // 2)]:
        try:
            sample = pd.read_csv(path, nrows=5)
            if not required.issubset(sample.columns):
                continue
            part = pd.read_csv(path, usecols=list(required))
            chunks.append(part)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
    if not chunks:
        raise ValueError("No London block CSV had LCLid/tstp/energy(kWh/hh)")
    long = pd.concat(chunks, ignore_index=True)
    long["tstp"] = pd.to_datetime(long["tstp"], errors="coerce")
    long["energy(kWh/hh)"] = pd.to_numeric(long["energy(kWh/hh)"], errors="coerce")
    counts = long.groupby("LCLid")["energy(kWh/hh)"].count().nlargest(config.n_endpoints)
    long = long[long["LCLid"].isin(counts.index)].dropna()
    wide = long.pivot_table(index="tstp", columns="LCLid", values="energy(kWh/hh)", aggfunc="mean")
    wide = wide.resample("h").sum(min_count=1).interpolate(limit_direction="both")
    wide = wide.loc[:, counts.index.intersection(wide.columns)]
    evening = wide.between_time("17:00", "23:00").mean()
    ratio = (evening / wide.mean().replace(0, np.nan)).fillna(0)
    return wide, (ratio > ratio.median()).astype(int).to_numpy()


FEATURE_NAMES = [
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "lag_24",
    "mean_4",
    "mean_24",
    "std_24",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


def feature_matrix(values: np.ndarray, timestamps: pd.DatetimeIndex, times: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[np.ndarray] = []
    targets: List[float] = []
    n_endpoints = values.shape[1]
    for t in times:
        hour = timestamps[t].hour
        dow = timestamps[t].dayofweek
        cal = np.array(
            [
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
                math.sin(2 * math.pi * dow / 7),
                math.cos(2 * math.pi * dow / 7),
            ]
        )
        for i in range(n_endpoints):
            hist = values[t - 24 : t, i]
            row = np.concatenate(
                [
                    values[[t - 1, t - 2, t - 4, t - 8, t - 24], i],
                    [np.mean(hist[-4:]), np.mean(hist), np.std(hist)],
                    cal,
                ]
            )
            rows.append(row)
            targets.append(values[t, i])
    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def prepare_data(config: ExperimentConfig, seed: int) -> PreparedData:
    if config.dataset == "synthetic":
        wide, private_attribute = synthetic_load_data(config, seed)
        source_name = "Synthetic hierarchical smoke-test data"
        source_url = "generated locally"
        source_sha256 = "not applicable"
    elif config.dataset == "uci":
        wide, private_attribute = load_uci_data(config)
        source_path = Path(config.cache_dir) / "ElectricityLoadDiagrams20112014.txt"
        source_name = "UCI ElectricityLoadDiagrams20112014"
        source_url = "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014"
        source_sha256 = file_sha256(source_path)
    elif config.dataset == "opsd":
        wide, private_attribute = load_opsd_data(config)
        source_path = Path(config.cache_dir) / "household_data_60min_singleindex.csv"
        source_name = "Open Power System Data Household Data 2020-04-15"
        source_url = OPSD_URL
        source_sha256 = file_sha256(source_path)
    elif config.dataset == "london":
        wide, private_attribute = load_london_data(config)
        source_name = "Low Carbon London Smart Meter Data"
        source_url = "https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d"
        source_sha256 = "local multi-file input; see run manifest"
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    wide = wide.sort_index().replace([np.inf, -np.inf], np.nan)
    wide = wide.interpolate(limit_direction="both").fillna(0.0)
    if wide.shape[1] > config.n_endpoints:
        wide = wide.iloc[:, : config.n_endpoints]
        private_attribute = private_attribute[: config.n_endpoints]
    n = wide.shape[1]
    if n < max(4, config.n_zones):
        raise ValueError("Too few usable endpoints")
    total = len(wide)
    needed = config.history + config.calibration_steps + config.n_steps + 120
    if total < needed:
        raise ValueError(f"Dataset has {total} points; at least {needed} required")

    raw = np.log1p(np.clip(wide.to_numpy(dtype=float), 0, None))
    train_end = min(int(total * 0.65), total - config.calibration_steps - config.n_steps)
    eval_start = total - config.n_steps
    cal_start = eval_start - config.calibration_steps
    if train_end >= cal_start:
        train_end = cal_start - 1
    mean = raw[:train_end].mean(axis=0)
    std = raw[:train_end].std(axis=0)
    std[std < 1e-6] = 1.0
    values = (raw - mean) / std

    train_times = np.arange(config.history, train_end)
    cal_times = np.arange(cal_start, eval_start)
    eval_times = np.arange(eval_start, total)
    X_train, y_train = feature_matrix(values, wide.index, train_times)
    X_cal_flat, y_cal_flat = feature_matrix(values, wide.index, cal_times)
    X_eval_flat, y_eval_flat = feature_matrix(values, wide.index, eval_times)

    rng = np.random.default_rng(seed)
    if len(X_train) > config.train_max_rows:
        idx = rng.choice(len(X_train), config.train_max_rows, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    d = X_train.shape[1]
    X_cal = X_cal_flat.reshape(len(cal_times), n, d)
    y_cal = y_cal_flat.reshape(len(cal_times), n)
    X_eval = X_eval_flat.reshape(len(eval_times), n, d)
    y_eval = y_eval_flat.reshape(len(eval_times), n)
    zones = np.arange(n) % config.n_zones
    baseline = np.median(X_train, axis=0)
    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_cal=X_cal,
        y_cal=y_cal,
        X_eval=X_eval,
        y_eval=y_eval,
        cal_load=y_cal,
        eval_load=y_eval,
        feature_names=FEATURE_NAMES.copy(),
        feature_baseline=baseline,
        zones=zones,
        private_attribute=np.asarray(private_attribute, dtype=int),
        eval_timestamps=wide.index[eval_times],
        endpoint_ids=[str(column) for column in wide.columns],
        source_window_start=str(wide.index.min()),
        source_window_end=str(wide.index.max()),
        source_name=source_name,
        source_url=source_url,
        source_sha256=source_sha256,
    )


class TreeForecastModel:
    def __init__(self, config: ExperimentConfig, seed: int):
        self.config = config
        self.seed = seed
        self.model: RegressorMixin
        self.backend = "extra_trees"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TreeForecastModel":
        if self.config.preset == "full":
            try:
                from lightgbm import LGBMRegressor

                self.model = LGBMRegressor(
                    n_estimators=self.config.tree_estimators,
                    learning_rate=0.035,
                    num_leaves=31,
                    max_depth=-1,
                    subsample=0.85,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=self.seed,
                    n_jobs=-1,
                    verbosity=-1,
                )
                self.backend = "lightgbm"
            except ImportError:
                print("lightgbm unavailable; falling back to ExtraTreesRegressor")
                self.model = ExtraTreesRegressor(
                    n_estimators=self.config.tree_estimators,
                    min_samples_leaf=2,
                    max_features=0.9,
                    n_jobs=-1,
                    random_state=self.seed,
                )
        else:
            self.model = ExtraTreesRegressor(
                n_estimators=self.config.quick_tree_estimators,
                min_samples_leaf=2,
                max_features=0.9,
                n_jobs=-1,
                random_state=self.seed,
            )
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float)


class NeuralForecastModel:
    def __init__(self, config: ExperimentConfig, seed: int, n_features: int):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("The neural experiment requires torch>=2.2") from exc
        self.torch = torch
        self.config = config
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class GatedResidualBlock(nn.Module):
            def __init__(self, width: int, dropout: float):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(width, width * 2),
                    nn.BatchNorm1d(width * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(width * 2, width),
                    nn.BatchNorm1d(width),
                )
                self.gate = nn.Sequential(nn.Linear(width, width), nn.Sigmoid())
                self.activation = nn.SiLU()

            def forward(self, x):
                update = self.net(x)
                return self.activation(x + self.gate(x) * update)

        class ResidualTabularDNN(nn.Module):
            def __init__(self, d: int):
                super().__init__()
                width = 192 if d >= 12 else 128
                self.stem = nn.Sequential(
                    nn.Linear(d, width),
                    nn.BatchNorm1d(width),
                    nn.SiLU(),
                    nn.Dropout(0.05),
                )
                self.blocks = nn.Sequential(
                    GatedResidualBlock(width, 0.08),
                    GatedResidualBlock(width, 0.08),
                    GatedResidualBlock(width, 0.06),
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, 96),
                    nn.SiLU(),
                    nn.Dropout(0.04),
                    nn.Linear(96, 32),
                    nn.SiLU(),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                z = self.stem(x)
                z = self.blocks(z)
                return self.head(z).squeeze(-1)

        self.model = ResidualTabularDNN(n_features).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralForecastModel":
        torch = self.torch
        from torch.utils.data import DataLoader, TensorDataset

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(X))
        val_count = min(max(256, int(0.12 * len(X))), max(1, len(X) // 3))
        val_idx, train_idx = order[:val_count], order[val_count:]
        train_ds = TensorDataset(
            torch.tensor(X[train_idx], dtype=torch.float32),
            torch.tensor(y[train_idx], dtype=torch.float32),
        )
        batch_size = 1024 if len(train_idx) > 20000 else 512
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32, device=self.device)
        y_val = torch.tensor(y[val_idx], dtype=torch.float32, device=self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1.2e-3, weight_decay=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.config.neural_epochs)
        )
        loss_fn = torch.nn.SmoothL1Loss(beta=0.25)
        best_state, best_loss, patience = None, float("inf"), 14
        averaged_state, averaged_count = None, 0
        stale = 0
        for epoch in range(self.config.neural_epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                optimizer.step()
            scheduler.step()
            self.model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model(X_val), y_val).cpu())
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if epoch >= max(3, self.config.neural_epochs // 2):
                current = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                    if torch.is_floating_point(v)
                }
                if averaged_state is None:
                    averaged_state = current
                else:
                    for key, value in current.items():
                        averaged_state[key].mul_(averaged_count / (averaged_count + 1))
                        averaged_state[key].add_(value / (averaged_count + 1))
                averaged_count += 1
            if stale >= patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if averaged_state and averaged_count >= 3:
            candidate_state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }
            for key, value in averaged_state.items():
                candidate_state[key] = value
            self.model.load_state_dict(candidate_state, strict=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), 2048):
                xb = torch.tensor(X[start : start + 2048], dtype=torch.float32, device=self.device)
                outputs.append(self.model(xb).cpu().numpy())
        return np.concatenate(outputs).astype(float)

    def integrated_gradients(self, X: np.ndarray, baseline: np.ndarray, steps: int) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        baseline_t = torch.tensor(baseline, dtype=torch.float32, device=self.device)
        results: List[np.ndarray] = []
        alphas = torch.linspace(0.0, 1.0, steps, device=self.device)
        for start in range(0, len(X), 256):
            xb = torch.tensor(X[start : start + 256], dtype=torch.float32, device=self.device)
            total_grad = torch.zeros_like(xb)
            for alpha in alphas:
                interp = (baseline_t + alpha * (xb - baseline_t)).detach().requires_grad_(True)
                output = self.model(interp).sum()
                grad = torch.autograd.grad(output, interp, retain_graph=False)[0]
                total_grad += grad
            ig = (xb - baseline_t) * total_grad / steps
            results.append(ig.detach().cpu().numpy())
        return np.concatenate(results).astype(float)

    def gradient_x_input(self, X: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        baseline_t = torch.tensor(baseline, dtype=torch.float32, device=self.device)
        results: List[np.ndarray] = []
        for start in range(0, len(X), 512):
            xb = torch.tensor(
                X[start : start + 512],
                dtype=torch.float32,
                device=self.device,
            ).requires_grad_(True)
            output = self.model(xb).sum()
            grad = torch.autograd.grad(output, xb, retain_graph=False)[0]
            attribution = (xb - baseline_t) * grad
            results.append(attribution.detach().cpu().numpy())
        return np.concatenate(results).astype(float)


def ablation_explanations(model: Any, X: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Prediction difference after replacing one standardized feature."""
    original = model.predict(X)
    phi = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        perturbed = X.copy()
        perturbed[:, j] = baseline[j]
        phi[:, j] = original - model.predict(perturbed)
    return phi


def tree_explanations(
    model: TreeForecastModel,
    X: np.ndarray,
    baseline: np.ndarray,
    xai_method: str,
) -> Tuple[np.ndarray, str]:
    if xai_method == "default":
        xai_method = "treeshap" if model.config.preset == "full" else "feature_ablation"
    if xai_method == "treeshap":
        try:
            import shap

            explainer = shap.TreeExplainer(model.model)
            values = explainer.shap_values(X)
            if isinstance(values, list):
                values = values[0]
            return np.asarray(values, dtype=float), "TreeSHAP"
        except Exception as exc:
            print(f"TreeSHAP failed ({exc}); using feature ablation")
            return ablation_explanations(model, X, baseline), "feature_ablation"
    if xai_method == "feature_ablation":
        return ablation_explanations(model, X, baseline), "feature_ablation"
    raise ValueError(f"XAI method {xai_method!r} is not valid for tree models")


@dataclass
class ModelOutputs:
    phi_cal: np.ndarray
    phi_eval: np.ndarray
    pred_cal: np.ndarray
    pred_eval: np.ndarray
    forecast_rmse: float
    forecast_mae: float
    model_backend: str
    explainer_backend: str


def train_and_explain(
    data: PreparedData,
    config: ExperimentConfig,
    model_family: str,
    seed: int,
    xai_method: str = "default",
) -> ModelOutputs:
    scaler = StandardScaler().fit(data.X_train)
    X_train = scaler.transform(data.X_train)
    X_cal_flat = scaler.transform(data.X_cal.reshape(-1, data.n_features))
    X_eval_flat = scaler.transform(data.X_eval.reshape(-1, data.n_features))
    baseline = scaler.transform(data.feature_baseline.reshape(1, -1))[0]

    if model_family == "tree":
        model = TreeForecastModel(config, seed).fit(X_train, data.y_train)
        phi_cal_flat, explainer_backend = tree_explanations(
            model, X_cal_flat, baseline, xai_method
        )
        phi_eval_flat, _ = tree_explanations(model, X_eval_flat, baseline, xai_method)
        model_backend = model.backend
    elif model_family == "neural":
        model = NeuralForecastModel(config, seed, data.n_features).fit(X_train, data.y_train)
        method = "integrated_gradients" if xai_method == "default" else xai_method
        if method == "integrated_gradients":
            phi_cal_flat = model.integrated_gradients(X_cal_flat, baseline, config.ig_steps)
            phi_eval_flat = model.integrated_gradients(X_eval_flat, baseline, config.ig_steps)
            explainer_backend = "IntegratedGradients"
        elif method == "gradient_x_input":
            phi_cal_flat = model.gradient_x_input(X_cal_flat, baseline)
            phi_eval_flat = model.gradient_x_input(X_eval_flat, baseline)
            explainer_backend = "GradientXInput"
        else:
            raise ValueError(f"XAI method {xai_method!r} is not valid for neural models")
        model_backend = "PyTorchResidualDNN"
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    pred_cal = model.predict(X_cal_flat).reshape(data.X_cal.shape[:2])
    pred_eval = model.predict(X_eval_flat).reshape(data.X_eval.shape[:2])
    phi_cal = phi_cal_flat.reshape(data.X_cal.shape)
    phi_eval = phi_eval_flat.reshape(data.X_eval.shape)
    return ModelOutputs(
        phi_cal=phi_cal,
        phi_eval=phi_eval,
        pred_cal=pred_cal,
        pred_eval=pred_eval,
        forecast_rmse=float(np.sqrt(mean_squared_error(data.y_eval.ravel(), pred_eval.ravel()))),
        forecast_mae=float(mean_absolute_error(data.y_eval.ravel(), pred_eval.ravel())),
        model_backend=model_backend,
        explainer_backend=explainer_backend,
    )


def compatible_xai_methods(model_family: str, configured: Sequence[str]) -> List[str]:
    if not configured:
        configured = ["default"]
    if "default" in configured:
        return ["default"]
    allowed = {
        "tree": {"treeshap", "feature_ablation"},
        "neural": {"integrated_gradients", "gradient_x_input"},
    }[model_family]
    selected = [method for method in configured if method in allowed]
    if not selected:
        raise ValueError(
            f"No configured XAI method is compatible with {model_family}; "
            f"allowed methods are {sorted(allowed)}"
        )
    return selected


class ZCDPAccountant:
    """Scalar Gaussian accountant with exact additive zCDP composition.

    A Gaussian mechanism with L2 sensitivity C and noise sigma*C spends
    rho=1/(2*sigma^2)-zCDP. Conversion uses
    epsilon = rho + 2*sqrt(rho*log(1/delta)). zCDP implies RDP at every
    order, so this is compatible with the manuscript's cumulative RDP view.
    """

    def __init__(self, epsilon_max: float, delta: float):
        if epsilon_max <= 0 or not 0 < delta < 1:
            raise ValueError("Invalid privacy parameters")
        self.epsilon_max = float(epsilon_max)
        self.delta = float(delta)
        log_term = math.log(1.0 / delta)
        self.max_rho = (math.sqrt(log_term + epsilon_max) - math.sqrt(log_term)) ** 2
        self.spent_rho = 0.0

    @staticmethod
    def cost(sigma: float) -> float:
        return 1.0 / (2.0 * sigma * sigma)

    def can_spend(self, sigma: float) -> bool:
        return self.spent_rho + self.cost(sigma) <= self.max_rho + 1e-12

    def spend(self, sigma: float) -> None:
        cost = self.cost(sigma)
        if self.spent_rho + cost > self.max_rho + 1e-10:
            raise RuntimeError("Privacy budget exceeded")
        self.spent_rho += cost

    @property
    def remaining_rho(self) -> float:
        return max(0.0, self.max_rho - self.spent_rho)

    @property
    def epsilon_spent(self) -> float:
        if self.spent_rho <= 0:
            return 0.0
        return self.spent_rho + 2.0 * math.sqrt(self.spent_rho * math.log(1.0 / self.delta))


@dataclass
class NetworkTrace:
    spectral_efficiency: np.ndarray
    edge_delay_ms: np.ndarray
    packet_failure: np.ndarray
    distances_m: np.ndarray
    base_gaussian: np.ndarray


def make_network_trace(config: ExperimentConfig, data: PreparedData, seed: int) -> NetworkTrace:
    rng = np.random.default_rng(seed + 100_003)
    t_steps, n = data.n_steps, data.n_endpoints
    distances = rng.uniform(40.0, 450.0, n)
    shadow = np.zeros((t_steps, n))
    innovations = rng.normal(0, 2.5, size=(t_steps, n))
    for t in range(1, t_steps):
        shadow[t] = 0.92 * shadow[t - 1] + math.sqrt(1 - 0.92**2) * innovations[t]
    fading_power = rng.exponential(1.0, size=(t_steps, n))
    # Simplified 3GPP UMi LOS form: PL=32.4+21log10(d)+20log10(fc_GHz).
    pathloss = 32.4 + 21.0 * np.log10(distances[None, :]) + 20.0 * math.log10(config.carrier_ghz)
    rx_dbm = config.tx_power_dbm - pathloss - shadow + 10.0 * np.log10(fading_power + 1e-12)
    noise_dbm = -174.0 + 10.0 * math.log10(config.rb_bandwidth_hz) + config.noise_figure_db
    snr_linear = 10 ** ((rx_dbm - noise_dbm) / 10.0)
    se = np.clip(np.log2(1.0 + snr_linear), 0.10, 7.40)
    edge_delay = rng.lognormal(mean=math.log(max(config.edge_delay_ms, 0.1)), sigma=0.18, size=(t_steps, n))

    bad = np.zeros((t_steps, n), dtype=bool)
    for t in range(1, t_steps):
        enter_bad = rng.random(n) < 0.03
        leave_bad = rng.random(n) < 0.40
        bad[t] = np.where(bad[t - 1], ~leave_bad, enter_bad)
    fail_probability = np.where(bad, 0.45, 0.02)
    packet_failure = rng.random((t_steps, n)) < fail_probability
    base_gaussian = rng.normal(size=(t_steps, n, data.n_features))
    return NetworkTrace(se, edge_delay, packet_failure, distances, base_gaussian)


def apply_stress_to_trace(trace: NetworkTrace, seed: int, scenario: str) -> NetworkTrace:
    """Create controlled wireless stress scenarios from the base trace."""
    if scenario == "baseline":
        return trace
    rng = np.random.default_rng(seed + 909_001)
    se = trace.spectral_efficiency.copy()
    edge_delay = trace.edge_delay_ms.copy()
    packet_failure = trace.packet_failure.copy()
    distances = trace.distances_m.copy()
    base_gaussian = trace.base_gaussian.copy()
    t_steps, n = se.shape

    if scenario == "burst_loss":
        bad = np.zeros((t_steps, n), dtype=bool)
        for t in range(1, t_steps):
            enter_bad = rng.random(n) < 0.06
            leave_bad = rng.random(n) < 0.12
            bad[t] = np.where(bad[t - 1], ~leave_bad, enter_bad)
        stress_failure = rng.random((t_steps, n)) < np.where(bad, 0.70, 0.04)
        packet_failure = packet_failure | stress_failure
        edge_delay = edge_delay * np.where(bad, 1.35, 1.0)
    elif scenario == "weak_channel":
        weak_count = max(1, int(math.ceil(0.25 * n)))
        weak_idx = np.argsort(distances)[-weak_count:]
        se[:, weak_idx] = np.maximum(0.10, 0.45 * se[:, weak_idx])
        extra_fail = np.zeros((t_steps, n), dtype=bool)
        extra_fail[:, weak_idx] = rng.random((t_steps, weak_count)) < 0.12
        packet_failure = packet_failure | extra_fail
    else:
        raise ValueError(f"Unknown stress scenario: {scenario}")

    return NetworkTrace(se, edge_delay, packet_failure, distances, base_gaussian)


def packet_size_bits(mode: PacketMode, d: int, base_bits: int) -> Tuple[int, int]:
    k = max(1, min(d, int(math.ceil(mode.k_fraction * d))))
    index_bits = int(math.ceil(math.log2(max(2, d))))
    # Value, sign, index, quantization scale, uncertainty/certificate overhead.
    group_count = 4 if d >= 12 else max(1, min(4, d))
    bits = (
        base_bits
        + k * (index_bits + mode.quant_bits + 1 + 16)
        + group_count * (mode.quant_bits + 16)
    )
    return bits, k


def network_cost(
    mode: PacketMode, d: int, config: ExperimentConfig, trace: NetworkTrace, t: int, i: int
) -> Tuple[int, float, float, int]:
    bits, k = packet_size_bits(mode, d, config.base_packet_bits)
    tx_budget_ms = config.deadline_ms - trace.edge_delay_ms[t, i]
    if tx_budget_ms <= 0:
        return config.n_resource_blocks + 1, float("inf"), 0.0, k
    capacity_per_rb = (
        trace.spectral_efficiency[t, i]
        * config.rb_bandwidth_hz
        * (tx_budget_ms / 1000.0)
    )
    rbs = int(math.ceil(bits / max(capacity_per_rb, 1e-12)))
    tx_ms = 1000.0 * bits / max(
        rbs * config.rb_bandwidth_hz * trace.spectral_efficiency[t, i], 1e-12
    )
    latency_ms = tx_ms + trace.edge_delay_ms[t, i]
    tx_power_w = 10 ** ((config.tx_power_dbm - 30.0) / 10.0)
    energy_mj = tx_power_w * tx_ms
    return rbs, latency_ms, energy_mj, k


def privatize_and_compress(
    delta: np.ndarray,
    clip_norm: float,
    sensitivity: float,
    mode: PacketMode,
    base_noise: np.ndarray,
) -> Tuple[np.ndarray, float]:
    norm = np.linalg.norm(delta)
    clipped = delta * min(1.0, clip_norm / max(norm, 1e-12))
    private = clipped + mode.sigma * sensitivity * base_noise
    d = len(private)
    k = max(1, min(d, int(math.ceil(mode.k_fraction * d))))
    idx = np.argpartition(np.abs(private), -k)[-k:]
    sparse = np.zeros(d, dtype=float)
    dynamic_range = max(np.max(np.abs(private[idx])), 1e-8)
    levels = 2 ** mode.quant_bits - 1
    normalized = np.clip(private[idx] / dynamic_range, -1.0, 1.0)
    quantized = np.round((normalized + 1.0) * 0.5 * levels) / levels
    sparse[idx] = (2.0 * quantized - 1.0) * dynamic_range
    # Task-semantic group summaries are post-processing of the already private
    # vector. They preserve the operator's driver categories even when only a
    # few individual coordinates are transmitted.
    if d >= 12:
        groups = [np.arange(0, 4), np.arange(4, 8), np.arange(8, 10), np.arange(10, 12)]
    else:
        groups = [group for group in np.array_split(np.arange(d), min(4, d)) if len(group)]
    for group in groups:
        target_sum = float(np.sum(private[group]))
        scale = max(abs(target_sum), 1e-8)
        target_quantized = round(((target_sum / scale) + 1.0) * 0.5 * levels) / levels
        target_quantized = (2.0 * target_quantized - 1.0) * scale
        correction = (target_quantized - float(np.sum(sparse[group]))) / len(group)
        sparse[group] += correction
    compression_error = float(np.linalg.norm(private - sparse))
    return sparse, compression_error


@dataclass
class Candidate:
    endpoint: int
    mode: PacketMode
    utility: float
    rbs: int
    latency_ms: float
    energy_mj: float


@dataclass
class AblationFlags:
    use_aox: bool = True
    use_privacy_price: bool = True
    adaptive_resolution: bool = True
    use_hierarchy: bool = True
    use_fairness: bool = True
    residual_accumulation: bool = True


def mode_quality(mode: PacketMode) -> float:
    # Four driver-group summaries are always carried, so even a sparse packet
    # retains task-level semantics while detailed coordinates improve full
    # attribution fidelity.
    resolution = 0.40 + 0.60 * mode.k_fraction
    quantization = 1.0 - 2.0 ** (-mode.quant_bits)
    privacy = math.exp(-0.075 * mode.sigma)
    return float(resolution * quantization * privacy)


def multiple_choice_knapsack(candidates: Mapping[int, List[Candidate]], capacity: int) -> List[Candidate]:
    """Exact per-slot multiple-choice knapsack; one mode per endpoint."""
    dp: List[Tuple[float, List[Candidate]]] = [(-float("inf"), []) for _ in range(capacity + 1)]
    dp[0] = (0.0, [])
    for endpoint in sorted(candidates):
        options: List[Optional[Candidate]] = [None] + candidates[endpoint]
        new_dp: List[Tuple[float, List[Candidate]]] = [(-float("inf"), []) for _ in range(capacity + 1)]
        for used, (value, chosen) in enumerate(dp):
            if not np.isfinite(value):
                continue
            for option in options:
                cost = 0 if option is None else option.rbs
                score = 0.0 if option is None else option.utility
                if used + cost <= capacity and value + score > new_dp[used + cost][0]:
                    new_dp[used + cost] = (value + score, chosen + ([] if option is None else [option]))
        dp = new_dp
    _, selection = max(dp, key=lambda item: item[0])
    return selection


def driver_group(phi: np.ndarray) -> int:
    groups = (np.arange(0, 4), np.arange(4, 8), np.arange(8, 10), np.arange(10, 12))
    scores = [float(np.sum(np.abs(phi[idx]))) for idx in groups]
    return int(np.argmax(scores))


def intervention_active(phi: np.ndarray, threshold: float) -> bool:
    """Whether a reconstructed explanation is strong enough to drive action."""
    return bool(np.linalg.norm(phi) > threshold)


def expected_random_topk(k: int, d: int) -> float:
    if d <= 0:
        return float("nan")
    k = max(1, min(k, d))
    return float(k / d)


def sensitivity_multiplier(config: ExperimentConfig) -> float:
    if config.adjacency == "add_remove":
        return 1.0
    if config.adjacency == "replace_one":
        return 2.0
    raise ValueError("adjacency must be 'add_remove' or 'replace_one'")


def budget_calibrated_sigma(accountant: ZCDPAccountant, config: ExperimentConfig) -> float:
    """Noise level that spends the remaining horizon budget over a release cap.

    The feasibility ladder showed that using very noisy packets many times can
    erase Top-K signal. Selective PRISM instead spends privacy only on public
    high-value releases and calibrates each release to the declared cap.
    """
    cap = max(1, int(config.max_private_releases_per_endpoint))
    per_release_rho = accountant.max_rho / cap
    sigma = math.sqrt(1.0 / (2.0 * max(per_release_rho, 1e-12)))
    sigma *= max(float(config.release_cap_sigma_safety), 1.0)
    return max(float(config.min_budget_calibrated_sigma), sigma)


def selective_mode(mode: PacketMode, accountant: ZCDPAccountant, config: ExperimentConfig) -> PacketMode:
    if not config.selective_private_release:
        return mode
    sigma = min(float(mode.sigma), budget_calibrated_sigma(accountant, config))
    return PacketMode(f"{mode.name}_selective", mode.k_fraction, mode.quant_bits, sigma)


def is_map_fallback_mode(mode: PacketMode) -> bool:
    return mode.name == REPAIRED_MAP_PACKET.name or mode.name.startswith(
        f"{REPAIRED_MAP_PACKET.name}_"
    )


def public_release_score(
    aox_value: float,
    critical_value: float,
    novelty_value: float,
    fairness_so_far: float,
    config: ExperimentConfig,
) -> float:
    fairness_deficit = max(0.0, config.min_jain_fairness - fairness_so_far)
    return (
        3.0 * float(critical_value)
        + 1.5 * float(novelty_value)
        + 0.05 * float(aox_value)
        + 2.0 * fairness_deficit
    )


def student_t_ci(values: Sequence[float]) -> Tuple[float, float, int]:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    n = int(len(clean))
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(clean.mean())
    if n == 1:
        return mean, 0.0, 1
    std = float(clean.std(ddof=1))
    critical = float(student_t.ppf(0.975, df=n - 1))
    return mean, critical * std / math.sqrt(n), n


def zone_mean(matrix: np.ndarray, zones: np.ndarray, z: int) -> np.ndarray:
    return np.mean(matrix[zones == z], axis=0)


def calibrate_simulation(
    outputs: ModelOutputs, data: PreparedData, config: ExperimentConfig
) -> Dict[str, np.ndarray]:
    phi = outputs.phi_cal
    diff = np.diff(phi, axis=0)
    clip_norms = np.quantile(np.linalg.norm(diff, axis=2), config.clip_quantile, axis=0)
    positive = clip_norms[clip_norms > 1e-7]
    floor = np.median(positive) * 0.1 if len(positive) else 1e-3
    clip_norms = np.maximum(clip_norms, max(floor, 1e-5))

    residuals = np.zeros_like(phi)
    zone_norms: List[float] = []
    for t in range(phi.shape[0]):
        for z in range(config.n_zones):
            zphi = zone_mean(phi[t], data.zones, z)
            zone_norms.append(float(np.linalg.norm(zphi)))
            residuals[t, data.zones == z] = phi[t, data.zones == z] - zphi
    residual_mu = residuals.mean(axis=0)
    residual_scale = residuals.std(axis=0)
    residual_scale[residual_scale < 1e-5] = 1e-5

    pred_cal_z = np.column_stack(
        [outputs.pred_cal[:, data.zones == z].mean(axis=1) for z in range(config.n_zones)]
    )
    pred_level_threshold = np.quantile(pred_cal_z, 0.80, axis=0)
    pred_ramp_threshold = np.quantile(np.abs(np.diff(pred_cal_z, axis=0)), 0.90, axis=0)
    endpoint_ramp_scale = np.quantile(np.abs(np.diff(outputs.pred_cal, axis=0)), 0.90, axis=0)
    endpoint_ramp_scale = np.maximum(endpoint_ramp_scale, 1e-5)
    return {
        "clip_norms": clip_norms,
        "residual_mu": residual_mu,
        "residual_scale": residual_scale,
        "pred_level_threshold": pred_level_threshold,
        "pred_ramp_threshold": pred_ramp_threshold,
        "endpoint_ramp_scale": endpoint_ramp_scale,
        "zone_norm_threshold": np.array(np.quantile(zone_norms, 0.90)),
    }


def criticality_matrix(
    pred_eval: np.ndarray, data: PreparedData, calibration: Mapping[str, np.ndarray], config: ExperimentConfig
) -> np.ndarray:
    zpred = np.column_stack(
        [pred_eval[:, data.zones == z].mean(axis=1) for z in range(config.n_zones)]
    )
    ramp = np.vstack([np.zeros((1, config.n_zones)), np.abs(np.diff(zpred, axis=0))])
    critical = (zpred > calibration["pred_level_threshold"][None, :]) | (
        ramp > calibration["pred_ramp_threshold"][None, :]
    )
    return critical.astype(bool)


def hierarchy_proxy(
    recon: np.ndarray,
    zones: np.ndarray,
    residual_mu: np.ndarray,
    residual_scale: np.ndarray,
    use_hierarchy: bool,
) -> np.ndarray:
    n = len(recon)
    if not use_hierarchy:
        return np.zeros(n)
    proxy = np.zeros(n)
    for z in np.unique(zones):
        idx = np.where(zones == z)[0]
        zrecon = recon[idx].mean(axis=0)
        normalized = ((recon[idx] - zrecon) - residual_mu[idx]) / residual_scale[idx]
        proxy[idx] = np.linalg.norm(normalized, axis=1) / math.sqrt(recon.shape[1])
    return np.clip(proxy, 0.0, 10.0)


def fixed_mode() -> PacketMode:
    return PACKET_MODES[1]


def build_candidates(
    scheduler: str,
    flags: AblationFlags,
    t: int,
    aox: np.ndarray,
    critical_by_endpoint: np.ndarray,
    forecast_novelty: np.ndarray,
    hproxy: np.ndarray,
    delivered_counts: np.ndarray,
    private_release_counts: np.ndarray,
    residual_norm: np.ndarray,
    noise_floor: np.ndarray,
    critical_coverage_so_far: float,
    fairness_so_far: float,
    accountants: Sequence[ZCDPAccountant],
    config: ExperimentConfig,
    trace: NetworkTrace,
    d: int,
) -> Mapping[int, List[Candidate]]:
    candidates: Dict[int, List[Candidate]] = {}
    for i in range(len(aox)):
        if (
            config.selective_private_release
            and scheduler in {"prism_xai", "repaired_mapxai"}
            and private_release_counts[i] >= config.max_private_releases_per_endpoint
        ):
            continue
        release_score = public_release_score(
            aox[i],
            critical_by_endpoint[i],
            forecast_novelty[i],
            fairness_so_far,
            config,
        )
        coverage_deficit = max(0.0, config.min_critical_coverage - critical_coverage_so_far)
        if (
            config.selective_private_release
            and scheduler in {"prism_xai", "repaired_mapxai"}
            and release_score < config.public_release_score_threshold
            and not (critical_by_endpoint[i] and coverage_deficit > 0.0)
        ):
            continue
        fallback_gain_per_bit = 0.0
        if scheduler in {"periodic_fixed_dp", "channel_fixed_dp", "aox_fixed_dp"}:
            modes = [fixed_mode()]
        elif scheduler == "repaired_mapxai":
            if hproxy[i] > config.hierarchy_threshold or aox[i] > config.reliability_aox_threshold:
                modes = [PACKET_MODES[2]]
            elif hproxy[i] < 1.0:
                modes = [PACKET_MODES[0]]
            else:
                modes = [PACKET_MODES[1]]
        elif scheduler == "joint_no_compression":
            modes = [PacketMode(f"full_{m.name}", 1.0, 10, m.sigma) for m in PACKET_MODES]
        else:
            if flags.adaptive_resolution:
                # The MAP-XAI fallback is the default valid PRISM action.
                # Higher-detail adaptive modes must beat this fallback on
                # expected distortion reduction per transmitted bit.
                modes = [REPAIRED_MAP_PACKET] + list(PACKET_MODES) + [
                    PacketMode("operational", 0.25, 8, 2.0)
                ]
            else:
                modes = [PacketMode(f"fixed_{m.name}", 0.5, 8, m.sigma) for m in PACKET_MODES]

        endpoint_options: List[Candidate] = []
        signal_to_noise = residual_norm[i] / max(noise_floor[i], 1e-12)
        for mode in modes:
            if config.selective_private_release and scheduler in {"prism_xai", "repaired_mapxai"}:
                mode = selective_mode(mode, accountants[i], config)
            if (
                scheduler == "prism_xai"
                and not is_map_fallback_mode(mode)
                and signal_to_noise < config.residual_noise_margin
            ):
                # Explicit do-not-transmit option for optional adaptive
                # upgrades: the MAP-XAI fallback remains selectable, but
                # additional PRISM packets are suppressed until the residual
                # exceeds the DP noise floor.
                continue
            if not accountants[i].can_spend(mode.sigma):
                continue
            rbs, latency_ms, energy_mj, _ = network_cost(mode, d, config, trace, t, i)
            if rbs < 1 or rbs > config.n_resource_blocks or latency_ms > config.deadline_ms:
                continue
            # Forecasts are assumed to be part of the existing operational
            # telemetry. Their normalized change is therefore a public/task
            # signal that helps prioritize explanation updates without reading
            # a clean explanation at the edge.
            urgency = 1.0 + 3.0 * float(critical_by_endpoint[i]) + 2.0 * float(
                forecast_novelty[i]
            )
            if flags.use_aox:
                urgency *= float(aox[i])
            residual_signal = min(signal_to_noise, 8.0)
            expected_distortion_reduction = (
                residual_norm[i]
                * mode_quality(mode)
                * math.log1p(residual_signal)
                * urgency
            )
            if flags.use_hierarchy:
                expected_distortion_reduction *= 1.0 + 0.03 * hproxy[i]
            fairness = 0.0
            if flags.use_fairness:
                fairness_deficit = max(0.0, config.min_jain_fairness - fairness_so_far)
                fairness = config.fairness_weight * (
                    aox[i] / (1.0 + delivered_counts[i]) + fairness_deficit
                )
            privacy_price = 0.0
            if flags.use_privacy_price:
                spent_fraction = accountants[i].spent_rho / max(accountants[i].max_rho, 1e-12)
                cost_fraction = ZCDPAccountant.cost(mode.sigma) / max(
                    accountants[i].max_rho, 1e-12
                )
                opportunity_weight = 0.10 if critical_by_endpoint[i] else 1.20
                privacy_price = (
                    config.privacy_price_weight
                    * opportunity_weight
                    * cost_fraction
                    * (1.0 + spent_fraction)
                )
            if scheduler == "channel_fixed_dp":
                utility = trace.spectral_efficiency[t, i] / rbs
            elif scheduler == "aox_fixed_dp":
                utility = urgency / rbs + fairness
            elif scheduler == "periodic_fixed_dp":
                utility = 1.0
            elif scheduler == "repaired_mapxai":
                utility = (1.0 + hproxy[i] + 0.2 * aox[i]) / rbs
            else:
                bits, _ = packet_size_bits(mode, d, config.base_packet_bits)
                gain_per_bit = expected_distortion_reduction / max(bits, 1)
                if is_map_fallback_mode(mode):
                    fallback_gain_per_bit = gain_per_bit
                coverage_bonus = (
                    config.critical_recall_weight * coverage_deficit * float(critical_by_endpoint[i])
                )
                if (
                    scheduler == "prism_xai"
                    and not is_map_fallback_mode(mode)
                    and (
                        signal_to_noise < config.residual_upgrade_margin
                        or gain_per_bit
                        < fallback_gain_per_bit * (1.0 + config.override_gain_margin)
                    )
                ):
                    continue
                utility = (
                    config.gain_per_bit_weight * gain_per_bit
                    + config.aox_weight * float(aox[i]) * float(critical_by_endpoint[i]) / max(bits, 1)
                    + fairness / max(bits, 1)
                    + coverage_bonus / max(bits, 1)
                    - config.energy_weight * energy_mj / max(bits, 1)
                    - privacy_price
                )
                if utility <= 0.0 and not (
                    critical_by_endpoint[i] and critical_coverage_so_far < config.min_critical_coverage
                ):
                    continue
            endpoint_options.append(Candidate(i, mode, float(utility), rbs, latency_ms, energy_mj))
        if endpoint_options:
            candidates[i] = endpoint_options
    return candidates


def select_candidates(
    scheduler: str,
    candidates: Mapping[int, List[Candidate]],
    t: int,
    n_endpoints: int,
    capacity: int,
) -> List[Candidate]:
    if scheduler in {"prism_xai", "joint_no_compression"}:
        return multiple_choice_knapsack(candidates, capacity)
    if scheduler == "periodic_fixed_dp":
        selected: List[Candidate] = []
        used = 0
        for offset in range(n_endpoints):
            i = (t + offset) % n_endpoints
            if i not in candidates:
                continue
            option = candidates[i][0]
            if used + option.rbs <= capacity:
                selected.append(option)
                used += option.rbs
        return selected
    flat = [option for options in candidates.values() for option in options]
    flat.sort(key=lambda item: item.utility, reverse=True)
    selected, used, seen = [], 0, set()
    for option in flat:
        if option.endpoint in seen or used + option.rbs > capacity:
            continue
        selected.append(option)
        seen.add(option.endpoint)
        used += option.rbs
    return selected


def select_prism_matched_fallback(
    fallback_selected: Sequence[Candidate],
    prism_candidates: Mapping[int, List[Candidate]],
    d: int,
    config: ExperimentConfig,
    current_bits: int,
    reference_bits: float,
) -> List[Candidate]:
    """Preserve the MAP-XAI fallback coverage skeleton.

    PRISM is allowed to replace an internal fallback packet only when the
    replacement serves the same endpoint and fits the near-matched traffic
    envelope. This prevents the scheduler from winning by abstention or by
    starving endpoints that the fallback would cover.
    """
    selected: List[Candidate] = []
    projected_bits = float(current_bits)
    traffic_ceiling = config.prism_traffic_guardrail_ratio * max(reference_bits, 0.0)
    for fallback in fallback_selected:
        fallback_bits, _ = packet_size_bits(fallback.mode, d, config.base_packet_bits)
        options = list(prism_candidates.get(fallback.endpoint, []))
        options.append(fallback)
        eligible: List[Candidate] = []
        for option in options:
            bits, _ = packet_size_bits(option.mode, d, config.base_packet_bits)
            same_or_small = bits <= fallback_bits * config.prism_upgrade_budget_ratio + 1e-9
            within_total = projected_bits + bits <= traffic_ceiling + 1e-9
            if same_or_small and within_total:
                eligible.append(option)
        if not eligible:
            continue
        best = max(eligible, key=lambda item: item.utility)
        selected.append(best)
        bits, _ = packet_size_bits(best.mode, d, config.base_packet_bits)
        projected_bits += bits
    return selected


def attribute_attack_auc(recon_history: np.ndarray, labels: np.ndarray, seed: int) -> float:
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2 or np.min(np.bincount(labels)) < 3:
        return float("nan")
    # Endpoint-level aggregates avoid leaking the same endpoint across folds.
    features = np.concatenate(
        [
            recon_history.mean(axis=0),
            recon_history.std(axis=0),
            np.max(np.abs(recon_history), axis=0),
        ],
        axis=1,
    )
    splits = min(3, int(np.min(np.bincount(labels))))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    scores = cross_val_predict(model, features, labels, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(labels, scores))
    # A capable attacker can reverse a consistently inverted score. Reporting
    # max(AUC, 1-AUC) prevents values below 0.5 from being misread as privacy.
    return max(auc, 1.0 - auc)


def simulate_scheduler(
    scheduler: str,
    data: PreparedData,
    outputs: ModelOutputs,
    trace: NetworkTrace,
    config: ExperimentConfig,
    epsilon_budget: float,
    seed: int,
    flags: Optional[AblationFlags] = None,
    run_label: Optional[str] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    flags = flags or AblationFlags()
    calibration = calibrate_simulation(outputs, data, config)
    critical_zone = criticality_matrix(outputs.pred_eval, data, calibration, config)
    t_steps, n, d = outputs.phi_eval.shape
    accountants = [ZCDPAccountant(epsilon_budget, config.delta) for _ in range(n)]
    fallback_reference_accountants = [
        ZCDPAccountant(epsilon_budget, config.delta) for _ in range(n)
    ]
    private_release_counts = np.zeros(n, dtype=int)
    fallback_reference_release_counts = np.zeros(n, dtype=int)
    recon = np.zeros((n, d), dtype=float)
    # The endpoint mirrors the edge reconstruction from acknowledged packets.
    # It therefore sends an error-feedback innovation relative to that shared
    # private state, avoiding unbounded accumulation of old DP noise.
    last_certified_quality = np.zeros(n, dtype=float)
    aox = np.ones(n, dtype=float)
    delivered_counts = np.zeros(n, dtype=int)
    transmitted_counts = np.zeros(n, dtype=int)
    residual_buffer = np.zeros((n, d), dtype=float)
    previous_oracle_phi = outputs.phi_eval[0].copy()
    recon_history = np.zeros((t_steps, n, d), dtype=float)
    prism_reference_bits = 0.0

    total_bits = 0
    total_energy_mj = 0.0
    scheduled_count = delivered_count = failure_count = 0
    latency_values: List[float] = []
    step_cosine: List[float] = []
    step_spearman: List[float] = []
    step_topk: List[float] = []
    step_sign: List[float] = []
    step_distortion: List[float] = []
    critical_event_count = 0
    covered_critical_event_count = 0
    correct_critical_event_count = 0
    accepted_error_sum = 0.0
    accepted_error_count = 0
    noncritical_event_count = 0
    false_intervention_count = 0
    false_activation_count = 0
    noncritical_driver_mismatch_count = 0
    trace_rows: List[Dict[str, Any]] = []

    for t in range(t_steps):
        critical_endpoint = critical_zone[t, data.zones].astype(float)
        previous_prediction = outputs.pred_cal[-1] if t == 0 else outputs.pred_eval[t - 1]
        forecast_novelty = np.clip(
            np.abs(outputs.pred_eval[t] - previous_prediction)
            / calibration["endpoint_ramp_scale"],
            0.0,
            5.0,
        )
        hproxy = hierarchy_proxy(
            recon,
            data.zones,
            calibration["residual_mu"],
            calibration["residual_scale"],
            flags.use_hierarchy,
        )
        if flags.residual_accumulation:
            residual_buffer *= config.residual_decay
            if t == 0:
                residual_buffer += outputs.phi_eval[t] - recon
            else:
                residual_buffer += outputs.phi_eval[t] - previous_oracle_phi
        else:
            residual_buffer = outputs.phi_eval[t] - recon
        previous_oracle_phi = outputs.phi_eval[t].copy()
        # EXPERIMENTAL PUBLIC-SCORE VARIANT.
        # Candidate selection does not inspect the raw attribution residual norm.
        # This assumes forecast_novelty, calibration clip norms, criticality, AoX,
        # and channel state are public. That assumption must be stated and proved
        # separately; this edit is not a full-transcript DP certificate.
        residual_norm = np.maximum(forecast_novelty, 0.0) * np.array(
            [PACKET_MODES[0].sigma * sensitivity_multiplier(config)
             * float(calibration["clip_norms"][j]) * math.sqrt(d)
             for j in range(n)]
        ) * 2.0
        noise_floor = np.array(
            [
                PACKET_MODES[0].sigma
                * sensitivity_multiplier(config)
                * float(calibration["clip_norms"][i])
                * math.sqrt(d)
                for i in range(n)
            ]
        )
        critical_coverage_so_far = (
            covered_critical_event_count / critical_event_count
            if critical_event_count
            else config.min_critical_coverage
        )
        fairness_so_far = jain_index(delivered_counts)
        fallback_selected_for_prism: List[Candidate] = []
        if scheduler == "prism_xai":
            fallback_candidate_map = build_candidates(
                "repaired_mapxai",
                flags,
                t,
                aox,
                critical_endpoint,
                forecast_novelty,
                hproxy,
                delivered_counts,
                fallback_reference_release_counts,
                residual_norm,
                noise_floor,
                critical_coverage_so_far,
                fairness_so_far,
                fallback_reference_accountants,
                config,
                trace,
                d,
            )
            fallback_selected = select_candidates(
                "repaired_mapxai",
                fallback_candidate_map,
                t,
                n,
                config.n_resource_blocks,
            )
            fallback_selected_for_prism = list(fallback_selected)
            fallback_step_bits = sum(
                packet_size_bits(option.mode, d, config.base_packet_bits)[0]
                for option in fallback_selected
            )
            for option in fallback_selected:
                fallback_reference_accountants[option.endpoint].spend(option.mode.sigma)
                fallback_reference_release_counts[option.endpoint] += 1
            prism_reference_bits += fallback_step_bits
        candidate_map = build_candidates(
            scheduler,
            flags,
            t,
            aox,
            critical_endpoint,
            forecast_novelty,
            hproxy,
            delivered_counts,
            private_release_counts,
            residual_norm,
            noise_floor,
            critical_coverage_so_far,
            fairness_so_far,
            accountants,
            config,
            trace,
            d,
        )
        if scheduler == "prism_xai":
            selected = select_prism_matched_fallback(
                fallback_selected_for_prism,
                candidate_map,
                d,
                config,
                total_bits,
                prism_reference_bits,
            )
        else:
            selected = select_candidates(
                scheduler, candidate_map, t, n, config.n_resource_blocks
            )

        successful_endpoints: set[int] = set()
        step_bits = 0
        step_failures = 0
        for option in selected:
            i, mode = option.endpoint, option.mode
            bits, _ = packet_size_bits(mode, d, config.base_packet_bits)
            if (
                scheduler == "prism_xai"
                and not config.allow_prism_over_budget
                and total_bits + bits
                > config.prism_traffic_guardrail_ratio * prism_reference_bits + 1e-9
            ):
                continue
            if (
                config.selective_private_release
                and scheduler in {"prism_xai", "repaired_mapxai"}
                and private_release_counts[i] >= config.max_private_releases_per_endpoint
            ):
                continue
            if not accountants[i].can_spend(mode.sigma):
                continue
            # Privacy is spent on transmission, not only successful decoding.
            accountants[i].spend(mode.sigma)
            private_release_counts[i] += 1
            transmitted_counts[i] += 1
            scheduled_count += 1
            total_bits += bits
            step_bits += bits
            total_energy_mj += option.energy_mj
            delta = residual_buffer[i] if scheduler == "prism_xai" else outputs.phi_eval[t, i] - recon[i]
            decoded_delta, _ = privatize_and_compress(
                delta,
                float(calibration["clip_norms"][i]),
                sensitivity_multiplier(config) * float(calibration["clip_norms"][i]),
                mode,
                trace.base_gaussian[t, i],
            )
            failed = bool(trace.packet_failure[t, i])
            if failed:
                failure_count += 1
                step_failures += 1
                latency_values.append(config.deadline_ms * 1.5)
            else:
                recon[i] += decoded_delta
                if scheduler == "prism_xai":
                    residual_buffer[i] = outputs.phi_eval[t, i] - recon[i]
                last_certified_quality[i] = mode_quality(mode)
                delivered_counts[i] += 1
                delivered_count += 1
                successful_endpoints.add(i)
                latency_values.append(option.latency_ms)

        aox += 1.0
        if successful_endpoints:
            aox[list(successful_endpoints)] = 1.0
        recon_history[t] = recon

        cosines, spearmans, overlaps, signs, distortions = [], [], [], [], []
        critical_count = covered_count = correct_covered = correct_all = 0
        false_count = noncritical_count = 0
        for z in range(config.n_zones):
            idx = np.where(data.zones == z)[0]
            oracle_z = outputs.phi_eval[t, idx].mean(axis=0)
            recon_z = recon[idx].mean(axis=0)
            cos = cosine_similarity(oracle_z, recon_z)
            cosines.append(cos)
            spearmans.append(spearman_similarity(oracle_z, recon_z))
            overlaps.append(topk_overlap(oracle_z, recon_z, config.topk_eval))
            signs.append(topk_sign_agreement(oracle_z, recon_z, config.topk_eval))
            distortions.append(float(np.linalg.norm(recon_z - oracle_z) / (np.linalg.norm(oracle_z) + 1e-8)))
            # This is an online certificate using only AoX and the last
            # transmitted mode's public theoretical quality. The oracle
            # cosine is never used to decide coverage.
            covered = bool(
                np.mean(aox[idx]) <= config.reliability_aox_threshold
                and np.mean(last_certified_quality[idx]) >= 0.10
            )
            agreement = driver_group(oracle_z) == driver_group(recon_z)
            if critical_zone[t, z]:
                critical_count += 1
                covered_count += int(covered)
                correct_covered += int(covered and agreement)
                correct_all += int(covered and agreement)
                if covered:
                    accepted_error_sum += float(
                        np.linalg.norm(recon_z - oracle_z) / (np.linalg.norm(oracle_z) + 1e-8)
                    )
                    accepted_error_count += 1
            else:
                noncritical_count += 1
                active = intervention_active(
                    recon_z, float(calibration["zone_norm_threshold"])
                )
                mismatch = driver_group(oracle_z) != driver_group(recon_z)
                false_activation_count += int(active)
                noncritical_driver_mismatch_count += int(mismatch)
                # A false intervention should correspond to an actionable
                # false decision, not merely a nonzero explanation vector.
                false_count += int(covered and active and mismatch)

        step_cosine.append(float(np.mean(cosines)))
        step_spearman.append(float(np.mean(spearmans)))
        step_topk.append(float(np.mean(overlaps)))
        step_sign.append(float(np.mean(signs)))
        step_distortion.append(float(np.mean(distortions)))
        # Pool event counts across the full evaluation window. Averaging
        # per-step ratios would give steps with one critical zone the same
        # weight as steps with several critical zones.
        critical_event_count += critical_count
        covered_critical_event_count += covered_count
        correct_critical_event_count += correct_all
        noncritical_event_count += noncritical_count
        false_intervention_count += false_count

        trace_rows.append(
            {
                "step": t,
                "timestamp": str(data.eval_timestamps[t]),
                "scheduler": run_label or scheduler,
                "mean_cosine": step_cosine[-1],
                "topk_overlap": step_topk[-1],
                "hierarchical_distortion": step_distortion[-1],
                "mean_aox": float(np.mean(aox)),
                "p95_aox": float(np.quantile(aox, 0.95)),
                "critical_zone_fraction": float(np.mean(critical_zone[t])),
                "remaining_rho_fraction": float(
                    np.mean([acc.remaining_rho / acc.max_rho for acc in accountants])
                ),
                "step_bits": step_bits,
                "step_failures": step_failures,
                "cumulative_bits": total_bits,
            }
        )

    epsilons = np.array([acc.epsilon_spent for acc in accountants])
    exhausted = np.array(
        [acc.remaining_rho < ZCDPAccountant.cost(PACKET_MODES[0].sigma) for acc in accountants]
    )
    attack_auc = attribute_attack_auc(recon_history, data.private_attribute, seed)
    collapse_mask = np.asarray(step_cosine) < 0.5
    operational_agreement_all = (
        correct_critical_event_count / critical_event_count
        if critical_event_count
        else float("nan")
    )
    operational_agreement_covered = (
        correct_critical_event_count / covered_critical_event_count
        if covered_critical_event_count
        else float("nan")
    )
    critical_coverage = (
        covered_critical_event_count / critical_event_count
        if critical_event_count
        else float("nan")
    )
    critical_event_recall = operational_agreement_all
    intervention_coverage = critical_coverage
    intervention_precision = (
        correct_critical_event_count
        / max(correct_critical_event_count + false_intervention_count, 1)
        if critical_event_count or noncritical_event_count
        else float("nan")
    )
    abstention_rate = (
        1.0 - covered_critical_event_count / critical_event_count
        if critical_event_count
        else float("nan")
    )
    conditional_error_accepted = (
        accepted_error_sum / accepted_error_count
        if accepted_error_count
        else float("nan")
    )
    false_intervention_rate = (
        false_intervention_count / noncritical_event_count
        if noncritical_event_count
        else float("nan")
    )
    false_activation_rate = (
        false_activation_count / noncritical_event_count
        if noncritical_event_count
        else float("nan")
    )
    noncritical_driver_mismatch_rate = (
        noncritical_driver_mismatch_count / noncritical_event_count
        if noncritical_event_count
        else float("nan")
    )
    result: Dict[str, Any] = {
        "scheduler": run_label or scheduler,
        "scheduler_base": scheduler,
        "epsilon_budget": epsilon_budget,
        "delta": config.delta,
        "cosine_similarity": float(np.mean(step_cosine)),
        "spearman_rho": float(np.mean(step_spearman)),
        "topk_overlap": float(np.mean(step_topk)),
        "topk_sign_agreement": float(np.mean(step_sign)),
        "hierarchical_distortion": float(np.mean(step_distortion)),
        "collapse_rate": float(np.mean(collapse_mask)),
        "max_collapse_duration": int(longest_true_run(collapse_mask)),
        "critical_events": int(critical_event_count),
        "covered_critical_events": int(covered_critical_event_count),
        "correct_critical_events": int(correct_critical_event_count),
        "noncritical_events": int(noncritical_event_count),
        "false_interventions": int(false_intervention_count),
        "false_activations": int(false_activation_count),
        "noncritical_driver_mismatches": int(noncritical_driver_mismatch_count),
        "operational_agreement_all": operational_agreement_all,
        "operational_agreement_covered": operational_agreement_covered,
        "critical_event_recall": critical_event_recall,
        "intervention_precision": intervention_precision,
        "intervention_coverage": intervention_coverage,
        "abstention_rate": abstention_rate,
        "conditional_error_accepted": conditional_error_accepted,
        "critical_coverage": critical_coverage,
        "missed_critical_rate": (
            1.0 - operational_agreement_all
            if np.isfinite(operational_agreement_all)
            else float("nan")
        ),
        "false_intervention_rate": false_intervention_rate,
        "false_activation_rate": false_activation_rate,
        "noncritical_driver_mismatch_rate": noncritical_driver_mismatch_rate,
        "transmitted_bits": int(total_bits),
        "transmitted_kbits": float(total_bits / 1000.0),
        "fallback_reference_kbits": float(prism_reference_bits / 1000.0)
        if scheduler == "prism_xai"
        else float("nan"),
        "bits_per_correct_intervention": float(
            total_bits / correct_critical_event_count
        )
        if correct_critical_event_count
        else float("nan"),
        "scheduled_packets": int(scheduled_count),
        "delivered_packets": int(delivered_count),
        "delivery_ratio": float(delivered_count / scheduled_count) if scheduled_count else 0.0,
        "deadline_miss_rate": float(failure_count / scheduled_count) if scheduled_count else 0.0,
        "mean_latency_ms": float(np.mean(latency_values)) if latency_values else float("nan"),
        "p95_latency_ms": float(np.quantile(latency_values, 0.95)) if latency_values else float("nan"),
        "mean_aox": float(np.mean([row["mean_aox"] for row in trace_rows])),
        "p95_aox": float(np.mean([row["p95_aox"] for row in trace_rows])),
        "energy_mj": float(total_energy_mj),
        "epsilon_spent_mean": float(np.mean(epsilons)),
        "epsilon_spent_max": float(np.max(epsilons)),
        "privacy_exhaustion_fraction": float(np.mean(exhausted)),
        "jain_delivery_fairness": jain_index(delivered_counts),
        "attribute_attack_auc": attack_auc,
        "random_topk_baseline": expected_random_topk(config.topk_eval, d),
        "mean_updates_per_endpoint": float(np.mean(delivered_counts)),
        "min_updates_per_endpoint": int(np.min(delivered_counts)),
        "mean_private_releases_per_endpoint": float(np.mean(private_release_counts)),
        "max_private_releases_per_endpoint": int(np.max(private_release_counts)),
        "private_release_cap": int(config.max_private_releases_per_endpoint),
        "selective_private_release": bool(config.selective_private_release),
        "private_release_cap_applies": bool(
            config.selective_private_release and scheduler in {"prism_xai", "repaired_mapxai"}
        ),
    }
    return result, pd.DataFrame(trace_rows)


def summarize_with_ci(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "model_family", "epsilon_budget", "scheduler"]
    if "xai_method" in raw.columns:
        group_cols.insert(2, "xai_method")
    if "stress_scenario" in raw.columns:
        group_cols = ["stress_scenario"] + group_cols
    excluded = set(group_cols + ["seed", "scheduler_base", "model_backend", "explainer_backend"])
    metric_cols = [
        column
        for column in raw.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(raw[column])
    ]
    rows: List[Dict[str, Any]] = []
    for keys, group in raw.groupby(group_cols, dropna=False):
        row: Dict[str, Any] = dict(zip(group_cols, keys))
        row["n_runs"] = len(group)
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            n_valid = len(values)
            row[f"{metric}_n"] = n_valid
            if n_valid == 0:
                mean = std = ci = float("nan")
            else:
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if n_valid > 1 else 0.0
                critical = float(student_t.ppf(0.975, df=n_valid - 1)) if n_valid > 1 else 0.0
                ci = critical * std / math.sqrt(n_valid) if n_valid > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots(
    raw: pd.DataFrame,
    trace: pd.DataFrame,
    ablations: pd.DataFrame,
    output_dir: Path,
) -> None:
    if sns is None:
        print("seaborn is not installed; skipping figure rendering.")
        return
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    palette = dict(
        zip(
            SCHEDULER_LABELS.values(),
            sns.color_palette("colorblind", len(SCHEDULER_LABELS)),
        )
    )
    plot_data = raw.copy()
    plot_data["method"] = plot_data["scheduler"].map(SCHEDULER_LABELS).fillna(plot_data["scheduler"])
    aggregate_cols = [
        "transmitted_kbits",
        "operational_agreement_all",
        "epsilon_spent_max",
        "cosine_similarity",
        "jain_delivery_fairness",
    ]
    grouped = (
        plot_data.groupby(
            ["dataset", "model_family", "epsilon_budget", "scheduler", "method"],
            as_index=False,
            dropna=False,
        )[aggregate_cols]
        .mean()
    )
    datasets = sorted(grouped["dataset"].astype(str).unique())
    models = sorted(grouped["model_family"].astype(str).unique())

    def faceted_axes(height: float = 3.7) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(
            len(datasets),
            len(models),
            figsize=(5.4 * len(models), height * len(datasets)),
            squeeze=False,
        )
        return fig, axes

    def padded_limits(values: pd.Series, hard_min: float, hard_max: float) -> Tuple[float, float]:
        clean = pd.to_numeric(values, errors="coerce").dropna()
        if clean.empty:
            return hard_min, hard_max
        lo, hi = float(clean.min()), float(clean.max())
        span = max(hi - lo, 0.05 * (hard_max - hard_min))
        pad = 0.12 * span
        return max(hard_min, lo - pad), min(hard_max, hi + pad)

    def draw_method_paths(
        axes: np.ndarray,
        x_metric: str,
        y_metric: str,
        x_label: str,
        y_label: str,
        annotate_epsilon: bool = False,
        x_bounds: Tuple[float, float] = (0.0, float("inf")),
        y_bounds: Tuple[float, float] = (-float("inf"), float("inf")),
    ) -> None:
        for row, dataset in enumerate(datasets):
            for col, model_family in enumerate(models):
                ax = axes[row, col]
                panel = grouped[
                    (grouped["dataset"].astype(str) == dataset)
                    & (grouped["model_family"].astype(str) == model_family)
                ]
                for method, method_frame in panel.groupby("method", sort=True):
                    method_frame = method_frame.sort_values("epsilon_budget")
                    ax.plot(
                        method_frame[x_metric],
                        method_frame[y_metric],
                        marker="o",
                        linewidth=1.2,
                        markersize=4.5,
                        color=palette.get(method, "#333333"),
                        label=method,
                    )
                    if annotate_epsilon:
                        for _, point in method_frame.iterrows():
                            ax.annotate(
                                f"{point['epsilon_budget']:g}",
                                (point[x_metric], point[y_metric]),
                                xytext=(3, 2),
                                textcoords="offset points",
                                fontsize=5.5,
                                color=palette.get(method, "#333333"),
                            )
                ax.set_title(f"{dataset.upper()} — {model_family}")
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                if not panel.empty:
                    ax.legend(fontsize=6, ncol=2, frameon=True)
                    finite_x_max = (
                        x_bounds[1]
                        if np.isfinite(x_bounds[1])
                        else max(float(panel[x_metric].max()) * 1.08, 1.0)
                    )
                    finite_x_min = x_bounds[0] if np.isfinite(x_bounds[0]) else float(panel[x_metric].min())
                    finite_y_max = (
                        y_bounds[1]
                        if np.isfinite(y_bounds[1])
                        else max(float(panel[y_metric].max()) * 1.08, 1.0)
                    )
                    finite_y_min = y_bounds[0] if np.isfinite(y_bounds[0]) else float(panel[y_metric].min())
                    ax.set_xlim(*padded_limits(panel[x_metric], finite_x_min, finite_x_max))
                    ax.set_ylim(*padded_limits(panel[y_metric], finite_y_min, finite_y_max))

    fig, axes = faceted_axes()
    draw_method_paths(
        axes,
        "transmitted_kbits",
        "operational_agreement_all",
        "Transmitted explanation traffic (kbit)",
        "Operational agreement",
        annotate_epsilon=True,
        x_bounds=(0.0, float("inf")),
        y_bounds=(0.0, 1.0),
    )
    fig.suptitle("Communication–operation trade-off (labels show privacy budget)", y=1.01)
    fig.tight_layout()
    save_figure(fig, output_dir, "pareto_bits_fidelity")

    fig, axes = faceted_axes()
    draw_method_paths(
        axes,
        "epsilon_spent_max",
        "cosine_similarity",
        r"Maximum composed $\epsilon$ spent",
        "Zonal explanation cosine similarity",
        x_bounds=(0.0, float("inf")),
        y_bounds=(-1.0, 1.0),
    )
    fig.suptitle("Privacy–explanation trade-off", y=1.01)
    fig.tight_layout()
    save_figure(fig, output_dir, "privacy_fidelity")

    fig, axes = faceted_axes()
    draw_method_paths(
        axes,
        "jain_delivery_fairness",
        "cosine_similarity",
        "Jain delivery fairness",
        "Explanation cosine similarity",
        annotate_epsilon=True,
        x_bounds=(0.0, 1.0),
        y_bounds=(-1.0, 1.0),
    )
    fig.suptitle("Explanation fidelity versus endpoint fairness", y=1.01)
    fig.tight_layout()
    save_figure(fig, output_dir, "fairness_reliability")

    if not trace.empty:
        trace_plot = trace.copy()
        trace_plot["method"] = trace_plot["scheduler"].map(SCHEDULER_LABELS).fillna(trace_plot["scheduler"])
        trace_datasets = sorted(trace_plot["dataset"].astype(str).unique())
        fig, axes = plt.subplots(
            4,
            len(trace_datasets),
            figsize=(6.1 * len(trace_datasets), 9.0),
            sharex="col",
            squeeze=False,
        )
        for col, dataset in enumerate(trace_datasets):
            panel = trace_plot[trace_plot["dataset"].astype(str) == dataset]
            for method, group in panel.groupby("method", sort=True):
                group = group.sort_values("step")
                color = palette.get(method, "#333333")
                axes[0, col].plot(group["step"], group["mean_cosine"], label=method, color=color)
                axes[1, col].plot(group["step"], group["mean_aox"], label=method, color=color)
                axes[2, col].plot(group["step"], group["remaining_rho_fraction"], label=method, color=color)
            critical = panel.groupby("step")["critical_zone_fraction"].max()
            axes[3, col].fill_between(
                critical.index, 0, critical.values, alpha=0.35, color="#CC6677"
            )
            axes[0, col].set_title(dataset.upper())
            axes[0, col].set_ylim(-1.05, 1.05)
            axes[2, col].set_ylim(-0.05, 1.05)
            axes[3, col].set_ylim(0, 1.05)
            axes[3, col].set_xlabel("Evaluation step")
            axes[0, col].legend(fontsize=7, ncol=1, loc="best")
        for row, label in enumerate(["Cosine", "Mean AoX", "Privacy left", "Critical zones"]):
            axes[row, 0].set_ylabel(label)
        fig.suptitle("Regime-change reliability and privacy trace", y=0.995)
        fig.tight_layout()
        save_figure(fig, output_dir, "regime_trace")

    if not ablations.empty:
        ablation_plot = ablations.copy()
        ablation_plot["method"] = ablation_plot["scheduler"].map(SCHEDULER_LABELS).fillna(ablation_plot["scheduler"])
        ablation_datasets = sorted(ablation_plot["dataset"].astype(str).unique())
        metrics = [
            ("topk_overlap", "Top-K overlap", (0.0, 1.0)),
            ("operational_agreement_all", "Operational agreement", (0.0, 1.0)),
            ("false_intervention_rate", "False intervention", (0.0, 1.0)),
            ("jain_delivery_fairness", "Delivery fairness", (0.0, 1.0)),
        ]
        method_order = [
            SCHEDULER_LABELS[key]
            for key in [
                "prism_xai",
                "prism_no_aox",
                "prism_no_privacy_price",
                "prism_fixed_resolution",
                "prism_no_hierarchy",
                "prism_no_fairness",
                "prism_no_residual_accumulation",
            ]
            if SCHEDULER_LABELS[key] in set(ablation_plot["method"])
        ]
        fig, axes = plt.subplots(
            len(ablation_datasets),
            len(metrics),
            figsize=(13.5, 3.2 * len(ablation_datasets)),
            squeeze=False,
        )
        for row, dataset in enumerate(ablation_datasets):
            panel = ablation_plot[ablation_plot["dataset"].astype(str) == dataset]
            for col, (metric, title, limits) in enumerate(metrics):
                ax = axes[row, col]
                sns.barplot(
                    data=panel,
                    y="method",
                    x=metric,
                    order=method_order,
                    errorbar=("ci", 95),
                    seed=0,
                    color="#4477AA",
                    ax=ax,
                )
                ax.set_xlim(*padded_limits(panel[metric], limits[0], limits[1]))
                ax.set_title(f"{dataset.upper()} — {title}")
                ax.set_xlabel("Score")
                ax.set_ylabel("")
                if col > 0:
                    ax.set_yticklabels([])
        fig.suptitle("PRISM-XAI ablations (full PRISM-XAI included as control)", y=1.01)
        fig.tight_layout()
        save_figure(fig, output_dir, "ablation_results")


def write_latex_table(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    tree = summary[summary["model_family"] == "tree"].copy()
    if tree.empty:
        raise ValueError("Cannot build the main table: no tree-model results are available")
    target_epsilon = float(
        tree.loc[(tree["epsilon_budget"] - 4.0).abs().idxmin(), "epsilon_budget"]
    )
    table = tree[tree["epsilon_budget"] == target_epsilon].copy()
    metrics = [
        ("topk_overlap", "Top-$K$"),
        ("cosine_similarity", "Cosine"),
        ("operational_agreement_all", "Decision"),
        ("critical_event_recall", "Recall"),
        ("intervention_precision", "Prec."),
        ("abstention_rate", "Abst."),
        ("p95_latency_ms", "P95 ms"),
        ("transmitted_kbits", "kbit"),
        ("epsilon_spent_max", "Spent $\\epsilon$"),
        ("false_intervention_rate", "False int."),
        ("bits_per_correct_intervention", "bit/correct"),
        ("jain_delivery_fairness", "Fairness"),
    ]
    rows: List[Dict[str, str]] = []
    for _, item in table.sort_values(["dataset", "scheduler"]).iterrows():
        row: Dict[str, str] = {
            "Dataset": str(item["dataset"]).upper(),
            "Model": "Tree",
            "$\\epsilon_{\\max}$": f"{target_epsilon:g}",
            "Method": SCHEDULER_LABELS.get(item["scheduler"], item["scheduler"]),
        }
        for metric, label in metrics:
            mean = item.get(f"{metric}_mean", float("nan"))
            ci = item.get(f"{metric}_ci95", float("nan"))
            row[label] = f"{mean:.3f} $\\pm$ {ci:.3f}" if np.isfinite(mean) else "--"
        rows.append(row)
    frame = pd.DataFrame(rows)
    columns = list(frame.columns)
    align = "l" + "c" * (len(columns) - 1)
    latex_lines = [
        "% Values are means with 95% Student-t confidence intervals across seeds.",
        "\\begin{tabular}{" + align + "}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        latex_lines.append(" & ".join(str(row[column]) for column in columns) + " \\\\")
    latex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    latex = "\n".join(latex_lines)
    (output_dir / "main_results_table.tex").write_text(latex, encoding="utf-8")


def write_appendix_condition_table(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    keep = summary[
        summary["scheduler"].isin(["prism_xai", "joint_no_compression", "periodic_fixed_dp"])
    ].copy()
    keep["Method"] = keep["scheduler"].map(SCHEDULER_LABELS)
    rows: List[Dict[str, str]] = []
    metrics = [
        ("transmitted_kbits", "kbit"),
        ("topk_overlap", "Top-$K$"),
        ("random_topk_baseline", "Random Top-$K$"),
        ("cosine_similarity", "Cosine"),
        ("operational_agreement_all", "Decision"),
        ("critical_event_recall", "Recall"),
        ("intervention_precision", "Precision"),
        ("intervention_coverage", "Coverage"),
        ("abstention_rate", "Abstain"),
        ("false_intervention_rate", "False int."),
        ("epsilon_spent_max", "Spent $\\epsilon$"),
    ]
    for _, item in keep.sort_values(["dataset", "model_family", "epsilon_budget", "scheduler"]).iterrows():
        row = {
            "Dataset": str(item["dataset"]).upper(),
            "Model": str(item["model_family"]).title(),
            "$\\epsilon_{\\max}$": f"{float(item['epsilon_budget']):g}",
            "Method": str(item["Method"]),
        }
        for metric, label in metrics:
            mean = item.get(f"{metric}_mean", float("nan"))
            ci = item.get(f"{metric}_ci95", float("nan"))
            row[label] = f"{mean:.3f} $\\pm$ {ci:.3f}" if np.isfinite(mean) else "--"
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "appendix_condition_means.csv", index=False)
    columns = list(frame.columns)
    align = "llll" + "c" * (len(columns) - 4)
    latex_lines = [
        "% Appendix table: all dataset/model/privacy condition means.",
        "\\begin{table*}[t]",
        "\\caption{Appendix summary across all dataset/model/privacy conditions.}",
        "\\label{tab:appendix-conditions}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{" + align + "}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        latex_lines.append(" & ".join(str(row[column]) for column in columns) + " \\\\")
    latex_lines.extend(
        ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    )
    (output_dir / "appendix_condition_table.tex").write_text(
        "\n".join(latex_lines), encoding="utf-8"
    )


def write_paired_comparison_tables(raw: pd.DataFrame, output_dir: Path) -> None:
    if raw.empty:
        return
    key_cols = ["dataset", "model_family", "epsilon_budget", "seed"]
    group_levels = [0, 1, 2]
    if "xai_method" in raw.columns:
        key_cols.insert(2, "xai_method")
        group_levels = [0, 1, 2, 3]
    baselines = [item for item in SCHEDULERS if item != "prism_xai"]
    metrics = [
        ("transmitted_kbits", "Delta kbit"),
        ("topk_overlap", "Delta Top-$K$"),
        ("cosine_similarity", "Delta cosine"),
        ("critical_event_recall", "Delta recall"),
        ("intervention_precision", "Delta precision"),
        ("intervention_coverage", "Delta coverage"),
        ("deadline_miss_rate", "Delta miss"),
        ("jain_delivery_fairness", "Delta fairness"),
        ("energy_mj", "Delta energy"),
        ("bits_per_correct_intervention", "Delta bit/correct"),
    ]
    prism = raw[raw["scheduler"] == "prism_xai"].set_index(key_cols)
    rows: List[Dict[str, Any]] = []
    for baseline in baselines:
        other = raw[raw["scheduler"] == baseline].set_index(key_cols)
        joined = prism.join(other, lsuffix="_prism", rsuffix="_baseline", how="inner")
        if joined.empty:
            continue
        for keys, group in joined.groupby(level=group_levels):
            if "xai_method" in key_cols:
                dataset, model_family, xai_method, epsilon_budget = keys
            else:
                dataset, model_family, epsilon_budget = keys
                xai_method = "default"
            row: Dict[str, Any] = {
                "dataset": dataset,
                "model_family": model_family,
                "xai_method": xai_method,
                "epsilon_budget": epsilon_budget,
                "baseline": baseline,
                "baseline_label": SCHEDULER_LABELS.get(baseline, baseline),
                "n_pairs": int(len(group)),
            }
            for metric, _ in metrics:
                left = pd.to_numeric(group[f"{metric}_prism"], errors="coerce")
                right = pd.to_numeric(group[f"{metric}_baseline"], errors="coerce")
                diff = left - right
                mean, ci, n = student_t_ci(diff)
                row[f"{metric}_diff_mean"] = mean
                row[f"{metric}_diff_ci95"] = ci
                row[f"{metric}_diff_n"] = n
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "paired_prism_minus_baselines.csv", index=False)
    if frame.empty:
        return

    target = frame[np.isclose(frame["epsilon_budget"].astype(float), 4.0)].copy()
    if target.empty:
        target = frame.copy()
    target = target[target["model_family"] == "tree"].copy()
    columns = ["Dataset", "Baseline", "Delta kbit", "Delta Top-$K$", "Delta recall", "Delta fairness"]
    latex_rows: List[Dict[str, str]] = []
    for _, item in target.sort_values(["dataset", "baseline"]).iterrows():
        out = {
            "Dataset": str(item["dataset"]).upper(),
            "Baseline": str(item["baseline_label"]),
        }
        for metric, label in metrics:
            if label not in columns:
                continue
            mean = item[f"{metric}_diff_mean"]
            ci = item[f"{metric}_diff_ci95"]
            out[label] = f"{mean:.3f} $\\pm$ {ci:.3f}" if np.isfinite(mean) else "--"
        latex_rows.append(out)
    latex_lines = [
        "% Paired PRISM-XAI minus baseline differences; negative kbit is better.",
        "\\begin{table*}[t]",
        "\\caption{Paired differences for PRISM-XAI minus each baseline at $\\epsilon_{\\max}=4$ with the tree model.}",
        "\\label{tab:paired-differences}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for row in latex_rows:
        latex_lines.append(" & ".join(row[column] for column in columns) + " \\\\")
    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (output_dir / "paired_prism_minus_baselines_table.tex").write_text(
        "\n".join(latex_lines), encoding="utf-8"
    )


def write_stress_tables(raw: pd.DataFrame, output_dir: Path) -> None:
    if raw.empty or "stress_scenario" not in raw.columns:
        return
    metrics = [
        "transmitted_kbits",
        "topk_overlap",
        "operational_agreement_all",
        "false_intervention_rate",
        "false_activation_rate",
        "delivery_ratio",
        "deadline_miss_rate",
        "jain_delivery_fairness",
    ]
    summary = summarize_with_ci(raw)
    summary.to_csv(output_dir / "stress_summary_ci.csv", index=False)
    focus = summary[
        summary["scheduler"].isin(["prism_xai", "joint_no_compression", "periodic_fixed_dp"])
    ].copy()
    focus["Method"] = focus["scheduler"].map(SCHEDULER_LABELS)
    rows: List[Dict[str, str]] = []
    for _, item in focus.sort_values(["stress_scenario", "dataset", "scheduler"]).iterrows():
        row = {
            "Stress": str(item["stress_scenario"]).replace("_", " ").title(),
            "Dataset": str(item["dataset"]).upper(),
            "Method": str(item["Method"]),
        }
        for metric in metrics:
            mean = item.get(f"{metric}_mean", float("nan"))
            ci = item.get(f"{metric}_ci95", float("nan"))
            row[metric.replace("_", " ")] = (
                f"{mean:.3f} $\\pm$ {ci:.3f}" if np.isfinite(mean) else "--"
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "stress_focus_table.csv", index=False)
    columns = list(frame.columns)
    align = "lll" + "c" * (len(columns) - 3)
    latex_lines = [
        "% Stress-test table for appendix/supplement.",
        "\\begin{table*}[t]",
        "\\caption{Wireless stress tests under burst-loss and weak-channel conditions.}",
        "\\label{tab:stress-tests}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{" + align + "}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        latex_lines.append(" & ".join(str(row[column]) for column in columns) + " \\\\")
    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (output_dir / "stress_test_table.tex").write_text(
        "\n".join(latex_lines), encoding="utf-8"
    )


def make_stress_plot(raw: pd.DataFrame, output_dir: Path) -> None:
    if sns is None:
        print("seaborn is not installed; skipping stress figure rendering.")
        return
    if raw.empty or "stress_scenario" not in raw.columns:
        return
    plot_data = raw.copy()
    plot_data["method"] = plot_data["scheduler"].map(SCHEDULER_LABELS).fillna(plot_data["scheduler"])
    focus = plot_data[
        plot_data["scheduler"].isin(["prism_xai", "joint_no_compression", "periodic_fixed_dp"])
    ]
    if focus.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), squeeze=False)
    panels = [
        ("transmitted_kbits", "Traffic (kbit)"),
        ("operational_agreement_all", "Decision agreement"),
        ("false_intervention_rate", "Strict false intervention"),
    ]
    for ax, (metric, title) in zip(axes.flat, panels):
        sns.barplot(
            data=focus,
            x="stress_scenario",
            y=metric,
            hue="method",
            errorbar=("ci", 95),
            seed=0,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Stress scenario")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
        if metric != "transmitted_kbits":
            ax.set_ylim(0.0, min(1.0, max(0.1, float(focus[metric].max()) * 1.25)))
    axes[0, 0].legend(fontsize=7, frameon=True)
    for ax in axes.flat[1:]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.suptitle("Wireless robustness stress tests", y=1.03)
    fig.tight_layout()
    save_figure(fig, output_dir, "stress_wireless_robustness")


def write_wireless_assumptions(output_dir: Path) -> None:
    text = """# Wireless Assumptions and 3GPP Trace Abstraction

The simulator uses a trace-driven uplink abstraction inspired by 3GPP TR 38.901 urban microcell modeling. Each run samples endpoint distances, correlated log-normal shadowing, exponential small-scale fading, 3.5 GHz carrier frequency, 180 kHz resource-block bandwidth, transmit power, noise figure, edge-processing delay, and bursty packet failures. The resulting spectral-efficiency traces drive packet feasibility, latency, energy, deadline misses, and delivery fairness.

This abstraction is not a full ns-3 or link-level 3GPP implementation. It is used to test scheduler behavior under repeatable time-varying wireless capacity. The v1.4 stress suite adds two controlled robustness cases: burst_loss increases bad-channel persistence and failure probability, while weak_channel reduces spectral efficiency for the farthest quartile of endpoints and adds extra packet failures. These tests should be reported as wireless stress tests rather than as physical network deployment validation.
"""
    (output_dir / "WIRELESS_ASSUMPTIONS.md").write_text(text, encoding="utf-8")


def validate_results(
    raw: pd.DataFrame,
    config: ExperimentConfig,
    trace: Optional[pd.DataFrame] = None,
    summary: Optional[pd.DataFrame] = None,
    ablations: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    key_cols = ["dataset", "model_family", "epsilon_budget", "scheduler", "seed"]
    if "xai_method" in raw.columns:
        key_cols.insert(2, "xai_method")
    xai_count_by_model = {
        model: len(compatible_xai_methods(model, config.xai_methods))
        for model in config.models
    }
    expected_model_xai_count = sum(xai_count_by_model.values())
    expected_rows = (
        expected_model_xai_count
        * len(config.epsilon_budgets)
        * len(SCHEDULERS)
        * len(config.seeds)
    )
    checks["expected_run_count"] = expected_rows
    checks["actual_run_count"] = int(len(raw))
    checks["run_matrix_complete"] = len(raw) == expected_rows
    checks["unique_run_keys"] = bool(
        set(key_cols).issubset(raw.columns) and not raw.duplicated(key_cols).any()
    )
    checks["all_schedulers_present"] = set(SCHEDULERS) == set(raw["scheduler"])
    checks["privacy_never_exceeded"] = bool(
        np.all(raw["epsilon_spent_max"] <= raw["epsilon_budget"] + 1e-6)
    )

    bounded_zero_one = [
        "topk_overlap",
        "topk_sign_agreement",
        "collapse_rate",
        "operational_agreement_all",
        "operational_agreement_covered",
        "critical_event_recall",
        "intervention_precision",
        "intervention_coverage",
        "abstention_rate",
        "critical_coverage",
        "missed_critical_rate",
        "false_intervention_rate",
        "false_activation_rate",
        "noncritical_driver_mismatch_rate",
        "delivery_ratio",
        "deadline_miss_rate",
        "privacy_exhaustion_fraction",
        "jain_delivery_fairness",
        "attribute_attack_auc",
    ]
    bounded_correlations = ["cosine_similarity", "spearman_rho"]
    checks["bounded_metrics_valid"] = bool(
        all(raw[column].dropna().between(0.0, 1.0).all() for column in bounded_zero_one)
        and all(raw[column].dropna().between(-1.0, 1.0).all() for column in bounded_correlations)
    )
    checks["attribute_attack_auc_orientation_valid"] = bool(
        raw["attribute_attack_auc"].dropna().between(0.5, 1.0).all()
    )
    checks["nonnegative_costs_valid"] = bool(
        (raw[["transmitted_bits", "transmitted_kbits", "energy_mj"]] >= 0).all().all()
        and (raw[["mean_latency_ms", "p95_latency_ms"]].dropna() >= 0).all().all()
    )

    required_finite = [
        "cosine_similarity",
        "spearman_rho",
        "topk_overlap",
        "topk_sign_agreement",
        "hierarchical_distortion",
        "transmitted_bits",
        "transmitted_kbits",
        "delivery_ratio",
        "deadline_miss_rate",
        "epsilon_spent_mean",
        "epsilon_spent_max",
        "jain_delivery_fairness",
        "forecast_rmse",
        "forecast_mae",
    ]
    checks["core_metrics_finite"] = bool(
        np.isfinite(raw[required_finite].to_numpy(dtype=float)).all()
    )
    checks["conditional_missingness_valid"] = bool(
        (
            raw["operational_agreement_covered"].notna()
            == (raw["covered_critical_events"] > 0)
        ).all()
        and (
            raw["false_intervention_rate"].notna()
            == (raw["noncritical_events"] > 0)
        ).all()
        and (
            raw["operational_agreement_all"].notna()
            == (raw["critical_events"] > 0)
        ).all()
    )

    paired_group_cols = ["model_family", "epsilon_budget", "scheduler"]
    if "xai_method" in raw.columns:
        paired_group_cols.insert(1, "xai_method")
    paired = raw.groupby(paired_group_cols)["seed"].nunique()
    checks["paired_seed_counts_valid"] = bool(
        len(paired) == expected_model_xai_count * len(config.epsilon_budgets) * len(SCHEDULERS)
        and (paired == len(set(config.seeds))).all()
    )
    checks["paired_seed_counts"] = [
        {
            "model_family": key[0],
            "xai_method": key[1] if len(key) == 4 else "default",
            "epsilon_budget": float(key[2] if len(key) == 4 else key[1]),
            "scheduler": key[3] if len(key) == 4 else key[2],
            "n_seeds": int(value),
        }
        for key, value in paired.items()
    ]

    if summary is not None and not summary.empty:
        metric_n_columns = [column for column in summary if column.endswith("_n")]
        checks["summary_effective_sample_sizes_present"] = bool(metric_n_columns)
        checks["summary_run_counts_valid"] = bool((summary["n_runs"] == len(set(config.seeds))).all())
    else:
        checks["summary_effective_sample_sizes_present"] = False
        checks["summary_run_counts_valid"] = False

    if trace is not None and not trace.empty:
        trace_key = [
            "dataset",
            "model_family",
            "xai_method",
            "epsilon_budget",
            "seed",
            "scheduler",
            "step",
        ]
        checks["trace_provenance_complete"] = set(trace_key).issubset(trace.columns)
        checks["trace_keys_unique"] = bool(
            checks["trace_provenance_complete"] and not trace.duplicated(trace_key).any()
        )
    else:
        trace_expected = "tree" in config.models
        checks["trace_provenance_complete"] = not trace_expected
        checks["trace_keys_unique"] = not trace_expected

    if config.run_ablations and "tree" in config.models:
        checks["ablation_control_present"] = bool(
            ablations is not None
            and not ablations.empty
            and "prism_xai" in set(ablations["scheduler"])
        )
    else:
        checks["ablation_control_present"] = True

    boolean_checks = [value for value in checks.values() if isinstance(value, (bool, np.bool_))]
    checks["passed"] = bool(all(boolean_checks))
    return checks


def package_versions() -> Dict[str, str]:
    versions = {
        "prism_xai_script": SCRIPT_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import sklearn

        versions["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    for name in ["lightgbm", "shap", "torch", "matplotlib", "seaborn"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not installed"
    return versions


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    output_dir: Path,
    validation: Mapping[str, Any],
    elapsed_s: float,
    filename: str = "run_manifest.json",
) -> None:
    files = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != filename:
            files[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "elapsed_seconds": elapsed_s,
        "validation": validation,
        "files": files,
    }
    (output_dir / filename).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_pipeline(config: ExperimentConfig) -> None:
    validate_config(config)
    started = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir = str(Path(config.cache_dir).resolve())
    (output_dir / "run_config.json").write_text(
        json.dumps({"config": asdict(config), "versions": package_versions()}, indent=2),
        encoding="utf-8",
    )

    records: List[Dict[str, Any]] = []
    ablation_records: List[Dict[str, Any]] = []
    representative_traces: List[pd.DataFrame] = []
    representative_epsilon = min(config.epsilon_budgets, key=lambda value: abs(value - 4.0))
    ablation_definitions = {
        "prism_no_aox": AblationFlags(use_aox=False),
        "prism_no_privacy_price": AblationFlags(use_privacy_price=False),
        "prism_fixed_resolution": AblationFlags(adaptive_resolution=False),
        "prism_no_hierarchy": AblationFlags(use_hierarchy=False),
        "prism_no_fairness": AblationFlags(use_fairness=False),
        "prism_no_residual_accumulation": AblationFlags(residual_accumulation=False),
    }

    for seed in config.seeds:
        set_global_seed(seed)
        print(f"\n=== Preparing {config.dataset} data, seed={seed} ===")
        data = prepare_data(config, seed)
        provenance_path = output_dir / "data_provenance.json"
        if not provenance_path.exists():
            provenance_path.write_text(
                json.dumps(
                    {
                        "dataset": config.dataset,
                        "source_name": data.source_name,
                        "source_url": data.source_url,
                        "source_file_sha256": data.source_sha256,
                        "n_endpoints_used": data.n_endpoints,
                        "endpoint_ids": data.endpoint_ids,
                        "n_zones": config.n_zones,
                        "n_calibration_steps": config.calibration_steps,
                        "n_evaluation_steps": config.n_steps,
                        "source_window_start": data.source_window_start,
                        "source_window_end": data.source_window_end,
                        "evaluation_window_start": str(data.eval_timestamps.min()),
                        "evaluation_window_end": str(data.eval_timestamps.max()),
                        "synthetic_hierarchy": config.dataset in {"uci", "opsd", "london"},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        network_trace = make_network_trace(config, data, seed)
        for model_family in config.models:
            for xai_method in compatible_xai_methods(model_family, config.xai_methods):
                print(
                    f"Training/explaining model={model_family}, xai={xai_method}, seed={seed}"
                )
                outputs = train_and_explain(data, config, model_family, seed, xai_method)
                print(
                    f"Forecast RMSE={outputs.forecast_rmse:.4f}, MAE={outputs.forecast_mae:.4f}, "
                    f"backend={outputs.model_backend}/{outputs.explainer_backend}"
                )
                for epsilon_budget in config.epsilon_budgets:
                    for scheduler in SCHEDULERS:
                        print(
                            f"Simulating scheduler={scheduler}, model={model_family}, "
                            f"xai={outputs.explainer_backend}, epsilon={epsilon_budget}, seed={seed}"
                        )
                        result, trace_frame = simulate_scheduler(
                            scheduler,
                            data,
                            outputs,
                            network_trace,
                            config,
                            epsilon_budget,
                            seed,
                        )
                        result.update(
                            {
                                "dataset": config.dataset,
                                "model_family": model_family,
                                "xai_method": outputs.explainer_backend,
                                "model_backend": outputs.model_backend,
                                "explainer_backend": outputs.explainer_backend,
                                "seed": seed,
                                "forecast_rmse": outputs.forecast_rmse,
                                "forecast_mae": outputs.forecast_mae,
                            }
                        )
                        trace_frame = trace_frame.assign(
                            dataset=config.dataset,
                            model_family=model_family,
                            xai_method=outputs.explainer_backend,
                            model_backend=outputs.model_backend,
                            explainer_backend=outputs.explainer_backend,
                            epsilon_budget=float(epsilon_budget),
                            seed=int(seed),
                        )
                        records.append(result)
                        if (
                            seed == config.seeds[0]
                            and model_family == "tree"
                            and epsilon_budget == representative_epsilon
                            and scheduler in {"aox_fixed_dp", "repaired_mapxai", "prism_xai"}
                        ):
                            representative_traces.append(trace_frame)
                        if config.save_all_traces:
                            trace_frame.to_csv(
                                output_dir
                                / (
                                    f"trace_{config.dataset}_{model_family}_"
                                    f"{outputs.explainer_backend}_eps{epsilon_budget:g}_"
                                    f"{scheduler}_seed{seed}.csv"
                                ),
                                index=False,
                            )

                if config.run_ablations and model_family == "tree":
                    for label, flags in ablation_definitions.items():
                        print(
                            f"Running ablation={label}, xai={outputs.explainer_backend}, seed={seed}"
                        )
                        result, _ = simulate_scheduler(
                            "prism_xai",
                            data,
                            outputs,
                            network_trace,
                            config,
                            representative_epsilon,
                            seed,
                            flags=flags,
                            run_label=label,
                        )
                        result.update(
                            {
                                "dataset": config.dataset,
                                "model_family": model_family,
                                "xai_method": outputs.explainer_backend,
                                "model_backend": outputs.model_backend,
                                "explainer_backend": outputs.explainer_backend,
                                "seed": seed,
                                "forecast_rmse": outputs.forecast_rmse,
                                "forecast_mae": outputs.forecast_mae,
                            }
                        )
                        ablation_records.append(result)

    raw = pd.DataFrame(records)
    summary = summarize_with_ci(raw)
    ablations = pd.DataFrame(ablation_records)
    if config.run_ablations:
        baseline = raw[
            (raw["model_family"] == "tree")
            & (raw["epsilon_budget"] == representative_epsilon)
            & (raw["scheduler"] == "prism_xai")
        ].copy()
        ablations = pd.concat([baseline, ablations], ignore_index=True, sort=False)
    trace = pd.concat(representative_traces, ignore_index=True) if representative_traces else pd.DataFrame()
    raw.to_csv(output_dir / "raw_run_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_ci.csv", index=False)
    trace.to_csv(output_dir / "representative_trace.csv", index=False)
    ablations.to_csv(output_dir / "ablation_metrics.csv", index=False)
    write_latex_table(summary, output_dir)
    write_appendix_condition_table(summary, output_dir)
    write_paired_comparison_tables(raw, output_dir)
    write_wireless_assumptions(output_dir)
    make_plots(raw, trace, ablations, output_dir)
    validation = validate_results(raw, config, trace, summary, ablations)
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2, default=str), encoding="utf-8"
    )
    write_manifest(output_dir, validation, time.time() - started)
    if not validation["passed"]:
        raise RuntimeError(f"Validation failed: {validation}")
    print(f"\nCompleted successfully. Results: {output_dir.resolve()}")


def read_optional_csv(path: Path) -> pd.DataFrame:
    """Read an optional result CSV, including intentionally empty files."""
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def run_real_data_suite(
    output_dir: Path,
    mode: str,
    overrides: Optional[Mapping[str, Any]] = None,
) -> None:
    """Run UCI and OPSD and create combined real-data outputs.

    This lives in the main file so a Colab user needs only this one script.
    Verification is a one-seed engineering check; full is the paper matrix.
    """
    started = time.time()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    dataset_dirs: List[Path] = []
    for dataset in ("uci", "opsd"):
        config = ExperimentConfig.full_defaults()
        config.dataset = dataset
        config.output_dir = str(root / dataset)
        config.cache_dir = str(root / "data_cache")
        if dataset == "opsd":
            config.n_endpoints = 11
            config.n_zones = 3
        for key, value in (overrides or {}).items():
            setattr(config, key, value)
        if mode == "verification":
            config.preset = "quick"
            config.models = ["tree"]
            config.seeds = [11]
            config.epsilon_budgets = [4.0]
            config.n_steps = 48
            config.calibration_steps = 24
            config.train_max_rows = 30000
            config.quick_tree_estimators = 100
            config.run_ablations = False
        run_pipeline(config)
        dataset_dirs.append(Path(config.output_dir))

    raw = pd.concat(
        [pd.read_csv(path / "raw_run_metrics.csv") for path in dataset_dirs],
        ignore_index=True,
    )
    traces = pd.concat(
        [pd.read_csv(path / "representative_trace.csv") for path in dataset_dirs],
        ignore_index=True,
    )
    ablation_frames = [
        frame
        for frame in (read_optional_csv(path / "ablation_metrics.csv") for path in dataset_dirs)
        if not frame.empty
    ]
    ablations = (
        pd.concat(ablation_frames, ignore_index=True) if ablation_frames else pd.DataFrame()
    )
    summary = summarize_with_ci(raw)
    raw.to_csv(root / "combined_raw_run_metrics.csv", index=False)
    summary.to_csv(root / "combined_summary_ci.csv", index=False)
    traces.to_csv(root / "combined_representative_trace.csv", index=False)
    ablations.to_csv(root / "combined_ablation_metrics.csv", index=False)
    make_plots(raw, traces, ablations, root)
    write_latex_table(summary, root)
    write_appendix_condition_table(summary, root)
    write_paired_comparison_tables(raw, root)
    write_wireless_assumptions(root)
    (root / "DATA_SOURCES.md").write_text(
        "# Real datasets\n\n"
        "- UCI ElectricityLoadDiagrams20112014: https://doi.org/10.24432/C58C86\n"
        "- OPSD Household Data 2020-04-15: "
        "https://data.open-power-system-data.org/household_data/2020-04-15/\n\n"
        "Each dataset directory records the source URL, SHA-256, and exact slice.\n",
        encoding="utf-8",
    )
    dataset_validations = [
        json.loads((path / "validation_report.json").read_text(encoding="utf-8"))
        for path in dataset_dirs
    ]
    combined_key = ["dataset", "model_family", "epsilon_budget", "scheduler", "seed"]
    if "xai_method" in raw.columns:
        combined_key.insert(2, "xai_method")
    trace_key = [
        "dataset",
        "model_family",
        "xai_method",
        "epsilon_budget",
        "seed",
        "scheduler",
        "step",
    ]
    suite_validation = {
        "mode": mode,
        "datasets": sorted(raw["dataset"].unique().tolist()),
        "all_dataset_validations_passed": all(item["passed"] for item in dataset_validations),
        "combined_run_keys_unique": bool(not raw.duplicated(combined_key).any()),
        "combined_trace_provenance_complete": bool(set(trace_key).issubset(traces.columns)),
        "combined_trace_keys_unique": bool(
            set(trace_key).issubset(traces.columns) and not traces.duplicated(trace_key).any()
        ),
        "combined_ablation_control_present": bool(
            mode != "full"
            or (
                not ablations.empty
                and set(raw["dataset"].unique()).issubset(
                    set(ablations.loc[ablations["scheduler"] == "prism_xai", "dataset"])
                )
            )
        ),
        "note": (
            "Engineering verification only; do not use as a statistical paper result."
            if mode == "verification"
            else "Full paired real-data experiment matrix."
        ),
    }
    suite_boolean_checks = [
        value for value in suite_validation.values() if isinstance(value, bool)
    ]
    suite_validation["passed"] = bool(all(suite_boolean_checks))
    (root / "combined_validation_report.json").write_text(
        json.dumps(suite_validation, indent=2), encoding="utf-8"
    )
    write_manifest(
        root,
        suite_validation,
        time.time() - started,
        filename="combined_run_manifest.json",
    )
    if not suite_validation["passed"]:
        raise RuntimeError(f"Combined validation failed: {suite_validation}")
    print(f"\nCombined real-data suite completed: {root.resolve()}")


def run_wireless_stress_suite(output_dir: Path, mode: str) -> None:
    """Run compact wireless robustness tests for the appendix/supplement."""
    started = time.time()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    scenarios = ["baseline", "burst_loss", "weak_channel"]
    seeds = [11] if mode == "verification" else [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

    for dataset in ("uci", "opsd"):
        config = ExperimentConfig.full_defaults()
        config.dataset = dataset
        config.output_dir = str(root / f"{dataset}_stress")
        config.cache_dir = str(root / "data_cache")
        config.models = ["tree"]
        config.epsilon_budgets = [4.0]
        config.seeds = seeds
        config.run_ablations = False
        if dataset == "opsd":
            config.n_endpoints = 11
            config.n_zones = 3
        if mode == "verification":
            config.preset = "quick"
            config.n_steps = 48
            config.calibration_steps = 24
            config.train_max_rows = 30000
            config.quick_tree_estimators = 100
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        validate_config(config)

        for seed in config.seeds:
            set_global_seed(seed)
            print(f"\n=== Stress data={dataset}, seed={seed} ===")
            data = prepare_data(config, seed)
            base_trace = make_network_trace(config, data, seed)
            outputs = train_and_explain(data, config, "tree", seed, "default")
            for scenario in scenarios:
                stressed_trace = apply_stress_to_trace(base_trace, seed, scenario)
                for scheduler in SCHEDULERS:
                    print(
                        f"Stress scenario={scenario}, scheduler={scheduler}, "
                        f"dataset={dataset}, seed={seed}"
                    )
                    result, _ = simulate_scheduler(
                        scheduler,
                        data,
                        outputs,
                        stressed_trace,
                        config,
                        4.0,
                        seed,
                    )
                    result.update(
                        {
                            "stress_scenario": scenario,
                            "dataset": dataset,
                            "model_family": "tree",
                            "xai_method": outputs.explainer_backend,
                            "model_backend": outputs.model_backend,
                            "explainer_backend": outputs.explainer_backend,
                            "seed": seed,
                            "forecast_rmse": outputs.forecast_rmse,
                            "forecast_mae": outputs.forecast_mae,
                        }
                    )
                    records.append(result)

    raw = pd.DataFrame(records)
    raw.to_csv(root / "stress_raw_run_metrics.csv", index=False)
    write_stress_tables(raw, root)
    make_stress_plot(raw, root)
    write_wireless_assumptions(root)
    validation = {
        "mode": mode,
        "script_version": SCRIPT_VERSION,
        "stress_scenarios": scenarios,
        "datasets": ["opsd", "uci"],
        "seeds": seeds,
        "expected_run_count": len(scenarios) * 2 * len(seeds) * len(SCHEDULERS),
        "actual_run_count": int(len(raw)),
        "privacy_never_exceeded": bool(
            np.all(raw["epsilon_spent_max"] <= raw["epsilon_budget"] + 1e-6)
        ),
        "strict_false_intervention_present": "false_intervention_rate" in raw.columns,
        "legacy_false_activation_present": "false_activation_rate" in raw.columns,
    }
    validation["passed"] = bool(
        validation["expected_run_count"] == validation["actual_run_count"]
        and validation["privacy_never_exceeded"]
        and validation["strict_false_intervention_present"]
        and validation["legacy_false_activation_present"]
    )
    (root / "stress_validation_report.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    write_manifest(root, validation, time.time() - started, filename="stress_run_manifest.json")
    if not validation["passed"]:
        raise RuntimeError(f"Stress validation failed: {validation}")
    print(f"\nWireless stress suite completed: {root.resolve()}")


def score_tuning_candidate(raw: pd.DataFrame) -> Dict[str, float]:
    """Score a candidate against the internal MAP-XAI fallback.

    The objective rewards matched-or-lower traffic only when PRISM keeps
    enough useful behavior. This prevents the degenerate all-abstain solution
    from looking good merely because it sends no packets.
    """
    means = (
        raw.groupby(
            [
                column
                for column in [
                    "dataset",
                    "model_family",
                    "xai_method",
                    "epsilon_budget",
                    "scheduler",
                ]
                if column in raw.columns
            ]
        )
        .mean(numeric_only=True)
        .reset_index()
    )
    index_cols = [
        column
        for column in ["dataset", "model_family", "xai_method", "epsilon_budget"]
        if column in means.columns
    ]
    prism = means[means["scheduler"] == "prism_xai"].set_index(index_cols)
    fallback = means[means["scheduler"] == "repaired_mapxai"].set_index(index_cols)
    common = prism.index.intersection(fallback.index)
    if len(common) == 0:
        return {"score": float("-inf")}
    p = prism.loc[common]
    b = fallback.loc[common]
    traffic_saving = (b["transmitted_kbits"] - p["transmitted_kbits"]) / b[
        "transmitted_kbits"
    ].replace(0, np.nan)
    topk_delta = p["topk_overlap"] - b["topk_overlap"]
    recall_delta = p["critical_event_recall"] - b["critical_event_recall"]
    coverage_delta = p["intervention_coverage"] - b["intervention_coverage"]
    precision_delta = p["intervention_precision"] - b["intervention_precision"]
    fairness_delta = p["jain_delivery_fairness"] - b["jain_delivery_fairness"]
    random_gap = p["topk_overlap"] - p["random_topk_baseline"]
    abstention_penalty = np.maximum(0.0, p["abstention_rate"] - b["abstention_rate"])
    zero_action_penalty = (p["correct_critical_events"] <= 0).astype(float)
    score = (
        100.0 * float(np.nanmean(traffic_saving))
        + 250.0 * float(np.nanmean(topk_delta))
        + 500.0 * float(np.nanmean(recall_delta))
        + 250.0 * float(np.nanmean(coverage_delta))
        + 100.0 * float(np.nanmean(precision_delta))
        + 40.0 * float(np.nanmean(fairness_delta))
        + 100.0 * float(np.nanmean(random_gap))
        - 250.0 * float(np.nanmean(abstention_penalty))
        - 200.0 * float(np.nanmean(zero_action_penalty))
    )
    return {
        "score": score,
        "traffic_saving_vs_fallback": float(np.nanmean(traffic_saving)),
        "topk_delta_vs_fallback": float(np.nanmean(topk_delta)),
        "recall_delta_vs_fallback": float(np.nanmean(recall_delta)),
        "coverage_delta_vs_fallback": float(np.nanmean(coverage_delta)),
        "precision_delta_vs_fallback": float(np.nanmean(precision_delta)),
        "fairness_delta_vs_fallback": float(np.nanmean(fairness_delta)),
        "topk_minus_random": float(np.nanmean(random_gap)),
        "prism_abstention": float(np.nanmean(p["abstention_rate"])),
        "fallback_abstention": float(np.nanmean(b["abstention_rate"])),
    }


def run_tuning_suite(output_dir: Path) -> None:
    """Run a compact, reproducible scheduler hyperparameter search."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    candidates: List[Dict[str, Any]] = [
        {
            "name": "matched_default",
            "residual_noise_margin": 1.85,
            "residual_upgrade_margin": 4.0,
            "residual_decay": 0.65,
            "override_gain_margin": 0.35,
            "prism_traffic_guardrail_ratio": 1.02,
            "prism_upgrade_budget_ratio": 1.02,
            "fairness_weight": 120.0,
            "critical_recall_weight": 160.0,
        },
        {
            "name": "coverage_balanced",
            "residual_noise_margin": 1.25,
            "residual_upgrade_margin": 3.0,
            "residual_decay": 0.72,
            "override_gain_margin": 0.20,
            "prism_traffic_guardrail_ratio": 1.04,
            "prism_upgrade_budget_ratio": 1.05,
            "fairness_weight": 220.0,
            "critical_recall_weight": 320.0,
        },
        {
            "name": "topk_balanced",
            "residual_noise_margin": 1.10,
            "residual_upgrade_margin": 2.75,
            "residual_decay": 0.80,
            "override_gain_margin": 0.12,
            "prism_traffic_guardrail_ratio": 1.05,
            "prism_upgrade_budget_ratio": 1.08,
            "gain_per_bit_weight": 3200.0,
            "critical_recall_weight": 260.0,
        },
        {
            "name": "fairness_balanced",
            "residual_noise_margin": 1.35,
            "residual_upgrade_margin": 3.25,
            "residual_decay": 0.75,
            "override_gain_margin": 0.18,
            "prism_traffic_guardrail_ratio": 1.04,
            "prism_upgrade_budget_ratio": 1.05,
            "fairness_weight": 360.0,
            "critical_recall_weight": 260.0,
        },
        {
            "name": "traffic_strict",
            "residual_noise_margin": 2.10,
            "residual_upgrade_margin": 4.5,
            "residual_decay": 0.55,
            "override_gain_margin": 0.45,
            "prism_traffic_guardrail_ratio": 1.00,
            "prism_upgrade_budget_ratio": 1.00,
            "privacy_price_weight": 0.25,
        },
    ]
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        name = str(candidate["name"])
        overrides = {key: value for key, value in candidate.items() if key != "name"}
        candidate_dir = root / name
        print(f"\n=== Tuning candidate: {name} ===")
        run_real_data_suite(candidate_dir, "verification", overrides=overrides)
        raw = pd.read_csv(candidate_dir / "combined_raw_run_metrics.csv")
        score_row = score_tuning_candidate(raw)
        rows.append({"candidate": name, **overrides, **score_row})
    tuning = pd.DataFrame(rows).sort_values("score", ascending=False)
    tuning.to_csv(root / "tuning_results.csv", index=False)
    best = tuning.iloc[0].to_dict()
    (root / "best_tuning_config.json").write_text(
        json.dumps(best, indent=2, default=str), encoding="utf-8"
    )
    print("\nTuning complete. Ranked candidates:")
    print(tuning.to_string(index=False))
    print(f"\nBest candidate: {best['candidate']}")


def validate_config(config: ExperimentConfig) -> None:
    if config.preset not in {"quick", "full"}:
        raise ValueError("preset must be 'quick' or 'full'")
    if config.dataset not in {"synthetic", "uci", "opsd", "london"}:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    if not config.models or len(set(config.models)) != len(config.models):
        raise ValueError("models must be a non-empty list without duplicates")
    if not set(config.models).issubset({"tree", "neural"}):
        raise ValueError("models may contain only 'tree' and 'neural'")
    valid_xai = {
        "default",
        "treeshap",
        "feature_ablation",
        "integrated_gradients",
        "gradient_x_input",
    }
    if not config.xai_methods or not set(config.xai_methods).issubset(valid_xai):
        raise ValueError(f"xai_methods must be drawn from {sorted(valid_xai)}")
    for model in config.models:
        compatible_xai_methods(model, config.xai_methods)
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be a non-empty list of unique integers")
    if not config.epsilon_budgets or len(set(config.epsilon_budgets)) != len(
        config.epsilon_budgets
    ):
        raise ValueError("epsilon_budgets must be non-empty and unique")
    if any(value <= 0 for value in config.epsilon_budgets):
        raise ValueError("epsilon budgets must be positive")
    if not 0 < config.delta < 1:
        raise ValueError("delta must lie strictly between zero and one")
    if config.n_endpoints < 4:
        raise ValueError("n_endpoints must be at least four")
    if config.n_zones < 1 or config.n_zones > config.n_endpoints:
        raise ValueError("n_zones must be between one and n_endpoints")
    if config.history < 24:
        raise ValueError("history must be at least 24 because lag_24 is used")
    if config.n_steps < 1 or config.calibration_steps < 2:
        raise ValueError("n_steps must be positive and calibration_steps must be at least two")
    if config.adjacency not in {"add_remove", "replace_one"}:
        raise ValueError("adjacency must be 'add_remove' or 'replace_one'")
    if config.train_max_rows < 1 or config.n_resource_blocks < 1:
        raise ValueError("train_max_rows and n_resource_blocks must be positive")
    if not 0 < config.clip_quantile <= 1:
        raise ValueError("clip_quantile must lie in (0, 1]")
    if config.topk_eval < 1:
        raise ValueError("topk_eval must be positive")
    if not 0.0 <= config.residual_decay <= 1.0:
        raise ValueError("residual_decay must lie in [0, 1]")
    if (
        config.residual_noise_margin < 0
        or config.residual_upgrade_margin < 0
        or config.override_gain_margin < 0
    ):
        raise ValueError(
            "residual_noise_margin, residual_upgrade_margin, and "
            "override_gain_margin must be non-negative"
        )
    if config.prism_traffic_guardrail_ratio <= 0:
        raise ValueError("prism_traffic_guardrail_ratio must be positive")
    if config.prism_upgrade_budget_ratio <= 0:
        raise ValueError("prism_upgrade_budget_ratio must be positive")
    if config.max_private_releases_per_endpoint < 1:
        raise ValueError("max_private_releases_per_endpoint must be at least one")
    if config.public_release_score_threshold < 0:
        raise ValueError("public_release_score_threshold must be non-negative")
    if config.release_cap_sigma_safety < 1.0:
        raise ValueError("release_cap_sigma_safety must be at least one")
    if config.min_budget_calibrated_sigma <= 0:
        raise ValueError("min_budget_calibrated_sigma must be positive")


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    payload: Dict[str, Any] = {}
    if args.config:
        import yaml

        payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        valid_fields = {item.name for item in dataclasses.fields(ExperimentConfig)}
        unknown = set(payload) - valid_fields
        if unknown:
            raise ValueError(f"Unknown YAML config keys: {sorted(unknown)}")
    preset = args.preset or payload.get("preset", "quick")
    config = ExperimentConfig.full_defaults() if preset == "full" else ExperimentConfig()
    for key, value in payload.items():
        setattr(config, key, value)
    overrides = {
        "dataset": args.dataset,
        "london_dir": args.london_dir,
        "cache_dir": args.cache_dir,
        "output_dir": args.output_dir,
        "models": args.models,
        "xai_methods": args.xai_methods,
        "seeds": args.seeds,
        "epsilon_budgets": args.epsilon_budgets,
        "n_endpoints": args.n_endpoints,
        "n_zones": args.n_zones,
        "n_steps": args.n_steps,
        "calibration_steps": args.calibration_steps,
        "adjacency": args.adjacency,
        "min_critical_coverage": args.min_critical_coverage,
        "min_jain_fairness": args.min_jain_fairness,
        "residual_noise_margin": args.residual_noise_margin,
        "residual_upgrade_margin": args.residual_upgrade_margin,
        "residual_decay": args.residual_decay,
        "override_gain_margin": args.override_gain_margin,
        "prism_traffic_guardrail_ratio": args.prism_traffic_guardrail_ratio,
        "prism_upgrade_budget_ratio": args.prism_upgrade_budget_ratio,
        "gain_per_bit_weight": args.gain_per_bit_weight,
        "aox_weight": args.aox_weight,
        "energy_weight": args.energy_weight,
        "privacy_price_weight": args.privacy_price_weight,
        "fairness_weight": args.fairness_weight,
        "critical_recall_weight": args.critical_recall_weight,
        "selective_private_release": args.selective_private_release,
        "max_private_releases_per_endpoint": args.max_private_releases_per_endpoint,
        "public_release_score_threshold": args.public_release_score_threshold,
        "release_cap_sigma_safety": args.release_cap_sigma_safety,
        "min_budget_calibrated_sigma": args.min_budget_calibrated_sigma,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    if args.run_ablations is not None:
        config.run_ablations = args.run_ablations
    if args.allow_prism_over_budget:
        config.allow_prism_over_budget = True
    if args.save_all_traces:
        config.save_all_traces = True
    config.preset = preset
    validate_config(config)
    return config


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["quick", "full"],
        default=None,
        help="Preset override. When omitted, a YAML preset is honored; otherwise quick is used.",
    )
    parser.add_argument(
        "--real-suite",
        choices=["verification", "full"],
        default=None,
        help="Run both real datasets from this single script.",
    )
    parser.add_argument(
        "--stress-suite",
        choices=["verification", "full"],
        default=None,
        help="Run compact wireless stress tests for burst loss and weak-channel users.",
    )
    parser.add_argument(
        "--tune-suite",
        action="store_true",
        help="Run a compact ML-style scheduler hyperparameter search.",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML configuration")
    parser.add_argument("--dataset", choices=["synthetic", "uci", "opsd", "london"], default=None)
    parser.add_argument("--london-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--models", nargs="+", choices=["tree", "neural"], default=None)
    parser.add_argument(
        "--xai-methods",
        nargs="+",
        choices=[
            "default",
            "treeshap",
            "feature_ablation",
            "integrated_gradients",
            "gradient_x_input",
        ],
        default=None,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epsilon-budgets", nargs="+", type=float, default=None)
    parser.add_argument("--n-endpoints", type=int, default=None)
    parser.add_argument("--n-zones", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--calibration-steps", type=int, default=None)
    parser.add_argument("--adjacency", choices=["add_remove", "replace_one"], default=None)
    parser.add_argument("--min-critical-coverage", type=float, default=None)
    parser.add_argument("--min-jain-fairness", type=float, default=None)
    parser.add_argument("--residual-noise-margin", type=float, default=None)
    parser.add_argument("--residual-upgrade-margin", type=float, default=None)
    parser.add_argument("--residual-decay", type=float, default=None)
    parser.add_argument("--override-gain-margin", type=float, default=None)
    parser.add_argument("--prism-traffic-guardrail-ratio", type=float, default=None)
    parser.add_argument("--prism-upgrade-budget-ratio", type=float, default=None)
    parser.add_argument("--allow-prism-over-budget", action="store_true")
    parser.add_argument("--gain-per-bit-weight", type=float, default=None)
    parser.add_argument("--aox-weight", type=float, default=None)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--privacy-price-weight", type=float, default=None)
    parser.add_argument("--fairness-weight", type=float, default=None)
    parser.add_argument("--critical-recall-weight", type=float, default=None)
    parser.add_argument(
        "--selective-private-release",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Spend privacy only on selected public high-value events.",
    )
    parser.add_argument("--max-private-releases-per-endpoint", type=int, default=None)
    parser.add_argument("--public-release-score-threshold", type=float, default=None)
    parser.add_argument("--release-cap-sigma-safety", type=float, default=None)
    parser.add_argument("--min-budget-calibrated-sigma", type=float, default=None)
    parser.add_argument(
        "--run-ablations",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--save-all-traces", action="store_true")
    return parser.parse_args(argv)


def suite_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    keys = [
        "models",
        "xai_methods",
        "seeds",
        "epsilon_budgets",
        "n_endpoints",
        "n_zones",
        "n_steps",
        "calibration_steps",
        "adjacency",
        "min_critical_coverage",
        "min_jain_fairness",
        "residual_noise_margin",
        "residual_upgrade_margin",
        "residual_decay",
        "override_gain_margin",
        "prism_traffic_guardrail_ratio",
        "prism_upgrade_budget_ratio",
        "gain_per_bit_weight",
        "aox_weight",
        "energy_weight",
        "privacy_price_weight",
        "fairness_weight",
        "critical_recall_weight",
        "selective_private_release",
        "max_private_releases_per_endpoint",
        "public_release_score_threshold",
        "release_cap_sigma_safety",
        "min_budget_calibrated_sigma",
    ]
    overrides = {
        key: getattr(args, key)
        for key in keys
        if getattr(args, key, None) is not None
    }
    if args.allow_prism_over_budget:
        overrides["allow_prism_over_budget"] = True
    if args.run_ablations is not None:
        overrides["run_ablations"] = args.run_ablations
    if args.save_all_traces:
        overrides["save_all_traces"] = True
    return overrides


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    active_modes = sum(
        int(value)
        for value in [
            args.real_suite is not None,
            args.stress_suite is not None,
            args.tune_suite,
        ]
    )
    if active_modes > 1:
        raise ValueError("Choose only one suite mode")
    if args.tune_suite:
        output_dir = Path(args.output_dir or "prism_v18_tuning")
        run_tuning_suite(output_dir)
        return
    if args.real_suite is not None:
        output_dir = Path(
            args.output_dir
            or (
                "prism_real_verification"
                if args.real_suite == "verification"
                else "prism_icc2027_results"
            )
        )
        run_real_data_suite(output_dir, args.real_suite, overrides=suite_overrides_from_args(args))
        return
    if args.stress_suite is not None:
        output_dir = Path(
            args.output_dir
            or (
                "prism_stress_verification"
                if args.stress_suite == "verification"
                else "prism_icc2027_stress_results"
            )
        )
        run_wireless_stress_suite(output_dir, args.stress_suite)
        return
    config = build_config(args)
    if config.output_dir == "prism_results":
        config.output_dir = f"prism_{config.dataset}_{config.preset}_results"
    run_pipeline(config)


if __name__ == "__main__":
    main()
