#!/usr/bin/env python3
"""
SoliTOC processor for TXT data files.

This Tkinter-based application reads SoliTOC TXT files, performs baseline
correction, temperature smoothing/extrapolation, and peak integration, then
exports processed data and summary reports as CSV and text files.

Recent additions include:
- a fixed-value baseline mode
- preview skipping for fixed-value processing
- weighted temperature statistics for the CO2 peak
- a normalized CO2 column in the exported CSV output
"""

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Default configuration values.
DEFAULT_BASELINE_START = 800
DEFAULT_BASELINE_END = 1800
DEFAULT_FIXED_BASELINE = (1200.0, 1800.0)
DEFAULT_FIXED_VAL = 1000.0
DEFAULT_SHIFT = 88
DEFAULT_FACTOR = 0.000012563
DEFAULT_INTERP_EVERY = 100
DEFAULT_EXTRAP_RATE = 0.01
DEFAULT_AUTO_WINDOW = 600
DEFAULT_PREVIEW = True
DEFAULT_MAX_ENERGY_TEMP = 900.0
DEFAULT_FLOW_COL_IDX = 5
DEFAULT_ZONE_FLOW_SUSTAIN_POINTS = 5
DEFAULT_ZONE_FLOW_INCREASE = 1.0
DEFAULT_ZONE_FLOW_RELATIVE_INCREASE = 0.25
TEMP_BIN_EDGES = (200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0)
MIN_TIME_POINTS_FOR_SEARCH = 2
MIN_B_MARKERS_FOR_D3 = 3
MIN_POINTS_FOR_INTEGRATION = 2

OUT_SUBDIRS = {
    "csv": "processed_csv",
    "rp_input": "RampedPyrox_Input",
    "summaries": "summaries",
}


def get_unique_output_path(path: str) -> str:
    """Return a file path that does not overwrite an existing file."""
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    index = 1
    while True:
        candidate = f"{base}({index}){ext}"
        if not os.path.exists(candidate):
            return candidate
        index += 1


# -------------------------
# Part 1: parsing and helper functions
# -------------------------


def calculate_weighted_temp_stats(
    df: pd.DataFrame, temp_column: str = "Temperature_Energy"
) -> tuple[float, float]:
    """
    Calculates the weighted mean and standard deviation of the temperature
    using the CO2 signal as the statistical weight.

    Args:
        df (pd.DataFrame): The DataFrame containing the selected temperature column and 'co2_shifted'.

    Returns:
        Tuple[float, float]: The weighted mean temperature and weighted standard deviation.
                             Returns (NaN, NaN) if calculation is impossible (e.g., zero CO2).
    """
    # Keep only rows with usable temperature and CO2 values.
    subset = df[[temp_column, "co2_shifted"]].dropna()

    # Convert the relevant columns to arrays for the weighted calculation.
    temp = subset[temp_column]
    co2 = subset["co2_shifted"]

    # Compute the total CO2 signal, which serves as the weighting denominator.
    total_co2 = co2.sum()

    # Guard against empty or flat inputs that would make the statistics undefined.
    if total_co2 == 0 or pd.isna(total_co2):
        return float("nan"), float("nan")

    # Calculate the weighted mean temperature.
    weighted_mean_temp = (temp * co2).sum() / total_co2

    # Calculate the weighted variance of the temperature distribution.
    variance = (co2 * ((temp - weighted_mean_temp) ** 2)).sum() / total_co2

    # Convert the variance to a standard deviation.
    weighted_std_dev = np.sqrt(variance)

    # Return the results as native floating-point values.
    return float(weighted_mean_temp), float(weighted_std_dev)


def parse_solitoc_txt_to_df(path: str) -> tuple[str | None, pd.DataFrame]:
    """
    Parses a SoliTOC text file into a header string and a Pandas DataFrame.

    Args:
        path (str): The file path to the TXT document.

    Returns:
        Tuple[Optional[str], pd.DataFrame]: The extracted header line and the
        parsed numerical data as a Pandas DataFrame.
    """
    header_line = None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        lines = [ln.rstrip("\n") for ln in fh.readlines()]

    # Locate the first non-empty line to use as the file header.
    for ln in lines:
        if ln.strip() != "":
            header_line = ln
            break

    data_lines = []
    # Parse numeric data rows while skipping blank lines and comment lines.
    for ln in lines:
        if ln.strip() == "" or ln.strip().startswith("#"):
            continue
        parts = ln.strip().split()
        row = []
        for p in parts:
            try:
                row.append(float(p))
            except Exception:
                row = []
                break
        if row:
            data_lines.append(row)

    if not data_lines:
        raise ValueError(f"No numeric data parsed from {path}")

    # Standardize the row widths by padding shorter rows with NaN values.
    maxcols = max(len(r) for r in data_lines)
    data2 = [r + [np.nan] * (maxcols - len(r)) for r in data_lines]
    cols = [f"c{i}" for i in range(maxcols)]

    # Store the parsed values in a Pandas DataFrame for downstream processing.
    df = pd.DataFrame(data2, columns=cols)
    return header_line, df


def parse_program_metadata(
    path: str, header_line: str | None = None
) -> dict[str, float | None]:
    """Extract Pyro program values and the D3-like B-line marker from a TXT file."""
    if header_line is None:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            lines = [ln.rstrip("\n") for ln in fh.readlines()]
        for ln in lines:
            if ln.strip():
                header_line = ln
                break
    else:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            lines = [ln.rstrip("\n") for ln in fh.readlines()]

    pyro_rate = None
    c1_seconds = None
    c4_seconds = None
    b_d3_time = None

    if header_line:
        m = re.search(
            r"Pyro\s*([+-]?\d+(?:\.\d+)?)\s*HZ\s*C1\s*([+-]?\d+(?:\.\d+)?)s\s*C4\s*([+-]?\d+(?:\.\d+)?)s",
            header_line,
            flags=re.IGNORECASE,
        )
        if m:
            pyro_rate = float(m.group(1))
            c1_seconds = float(m.group(2))
            c4_seconds = float(m.group(3))

    b_line = next((ln for ln in lines if ln.startswith("#  B=")), None)
    if b_line:
        # Keep the third marker (spreadsheet-like D3 reference) if available.
        b_markers = [
            float(x) for x in re.findall(r"([+-]?\d+(?:\.\d+)?)\s*-\s*-1", b_line)
        ]
        if len(b_markers) >= MIN_B_MARKERS_FOR_D3:
            b_d3_time = b_markers[2]

    return {
        "pyro_heat_rate_c_per_min": pyro_rate,
        "c1_s": c1_seconds,
        "c4_s": c4_seconds,
        "b_d3_time_s": b_d3_time,
    }


def get_new_base_filename(header_line: str, fallback_name: str) -> str:
    """
    Extracts the new base filename from the SoliTOC header line.

    Args:
        header_line (str): The raw header line extracted from the text file.
        fallback_name (str): The original filename to fall back on if parsing fails.

    Returns:
        str: The newly extracted filename, or the fallback name.
    """
    if not header_line:
        return fallback_name

    # Isolate part b) by finding "# Sample no [float] :" and capturing everything after it
    header_match = re.search(
        r"#\s*Sample\s+no\s+\d+(?:\.\d+)?\s*:(.*)", header_line, flags=re.IGNORECASE
    )

    if not header_match:
        return fallback_name

    part_b = header_match.group(1).strip()

    # Search part b) for a space, followed by a float, followed by "mg;"
    name_match = re.search(r"^(.*?)\s+\d+(?:\.\d+)?mg;", part_b)

    if name_match:
        return name_match.group(1).strip()
    else:
        return part_b


def extract_sample_weight_mg(header_line: str) -> float | None:
    """
    Extracts the sample weight in milligrams from the header string.

    Args:
        header_line (str): The header line containing the sample metadata.

    Returns:
        Optional[float]: The extracted weight as a float, or None if not found.
    """
    if not header_line:
        return None
    # Regex to find weight followed by mg
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mg[;,]?", header_line, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except (TypeError, ValueError):
            return None
    return None


def baseline_from_indices(
    co2: np.ndarray, start_idx: int, end_idx: int
) -> tuple[float, int, int]:
    """
    Calculates the mean baseline from specific data indices.

    Args:
        co2 (np.ndarray): The raw CO2 data array.
        start_idx (int): The starting index.
        end_idx (int): The ending index.

    Returns:
        Tuple[float, int, int]: The calculated baseline value, start index, and end index.
    """
    s = int(max(0, start_idx))
    e = int(min(len(co2), end_idx))
    if e <= s:
        raise ValueError("Baseline indices invalid (end <= start).")
    # Calculate the mean ignoring NaNs
    return float(np.nanmean(co2[s:e])), s, e


def baseline_from_time_range(
    time: np.ndarray, co2: np.ndarray, tstart: float, tend: float
) -> tuple[float, int, int]:
    """
    Calculates the mean baseline from a specific time range in seconds.

    Args:
        time (np.ndarray): The time array.
        co2 (np.ndarray): The raw CO2 data array.
        tstart (float): Start time in seconds.
        tend (float): End time in seconds.

    Returns:
        Tuple[float, int, int]: The calculated baseline value, start index, and end index.
    """
    mask = (time >= tstart) & (time <= tend)
    if np.sum(mask) == 0:
        raise ValueError(f"No data in baseline time range {tstart}-{tend} s.")
    idxs = np.where(mask)[0]
    return float(np.nanmean(co2[mask])), int(idxs[0]), int(idxs[-1])


def detect_best_baseline_window(
    time: np.ndarray,
    co2: np.ndarray,
    search_start: float | None = None,
    search_end: float | None = None,
    window_size: float = DEFAULT_AUTO_WINDOW,
) -> tuple[float, int, int]:
    """
    Automatically finds the most stable window for baseline calculation based on standard deviation.

    Args:
        time (np.ndarray): The time array.
        co2 (np.ndarray): The raw CO2 data array.
        search_start (Optional[float]): The start time to begin searching.
        search_end (Optional[float]): The end time to stop searching.
        window_size (float): The size of the window to check for stability.

    Returns:
        Tuple[float, int, int]: The calculated baseline value, start index, and end index.
    """
    if search_start is None:
        search_start = 0.0
    if search_end is None:
        search_end = 2000.0
    if len(time) < MIN_TIME_POINTS_FOR_SEARCH:
        raise ValueError("Too few points for auto baseline detection.")

    dt = np.median(np.diff(time))
    if dt <= 0:
        raise ValueError("Non-increasing time vector.")

    w_pts = max(2, int(round(window_size / dt)))
    idx_candidates = np.where((time >= search_start) & (time <= search_end))[0]

    if len(idx_candidates) == 0:
        raise ValueError("No data in auto-detect search range.")

    best_score = None
    best_i = None

    # Iterate through candidates to find the lowest deviation score
    for i in idx_candidates:
        j = i + w_pts
        if j >= len(co2) or time[j] > search_end:
            break
        window = co2[i : j + 1]
        if np.all(np.isnan(window)):
            continue
        sd = float(np.nanstd(window))
        mean_abs_step = float(np.nanmean(np.abs(np.diff(window))))
        score = sd + 0.05 * mean_abs_step
        if best_score is None or score < best_score:
            best_score = score
            best_i = i

    if best_i is None:
        raise ValueError("Auto baseline detection failed to find a stable window.")

    i0, i1 = best_i, min(best_i + w_pts, len(co2) - 1)
    baseline_val = float(np.nanmean(co2[i0 : i1 + 1]))
    return baseline_val, int(i0), int(i1)


def detect_oxidation_start(  # noqa: PLR0913
    time: np.ndarray,
    flow_lance: np.ndarray,
    search_start_idx: int,
    sustain_points: int = DEFAULT_ZONE_FLOW_SUSTAIN_POINTS,
    min_increase: float = DEFAULT_ZONE_FLOW_INCREASE,
    min_relative_increase: float = DEFAULT_ZONE_FLOW_RELATIVE_INCREASE,
) -> int:
    """Find the first sustained flow-lance increase after the second temperature plateau."""
    flow = np.asarray(flow_lance, dtype=float)
    if search_start_idx >= len(flow) - 1:
        return len(flow) - 1

    plateau_slice = flow[
        max(0, search_start_idx - sustain_points) : search_start_idx + sustain_points
    ]
    finite = plateau_slice[np.isfinite(plateau_slice)]
    baseline_flow = (
        float(np.nanmedian(finite)) if len(finite) else float(flow[search_start_idx])
    )
    target_flow = baseline_flow + max(
        min_increase, baseline_flow * min_relative_increase
    )
    start = max(0, int(search_start_idx))
    stop = max(start + 1, len(flow) - sustain_points)

    for idx in range(start + 1, stop):
        window = flow[idx : idx + sustain_points]
        if len(window) < sustain_points or np.any(~np.isfinite(window)):
            continue
        if float(np.nanmedian(window)) >= target_flow:
            return idx

    diffs = np.diff(flow, prepend=flow[0])
    tail = diffs[start + 1 :]
    finite_tail = np.where(np.isfinite(tail))[0]
    if len(finite_tail) == 0:
        return min(len(flow) - 1, start + 1)
    best_offset = int(finite_tail[int(np.nanargmax(tail[finite_tail]))])
    return min(len(flow) - 1, start + 1 + best_offset)


def detect_zone_boundaries(  # noqa: PLR0913
    time: np.ndarray,
    oxid_tube: np.ndarray,
    flow_lance: np.ndarray,
    flow_relative_increase: float = DEFAULT_ZONE_FLOW_RELATIVE_INCREASE,
    ramp_reference_time: float | None = None,
    c4_seconds: float | None = None,
) -> dict[str, Any]:
    """Determine the five named zone boundaries from the time, temperature, and flow signals."""
    time = np.asarray(time, dtype=float)
    oxid_tube = np.asarray(oxid_tube, dtype=float)
    flow_lance = np.asarray(flow_lance, dtype=float)
    if len(time) == 0:
        raise ValueError("Cannot detect zone boundaries on an empty file.")

    start_idx = 0
    end_idx = len(time) - 1
    oxidation_search_start_idx = max(1, int(round((end_idx - start_idx) * 0.70)))
    start_oxidation_idx = detect_oxidation_start(
        time,
        flow_lance,
        oxidation_search_start_idx,
        min_relative_increase=flow_relative_increase,
    )
    start_oxidation_idx = min(
        end_idx, max(start_oxidation_idx, oxidation_search_start_idx)
    )

    t_start = float(time[start_idx])
    t_oxid = float(time[start_oxidation_idx])

    if ramp_reference_time is not None and np.isfinite(ramp_reference_time):
        start_ramp_time = float(ramp_reference_time)
    elif c4_seconds is not None and np.isfinite(c4_seconds):
        start_ramp_time = t_oxid - float(c4_seconds)
    else:
        start_ramp_time = t_start

    if c4_seconds is not None and np.isfinite(c4_seconds):
        start_plateau_time = t_oxid - float(c4_seconds)
    elif ramp_reference_time is not None and np.isfinite(ramp_reference_time):
        start_plateau_time = float(ramp_reference_time)
    else:
        start_plateau_time = start_ramp_time

    start_ramp_time = float(np.clip(start_ramp_time, t_start, t_oxid))
    start_plateau_time = float(np.clip(start_plateau_time, start_ramp_time, t_oxid))

    idx_lookup = np.arange(len(time))

    def _nearest_idx(target_time: float) -> int:
        return int(idx_lookup[np.nanargmin(np.abs(time - target_time))])

    start_ramp_idx = _nearest_idx(start_ramp_time)
    start_plateau_idx = _nearest_idx(start_plateau_time)

    boundary_indices = {
        "Start-Run": start_idx,
        "Start-Ramp": start_ramp_idx,
        "Start-Plateau": start_plateau_idx,
        "Start-Oxidation": start_oxidation_idx,
        "End-Run": end_idx,
    }
    boundary_times = {name: float(time[idx]) for name, idx in boundary_indices.items()}
    return {
        "indices": boundary_indices,
        "times": boundary_times,
        "plateau_1_window": (None, None),
        "plateau_2_window": (None, None),
    }


def build_zone_labels(length: int, boundary_indices: dict[str, int]) -> np.ndarray:
    """Label each row with its inferred processing zone."""
    labels = np.full(length, "oxidation", dtype=object)
    start_ramp = boundary_indices["Start-Ramp"]
    start_plateau = boundary_indices["Start-Plateau"]
    start_oxidation = boundary_indices["Start-Oxidation"]
    labels[:start_ramp] = "flush"
    labels[start_ramp:start_plateau] = "pyrolysis-ramp"
    labels[start_plateau:start_oxidation] = "pyrolysis-plateau"
    labels[start_oxidation:] = "oxidation"
    return labels


def enforce_strictly_increasing(
    values: np.ndarray, start_idx: int, end_idx: int, eps: float = 1e-4
) -> None:
    """Force strict monotonic increase in-place on a closed index interval."""
    if end_idx <= start_idx:
        return
    for idx in range(start_idx + 1, end_idx + 1):
        if not np.isfinite(values[idx]):
            values[idx] = values[idx - 1] + eps
        elif values[idx] <= values[idx - 1]:
            values[idx] = values[idx - 1] + eps


def snapshot_smooth_between_points(  # noqa: PLR0913
    time: np.ndarray,
    temp_raw: np.ndarray,
    start_idx: int,
    end_idx: int,
    snapshot_seconds: float,
    end_temp: float,
) -> np.ndarray:
    """Apply snapshot-based linear smoothing only between two boundary points."""
    out = np.asarray(temp_raw, dtype=float).copy()
    if end_idx <= start_idx:
        return out

    ts = np.asarray(time, dtype=float)
    tr = np.asarray(temp_raw, dtype=float)
    t0 = float(ts[start_idx])
    t1 = float(ts[end_idx])

    step = max(float(snapshot_seconds), 1e-6)
    anchor_times = [t0]
    cur = t0 + step
    while cur < t1:
        anchor_times.append(cur)
        cur += step
    anchor_times.append(t1)

    idx_lookup = np.arange(len(ts))

    def _nearest_idx(target_t: float) -> int:
        return int(idx_lookup[np.nanargmin(np.abs(ts - target_t))])

    anchor_indices = [_nearest_idx(t) for t in anchor_times]
    anchor_values = [
        float(tr[idx]) if np.isfinite(tr[idx]) else float(tr[start_idx])
        for idx in anchor_indices
    ]
    anchor_values[-1] = float(end_temp)

    seg_time = ts[start_idx : end_idx + 1]
    seg_vals = np.interp(
        seg_time,
        np.asarray(anchor_times, dtype=float),
        np.asarray(anchor_values, dtype=float),
    )
    out[start_idx : end_idx + 1] = seg_vals
    enforce_strictly_increasing(out, start_idx, end_idx)
    return out


def build_temperature_profiles(  # noqa: PLR0913
    time: np.ndarray,
    temp_raw: np.ndarray,
    boundary_indices: dict[str, int],
    snapshot_seconds: float,
    pyro_heat_rate_c_per_min: float | None,
    max_energy_temp_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Temperature_thermogram and Temperature_Energy profiles linked to time."""
    ts = np.asarray(time, dtype=float)
    tr = np.asarray(temp_raw, dtype=float)
    n = len(ts)
    if n == 0:
        return tr.copy(), tr.copy()

    start_ramp_idx = int(boundary_indices["Start-Ramp"])
    start_plateau_idx = int(boundary_indices["Start-Plateau"])
    start_oxid_idx = int(boundary_indices["Start-Oxidation"])

    start_ramp_idx = int(np.clip(start_ramp_idx, 0, n - 1))
    start_plateau_idx = int(np.clip(start_plateau_idx, start_ramp_idx, n - 1))
    start_oxid_idx = int(np.clip(start_oxid_idx, start_plateau_idx, n - 1))

    plateau_temp = (
        float(tr[start_plateau_idx])
        if np.isfinite(tr[start_plateau_idx])
        else float(tr[start_ramp_idx])
    )
    pyro_rate_per_sec = max(0.0, float(pyro_heat_rate_c_per_min or 0.0) / 60.0)

    thermogram = snapshot_smooth_between_points(
        ts,
        tr,
        start_ramp_idx,
        start_plateau_idx,
        snapshot_seconds=snapshot_seconds,
        end_temp=plateau_temp,
    )
    energy = thermogram.copy()

    # Thermogram: steady increase from Start-Plateau to end with Pyro rate.
    for idx in range(start_plateau_idx + 1, n):
        dt = max(0.0, float(ts[idx] - ts[idx - 1]))
        thermogram[idx] = thermogram[idx - 1] + pyro_rate_per_sec * dt
    enforce_strictly_increasing(thermogram, start_ramp_idx, n - 1)

    # Energy: linear ramp to Start-Oxidation with minimum +2 degC, then steady Pyro-rate increase.
    target_oxid_temp = max(
        float(max_energy_temp_c), float(energy[start_plateau_idx]) + 2.0
    )
    if start_oxid_idx > start_plateau_idx:
        t_plateau = float(ts[start_plateau_idx])
        t_oxid = float(ts[start_oxid_idx])
        for idx in range(start_plateau_idx, start_oxid_idx + 1):
            frac = (
                0.0
                if t_oxid <= t_plateau
                else (float(ts[idx]) - t_plateau) / (t_oxid - t_plateau)
            )
            energy[idx] = float(energy[start_plateau_idx]) + frac * (
                target_oxid_temp - float(energy[start_plateau_idx])
            )
        enforce_strictly_increasing(energy, start_plateau_idx, start_oxid_idx)

    for idx in range(start_oxid_idx + 1, n):
        dt = max(0.0, float(ts[idx] - ts[idx - 1]))
        energy[idx] = energy[idx - 1] + pyro_rate_per_sec * dt
    enforce_strictly_increasing(energy, start_ramp_idx, n - 1)

    return np.round(thermogram, 4), np.round(energy, 4)


def calculate_zone_and_bin_metrics(df_out: pd.DataFrame) -> dict[str, float]:
    """Calculate CO2-derived zone and temperature-bin metrics for summaries."""
    co2 = pd.to_numeric(df_out["co2_shifted"], errors="coerce").fillna(0.0)
    zone = df_out["zone"]
    temp_e = pd.to_numeric(df_out["Temperature_Energy"], errors="coerce")

    mask_ramp = zone == "pyrolysis-ramp"
    mask_plateau = zone == "pyrolysis-plateau"
    mask_oxid = zone == "oxidation"
    mask_pyro = mask_ramp | mask_plateau

    ramp_sum = float(co2[mask_ramp].sum())
    plateau_sum = float(co2[mask_plateau].sum())
    oxid_sum = float(co2[mask_oxid].sum())

    denom_total = ramp_sum + plateau_sum + oxid_sum
    non_pyro = float("nan") if denom_total <= 0 else (oxid_sum / denom_total) * 100.0

    pyro_sum = float(co2[mask_pyro].sum())
    metrics: dict[str, float] = {
        "non-pyrolyzable [%total CO2]": non_pyro,
    }

    # Lowest bin includes all ramp/plateau CO2 up to 200°C.
    bin_defs = [
        ("100-200°C [%]", mask_pyro & (temp_e < TEMP_BIN_EDGES[0])),
        ("200-300°C [%]", mask_pyro & (temp_e >= TEMP_BIN_EDGES[0]) & (temp_e < TEMP_BIN_EDGES[1])),
        ("300-400°C [%]", mask_pyro & (temp_e >= TEMP_BIN_EDGES[1]) & (temp_e < TEMP_BIN_EDGES[2])),
        ("400-500°C [%]", mask_pyro & (temp_e >= TEMP_BIN_EDGES[2]) & (temp_e < TEMP_BIN_EDGES[3])),
        ("500-600°C [%]", mask_pyro & (temp_e >= TEMP_BIN_EDGES[3]) & (temp_e < TEMP_BIN_EDGES[4])),
        ("600-700°C [%]", mask_pyro & (temp_e >= TEMP_BIN_EDGES[4]) & (temp_e < TEMP_BIN_EDGES[5])),
        ("700-800°C [%]", mask_pyro & (temp_e >= TEMP_BIN_EDGES[5]) & (temp_e < TEMP_BIN_EDGES[6])),
        # Highest bin includes all ramp/plateau CO2 from 800°C to Start-Oxidation boundary.
        ("800-900°C [%]", mask_pyro & (temp_e >= TEMP_BIN_EDGES[6])),
    ]

    for name, mask in bin_defs:
        bin_sum = float(co2[mask.fillna(False)].sum())
        metrics[name] = float("nan") if pyro_sum <= 0 else (bin_sum / pyro_sum) * 100.0

    return metrics


def format_batch_summary_row(result: dict[str, Any], file_name: str) -> dict[str, Any]:
    """Format one batch summary row using the public display column names."""
    row = {
        "file": file_name,
        "area": result.get("area"),
        "ugC/sample": result.get("ugC"),
        "sample weight [mg]": result.get("sample_weight_mg"),
        "TOC [mgC]": result.get("TOC_mgC"),
        "TOC [wt.%]": result.get("TOC_wtpercent"),
        "Heat rate [C]": result.get("pyro_heat_rate_c_per_min"),
        "C1 [s]": result.get("c1_s"),
        "C4 [s]": result.get("c4_s"),
        "mean temperature pyrolysis [C]": result.get("mean_temp_co2"),
        "std-div temperature pyrolysis [C]": result.get("std_dev_temp_co2"),
        "non-pyrolyzable [%total CO2]": result.get("non-pyrolyzable [%total CO2]"),
        "100-200C [%]": result.get("100-200°C [%]"),
        "200-300C [%]": result.get("200-300°C [%]"),
        "300-400C [%]": result.get("300-400°C [%]"),
        "400-500C [%]": result.get("400-500°C [%]"),
        "500-600C [%]": result.get("500-600°C [%]"),
        "600-700C [%]": result.get("600-700°C [%]"),
        "700-800C [%]": result.get("700-800°C [%]"),
        "800-900C [%]": result.get("800-900°C [%]"),
    }

    for key, value in result.get("zone_timepoints", {}).items():
        row[f"{key} [s]"] = value

    return row


def _sanitize_excel_sheet_name(name: str) -> str:
    cleaned = re.sub(r"[\\/?*:\[\]]", "_", str(name)).strip() or "Sheet1"
    return cleaned[:31]


def write_batch_exports(
    output_dir: Path,
    batch_summary_name: str,
    summary_rows: list[dict[str, Any]],
    thermogram_exports: list[tuple[str, pd.DataFrame]],
) -> dict[str, str]:
    """Write the batch summary CSV/XLSX pair and the thermogram workbook."""
    batch_name = (
        re.sub(r"[^A-Za-z0-9._ -]+", "_", batch_summary_name).strip().strip(" .")
        or "batch_summary"
    )

    summary_df = pd.DataFrame(summary_rows)
    batch_csv_path = get_unique_output_path(
        str(output_dir / f"{batch_name}_batch_summary.csv")
    )
    summary_df.to_csv(batch_csv_path, index=False)

    batch_xlsx_path = get_unique_output_path(
        str(output_dir / f"{batch_name}_batch_summary.xlsx")
    )
    with pd.ExcelWriter(batch_xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="batch_summary")

    thermogram_xlsx_path = get_unique_output_path(
        str(output_dir / f"{batch_name}_thermograms.xlsx")
    )
    with pd.ExcelWriter(thermogram_xlsx_path, engine="openpyxl") as writer:
        for sheet_name, thermogram_df in thermogram_exports:
            thermogram_df.to_excel(
                writer, index=False, sheet_name=_sanitize_excel_sheet_name(sheet_name)
            )

    return {
        "batch_csv_path": batch_csv_path,
        "batch_xlsx_path": batch_xlsx_path,
        "thermogram_xlsx_path": thermogram_xlsx_path,
    }


def shift_co2_array(co2: np.ndarray, shift: int = DEFAULT_SHIFT) -> np.ndarray:
    """Shift the CO2 signal earlier by a fixed number of points."""
    n = len(co2)
    if shift <= 0:
        return np.asarray(co2, dtype=float).copy()
    shifted = np.full_like(np.asarray(co2, dtype=float), np.nan)
    if shift < n:
        shifted[: n - shift] = np.asarray(co2, dtype=float)[shift:]
    return shifted


def integrate_trap(y: np.ndarray, x: np.ndarray) -> float:
    """
    Performs trapezoidal integration over a curve.

    Args:
        y (np.ndarray): The Y values (e.g., CO2).
        x (np.ndarray): The X values (e.g., Time).

    Returns:
        float: The integrated area.
    """
    y, x = np.asarray(y), np.asarray(x)
    mask = ~np.isnan(y) & ~np.isnan(x)
    if np.sum(mask) < MIN_POINTS_FOR_INTEGRATION:
        return 0.0

    y = y[mask]
    x = x[mask]
    dx = np.diff(x)
    if np.any(dx < 0):
        # Ensure time is monotonic increasing for trapezoid calculation
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        dx = np.diff(x)

    return float(np.sum(0.5 * (y[:-1] + y[1:]) * dx))


# -------------------------
# Part 2: preview helpers
# -------------------------
def compute_baseline_and_preview_data(  # noqa: PLR0913
    path: str,
    time_col_index: int,
    temp_col_index: int,
    co2_col_index: int,
    baseline_mode: str,
    manual_baseline: tuple[int, int],
    fixed_baseline: tuple[float, float],
    auto_search_range: tuple[float | None, float | None],
    auto_window_size: float,
    fixed_val: float = DEFAULT_FIXED_VAL,
) -> tuple[str | None, np.ndarray, np.ndarray, float, int, int, pd.DataFrame]:
    """
    Calculates baseline values based on the selected GUI mode and extracts raw arrays.

    Args:
        path (str): File path to read.
        time_col_index (int): Column index for time.
        temp_col_index (int): Column index for temperature.
        co2_col_index (int): Column index for CO2.
        baseline_mode (str): The chosen mode ('manual', 'fixed', 'auto', 'fixed_val').
        manual_baseline (Tuple[int, int]): Start and end indices for manual mode.
        fixed_baseline (Tuple[float, float]): Start and end times for fixed window mode.
        auto_search_range (Tuple[Optional[float], Optional[float]]): Search bounds for auto mode.
        auto_window_size (float): The stability window size for auto mode.
        fixed_val (float): The user-defined constant to subtract (for 'fixed_val' mode).

    Returns:
        Tuple containing the header line, time array, CO2 array, calculated baseline value,
        start index, end index, and the full DataFrame.
    """
    header_line, df = parse_solitoc_txt_to_df(path)
    ncols = df.shape[1]

    # Handle negative indexing for the last columns.
    if co2_col_index < 0:
        co2_col_index = ncols + co2_col_index
    if time_col_index < 0 or time_col_index >= ncols:
        raise IndexError("time_col_index out of bounds")
    if co2_col_index < 0 or co2_col_index >= ncols:
        raise IndexError("co2_col_index out of bounds")

    # Extract columns to arrays.
    time = df.iloc[:, time_col_index].to_numpy()
    co2_raw = df.iloc[:, co2_col_index].to_numpy()

    # Determine the baseline value based on the selected mode.
    if baseline_mode == "manual":
        bstart, bend = manual_baseline
        baseline_val, bi0, bi1 = baseline_from_indices(co2_raw, bstart, bend)
    elif baseline_mode == "fixed":
        t0, t1 = fixed_baseline
        baseline_val, bi0, bi1 = baseline_from_time_range(time, co2_raw, t0, t1)
    elif baseline_mode == "auto":
        s0, s1 = auto_search_range
        baseline_val, bi0, bi1 = detect_best_baseline_window(
            time, co2_raw, search_start=s0, search_end=s1, window_size=auto_window_size
        )
    elif baseline_mode == "fixed_val":
        baseline_val = fixed_val
        bi0, bi1 = 0, 0
    else:
        raise ValueError("Unknown baseline_mode for preview")

    return header_line, time, co2_raw, baseline_val, bi0, bi1, df


# -------------------------
# Part 3: file processing
# -------------------------
def process_file(  # noqa: PLR0912, PLR0913, PLR0915
    path: str,
    out_dir: str,
    time_col_index: int = 0,
    temp_col_index: int = 1,
    co2_col_index: int = -1,
    flow_col_index: int = 2,
    baseline_mode: str = "manual",
    manual_baseline: tuple[int, int] = (DEFAULT_BASELINE_START, DEFAULT_BASELINE_END),
    fixed_baseline: tuple[float, float] = DEFAULT_FIXED_BASELINE,
    auto_search_range: tuple[float | None, float | None] = (None, None),
    auto_window_size: float = DEFAULT_AUTO_WINDOW,
    fixed_val: float = DEFAULT_FIXED_VAL,
    shift: int = DEFAULT_SHIFT,
    factor: float = DEFAULT_FACTOR,
    interp_every: float = DEFAULT_INTERP_EVERY,
    max_energy_temp: float = DEFAULT_MAX_ENERGY_TEMP,
    temp_interval: tuple[float, float] | None = None,
    zone_flow_relative_increase: float = DEFAULT_ZONE_FLOW_RELATIVE_INCREASE,
    plot_Tmin: float | None = None,
    plot_Tmax: float | None = None,
    plot_Ymin: float | None = None,
    plot_Ymax: float | None = None,
) -> dict[str, Any]:
    """
    Main controller function to process a single file, calculate results, and save to CSV.

    Args:
        path (str): File path to the input text file.
        out_dir (str): Directory where outputs should be saved.
        time_col_index (int): Column index for time.
        temp_col_index (int): Column index for temperature.
        co2_col_index (int): Column index for CO2.
        flow_col_index (int): Column index for Flow.
        baseline_mode (str): Baseline mode ('manual', 'fixed', 'auto', 'fixed_val').
        manual_baseline (Tuple[int, int]): Indices for manual mode.
        fixed_baseline (Tuple[float, float]): Times for fixed mode.
        auto_search_range (Tuple[Optional[float], Optional[float]]): Search times for auto mode.
        auto_window_size (float): Window size for auto mode.
        fixed_val (float): Constant value for 'fixed_val' mode.
        preview_baseline (bool): If True, shows the preview modal.
        shift (int): Number of points to shift.
        factor (float): Conversion factor for Area to ugC.
        interp_every (float): Snapshot spacing in seconds for temperature smoothing.
        max_energy_temp (float): Target temperature at Start-Oxidation for Temperature_Energy.
        temp_interval (Optional[Tuple[float, float]]): Optional temperature bounds for integration.
        plot_Tmin/plot_Tmax/plot_Ymin/plot_Ymax: (Legacy unused arguments).

    Returns:
        Dict[str, Any]: A dictionary containing processing results and metadata.
    """
    # Ensure the output subfolders exist before writing results.
    for sub in OUT_SUBDIRS.values():
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    header_line, df = parse_solitoc_txt_to_df(path)
    ncols = df.shape[1]

    # Resolve negative column indices so they refer to the end of the row.
    if co2_col_index < 0:
        co2_col_index = ncols + co2_col_index
    if flow_col_index < 0:
        flow_col_index = ncols + flow_col_index

    # Validate the requested column indices before accessing them.
    for idx in (time_col_index, temp_col_index, co2_col_index, flow_col_index):
        if idx < 0 or idx >= ncols:
            raise IndexError(
                f"Column index {idx} out of bounds (file has {ncols} cols)."
            )

    time = df.iloc[:, time_col_index].to_numpy()
    temp_raw = df.iloc[:, temp_col_index].to_numpy()
    co2_raw = df.iloc[:, co2_col_index].to_numpy()
    flow_raw = df.iloc[:, flow_col_index].to_numpy()
    program_meta = parse_program_metadata(path, header_line=header_line)
    zone_boundaries = detect_zone_boundaries(
        time,
        temp_raw,
        flow_raw,
        flow_relative_increase=zone_flow_relative_increase,
        ramp_reference_time=program_meta.get("b_d3_time_s"),
        c4_seconds=program_meta.get("c4_s"),
    )

    # Determine the baseline level from the selected processing mode.
    if baseline_mode == "manual":
        bstart, bend = int(manual_baseline[0]), int(manual_baseline[1])
        baseline_val, bi0, bi1 = baseline_from_indices(co2_raw, bstart, bend)
    elif baseline_mode == "fixed":
        t0, t1 = fixed_baseline
        baseline_val, bi0, bi1 = baseline_from_time_range(time, co2_raw, t0, t1)
    elif baseline_mode == "auto":
        s0, s1 = auto_search_range
        baseline_val, bi0, bi1 = detect_best_baseline_window(
            time, co2_raw, search_start=s0, search_end=s1, window_size=auto_window_size
        )
    elif baseline_mode == "fixed_val":
        baseline_val = fixed_val
        bi0, bi1 = 0, 0
    else:
        raise ValueError("Unknown baseline_mode.")

    # Remove the baseline contribution from the CO2 signal.
    co2_bc = co2_raw - baseline_val
    co2_bc[co2_bc < 0] = 0.0

    # Align CO2 with the configured lag correction; temperature remains in native time.
    co2_shifted = shift_co2_array(co2_bc, shift=shift)

    # Build both temperature profiles linked to time and zone boundaries.
    temp_thermogram, temp_energy = build_temperature_profiles(
        time,
        temp_raw,
        zone_boundaries["indices"],
        snapshot_seconds=float(interp_every),
        pyro_heat_rate_c_per_min=program_meta.get("pyro_heat_rate_c_per_min"),
        max_energy_temp_c=float(max_energy_temp),
    )

    # Integrate the corrected signal to estimate TOC.
    if temp_interval is None:
        idx_start = int(max(0, zone_boundaries["indices"]["Start-Ramp"]))
        mask = np.arange(len(time)) >= idx_start
        area = integrate_trap(co2_shifted[mask], time[mask])
    else:
        Tmin, Tmax = temp_interval
        mask = (temp_energy >= Tmin) & (temp_energy <= Tmax) & ~np.isnan(co2_shifted)
        if np.sum(mask) < MIN_POINTS_FOR_INTEGRATION:
            area = 0.0
        else:
            area = float(np.trapz(co2_shifted[mask], time[mask]))

    ugC = area * factor

    # Convert the integrated result to mass-based and weight-percent values using the sample mass.
    sample_wt_mg = extract_sample_weight_mg(header_line)
    TOC_mgC = None
    TOC_wtpercent = None
    user_pctC = None
    if sample_wt_mg is not None:
        TOC_mgC = ugC / 1000.0
        TOC_wtpercent = 100.0 * (TOC_mgC / sample_wt_mg) if sample_wt_mg != 0 else None
        user_pctC = (ugC * 1000.0 / sample_wt_mg) * 12.0 if sample_wt_mg != 0 else None

    # Derive a clean base filename from the input file and header metadata.
    orig_basefn = os.path.splitext(os.path.basename(path))[0]
    basefn = get_new_base_filename(header_line, orig_basefn)

    # Build the output table with the processed signal columns.
    df_out = pd.DataFrame(
        {
            "time_s": time,
            "temp_raw": temp_raw,
            "Temperature_thermogram": temp_thermogram,
            "Temperature_Energy": temp_energy,
            "co2_raw": co2_raw,
            "co2_baseline_corrected": co2_bc,
            "co2_shifted": co2_shifted,
            "flow_lance_ml_min": flow_raw,
        }
    )
    df_out["zone"] = build_zone_labels(len(df_out), zone_boundaries["indices"])

    # Calculate the normalized CO2 fraction from the shifted signal.
    total_co2_shifted = df_out["co2_shifted"].sum(skipna=True)
    if total_co2_shifted != 0 and not pd.isna(total_co2_shifted):
        df_out["co2_normalized"] = df_out["co2_shifted"] / total_co2_shifted
    else:
        df_out["co2_normalized"] = pd.NA

    zone_metrics = calculate_zone_and_bin_metrics(df_out)

    # Compute weighted temperature statistics only over the ramp-to-oxidation interval.
    stats_start_idx = int(max(0, zone_boundaries["indices"]["Start-Ramp"]))
    stats_end_idx = int(
        min(len(df_out) - 1, zone_boundaries["indices"]["Start-Oxidation"])
    )
    if stats_end_idx < stats_start_idx:
        mean_temp_co2, std_dev_temp_co2 = float("nan"), float("nan")
    else:
        stats_df = df_out.iloc[stats_start_idx : stats_end_idx + 1]
        mean_temp_co2, std_dev_temp_co2 = calculate_weighted_temp_stats(
            stats_df, temp_column="Temperature_Energy"
        )

    csv_path = get_unique_output_path(
        os.path.join(out_dir, OUT_SUBDIRS["csv"], f"{basefn}_thermograms.csv")
    )
    # Save the file retaining all decimals
    df_out.to_csv(csv_path, index=False)

    rp_input_basefn = f"{basefn}_RP-Input"
    rp_input_path = get_unique_output_path(
        os.path.join(out_dir, OUT_SUBDIRS["rp_input"], f"{rp_input_basefn}.csv")
    )
    rp_start = int(max(0, zone_boundaries["indices"]["Start-Ramp"]))
    rp_end = int(min(len(df_out) - 1, zone_boundaries["indices"]["Start-Oxidation"]))
    if rp_end >= rp_start:
        rp_source = df_out.iloc[rp_start : rp_end + 1]
        rp_df = pd.DataFrame(
            {
                "date_time": rp_source["time_s"].to_numpy(),
                "temp": rp_source["Temperature_Energy"].to_numpy(),
                "CO2_scaled": rp_source["co2_shifted"].to_numpy(),
            }
        )
    else:
        rp_df = pd.DataFrame(columns=["date_time", "temp", "CO2_scaled"])
    rp_df.to_csv(rp_input_path, index=False)

    # Write a text summary with the key processing results.
    summary_path = get_unique_output_path(
        os.path.join(out_dir, OUT_SUBDIRS["summaries"], f"{basefn}_summary.txt")
    )
    with open(summary_path, "w") as fh:
        fh.write(f"File: {path}\n")
        fh.write(f"New Filename: {basefn}\n")
        fh.write(f"Baseline mode: {baseline_mode}\n")
        fh.write(f"Baseline value: {baseline_val}\n")
        if baseline_mode == "fixed_val":
            fh.write("Baseline index window: N/A (Fixed User Value)\n")
        else:
            fh.write(f"Baseline index window: {bi0} .. {bi1}\n")
        fh.write(f"Integration area: {area}\n")
        fh.write(f"µgC (area * {factor}): {ugC}\n")
        fh.write(f"Sample weight (mg): {sample_wt_mg}\n")
        fh.write(f"TOC (mgC): {TOC_mgC}\n")
        fh.write(f"TOC (wt%): {TOC_wtpercent}\n")
        fh.write(f"user_pctC (your formula): {user_pctC}\n")
        fh.write(f"Temp snapshot spacing (s): {interp_every}\n")
        fh.write(f"Energy max temp at Start-Oxidation (°C): {max_energy_temp}\n")
        fh.write(
            f"Pyro heat rate (°C/min): {program_meta.get('pyro_heat_rate_c_per_min')}\n"
        )
        fh.write(f"C1 (s): {program_meta.get('c1_s')}\n")
        fh.write(f"C4 (s): {program_meta.get('c4_s')}\n")
        fh.write(f"Start-Run: {zone_boundaries['times']['Start-Run']}\n")
        fh.write(f"Start-Ramp: {zone_boundaries['times']['Start-Ramp']}\n")
        fh.write(f"Start-Plateau: {zone_boundaries['times']['Start-Plateau']}\n")
        fh.write(f"Start-Oxidation: {zone_boundaries['times']['Start-Oxidation']}\n")
        fh.write(f"End-Run: {zone_boundaries['times']['End-Run']}\n")
        if temp_interval is not None:
            fh.write(f"Integration temp interval: {temp_interval}\n")

        # Include the weighted temperature statistics in the summary output.
        fh.write(f"Weighted Mean Temp (°C): {mean_temp_co2}\n")
        fh.write(f"Weighted Temp StdDev: {std_dev_temp_co2}\n")

    return {
        "area": area,
        "ugC": ugC,
        "sample_weight_mg": sample_wt_mg,
        "TOC_mgC": TOC_mgC,
        "TOC_wtpercent": TOC_wtpercent,
        "user_pctC": user_pctC,
        "baseline": baseline_val,
        "baseline_idx": (bi0, bi1),
        "basefn": basefn,
        "zone_timepoints": zone_boundaries["times"],
        "zone_indices": zone_boundaries["indices"],
        "pyro_heat_rate_c_per_min": program_meta.get("pyro_heat_rate_c_per_min"),
        "c1_s": program_meta.get("c1_s"),
        "c4_s": program_meta.get("c4_s"),
        "b_d3_time_s": program_meta.get("b_d3_time_s"),
        "mean_temp_co2": mean_temp_co2,
        "std_dev_temp_co2": std_dev_temp_co2,
        "thermogram_df": df_out,
        **zone_metrics,
    }


def run_zone_analysis(params: dict[str, Any]) -> None:
    """Run the Kedro batch analysis over a directory of SoliTOC files."""
    input_dir = Path(params["input_dir"])
    output_dir = Path(params["output_dir"])
    batch_summary_name = (
        str(params.get("batch_summary_name", "batch_summary")).strip()
        or "batch_summary"
    )
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise ValueError(f"No TXT files found in {input_dir}")

    summary_rows = []
    thermogram_exports: list[tuple[str, pd.DataFrame]] = []
    for file_path in files:
        res = process_file(
            str(file_path),
            str(output_dir),
            time_col_index=int(params.get("time_col_index", 0)),
            temp_col_index=int(params.get("temp_col_index", 1)),
            co2_col_index=int(params.get("co2_col_index", -1)),
            flow_col_index=int(params.get("flow_col_index", DEFAULT_FLOW_COL_IDX)),
            baseline_mode=str(params.get("baseline_mode", "manual")),
            manual_baseline=tuple(
                params.get(
                    "manual_baseline", (DEFAULT_BASELINE_START, DEFAULT_BASELINE_END)
                )
            ),
            fixed_baseline=tuple(params.get("fixed_baseline", DEFAULT_FIXED_BASELINE)),
            auto_search_range=tuple(params.get("auto_search_range", (None, None))),
            auto_window_size=float(params.get("auto_window_size", DEFAULT_AUTO_WINDOW)),
            fixed_val=float(params.get("fixed_val", DEFAULT_FIXED_VAL)),
            shift=int(params.get("shift", DEFAULT_SHIFT)),
            factor=float(params.get("factor", DEFAULT_FACTOR)),
            interp_every=float(params.get("interp_every", DEFAULT_INTERP_EVERY)),
            max_energy_temp=float(
                params.get("max_energy_temp", DEFAULT_MAX_ENERGY_TEMP)
            ),
            zone_flow_relative_increase=float(
                params.get(
                    "zone_flow_relative_increase", DEFAULT_ZONE_FLOW_RELATIVE_INCREASE
                )
            ),
        )
        summary_rows.append(
            format_batch_summary_row(res, res.get("basefn", file_path.stem))
        )
        thermogram_df = res.get("thermogram_df")
        if isinstance(thermogram_df, pd.DataFrame):
            thermogram_exports.append(
                (res.get("basefn", file_path.stem), thermogram_df)
            )

    write_batch_exports(
        output_dir, batch_summary_name, summary_rows, thermogram_exports
    )


# The Tkinter GUI has been moved to app.py.
