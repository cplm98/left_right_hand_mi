from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np
from flask import Flask, abort, jsonify, render_template, request
from scipy.io import loadmat
from scipy.interpolate import griddata
from scipy.signal import welch
from lib import apply_basic_filters

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "Data"
MAX_POINTS = 4000  # limit for plotting performance

BAND_DEFINITIONS: Dict[str, Tuple[float, float]] = {
    "delta_band": (0.5, 4),
    "theta_band": (4, 8),
    "mu_band": (8, 13),
    "beta_band": (13, 30),
    "gamma_band": (30, 45),
}

ERD_DATASETS = {
    "movement_left": {
        "label": "Movement · Left",
        "event_key": "movement_event",
        "trial_key": "n_movement_trials",
    },
    "movement_right": {
        "label": "Movement · Right",
        "event_key": "movement_event",
        "trial_key": "n_movement_trials",
    },
    "imagery_left": {
        "label": "Imagery · Left",
        "event_key": "imagery_event",
        "trial_key": "n_imagery_trials",
    },
    "imagery_right": {
        "label": "Imagery · Right",
        "event_key": "imagery_event",
        "trial_key": "n_imagery_trials",
    },
}

DEFAULT_FILTERS = {
    "hp": 1.0,
    "lp": 40.0,
    "notch_base": 60.0,
    "n_harmonics": 2,
    "notch_q": 30.0,
    "order": 4,
}
DEFAULT_BASELINE_SAMPLES = 1000

app = Flask(__name__)


def list_mat_files(directory: Path) -> List[str]:
    """Return sorted list of .mat files relative to DATA_DIR."""
    if not directory.exists():
        return []
    return sorted(str(path.name) for path in directory.glob("*.mat"))


def load_eeg_struct(file_path: Path) -> MutableMapping[str, object]:
    """Load EEG struct from MAT file as a Python mapping."""
    mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    data_dict = {
        key: value for key, value in mat.items() if not key.startswith("__")
    }

    eeg_raw = data_dict.get("eeg")
    if eeg_raw is None:
        raise ValueError("Key 'eeg' not found in MAT file.")

    eeg_data = matlab_to_python(eeg_raw)
    if not isinstance(eeg_data, MutableMapping):
        raise ValueError("Unexpected structure for 'eeg' key.")
    return eeg_data


def matlab_to_python(value):
    """Recursively convert MATLAB structs/object arrays into Python types."""
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [matlab_to_python(element) for element in value]
        return value
    if hasattr(value, "_fieldnames"):
        return {
            field: matlab_to_python(getattr(value, field))
            for field in value._fieldnames  # type: ignore[attr-defined]
        }
    return value


def normalize_array(candidate: np.ndarray) -> np.ndarray:
    """Convert candidate to 2D array shaped (channels, samples)."""
    arr = np.array(candidate)

    # Remove singleton dimensions while preserving at least 2 dims.
    arr = np.atleast_2d(np.squeeze(arr))

    # If channels appear on last axis, transpose to channels-first.
    if arr.shape[0] < arr.shape[-1] and arr.shape[0] < 3 and arr.shape[-1] >= 3:
        arr = arr.T

    return arr


def extract_segments(eeg_data: Mapping[str, object]) -> Dict[str, np.ndarray]:
    """Collect candidate EEG segments keyed by descriptive names."""
    segments: Dict[str, np.ndarray] = {}

    for key, raw_value in eeg_data.items():
        value = matlab_to_python(raw_value)

        if isinstance(value, np.ndarray):
            arr = normalize_array(value)
            if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 32:
                segments[key] = arr
        elif isinstance(value, (list, tuple)):
            for idx, element in enumerate(value):
                if element is None:
                    continue
                arr = normalize_array(np.asarray(element))
                if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 32:
                    segments[f"{key}_{idx}"] = arr

    return segments


def rank_segments(segments: Mapping[str, np.ndarray]) -> List[str]:
    """Order segments using a preferred list first, then remaining alphabetically."""
    preferred = [
        "rest",
        "movement_left",
        "movement_right",
        "imagery_left",
        "imagery_right",
    ]
    remaining = [key for key in segments if key not in preferred]
    ordered = [key for key in preferred if key in segments]
    ordered.extend(sorted(remaining))
    return ordered


def downsample(channels: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample channels evenly if they exceed max_points."""
    samples = channels.shape[1]
    if samples <= max_points:
        indices = np.arange(samples)
        return channels, indices

    step = max(1, samples // max_points)
    indices = np.arange(0, samples, step)
    return channels[:, indices], indices


def load_channels(
    file_path: Path, segment: str | None = None, channel_count: int | None = None
) -> Dict[str, Iterable]:
    """Load channels from selected segment; default returns all available channels."""
    print(f"[load_channels] file={file_path.name} requested_segment={segment} limit={channel_count}")
    eeg_data = load_eeg_struct(file_path)

    segments = extract_segments(eeg_data)
    if not segments:
        raise ValueError("No 2D array with at least three channels found.")

    ordered_segments = rank_segments(segments)
    print(
        f"[load_channels] available_segments={ordered_segments[:10]}"
        + ("..." if len(ordered_segments) > 10 else "")
    )

    if segment and segment not in segments:
        print(f"[load_channels] segment '{segment}' not found, falling back.")
        segment = None

    target_segment = segment or (
        "movement_left"
        if "movement_left" in segments
        else "rest"
        if "rest" in segments
        else ordered_segments[0]
    )
    if target_segment not in segments:
        raise ValueError(f"Segment '{target_segment}' not available.")

    segment_matrix = segments[target_segment]
    available_channels = segment_matrix.shape[0]
    channels = segment_matrix
    if channel_count is not None:
        channels = segment_matrix[:channel_count]

    channel_total = channels.shape[0]
    print(
        f"[load_channels] using_segment='{target_segment}' "
        f"channels_shape={channels.shape} available_channels={available_channels}"
    )

    reduced, indices = downsample(channels, MAX_POINTS)
    time_axis = indices.tolist()
    if indices.size:
        print(
            f"[load_channels] reduced_shape={reduced.shape} "
            f"indices_span=({indices[0]}, {indices[-1]})"
        )
    else:
        print("[load_channels] reduced empty selection")

    srate_value = eeg_data.get("srate")
    sample_rate = int(srate_value) if isinstance(srate_value, (int, np.integer)) else None

    if sample_rate:
        print(f"[load_channels] sample_rate={sample_rate}")

    return {
        "file": file_path.name,
        "segments": ordered_segments,
        "segment": target_segment,
        "sampleRate": sample_rate,
        "channelCount": channel_total,
        "availableChannels": available_channels,
        "time": time_axis,
        "channels": [
            {
                "name": f"Channel {idx + 1}",
                "values": channel.tolist(),
            }
            for idx, channel in enumerate(reduced)
        ],
    }


def compute_bandpowers(signal: np.ndarray, fs: int) -> Dict[str, float]:
    """Compute band powers for a 1D signal using Welch PSD."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        signal = signal.ravel()
    if signal.size < 8 or fs <= 0:
        return {name: float("nan") for name in BAND_DEFINITIONS}

    nperseg = min(signal.size, max(fs * 2, 64))
    freqs, pxx = welch(signal, fs=fs, nperseg=nperseg)

    bandpowers: Dict[str, float] = {}
    for name, (low, high) in BAND_DEFINITIONS.items():
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            bandpowers[name] = float("nan")
            continue
        bandpowers[name] = float(np.trapezoid(pxx[mask], freqs[mask]))

    return bandpowers


def sanitize_number(value: float | None) -> float | None:
    """Return value if finite, else None."""
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def normalize_psenloc(psenloc: object) -> np.ndarray | None:
    """Return normalized 2D electrode coordinates from psenloc."""
    try:
        arr = np.asarray(psenloc, dtype=float)
    except Exception:
        return None

    if arr.ndim < 2 or arr.shape[0] < 64 or arr.shape[1] < 2:
        return None

    xy = arr[:64, :2].copy()
    xy -= np.nanmean(xy, axis=0)
    max_abs = np.nanmax(np.abs(xy))
    if not np.isfinite(max_abs) or max_abs <= 0:
        return None
    xy /= max_abs
    return xy


def matrix_to_serializable(matrix: np.ndarray) -> List[List[float | None]]:
    """Convert numeric matrix to lists, mapping NaNs to None."""
    rows: List[List[float | None]] = []
    for row in matrix:
        rows.append(
            [float(val) if np.isfinite(val) else None for val in row]
        )
    return rows


def compute_topomap_grid(
    xy: np.ndarray,
    values: np.ndarray,
    grid_size: int = 180,
    label: str = "",
) -> Dict[str, object] | None:
    """Interpolate ERD values onto a dense grid for visualization."""
    finite_mask = np.isfinite(values)
    if finite_mask.sum() < 3:
        if label:
            print(f"[compute_topomap_grid] {label}: insufficient finite channels ({finite_mask.sum()})")
        return None

    x = xy[:, 0]
    y = xy[:, 1]
    radius = float(np.sqrt(x ** 2 + y ** 2).max())
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    padding = 0.15
    xmin, xmax = x.min() - padding, x.max() + padding
    ymin, ymax = y.min() - padding, y.max() + padding

    grid_x, grid_y = np.mgrid[
        xmin:xmax:complex(0, grid_size),
        ymin:ymax:complex(0, grid_size),
    ]

    try:
        grid_z = griddata(
            (x[finite_mask], y[finite_mask]),
            values[finite_mask],
            (grid_x, grid_y),
            method="cubic",
        )
    except Exception:
        grid_z = griddata(
            (x[finite_mask], y[finite_mask]),
            values[finite_mask],
            (grid_x, grid_y),
            method="linear",
        )

    if grid_z is None:
        if label:
            print(f"[compute_topomap_grid] {label}: griddata returned None")
        return None

    mask = (grid_x ** 2 + grid_y ** 2) <= (radius * 1.05) ** 2
    grid_z = np.where(mask, grid_z, np.nan)

    if not np.isfinite(grid_z).any():
        if label:
            print(f"[compute_topomap_grid] {label}: interpolated grid contains no finite values")
        return None

    return {
        "x": grid_x[:, 0].tolist(),
        "y": grid_y[0, :].tolist(),
        "z": matrix_to_serializable(grid_z.T),
        "radius": float(radius * 1.05),
    }


def parse_optional_float_arg(name: str) -> float | None:
    raw = request.args.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name!r}") from exc


def parse_optional_int_arg(name: str) -> int | None:
    raw = request.args.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name!r}") from exc


def compute_epoch_erd(
    file_path: Path,
    dataset_key: str,
    epoch_index: int,
    band_name: str,
    filter_options: Dict[str, float] | None = None,
    baseline_samples: int = 1000,
) -> Dict[str, object]:
    """Compute ERD for a given epoch, dataset, and band."""
    if dataset_key not in ERD_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_key}'.")
    if band_name not in BAND_DEFINITIONS:
        raise ValueError(f"Unknown band '{band_name}'.")

    eeg_data = load_eeg_struct(file_path)
    cfg = ERD_DATASETS[dataset_key]

    segment = eeg_data.get(dataset_key)
    if segment is None:
        raise ValueError(f"Segment '{dataset_key}' not found in file.")

    data_matrix = np.asarray(segment, dtype=float)
    if data_matrix.ndim != 2:
        data_matrix = np.atleast_2d(data_matrix)

    eeg_channels = data_matrix[:64, :]

    fs_value = eeg_data.get("srate")
    if not isinstance(fs_value, (int, np.integer)):
        raise ValueError("Sample rate 'srate' missing or invalid.")
    fs = int(fs_value)

    trial_count_raw = eeg_data.get(cfg["trial_key"])
    if not isinstance(trial_count_raw, (int, np.integer)):
        raise ValueError(f"Trial count '{cfg['trial_key']}' missing or invalid.")
    trial_count = int(trial_count_raw)
    if trial_count <= 0:
        raise ValueError("Trial count must be positive.")

    total_samples = eeg_channels.shape[1]
    samples_per_trial = total_samples // trial_count
    if samples_per_trial * trial_count != total_samples:
        raise ValueError("Samples do not divide evenly into trials.")

    filter_opts = filter_options or {}
    hp_val = filter_opts.get("hp", DEFAULT_FILTERS["hp"])
    lp_val = filter_opts.get("lp", DEFAULT_FILTERS["lp"])
    notch_base = filter_opts.get("notch_base", DEFAULT_FILTERS["notch_base"])
    n_harmonics = filter_opts.get("n_harmonics", DEFAULT_FILTERS["n_harmonics"])
    notch_q = filter_opts.get("notch_q", DEFAULT_FILTERS["notch_q"])
    filter_order = max(1, int(filter_opts.get("order", DEFAULT_FILTERS["order"])))

    nyquist = fs / 2.0
    hp_cutoff = hp_val if hp_val and 0 < hp_val < nyquist else None
    lp_cutoff = lp_val if lp_val and 0 < lp_val < nyquist else None
    if hp_cutoff is not None and lp_cutoff is not None and lp_cutoff <= hp_cutoff:
        raise ValueError("Low-pass cutoff must be greater than high-pass cutoff.")

    notch_base_val = notch_base if notch_base and 0 < notch_base < nyquist else None
    notch_q_val = notch_q if notch_q and notch_q > 0 else 30.0
    n_harmonics_val = int(n_harmonics) if n_harmonics is not None else 0
    n_harmonics_val = max(0, n_harmonics_val)

    filters_used = {
        "hp": hp_cutoff,
        "lp": lp_cutoff,
        "notchBase": notch_base_val,
        "notchHarmonics": n_harmonics_val,
        "notchQ": notch_q_val if notch_base_val else None,
        "order": filter_order,
    }

    eeg_filtered = apply_basic_filters(
        eeg_channels,
        fs,
        hp=hp_cutoff,
        lp=lp_cutoff,
        notch_base=notch_base_val,
        n_harm=n_harmonics_val,
        q=notch_q_val,
        order=filter_order,
    )

    epochs = eeg_filtered.reshape(64, samples_per_trial, trial_count, order="F")
    print(
        "[compute_epoch_erd] epochs shape",
        epochs.shape,
        f"total_samples={total_samples}",
        f"samples_per_trial={samples_per_trial}",
        f"trial_count={trial_count}",
    )

    event_array = eeg_data.get(cfg["event_key"])
    if event_array is None:
        raise ValueError(f"Event array '{cfg['event_key']}' missing.")
    event_array = np.asarray(event_array)
    if event_array.size != total_samples:
        raise ValueError("Event array length does not match segment samples.")
    event_epochs = event_array.reshape(samples_per_trial, trial_count, order="F")

    epoch_index = int(np.clip(epoch_index, 0, trial_count - 1))
    epoch_signal = epochs[:, :, epoch_index]
    epoch_events = event_epochs[:, epoch_index]

    movement_indices = np.where(epoch_events > 0)[0]
    if movement_indices.size:
        active_start = int(movement_indices[0])
        active_end = int(movement_indices[-1]) + 1
    else:
        active_start = samples_per_trial // 2
        active_end = min(samples_per_trial, active_start + fs)

    active_start = max(1, min(active_start, samples_per_trial - 2))
    active_end = max(active_start + 1, min(active_end, samples_per_trial))

    baseline_len_request = int(max(1, baseline_samples))
    baseline_start = max(0, active_start - baseline_len_request)
    baseline_end = active_start

    baseline_signal = epoch_signal[:, baseline_start:baseline_end]
    if baseline_signal.shape[1] < max(8, fs // 4):
        baseline_start = max(0, baseline_end - max(fs // 2, baseline_len_request))
        baseline_signal = epoch_signal[:, baseline_start:baseline_end]
    if baseline_signal.shape[1] == 0:
        baseline_start = 0
        baseline_end = min(active_start, max(1, baseline_len_request))
        baseline_signal = epoch_signal[:, baseline_start:baseline_end]

    active_signal = epoch_signal[:, active_start:active_end]
    if active_signal.shape[1] < max(8, fs // 4):
        active_end = min(samples_per_trial, active_start + max(fs // 2, active_end - active_start))
        active_signal = epoch_signal[:, active_start:active_end]
    if active_signal.shape[1] == 0:
        active_end = min(samples_per_trial, active_start + max(fs // 2, 1))
        active_signal = epoch_signal[:, active_start:active_end]

    baseline_samples_actual = baseline_signal.shape[1]
    active_samples_actual = active_signal.shape[1]

    channels: List[Dict[str, object]] = []
    erd_values: List[float] = []

    for idx in range(epoch_signal.shape[0]):
        ch_baseline = baseline_signal[idx]
        ch_active = active_signal[idx]

        baseline_powers = compute_bandpowers(ch_baseline, fs)
        active_powers = compute_bandpowers(ch_active, fs)

        baseline_val = baseline_powers.get(band_name)
        active_val = active_powers.get(band_name)
        erd_val: float | None = None
        if (
            baseline_val is not None
            and active_val is not None
            and np.isfinite(baseline_val)
            and np.isfinite(active_val)
        ):
            denom = baseline_val if abs(baseline_val) > 1e-12 else np.sign(baseline_val) * 1e-12 or 1e-12
            erd_val = float((active_val - baseline_val) / denom)
        if erd_val is not None and np.isfinite(erd_val):
            erd_values.append(erd_val)

        channels.append(
            {
                "index": idx + 1,
                "baseline": {k: sanitize_number(v) for k, v in baseline_powers.items()},
                "active": {k: sanitize_number(v) for k, v in active_powers.items()},
                "erd": sanitize_number(erd_val),
            }
        )

    if erd_values:
        bound = float(max(abs(np.min(erd_values)), abs(np.max(erd_values))))
        bound = max(bound, 1e-6)
        erd_range = [-bound, bound]
    else:
        erd_range = [-1.0, 1.0]

    erd_preview = np.array(erd_values, dtype=float)
    if erd_preview.size:
        erd_preview = np.nan_to_num(erd_preview, nan=0.0)
        print(
            f"[compute_epoch_erd] ERD stats min={erd_preview.min():.4f} "
            f"max={erd_preview.max():.4f} mean={erd_preview.mean():.4f}"
        )
        print(
            "[compute_epoch_erd] ERD first10",
            np.round(erd_preview[:10], 4).tolist(),
        )
    else:
        print("[compute_epoch_erd] ERD has no finite values.")

    positions_xy = normalize_psenloc(eeg_data.get("psenloc"))
    positions_payload: List[Dict[str, object]] = []
    topomap_payload: Dict[str, object] | None = None
    if positions_xy is not None:
        erd_array = np.array(
            [
                channel["erd"] if channel["erd"] is not None else np.nan
                for channel in channels
            ],
            dtype=float,
        )
        topomap_payload = compute_topomap_grid(
            positions_xy,
            erd_array,
            label=f"{file_path.name}:{dataset_key}:{band_name}:epoch{epoch_index}",
        )
        limit = min(len(channels), positions_xy.shape[0])
        for idx in range(limit):
            channel = channels[idx]
            erd_val = channel["erd"]
            positions_payload.append(
                {
                    "index": channel["index"],
                    "x": float(positions_xy[idx, 0]),
                    "y": float(positions_xy[idx, 1]),
                    "erd": sanitize_number(erd_val),
                }
            )
    else:
        print(f"[compute_epoch_erd] {file_path.name}:{dataset_key}:{band_name}:epoch{epoch_index} missing psenloc data")

    print(
        "[compute_epoch_erd]",
        file_path.name,
        dataset_key,
        f"epoch={epoch_index + 1}/{trial_count}",
        f"band={band_name}",
        f"finite_channels={len(erd_values)}",
        f"baseline_samples={baseline_samples_actual}",
        f"active_samples={active_samples_actual}",
        f"topomap={'yes' if topomap_payload else 'no'}",
    )

    windows = {
        "baseline": {
            "startSample": baseline_start,
            "endSample": baseline_end,
            "samples": baseline_samples_actual,
            "startTime": baseline_start / fs,
            "endTime": baseline_end / fs,
        },
        "active": {
            "startSample": active_start,
            "endSample": active_end,
            "samples": active_samples_actual,
            "startTime": active_start / fs,
            "endTime": active_end / fs,
        },
    }

    return {
        "file": file_path.name,
        "dataset": dataset_key,
        "band": band_name,
        "epoch": epoch_index,
        "epochCount": trial_count,
        "sampleRate": fs,
        "samplesPerTrial": samples_per_trial,
        "windows": windows,
        "baseline": {
            "samples": baseline_samples_actual,
            "seconds": baseline_samples_actual / fs,
        },
        "filters": filters_used,
        "channels": channels,
        "erdRange": erd_range,
        "positions": positions_payload,
        "topomap": topomap_payload,
    }


@app.route("/")
def index():
    files = list_mat_files(DATA_DIR)
    return render_template("index.html", files=files)


@app.route("/erd")
def erd():
    files = list_mat_files(DATA_DIR)
    dataset_options = [
        {"value": key, "label": cfg["label"]} for key, cfg in ERD_DATASETS.items()
    ]
    band_options = list(BAND_DEFINITIONS.keys())
    filter_defaults = {
        "hp": DEFAULT_FILTERS["hp"],
        "lp": DEFAULT_FILTERS["lp"],
        "notchBase": DEFAULT_FILTERS["notch_base"],
        "notchHarmonics": DEFAULT_FILTERS["n_harmonics"],
        "notchQ": DEFAULT_FILTERS["notch_q"],
        "order": DEFAULT_FILTERS["order"],
        "baselineSamples": DEFAULT_BASELINE_SAMPLES,
    }
    config = {
        "files": files,
        "datasets": dataset_options,
        "bands": band_options,
        "filters": filter_defaults,
    }
    return render_template("erd.html", config=config)


@app.route("/data")
def get_data():
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing file parameter.")
    segment = request.args.get("segment")

    target_path = DATA_DIR / filename
    if not target_path.is_file():
        abort(404, f"{filename} not found.")

    try:
        payload = load_channels(target_path, segment=segment)
    except Exception as exc:  # pragma: no cover - surfaced to client
        abort(500, f"Failed to load {filename}: {exc}")

    return jsonify(payload)


@app.route("/erd/data")
def get_erd_data():
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing file parameter.")

    dataset_key = request.args.get("dataset")
    if not dataset_key:
        abort(400, "Missing dataset parameter.")

    band_name = request.args.get("band", "mu_band")

    epoch_param = request.args.get("epoch", "0")
    try:
        epoch_index = int(epoch_param)
    except ValueError:
        abort(400, "Invalid epoch parameter.")

    target_path = DATA_DIR / filename
    if not target_path.is_file():
        abort(404, f"{filename} not found.")

    try:
        hp = parse_optional_float_arg("hp")
        lp = parse_optional_float_arg("lp")
        notch_base = parse_optional_float_arg("notchBase")
        notch_q = parse_optional_float_arg("notchQ")
        n_harmonics = parse_optional_int_arg("notchHarmonics")
        filter_order = parse_optional_int_arg("order")
        baseline_samples_arg = parse_optional_int_arg("baselineSamples")
    except ValueError as exc:
        abort(400, str(exc))

    filter_options = {
        "hp": DEFAULT_FILTERS["hp"] if hp is None else hp,
        "lp": DEFAULT_FILTERS["lp"] if lp is None else lp,
        "notch_base": DEFAULT_FILTERS["notch_base"] if notch_base is None else notch_base,
        "n_harmonics": DEFAULT_FILTERS["n_harmonics"] if n_harmonics is None else n_harmonics,
        "notch_q": DEFAULT_FILTERS["notch_q"] if notch_q is None else notch_q,
        "order": DEFAULT_FILTERS["order"] if filter_order is None else filter_order,
    }
    baseline_samples = (
        DEFAULT_BASELINE_SAMPLES if baseline_samples_arg is None else baseline_samples_arg
    )

    try:
        payload = compute_epoch_erd(
            target_path,
            dataset_key,
            epoch_index,
            band_name,
            filter_options=filter_options,
            baseline_samples=baseline_samples,
        )
    except ValueError as exc:
        abort(400, str(exc))
    except Exception as exc:  # pragma: no cover - surfaced to client
        abort(500, f"Failed to compute ERD for {filename}: {exc}")

    return jsonify(payload)


if __name__ == "__main__":
    app.run(debug=True)
