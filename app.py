from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np
from flask import Flask, abort, jsonify, render_template, request
from scipy.io import loadmat
from scipy.interpolate import griddata
from scipy.signal import welch

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


def compute_epoch_erd(
    file_path: Path,
    dataset_key: str,
    epoch_index: int,
    band_name: str,
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

    fs = eeg_data.get("srate")
    if not isinstance(fs, (int, np.integer)):
        raise ValueError("Sample rate 'srate' missing or invalid.")
    fs = int(fs)

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

    epochs = eeg_channels.reshape(64, samples_per_trial, trial_count, order="F")
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
        start = int(movement_indices[0])
        end = int(movement_indices[-1]) + 1
    else:
        start = samples_per_trial // 2
        end = samples_per_trial

    start = max(1, min(start, samples_per_trial - 2))
    end = max(start + 1, min(end, samples_per_trial))

    before_signal = epoch_signal[:, :start]
    during_signal = epoch_signal[:, start:end]

    if before_signal.shape[1] < max(8, fs // 4):
        before_signal = epoch_signal[:, :max(start, min(samples_per_trial, fs))]
    if during_signal.shape[1] < max(8, fs // 4):
        during_signal = epoch_signal[:, start:]

    channels: List[Dict[str, object]] = []
    erd_values: List[float] = []

    for idx in range(epoch_signal.shape[0]):
        ch_before = before_signal[idx]
        ch_during = during_signal[idx]

        before_powers = compute_bandpowers(ch_before, fs)
        during_powers = compute_bandpowers(ch_during, fs)

        before_val = before_powers.get(band_name)
        during_val = during_powers.get(band_name)
        erd_val: float | None = None
        if (
            before_val is not None
            and during_val is not None
            and np.isfinite(before_val)
            and np.isfinite(during_val)
        ):
            denom = before_val if abs(before_val) > 1e-12 else np.sign(before_val) * 1e-12 or 1e-12
            erd_val = float((during_val - before_val) / denom)
        if erd_val is not None and np.isfinite(erd_val):
            erd_values.append(erd_val)

        channels.append(
            {
                "index": idx + 1,
                "before": {k: sanitize_number(v) for k, v in before_powers.items()},
                "during": {k: sanitize_number(v) for k, v in during_powers.items()},
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
        f"topomap={'yes' if topomap_payload else 'no'}",
    )

    return {
        "file": file_path.name,
        "dataset": dataset_key,
        "band": band_name,
        "epoch": epoch_index,
        "epochCount": trial_count,
        "sampleRate": fs,
        "samplesPerTrial": samples_per_trial,
        "movementWindow": {
            "startSample": start,
            "endSample": end,
            "startTime": start / fs,
            "endTime": end / fs,
        },
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
    config = {"files": files, "datasets": dataset_options, "bands": band_options}
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
        payload = compute_epoch_erd(target_path, dataset_key, epoch_index, band_name)
    except ValueError as exc:
        abort(400, str(exc))
    except Exception as exc:  # pragma: no cover - surfaced to client
        abort(500, f"Failed to compute ERD for {filename}: {exc}")

    return jsonify(payload)


if __name__ == "__main__":
    app.run(debug=True)
