from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np
from flask import Flask, abort, jsonify, render_template, request
from scipy.io import loadmat

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "Data"
MAX_POINTS = 4000  # limit for plotting performance

app = Flask(__name__)


def list_mat_files(directory: Path) -> List[str]:
    """Return sorted list of .mat files relative to DATA_DIR."""
    if not directory.exists():
        return []
    return sorted(str(path.name) for path in directory.glob("*.mat"))


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


def load_channels(file_path: Path, segment: str | None = None, channel_count: int = 3) -> Dict[str, Iterable]:
    """Load first `channel_count` channels from selected segment."""
    print(f"[load_channels] file={file_path.name} requested_segment={segment}")
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

    channels = segments[target_segment][:channel_count]
    print(
        f"[load_channels] using_segment='{target_segment}' "
        f"channels_shape={channels.shape}"
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
        "time": time_axis,
        "channels": [
            {
                "name": f"Channel {idx + 1}",
                "values": channel.tolist(),
            }
            for idx, channel in enumerate(reduced)
        ],
    }


@app.route("/")
def index():
    files = list_mat_files(DATA_DIR)
    return render_template("index.html", files=files)


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


if __name__ == "__main__":
    app.run(debug=True)
