from scipy.io import loadmat
import numpy as np
from scipy.signal import welch, butter, filtfilt, iirnotch

# Pre-decided trial names containing time series data.
DATA_LABELS = [
    "movement_left",
    "movement_right",
    "imagery_left",
    "imagery_right",
]

N_TRIAL_LABELS = [
    "n_movement_trials",
    "n_imagery_trials"
]

# Key frequency bands for brain activity.
BANDS = {
    "delta_band": (.5, 4),
    "theta_band": (4, 8),
    "mu_band": (8, 13),
    "beta_band": (13, 30),
    "gamma_band": (30, 45)
}

FS = 512

def make_band_masks(f, bands=BANDS):
    """
    Takes in a list of frequencies and a dictionary of the bands of interest.
    """
    names  = list(bands.keys())
    bounds = np.array([bands[name] for name in names])     # (n_bands, 2)
    # mask: (n_freq, n_bands)
    mask = (f[:, None] >= bounds[:, 0]) & (f[:, None] <= bounds[:, 1])
    return names, mask

def make_epochs(eeg, data_labels=DATA_LABELS, n_trial_labels=N_TRIAL_LABELS):
    eeg_epochs = {}
    for data_label in data_labels:
        if "movement" in data_label:
            n_trials = eeg[n_trial_labels[0]] # this is a bit magic numbery but leaving for now.
        else: #Imagery
            n_trials = eeg[n_trial_labels[1]] # this is a bit magic numbery but leaving for now.
        eeg_data = eeg[data_label][:64,:]
        aux_data = eeg[data_label][64:,:]
        samples_per_trial = eeg_data.shape[1] // n_trials
        eeg_data = np.reshape(eeg_data, (64, samples_per_trial, n_trials), order="F")
        aux_data = np.reshape(aux_data,(aux_data.shape[0], samples_per_trial, n_trials), order="F")
        eeg_epochs[data_label] = {
            "eeg_data" : eeg_data,
            "aux_data" : aux_data
        }
    return eeg_epochs

def calculate_bandpowers(single_channel_data, fs=FS):
    f, Pxx = welch(single_channel_data, fs=fs, nperseg=fs*2)
    band_names, band_masks = make_band_masks(f) # don't love this but it will work for now.
    band_pxx = np.where(band_masks, Pxx[:, None], 0.0)              # (n_freq, n_bands)
    bandpowers = np.trapezoid(band_pxx, f, axis=0)               # (n_bands,)
    bandpowers_dict = dict(zip(band_names, bandpowers))
    return f, Pxx, bandpowers_dict


# ---------- Filtering utilities ----------

def butter_filter(data, fs, hp=None, lp=None, order=4):
    """
    Zero-phase Butterworth filtering with optional high- and/or low-pass.
    data: array-like (..., samples)
    fs: sampling rate (Hz)
    hp: high-pass cutoff in Hz (None to skip)
    lp: low-pass cutoff in Hz (None to skip)
    order: filter order (per section)
    """
    x = np.asarray(data, dtype=float, copy=True)
    if x.size == 0:
        return x

    nyq = fs / 2.0
    if hp is not None:
        if not (0 < hp < nyq):
            raise ValueError(f"high-pass cutoff must satisfy 0 < hp < {nyq}")
        b, a = butter(order, hp / nyq, btype="highpass")
        x = filtfilt(b, a, x, axis=-1)
    if lp is not None:
        if not (0 < lp < nyq):
            raise ValueError(f"low-pass cutoff must satisfy 0 < lp < {nyq}")
        b, a = butter(order, lp / nyq, btype="lowpass")
        x = filtfilt(b, a, x, axis=-1)
    return x


def butter_bandpass(data, fs, f_lo, f_hi, order=4):
    """Zero-phase Butterworth band-pass (f_lo < f_hi)."""
    if not (0 < f_lo < f_hi < fs / 2):
        raise ValueError("Cutoffs must satisfy 0 < f_lo < f_hi < fs/2")
    x = np.asarray(data, dtype=float, copy=True)
    nyq = fs / 2.0
    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype="bandpass")
    return filtfilt(b, a, x, axis=-1)


def notch_60hz(data, fs, q=30.0):
    """
    Single-frequency IIR notch at 60 Hz (US mains), zero-phase.
    q: quality factor (higher => narrower notch). Typical 20–50.
    """
    b, a = iirnotch(w0=60.0, Q=q, fs=fs)
    return filtfilt(b, a, np.asarray(data, dtype=float, copy=True), axis=-1)


def notch_harmonics(data, fs, base=60.0, n_harmonics=3, q=30.0, nyquist_guard=0.9):
    """
    Apply notches at base, 2*base, ..., up to n_harmonics.
    Skips any harmonic >= nyquist_guard * (fs/2) to avoid instability.
    """
    x = np.asarray(data, dtype=float, copy=True)
    if x.size == 0:
        return x

    nyq = fs / 2.0
    for k in range(1, n_harmonics + 1):
        f0 = k * base
        if f0 < nyquist_guard * nyq:
            b, a = iirnotch(w0=f0, Q=q, fs=fs)
            x = filtfilt(b, a, x, axis=-1)
    return x


def apply_basic_filters(
    data,
    fs,
    hp=1.0,
    lp=40.0,
    notch_base=60.0,
    n_harm=2,
    q=30.0,
    order=4,
):
    """
    Typical order: High-pass -> Notch(es) -> Low-pass (all zero-phase).
    Returns filtered array with the same shape as input.
    """
    x = np.asarray(data, dtype=float, copy=True)
    if x.size == 0:
        return x

    # High-pass (optional)
    hp_cutoff = hp if hp is not None and hp > 0 else None
    lp_cutoff = lp if lp is not None and lp > 0 else None
    notch_base_val = notch_base if notch_base is not None and notch_base > 0 else None
    n_harmonics = int(n_harm) if n_harm is not None else 0

    if hp_cutoff is not None:
        x = butter_filter(x, fs, hp=hp_cutoff, lp=None, order=order)

    if notch_base_val is not None and n_harmonics > 0:
        x = notch_harmonics(
            x,
            fs,
            base=notch_base_val,
            n_harmonics=n_harmonics,
            q=q,
        )

    if lp_cutoff is not None:
        x = butter_filter(x, fs, hp=None, lp=lp_cutoff, order=order)

    return x

