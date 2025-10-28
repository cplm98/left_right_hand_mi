from scipy.io import loadmat
import numpy as np
from scipy.signal import welch

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

def make_band_masks(f, bands=BANDS):
    """
    Takes in a list of frequencies and a dictionary of the bands of interest.
    """
    names  = list(bands.keys())
    bounds = np.array([bands[name] for name in names])     # (n_bands, 2)
    # mask: (n_freq, n_bands)
    mask = (f[:, None] >= bounds[:, 0]) & (f[:, None] <= bounds[:, 1])
    return names, mask

def make_epochs(eeg, baseline_dur, sample_dur, data_labels=DATA_LABELS, n_trial_labels=N_TRIAL_LABELS):
    
    for data_label in data_labels:
        if "movement" in data_label:
            n_trials = eeg[n_trial_labels[0]] # this is a bit magic numbery but leaving for now.
        else:
            n_trials = eeg[n_trial_labels[1]] # this is a bit magic numbery but leaving for now.
        eeg_data = eeg[data_label][:64,:]
        aux_data = eeg[data_label][64:,:]
        samples_per_trial = eeg_data.shape[1] // n_trials
        eeg_data = np.reshape(eeg_data, (64, samples_per_trial, n_trials), order="F")
        aux_data = np.reshape(aux_data,(aux_data.shape[0], samples_per_trial, n_trials), order="F")
        for i in range(eeg_data.shape[0]): #iterate through every channel for every epoch
            for j in range(eeg_data.shape[2]):
                channel_data = eeg_data[i,:,j]
#######
# Left off here - need to finish the epoch making function for breaking into baseline and sample.
#######



def calculate_bandpowers(single_channel_data, masks):
    f, Pxx = welch(single_channel_data, fs=fs, nperseg=fs*2)
    # print("Pxx shape: ", Pxx.shape)
    band_pxx = np.where(masks, Pxx[:, None], 0.0)              # (n_freq, n_bands)
    bandpowers = np.trapezoid(band_pxx, during_f, axis=0)               # (n_bands,)
    bandpowers_dict = dict(zip(names, bandpowers))
    return bandpowers_dict

channel_bandpowers = {}

for i in range(movement_left_epochs.shape[0]):
    channel_data = movement_left_epochs[i,:,0] # working with all channels (i), for full epoch time series (:), but only the first epoch (0)
    # print(channel_data[:10])
    before = channel_data[:1024]
    during = channel_data[1024:]
    # print(before.shape)
    # print(during.shape)
    before_bandpowers = calculate_bandpowers(before, mask)
    during_bandpowers = calculate_bandpowers(during, mask)
    # print(before_bandpowers)
    # print(during_bandpowers)
    channel_bandpowers[str(i+1)] = {
        "before": before_bandpowers,
        "during": during_bandpowers
    }

