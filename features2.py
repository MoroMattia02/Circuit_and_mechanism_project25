import numpy as np
import pandas as pd
from scipy.signal import welch, correlate
from scipy.stats import skew, kurtosis


# --------------------------
# Utility: time → index
# --------------------------
def t2i(t, fs):
    return int(t * fs)


def normalize_intervals(x):
    """
    Convert x to a (n, 2) array of [start, end] intervals.
    Return None if x is NaN, empty, or malformed.
    """
    if x is None:
        return None

    if isinstance(x, float) or isinstance(x, np.floating):
        if np.isnan(x):
            return None

    arr = np.asarray(x)

    if arr.ndim == 0 or arr.size == 0:
        return None

    if arr.ndim == 1:
        return None

    if arr.shape[1] != 2:
        return None

    return arr


# --------------------------
# mean Vm over the given intervals, will be active, inactive or free whisking
# --------------------------

def mean_vm_over_intervals(vm, fs, intervals):
    if intervals is None or isinstance(intervals, float) and np.isnan(intervals):
        return np.nan

    arr = np.asarray(intervals)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.nan
    
    segments = []
    for start, end in arr:
        i1 = int(start * fs)
        i2 = int(end * fs)
        i1 = max(i1, 0)
        i2 = min(i2, len(vm))
        if i2 > i1:
            segments.append(vm[i1:i2])
    
    if not segments:
        return np.nan
    
    return np.concatenate(segments).mean()



#same but with slope, in those intervals
def mean_slope_pre_post(vm, fs, intervals, pre_win=0.05, post_win=0.05):
    # If intervals array is malformed or missing
    if intervals is None or isinstance(intervals, float) and np.isnan(intervals):
        return np.nan, np.nan
    
    arr = np.asarray(intervals)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.nan, np.nan

    start_t, end_t = arr[0]     # first interval
    start_i = int(start_t * fs)
    end_i   = int(end_t * fs)

    dvm = np.diff(vm) * fs      # slope estimate
    
    pre_samp = int(pre_win * fs)
    post_samp = int(post_win * fs)

    # Pre-event slope
    pre_start = max(0, start_i - pre_samp)
    pre_end   = max(0, start_i)
    pre_slope = dvm[pre_start:pre_end].mean() if pre_end > pre_start else np.nan

    # Post-event slope
    post_start = min(len(dvm), end_i)
    post_end   = min(len(dvm), end_i + post_samp)
    post_slope = dvm[post_start:post_end].mean() if post_end > post_start else np.nan

    return pre_slope, post_slope



# --------------------------
# FFT-based features
# --------------------------
def compute_fft_features(vm, fs):
    """Return low-frequency power, high-frequency power, and spectral entropy."""
    freqs, psd = welch(vm, fs, nperseg=4096)

    low_power = psd[(freqs >= 1) & (freqs <= 10)].mean()
    high_power = psd[(freqs >= 30) & (freqs <= 90)].mean()

    psd_norm = psd / psd.sum()
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))

    return low_power, high_power, spectral_entropy


# --------------------------
# Autocorrelation decay
# --------------------------
def compute_autocorr_decay(vm):
    """Return the lag where the autocorrelation first drops below 1/e."""
    vm = vm - vm.mean()
    ac = correlate(vm, vm, mode="full")
    ac = ac[len(ac)//2:]  # positive lags
    ac = ac / ac[0]

    below = np.where(ac < 1/np.e)[0]
    return below[0] if len(below) > 0 else len(ac)


# --------------------------
# Rate of sharp Vm bumps
# --------------------------
def compute_bump_rate(vm, threshold=0.5):
    diffs = np.diff(vm)
    return np.sum(diffs > threshold) / len(vm)


# --------------------------
# Whisking-onset Vm feature
# --------------------------
def compute_whisk_features(vm, fs, whisk_times):
    wt = normalize_intervals(whisk_times)
    if wt is None:
        return np.nan

    onset = float(wt[0, 0])
    onset_i = t2i(onset, fs)
    win = int(0.1 * fs)

    if onset_i <= win or onset_i + win >= len(vm):
        return np.nan

    baseline = vm[onset_i - win : onset_i].mean()
    response = vm[onset_i : onset_i + win].mean()
    return response - baseline



# --------------------------
# Touch-onset Vm feature
# --------------------------
def compute_touch_features(vm, fs, contact_times):
    ct = normalize_intervals(contact_times)
    if ct is None:
        return np.nan

    onset = float(ct[0, 0])
    onset_i = t2i(onset, fs)
    win = int(0.1 * fs)

    if onset_i <= win or onset_i + win >= len(vm):
        return np.nan

    baseline = vm[onset_i - win : onset_i].mean()
    response = vm[onset_i : onset_i + win].mean()
    return response - baseline



# --------------------------
# Main feature extraction
# --------------------------
def extract_features(row):
    """
    Extract interpretable features from a single sweep:
    - statistical features
    - FFT features
    - autocorrelation features
    - whisking onset Vm change
    - touch onset Vm change
    """

    vm = np.array(row["Sweep_MembranePotential"])
    fs = float(row["Sweep_MembranePotential_SamplingRate"])

    whisk_times = row["Sweep_WhiskingTimes"]
    contact_times = row["Sweep_ActiveContactTimes"]
    quiet_times   = row["Sweep_QuietTimes"]

    mean_vm_quiet = mean_vm_over_intervals(vm, fs, quiet_times)
    mean_vm_whisk = mean_vm_over_intervals(vm, fs, whisk_times)
    mean_vm_touch = mean_vm_over_intervals(vm, fs, contact_times)

    whisk_slope_pre, whisk_slope_post = mean_slope_pre_post(vm, fs, whisk_times)
    touch_slope_pre, touch_slope_post = mean_slope_pre_post(vm, fs, contact_times)
    # Basic statistics
    mean_vm = vm.mean()
    std_vm = vm.std()
    skew_vm = skew(vm)
    kurt_vm = kurtosis(vm)

    # Frequency domain
    low_power, high_power, spectral_entropy = compute_fft_features(vm, fs)

    # Temporal dynamics
    dvm_std = np.diff(vm).std()
    autocorr_decay = compute_autocorr_decay(vm)
    bump_rate = compute_bump_rate(vm)

    # Event-triggered features
    whisk_vm_change = compute_whisk_features(vm, fs, whisk_times)
    touch_vm_change = compute_touch_features(vm, fs, contact_times)

    return pd.Series({
        "mean_vm": mean_vm,
        "std_vm": std_vm,
        "skew_vm": skew_vm,
        "kurt_vm": kurt_vm,
        "low_power": low_power,
        "high_power": high_power,
        "spectral_entropy": spectral_entropy,
        "dvm_std": dvm_std,
        "autocorr_decay": autocorr_decay,
        "bump_rate": bump_rate,
        "whisk_vm_change": whisk_vm_change,
        "touch_vm_change": touch_vm_change,
        "mean_vm_quiet": mean_vm_quiet,
        "mean_vm_whisk": mean_vm_whisk,
        "mean_vm_touch": mean_vm_touch,
        "whisk_slope_pre": whisk_slope_pre,
        "whisk_slope_post": whisk_slope_post,
        "touch_slope_pre": touch_slope_pre,
        "touch_slope_post": touch_slope_post
    })


# --------------------------
# Apply to full dataframe
# --------------------------
def build_feature_dataset(df):
    print("Extracting features for", len(df), "sweeps...")
    features_df = df.apply(extract_features, axis=1)

    # attach labels for downstream models
    features_df["cell_type"] = df["Cell_Type"]
    features_df["cell_layer"] = df["Cell_Layer"]
    features_df["cell_depth"] = df["Cell_Depth"]

    return features_df


# Example usage (uncomment in your script or notebook):
# df = pd.read_pickle("your_dataframe.pkl")
# features = build_feature_dataset(df)
# print(features.info())
# features.to_csv("extracted_features.csv", index=False)
