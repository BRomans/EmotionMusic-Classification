import numpy as np
from mbt_pyspt.models.mybraineegdata import MyBrainEEGData
from mbt_pyspt.modules.preprocessingflow import PreprocessingFlow
from meegkit.asr import ASR
from meegkit.utils.matrix import sliding_window
from scipy import sparse
from scipy.sparse.linalg import spsolve

#  Resample the annotations and stretch them over 60 seconds depending on the amount of data points


def preprocess_em_participant(data, list_pp, asr_cleaning=False):
    baseline_eo = data['trials']['enter/resting_EO']['eeg']
    baseline_ec = data['trials']['enter/resting_EC']['eeg']
    sampling_rate = data['sampRate']
    channel_locations = data['acquisitionLocation']
    raw_b_eo = MyBrainEEGData(baseline_eo, sampling_rate, channel_locations)
    raw_b_ec = MyBrainEEGData(baseline_ec, sampling_rate, channel_locations)
    ppflow_b_eo = PreprocessingFlow(eeg_data=raw_b_eo, preprocessing_list=list_pp)
    ppflow_b_ec = PreprocessingFlow(eeg_data=raw_b_ec, preprocessing_list=list_pp)
    prep_b_eo = ppflow_b_eo()
    prep_b_ec = ppflow_b_ec()
    for trial in data['trials']:
        data['trials'][trial]['prep_eeg'] = preprocess_trial(raw_eeg=data['trials'][trial]['eeg'],
                                                             list_pp=list_pp,
                                                             loc=channel_locations,
                                                             sr=sampling_rate,
                                                             asr_cleaning=asr_cleaning,
                                                             asr_baseline=prep_b_ec)
    return data


def preprocess_trial(raw_eeg, list_pp, loc, sr=250, asr_cleaning=False, asr_baseline=None):
    raw = MyBrainEEGData(raw_eeg, sr, loc)
    ppflow = PreprocessingFlow(eeg_data=raw, preprocessing_list=list_pp)
    preprocessed = ppflow()
    if asr_cleaning:
        clean_preprocessed = compute_asr_reconstruction(preprocessed,
                                                        train_duration=60,
                                                        train_baseline=asr_baseline,
                                                        sfreq=sr,
                                                        win_len=0.5,
                                                        win_overlap=0.25)
        preprocessed = MyBrainEEGData(clean_preprocessed, sr, loc)
    return preprocessed


def compute_asr_reconstruction(eeg, train_duration=10, train_baseline=None, sfreq=250, win_len=0.5, win_overlap=0.66):
    # If no baseline is provided, use the a portion of the eeg signal itself
    if train_baseline is None:
        train_baseline = eeg

    # Train on a clean portion of data
    asr = ASR(method='euclid', win_len=win_len, win_overlap=win_overlap)
    train_idx = np.arange(0 * sfreq, train_duration * sfreq, dtype=int)
    _, sample_mask = asr.fit(train_baseline.matrix_data[:, train_idx])

    # Apply filter using sliding (non-overlapping) windows
    X = sliding_window(eeg.matrix_data, window=int(sfreq), step=int(sfreq))
    Y = np.zeros_like(X)
    for i in range(X.shape[1]):
        Y[:, i, :] = asr.transform(X[:, i, :])

    # raw_data = X.reshape(2, -1)  # reshape to (n_chans, n_times)
    clean_data = Y.reshape(2, -1)
    return clean_data


def baseline_als_optimized(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def participant_avg_annotation_windows(data, n_windows):
    for trial in data['trials']:
        if trial.startswith('EO'):
            annotations = data['trials'][trial]['annotations']
            avg_valence, avg_arousal = compute_avg_annotation_windows(annotations, n_windows)
            data['trials'][trial]['annotations']['avg_x'] = avg_valence
            data['trials'][trial]['annotations']['avg_y'] = avg_arousal
    return data


def compute_avg_annotation_windows(annotations, n_windows):
    windowed_v = np.array_split(annotations['x'], n_windows)
    avg_valence = []
    for window in windowed_v:
        avg_window = np.mean(window)
        avg_valence.append(avg_window)

    windowed_a = np.array_split(annotations['y'], n_windows)
    avg_arousal = []
    for window in windowed_a:
        avg_window = np.mean(window)
        avg_arousal.append(avg_window)

    return avg_valence, avg_arousal
