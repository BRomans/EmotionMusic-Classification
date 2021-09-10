import numpy as np
import copy
from statistics import mean
from mbt_pyspt.models.mybraineegdata import MyBrainEEGData
from mbt_pyspt.modules.preprocessingflow import PreprocessingFlow
from mbt_pyspt.modules.featuresextractionflow import FeaturesExtractionFlow
from meegkit.asr import ASR
from meegkit.utils.matrix import sliding_window
from scipy import sparse
from scipy.sparse.linalg import spsolve


#  Resample the annotations and stretch them over 60 seconds depending on the amount of data points


def preprocess_em_participant(data, list_pp, bpass_freqs=None, notch_freqs=None, asr_cleaning=False):
    baseline_ec = data['trials']['enter/resting_EC']['eeg']
    sampling_rate = data['sampRate']
    channel_locations = data['acquisitionLocation']
    prep_b_ec = preprocess_trial(raw_eeg=baseline_ec,
                                 list_pp=list_pp,
                                 loc=channel_locations,
                                 sr=sampling_rate,
                                 bpass_freqs=bpass_freqs,
                                 notch_freqs=notch_freqs,
                                 asr_cleaning=asr_cleaning)
    data['trials']['enter/resting_EC']['prep_eeg'] = prep_b_ec
    for trial in data['trials']:
        if trial.startswith('EO') or trial.startswith('EC'):
            data['trials'][trial]['prep_eeg'] = preprocess_trial(raw_eeg=data['trials'][trial]['eeg'],
                                                                 list_pp=list_pp,
                                                                 loc=channel_locations,
                                                                 sr=sampling_rate,
                                                                 bpass_freqs=bpass_freqs,
                                                                 notch_freqs=notch_freqs,
                                                                 asr_cleaning=asr_cleaning,
                                                                 asr_baseline=prep_b_ec)
    return data


def preprocess_trial(raw_eeg, list_pp, loc, sr=250, bpass_freqs=None, notch_freqs=None, asr_cleaning=False,
                     asr_baseline=None):
    if bpass_freqs is None:
        bpass_freqs = {
            'l_freq': 0.1,
            'h_freq': 30
        }
    if notch_freqs is None:
        notch_freqs = (50, 100)

    raw = MyBrainEEGData(raw_eeg, sr, loc)
    raw.mne_data.notch_filter(freqs=notch_freqs)
    # eeg_data.mne_data.resample(sfreq=64)
    raw.mne_data.filter(l_freq=bpass_freqs['l_freq'], h_freq=bpass_freqs['h_freq'])
    ppflow = PreprocessingFlow(eeg_data=raw, preprocessing_list=list_pp)
    preprocessed = ppflow()
    if asr_cleaning:
        clean_preprocessed = compute_asr_reconstruction(preprocessed.matrix_data,
                                                        train_duration=60,
                                                        train_baseline=asr_baseline,
                                                        sfreq=sr,
                                                        win_len=0.5,
                                                        win_overlap=0.25)
        return clean_preprocessed
    else:
        return preprocessed.matrix_data


def compute_participant_features(data, ff_list, split_data, sr, loc, cleaned_eeg=False, skip_qc=True):
    """ Retrieve the alpha, beta and theta power in specified time windows to calculate the neuromarkers
    Normalization: should I normalize? while extracting frequency bands or after computing the neuromarkers?

    Alpha Power: increasing/decreasing frontal alpha power can correlate with increasing/decreasing arousal

    SASI index:increases for Negative emotions and decreases for
    positive emotions  https://pubmed.ncbi.nlm.nih.gov/26738175/

    AW Index: alphaAF4 - alphaAF3. Left hemisphere (AF3) for Positive Valence, right hemisphere (AF4) for Negative Valence.
            If AWIdx is positive, there is a right tendency and NV. if AWIdx is negative, there is left tendency and PV.
            What if vaues are negatives? Should we subtract absolute values?

    FMT Index: mean Theta power stimulus/ mean Theta power baseline for Fz channel. It follows approach-withdrawal tendencies
            similarly to AWIndex but correlates more with the appreciation. Should we investigate correlation with liking?

    FFT-based or wavelet based features:
    """
    trials = data['trials']
    eeg_label = 'prep_eeg'
    if cleaned_eeg:
        eeg_label = 'clean_eeg'
    for trial in trials:
        if trial.startswith('EO') or trial.startswith('EC'):
            n_win = trials[trial]['c_windows']
            if not trials[trial]['bad_quality'] or skip_qc:
                eeg = MyBrainEEGData(trials[trial][eeg_label], sr, loc)
                extraction = FeaturesExtractionFlow(eeg, features_list=ff_list, split_data=split_data)
                features, labels = extraction()
                alpha_powers = np.array([features[0][0:n_win], features[1][0:n_win]])
                # alpha_labels = labels[0:n_win]
                print("Alpha Powers", alpha_powers)
                theta_powers = np.array([features[0][n_win: n_win * 2], features[1][n_win: n_win * 2]])
                # theta_labels = labels[n_win: n_win*2]
                # print("Theta", theta_powers, theta_labels)
                beta_powers = np.array([features[0][n_win * 2: n_win * 3], features[1][n_win * 2: n_win * 3]])
                # beta_labels = labels[n_win*2: n_win*3]
                # print("Beta", beta_powers, beta_labels)

                aw_idx = np.subtract(alpha_powers[0], alpha_powers[1])  # alpha AF4 - alpha AF3
                print("AWI", aw_idx)

                fmt_idx = np.mean(theta_powers, axis=0)  # mean(theta AF4, theta AF3) element-wise
                print("FMT", fmt_idx)

                sasi_idx = np.array([
                    np.divide(np.subtract(beta_powers[0], theta_powers[0]), np.add(beta_powers[0], theta_powers[0])),
                    np.divide(np.subtract(beta_powers[1], theta_powers[1]), np.add(beta_powers[1], theta_powers[1]))
                ])  # (beta - theta / beta + theta)AF4, (beta - theta / beta + theta)AF3
                print("SASI", sasi_idx)

                if "features" not in data['trials'][trial]:
                    trials[trial]['features'] = dict()
                trials[trial]['features']['alpha_pow'] = alpha_powers
                trials[trial]['features']['aw_idx'] = aw_idx
                trials[trial]['features']['sasi_idx'] = sasi_idx
                trials[trial]['features']['fmt_idx'] = fmt_idx
                trials[trial]['features']['familiarity'] = trials[trial]['annotations']['familiarity']
                trials[trial]['features']['liking'] = trials[trial]['annotations']['liking']


def copy_annotations_to_ec(participant):
    """ Copy all the annotations from the EO condition to the respective EC condition trial"""
    trials = participant['trials']
    for trial in trials:
        if trial.startswith('EC'):
            trial_name = trial.split('/')[1]
            eo_trial = 'EO/' + trial_name
            trials[trial]['annotations']['x'] = copy.deepcopy(trials[eo_trial]['annotations']['x'])
            trials[trial]['annotations']['y'] = copy.deepcopy(trials[eo_trial]['annotations']['y'])


def remove_baseline(data):
    trials = data['trials']
    print("Removing mean baseline from each trial of participant: ", data['participant'])
    baseline_eo = trials['enter/resting_EO']['prep_eeg']
    baseline_ec = trials['enter/resting_EC']['prep_eeg']
    mean_b_eo_f4 = mean(baseline_eo.matrix_data[0])
    mean_b_eo_f3 = mean(baseline_eo.matrix_data[1])
    mean_b_ec_f4 = mean(baseline_ec.matrix_data[0])
    mean_b_ec_f3 = mean(baseline_ec.matrix_data[1])
    print("Mean Baseline F4/F3 EO: ", mean_b_eo_f4, mean_b_eo_f3)
    print("Mean Baseline F4/F3 EC: ", mean_b_ec_f4, mean_b_ec_f3)

    for trial in trials:
        if trial.startswith('EO'):
            trial_data = trials[trial]['prep_eeg'].matrix_data
            trial_data[0] = trial_data[0] - mean_b_eo_f4
            trial_data[1] = trial_data[1] - mean_b_eo_f3
        if trial.startswith('EC'):
            trial_data = trials[trial]['prep_eeg'].matrix_data
            trial_data[0] = trial_data[0] - mean_b_ec_f4
            trial_data[1] = trial_data[1] - mean_b_ec_f3


def compute_asr_reconstruction(eeg, train_duration=10, train_baseline=None, sfreq=250, win_len=0.5, win_overlap=0.66):
    """ Computes ASR for the given segment of EEG data. Taken from https://nbara.github.io/python-meegkit/ """
    # If no baseline is provided, use the a portion of the eeg signal itself
    if train_baseline is None:
        train_baseline = eeg

    # Train on a clean portion of data
    asr = ASR(method='euclid', win_len=win_len, win_overlap=win_overlap)
    train_idx = np.arange(0 * sfreq, train_duration * sfreq, dtype=int)
    _, sample_mask = asr.fit(train_baseline[:, train_idx])

    # Apply filter using sliding (non-overlapping) windows
    X = sliding_window(eeg, window=int(sfreq), step=int(sfreq))
    Y = np.zeros_like(X)
    for i in range(X.shape[1]):
        Y[:, i, :] = asr.transform(X[:, i, :])

    # raw_data = X.reshape(2, -1)  # reshape to (n_chans, n_times)
    clean_data = Y.reshape(2, -1)
    return clean_data


# https://stackoverflow.com/questions/29156532/python-baseline-correction-library
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


def participant_avg_annotation_windows(data, n_windows, w_size=1.0, cleaned_eeg=False, skip_qc=True):
    windows = n_windows / w_size
    trials = data['trials']
    for trial in trials:
        if trial.startswith('EO') or trial.startswith('EC'):
            if not trials[trial]['bad_quality'] or skip_qc:
                if cleaned_eeg:
                    windows = trials[trial]['c_windows']
                if "features" not in trials[trial]:
                    trials[trial]['features'] = dict()
                annotations = trials[trial]['annotations']
                avg_valence, avg_arousal = compute_avg_annotation_windows(annotations, windows, cleaned_eeg=cleaned_eeg)
                trials[trial]['features']['avg_x'] = avg_valence
                trials[trial]['features']['avg_y'] = avg_arousal
    return data


def compute_avg_annotation_windows(annotations, n_windows, cleaned_eeg=False):
    v_label = 'x'
    a_label = 'y'
    if cleaned_eeg:
        v_label = 'c_x'
        a_label = 'c_y'

    windowed_v = np.array_split(annotations[v_label], n_windows)
    avg_valence = []
    for window in windowed_v:
        avg_window = np.mean(window)
        avg_valence.append(avg_window)

    windowed_a = np.array_split(annotations[a_label], n_windows)
    avg_arousal = []
    for window in windowed_a:
        avg_window = np.mean(window)
        avg_arousal.append(avg_window)

    return avg_valence, avg_arousal


def find_none_parameters(dataset, parameter):
    for participant_id in dataset:
        for trial in dataset[participant_id]['trials']:
            if trial.startswith('EO') or trial.startswith('EC'):
                param = dataset[participant_id]['trials'][trial]['features'][parameter]
                if param == 'None':
                    print("Found None parameter!", participant_id, trial, param)
                    return participant_id, trial, param
