import numpy as np
from mbt_pyspt.models.mybraineegdata import MyBrainEEGData
from mbt_pyspt.modules.featuresextractionflow import FeaturesExtractionFlow


def compute_participant_features(data, ff_list, split_data, sr, loc, cleaned_eeg=False, skip_qc=True):
    """ Retrieve the alpha, beta and theta power in specified time windows to calculate the neuromarkers
    Normalization: should I normalize? while extracting frequency bands or after computing the neuromarkers?

    Alpha Power:
    increasing/decreasing frontal alpha power can correlate with increasing/decreasing arousal
    https://www.tandfonline.com/doi/abs/10.1080/02699930126048

    SASI index:
    (beta - theta)/(beta + theta). Increases for Negative emotions and decreases for
    positive emotions
    https://pubmed.ncbi.nlm.nih.gov/26738175/
    https://ieeexplore.ieee.org/document/8217815

    AW Index:
    alphaAF4 - alphaAF3. Left hemisphere (AF3) for Positive Valence, right hemisphere (AF4) for Negative Valence.
    If AWIdx is positive, there is a right tendency and NV. if AWIdx is negative, there is left tendency and PV.
    What if values are negatives? Should we subtract absolute values?
    https://www.tandfonline.com/doi/abs/10.1080/02699930126048

    FMT Index:
    mean Theta power stimulus/ mean Theta power baseline for Fz channel. It follows approach-withdrawal
    tendencies similarly to AWIndex but it increases/decreases in correlation with pleasant/unpleasant music.
    Should we investigate correlation with liking?
    https://stefan-koelsch.de/papers/Sammler_2007_music-emotion-Fm-theta-HR-EEG.pdf

    FFT-based or wavelet based features:
    Nothing for the moment

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




