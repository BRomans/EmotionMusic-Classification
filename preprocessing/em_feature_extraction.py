import numpy as np
from mbt_pyspt.models.mybraineegdata import MyBrainEEGData
from mbt_pyspt.modules.featuresextractionflow import FeaturesExtractionFlow


def compute_participant_features_baseline_normalized(participant, split_data, sr, loc, cleaned_eeg=False, skip_qc=True):
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
    What if values are both negatives? Should we subtract absolute values?
    https://www.tandfonline.com/doi/abs/10.1080/02699930126048

    FMT Index:
    mean Theta power stimulus/ mean Theta power baseline for Fz channel. It follows approach-withdrawal
    tendencies similarly to AWIndex but it increases/decreases in correlation with pleasant/unpleasant music.
    Should we investigate correlation with liking?
    https://stefan-koelsch.de/papers/Sammler_2007_music-emotion-Fm-theta-HR-EEG.pdf

    FFT-based or wavelet based features:
    Nothing for the moment

    """
    # Power features extraction methods
    baseline_extraction = [("get_amp_filtered_signal", {'samp_rate': 250, 'l_freq': 4.0, 'h_freq': 28.0})]
    theta_extraction = [("get_power_theta", {'samp_rate': 250, 'l_freq': 4.0, 'h_freq': 8.0})]
    alpha_extraction = [("get_power_alpha", {'samp_rate': 250, 'l_freq': 8.0, 'h_freq': 13.0})]
    beta_extraction = [("get_power_beta", {'samp_rate': 250, 'l_freq': 13.0, 'h_freq': 28.0})]

    # Other spectral features extraction methods
    skewness_theta_ext = [("get_skewness_theta", {})]  # skewness of theta EEG in eeg_data
    skewness_alpha_ext = [("get_skewness_alpha", {})]  # skewness of alpha EEG in eeg_data
    skewness_beta_ext = [("get_skewness_beta", {})]  # skewness of beta EEG in eeg_data
    kurtosis_theta_ext = [("get_kurtosis_theta", {})]  # kurtosis of theta EEG in eeg_data
    kurtosis_alpha_ext = [("get_kurtosis_alpha", {})]  # kurtosis of alpha EEG in eeg_data
    kurtosis_beta_ext = [("get_kurtosis_beta", {})]  # kurtosis of beta EEG in eeg_data
    std_theta_ext = [("get_std_theta", {})]  # standard deviation of theta EEG in eeg_data
    std_alpha_ext = [("get_std_alpha", {})]  # standard deviation of alpha EEG in eeg_data
    std_beta_ext = [("get_std_beta", {})]  # standard deviation of beta EEG in eeg_data
    ratio_theta_ext = [("get_ratio_theta", {})]  # theta ratio for EEG in eeg_data
    ratio_alpha_ext = [("get_ratio_alpha", {})]  # alpha ratio for EEG in eeg_data
    ratio_beta_ext = [("get_ratio_beta", {})]  # beta ratio for EEG in eeg_data
    rsd_theta_ext = [("get_rsd_theta", {})]  # relative spectral difference for theta in eeg_data
    rsd_alpha_ext = [("get_rsd_alpha", {})]  # relative spectral difference for alpha in eeg_data
    rsd_beta_ext = [("get_rsd_beta", {})]  # relative spectral difference for beta in eeg_data

    baseline_eeg = MyBrainEEGData(participant['baseline_eeg'], sr, loc)

    extraction = FeaturesExtractionFlow(baseline_eeg, features_list=baseline_extraction)
    avg_bas, avg_bas_labels = extraction()
    print("Baseline Avg Amplitude", avg_bas, avg_bas_labels)

    extraction = FeaturesExtractionFlow(baseline_eeg, features_list=theta_extraction)
    theta_bas, theta_bas_labels = extraction()
    print("Baseline Theta Power", theta_bas, theta_bas_labels)

    extraction = FeaturesExtractionFlow(baseline_eeg, features_list=alpha_extraction)
    alpha_bas, alpha_bas_labels = extraction()
    print("Baseline Alpha Power", alpha_bas, alpha_bas_labels)

    extraction = FeaturesExtractionFlow(baseline_eeg, features_list=beta_extraction)
    beta_bas, beta_bas_labels = extraction()
    print("Baseline Beta Power", beta_bas, beta_bas_labels)

    trials = participant['trials']
    eeg_label = 'prep_eeg'
    if cleaned_eeg:
        eeg_label = 'clean_eeg'
    for trial in trials:
        if trial.startswith('EO') or trial.startswith('EC'):
            n_win = trials[trial]['c_windows']
            if not trials[trial]['bad_quality'] or skip_qc:
                eeg = MyBrainEEGData(trials[trial][eeg_label], sr, loc)

                extraction = FeaturesExtractionFlow(eeg, features_list=theta_extraction, split_data=split_data)
                theta_powers, labels = extraction()
                norm_theta_pow = decibel_normalization(theta_powers, theta_bas)
                print("Theta Powers", theta_powers)
                print("Norm Theta Powers", norm_theta_pow)

                extraction = FeaturesExtractionFlow(eeg, features_list=alpha_extraction, split_data=split_data)
                alpha_powers, labels = extraction()
                norm_alpha_pow = decibel_normalization(alpha_powers, alpha_bas)
                print("Alpha Powers", alpha_powers)
                print("Norm Alpha Powers", norm_alpha_pow)

                extraction = FeaturesExtractionFlow(eeg, features_list=beta_extraction, split_data=split_data)
                beta_powers, labels = extraction()
                norm_beta_pow = decibel_normalization(beta_powers, beta_bas)
                print("Beta Powers", beta_powers)
                print("Norm Beta Powers", norm_beta_pow)

                # Computing neuromarkers using decibel normalized powers
                aw_idx = np.subtract(norm_alpha_pow[0], norm_alpha_pow[1])  # alpha AF4 - alpha AF3
                print("AWI", aw_idx)

                fmt_idx = np.median(norm_theta_pow, axis=0)  # median(theta AF4, theta AF3) element-wise
                print("FMT", fmt_idx)

                sasi_idx = np.array([
                    np.divide(np.subtract(norm_beta_pow[0], norm_theta_pow[0]),
                              np.add(norm_beta_pow[0], norm_theta_pow[0])),
                    np.divide(np.subtract(norm_beta_pow[1], norm_theta_pow[1]),
                              np.add(norm_beta_pow[1], norm_theta_pow[1]))
                ])  # (beta - theta / beta + theta)AF4, (beta - theta / beta + theta)AF3
                print("SASI", sasi_idx)

                extraction = FeaturesExtractionFlow(eeg, features_list=skewness_theta_ext, split_data=split_data)
                skewness_theta, labels = extraction()
                print("Skewness Theta", skewness_theta)

                extraction = FeaturesExtractionFlow(eeg, features_list=skewness_alpha_ext, split_data=split_data)
                skewness_alpha, labels = extraction()
                print("Skewness Alpha", skewness_alpha)

                extraction = FeaturesExtractionFlow(eeg, features_list=skewness_beta_ext, split_data=split_data)
                skewness_beta, labels = extraction()
                print("Skewness Beta", skewness_beta)

                extraction = FeaturesExtractionFlow(eeg, features_list=kurtosis_theta_ext, split_data=split_data)
                kurtosis_theta, labels = extraction()
                print("Kurtosis Theta", kurtosis_theta)

                extraction = FeaturesExtractionFlow(eeg, features_list=kurtosis_alpha_ext, split_data=split_data)
                kurtosis_alpha, labels = extraction()
                print("Kurtosis Alpha", kurtosis_alpha)

                extraction = FeaturesExtractionFlow(eeg, features_list=kurtosis_beta_ext, split_data=split_data)
                kurtosis_beta, labels = extraction()
                print("Kurtosis Beta", kurtosis_beta)

                extraction = FeaturesExtractionFlow(eeg, features_list=std_theta_ext, split_data=split_data)
                std_theta, labels = extraction()
                print("Standard Deviation Theta", std_theta)

                extraction = FeaturesExtractionFlow(eeg, features_list=std_alpha_ext, split_data=split_data)
                std_alpha, labels = extraction()
                print("Standard Deviation Alpha", std_alpha)

                extraction = FeaturesExtractionFlow(eeg, features_list=std_beta_ext, split_data=split_data)
                std_beta, labels = extraction()
                print("Standard Deviation Beta", std_beta)

                extraction = FeaturesExtractionFlow(eeg, features_list=ratio_theta_ext, split_data=split_data)
                ratio_theta, labels = extraction()
                print("Ratio Theta", ratio_theta)

                extraction = FeaturesExtractionFlow(eeg, features_list=ratio_alpha_ext, split_data=split_data)
                ratio_alpha, labels = extraction()
                print("Ratio Alpha", ratio_alpha)

                extraction = FeaturesExtractionFlow(eeg, features_list=ratio_beta_ext, split_data=split_data)
                ratio_beta, labels = extraction()
                print("Ratio Beta", ratio_beta)

                extraction = FeaturesExtractionFlow(eeg, features_list=rsd_theta_ext, split_data=split_data)
                rsd_theta, labels = extraction()
                print("RSD Theta", rsd_theta)

                extraction = FeaturesExtractionFlow(eeg, features_list=rsd_alpha_ext, split_data=split_data)
                rsd_alpha, labels = extraction()
                print("RSD Alpha", rsd_alpha)

                extraction = FeaturesExtractionFlow(eeg, features_list=rsd_beta_ext, split_data=split_data)
                rsd_beta, labels = extraction()
                print("RSD Beta", rsd_beta)

                # Save all features in the participant object
                if "features" not in trials[trial]:
                    trials[trial]['features'] = dict()
                trials[trial]['features']['norm_theta_pow'] = norm_theta_pow
                trials[trial]['features']['norm_alpha_pow'] = norm_alpha_pow
                trials[trial]['features']['norm_beta_pow'] = norm_beta_pow
                trials[trial]['features']['aw_idx'] = aw_idx
                trials[trial]['features']['sasi_idx'] = sasi_idx
                trials[trial]['features']['fmt_idx'] = fmt_idx
                trials[trial]['features']['skewness_theta'] = skewness_theta
                trials[trial]['features']['skewness_alpha'] = skewness_alpha
                trials[trial]['features']['skewness_beta'] = skewness_beta
                trials[trial]['features']['kurtosis_theta'] = kurtosis_theta
                trials[trial]['features']['kurtosis_alpha'] = kurtosis_alpha
                trials[trial]['features']['kurtosis_beta'] = kurtosis_beta
                trials[trial]['features']['std_theta'] = std_theta
                trials[trial]['features']['std_alpha'] = std_alpha
                trials[trial]['features']['std_beta'] = std_beta
                trials[trial]['features']['ratio_theta'] = ratio_theta
                trials[trial]['features']['ratio_alpha'] = ratio_alpha
                trials[trial]['features']['ratio_beta'] = ratio_beta
                trials[trial]['features']['rsd_theta'] = rsd_theta
                trials[trial]['features']['rsd_alpha'] = rsd_alpha
                trials[trial]['features']['rsd_beta'] = rsd_beta
                trials[trial]['features']['familiarity'] = trials[trial]['annotations']['familiarity']
                trials[trial]['features']['liking'] = trials[trial]['annotations']['liking']


def percent_normalization(freq_power, bas_freq_power):
    normalized_channels = []
    for i in range(0, freq_power.shape[0]):
        norm_channel_pow = 100 * ((freq_power[i] - bas_freq_power[i]) / bas_freq_power[i])
        normalized_channels.append(norm_channel_pow)
    return np.array(normalized_channels)


def decibel_normalization(freq_power, bas_freq_power):
    normalized_channels = []
    for i in range(0, freq_power.shape[0]):
        norm_channel_pow = 10 * (np.log10(freq_power[i] / bas_freq_power[i]))
        normalized_channels.append(norm_channel_pow)
    return np.array(normalized_channels)


@DeprecationWarning
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
