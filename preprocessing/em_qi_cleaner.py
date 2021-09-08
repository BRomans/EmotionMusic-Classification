import copy

import numpy as np
from mbt_pyspt.models.mybraineegdata import MyBrainEEGData


def qi_data_removal(participant, trial_duration=60, qi_window_size=6, qi_threshold=1.0, allowed_loss=50, sr=250):
    trials = participant['trials']
    channel_locations = participant['acquisitionLocation']
    participant['bad_trials'] = 0
    for trial in trials:
        if trial.startswith('EO') or trial.startswith('EC'):
            print(trial)
            # find time windows that should be removed according to QI

            prep_eeg = participant['trials'][trial]['prep_eeg']
            qualities = participant['trials'][trial]['qualities']
            annotations = participant['trials'][trial]['annotations']
            participant['trials'][trial]['bad_quality'] = False

            qualities_norm = [[], []]
            qualities_norm[0] = normalize_qualities(qualities[0])
            qualities_norm[1] = normalize_qualities(qualities[1])

            # save the indexes that should be removed as 1 and to be kept as 0. Both channels are iterated and if one of them is below the desired quality, remove both
            win_to_remove = windows_to_remove(trial_duration, qi_threshold, qualities_norm, qi_window_size)
            n_windows = len(win_to_remove)
            print("Windows to remove", win_to_remove)

            # Now loop through the time windows and save the preprocessed EEG in a new variable withouht the low quality data by copying window by window (1 * sampling_rate)
            cleaned_eeg = [[], []]
            split_eeg_F4 = np.array_split(prep_eeg[0], n_windows)
            split_eeg_F3 = np.array_split(prep_eeg[1], n_windows)
            annotations['c_x'] = []
            annotations['c_y'] = []
            ann_sr = int(round(len(annotations["x"]) / trial_duration))
            split_annotations_x = np.array_split(annotations["x"], n_windows)
            split_annotations_y = np.array_split(annotations["y"], n_windows)

            for idx in range(0, n_windows):
                if win_to_remove[idx] == 0:
                    cleaned_eeg[0].extend(copy.deepcopy(split_eeg_F4[idx]))
                    cleaned_eeg[1].extend(copy.deepcopy(split_eeg_F3[idx]))
                    annotations['c_x'].extend(split_annotations_x[idx])
                    annotations['c_y'].extend(split_annotations_y[idx])
            cleaned_eeg = np.array(cleaned_eeg)
            print("Clean seconds of EEG:", cleaned_eeg.shape[1] / 250)
            print("Clean seconds of annotations:", len(annotations['c_x']) / ann_sr,
                  len(annotations['c_y']) / ann_sr)

            participant['trials'][trial]['clean_eeg'] = cleaned_eeg
            participant['trials'][trial]['c_windows'] = n_windows - np.count_nonzero(win_to_remove)
            participant['trials'][trial]['all_windows'] = win_to_remove
            perc_bad = round(np.count_nonzero(win_to_remove) / len(win_to_remove) * 100, 2)
            print("Percentage of pruned data: " + str(perc_bad) + "%")
            if perc_bad > allowed_loss:
                print("Marked trial for rejection: ", trial)
                participant['bad_trials'] += 1
                participant['trials'][trial]['bad_quality'] = True


def normalize_qualities(qualities):
    new_q = []
    for value in qualities:
        if value == -1:
            new_q.append(0)
        elif value == 0.25:
            new_q.append(0.5)
        else:
            new_q.append(value)
    return new_q


def windows_to_remove(trial_duration, qi_threshold, qualities, qi_window_size):
    qi_windows = int(trial_duration / qi_window_size)
    win_to_remove = np.zeros(qi_windows, dtype=int)

    avg_qi_0 = np.average(np.array(qualities[0]).reshape(-1, qi_window_size), axis=1)
    avg_qi_1 = np.average(np.array(qualities[1]).reshape(-1, qi_window_size), axis=1)

    for i in range(0, qi_windows):
        if float(avg_qi_0[i]) < qi_threshold or float(avg_qi_1[i]) < qi_threshold:
            win_to_remove[i] = 1

    return win_to_remove


