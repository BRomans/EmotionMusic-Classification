import numpy as np


def get_arousal_annotations(prep_dataset, participants_subset, condition, skip_qc=True):
    """ All arousal annotations filtered by condition"""
    arousal_ann = np.array([])
    t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
    for participant_id in participants_subset:
        trials = prep_dataset[participant_id]['trials']
        for trial_class in t_classes:
            trial = trials[condition + '/' + trial_class + 'A']
            if not trial['bad_quality'] or skip_qc:
                avg_arousal = trial['features']['avg_y']
                arousal_ann = np.concatenate((arousal_ann, np.array(avg_arousal)))

            trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                avg_arousal = trial['features']['avg_y']
                arousal_ann = np.concatenate((arousal_ann, np.array(avg_arousal)))
    return arousal_ann


def get_arousal_labels(prep_dataset, participants_subset, condition, skip_qc=True, string_labels=True):
    """ All arousal labels filtered by condition"""
    arousal_labels = np.array([])
    t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
    pos_label = "HA" if string_labels else 1
    neg_label = "LA" if string_labels else -1

    for participant_id in participants_subset:
        trials = prep_dataset[participant_id]['trials']
        for trial_class in t_classes:
            trial = trials[condition + '/' + trial_class + 'A']
            if not trial['bad_quality'] or skip_qc:
                avg_arousal = trial['features']['avg_y']
                labels = [pos_label if i > 0 else neg_label for i in avg_arousal]
                arousal_labels = np.concatenate((arousal_labels, np.array(labels)))

            trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                avg_arousal = trial['features']['avg_y']
                labels = [pos_label if i > 0 else neg_label for i in avg_arousal]
                arousal_labels = np.concatenate((arousal_labels, np.array(labels)))
    return arousal_labels


def get_arousal_labels_with_neutral(prep_dataset, participants_subset, condition, skip_qc=True, string_labels=True):
    """ All arousal labels filtered by condition and converted into three classes: HA, LA and Neutral"""
    arousal_labels = np.array([])
    t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
    pos_label = "HA" if string_labels else 1
    neg_label = "LA" if string_labels else -1
    neu_label = "N" if string_labels else 0

    for participant_id in participants_subset:
        trials = prep_dataset[participant_id]['trials']
        for trial_class in t_classes:
            trial = trials[condition + '/' + trial_class + 'A']
            if not trial['bad_quality'] or skip_qc:
                avg_arousal = trial['features']['avg_y']
                labels = []
                for i in avg_arousal:
                    if i > 0.1:
                        labels.append(pos_label)
                    elif i < -0.1:
                        labels.append(neg_label)
                    else:
                        labels.append(neu_label)
                arousal_labels = np.concatenate((arousal_labels, np.array(labels)))

            trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                avg_arousal = trial['features']['avg_y']
                labels = []
                for i in avg_arousal:
                    if i > 0.1:
                        labels.append(pos_label)
                    elif i < -0.1:
                        labels.append(neg_label)
                    else:
                        labels.append(neu_label)
                arousal_labels = np.concatenate((arousal_labels, np.array(labels)))
    return arousal_labels


def get_valence_annotations(prep_dataset, participants_subset, condition, skip_qc=True):
    """ All valence annotations filtered by condition"""
    valence_ann = np.array([])
    t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
    for participant_id in participants_subset:
        trials = prep_dataset[participant_id]['trials']
        for trial_class in t_classes:
            trial = trials[condition + '/' + trial_class + 'A']
            if not trial['bad_quality'] or skip_qc:
                avg_valence = trial['features']['avg_x']
                valence_ann = np.concatenate((valence_ann, np.array(avg_valence)))

            trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                avg_valence = trial['features']['avg_x']
                valence_ann = np.concatenate((valence_ann, np.array(avg_valence)))
    return valence_ann


def get_valence_labels(prep_dataset, participants_subset, condition, skip_qc=True, string_labels=True):
    """ All valence labels filtered by condition"""
    valence_labels = np.array([])
    t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
    pos_label = "HV" if string_labels else 1
    neg_label = "LV" if string_labels else -1

    for participant_id in participants_subset:
        trials = prep_dataset[participant_id]['trials']
        for trial_class in t_classes:
            trial = trials[condition + '/' + trial_class + 'A']
            if not trial['bad_quality'] or skip_qc:
                avg_valence = trial['features']['avg_x']
                labels = [pos_label if i > 0 else neg_label for i in avg_valence]
                valence_labels = np.concatenate((valence_labels, np.array(labels)))

            trial = trials[condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                avg_valence = trial['features']['avg_x']
                labels = [pos_label if i > 0 else neg_label for i in avg_valence]
                valence_labels = np.concatenate((valence_labels, np.array(labels)))
    return valence_labels


def get_valence_labels_with_neutral(prep_dataset, participants_subset, condition, skip_qc=True, string_labels=True):
    """ All valence labels filtered by condition converted into three classes: HV, LV and Neutral"""
    valence_labels = np.array([])
    t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
    pos_label = "HV" if string_labels else 1
    neg_label = "LV" if string_labels else -1
    neu_label = "N" if string_labels else 0

    for participant_id in participants_subset:
        trials = prep_dataset[participant_id]['trials']
        for trial_class in t_classes:
            trial = trials[condition + '/' + trial_class + 'A']
            if not trial['bad_quality'] or skip_qc:
                avg_valence = trial['features']['avg_x']
                labels = []
                for i in avg_valence:
                    if i > 0.1:
                        labels.append(pos_label)
                    elif i < -0.1:
                        labels.append(neg_label)
                    else:
                        labels.append("N")
                valence_labels = np.concatenate((valence_labels, np.array(labels)))

            trial = trials[condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                avg_valence = trial['features']['avg_x']
                labels = []
                for i in avg_valence:
                    if i > 0.1:
                        labels.append(pos_label)
                    elif i < -0.1:
                        labels.append(neg_label)
                    else:
                        labels.append(neu_label)
                valence_labels = np.concatenate((valence_labels, np.array(labels)))
    return valence_labels


def get_valence_arousal_labels(prep_dataset, participants_subset, condition, skip_qc=True, string_labels=True):
    """ All VA labels filtered by condition"""
    va_labels = np.array([])
    t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
    hahv_label = "HAHV" if string_labels else 1
    lahv_label = "LAHV" if string_labels else 2
    lalv_label = "LALV" if string_labels else 3
    halv_label = "HALV" if string_labels else 4

    for participant_id in participants_subset:
        trials = prep_dataset[participant_id]['trials']
        for trial_class in t_classes:
            trial = trials[condition + '/' + trial_class + 'A']
            if not trial['bad_quality'] or skip_qc:
                avg_valence = trial['features']['avg_x']
                avg_arousal = trial['features']['avg_y']
                labels = []
                for i in range(0, len(avg_arousal)):
                    if avg_arousal[i] > 0 and avg_valence[i] > 0:
                        labels.append(hahv_label)
                    elif avg_arousal[i] <= 0 and avg_valence[i] > 0:
                        labels.append(lahv_label)
                    elif avg_arousal[i] <= 0 and avg_valence[i] <= 0:
                        labels.append(lalv_label)
                    elif avg_arousal[i] > 0 and avg_valence[i] <= 0:
                        labels.append(halv_label)
                va_labels = np.concatenate((va_labels, np.array(labels)))

            trial = trials[condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                avg_valence = trial['features']['avg_x']
                avg_arousal = trial['features']['avg_y']
                labels = []
                for i in range(0, len(avg_arousal)):
                    if avg_arousal[i] > 0 and avg_valence[i] > 0:
                        labels.append(hahv_label)
                    elif avg_arousal[i] <= 0 and avg_valence[i] > 0:
                        labels.append(lahv_label)
                    elif avg_arousal[i] <= 0 and avg_valence[i] <= 0:
                        labels.append(lalv_label)
                    elif avg_arousal[i] > 0 and avg_valence[i] <= 0:
                        labels.append(halv_label)
                va_labels = np.concatenate((va_labels, np.array(labels)))
    return va_labels


def get_neuromarker_1D(prep_dataset, participants_subset, condition, neuromarker, skip_qc=True):
    """ Get all instances of a chosen neuromarker filtered by condition"""
    indexes = np.array([])
    t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
    for participant_id in participants_subset:
        trials = prep_dataset[participant_id]['trials']
        for trial_class in t_classes:
            trial = trials[condition + '/' + trial_class + 'A']
            if not trial['bad_quality'] or skip_qc:
                idx = trial['features'][neuromarker]
                indexes = np.concatenate((indexes, np.array(idx)))

            trial = trials[condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                idx = trial['features'][neuromarker]
                indexes = np.concatenate((indexes, np.array(idx)))
    return indexes


def get_neuromarker_2D(prep_dataset, participants_subset, condition, neuromarker, skip_qc=True):
    """ Get all instances of a chosen neuromarker filtered by condition"""
    indexes = np.array([[], []])
    t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
    for participant_id in participants_subset:
        trials = prep_dataset[participant_id]['trials']
        for trial_class in t_classes:
            trial = trials[condition + '/' + trial_class + 'A']
            if not trial['bad_quality'] or skip_qc:
                idx = trial['features'][neuromarker]
                indexes = np.array(
                    [np.concatenate((indexes[0], np.array(idx[0]))), np.concatenate((indexes[1], np.array(idx[1])))])

            trial = trials[condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                idx = trial['features'][neuromarker]
                indexes = np.array(
                    [np.concatenate((indexes[0], np.array(idx[0]))), np.concatenate((indexes[1], np.array(idx[1])))])
    return indexes


def get_continuous_eeg_feature(prep_dataset, participants_subset, condition, feature, skip_qc=True, n_channels=2):
    features = [[], []]
    t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
    for participant_id in participants_subset:
        trials = prep_dataset[participant_id]['trials']
        for trial_class in t_classes:
            trial = trials[condition + '/' + trial_class + 'A']
            if not trial['bad_quality'] or skip_qc:
                idx = trial['features'][feature]
                for i in range(0, n_channels):
                    features[i] = np.concatenate((np.array(features[i]), np.array(idx[i])))

            trial = trials[condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                idx = trial['features'][feature]
                for i in range(0, n_channels):
                    features[i] = np.concatenate((np.array(features[i]), np.array(idx[i])))
    return np.array(features)


def get_discrete_feature(prep_dataset, participants_subset, condition, feature, skip_qc=True):
    """ Get all instances of a chosen trial feature, i.e. liking or familiarity, filtered by condition"""
    features = np.array([])
    t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
    for participant_id in participants_subset:
        trials = prep_dataset[participant_id]['trials']
        for trial_class in t_classes:
            trial = trials[condition + '/' + trial_class + 'A']
            if not trial['bad_quality'] or skip_qc:
                feat = trial['features'][feature]
                features = np.append(features, int(feat))

            trial = trials[condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                feat = trial['features'][feature]
                features = np.append(features, int(feat))
    return features


def get_discrete_feature_as_continuous(prep_dataset, participants_subset, condition, feature, skip_qc=True):
    """ Get all instances of a chosen trial feature, i.e. liking or familiarity, filtered by condition"""
    features = np.array([])
    t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
    for participant_id in participants_subset:
        trials = prep_dataset[participant_id]['trials']
        for trial_class in t_classes:
            trial = trials[condition + '/' + trial_class + 'A']
            if not trial['bad_quality'] or skip_qc:
                feat = trial['features'][feature]
                n_win = trial['c_windows']
                feat_array = np.full(shape=n_win, fill_value=int(feat), dtype=np.int)
                features = np.concatenate((features, feat_array))

            trial = trials[condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                feat = trial['features'][feature]
                n_win = trial['c_windows']
                feat_array = np.full(shape=n_win, fill_value=int(feat), dtype=np.int)
                features = np.concatenate((features, feat_array))
    return features
