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


def get_arousal_labels(prep_dataset, participants_subset, condition, skip_qc=True):
    """ All arousal labels filtered by condition"""
    arousal_labels = np.array([])
    t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
    for participant_id in participants_subset:
        trials = prep_dataset[participant_id]['trials']
        for trial_class in t_classes:
            trial = trials[condition + '/' + trial_class + 'A']
            if not trial['bad_quality'] or skip_qc:
                avg_arousal = trial['features']['avg_y']
                labels = ["HA" if i > 0 else "LA" for i in avg_arousal]
                arousal_labels = np.concatenate((arousal_labels, np.array(labels)))

            trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                avg_arousal = trial['features']['avg_y']
                labels = ["HA" if i > 0 else "LA" for i in avg_arousal]
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


def get_valence_labels(prep_dataset, participants_subset, condition, skip_qc=True):
    """ All valence labels filtered by condition"""
    valence_labels = np.array([])
    t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
    for participant_id in participants_subset:
        trials = prep_dataset[participant_id]['trials']
        for trial_class in t_classes:
            trial = trials[condition + '/' + trial_class + 'A']
            if not trial['bad_quality'] or skip_qc:
                avg_valence = trial['features']['avg_x']
                labels = ["HV" if i > 0 else "LV" for i in avg_valence]
                valence_labels = np.concatenate((valence_labels, np.array(labels)))

            trial = trials[condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                avg_valence = trial['features']['avg_x']
                labels = ["HV" if i > 0 else "LV" for i in avg_valence]
                valence_labels = np.concatenate((valence_labels, np.array(labels)))
    return valence_labels


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
                indexes = np.array([np.concatenate((indexes[0], np.array(idx[0]))), np.concatenate((indexes[1], np.array(idx[1])))])

            trial = trials[condition + '/' + trial_class + 'B']
            if not trial['bad_quality'] or skip_qc:
                idx = trial['features'][neuromarker]
                indexes = np.array([np.concatenate((indexes[0], np.array(idx[0]))), np.concatenate((indexes[1], np.array(idx[1])))])
    return indexes


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
