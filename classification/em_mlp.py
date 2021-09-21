import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from classification.em_features_aggregator import get_arousal_labels, get_valence_labels, get_neuromarker_2D, \
    get_neuromarker_1D, get_discrete_feature, get_valence_arousal_labels


def run_subject_dependent_experiment_mlp(dataset, condition):
    print("Running new subject-dependent experiment with condition " + condition)
    arousal_results = [[], [], []]
    valence_results = [[], [], []]
    va_results = [[], [], []]

    for participant in dataset:
        train_participants = [participant]

        print("Testing with participant", participant)

        arousal_labels = get_arousal_labels(dataset, train_participants, condition, False)
        valence_labels = get_valence_labels(dataset, train_participants, condition, False)
        alpha_norm = get_neuromarker_2D(dataset, train_participants, condition, 'alpha_pow', False)
        aw_idx = get_neuromarker_1D(dataset, train_participants, condition, 'aw_idx', False)
        fmt_idx = get_neuromarker_1D(dataset, train_participants, condition, 'fmt_idx', False)
        sasi_idx = get_neuromarker_2D(dataset, train_participants, condition, 'sasi_idx', False)
        liking = get_discrete_feature(dataset, train_participants, condition, 'liking', False)
        familiarity = get_discrete_feature(dataset, train_participants, condition, 'familiarity', False)

        va_labels = get_valence_arousal_labels(dataset, train_participants, condition, False)

        X_a = np.array([aw_idx, fmt_idx, sasi_idx[0], sasi_idx[1], alpha_norm[0], alpha_norm[1]]).reshape(-1, 6)
        X_v = np.array([aw_idx, fmt_idx, sasi_idx[0], sasi_idx[1], alpha_norm[0], alpha_norm[1]]).reshape(-1, 6)
        X_va = np.array([aw_idx, fmt_idx, sasi_idx[0], sasi_idx[1], alpha_norm[0], alpha_norm[1]]).reshape(-1, 6)

        y_a = np.array(arousal_labels)
        y_v = np.array(valence_labels)
        y_va = np.array(va_labels)

        x_a_train, x_a_test, y_a_train, y_a_test = train_test_split(X_a, y_a, test_size=0.2, random_state=1)
        x_v_train, x_v_test, y_v_train, y_v_test = train_test_split(X_v, y_v, test_size=0.2, random_state=1)
        x_va_train, x_va_test, y_va_train, y_va_test = train_test_split(X_va, y_va, test_size=0.2, random_state=1)

        # Arousal classifier
        arousal_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                    hidden_layer_sizes=(5, 2), random_state=1)
        arousal_clf.fit(x_a_train, y_a_train)

        # Valence classifier
        valence_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                    hidden_layer_sizes=(5, 2), random_state=1)
        valence_clf.fit(x_v_train, y_v_train)

        # Valence-Arousal classifier
        va_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(5, 2), random_state=1)
        va_clf.fit(x_va_train, y_va_train)

        # Arousal prediction
        arousal_pred = arousal_clf.predict(x_a_test)
        arousal_accuracy = accuracy_score(y_a_test, arousal_pred)
        arousal_f1 = f1_score(y_a_test, arousal_pred, average='weighted')
        arousal_precision = precision_score(y_a_test, arousal_pred, labels=["LA"], average='weighted')
        arousal_results[0].append(arousal_accuracy)
        arousal_results[1].append(arousal_f1)
        arousal_results[2].append(arousal_precision)

        # Valence prediction
        valence_pred = valence_clf.predict(x_v_test)
        valence_accuracy = accuracy_score(y_v_test, valence_pred)
        valence_f1 = f1_score(y_v_test, valence_pred, average='weighted')
        valence_precision = precision_score(y_v_test, valence_pred, labels=["LV"], average='weighted')
        valence_results[0].append(valence_accuracy)
        valence_results[1].append(valence_f1)
        valence_results[2].append(valence_precision)

        # Valence-Arousal prediction
        va_pred = va_clf.predict(x_va_test)
        va_accuracy = accuracy_score(y_va_test, va_pred)
        va_f1 = f1_score(y_va_test, va_pred, average='weighted')
        va_precision = precision_score(y_va_test, va_pred, average='weighted')
        va_results[0].append(va_accuracy)
        va_results[1].append(va_f1)
        va_results[2].append(va_precision)
    arousal_results = np.array(arousal_results)
    valence_results = np.array(valence_results)
    va_results = np.array(va_results)

    return arousal_results, valence_results, va_results


def run_subject_independent_experiment_mlp(dataset, condition):
    arousal_results = [[], [], []]
    valence_results = [[], [], []]
    va_results = [[], [], []]
    participant_keys = [*dataset]
    print("Running new subject-independent experiment with condition " + condition)

    for participant in dataset:
        train_participants = deepcopy(participant_keys)
        train_participants.remove(participant)
        test_participants = [participant]
        print("Testing with participant", test_participants)

        arousal_labels = get_arousal_labels(dataset, train_participants, condition, False)
        valence_labels = get_valence_labels(dataset, train_participants, condition, False)
        alpha_norm = get_neuromarker_2D(dataset, train_participants, condition, 'alpha_pow', False)
        aw_idx = get_neuromarker_1D(dataset, train_participants, condition, 'aw_idx', False)
        fmt_idx = get_neuromarker_1D(dataset, train_participants, condition, 'fmt_idx', False)
        sasi_idx = get_neuromarker_2D(dataset, train_participants, condition, 'sasi_idx', False)
        liking = get_discrete_feature(dataset, train_participants, condition, 'liking', False)
        familiarity = get_discrete_feature(dataset, train_participants, condition, 'familiarity', False)

        va_labels = get_valence_arousal_labels(dataset, train_participants, condition, False)

        # Input
        X_a = np.array([aw_idx, fmt_idx, sasi_idx[0], sasi_idx[1], alpha_norm[0], alpha_norm[1]]).reshape(-1, 6)
        X_v = np.array([aw_idx, fmt_idx, sasi_idx[0], sasi_idx[1], alpha_norm[0], alpha_norm[1]]).reshape(-1, 6)
        X_va = np.array([aw_idx, fmt_idx, sasi_idx[0], sasi_idx[1], alpha_norm[0], alpha_norm[1]]).reshape(-1, 6)

        y_a = np.array(arousal_labels)
        y_v = np.array(valence_labels)
        y_va = np.array(va_labels)

        # Arousal classifier
        arousal_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                    hidden_layer_sizes=(5, 2), random_state=1)
        arousal_clf.fit(X_a, y_a)

        # Valence classifier
        valence_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                    hidden_layer_sizes=(5, 2), random_state=1)
        valence_clf.fit(X_v, y_v)

        # Valence-Arousal classifier
        va_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(5, 2), random_state=1)
        va_clf.fit(X_va, y_va)

        t_aw_idx = get_neuromarker_1D(dataset, test_participants, condition, 'aw_idx', False)
        t_fmt_idx = get_neuromarker_1D(dataset, test_participants, condition, 'fmt_idx', False)
        t_alpha_norm = get_neuromarker_2D(dataset, test_participants, condition, 'alpha_pow', False)
        t_sasi_idx = get_neuromarker_2D(dataset, test_participants, condition, 'sasi_idx', False)

        X_test = np.array(
            [t_aw_idx, t_fmt_idx, t_sasi_idx[0], t_sasi_idx[1], t_alpha_norm[0], t_alpha_norm[1]]).reshape(-1, 6)

        # Arousal prediction
        y_test = get_arousal_labels(dataset, test_participants, condition, False)
        arousal_pred = arousal_clf.predict(X_test)
        arousal_accuracy = accuracy_score(y_test, arousal_pred)
        arousal_f1 = f1_score(y_test, arousal_pred, average='weighted')
        arousal_precision = precision_score(y_test, arousal_pred, labels=["LA"], average='weighted')
        arousal_results[0].append(arousal_accuracy)
        arousal_results[1].append(arousal_f1)
        arousal_results[2].append(arousal_precision)

        # Valence prediction
        y_test = get_valence_labels(dataset, test_participants, condition, False)
        valence_pred = valence_clf.predict(X_test)
        valence_accuracy = accuracy_score(y_test, valence_pred)
        valence_f1 = f1_score(y_test, valence_pred, average='weighted')
        valence_precision = precision_score(y_test, valence_pred, labels=["LV"], average='weighted')
        valence_results[0].append(valence_accuracy)
        valence_results[1].append(valence_f1)
        valence_results[2].append(valence_precision)

        # Valence-Arousal prediction
        y_test = get_valence_arousal_labels(dataset, test_participants, condition, False)
        va_pred = va_clf.predict(X_test)
        va_accuracy = accuracy_score(y_test, va_pred)
        va_f1 = f1_score(y_test, va_pred, average='weighted')
        va_precision = precision_score(y_test, va_pred, average='weighted')
        va_results[0].append(va_accuracy)
        va_results[1].append(va_f1)
        va_results[2].append(va_precision)
    arousal_results = np.array(arousal_results)
    valence_results = np.array(valence_results)
    va_results = np.array(va_results)
    return arousal_results, valence_results, va_results
