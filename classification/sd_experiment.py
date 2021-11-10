import copy
from datetime import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler

from classification.em_features_aggregator import get_arousal_labels, get_valence_labels, get_valence_arousal_labels, \
    get_arousal_class_labels, get_valence_class_labels, get_valence_arousal_class_labels, \
    get_discrete_feature_as_labels, get_neuromarker_1D, get_neuromarker_2D, get_continuous_eeg_feature, \
    get_discrete_feature_as_continuous
from utils.em_data_loader import load_dataset
from utils.em_plotting import plot_labels_distribution, plot_kfold_roc_curve, plot_confusion_matrix_for_classifier


def majority_chance_level(labels, classes):
    c1_count = np.count_nonzero(labels == classes[0])
    c2_count = np.count_nonzero(labels == classes[1])
    if (c1_count >= c2_count):
        return c1_count / (c1_count + c2_count)
    else:
        return c2_count / (c1_count + c2_count)


def retrieve_best_parameters_grid_search_svm(parameter_space, X, y, weights, scoring='mcc',d_function='ovo'):
    if scoring == 'mcc':
        scoring = make_scorer(matthews_corrcoef)
    clf = svm.SVC(decision_function_shape=d_function, probability=True, class_weight=weights)
    clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=cv, scoring=scoring, refit=True)
    clf.fit(X, y)
    print('Best parameters found:\n', clf.best_params_)
    return clf.best_params_


def retrieve_best_parameters_grid_search_mlp(parameter_space, X, y, scoring='mcc', max_iter=10000):
    if scoring == 'mcc':
        scoring = make_scorer(matthews_corrcoef)
    mlp = MLPClassifier(max_iter=max_iter)
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=cv, scoring=scoring, refit=True)
    clf.fit(X, y)
    print('Best parameters found:\n', clf.best_params_)
    return clf.best_params_


time_window = "5"
preprocessed_folder = '../data/preprocessed_' + time_window
prep_dataset = load_dataset(preprocessed_folder)
participants = prep_dataset.keys()

condition = 'EO&EC'
full_features_set = True
user_labels = True
plotting = True
liking_class = False
normalize_labels = False
grid_optimization = True
n_top_features = 5
cv = StratifiedKFold(5, shuffle=False)
mcc_scoring = make_scorer(matthews_corrcoef)
template_table = {
    'Participant': [],
    'Condition': condition,
    'TimeWindow': time_window,
    'Classifier': [],
    'Features': [],
    'Algorithme': [],
    'Chance Level': [],
    'Mean': [],
    'Std': [],
    'Max': [],
    'Min': [],
    'F1 Score': [],
    'MCC': [],
    'Scores': [],
    'Avg Chance Level': [],
    'Avg Mean Score': [],
    'Avg Std': [],
    'Max Mean Score': [],
    'Min Mean Score': [],
    'Avg F1 Score': [],
    'Avg MCC': []
}

arousal_svm_table = copy.deepcopy(template_table)
valence_svm_table = copy.deepcopy(template_table)
va_svm_table = copy.deepcopy(template_table)
arousal_mlp_table = copy.deepcopy(template_table)
valence_mlp_table = copy.deepcopy(template_table)
va_mlp_table = copy.deepcopy(template_table)
liking_mlp_table = copy.deepcopy(template_table)

va_balanced = ['s010704', 's250602', 's050702', 's280603', 's230603']

best_valence = ['s060703', 's050704', 's290601', 's020703']
best_arousal = ['s060703', 's220602', 's010703', 's230603']
worst_arousal = ['s290603', 's010704', 's230604', 's280603', 's250601', 's290604']
worst_valence = ['s280603', 's230603', 's230602', 's050702', 's280604']

np.random.seed(4)
t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
np.random.shuffle(t_classes)
print(t_classes)

participant_keys = va_balanced
for participant in participant_keys:
    participant_tic_fwd = time()

    train_participants = [participant]

    participant_id = train_participants[0]
    print("Processing", participant_id)

    # Features preparation

    # Labels
    if user_labels:
        arousal_labels = get_arousal_labels(prep_dataset, train_participants, condition, skip_qc=False,
                                            string_labels=True, t_classes=t_classes, normalize_labels=normalize_labels)
        valence_labels = get_valence_labels(prep_dataset, train_participants, condition, skip_qc=False,
                                            string_labels=True, t_classes=t_classes, normalize_labels=normalize_labels)
        va_labels = get_valence_arousal_labels(prep_dataset, train_participants, condition, skip_qc=False,
                                               string_labels=True, t_classes=t_classes,
                                               normalize_labels=normalize_labels)
    else:
        arousal_labels = get_arousal_class_labels(prep_dataset, train_participants, condition, skip_qc=False,
                                                  string_labels=True, t_classes=t_classes)
        valence_labels = get_valence_class_labels(prep_dataset, train_participants, condition, skip_qc=False,
                                                  string_labels=True, t_classes=t_classes)
        va_labels = get_valence_arousal_class_labels(prep_dataset, train_participants, condition, skip_qc=False,
                                                     string_labels=True, t_classes=t_classes)

    if liking_class:
        liking_labels = get_discrete_feature_as_labels(prep_dataset, train_participants, condition, 'liking',
                                                       skip_qc=False, t_classes=t_classes)

    arousal_num_labels = get_arousal_labels(prep_dataset, train_participants, condition, skip_qc=False,
                                            string_labels=False)
    valence_num_labels = get_valence_labels(prep_dataset, train_participants, condition, skip_qc=False,
                                            string_labels=False)
    va_num_labels = get_valence_arousal_labels(prep_dataset, train_participants, condition, skip_qc=False,
                                               string_labels=False)

    # Calculate chance level based on majority class
    chance_arousal = np.round(majority_chance_level(arousal_labels, ['HA', 'LA']), 2)
    chance_valence = np.round(majority_chance_level(valence_labels, ['HV', 'LV']), 2)

    # Neuromarkers
    aw_idx = get_neuromarker_1D(prep_dataset, train_participants, condition, 'aw_idx', False)
    fmt_idx = get_neuromarker_1D(prep_dataset, train_participants, condition, 'fmt_idx', False)
    sasi_idx = get_neuromarker_2D(prep_dataset, train_participants, condition, 'sasi_idx', False)

    # Continuous EEG features
    theta_norm = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'norm_theta_pow', False)
    alpha_norm = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'norm_alpha_pow', False)
    beta_norm = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'norm_beta_pow', False)

    skewness_theta = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'skewness_theta', False)
    skewness_alpha = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'skewness_alpha', False)
    skewness_beta = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'skewness_beta', False)

    kurtosis_theta = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'kurtosis_theta', False)
    kurtosis_alpha = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'kurtosis_alpha', False)
    kurtosis_beta = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'kurtosis_beta', False)

    std_theta = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'std_theta', False)
    std_alpha = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'std_alpha', False)
    std_beta = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'std_beta', False)

    ratio_theta = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'ratio_theta', False)
    ratio_alpha = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'ratio_alpha', False)
    ratio_beta = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'ratio_beta', False)

    rsd_theta = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'rsd_theta', False)
    rsd_alpha = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'rsd_alpha', False)
    rsd_beta = get_continuous_eeg_feature(prep_dataset, train_participants, condition, 'rsd_beta', False)

    # Discrete features
    liking = get_discrete_feature_as_continuous(prep_dataset, train_participants, condition, 'liking', False)
    familiarity = get_discrete_feature_as_continuous(prep_dataset, train_participants, condition, 'familiarity', False)

    # Plot labels distribution
    if plotting:
        plot_labels_distribution(participant, arousal_labels, valence_labels, va_labels)

    # Full features set

    features = np.array([aw_idx, fmt_idx, sasi_idx[0], sasi_idx[1],
                         theta_norm[0], theta_norm[1],
                         alpha_norm[0], alpha_norm[1],
                         beta_norm[0], beta_norm[1],
                         skewness_theta[0], skewness_theta[1],
                         skewness_alpha[0], skewness_alpha[1],
                         skewness_beta[0], skewness_beta[1],
                         kurtosis_theta[0], kurtosis_theta[1],
                         kurtosis_alpha[0], kurtosis_alpha[1],
                         kurtosis_beta[0], kurtosis_beta[1],
                         std_theta[0], std_theta[1],
                         std_alpha[0], std_alpha[1],
                         std_beta[0], std_beta[1],
                         ratio_theta[0], ratio_theta[1],
                         ratio_alpha[0], ratio_alpha[1],
                         ratio_beta[0], ratio_beta[1],
                         rsd_theta[0], rsd_theta[1],
                         rsd_alpha[0], rsd_alpha[1],
                         rsd_beta[0], rsd_beta[1]

                         ])

    feature_names = np.array(['aw_idx', 'fmt_idx', 'sasi_idx_f4', 'sasi_idx_f3',
                              'theta_norm_f4', 'theta_norm_f3',
                              'alpha_norm_f4', 'alpha_norm_f3',
                              'beta_norm_f4', 'beta_norm_f3',
                              'skewness_theta_f4', 'skewness_theta_f3',
                              'skewness_alpha_f4', 'skewness_alpha_f3',
                              'skewness_beta_f4', 'skewness_beta_f3',
                              'kurtosis_theta_f4', 'kurtosis_theta_f3',
                              'kurtosis_alpha_f4', 'kurtosis_alpha_f3',
                              'kurtosis_beta_f4', 'kurtosis_beta_f3',
                              'std_theta_f4', 'std_theta_f3',
                              'std_alpha_f4', 'std_alpha_f3',
                              'std_beta_f4', 'std_beta_f3',
                              'ratio_theta_f4', 'ratio_theta_f3',
                              'ratio_alpha_f4', 'ratio_alpha_f3',
                              'ratio_beta_f4', 'ratio_beta_f3',
                              'rsd_theta_f4', 'rsd_theta_f3',
                              'rsd_alpha_f4', 'rsd_alpha_f3',
                              'rsd_beta_f4', 'rsd_beta_f3'
                              ])

    # PCA features selection
    features_table = {}
    for i in range(0, len(feature_names)):
        features_table[feature_names[i]] = features[i]
    raw_features_df = pd.DataFrame(features_table)
    X = raw_features_df.values

    # Scale the data before applying PCA
    scaler = MaxAbsScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    if plotting:
        pca_40 = PCA(n_components=len(features), random_state=2020)
        pca_40 = pca_40.fit(X_scaled)
        X_pca_40 = pca_40.transform(X_scaled)
        pca_2 = PCA(n_components=2, random_state=2020)
        pca_2.fit(X_scaled)
        X_pca_2 = pca_2.transform(X_scaled)

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=X_pca_2[:, 0], y=X_pca_2[:, 1], s=70, hue=valence_labels, palette=['green', 'blue'])

        plt.title(participant + ": 2D Scatterplot: " + str(
            round(np.cumsum(pca_40.explained_variance_ratio_ * 100)[1], 2)) + "% of variability captured", pad=15)
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=X_pca_2[:, 0], y=X_pca_2[:, 1], s=70, hue=arousal_labels, palette=['orange', 'yellow'])

        plt.title(participant + ": 2D Scatterplot: " + str(
            round(np.cumsum(pca_40.explained_variance_ratio_ * 100)[1], 2)) + "% of variability captured", pad=15)
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")

    # Select the components that represent 95% of the variability
    pca_95 = PCA(n_components=0.95, random_state=2020)
    pca_95.fit(X_scaled)
    X_pca_95 = pca_95.transform(X_scaled)
    components = []
    for i in range(0, X_pca_95.shape[1]):
        components.append("PC" + str(i + 1))
    components = np.array(components)
    print(components)
    pc_df = pd.DataFrame(X_pca_95, columns=components)
    print(pc_df.values.shape, arousal_labels.shape)

    X_a = pc_df.values
    y_a = np.array(arousal_labels)

    X_v = pc_df.values
    y_v = np.array(valence_labels)

    X_va = pc_df.values
    y_va = np.array(va_labels)

    x_a_train, x_a_test, y_a_train, y_a_test = train_test_split(X_a, y_a, test_size=0.2, random_state=2020)
    x_v_train, x_v_test, y_v_train, y_v_test = train_test_split(X_v, y_v, test_size=0.2, random_state=2020)
    x_va_train, x_va_test, y_va_train, y_va_test = train_test_split(X_va, y_va, test_size=0.2, random_state=2020)

    # SVM Cross-validation
    C = 0.0001
    gamma = 0.1
    kernel = 'rbf'
    parameter_space = {
        'C': [0.1, 1, 10, 100, 1000, 10000],
        'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf']
    }
    # Arousal classifiers
    df_a = pd.DataFrame(y_a)
    a_weights = df_a.value_counts(normalize=True)
    weights = {'HA': a_weights['HA'], 'LA': a_weights['LA']}
    if grid_optimization:
        best_params = retrieve_best_parameters_grid_search_svm(parameter_space, X_a, y_a, weights, d_function='ovo')
        kernel = best_params['kernel']
        C = best_params['C']
        gamma = best_params['gamma']
    arousal_clf = svm.SVC(kernel=kernel, probability=True, C=C, class_weight=weights, gamma=gamma)
    scores = cross_val_score(arousal_clf, X_a, y_a, cv=cv, scoring='balanced_accuracy')
    mcc = cross_val_score(arousal_clf, X_a, y_a, cv=cv, scoring=mcc_scoring)
    f1 = cross_val_score(arousal_clf, X_a, y_a, cv=cv, scoring='f1_weighted')
    mcc = np.nanmean(mcc)
    f1 = np.nanmean(f1)
    print(
        "Arousal SVM: %0.2f accuracy with a standard deviation of %0.2f, Max of %0.2f and Min of %0.2f, MCC %0.2f " % (
        np.nanmean(scores), np.nanstd(scores), max(scores), min(scores), mcc))
    if plotting:
        plot_kfold_roc_curve(arousal_clf, X_a, y_a, cv, 'HA', "Arousal")
        plot_confusion_matrix_for_classifier(arousal_clf, x_a_test, y_a_test, ['LA', 'HA'])

    arousal_svm_table['Participant'].append(participant_id)
    arousal_svm_table['Classifier'].append('Arousal')
    arousal_svm_table['Algorithme'].append('SVM')
    arousal_svm_table['Features'].append(components)
    arousal_svm_table['Mean'].append(round(np.nanmean(scores), 2))
    arousal_svm_table['Std'].append(round(np.nanstd(scores), 2))
    arousal_svm_table['Max'].append(round(max(scores), 2))
    arousal_svm_table['Min'].append(round(min(scores), 2))
    arousal_svm_table['Chance Level'].append(chance_arousal)
    arousal_svm_table['F1 Score'].append(f1)
    arousal_svm_table['MCC'].append(mcc)
    arousal_svm_table['Scores'].append(np.round(scores, 2))

    # Valence classifier
    df_v = pd.DataFrame(y_v)
    v_weights = df_v.value_counts(normalize=True)
    weights = {'HV': v_weights['HV'], 'LV': v_weights['LV']}
    if grid_optimization:
        best_params = retrieve_best_parameters_grid_search_svm(parameter_space, X_v, y_v, weights, d_function='ovo')
        kernel = best_params['kernel']
        C = best_params['C']
        gamma = best_params['gamma']
    valence_clf = svm.SVC(kernel=kernel, probability=True, C=C, class_weight=weights, gamma=gamma)
    scores = cross_val_score(valence_clf, X_v, y_v, cv=cv, scoring='balanced_accuracy')
    mcc = cross_val_score(valence_clf, X_v, y_v, cv=cv, scoring=mcc_scoring)
    f1 = cross_val_score(valence_clf, X_v, y_v, cv=cv, scoring='f1_weighted')
    mcc = np.nanmean(mcc)
    f1 = np.nanmean(f1)
    print(
        "Valence SVM: %0.2f accuracy with a standard deviation of %0.2f, Max of %0.2f and Min of %0.2f, MCC %0.2f " % (
        np.nanmean(scores), np.nanstd(scores), max(scores), min(scores), mcc))
    if plotting:
        plot_kfold_roc_curve(valence_clf, X_v, y_v, cv, 'HV', "Valence")
        plot_confusion_matrix_for_classifier(valence_clf, x_v_test, y_v_test, ['LV', 'HV'])

    valence_svm_table['Participant'].append(participant_id)
    valence_svm_table['Classifier'].append('Valence')
    valence_svm_table['Algorithme'].append('SVM')
    valence_svm_table['Features'].append(components)
    valence_svm_table['Mean'].append(round(np.nanmean(scores), 2))
    valence_svm_table['Std'].append(round(np.nanstd(scores), 2))
    valence_svm_table['Max'].append(round(max(scores), 2))
    valence_svm_table['Min'].append(round(min(scores), 2))
    valence_svm_table['Chance Level'].append(chance_valence)
    valence_svm_table['F1 Score'].append(f1)
    valence_svm_table['MCC'].append(mcc)
    valence_svm_table['Scores'].append(np.round(scores, 2))

    # Valence-Arousal classifier
    df_va = pd.DataFrame(y_va)
    va_weights = df_va.value_counts(normalize=True)
    weights = dict()
    for class_key in va_weights.index.format():
        weights[class_key] = va_weights[class_key]
    if grid_optimization:
        best_params = retrieve_best_parameters_grid_search_svm(parameter_space, X_va, y_va, weights, d_function='ovr')
        kernel = best_params['kernel']
        C = best_params['C']
        gamma = best_params['gamma']
    va_clf = svm.SVC(kernel=kernel, decision_function_shape="ovr", probability=True, C=C, class_weight=weights,
                     gamma=gamma)
    scores = cross_val_score(va_clf, X_va, y_va, cv=cv, scoring='balanced_accuracy')
    mcc = cross_val_score(va_clf, X_va, y_va, cv=cv, scoring=mcc_scoring)
    f1 = cross_val_score(va_clf, X_va, y_va, cv=cv, scoring='f1_weighted')
    mcc = np.nanmean(mcc)
    f1 = np.nanmean(f1)
    print("VA SVM: %0.2f accuracy with a standard deviation of %0.2f, Max of %0.2f and Min of %0.2f , MCC %0.2f " % (
    np.nanmean(scores), np.nanstd(scores), max(scores), min(scores), mcc))
    va_svm_table['Participant'].append(participant_id)
    va_svm_table['Classifier'].append('Valence-Arousal')
    va_svm_table['Algorithme'].append('SVM')
    va_svm_table['Features'].append(components)
    va_svm_table['Mean'].append(round(np.nanmean(scores), 2))
    va_svm_table['Std'].append(round(np.nanstd(scores), 2))
    va_svm_table['Max'].append(round(max(scores), 2))
    va_svm_table['Min'].append(round(min(scores), 2))
    va_svm_table['Chance Level'].append(0)
    va_svm_table['F1 Score'].append(f1)
    va_svm_table['MCC'].append(mcc)
    va_svm_table['Scores'].append(np.round(scores, 2))

    # MLP Cross-validation
    solver = 'lbfgs'
    hidden_layer_sizes = (4, 4)
    learning_rate = 'adaptive'
    alpha = 0.0001
    activation = 'relu'
    parameter_space = {
        'hidden_layer_sizes': [(10,), (5, 2,), (4, 4,), (10, 4, 2)],
        'activation': ['relu'],
        'solver': ['lbfgs'],
        'alpha': [0.00001, 0.001, 0.01, 0.1, 1, 10],
        'learning_rate': ['adaptive'],
    }

    # Arousal classifiers
    if grid_optimization:
        best_params = retrieve_best_parameters_grid_search_mlp(parameter_space, X_a, y_a, max_iter=10000)
        hidden_layer_sizer = best_params['hidden_layer_sizes']
        learning_rate = best_params['learning_rate']
        alpha = best_params['alpha']
        activation = best_params['activation']
    arousal_clf = MLPClassifier(solver=solver, alpha=alpha, activation=activation, learning_rate=learning_rate,
                                hidden_layer_sizes=hidden_layer_sizes, random_state=2020, max_iter=10000)
    scores = cross_val_score(arousal_clf, X_a, y_a, cv=cv, scoring='balanced_accuracy')
    mcc = cross_val_score(arousal_clf, X_a, y_a, cv=cv, scoring=mcc_scoring)
    f1 = cross_val_score(arousal_clf, X_a, y_a, cv=cv, scoring='f1_weighted')
    mcc = np.nanmean(mcc)
    f1 = np.nanmean(f1)
    print(
        "Arousal MLP: %0.2f accuracy with a standard deviation of %0.2f, Max of %0.2f and Min of %0.2f, MCC %0.2f " % (
        np.nanmean(scores), np.nanstd(scores), max(scores), min(scores), mcc))
    if plotting:
        plot_kfold_roc_curve(arousal_clf, X_a, y_a, cv, 'HA', "Arousal")
        plot_confusion_matrix_for_classifier(arousal_clf, x_a_test, y_a_test, ['LA', 'HA'])

    arousal_mlp_table['Participant'].append(participant_id)
    arousal_mlp_table['Classifier'].append('Arousal')
    arousal_mlp_table['Algorithme'].append('MLP')
    arousal_mlp_table['Features'].append(components)
    arousal_mlp_table['Mean'].append(round(np.nanmean(scores), 2))
    arousal_mlp_table['Std'].append(round(np.nanstd(scores), 2))
    arousal_mlp_table['Max'].append(round(max(scores), 2))
    arousal_mlp_table['Min'].append(round(min(scores), 2))
    arousal_mlp_table['Chance Level'].append(chance_arousal)
    arousal_mlp_table['F1 Score'].append(f1)
    arousal_mlp_table['MCC'].append(mcc)
    arousal_mlp_table['Scores'].append(np.round(scores, 2))

    # Valence classifier
    if grid_optimization:
        best_params = retrieve_best_parameters_grid_search_mlp(parameter_space, X_v, y_v, max_iter=10000)
        hidden_layer_sizer = best_params['hidden_layer_sizes']
        learning_rate = best_params['learning_rate']
        alpha = best_params['alpha']
        activation = best_params['activation']
    valence_clf = MLPClassifier(solver=solver, alpha=alpha, activation=activation, learning_rate=learning_rate,
                                hidden_layer_sizes=hidden_layer_sizes, random_state=1, max_iter=10000)
    scores = cross_val_score(valence_clf, X_v, y_v, cv=cv, scoring='balanced_accuracy')
    mcc = cross_val_score(valence_clf, X_v, y_v, cv=cv, scoring=mcc_scoring)
    f1 = cross_val_score(valence_clf, X_v, y_v, cv=cv, scoring='f1_weighted')
    mcc = np.nanmean(mcc)
    f1 = np.nanmean(f1)
    print(
        "Valence MLP: %0.2f accuracy with a standard deviation of %0.2f, Max of %0.2f and Min of %0.2f, MCC %0.2f " % (
        np.nanmean(scores), np.nanstd(scores), max(scores), min(scores), mcc))
    if plotting:
        plot_kfold_roc_curve(valence_clf, X_v, y_v, cv, 'HV', "Valence")
        plot_confusion_matrix_for_classifier(valence_clf, x_v_test, y_v_test, ['LV', 'HV'])

    valence_mlp_table['Participant'].append(participant_id)
    valence_mlp_table['Classifier'].append('Valence')
    valence_mlp_table['Algorithme'].append('MLP')
    valence_mlp_table['Features'].append('')
    valence_mlp_table['Mean'].append(round(np.nanmean(scores), 2))
    valence_mlp_table['Std'].append(round(np.nanstd(scores), 2))
    valence_mlp_table['Max'].append(round(max(scores), 2))
    valence_mlp_table['Min'].append(round(min(scores), 2))
    valence_mlp_table['Chance Level'].append(chance_valence)
    valence_mlp_table['F1 Score'].append(f1)
    valence_mlp_table['MCC'].append(mcc)
    valence_mlp_table['Scores'].append(np.round(scores, 2))

    # Valence-Arousal classifier
    if grid_optimization:
        best_params = retrieve_best_parameters_grid_search_mlp(parameter_space, X_va, y_va, max_iter=10000)
        hidden_layer_sizer = best_params['hidden_layer_sizes']
        learning_rate = best_params['learning_rate']
        alpha = best_params['alpha']
        activation = best_params['activation']
    va_clf = MLPClassifier(solver=solver, alpha=alpha, activation=activation, learning_rate=learning_rate,
                           hidden_layer_sizes=hidden_layer_sizes, random_state=1, max_iter=10000)
    scores = cross_val_score(va_clf, X_va, y_va, cv=cv, scoring='balanced_accuracy')
    mcc = cross_val_score(va_clf, X_va, y_va, cv=cv, scoring=mcc_scoring)
    f1 = cross_val_score(va_clf, X_va, y_va, cv=cv, scoring='f1_weighted')
    mcc = np.nanmean(mcc)
    f1 = np.nanmean(f1)
    print("VA MLP: %0.2f accuracy with a standard deviation of %0.2f, Max of %0.2f and Min of %0.2f, MCC %0.2f " % (
    np.nanmean(scores), np.nanstd(scores), max(scores), min(scores), mcc))
    va_mlp_table['Participant'].append(participant_id)
    va_mlp_table['Classifier'].append('Valence-Arousal')
    va_mlp_table['Algorithme'].append('MLP')
    va_mlp_table['Features'].append('')
    va_mlp_table['Mean'].append(round(np.nanmean(scores), 2))
    va_mlp_table['Std'].append(round(np.nanstd(scores), 2))
    va_mlp_table['Max'].append(round(max(scores), 2))
    va_mlp_table['Min'].append(round(min(scores), 2))
    va_mlp_table['Chance Level'].append(0)
    va_mlp_table['F1 Score'].append(f1)
    va_mlp_table['MCC'].append(mcc)
    va_mlp_table['Scores'].append(np.round(scores, 2))

    participant_toc_fwd = time()
    print(f"Participant processed in {participant_toc_fwd - participant_tic_fwd:.3f}s")
    print('------------------------------------------------------------------')

arousal_svm_table['Avg Mean Score'] = np.mean(arousal_svm_table['Mean'])
arousal_svm_table['Max Mean Score'] = max(arousal_svm_table['Mean'])
arousal_svm_table['Min Mean Score'] = min(arousal_svm_table['Mean'])
arousal_svm_table['Avg Std'] = np.mean(arousal_svm_table['Std'])
arousal_svm_table['Avg Chance Level'] = np.mean(arousal_svm_table['Chance Level'])
arousal_svm_table['Avg F1 Score'] = np.mean(arousal_svm_table['F1 Score'])
arousal_svm_table['Avg MCC'] = np.mean(arousal_svm_table['MCC'])

valence_svm_table['Avg Mean Score'] = np.mean(valence_svm_table['Mean'])
valence_svm_table['Max Mean Score'] = max(valence_svm_table['Mean'])
valence_svm_table['Min Mean Score'] = min(valence_svm_table['Mean'])
valence_svm_table['Avg Std'] = np.mean(valence_svm_table['Std'])
valence_svm_table['Avg Chance Level'] = np.mean(valence_svm_table['Chance Level'])
valence_svm_table['Avg F1 Score'] = np.mean(valence_svm_table['F1 Score'])
valence_svm_table['Avg MCC'] = np.mean(valence_svm_table['MCC'])

va_svm_table['Avg Mean Score'] = np.mean(va_svm_table['Mean'])
va_svm_table['Max Mean Score'] = max(va_svm_table['Mean'])
va_svm_table['Min Mean Score'] = min(va_svm_table['Mean'])
va_svm_table['Avg Std'] = np.mean(va_svm_table['Std'])
va_svm_table['Avg Chance Level'] = np.mean(va_svm_table['Chance Level'])
va_svm_table['Avg F1 Score'] = np.mean(va_svm_table['F1 Score'])
va_svm_table['Avg MCC'] = np.mean(va_svm_table['MCC'])

arousal_mlp_table['Avg Mean Score'] = np.mean(arousal_mlp_table['Mean'])
arousal_mlp_table['Max Mean Score'] = max(arousal_mlp_table['Mean'])
arousal_mlp_table['Min Mean Score'] = min(arousal_mlp_table['Mean'])
arousal_mlp_table['Avg Std'] = np.mean(arousal_mlp_table['Std'])
arousal_mlp_table['Avg Chance Level'] = np.mean(arousal_mlp_table['Chance Level'])
arousal_mlp_table['Avg F1 Score'] = np.mean(arousal_mlp_table['F1 Score'])
arousal_mlp_table['Avg MCC'] = np.mean(arousal_mlp_table['MCC'])

valence_mlp_table['Avg Mean Score'] = np.mean(valence_mlp_table['Mean'])
valence_mlp_table['Max Mean Score'] = max(valence_mlp_table['Mean'])
valence_mlp_table['Min Mean Score'] = min(valence_mlp_table['Mean'])
valence_mlp_table['Avg Std'] = np.mean(valence_mlp_table['Std'])
valence_mlp_table['Avg Chance Level'] = np.mean(valence_mlp_table['Chance Level'])
valence_mlp_table['Avg F1 Score'] = np.mean(valence_mlp_table['F1 Score'])
valence_mlp_table['Avg MCC'] = np.mean(valence_mlp_table['MCC'])

va_mlp_table['Avg Mean Score'] = np.mean(va_mlp_table['Mean'])
va_mlp_table['Max Mean Score'] = max(va_mlp_table['Mean'])
va_mlp_table['Min Mean Score'] = min(va_mlp_table['Mean'])
va_mlp_table['Avg Std'] = np.mean(va_mlp_table['Std'])
va_mlp_table['Avg Chance Level'] = np.mean(va_mlp_table['Chance Level'])
va_mlp_table['Avg F1 Score'] = np.mean(va_mlp_table['F1 Score'])
va_mlp_table['Avg MCC'] = np.mean(va_mlp_table['MCC'])

if liking_class:
    liking_mlp_table['Avg Mean Score'] = np.mean(va_mlp_table['Mean'])
    liking_mlp_table['Max Mean Score'] = max(va_mlp_table['Mean'])
    liking_mlp_table['Min Mean Score'] = min(va_mlp_table['Mean'])
    liking_mlp_table['Avg Std'] = np.mean(va_mlp_table['Std'])




folder = 'PCA'
sort_criterion = 'MCC'
y_val = 'MCC'
ylim = (-1, 1)
# Plot ranking and Save results
plt.rcParams["figure.figsize"] = (14,6)
df = pd.DataFrame(data=arousal_svm_table)
df.set_index('Participant', inplace=True)
df.sort_values(sort_criterion, inplace=True, ascending=False)
df.plot(y=y_val, kind='bar', legend=False, title='Participants ranking for Arousal SVM, sorted by ' + sort_criterion, color='orange', ylim=ylim, ylabel=y_val)
df.to_csv('../data/results/SD/'+ folder +'/'+ str(time_window) + 's/arousal_svm_' + condition + '_top_'+ str(time_window) + 's_full.csv')

df = pd.DataFrame(data=valence_svm_table)
df.set_index('Participant', inplace=True)
df.sort_values(sort_criterion, inplace=True, ascending=False)
df.plot(y=y_val, kind='bar', legend=False, title='Participants ranking for Valence SVM, sorted by ' + sort_criterion, color='limegreen', ylim=ylim, ylabel=y_val)
df.to_csv('../data/results/SD/'+ folder +'/'+ str(time_window) + 's/valence_svm_' + condition + '_top_'+ str(time_window) + 's_full.csv')

df = pd.DataFrame(data=va_svm_table)
df.set_index('Participant', inplace=True)
df.sort_values(sort_criterion, inplace=True, ascending=False)
df.plot(y=y_val, kind='bar', legend=False, title='Participants ranking for Valence-Arousal SVM, sorted by ' + sort_criterion, color='red', ylim=ylim, ylabel=y_val)
df.to_csv('../data/results/SD/'+ folder +'/'+ str(time_window) + 's/va_svm_' + condition + '_top_'+ str(time_window) + 's_full.csv')

df = pd.DataFrame(data=arousal_mlp_table)
df.set_index('Participant', inplace=True)
df.sort_values(sort_criterion, inplace=True, ascending=False)
df.plot(y=y_val, kind='bar', legend=False, title='Participants ranking for Arousal MLP, sorted by ' + sort_criterion, color='orange', ylim=ylim, ylabel=y_val)
df.to_csv('../data/results/SD/'+ folder +'/'+ str(time_window) + 's/arousal_mlp_' + condition + '_top_'+ str(time_window) + 's_full.csv')

df = pd.DataFrame(data=valence_mlp_table)
df.set_index('Participant', inplace=True)
df.sort_values(sort_criterion, inplace=True, ascending=False)
df.plot(y=y_val, kind='bar', legend=False, title='Participants ranking for Valence MLP, sorted by ' + sort_criterion, color='limegreen', ylim=ylim, ylabel=y_val)
df.to_csv('../data/results/SD/'+ folder +'/'+ str(time_window) + 's/valence_mlp_' + condition + '_top_'+ str(time_window) + 's_full.csv')

df = pd.DataFrame(data=va_mlp_table)
df.set_index('Participant', inplace=True)
df.sort_values(sort_criterion, inplace=True, ascending=False)
df.plot(y=y_val, kind='bar', legend=False, title='Participants ranking for Valence-Arousal MLP, sorted by ' + sort_criterion, color='red', ylim=ylim, ylabel=y_val)
df.to_csv('../data/results/SD/'+ folder +'/'+ str(time_window) + 's/va_mlp_' + condition + '_top_'+ str(time_window) + 's_full.csv')

if liking_class:
    df = pd.DataFrame(data=liking_mlp_table)
    df.set_index('Participant', inplace=True)
    df.sort_values(sort_criterion, inplace=True, ascending=False)
    df.plot(y='Mean', kind='bar', legend=False, title='Participants ranking for Liking MLP', color='purple', ylim=(0,1), ylabel=y_val)
    df.to_csv('../data/results/SD/'+ folder +'/'+ str(time_window) + 's/liking_mlp_' + condition + '_top_'+ str(time_window) + 's_full.csv')