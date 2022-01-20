import copy
from datetime import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, matthews_corrcoef, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler

from classification.em_features_aggregator import get_arousal_labels, get_valence_labels, get_valence_arousal_labels, \
    get_arousal_class_labels, get_valence_class_labels, get_valence_arousal_class_labels, \
    get_discrete_feature_as_labels, get_neuromarker_1D, get_neuromarker_2D, get_continuous_eeg_feature, \
    get_discrete_feature_as_continuous, majority_chance_level
from classification.em_grid_search import retrieve_best_parameters_grid_search_svm, \
    retrieve_best_parameters_grid_search_mlp
from utils.em_data_loader import load_dataset
from utils.em_plotting import plot_labels_distribution, plot_kfold_roc_curve, plot_confusion_matrix_for_classifier

time_window = "5"
preprocessed_folder = '../data/preprocessed_' + time_window
prep_dataset = load_dataset(preprocessed_folder)
participants = prep_dataset.keys()

# condition = 'EO&EC'
user_labels = False
plotting = True
liking_class = False
normalize_labels = False
grid_optimization = False
n_top_features = 5
cv = StratifiedKFold(5, shuffle=False)
mcc_scoring = make_scorer(matthews_corrcoef)

template_table = {
    'Test Participant': [],
    'Condition': condition,
    'TimeWindow': time_window,
    'Classifier': [],
    'Features': [],
    'Algorithme': [],
    'Chance Level': [],
    'Train Mean Score': [],
    'Train Std': [],
    'Max Train': [],
    'Min Train': [],
    'Test Accuracy': [],
    'Test F1 Score': [],
    'MCC': [],
    'MCC Std': [],
    'CV MCC': [],
    'Train Scores': [],
    'Avg Chance Level': [],
    'Avg Chance Level Std': [],
    'Avg Train Score': [],
    'Avg Train Std': [],
    'Max Train Score': [],
    'Min Train Score': [],
    'Max Test Score': [],
    'Min Test Score': [],
    'Avg F1 Test Score': [],
    'Avg MCC Test': [],
    'Avg Test Accuracy': [],
    'Avg Test Accuracy Std': [],
    'Avg CV F1 Score': [],
    'Avg CV MCC': [],
    'Avg CV MCC Std': []

}

arousal_svm_table = copy.deepcopy(template_table)
valence_svm_table = copy.deepcopy(template_table)
arousal_mlp_table = copy.deepcopy(template_table)
valence_mlp_table = copy.deepcopy(template_table)

np.random.seed(4)
t_classes = ['class_1_', 'class_2_', 'class_3_', 'class_4_']
np.random.shuffle(t_classes)
print(t_classes)

participant_keys = [*prep_dataset]
for participant in participant_keys:
    participant_tic_fwd = time()
    train_participants = copy.deepcopy(participant_keys)
    train_participants.remove(participant)
    test_participants = [participant]
    print("Training with " + str(len(train_participants)) + " participants:", train_participants)
    print("Testing with participant", test_participants)
    participant_id = participant
    print("Processing", participant)

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

    # Features for Test participant
    # Labels
    test_arousal_labels = get_arousal_labels(prep_dataset, test_participants, condition, skip_qc=False,
                                             string_labels=True)
    test_valence_labels = get_valence_labels(prep_dataset, test_participants, condition, skip_qc=False,
                                             string_labels=True)
    test_va_labels = get_valence_arousal_labels(prep_dataset, test_participants, condition, skip_qc=False,
                                                string_labels=True)

    test_arousal_num_labels = get_arousal_labels(prep_dataset, test_participants, condition, skip_qc=False,
                                                 string_labels=False)
    test_valence_num_labels = get_valence_labels(prep_dataset, test_participants, condition, skip_qc=False,
                                                 string_labels=False)
    test_va_num_labels = get_valence_arousal_labels(prep_dataset, test_participants, condition, skip_qc=False,
                                                    string_labels=False)

    # Neuromarkers
    test_aw_idx = get_neuromarker_1D(prep_dataset, test_participants, condition, 'aw_idx', False)
    test_fmt_idx = get_neuromarker_1D(prep_dataset, test_participants, condition, 'fmt_idx', False)
    test_sasi_idx = get_neuromarker_2D(prep_dataset, test_participants, condition, 'sasi_idx', False)

    # Continuous EEG features
    test_theta_norm = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'norm_theta_pow', False)
    test_alpha_norm = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'norm_alpha_pow', False)
    test_beta_norm = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'norm_beta_pow', False)

    test_skewness_theta = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'skewness_theta',
                                                     False)
    test_skewness_alpha = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'skewness_alpha',
                                                     False)
    test_skewness_beta = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'skewness_beta', False)

    test_kurtosis_theta = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'kurtosis_theta',
                                                     False)
    test_kurtosis_alpha = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'kurtosis_alpha',
                                                     False)
    test_kurtosis_beta = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'kurtosis_beta', False)

    test_std_theta = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'std_theta', False)
    test_std_alpha = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'std_alpha', False)
    test_std_beta = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'std_beta', False)

    test_ratio_theta = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'ratio_theta', False)
    test_ratio_alpha = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'ratio_alpha', False)
    test_ratio_beta = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'ratio_beta', False)

    test_rsd_theta = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'rsd_theta', False)
    test_rsd_alpha = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'rsd_alpha', False)
    test_rsd_beta = get_continuous_eeg_feature(prep_dataset, test_participants, condition, 'rsd_beta', False)

    # Discrete features
    liking = get_discrete_feature_as_continuous(prep_dataset, train_participants, condition, 'liking', False)
    familiarity = get_discrete_feature_as_continuous(prep_dataset, train_participants, condition, 'familiarity', False)

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

    t_features = np.array([test_aw_idx, test_fmt_idx, test_sasi_idx[0], test_sasi_idx[1],
                           test_theta_norm[0], test_theta_norm[1],
                           test_alpha_norm[0], test_alpha_norm[1],
                           test_beta_norm[0], test_beta_norm[1],
                           test_skewness_theta[0], test_skewness_theta[1],
                           test_skewness_alpha[0], test_skewness_alpha[1],
                           test_skewness_beta[0], test_skewness_beta[1],
                           test_kurtosis_theta[0], test_kurtosis_theta[1],
                           test_kurtosis_alpha[0], test_kurtosis_alpha[1],
                           test_kurtosis_beta[0], test_kurtosis_beta[1],
                           test_std_theta[0], test_std_theta[1],
                           test_std_alpha[0], test_std_alpha[1],
                           test_std_beta[0], test_std_beta[1],
                           test_ratio_theta[0], test_ratio_theta[1],
                           test_ratio_alpha[0], test_ratio_alpha[1],
                           test_ratio_beta[0], test_ratio_beta[1],
                           test_rsd_theta[0], test_rsd_theta[1],
                           test_rsd_alpha[0], test_rsd_alpha[1],
                           test_rsd_beta[0], test_rsd_beta[1]

                           ])

    t_feature_names = np.array(['test_aw_idx', 'test_fmt_idx', 'test_sasi_idx_f4', 'test_sasi_idx_f3',
                                'test_theta_norm_f4', 'test_theta_norm_f3',
                                'test_alpha_norm_f4', 'test_alpha_norm_f3',
                                'test_beta_norm_f4', 'test_beta_norm_f3',
                                'test_skewness_theta_f4', 'test_skewness_theta_f3',
                                'test_skewness_alpha_f4', 'test_skewness_alpha_f3',
                                'test_skewness_beta_f4', 'test_skewness_beta_f3',
                                'test_kurtosis_theta_f4', 'test_kurtosis_theta_f3',
                                'test_kurtosis_alpha_f4', 'test_kurtosis_alpha_f3',
                                'test_kurtosis_beta_f4', 'test_kurtosis_beta_f3',
                                'test_std_theta_f4', 'test_std_theta_f3',
                                'test_std_alpha_f4', 'test_std_alpha_f3',
                                'test_std_beta_f4', 'test_std_beta_f3',
                                'test_ratio_theta_f4', 'test_ratio_theta_f3',
                                'test_ratio_alpha_f4', 'test_ratio_alpha_f3',
                                'test_ratio_beta_f4', 'test_ratio_beta_f3',
                                'test_rsd_theta_f4', 'test_rsd_theta_f3',
                                'test_rsd_alpha_f4', 'test_rsd_alpha_f3',
                                'test_rsd_beta_f4', 'test_rsd_beta_f3'
                                ])

    # PCA features selection
    features_table = {}
    for i in range(0, len(feature_names)):
        features_table[feature_names[i]] = features[i]
    raw_features_df = pd.DataFrame(features_table)
    X = raw_features_df.values

    t_features_table = {}
    for i in range(0, len(t_feature_names)):
        t_features_table[t_feature_names[i]] = t_features[i]
    t_raw_features_df = pd.DataFrame(t_features_table)
    X_test = t_raw_features_df.values

    # Scale the data before applying PCA
    scaler = MaxAbsScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    scaler = MaxAbsScaler()
    scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)

    # Select 12 components that on average represent 95% of the variability for each participant
    pca_95 = PCA(n_components=12, random_state=2020)
    pca_95.fit(X_scaled)
    X_pca_95 = pca_95.transform(X_scaled)
    components = []
    for i in range(0, X_pca_95.shape[1]):
        components.append("PC" + str(i + 1))
    components = np.array(components)
    pc_df = pd.DataFrame(X_pca_95, columns=components)

    t_pca_95 = PCA(n_components=12, random_state=2020)
    t_pca_95.fit(X_test_scaled)
    t_X_pca_95 = t_pca_95.transform(X_test_scaled)
    t_components = []
    for i in range(0, t_X_pca_95.shape[1]):
        t_components.append("PC" + str(i + 1))
    t_components = np.array(t_components)
    t_pc_df = pd.DataFrame(t_X_pca_95, columns=t_components)

    X_a = pc_df.values
    y_a = np.array(arousal_labels)

    X_v = pc_df.values
    y_v = np.array(valence_labels)

    X_va = pc_df.values
    y_va = np.array(va_labels)

    # Scaling the test data using MaxAbsScaler
    test_X_a = t_pc_df.values
    test_y_a = np.array(test_arousal_labels)

    test_X_v = t_pc_df.values
    test_y_v = np.array(test_valence_labels)

    test_X_va = t_pc_df
    test_y_va = np.array(test_va_labels)

    # SVM Cross-validation
    # SVM parameters
    C = 1000
    kernel = 'rbf'
    gamma = 0.1
    parameter_space = {
        'C': [0.1, 1, 10, 1000],
        'gamma': [10, 0.1, 0.01, 0.0001],
        'kernel': ['rbf']
    }
    # Arousal classifier

    # Calculate class weights beforehand
    df_a = pd.DataFrame(y_a)
    a_weights = df_a.value_counts(normalize=True)
    weights = {'HA': a_weights['HA'], 'LA': a_weights['LA']}
    if grid_optimization:
        best_params = retrieve_best_parameters_grid_search_svm(parameter_space, X_a, y_a, weights,
                                                               d_function='ovo')
        kernel = best_params['kernel']
        C = best_params['C']
        gamma = best_params['gamma']
    arousal_clf = svm.SVC(kernel=kernel, probability=True, C=C, class_weight=weights, gamma=gamma)

    scores = cross_val_score(arousal_clf, X_a, y_a, cv=cv, scoring='balanced_accuracy')
    cv_mcc = cross_val_score(arousal_clf, X_a, y_a, cv=cv, scoring=mcc_scoring)
    cv_f1 = cross_val_score(arousal_clf, X_a, y_a, cv=cv, scoring='f1_weighted')

    arousal_clf.fit(X_a, y_a)
    print(
        "Arousal SVM: %0.2f cross-validated accuracy with a standard deviation of %0.2f, Max of %0.2f and Min of %0.2f " % (
            np.nanmean(scores), np.nanstd(scores), max(scores), min(scores)))
    y_pred = arousal_clf.predict(test_X_a)
    accuracy = accuracy_score(test_y_a, y_pred)
    f1 = f1_score(test_y_a, y_pred, average='weighted')
    mcc = matthews_corrcoef(test_y_a, y_pred)
    if plotting:
        # ROC Curve
        plot_kfold_roc_curve(arousal_clf, X_a, y_a, cv, 'HA', "Arousal")
        # Confusion Matrix
        plot_confusion_matrix_for_classifier(arousal_clf, test_X_a, test_y_a, ['LA', 'HA'])

    print("Arousal SVM: %0.2f accuracy, %0.2f F1 score, %0.2f MCC" % (accuracy, f1, mcc))
    arousal_svm_table['Test Participant'].append(participant)
    arousal_svm_table['Classifier'].append('Arousal')
    arousal_svm_table['Algorithme'].append('SVM')
    arousal_svm_table['Features'].append('')
    arousal_svm_table['Chance Level'].append(chance_arousal)
    arousal_svm_table['Train Mean Score'].append(np.round(np.nanmean(scores), 2))
    arousal_svm_table['Train Std'].append(np.nanstd(scores))
    arousal_svm_table['Max Train'].append(np.round(max(scores), 2))
    arousal_svm_table['Min Train'].append(np.round(min(scores), 2))
    arousal_svm_table['Train Scores'].append(np.round(scores, 2))
    arousal_svm_table['Test Accuracy'].append(np.round(accuracy, 2))
    arousal_svm_table['Test F1 Score'].append(np.round(f1, 2))
    arousal_svm_table['MCC'].append(np.round(mcc, 2))
    arousal_svm_table['CV MCC'].append(np.round(cv_mcc, 2))

    # Valence classifier

    # Calculate class weights beforehand
    df_v = pd.DataFrame(y_v)
    v_weights = df_v.value_counts(normalize=True)
    weights = {'HV': v_weights['HV'], 'LV': v_weights['LV']}
    if grid_optimization:
        best_params = retrieve_best_parameters_grid_search_svm(parameter_space, X_a, y_a, weights,
                                                               d_function='ovo')
        kernel = best_params['kernel']
        C = best_params['C']
        gamma = best_params['gamma']

    valence_clf = svm.SVC(kernel=kernel, probability=True, C=C, class_weight=weights)

    scores = cross_val_score(valence_clf, X_v, y_v, cv=cv, scoring='balanced_accuracy')
    cv_mcc = cross_val_score(valence_clf, X_v, y_v, cv=cv, scoring=mcc_scoring)
    cv_f1 = cross_val_score(valence_clf, X_v, y_v, cv=cv, scoring='f1_weighted')

    valence_clf.fit(X_v, y_v)

    print(
        "Valence SVM: %0.2f cross-validated accuracy with a standard deviation of %0.2f, Max of %0.2f and Min of %0.2f " % (
            np.nanmean(scores), np.nanstd(scores), max(scores), min(scores)))
    y_pred = valence_clf.predict(test_X_v)
    accuracy = accuracy_score(test_y_v, y_pred)
    f1 = f1_score(test_y_v, y_pred, average='weighted')
    mcc = matthews_corrcoef(test_y_v, y_pred)
    print("Valence SVM: %0.2f accuracy, %0.2f F1 score, %0.2f MCC" % (accuracy, f1, mcc))
    if plotting:
        # ROC Curve
        plot_kfold_roc_curve(valence_clf, X_v, y_v, cv, 'HV', "Valence")
        # Confusion Matrix
        plot_confusion_matrix_for_classifier(valence_clf, test_X_v, test_y_v, ['LV', 'HV'])
    valence_svm_table['Test Participant'].append(participant)
    valence_svm_table['Classifier'].append('Valence')
    valence_svm_table['Algorithme'].append('SVM')
    valence_svm_table['Features'].append('')
    valence_svm_table['Chance Level'].append(chance_valence)
    valence_svm_table['Train Mean Score'].append(np.round(np.nanmean(scores), 2))
    valence_svm_table['Train Std'].append(np.nanstd(scores))
    valence_svm_table['Max Train'].append(np.round(max(scores), 2))
    valence_svm_table['Min Train'].append(np.round(min(scores), 2))
    valence_svm_table['Train Scores'].append(np.round(scores, 2))
    valence_svm_table['Test Accuracy'].append(np.round(accuracy, 2))
    valence_svm_table['Test F1 Score'].append(np.round(f1, 2))
    valence_svm_table['MCC'].append(np.round(mcc, 2))
    valence_svm_table['CV MCC'].append(np.round(cv_mcc, 2))

    # MLP Classifiers

    # MLP hyperparameters
    solver = 'lbfgs'
    activation = 'relu'
    learning_rate = 'adaptive'
    hidden_layer_sizes = (4, 4)
    alpha = 0.0001

    max_iter = 10000
    parameter_space = {
        'hidden_layer_sizes': [(5, 2,), (4, 4,), (10, 4, 2)],
        'activation': ['relu'],
        'solver': ['lbfgs'],
        'alpha': [0.00001, 0.01, 1],
        'learning_rate': ['adaptive'],
    }

    # Arousal classifier
    alpha = 10
    hidden_layer_sizes = (10,)
    if grid_optimization:
        best_params = retrieve_best_parameters_grid_search_mlp(parameter_space, X_a, y_a, max_iter=10000)
        hidden_layer_sizes = best_params['hidden_layer_sizes']
        learning_rate = best_params['learning_rate']
        alpha = best_params['alpha']
        activation = best_params['activation']
    arousal_clf = MLPClassifier(solver=solver, alpha=alpha, activation=activation, learning_rate=learning_rate,
                                hidden_layer_sizes=hidden_layer_sizes, random_state=1, max_iter=max_iter)

    scores = cross_val_score(arousal_clf, X_a, y_a, cv=cv, scoring='balanced_accuracy')
    cv_mcc = cross_val_score(arousal_clf, X_a, y_a, cv=cv, scoring=mcc_scoring)
    cv_f1 = cross_val_score(arousal_clf, X_a, y_a, cv=cv, scoring='f1_weighted')

    arousal_clf.fit(X_a, y_a)

    print(
        "Arousal MLP: %0.2f cross-validated accuracy with a standard deviation of %0.2f, Max of %0.2f and Min of %0.2f " % (
            scores.mean(), scores.std(), max(scores), min(scores)))
    y_pred = arousal_clf.predict(test_X_a)
    accuracy = accuracy_score(test_y_a, y_pred)
    f1 = f1_score(test_y_a, y_pred, average='weighted')
    mcc = matthews_corrcoef(test_y_a, y_pred)
    if plotting:
        # ROC Curve
        plot_kfold_roc_curve(arousal_clf, X_a, y_a, cv, 'HA', "Arousal")
        # Confusion Matrix
        plot_confusion_matrix_for_classifier(arousal_clf, test_X_a, test_y_a, ['LA', 'HA'])
    print("Arousal MLP: %0.2f accuracy, %0.2f, %0.2f MCC" % (accuracy, f1, mcc))
    arousal_mlp_table['Test Participant'].append(participant)
    arousal_mlp_table['Classifier'].append('Arousal')
    arousal_mlp_table['Algorithme'].append('MLP')
    arousal_mlp_table['Features'].append('')
    arousal_mlp_table['Chance Level'].append(chance_arousal)
    arousal_mlp_table['Train Mean Score'].append(np.round(np.nanmean(scores), 2))
    arousal_mlp_table['Train Std'].append(np.std(scores))
    arousal_mlp_table['Max Train'].append(np.round(max(scores), 2))
    arousal_mlp_table['Min Train'].append(np.round(min(scores), 2))
    arousal_mlp_table['Train Scores'].append(np.round(scores, 2))
    arousal_mlp_table['Test Accuracy'].append(np.round(accuracy, 2))
    arousal_mlp_table['Test F1 Score'].append(np.round(f1, 2))
    arousal_mlp_table['MCC'].append(np.round(mcc, 2))
    arousal_mlp_table['CV MCC'].append(np.round(cv_mcc, 2))

    # Valence classifier
    alpha = 10
    hidden_layer_sizes = (10, 4, 2)

    if grid_optimization:
        best_params = retrieve_best_parameters_grid_search_mlp(parameter_space, X_v, y_v, max_iter=10000)
        hidden_layer_sizes = best_params['hidden_layer_sizes']
        learning_rate = best_params['learning_rate']
        alpha = best_params['alpha']
        activation = best_params['activation']
    valence_clf = MLPClassifier(solver=solver, alpha=alpha, activation=activation, learning_rate=learning_rate,
                                hidden_layer_sizes=hidden_layer_sizes, random_state=1, max_iter=max_iter)

    scores = cross_val_score(valence_clf, X_v, y_v, cv=cv, scoring='balanced_accuracy')
    cv_mcc = cross_val_score(valence_clf, X_v, y_v, cv=cv, scoring=mcc_scoring)
    cv_f1 = cross_val_score(valence_clf, X_v, y_v, cv=cv, scoring='f1_weighted')

    valence_clf.fit(X_v, y_v)
    print(
        "Valence MLP: %0.2f cross-validated accuracy with a standard deviation of %0.2f, Max of %0.2f and Min of %0.2f " % (
            np.nanmean(scores), np.nanstd(scores), max(scores), min(scores)))
    y_pred = valence_clf.predict(test_X_v)
    accuracy = accuracy_score(test_y_v, y_pred)
    f1 = f1_score(test_y_v, y_pred, average='weighted')
    mcc = matthews_corrcoef(test_y_v, y_pred)
    if plotting:
        # ROC Curve
        plot_kfold_roc_curve(valence_clf, X_v, y_v, cv, 'HV', "Valence")
        # Confusion Matrix
        plot_confusion_matrix_for_classifier(valence_clf, test_X_v, test_y_v, ['LV', 'HV'])
    print("Valence MLP: %0.2f accuracy, %0.2f F1 score, %0.2f MCC" % (accuracy, f1, mcc))
    valence_mlp_table['Test Participant'].append(participant)
    valence_mlp_table['Classifier'].append('Valence')
    valence_mlp_table['Algorithme'].append('MLP')
    valence_mlp_table['Features'].append('')
    valence_mlp_table['Chance Level'].append(chance_valence)
    valence_mlp_table['Train Mean Score'].append(np.round(np.nanmean(scores), 2))
    valence_mlp_table['Train Std'].append(np.nanstd(scores))
    valence_mlp_table['Max Train'].append(np.round(max(scores), 2))
    valence_mlp_table['Min Train'].append(np.round(min(scores), 2))
    valence_mlp_table['Train Scores'].append(np.round(scores, 2))
    valence_mlp_table['Test Accuracy'].append(np.round(accuracy, 2))
    valence_mlp_table['Test F1 Score'].append(np.round(f1, 2))
    valence_mlp_table['MCC'].append(np.round(mcc, 2))
    valence_mlp_table['CV MCC'].append(np.round(cv_mcc, 2))

    participant_toc_fwd = time()
    print(f"Participant processed in {participant_toc_fwd - participant_tic_fwd:.3f}s")
    print('------------------------------------------------------------------')

# Computing results for the .csv tables
arousal_svm_table['Avg Test Accuracy'] = np.nanmean(arousal_svm_table['Test Accuracy'])
arousal_svm_table['Avg Test Accuracy Std'] = np.nanmean(arousal_svm_table['Test Accuracy Std'])
arousal_svm_table['Max Test Score'] = max(arousal_svm_table['Test Accuracy'])
arousal_svm_table['Min Test Score'] = min(arousal_svm_table['Test Accuracy'])
arousal_svm_table['Avg F1 Test Score'] = np.nanmean(arousal_svm_table['Test F1 Score'])
arousal_svm_table['Avg Train Score'] = np.nanmean(arousal_svm_table['Train Mean Score'])
arousal_svm_table['Max Train Score'] = max(arousal_svm_table['Train Mean Score'])
arousal_svm_table['Min Train Score'] = min(arousal_svm_table['Train Mean Score'])
arousal_svm_table['Avg Train Std'] = np.nanmean(arousal_svm_table['Train Std'])
arousal_svm_table['Avg Chance Level'] = np.nanmean(arousal_svm_table['Chance Level'])
arousal_svm_table['Avg Chance Level Std'] = np.nanstd(arousal_svm_table['Chance Level'])
arousal_svm_table['Avg MCC'] = np.nanmean(arousal_svm_table['MCC'])
arousal_svm_table['Avg MCC Std'] = np.nanstd(arousal_svm_table['MCC'])
arousal_svm_table['Avg CV MCC'] = np.nanmean(arousal_svm_table['CV MCC'])
arousal_svm_table['Avg CV MCC Std'] = np.nanstd(arousal_svm_table['CV MCC'])

valence_svm_table['Avg Test Accuracy'] = np.nanmean(valence_svm_table['Test Accuracy'])
valence_svm_table['Avg Test Accuracy Std'] = np.nanmean(valence_svm_table['Test Accuracy Std'])
valence_svm_table['Max Test Score'] = max(valence_svm_table['Test Accuracy'])
valence_svm_table['Min Test Score'] = min(valence_svm_table['Test Accuracy'])
valence_svm_table['Avg F1 Test Score'] = np.nanmean(valence_svm_table['Test F1 Score'])
valence_svm_table['Avg Train Score'] = np.nanmean(valence_svm_table['Train Mean Score'])
valence_svm_table['Max Train Score'] = max(valence_svm_table['Train Mean Score'])
valence_svm_table['Min Train Score'] = min(valence_svm_table['Train Mean Score'])
valence_svm_table['Avg Train Std'] = np.nanmean(valence_svm_table['Train Std'])
valence_svm_table['Avg Chance Level'] = np.nanmean(valence_svm_table['Chance Level'])
valence_svm_table['Avg Chance Level Std'] = np.nanstd(valence_svm_table['Chance Level'])
valence_svm_table['Avg MCC'] = np.nanmean(valence_svm_table['MCC'])
valence_svm_table['Avg MCC Std'] = np.nanstd(valence_svm_table['MCC'])
valence_svm_table['Avg CV MCC'] = np.nanmean(valence_svm_table['CV MCC'])
valence_svm_table['Avg CV MCC Std'] = np.nanstd(valence_svm_table['CV MCC'])

arousal_mlp_table['Avg Test Accuracy'] = np.nanmean(arousal_mlp_table['Test Accuracy'])
arousal_mlp_table['Avg Test Accuracy Std'] = np.nanmean(arousal_mlp_table['Test Accuracy Std'])
arousal_mlp_table['Max Test Score'] = max(arousal_mlp_table['Test Accuracy'])
arousal_mlp_table['Min Test Score'] = min(arousal_mlp_table['Test Accuracy'])
arousal_mlp_table['Avg F1 Test Score'] = np.nanmean(arousal_mlp_table['Test F1 Score'])
arousal_mlp_table['Avg Train Score'] = np.nanmean(arousal_mlp_table['Train Mean Score'])
arousal_mlp_table['Max Train Score'] = max(arousal_mlp_table['Train Mean Score'])
arousal_mlp_table['Min Train Score'] = min(arousal_mlp_table['Train Mean Score'])
arousal_mlp_table['Avg Train Std'] = np.nanmean(arousal_mlp_table['Train Std'])
arousal_mlp_table['Avg Chance Level'] = np.nanmean(arousal_mlp_table['Chance Level'])
arousal_mlp_table['Avg Chance Level Std'] = np.nanstd(arousal_mlp_table['Chance Level'])
arousal_mlp_table['Avg MCC'] = np.nanmean(arousal_mlp_table['MCC'])
arousal_mlp_table['Avg MCC Std'] = np.nanstd(arousal_mlp_table['MCC'])
arousal_mlp_table['Avg CV MCC'] = np.nanmean(arousal_mlp_table['CV MCC'])
arousal_mlp_table['Avg CV MCC Std'] = np.nanstd(arousal_mlp_table['CV MCC'])

valence_mlp_table['Avg Test Accuracy'] = np.nanmean(valence_mlp_table['Test Accuracy'])
valence_mlp_table['Avg Test Accuracy Std'] = np.nanmean(valence_mlp_table['Test Accuracy Std'])
valence_mlp_table['Max Test Score'] = max(valence_mlp_table['Test Accuracy'])
valence_mlp_table['Min Test Score'] = min(valence_mlp_table['Test Accuracy'])
valence_mlp_table['Avg F1 Test Score'] = np.nanmean(valence_mlp_table['Test F1 Score'])
valence_mlp_table['Avg Train Score'] = np.nanmean(valence_mlp_table['Train Mean Score'])
valence_mlp_table['Max Train Score'] = max(valence_mlp_table['Train Mean Score'])
valence_mlp_table['Min Train Score'] = min(valence_mlp_table['Train Mean Score'])
valence_mlp_table['Avg Train Std'] = np.nanmean(valence_mlp_table['Train Std'])
valence_mlp_table['Avg Chance Level'] = np.nanmean(valence_mlp_table['Chance Level'])
valence_mlp_table['Avg Chance Level Std'] = np.nanstd(valence_mlp_table['Chance Level'])
valence_mlp_table['Avg MCC'] = np.nanmean(valence_mlp_table['MCC'])
valence_mlp_table['Avg MCC Std'] = np.nanmean(valence_mlp_table['MCC'])
valence_mlp_table['Avg CV MCC'] = np.nanmean(valence_mlp_table['CV MCC'])
valence_mlp_table['Avg CV MCC Std'] = np.nanmean(valence_mlp_table['CV MCC'])

folder = 'PCA'
sort_criterion = 'Mean'
y_val = 'Mean'
ylim = (-1, 1)
# Plot ranking and Save results
df = pd.DataFrame(data=arousal_svm_table)
df.set_index('Test Participant', inplace=True)
df.sort_values(sort_criterion, inplace=True, ascending=False)
df.plot(y=y_val, kind='bar', legend=False, title='Participants ranking for Arousal SVM, sorted by ' + sort_criterion, color='orange', ylim=ylim,ylabel=y_val)
df.to_csv('../data/results/SI/PCA/'+ str(time_window) + 's/arousal_svm_' + condition + '_top_'+ str(time_window) + 's_full.csv')

df = pd.DataFrame(data=valence_svm_table)
df.set_index('Test Participant', inplace=True)
df.sort_values(sort_criterion, inplace=True, ascending=False)
df.plot(y=y_val, kind='bar', legend=False, title='Participants ranking for Valence SVM, sorted by ' + sort_criterion, color='limegreen', ylim=ylim,ylabel=y_val)
df.to_csv('../data/results/SI/PCA/'+ str(time_window) + 's/valence_svm_' + condition + '_top_'+ str(time_window) + 's_full.csv')

df = pd.DataFrame(data=arousal_mlp_table)
df.set_index('Test Participant', inplace=True)
df.sort_values(sort_criterion, inplace=True, ascending=False)
df.plot(y=y_val, kind='bar', legend=False, title='Participants ranking for Arousal MLP, sorted by ' + sort_criterion, color='orange', ylim=ylim,ylabel=y_val)
df.to_csv('../data/results/SI/PCA/'+ str(time_window) + 's/arousal_mlp_' + condition + '_top_'+ str(time_window) + 's_full.csv')

df = pd.DataFrame(data=valence_mlp_table)
df.set_index('Test Participant', inplace=True)
df.sort_values(sort_criterion, inplace=True, ascending=False)
df.plot(y=y_val, kind='bar', legend=False, title='Participants ranking for Valence MLP, sorted by ' + sort_criterion, color='limegreen', ylim=ylim,ylabel=y_val)
df.to_csv('../data/results/SI/PCA/'+ str(time_window) + 's/valence_mlp_' + condition + '_top_'+ str(time_window) + 's_full.csv')
