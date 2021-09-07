import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from utils.em_plotting import plot_linear_regression


def lin_regression(x_train, y_train, x_test, y_test, d_label='Y', i_label='X', plot=False):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    msq = mean_squared_error(y_test, y_pred)
    coef = regr.coef_
    r2 = r2_score(y_test, y_pred)

    print("Prediction with " + d_label + " as dependent variable and " + i_label + " as independent variable")
    # The coefficients
    print('Coefficients: \n', coef)
    # The mean squared error
    print('Mean squared error: %.2f' % msq)
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f' % r2)

    # Plot outputs
    if plot:
        plot_linear_regression(x_test, y_test, y_pred, d_label, i_label)

    return msq, coef, r2


def valence_and_neuromarker(prep_dataset, train_participants, test_participants, trial_class, condition, neuromarker, skip_qc=True):
    """ Valence annotations and a chosen neuromarker"""
    x_train = np.array([])
    y_train = np.array([])
    x_test = np.array([])
    y_test = np.array([])
    for participant_id in train_participants:
        trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'A']
        if not trial['bad_quality'] or skip_qc:
            idx = trial['features'][neuromarker]
            avg_valence = trial['features']['avg_x']
            x_train = np.concatenate((x_train, np.array(avg_valence)))
            y_train = np.concatenate((y_train, np.array(idx)))

        trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'B']
        if not trial['bad_quality'] or skip_qc:
            idx = trial['features'][neuromarker]
            avg_valence = trial['features']['avg_x']
            x_train = np.concatenate((x_train, np.array(avg_valence)))
            y_train = np.concatenate((y_train, np.array(idx)))

    for participant_id in test_participants:
        trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'A']
        if not trial['bad_quality'] or skip_qc:
            idx = trial['features'][neuromarker]
            avg_valence = trial['features']['avg_x']
            x_test = np.concatenate((x_test, np.array(avg_valence)))
            y_test = np.concatenate((y_test, np.array(idx)))

        trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'B']
        if not trial['bad_quality'] or skip_qc:
            idx = trial['features'][neuromarker]
            avg_valence = trial['features']['avg_x']
            x_test = np.concatenate((x_test, np.array(avg_valence)))
            y_test = np.concatenate((y_test, np.array(idx)))
    return x_train, y_train, x_test, y_test


def arousal_and_neuromarker(prep_dataset, train_participants, test_participants, trial_class, condition, neuromarker, skip_qc=True):
    """ Arousal annotations and a chosen neuromarker"""
    x_train = np.array([])
    y_train = np.array([])
    x_test = np.array([])
    y_test = np.array([])
    for participant_id in train_participants:
        trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'A']
        if not trial['bad_quality'] or skip_qc:
            idx = trial['features'][neuromarker]
            avg_arousal = trial['features']['avg_y']
            x_train = np.concatenate((x_train, np.array(avg_arousal)))
            y_train = np.concatenate((y_train, np.array(idx)))

        trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'B']
        if not trial['bad_quality'] or skip_qc:
            idx = trial['features'][neuromarker]
            avg_arousal = trial['features']['avg_y']
            x_train = np.concatenate((x_train, np.array(avg_arousal)))
            y_train = np.concatenate((y_train, np.array(idx)))

    for participant_id in test_participants:
        trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'A']
        if not trial['bad_quality'] or skip_qc:
            idx = trial['features'][neuromarker]
            avg_arousal = trial['features']['avg_y']
            x_test = np.concatenate((x_test, np.array(avg_arousal)))
            y_test = np.concatenate((y_test, np.array(idx)))

        trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'B']
        if not trial['bad_quality'] or skip_qc:
            idx = trial['features'][neuromarker]
            avg_arousal = trial['features']['avg_y']
            x_test = np.concatenate((x_test, np.array(avg_arousal)))
            y_test = np.concatenate((y_test, np.array(idx)))
    return x_train, y_train, x_test, y_test


# NOT working, to adjust for automation
def prepare_regression_data_annotations(prep_dataset,train_participants, test_participants, independent, dependent, condition,
                                        trial_class):
    x_train = np.array([])
    y_train = np.array([])
    x_test = np.array([])
    y_test = np.array([])
    for participant_id in train_participants:
        i = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'A']['features'][independent]
        d = prep_dataset[participant_id]['trials']['EO' + '/' + trial_class + 'A']['features'][dependent]
        x_train = np.concatenate((x_train, np.array(i)))
        y_train = np.concatenate((y_train, np.array(d)))

        i = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'B']['features'][independent]
        d = prep_dataset[participant_id]['trials']['EO' + '/' + trial_class + 'B']['features'][dependent]
        x_train = np.concatenate((x_train, np.array(i)))
        y_train = np.concatenate((y_train, np.array(d)))

    for participant_id in test_participants:
        i = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'A']['features'][independent]
        d = prep_dataset[participant_id]['trials']['EO' + '/' + trial_class + 'A']['features'][dependent]
        x_test = np.concatenate((x_test, np.array(i)))
        y_test = np.concatenate((y_test, np.array(d)))

        i = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'B']['features'][independent]
        d = prep_dataset[participant_id]['trials']['EO' + '/' + trial_class + 'B']['features'][dependent]
        x_test = np.concatenate((x_test, np.array(i)))
        y_test = np.concatenate((y_test, np.array(d)))

    return x_train, y_train, x_test, y_test


def liking_familiarity_correlation(prep_dataset, train_participants, test_participants, condition, trial_class, dependent, independent, skip_qc=True):
    """ Correlations between liking and familiarity """
    x_train = np.array([])
    y_train = np.array([])
    x_test = np.array([])
    y_test = np.array([])
    for participant_id in train_participants:
        trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'A']
        if not trial['bad_quality'] or skip_qc:
            i = trial['features'][independent]
            d = trial['features'][dependent]
            x_train = np.append(x_train, int(i))
            y_train = np.append(y_train, int(d))

        trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'B']
        if not trial['bad_quality'] or skip_qc:
            i = trial['features'][independent]
            d = trial['features'][dependent]
            x_train = np.append(x_train, int(i))
            y_train = np.append(y_train, int(d))

    for participant_id in test_participants:
        trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'A']
        if not trial['bad_quality'] or skip_qc:
            i = trial['features'][independent]
            d = trial['features'][dependent]
            x_test = np.append(x_test, int(i))
            y_test = np.append(y_test, int(d))

        trial = prep_dataset[participant_id]['trials'][condition + '/' + trial_class + 'B']
        if not trial['bad_quality'] or skip_qc:
            i = trial['features'][independent]
            d = trial['features'][dependent]
            x_test = np.append(x_test, int(i))
            y_test = np.append(y_test, int(d))
    return x_train, y_train, x_test, y_test
