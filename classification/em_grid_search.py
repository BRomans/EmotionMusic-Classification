from sklearn import svm
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


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
