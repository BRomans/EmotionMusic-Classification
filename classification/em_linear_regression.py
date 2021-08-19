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

