import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from algorithms.linear_regression import LinearRegression


def test_recovers_known_coefficients_with_no_noise():
    rng = np.random.default_rng(0)
    X = rng.uniform(-10, 10, size=(200, 2))
    true_coefficients = np.array([3.0, -2.0])
    true_intercept = 5.0
    y = X @ true_coefficients + true_intercept

    model = LinearRegression(learning_rate=0.5, n_iterations=2000).fit(X, y)

    assert np.allclose(model.coefficients, true_coefficients, atol=0.05)
    assert abs(model.intercept - true_intercept) < 0.1


def test_matches_sklearn_on_noisy_data():
    rng = np.random.default_rng(1)
    X = rng.uniform(-5, 5, size=(300, 3))
    true_coefficients = np.array([1.5, -4.0, 0.5])
    y = X @ true_coefficients + 2.0 + rng.normal(0, 1.0, size=300)

    ours = LinearRegression(learning_rate=0.5, n_iterations=3000).fit(X, y)
    sklearn_model = SklearnLinearRegression().fit(X, y)

    assert np.allclose(ours.coefficients, sklearn_model.coef_, atol=0.1)
    assert abs(ours.intercept - sklearn_model.intercept_) < 0.2


def test_predict_matches_fitted_line():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])  # y = 2x + 1, exact
    model = LinearRegression(learning_rate=0.3, n_iterations=2000).fit(X, y)
    predictions = model.predict(np.array([[4.0], [5.0]]))
    assert np.allclose(predictions, [9.0, 11.0], atol=0.1)
