import numpy as np

from algorithms.bayesian_regression import BayesianLinearRegression


def test_posterior_mean_approaches_true_coefficients_as_data_grows():
    rng = np.random.default_rng(0)
    true_weights = np.array([2.0, -1.5])  # [intercept, slope]

    small_X = rng.uniform(-3, 3, size=(5, 1))
    small_y = true_weights[0] + true_weights[1] * small_X[:, 0] + rng.normal(0, 0.5, size=5)

    large_X = rng.uniform(-3, 3, size=(500, 1))
    large_y = true_weights[0] + true_weights[1] * large_X[:, 0] + rng.normal(0, 0.5, size=500)

    small_model = BayesianLinearRegression(alpha=1.0, beta=4.0).fit(small_X, small_y)
    large_model = BayesianLinearRegression(alpha=1.0, beta=4.0).fit(large_X, large_y)

    small_error = np.abs(small_model.posterior_mean - true_weights).sum()
    large_error = np.abs(large_model.posterior_mean - true_weights).sum()

    assert large_error < small_error


def test_predictive_uncertainty_shrinks_with_more_data():
    """The defining property of a Bayesian method: as more evidence comes
    in, the model should become more confident, not just more accurate."""
    rng = np.random.default_rng(1)
    true_weights = np.array([1.0, 3.0])

    small_X = rng.uniform(-2, 2, size=(5, 1))
    small_y = true_weights[0] + true_weights[1] * small_X[:, 0] + rng.normal(0, 0.3, size=5)

    large_X = rng.uniform(-2, 2, size=(300, 1))
    large_y = true_weights[0] + true_weights[1] * large_X[:, 0] + rng.normal(0, 0.3, size=300)

    small_model = BayesianLinearRegression(alpha=1.0, beta=10.0).fit(small_X, small_y)
    large_model = BayesianLinearRegression(alpha=1.0, beta=10.0).fit(large_X, large_y)

    query_point = np.array([[0.5]])
    _, small_std = small_model.predict(query_point, return_std=True)
    _, large_std = large_model.predict(query_point, return_std=True)

    assert large_std[0] < small_std[0]


def test_converges_to_ols_with_a_weak_prior_and_lots_of_data():
    rng = np.random.default_rng(2)
    true_weights = np.array([0.5, 2.0, -1.0])
    X = rng.uniform(-5, 5, size=(2000, 2))
    y = true_weights[0] + X @ true_weights[1:] + rng.normal(0, 0.5, size=2000)

    model = BayesianLinearRegression(alpha=1e-4, beta=4.0).fit(X, y)

    assert np.allclose(model.posterior_mean, true_weights, atol=0.1)
