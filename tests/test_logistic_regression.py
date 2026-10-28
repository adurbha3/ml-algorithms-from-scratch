import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from algorithms.logistic_regression import LogisticRegression


def test_perfectly_separates_well_separated_classes():
    rng = np.random.default_rng(0)
    class_0 = rng.normal(loc=[-3, -3], scale=0.5, size=(50, 2))
    class_1 = rng.normal(loc=[3, 3], scale=0.5, size=(50, 2))
    X = np.vstack([class_0, class_1])
    y = np.array([0] * 50 + [1] * 50)

    model = LogisticRegression(learning_rate=0.5, n_iterations=5000).fit(X, y)
    predictions = model.predict(X)

    assert (predictions == y).mean() == 1.0


def test_predict_proba_is_between_zero_and_one():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    probabilities = model.predict_proba(X)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_matches_sklearn_accuracy_on_noisy_data():
    rng = np.random.default_rng(2)
    class_0 = rng.normal(loc=[-1, -1], scale=1.5, size=(150, 2))
    class_1 = rng.normal(loc=[1, 1], scale=1.5, size=(150, 2))
    X = np.vstack([class_0, class_1])
    y = np.array([0] * 150 + [1] * 150)

    ours = LogisticRegression(learning_rate=0.3, n_iterations=5000).fit(X, y)
    sklearn_model = SklearnLogisticRegression().fit(X, y)

    our_accuracy = (ours.predict(X) == y).mean()
    sklearn_accuracy = (sklearn_model.predict(X) == y).mean()

    assert abs(our_accuracy - sklearn_accuracy) < 0.05
