"""Runs each from-scratch model next to its scikit-learn equivalent on the
same synthetic data and prints a side-by-side comparison, so the claim
"matches scikit-learn" is something you can see run, not just read in a
test file."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

from algorithms.bayesian_regression import BayesianLinearRegression
from algorithms.decision_tree import DecisionTreeClassifier
from algorithms.kmeans import KMeans
from algorithms.linear_regression import LinearRegression
from algorithms.logistic_regression import LogisticRegression

SEED = 0


def compare_linear_regression():
    rng = np.random.default_rng(SEED)
    X = rng.uniform(-5, 5, size=(300, 3))
    y = X @ np.array([1.5, -4.0, 0.5]) + 2.0 + rng.normal(0, 1.0, size=300)

    ours = LinearRegression().fit(X, y)
    theirs = SklearnLinearRegression().fit(X, y)

    print("Linear Regression")
    print(f"  ours   R2={r2_score(y, ours.predict(X)):.4f}  coef={np.round(ours.coefficients, 3)}")
    print(f"  sklearn R2={r2_score(y, theirs.predict(X)):.4f}  coef={np.round(theirs.coef_, 3)}")


def compare_logistic_regression():
    rng = np.random.default_rng(SEED)
    class_0 = rng.normal(loc=[-1, -1], scale=1.5, size=(150, 2))
    class_1 = rng.normal(loc=[1, 1], scale=1.5, size=(150, 2))
    X = np.vstack([class_0, class_1])
    y = np.array([0] * 150 + [1] * 150)

    ours = LogisticRegression().fit(X, y)
    theirs = SklearnLogisticRegression().fit(X, y)

    print("Logistic Regression")
    print(f"  ours    accuracy={(ours.predict(X) == y).mean():.4f}")
    print(f"  sklearn accuracy={(theirs.predict(X) == y).mean():.4f}")


def compare_decision_tree():
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=300, centers=4, n_features=3, cluster_std=1.5, random_state=SEED)
    ours = DecisionTreeClassifier(max_depth=5).fit(X, y)
    theirs = SklearnDecisionTreeClassifier(max_depth=5, random_state=SEED).fit(X, y)

    print("Decision Tree")
    print(f"  ours    accuracy={(ours.predict(X) == y).mean():.4f}")
    print(f"  sklearn accuracy={(theirs.predict(X) == y).mean():.4f}")


def compare_kmeans():
    rng = np.random.default_rng(SEED)
    true_centers = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
    X = np.vstack([rng.normal(loc=c, scale=0.5, size=(40, 2)) for c in true_centers])

    ours = KMeans(n_clusters=4, seed=SEED).fit(X)
    theirs = SklearnKMeans(n_clusters=4, n_init=10, random_state=SEED).fit(X)

    print("K-Means")
    print(f"  ours    inertia={ours.inertia(X):.2f}")
    print(f"  sklearn inertia={theirs.inertia_:.2f}")


def demo_bayesian_regression():
    rng = np.random.default_rng(SEED)
    true_weights = np.array([2.0, -1.5])
    X = rng.uniform(-3, 3, size=(20, 1))
    y = true_weights[0] + true_weights[1] * X[:, 0] + rng.normal(0, 0.5, size=20)

    model = BayesianLinearRegression(alpha=1.0, beta=4.0).fit(X, y)
    mean, std = model.predict(np.array([[0.0], [2.0]]), return_std=True)

    print("Bayesian Linear Regression (no scikit-learn equivalent -- this is the point)")
    print(f"  posterior mean (intercept, slope) = {np.round(model.posterior_mean, 3)}, true = {true_weights}")
    print(f"  prediction at x=0: {mean[0]:.3f} +/- {std[0]:.3f}")
    print(f"  prediction at x=2: {mean[1]:.3f} +/- {std[1]:.3f}")


if __name__ == "__main__":
    compare_linear_regression()
    print()
    compare_logistic_regression()
    print()
    compare_decision_tree()
    print()
    compare_kmeans()
    print()
    demo_bayesian_regression()
