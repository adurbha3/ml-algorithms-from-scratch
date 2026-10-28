import numpy as np
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

from algorithms.decision_tree import DecisionTreeClassifier


def test_perfectly_fits_a_minimal_xor_pattern_no_linear_model_can_separate():
    """A known correctness case: (x1 > 0) XOR (x2 > 0) is not linearly
    separable, so a tree must actually split on both features to solve
    it. The first split (on either feature) ties the parent's Gini
    exactly -- no single split improves purity on pure XOR, a textbook
    limitation of greedy top-down induction -- but it still cleanly
    divides the data into two pairs, and the second-level split on the
    other feature then separates each pair perfectly. (A larger,
    asymmetric version of this pattern with extra redundant points was
    tried first and is NOT perfectly solvable by greedy splitting --
    verified by comparing against scikit-learn's tree, which gets the
    exact same two points wrong. That is a real, shared limitation of
    greedy induction, not a bug, so this test uses the minimal case
    that greedy splitting can actually solve.)"""
    X = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(max_depth=4).fit(X, y)
    predictions = model.predict(X)

    assert (predictions == y).all()


def test_shallow_tree_cannot_perfectly_fit_complex_pattern():
    X = np.array([
        [1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0],
    ])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(max_depth=1).fit(X, y)
    predictions = model.predict(X)
    assert (predictions == y).sum() < len(y)


def test_matches_sklearn_accuracy_on_blobs():
    X, y = make_blobs(n_samples=300, centers=4, n_features=3, cluster_std=1.5, random_state=0)

    ours = DecisionTreeClassifier(max_depth=5).fit(X, y)
    sklearn_model = SklearnDecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)

    our_accuracy = (ours.predict(X) == y).mean()
    sklearn_accuracy = (sklearn_model.predict(X) == y).mean()

    assert abs(our_accuracy - sklearn_accuracy) < 0.05
