"""A CART-style binary decision tree classifier built from first
principles: recursive Gini-impurity splits over candidate thresholds, no
scikit-learn. Exists specifically to demonstrate the thing a linear model
cannot do -- carve out a non-linear decision boundary."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Node:
    feature_index: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    predicted_class: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.predicted_class is not None


def _gini(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels)
    proportions = counts / len(labels)
    return 1.0 - np.sum(proportions**2)


def _best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float] | None:
    """Returns the single lowest-weighted-impurity split among all
    candidates, whatever it is -- NOT only a split that strictly improves
    on the parent's own impurity.

    Requiring strict improvement was a real bug: on a symmetric XOR-like
    pattern, every single-feature split at the root produces exactly the
    same weighted Gini as the parent (a well-known limitation of greedy
    top-down trees -- no single axis-aligned split can separate XOR at
    all). Rejecting every tied candidate left the tree unable to split
    past the root, predicting the majority class for every row. The fix
    is to always take the best candidate split and let depth and
    min_samples_split be the actual stopping criteria, matching what a
    real CART implementation does: a "wasted" first split still subdivides
    the data so a genuinely separating split can be found one level down."""
    n_samples, n_features = X.shape
    best_weighted_gini: float | None = None
    best_split: tuple[int, float] | None = None

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        if len(thresholds) < 2:
            continue
        candidates = (thresholds[:-1] + thresholds[1:]) / 2
        for threshold in candidates:
            left_mask = X[:, feature_index] <= threshold
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            weighted_gini = (
                left_mask.sum() * _gini(y[left_mask]) + right_mask.sum() * _gini(y[right_mask])
            ) / n_samples
            if best_weighted_gini is None or weighted_gini < best_weighted_gini:
                best_weighted_gini = weighted_gini
                best_split = (feature_index, threshold)

    return best_split


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: _Node | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        majority_class = int(np.bincount(y).argmax())

        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return _Node(predicted_class=majority_class)

        split = _best_split(X, y)
        if split is None:
            return _Node(predicted_class=majority_class)

        feature_index, threshold = split
        left_mask = X[:, feature_index] <= threshold
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return _Node(feature_index=feature_index, threshold=threshold, left=left, right=right)

    def _predict_one(self, x: np.ndarray, node: _Node) -> int:
        if node.is_leaf:
            return node.predicted_class
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root) for x in X])
