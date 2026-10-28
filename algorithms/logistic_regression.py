"""Binary logistic regression fit by batch gradient descent on the
cross-entropy loss -- no scikit-learn. Same internal standardization and
unstandardization approach as the linear regression module, for the same
reason: stable gradient descent, interpretable final coefficients."""
from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coefficients: np.ndarray | None = None
        self.intercept: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X_scaled = (X - mean) / std

        w = np.zeros(n_features)
        b = 0.0
        for _ in range(self.n_iterations):
            predictions = _sigmoid(X_scaled @ w + b)
            error = predictions - y
            grad_w = (1 / n_samples) * (X_scaled.T @ error)
            grad_b = (1 / n_samples) * np.sum(error)
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

        self.coefficients = w / std
        self.intercept = b - np.sum(w * mean / std)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return _sigmoid(X @ self.coefficients + self.intercept)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
