"""Linear regression fit by batch gradient descent on mean squared error
-- no closed-form normal equation shortcut, no scikit-learn. Features are
standardized internally for stable gradient descent, then the learned
coefficients are converted back to the original feature scale so they
stay directly interpretable."""
from __future__ import annotations

import numpy as np


class LinearRegression:
    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coefficients: np.ndarray | None = None
        self.intercept: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
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
            predictions = X_scaled @ w + b
            error = predictions - y
            grad_w = (2 / n_samples) * (X_scaled.T @ error)
            grad_b = (2 / n_samples) * np.sum(error)
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

        self.coefficients = w / std
        self.intercept = b - np.sum(w * mean / std)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coefficients + self.intercept
