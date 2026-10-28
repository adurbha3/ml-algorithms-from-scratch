"""Bayesian linear regression with a conjugate Gaussian prior, solved in
closed form -- no scikit-learn, no sampling. Unlike the other four models
here, the point is not just a prediction: the posterior covariance gives
an honest uncertainty estimate that should shrink as more data arrives,
which is the actual defining property of a Bayesian method and the thing
this module's tests check for directly."""
from __future__ import annotations

import numpy as np


class BayesianLinearRegression:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """alpha: prior precision (1 / prior variance) on the weights.
        beta: noise precision (1 / noise variance), assumed known."""
        self.alpha = alpha
        self.beta = beta
        self.posterior_mean: np.ndarray | None = None
        self.posterior_cov: np.ndarray | None = None

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((len(X), 1)), X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLinearRegression":
        X = self._add_intercept(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]

        prior_precision = self.alpha * np.eye(n_features)
        posterior_precision = prior_precision + self.beta * (X.T @ X)
        self.posterior_cov = np.linalg.inv(posterior_precision)
        self.posterior_mean = self.beta * self.posterior_cov @ X.T @ y
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        X = self._add_intercept(np.asarray(X, dtype=float))
        mean = X @ self.posterior_mean
        if not return_std:
            return mean

        parameter_variance = np.sum((X @ self.posterior_cov) * X, axis=1)
        predictive_variance = 1.0 / self.beta + parameter_variance
        return mean, np.sqrt(predictive_variance)
