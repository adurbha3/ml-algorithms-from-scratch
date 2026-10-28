"""K-means clustering via Lloyd's algorithm from scratch: random
initialization from the data, then alternating assignment and update
steps until the centers stop moving or a maximum iteration count is hit.

Runs several random restarts and keeps the lowest-inertia result, not
just one. A single run found a real failure mode during testing: on a
well-separated four-cluster case, a single unlucky random initialization
put two starting points in the same true cluster, so the algorithm
converged with two of its final centers splitting that one cluster while
a third center drifted to the midpoint between two OTHER true clusters it
never got an initial point near -- a textbook bad local optimum for
randomly initialized k-means, which is exactly why real implementations
default to multiple restarts instead of trusting a single run."""
from __future__ import annotations

import numpy as np


class KMeans:
    def __init__(
        self,
        n_clusters: int = 3,
        max_iterations: int = 300,
        tol: float = 1e-6,
        n_init: int = 10,
        seed: int = 0,
    ):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tol = tol
        self.n_init = n_init
        self.seed = seed
        self.centers: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def _run_once(self, X: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        initial_indices = rng.choice(len(X), size=self.n_clusters, replace=False)
        centers = X[initial_indices].copy()
        labels = np.zeros(len(X), dtype=int)

        for _ in range(self.max_iterations):
            distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            new_centers = np.empty_like(centers)
            for cluster in range(self.n_clusters):
                members = X[labels == cluster]
                new_centers[cluster] = members.mean(axis=0) if len(members) > 0 else centers[cluster]

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break

        return centers, labels

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)

        best_centers = None
        best_labels = None
        best_inertia = None

        for _ in range(self.n_init):
            centers, labels = self._run_once(X, rng)
            inertia = float(np.sum((X - centers[labels]) ** 2))
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels

        self.centers = best_centers
        self.labels_ = best_labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        distances = np.linalg.norm(X[:, None, :] - self.centers[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def inertia(self, X: np.ndarray) -> float:
        X = np.asarray(X, dtype=float)
        labels = self.predict(X)
        return float(np.sum((X - self.centers[labels]) ** 2))
