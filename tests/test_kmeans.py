import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans

from algorithms.kmeans import KMeans


def _match_centers_to_truth(found_centers, true_centers):
    """Greedily pairs each found center with its nearest true center, since
    cluster labels are arbitrary -- cluster 0 from our model might be
    cluster 2 in the ground truth."""
    matched = []
    for center in found_centers:
        distances = np.linalg.norm(true_centers - center, axis=1)
        matched.append(true_centers[np.argmin(distances)])
    return np.array(matched)


def test_recovers_well_separated_cluster_centers():
    rng = np.random.default_rng(0)
    true_centers = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 0.0]])
    X = np.vstack([rng.normal(loc=center, scale=0.3, size=(50, 2)) for center in true_centers])

    model = KMeans(n_clusters=3, seed=0).fit(X)
    matched_truth = _match_centers_to_truth(model.centers, true_centers)

    assert np.allclose(model.centers, matched_truth, atol=0.5)


def test_every_point_assigned_to_nearest_center():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    model = KMeans(n_clusters=2, seed=1).fit(X)
    predicted = model.predict(X)
    assert np.array_equal(predicted, model.labels_)


def test_multiple_restarts_avoid_bad_local_optima():
    """Regression test: a single random initialization can put two
    starting centers in the same true cluster, converging to a bad local
    optimum where one true cluster gets split and two others get merged.
    Averaged over several seeds, n_init=10 should reliably beat a single
    run's inertia (never meaningfully worse, often much better)."""
    rng = np.random.default_rng(3)
    true_centers = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
    X = np.vstack([rng.normal(loc=center, scale=0.3, size=(50, 2)) for center in true_centers])

    single_run_inertias = [KMeans(n_clusters=4, n_init=1, seed=seed).fit(X).inertia(X) for seed in range(10)]
    multi_run_inertia = KMeans(n_clusters=4, n_init=10, seed=0).fit(X).inertia(X)

    assert multi_run_inertia <= min(single_run_inertias) + 1e-6


def test_inertia_comparable_to_sklearn_on_separated_blobs():
    rng = np.random.default_rng(2)
    true_centers = np.array([[0.0, 0.0], [8.0, 8.0], [8.0, 0.0], [0.0, 8.0]])
    X = np.vstack([rng.normal(loc=center, scale=0.5, size=(40, 2)) for center in true_centers])

    ours = KMeans(n_clusters=4, seed=2).fit(X)
    sklearn_model = SklearnKMeans(n_clusters=4, n_init=10, random_state=2).fit(X)

    our_inertia = ours.inertia(X)
    sklearn_inertia = sklearn_model.inertia_

    assert our_inertia < sklearn_inertia * 1.1  # within 10% of sklearn's optimum
