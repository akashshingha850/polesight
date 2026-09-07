from scipy.spatial import KDTree
import numpy as np

from sklearn.cluster import DBSCAN
import logging

CLUSTER_CLOSEST: int = 1
CLUSTER_LARGEST: int = 2
CLUSTER_DENSEST: int = 3

import json
from pathlib import Path

DEFAULT_CONFIG = {
    "pipeline": [
        {
            "step": "remove_3d_outliers",
            "distance_threshold": 0.5,
            "number_of_neighhbors": 4,
            "max_iterations": 15,
        },
        {
            "step": "expand_cluster",
            "radii": (
                (15, 0.05),
                (5, 0.05),
            ),
        },
        {
            "step": "select_closest_cluster",
            "multiplier": 10,
            "min_dist": 0.05,
            "min_size": 15,
        },
        {
            "step": "expand_cluster",
            "radii": (
                (40, 0.2),
                (np.inf, 0.1),
                (np.inf, 0.05),
                (np.inf, 0.02),
                (np.inf, 0.02),
                (np.inf, 0.02),
                (np.inf, 0.02),
                (np.inf, 0.02),
            ),
        },
        # {"step": "remove_3d_outliers", "distance_threshold": 0.50, "number_of_neighhbors": 4, "max_iterations": 5},
        {
            "step": "select_cluster_dbscan",
            "min_samples": 15,
            "eps": 0.25,
            "mode": CLUSTER_LARGEST,
        },
    ]
}

PIPELINE_REGISTRY = {}


def register_step(name):

    def decorator(func):
        PIPELINE_REGISTRY[name] = func
        return func

    return decorator


def load_config(config_path=None):
    if config_path is None:
        return DEFAULT_CONFIG
    return json.loads(Path(config_path).read_text())


def expand_labels(
    mask_indices, point_cloud, kd_tree, config_path=None, config=None, densities=None
):

    config = config or load_config(config_path)
    progress = [mask_indices]

    if len(mask_indices) == 0:
        logging.debug(("TOO SHORT"))
        return [], []
    for stage in config["pipeline"]:
        step_name = stage["step"]

        step_fn = PIPELINE_REGISTRY.get(step_name)

        if step_fn is None:
            raise ValueError(f"Unknown pipeline step '{step_name}'")

        params = {k: v for k, v in stage.items() if k != "step"}
        params["kd_tree"] = kd_tree
        params["densities"] = densities

        # logging.debug(( step_fn, params, step_name))

        mask_indices = step_fn(
            point_cloud,
            mask_indices,
            **params,
        )

        progress.append(mask_indices)

        if len(mask_indices) == 0:
            return [], progress

    return mask_indices, progress


@register_step("remove_3d_outliers")
def remove_3d_outliers(
    points,
    idxs,
    distance_threshold=0.5,
    number_of_neighhbors=5,
    max_iterations=1,
    **kwargs,
):
    """Remove 3D spatial outliers using nearest-neighbor distances.

    Parameters
    ----------
    points : np.ndarray
        Point cloud.
    idxs : np.ndarray
        Candidate indices.
    distance_threshold : float
        Maximum allowed neighbor distance.

    Returns
    -------
    idxs : np.ndarray
        Filtered indices.
    """
    for iteration in range(max_iterations):
        local_kd_tree = KDTree(points[idxs, :3])
        dists, _ = local_kd_tree.query(
            points[idxs, :3],
            range(1, number_of_neighhbors),
            distance_upper_bound=distance_threshold * 1.5,
            workers=1,
            p=2,
        )

        to_remove = []
        for i, dist in enumerate(dists):
            if np.any(np.array(dist) > distance_threshold):
                to_remove.append(i)
        if len(to_remove) == 0:
            logging.debug((f"remove_3d_outliers BREAK {iteration=} {max_iterations}"))
            break
        idxs = np.delete(idxs, to_remove)
        if len(idxs) == 0:
            logging.debug(("Removed all"))
            break

    return idxs


@register_step("select_closest_cluster")
def select_closest_cluster(
    original_pc,
    idxs,
    multiplier=2,
    mode=CLUSTER_CLOSEST,
    min_size=10,
    min_dist=None,
    **kwargs,
):
    """Perform 1D distance-based filtering to select the closest dense cluster.

    Parameters
    ----------
    original_pc : np.ndarray
        Original point cloud.
    idxs : np.ndarray
        Candidate indices.

    Returns
    -------
    idxs : np.ndarray
        Filtered indices.
    """
    points = original_pc[idxs, :3]
    distances = np.linalg.norm(points, axis=1)
    order = np.argsort(distances)

    diffs = distances[order[1:]] - distances[order[:-1]]

    std = np.std(diffs) * multiplier
    min_dist = 0 if min_dist is not None else min_dist
    std = np.max(std, min_dist)

    selected = []
    prev = distances[order[0]]
    if mode == CLUSTER_LARGEST:
        clusters = []

    for idx in order:
        if distances[idx] - prev > std:
            if len(selected) < min_size:
                selected = []
                prev = distances[idx]
                continue
            if mode == CLUSTER_CLOSEST:
                break
            clusters.append(selected)
            selected = []

        selected.append(int(idx))
        prev = distances[idx]

    if mode == CLUSTER_CLOSEST:
        if len(selected) > 0:
            return idxs[selected]
        return np.zeros(0)

    if len(clusters) == 0:
        # There's no variation in distances -> single cluster
        return idxs

    clusters.sort(key=len, reverse=True)
    return idxs[clusters[0]]


@register_step("expand_cluster")
def expand_cluster(
    original_pc,
    idxs,
    kd_tree,
    radii=(
        (40, 0.2),
        (np.inf, 0.1),
        (np.inf, 0.05),
        (np.inf, 0.02),
        (np.inf, 0.02),
        (np.inf, 0.02),
        (np.inf, 0.02),
        (np.inf, 0.02),
    ),
    **kwargs,
):
    """Iteratively expand a cluster using fixed-radius KD-tree queries.

    Parameters
    ----------
    original_pc : np.ndarray
        Original point cloud.
    idxs : np.ndarray
        Seed indices.
    kd_tree : KDTree
        KD-tree built on the full point cloud.

    Returns
    -------
    idxs : np.ndarray
        Expanded cluster indices.
    """
    for k, radius in radii:
        number_of_points = len(idxs)
        positions = original_pc[idxs, :3]
        if k < np.inf:
            _, neighbors = kd_tree.query(
                positions, k, distance_upper_bound=radius, workers=1, p=2
            )
        else:
            neighbors = np.asanyarray(kd_tree.query_ball_point(positions, r=radius, p=2))
            neighbors = [n for list in neighbors for n in list]
        neighbors = np.reshape(neighbors, -1)

        idxs = np.append(idxs, neighbors)
        idxs = idxs[idxs < original_pc.shape[0]]
        idxs = np.unique(idxs)

        if len(idxs) == number_of_points:
            logging.debug((f"No new points in expanding cluster {k=} {radius=}"))
            return idxs
    logging.debug(("CLUSTER EXPANDION FINISHED"))
    return idxs


@register_step("select_cluster_dbscan")
def select_cluster_dbscan(
    original_pc,
    idxs,
    eps=0.15,
    min_samples=5,
    mode=CLUSTER_CLOSEST,
    **kwargs,
):
    """Cluster candidate points using DBSCAN and return one cluster.

    Parameters
    ----------
    original_pc : np.ndarray
    idxs : np.ndarray
    eps : float
        DBSCAN neighborhood radius.
    min_samples : int
        Minimum points for a core point.
    mode : int
        CLUSTER_CLOSEST
        CLUSTER_LARGEST
        CLUSTER_DENSEST

    Returns
    -------
    np.ndarray
        Selected cluster indices.
    """

    if len(idxs) == 0:
        return np.asarray([], dtype=np.int64)

    points = original_pc[idxs, :3]

    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    ).fit_predict(points)

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    if len(unique_labels) == 0:
        return np.asarray([], dtype=np.int64)

    clusters = []

    for label in unique_labels:
        local_mask = labels == label

        cluster_local_idxs = np.where(local_mask)[0]
        cluster_global_idxs = idxs[cluster_local_idxs]

        cluster_points = original_pc[cluster_global_idxs, :3]

        mean_distance = np.mean(np.linalg.norm(cluster_points, axis=1))
        score = mean_distance / len(cluster_global_idxs)
        clusters.append(
            {
                "indices": cluster_global_idxs,
                "size": len(cluster_global_idxs),
                "distance": mean_distance,
                "score": score,
            }
        )

    if mode == CLUSTER_CLOSEST:
        selected = min(clusters, key=lambda c: c["distance"])

    elif mode == CLUSTER_LARGEST:
        selected = max(clusters, key=lambda c: c["size"])

    else:
        raise ValueError(f"Unsupported mode {mode}")

    return selected["indices"]
