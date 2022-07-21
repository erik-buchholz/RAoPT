#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
"""
This file contains the different evaluation metrics.
"""
import logging

import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit
from shapely.geometry import Polygon

from raopt.utils.helpers import get_latlon_matrix
from raopt.preprocessing.coordinates import is_polar_coord_pd

log = logging.getLogger()


def euclidean_distance_pd(traj1: pd.DataFrame, traj2: pd.DataFrame,
                          use_haversine=True, disable_checks: bool = False) -> float:
    """
    Compute the Euclidean distance of 2 DataFrames, based on longitude and latitude.
    Follow definition in
    Shao M, Li J, Yan Q, Chen F, Huang H, Chen X. Structured Sparsity Model Based Trajectory Tracking Using Private
    Location Data Release. IEEE Trans Dependable Secure Computations 2020;5971(c):1-1. doi:10.1109/TDSC.2020.2972334
    &
    Su H, Liu S, Zheng B, Zhou X, Zheng K. A survey of trajectory distance measures and performance evaluation.
    VLDB J. 2020;29(1):3-32. doi:10.1007/s00778-019-00574-9


    :param disable_checks: Necessary for GNoise and PNoise as results are too noisy for consistency checks
    :param use_haversine: Use haversine distance instead of simple subtraction
    :param traj1: Trajectory 1
    :param traj2: Trajectory 2
    :return: The computed Euclidean distance
    """
    if not disable_checks:
        check_haversine_usable_pd(traj1, traj2, use_haversine)
    t2 = get_latlon_matrix(traj2)
    t1 = get_latlon_matrix(traj1)
    if len(t2) != len(t1):
        if 'trajectory_id' in traj1:
            log.warning("Using asynchronous euclidean distance!"
                        f"Trajectory ID 1: {traj1['trajectory_id'][0]}; Length: {len(traj1)}'"
                        f"Trajectory ID 2: {traj2['trajectory_id'][0]}; Length: {len(traj2)}'")
        else:
            log.warning("Using asynchronous euclidean distance, but cannot check IDs.")
        return _async_euclidean_distance(t1, t2, use_haversine)
    return euclidean_distance(t1, t2, use_haversine)


def euclidean_distance(t1: np.ndarray, t2: np.ndarray, use_haversine=True):
    """Compute the euclidean distance between two trajectories."""
    n = len(t1)
    if use_haversine:
        d = haversine_vector(t1, t2, unit=Unit.METERS)
    else:
        d = np.linalg.norm((t2 - t1), axis=1)
    res = 1 / n * np.sum(d)
    return res


def _async_euclidean_distance(t1: np.ndarray, t2: np.ndarray, use_haversine=True) -> float:
    """
    Compute the Euclidean distance of 2 DataFrames, based on longitude and latitude, for DataFrames with differing
    lengths.
    Follow definition in
    1. Su H, Liu S, Zheng B, Zhou X, Zheng K. A survey of trajectory distance measures and performance evaluation.
    VLDB J. 2020;29(1):3-32. doi:10.1007/s00778-019-00574-9


    :param use_haversine: Use haversine distance instead of simple subtraction
    :param t1: Trajectory 1
    :param t2: Trajectory 2
    :return: The computed Euclidean distance
    """
    if len(t1) > len(t2):
        t1, t2 = t2, t1
    n = len(t1)
    m = len(t2)
    if m == n:
        raise ValueError("The synchronous definition should have been used.")
    results = []
    for j in range(0, m - n + 1):
        tmp = t2[j:j + n, ]
        if use_haversine:
            d = haversine_vector(t1, tmp, unit=Unit.METERS)
        else:
            d = np.linalg.norm((tmp - t1), axis=1)
        results.append(np.sum(d))
    assert len(results) == m - n + 1
    return 1 / n * min(results)


def hausdorff_distance_pd(t1: pd.DataFrame, t2: pd.DataFrame,
                          use_haversine: bool = True, disable_checks: bool = False) -> float:
    """
    Compute the Hausdorff distance of 2 DataFrames, based on longitude and latitude.

    :param disable_checks: Necessary for GNoise and PNoise as results are too bad
    :param use_haversine: Use haversine distance instead of simple subtraction
    :param t1: Trajectory 1
    :param t2: Trajectory 2
    :return: The computed Hausdorff distance
    """
    if not disable_checks:
        check_haversine_usable_pd(t1, t2, use_haversine)
    t1 = get_latlon_matrix(t1)
    t2 = get_latlon_matrix(t2)
    return hausdorff_distance(t1, t2, use_haversine)


def hausdorff_distance(t1: np.ndarray, t2: np.ndarray, use_haversine: bool = True) -> float:
    """
        Compute the Hausdorff distance of 2 trajectories, based on longitude and latitude or x and y.

        :param use_haversine: Use haversine distance instead of simple subtraction
        :param t1: Trajectory 1
        :param t2: Trajectory 2
        :return: The computed Hausdorff distance
        """
    return max([
        _hausdorff_distance_directed(t1, t2, use_haversine=use_haversine),
        _hausdorff_distance_directed(t2, t1, use_haversine=use_haversine),
    ])


def _hausdorff_distance_directed(t1: np.ndarray, t2: np.ndarray, use_haversine: bool = True) -> float:
    results = []
    for v in t1:
        tmp = np.broadcast_to(v, (len(t2), len(v)))
        # Use the haversine formula for distance
        if use_haversine:
            distances = haversine_vector(t2, tmp, unit=Unit.METERS)
        else:
            distances = np.linalg.norm((t2 - tmp), axis=1)
        results.append(min(distances))
    return max(results)


def check_haversine_usable_pd(t1: np.ndarray, t2: np.ndarray, use_haversine: bool = True) -> None:
    """
    Check if both trajectories are defined in polat coordinates, i.e., haversine formula is applicable
    :param t1: Trajectory 1
    :param t2: trajectory 2
    :param use_haversine: Expected use
    :return: None, raises error on mismatch
    """
    if use_haversine:
        # Check that coordinates are longitude and latitude
        if not is_polar_coord_pd(t1) or not is_polar_coord_pd(t2):
            raise ValueError(
                "Haversine Formula only works with longitude and latitude."
            )
    else:
        if is_polar_coord_pd(t1) or is_polar_coord_pd(t2):
            logging.warning(
                "It looks like you are not using haversine distance although dealing with polar coordinates."
            )
            # raise ValueError("For polar coordinates, the haversine formula should be used.")


def jaccard_index(t1: np.ndarray, t2: np.ndarray) -> float:
    """
    Compute Jaccard Index of convex hull
    :param t1: Trajectory 1
    :param t2: trajectory 2
    :return: float
    """
    p1, p2 = Polygon(t1).convex_hull, Polygon(t2).convex_hull
    intersection = p1.intersection(p2)
    union = p1.union(p2)
    jaccard_index = intersection.area / union.area
    return jaccard_index


def jaccard_index_pd(t1: pd.DataFrame, t2: pd.DataFrame) -> float:
    """
    Compute Jaccard Index of convex hull
    :param t1: Trajectory 1
    :param t2: trajectory 2
    :return: float
    """
    return jaccard_index(get_latlon_matrix(t1), get_latlon_matrix(t2))
