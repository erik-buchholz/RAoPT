#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import logging
import multiprocessing as mp
import pickle
import warnings
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit, haversine
from tqdm import tqdm

from raopt.utils.config import Config
from raopt.utils.helpers import ms_to_kmh

log = logging.getLogger()


def load_cache(filename: str):
    if Config.is_caching() and Path(filename).exists():
        log.warning(
            f"Loading data from cached file '{filename}'.")
        return pickle.load(open(filename, "rb"))
    else:
        return None


def drop_out_of_bounds(arg: tuple) -> pd.DataFrame:
    """Adapter to use _drop_out_of_points with mp.Pool().map()."""
    taxi, b1, b2, b3, b4 = arg
    return _drop_out_of_bounds(taxi, min_lon=b1, max_lon=b2, min_lat=b3, max_lat=b4)


def _drop_out_of_bounds(
        df: pd.DataFrame,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        lat_label: str = 'latitude',
        lon_label: str = 'longitude',
) -> pd.DataFrame:
    """Drop all points with longitude or latitude outside the given bounding box."""
    df.drop(df[
        (df[lon_label] < min_lon) | (df[lon_label] > max_lon) | (
            df[lat_label] < min_lat) | (df[lat_label] > max_lat)
    ].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def drop_duplicate_points(df: pd.DataFrame,
                          lat_label: str = 'latitude',
                          lon_label: str = 'longitude',
                          ) -> pd.DataFrame:
    """
    Remove points with duplicate timestamps. Choose the closer location to the surrounding points
    :param df: The DataFrame to modify (in place)
    :param lat_label: Label of latitude column
    :param lon_label: Label of longitude column
    :return: Return the modified DataFrame, but input was modified in-place
    """
    columns = [lat_label, lon_label]
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['tdiff'] = df['timestamp'].diff()
    while len(df[(df.tdiff == timedelta(seconds=0))]) > 0:
        df['lon_diff'] = df[lon_label].diff()
        df['lat_diff'] = df[lat_label].diff()
        duplicates = list(df[(df.tdiff == timedelta(seconds=0)) & (
            df.lon_diff == 0) & (df.lat_diff == 0)].index)
        indices = df[(df.tdiff == timedelta(seconds=0)) & (
            (df.lon_diff != 0) | (df.lat_diff != 0))].index
        for i in indices:
            if i == len(df) - 1:
                # If the last point is an outlier, remove it
                duplicates.append(i)
                continue
            prev = df.iloc[i - 2][columns]
            p1 = df.iloc[i - 1][columns]
            p2 = df.iloc[i][columns]
            next_p = df.iloc[i + 1][columns]
            dist_p1 = haversine(prev, p1, Unit.METERS) + \
                haversine(p1, next_p, Unit.METERS)
            dist_p2 = haversine(prev, p2, Unit.METERS) + \
                haversine(p2, next_p, Unit.METERS)
            if dist_p1 > dist_p2:
                duplicates.append(i - 1)
            else:
                duplicates.append(i)

        df.drop(duplicates, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['tdiff'] = df['timestamp'].diff()

    df.drop(columns=['tdiff', 'lon_diff', 'lat_diff'], inplace=True, errors='ignore')

    return df


def drop_speed_outliers(df: pd.DataFrame,
                        max_speed: float,
                        lat_label: str = 'latitude',
                        lon_label: str = 'longitude',
                        ) -> int:
    """
    Remove points with too large distance to surrounding points.
    :param df: The DataFrame to modify (in place)
    :param max_speed: the maximum allowed speed
    :param lat_label: Label of latitude column
    :param lon_label: Label of longitude column
    :return: Return the modified DataFrame, but input was modified in-place
    """
    if len(df) == 0:
        return df
    columns = [lat_label, lon_label]
    changed = True
    i = -1
    while changed:
        # We might need multiple rounds as removing a point changes the speed of the next one
        changed = False
        i += 1

        df['haversine'] = haversine_vector(
            np.array(df[columns].shift(1)), (np.array(df[columns])), unit=Unit.METERS)
        df['tdiff'] = df['timestamp'].diff()
        df['speed'] = ms_to_kmh(df.haversine / df.tdiff.dt.seconds)

        # Drop high speeds
        if i == 0:
            # In case of outliers we do not want to remove the next point, too.
            # Therefore, we check that the speed would be too high if the previous point was removed.
            # However, we only do this in the first round b/c then all the bad outliers should have been removed.
            df['haversine2'] = haversine_vector(
                np.array(df[columns].shift(2)), (np.array(df[columns])), unit=Unit.METERS)
            df['tdiff2'] = df['timestamp'].diff(periods=2)
            df['speed2'] = ms_to_kmh(df.haversine2 / df.tdiff2.dt.seconds)
            drop_speed_id = df[(df.speed > max_speed) &
                               (df.speed2 > max_speed)].index
            changed = True
        else:
            drop_speed_id = df[(df.speed > max_speed)].index
        if len(drop_speed_id) > 0:
            # We repeat until no changes are made anymore
            changed = True
        if 1 in drop_speed_id:
            try:
                d1 = haversine(df.iloc[0][columns], df.iloc[2][columns], Unit.METERS)
                d2 = haversine(df.iloc[1][columns], df.iloc[2][columns], Unit.METERS)
                if d1 < d2:
                    # First point is outlier, not second
                    drop_speed_id.drop(1)
                    drop_speed_id.append(pd.Index([0]))
            except IndexError:
                # First point already removed
                pass
        df.drop(drop_speed_id, inplace=True)

    df.drop(columns=['haversine', 'tdiff', 'speed', 'tdiff2', 'haversine2', 'speed2'], inplace=True, errors='ignore')
    df.reset_index(drop=True, inplace=True)

    return df


def verify_trajectory(df: pd.DataFrame,
                      interval: float,
                      max_dist: float,
                      min_len: int,
                      max_len: int,
                      lat_label: str = 'latitude',
                      lon_label: str = 'longitude',
                      ) -> bool:
    """
    Return true, if the trajectory is valid or raises a ValueError otherwise.
    :param df: Trajectory to verify
    :param interval: [s] Maximal time interval between two locations
    :param max_dist: [m] Maximal distance between two locations
    :param min_len: Minimal length of a trajectory
    :param max_len: Maximal length of a trajectory
    :param lat_label: Label of latitude column
    :param lon_label: Label of longitude column
    :return: True on success, raises error otherwise
    """
    columns = [lat_label, lon_label]
    distances = haversine_vector(
        np.array(df[columns].shift(1)), (np.array(df[columns])), unit=Unit.METERS)
    tdiff = df['timestamp'].diff()
    if len(df[tdiff.dt.seconds > interval]) > 0:
        print(df)
        raise ValueError(
            "There is a trajectories exceeding the maximal time gap.")
    if len(df[distances > max_dist]) > 0:
        print(df)
        raise ValueError(
            "There is a trajectories exceeding the maximal space gap.")
    if min_len > len(df) or max_len < len(df):
        print(df)
        raise ValueError("One trajectory has a bad length.")
    return True


def split_based_on_timediff(df: pd.DataFrame, interval: float) -> List[pd.DataFrame]:
    """
    Split one trajectory into a list of shorter trajectories based on the gap between data points.

    :param df: The trajectory to split.
    :param interval: [SECONDS] The maximal time between to data points within a trajectory.
    :return: List of trajectories, each represented as one pandas DataFrame.
    """
    interval = timedelta(seconds=interval)
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['tdiff'] = df['timestamp'].diff()
    split_indices: pd.DataFrame = df.loc[df['tdiff'] > interval]
    splits = [0] + list(split_indices.index) + [len(df)]
    dfs = [df.iloc[splits[i]: splits[i + 1]
                   ].drop(columns=['tdiff'], errors='ignore') for i in range(len(splits) - 1)]
    return dfs


def divide_by_size(dfs: List[pd.DataFrame], size: int) -> List[pd.DataFrame]:
    """
    Split trajectories such that each resulting trajectory has the given size. Offsets are just dropped.
    :param dfs: The trajectories to split.
    :param size: Size of the resulting trajectories.
    :return: List of trajectories, each represented as one pandas DataFrame.
    """
    result = []
    for df in dfs:
        tmp = [df.iloc[i:(i + size)] for i in range(0, len(df) + 1, size)]
        result.extend(tmp)
    result = [r for r in result if len(r) == size]
    return result


def _prepare_quantile_computation(
        t: pd.DataFrame,
        lat_label: str = 'latitude',
        lon_label: str = 'longitude',
) -> pd.DataFrame:
    cols = [lat_label, lon_label]
    if len(t) == 0:
        return t
    t['tdiff'] = t['timestamp'].diff().dt.seconds
    t['dist'] = haversine_vector(np.array(t[cols].shift(1)), (np.array(t[cols])), unit=Unit.METERS)
    t['speed'] = ms_to_kmh(t.dist / t.tdiff)
    return t


def compute_quantiles(
        trajs: List[pd.DataFrame],
        title: str,
        output_file: str = None,
        lat_label: str = 'latitude',
        lon_label: str = 'longitude',
        percentages: Tuple[float] = (0.9, 0.99, 0.999, 0.9999),
) -> None:
    """
    Compute the quantiles for the given dataset.
    :param trajs: List of trajectories as pandas DataFrames
    :param output_file: File to store the results
    :param title: Usually Dataset name
    :param lat_label: Label identifying latitude column
    :param lon_label: Label identifying longitude column
    :param percentages: The quantiles to compute
    :return:
    """
    values = [lat_label, lon_label, 'dist', 'speed', 'tdiff']
    upper_quantiles = {q: {v: [] for v in values} for q in percentages}
    lower_quantiles = {q: {v: [] for v in values} for q in percentages}

    print("Prepare dataset...")
    if Config.parallelization_enabled():
        with mp.Pool(mp.cpu_count()) as pool:
            res = [r for r in tqdm(pool.imap(_prepare_quantile_computation, trajs, chunksize=10), total=len(trajs))]
    else:
        res = [_prepare_quantile_computation(r) for r in tqdm(trajs)]
    ts = pd.concat(res)
    ts.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Compute Quantiles...")
    for q in percentages:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uq = ts.quantile(q=q)
            lq = ts.quantile(q=(1 - q))
        for v in values:
            upper_quantiles[q][v] = uq[v]
            lower_quantiles[q][v] = lq[v]

    for q in percentages:
        print("-" * 50, f"{q * 100}%-Quantile", "-" * 50)
        print(
            f"{q * 100}%-Quantile Time: "
            f"{round(upper_quantiles[q]['tdiff'], 2)}s -- {round(lower_quantiles[q]['tdiff'], 2)}s"
        )
        print(
            f"{q * 100}%-Quantile Distance: "
            f"{round(upper_quantiles[q]['dist'], 4)}km -- {round(lower_quantiles[q]['dist'], 4)}km"
        )
        print(
            f"{q * 100}%-Quantile Speed: "
            f"{round(upper_quantiles[q]['speed'], 2)}km/h -- {round(lower_quantiles[q]['speed'], 2)}km/h"
        )
        print(
            f"{q * 100}%-Quantile Latitude: "
            f"{round(upper_quantiles[q][lat_label], 2)} -- {round(lower_quantiles[q][lat_label])}"
        )
        print(
            f"{q * 100}%-Quantile Longitude: "
            f"{round(upper_quantiles[q][lon_label], 2)} -- {round(lower_quantiles[q][lon_label])}"
        )

    names = {
        'tdiff': "Time",
        'dist': "Distance",
        'speed': "Speed",
        lat_label: "Latitude",
        lon_label: "Longitude"
    }

    if output_file is not None:
        with open(output_file, 'w') as fd:
            fd.write(f"# {title}\n\n")
            # fd.write("The following values have been produced without pre-processing. \n\n")
            for q in percentages:
                fd.write(f"## {q * 100}%-Quantile\n")
                for v in values:
                    fd.write(f"* **{names[v]}:** {round(upper_quantiles[q][v], 4)} -- "
                             f"{round(lower_quantiles[q][v], 4)}\n")
