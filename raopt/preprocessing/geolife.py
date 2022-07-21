#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import logging
import multiprocessing as mp
import re
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from raopt.preprocessing.preprocess import drop_out_of_bounds, drop_duplicate_points, \
    drop_speed_outliers, split_based_on_timediff, verify_trajectory, load_cache
from raopt.utils import logger
from raopt.utils.config import Config
from raopt.utils.helpers import store, kmh_to_ms, trajectories_to_csv, find_bbox

# -----------------------------CONSTANTS--------------------------------------------------------------------------------
CHUNKSIZE = 1  # For multiprocessing
pdir = Config.get_cache_dir('geolife')
Path(pdir).mkdir(exist_ok=True, parents=True)
csv_dir = Config.get_csv_dir('geolife')
data_dir = Config.get_dataset_dir('geolife')
_geolife_cache = pdir + 'geolife_cleaned.pickle'
_geolife_trajectory_dict = pdir + 'originals_dict.pickle'

if __name__ == '__main__':
    log = logger.configure_root_loger(
        logging.INFO, Config.get_logdir() + "geolife.log")
else:
    log = logging.getLogger()


def _read_geolife_file(user: int, trajectory_id: int) -> pd.DataFrame:
    """Read the t-drive files into dataFrame."""
    filename = f"{data_dir}{user:03d}/Trajectory/{trajectory_id}.plt"
    datatypes = OrderedDict({
        "latitude": float,
        "longitude": float,
        "ignore": str,
        "altitude": float,
        "days": float,
        "date": str,
        "time": str,
    })
    dates = [['date', 'time']]
    df = pd.read_csv(
        filename,
        delimiter=',',
        header=None,
        names=list(datatypes.keys()),
        skiprows=6,
        dtype=datatypes,
        parse_dates=dates
    )
    df.rename(columns={'date_time': 'timestamp'}, inplace=True)
    df.drop(inplace=True, columns=['ignore', 'days'])
    df['uid'] = user
    df['trajectory_id'] = f"{user:03d}_{trajectory_id}"
    return df


def _drop_speed_outliers_geolife(df: pd.DataFrame) -> int:
    max_speed = float(Config.get('GEOLIFE', 'OUTLIER_SPEED'))
    return drop_speed_outliers(df, max_speed=max_speed)


def _clean_trajectories(trajs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Remove outliers and clean the locations."""
    # Remove invalid points
    bbox = find_bbox(trajs, 0.95)
    log.info(f"Using bounding box: {np.around(bbox, 2)}")
    with mp.Pool(mp.cpu_count()) as pool:
        args = ((t, *bbox) for t in trajs)
        trajs = [r for r in tqdm(
            pool.imap(drop_out_of_bounds, args, chunksize=CHUNKSIZE),
            total=len(trajs),
            desc='Removing out-of-bounds points'
        )]
        trajs = [r for r in tqdm(
            pool.imap(drop_duplicate_points, trajs, chunksize=CHUNKSIZE),
            total=len(trajs),
            desc='Dropping duplicates'
        )]
        trajs = [r for r in tqdm(
            pool.imap(_drop_speed_outliers_geolife, trajs, chunksize=CHUNKSIZE),
            total=len(trajs),
            desc="Dropping speed outliers"
        )]

    # Remove nearly empty trajectories
    trajs = [t for t in trajs if len(t) >= int(Config.get('GEOLIFE', 'MIN_LENGTH'))]
    return trajs


def _process_user(uid: int) -> List[pd.DataFrame]:
    """Return the trajectories of one user with ID *uid*."""
    tdir = data_dir + f"{uid:03d}/Trajectory/"
    files = Path(tdir).glob("*.plt")
    trajs = []
    for file in files:
        tid = re.search(r'.*/Trajectory/([0-9]*)\.plt', str(file)).group(1)
        trajs.append(_read_geolife_file(uid, tid))
    return trajs


def get_geolife(disable_cleaning: bool = False) -> List[pd.DataFrame]:
    """
    Read all original geolife trajectories into pandas DataFrames.
    :param disable_cleaning: Return trajectories as they are
    :return: List of Trajectories as pandas DataFrames
    """
    trajs = load_cache(_geolife_cache)
    if trajs is None:
        uids = range(int(Config.get('GEOLIFE', 'MIN_UID')), int(Config.get('GEOLIFE', 'MAX_UID')) + 1)
        trajs = []
        with mp.Pool(mp.cpu_count()) as pool:
            for r in tqdm(pool.imap(_process_user, uids, chunksize=1), total=len(uids), desc='Reading Files'):
                trajs.extend(r)

        if not disable_cleaning:
            trajs = _clean_trajectories(trajs)

        log.info(f"Generated {len(trajs)} GeoLife DataFrames.")

        # Write to cache
        if Config.is_caching():
            store(trajs, _geolife_cache)
    return trajs


def _split_based_on_time_geolife(df: pd.DataFrame) -> List[pd.DataFrame]:
    interval = float(Config.get('GEOLIFE', 'INTERVAL'))  # seconds
    return split_based_on_timediff(df, interval)


def _verify_geolife_trajectory(df: pd.DataFrame) -> bool:
    """
    Return true, if the trajectory is valid or raises a ValueError otherwise.
    :param df: Trajectory to verify
    :return: True on success
    """
    interval = float(Config.get('GEOLIFE', 'INTERVAL'))
    speed = int(Config.get('GEOLIFE', 'OUTLIER_SPEED'))
    max_dist = kmh_to_ms(speed) * interval  # in Meter
    return verify_trajectory(df, interval, max_dist,
                             int(Config.get('GEOLIFE', 'MIN_LENGTH')),
                             int(Config.get('GEOLIFE', 'MAX_LENGTH')))


def get_geolife_trajectories() -> List[pd.DataFrame]:
    """
    Returns the GeoLife trajectory used for our evaluation.
    :return: List of Trajectories as pandas DataFrames
    """
    result = load_cache(_geolife_trajectory_dict)
    if result is None:
        trajs: List[pd.DataFrame] = get_geolife()

        # Split trajectories if the break between two locations exceeds a value
        with mp.Pool(mp.cpu_count()) as pool:
            result = []
            for r in tqdm(pool.imap(_split_based_on_time_geolife, trajs, chunksize=CHUNKSIZE),
                          total=len(trajs),
                          leave=True,
                          desc='Splitting Trajectories'):
                result.extend(r)

            # Remove too short or too long trajectories
            result = [r.reset_index(drop=True)
                      for r in result
                      if int(Config.get('GEOLIFE', 'MAX_LENGTH')) >= len(r) >= int(Config.get('GEOLIFE', 'MIN_LENGTH'))]

            # Add a unique ID
            odict = {}
            for t in tqdm(result, leave=True, desc='Creating unique indices'):
                idx = t.trajectory_id[0]
                i = 0
                while f"{idx}_{i}" in odict:
                    i += 1
                idx = f"{idx}_{i}"
                t['trajectory_id'] = idx
                odict[idx] = t

            for _ in tqdm(
                    pool.imap_unordered(_verify_geolife_trajectory, result, chunksize=CHUNKSIZE),
                    total=len(result),
                    desc='Verification'
            ):
                pass

        log.info(f"Generated {len(result)} GeoLife trajectories.")

        if Config.is_caching():
            store(odict, _geolife_trajectory_dict)
        trajectories_to_csv(result, csv_dir + 'originals.csv')

        assert len(result) != len(trajs)

    return result


if __name__ == '__main__':
    get_geolife_trajectories()
