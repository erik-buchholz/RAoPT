#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import logging
import pickle
import shutil
from pathlib import Path
from typing import List, Any, Dict, Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from timeit import default_timer as timer

from raopt.utils.config import Config

log = logging.getLogger()


def kmh_to_ms(v: float) -> float:
    """
    Convert km/h into m/s
    :param v: velocity in km/h
    :return: velocity in m/s
    """
    return v * 1000 / 3600


def ms_to_kmh(v: float) -> float:
    """
    Convert m/s into km/h
    :param v: velocity in m/s
    :return: velocity in km/h
    """
    return v / 1000 * 3600


def remove_cache() -> None:
    """
    Remove the trajectory cache files.
    """
    temp_dir = Config.get_temp_dir()
    for file in Path(temp_dir).glob("*.pickle"):
        log.info(f"Removing {file}.")
        file.unlink()


def get_latlon_arrays(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Extract longitude and latitude from a DataFrame
    :param df: DataFrame to extract from
    :return: (Latitudes: np.ndarray, Longitudes: np.ndarray)
    """
    return df['latitude'].to_numpy(), df['longitude'].to_numpy()


def get_latlon_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extract longitude and latitude from a DataFrame as a list of points
    :param df: DataFrame to extract from
    :return: Points: np.ndarray [[x1, y1], [x2, y2], ..., [xn, yn]]
    """
    cols = ['latitude', 'longitude']
    return df[cols].to_numpy()


def set_latlon(df: pd.DataFrame, lat: np.ndarray, lon: np.ndarray,
               lat_label: str = 'latitude', lon_label: str = 'longitude') -> pd.DataFrame:
    """
    Set the latitude and longitude values of DataFrame.
    :param df: DataFrame to modify
    :param lat: New Latitude
    :param lon: New Longitude
    :param lat_label: Latitude column name
    :param lon_label: Longitude column name
    :return:
    """
    df[lon_label] = lon
    df[lat_label] = lat
    return df


def plot_progress(history, filename: str = None) -> None:
    """
    Plot progress during the training.

    :param history: The history object returned by model.fit() [tf.keras.callbacks.History]
    :param filename: Write to file instead of showing
    :return: None
    """
    metrics = history.history.keys()
    plt.title('Training Metrics')
    for m in metrics:
        plt.plot(history.history[m], label=m)
    plt.ylim(0, 1)
    plt.legend()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def compute_reference_point(df: pd.DataFrame or Iterable[pd.DataFrame],
                            lat_label: str = 'latitude',
                            lon_label: str = 'longitude') -> (float, float):
    """
    Compute the weighted centroid of (a) DataFrame(s) containing longitude and latitude values.
    :param df: The dataset to consider
    :param lat_label: Label of the latitude column
    :param lon_label: Label of the longitude column
    :return: (Latitude, Longitude)
    """
    start_time = timer()
    if isinstance(df, Iterable) and not isinstance(df, pd.DataFrame):
        df = pd.concat(df)
    lat0 = df[lat_label].sum() / len(df)
    lon0 = df[lon_label].sum() / len(df)
    log.debug(f"Computed reference point in {timer() - start_time:.2f}s.")
    return lat0, lon0


def find_bbox(
        trajs: List[pd.DataFrame],
        quantile: float = 1,
        x_label: str = 'longitude',
        y_label: str = 'latitude'
) -> (float, float, float, float):
    """Find a bounding box enclosing the defined quantile of points.

    :return: (Minimum X, Maximum X, Minimum Y, Minimum Y)
    """
    single_db = pd.concat(trajs)
    upper_quantiles = single_db.quantile(q=quantile)
    lower_quantiles = single_db.quantile(q=(1 - quantile))
    return lower_quantiles[x_label], upper_quantiles[x_label], \
           lower_quantiles[y_label], upper_quantiles[y_label]


def compute_scaling_factor(trajectories: Iterable[pd.DataFrame], lat0: float, lon0: float) -> (float, float):
    """
    Return the scaling factor for latitude and longitude from a list of pandas DataFrames.
    :param trajectories: List[pd.Dataframe]
    :param lat0: Latitude of reference point
    :param lon0: Longitude of reference point
    :return: (latitude, longitude)
    """
    start_time = timer()
    # Concatenate to one list
    df = pd.concat(trajectories)
    # Scaling
    scale_lat = max(abs(df['latitude'].max() - lat0),
                    abs(df['latitude'].min() - lat0),
                    )
    scale_lon = max(abs(df['longitude'].max() - lon0),
                    abs(df['longitude'].min() - lon0),
                    )
    log.debug(f"Computed scaling factor in {timer() - start_time:.2f}s.")
    return scale_lat, scale_lon


def clear_tensorboard_logs() -> None:
    """
    Remove all existing tensorboard logs.
    :return:
    """
    tdir = Config.get_tensorboard_dir()
    for file in Path(tdir).glob('*'):
        print(f"Removing {file}.")
        if file.is_file():
            file.unlink(missing_ok=True)
        else:
            shutil.rmtree(file.absolute())


def store(obj: object, filename: str, mute: bool = False) -> None:
    """Store the given object into a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    if not mute:
        log.info(f"Wrote data to file {filename}")


def load(filename: str, mute: bool = False) -> Any:
    """Load an object from a pickle file."""
    if not mute:
        log.info(f"Loading data from {filename}")
    with open(filename, 'rb') as f:
        res = pickle.load(f)
    return res


def dictify_trajectories(lst: List[pd.DataFrame], tid_label: str = 'trajectory_id') -> Dict[int or str, pd.DataFrame]:
    """
    Load a list of trajectories, convert to a dictionary and save into a destination file.
    :param lst: List of trajectories
    :param tid_label: column name of trajectory ID
    :return: Dictionary containing mapping d[t.trajectory_id] = t
    """
    dct = {}
    for t in tqdm(lst, leave=False, desc='Dictify'):
        dct[str(t[tid_label].iloc[0])] = t
    return dct


def load_cached_trajectory_dict(base_filename: str, tid_label: str = 'trajectory_id') -> Dict[int or str, pd.DataFrame]:
    """
    Load a dictionary containing tID: trajectory from file or create it if no such file exists.
    :param base_filename: base filename for trajectories
    :param tid_label: column name of trajectory IDs
    :return:
    """
    protected_file = base_filename + ".pickle"
    protected_dict_file = base_filename + "_dict.pickle"
    if not Path(protected_dict_file).exists():
        protected_list = load(protected_file)
        protected_dict = dictify_trajectories(protected_list, tid_label=tid_label)
        if Config.is_caching():
            store(protected_dict, protected_dict_file)
    else:
        protected_dict = load(protected_dict_file)
    return protected_dict


def load_trajectory_dict(dataset: str,
                         basename: str,
                         tid_label: str = 'trajectory_id') -> Dict[int or str, pd.DataFrame]:
    """
    Load a trajectory dict either from cache or from CSV file.
    :param dataset: Trajectories of which dataset
    :param basename: original or the protection mechanism used
    :param tid_label: label of trajectory id
    :return: Trajectory dictionary
    """
    if Config.is_caching():
        try:
            return load_cached_trajectory_dict(
                base_filename=Config.get_cache_dir(dataset=dataset) + basename, tid_label=tid_label)
        except FileNotFoundError as e:
            log.warning(f"No cached file exist ({e.filename}). Loading CSV instead.")
    return read_trajectories_from_csv(
        filename=Config.get_csv_dir(dataset=dataset) + basename + '.csv', tid_label=tid_label
    )


def split_set_into_xy(lst: List[tuple]):
    """Split training set into (X, Y)"""
    X = np.array([x for x, _ in lst], dtype='object')
    Y = np.array([y for _, y in lst], dtype='object')
    return X, Y


def read_trajectories_from_csv(filename: str,
                               latitude_label: str = 'latitude',
                               longitude_label: str = 'longitude',
                               tid_label: str = 'trajectory_id',
                               user_label: str = 'uid',
                               tid_type: str = 'str',
                               user_type: str = 'int32',
                               date_columns: bool or list = ['timestamp'],
                               as_dict: bool = True
                               ) -> Dict[str or int, pd.DataFrame]:
    """
    Read a CSV file containing multiple trajectories and return a list with each trajectory identified as
    pandas DataFrame.
    :param filename: Path to CSV file
    :param latitude_label: column name of latitude values
    :param longitude_label: column name of longitude values
    :param tid_label: column name of trajectory IDs
    :param user_label: column name of user IDs
    :param tid_type: type of trajectory IDs (e.g., int32)
    :param user_type: type of user IDs (e.g., int32)
    :param date_columns: parse columns as dates
    :param as_dict: Return as dict with the trajectory ID as key. Otherwise, return a list
    :return:
    """
    log.info(f"Reading Trajectories from {filename}.")
    df = pd.read_csv(
        filename,
        # parse_dates=date_columns,
        dtype={tid_label: tid_type, user_label: user_type}
    )
    if type(date_columns) is list and date_columns[0] in df:
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
    conv = {latitude_label: 'latitude', longitude_label: 'longitude', tid_label: 'trajectory_id', user_label: 'uid'}
    df.rename(columns=conv, inplace=True)
    trajectories: Dict[str or int, pd.DataFrame] = {key: t.reset_index(drop=True) for key,
                                                                                      t in df.groupby('trajectory_id')}
    if as_dict:
        return trajectories
    else:
        return list(trajectories.values())


def trajectories_to_csv(trajectories: List[pd.DataFrame] or dict, filename: str):
    """
    Write DataFrames of trajectories into a CSV file
    :param trajectories: List of DataFrames, each representing one trajectory
    :param filename: The file to save to
    :return:
    """
    if type(trajectories) is dict:
        key = int if type(next(iter(trajectories.keys()))) is int or next(iter(trajectories.keys())) else None
        trajectories = [trajectories[key] for key in sorted(trajectories.keys(), key=key)]
    Path(filename).parent.mkdir(exist_ok=True, parents=True)
    pd.concat(trajectories).to_csv(filename, index=False)
    log.info(f"Wrote Trajectories to CSV file {filename}.")


def append_trajectory(t: pd.DataFrame or List[pd.DataFrame], filename: str):
    """
    Add trajectories to an existing CSV file.
    :param t: Single trajectory as DataFrame or list of trajectories
    :param filename: The file to write to
    :return:
    """
    if type(t) is list:
        t = pd.concat(t)
    t.to_csv(filename, mode='a', index=False, header=False)
