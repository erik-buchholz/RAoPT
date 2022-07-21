#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
"""
This files contains the trajectory encoding that has to be performed to feed the trajectories into the model.
"""
import logging
import multiprocessing as mp
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from raopt.utils.config import Config
from raopt.utils.helpers import store, load, load_trajectory_dict

log = logging.getLogger()


def encode_timestamp(df: pd.DataFrame, label: str) -> (np.ndarray, np.ndarray):
    """
    Encode a timestamp into hour-of-day and day-of-week.
    :param df: Trajectory
    :param label: Timestamp column's label
    :return: (hour-of-day, day-of-week) both as one-hot encodings
    """
    hour = df[label].dt.hour.to_numpy()
    dow = df[label].dt.dayofweek
    try:
        hour_encoded = tf.keras.utils.to_categorical(hour, num_classes=24)
        dow_encoded = tf.keras.utils.to_categorical(dow, num_classes=7)
    except IndexError as e:
        log.error(
            f"There is an error in the time information of trajectory {df['trajectory_id']}."
        )
        raise e
    return hour_encoded, dow_encoded


def encode_trajectory(t: pd.DataFrame,
                      ignore_time: bool = False,
                      categorical_features: list = list(),
                      vocab_size: dict = dict(),
                      numerical_features: list = list()
                      ) -> np.ndarray:
    """
    Encode a trajectory as matrix.
    Each row is one stop, each column one feature.
    :param t: Trajectory
    :param ignore_time: Only convert latitude and longitude
    :param categorical_features: List of categorical features to encode
    :param vocab_size: A dictionary stating the size of the encoding for each categorical feature as values
    :param numerical_features: List of features that are supposed to be encoded as floats.
    :return: out.shape = (len(t), # Features)
    """
    # Drop all NaN values before encoding
    t.dropna(inplace=True)
    t.reset_index(inplace=True, drop=True)
    lat = t['latitude'].to_numpy().reshape((-1, 1))
    lon = t['longitude'].to_numpy().reshape((-1, 1))
    parts = [lat, lon]
    if not ignore_time and 'timestamp' in t:
        parts.extend(encode_timestamp(t, 'timestamp'))
    # Categorical attributes
    for f in categorical_features:
        parts.append(
            tf.keras.utils.to_categorical(t[f], num_classes=vocab_size[f])
        )
    for f in numerical_features:
        parts.append(t[f].to_numpy().reshape((-1, 1)))
    res = np.concatenate(parts, axis=1)
    return res


def decode_trajectory(t: np.ndarray, ignore_time: bool = False) -> pd.DataFrame:
    """
    Decode the trajectory encoded with encode_trajectory.
    :param t: The encoded trajectory
    :param ignore_time: Only convert latitude and longitude
    :return:
    """
    d = {}
    if ignore_time:
        if len(t) > 2:
            lat, lon, _ = np.split(t, [1, 2], axis=-1)
        else:
            lat, lon = np.split(t, [1], axis=-1)
    else:
        lat, lon, hour_encoded, dow_encoded = np.split(t, [1, 2, 26], axis=-1)
        hour = np.argmax(hour_encoded, axis=-1)
        dow = np.argmax(dow_encoded, axis=-1)
        d['hour'] = hour
        d['dow'] = dow
    lat = lat.flatten()
    d['latitude'] = lat
    lon = lon.flatten()
    d['longitude'] = lon

    res = pd.DataFrame(d)
    return res


def subtract_reference_point(t: np.ndarray, lat0: float, lon0: float) -> pd.DataFrame:
    """
    Subtract the reference point from longitude and latitude
    :param t: Trajectory to modify
    :param lat0: Latitude of reference point
    :param lon0: Longitude of reference point
    :return:
    """
    t['latitude'] -= lat0
    t['longitude'] -= lon0
    return t


def add_reference_point(t: np.ndarray, lat0: float, lon0: float) -> pd.DataFrame:
    """
    Add the reference point from longitude and latitude
    :param t: Trajectory to modify
    :param lat0: Latitude of reference point
    :param lon0: Longitude of reference point
    :return:
    """
    t['latitude'] += lat0
    t['longitude'] += lon0
    return t


def _encode_wrapper(args):
    return encode_trajectory(*args)


def encode_trajectory_dict(trajectory_dict: Dict[str or int, pd.DataFrame],
                           ignore_time: bool = False) -> Dict[str or int, np.ndarray]:
    """
    Create a dict containing encoded trajectories from a dict containing pandas.DataFrame trajectories
    :param trajectory_dict: {trajectory_id: pd.DataFrame}
    :param ignore_time: Only convert latitude and longitude
    :return: encoded_dict: {trajectory_id: np.ndarray}
    """
    log.info("Encoding trajectories...")
    start = timer()
    if Config.parallelization_enabled():
        keys = list(trajectory_dict.keys())
        generator = ((trajectory_dict[key], ignore_time) for key in keys)
        with mp.Pool(mp.cpu_count()) as pool:
            encoded_dict = dict(zip(keys, pool.map(_encode_wrapper, generator)))
    else:
        encoded_dict = {
            key: encode_trajectory(trajectory_dict[key], ignore_time=ignore_time) for key in tqdm(
                trajectory_dict, desc='Encoding', total=len(trajectory_dict))
        }
    log.info(f"Encoded trajectories in {round(timer() - start)}s.")
    return encoded_dict


def get_encoded_trajectory_dict(dataset: str, basename: str, encoded_file: str = None, ignore_time: bool = False,
                                trajectory_dict:  Dict[str or int, pd.DataFrame] = None) -> dict:
    """
    Load encoded trajectories from cache file if the file exists or generate the encodings.
    :param dataset: Trajectories of which dataset
    :param basename: original or the protection mechanism used
    :param encoded_file: File to store the encoded trajectories
    :param ignore_time: Only convert latitude and longitude
    :param trajectory_dict: provide a dict of trajectories to avoid loading from file
    :return:
    """
    encoded_file = Config.get_cache_dir(dataset=dataset) + basename + "_encoded_dict.pickle" if \
        encoded_file is None else encoded_file
    if Config.is_caching() and Path(encoded_file).exists():
        encoded_dict = load(encoded_file)
    else:
        if trajectory_dict is None:
            trajectory_dict = load_trajectory_dict(dataset=dataset, basename=basename)
        encoded_dict = encode_trajectory_dict(trajectory_dict, ignore_time=ignore_time)
        if Config.is_caching():
            store(encoded_dict, encoded_file)
    return encoded_dict


class SemanticEncoder:
    """Class used to encode semantic trajectories that contain more semantic information than timestamps."""

    def __init__(self, categorical_features: List[str], numerical_features: List[str]):
        self.vocabulary = {}
        self.encoders = {}
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

    def transform_categorical(
            self,
            trajectories: dict,
    ) -> dict:
        """
        Transform all categorical attributes of the trajectories into integers
        :param trajectories: dict[trajectory_id] = pd.DataFrame(trajectory)
        :return: Transformed trajectories
        """
        df = pd.concat(list(trajectories.values()))
        for f in self.categorical_features:
            if f not in self.encoders:
                # Fit encoder
                self.encoders[f] = LabelEncoder().fit(df[f])
            df[f] = self.encoders[f].transform(df[f])
        trajectories = {key: t.reset_index(drop=True) for key, t in df.groupby('trajectory_id')}
        return trajectories

    def get_vocab_sizes(self):
        """
        Return the number of values for each categorical attribute
        (transform_categorical needs to be called beforehand!)
        :return: dict[feature_name] = # values
        """
        return {f: len(self.encoders[f].classes_) for f in self.categorical_features}

    def encode_semantic(self, trajectories: dict) -> np.ndarray:
        """Encode the given dictionary of semantic trajectories.
        :param trajectories: dict[trajectory_id] = pd.DataFrame(trajectory)
        :return: Encoded trajectories
        """
        trajectories = self.transform_categorical(trajectories)
        encoded = {k: encode_trajectory(trajectories[k],
                                        categorical_features=self.categorical_features,
                                        vocab_size=self.get_vocab_sizes(),
                                        numerical_features=self.numerical_features
                                        ) for k in trajectories}
        return encoded



