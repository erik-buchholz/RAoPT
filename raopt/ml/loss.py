#!/usr/bin/env python3
"""
This files contains custom loss function(s).
"""
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf

EARTH_RADIUS = 6371008.8  # [meter]


def degrees_to_radians(deg):
    """Convert degrees into radians with tensorflow methods."""
    return tf.constant(np.pi) / tf.constant(180, dtype=tf.float32) * deg


def haversine_distance_tf(y_true, y_pred):
    """
    Input Shape: (batch_size, # steps (= trajectory length), # features (= 2 [lat:lon]))
    Output Shape: (batch_size, # steps (= trajectory length))

    This function is based on function haversine_vector from the haversine package:
    https://github.com/mapado/haversine

    :param y_true: Original Trajectories
    :param y_pred: Reconstructed Trajectories
    :return: Haversine distance between trajectories
    """
    # unpack latitude/longitude
    lat_true, lon_true = y_true[:, :, 0], y_true[:, :, 1]
    lat_pred, lon_pred = y_pred[:, :, 0], y_pred[:, :, 1]

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1 = degrees_to_radians(lat_true)
    lon1 = degrees_to_radians(lon_true)
    lat2 = degrees_to_radians(lat_pred)
    lon2 = degrees_to_radians(lon_pred)

    lat = lat2 - lat1
    lng = lon2 - lon1

    d = (tf.math.pow(tf.math.sin(lat * tf.constant(0.5)), 2)
         + tf.math.cos(lat1) * tf.math.cos(lat2) * tf.math.pow(tf.math.sin(lng * tf.constant(0.5)), 2))

    # The protection mechanisms might lead to coordinates that are invalid
    d = tf.where(d > 1.0, tf.ones_like(d), d)
    d = tf.where(d < 0.0, tf.zeros_like(d), d)

    x = tf.constant(2, dtype=tf.float32) * tf.constant(EARTH_RADIUS) * tf.math.asin(tf.math.sqrt(d))
    return x


@tf.autograph.experimental.do_not_convert
def euclidean_loss(y_true, y_pred):
    """
    Utilize the euclidean loss.

    Shape of y_true/y_pred: (batch_size, # steps (= trajectory length), # features (= 2 [lat:lon]))

    :param y_true: True output trajectories
    :param y_pred: Predicted Trajectories
    :return:
    """
    mask_value = tf.constant([0.0, 0.0], dtype=tf.float32)
    mask = tf.cast(tf.keras.backend.all(tf.keras.backend.not_equal(y_true, mask_value), axis=-1), dtype=tf.float32)
    hd = tf.abs((haversine_distance_tf(y_true, y_pred)))  # Shape = (batch_size, trajectory_length)
    mae = tf.math.reduce_sum(hd * mask, axis=1) / tf.math.reduce_sum(mask, axis=1)
    return mae
