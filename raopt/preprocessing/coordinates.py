#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
"""
Conversion of coordinates between different systems.
"""
import logging

import numpy as np
import pandas as pd
import utm
from numpy import radians, degrees

R_e = 6317000  # [m] Average radius of earth in _meters_
LAT_DISTANCE = 111319.44  # [m] Distance in _meters_ between two latitude degrees
log = logging.getLogger()


def convert_coord_to_meter_jiang(latitudes: np.ndarray, longitudes: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Convert geo-coordinates into meters as proposed by Jiang et al. However, this is a strong approximation and NOT
    actually correct.
    """
    raise DeprecationWarning('Should not be used in production!')
    if (abs(latitudes) > 90).any() or (abs(longitudes) > 180).any():
        log.error(
            f'The given locations are not to represented in longitude and latitude.'
            f'Latitude: {latitudes}\nLongitudes: {longitudes}'
        )
        raise ValueError(
            'The given locations are not to represented as longitude and latitude.')
    alpha = radians(latitudes)
    beta = radians(longitudes)
    x = R_e * np.cos(alpha) * np.cos(beta)
    y = R_e * np.cos(alpha) * np.sin(beta)
    return x, y


def revert_conversion_jiang(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Revert the approx conversion of Jiang et al.
    """
    raise DeprecationWarning('Should not be used in production!')
    if (abs(x) <= 90).all() or (abs(y) <= 180).all():
        log.warning('The DataFrame seems to be in lat/long format already!')
    beta = np.arctan2(y, x)
    lon = degrees(beta)
    inner_term = x / (R_e * np.cos(beta))
    alpha = np.arccos(inner_term)
    lat = degrees(alpha)
    """
    Uses the format "DDD.dddd." In this format, South latitudes and West longitudes are preceded by a minus sign.
    Latitudes range from -90 to 90, and longitudes range from -180 to 180.
    """
    if not (abs(lat) <= 90).all():
        raise ValueError("Not all latitudes below 90.")
    if not (abs(lon) <= 180).all():
        raise ValueError("Not all longitudes below 180.")
    return lat, lon


def to_utm(latitude: np.ndarray, longitude: np.ndarray, zone=None, letter=None) -> (np.ndarray, np.ndarray, int):
    """
    Convert latitude-longitude values into utm
    :param latitude: Latitudes as Array
    :param longitude: Longitudes as array
    :param zone: Enforce a zone
    :param letter: Enforce a zone
    :return: (x, y, zone, letter) in UTM
    """
    x, y, zone, letter = utm.from_latlon(
        latitude, longitude, force_zone_number=zone, force_zone_letter=letter)
    return x, y, zone, letter


def from_utm(x: np.ndarray, y: np.ndarray, zone: int, letter: str) -> (np.ndarray, np.ndarray):
    """
    Convert UTM coordinates into longitude and latitude.
    :param x: UTM x coordinate
    :param y: UTM y coordinate
    :param zone: UTM zone
    :param letter: UTM zone letter
    :return: (latitude, longitude)
    """
    lat, lon = utm.to_latlon(x, y, zone, letter)
    return lat, lon


def to_offset_coords(latitude: np.ndarray, longitude: np.ndarray, lat0, lon0) -> (np.ndarray, np.ndarray):
    """
    Compute cartesian coordinates as distances to a reference point.
    :param latitude: float or np.ndarray
    :param longitude: float or np.ndarray
    :param lat0: Reference point latitude
    :param lon0: Reference point longitude
    :return: (X, Y)
    """
    dist_lat = LAT_DISTANCE  # m
    dist_lon = LAT_DISTANCE * np.cos(radians(lat0))  # m
    x = dist_lon * (longitude - lon0)
    y = dist_lat * (latitude - lat0)
    return x, y


def from_offset_coords(x: np.ndarray, y: np.ndarray, lat0, lon0) -> (np.ndarray, np.ndarray):
    """
    Converts cartesian coordinates based on a reference point to longitude and latitude
    :param x: float or np.ndarray
    :param y: float or np.ndarray
    :param lat0: Reference point latitude
    :param lon0: Reference point longitude
    :return: (latitude, longitude)
    """
    dist_lat = LAT_DISTANCE  # m
    dist_lon = LAT_DISTANCE * np.cos(radians(lat0))  # m
    lat = y / dist_lat + lat0
    lon = x / dist_lon + lon0
    return lat, lon


def check_coordinate_range(x: np.ndarray, y: np.ndarray):
    """
    Print a warning if x and y appear to be latitude and longitude rather than cartesian coordinates. Used for
    protection mechanism of Jiang2013.
    :param x:
    :param y:
    :return:
    """
    pass  # Was used during debugging
    # if is_polar_coord(x, y):
    #     log.warning(
    #         f"Coordinates in meters expected, but the provided coordinates appear to be longitude and latitude."
    #         f"\nX/Latitude: {x};\nY/Longitude: {y}")


def is_polar_coord(latitudes: np.ndarray, longitudes: np.ndarray) -> bool:
    """
    Return true if the values are valid longitudes and latitudes.
    :param longitudes:
    :param latitudes:
    :return:
    """
    return (abs(latitudes) <= 90).all() and (abs(longitudes) <= 180).all()


def is_polar_coord_pd(df: pd.DataFrame, lat_label: str = 'latitude', lon_label: str = 'longitude') -> bool:
    """
    Return true if the DataFrame coordinates are valid longitudes and latitudes.
    :param lat_label: Column name of latitude
    :param lon_label: Column name of longitude
    :param df: DataFrame to Test
    :return:
    """
    lat = df[lat_label]
    lon = df[lon_label]
    return is_polar_coord(lat, lon)


def latlon_to_xy(latitude: np.ndarray, longitude: np.ndarray, lat0: float, lon0: float) -> (np.ndarray, np.ndarray):
    """
    Transform latitude, longitude into cartesian x, y coordinates.
    Use this function as an abstraction layer in order to easily exchange the concrete coordinate variant
    :param latitude: Dezimal Point
    :param longitude: Dezimal Point
    :param lat0: Latitude of reference point
    :param lon0: Longitude of reference point
    """
    return to_offset_coords(latitude, longitude, lat0, lon0)


def xy_to_latlon(x: np.ndarray, y: np.ndarray, lat0: float, lon0: float) -> (np.ndarray, np.ndarray):
    """
    Back-conversion xy to latitude and longitude
    :param x: X-Coordinate (Meters)
    :param y: Y-Coordinate (Meters)
    :param lat0: Latitude of reference point
    :param lon0: Longitude of reference point
    :return:
    """
    return from_offset_coords(x, y, lat0, lon0)


def latlon_to_xy_matrix(t: np.ndarray, lat0: float, lon0: float) -> (np.ndarray, np.ndarray):
    """
    Transform latitude, longitude into cartesian x, y coordinates.
    Use this function as a layer in order to easily exchange the concrete coordinate variant
    :param t: Trajectory to convert
    :param lat0: Latitude of reference point
    :param lon0: Longitude of reference point
    """
    x, y = to_offset_coords(t[:, 0], t[:, 1], lat0, lon0)
    return np.column_stack((x, y))


def xy_to_latlon_matrix(t: np.ndarray, lat0: float, lon0: float) -> (np.ndarray, np.ndarray):
    """
    Back-conversion xy to latitude and longitude
    :param t: Trajectory to convert
    :param lat0: Latitude of reference point
    :param lon0: Longitude of reference point
    :return:
    """
    lat, lon = from_offset_coords(t[:, 0], t[:, 1], lat0, lon0)
    return np.column_stack((lat, lon))
