# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import logging
import pickle
import random
import tempfile
from unittest import TestCase

import numpy as np
import pandas as pd

from raopt.utils import helpers
from raopt.utils.config import Config

with open(Config.get_test_dir() + 'resources/sample_trajectories.pickle', 'rb') as f:
    trajs = pickle.load(f)
with open(Config.get_test_dir() + 'resources/sample_encodings.pickle', 'rb') as f:
    encodings = pickle.load(f)

logging.basicConfig(level=logging.ERROR)


class Test(TestCase):

    def test_kmh_to_ms(self):
        self.assertEqual(
            round(helpers.kmh_to_ms(33), 5),
            9.16667
        )
        self.assertEqual(
            5,
            helpers.kmh_to_ms(helpers.ms_to_kmh(5))
        )

    def test_ms_to_kmh(self):
        self.assertEqual(
            round(helpers.ms_to_kmh(33), 5),
            118.8
        )
        self.assertEqual(
            5,
            helpers.ms_to_kmh(helpers.kmh_to_ms(5))
        )

    def test_get_latlon_arrays(self):
        lat, lon = helpers.get_latlon_arrays(trajs[0])
        np.testing.assert_allclose(
            lat,
            np.array([39.85168, 39.85167, 39.85169, 39.85172, 39.85174, 39.85175,
                      39.85176, 39.85208, 39.85199, 39.85196, 39.85182, 39.85184,
                      39.85184, 39.85184, 39.85146, 39.85165, 39.85165, 39.85163,
                      39.85159])
        )
        np.testing.assert_allclose(
            lon,
            np.array([116.69168, 116.69168, 116.69167, 116.69167, 116.69167, 116.69167,
                      116.69167, 116.69172, 116.69172, 116.69171, 116.69171, 116.69171,
                      116.6917, 116.6917, 116.69168, 116.69172, 116.69176, 116.69168,
                      116.69166])
        )

    def test_get_latlon_matrix(self):
        matrix = helpers.get_latlon_matrix(trajs[0])
        np.testing.assert_allclose(
            matrix,
            np.array([[39.85168, 116.69168],
                      [39.85167, 116.69168],
                      [39.85169, 116.69167],
                      [39.85172, 116.69167],
                      [39.85174, 116.69167],
                      [39.85175, 116.69167],
                      [39.85176, 116.69167],
                      [39.85208, 116.69172],
                      [39.85199, 116.69172],
                      [39.85196, 116.69171],
                      [39.85182, 116.69171],
                      [39.85184, 116.69171],
                      [39.85184, 116.6917],
                      [39.85184, 116.6917],
                      [39.85146, 116.69168],
                      [39.85165, 116.69172],
                      [39.85165, 116.69176],
                      [39.85163, 116.69168],
                      [39.85159, 116.69166]])
        )

    def test_set_latlon(self):
        tmp = pd.DataFrame({
            'id': list(range(100))
        })
        a = [2 * i for i in range(100)]
        b = [5 * i for i in range(100)]
        helpers.set_latlon(tmp, a, b)
        self.assertEqual(
            a,
            list(tmp['latitude'])
        )
        self.assertEqual(
            b,
            list(tmp['longitude'])
        )

    def test_compute_reference_point(self):
        tmp = pd.DataFrame({
            'latitude': list(range(66)) + list(range(33)),
            'longitude': list(range(99))
        })
        self.assertEqual(
            (27.0, 49.0),
            helpers.compute_reference_point(tmp)
        )

    def test_compute_scaling_factor(self):
        tmp = pd.DataFrame({
            'latitude': list(range(66)) + list(range(33)),
            'longitude': list(range(99))
        })
        tmp1 = pd.DataFrame({
            'latitude': list(range(200)) + list(range(100)),
            'longitude': list(range(300))
        })
        self.assertEqual(
            (199 - 50, 299 - 50),
            helpers.compute_scaling_factor([tmp, tmp1], 50, 50)
        )

    def test_store(self):
        filename = tempfile.gettempdir() + f'/loadsave.{random.randint(0, 10 ** 10)}'
        obj = random.random()
        helpers.store(obj, filename, mute=True)
        self.assertEqual(
            obj,
            helpers.load(filename, mute=True)
        )

    def test_split_set_into_xy(self):
        lst = [
            ('a', 1),
            ('b', 2),
            ('c', 3)
        ]
        resA, resB = helpers.split_set_into_xy(lst)
        self.assertListEqual(
            list(resA), ['a', 'b', 'c']
        )
        self.assertListEqual(
            [1, 2, 3],
            list(resB)
        )

    def test_read_trajectories_from_csv(self):
        samples = helpers.read_trajectories_from_csv(
            Config.get_test_dir() + 'resources/sample_trajectories.csv',
            tid_label='trajectory_id',
            tid_type='int64',
            user_label='uid',
            user_type='int64',
            date_columns=['timestamp'],
            as_dict=True
        )
        for i in samples:
            pd.testing.assert_frame_equal(
                samples[i],
                trajs[i]
            )
