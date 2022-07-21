# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import logging
from unittest import TestCase

import numpy as np
import pandas as pd
from haversine import haversine, haversine_vector
from raopt.preprocessing import metrics as m

logging.basicConfig(level=logging.ERROR)


class Test(TestCase):

    def setUp(self) -> None:
        self.t1 = {
            'latitude': [
                1, 2, 3, 4, 5, 6, 7, 8, 9
            ],
            'longitude': [
                1, 2, 3, 4, 5, 6, 7, 8, 9
            ]
        }
        self.t2 = {
            'latitude': [
                5, 5, 5, 5, 5, 5, 5, 5, 5
            ],
            'longitude': [
                5, 5, 5, 5, 5, 5, 5, 5, 5
            ]
        }
        super().setUpClass()

    def test_euclidean_distance(self):
        t1 = pd.DataFrame(self.t1)
        t2 = pd.DataFrame(self.t2)
        self.assertEqual(
            0,
            m.euclidean_distance_pd(t1, t1)
        )
        self.assertEqual(
            0,
            m.euclidean_distance_pd(t1, t1, False)
        )
        self.assertEqual(
            0,
            m.euclidean_distance_pd(t2, t2) / 1000
        )
        self.assertEqual(
            0,
            m.euclidean_distance_pd(t2, t2, False)
        )
        self.assertAlmostEqual(
            3.14,
            m.euclidean_distance_pd(t1, t2, False),
            2
        )
        self.assertAlmostEqual(
            349,
            m.euclidean_distance_pd(t1, t2) / 1000,
            0
        )

    def test__async_euclidean_distance(self):
        t1 = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [
            5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
        t2 = np.array([[5, 5], [5, 5], [5, 5], [5, 5], [
            5, 5], [5, 5], [5, 5], [5, 5], [5, 5]])
        with self.assertRaises(ValueError):
            m._async_euclidean_distance(t1, t2)
        t1 = np.append(t1, [[10, 10]], axis=0)
        self.assertAlmostEqual(
            3.14,
            m._async_euclidean_distance(t1, t2, False),
            2
        )
        self.assertAlmostEqual(
            349,
            m._async_euclidean_distance(t1, t2) / 1000,
            0
        )

    def test_hausdorff_distance(self):
        t1 = pd.DataFrame(self.t1)
        t2 = pd.DataFrame(self.t2)
        self.assertEqual(
            0,
            m.hausdorff_distance_pd(t1, t1)
        )
        self.assertEqual(
            m.hausdorff_distance_pd(t1, t2),
            m.hausdorff_distance_pd(t2, t1)
        )
        self.assertAlmostEqual(
            629,
            m.hausdorff_distance_pd(t2, t1) / 1000,
            0
        )
        self.assertAlmostEqual(
            np.sqrt(np.sum(np.square(np.array([5, 5]) - np.array([1, 1])))),
            m.hausdorff_distance_pd(t2, t1, use_haversine=False),
            0
        )

    def test_haversine_vector(self):
        self.assertAlmostEqual(
            941,
            haversine(
                (50.0359, 5.4253), (58.3838, 3.0412)
            ),
            0
        )
        self.assertAlmostEqual(
            132,
            haversine(
                (50., 40.), (51., 41.)
            ),
            0
        )
        self.assertAlmostEqual(
            2356,
            haversine(
                (5., 80.), (15., 99.)
            ),
            0
        )

    def test_haversine_vector_pd(self):
        t1 = [[50.0359, 5.4253]]
        t2 = [[58.3838, 3.0412]]
        self.assertAlmostEqual(
            941,
            haversine_vector(t1, t2)[0],
            0
        )
        t1 = [
            [50.0359, 5.4253],
            [50., 40.],
            [5., 80.]
        ]
        t2 = [
            [58.3838, 3.0412],
            [51., 41.],
            [15., 99.]
        ]
        tmp = m.haversine_vector(t1, t2)
        for i, v in enumerate([941, 132, 2356]):
            self.assertAlmostEqual(
                v,
                tmp[i],
                0
            )

    def test_jaccard_index(self):
        t1 = [
            [0, 0],
            [2, 2],
            [2, 0],
            [0, 2],
            [1, 1]
        ]
        self.assertAlmostEqual(
            1,
            m.jaccard_index(t1, t1)
        )
        t2 = [
            [0, 0],
            [-2, -2],
            [-2, 0],
            [0, -2],
            [-1, -1]
        ]
        self.assertAlmostEqual(
            0,
            m.jaccard_index(t1, t2)
        )
        t3 = [
            [0, 0],
            [1, 1],
            [1, 0],
            [0, 1],
            [0.5, 0.5]
        ]
        self.assertAlmostEqual(
            0.25,
            m.jaccard_index(t1, t3)
        )
        t3 = [
            [0, 0],
            [1, 2],
            [1, 0],
            [0, 2],
            [0.5, 0.5]
        ]
        self.assertAlmostEqual(
            0.5,
            m.jaccard_index(t1, t3)
        )
