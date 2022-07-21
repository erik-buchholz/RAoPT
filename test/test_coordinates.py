# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import logging
from unittest import TestCase

import numpy as np

from raopt.preprocessing.coordinates import latlon_to_xy_matrix, xy_to_latlon_matrix

logging.basicConfig(level=logging.ERROR)


class Test(TestCase):

    def test_latlon_to_xy_matrix(self):
        a = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8]
        ])
        lat0 = (39.6 + 40.48) / 2
        lon0 = (115.96 + 117.16) / 2
        b = latlon_to_xy_matrix(a, lat0, lon0)
        res = [[-9763451.95350794, - 4345910.9376],
               [-9593000.62750397, - 4123272.0576],
               [-9422549.30149999, - 3900633.1776],
               [-9252097.97549601, - 3677994.2976]]
        for i, row in enumerate(a):
            for j, v in enumerate(row):
                self.assertAlmostEqual(
                    res[i][j],
                    b[i, j]
                )

    def test_xy_to_latlon_matrix(self):
        a = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8]
        ])
        b = latlon_to_xy_matrix(xy_to_latlon_matrix(a, 2, 2), 2, 2)
        for i, row in enumerate(a):
            for j, v in enumerate(row):
                self.assertAlmostEqual(
                    v,
                    b[i, j]
                )
