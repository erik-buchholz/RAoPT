# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import logging
from unittest import TestCase
from unittest.mock import patch, Mock

import pandas as pd

import raopt.preprocessing.preprocess as pp
from raopt.preprocessing import tdrive
from raopt.utils import config, helpers

logging.basicConfig(level=logging.ERROR)


class Test(TestCase):

    t: pd.DataFrame = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.t = tdrive._read_tdrive_file(
            config.Config.get_test_dir() + 'resources/simple.csv')

    def test_find_bbox(self):
        l1 = []
        with self.assertRaises(ValueError):
            helpers.find_bbox(l1)
        l2 = [self.t]
        res = helpers.find_bbox(l2, quantile=0.99)
        res = (round(i, 3) for i in res)
        self.assertEqual((116.520, 117.482, 39.566, 40.), tuple(res))

    @patch("raopt.preprocessing.tdrive.get_tdrive_data")
    @patch("raopt.preprocessing.tdrive.pd.concat")
    def test_get_single_traj_db(self, concat: Mock, get: Mock):
        get.return_value = 123
        tdrive.get_single_tdrive_db()
        concat.assert_called_once_with(123)

    @patch("raopt.preprocessing.tdrive.get_hua2015_trajs", return_value=list(range(10000)))
    def test_get_li2017_trajs(self, m):
        res = tdrive.get_li2017_trajs()
        self.assertEqual(850, len(res))

    def test_drop_out_of_bounds(self):
        t = self.t.copy()
        self.assertEqual(
            10,
            len(t)
        )
        t = pp.drop_out_of_bounds((t, 116.8, 119, 39, 39.93))
        self.assertEqual(
            4,
            len(t)
        )
