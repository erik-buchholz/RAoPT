# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import logging
import pickle
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from raopt.ml import encoder
from raopt.utils.config import Config

logging.basicConfig(level=logging.ERROR)


with open(Config.get_test_dir() + 'resources/sample_trajectories.pickle', 'rb') as f:
    trajs = pickle.load(f)
with open(Config.get_test_dir() + 'resources/sample_encodings.pickle', 'rb') as f:
    encodings = pickle.load(f)


class Test(TestCase):

    def test_encode_trajectory_dict(self):
        dct = {
            1: trajs[1],
            2: trajs[2]
        }
        exp = {
            1: encodings[1],
            2: encodings[2]
        }
        with patch.object(encoder.Config, 'is_caching', return_value=False):
            res = encoder.encode_trajectory_dict(dct)
            for i, key in enumerate(exp):
                np.testing.assert_allclose(res[key], exp[key], err_msg=f'Failed on iteration {i}')
