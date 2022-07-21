# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import logging
import os
import pickle
from unittest.mock import patch, Mock, mock_open

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from raopt.ml.tensorflow_preamble import TensorflowConfig
TensorflowConfig.configure_tensorflow()
import tensorflow as tf
from raopt.ml.encoder import encode_trajectory, decode_trajectory
from raopt.utils.config import Config
from raopt.ml import model

logging.basicConfig(level=logging.ERROR)

with open(Config.get_test_dir() + 'resources/sample_trajectories.pickle', 'rb') as f:
    subset = pickle.load(f)
with open(Config.get_test_dir() + 'resources/sample_encodings.pickle', 'rb') as f:
    encodings = pickle.load(f)


@patch.object(model, 'ADD_REF_POINT_IN_MODEL', False)
class Test(tf.test.TestCase):

    dataset = 'TDRIVE'

    def setUp(self):
        # super(Test, self).setUp()
        self.x = [
            np.array([[5, 6], [3, 7], [5, 2], [7, 9]]).astype(np.float32),
            np.array([[5, 6], [3, 7], [5, 2]]).astype(np.float32),
            np.array([[3, 3], [5, 9], [2, 2], [9, 9], [7., 8.]]).astype(np.float32),
        ]
        self.y = np.array([
            np.array([[0., 0.]] * 6 + [[3., 3.], [1., 4.], [3., -1.], [5., 6.]]),
            np.array([[0., 0.]] * 7 + [[3., 3.], [1., 4.], [3., -1.]]),
            np.array([[0., 0.]] * 5 + [[1., 0.], [3., 6.], [0., -1.], [7., 6.], [5., 5.]])
        ])

    def test__encode(self):
        encoded = model._encode(subset)
        for i, t in enumerate(subset):
            self.assertTrue(
                (encoded[i] ==
                 encode_trajectory(t)).all()
            )

    def test__decode(self):
        decoded = model._decode(encodings, subset, ignore_time=False)
        for i, t in enumerate(encodings):
            self.assertTrue(
                (decoded[i][['latitude', 'longitude']].to_numpy() ==
                 decode_trajectory(t)[['latitude', 'longitude']].to_numpy()).all()
            )

    @patch("builtins.open", mock_open())
    def test_attack_model(self):
        m = model.AttackModel(
            max_length=10,
            features=model.FEATURES,
            scale_factor=(1., 1.),
            reference_point=(0., 0.)
        )
        self.assertIsNotNone(m.model)
        with self.assertRaises(AssertionError):
            model.AttackModel(
                max_length=10,
                features=['hour', 'dow'],
                scale_factor=(1., 1.),
                reference_point=(0., 0.)
            )

    @patch("builtins.open", mock_open())
    def test_preprocess_x(self):
        m = model.AttackModel(
            max_length=10,
            features=['latlon'],
            scale_factor=(1., 1.),
            reference_point=(2., 3.)
        )
        exp_y = self.y
        y = m.preprocess_x(self.x)
        np.testing.assert_allclose(y, exp_y)

    @patch("builtins.open", mock_open())
    def test_postprocess_x(self):
        m = model.AttackModel(
            max_length=10,
            features=['latlon'],
            scale_factor=(1., 1.),
            reference_point=(2., 3.)
        )
        exp_x = self.x
        y = self.y
        x = m.postprocess(exp_x, y)
        for i, _ in enumerate(x):
            np.testing.assert_allclose(x[i], exp_x[i], err_msg=f'Failed on i = {i}.')

    @patch("builtins.open", mock_open())
    def test_scaling(self):
        m = model.AttackModel(
            max_length=10,
            features=['latlon'],
            scale_factor=(10., 10.),
            reference_point=(2., -1.)
        )

        y = m.preprocess_x(self.x)
        new_x = m.postprocess(self.x, y)
        for i, _ in enumerate(self.x):
            np.testing.assert_allclose(new_x[i], self.x[i], err_msg=f'Failed on i = {i}.')

    @patch("builtins.open", mock_open())
    def test_evaluate(self):
        m = model.AttackModel(
            max_length=10,
            features=['latlon'],
            scale_factor=(10., 10.),
            reference_point=(2., -1.)
        )
        mock = Mock()
        mock.return_value = 5
        with patch.object(m.model, 'evaluate', mock):
            self.assertEqual(
                m.evaluate(self.x, self.y),
                5
            )
        mock.assert_called_once()

    @patch("builtins.open", mock_open())
    def test_predict(self):
        m = model.AttackModel(
            max_length=10,
            features=['latlon'],
            scale_factor=(10., 10.),
            reference_point=(2., -1.)
        )
        # Test for encoded List
        mock = Mock()
        mock.side_effect = lambda x: x
        with patch.object(m.model, 'predict', mock):
            res = m.predict(self.x)
        for i, _ in enumerate(self.x):
            np.testing.assert_allclose(res[i], self.x[i], err_msg=f'Failed on i = {i}.')
        # Test with single element
        t = self.x[0]
        with patch.object(m.model, 'predict', mock):
            res = m.predict(t)
        np.testing.assert_allclose(res, t)
