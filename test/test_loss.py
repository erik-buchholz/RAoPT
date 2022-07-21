# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
from unittest import TestCase
from keras_preprocessing.sequence import pad_sequences

from raopt.ml import loss
from raopt.preprocessing.metrics import euclidean_distance


class Test(TestCase):
    def test_euclidean_loss(self):
        a_orig = [
            [
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5]
            ],
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1]
            ]
        ]
        b_orig = [
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1]
            ],
            [
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5]
            ]
        ]
        a = tf.constant(a_orig, dtype=tf.float32)
        b = tf.constant(b_orig, dtype=tf.float32)
        haversine_traditional = np.array([euclidean_distance(a_orig[0], b_orig[0]),
                                          euclidean_distance(a_orig[1], b_orig[1])])
        np.testing.assert_allclose(
             haversine_traditional,
             loss.euclidean_loss(a, b),
             atol=1
        )

    def test_euclidean_loss_masking(self):
        true_orig = [
            [
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5]
            ],
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1]
            ]
        ]
        pred_orig = [
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1]
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5]
            ]
        ]
        pred_orig_padded = np.array([
            [
                [1, 1],  # Masked
                [1, 1],  # Masked
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1]
            ],
            [
                [1, 5],  # Masked
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5]
            ]
        ])
        true_orig_padded = pad_sequences(
            true_orig, maxlen=7, padding='pre', dtype='float64'
        )
        y_true = tf.constant(true_orig_padded, dtype=tf.float32)
        y_pred = tf.constant(pred_orig_padded, dtype=tf.float32)
        haversine_traditional = np.array([euclidean_distance(true_orig[0], pred_orig[0]),
                                          euclidean_distance(true_orig[1], pred_orig[1])])
        np.testing.assert_allclose(
             haversine_traditional,
             loss.euclidean_loss(y_true, y_pred),
             atol=1
        )
