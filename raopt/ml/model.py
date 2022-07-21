#!/usr/bin/env python3
"""
This files contains the actual model.
"""
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import copy
import logging
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from raopt.ml.tensorflow_preamble import TensorflowConfig
TensorflowConfig.configure_tensorflow()

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Bidirectional, \
    Masking, Concatenate, TimeDistributed, LSTM
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback

from raopt.ml.encoder import encode_trajectory, decode_trajectory
from raopt.ml.loss import euclidean_loss
from raopt.utils.config import Config

# CONSTANTS ####################################################################
ADD_REF_POINT_IN_MODEL = False  # If this option is active, we add the reference point at the end of the model
# such that the loss function computes the distances on the final latitude/longitude rather than on offsets
# Our tests show that model quality improves if deactivated.
SCALE_IN_MODEL = True  # If active, the re-scaling with the scaling factor is done within the model
# Evaluation yields better results if activated
param_path = Config.get_parameter_path()
Path(param_path).mkdir(exist_ok=True)
LEARNING_RATE = Config.get_learning_rate()
MODEL_NAME = 'RAoPT'
FEATURES = ['latlon', 'hour', 'dow']
log = logging.getLogger()
# CONSTANTS ####################################################################


def _encode(x: List[pd.DataFrame]) -> np.ndarray:
    """
    Encode a list of trajectories represented as pandas.DataFrame
    :param x: list of trajectories as DataFrames
    :return: list of trajectories as numpy arrays
    """
    encodings = [encode_trajectory(t) for t in tqdm(x, total=len(x), desc='Encoding', leave=False)]
    return encodings


def _decode(x: np.ndarray, originals: List[pd.DataFrame], ignore_time: bool = True) -> List[pd.DataFrame]:
    """
    Decode a predicted list of trajectories
    :param x: list of trajectories as numpy arrays
    :param originals: list of the original trajectories containing additional information such a tid
    :param ignore_time: By default the predicted trajectories only contain location information. Otherwise,
    set this flag to False
    :return: list of trajectories as DataFrames
    """

    if 'taxi_id' in originals[0]:
        uid = 'taxi_id'
    else:
        uid = 'uid'

    decodings = []
    for i, t in tqdm(enumerate(x), leave=False, desc='Decoding Trajectories', total=len(x)):
        decoded = decode_trajectory(t, ignore_time=ignore_time)
        decoded['trajectory_id'] = originals[i]['trajectory_id'][0]
        decoded['uid'] = originals[i][uid][0]
        if 'timestamp' in originals[i]:
            decoded['timestamp'] = originals[i]['timestamp']
        decodings.append(decoded)
    return decodings


class AttackModel:

    def __init__(
            self,
            reference_point: (float, float),
            scale_factor: (float, float),
            max_length: int,
            features: List[str] = FEATURES,
            vocab_size: Dict[str, int] = None,
            embedding_size: Dict[str, int] = None,
            learning_rate: float = LEARNING_RATE,
            parameter_file: str = None,
    ):
        """
        Initialize the model
        :param max_length: Maximal length of any trajectory (for padding)
        :param features: List of features used by the model. First feature has to be latlon!
        :param vocab_size: Dict stating the number of values for each feature
                           (e.g., 7 for an onehot encoded day of week)
        :param scale_factor: Scale factor for latitude and longitude (lat: float, lon: float)
        :param learning_rate: The Adam learning rate to use
        :param parameter_file: File to save parameters of the model
        :param reference_point: (lat, lon)
        """
        self.history = None
        self.max_length = max_length
        self.features = features
        assert self.features[0] == 'latlon'
        if vocab_size is None:
            self.vocab_size = {
                'latlon': 2,
                'hour': 24,
                'dow': 7,
            }
        else:
            self.vocab_size = vocab_size
        if embedding_size is None:
            self.embedding_size = {
                'latlon': 64,
                'hour': 24,
                'dow': 7,
            }
        else:
            self.embedding_size = embedding_size
        self.num_features = sum(self.vocab_size[k] for k in features)
        self.scale_factor = scale_factor
        self.lat0, self.lon0 = reference_point
        self.param_file = parameter_file

        # Define the optimizer
        self.optimizer = Adam(learning_rate)  # Good default choice

        # Build Model
        self.model = self.build_model()

        # Loss: For very bad protection mechanisms like CNoise 0.01 the euclidean loss is unstable
        # because the coordinates are not true latitudes/longitudes
        loss = euclidean_loss if self.scale_factor[0] < 90 else 'mse'

        # Compile Model
        self.model.compile(
            loss=loss,
            # loss='mse',
            optimizer=self.optimizer,
            # metrics=['accuracy'],  # Metrics to track in addition to loss
        )

    def build_model(self) -> Model:
        """
        Return an LSTM-based model that reconstructs original trajectories from protected versions.

        :return: The keras.models.Model.
        """

        # Input layer-----------------------------------------------------------
        inputs = Input(shape=(self.max_length, self.num_features),
                       name="Input_Encoding")

        # Masking Layer! (Inputs are padded)------------------------------------
        masked = Masking(name="Mask_Padding")(inputs)

        # Split Inputs ---------------------------------------------------------
        # Example for T-Drive dataset:
        # Split along feature axis into 3 inputs. I.e., the result is a list containing 3 arrays:
        # in_latlon.shape = (# samples, self.max_length, 2)  -> Latitude & Longitude
        # in_hour.shape = (# samples, self.max_length, 24)  -> Hour
        # in_dow.shape = (# samples, self.max_length, 7)  -> Day of Week
        split_points = [
            self.vocab_size[feature]
            for feature in self.features

        ]
        in_elements = tf.split(
            masked, split_points,
            axis=-1,
            name="split_features"
        )
        # ----------------------------------------------------------------------

        # Embedding Layer-------------------------------------------------------
        embeddings = []
        for i, feature in enumerate(self.features):
            emb = Dense(units=self.embedding_size[feature],
                        activation='relu',
                        name=f'Embedding_{feature}_dense')
            embedding = TimeDistributed(emb, name=f'Embedding_{feature}')(in_elements[i])
            embeddings.append(embedding)
        # ----------------------------------------------------------------------

        # Feature Fusion -------------------------------------------------------
        # Some datasets only contain latitude and longitude. In that case, we do
        # not need to fuse any features
        if len(embeddings) > 1:
            concatenation = Concatenate(axis=-1, name="Join_Features")(embeddings)
        else:
            concatenation = embeddings[0]
        feature_fusion = Dense(
            units=100,
            activation='relu',
            name='Feature_Fusion'
        )(concatenation)
        # ----------------------------------------------------------------------

        # Bidirectional LSTM layer ---------------------------------------------
        bidirectional_lstm = \
            Bidirectional(
                LSTM(units=100,
                     return_sequences=True,
                     ),
                # merge_mode=None,
                name="Bidirectional_LSTM",
            )(feature_fusion)
        # ----------------------------------------------------------------------

        # Output Layer ---------------------------------------------------------
        output_lat = TimeDistributed(
            Dense(1, activation='tanh'), name='Output_lat')(bidirectional_lstm)
        output_lon = TimeDistributed(
            Dense(1, activation='tanh'), name='Output_lon')(bidirectional_lstm)
        if SCALE_IN_MODEL:
            # We need to scale up because tanh outputs are between -1 an 1
            offset = (self.lat0, self.lon0) if ADD_REF_POINT_IN_MODEL else (0., 0.)
            lat_scaled = TimeDistributed(Rescaling(scale=self.scale_factor[0], offset=offset[0]),
                                         name='Output_lat_scaled')(output_lat)
            lon_scaled = TimeDistributed(Rescaling(scale=self.scale_factor[1], offset=offset[1]),
                                         name='Output_lon_scaled')(output_lon)
            outputs = [lat_scaled, lon_scaled]
        else:
            outputs = [output_lat, output_lon]

        # We actually just need to compute latitude and longitude
        # This code assumes that all other features are categorical
        # for feature in self.features[1:]:
        #     out = TimeDistributed(Dense(
        #         self.vocab_size[f'{feature}'], activation='softmax'), name=f'Output_{feature}')(bidirectional_lstm)
        #     outputs.append(out)

        if len(outputs) > 1:
            output = Concatenate(axis=-1, name="Output_Concatenation")(outputs)
        else:
            output = outputs

        model = Model(inputs=inputs, outputs=output,
                      name=MODEL_NAME)

        # Save Model-----------------------------------------------------------
        file = param_path + f"{MODEL_NAME}.json"
        with open(file, 'w') as f:
            f.write(model.to_json())
        # ----------------------------------------------------------------------

        return model

    def preprocess_x(self, x: np.ndarray) -> np.ndarray:
        """
        Pre-process data before feeding it into the model.
        I.e.,
        1. Subtract the reference point from longitude and latitude
        2. Pad all trajectories to the same length for batch processing
        :param x: Input data samples
        :return: Pre-processed values
        """

        # Copy b/c pre-processing modifies data
        x = copy.deepcopy(x)

        # Subtract reference point
        for i in range(len(x)):
            x[i][:, 0] -= self.lat0
            x[i][:, 1] -= self.lon0

        # Padding
        x = pad_sequences(
            x, maxlen=self.max_length, padding='pre', dtype='float64'
        )

        return x

    def preprocess_y(self, y: np.ndarray) -> np.ndarray:
        """Drop everything except for latitude and longitude."""

        # Copy b/c pre-processing modifies data
        y = copy.deepcopy(y)

        if not ADD_REF_POINT_IN_MODEL:
            # Subtract reference point
            for i in range(len(y)):
                y[i][:, 0] -= self.lat0
                y[i][:, 1] -= self.lon0

        # Padding
        y = pad_sequences(
            y, maxlen=self.max_length, padding='pre', dtype='float64'
        )

        if not SCALE_IN_MODEL:
            # Scale longitude and latitude
            # ***Moved into Model via Rescaling Layer***
            y[:, :, 0] /= self.scale_factor[0]
            y[:, :, 1] /= self.scale_factor[1]

        # Only use lat/lon information
        y = y[:, :, :2]

        return y

    def postprocess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Revert the pre-processing in case the model is used for prediction.
        :param x: The input trajectories (only required for the length of each trajectory to remove padding)
        :param y: The predicted trajectories
        :return: post-processed predicted trajectories
        """
        if not SCALE_IN_MODEL:
            # Revert Scaling
            y[:, :, 0] *= self.scale_factor[0]
            y[:, :, 1] *= self.scale_factor[1]

        # Add reference point
        if not ADD_REF_POINT_IN_MODEL:
            y[:, :, 0] += self.lat0
            y[:, :, 1] += self.lon0

        # Remove padded values
        result = []
        for i in range(len(y)):
            n = len(y[i]) - len(x[i])
            result.append(y[i][n:])
        return result

    def train(self, x: np.ndarray,
              y: np.ndarray,
              epochs: int = Config.get_epochs(),
              batch_size: int = Config.get_batch_size(),
              val_x: np.ndarray = None,
              val_y: np.ndarray = None,
              tensorboard: bool = False,
              use_val_loss: bool = False,
              early_stopping: int = Config.get_early_stop()
              ):
        """
        :param x: Protected Trajectories (Encoded)
                    x.shape = (# samples, trajectory length, # features)
        :param y: Original Trajectories (Encoded)
                    y.shape = (# samples, trajectory length, # features)
        :param epochs: The number of epochs to train for
        :param batch_size: Batch Size
        :param val_x: Validation X
        :param val_y: Validation X
        :param tensorboard: If Tensorboard should be used
        :param use_val_loss: Use validation_split in training and val_loss as metric
        :param early_stopping: Early Stop Patience or 0 = deactivate
        :return:
        """

        # Pre-processing of x
        x_train = self.preprocess_x(x)

        # Pre-processing of y
        y_train = self.preprocess_y(y)

        # Validation Set or Split used?
        if val_x is None:
            validation_data = None
        else:
            # Pre-Process Validation Data, too
            val_x = self.preprocess_x(val_x)
            val_y = self.preprocess_y(val_y)
            validation_data = (val_x, val_y)
        if use_val_loss:
            validation_split = 0.1
            stop_metric = 'val_loss'
        else:
            validation_split = 0.0
            stop_metric = 'loss'

        # Checkpointing
        if self.param_file is None:
            checkpoint_path = param_path + \
                MODEL_NAME + "_epoch{epoch:03d}.hdf5"
        else:
            checkpoint_path = self.param_file
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,  # where to save
            monitor=stop_metric,  # metric to monitor
            save_best_only=True,  # only save on improved accuracy
            mode='min',  # Max for accuracy, Min for loss etc.
            save_weights_only=True,  # Don't save the model
            verbose=0,
            save_freq='epoch'  # Save all X batches or 'epoch'
        )

        callbacks = [
            TqdmCallback(verbose=1, leave=False),  # verbose: 0 -> Only show epoch bar, 1-> Show batches
            checkpoint,
        ]

        if early_stopping > 0:
            callbacks.append(
                EarlyStopping(
                    monitor=stop_metric,
                    patience=early_stopping,  # Only if stagnant for X epochs
                    mode='min',
                    restore_best_weights=True,
                    verbose=0
                )
            )

        # Tensorboard
        if tensorboard:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=Config.get_tensorboard_dir())
            callbacks.append(tensorboard_callback)

        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_split=validation_split
        )

        return self.history

    def _predict(self, x: np.ndarray) -> np.ndarray:
        # Preprocessing
        x_test = self.preprocess_x(x)

        prediction = self.model.predict(x_test)

        return self.postprocess(x, prediction)

    def predict(self, x: np.ndarray or list or pd.DataFrame) -> np.ndarray or List[pd.DataFrame] or pd.DataFrame:
        """
        Recover trajectories through the trained model.
        Loads parameters from file.
        :param x: The protected trajectories to reconstruct, either as list of DataFrames or (Encoded)
        :return: List/array of reconstructed trajectories (Same Format as input)
        """

        if type(x) is pd.DataFrame or len(x[0]) == self.num_features:
            # The function was called with a single element rather than multiple ones
            x = [x]
            single = True
        else:
            single = False

        if type(x[0]) is np.ndarray:
            # Input is already encoded
            result = self._predict(x)
        elif type(x[0]) is pd.DataFrame:
            # Encoding necessary
            start = timer()
            encoded = _encode(x)
            log.info(f"Encoded trajectories in {round(timer() - start)}s")

            start = timer()
            prediction = self._predict(encoded)
            log.info(f"Prediction in {round(timer() - start)}s")

            start = timer()
            decoded = _decode(prediction, x)
            log.info(f"Decoded trajectories in {round(timer() - start)}s")
            result = decoded
        else:
            log.error(f"Unexpected input type {type(x[0])}")
            raise ValueError(f"Unexpected input type {type(x[0])}")

        if single:
            return result[0]
        else:
            return result

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model performance.
        :param x: Protected Trajectories (Encoded)
        :param y: Original Trajectories (Encoded)
        :return:
        """

        # Preprocessing
        x_test = self.preprocess_x(x)
        y_test = self.preprocess_y(y)

        # Evaluate performance
        d = self.model.evaluate(x_test, y_test, return_dict=True)
        return d
