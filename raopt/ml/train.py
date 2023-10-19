#!/usr/bin/env python3
"""
This file is used for training of the model.
"""
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import argparse
import logging
import pickle

from timeit import default_timer as timer

from raopt.ml.model import AttackModel
from raopt.ml import encoder
from raopt.utils import logger, helpers
from raopt.utils.config import Config

log = logger.configure_root_loger(
    logging.INFO, Config.get_logdir() + "train.log")
features = ['latlon', 'hour', 'dow']
vocab_size = {
                'latlon': 2,
                'hour': 24,
                'dow': 7,
}
embedding_size = {
                'latlon': 64,
                'hour': 24,
                'dow': 7,
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='RAoPT Training.')
    parser.add_argument('ofile', metavar='ORIGINAL_FILE', type=str,
                        help='CSV file with unprotected trajectories.')
    parser.add_argument('pfile', metavar='PROTECTED_FILE', type=str,
                        help='CSV file with protected trajectories.')
    parser.add_argument('parameter_file', metavar='PARAMETER_FILE', type=str,
                        help='File to store model parameters.')
    parser.add_argument('max_len', metavar='MAX_LENGTH', type=int,
                        help='Upper Bound of Trajectory Length.')
    parser.add_argument('-b', '--batch', dest='batch', type=int,
                        help='Batch Size', default=Config.get_batch_size())
    parser.add_argument('-e', '--epochs', dest='epochs', type=int,
                        help='Number of Epochs', default=Config.get_epochs())
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float,
                        help='Learning Rate', default=Config.get_learning_rate())
    parser.add_argument('-s', '--early_stop', dest='early_stop', default=Config.get_early_stop(), type=int,
                        help=f'Early Stop Patience (0 = Deactivated) [DEFAULT = {Config.get_early_stop()}]')

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    ###########################################################################
    # Load Data
    ###########################################################################
    log.info("Loading Data...")
    start_time = timer()
    originals = helpers.read_trajectories_from_csv(args.ofile)
    protected = helpers.read_trajectories_from_csv(args.pfile)
    log.debug(f"Loading data took {timer()-start_time:.2f}s.")
    ###########################################################################
    # Model Parameters
    ###########################################################################
    log.info("Compute Parameters...")
    max_length = args.max_len
    # We do not consider the test originals as the attacker does not have access to these
    all_trajectories = list(originals.values()) + list(protected.values()) + helpers.read_trajectories_from_csv(
        args.pfile.replace('train', 'test'), as_dict=False)  # Load test_protected b/c known to attacker
    lat0, lon0 = helpers.compute_reference_point(all_trajectories)
    log.info(f"Reference Point: ({lat0:.2f}, {lon0:.2f})")
    scale_factor = helpers.compute_scaling_factor(all_trajectories, lat0, lon0)
    log.info(f"Scale Factor: ({scale_factor[0]:.2f}, {scale_factor[1]:.2f})")
    pickle.dump({'lat0': lat0, 'lon0': lon0, 'sf': scale_factor},
                open(args.parameter_file.replace('hdf5', '_val.pickle'), 'wb'))
    ###########################################################################
    # Encoding
    ###########################################################################
    encoded_originals = encoder.encode_trajectory_dict(originals, ignore_time=False)
    encoded_protected = encoder.encode_trajectory_dict(protected, ignore_time=False)
    keys = list(encoded_protected.keys())
    trainX = [encoded_protected[i] for i in keys]
    trainY = [encoded_originals[i] for i in keys]
    ###########################################################################
    # Model
    ###########################################################################
    lstm = AttackModel(
            max_length=max_length,
            features=features,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            scale_factor=scale_factor,
            learning_rate=args.learning_rate,
            reference_point=(lat0, lon0),
            parameter_file=args.parameter_file
    )
    log.info('Model Summary:')
    print(lstm.model.summary())
    ###########################################################################
    # Training
    ###########################################################################
    h = lstm.train(
        trainX,
        trainY,
        epochs=args.epochs,
        batch_size=args.batch,
        tensorboard=False,
        use_val_loss=True,
        early_stopping=args.early_stop)

    log.info(f"Training Completed After {len(h.history['loss'])} Epochs.")
