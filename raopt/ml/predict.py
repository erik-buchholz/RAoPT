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
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict
import pandas as pd

from raopt.ml.model import AttackModel
from raopt.utils import logger, helpers
from raopt.utils.config import Config
from raopt.eval import main as eval_main
from raopt.ml import train

log = logger.configure_root_loger(
    logging.INFO, Config.get_logdir() + "predict.log")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='RAoPT Prediction.')
    parser.add_argument('infile', metavar='INPUT_FILE', type=str,
                        help='CSV file with protected trajectories.')
    parser.add_argument('outfile', metavar='OUTPUT_FILE', type=str, default="",
                        help='Prediction Mode: File to store the reconstructed trajectories (DEFAULT).\t'
                             "Evaluation Mode: Store evaluation results (Only if '-e/--evaluate' specified).")
    parser.add_argument('parameter_file', metavar='PARAMETER_FILE', type=str,
                        help='File to load model parameters from.')
    parser.add_argument('-e', '--evaluate', metavar='ORIGINAL_FILE', type=str,
                        dest='evaluate', default="",
                        help='Activate Evaluation Mode: Compute distances to the original trajectories in this file.')
    parser.add_argument('max_len', metavar='MAX_LENGTH', type=int,
                        help='Upper Bound of Trajectory Length.')
    parser.add_argument('-r', '--reference_point', type=float, nargs=2, metavar=('LATITUDE', 'LONGITUDE'),
                        default=None, help="Reference Point used during training.")
    parser.add_argument('-s', '--scaling_factor', type=float, nargs=2, metavar=('LATITUDE', 'LONGITUDE'),
                        default=None, help="Scaling Factor used during training.")

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    ###########################################################################
    # Load Data
    ###########################################################################
    log.info("Loading Data...")
    start_time = timer()
    protected = helpers.read_trajectories_from_csv(args.infile)
    log.debug(f"Loading data took {timer()-start_time:.2f}s.")
    ###########################################################################
    # Model Parameters
    ###########################################################################
    log.info("Compute Parameters...")
    max_length = args.max_len
    # We do not consider the test originals as the attacker does not have access to these
    val_file = args.parameter_file.replace('hdf5', '_val.pickle')
    if Path(val_file).exists():
        log.info(f"Reading reference point/scale factor from {val_file}.")
        dct = pickle.load(open(val_file, 'rb'))
        lat0, lon0, scale_factor = dct['lat0'], dct['lon0'], dct['sf']
    else:
        lat0, lon0 = args.reference_point if args.reference_point is not None else helpers.compute_reference_point(
            protected.values())
        scale_factor = helpers.compute_scaling_factor(
            protected.values(), lat0, lon0) if args.scaling_factor is None else args.scaling_factor
    log.info(f"Reference Point: ({lat0:.2f}, {lon0:.2f})")
    log.info(f"Scale Factor: ({scale_factor[0]:.2f}, {scale_factor[1]:.2f})")
    features = train.features
    vocab_size = train.vocab_size
    embedding_size = train.embedding_size
    ###########################################################################
    # Model
    ###########################################################################
    lstm = AttackModel(
            max_length=max_length,
            features=features,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            scale_factor=scale_factor,
            reference_point=(lat0, lon0),
            parameter_file=args.parameter_file
    )
    log.info('Model Summary:')
    print(lstm.model.summary())
    ###########################################################################
    # Load Parameters
    ###########################################################################
    lstm.model.load_weights(args.parameter_file)
    ###########################################################################
    # Prediction
    ###########################################################################
    start_time = timer()
    reconstructed = lstm.predict(x=list(protected.values()))
    log.info(f"Prediction Completed in {timer()-start_time:.2f}s.")

    if args.evaluate == "":
        ###########################################################################
        # Prediction Mode: Storage of reconstructed trajectories
        ###########################################################################
        helpers.trajectories_to_csv(reconstructed, args.outfile)
    else:
        ###########################################################################
        # Evaluation Mode: Compute Distance to Originals
        ###########################################################################
        reconstructed: Dict[str, pd.DataFrame] = {str(p['trajectory_id'][0]): p for p in reconstructed}
        originals = helpers.read_trajectories_from_csv(args.evaluate)
        distances = eval_main.parallelized_distance_computation(
            test_orig=originals, reconstructed=reconstructed, test_p=protected, fold=0)
        # Store results
        df = pd.DataFrame(distances)
        df.to_csv(args.outfile, index=False)
        log.info(f"Wrote distances to: {args.outfile}")
        eval_main.print_results_detailed(df)
