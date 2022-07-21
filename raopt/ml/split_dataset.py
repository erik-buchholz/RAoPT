#!/usr/bin/env python3
"""
Split a dataset of trajectories into train and test set.
"""
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------

import argparse
import logging
import random
import sys
from pathlib import Path
from timeit import default_timer as timer

from raopt.utils import logger, helpers
from raopt.utils.config import Config

log = logger.configure_root_loger(
    logging.INFO, Config.get_logdir() + "train.log")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Create Train and Test Set.')
    parser.add_argument('original_file', metavar='ORIGINAL_FILE', type=str,
                        help='CSV file with unprotected trajectories.')
    parser.add_argument('protected_file', metavar='PROTECTED_FILE', type=str,
                        help='CSV file with protected trajectories.')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR', type=str,
                        help='Directory to save output files.')
    parser.add_argument('-s', '--split', metavar='SPLIT', type=float, dest='split',
                        default=0.2, help='Test Split from [0;1] [DEFAULT: 0.2]')
    return parser


args = get_parser().parse_args()
###########################################################################
# Check output dir available
###########################################################################
files = ['train_o', 'train_p', 'test_o', 'test_p']
for i, file in enumerate(files):
    files[i] = args.output_dir + file + '.csv'
    if Path(files[i]).exists():
        while True:
            v = input(f"The file {files[i]} already exists. Do you want to overwrite? [Y/N]").upper()
            if v in ['Y', 'N']:
                break
        if v == 'N':
            sys.exit(0)
###########################################################################
# Load Data
###########################################################################
log.info("Loading Data...")
start_time = timer()
originals = helpers.read_trajectories_from_csv(args.original_file)
protected = helpers.read_trajectories_from_csv(args.protected_file)
log.debug(f"Loading data took {timer()-start_time:.2f}s.")
###########################################################################
# Compute indices
###########################################################################
indices = list(protected.keys())
test_idx = random.sample(indices, int(len(indices) * args.split))
train_idx = [i for i in indices if i not in test_idx]
train_o = {i: originals[i] for i in train_idx}
train_p = {i: protected[i] for i in train_idx}
test_o = {i: originals[i] for i in test_idx}
test_p = {i: protected[i] for i in test_idx}
###########################################################################
# Store
###########################################################################
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
helpers.trajectories_to_csv(train_o, files[0])
helpers.trajectories_to_csv(train_p, files[1])
helpers.trajectories_to_csv(test_o, files[2])
helpers.trajectories_to_csv(test_p, files[3])
log.info(f"Stored files into: {args.output_dir}")
