#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
"""
Measures the reconstruction time for a trained model.
"""
import argparse
import csv
import logging
import random
from timeit import default_timer as timer
from typing import Iterable

import numpy as np
import scipy.stats
from tqdm import tqdm

from raopt.ml.model import AttackModel
from raopt.ml.tensorflow_preamble import TensorflowConfig
from raopt.utils import logger, helpers

CONF_INTERVAL = 0.99


def get_parser() -> argparse.ArgumentParser:
    """Returns argument parser for this evaluation script."""
    parser = argparse.ArgumentParser(description='RAoPT Runtime Evaluation.')
    parser.add_argument('param_file', metavar='PARAMETER_FILE', type=str,
                        help='hdf5 file containing parameters for a trained model')
    parser.add_argument('protected_file', metavar="PROTECTED_FILE", type=str,
                        help="File containing the protected trajectories to reconstruct")
    parser.add_argument('output_file', metavar="OUTPUT_FILE", type=str,
                        help="File to store the measurement results")
    parser.add_argument('-g', '--gpu', help="GPU to use", type=int, default=None)
    parser.add_argument('-s', '--sample', type=int, default=None,
                        help="Number of trajectories to use for the evaluation [DEFAULT: All]")
    return parser


log = logger.configure_root_loger(
    logging.INFO, None)
args = get_parser().parse_args()
TensorflowConfig.configure_tensorflow(args.gpu)

protected = helpers.read_trajectories_from_csv(args.protected_file)

all_t = list(protected.values())
lat0, lon0 = helpers.compute_reference_point(all_t)
log.info(f"Using Reference Point:\t({lat0}, {lon0})")
scale_factor = helpers.compute_scaling_factor(all_t, lat0, lon0)
log.info(f"Using Scaling Factor:\t{scale_factor}")
max_length = max([len(t) for t in all_t])
log.info(f"Using Max Length:\t\t{max_length}")
if args.sample is not None:
    all_t = random.sample(all_t, args.sample)
log.info(f"Number of Trajectories:\t{len(all_t)}")

raopt = AttackModel(
    max_length=max_length,
    scale_factor=scale_factor,
    reference_point=(lat0, lon0),
    parameter_file=args.param_file
)

raopt.model.load_weights(args.param_file)

times = list()

log.setLevel(logging.WARNING)
for t in tqdm(all_t, desc="Measuring"):
    start = timer()
    p = raopt.predict(t)
    runtime = timer() - start
    assert len(t) == len(p)
    times.append(runtime)
log.setLevel(logging.INFO)


def mean_confidence_interval(data: Iterable[float], confidence: float = 0.99) -> (float, float):
    """Compute the mean and the corresponding confidence interval of the
    given data.

    :param confidence: Confidence interval to use, default: 0.99
    :param data: List of number to compute mean and interval for
    """
    a = 1.0 * np.array(data)
    n = len(a)
    if n == 1:
        return a[0], 0
    m, se = np.mean(a), scipy.stats.sem(a)
    h: float = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return float(m), h


mean, e = mean_confidence_interval(times, CONF_INTERVAL)
print(f"The mean prediction time is:\t{mean}s\n"
      f"The {int(CONF_INTERVAL * 100)}%-Confidence Interval is:\t[{mean - e};{mean + e}]s"
      )

with open(args.output_file, 'w', newline='') as fd:
    writer = csv.writer(fd, delimiter=',')
    writer.writerow(['RUN', 'RUNTIME[s]'])
    rows = [(i, t) for i, t in enumerate(times)]
    writer.writerows(rows)
    log.info(f"Wrote all results to {args.output_file}.")
