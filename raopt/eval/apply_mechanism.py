#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
"""
This file generates protected trajectories that can be used for the training of the attack model.
"""
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from typing import Dict, List

from raopt.dp.sdd import execute_mechanism, sdd, StuckException, cnoise
from raopt.eval.parser import parse_eval
from raopt.utils import logger, helpers
from raopt.utils.config import Config, get_basename
from raopt.utils.helpers import store, load, trajectories_to_csv, load_trajectory_dict


log = logger.configure_root_loger(
        logging.INFO, Config.get_logdir() + "trajectory_generator.log")
mechanisms = {'SDD': sdd, 'CNOISE': cnoise}


def protect_trajectories(
        unprotected_trajectories: List[pd.DataFrame],
        mechanism: str,
        epsilon: float,
        M: float,
        tmp_file: str,
) -> List[pd.DataFrame]:
    """
    Protect the given trajectories with the specified protection mechanism.
    :param unprotected_trajectories: List of original trajectories as pandas DataFrames
    :param mechanism: [SDD, CNoise, PNoise, GNoise]
    :param epsilon: float
    :param M: sensitivity of dataset
    :param tmp_file: File to store intermediate results for the long execution of the SDD mechanism
    :return:
    """
    # Compute reference point
    log.info("Computing Reference Point...")
    lat0, lon0 = helpers.compute_reference_point(unprotected_trajectories)
    log.info(f"Using reference point ({lat0:.2f}, {lon0:.2f}).")
    todo = [(t, mechanism, lat0, lon0, epsilon, M) for t in unprotected_trajectories]
    protected = []
    try:
        if Config.parallelization_enabled():
            n_cpu = cpu_count() if Config.use_all_cpus() else int(cpu_count() * 3 / 4)
            with Pool(n_cpu) as pool:
                for i, res in tqdm(
                        pool.imap_unordered(_generate, todo, chunksize=10),
                        total=len(todo),
                        leave=False,
                        ncols=120):
                    if res is not None:  # Execution failed if None
                        protected.append(res)
                        if mechanism.upper() == 'SDD' and len(protected) % 1000 == 0:
                            store(protected, tmp_file, mute=True)
        else:
            protected = [_generate(t) for t in tqdm(todo, leave=False)]
    except KeyboardInterrupt:
        # Quit gracefully and store current results
        pass
    return protected


def apply_mechanism(
        dataset: str,
        mechanism: str,
        epsilon: float,
        sensitivity: float = 0,
        version: int = 1,
        output_prefix: str = '',
        originals: Dict[str or int, pd.DataFrame] = None,
):
    """
    Apply a protection mechanism to trajectories.
    :param dataset: Dataset to consider
    :param mechanism: The mechanism to protect the trajectories with
    :param epsilon: Epsilon for protection mechanism
    :param sensitivity: Sensitivity of the dataset
    :param version: Version of output file
    :param originals: The trajectories to protect if not None
    :param output_prefix: prefix for output file (e.g., test/train)
    :return:
    """
    M = Config.get_M(dataset) if sensitivity == 0 else sensitivity
    log.info(f'Using M = {M}m.')  # See dp.sdd for meaning

    pdir = Config.get_cache_dir(dataset)
    Path(pdir).mkdir(parents=True, exist_ok=True)
    basename = get_basename(mechanism, epsilon, M, version)
    dict_pickle_file = pdir + f"{output_prefix}{basename}_dict.pickle"
    tmpfile = pdir + f"{output_prefix}{basename}.pickle"
    csv_out_file = Config.get_csv_dir(dataset) + f"{output_prefix}{basename}.csv"
    log.info(f"Will be saving to {csv_out_file}.")
    if Config.is_caching() and Path(dict_pickle_file).exists():
        log.warning(f"Continue with: {dict_pickle_file}")
        protected = list(load(dict_pickle_file).values())
        done = [t['trajectory_id'][0] for t in protected]
    else:
        protected, done = [], []
    if originals is None:
        # Load originals
        originals: Dict[str or int, pd.DataFrame] = load_trajectory_dict(dataset=dataset, basename='originals')

    todo = [originals[t] for t in originals if t not in done]

    protected.extend(protect_trajectories(todo, mechanism=mechanism, epsilon=epsilon, M=M, tmp_file=tmpfile))

    dct = helpers.dictify_trajectories(protected)
    if Config.is_caching():
        store(dct, dict_pickle_file)
    trajectories_to_csv(dct, csv_out_file)
    log.info(f"Stored {len(protected)} generated trajectories.")


def _generate(args: tuple):
    """Call the actual protection mechanism."""
    t: pd.DataFrame
    t, mechanism, lat0, lon0, epsilon, M = args
    i = (t['trajectory_id'][0])
    if mechanism == 'SDD':
        error_counter = 0
        while True:
            try:
                tp = execute_mechanism(t, sdd, lat0=lat0, lon0=lon0, kwargs={
                    'epsilon': epsilon,
                    'M': M,
                    'noisy_endpoints': True,
                    'show_progress': False,
                    'enforce_line11': True
                })
                break
            except StuckException:
                error_counter += 1
                if error_counter >= 1000:
                    # This trajectory seems to be unusable.
                    log.warning(
                        f"Execution aborted for trajectory {i} because 1000 tries unsuccessful."
                    )
                    return i, None
                # log.warning("Restarting SDD mechanism because the algorithm is stuck at line 11.")
    else:
        tp = execute_mechanism(
            t, mechanisms[mechanism], lat0=lat0, lon0=lon0, kwargs={'epsilon': epsilon, 'M': M}
        )
    return i, tp


if __name__ == '__main__':
    args = parse_eval().parse_args()
    apply_mechanism(args.dataset, args.mechanism, args.epsilon,
                    sensitivity=args.sensitivity, version=args.version)
