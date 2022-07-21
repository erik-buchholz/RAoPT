#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
"""
Reads the cases from config and performs missing evaluations.
"""
import argparse
import copy
import logging
import traceback
import multiprocessing as mp
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Iterable, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold

from raopt.ml.tensorflow_preamble import TensorflowConfig
from raopt.preprocessing.metrics import euclidean_distance_pd, hausdorff_distance_pd, jaccard_index_pd


def get_parser() -> argparse.ArgumentParser:
    """Returns argument parser for this main evaluation script."""
    parser = argparse.ArgumentParser(description='RAoPT Evaluation.')
    parser.add_argument('-c', '--case', metavar='CASE', type=str,
                        help='Execute specific case', default=None)
    parser.add_argument('-g', '--gpu', help="GPU to use", type=int, default=None)
    return parser


CASE = None
if __name__ == '__main__':
    from raopt.utils import logger
    log = logger.configure_root_loger(
        logging.INFO, None)
    args = get_parser().parse_args()
    TensorflowConfig.configure_tensorflow(args.gpu)
    CASE = args.case
else:
    log = logging.getLogger()


# ###############################CONSTANTS###############################################
from raopt.utils.config import Config, get_basename
n_kfold = int(Config.get('DEFAULT', 'KFOLD'))
kfold_random_state = 7
learning_rate = Config.get_learning_rate()
EPOCHS = Config.get_epochs()
batch_size = Config.get_batch_size()
# ###############################CONSTANTS###############################################


def get_cases_file() -> str:
    """Return the filename of the file containing all evaluation cases."""
    return Config.get_basedir() + 'config/cases.csv'


def read_cases(filename: str) -> pd.DataFrame:
    """Read the case file and return pandas DataFrame."""
    df = pd.read_csv(
        filename,
        delimiter=',',
        dtype={
            'ID': str,
            'Epsilon Train': float,
            'Epsilon Test': float,
            'M Train': float,
            'M Test': float,
            'Done': bool
        }
    )
    return df


def get_cases() -> List[dict]:
    """Get all evaluation cases represented as dicts."""
    filename = get_cases_file()
    df = read_cases(filename)
    return df.to_dict('records')


def mark_case_complete(case_id: str, mark_as=True, filename=get_cases_file()) -> None:
    """Mark an evaluation case as done within the case file."""
    cases = read_cases(filename)
    cases.loc[cases['ID'] == str(case_id), 'Done'] = mark_as
    cases.to_csv(filename, index=False)


def compute_distances(val: (pd.DataFrame, pd.DataFrame, pd.DataFrame, int)) -> dict:
    """
    Compute our metrics for the given tuple.
    :param val: (Original Traj., Reconstructed Traj., Protected Traj., Fold Number)
    :return: All metrics as dict
    """
    (o, r, p, fold) = val
    res = \
        {
            'Fold': fold,
            'Euclidean Original - Protected':
                euclidean_distance_pd(o, p, use_haversine=True, disable_checks=True),
            'Euclidean Original - Reconstructed':
                euclidean_distance_pd(o, r, use_haversine=True, disable_checks=True),
            'Hausdorff Original - Protected':
                hausdorff_distance_pd(o, p, use_haversine=True, disable_checks=True),
            'Hausdorff Original - Reconstructed':
                hausdorff_distance_pd(o, r, use_haversine=True, disable_checks=True),
            'Jaccard Original - Protected':
                jaccard_index_pd(o, p),
            'Jaccard Original - Reconstructed':
                jaccard_index_pd(o, r),
        }
    return res


def parallelized_distance_computation(
        test_orig: dict, reconstructed: dict, test_p: dict, fold: int = 0) -> List[dict]:
    """
    Compute our metrics in parallel.
    :param test_orig: Original Trajectories [d[id] = pd.DataFrame]
    :param reconstructed: Reconstructed Trajectories [d[id] = pd.DataFrame]
    :param test_p: Protected Trajectories [d[id] = pd.DataFrame]
    :param fold: Fold number
    :return: List of Results for each tuple of trajectories with same ID.
    """
    start = timer()
    parallel_input = [
        (test_orig[id],
         reconstructed[id],
         test_p[id],
         fold)
        for id in reconstructed
    ]
    if Config.parallelization_enabled():
        with mp.Pool(mp.cpu_count()) as pool:
            fold_results = [
                x for x in tqdm(pool.imap(compute_distances, parallel_input, chunksize=10),
                                desc='Computing distances',
                                total=len(parallel_input))]
    else:
        fold_results = [
            compute_distances(x) for x in tqdm(parallel_input, desc='Computing distances', total=len(parallel_input))]
    log.info(f"Completed distance computation in {round(timer() - start)}s.")
    return fold_results


def compute_decrease_percent(before: float or np.ndarray, after: float or np.ndarray) -> float or np.ndarray:
    """
    Compute the decrease in percentage.
    :param before: The value before the change
    :param after: The value after the change
    :return: Improvement [%]
    """
    return 100 * (before - after) / abs(before)


def compute_increase_percent(before: float or np.ndarray, after: float or np.ndarray) -> float or np.ndarray:
    """
    Compute the increase in percentage.
    :param before: The value before the change
    :param after: The value after the change
    :return: Improvement [%]
    """
    return 100 * (after - before) / abs(before)


def comp_results(df: pd.DataFrame) -> (float, float, float, float):
    """
    :param df: Pandas DataFrame containing results
    :return: (Euclidean Improvement, Hausdorff Improvement, Jaccard Protected, Jaccard Reconstructed)
    """
    ep = df['Euclidean Original - Protected']
    er = df['Euclidean Original - Reconstructed']
    hp = df['Hausdorff Original - Protected']
    hr = df['Hausdorff Original - Reconstructed']
    e_imp = (compute_decrease_percent(ep, er)).mean()
    h_imp = (compute_decrease_percent(hp, hr)).mean()
    jp = df['Jaccard Original - Protected'].mean()
    jr = df['Jaccard Original - Reconstructed'].mean()
    return e_imp, h_imp, jp, jr


def print_results_detailed(df: pd.DataFrame) -> None:
    """Receives a pandas DataFrame containing all evaluation results."""
    e_imp, h_imp, jp, jr = comp_results(df)
    # Print Results
    print(f"Average Euclidean Distance protected\t\t<-->\toriginal:\t{df['Euclidean Original - Protected'].mean()}")
    print(
        f"Average Euclidean Distance reconstructed\t<-->\toriginal:\t{df['Euclidean Original - Reconstructed'].mean()}")
    print(f"Improvement by {round(e_imp, 1)}%.")
    print(f"Average Hausdorff Distance protected\t\t<-->\toriginal:\t{df['Hausdorff Original - Protected'].mean()}")
    print(
        f"Average Hausdorff Distance reconstructed\t<-->\toriginal:\t{df['Hausdorff Original - Reconstructed'].mean()}")
    print(f"Improvement by {round(h_imp, 1)}%.")
    print(f"Average Jaccard Distance protected\t\t<-->\toriginal:\t{jp}")
    print(f"Average Jaccard Distance reconstructed\t\t<-->\toriginal:\t{jr}")
    j_imp = compute_increase_percent(jp, jr) if jp != 0.0 else np.inf
    print(f"Improvement by {round(j_imp, 1)}%.")


def print_all_results(output_dir: str, res_file: str = None) -> None:
    """Print the improvements for all metrics based on a directory containing the results.
    :param output_dir: The parent directory containing all the cases' output directories
    :param res_file: File to write the computed results to
    """
    filename = get_cases_file()
    cases = read_cases(filename)
    all_cases = cases['ID']

    for cid in all_cases:
        filename = output_dir + f"case{cid}/results.csv"
        try:
            df = pd.read_csv(filename)
            e_imp, h_imp, jp, jr = comp_results(df)
            if 'Jaccard Original - Protected' in df:
                j_imp = compute_increase_percent(jp, jr) if jp != 0.0 else np.inf
            else:
                j_imp, jp, jr = "N/A", "N/A", "N/A"

            cases.loc[cases['ID'] == str(cid), 'Euclid'] = f"{e_imp}%"
            cases.loc[cases['ID'] == str(cid), 'Hausdorff'] = f"{h_imp}%"
            cases.loc[cases['ID'] == str(cid), 'Jaccard Before'] = f"{jp}"
            cases.loc[cases['ID'] == str(cid), 'Jaccard After'] = f"{jr}"

            print(f"\033[31m Case {cid} Results: \033[0m\t"
                  f"Improvement by ({round(e_imp, 1)}%;"
                  f"\t{round(h_imp, 1)}%;"
                  f"\t{round(j_imp, 1)}%).")
        except FileNotFoundError:
            print(f"\033[31m Case {cid} not found.\033[0m")

    if filename is not None:
        cases.to_csv(res_file, index=False)


def store_metadata(odir: str, case: dict) -> None:
    """Store the metadata of an evaluation for honest reproducibility.
    :param odir: The case's output directory
    :param case: The dictionary containing all the case parameters
    """
    metafile = odir + 'metadata.txt'
    with open(metafile, 'w') as f:
        f.write(f"# Case {case['ID']}\n")
        for key in case:
            f.write(f"{key} = {case[key]}\n")
        f.write(f"Number of Splits = {n_kfold}\n")
        f.write(f"Seed for KFold/Split = {kfold_random_state}\n")
    log.info(f"Wrote Metadate to {metafile}.")


def determine_max_length(case: dict) -> int:
    """Determine the padding length that needs to be used by the model. Necessary if different datasets used.
    :param case: Dict containing all case information
    :return: Maximal length of any trajectory in the datasets
    """
    return max([Config.get_max_len(case['Dataset Train']), Config.get_max_len(case['Dataset Test'])])


def run_case(case: dict) -> bool:
    """
    Run the entire evaluation for a given case.
    :param case: Dict containing all case information
    :return: True on success, False if an error occurred.
    """
    from raopt.utils import helpers
    from raopt.ml.model import AttackModel
    import tensorflow as tf
    cid = case['ID']
    print(f"\033[33m Running Case {cid}.\033[0m")

    same_dataset = True if case['Dataset Train'] == case['Dataset Test'] else False

    # Create the output directory
    odir = Config.get_output_dir() + f"case{cid}/"
    Path(odir).mkdir(parents=True, exist_ok=True)

    ###########################################################################
    # Load Data
    ###########################################################################
    log.info("Loading Training/Testing Data")
    train_basename = get_basename(case['Protection Train'], case['Epsilon Train'], case['M Train'], 1)
    test_basename = get_basename(case['Protection Test'], case['Epsilon Test'], case['M Test'], 1)
    try:
        train_originals = helpers.load_trajectory_dict(dataset=case['Dataset Train'], basename='originals')
        train_protected = helpers.load_trajectory_dict(dataset=case['Dataset Train'], basename=train_basename)
        test_originals = copy.deepcopy(train_originals) if same_dataset else helpers.load_trajectory_dict(
            dataset=case['Dataset Test'], basename='originals')
        test_protected = copy.deepcopy(train_protected) if same_dataset and train_basename == test_basename \
            else helpers.load_trajectory_dict(dataset=case['Dataset Test'], basename=test_basename)
    except FileNotFoundError as e:
        log.error(f"Aborting case {cid} because file {e.filename} not found.")
        raise RuntimeError(f"Case {cid}: {e.filename} not found.")

    ###########################################################################
    # Determine Indices / K-Fold
    ###########################################################################
    splits: Iterable[(list, list)]  # Each tuple contains (train_index, test_index)
    if same_dataset:
        # Use k-fold if same dataset
        log.info(f"Using {n_kfold}-Fold Cross Validation.")
        # Use fixed random state for reproducibility
        keys = sorted(train_protected.keys())
        kf = KFold(n_splits=n_kfold, shuffle=True, random_state=kfold_random_state)
        splits = [([keys[i] for i in train_idx], [keys[i] for i in test_idx]) for train_idx, test_idx in kf.split(keys)]
    else:
        # If we do not use the same dataset, we can just train on one and test on the other.
        splits: List = [(list(train_protected.keys()), list(test_protected.keys()))]
        splits = splits * n_kfold
    ###########################################################################
    # Store Meta Data for Reproduction
    ###########################################################################
    store_metadata(odir, case)
    ###########################################################################
    # Model Parameters
    ###########################################################################
    max_length = determine_max_length(case)
    # We do not consider the test originals as the attacker does not have access to these
    all_trajectories = list(train_originals.values()) + list(train_protected.values()) + list(test_protected.values())
    lat0, lon0 = helpers.compute_reference_point(all_trajectories)
    log.info(f"Reference Point: ({lat0:.2f},{lon0:.2f})")
    scale_factor = helpers.compute_scaling_factor(all_trajectories, lat0, lon0)
    log.info(f"Scale Factor: ({scale_factor[0]:.2f},{scale_factor[1]:.2f})")
    ###########################################################################
    # Encoding of Training Set
    ###########################################################################
    from raopt.ml.encoder import get_encoded_trajectory_dict
    train_originals_encoded = get_encoded_trajectory_dict(
        dataset=case['Dataset Train'], basename="originals", trajectory_dict=train_originals)
    train_protected_encoded = get_encoded_trajectory_dict(
        dataset=case['Dataset Train'], basename=train_basename, trajectory_dict=train_protected)
    ###########################################################################
    # Training and Testing
    ###########################################################################
    fold, results = 0, []
    for train_idx, test_idx in splits:

        # Against memory leakage
        tf.keras.backend.clear_session()

        fold += 1
        log.info(f"Processing round {fold}/{len(splits)}")
        parameter_file = odir + f'parameters_fold_{fold}.hdf5'

        lstm = AttackModel(
            max_length=max_length,
            scale_factor=scale_factor,
            learning_rate=learning_rate,
            reference_point=(lat0, lon0),
            parameter_file=parameter_file
        )

        # lstm.model.summary( )

        if Config.continue_evaluation() and Path(parameter_file).exists():
            log.warning(f"Loading existing parameters from {parameter_file}.")
            lstm.model.load_weights(parameter_file)

        # Smaller M cases need more training
        # if case['M Train'] < 2000:
        #     epochs = EPOCHS * 2
        # else:
        epochs = EPOCHS

        # Check if training might have been completed already
        fold_completion_file = odir + f'fold_{fold}_complete.txt'
        if not Config.continue_evaluation() or (not Path(fold_completion_file).exists() or
                                                helpers.load(fold_completion_file) < epochs):
            ###########################################################################
            # Training
            ###########################################################################
            trainX = [train_protected_encoded[key]for key in train_idx]
            trainY = [train_originals_encoded[key] for key in train_idx]
            log.info("Start Training")
            history = lstm.train(trainX, trainY, epochs=epochs, batch_size=batch_size,
                                 use_val_loss=True, tensorboard=Config.use_tensorboard())
            n_epochs = len(history.history['loss'])
            log.info(f"Training complete after {n_epochs} epochs.")
            # Remember that this fold has been fully trained
            helpers.store(epochs, fold_completion_file)
            ###########################################################################
            # End Training
            ###########################################################################

        ###########################################################################
        # Testing
        ###########################################################################
        log.info(f"Prediction on {len(test_idx)} indices.")
        reconstructed = lstm.predict([test_protected[i] for i in test_idx if i in test_protected])
        reconstructed: Dict[str or int, pd.DataFrame] = {str(p['trajectory_id'][0]): p for p in reconstructed}
        ###########################################################################
        # Compute Results
        ###########################################################################
        fold_results = parallelized_distance_computation(test_orig=test_originals, reconstructed=reconstructed,
                                                         test_p=test_protected, fold=fold)
        # Store results of this fold
        result_file = odir + f'fold_results_{fold}.csv'
        df = pd.DataFrame(fold_results)
        df.to_csv(result_file, index=False)
        log.info(f"Wrote Fold {fold} results to: {result_file}")
        print_results_detailed(df)
        # All Fold's results to total results
        results.extend(fold_results)

    ###########################################################################
    # store final results
    ###########################################################################
    result_file = odir + f'results.csv'
    df = pd.DataFrame(results)
    df.to_csv(result_file, index=False)
    log.info(f"Wrote final results to: {result_file}")
    print_results_detailed(df)

    # Returns true on success only
    return True


def run_cases():
    """Execute all evaluation cases marked as to-do."""
    cases = get_cases()
    for case in cases:
        if not case['Todo']:
            log.info(f"Skipping case {case['ID']} because Todo = {case['Todo']}.")
            continue
        case_logfile_handler = logger.add_filehandler(Config.get_logdir() + f"case_{case['ID']}.log", log)
        try:
            success = run_case(case)
            if success:
                mark_case_complete(case['ID'])
        except Exception as e:
            err_file = Config.get_logdir() + f"case_{case['ID']}.err"
            log.error(f"Case {case['ID']}: Aborted due to the following error: {str(e)}")
            with open(err_file, 'a') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
            log.error(f"Full stack trace written to {err_file}.")
        finally:
            # Remove case specific log file
            log.removeHandler(case_logfile_handler)
    log.info("All test cases completed. Terminating.")


if __name__ == '__main__':
    if CASE is None:
        run_cases()
    else:
        cases = get_cases()
        for case in cases:
            if case['ID'] == CASE:
                run_case(case)
