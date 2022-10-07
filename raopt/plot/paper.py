#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
"""
Plots for ACSAC Paper.
"""

import logging
from pathlib import Path
from typing import List, Iterable, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from raopt.eval.main import comp_results
from raopt.plot import plot
from raopt.plot.plot import cm
from raopt.utils.config import Config

log = logging.getLogger()
COLUMN_WIDTH = plot.IEEE_WIDTH
DOUBLE_COLUMN_WIDTH = 506.295 * 0.03514
result_file = Config.get_output_dir() + "case{}/results.csv"  # Needs format with case ID
plot_dir = "plots/"
Path(plot_dir).mkdir(parents=True, exist_ok=True)
ALL_COLUMNS = ['ID', 'Dataset Train', 'Dataset Test', 'Protection Train', 'Protection Test',
               'Epsilon Train', 'Epsilon Test',
               ]
font_size = 7
settings = {
    'font.size': font_size,
    'legend.fontsize': font_size - 2,
    'axes.titlesize': font_size - 2,
    'axes.labelsize': font_size - 2,
    'ytick.labelsize': font_size - 2,
    'xtick.labelsize': font_size - 2,
    'lines.markersize': 3,
    'figure.autolayout': True
}
plt.rcParams.update(settings)


def modify_strings(cases: List[dict]) -> List[dict]:
    """
    Modify constant so that they look better when printed onto plots.
    :param cases: List with cases
    :return: Modified list of cases
    """
    replacements = {
        'TDRIVE': 'T-Drive',
        'CNOISE': 'CNoise',
        'GEOLIFE': 'GeoLife'
    }
    for case in cases:
        for col in case:
            for old in replacements:
                if type(case[col]) is str and old in case[col]:
                    case[col] = case[col].replace(old, replacements[old])
    return cases


def example_trajectories(originals: Dict[str, pd.DataFrame],
                         protected: Dict[str, pd.DataFrame],
                         reconstructed: Dict[str, pd.DataFrame],
                         tids: List[str],
                         n_rows: int = 1,
                         protection_mechanism='SDD 0.1',
                         filename: bool = None):
    """

    :param originals: dict containing original trajectories: d[trajectory_id] = pd.DataFrame
    :param protected: dict containing protected trajectories: d[trajectory_id] = pd.DataFrame
    :param reconstructed: dict containing reconstructed trajectories: d[trajectory_id] = pd.DataFrame
    :param tids: List of trajectory ids
    :param n_rows: number of rows
    :param protection_mechanism: The protection mechanisms used for these examples
    :param filename: None or location of output PNG
    :return: None
    """
    n_cols = int(len(tids) / n_rows)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(cm(DOUBLE_COLUMN_WIDTH), cm(3)))
    if n_rows == 1:
        ax = [ax]
    labels = ['Original', protection_mechanism, 'Reconst.']
    lines = list()
    for row in range(n_rows):
        ax[row][0].set_ylabel("Latitude")
        for col in range(n_cols):
            ax[row][col]: plt.Axes
            tid = tids[row * n_cols + col]
            ts = [originals[tid], protected[tid], reconstructed[tid]]

            for i, t in enumerate(ts):
                line, = ax[row][col].plot(t.longitude, t.latitude, 'o-', linewidth=0.5)
                lines.append(line)

            ax[row][col].set_xlabel("Longitude")
            ax[row][col].set_title(f"TID: {tid}", x=0.83, y=0.75)

    plot.Legend(lines[:len(labels)], labels, axis=fig, location='top', ncols=3).make()
    plt.subplots_adjust(wspace=-1.3, hspace=None)
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Stored to {filename}.")


def plot_line(axis: plt.Axes, x: Iterable[float], data: Iterable[float], label: str = None, fmt: str = None):
    """
    Plot a line with error-bars at each point.
    :param axis: The Axes object to plot onto
    :param x: The x coordinates len(x) == len(data)
    :param data: List of y values to compute the mean and confidence interval from
    :param label: Line label for legend
    :param fmt: Line format/style
    :return:
    """
    y, error = [], []
    for d in data:
        tmp_y, tmp_e = plot.mean_confidence_interval(d)
        y.append(tmp_y)
        error.append(tmp_e)
    return axis.errorbar(
        x,
        y,
        fmt=fmt,
        yerr=error,
        label=label,
    )


def plot_lines(axis: plt.Axes, x: Iterable[float], data: Iterable[Iterable[float]], labels: List[str],
               title: str = None):
    """
    Plot multiple lines with y-errorbars.
    :param axis: The axis object to print ontoz
    :param x: The x coordinates len(x) == len(data[i]) for all i in range(len(data))
    :param data: List of lists, one list for each line
                 I.e., each element is a list of values. data = [[], [], []]
                 len(x) == len(data[i]) for all i in range(len(data))
    :param labels: Labels for each line. len(labels) == len(data)
    :param title: Title of the plot
    :return:
    """
    fmts = ['-x', '-o', '-D', '-s']
    pad = 2
    lines = []
    for i, y in enumerate(data):
        lines.append(plot_line(axis, x, y, label=labels[i], fmt=fmts[i]))
    axis.set_title(title, pad=pad)
    axis.set_xticks(x)
    return lines


def return_results(idx: List[str]) -> (List[Iterable[float]], List[Iterable[float]]):
    """
    Return the results for the given cases
    :param idx: List of case IDs
    :return: (List[Euclidean Improvements], List[Hausdorff Imp.])
    """
    cases = []
    for cid in idx:
        try:
            cases.append(pd.read_csv(result_file.format(cid)))
        except FileNotFoundError as e:
            log.warning(f"[SKIPPED] Case(s) missing: {e.filename}")

    data_euclid = []
    data_hausdorff = []
    for case in cases:
        ep = case['Euclidean Original - Protected']
        er = case['Euclidean Original - Reconstructed']
        hp = case['Hausdorff Original - Protected']
        hr = case['Hausdorff Original - Reconstructed']
        improvement_euclid = 100. * (ep - er) / abs(ep)
        improvement_hausdorff = 100. * (hp - hr) / abs(hp)
        data_euclid.append(improvement_euclid)
        data_hausdorff.append(improvement_hausdorff)
    return data_euclid, data_hausdorff


def adversary_1_figure(save_to_file: bool = False) -> None:
    """
    Figure 4 in Paper. Shows basic cases.
    :param save_to_file: Store into output directory
    :return:
    """
    tdrive_sdd_euclid, tdrive_sdd_hausdorff = return_results(range(6, 9 + 1))
    tdrive_cnoise_euclid, tdrive_cnoise_hausdorff = return_results(range(1, 5 + 1))
    geolife_sdd_euclid, geolife_sdd_hausdorff = return_results(range(15, 18 + 1))
    geolife_cnoise_euclid, geolife_cnoise_hausdorff = return_results(range(10, 14 + 1))

    labels = ("TDRIVE Euclid", "TDRIVE Hausdorff", "GEOLIFE Euclid", "GEOLIFE Hausdorff")
    fig, ax = plt.subplots(1, 2, figsize=(cm(COLUMN_WIDTH), cm(4)), sharey=True)

    # One plot for Cnoise and one for SDD=
    # Cnoise
    x = [0.01, 0.1, 1, 10, 100]
    data = [tdrive_cnoise_euclid, tdrive_cnoise_hausdorff, geolife_cnoise_euclid, geolife_cnoise_hausdorff]
    plot_lines(ax[0], x, data, labels, 'Cnoise')

    # SDD
    x = [0.01, 0.1, 1, 10]
    data = [tdrive_sdd_euclid, tdrive_sdd_hausdorff, geolife_sdd_euclid, geolife_sdd_hausdorff]
    lines = plot_lines(ax[1], x, data, labels, 'SDD')

    ax[0].set_ylabel('Distance Reduction [%]')
    for axis in ax:
        axis.set_xlabel('\u03B5')
        axis.set_ylim(0, 100)
        axis.set_xscale('log')

    plot.Legend(lines, labels, axis=fig, location='top', ncols=4).make()
    fig.tight_layout()
    if save_to_file:
        filename = plot_dir + 'distances_basic.png'
        plt.savefig(filename, bbox_inches='tight')
        print(f"Stored to {filename}.")


def print_all_results() -> None:
    """
    Print a table with all results for cases 1 - 36 for the appendix.
    :return: None
    """
    print_partial_table([str(i) for i in range(1, 37)])


def print_partial_table(case_ids: List[int], columns: List[str] = ALL_COLUMNS, no_results=False) -> None:
    """
    Print a table with the results for the given cases.
    :param case_ids: IDs of the cases to print
    :param columns: Columns of the table
    :param no_results: Only print the case properties
    :return: None
    """
    # Create complete table
    from raopt.eval.main import get_cases
    cases = [case for case in get_cases() if case['ID'] in case_ids]
    cases = modify_strings(cases)

    for case in cases:
        cid = case['ID']
        filename = result_file.format(cid)
        try:
            df = pd.read_csv(filename)
            line = [f"{case[column]}" for column in columns]
            if not no_results:
                e_imp, h_imp, jp, jr = comp_results(df)
                line.extend([
                    f"\SI{{{e_imp:.1f}}}{{\%}}",
                    f"\SI{{{h_imp:.1f}}}{{\%}}",
                    f"\\num{{{jp:1.2e}}}",
                    f"\\num{{{jr:1.2e}}}"
                ])
            line[-1] += '\\\\'
            print(
                ' & '.join(line)
            )
        except FileNotFoundError:
            print(f"\033[31m Case {cid} not found.\033[0m")


def plot_bar(axis: plt.Axes, x: Iterable, data: List[List[float]], bar_width=0.5, labels=None):
    """
    Create a bar plot.
    :param axis: The Axes object to produce the bars on
    :param x: The x values for the bars (left edge of the bar!) len(x) == len(data)
    :param data: The y values for a bar. Each bar is represented by a List[float] so that mean and conf-intervals
                 can be computed.
    :param bar_width: Width of each bar
    :param labels: Bar labels for the figure's legend len(labels) == len(data)
    :return:
    """
    y, error = [], []
    for d in data:
        tmp_y, tmp_e = plot.mean_confidence_interval(d)
        y.append(tmp_y)
        error.append(tmp_e)

    bars = axis.bar(
        x=x,
        height=y,
        width=bar_width,
        yerr=error,
        label=labels,
        align='edge'
    )
    bar_labels = axis.bar_label(bars, fmt='%.1f%%', label_type='center', fontsize=font_size - 2)
    for i in range(len(y)):
        if y[i] < 0:
            bar_labels[i].xy = (bar_labels[i].xy[0], 10)
    return bars


def transfer_figure(title: str,
                    euclid1: Iterable[Iterable[float]], hausdorff1: Iterable[Iterable[float]],
                    euclid2: Iterable[Iterable[float]], hausdorff2: Iterable[Iterable[float]],
                    x_ticks: Iterable[float], x_labels: Iterable, filename: str = None) -> plt.Figure:
    """
    For Figures 5 & 7
    The data Iterables have one list for each location:
    euclidX/hausdorffX = [[x1_1,...,x1_m], [x2_1,...,x2_m]], ..., [xn_1,...,xn_m]]]
    :param title: Title of the plot
    :param euclid1: One list for each x-value (so that mean and confidence intervals can be computed)
    :param hausdorff1: One list for each x-value (so that mean and confidence intervals can be computed)
    :param euclid2: One list for each x-value (so that mean and confidence intervals can be computed)
    :param hausdorff2: One list for each x-value (so that mean and confidence intervals can be computed)
    :param x_ticks: Location of x_ticks len(x_ticks) == len(euclidX) == len(hausdorffX)
    :param x_labels: Test for x_ticks len(x_labels) == len(x_ticks)
    :param filename: If provided the Figure is written to this file
    :return: Produced Figure
    """
    fig, ax = plt.subplots(1, 2, figsize=(cm(COLUMN_WIDTH), cm(3)), sharey=True)
    # One plot for Cnoise and one for SDD=
    pad = 2
    labels = ("Euclidean Distance", "Hausdorff Distance")
    x = np.array([0.6, 1.6])
    bar_width = 0.4

    ax[0].set_title(f'T-Drive to GeoLife [{title}]', pad=pad)
    ax[0].set_xticks(x_ticks, x_labels)
    lines = list()
    lines.append(plot_bar(ax[0], x, euclid1, bar_width))
    lines.append(plot_bar(ax[0], x + bar_width, hausdorff1, bar_width))

    ax[1].set_title(f'GeoLife to T-Drive [{title}]', pad=pad)
    ax[1].set_xticks(x_ticks, x_labels)
    lines = list()
    lines.append(plot_bar(ax[1], x, euclid2, bar_width))
    lines.append(plot_bar(ax[1], x + bar_width, hausdorff2, bar_width))

    for axis in ax:
        axis.set_xticks(x_ticks, x_labels)
        axis.set_xlabel('\u03B5', labelpad=-10)
        # axis.set_ylabel('Distance Reduction [%]')
        axis.set_ylim(0, 100)

    ax[0].set_ylabel('Distance Reduction [%]')
    # fig.legend(lines, labels, ncol=2, )#bbox_to_anchor=(0.5, 0.8),  loc='center',)
    plot.Legend(lines, labels, axis=fig, location='top', ncols=2).make()
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Stored to {filename}.")
    return fig


def adversary_2_figure(save_to_file: bool = False) -> None:
    """
    Figures 5 & 7. Transfer from one dataset to another.
    :param save_to_file: Store into output directory
    :return: None
    """
    tdrive_geolife_sdd_euclid, tdrive_geolife_sdd_hausdorff = return_results(range(21, 22 + 1))
    tdrive_geolife_cnoise_euclid, tdrive_geolife_cnoise_hausdorff = return_results(range(19, 20 + 1))
    geolife_tdrive_sdd_euclid, geolife_tdrive_sdd_hausdorff = return_results(range(25, 26 + 1))
    geolife_tdrive_cnoise_euclid, geolife_tdrive_cnoise_hausdorff = return_results(range(23, 24 + 1))

    # SDD
    x_ticks = [1, 2]
    x_labels = [0.1, 1]
    filename = plot_dir + 'distance_transfer_sdd.png' if save_to_file else None
    transfer_figure(
        'SDD', tdrive_geolife_sdd_euclid, tdrive_geolife_sdd_hausdorff,
        geolife_tdrive_sdd_euclid, geolife_tdrive_sdd_hausdorff, x_ticks, x_labels,
        filename=filename
    )
    # Cnoise
    x_ticks = [1, 2]
    x_labels = [1, 10]
    filename = plot_dir + 'distance_transfer_cnoise.png' if save_to_file else None
    transfer_figure(
        'CNOISE', tdrive_geolife_cnoise_euclid, tdrive_geolife_cnoise_hausdorff,
        geolife_tdrive_cnoise_euclid, geolife_tdrive_cnoise_hausdorff, x_ticks, x_labels,
        filename=filename
    )


def worst_case_plot(save_to_file: bool = False) -> plt.Figure:
    """
    Figure 6 in paper: Show the 4 worst-case measurements
    :param save_to_file: Store into output directory?
    :return: Figure
    """
    euclid, hausdorff = return_results(range(33, 36 + 1))

    x_ticks = np.arange(1, 5, 1)
    x_labels = np.arange(33, 37, 1)
    bar_width = 0.4
    x = np.arange(1, 5, 1) - bar_width

    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(cm(COLUMN_WIDTH), cm(3)))
    # One plot for Cnoise and one for SDD
    pad = 2
    labels = ("Euclidean Distance", "Hausdorff Distance")

    bars = list()
    bars.append(plot_bar(ax, x, euclid, bar_width))
    bars.append(plot_bar(ax, x + bar_width, hausdorff, bar_width))

    ax.set_title('Adversary Case 3', pad=pad)
    ax.set_xticks(x_ticks, x_labels)
    ax.set_xlabel('Measurement ID')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Distance Reduction [%]')
    # ax.legend(bars, labels, loc='upper left', ncol=2)
    plot.Legend(bars, labels, axis=fig, location='top', ncols=2).make()
    fig.tight_layout()
    if save_to_file:
        filename = plot_dir + 'distances_worst_case.png'
        plt.savefig(filename, bbox_inches='tight')
        print(f"Stored to {filename}.")


def figure_3(save_to_file: bool = False, train: bool = False) -> None:
    """Create Figure 3 from Paper

    :param save_to_file: Store into output directory?
    :param train: Train the model from scratch instead of using existing parameters
    :return: None
    """
    import random

    from raopt.utils import helpers
    from raopt.ml import model

    filename = plot_dir + 'example-trajs.png' if save_to_file else None

    originals = helpers.read_trajectories_from_csv(
        "processed_csv/tdrive/originals.csv")
    protected = helpers.read_trajectories_from_csv(
        "processed_csv/tdrive/sdd_M16500_e0.1_1.csv")
    tid_range = list(protected.keys())
    tid_range_file = Config.get_parameter_path() + 'tid_range_fig3.pickle'

    parameter_file = Config.get_parameter_path() + 'figure3.hdf5'
    if not train and not Path(parameter_file).exists():
        parameter_file = "output/case7/parameters_fold_1.hdf5"
    elif not train:
        tid_range = helpers.load(tid_range_file)

    all_trajs = list(protected.values())
    lat0, lon0 = helpers.compute_reference_point(all_trajs)
    scale_factor = helpers.compute_scaling_factor(all_trajs, lat0, lon0)
    log.info(f"Reference Point: ({lat0}, {lon0})")
    log.info(f"Scaling factor: {scale_factor}")

    raopt = model.AttackModel(
        reference_point=(lat0, lon0),
        scale_factor=scale_factor,
        max_length=100,
    )

    if train:
        from raopt.ml import encoder
        keys = list(protected.keys())
        random.shuffle(keys)
        n_test = int(0.2 * len(keys))
        tid_range = keys[:n_test]
        helpers.store(tid_range, tid_range_file)
        train_idx = keys[n_test:]
        protected_encoded = encoder.encode_trajectory_dict(protected)
        originals_encoded = encoder.encode_trajectory_dict(originals)
        trainX = [protected_encoded[key] for key in train_idx]
        trainY = [originals_encoded[key] for key in train_idx]
        log.info("Start Training")
        history = raopt.train(trainX, trainY, use_val_loss=True)
        log.info(f"Training complete after {len(history.history['loss'])} epochs.")
    else:
        log.info(f"Loading parameters from: {parameter_file}")
        raopt.model.load_weights(parameter_file)

    tids = random.sample(tid_range, 4)

    reconstructed = raopt.predict([protected[i] for i in tids])
    reconstructed = helpers.dictify_trajectories(reconstructed)

    example_trajectories(
        originals=originals,
        protected=protected,
        reconstructed=reconstructed,
        tids=tids,
        protection_mechanism='SDD 0.1',
        filename=filename
    )


if __name__ == '__main__':
    # Table 5
    print("#" * 80)
    print("Table 5: ")
    print("#" * 80)
    print_all_results()
    print("#" * 80)

    # Table 2
    print("#" * 80)
    print("Table 2")
    print("#" * 80)
    print_partial_table(['27', '28', '29', '30'], ['ID', 'Protection Test', 'Epsilon Train', 'Epsilon Test'])
    print("#" * 80)

    # Table 3
    print("#" * 80)
    print("Table 3")
    print("#" * 80)
    print_partial_table(['31', '32'], ['ID', 'Protection Train', 'Protection Test', 'Epsilon Test'])
    print("#" * 80)
    #
    # Table 4
    print("#" * 80)
    print("Table 4")
    print("#" * 80)
    print_partial_table(['33', '34', '35', '36'],
                        ['ID', 'Dataset Train', 'Dataset Test', 'Protection Train', 'Protection Test',
                         'Epsilon Train', 'Epsilon Test'], no_results=True)
    print("#" * 80)
    # Figure 3
    # Warning! Requires Model execution
    # print("#" * 80)
    # print("Figure 3")
    # print("#" * 80)
    # figure_3(True, False)
    # print("#" * 80)
    #
    # Figure 4
    print("#" * 80)
    print("Figure 4")
    print("#" * 80)
    adversary_1_figure(True)
    print("#" * 80)
    #
    # Figure 5 & 7
    print("#" * 80)
    print("Figure 5 & Figure 7")
    print("#" * 80)
    adversary_2_figure(True)
    print("#" * 80)
    #
    # Figure 6
    print("#" * 80)
    print("Figure 6")
    print("#" * 80)
    worst_case_plot(True)
    print("#" * 80)
