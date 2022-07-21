#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import logging
from typing import List

import pandas as pd
from matplotlib import pyplot as plt, axes
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from raopt.utils.helpers import find_bbox

log = logging.getLogger()
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def set_bbox(ax: axes.Axes, bbox: (float, float, float, float)) -> None:
    """
    Set the axis limits for the given axis.
    :param ax: The axis to modify
    :param bbox: [Optional] If not provided, read from config file.
    :return: None
    """
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])


def visualize_trajectories_tight(ts: List[pd.DataFrame], labels: List[str] = None) -> None:
    """
    Display the given trajectories within a tight bounding box.
    :param labels: Labels of the curves
    :rtype: None
    :param ts: List of min. one trajectory
    :return: None
    """
    log.debug(f"Trajectory Length: {len(ts[0])}")
    fig, ax = plt.subplots(figsize=(8, 7))
    lines = []
    for i, t in enumerate(ts):
        line, = ax.plot(t.longitude, t.latitude, 'o-', linewidth=0.5)
        lines.append(line)
    if labels is not None:
        ax.legend(lines, labels)
    ax.set_title(f"Trajectory")
    # bbox = pp.find_bbox(ts, quantile=1) + (0.2, 0.2, 0.2, 0.2)
    # set_bbox(ax, bbox)
    plt.show()


def scatterplot(df: pd.DataFrame, x_label='longitude', y_label='latitude') -> None:
    """
    Plot a scatterplot of the locations contained in the given DataFrame.
    """
    fig: Figure
    ax: axes.Axes
    fig, ax = plt.subplots()
    ax.set_title("Heatmap")

    # Set bounding box
    bbox = find_bbox(df, quantile=1) + (0.2, 0.2, 0.2, 0.2)
    set_bbox(ax, bbox)

    # Create points
    ax.scatter(df[x_label], df[y_label], zorder=1, alpha=0.2, s=0.01)

    # Display map as background
    # img = plt.imread(config.Config.get_temp_dir() + "map.png")
    # plt.imshow(img, zorder=0, extent=bbox, aspect='equal')

    plt.show()
    # plt.savefig(config.Config.get_temp_dir() + 'scatter.png', dpi=400)


def heatmap(df: pd.DataFrame, x_label='longitude', y_label='latitude') -> None:
    """
    Plot a heatmap of the entire dataset.
    :return: None
    """
    fig: Figure
    ax: axes.Axes
    fig, ax = plt.subplots()
    ax.set_title("Heatmap")

    # Set bounding box
    bbox = find_bbox(df, quantile=1) + (0.2, 0.2, 0.2, 0.2)
    set_bbox(ax, bbox)

    cax = ax.hexbin(
        x=df[x_label],
        y=df[y_label],
        extent=bbox,
        gridsize=1000,
        cmap='hot',
        bins='log',
        zorder=1,
    )

    ax.set_facecolor('black')

    fig.colorbar(cax, )

    plt.show()
    # plt.savefig(config.Config.get_temp_dir() + 'heatmap.png', dpi=900)


def get_size_overview(data: List[pd.DataFrame], upper_limit: int = 200, title: str = None) -> Axes:
    """
    Plot a histogram of the trajectory lengths.
    :param title: Title of the plot
    :param data: A list of trajectories as pandas DataFrames
    :param upper_limit: The upper length to show
    :return:
    """
    fig, ax = plt.subplots()
    sizes = [len(d) for d in data]
    ax.hist(sizes, bins=max(sizes))
    ax.set_title(title)
    ax.set_xlim(0, upper_limit)
    return ax
