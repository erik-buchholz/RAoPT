#!/usr/bin/env python3
#
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
"""
This file contains general plot functionality.
"""

from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib import patches


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Default Settings
# -----------------------------------------------------------------------------
def cm(value: float or int) -> float:
    """Calculate the number that has to be given as size to obtain cm."""
    return value / 2.54


IEEE_WIDTH = 241.14749 * 0.03514  # pt in cm
font_size = 9
ticks_fontsize = font_size - 1
legend_font_size = font_size - 1
default_settings = {
    'font.size': font_size,
    'legend.fontsize': legend_font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'ytick.labelsize': ticks_fontsize,
    'xtick.labelsize': ticks_fontsize,
    'hatch.linewidth': 0.8,
    'xtick.minor.pad': 1,
    'axes.labelpad': 3,
    'legend.framealpha': 1,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'legend.handletextpad': 0.2,
    'legend.columnspacing': 0.8,
    'figure.dpi': 1000,
    # 'figure.autolayout': True,
    'legend.facecolor': 'white',
    'lines.linewidth': 1.5,
    'errorbar.capsize': 3,  # Länge der Hüte
    'lines.markeredgewidth': 0.7,  # Dicke des horizontalen Strichs/Error Caps
    'lines.markersize': 3,
    # 'text.usetex' : True
}
plt.rcParams.update(default_settings)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


class Legend(object):
    """Represents a legend object

    empty_positions: Empty spaces in legend
    order: Reorder legend according to list
    axis: The axis to add the legend to
    markers: List of plot objects
    legend_labels: The labels to use
    location: 'top' or default matplotlib values
    """
    STACKS: str = 'STACKS'
    BARS: str = 'BARS'
    TOP: str = 'top'
    target: str = STACKS
    location: str = None
    markers: list = []
    ncols: int = None
    labels: List[str] = []
    axis = plt
    order: Union[Tuple[int], List[int], None] = None
    empty_positions: Union[List[int], None] = None
    custom_labels: List[tuple] = None

    def __init__(self, handles, labels: List[str], axis=plt, location: str = None,
                 custom_labels: List[tuple] = None,
                 empty_positions: Union[List[int], None] = None,
                 order: Union[Tuple[int], List[int], None] = None,
                 ncols: int = None):
        self.markers = handles
        self.labels = labels
        self.axis = axis
        self.location = location
        self.custom_labels = custom_labels
        self.empty_positions = empty_positions
        self.order = order
        self.ncols = ncols

    def make(self):
        """Add legend to axis"""
        if self.empty_positions is not None:
            r = patches.Rectangle((0, 0), 1, 1, fill=False,
                                  edgecolor='none',
                                  visible=False)
            for pos in self.empty_positions:
                self.labels.insert(pos, "")
                self.markers.insert(pos, r)
        if self.custom_labels is not None:
            if len(self.labels) == 0:
                # Only custom labels wanted
                self.markers = []
            for m, t in self.custom_labels:
                self.markers.append(m)
                self.labels.append(t)
        if self.order is not None:
            self.markers = [self.markers[i] for i in self.order]
            self.labels = [self.labels[i] for i in self.order]
        if self.ncols is not None:
            columns = self.ncols
        elif len(self.labels) <= 5:
            columns = len(self.labels)
        else:
            columns = np.ceil(len(self.labels) / 2)
        if self.location == "top":
            legend = self.axis.legend(self.markers, self.labels,
                                      loc='center', ncol=columns,
                                      bbox_to_anchor=(0.5, 1))
        elif self.location == 'above':
            legend = self.axis.legend(self.markers, self.labels,
                                      loc='lower center', ncol=columns,
                                      bbox_to_anchor=(0.5, 1))
        else:
            # Best location
            legend = self.axis.legend(self.markers, self.labels,
                                      loc=self.location, ncol=columns)
        legend.get_frame().set_linewidth(0.4)
        return legend


def mean_confidence_interval(data: list, confidence: float = 0.99) -> \
        Tuple[float, float]:
    """Compute the mean and the corresponding confidence interval of the
    given data.

    :param confidence: Confidence interval to use, default: 99%
    :param data: List of number to compute mean and interval for
    """
    a = 1.0 * np.array(data)
    n = len(a)
    if n == 1:
        return a[0], 0
    m, se = np.mean(a), scipy.stats.sem(a)
    h: float = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return float(m), h
