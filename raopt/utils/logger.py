#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
"""
Custom logger supporting colors.

Adapted from:
https://stackoverflow.com/questions/384076/how-can-i-color-python-logging -output
"""
import copy
import logging
import os
import sys

from raopt.utils import config

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# The background is set with 40 plus the number of the color, and the
# foreground with 30

# These are the sequences need to get colored output
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

COLORS = {
    'WARNING': YELLOW,
    'INFO': BLUE,
    'DEBUG': WHITE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}

# FORMAT = "[%(asctime)s][%(levelname)-18s][$BOLD%(name)-22s$RESET]  " \
#          "%(message)s ($BOLD%(filename)s$RESET:%(lineno)d) "
FORMAT = "[%(asctime)s][%(levelname)-18s] " \
         "%(message)s ($BOLD%(filename)s$RESET:%(lineno)d) "


class ColoredFormatter(logging.Formatter):
    """Formatter for colored console output."""

    def __init__(self, use_color=True):
        if use_color:
            fmt = FORMAT.replace("$RESET", RESET_SEQ).replace("$BOLD",
                                                              BOLD_SEQ)
        else:  # pragma no cover
            fmt = FORMAT.replace("$RESET", "").replace("$BOLD", "")
        logging.Formatter.__init__(self, fmt)
        self.use_color = use_color

    def format(self, record):  # pragma no cover
        """Format the given message."""
        record = copy.copy(record)
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            color: str = COLOR_SEQ % (30 + COLORS[levelname])
            record.msg = color + record.msg + RESET_SEQ
            record.levelname = color + levelname + RESET_SEQ
        return logging.Formatter.format(self, record)


def add_colored_formatter(
        logger: logging.Logger = logging.getLogger()) -> None:
    """Add ColoredFormatter to a given logger or to the root logger."""
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColoredFormatter())
    logger.addHandler(console)


def add_filehandler(file: str,
                    logger: logging.Logger = logging.getLogger()
                    ) -> logging.FileHandler:
    """Add filehandler for given file to a given logger or to the root
    logger."""
    filehandler = logging.FileHandler(file)
    filehandler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)-7s]  "  # [%(name)-22s]
        "%(message)s (%(filename)s:%(lineno)d) "))
    logger.addHandler(filehandler)
    return filehandler


def configure_root_loger(logging_level: int,
                         file: str or None = None) -> logging.Logger:
    """
    Add both the colored formatter and the filehandler to the root logger.
    """
    root = logging.getLogger()
    for h in root.handlers:
        root.removeHandler(h)
    root.setLevel(logging_level)
    add_colored_formatter(logger=root)
    if file is not None:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        add_filehandler(file, logger=root)
    error_handler = add_filehandler(
        config.Config.get_logdir() + 'error.log',
        logger=root)
    error_handler.setLevel(logging.ERROR)
    return root
