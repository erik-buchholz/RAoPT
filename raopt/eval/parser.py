#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
"""
Parse arguments for evaluation scripts.
"""
import argparse

from raopt.utils import config


def parse_eval() -> argparse.ArgumentParser:
    """Return argument parser for the evaluation scripts."""
    parser = argparse.ArgumentParser(description='RAoPT Evaluation.')
    parser.add_argument('dataset', metavar='DATASET', type=str.upper,
                        help='The dataset to use', choices=config.DATASETS)
    parser.add_argument('mechanism', metavar='MECHANISM', type=str.upper,
                        choices=config.MECHANISMS,
                        help='Mechanism to use for protection')
    parser.add_argument('epsilon', metavar='EPSILON', type=float,
                        help='Value for epsilon used by mechanism')
    parser.add_argument('version', metavar='VERSION', type=int,
                        help='Version number of output file')
    parser.add_argument('-m', '--sensitivity', metavar='M', dest='sensitivity', type=float,
                        help='Sensitivity M to use for protection mechanism', default=0)
    return parser
