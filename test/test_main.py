# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import logging
import pickle
from contextlib import redirect_stdout
from io import StringIO
from unittest import TestCase
from unittest.mock import patch, Mock

import pandas as pd

import raopt.eval.main as main
from raopt.utils.config import Config

logging.basicConfig(level=logging.ERROR)

with open(Config.get_test_dir() + 'resources/sample_trajectories.pickle', 'rb') as f:
    trajs = pickle.load(f)


class Test(TestCase):

    def test__read_cases(self):
        cases = main.read_cases(Config.get_test_dir() + 'resources/cases.csv')
        self.assertTrue(
            type(cases) == pd.DataFrame
        )
        self.assertFalse(cases['Done'][7])
        self.assertTrue(cases['Done'][5])

    def test_read_cases(self):
        m = Mock()
        with patch.object(main, 'read_cases') as p:
            p.return_value = m
            main.get_cases()
            p.assert_called_once()
            m.to_dict.assert_called_once()

    def test_mark_case_complete(self):
        filename = Config.get_test_dir() + 'resources/cases.csv'
        cases = main.read_cases(filename)
        self.assertFalse(cases['Done'][8], msg='Case 9 is marked as done but should not!')
        main.mark_case_complete(9, filename=filename)
        cases = main.read_cases(filename)
        self.assertTrue(cases['Done'][8], msg='Case 9 was not marked!')
        # Revert marking
        main.mark_case_complete(9, mark_as=False, filename=filename)

    def test_compute_distances(self):
        a, b, c = trajs[1], trajs[2], trajs[3]
        fold = 77
        self.assertEqual(
            {
                'Fold': fold,
                'Euclidean Original - Protected':
                    0.0,
                'Euclidean Original - Reconstructed':
                    0.0,
                'Hausdorff Original - Protected':
                    0.0,
                'Hausdorff Original - Reconstructed':
                    0.0,
                'Jaccard Original - Protected': 1.0,
                'Jaccard Original - Reconstructed': 1.0
            },
            main.compute_distances((a, a, a, fold))
        )
        self.assertEqual(
            {
                'Fold': fold,
                'Euclidean Original - Protected':
                    6262.082387002696,
                'Euclidean Original - Reconstructed':
                    0.0,
                'Hausdorff Original - Protected':
                    22545.64698114857,
                'Hausdorff Original - Reconstructed':
                    0.0,
                'Jaccard Original - Protected': 1.821814887138933e-05,
                'Jaccard Original - Reconstructed': 1.0
            },
            main.compute_distances((a, a, b, fold))
        )

    def test_parallelized_distance_computation(self):
        orig = {1: trajs[1], 2: trajs[3]}
        protected = {1: trajs[2], 2: trajs[3]}
        rec = {1: trajs[1], 2: trajs[3]}
        fold = 77
        res = main.parallelized_distance_computation(orig, rec, protected, fold)
        self.assertEqual(
            [
                {
                    'Fold': fold,
                    'Euclidean Original - Protected':
                        6262.082387002696,
                    'Euclidean Original - Reconstructed':
                        0.0,
                    'Hausdorff Original - Protected':
                        22545.64698114857,
                    'Hausdorff Original - Reconstructed':
                        0.0,
                    'Jaccard Original - Protected': 1.821814887138933e-05,
                    'Jaccard Original - Reconstructed': 1.0
                },
                {
                    'Fold': fold,
                    'Euclidean Original - Protected':
                        0.0,
                    'Euclidean Original - Reconstructed':
                        0.0,
                    'Hausdorff Original - Protected':
                        0.0,
                    'Hausdorff Original - Reconstructed':
                        0.0,
                    'Jaccard Original - Protected': 1.0,
                    'Jaccard Original - Reconstructed': 1.0
                }
            ],
            list(res)
        )

    def test_print_results(self):
        df = pd.read_csv(Config.get_test_dir() + 'resources/results_sample.csv')
        with redirect_stdout(StringIO()) as stdout:
            main.print_results_detailed(df)
        text = stdout.getvalue()
        self.assertIn(
            'Improvement',
            text
        )
