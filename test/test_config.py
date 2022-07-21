# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import logging
from unittest import TestCase
from unittest.mock import patch

from raopt.utils import config

logging.basicConfig(level=logging.ERROR)


class TestConfig(TestCase):

    def setUp(self) -> None:
        config.Config._config = None

    def test_load_config(self):
        with patch("raopt.utils.config.Config._load_config") as m:
            config.Config.load_config()
            self.assertEqual(1, m.call_count)
        config.Config._config = "Value"
        with patch("raopt.utils.config.Config._load_config") as m:
            config.Config.load_config()
            self.assertEqual(0, m.call_count)
        config.Config._config = None

    @patch("raopt.utils.config.Config.load_config")
    def test_get_dataset_dir(self, m):
        config.Config._config = {"TDRIVE": {"DATASET_PATH": "1234"}}
        config.Config._base_dir = ""
        self.assertEqual("1234", config.Config.get_dataset_dir('tdrive'))
        self.assertEqual(1, m.call_count)
