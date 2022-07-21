#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
import configparser
import logging
from pathlib import Path
from typing import Iterable

DATASETS = ['TDRIVE', 'GEOLIFE', 'FOURSQUARE_NYC', 'FOURSQUARE_GLOBAL', 'FOURSQUARE_SEMANTIC']
MECHANISMS = ['SDD', 'CNOISE']


log = logging.getLogger()


def test_dataset(dataset: str):
    """Test whether argument is valid dataset"""
    if dataset.upper() not in DATASETS:
        raise ValueError(f"Unknown dataset name {dataset}")


def get_basename(mechanism: str, epsilon: float, sensitivity: float, version: int) -> str:
    """Return the base name of files containing protected trajectories."""
    return f"{mechanism.lower()}_M{round(sensitivity)}_e{float(epsilon)}_{int(version)}"


def get_bool(k: str, v: str) -> bool:
    if v.upper() == 'TRUE':
        return True
    if v.upper() == 'FALSE':
        return False
    else:
        return RuntimeError(f"Invalid boolean argument {v} for key {k}, must be False or True.")


class Config:
    _config: configparser.ConfigParser = None
    _base_dir = str(Path(__file__).parent.parent.parent.resolve()) + '/'
    _config_filepath = f"{_base_dir}config/config.ini"

    @classmethod
    def _get_absolute_dir(cls, rel_dir: str):
        if rel_dir[0] == '/':
            return rel_dir
        else:
            return cls._base_dir + rel_dir

    @classmethod
    def _load_config(cls) -> None:
        cls._config = configparser.ConfigParser()
        cls._config.read(cls._config_filepath)

    @classmethod
    def load_config(cls) -> None:
        if cls._config is None:
            cls._load_config()

    @classmethod
    def get_dataset_dir(cls, dataset: str) -> str:
        test_dataset(dataset)
        cls.load_config()
        return cls._config[dataset.upper()]['DATASET_PATH']

    @classmethod
    def get_config(cls) -> configparser.ConfigParser:
        cls.load_config()
        return cls._config

    @classmethod
    def get(cls, category: str, key: str):
        cls.load_config()
        return cls._config[category][key]

    @classmethod
    def get_logdir(cls):
        cls.load_config()
        return cls._get_absolute_dir(cls._config['DEFAULT']['LOG_DIR'])

    @classmethod
    def get_eval_dir(cls):
        cls.load_config()
        return cls._base_dir + 'results/eval/'

    @classmethod
    def get_tensorboard_dir(cls):
        cls.load_config()
        return cls._get_absolute_dir(cls._config['DEFAULT']['TENSORBOARD_DIR'])

    @classmethod
    def get_output_dir(cls):
        cls.load_config()
        return cls._get_absolute_dir(cls._config['DEFAULT']['OUTPUT_DIR'])

    @classmethod
    def get_filenames_tdrive(cls) -> Iterable:
        cls.load_config()
        start = int(cls._config['TDRIVE']['MIN_FILE'])
        stop = int(cls._config['TDRIVE']['MAX_FILE'])
        return (f"{i}.txt" for i in range(start, stop + 1, 1))

    @classmethod
    def get_temp_dir(cls) -> str:
        cls.load_config()
        return cls._get_absolute_dir(cls._config['DEFAULT']['TEMP_DIR'])

    @classmethod
    def get_cache_dir(cls, dataset: str) -> str:
        test_dataset(dataset)
        cls.load_config()
        return cls._get_absolute_dir(cls._config['DEFAULT']['CACHE_DIR'] + f'{dataset.lower()}/')

    @classmethod
    def get_csv_dir(cls, dataset: str) -> str:
        test_dataset(dataset)
        cls.load_config()
        return cls._get_absolute_dir(cls._config['DEFAULT']['CSV_DIR'] + f'{dataset.lower()}/')

    @classmethod
    def get_basedir(cls) -> str:
        cls.load_config()
        return cls._base_dir

    @classmethod
    def get_test_dir(cls) -> str:
        return cls.get_basedir() + 'test/'

    @classmethod
    def get_parameter_path(cls) -> str:
        cls.load_config()
        return cls.get_basedir() + "model_parameters/"

    @classmethod
    def is_caching(cls) -> bool:
        """
        Is caching active?
        :return: Caching Active?
        """
        cls.load_config()
        return cls._config['DEFAULT']['CACHING'] == 'True'

    @classmethod
    def get_gpu_num(cls) -> str:
        cls.load_config()
        return cls._config['DEFAULT']['GPU_NUM']

    @classmethod
    def get_early_stop(cls) -> int:
        cls.load_config()
        return int(cls._config['DEFAULT']['EARLY_STOP'])

    @classmethod
    def get_batch_size(cls) -> int:
        cls.load_config()
        return int(cls._config['DEFAULT']['BATCH_SIZE'])

    @classmethod
    def get_epochs(cls) -> int:
        cls.load_config()
        return int(cls._config['DEFAULT']['EPOCHS'])

    @classmethod
    def get_learning_rate(cls) -> int:
        cls.load_config()
        return float(cls._config['DEFAULT']['LEARNING_RATE'])

    @classmethod
    def get_max_len(cls, dataset: str):
        cls.load_config()
        test_dataset(dataset)
        return int(cls._config[dataset.upper()]['MAX_LENGTH'])

    @classmethod
    def get_tul_config(cls, key: str):
        cls.load_config()
        return cls._config['TUL'][key]

    # @classmethod
    # def use_csv(cls):
    #     cls.load_config()
    #     return bool(cls._config['DEFAULT']['READ_FROM_CSV'])

    @classmethod
    def parallelization_enabled(cls) -> bool:
        cls.load_config()
        return get_bool('PARALLEL', cls._config['DEFAULT']['PARALLEL'])

    @classmethod
    def continue_evaluation(cls) -> bool:
        cls.load_config()
        return get_bool('CONTINUE_EVAL', cls._config['DEFAULT']['CONTINUE_EVAL'])

    @classmethod
    def use_tensorboard(cls) -> bool:
        cls.load_config()
        return get_bool('TENSORBOARD', cls._config['DEFAULT']['TENSORBOARD'])

    @classmethod
    def use_all_cpus(cls) -> bool:
        cls.load_config()
        return get_bool('USE_ALL_CPUS', cls._config['DEFAULT']['USE_ALL_CPUS'])

    @classmethod
    def get_M(cls, dataset):
        cls.load_config()
        test_dataset(dataset)
        try:
            # If M is defined it overwrites all other choices
            M = int(Config.get(dataset, 'M'))
        except KeyError:
            # In case M is not defined, we default to MAX_SPEED * INTERVAL
            M = int(Config.get(dataset, 'OUTLIER_SPEED')) * 1000 / 3600 * int(Config.get(dataset, 'INTERVAL'))
        return M
