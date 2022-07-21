#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------
"""
Import this first when using tensorflow to remove the warning messages and only use the specified GPU.
Must be imported BEFORE tensorflow!
"""


class TensorflowConfig:
    """Used for appropriate configuration of TensorFlow."""
    called = False

    @classmethod
    def configure_tensorflow(cls, gpu_num: int = None):
        import os
        from raopt.utils.config import Config
        if not cls.called:
            os.environ['TF_XLA_FLAGS'] = "--tf_xla_enable_xla_devices"
            if gpu_num is None:
                gpu_num = Config.get_gpu_num()
            # Only use one GPU
            if int(gpu_num) >= 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
                print(f"Using GPU {gpu_num}!")
            # Reduce all the Keras/TensorFlow info messages (only show warning and above)
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
        cls.called = True
