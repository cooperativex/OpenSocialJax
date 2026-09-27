"""Test defaults: JAX on the CPU (the tests never need a GPU, and CI has none), and no "CUDA_ERROR_NO_DEVICE"
plugin error from JAX on a machine without a GPU. The same defaults as llm_policy/jax_defaults.py."""
import logging
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")


class _NoCudaDevice(logging.Filter):
    def filter(self, record):
        return not ("plugin configuration error" in record.getMessage() and record.exc_info
                    and "CUDA_ERROR_NO_DEVICE" in str(record.exc_info[1]))


logging.getLogger("jax._src.xla_bridge").addFilter(_NoCudaDevice())
