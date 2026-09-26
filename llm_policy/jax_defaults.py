"""JAX defaults for the LLM harness, applied before JAX is imported.

The environment steps on the CPU: the harness plays one small environment per run, and the GPUs are for the model
server. Set JAX_PLATFORMS yourself to override. A machine without a GPU also gets JAX's "plugin configuration error:
CUDA_ERROR_NO_DEVICE" (logged at ERROR level, with a traceback, at every start-up); that one message is dropped,
everything else JAX logs goes through."""
import logging
import os


class _NoCudaDevice(logging.Filter):
    def filter(self, record):
        return not ("plugin configuration error" in record.getMessage() and record.exc_info
                    and "CUDA_ERROR_NO_DEVICE" in str(record.exc_info[1]))


def apply():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    logging.getLogger("jax._src.xla_bridge").addFilter(_NoCudaDevice())
