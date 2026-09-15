"""Differentiable particle filtering for the Dacca benchmark.

This is a small JAX implementation of the differentiable ensemble transform
of Corenflos et al. (2021).  It lives outside Pypomp deliberately: the module
is benchmark code, not a new Pypomp API.
"""

import os

# This benchmark shares a GPU with unrelated work.  Set conservative JAX
# allocation defaults before importing any module that initializes JAX.  A
# caller can still override either variable explicitly.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from .dpf import DPFConfig, DPFResult, DaccaArrays, dacca_dpf, dpf_value_and_grad
from .transport import TransportConfig, ensemble_transform, transport_matrix

__all__ = [
    "DPFConfig",
    "DPFResult",
    "DaccaArrays",
    "TransportConfig",
    "dacca_dpf",
    "dpf_value_and_grad",
    "ensemble_transform",
    "transport_matrix",
]
