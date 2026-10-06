"""
Algorithms and programs for analysing and tracking electric field
recordings of weakly electric fish.
"""

import os

# legacy numba ctypes CUDA bindings crash on recent NVIDIA drivers; must be
# set before numba is imported anywhere, so set it on package import
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")
