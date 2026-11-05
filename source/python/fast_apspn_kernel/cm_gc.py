"""
CM-GC wrapper

This module provides a Python interface to the CM-GC algorithm for computing network geometrical congruence.
"""

import ctypes
import numpy as np
from .utils import load_dll, prepare_reference_distance, prepare_adjacency

# Load the DLL once
_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        _lib = load_dll('cm_gc.dll')
        # Define the C function signature
        # int cm_gc_py(const double* adj_matrix_data, int64_t n,
        #              const double* rd_matrix_data, double* output_gc,
        #              double* elapsed_time)
        _lib.cm_gc_py.argtypes = [
            ctypes.POINTER(ctypes.c_double),   # adj_matrix_data
            ctypes.c_int64,                     # n
            ctypes.POINTER(ctypes.c_double),   # rd_matrix_data
            ctypes.POINTER(ctypes.c_double),   # output_gc
            ctypes.POINTER(ctypes.c_double)    # elapsed_time
        ]
        _lib.cm_gc_py.restype = ctypes.c_int
    return _lib


def cm_gc(adj_matrix, rd_matrix):
    """
    Compute network geometrical congruence using the CM-GC algorithm
    
    Args:
        adj_matrix: Binary undirected adjacency, n×n; NumPy dense or SciPy sparse.
            This legacy interface converts sparse input to dense internally.
        rd_matrix: Finite symmetric reference distance matrix, n×n
        
    Returns:
        dict: Dictionary containing:
            - 'result': GC scalar value
            - 'elapsed_time': Execution time in seconds
            
    Raises:
        ValueError: If matrices are not square or dimensions don't match
        RuntimeError: If the algorithm fails
    """
    lib = _get_lib()
    
    # Prepare input matrices
    adj_matrix = prepare_adjacency(adj_matrix)
    rd_matrix = prepare_reference_distance(rd_matrix)
    
    n = adj_matrix.shape[0]
    
    if rd_matrix.shape != (n, n):
        raise ValueError(f"rd_matrix shape {rd_matrix.shape} doesn't match adj_matrix shape {adj_matrix.shape}")
    
    # Allocate output (single scalar value)
    output_gc = np.zeros(1, dtype=np.float64)
    elapsed_time = ctypes.c_double()
    
    # Call the C function
    result_code = lib.cm_gc_py(
        adj_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(n),
        rd_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        output_gc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(elapsed_time)
    )
    
    if result_code != 0:
        raise RuntimeError(f"CM-GC failed with error code {result_code}")
    
    return {
        'result': output_gc[0],
        'elapsed_time': elapsed_time.value
    }

