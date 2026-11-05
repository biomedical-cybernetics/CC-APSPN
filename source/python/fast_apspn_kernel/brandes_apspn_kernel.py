"""
Brandes-APSPN-kernel wrapper

This module provides a Python interface to the Brandes APSPN kernel algorithm,
which solves multiple APSPN-based problems using the Brandes algorithm framework.

Supported problem types:
- 'apspn': All-Pairs Shortest Path Number
- 'gc': Network Geometrical Congruence (requires reference distance)
- 'bc': Betweenness Centrality
- 'apslp': All-Pairs Shortest-Lightest Path (requires reference distance)
- 'phd': Path Hesitation Degree
"""

import ctypes
import numpy as np
from .utils import load_dll, prepare_reference_distance, prepare_adjacency

# Load the DLL once
_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        _lib = load_dll('brandes_apspn_kernel.dll')
        # Define the C function signature
        # int brandes_apspn_kernel_py(const char* algo_type_str, const double* adj_matrix_data,
        #                              int64_t n, const double* rd_matrix_data,
        #                              double* output_data, double* elapsed_time)
        _lib.brandes_apspn_kernel_py.argtypes = [
            ctypes.c_char_p,                   # algo_type_str
            ctypes.POINTER(ctypes.c_double),   # adj_matrix_data
            ctypes.c_int64,                     # n
            ctypes.POINTER(ctypes.c_double),   # rd_matrix_data
            ctypes.POINTER(ctypes.c_double),   # output_data
            ctypes.POINTER(ctypes.c_double)    # elapsed_time
        ]
        _lib.brandes_apspn_kernel_py.restype = ctypes.c_int
    return _lib


def brandes_apspn_kernel(adj_matrix, problem_type='apspn', rd_matrix=None):
    """
    Run Brandes-APSPN-kernel algorithm for various APSPN-based problems
    
    Args:
        adj_matrix: Binary undirected adjacency, n×n; NumPy dense or SciPy sparse.
            This legacy interface converts sparse input to dense internally.
        problem_type (str): Type of problem to solve. One of:
            - 'apspn': All-Pairs Shortest Path Number (default)
            - 'gc': Network Geometrical Congruence (requires rd_matrix)
            - 'bc': Betweenness Centrality
            - 'apslp': All-Pairs Shortest-Lightest Path (requires rd_matrix)
            - 'phd': Path Hesitation Degree
        rd_matrix: Finite symmetric reference distances (required for 'gc' and 'apslp'), n×n
        
    Returns:
        dict: Dictionary containing:
            - 'result': Output (format depends on problem_type)
                - For 'apspn', 'apslp': n×n matrix
                - For 'bc', 'phd': n-length vector
                - For 'gc': scalar value
            - 'elapsed_time': Execution time in seconds
            
    Raises:
        ValueError: If problem_type is invalid or rd_matrix is required but not provided
        RuntimeError: If the algorithm fails
    """
    lib = _get_lib()
    
    # Validate problem type
    valid_types = ['apspn', 'gc', 'bc', 'apslp', 'phd']
    problem_type = problem_type.lower()
    if problem_type not in valid_types:
        raise ValueError(f"Invalid problem_type '{problem_type}'. Must be one of {valid_types}")
    
    # Check if rd_matrix is required
    requires_rd = problem_type in ['gc', 'apslp']
    if requires_rd and rd_matrix is None:
        raise ValueError(f"Problem type '{problem_type}' requires rd_matrix (reference distance)")
    
    # Prepare input matrix
    adj_matrix = prepare_adjacency(adj_matrix)
    n = adj_matrix.shape[0]
    
    # Prepare rd_matrix if needed
    rd_ptr = None
    if rd_matrix is not None:
        rd_matrix = prepare_reference_distance(rd_matrix)
        if rd_matrix.shape != (n, n):
            raise ValueError(f"rd_matrix shape {rd_matrix.shape} doesn't match adj_matrix shape {adj_matrix.shape}")
        rd_ptr = rd_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        # Create a null pointer for C++
        rd_ptr = ctypes.POINTER(ctypes.c_double)()
    
    # Allocate output array based on problem type
    # BC and PHD return vectors (n×1), GC returns scalar, others return matrices (n×n)
    if problem_type in ['bc', 'phd']:
        output = np.zeros(n, dtype=np.float64)
    elif problem_type == 'gc':
        output = np.zeros(1, dtype=np.float64)
    else:  # apspn, apslp
        output = np.zeros((n, n), dtype=np.float64, order='F')
    
    elapsed_time = ctypes.c_double()
    
    # Call the C function
    result_code = lib.brandes_apspn_kernel_py(
        problem_type.encode('utf-8'),
        adj_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(n),
        rd_ptr,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(elapsed_time)
    )
    
    if result_code != 0:
        raise RuntimeError(f"Brandes-APSPN-kernel failed with error code {result_code}")
    
    # For GC, return scalar value
    if problem_type == 'gc':
        output = output[0]
    
    return {
        'result': output,
        'elapsed_time': elapsed_time.value
    }

