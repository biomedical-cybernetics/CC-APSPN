"""
CC-SE-BCC-Opts-APSPN-kernel wrapper

This module provides a Python interface to the CC-APSPN-kernel algorithm with 
SE and BCC optimization options.

Supported problem types:
- 'apspn': All-Pairs Shortest Path Number
- 'bc': Betweenness Centrality

SE and BCC optimizations can be enabled/disabled independently.
"""

import ctypes
import numpy as np
from .utils import load_dll, prepare_adjacency

# Load the DLL once
_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        _lib = load_dll('cc_sebccopts_apspn_kernel.dll')
        # Define the C function signature
        # int cc_sebccopts_apspn_kernel_py(const char* algo_type_str, int use_se,
        #                                   int use_bcc, const double* adj_matrix_data,
        #                                   int64_t n, double* output_data,
        #                                   double* elapsed_time)
        _lib.cc_sebccopts_apspn_kernel_py.argtypes = [
            ctypes.c_char_p,                   # algo_type_str
            ctypes.c_int,                       # use_se
            ctypes.c_int,                       # use_bcc
            ctypes.POINTER(ctypes.c_double),   # adj_matrix_data
            ctypes.c_int64,                     # n
            ctypes.POINTER(ctypes.c_double),   # output_data
            ctypes.POINTER(ctypes.c_double)    # elapsed_time
        ]
        _lib.cc_sebccopts_apspn_kernel_py.restype = ctypes.c_int
    return _lib


def cc_sebccopts_apspn_kernel(adj_matrix, problem_type='apspn', use_se=True, use_bcc=True):
    """
    Run CC-APSPN-kernel with SE and BCC optimization options
    
    Args:
        adj_matrix: Binary undirected adjacency, n×n; NumPy dense or SciPy sparse.
            This legacy interface converts sparse input to dense internally.
        problem_type (str): Type of problem to solve. One of:
            - 'apspn': All-Pairs Shortest Path Number (default)
            - 'bc': Betweenness Centrality
        use_se (bool): Enable SE
        use_bcc (bool): Enable BCC
        
    Returns:
        dict: Dictionary containing:
            - 'result': Output (format depends on problem_type)
                - For 'apspn': n×n matrix
                - For 'bc': n-length vector
            - 'elapsed_time': Execution time in seconds
            
    Raises:
        ValueError: If problem_type is invalid
        RuntimeError: If the algorithm fails
    """
    lib = _get_lib()
    
    # Validate problem type
    valid_types = ['apspn', 'bc']
    problem_type = problem_type.lower()
    if problem_type not in valid_types:
        raise ValueError(f"Invalid problem_type '{problem_type}'. Must be one of {valid_types}")
    
    # Prepare input matrix
    adj_matrix = prepare_adjacency(adj_matrix)
    n = adj_matrix.shape[0]
    
    # Allocate output array based on problem type
    if problem_type == 'bc':
        output = np.zeros(n, dtype=np.float64)
    else:  # apspn
        output = np.zeros((n, n), dtype=np.float64, order='F')
    
    elapsed_time = ctypes.c_double()
    
    # Call the C function
    result_code = lib.cc_sebccopts_apspn_kernel_py(
        problem_type.encode('utf-8'),
        ctypes.c_int(1 if use_se else 0),
        ctypes.c_int(1 if use_bcc else 0),
        adj_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(n),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(elapsed_time)
    )
    
    if result_code != 0:
        raise RuntimeError(f"CC-SE-BCC-Opts-APSPN-kernel failed with error code {result_code}")
    
    return {
        'result': output,
        'elapsed_time': elapsed_time.value
    }

