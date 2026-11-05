"""
Sequential CC-APSPN wrapper for exact and approximate problem modes.

Like the combined MATLAB interface, this function selects the computation
through problem_type and calls the existing native library.
"""

import ctypes
import numpy as np
from .utils import load_dll, prepare_reference_distance, prepare_adjacency


_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = load_dll("cc_apspn_kernel_approx.dll")
        _lib.cc_apspn_kernel_py.argtypes = [
            ctypes.c_char_p,                   # algo_type_str
            ctypes.c_char_p,                   # ablation_type_str
            ctypes.POINTER(ctypes.c_double),   # adj_matrix_data
            ctypes.c_int64,                    # n
            ctypes.POINTER(ctypes.c_double),   # rd_matrix_data
            ctypes.POINTER(ctypes.c_double),   # output_data
            ctypes.POINTER(ctypes.c_double),   # elapsed_time
            ctypes.c_double,                   # sampling_factor
            ctypes.c_double,                   # delta
            ctypes.c_double,                   # err
            ctypes.c_double,                   # emp_peeling_a
            ctypes.c_char_p,                   # memoization_depth_str
            ctypes.c_uint32,                   # seed
            ctypes.c_int,                      # verbose
        ]
        _lib.cc_apspn_kernel_py.restype = ctypes.c_int
    return _lib


def cc_apspn_kernel_approx(
    adj_matrix,
    problem_type="gc_approx",
    rd_matrix=None,
    ablation_type="None",
    sampling_factor=None,
    delta=None,
    err=None,
    emp_peeling_a=None,
    memoization_depth=None,
    seed=None,
    verbose=True,
):
    """
    Run the sequential exact CC-APSPN kernel, CC-ApproxGC or CC-ApproxBC.

    Args:
        adj_matrix: Binary adjacency, n x n; NumPy dense or SciPy sparse.
            This interface densifies sparse input. All modes require an
            undirected graph except BC_APPROX, which also accepts directed graphs.
        problem_type (str): 'apspn', 'bc', 'gc', 'apslp', 'phd', 'bc_approx'
            or 'gc_approx'. The existing default remains 'gc_approx'.
        rd_matrix: Finite symmetric reference distances, n x n, required for
            'gc', 'gc_approx' and 'apslp'.
        ablation_type (str): Exact-mode ablation: 'None' (full algorithm),
            'Pri', 'Mem', 'Symm' or 'DAP'. Approximate modes do not use it.
        sampling_factor: Optional sampling factor for approximate GC.
        delta: Optional confidence parameter for approximate BC (default 0.05).
        err: Optional error parameter for approximate BC (default 0.01).
        emp_peeling_a: Optional empirical peeling parameter for approximate BC (default 2.0).
        memoization_depth: Optional memoization strategy string for approximate BC (default 'log_n_w').
        seed: Optional random seed for approximate BC (default 42).
        verbose (bool): Verbose flag passed to the C++ implementation.

    Returns:
        dict with 'result' and native 'elapsed_time' in seconds. The result is
        an n x n matrix for APSPN/APSLP, an n-length vector for BC/BC_APPROX/PHD,
        or a scalar for GC/GC_APPROX.
    """
    lib = _get_lib()

    problem_type = problem_type.lower()
    valid_types = {mode: mode.upper() for mode in
                   ("apspn", "bc", "gc", "apslp", "phd", "bc_approx", "gc_approx")}
    if problem_type not in valid_types:
        raise ValueError(f"Invalid problem_type '{problem_type}'. Must be one of {sorted(valid_types)}")

    valid_ablations = ["None", "Pri", "Mem", "Symm", "DAP"]
    if ablation_type not in valid_ablations:
        raise ValueError(f"Invalid ablation_type '{ablation_type}'. Must be one of {valid_ablations}")

    adj_matrix = prepare_adjacency(adj_matrix, allow_directed=(problem_type == "bc_approx"))
    n = adj_matrix.shape[0]

    rd_ptr = ctypes.POINTER(ctypes.c_double)()
    if problem_type in ("gc", "gc_approx", "apslp"):
        if rd_matrix is None:
            raise ValueError(f"problem_type '{problem_type}' requires rd_matrix")
        rd_matrix = prepare_reference_distance(rd_matrix)
        if rd_matrix.shape != (n, n):
            raise ValueError(f"rd_matrix shape {rd_matrix.shape} doesn't match adj_matrix shape {adj_matrix.shape}")
        rd_ptr = rd_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    if problem_type in ("bc", "bc_approx", "phd"):
        output = np.zeros(n, dtype=np.float64)
    elif problem_type in ("gc", "gc_approx"):
        output = np.zeros(1, dtype=np.float64)
    else:
        output = np.zeros((n, n), dtype=np.float64, order="F")

    memoization_depth_bytes = (
        memoization_depth.encode("utf-8")
        if memoization_depth is not None
        else None
    )
    elapsed_time = ctypes.c_double()

    result_code = lib.cc_apspn_kernel_py(
        valid_types[problem_type].encode("utf-8"),
        ablation_type.encode("utf-8"),
        adj_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(n),
        rd_ptr,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(elapsed_time),
        ctypes.c_double(-1.0 if sampling_factor is None else sampling_factor),
        ctypes.c_double(-1.0 if delta is None else delta),
        ctypes.c_double(-1.0 if err is None else err),
        ctypes.c_double(-1.0 if emp_peeling_a is None else emp_peeling_a),
        memoization_depth_bytes,
        ctypes.c_uint32(0 if seed is None else seed),
        ctypes.c_int(1 if verbose else 0),
    )

    if result_code != 0:
        raise RuntimeError(f"Sequential CC-APSPN failed with error code {result_code}")

    return {
        "result": output[0] if problem_type in ("gc", "gc_approx") else output,
        "elapsed_time": elapsed_time.value,
    }
