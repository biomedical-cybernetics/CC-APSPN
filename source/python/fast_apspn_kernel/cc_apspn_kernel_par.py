"""Five parallel exact CC-APSPN modes and two sequential approximations."""

import ctypes
import operator

import numpy as np
from scipy import sparse

from .utils import load_dll, prepare_adjacency, prepare_reference_distance

_lib = None
_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)
_INT64_PTR = ctypes.POINTER(ctypes.c_int64)
_MODES = {"apspn", "bc", "gc", "phd", "apslp", "gc_approx", "bc_approx"}


def _get_lib():
    global _lib
    if _lib is None:
        lib = load_dll("cc_apspn_kernel_par.dll")
        lib.cc_apspn_kernel_par_py.argtypes = [
            ctypes.c_char_p, _DOUBLE_PTR, _INT64_PTR, _INT64_PTR,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_char_p, ctypes.c_uint32, ctypes.c_int,
        ]
        lib.cc_apspn_kernel_par_py.restype = ctypes.c_int
        lib.cc_apspn_kernel_par_last_error.argtypes = []
        lib.cc_apspn_kernel_par_last_error.restype = ctypes.c_char_p
        lib.cc_apspn_kernel_par_effective_threads.argtypes = []
        lib.cc_apspn_kernel_par_effective_threads.restype = ctypes.c_int64
        _lib = lib
    return _lib


def cc_apspn_kernel_par(
    adj_matrix, problem_type="apspn", rd_matrix=None, num_threads=1,
    *, sampling_factor=1.0, delta=0.05, err=0.01, emp_peeling_a=2.0,
    memoization_depth="log_n_w", seed=42, verbose=False,
):
    """Run the settled parallel CC-APSPN kernel.

    ``adj_matrix`` accepts dense NumPy arrays (including bool and float64),
    or SciPy sparse CSR/CSC/COO matrices and arrays. It must be finite, binary,
    square and have at least two nodes. All modes except ``bc_approx`` require
    a symmetric adjacency matrix. Sparse exact input is passed as CSR without
    allocating a dense adjacency matrix; the algorithm may still allocate
    quadratic distance/output matrices. Approximate modes remain sequential
    and their native compatibility path densifies sparse adjacency.

    ``problem_type`` is one of apspn, bc, gc, phd, apslp, gc_approx, bc_approx.
    GC, GC_APPROX and APSLP require a finite symmetric reference-distance ``rd_matrix``
    of the same shape. ``num_threads`` is a positive integer no larger than
    the machine's physical core count (default one). ``seed`` is a uint32
    integer used by BC_APPROX, including zero. Approximate GC uses its native
    sampling scheme and is not controlled by this seed.

    Returns a dict containing ``result`` (matrix, vector, or GC scalar),
    native ``elapsed_time`` in seconds, ``requested_threads`` and actual
    ``effective_threads``. Input conversion/validation time is not included
    in the native elapsed time.
    """
    mode = str(problem_type).lower()
    if mode not in _MODES:
        raise ValueError(f"Invalid problem_type {problem_type!r}; choose from {sorted(_MODES)}")
    try:
        threads = operator.index(num_threads)
    except TypeError as exc:
        raise ValueError("num_threads must be a positive integer") from exc
    if isinstance(num_threads, (bool, np.bool_)) or not 1 <= threads <= np.iinfo(np.int32).max:
        raise ValueError("num_threads must be a positive 32-bit integer")

    adjacency = prepare_adjacency(adj_matrix, allow_directed=(mode == "bc_approx"), preserve_sparse=True)
    n = adjacency.shape[0]
    dense_ptr, indptr_ptr, indices_ptr = _DOUBLE_PTR(), _INT64_PTR(), _INT64_PTR()
    nnz = 0
    if sparse.issparse(adjacency):
        indptr = np.ascontiguousarray(adjacency.indptr, dtype=np.int64)
        indices = np.ascontiguousarray(adjacency.indices, dtype=np.int64)
        indptr_ptr = indptr.ctypes.data_as(_INT64_PTR)
        indices_ptr = indices.ctypes.data_as(_INT64_PTR)
        nnz = adjacency.nnz
    else:
        dense_ptr = adjacency.ctypes.data_as(_DOUBLE_PTR)

    rd_ptr = _DOUBLE_PTR()
    if mode in {"gc", "gc_approx", "apslp"}:
        if rd_matrix is None:
            raise ValueError(f"problem_type {mode!r} requires rd_matrix")
        reference = prepare_reference_distance(rd_matrix)
        if reference.shape != (n, n):
            raise ValueError("rd_matrix must have the same shape as adj_matrix")
        rd_ptr = reference.ctypes.data_as(_DOUBLE_PTR)

    options = {"sampling_factor": sampling_factor, "delta": delta,
               "err": err, "emp_peeling_a": emp_peeling_a}
    for name, value in options.items():
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive scalar")
    if sampling_factor > n:
        raise ValueError("sampling_factor must not exceed the number of nodes")
    if delta >= 1:
        raise ValueError("delta must lie strictly between 0 and 1")
    if err >= 1:
        raise ValueError("err must lie strictly between 0 and 1")
    memoization_depth = str(memoization_depth).lower()
    if memoization_depth not in {"avg_deg", "log_n", "sqrt_avg_deg", "log_n_w", "none"}:
        raise ValueError("Unsupported memoization_depth")
    try:
        seed = operator.index(seed)
    except TypeError as exc:
        raise ValueError("seed must be a uint32 integer") from exc
    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("seed must be a uint32 integer")

    if mode in {"apspn", "apslp"}:
        output = np.zeros((n, n), dtype=np.float64, order="F")
    elif mode in {"gc", "gc_approx"}:
        output = np.zeros(1, dtype=np.float64)
    else:
        output = np.zeros(n, dtype=np.float64)
    elapsed = ctypes.c_double()
    lib = _get_lib()
    status = lib.cc_apspn_kernel_par_py(
        mode.upper().encode("ascii"), dense_ptr, indptr_ptr, indices_ptr,
        n, nnz, threads, rd_ptr, output.ctypes.data_as(_DOUBLE_PTR), ctypes.byref(elapsed),
        *(float(value) for value in options.values()),
        memoization_depth.encode("ascii"), seed, int(bool(verbose)),
    )
    if status:
        detail = lib.cc_apspn_kernel_par_last_error()
        message = detail.decode("utf-8", errors="replace") if detail else "unknown native error"
        raise RuntimeError(f"Parallel CC-APSPN failed ({status}): {message}")
    return {"result": output[0] if mode in {"gc", "gc_approx"} else output,
            "elapsed_time": elapsed.value, "requested_threads": threads,
            "effective_threads": int(lib.cc_apspn_kernel_par_effective_threads())}
