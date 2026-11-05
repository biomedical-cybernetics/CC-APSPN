"""Native-library loading and matrix preparation shared by the Python APIs."""

import ctypes
import os
from pathlib import Path
import sys

import numpy as np
from scipy import sparse


def load_dll(dll_name):
    """Load the platform library from an installed package or a source checkout."""
    stem = Path(dll_name).stem
    suffix = ".dll" if sys.platform == "win32" else ".so"
    package_dir = Path(__file__).resolve().parent
    candidates = [package_dir / "lib" / (stem + suffix),
                  package_dir.parent / "lib" / (stem + suffix)]
    for path in candidates:
        if path.is_file():
            try:
                return ctypes.CDLL(str(path))
            except OSError as exc:
                runtime = "the Microsoft Visual C++ x64 runtime" if os.name == "nt" else "the OpenMP runtime (libgomp)"
                raise OSError(f"Could not load {path}; ensure {runtime} is installed. {exc}") from exc
    raise OSError(
        f"Native library {stem + suffix} not found. From the release checkout, "
        "run python source/python/build_native.py --eigen <Eigen include directory>, "
        "then install source/python with pip."
    )


def prepare_matrix(matrix):
    """Return a real square float64 matrix in Eigen's column-major order.

    Legacy native entry points require dense matrices, so SciPy sparse input
    is converted here. The parallel wrapper bypasses this helper for adjacency
    input and passes CSR arrays without densifying.
    """
    if sparse.issparse(matrix):
        if np.iscomplexobj(matrix.data):
            raise ValueError("Matrix must be real")
        matrix = matrix.toarray()
    if np.iscomplexobj(matrix):
        raise ValueError("Matrix must be real")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Matrix must contain only finite values")
    return np.asfortranarray(matrix)


def prepare_adjacency(matrix, *, allow_directed=False, preserve_sparse=False):
    """Validate an unweighted graph and return float64 dense or canonical CSR.

    Repeated coordinates are summed using SciPy's normal matrix semantics
    before the binary check; duplicates adding to a weight other than one
    are rejected. Explicit zeros are removed without densifying sparse input.
    """
    if sparse.issparse(matrix):
        if np.iscomplexobj(matrix.data):
            raise ValueError("Adjacency must be real")
        matrix = matrix.astype(np.float64).tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
            raise ValueError("Adjacency must be square with at least two nodes")
        if not np.isfinite(matrix.data).all() or not np.isin(matrix.data, [0, 1]).all():
            raise ValueError("Adjacency must contain only finite binary values (0 or 1)")
        if not allow_directed and (matrix != matrix.T).nnz:
            raise ValueError("This problem requires a symmetric, undirected adjacency matrix")
        if preserve_sparse:
            return matrix
        return np.asfortranarray(matrix.toarray())
    matrix = prepare_matrix(matrix)
    if matrix.shape[0] < 2:
        raise ValueError("Adjacency must have at least two nodes")
    if not np.isin(matrix, [0, 1]).all():
        raise ValueError("Adjacency must contain only binary values (0 or 1)")
    if not allow_directed and not np.array_equal(matrix, matrix.T):
        raise ValueError("This problem requires a symmetric, undirected adjacency matrix")
    return matrix


def prepare_reference_distance(matrix):
    """Prepare the symmetric reference distances required by undirected modes."""
    matrix = prepare_matrix(matrix)
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError("rd_matrix must be symmetric for undirected GC/APSLP modes")
    return matrix

