"""Dense and sparse BC examples; run after building and installing the package."""

import numpy as np
from scipy.sparse import csr_matrix

from fast_apspn_kernel import cc_apspn_kernel_par


def main():
    adjacency = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=bool)
    dense_result = cc_apspn_kernel_par(adjacency, "bc", num_threads=1)
    # Set num_threads to the desired number of physical cores.
    sparse_result = cc_apspn_kernel_par(csr_matrix(adjacency), "bc", num_threads=2)
    np.testing.assert_allclose(dense_result["result"], sparse_result["result"])
    print("BC:", sparse_result["result"])
    print("Native elapsed time (s):", sparse_result["elapsed_time"])
    print("Effective threads:", sparse_result["effective_threads"])


if __name__ == "__main__":
    main()
