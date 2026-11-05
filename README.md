# Network-science-informed speed-up of shortest-path counting boosts betweenness centrality and geometrical congruence computation

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/figures/cc_apspn_logo_dark.svg">
    <img src="docs/figures/cc_apspn_logo.svg" alt="CC-APSPN logo" width="480">
  </picture>
</p>

Many network metrics, including betweenness centrality (BC) and network geometrical congruence (GC), aggregate information over all shortest paths between node pairs. We formalize the **all-pairs shortest path number (APSPN)** problem that underlies this class of **APSPN-based problems** and provide C++ implementations, with MATLAB and Python interfaces, of:

- the **CC-APSPN kernel**, an exact APSPN kernel that uses network-science principles (centrality-driven prioritization, all-paths memoization, direction-adaptive propagation and symmetric mapping) to reuse shortest-path information across searches. It reduces runtime by up to 90% relative to the Brandes-APSPN kernel across diverse network topologies;
- the **parallel CC-APSPN kernel**, which carries this acceleration onto multicore CPUs with OpenMP, including a dedicated scheduling strategy for exact BC;
- **CC-ApproxBC** and **CC-ApproxGC**, sampling-based approximations of BC and GC built on CC-bbBFS, a single-pair shortest-path primitive. CC-ApproxBC keeps SILVAN's progressive-sampling strategy and replaces its balanced bidirectional BFS step with CC-bbBFS.

The repository also contains the **Brandes-APSPN kernel** baseline (generalized from Brandes' BC algorithm), the CC-APSPN kernel combined with **SE/BCC** graph manipulation heuristics, and **CM-GC**, an earlier exact GC algorithm included for comparison.

## Algorithms and where to find them

| Algorithm | Computes | MATLAB MEX | Python function | Source folder |
| :-- | :-- | :-- | :-- | :-- |
| Brandes-APSPN kernel | exact APSPN, BC, GC, APSLP, PHD | `brandes_apspn_kernel_cpp` | `brandes_apspn_kernel` | `source/cpp/brandes_apspn_kernel/` |
| CC-APSPN kernel, CC-ApproxBC, CC-ApproxGC (sequential) | exact APSPN, BC, GC, APSLP, PHD; approximate BC and GC | `cc_apspn_kernel_cpp_ablation_approx` | `cc_apspn_kernel_approx` | `source/cpp/cc_apspn_kernel_approx/` |
| Parallel CC-APSPN kernel | the same seven modes; exact modes run in parallel, approximate modes sequentially | `cc_apspn_kernel_par` | `cc_apspn_kernel_par` | `source/cpp/cc_apspn_kernel_par/` |
| CC-APSPN kernel with SE/BCC | exact APSPN, BC | `cc_sebccopts_apspn_kernel_cpp` | `cc_sebccopts_apspn_kernel` | `source/cpp/cc_sebccopts_apspn_kernel/` |
| CM-GC | exact GC | `cm_gc_cpp` | `cm_gc` | `source/cpp/cm_gc/` |

The sequential CC-APSPN kernel and the two approximate algorithms share one native target. Despite its name, `cc_apspn_kernel_approx` also runs the exact modes: the problem type (`bc` or `bc_approx`, for example) selects exact or approximate computation.

SE/BCC combines the kernel with structural-equivalence (SE) reduction and biconnected-component (BCC) decomposition. It implements APSPN and BC only. SE reduction cannot be applied when reference distances are involved, so GC and APSLP are not supported; PHD is not implemented.

## Problems

| Problem | Description | Needs reference distances | Output |
| :-- | :-- | :-- | :-- |
| **APSPN** | Number of shortest paths between every node pair | No | n×n matrix |
| **BC** | Betweenness centrality | No | length-n vector |
| **GC** | Network geometrical congruence | Yes | scalar |
| **APSLP** | All-pairs shortest-lightest path | Yes | n×n matrix |
| **PHD** | Path hesitation degree | No | length-n vector |

## Input requirements

- Graphs are **unweighted and undirected**. CC-ApproxBC also accepts directed graphs.
- Adjacency matrices are square, binary and finite. MATLAB accepts dense or sparse `double`/`logical` matrices; Python accepts NumPy arrays and SciPy sparse matrices or arrays.
- GC and APSLP need a **reference-distance matrix**: finite, symmetric, the same size as the adjacency matrix and zero on the diagonal.
- Graphs must be **connected**. Disconnected graphs are not supported; run the algorithms on the largest connected component.

Memory use of the matrix-valued problems (APSPN, APSLP) and of the distance matrix grows quadratically with the number of nodes. See the [Installation Guide](docs/INSTALL.md) for details on sparse input.

## Quick start

Precompiled MEX files (MATLAB) and DLLs (Python) are included for Windows x64. On Linux, build the Python libraries locally first. Building, the full Python and MATLAB interfaces and troubleshooting are described in the **[Installation Guide](docs/INSTALL.md)**.

### MATLAB

Run from the repository root:

```matlab
addpath(fullfile(pwd, 'source', 'matlab', 'mex_functions'));

adj_matrix = [0 1 1 0; 1 0 0 1; 1 0 0 1; 0 1 1 0];

% Exact APSPN with the Brandes-APSPN kernel
[apspn_result, elapsed_time] = brandes_apspn_kernel_cpp('apspn', adj_matrix);

% Exact BC with the sequential CC-APSPN kernel ('None' = full algorithm, no ablation)
[bc_result, elapsed_time] = cc_apspn_kernel_cpp_ablation_approx('bc', 'None', adj_matrix);

% Exact BC with the parallel CC-APSPN kernel on four threads
[bc_parallel, elapsed_time, counters] = cc_apspn_kernel_par( ...
    'BC', sparse(adj_matrix), 4, false);

% Exact BC with SE and BCC enabled (this MEX accepts only lowercase mode names)
[bc_sebcc, elapsed_time] = cc_sebccopts_apspn_kernel_cpp('bc', true, true, adj_matrix, false);

% GC needs a reference-distance matrix
rd_matrix = rand(4, 4);
rd_matrix = rd_matrix + rd_matrix';
rd_matrix = rd_matrix - diag(diag(rd_matrix));
[gc_result, elapsed_time] = cm_gc_cpp(adj_matrix, rd_matrix);

% Approximate GC (sampling factor 2) and approximate BC
[gc_approx_result, elapsed_time] = cc_apspn_kernel_cpp_ablation_approx( ...
    'GC_APPROX', 'None', adj_matrix, rd_matrix, 2, false);
[bc_approx_result, elapsed_time] = cc_apspn_kernel_cpp_ablation_approx( ...
    'BC_APPROX', 'None', adj_matrix);

disp(full(apspn_result));
```

### Python

Run from the repository root:

```python
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path("source/python").resolve()))
from fast_apspn_kernel import (
    brandes_apspn_kernel, cc_apspn_kernel_approx, cc_apspn_kernel_par,
    cc_sebccopts_apspn_kernel, cm_gc,
)

adj_matrix = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
], dtype=np.float64)

# Exact APSPN with the Brandes-APSPN kernel
result_apspn = brandes_apspn_kernel(adj_matrix, problem_type="apspn")
print("APSPN matrix:\n", result_apspn["result"])
print(f"Elapsed time: {result_apspn['elapsed_time']:.6f} s")

# Exact BC with the sequential CC-APSPN kernel
result_bc = cc_apspn_kernel_approx(adj_matrix, problem_type="bc", verbose=False)
print("BC:", result_bc["result"])

# Exact BC with the parallel CC-APSPN kernel on four threads
parallel_bc = cc_apspn_kernel_par(
    sparse.csr_matrix(adj_matrix), problem_type="bc", num_threads=4,
)
print("Parallel BC:", parallel_bc["result"], "threads:", parallel_bc["effective_threads"])

# Exact BC with SE and BCC enabled
result_sebcc = cc_sebccopts_apspn_kernel(adj_matrix, problem_type="bc", use_se=True, use_bcc=True)
print("SE/BCC BC:", result_sebcc["result"])

# GC needs a reference-distance matrix
rd_matrix = np.random.rand(4, 4)
rd_matrix = (rd_matrix + rd_matrix.T) / 2
np.fill_diagonal(rd_matrix, 0)
print("GC:", cm_gc(adj_matrix, rd_matrix)["result"])

# Approximate GC (sampling factor 2) and approximate BC
result_gc_approx = cc_apspn_kernel_approx(
    adj_matrix, problem_type="gc_approx", rd_matrix=rd_matrix,
    sampling_factor=2, verbose=False,
)
print("Approximate GC:", result_gc_approx["result"])
result_bc_approx = cc_apspn_kernel_approx(adj_matrix, problem_type="bc_approx", verbose=False)
print("Approximate BC:", result_bc_approx["result"])
```

### C++

```cpp
#include "brandes_apspn_kernel.h"
#include <Eigen/Dense>
#include <iostream>
#include <variant>

int main() {
    Eigen::MatrixXd adj_matrix(4, 4);
    adj_matrix << 0, 1, 1, 0,
                  1, 0, 0, 1,
                  1, 0, 0, 1,
                  0, 1, 1, 0;

    AlgorithmParams params;
    Result result = BrandesApspnKernel(adj_matrix, AlgorithmType::APSPN, params);

    const auto& apspn_matrix = std::get<Eigen::MatrixXd>(result.result);
    std::cout << "APSPN matrix:\n" << apspn_matrix << std::endl;
    std::cout << "Elapsed time: " << result.total_time << " s" << std::endl;
    return 0;
}
```

Build it from the repository root together with the kernel source and the shared graph utilities, for example with GCC:

```bash
g++ -std=c++20 -O2 -I/path/to/eigen3 -Isource/cpp/brandes_apspn_kernel example.cpp \
    source/cpp/brandes_apspn_kernel/brandes_apspn_kernel_cpp.cpp \
    source/common_include/graph_utils.cpp -o example
```

## Benchmark scripts

To time all algorithms on the networks in `data/example_networks/`:

- MATLAB, from the repository root: `run('scripts/benchmark_on_examples.m')`
- Python: install the benchmark dependencies with `python -m pip install pandas h5py`,
  then run `python scripts/benchmark_on_examples.py`.

Both scripts include Brandes, sequential and parallel CC-APSPN, CC-APSPN with
SE+BCC, CM-GC, CC-ApproxBC and CC-ApproxGC, each on its supported problems.
Parallel CC-APSPN requests four threads by default; change `num_threads` in the
script to suit your machine. Requested and effective thread counts are saved
with the timings. Problems requiring reference distances are skipped when the
network file does not provide them.

Results are saved to `results/benchmark/`.

Networks are stored as adjacency matrices (sparse or dense) in MATLAB `.mat` files. The scripts read the adjacency matrix from a variable named `x`, `adjMatrix` or `x_matrices` (a cell array whose first element is used), and the reference-distance matrix, when needed, from `GSP` or `geo_matrices` (cell array).

## Repository layout

```text
source/cpp/<algorithm>/   C++ sources, one folder per algorithm (see the table above)
source/common_include/    shared C++ helpers
source/matlab/            MEX build scripts and prebuilt Windows MEX files
source/python/            Python package fast_apspn_kernel and its build script
third_party/              CC-BFS and CC-BFS-P, which compute the APSP distances used by the kernels
scripts/                  benchmark scripts, nPSO network generator and release tests
data/example_networks/    example networks for the benchmark scripts
docs/                     installation guide and figures
```

## Requirements

- **C++**: a C++20 compiler with OpenMP (tested with MSVC 2022 and GCC 11) and Eigen 3.4.
- **MATLAB**: tested with R2023b on Windows. Compiling the MEX files needs a C++20 compiler configured for MEX.
- **Python**: Python 3.8 or later, NumPy 1.19 or later and SciPy 1.5 or later.

## References

- **Brandes' BC algorithm**: Brandes, U. A faster algorithm for betweenness centrality. *J. Math. Sociol.* 25, 163–177 (2001).
- **Network geometrical congruence and CM-GC**: Cannistraci, C. V. & Muscoloni, A. Geometrical congruence, greedy navigability and myopic transfer in complex networks and brain connectomes. *Nat. Commun.* 13, 7308 (2022).
- **APSLP**: Zwick, U. All pairs lightest shortest paths. In *Proceedings of the 31st Annual ACM Symposium on Theory of Computing*, 61–69 (1999).
- **PHD**: Deng, Y. & Wu, J. Power law of path multiplicity in complex networks. *PNAS Nexus* 3, pgae228 (2024).
- **SILVAN**: Pellegrina, L. & Vandin, F. SILVAN: Estimating betweenness centralities with progressive sampling and non-uniform Rademacher bounds. *ACM Trans. Knowl. Discov. Data* 18, 1–55 (2024).
- **KADABRA**: Borassi, M. & Natale, E. KADABRA is an ADaptive Algorithm for Betweenness via Random Approximation. *ACM J. Exp. Algorithmics* 24, 1–35 (2019).
- **nPSO model**: Muscoloni, A. & Cannistraci, C. V. A nonuniform popularity-similarity optimization (nPSO) model to efficiently generate realistic complex networks with communities. *New J. Phys.* 20, 052002 (2018).

## Citation

If you use this code or the concepts presented in your research, please cite:

```bibtex
@article{xxx,
  title={xxx},
  author={Chen, Junda and Cannistraci, Carlo Vittorio},
  journal={xxx},
  year={xxx}
}
```

## License

This code is released under the MIT License (see [LICENSE](LICENSE)). The approximate-BC sampler is derived from SILVAN and KADABRA (Apache License 2.0), and `timer.h` comes from the GAP Benchmark Suite (BSD 3-Clause License). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contact

For questions or issues, please open an issue on the GitHub repository or email kalokagathos.agon@gmail.com.
