# Installation and parallel-kernel usage

Run these commands from the repository root. Five native targets are provided:
Brandes, CC-APSPN (exact and approximate), CC-APSPN with SE/BCC, CM-GC, and
parallel CC-APSPN. They share the Python package `fast_apspn_kernel`.

Windows x64 MATLAB MEX files and Python DLLs are included. Linux Python users
should build shared libraries on their own machine. A Linux MEX build path is
provided, but Linux MATLAB execution has not been tested for this release.
macOS is not covered by these build scripts.

## Python on Linux

Requirements: Python 3.8 or newer, NumPy, SciPy, a C++20 compiler with OpenMP,
and Eigen headers. The checked environment uses Ubuntu 22.04, GCC 11.4,
Python 3.11 and Eigen 3.4. GCC's OpenMP runtime (`libgomp`) is needed at run time.

On Ubuntu, install the compiler and headers if needed:

```bash
sudo apt-get install build-essential libeigen3-dev
python -m pip install numpy scipy setuptools wheel
python source/python/build_native.py --eigen /usr/include/eigen3
python -m pip install ./source/python --no-build-isolation
```

The build script compiles all five targets using C++20, position-independent
code and OpenMP. It writes `.so` files inside
`source/python/fast_apspn_kernel/lib/`. Do not copy a Windows DLL to Linux or
assume that a locally built `.so` is portable across unrelated Linux systems.

## Python on Windows

To use the included x64 DLLs:

```powershell
python -m pip install numpy scipy setuptools wheel
python -m pip install ./source/python --no-build-isolation
```

The Microsoft Visual C++ x64 runtime must be installed. To rebuild, install
Visual Studio 2022 Build Tools with Desktop development with C++, obtain Eigen,
and run:

```powershell
python source/python/build_native.py --eigen C:/path/to/eigen-3.4.0
python -m pip install ./source/python --no-build-isolation --force-reinstall --no-deps
```

The builder discovers Visual Studio automatically. `EIGEN3_INCLUDE_DIR`
can replace `--eigen`. The existing
`source/python/compile_windows_msvc.bat` launches this same builder. Only
libraries for the current operating system are included when building a wheel.

The example benchmark script also needs Pandas and h5py:

```bash
python -m pip install pandas h5py
```

These two packages are not needed for ordinary calls to the algorithms.

## Python API

`cc_apspn_kernel_approx` is the single sequential entry point for exact and
approximate computation, matching the combined MATLAB interface. Select the
computation with `problem_type`; its default is `gc_approx`, which needs
`rd_matrix`. The function prints progress by default (`verbose=True`); pass
`verbose=False` to reduce it. Approximate BC and CM-GC still print some
diagnostic output.

```python
from fast_apspn_kernel import cc_apspn_kernel_approx

# A is your binary adjacency matrix; RD is its reference-distance matrix.
# exact_bc = cc_apspn_kernel_approx(A, problem_type="bc")
# approx_bc = cc_apspn_kernel_approx(A, problem_type="bc_approx")
# exact_gc = cc_apspn_kernel_approx(A, problem_type="gc", rd_matrix=RD)
```

All seven modes run sequentially through this function. The exact modes accept
`ablation_type="None"` (the full algorithm), `"Pri"`, `"Mem"`, `"Symm"` or
`"DAP"`; in Python these names are case-sensitive, while the MATLAB interface
accepts any case. Approximate modes accept sampling options. The result
dictionary contains `result` and `elapsed_time`; matrix outputs are n-by-n,
exact/approximate BC and PHD outputs are n-length vectors, and exact/approximate
GC outputs are scalars.

For parallel exact computation, use `cc_apspn_kernel_par`:

```python
import numpy as np
from scipy.sparse import csr_matrix
from fast_apspn_kernel import cc_apspn_kernel_par

A = np.array([[0, 1, 0, 1],
              [1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 0, 1, 0]], dtype=bool)
answer = cc_apspn_kernel_par(csr_matrix(A), problem_type="bc", num_threads=4)
print(answer["result"])
print(answer["elapsed_time"], answer["effective_threads"])
```

Dense NumPy arrays and SciPy CSR/CSC/COO matrices or arrays are accepted. Inputs
are copied/canonicalized when necessary; callers need not choose a particular
NumPy memory order. Adjacency must be finite, binary and square, with at least
two nodes. Exact modes and approximate GC require symmetry; approximate BC
also accepts directed graphs.

`problem_type` accepts `apspn`, `bc`, `gc`, `phd`, `apslp`, `gc_approx` and
`bc_approx`. GC, approximate GC and APSLP require `rd_matrix`: a finite,
symmetric reference-distance matrix matching the adjacency shape. Geometric
reference distances should be zero on the diagonal and positive off it.

The five exact modes run in parallel. `num_threads` defaults to 1; select a
positive number within the machine's physical-core limit. Returned
`requested_threads` and `effective_threads` make the setting explicit. Native
elapsed time excludes Python input conversion and validation.

The two approximate modes retain the established sequential algorithms, even
when `num_threads` exceeds 1. Their effective thread count is 1. Options are
keyword-only:

```python
approx_bc = cc_apspn_kernel_par(
    A, "bc_approx", delta=0.05, err=0.01,
    emp_peeling_a=2, memoization_depth="log_n_w", seed=42,
)
# RD is your reference-distance matrix, with the same shape as A.
# approx_gc = cc_apspn_kernel_par(A, "gc_approx", rd_matrix=RD, sampling_factor=2)
```

The `seed` controls approximate BC, not approximate GC's native sampling
scheme. A fixed seed need not produce identical samples across Windows and
Linux, whose C++ random-number distribution implementations differ.
Sparse adjacency stays sparse in the parallel **exact** input path.
The parallel Python approximate path and the other Python wrappers densify
sparse adjacency. Distance matrices and matrix-valued outputs may also require
quadratic memory; sparse input alone does not remove that cost.

Use `num_threads` for internal parallelism; concurrent calls from multiple
Python threads are not supported, especially for approximate routines with
process-static state.

The CC-APSPN kernel with SE/BCC graph manipulation supports APSPN and BC:

```python
from fast_apspn_kernel import cc_sebccopts_apspn_kernel

answer = cc_sebccopts_apspn_kernel(A, problem_type="bc", use_se=True, use_bcc=True)
print(answer["result"], answer["elapsed_time"])
```

## MATLAB

Use the included Windows x64 MEX files:

```matlab
addpath(fullfile(pwd, 'source', 'matlab', 'mex_functions'));
A = sparse([0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0]);
[bc, seconds, counters] = cc_apspn_kernel_par('BC', A, 4, false);
```

Dense and sparse `double` and `logical` adjacency are supported. Exact calls:

```matlab
[value, seconds, counters] = cc_apspn_kernel_par('APSPN', A, threads, false);
[value, seconds, counters] = cc_apspn_kernel_par('BC',    A, threads, false);
[value, seconds, counters] = cc_apspn_kernel_par('PHD',   A, threads, false);
[value, seconds, counters] = cc_apspn_kernel_par('GC',    A, threads, RD, false);
[value, seconds, counters] = cc_apspn_kernel_par('APSLP', A, threads, RD, false);
```

Approximate modes run sequentially:

```matlab
[value, seconds, counters] = cc_apspn_kernel_par( ...
    'GC_APPROX', A, threads, RD, 2, false);
[value, seconds, counters] = cc_apspn_kernel_par( ...
    'BC_APPROX', A, threads, 0.05, 0.01, 2, 'log_n_w', 42, false);
```

The SE/BCC kernel takes positional flags and accepts only the lowercase mode
names `'apspn'` and `'bc'`. Verbose output is on unless the last argument is
`false`:

```matlab
[value, seconds, counters] = cc_sebccopts_apspn_kernel_cpp('bc', use_se, use_bcc, A, false);
```

To compile all five MEX targets:

```matlab
addpath(fullfile(pwd, 'source', 'matlab'));
setenv('EIGEN_INCLUDE_DIR', 'C:/path/to/eigen-3.4.0'); % Windows
% setenv('EIGEN_INCLUDE_DIR', '/usr/include/eigen3');   % Linux
compile_all_mex;
```

Windows compilation selects MATLAB's MSVC 2022 compiler configuration to avoid
an older installed compiler without C++20 support. Set `MEX_CXX_OPTIONS` to a
suitable compiler-options XML file if needed. Linux uses GCC's
`-std=c++20 -fopenmp` flags. Windows MATLAB R2023b is the tested version; Linux
MATLAB testing is deferred.

The parallel Windows MEX keeps its VCOMP OpenMP module loaded after a
multithreaded call. Restart MATLAB before replacing/rebuilding that loaded MEX;
do not force-unlock it. This protects OpenMP runtime state.

## Checks and troubleshooting

From the repository root:

```bash
python -m unittest discover -s scripts/tests -p "test_*.py" -v
```

In MATLAB:

```matlab
addpath(fullfile(pwd, 'scripts', 'tests'));
test_release_parallel;
```

These are correctness/compatibility checks, not benchmarks.

- Missing library: run `build_native.py` and reinstall the package.
- Windows DLL dependency error: install the x64 Visual C++ runtime and use
  64-bit Python.
- Linux OpenMP dependency error: ensure the GCC OpenMP runtime is available.
- Eigen not found: point the build option at the directory containing
  `Eigen/Dense`, not its parent or `Eigen` itself.
- Invalid input: check adjacency symmetry/binary values and the reference
  matrix's dimensions and symmetry. Approximate BC allows directed adjacency.
- NaN or unexpected results: check that the graph is connected. Disconnected
  graphs are not supported; use the largest connected component.

## Native C++ integration

Keep the repository's source layout: several helpers use relative includes.
Reproducible source lists are in `source/python/build_native.py` and
`source/matlab/compile_all_mex.m`. `cc_bfs_p_apsp_l.cpp` includes its embedded
CC-BFS-P implementation; do not also compile that embedded `.cpp` separately.
Use C++20, Eigen and the platform's OpenMP toolchain consistently.


