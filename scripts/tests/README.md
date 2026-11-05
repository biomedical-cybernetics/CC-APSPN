# Release checks

These small tests check packaging and interface compatibility, not performance.
Build the release libraries first. The test suite needs Python 3.9 or later,
NumPy, and SciPy 1.8 or later (to exercise sparse arrays as well as matrices);
no additional test framework is required. These test requirements are stricter
than the package's runtime minimums.

```text
python scripts/tests/test_release_parallel.py
```

The Python tests cover all seven modes through the unified sequential
`cc_apspn_kernel_approx` interface. The five exact modes are compared with
Brandes and parallel CC-APSPN; both approximate modes are compared with the
parallel API's sequential approximation paths in fresh processes. Coverage
includes dense and SciPy sparse inputs (CSR, CSC,
COO, matrices and arrays), logical and double inputs, output layout, unchanged
inputs, rejected invalid inputs, and thread changes `1, 4, 1`. They also check
that the parallel exact path does not convert sparse adjacency to a dense
matrix. The sequential wrapper still densifies sparse adjacency. Its tests
also check exact matrix-output orientation, ablation forwarding and required
reference distances. The test configuration needs four physical cores.
Approximate modes still run sequentially.

One small smoke test also checks CM-GC and the four SE/BCC combinations
(APSPN and BC), since these release libraries share matrix-conversion helpers.

Use `--installed` to test a pip-installed package rather than the source
checkout, or `--json /path/to/results.json` to save results for cross-platform
comparison. Exact numerical results should agree within floating-point
tolerance, not necessarily bit for bit. Approximate fixed-seed results are
compared with the sequential implementation on each platform; different C++
standard-library random implementations can give different samples across
platforms.

For MATLAB, add `scripts/tests` to the path and run:

```matlab
test_release_parallel
```

Run the following approximate probes in two **fresh MATLAB processes**:

```matlab
test_release_parallel('approx', 'sequential', '/temporary/path/seq.json')
test_release_parallel('approx', 'parallel', '/temporary/path/par.json')
```

Then compare them with:

```matlab
test_release_parallel('compare', '/temporary/path/seq.json', '/temporary/path/par.json')
```

The MATLAB tests additionally verify column-vector outputs and requested vs.
effective thread counts. These tests require the release MEX files under
`source/matlab/mex_functions`. They do not compile or modify algorithms.

## Fixture provenance

`fixtures/grid_4x5.mat` (a 4×5 grid graph) and `fixtures/cycle_n11.mat` (an
11-node cycle) are small constructed test graphs. Each contains one variable,
`x`: an undirected binary sparse adjacency matrix. The directed-input check for
approximate BC orients the cycle. Reference distances are Euclidean distances
between fixed, deterministic node coordinates defined in the test scripts.
