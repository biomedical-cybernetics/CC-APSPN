# CC-BFS-P

CC-BFS-P is the parallel (OpenMP) version of CC-BFS, the authors'
network-science-informed algorithm for all-pairs shortest path (APSP) distances
in unweighted, undirected graphs. The parallel CC-APSPN kernel uses it to
compute the distance matrix in its first step. The sequential kernels use CC-BFS
(`third_party/cc_bfs/`).

The files in `cpp/` are included unchanged from the authors' CC-BFS-P
implementation.

## How it works

CC-BFS first processes a small set of influential hub nodes and then reuses
their shortest-path structures to prune the breadth-first searches from all
other nodes. CC-BFS-P keeps this search logic and parallelizes it in two phases:

1. **Hubs.** All hub searches run in parallel.
2. **Other sources.** The remaining sources are processed in batches of at most
   `4 × num_threads`, scheduled dynamically within each batch. Every source in
   a batch reuses only results published before the batch started. When the
   batch finishes, its distance columns are published together in source
   order, so which results a search can reuse does not depend on thread
   timing.

Each thread keeps its own search buffers for the whole run. This avoids
repeated allocation at the cost of O(n) extra memory per thread.

## Use in CC-APSPN

CC-APSPN embeds CC-BFS-P through `source/common_include/cc_bfs_p_apsp_l.cpp`,
which includes `cpp/cc_bfs_p.cpp`. Do not compile `cpp/cc_bfs_p.cpp` separately.
Add `cpp/include/` to the include path, as the MATLAB and Python build scripts
do.

`cpp/cc_bfs_p.cpp` also contains a standalone MATLAB entry point,
`[D, t, counters] = cc_bfs_p(A, num_threads, ...)`, which this release does not
build.
