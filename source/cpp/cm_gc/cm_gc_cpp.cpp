#include "cm_gc.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <limits>
#include <chrono>
#include <queue>
#include <unordered_map>
#include <iostream>
#include "string"
#include <cassert>
#include <numeric>
#include "../../common_include/graph_utils.h"
#include "../../common_include/timer.h"
#ifdef MATLAB_BUILD
#include "mex.h"
#include "../../common_include/mex_utils.h"
#endif

// Function to format progress with at least two digits
std::string formatProgress(int64_t progress) {
    // Convert the progress to a string
    std::string progressStr = std::to_string(progress);

    // Ensure at least two digits
    if (progressStr.length() < 2) {
        progressStr = "0" + progressStr;
    }

    return progressStr;
}


// BFS to compute unweighted shortest paths from a single source
Eigen::VectorXd bfs_sss(
    const std::vector<std::vector<int64_t>>& A,
    const int64_t n,
    const int64_t s) {
    Eigen::VectorXd distances = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
    distances(s) = 0;
    std::queue<int64_t> queue;
    queue.push(s);

    while (!queue.empty()) {
        int64_t curr = queue.front();
        queue.pop();

        for (int64_t neighbor : A[curr]) {
            if (distances(neighbor) == std::numeric_limits<double>::infinity()) {  // Not visited
                distances(neighbor) = distances(curr) + 1;
                queue.push(neighbor);
            }
        }
    }
    return distances;
}

// Recursive function to compute pTSP
void compute_pTSP_rec(int64_t N,
                      const std::vector<std::vector<int64_t>>& A,
                      const Eigen::MatrixXd& G,
                      const Eigen::MatrixXd& T,
                      const Eigen::VectorXd& L,
                      Eigen::MatrixXd& P,
                      Eigen::VectorXd& Pcount,
                      Eigen::VectorXd& inpath, int64_t s, int64_t t, int64_t lt, double lg, double g) {
    inpath(t) = 1;  // Mark node as visited
    lt += 1;        // Increment topological length
    lg += g;        // Increment geometrical length

    // Update P and Pcount if conditions are met
    if (lt == T(s, t) && (L(s) > L(t) || (L(s) == L(t) && s < t))) {
        Pcount(t) += 1;
        P(s, t) += lg;
    }

    // Continue recursion if maximum path length not reached
    if (lt < L(s)) {
        for (int64_t u : A[t]) {
            if (inpath(u) == 0) {
                compute_pTSP_rec(N, A, G, T, L, P, Pcount, inpath, s, u, lt, lg, G(t, u));
            }
        }
    }

    inpath(t) = 0;  // Mark node as unvisited
}

// The main function implementing CM_GC
Result CM_GC_Cpp(const Eigen::MatrixXd& adj_matrix,
                 const Eigen::MatrixXd& RD) {

    /// Computation starts
    Timer global_timer;
    global_timer.Start();

    int64_t n = adj_matrix.rows();
    /// compute adjacency list
    std::cout << "Computing adjacency list..." << std::endl;
    auto A = ComputeAdjList(adj_matrix, n);

    /// Compute APSP using BFS
    std::cout << "Computing APSP distances...\n";
    Eigen::MatrixXd distances(n, n);
    for (int64_t i = 0; i < n; ++i) {
        distances.row(i) = bfs_sss(A, n, i);
    }
    std::cout << "APSP distance computed." << std::endl;

    /// Compute the mean of the topological shortest paths
    std::cout << "Computing mean of topological shortest paths...\n";
    Eigen::VectorXd Tmean = Eigen::VectorXd::Zero(n);
    for (int64_t s = 0; s < n; ++s) {
        for (int64_t t = s + 1; t < n; ++t) {
            Tmean(s) += distances(s, t);
            Tmean(t) += distances(s, t);
        }
        Tmean(s) /= (n - 1);
    }

    // Sort nodes by decreasing mean of the topological shortest paths
    std::vector<int64_t> order(n);
    std::iota(order.begin(), order.end(), 0);  // Fill order with 0, 1, ..., N-1
    std::sort(order.begin(), order.end(), [&Tmean](int64_t i, int64_t j) { return Tmean(i) > Tmean(j); });

    // Compute L, which indicates the maximum path length to evaluate for each node
    std::cout << "Computing maximum path lengths..." << std::endl;
    Eigen::VectorXd L = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd mask = Eigen::VectorXd::Zero(n);
    for (int64_t i = 0; i < n; ++i) {
        int64_t s = order[i];
        for (int64_t t = 0; t < n; ++t) {
            if (mask(t) == 0 && distances(s, t) > L(s)) {
                L(s) = distances(s, t);
            }
        }
        mask(s) = 1;
    }
    //
    std::cout << "overall recursion depths: " << L.sum() << std::endl;

    /// Main computation of P matrix
    std::cout << "Computing P matrix..." << std::endl;
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(n, n);
    for (int64_t i = 0; i < n; ++i) {
        // Display progress
        int64_t progress = (i * 100.0) / n;
        std::cout << "\b\b\b\b\b\b\b\b\b\b\b\bProgress:" << formatProgress(progress) << "%" << std::flush;

        int64_t lt = 0;
        int64_t lg = 0;
        Eigen::VectorXd Pcount = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd inpath = Eigen::VectorXd::Zero(n);
        int64_t s = order[i];
        inpath[s] = 1;

        // Start recursive computation for each neighbor of s
        for (int64_t t : A[s]) {
            compute_pTSP_rec(n, A, RD, distances, L, P, Pcount, inpath, s, t, lt, lg, RD(s, t));
        }

        // Update P matrix
        for (int64_t t = 0; t < n; ++t) {
            if (L(s) > L(t) || (L(s) == L(t) && s < t)) {
                P(s, t) = P(s, t) / Pcount(t);
                P(t, s) = P(s, t);  // Symmetric
            }
        }


    }

    // Compute geometrical congruence
    std::cout << "Computing geometrical congruence..." << std::endl;
    Eigen::MatrixXd RD_ = RD.array() * (adj_matrix.array()==0).cast<double>();
    RD_.diagonal().setZero();
    P.diagonal().setOnes();
    double num_edges = adj_matrix.sum();
    double nonadj_pairs = n * (n-1) - num_edges;
    double GC = (RD_.array() / P.array()).sum() / nonadj_pairs;

    global_timer.Stop();
    double durationInSeconds = global_timer.Seconds();

    return Result{GC, durationInSeconds};
}

// ============================================================================
// Python-callable interface (for ctypes)
// ============================================================================
#ifdef PYTHON_BUILD

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

/**
 * @brief Python-callable wrapper for CM-GC algorithm
 * 
 * @param adj_matrix_data Flat array of adjacency matrix (row-major, n*n elements)
 * @param n Number of nodes
 * @param rd_matrix_data Flat array of referential distance matrix (row-major, n*n elements)
 * @param output_gc Output parameter for GC value
 * @param elapsed_time Output parameter for computation time
 * @return 0 on success, -1 on error
 */
EXPORT int cm_gc_py(
    const double* adj_matrix_data,
    int64_t n,
    const double* rd_matrix_data,
    double* output_gc,
    double* elapsed_time)
{
    try {
        // Convert input arrays to Eigen matrices
        Eigen::Map<const Eigen::MatrixXd> adj_matrix_map(adj_matrix_data, n, n);
        Eigen::MatrixXd adj_matrix = adj_matrix_map;
        
        Eigen::Map<const Eigen::MatrixXd> rd_matrix_map(rd_matrix_data, n, n);
        Eigen::MatrixXd rd_matrix = rd_matrix_map;
        
        // Call the algorithm
        Result result = CM_GC_Cpp(adj_matrix, rd_matrix);
        
        // Copy output
        *output_gc = result.gc;
        *elapsed_time = result.total_time;
        
        return 0;  // Success
    }
    catch (...) {
        return -1;  // Error
    }
}

#ifdef __cplusplus
}
#endif
#endif // PYTHON_BUILD


// ============================================================================
// MATLAB MEX Interface
// ============================================================================

#ifdef MATLAB_BUILD
/**
 * @brief MEX interface function for the CM-GC algorithm
 *
 * @param nlhs Number of output arguments (left-hand side)
 * @param plhs Array of output arguments
 * @param nrhs Number of input arguments (right-hand side)
 * @param prhs Array of input arguments
 *
 * INPUT PARAMETERS:
 * prhs[0] - Adjacency matrix (double or logical matrix):
 *           Square matrix representing the graph structure (n×n).
 *           Can be sparse or dense format. Nonzero entries are treated as
 *           unweighted edges. The graph must be undirected, so the matrix
 *           should be symmetric.
 *
 * prhs[1] - RD matrix (double matrix):
 *           Reference distance matrix with same dimensions as adjacency matrix.
 *           Represents geometric distances between node pairs.
 *           Required for GC computation.
 *
 * OUTPUT PARAMETERS:
 * plhs[0] - GC value (double scalar):
 *           The computed geometrical congruence value for the network.
 *
 * plhs[1] - (Optional) Computation time (double scalar):
 *           Total execution time in seconds.
 *
 * USAGE EXAMPLES:
 * Basic GC computation:
 *   gc = cm_gc_cpp(adj_matrix, rd_matrix);
 *
 * With timing information:
 *   [gc, time] = cm_gc_cpp(adj_matrix, rd_matrix);
 *
 * @author Junda
 * @date 2024/4/1
 * @version V1
 */
// Entry point for the MEX-file
void mexFunction(int64_t nlhs, mxArray *plhs[], int64_t nrhs, const mxArray *prhs[]) {
    // Check the number of input arguments.
    if (nrhs < 2) {
        mexErrMsgIdAndTxt("mexFunction:invalidNumInputs", "At least two inputs required: adjacency matrix and RD matrix.");
    }

    // Parse the input adjacency matrix
    Eigen::Index numVertices = mxGetN(prhs[0]);
    // Parse the adjacency matrix
    Eigen::MatrixXd adjMatrix = parseSparseDenseMatrix(prhs[0]);
    Eigen::MatrixXd rdMatrix = parseSparseDenseMatrix(prhs[1]);
    if (adjMatrix.rows() <= 1) {
        mexErrMsgIdAndTxt("mexFunction:invalidGraphSize",
                          "Input graph must contain at least 2 nodes.");
    }

    // Call the algorithm function
    Result result = CM_GC_Cpp(adjMatrix, rdMatrix);

    // Allocate memory for the output
    plhs[0] = mxCreateDoubleScalar(result.gc);
    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleScalar(result.total_time);
    }
}

#endif // MATLAB_BUILD
