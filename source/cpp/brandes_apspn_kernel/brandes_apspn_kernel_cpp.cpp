//
// Created by Junda on 2024/9/17.
//
#include "brandes_apspn_kernel.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <limits>
#include <queue>
#include <map>
#include "string"
#include <optional>
#include <variant>
#include <stack>
#include "iostream"
#include "../../common_include/graph_utils.h"
#include "../../common_include/timer.h"
#ifdef MATLAB_BUILD
#include "mex.h"
#include "../../common_include/mex_utils.h"
#endif


#ifdef MATLAB_BUILD
mxArray* CreateCountersStruct(const COUNTERS& counters) {
    // Create field names array
    const char* fieldnames[] = {
            "durationInSecComputeAdjList",
            "durationInSecTotal"
    };
    constexpr size_t numFields = sizeof(fieldnames) / sizeof(fieldnames[0]);

    // Create the struct
    mxArray* matlabStruct = mxCreateStructMatrix(1, 1, numFields, fieldnames);

    // Set each field manually
    mxSetFieldByNumber(matlabStruct, 0, 0, mxCreateDoubleScalar(counters.durationInSecComputeAdjList));
    mxSetFieldByNumber(matlabStruct, 0, 1, mxCreateDoubleScalar(counters.durationInSecTotal));

    return matlabStruct;
}
#endif // MATLAB_BUILD

// BFS algorithm to find the shortest path from a single source to all other vertices in the graph
void BfsSss(
                const std::vector<std::vector<int64_t>>& A,
                const int64_t n,
                const int64_t s,
                Eigen::MatrixXd &D,
                Eigen::MatrixXd &Pcount,
                const AlgorithmType& algo_type,
                std::optional<Eigen::MatrixXd> &P,
                std::optional<Eigen::VectorXd> &BC,
                std::optional<Eigen::MatrixXd> &APSLP,
                const std::optional<Eigen::MatrixXd> &RD) {
    // initialize intermediate variables
    std::queue<int64_t> Q;
    Q.push(s);
    auto D_s = D.col(s);
    D_s(s) = 0;
    auto Pcount_s = Pcount.col(s);
    Pcount_s(s) = 1;
    // initialize optional variables
    std::optional<std::vector<std::vector<int64_t>>> Preds;
    std::optional<std::stack<int64_t>> S;
    std::optional<Eigen::Ref<Eigen::VectorXd>> SSSLP;  // single-source shortest-lightest path

    switch (algo_type) {
    case AlgorithmType::BC:
        assert(BC.has_value());
        Preds = std::vector<std::vector<int64_t>>(n);
        S = std::stack<int64_t>();
        break;
    case AlgorithmType::GC:
        assert(P.has_value());
        assert(RD.has_value());
        break;
    case AlgorithmType::APSLP:
        assert(APSLP.has_value());
        assert(RD.has_value());
        SSSLP = (*APSLP).col(s);
        (*SSSLP)(s) = 0;
        break;
    }

    while (!Q.empty()) {
        int64_t curr = Q.front();
        Q.pop();
        switch (algo_type) {
        case AlgorithmType::BC:
            (*S).push(curr);
            break;
        }

        double alt_dst = D_s(curr) + 1;
        for (int64_t t: A[curr]) {
            if (alt_dst < D_s(t)) {
                Q.push(t);
                D_s(t) = alt_dst;
            }
            if (alt_dst == D_s[t]) {  // found a new shortest path to t
                Pcount_s[t] +=  Pcount_s[curr];
                switch (algo_type) {
                case AlgorithmType::GC:
                    (*P)(s,t) += (*P)(s, curr) + (*RD)(curr, t) * Pcount_s(curr);
                    break;
                case AlgorithmType::BC:
                    (*Preds)[t].push_back(curr);
                    break;
                case AlgorithmType::APSLP:
                    double alt_slp = (*SSSLP)(curr) + (*RD)(curr, t); // alternative shortest-lightest path
                    if ((*SSSLP)(t) > alt_slp) {
                        (*SSSLP)(t) = alt_slp;
                    }
                }
            }
        }
    }
    switch (algo_type) {
    case AlgorithmType::BC:  // Backward dependency propagation
        {
            Eigen::VectorXd BC_s = Eigen::VectorXd::Zero(n);
            while (!(*S).empty())
            {
                int64_t curr = (*S).top();
                (*S).pop();
                for (int64_t v : (*Preds)[curr])
                {
                    BC_s(v) += (Pcount_s(v) / Pcount_s(curr)) * (1 + BC_s(curr));
                }
                if (curr != s)
                {
                    (*BC)(curr) += BC_s(curr);
                }
            }
            break;
        }
    }
}


// The actual algorithm implementation
Result BrandesApspnKernel(const Eigen::MatrixXd& adj_matrix,
                            const AlgorithmType& algo_type,
                            const AlgorithmParams& params) {

    /// computation starts
    Timer global_timer;
    Timer local_timer;
    global_timer.Start();

    int64_t n = adj_matrix.rows();
    COUNTERS counters;

    /// compute adjacency list
    local_timer.Start();

    auto A = ComputeAdjList(adj_matrix, n);

    local_timer.Stop();
    counters.durationInSecComputeAdjList = local_timer.Seconds();
    std::cout << "@@ Adjacency list computation costs: " << local_timer.Seconds() << " seconds" << std::endl;


    /// initialize variables
    Eigen::MatrixXd D = Eigen::MatrixXd::Constant(n, n, std::numeric_limits<double>::infinity());
    Eigen::MatrixXd Pcount = Eigen::MatrixXd::Constant(n, n, 0);
    std::optional<Eigen::MatrixXd> RD;
    std::optional<Eigen::MatrixXd> P;
    std::optional<Eigen::VectorXd> BC;
    std::optional<Eigen::VectorXd> PHD;
    std::optional<Eigen::MatrixXd> APSLP;
    std::optional<double> GC;


    // Initialize variables based on algorithm type
    switch (algo_type) {
        case AlgorithmType::GC:
            // parse RD matrix
            if (!params.rd_matrix.has_value()) {
                throw std::runtime_error("GC algorithm requires RD matrix.");
            }
            RD = params.rd_matrix;
            P = Eigen::MatrixXd::Constant(n, n, 0);
            break;
        case AlgorithmType::BC:
            BC = Eigen::VectorXd::Zero(n);
            break;
        case AlgorithmType::PHD:
            PHD = Eigen::VectorXd::Zero(n);
            break;
    case AlgorithmType::APSLP:
            // parse RD matrix
            if (!params.rd_matrix.has_value()) {
                throw std::runtime_error("APSLP algorithm requires RD matrix.");
            }
            RD = params.rd_matrix;
            APSLP = Eigen::MatrixXd::Constant(n, n, INF);
            break;
    }

    // APSPN main loop
    for (int64_t idx = 0; idx < n; ++idx) {
        int64_t s = idx;
        BfsSss(A, n, s, D, Pcount, algo_type, P, BC, APSLP, RD);
    }

    // post processing
    switch (algo_type) {
    case AlgorithmType::BC:
        if (adj_matrix.isApprox(adj_matrix.transpose())) {
            (*BC) /= 2;
        }
        break;
    case AlgorithmType::GC:
        {
            GC = 0;
            (*P) = (*P).array() / Pcount.array();
            // Mask the elements in RD where x is true
            Eigen::MatrixXd RD_ = (*RD).array() * (adj_matrix.array() == 0).cast<double>();
            RD_.diagonal().setZero();
            (*P).diagonal().setOnes();
            double num_edges = adj_matrix.sum();
            double num_nonadj_pairs = n * (n - 1) - num_edges;
            GC = (RD_.array() / (*P).array()).sum() / num_nonadj_pairs;
            break;
        }
    case AlgorithmType::PHD:
        // compute the average of Pcount for each node
        for (int64_t i = 0; i < n; ++i) {
            (*PHD)(i) = (Pcount.col(i).sum()-1)/ (n-1); // exclude self-loop
        }
    }

    // computation ends
    global_timer.Stop();
    counters.durationInSecTotal = global_timer.Seconds();

    // Return the result
    switch (algo_type) {
    case AlgorithmType::APSPN:
        return Result{Pcount,counters, global_timer.Seconds()};
    case AlgorithmType::BC:
        return Result{BC.value(), counters,global_timer.Seconds()};
    case AlgorithmType::GC:
        return Result{GC.value(), counters,global_timer.Seconds()};
    case AlgorithmType::PHD:
        return Result{PHD.value(), counters,global_timer.Seconds()};
    case AlgorithmType::APSLP:
        return Result{APSLP.value(), counters, global_timer.Seconds()};
    default:
        throw std::runtime_error("Unknown algorithm type.");
    }
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
 * @brief Unified Python-callable wrapper for Brandes-APSPN-kernel algorithms
 * 
 * This function provides a single entry point for all algorithm types, similar to the MEX interface.
 * 
 * @param algo_type_str Algorithm type string: "APSPN", "BC", "GC", "PHD", or "APSLP" (case-insensitive)
 * @param adj_matrix_data Flat array of adjacency matrix (row-major, n*n elements)
 * @param n Number of nodes
 * @param rd_matrix_data Flat array of referential distance matrix (row-major, n*n elements), NULL if not needed
 * @param output_data Output array allocated by caller:
 *                    - APSPN/APSLP: n*n array for matrix result
 *                    - BC/PHD: n array for vector result
 *                    - GC: single scalar (1 element array)
 * @param elapsed_time Output parameter for computation time
 * @return 0 on success, negative error code on failure:
 *         -1: Unknown error
 *         -2: Invalid algorithm type
 *         -3: Missing RD matrix (required for GC/APSLP)
 */
EXPORT int brandes_apspn_kernel_py(
    const char* algo_type_str,
    const double* adj_matrix_data,
    int64_t n,
    const double* rd_matrix_data,
    double* output_data,
    double* elapsed_time)
{
    try {
        // Parse algorithm type
        std::string algo_str(algo_type_str);
        std::transform(algo_str.begin(), algo_str.end(), algo_str.begin(), ::toupper);
        
        AlgorithmType algo_type;
        if (algo_str == "APSPN") {
            algo_type = AlgorithmType::APSPN;
        } else if (algo_str == "BC") {
            algo_type = AlgorithmType::BC;
        } else if (algo_str == "GC") {
            algo_type = AlgorithmType::GC;
        } else if (algo_str == "PHD") {
            algo_type = AlgorithmType::PHD;
        } else if (algo_str == "APSLP") {
            algo_type = AlgorithmType::APSLP;
        } else {
            return -2;  // Invalid algorithm type
        }
        
        // Convert adjacency matrix
        Eigen::Map<const Eigen::MatrixXd> adj_matrix_map(adj_matrix_data, n, n);
        Eigen::MatrixXd adj_matrix = adj_matrix_map;
        
        // Prepare algorithm parameters
        AlgorithmParams params;
        if (algo_type == AlgorithmType::GC || algo_type == AlgorithmType::APSLP) {
            if (rd_matrix_data == nullptr) {
                return -3;  // Missing RD matrix
            }
            Eigen::Map<const Eigen::MatrixXd> rd_matrix_map(rd_matrix_data, n, n);
            Eigen::MatrixXd rd_matrix = rd_matrix_map;
            params = AlgorithmParams(rd_matrix);
        }
        
        // Call the algorithm
        Result result = BrandesApspnKernel(adj_matrix, algo_type, params);
        
        // Copy output based on algorithm type
        switch (algo_type) {
            case AlgorithmType::APSPN:
            case AlgorithmType::APSLP: {
                const auto& matrix_result = std::get<Eigen::MatrixXd>(result.result);
                Eigen::Map<Eigen::MatrixXd>(output_data, n, n) = matrix_result;
                break;
            }
            case AlgorithmType::BC:
            case AlgorithmType::PHD: {
                const auto& vector_result = std::get<Eigen::VectorXd>(result.result);
                Eigen::Map<Eigen::VectorXd>(output_data, n) = vector_result;
                break;
            }
            case AlgorithmType::GC: {
                *output_data = std::get<double>(result.result);
                break;
            }
        }
        
        // Set elapsed time
        *elapsed_time = result.total_time;
        
        return 0;  // Success
    }
    catch (const std::bad_variant_access&) {
        return -1;  // Type mismatch error
    }
    catch (...) {
        return -1;  // Unknown error
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
// Helper functions for MEX interface
template<typename EnumType>
EnumType parseEnumFromString(const char* str, const std::map<std::string, EnumType>& enumMap, const std::string& errorMsg) {
    std::string upperStr = str;
    std::transform(upperStr.begin(), upperStr.end(), upperStr.begin(), ::toupper);
    
    auto it = enumMap.find(upperStr);
    if (it != enumMap.end()) {
        return it->second;
    }
    
    mexErrMsgIdAndTxt("MyToolbox:unified:unknownType", errorMsg.c_str());
    return enumMap.begin()->second; // This line never executes due to mexErrMsgIdAndTxt
}

AlgorithmType parseAlgorithmType(const mxArray* prhs) {
    if (!mxIsChar(prhs)) {
        mexErrMsgIdAndTxt("MyToolbox:unified:invalidInput", "Algorithm type must be a string.");
    }
    
    char* algoTypeStr = mxArrayToString(prhs);
    static const std::map<std::string, AlgorithmType> algoMap = {
        {"APSPN", AlgorithmType::APSPN},
        {"BC", AlgorithmType::BC},
        {"GC", AlgorithmType::GC},
        {"PHD", AlgorithmType::PHD},
        {"APSLP", AlgorithmType::APSLP}
    };
    
    AlgorithmType result = parseEnumFromString(algoTypeStr, algoMap, 
        "Unknown algorithm type. Supported types: APSPN, BC, GC, PHD, APSLP");
    mxFree(algoTypeStr);
    return result;
}

struct ParsedInputs {
    AlgorithmType algo_type;
    Eigen::MatrixXd adjMatrix;
    AlgorithmParams params;
};

ParsedInputs parseInputArguments(int nrhs, const mxArray *prhs[]) {
    // Check minimum number of arguments
    if (nrhs < 2) {
        mexErrMsgIdAndTxt("MyToolbox:unified:nrhs",
                          "At least two inputs required (algorithm type and adjacency matrix).");
    }
    
    ParsedInputs inputs;
    
    // Parse algorithm type
    inputs.algo_type = parseAlgorithmType(prhs[0]);
    
    // Parse adjacency matrix
    inputs.adjMatrix = parseSparseDenseMatrix(prhs[1]);
    Eigen::Index numVertices = inputs.adjMatrix.rows();
    if (numVertices <= 1) {
        mexErrMsgIdAndTxt("MyToolbox:unified:invalidGraphSize",
                          "Input graph must contain at least 2 nodes.");
    }
    
    // Parse optional RD matrix for GC/APSLP algorithms
    if (inputs.algo_type == AlgorithmType::GC || inputs.algo_type == AlgorithmType::APSLP) {
        if (nrhs < 3) {
            mexErrMsgIdAndTxt("MyToolbox:unified:nrhs",
                              "GC and APSLP algorithms require RD matrix.");
        }
        Eigen::MatrixXd rdMatrix = parseSparseDenseMatrix(prhs[2]);
        if (rdMatrix.rows() != numVertices || rdMatrix.cols() != numVertices) {
            mexErrMsgIdAndTxt("MyToolbox:unified:dimensionMismatch",
                              "RD matrix dimensions must match adjacency matrix.");
        }
        inputs.params = AlgorithmParams(rdMatrix);
    }
    
    return inputs;
}

void createOutputArguments(int nlhs, mxArray *plhs[], const Result& result, 
                          AlgorithmType algo_type, Eigen::Index numVertices) {
    // Create main output based on algorithm type
    switch(algo_type) {
        case AlgorithmType::GC:
            try {
                double gc_result = std::get<double>(result.result);
                plhs[0] = mxCreateDoubleScalar(gc_result);
            } catch (const std::bad_variant_access&) {
                mexErrMsgIdAndTxt("MyToolbox:unified:unexpectedType",
                                  "GC algorithm should return a double value.");
            }
            break;

        case AlgorithmType::APSPN:
            try {
                const auto& matrix_result = std::get<Eigen::MatrixXd>(result.result);
                plhs[0] = mxCreateDoubleMatrix(numVertices, numVertices, mxREAL);
                Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), numVertices, numVertices) = matrix_result;
            } catch (const std::bad_variant_access&) {
                mexErrMsgIdAndTxt("MyToolbox:unified:unexpectedType",
                                  "APSPN algorithm should return a matrix.");
            }
            break;

        case AlgorithmType::BC:
        case AlgorithmType::PHD:
            try {
                const auto& vector_result = std::get<Eigen::VectorXd>(result.result);
                plhs[0] = mxCreateDoubleMatrix(numVertices, 1, mxREAL);
                Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[0]), numVertices) = vector_result;
            } catch (const std::bad_variant_access&) {
                mexErrMsgIdAndTxt("MyToolbox:unified:unexpectedType",
                                  "BC/PHD algorithm should return a vector.");
            }
            break;
            
        case AlgorithmType::APSLP:
            try {
                const auto& matrix_result = std::get<Eigen::MatrixXd>(result.result);
                plhs[0] = mxCreateDoubleMatrix(numVertices, numVertices, mxREAL);
                Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), numVertices, numVertices) = matrix_result;
            } catch (const std::bad_variant_access&) {
                mexErrMsgIdAndTxt("MyToolbox:unified:unexpectedType",
                                  "APSLP algorithm should return a matrix.");
            }
            break;
            
        default:
            mexErrMsgIdAndTxt("MyToolbox:unified:unknownAlgorithm", "Unknown algorithm type.");
    }

    // Optional outputs
    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleScalar(result.total_time);
    }

    if (nlhs > 2) {
        plhs[2] = CreateCountersStruct(result.counters);
    }
}

/**
 * @brief MEX interface function for the Brandes-APSPN-kernel algorithm framework
 *
 * This function provides a unified interface to the Brandes-APSPN-kernel graph algorithms
 *
 * @param nlhs Number of output arguments (left-hand side)
 * @param plhs Array of output arguments  
 * @param nrhs Number of input arguments (right-hand side)
 * @param prhs Array of input arguments
 *
 * INPUT PARAMETERS:
 * prhs[0] - Algorithm type (string):
 *           - 'APSPN': All-Pairs Shortest Path Number 
 *           - 'BC': Betweenness Centrality
 *           - 'GC': Network Geometrical Congruence
 *           - 'PHD': Path Hesitation Degree 
 *           - 'APSLP': All-Pairs Shortest-Lightest Paths
 *
 * prhs[1] - Adjacency matrix (double or logical matrix):
 *           Square matrix representing the graph structure (n×n).
 *           Can be sparse or dense format. Nonzero entries are treated as
 *           unweighted edges. Directed or undirected graphs are supported;
 *           use a symmetric matrix for undirected graphs.
 *
 * prhs[2] - (Optional) RD matrix (double matrix):
 *           Required for GC and APSLP algorithms only.
 *           Referential distance matrix with same dimensions as adjacency matrix.
 *           For GC: represents geometric distances between non-adjacent node pairs.
 *           For APSLP: represents edge weights for shortest-lightest path computation.
 *
 * OUTPUT PARAMETERS:
 * plhs[0] - Algorithm result:
 *           - APSPN: n×n double matrix of shortest path counts
 *           - BC: n×1 double vector of betweenness centrality values
 *           - GC: scalar double value of geometrical congruence
 *           - PHD: n×1 double vector of path hesitation degrees  
 *           - APSLP: n×n double matrix of shortest-lightest path weights
 *
 * plhs[1] - (Optional) Computation time (double scalar):
 *           Total execution time in seconds
 *
 * plhs[2] - (Optional) Performance counters (struct):
 *           Performance statistics including:
 *           - durationInSecComputeAdjList: Adjacency list computation time
 *           - durationInSecTotal: Total computation time
 *
 * @author Junda
 * @date 2024/9/17  
 * @version V1
 */
// Entry point for the MEX-file
void mexFunction(int64_t nlhs, mxArray *plhs[], int64_t nrhs, const mxArray *prhs[]) {
    try {
        // Parse all input arguments
        ParsedInputs inputs = parseInputArguments(nrhs, prhs);
        
        // Run the algorithm
        auto result = BrandesApspnKernel(inputs.adjMatrix, inputs.algo_type, inputs.params);
        
        // Create output arguments
        createOutputArguments(nlhs, plhs, result, inputs.algo_type, inputs.adjMatrix.rows());
        
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("MyToolbox:unified:exception", e.what());
    }
}

#endif // MATLAB_BUILD
