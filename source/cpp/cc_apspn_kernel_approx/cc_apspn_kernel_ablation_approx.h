//
// Created by Junda on 2024/4/4.
//

#ifndef CC_APSPN_KERNEL_ABLATION_APPROX_H
#define CC_APSPN_KERNEL_ABLATION_APPROX_H

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <variant>
#include <optional>
#include <string>
#include <iostream>
#include <iomanip>

// Forward declaration for MATLAB types
#ifdef MATLAB_BUILD
#include "mex.h"
#else
typedef struct mxArray_tag mxArray;
#endif

enum class AlgorithmType {
    APSPN,
    BC,
    GC,
    GC_approx,
    BC_approx,
    PHD,
    APSLP
};

enum class AblationType {
    Pri,
    Mem,
    Symm,
    DAP,
    None
};

struct COUNTERS {
    int64_t mem_util_count = 0;   // overall number of utilized ps nodes
    int64_t graph_trav_count = 0; // number of nodes performing graph traversal
    double apspn_mem_con = 0.0;      // apspn memoization constraint

    // visited nodes components
    int64_t fwd_visit_count = 0;  // number of visited nodes in forward traversal
    int64_t bwd_visit_count = 0;  // number of visited nodes in backward traversal
    int64_t bc_bwd_visit_count = 0;  // number of visited nodes in backward traversal for BC
    int64_t bc_fwd_visit_count = 0;  // number of visited nodes in forward traversal for BC

    // times
    double durationInSecPri = 0.0; // time to compute IT and priority
    double durationInSecComputeAdjList = 0.0; // time to compute adjacency list
    // time for the APSP stage
    double durationInSecAPSPComputation = 0.0; // time to compute APSP
    // time for the APSPN stage
    double durationInSecAPSPNInit = 0.0; // time to initialize
    double durationInSecAPSPNLevelComputation = 0.0; // time to compute levels
    double durationInSecAPSPNLevelTraversal = 0.0; // time to traverse levels
    double durationInSecAPSPNComputation = 0.0;   // overall time to compute APSPN (given APSP and x as inputs)
    // total
    double durationInSecTotal = 0.0; // total time

    // Memory performance statistics (for approximate GC)
    double durationInSecCcBbBFS = 0.0; // time in CcBbBFS
    double durationInSecMemReplay = 0.0; // time in memory replay
    double durationInSecMemPopulation = 0.0; // time in memory population
    double durationInSecBbBFS = 0.0; // time in bbBFS
    int64_t nodes_mem_replayed = 0; // nodes handled by memory replay
    int64_t nodes_bfs = 0; // nodes handled by actual bbBFS
};

// Result types for different algorithms
struct Result {
    std::variant<
        Eigen::MatrixXd,     // For APSPN
        Eigen::VectorXd,     // For BC
        double               // For GC
    > result;
    COUNTERS counters;
    double total_time;  // In seconds
};

// Parameters for different algorithms
    struct AlgorithmParams {
        bool verbose{true};
        std::optional<Eigen::MatrixXd> rd_matrix;
    double sampling_factor{1.0};  // Default sampling factor for approximate GC (num_samples = n * sampling_factor)

        // SILVAN approximate BC parameters
        double delta{0.05};  // Confidence parameter for BC_approx
        double err{0.01};    // Error parameter for BC_approx
        double emp_peeling_a{2.0};  // Empirical peeling parameter for BC_approx
    std::string memoization_depth{"log_n_w"};  // Memoization depth strategy for BC_approx
        uint32_t seed{42};   // Random seed for BC_approx

        // Default constructor
        AlgorithmParams() = default;

        // Constructor with RD matrix
        explicit AlgorithmParams(const Eigen::MatrixXd& rd) : rd_matrix(rd) {}

        // Constructor with RD matrix and sampling factor
        AlgorithmParams(const Eigen::MatrixXd& rd, double factor) : rd_matrix(rd), sampling_factor(factor) {}
};

inline std::string AlgorithmTypeDisplayName(AlgorithmType algo_type) {
    switch (algo_type) {
        case AlgorithmType::GC_approx:
            return "Approximate Geometrical Congruence (GC_approx)";
        case AlgorithmType::BC_approx:
            return "Approximate Betweenness Centrality (BC_approx)";
        case AlgorithmType::GC:
            return "Geometrical Congruence (GC)";
        case AlgorithmType::BC:
            return "Betweenness Centrality (BC)";
        case AlgorithmType::APSPN:
            return "APSPN";
        case AlgorithmType::PHD:
            return "PHD";
        case AlgorithmType::APSLP:
            return "APSLP";
        default:
            return "Unknown";
    }
}

inline void PrintAlgorithmHeader(AlgorithmType algo_type, const AlgorithmParams& params) {
    if (!params.verbose) {
        return;
    }
    std::cout << "========================================" << std::endl;
    std::cout << "Problem mode: " << AlgorithmTypeDisplayName(algo_type) << std::endl;
    if (algo_type == AlgorithmType::GC_approx) {
        auto flags = std::cout.flags();
        auto precision = std::cout.precision();
        std::cout << "Sampling factor: " << std::fixed << std::setprecision(2)
                  << params.sampling_factor << std::endl;
        std::cout.flags(flags);
        std::cout.precision(precision);
    } else if (algo_type == AlgorithmType::BC_approx) {
        auto flags = std::cout.flags();
        auto precision = std::cout.precision();
        std::cout << "delta: " << std::fixed << std::setprecision(4) << params.delta << std::endl;
        std::cout << "err: " << std::fixed << std::setprecision(4) << params.err << std::endl;
        std::cout << "emp_peeling_a: " << std::fixed << std::setprecision(2) << params.emp_peeling_a << std::endl;
        std::cout << "memoization_depth: " << params.memoization_depth << std::endl;
        std::cout << "seed: " << params.seed << std::endl;
        std::cout.flags(flags);
        std::cout.precision(precision);
    }
    std::cout << "========================================" << std::endl;
}

inline void PrintTotalComputationTime(double total_seconds, bool verbose) {
    if (!verbose) {
        return;
    }
    auto flags = std::cout.flags();
    auto precision = std::cout.precision();
    std::cout << "====================================" << std::endl;
    std::cout << "@@ Total computation time: " << std::fixed << std::setprecision(2)
              << total_seconds << " seconds" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout.flags(flags);
    std::cout.precision(precision);
}

Result CcApspnKernelExact(
        const Eigen::MatrixXd& adj_matrix,
        const AlgorithmType& algo_type,
        const AblationType& ablation_type,
        const AlgorithmParams& params,
        const std::string& pri_criteria);

Result CcApproxGc(
        const Eigen::MatrixXd& adj_matrix,
        const AlgorithmParams& params);

Result CcApproxBc(
        const std::variant<const mxArray*, const Eigen::MatrixXd*>& adj_input,
        const AlgorithmParams& params);

#endif //CC_APSPN_KERNEL_ABLATION_APPROX_H
