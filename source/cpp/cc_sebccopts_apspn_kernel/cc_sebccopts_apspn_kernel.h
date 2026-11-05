//
// Created by Junda on 2024/10/21.
//

#ifndef CC_SEBCCOPTS_APSPN_KERNEL_H
#define CC_SEBCCOPTS_APSPN_KERNEL_H

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <variant>
#include <optional>
#include <string>

enum class AlgorithmType {
    APSPN,
    BC
};

struct HEURISTICS {
    bool SE = false;
    bool BCC = false;
};

struct COUNTERS {
    int64_t mem_util_count = 0;   // over all number of encountered ps nodes
    int64_t graph_trav_count = 0; // number of visited normal nodes
    int64_t cv_count = 0;       // number of cut vertices
    int64_t bc_count = 0;       // number of biconnected components
    int64_t eqc_n = 0;          // number of equivalence classes
    double apspn_mem_con = 0.0;      // limit of pse nodes

    // visited nodes components
    int64_t fwd_visit_count = 0;  // number of visited nodes in forward traversal
    int64_t bwd_visit_count = 0;  // number of visited nodes in backward traversal
    int64_t bc_bwd_visit_count = 0;  // number of visited nodes in backward traversal for BC
    int64_t bc_fwd_visit_count = 0;  // number of visited nodes in forward traversal for BC

    double org_graph_density = 0.0; // density of the original grap
    double reduced_graph_density = 0.0; // density of the reduced graph
    double biconn_sep_ratio = 0.0; // ratio of nodes seperated by biconnected components
    // time
    double durationInSecComputeAdjList = 0.0; // time to compute adjacency list
    double durationInSecFindEqc = 0.0; // time to find equivalence classes
    double durationInSecBiconnDecomp = 0.0; // time to decompose biconnected components
    double durationInSecCompLocalDeg = 0.0; // time to compute local degree
    double durationInSecPri = 0.0; // time to prioritize sources
    double durationInSecLocalSSS = 0.0; // time to perform Local Single-source Searches
    double durationInSecTotal = 0.0; // total time
    double durationInSecAPSPComputation = 0.0; // time to compute APSP
};

// Result types for different algorithms
struct Result {
    std::variant<
        Eigen::MatrixXd,     // For APSPN
        Eigen::VectorXd     // For BC
    > result;
    COUNTERS counters;
    double total_time;
};

// Parameters for different algorithms
struct AlgorithmParams {
    bool use_se{true};
    bool use_bcc{true};
    AlgorithmType algo_type{AlgorithmType::APSPN};
    bool verbose{true};
    std::optional<Eigen::MatrixXd> rd_matrix;

    // Default constructor
    AlgorithmParams() = default;

    // Constructor with RD matrix
    explicit AlgorithmParams(const Eigen::MatrixXd& rd) : rd_matrix(rd) {}

    // Constructor with verbose
    AlgorithmParams(bool use_se, bool use_bcc, bool verbose)
            : use_se(use_se), use_bcc(use_bcc), verbose(verbose) {}
};

Result CcSeBccOptionalApspnKernel(
        const Eigen::MatrixXd& adj_matrix,
        const AlgorithmParams& params);

#endif //CC_SEBCCOPTS_APSPN_KERNEL_H

