//
// Created by Junda on 2024/4/1.
//

#ifndef CPP_ALGOS_H
#define CPP_ALGOS_H

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <variant>
#include <optional>

enum class AlgorithmType {
    APSPN,
    BC,
    GC,
    PHD,
    APSLP
};

struct COUNTERS {
    // time
    double durationInSecComputeAdjList = 0.0; // time to compute adjacency list
    double durationInSecTotal = 0.0;          // total time
};

// Result types for different algorithms
struct Result {
    std::variant<
            Eigen::MatrixXd,     // For APSPN
            Eigen::VectorXd,     // For BC
            double               // For GC
    > result;
    COUNTERS counters;
    double total_time;
};

// Parameters for different algorithms
struct AlgorithmParams {
    std::optional<Eigen::MatrixXd> rd_matrix;

    // Default constructor
    AlgorithmParams() = default;

    // Constructor with RD matrix
    explicit AlgorithmParams(const Eigen::MatrixXd& rd) : rd_matrix(rd) {}
};

Result BrandesApspnKernel(const Eigen::MatrixXd &adj_matrix,
                            const AlgorithmType& algo_type,
                            const AlgorithmParams &params);

#endif //CPP_ALGOS_H
