//
// Created by Junda on 2025/6/30.
//

#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <vector>
#include <unordered_map>
#include <queue>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <algorithm>
#include <utility>
#include <Eigen/Dense>

#ifdef MATLAB_BUILD
#include "mex.h"
#endif


// define infinity value
double constexpr INF = std::numeric_limits<double>::infinity();

struct BiconnectedComponents {
    std::unordered_map<int64_t, std::vector<int64_t>> block2nodes;
    std::vector<int64_t> cutVertices;
    std::vector<std::vector<int64_t>> bcTreeAdj;
    std::vector<std::vector<bool>> blockToNode;
};



BiconnectedComponents find_biconnected_components_tree(const std::vector<std::vector<int64_t>>& A);

std::tuple<std::vector<std::vector<int64_t>>, std::vector<int64_t>>
find_structural_equivalence(const std::vector<std::vector<int64_t>>& A);

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
compute_eqc_comp_tree_weights(
        const std::unordered_map<int64_t, std::vector<int64_t>>& eqc_block2nodes,
        const std::vector<std::vector<int64_t>>& eqc_bcTreeAdj,
        const int64_t eqc_n,
        const std::vector<int64_t>& eqc_node_mul,
        const int64_t n);

void compute_comm_intensity_matrix(
        const std::vector<std::vector<int64_t>>& A,
        const std::vector<std::vector<int64_t>>& equivalence_classes,
        const std::vector<std::vector<int64_t>>& eqc_A,
        int64_t eqc_B_id,
        const std::vector<int64_t>& eqc_block,
        const Eigen::MatrixXd& eqc_D_l,
        Eigen::MatrixXd& h,
        const std::vector<bool>& eqc_cv_mask,
        const std::vector<int64_t>& eqc_node_mul);

int64_t compute_eqc_block_m(const std::vector<int64_t>& eqc_block,
                            const std::vector<std::vector<int64_t>>& eqc_A,
                            const std::vector<bool>& eqc_cv_mask,
                            const std::vector<bool>& eqc_block_mask);

void compute_inter_block_Pcount(
        const std::unordered_map<int64_t, std::vector<int64_t>>& eqc_block2nodes,
        const std::vector<std::vector<int64_t>>& eqc_bcTreeAdj,
        const std::vector<int64_t>& eqc_node_mul,
        Eigen::MatrixXd& eqc_Pcount,
        int64_t offset,
        int64_t eqc_n);

std::pair<Eigen::VectorXd, std::vector<std::vector<int64_t>>> ComputeDegreeAndAdjList(
    const Eigen::MatrixXd& adj_matrix,
    const int64_t n);

#ifdef MATLAB_BUILD
namespace graph_utils {

struct ParsedGraphInput {
    std::vector<std::vector<int64_t>> adjacency;
    Eigen::VectorXd degrees;
};

// Parse a binary MATLAB graph without densifying sparse input.
// Directedness and self-loop restrictions belong to the receiving algorithm.
// Input preparation is separate from the receiving algorithm's runtime.
ParsedGraphInput ParseGraphInputParallel(const mxArray* input, int64_t num_threads);

} // namespace graph_utils
#endif

std::pair<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>> ComputeAdjAndNonAdjList(
    const Eigen::MatrixXd& adj_matrix,
    const int64_t n);

std::vector<std::vector<int64_t>> ComputeAdjList(
    const Eigen::MatrixXd& adj_matrix,
    const int64_t n);

#endif //GRAPH_UTILS_H
