#ifndef PARALLEL_CC_CC_BFS_P_APSP_L_HPP_
#define PARALLEL_CC_CC_BFS_P_APSP_L_HPP_

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

#include "../../third_party/cc_bfs_p/cpp/cc_bfs_p.h"

namespace parallel_cc {

// Owning compact result for exact continuations that can consume CC-BFS-P's
// lossless uint8/uint16/uint32/uint64 distance matrix directly.
cc_bfs_p::Result CC_BFS_P_APSP_L_Compact_FromAdjacency(
    const std::vector<std::vector<int64_t>>& adjacency,
    const Eigen::VectorXd& degrees,
    int64_t num_threads);

// Thin boundary to the settled parallel CC-BFS-P core. The caller supplies
// the adjacency already parsed for CC-APSPN, so this adds no second graph
// representation and changes no CC-BFS-P logic.
Eigen::MatrixXd CC_BFS_P_APSP_L_FromAdjacency(
    const std::vector<std::vector<int64_t>>& adjacency,
    const Eigen::VectorXd& degrees,
    int64_t num_threads);

}  // namespace parallel_cc

#endif
