#ifndef CC_BFS_APSP_L_CPP_HPP
#define CC_BFS_APSP_L_CPP_HPP

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

// Rename upstream global symbols so the vendored CC-BFS code can be embedded in
// kernel translation units without colliding with this project's own globals.
#define CC_BFS_EMBEDDED
#define COUNTERS CC_BFS_UPSTREAM_COUNTERS
#define RESULT CC_BFS_UPSTREAM_RESULT
#define CC_BFS_APSP_L CC_BFS_APSP_L_UPSTREAM
#define SelectHubNodes CC_BFS_SelectHubNodes_UPSTREAM
#define FindHighestDegreeNeighborHub CC_BFS_FindHighestDegreeNeighborHub_UPSTREAM
#define ComputeDegreeAndAdjList CC_BFS_ComputeDegreeAndAdjList_UPSTREAM
#define EstimateBackwardCost CC_BFS_EstimateBackwardCost_UPSTREAM
#define ComputeForwardCost CC_BFS_ComputeForwardCost_UPSTREAM
#define BuildSPStructures CC_BFS_BuildSPStructures_UPSTREAM
#define NodeProp CC_BFS_NodeProp_UPSTREAM
#define MemoizedSSS CC_BFS_MemoizedSSS_UPSTREAM
#define HubGuidedSSS CC_BFS_HubGuidedSSS_UPSTREAM
#define CompHubConstraint CC_BFS_CompHubConstraint_UPSTREAM
#define CC_BFS_APSP_L_mem CC_BFS_APSP_L_mem_UPSTREAM
#define CC_BFS_APSP_L_mem_FromAdjacency CC_BFS_APSP_L_mem_FromAdjacency_UPSTREAM
#define createCountersStruct CC_BFS_createCountersStruct_UPSTREAM
#define mexFunction CC_BFS_mexFunction_UPSTREAM
#define cc_bfs_apsp_py cc_bfs_apsp_py_upstream
#define CC_BFS_PY_COUNTERS CC_BFS_PY_COUNTERS_UPSTREAM

#include "../../third_party/cc_bfs/cpp/cc_bfs_apsp_l.h"
#include "../../third_party/cc_bfs/cpp/cc_bfs_apsp_l_cpp.cpp"

#undef CC_BFS_PY_COUNTERS
#undef cc_bfs_apsp_py
#undef mexFunction
#undef createCountersStruct
#undef CC_BFS_APSP_L_mem_FromAdjacency
#undef CC_BFS_APSP_L_mem
#undef CompHubConstraint
#undef HubGuidedSSS
#undef MemoizedSSS
#undef NodeProp
#undef BuildSPStructures
#undef ComputeForwardCost
#undef EstimateBackwardCost
#undef ComputeDegreeAndAdjList
#undef FindHighestDegreeNeighborHub
#undef SelectHubNodes
#undef CC_BFS_APSP_L
#undef RESULT
#undef COUNTERS
#undef CC_BFS_EMBEDDED

namespace CC_BFS_APSP {

struct COUNTERS {
    ::CC_BFS_UPSTREAM_COUNTERS upstream;
};

struct RESULT {
    Eigen::MatrixXd D;
    COUNTERS counters;
    double computation_time;
};

inline RESULT CC_BFS_APSP_L(const Eigen::MatrixXd& adj_matrix,
                            const std::string& hub_constraint = "bounded-avg-deg") {
    MatrixType adj_matrix_upstream = adj_matrix;
    auto upstream_result = ::CC_BFS_APSP_L_UPSTREAM(adj_matrix_upstream, hub_constraint, false, false);

    RESULT result;
    result.D = ResultMatrixToDouble(upstream_result.D);
    result.counters.upstream = std::move(upstream_result.counters);
    result.computation_time = upstream_result.computation_time;
    return result;
}

// Reuse the caller's prepared adjacency and degree vector by const reference.
inline RESULT CC_BFS_APSP_L_FromAdjacency(
        const std::vector<std::vector<int64_t>>& adjacency,
        const Eigen::VectorXd& degrees,
        const std::string& hub_constraint = "bounded-avg-deg") {
    auto upstream_result = ::CC_BFS_APSP_L_mem_FromAdjacency_UPSTREAM(
        adjacency, degrees, 0.0, hub_constraint, false, false, false);

    RESULT result;
    result.D = ResultMatrixToDouble(upstream_result.D);
    result.counters.upstream = std::move(upstream_result.counters);
    result.computation_time = upstream_result.computation_time;
    return result;
}

} // namespace CC_BFS_APSP

#endif // CC_BFS_APSP_L_CPP_HPP
