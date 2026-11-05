#include "cc_apspn_kernel_par.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <iomanip>
#include <numeric>
#include <variant>
#include <optional>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <exception>
#include <cctype>
#include <limits>
#include <type_traits>
#include <omp.h>
#include "../../common_include/cc_bfs_p_apsp_l.hpp"
#include "../../common_include/graph_utils.h"
#include "../../common_include/timer.h"
#include "../../common_include/parallel_utils.hpp"
#include "../../common_include/process_memory_tracker.h"
#ifdef MATLAB_BUILD
#include "mex.h"
#include "../../common_include/matlab_vcomp_lifetime.hpp"
#include "../../common_include/matlab_adjacency_list.h"
#include "../../common_include/mex_utils.h"
#endif

#ifdef MATLAB_BUILD
mxArray* CreateCountersStruct(const COUNTERS& counters) {
    // Create field names array
    const char* fieldnames[] = {
            "mem_util_count",
            "graph_trav_count",
            "apspn_mem_con",
            "fwd_visit_count",
            "bwd_visit_count",
            "bc_bwd_visit_count",
            "bc_fwd_visit_count",
            "durationInSecPri",
            "durationInSecComputeAdjList",
            "durationInSecAPSPComputation",
            "durationInSecAPSPNInit",
            "durationInSecAPSPNLevelComputation",
            "durationInSecAPSPNLevelTraversal",
            "durationInSecAPSPNComputation",
            "durationInSecTotal",
            "durationInSecCcBbBFS",
            "durationInSecPostprocess",
            "requested_threads",
            "effective_threads",
            "process_baseline_working_set_bytes",
            "process_peak_working_set_bytes",
            "process_delta_peak_working_set_bytes",
            "process_baseline_private_bytes",
            "process_peak_private_bytes",
            "process_delta_peak_private_bytes",
            "memory_measurement",
            "distance_matrix_bytes",
            "distance_data_type",
            "distance_diameter_upper_bound",
            "durationInSecOutputAllocation",
            "durationInSecEccentricity"
    };
    constexpr size_t numFields = sizeof(fieldnames) / sizeof(fieldnames[0]);

    // Create the struct
    mxArray* matlabStruct = mxCreateStructMatrix(1, 1, numFields, fieldnames);

    // Set each field manually
    mxSetFieldByNumber(matlabStruct, 0, 0, mxCreateDoubleScalar(counters.mem_util_count));
    mxSetFieldByNumber(matlabStruct, 0, 1, mxCreateDoubleScalar(counters.graph_trav_count));
    mxSetFieldByNumber(matlabStruct, 0, 2, mxCreateDoubleScalar(counters.apspn_mem_con));
    mxSetFieldByNumber(matlabStruct, 0, 3, mxCreateDoubleScalar(counters.fwd_visit_count));
    mxSetFieldByNumber(matlabStruct, 0, 4, mxCreateDoubleScalar(counters.bwd_visit_count));
    mxSetFieldByNumber(matlabStruct, 0, 5, mxCreateDoubleScalar(counters.bc_bwd_visit_count));
    mxSetFieldByNumber(matlabStruct, 0, 6, mxCreateDoubleScalar(counters.bc_fwd_visit_count));
    mxSetFieldByNumber(matlabStruct, 0, 7, mxCreateDoubleScalar(counters.durationInSecPri));
    mxSetFieldByNumber(matlabStruct, 0, 8, mxCreateDoubleScalar(counters.durationInSecComputeAdjList));
    mxSetFieldByNumber(matlabStruct, 0, 9, mxCreateDoubleScalar(counters.durationInSecAPSPComputation));
    mxSetFieldByNumber(matlabStruct, 0, 10, mxCreateDoubleScalar(counters.durationInSecAPSPNInit));
    mxSetFieldByNumber(matlabStruct, 0, 11, mxCreateDoubleScalar(counters.durationInSecAPSPNLevelComputation));
    mxSetFieldByNumber(matlabStruct, 0, 12, mxCreateDoubleScalar(counters.durationInSecAPSPNLevelTraversal));
    mxSetFieldByNumber(matlabStruct, 0, 13, mxCreateDoubleScalar(counters.durationInSecAPSPNComputation));
    mxSetFieldByNumber(matlabStruct, 0, 14, mxCreateDoubleScalar(counters.durationInSecTotal));
    mxSetFieldByNumber(matlabStruct, 0, 15, mxCreateDoubleScalar(counters.durationInSecCcBbBFS));
    mxSetFieldByNumber(matlabStruct, 0, 16, mxCreateDoubleScalar(counters.durationInSecPostprocess));
    mxSetFieldByNumber(matlabStruct, 0, 17, mxCreateDoubleScalar(static_cast<double>(counters.requested_threads)));
    mxSetFieldByNumber(matlabStruct, 0, 18, mxCreateDoubleScalar(static_cast<double>(counters.effective_threads)));
    mxSetFieldByNumber(matlabStruct, 0, 19, mxCreateDoubleScalar(static_cast<double>(counters.process_baseline_working_set_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 20, mxCreateDoubleScalar(static_cast<double>(counters.process_peak_working_set_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 21, mxCreateDoubleScalar(static_cast<double>(counters.process_delta_peak_working_set_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 22, mxCreateDoubleScalar(static_cast<double>(counters.process_baseline_private_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 23, mxCreateDoubleScalar(static_cast<double>(counters.process_peak_private_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 24, mxCreateDoubleScalar(static_cast<double>(counters.process_delta_peak_private_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 25, mxCreateString("checkpoint-sampled process memory"));
    mxSetFieldByNumber(matlabStruct, 0, 26, mxCreateDoubleScalar(static_cast<double>(counters.distance_matrix_bytes)));
    mxSetFieldByNumber(matlabStruct, 0, 27, mxCreateString(counters.distance_data_type.c_str()));
    mxSetFieldByNumber(matlabStruct, 0, 28, mxCreateDoubleScalar(static_cast<double>(counters.distance_diameter_upper_bound)));
    mxSetFieldByNumber(matlabStruct, 0, 29, mxCreateDoubleScalar(counters.durationInSecOutputAllocation));
    mxSetFieldByNumber(matlabStruct, 0, 30, mxCreateDoubleScalar(counters.durationInSecEccentricity));

    return matlabStruct;
}
#endif // MATLAB_BUILD

template <typename DistScalar>
Eigen::VectorXd DistanceColumnSums(const DistMat<DistScalar>& D) {
    Eigen::VectorXd sums = Eigen::VectorXd::Zero(D.cols());
    for (Eigen::Index column = 0; column < D.cols(); ++column) {
        double value = 0.0;
        for (Eigen::Index row = 0; row < D.rows(); ++row) {
            value += static_cast<double>(D(row, column));
        }
        sums(column) = value;
    }
    return sums;
}

// Merge one completed source's counters after its worker has finished. This
// keeps counter writes out of the parallel source-search loops.
void AddSourceCounters(COUNTERS& total, const COUNTERS& source) {
    total.mem_util_count += source.mem_util_count;
    total.graph_trav_count += source.graph_trav_count;
    total.fwd_visit_count += source.fwd_visit_count;
    total.bwd_visit_count += source.bwd_visit_count;
    total.bc_bwd_visit_count += source.bc_bwd_visit_count;
    total.bc_fwd_visit_count += source.bc_fwd_visit_count;
    total.durationInSecAPSPNInit += source.durationInSecAPSPNInit;
    total.durationInSecAPSPNLevelComputation += source.durationInSecAPSPNLevelComputation;
    total.durationInSecAPSPNLevelTraversal += source.durationInSecAPSPNLevelTraversal;
}

// A workspace is used by only one source search at a time. Its buffers are
// reset before reuse; the ordinary schedule assigns workspaces to workers,
// while the pipeline assigns them to bank slots.
struct SourceSearchWorkspace {
    SourceSearchWorkspace(const int64_t n, const bool needs_projection)
        : pcount_mem(n), frontier_mask(n, false), level_mask(n, false) {
        if (needs_projection) projection_mem.emplace(n);
    }

    Eigen::VectorXd pcount_mem;
    std::optional<Eigen::VectorXd> projection_mem;
    std::vector<std::vector<int64_t>> levels;
    size_t active_level_count = 0;
    std::vector<char> frontier_mask;
    std::vector<char> level_mask;
    std::vector<int64_t> mem_sources;
};

template <typename Derived>
void BuildSpLevels(
        const int64_t n,
        const int64_t max_depth,
        const Eigen::MatrixBase<Derived>& D_s,
        SourceSearchWorkspace& workspace) {
    // Clear every level used by the preceding source. Levels outside that
    // range were cleared when they last became inactive and retain only their
    // capacity.
    for (size_t level = 0; level < workspace.active_level_count; ++level) {
        workspace.levels[level].clear();
    }
    const size_t required_levels = static_cast<size_t>(max_depth + 1);
    if (workspace.levels.size() < required_levels) {
        workspace.levels.resize(required_levels);
    }
    workspace.active_level_count = required_levels;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t depth = static_cast<int64_t>(D_s(i));
        if (depth < 0 || depth > max_depth) {
            throw std::runtime_error("Compact distance is outside its eccentricity range.");
        }
        workspace.levels[static_cast<size_t>(depth)].push_back(i);
    }
}


// Shared source path-counting phase for every exact mode. GC and APSLP add
// their mode-specific values while following this same traversal.
template <typename DistScalar>
void DistanceGuidedSssPathPhase(
        const std::vector<std::vector<int64_t>>& A,
        const int64_t n,
        const int64_t s,
        const int64_t mem_con,
        const DistMat<DistScalar>& D,
        Eigen::MatrixXd& Pcount,
        const Eigen::VectorXd& eccs,
        const AlgorithmType& algo_type,
        std::optional<Eigen::MatrixXd>& P,
        std::optional<Eigen::MatrixXd>& APSLP,
        const std::optional<Eigen::Map<const Eigen::MatrixXd>>& RD,
        const std::vector<char>& processed,
        COUNTERS& counters,
        SourceSearchWorkspace& workspace) {
    Timer local_timer;
    local_timer.Start();

    auto D_s = D.col(s);
    auto Pcount_s = Pcount.col(s);
    // This source owns column s. Clear it before any source-local memoization
    // reads it, replacing the entry thread's serial full-matrix zero-fill.
    Pcount_s.setZero();
    workspace.pcount_mem = Pcount_s;  // memoized Pcount_s
    Eigen::VectorXd& Pcount_s_mem = workspace.pcount_mem;
    Pcount_s(s) = 1;
    // GC and APSLP compute their mode-specific values during the same path pass.
    std::optional<Eigen::Ref<Eigen::VectorXd>> P_s;
    Eigen::VectorXd* P_s_mem = nullptr;
    std::optional<Eigen::Ref<Eigen::VectorXd>> SSSLP;  // single-source shortest-lightest path

    switch (algo_type) {
    case AlgorithmType::BC:
        break;
    case AlgorithmType::GC:
        assert(P.has_value());
        assert(RD.has_value());
        P_s = (*P).col(s);
        // A worker initializes its column before any traversal or publication.
        (*P_s).setZero();
        workspace.projection_mem.value() = P_s.value();
        P_s_mem = &workspace.projection_mem.value();
        break;
    case AlgorithmType::APSLP:
        assert(APSLP.has_value());
        assert(RD.has_value());
        SSSLP = (*APSLP).col(s);
        (*SSSLP).setConstant(INF);
        (*SSSLP)(s) = 0;
        break;
    }

    local_timer.Stop();
    counters.durationInSecAPSPNInit += local_timer.Seconds();

    local_timer.Start();
    // Group vertices by distance from the source.
    const int64_t max_depth = static_cast<int64_t>(eccs(s));
    BuildSpLevels(n, max_depth, D_s, workspace);
    auto& levels = workspace.levels;
    local_timer.Stop();
    counters.durationInSecAPSPNLevelComputation += local_timer.Seconds();

    // Count paths from this source, reusing completed neighbor columns when allowed.
    local_timer.Start();

    std::fill(workspace.frontier_mask.begin(), workspace.frontier_mask.end(), false);
    std::fill(workspace.level_mask.begin(), workspace.level_mask.end(), false);
    auto& frontier_mask = workspace.frontier_mask;
    auto& level_mask = workspace.level_mask;
    frontier_mask[s] = true;

    // Every neighbor of the source has one direct shortest path.
    int64_t curr = s;
    level_mask[s] = true;
    for (const int64_t t : A[curr])
    {
        Pcount_s(t) = 1;  // initialize Pcount for the first level
        switch (algo_type) {
        case AlgorithmType::GC:
            (*P_s)[t] += (*P_s)[curr] + (*RD).col(curr)(t) * Pcount_s(curr);
            break;
        case AlgorithmType::APSLP: // the TSP from s to t via curr is lighter?
            (*SSSLP)(t) = std::min<double>((*SSSLP)(t), (*SSSLP)(curr) + (*RD).col(curr)(t));
            break;
        }
        frontier_mask[t] = true;
    }
    // Previously completed neighbors can supply their path-count columns.
    workspace.mem_sources.clear();
    auto& mem_sources = workspace.mem_sources;
    if (A[s].size() <= mem_con)
    {
        for (const int64_t t : A[curr])
        {
            if (processed[t]) // previously processed neighbor can replay its path counts
            {
                mem_sources.push_back(t);
            }
        }
    }
    // Replay those completed columns before traversing the remaining graph.
    if (!mem_sources.empty())
    {
        // Replayed neighbors no longer need graph traversal from this source.
        for (auto ms : mem_sources) {
            frontier_mask[ms] = false;
        }
        for (int64_t curr_depth = 1; curr_depth < max_depth; ++curr_depth)
        {
            const auto& dst_level = levels[curr_depth + 1];  // destination level
            for (auto ms : mem_sources)
            {
                if (curr_depth > eccs[ms]) { // reached the end of ps's SP level
                    continue;
                }
                const auto& D_ms = D.col(ms);
                const auto& Pcount_ms = Pcount.col(ms);

                for (const int64_t i : dst_level) {
                    if (static_cast<int64_t>(D_ms(i)) == curr_depth) {  // shortest path from s to i pass through ps
                        Pcount_s_mem(i) += Pcount_ms(i);
                        switch (algo_type) {
                        case AlgorithmType::GC:
                            (*P_s_mem)(i) += (*P).col(ms)(i) + (*P_s)(ms) * Pcount_ms(i);
                            break;
                        case AlgorithmType::APSLP:  // the TSP from s to i via ps is lighter?
                            (*SSSLP)(i) = std::min<double>((*SSSLP)(i), (*SSSLP)(ms) + (*APSLP).col(ms)(i));
                            break;
                        }
                    }
                }
            }
        }
    }

    // Count the remaining paths one distance level at a time.
    for (int64_t curr_depth = 1; curr_depth < max_depth; ++curr_depth) {
        const auto& curr_level = levels[curr_depth];
        const auto& dst_level = levels[curr_depth + 1];  // destination level
        // Mark every level already reached so forward scans only enter the next level.
        for (auto& curr : curr_level) {
            level_mask[curr] = true;
        }
        // Use the cheaper of pushing from the frontier and pulling from destinations.
        int64_t fwd_cost = 0;
        int64_t bwd_cost = 0;

        for (const int64_t curr : curr_level) {
            if (!frontier_mask[curr]) continue;
            fwd_cost += A[curr].size();
        }
        if (fwd_cost == 0) continue; // no more frontier nodes
        for (const int64_t t : dst_level) {
            if (processed[t]) continue;
            bwd_cost += A[t].size();
        }
        if (fwd_cost < bwd_cost) { // forward traversal is cheaper
            for (const int64_t curr : curr_level) {
                if (!frontier_mask[curr]) continue;
                counters.graph_trav_count++;
                counters.fwd_visit_count++;
                const double curr_pcount = Pcount_s(curr);
                for (const int64_t t : A[curr]) {
                    if (level_mask[t]) continue;
                    // Unmasked neighbors lie one level farther from the source.
                    Pcount_s(t) += curr_pcount;
                    switch (algo_type) {
                    case AlgorithmType::GC:
                        (*P_s)[t] += (*P_s)[curr] + (*RD).col(curr)(t) * Pcount_s(curr);
                        break;
                    case AlgorithmType::APSLP: // the TSP from s to t via curr is lighter?
                        (*SSSLP)(t) = (std::min<double>((*SSSLP)(t), (*SSSLP)(curr) + (*RD).col(curr)(t)));
                        break;
                    }
                    frontier_mask[t] = true;
                }
            }
        }
        else
        { // backward traversal is cheaper
            for (const int64_t curr : dst_level) {
                if (processed[curr])  // apply graph symmetry if possible
                {
                    Pcount_s(curr) = Pcount.col(curr)(s) - Pcount_s_mem(curr);
                    switch (algo_type) {
                    case AlgorithmType::GC:
                        (*P_s)[curr] = (*P).col(curr)(s) - (*P_s_mem)(curr);
                        break;
                    case AlgorithmType::APSLP: // the TSP from s to t via curr is lighter?
                        (*SSSLP)(curr) = (*APSLP).col(curr)(s);
                        break;
                    }
                    if (Pcount_s(curr) > 0) frontier_mask[curr] = true;  // curr not reachable via graph traversal
                }
                else
                {
                    counters.graph_trav_count++;
                    counters.bwd_visit_count++;
                    for (const int64_t t : A[curr]) {
                        if (!frontier_mask[t] || !level_mask[t]) continue;
                        // t is a valid predecessor of curr in the SP-DAG
                        Pcount_s(curr) += Pcount_s(t);
                        switch (algo_type) {
                        case AlgorithmType::GC:
                            (*P_s)[curr] += (*P_s)[t] + (*RD).col(t)(curr) * Pcount_s(t);
                            break;
                        case AlgorithmType::APSLP: // the TSP from s to curr via t is lighter?
                            (*SSSLP)(curr) = std::min<double>((*SSSLP)(curr), (*SSSLP)(t) + (*RD).col(t)(curr));
                            break;
                        }
                        frontier_mask[curr] = true;
                    }
                }
            }
        }
    }

    local_timer.Stop();
    counters.durationInSecAPSPNLevelTraversal += local_timer.Seconds();
    // Add the replayed contributions to the paths found by graph traversal.
    Pcount_s += Pcount_s_mem;
    if (algo_type == AlgorithmType::GC) {
        (*P_s) += *P_s_mem;
    }
    counters.mem_util_count += mem_sources.size();
}

// Exact-BC dependency work is independent after the source path-count column
// has been combined. It reads the retained source levels and Pcount column and
// writes only the source slot's private BC vector.
template <typename DistScalar>
void RunBcDependencyPhase(
        const std::vector<std::vector<int64_t>>& A,
        const int64_t s,
        const DistMat<DistScalar>& D,
        Eigen::MatrixXd& Pcount,
        const Eigen::VectorXd& eccs,
        COUNTERS& counters,
        SourceSearchWorkspace& workspace,
        Eigen::VectorXd& BC_s) {
    const auto D_s = D.col(s);
    const auto Pcount_s = Pcount.col(s);
    const int64_t max_depth = static_cast<int64_t>(eccs(s));
    const auto& levels = workspace.levels;

    BC_s.setZero();
    BC_s(s) = 0.0;
    for (int64_t depth = max_depth; depth >= 2; --depth) {
        int64_t bwd_cost = 0;
        int64_t fwd_cost = 0;
        for (const int64_t u : levels[static_cast<size_t>(depth)]) {
            bwd_cost += static_cast<int64_t>(A[static_cast<size_t>(u)].size());
        }
        for (const int64_t u : levels[static_cast<size_t>(depth - 1)]) {
            fwd_cost += static_cast<int64_t>(A[static_cast<size_t>(u)].size());
        }
        if (bwd_cost < fwd_cost) {
            for (const int64_t u : levels[static_cast<size_t>(depth)]) {
                ++counters.bc_bwd_visit_count;
                for (const int64_t v : A[static_cast<size_t>(u)]) {
                    if (static_cast<int64_t>(D_s(v)) ==
                        static_cast<int64_t>(D_s(u)) - 1) {
                        BC_s(v) +=
                            (Pcount_s(v) / Pcount_s(u)) * (1 + BC_s(u));
                    }
                }
            }
        } else {
            for (const int64_t v : levels[static_cast<size_t>(depth - 1)]) {
                ++counters.bc_fwd_visit_count;
                for (const int64_t u : A[static_cast<size_t>(v)]) {
                    if (static_cast<int64_t>(D_s(v)) ==
                        static_cast<int64_t>(D_s(u)) - 1) {
                        BC_s(v) +=
                            (Pcount_s(v) / Pcount_s(u)) * (1 + BC_s(u));
                    }
                }
            }
        }
    }
}

// All exact modes share prepared adjacency; none needs a dense adjacency copy.
Result CcApspnKernelExact(
        const std::vector<std::vector<int64_t>>& A,
        const Eigen::VectorXd& DEG,
        const AlgorithmType& algo_type,
        const AlgorithmParams& params,
        const int64_t num_threads) {

    // The graph is already parsed. Measure algorithm computation only,
    // including working-memory initialization but excluding result delivery.
    COUNTERS counters;
    Timer global_timer;
    Timer local_timer;
    global_timer.Start();
    const int64_t n = static_cast<int64_t>(A.size());

    PrintAlgorithmHeader(algo_type, params);

    // Initialize per-call outputs and source state.
    local_timer.Start();
    // Each source worker clears its owned column immediately before use.
    // Avoid a serial n-by-n zero-fill on the MATLAB entry thread.
    Eigen::MatrixXd Pcount(n, n);  // Number of shortest paths
    std::optional<Eigen::Map<const Eigen::MatrixXd>> RD;
    std::optional<Eigen::MatrixXd> P;  // Average shortest path projection
    std::optional<Eigen::VectorXd> BC;  // Betweenness centrality
    std::optional<Eigen::VectorXd> PHD;  // Path-hesitation degree
    std::optional<Eigen::MatrixXd> APSLP; // All-pairs shortest-lightest paths
    std::optional<double> GC;  // Geometrical congruence

    std::vector<char> processed(n, false);
    std::vector<int64_t> source_list(n); std::iota(source_list.begin(), source_list.end(), 0);
    // GC and APSLP read the same referential-distance view. A converted sparse
    // or logical input is owned by params for the lifetime of this call.
    if (algo_type == AlgorithmType::GC || algo_type == AlgorithmType::APSLP) {
        if (params.rd_data != nullptr) {
            RD.emplace(params.rd_data, n, n);
        } else if (params.rd_matrix.has_value()) {
            RD.emplace(params.rd_matrix->data(), n, n);
        } else {
            throw std::runtime_error("GC and APSLP require an RD matrix.");
        }
    }
    // Initialize variables based on problem mode
    switch (algo_type) {
    case AlgorithmType::GC:
        P.emplace(n, n);
        break;
    case AlgorithmType::BC:
        BC = Eigen::VectorXd::Zero(n);
        break;
    case AlgorithmType::PHD:
        PHD = Eigen::VectorXd::Zero(n);
        break;
    case AlgorithmType::APSLP:
        APSLP.emplace(n, n);
        break;
    }

    local_timer.Stop();
    counters.durationInSecAPSPNInit = local_timer.Seconds();
    counters.durationInSecOutputAllocation = local_timer.Seconds();
    parallel_cc::memory::UpdateActivePeak();
    if (params.verbose) {
        std::cout << "@@ Initialization time: " << local_timer.Seconds() << " seconds" << std::endl;
    }

    // Compute the compact all-pairs distance matrix.
    local_timer.Start();
    // All exact modes share the same distance stage. The adapter owns and
    // joins its OpenMP team before returning, so the later persistent
    // CC-APSPN source team begins at a serial boundary and is never nested.
    auto distance_result = parallel_cc::CC_BFS_P_APSP_L_Compact_FromAdjacency(
        A, DEG, num_threads);
    local_timer.Stop();
    // This phase includes diameter selection and parallel CC-BFS execution.
    // The owning compact matrix stays live through the exact computation.
    counters.durationInSecAPSPComputation = local_timer.Seconds();
    counters.distance_matrix_bytes =
        distance_result.counters.distance_matrix_bytes;
    counters.distance_data_type = distance_result.counters.data_type;
    counters.distance_diameter_upper_bound =
        distance_result.counters.diam_upper;
    parallel_cc::memory::UpdateActivePeak();
    if (params.verbose) {
        std::cout << "@@ APSP computation time: " << local_timer.Seconds() << " seconds" << std::endl;
    }

    // Run the same exact algorithm for whichever lossless distance type
    // CC-BFS-P selected. Scientific outputs and path-count arithmetic remain
    // double in every case.
    auto run_exact = [&](const auto &D) -> Result {
        // Process high-degree sources first, matching the complete CC-APSPN algorithm.
        std::sort(source_list.begin(), source_list.end(), [&](int64_t a, int64_t b) {
            return DEG[a] > DEG[b];
        });
        int64_t m = DEG.sum();
        int64_t avg_deg = m / n;
        int64_t mem_con = std::floor(std::sqrt(avg_deg));
        counters.apspn_mem_con = mem_con;

        // Compute each source's eccentricity from the compact distance matrix.
        local_timer.Start();
        Eigen::VectorXd eccs(D.cols());
#pragma omp parallel for num_threads(num_threads) schedule(static)
        for (Eigen::Index column = 0; column < D.cols(); ++column) {
            eccs(column) = static_cast<double>(D.col(column).maxCoeff());
        }
        local_timer.Stop();
        counters.durationInSecEccentricity = local_timer.Seconds();
        if (params.verbose) {
            std::cout << "@@ Eccentricity computation time: " << local_timer.Seconds() << " seconds"
                      << std::endl;
        }

        // Process sources with the selected parallel schedule.
        local_timer.Start();
        int64_t max_effective_threads = 1;
        if (algo_type == AlgorithmType::BC) {
            // Overlap one batch's BC dependencies with the next batch's
            // path-counting work.
            const int64_t max_batch_size = std::min<int64_t>(num_threads, n);
            const int64_t batch_count =
                (static_cast<int64_t>(source_list.size()) + num_threads - 1) / num_threads;

            std::vector<COUNTERS> source_counters[2] = {
                std::vector<COUNTERS>(static_cast<size_t>(max_batch_size)),
                std::vector<COUNTERS>(static_cast<size_t>(max_batch_size))};
            std::vector<std::exception_ptr> path_errors[2] = {
                std::vector<std::exception_ptr>(static_cast<size_t>(max_batch_size)),
                std::vector<std::exception_ptr>(static_cast<size_t>(max_batch_size))};
            std::vector<std::exception_ptr> dependency_errors[2] = {
                std::vector<std::exception_ptr>(static_cast<size_t>(max_batch_size)),
                std::vector<std::exception_ptr>(static_cast<size_t>(max_batch_size))};
            std::vector<SourceSearchWorkspace> workspace_banks[2];
            for (int bank = 0; bank < 2; ++bank) {
                workspace_banks[bank].reserve(static_cast<size_t>(max_batch_size));
                for (int64_t slot = 0; slot < max_batch_size; ++slot) {
                    workspace_banks[bank].emplace_back(n, false);
                }
            }
            std::vector<Eigen::VectorXd> source_bc;
            source_bc.reserve(static_cast<size_t>(max_batch_size));
            for (int64_t slot = 0; slot < max_batch_size; ++slot) {
                source_bc.emplace_back(Eigen::VectorXd::Zero(n));
            }

            int64_t bank_batch_size[2] = {0, 0};
            int64_t bank_path_success[2] = {0, 0};
            int64_t bank_dependency_success[2] = {0, 0};
            int64_t team_size = 1;
            std::exception_ptr pending_error;

#pragma omp parallel num_threads(num_threads)
            {
#pragma omp master
                team_size = omp_get_num_threads();
#pragma omp barrier

// Fill bank 0 with the first path-count batch. Later waves
// overlap dependencies from one bank with path counts in the other.
#pragma omp master
                bank_batch_size[0] =
                    std::min<int64_t>(num_threads, static_cast<int64_t>(source_list.size()));
#pragma omp barrier
#pragma omp for schedule(static, 1)
                for (int64_t slot = 0; slot < bank_batch_size[0]; ++slot) {
                    const int64_t s = source_list[static_cast<size_t>(slot)];
                    source_counters[0][static_cast<size_t>(slot)] = COUNTERS{};
                    path_errors[0][static_cast<size_t>(slot)] = nullptr;
                    try {
                        DistanceGuidedSssPathPhase(A, n, s, mem_con, D, Pcount, eccs,
                                                   AlgorithmType::BC, P, APSLP, RD, processed,
                                                   source_counters[0][static_cast<size_t>(slot)],
                                                   workspace_banks[0][static_cast<size_t>(slot)]);
                    } catch (...) {
                        path_errors[0][static_cast<size_t>(slot)] = std::current_exception();
                    }
                }

#pragma omp master
                {
                    bank_path_success[0] = 0;
                    for (int64_t slot = 0; slot < bank_batch_size[0]; ++slot) {
                        if (path_errors[0][static_cast<size_t>(slot)]) {
                            pending_error = path_errors[0][static_cast<size_t>(slot)];
                            break;
                        }
                        processed[source_list[static_cast<size_t>(slot)]] = true;
                        ++bank_path_success[0];
                    }
                    max_effective_threads = std::max<int64_t>(
                        max_effective_threads, std::min<int64_t>(bank_batch_size[0], team_size));
                    if (!pending_error)
                        parallel_cc::memory::UpdateActivePeak();
                }
#pragma omp barrier

                for (int64_t batch = 0; batch < batch_count; ++batch) {
                    const int current_bank = static_cast<int>(batch & 1);
                    const int next_bank = current_bank ^ 1;
                    const bool has_next = batch + 1 < batch_count;
                    const int64_t next_begin = (batch + 1) * num_threads;
                    const int64_t next_size =
                        has_next ? std::min<int64_t>(num_threads,
                                                     static_cast<int64_t>(source_list.size()) -
                                                         next_begin)
                                 : 0;

#pragma omp master
                    {
                        bank_dependency_success[current_bank] = 0;
                        for (int64_t slot = 0; slot < bank_path_success[current_bank]; ++slot) {
                            dependency_errors[current_bank][static_cast<size_t>(slot)] = nullptr;
                        }
                        if (has_next && !pending_error) {
                            bank_batch_size[next_bank] = next_size;
                        }
                    }
#pragma omp barrier

                    const int64_t dependency_jobs = bank_path_success[current_bank];
                    const int64_t path_jobs = (has_next && !pending_error) ? next_size : 0;
                    const int64_t combined_jobs = dependency_jobs + path_jobs;

// Dependency jobs have the lower indices. Dynamic scheduling lets
// a worker begin the next batch's path count after it finishes a
// dependency job, while slower dependencies continue elsewhere.
#pragma omp for schedule(dynamic, 1)
                    for (int64_t job = 0; job < combined_jobs; ++job) {
                        if (job < dependency_jobs) {
                            const int64_t slot = job;
                            const int64_t source_index = batch * num_threads + slot;
                            const int64_t s = source_list[static_cast<size_t>(source_index)];
                            try {
                                RunBcDependencyPhase(
                                    A, s, D, Pcount, eccs,
                                    source_counters[current_bank][static_cast<size_t>(slot)],
                                    workspace_banks[current_bank][static_cast<size_t>(slot)],
                                    source_bc[static_cast<size_t>(slot)]);
                            } catch (...) {
                                dependency_errors[current_bank][static_cast<size_t>(slot)] =
                                    std::current_exception();
                            }
                        } else {
                            const int64_t slot = job - dependency_jobs;
                            const int64_t source_index = next_begin + slot;
                            const int64_t s = source_list[static_cast<size_t>(source_index)];
                            source_counters[next_bank][static_cast<size_t>(slot)] = COUNTERS{};
                            path_errors[next_bank][static_cast<size_t>(slot)] = nullptr;
                            try {
                                DistanceGuidedSssPathPhase(
                                    A, n, s, mem_con, D, Pcount, eccs, AlgorithmType::BC, P, APSLP,
                                    RD, processed,
                                    source_counters[next_bank][static_cast<size_t>(slot)],
                                    workspace_banks[next_bank][static_cast<size_t>(slot)]);
                            } catch (...) {
                                path_errors[next_bank][static_cast<size_t>(slot)] =
                                    std::current_exception();
                            }
                        }
                    }

#pragma omp master
                    {
                        for (int64_t slot = 0; slot < dependency_jobs; ++slot) {
                            if (dependency_errors[current_bank][static_cast<size_t>(slot)]) {
                                if (!pending_error) {
                                    pending_error =
                                        dependency_errors[current_bank][static_cast<size_t>(slot)];
                                }
                                break;
                            }
                            AddSourceCounters(
                                counters, source_counters[current_bank][static_cast<size_t>(slot)]);
                            ++bank_dependency_success[current_bank];
                        }

                        if (has_next && !pending_error) {
                            bank_path_success[next_bank] = 0;
                            for (int64_t slot = 0; slot < next_size; ++slot) {
                                if (path_errors[next_bank][static_cast<size_t>(slot)]) {
                                    pending_error =
                                        path_errors[next_bank][static_cast<size_t>(slot)];
                                    break;
                                }
                                processed[source_list[static_cast<size_t>(next_begin + slot)]] =
                                    true;
                                ++bank_path_success[next_bank];
                            }
                            max_effective_threads = std::max<int64_t>(
                                max_effective_threads, std::min<int64_t>(next_size, team_size));
                            if (!pending_error) {
                                parallel_cc::memory::UpdateActivePeak();
                            }
                        }
                    }
#pragma omp barrier

                    const int64_t vertex_begin =
                        n * static_cast<int64_t>(omp_get_thread_num()) / team_size;
                    const int64_t vertex_end =
                        n * static_cast<int64_t>(omp_get_thread_num() + 1) / team_size;
                    for (int64_t vertex = vertex_begin; vertex < vertex_end; ++vertex) {
                        double value = BC.value()(vertex);
                        for (int64_t slot = 0; slot < bank_dependency_success[current_bank];
                             ++slot) {
                            value += source_bc[static_cast<size_t>(slot)](vertex);
                        }
                        BC.value()(vertex) = value;
                    }
#pragma omp barrier
                    if (pending_error)
                        break;
                }
            }

            if (pending_error)
                std::rethrow_exception(pending_error);
        } else {
            // The other exact modes have no BC dependency work to overlap.
            const int64_t max_batch_size = std::min<int64_t>(num_threads, n);
            std::vector<COUNTERS> source_counters(static_cast<size_t>(max_batch_size));
            std::vector<std::exception_ptr> source_errors(static_cast<size_t>(max_batch_size));
            // Allocate one reusable workspace per requested worker. During
            // the parallel region, only that worker accesses its workspace.
            std::vector<SourceSearchWorkspace> workspaces;
            workspaces.reserve(static_cast<size_t>(num_threads));
            for (int64_t worker = 0; worker < num_threads; ++worker) {
                workspaces.emplace_back(n, algo_type == AlgorithmType::GC);
            }
            int64_t team_size = 1;
            std::exception_ptr pending_error;
            const int64_t batch_count =
                (static_cast<int64_t>(source_list.size()) + num_threads - 1) / num_threads;

// One team serves every ordered batch. The barriers are the visibility
// boundary: all workers compute against one frozen `processed` snapshot,
// then the entry thread publishes successful slots in source order before
// any worker starts the next batch.
#pragma omp parallel num_threads(num_threads)
            {
                const int worker_id = omp_get_thread_num();
                SourceSearchWorkspace &workspace = workspaces[static_cast<size_t>(worker_id)];

#pragma omp master
                team_size = omp_get_num_threads();
#pragma omp barrier

                for (int64_t batch = 0; batch < batch_count; ++batch) {
                    const int64_t batch_begin = batch * num_threads;
                    const int64_t batch_size = std::min<int64_t>(
                        num_threads, static_cast<int64_t>(source_list.size()) - batch_begin);
// Slot-owned results remain separate from worker-owned scratch:
// an OpenMP runtime may create fewer workers than requested, in
// which case one worker can compute more than one slot.
#pragma omp for schedule(static, 1)
                    for (int64_t slot = 0; slot < batch_size; ++slot) {
                        const int64_t s = source_list[static_cast<size_t>(batch_begin + slot)];
                        // The worker that owns this source slot prepares all of its
                        // reusable per-source state before starting the search. The
                        // omp-for's implicit barrier remains the publication boundary.
                        source_counters[static_cast<size_t>(slot)] = COUNTERS{};
                        source_errors[static_cast<size_t>(slot)] = nullptr;
                        try {
                            DistanceGuidedSssPathPhase(A, n, s, mem_con, D, Pcount, eccs, algo_type,
                                                       P, APSLP, RD, processed,
                                                       source_counters[static_cast<size_t>(slot)],
                                                       workspace);
                        } catch (...) {
                            // Workers record exceptions but never call MATLAB or throw
                            // across the OpenMP boundary. The entry thread rethrows
                            // only after the persistent team has joined.
                            source_errors[static_cast<size_t>(slot)] = std::current_exception();
                        }
                    }

#pragma omp master
                    {
                        max_effective_threads = std::max<int64_t>(
                            max_effective_threads, std::min<int64_t>(batch_size, team_size));
                        for (int64_t slot = 0; slot < batch_size; ++slot) {
                            if (source_errors[static_cast<size_t>(slot)]) {
                                pending_error = source_errors[static_cast<size_t>(slot)];
                                break;
                            }
                            AddSourceCounters(counters, source_counters[static_cast<size_t>(slot)]);
                            processed[source_list[static_cast<size_t>(batch_begin + slot)]] = true;
                        }
                        if (!pending_error) {
                            parallel_cc::memory::UpdateActivePeak();
                        }
                    }
#pragma omp barrier
                    if (pending_error)
                        break;
                }
            }
            if (pending_error) {
                std::rethrow_exception(pending_error);
            }
        }
        local_timer.Stop();
        parallel_cc::memory::UpdateActivePeak();
        counters.durationInSecAPSPNComputation = local_timer.Seconds();
        counters.requested_threads = num_threads;
        counters.effective_threads = max_effective_threads;

        if (params.verbose) {
            std::cout << "@@ APSPN computation costs: " << std::fixed << std::setprecision(2)
                      << local_timer.Seconds() << std::endl;
            std::cout << "    Worker timings below are summed across sources, not elapsed wall time."
                      << std::endl;
            std::cout << "    Source initialization (summed worker seconds): "
                      << counters.durationInSecAPSPNInit - counters.durationInSecOutputAllocation
                      << std::endl;
            std::cout << "    Level computation (summed worker seconds): "
                      << counters.durationInSecAPSPNLevelComputation << std::endl;
            std::cout << "    Level traversal (summed worker seconds): "
                      << counters.durationInSecAPSPNLevelTraversal << std::endl;
            std::cout << "    Forward traversal visited nodes count: " << counters.fwd_visit_count
                      << ", proportion: " << std::fixed << std::setprecision(2)
                      << (double)counters.fwd_visit_count / counters.graph_trav_count << std::endl;
            std::cout << "    Backward traversal visited nodes count: " << counters.bwd_visit_count
                      << ", proportion: " << std::fixed << std::setprecision(2)
                      << (double)counters.bwd_visit_count / counters.graph_trav_count << std::endl;
        }

        if (algo_type == AlgorithmType::BC && params.verbose) {
            std::cout << "    BC forward visited nodes count: " << counters.bc_fwd_visit_count
                      << ", proportion: " << std::fixed << std::setprecision(2)
                      << (double)counters.bc_fwd_visit_count / (n * n) << std::endl;
            std::cout << "    BC backward visited nodes count: " << counters.bc_bwd_visit_count
                      << ", proportion: " << std::fixed << std::setprecision(2)
                      << (double)counters.bc_bwd_visit_count / (n * n) << std::endl;
        }

        // Convert internal results into the selected mode's final output.
        local_timer.Start();

        switch (algo_type) {
        case AlgorithmType::BC:
            // Adjacency symmetry was checked before entering the exact kernel.
            (*BC) /= 2;
            break;
        case AlgorithmType::GC: {
            // Traversal is finished: Pcount can now hold the masked RD
            // numerator. Each worker updates independent contiguous columns,
            // avoiding a further n-by-n RD temporary and dense adjacency mask.
#pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int64_t column = 0; column < n; ++column) {
                auto projection = P->col(column);
                auto numerator = Pcount.col(column);
                projection.array() /= numerator.array();
                numerator = RD->col(column);
                for (const int64_t neighbor : A[static_cast<size_t>(column)]) {
                    // Preserve the parent's multiplication semantics for
                    // nonfinite RD values, including infinity times zero.
                    numerator(neighbor) *= 0.0;
                }
                numerator(column) = 0.0;
                projection(column) = 1.0;
                projection = numerator.array() / projection.array();
            }
            const double num_nonadj_pairs = n * (n - 1) - static_cast<double>(m);
            // Keep the original full-matrix Eigen summation order. Only the
            // independent elementwise preparation above has been parallelized.
            GC = P->array().sum() / num_nonadj_pairs;
            break;
        }
        case AlgorithmType::PHD: {
            // Parallelize columns while retaining each column's sum order.
#pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                (*PHD)(i) = (Pcount.col(i).sum() - 1) / (n - 1); // exclude self-loop
            }
            break;
        }
        }

        local_timer.Stop();
        counters.durationInSecPostprocess = local_timer.Seconds();
        parallel_cc::memory::UpdateActivePeak();
        if (params.verbose) {
            std::cout << "@@ Post processing costs: " << local_timer.Seconds() << " seconds"
                      << std::endl;
        }

        // Stop before packaging or copying the computed result.
        global_timer.Stop();
        const double exact_seconds = global_timer.Seconds();
        counters.durationInSecTotal = exact_seconds;
        if (params.verbose) {
            std::cout << "Overall memoized sources during APSPN: " << counters.mem_util_count
                      << ", avg per source: " << std::fixed << std::setprecision(1)
                      << (double)counters.mem_util_count / n
                      << ", avg node degree: " << (double)m / n << std::endl;
            std::cout << "APSPN memoization constraint: " << counters.apspn_mem_con << std::endl;
            std::cout << "Graph traversal nodes: " << counters.graph_trav_count
                      << ", proportion: " << std::fixed << std::setprecision(2)
                      << (double)(counters.graph_trav_count) / (n * n) << std::endl;
            std::cout << "Characteristic path length (CPL): " << std::fixed << std::setprecision(2)
                      << DistanceColumnSums(D).sum() / (n * (n - 1)) << std::endl;
        }
        PrintTotalComputationTime(counters.durationInSecTotal, params.verbose);

        // Return the selected exact-mode result.
        switch (algo_type) {
        case AlgorithmType::APSPN:
            return Result{std::move(Pcount), counters, exact_seconds};
        case AlgorithmType::BC:
            return Result{BC.value(), counters, exact_seconds};
        case AlgorithmType::GC:
            return Result{GC.value(), counters, exact_seconds};
        case AlgorithmType::PHD:
            return Result{PHD.value(), counters, exact_seconds};
        case AlgorithmType::APSLP:
            return Result{std::move(APSLP.value()), counters, exact_seconds};
        default:
            throw std::runtime_error("Unknown problem mode.");
        }
    };

    Result result{Eigen::MatrixXd{}, COUNTERS{}, 0.0};
    switch (distance_result.distances.index()) {
    case 0:
        throw std::runtime_error("Compact distance matrix is unavailable.");
    case 1:
        result = run_exact(std::get<DistMat<uint8_t>>(distance_result.distances));
        break;
    case 2:
        result = run_exact(std::get<DistMat<uint16_t>>(distance_result.distances));
        break;
    case 3:
        result = run_exact(std::get<DistMat<uint32_t>>(distance_result.distances));
        break;
    case 4:
        result = run_exact(std::get<DistMat<uint64_t>>(distance_result.distances));
        break;
    case 5:
        result = run_exact(std::get<DistMat<double>>(distance_result.distances));
        break;
    default:
        throw std::runtime_error("Unknown compact distance matrix type.");
    }
    return result;
}

#if defined(PYTHON_BUILD) || defined(MATLAB_BUILD)
namespace {
// Boundary validation only: symmetry reuse in GC/APSLP assumes undirected
// reference weights. Match allclose(RD, RD.T, rtol=1e-12, atol=1e-12),
// including both directions of its relative tolerance; do not alter RD.
void ValidateReferenceDistances(const double* rd, const int64_t n) {
    constexpr double tolerance = 1e-12;
    for (int64_t column = 0; column < n; ++column) {
        for (int64_t row = 0; row <= column; ++row) {
            const double forward = rd[row + column * n];
            const double reverse = rd[column + row * n];
            if (!std::isfinite(forward) || !std::isfinite(reverse)) {
                throw std::invalid_argument("Reference-distance matrix must contain only finite values.");
            }
            const double limit = tolerance + tolerance *
                (std::min)(std::abs(forward), std::abs(reverse));
            if (std::abs(forward - reverse) > limit) {
                throw std::invalid_argument(
                    "Reference-distance matrix must be symmetric (rtol=atol=1e-12).");
            }
        }
    }
}
}  // namespace
#endif

// Python-callable interface (for ctypes)
#ifdef PYTHON_BUILD

namespace {
// Keep error and counter retrieval local to the calling Python thread.
thread_local std::string python_last_error;
thread_local int64_t python_effective_threads = 0;

struct PythonGraphInput {
    std::vector<std::vector<int64_t>> adjacency;
    Eigen::VectorXd degrees;
};

PythonGraphInput ParsePythonGraph(
        const double* dense_adjacency,
        const int64_t* csr_indptr,
        const int64_t* csr_indices,
        const int64_t n,
        const int64_t nnz,
        const bool require_undirected) {
    if (n < 2 || n > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("Input graph must contain 2 through UINT32_MAX nodes.");
    }
    // Exact distances and the approximate compatibility path can allocate n*n.
    if (n > std::numeric_limits<Eigen::Index>::max() / n ||
        static_cast<uint64_t>(n) >
            std::numeric_limits<size_t>::max() / sizeof(double) / static_cast<uint64_t>(n)) {
        throw std::invalid_argument("Graph dimensions exceed the addressable matrix size.");
    }
    if (nnz < 0 || (dense_adjacency && (csr_indptr || csr_indices || nnz != 0)) ||
        (!dense_adjacency && (!csr_indptr || (nnz > 0 && !csr_indices)))) {
        throw std::invalid_argument("Provide either column-major dense adjacency or valid CSR arrays, not both.");
    }

    PythonGraphInput graph;
    graph.adjacency.resize(static_cast<size_t>(n));
    graph.degrees.resize(n);
    if (dense_adjacency) {
        // CSR and dense inputs both produce sorted, row-oriented neighbors.
        // Exact modes are undirected, so this also matches MATLAB's columns.
        for (int64_t row = 0; row < n; ++row) {
            auto& neighbors = graph.adjacency[static_cast<size_t>(row)];
            for (int64_t column = 0; column < n; ++column) {
                const double value = dense_adjacency[row + column * n];
                if (!std::isfinite(value) || (value != 0.0 && value != 1.0)) {
                    throw std::invalid_argument("Adjacency matrix must contain only finite binary values 0 or 1.");
                }
                if (value != 0.0) neighbors.push_back(column);
            }
            graph.degrees(row) = static_cast<double>(neighbors.size());
        }
    } else {
        if (csr_indptr[0] != 0 || csr_indptr[n] != nnz) {
            throw std::invalid_argument("CSR indptr must start at zero and end at nnz.");
        }
        for (int64_t row = 0; row < n; ++row) {
            const int64_t begin = csr_indptr[row];
            const int64_t end = csr_indptr[row + 1];
            if (begin < 0 || end < begin || end > nnz) {
                throw std::invalid_argument("CSR indptr must be nondecreasing and within [0, nnz].");
            }
            auto& neighbors = graph.adjacency[static_cast<size_t>(row)];
            neighbors.reserve(static_cast<size_t>(end - begin));
            for (int64_t position = begin; position < end; ++position) {
                const int64_t column = csr_indices[position];
                if (column < 0 || column >= n) {
                    throw std::invalid_argument("CSR column index is outside the graph.");
                }
                neighbors.push_back(column);
            }
            std::sort(neighbors.begin(), neighbors.end());
            if (std::adjacent_find(neighbors.begin(), neighbors.end()) != neighbors.end()) {
                throw std::invalid_argument("CSR input must not contain duplicate column indices.");
            }
            graph.degrees(row) = static_cast<double>(neighbors.size());
        }
    }

    if (require_undirected) {
        for (int64_t row = 0; row < n; ++row) {
            for (const int64_t column : graph.adjacency[static_cast<size_t>(row)]) {
                const auto& reverse = graph.adjacency[static_cast<size_t>(column)];
                if (!std::binary_search(reverse.begin(), reverse.end(), row)) {
                    throw std::invalid_argument("Only undirected graphs are allowed (adjacency must be symmetric).");
                }
            }
        }
    }
    return graph;
}
}  // namespace

extern "C" {

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

/**
 * @brief Parallel Python C ABI. All dense matrices use column-major order.
 *
 * @param algo_type_str Problem mode string: "APSPN", "BC", "GC", "GC_APPROX", "BC_APPROX", "PHD", or "APSLP"
 * @param dense_adjacency Binary adjacency, n*n doubles, or NULL for CSR input.
 * @param csr_indptr CSR row offsets, n+1 int64 elements, or NULL for dense input.
 * @param csr_indices CSR column indices, nnz int64 elements; explicit zeros have
 *                    already been removed and nonzero values validated as one.
 * @param n Number of nodes
 * @param nnz Number of CSR entries; zero for dense input.
 * @param num_threads Requested workers for exact modes; approximate modes use one.
 * @param rd_matrix_data Flat array of referential distance matrix (NULL if not needed)
 * @param output_data Output array allocated by caller
 * @param elapsed_time Output parameter for computation time
 * @param sampling_factor Sampling factor for approximate GC.
 * @param delta Confidence parameter for approximate BC.
 * @param err Error parameter for approximate BC.
 * @param emp_peeling_a Empirical peeling parameter for approximate BC.
 * @param memoization_depth_str Optional memoization depth string for approximate BC; pass NULL to use default
 * @param seed Random seed for approximate BC (zero is a valid explicit seed).
 * @param verbose Verbose flag for all algorithms
 * @return 0 on success, negative error code on failure
 */
EXPORT int cc_apspn_kernel_par_py(
    const char* algo_type_str,
    const double* dense_adjacency,
    const int64_t* csr_indptr,
    const int64_t* csr_indices,
    int64_t n,
    int64_t nnz,
    int64_t num_threads,
    const double* rd_matrix_data,
    double* output_data,
    double* elapsed_time,
    double sampling_factor,
    double delta,
    double err,
    double emp_peeling_a,
    const char* memoization_depth_str,
    uint32_t seed,
    int verbose)
{
    python_last_error.clear();
    python_effective_threads = 0;
    try {
        if (!algo_type_str || !output_data || !elapsed_time) {
            throw std::invalid_argument("Mode, output buffer and elapsed-time pointer must not be null.");
        }
        *elapsed_time = 0.0;
        if (num_threads < 1 || num_threads > static_cast<int64_t>(GetPhysicalCoreCount()) ||
            num_threads > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("num_threads must be an integer from 1 through the physical-core count.");
        }
        // Parse problem mode
        std::string algo_str(algo_type_str);
        std::transform(algo_str.begin(), algo_str.end(), algo_str.begin(),
            [](unsigned char value) { return static_cast<char>(std::toupper(value)); });

        AlgorithmType algo_type;
        if (algo_str == "APSPN") algo_type = AlgorithmType::APSPN;
        else if (algo_str == "BC") algo_type = AlgorithmType::BC;
        else if (algo_str == "GC") algo_type = AlgorithmType::GC;
        else if (algo_str == "GC_APPROX") algo_type = AlgorithmType::GC_approx;
        else if (algo_str == "BC_APPROX") algo_type = AlgorithmType::BC_approx;
        else if (algo_str == "PHD") algo_type = AlgorithmType::PHD;
        else if (algo_str == "APSLP") algo_type = AlgorithmType::APSLP;
        else throw std::invalid_argument("Unknown mode. Supported: APSPN, BC, GC, GC_APPROX, BC_APPROX, PHD, APSLP.");

        const bool is_approximate = algo_type == AlgorithmType::GC_approx ||
                                    algo_type == AlgorithmType::BC_approx;
        PythonGraphInput graph = ParsePythonGraph(
            dense_adjacency, csr_indptr, csr_indices, n, nnz,
            algo_type != AlgorithmType::BC_approx);

        // Prepare algorithm parameters
        AlgorithmParams params;
        params.verbose = (verbose != 0);
        if (algo_type == AlgorithmType::GC ||
            algo_type == AlgorithmType::GC_approx ||
            algo_type == AlgorithmType::APSLP) {
            if (rd_matrix_data == nullptr) {
                throw std::invalid_argument("GC, GC_APPROX and APSLP require an n-by-n referential-distance matrix.");
            }
            ValidateReferenceDistances(rd_matrix_data, n);
            if (algo_type == AlgorithmType::GC_approx) {
                Eigen::Map<const Eigen::MatrixXd> rd_matrix_map(rd_matrix_data, n, n);
                params.rd_matrix = rd_matrix_map;
            } else {
                params.rd_data = rd_matrix_data;
            }
        }
        if (algo_type == AlgorithmType::GC_approx) {
            if (!std::isfinite(sampling_factor) || sampling_factor <= 0.0 ||
                sampling_factor > static_cast<double>(n)) {
                throw std::invalid_argument("GC_APPROX sampling_factor must be finite and in (0, n].");
            }
            params.sampling_factor = sampling_factor;
        }
        if (algo_type == AlgorithmType::BC_approx) {
            if (!std::isfinite(delta) || delta <= 0.0 || delta >= 1.0 ||
                !std::isfinite(err) || err <= 0.0 || err >= 1.0 ||
                !std::isfinite(emp_peeling_a) || emp_peeling_a <= 0.0) {
                throw std::invalid_argument("BC_APPROX requires 0 < delta, err < 1 and finite emp_peeling_a > 0.");
            }
            params.delta = delta;
            params.err = err;
            params.emp_peeling_a = emp_peeling_a;
            if (memoization_depth_str != nullptr) {
                params.memoization_depth = std::string(memoization_depth_str);
            }
            params.seed = seed;
        }

        // Keep the settled algorithm bodies unchanged. Exact CSR input goes
        // straight to adjacency lists; the sequential approximate interfaces
        // still require an owning dense Eigen matrix in Python.
        Eigen::MatrixXd adj_matrix;
        if (is_approximate) {
            adj_matrix = Eigen::MatrixXd::Zero(n, n);
            for (int64_t row = 0; row < n; ++row) {
                for (const int64_t column : graph.adjacency[static_cast<size_t>(row)]) {
                    adj_matrix(row, column) = 1.0;
                }
            }
        }
        omp_set_dynamic(0);
        omp_set_num_threads(is_approximate ? 1 : static_cast<int>(num_threads));

        Result result;
        if (algo_type == AlgorithmType::GC_approx) {
            result = CcApproxGc(adj_matrix, params);
        } else if (algo_type == AlgorithmType::BC_approx) {
            result = CcApproxBc(std::variant<const mxArray*, const Eigen::MatrixXd*>(&adj_matrix), params);
        } else {
            // Python input preparation is outside the same kernel timer.
            result = CcApspnKernelExact(graph.adjacency, graph.degrees, algo_type, params, num_threads);
        }

        // Copy output based on problem mode
        switch (algo_type) {
            case AlgorithmType::APSPN:
            case AlgorithmType::APSLP: {
                const auto& matrix_result = std::get<Eigen::MatrixXd>(result.result);
                Eigen::Map<Eigen::MatrixXd>(output_data, n, n) = matrix_result;
                break;
            }
            case AlgorithmType::BC:
            case AlgorithmType::BC_approx:
            case AlgorithmType::PHD: {
                const auto& vector_result = std::get<Eigen::VectorXd>(result.result);
                Eigen::Map<Eigen::VectorXd>(output_data, n) = vector_result;
                break;
            }
            case AlgorithmType::GC:
            case AlgorithmType::GC_approx: {
                *output_data = std::get<double>(result.result);
                break;
            }
        }

        *elapsed_time = result.counters.durationInSecTotal;
        python_effective_threads = is_approximate ? 1 : result.counters.effective_threads;
        return 0;  // Success
    }
    catch (const std::exception& error) {
        python_last_error = error.what();
        return -1;
    }
    catch (...) {
        python_last_error = "Unknown native error in parallel CC-APSPN.";
        return -1;
    }
}

EXPORT const char* cc_apspn_kernel_par_last_error() {
    return python_last_error.c_str();
}

EXPORT int64_t cc_apspn_kernel_par_effective_threads() {
    return python_effective_threads;
}

// Retain the old dense, single-worker ABI. Despite its old row-major comment,
// its Eigen maps already used column-major order; preserve that actual layout.
EXPORT int cc_apspn_kernel_py(
    const char* algo_type_str, const double* adj_matrix_data, int64_t n,
    const double* rd_matrix_data, double* output_data, double* elapsed_time,
    double sampling_factor, double delta, double err, double emp_peeling_a,
    const char* memoization_depth_str, uint32_t seed, int verbose) {
    return cc_apspn_kernel_par_py(
        algo_type_str, adj_matrix_data, nullptr, nullptr, n, 0, 1,
        rd_matrix_data, output_data, elapsed_time,
        sampling_factor > 0.0 ? sampling_factor : 1.0,
        delta > 0.0 ? delta : 0.05, err > 0.0 ? err : 0.01,
        emp_peeling_a > 0.0 ? emp_peeling_a : 2.0,
        memoization_depth_str, seed != 0 ? seed : 42, verbose);
}
}  // extern "C"
#endif // PYTHON_BUILD


// MATLAB MEX Interface

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
        mexErrMsgIdAndTxt("MyToolbox:unified:invalidInput", "Problem mode must be a string.");
    }

    char* algoTypeStr = mxArrayToString(prhs);
    static const std::map<std::string, AlgorithmType> algoMap = {
        {"APSPN", AlgorithmType::APSPN},
        {"BC", AlgorithmType::BC},
        {"GC", AlgorithmType::GC},
        {"GC_APPROX", AlgorithmType::GC_approx},
        {"BC_APPROX", AlgorithmType::BC_approx},
        {"PHD", AlgorithmType::PHD},
        {"APSLP", AlgorithmType::APSLP}
    };

    AlgorithmType result = parseEnumFromString(algoTypeStr, algoMap,
        "Unknown problem mode. Supported modes: APSPN, BC, GC, GC_APPROX, BC_APPROX, PHD, APSLP");
    mxFree(algoTypeStr);
    return result;
}

struct ParsedInputs {
    AlgorithmType algo_type;
    graph_utils::ParsedGraphInput graph;
    // Only the unchanged sequential approximate-GC path needs this matrix.
    Eigen::MatrixXd adjMatrix;
    Eigen::Index num_vertices = 0;
    int64_t num_threads;
    AlgorithmParams params;
};

int64_t parseThreadCount(const mxArray* value) {
    if (!mxIsDouble(value) || mxIsComplex(value) || mxGetNumberOfElements(value) != 1) {
        mexErrMsgIdAndTxt(
            "parallel_cc:cc_apspn_kernel_par:threads",
            "num_threads must be a real scalar integer.");
    }
    const double raw = mxGetScalar(value);
    const size_t physical_cores = GetPhysicalCoreCount();
    if (!std::isfinite(raw) || raw < 1.0 || std::floor(raw) != raw ||
        raw > static_cast<double>(physical_cores)) {
        mexErrMsgIdAndTxt(
            "parallel_cc:cc_apspn_kernel_par:threads",
            "num_threads must be an integer from 1 through the physical-core count.");
    }
    return static_cast<int64_t>(raw);
}

ParsedInputs parseInputArguments(int nrhs, const mxArray *prhs[]) {
    // Check minimum number of arguments
    if (nrhs < 3) {
        mexErrMsgIdAndTxt("MyToolbox:unified:nrhs",
                          "At least three inputs required (problem mode, adjacency matrix, num_threads).");
    }

    ParsedInputs inputs;

    // Parse the problem mode and requested worker count.
    inputs.algo_type = parseAlgorithmType(prhs[0]);
    inputs.num_threads = parseThreadCount(prhs[2]);

    if (inputs.algo_type != AlgorithmType::GC_approx) {
        inputs.graph = graph_utils::ParseGraphInputParallel(prhs[1], inputs.num_threads);
        inputs.num_vertices = static_cast<Eigen::Index>(inputs.graph.adjacency.size());
        if (!parallel_cc::matlab::IsUndirectedAdjacencyList(inputs.graph.adjacency)) {
            throw std::runtime_error("Only undirected graphs are allowed (adjacency must be symmetric).");
        }
    } else {
        // Approximate GC retains its original sequential input path.
        inputs.adjMatrix = parseSparseDenseMatrix(prhs[1]);
        inputs.num_vertices = inputs.adjMatrix.rows();
        if (!inputs.adjMatrix.isApprox(inputs.adjMatrix.transpose())) {
            mexErrMsgIdAndTxt("MyToolbox:unified:directedGraphNotAllowed",
                             "Only undirected graphs are allowed (adjacency must be symmetric).");
        }
    }
    if (inputs.num_vertices <= 1) {
        mexErrMsgIdAndTxt("MyToolbox:unified:invalidGraphSize",
                          "Input graph must contain at least 2 nodes.");
    }
    parallel_cc::memory::UpdateActivePeak();

    // Parse optional RD matrix for GC/GC_APPROX/APSLP modes
    int paramOffset = 3;
    if (inputs.algo_type == AlgorithmType::GC ||
        inputs.algo_type == AlgorithmType::GC_approx ||
        inputs.algo_type == AlgorithmType::APSLP) {
        if (nrhs < 4) {
            mexErrMsgIdAndTxt("MyToolbox:unified:nrhs",
                              "GC, GC_APPROX, and APSLP modes require RD matrix.");
        }
        if (mxGetM(prhs[3]) != inputs.num_vertices ||
            mxGetN(prhs[3]) != inputs.num_vertices) {
            mexErrMsgIdAndTxt("MyToolbox:unified:dimensionMismatch",
                              "RD matrix dimensions must match adjacency matrix.");
        }
        if (inputs.algo_type == AlgorithmType::GC_approx) {
            // Approximation keeps its original input representation.
            Eigen::MatrixXd rdMatrix = parseSparseDenseMatrix(prhs[3]);
            inputs.params = AlgorithmParams(rdMatrix);
        } else {
            if ((!mxIsDouble(prhs[3]) && !mxIsLogical(prhs[3])) ||
                mxIsComplex(prhs[3]) || mxGetNumberOfDimensions(prhs[3]) != 2) {
                throw std::runtime_error("RD must be a real double or logical matrix.");
            }
            if (mxIsDouble(prhs[3]) && !mxIsSparse(prhs[3])) {
                // The MATLAB input stays alive and unmodified until MEX returns.
                inputs.params.rd_data = mxGetPr(prhs[3]);
            } else {
                inputs.params.rd_matrix = parseSparseDenseMatrix(prhs[3]);
            }
        }
        const double* rd_data = inputs.params.rd_data != nullptr
            ? inputs.params.rd_data : inputs.params.rd_matrix->data();
        ValidateReferenceDistances(rd_data, inputs.num_vertices);
        paramOffset = 4;
    }

    // Parse the optional GC_APPROX sampling factor.
    if (inputs.algo_type == AlgorithmType::GC_approx) {
        if (nrhs > paramOffset && !mxIsEmpty(prhs[paramOffset])) {
            if (!mxIsScalar(prhs[paramOffset]) || !mxIsDouble(prhs[paramOffset])) {
                mexErrMsgIdAndTxt("MyToolbox:unified:invalidInput",
                                  "Sampling factor must be a scalar double.");
            }
            inputs.params.sampling_factor = mxGetScalar(prhs[paramOffset]);
            if (inputs.params.sampling_factor <= 0 ||
                inputs.params.sampling_factor > inputs.num_vertices) {
                mexErrMsgIdAndTxt("MyToolbox:unified:invalidInput",
                                  "Sampling factor must be in the range (0, n].");
            }
            paramOffset++;
        }
    }

    // Parse the optional verbosity flag.
    if (nrhs > paramOffset) {
        if (mxIsLogicalScalar(prhs[paramOffset])) {
            inputs.params.verbose = mxIsLogicalScalarTrue(prhs[paramOffset]);
        } else if (!mxIsEmpty(prhs[paramOffset])) {
            mexErrMsgIdAndTxt("MyToolbox:unified:invalidInput",
                              "Verbose flag must be a logical scalar or empty [].");
        }
    }

    return inputs;
}

void createOutputArguments(int nlhs, mxArray *plhs[], const Result& result,
                          AlgorithmType algo_type, Eigen::Index numVertices) {
    // Create main output based on problem mode
    switch(algo_type) {
        case AlgorithmType::GC:
        case AlgorithmType::GC_approx:
            try {
                double gc_result = std::get<double>(result.result);
                plhs[0] = mxCreateDoubleScalar(gc_result);
            } catch (const std::bad_variant_access&) {
                mexErrMsgIdAndTxt("MyToolbox:unified:unexpectedType",
                                  "GC/GC_APPROX mode should return a double value.");
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
        case AlgorithmType::BC_approx:
        case AlgorithmType::PHD:
            try {
                const auto& vector_result = std::get<Eigen::VectorXd>(result.result);
                plhs[0] = mxCreateDoubleMatrix(numVertices, 1, mxREAL);
                Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[0]), numVertices) = vector_result;
            } catch (const std::bad_variant_access&) {
                mexErrMsgIdAndTxt("MyToolbox:unified:unexpectedType",
                                  "BC/BC_APPROX/PHD mode should return a vector.");
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
            mexErrMsgIdAndTxt("MyToolbox:unified:unknownAlgorithm", "Unknown problem mode.");
    }

    // Optional outputs
    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleScalar(result.counters.durationInSecTotal);
    }

    if (nlhs > 2) {
        plhs[2] = CreateCountersStruct(result.counters);
    }
}

/**
 * @brief MEX interface function for unified implementation of CC-APSPN kernel, CC-ApproxGC and CC-ApproxBC
 *
 * This function provides a unified interface to the CC-APSPN kernel, CC-ApproxGC and CC-ApproxBC algorithms
 *
 * @param nlhs Number of output arguments (left-hand side)
 * @param plhs Array of output arguments
 * @param nrhs Number of input arguments (right-hand side)
 * @param prhs Array of input arguments
 *
 * INPUT PARAMETERS:
 * prhs[0] - Problem mode (string): One of
 *           - 'APSPN': All-Pairs Shortest Path Number
 *           - 'BC': Betweenness Centrality
 *           - 'GC': Network Geometrical Congruence
 *           - 'GC_APPROX': Approximate Network Geometrical Congruence
 *           - 'BC_APPROX': Approximate Betweenness Centrality
 *           - 'PHD': Path Hesitation Degree
 *           - 'APSLP': All-Pairs Shortest-Lightest Path
 *
 * prhs[1] - Adjacency matrix (double or logical matrix):
 *           Square matrix representing the graph structure.
 *           Can be sparse or dense format. Exact modes require a binary,
 *           undirected adjacency matrix; entries equal to one are edges.
 *           All problem modes except BC_APPROX require symmetric adjacency matrices
 *           (undirected graphs). BC_APPROX accepts directed or undirected graphs.
 *
 * prhs[2] - Number of threads (real integer scalar):
 *           Required for every mode. Exact modes use up to this many workers
 *           in the ordered source batches. GC_APPROX and BC_APPROX validate
 *           and report the value but retain their serial samplers.
 *
 * prhs[3] - (Mode-dependent) RD matrix or first BC_APPROX parameter:
 *           RD is required for GC, GC_APPROX and APSLP modes.
 *           Reference distance matrix with same dimensions as adjacency matrix.
 *
 * Subsequent mode-specific inputs preserve the integrated-kernel order:
 *           GC_APPROX: RD, sampling_factor, verbose
 *           BC_APPROX: delta, err, emp_peeling_a, memoization_depth, seed,
 *                      verbose
 *           exact GC/APSLP: RD, verbose
 *           other exact modes: verbose
 *
 *
 * OUTPUT PARAMETERS:
 * plhs[0] - Result:
 *           - APSPN: n×n matrix of shortest path counts
 *           - BC: n×1 vector of betweenness centrality values
 *           - GC: scalar value of geometrical congruence
 *           - GC_APPROX: scalar value of approximate geometrical congruence
 *           - BC_APPROX: n×1 vector of approximate betweenness centrality values
 *           - PHD: n×1 vector of path hesitation degrees
 *           - APSLP: n×n matrix of shortest-lightest path weights
 *
 * plhs[1] - (Optional) Computation time (double scalar):
 *           Total execution time in seconds
 *
 * plhs[2] - (Optional) Performance counters (struct)
 *
 * USAGE EXAMPLES:
 *
 * 1. APSPN:
 *    [Pcount, elapsed, counters] = cc_apspn_kernel_par('APSPN', adj_matrix, num_threads);
 *
 * 2. BC (Exact):
 *    [BC_vals, elapsed, counters] = cc_apspn_kernel_par('BC', adj_matrix, num_threads);
 *
 * 3. GC (Exact):
 *    [GC_val, elapsed, counters] = cc_apspn_kernel_par('GC', adj_matrix, num_threads, rd_matrix);
 *
 * 4. GC (Approximate):
 *    [GC_val, elapsed, counters] = cc_apspn_kernel_par('GC_APPROX', adj_matrix, num_threads, rd_matrix, sampling_factor);
 *    % sampling_factor: (Optional) Number of source nodes to sample (default: n)
 *
 * 5. BC (Approximate):
 *    [BC_vals, elapsed, counters] = cc_apspn_kernel_par('BC_APPROX', adj_matrix, num_threads, delta, err, emp_peeling_a, memoization_depth, seed, verbose);
 *    % delta: (Optional) Confidence parameter (default: 0.05)
 *    % err: (Optional) Error parameter (default: 0.01)
 *    % emp_peeling_a: (Optional) Peeling parameter (default: 2.0)
 *    % memoization_depth: (Optional) 'none', 'avg_deg', 'log_n', 'log_n_w', etc. (default: 'log_n_w')
 *    % seed: (Optional) Random seed (default: 42)
 *    % verbose: (Optional) Print progress (default: true)
 *
 * 6. PHD:
 *    [PHD_vals, elapsed, counters] = cc_apspn_kernel_par('PHD', adj_matrix, num_threads);
 *
 * 7. APSLP:
 *    [Weights, elapsed, counters] = cc_apspn_kernel_par('APSLP', adj_matrix, num_threads, rd_matrix);
 *
 * @author Junda
 */

/**
 * @brief Parse parameters for the BC_APPROX mode from MATLAB inputs.
 */
AlgorithmParams parseBcApproxParams(int nrhs, const mxArray *prhs[]) {
    AlgorithmParams params;

    // Parameters start after problem mode, adjacency, and num_threads.
    int paramOffset = 3;

    // Parse delta (optional, default: 0.05)
    if (nrhs > paramOffset && !mxIsEmpty(prhs[paramOffset])) {
        params.delta = mxGetScalar(prhs[paramOffset]);
    }
    paramOffset++;

    // Parse err (optional, default: 0.01)
    if (nrhs > paramOffset && !mxIsEmpty(prhs[paramOffset])) {
        params.err = mxGetScalar(prhs[paramOffset]);
    }
    paramOffset++;

    // Parse emp_peeling_a (optional, default: 2.0)
    if (nrhs > paramOffset && !mxIsEmpty(prhs[paramOffset])) {
        params.emp_peeling_a = mxGetScalar(prhs[paramOffset]);
    }
    paramOffset++;

    // Parse memoization_depth (optional, default: "log_n_w")
    if (nrhs > paramOffset && !mxIsEmpty(prhs[paramOffset])) {
        if (mxIsChar(prhs[paramOffset])) {
            char* buffer = mxArrayToString(prhs[paramOffset]);
            if (buffer) {
                params.memoization_depth = std::string(buffer);
                mxFree(buffer);
            }
        }
    }
    paramOffset++;

    // Parse seed (optional, default: 42)
    if (nrhs > paramOffset && !mxIsEmpty(prhs[paramOffset])) {
        params.seed = static_cast<uint32_t>(mxGetScalar(prhs[paramOffset]));
    }
    paramOffset++;

    // Parse verbose parameter (optional, default: true)
    if (nrhs > paramOffset) {
        if (mxIsLogicalScalar(prhs[paramOffset])) {
            params.verbose = mxIsLogicalScalarTrue(prhs[paramOffset]);
        }
    }

    return params;
}

// Entry point for the MEX-file
void mexFunction(int64_t nlhs, mxArray *plhs[], int64_t nrhs, const mxArray *prhs[]) {
    try {
        parallel_cc::memory::PeakTracker memory_tracker;
        memory_tracker.CaptureBaseline();
        parallel_cc::memory::ScopedPeakTracker tracker_scope(&memory_tracker);

        // Check minimum number of arguments first
        if (nrhs < 3) {
            mexErrMsgIdAndTxt("MyToolbox:unified:nrhs",
                              "At least three inputs required (problem mode, adjacency matrix, num_threads).");
        }

        // Parse problem mode first to decide parsing strategy
        AlgorithmType algo_type = parseAlgorithmType(prhs[0]);
        const int64_t requested_threads = parseThreadCount(prhs[2]);
        const bool is_approximate = algo_type == AlgorithmType::BC_approx ||
                                    algo_type == AlgorithmType::GC_approx;
        if (is_approximate) {
            if (requested_threads > 1) {
                mexWarnMsgIdAndTxt("parallel_cc:cc_apspn_kernel_par:approximateThreads",
                    "Multithreading is not supported for BC_APPROX or GC_APPROX; using 1 thread.");
            }
        } else {
            // Only exact modes use the parallel implementation.
            parallel_cc::matlab::ProtectVcompMexLifetime(requested_threads);
            omp_set_dynamic(0);
            omp_set_num_threads(static_cast<int>(requested_threads));
        }

        Result result;

        // Branch based on problem mode
        if (algo_type == AlgorithmType::BC_approx) {
            // BC_APPROX uses sparse parsing to avoid materializing a dense matrix.
            const mxArray* adj_matrix_mx = prhs[1];
            if (mxGetM(adj_matrix_mx) <= 1) {
                mexErrMsgIdAndTxt("MyToolbox:unified:invalidGraphSize",
                                  "Input graph must contain at least 2 nodes.");
            }
            AlgorithmParams params = parseBcApproxParams(nrhs, prhs);

            result = CcApproxBc(std::variant<const mxArray*, const Eigen::MatrixXd*>(adj_matrix_mx), params);
        } else {
            // Prepare inputs before entering the timed algorithm.
            ParsedInputs inputs = parseInputArguments(nrhs, prhs);

            // Run the appropriate algorithm (exact or approximate)
            if (inputs.algo_type == AlgorithmType::GC_approx) {
                // Call approximate GC algorithm
                result = CcApproxGc(inputs.adjMatrix, inputs.params);
            } else {
                result = CcApspnKernelExact(
                    inputs.graph.adjacency, inputs.graph.degrees,
                    inputs.algo_type, inputs.params, inputs.num_threads);
            }
        }

        memory_tracker.UpdatePeak();
        result.counters.requested_threads = requested_threads;
        if (is_approximate) {
            result.counters.effective_threads = 1;
        }
        parallel_cc::memory::FillCounters(result.counters, memory_tracker);

        // Keep the algorithm's runtime unchanged. MATLAB allocation and copying
        // return the computed result only after the kernel timer has stopped.
        createOutputArguments(nlhs, plhs, result, algo_type, mxGetM(prhs[1]));

    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("MyToolbox:unified:exception", e.what());
    }
}

#endif // MATLAB_BUILD
