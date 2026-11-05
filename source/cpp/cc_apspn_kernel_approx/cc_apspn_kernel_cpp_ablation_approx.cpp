//
// Created by Junda on 2024/4/4.
//

#include "cc_apspn_kernel_ablation_approx.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <map>
#include "string"
#include <cassert>
#include <iomanip>
#include <numeric>
#include <variant>
#include <optional>
#include "iostream"
#include "random"
#include "../../common_include/cc_bfs_apsp_l_cpp.hpp"
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
            "durationInSecCcBbBFS"
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

    return matlabStruct;
}
#endif // MATLAB_BUILD

void PrioritizeSourceNodes(
        std::vector<int64_t>& source_list,
        const Eigen::MatrixXd& D,
        const Eigen::VectorXd& DEG,
        const std::string& pri_criteria) {
    // prioritize source nodes
    if (pri_criteria == "deg") {
        std::sort(source_list.begin(), source_list.end(), [&](int64_t a, int64_t b) {
            return DEG[a] > DEG[b];
        });
    } else if (pri_criteria == "closeness")
    {
        auto sum_dist = D.colwise().sum();
        std::sort(source_list.begin(), source_list.end(), [&](int64_t a, int64_t b) {
            return sum_dist[a] < sum_dist[b];
        });
    } else if (pri_criteria == "deg-closeness")
    {
        auto sum_dist = D.colwise().sum().transpose();  // Convert to column vector
        Eigen::VectorXd deg_closeness = (DEG.array() / (sum_dist.array() + 1e-10)).matrix();
        std::sort(source_list.begin(), source_list.end(), [&](int64_t a, int64_t b) {
            return deg_closeness[a] > deg_closeness[b];
        });
    } else if (pri_criteria == "mean-depth") {
        auto mean_depth = D.colwise().mean();
        std::sort(source_list.begin(), source_list.end(), [&](int64_t a, int64_t b) {
            return mean_depth[a] < mean_depth[b];
        });
    } else if (pri_criteria == "max-depth") {
        auto max_depth = D.colwise().maxCoeff();
        std::sort(source_list.begin(), source_list.end(), [&](int64_t a, int64_t b) {
            return max_depth[a] < max_depth[b];
        });
    } else if (pri_criteria == "random") {
        std::random_device rd;  // Obtain a random number from hardware
        std::mt19937 rng(rd()); // Seed the generator
        std::shuffle(source_list.begin(), source_list.end(), rng);
    } else {
        throw std::runtime_error("Unrecognized prioritization criteria.");
    }
}

std::vector<std::vector<int64_t>> BuildSpLevels(
        const int64_t n,
        const int64_t s,
        const int64_t max_depth,
        const Eigen::VectorXd& D_s) {
    std::vector<std::vector<int64_t>> levels(max_depth + 1);
    for (auto& level : levels) {
        level.reserve(n / (max_depth + 1));
    }
    for (int64_t i = 0; i < n; ++i) {
        levels[static_cast<int64_t>(D_s(i))].push_back(i);
    }
    return levels;
}


void DistanceGuidedSss(
        const std::vector<std::vector<int64_t>>& A,
        const int64_t n,
        const int64_t s,
        const int64_t mem_con,
        const Eigen::MatrixXd& D,
        Eigen::MatrixXd& Pcount,
        const Eigen::VectorXd& eccs,
        const AlgorithmType& algo_type,
        std::optional<Eigen::MatrixXd>& P,
        std::optional<Eigen::VectorXd>& BC,
        std::optional<Eigen::MatrixXd>& APSLP,
        const std::optional<Eigen::MatrixXd>& RD,
        const std::vector<char>& processed,
        const AblationType& ablation_type,
        COUNTERS& counters) {
    //////////////////////////////
    Timer local_timer;
    local_timer.Start();

    auto D_s = D.col(s);
    auto Pcount_s = Pcount.col(s);
    bool use_symm = ablation_type != AblationType::Symm;

    Eigen::VectorXd Pcount_s_mem = Pcount_s;  // memoized Pcount_s
    Pcount_s(s) = 1;
    // initialize optional variables
    std::optional<Eigen::Ref<Eigen::VectorXd>> P_s;
    std::optional<Eigen::VectorXd> P_s_mem;
    std::optional<Eigen::Ref<Eigen::VectorXd>> SSSLP;  // single-source shortest-lightest path

    switch (algo_type) {
    case AlgorithmType::BC:
        assert(BC.has_value());
        break;
    case AlgorithmType::GC:
        assert(P.has_value());
        assert(RD.has_value());
        P_s = (*P).col(s);
        P_s_mem = P_s.value();
        break;
    case AlgorithmType::APSLP:
        assert(APSLP.has_value());
        assert(RD.has_value());
        SSSLP = (*APSLP).col(s);
        (*SSSLP)(s) = 0;
        break;
    }

    local_timer.Stop();
    counters.durationInSecAPSPNInit += local_timer.Seconds();

    //////////////////////////////
    local_timer.Start();
    // Build SP levels
    const int64_t max_depth = static_cast<int64_t>(eccs(s));
    auto levels = BuildSpLevels(n, s, max_depth, D_s);
    local_timer.Stop();
    counters.durationInSecAPSPNLevelComputation += local_timer.Seconds();

    // main loop
    local_timer.Start();

    std::vector<char> frontier_mask(n, false);  // nodes in current and past frontier (for graph traversal)
    std::vector<char> level_mask(n, false);  // nodes in levels that are explored or being explored
    frontier_mask[s] = true;

    // handle first level
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
    // find memoized sources
    std::vector<int64_t> mem_sources;
    if (ablation_type != AblationType::Mem && A[s].size() <= mem_con)
    {
        for (const int64_t t : A[curr])
        {
            if (processed[t]) // previously processed neighbor can replay its path counts
            {
                mem_sources.push_back(t);
            }
        }
    }
    // perform all-paths memoization
    if (!mem_sources.empty())
    {
        // unmask mem_source
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
                    if (D_ms(i) == curr_depth) {  // shortest path from s to i pass through ps
                        Pcount_s_mem(i) += Pcount_ms(i);
                        switch (algo_type) {
                        case AlgorithmType::GC:
                            (*P_s_mem)[i] += (*P).col(ms)(i) + (*P_s)(ms) * Pcount_ms(i);
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

    // perform graph traversal
    for (int64_t curr_depth = 1; curr_depth < max_depth; ++curr_depth) {
        const auto& curr_level = levels[curr_depth];
        const auto& dst_level = levels[curr_depth + 1];  // destination level
        // mask nodes at the current level
        for (auto& curr : curr_level) {
            level_mask[curr] = true;
        }
        // computes forward and backward cost
        int64_t fwd_cost = 0;
        int64_t bwd_cost = 0;

        for (const int64_t curr : curr_level) {
            if (!frontier_mask[curr]) continue;
            fwd_cost += A[curr].size();
        }
        if (fwd_cost == 0) continue; // no more frontier nodes
        for (const int64_t t : dst_level) {
            if (use_symm && processed[t]) continue;
            bwd_cost += A[t].size();
        }
        if (ablation_type == AblationType::DAP) {
            bwd_cost = 2*fwd_cost; // disable DAP
        }
        if (fwd_cost < bwd_cost) { // forward traversal is cheaper
            for (const int64_t curr : curr_level) {
                if (!frontier_mask[curr]) continue;
                counters.graph_trav_count++;
                counters.fwd_visit_count++;
                const double curr_pcount = Pcount_s(curr);
                for (const int64_t t : A[curr]) {
                    if (level_mask[t]) continue;
                    // t is at the next level
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
                if (use_symm && processed[curr])  // apply graph symmetry if possible
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
    //////////////////////////////

    // combine the results
    Pcount_s += Pcount_s_mem;
    switch (algo_type) {
        case AlgorithmType::GC:
            (*P_s) += (*P_s_mem);
            break;
        case AlgorithmType::BC:
            {
                Eigen::VectorXd BC_s = Eigen::VectorXd::Zero(n);
                // Traverse SP levels in reversed order to propagate BC dependencies
                for (int64_t depth = max_depth; depth >= 2; depth--)
                {
                    int64_t bwd_cost = 0;
                    int64_t fwd_cost = 0;
                    for (int64_t u : levels[depth])
                    {
                        bwd_cost += A[u].size();
                    }
                    for (int64_t u : levels[depth - 1])
                    {
                        fwd_cost += A[u].size();
                    }
                    if (bwd_cost < fwd_cost)
                    {
                        // backward propagation is cheaper
                        for (int64_t u : levels[depth])
                        {
                            counters.bc_bwd_visit_count++;
                            for (int64_t v : A[u])
                            {
                                if (D_s(v) == D_s(u) - 1)
                                {
                                    BC_s(v) += (Pcount_s(v) / Pcount_s(u)) * (1 + BC_s(u));
                                }
                            }
                        }
                    }
                    else
                    {
                        // forward propagation is cheaper
                        for (int64_t v : levels[depth - 1])
                        {
                            counters.bc_fwd_visit_count++;
                            for (int64_t u : A[v])
                            {
                                if (D_s(v) == D_s(u) - 1)
                                {
                                    // v is a valid predecessor of u in the SP-DAG
                                    BC_s(v) += (Pcount_s(v) / Pcount_s(u)) * (1 + BC_s(u));
                                }
                            }
                        }
                    }
                }
                // Add pairwise dependencies to BC
                for (int64_t curr = 0; curr < n; ++curr)
                {
                    if (curr != s)
                    {
                        (*BC)(curr) += BC_s(curr);
                    }
                }
                break;
            }
    }
    counters.mem_util_count += mem_sources.size();
}

// The exact algorithm implementation
Result CcApspnKernelExact(
        const Eigen::MatrixXd& adj_matrix,
        const AlgorithmType& algo_type,
        const AblationType& ablation_type,
        const AlgorithmParams& params,
        const std::string& pri_criteria) {

    /// computation starts
    COUNTERS counters;
    Timer global_timer;
    Timer local_timer;
    global_timer.Start();
    int64_t n = adj_matrix.rows();

    PrintAlgorithmHeader(algo_type, params);

    // declare ablation type
    if (params.verbose) {
        std::cout << "Ablation type: ";
        switch (ablation_type) {
            case AblationType::Pri:
                std::cout << "Prioritization ablation (Pri)" << std::endl;
                break;
            case AblationType::Mem:
                std::cout << "Memoization ablation (Mem)" << std::endl;
                break;
            case AblationType::Symm:
                std::cout << "Symmetry ablation (Symm)" << std::endl;
                break;
            case AblationType::DAP:
                std::cout << "Direction-Adaptive Propagation ablation (DAP)" << std::endl;
                break;
            case AblationType::None:
                std::cout << "No ablation (None)" << std::endl;
                break;
        }
    }

    /// compute degree and adjacency list
    local_timer.Start();
    auto [DEG, A] = ComputeDegreeAndAdjList(adj_matrix, n);
    local_timer.Stop();
    counters.durationInSecComputeAdjList = local_timer.Seconds();
    if (params.verbose) {
        std::cout << "@@ Degree and Adjacency list computation time: " << local_timer.Seconds() << std::endl;
    }

    /// initialize variables
    local_timer.Start();
    Eigen::MatrixXd D;
    Eigen::MatrixXd Pcount = Eigen::MatrixXd::Constant(n, n, 0);  // Number of shortest paths
    std::optional<Eigen::MatrixXd> RD;  // Referential distance
    std::optional<Eigen::MatrixXd> P;  // Average shortest path projection
    std::optional<Eigen::VectorXd> BC;  // Betweenness centrality
    std::optional<Eigen::VectorXd> PHD;  // Path-hesitation degree
    std::optional<Eigen::MatrixXd> APSLP; // All-pairs shortest-lightest paths
    std::optional<double> GC;  // Geometrical congruence

    std::vector<char> processed(n, false);
    std::vector<int64_t> source_list(n); std::iota(source_list.begin(), source_list.end(), 0);
    // Initialize variables based on problem mode
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

    local_timer.Stop();
    counters.durationInSecAPSPNInit = local_timer.Seconds();
    if (params.verbose) {
        std::cout << "@@ Initialization time: " << local_timer.Seconds() << " seconds" << std::endl;
    }

    /// compute APSP
    local_timer.Start();
    CC_BFS_APSP::RESULT result = CC_BFS_APSP::CC_BFS_APSP_L_FromAdjacency(
        A, DEG, "bounded-avg-deg");
    D = std::move(result.D);
    local_timer.Stop();
    counters.durationInSecAPSPComputation = local_timer.Seconds();
    if (params.verbose) {
        std::cout << "@@ APSP computation time: " << local_timer.Seconds() << " seconds" << std::endl;
    }

    /// prioritize source nodes
    if (ablation_type == AblationType::Pri) {
        PrioritizeSourceNodes(source_list, D, DEG, "random");
    } else {
        PrioritizeSourceNodes(source_list, D, DEG, pri_criteria);
    }
    int64_t m = DEG.sum();
    int64_t avg_deg = m / n;
    int64_t mem_con = std::floor(std::sqrt(avg_deg));
    counters.apspn_mem_con = mem_con;

    /// Compute eccentricities
    local_timer.Start();
    Eigen::VectorXd eccs = D.colwise().maxCoeff();
    local_timer.Stop();
    if (params.verbose) {
        std::cout << "@@ Eccentricity computation time: " << local_timer.Seconds() << " seconds" << std::endl;
    }

    /// APSPN main loop
    local_timer.Start();

    for (int64_t idx = 0; idx < source_list.size(); ++idx) {
        int64_t s = source_list[idx];
        DistanceGuidedSss(A, n, s, mem_con, D, Pcount, eccs, algo_type, P, BC, APSLP, RD, processed, ablation_type, counters);
        processed[s] = true;
    }
    local_timer.Stop();
    counters.durationInSecAPSPNComputation = local_timer.Seconds();


    if (params.verbose) {
        std::cout << "@@ APSPN computation costs: " << std::fixed << std::setprecision(2) << local_timer.Seconds() << std::endl;
        std::cout << "    Details: " << std::endl;
        std::cout << "    APSPN init: " << std::fixed << std::setprecision(2) << counters.durationInSecAPSPNInit
                  << ", " << std::fixed << std::setprecision(2) << (counters.durationInSecAPSPNInit / local_timer.Seconds() * 100)
                  << "%" << std::endl;
        std::cout << "    APSPN level computation: " << std::fixed << std::setprecision(2) << counters.durationInSecAPSPNLevelComputation
                  << ", " << std::fixed << std::setprecision(2) << (counters.durationInSecAPSPNLevelComputation / local_timer.Seconds() * 100)
                  << "%" << std::endl;
        std::cout << "    APSPN level traversal: " << std::fixed << std::setprecision(2) << counters.durationInSecAPSPNLevelTraversal
                  << ", " << std::fixed << std::setprecision(2) << (counters.durationInSecAPSPNLevelTraversal / local_timer.Seconds() * 100)
                  << "%" << std::endl;
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
                  << (double)counters.bc_fwd_visit_count / (n*n) << std::endl;
        std::cout << "    BC backward visited nodes count: " << counters.bc_bwd_visit_count
                  << ", proportion: " << std::fixed << std::setprecision(2)
                  << (double)counters.bc_bwd_visit_count / (n*n) << std::endl;
    }

    // post processing
    local_timer.Start();

    switch (algo_type) {
    case AlgorithmType::BC:
        // Divide by 2 for undirected graph
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
        break;
    }

    local_timer.Stop();
    if (params.verbose) {
        std::cout << "@@ Post processing costs: " << local_timer.Seconds() << " seconds" << std::endl;
    }

    /// computation ends
    global_timer.Stop();
    counters.durationInSecTotal = global_timer.Seconds();
    if (params.verbose) {
        std::cout << "Overall memoized sources during APSPN: " << counters.mem_util_count
                  << ", avg per source: " << std::fixed << std::setprecision(1)
                  << (double)counters.mem_util_count / n
                  << ", avg node degree: " << (double)m / n << std::endl;
        std::cout << "APSPN memoization constraint: " << counters.apspn_mem_con << std::endl;
        std::cout << "Graph traversal nodes: " << counters.graph_trav_count
                  << ", proportion: " << std::fixed << std::setprecision(2)
                  << (double)(counters.graph_trav_count) / (n*n) << std::endl;
        std::cout << "Characteristic path length (CPL): " << std::fixed << std::setprecision(2)
                  << D.sum() / (n*(n-1)) << std::endl;
    }
    PrintTotalComputationTime(counters.durationInSecTotal, params.verbose);

    /// Return the result
    switch (algo_type) {
    case AlgorithmType::APSPN:
        return Result{Pcount, counters,global_timer.Seconds()};
    case AlgorithmType::BC:
        return Result{BC.value(), counters,global_timer.Seconds()};
    case AlgorithmType::GC:
        return Result{GC.value(), counters, global_timer.Seconds()};
    case AlgorithmType::PHD:
        return Result{PHD.value(), counters,global_timer.Seconds()};
    case AlgorithmType::APSLP:
        return Result{APSLP.value(), counters, global_timer.Seconds()};
    default:
        throw std::runtime_error("Unknown problem mode.");
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
 * @brief Unified Python-callable wrapper for CC-APSPN-kernel algorithms
 *
 * @param algo_type_str Problem mode string: "APSPN", "BC", "GC", "GC_APPROX", "BC_APPROX", "PHD", or "APSLP"
 * @param ablation_type_str Ablation type string: "Pri", "Mem", "Symm", "DAP", or "None"
 * @param adj_matrix_data Flat array of adjacency matrix (row-major, n*n elements)
 * @param n Number of nodes
 * @param rd_matrix_data Flat array of referential distance matrix (NULL if not needed)
 * @param output_data Output array allocated by caller
 * @param elapsed_time Output parameter for computation time
 * @param sampling_factor Optional sampling factor for approximate GC; pass <= 0 to use default
 * @param delta Optional confidence parameter for approximate BC; pass <= 0 to use default
 * @param err Optional error parameter for approximate BC; pass <= 0 to use default
 * @param emp_peeling_a Optional empirical peeling parameter for approximate BC; pass <= 0 to use default
 * @param memoization_depth_str Optional memoization depth string for approximate BC; pass NULL to use default
 * @param seed Optional random seed for approximate BC; pass 0 to use default
 * @param verbose Verbose flag for all algorithms
 * @return 0 on success, negative error code on failure
 */
EXPORT int cc_apspn_kernel_py(
    const char* algo_type_str,
    const char* ablation_type_str,
    const double* adj_matrix_data,
    int64_t n,
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
    try {
        // Parse problem mode
        std::string algo_str(algo_type_str);
        std::transform(algo_str.begin(), algo_str.end(), algo_str.begin(), ::toupper);

        AlgorithmType algo_type;
        if (algo_str == "APSPN") algo_type = AlgorithmType::APSPN;
        else if (algo_str == "BC") algo_type = AlgorithmType::BC;
        else if (algo_str == "GC") algo_type = AlgorithmType::GC;
        else if (algo_str == "GC_APPROX") algo_type = AlgorithmType::GC_approx;
        else if (algo_str == "BC_APPROX") algo_type = AlgorithmType::BC_approx;
        else if (algo_str == "PHD") algo_type = AlgorithmType::PHD;
        else if (algo_str == "APSLP") algo_type = AlgorithmType::APSLP;
        else return -2;  // Invalid problem mode

        // Parse ablation type
        std::string ablation_str(ablation_type_str);
        std::transform(ablation_str.begin(), ablation_str.end(), ablation_str.begin(), ::toupper);

        AblationType ablation_type;
        if (ablation_str == "PRI") ablation_type = AblationType::Pri;
        else if (ablation_str == "MEM") ablation_type = AblationType::Mem;
        else if (ablation_str == "SYMM") ablation_type = AblationType::Symm;
        else if (ablation_str == "DAP") ablation_type = AblationType::DAP;
        else if (ablation_str == "NONE") ablation_type = AblationType::None;
        else return -4;  // Invalid ablation type

        // Convert adjacency matrix
        Eigen::Map<const Eigen::MatrixXd> adj_matrix_map(adj_matrix_data, n, n);
        Eigen::MatrixXd adj_matrix = adj_matrix_map;

        // Prepare algorithm parameters
        AlgorithmParams params;
        params.verbose = (verbose != 0);
        if (algo_type == AlgorithmType::GC ||
            algo_type == AlgorithmType::GC_approx ||
            algo_type == AlgorithmType::APSLP) {
            if (rd_matrix_data == nullptr) return -3;  // Missing RD matrix
            Eigen::Map<const Eigen::MatrixXd> rd_matrix_map(rd_matrix_data, n, n);
            Eigen::MatrixXd rd_matrix = rd_matrix_map;
            params = AlgorithmParams(rd_matrix);
            params.verbose = (verbose != 0);
        }
        if (algo_type == AlgorithmType::GC_approx && sampling_factor > 0.0) {
            if (sampling_factor > static_cast<double>(n)) return -5;  // Invalid sampling factor
            params.sampling_factor = sampling_factor;
        }
        if (algo_type == AlgorithmType::BC_approx) {
            if (delta > 0.0) {
                params.delta = delta;
            }
            if (err > 0.0) {
                params.err = err;
            }
            if (emp_peeling_a > 0.0) {
                params.emp_peeling_a = emp_peeling_a;
            }
            if (memoization_depth_str != nullptr) {
                params.memoization_depth = std::string(memoization_depth_str);
            }
            if (seed != 0) {
                params.seed = seed;
            }
        }

        // Call the algorithm
        std::string pri_criteria = "deg";
        Result result;
        if (algo_type == AlgorithmType::GC_approx) {
            result = CcApproxGc(adj_matrix, params);
        } else if (algo_type == AlgorithmType::BC_approx) {
            result = CcApproxBc(std::variant<const mxArray*, const Eigen::MatrixXd*>(&adj_matrix), params);
        } else {
            result = CcApspnKernelExact(adj_matrix, algo_type, ablation_type, params, pri_criteria);
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

AblationType parseAblationType(const mxArray* prhs) {
    if (!mxIsChar(prhs)) {
        mexErrMsgIdAndTxt("MyToolbox:unified:invalidInput", "Ablation type must be a string.");
    }

    char* ablationTypeStr = mxArrayToString(prhs);
    static const std::map<std::string, AblationType> ablationMap = {
        {"PRI", AblationType::Pri},
        {"MEM", AblationType::Mem},
        {"SYMM", AblationType::Symm},
        {"DAP", AblationType::DAP},
        {"NONE", AblationType::None}
    };

    AblationType result = parseEnumFromString(ablationTypeStr, ablationMap,
        "Unknown ablation type. Supported types: Pri, Mem, Symm, DAP, None");
    mxFree(ablationTypeStr);
    return result;
}

struct ParsedInputs {
    AlgorithmType algo_type;
    AblationType ablation_type;
    Eigen::MatrixXd adjMatrix;
    AlgorithmParams params;
    std::string pri_criteria;
};

ParsedInputs parseInputArguments(int nrhs, const mxArray *prhs[]) {
    // Check minimum number of arguments
    if (nrhs < 3) {
        mexErrMsgIdAndTxt("MyToolbox:unified:nrhs",
                          "At least three inputs required (problem mode, ablation type, adjacency matrix).");
    }

    ParsedInputs inputs;

    // Parse algorithm and ablation types
    inputs.algo_type = parseAlgorithmType(prhs[0]);
    inputs.ablation_type = parseAblationType(prhs[1]);

    // Parse adjacency matrix
    inputs.adjMatrix = parseSparseDenseMatrix(prhs[2]);
    Eigen::Index numVertices = inputs.adjMatrix.rows();
    if (numVertices <= 1) {
        mexErrMsgIdAndTxt("MyToolbox:unified:invalidGraphSize",
                          "Input graph must contain at least 2 nodes.");
    }

    // For all problem modes except BC_APPROX, only allow undirected graphs as input.
    // We treat a graph as undirected iff the adjacency matrix is symmetric.
    if (inputs.algo_type != AlgorithmType::BC_approx) {
        if (!inputs.adjMatrix.isApprox(inputs.adjMatrix.transpose())) {
            mexErrMsgIdAndTxt(
                    "MyToolbox:unified:directedGraphNotAllowed",
                    "For modes other than BC_APPROX, only undirected graphs are allowed (adjacency must be symmetric).");
        }
    }

    // Parse optional RD matrix for GC/GC_APPROX/APSLP modes
    int paramOffset = 3;
    if (inputs.algo_type == AlgorithmType::GC ||
        inputs.algo_type == AlgorithmType::GC_approx ||
        inputs.algo_type == AlgorithmType::APSLP) {
        if (nrhs < 4) {
            mexErrMsgIdAndTxt("MyToolbox:unified:nrhs",
                              "GC, GC_APPROX, and APSLP modes require RD matrix.");
        }
        Eigen::MatrixXd rdMatrix = parseSparseDenseMatrix(prhs[3]);
        if (rdMatrix.rows() != numVertices || rdMatrix.cols() != numVertices) {
            mexErrMsgIdAndTxt("MyToolbox:unified:dimensionMismatch",
                              "RD matrix dimensions must match adjacency matrix.");
        }
        inputs.params = AlgorithmParams(rdMatrix);
        paramOffset = 4;
    }

    // Parse sampling_factor for GC_approx (optional)
    if (inputs.algo_type == AlgorithmType::GC_approx) {
        if (nrhs > paramOffset && !mxIsEmpty(prhs[paramOffset])) {
            if (!mxIsScalar(prhs[paramOffset]) || !mxIsDouble(prhs[paramOffset])) {
                mexErrMsgIdAndTxt("MyToolbox:unified:invalidInput",
                                  "Sampling factor must be a scalar double.");
            }
            inputs.params.sampling_factor = mxGetScalar(prhs[paramOffset]);
            if (inputs.params.sampling_factor <= 0 || inputs.params.sampling_factor > numVertices) {
                mexErrMsgIdAndTxt("MyToolbox:unified:invalidInput",
                                  "Sampling factor must be in the range (0, n].");
            }
            paramOffset++;
        }
    }

    // Set default values
    inputs.pri_criteria = "deg";

    // Parse verbose parameter (optional)
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
        plhs[1] = mxCreateDoubleScalar(result.total_time);
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
 * prhs[1] - Ablation type (string): One of
 *           - 'Pri': Prioritization ablation
 *           - 'Mem': Memoization ablation
 *           - 'Symm': Symmetry ablation
 *           - 'DAP': Direction-Adaptive Propagation ablation
 *           - 'None': No ablation
 *
 * prhs[2] - Adjacency matrix (double or logical matrix):
 *           Square matrix representing the graph structure.
 *           Can be sparse or dense format. Nonzero entries are treated as edges.
 *           All problem modes except BC_APPROX require symmetric adjacency matrices
 *           (undirected graphs). BC_APPROX accepts directed or undirected graphs.
 *
 * prhs[3] - (Optional) RD matrix (double matrix):
 *           Required for GC, GC_APPROX and APSLP modes.
 *           Reference distance matrix with same dimensions as adjacency matrix.
 *
 * prhs[4] - (Optional) Verbose flag (logical scalar):
 *           Controls printing of progress and statistics to the console.
 *           Default: true
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
 *    [Pcount, elapsed, counters] = cc_apspn_kernel('APSPN', 'None', adj_matrix);
 *
 * 2. BC (Exact):
 *    [BC_vals, elapsed, counters] = cc_apspn_kernel('BC', 'None', adj_matrix);
 *
 * 3. GC (Exact):
 *    [GC_val, elapsed, counters] = cc_apspn_kernel('GC', 'None', adj_matrix, rd_matrix);
 *
 * 4. GC (Approximate):
 *    [GC_val, elapsed, counters] = cc_apspn_kernel('GC_APPROX', 'None', adj_matrix, rd_matrix, sampling_factor);
 *    % sampling_factor: (Optional) Number of source nodes to sample (default: n)
 *
 * 5. BC (Approximate):
 *    [BC_vals, elapsed, counters] = cc_apspn_kernel('BC_APPROX', 'None', adj_matrix, delta, err, emp_peeling_a, memoization_depth, seed, verbose);
 *    % delta: (Optional) Confidence parameter (default: 0.05)
 *    % err: (Optional) Error parameter (default: 0.01)
 *    % emp_peeling_a: (Optional) Peeling parameter (default: 2.0)
 *    % memoization_depth: (Optional) 'none', 'avg_deg', 'log_n', 'log_n_w', etc. (default: 'log_n_w')
 *    % seed: (Optional) Random seed (default: 42)
 *    % verbose: (Optional) Print progress (default: true)
 *
 * 6. PHD:
 *    [PHD_vals, elapsed, counters] = cc_apspn_kernel('PHD', 'None', adj_matrix);
 *
 * 7. APSLP:
 *    [Weights, elapsed, counters] = cc_apspn_kernel('APSLP', 'None', adj_matrix, rd_matrix);
 *
 * @author Junda
 * @date 2025/9/3
 * @version V3
 */

/**
 * @brief Parse parameters for the BC_APPROX mode from MATLAB inputs.
 */
AlgorithmParams parseBcApproxParams(int nrhs, const mxArray *prhs[]) {
    AlgorithmParams params;

    // Parameters start at index 3 (after algo_type, ablation_type, adj_matrix)
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
        // Check minimum number of arguments first
        if (nrhs < 3) {
            mexErrMsgIdAndTxt("MyToolbox:unified:nrhs",
                              "At least three inputs required (problem mode, ablation type, adjacency matrix).");
        }

        // Parse problem mode first to decide parsing strategy
        AlgorithmType algo_type = parseAlgorithmType(prhs[0]);

        Result result;

        // Branch based on problem mode
        if (algo_type == AlgorithmType::BC_approx) {
            // BC_approx: Use sparse parsing path to avoid materializing full matrix
            const mxArray* adj_matrix_mx = prhs[2];
            if (mxGetM(adj_matrix_mx) <= 1) {
                mexErrMsgIdAndTxt("MyToolbox:unified:invalidGraphSize",
                                  "Input graph must contain at least 2 nodes.");
            }
            AlgorithmParams params = parseBcApproxParams(nrhs, prhs);

            result = CcApproxBc(std::variant<const mxArray*, const Eigen::MatrixXd*>(adj_matrix_mx), params);

            // Create output arguments
            createOutputArguments(nlhs, plhs, result, algo_type, mxGetM(adj_matrix_mx));

        } else {
            // All other algorithms: Use dense matrix parsing path
            ParsedInputs inputs = parseInputArguments(nrhs, prhs);

            // Run the appropriate algorithm (exact or approximate)
            if (inputs.algo_type == AlgorithmType::GC_approx) {
                // Call approximate GC algorithm
                result = CcApproxGc(inputs.adjMatrix, inputs.params);
            } else {
                // Call exact algorithm
                result = CcApspnKernelExact(inputs.adjMatrix, inputs.algo_type, inputs.ablation_type,
                                            inputs.params, inputs.pri_criteria);
            }

            // Create output arguments
            createOutputArguments(nlhs, plhs, result, inputs.algo_type, inputs.adjMatrix.rows());
        }

    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("MyToolbox:unified:exception", e.what());
    }
}

#endif // MATLAB_BUILD
