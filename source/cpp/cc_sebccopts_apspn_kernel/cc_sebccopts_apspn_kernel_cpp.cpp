//
// Created by Junda on 2024/10/21.
//
// This version use CC-APSP compute the APSP for the reduced graph and each biconnected component
#include "cc_sebccopts_apspn_kernel.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <variant>
#include <optional>
#include <iomanip>
#include "random"
#include "cmath"
#include "iostream"
#include "../../common_include/graph_utils.h"
#include "../../common_include/timer.h"
#include "../../common_include/cc_bfs_apsp_l_cpp.hpp"
#ifdef MATLAB_BUILD
#include "mex.h"
#include "../../common_include/mex_utils.h"
#endif


//  This function computes the local BC and eqc_Pcount for nodes within the same block
void DistanceGuidedSss(
        const AlgorithmType& algo_type,
        const HEURISTICS& heuristics,
        const std::vector<std::vector<int64_t>>& eqc_A,
        const int64_t eqc_s,
        const int64_t eqc_n,
        const std::vector<int64_t>& eqc_block,
        const std::vector<char>& eqc_block_mask,
        const Eigen::MatrixXd& eqc_D,
        Eigen::MatrixXd& eqc_Pcount,
        const Eigen::VectorXd& eqc_max_depths,
        const std::vector<char>& eqc_processed,
        std::optional<Eigen::VectorXd>& eqc_BC,
        const std::optional<Eigen::MatrixXd>& h,
        const std::vector<int64_t>& eqc_node_mul,
        const double& eqc_mem_con,
        COUNTERS& counters) {

    auto eqc_D_s = eqc_D.col(eqc_s);
    auto eqc_Pcount_s = eqc_Pcount.col(eqc_s);
    Eigen::VectorXd Pcount_s_mem = Eigen::VectorXd::Zero(eqc_n);
    eqc_Pcount_s[eqc_s] = eqc_node_mul[eqc_s];

    switch (algo_type) {
        case AlgorithmType::BC:
            assert(eqc_BC.has_value());
            break;
    }

    // Build SP levels
    const int64_t max_depth = static_cast<int64_t>(eqc_max_depths(eqc_s));
    std::vector<std::vector<int64_t>> levels(max_depth + 1);
    for (auto& level : levels) {
        level.reserve(eqc_n / (max_depth + 1));
    }
    for (int64_t i : eqc_block) {
        levels[static_cast<int64_t>(eqc_D_s(i))].push_back(i);
    }

    // main loop
    std::vector<char> frontier_mask(eqc_n, false);
    std::vector<char> level_mask(eqc_n, false);
    frontier_mask[eqc_s] = true;

    // handle first level
    int64_t curr = eqc_s;
    level_mask[eqc_s] = true;
    for (const int64_t i : levels[1])
    {
        if (heuristics.SE) {
            eqc_Pcount_s(i) += eqc_Pcount_s[curr] * eqc_node_mul[i]; // overhead
        } 
        else 
        {
            eqc_Pcount_s(i) += eqc_Pcount_s[curr];
        }
        frontier_mask[i] = true;
    }
    // find memoized sources
    std::vector<int64_t> mem_sources;
    if (eqc_A[eqc_s].size() <= eqc_mem_con)
    {
        for (const int64_t t : levels[1])
        {
            if (eqc_processed[t]) // previously processed neighbor can replay its path counts
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
                if (curr_depth > eqc_max_depths[ms]) { // reached the end of ps's SP level
                    continue;
                }
                const auto& D_ms = eqc_D.col(ms);
                const auto& Pcount_ms = eqc_Pcount.col(ms);

                for (const int64_t i : dst_level) {
                    if (D_ms(i) == curr_depth) {  // shortest path from s to i pass through ps
                        if (heuristics.SE) 
                        {
                            Pcount_s_mem(i) += Pcount_ms(i) * eqc_Pcount_s(ms) / eqc_node_mul[ms];
                        } 
                        else 
                        {
                            Pcount_s_mem(i) += Pcount_ms(i) * eqc_Pcount_s(ms);
                        }
                    }
                }
            }
        }
    }

    // perform graph traversal
    for (int64_t curr_depth = 1; curr_depth < max_depth; ++curr_depth) {
        const auto& curr_level = levels[curr_depth];
        const auto& dst_level = levels[curr_depth + 1];
        // mask nodes at the current level
        for (const int64_t curr : curr_level) {
            level_mask[curr] = true;
        }

        // computes forward and backward cost
        int64_t fwd_cost = 0;
        int64_t bwd_cost = 0;

        for (const int64_t curr : curr_level) {
            if (!frontier_mask[curr]) continue;
            fwd_cost += eqc_A[curr].size();
        }
        if (fwd_cost == 0) continue; // no more frontier nodes
        for (const int64_t t : dst_level) {
            if (eqc_processed[t]) continue;
            bwd_cost += eqc_A[t].size();
        }
        if (fwd_cost < bwd_cost) { // forward traversal is cheaper
            for (const int64_t curr : curr_level) {
                if (!frontier_mask[curr]) continue;
                counters.graph_trav_count++;
                counters.fwd_visit_count++;
                const double curr_pcount = eqc_Pcount_s(curr);
                for (const int64_t t : eqc_A[curr]) {
                    if (level_mask[t] || !eqc_block_mask[t]) continue;
                    if (heuristics.SE) {
                        eqc_Pcount_s(t) += curr_pcount * eqc_node_mul[t]; // overhead
                    } else {
                        eqc_Pcount_s(t) += curr_pcount;
                    }
                    frontier_mask[t] = true;
                }
            }
        } 
        else 
        { // backward traversal is cheaper
            for (const int64_t curr : dst_level) {
                if (eqc_processed[curr])  // apply graph symmetry if possible
                {
                    eqc_Pcount_s(curr) = eqc_Pcount.col(curr)(eqc_s) - Pcount_s_mem(curr);
                    if (eqc_Pcount_s(curr) > 0) frontier_mask[curr] = true;  // curr not reachable via graph traversal
                }
                else 
                {
                    counters.graph_trav_count++;
                    counters.fwd_visit_count++;
                    for (const int64_t t : eqc_A[curr]) {
                        if (!frontier_mask[t] || !level_mask[t]) continue;
                        if (heuristics.SE) {
                            eqc_Pcount_s(curr) += eqc_Pcount_s(t) * eqc_node_mul[curr]; // overhead
                        } 
                        else 
                        {
                            eqc_Pcount_s(curr) += eqc_Pcount_s(t);
                        }
                        frontier_mask[curr] = true;
                    }
                }
            }
        }

    }
    // combine the results
    eqc_Pcount_s += Pcount_s_mem;

    switch (algo_type) {
        case AlgorithmType::BC:
            Eigen::VectorXd BC_s = Eigen::VectorXd::Zero(eqc_n);
            // Dependency propagation: Traverse levels in reverse order
            for (int64_t depth = max_depth; depth >= 2; depth--) {

                int64_t bwd_cost = 0;
                int64_t fwd_cost = 0;
                for (int64_t u : levels[depth]) {
                    bwd_cost += eqc_A[u].size();
                }
                for (int64_t u : levels[depth - 1]) {
                    fwd_cost += eqc_A[u].size();
                }
                if (bwd_cost < fwd_cost)
                {   // backward traversal is cheaper
                    for (int64_t u: levels[depth]) {
                        counters.bc_bwd_visit_count++;
                        for (int64_t v: eqc_A[u]) {
                            if (eqc_D_s(v) == eqc_D_s(u) - 1) {
                                if (heuristics.SE || heuristics.BCC) {
                                    BC_s(v) += (eqc_Pcount_s(v) / eqc_Pcount_s(u)) * ((*h)(eqc_s, u) + BC_s(u)) * eqc_node_mul[u]; // overhead
                                } else {
                                    BC_s(v) += (eqc_Pcount_s(v) / eqc_Pcount_s(u)) * (1 + BC_s(u));
                                }
                            }
                        }
                    }
                }
                else
                {   // forward traversal is cheaper
                    for (int64_t v: levels[depth-1]) {
                        counters.bc_fwd_visit_count++;
                        for (int64_t u: eqc_A[v]) {
                            if (eqc_D_s(v) == eqc_D_s(u) - 1 && eqc_block_mask[u]) {
                                if (heuristics.SE || heuristics.BCC) { // v is a valid predecessor of u in the SP-DAG
                                    BC_s(v) += (eqc_Pcount_s(v) / eqc_Pcount_s(u)) * ((*h)(eqc_s, u) + BC_s(u)) * eqc_node_mul[u]; // overhead
                                } else {
                                    BC_s(v) += (eqc_Pcount_s(v) / eqc_Pcount_s(u)) * (1 + BC_s(u));
                                }
                            }
                        }
                    }
                }
            }
            // Add pairwise dependencies to BC
            for (int64_t curr = 0; curr < eqc_n; ++curr) {
                if (curr != eqc_s) {
                    (*eqc_BC)[curr] += BC_s[curr];
                }
            }
            break;
    }

    // update counter
    counters.mem_util_count += mem_sources.size();
}

void PrioritizeSourceNodes(
        std::unordered_map<int64_t, std::vector<int64_t>>& block2nodes,
        std::unordered_map<int64_t, std::unordered_map<int64_t, int64_t>>& block_node_deg,
        const std::string& pri_criteria) {
    // prioritize source nodes within each block
    if (pri_criteria == "deg") {
        for (auto& [block_id, nodes] : block2nodes) {
            std::sort(nodes.begin(), nodes.end(), [&](int64_t a, int64_t b) {
                return block_node_deg[block_id][a] > block_node_deg[block_id][b];
            });
        }
    } else if (pri_criteria == "random") {
        std::random_device rd;  // Obtain a random number from hardware
        std::mt19937 rng(rd()); // Seed the generator
        for (auto& [block_id, nodes] : block2nodes) {
            std::shuffle(nodes.begin(), nodes.end(), rng);
        }
    } else {
        throw std::runtime_error("Unrecognized prioritization criteria.");
    }
}

// Function to calculate local degrees for nodes within each block
std::unordered_map<int64_t, std::unordered_map<int64_t, int64_t>>
CalLocalDegree(
        const std::unordered_map<int64_t, std::vector<int64_t>>& block2nodes,
        const std::vector<std::vector<bool>>& blockToNode,
        const std::vector<std::vector<int64_t>>& adj_list) {
    std::unordered_map<int64_t, std::unordered_map<int64_t, int64_t>> block_node_deg;
    for (const auto& [block_id, nodes] : block2nodes) {
        for (int64_t node : nodes) {
            int64_t local_degree = 0;
            for (int64_t neighbor : adj_list[node]) {
                if (blockToNode[block_id][neighbor]) {
                    local_degree++;
                }
            }
            block_node_deg[block_id][node] = local_degree;
        }
    }
    return block_node_deg;
}


Eigen::MatrixXd RecoverPcount(
        const std::vector<std::vector<int64_t>>& equivalence_classes,
        const std::vector<int64_t>& node_to_eqc_id,
        const std::vector<std::vector<int64_t>>& A,
        const Eigen::MatrixXd& eqc_Pcount) {
    int64_t n = A.size();
    Eigen::MatrixXd Pcount = Eigen::MatrixXd::Zero(n, n);

    // Pre-compute sizes to avoid repeated calculations
    std::vector<double> eqc_sizes(equivalence_classes.size());
    std::vector<size_t> adj_sizes(n);
    for (size_t i = 0; i < equivalence_classes.size(); ++i) {
        eqc_sizes[i] = static_cast<double>(equivalence_classes[i].size());
    }
    for (size_t i = 0; i < n; ++i) {
        adj_sizes[i] = A[i].size();
    }

    // Set diagonal elements first (vectorized operation)
    Pcount.diagonal().setOnes();

    // Process off-diagonal elements
    for (int64_t s = 0; s < n; ++s) {
        int64_t eqc_s = node_to_eqc_id[s];
        double eqc_s_size_inv = 1.0 / eqc_sizes[eqc_s];

        // Process elements after diagonal (upper triangular)
        for (int64_t t = s + 1; t < n; ++t) {
            int64_t eqc_t = node_to_eqc_id[t];
            double value;

            if (eqc_s == eqc_t) {
                value = adj_sizes[s];
            } else {
                value = eqc_Pcount(eqc_s, eqc_t) * eqc_s_size_inv / eqc_sizes[eqc_t];
            }


            // Set both symmetric elements at once
            Pcount(s, t) = value;
            Pcount(t, s) = value;
        }
    }

    return Pcount;
}

double CompMemConCurrBlock(int64_t n, int64_t m, const std::string& pse_restriction) {
    double density = (double)m / (n * (n - 1));
    double avg_deg = (double)m / n;
    double pse_limit;
    if (pse_restriction == "avg-deg") {
        pse_limit = avg_deg;
    } else if (pse_restriction == "den-deg") {
        pse_limit = std::pow(avg_deg, 1 / (1 + density));
    } else if (pse_restriction == "sqrt-deg") {
        pse_limit = std::pow(avg_deg, 0.5);
    } else if (pse_restriction == "cri-den") {
        // log(N)/N
        // 0~1 confidence level of which you believe memoization is useful at this density
        double confidence = 0.5;
        double rho_critical = std::log(n) / n;
        double coeff = std::log(confidence) / rho_critical;
        pse_limit = std::exp(density*coeff)*avg_deg;
    } else if (pse_restriction == "none") {
        pse_limit = n; // means no limitation
    } else {
        // Create an output string stream to format your message
        std::ostringstream oss;
        oss << "Unrecognized information throttling strategy: " << pse_restriction << ".";
        throw std::runtime_error(oss.str());
    }
    return pse_limit;
}

std::vector<double> CompMemConAllBlock(const std::unordered_map<int64_t, std::vector<int64_t>>& eqc_block2nodes,
                                             const std::vector<std::vector<int64_t>>& eqc_A,
                                             const std::vector<bool>& eqc_cv_mask,
                                             const std::vector<std::vector<bool>> & eqc_blockToNode,
                                             const std::string& pse_restriction) {
    std::vector<double> pse_limit_all_blocks(eqc_block2nodes.size(), 0);
    for (const auto& [eqc_B_id, eqc_block] : eqc_block2nodes) {
        int64_t eqc_block_m = 0;
        for (int64_t eqc_node : eqc_block) {
            if (eqc_cv_mask[eqc_node]) {
                for (int64_t neighbor : eqc_A[eqc_node]) {
                    if (eqc_blockToNode[eqc_B_id][neighbor]) {
                        eqc_block_m++;
                    }
                }
            } else {
                eqc_block_m += eqc_A[eqc_node].size();
            }
        }
        double pse_limit = CompMemConCurrBlock(eqc_block.size(), eqc_block_m, pse_restriction);
        pse_limit_all_blocks[eqc_B_id] = pse_limit;
    }
    return pse_limit_all_blocks;
}

/*
 * This function computes APSPN or BC
 */
Result CcSeBccOptionalApspnKernel(
        const Eigen::MatrixXd& adj_matrix,
        const AlgorithmParams& params) {

    // computation starts
    if (params.verbose) {
        std::cout << "Computation started." << std::endl;
    }
    COUNTERS counters;
    Timer global_timer;
    Timer local_timer;
    global_timer.Start();
    int64_t n = adj_matrix.rows();

    const AlgorithmType algo_type = params.algo_type;
    bool use_se = params.use_se;
    bool use_bcc = params.use_bcc;

    if (params.verbose) {
        std::cout << "Use SE: " << use_se << std::endl;
        std::cout << "Use BCC: " << use_bcc << std::endl;
    }

    // ------------- Compute adjacency list -------------
    local_timer.Start();

    // compute adjacency list
    const auto& A = ComputeAdjList(adj_matrix, n);
    Eigen::VectorXd degrees(n);
    int64_t m = 0; // number of edges in the original graph
    for (int64_t i = 0; i < n; ++i) {
        degrees[i] = static_cast<double>(A[i].size());
        m += A[i].size();
    }

    local_timer.Stop();
    counters.durationInSecComputeAdjList = local_timer.Seconds();
    if (params.verbose) {
        std::cout << "@@ Adjacency list computation cost time: " << local_timer.Seconds() << std::endl;
    }


    // ------------- Compute APSP of the original graph -------------
    local_timer.Start();
    CC_BFS_APSP::RESULT result = CC_BFS_APSP::CC_BFS_APSP_L_FromAdjacency(
        A, degrees, "bounded-avg-deg");
    Eigen::MatrixXd D = std::move(result.D);
    local_timer.Stop();
    counters.durationInSecAPSPComputation = local_timer.Seconds();
    if (params.verbose) {
        std::cout << "@@ APSP computation time: " << local_timer.Seconds() << " seconds" << std::endl;
    }


    // ------------- SE: Find structural equivalence classes -------------
    local_timer.Start();

    std::vector<std::vector<int64_t>> equivalence_classes;
    std::vector<int64_t> eqc_node_mul;
    std::vector<int64_t> node_to_eqc_id(n, 0);
    std::vector<std::vector<int64_t>> eqc_A;
    int64_t eqc_n;
    int64_t eqc_m;

    if (use_se) {
        std::tie(equivalence_classes, eqc_node_mul) = find_structural_equivalence(A);
        eqc_n = equivalence_classes.size();
        for (int64_t eqc_id = 0; eqc_id < equivalence_classes.size(); ++eqc_id) {
            const auto &eqc = equivalence_classes[eqc_id];
            for (const int64_t& node: eqc) {
                node_to_eqc_id[node] = eqc_id;
            }
        }

        // compute the adjacency list of the reduced graph
        std::vector<std::unordered_set<int64_t>> eqc_A_set(eqc_n);
        for (int64_t eqc_node = 0; eqc_node < eqc_n; ++eqc_node) {
            int64_t sample_graph_node = equivalence_classes[eqc_node][0];
            const auto &sample_graph_node_neig = A[sample_graph_node];
            for (int64_t neighbor: sample_graph_node_neig) {
                eqc_A_set[eqc_node].insert(node_to_eqc_id[neighbor]);
            }
        }
        eqc_m = 0;
        eqc_A = std::vector<std::vector<int64_t>>(eqc_n);
        for (int64_t i = 0; i < eqc_n; ++i) {
            eqc_A[i] = std::vector<int64_t>(eqc_A_set[i].begin(), eqc_A_set[i].end());
            eqc_m += eqc_A[i].size();
        }

        local_timer.Stop();
        counters.eqc_n = eqc_n;
        counters.durationInSecFindEqc = local_timer.Seconds();
        if (params.verbose) {
            std::cout << std::fixed << std::setprecision(4) << "@@ structural equivalence classes computation cost time: "
                      << counters.durationInSecFindEqc << std::endl;
        }
    } else {
        equivalence_classes.resize(n);
        for (int64_t i = 0; i < n; ++i) {
            equivalence_classes[i].push_back(i);
        }
        eqc_A = A;
        eqc_node_mul = std::vector<int64_t>(n, 1);
        eqc_n = n;
        eqc_m = m;
        if (params.verbose) {
            std::cout << "Skip structural equivalence classes computation." << std::endl;
        }
    }

    if (use_se && eqc_n == n) {
        // we treat the graph as the original graph if no se vertice exists, to avoid processing overhead
        eqc_A = A;
    }

    // ------------- BCC: Find bi-connected components -------------
    local_timer.Start();

    std::unordered_map<int64_t, std::vector<int64_t>> eqc_block2nodes;
    std::vector<int64_t> eqc_cv_set;
    std::vector<std::vector<int64_t>> eqc_bcTreeAdj;
    std::vector<std::vector<bool>> eqc_blockToNode;
    std::vector<bool> eqc_cv_mask(n, false);

    if (!use_bcc) { // to remove the BCC heuristic, we treat the graph as if it is a single block
        eqc_block2nodes[0] = std::vector<int64_t>(eqc_n);
        std::iota(eqc_block2nodes[0].begin(), eqc_block2nodes[0].end(), 0);
        eqc_blockToNode.push_back(std::vector<bool>(n, true));
        eqc_cv_set = std::vector<int64_t>(0);
        eqc_bcTreeAdj = std::vector<std::vector<int64_t>>(0);

        if (params.verbose) {
            std::cout << "Skip bi-connected components decomposition." << std::endl;
        }
    } else {
        // find bi-connected components and build the block-cut tree
        BiconnectedComponents bcc = find_biconnected_components_tree(eqc_A);
        eqc_block2nodes = bcc.block2nodes;
        eqc_cv_set = bcc.cutVertices;
        eqc_bcTreeAdj = bcc.bcTreeAdj;
        eqc_blockToNode = bcc.blockToNode;
        for (int64_t cv: eqc_cv_set) eqc_cv_mask[cv] = true;

        local_timer.Stop();
        counters.bc_count = eqc_block2nodes.size();
        counters.cv_count = eqc_cv_set.size();
        counters.durationInSecBiconnDecomp = local_timer.Seconds();
        if (params.verbose) {
            std::cout << "@@ bi-connected component decomposition cost time: " << counters.durationInSecBiconnDecomp << std::endl;
        }
    }
    int64_t num_blocks = eqc_block2nodes.size();
    int64_t offset = num_blocks;

    // determine the heuristic type
    HEURISTICS heuristics;
    if (eqc_n < n) {
        heuristics.SE = true;
    }
    if (num_blocks > 1) {
        heuristics.BCC = true;
    }

    // ------------- Compute local degrees -------------
    local_timer.Start();

    auto eqc_block_node_deg = CalLocalDegree(eqc_block2nodes, eqc_blockToNode, eqc_A);

    local_timer.Stop();
    counters.durationInSecCompLocalDeg = local_timer.Seconds();
    if (params.verbose) {
        std::cout << std::fixed << std::setprecision(4) << "@@ local degrees computation cost time: " << counters.durationInSecCompLocalDeg << std::endl;
    }

    // ------------- Prioritize source nodes and compute memoization constraint for each node -------------
    local_timer.Start();

    PrioritizeSourceNodes(eqc_block2nodes, eqc_block_node_deg, "deg");
    auto mem_cons = CompMemConAllBlock(eqc_block2nodes, eqc_A, eqc_cv_mask, eqc_blockToNode, "sqrt-deg");
    double sum_mem_con = 0;
    int64_t sum_source_nodes = 0;
    for (const auto& [eqc_block_id, eqc_block] : eqc_block2nodes) {
        sum_mem_con += eqc_block.size() * mem_cons[eqc_block_id];
        sum_source_nodes += eqc_block.size();
    }
    double avg_mem_con = sum_mem_con / sum_source_nodes;

    local_timer.Stop();
    counters.apspn_mem_con = avg_mem_con;
    counters.durationInSecPri = local_timer.Seconds();
    if (params.verbose) {
        std::cout << std::fixed << std::setprecision(4) << "@@ prioritization cost time: " << counters.durationInSecPri << std::endl;
    }

    // ------------- Pre-processing -------------
    Eigen::MatrixXd eqc_D = Eigen::MatrixXd::Constant(eqc_n, eqc_n, std::numeric_limits<double>::infinity());
    Eigen::MatrixXd eqc_Pcount = Eigen::MatrixXd::Zero(eqc_n, eqc_n);
    std::optional<Eigen::VectorXd> eqc_BC;
    std::optional<Eigen::VectorXd> BC;
    std::optional<Eigen::MatrixXd> Pcount;
    std::optional<Eigen::MatrixXd> h;
    std::optional<Eigen::MatrixXd> eqc_D_l;
    std::optional<Eigen::MatrixXd> eqc_D_r;

    // map D to eqc_D
    local_timer.Start();

    if (eqc_n == n)
    {
        eqc_D = D;
    }
    else
    {
        // Create index vectors for efficient matrix indexing
        Eigen::Matrix<int64_t, Eigen::Dynamic, 1> eqc_indices(eqc_n);
        for (int64_t i = 0; i < eqc_n; ++i) {
            eqc_indices[i] = equivalence_classes[i][0];
        }
        Eigen::VectorXi eqc_indices_int = eqc_indices.cast<int>();
        eqc_D = D(eqc_indices_int, eqc_indices_int);
    }

    local_timer.Stop();
    if (params.verbose) {
        std::cout << "@@ Mapping D to eqc_D cost time: " << local_timer.Seconds() << " seconds" << std::endl;
    }

    switch (algo_type) {
        case AlgorithmType::APSPN:
            break;
        case AlgorithmType::BC:
            eqc_BC = Eigen::VectorXd::Zero(eqc_n);
            h = Eigen::MatrixXd::Constant(eqc_n, eqc_n, 1);
            if (heuristics.BCC) {
                // compute the block-cut tree weights D_r and D_l
                std::tie(eqc_D_r.emplace(), eqc_D_l.emplace()) =
                        compute_eqc_comp_tree_weights(eqc_block2nodes, eqc_bcTreeAdj, eqc_n, eqc_node_mul, n);
                // compute the inter-block dependency for each cut vertex
                for (int64_t eqc_cv : eqc_cv_set) {
                    double weight_sum = 0;
                    for (int64_t eqc_B : eqc_bcTreeAdj[eqc_cv + offset]) {
                        weight_sum += (*eqc_D_l)(eqc_B, eqc_cv) * (*eqc_D_r)(eqc_B, eqc_cv);
                    }
                    (*eqc_BC)(eqc_cv) += weight_sum;
                }
            } else {
                // there is no bi-connected component in the graph
                // these two matrices wouldn't be used
                eqc_D_l = Eigen::MatrixXd::Zero(0, 0);
                eqc_D_r = Eigen::MatrixXd::Zero(0, 0);
            }

            // compute the communication intensity matrix h, which is determined by both SE and BCC
            if (heuristics.SE || heuristics.BCC) {
                for (auto& [eqc_block_id, eqc_block] : eqc_block2nodes) {
                    compute_comm_intensity_matrix(A, equivalence_classes, eqc_A, eqc_block_id, eqc_block, *eqc_D_l, *h, eqc_cv_mask, eqc_node_mul);
                }
            }
            break;
        default:
            throw std::runtime_error("Unrecognized algorithm type.");
    }


    // ------------- Local APSPN Single-source search -------------
    local_timer.Start();

    Eigen::VectorXd eqc_max_depths = Eigen::VectorXd::Zero(eqc_n);
    std::vector<char> apspn_eqc_processed(eqc_n, false);
    for (auto& pair : eqc_block2nodes) { // process each block
        auto& eqc_block_id = pair.first;
        auto& eqc_block = pair.second;
        // mask for the current block
        std::vector<char> eqc_block_mask(eqc_n, false);
        for (int64_t eqc_node : eqc_block) eqc_block_mask[eqc_node] = true;

        // compute the max depth for each node within the its block
        for (int64_t eqc_s : eqc_block) {
            for (int64_t eqc_node : eqc_block) {
                eqc_max_depths[eqc_s] = std::max<int64_t>(eqc_max_depths[eqc_s], eqc_D.col(eqc_s)(eqc_node));
            }
        }

        // compute local APSPN
        for (int64_t eqc_s: eqc_block) {
            DistanceGuidedSss(
                    algo_type,
                    heuristics,
                    eqc_A,
                    eqc_s,
                    eqc_n,
                    eqc_block,
                    eqc_block_mask,
                    eqc_D,
                    eqc_Pcount,
                    eqc_max_depths,
                    apspn_eqc_processed,
                    eqc_BC,
                    h,
                    eqc_node_mul,
                    mem_cons[eqc_block_id],
                    counters);
            apspn_eqc_processed[eqc_s] = true;
        }
        // clear ps_mask[cv] to avoid false usage
        for (int64_t s: eqc_cv_set) {
            apspn_eqc_processed[s] = false;
        }
    }

    //----------------- Post-processing -----------------
    switch (algo_type) {
        case AlgorithmType::APSPN:
            if (heuristics.BCC) {
                // compute inter-block eqc_Pcount
                compute_inter_block_Pcount(eqc_block2nodes, eqc_bcTreeAdj, eqc_node_mul, eqc_Pcount, offset, eqc_n);
            }
            if (heuristics.SE) {
                // recover Pcount from the eqc graph
                Pcount = RecoverPcount(equivalence_classes, node_to_eqc_id, A, eqc_Pcount);
            } else {
                Pcount = eqc_Pcount;
            }
            break;
        case AlgorithmType::BC:
            BC = Eigen::VectorXd::Zero(n);
            if (heuristics.SE) {
                // add up the intra-eqc contributions stored in the h diagonal
                for (int64_t eqc_node = 0; eqc_node < eqc_n; ++eqc_node) {
                    (*eqc_BC)(eqc_node) += (*h)(eqc_node, eqc_node);
                }
                // recover the BC value for each v in original graph
                for (int64_t eqc_node = 0; eqc_node < eqc_n; ++eqc_node) {
                    const auto &class_nodes = equivalence_classes[eqc_node];
                    for (int64_t i: class_nodes) {
                        (*BC)(i) = (*eqc_BC)(eqc_node) / class_nodes.size();
                    }
                }
            } else {
                // copy eqc_BC to BC
                (*BC) = eqc_BC.value();
            }
            if (adj_matrix.isApprox(adj_matrix.transpose())) {
                (*BC) /= 2;
            }
            break;
        default:
            throw std::runtime_error("Unrecognized algorithm type.");

    }

    local_timer.Stop();
    counters.durationInSecLocalSSS = local_timer.Seconds();
    if (params.verbose) {
        std::cout << std::fixed << std::setprecision(4) << "@@ local single-source searches cost time: " << counters.durationInSecLocalSSS << std::endl;
    }

    // computation ends
    global_timer.Stop();
    counters.durationInSecTotal = global_timer.Seconds();

    // ------------- Print statistics -------------
    int64_t source_count = 0;
    int64_t bfs_vn_count = 0; // overall nodes require to be visited by standard BFS
    for (const auto& [block, block_nodes] : eqc_block2nodes) {
        bfs_vn_count += block_nodes.size()*block_nodes.size();
        source_count += block_nodes.size();
    }
    counters.org_graph_density = (double)m / (n * (n - 1));
    counters.reduced_graph_density = (double)eqc_m / (eqc_n * (eqc_n - 1));
    counters.biconn_sep_ratio = (double)(eqc_n*eqc_n-bfs_vn_count) / (eqc_n*eqc_n);

    // check for abnormal data
    if (n<=0 || m<=0 || eqc_n<=0 || eqc_m<=0) {
        throw std::runtime_error("Non-positive graph size detected. Check for numeric overflow.");
    }
    if (params.verbose) {
        std::cout << "overall equivalence classes number: " << eqc_n << ", proportion: " << (double)eqc_n / n << std::endl;
        std::cout << "overall cut-vertex number: " << eqc_cv_set.size() << ", proportion: " << (double)eqc_cv_set.size() / n << std::endl;

        std::cout << "Original connectivity density: " << std::fixed << std::setprecision(4)
                  << counters.org_graph_density * 100 << "%, after reduction: "
                  << counters.reduced_graph_density * 100 << "%, decreased: "
                  << (counters.org_graph_density - counters.reduced_graph_density) * 100 << "%" << std::endl;
        std::cout << "Encountered ps nodes: " << counters.mem_util_count << ", avg per source: "
                  << static_cast<double>(counters.mem_util_count) / source_count << ", (eqc) avg node degree: "
                  << static_cast<double>(eqc_m) / eqc_n << ", avg pse limit: " << avg_mem_con << std::endl;
        std::cout << "Visited normal nodes: " << counters.graph_trav_count << ", proportion: "
                  << static_cast<double>(counters.graph_trav_count) / bfs_vn_count << std::endl;
        std::cout << "Bi-conn separated nodes: " << eqc_n * eqc_n - bfs_vn_count << ", proportion: "
                  << counters.biconn_sep_ratio << std::endl;

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
    // Return the result
    switch (algo_type) {
        case AlgorithmType::APSPN:
            return Result{Pcount.value(), counters,counters.durationInSecTotal};
        case AlgorithmType::BC:
            return Result{BC.value(),counters, counters.durationInSecTotal};
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
 * @brief Python-callable wrapper for CC-SE-BCC-APSPN-kernel algorithms
 * 
 * @param algo_type_str Algorithm type string: "apspn" or "bc" (case-insensitive)
 * @param use_se Use SE flag (1 or 0)
 * @param use_bcc Use BCC flag (1 or 0)
 * @param adj_matrix_data Flat array of adjacency matrix (row-major, n*n elements)
 * @param n Number of nodes
 * @param output_data Output array allocated by caller
 * @param elapsed_time Output parameter for computation time
 * @return 0 on success, negative error code on failure
 */
EXPORT int cc_sebccopts_apspn_kernel_py(
    const char* algo_type_str,
    int use_se,
    int use_bcc,
    const double* adj_matrix_data,
    int64_t n,
    double* output_data,
    double* elapsed_time)
{
    try {
        // Parse algorithm type
        std::string algo_str(algo_type_str);
        std::transform(algo_str.begin(), algo_str.end(), algo_str.begin(), ::tolower);
        
        AlgorithmType algo_type;
        if (algo_str == "apspn") algo_type = AlgorithmType::APSPN;
        else if (algo_str == "bc") algo_type = AlgorithmType::BC;
        else return -2;  // Invalid algorithm type
        
        // Convert adjacency matrix
        Eigen::Map<const Eigen::MatrixXd> adj_matrix_map(adj_matrix_data, n, n);
        Eigen::MatrixXd adj_matrix = adj_matrix_map;
        
        // Set up algorithm parameters
        AlgorithmParams params(use_se != 0, use_bcc != 0, false);  // verbose = false for Python
        params.algo_type = algo_type;
        
        // Call the algorithm
        Result result = CcSeBccOptionalApspnKernel(adj_matrix, params);
        
        // Copy output based on algorithm type
        switch (algo_type) {
            case AlgorithmType::APSPN: {
                const auto& matrix_result = std::get<Eigen::MatrixXd>(result.result);
                Eigen::Map<Eigen::MatrixXd>(output_data, n, n) = matrix_result;
                break;
            }
            case AlgorithmType::BC: {
                const auto& vector_result = std::get<Eigen::VectorXd>(result.result);
                Eigen::Map<Eigen::VectorXd>(output_data, n) = vector_result;
                break;
            }
            default:
                return -2;
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
AlgorithmType parseAlgorithmType(const mxArray* prhs) {
    if (!mxIsChar(prhs)) {
        mexErrMsgIdAndTxt("MyToolbox:unified:invalidInput", "Algorithm type must be a string.");
    }
    
    char* algoTypeStr = mxArrayToString(prhs);
    std::string algo_type_str(algoTypeStr);
    mxFree(algoTypeStr);
    
    if (algo_type_str == "apspn") {
        return AlgorithmType::APSPN;
    } else if (algo_type_str == "bc") {
        return AlgorithmType::BC;
    } else if (algo_type_str == "gc") {
        mexErrMsgIdAndTxt("MyToolbox:unified:invalidAlgorithm",
                          "GC algorithm not supported in this function.");
    } else {
        mexErrMsgIdAndTxt("MyToolbox:unified:invalidAlgorithm",
                          "Unknown algorithm type. Must be 'apspn', or 'bc'.");
    }
    // This line should never be reached due to mexErrMsgIdAndTxt
    return AlgorithmType::APSPN;
}

struct ParsedInputs {
    AlgorithmType algo_type;
    bool use_se;
    bool use_bcc;
    Eigen::MatrixXd adjMatrix;
    bool verbose;
};

ParsedInputs parseInputArguments(int nrhs, const mxArray *prhs[]) {
    // Check minimum number of arguments
    if (nrhs < 4) {
        mexErrMsgIdAndTxt("MyToolbox:unified:nrhs",
                          "At least 4 inputs required (algorithm type, use_se, use_bcc, adjacency matrix).");
    }
    
    ParsedInputs inputs;
    
    // Parse algorithm type
    inputs.algo_type = parseAlgorithmType(prhs[0]);
    
    // Parse use_se and use_bcc flags
    inputs.use_se = static_cast<bool>(mxGetScalar(prhs[1]));
    inputs.use_bcc = static_cast<bool>(mxGetScalar(prhs[2]));
    
    // Parse adjacency matrix
    inputs.adjMatrix = parseSparseDenseMatrix(prhs[3]);
    if (inputs.adjMatrix.rows() <= 1) {
        mexErrMsgIdAndTxt("MyToolbox:unified:invalidGraphSize",
                          "Input graph must contain at least 2 nodes.");
    }
    
    // Set default verbose
    inputs.verbose = true;
    
    // Parse optional verbose parameter
    if (nrhs > 4) {
        if (mxIsLogicalScalar(prhs[4])) {
            inputs.verbose = mxIsLogicalScalarTrue(prhs[4]);
        } else if (!mxIsEmpty(prhs[4])) {
            mexErrMsgIdAndTxt("MyToolbox:unified:invalidInput", 
                              "Verbose flag must be a logical scalar or empty [].");
        }
    }
    
    return inputs;
}

void createOutputArguments(int nlhs, mxArray *plhs[], const Result& result, 
                          AlgorithmType algo_type, Eigen::Index numVertices) {
    // Create main output based on algorithm type
    switch(algo_type) {
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
            try {
                const auto& vector_result = std::get<Eigen::VectorXd>(result.result);
                plhs[0] = mxCreateDoubleMatrix(numVertices, 1, mxREAL);
                Eigen::Map<Eigen::VectorXd>(mxGetPr(plhs[0]), numVertices) = vector_result;
            } catch (const std::bad_variant_access&) {
                mexErrMsgIdAndTxt("MyToolbox:unified:unexpectedType",
                                  "BC algorithm should return a vector.");
            }
            break;
        default:
            mexErrMsgIdAndTxt("MyToolbox:unified:unknownAlgorithm",
                              "Unknown algorithm type.");
    }

    // Optional outputs
    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleScalar(result.total_time);
    }

    if (nlhs > 2) {
        // parse counters into matlab struct
        COUNTERS counters = result.counters;
        const char *field_names[] = {"pse_count", "nn_visit_count", "cv_count", "bc_count", "eqc_n", "pse_limit",
                                     "org_graph_density", "reduced_graph_density", "biconn_sep_ratio",
                                     "durationInSecComputeAdjList", "durationInSecFindEqc", "durationInSecBiconnDecomp",
                                     "durationInSecCompLocalDeg", "durationInSecITandPrior", "durationInSecLocalSSS",
                                     "durationInSecTotal", "durationInSecAPSPComputation"};
        mwSize num_fields = sizeof(field_names) / sizeof(*field_names);
        plhs[2] = mxCreateStructMatrix(1, 1, num_fields, field_names);
        mxSetField(plhs[2], 0, "pse_count", mxCreateDoubleScalar(counters.mem_util_count));
        mxSetField(plhs[2], 0, "nn_visit_count", mxCreateDoubleScalar(counters.graph_trav_count));
        mxSetField(plhs[2], 0, "cv_count", mxCreateDoubleScalar(counters.cv_count));
        mxSetField(plhs[2], 0, "bc_count", mxCreateDoubleScalar(counters.bc_count));
        mxSetField(plhs[2], 0, "eqc_n", mxCreateDoubleScalar(counters.eqc_n));
        mxSetField(plhs[2], 0, "pse_limit", mxCreateDoubleScalar(counters.apspn_mem_con));
        mxSetField(plhs[2], 0, "org_graph_density", mxCreateDoubleScalar(counters.org_graph_density));
        mxSetField(plhs[2], 0, "reduced_graph_density", mxCreateDoubleScalar(counters.reduced_graph_density));
        mxSetField(plhs[2], 0, "biconn_sep_ratio", mxCreateDoubleScalar(counters.biconn_sep_ratio));
        mxSetField(plhs[2], 0, "durationInSecComputeAdjList", mxCreateDoubleScalar(counters.durationInSecComputeAdjList));
        mxSetField(plhs[2], 0, "durationInSecFindEqc", mxCreateDoubleScalar(counters.durationInSecFindEqc));
        mxSetField(plhs[2], 0, "durationInSecBiconnDecomp", mxCreateDoubleScalar(counters.durationInSecBiconnDecomp));
        mxSetField(plhs[2], 0, "durationInSecCompLocalDeg", mxCreateDoubleScalar(counters.durationInSecCompLocalDeg));
        mxSetField(plhs[2], 0, "durationInSecITandPrior", mxCreateDoubleScalar(counters.durationInSecPri));
        mxSetField(plhs[2], 0, "durationInSecLocalSSS", mxCreateDoubleScalar(counters.durationInSecLocalSSS));
        mxSetField(plhs[2], 0, "durationInSecTotal", mxCreateDoubleScalar(counters.durationInSecTotal));
        mxSetField(plhs[2], 0, "durationInSecAPSPComputation", mxCreateDoubleScalar(counters.durationInSecAPSPComputation));
    }
}

/**
 * @brief MEX interface function for CC-SE-BCC-APSPN family of graph algorithms
 *
 * @param nlhs Number of output arguments (left-hand side)
 * @param plhs Array of output arguments
 * @param nrhs Number of input arguments (right-hand side)
 * @param prhs Array of input arguments
 *
 * INPUT PARAMETERS:
 * prhs[0] - Algorithm type (string): {'apspn', 'bc'}
 * prhs[1] - Use SE flag (logical): Enable/disable Structural Equivalence optimization
 * prhs[2] - Use BCC flag (logical): Enable/disable Biconnected Components optimization
 * prhs[3] - Adjacency matrix (double or logical matrix): Square symmetric matrix
 *           representing an undirected, unweighted graph.
 * prhs[4] - (Optional) Verbose flag (logical): Enable/disable progress output (default: true)
 *
 * OUTPUT PARAMETERS:
 * plhs[0] - Algorithm result (matrix for APSPN, vector for BC)
 * plhs[1] - (Optional) Computation time (double scalar)
 * plhs[2] - (Optional) Performance counters (struct)
 *
 * @author Junda
 * @version 3.0
 */
// Entry point for the MEX-file
void mexFunction(int64_t nlhs, mxArray *plhs[], int64_t nrhs, const mxArray *prhs[]) {
    try {
        // Parse all input arguments
        ParsedInputs inputs = parseInputArguments(nrhs, prhs);
        
        // Set up algorithm parameters
        AlgorithmParams params(inputs.use_se, inputs.use_bcc, inputs.verbose);
        params.algo_type = inputs.algo_type;
        
        // Run the algorithm
        auto result = CcSeBccOptionalApspnKernel(inputs.adjMatrix, params);
        
        // Create output arguments
        createOutputArguments(nlhs, plhs, result, inputs.algo_type, inputs.adjMatrix.rows());
        
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("MyToolbox:unified:exception", e.what());
    }
}

#endif // MATLAB_BUILD
