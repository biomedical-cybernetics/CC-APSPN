//
// Created by Junda on 2024/4/4.
//

#include "../cc_apspn_kernel_ablation_approx.h"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include "../../../common_include/graph_utils.h"
#include "../../../common_include/timer.h"

// Constants for ball_indicator (used in approximate algorithms)
#define UNVISITED 0
#define VISITED_U 1
#define VISITED_V 2

// Structure to store memoized information for a node
struct MemNode {
    uint32_t node_id;
    uint32_t dist;
    uint64_t n_paths;
    double p;  // Accumulated projection
};

// Static random number generator for RandomNode function
static std::mt19937_64 random_engine(42ULL);

// Returns a random node index between 0 and n-1
// Uses rejection sampling to avoid modulo bias (same as SILVAN's Rand_gen::get_max)
inline uint32_t RandomNode(uint32_t n) {
    uint64_t r;
    uint64_t threshold = UINT64_MAX - UINT64_MAX % n;
    while ((r = random_engine()) >= threshold) {}
    return static_cast<uint32_t>(r % n);
}

inline uint64_t RandomIndex(uint64_t max) {
    if (max == 0) {
        return 0;
    }
    uint64_t r;
    uint64_t threshold = UINT64_MAX - UINT64_MAX % max;
    while ((r = random_engine()) >= threshold) {}
    return r % max;
}

// Accumulate projection over meeting edges
double AccumulateProjection(
    const std::vector<std::pair<int64_t, int64_t>> &sp_edges,
    std::vector<uint32_t>& ball_indicator,
    const Eigen::MatrixXd& RD,
    const std::vector<double>& n_paths_u,
    const std::vector<double>& n_paths_v,
    const std::vector<double>& p_u,
    const std::vector<double>& p_v) {
    // Compute accumulated projection p_uv by summing over meeting edges
    double p_uv = 0.0;

    for (const std::pair<int64_t, int64_t>& edge : sp_edges) {
        int64_t x = edge.first;
        int64_t y = edge.second;

        if (ball_indicator[x] == VISITED_U) {
            p_uv += p_u[x] * n_paths_v[y] + RD(x, y) * n_paths_u[x] * n_paths_v[y] + p_v[y] * n_paths_u[x];
        } else {
            p_uv += p_u[y] * n_paths_v[x] + RD(y, x) * n_paths_u[y] * n_paths_v[x] + p_v[x] * n_paths_u[y];
        }
    }

    return p_uv;
}

// Compute all shortest paths for a sampled pair of nodes
double CcBbBFS(
    const uint32_t n,
    const Eigen::VectorXd& DEG,
    const std::vector<std::vector<int64_t>>& A,
    const uint32_t u,
    const uint32_t v,
    std::vector<uint32_t>& ball_indicator,
    const Eigen::MatrixXd& RD,
    std::vector<int64_t>& q,
    std::vector<double>& n_paths_u,
    std::vector<double>& n_paths_v,
    std::vector<int32_t>& dist,
    std::vector<std::vector<MemNode>>& mem_per_node,
    std::vector<uint32_t>& mem_depth_per_node,
    const uint32_t max_mem_depth,
    std::vector<double>& p_u,
    std::vector<double>& p_v,
    COUNTERS& counters) {

    auto start_estimator = std::chrono::high_resolution_clock::now();

    double num_paths = 0.0;
    double p_uv_mean = 0.0;  // mean projection of the topological shortest paths (TSPs) between u and v
    double p_uv = 0.0;  // accumulated projection of the TSPs between u and v
    double contrib = 0.0;

    // Check for small distance
    uint32_t first_node, second_node;
    const std::vector<int64_t>* first_adj;
    const std::vector<int64_t>* second_adj;
    if (DEG(u) <= DEG(v)) {
        first_node = u;
        second_node = v;
        first_adj = &A[first_node];
        second_adj = &A[second_node];
    } else {
        first_node = v;
        second_node = u;
        first_adj = &A[first_node];
        second_adj = &A[second_node];
    }
    // Scan neighbors of the first node
    for (auto w : *first_adj) {
        if (w == second_node) {
            p_uv += RD(first_node, second_node);
            num_paths = 1.0;
            break;
        }
        ball_indicator[w] = VISITED_U;
    }

    // Scan neighbors of the second node
    if (num_paths == 0.0) {
        for (auto w : *second_adj) {
            if (ball_indicator[w] == VISITED_U) {
                p_uv += RD(first_node, w) + RD(w, second_node);
                num_paths += 1.0;
            }
        }
    }

    // Reset ball_indicator
    for (auto i : *first_adj) {
        ball_indicator[i] = UNVISITED;
    }

    // If short path found, return directly
    if (num_paths > 0.0) {
        p_uv_mean = p_uv / num_paths;
        contrib = RD(first_node, second_node) / p_uv_mean;
        auto end_estimator = std::chrono::high_resolution_clock::now();
        counters.durationInSecCcBbBFS += std::chrono::duration_cast<std::chrono::nanoseconds>(end_estimator - start_estimator).count() / 1e9;
        return contrib;
    }

    // Bidirect_bfs
    int64_t end_q = 0;
    std::vector<std::pair< int64_t, int64_t > > sp_edges;
    int64_t x, y;
    bool have_to_stop = false;
    int64_t start_u = 0, start_v = 1, end_u = 1, end_v = 2, start_cur, end_cur, *new_end_cur;
    int64_t sum_degs_u = 0, sum_degs_v = 0, *sum_degs_cur;

    end_q = 2;

    q[0] = u;
    q[1] = v;

    ball_indicator[u] = VISITED_U;
    ball_indicator[v] = VISITED_V;

    n_paths_u[u] = 1;
    dist[u] = 0;
    p_u[u] = 0.0;  // Initialize projection from u to u

    n_paths_v[v] = 1;
    dist[v] = 0;
    p_v[v] = 0.0;  // Initialize projection from v to v

    std::vector<double> *cur_n_paths;
    std::vector<double> *other_n_paths;
    double tot_weight = 0;

    // --- SYNCHRONIZED MEMORY REPLAY LOGIC START ---
    auto start_mem_replay = std::chrono::high_resolution_clock::now();
    const auto& mem_u = mem_per_node[u];
    const auto& mem_v = mem_per_node[v];
    bool collision_detected = false;

    // Skip memory replay if depth <= 1
    if (!mem_u.empty() || !mem_v.empty()) {
        size_t idx_u = 0;
        size_t idx_v = 0;
        uint32_t curr_dist_u = 1;  // Track distance for U's replay
        uint32_t curr_dist_v = 1;  // Track distance for V's replay

        // Synchronized replay: alternate between U and V based on sum_degs
        // Continue replay while either endpoint has a stored level left.
        while (idx_u < mem_u.size() || idx_v < mem_v.size()) {
            // Decide which memory to replay next (similar to BFS expansion strategy)
            if (idx_v >= mem_v.size() || (idx_u < mem_u.size() && sum_degs_u <= sum_degs_v)) {
                // Replay next level from U's memory
                // Save state before processing this level
                uint32_t saved_start_u = start_u;
                uint32_t saved_end_u = end_u;
                uint32_t saved_end_q = end_q;
                uint32_t saved_sum_degs_u = sum_degs_u;
                size_t saved_idx_u = idx_u;

                start_u = end_q;
                sum_degs_u = 0;

                while (idx_u < mem_u.size() && mem_u[idx_u].dist == curr_dist_u) {
                    uint32_t y = mem_u[idx_u].node_id;

                    // Check for collision
                    if (ball_indicator[y] == VISITED_V) {
                        collision_detected = true;
                        break;  // Stop processing this level
                    }

                    ball_indicator[y] = VISITED_U;
                    dist[y] = curr_dist_u;
                    n_paths_u[y] = mem_u[idx_u].n_paths;
                    p_u[y] = mem_u[idx_u].p;  // Restore accumulated projection
                    q[end_q++] = y;
                    sum_degs_u += static_cast<uint32_t>(DEG(y));
                    counters.nodes_mem_replayed++;  // Count replayed node
                    idx_u++;
                }

                if (collision_detected) {
                    // Rollback: reset ball_indicator for nodes we just added
                    for (uint32_t i = saved_end_q; i < end_q; i++) {
                        ball_indicator[q[i]] = UNVISITED;
                    }
                    // Restore state
                    start_u = saved_start_u;
                    end_u = saved_end_u;
                    end_q = saved_end_q;
                    sum_degs_u = saved_sum_degs_u;
                    idx_u = saved_idx_u;
                    break;  // Stop replay
                }

                end_u = end_q;
                curr_dist_u++;
            } else {
                // Replay next level from V's memory
                // Save state before processing this level
                uint32_t saved_start_v = start_v;
                uint32_t saved_end_v = end_v;
                uint32_t saved_end_q = end_q;
                uint32_t saved_sum_degs_v = sum_degs_v;
                size_t saved_idx_v = idx_v;

                start_v = end_q;
                sum_degs_v = 0;


                while (idx_v < mem_v.size() && mem_v[idx_v].dist == curr_dist_v) {
                    uint32_t y = mem_v[idx_v].node_id;

                    // Check for collision
                    if (ball_indicator[y] == VISITED_U) {
                        collision_detected = true;
                        break;  // Stop processing this level
                    }

                    ball_indicator[y] = VISITED_V;
                    dist[y] = curr_dist_v;
                    n_paths_v[y] = mem_v[idx_v].n_paths;
                    p_v[y] = mem_v[idx_v].p;  // Restore accumulated projection
                    q[end_q++] = y;
                    sum_degs_v += static_cast<uint32_t>(DEG(y));
                    counters.nodes_mem_replayed++;  // Count replayed node
                    idx_v++;
                }

                if (collision_detected) {

                    // Rollback: reset ball_indicator for nodes we just added
                    for (uint32_t i = saved_end_q; i < end_q; i++) {
                        ball_indicator[q[i]] = UNVISITED;
                    }
                    // Restore state
                    start_v = saved_start_v;
                    end_v = saved_end_v;
                    end_q = saved_end_q;
                    sum_degs_v = saved_sum_degs_v;
                    idx_v = saved_idx_v;
                    break;  // Stop replay
                }

                end_v = end_q;
                curr_dist_v++;
            }
        }
    }

    uint32_t replay_end_q = end_q;
    auto end_mem_replay = std::chrono::high_resolution_clock::now();
    counters.durationInSecMemReplay += std::chrono::duration_cast<std::chrono::nanoseconds>(end_mem_replay - start_mem_replay).count() / 1e9;
    // --- SYNCHRONIZED MEMORY REPLAY LOGIC END ---

    std::vector<double> *cur_p;  // Current projection vector being updated
    std::vector<double> *other_p;  // Other projection vector

    auto start_bfs = std::chrono::high_resolution_clock::now();
    while( !have_to_stop ) {
        // Decide which ball should be expanded
        if (sum_degs_u <= sum_degs_v) {
            // Expand from u
            start_cur = start_u; // The index in the queue q of the first node in the current level that needs to be processed (expanded).
            end_cur = end_u; // The index in the queue q that marks the end of the current level (exclusive). It points to the slot after the last node of the current frontier.
            start_u = end_q; // The index in the q array where the current level of the forward search begins.
            new_end_cur = &end_u;
            end_u = end_q; // The index in the q array where the current level of the forward search ends.
            sum_degs_u = 0;
            sum_degs_cur = &sum_degs_u;
            cur_n_paths = &n_paths_u;
            other_n_paths = &n_paths_v;
            cur_p = &p_u;
            other_p = &p_v;
        } else {
            // Expand from v
            start_cur = start_v;
            end_cur = end_v;
            start_v = end_q;
            new_end_cur = &end_v;
            end_v = end_q;
            sum_degs_v = 0;
            sum_degs_cur = &sum_degs_v;
            cur_n_paths = &n_paths_v;
            other_n_paths = &n_paths_u;
            cur_p = &p_v;
            other_p = &p_u;
        }

        // Expand first ball.
        while( start_cur < end_cur ) {
            x = q[start_cur++];
            for( auto y : A[x]) {
                // vis_edges++;
                if( ball_indicator[y] == 0 ) {
                    (*sum_degs_cur) += DEG(y);
                    (*cur_n_paths)[y] = (*cur_n_paths)[x];
                    (*cur_p)[y] = (*cur_p)[x] + RD(x, y) * (*cur_n_paths)[x];  // Initialize projection
                    ball_indicator[y] = ball_indicator[x];
                    counters.nodes_bfs++;  // Count node handled by actual BFS
                    q[end_q++] = y;
                    (*new_end_cur)++;
                    dist[y] = dist[x] + 1;
                } else if (ball_indicator[y]!=ball_indicator[x]) {
                    have_to_stop = true;
                    sp_edges.push_back( std::pair< int64_t, int64_t > (x, y) );
                    tot_weight += (*cur_n_paths)[x] * (*other_n_paths)[y];
                }
                else if (dist[y] == dist[x] + 1) {
                    (*cur_n_paths)[y] += (*cur_n_paths)[x];
                    (*cur_p)[y] += (*cur_p)[x] + RD(x, y) * (*cur_n_paths)[x];  // Accumulate projection
                }
            }
        }
        if (*sum_degs_cur == 0) {
            have_to_stop = true;
        }
    }




    auto end_bfs = std::chrono::high_resolution_clock::now();
    counters.durationInSecBbBFS += std::chrono::duration_cast<std::chrono::nanoseconds>(end_bfs - start_bfs).count() / 1e9;


    // At the end, end_q is the length of the queue q.
    if (sp_edges.size() == 0) {
        // No shortest path found (disconnected nodes)
        // Have to reset ball_indicator!
        for (uint32_t i = 0; i < end_q; i++) {
            ball_indicator[q[i]] = UNVISITED;
        }
        // Throw error
        std::cout << "Error: sp_edges empty for u: " << u << " and v: " << v << std::endl;
        auto end_estimator = std::chrono::high_resolution_clock::now();
        counters.durationInSecCcBbBFS += std::chrono::duration_cast<std::chrono::nanoseconds>(end_estimator - start_estimator).count() / 1e9;
        return 0.0;  // Return 0 for disconnected nodes
    }

    // Compute length of the shortest path from u to v.
    int path_length = dist[sp_edges[0].first] + dist[sp_edges[0].second] + 1;
    // Track distance statistics
    // total_path_distance += path_length;
    // total_valid_paths++;

    // --- MEMORY POPULATION LOGIC START ---
    auto start_mem_pop = std::chrono::high_resolution_clock::now();

    uint32_t prev_mem_depth_u = mem_depth_per_node[u];
    uint32_t prev_mem_depth_v = mem_depth_per_node[v];
    if (std::max<uint32_t>(dist[sp_edges[0].first], dist[sp_edges[0].second]) <= max_mem_depth) {

        // Pre-allocate space to avoid repeated reallocations during memory population
        if (end_cur > replay_end_q && !mem_per_node.empty()) {
            uint32_t potential_new_entries = end_cur - replay_end_q;

            size_t current_size_u = mem_per_node[u].size();
            size_t current_capacity_u = mem_per_node[u].capacity();
            if (current_size_u + potential_new_entries > current_capacity_u) {
                mem_per_node[u].reserve(current_size_u + potential_new_entries);
            }

            size_t current_size_v = mem_per_node[v].size();
            size_t current_capacity_v = mem_per_node[v].capacity();
            if (current_size_v + potential_new_entries > current_capacity_v) {
                mem_per_node[v].reserve(current_size_v + potential_new_entries);
            }
        }
        for (uint32_t i = replay_end_q; i < end_cur; i++) { // Must not use end_q because some nodes at the end of q may be incomplete for one distance level
            uint32_t node = q[i];

            if (ball_indicator[node] == VISITED_U) {
                // Node belongs to U-ball
                // Check if we should memoize this node: it must be deeper than previous memory AND within max_mem_depth limit
                if ((dist[node] > prev_mem_depth_u)) {
                    mem_per_node[u].push_back({node, (uint32_t)dist[node], (uint64_t)n_paths_u[node], p_u[node]});
                    mem_depth_per_node[u] = std::max<uint32_t>(mem_depth_per_node[u], dist[node]);
                }
            } else if (ball_indicator[node] == VISITED_V) {
                // Node belongs to V-ball
                // Check if we should memoize this node: it must be deeper than previous memory AND within max_mem_depth limit
                if ((dist[node] > prev_mem_depth_v)) {
                    mem_per_node[v].push_back({node, (uint32_t)dist[node], (uint64_t)n_paths_v[node], p_v[node]});
                    mem_depth_per_node[v] = std::max<uint32_t>(mem_depth_per_node[v], dist[node]);
                }
            }
        }
    }

    auto end_mem_pop = std::chrono::high_resolution_clock::now();
    counters.durationInSecMemPopulation += std::chrono::duration_cast<std::chrono::nanoseconds>(end_mem_pop - start_mem_pop).count() / 1e9;
    // --- MEMORY POPULATION LOGIC END ---

    // For exact estimator, we consider ALL paths, so num_paths is the total number of shortest paths.
    // We do not scale the intermediate counts, so we effectively pass alpha=1.0 conceptually,
    // but without the truncation error.
    num_paths = (double)tot_weight;
    p_uv = AccumulateProjection(sp_edges, ball_indicator, RD, n_paths_u, n_paths_v, p_u, p_v);

    if (p_uv <= 0.0) {
        // throw error
        std::cout << "Error: p_uv <= 0.0 for u: " << u << " and v: " << v << std::endl;
        std::cout << "  Distance from u to v: " << path_length << std::endl;
        std::cout << "  Number of shortest paths: " << num_paths << std::endl;
        std::cout << "  sp_edges size: " << sp_edges.size() << std::endl;
        if (!sp_edges.empty()) {
            std::cout << "  First edge: (" << sp_edges[0].first << ", " << sp_edges[0].second << ")" << std::endl;
        }
        // Have to reset ball_indicator!
        for (uint32_t i = 0; i < end_q; i++) {
            ball_indicator[q[i]] = UNVISITED;
        }
        auto end_estimator = std::chrono::high_resolution_clock::now();
        counters.durationInSecCcBbBFS += std::chrono::duration_cast<std::chrono::nanoseconds>(end_estimator - start_estimator).count() / 1e9;
        return 0.0;
    }

    // Have to reset ball_indicator!
    for (uint32_t i = 0; i < end_q; i++) {
        ball_indicator[q[i]] = UNVISITED;
    }

    p_uv_mean = p_uv / num_paths;
    contrib = RD(u, v) / p_uv_mean;

    auto end_estimator = std::chrono::high_resolution_clock::now();
    counters.durationInSecCcBbBFS += std::chrono::duration_cast<std::chrono::nanoseconds>(end_estimator - start_estimator).count() / 1e9;

    return contrib;
}

// The approximate algorithm implementation
Result CcApproxGc(
        const Eigen::MatrixXd& adj_matrix,
        const AlgorithmParams& params) {

    /// computation starts
    COUNTERS counters;
    Timer global_timer;
    Timer local_timer;
    global_timer.Start();
    int64_t n = adj_matrix.rows();

    PrintAlgorithmHeader(AlgorithmType::GC_approx, params);

    /// compute degree and adjacency list
    local_timer.Start();
    Eigen::VectorXd DEG(n);
    for (int64_t i = 0; i < n; ++i) {
        DEG(i) = adj_matrix.col(i).sum();
    }
    int64_t m = DEG.sum();
    double density = 0.0;
    if (n > 1) {
        density = static_cast<double>(m) / (static_cast<double>(n) * static_cast<double>(n - 1));
    }
    std::vector<std::vector<int64_t>> A;
    std::vector<std::vector<int64_t>> nonA;
    bool use_nonadj_sampling = density >= 0.9;
    if (use_nonadj_sampling) {
        auto lists = ComputeAdjAndNonAdjList(adj_matrix, n);
        A = std::move(lists.first);
        nonA = std::move(lists.second);
    } else {
        A = ComputeAdjList(adj_matrix, n);
    }
    local_timer.Stop();
    counters.durationInSecComputeAdjList = local_timer.Seconds();
    if (params.verbose) {
        std::cout << "@@ Degree and Adjacency list computation time: " << local_timer.Seconds() << std::endl;
    }

    if (m == n * (n - 1)) {
        throw std::runtime_error("GC_approx requires at least one non-adjacent node pair to sample.");
    }

    // Calculate average degree and max_mem_depth (log_n_w)
    double avg_deg = static_cast<double>(m) / static_cast<double>(n);
    uint32_t max_mem_depth = static_cast<uint32_t>(std::log2(static_cast<double>(n)) * std::log2(avg_deg));
    if (params.verbose) {
        std::cout << "Average degree: " << avg_deg << std::endl;
        std::cout << "Max mem depth (log_n_w): " << max_mem_depth << std::endl;
    }

    /// initialize variables
    local_timer.Start();

    // Check if RD matrix is provided
    if (!params.rd_matrix.has_value()) {
        throw std::invalid_argument("RD matrix is required for approximate GC computation.");
    }
    Eigen::MatrixXd RD = params.rd_matrix.value();  // Referential distance
    double GC_approx = 0.0;  // Geometrical congruence

    // Allocate BFS data structures once for reuse across samples
    std::vector<uint32_t> ball_indicator(n, UNVISITED);
    std::vector<int64_t> q(n);  // BFS queue
    std::vector<double> n_paths_u(n);  // Number of shortest paths from u
    std::vector<double> n_paths_v(n);  // Number of shortest paths from v
    std::vector<int32_t> dist(n);  // Distance array
    std::vector<std::vector<MemNode>> mem_per_node(n);  // Memory for visited nodes
    std::vector<uint32_t> mem_depth_per_node(n, 0);  // Depth of memory per node
    std::vector<double> p_u(n);  // Accumulated TSP projection from u to other nodes
    std::vector<double> p_v(n);  // Accumulated TSP projection from v to other nodes

    local_timer.Stop();
    counters.durationInSecAPSPNInit = local_timer.Seconds();
    if (params.verbose) {
        std::cout << "@@ Initialization time: " << local_timer.Seconds() << std::endl;
    }

    /// Main approximate computation loop
    local_timer.Start();

    // Calculate number of samples based on sampling factor
    // num_samples = n * sampling_factor
    int64_t num_samples = static_cast<int64_t>(std::ceil(n * params.sampling_factor));

    if (params.verbose) {
        std::cout << "Number of nodes: " << n << std::endl;
        std::cout << "Sampling factor: " << params.sampling_factor << std::endl;
        std::cout << "Number of samples: " << num_samples << std::endl;
    }

    // Sample node pairs and accumulate projections
    GC_approx = 0.0;
    std::vector<uint64_t> nonadj_prefix;
    uint64_t total_nonadj = 0;
    if (use_nonadj_sampling) {
        nonadj_prefix.resize(n + 1, 0);
        for (int64_t i = 0; i < n; ++i) {
            nonadj_prefix[i + 1] = nonadj_prefix[i] + nonA[i].size();
        }
        total_nonadj = nonadj_prefix[n];
        if (total_nonadj == 0) {
            throw std::runtime_error("GC_approx requires at least one non-adjacent node pair to sample.");
        }
    }
    uint32_t u = 0;
    uint32_t v = 0;
    for (int64_t sample = 0; sample < num_samples; sample++) {
        if (use_nonadj_sampling) {
            uint64_t r = RandomIndex(total_nonadj);
            auto it = std::upper_bound(nonadj_prefix.begin(), nonadj_prefix.end(), r);
            uint32_t u_idx = static_cast<uint32_t>(it - nonadj_prefix.begin() - 1);
            uint64_t offset = r - nonadj_prefix[u_idx];
            u = u_idx;
            v = static_cast<uint32_t>(nonA[u_idx][offset]);
        } else {
            u = RandomNode(n);
            v = RandomNode(n);
            while (u == v || adj_matrix(u, v) != 0.0) {
                u = RandomNode(n);
                v = RandomNode(n);
            }
        }
        double contrib = CcBbBFS(n, DEG, A, u, v, ball_indicator, RD,
                                            q, n_paths_u, n_paths_v, dist,
                                            mem_per_node, mem_depth_per_node, max_mem_depth,
                                            p_u, p_v, counters);
        GC_approx += contrib;
    }

    // Compute the average
    GC_approx /= num_samples;

    if (params.verbose) {
        std::cout << "Approximate GC: " << GC_approx << std::endl;
    }

    local_timer.Stop();
    counters.durationInSecAPSPNComputation = local_timer.Seconds();

    if (params.verbose) {
        std::cout << "@@ Approximate computation costs: " << std::fixed << std::setprecision(2)
                  << local_timer.Seconds() << std::endl;

        // Print memory performance statistics
        std::cout << "=== Memory Performance Statistics ===" << std::endl;
        std::cout << "@@Time in CcBbBFS (s): " << counters.durationInSecCcBbBFS << std::endl;
        std::cout << "@@Time in memory replay (s): " << counters.durationInSecMemReplay << std::endl;
        std::cout << "@@Time in memory population (s): " << counters.durationInSecMemPopulation << std::endl;
        std::cout << "@@Time in bbBFS (s): " << counters.durationInSecBbBFS << std::endl;

        int64_t total_nodes_handled = counters.nodes_mem_replayed + counters.nodes_bfs;
        std::cout << "Nodes handled by memory replay: " << counters.nodes_mem_replayed << std::endl;
        std::cout << "Nodes handled by actual bbBFS: " << counters.nodes_bfs << std::endl;
        if (total_nodes_handled > 0) {
            double replay_ratio = (double)counters.nodes_mem_replayed / (double)total_nodes_handled;
            std::cout << "Memory replay ratio: " << (replay_ratio * 100.0) << "%" << std::endl;
        } else {
            std::cout << "Memory replay ratio: N/A (no nodes handled)" << std::endl;
        }
        std::cout << "====================================" << std::endl;
    }

    /// computation ends
    global_timer.Stop();
    counters.durationInSecTotal = global_timer.Seconds();

    if (params.verbose) {
        PrintTotalComputationTime(counters.durationInSecTotal, params.verbose);
    }

    /// Return the result
    return Result{GC_approx, counters, global_timer.Seconds()};
}
