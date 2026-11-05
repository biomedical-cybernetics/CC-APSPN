// Derived from SILVAN by L. Pellegrina and F. Vandin
// (https://github.com/leonardopellegrina/SILVAN), which builds on KADABRA by
// M. Borassi and E. Natale. Licensed under the Apache License 2.0; see
// THIRD_PARTY_NOTICES.md in the CC-APSPN release. Modified for CC-APSPN.
#define NOMINMAX
#include "Sp_sampler.h"
#include "Rand_gen.h"
#include <iostream>
#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_map>

#define UNVISITED 0
#define VISITED_U 1
#define VISITED_V 2

using namespace std;

namespace cc_approxbc_lib {

static std::atomic<uint64_t> global_n_exact_calls(0);
static std::atomic<uint64_t> global_n_sampling_calls(0);
static std::atomic<uint64_t> global_time_exact_estimator_ns(0);
static std::atomic<uint64_t> global_time_bbbfs_ns(0);
static std::atomic<uint64_t> global_time_mem_population_ns(0);
static std::atomic<uint64_t> global_time_random_path_ns(0);
static std::atomic<uint64_t> global_mem_collisions(0);
static std::atomic<uint64_t> global_nodes_replayed(0);
static std::atomic<uint64_t> global_nodes_bfs(0);

// Sampling statistics
static std::mutex sampling_stats_mutex;
static std::unordered_map<uint32_t, uint64_t> node_sampling_count;
static std::atomic<uint64_t> total_node_samples(0);
static std::atomic<uint64_t> total_path_distance(0);
static std::atomic<uint64_t> total_valid_paths(0);

// Define static mem members
std::vector<std::vector<MemNode>> Sp_sampler::mem_per_node;
std::vector<uint32_t> Sp_sampler::mem_depth_per_node;
uint32_t Sp_sampler::max_mem_depth = UINT32_MAX;  // Default: infinity (no limit)
bool Sp_sampler::enable_mem = true;

void Sp_sampler::set_enable_mem(bool enable) {
    enable_mem = enable;
}

void Sp_sampler::reset_global_counters() {
    global_n_exact_calls = 0;
    global_n_sampling_calls = 0;
    global_time_exact_estimator_ns = 0;
    global_time_bbbfs_ns = 0;
    global_time_mem_population_ns = 0;
    global_time_random_path_ns = 0;
    global_mem_collisions = 0;
    global_nodes_replayed = 0;
    global_nodes_bfs = 0;

    // Reset sampling statistics
    std::lock_guard<std::mutex> lock(sampling_stats_mutex);
    node_sampling_count.clear();
    total_node_samples = 0;
    total_path_distance = 0;
    total_valid_paths = 0;
}

void Sp_sampler::print_global_counters() {
    uint64_t exact = global_n_exact_calls.load();
    uint64_t sampling = global_n_sampling_calls.load();
    uint64_t time_exact = global_time_exact_estimator_ns.load();
    uint64_t time_bbbfs = global_time_bbbfs_ns.load();
    uint64_t time_memoization_pop = global_time_mem_population_ns.load();
    uint64_t time_random_path = global_time_random_path_ns.load();
    uint64_t collisions = global_mem_collisions.load();
    uint64_t nodes_replayed = global_nodes_replayed.load();
    uint64_t nodes_bfs = global_nodes_bfs.load();

    std::cout << "Exact AB estimator calls: " << exact << std::endl;
    std::cout << "Sampling AB estimator calls: " << sampling << std::endl;
    if (sampling > 0) {
        std::cout << "Ratio (Exact/Sampling): " << (double)exact / (double)sampling << std::endl;
    } else {
        std::cout << "Ratio (Exact/Sampling): N/A (Sampling=0)" << std::endl;
    }
    std::cout << "@@Time in cc_bbbfs (s): " << (double)time_random_path / 1e9 << std::endl;
    std::cout << "@@Time in Exact AB estimator (s): " << (double)time_exact / 1e9 << std::endl;
    std::cout << "@@Time in bbBFS (s): " << (double)time_bbbfs / 1e9 << std::endl;
    std::cout << "@@Time in memoization population (s): " << (double)time_memoization_pop / 1e9 << std::endl;


    // Cache statistics
    if (!mem_depth_per_node.empty()) {
        uint64_t total_depth = 0;
        uint64_t nodes_with_memoization = 0;
        uint64_t max_depth = 0;

        for (size_t i = 0; i < mem_depth_per_node.size(); i++) {
            uint32_t depth = mem_depth_per_node[i];
            if (depth > 0) {
                total_depth += depth;
                nodes_with_memoization++;
                if (depth > max_depth) {
                    max_depth = depth;
                }
            }
        }

        std::cout << "=== Memoization Statistics ===" << std::endl;
        std::cout << "Memoization replay collisions: " << collisions << std::endl;
        std::cout << "Nodes with memoization: " << nodes_with_memoization << " / " << mem_depth_per_node.size() << std::endl;
        if (nodes_with_memoization > 0) {
            double avg_depth = (double)total_depth / (double)nodes_with_memoization;
            std::cout << "Average memoization depth: " << avg_depth << std::endl;
            std::cout << "Max memoization depth: " << max_depth << std::endl;
        } else {
            std::cout << "Average memoization depth: N/A (no memoization)" << std::endl;
        }

        // Node handling statistics
        uint64_t total_nodes_handled = nodes_replayed + nodes_bfs;
        std::cout << "Nodes handled by memoization replay: " << nodes_replayed << std::endl;
        std::cout << "Nodes handled by actual bbBFS: " << nodes_bfs << std::endl;
        if (total_nodes_handled > 0) {
            double replay_ratio = (double)nodes_replayed / (double)total_nodes_handled;
            std::cout << "Memoization replay ratio: " << (replay_ratio * 100.0) << "%" << std::endl;
        } else {
            std::cout << "Memoization replay ratio: N/A (no nodes handled)" << std::endl;
        }
        std::cout << "========================" << std::endl;
    }
}

void Sp_sampler::print_sampling_stats() {
    std::lock_guard<std::mutex> lock(sampling_stats_mutex);

    uint64_t total_samples = total_node_samples.load();
    uint64_t distinct_nodes = node_sampling_count.size();
    uint64_t total_dist = total_path_distance.load();
    uint64_t valid_paths = total_valid_paths.load();

    std::cout << "=== Node Sampling Statistics ===" << std::endl;
    std::cout << "Total node samples: " << total_samples << std::endl;
    std::cout << "Distinct nodes sampled: " << distinct_nodes << std::endl;

    if (distinct_nodes > 0) {
        double avg_samples_per_node = (double)total_samples / (double)distinct_nodes;
        std::cout << "Average samples per distinct node: " << avg_samples_per_node << std::endl;
    } else {
        std::cout << "Average samples per distinct node: N/A (no nodes sampled)" << std::endl;
    }

    if (valid_paths > 0) {
        double avg_distance = (double)total_dist / (double)valid_paths;
        std::cout << "Average distance between sampled u and v: " << avg_distance << std::endl;
    } else {
        std::cout << "Average distance between sampled u and v: N/A (no valid paths)" << std::endl;
    }
    std::cout << "================================" << std::endl;
}

void Sp_sampler::reset_mem() {
    mem_per_node.clear();
    mem_depth_per_node.clear();
}

void Sp_sampler::set_max_mem_depth(uint32_t depth) {
    max_mem_depth = depth;
}

// Instantiates an object, that samples from the graph g.
// INPUT: g (a graphs), seed (the seed for the random sampler).
Sp_sampler::Sp_sampler( const Graph *g, const uint32_t seed ) {
    uint32_t n = g->get_nn();
    q = (uint32_t*) malloc( n*sizeof(uint32_t));
    ball_indicator = (uint32_t*) calloc( n, sizeof(uint32_t));
    dist = (uint32_t*) malloc( n*sizeof(uint32_t));

    randgen = new Rand_gen( seed );
    n_paths_u = (double*) malloc( n*sizeof(double));
    n_paths_v = (double*) malloc( n*sizeof(double));
    this->g = g;
    n_exact_calls = 0;
    n_sampling_calls = 0;

    // Initialize static per-node mem structures only once (on first instantiation)
    if (enable_mem && mem_per_node.empty()) {
        mem_per_node.resize(n);
        mem_depth_per_node.assign(n, 0);

        // Pre-allocate mem for each node (heuristic: 2x degree)
        for (uint32_t i = 0; i < n; i++) {
            mem_per_node[i].reserve(2 * g->degrees[i]);
        }
    }
}

// Returns a random node.
inline uint32_t Sp_sampler::random_node() const {
    return randgen->get_max(g->get_nn());
}

// Return a map of nodes to the number of shortest paths touching them
map<uint32_t, double>/*vector<uint32_t>*/ Sp_sampler::cc_bbbfs(int &path_length , double &num_paths) {
    auto start_random_path = std::chrono::high_resolution_clock::now();

    uint32_t v = random_node(), u = random_node();
    while (u == v) {
        v = random_node();
    }

    uint32_t *degrees = g->degrees;
    uint32_t *in_degrees = g->in_degrees;
    uint32_t **adj = g->adj;
    uint32_t **inc = g->inc;
    std::map<uint32_t, double> path_map;
    num_paths = 0.0;

    // Check for small distance
    uint32_t first_node, second_node;
    uint32_t first_deg, second_deg;
    uint32_t* first_adj;
    uint32_t* second_adj;
    bool short_path = false;
    if (degrees[u] <= in_degrees[v]) {
        first_node = u;
        second_node = v;
        first_adj = adj[first_node];
        first_deg = degrees[first_node];
        second_adj = inc[second_node];
        second_deg = in_degrees[second_node];

    } else {
        first_node = v;
        second_node = u;
        first_adj = inc[first_node];
        first_deg = in_degrees[first_node];
        second_adj = adj[second_node];
        second_deg = degrees[second_node];
    }
    // Mark neighbors of first node
    for (uint32_t i = 0; i < first_deg; i++) {
        uint32_t w = first_adj[i];
        if (w == second_node) {
            short_path = true;
            path_length = 1;
            num_paths = 1.0;
            break;
        }
        ball_indicator[w] = 1;
    }

    // Scan neighbors of second node
    if (!short_path) {
        for (uint32_t i = 0; i < second_deg; i++) {
            uint32_t w = second_adj[i];
            if (ball_indicator[w] == 1) {
                short_path = true;
                path_length = 2;
                path_map[w] = 1.0;
                num_paths += 1.0;
            }
        }
    }

    // Reset ball_indicator
    for (uint32_t i = 0; i < first_deg; i++) {
        ball_indicator[first_adj[i]] = UNVISITED;
    }

    // If common neighbors found, return directly
    if (short_path) {
        total_path_distance += path_length;
        total_valid_paths++;
        auto end_random_path = std::chrono::high_resolution_clock::now();
        global_time_random_path_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_random_path - start_random_path).count();
        return path_map;
    }

    // Sample sp
    uint32_t end_q = 0;
    vis_edges = 0;
    sp_dag_edges = 0;
    // Bidirect_bfs
    vector<pair< uint32_t, uint32_t > > sp_edges;
    uint32_t x, y;
    bool have_to_stop = false;
    uint32_t start_u = 0, start_v = 1, end_u = 1, end_v = 2, start_cur, end_cur, *new_end_cur;
    uint32_t sum_degs_u = 0, sum_degs_v = 0, *sum_degs_cur;
    uint32_t neigh_num;


    // Track sampled nodes (only after ensuring u != v)
    {
        std::lock_guard<std::mutex> lock(sampling_stats_mutex);
        node_sampling_count[u]++;
        node_sampling_count[v]++;
        total_node_samples += 2;
    }

    bool guess_void = false;
    if(g->directed){
      if(g->cc[u] < g->cc[v]){
        guess_void = true;
        auto end_random_path = std::chrono::high_resolution_clock::now();
        global_time_random_path_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_random_path - start_random_path).count();
        return std::map<uint32_t,double>();//vector<uint32_t>();
      }
    }
    else{
      if(g->cc[u] != g->cc[v]){
        guess_void = true;
        auto end_random_path = std::chrono::high_resolution_clock::now();
        global_time_random_path_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_random_path - start_random_path).count();
        return std::map<uint32_t,double>();//vector<uint32_t>();
      }
    }

    auto start_bfs = std::chrono::high_resolution_clock::now();

    end_q = 2;

    q[0] = u;
    q[1] = v;

    ball_indicator[u] = VISITED_U;
    ball_indicator[v] = VISITED_V;

    n_paths_u[u] = 1;
    dist[u] = 0;
    n_paths_v[v] = 1;
    dist[v] = 0;

    double *cur_n_paths;
    double *other_n_paths;
    double tot_weight = 0;

    // --- SYNCHRONIZED MEMORY REPLAY LOGIC START ---
    uint32_t replay_end_q = end_q;
    if (enable_mem) {
        const auto& memo_u = mem_per_node[u];
        const auto& memo_v = mem_per_node[v];
        bool collision_detected = false;

        // Skip memory replay if depth <= 1
        if (!memo_u.empty() || !memo_v.empty()) {
            size_t idx_u = 0;
            size_t idx_v = 0;
            uint32_t curr_dist_u = 1;  // Track distance for U's replay
            uint32_t curr_dist_v = 1;  // Track distance for V's replay

            // alternate between U and V based on sum_degs
            // Continue replay while either endpoint has a stored level left.
            while (idx_u < memo_u.size() || idx_v < memo_v.size()) {
                // Decide which source to replay next (similar to BFS expansion strategy)
                if (idx_v >= memo_v.size() || (idx_u < memo_u.size() && sum_degs_u <= sum_degs_v)) {
                    // Replay next level from U's memory
                    // Save state before processing this level
                    uint32_t saved_start_u = start_u;
                    uint32_t saved_end_u = end_u;
                    uint32_t saved_end_q = end_q;
                    uint32_t saved_sum_degs_u = sum_degs_u;
                    size_t saved_idx_u = idx_u;

                    start_u = end_q;
                    sum_degs_u = 0;

                    while (idx_u < memo_u.size() && memo_u[idx_u].dist == curr_dist_u) {
                        uint32_t y = memo_u[idx_u].node_id;

                        // Check for collision
                        if (ball_indicator[y] == VISITED_V) {
                            collision_detected = true;
                            break;  // Stop processing this level
                        }

                        ball_indicator[y] = VISITED_U;
                        dist[y] = curr_dist_u;
                        n_paths_u[y] = memo_u[idx_u].n_paths;
                        q[end_q++] = y;
                        sum_degs_u += g->degrees[y];
                        global_nodes_replayed++;  // Count replayed node
                        idx_u++;
                    }

                    if (collision_detected) {
                        // Count collision
                        global_mem_collisions++;

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


                    while (idx_v < memo_v.size() && memo_v[idx_v].dist == curr_dist_v) {
                        uint32_t y = memo_v[idx_v].node_id;

                        // Check for collision
                        if (ball_indicator[y] == VISITED_U) {
                            collision_detected = true;
                            break;  // Stop processing this level
                        }

                        ball_indicator[y] = VISITED_V;
                        dist[y] = curr_dist_v;
                        n_paths_v[y] = memo_v[idx_v].n_paths;
                        q[end_q++] = y;
                        sum_degs_v += g->in_degrees[y];
                        global_nodes_replayed++;  // Count replayed node
                        idx_v++;
                    }

                    if (collision_detected) {
                        // Count collision
                        global_mem_collisions++;

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

        replay_end_q = end_q;
    }
    // --- SYNCHRONIZED MEMORY REPLAY LOGIC END ---

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
            degrees = g->degrees;
            adj = g->adj;
            cur_n_paths = n_paths_u;
            other_n_paths = n_paths_v;
        } else {
            // Expand from v
            start_cur = start_v;
            end_cur = end_v;
            start_v = end_q;
            new_end_cur = &end_v;
            end_v = end_q;
            sum_degs_v = 0;
            sum_degs_cur = &sum_degs_v;
            degrees = g->in_degrees;
            adj = g->inc;
            cur_n_paths = n_paths_v;
            other_n_paths = n_paths_u;
        }

        // Expand first ball.
        while( start_cur < end_cur ) {
            x = q[start_cur++];
            neigh_num = degrees[x];

            for( uint32_t j=0; j<neigh_num; j++ ){
                vis_edges++;
                y = adj[x][j];
                if( ball_indicator[y] == 0 ) {
                    (*sum_degs_cur) += degrees[y];
                    cur_n_paths[y] = cur_n_paths[x];
                    ball_indicator[y] = ball_indicator[x];
                    global_nodes_bfs++;  // Count node handled by actual BFS
                    q[end_q++] = y;
                    (*new_end_cur)++;
                    dist[y] = dist[x] + 1;
                } else if (ball_indicator[y]!=ball_indicator[x]) {
                    have_to_stop = true;
                    sp_edges.push_back( pair< uint32_t, uint32_t > (x, y) );
                    tot_weight += cur_n_paths[x] * other_n_paths[y];
                }
                else if (dist[y] == dist[x] + 1) {
                    cur_n_paths[y] += cur_n_paths[x];
                }
            }
        }
        if (*sum_degs_cur == 0) {
            have_to_stop = true;
        }
    }




    auto end_bfs = std::chrono::high_resolution_clock::now();
    global_time_bbbfs_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_bfs - start_bfs).count();


    // At the end, end_q is the length of the queue q.
    if (sp_edges.size() == 0) {

        if(g->directed == false && guess_void == false){
          std::cout << "Error: guess void false, sp_edges empty, path.size() == 0 " << std::endl;
          std::cout << " u: " << u << " v: " << v <<  " u sampling count: " << node_sampling_count[u] << " v sampling count: " << node_sampling_count[v] << std::endl;
        }

        // Have to reset ball_indicator!
        for (uint32_t i = 0; i < end_q; i++) {
            ball_indicator[q[i]] = UNVISITED;
        }

        auto end_random_path = std::chrono::high_resolution_clock::now();
        global_time_random_path_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_random_path - start_random_path).count();
        return std::map<uint32_t,double>();//vector<uint32_t>();
    }

    // Compute length of the shortest path from u to v.
    path_length = 0;
    path_length = dist[sp_edges[0].first] + dist[sp_edges[0].second] + 1;
    // Track distance statistics
    total_path_distance += path_length;
    total_valid_paths++;

    // --- MEMORY POPULATION LOGIC START ---
    auto start_mem_pop = std::chrono::high_resolution_clock::now();

    if (enable_mem) {
        uint32_t prev_memoization_depth_u = mem_depth_per_node[u];
        uint32_t prev_memoization_depth_v = mem_depth_per_node[v];
        if (std::max(dist[sp_edges[0].first], dist[sp_edges[0].second]) <= max_mem_depth) {

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
                    // Check if we should memoize this node: it must be deeper than previous memoization AND within max_mem_depth limit
                    if ((dist[node] > prev_memoization_depth_u)) {
                        mem_per_node[u].push_back({node, dist[node], (uint64_t)n_paths_u[node]});
                        mem_depth_per_node[u] = std::max(mem_depth_per_node[u], dist[node]);
                    }
                } else if (ball_indicator[node] == VISITED_V) {
                    // Node belongs to V-ball
                    // Check if we should memoize this node: it must be deeper than previous memoization AND within max_mem_depth limit
                    if ((dist[node] > prev_memoization_depth_v)) {
                        mem_per_node[v].push_back({node, dist[node], (uint64_t)n_paths_v[node]});
                        mem_depth_per_node[v] = std::max(mem_depth_per_node[v], dist[node]);
                    }
                }
            }
        }
    }

    auto end_mem_pop = std::chrono::high_resolution_clock::now();
    global_time_mem_population_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_mem_pop - start_mem_pop).count();
    // --- MEMORY POPULATION LOGIC END ---

    // For AB estimator, we consider ALL paths, so num_paths is the total number of shortest paths.
    // We do not scale the intermediate counts, so we effectively pass alpha=1.0 conceptually,
    // but without the truncation error.
    num_paths = (double)tot_weight;
    n_exact_calls++;
    auto start = std::chrono::high_resolution_clock::now();

    path_map = exact_dep_agg(u, v, end_q, sp_edges);
    auto end = std::chrono::high_resolution_clock::now();
    global_time_exact_estimator_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Have to reset ball_indicator!
    for (uint32_t i = 0; i < end_q; i++) {
        ball_indicator[q[i]] = UNVISITED;
    }
    if(guess_void && path_map.size() > 0){
      std::cout << "Error: guess void true, path.size() > 0 " << std::endl;
      std::cout << "   cc[u] " << g->cc[u] << " cc[v] " << g->cc[v] << std::endl;
    }
    //return path;
    auto end_random_path = std::chrono::high_resolution_clock::now();
    global_time_random_path_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_random_path - start_random_path).count();
    return path_map;
}

std::map<uint32_t, double> Sp_sampler::exact_dep_agg(const uint32_t u, const uint32_t v, const uint32_t end_q, const std::vector<std::pair<uint32_t, uint32_t>> &sp_edges) {
    // Propagate number of shortest paths from the meeting edges back to u and v
    // First, initialize the counts for the "other" side to 0 for visited nodes
    for (uint32_t i = 0; i < end_q; i++) {
        if (ball_indicator[q[i]] == VISITED_U) {
            n_paths_v[q[i]] = 0;
        } else if (ball_indicator[q[i]] == VISITED_V) {
            n_paths_u[q[i]] = 0;
        }
    }

    // Seed the counts at the meeting edges
    for (pair<uint32_t, uint32_t> p : sp_edges) {
        if (ball_indicator[p.first] == VISITED_U) {
             n_paths_v[p.first] += n_paths_v[p.second];
             n_paths_u[p.second] += n_paths_u[p.first];
        } else {
             n_paths_u[p.first] += n_paths_u[p.second];
             n_paths_v[p.second] += n_paths_v[p.first];
        }
    }

    // Propagate backwards
    uint32_t i = end_q;
    while(i > 0) {
        i--;
        uint32_t z = q[i];

        if (ball_indicator[z] == VISITED_U) {
            if (n_paths_v[z] == 0) continue;

            // Parents are incoming neighbors (from u's perspective)
            const uint32_t *parents = g->inc[z];
            uint32_t n_parents = g->in_degrees[z];

            for(uint32_t k=0; k<n_parents; k++){
                uint32_t p = parents[k];
                if (ball_indicator[p] == VISITED_U && dist[p] == dist[z] - 1) {
                    n_paths_v[p] += n_paths_v[z];
                }
            }
        } else if (ball_indicator[z] == VISITED_V) {
            if (n_paths_u[z] == 0) continue;

            // Parents are outgoing neighbors (from v's perspective, backward search follows incoming edges of BFS tree, which are outgoing edges in graph)
            const uint32_t *parents = g->adj[z];
            uint32_t n_parents = g->degrees[z];

            for(uint32_t k=0; k<n_parents; k++){
                uint32_t p = parents[k];
                if (ball_indicator[p] == VISITED_V && dist[p] == dist[z] - 1) {
                    n_paths_u[p] += n_paths_u[z];
                }
            }
        }
    }

    std::map<uint32_t, double> path_map;

    for (uint32_t i = 0; i < end_q; i++) {
        uint32_t z = q[i];
        if (z == u || z == v) continue;
        if (n_paths_u[z] == 0 || n_paths_v[z] == 0) continue; // z does not participate in any shortest path

        // Exact number of shortest paths through z
        double expected_val = (double)n_paths_u[z] * (double)n_paths_v[z];
        path_map[z] = expected_val;
    }

    return path_map;
}



Sp_sampler::~Sp_sampler()
{
    global_n_exact_calls += n_exact_calls;
    global_n_sampling_calls += n_sampling_calls;
    free(q);
    free(ball_indicator);
    free(dist);
    free(n_paths_u);
    free(n_paths_v);
    delete(randgen);
}

} // namespace cc_approxbc_lib
