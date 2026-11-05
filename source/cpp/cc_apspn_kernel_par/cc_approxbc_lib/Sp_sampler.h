// Derived from SILVAN by L. Pellegrina and F. Vandin
// (https://github.com/leonardopellegrina/SILVAN), which builds on KADABRA by
// M. Borassi and E. Natale. Licensed under the Apache License 2.0; see
// THIRD_PARTY_NOTICES.md in the CC-APSPN release. Modified for CC-APSPN.
#ifndef SP_SAMPLER_H
#define SP_SAMPLER_H

#include <cstdint>
#include <vector>
#include <map>
#include <ctype.h>

#include "Rand_gen.h"
#include "Graph.h"

namespace cc_approxbc_lib {

// MemNode stores memoized (cached) information for a node
struct MemNode {
    uint32_t node_id;
    uint32_t dist;
    uint64_t n_paths;
};

class Sp_sampler
{
    public:
        Sp_sampler( const Graph *g, const uint32_t seed );
        virtual ~Sp_sampler();
        std::map<uint32_t, double>/*vector<uint32_t>*/ cc_bbbfs(int &path_length , double &num_paths);
        static void reset_global_counters();
        static void print_global_counters();
        static void print_sampling_stats();
        static void reset_mem();
        static void set_max_mem_depth(uint32_t depth);
        static void set_enable_mem(bool enable);
        uint64_t vis_edges;
        uint64_t sp_dag_edges;
    protected:
    private:
        uint64_t n_exact_calls;
        uint64_t n_sampling_calls;
        std::map<uint32_t, double> exact_dep_agg(const uint32_t u, const uint32_t v, const uint32_t end_q, const std::vector<std::pair<uint32_t, uint32_t>> &sp_edges);
        inline uint32_t random_node() const;
        uint32_t *ball_indicator;
        uint32_t *dist;
        uint32_t *q;
        double *n_paths_u;
        double *n_paths_v;
        Rand_gen *randgen;
        const Graph *g;

        // Static per-node mem: shared across all Sp_sampler instances
        static std::vector<std::vector<MemNode>> mem_per_node;
        static std::vector<uint32_t> mem_depth_per_node;
        static uint32_t max_mem_depth;  // Maximum memoization depth (UINT32_MAX means infinity)
        static bool enable_mem;
};

} // namespace cc_approxbc_lib

#endif // SP_SAMPLER_H
