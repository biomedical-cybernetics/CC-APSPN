/**
 * SILVAN approximate betweenness centrality wrapper for CC-APSPN kernel integration.
 *
 * This file provides a C++ wrapper around the SILVAN approximate BC algorithm
 * that can be called from the unified CC-APSPN kernel MEX interface.
 */
#define NOMINMAX
#ifdef MATLAB_BUILD
#include "matrix.h"
#endif
#include "../cc_apspn_kernel_par.h"
#include "Probabilistic.h"
#include "Sp_sampler.h"
#include "utilities.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <variant>
#include <stdexcept>

using namespace cc_approxbc_lib;

Result CcApproxBc(const std::variant<const mxArray*, const Eigen::MatrixXd*>& adj_input,
                  const AlgorithmParams& params) {
    uint32_t n = 0;
    bool directed = false;
    std::vector<uint32_t> degrees;
    std::vector<uint32_t> in_degrees;
    preprocess_adj_input(adj_input, n, directed, degrees, in_degrees);

    PrintAlgorithmHeader(AlgorithmType::BC_approx, params);

    // Extract parameters from AlgorithmParams
    uint32_t k = 0;  // Always use k=0 for absolute approximation
    double delta = params.delta;
    double err = params.err;
    double emp_peeling_a = params.emp_peeling_a;
    std::string memoization_depth_str = params.memoization_depth;
    uint32_t seed = params.seed;

    // Calculate memoization depth
    uint32_t max_mem_depth = UINT32_MAX;  // Default: infinity (no limit)
    std::string memoization_depth_lower = memoization_depth_str;
    std::transform(memoization_depth_lower.begin(), memoization_depth_lower.end(),
                   memoization_depth_lower.begin(), ::tolower);
    uint64_t total_degree = 0;
    for (uint32_t i = 0; i < n; i++) {
        total_degree += degrees[i];
    }
    double avg_deg = static_cast<double>(total_degree) / static_cast<double>(n);
    if (memoization_depth_lower == "avg_deg") {
        max_mem_depth = static_cast<uint32_t>(avg_deg);
    } else if (memoization_depth_lower == "log_n") {
        max_mem_depth = static_cast<uint32_t>(std::log2(static_cast<double>(n)));
    } else if (memoization_depth_lower == "sqrt_avg_deg") {
        max_mem_depth = static_cast<uint32_t>(std::sqrt(avg_deg));
    } else if (memoization_depth_lower == "log_n_w") { // log(n)*log(avg_deg)
        max_mem_depth = static_cast<uint32_t>(std::log2(static_cast<double>(n)) * std::log2(avg_deg));
    } else if (memoization_depth_lower == "none") {
        max_mem_depth = UINT32_MAX;
    } else {
        throw std::runtime_error("unrecognized memoization depth: " + memoization_depth_str);
    }

    if (params.verbose) {
        std::cout << "Memoization depth parameter: " << memoization_depth_str << std::endl;
    }
    if (memoization_depth_lower != "none" && params.verbose) {
        std::cout << "Computed memoization depth limit: " << max_mem_depth << std::endl;
        std::cout << "Average degree: " << avg_deg << std::endl;
    }

    // Reset ALL static state from previous runs to avoid state contamination (Not necessary)
    Sp_sampler::reset_mem();           // Clears memoization vectors
    Sp_sampler::reset_global_counters(); // Clears sampling statistics and counters
    Sp_sampler::set_max_mem_depth(max_mem_depth);
    Sp_sampler::set_enable_mem(!directed);

    // Pass false for alpha_given since we removed alpha parameter
    Probabilistic G(n, degrees.data(), directed ? in_degrees.data() : nullptr,
                    directed, 0.0, false, emp_peeling_a, true, "", seed);

    if (std::holds_alternative<const Eigen::MatrixXd*>(adj_input)) {
        const Eigen::MatrixXd* adj = std::get<const Eigen::MatrixXd*>(adj_input);
        add_edges_from_eigen(*adj, G);
    } else {
#ifdef MATLAB_BUILD
        const mxArray* mxA = std::get<const mxArray*>(adj_input);
        add_edges_from_mxA(mxA, G);
#else
        throw std::runtime_error("CcApproxBc mxArray input requires MATLAB_BUILD.");
#endif
    }

    double startTime = get_time_sec();
    G.run(k, delta, err, 0, 100);
    double elapsed = get_time_sec() - startTime;
    PrintTotalComputationTime(elapsed, params.verbose);

    // Prepare result vector
    Eigen::VectorXd bc_result(n);
    double scale_factor = static_cast<double>(n) * static_cast<double>(n - 1);
    if (!directed) {
        scale_factor /= 2.0;
    }

    for (uint32_t i = 0; i < n; i++) {
        bc_result[i] = G.get_centrality(i) * scale_factor;
    }

    // Create Result struct
    Result result;
    result.result = bc_result;
    result.total_time = elapsed;
    result.counters = COUNTERS();  // Initialize with default values (SILVAN doesn't expose detailed counters)
    result.counters.durationInSecTotal = elapsed;

    return result;
}
