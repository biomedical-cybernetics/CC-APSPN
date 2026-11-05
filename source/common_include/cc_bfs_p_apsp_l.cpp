#include "cc_bfs_p_apsp_l.hpp"

#include <omp.h>

#include <stdexcept>

#include "process_memory_tracker.h"

// Embed the reviewed kernel implementation but not its standalone MATLAB
// entry point. This avoids MEX/global-type collisions with the CC-APSPN
// boundary while leaving the source algorithm byte-for-byte unchanged.
#ifdef MATLAB_BUILD
#define PARALLEL_CC_RESTORE_MATLAB_BUILD
#undef MATLAB_BUILD
#endif
#include "../../third_party/cc_bfs_p/cpp/cc_bfs_p.cpp"
#ifdef PARALLEL_CC_RESTORE_MATLAB_BUILD
#define MATLAB_BUILD
#undef PARALLEL_CC_RESTORE_MATLAB_BUILD
#endif

namespace parallel_cc {

cc_bfs_p::Result CC_BFS_P_APSP_L_Compact_FromAdjacency(
    const std::vector<std::vector<int64_t>>& adjacency,
    const Eigen::VectorXd& degrees,
    int64_t num_threads) {
  if (omp_in_parallel()) {
    throw std::runtime_error(
        "Parallel CC-BFS distance stage must start outside an OpenMP team.");
  }

  namespace DistanceCore = parallel_cc::cc_bfs_p;
  Timer diameter_timer;
  diameter_timer.Start();
  const int64_t diameter = adjacency.size() <= 1
      ? 0
      : DistanceCore::DiamUpperBound(adjacency, degrees);
  diameter_timer.Stop();
  DistanceCore::Result result = DistanceCore::Run(
      adjacency, degrees, "bounded-avg-deg", num_threads, false, false,
      false, 0.0, diameter, diameter_timer.Seconds());

  // The compact matrix is already owning and source-column oriented.
  parallel_cc::memory::UpdateActivePeak();
  return result;
}

Eigen::MatrixXd CC_BFS_P_APSP_L_FromAdjacency(
    const std::vector<std::vector<int64_t>>& adjacency,
    const Eigen::VectorXd& degrees,
    int64_t num_threads) {
  cc_bfs_p::Result result = CC_BFS_P_APSP_L_Compact_FromAdjacency(
      adjacency, degrees, num_threads);

  // The reviewed core stores one source's distances in each matrix column.
  // CC-APSPN consumes the same orientation, so conversion only widens the
  // compact scalar type to double; it must not transpose the matrix.
  Eigen::MatrixXd distances = ::ResultMatrixToDouble(result.distances);
  // Sample while the compact CC-BFS result and converted double matrix both
  // exist; sampling only after return would miss this conversion peak.
  parallel_cc::memory::UpdateActivePeak();
  return distances;
}

}  // namespace parallel_cc
