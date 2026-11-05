#ifndef PARALLEL_CC_MATLAB_VCOMP_LIFETIME_HPP
#define PARALLEL_CC_MATLAB_VCOMP_LIFETIME_HPP

#include "mex.h"

#include <cstdint>

namespace parallel_cc::matlab {

// MATLAB R2023b can unload a VCOMP-backed MEX while VCOMP still retains
// worker-runtime state, causing a callback into the unloaded image. Lock the
// MEX once before its first request above one thread. MATLAB releases the
// module at process shutdown in a safe order. This changes module lifetime,
// not thread placement or algorithm behavior.
inline void ProtectVcompMexLifetime(int64_t requested_threads) {
#if defined(_WIN32)
  if (requested_threads <= 1) return;
  static const bool locked = []() {
    mexLock();
    return true;
  }();
  (void)locked;
#else
  (void)requested_threads;
#endif
}

}  // namespace parallel_cc::matlab

#endif  // PARALLEL_CC_MATLAB_VCOMP_LIFETIME_HPP
