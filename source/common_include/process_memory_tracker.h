#ifndef PARALLEL_CC_PROCESS_MEMORY_TRACKER_H_
#define PARALLEL_CC_PROCESS_MEMORY_TRACKER_H_

#include <cstdint>
#include <limits>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace parallel_cc::memory {

struct Snapshot {
  uint64_t working_set_bytes = 0;
  uint64_t private_bytes = 0;
  bool valid = false;
};

inline Snapshot ReadCurrentProcess() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS_EX counters{};
  counters.cb = sizeof(counters);
  Snapshot snapshot;
  if (K32GetProcessMemoryInfo(
          GetCurrentProcess(),
          reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
          sizeof(counters))) {
    snapshot.working_set_bytes =
        static_cast<uint64_t>(counters.WorkingSetSize);
    snapshot.private_bytes = static_cast<uint64_t>(counters.PrivateUsage);
    snapshot.valid = true;
  }
  return snapshot;
#else
  return Snapshot{};
#endif
}

struct PeakTracker {
  Snapshot baseline;
  Snapshot peak;
  bool started = false;

  void CaptureBaseline() {
    baseline = ReadCurrentProcess();
    peak = baseline;
    started = baseline.valid;
  }

  void UpdatePeak() {
    const Snapshot current = ReadCurrentProcess();
    if (!current.valid) return;
    if (!started) {
      baseline = current;
      peak = current;
      started = true;
      return;
    }
    if (current.working_set_bytes > peak.working_set_bytes) {
      peak.working_set_bytes = current.working_set_bytes;
    }
    if (current.private_bytes > peak.private_bytes) {
      peak.private_bytes = current.private_bytes;
    }
    peak.valid = true;
  }

  uint64_t DeltaPeakWorkingSetBytes() const {
    return peak.working_set_bytes >= baseline.working_set_bytes
        ? peak.working_set_bytes - baseline.working_set_bytes : 0;
  }

  uint64_t DeltaPeakPrivateBytes() const {
    return peak.private_bytes >= baseline.private_bytes
        ? peak.private_bytes - baseline.private_bytes : 0;
  }
};

inline int64_t ClampToInt64(uint64_t value) {
  const uint64_t maximum =
      static_cast<uint64_t>((std::numeric_limits<int64_t>::max)());
  return static_cast<int64_t>(value > maximum ? maximum : value);
}

template <typename Counters>
inline void FillCounters(Counters& output, const PeakTracker& tracker) {
  if (!tracker.started) return;
  output.process_baseline_working_set_bytes =
      ClampToInt64(tracker.baseline.working_set_bytes);
  output.process_peak_working_set_bytes =
      ClampToInt64(tracker.peak.working_set_bytes);
  output.process_delta_peak_working_set_bytes =
      ClampToInt64(tracker.DeltaPeakWorkingSetBytes());
  output.process_baseline_private_bytes =
      ClampToInt64(tracker.baseline.private_bytes);
  output.process_peak_private_bytes =
      ClampToInt64(tracker.peak.private_bytes);
  output.process_delta_peak_private_bytes =
      ClampToInt64(tracker.DeltaPeakPrivateBytes());
}

inline PeakTracker*& ActiveTracker() {
  static PeakTracker* tracker = nullptr;
  return tracker;
}

inline void UpdateActivePeak() {
  if (ActiveTracker() != nullptr) ActiveTracker()->UpdatePeak();
}

class ScopedPeakTracker {
 public:
  explicit ScopedPeakTracker(PeakTracker* tracker)
      : previous_(ActiveTracker()) {
    ActiveTracker() = tracker;
  }

  ScopedPeakTracker(const ScopedPeakTracker&) = delete;
  ScopedPeakTracker& operator=(const ScopedPeakTracker&) = delete;

  ~ScopedPeakTracker() { ActiveTracker() = previous_; }

 private:
  PeakTracker* previous_;
};

}  // namespace parallel_cc::memory

#endif  // PARALLEL_CC_PROCESS_MEMORY_TRACKER_H_
