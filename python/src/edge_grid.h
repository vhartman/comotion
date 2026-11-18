#pragma once
#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <comotion/collision/detail/VampPackingUtils.h>

namespace comotion_python {
// Shared finite-grid contract. begin/end index the binary traversal before
// endpoint removal; partial ranges always retain that order for resumption.
// The returned last index is the last submitted sample, not a safe prefix.
template <typename Check>
std::tuple<bool, int, int> check_edge_grid(
    const double *a, const double *b, int d, int n, int begin, int end,
    bool endpoints, const std::string &packing, Check check) {
  if (n < 2 || n > 10000000)
    throw std::invalid_argument("Edge sample count must be between 2 and 10000000");
  if (packing != "binary" && packing != "rake" && packing != "linear")
    throw std::invalid_argument("Unknown edge packing");
  if (begin < 0 || end < begin || end > n)
    throw std::invalid_argument("Invalid edge sample range");
  if (begin == end)
    return {true, -1, 0};
  std::vector<double> samples(16 * d);
  int rows = 0, last = -1, checked = 0;
  auto flush = [&]() {
    checked += rows;
    bool valid = check(samples.data(), rows);
    rows = 0;
    return valid;
  };
  auto append = [&](int index) {
    double t = double(index) / (n - 1);
    for (int j = 0; j < d; ++j) {
      volatile double offset = t * (b[j] - a[j]);
      samples[rows * d + j] = a[j] + offset;
    }
    last = index;
    return ++rows < 16 || flush();
  };
  if (packing == "binary" || begin != 0 || end != n) {
    std::vector<std::pair<int, int>> queue;
    queue.reserve(end + 1);
    queue.emplace_back(0, n - 1);
    for (int pos = 0; pos < end; ++pos) {
      auto [lo, hi] = queue[pos];
      int mid = (lo + hi) / 2;
      if (lo < mid) queue.emplace_back(lo, mid - 1);
      if (mid < hi) queue.emplace_back(mid + 1, hi);
      if (pos < begin || (!endpoints && (mid == 0 || mid == n - 1))) continue;
      if (!append(mid)) return {false, last, checked};
    }
  } else {
    auto order = packing == "rake" ? comotion::VampBatchPacking::Rake
                                   : comotion::VampBatchPacking::Linear;
    for (const auto &pack : comotion::detail::makeVampPackingLayouts(0, n, order)) {
      for (size_t lane = 0; lane < pack.lanes; ++lane) {
        int i = pack.timesteps[lane];
        if (!endpoints && (i == 0 || i == n - 1)) continue;
        if (!append(i)) return {false, last, checked};
      }
      if (rows && !flush()) return {false, last, checked};
    }
  }
  if (rows && !flush()) return {false, last, checked};
  return {true, last, checked};
}
} // namespace comotion_python
