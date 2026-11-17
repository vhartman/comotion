#pragma once
// Optional Panda fast path. Keep upstream self/environment predicates intact;
// reuse VAMP FK across inter-robot pairs, and pack independent state proposals.
// No worker threads, changed collision geometry, or relaxed collision tests.
#include <chrono>
#include <comotion/collision/CollisionChecker.h>
#include <comotion/collision/detail/VampPackingUtils.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vamp/collision/sphere_sphere.hh>
#include <vamp/robots/panda.hh>
#include <vamp/robots/ur5.hh>

namespace comotion_python {
constexpr std::size_t width = vamp::FloatVectorWidth;
constexpr std::size_t max_spheres =
    std::max(vamp::robots::Panda::n_spheres, vamp::robots::UR5::n_spheres);
struct Geometry {
  std::array<vamp::FloatVector<width>, max_spheres> x, y, z, r;
  std::size_t count = 0;
};
using Paths = std::vector<comotion::Path>;
using Robots = std::vector<const comotion::RobotModel *>;

template <typename Robot>
inline Geometry robot_geometry(const comotion::RobotModel &model,
                               const comotion::Path &path,
                               const std::vector<std::size_t> &rows) {
  typename Robot::template ConfigurationBlock<width> block;
  for (std::size_t d = 0; d < Robot::dimension; ++d) {
    std::array<float, width> values{};
    for (std::size_t lane = 0; lane < width; ++lane)
      values[lane] =
          static_cast<float>(path[rows[lane < rows.size() ? lane : 0]][d]);
    block[d] = vamp::FloatVector<width>(values);
  }
  typename Robot::template Spheres<width> packed;
  Robot::template sphere_fk<width>(block, packed);
  Geometry out;
  out.count = Robot::n_spheres;
  for (std::size_t k = 0; k < Robot::n_spheres; ++k) {
    out.x[k] = packed.x[k];
    out.y[k] = packed.y[k];
    out.z[k] = packed.z[k];
    out.r[k] = packed.r[k];
  }
  const auto transform = model.getBaseTransform().cast<float>();
  const auto r = transform.linear();
  const auto t = transform.translation();
  for (std::size_t k = 0; k < Robot::n_spheres; ++k) {
    const auto x = out.x[k], y = out.y[k], z = out.z[k];
    out.x[k] = r(0, 0) * x + r(0, 1) * y + r(0, 2) * z + t[0];
    out.y[k] = r(1, 0) * x + r(1, 1) * y + r(1, 2) * z + t[1];
    out.z[k] = r(2, 0) * x + r(2, 1) * y + r(2, 2) * z + t[2];
  }
  return out;
}

inline Geometry geometry(const comotion::RobotModel &model,
                         const comotion::Path &path,
                         const std::vector<std::size_t> &rows) {
  using Family = comotion::RobotModel::RobotFamily;
  if (model.robotFamily() == Family::Panda)
    return robot_geometry<vamp::robots::Panda>(model, path, rows);
  if (model.robotFamily() == Family::UR5)
    return robot_geometry<vamp::robots::UR5>(model, path, rows);
  throw std::invalid_argument("Optimized FK requires a Panda or UR5 model");
}

inline bool pairs(const Robots &robots, const Paths &paths,
                  const std::vector<std::size_t> &rows,
                  std::vector<unsigned char> &valid, bool require_all = false) {
  std::vector<Geometry> cache(robots.size());
  std::vector<bool> ready(robots.size(), false);
  const auto get = [&](std::size_t i) -> const Geometry & {
    if (!ready[i]) {
      cache[i] = geometry(*robots[i], paths[i], rows);
      ready[i] = true;
    }
    return cache[i];
  };
  for (std::size_t i = 0; i < robots.size(); ++i)
    for (std::size_t j = i + 1; j < robots.size(); ++j) {
      const auto &a = get(i);
      const auto &b = get(j);
      std::array<bool, width> colliding{};
      for (std::size_t u = 0; u < a.count; ++u)
        for (std::size_t v = 0; v < b.count; ++v) {
          auto distances = vamp::collision::sphere_sphere_sql2(
              a.x[u], a.y[u], a.z[u], a.r[u], b.x[v], b.y[v], b.z[v], b.r[v]);
          std::array<float, width> values{};
          distances.to_array_unaligned(values.data());
          for (std::size_t lane = 0; lane < rows.size(); ++lane)
            if (values[lane] < 0.f)
              colliding[lane] = true;
        }
      bool any = false;
      for (std::size_t lane = 0; lane < rows.size(); ++lane) {
        if (require_all && colliding[lane])
          return false;
        if (colliding[lane])
          valid[rows[lane]] = false;
        any = any || valid[rows[lane]];
      }
      if (!any)
        return false;
    }
  return true;
}

// SIMD self/environment predicate is aggregate. Recursively split only failed
// packs to recover an exact per-state mask; never discard valid neighbors of
// an invalid SIMD lane. Survivors are compacted before pair checking.
inline std::vector<unsigned char>
states(const comotion::CollisionChecker &checker, const Robots &robots,
       const Paths &paths) {
  const std::size_t count = paths.empty() ? 0 : paths.front().size();
  std::vector<unsigned char> valid(count, true);
  for (std::size_t r = 0; r < robots.size(); ++r) {
    std::vector<std::size_t> rows;
    for (std::size_t k = 0; k < count; ++k)
      if (valid[k])
        rows.push_back(k);
    std::function<void(std::size_t, std::size_t)> check = [&](std::size_t begin,
                                                              std::size_t end) {
      comotion::Path pack;
      for (std::size_t k = begin; k < end; ++k)
        pack.push_back(paths[r][rows[k]]);
      if (checker.isRobotPathValid(*robots[r], pack))
        return;
      if (end - begin == 1) {
        valid[rows[begin]] = false;
        return;
      }
      auto mid = begin + (end - begin) / 2;
      check(begin, mid);
      check(mid, end);
    };
    for (std::size_t begin = 0; begin < rows.size(); begin += width)
      check(begin, std::min(begin + width, rows.size()));
  }
  std::vector<std::size_t> rows;
  for (std::size_t k = 0; k < count; ++k)
    if (valid[k])
      rows.push_back(k);
  for (std::size_t begin = 0; begin < rows.size(); begin += width)
    pairs(robots, paths,
          {rows.begin() + begin,
           rows.begin() + std::min(begin + width, rows.size())},
          valid);
  return valid;
}

inline bool motion(const comotion::CollisionChecker &checker,
                   const Robots &robots,
                   const std::vector<std::vector<double>> &from,
                   const std::vector<std::vector<double>> &to, int steps) {
  // Identical samples and rake ordering to upstream. Build only one SIMD
  // pack at a time, instead of materializing every interpolation sample.
  const auto packs = comotion::detail::makeVampPackingLayouts(
      0, steps + 1, comotion::VampBatchPacking::Rake);
  for (const auto &pack : packs) {
    Paths paths(robots.size());
    for (std::size_t r = 0; r < robots.size(); ++r) {
      for (std::size_t lane = 0; lane < pack.lanes; ++lane) {
        double t = double(pack.timesteps[lane]) / steps;
        std::vector<double> q(from[r].size());
        for (std::size_t d = 0; d < from[r].size(); ++d)
          q[d] = from[r][d] + t * (to[r][d] - from[r][d]);
        paths[r].push_back(std::move(q));
      }
      if (!checker.isRobotPathValid(*robots[r], paths[r]))
        return false;
    }
    std::vector<std::size_t> rows(pack.lanes);
    std::iota(rows.begin(), rows.end(), 0);
    std::vector<unsigned char> valid(pack.lanes, true);
    if (!pairs(robots, paths, rows, valid, true))
      return false;
  }
  return true;
}

// Diagnostic counters are compiled out of the ordinary query kernels. Timed
// replay is separate from profiled replay: per-pack clocks perturb tiny
// queries.
struct EdgeStats {
  double preparation_seconds = 0, self_environment_seconds = 0, fk_seconds = 0,
         pair_seconds = 0;
  std::uint64_t edges = 0, planned_samples = 0, packs = 0, samples = 0,
                self_calls = 0;
  std::uint64_t fk_calls = 0, robot_pairs = 0, sphere_pairs = 0;
  std::uint64_t self_rejections = 0, pair_rejections = 0;
};
using Clock = std::chrono::steady_clock;
template <bool Profile> inline Clock::time_point stamp() {
  if constexpr (Profile)
    return Clock::now();
  return {};
}
template <bool Profile>
inline void charge(double &field, Clock::time_point begin) {
  if constexpr (Profile)
    field += std::chrono::duration<double>(Clock::now() - begin).count();
}

struct EdgeWorkspace {
  Paths paths;
  std::vector<Geometry> cache;
  std::vector<unsigned char> ready;
  std::vector<std::size_t> rows;
  explicit EdgeWorkspace(std::size_t n) : paths(n), ready(n), rows(width) {
    std::iota(rows.begin(), rows.end(), 0);
  }
};

// Kernel 0: scalar lane extraction; 1: SIMD masks; 2: masks + early exit;
// 3: same decisions as 2, plus edge-local reusable storage/lazy rake layouts.
// Only edge queries use this route. State batches still need a per-state mask.
template <int Kernel, bool Profile>
inline bool edge_pairs(const Robots &robots, EdgeWorkspace &work,
                       EdgeStats &stats) {
  auto setup = stamp<Profile>();
  if (work.cache.size() != robots.size())
    work.cache.resize(robots.size());
  std::fill(work.ready.begin(), work.ready.end(), false);
  charge<Profile>(stats.preparation_seconds, setup);
  for (std::size_t i = 0; i < robots.size(); ++i)
    for (std::size_t j = i + 1; j < robots.size(); ++j) {
      for (auto r : {i, j})
        if (!work.ready[r]) {
          auto begin = stamp<Profile>();
          work.cache[r] = geometry(*robots[r], work.paths[r], work.rows);
          charge<Profile>(stats.fk_seconds, begin);
          work.ready[r] = true;
          if constexpr (Profile)
            ++stats.fk_calls;
        }
      auto begin = stamp<Profile>();
      if constexpr (Profile)
        ++stats.robot_pairs;
      const auto &a = work.cache[i];
      const auto &b = work.cache[j];
      auto mask = vamp::FloatVector<width>::zero_vector();
      std::array<bool, width> colliding{};
      for (std::size_t u = 0; u < a.count; ++u) {
        for (std::size_t v = 0; v < b.count; ++v) {
          auto distances = vamp::collision::sphere_sphere_sql2(
              a.x[u], a.y[u], a.z[u], a.r[u], b.x[v], b.y[v], b.z[v], b.r[v]);
          if constexpr (Kernel == 0) {
            std::array<float, width> values{};
            distances.to_array_unaligned(values.data());
            for (std::size_t lane = 0; lane < work.rows.size(); ++lane)
              if (values[lane] < 0.f)
                colliding[lane] = true;
          } else {
            mask = mask | (distances < 0.f);
          }
        }
        if constexpr (Profile)
          stats.sphere_pairs += b.count;
        // Test once per sphere row rather than reducing after every sphere
        // pair. Invalid padding cannot appear: geometry duplicates lane 0.
        if constexpr (Kernel >= 2)
          if (mask.any()) {
            charge<Profile>(stats.pair_seconds, begin);
            if constexpr (Profile)
              ++stats.pair_rejections;
            return false;
          }
      }
      bool hit;
      if constexpr (Kernel == 0)
        hit = std::any_of(colliding.begin(), colliding.end(),
                          [](bool v) { return v; });
      else
        hit = mask.any();
      charge<Profile>(stats.pair_seconds, begin);
      if (hit) {
        if constexpr (Profile)
          ++stats.pair_rejections;
        return false;
      }
    }
  return true;
}

template <int Kernel, bool Profile>
inline bool edge_kernel(const comotion::CollisionChecker &checker,
                        const Robots &robots,
                        const std::vector<std::vector<double>> &from,
                        const std::vector<std::vector<double>> &to, int steps,
                        EdgeStats &stats) {
  auto begin = stamp<Profile>();
  if constexpr (Profile) {
    ++stats.edges;
    stats.planned_samples += steps + 1;
  }
  std::vector<comotion::detail::VampPackingLayout> layouts;
  std::unique_ptr<EdgeWorkspace> workspace;
  const std::size_t stride = (std::size_t(steps) + 1 + width - 1) / width;
  if constexpr (Kernel == 3) {
    workspace = std::make_unique<EdgeWorkspace>(robots.size());
    for (std::size_t r = 0; r < robots.size(); ++r)
      workspace->paths[r].resize(width, std::vector<double>(from[r].size()));
  } else
    layouts = comotion::detail::makeVampPackingLayouts(
        0, steps + 1, comotion::VampBatchPacking::Rake);
  charge<Profile>(stats.preparation_seconds, begin);
  for (std::size_t offset = 0; offset < stride; ++offset) {
    begin = stamp<Profile>();
    comotion::detail::VampPackingLayout pack;
    if constexpr (Kernel == 3) {
      for (std::size_t lane = 0; lane < width; ++lane) {
        const auto timestep = offset + lane * stride;
        if (timestep > std::size_t(steps))
          break;
        pack.timesteps[pack.lanes++] = timestep;
      }
    } else
      pack = layouts[offset];
    if constexpr (Kernel != 3)
      workspace = std::make_unique<EdgeWorkspace>(robots.size());
    auto &work = *workspace;
    work.rows.resize(pack.lanes);
    if constexpr (Profile) {
      ++stats.packs;
      stats.samples += pack.lanes;
    }
    charge<Profile>(stats.preparation_seconds, begin);
    for (std::size_t r = 0; r < robots.size(); ++r) {
      begin = stamp<Profile>();
      if constexpr (Kernel == 3)
        work.paths[r].resize(pack.lanes);
      for (std::size_t lane = 0; lane < pack.lanes; ++lane) {
        double t = double(pack.timesteps[lane]) / steps;
        if constexpr (Kernel == 3) {
          auto &q = work.paths[r][lane];
          for (std::size_t d = 0; d < from[r].size(); ++d)
            q[d] = from[r][d] + t * (to[r][d] - from[r][d]);
        } else {
          std::vector<double> q(from[r].size());
          for (std::size_t d = 0; d < from[r].size(); ++d)
            q[d] = from[r][d] + t * (to[r][d] - from[r][d]);
          work.paths[r].push_back(std::move(q));
        }
      }
      charge<Profile>(stats.preparation_seconds, begin);
      begin = stamp<Profile>();
      bool valid = checker.isRobotPathValid(*robots[r], work.paths[r]);
      charge<Profile>(stats.self_environment_seconds, begin);
      if constexpr (Profile)
        ++stats.self_calls;
      if (!valid) {
        if constexpr (Profile)
          ++stats.self_rejections;
        return false;
      }
    }
    if (!edge_pairs<Kernel, Profile>(robots, work, stats))
      return false;
  }
  return true;
}

template <bool Profile>
inline bool edge_dispatch(int kernel, const comotion::CollisionChecker &checker,
                          const Robots &robots,
                          const std::vector<std::vector<double>> &from,
                          const std::vector<std::vector<double>> &to, int steps,
                          EdgeStats &stats) {
  switch (kernel) {
  case 0:
    if constexpr (!Profile)
      return motion(checker, robots, from, to, steps);
    else
      return edge_kernel<0, true>(checker, robots, from, to, steps, stats);
  case 1:
    return edge_kernel<1, Profile>(checker, robots, from, to, steps, stats);
  case 2:
    return edge_kernel<2, Profile>(checker, robots, from, to, steps, stats);
  case 3:
    return edge_kernel<3, Profile>(checker, robots, from, to, steps, stats);
  default:
    throw std::invalid_argument("Unknown edge kernel");
  }
}
} // namespace comotion_python
