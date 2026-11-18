#include "primitive_scene.h"
#include "edge_grid.h"
#include "edge_batch.h"

namespace comotion_python::primitive {
std::tuple<bool, int, int>
Scene::edge(Array from, Array to, std::shared_ptr<Mode> m, int n, int begin,
            int end, bool endpoints, double tolerance,
            const std::string &packing, bool early_exit) {
  check(m);
  int d = joints.size();
  if (from.ndim() != 1 || to.ndim() != 1 || from.size() != d || to.size() != d)
    throw std::invalid_argument("Invalid edge dimensions");
  if (!std::isfinite(tolerance) || tolerance < 0)
    throw std::invalid_argument("Invalid collision tolerance");
  // Retain the low-level primitive API's historical slice normalization.
  begin = std::clamp(begin < 0 ? n + begin : begin, 0, std::max(n, 0));
  end = std::clamp(end < 0 ? n + end : end, 0, std::max(n, 0));
  end = std::max(begin, end);
  return check_edge_grid(from.data(), to.data(), d, n, begin, end, endpoints,
                         packing, [&](const double *samples, int rows) {
    Array chunk({rows, d}, samples, py::none());
    auto valid = batch_impl(chunk, m, {}, true, tolerance, py::none(), early_exit);
    return std::all_of(valid.data(), valid.data() + rows, [](bool v) { return v; });
  });
}

py::array_t<bool> Scene::batch(Array qs, std::shared_ptr<Mode> m,
                               std::vector<int> group, bool simd,
                               double tolerance,
                               py::object diagnostics) {
  return batch_impl(qs, m, group, simd, tolerance, diagnostics, false);
}

py::array_t<bool> Scene::batch_impl(Array qs, std::shared_ptr<Mode> m,
                                    const std::vector<int> &group, bool simd,
                                    double tolerance, py::object diagnostics,
                                    bool stop_on_collision) {
  if (!std::isfinite(tolerance) || tolerance < 0)
    throw std::invalid_argument("Invalid collision tolerance");
  check(m);
  if (qs.ndim() != 2 || qs.shape(1) != int(joints.size()))
    throw std::invalid_argument("Expected (N, dof) configurations");
  int n = qs.shape(0), d = joints.size();
  py::array_t<bool> out(n);
  bool *result = out.mutable_data();
  // Edge-only queries need one rejection, not a complete validity mask.
  if (stop_on_collision)
    std::fill(result, result + n, true);
  const double *data = qs.data();
  bool profiling = !diagnostics.is_none();
  using Clock = std::chrono::steady_clock;
  auto elapsed = [](auto start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
  };
  double fk_seconds = 0, pair_seconds = 0, fallback_seconds = 0;
  size_t pair_count = 0, fallback_count = 0;
  if (!m->prepared)
    prepare(*m);
  py::gil_scoped_release release;
  if (simd && n == 1 && !profiling) {
    result[0] = scalar_valid(*m, data, group, tolerance);
    return out;
  }
  if (!simd) {
    for (int i = 0; i < n; ++i)
      result[i] = double_valid(*m, data + i * d, group, tolerance);
    return out;
  }
  for (int start = 0; start < n; start += W) {
    int rows = std::min(W, n - start);
    std::array<float, W> flags{};
    std::array<std::vector<std::pair<int, int>>, W> pending;
    for (int k = 0; k < rows; ++k)
      flags[k] = valid_q(data + (start + k) * d) &&
                 (m->static_penetration <= tolerance || !group.empty());
    if (stop_on_collision)
      for (int k = 0; k < rows; ++k)
        if (!flags[k]) {
          result[0] = false;
          return out;
        }
    V alive = V(flags) > V(0);
    auto clock_start = profiling ? Clock::now() : Clock::time_point{};
    auto g = geometry<V>(*m, [&](int j) {
      std::array<float, W> v;
      for (int k = 0; k < W; ++k)
        v[k] = float(data[(start + (k < rows ? k : 0)) * d + j]);
      return V(v);
    });
    if (profiling) {
      fk_seconds += elapsed(clock_start);
      clock_start = Clock::now();
    }
    auto centers = bound_centers(*m, g);
    bool any_valid = true;
    for (const auto &block : m->topology->blocks) {
      const auto &A = centers[block.a], &B = centers[block.b];
      V x = A[0] - B[0], y = A[1] - B[1], z = A[2] - B[2];
      // Inflate for float FK/transform rounding; close bounds are never used
      // to decide collision and retain the double-precision narrow phase.
      V radius = V(m->bounds[block.a].radius + m->bounds[block.b].radius);
      V margin = V(1e-4) * (V(1) + radius + absolute(A[0]) + absolute(A[1]) +
                            absolute(A[2]) + absolute(B[0]) + absolute(B[1]) +
                            absolute(B[2]));
      V r = radius + margin;
      if (((x * x + y * y + z * z - r * r <= V(0)) & alive).none())
        continue;
      for (auto [a, b] : block.pairs) {
        if (!relevant(*m, a, b, group))
          continue;
        if (profiling)
          ++pair_count;
        V gap = clearance<V>(a, b, g);
        // Capsule pairs have an exact clearance, so lanes outside a float
        // rounding band (bounded like the block bounds above) are decided
        // here; only the band goes to the double-precision check.
        bool capsules = kinds[a] == 1 && kinds[b] == 1;
        V band = V(1e-5f);
        if (capsules) {
          const auto &PA = g[a].m, &PB = g[b].m;
          band = V(1e-4) * (V(1) + V(sizes[a][1] + sizes[b][1]) +
                            absolute(PA[3]) + absolute(PA[7]) +
                            absolute(PA[11]) + absolute(PB[3]) +
                            absolute(PB[7]) + absolute(PB[11]));
        }
        double threshold = -tolerance;
        if (kinds[a] == 0 || kinds[b] == 0) {
          int sphere = kinds[a] == 0 ? a : b, other = kinds[a] == 0 ? b : a;
          double radius =
              sizes[sphere][0] +
              ((kinds[other] == 0 || kinds[other] == 1) ? sizes[other][0] : 0.);
          double t = std::min(tolerance, radius);
          threshold = -t * (2 * radius - t);
        }
        if (!convex_pair(a, b) || capsules) {
          // Round the rejection threshold down; boundary lanes go to the
          // double-precision check instead of relying on a float conversion.
          float reject = std::nextafter(
              float(threshold - 1e-5), -std::numeric_limits<float>::infinity());
          V colliding = (capsules ? (gap < V(float(-tolerance)) - band)
                                  : (gap < V(reject))) &
                        alive;
          if (stop_on_collision && colliding.any()) {
            result[0] = false;
            return out;
          }
          alive = alive & ~colliding;
        }
        if (alive.none()) {
          any_valid = false;
          break;
        }
        V uncertain = (gap <= band) & alive;
        if (uncertain.any()) {
          uncertain.to_array_unaligned(flags.data());
          for (int k = 0; k < rows; ++k)
            if (flags[k] != 0.f)
              pending[k].emplace_back(a, b);
        }
      }
      if (!any_valid)
        break;
    }
    if (profiling) {
      pair_seconds += elapsed(clock_start);
      clock_start = Clock::now();
    }
    alive.to_array_unaligned(flags.data());
    for (int k = 0; k < rows; ++k) {
      bool valid = flags[k] != 0.f;
      if (valid && !pending[k].empty()) {
        if (profiling)
          ++fallback_count;
        auto exact = geometry<double>(
            *m, [&](int j) { return data[(start + k) * d + j]; });
        double depth = group.empty() ? m->static_penetration : 0.;
        // Preserve the original summation order for positive tolerances.
        std::sort(pending[k].begin(), pending[k].end());
        for (auto [a, b] : pending[k]) {
          if (tolerance == 0) {
            if (!pair_valid(a, b, exact)) {
              valid = false;
              break;
            }
          } else {
            depth += pair_depth(a, b, exact);
            if (depth > tolerance) {
              valid = false;
              break;
            }
          }
        }
      }
      result[start + k] = valid;
      if (stop_on_collision && !valid)
        return out;
    }
    if (profiling)
      fallback_seconds += elapsed(clock_start);
  }
  if (profiling) {
    py::gil_scoped_acquire acquire;
    diagnostics["fk_seconds"] = fk_seconds;
    diagnostics["pair_seconds"] = pair_seconds;
    diagnostics["fallback_seconds"] = fallback_seconds;
    diagnostics["pair_pack_checks"] = pair_count;
    diagnostics["fallback_rows"] = fallback_count;
    diagnostics["moving_pairs"] = m->topology->moving_pairs.size();
    diagnostics["pair_blocks"] = m->topology->blocks.size();
    diagnostics["shapes"] = shapes.size();
    diagnostics["joints"] = joints.size();
  }
  return out;
}
} // namespace comotion_python::primitive
PYBIND11_MODULE(_primitive, m) {
  using namespace comotion_python::primitive;
  m.attr("simd_width") = W;
  m.attr("uses_fcl") = false;
  py::class_<Mode, std::shared_ptr<Mode>>(m, "ModeHandle");
  py::class_<Scene>(m, "Scene")
      .def(py::init<py::dict>())
      .def("initial", &Scene::initial, py::keep_alive<0, 1>())
      .def_property_readonly("prepared_modes", &Scene::prepared_count)
      .def("transition", &Scene::transition, py::keep_alive<0, 1>())
      .def("poses", &Scene::poses)
      .def("info", &Scene::info)
      .def("state_valid", [](Scene &scene, Array q) {
        return scene.state(q, scene.initial(), {}, 0.);
      }, py::arg("q"))
      .def("state_valid", &Scene::state, py::arg("q"), py::arg("mode"),
           py::arg("group") = std::vector<int>{}, py::arg("tolerance") = 0.)
      .def("batch_edge_valid", [](Scene &scene, Array from, Array to,
                                    std::shared_ptr<Mode> mode, py::array_t<int, py::array::c_style> counts,
                                    bool endpoints, double tolerance, const std::string &packing) {
        return comotion_python::check_edge_batch(from, to, counts,
          [&](Array a, Array b, int n) {
            return scene.edge(a, b, mode, n, 0, n, endpoints, tolerance, packing, true);
          });
      }, py::arg("from_q"), py::arg("to_q"), py::arg("mode"), py::arg("counts"),
         py::arg("include_endpoints") = false, py::arg("tolerance") = 0.,
         py::arg("packing") = "binary")
      .def("edge_valid", &Scene::edge, py::arg("from_q"), py::arg("to_q"),
           py::arg("mode"), py::arg("n"), py::arg("begin"), py::arg("end"),
           py::arg("include_endpoints"), py::arg("tolerance") = 0.,
           py::arg("packing") = "binary", py::arg("early_exit") = true)
      .def("batch_valid", &Scene::batch, py::arg("qs"), py::arg("mode"),
           py::arg("group") = std::vector<int>{}, py::arg("simd") = true,
           py::arg("tolerance") = 0., py::arg("diagnostics") = py::none());
}
