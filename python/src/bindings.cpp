// Optional collision-only bridge. No planner, sampler or cost implementation.
#include "cached_vamp.h"
#include "edge_grid.h"
#include "edge_batch.h"
#include "runtime_spheres.h"
#include <algorithm>
#include <cmath>
#include <comotion/collision/CollisionChecker.h>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Configs = std::vector<std::vector<double>>;
#include "primitive_overlay.h"

namespace comotion_python {
class Scene {
  std::vector<std::unique_ptr<comotion::RobotModel>> models_;
  std::vector<const comotion::RobotModel *> robots_;
  std::vector<std::size_t> offsets_{0};
  comotion::CollisionChecker checker_;
  PrimitiveOverlay extras_;
  bool vamp_ = false;
  bool runtime_ = false;
  bool packed_world_ = false;
  bool packed_world_active() const { return packed_world_ && runtime_; }
  bool overlay_valid(const double *q) const {
    if (packed_world_active() && extras_.world_only())
      return true;
    return extras_.valid(robots_, unpack(q), nullptr, true,
                         packed_world_active());
  }
  std::vector<comotion_python::RuntimeRobot> runtime_robots_;
  std::vector<comotion::ObstacleSphere> runtime_spheres_;
  std::vector<comotion::ObstacleCylinder> runtime_cylinders_;
  std::array<bool, vamp::FloatVectorWidth>
  runtime_pack(const double *q, std::size_t stride, std::size_t rows) const {
    std::array<bool, vamp::FloatVectorWidth> uncertain{};
    auto extra = [&](std::size_t r, const auto &geometry, auto reject) {
      if (!packed_world_active())
        return true;
      auto world_reject = [&](const comotion_python::Packed &distance) {
        std::array<float, vamp::FloatVectorWidth> d;
        distance.to_array_unaligned(d.data());
        // Near-contact world decisions use the original double-precision
        // checker. Defer uncertain lanes without skipping robot checks.
        for (std::size_t k = 0; k < rows; ++k)
          if (std::abs(d[k]) < 1.e-5f) {
            uncertain[k] = true;
            d[k] = 0.f;
          }
        return reject(comotion_python::Packed(d));
      };
      return extras_.packed_world(runtime_robots_[r], geometry, world_reject);
    };
    auto valid = comotion_python::runtime_valid(
        runtime_robots_, offsets_, runtime_spheres_, runtime_cylinders_, q,
        stride, rows, extra);
    if (packed_world_active())
      for (std::size_t k = 0; k < rows; ++k)
        if (valid[k] && uncertain[k])
          valid[k] = extras_.valid(robots_, unpack(q + k * stride));
    return valid;
  }
  bool optimized_ = false;
  int edge_kernel_ =
      3; // buffered SIMD edge kernel; reference remains selectable

  static comotion::CollisionChecker::Backend backend(const std::string &name) {
    if (name == "vamp")
      return comotion::CollisionChecker::Backend::Vamp;
    if (name == "spheres" || name == "runtime")
      return comotion::CollisionChecker::Backend::Spheres;
    if (name == "fcl")
      return comotion::CollisionChecker::Backend::Fcl;
    throw std::invalid_argument("backend must be vamp, spheres or fcl");
  }
  Configs unpack(const double *q) const {
    Configs result;
    result.reserve(robots_.size());
    for (std::size_t i = 0; i < robots_.size(); ++i)
      result.emplace_back(q + offsets_[i], q + offsets_[i + 1]);
    return result;
  }
  void shape(const Array &q) const {
    if (q.ndim() != 1 || std::size_t(q.size()) != offsets_.back())
      throw std::invalid_argument("Expected a flat full-system configuration");
  }
  bool in_bounds(const double *q) const {
    for (std::size_t i = 0; i < robots_.size(); ++i)
      for (int j = 0; j < robots_[i]->numJoints(); ++j) {
        double x = q[offsets_[i] + j];
        if (!std::isfinite(x) || x < robots_[i]->jointLower(j) ||
            x > robots_[i]->jointUpper(j))
          return false;
      }
    return true;
  }

public:
  Scene(const std::string &urdf, const std::string &srdf, const Array &bases,
        const Array &spheres, const Array &cylinders, const std::string &kind)
      : checker_(backend(kind)), vamp_(kind == "vamp"),
        runtime_(kind == "runtime") {
    if (bases.ndim() != 3 || bases.shape(1) != 4 || bases.shape(2) != 4 ||
        bases.shape(0) < 1)
      throw std::invalid_argument("bases must have shape (robots,4,4)");
    auto b = bases.unchecked<3>();
    for (py::ssize_t i = 0; i < bases.shape(0); ++i) {
      auto model = std::make_unique<comotion::RobotModel>();
      model->loadURDF(urdf);
      model->loadSRDF(srdf);
      Eigen::Affine3d tf = Eigen::Affine3d::Identity();
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
          if (!std::isfinite(b(i, r, c)))
            throw std::invalid_argument("Nonfinite base transform");
          tf.matrix()(r, c) = b(i, r, c);
        }
      if (!tf.linear().transpose().isApprox(tf.linear().inverse(), 1e-8) ||
          std::abs(tf.linear().determinant() - 1) > 1e-8 ||
          !tf.matrix().row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-8))
        throw std::invalid_argument("Base transform must be rigid");
      model->setBaseTransform(tf);
      offsets_.push_back(offsets_.back() + model->numJoints());
      robots_.push_back(model.get());
      models_.push_back(std::move(model));
      if (runtime_)
        runtime_robots_.emplace_back(*models_.back());
    }
    if (spheres.ndim() != 2 || spheres.shape(1) != 4 || cylinders.ndim() != 2 ||
        cylinders.shape(1) != 8)
      throw std::invalid_argument("Expected spheres (n,4), cylinders (n,8)");
    auto s = spheres.unchecked<2>();
    std::vector<comotion::ObstacleSphere> ss;
    for (py::ssize_t i = 0; i < spheres.shape(0); ++i) {
      for (int j = 0; j < 4; ++j)
        if (!std::isfinite(s(i, j)))
          throw std::invalid_argument("Nonfinite sphere");
      if (s(i, 3) <= 0)
        throw std::invalid_argument("Nonpositive sphere radius");
      ss.push_back({{s(i, 0), s(i, 1), s(i, 2)}, s(i, 3)});
    }
    checker_.setObstacles(ss);
    extras_.spheres(ss);
    runtime_spheres_ = ss;
    auto c = cylinders.unchecked<2>();
    std::vector<comotion::ObstacleCylinder> cs;
    for (py::ssize_t i = 0; i < cylinders.shape(0); ++i) {
      for (int j = 0; j < 8; ++j)
        if (!std::isfinite(c(i, j)))
          throw std::invalid_argument("Nonfinite cylinder");
      Eigen::Vector3d axis(c(i, 3), c(i, 4), c(i, 5));
      if (axis.norm() < 1e-12 || c(i, 6) <= 0 || c(i, 7) <= 0)
        throw std::invalid_argument("Invalid cylinder");
      cs.push_back(
          {{c(i, 0), c(i, 1), c(i, 2)}, axis.normalized(), c(i, 6), c(i, 7)});
    }
    checker_.setCylinderObstacles(cs);
    extras_.cylinders(cs);
    runtime_cylinders_ = cs;
  }
  py::array_t<double> limits() const {
    py::array_t<double> out({py::ssize_t(2), py::ssize_t(offsets_.back())});
    auto a = out.mutable_unchecked<2>();
    for (std::size_t i = 0; i < robots_.size(); ++i)
      for (int j = 0; j < robots_[i]->numJoints(); ++j) {
        a(0, offsets_[i] + j) = robots_[i]->jointLower(j);
        a(1, offsets_[i] + j) = robots_[i]->jointUpper(j);
      }
    return out;
  }
  py::array_t<double> spheres(const Array &q) const {
    shape(q);
    auto configs = unpack(q.data());
    std::vector<comotion::CollisionSphere> all;
    for (std::size_t i = 0; i < robots_.size(); ++i) {
      auto block = robots_[i]->getCollisionSpheres(configs[i]);
      all.insert(all.end(), block.begin(), block.end());
    }
    py::array_t<double> out({py::ssize_t(all.size()), py::ssize_t(4)});
    auto a = out.mutable_unchecked<2>();
    for (std::size_t i = 0; i < all.size(); ++i) {
      for (int j = 0; j < 3; ++j)
        a(i, j) = all[i].center[j];
      a(i, 3) = all[i].radius;
    }
    return out;
  }
  Array link_poses(const Array &q) const {
    shape(q);
    if (!in_bounds(q.data()))
      throw std::invalid_argument("Configuration outside limits");
    auto cfg = unpack(q.data());
    Array out({py::ssize_t(robots_.size()),
               py::ssize_t(robots_[0]->links().size()), py::ssize_t(7)});
    auto a = out.mutable_unchecked<3>();
    for (std::size_t r = 0; r < robots_.size(); ++r) {
      auto poses = robots_[r]->getLinkTransforms(cfg[r]);
      for (std::size_t j = 0; j < poses.size(); ++j) {
        Eigen::Quaterniond rotation(poses[j].linear());
        for (int k = 0; k < 3; ++k)
          a(r, j, k) = poses[j].translation()[k];
        a(r, j, 3) = rotation.w();
        a(r, j, 4) = rotation.x();
        a(r, j, 5) = rotation.y();
        a(r, j, 6) = rotation.z();
      }
    }
    return out;
  }
  bool valid(const Array &q) const {
    shape(q);
    py::gil_scoped_release release;
    if (!in_bounds(q.data()))
      return false;
    if (!optimized_) {
      auto cfg = unpack(q.data());
      return checker_.isValidComposite(robots_, cfg) &&
             extras_.valid(robots_, cfg);
    }
    if (runtime_)
      return runtime_pack(q.data(), offsets_.back(), 1)[0] &&
             overlay_valid(q.data());
    comotion_python::Paths paths(robots_.size());
    auto configs = unpack(q.data());
    for (std::size_t r = 0; r < robots_.size(); ++r)
      paths[r].push_back(configs[r]);
    return comotion_python::states(checker_, robots_, paths)[0] &&
           extras_.valid(robots_, configs);
  }
  bool group_valid(const Array &q, const std::vector<int> &group,
                   bool include_other_robots = true) const {
    shape(q);
    std::vector<bool> selected(robots_.size(), false);
    for (int i : group) {
      if (i < 0 || std::size_t(i) >= robots_.size())
        throw std::invalid_argument("Invalid robot index");
      selected[i] = true;
    }
    py::gil_scoped_release release;
    if (!in_bounds(q.data()))
      return false;
    auto configs = unpack(q.data());
    for (std::size_t i = 0; i < robots_.size(); ++i) {
      if (selected[i] && !checker_.isValidSingleFull(*robots_[i], configs[i]))
        return false;
      for (std::size_t j = i + 1; j < robots_.size(); ++j)
        if ((include_other_robots ? selected[i] || selected[j]
                                  : selected[i] && selected[j]) &&
            !checker_.isValidPair(*robots_[i], configs[i], *robots_[j],
                                  configs[j]))
          return false;
    }
    return extras_.valid(robots_, configs, &group, include_other_robots);
  }
  bool motion_valid(const Array &from, const Array &to,
                    double resolution) const {
    shape(from);
    shape(to);
    if (!std::isfinite(resolution) || resolution <= 0)
      throw std::invalid_argument("Invalid resolution");
    py::gil_scoped_release release;
    return motion_impl(from.data(), to.data(), resolution);
  }
  bool motion_impl(const double *from, const double *to, double resolution,
                   comotion_python::EdgeStats *profile = nullptr) const {
    if (!in_bounds(from) || !in_bounds(to))
      return false;
    double distance = 0;
    for (std::size_t i = 0; i < offsets_.back(); ++i)
      distance = std::max(distance, std::abs(from[i] - to[i]));
    double steps = std::max(1., std::ceil(distance / resolution));
    if (steps > 10000000)
      throw std::invalid_argument("Edge requires over ten million samples");
    if (runtime_ && optimized_) {
      const std::size_t width = vamp::FloatVectorWidth, dof = offsets_.back();
      std::vector<double> samples(width * dof);
      for (int begin = 0; begin <= int(steps); begin += int(width)) {
        auto count = std::min(width, std::size_t(int(steps) + 1 - begin));
        for (std::size_t k = 0; k < count; ++k)
          for (std::size_t j = 0; j < dof; ++j)
            samples[k * dof + j] =
                from[j] + (double(begin + k) / steps) * (to[j] - from[j]);
        auto valid = runtime_pack(samples.data(), dof, count);
        for (std::size_t k = 0; k < count; ++k)
          if (!valid[k] || !overlay_valid(samples.data() + k * dof))
            return false;
      }
      return true;
    }
    comotion::CompositePathValidationOptions opts;
    opts.discrete_num_checks_hint = static_cast<int>(steps);
    opts.conflict_find_parallel_workers = 1;
    bool accepted;
    if (optimized_) {
      comotion_python::EdgeStats unused;
      accepted = profile ? comotion_python::edge_dispatch<true>(
                               edge_kernel_, checker_, robots_, unpack(from),
                               unpack(to), static_cast<int>(steps), *profile)
                         : comotion_python::edge_dispatch<false>(
                               edge_kernel_, checker_, robots_, unpack(from),
                               unpack(to), static_cast<int>(steps), unused);
    } else
      accepted = checker_.isCompositeMotionValid(robots_, unpack(from),
                                                 unpack(to), opts);
    if (!accepted || extras_.empty())
      return accepted;
    std::vector<double> q(offsets_.back());
    for (int k = 0; k <= int(steps); ++k) {
      double t = double(k) / steps;
      for (std::size_t j = 0; j < q.size(); ++j)
        q[j] = from[j] + t * (to[j] - from[j]);
      if (!extras_.valid(robots_, unpack(q.data())))
        return false;
    }
    return true;
  }
  py::array_t<bool> batch_valid(const Array &qs) const {
    if (qs.ndim() != 2 || std::size_t(qs.shape(1)) != offsets_.back())
      throw std::invalid_argument("Expected (batch,dof)");
    py::array_t<bool> result(qs.shape(0));
    auto out = result.mutable_data();
    py::gil_scoped_release release;
    if (runtime_ && optimized_) {
      for (py::ssize_t begin = 0; begin < qs.shape(0);
           begin += vamp::FloatVectorWidth) {
        auto count = std::min(std::size_t(vamp::FloatVectorWidth),
                              std::size_t(qs.shape(0) - begin));
        auto valid =
            runtime_pack(qs.data() + begin * qs.shape(1), qs.shape(1), count);
        for (std::size_t k = 0; k < count; ++k) {
          const double *q = qs.data() + (begin + k) * qs.shape(1);
          out[begin + k] = in_bounds(q) && valid[k] && overlay_valid(q);
        }
      }
      return result;
    }
    if (optimized_) {
      comotion_python::Paths paths(robots_.size());
      std::vector<py::ssize_t> rows;
      for (py::ssize_t k = 0; k < qs.shape(0); ++k) {
        const double *q = qs.data() + k * qs.shape(1);
        out[k] = false;
        if (!in_bounds(q))
          continue;
        rows.push_back(k);
        auto configs = unpack(q);
        for (std::size_t r = 0; r < robots_.size(); ++r)
          paths[r].push_back(std::move(configs[r]));
      }
      auto valid = comotion_python::states(checker_, robots_, paths);
      for (std::size_t k = 0; k < rows.size(); ++k)
        out[rows[k]] =
            valid[k] &&
            extras_.valid(robots_, unpack(qs.data() + rows[k] * qs.shape(1)));
      return result;
    }
    for (py::ssize_t k = 0; k < qs.shape(0); ++k) {
      const double *q = qs.data() + k * qs.shape(1);
      out[k] = in_bounds(q) && checker_.isValidComposite(robots_, unpack(q)) &&
               extras_.valid(robots_, unpack(q));
    }
    return result;
  }
  static void check_query(const py::object &mode, double tolerance) {
    if (!mode.is_none())
      throw std::invalid_argument("Generated scene expects its initial (None) mode");
    if (tolerance != 0.)
      throw std::invalid_argument("Generated scene supports zero collision tolerance only");
  }
  bool state_valid(const Array &q, const py::object &mode,
                   const std::vector<int> &group, double tolerance) const {
    check_query(mode, tolerance);
    return group.empty() ? valid(q) : group_valid(q, group, false);
  }
  py::array_t<bool> query_batch(const Array &qs, const py::object &mode,
                               const std::vector<int> &group, bool simd,
                               double tolerance) const {
    check_query(mode, tolerance);
    if (group.empty() && simd) return batch_valid(qs);
    if (qs.ndim() != 2 || std::size_t(qs.shape(1)) != offsets_.back())
      throw std::invalid_argument("Expected (batch,dof)");
    py::array_t<bool> out(qs.shape(0));
    for (py::ssize_t i = 0; i < qs.shape(0); ++i) {
      Array row({qs.shape(1)}, qs.data(i), qs);
      out.mutable_data()[i] = state_valid(row, mode, group, tolerance);
    }
    return out;
  }
  std::tuple<bool, int, int> edge_valid(
      const Array &a, const Array &b, const py::object &mode, int n, int begin,
      int end, bool endpoints, double tolerance, const std::string &packing,
      bool early_exit) const {
    check_query(mode, tolerance);
    shape(a); shape(b);
    // Use the same low-level slice semantics as PrimitiveScene.
    begin = std::clamp(begin < 0 ? n + begin : begin, 0, std::max(n, 0));
    end = std::clamp(end < 0 ? n + end : end, 0, std::max(n, 0));
    end = std::max(begin, end);
    const int d = offsets_.back();
    EdgeWorkspace work(robots_.size());
    EdgeStats unused;
    for (std::size_t r = 0; r < robots_.size(); ++r)
      work.paths[r].resize(16, std::vector<double>(robots_[r]->numJoints()));
    py::gil_scoped_release release;
    return check_edge_grid(a.data(), b.data(), d, n, begin, end, endpoints,
                           packing, [&](const double *samples, int rows) {
      for (int k = 0; k < rows; ++k)
        if (!in_bounds(samples + k * d)) return false;
      if (!vamp_ || !optimized_) {
        for (int k = 0; k < rows; ++k) {
          auto q = unpack(samples + k * d);
          if (!checker_.isValidComposite(robots_, q) || !extras_.valid(robots_, q))
            return false;
        }
        return true;
      }
      bool accepted = true;
      for (int offset = 0; offset < rows; offset += int(width)) {
        int count = std::min(int(width), rows - offset);
        work.rows.resize(count);
        std::iota(work.rows.begin(), work.rows.end(), 0);
        for (std::size_t r = 0; r < robots_.size(); ++r) {
          work.paths[r].resize(count);
          for (int k = 0; k < count; ++k) {
            auto &q = work.paths[r][k];
            q.resize(robots_[r]->numJoints());
            std::copy_n(samples + (offset + k) * d + offsets_[r], q.size(), q.begin());
          }
          if (!checker_.isRobotPathValid(*robots_[r], work.paths[r])) {
            accepted = false;
            if (early_exit) return false;
          }
        }
        if (!edge_pairs<3, false>(robots_, work, unused)) {
          accepted = false;
          if (early_exit) return false;
        }
        if (!extras_.empty())
          for (int k = 0; k < count; ++k)
            if (!extras_.valid(robots_, unpack(samples + (offset + k) * d))) {
              accepted = false;
              if (early_exit) return false;
            }
      }
      return accepted;
    });
  }
  void set_primitives(const std::vector<int> &owners,
                      const std::vector<std::string> &links,
                      const std::vector<int> &kinds, const Array &sizes,
                      const Array &poses,
                      const std::vector<std::vector<std::string>> &ignore) {
    extras_.set(robots_, owners, links, kinds, sizes, poses, ignore);
  }
  void reparent_primitive(int index, int robot, const std::string &link,
                          const Array &pose,
                          const std::vector<std::string> &ignore) {
    extras_.reparent(robots_, index, robot, link, pose, ignore);
  }
  Array primitive_poses(const Array &q) const {
    shape(q);
    if (!in_bounds(q.data()))
      throw std::invalid_argument("Configuration outside limits");
    auto poses = extras_.poses(robots_, unpack(q.data()));
    Array out({py::ssize_t(poses.size()), py::ssize_t(7)});
    for (std::size_t i = 0; i < poses.size(); ++i) {
      double *p = out.mutable_data() + 7 * i;
      const auto &t = poses[i];
      Eigen::Quaterniond r(t.linear());
      for (int j = 0; j < 3; ++j)
        p[j] = t.translation()[j];
      p[3] = r.w();
      p[4] = r.x();
      p[5] = r.y();
      p[6] = r.z();
    }
    return out;
  }
  void set_packed_world(bool enabled) { packed_world_ = enabled; }
  void set_optimized(bool enabled) {
    if (enabled && !runtime_ &&
        (!vamp_ ||
         std::any_of(robots_.begin(), robots_.end(), [](const auto *r) {
           return r->robotFamily() !=
                      comotion::RobotModel::RobotFamily::Panda &&
                  r->robotFamily() != comotion::RobotModel::RobotFamily::UR5;
         })))
      throw std::invalid_argument(
          "Cached SIMD path supports VAMP Panda and UR5 models");
    optimized_ = enabled;
  }
  void set_edge_kernel(const std::string &name) {
    if (name == "reference")
      edge_kernel_ = 0;
    else if (name == "mask")
      edge_kernel_ = 1;
    else if (name == "early")
      edge_kernel_ = 2;
    else if (name == "buffered")
      edge_kernel_ = 3;
    else
      throw std::invalid_argument(
          "edge kernel must be reference, mask, early or buffered");
  }
  py::dict profile_motion_batch(const Array &from, const Array &to,
                                double resolution) const {
    if (!optimized_ || runtime_)
      throw std::invalid_argument(
          "Edge profiling requires generated VAMP mode");
    if (from.ndim() != 2 || to.ndim() != 2 || from.shape(0) != to.shape(0) ||
        std::size_t(from.shape(1)) != offsets_.back() ||
        to.shape(1) != from.shape(1))
      throw std::invalid_argument("Expected matching (batch,dof) edge arrays");
    if (!std::isfinite(resolution) || resolution <= 0)
      throw std::invalid_argument("Invalid resolution");
    comotion_python::EdgeStats stats;
    py::array_t<bool> result(from.shape(0));
    auto out = result.mutable_data();
    double elapsed;
    {
      py::gil_scoped_release release;
      auto begin = comotion_python::Clock::now();
      for (py::ssize_t k = 0; k < from.shape(0); ++k)
        out[k] =
            motion_impl(from.data() + k * offsets_.back(),
                        to.data() + k * offsets_.back(), resolution, &stats);
      elapsed =
          std::chrono::duration<double>(comotion_python::Clock::now() - begin)
              .count();
    }
    py::dict report;
    report["valid"] = result;
    report["elapsed_seconds"] = elapsed;
#define EDGE_FIELD(name) report[#name] = stats.name
    EDGE_FIELD(preparation_seconds);
    EDGE_FIELD(self_environment_seconds);
    EDGE_FIELD(fk_seconds);
    EDGE_FIELD(pair_seconds);
    EDGE_FIELD(edges);
    EDGE_FIELD(planned_samples);
    EDGE_FIELD(packs);
    EDGE_FIELD(samples);
    EDGE_FIELD(self_calls);
    EDGE_FIELD(fk_calls);
    EDGE_FIELD(robot_pairs);
    EDGE_FIELD(sphere_pairs);
    EDGE_FIELD(self_rejections);
    EDGE_FIELD(pair_rejections);
#undef EDGE_FIELD
    return report;
  }
  py::array_t<bool> motion_batch_valid(const Array &from, const Array &to,
                                       double resolution) const {
    if (from.ndim() != 2 || to.ndim() != 2 || from.shape(0) != to.shape(0) ||
        std::size_t(from.shape(1)) != offsets_.back() ||
        to.shape(1) != from.shape(1))
      throw std::invalid_argument("Expected matching (batch,dof) edge arrays");
    if (!std::isfinite(resolution) || resolution <= 0)
      throw std::invalid_argument("Invalid resolution");
    py::array_t<bool> result(from.shape(0));
    auto out = result.mutable_data();
    py::gil_scoped_release release;
    for (py::ssize_t k = 0; k < from.shape(0); ++k)
      out[k] = motion_impl(from.data() + k * offsets_.back(),
                           to.data() + k * offsets_.back(), resolution);
    return result;
  }
};

} // namespace comotion_python

PYBIND11_MODULE(_collision, m) {
  using comotion_python::Scene;
  m.doc() = "CoMotion collision-only bridge; objects are not safe for "
            "concurrent queries";
  py::class_<Scene>(m, "Scene")
      .def(py::init<const std::string &, const std::string &, const Array &,
                    const Array &, const Array &, const std::string &>())
      .def("limits", &Scene::limits)
      .def("spheres", &Scene::spheres)
      .def("link_poses", &Scene::link_poses)
      .def("valid", &Scene::valid)
      .def("group_valid", &Scene::group_valid, py::arg("q"), py::arg("group"),
           py::arg("include_other_robots") = true)
      .def("motion_valid", &Scene::motion_valid)
      .def("initial", [](const Scene &) { return py::none(); })
      .def("state_valid", &Scene::valid, py::arg("q"))
      .def("state_valid", [](const Scene &scene, const Array &q, const py::object &mode) {
        Scene::check_query(mode, 0.);
        return scene.valid(q);
      }, py::arg("q"), py::arg("mode"))
      .def("state_valid", &Scene::state_valid, py::arg("q"), py::arg("mode"),
           py::arg("group") = std::vector<int>{}, py::arg("tolerance") = 0.)
      .def("batch_valid", &Scene::query_batch, py::arg("qs"), py::arg("mode") = py::none(),
           py::arg("group") = std::vector<int>{}, py::arg("simd") = true, py::arg("tolerance") = 0.)
      .def("batch_edge_valid", [](Scene &scene, Array from, Array to,
                                    py::object mode, py::array_t<int, py::array::c_style> counts,
                                    bool endpoints, double tolerance, const std::string &packing) {
        return comotion_python::check_edge_batch(from, to, counts,
          [&](Array a, Array b, int n) {
            return scene.edge_valid(a, b, mode, n, 0, n, endpoints, tolerance, packing, true);
          });
      }, py::arg("from_q"), py::arg("to_q"), py::arg("mode"), py::arg("counts"),
         py::arg("include_endpoints") = false, py::arg("tolerance") = 0.,
         py::arg("packing") = "binary")
      .def("edge_valid", &Scene::edge_valid, py::arg("from_q"), py::arg("to_q"),
           py::arg("mode"), py::arg("n"), py::arg("begin"), py::arg("end"),
           py::arg("include_endpoints"), py::arg("tolerance") = 0.,
           py::arg("packing") = "binary", py::arg("early_exit") = true)
      .def("set_optimized", &Scene::set_optimized)
      .def("set_packed_world", &Scene::set_packed_world)
      .def("set_primitives", &Scene::set_primitives)
      .def("reparent_primitive", &Scene::reparent_primitive, py::arg("index"),
           py::arg("robot"), py::arg("link"), py::arg("pose"),
           py::arg("ignore_links") = std::vector<std::string>{})
      .def("primitive_poses", &Scene::primitive_poses)
      .def("motion_batch_valid", &Scene::motion_batch_valid)
      .def("set_edge_kernel", &Scene::set_edge_kernel)
      .def("profile_motion_batch", &Scene::profile_motion_batch);
}
