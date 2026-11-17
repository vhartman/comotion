#pragma once
// Frozen articulated sphere/box scenes. SIMD over configurations; no FCL.
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <ccd/ccd.h>
#include <chrono>
#include <cmath>
#include <comotion/collision/detail/VampPackingUtils.h>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <set>
#include <vamp/vector.hh>
#include <vector>
namespace py = pybind11;
namespace comotion_python::primitive {
using V = vamp::FloatVector<vamp::FloatVectorWidth>;
using Transform = Eigen::Isometry3d;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
constexpr int W = vamp::FloatVectorWidth;
inline Transform from_pose(const std::vector<double> &p) {
  if (p.size() != 7)
    throw std::invalid_argument("Expected xyz,wxyz pose");
  Transform t = Transform::Identity();
  t.translation() = Eigen::Vector3d(p[0], p[1], p[2]);
  Eigen::Quaterniond r(p[3], p[4], p[5], p[6]);
  if (!t.matrix().allFinite() || !r.coeffs().allFinite() || r.norm() < 1e-12)
    throw std::invalid_argument("Invalid pose");
  t.linear() = r.normalized().toRotationMatrix();
  return t;
}
inline std::array<double, 7> to_pose(const Transform &t) {
  Eigen::Quaterniond r(t.linear());
  r.normalize();
  return {t.translation().x(),
          t.translation().y(),
          t.translation().z(),
          r.w(),
          r.x(),
          r.y(),
          r.z()};
}
template <class T> struct Pose { std::array<T, 12> m; };
template <class T> Pose<T> constant(const Transform &a) {
  Pose<T> p;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      p.m[4 * i + j] = T(a.matrix()(i, j));
  return p;
}
template <class T> Pose<T> compose(const Pose<T> &a, const Pose<T> &b) {
  Pose<T> c;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j)
      c.m[4 * i + j] = a.m[4 * i] * b.m[j] + a.m[4 * i + 1] * b.m[4 + j] +
                       a.m[4 * i + 2] * b.m[8 + j];
    c.m[4 * i + 3] = a.m[4 * i + 3] + a.m[4 * i] * b.m[3] +
                     a.m[4 * i + 1] * b.m[7] + a.m[4 * i + 2] * b.m[11];
  }
  return c;
}
inline double sine(double x) { return std::sin(x); }
inline V sine(V x) { return x.sin(); }
inline double cosine(double x) { return std::cos(x); }
inline V cosine(V x) { return x.cos(); }
inline double absolute(double x) { return std::abs(x); }
inline V absolute(V x) { return x.abs(); }
inline double maximum(double x, double y) { return std::max(x, y); }
inline V maximum(V x, V y) { return x.max(y); }
inline double root(double x) { return std::sqrt(x); }
inline V root(V x) { return x.sqrt(); }
inline double clamp01(double x) { return std::min(std::max(x, 0.), 1.); }
inline V clamp01(V x) { return x.clamp(0.f, 1.f); }

// Exact capsule-capsule clearance: distance between the core segments (local
// z, half-lengths h) minus the radii; positive means disjoint. Clamps only, no
// branches, so it runs per SIMD lane. The alternation after the closed-form s
// makes it exact, including parallel and zero-length cores.
template <class T>
T capsule_gap(const std::array<T, 12> &A, double ra, double ha,
              const std::array<T, 12> &B, double rb, double hb) {
  std::array<T, 3> d1, d2, r;
  for (int i = 0; i < 3; ++i) {
    d1[i] = T(2 * ha) * A[4 * i + 2];
    d2[i] = T(2 * hb) * B[4 * i + 2];
    r[i] = (A[4 * i + 3] - T(ha) * A[4 * i + 2]) -
           (B[4 * i + 3] - T(hb) * B[4 * i + 2]);
  }
  auto dot = [](const std::array<T, 3> &x, const std::array<T, 3> &y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  };
  T a = dot(d1, d1), e = dot(d2, d2), b = dot(d1, d2), c = dot(d1, r),
    f = dot(d2, r), tiny = T(1e-30);
  T ai = maximum(a, tiny), ei = maximum(e, tiny);
  T s = clamp01((b * f - c * e) / maximum(a * e - b * b, tiny));
  T t = clamp01((b * s + f) / ei);
  s = clamp01((b * t - c) / ai);
  t = clamp01((b * s + f) / ei);
  T w0 = r[0] + s * d1[0] - t * d2[0], w1 = r[1] + s * d1[1] - t * d2[1],
    w2 = r[2] + s * d1[2] - t * d2[2];
  return root(w0 * w0 + w1 * w1 + w2 * w2) - T(ra + rb);
}
struct Joint {
  int frame, kind;
  Eigen::Vector3d axis;
};
struct Bound {
  int shape;
  Eigen::Vector3d center;
  double radius;
};
struct PairBlock {
  int a, b;
  std::vector<std::pair<int, int>> pairs;
};
struct Topology {
  std::vector<int> order, owners;
  std::vector<std::pair<int, int>> pairs, moving_pairs, static_pairs;
  std::vector<std::vector<int>> groups;
  std::vector<PairBlock> blocks;
  bool prepared = false;
};
struct Mode {
  const void *scene = nullptr;
  std::vector<int> parents, contacts, has_joint;
  std::vector<Transform> local;
  std::vector<double> q;
  std::shared_ptr<Topology> topology;
  // Collapse fixed frames into the preceding active joint for query FK.
  std::vector<int> joint_parent, shape_parent;
  std::vector<Transform> joint_origin, shape_origin;
  std::vector<Bound> bounds;
  bool prepared = false;
  bool static_valid = true;
  double static_penetration = 0;
};
class Scene {
  std::set<std::pair<int, int>> excluded_pairs;
  std::vector<Joint> joints;
  std::vector<std::vector<int>> frame_joints;
  std::vector<int> shapes, kinds, frame_robot, movable;
  std::vector<Eigen::Vector3d> sizes;
  std::vector<std::array<double, 2>> limits;
  std::map<std::vector<int>, std::shared_ptr<Topology>> topology_cache;
  std::shared_ptr<Mode> first;
  size_t prepared_modes = 0;
  void check(const std::shared_ptr<Mode> &m) const;
  std::shared_ptr<Topology> topology(const Mode &m);
  void prepare(Mode &m);
  template <class T>
  std::vector<std::array<T, 3>>
  bound_centers(const Mode &m, const std::vector<Pose<T>> &g) const {
    std::vector<std::array<T, 3>> out;
    out.reserve(m.bounds.size());
    for (const auto &b : m.bounds) {
      const auto &p = g[b.shape].m;
      std::array<T, 3> c;
      for (int i = 0; i < 3; ++i)
        c[i] = p[4 * i + 3] + p[4 * i] * T(b.center[0]) +
               p[4 * i + 1] * T(b.center[1]) + p[4 * i + 2] * T(b.center[2]);
      out.push_back(c);
    }
    return out;
  }
  Transform local_pose(const Mode &m, const double *q, int frame) const;
  Transform frame_pose(const Mode &m, const double *q, int frame) const;
  std::vector<Transform> poses_raw(const Mode &m, const double *q) const;
  template <class T, class Load>
  std::vector<Pose<T>> geometry(const Mode &m, Load load) const {
    std::vector<Pose<T>> jp(joints.size());
    std::vector<char> ready(joints.size());
    std::function<void(int)> evaluate = [&](int j) {
      if (ready[j])
        return;
      int p = m.joint_parent[j];
      if (p >= 0)
        evaluate(p);
      auto local = constant<T>(m.joint_origin[j]),
           motion = constant<T>(Transform::Identity());
      T x = load(j);
      const auto &a = joints[j].axis;
      if (joints[j].kind)
        for (int i = 0; i < 3; ++i)
          motion.m[4 * i + 3] = T(a[i]) * x;
      else {
        T c = cosine(x), s = sine(x), d = T(1) - c;
        for (int i = 0; i < 3; ++i)
          for (int k = 0; k < 3; ++k)
            motion.m[4 * i + k] = T(a[i] * a[k]) * d + (i == k ? c : T(0));
        motion.m[1] = motion.m[1] - T(a.z()) * s;
        motion.m[2] = motion.m[2] + T(a.y()) * s;
        motion.m[4] = motion.m[4] + T(a.z()) * s;
        motion.m[6] = motion.m[6] - T(a.x()) * s;
        motion.m[8] = motion.m[8] - T(a.y()) * s;
        motion.m[9] = motion.m[9] + T(a.x()) * s;
      }
      local = compose(local, motion);
      jp[j] = p < 0 ? local : compose(jp[p], local);
      ready[j] = 1;
    };
    for (int j = 0; j < int(joints.size()); ++j)
      evaluate(j);
    std::vector<Pose<T>> g;
    g.reserve(shapes.size());
    for (int i = 0; i < int(shapes.size()); ++i) {
      auto local = constant<T>(m.shape_origin[i]);
      int p = m.shape_parent[i];
      g.push_back(p < 0 ? local : compose(jp[p], local));
    }
    return g;
  }
  Eigen::Vector3d box_extent(int i) const;
  bool convex_pair(int a, int b) const;
  struct Convex {
    int kind;
    Eigen::Vector3d size, center;
    Eigen::Matrix3d rotation;
  };
  static void support(const void *object, const ccd_vec3_t *direction,
                      ccd_vec3_t *result);
  static void center(const void *object, ccd_vec3_t *result);
  double convex_depth(int a, int b, const std::vector<Pose<double>> &g) const;
  // Positive separation means disjoint. Box SAT uses normalized axes.
  template <class T>
  T clearance(int a, int b, const std::vector<Pose<T>> &g) const {
    const auto &A = g[a].m;
    const auto &B = g[b].m;
    std::array<T, 3> d{B[3] - A[3], B[7] - A[7], B[11] - A[11]};
    if (kinds[a] == 0 && kinds[b] == 0) {
      T r = T(sizes[a][0] + sizes[b][0]);
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - r * r;
    }
    if (kinds[a] == 0 || kinds[b] == 0) {
      int s = kinds[a] == 0 ? a : b, box = kinds[a] == 0 ? b : a;
      const auto &S = g[s].m;
      const auto &O = g[box].m;
      std::array<T, 3> delta{S[3] - O[3], S[7] - O[7], S[11] - O[11]};
      if (kinds[box] == 1 || kinds[box] == 3) {
        T x = delta[0] * O[0] + delta[1] * O[4] + delta[2] * O[8];
        T y = delta[0] * O[1] + delta[1] * O[5] + delta[2] * O[9];
        T z = delta[0] * O[2] + delta[1] * O[6] + delta[2] * O[10];
        T axial = maximum(absolute(z) - T(sizes[box][1]), T(0));
        if (kinds[box] == 1) {
          T radius = T(sizes[s][0] + sizes[box][0]);
          return x * x + y * y + axial * axial - radius * radius;
        }
        T radial = maximum(root(x * x + y * y) - T(sizes[box][0]), T(0));
        return radial * radial + axial * axial - T(sizes[s][0] * sizes[s][0]);
      }
      T sum = T(0);
      for (int i = 0; i < 3; ++i) {
        T v = delta[0] * O[i] + delta[1] * O[4 + i] + delta[2] * O[8 + i];
        T e = maximum(absolute(v) - T(sizes[box][i]), T(0));
        sum = sum + e * e;
      }
      return sum - T(sizes[s][0] * sizes[s][0]);
    }
    if (kinds[a] == 1 && kinds[b] == 1)
      return capsule_gap<T>(A, sizes[a][0], sizes[a][1], B, sizes[b][0],
                            sizes[b][1]);
    T best = T(-1e30);
    auto axis = [&](std::array<T, 3> v) {
      T length =
          root(maximum(v[0] * v[0] + v[1] * v[1] + v[2] * v[2], T(1e-30)));
      T dist = absolute(d[0] * v[0] + d[1] * v[1] + d[2] * v[2]), ra = T(0),
        rb = T(0);
      for (int i = 0; i < 3; ++i) {
        ra = ra + T(box_extent(a)[i]) *
                      absolute(v[0] * A[i] + v[1] * A[4 + i] + v[2] * A[8 + i]);
        rb = rb + T(box_extent(b)[i]) *
                      absolute(v[0] * B[i] + v[1] * B[4 + i] + v[2] * B[8 + i]);
      }
      // A degenerate cross axis cannot separate the boxes. Its zero gap
      // is intentionally left uncertain and resolved in double precision.
      best = maximum(best, (dist - ra - rb) / length);
    };
    for (int i = 0; i < 3; ++i) {
      axis({A[i], A[4 + i], A[8 + i]});
      axis({B[i], B[4 + i], B[8 + i]});
    }
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        axis({A[4 + i] * B[8 + j] - A[8 + i] * B[4 + j],
              A[8 + i] * B[j] - A[i] * B[8 + j],
              A[i] * B[4 + j] - A[4 + i] * B[j]});
    return best;
  }
  bool relevant(const Mode &m, int a, int b,
                const std::vector<int> &group) const;
  bool valid_q(const double *q) const;
  bool pair_valid(int a, int b, const std::vector<Pose<double>> &g) const;
  double pair_depth(int a, int b, const std::vector<Pose<double>> &g) const;
  bool double_valid(const Mode &m, const double *q,
                    const std::vector<int> &group, double tolerance) const;

  bool scalar_valid(const Mode &m, const double *q,
                    const std::vector<int> &group, double tolerance) const;

public:
  Scene(py::dict d);
  std::shared_ptr<Mode> initial() const;
  std::shared_ptr<Mode> transition(std::shared_ptr<Mode> prev,
                                   const std::vector<int> &indices, Array entry,
                                   int object, int parent, int effect,
                                   std::vector<double> appear);
  py::array_t<double> poses(Array q, std::shared_ptr<Mode> m) const;
  py::list info(std::shared_ptr<Mode> m) const;
  size_t prepared_count() const;
  bool state(Array q, std::shared_ptr<Mode> m, const std::vector<int> &group,
             double tolerance);
  std::tuple<bool, int, int> edge(Array from, Array to, std::shared_ptr<Mode> m,
                                  int n, int begin, int end, bool endpoints,
                                  double tolerance, const std::string &packing,
                                  bool early_exit);
  py::array_t<bool> batch(Array qs, std::shared_ptr<Mode> m,
                          std::vector<int> group, bool simd, double tolerance,
                          py::object diagnostics = py::none());
  py::array_t<bool> batch_impl(Array qs, std::shared_ptr<Mode> m,
                               const std::vector<int> &group, bool simd,
                               double tolerance, py::object diagnostics,
                               bool stop_on_collision);
};

} // namespace comotion_python::primitive
