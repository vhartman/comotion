#pragma once
// Runtime URDF sphere models: SIMD FK across configurations, reusable pair
// lists. Geometry edits are read at construction; no robot-specific generated
// constants.
#include <array>
#include <comotion/robot/RobotModel.h>
#include <stdexcept>
#include <vamp/vector.hh>
#include <vector>

namespace comotion_python {
using Packed = vamp::FloatVector<vamp::FloatVectorWidth>;
struct PackedPose {
  std::array<Packed, 12> m;
}; // row-major 3x4
struct PackedSphere {
  Packed x, y, z;
  float r;
};

inline PackedPose constant_pose(const Eigen::Affine3d &t) {
  PackedPose p;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      p.m[4 * i + j] = float(t.matrix()(i, j));
  return p;
}
inline PackedPose compose(const PackedPose &a, const PackedPose &b) {
  PackedPose c;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j)
      c.m[4 * i + j] = a.m[4 * i] * b.m[j] + a.m[4 * i + 1] * b.m[4 + j] +
                       a.m[4 * i + 2] * b.m[8 + j];
    c.m[4 * i + 3] = a.m[4 * i + 3] + a.m[4 * i] * b.m[3] +
                     a.m[4 * i + 1] * b.m[7] + a.m[4 * i + 2] * b.m[11];
  }
  return c;
}
struct RuntimeRobot {
  struct Link {
    int parent = -1, coordinate = -1;
    bool slide = false;
    Eigen::Affine3d origin;
    Eigen::Vector3d axis;
  };
  std::vector<Link> links;
  std::vector<comotion::CollisionSphere> spheres;
  std::vector<std::pair<int, int>> self_pairs;
  explicit RuntimeRobot(const comotion::RobotModel &robot) {
    auto names = robot.activeJointNames();
    for (const auto &source : robot.links()) {
      if (!source.collision_boxes.empty() ||
          !source.collision_cylinders.empty() ||
          !source.collision_meshes.empty())
        throw std::invalid_argument(
            "Runtime sphere models require sphere-only collision geometry");
      Link link;
      link.origin = robot.getBaseTransform();
      link.axis = Eigen::Vector3d::UnitZ();
      if (source.parent_joint_index >= 0) {
        const auto &j = robot.joints()[source.parent_joint_index];
        link.parent = robot.linkIndex(j.parent_link);
        link.origin = j.origin;
        link.axis = j.axis;
        if (link.parent < 0 || link.parent >= int(links.size()))
          throw std::invalid_argument(
              "URDF links must be in parent-before-child order");
        if (j.type != "fixed" && j.type != "revolute" && j.type != "prismatic")
          throw std::invalid_argument("Unsupported runtime joint type");
        if (j.type != "fixed") {
          if (std::abs(j.axis.norm() - 1) > 1e-8)
            throw std::invalid_argument("Joint axes must be unit length");
          auto it = std::find(names.begin(), names.end(), j.name);
          if (it == names.end())
            throw std::invalid_argument("Missing active joint");
          link.coordinate = int(it - names.begin());
          link.slide = j.type == "prismatic";
        }
      }
      links.push_back(link);
      for (const auto &s : source.collision_spheres) {
        if (!s.center.allFinite() || !std::isfinite(s.radius) || s.radius <= 0)
          throw std::invalid_argument("Invalid collision sphere");
        spheres.push_back(s);
      }
    }
    if (spheres.empty())
      throw std::invalid_argument("Robot has no collision spheres");
    for (int i = 0; i < int(spheres.size()); ++i)
      for (int j = i + 1; j < int(spheres.size()); ++j) {
        int a = spheres[i].link_index, b = spheres[j].link_index;
        if (a != b && !robot.isSelfCollisionDisabled(robot.links()[a].name,
                                                     robot.links()[b].name))
          self_pairs.emplace_back(i, j);
      }
  }
  std::vector<PackedSphere> geometry(const double *q, std::size_t stride,
                                     std::size_t rows) const {
    std::vector<PackedPose> poses;
    poses.reserve(links.size());
    for (const auto &link : links) {
      auto local = constant_pose(link.origin);
      if (link.coordinate >= 0) {
        std::array<float, vamp::FloatVectorWidth> values{};
        for (std::size_t k = 0; k < values.size(); ++k)
          values[k] = float(q[(k < rows ? k : 0) * stride + link.coordinate]);
        Packed x(values);
        auto motion = constant_pose(Eigen::Affine3d::Identity());
        if (link.slide)
          for (int i = 0; i < 3; ++i)
            motion.m[4 * i + 3] = float(link.axis[i]) * x;
        else {
          Packed c = x.cos(), s = x.sin(), d = Packed(1.f) - c;
          const auto &a = link.axis;
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              motion.m[4 * i + j] =
                  float(a[i] * a[j]) * d + (i == j ? c : Packed(0.f));
          motion.m[1] = motion.m[1] - float(a.z()) * s;
          motion.m[2] = motion.m[2] + float(a.y()) * s;
          motion.m[4] = motion.m[4] + float(a.z()) * s;
          motion.m[6] = motion.m[6] - float(a.x()) * s;
          motion.m[8] = motion.m[8] - float(a.y()) * s;
          motion.m[9] = motion.m[9] + float(a.x()) * s;
        }
        local = compose(local, motion);
      }
      poses.push_back(link.parent < 0 ? local
                                      : compose(poses[link.parent], local));
    }
    std::vector<PackedSphere> result;
    result.reserve(spheres.size());
    for (const auto &s : spheres) {
      const auto &t = poses[s.link_index].m;
      const auto p = s.center.cast<float>();
      result.push_back({t[0] * p.x() + t[1] * p.y() + t[2] * p.z() + t[3],
                        t[4] * p.x() + t[5] * p.y() + t[6] * p.z() + t[7],
                        t[8] * p.x() + t[9] * p.y() + t[10] * p.z() + t[11],
                        float(s.radius)});
    }
    return result;
  }
};
inline Packed sphere_clearance(const PackedSphere &a, const PackedSphere &b) {
  auto x = a.x - b.x, y = a.y - b.y, z = a.z - b.z;
  float r = a.r + b.r;
  return x * x + y * y + z * z - r * r;
}
template <class ExtraCheck>
inline std::array<bool, vamp::FloatVectorWidth>
runtime_valid(const std::vector<RuntimeRobot> &robots,
              const std::vector<std::size_t> &offsets,
              const std::vector<comotion::ObstacleSphere> &obstacles,
              const std::vector<comotion::ObstacleCylinder> &cylinders,
              const double *q, std::size_t stride, std::size_t rows,
              ExtraCheck extra) {
  std::array<bool, vamp::FloatVectorWidth> valid;
  valid.fill(true);
  auto reject = [&](const Packed &separation) {
    std::array<float, vamp::FloatVectorWidth> s;
    separation.to_array_unaligned(s.data());
    bool any = false;
    for (std::size_t k = 0; k < rows; ++k) {
      valid[k] = valid[k] && s[k] >= 0.f;
      any = any || valid[k];
    }
    return any;
  };
  std::vector<std::vector<PackedSphere>> all;
  all.reserve(robots.size());
  for (std::size_t r = 0; r < robots.size(); ++r) {
    all.push_back(robots[r].geometry(q + offsets[r], stride, rows));
    const auto &g = all.back();
    for (auto [i, j] : robots[r].self_pairs)
      if (!reject(sphere_clearance(g[i], g[j])))
        return valid;
    if (!extra(r, g, reject))
      return valid;
    for (const auto &a : g) {
      for (const auto &b : obstacles) {
        PackedSphere p{Packed(float(b.center.x())), Packed(float(b.center.y())),
                       Packed(float(b.center.z())), float(b.radius)};
        if (!reject(sphere_clearance(a, p)))
          return valid;
      }
      for (const auto &b : cylinders) {
        auto x = a.x - float(b.center.x()), y = a.y - float(b.center.y()),
             z = a.z - float(b.center.z());
        auto t = x * float(b.axis.x()) + y * float(b.axis.y()) +
                 z * float(b.axis.z());
        auto u = x - t * float(b.axis.x()), v = y - t * float(b.axis.y()),
             w = z - t * float(b.axis.z());
        auto radial =
            ((u * u + v * v + w * w).sqrt() - float(b.radius)).max(0.f);
        auto axial = (t.abs() - float(b.half_height)).max(0.f);
        if (!reject(radial * radial + axial * axial - a.r * a.r))
          return valid;
      }
    }
    for (std::size_t previous = 0; previous < r; ++previous)
      for (const auto &a : g)
        for (const auto &b : all[previous])
          if (!reject(sphere_clearance(a, b)))
            return valid;
  }
  return valid;
}
} // namespace comotion_python
