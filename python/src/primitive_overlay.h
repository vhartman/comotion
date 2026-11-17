#pragma once
// Exact primitive additions to generated VAMP robot sphere models.
#include <comotion/robot/RobotModel.h>
#include <fcl/fcl.h>
#include <set>

class PrimitiveOverlay {
  struct Object {
    int robot = -1, link = -1, kind = 0;
    Eigen::Vector3d size = Eigen::Vector3d::Zero();
    Eigen::Isometry3d local = Eigen::Isometry3d::Identity();
    std::shared_ptr<fcl::CollisionGeometryd> geometry;
    std::set<int> ignored_links;
  };
  std::vector<Object> objects_, environment_;
  using Robots = std::vector<const comotion::RobotModel *>;
  using Configs = std::vector<std::vector<double>>;
  static bool intersects(const Object &a, const Eigen::Isometry3d &ta,
                         const Object &b, const Eigen::Isometry3d &tb) {
    fcl::CollisionObjectd x(a.geometry, ta), y(b.geometry, tb);
    if (!x.getAABB().overlap(y.getAABB()))
      return false;
    fcl::CollisionRequestd req;
    fcl::CollisionResultd result;
    fcl::collide(&x, &y, req, result);
    return result.isCollision();
  }
  static bool sphere_intersects(const Object &o, const Eigen::Isometry3d &t,
                                const comotion::CollisionSphere &sphere) {
    Eigen::Vector3d p =
        t.linear().transpose() * (sphere.center - t.translation());
    double radius = sphere.radius, distance;
    if (o.kind == 0) {
      radius += o.size[0];
      distance = p.squaredNorm();
    } else if (o.kind == 1) {
      p.z() -= std::clamp(p.z(), -o.size[1], o.size[1]);
      radius += o.size[0];
      distance = p.squaredNorm();
    } else if (o.kind == 2)
      distance = (p.cwiseAbs() - o.size).cwiseMax(0.).squaredNorm();
    else {
      double radial = std::max(0., p.head<2>().norm() - o.size[0]);
      double axial = std::max(0., std::abs(p.z()) - o.size[1]);
      distance = radial * radial + axial * axial;
    }
    return distance < radius * radius;
  }
  static Eigen::Isometry3d transform(const double *p) {
    for (int i = 0; i < 7; ++i)
      if (!std::isfinite(p[i]))
        throw std::invalid_argument("Nonfinite primitive pose");
    Eigen::Quaterniond q(p[3], p[4], p[5], p[6]);
    if (std::abs(q.norm() - 1) > 1e-6)
      throw std::invalid_argument("Primitive quaternion must be normalized");
    Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
    t.linear() = q.normalized().toRotationMatrix();
    t.translation() = Eigen::Vector3d(p[0], p[1], p[2]);
    return t;
  }

public:
  bool empty() const { return objects_.empty(); }
  bool world_only() const {
    return std::all_of(objects_.begin(), objects_.end(),
                       [](const auto &o) { return o.robot < 0; });
  }
  template <class Reject>
  bool packed_world(const comotion_python::RuntimeRobot &robot,
                    const std::vector<comotion_python::PackedSphere> &spheres,
                    Reject reject) const {
    using comotion_python::Packed;
    for (const auto &o : objects_) {
      if (o.robot >= 0)
        continue;
      for (std::size_t i = 0; i < spheres.size(); ++i) {
        if (o.ignored_links.count(robot.spheres[i].link_index))
          continue;
        const auto &s = spheres[i];
        const auto &R = o.local.linear();
        const auto &t = o.local.translation();
        auto dx = s.x - float(t.x()), dy = s.y - float(t.y()),
             dz = s.z - float(t.z());
        auto x =
            float(R(0, 0)) * dx + float(R(1, 0)) * dy + float(R(2, 0)) * dz;
        auto y =
            float(R(0, 1)) * dx + float(R(1, 1)) * dy + float(R(2, 1)) * dz;
        auto z =
            float(R(0, 2)) * dx + float(R(1, 2)) * dy + float(R(2, 2)) * dz;
        float radius = s.r;
        Packed distance;
        if (o.kind == 0) {
          radius += float(o.size[0]);
          distance = x * x + y * y + z * z;
        } else if (o.kind == 1) {
          z = z +
              (Packed(0.f) - z.max(float(-o.size[1]))).max(float(-o.size[1]));
          radius += float(o.size[0]);
          distance = x * x + y * y + z * z;
        } else if (o.kind == 2) {
          x = (x.abs() - float(o.size[0])).max(0.f);
          y = (y.abs() - float(o.size[1])).max(0.f);
          z = (z.abs() - float(o.size[2])).max(0.f);
          distance = x * x + y * y + z * z;
        } else {
          auto radial = ((x * x + y * y).sqrt() - float(o.size[0])).max(0.f);
          auto axial = (z.abs() - float(o.size[1])).max(0.f);
          distance = radial * radial + axial * axial;
        }
        if (!reject(distance - radius * radius))
          return false;
      }
    }
    return true;
  }

  void spheres(const std::vector<comotion::ObstacleSphere> &values) {
    for (const auto &s : values) {
      Object o;
      o.geometry = std::make_shared<fcl::Sphered>(s.radius);
      o.local.translation() = s.center;
      environment_.push_back(o);
    }
  }
  void cylinders(const std::vector<comotion::ObstacleCylinder> &values) {
    for (const auto &c : values) {
      Object o;
      o.geometry =
          std::make_shared<fcl::Cylinderd>(c.radius, 2 * c.half_height);
      o.local.translation() = c.center;
      o.local.linear() =
          Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), c.axis)
              .toRotationMatrix();
      environment_.push_back(o);
    }
  }
  void set(const Robots &robots, const std::vector<int> &owners,
           const std::vector<std::string> &links, const std::vector<int> &kinds,
           const Array &sizes, const Array &poses,
           const std::vector<std::vector<std::string>> &ignore) {
    auto n = owners.size();
    if (links.size() != n || kinds.size() != n || ignore.size() != n ||
        sizes.ndim() != 2 || std::size_t(sizes.shape(0)) != n ||
        sizes.shape(1) != 3 || poses.ndim() != 2 ||
        std::size_t(poses.shape(0)) != n || poses.shape(1) != 7)
      throw std::invalid_argument("Invalid primitive arrays");
    std::vector<Object> proposed;
    for (std::size_t i = 0; i < n; ++i) {
      Object o;
      o.robot = owners[i];
      o.local = transform(poses.data() + 7 * i);
      if (o.robot < -1 || o.robot >= int(robots.size()))
        throw std::invalid_argument("Invalid primitive owner");
      if (o.robot >= 0) {
        o.link = robots[o.robot]->linkIndex(links[i]);
        if (o.link < 0)
          throw std::invalid_argument("Unknown attachment link");
        for (const auto &name : ignore[i]) {
          int link = robots[o.robot]->linkIndex(name);
          if (link < 0)
            throw std::invalid_argument("Unknown contact-exclusion link");
          o.ignored_links.insert(link);
        }
      } else {
        if (!links[i].empty())
          throw std::invalid_argument("World primitives have no parent link");
        for (const auto &name : ignore[i]) {
          int link = robots[0]->linkIndex(name);
          if (link < 0)
            throw std::invalid_argument("Unknown world contact-exclusion link");
          o.ignored_links.insert(link);
        }
      }
      const double *s = sizes.data() + 3 * i;
      o.kind = kinds[i];
      o.size = Eigen::Vector3d(s[0], s[1], s[2]);
      for (int k = 0; k < 3; ++k)
        if (!std::isfinite(s[k]) || s[k] < 0)
          throw std::invalid_argument("Invalid primitive dimensions");
      if (kinds[i] == 0 && s[0] > 0)
        o.geometry = std::make_shared<fcl::Sphered>(s[0]);
      else if (kinds[i] == 1 && s[0] > 0)
        o.geometry = std::make_shared<fcl::Capsuled>(s[0], 2 * s[1]);
      else if (kinds[i] == 2 && s[0] > 0 && s[1] > 0 && s[2] > 0)
        o.geometry = std::make_shared<fcl::Boxd>(2 * s[0], 2 * s[1], 2 * s[2]);
      else if (kinds[i] == 3 && s[0] > 0 && s[1] > 0)
        o.geometry = std::make_shared<fcl::Cylinderd>(s[0], 2 * s[1]);
      else
        throw std::invalid_argument("Unsupported primitive kind or dimensions");
      proposed.push_back(std::move(o));
    }
    objects_ = std::move(proposed);
  }
  void reparent(const Robots &robots, int index, int robot,
                const std::string &link, const Array &p,
                const std::vector<std::string> &ignore) {
    if (index < 0 || index >= int(objects_.size()) || p.ndim() != 1 ||
        p.size() != 7)
      throw std::invalid_argument("Invalid attachment update");
    if (robot < -1 || robot >= int(robots.size()))
      throw std::invalid_argument("Invalid attachment robot");
    Object proposed = objects_[index];
    proposed.robot = robot;
    proposed.local = transform(p.data());
    proposed.link = -1;
    proposed.ignored_links.clear();
    if (robot >= 0) {
      proposed.link = robots[robot]->linkIndex(link);
      if (proposed.link < 0)
        throw std::invalid_argument("Unknown attachment link");
      for (const auto &name : ignore) {
        int i = robots[robot]->linkIndex(name);
        if (i < 0)
          throw std::invalid_argument("Unknown excluded link");
        proposed.ignored_links.insert(i);
      }
    } else if (!link.empty() || !ignore.empty())
      throw std::invalid_argument("World primitives have no robot links");
    objects_[index] = std::move(proposed);
  }
  std::vector<Eigen::Isometry3d> poses(const Robots &robots,
                                       const Configs &configs) const {
    std::vector<std::vector<Eigen::Affine3d>> transforms(robots.size());
    std::vector<Eigen::Isometry3d> result;
    for (const auto &o : objects_) {
      Eigen::Isometry3d t = o.local;
      if (o.robot >= 0) {
        if (transforms[o.robot].empty())
          transforms[o.robot] =
              robots[o.robot]->getLinkTransforms(configs[o.robot]);
        t.matrix() = transforms[o.robot][o.link].matrix() * o.local.matrix();
      }
      result.push_back(t);
    }
    return result;
  }
  bool valid(const Robots &robots, const Configs &configs,
             const std::vector<int> *group = nullptr, bool others = true,
             bool skip_world_robot = false) const {
    if (empty())
      return true;
    // Attachment poses and robot spheres share one FK evaluation per robot.
    std::vector<std::vector<Eigen::Affine3d>> transforms(robots.size());
    auto fk = [&](std::size_t r) -> const std::vector<Eigen::Affine3d> & {
      if (transforms[r].empty())
        transforms[r] = robots[r]->getLinkTransforms(configs[r]);
      return transforms[r];
    };
    std::vector<Eigen::Isometry3d> world;
    world.reserve(objects_.size());
    for (const auto &o : objects_) {
      Eigen::Isometry3d t = o.local;
      if (o.robot >= 0)
        t.matrix() = fk(o.robot)[o.link].matrix() * o.local.matrix();
      world.push_back(t);
    }
    auto selected = [&](int owner) {
      return !group || (owner >= 0 && std::find(group->begin(), group->end(),
                                                owner) != group->end());
    };
    auto relevant = [&](int a, int b) {
      return group ? (selected(a) || selected(b)) &&
                         (others ||
                          ((a < 0 || selected(a)) && (b < 0 || selected(b))))
                   : (a >= 0 || b >= 0);
    };
    std::vector<std::vector<comotion::CollisionSphere>> spheres(robots.size());
    for (std::size_t i = 0; i < objects_.size(); ++i) {
      const auto &o = objects_[i];
      for (const auto &e : environment_)
        if (relevant(o.robot, -1) && intersects(o, world[i], e, e.local))
          return false;
      for (std::size_t j = 0; j < i; ++j)
        if (relevant(o.robot, objects_[j].robot) &&
            intersects(o, world[i], objects_[j], world[j]))
          return false;
      for (std::size_t r = 0; r < robots.size(); ++r) {
        if (!relevant(o.robot, r) || (skip_world_robot && o.robot < 0))
          continue;
        if (spheres[r].empty()) {
          const auto &tf = fk(r);
          for (const auto &link : robots[r]->links())
            for (const auto &local : link.collision_spheres) {
              auto sphere = local;
              sphere.center = tf[link.index] * local.center;
              spheres[r].push_back(sphere);
            }
        }
        for (const auto &s : spheres[r]) {
          if ((o.robot < 0 || o.robot == int(r)) &&
              o.ignored_links.count(s.link_index))
            continue;
          if (sphere_intersects(o, world[i], s))
            return false;
        }
      }
    }
    return true;
  }
};
