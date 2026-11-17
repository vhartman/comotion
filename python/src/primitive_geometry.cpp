#include "primitive_scene.h"

namespace comotion_python::primitive {
Eigen::Vector3d Scene::box_extent(int i) const {
  if (kinds[i] == 1)
    return {sizes[i][0], sizes[i][0], sizes[i][1] + sizes[i][0]};
  if (kinds[i] == 3)
    return {sizes[i][0], sizes[i][0], sizes[i][1]};
  return sizes[i];
}

bool Scene::convex_pair(int a, int b) const {
  return kinds[a] != 0 && kinds[b] != 0 && (kinds[a] != 2 || kinds[b] != 2);
}

void Scene::support(const void *object, const ccd_vec3_t *direction,
                    ccd_vec3_t *result) {
  const auto &c = *static_cast<const Convex *>(object);
  Eigen::Vector3d world(ccdVec3X(direction), ccdVec3Y(direction),
                        ccdVec3Z(direction));
  Eigen::Vector3d d = c.rotation.transpose() * world, point;
  if (c.kind == 2)
    for (int i = 0; i < 3; ++i)
      point[i] = d[i] >= 0 ? c.size[i] : -c.size[i];
  else if (c.kind == 1) {
    double length = d.norm();
    point = length > 0 ? Eigen::Vector3d(d * (c.size[0] / length))
                       : Eigen::Vector3d(c.size[0], 0, 0);
    point.z() += d.z() > 0 ? c.size[1] : -c.size[1];
  } else {
    double length = std::hypot(d.x(), d.y());
    point = {length > 0 ? c.size[0] * d.x() / length : 0.,
             length > 0 ? c.size[0] * d.y() / length : 0.,
             (d.z() > 0 ? 1. : (d.z() < 0 ? -1. : 0.)) * c.size[1]};
  }
  point = c.center + c.rotation * point;
  ccdVec3Set(result, point.x(), point.y(), point.z());
}

void Scene::center(const void *object, ccd_vec3_t *result) {
  const auto &c = *static_cast<const Convex *>(object);
  ccdVec3Set(result, c.center.x(), c.center.y(), c.center.z());
}

double Scene::convex_depth(int a, int b,
                           const std::vector<Pose<double>> &g) const {
  Convex shapes[2];
  int ids[2] = {a, b};
  for (int k = 0; k < 2; ++k) {
    auto &c = shapes[k];
    int i = ids[k];
    c.kind = kinds[i];
    c.size = sizes[i];
    for (int row = 0; row < 3; ++row) {
      c.center[row] = g[i].m[4 * row + 3];
      for (int col = 0; col < 3; ++col)
        c.rotation(row, col) = g[i].m[4 * row + col];
    }
  }
  ccd_t ccd;
  CCD_INIT(&ccd);
  ccd.support1 = support;
  ccd.support2 = support;
  ccd.center1 = center;
  ccd.center2 = center;
  ccd.max_iterations = 1000;
  ccd.dist_tolerance = 1e-12;
  ccd.epa_tolerance = 1e-10;
  ccd.mpr_tolerance = 1e-6;
  ccd_real_t depth = 0;
  ccd_vec3_t direction, position;
  int status = ccdMPRPenetration(&shapes[0], &shapes[1], &ccd, &depth,
                                 &direction, &position);
  if (status == -2)
    throw std::runtime_error("Convex collision allocation failed");
  if (status == -1)
    return 0.;
  if (status != 0)
    throw std::runtime_error("Unexpected convex collision result");
  return std::max(0., double(depth));
}

bool Scene::relevant(const Mode &m, int a, int b,
                     const std::vector<int> &group) const {
  if (group.empty())
    return true;
  int x = m.topology->owners[shapes[a]], y = m.topology->owners[shapes[b]];
  auto selected = [&](int r) {
    return std::find(group.begin(), group.end(), r) != group.end();
  };
  return (selected(x) || selected(y)) && (x < 0 || selected(x)) &&
         (y < 0 || selected(y));
}

bool Scene::valid_q(const double *q) const {
  for (int j = 0; j < int(joints.size()); ++j)
    if (!std::isfinite(q[j]) || q[j] < limits[j][0] || q[j] > limits[j][1])
      return false;
  return true;
}

bool Scene::pair_valid(int a, int b, const std::vector<Pose<double>> &g) const {
  if (kinds[a] == 1 && kinds[b] == 1) // exact; MPR overestimates elongated pairs
    return clearance<double>(a, b, g) >= 0;
  if (convex_pair(a, b))
    return clearance<double>(a, b, g) > 0 || (convex_depth(a, b, g) <= 0);
  if (kinds[a] != 2 || kinds[b] != 2)
    return clearance<double>(a, b, g) >= 0;
  const auto &A = g[a].m;
  const auto &B = g[b].m;
  Eigen::Matrix3d ar, br;
  Eigen::Vector3d d;
  for (int i = 0; i < 3; ++i) {
    d[i] = B[4 * i + 3] - A[4 * i + 3];
    for (int j = 0; j < 3; ++j) {
      ar(i, j) = A[4 * i + j];
      br(i, j) = B[4 * i + j];
    }
  }
  bool separated = false;
  auto test = [&](Eigen::Vector3d v) {
    if (v.squaredNorm() < 1e-24)
      return;
    double r = (ar.transpose() * v).cwiseAbs().dot(sizes[a]) +
               (br.transpose() * v).cwiseAbs().dot(sizes[b]);
    if (std::abs(d.dot(v)) >= r)
      separated = true;
  };
  for (int i = 0; i < 3; ++i) {
    test(ar.col(i));
    test(br.col(i));
    for (int j = 0; j < 3; ++j)
      test(ar.col(i).cross(br.col(j)));
  }
  return separated;
}

double Scene::pair_depth(int a, int b,
                         const std::vector<Pose<double>> &g) const {
  if (kinds[a] == 1 && kinds[b] == 1) // exact; MPR overestimates elongated pairs
    return std::max(0., -clearance<double>(a, b, g));
  if (convex_pair(a, b))
    return clearance<double>(a, b, g) > 0 ? 0. : convex_depth(a, b, g);
  const auto &A = g[a].m;
  const auto &B = g[b].m;
  Eigen::Vector3d delta(B[3] - A[3], B[7] - A[7], B[11] - A[11]);
  if (kinds[a] == 0 && kinds[b] == 0)
    return std::max(0., sizes[a][0] + sizes[b][0] - delta.norm());
  if (kinds[a] == 0 || kinds[b] == 0) {
    int sphere = kinds[a] == 0 ? a : b, other = kinds[a] == 0 ? b : a;
    const auto &S = g[sphere].m;
    const auto &O = g[other].m;
    Eigen::Vector3d p(S[3] - O[3], S[7] - O[7], S[11] - O[11]), local;
    for (int i = 0; i < 3; ++i)
      local[i] = p[0] * O[i] + p[1] * O[4 + i] + p[2] * O[8 + i];
    double distance;
    if (kinds[other] == 1) {
      double axial = std::max(0., std::abs(local.z()) - sizes[other][1]);
      distance = std::sqrt(local.x() * local.x() + local.y() * local.y() +
                           axial * axial) -
                 sizes[other][0];
    } else if (kinds[other] == 3) {
      double radial = std::hypot(local.x(), local.y()) - sizes[other][0],
             axial = std::abs(local.z()) - sizes[other][1];
      distance = std::hypot(std::max(0., radial), std::max(0., axial)) +
                 std::min(0., std::max(radial, axial));
    } else {
      Eigen::Vector3d d = local.cwiseAbs() - sizes[other];
      distance = d.cwiseMax(0.).norm() + std::min(0., d.maxCoeff());
    }
    return std::max(0., sizes[sphere][0] - distance);
  }
  Eigen::Matrix3d ar, br;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      ar(i, j) = A[4 * i + j];
      br(i, j) = B[4 * i + j];
    }
  double depth = std::numeric_limits<double>::infinity();
  auto axis = [&](Eigen::Vector3d v) {
    double length = v.norm();
    if (length < 1e-12)
      return;
    double radius = (ar.transpose() * v).cwiseAbs().dot(sizes[a]) +
                    (br.transpose() * v).cwiseAbs().dot(sizes[b]);
    depth = std::min(depth, (radius - std::abs(delta.dot(v))) / length);
  };
  for (int i = 0; i < 3; ++i) {
    axis(ar.col(i));
    axis(br.col(i));
    for (int j = 0; j < 3; ++j)
      axis(ar.col(i).cross(br.col(j)));
  }
  return std::max(0., depth);
}

bool Scene::double_valid(const Mode &m, const double *q,
                         const std::vector<int> &group,
                         double tolerance) const {
  double depth = group.empty() ? m.static_penetration : 0.;
  if (!valid_q(q) || depth > tolerance)
    return false;
  auto g = geometry<double>(m, [&](int j) { return q[j]; });
  for (auto [a, b] : m.topology->moving_pairs)
    if (relevant(m, a, b, group)) {
      if (tolerance == 0) {
        if (!pair_valid(a, b, g))
          return false;
      } else {
        depth += pair_depth(a, b, g);
        if (depth > tolerance)
          return false;
      }
    }
  return true;
}

bool Scene::scalar_valid(const Mode &m, const double *q,
                         const std::vector<int> &group,
                         double tolerance) const {
  double depth = group.empty() ? m.static_penetration : 0.;
  if (!valid_q(q) || depth > tolerance)
    return false;
  auto g = geometry<double>(m, [&](int j) { return q[j]; });
  auto centers = bound_centers(m, g);
  std::vector<std::pair<std::pair<int, int>, double>> depths;
  for (const auto &block : m.topology->blocks) {
    const auto &a = centers[block.a], &b = centers[block.b];
    double x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
    double r = m.bounds[block.a].radius + m.bounds[block.b].radius;
    r += 1e-12 * (1 + r + std::abs(a[0]) + std::abs(a[1]) + std::abs(a[2]) +
                  std::abs(b[0]) + std::abs(b[1]) + std::abs(b[2]));
    if (x * x + y * y + z * z > r * r)
      continue;
    for (auto [i, j] : block.pairs) {
      if (!relevant(m, i, j, group))
        continue;
      if (tolerance == 0) {
        if (!pair_valid(i, j, g))
          return false;
      } else if (clearance<double>(i, j, g) <= 0.) {
        double value = pair_depth(i, j, g);
        if (value > tolerance)
          return false;
        if (value > 0.)
          depths.push_back({{i, j}, value});
      }
    }
  }
  std::sort(depths.begin(), depths.end());
  for (const auto &item : depths) {
    depth += item.second;
    if (depth > tolerance)
      return false;
  }
  return true;
}
} // namespace comotion_python::primitive
