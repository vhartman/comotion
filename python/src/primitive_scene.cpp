#include "primitive_scene.h"

namespace comotion_python::primitive {
void Scene::check(const std::shared_ptr<Mode> &m) const {
  if (!m || m->scene != this)
    throw std::invalid_argument("Mode belongs to another scene");
}

std::shared_ptr<Topology> Scene::topology(const Mode &m) {
  std::vector<int> key = m.parents;
  key.insert(key.end(), m.contacts.begin(), m.contacts.end());
  key.insert(key.end(), m.has_joint.begin(), m.has_joint.end());
  auto found = topology_cache.find(key);
  if (found != topology_cache.end())
    return found->second;
  auto t = std::make_shared<Topology>();
  int n = m.parents.size();
  t->owners.resize(n, -1);
  std::vector<int> marks(n), link(n);
  std::vector<std::map<int, int>> depths(n);
  std::function<void(int)> visit = [&](int i) {
    if (i < 0 || i >= n)
      throw std::invalid_argument("Invalid parent");
    if (marks[i] == 2)
      return;
    if (marks[i] == 1)
      throw std::invalid_argument("Cyclic attachments");
    marks[i] = 1;
    int p = m.parents[i];
    if (p >= 0)
      visit(p);
    else if (p != -1)
      throw std::invalid_argument("Invalid root");
    t->order.push_back(i);
    t->owners[i] =
        frame_robot[i] >= 0 ? frame_robot[i] : (p < 0 ? -1 : t->owners[p]);
    link[i] = (p < 0 || m.has_joint[i]) ? i : link[p];
    marks[i] = 2;
  };
  for (int i = 0; i < n; ++i)
    visit(i);
  for (int i = 0; i < n; ++i) {
    int d = 0;
    for (int p = m.parents[i]; p >= 0; p = m.parents[p]) {
      d += m.has_joint[p];
      depths[i][p] = d;
    }
  }
  for (int a = 0; a < int(shapes.size()); ++a)
    for (int b = a + 1; b < int(shapes.size()); ++b) {
      int i = shapes[a], j = shapes[b], x = link[i], y = link[j];
      if (!m.contacts[i] || !m.contacts[j] || x == y ||
          excluded_pairs.count({a, b}))
        continue;
      auto excluded = [&](int f, int a, int b) {
        auto it = depths[a].find(b);
        return m.contacts[f] < 0 && it != depths[a].end() &&
               it->second <= -m.contacts[f];
      };
      if (!excluded(i, x, y) && !excluded(j, y, x))
        t->pairs.emplace_back(a, b);
    }
  if (topology_cache.size() >= 128)
    topology_cache.erase(topology_cache.begin());
  topology_cache.emplace(std::move(key), t);
  return t;
}

void Scene::prepare(Mode &m) {
  if (!m.topology)
    m.topology = topology(m);
  int n = m.parents.size();
  std::vector<int> anchor(n, -1);
  std::vector<Transform> fixed(n);
  m.joint_parent.resize(joints.size());
  m.joint_origin.resize(joints.size());
  for (int f : m.topology->order) {
    int p = m.parents[f];
    anchor[f] = p < 0 ? -1 : anchor[p];
    fixed[f] = p < 0 ? m.local[f] : fixed[p] * m.local[f];
    for (int j : frame_joints[f]) {
      m.joint_parent[j] = anchor[f];
      m.joint_origin[j] = fixed[f];
      anchor[f] = j;
      fixed[f] = Transform::Identity();
    }
  }
  m.shape_parent.resize(shapes.size());
  m.shape_origin.resize(shapes.size());
  for (int i = 0; i < int(shapes.size()); ++i) {
    m.shape_parent[i] = anchor[shapes[i]];
    m.shape_origin[i] = fixed[shapes[i]];
  }
  auto &top = *m.topology;
  if (!top.prepared) {
    std::map<int, std::vector<int>> groups;
    for (int i = 0; i < int(shapes.size()); ++i)
      groups[m.shape_parent[i] < 0 ? -i - 1 : m.shape_parent[i]].push_back(i);
    std::vector<int> group_for_shape(shapes.size());
    for (const auto &[key, ids] : groups) {
      for (int i : ids)
        group_for_shape[i] = top.groups.size();
      top.groups.push_back(ids);
    }
    std::map<std::pair<int, int>, size_t> block_index;
    for (auto pair : top.pairs) {
      if (m.shape_parent[pair.first] < 0 && m.shape_parent[pair.second] < 0) {
        top.static_pairs.push_back(pair);
        continue;
      }
      top.moving_pairs.push_back(pair);
      int a = group_for_shape[pair.first], b = group_for_shape[pair.second];
      if (a > b)
        std::swap(a, b);
      auto [it, inserted] =
          block_index.emplace(std::make_pair(a, b), top.blocks.size());
      if (inserted)
        top.blocks.push_back({a, b, {}});
      top.blocks[it->second].pairs.push_back(pair);
    }
    top.prepared = true;
  }
  m.static_penetration = 0.;
  // Static shape origins are already in world coordinates.
  std::vector<Pose<double>> g(shapes.size());
  for (auto [a, b] : top.static_pairs) {
    g[a] = constant<double>(m.shape_origin[a]);
    g[b] = constant<double>(m.shape_origin[b]);
    m.static_penetration += pair_depth(a, b, g);
  }
  m.static_valid = m.static_penetration <= 0;
  // Membership is shared by attachment topology; poses and bounds are
  // specific to this mode, including its grasp transform.
  m.bounds.clear();
  for (const auto &ids : top.groups) {
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (int i : ids)
      center += m.shape_origin[i].translation();
    center /= double(ids.size());
    double radius = 0.;
    for (int i : ids) {
      double r = kinds[i] == 0   ? sizes[i][0]
                 : kinds[i] == 1 ? sizes[i][0] + sizes[i][1]
                 : kinds[i] == 3 ? std::hypot(sizes[i][0], sizes[i][1])
                                 : sizes[i].norm();
      radius = std::max(radius,
                        (m.shape_origin[i].translation() - center).norm() + r);
    }
    int first = ids.front();
    m.bounds.push_back(
        {first, m.shape_origin[first].inverse() * center, radius});
  }
  m.prepared = true;
  ++prepared_modes;
}

Transform Scene::local_pose(const Mode &m, const double *q, int frame) const {
  Transform v = m.local[frame];
  for (int j : frame_joints[frame]) {
    Transform motion = Transform::Identity();
    if (joints[j].kind)
      motion.translation() = joints[j].axis * q[j];
    else
      motion.linear() =
          Eigen::AngleAxisd(q[j], joints[j].axis).toRotationMatrix();
    v = v * motion;
  }
  return v;
}

Transform Scene::frame_pose(const Mode &m, const double *q, int frame) const {
  std::vector<int> chain;
  for (int f = frame; f >= 0; f = m.parents[f])
    chain.push_back(f);
  Transform pose = local_pose(m, q, chain.back());
  // Preserve the root-to-leaf multiplication order used by poses_raw.
  for (auto it = chain.rbegin() + 1; it != chain.rend(); ++it)
    pose = pose * local_pose(m, q, *it);
  return pose;
}

std::vector<Transform> Scene::poses_raw(const Mode &m, const double *q) const {
  std::vector<Transform> p(m.parents.size());
  for (int f : m.topology->order) {
    Transform v = local_pose(m, q, f);
    p[f] = m.parents[f] < 0 ? v : p[m.parents[f]] * v;
  }
  return p;
}

Scene::Scene(py::dict d) {
  auto m = std::make_shared<Mode>();
  m->scene = this;
  m->parents = d["parents"].cast<std::vector<int>>();
  m->contacts = d["contacts"].cast<std::vector<int>>();
  m->has_joint = d["has_joint"].cast<std::vector<int>>();
  m->q = d["start_q"].cast<std::vector<double>>();
  for (auto p : d["poses"].cast<std::vector<std::vector<double>>>())
    m->local.push_back(from_pose(p));
  auto names = d["frame_names"].cast<std::vector<std::string>>();
  int n = names.size();
  if (std::set<std::string>(names.begin(), names.end()).size() != names.size())
    throw std::invalid_argument("Duplicate frame names");
  if (int(m->parents.size()) != n || int(m->contacts.size()) != n ||
      int(m->has_joint.size()) != n || int(m->local.size()) != n)
    throw std::invalid_argument("Invalid frame arrays");
  frame_robot.resize(n, -1);
  frame_joints.resize(n);
  auto owners = d["frame_robot"].cast<std::map<std::string, int>>();
  for (int i = 0; i < n; ++i)
    if (owners.count(names[i]))
      frame_robot[i] = owners[names[i]];
  for (auto name : d["movable"].cast<std::vector<std::string>>()) {
    auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end())
      throw std::invalid_argument("Unknown movable frame");
    movable.push_back(it - names.begin());
  }
  for (auto v : d["joints"].cast<std::vector<std::vector<double>>>()) {
    if (v.size() != 5 ||
        !std::all_of(v.begin(), v.end(),
                     [](double x) { return std::isfinite(x); }) ||
        v[0] < 0 || v[0] >= n || v[0] != std::floor(v[0]) ||
        (v[1] != 0 && v[1] != 1))
      throw std::invalid_argument("Invalid joint");
    Joint j{int(v[0]), int(v[1]), {v[2], v[3], v[4]}};
    if (j.frame < 0 || j.frame >= n || (j.kind != 0 && j.kind != 1) ||
        std::abs(j.axis.norm() - 1) > 1e-10)
      throw std::invalid_argument("Unsupported joint");
    frame_joints[j.frame].push_back(joints.size());
    joints.push_back(j);
  }
  limits = d["limits"].cast<std::vector<std::array<double, 2>>>();
  if (limits.size() != joints.size() || m->q.size() != joints.size())
    throw std::invalid_argument("Invalid joint arrays");
  for (size_t i = 0; i < limits.size(); ++i)
    if (std::isnan(limits[i][0]) || std::isnan(limits[i][1]) ||
        limits[i][0] > limits[i][1] || !std::isfinite(m->q[i]))
      throw std::invalid_argument(
          "Invalid joint limits or initial configuration");
  shapes = d["shape_frames"].cast<std::vector<int>>();
  kinds = d["kinds"].cast<std::vector<int>>();
  for (auto v : d["sizes"].cast<std::vector<std::array<double, 3>>>())
    sizes.emplace_back(v[0], v[1], v[2]);
  if (shapes.size() != kinds.size() || shapes.size() != sizes.size())
    throw std::invalid_argument("Invalid shape arrays");
  for (int i = 0; i < int(shapes.size()); ++i)
    if (shapes[i] < 0 || shapes[i] >= n || (kinds[i] < 0 || kinds[i] > 3) ||
        !sizes[i].allFinite() || sizes[i][0] <= 0 ||
        (kinds[i] == 2 && sizes[i].minCoeff() <= 0) ||
        ((kinds[i] == 1 || kinds[i] == 3) && sizes[i][1] <= 0))
      throw std::invalid_argument("SIMD conversion supports finite spheres, "
                                  "capsules, cylinders and sharp boxes only");
  if (d.contains("excluded_pairs"))
    for (auto pair :
         d["excluded_pairs"].cast<std::vector<std::pair<int, int>>>()) {
      if (pair.first > pair.second)
        std::swap(pair.first, pair.second);
      if (pair.first < 0 || pair.second >= int(shapes.size()) ||
          pair.first == pair.second)
        throw std::invalid_argument("Invalid excluded shape pair");
      excluded_pairs.insert(pair);
    }
  prepare(*m);
  first = m;
}

std::shared_ptr<Mode> Scene::initial() const { return first; }

std::shared_ptr<Mode> Scene::transition(std::shared_ptr<Mode> prev,
                                        const std::vector<int> &indices,
                                        Array entry, int object, int parent,
                                        int effect,
                                        std::vector<double> appear) {
  check(prev);
  if (entry.ndim() != 1 || entry.size() != int(joints.size()))
    throw std::invalid_argument("Invalid entry configuration");
  auto m =
      object < 0 ? std::make_shared<Mode>(*prev) : std::make_shared<Mode>();
  if (object >= 0) {
    m->scene = this;
    m->parents = prev->parents;
    m->contacts = prev->contacts;
    m->has_joint = prev->has_joint;
    m->local = prev->local;
    m->q = prev->q;
    m->topology = prev->topology;
  }
  if (effect < 0 || effect > 2 || (object < 0 && effect != 0))
    throw std::invalid_argument("Invalid transition effect");
  auto q = entry.data();
  if (!std::all_of(q, q + entry.size(),
                   [](double x) { return std::isfinite(x); }))
    throw std::invalid_argument("Non-finite transition configuration");
  for (int i : indices) {
    if (i < 0 || i >= int(joints.size()))
      throw std::invalid_argument("Invalid transition index");
    m->q[i] = q[i];
  }
  if (object < 0)
    return m;
  if (object >= int(m->parents.size()) || parent < 0 ||
      parent >= int(m->parents.size()) || !frame_joints[object].empty())
    throw std::invalid_argument("Invalid attachment");
  if (effect == 1) {
    m->local[object] = from_pose(appear);
    m->contacts[object] = 1;
  }
  auto parent_pose = frame_pose(*m, m->q.data(), parent);
  auto object_pose = frame_pose(*m, m->q.data(), object);
  m->local[object] = parent_pose.inverse() * object_pose;
  m->parents[object] = parent;
  m->has_joint[object] = 1;
  if (m->contacts[object])
    m->contacts[object] = -1;
  // Validate topology before traversing descendants.
  m->topology = topology(*m);
  if (effect == 2)
    for (int f : m->topology->order)
      for (int a = f; a >= 0; a = m->parents[a])
        if (a == object) {
          m->contacts[f] = 0;
          break;
        }
  m->topology = topology(*m);
  return m;
}

py::array_t<double> Scene::poses(Array q, std::shared_ptr<Mode> m) const {
  check(m);
  if (q.ndim() != 1 || q.size() != int(joints.size()))
    throw std::invalid_argument("Invalid configuration");
  auto p = poses_raw(*m, q.data());
  py::array_t<double> out({int(p.size()), 7});
  auto v = out.mutable_unchecked<2>();
  for (int i = 0; i < int(p.size()); ++i) {
    auto a = to_pose(p[i]);
    for (int j = 0; j < 7; ++j)
      v(i, j) = a[j];
  }
  return out;
}

py::list Scene::info(std::shared_ptr<Mode> m) const {
  check(m);
  py::list out;
  for (int i : movable)
    if (m->contacts[i])
      out.append(py::make_tuple(i, m->parents[i], to_pose(m->local[i])));
  return out;
}

size_t Scene::prepared_count() const { return prepared_modes; }

bool Scene::state(Array q, std::shared_ptr<Mode> m,
                  const std::vector<int> &group, double tolerance) {
  check(m);
  if (q.ndim() != 1 || q.size() != int(joints.size()))
    throw std::invalid_argument("Expected a full configuration");
  if (!std::isfinite(tolerance) || tolerance < 0)
    throw std::invalid_argument("Invalid collision tolerance");
  if (!m->prepared)
    prepare(*m);
  py::gil_scoped_release release;
  return scalar_valid(*m, q.data(), group, tolerance);
}
} // namespace comotion_python::primitive
