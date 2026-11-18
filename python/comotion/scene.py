"""One geometry definition, with explicit selection of a collision implementation.

Importers preserve supported collision shapes. They never approximate meshes,
rounded boxes or unrecognized joints. Generated kernels are admitted only for
an exact match to their supported model. They do not replace supplied geometry.
"""
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import xml.etree.ElementTree as ET

import numpy as np

from ._primitive import Scene as PrimitiveScene
from .resources import resource_directory


class UnsupportedScene(ValueError):
    """The requested implementation cannot represent this scene exactly."""


def _vector(value, size, label):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'{label}: expected {size} finite numbers')
    return a.tolist()


def _pose(origin):
    if origin is None:
        return [0., 0., 0., 1., 0., 0., 0.]
    xyz = _vector(origin.get('xyz', '0 0 0').split(), 3, 'origin xyz')
    r, p, y = np.asarray(_vector(origin.get('rpy', '0 0 0').split(), 3, 'origin rpy')) / 2
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    return xyz + [cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                  cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy]


def _rotation(q):
    w, x, y, z = _vector(q, 4, 'quaternion')
    if abs(np.linalg.norm(q) - 1) > 1e-10:
        raise ValueError('Expected a unit quaternion')
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _axis_pose(axis):
    axis = np.asarray(_vector(axis, 3, 'cylinder axis'))
    if abs(np.linalg.norm(axis) - 1) > 1e-10:
        raise ValueError('Expected a unit cylinder axis')
    if axis[2] < -1 + 1e-12:
        return [0., 1., 0., 0.]
    q = np.array([1 + axis[2], -axis[1], axis[0], 0.])
    return (q / np.linalg.norm(q)).tolist()


def import_robot_scene(urdf, srdf, bases, obstacles=(), *, start_q=None):
    """Import repeated URDF robots and primitive obstacles without approximation.

    Bases use ``position`` and optional ``quaternion_xyzw``. Obstacles use the
    built-in scene format: sphere (center/radius), cylinder (center/unit axis/
    radius/half_height), or box (center/half_extents/quaternion_xyzw).
    Joint coordinates follow URDF joint order. Only fixed, revolute and
    prismatic joints without mimic are supported. Mimic metadata on fixed
    joints has no kinematic effect (as in the packaged Panda fingers). Collision meshes are rejected.
    """
    root = ET.parse(urdf).getroot()
    links = root.findall('link'); joints = root.findall('joint')
    names = [link.get('name') for link in links]
    if not links or len(set(names)) != len(names) or None in names:
        raise ValueError('URDF requires unique link names')
    by_child = {}
    for joint in joints:
        if joint.get('type') not in ('fixed', 'revolute', 'prismatic') or (joint.get('type') != 'fixed' and joint.find('mimic') is not None):
            raise UnsupportedScene(f'Unsupported URDF joint: {joint.get("name")}')
        parent = joint.find('parent').get('link'); child = joint.find('child').get('link')
        if parent not in names or child not in names or child in by_child:
            raise ValueError('Invalid URDF joint tree')
        by_child[child] = joint
    roots = set(names) - set(by_child)
    if len(roots) != 1:
        raise ValueError('Expected one URDF root link')
    ordered = []
    pending = list(links)
    while pending:
        before = len(pending)
        for link in pending[:]:
            joint = by_child.get(link.get('name'))
            if joint is None or joint.find('parent').get('link') in {l.get('name') for l in ordered}:
                ordered.append(link)
                pending.remove(link)
        if len(pending) == before:
            raise ValueError('Cyclic URDF joint tree')
    active = [j for j in joints if j.get('type') != 'fixed']
    disabled = set()
    if srdf is not None:
        for entry in ET.parse(srdf).getroot().findall('disable_collisions'):
            pair = frozenset((entry.get('link1'), entry.get('link2')))
            if not pair.issubset(names):
                raise ValueError('SRDF exclusion references an unknown link')
            disabled.add(pair)
    if not len(bases):
        raise ValueError('Expected at least one robot base')
    d = dict(frame_names=[], parents=[], poses=[], has_joint=[], contacts=[],
             joints=[], joint_names=[], limits=[], frame_robot={}, robot_names=[],
             shape_frames=[], kinds=[], sizes=[], movable=[], actions=[], tolerance=0.)
    def frame(name, parent, pose, boundary=False, contact=0, owner=None):
        i = len(d['frame_names'])
        d['frame_names'].append(name); d['parents'].append(parent); d['poses'].append(pose)
        d['has_joint'].append(int(boundary)); d['contacts'].append(contact)
        if owner is not None:
            d['frame_robot'][name] = owner
        return i
    world = frame('world', -1, [0., 0., 0., 1., 0., 0., 0.], True)
    robot_shapes = []
    for r, base in enumerate(bases):
        if set(base) - {'position', 'quaternion_xyzw'}:
            raise UnsupportedScene('Unknown robot base fields')
        xyz = _vector(base['position'], 3, 'base position')
        q = np.roll(_vector(base.get('quaternion_xyzw', [0, 0, 0, 1]), 4, 'base quaternion'), 1).tolist()
        _rotation(q)
        d['robot_names'].append(f'robot{r}')
        frames = {}
        for link in ordered:
            name = link.get('name'); joint = by_child.get(name)
            parent = world if joint is None else frames[joint.find('parent').get('link')]
            pose = xyz + q if joint is None else _pose(joint.find('origin'))
            f = frame(f'r{r}/{name}', parent, pose, True, owner=r); frames[name] = f
            for k, collision in enumerate(link.findall('collision')):
                geometry = collision.find('geometry')
                if geometry is None or len(geometry) != 1:
                    raise UnsupportedScene(f'{name}: expected one collision primitive')
                shape = geometry[0]
                if shape.tag == 'sphere':
                    kind, size = 0, [float(shape.get('radius')), 0., 0.]
                elif shape.tag == 'cylinder':
                    kind, size = 3, [float(shape.get('radius')), float(shape.get('length'))/2, 0.]
                elif shape.tag == 'box':
                    kind, size = 2, (np.asarray(_vector(shape.get('size').split(), 3, 'box size'))/2).tolist()
                else:
                    raise UnsupportedScene(f'{name}: unsupported collision shape {shape.tag}')
                cf = frame(f'r{r}/{name}/shape{k}', f, _pose(collision.find('origin')), contact=1)
                d['shape_frames'].append(cf); d['kinds'].append(kind); d['sizes'].append(size)
                robot_shapes.append((r, name))
        for joint in active:
            axis = joint.find('axis'); limit = joint.find('limit')
            if limit is None or 'lower' not in limit.attrib or 'upper' not in limit.attrib:
                raise UnsupportedScene(f'{joint.get("name")}: finite joint limits required')
            d['joints'].append([frames[joint.find('child').get('link')], int(joint.get('type') == 'prismatic'),
                                *_vector(('1 0 0' if axis is None else axis.get('xyz')).split(), 3, 'joint axis')])
            d['joint_names'].append(f'r{r}/{joint.get("name")}')
            d['limits'].append([float(limit.get('lower')), float(limit.get('upper'))])
    nr = len(robot_shapes)
    for k, ob in enumerate(obstacles):
        typ = ob['type']; q = [1., 0., 0., 0.]
        allowed = {'type', 'center'}
        if typ == 'sphere':
            kind, size = 0, [ob['radius'], 0., 0.]; allowed |= {'radius'}
        elif typ == 'cylinder':
            kind, size = 3, [ob['radius'], ob['half_height'], 0.]
            q = _axis_pose(ob['axis']); allowed |= {'radius', 'half_height', 'axis'}
        elif typ == 'box':
            kind, size = 2, _vector(ob['half_extents'], 3, 'box half extents')
            q = np.roll(_vector(ob.get('quaternion_xyzw', [0,0,0,1]), 4, 'box quaternion'), 1).tolist()
            _rotation(q)
            allowed |= {'half_extents', 'quaternion_xyzw'}
        else:
            raise UnsupportedScene(f'Unsupported obstacle: {typ}')
        if set(ob) - allowed:
            raise UnsupportedScene(f'Unknown {typ} obstacle fields: {set(ob) - allowed}')
        f = frame(f'obstacle{k}', world, _vector(ob['center'], 3, 'obstacle center') + q, True, 1)
        d['shape_frames'].append(f); d['kinds'].append(kind); d['sizes'].append(size)
    excluded = []
    for r in range(len(bases)):
        indices = [i for i, (owner, _) in enumerate(robot_shapes) if owner == r]
        excluded.extend([a, b] for a, b in combinations(indices, 2)
                        if robot_shapes[a][1] == robot_shapes[b][1] or
                        frozenset((robot_shapes[a][1], robot_shapes[b][1])) in disabled)
    excluded += [list(pair) for pair in combinations(range(nr, len(d['shape_frames'])), 2)]
    d['excluded_pairs'] = excluded
    d['start_q'] = ([0.] * len(d['joints']) if start_q is None else
                    _vector(start_q, len(d['joints']), 'start configuration'))
    return d


# These files are the model supported by the compiled Panda kernels. Updating
# either requires checking the kernel geometry and collision exclusions too.
_PANDA_FILES = {'panda_spherized.urdf': 'bb56ebf99ea158d0e705bcdbed3d60460acde3eb53b76b7142eeadc847d67442',
                'panda.srdf': '1150719ea9d81139418198a50faea17e155323547d056c4edcb7ecc82fd8d317'}


def _generated_inputs(d):
    resources = resource_directory() / 'panda'
    for name, digest in _PANDA_FILES.items():
        if sha256((resources/name).read_bytes()).hexdigest() != digest:
            raise UnsupportedScene(f'Generated Panda model resource changed: {name}')
    if d['movable'] or d['tolerance'] != 0:
        raise UnsupportedScene('Generated scene requires static geometry and zero tolerance')
    if any(a.get('type') not in (None, 'goto') or a.get('side_effect') is not None for a in d['actions']):
        raise UnsupportedScene('Generated scene does not support attachment/contact transitions')
    names = d['frame_names']; bases = []; transforms = []
    for r in range(len(d['robot_names'])):
        try:
            pose = d['poses'][names.index(f'r{r}/panda_link0')]
        except ValueError as exc:
            raise UnsupportedScene('Scene is not the supported Panda frame/joint layout') from exc
        bases.append(dict(position=pose[:3], quaternion_xyzw=np.roll(pose[3:], -1).tolist()))
        matrix = np.eye(4)
        matrix[:3,:3] = _rotation(pose[3:])
        matrix[:3,3] = pose[:3]
        transforms.append(matrix)
    obstacles = []
    spheres = []
    cylinders = []
    for f, kind, size in zip(d['shape_frames'], d['kinds'], d['sizes']):
        if not names[f].startswith('obstacle'):
            continue
        pose = d['poses'][f]
        if kind == 0:
            obstacles.append(dict(type='sphere', center=pose[:3], radius=size[0]))
            spheres.append([*pose[:3], size[0]])
        elif kind == 3:
            axis = _rotation(pose[3:])[:,2].tolist()
            obstacles.append(dict(type='cylinder', center=pose[:3], axis=axis, radius=size[0], half_height=size[1]))
            cylinders.append([*pose[:3], *axis, size[0], size[1]])
        else:
            raise UnsupportedScene('Generated scene supports sphere/cylinder obstacles only')
    reference = import_robot_scene(resources/'panda_spherized.urdf', resources/'panda.srdf', bases, obstacles, start_q=d['start_q'])
    # Cylinder rotation around its axis and sphere rotation do not affect their
    # geometry. Keep the supplied rotations for the otherwise exact comparison.
    for i, name in enumerate(reference['frame_names']):
        if name.startswith('obstacle') and i < len(names) and names[i] == name:
            reference['poses'][i][3:] = d['poses'][i][3:]
    reference['robot_names'] = d['robot_names']
    reference['actions'] = d['actions']  # goto task metadata does not change geometry
    for key in reference:
        if key not in d or d[key] != reference[key]:
            raise UnsupportedScene(f'Generated Panda scene does not match supplied {key}')
    return (str(resources/'panda_spherized.urdf'), str(resources/'panda.srdf'),
            np.asarray(transforms), np.asarray(spheres).reshape(-1,4), np.asarray(cylinders).reshape(-1,8), 'vamp')


class Scene:
    """Common query API. Choose ``generic`` (default) or ``generated`` explicitly.

    The definition is copied at construction. Changing the source dictionary
    afterwards has no effect. Queries are bound directly to native methods;
    kernel selection and model matching happen once, never per query.
    """
    def __init__(self, definition, *, implementation='generic'):
        if implementation not in ('generic', 'generated', 'runtime', 'spheres', 'fcl'):
            raise ValueError('implementation must be generic, generated, runtime, spheres or fcl')
        d = deepcopy(definition)
        allowed = {'frame_names','parents','poses','has_joint','contacts','joints','joint_names',
                   'limits','frame_robot','robot_names','shape_frames','kinds','sizes','movable',
                   'actions','tolerance','start_q','excluded_pairs'}
        if set(d) - allowed:
            raise UnsupportedScene(f'Unknown scene fields: {set(d) - allowed}')
        if not np.isfinite(d['tolerance']) or d['tolerance'] < 0:
            raise ValueError('Invalid collision tolerance')
        if (not d['robot_names'] or any(not isinstance(n, str) for n in d['robot_names'])
                or len(set(d['robot_names'])) != len(d['robot_names'])):
            raise ValueError('Expected unique robot names')
        self._definition = d
        self.implementation = implementation
        self._inspection = None
        if implementation == 'generic':
            kernel = PrimitiveScene(d)
            self.transition = kernel.transition
            self.poses = kernel.poses
            self.info = kernel.info
        else:
            from ._collision import Scene as RobotScene
            args = _generated_inputs(d)
            kind = 'vamp' if implementation == 'generated' else implementation
            kernel = RobotScene(*args[:-1], kind)
            kernel.set_optimized(implementation in ('generated', 'runtime'))
        self._kernel = kernel
        for name in ('initial', 'state_valid', 'batch_valid', 'edge_valid', 'batch_edge_valid'):
            setattr(self, name, getattr(kernel, name))

    def set_optimized(self, enabled):
        if self.implementation != 'generated':
            raise UnsupportedScene('Optimization toggle is available for generated kernels only')
        self._kernel.set_optimized(bool(enabled))

    @property
    def definition(self):
        return deepcopy(self._definition)

    def transition(self, *args, **kwargs):
        raise UnsupportedScene('Generated scene does not support attachment/contact transitions')

    def _inspection_scene(self, mode):
        if mode is not None:
            raise ValueError('Generated scene expects its initial (None) mode')
        if self._inspection is None:
            self._inspection = PrimitiveScene(self._definition)
        return self._inspection

    def poses(self, q, mode):
        scene = self._inspection_scene(mode)
        return scene.poses(q, scene.initial())

    def info(self, mode):
        scene = self._inspection_scene(mode)
        return scene.info(scene.initial())
