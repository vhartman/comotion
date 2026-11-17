# Python collision queries

This optional package exposes collision checking without building CoMotion's planners.
It provides `RobotScene` for URDF/generated robot models and `PrimitiveScene` for
articulated sphere, box, capsule and cylinder scenes with rigid attachments.
The primitive scene is an additional runtime backend, not an implementation of
CoMotion's planning algorithms. Each scene is mutable and must not be queried
concurrently from multiple threads.

## Install

Install a C++17 compiler, CMake, Eigen3, libccd, and FCL development packages.
Clone this repository recursively, then run `python -m pip install .` at its root.
Compilation uses one job by default. No extension is written into a downstream
project and no simulator is required at runtime.

The default build targets the local CPU. Rebuild on a different CPU, or choose
explicit SIMD flags with `-Ccmake.define.VAMP_ARCH="-mavx2 -mfma"` on compatible
x86 machines. For primitive scenes only, pass
`-Ccmake.define.COMOTION_PYTHON_ROBOT_SCENES=OFF` to omit FCL robot bindings.
The normal C++ build is unchanged unless `COMOTION_BUILD_PYTHON=ON` is selected.

```python
from comotion import PrimitiveScene, RobotScene, load_builtin, resource_directory
```

Primitive scenes accept a dictionary of frame parents, local poses (`xyz,wxyz`),
ordered hinge/slide joints, joint bounds, robot ownership and collision shapes.
Joint kinds are 0 (hinge) and 1 (slide); shape kinds are 0 (sphere), 1 (capsule),
2 (box) and 3 (cylinder). Box dimensions are half extents; capsules/cylinders use
radius and half axial length. Unsupported geometry must be converted explicitly.

`initial()` returns an immutable mode handle. `transition()` creates a new handle
with a rigid attachment/contact change; it does not mutate the old handle.
`state_valid()`, `batch_valid()` and `edge_valid()` share geometry and tolerance
semantics. Batches return a Boolean mask in input order. Positive tolerances sum
penetration depths; different convex solvers can disagree near a nonzero
penetration threshold even with identical geometry. An edge checks a finite interpolation grid, not continuous
collision freedom. Its returned last sample is an inspection cursor, not a
certified last-valid configuration.

The Python task loader ships the upstream Panda cage/flat definitions; it does
not download resources or require an upstream checkout at runtime.

## Minimal primitive query

```python
import numpy as np
from comotion import PrimitiveScene

# A translating sphere and a fixed sphere, both of radius 0.1.
scene = PrimitiveScene(dict(
    frame_names=["slider", "obstacle"], parents=[-1, -1],
    poses=[[0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0]],
    has_joint=[1, 0], contacts=[1, 1],
    joints=[[0, 1, 1, 0, 0]], limits=[[-2, 2]], start_q=[0],
    frame_robot={"slider": 0}, movable=[],
    shape_frames=[0, 1], kinds=[0, 0], sizes=[[0.1, 0, 0], [0.1, 0, 0]],
))
mode = scene.initial()
assert scene.state_valid(np.array([0.]), mode)
assert not scene.state_valid(np.array([1.]), mode)
mask = scene.batch_valid(np.array([[0.], [1.]]), mode)
assert mask.tolist() == [True, False]
```

Frame parents use `-1` for world roots. Joint records contain frame index, joint
kind, and a unit axis; their order is the configuration-vector order. Collision
contacts and rigid-link exclusions follow the exported scene definition. An
optional `excluded_pairs` list contains pairs of **shape indices**.
