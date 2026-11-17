# Python collision queries

Install a C++17 compiler, CMake, Eigen3 and FCL development packages.
Clone this repository recursively and run `python -m pip install .`.
The package builds collision checking without the CoMotion planners, using one
compilation job and native CPU SIMD flags by default. Rebuild on another CPU.

`comotion.RobotScene` accepts a URDF, SRDF, robot base transforms and world
obstacles. Scalar and batched validity queries share the same model.
`load_builtin` and `resource_directory` expose the installed Panda task data.
Scenes are mutable and must not be queried concurrently.
