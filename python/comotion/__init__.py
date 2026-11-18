"""Collision queries without importing a planner or simulator."""
from ._primitive import Scene as PrimitiveScene, ModeHandle, simd_width
from .scene import Scene, UnsupportedScene, import_robot_scene
from .resources import resource_directory, load_builtin

__all__ = ["Scene", "UnsupportedScene", "import_robot_scene", "PrimitiveScene", "ModeHandle", "RobotScene", "simd_width",
           "resource_directory", "load_builtin"]

def __getattr__(name):
    if name == "RobotScene":
        from ._collision import Scene
        return Scene
    raise AttributeError(name)
