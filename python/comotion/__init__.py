"""Collision queries for URDF/generated robot models."""
from .resources import resource_directory, load_builtin

__all__ = ["RobotScene", "resource_directory", "load_builtin"]

def __getattr__(name):
    if name == "RobotScene":
        from ._collision import Scene
        return Scene
    raise AttributeError(name)
