"""Installed robot descriptions and frozen upstream Panda task definitions."""
import json
from pathlib import Path

def resource_directory():
    return Path(__file__).resolve().parent / "resources"

def load_builtin(family, robots):
    if family not in ("flat", "cage") or isinstance(robots, bool) or not isinstance(robots, int):
        raise ValueError("Expected flat/cage and an integer robot count")
    path = resource_directory() / "tasks" / f"panda_{family}_{robots}.json"
    if not path.is_file():
        raise ValueError(f"No built-in Panda {family} scene for {robots} robots")
    return json.loads(path.read_text())
