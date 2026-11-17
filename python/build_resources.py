"""Extract the upstream embedded task data for the installed Python package."""
import json
from pathlib import Path
import re
import sys

source, destination = map(Path, sys.argv[1:])
destination.mkdir(parents=True, exist_ok=True)
for family in ("flat", "cage"):
    text = (source / "apps" / f"panda_{family}_builtin_tasks.hpp").read_text()
    pattern = r'kTasks(\d+)\[\]\s*=\s*R"([^ (]+)\((.*?)\)\2";'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        raise ValueError(f"No built-in {family} tasks found")
    for count, _, data in matches:
        (destination / f"panda_{family}_{count}.json").write_text(
            json.dumps(json.loads(data)) + "\n")
