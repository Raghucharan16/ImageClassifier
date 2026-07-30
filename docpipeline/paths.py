"""
Resolve bundled resource paths whether running from source or as a frozen
PyInstaller .exe.

When frozen, PyInstaller unpacks bundled data under sys._MEIPASS. Model files
are looked up there first, then next to the executable, then beside this module,
then in the current working directory (dev mode). Output paths are NOT routed
through here -- they stay relative to the cwd so results land where the user runs
the .exe.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def resource_path(rel_path: str) -> str:
    candidates = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(os.path.join(sys._MEIPASS, rel_path))
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(os.path.dirname(sys.executable), rel_path))
    # Beside this module: models/ ships inside the package folder, so this is
    # what resolves them when the pipeline is launched from any working
    # directory (main.py is normally run from the repo root, not docpipeline/).
    candidates.append(os.path.join(_HERE, rel_path))
    candidates.append(os.path.abspath(rel_path))
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[-1]
