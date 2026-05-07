"""Locate a local unitree_sdk2_python checkout when it is not installed."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def ensure_unitree_sdk2py() -> None:
    """Make the local Unitree SDK importable before importing unitree_sdk2py."""
    try:
        import unitree_sdk2py  # noqa: F401
        return
    except ModuleNotFoundError as exc:
        if exc.name != "unitree_sdk2py":
            raise

    here = Path(__file__).resolve()
    candidates = []

    env_path = os.environ.get("UNITREE_SDK2_PYTHON")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            Path.home() / "unitree_sdk2_python",
            here.parents[1].parent / "unitree_sdk2_python",
        ]
    )

    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for root in candidates:
        if not root.exists():
            continue

        paths = [
            root,
            root / ".venv" / "lib" / py_version / "site-packages",
        ]
        for path in paths:
            if path.exists():
                site.addsitedir(str(path))

        try:
            import unitree_sdk2py  # noqa: F401
            return
        except ModuleNotFoundError:
            continue

    searched = ", ".join(str(path) for path in candidates)
    raise ModuleNotFoundError(
        "No module named 'unitree_sdk2py'. Install unitree_sdk2_python with "
        "`pip3 install -e /path/to/unitree_sdk2_python`, or set "
        f"UNITREE_SDK2_PYTHON to its checkout path. Searched: {searched}"
    )
