from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_project_python(required_modules: tuple[str, ...] = ("numpy", "mujoco")) -> None:
    """Re-exec under the local project interpreter when core deps are missing."""
    missing = []
    for module_name in required_modules:
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)

    if not missing:
        return

    project_python = Path(__file__).resolve().parent / ".mambaenv" / "bin" / "python"
    if not project_python.exists():
        missing_list = ", ".join(missing)
        raise ModuleNotFoundError(
            f"Missing Python module(s): {missing_list}. "
            "Install dependencies or run with the project interpreter."
        )

    os.execv(str(project_python), [str(project_python), *sys.argv])


def confirm_real_robot_start(prompt: str) -> None:
    """Require explicit confirmation for real-robot runs unless overridden."""
    if os.environ.get("UNITREE_SKIP_CONFIRM") == "1":
        return

    if not sys.stdin.isatty():
        raise RuntimeError(
            f"{prompt} Run from an interactive terminal, pass `--sim`, "
            "or set UNITREE_SKIP_CONFIRM=1 if you intentionally want to bypass the prompt."
        )

    input(prompt)
