#!/usr/bin/env python3
"""Phase 0 — verify/build the Python environment the DMMM pipeline (phases 1-4) runs in.

Idempotent and OS-agnostic (Windows/macOS/Linux). Run it with any system Python
(python3 / python / py); it only uses the standard library, like report.py:

  1. Check the running interpreter meets the minimum version.
  2. If <skill_dir>/.venv exists and all required imports pass -> done (fast path).
  3. Otherwise create the venv (stdlib `venv`) and install dependencies — from
     requirements.lock (pinned, demo-verified versions) when present, falling back
     to requirements.txt (unpinned) if the pinned install fails (e.g. no wheels for
     this Python) — then re-verify the imports.

Prints a JSON summary to stdout for the caller to report on (progress goes to
stderr). Later phases must use the "python" path from that JSON.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import venv
from pathlib import Path

MIN_PYTHON = (3, 9)
# Import names for everything in requirements.txt (sklearn = scikit-learn).
REQUIRED_IMPORTS = ["pandas", "numpy", "xgboost", "optuna", "sklearn", "cmaes"]


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fail(stage: str, error: str, hint: str = "") -> None:
    summary = {"status": "error", "stage": stage, "error": error}
    if hint:
        summary["hint"] = hint
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    raise SystemExit(1)


def verify_imports(python: Path) -> str:
    """Return '' if all required imports work, else the import error text."""
    proc = subprocess.run(
        [str(python), "-c", "import " + ", ".join(REQUIRED_IMPORTS)],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return ""
    return (proc.stderr or proc.stdout).strip().splitlines()[-1] if (proc.stderr or proc.stdout) else "import check failed"


def probe_version(python: Path) -> str:
    proc = subprocess.run(
        [str(python), "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"],
        capture_output=True, text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description="DMMM phase 0 — environment check/setup")
    p.add_argument("--skill-dir", default=None,
                   help="Skill root holding requirements.txt (default: parent of this script)")
    p.add_argument("--force", action="store_true",
                   help="Recreate the venv from scratch even if it exists")
    args = p.parse_args()

    t0 = time.time()
    if sys.version_info < MIN_PYTHON:
        fail(
            "python",
            f"Python {sys.version.split()[0]} is too old (need >= {'.'.join(map(str, MIN_PYTHON))})",
            hint="Install a recent Python 3 (python.org, brew, or the Microsoft Store) and rerun.",
        )

    skill_dir = Path(args.skill_dir).resolve() if args.skill_dir else Path(__file__).resolve().parent.parent
    requirements = skill_dir / "requirements.txt"
    lock = skill_dir / "requirements.lock"
    if not requirements.is_file():
        fail("skill-dir", f"requirements.txt not found in {skill_dir}",
             hint="Pass the skill root explicitly with --skill-dir.")
    # Prefer the pinned, demo-verified versions; keep the unpinned list as fallback.
    req_candidates = [lock, requirements] if lock.is_file() else [requirements]

    venv_dir = skill_dir / ".venv"
    py = venv_python(venv_dir)
    created = False
    installed = False
    warnings: list[str] = []

    # Fast path: existing venv with all dependencies -> nothing to do.
    if py.is_file() and not args.force:
        log(f"checking existing venv: {venv_dir}")
        err = verify_imports(py)
        if not err:
            json.dump({
                "status": "ok",
                "python": str(py),
                "python_version": probe_version(py),
                "created": False,
                "installed": False,
                "requirements": None,
                "elapsed_sec": round(time.time() - t0, 1),
                "warnings": warnings,
            }, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
            return
        warnings.append(f"existing venv missing dependencies ({err}); installing")
        log(warnings[-1])
    else:
        if venv_dir.exists():
            log(f"rebuilding venv ({'--force' if args.force else 'python executable missing'}): {venv_dir}")
        else:
            log(f"creating venv: {venv_dir}")
        try:
            venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
        except Exception as e:  # noqa: BLE001 - surface a clean message to the caller
            fail("venv", f"could not create venv at {venv_dir}: {e}")
        created = True

    requirements_used = None
    for req in req_candidates:
        log(f"installing dependencies from {req} (needs network access to PyPI)")
        proc = subprocess.run(
            [str(py), "-m", "pip", "install", "-r", str(req)],
            stdout=sys.stderr, stderr=sys.stderr,
        )
        if proc.returncode == 0:
            requirements_used = req.name
            break
        if req is not req_candidates[-1]:
            warnings.append(
                f"pinned install from {req.name} failed (exit {proc.returncode}); "
                "retrying with unpinned requirements.txt"
            )
            log(warnings[-1])
    if requirements_used is None:
        fail("pip", f"pip install failed for {[r.name for r in req_candidates]}",
             hint="Check the log above; a network/proxy issue is the usual cause.")
    installed = True

    err = verify_imports(py)
    if err:
        fail("verify", f"dependencies still missing after install: {err}")

    json.dump({
        "status": "ok",
        "python": str(py),
        "python_version": probe_version(py),
        "created": created,
        "installed": installed,
        "requirements": requirements_used,
        "elapsed_sec": round(time.time() - t0, 1),
        "warnings": warnings,
    }, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
