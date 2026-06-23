#!/usr/bin/env python3
"""Build a Kaggle-ready Orbit Wars submission tarball for the DDPM policy.

Bundles ``ddpm_act.py`` and its import closure plus a trained checkpoint into
``submission.tar.gz`` with a generated top-level ``main.py`` whose last ``def``
(``agent``) is the Kaggle entrypoint. Torch/pandas/yaml are preinstalled on the
Kaggle simulation image, so no requirements/vendored wheels are needed.

Usage:
    python pack.py                          # build submission.tar.gz
    python pack.py --smoke                  # build + run an in-process smoke test
    python pack.py --checkpoint checkpoints/latest.pt --out my_sub.tar.gz
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BUILD_DIR = REPO_ROOT / "build" / "submission"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "step_1800500.pt"
DEFAULT_OUT = REPO_ROOT / "submission.tar.gz"

# Import closure of ddpm_act, copied with their layout preserved so that
# ``from src.methods.ddpm import DDPM`` keeps resolving inside the bundle.
CLOSURE_FILES = (
    "ddpm_act.py",
    "train_orbit.py",
    "src/methods/__init__.py",
    "src/methods/base.py",
    "src/methods/ddpm.py",
)

# Generated at bundle root; last def (agent) is the Kaggle entrypoint. Force CPU
# because the eval image has no GPU (make_ddpm_agent would otherwise pick cuda).
#
# NOTE: Kaggle loads this file via `env = {}; exec(code_object, env)` (see
# kaggle_environments.agent.get_last_callable), so `__file__` is NOT defined and
# `Path(__file__)` raises NameError. Anchor on the imported `ddpm_act` module
# instead — it lives next to checkpoint.pt and its __file__ is set normally
# because it's imported (not exec'd).
MAIN_PY = '''\
from pathlib import Path

import ddpm_act
from ddpm_act import make_ddpm_agent

_AGENT = make_ddpm_agent(
    Path(ddpm_act.__file__).resolve().parent / "checkpoint.pt",
    device="cpu",
)


def agent(obs):
    return _AGENT(obs)
'''


def _copy_closure(build_dir: Path) -> None:
    for rel in CLOSURE_FILES:
        src = REPO_ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(f"required closure file missing: {src}")
        dst = build_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    # `src` has no __init__.py in the repo (works as a namespace package); add an
    # empty one in the bundle so the import is robust regardless of cwd handling.
    (build_dir / "src" / "__init__.py").touch()


def build(checkpoint: Path, out: Path) -> Path:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    _copy_closure(BUILD_DIR)
    shutil.copy2(checkpoint, BUILD_DIR / "checkpoint.pt")
    (BUILD_DIR / "main.py").write_text(MAIN_PY)

    out = out.resolve()
    if out.exists():
        out.unlink()
    # Root-relative entries (./main.py, ./ddpm_act.py, ./src/...) so Kaggle finds
    # main.py at the archive root rather than nested under submission/.
    subprocess.run(
        ["tar", "-czf", str(out), "-C", str(BUILD_DIR), "."],
        check=True,
    )
    size_mb = out.stat().st_size / 1e6
    print(f"[pack] wrote {out} ({size_mb:.1f} MB)")
    print(f"[pack] bundle staged at {BUILD_DIR}")
    return out


SMOKE_PROGRAM = textwrap.dedent(
    """
    import os, sys, time
    from pathlib import Path

    # Reproduce EXACTLY how Kaggle loads the agent: compile + exec into an empty
    # namespace with NO __file__, and put the agent dir on sys.path. `import main`
    # would mask __file__ bugs because import machinery sets __file__.
    here = os.getcwd()
    sys.path.append(here)
    src = Path("main.py").read_text()
    env = {}
    exec(compile(src, os.path.join(here, "main.py"), "exec"), env)
    agent = [v for v in env.values() if callable(v)][-1]

    # Minimal 2-player observation: planet 0 is ours (owner 0, ships -> source),
    # planet 1 is the opponent. Row layout = (id, owner, x, y, radius, ships, production).
    planets = [
        [0, 0, 20.0, 20.0, 1.0, 25, 1.0],
        [1, 1, 80.0, 80.0, 1.0, 25, 1.0],
    ]
    obs = {
        "player": 0,
        "step": 0,
        "angular_velocity": 0.01,
        "initial_planets": planets,
        "planets": planets,
        "fleets": [],
        "comet_planet_ids": [],
    }

    t0 = time.perf_counter()
    moves = agent(obs)
    dt = time.perf_counter() - t0

    assert isinstance(moves, list), f"agent must return a list, got {type(moves)}"
    print(f"[smoke] OK: agent returned {len(moves)} move(s) in {dt * 1000:.0f} ms")
    print(f"[smoke] sample moves: {moves[:3]}")
    print(f"[smoke] per-turn latency ~ {dt:.2f}s (CPU; watch the Kaggle turn budget)")
    """
).strip()


def smoke(build_dir: Path) -> None:
    """Run the packed bundle in a fresh subprocess with cwd = bundle dir.

    Mimics Kaggle's flat working directory, catching missing-file, import-path,
    and torch.load failures before a real submission is spent.
    """
    print("[smoke] importing bundle and running one turn...")
    result = subprocess.run(
        [sys.executable, "-c", SMOKE_PROGRAM],
        cwd=str(build_dir),
    )
    if result.returncode != 0:
        raise SystemExit(f"[smoke] FAILED (exit {result.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"trained .pt checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output tarball (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="after building, import the bundle and run one synthetic turn",
    )
    ap.add_argument(
        "--keep-build",
        action="store_true",
        help="keep build/submission/ after packing (default: keep it anyway for --smoke)",
    )
    args = ap.parse_args()

    build(args.checkpoint.resolve(), args.out)

    if args.smoke:
        smoke(BUILD_DIR)

    if not args.smoke and not args.keep_build:
        # Build dir is harmless to keep, but tidy up when nothing else needs it.
        shutil.rmtree(BUILD_DIR, ignore_errors=True)

    print("\nSubmit with:")
    print(
        f"  kaggle competitions submit -c orbit-wars -f {args.out} "
        f'-m "ddpm {args.checkpoint.stem}"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
