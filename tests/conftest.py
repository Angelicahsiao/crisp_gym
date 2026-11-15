"""Shared pytest setup: every test file gets its own fake modules, and tests that
need lerobot or ROS skip themselves where those are missing.

WHY THIS FILE EXISTS
--------------------
Most test files here were written to run alone (`python tests/test_x.py`) and
make that possible by putting FAKE modules into `sys.modules` — a hand-made
`torch`, `rclpy`, `lerobot`, `crisp_py` — before importing the code under test.
Run alone, that is fine. Run together in one pytest process, the fakes of one
file were still there when the next file was imported:

    test_pose_math installs its fake torch
    test_action_swap sees "torch is already there", skips its own fake, and
    fails against the wrong one

so tests passed or failed depending on which files ran before them. Nothing in
the test files had to be wrong for that to happen.

WHAT IT DOES
------------
1. ISOLATION. Around the import of each test file (collection) and again around
   its tests (run), `sys.modules` is compared with what it was before. Modules
   the file brought in that belong to it are taken out again afterwards and put
   back only while that file's tests run:
     * fakes: modules that were not loaded from anywhere (no spec, no file);
     * project modules (crisp_gym/, scripts/, examples/): they were imported
       while the fakes were in place and hold references to them, so the next
       file must import its own fresh copy.
   Real third-party modules (numpy, pandas, a real torch ...) are never touched:
   importing those twice in one process is not safe.
   The test files themselves are unchanged and still run as scripts.

2. MARKERS. `lerobot` and `ros` (see MARKER_NEEDS). A marked test is skipped
   with a reason when its packages are not installed, instead of failing on the
   import. Mark a test with `@pytest.mark.lerobot`, or a whole file by listing
   it in FILE_MARKERS below (which leaves the file runnable as a script).

3. `tmp_dir`: the fixture some script-style tests expect as an argument.

Set CRISP_TEST_NO_ISOLATION=1 to switch part 1 off (to rule it out when a test
behaves differently under pytest than as a script).
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROJECT_DIRS = (REPO / "crisp_gym", REPO / "scripts", REPO / "examples")

# marker -> packages that must be REALLY installed for the test to run
MARKER_NEEDS = {
    "lerobot": ("lerobot", "torch"),
    "ros": ("rclpy", "crisp_py"),
}

# whole files that need a marker (the file stays free of a pytest import)
FILE_MARKERS = {
    "test_lerobot_record.py": ("lerobot",),
}

# Modules that are created by C runtimes rather than loaded from a file, and so
# look like hand-made fakes. They are real and shared: never take them out.
_NOT_A_FAKE = re.compile(r"^(cython_runtime|_cython_\w+|__main__|__mp_main__)(\.|$)")

_ISOLATE = os.environ.get("CRISP_TEST_NO_ISOLATION", "") in ("", "0")
_STASH = pytest.StashKey[dict]()


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# Asked ONCE, here, before any test file is imported — afterwards a fake of the
# same name may sit in sys.modules and answer for the real package.
_AVAILABLE = {name: _installed(name) for needs in MARKER_NEEDS.values() for name in needs}


# ── 1. isolation ─────────────────────────────────────────────────────────────

def _is_real(module: object) -> bool:
    """Loaded from somewhere (has a spec or a file), as opposed to hand-made."""
    return isinstance(module, types.ModuleType) and (
        getattr(module, "__spec__", None) is not None
        or bool(getattr(module, "__file__", None))
    )


def _is_project(module: object) -> bool:
    """Loaded from crisp_gym/, scripts/ or examples/ of this repository.

    A package without an __init__.py (crisp_gym.util, crisp_gym.record ...) has
    no __file__; its directories are in __path__ instead.
    """
    file = getattr(module, "__file__", None)
    places = [file] if isinstance(file, str) else []
    if not places:
        try:
            places = [str(p) for p in getattr(module, "__path__", ())]
        except Exception:  # noqa: BLE001 - a namespace path whose parent is gone
            return False
    for place in places:
        try:
            path = Path(place).resolve()
        except OSError:
            continue
        if any(d == path or d in path.parents for d in PROJECT_DIRS):
            return True
    return False


def _belongs_to_a_test_file(name: str, module: object, now: dict) -> bool:
    """Is this sys.modules entry something one test file set up for itself?"""
    if _NOT_A_FAKE.match(name):
        return False
    top = now.get(name.partition(".")[0])
    if _is_project(module) or (top is not None and _is_project(top)):
        return True                       # a whole project package goes together
    if _is_real(module):
        return False                      # real third-party: shared by everyone
    # A module that was not loaded from anywhere. It is a test's fake unless it
    # sits under a real third-party package, where the library made it itself
    # (torch._C._nn, cv2.gapi, six.moves, pyexpat.errors ...).
    return top is None or top is module or not _is_real(top)


def _take_back(before: dict) -> dict:
    """Return sys.modules to `before` as far as test-owned entries go.

    Returns what was taken out, so it can be put back for the file's own tests.
    """
    now = dict(sys.modules)
    # decide everything first: a package's path cannot be read once its parent
    # has been taken out
    taken = {name: module for name, module in now.items()
             if before.get(name) is not module and _belongs_to_a_test_file(name, module, now)}
    for name in taken:
        if name in before:
            sys.modules[name] = before[name]
        else:
            del sys.modules[name]
    for name, module in before.items():   # a real module a test removed
        if name not in sys.modules and not _belongs_to_a_test_file(name, module, before):
            sys.modules[name] = module
    return taken


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector: pytest.Collector):  # noqa: ANN201
    """Importing a test file must leave no fakes behind for the next file."""
    if not (_ISOLATE and isinstance(collector, pytest.Module)):
        yield
        return
    before = dict(sys.modules)
    try:
        yield
    finally:
        collector.stash[_STASH] = _take_back(before)


@pytest.fixture(scope="module", autouse=True)
def _own_modules(request: pytest.FixtureRequest):  # noqa: ANN202
    """While a file's tests run, its fakes and project modules are in place."""
    if not _ISOLATE:
        yield
        return
    before = dict(sys.modules)
    sys.modules.update(request.node.stash.get(_STASH, {}))
    try:
        yield
    finally:
        _take_back(before)


# ── 2. markers ───────────────────────────────────────────────────────────────

def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply FILE_MARKERS and skip marked tests whose packages are missing."""
    for item in items:
        for marker in FILE_MARKERS.get(Path(str(item.fspath)).name, ()):
            item.add_marker(getattr(pytest.mark, marker))
        for marker, needs in MARKER_NEEDS.items():
            missing = [name for name in needs if not _AVAILABLE[name]]
            if missing and item.get_closest_marker(marker):
                item.add_marker(pytest.mark.skip(
                    reason=f"marked '{marker}': needs {', '.join(missing)} (not installed)"))


# ── 3. fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """A fresh directory, under the name the script-style tests use."""
    return tmp_path
