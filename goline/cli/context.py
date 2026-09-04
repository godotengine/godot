"""Goline context packs (Stage 7 core, delivered for Stage 3).

Assemble grounded, scoped context for an external agent CLI about either the
Goline engine repo or a game project built with Goline, so the agent's work
is accurate rather than generic.

Dependency-free (stdlib). Subprocess use is limited and guarded (git, tool
detection) so the module stays testable and safe offline:
  - Every subprocess call is wrapped in a try/except and returns None on
    failure.
  - Nothing writes to the filesystem.
  - File scans are depth-bounded and entry-count-bounded to stay fast.
"""

from __future__ import annotations

import os
import shutil
import subprocess

# ---------------------------------------------------------------------------
# Small, guarded subprocess helper
# ---------------------------------------------------------------------------


def _run(argv: list[str], cwd: str, timeout: int = 5) -> str | None:
    """Run a read-only command and return trimmed stdout, or None on any
    failure. Never raises."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (proc.stdout or "").strip() or None


def _git(cwd: str, *args: str) -> str | None:
    return _run(["git", *args], cwd)


def _which(name: str) -> str | None:
    return shutil.which(name)


# ---------------------------------------------------------------------------
# Engine context
# ---------------------------------------------------------------------------

_ENGINE_MARKERS = ("version.py", "SConstruct", "godot")


def _first_existing(root: str, names: tuple[str, ...]) -> str | None:
    for n in names:
        if os.path.exists(os.path.join(root, n)):
            return n
    return None


def detect_engine_root(cwd: str | None = None) -> str | None:
    """Walk up from cwd to find a Goline/Godot engine repo root."""
    cur = os.path.abspath(cwd or os.getcwd())
    while True:
        if _first_existing(cur, _ENGINE_MARKERS):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _list_names(path: str) -> list[str]:
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def build_engine_context(root: str | None = None, use_git: bool = True) -> str:
    """Produce a plain-text context pack for the Goline engine repo."""
    root = root or detect_engine_root() or os.getcwd()
    lines: list[str] = []
    lines.append("# Goline engine context")
    lines.append(f"repo_root: {root}")

    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD") if use_git else None
    commit = (_git(root, "rev-parse", "--short", "HEAD") if use_git else None)
    clean = (_git(root, "status", "--porcelain") if use_git else None)
    if branch:
        lines.append(f"git_branch: {branch}")
    if commit:
        lines.append(f"git_commit: {commit}")
    lines.append(f"git_clean: {'yes' if clean is None else 'no'}")

    present = [m for m in _ENGINE_MARKERS if os.path.exists(os.path.join(root, m))]
    lines.append(f"engine_markers: {', '.join(present) or 'none'}")

    # Toolchain detection (best-effort, non-fatal).
    tools = []
    if _which("scons"):
        tools.append("scons")
    if _which("python"):
        tools.append("python")
    if _which("cl") or _which("g++"):
        tools.append("c++-compiler")
    if not tools:
        tools.append("none-detected")
    lines.append("tools_on_path: " + ", ".join(tools))

    # Key directories.
    present_dirs = []
    for d in ("core", "scene", "editor", "modules", "platform", "goline", "docs"):
        if os.path.isdir(os.path.join(root, d)):
            present_dirs.append(d)
    lines.append("present_dirs: " + ", ".join(present_dirs))

    # Version info (fast, offline).
    ver = _read_simple(root, "version.py")
    if ver:
        lines.append("version.py_present: yes")
        ident = []
        for key in ("name", "short_name", "major", "minor", "patch"):
            val = _scan_cfg(ver, key)
            if val is not None:
                ident.append(f"{key}={val}")
        if ident:
            lines.append("identity: " + ", ".join(ident))
    else:
        lines.append("version.py_present: no")

    # Goline handoff documents.
    goline_docs = [f for f in ("GOLINE.md",) if os.path.exists(os.path.join(root, f))]
    docs = _list_names(os.path.join(root, "docs")) if os.path.isdir(os.path.join(root, "docs")) else []
    lines.append("goline_handoff: " + ", ".join(goline_docs))
    lines.append("docs_entries: " + (", ".join(docs[:20]) if docs else "none"))

    # Build/run guidance (static map; the agent reads the files for details).
    lines.append("")
    lines.append("## Guidance")
    lines.append(
        "This is a Godot Engine 4 fork named Goline. Build/test commands and "
        "project layout are documented in AGENTS.md / GOLINE.md at the repo "
        "root. Forward new features to opencode/claude with this context. "
        "Do not modify upstream engine internals unless a roadmap stage "
        "authorizes it."
    )

    # Security: the agent must "hold" the Goline permission policy.
    from goline.cli.policy import guidance_notice
    lines.append("")
    lines.append(guidance_notice())
    return "\n".join(lines)


def _read_simple(path: str, name: str) -> str | None:
    try:
        with open(os.path.join(path, name), "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Game project context
# ---------------------------------------------------------------------------

# Extensions we surface for a game project, bounded to avoid flooding the pack.
_INTERESTING_EXT = {".gd", ".cs", ".tscn", ".tres"}
_MAX_GAME_FILES = 60
_MAX_SCAN_DEPTH = 4


def _is_project_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "project.godot"))


def build_game_context(project: str | None = None, depth: int = _MAX_SCAN_DEPTH) -> str:
    """Produce a plain-text context pack for a Goline game project.

    `project` is a directory containing project.godot (defaults to cwd).
    Returns a message instead of a pack if no project is found.
    """
    root = os.path.abspath(project or os.getcwd())
    if not _is_project_dir(root):
        return (
            f"No game project found at {root} (expected project.godot). "
            "Pass --project <dir containing project.godot>"
        )

    lines: list[str] = []
    lines.append("# Goline game project context")
    lines.append(f"project_root: {root}")

    cfg = _read_simple(root, "project.godot")
    if cfg:
        name = _scan_cfg(cfg, "config/name")
        main = _scan_cfg(cfg, "run/main_scene")
        config_version = _scan_cfg(cfg, "config_version")
        if name:
            lines.append(f"project_name: {name}")
        if main:
            lines.append(f"main_scene: {main}")
        if config_version:
            lines.append(f"config_version: {config_version}")
    else:
        lines.append("project.godot: unreadable")

    # Scoped file listing (bounded).
    interesting: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        rel_depth = dirpath[len(root):].count(os.sep)
        if rel_depth > depth:
            continue
        for fn in filenames:
            if os.path.splitext(fn)[1] in _INTERESTING_EXT:
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                interesting.append(rel.replace(os.sep, "/"))
                if len(interesting) >= _MAX_GAME_FILES:
                    break
        if len(interesting) >= _MAX_GAME_FILES:
            break
    interesting.sort()
    lines.append(f"files ({len(interesting)} shown): " + ", ".join(interesting))

    lines.append("")
    lines.append("## Guidance")
    lines.append(
        "This is a game project for Goline (a Godot 4 fork). Help with "
        "GDScript/C# scenes and scripts grounded in the files above. "
        "Main scene is the run entry point unless stated otherwise."
    )

    # Security: the agent must "hold" the Goline permission policy.
    from goline.cli.policy import guidance_notice
    lines.append("")
    lines.append(guidance_notice())
    return "\n".join(lines)


def _scan_cfg(text: str, key: str) -> str | None:
    """Pull `key=value` (or `key = value`) from a config-style line."""
    prefix = key + "="
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(key) and "=" in s:
            left, _, right = s.partition("=")
            if left.strip() == key:
                val = right.strip().strip('"')
                return val or None
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_context(kind: str, project: str | None = None, use_git: bool = True) -> str:
    """Dispatch to the right context pack. kind in {'engine', 'game'}."""
    kind = (kind or "").lower()
    if kind == "engine":
        return build_engine_context(use_git=use_git)
    if kind == "game":
        return build_game_context(project)
    raise ValueError(f"unknown context kind '{kind}' (expected 'engine' or 'game')")
