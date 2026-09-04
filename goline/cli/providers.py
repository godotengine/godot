"""Goline provider drivers (parsed from T3 Code's ProviderDriver SPI).

Option A port: adopt T3 Code's clean provider-abstraction ideas into a
dependency-free Python layer. T3 Code (pingdotgg/t3code) wraps coding-agent
CLIs behind a ProviderDriver -> ProviderInstance (snapshot / adapter /
events) model and a normalized event stream. We keep the *shape* but drop the
Effect/Node stack, emitting a small discriminated-union event vocabulary and
dispatching via each agent's native headless CLI mode:

  - OpenCode: `opencode run --format json --file <ctx> --model <m> <prompt>`
  - Claude:   `claude -p <prompt>` (print/headless mode)

This is deliberately one-shot (dispatch a prompt, stream events) rather than
a long-lived orchestration server — appropriate for our lightweight CLI.

Pure + offline-testable: every concrete driver takes an injectable executor
so tests never spawn a process.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

# ---------------------------------------------------------------------------
# Normalized provider event vocabulary (mirrors T3's ProviderRuntimeEventV2)
# ---------------------------------------------------------------------------


@dataclass
class ProviderEvent:
    """A normalized, provider-agnostic event from an agent session."""

    kind: str            # one of: content.delta, message.start, request.opened,
                         # task.started, task.completed, permission, error, done
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, **self.data}


class _NOOP:
    pass


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class DispatchResult:
    """Outcome of dispatching one prompt to a provider."""

    provider: str
    model: str
    events: List[ProviderEvent] = field(default_factory=list)
    exit_code: int = 0
    error: Optional[str] = None

    @property
    def text(self) -> str:
        """Concatenated assistant content deltas."""
        parts = []
        for e in self.events:
            if e.kind == "content.delta" and e.data.get("text"):
                parts.append(e.data["text"])
        return "".join(parts)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "events": [e.to_dict() for e in self.events],
            "exit_code": self.exit_code,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# ProviderDriver SPI (Protocol): what each external agent adapter must do
# ---------------------------------------------------------------------------


class ProviderDriver(Protocol):
    """Common interface every provider adapter implements.

    Mirrors T3 Code's ProviderDriver, reduced to what a one-shot CLI needs.
    """

    driver_kind: str
    display_name: str

    def dispatch(
        self,
        prompt: str,
        context_file: Optional[str],
        *,
        model: Optional[str] = None,
        workdir: Optional[str] = None,
        executor: Optional[Callable[[List[str], str], "subprocess_result_proto"]]
        = None,
    ) -> DispatchResult:
        """Send a prompt to the agent and return normalized events.

        `executor` is an injectable `(argv, cwd) -> CompletedProcess` for
        tests; default is a real subprocess runner.
        """
        ...


# A minimal structural type for subprocess results (duck-typed).
class subprocess_result_proto(Protocol):
    returncode: int
    stdout: str
    stderr: str


def _executable_argv(command: str, args: List[str]) -> List[str]:
    """Build an argv that actually executes `command` across platforms.

    Some CLIs (opencode, claude) resolve to PowerShell/Cmd script wrappers on
    Windows (`*.ps1`, `*.cmd`, `*.bat`) that CreateProcess cannot run
    directly. Route through the appropriate interpreter when that is the case;
    otherwise use the command as-is. Mirrors goline_cli._command_argv().
    """
    import shutil

    path = shutil.which(command)
    if path is None:
        return [command] + args
    lower = path.lower()
    if lower.endswith(".ps1"):
        return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path] + args
    if lower.endswith((".cmd", ".bat")):
        return ["cmd", "/d", "/c", path] + args
    return [command] + args


def _default_executor(argv: List[str], cwd: str) -> subprocess_result_proto:
    import subprocess

    return subprocess.run(argv, capture_output=True, text=True, cwd=cwd, timeout=600)


# ---------------------------------------------------------------------------
# Event parsing helpers
# ---------------------------------------------------------------------------


def _parse_opencode_events(raw: str) -> List[ProviderEvent]:
    """Parse `opencode run --format json` output (one JSON object per line)
    into normalized ProviderEvents. Unknown/blank lines are skipped."""
    events: List[ProviderEvent] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        events.append(_normalize_opencode_message(msg))
    if not events:
        # Fallback: no parseable JSON (e.g. plain text output) -> one text event.
        if raw.strip():
            events.append(ProviderEvent("content.delta", {"text": raw.strip()}))
    return events


def _normalize_opencode_message(msg: dict) -> ProviderEvent:
    """Map an @opencode-ai/sdk `message` object to our vocabulary."""
    typ = msg.get("type", "")
    if typ == "message.part.updated":
        part = msg.get("part") or {}
        ptype = part.get("type")
        if ptype == "text":
            return ProviderEvent("content.delta", {"text": part.get("text", ""), "session_id": msg.get("sessionID")})
        if ptype == "reasoning":
            return ProviderEvent("reasoning", {"text": part.get("text", "")})
        return ProviderEvent("content.delta", {"type": ptype})
    if typ == "message.updated":
        role = (msg.get("info") or {}).get("role", "")
        if role == "user":
            return ProviderEvent("prompt.recorded")
        return ProviderEvent("message.updated", {"role": role})
    if typ == "session.updated":
        return ProviderEvent("session.updated", {"session_id": msg.get("sessionID")})
    # `opencode run --format json` native schema:
    if typ == "text":
        part = msg.get("part") or {}
        text = part.get("text") or msg.get("text") or ""
        return ProviderEvent("content.delta", {"text": text, "session_id": msg.get("sessionID")})
    if typ == "reasoning":
        part = msg.get("part") or {}
        return ProviderEvent("reasoning", {"text": part.get("text") or msg.get("text") or ""})
    if typ == "step_start":
        return ProviderEvent("step.start", {"session_id": msg.get("sessionID")})
    if typ == "step_finish":
        return ProviderEvent("done", {"reason": msg.get("reason") or (msg.get("part") or {}).get("reason")})
    return ProviderEvent("raw", {"type": typ})


# ---------------------------------------------------------------------------
# Concrete drivers
# ---------------------------------------------------------------------------


@dataclass
class OpenCodeDriver:
    """Drive OpenCode's headless `run` mode (grounded, JSON events)."""

    driver_kind: str = "opencode"
    display_name: str = "OpenCode"

    def dispatch(
        self,
        prompt: str,
        context_file: Optional[str],
        *,
        model: Optional[str] = None,
        workdir: Optional[str] = None,
        executor: Optional[Callable[[List[str], str], subprocess_result_proto]] = None,
    ) -> DispatchResult:
        exe = executor or _default_executor
        cwd = os.path.abspath(workdir or os.getcwd())
        argv = _executable_argv("opencode", ["run", "--format", "json"])
        if model:
            argv += ["--model", model]
        # The prompt must precede --file: opencode treats a trailing positional
        # AFTER --file as another file path, not as inline message text.
        argv.append(prompt)
        if context_file:
            argv += ["--file", context_file]
        try:
            proc = exe(argv, cwd)
        except Exception as exc:  # noqa: BLE001 - surface as error event
            return DispatchResult(
                provider=self.driver_kind,
                model=model or "default",
                events=[ProviderEvent("error", {"message": str(exc)})],
                exit_code=-1,
                error=str(exc),
            )
        events = _parse_opencode_events(proc.stdout or "")
        if proc.returncode != 0 and not events:
            events = [ProviderEvent("error", {"message": (proc.stderr or "").strip()})]
        return DispatchResult(
            provider=self.driver_kind,
            model=model or "default",
            events=events,
            exit_code=proc.returncode,
            error=None if proc.returncode == 0 else (proc.stderr or "").strip() or None,
        )


@dataclass
class ClaudeDriver:
    """Drive Claude Code's headless print mode (`claude -p`) for parity."""

    driver_kind: str = "claude"
    display_name: str = "Claude Code"

    def dispatch(
        self,
        prompt: str,
        context_file: Optional[str],
        *,
        model: Optional[str] = None,
        workdir: Optional[str] = None,
        executor: Optional[Callable[[List[str], str], subprocess_result_proto]] = None,
    ) -> DispatchResult:
        exe = executor or _default_executor
        cwd = os.path.abspath(workdir or os.getcwd())
        argv = _executable_argv("claude", ["-p", "--output-format", "json"])
        if model:
            argv += ["--model", model]
        if context_file:
            with open(context_file, "r", encoding="utf-8") as fh:
                prompt = f"<context>\n{fh.read()}\n</context>\n\n{prompt}"
        argv.append(prompt)
        try:
            proc = exe(argv, cwd)
        except Exception as exc:  # noqa: BLE001
            return DispatchResult(
                provider=self.driver_kind,
                model=model or "default",
                events=[ProviderEvent("error", {"message": str(exc)})],
                exit_code=-1,
                error=str(exc),
            )
        text = (proc.stdout or "").strip()
        events = [ProviderEvent("content.delta", {"text": text})] if text else []
        if proc.returncode != 0 and not events:
            events = [ProviderEvent("error", {"message": (proc.stderr or "").strip()})]
        return DispatchResult(
            provider=self.driver_kind,
            model=model or "default",
            events=events,
            exit_code=proc.returncode,
            error=None if proc.returncode == 0 else (proc.stderr or "").strip() or None,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_driver(provider: str) -> ProviderDriver:
    """Return the driver for a provider name (case-insensitive alias)."""
    p = (provider or "").lower()
    if p in ("opencode", "o"):
        return OpenCodeDriver()
    if p in ("claude", "claudeagent", "c"):
        return ClaudeDriver()
    raise ValueError(f"unknown provider '{provider}' (available: opencode, claude)")


# ---------------------------------------------------------------------------
# Handover (our lightweight port of `t3code handover`)
# ---------------------------------------------------------------------------


def write_context_file(pack: str) -> str:
    """Write a context pack to a temp file and return its path."""
    fd, path = tempfile.mkstemp(prefix="goline-handover-", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(pack)
    return path


def resolve_git_root(cwd: Optional[str] = None, git_exec=None) -> Optional[str]:
    """Find the git repository root for `cwd` (like t3code's workspaceMode=repo)."""
    import subprocess

    cwd = os.path.abspath(cwd or os.getcwd())
    cmd = git_exec or ["git", "rev-parse", "--show-toplevel"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    out = (proc.stdout or "").strip()
    return out or None
