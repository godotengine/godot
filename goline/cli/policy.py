"""Goline command permission / audit gate (ARCHITECTURE §8/§9, Stage 8).

Pure, dependency-free policy that decides whether a command string may be
run by an agent, plus an append-only audit log. Nothing here executes
commands; it only *classifies* them so callers (the orchestrator or a human)
can enforce the decision.

Core ideas:
  - Every command is reduced to its executable name + a normalized lowercased
    form, then matched against deny patterns (destructive) and allow patterns.
  - Default posture: allow read-only/inspect commands; DENY destructive ones
    unless a policy explicitly allows them.
  - Audit entries are timestamped {decision, reason, command} and appended as
    JSON lines; the log is never mutated in place (append-only).

This module is deliberately pure (no subprocess) and fully offline-testable.
"""

from __future__ import annotations

import datetime
import json
import os
import re

# ---------------------------------------------------------------------------
# Destructive command patterns (default deny). Each is an executable name or
# a regex matched against the whole normalized command.
# ---------------------------------------------------------------------------

# Executable names that are destructive by nature, whatever their args.
_DENY_EXECUTABLES = frozenset(
    {
        # file / filesystem destruction
        "rm",
        "rmdir",
        "shred",
        "dd",
        "deltree",
        "format",
        "wipefs",
        "del",
        "rd",
        "remove-item",
        "ri",
        "clear-item",
        "clear-content",
        "remove-itemproperty",
        # partitioning / low level storage
        "mkfs",
        "fdisk",
        "parted",
        "sfdisk",
        "gdisk",
        "cryptsetup",
        "pvcreate",
        "vgremove",
        "lvremove",
        # system / power
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init",
        # process killing (force)
        "kill",
        "killall",
        "pkill",
        "taskkill",
        "stop-process",
    }
)

# Whole-command regexes for dangerous flags/combinations.
_DENY_PATTERNS = (
    # --- git history / working-tree destruction ---
    re.compile(r"\bgit\s+(reset|clean|rebase|merge|prune|gc)\b"),
    re.compile(r"\bgit\s+checkout\s+--\s+"),
    re.compile(r"\bgit\s+(branch|tag)\s+-(d|D)\b"),
    re.compile(r"\bgit\s+stash\s+(drop|clear)\b"),
    re.compile(r"\bgit\s+rm\b"),
    # --- git force push (any form, incl. +refspec) ---
    re.compile(r"\bgit\s+push\b.*(--force|\\-\\-force|\s-f\b)"),
    re.compile(r"\bgit\s+push\b.*\+\s*[A-Za-z0-9_:./-]+"),
    # --- recursive / forced deletion across shells ---
    re.compile(r"\b(rm|rmdir)\b.*-\w*r\b"),
    re.compile(r"\b(rm|rmdir|del|rd|deltree|format)\b.*[-/][Ss]"),
    re.compile(r"\b(Remove-Item|Clear-Item|Clear-Content)\b.*\s(-Recurse|-Force)"),
    re.compile(r"\bremove-item\b.*\s-(recurse|force|r|f)\b"),
    # --- privilege escalation / shell takeover ---
    re.compile(r"\bsudo\b"),
    re.compile(r"\b>:\(\)|:\s*\(\s*\)\s*\{"),
    re.compile(r"\b(eval|exec)\b.*\$\("),
    # --- remote code install pipelines (supply chain) ---
    re.compile(r"\b(curl|wget|iwr|Invoke-WebRequest)\b.*\|\s*(sh|bash|zsh|python|python3|node|iex)\b"),
    re.compile(r"\b(iwr|Invoke-WebRequest)\b.*\|\s*iex\b"),
    # --- system mutation ---
    re.compile(r"\b(reg|reg\.exe)\s+delete\b"),
    re.compile(r"\b(taskkill|kill|Stop-Process|pkill|killall)\b.*\b(-9|/f|-Force|force)\b"),
    re.compile(r"\b(shutdown|reboot|halt|poweroff|Stop-Computer|Restart-Computer)\b"),
    re.compile(r"\binit\s+(0|6)\b"),
    re.compile(r"\bchmod\s+[0-7]{3}\b"),
    re.compile(r"\b>+\s*\S+(?:\s|$)|&&\s*>"),
    re.compile(r"\b(powershell|cmd)\s+/c\s+(rm|del|format|rd|deltree|reset|clean|iwr|curl)"),
)

# Read-only / inspect commands we permit by default. Anything not matching a
# deny pattern and whose executable is known-safe is allowed; a small
# allow-list makes the intent explicit and tightens the default.
_ALLOW_EXECUTABLES = frozenset(
    {
        "git",
        "python",
        "node",
        "scons",
        "cl",
        "g++",
        "clang",
        "where",
        "Get-ChildItem",
        "dir",
        "ls",
        "cat",
        "type",
        "echo",
        "opencode",
        "claude",
    }
)

# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------

DENY = "deny"
ALLOW = "allow"
ERROR = "error"


class Decision:
    """The verdict for a command plus a human-readable reason."""

    __slots__ = ("decision", "reason", "command")

    def __init__(self, decision: str, reason: str, command: str) -> None:
        self.decision = decision
        self.reason = reason
        self.command = command

    @property
    def allowed(self) -> bool:
        return self.decision == ALLOW

    def to_dict(self) -> dict:
        return {"decision": self.decision, "reason": self.reason, "command": self.command}


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class Policy:
    """Default-deny-destructive policy. Pure and deterministic."""

    def __init__(
        self,
        custom_deny: "list[str] | None" = None,
        custom_allow: "list[str] | None" = None,
        deny_all: bool = False,
    ) -> None:
        # Extra deny regexes (strings) add to the built-in set.
        self._extra_deny = [re.compile(p) for p in (custom_deny or [])]
        # Extra allow patterns as regexes on the normalized command.
        self._extra_allow = [re.compile(p) for p in (custom_allow or [])]
        self._deny_all = deny_all

    @staticmethod
    def _executable(command: str) -> str:
        """Return the leading executable token, lowercased."""
        cmd = command.strip()
        if not cmd:
            return ""
        # Handle `exe arg...` and `exe "arg with space"` first token.
        tok = cmd.split(None, 1)[0].strip('"').strip("'")
        return tok.lower()

    def classify(self, command: str) -> Decision:
        """Return an allow/deny decision for `command` (never executes it)."""
        cmd = (command or "").strip()
        if self._deny_all:
            return Decision(DENY, "deny-all policy active", cmd)
        if not cmd:
            return Decision(ERROR, "empty command", cmd)

        norm = cmd.lower()
        exe = self._executable(cmd)

        # Extra allow rules first (explicitly permitted by the operator).
        if any(p.search(norm) for p in self._extra_allow):
            return Decision(ALLOW, "explicit allow rule matched", cmd)

        # Check caller-provided deny rules, then built-in deny pattern set.
        for pat in self._extra_deny + list(_DENY_PATTERNS):
            if pat.search(norm):
                return Decision(DENY, f"deny pattern: {pat.pattern}", cmd)

        if exe in _DENY_EXECUTABLES:
            return Decision(DENY, f"destructive executable: {exe}", cmd)

        if exe == "git":
            # Already handled destructive git ops above; remaining git is
            # read-only-ish (status/diff/log/rev-parse) and allowed.
            return Decision(ALLOW, "git command not matching destructive patterns", cmd)

        if exe in _ALLOW_EXECUTABLES:
            return Decision(ALLOW, f"known-safe executable: {exe}", cmd)

        # Unknown executable: default-deny (safer) unless allow-listed above.
        return Decision(DENY, f"unrecognized executable: {exe}, not allow-listed", cmd)


# ---------------------------------------------------------------------------
# Audit log (append-only)
# ---------------------------------------------------------------------------

class AuditLog:
    """Append-only JSONL audit log. Never rewrites existing entries."""

    def __init__(self, path: "str | None" = None) -> None:
        self.path = path
        self._memory: list[dict] = []

    def record(self, decision: Decision) -> None:
        entry = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **decision.to_dict(),
        }
        self._memory.append(entry)
        if self.path:
            self._append_to_disk(entry)

    def _append_to_disk(self, entry: dict) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            # Logging failure must never crash the caller.
            pass

    @property
    def entries(self) -> "list[dict]":
        return list(self._memory)


DEFAULT_DENY_NOTICE = (
    "Destructive commands (rm/del, git reset/clean/force-push, sudo, ...) are "
    "denied by the default Goline policy. Override with an explicit allow rule "
    "only when you intend it."
)


def default_kickoff_notice() -> str:
    return DEFAULT_DENY_NOTICE


# Human-readable summary of the DENY *patterns* (regex-based rules that are
# not expressed as a single denied executable). Derived from _DENY_PATTERNS so
# an agent can "hold" the policy as an instruction.
_DENY_PATTERN_NOTICE = (
    "git history/destruction: reset, clean, rebase, merge, prune, gc, "
    "checkout --, rm, stash drop/clear, branch/tag -d, and ANY force push "
    "(--force, -f, or +refspec)",
    "privilege escalation: sudo; shell redirection to a file",
    "remote code install: curl | wget | iwr piped to sh | bash | python | node | iex",
    "registry deletion and Windows/system mutation (reg delete, init 0, ...)",
    "any executable NOT in the allow list (unknown tools are denied by default)",
)


def guidance_notice() -> str:
    """Render the gate policy as a MANDATORY instruction block for an agent."""
    exes = ", ".join(sorted(_DENY_EXECUTABLES))
    patterns = "".join(f"- {line}\n" for line in _DENY_PATTERN_NOTICE)
    return (
        "## Agent permission policy (MANDATORY)\n"
        "You MUST NOT run any command matching these rules without first "
        "asking the human for explicit approval:\n"
        + f"- Denied executables: {exes}\n"
        + patterns
        + "If your next action would run such a command, STOP and ask first."
    )
