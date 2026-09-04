"""Tests for goline.cli.goline_cli (Stage 2). Pure / offline."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from goline.cli import goline_cli as cli


class DiscoverAgentsTest(unittest.TestCase):
    def test_no_tools_on_path_returns_empty(self):
        with mock.patch.object(cli, "_which", return_value=None):
            found = cli.discover_agents()
        self.assertEqual(found, {})

    def test_discovers_and_orders_file_edit_first(self):
        # Pretend opencode, aichat, and claude are all on PATH.
        def fake_which(command: str):
            return {n: f"/usr/bin/{n}" for n in cli._KNOWN_NAMES}.get(command)

        def fake_probe(command, flag):
            return f"{command} 9.9.9"

        with mock.patch.object(cli, "_which", side_effect=fake_which), mock.patch.object(
            cli, "_probe_version", side_effect=fake_probe
        ):
            found = cli.discover_agents()

        names = list(found.keys())
        # file-edit kinds sort before 'chat'; within a kind, alphabetical.
        self.assertNotEqual(names[0], "aichat")  # aichat is 'chat', sorted last
        self.assertEqual(found[names[0]]["kind"], "file-edit")
        self.assertEqual(found["aichat"]["kind"], "chat")
        self.assertEqual(names[-1], "aichat")  # chat kind sorts last
        self.assertIn("claude", names)
        self.assertIn("opencode", names)
        self.assertEqual(found["claude"]["version"], "claude 9.9.9")
        self.assertEqual(found["claude"]["kind"], "file-edit")


class CommandArgvTest(unittest.TestCase):
    def test_plain_binary_unchanged(self):
        with mock.patch.object(cli, "_which", return_value="/usr/bin/claude"):
            self.assertEqual(cli._command_argv("claude"), ["claude"])

    def test_ps1_wrapper_routed_through_powershell(self):
        with mock.patch.object(cli, "_which", return_value=r"C:\x\claude.ps1"):
            argv = cli._command_argv("claude")
        self.assertEqual(argv[0], "powershell")
        self.assertTrue(argv[-1].endswith("claude.ps1"))

    def test_cmd_wrapper_routed_through_cmd(self):
        with mock.patch.object(cli, "_which", return_value=r"C:\x\opencode.cmd"):
            argv = cli._command_argv("opencode")
        self.assertEqual(argv[0], "cmd")
        self.assertTrue(argv[-1].endswith("opencode.cmd"))

    def test_missing_binary_falls_back_to_command(self):
        with mock.patch.object(cli, "_which", return_value=None):
            self.assertEqual(cli._command_argv("nope"), ["nope"])


class ResolveAgentTest(unittest.TestCase):
    def _agents(self, names):
        return {
            n: {"kind": "file-edit" if n != "aichat" else "chat", "binary": n}
            for n in names
        }

    def test_requested_wins(self):
        a = self._agents(["claude", "opencode"])
        self.assertEqual(cli.resolve_agent("opencode", a), "opencode")

    def test_env_var(self):
        a = self._agents(["claude", "opencode"])
        with mock.patch.dict("os.environ", {"GOLINE_AGENT": "opencode"}, clear=True):
            self.assertEqual(cli.resolve_agent(None, a), "opencode")

    def test_defaults_to_first_file_edit(self):
        a = self._agents(["aichat", "claude"])
        self.assertEqual(cli.resolve_agent(None, a), "claude")

    def test_raises_when_requested_missing(self):
        with self.assertRaises(RuntimeError):
            cli.resolve_agent("ghost", self._agents(["claude"]))

    def test_raises_when_nothing_found(self):
        with self.assertRaises(RuntimeError):
            cli.resolve_agent(None, {})


class SmokeTest(unittest.TestCase):
    """Optional: runs against whatever is actually on PATH."""

    def test_self_test_never_crashes(self):
        # Should return 0 whether or not agents are installed.
        self.assertEqual(cli.main(["--self-test"]), 0)


class ContextLaunchTest(unittest.TestCase):
    """Exercise --context through main() with the agent launch mocked out."""

    def _fake_discovery(self):
        return {
            "opencode": {"kind": "file-edit", "binary": "opencode"},
            "claude": {"kind": "file-edit", "binary": "claude"},
        }

    def test_context_game_launch_builds_prompt_with_project(self):
        with mock.patch.object(
            cli, "discover_agents", side_effect=self._fake_discovery
        ), mock.patch.object(cli, "launch_agent", return_value=0) as launch:
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, "project.godot"), "w", encoding="utf-8") as fh:
                    fh.write('config_version=5\n[application]\nconfig/name="T"\n')
                code = cli.main(["--agent", "opencode", "--context", "game", "--project", d, "--", "hello"])
        self.assertEqual(code, 0)
        self.assertEqual(launch.call_count, 1)
        args = launch.call_args[0][1]
        self.assertEqual(args[0], "hello")

    def test_context_engine_launch_defaults_prompt_to_context_path(self):
        with mock.patch.object(
            cli, "discover_agents", side_effect=self._fake_discovery
        ), mock.patch.object(cli, "launch_agent", return_value=0) as launch, mock.patch.object(
            cli.goline_context, "detect_engine_root", return_value=os.getcwd()
        ), mock.patch.object(cli.goline_context, "_git", return_value=None), mock.patch.object(
            cli.goline_context, "_which", return_value=None
        ):
            code = cli.main(["--agent", "opencode", "--context", "engine"])
        self.assertEqual(code, 0)
        args = launch.call_args[0][1]
        self.assertTrue(args[0].startswith("Read the grounded context at "))


class EventAuditHelpersTest(unittest.TestCase):
    """Unit coverage for goline_cli._classify_event_command/_audit_agent_events."""

    def _ev(self, kind, **data):
        from goline.cli import providers

        return providers.ProviderEvent(kind, data)

    def test_classify_extracts_and_denies_destructive_command(self):
        ev = self._ev("tool", command="rm -rf /tmp/x")
        decision = cli._classify_event_command(ev)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.decision, "deny")

    def test_classify_allows_read_only_git(self):
        ev = self._ev("permission", command="git status")
        decision = cli._classify_event_command(ev)
        self.assertTrue(decision.allowed)

    def test_classify_ignores_event_without_command(self):
        ev = self._ev("content.delta", text="hello")
        self.assertIsNone(cli._classify_event_command(ev))

    def test_audit_records_only_command_events(self):
        import json as _json
        from goline.cli import goline_cli as _cli
        from goline.cli import providers

        events = [
            self._ev("tool", command="git status"),
            self._ev("tool", command="rm x"),
            self._ev("content.delta", text="no command"),
        ]
        with tempfile.TemporaryDirectory() as d:
            audit_path = os.path.join(d, "audit.jsonl")
            log = cli.goline_policy.AuditLog(audit_path)
            cli._audit_agent_events(events, log)
            with open(audit_path, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        self.assertEqual(len(lines), 2)
        recs = [_json.loads(l) for l in lines]
        self.assertEqual(recs[0]["decision"], "allow")
        self.assertEqual(recs[1]["decision"], "deny")


if __name__ == "__main__":
    unittest.main()
