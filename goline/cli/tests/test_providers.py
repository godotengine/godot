"""Tests for goline.cli.providers (T3-ported provider SPI + handover). Offline."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from goline.cli import goline_cli
from goline.cli import providers


def _make_proc(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class EventParsingTest(unittest.TestCase):
    def test_parses_text_delta_from_json_lines(self):
        raw = (
            json.dumps({"type": "message.part.updated",
                        "part": {"type": "text", "text": "hello"}}) + "\n" +
            json.dumps({"type": "message.part.updated",
                        "part": {"type": "text", "text": " world"}}) + "\n"
        )
        events = providers._parse_opencode_events(raw)
        joined = "".join(e.data["text"] for e in events if e.kind == "content.delta")
        self.assertEqual(joined, "hello world")

    def test_plain_text_fallback(self):
        events = providers._parse_opencode_events("just plain text")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "content.delta")
        self.assertEqual(events[0].data["text"], "just plain text")

    def test_blank_output_no_events(self):
        self.assertEqual(providers._parse_opencode_events(" \n \n "), [])

    def test_native_opencode_run_schema(self):
        raw = (
            json.dumps({"type": "step_start", "sessionID": "s1", "part": {"type": "step-start"}}) + "\n" +
            json.dumps({"type": "text", "sessionID": "s1",
                        "part": {"type": "text", "text": "GOLINE-OK"}}) + "\n" +
            json.dumps({"type": "step_finish", "sessionID": "s1",
                        "reason": "stop", "part": {"type": "step-finish"}}) + "\n"
        )
        events = providers._parse_opencode_events(raw)
        kinds = [e.kind for e in events]
        self.assertIn("content.delta", kinds)
        self.assertIn("step.start", kinds)
        self.assertIn("done", kinds)
        joined = "".join(e.data["text"] for e in events if e.kind == "content.delta")
        self.assertEqual(joined, "GOLINE-OK")
        done = next(e for e in events if e.kind == "done")
        self.assertEqual(done.data["reason"], "stop")


class OpenCodeDriverTest(unittest.TestCase):
    def test_dispatch_builds_correct_argv_and_streams(self):
        raw = json.dumps({"type": "message.part.updated",
                          "part": {"type": "text", "text": "hi"}}) + "\n"
        calls = {}

        def fake_executor(argv, cwd):
            calls["argv"] = argv
            calls["cwd"] = cwd
            return _make_proc(0, raw)

        with tempfile.TemporaryDirectory() as d:
            result = providers.OpenCodeDriver().dispatch(
                "do the thing", "ctx.txt", model="opencode/gpt-4o",
                workdir=d, executor=fake_executor,
            )
        argv = calls["argv"]
        # Wrapper-aware argv: on Windows the executable may be prefixed by
        # powershell/cmd + the resolved script path. Assert the logical CLI
        # arguments appear in the right order regardless of that prefix.
        self.assertIn("run", argv)
        run_idx = argv.index("run")
        logical = argv[run_idx:]
        self.assertEqual(logical, [
            "run", "--format", "json",
            "--model", "opencode/gpt-4o",
            "do the thing",
            "--file", "ctx.txt",
        ])
        self.assertEqual(calls["cwd"], os.path.abspath(d))
        self.assertEqual(result.text, "hi")
        self.assertEqual(result.exit_code, 0)

    def test_nonzero_without_events_yields_error(self):
        with mock.patch.object(
            providers.OpenCodeDriver, "dispatch",
            side_effect=lambda *a, **k: None,
        ):
            pass  # placeholder to keep the patch honest
        drv = providers.OpenCodeDriver()
        result = drv.dispatch("x", None, workdir=tempfile.gettempdir(),
                              executor=lambda argv, cwd: _make_proc(1, "", "boom"))
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(result.events[0].kind, "error")


class HandoverCLITest(unittest.TestCase):
    def test_handover_requires_prompt(self):
        code = goline_cli.main(["--handover", "--provider", "opencode"])
        self.assertEqual(code, 1)

    def test_handover_end_to_end_mocked(self):
        """Full --handover path with driver + subprocess mocked out."""
        result = providers.DispatchResult(
            provider="opencode", model="m",
            events=[providers.ProviderEvent("content.delta", {"text": "done"})],
            exit_code=0,
        )
        fake_driver = mock.Mock(driver_kind="opencode")
        fake_driver.dispatch.return_value = result

        with mock.patch.object(providers, "get_driver", return_value=fake_driver), \
             mock.patch.object(providers, "write_context_file", return_value="/tmp/ctx.txt"), \
             mock.patch.object(goline_cli.goline_context, "build_context", return_value="PACK"):
            code = goline_cli.main([
                "--handover", "--provider", "opencode", "--context", "engine",
                "--model", "opencode/gpt-4o", "--", "hello"
            ])
        self.assertEqual(code, 0)
        fake_driver.dispatch.assert_called_once()
        call = fake_driver.dispatch.call_args
        # (prompt, context_file) positional; model/workdir keyword.
        self.assertEqual(call.args[0], "hello")
        self.assertEqual(call.args[1], "/tmp/ctx.txt")
        self.assertEqual(call.kwargs["model"], "opencode/gpt-4o")


class GitRootTest(unittest.TestCase):
    def test_resolve_git_root(self):
        with mock.patch(
            "subprocess.run", return_value=_make_proc(0, "/repo/root\n")
        ) as m:
            root = providers.resolve_git_root("C:/some/dir")
        self.assertEqual(root, "/repo/root")
        argv = m.call_args.args[0]
        self.assertIn("--show-toplevel", argv)

    def test_resolve_git_root_failure_returns_none(self):
        with mock.patch(
            "subprocess.run", return_value=_make_proc(128, "", "fatal: not a git repo")
        ):
            self.assertIsNone(providers.resolve_git_root("C:/x"))


class ExecutableArgvTest(unittest.TestCase):
    def test_bare_command_when_not_a_wrapper(self):
        with mock.patch("shutil.which", return_value="C:\\bin\\opencode.exe"):
            argv = providers._executable_argv("opencode", ["run", "--format", "json"])
        self.assertEqual(argv, ["opencode", "run", "--format", "json"])

    def test_ps1_wrapper_routes_through_powershell(self):
        with mock.patch("shutil.which", return_value="C:\\Users\\u\\AppData\\Roaming\\npm\\opencode.ps1"):
            argv = providers._executable_argv("opencode", ["run"])
        self.assertEqual(argv[0], "powershell")
        self.assertEqual(argv[-1], "run")

    def test_missing_command_returns_bare(self):
        with mock.patch("shutil.which", return_value=None):
            argv = providers._executable_argv("nope", ["x"])
        self.assertEqual(argv, ["nope", "x"])


if __name__ == "__main__":
    unittest.main()
