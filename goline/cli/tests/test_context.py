"""Tests for goline.cli.context (Stage 3 grounded context packs). Offline."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from goline.cli import context as ctx


class EngineContextTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        # Minimal engine skeleton.
        for f in ("version.py", "SConstruct", "GOLINE.md"):
            with open(os.path.join(self.root, f), "w", encoding="utf-8") as fh:
                fh.write("x\n")
        for d in ("core", "editor", "goline", "docs"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        with open(os.path.join(self.root, "version.py"), "w", encoding="utf-8") as fh:
            fh.write('short_name = "godot"\nname = "Goline"\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_detects_engine_root_from_marker(self):
        self.assertEqual(ctx.detect_engine_root(self.root), self.root)

    def test_engine_context_includes_identity_and_dirs(self):
        with mock.patch.object(ctx, "_git", return_value=None), mock.patch.object(
            ctx, "_which", return_value=None
        ):
            pack = ctx.build_engine_context(root=self.root)
        self.assertIn("name=Goline", pack)
        self.assertIn("core", pack)
        self.assertIn("editor", pack)
        self.assertIn("goline", pack)
        self.assertIn("GOLINE.md", pack)

    def test_engine_context_embeds_permission_policy(self):
        with mock.patch.object(ctx, "_git", return_value=None), mock.patch.object(
            ctx, "_which", return_value=None
        ):
            pack = ctx.build_engine_context(root=self.root)
        self.assertIn("Agent permission policy", pack)
        self.assertIn("MUST NOT run", pack)
        self.assertIn("rm", pack)
        self.assertIn("STOP and ask", pack)

    def test_engine_context_reports_tools_and_git(self):
        def fake_git(cwd, *args):
            if args[0] == "rev-parse" and args[1] == "--abbrev-ref":
                return "myfork"
            if args[0] == "rev-parse":
                return "abc1234"
            if args[0] == "status":
                return " M version.py"
            return ""

        def fake_which(name):
            return "found" if name == "python" else None

        with mock.patch.object(ctx, "_git", side_effect=fake_git), mock.patch.object(
            ctx, "_which", side_effect=fake_which
        ):
            pack = ctx.build_engine_context(root=self.root)
        self.assertIn("git_branch: myfork", pack)
        self.assertIn("git_commit: abc1234", pack)
        self.assertIn("git_clean: no", pack)
        self.assertIn("python", pack)

    def test_git_failure_degrades_gracefully(self):
        with mock.patch.object(ctx, "_git", return_value=None), mock.patch.object(
            ctx, "_which", return_value=None
        ):
            pack = ctx.build_engine_context(root=self.root)
        self.assertNotIn("git_branch:", pack)


class GameContextTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "scripts"), exist_ok=True)
        with open(os.path.join(self.root, "project.godot"), "w", encoding="utf-8") as fh:
            fh.write(
                "; Engine configuration\n"
                'config_version=5\n\n'
                "[application]\n"
                'config/name="MyGame"\n'
                'run/main_scene="res://scenes/Main.tscn"\n'
            )
        with open(os.path.join(self.root, "scripts", "player.gd"), "w", encoding="utf-8") as fh:
            fh.write("extends CharacterBody2D\n")
        with open(os.path.join(self.root, "scripts", "enemy.cs"), "w", encoding="utf-8") as fh:
            fh.write("using Godot;\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_game_context_reads_project(self):
        pack = ctx.build_game_context(project=self.root)
        self.assertIn("project_name: MyGame", pack)
        self.assertIn("config_version: 5", pack)
        self.assertIn("main_scene: res://scenes/Main.tscn", pack)
        self.assertIn("scripts/player.gd", pack)
        self.assertIn("scripts/enemy.cs", pack)
        self.assertIn("Agent permission policy", pack)

    def test_non_project_dir_returns_message(self):
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        msg = ctx.build_game_context(project=empty.name)
        self.assertTrue(msg.startswith("No game project found"))

    def test_dispatch(self):
        with mock.patch.object(ctx, "_git", return_value=None), mock.patch.object(
            ctx, "_which", return_value=None
        ):
            self.assertIn("engine context", ctx.build_context("engine", None))
        self.assertIn("game project", ctx.build_context("game", self.root).lower())

    def test_dispatch_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            ctx.build_context("bogus")


if __name__ == "__main__":
    unittest.main()
