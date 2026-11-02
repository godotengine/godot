import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeEnvironment(dict[str, list[str]]):
    def AppendUnique(self, **values):
        for name, items in values.items():
            current = self.setdefault(name, [])
            current.extend(item for item in items if item not in current)

    def Append(self, **values):
        for name, items in values.items():
            self.setdefault(name, []).extend(items)

    def Prepend(self, **values):
        for name, items in values.items():
            self[name] = items + self.get(name, [])

    def File(self, name):
        return Path(name)

    def Exit(self, code):
        raise SystemExit(code)


class NativeConfigTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "native_config_under_test", Path(__file__).resolve().parents[3] / "drivers/webgpu/native_config.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        self.messages = []
        with patch.dict(sys.modules, {"methods": types.SimpleNamespace(print_error=self.messages.append)}):
            spec.loader.exec_module(self.module)

    def test_disabled_backend_does_not_require_or_add_dawn(self):
        env = FakeEnvironment(webgpu=False, platform="windows")
        self.module.configure_native_webgpu(env)
        self.assertEqual(env, {"webgpu": False, "platform": "windows"})

    def test_missing_sdk_fails_with_setup_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            env = FakeEnvironment(webgpu=True, platform="windows", webgpu_native_path=directory)
            with self.assertRaises(SystemExit):
                self.module.configure_native_webgpu(env)
            self.assertIn("native/README.md", self.messages[0])

    def test_desktop_sdks_stage_dawn_without_linking_it_into_other_renderer_paths(self):
        for platform, runtime in (
            ("windows", "bin/godot_webgpu_dawn.dll"),
            ("linuxbsd", "lib/libgodot_webgpu_dawn.so"),
            ("macos", "lib/libgodot_webgpu_dawn.dylib"),
        ):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                for name in ("include/dawn/webgpu.h", runtime):
                    file = root / name
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.touch()
                env = FakeEnvironment(webgpu=True, platform=platform, webgpu_native_path=directory)
                self.module.configure_native_webgpu(env)
                self.assertEqual(env["webgpu_native_path"], str(root))
                self.assertNotIn("LIBS", env)
                self.assertEqual(env["webgpu_native_runtimes"], [str(root / runtime)])
                self.assertIn("RD_ENABLED", env["CPPDEFINES"])
                self.assertIn("WEBGPU_ENABLED", env["CPPDEFINES"])
                self.assertNotIn("LINKFLAGS", env)

    def test_relative_sdk_path_is_resolved_from_the_build_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in ("sdk/include/dawn/webgpu.h", "sdk/lib/libgodot_webgpu_dawn.so"):
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.touch()
            previous = os.getcwd()
            os.chdir(root)
            try:
                env = FakeEnvironment(webgpu=True, platform="linuxbsd", webgpu_native_path="sdk")
                self.module.configure_native_webgpu(env)
            finally:
                os.chdir(previous)
            self.assertEqual(env["webgpu_native_path"], str(root / "sdk"))

    def test_windows_compiler_sidecars_are_staged_when_supplied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("include/dawn/webgpu.h", "bin/godot_webgpu_dawn.dll", "bin/dxcompiler.dll", "bin/dxil.dll"):
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.touch()
            env = FakeEnvironment(webgpu=True, platform="windows", webgpu_native_path=directory)
            self.module.configure_native_webgpu(env)
            self.assertEqual(
                {Path(name).name for name in env["webgpu_native_runtimes"]},
                {"godot_webgpu_dawn.dll", "dxcompiler.dll", "dxil.dll"},
            )


if __name__ == "__main__":
    unittest.main()
