import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class EngineShellTests(unittest.TestCase):
    def test_capability_substitution_precedes_optional_closure_compilation(self):
        spec = importlib.util.spec_from_file_location(
            "web_helpers", Path(__file__).resolve().parents[3] / "platform/web/emscripten_helpers.py"
        )
        helpers = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "SCons.Util": types.SimpleNamespace(WhereIs=lambda _: None),
                "platform_methods": types.SimpleNamespace(get_build_version=lambda _: "test"),
            },
        ):
            spec.loader.exec_module(helpers)

        class Environment(dict):
            def File(self, path):
                return path

            def Substfile(self, **kwargs):
                self.prepared = kwargs
                return ["prepared.js"]

            def BuildJS(self, target, source, **kwargs):
                self.compiled = (target, source, kwargs)
                return ["compiled.js"]

        for closure in (False, True):
            for threads, precision, webgpu, message in (
                (False, "single", True, ""),
                (True, "single", False, "multithreaded browser"),
                (False, "double", False, "double-precision"),
                (False, "single", False, "not included"),
            ):
                with self.subTest(closure=closure, threads=threads, precision=precision, webgpu=webgpu):
                    env = Environment(use_closure_compiler=closure, precision=precision, webgpu=webgpu)
                    result = helpers.create_engine_file(env, "engine.js", ["source.js"], ["externs.js"], threads)
                    reason = json.loads(env.prepared["SUBST_DICT"]["___GODOT_WEBGPU_UNAVAILABLE_REASON___"])
                    self.assertIn(message, reason)
                    if not message:
                        self.assertEqual("", reason)
                    self.assertEqual(["compiled.js"] if closure else ["prepared.js"], result)
                    if closure:
                        self.assertEqual(("engine.js", ["prepared.js"], {"JSEXTERNS": ["externs.js"]}), env.compiled)
                    else:
                        self.assertEqual("engine.js", env.prepared["target"])


if __name__ == "__main__":
    unittest.main()
