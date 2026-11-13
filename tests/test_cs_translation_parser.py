"""Guards the C# translation parser so project define constants are computed once.

GetProjectDefineConstants runs an MSBuild evaluation, so it must not run for every
parsed file (see godot issue #125395). The engine cannot be run from here, so this
checks the structure of the plugin source instead.
"""

import os
import re
import unittest

PLUGIN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    "modules",
    "mono",
    "editor",
    "GodotTools",
    "GodotTools",
    "CsTranslationParserPlugin.cs",
)


def _parse_file_body(source):
    match = re.search(r"public override Array<string\[\]> _ParseFile\(string path\)\s*\{", source)
    if match is None:
        raise AssertionError("_ParseFile not found in CsTranslationParserPlugin.cs")
    depth = 0
    for index in range(match.end() - 1, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[match.end() : index]
    raise AssertionError("unterminated _ParseFile body")


class CsTranslationParserCacheTest(unittest.TestCase):
    def setUp(self):
        with open(PLUGIN_PATH, encoding="utf-8") as f:
            self.source = f.read()
        self.body = _parse_file_body(self.source)

    def test_define_constants_cached_outside_per_file_loop(self):
        self.assertIn(
            "_projectDefineConstants",
            self.source,
            "CsTranslationParserPlugin must cache _projectDefineConstants across files",
        )
        guard = self.body.find("if (_projectDefineConstants == null ||")
        self.assertNotEqual(guard, -1, "_ParseFile must compute _projectDefineConstants only when it is null or stale")
        computed = self.body.find("GetProjectDefineConstants(")
        self.assertGreater(computed, guard, "GetProjectDefineConstants must run inside the guarded cache block")
        stored = self.body.find("_projectDefineConstants = defineConstants;")
        self.assertGreater(
            stored,
            computed,
            "_projectDefineConstants must be assigned only after all configuration/platform pairs are computed",
        )
        self.assertIn(
            "_projectDefineConstantsWriteTime",
            self.body,
            "the cache must be invalidated when the project file changes",
        )
        self.assertEqual(
            self.body.count("GetProjectDefineConstants("),
            1,
            "GetProjectDefineConstants must be called once in _ParseFile, not per file",
        )

    def test_parse_loop_reuses_cached_symbols(self):
        self.assertIn(
            "foreach (string[] symbols in _projectDefineConstants)",
            self.body,
            "_ParseFile must iterate the cached define constants",
        )


if __name__ == "__main__":
    unittest.main()
