# P00 baseline, 2026-09-23

- Local HEAD before changes: `ca871ccc9cd8e5c2ce829ec5c0fae1ac32c6269f`, branch `master`. Remote `origin` is `https://github.com/slimedragonair/SlimeEngine`; `upstream` is `https://github.com/godotengine/godot.git`. One worktree was present. Dirty paths were only the two user supplied, untracked `SlimeEngine_Agent_Kickoff.md` and `SlimeEngine_AI_Implementation_Plan.md`; both remain untouched. No repository `AGENTS.md` was found, and `CONTRIBUTING.md` was read.
- Work branch created after the baseline passed: `codex/slime-ai-p00-p03`; no reset, pull, stash, or cleanup was performed.
- Python 3.12.10, SCons 4.10.1, Node 24.12.0, npm 11.6.2. Visual Studio 2022 Community supplied MSVC 14.3 and Windows SDK 10.0.26100.0. RAM 103000600576 bytes; E: free space at inspection 198178156544 bytes.
- The initial `python -m SCons platform=windows target=editor arch=x86_64 tests=yes dev_build=yes -j8` exited 2 before compilation: installed `site-packages/misc` shadowed this source tree's `misc/utility` package. This is a local Python import collision, not a C++ failure. The correction below changes import resolution only in the SCons process.
- Successful unmodified build command (exit 0, 00:06:41.91): `python -c "import misc; misc.__path__.insert(0, r'E:\SlimeEngine\misc'); import SCons.Script; SCons.Script.main()" platform=windows target=editor arch=x86_64 tests=yes dev_build=yes -j8`.
- Actual binary: `E:\SlimeEngine\bin\godot.windows.editor.dev.x86_64.console.exe`, SHA-256 `29E4813CE97AD2E1BB65D113AE779374DFE91A742DFBBE808D268D733FE58ABA`; `--version` returned `4.8.dev.custom_build.ca871ccc9` (exit 0).
- Baseline editor smoke: `& E:\SlimeEngine\bin\godot.windows.editor.dev.x86_64.console.exe --headless --editor --path $env:TEMP\slimeengine-p00-fixture --quit-after 60` returned 0 after project scan and editor layout load. The disposable fixture had one Node2D root and a Node2D Marker.
- Baseline native test: `& E:\SlimeEngine\bin\godot.windows.editor.dev.x86_64.console.exe --test --test-case='*[Node2D]*' --no-colors` returned 0: 2 cases passed, 0 failed, 1427 skipped; 45 assertions passed. This is a relevant subset, not a full suite.
- Evidence: `evidence/p00-baseline-build.log`, `evidence/p00-editor-smoke.log`, and `evidence/p00-native-tests.log`. The baseline binary hash and paths above tie the logs to the source revision.

Trusted repository fixtures were added only after the baseline build and tests. Initial SHA-256 hashes:

| File | SHA-256 |
|---|---|
| `tools/slime_ai/fixtures/2d/project.godot` | `732C273905B1289E6B289D7563A509A18AD2257723619792962205AFCFEC9D27` |
| `tools/slime_ai/fixtures/2d/main.tscn` | `FE8557743069A5084B2A19C9DCBB9FBC2964D9BA729C1FA6DEDC15CE1674A88F` |
| `tools/slime_ai/fixtures/3d/project.godot` | `13DD8D5FF356F3914B6DF7294DEFB1A134DE95107FB4374CBD5F48076D92C73F` |
| `tools/slime_ai/fixtures/3d/main.tscn` | `8A9D96C7FBD68F383A4449614C5020396E2946D4B86CED85629F154AE0979D03` |

Test runs must copy these fixtures to a disposable directory before mutations. The fixture source files contain only built-in nodes, no scripts or plugins.

Both repository fixtures were copied to fresh GUID-named directories under `%TEMP%` and opened with the unmodified editor using `--headless --editor --path <copy> --quit-after 60`. Both exited 0. Logs: `evidence/p00-fixture-2d.log` and `evidence/p00-fixture-3d.log`.
