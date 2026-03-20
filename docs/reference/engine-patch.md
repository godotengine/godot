# Engine Patch Report

This report lists fork deltas against pinned upstream Godot source for files that originate from upstream.

## Metadata

| Field | Value |
| --- | --- |
| Status | `ok` |
| Upstream repo | `https://github.com/godotengine/godot.git` |
| Upstream ref (configured) | `876b290332ec6f2e6d173d08162a02aa7e6ca46d` |
| Upstream commit (resolved) | `876b290332ec6f2e6d173d08162a02aa7e6ca46d` |
| Fork-base commit (merge-base) | `876b290332ec6f2e6d173d08162a02aa7e6ca46d` |
| Head ref | `4cc28822021ed7e8a032cd64bd1202d8a38ac32d` |
| Head commit | `4cc28822021ed7e8a032cd64bd1202d8a38ac32d` |
| Rename threshold | `-M70%` |

## Summary

| Metric | Count |
| --- | ---: |
| Total upstream-origin changed files | 134 |
| Modified (`M`) | 95 |
| Renamed (`R*`) | 0 |
| Deleted (`D`) | 39 |

## Engine-source-only summary

Filter: files under runtime engine source roots with code/shader extensions (C/C++/Obj-C headers/sources and `.glsl`).

| Metric | Count |
| --- | ---: |
| Total engine-source changed files | 32 |
| Modified (`M`) | 32 |
| Renamed (`R*`) | 0 |
| Deleted (`D`) | 0 |

## Engine-source-only changed files

| Status | Path | Subsystem | + | - | Last touch |
| --- | --- | --- | ---: | ---: | --- |
| `M` | `core/config/engine.cpp` | `core` | 4 | 0 | `aeeb1475cb34` |
| `M` | `core/config/engine.h` | `core` | 6 | 5 | `aeeb1475cb34` |
| `M` | `drivers/gles3/rasterizer_gles3.h` | `drivers` | 5 | 4 | `aeeb1475cb34` |
| `M` | `drivers/gles3/rasterizer_scene_gles3.cpp` | `drivers` | 1 | 1 | `aeeb1475cb34` |
| `M` | `drivers/gles3/rasterizer_scene_gles3.h` | `drivers` | 1 | 1 | `aeeb1475cb34` |
| `M` | `editor/docks/scene_tree_dock.cpp` | `editor` | 75 | 1 | `aeeb1475cb34` |
| `M` | `editor/docks/scene_tree_dock.h` | `editor` | 3 | 0 | `aeeb1475cb34` |
| `M` | `main/main.cpp` | `main` | 8 | 4 | `aeeb1475cb34` |
| `M` | `platform/windows/os_windows.cpp` | `platform/windows` | 3 | 2 | `aeeb1475cb34` |
| `M` | `servers/rendering/dummy/rasterizer_dummy.h` | `servers` | 4 | 3 | `aeeb1475cb34` |
| `M` | `servers/rendering/dummy/rasterizer_scene_dummy.h` | `servers` | 1 | 1 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_compositor.h` | `servers` | 8 | 3 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_rd/effects/copy_effects.cpp` | `servers` | 65 | 22 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_rd/effects/copy_effects.h` | `servers` | 15 | 8 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_rd/forward_clustered/render_forward_clustered.cpp` | `servers` | 252 | 1 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_rd/renderer_compositor_rd.cpp` | `servers` | 10 | 8 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_rd/renderer_compositor_rd.h` | `servers` | 14 | 11 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_rd/renderer_scene_render_rd.cpp` | `servers` | 119 | 9 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_rd/renderer_scene_render_rd.h` | `servers` | 10 | 4 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_rd/storage_rd/light_storage.h` | `servers` | 2 | 0 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_rd/storage_rd/render_data_rd.h` | `servers` | 17 | 3 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_rd/storage_rd/utilities.cpp` | `servers` | 16 | 9 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_scene_cull.cpp` | `servers` | 179 | 45 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_scene_cull.h` | `servers` | 45 | 25 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_scene_render.h` | `servers` | 1 | 1 | `aeeb1475cb34` |
| `M` | `servers/rendering/renderer_viewport.cpp` | `servers` | 4 | 0 | `aeeb1475cb34` |
| `M` | `servers/rendering/rendering_device.cpp` | `servers` | 44 | 11 | `aeeb1475cb34` |
| `M` | `servers/rendering/rendering_device.h` | `servers` | 14 | 7 | `aeeb1475cb34` |
| `M` | `servers/rendering/rendering_server_default.cpp` | `servers` | 5 | 4 | `aeeb1475cb34` |
| `M` | `servers/rendering/rendering_server_globals.cpp` | `servers` | 1 | 0 | `aeeb1475cb34` |
| `M` | `servers/rendering/rendering_server_globals.h` | `servers` | 8 | 3 | `aeeb1475cb34` |
| `M` | `servers/rendering_server.h` | `servers` | 5 | 4 | `aeeb1475cb34` |

## Subsystem distribution

| Subsystem | Files |
| --- | ---: |
| `.gitattributes` | 1 |
| `.github` | 22 |
| `.gitignore` | 1 |
| `.pre-commit-config.yaml` | 1 |
| `CHANGELOG.md` | 1 |
| `CONTRIBUTING.md` | 1 |
| `README.md` | 1 |
| `core` | 2 |
| `doc` | 12 |
| `drivers` | 3 |
| `editor` | 18 |
| `glsl_builders.py` | 1 |
| `main` | 1 |
| `misc` | 17 |
| `modules/gdscript` | 1 |
| `modules/mono` | 18 |
| `platform/android` | 1 |
| `platform/web` | 1 |
| `platform/windows` | 1 |
| `servers` | 24 |
| `tests` | 2 |
| `thirdparty` | 3 |
| `version.py` | 1 |

## Changed upstream-origin files

| Status | Path | Subsystem | + | - | Last touch | Deletion commit | Pre-deletion commit |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `M` | `.gitattributes` | `.gitattributes` | 5 | 0 | `d9f052c12dd4` | `-` | `-` |
| `D` | `.github/CODEOWNERS` | `.github` | 0 | 259 | `aeeb1475cb34` | `aeeb1475cb34` | `8bb22eb66802` |
| `D` | `.github/ISSUE_TEMPLATE/bug_report.yml` | `.github` | 0 | 72 | `aeeb1475cb34` | `aeeb1475cb34` | `41a81f5360d2` |
| `D` | `.github/ISSUE_TEMPLATE/config.yml` | `.github` | 0 | 14 | `aeeb1475cb34` | `aeeb1475cb34` | `389227bff13c` |
| `D` | `.github/PULL_REQUEST_TEMPLATE.md` | `.github` | 0 | 8 | `aeeb1475cb34` | `aeeb1475cb34` | `500c83305dbb` |
| `D` | `.github/actions/download-artifact/action.yml` | `.github` | 0 | 20 | `aeeb1475cb34` | `aeeb1475cb34` | `41a81f5360d2` |
| `D` | `.github/actions/godot-build/action.yml` | `.github` | 0 | 42 | `aeeb1475cb34` | `aeeb1475cb34` | `d38bda7e7d21` |
| `D` | `.github/actions/godot-cache-restore/action.yml` | `.github` | 0 | 39 | `aeeb1475cb34` | `aeeb1475cb34` | `423ba3da00f9` |
| `D` | `.github/actions/godot-cache-save/action.yml` | `.github` | 0 | 22 | `aeeb1475cb34` | `aeeb1475cb34` | `423ba3da00f9` |
| `D` | `.github/actions/godot-converter-test/action.yml` | `.github` | 0 | 20 | `aeeb1475cb34` | `aeeb1475cb34` | `41a81f5360d2` |
| `D` | `.github/actions/godot-cpp-build/action.yml` | `.github` | 0 | 38 | `aeeb1475cb34` | `aeeb1475cb34` | `715d4bf7e603` |
| `D` | `.github/actions/godot-deps/action.yml` | `.github` | 0 | 35 | `aeeb1475cb34` | `aeeb1475cb34` | `423ba3da00f9` |
| `D` | `.github/actions/godot-project-test/action.yml` | `.github` | 0 | 45 | `aeeb1475cb34` | `aeeb1475cb34` | `57eac93820de` |
| `D` | `.github/actions/upload-artifact/action.yml` | `.github` | 0 | 22 | `aeeb1475cb34` | `aeeb1475cb34` | `423ba3da00f9` |
| `D` | `.github/workflows/android_builds.yml` | `.github` | 0 | 125 | `aeeb1475cb34` | `aeeb1475cb34` | `486abd481a5b` |
| `D` | `.github/workflows/cache_cleanup.yml` | `.github` | 0 | 32 | `2f4234d31599` | `2f4234d31599` | `1e5b075f4886` |
| `D` | `.github/workflows/ios_builds.yml` | `.github` | 0 | 44 | `aeeb1475cb34` | `aeeb1475cb34` | `486abd481a5b` |
| `D` | `.github/workflows/linux_builds.yml` | `.github` | 0 | 257 | `aeeb1475cb34` | `aeeb1475cb34` | `76ffc3adf6ee` |
| `D` | `.github/workflows/macos_builds.yml` | `.github` | 0 | 108 | `aeeb1475cb34` | `aeeb1475cb34` | `76ffc3adf6ee` |
| `D` | `.github/workflows/runner.yml` | `.github` | 0 | 46 | `aeeb1475cb34` | `aeeb1475cb34` | `2820bc97de6c` |
| `D` | `.github/workflows/static_checks.yml` | `.github` | 0 | 46 | `aeeb1475cb34` | `aeeb1475cb34` | `2820bc97de6c` |
| `D` | `.github/workflows/web_builds.yml` | `.github` | 0 | 77 | `aeeb1475cb34` | `aeeb1475cb34` | `c5bf809d160b` |
| `D` | `.github/workflows/windows_builds.yml` | `.github` | 0 | 128 | `aeeb1475cb34` | `aeeb1475cb34` | `dd0387fcb0fc` |
| `M` | `.gitignore` | `.gitignore` | 124 | 0 | `4cc28822021e` | `-` | `-` |
| `M` | `.pre-commit-config.yaml` | `.pre-commit-config.yaml` | 23 | 25 | `aeeb1475cb34` | `-` | `-` |
| `M` | `CHANGELOG.md` | `CHANGELOG.md` | 36 | 2518 | `a9aa717a87f8` | `-` | `-` |
| `M` | `CONTRIBUTING.md` | `CONTRIBUTING.md` | 6 | 208 | `aeeb1475cb34` | `-` | `-` |
| `M` | `README.md` | `README.md` | 25 | 66 | `d9f052c12dd4` | `-` | `-` |
| `M` | `core/config/engine.cpp` | `core` | 4 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `core/config/engine.h` | `core` | 6 | 5 | `aeeb1475cb34` | `-` | `-` |
| `M` | `doc/tools/doc_status.py` | `doc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `doc/tools/make_rst.py` | `doc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `doc/translations/es.po` | `doc` | 321 | 11446 | `aeeb1475cb34` | `-` | `-` |
| `M` | `doc/translations/fr.po` | `doc` | 515 | 5517 | `aeeb1475cb34` | `-` | `-` |
| `M` | `doc/translations/ga.po` | `doc` | 424 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `doc/translations/it.po` | `doc` | 628 | 377 | `aeeb1475cb34` | `-` | `-` |
| `M` | `doc/translations/ko.po` | `doc` | 66 | 282 | `aeeb1475cb34` | `-` | `-` |
| `M` | `doc/translations/ru.po` | `doc` | 673 | 1203 | `aeeb1475cb34` | `-` | `-` |
| `M` | `doc/translations/ta.po` | `doc` | 580 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `doc/translations/uk.po` | `doc` | 707 | 1241 | `aeeb1475cb34` | `-` | `-` |
| `M` | `doc/translations/zh_CN.po` | `doc` | 598 | 1076 | `aeeb1475cb34` | `-` | `-` |
| `M` | `doc/translations/zh_TW.po` | `doc` | 146 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `drivers/gles3/rasterizer_gles3.h` | `drivers` | 5 | 4 | `aeeb1475cb34` | `-` | `-` |
| `M` | `drivers/gles3/rasterizer_scene_gles3.cpp` | `drivers` | 1 | 1 | `aeeb1475cb34` | `-` | `-` |
| `M` | `drivers/gles3/rasterizer_scene_gles3.h` | `drivers` | 1 | 1 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/docks/scene_tree_dock.cpp` | `editor` | 75 | 1 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/docks/scene_tree_dock.h` | `editor` | 3 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/editor/de.po` | `editor` | 2 | 36 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/editor/es.po` | `editor` | 22 | 72 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/editor/fa.po` | `editor` | 1 | 60 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/editor/fr.po` | `editor` | 3 | 41 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/editor/ga.po` | `editor` | 13 | 3688 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/editor/ko.po` | `editor` | 10 | 56 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/editor/pl.po` | `editor` | 1 | 49 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/editor/ru.po` | `editor` | 2 | 49 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/editor/uk.po` | `editor` | 3 | 50 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/editor/zh_CN.po` | `editor` | 2 | 43 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/properties/de.po` | `editor` | 3 | 6 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/properties/es.po` | `editor` | 4 | 7 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/properties/ko.po` | `editor` | 16 | 19 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/properties/pl.po` | `editor` | 4 | 4 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/properties/ru.po` | `editor` | 3 | 7 | `aeeb1475cb34` | `-` | `-` |
| `M` | `editor/translations/properties/zh_CN.po` | `editor` | 2 | 5 | `aeeb1475cb34` | `-` | `-` |
| `M` | `glsl_builders.py` | `glsl_builders.py` | 21 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `main/main.cpp` | `main` | 8 | 4 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/dist/visionos_xcode/libgodot.visionos.debug.xcframework/Info.plist` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/dist/visionos_xcode/libgodot.visionos.release.xcframework/Info.plist` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/char_range_fetch.py` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/check_ci_log.py` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/copyright_headers.py` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/dotnet_format.py` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/file_format.py` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/gitignore_check.sh` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/header_guards.py` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/install_d3d12_sdk_windows.py` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/install_vulkan_sdk_macos.sh` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/make_icons.sh` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/make_tarball.sh` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/purge_cache.py` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/ucaps_fetch.py` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/unicode_ranges_fetch.py` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `misc/scripts/validate_extension_api.sh` | `misc` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `D` | `modules/gdscript/tests/scripts/completion/get_node/local/local.cfg` | `modules/gdscript` | 0 | 7 | `aeeb1475cb34` | `aeeb1475cb34` | `ae853e1a4282` |
| `D` | `modules/mono/editor/Godot.NET.Sdk/Godot.NET.Sdk/Godot.NET.Sdk.csproj` | `modules/mono` | 0 | 37 | `aeeb1475cb34` | `aeeb1475cb34` | `6b441ab6a449` |
| `D` | `modules/mono/editor/Godot.NET.Sdk/Godot.SourceGenerators.Sample/Godot.SourceGenerators.Sample.csproj` | `modules/mono` | 0 | 35 | `aeeb1475cb34` | `aeeb1475cb34` | `fb8553e4d763` |
| `D` | `modules/mono/editor/Godot.NET.Sdk/Godot.SourceGenerators.Tests/Godot.SourceGenerators.Tests.csproj` | `modules/mono` | 0 | 44 | `aeeb1475cb34` | `aeeb1475cb34` | `580a225a4a16` |
| `D` | `modules/mono/editor/Godot.NET.Sdk/Godot.SourceGenerators/Godot.SourceGenerators.csproj` | `modules/mono` | 0 | 40 | `aeeb1475cb34` | `aeeb1475cb34` | `6b441ab6a449` |
| `D` | `modules/mono/editor/GodotTools/GodotTools.BuildLogger/GodotTools.BuildLogger.csproj` | `modules/mono` | 0 | 11 | `aeeb1475cb34` | `aeeb1475cb34` | `1036bfd7ad56` |
| `D` | `modules/mono/editor/GodotTools/GodotTools.Core/GodotTools.Core.csproj` | `modules/mono` | 0 | 10 | `aeeb1475cb34` | `aeeb1475cb34` | `fb8553e4d763` |
| `D` | `modules/mono/editor/GodotTools/GodotTools.IdeMessaging.CLI/GodotTools.IdeMessaging.CLI.csproj` | `modules/mono` | 0 | 15 | `aeeb1475cb34` | `aeeb1475cb34` | `1036bfd7ad56` |
| `D` | `modules/mono/editor/GodotTools/GodotTools.IdeMessaging/GodotTools.IdeMessaging.csproj` | `modules/mono` | 0 | 26 | `aeeb1475cb34` | `aeeb1475cb34` | `64f2e8b64f8a` |
| `D` | `modules/mono/editor/GodotTools/GodotTools.OpenVisualStudio/GodotTools.OpenVisualStudio.csproj` | `modules/mono` | 0 | 19 | `aeeb1475cb34` | `aeeb1475cb34` | `1036bfd7ad56` |
| `D` | `modules/mono/editor/GodotTools/GodotTools.ProjectEditor/GodotTools.ProjectEditor.csproj` | `modules/mono` | 0 | 22 | `aeeb1475cb34` | `aeeb1475cb34` | `8d41b5a582f8` |
| `D` | `modules/mono/editor/GodotTools/GodotTools.Shared/GodotTools.Shared.csproj` | `modules/mono` | 0 | 12 | `aeeb1475cb34` | `aeeb1475cb34` | `fb8553e4d763` |
| `D` | `modules/mono/editor/GodotTools/GodotTools/GodotTools.csproj` | `modules/mono` | 0 | 62 | `aeeb1475cb34` | `aeeb1475cb34` | `b53af55462b3` |
| `D` | `modules/mono/glue/GodotSharp/Godot.SourceGenerators.Internal/Godot.SourceGenerators.Internal.csproj` | `modules/mono` | 0 | 15 | `aeeb1475cb34` | `aeeb1475cb34` | `4047e4b89442` |
| `D` | `modules/mono/glue/GodotSharp/GodotPlugins/GodotPlugins.csproj` | `modules/mono` | 0 | 18 | `aeeb1475cb34` | `aeeb1475cb34` | `fb8553e4d763` |
| `D` | `modules/mono/glue/GodotSharp/GodotSharp/GodotSharp.csproj` | `modules/mono` | 0 | 158 | `aeeb1475cb34` | `aeeb1475cb34` | `6b441ab6a449` |
| `D` | `modules/mono/glue/GodotSharp/GodotSharpEditor/GodotSharpEditor.csproj` | `modules/mono` | 0 | 49 | `aeeb1475cb34` | `aeeb1475cb34` | `6b441ab6a449` |
| `M` | `modules/mono/build_scripts/build_assemblies.py` | `modules/mono` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `modules/mono/thirdparty/libSystem.Security.Cryptography.Native.Android.jar` | `modules/mono` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `platform/android/java/gradlew` | `platform/android` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `platform/web/serve.py` | `platform/web` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `platform/windows/os_windows.cpp` | `platform/windows` | 3 | 2 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/dummy/rasterizer_dummy.h` | `servers` | 4 | 3 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/dummy/rasterizer_scene_dummy.h` | `servers` | 1 | 1 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_compositor.h` | `servers` | 8 | 3 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_rd/effects/SCsub` | `servers` | 1 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_rd/effects/copy_effects.cpp` | `servers` | 65 | 22 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_rd/effects/copy_effects.h` | `servers` | 15 | 8 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_rd/forward_clustered/render_forward_clustered.cpp` | `servers` | 252 | 1 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_rd/renderer_compositor_rd.cpp` | `servers` | 10 | 8 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_rd/renderer_compositor_rd.h` | `servers` | 14 | 11 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_rd/renderer_scene_render_rd.cpp` | `servers` | 119 | 9 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_rd/renderer_scene_render_rd.h` | `servers` | 10 | 4 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_rd/storage_rd/light_storage.h` | `servers` | 2 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_rd/storage_rd/render_data_rd.h` | `servers` | 17 | 3 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_rd/storage_rd/utilities.cpp` | `servers` | 16 | 9 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_scene_cull.cpp` | `servers` | 179 | 45 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_scene_cull.h` | `servers` | 45 | 25 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_scene_render.h` | `servers` | 1 | 1 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/renderer_viewport.cpp` | `servers` | 4 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/rendering_device.cpp` | `servers` | 44 | 11 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/rendering_device.h` | `servers` | 14 | 7 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/rendering_server_default.cpp` | `servers` | 5 | 4 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/rendering_server_globals.cpp` | `servers` | 1 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering/rendering_server_globals.h` | `servers` | 8 | 3 | `aeeb1475cb34` | `-` | `-` |
| `M` | `servers/rendering_server.h` | `servers` | 5 | 4 | `aeeb1475cb34` | `-` | `-` |
| `M` | `tests/create_test.py` | `tests` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `tests/test_main.cpp` | `tests` | 3 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `thirdparty/basis_universal/encoder/basisu_astc_hdr_6x6_enc.h` | `thirdparty` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `thirdparty/sdl/update-sdl.sh` | `thirdparty` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `thirdparty/thorvg/update-thorvg.sh` | `thirdparty` | 0 | 0 | `aeeb1475cb34` | `-` | `-` |
| `M` | `version.py` | `version.py` | 2 | 2 | `aeeb1475cb34` | `-` | `-` |

## Touchpoints (compact)

Line-range touchpoints are compact in Markdown; full per-hunk details are canonical in `engine-patch.json`.

## Regeneration

```bash
python3 scripts/generate_engine_patch_report.py
```

