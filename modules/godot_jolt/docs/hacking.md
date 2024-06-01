# Hacking

*See [`building.md`][bld] for per-platform prerequisites.*

This document contains details and guidelines aimed at developers looking to hack on Godot Jolt.

Most CMake [generators][gen] should be compatible, meaning you can generate project files for
environments like Visual Studio, Xcode or straight Makefiles. This is done either through their CLI
or GUI application. Alternatively you can use the [presets](#presets) described below.

## Table of Contents

- [Dependencies](#dependencies)
- [Options](#options)
- [Presets](#presets)
- [User Presets](#user-presets)
- [Formatting](#formatting)
- [Linting](#linting)
- [Updating Godot](#updating-godot)

## Dependencies

Godot Jolt makes use of CMake's [`ExternalProject`][exp] module to clone, build and integrate any
third-party dependencies like Jolt itself or the GDExtension C++ bindings. These are all lazily
cloned during your first build and as such won't appear anywhere in your workspace when you first
clone Godot Jolt. You can find the Git references for all the third-party dependencies under
`cmake/GodotJoltExternal*.cmake`.

The reasons for doing it this way, as opposed to using something like CMake's `FetchContent` module
or Git submodules, are many and boring, but this does come with the benefit that you shouldn't have
to care about appending `--recursive` to any of your Git commands and you should instead (for the
most part) be able to treat this project as if it had no external dependencies at all.

Note that every third-party dependency has been forked in order to ensure the future stability of
Godot Jolt, as there would otherwise be nothing stopping dependencies from simply deleting their
repository and thereby rendering Godot Jolt impossible to build. In some cases these forks have also
been modified to make integration easier. You can find all of these forks under the
[`godot-jolt`][org] organization on GitHub.

## Options

These are the project-specific CMake options that are available. They are only relevant if you
decide *not* to use the presets described [below](#presets), but you can also override the presets'
defaults by passing `-DGDJ_SOME_VARIABLE=VALUE` to CMake.

- `GDJ_X86_INSTRUCTION_SET`
  - Sets the minimum required CPU instruction set when compiling for x86.
  - ⚠️ This flag is not available on Apple platforms.
  - Possible values are [`NONE`, `SSE2`, `AVX`, `AVX2`, `AVX512`]
  - Default is `SSE2`
- `GDJ_INTERPROCEDURAL_OPTIMIZATION`
  - Enables interprocedural optimizations for any optimized builds, also known as link-time
    optimizations or link-time code generation.
  - Default is `TRUE`.
- `GDJ_PRECOMPILE_HEADERS`
  - Enables precompiling of header files that don't change often, like external ones, which speeds
    up compilations.
  - Default is `TRUE`.
- `GDJ_STATIC_RUNTIME_LIBRARY`
  - Whether to statically link against the platform-specific C++ runtime, for added portability.
  - ⚠️ This flag is not available on Apple or Android platforms.
  - Default is `TRUE`.
- `GDJ_USE_MIMALLOC`
  - Whether to use mimalloc as the default general-purpose memory allocator.
  - ⚠️ This flag is not available for iOS or Android.
  - Default is `TRUE`.
- `GDJ_INSTALL_DEBUG_SYMBOLS`
  - Whether to install debug symbols along with the binaries
  - Default is `FALSE`.
- `GDJ_DOUBLE_PRECISION`
  - Whether to build with 64-bit floating-point precision.
  - ⚠️ This only applies to positions, everything else will use 32-bit precision.
  - Default is `FALSE`.

## Presets

There are configuration and build presets available that utilize the relatively new
[`CMakePresets.json`][prs]. These make for a less verbose command-line interface, but also help
unify behavior across environments. Visual Studio (through a [component][mvs]) and Visual Studio
Code (through an [extension][vsc]) both support these and lets you choose these presets from within
the editor.

All these presets use the `Ninja Multi-Config` generator, which uses the [Ninja][nnj] build system.
The binaries for Ninja are bundled in this repository (under `tools/ninja`) and do not need to
installed separately.

The following configuration presets are currently available:

- `windows-msvc-x64` (Microsoft Visual C++, x86-64)
- `windows-msvc-x86` (Microsoft Visual C++, x86)
- `windows-clangcl-x64` (LLVM clang-cl, x86-64)
- `windows-clangcl-x86` (LLVM clang-cl, x86)
- `linux-gcc-x64` (GCC, x86-64)
- `linux-gcc-x86` (GCC, x86)
- `linux-clang-x64` (LLVM Clang, x86-64)
- `linux-clang-x86` (LLVM Clang, x86)
- `macos-clang` (Apple Clang, [universal][uvb])

One of the following suffixes are then applied to the configuration presets to create the build
preset:

- `-debug`
- `-development`
- `-distribution`
- `-editor-debug`
- `-editor-development`
- `-editor-distribution`

`-debug` signifies a build with optimizations disabled and extension-related debugging features
enabled.

`-development` signifies a build with optimizations enabled and extension-related debugging features
enabled.

`-distribution` signifies a build with optimizations enabled and extension-related debugging
features disabled.

`-editor` signifies what Godot calls a "debug" build and will build the version of the library that
gets used when inside the Godot editor as well as what gets bundled when exporting a "debug" build
from Godot.

You then use these presets like so:

```sh
# Using the above mentioned configure preset
cmake --preset windows-msvc-x64

# Using the above mentioned build preset
cmake --build --preset windows-msvc-x64-editor-debug
```

Note that **all** configurations currently include debug symbols when building from source. Debug
symbols will be stripped from `-distribution` binaries and provided as a separate download on
GitHub.

## User Presets

CMake offers the ability to have local/personal presets through a `CMakeUserPresets.json` file. This
file lets you define your own presets that can inherit from and extend the presets found in
`CMakePresets.json`. These user presets will show up in editors that support it just like the
regular presets.

It is encouraged to have your user presets inherit from the `dev-base` preset, which enables certain
developer-specific settings, like more pedantic CMake warnings.

Simply create a `CMakeUserPresets.json` next to `CMakePresets.json` and have it look something like
this:

```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "dev",
      "inherits": ["windows-msvc-x64", "dev-base"],
      "displayName": "MSVC, 64-bit, Development"
    }
  ],
  "buildPresets": [
    {
      "name": "dev-debug",
      "configurePreset": "dev",
      "configuration": "EditorDebug",
      "displayName": "EditorDebug",
      "targets": ["install"]
    },
    {
      "name": "dev-development",
      "configurePreset": "dev",
      "configuration": "EditorDevelopment",
      "displayName": "EditorDevelopment",
      "targets": ["install"]
    },
    {
      "name": "dev-distribution",
      "configurePreset": "dev",
      "configuration": "EditorDistribution",
      "displayName": "EditorDistribution",
      "targets": ["install"]
    }
  ]
}
```

Then you can configure/build it like any other preset:

```sh
cmake --preset dev
cmake --build --preset dev-debug
```

## Formatting

Prerequisites:

- PowerShell 7.2.7 or newer

[clang-format][clf] is used to format code in Godot Jolt. There are numerous extensions that allow
you to use clang-format from within the editor of your choosing, such as [the C++ extension][cpp]
for Visual Studio Code, which can save you from the hassle of running the commands shown below.

⚠️ The clang-format configuration that Godot Jolt uses is written for **LLVM 16.0** and won't work
with earlier versions, possibly not newer ones either.

There is a PowerShell script, `scripts/run_clang_format.ps1`, that runs clang-format on all source
files in the directory you provide, with the option to fix any errors it encounters.

To see if you have any formatting errors:

```sh
./scripts/run_clang_format.ps1 -SourcePath ./src
```

To also automatically fix any formatting errors it might encounter:

```sh
./scripts/run_clang_format.ps1 -SourcePath ./src -Fix
```

## Linting

Prerequisites:

- PowerShell 7.2.7 or newer

[clang-tidy][clt] is used to lint code in Godot Jolt. There are numerous extensions that allow you
to use clang-tidy from within the editor of your choosing, such as [the C++ extension][cpp] for
Visual Studio Code, which can save you from the hassle of running the commands shown below.

⚠️ The clang-tidy configuration that Godot Jolt uses is written for **LLVM 16.0** and won't work
with earlier versions, possibly not newer ones either.

⚠️ Because clang-tidy effectively compiles the code in order to analyze it, it's highly recommended
that you provide the script below with build files for a Clang-based compiler, such as `clang` or
`clang-cl`, in order to avoid strange errors.

There is a PowerShell script, `scripts/run_clang_tidy.ps1`, that runs clang-tidy on all source files
in the directory you provide, with the option to try to fix any errors it encounters.

To see if you have any linting errors:

```sh
# Generate build files, and disable precompiled headers to prevent compatibility issues
cmake --preset windows-clangcl-x64 -DGDJ_PRECOMPILE_HEADERS=NO

# Build any configuration, so that we fetch and prepare all of our dependencies
cmake --build --preset windows-clangcl-x64-editor-debug

# Run the script, and provide paths to source files and the generated compile_commands.json
./scripts/run_clang_tidy.ps1 -SourcePath ./src -BuildPath ./build/windows-clangcl-x64
```

To make clang-tidy attempt to fix any linting errors, you can provide the `-Fix` argument:

⚠️ This is very slow, as `-Fix` can't run multiple instances of clang-tidy in parallel.

```sh
./scripts/run_clang_tidy.ps1 -SourcePath ./src -BuildPath ./build/windows-clangcl-x64 -Fix
```

## Updating Godot

If you wish to target a version of Godot other than the current stable version then you will need to
make modifications to [the fork][gpp] of the GDExtension C++ bindings that Godot Jolt uses, in order
to update the GDExtension API to your desired version.

⚠️ This process is somewhat complicated and could lead to having to resolve compilation errors that
might appear from API differences.

⚠️ This process requires a moderate proficiency with [Git][git].

⚠️ This process will *only* work with Godot Jolt's [fork][gpp] of `godot-cpp`, as it contains
changes that are crucial for building and running Godot Jolt.

1. Clone [the fork][gpp] of `godot-cpp`
2. Run `godot --dump-extension-api extension_api.json` using the desired Godot version
3. Move `extension_api.json` into the `gdextension` directory of your `godot-cpp` clone
4. Copy `core/extension/gdextension_interface.h` from the source code of your desired Godot version
   into the `gdextension` directory of your `godot-cpp` clone
5. Commit those two files to your `godot-cpp` clone
6. Run `git rev-parse HEAD` within your `godot-cpp` clone to see the commit hash
7. Open `godot-jolt/cmake/GodotJoltExternalGodotCpp.cmake`
8. Change `GIT_REPOSITORY` to be the absolute path (on disk) of your `godot-cpp` clone
9. Change `GIT_COMMIT` to be the commit hash that you got previously
10. Replace any `\` in the repository path with `/`
11. Build Godot Jolt

To make these changes portable across workspaces you would need to push them to a remote repository
and link to that instead of your local one.

[bld]: building.md
[gen]: https://cmake.org/cmake/help/v3.22/manual/cmake-generators.7.html
[exp]: https://cmake.org/cmake/help/v3.22/module/ExternalProject.html
[org]: https://github.com/orgs/godot-jolt/repositories
[prs]: https://cmake.org/cmake/help/v3.22/manual/cmake-presets.7.html
[mvs]: https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio
[vsc]: https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools
[nnj]: https://ninja-build.org/
[uvb]: https://en.wikipedia.org/wiki/Universal_binary
[clf]: https://releases.llvm.org/15.0.0/tools/clang/docs/ClangFormat.html
[clt]: https://releases.llvm.org/15.0.0/tools/clang/tools/extra/docs/clang-tidy/index.html
[cpp]: https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools
[gpp]: https://github.com/godot-jolt/godot-cpp
[git]: https://git-scm.com/
