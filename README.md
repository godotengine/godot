# Godot Engine

<p align="center">
  <a href="https://godotengine.org">
    <img src="logo_outlined.svg" width="400" alt="Godot Engine logo">
  </a>
</p>

## 2D and 3D cross-platform game engine

**[Godot Engine](https://godotengine.org) is a feature-packed, cross-platform
game engine to create 2D and 3D games from a unified interface.** It provides a
comprehensive set of [common tools](https://godotengine.org/features), so that
users can focus on making games without having to reinvent the wheel. Games can
be exported with one click to a number of platforms, including the major desktop
platforms (Linux, macOS, Windows), mobile platforms (Android, iOS), as well as
Web-based platforms and [consoles](https://docs.godotengine.org/en/latest/tutorials/platform/consoles.html).

## Free, open source and community-driven

Godot is completely free and open source under the very permissive [MIT license](https://godotengine.org/license).
No strings attached, no royalties, nothing. The users' games are theirs, down
to the last line of engine code. Godot's development is fully independent and
community-driven, empowering users to help shape their engine to match their
expectations. It is supported by the [Godot Foundation](https://godot.foundation/)
not-for-profit.

Before being open sourced in [February 2014](https://github.com/godotengine/godot/commit/0b806ee0fc9097fa7bda7ac0109191c9c5e0a1ac),
Godot had been developed by [Juan Linietsky](https://github.com/reduz) and
[Ariel Manzur](https://github.com/punto-) (both still maintaining the project)
for several years as an in-house engine, used to publish several work-for-hire
titles.

![Screenshot of a 3D scene in the Godot Engine editor](https://raw.githubusercontent.com/godotengine/godot-design/master/screenshots/editor_tps_demo_1920x1080.jpg)

## Getting the engine

### Binary downloads

Official binaries for the Godot editor and the export templates can be found
[on the Godot website](https://godotengine.org/download).

### Compiling from source

[See the official docs](https://docs.godotengine.org/en/latest/contributing/development/compiling)
for compilation instructions for every supported platform.

## Community and contributing

Godot is not only an engine but an ever-growing community of users and engine
developers. The main community channels are listed [on the homepage](https://godotengine.org/community).

The best way to get in touch with the core engine developers is to join the
[Godot Contributors Chat](https://chat.godotengine.org).

To get started contributing to the project, see the [contributing guide](CONTRIBUTING.md).
This document also includes guidelines for reporting bugs.

## Documentation and demos

The official documentation is hosted on [Read the Docs](https://docs.godotengine.org).
It is maintained by the Godot community in its own [GitHub repository](https://github.com/godotengine/godot-docs).

The [class reference](https://docs.godotengine.org/en/latest/classes/)
is also accessible from the Godot editor.

We also maintain official demos in their own [GitHub repository](https://github.com/godotengine/godot-demo-projects)
as well as a list of [awesome Godot community resources](https://github.com/godotengine/awesome-godot).

There are also a number of other
[learning resources](https://docs.godotengine.org/en/latest/community/tutorials.html)
provided by the community, such as text and video tutorials, demos, etc.
Consult the [community channels](https://godotengine.org/community)
for more information.

[![Code Triagers Badge](https://www.codetriage.com/godotengine/godot/badges/users.svg)](https://www.codetriage.com/godotengine/godot)
[![Translate on Weblate](https://hosted.weblate.org/widgets/godot-engine/-/godot/svg-badge.svg)](https://hosted.weblate.org/engage/godot-engine/?utm_source=widget)
[![TODOs](https://badgen.net/https/api.tickgit.com/badgen/github.com/godotengine/godot)](https://www.tickgit.com/browse?repo=github.com/godotengine/godot)

## Personal edg3 Additions
This branch is for my personal style adjustments for Godot 4.3 C# use, in the future at some point, so it won't ever touch the master branch.

#### Features
- My first personal build

### 0. Unimportant Notes
- ```shift + right-click``` in win file explorer is old menu immediately, neat
- in ```powershell``` command ```cls``` clears all text from console

### 1. Personal Compiling Tutorial
- Install VS2022; make sure Desktop C++ is selected in install
- ```git clone https://github.com/edg3/godot.git```
- Python 3.11 Used - ```https://www.python.org/downloads/```
- Used Scoop
    - Powershell Not as Admin:
        - ```Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser```
        - ```Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression```
    - Powershell as Admin:
        - ```scoop install gcc scons make mingw```
- in the cloned ```godot``` folder: ```scons platform=windows``` - this compiles it for windows, first Time elapsed: 00:51:43.37


- tried again with ```scons -j4 platform=windows``` uses 4 thread instead, second Time elapsed: 00:00:24.93, so noted it likely doesn't recompile if no changes on files
- in ```godot\bin``` you will see the binaries; run ```godot.windows.editor.x86_64.exe``` and it will launch the editor
- To swap to build and run in VS2022; run ```scons -j4 platform=windows vsproj=yes```, you will see a new ```godot\godot.sln``` file in the godot folder
- You can add export template if you'd like, check documentation for more info if you need

```
\godot> scons platform=windows target=template_debug arch=x86_32
\godot> scons platform=windows target=template_release arch=x86_32
\godot> scons platform=windows target=template_debug arch=x86_64
\godot> scons platform=windows target=template_release arch=x86_64
\godot> scons platform=windows target=template_debug arch=arm64
\godot> scons platform=windows target=template_release arch=arm64
```

- run ```scons -j4 platform=windows target=editor module_mono_enabled=yes```
<!-- - then ```scons platform=windows target=template_debug module_mono_enabled=yes```
- finally ```scons platform=windows target=template_release module_mono_enabled=yes``` -->
- now run ```.\bin\godot.windows.editor.x86_64.mono.exe --headless --generate-mono-glue modules/mono/glue```
    - You can find generated files in ```godot\modules\mono\glue```
- finally ```python ./modules/mono/build_scripts/build_assemblies.py --godot-output-dir=./bin --godot-platform=windows```
- first run took ages; created ```build.ps1``` and it was quick

#### Notes:
- need to review; might have broken things with ```#define TOOLS_ENABLED 1``` in ```main.h``` and below in ```SConstruct```
```
#if env.editor_build:
env.Append(CPPDEFINES=["TOOLS_ENABLED"])
```
- ```git remote remove <name, e.g. origin>``` - to remove origin so I can make my own origin
- ```git remote add github https://github.com/godotengine/godot.git``` - so I can pull latest off GitHub to stay up to date