# Contributing To Terrain3D

We need your help to make this the best terrain plugin for Godot.

Please see [System Architecture](https://terrain3d.readthedocs.io/en/stable/docs/system_architecture.html) to gain an understanding of how the system works. Then review the [roadmap](https://github.com/users/TokisanGames/projects/3) for priority of issues.

If you wish to take on a major component, it's best to join our [discord server](https://tokisan.com/discord) and discuss your plans with Cory to make sure your efforts are aligned with other plans.

**Table of Contents**
* [Setup Your System](#setup-your-system)
* [PR Workflow](#pr-workflow)
* [Code Style](#code-style)
* [Documentation](#documentation)

## Setup Your System

Make sure you are setup to [build the plugin from source](https://terrain3d.readthedocs.io/en/stable/docs/building_from_source.html). 

### Install clang-format

clang-format will adjust the style of your code to a consistent standard. Once you install it you can manually run it on all of your code to see or apply changes, and you can set it up to run automatically upon each commit.

#### Installing clang-format binary onto your system.
* Download version 13 or later
* Make sure the LLVM binary directory where `clang-format` is stored gets added to the `PATH` during installation
* Linux/OSX: Install the `clang-format` package, or all of `LLVM` or `clang` if your distribution doesn't provide the standalone tool
* Windows: Download LLVM for Windows from <https://releases.llvm.org/download.html>

#### Using clang-format automatically

We use Godot's clang-format hooks that will format your code upon making a commit. Install the hooks into your repo after cloning.

* Copy `tools/hooks/*` into `.git/hooks` or run `python tools/install-hooks.py`

#### Using clang-format manually

* View a formatted file, no changes on disk: `clang-format <filenames>`
* See what changes would be made: `git-clang-format --diff <filenames>`
* Change the files in place: `clang-format -i <filenames>`

 
## PR Workflow

We use the standard [Godot PR workflow](https://docs.godotengine.org/en/stable/contributing/workflow/pr_workflow.html). Please submit PRs according to the same process Godot uses.

## Code Style

### GDScript

In general, follow the [Godot GDScript style guidelines](https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_styleguide.html). 
In addition:
* All variables and functions are static typed, with a colon then space (eg. `var state: int = 3`)
* Auto static typing can be used *only* when the type is specifically assigned (eg. `var point := Vector2(1, 1)`)
* Two blank lines between functions

### GLSL

* Similar to C++ formatting below, except use `float` and no clang-format
* Private uniforms are prefaced with `_` and are hidden from the inspector and not accessible via set/get_shader_param()

### C++

In general, follow the [Godot C++ style guidelines](https://docs.godotengine.org/en/stable/contributing/development/code_style_guidelines.html).
In addition:

Use const correctness:
* Function parameters that won't be changed (almost all) should be marked const. Exceptions are pointers, or where passing a variable the function is supposed to modify, eg. Terrain3D::_generate_triangles
* Functions that won't change the object should be marked const (e.g. most get_ functions)

Pass by reference:
* Pass everything larger than 4 bytes by reference, including Ref<> and arrays, dictionaries, RIDs. e.g. `const Transform3D &xform`

* Floats:
* Use `real_t` instead of `float`
* Format float literals like `0.0f`
* Float literals and `real_t` variables can share operations (e.g. `mydouble += 1.0f`) unless the compiler complains. e.g. `Math::lerp(mydouble, real_t(0.0f), real_t(1.0f))`

Braces:
* Everything braced - no if/for one-liners. Including switch cases
* One line setters/getters can go in the header file
* Opening brace on the initial line (eg. `if (condition) {`), and ending brace at the same tab stop as the initial line

Private & Public:
* Private variables/functions prefaced with `_`
* One initial public section for constants
* Private/public/protected for members and functions in that order, in header and cpp files
* Functions in h and cpp files in same order

Other formatting:
* One blank line between functions
* All code passed through clang-format. See below


## Documentation

All PRs that include new methods and features or changed functionality should include documentation updates. This could be in the form of a tutorial page for the user manual, or API changes to the XML Class Reference.

### User Manual

Tutorials and usage documentation lives in [doc/docs](https://github.com/TokisanGames/Terrain3D/tree/main/doc/docs) and is written in Markdown (*.md). Images are stored in `images` and videos are stored [_static/video](https://github.com/TokisanGames/Terrain3D/tree/main/doc/_static/video). 

Pages also need to be included in the table of contents `doc/index.rst`. Readthedocs will then be able to find everything it needs to build the html documentation upon a commit.

### Class Reference

The class reference documentation that contributors edit is stored in [XML files](https://github.com/TokisanGames/Terrain3D/tree/main/doc/classes). These files are used as the source for generated documentation.

Edit the class reference according to the [Godot class reference primer](https://docs.godotengine.org/en/stable/contributing/documentation/class_reference_primer.html#doc-class-reference-primer).

Godot's doc-tool is used to extract or update the class structure from the compiled addon. See below for instructions.

### Using the Documentation Tools

This step isn't required for contributors. You may ask for help generating the XML class structure so you can edit it, or generating the resulting RST files. 

#### To setup your system

1. Use a bash shell available in linux, [gitforwindows](https://gitforwindows.org), or [Microsoft's WSL](https://learn.microsoft.com/en-us/windows/wsl/install).
2. Install the following modules using python's pip: `pip install docutils myst-parser sphinx sphinx-rtd-theme sphinx-rtd-dark-mode`.
3. Edit `doc/build_docs.sh` and adjust the paths to your Godot executable and `make_rst.py`, found in the Godot repository.

#### To edit the documentation

1. Build Terrain3D with your updated code.
2. Within the `doc` folder, run `./build_docs.sh`. The following will occur:
  - The Godot executable dumps the XML structure for all classes, including those of installed addons.
  - Any existing XML files (eg Terrain3D*) will be updated with the new structure, leaving prior written documentation.
  - Sphinx RST files are generated from the XML files.
  - All non-Terrain3D XML files are removed.
  - A local html copy of the docs are generated from the Markdown and RST files, and a browser is open to view them.
3. Fill in the XML files with documentation of the new generated structure and make any other changes to the Markdown files.
4. Run the script again to update the RST files. This isn't necessary for Markdown updates, except to view the changes locally.
5. Push your updates to the Markdown, XML, and RST files to the repository. Due to the nature of generation scripts, carefully review the changes so you only push those you intend.
6. Readthedocs will detect commits to the main tree and will build the online html docs from the Markdown and RST files.

