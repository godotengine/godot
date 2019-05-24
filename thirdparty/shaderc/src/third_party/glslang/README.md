Also see the Khronos landing page for glslang as a reference front end:

https://www.khronos.org/opengles/sdk/tools/Reference-Compiler/

The above page includes where to get binaries, and is kept up to date
regarding the feature level of glslang.

glslang
=======

[![Build Status](https://travis-ci.org/KhronosGroup/glslang.svg?branch=master)](https://travis-ci.org/KhronosGroup/glslang)
[![Build status](https://ci.appveyor.com/api/projects/status/q6fi9cb0qnhkla68/branch/master?svg=true)](https://ci.appveyor.com/project/Khronoswebmaster/glslang/branch/master)

An OpenGL and OpenGL ES shader front end and validator.

There are several components:

1. A GLSL/ESSL front-end for reference validation and translation of GLSL/ESSL into an AST.

2. An HLSL front-end for translation of a broad generic HLL into the AST. See [issue 362](https://github.com/KhronosGroup/glslang/issues/362) and [issue 701](https://github.com/KhronosGroup/glslang/issues/701) for current status.

3. A SPIR-V back end for translating the AST to SPIR-V.

4. A standalone wrapper, `glslangValidator`, that can be used as a command-line tool for the above.

How to add a feature protected by a version/extension/stage/profile:  See the
comment in `glslang/MachineIndependent/Versions.cpp`.

Tasks waiting to be done are documented as GitHub issues.

Execution of Standalone Wrapper
-------------------------------

To use the standalone binary form, execute `glslangValidator`, and it will print
a usage statement.  Basic operation is to give it a file containing a shader,
and it will print out warnings/errors and optionally an AST.

The applied stage-specific rules are based on the file extension:
* `.vert` for a vertex shader
* `.tesc` for a tessellation control shader
* `.tese` for a tessellation evaluation shader
* `.geom` for a geometry shader
* `.frag` for a fragment shader
* `.comp` for a compute shader

There is also a non-shader extension
* `.conf` for a configuration file of limits, see usage statement for example

Building
--------

Instead of building manually, you can also download the binaries for your
platform directly from the [master-tot release][master-tot-release] on GitHub.
Those binaries are automatically uploaded by the buildbots after successful
testing and they always reflect the current top of the tree of the master
branch.

### Dependencies

* A C++11 compiler.
  (For MSVS: 2015 is recommended, 2013 is fully supported/tested, and 2010 support is attempted, but not tested.)
* [CMake][cmake]: for generating compilation targets.
* make: _Linux_, ninja is an alternative, if configured.
* [Python 2.7][python]: for executing SPIRV-Tools scripts. (Optional if not using SPIRV-Tools.)
* [bison][bison]: _optional_, but needed when changing the grammar (glslang.y).
* [googletest][googletest]: _optional_, but should use if making any changes to glslang.

### Build steps

The following steps assume a Bash shell. On Windows, that could be the Git Bash
shell or some other shell of your choosing.

#### 1) Check-Out this project 

```bash
cd <parent of where you want glslang to be>
git clone https://github.com/KhronosGroup/glslang.git
```

#### 2) Check-Out External Projects

```bash
cd <the directory glslang was cloned to, "External" will be a subdirectory>
git clone https://github.com/google/googletest.git External/googletest
```

If you want to use googletest with Visual Studio 2013, you also need to check out an older version:

```bash
# to use googletest with Visual Studio 2013
cd External/googletest
git checkout 440527a61e1c91188195f7de212c63c77e8f0a45
cd ../..
```

If you wish to assure that SPIR-V generated from HLSL is legal for Vulkan,
or wish to invoke -Os to reduce SPIR-V size from HLSL or GLSL, install
spirv-tools with this:

```bash
./update_glslang_sources.py
```

#### 3) Configure

Assume the source directory is `$SOURCE_DIR` and the build directory is
`$BUILD_DIR`. First ensure the build directory exists, then navigate to it:

```bash
mkdir -p $BUILD_DIR
cd $BUILD_DIR
```

For building on Linux:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$(pwd)/install" $SOURCE_DIR
# "Release" (for CMAKE_BUILD_TYPE) could also be "Debug" or "RelWithDebInfo"
```

For building on Windows:

```bash
cmake $SOURCE_DIR -DCMAKE_INSTALL_PREFIX="$(pwd)/install"
# The CMAKE_INSTALL_PREFIX part is for testing (explained later).
```

The CMake GUI also works for Windows (version 3.4.1 tested).

Also, consider using `git config --global core.fileMode false` (or with `--local`) on Windows
to prevent the addition of execution permission on files.

#### 4) Build and Install

```bash
# for Linux:
make -j4 install

# for Windows:
cmake --build . --config Release --target install
# "Release" (for --config) could also be "Debug", "MinSizeRel", or "RelWithDebInfo"
```

If using MSVC, after running CMake to configure, use the
Configuration Manager to check the `INSTALL` project.

### If you need to change the GLSL grammar

The grammar in `glslang/MachineIndependent/glslang.y` has to be recompiled with
bison if it changes, the output files are committed to the repo to avoid every
developer needing to have bison configured to compile the project when grammar
changes are quite infrequent. For windows you can get binaries from
[GnuWin32][bison-gnu-win32].

The command to rebuild is:

```bash
bison --defines=MachineIndependent/glslang_tab.cpp.h
      -t MachineIndependent/glslang.y
      -o MachineIndependent/glslang_tab.cpp
```

The above command is also available in the bash script at
`glslang/updateGrammar`.

Testing
-------

Right now, there are two test harnesses existing in glslang: one is [Google
Test](gtests/), one is the [`runtests` script](Test/runtests). The former
runs unit tests and single-shader single-threaded integration tests, while
the latter runs multiple-shader linking tests and multi-threaded tests.

### Running tests

The [`runtests` script](Test/runtests) requires compiled binaries to be
installed into `$BUILD_DIR/install`. Please make sure you have supplied the
correct configuration to CMake (using `-DCMAKE_INSTALL_PREFIX`) when building;
otherwise, you may want to modify the path in the `runtests` script.

Running Google Test-backed tests:

```bash
cd $BUILD_DIR

# for Linux:
ctest

# for Windows:
ctest -C {Debug|Release|RelWithDebInfo|MinSizeRel}

# or, run the test binary directly
# (which gives more fine-grained control like filtering):
<dir-to-glslangtests-in-build-dir>/glslangtests
```

Running `runtests` script-backed tests:

```bash
cd $SOURCE_DIR/Test && ./runtests
```

### Contributing tests

Test results should always be included with a pull request that modifies
functionality.

If you are writing unit tests, please use the Google Test framework and
place the tests under the `gtests/` directory.

Integration tests are placed in the `Test/` directory. It contains test input
and a subdirectory `baseResults/` that contains the expected results of the
tests.  Both the tests and `baseResults/` are under source-code control.

Google Test runs those integration tests by reading the test input, compiling
them, and then compare against the expected results in `baseResults/`. The
integration tests to run via Google Test is registered in various
`gtests/*.FromFile.cpp` source files. `glslangtests` provides a command-line
option `--update-mode`, which, if supplied, will overwrite the golden files
under the `baseResults/` directory with real output from that invocation.
For more information, please check `gtests/` directory's
[README](gtests/README.md).

For the `runtests` script, it will generate current results in the
`localResults/` directory and `diff` them against the `baseResults/`.
When you want to update the tracked test results, they need to be
copied from `localResults/` to `baseResults/`.  This can be done by
the `bump` shell script.

You can add your own private list of tests, not tracked publicly, by using
`localtestlist` to list non-tracked tests.  This is automatically read
by `runtests` and included in the `diff` and `bump` process.

Programmatic Interfaces
-----------------------

Another piece of software can programmatically translate shaders to an AST
using one of two different interfaces:
* A new C++ class-oriented interface, or
* The original C functional interface

The `main()` in `StandAlone/StandAlone.cpp` shows examples using both styles.

### C++ Class Interface (new, preferred)

This interface is in roughly the last 1/3 of `ShaderLang.h`.  It is in the
glslang namespace and contains the following.

```cxx
const char* GetEsslVersionString();
const char* GetGlslVersionString();
bool InitializeProcess();
void FinalizeProcess();

class TShader
    setStrings(...);
    setEnvInput(EShSourceHlsl or EShSourceGlsl, stage,  EShClientVulkan or EShClientOpenGL, 100);
    setEnvClient(EShClientVulkan or EShClientOpenGL, EShTargetVulkan_1_0 or EShTargetVulkan_1_1 or EShTargetOpenGL_450);
    setEnvTarget(EShTargetSpv, EShTargetSpv_1_0 or EShTargetSpv_1_3);
    bool parse(...);
    const char* getInfoLog();

class TProgram
    void addShader(...);
    bool link(...);
    const char* getInfoLog();
    Reflection queries
```

See `ShaderLang.h` and the usage of it in `StandAlone/StandAlone.cpp` for more
details.

### C Functional Interface (orignal)

This interface is in roughly the first 2/3 of `ShaderLang.h`, and referred to
as the `Sh*()` interface, as all the entry points start `Sh`.

The `Sh*()` interface takes a "compiler" call-back object, which it calls after
building call back that is passed the AST and can then execute a backend on it.

The following is a simplified resulting run-time call stack:

```c
ShCompile(shader, compiler) -> compiler(AST) -> <back end>
```

In practice, `ShCompile()` takes shader strings, default version, and
warning/error and other options for controlling compilation.

Basic Internal Operation
------------------------

* Initial lexical analysis is done by the preprocessor in
  `MachineIndependent/Preprocessor`, and then refined by a GLSL scanner
  in `MachineIndependent/Scan.cpp`.  There is currently no use of flex.

* Code is parsed using bison on `MachineIndependent/glslang.y` with the
  aid of a symbol table and an AST.  The symbol table is not passed on to
  the back-end; the intermediate representation stands on its own.
  The tree is built by the grammar productions, many of which are
  offloaded into `ParseHelper.cpp`, and by `Intermediate.cpp`.

* The intermediate representation is very high-level, and represented
  as an in-memory tree.   This serves to lose no information from the
  original program, and to have efficient transfer of the result from
  parsing to the back-end.  In the AST, constants are propogated and
  folded, and a very small amount of dead code is eliminated.

  To aid linking and reflection, the last top-level branch in the AST
  lists all global symbols.

* The primary algorithm of the back-end compiler is to traverse the
  tree (high-level intermediate representation), and create an internal
  object code representation.  There is an example of how to do this
  in `MachineIndependent/intermOut.cpp`.

* Reduction of the tree to a linear byte-code style low-level intermediate
  representation is likely a good way to generate fully optimized code.

* There is currently some dead old-style linker-type code still lying around.

* Memory pool: parsing uses types derived from C++ `std` types, using a
  custom allocator that puts them in a memory pool.  This makes allocation
  of individual container/contents just few cycles and deallocation free.
  This pool is popped after the AST is made and processed.

  The use is simple: if you are going to call `new`, there are three cases:

  - the object comes from the pool (its base class has the macro
    `POOL_ALLOCATOR_NEW_DELETE` in it) and you do not have to call `delete`

  - it is a `TString`, in which case call `NewPoolTString()`, which gets
    it from the pool, and there is no corresponding `delete`

  - the object does not come from the pool, and you have to do normal
    C++ memory management of what you `new`


[cmake]: https://cmake.org/
[python]: https://www.python.org/
[bison]: https://www.gnu.org/software/bison/
[googletest]: https://github.com/google/googletest
[bison-gnu-win32]: http://gnuwin32.sourceforge.net/packages/bison.htm
[master-tot-release]: https://github.com/KhronosGroup/glslang/releases/tag/master-tot
