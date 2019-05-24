# Effcee

[![Linux and OSX Build Status](https://travis-ci.org/google/effcee.svg)](https://travis-ci.org/google/effcee "Linux and OSX Build Status")

Effcee is a C++ library for stateful pattern matching of strings,
inspired by LLVM's [FileCheck][FileCheck] command.

Effcee:
- Is a library, so it can be used for quickly running tests in your own process.
- Is largely compatible with FileCheck, so tests and test-writing skills are
  transferable.
- Has few dependencies:
  - The C++11 standard library, and
  - [RE2][RE2] for regular expression matching.

## Example

The following is from [examples/main.cc](examples/main.cc):

```C++

    #include <iostream>
    #include <sstream>

    #include "effcee/effcee.h"

    // Checks standard input against the list of checks provided as command line
    // arguments.
    //
    // Example:
    //    cat <<EOF >sample_data.txt
    //    Bees
    //    Make
    //    Delicious Honey
    //    EOF
    //    effcee-example <sample_data.txt "CHECK: Bees" "CHECK-NOT:Sting" "CHECK: Honey"
    int main(int argc, char* argv[]) {
      // Read the command arguments as a list of check rules.
      std::ostringstream checks_stream;
      for (int i = 1; i < argc; ++i) {
        checks_stream << argv[i] << "\n";
      }
      // Read stdin as the input to match.
      std::stringstream input_stream;
      std::cin >> input_stream.rdbuf();

      // Attempt to match.  The input and checks arguments can be provided as
      // std::string or pointer to char.
      auto result = effcee::Match(input_stream.str(), checks_stream.str(),
                                  effcee::Options().SetChecksName("checks"));

      // Successful match result converts to true.
      if (result) {
        std::cout << "The input matched your check list!" << std::endl;
      } else {
        // Otherwise, you can get a status code and a detailed message.
        switch (result.status()) {
          case effcee::Result::Status::NoRules:
            std::cout << "error: Expected check rules as command line arguments\n";
            break;
          case effcee::Result::Status::Fail:
            std::cout << "The input failed to match your check rules:\n";
            break;
          default:
            break;
        }
        std::cout << result.message() << std::endl;
        return 1;
      }
      return 0;
    }

```

For more examples, see the matching tests
in [effcee/match_test.cc](effcee/match_test.cc).

## Status

Effcee is mature enough to be relied upon by
[third party projects](#what-uses-effcee), but could be improved.

What works:
* All check types: CHECK, CHECK-NEXT, CHECK-SAME, CHECK-DAG, CHECK-LABEL, CHECK-NOT.
* Check strings can contain:
  * fixed strings
  * regular expressions
  * variable definitions and uses
* Setting a custom check prefix.
* Accurate and helpful reporting of match failures.

What is left to do:
* Add an option to define shorthands for regular expressions.
  * For example, you could express that if the string `%%` appears where a
    regular expression is expected, then it expands to the regular expression
    for a local identifier in LLVM assembly language, i.e.
    `%[-a-zA-Z$._][-a-zA-Z$._0-9]*`.
    This enables you to write precise tests with less fuss.
* Better error reporting for failure to parse the checks list.
* Write a check language reference and tutorial.

What is left to do, but lower priority:
* Match full lines.
* Strict whitespace.
* Implicit check-not.
* Variable scoping.

## Licensing and contributing

Effcee is licensed under terms of the [Apache 2.0 license](LICENSE).  If you
are interested in contributing to this project, please see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

This is not an official Google product (experimental or otherwise), it is just
code that happens to be owned by Google.  That may change if Effcee gains
contributions from others.  See the [`CONTRIBUTING.md`](CONTRIBUTING.md) file
for more information. See also the [`AUTHORS`](AUTHORS) and
[`CONTRIBUTORS`](CONTRIBUTORS) files.

## File organization

- [`effcee`/](effcee) : library source code, and tests
- `third_party/`: third party open source packages, downloaded
  separately
- [`examples/`](examples): example programs

Effcee depends on the [RE2][RE2] regular expression library.

Effcee tests depend on [Googletest][Googletest] and [Python][Python].

In the following sections, `$SOURCE_DIR` is the directory containing the
Effcee source code.

## Getting and building Effcee

1) Check out the source code:

```sh
git clone https://github.com/google/effcee $SOURCE_DIR
cd $SOURCE_DIR/third_party
git clone https://github.com/google/googletest.git
git clone https://github.com/google/re2.git
cd $SOURCE_DIR/
```

Note: There are two other ways to manage third party sources:
- If you are building Effcee as part of a larger CMake-based project,
  add the RE2 and `googletest` projects before adding Effcee.
- Otherwise, you can set CMake variables to point to third party sources
  if they are located somewhere else.  See the [Build options](#build-options) below.

2) Ensure you have the requisite tools -- see the tools subsection below.

3) Decide where to place the build output. In the following steps, we'll call it
   `$BUILD_DIR`. Any new directory should work. We recommend building outside
   the source tree, but it is also common to build in a (new) subdirectory of
   `$SOURCE_DIR`, such as `$SOURCE_DIR/build`.

4a) Build and test with Ninja on Linux or Windows:

```sh
cd $BUILD_DIR
cmake -GNinja -DCMAKE_BUILD_TYPE={Debug|Release|RelWithDebInfo} $SOURCE_DIR
ninja
ctest
```

4b) Or build and test with MSVC on Windows:

```sh
cd $BUILD_DIR
cmake $SOURCE_DIR
cmake --build . --config {Release|Debug|MinSizeRel|RelWithDebInfo}
ctest -C {Release|Debug|MinSizeRel|RelWithDebInfo}
```

4c) Or build with MinGW on Linux for Windows:
(Skip building threaded unit tests due to
[Googletest bug 606](https://github.com/google/googletest/issues/606))

```sh
cd $BUILD_DIR
cmake -GNinja -DCMAKE_BUILD_TYPE={Debug|Release|RelWithDebInfo} $SOURCE_DIR \
   -DCMAKE_TOOLCHAIN_FILE=$SOURCE_DIR/cmake/linux-mingw-toolchain.cmake \
   -Dgtest_disable_pthreads=ON
ninja
```

After a successful build, you should have a `libeffcee` library under
the `$BUILD_DIR/effcee/` directory.

The default behavior on MSVC is to link with the static CRT. If you would like
to change this behavior `-DEFFCEE_ENABLE_SHARED_CRT` may be passed on the
cmake configure line.

### Tests

By default, Effcee registers two tests with `ctest`:

* `effcee-test`: All library tests, based on Googletest.
* `effcee-example`: Executes the example executable with sample inputs.

Running `ctest` without arguments will run the tests for Effcee as well as for
RE2.

You can disable Effcee's tests by using `-DEFFCEE_BUILD_TESTING=OFF` at
configuration time:

```sh
cmake -GNinja -DEFFCEE_BUILD_TESTING=OFF ...
```

The RE2 tests run much longer, so if you're working on Effcee alone, we
suggest limiting ctest to tests with prefix `effcee`:

    ctest -R effcee

Alternately, you can turn off RE2 tests entirely by using
`-DRE2_BUILD_TESTING=OFF` at configuration time:

```sh
cmake -GNinja -DRE2_BUILD_TESTING=OFF ...
```

### Tools you'll need

For building, testing, and profiling Effcee, the following tools should be
installed regardless of your OS:

- A compiler supporting C++11.
- [CMake][CMake]: for generating compilation targets.
- [Python][Python]: for a test script.

On Linux, if cross compiling to Windows:
- [MinGW][MinGW]: A GCC-based cross compiler targeting Windows
    so that generated executables use the Microsoft C runtime libraries.

On Windows, the following tools should be installed and available on your path:

- Visual Studio 2015 or later. Previous versions of Visual Studio are not usable
  with RE2 or Googletest.
- Git - including the associated tools, Bash, `diff`.

### Build options

Third party source locations:
- `EFFCEE_GOOGLETEST_DIR`: Location of `googletest` sources, if not under
  `third_party`.
- `EFFCEE_RE2_DIR`: Location of `re2` sources, if not under `third_party`.
- `EFFCEE_THIRD_PARTY_ROOT_DIR`: Alternate location for `googletest` and
  `re2` subdirectories.  This is used if the sources are not located under
  the `third_party` directory, and if the previous two variables are not set.

Compilation options:
- `DISABLE_RTTI`. Disable runtime type information. Default is enabled.
- `DISABLE_EXCEPTIONS`.  Disable exceptions. Default is enabled.
- `EFFCEE_ENABLE_SHARED_CRT`. See above.

Controlling samples and tests:
- `EFFCEE_BUILD_SAMPLES`. Should Effcee examples be built?  Defaults to `ON`.
- `EFFCEE_BUILD_TESTING`. Should Effcee tests be built?  Defaults to `ON`.
- `RE2_BUILD_TESTING`. Should RE2 tests be built?  Defaults to `ON`.

## Bug tracking

We track bugs using GitHub -- click on the "Issues" button on
[the project's GitHub page](https://github.com/google/effcee).

## What uses Effcee?

- [Tests](https://github.com/Microsoft/DirectXShaderCompiler/tree/master/tools/clang/test/CodeGenSPIRV)
  for SPIR-V code generation in the [DXC][DXC] HLSL compiler.
- Tests for [SPIRV-Tools][SPIRV-Tools]

## References

[CMake]: https://cmake.org/
[DXC]: https://github.com/Microsoft/DirectXShaderCompiler
[FileCheck]: http://llvm.org/docs/CommandGuide/FileCheck.html
[Googletest]: https://github.com/google/googletest
[MinGW]: http://www.mingw.org/
[Python]: https://www.python.org/
[RE2]: https://github.com/google/re2
[SPIRV-Tools]: https://github.com/KhronosGroup/SPIRV-Tools
