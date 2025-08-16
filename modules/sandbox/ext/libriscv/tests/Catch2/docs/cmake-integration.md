<a id="top"></a>
# CMake integration

**Contents**<br>
[CMake targets](#cmake-targets)<br>
[Automatic test registration](#automatic-test-registration)<br>
[CMake project options](#cmake-project-options)<br>
[`CATCH_CONFIG_*` customization options in CMake](#catch_config_-customization-options-in-cmake)<br>
[Installing Catch2 from git repository](#installing-catch2-from-git-repository)<br>
[Installing Catch2 from vcpkg](#installing-catch2-from-vcpkg)<br>
[Installing Catch2 from Bazel](#installing-catch2-from-bazel)<br>

Because we use CMake to build Catch2, we also provide a couple of
integration points for our users.

1) Catch2 exports a (namespaced) CMake target
2) Catch2's repository contains CMake scripts for automatic registration
of `TEST_CASE`s in CTest

## CMake targets

Catch2's CMake build exports two targets, `Catch2::Catch2`, and
`Catch2::Catch2WithMain`. If you do not need custom `main` function,
you should be using the latter (and only the latter). Linking against
it will add the proper include paths and link your target together with
2 static libraries that implement Catch2 and its main respectively.
If you need custom `main`, you should link only against `Catch2::Catch2`.

This means that if Catch2 has been installed on the system, it should
be enough to do
```cmake
find_package(Catch2 3 REQUIRED)
# These tests can use the Catch2-provided main
add_executable(tests test.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

# These tests need their own main
add_executable(custom-main-tests test.cpp test-main.cpp)
target_link_libraries(custom-main-tests PRIVATE Catch2::Catch2)
```

These targets are also provided when Catch2 is used as a subdirectory.
Assuming Catch2 has been cloned to `lib/Catch2`, you only need to replace
the `find_package` call with `add_subdirectory(lib/Catch2)` and the snippet
above still works.


Another possibility is to use [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html):
```cmake
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

add_executable(tests test.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
```


## Automatic test registration

Catch2's repository also contains three CMake scripts that help users
with automatically registering their `TEST_CASE`s with CTest. They
can be found in the `extras` folder, and are

1) `Catch.cmake` (and its dependency `CatchAddTests.cmake`)
2) `ParseAndAddCatchTests.cmake` (deprecated)
3) `CatchShardTests.cmake` (and its dependency `CatchShardTestsImpl.cmake`)

If Catch2 has been installed in system, both of these can be used after
doing `find_package(Catch2 REQUIRED)`. Otherwise you need to add them
to your CMake module path.

<a id="catch_discover_tests"></a>
### `Catch.cmake` and `CatchAddTests.cmake`

`Catch.cmake` provides function `catch_discover_tests` to get tests from
a target. This function works by running the resulting executable with
`--list-test` flag, and then parsing the output to find all existing tests.

#### Usage
```cmake
cmake_minimum_required(VERSION 3.16)

project(baz LANGUAGES CXX VERSION 0.0.1)

find_package(Catch2 REQUIRED)
add_executable(tests test.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2)

include(CTest)
include(Catch)
catch_discover_tests(tests)
```

When using `FetchContent`, `include(Catch)` will fail unless
`CMAKE_MODULE_PATH` is explicitly updated to include the extras
directory.

```cmake
# ... FetchContent ...
#
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(tests)
```

#### Customization
`catch_discover_tests` can be given several extra arguments:
```cmake
catch_discover_tests(target
                     [TEST_SPEC arg1...]
                     [EXTRA_ARGS arg1...]
                     [WORKING_DIRECTORY dir]
                     [TEST_PREFIX prefix]
                     [TEST_SUFFIX suffix]
                     [PROPERTIES name1 value1...]
                     [TEST_LIST var]
                     [REPORTER reporter]
                     [OUTPUT_DIR dir]
                     [OUTPUT_PREFIX prefix]
                     [OUTPUT_SUFFIX suffix]
                     [DISCOVERY_MODE <POST_BUILD|PRE_TEST>]
                     [SKIP_IS_FAILURE]
                     [ADD_TAGS_AS_LABELS]
)
```

* `TEST_SPEC arg1...`

Specifies test cases, wildcarded test cases, tags and tag expressions to
pass to the Catch executable alongside the `--list-test-names-only` flag.


* `EXTRA_ARGS arg1...`

Any extra arguments to pass on the command line to each test case.


* `WORKING_DIRECTORY dir`

Specifies the directory in which to run the discovered test cases.  If this
option is not provided, the current binary directory is used.


* `TEST_PREFIX prefix`

Specifies a _prefix_ to be added to the name of each discovered test case.
This can be useful when the same test executable is being used in multiple
calls to `catch_discover_tests()`, with different `TEST_SPEC` or `EXTRA_ARGS`.


* `TEST_SUFFIX suffix`

Same as `TEST_PREFIX`, except it specific the _suffix_ for the test names.
Both `TEST_PREFIX` and `TEST_SUFFIX` can be specified at the same time.


* `PROPERTIES name1 value1...`

Specifies additional properties to be set on all tests discovered by this
invocation of `catch_discover_tests`.


* `TEST_LIST var`

Make the list of tests available in the variable `var`, rather than the
default `<target>_TESTS`.  This can be useful when the same test
executable is being used in multiple calls to `catch_discover_tests()`.
Note that this variable is only available in CTest.

* `REPORTER reporter`

Use the specified reporter when running the test case. The reporter will
be passed to the test runner as `--reporter reporter`.

* `OUTPUT_DIR dir`

If specified, the parameter is passed along as
`--out dir/<test_name>` to test executable. The actual file name is the
same as the test name. This should be used instead of
`EXTRA_ARGS --out foo` to avoid race conditions writing the result output
when using parallel test execution.

* `OUTPUT_PREFIX prefix`

May be used in conjunction with `OUTPUT_DIR`.
If specified, `prefix` is added to each output file name, like so
`--out dir/prefix<test_name>`.

* `OUTPUT_SUFFIX suffix`

May be used in conjunction with `OUTPUT_DIR`.
If specified, `suffix` is added to each output file name, like so
`--out dir/<test_name>suffix`. This can be used to add a file extension to
the output file name e.g. ".xml".

* `DISCOVERY_MODE mode`

If specified allows control over when test discovery is performed.
For a value of `POST_BUILD` (default) test discovery is performed at build time.
For a value of `PRE_TEST` test discovery is delayed until just prior to test
execution (useful e.g. in cross-compilation environments).
``DISCOVERY_MODE`` defaults to the value of the
``CMAKE_CATCH_DISCOVER_TESTS_DISCOVERY_MODE`` variable if it is not passed when
calling ``catch_discover_tests``. This provides a mechanism for globally
selecting a preferred test discovery behavior.

* `SKIP_IS_FAILURE`

Skipped tests will be marked as failed instead.

* `ADD_TAGS_AS_LABELS`

Add the tags from tests as labels to CTest.


### `ParseAndAddCatchTests.cmake`

⚠ This script is [deprecated](https://github.com/catchorg/Catch2/pull/2120)
in Catch2 2.13.4 and superseded by the above approach using `catch_discover_tests`.
See [#2092](https://github.com/catchorg/Catch2/issues/2092) for details.

`ParseAndAddCatchTests` works by parsing all implementation files
associated with the provided target, and registering them via CTest's
`add_test`. This approach has some limitations, such as the fact that
commented-out tests will be registered anyway. More serious, only a
subset of the assertion macros currently available in Catch can be
detected by this script and tests with any macros that cannot be
parsed are *silently ignored*.


#### Usage

```cmake
cmake_minimum_required(VERSION 3.16)

project(baz LANGUAGES CXX VERSION 0.0.1)

find_package(Catch2 REQUIRED)
add_executable(tests test.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2)

include(CTest)
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(tests)
```


#### Customization

`ParseAndAddCatchTests` provides some customization points:
* `PARSE_CATCH_TESTS_VERBOSE` -- When `ON`, the script prints debug
messages. Defaults to `OFF`.
* `PARSE_CATCH_TESTS_NO_HIDDEN_TESTS` -- When `ON`, hidden tests (tests
tagged with either of `[.]` or `[.foo]`) will not be registered.
Defaults to `OFF`.
* `PARSE_CATCH_TESTS_ADD_FIXTURE_IN_TEST_NAME` -- When `ON`, adds fixture
class name to the test name in CTest. Defaults to `ON`.
* `PARSE_CATCH_TESTS_ADD_TARGET_IN_TEST_NAME` -- When `ON`, adds target
name to the test name in CTest. Defaults to `ON`.
* `PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS` -- When `ON`, adds test
file to `CMAKE_CONFIGURE_DEPENDS`. This means that the CMake configuration
step will be re-ran when the test files change, letting new tests be
automatically discovered. Defaults to `OFF`.


Optionally, one can specify a launching command to run tests by setting the
variable `OptionalCatchTestLauncher` before calling `ParseAndAddCatchTests`. For
instance to run some tests using `MPI` and other sequentially, one can write
```cmake
set(OptionalCatchTestLauncher ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NUMPROC})
ParseAndAddCatchTests(mpi_foo)
unset(OptionalCatchTestLauncher)
ParseAndAddCatchTests(bar)
```


### `CatchShardTests.cmake`

> `CatchShardTests.cmake` was introduced in Catch2 3.1.0.

`CatchShardTests.cmake` provides a function
`catch_add_sharded_tests(TEST_BINARY)` that splits tests from `TEST_BINARY`
into multiple shards. The tests in each shard and their order is randomized,
and the seed changes every invocation of CTest.

Currently there are 3 customization points for this script:

 * SHARD_COUNT - number of shards to split target's tests into
 * REPORTER    - reporter spec to use for tests
 * TEST_SPEC   - test spec used for filtering tests

Example usage:

```
include(CatchShardTests)

catch_add_sharded_tests(foo-tests
  SHARD_COUNT 4
  REPORTER "xml::out=-"
  TEST_SPEC "A"
)

catch_add_sharded_tests(tests
  SHARD_COUNT 8
  REPORTER "xml::out=-"
  TEST_SPEC "B"
)
```

This registers total of 12 CTest tests (4 + 8 shards) to run shards
from `foo-tests` test binary, filtered by a test spec.

_Note that this script is currently a proof-of-concept for reseeding
shards per CTest run, and thus does not support (nor does it currently
aim to support) all customization points from
[`catch_discover_tests`](#catch_discover_tests)._


## CMake project options

Catch2's CMake project also provides some options for other projects
that consume it. These are:

* `BUILD_TESTING` -- When `ON` and the project is not used as a subproject,
Catch2's test binary will be built. Defaults to `ON`.
* `CATCH_INSTALL_DOCS` -- When `ON`, Catch2's documentation will be
included in the installation. Defaults to `ON`.
* `CATCH_INSTALL_EXTRAS` -- When `ON`, Catch2's extras folder (the CMake
scripts mentioned above, debugger helpers) will be included in the
installation. Defaults to `ON`.
* `CATCH_DEVELOPMENT_BUILD` -- When `ON`, configures the build for development
of Catch2. This means enabling test projects, warnings and so on.
Defaults to `OFF`.


Enabling `CATCH_DEVELOPMENT_BUILD` also enables further configuration
customization options:

* `CATCH_BUILD_TESTING` -- When `ON`, Catch2's SelfTest project will be
built. Defaults to `ON`. Note that Catch2 also obeys `BUILD_TESTING` CMake
variable, so _both_ of them need to be `ON` for the SelfTest to be built,
and either of them can be set to `OFF` to disable building SelfTest.
* `CATCH_BUILD_EXAMPLES` -- When `ON`, Catch2's usage examples will be
built. Defaults to `OFF`.
* `CATCH_BUILD_EXTRA_TESTS` -- When `ON`, Catch2's extra tests will be
built. Defaults to `OFF`.
* `CATCH_BUILD_FUZZERS` -- When `ON`, Catch2 fuzzing entry points will
be built. Defaults to `OFF`.
* `CATCH_ENABLE_WERROR` -- When `ON`, adds `-Werror` or equivalent flag
to the compilation. Defaults to `ON`.
* `CATCH_BUILD_SURROGATES` -- When `ON`, each header in Catch2 will be
compiled separately to ensure that they are self-sufficient.
Defaults to `OFF`.


## `CATCH_CONFIG_*` customization options in CMake

> CMake support for `CATCH_CONFIG_*` options was introduced in Catch2 3.0.1

Due to the new separate compilation model, all the options from the
[Compile-time configuration docs](configuration.md#top) can also be set
through Catch2's CMake. To set them, define the option you want as `ON`,
e.g. `-DCATCH_CONFIG_NOSTDOUT=ON`.

Note that setting the option to `OFF` doesn't disable it. To force disable
an option, you need to set the `_NO_` form of it to `ON`, e.g.
`-DCATCH_CONFIG_NO_COLOUR_WIN32=ON`.


To summarize the configuration option behaviour with an example:

| `-DCATCH_CONFIG_COLOUR_WIN32` | `-DCATCH_CONFIG_NO_COLOUR_WIN32` |      Result |
|-------------------------------|----------------------------------|-------------|
|                          `ON` |                             `ON` |       error |
|                          `ON` |                            `OFF` |    force-on |
|                         `OFF` |                             `ON` |   force-off |
|                         `OFF` |                            `OFF` | auto-detect |



## Installing Catch2 from git repository

If you cannot install Catch2 from a package manager (e.g. Ubuntu 16.04
provides catch only in version 1.2.0) you might want to install it from
the repository instead. Assuming you have enough rights, you can just
install it to the default location, like so:
```
$ git clone https://github.com/catchorg/Catch2.git
$ cd Catch2
$ cmake -B build -S . -DBUILD_TESTING=OFF
$ sudo cmake --build build/ --target install
```

If you do not have superuser rights, you will also need to specify
[CMAKE_INSTALL_PREFIX](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)
when configuring the build, and then modify your calls to
[find_package](https://cmake.org/cmake/help/latest/command/find_package.html)
accordingly.

## Installing Catch2 from vcpkg

Alternatively, you can build and install Catch2 using [vcpkg](https://github.com/microsoft/vcpkg/) dependency manager:
```
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
./vcpkg install catch2
```

The catch2 port in vcpkg is kept up to date by microsoft team members and community contributors.
If the version is out of date, please [create an issue or pull request](https://github.com/Microsoft/vcpkg) on the vcpkg repository.

## Installing Catch2 from Bazel

Catch2 is now a supported module in the Bazel Central Registry. You only need to add one line to your MODULE.bazel file;
please see https://registry.bazel.build/modules/catch2 for the latest supported version.

You can then add `catch2_main` to each of your C++ test build rules as follows:

```
cc_test(
    name = "example_test",
    srcs = ["example_test.cpp"],
    deps = [
        ":example",
        "@catch2//:catch2_main",
    ],
)
```

---

[Home](Readme.md#top)
