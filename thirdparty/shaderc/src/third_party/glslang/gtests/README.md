Glslang Tests based on the Google Test Framework
================================================

This directory contains [Google Test][gtest] based test fixture and test
cases for glslang.

Apart from typical unit tests, necessary utility methods are added into
the [`GlslangTests`](TestFixture.h) fixture to provide the ability to do
file-based integration tests. Various `*.FromFile.cpp` files lists names
of files containing input shader code in the `Test/` directory. Utility
methods will load the input shader source, compile them, and compare with
the corresponding expected output in the `Test/baseResults/` directory.

How to run the tests
--------------------

Please make sure you have a copy of [Google Test][gtest] checked out under
the `External` directory before building. After building, just run the
`ctest` command or the `gtests/glslangtests` binary in your build directory.

The `gtests/glslangtests` binary also provides an `--update-mode` command
line option, which, if supplied, will overwrite the golden files under
the `Test/baseResults/` directory with real output from that invocation.
This serves as an easy way to update golden files.

[gtest]: https://github.com/google/googletest
