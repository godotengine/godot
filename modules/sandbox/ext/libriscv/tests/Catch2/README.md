<a id="top"></a>
![Catch2 logo](data/artwork/catch2-logo-full-with-background.svg)

[![Github Releases](https://img.shields.io/github/release/catchorg/catch2.svg)](https://github.com/catchorg/catch2/releases)
[![Linux build status](https://github.com/catchorg/Catch2/actions/workflows/linux-simple-builds.yml/badge.svg)](https://github.com/catchorg/Catch2/actions/workflows/linux-simple-builds.yml)
[![Linux build status](https://github.com/catchorg/Catch2/actions/workflows/linux-other-builds.yml/badge.svg)](https://github.com/catchorg/Catch2/actions/workflows/linux-other-builds.yml)
[![MacOS build status](https://github.com/catchorg/Catch2/actions/workflows/mac-builds.yml/badge.svg)](https://github.com/catchorg/Catch2/actions/workflows/mac-builds.yml)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/catchorg/Catch2?svg=true&branch=devel)](https://ci.appveyor.com/project/catchorg/catch2)
[![Code Coverage](https://codecov.io/gh/catchorg/Catch2/branch/devel/graph/badge.svg)](https://codecov.io/gh/catchorg/Catch2)
[![Try online](https://img.shields.io/badge/try-online-blue.svg)](https://godbolt.org/z/EdoY15q9G)
[![Join the chat in Discord: https://discord.gg/4CWS9zD](https://img.shields.io/badge/Discord-Chat!-brightgreen.svg)](https://discord.gg/4CWS9zD)


## What is Catch2?

Catch2 is mainly a unit testing framework for C++, but it also
provides basic micro-benchmarking features, and simple BDD macros.

Catch2's main advantage is that using it is both simple and natural.
Test names do not have to be valid identifiers, assertions look like
normal C++ boolean expressions, and sections provide a nice and local way
to share set-up and tear-down code in tests.

**Example unit test**
```cpp
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

uint32_t factorial( uint32_t number ) {
    return number <= 1 ? number : factorial(number-1) * number;
}

TEST_CASE( "Factorials are computed", "[factorial]" ) {
    REQUIRE( factorial( 1) == 1 );
    REQUIRE( factorial( 2) == 2 );
    REQUIRE( factorial( 3) == 6 );
    REQUIRE( factorial(10) == 3'628'800 );
}
```

**Example microbenchmark**
```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <cstdint>

uint64_t fibonacci(uint64_t number) {
    return number < 2 ? number : fibonacci(number - 1) + fibonacci(number - 2);
}

TEST_CASE("Benchmark Fibonacci", "[!benchmark]") {
    REQUIRE(fibonacci(5) == 5);

    REQUIRE(fibonacci(20) == 6'765);
    BENCHMARK("fibonacci 20") {
        return fibonacci(20);
    };

    REQUIRE(fibonacci(25) == 75'025);
    BENCHMARK("fibonacci 25") {
        return fibonacci(25);
    };
}
```

_Note that benchmarks are not run by default, so you need to run it explicitly
with the `[!benchmark]` tag._


## Catch2 v3 has been released!

You are on the `devel` branch, where the v3 version is being developed.
v3 brings a bunch of significant changes, the big one being that Catch2
is no longer a single-header library. Catch2 now behaves as a normal
library, with multiple headers and separately compiled implementation.

The documentation is slowly being updated to take these changes into
account, but this work is currently still ongoing.

For migrating from the v2 releases to v3, you should look at [our
documentation](docs/migrate-v2-to-v3.md#top). It provides a simple
guidelines on getting started, and collects most common migration
problems.

For the previous major version of Catch2 [look into the `v2.x` branch
here on GitHub](https://github.com/catchorg/Catch2/tree/v2.x).


## How to use it
This documentation comprises these three parts:

* [Why do we need yet another C++ Test Framework?](docs/why-catch.md#top)
* [Tutorial](docs/tutorial.md#top) - getting started
* [Reference section](docs/Readme.md#top) - all the details


## More
* Issues and bugs can be raised on the [Issue tracker on GitHub](https://github.com/catchorg/Catch2/issues)
* For discussion or questions please use [our Discord](https://discord.gg/4CWS9zD)
* See who else is using Catch2 in [Open Source Software](docs/opensource-users.md#top)
or [commercially](docs/commercial-users.md#top).
