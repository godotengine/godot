<a id="top"></a>
# Migrating from v2 to v3

v3 is the next major version of Catch2 and brings three significant changes:
 * Catch2 is now split into multiple headers
 * Catch2 is now compiled as a static library
 * C++14 is the minimum required C++ version

There are many reasons why we decided to go from the old single-header
distribution model to a more standard library distribution model. The
big one is compile-time performance, but moving over to a split header
distribution model also improves the future maintainability and
extendability of the codebase. For example v3 adds a new kind of matchers
without impacting the compilation times of users that do not use matchers
in their tests. The new model is also more friendly towards package
managers, such as vcpkg and Conan.

The result of this move is a significant improvement in compilation
times, e.g. the inclusion overhead of Catch2 in the common case has been
reduced by roughly 80%. The improved ease of maintenance also led to
various runtime performance improvements and the introduction of new features.
For details, look at [the release notes of 3.0.1](release-notes.md#301).

_Note that we still provide one header + one translation unit (TU)
distribution but do not consider it the primarily supported option. You
should also expect that the compilation times will be worse if you use
this option._


## How to migrate projects from v2 to v3

To migrate to v3, there are two basic approaches to do so.

1. Use `catch_amalgamated.hpp` and `catch_amalgamated.cpp`.
2. Build Catch2 as a proper (static) library, and move to piecewise headers

Doing 1 means downloading the [amalgamated header](/extras/catch_amalgamated.hpp)
and the [amalgamated sources](/extras/catch_amalgamated.cpp) from `extras`,
dropping them into your test project, and rewriting your includes from
`<catch2/catch.hpp>` to `"catch_amalgamated.hpp"` (or something similar,
based on how you set up your paths).

The disadvantage of using this approach are increased compilation times,
at least compared to the second approach, but it does let you avoid
dealing with consuming libraries in your build system of choice.


However, we recommend doing 2, and taking extra time to migrate to v3
properly. This lets you reap the benefits of significantly improved
compilation times in the v3 version. The basic steps to do so are:

1. Change your CMakeLists.txt to link against `Catch2WithMain` target if
you use Catch2's default main. (If you do not, keep linking against
the `Catch2` target.). If you use pkg-config, change `pkg-config catch2` to
`pkg-config catch2-with-main`.
2. Delete TU with `CATCH_CONFIG_RUNNER` or `CATCH_CONFIG_MAIN` defined,
as it is no longer needed.
3. Change `#include <catch2/catch.hpp>` to `#include <catch2/catch_all.hpp>`
4. Check that everything compiles. You might have to modify namespaces,
or perform some other changes (see the
[Things that can break during porting](#things-that-can-break-during-porting)
section for the most common things).
5. Start migrating your test TUs from including `<catch2/catch_all.hpp>`
to piecemeal includes. You will likely want to start by including
`<catch2/catch_test_macros.hpp>`, and then go from there. (see
[other notes](#other-notes) for further ideas)

## Other notes

* The main test include is now `<catch2/catch_test_macros.hpp>`

* Big "subparts" like Matchers, or Generators, have their own folder, and
also their own "big header", so if you just want to include all matchers,
you can include `<catch2/matchers/catch_matchers_all.hpp>`,
or `<catch2/generators/catch_generators_all.hpp>`


## Things that can break during porting

* The namespaces of Matchers were flattened and cleaned up.

Matchers are no longer declared deep within an internal namespace and
then brought up into `Catch` namespace. All Matchers now live in the
`Catch::Matchers` namespace.

* The `Contains` string matcher was renamed to `ContainsSubstring`.

* The reporter interfaces changed in a breaking manner.

If you are using a custom reporter or listener, you will likely need to
modify them to conform to the new interfaces. Unlike before in v2,
the [interfaces](reporters.md#top) and the [events](reporter-events.md#top)
are now documented.


---

[Home](Readme.md#top)
