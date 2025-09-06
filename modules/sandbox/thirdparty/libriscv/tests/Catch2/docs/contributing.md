<a id="top"></a>
# Contributing to Catch2

**Contents**<br>
[Using Git(Hub)](#using-github)<br>
[Testing your changes](#testing-your-changes)<br>
[Writing documentation](#writing-documentation)<br>
[Writing code](#writing-code)<br>
[CoC](#coc)<br>

So you want to contribute something to Catch2? That's great! Whether it's
a bug fix, a new feature, support for additional compilers - or just
a fix to the documentation - all contributions are very welcome and very
much appreciated. Of course so are bug reports, other comments, and
questions, but generally it is a better idea to ask questions in our
[Discord](https://discord.gg/4CWS9zD), than in the issue tracker.


This page covers some guidelines and helpful tips for contributing
to the codebase itself.

## Using Git(Hub)

Ongoing development happens in the `devel` branch for Catch2 v3, and in
`v2.x` for maintenance updates to the v2 versions.

Commits should be small and atomic. A commit is atomic when, after it is
applied, the codebase, tests and all, still works as expected. Small
commits are also preferred, as they make later operations with git history,
whether it is bisecting, reverting, or something else, easier.

_When submitting a pull request please do not include changes to the
amalgamated distribution files. This means do not include them in your
git commits!_

When addressing review comments in a MR, please do not rebase/squash the
commits immediately. Doing so makes it harder to review the new changes,
slowing down the process of merging a MR. Instead, when addressing review
comments, you should append new commits to the branch and only squash
them into other commits when the MR is ready to be merged. We recommend
creating new commits with `git commit --fixup` (or `--squash`) and then
later squashing them with `git rebase --autosquash` to make things easier.



## Testing your changes

_Note: Running Catch2's tests requires Python3_


Catch2 has multiple layers of tests that are then run as part of our CI.
The most obvious one are the unit tests compiled into the `SelfTest`
binary. These are then used in "Approval tests", which run (almost) all
tests from `SelfTest` through a specific reporter and then compare the
generated output with a known good output ("Baseline"). By default, new
tests should be placed here.

To configure a Catch2 build with just the basic tests, use the `basic-tests`
preset, like so:

```
# Assuming you are in Catch2's root folder

cmake -B basic-test-build -S . -DCMAKE_BUILD_TYPE=Debug --preset basic-tests
```

However, not all tests can be written as plain unit tests. For example,
checking that Catch2 orders tests randomly when asked to, and that this
random ordering is subset-invariant, is better done as an integration
test using an external check script. Catch2 integration tests are written
using CTest, either as a direct command invocation + pass/fail regex,
or by delegating the check to a Python script.

Catch2 is slowly gaining more and more types of tests, currently Catch2
project also has buildable examples, "ExtraTests", and CMake config tests.
Examples present a small and self-contained snippets of code that
use Catch2's facilities for specific purpose. Currently they are assumed
passing if they compile.

ExtraTests then are expensive tests, that we do not want to run all the
time. This can be either because they take a long time to run, or because
they take a long time to compile, e.g. because they test compile time
configuration and require separate compilation.

Finally, CMake config tests test that you set Catch2's compile-time
configuration options through CMake, using CMake options of the same name.

These test categories can be enabled one by one, by passing
`-DCATCH_BUILD_EXAMPLES=ON`, `-DCATCH_BUILD_EXTRA_TESTS=ON`, and
`-DCATCH_ENABLE_CONFIGURE_TESTS=ON` when configuring the build.

Catch2 also provides a preset that promises to enable _all_ test types,
`all-tests`.

The snippet below will build & run all tests, in `Debug` compilation mode.

<!-- snippet: catch2-build-and-test -->
<a id='snippet-catch2-build-and-test'></a>
```sh
# 1. Regenerate the amalgamated distribution (some tests are built against it)
./tools/scripts/generateAmalgamatedFiles.py

# 2. Configure the full test build
cmake -B debug-build -S . -DCMAKE_BUILD_TYPE=Debug --preset all-tests

# 3. Run the actual build
cmake --build debug-build

# 4. Run the tests using CTest
ctest -j 4 --output-on-failure -C Debug --test-dir debug-build
```
<sup><a href='/tools/scripts/buildAndTest.sh#L6-L19' title='File snippet `catch2-build-and-test` was extracted from'>snippet source</a> | <a href='#snippet-catch2-build-and-test' title='Navigate to start of snippet `catch2-build-and-test`'>anchor</a></sup>
<!-- endSnippet -->

For convenience, the above commands are in the script `tools/scripts/buildAndTest.sh`, and can be run like this:

```bash
cd Catch2
./tools/scripts/buildAndTest.sh
```

A Windows version of the script is available at `tools\scripts\buildAndTest.cmd`.

If you added new tests, you will likely see `ApprovalTests` failure.
After you check that the output difference is expected, you should
run `tools/scripts/approve.py` to confirm them, and include these changes
in your commit.


## Writing documentation

If you have added new feature to Catch2, it needs documentation, so that
other people can use it as well. This section collects some technical
information that you will need for updating Catch2's documentation, and
possibly some generic advise as well.


### Technicalities

First, the technicalities:

* If you have introduced a new document, there is a simple template you
should use. It provides you with the top anchor mentioned to link to
(more below), and also with a backlink to the top of the documentation:
```markdown
<a id="top"></a>
# Cool feature

> [Introduced](https://github.com/catchorg/Catch2/pull/123456) in Catch2 X.Y.Z

Text that explains how to use the cool feature.


---

[Home](Readme.md#top)
```

* Crosslinks to different pages should target the `top` anchor, like this
`[link to contributing](contributing.md#top)`.

* We introduced version tags to the documentation, which show users in
which version a specific feature was introduced. This means that newly
written documentation should be tagged with a placeholder, that will
be replaced with the actual version upon release. There are 2 styles
of placeholders used through the documentation, you should pick one that
fits your text better (if in doubt, take a look at the existing version
tags for other features).
  * `> [Introduced](link-to-issue-or-PR) in Catch2 X.Y.Z` - this
  placeholder is usually used after a section heading
  * `> X (Y and Z) was [introduced](link-to-issue-or-PR) in Catch2 X.Y.Z`
  - this placeholder is used when you need to tag a subpart of something,
  e.g. a list

* For pages with more than 4 subheadings, we provide a table of contents
(ToC) at the top of the page. Because GitHub markdown does not support
automatic generation of ToC, it has to be handled semi-manually. Thus,
if you've added a new subheading to some page, you should add it to the
ToC. This can be done either manually, or by running the
`updateDocumentToC.py` script in the `scripts/` folder.

### Contents

Now, for some content tips:

* Usage examples are good. However, having large code snippets inline
can make the documentation less readable, and so the inline snippets
should be kept reasonably short. To provide more complex compilable
examples, consider adding new .cpp file to `examples/`.

* Don't be afraid to introduce new pages. The current documentation
tends towards long pages, but a lot of that is caused by legacy, and
we know that some of the pages are overly big and unfocused.

* When adding information to an existing page, please try to keep your
formatting, style and changes consistent with the rest of the page.

* Any documentation has multiple different audiences, that desire
different information from the text. The 3 basic user-types to try and
cover are:
  * A beginner to Catch2, who requires closer guidance for the usage of Catch2.
  * Advanced user of Catch2, who want to customize their usage.
  * Experts, looking for full reference of Catch2's capabilities.


## Writing code

If want to contribute code, this section contains some simple rules
and tips on things like code formatting, code constructions to avoid,
and so on.

### C++ standard version

Catch2 currently targets C++14 as the minimum supported C++ version.
Features from higher language versions should be used only sparingly,
when the benefits from using them outweigh the maintenance overhead.

Example of good use of polyfilling features is our use of `conjunction`,
where if available we use `std::conjunction` and otherwise provide our
own implementation. The reason it is good is that the surface area for
maintenance is quite small, and `std::conjunction` can directly use
compiler built-ins, thus providing significant compilation benefits.

Example of bad use of polyfilling features would be to keep around two
sets of metaprogramming in the stringification implementation, once
using C++14 compliant TMP and once using C++17's `if constexpr`. While
the C++17 would provide significant compilation speedups, the maintenance
cost would be too high.


### Formatting

To make code formatting simpler for the contributors, Catch2 provides
its own config for `clang-format`. However, because it is currently
impossible to replicate existing Catch2's formatting in clang-format,
using it to reformat a whole file would cause massive diffs. To keep
the size of your diffs reasonable, you should only use clang-format
on the newly changed code.


### Code constructs to watch out for

This section is a (sadly incomplete) listing of various constructs that
are problematic and are not always caught by our CI infrastructure.


#### Naked exceptions and exceptions-related function

If you are throwing an exception, it should be done via `CATCH_ERROR`
or `CATCH_RUNTIME_ERROR` in `internal/catch_enforce.hpp`. These macros will handle
the differences between compilation with or without exceptions for you.
However, some platforms (IAR) also have problems with exceptions-related
functions, such as `std::current_exceptions`. We do not have IAR in our
CI, but luckily there should not be too many reasons to use these.
However, if you do, they should be kept behind a
`CATCH_CONFIG_DISABLE_EXCEPTIONS` macro.


#### Avoid `std::move` and `std::forward`

`std::move` and `std::forward` provide nice semantic name for a specific
`static_cast`. However, being function templates they have surprisingly
high cost during compilation, and can also have a negative performance
impact for low-optimization builds.

You should be using `CATCH_MOVE` and `CATCH_FORWARD` macros from
`internal/catch_move_and_forward.hpp` instead. They expand into the proper
`static_cast`, and avoid the overhead of `std::move` and `std::forward`.


#### Unqualified usage of functions from C's stdlib

If you are using a function from C's stdlib, please include the header
as `<cfoo>` and call the function qualified. The common knowledge that
there is no difference is wrong, QNX and VxWorks won't compile if you
include the header as `<cfoo>` and call the function unqualified.


#### User-Defined Literals (UDL) for Catch2' types

Due to messy standardese and ... not great ... implementation of
`-Wreserved-identifier` in Clang, avoid declaring UDLs as
```cpp
Approx operator "" _a(long double);
```
and instead declare them as
```cpp
Approx operator ""_a(long double);
```

Notice that the second version does not have a space between the `""` and
the literal suffix.



### New source file template

If you are adding new source file, there is a template you should use.
Specifically, every source file should start with the licence header:
```cpp

    //              Copyright Catch2 Authors
    // Distributed under the Boost Software License, Version 1.0.
    //   (See accompanying file LICENSE.txt or copy at
    //        https://www.boost.org/LICENSE_1_0.txt)

    // SPDX-License-Identifier: BSL-1.0
```

The include guards for header files should follow the pattern `{FILENAME}_INCLUDED`.
This means that for file `catch_matchers_foo.hpp`, the include guard should
be `CATCH_MATCHERS_FOO_HPP_INCLUDED`, for `catch_generators_bar.hpp`, the include
guard should be `CATCH_GENERATORS_BAR_HPP_INCLUDED`, and so on.


### Adding new `CATCH_CONFIG` option

When adding new `CATCH_CONFIG` option, there are multiple places to edit:
  * `CMake/CatchConfigOptions.cmake` - this is used to generate the
    configuration options in CMake, so that CMake frontends know about them.
  * `docs/configuration.md` - this is where the options are documented
  * `src/catch2/catch_user_config.hpp.in` - this is template for generating
    `catch_user_config.hpp` which contains the materialized configuration
  * `BUILD.bazel` - Bazel does not have configuration support like CMake,
    and all expansions need to be done manually
  * other files as needed, e.g. `catch2/internal/catch_config_foo.hpp`
    for the logic that guards the configuration


## CoC

This project has a [CoC](../CODE_OF_CONDUCT.md). Please adhere to it
while contributing to Catch2.

-----------

_This documentation will always be in-progress as new information comes
up, but we are trying to keep it as up to date as possible._

---

[Home](Readme.md#top)
