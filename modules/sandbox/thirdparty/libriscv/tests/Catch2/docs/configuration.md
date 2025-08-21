<a id="top"></a>
# Compile-time configuration

**Contents**<br>
[Prefixing Catch macros](#prefixing-catch-macros)<br>
[Terminal colour](#terminal-colour)<br>
[Console width](#console-width)<br>
[stdout](#stdout)<br>
[Fallback stringifier](#fallback-stringifier)<br>
[Default reporter](#default-reporter)<br>
[Bazel support](#bazel-support)<br>
[C++11 toggles](#c11-toggles)<br>
[C++17 toggles](#c17-toggles)<br>
[Other toggles](#other-toggles)<br>
[Enabling stringification](#enabling-stringification)<br>
[Disabling exceptions](#disabling-exceptions)<br>
[Disabling deprecation warnings](#disabling-deprecation-warnings)<br>
[Overriding Catch's debug break (`-b`)](#overriding-catchs-debug-break--b)<br>
[Static analysis support](#static-analysis-support)<br>
[Experimental thread safety](#experimental-thread-safety)<br>

Catch2 is designed to "just work" as much as possible, and most of the
configuration options below are changed automatically during compilation,
according to the detected environment. However, this detection can also
be overridden by users, using macros documented below, and/or CMake options
with the same name.


## Prefixing Catch macros

    CATCH_CONFIG_PREFIX_ALL       // Prefix all macros with CATCH_
    CATCH_CONFIG_PREFIX_MESSAGES  // Prefix only INFO, UNSCOPED_INFO, WARN and CAPTURE

To keep test code clean and uncluttered Catch uses short macro names (e.g. ```TEST_CASE``` and ```REQUIRE```). Occasionally these may conflict with identifiers from platform headers or the system under test. In this case the above identifier can be defined. This will cause all the Catch user macros to be prefixed with ```CATCH_``` (e.g. ```CATCH_TEST_CASE``` and ```CATCH_REQUIRE```).


## Terminal colour

    CATCH_CONFIG_COLOUR_WIN32     // Force enables compiling colouring impl based on Win32 console API
    CATCH_CONFIG_NO_COLOUR_WIN32  // Force disables ...

Yes, Catch2 uses the british spelling of colour.

Catch2 attempts to autodetect whether the Win32 console colouring API,
`SetConsoleTextAttribute`, is available, and if it is available it compiles
in a console colouring implementation that uses it.

This option can be used to override Catch2's autodetection and force the
compilation either ON or OFF.


## Console width

    CATCH_CONFIG_CONSOLE_WIDTH = x // where x is a number

Catch formats output intended for the console to fit within a fixed number of characters. This is especially important as indentation is used extensively and uncontrolled line wraps break this.
By default a console width of 80 is assumed but this can be controlled by defining the above identifier to be a different value.

## stdout

    CATCH_CONFIG_NOSTDOUT

To support platforms that do not provide `std::cout`, `std::cerr` and
`std::clog`, Catch does not use them directly, but rather calls
`Catch::cout`, `Catch::cerr` and `Catch::clog`. You can replace their
implementation by defining `CATCH_CONFIG_NOSTDOUT` and implementing
them yourself, their signatures are:

    std::ostream& cout();
    std::ostream& cerr();
    std::ostream& clog();

[You can see an example of replacing these functions here.](
../examples/231-Cfg-OutputStreams.cpp)


## Fallback stringifier

By default, when Catch's stringification machinery has to stringify
a type that does not specialize `StringMaker`, does not overload `operator<<`,
is not an enumeration and is not a range, it uses `"{?}"`. This can be
overridden by defining `CATCH_CONFIG_FALLBACK_STRINGIFIER` to name of a
function that should perform the stringification instead.

All types that do not provide `StringMaker` specialization or `operator<<`
overload will be sent to this function (this includes enums and ranges).
The provided function must return `std::string` and must accept any type,
e.g. via overloading.

_Note that if the provided function does not handle a type and this type
requires to be stringified, the compilation will fail._


## Default reporter

Catch's default reporter can be changed by defining macro
`CATCH_CONFIG_DEFAULT_REPORTER` to string literal naming the desired
default reporter.

This means that defining `CATCH_CONFIG_DEFAULT_REPORTER` to `"console"`
is equivalent with the out-of-the-box experience.


## Bazel support

Compiling Catch2 with `CATCH_CONFIG_BAZEL_SUPPORT` force-enables Catch2's
support for Bazel's environment variables (normally Catch2 looks for
`BAZEL_TEST=1` env var first).

This can be useful if you are using older versions of Bazel, that do not
yet have `BAZEL_TEST` env var support.

> `CATCH_CONFIG_BAZEL_SUPPORT` was [introduced](https://github.com/catchorg/Catch2/pull/2399) in Catch2 3.0.1.

> `CATCH_CONFIG_BAZEL_SUPPORT` was [deprecated](https://github.com/catchorg/Catch2/pull/2459) in Catch2 3.1.0.


## C++11 toggles

    CATCH_CONFIG_CPP11_TO_STRING // Use `std::to_string`

Because we support platforms whose standard library does not contain
`std::to_string`, it is possible to force Catch to use a workaround
based on `std::stringstream`. On platforms other than Android,
the default is to use `std::to_string`. On Android, the default is to
use the `stringstream` workaround. As always, it is possible to override
Catch's selection, by defining either `CATCH_CONFIG_CPP11_TO_STRING` or
`CATCH_CONFIG_NO_CPP11_TO_STRING`.


## C++17 toggles

    CATCH_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS  // Override std::uncaught_exceptions (instead of std::uncaught_exception) support detection
    CATCH_CONFIG_CPP17_STRING_VIEW          // Override std::string_view support detection (Catch provides a StringMaker specialization by default)
    CATCH_CONFIG_CPP17_VARIANT              // Override std::variant support detection (checked by CATCH_CONFIG_ENABLE_VARIANT_STRINGMAKER)
    CATCH_CONFIG_CPP17_OPTIONAL             // Override std::optional support detection (checked by CATCH_CONFIG_ENABLE_OPTIONAL_STRINGMAKER)
    CATCH_CONFIG_CPP17_BYTE                 // Override std::byte support detection (Catch provides a StringMaker specialization by default)

> `CATCH_CONFIG_CPP17_STRING_VIEW` was [introduced](https://github.com/catchorg/Catch2/issues/1376) in Catch2 2.4.1.

Catch contains basic compiler/standard detection and attempts to use
some C++17 features whenever appropriate. This automatic detection
can be manually overridden in both directions, that is, a feature
can be enabled by defining the macro in the table above, and disabled
by using `_NO_` in the macro, e.g. `CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS`.


## Other toggles

    CATCH_CONFIG_COUNTER                    // Use __COUNTER__ to generate unique names for test cases
    CATCH_CONFIG_WINDOWS_SEH                // Enable SEH handling on Windows
    CATCH_CONFIG_FAST_COMPILE               // Sacrifices some (rather minor) features for compilation speed
    CATCH_CONFIG_POSIX_SIGNALS              // Enable handling POSIX signals
    CATCH_CONFIG_WINDOWS_CRTDBG             // Enable leak checking using Windows's CRT Debug Heap
    CATCH_CONFIG_DISABLE_STRINGIFICATION    // Disable stringifying the original expression
    CATCH_CONFIG_DISABLE                    // Disables assertions and test case registration
    CATCH_CONFIG_WCHAR                      // Enables use of wchart_t
    CATCH_CONFIG_EXPERIMENTAL_REDIRECT      // Enables the new (experimental) way of capturing stdout/stderr
    CATCH_CONFIG_USE_ASYNC                  // Force parallel statistical processing of samples during benchmarking
    CATCH_CONFIG_ANDROID_LOGWRITE           // Use android's logging system for debug output
    CATCH_CONFIG_GLOBAL_NEXTAFTER           // Use nextafter{,f,l} instead of std::nextafter
    CATCH_CONFIG_GETENV                     // System has a working `getenv`
    CATCH_CONFIG_USE_BUILTIN_CONSTANT_P     // Use __builtin_constant_p to trigger warnings

> [`CATCH_CONFIG_ANDROID_LOGWRITE`](https://github.com/catchorg/Catch2/issues/1743) and [`CATCH_CONFIG_GLOBAL_NEXTAFTER`](https://github.com/catchorg/Catch2/pull/1739) were introduced in Catch2 2.10.0

> `CATCH_CONFIG_GETENV` was [introduced](https://github.com/catchorg/Catch2/pull/2562) in Catch2 3.2.0

> `CATCH_CONFIG_USE_BUILTIN_CONSTANT_P` was introduced in Catch2 3.8.0

Currently Catch enables `CATCH_CONFIG_WINDOWS_SEH` only when compiled with MSVC, because some versions of MinGW do not have the necessary Win32 API support.

`CATCH_CONFIG_POSIX_SIGNALS` is on by default, except when Catch is compiled under `Cygwin`, where it is disabled by default (but can be force-enabled by defining `CATCH_CONFIG_POSIX_SIGNALS`).

`CATCH_CONFIG_GETENV` is on by default, except when Catch2 is compiled for
platforms that lacks working `std::getenv` (currently Windows UWP and
Playstation).

`CATCH_CONFIG_WINDOWS_CRTDBG` is off by default. If enabled, Windows's
CRT is used to check for memory leaks, and displays them after the tests
finish running. This option only works when linking against the default
main, and must be defined for the whole library build.

`CATCH_CONFIG_WCHAR` is on by default, but can be disabled. Currently
it is only used in support for DJGPP cross-compiler.

With the exception of `CATCH_CONFIG_EXPERIMENTAL_REDIRECT`,
these toggles can be disabled by using `_NO_` form of the toggle,
e.g. `CATCH_CONFIG_NO_WINDOWS_SEH`.

`CATCH_CONFIG_USE_BUILTIN_CONSTANT_P` is ON by default for Clang and GCC
(but as far as possible, not for other compilers masquerading for these
two). However, it can cause bugs where the enclosed code is evaluated, even
though it should not be, e.g. in [#2925](https://github.com/catchorg/Catch2/issues/2925).


### `CATCH_CONFIG_FAST_COMPILE`
This compile-time flag speeds up compilation of assertion macros by ~20%,
by disabling the generation of assertion-local try-catch blocks for
non-exception family of assertion macros ({`REQUIRE`,`CHECK`}{``,`_FALSE`, `_THAT`}).
This disables translation of exceptions thrown under these assertions, but
should not lead to false negatives.

`CATCH_CONFIG_FAST_COMPILE` has to be either defined, or not defined,
in all translation units that are linked into single test binary.

### `CATCH_CONFIG_DISABLE_STRINGIFICATION`
This toggle enables a workaround for VS 2017 bug. For details see [known limitations](limitations.md#visual-studio-2017----raw-string-literal-in-assert-fails-to-compile).

### `CATCH_CONFIG_DISABLE`
This toggle removes most of Catch from given file. This means that `TEST_CASE`s are not registered and assertions are turned into no-ops. Useful for keeping tests within implementation files (ie for functions with internal linkage), instead of in external files.

This feature is considered experimental and might change at any point.

_Inspired by Doctest's `DOCTEST_CONFIG_DISABLE`_


## Enabling stringification

By default, Catch does not stringify some types from the standard library. This is done to avoid dragging in various standard library headers by default. However, Catch does contain these and can be configured to provide them, using these macros:

    CATCH_CONFIG_ENABLE_PAIR_STRINGMAKER     // Provide StringMaker specialization for std::pair
    CATCH_CONFIG_ENABLE_TUPLE_STRINGMAKER    // Provide StringMaker specialization for std::tuple
    CATCH_CONFIG_ENABLE_VARIANT_STRINGMAKER  // Provide StringMaker specialization for std::variant, std::monostate (on C++17)
    CATCH_CONFIG_ENABLE_OPTIONAL_STRINGMAKER // Provide StringMaker specialization for std::optional (on C++17)
    CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS     // Defines all of the above

> `CATCH_CONFIG_ENABLE_VARIANT_STRINGMAKER` was [introduced](https://github.com/catchorg/Catch2/issues/1380) in Catch2 2.4.1.

> `CATCH_CONFIG_ENABLE_OPTIONAL_STRINGMAKER` was [introduced](https://github.com/catchorg/Catch2/issues/1510) in Catch2 2.6.0.

## Disabling exceptions

> Introduced in Catch2 2.4.0.

By default, Catch2 uses exceptions to signal errors and to abort tests
when an assertion from the `REQUIRE` family of assertions fails. We also
provide an experimental support for disabling exceptions. Catch2 should
automatically detect when it is compiled with exceptions disabled, but
it can be forced to compile without exceptions by defining

    CATCH_CONFIG_DISABLE_EXCEPTIONS

Note that when using Catch2 without exceptions, there are 2 major
limitations:

1) If there is an error that would normally be signalled by an exception,
the exception's message will instead be written to `Catch::cerr` and
`std::terminate` will be called.
2) If an assertion from the `REQUIRE` family of macros fails,
`std::terminate` will be called after the active reporter returns.


There is also a customization point for the exact behaviour of what
happens instead of exception being thrown. To use it, define

    CATCH_CONFIG_DISABLE_EXCEPTIONS_CUSTOM_HANDLER

and provide a definition for this function:

```cpp
namespace Catch {
    [[noreturn]]
    void throw_exception(std::exception const&);
}
```


## Disabling deprecation warnings

> Introduced in Catch2 3.9.0

Catch2 has started using the C++ macro `[[deprecated]]` to mark things
that are deprecated and should not be used any more. If you need to
temporarily disable these warnings, use

    CATCH_CONFIG_NO_DEPRECATION_ANNOTATIONS

Catch2 currently does not support more fine-grained deprecation warning
control, nor do we plan to.


## Overriding Catch's debug break (`-b`)

> [Introduced](https://github.com/catchorg/Catch2/pull/1846) in Catch2 2.11.2.

You can override Catch2's break-into-debugger code by defining the
`CATCH_BREAK_INTO_DEBUGGER()` macro. This can be used if e.g. Catch2 does
not know your platform, or your platform is misdetected.

The macro will be used as is, that is, `CATCH_BREAK_INTO_DEBUGGER();`
must compile and must break into debugger.


## Static analysis support

> Introduced in Catch2 3.4.0.

Some parts of Catch2, e.g. `SECTION`s, can be hard for static analysis
tools to reason about. Catch2 can change its internals to help static
analysis tools reason about the tests.

Catch2 automatically detects some static analysis tools (initial
implementation checks for clang-tidy and Coverity), but you can override
its detection (in either direction) via

```
CATCH_CONFIG_EXPERIMENTAL_STATIC_ANALYSIS_SUPPORT     // force enables static analysis help
CATCH_CONFIG_NO_EXPERIMENTAL_STATIC_ANALYSIS_SUPPORT  // force disables static analysis help
```

_As the name suggests, this is currently experimental, and thus we provide
no backwards compatibility guarantees._

**DO NOT ENABLE THIS FOR BUILDS YOU INTEND TO RUN.** The changed internals
are not meant to be runnable, only "scannable".


## Experimental thread safety

> Introduced in Catch2 3.9.0

Catch2 can optionally support thread-safe assertions, that means, multiple
user-spawned threads can use the assertion macros at the same time. Due
to the performance cost this imposes even on single-threaded usage, Catch2
defaults to non-thread-safe assertions.

    CATCH_CONFIG_EXPERIMENTAL_THREAD_SAFE_ASSERTIONS     // enables thread safe assertions
    CATCH_CONFIG_NO_EXPERIMENTAL_THREAD_SAFE_ASSERTIONS  // force-disables thread safe assertions

See [the documentation on thread safety in Catch2](thread-safety.md#top)
for details on which macros are safe and other notes.


---

[Home](Readme.md#top)
