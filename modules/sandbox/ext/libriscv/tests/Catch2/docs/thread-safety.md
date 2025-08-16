<a id="top"></a>
# Thread safety in Catch2

**Contents**<br>
[Using assertion macros from multiple threads](#using-assertion-macros-from-multiple-threads)<br>
[examples](#examples)<br>
[`STATIC_REQUIRE` and `STATIC_CHECK`](#static_require-and-static_check)<br>
[Fatal errors and multiple threads](#fatal-errors-and-multiple-threads)<br>
[Performance overhead](#performance-overhead)<br>

> Thread safe assertions were introduced in Catch2 3.9.0

Thread safety in Catch2 is currently limited to all the assertion macros.
Interacting with benchmark macros, message macros (e.g. `INFO` or `CAPTURE`),
sections macros, generator macros, or test case macros is not thread-safe.
The message macros are likely to be made thread-safe in the future, but
the way sections define test runs is incompatible with user being able
to spawn threads arbitrarily, thus that limitation is here to stay.

**Important: thread safety in Catch2 is [opt-in](configuration.md#experimental-thread-safety)**


## Using assertion macros from multiple threads

The full set of Catch2's runtime assertion macros is thread-safe. However,
it is important to keep in mind that their semantics might not support
being used from user-spawned threads.

Specifically, the `REQUIRE` family of assertion macros have semantics
of stopping the test execution on failure. This is done by throwing
an exception, but since the user-spawned thread will not have the test-level
try-catch block ready to catch the test failure exception, failing a
`REQUIRE` assertion inside this thread will terminate the process.

The `CHECK` family of assertions does not have this issue, because it
does not try to stop the test execution.

Note that `CHECKED_IF` and `CHECKED_ELSE` are also thread safe (internally
they are assertion macro + an if).

**`SKIP()`, `FAIL()`, `SUCCEED()` are not assertion macros, and are not
thread-safe.**


## examples

### `REQUIRE` from main thread, `CHECK` from spawned threads

```cpp
TEST_CASE( "Failed REQUIRE in main thread is fine" ) {
    std::vector<std::jthread> threads;
    for ( size_t t = 0; t < 16; ++t) {
        threads.emplace_back( []() {
            for (size_t i = 0; i < 10'000; ++i) {
                CHECK( true );
                CHECK( false );
            }
        } );
    }

    REQUIRE( false );
}
```
This will work as expected, that is, the process will finish running
normally, the test case will fail and there will be the correct count of
passing and failing assertions (160000 and 160001 respectively). However,
it is important to understand that when the main thread fails its assertion,
the spawned threads will keep running.


### `REQUIRE` from spawned threads

```cpp
TEST_CASE( "Successful REQUIRE in spawned thread is fine" ) {
    std::vector<std::jthread> threads;
    for ( size_t t = 0; t < 16; ++t) {
        threads.emplace_back( []() {
            for (size_t i = 0; i < 10'000; ++i) {
                REQUIRE( true );
            }
        } );
    }
}
```
This will also work as expected, because the `REQUIRE` is successful.

```cpp
TEST_CASE( "Failed REQUIRE in spawned thread is fine" ) {
    std::vector<std::jthread> threads;
    for ( size_t t = 0; t < 16; ++t) {
        threads.emplace_back( []() {
            for (size_t i = 0; i < 10'000; ++i) {
                REQUIRE( false );
            }
        } );
    }
}
```
This will fail catastrophically and terminate the process.


## `STATIC_REQUIRE` and `STATIC_CHECK`

None of `STATIC_REQUIRE`, `STATIC_REQUIRE_FALSE`, `STATIC_CHECK`, and
`STATIC_CHECK_FALSE` are currently thread safe. This might be surprising
given that they are a compile-time checks, but they also rely on the
message macros to register the result with reporter at runtime.


## Fatal errors and multiple threads

By default, Catch2 tries to catch fatal errors (POSIX signals/Windows
Structured Exceptions) and report something useful to the user. This
always happened on a best-effort basis, but in presence of multiple
threads and locks the chance of it working decreases. If this starts
being an issue for you, [you can disable it](configuration.md#other-toggles).


## Performance overhead

In the worst case, which is optimized build and assertions using the
fast path for successful assertions, the performance overhead of using
the thread-safe assertion implementation can reach 40%. In other cases,
the overhead will be smaller, between 4% and 20%.



---

[Home](Readme.md#top)
