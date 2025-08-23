<a id="top"></a>
# Thread safety in Catch2

**Contents**<br>
[Using assertion macros from spawned threads](#using-assertion-macros-from-spawned-threads)<br>
[Assertion-like message macros and spawned threads](#assertion-like-message-macros-and-spawned-threads)<br>
[Message macros and spawned threads](#message-macros-and-spawned-threads)<br>
[examples](#examples)<br>
[`STATIC_REQUIRE` and `STATIC_CHECK`](#static_require-and-static_check)<br>
[Fatal errors and multiple threads](#fatal-errors-and-multiple-threads)<br>
[Performance overhead](#performance-overhead)<br>

> Thread safe assertions were introduced in Catch2 3.9.0

Thread safety in Catch2 is currently limited to all the assertion macros,
and to message or message-adjacent macros (e.g. `INFO` or `WARN`).

Interacting with benchmark macros, sections macros, generator macros, or
test case macros is not thread-safe. The way sections define paths through
the test is incompatible with user spawning threads arbitrarily, so this
limitation is here to stay.

**Important: thread safety in Catch2 is [opt-in](configuration.md#experimental-thread-safety)**


## Using assertion macros from spawned threads

The full set of Catch2's runtime assertion macros is thread-safe. However,
it is important to keep in mind that their semantics might not support
being used from user-spawned threads.

Specifically, the `REQUIRE` family of assertion macros have semantics
of stopping the test execution on failure. This is done by throwing
an exception, but since the user-spawned thread will not have the test-level
try-catch block ready to catch the test failure exception, failing a
`REQUIRE` assertion inside user-spawned thread will terminate the process.

The `CHECK` family of assertions does not have this issue, because it
does not try to stop the test execution.

Note that `CHECKED_IF` and `CHECKED_ELSE` are also thread safe (internally
they are assertion macro + an if).


## Assertion-like message macros and spawned threads

Similarly to assertion macros, not all assertion-like message macros can
be used from spawned thread.

`SKIP` and `FAIL` macros stop the test execution. Just like with `REQUIRE`,
this means that they cannot be used inside user-spawned threads. `SUCCEED`,
`FAIL_CHECK` and `WARN` do not attempt to stop the test execution and
thus can be used from any thread.


## Message macros and spawned threads

Macros that add extra messages to following assertion, such as `INFO`
or `CAPTURE`, are all thread safe and can be used in any thread. Note
that these messages are per-thread, and thus `INFO` inside a user-spawned
thread will not be seen by the main thread, and vice versa.


## examples

### `REQUIRE` from the main thread, `CHECK` from spawned threads

```cpp
TEST_CASE( "Failed REQUIRE in the main thread is fine" ) {
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
TEST_CASE( "Failed REQUIRE in spawned thread kills the process" ) {
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


### INFO across threads

```cpp
TEST_CASE( "messages don't cross threads" ) {
    std::jthread t1( [&]() {
        for ( size_t i = 0; i < 100; ++i ) {
            INFO( "spawned thread #1" );
            CHECK( 1 == 1 );
        }
    } );

    std::thread t2( [&]() {
        for (size_t i = 0; i < 100; ++i) {
            UNSCOPED_INFO( "spawned thread #2" );
        }
    } );

    for (size_t i = 0; i < 100; ++i) {
        CHECK( 1 == 2 );
    }
}
```
None of the failed checks will show the "spawned thread #1" message, as
that message is for the `t1` thread. If the reporter shows passing
assertions (e.g. due to the tests being run with `-s`), you will see the
"spawned thread #1" message alongside the passing `CHECK( 1 == 1 )` assertion.

The message "spawned thread #2" will never be shown, because there are no
assertions in `t2`.


### FAIL/SKIP from the main thread

```cpp
TEST_CASE( "FAIL in the main thread is fine" ) {
    std::vector<std::jthread> threads;
    for ( size_t t = 0; t < 16; ++t) {
        threads.emplace_back( []() {
            for (size_t i = 0; i < 10; ++i) {
                CHECK( true );
                CHECK( false );
            }
        } );
    }

    FAIL();
}
```

This will work as expected, that is, the process will finish running
normally, the test case will fail and there will be 321 total assertions,
160 passing and 161 failing (`FAIL` counts as failed assertion).

However, when the main thread hits `FAIL`, it will wait for the other
threads to finish due to `std::jthread`'s destructor joining the spawned
thread. Due to this, using `SKIP` is not recommended once more threads
are spawned; while the main thread will bail from the test execution,
the spawned threads will keep running and may fail the test case.


### FAIL/SKIP from spawned threads

```cpp
TEST_CASE( "FAIL/SKIP in spawned thread kills the process" ) {
    std::vector<std::jthread> threads;
    for ( size_t t = 0; t < 16; ++t) {
        threads.emplace_back( []() {
            for (size_t i = 0; i < 10'000; ++i) {
                FAIL();
            }
        } );
    }
}
```
As with failing `REQUIRE`, both `FAIL` and `SKIP` in spawned threads
terminate the process.


## `STATIC_REQUIRE` and `STATIC_CHECK`

All of `STATIC_REQUIRE`, `STATIC_REQUIRE_FALSE`, `STATIC_CHECK`, and
`STATIC_CHECK_FALSE` are thread safe in the delayed evaluation configuration.


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
