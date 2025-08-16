<a id="top"></a>
# Explicitly skipping, passing, and failing tests at runtime

## Skipping Test Cases at Runtime

> [Introduced](https://github.com/catchorg/Catch2/pull/2360) in Catch2 3.3.0.

In some situations it may not be possible to meaningfully execute a test case,
for example when the system under test is missing certain hardware capabilities.
If the required conditions can only be determined at runtime, it often
doesn't make sense to consider such a test case as either passed or failed,
because it simply cannot run at all.

To properly express such scenarios, Catch2 provides a way to explicitly
_skip_ test cases, using the `SKIP` macro:

```
SKIP( [streamable expression] )
```

Example usage:

```c++
TEST_CASE("copy files between drives") {
    if(getNumberOfHardDrives() < 2) {
        SKIP("at least two hard drives required");
    }
    // ...
}
```

This test case is then reported as _skipped_ instead of _passed_ or _failed_.

The `SKIP` macro behaves similarly to an explicit [`FAIL`](#passing-and-failing-test-cases),
in that it is the last expression that will be executed:

```c++
TEST_CASE("my test") {
    printf("foo");
    SKIP();
    printf("bar"); // not printed
}
```

However a failed assertion _before_ a `SKIP` still causes the entire
test case to fail:

```c++
TEST_CASE("failing test") {
    CHECK(1 == 2);
    SKIP();
}
```

Same applies for a `SKIP` nested inside an assertion:

```cpp
static bool do_skip() {
    SKIP();
    return true;
}

TEST_CASE("Another failing test") {
    CHECK(do_skip());
}
```


### Interaction with Sections and Generators

Sections, nested sections as well as specific outputs from [generators](generators.md#top)
can all be individually skipped, with the rest executing as usual:

```c++
TEST_CASE("complex test case") {
  int value = GENERATE(2, 4, 6);
  SECTION("a") {
    SECTION("a1") { CHECK(value < 8); }
    SECTION("a2") {
      if (value == 4) {
        SKIP();
      }
      CHECK(value % 2 == 0);
    }
  }
}
```

This test case will report 5 passing assertions; one for each of the three
values in section `a1`, and then two in section `a2`, from values 2 and 4.

Note that as soon as one section is skipped, the entire test case will
be reported as _skipped_ (unless there is a failing assertion, in which
case the test is handled as _failed_ instead).

Note that if all test cases in a run are skipped, Catch2 returns a non-zero
exit code, same as it does if no test cases have run. This behaviour can
be overridden using the [--allow-running-no-tests](command-line.md#no-tests-override)
flag.

### `SKIP` inside generators

You can also use the `SKIP` macro inside generator's constructor to handle
cases where the generator is empty, but you do not want to fail the test
case.


## Passing and failing test cases

Test cases can also be explicitly passed or failed, without the use of
assertions, and with a specific message. This can be useful to handle
complex preconditions/postconditions and give useful error messages
when they fail.

* `SUCCEED( [streamable expression] )`

`SUCCEED` is morally equivalent with `INFO( [streamable expression] ); REQUIRE( true );`.
Note that it does not stop further test execution, so it cannot be used
to guard failing assertions from being executed.

_In practice, `SUCCEED` is usually used as a test placeholder, to avoid
[failing a test case due to missing assertions](command-line.md#warnings)._

```cpp
TEST_CASE( "SUCCEED showcase" ) {
    int I = 1;
    SUCCEED( "I is " << I );
    // ... execution continues here ...
}
```

* `FAIL( [streamable expression] )`

`FAIL` is morally equivalent with `INFO( [streamable expression] ); REQUIRE( false );`.

_In practice, `FAIL` is usually used to stop executing test that is currently
known to be broken, but has to be fixed later._

```cpp
TEST_CASE( "FAIL showcase" ) {
    FAIL( "This test case causes segfault, which breaks CI." );
    // ... this will not be executed ...
}
```


---

[Home](Readme.md#top)
