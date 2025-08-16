<a id="top"></a>
# Reporter events

**Contents**<br>
[Test running events](#test-running-events)<br>
[Benchmarking events](#benchmarking-events)<br>
[Listings events](#listings-events)<br>
[Miscellaneous events](#miscellaneous-events)<br>

Reporter events are one of the customization points for user code. They
are used by [reporters](reporters.md#top) to customize Catch2's output,
and by [event listeners](event-listeners.md#top) to perform in-process
actions under some conditions.

There are currently 21 reporter events in Catch2, split between 4 distinct
event groups:
* test running events (10 events)
* benchmarking (4 events)
* listings (3 events)
* miscellaneous (4 events)

## Test running events

Test running events are always paired so that for each `fooStarting` event,
there is a `fooEnded` event. This means that the 10 test running events
consist of 5 pairs of events:

* `testRunStarting` and `testRunEnded`,
* `testCaseStarting` and `testCaseEnded`,
* `testCasePartialStarting` and `testCasePartialEnded`,
* `sectionStarting` and `sectionEnded`,
* `assertionStarting` and `assertionEnded`

### `testRun` events

```cpp
void testRunStarting( TestRunInfo const& testRunInfo );
void testRunEnded( TestRunStats const& testRunStats );
```

The `testRun` events bookend the entire test run. `testRunStarting` is
emitted before the first test case is executed, and `testRunEnded` is
emitted after all the test cases have been executed.

### `testCase` events

```cpp
void testCaseStarting( TestCaseInfo const& testInfo );
void testCaseEnded( TestCaseStats const& testCaseStats );
```

The `testCase` events bookend one _full_ run of a specific test case.
Individual runs through a test case, e.g. due to `SECTION`s or `GENERATE`s,
are handled by a different event.


### `testCasePartial` events

> Introduced in Catch2 3.0.1

```cpp
void testCasePartialStarting( TestCaseInfo const& testInfo, uint64_t partNumber );
void testCasePartialEnded(TestCaseStats const& testCaseStats, uint64_t partNumber );
```

`testCasePartial` events bookend one _partial_ run of a specific test case.
This means that for any given test case, these events can be emitted
multiple times, e.g. due to multiple leaf sections.

In regards to nesting with `testCase` events, `testCasePartialStarting`
will never be emitted before the corresponding `testCaseStarting`, and
`testCasePartialEnded` will always be emitted before the corresponding
`testCaseEnded`.


### `section` events

```cpp
void sectionStarting( SectionInfo const& sectionInfo );
void sectionEnded( SectionStats const& sectionStats );
```

`section` events are emitted only for active `SECTION`s, that is, sections
that are entered. Sections that are skipped in this test case run-through
do not cause events to be emitted.

_Note that test cases always contain one implicit section. The event for
this section is emitted after the corresponding `testCasePartialStarting`
event._


### `assertion` events

```cpp
void assertionStarting( AssertionInfo const& assertionInfo );
void assertionEnded( AssertionStats const& assertionStats );
```

The `assertionStarting` event is emitted before the expression in the
assertion is captured or evaluated and `assertionEnded` is emitted
afterwards. This means that given assertion like `REQUIRE(a + b == c + d)`,
Catch2 first emits `assertionStarting` event, then `a + b` and `c + d`
are evaluated, then their results are captured, the comparison is evaluated,
and then `assertionEnded` event is emitted.


## Benchmarking events

> [Introduced](https://github.com/catchorg/Catch2/issues/1616) in Catch2 2.9.0.

```cpp
void benchmarkPreparing( StringRef name ) override;
void benchmarkStarting( BenchmarkInfo const& benchmarkInfo ) override;
void benchmarkEnded( BenchmarkStats<> const& benchmarkStats ) override;
void benchmarkFailed( StringRef error ) override;
```

Due to the benchmark lifecycle being bit more complicated, the benchmarking
events have their own category, even though they could be seen as parallel
to the `assertion*` events. You should expect running a benchmark to
generate at least 2 of the events above.

To understand the explanation below, you should read the [benchmarking
documentation](benchmarks.md#top) first.

* `benchmarkPreparing` event is sent after the environmental probe
finishes, but before the user code is first estimated.
* `benchmarkStarting` event is sent after the user code is estimated,
but has not been benchmarked yet.
* `benchmarkEnded` event is sent after the user code has been benchmarked,
and contains the benchmarking results.
* `benchmarkFailed` event is sent if either the estimation or the
benchmarking itself fails.


## Listings events

> Introduced in Catch2 3.0.1.

Listings events are events that correspond to the test binary being
invoked with `--list-foo` flag.

There are currently 3 listing events, one for reporters, one for tests,
and one for tags. Note that they are not exclusive to each other.

```cpp
void listReporters( std::vector<ReporterDescription> const& descriptions );
void listTests( std::vector<TestCaseHandle> const& tests );
void listTags( std::vector<TagInfo> const& tagInfos );
```


## Miscellaneous events

```cpp
void reportInvalidTestSpec( StringRef unmatchedSpec );
void fatalErrorEncountered( StringRef error );
void noMatchingTestCases( StringRef unmatchedSpec );
```

These are one-off events that do not neatly fit into other categories.

`reportInvalidTestSpec` is sent for each [test specification command line
argument](command-line.md#specifying-which-tests-to-run) that wasn't
parsed into a valid spec.

`fatalErrorEncountered` is sent when Catch2's POSIX signal handling
or Windows SE handler is called into with a fatal signal/exception.

`noMatchingTestCases` is sent for each user provided test specification
that did not match any registered tests.

---

[Home](Readme.md#top)
