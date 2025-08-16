<a id="top"></a>
# Reporters

Reporters are a customization point for most of Catch2's output, e.g.
formatting and writing out [assertions (whether passing or failing),
sections, test cases, benchmarks, and so on](reporter-events.md#top).

Catch2 comes with a bunch of reporters by default (currently 9), and
you can also write your own reporter. Because multiple reporters can
be active at the same time, your own reporters do not even have to handle
all reporter event, just the ones you are interested in, e.g. benchmarks.


## Using different reporters

You can see which reporters are available by running the test binary
with `--list-reporters`. You can then pick one of them with the [`-r`,
`--reporter` option](command-line.md#choosing-a-reporter-to-use), followed
by the name of the desired reporter, like so:

```
--reporter xml
```

You can also select multiple reporters to be used at the same time.
In that case you should read the [section on using multiple
reporters](#multiple-reporters) to avoid any surprises from doing so.


<a id="multiple-reporters"></a>
## Using multiple reporters

> Support for having multiple parallel reporters was [introduced](https://github.com/catchorg/Catch2/pull/2183) in Catch2 3.0.1

Catch2 supports using multiple reporters at the same time while having
them write into different destinations. The two main uses of this are

* having both human-friendly and machine-parseable (e.g. in JUnit format)
  output from one run of binary
* having "partial" reporters that are highly specialized, e.g. having one
  reporter that writes out benchmark results as markdown tables and does
  nothing else, while also having standard testing output separately

Specifying multiple reporter looks like this:
```
--reporter JUnit::out=result-junit.xml --reporter console::out=-::colour-mode=ansi
```

This tells Catch2 to use two reporters, `JUnit` reporter that writes
its machine-readable XML output to file `result-junit.xml`, and the
`console` reporter that writes its user-friendly output to stdout and
uses ANSI colour codes for colouring the output.

Using multiple reporters (or one reporter and one-or-more [event
listeners](event-listeners.md#top)) can have surprisingly complex semantics
when using customization points provided to reporters by Catch2, namely
capturing stdout/stderr from test cases.

As long as at least one reporter (or listener) asks Catch2 to capture
stdout/stderr, captured stdout and stderr will be available to all
reporters and listeners.

Because this might be surprising to the users, if at least one active
_reporter_ is non-capturing, then Catch2 tries to roughly emulate
non-capturing behaviour by printing out the captured stdout/stderr
just before `testCasePartialEnded` event is sent out to the active
reporters and listeners. This means that stdout/stderr is no longer
printed out from tests as it is being written, but instead it is written
out in batch after each runthrough of a test case is finished.



## Writing your own reporter

You can also write your own custom reporter and tell Catch2 to use it.
When writing your reporter, you have two options:

* Derive from `Catch::ReporterBase`. When doing this, you will have
  to provide handling for all [reporter events](reporter-events.md#top).
* Derive from one of the provided [utility reporter bases in
  Catch2](#utility-reporter-bases).

Generally we recommend doing the latter, as it is less work.

Apart from overriding handling of the individual reporter events, reporters
have access to some extra customization points, described below.


### Utility reporter bases

Catch2 currently provides two utility reporter bases:

* `Catch::StreamingReporterBase`
* `Catch::CumulativeReporterBase`

`StreamingReporterBase` is useful for reporters that can format and write
out the events as they come in. It provides (usually empty) implementation
for all reporter events, and if you let it handle the relevant events,
it also handles storing information about active test run and test case.

`CumulativeReporterBase` is a base for reporters that need to see the whole
test run, before they can start writing the output, such as the JUnit
and SonarQube reporters. This post-facto approach requires the assertions
to be stringified when it is finished, so that the assertion can be written
out later. Because the stringification can be expensive, and not all
cumulative reporters need the assertions, this base provides customization
point to change whether the assertions are saved or not, separate for
passing and failing assertions.


_Generally we recommend that if you override a member function from either
of the bases, you call into the base's implementation first. This is not
necessarily in all cases, but it is safer and easier._


Writing your own reporter then looks like this:

```cpp
#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <iostream>

class PartialReporter : public Catch::StreamingReporterBase {
public:
    using StreamingReporterBase::StreamingReporterBase;

    static std::string getDescription() {
        return "Reporter for testing TestCasePartialStarting/Ended events";
    }

    void testCasePartialStarting(Catch::TestCaseInfo const& testInfo,
                                 uint64_t partNumber) override {
        std::cout << "TestCaseStartingPartial: " << testInfo.name << '#' << partNumber << '\n';
    }

    void testCasePartialEnded(Catch::TestCaseStats const& testCaseStats,
                              uint64_t partNumber) override {
        std::cout << "TestCasePartialEnded: " << testCaseStats.testInfo->name << '#' << partNumber << '\n';
    }
};


CATCH_REGISTER_REPORTER("partial", PartialReporter)
```

This create a simple reporter that responds to `testCasePartial*` events,
and calls itself "partial" reporter, so it can be invoked with
`--reporter partial` command line flag.


### `ReporterPreferences`

Each reporter instance contains instance of `ReporterPreferences`, a type
that holds flags for the behaviour of Catch2 when this reporter run.
Currently there are three customization options:

* `shouldRedirectStdOut` - whether the reporter wants to handle
   writes to stdout/stderr from user code, or not. This is useful for
   reporters that output machine-parseable output, e.g. the JUnit
   reporter, or the XML reporter.
* `shouldReportAllAssertions` - whether the reporter wants to handle
  `assertionEnded` events for passing assertions as well as failing
   assertions. Usually reporters do not report successful assertions
   and don't need them for their output, but sometimes the desired output
   format includes passing assertions even without the `-s` flag.
* `shouldReportAllAssertionStarts` - whether the reporter wants to handle
  `assertionStarting` events. Most reporters do not, and opting out
   explicitly enables a fast-path in Catch2's handling of assertions.

> `shouldReportAllAssertionStarts` was introduced in Catch2 3.9.0


### Per-reporter configuration

> Per-reporter configuration was introduced in Catch2 3.0.1

Catch2 supports some configuration to happen per reporter. The configuration
options fall into one of two categories:

* Catch2-recognized options
* Reporter-specific options

The former is a small set of universal options that Catch2 handles for
the reporters, e.g. output file or console colour mode. The latter are
options that the reporters have to handle themselves, but the keys and
values can be arbitrary strings, as long as they don't contain `::`. This
allows writing reporters that can be significantly customized at runtime.

Reporter-specific options always have to be prefixed with "X" (large
letter X).


### Other expected functionality of a reporter

When writing a custom reporter, there are few more things that you should
keep in mind. These are not important for correctness, but they are
important for the reporter to work _nicely_.

* Catch2 provides a simple verbosity option for users. There are three
  verbosity levels, "quiet", "normal", and "high", and if it makes sense
  for reporter's output format, it should respond to these by changing
  what, and how much, it writes out.

* Catch2 operates with an rng-seed. Knowing what seed a test run had
  is important if you want to replicate it, so your reporter should
  report the rng-seed, if at all possible given the target output format.

* Catch2 also operates with test filters, or test specs. If a filter
  is present, you should also report the filter, if at all possible given
  the target output format.



---

[Home](Readme.md#top)
