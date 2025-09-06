<a id="top"></a>
# Command line

**Contents**<br>
[Specifying which tests to run](#specifying-which-tests-to-run)<br>
[Choosing a reporter to use](#choosing-a-reporter-to-use)<br>
[Breaking into the debugger](#breaking-into-the-debugger)<br>
[Showing results for successful tests](#showing-results-for-successful-tests)<br>
[Aborting after a certain number of failures](#aborting-after-a-certain-number-of-failures)<br>
[Listing available tests, tags or reporters](#listing-available-tests-tags-or-reporters)<br>
[Sending output to a file](#sending-output-to-a-file)<br>
[Naming a test run](#naming-a-test-run)<br>
[Eliding assertions expected to throw](#eliding-assertions-expected-to-throw)<br>
[Make whitespace visible](#make-whitespace-visible)<br>
[Warnings](#warnings)<br>
[Reporting timings](#reporting-timings)<br>
[Load test names to run from a file](#load-test-names-to-run-from-a-file)<br>
[Specify the order test cases are run](#specify-the-order-test-cases-are-run)<br>
[Specify a seed for the Random Number Generator](#specify-a-seed-for-the-random-number-generator)<br>
[Identify framework and version according to the libIdentify standard](#identify-framework-and-version-according-to-the-libidentify-standard)<br>
[Wait for key before continuing](#wait-for-key-before-continuing)<br>
[Skip all benchmarks](#skip-all-benchmarks)<br>
[Specify the number of benchmark samples to collect](#specify-the-number-of-benchmark-samples-to-collect)<br>
[Specify the number of resamples for bootstrapping](#specify-the-number-of-resamples-for-bootstrapping)<br>
[Specify the confidence-interval for bootstrapping](#specify-the-confidence-interval-for-bootstrapping)<br>
[Disable statistical analysis of collected benchmark samples](#disable-statistical-analysis-of-collected-benchmark-samples)<br>
[Specify the amount of time in milliseconds spent on warming up each test](#specify-the-amount-of-time-in-milliseconds-spent-on-warming-up-each-test)<br>
[Usage](#usage)<br>
[Specify the section to run](#specify-the-section-to-run)<br>
[Filenames as tags](#filenames-as-tags)<br>
[Override output colouring](#override-output-colouring)<br>
[Test Sharding](#test-sharding)<br>
[Allow running the binary without tests](#allow-running-the-binary-without-tests)<br>
[Output verbosity](#output-verbosity)<br>

Catch works quite nicely without any command line options at all - but for those times when you want greater control the following options are available.
Click one of the following links to take you straight to that option - or scroll on to browse the available options.

<a href="#specifying-which-tests-to-run">               `    <test-spec> ...`</a><br />
<a href="#usage">                                       `    -h, -?, --help`</a><br />
<a href="#showing-results-for-successful-tests">        `    -s, --success`</a><br />
<a href="#breaking-into-the-debugger">                  `    -b, --break`</a><br />
<a href="#eliding-assertions-expected-to-throw">        `    -e, --nothrow`</a><br />
<a href="#invisibles">                                  `    -i, --invisibles`</a><br />
<a href="#sending-output-to-a-file">                    `    -o, --out`</a><br />
<a href="#choosing-a-reporter-to-use">                  `    -r, --reporter`</a><br />
<a href="#naming-a-test-run">                           `    -n, --name`</a><br />
<a href="#aborting-after-a-certain-number-of-failures"> `    -a, --abort`</a><br />
<a href="#aborting-after-a-certain-number-of-failures"> `    -x, --abortx`</a><br />
<a href="#warnings">                                    `    -w, --warn`</a><br />
<a href="#reporting-timings">                           `    -d, --durations`</a><br />
<a href="#input-file">                                  `    -f, --input-file`</a><br />
<a href="#run-section">                                 `    -c, --section`</a><br />
<a href="#filenames-as-tags">                           `    -#, --filenames-as-tags`</a><br />


</br>

<a href="#listing-available-tests-tags-or-reporters">   `    --list-tests`</a><br />
<a href="#listing-available-tests-tags-or-reporters">   `    --list-tags`</a><br />
<a href="#listing-available-tests-tags-or-reporters">   `    --list-reporters`</a><br />
<a href="#listing-available-tests-tags-or-reporters">   `    --list-listeners`</a><br />
<a href="#order">                                       `    --order`</a><br />
<a href="#rng-seed">                                    `    --rng-seed`</a><br />
<a href="#libidentify">                                 `    --libidentify`</a><br />
<a href="#wait-for-keypress">                           `    --wait-for-keypress`</a><br />
<a href="#skip-benchmarks">                             `    --skip-benchmarks`</a><br />
<a href="#benchmark-samples">                           `    --benchmark-samples`</a><br />
<a href="#benchmark-resamples">                         `    --benchmark-resamples`</a><br />
<a href="#benchmark-confidence-interval">               `    --benchmark-confidence-interval`</a><br />
<a href="#benchmark-no-analysis">                       `    --benchmark-no-analysis`</a><br />
<a href="#benchmark-warmup-time">                       `    --benchmark-warmup-time`</a><br />
<a href="#colour-mode">                                 `    --colour-mode`</a><br />
<a href="#test-sharding">                               `    --shard-count`</a><br />
<a href="#test-sharding">                               `    --shard-index`</a><br />
<a href=#no-tests-override>                             `    --allow-running-no-tests`</a><br />
<a href=#output-verbosity>                              `    --verbosity`</a><br />

</br>



<a id="specifying-which-tests-to-run"></a>
## Specifying which tests to run

<pre>&lt;test-spec> ...</pre>

By providing a test spec, you filter which tests will be run. If you call
Catch2 without any test spec, then it will run all non-hidden test
cases. A test case is hidden if it has the `[!benchmark]` tag, any tag
with a dot at the start, e.g. `[.]` or `[.foo]`.

There are three basic test specs that can then be combined into more
complex specs:

  * Full test name, e.g. `"Test 1"`.

    This allows only test cases whose name is "Test 1".

  * Wildcarded test name, e.g. `"*Test"`, or `"Test*"`, or `"*Test*"`.

    This allows any test case whose name ends with, starts with, or contains
    in the middle the string "Test". Note that the wildcard can only be at
    the start or end.

  * Tag name, e.g. `[some-tag]`.

    This allows any test case tagged with "[some-tag]". Remember that some
    tags are special, e.g. those that start with "." or with "!".


You can also combine the basic test specs to create more complex test
specs. You can:

  * Concatenate specs to apply all of them, e.g. `[some-tag][other-tag]`.

    This allows test cases that are tagged with **both** "[some-tag]" **and**
    "[other-tag]". A test case with just "[some-tag]" will not pass the filter,
    nor will test case with just "[other-tag]".

  * Comma-join specs to apply any of them, e.g. `[some-tag],[other-tag]`.

    This allows test cases that are tagged with **either** "[some-tag]" **or**
    "[other-tag]". A test case with both will obviously also pass the filter.

    Note that commas take precedence over simple concatenation. This means
    that `[a][b],[c]` accepts tests that are tagged with either both "[a]" and
    "[b]", or tests that are tagged with just "[c]".

  * Negate the spec by prepending it with `~`, e.g. `~[some-tag]`.

    This rejects any test case that is tagged with "[some-tag]". Note that
    rejection takes precedence over other filters.

    Note that negations always binds to the following _basic_ test spec.
    This means that `~[foo][bar]` negates only the "[foo]" tag and not the
    "[bar]" tag.

Note that when Catch2 is deciding whether to include a test, first it
checks whether the test matches any negative filters. If it does,
the test is rejected. After that, the behaviour depends on whether there
are positive filters as well. If there are no positive filters, all
remaining non-hidden tests are included. If there are positive filters,
only tests that match the positive filters are included.

You can also match test names with special characters by escaping them
with a backslash (`"\"`), e.g. a test named `"Do A, then B"` is matched
by `"Do A\, then B"` test spec. Backslash also escapes itself.


### Examples

Given these TEST_CASEs,
```
TEST_CASE("Test 1") {}

TEST_CASE("Test 2", "[.foo]") {}

TEST_CASE("Test 3", "[.bar]") {}

TEST_CASE("Test 4", "[.][foo][bar]") {}
```

this is the result of these filters
```
./tests                      # Selects only the first test, others are hidden
./tests "Test 1"             # Selects only the first test, other do not match
./tests ~"Test 1"            # Selects no tests. Test 1 is rejected, other tests are hidden
./tests "Test *"             # Selects all tests.
./tests [bar]                # Selects tests 3 and 4. Other tests are not tagged [bar]
./tests ~[foo]               # Selects test 1, because it is the only non-hidden test without [foo] tag
./tests [foo][bar]           # Selects test 4.
./tests [foo],[bar]          # Selects tests 2, 3, 4.
./tests ~[foo][bar]          # Selects test 3. 2 and 4 are rejected due to having [foo] tag
./tests ~"Test 2"[foo]       # Selects test 4, because test 2 is explicitly rejected
./tests [foo][bar],"Test 1"  # Selects tests 1 and 4.
./tests "Test 1*"            # Selects test 1, wildcard can match zero characters
```

_Note: Using plain asterisk on a command line can cause issues with shell
expansion. Make sure that the asterisk is passed to Catch2 and is not
interpreted by the shell._


<a id="choosing-a-reporter-to-use"></a>
## Choosing a reporter to use

<pre>-r, --reporter &lt;reporter[::key=value]*&gt;</pre>

Reporters are how the output from Catch2 (results of assertions, tests,
benchmarks and so on) is formatted and written out. The default reporter
is called the "Console" reporter and is intended to provide relatively
verbose and human-friendly output.

Reporters are also individually configurable. To pass configuration options
to the reporter, you append `::key=value` to the reporter specification
as many times as you want, e.g. `--reporter xml::out=someFile.xml` or
`--reporter custom::colour-mode=ansi::Xoption=2`.

The keys must either be prefixed by "X", in which case they are not parsed
by Catch2 and are only passed down to the reporter, or one of options
hardcoded into Catch2. Currently there are only 2,
["out"](#sending-output-to-a-file), and ["colour-mode"](#colour-mode).

_Note that the reporter might still check the X-prefixed options for
validity, and throw an error if they are wrong._

> Support for passing arguments to reporters through the `-r`, `--reporter` flag was introduced in Catch2 3.0.1

There are multiple built-in reporters, you can see what they do by using the
[`--list-reporters`](command-line.md#listing-available-tests-tags-or-reporters)
flag. If you need a reporter providing custom format outside of the already
provided ones, look at the ["write your own reporter" part of the reporter
documentation](reporters.md#writing-your-own-reporter).

This option may be passed multiple times to use multiple (different)
reporters  at the same time. See the [reporter documentation](reporters.md#multiple-reporters)
for details on what the resulting behaviour is. Also note that at most one
reporter can be provided without the output-file part of reporter spec.
This reporter will use the "default" output destination, based on
the [`-o`, `--out`](#sending-output-to-a-file) option.

> Support for using multiple different reporters at the same time was [introduced](https://github.com/catchorg/Catch2/pull/2183) in Catch2 3.0.1


_Note: There is currently no way to escape `::` in the reporter spec,
and thus the reporter names, or configuration keys and values, cannot
contain `::`. As `::` in paths is relatively obscure (unlike ':'), we do
not consider this an issue._


<a id="breaking-into-the-debugger"></a>
## Breaking into the debugger
<pre>-b, --break</pre>

Under most debuggers Catch2 is capable of automatically breaking on a test
failure. This allows the user to see the current state of the test during
failure.

<a id="showing-results-for-successful-tests"></a>
## Showing results for successful tests
<pre>-s, --success</pre>

Usually you only want to see reporting for failed tests. Sometimes it's useful to see *all* the output (especially when you don't trust that that test you just added worked first time!).
To see successful, as well as failing, test results just pass this option. Note that each reporter may treat this option differently. The Junit reporter, for example, logs all results regardless.

<a id="aborting-after-a-certain-number-of-failures"></a>
## Aborting after a certain number of failures
<pre>-a, --abort
-x, --abortx [&lt;failure threshold>]
</pre>

If a ```REQUIRE``` assertion fails the test case aborts, but subsequent test cases are still run.
If a ```CHECK``` assertion fails even the current test case is not aborted.

Sometimes this results in a flood of failure messages and you'd rather just see the first few. Specifying ```-a``` or ```--abort``` on its own will abort the whole test run on the first failed assertion of any kind. Use ```-x``` or ```--abortx``` followed by a number to abort after that number of assertion failures.

<a id="listing-available-tests-tags-or-reporters"></a>
## Listing available tests, tags or reporters
```
--list-tests
--list-tags
--list-reporters
--list-listeners
```

> The `--list*` options became customizable through reporters in Catch2 3.0.1

> The `--list-listeners` option was added in Catch2 3.0.1

`--list-tests` lists all registered tests matching specified test spec.
Usually this listing also includes tags, and potentially also other
information, like source location, based on verbosity and reporter's design.

`--list-tags` lists all tags from registered tests matching specified test
spec. Usually this also includes number of tests cases they match and
similar information.

`--list-reporters` lists all available reporters and their descriptions.

`--list-listeners` lists all registered listeners and their descriptions.

The [`--verbosity` argument](#output-verbosity) modifies the level of detail provided by the default `--list*` options
as follows:

| Option             | `normal` (default)              | `quiet`             | `high`                                  |
|--------------------|---------------------------------|---------------------|-----------------------------------------|
| `--list-tests`     | Test names and tags             | Test names only     | Same as `normal`, plus source code line |
| `--list-tags`      | Tags and counts                 | Same as `normal`    | Same as `normal`                        |
| `--list-reporters` | Reporter names and descriptions | Reporter names only | Same as `normal`                        |
| `--list-listeners` | Listener names and descriptions | Same as `normal`    | Same as `normal`                        |

<a id="sending-output-to-a-file"></a>
## Sending output to a file
<pre>-o, --out &lt;filename&gt;
</pre>

Use this option to send all output to a file, instead of stdout. You can
use `-` as the filename to explicitly send the output to stdout (this is
useful e.g. when using multiple reporters).

> Support for `-` as the filename was introduced in Catch2 3.0.1

Filenames starting with "%" (percent symbol) are reserved by Catch2 for
meta purposes, e.g. using `%debug` as the filename opens stream that
writes to platform specific debugging/logging mechanism.

Catch2 currently recognizes 3 meta streams:

* `%debug` - writes to platform specific debugging/logging output
* `%stdout` - writes to stdout
* `%stderr` - writes to stderr

> Support for `%stdout` and `%stderr` was introduced in Catch2 3.0.1


<a id="naming-a-test-run"></a>
## Naming a test run
<pre>-n, --name &lt;name for test run></pre>

If a name is supplied it will be used by the reporter to provide an overall name for the test run. This can be useful if you are sending to a file, for example, and need to distinguish different test runs - either from different Catch executables or runs of the same executable with different options. If not supplied the name is defaulted to the name of the executable.

<a id="eliding-assertions-expected-to-throw"></a>
## Eliding assertions expected to throw
<pre>-e, --nothrow</pre>

Skips all assertions that test that an exception is thrown, e.g. ```REQUIRE_THROWS```.

These can be a nuisance in certain debugging environments that may break when exceptions are thrown (while this is usually optional for handled exceptions, it can be useful to have enabled if you are trying to track down something unexpected).

Sometimes exceptions are expected outside of one of the assertions that tests for them (perhaps thrown and caught within the code-under-test). The whole test case can be skipped when using ```-e``` by marking it with the ```[!throws]``` tag.

When running with this option any throw checking assertions are skipped so as not to contribute additional noise. Be careful if this affects the behaviour of subsequent tests.

<a id="invisibles"></a>
## Make whitespace visible
<pre>-i, --invisibles</pre>

If a string comparison fails due to differences in whitespace - especially leading or trailing whitespace - it can be hard to see what's going on.
This option transforms tabs and newline characters into ```\t``` and ```\n``` respectively when printing.

<a id="warnings"></a>
## Warnings
<pre>-w, --warn &lt;warning name></pre>

You can think of Catch2's warnings as the equivalent of `-Werror` (`/WX`)
flag for C++ compilers. It turns some suspicious occurrences, like a section
without assertions, into errors. Because these might be intended, warnings
are not enabled by default, but user can opt in.

You can enable multiple warnings at the same time.

There are currently two warnings implemented:

```
    NoAssertions        // Fail test case / leaf section if no assertions
                        // (e.g. `REQUIRE`) is encountered.
    UnmatchedTestSpec   // Fail test run if any of the CLI test specs did
                        // not match any tests.
```

> `UnmatchedTestSpec` was introduced in Catch2 3.0.1.


<a id="reporting-timings"></a>
## Reporting timings
<pre>-d, --durations &lt;yes/no></pre>

When set to ```yes``` Catch will report the duration of each test case, in seconds with millisecond precision. Note that it does this regardless of whether a test case passes or fails. Note, also, the certain reporters (e.g. Junit) always report test case durations regardless of this option being set or not.

<pre>-D, --min-duration &lt;value></pre>

> `--min-duration` was [introduced](https://github.com/catchorg/Catch2/pull/1910) in Catch2 2.13.0

When set, Catch will report the duration of each test case that took more
than &lt;value> seconds, in seconds with millisecond precision. This option is overridden by both
`-d yes` and `-d no`, so that either all durations are reported, or none
are.


<a id="input-file"></a>
## Load test names to run from a file
<pre>-f, --input-file &lt;filename></pre>

Provide the name of a file that contains a list of test case names,
one per line. Blank lines are skipped.

A useful way to generate an initial instance of this file is to combine
the [`--list-tests`](#listing-available-tests-tags-or-reporters) flag with
the [`--verbosity quiet`](#output-verbosity) option. You can also
use test specs to filter this list down to what you want first.


<a id="order"></a>
## Specify the order test cases are run
<pre>--order &lt;decl|lex|rand&gt;</pre>

Test cases are ordered one of three ways:

### decl
Declaration order.

Tests in the same translation unit are sorted using their declaration orders,
different TUs are sorted in an implementation (linking) dependent order.


### lex
Lexicographic order.

Tests are sorted by their name, their tags are ignored.


### rand
Randomized order. The default order.

> Randomized order has been made default in Catch2 3.9.0

The order is dependent on Catch2's random seed (see
[`--rng-seed`](#rng-seed)), and is subset invariant. What this means
is that as long as the random seed is fixed, running only some tests
(e.g. via tag) does not change their relative order.

> The subset stability was introduced in Catch2 v2.12.0

Since the random order was made subset stable, we promise that given
the same random seed, the order of test cases will be the same across
different platforms, as long as the tests were compiled against identical
version of Catch2. We reserve the right to change the relative order
of tests cases between Catch2 versions, but it is unlikely to happen often.


<a id="rng-seed"></a>
## Specify a seed for the Random Number Generator
<pre>--rng-seed &lt;'time'|'random-device'|number&gt;</pre>

Sets the seed for random number generators used by Catch2. These are used
e.g. to shuffle tests when user asks for tests to be in random order.

Using `time` as the argument asks Catch2 generate the seed through call
to `std::time(nullptr)`. This provides very weak randomness and multiple
runs of the binary can generate the same seed if they are started close
to each other.

Using `random-device` asks for `std::random_device` to be used instead.
If your implementation provides working `std::random_device`, it should
be preferred to using `time`. Catch2 uses `std::random_device` by default.


<a id="libidentify"></a>
## Identify framework and version according to the libIdentify standard
<pre>--libidentify</pre>

See [The LibIdentify repo for more information and examples](https://github.com/janwilmans/LibIdentify).

<a id="wait-for-keypress"></a>
## Wait for key before continuing
<pre>--wait-for-keypress &lt;never|start|exit|both&gt;</pre>

Will cause the executable to print a message and wait until the return/ enter key is pressed before continuing -
either before running any tests, after running all tests - or both, depending on the argument.

<a id="skip-benchmarks"></a>
## Skip all benchmarks
<pre>--skip-benchmarks</pre>

> [Introduced](https://github.com/catchorg/Catch2/issues/2408) in Catch2 3.0.1.

This flag tells Catch2 to skip running all benchmarks. Benchmarks in this
case mean code blocks in `BENCHMARK` and `BENCHMARK_ADVANCED` macros, not
test cases with the `[!benchmark]` tag.

<a id="benchmark-samples"></a>
## Specify the number of benchmark samples to collect
<pre>--benchmark-samples &lt;# of samples&gt;</pre>

> [Introduced](https://github.com/catchorg/Catch2/issues/1616) in Catch2 2.9.0.

When running benchmarks a number of "samples" is collected. This is the base data for later statistical analysis.
Per sample a clock resolution dependent number of iterations of the user code is run, which is independent of the number of samples. Defaults to 100.

<a id="benchmark-resamples"></a>
## Specify the number of resamples for bootstrapping
<pre>--benchmark-resamples &lt;# of resamples&gt;</pre>

> [Introduced](https://github.com/catchorg/Catch2/issues/1616) in Catch2 2.9.0.

After the measurements are performed, statistical [bootstrapping] is performed
on the samples. The number of resamples for that bootstrapping is configurable
but defaults to 100000. Due to the bootstrapping it is possible to give
estimates for the mean and standard deviation. The estimates come with a lower
bound and an upper bound, and the confidence interval (which is configurable but
defaults to 95%).

 [bootstrapping]: http://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29

<a id="benchmark-confidence-interval"></a>
## Specify the confidence-interval for bootstrapping
<pre>--benchmark-confidence-interval &lt;confidence-interval&gt;</pre>

> [Introduced](https://github.com/catchorg/Catch2/issues/1616) in Catch2 2.9.0.

The confidence-interval is used for statistical bootstrapping on the samples to
calculate the upper and lower bounds of mean and standard deviation.
Must be between 0 and 1 and defaults to 0.95.

<a id="benchmark-no-analysis"></a>
## Disable statistical analysis of collected benchmark samples
<pre>--benchmark-no-analysis</pre>

> [Introduced](https://github.com/catchorg/Catch2/issues/1616) in Catch2 2.9.0.

When this flag is specified no bootstrapping or any other statistical analysis is performed.
Instead the user code is only measured and the plain mean from the samples is reported.

<a id="benchmark-warmup-time"></a>
## Specify the amount of time in milliseconds spent on warming up each test
<pre>--benchmark-warmup-time</pre>

> [Introduced](https://github.com/catchorg/Catch2/pull/1844) in Catch2 2.11.2.

Configure the amount of time spent warming up each test.

<a id="usage"></a>
## Usage
<pre>-h, -?, --help</pre>

Prints the command line arguments to stdout


<a id="run-section"></a>
## Specify the section to run
<pre>-c, --section &lt;section name&gt;</pre>

To limit execution to a specific section within a test case, use this option one or more times.
To narrow to sub-sections use multiple instances, where each subsequent instance specifies a deeper nesting level.

E.g. if you have:

<pre>
TEST_CASE( "Test" ) {
  SECTION( "sa" ) {
    SECTION( "sb" ) {
      /*...*/
    }
    SECTION( "sc" ) {
      /*...*/
    }
  }
  SECTION( "sd" ) {
    /*...*/
  }
}
</pre>

Then you can run `sb` with:
<pre>./MyExe Test -c sa -c sb</pre>

Or run just `sd` with:
<pre>./MyExe Test -c sd</pre>

To run all of `sa`, including `sb` and `sc` use:
<pre>./MyExe Test -c sa</pre>

There are some limitations of this feature to be aware of:
- Code outside of sections being skipped will still be executed - e.g. any set-up code in the TEST_CASE before the
start of the first section.</br>
- At time of writing, wildcards are not supported in section names.
- If you specify a section without narrowing to a test case first then all test cases will be executed
(but only matching sections within them).


<a id="filenames-as-tags"></a>
## Filenames as tags
<pre>-#, --filenames-as-tags</pre>

This option adds an extra tag to all test cases. The tag is `#` followed
by the unqualified filename the test case is defined in, with the _last_
extension stripped out.

For example, tests within the file `tests\SelfTest\UsageTests\BDD.tests.cpp`
will be given the `[#BDD.tests]` tag.


<a id="colour-mode"></a>
## Override output colouring
<pre>--colour-mode &lt;ansi|win32|none|default&gt;</pre>

> The `--colour-mode` option replaced the old `--colour` option in Catch2 3.0.1


Catch2 support two different ways of colouring terminal output, and by
default it attempts to make a good guess on which implementation to use
(and whether to even use it, e.g. Catch2 tries to avoid writing colour
codes when writing the results into a file).

`--colour-mode` allows the user to explicitly select what happens.

* `--colour-mode ansi` tells Catch2 to always use ANSI colour codes, even
when writing to a file
* `--colour-mode win32` tells Catch2 to use colour implementation based
  on Win32 terminal API
* `--colour-mode none` tells Catch2 to disable colours completely
* `--colour-mode default` lets Catch2 decide

`--colour-mode default` is the default setting.


<a id="test-sharding"></a>
## Test Sharding
<pre>--shard-count <#number of shards>, --shard-index <#shard index to run></pre>

> [Introduced](https://github.com/catchorg/Catch2/pull/2257) in Catch2 3.0.1.

When `--shard-count <#number of shards>` is used, the tests to execute
will be split evenly in to the given number of sets, identified by indices
starting at 0. The tests in the set given by
`--shard-index <#shard index to run>` will be executed. The default shard
count is `1`, and the default index to run is `0`.

_Shard index must be less than number of shards. As the name suggests,
it is treated as an index of the shard to run._

Sharding is useful when you want to split test execution across multiple
processes, as is done with the [Bazel test sharding](https://docs.bazel.build/versions/main/test-encyclopedia.html#test-sharding).


<a id="no-tests-override"></a>
## Allow running the binary without tests
<pre>--allow-running-no-tests</pre>

> Introduced in Catch2 3.0.1.

By default, Catch2 test binaries return non-0 exit code if no tests were run,
e.g. if the binary was compiled with no tests, the provided test spec matched no
tests, or all tests [were skipped at runtime](skipping-passing-failing.md#top). This flag
overrides that, so a test run with no tests still returns 0.

## Output verbosity
```
-v, --verbosity <quiet|normal|high>
```

Changing verbosity might change how many details Catch2's reporters output.
However, you should consider changing the verbosity level as a _suggestion_.
Not all reporters support all verbosity levels, e.g. because the reporter's
format cannot meaningfully change. In that case, the verbosity level is
ignored.

Verbosity defaults to _normal_.


---

[Home](Readme.md#top)
