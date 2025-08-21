<a id="top"></a>
# Best practices and other tips on using Catch2

## Running tests

Your tests should be run in a manner roughly equivalent with:

```
./tests --order rand --warn NoAssertions
```

Notice that all the tests are run in a large batch, their relative order
is randomized, and that you ask Catch2 to fail test whose leaf-path
does not contain an assertion.

The reason I recommend running all your tests in the same process is that
this exposes your tests to interference from their runs. This can be both
positive interference, where the changes in global state from previous
test allow later tests to pass, but also negative interference, where
changes in global state from previous test causes later tests to fail.

In my experience, interference, especially destructive interference,
usually comes from errors in the code under test, rather than the tests
themselves. This means that by allowing interference to happen, our tests
can find these issues. Obviously, to shake out interference coming from
different orderings of tests, the test order also need to be shuffled
between runs.

However, running all tests in a single batch eventually becomes impractical
as they will take too long to run, and you will want to run your tests
in parallel.


<a id="parallel-tests"></a>
## Running tests in parallel

There are multiple ways of running tests in parallel, with various level
of structure. If you are using CMake and CTest, then we provide a helper
function [`catch_discover_tests`](cmake-integration.md#automatic-test-registration)
that registers each Catch2 `TEST_CASE` as a single CTest test, which
is then run in a separate process. This is an easy way to set up parallel
tests if you are already using CMake & CTest to run your tests, but you
will lose the advantage of running tests in batches.


Catch2 also supports [splitting tests in a binary into multiple
shards](command-line.md#test-sharding). This can be used by any test
runner to run batches of tests in parallel. Do note that when selecting
on the number of shards, you should have more shards than there are cores,
to avoid issues with long-running tests getting accidentally grouped in
the same shard, and causing long-tailed execution time.

**Note that naively composing sharding and random ordering of tests will break.**

Invoking Catch2 test executable like this

```text
./tests --order rand --shard-index 0 --shard-count 3
./tests --order rand --shard-index 1 --shard-count 3
./tests --order rand --shard-index 2 --shard-count 3
```

does not guarantee covering all tests inside the executable, because
each invocation will have its own random seed, thus it will have its own
random order of tests and thus the partitioning of tests into shards will
be different as well.

To do this properly, you need the individual shards to share the random
seed, e.g.
```text
./tests --order rand --shard-index 0 --shard-count 3 --rng-seed 0xBEEF
./tests --order rand --shard-index 1 --shard-count 3 --rng-seed 0xBEEF
./tests --order rand --shard-index 2 --shard-count 3 --rng-seed 0xBEEF
```

Catch2 actually provides a helper to automatically register multiple shards
as CTest tests, with shared random seed that changes each CTest invocation.
For details look at the documentation of
[`CatchShardTests.cmake` CMake script](cmake-integration.md#catchshardtestscmake).


## Organizing tests into binaries

Both overly large and overly small test binaries can cause issues. Overly
large test binaries have to be recompiled and relinked often, and the
link times are usually also long. Overly small test binaries in turn pay
significant overhead from linking against Catch2 more often per compiled
test case, and also make it hard/impossible to run tests in batches.

Because there is no hard and fast rule for the right size of a test binary,
I recommend having 1:1 correspondence between libraries in project and test
binaries. (At least if it is possible, in some cases it is not.) Having
a test binary for each library in project keeps related tests together,
and makes tests easy to navigate by reflecting the project's organizational
structure.


---

[Home](Readme.md#top)
