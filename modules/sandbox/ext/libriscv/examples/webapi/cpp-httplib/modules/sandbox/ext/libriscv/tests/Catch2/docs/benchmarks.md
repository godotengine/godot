<a id="top"></a>
# Authoring benchmarks

> [Introduced](https://github.com/catchorg/Catch2/issues/1616) in Catch2 2.9.0.

Writing benchmarks is not easy. Catch simplifies certain aspects but you'll
always need to take care about various aspects. Understanding a few things about
the way Catch runs your code will be very helpful when writing your benchmarks.

First off, let's go over some terminology that will be used throughout this
guide.

- *User code*: user code is the code that the user provides to be measured.
- *Run*: one run is one execution of the user code. Sometimes also referred
  to as an _iteration_.
- *Sample*: one sample is one data point obtained by measuring the time it takes
  to perform a certain number of runs. One sample can consist of more than one
  run if the clock available does not have enough resolution to accurately
  measure a single run. All samples for a given benchmark execution are obtained
  with the same number of runs.

## Execution procedure

Now I can explain how a benchmark is executed in Catch. There are three main
steps, though the first does not need to be repeated for every benchmark.

1. *Environmental probe*: before any benchmarks can be executed, the clock's
resolution is estimated. A few other environmental artifacts are also estimated
at this point, like the cost of calling the clock function, but they almost
never have any impact in the results.

2. *Estimation*: the user code is executed a few times to obtain an estimate of
the amount of runs that should be in each sample. This also has the potential
effect of bringing relevant code and data into the caches before the actual
measurement starts.

3. *Measurement*: all the samples are collected sequentially by performing the
number of runs estimated in the previous step for each sample.

This already gives us one important rule for writing benchmarks for Catch: the
benchmarks must be repeatable. The user code will be executed several times, and
the number of times it will be executed during the estimation step cannot be
known beforehand since it depends on the time it takes to execute the code.
User code that cannot be executed repeatedly will lead to bogus results or
crashes.

## Benchmark specification

Benchmarks can be specified anywhere inside a Catch test case.
There is a simple and a slightly more advanced version of the `BENCHMARK` macro.

Let's have a look how a naive Fibonacci implementation could be benchmarked:
```c++
std::uint64_t Fibonacci(std::uint64_t number) {
    return number < 2 ? 1 : Fibonacci(number - 1) + Fibonacci(number - 2);
}
```
Now the most straight forward way to benchmark this function, is just adding a `BENCHMARK` macro to our test case:
```c++
TEST_CASE("Fibonacci") {
    CHECK(Fibonacci(0) == 1);
    // some more asserts..
    CHECK(Fibonacci(5) == 8);
    // some more asserts..

    // now let's benchmark:
    BENCHMARK("Fibonacci 20") {
        return Fibonacci(20);
    };

    BENCHMARK("Fibonacci 25") {
        return Fibonacci(25);
    };

    BENCHMARK("Fibonacci 30") {
        return Fibonacci(30);
    };

    BENCHMARK("Fibonacci 35") {
        return Fibonacci(35);
    };
}
```
There's a few things to note:
- As `BENCHMARK` expands to a lambda expression it is necessary to add a semicolon after
 the closing brace (as opposed to the first experimental version).
- The `return` is a handy way to avoid the compiler optimizing away the benchmark code.

Running this already runs the benchmarks and outputs something similar to:
```
-------------------------------------------------------------------------------
Fibonacci
-------------------------------------------------------------------------------
C:\path\to\Catch2\Benchmark.tests.cpp(10)
...............................................................................
benchmark name                                  samples       iterations    est run time
                                                mean          low mean      high mean
                                                std dev       low std dev   high std dev
-------------------------------------------------------------------------------
Fibonacci 20                                            100       416439   83.2878 ms
                                                       2 ns         2 ns         2 ns
                                                       0 ns         0 ns         0 ns

Fibonacci 25                                            100       400776   80.1552 ms
                                                       3 ns         3 ns         3 ns
                                                       0 ns         0 ns         0 ns

Fibonacci 30                                            100       396873   79.3746 ms
                                                      17 ns        17 ns        17 ns
                                                       0 ns         0 ns         0 ns

Fibonacci 35                                            100       145169   87.1014 ms
                                                     468 ns       464 ns       473 ns
                                                      21 ns        15 ns        34 ns
```

### Advanced benchmarking
The simplest use case shown above, takes no arguments and just runs the user code that needs to be measured.
However, if using the `BENCHMARK_ADVANCED` macro and adding a `Catch::Benchmark::Chronometer` argument after
the macro, some advanced features are available. The contents of the simple benchmarks are invoked once per run,
while the blocks of the advanced benchmarks are invoked exactly twice:
once during the estimation phase, and another time during the execution phase.

```c++
BENCHMARK("simple"){ return long_computation(); };

BENCHMARK_ADVANCED("advanced")(Catch::Benchmark::Chronometer meter) {
    set_up();
    meter.measure([] { return long_computation(); });
};
```

These advanced benchmarks no longer consist entirely of user code to be measured.
In these cases, the code to be measured is provided via the
`Catch::Benchmark::Chronometer::measure` member function. This allows you to set up any
kind of state that might be required for the benchmark but is not to be included
in the measurements, like making a vector of random integers to feed to a
sorting algorithm.

A single call to `Catch::Benchmark::Chronometer::measure` performs the actual measurements
by invoking the callable object passed in as many times as necessary. Anything
that needs to be done outside the measurement can be done outside the call to
`measure`.

The callable object passed in to `measure` can optionally accept an `int`
parameter.

```c++
meter.measure([](int i) { return long_computation(i); });
```

If it accepts an `int` parameter, the sequence number of each run will be passed
in, starting with 0. This is useful if you want to measure some mutating code,
for example. The number of runs can be known beforehand by calling
`Catch::Benchmark::Chronometer::runs`; with this one can set up a different instance to be
mutated by each run.

```c++
std::vector<std::string> v(meter.runs());
std::fill(v.begin(), v.end(), test_string());
meter.measure([&v](int i) { in_place_escape(v[i]); });
```

Note that it is not possible to simply use the same instance for different runs
and resetting it between each run since that would pollute the measurements with
the resetting code.

It is also possible to just provide an argument name to the simple `BENCHMARK` macro to get
the same semantics as providing a callable to `meter.measure` with `int` argument:

```c++
BENCHMARK("indexed", i){ return long_computation(i); };
```

### Constructors and destructors

All of these tools give you a lot mileage, but there are two things that still
need special handling: constructors and destructors. The problem is that if you
use automatic objects they get destroyed by the end of the scope, so you end up
measuring the time for construction and destruction together. And if you use
dynamic allocation instead, you end up including the time to allocate memory in
the measurements.

To solve this conundrum, Catch provides class templates that let you manually
construct and destroy objects without dynamic allocation and in a way that lets
you measure construction and destruction separately.

```c++
BENCHMARK_ADVANCED("construct")(Catch::Benchmark::Chronometer meter) {
    std::vector<Catch::Benchmark::storage_for<std::string>> storage(meter.runs());
    meter.measure([&](int i) { storage[i].construct("thing"); });
};

BENCHMARK_ADVANCED("destroy")(Catch::Benchmark::Chronometer meter) {
    std::vector<Catch::Benchmark::destructable_object<std::string>> storage(meter.runs());
    for(auto&& o : storage)
        o.construct("thing");
    meter.measure([&](int i) { storage[i].destruct(); });
};
```

`Catch::Benchmark::storage_for<T>` objects are just pieces of raw storage suitable for `T`
objects. You can use the `Catch::Benchmark::storage_for::construct` member function to call a constructor and
create an object in that storage. So if you want to measure the time it takes
for a certain constructor to run, you can just measure the time it takes to run
this function.

When the lifetime of a `Catch::Benchmark::storage_for<T>` object ends, if an actual object was
constructed there it will be automatically destroyed, so nothing leaks.

If you want to measure a destructor, though, we need to use
`Catch::Benchmark::destructable_object<T>`. These objects are similar to
`Catch::Benchmark::storage_for<T>` in that construction of the `T` object is manual, but
it does not destroy anything automatically. Instead, you are required to call
the `Catch::Benchmark::destructable_object::destruct` member function, which is what you
can use to measure the destruction time.

### The optimizer

Sometimes the optimizer will optimize away the very code that you want to
measure. There are several ways to use results that will prevent the optimiser
from removing them. You can use the `volatile` keyword, or you can output the
value to standard output or to a file, both of which force the program to
actually generate the value somehow.

Catch adds a third option. The values returned by any function provided as user
code are guaranteed to be evaluated and not optimised out. This means that if
your user code consists of computing a certain value, you don't need to bother
with using `volatile` or forcing output. Just `return` it from the function.
That helps with keeping the code in a natural fashion.

Here's an example:

```c++
// may measure nothing at all by skipping the long calculation since its
// result is not used
BENCHMARK("no return"){ long_calculation(); };

// the result of long_calculation() is guaranteed to be computed somehow
BENCHMARK("with return"){ return long_calculation(); };
```

However, there's no other form of control over the optimizer whatsoever. It is
up to you to write a benchmark that actually measures what you want and doesn't
just measure the time to do a whole bunch of nothing.

To sum up, there are two simple rules: whatever you would do in handwritten code
to control optimization still works in Catch; and Catch makes return values
from user code into observable effects that can't be optimized away.

<i>Adapted from nonius' documentation.</i>
