<a id="top"></a>
# Comparing floating point numbers with Catch2

If you are not deeply familiar with them, floating point numbers can be
unintuitive. This also applies to comparing floating point numbers for
(in)equality.

This page assumes that you have some understanding of both FP, and the
meaning of different kinds of comparisons, and only goes over what
functionality Catch2 provides to help you with comparing floating point
numbers. If you do not have this understanding, we recommend that you first
study up on floating point numbers and their comparisons, e.g. by [reading
this blog post](https://codingnest.com/the-little-things-comparing-floating-point-numbers/).


## Floating point matchers

```
#include <catch2/matchers/catch_matchers_floating_point.hpp>
```

[Matchers](matchers.md#top) are the preferred way of comparing floating
point numbers in Catch2. We provide 3 of them:

* `WithinAbs(double target, double margin)`,
* `WithinRel(FloatingPoint target, FloatingPoint eps)`, and
* `WithinULP(FloatingPoint target, uint64_t maxUlpDiff)`.

> `WithinRel` matcher was introduced in Catch2 2.10.0

As with all matchers, you can combine multiple floating point matchers
in a single assertion. For example, to check that some computation matches
a known good value within 0.1% or is close enough (no different to 5
decimal places) to zero, we would write this assertion:

```cpp
    REQUIRE_THAT( computation(input),
        Catch::Matchers::WithinRel(expected, 0.001)
     || Catch::Matchers::WithinAbs(0, 0.000001) );
```


### WithinAbs

`WithinAbs` creates a matcher that accepts floating point numbers whose
difference with `target` is less-or-equal to the `margin`. Since `float`
can be converted to `double` without losing precision, only `double`
overload exists.

```cpp
REQUIRE_THAT(1.0, WithinAbs(1.2, 0.2));
REQUIRE_THAT(0.f, !WithinAbs(1.0, 0.5));
// Notice that infinity == infinity for WithinAbs
REQUIRE_THAT(INFINITY, WithinAbs(INFINITY, 0));
```


### WithinRel

`WithinRel` creates a matcher that accepts floating point numbers that
are _approximately equal_ to the `target` with a tolerance of `eps.`
Specifically, it matches if
`|arg - target| <= eps * max(|arg|, |target|)` holds. If you do not
specify `eps`, `std::numeric_limits<FloatingPoint>::epsilon * 100`
is used as the default.

```cpp
// Notice that WithinRel comparison is symmetric, unlike Approx's.
REQUIRE_THAT(1.0, WithinRel(1.1, 0.1));
REQUIRE_THAT(1.1, WithinRel(1.0, 0.1));
// Notice that inifnity == infinity for WithinRel
REQUIRE_THAT(INFINITY, WithinRel(INFINITY));
```


### WithinULP

`WithinULP` creates a matcher that accepts floating point numbers that
are no more than `maxUlpDiff`
[ULPs](https://en.wikipedia.org/wiki/Unit_in_the_last_place)
away from the `target` value. The short version of what this means
is that there is no more than `maxUlpDiff - 1` representable floating
point numbers between the argument for matching and the `target` value.

When using the ULP matcher in Catch2, it is important to keep in mind
that Catch2 interprets ULP distance slightly differently than
e.g. `std::nextafter` does.

Catch2's ULP calculation obeys these relations:
  * `ulpDistance(-x, x) == 2 * ulpDistance(x, 0)`
  * `ulpDistance(-0, 0) == 0` (due to the above)
  * `ulpDistance(DBL_MAX, INFINITY) == 1`
  * `ulpDistancE(NaN, x) == infinity`


**Important**: The WithinULP matcher requires the platform to use the
[IEEE-754](https://en.wikipedia.org/wiki/IEEE_754) representation for
floating point numbers.

```cpp
REQUIRE_THAT( -0.f, WithinULP( 0.f, 0 ) );
```


## `Approx`

```
#include <catch2/catch_approx.hpp>
```

**We strongly recommend against using `Approx` when writing new code.**
You should be using floating point matchers instead.

Catch2 provides one more way to handle floating point comparisons. It is
`Approx`, a special type with overloaded comparison operators, that can
be used in standard assertions, e.g.

```cpp
REQUIRE(0.99999 == Catch::Approx(1));
```

`Approx` supports four comparison operators, `==`, `!=`, `<=`, `>=`, and can
also be used with strong typedefs over `double`s. It can be used for both
relative and margin comparisons by using its three customization points.
Note that the semantics of this is always that of an _or_, so if either
the relative or absolute margin comparison passes, then the whole comparison
passes.

The downside to `Approx` is that it has a couple of issues that we cannot
fix without breaking backwards compatibility. Because Catch2 also provides
complete set of matchers that implement different floating point comparison
methods, `Approx` is left as-is, is considered deprecated, and should
not be used in new code.

The issues are
  * All internal computation is done in `double`s, leading to slightly
    different results if the inputs were floats.
  * `Approx`'s relative margin comparison is not symmetric. This means
    that `Approx( 10 ).epsilon(0.1) != 11.1` but `Approx( 11.1 ).epsilon(0.1) == 10`.
  * By default, `Approx` only uses relative margin comparison. This means
    that `Approx(0) == X` only passes for `X == 0`.


### Approx details

If you still want/need to know more about `Approx`, read on.

Catch2 provides a UDL for `Approx`; `_a`. It resides in the `Catch::literals`
namespace, and can be used like this:

```cpp
using namespace Catch::literals;
REQUIRE( performComputation() == 2.1_a );
```

`Approx` has three customization points for the comparison:

* **epsilon** - epsilon sets the coefficient by which a result
can differ from `Approx`'s value before it is rejected.
_Defaults to `std::numeric_limits<float>::epsilon()*100`._

```cpp
Approx target = Approx(100).epsilon(0.01);
100.0 == target; // Obviously true
200.0 == target; // Obviously still false
100.5 == target; // True, because we set target to allow up to 1% difference
```


* **margin** - margin sets the absolute value by which
a result can differ from `Approx`'s value before it is rejected.
_Defaults to `0.0`._

```cpp
Approx target = Approx(100).margin(5);
100.0 == target; // Obviously true
200.0 == target; // Obviously still false
104.0 == target; // True, because we set target to allow absolute difference of at most 5
```

* **scale** - scale is used to change the magnitude of `Approx` for the relative check.
_By default, set to `0.0`._

Scale could be useful if the computation leading to the result worked
on a different scale than is used by the results. Approx's scale is added
to Approx's value when computing the allowed relative margin from the
Approx's value.


---

[Home](Readme.md#top)
