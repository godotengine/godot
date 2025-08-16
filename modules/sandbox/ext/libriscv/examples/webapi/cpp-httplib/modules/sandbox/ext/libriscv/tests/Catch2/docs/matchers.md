<a id="top"></a>
# Matchers

**Contents**<br>
[Using Matchers](#using-matchers)<br>
[Built-in matchers](#built-in-matchers)<br>
[Writing custom matchers (old style)](#writing-custom-matchers-old-style)<br>
[Writing custom matchers (new style)](#writing-custom-matchers-new-style)<br>

Matchers, as popularized by the [Hamcrest](https://en.wikipedia.org/wiki/Hamcrest)
framework are an alternative way to write assertions, useful for tests
where you work with complex types or need to assert more complex
properties. Matchers are easily composable and users can write their
own and combine them with the Catch2-provided matchers seamlessly.


## Using Matchers

Matchers are most commonly used in tandem with the `REQUIRE_THAT` or
`CHECK_THAT` macros. The `REQUIRE_THAT` macro takes two arguments,
the first one is the input (object/value) to test, the second argument
is the matcher itself.

For example, to assert that a string ends with the "as a service"
substring, you can write the following assertion

```cpp
using Catch::Matchers::EndsWith;

REQUIRE_THAT( getSomeString(), EndsWith("as a service") );
```

Individual matchers can also be combined using the C++ logical
operators, that is `&&`, `||`, and `!`, like so:

```cpp
using Catch::Matchers::EndsWith;
using Catch::Matchers::ContainsSubstring;

REQUIRE_THAT( getSomeString(),
              EndsWith("as a service") && ContainsSubstring("web scale"));
```

The example above asserts that the string returned from `getSomeString`
_both_ ends with the suffix "as a service" _and_ contains the string
"web scale" somewhere.


Both of the string matchers used in the examples above live in the
`catch_matchers_string.hpp` header, so to compile the code above also
requires `#include <catch2/matchers/catch_matchers_string.hpp>`.

### Combining operators and lifetimes

**IMPORTANT**: The combining operators do not take ownership of the
matcher objects being combined.

This means that if you store combined matcher object, you have to ensure
that the individual matchers being combined outlive the combined matcher.
Note that the negation matcher from `!` also counts as combining matcher
for this.

Explained on an example, this is fine
```cpp
CHECK_THAT(value, WithinAbs(0, 2e-2) && !WithinULP(0., 1));
```

and so is this
```cpp
auto is_close_to_zero = WithinAbs(0, 2e-2);
auto is_zero          = WithinULP(0., 1);

CHECK_THAT(value, is_close_to_zero && !is_zero);
```

but this is not
```cpp
auto is_close_to_zero = WithinAbs(0, 2e-2);
auto is_zero          = WithinULP(0., 1);
auto is_close_to_but_not_zero = is_close_to_zero && !is_zero;

CHECK_THAT(a_value, is_close_to_but_not_zero); // UAF
```

because `!is_zero` creates a temporary instance of Negation matcher,
which the `is_close_to_but_not_zero` refers to. After the line ends,
the temporary is destroyed and the combined `is_close_to_but_not_zero`
matcher now refers to non-existent object, so using it causes use-after-free.


## Built-in matchers

Every matcher provided by Catch2 is split into 2 parts, a factory
function that lives in the `Catch::Matchers` namespace, and the actual
matcher type that is in some deeper namespace and should not be used by
the user. In the examples above, we used `Catch::Matchers::Contains`.
This is the factory function for the
`Catch::Matchers::StdString::ContainsMatcher` type that does the actual
matching.

Out of the box, Catch2 provides the following matchers:


### `std::string` matchers

Catch2 provides 5 different matchers that work with `std::string`,
* `StartsWith(std::string str, CaseSensitive)`,
* `EndsWith(std::string str, CaseSensitive)`,
* `ContainsSubstring(std::string str, CaseSensitive)`,
* `Equals(std::string str, CaseSensitive)`, and
* `Matches(std::string str, CaseSensitive)`.

The first three should be fairly self-explanatory, they succeed if
the argument starts with `str`, ends with `str`, or contains `str`
somewhere inside it.

The `Equals` matcher matches a string if (and only if) the argument
string is equal to `str`.

Finally, the `Matches` matcher performs an ECMAScript regex match using
`str` against the argument string. It is important to know that
the match is performed against the string as a whole, meaning that
the regex `"abc"` will not match input string `"abcd"`. To match
`"abcd"`, you need to use e.g. `"abc.*"` as your regex.

The second argument sets whether the matching should be case-sensitive
or not. By default, it is case-sensitive.

> `std::string` matchers live in `catch2/matchers/catch_matchers_string.hpp`


### Vector matchers

_Vector matchers have been deprecated in favour of the generic
range matchers with the same functionality._

Catch2 provides 5 built-in matchers that work on `std::vector`.

These are

 * `Contains` which checks whether a specified vector is present in the result
 * `VectorContains` which checks whether a specified element is present in the result
 * `Equals` which checks whether the result is exactly equal (order matters) to a specific vector
 * `UnorderedEquals` which checks whether the result is equal to a specific vector under a permutation
 * `Approx` which checks whether the result is "approx-equal" (order matters, but comparison is done via `Approx`) to a specific vector
> Approx matcher was [introduced](https://github.com/catchorg/Catch2/issues/1499) in Catch2 2.7.2.

An example usage:
```cpp
    std::vector<int> some_vec{ 1, 2, 3 };
    REQUIRE_THAT(some_vec, Catch::Matchers::UnorderedEquals(std::vector<int>{ 3, 2, 1 }));
```

This assertions will pass, because the elements given to the matchers
are a permutation of the ones in `some_vec`.

> vector matchers live in `catch2/matchers/catch_matchers_vector.hpp`


### Floating point matchers

Catch2 provides 4 matchers that target floating point numbers. These
are:

* `WithinAbs(double target, double margin)`,
* `WithinULP(FloatingPoint target, uint64_t maxUlpDiff)`, and
* `WithinRel(FloatingPoint target, FloatingPoint eps)`.
* `IsNaN()`

> `WithinRel` matcher was introduced in Catch2 2.10.0

> `IsNaN` matcher was introduced in Catch2 3.3.2.

The first three serve to compare two floating pointe numbers. For more
details about how they work, read [the docs on comparing floating point
numbers](comparing-floating-point-numbers.md#floating-point-matchers).

`IsNaN` then does exactly what it says on the tin. It matches the input
if it is a NaN (Not a Number). The advantage of using it over just plain
`REQUIRE(std::isnan(x))`, is that if the check fails, with `REQUIRE` you
won't see the value of `x`, but with `REQUIRE_THAT(x, IsNaN())`, you will.


### Miscellaneous matchers

Catch2 also provides some matchers and matcher utilities that do not
quite fit into other categories.

The first one of them is the `Predicate(Callable pred, std::string description)`
matcher. It creates a matcher object that calls `pred` for the provided
argument. The `description` argument allows users to set what the
resulting matcher should self-describe as if required.

Do note that you will need to explicitly specify the type of the
argument, like in this example:

```cpp
REQUIRE_THAT("Hello olleH",
             Predicate<std::string>(
                 [] (std::string const& str) -> bool { return str.front() == str.back(); },
                 "First and last character should be equal")
);
```

> the predicate matcher lives in `catch2/matchers/catch_matchers_predicate.hpp`


The other miscellaneous matcher utility is exception matching.


#### Matching exceptions

Because exceptions are a bit special, Catch2 has a separate macro for them.


The basic form is

```
REQUIRE_THROWS_MATCHES(expr, ExceptionType, Matcher)
```

and it checks that the `expr` throws an exception, that exception is derived
from the `ExceptionType` type, and then `Matcher::match` is called on
the caught exception.

> `REQUIRE_THROWS_MATCHES` macro lives in `catch2/matchers/catch_matchers.hpp`

For one-off checks you can use the `Predicate` matcher above, e.g.

```cpp
REQUIRE_THROWS_MATCHES(parse(...),
                       parse_error,
                       Predicate<parse_error>([] (parse_error const& err) -> bool { return err.line() == 1; })
);
```

but if you intend to thoroughly test your error reporting, I recommend
defining a specialized matcher.


Catch2 also provides 2 built-in matchers for checking the error message
inside an exception (it must be derived from `std::exception`):
* `Message(std::string message)`.
* `MessageMatches(Matcher matcher)`.

> `MessageMatches` was [introduced](https://github.com/catchorg/Catch2/pull/2570) in Catch2 3.3.0

`Message` checks that the exception's
message, as returned from `what` is exactly equal to `message`.

`MessageMatches` applies the provided matcher on the exception's
message, as returned from `what`. This is useful in conjunctions with the `std::string` matchers (e.g. `StartsWith`)

Example use:
```cpp
REQUIRE_THROWS_MATCHES(throwsDerivedException(),  DerivedException,  Message("DerivedException::what"));
REQUIRE_THROWS_MATCHES(throwsDerivedException(),  DerivedException,  MessageMatches(StartsWith("DerivedException")));
```

> the exception message matchers live in `catch2/matchers/catch_matchers_exception.hpp`


### Generic range Matchers

> Generic range matchers were introduced in Catch2 3.0.1

Catch2 also provides some matchers that use the new style matchers
definitions to handle generic range-like types. These are:

* `IsEmpty()`
* `SizeIs(size_t target_size)`
* `SizeIs(Matcher size_matcher)`
* `Contains(T&& target_element, Comparator = std::equal_to<>{})`
* `Contains(Matcher element_matcher)`
* `AllMatch(Matcher element_matcher)`
* `AnyMatch(Matcher element_matcher)`
* `NoneMatch(Matcher element_matcher)`
* `AllTrue()`, `AnyTrue()`, `NoneTrue()`
* `RangeEquals(TargetRangeLike&&, Comparator = std::equal_to<>{})`
* `UnorderedRangeEquals(TargetRangeLike&&, Comparator = std::equal_to<>{})`

> `IsEmpty`, `SizeIs`, `Contains` were introduced in Catch2 3.0.1

> `All/Any/NoneMatch` were introduced in Catch2 3.0.1

> `All/Any/NoneTrue` were introduced in Catch2 3.1.0

> `RangeEquals` and `UnorderedRangeEquals` matchers were [introduced](https://github.com/catchorg/Catch2/pull/2377) in Catch2 3.3.0

`IsEmpty` should be self-explanatory. It successfully matches objects
that are empty according to either `std::empty`, or ADL-found `empty`
free function.

`SizeIs` checks range's size. If constructed with `size_t` arg, the
matchers accepts ranges whose size is exactly equal to the arg. If
constructed from another matcher, then the resulting matcher accepts
ranges whose size is accepted by the provided matcher.

`Contains` accepts ranges that contain specific element. There are
again two variants, one that accepts the desired element directly,
in which case a range is accepted if any of its elements is equal to
the target element. The other variant is constructed from a matcher,
in which case a range is accepted if any of its elements is accepted
by the provided matcher.

`AllMatch`, `NoneMatch`, and `AnyMatch` match ranges for which either
all, none, or any of the contained elements matches the given matcher,
respectively.

`AllTrue`, `NoneTrue`, and `AnyTrue` match ranges for which either
all, none, or any of the contained elements are `true`, respectively.
It works for ranges of `bool`s and ranges of elements (explicitly)
convertible to `bool`.

`RangeEquals` compares the range that the matcher is constructed with
(the "target range") against the range to be tested, element-wise. The
match succeeds if all elements from the two ranges compare equal (using
`operator==` by default). The ranges do not need to be the same type,
and the element types do not need to be the same, as long as they are
comparable. (e.g. you may compare `std::vector<int>` to `std::array<char>`).

`UnorderedRangeEquals` is similar to `RangeEquals`, but the order
does not matter. For example "1, 2, 3" would match "3, 2, 1", but not
"1, 1, 2, 3" As with `RangeEquals`, `UnorderedRangeEquals` compares
the individual elements using `operator==` by default.

Both `RangeEquals` and `UnorderedRangeEquals` optionally accept a
predicate which can be used to compare the containers element-wise.

To check a container elementwise against a given matcher, use
`AllMatch`.


## Writing custom matchers (old style)

The old style of writing matchers has been introduced back in Catch
Classic. To create an old-style matcher, you have to create your own
type that derives from `Catch::Matchers::MatcherBase<ArgT>`, where
`ArgT` is the type your matcher works for. Your type has to override
two methods, `bool match(ArgT const&) const`,
and `std::string describe() const`.

As the name suggests, `match` decides whether the provided argument
is matched (accepted) by the matcher. `describe` then provides a
human-oriented description of what the matcher does.

We also recommend that you create factory function, just like Catch2
does, but that is mostly useful for template argument deduction for
templated matchers (assuming you do not have CTAD available).

To combine these into an example, let's say that you want to write
a matcher that decides whether the provided argument is a number
within certain range. We will call it `IsBetweenMatcher<T>`:

```c++
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
// ...


template <typename T>
class IsBetweenMatcher : public Catch::Matchers::MatcherBase<T> {
    T m_begin, m_end;
public:
    IsBetweenMatcher(T begin, T end) : m_begin(begin), m_end(end) {}

    bool match(T const& in) const override {
        return in >= m_begin && in <= m_end;
    }

    std::string describe() const override {
        std::ostringstream ss;
        ss << "is between " << m_begin << " and " << m_end;
        return ss.str();
    }
};

template <typename T>
IsBetweenMatcher<T> IsBetween(T begin, T end) {
    return { begin, end };
}

// ...

TEST_CASE("Numbers are within range") {
    // infers `double` for the argument type of the matcher
    CHECK_THAT(3., IsBetween(1., 10.));
    // infers `int` for the argument type of the matcher
    CHECK_THAT(100, IsBetween(1, 10));
}
```

Obviously, the code above can be improved somewhat, for example you
might want to `static_assert` over the fact that `T` is an arithmetic
type... or generalize the matcher to cover any type for which the user
can provide a comparison function object.

Note that while any matcher written using the old style can also be
written using the new style, combining old style matchers should
generally compile faster. Also note that you can combine old and new
style matchers arbitrarily.

> `MatcherBase` lives in `catch2/matchers/catch_matchers.hpp`


## Writing custom matchers (new style)

> New style matchers were introduced in Catch2 3.0.1

To create a new-style matcher, you have to create your own type that
derives from `Catch::Matchers::MatcherGenericBase`. Your type has to
also provide two methods, `bool match( ... ) const` and overridden
`std::string describe() const`.

Unlike with old-style matchers, there are no requirements on how
the `match` member function takes its argument. This means that the
argument can be taken by value or by mutating reference, but also that
the matcher's `match` member function can be templated.

This allows you to write more complex matcher, such as a matcher that
can compare one range-like (something that responds to `begin` and
`end`) object to another, like in the following example:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
// ...

template<typename Range>
struct EqualsRangeMatcher : Catch::Matchers::MatcherGenericBase {
    EqualsRangeMatcher(Range const& range):
        range{ range }
    {}

    template<typename OtherRange>
    bool match(OtherRange const& other) const {
        using std::begin; using std::end;

        return std::equal(begin(range), end(range), begin(other), end(other));
    }

    std::string describe() const override {
        return "Equals: " + Catch::rangeToString(range);
    }

private:
    Range const& range;
};

template<typename Range>
auto EqualsRange(const Range& range) -> EqualsRangeMatcher<Range> {
    return EqualsRangeMatcher<Range>{range};
}

TEST_CASE("Combining templated matchers", "[matchers][templated]") {
    std::array<int, 3> container{{ 1,2,3 }};

    std::array<int, 3> a{{ 1,2,3 }};
    std::vector<int> b{ 0,1,2 };
    std::list<int> c{ 4,5,6 };

    REQUIRE_THAT(container, EqualsRange(a) || EqualsRange(b) || EqualsRange(c));
}
```

Do note that while you can rewrite any matcher from the old style to
a new style matcher, combining new style matchers is more expensive
in terms of compilation time. Also note that you can combine old style
and new style matchers arbitrarily.

> `MatcherGenericBase` lives in `catch2/matchers/catch_matchers_templated.hpp`


---

[Home](Readme.md#top)
