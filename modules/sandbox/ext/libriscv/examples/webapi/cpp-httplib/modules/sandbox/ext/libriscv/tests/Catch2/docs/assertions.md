<a id="top"></a>
# Assertion Macros

**Contents**<br>
[Natural Expressions](#natural-expressions)<br>
[Floating point comparisons](#floating-point-comparisons)<br>
[Exceptions](#exceptions)<br>
[Matcher expressions](#matcher-expressions)<br>
[Thread Safety](#thread-safety)<br>
[Expressions with commas](#expressions-with-commas)<br>

Most test frameworks have a large collection of assertion macros to capture all possible conditional forms (```_EQUALS```, ```_NOTEQUALS```, ```_GREATER_THAN``` etc).

Catch is different. Because it decomposes natural C-style conditional expressions most of these forms are reduced to one or two that you will use all the time. That said there is a rich set of auxiliary macros as well. We'll describe all of these here.

Most of these macros come in two forms:

## Natural Expressions

The ```REQUIRE``` family of macros tests an expression and aborts the test case if it fails.
The ```CHECK``` family are equivalent but execution continues in the same test case even if the assertion fails. This is useful if you have a series of essentially orthogonal assertions and it is useful to see all the results rather than stopping at the first failure.

* **REQUIRE(** _expression_ **)** and
* **CHECK(** _expression_ **)**

Evaluates the expression and records the result. If an exception is thrown, it is caught, reported, and counted as a failure. These are the macros you will use most of the time.

Examples:
```
CHECK( str == "string value" );
CHECK( thisReturnsTrue() );
REQUIRE( i == 42 );
```

Expressions prefixed with `!` cannot be decomposed. If you have a type
that is convertible to bool and you want to assert that it evaluates to
false, use the two forms below:


* **REQUIRE_FALSE(** _expression_ **)** and
* **CHECK_FALSE(** _expression_ **)**

Note that there is no reason to use these forms for plain bool variables,
because there is no added value in decomposing them.

Example:
```cpp
Status ret = someFunction();
REQUIRE_FALSE(ret); // ret must evaluate to false, and Catch2 will print
                    // out the value of ret if possibly
```


### Other limitations

Note that expressions containing either of the binary logical operators,
`&&` or `||`, cannot be decomposed and will not compile. The reason behind
this is that it is impossible to overload `&&` and `||` in a way that
keeps their short-circuiting semantics, and expression decomposition
relies on overloaded operators to work.

Simple example of an issue with overloading binary logical operators
is a common pointer idiom, `p && p->foo == 2`. Using the built-in `&&`
operator, `p` is only dereferenced if it is not null. With overloaded
`&&`, `p` is always dereferenced, thus causing a segfault if
`p == nullptr`.

If you want to test expression that contains `&&` or `||`, you have two
options.

1) Enclose it in parentheses. Parentheses force evaluation of the expression
   before the expression decomposition can touch it, and thus it cannot
   be used.

2) Rewrite the expression. `REQUIRE(a == 1 && b == 2)` can always be split
   into `REQUIRE(a == 1); REQUIRE(b == 2);`. Alternatively, if this is a
   common pattern in your tests, think about using [Matchers](#matcher-expressions).
   instead. There is no simple rewrite rule for `||`, but I generally
   believe that if you have `||` in your test expression, you should rethink
   your tests.


## Floating point comparisons

Comparing floating point numbers is complex, and [so it has its own
documentation page](comparing-floating-point-numbers.md#top).


## Exceptions

* **REQUIRE_NOTHROW(** _expression_ **)** and
* **CHECK_NOTHROW(** _expression_ **)**

Expects that no exception is thrown during evaluation of the expression.

* **REQUIRE_THROWS(** _expression_ **)** and
* **CHECK_THROWS(** _expression_ **)**

Expects that an exception (of any type) is be thrown during evaluation of the expression.

* **REQUIRE_THROWS_AS(** _expression_, _exception type_ **)** and
* **CHECK_THROWS_AS(** _expression_, _exception type_ **)**

Expects that an exception of the _specified type_ is thrown during evaluation of the expression. Note that the _exception type_ is extended with `const&` and you should not include it yourself.

* **REQUIRE_THROWS_WITH(** _expression_, _string or string matcher_ **)** and
* **CHECK_THROWS_WITH(** _expression_, _string or string matcher_ **)**

Expects that an exception is thrown that, when converted to a string, matches the _string_ or _string matcher_ provided (see next section for Matchers).

e.g.
```cpp
REQUIRE_THROWS_WITH( openThePodBayDoors(), ContainsSubstring( "afraid" ) && ContainsSubstring( "can't do that" ) );
REQUIRE_THROWS_WITH( dismantleHal(), "My mind is going" );
```

* **REQUIRE_THROWS_MATCHES(** _expression_, _exception type_, _matcher for given exception type_ **)** and
* **CHECK_THROWS_MATCHES(** _expression_, _exception type_, _matcher for given exception type_ **)**

Expects that exception of _exception type_ is thrown and it matches provided matcher (see the [documentation for Matchers](matchers.md#top)).


_Please note that the `THROW` family of assertions expects to be passed a single expression, not a statement or series of statements. If you want to check a more complicated sequence of operations, you can use a C++11 lambda function._

```cpp
REQUIRE_NOTHROW([&](){
    int i = 1;
    int j = 2;
    auto k = i + j;
    if (k == 3) {
        throw 1;
    }
}());
```



## Matcher expressions

To support Matchers a slightly different form is used. Matchers have [their own documentation](matchers.md#top).

* **REQUIRE_THAT(** _lhs_, _matcher expression_ **)** and
* **CHECK_THAT(** _lhs_, _matcher expression_ **)**

Matchers can be composed using `&&`, `||` and `!` operators.

## Thread Safety

Currently assertions in Catch are not thread safe.
For more details, along with workarounds, see the section on [the limitations page](limitations.md#thread-safe-assertions).

## Expressions with commas

Because the preprocessor parses code using different rules than the
compiler, multiple-argument assertions (e.g. `REQUIRE_THROWS_AS`) have
problems with commas inside the provided expressions. As an example
`REQUIRE_THROWS_AS(std::pair<int, int>(1, 2), std::invalid_argument);`
will fail to compile, because the preprocessor sees 3 arguments provided,
but the macro accepts only 2. There are two possible workarounds.

1) Use typedef:
```cpp
using int_pair = std::pair<int, int>;
REQUIRE_THROWS_AS(int_pair(1, 2), std::invalid_argument);
```

This solution is always applicable, but makes the meaning of the code
less clear.

2) Parenthesize the expression:
```cpp
TEST_CASE_METHOD((Fixture<int, int>), "foo", "[bar]") {
    SUCCEED();
}
```

This solution is not always applicable, because it might require extra
changes on the Catch's side to work.

---

[Home](Readme.md#top)
