<a id="top"></a>
# Test cases and sections

**Contents**<br>
[Tags](#tags)<br>
[Tag aliases](#tag-aliases)<br>
[BDD-style test cases](#bdd-style-test-cases)<br>
[Type parametrised test cases](#type-parametrised-test-cases)<br>
[Signature based parametrised test cases](#signature-based-parametrised-test-cases)<br>

While Catch fully supports the traditional, xUnit, style of class-based fixtures containing test case methods this is not the preferred style.

Instead Catch provides a powerful mechanism for nesting test case sections within a test case. For a more detailed discussion see the [tutorial](tutorial.md#test-cases-and-sections).

Test cases and sections are very easy to use in practice:

* **TEST_CASE(** _test name_ \[, _tags_ \] **)**
* **SECTION(** _section name_, \[, _section description_ \] **)**


_test name_ and _section name_ are free form, quoted, strings.
The optional _tags_ argument is a quoted string containing one or more
tags enclosed in square brackets, and are discussed below.
_section description_ can be used to provide long form description
of a section while keeping the _section name_ short for use with the
[`-c` command line parameter](command-line.md#specify-the-section-to-run).

**The combination of test names and tags must be unique within the Catch2
executable.**

For examples see the [Tutorial](tutorial.md#top)

## Tags

Tags allow an arbitrary number of additional strings to be associated with a test case. Test cases can be selected (for running, or just for listing) by tag - or even by an expression that combines several tags. At their most basic level they provide a simple way to group several related tests together.

As an example - given the following test cases:

    TEST_CASE( "A", "[widget]" ) { /* ... */ }
    TEST_CASE( "B", "[widget]" ) { /* ... */ }
    TEST_CASE( "C", "[gadget]" ) { /* ... */ }
    TEST_CASE( "D", "[widget][gadget]" ) { /* ... */ }

The tag expression, ```"[widget]"``` selects A, B & D. ```"[gadget]"``` selects C & D. ```"[widget][gadget]"``` selects just D and ```"[widget],[gadget]"``` selects all four test cases.

For more detail on command line selection see [the command line docs](command-line.md#specifying-which-tests-to-run)

Tag names are not case sensitive and can contain any ASCII characters.
This means that tags `[tag with spaces]` and `[I said "good day"]`
are both allowed tags and can be filtered on. However, escapes are not
supported and `[\]]` is not a valid tag.

The same tag can be specified multiple times for a single test case,
but only one of the instances of identical tags will be kept. Which one
is kept is functionally random.


### Special Tags

All tag names beginning with non-alphanumeric characters are reserved by Catch. Catch defines a number of "special" tags, which have meaning to the test runner itself. These special tags all begin with a symbol character. Following is a list of currently defined special tags and their meanings.

* `[.]` - causes test cases to be skipped from the default list (i.e. when no test cases have been explicitly selected through tag expressions or name wildcards). The hide tag is often combined with another, user, tag (for example `[.][integration]` - so all integration tests are excluded from the default run but can be run by passing `[integration]` on the command line). As a short-cut you can combine these by simply prefixing your user tag with a `.` - e.g. `[.integration]`.

* `[!throws]` - lets Catch know that this test is likely to throw an exception even if successful. This causes the test to be excluded when running with `-e` or `--nothrow`.

* `[!mayfail]` - doesn't fail the test if any given assertion fails (but still reports it). This can be useful to flag a work-in-progress, or a known issue that you don't want to immediately fix but still want to track in your tests.

* `[!shouldfail]` - like `[!mayfail]` but *fails* the test if it *passes*. This can be useful if you want to be notified of accidental, or third-party, fixes.

* `[!nonportable]` - Indicates that behaviour may vary between platforms or compilers.

* `[#<filename>]` - these tags are added to test cases when you run Catch2
                    with [`-#` or `--filenames-as-tags`](command-line.md#filenames-as-tags).

* `[@<alias>]` - tag aliases all begin with `@` (see below).

* `[!benchmark]` - this test case is actually a benchmark. Currently this only serves to hide the test case by default, to avoid the execution time costs.


## Tag aliases

Between tag expressions and wildcarded test names (as well as combinations of the two) quite complex patterns can be constructed to direct which test cases are run. If a complex pattern is used often it is convenient to be able to create an alias for the expression. This can be done, in code, using the following form:

    CATCH_REGISTER_TAG_ALIAS( <alias string>, <tag expression> )

Aliases must begin with the `@` character. An example of a tag alias is:

    CATCH_REGISTER_TAG_ALIAS( "[@nhf]", "[failing]~[.]" )

Now when `[@nhf]` is used on the command line this matches all tests that are tagged `[failing]`, but which are not also hidden.

## BDD-style test cases

In addition to Catch's take on the classic style of test cases, Catch supports an alternative syntax that allow tests to be written as "executable specifications" (one of the early goals of [Behaviour Driven Development](http://dannorth.net/introducing-bdd/)). This set of macros map on to ```TEST_CASE```s and ```SECTION```s, with a little internal support to make them smoother to work with.

* **SCENARIO(** _scenario name_ \[, _tags_ \] **)**

This macro maps onto ```TEST_CASE``` and works in the same way, except that the test case name will be prefixed by "Scenario: "

* **GIVEN(** _something_ **)**
* **WHEN(** _something_ **)**
* **THEN(** _something_ **)**

These macros map onto ```SECTION```s except that the section names are the _something_ texts prefixed by
"given: ", "when: " or "then: " respectively. These macros also map onto the AAA or A<sup>3</sup> test pattern
(standing either for [Assemble-Activate-Assert](http://wiki.c2.com/?AssembleActivateAssert) or
[Arrange-Act-Assert](http://wiki.c2.com/?ArrangeActAssert)), and in this context, the macros provide both code
documentation and reporting of these parts of a test case without the need for extra comments or code to do so.

Semantically, a `GIVEN` clause may have multiple _independent_ `WHEN` clauses within it. This allows a test
to have, e.g., one set of "given" objects and multiple subtests using those objects in various ways in each
of the `WHEN` clauses without repeating the initialisation from the `GIVEN` clause. When there are _dependent_
clauses -- such as a second `WHEN` clause that should only happen _after_ the previous `WHEN` clause has been
executed and validated -- there are additional macros starting with `AND_`:

* **AND_GIVEN(** _something_ **)**
* **AND_WHEN(** _something_ **)**
* **AND_THEN(** _something_ **)**

These are used to chain ```GIVEN```s, ```WHEN```s and ```THEN```s together. The `AND_*` clause is placed
_inside_ the clause on which it depends. There can be multiple _independent_ clauses that are all _dependent_
on a single outer clause.
```cpp
SCENARIO( "vector can be sized and resized" ) {
    GIVEN( "An empty vector" ) {
        auto v = std::vector<std::string>{};

        // Validate assumption of the GIVEN clause
        THEN( "The size and capacity start at 0" ) {
            REQUIRE( v.size() == 0 );
            REQUIRE( v.capacity() == 0 );
        }

        // Validate one use case for the GIVEN object
        WHEN( "push_back() is called" ) {
            v.push_back("hullo");

            THEN( "The size changes" ) {
                REQUIRE( v.size() == 1 );
                REQUIRE( v.capacity() >= 1 );
            }
        }
    }
}
```

This code will result in two runs through the scenario:
```
Scenario : vector can be sized and resized
  Given  : An empty vector
  Then   : The size and capacity start at 0

Scenario : vector can be sized and resized
  Given  : An empty vector
  When   : push_back() is called
  Then   : The size changes
```

See also [runnable example on godbolt](https://godbolt.org/z/eY5a64r99),
with a more complicated (and failing) example.

> `AND_GIVEN` was [introduced](https://github.com/catchorg/Catch2/issues/1360) in Catch2 2.4.0.

When any of these macros are used the console reporter recognises them and formats the test case header such that the Givens, Whens and Thens are aligned to aid readability.

Other than the additional prefixes and the formatting in the console reporter these macros behave exactly as ```TEST_CASE```s and ```SECTION```s. As such there is nothing enforcing the correct sequencing of these macros - that's up to the programmer!

## Type parametrised test cases

In addition to `TEST_CASE`s, Catch2 also supports test cases parametrised
by types, in the form of `TEMPLATE_TEST_CASE`,
`TEMPLATE_PRODUCT_TEST_CASE` and `TEMPLATE_LIST_TEST_CASE`. These macros
are defined in the `catch_template_test_macros.hpp` header, so compiling
the code examples below also requires
`#include <catch2/catch_template_test_macros.hpp>`.


* **TEMPLATE_TEST_CASE(** _test name_ , _tags_,  _type1_, _type2_, ..., _typen_ **)**

> [Introduced](https://github.com/catchorg/Catch2/issues/1437) in Catch2 2.5.0.

_test name_ and _tag_ are exactly the same as they are in `TEST_CASE`,
with the difference that the tag string must be provided (however, it
can be empty). _type1_ through _typen_ is the list of types for which
this test case should run, and, inside the test code, the current type
is available as the `TestType` type.

Because of limitations of the C++ preprocessor, if you want to specify
a type with multiple template parameters, you need to enclose it in
parentheses, e.g. `std::map<int, std::string>` needs to be passed as
`(std::map<int, std::string>)`.

Example:
```cpp
TEMPLATE_TEST_CASE( "vectors can be sized and resized", "[vector][template]", int, std::string, (std::tuple<int,float>) ) {

    std::vector<TestType> v( 5 );

    REQUIRE( v.size() == 5 );
    REQUIRE( v.capacity() >= 5 );

    SECTION( "resizing bigger changes size and capacity" ) {
        v.resize( 10 );

        REQUIRE( v.size() == 10 );
        REQUIRE( v.capacity() >= 10 );
    }
    SECTION( "resizing smaller changes size but not capacity" ) {
        v.resize( 0 );

        REQUIRE( v.size() == 0 );
        REQUIRE( v.capacity() >= 5 );

        SECTION( "We can use the 'swap trick' to reset the capacity" ) {
            std::vector<TestType> empty;
            empty.swap( v );

            REQUIRE( v.capacity() == 0 );
        }
    }
    SECTION( "reserving smaller does not change size or capacity" ) {
        v.reserve( 0 );

        REQUIRE( v.size() == 5 );
        REQUIRE( v.capacity() >= 5 );
    }
}
```

* **TEMPLATE_PRODUCT_TEST_CASE(** _test name_ , _tags_, (_template-type1_, _template-type2_, ..., _template-typen_), (_template-arg1_, _template-arg2_, ..., _template-argm_) **)**

> [Introduced](https://github.com/catchorg/Catch2/issues/1468) in Catch2 2.6.0.

_template-type1_ through _template-typen_ is list of template
types which should be combined with each of _template-arg1_ through
 _template-argm_, resulting in _n * m_ test cases. Inside the test case,
the resulting type is available under the name of `TestType`.

To specify more than 1 type as a single _template-type_ or _template-arg_,
you must enclose the types in an additional set of parentheses, e.g.
`((int, float), (char, double))` specifies 2 template-args, each
consisting of 2 concrete types (`int`, `float` and `char`, `double`
respectively). You can also omit the outer set of parentheses if you
specify only one type as the full set of either the _template-types_,
or the _template-args_.


Example:
```cpp
template< typename T>
struct Foo {
    size_t size() {
        return 0;
    }
};

TEMPLATE_PRODUCT_TEST_CASE("A Template product test case", "[template][product]", (std::vector, Foo), (int, float)) {
    TestType x;
    REQUIRE(x.size() == 0);
}
```

You can also have different arities in the _template-arg_ packs:
```cpp
TEMPLATE_PRODUCT_TEST_CASE("Product with differing arities", "[template][product]", std::tuple, (int, (int, double), (int, double, float))) {
    TestType x;
    REQUIRE(std::tuple_size<TestType>::value >= 1);
}
```

* **TEMPLATE_LIST_TEST_CASE(** _test name_, _tags_, _type list_ **)**

> [Introduced](https://github.com/catchorg/Catch2/issues/1627) in Catch2 2.9.0.

_type list_ is a generic list of types on which test case should be instantiated.
List can be `std::tuple`, `boost::mpl::list`, `boost::mp11::mp_list` or anything with
`template <typename...>` signature.

This allows you to reuse the _type list_ in multiple test cases.

Example:
```cpp
using MyTypes = std::tuple<int, char, float>;
TEMPLATE_LIST_TEST_CASE("Template test case with test types specified inside std::tuple", "[template][list]", MyTypes)
{
    REQUIRE(sizeof(TestType) > 0);
}
```


## Signature based parametrised test cases

> [Introduced](https://github.com/catchorg/Catch2/issues/1609) in Catch2 2.8.0.

In addition to [type parametrised test cases](#type-parametrised-test-cases) Catch2 also supports
signature base parametrised test cases, in form of `TEMPLATE_TEST_CASE_SIG` and `TEMPLATE_PRODUCT_TEST_CASE_SIG`.
These test cases have similar syntax like [type parametrised test cases](#type-parametrised-test-cases), with one
additional positional argument which specifies the signature. These macros are defined in the
`catch_template_test_macros.hpp` header, so compiling the code examples below also requires
`#include <catch2/catch_template_test_macros.hpp>`.

### Signature
Signature has some strict rules for these tests cases to work properly:
* signature with multiple template parameters e.g. `typename T, size_t S` must have this format in test case declaration
  `((typename T, size_t S), T, S)`
* signature with variadic template arguments e.g. `typename T, size_t S, typename...Ts` must have this format in test case declaration
  `((typename T, size_t S, typename...Ts), T, S, Ts...)`
* signature with single non type template parameter e.g. `int V` must have this format in test case declaration `((int V), V)`
* signature with single type template parameter e.g. `typename T` should not be used as it is in fact `TEMPLATE_TEST_CASE`

Currently Catch2 support up to 11 template parameters in signature

### Examples

* **TEMPLATE_TEST_CASE_SIG(** _test name_ , _tags_,  _signature_, _type1_, _type2_, ..., _typen_ **)**

Inside `TEMPLATE_TEST_CASE_SIG` test case you can use the names of template parameters as defined in _signature_.

```cpp
TEMPLATE_TEST_CASE_SIG("TemplateTestSig: arrays can be created from NTTP arguments", "[vector][template][nttp]",
  ((typename T, int V), T, V), (int,5), (float,4), (std::string,15), ((std::tuple<int, float>), 6)) {

    std::array<T, V> v;
    REQUIRE(v.size() > 1);
}
```

* **TEMPLATE_PRODUCT_TEST_CASE_SIG(** _test name_ , _tags_, _signature_, (_template-type1_, _template-type2_, ..., _template-typen_), (_template-arg1_, _template-arg2_, ..., _template-argm_) **)**

```cpp

template<typename T, size_t S>
struct Bar {
    size_t size() { return S; }
};

TEMPLATE_PRODUCT_TEST_CASE_SIG("A Template product test case with array signature", "[template][product][nttp]", ((typename T, size_t S), T, S), (std::array, Bar), ((int, 9), (float, 42))) {
    TestType x;
    REQUIRE(x.size() > 0);
}
```


---

[Home](Readme.md#top)
