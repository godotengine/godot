<a id="top"></a>
# String conversions

**Contents**<br>
[operator << overload for std::ostream](#operator--overload-for-stdostream)<br>
[Catch::StringMaker specialisation](#catchstringmaker-specialisation)<br>
[Catch::is_range specialisation](#catchis_range-specialisation)<br>
[Exceptions](#exceptions)<br>
[Enums](#enums)<br>
[Floating point precision](#floating-point-precision)<br>


Catch needs to be able to convert types you use in assertions and logging expressions into strings (for logging and reporting purposes).
Most built-in or std types are supported out of the box but there are two ways that you can tell Catch how to convert your own types (or other, third-party types) into strings.

## operator << overload for std::ostream

This is the standard way of providing string conversions in C++ - and the chances are you may already provide this for your own purposes. If you're not familiar with this idiom it involves writing a free function of the form:

```cpp
std::ostream& operator << ( std::ostream& os, T const& value ) {
    os << convertMyTypeToString( value );
    return os;
}
```

(where ```T``` is your type and ```convertMyTypeToString``` is where you'll write whatever code is necessary to make your type printable - it doesn't have to be in another function).

You should put this function in the same namespace as your type, or the global namespace, and have it declared before including Catch's header.

## Catch::StringMaker specialisation
If you don't want to provide an ```operator <<``` overload, or you want to convert your type differently for testing purposes, you can provide a specialization for `Catch::StringMaker<T>`:

```cpp
namespace Catch {
    template<>
    struct StringMaker<T> {
        static std::string convert( T const& value ) {
            return convertMyTypeToString( value );
        }
    };
}
```

## Catch::is_range specialisation
As a fallback, Catch attempts to detect if the type can be iterated
(`begin(T)` and `end(T)` are valid) and if it can be, it is stringified
as a range. For certain types this can lead to infinite recursion, so
it can be disabled by specializing `Catch::is_range` like so:

```cpp
namespace Catch {
    template<>
    struct is_range<T> {
        static const bool value = false;
    };
}

```


## Exceptions

By default all exceptions deriving from `std::exception` will be translated to strings by calling the `what()` method. For exception types that do not derive from `std::exception` - or if `what()` does not return a suitable string - use `CATCH_TRANSLATE_EXCEPTION`. This defines a function that takes your exception type, by reference, and returns a string. It can appear anywhere in the code - it doesn't have to be in the same translation unit. For example:

```cpp
CATCH_TRANSLATE_EXCEPTION( MyType const& ex ) {
    return ex.message();
}
```

## Enums

> Introduced in Catch2 2.8.0.

Enums that already have a `<<` overload for `std::ostream` will convert to strings as expected.
If you only need to convert enums to strings for test reporting purposes you can provide a `StringMaker` specialisations as any other type.
However, as a convenience, Catch provides the `CATCH_REGISTER_ENUM` helper macro that will generate the `StringMaker` specialisation for you with minimal code.
Simply provide it the (qualified) enum name, followed by all the enum values, and you're done!

E.g.

```cpp
enum class Fruits { Banana, Apple, Mango };

CATCH_REGISTER_ENUM( Fruits, Fruits::Banana, Fruits::Apple, Fruits::Mango )

TEST_CASE() {
    REQUIRE( Fruits::Mango == Fruits::Apple );
}
```

... or if the enum is in a namespace:
```cpp
namespace Bikeshed {
    enum class Colours { Red, Green, Blue };
}

// Important!: This macro must appear at top level scope - not inside a namespace
// You can fully qualify the names, or use a using if you prefer
CATCH_REGISTER_ENUM( Bikeshed::Colours,
    Bikeshed::Colours::Red,
    Bikeshed::Colours::Green,
    Bikeshed::Colours::Blue )

TEST_CASE() {
    REQUIRE( Bikeshed::Colours::Red == Bikeshed::Colours::Blue );
}
```

## Floating point precision

> [Introduced](https://github.com/catchorg/Catch2/issues/1614) in Catch2 2.8.0.

Catch provides a built-in `StringMaker` specialization for both `float`
and `double`. By default, it uses what we think is a reasonable precision,
but you can customize it by modifying the `precision` static variable
inside the `StringMaker` specialization, like so:

```cpp
        Catch::StringMaker<float>::precision = 15;
        const float testFloat1 = 1.12345678901234567899f;
        const float testFloat2 = 1.12345678991234567899f;
        REQUIRE(testFloat1 == testFloat2);
```

This assertion will fail and print out the `testFloat1` and `testFloat2`
to 15 decimal places.

---

[Home](Readme.md#top)
