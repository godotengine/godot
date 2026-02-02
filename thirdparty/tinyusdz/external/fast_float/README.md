
## fast_float number parsing library: 4x faster than strtod
[![Fuzzing Status](https://oss-fuzz-build-logs.storage.googleapis.com/badges/fast_float.svg)](https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&can=1&q=proj:fast_float)
[![VS17-CI](https://github.com/fastfloat/fast_float/actions/workflows/vs17-ci.yml/badge.svg)](https://github.com/fastfloat/fast_float/actions/workflows/vs17-ci.yml)
[![Ubuntu 22.04 CI (GCC 11)](https://github.com/fastfloat/fast_float/actions/workflows/ubuntu22.yml/badge.svg)](https://github.com/fastfloat/fast_float/actions/workflows/ubuntu22.yml)

The fast_float library provides fast header-only implementations for the C++ from_chars
functions for `float` and `double` types.  These functions convert ASCII strings representing
decimal values (e.g., `1.3e10`) into binary types. We provide exact rounding (including
round to even). In our experience, these `fast_float` functions many times faster than comparable number-parsing functions from existing C++ standard libraries.

Specifically, `fast_float` provides the following two functions with a C++17-like syntax (the library itself only requires C++11):

```C++
from_chars_result from_chars(const char* first, const char* last, float& value, ...);
from_chars_result from_chars(const char* first, const char* last, double& value, ...);
```

The return type (`from_chars_result`) is defined as the struct:
```C++
struct from_chars_result {
    const char* ptr;
    std::errc ec;
};
```

It parses the character sequence [first,last) for a number. It parses floating-point numbers expecting
a locale-independent format equivalent to the C++17 from_chars function.
The resulting floating-point value is the closest floating-point values (using either float or double),
using the "round to even" convention for values that would otherwise fall right in-between two values.
That is, we provide exact parsing according to the IEEE standard.


Given a successful parse, the pointer (`ptr`) in the returned value is set to point right after the
parsed number, and the `value` referenced is set to the parsed value. In case of error, the returned
`ec` contains a representative error, otherwise the default (`std::errc()`) value is stored.

The implementation does not throw and does not allocate memory (e.g., with `new` or `malloc`).

It will parse infinity and nan values.

Example:

``` C++
#include "fast_float/fast_float.h"
#include <iostream>

int main() {
    const std::string input =  "3.1416 xyz ";
    double result;
    auto answer = fast_float::from_chars(input.data(), input.data()+input.size(), result);
    if(answer.ec != std::errc()) { std::cerr << "parsing failure\n"; return EXIT_FAILURE; }
    std::cout << "parsed the number " << result << std::endl;
    return EXIT_SUCCESS;
}
```


Like the C++17 standard, the `fast_float::from_chars` functions take an optional last argument of
the type `fast_float::chars_format`. It is a bitset value: we check whether
`fmt & fast_float::chars_format::fixed` and `fmt & fast_float::chars_format::scientific` are set
to determine whether we allow the fixed point and scientific notation respectively.
The default is  `fast_float::chars_format::general` which allows both `fixed` and `scientific`.

The library seeks to follow the C++17 (see [20.19.3](http://eel.is/c++draft/charconv.from.chars).(7.1))  specification.
* The `from_chars` function does not skip leading white-space characters.
* [A leading `+` sign](https://en.cppreference.com/w/cpp/utility/from_chars) is forbidden.
* It is generally impossible to represent a decimal value exactly as binary floating-point number (`float` and `double` types). We seek the nearest value. We round to an even mantissa when we are in-between two binary floating-point numbers.

Furthermore, we have the following restrictions:
* We only support `float` and `double` types at this time.
* We only support the decimal format: we do not support hexadecimal strings.
* For values that are either very large or very small (e.g., `1e9999`), we represent it using the infinity or negative infinity value and the returned `ec` is set to `std::errc::result_out_of_range`.

We support Visual Studio, macOS, Linux, freeBSD. We support big and little endian. We support 32-bit and 64-bit systems.

We assume that the rounding mode is set to nearest (`std::fegetround() == FE_TONEAREST`).

## C++20: compile-time evaluation (constexpr)

In C++20, you may use `fast_float::from_chars` to parse strings
at compile-time, as in the following example:

```C++
// consteval forces compile-time evaluation of the function in C++20.
consteval double parse(std::string_view input) {
  double result;
  auto answer = fast_float::from_chars(input.data(), input.data()+input.size(), result);
  if(answer.ec != std::errc()) { return -1.0; }
  return result;
}

// This function should compile to a function which
// merely returns 3.1415.
constexpr double constexptest() {
  return parse("3.1415 input");
}
```

## Non-ASCII Inputs

We also support UTF-16 and UTF-32 inputs, as well as ASCII/UTF-8, as in the following example:

``` C++
#include "fast_float/fast_float.h"
#include <iostream>

int main() {
    const std::u16string input =  u"3.1416 xyz ";
    double result;
    auto answer = fast_float::from_chars(input.data(), input.data()+input.size(), result);
    if(answer.ec != std::errc()) { std::cerr << "parsing failure\n"; return EXIT_FAILURE; }
    std::cout << "parsed the number " << result << std::endl;
    return EXIT_SUCCESS;
}
```

## Using commas as decimal separator


The C++ standard stipulate that `from_chars` has to be locale-independent. In
particular, the decimal separator has to be the period (`.`). However,
some users still want to use the `fast_float` library with in a locale-dependent
manner. Using a separate function called `from_chars_advanced`, we allow the users
to pass a `parse_options` instance which contains a custom decimal separator (e.g.,
the comma). You may use it as follows.

```C++
#include "fast_float/fast_float.h"
#include <iostream>

int main() {
    const std::string input =  "3,1416 xyz ";
    double result;
    fast_float::parse_options options{fast_float::chars_format::general, ','};
    auto answer = fast_float::from_chars_advanced(input.data(), input.data()+input.size(), result, options);
    if((answer.ec != std::errc()) || ((result != 3.1416))) { std::cerr << "parsing failure\n"; return EXIT_FAILURE; }
    std::cout << "parsed the number " << result << std::endl;
    return EXIT_SUCCESS;
}
```

You can parse delimited numbers:
```C++
  const std::string input =   "234532.3426362,7869234.9823,324562.645";
  double result;
  auto answer = fast_float::from_chars(input.data(), input.data()+input.size(), result);
  if(answer.ec != std::errc()) {
    // check error
  }
  // we have result == 234532.3426362.
  if(answer.ptr[0] != ',') {
    // unexpected delimiter
  }
  answer = fast_float::from_chars(answer.ptr + 1, input.data()+input.size(), result);
  if(answer.ec != std::errc()) {
    // check error
  }
  // we have result == 7869234.9823.
  if(answer.ptr[0] != ',') {
    // unexpected delimiter
  }
  answer = fast_float::from_chars(answer.ptr + 1, input.data()+input.size(), result);
  if(answer.ec != std::errc()) {
    // check error
  }
  // we have result == 324562.645.
```


## Relation With Other Work

The fast_float library is part of:

- GCC (as of version 12): the `from_chars` function in GCC relies on fast_float.
- [WebKit](https://github.com/WebKit/WebKit), the engine behind Safari (Apple's web browser)


The fastfloat algorithm is part of the [LLVM standard libraries](https://github.com/llvm/llvm-project/commit/87c016078ad72c46505461e4ff8bfa04819fe7ba).

There is a [derived implementation part of AdaCore](https://github.com/AdaCore/VSS).


The fast_float library provides a performance similar to that of the [fast_double_parser](https://github.com/lemire/fast_double_parser) library but using an updated algorithm reworked from the ground up, and while offering an API more in line with the expectations of C++ programmers. The fast_double_parser library is part of the [Microsoft LightGBM machine-learning framework](https://github.com/microsoft/LightGBM).

## References

- Daniel Lemire, [Number Parsing at a Gigabyte per Second](https://arxiv.org/abs/2101.11408), Software: Practice and Experience 51 (8), 2021.
- Noble Mushtak, Daniel Lemire, [Fast Number Parsing Without Fallback](https://arxiv.org/abs/2212.06644), Software: Practice and Experience 53 (7), 2023.

## Other programming languages

- [There is an R binding](https://github.com/eddelbuettel/rcppfastfloat) called `rcppfastfloat`.
- [There is a Rust port of the fast_float library](https://github.com/aldanor/fast-float-rust/) called `fast-float-rust`.
- [There is a Java port of the fast_float library](https://github.com/wrandelshofer/FastDoubleParser) called `FastDoubleParser`. It used for important systems such as [Jackson](https://github.com/FasterXML/jackson-core).
- [There is a C# port of the fast_float library](https://github.com/CarlVerret/csFastFloat) called `csFastFloat`.


## Users

The fast_float library is used by [Apache Arrow](https://github.com/apache/arrow/pull/8494) where it multiplied the number parsing speed by two or three times. It is also used by [Yandex ClickHouse](https://github.com/ClickHouse/ClickHouse) and by [Google Jsonnet](https://github.com/google/jsonnet).


## How fast is it?

It can parse random floating-point numbers at a speed of 1 GB/s on some systems. We find that it is often twice as fast as the best available competitor, and many times faster than many standard-library implementations.

<img src="http://lemire.me/blog/wp-content/uploads/2020/11/fastfloat_speed.png" width="400">

```
$ ./build/benchmarks/benchmark
# parsing random integers in the range [0,1)
volume = 2.09808 MB
netlib                                  :   271.18 MB/s (+/- 1.2 %)    12.93 Mfloat/s
doubleconversion                        :   225.35 MB/s (+/- 1.2 %)    10.74 Mfloat/s
strtod                                  :   190.94 MB/s (+/- 1.6 %)     9.10 Mfloat/s
abseil                                  :   430.45 MB/s (+/- 2.2 %)    20.52 Mfloat/s
fastfloat                               :  1042.38 MB/s (+/- 9.9 %)    49.68 Mfloat/s
```

See https://github.com/lemire/simple_fastfloat_benchmark for our benchmarking code.


## Video

[![Go Systems 2020](http://img.youtube.com/vi/AVXgvlMeIm4/0.jpg)](http://www.youtube.com/watch?v=AVXgvlMeIm4)<br />

## Using as a CMake dependency

This library is header-only by design. The CMake file provides the `fast_float` target
which is merely a pointer to the `include` directory.

If you drop the `fast_float` repository in your CMake project, you should be able to use
it in this manner:

```cmake
add_subdirectory(fast_float)
target_link_libraries(myprogram PUBLIC fast_float)
```

Or you may want to retrieve the dependency automatically if you have a sufficiently recent version of CMake (3.11 or better at least):

```cmake
FetchContent_Declare(
  fast_float
  GIT_REPOSITORY https://github.com/lemire/fast_float.git
  GIT_TAG tags/v1.1.2
  GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(fast_float)
target_link_libraries(myprogram PUBLIC fast_float)

```

You should change the `GIT_TAG` line so that you recover the version you wish to use.

## Using as single header

The script `script/amalgamate.py` may be used to generate a single header
version of the library if so desired.
Just run the script from the root directory of this repository.
You can customize the license type and output file if desired as described in
the command line help.

You may directly download automatically generated single-header files:

https://github.com/fastfloat/fast_float/releases/download/v5.2.0/fast_float.h

## Credit

Though this work is inspired by many different people, this work benefited especially from exchanges with
Michael Eisel, who motivated the original research with his key insights, and with Nigel Tao who provided
invaluable feedback. Rémy Oudompheng first implemented a fast path we use in the case of long digits.

The library includes code adapted from Google Wuffs (written by Nigel Tao) which was originally published
under the Apache 2.0 license.

## License

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> or <a href="LICENSE-BOOST">BOOST license</a> .
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this repository by you, as defined in the Apache-2.0 license,
shall be triple licensed as above, without any additional terms or conditions.
</sub>
