#ifndef _C4_CHARCONV_HPP_
#define _C4_CHARCONV_HPP_

/** @file charconv.hpp Lightweight generic type-safe wrappers for
 * converting individual values to/from strings.
 *
 * These are the main functions:
 *
 * @code{.cpp}
 * // Convert the given value, writing into the string.
 * // The resulting string will NOT be null-terminated.
 * // Return the number of characters needed.
 * // This function is safe to call when the string is too small -
 * // no writes will occur beyond the string's last character.
 * template<class T> size_t c4::to_chars(substr buf, T const& C4_RESTRICT val);
 *
 *
 * // Convert the given value to a string using to_chars(), and
 * // return the resulting string, up to and including the last
 * // written character.
 * template<class T> substr c4::to_chars_sub(substr buf, T const& C4_RESTRICT val);
 *
 *
 * // Read a value from the string, which must be
 * // trimmed to the value (ie, no leading/trailing whitespace).
 * // return true if the conversion succeeded.
 * // There is no check for overflow; the value wraps around in a way similar
 * // to the standard C/C++ overflow behavior. For example,
 * // from_chars<int8_t>("128", &val) returns true and val will be
 * // set tot 0.
 * template<class T> bool c4::from_chars(csubstr buf, T * C4_RESTRICT val);
 *
 *
 * // Read the first valid sequence of characters from the string,
 * // skipping leading whitespace, and convert it using from_chars().
 * // Return the number of characters read for converting.
 * template<class T> size_t c4::from_chars_first(csubstr buf, T * C4_RESTRICT val);
 * @endcode
 */

#include "language.hpp"
#include <inttypes.h>
#include <type_traits>
#include <climits>
#include <limits>
#include <utility>

#include "config.hpp"
#include "substr.hpp"
#include "std/std_fwd.hpp"
#include "memory_util.hpp"
#include "szconv.hpp"

#ifndef C4CORE_NO_FAST_FLOAT
#   if (C4_CPP >= 17)
#       if defined(_MSC_VER)
#           if (C4_MSVC_VERSION >= C4_MSVC_VERSION_2019) // VS2017 and lower do not have these macros
#               include <charconv>
#               define C4CORE_HAVE_STD_TOCHARS 1
#               define C4CORE_HAVE_STD_FROMCHARS 0 // prefer fast_float with MSVC
#               define C4CORE_HAVE_FAST_FLOAT 1
#           else
#               define C4CORE_HAVE_STD_TOCHARS 0
#               define C4CORE_HAVE_STD_FROMCHARS 0
#               define C4CORE_HAVE_FAST_FLOAT 1
#           endif
#       else
#           if __has_include(<charconv>)
#               include <charconv>
#               if defined(__cpp_lib_to_chars)
#                   define C4CORE_HAVE_STD_TOCHARS 1
#                   define C4CORE_HAVE_STD_FROMCHARS 0 // glibc uses fast_float internally
#                   define C4CORE_HAVE_FAST_FLOAT 1
#               else
#                   define C4CORE_HAVE_STD_TOCHARS 0
#                   define C4CORE_HAVE_STD_FROMCHARS 0
#                   define C4CORE_HAVE_FAST_FLOAT 1
#               endif
#           else
#               define C4CORE_HAVE_STD_TOCHARS 0
#               define C4CORE_HAVE_STD_FROMCHARS 0
#               define C4CORE_HAVE_FAST_FLOAT 1
#           endif
#       endif
#   else
#       define C4CORE_HAVE_STD_TOCHARS 0
#       define C4CORE_HAVE_STD_FROMCHARS 0
#       define C4CORE_HAVE_FAST_FLOAT 1
#   endif
#   if C4CORE_HAVE_FAST_FLOAT
        C4_SUPPRESS_WARNING_GCC_WITH_PUSH("-Wsign-conversion")
        C4_SUPPRESS_WARNING_GCC("-Warray-bounds")
#       if defined(__GNUC__) && __GNUC__ >= 5
            C4_SUPPRESS_WARNING_GCC("-Wshift-count-overflow")
#       endif
#       include "ext/fast_float.hpp"
        C4_SUPPRESS_WARNING_GCC_POP
#   endif
#elif (C4_CPP >= 17)
#   define C4CORE_HAVE_FAST_FLOAT 0
#   if defined(_MSC_VER)
#       if (C4_MSVC_VERSION >= C4_MSVC_VERSION_2019) // VS2017 and lower do not have these macros
#           include <charconv>
#           define C4CORE_HAVE_STD_TOCHARS 1
#           define C4CORE_HAVE_STD_FROMCHARS 1
#       else
#           define C4CORE_HAVE_STD_TOCHARS 0
#           define C4CORE_HAVE_STD_FROMCHARS 0
#       endif
#   else
#       if __has_include(<charconv>)
#           include <charconv>
#           if defined(__cpp_lib_to_chars)
#               define C4CORE_HAVE_STD_TOCHARS 1
#               define C4CORE_HAVE_STD_FROMCHARS 1 // glibc uses fast_float internally
#           else
#               define C4CORE_HAVE_STD_TOCHARS 0
#               define C4CORE_HAVE_STD_FROMCHARS 0
#           endif
#       else
#           define C4CORE_HAVE_STD_TOCHARS 0
#           define C4CORE_HAVE_STD_FROMCHARS 0
#       endif
#   endif
#else
#   define C4CORE_HAVE_STD_TOCHARS 0
#   define C4CORE_HAVE_STD_FROMCHARS 0
#   define C4CORE_HAVE_FAST_FLOAT 0
#endif


#if !C4CORE_HAVE_STD_FROMCHARS
#include <cstdio>
#endif


#if defined(_MSC_VER)
#   pragma warning(push)
#   pragma warning(disable: 4996) // snprintf/scanf: this function or variable may be unsafe
#   if C4_MSVC_VERSION != C4_MSVC_VERSION_2017
#       pragma warning(disable: 4800) //'int': forcing value to bool 'true' or 'false' (performance warning)
#   endif
#endif

#if defined(__clang__)
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wtautological-constant-out-of-range-compare"
#   pragma clang diagnostic ignored "-Wformat-nonliteral"
#   pragma clang diagnostic ignored "-Wdouble-promotion" // implicit conversion increases floating-point precision
#   pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wformat-nonliteral"
#   pragma GCC diagnostic ignored "-Wdouble-promotion" // implicit conversion increases floating-point precision
#   pragma GCC diagnostic ignored "-Wuseless-cast"
#   pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

#if defined(__clang__)
#define C4_NO_UBSAN_IOVRFLW __attribute__((no_sanitize("signed-integer-overflow")))
#elif defined(__GNUC__)
#if __GNUC__ > 7
#define C4_NO_UBSAN_IOVRFLW __attribute__((no_sanitize("signed-integer-overflow")))
#else
#define C4_NO_UBSAN_IOVRFLW
#endif
#else
#define C4_NO_UBSAN_IOVRFLW
#endif


namespace c4 {

#if C4CORE_HAVE_STD_TOCHARS
/** @warning Use only the symbol. Do not rely on the type or naked value of this enum. */
typedef enum : std::underlying_type<std::chars_format>::type {
    /** print the real number in floating point format (like %f) */
    FTOA_FLOAT = static_cast<std::underlying_type<std::chars_format>::type>(std::chars_format::fixed),
    /** print the real number in scientific format (like %e) */
    FTOA_SCIENT = static_cast<std::underlying_type<std::chars_format>::type>(std::chars_format::scientific),
    /** print the real number in flexible format (like %g) */
    FTOA_FLEX = static_cast<std::underlying_type<std::chars_format>::type>(std::chars_format::general),
    /** print the real number in hexadecimal format (like %a) */
    FTOA_HEXA = static_cast<std::underlying_type<std::chars_format>::type>(std::chars_format::hex),
} RealFormat_e;
#else
/** @warning Use only the symbol. Do not rely on the type or naked value of this enum. */
typedef enum : char {
    /** print the real number in floating point format (like %f) */
    FTOA_FLOAT = 'f',
    /** print the real number in scientific format (like %e) */
    FTOA_SCIENT = 'e',
    /** print the real number in flexible format (like %g) */
    FTOA_FLEX = 'g',
    /** print the real number in hexadecimal format (like %a) */
    FTOA_HEXA = 'a',
} RealFormat_e;
#endif


/** in some platforms, int,unsigned int
 *  are not any of int8_t...int64_t and
 *  long,unsigned long are not any of uint8_t...uint64_t */
template<class T>
struct is_fixed_length
{
    enum : bool {
        /** true if T is one of the fixed length signed types */
        value_i = (std::is_integral<T>::value
                   && (std::is_same<T, int8_t>::value
                       || std::is_same<T, int16_t>::value
                       || std::is_same<T, int32_t>::value
                       || std::is_same<T, int64_t>::value)),
        /** true if T is one of the fixed length unsigned types */
        value_u = (std::is_integral<T>::value
                   && (std::is_same<T, uint8_t>::value
                       || std::is_same<T, uint16_t>::value
                       || std::is_same<T, uint32_t>::value
                       || std::is_same<T, uint64_t>::value)),
        /** true if T is one of the fixed length signed or unsigned types */
        value = value_i || value_u
    };
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

#ifdef _MSC_VER
#   pragma warning(push)
#elif defined(__clang__)
#   pragma clang diagnostic push
#elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wconversion"
#   if __GNUC__ >= 6
#       pragma GCC diagnostic ignored "-Wnull-dereference"
#   endif
#endif

namespace detail {

/* python command to get the values below:
def dec(v):
    return str(v)
for bits in (8, 16, 32, 64):
    imin, imax, umax = (-(1 << (bits - 1))), (1 << (bits - 1)) - 1, (1 << bits) - 1
    for vname, v in (("imin", imin), ("imax", imax), ("umax", umax)):
        for f in (bin, oct, dec, hex):
            print(f"{bits}b: {vname}={v} {f.__name__}: len={len(f(v)):2d}: {v} {f(v)}")
*/

// do not use the type as the template argument because in some
// platforms long!=int32 and long!=int64. Just use the numbytes
// which is more generic and spares lengthy SFINAE code.
template<size_t num_bytes, bool is_signed> struct charconv_digits_;
template<class T> using charconv_digits = charconv_digits_<sizeof(T), std::is_signed<T>::value>;

template<> struct charconv_digits_<1u, true> // int8_t
{
    enum : size_t {
        maxdigits_bin       = 1 + 2 + 8, // -128==-0b10000000
        maxdigits_oct       = 1 + 2 + 3, // -128==-0o200
        maxdigits_dec       = 1     + 3, // -128
        maxdigits_hex       = 1 + 2 + 2, // -128==-0x80
        maxdigits_bin_nopfx =         8, // -128==-0b10000000
        maxdigits_oct_nopfx =         3, // -128==-0o200
        maxdigits_dec_nopfx =         3, // -128
        maxdigits_hex_nopfx =         2, // -128==-0x80
    };
    // min values without sign!
    static constexpr csubstr min_value_dec() noexcept { return csubstr("128"); }
    static constexpr csubstr min_value_hex() noexcept { return csubstr("80"); }
    static constexpr csubstr min_value_oct() noexcept { return csubstr("200"); }
    static constexpr csubstr min_value_bin() noexcept { return csubstr("10000000"); }
    static constexpr csubstr max_value_dec() noexcept { return csubstr("127"); }
    static constexpr bool    is_oct_overflow(csubstr str) noexcept { return !((str.len < 3) || (str.len == 3 && str[0] <= '1')); }
};
template<> struct charconv_digits_<1u, false> // uint8_t
{
    enum : size_t {
        maxdigits_bin       = 2 + 8, // 255 0b11111111
        maxdigits_oct       = 2 + 3, // 255 0o377
        maxdigits_dec       =     3, // 255
        maxdigits_hex       = 2 + 2, // 255 0xff
        maxdigits_bin_nopfx =     8, // 255 0b11111111
        maxdigits_oct_nopfx =     3, // 255 0o377
        maxdigits_dec_nopfx =     3, // 255
        maxdigits_hex_nopfx =     2, // 255 0xff
    };
    static constexpr csubstr max_value_dec() noexcept { return csubstr("255"); }
    static constexpr bool    is_oct_overflow(csubstr str) noexcept { return !((str.len < 3) || (str.len == 3 && str[0] <= '3')); }
};
template<> struct charconv_digits_<2u, true> // int16_t
{
    enum : size_t {
        maxdigits_bin       = 1 + 2 + 16, // -32768 -0b1000000000000000
        maxdigits_oct       = 1 + 2 +  6, // -32768 -0o100000
        maxdigits_dec       = 1     +  5, // -32768 -32768
        maxdigits_hex       = 1 + 2 +  4, // -32768 -0x8000
        maxdigits_bin_nopfx =         16, // -32768 -0b1000000000000000
        maxdigits_oct_nopfx =          6, // -32768 -0o100000
        maxdigits_dec_nopfx =          5, // -32768 -32768
        maxdigits_hex_nopfx =          4, // -32768 -0x8000
    };
    // min values without sign!
    static constexpr csubstr min_value_dec() noexcept { return csubstr("32768"); }
    static constexpr csubstr min_value_hex() noexcept { return csubstr("8000"); }
    static constexpr csubstr min_value_oct() noexcept { return csubstr("100000"); }
    static constexpr csubstr min_value_bin() noexcept { return csubstr("1000000000000000"); }
    static constexpr csubstr max_value_dec() noexcept { return csubstr("32767"); }
    static constexpr bool    is_oct_overflow(csubstr str) noexcept { return !((str.len < 6)); }
};
template<> struct charconv_digits_<2u, false> // uint16_t
{
    enum : size_t {
        maxdigits_bin       = 2 + 16, // 65535 0b1111111111111111
        maxdigits_oct       = 2 +  6, // 65535 0o177777
        maxdigits_dec       =      6, // 65535 65535
        maxdigits_hex       = 2 +  4, // 65535 0xffff
        maxdigits_bin_nopfx =     16, // 65535 0b1111111111111111
        maxdigits_oct_nopfx =      6, // 65535 0o177777
        maxdigits_dec_nopfx =      6, // 65535 65535
        maxdigits_hex_nopfx =      4, // 65535 0xffff
    };
    static constexpr csubstr max_value_dec() noexcept { return csubstr("65535"); }
    static constexpr bool    is_oct_overflow(csubstr str) noexcept { return !((str.len < 6) || (str.len == 6 && str[0] <= '1')); }
};
template<> struct charconv_digits_<4u, true> // int32_t
{
    enum : size_t {
        maxdigits_bin       = 1 + 2 + 32, // len=35: -2147483648 -0b10000000000000000000000000000000
        maxdigits_oct       = 1 + 2 + 11, // len=14: -2147483648 -0o20000000000
        maxdigits_dec       = 1     + 10, // len=11: -2147483648 -2147483648
        maxdigits_hex       = 1 + 2 +  8, // len=11: -2147483648 -0x80000000
        maxdigits_bin_nopfx =         32, // len=35: -2147483648 -0b10000000000000000000000000000000
        maxdigits_oct_nopfx =         11, // len=14: -2147483648 -0o20000000000
        maxdigits_dec_nopfx =         10, // len=11: -2147483648 -2147483648
        maxdigits_hex_nopfx =          8, // len=11: -2147483648 -0x80000000
    };
    // min values without sign!
    static constexpr csubstr min_value_dec() noexcept { return csubstr("2147483648"); }
    static constexpr csubstr min_value_hex() noexcept { return csubstr("80000000"); }
    static constexpr csubstr min_value_oct() noexcept { return csubstr("20000000000"); }
    static constexpr csubstr min_value_bin() noexcept { return csubstr("10000000000000000000000000000000"); }
    static constexpr csubstr max_value_dec() noexcept { return csubstr("2147483647"); }
    static constexpr bool    is_oct_overflow(csubstr str) noexcept { return !((str.len < 11) || (str.len == 11 && str[0] <= '1')); }
};
template<> struct charconv_digits_<4u, false> // uint32_t
{
    enum : size_t {
        maxdigits_bin       = 2 + 32, // len=34: 4294967295 0b11111111111111111111111111111111
        maxdigits_oct       = 2 + 11, // len=13: 4294967295 0o37777777777
        maxdigits_dec       =     10, // len=10: 4294967295 4294967295
        maxdigits_hex       = 2 +  8, // len=10: 4294967295 0xffffffff
        maxdigits_bin_nopfx =     32, // len=34: 4294967295 0b11111111111111111111111111111111
        maxdigits_oct_nopfx =     11, // len=13: 4294967295 0o37777777777
        maxdigits_dec_nopfx =     10, // len=10: 4294967295 4294967295
        maxdigits_hex_nopfx =      8, // len=10: 4294967295 0xffffffff
    };
    static constexpr csubstr max_value_dec() noexcept { return csubstr("4294967295"); }
    static constexpr bool is_oct_overflow(csubstr str) noexcept { return !((str.len < 11) || (str.len == 11 && str[0] <= '3')); }
};
template<> struct charconv_digits_<8u, true> // int32_t
{
    enum : size_t {
        maxdigits_bin       = 1 + 2 + 64, // len=67: -9223372036854775808 -0b1000000000000000000000000000000000000000000000000000000000000000
        maxdigits_oct       = 1 + 2 + 22, // len=25: -9223372036854775808 -0o1000000000000000000000
        maxdigits_dec       = 1     + 19, // len=20: -9223372036854775808 -9223372036854775808
        maxdigits_hex       = 1 + 2 + 16, // len=19: -9223372036854775808 -0x8000000000000000
        maxdigits_bin_nopfx =         64, // len=67: -9223372036854775808 -0b1000000000000000000000000000000000000000000000000000000000000000
        maxdigits_oct_nopfx =         22, // len=25: -9223372036854775808 -0o1000000000000000000000
        maxdigits_dec_nopfx =         19, // len=20: -9223372036854775808 -9223372036854775808
        maxdigits_hex_nopfx =         16, // len=19: -9223372036854775808 -0x8000000000000000
    };
    static constexpr csubstr min_value_dec() noexcept { return csubstr("9223372036854775808"); }
    static constexpr csubstr min_value_hex() noexcept { return csubstr("8000000000000000"); }
    static constexpr csubstr min_value_oct() noexcept { return csubstr("1000000000000000000000"); }
    static constexpr csubstr min_value_bin() noexcept { return csubstr("1000000000000000000000000000000000000000000000000000000000000000"); }
    static constexpr csubstr max_value_dec() noexcept { return csubstr("9223372036854775807"); }
    static constexpr bool    is_oct_overflow(csubstr str) noexcept { return !((str.len < 22)); }
};
template<> struct charconv_digits_<8u, false>
{
    enum : size_t {
        maxdigits_bin       = 2 + 64, // len=66: 18446744073709551615 0b1111111111111111111111111111111111111111111111111111111111111111
        maxdigits_oct       = 2 + 22, // len=24: 18446744073709551615 0o1777777777777777777777
        maxdigits_dec       =     20, // len=20: 18446744073709551615 18446744073709551615
        maxdigits_hex       = 2 + 16, // len=18: 18446744073709551615 0xffffffffffffffff
        maxdigits_bin_nopfx =     64, // len=66: 18446744073709551615 0b1111111111111111111111111111111111111111111111111111111111111111
        maxdigits_oct_nopfx =     22, // len=24: 18446744073709551615 0o1777777777777777777777
        maxdigits_dec_nopfx =     20, // len=20: 18446744073709551615 18446744073709551615
        maxdigits_hex_nopfx =     16, // len=18: 18446744073709551615 0xffffffffffffffff
    };
    static constexpr csubstr max_value_dec() noexcept { return csubstr("18446744073709551615"); }
    static constexpr bool    is_oct_overflow(csubstr str) noexcept { return !((str.len < 22) || (str.len == 22 && str[0] <= '1')); }
};
} // namespace detail


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

// Helper macros, undefined below
#define _c4append(c) { if(C4_LIKELY(pos < buf.len)) { buf.str[pos++] = static_cast<char>(c); } else { ++pos; } }
#define _c4appendhex(i) { if(C4_LIKELY(pos < buf.len)) { buf.str[pos++] = hexchars[i]; } else { ++pos; } }

/** @name digits_dec return the number of digits required to encode a
 * decimal number.
 *
 * @note At first sight this code may look heavily branchy and
 * therefore inefficient. However, measurements revealed this to be
 * the fastest among the alternatives.
 *
 * @see https://github.com/biojppm/c4core/pull/77 */
/** @{ */

template<class T>
C4_CONSTEXPR14 C4_ALWAYS_INLINE
auto digits_dec(T v) noexcept
    -> typename std::enable_if<sizeof(T) == 1u, unsigned>::type
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v >= 0);
    return ((v >= 100) ? 3u : ((v >= 10) ? 2u : 1u));
}

template<class T>
C4_CONSTEXPR14 C4_ALWAYS_INLINE
auto digits_dec(T v) noexcept
    -> typename std::enable_if<sizeof(T) == 2u, unsigned>::type
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v >= 0);
    return ((v >= 10000) ? 5u : (v >= 1000) ? 4u : (v >= 100) ? 3u : (v >= 10) ? 2u : 1u);
}

template<class T>
C4_CONSTEXPR14 C4_ALWAYS_INLINE
auto digits_dec(T v) noexcept
    -> typename std::enable_if<sizeof(T) == 4u, unsigned>::type
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v >= 0);
    return ((v >= 1000000000) ? 10u : (v >= 100000000) ? 9u : (v >= 10000000) ? 8u :
            (v >= 1000000) ? 7u : (v >= 100000) ? 6u : (v >= 10000) ? 5u :
            (v >= 1000) ? 4u : (v >= 100) ? 3u : (v >= 10) ? 2u : 1u);
}

template<class T>
C4_CONSTEXPR14 C4_ALWAYS_INLINE
auto digits_dec(T v) noexcept
    -> typename std::enable_if<sizeof(T) == 8u, unsigned>::type
{
    // thanks @fargies!!!
    // https://github.com/biojppm/c4core/pull/77#issuecomment-1063753568
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v >= 0);
    if(v >= 1000000000) // 10
    {
        if(v >= 100000000000000) // 15 [15-20] range
        {
            if(v >= 100000000000000000) // 18 (15 + (20 - 15) / 2)
            {
                if((typename std::make_unsigned<T>::type)v >= 10000000000000000000u) // 20
                    return 20u;
                else
                    return (v >= 1000000000000000000) ? 19u : 18u;
            }
            else if(v >= 10000000000000000) // 17
                return 17u;
            else
                return(v >= 1000000000000000) ? 16u : 15u;
        }
        else if(v >= 1000000000000) // 13
            return (v >= 10000000000000) ? 14u : 13u;
        else if(v >= 100000000000) // 12
            return 12;
        else
            return(v >= 10000000000) ? 11u : 10u;
    }
    else if(v >= 10000) // 5 [5-9] range
    {
        if(v >= 10000000) // 8
            return (v >= 100000000) ? 9u : 8u;
        else if(v >= 1000000) // 7
            return 7;
        else
            return (v >= 100000) ? 6u : 5u;
    }
    else if(v >= 100)
        return (v >= 1000) ? 4u : 3u;
    else
        return (v >= 10) ? 2u : 1u;
}

/** @} */


template<class T>
C4_CONSTEXPR14 C4_ALWAYS_INLINE unsigned digits_hex(T v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v >= 0);
    return v ? 1u + (msb((typename std::make_unsigned<T>::type)v) >> 2u) : 1u;
}

template<class T>
C4_CONSTEXPR14 C4_ALWAYS_INLINE unsigned digits_bin(T v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v >= 0);
    return v ? 1u + msb((typename std::make_unsigned<T>::type)v) : 1u;
}

template<class T>
C4_CONSTEXPR14 C4_ALWAYS_INLINE unsigned digits_oct(T v_) noexcept
{
    // TODO: is there a better way?
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v_ >= 0);
    using U = typename
        std::conditional<sizeof(T) <= sizeof(unsigned),
                         unsigned,
                         typename std::make_unsigned<T>::type>::type;
    U v = (U) v_;  // safe because we require v_ >= 0
    unsigned __n = 1;
    const unsigned __b2 = 64u;
    const unsigned __b3 = __b2 * 8u;
    const unsigned long __b4 = __b3 * 8u;
    while(true)
	{
        if(v < 8u)
            return __n;
        if(v < __b2)
            return __n + 1;
        if(v < __b3)
            return __n + 2;
        if(v < __b4)
            return __n + 3;
        v /= (U) __b4;
        __n += 4;
	}
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

namespace detail {
C4_INLINE_CONSTEXPR const char hexchars[] = "0123456789abcdef";
C4_INLINE_CONSTEXPR const char digits0099[] =
    "0001020304050607080910111213141516171819"
    "2021222324252627282930313233343536373839"
    "4041424344454647484950515253545556575859"
    "6061626364656667686970717273747576777879"
    "8081828384858687888990919293949596979899";
} // namespace detail

C4_SUPPRESS_WARNING_GCC_PUSH
C4_SUPPRESS_WARNING_GCC("-Warray-bounds")  // gcc has false positives here
#if (defined(__GNUC__) && (__GNUC__ >= 7))
C4_SUPPRESS_WARNING_GCC("-Wstringop-overflow")  // gcc has false positives here
#endif

template<class T>
C4_HOT C4_ALWAYS_INLINE
void write_dec_unchecked(substr buf, T v, unsigned digits_v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v >= 0);
    C4_ASSERT(buf.len >= digits_v);
    C4_XASSERT(digits_v == digits_dec(v));
    // in bm_xtoa: checkoncelog_singlediv_write2
    while(v >= T(100))
    {
        T quo = v;
        quo /= T(100);
        const auto num = (v - quo * T(100)) << 1u;
        v = quo;
        buf.str[--digits_v] = detail::digits0099[num + 1];
        buf.str[--digits_v] = detail::digits0099[num];
    }
    if(v >= T(10))
    {
        C4_ASSERT(digits_v == 2);
        const auto num = v << 1u;
        buf.str[1] = detail::digits0099[num + 1];
        buf.str[0] = detail::digits0099[num];
    }
    else
    {
        C4_ASSERT(digits_v == 1);
        buf.str[0] = (char)('0' + v);
    }
}


template<class T>
C4_HOT C4_ALWAYS_INLINE
void write_hex_unchecked(substr buf, T v, unsigned digits_v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v >= 0);
    C4_ASSERT(buf.len >= digits_v);
    C4_XASSERT(digits_v == digits_hex(v));
    do {
        buf.str[--digits_v] = detail::hexchars[v & T(15)];
        v >>= 4;
    } while(v);
    C4_ASSERT(digits_v == 0);
}


template<class T>
C4_HOT C4_ALWAYS_INLINE
void write_oct_unchecked(substr buf, T v, unsigned digits_v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v >= 0);
    C4_ASSERT(buf.len >= digits_v);
    C4_XASSERT(digits_v == digits_oct(v));
    do {
        buf.str[--digits_v] = (char)('0' + (v & T(7)));
        v >>= 3;
    } while(v);
    C4_ASSERT(digits_v == 0);
}


template<class T>
C4_HOT C4_ALWAYS_INLINE
void write_bin_unchecked(substr buf, T v, unsigned digits_v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v >= 0);
    C4_ASSERT(buf.len >= digits_v);
    C4_XASSERT(digits_v == digits_bin(v));
    do {
        buf.str[--digits_v] = (char)('0' + (v & T(1)));
        v >>= 1;
    } while(v);
    C4_ASSERT(digits_v == 0);
}


/** write an integer to a string in decimal format. This is the
 * lowest level (and the fastest) function to do this task.
 * @note does not accept negative numbers
 * @note the resulting string is NOT zero-terminated.
 * @note it is ok to call this with an empty or too-small buffer;
 * no writes will occur, and the required size will be returned
 * @return the number of characters required for the buffer. */
template<class T>
C4_ALWAYS_INLINE size_t write_dec(substr buf, T v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v >= 0);
    unsigned digits = digits_dec(v);
    if(C4_LIKELY(buf.len >= digits))
        write_dec_unchecked(buf, v, digits);
    return digits;
}

/** write an integer to a string in hexadecimal format. This is the
 * lowest level (and the fastest) function to do this task.
 * @note does not accept negative numbers
 * @note does not prefix with 0x
 * @note the resulting string is NOT zero-terminated.
 * @note it is ok to call this with an empty or too-small buffer;
 * no writes will occur, and the required size will be returned
 * @return the number of characters required for the buffer. */
template<class T>
C4_ALWAYS_INLINE size_t write_hex(substr buf, T v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v >= 0);
    unsigned digits = digits_hex(v);
    if(C4_LIKELY(buf.len >= digits))
        write_hex_unchecked(buf, v, digits);
    return digits;
}

/** write an integer to a string in octal format. This is the
 * lowest level (and the fastest) function to do this task.
 * @note does not accept negative numbers
 * @note does not prefix with 0o
 * @note the resulting string is NOT zero-terminated.
 * @note it is ok to call this with an empty or too-small buffer;
 * no writes will occur, and the required size will be returned
 * @return the number of characters required for the buffer. */
template<class T>
C4_ALWAYS_INLINE size_t write_oct(substr buf, T v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v >= 0);
    unsigned digits = digits_oct(v);
    if(C4_LIKELY(buf.len >= digits))
        write_oct_unchecked(buf, v, digits);
    return digits;
}

/** write an integer to a string in binary format. This is the
 * lowest level (and the fastest) function to do this task.
 * @note does not accept negative numbers
 * @note does not prefix with 0b
 * @note the resulting string is NOT zero-terminated.
 * @note it is ok to call this with an empty or too-small buffer;
 * no writes will occur, and the required size will be returned
 * @return the number of characters required for the buffer. */
template<class T>
C4_ALWAYS_INLINE size_t write_bin(substr buf, T v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_ASSERT(v >= 0);
    unsigned digits = digits_bin(v);
    C4_ASSERT(digits > 0);
    if(C4_LIKELY(buf.len >= digits))
        write_bin_unchecked(buf, v, digits);
    return digits;
}


namespace detail {
template<class U> using NumberWriter = size_t (*)(substr, U);
template<class T, NumberWriter<T> writer>
size_t write_num_digits(substr buf, T v, size_t num_digits) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    size_t ret = writer(buf, v);
    if(ret >= num_digits)
        return ret;
    else if(ret >= buf.len || num_digits > buf.len)
        return num_digits;
    C4_ASSERT(num_digits >= ret);
    size_t delta = static_cast<size_t>(num_digits - ret);
    memmove(buf.str + delta, buf.str, ret);
    memset(buf.str, '0', delta);
    return num_digits;
}
} // namespace detail


/** same as c4::write_dec(), but pad with zeroes on the left
 * such that the resulting string is @p num_digits wide.
 * If the given number is requires more than num_digits, then the number prevails. */
template<class T>
C4_ALWAYS_INLINE size_t write_dec(substr buf, T val, size_t num_digits) noexcept
{
    return detail::write_num_digits<T, &write_dec<T>>(buf, val, num_digits);
}

/** same as c4::write_hex(), but pad with zeroes on the left
 * such that the resulting string is @p num_digits wide.
 * If the given number is requires more than num_digits, then the number prevails. */
template<class T>
C4_ALWAYS_INLINE size_t write_hex(substr buf, T val, size_t num_digits) noexcept
{
    return detail::write_num_digits<T, &write_hex<T>>(buf, val, num_digits);
}

/** same as c4::write_bin(), but pad with zeroes on the left
 * such that the resulting string is @p num_digits wide.
 * If the given number is requires more than num_digits, then the number prevails. */
template<class T>
C4_ALWAYS_INLINE size_t write_bin(substr buf, T val, size_t num_digits) noexcept
{
    return detail::write_num_digits<T, &write_bin<T>>(buf, val, num_digits);
}

/** same as c4::write_oct(), but pad with zeroes on the left
 * such that the resulting string is @p num_digits wide.
 * If the given number is requires more than num_digits, then the number prevails. */
template<class T>
C4_ALWAYS_INLINE size_t write_oct(substr buf, T val, size_t num_digits) noexcept
{
    return detail::write_num_digits<T, &write_oct<T>>(buf, val, num_digits);
}

C4_SUPPRESS_WARNING_GCC_POP


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


C4_SUPPRESS_WARNING_MSVC_PUSH
C4_SUPPRESS_WARNING_MSVC(4365) // '=': conversion from 'int' to 'I', signed/unsigned mismatch

/** read a decimal integer from a string. This is the
 * lowest level (and the fastest) function to do this task.
 * @note does not accept negative numbers
 * @note The string must be trimmed. Whitespace is not accepted.
 * @note the string must not be empty
 * @note there is no check for overflow; the value wraps around
 * in a way similar to the standard C/C++ overflow behavior.
 * For example, `read_dec<int8_t>("128", &val)` returns true
 * and val will be set to 0 because 127 is the max i8 value.
 * @see overflows<T>() to find out if a number string overflows a type range
 * @return true if the conversion was successful (no overflow check) */
template<class I>
C4_NO_UBSAN_IOVRFLW
C4_ALWAYS_INLINE bool read_dec(csubstr s, I *C4_RESTRICT v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<I>::value);
    C4_ASSERT(!s.empty());
    *v = 0;
    for(char c : s)
    {
        if(C4_UNLIKELY(c < '0' || c > '9'))
            return false;
        *v = (*v) * I(10) + (I(c) - I('0'));
    }
    return true;
}

/** read an hexadecimal integer from a string. This is the
 * lowest level (and the fastest) function to do this task.
 * @note does not accept negative numbers
 * @note does not accept leading 0x or 0X
 * @note the string must not be empty
 * @note the string must be trimmed. Whitespace is not accepted.
 * @note there is no check for overflow; the value wraps around
 * in a way similar to the standard C/C++ overflow behavior.
 * For example, `read_hex<int8_t>("80", &val)` returns true
 * and val will be set to 0 because 7f is the max i8 value.
 * @see overflows<T>() to find out if a number string overflows a type range
 * @return true if the conversion was successful (no overflow check) */
template<class I>
C4_NO_UBSAN_IOVRFLW
C4_ALWAYS_INLINE bool read_hex(csubstr s, I *C4_RESTRICT v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<I>::value);
    C4_ASSERT(!s.empty());
    *v = 0;
    for(char c : s)
    {
        I cv;
        if(c >= '0' && c <= '9')
            cv = I(c) - I('0');
        else if(c >= 'a' && c <= 'f')
            cv = I(10) + (I(c) - I('a'));
        else if(c >= 'A' && c <= 'F')
            cv = I(10) + (I(c) - I('A'));
        else
            return false;
        *v = (*v) * I(16) + cv;
    }
    return true;
}

/** read a binary integer from a string. This is the
 * lowest level (and the fastest) function to do this task.
 * @note does not accept negative numbers
 * @note does not accept leading 0b or 0B
 * @note the string must not be empty
 * @note the string must be trimmed. Whitespace is not accepted.
 * @note there is no check for overflow; the value wraps around
 * in a way similar to the standard C/C++ overflow behavior.
 * For example, `read_bin<int8_t>("10000000", &val)` returns true
 * and val will be set to 0 because 1111111 is the max i8 value.
 * @see overflows<T>() to find out if a number string overflows a type range
 * @return true if the conversion was successful (no overflow check) */
template<class I>
C4_NO_UBSAN_IOVRFLW
C4_ALWAYS_INLINE bool read_bin(csubstr s, I *C4_RESTRICT v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<I>::value);
    C4_ASSERT(!s.empty());
    *v = 0;
    for(char c : s)
    {
        *v <<= 1;
        if(c == '1')
            *v |= 1;
        else if(c != '0')
            return false;
    }
    return true;
}

/** read an octal integer from a string. This is the
 * lowest level (and the fastest) function to do this task.
 * @note does not accept negative numbers
 * @note does not accept leading 0o or 0O
 * @note the string must not be empty
 * @note the string must be trimmed. Whitespace is not accepted.
 * @note there is no check for overflow; the value wraps around
 * in a way similar to the standard C/C++ overflow behavior.
 * For example, `read_oct<int8_t>("200", &val)` returns true
 * and val will be set to 0 because 177 is the max i8 value.
 * @see overflows<T>() to find out if a number string overflows a type range
 * @return true if the conversion was successful (no overflow check) */
template<class I>
C4_NO_UBSAN_IOVRFLW
C4_ALWAYS_INLINE bool read_oct(csubstr s, I *C4_RESTRICT v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<I>::value);
    C4_ASSERT(!s.empty());
    *v = 0;
    for(char c : s)
    {
        if(C4_UNLIKELY(c < '0' || c > '7'))
            return false;
        *v = (*v) * I(8) + (I(c) - I('0'));
    }
    return true;
}

C4_SUPPRESS_WARNING_MSVC_POP


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

C4_SUPPRESS_WARNING_GCC_WITH_PUSH("-Wswitch-default")

namespace detail {
inline size_t _itoa2buf(substr buf, size_t pos, csubstr val) noexcept
{
    C4_ASSERT(pos + val.len <= buf.len);
    memcpy(buf.str + pos, val.str, val.len);
    return pos + val.len;
}
inline size_t _itoa2bufwithdigits(substr buf, size_t pos, size_t num_digits, csubstr val) noexcept
{
    num_digits = num_digits > val.len ? num_digits - val.len : 0;
    C4_ASSERT(num_digits + val.len <= buf.len);
    for(size_t i = 0; i < num_digits; ++i)
        _c4append('0');
    return detail::_itoa2buf(buf, pos, val);
}
template<class I>
C4_NO_INLINE size_t _itoadec2buf(substr buf) noexcept
{
    using digits_type = detail::charconv_digits<I>;
    if(C4_UNLIKELY(buf.len < digits_type::maxdigits_dec))
        return digits_type::maxdigits_dec;
    buf.str[0] = '-';
    return detail::_itoa2buf(buf, 1, digits_type::min_value_dec());
}
template<class I>
C4_NO_INLINE size_t _itoa2buf(substr buf, I radix) noexcept
{
    using digits_type = detail::charconv_digits<I>;
    size_t pos = 0;
    if(C4_LIKELY(buf.len > 0))
        buf.str[pos++] = '-';
    switch(radix)
    {
    case I(10):
        if(C4_UNLIKELY(buf.len < digits_type::maxdigits_dec))
            return digits_type::maxdigits_dec;
        pos =_itoa2buf(buf, pos, digits_type::min_value_dec());
        break;
    case I(16):
        if(C4_UNLIKELY(buf.len < digits_type::maxdigits_hex))
            return digits_type::maxdigits_hex;
        buf.str[pos++] = '0';
        buf.str[pos++] = 'x';
        pos = _itoa2buf(buf, pos, digits_type::min_value_hex());
        break;
    case I( 2):
        if(C4_UNLIKELY(buf.len < digits_type::maxdigits_bin))
            return digits_type::maxdigits_bin;
        buf.str[pos++] = '0';
        buf.str[pos++] = 'b';
        pos = _itoa2buf(buf, pos, digits_type::min_value_bin());
        break;
    case I( 8):
        if(C4_UNLIKELY(buf.len < digits_type::maxdigits_oct))
            return digits_type::maxdigits_oct;
        buf.str[pos++] = '0';
        buf.str[pos++] = 'o';
        pos = _itoa2buf(buf, pos, digits_type::min_value_oct());
        break;
    }
    return pos;
}
template<class I>
C4_NO_INLINE size_t _itoa2buf(substr buf, I radix, size_t num_digits) noexcept
{
    using digits_type = detail::charconv_digits<I>;
    size_t pos = 0;
    size_t needed_digits = 0;
    if(C4_LIKELY(buf.len > 0))
        buf.str[pos++] = '-';
    switch(radix)
    {
    case I(10):
        // add 1 to account for -
        needed_digits = num_digits+1 > digits_type::maxdigits_dec ? num_digits+1 : digits_type::maxdigits_dec;
        if(C4_UNLIKELY(buf.len < needed_digits))
            return needed_digits;
        pos = _itoa2bufwithdigits(buf, pos, num_digits, digits_type::min_value_dec());
        break;
    case I(16):
        // add 3 to account for -0x
        needed_digits = num_digits+3 > digits_type::maxdigits_hex ? num_digits+3 : digits_type::maxdigits_hex;
        if(C4_UNLIKELY(buf.len < needed_digits))
            return needed_digits;
        buf.str[pos++] = '0';
        buf.str[pos++] = 'x';
        pos = _itoa2bufwithdigits(buf, pos, num_digits, digits_type::min_value_hex());
        break;
    case I(2):
        // add 3 to account for -0b
        needed_digits = num_digits+3 > digits_type::maxdigits_bin ? num_digits+3 : digits_type::maxdigits_bin;
        if(C4_UNLIKELY(buf.len < needed_digits))
            return needed_digits;
        C4_ASSERT(buf.len >= digits_type::maxdigits_bin);
        buf.str[pos++] = '0';
        buf.str[pos++] = 'b';
        pos = _itoa2bufwithdigits(buf, pos, num_digits, digits_type::min_value_bin());
        break;
    case I(8):
        // add 3 to account for -0o
        needed_digits = num_digits+3 > digits_type::maxdigits_oct ? num_digits+3 : digits_type::maxdigits_oct;
        if(C4_UNLIKELY(buf.len < needed_digits))
            return needed_digits;
        C4_ASSERT(buf.len >= digits_type::maxdigits_oct);
        buf.str[pos++] = '0';
        buf.str[pos++] = 'o';
        pos = _itoa2bufwithdigits(buf, pos, num_digits, digits_type::min_value_oct());
        break;
    }
    return pos;
}
} // namespace detail


/** convert an integral signed decimal to a string.
 * @note the resulting string is NOT zero-terminated.
 * @note it is ok to call this with an empty or too-small buffer;
 * no writes will occur, and the needed size will be returned
 * @return the number of characters required for the buffer. */
template<class T>
C4_ALWAYS_INLINE size_t itoa(substr buf, T v) noexcept
{
    C4_STATIC_ASSERT(std::is_signed<T>::value);
    if(v >= T(0))
    {
        // write_dec() checks the buffer size, so no need to check here
        return write_dec(buf, v);
    }
    // when T is the min value (eg i8: -128), negating it
    // will overflow, so treat the min as a special case
    else if(C4_LIKELY(v != std::numeric_limits<T>::min()))
    {
        v = -v;
        unsigned digits = digits_dec(v);
        if(C4_LIKELY(buf.len >= digits + 1u))
        {
            buf.str[0] = '-';
            write_dec_unchecked(buf.sub(1), v, digits);
        }
        return digits + 1u;
    }
    return detail::_itoadec2buf<T>(buf);
}

/** convert an integral signed integer to a string, using a specific
 * radix. The radix must be 2, 8, 10 or 16.
 *
 * @note the resulting string is NOT zero-terminated.
 * @note it is ok to call this with an empty or too-small buffer;
 * no writes will occur, and the needed size will be returned
 * @return the number of characters required for the buffer. */
template<class T>
C4_ALWAYS_INLINE size_t itoa(substr buf, T v, T radix) noexcept
{
    C4_STATIC_ASSERT(std::is_signed<T>::value);
    C4_ASSERT(radix == 2 || radix == 8 || radix == 10 || radix == 16);
    C4_SUPPRESS_WARNING_GCC_PUSH
    #if (defined(__GNUC__) && (__GNUC__ >= 7))
        C4_SUPPRESS_WARNING_GCC("-Wstringop-overflow")  // gcc has a false positive here
    #endif
    // when T is the min value (eg i8: -128), negating it
    // will overflow, so treat the min as a special case
    if(C4_LIKELY(v != std::numeric_limits<T>::min()))
    {
        unsigned pos = 0;
        if(v < 0)
        {
            v = -v;
            if(C4_LIKELY(buf.len > 0))
                buf.str[pos] = '-';
            ++pos;
        }
        unsigned digits = 0;
        switch(radix)
        {
        case T(10):
            digits = digits_dec(v);
            if(C4_LIKELY(buf.len >= pos + digits))
                write_dec_unchecked(buf.sub(pos), v, digits);
            break;
        case T(16):
            digits = digits_hex(v);
            if(C4_LIKELY(buf.len >= pos + 2u + digits))
            {
                buf.str[pos + 0] = '0';
                buf.str[pos + 1] = 'x';
                write_hex_unchecked(buf.sub(pos + 2), v, digits);
            }
            digits += 2u;
            break;
        case T(2):
            digits = digits_bin(v);
            if(C4_LIKELY(buf.len >= pos + 2u + digits))
            {
                buf.str[pos + 0] = '0';
                buf.str[pos + 1] = 'b';
                write_bin_unchecked(buf.sub(pos + 2), v, digits);
            }
            digits += 2u;
            break;
        case T(8):
            digits = digits_oct(v);
            if(C4_LIKELY(buf.len >= pos + 2u + digits))
            {
                buf.str[pos + 0] = '0';
                buf.str[pos + 1] = 'o';
                write_oct_unchecked(buf.sub(pos + 2), v, digits);
            }
            digits += 2u;
            break;
        }
        return pos + digits;
    }
    C4_SUPPRESS_WARNING_GCC_POP
    // when T is the min value (eg i8: -128), negating it
    // will overflow
    return detail::_itoa2buf<T>(buf, radix);
}


/** same as c4::itoa(), but pad with zeroes on the left such that the
 * resulting string is @p num_digits wide, not accounting for radix
 * prefix (0x,0o,0b). The @p radix must be 2, 8, 10 or 16.
 *
 * @note the resulting string is NOT zero-terminated.
 * @note it is ok to call this with an empty or too-small buffer;
 * no writes will occur, and the needed size will be returned
 * @return the number of characters required for the buffer. */
template<class T>
C4_ALWAYS_INLINE size_t itoa(substr buf, T v, T radix, size_t num_digits) noexcept
{
    C4_STATIC_ASSERT(std::is_signed<T>::value);
    C4_ASSERT(radix == 2 || radix == 8 || radix == 10 || radix == 16);
    C4_SUPPRESS_WARNING_GCC_PUSH
    #if (defined(__GNUC__) && (__GNUC__ >= 7))
        C4_SUPPRESS_WARNING_GCC("-Wstringop-overflow")  // gcc has a false positive here
    #endif
    // when T is the min value (eg i8: -128), negating it
    // will overflow, so treat the min as a special case
    if(C4_LIKELY(v != std::numeric_limits<T>::min()))
    {
        unsigned pos = 0;
        if(v < 0)
        {
            v = -v;
            if(C4_LIKELY(buf.len > 0))
                buf.str[pos] = '-';
            ++pos;
        }
        unsigned total_digits = 0;
        switch(radix)
        {
        case T(10):
            total_digits = digits_dec(v);
            total_digits = pos + (unsigned)(num_digits > total_digits ? num_digits : total_digits);
            if(C4_LIKELY(buf.len >= total_digits))
                write_dec(buf.sub(pos), v, num_digits);
            break;
        case T(16):
            total_digits = digits_hex(v);
            total_digits = pos + 2u + (unsigned)(num_digits > total_digits ? num_digits : total_digits);
            if(C4_LIKELY(buf.len >= total_digits))
            {
                buf.str[pos + 0] = '0';
                buf.str[pos + 1] = 'x';
                write_hex(buf.sub(pos + 2), v, num_digits);
            }
            break;
        case T(2):
            total_digits = digits_bin(v);
            total_digits = pos + 2u + (unsigned)(num_digits > total_digits ? num_digits : total_digits);
            if(C4_LIKELY(buf.len >= total_digits))
            {
                buf.str[pos + 0] = '0';
                buf.str[pos + 1] = 'b';
                write_bin(buf.sub(pos + 2), v, num_digits);
            }
            break;
        case T(8):
            total_digits = digits_oct(v);
            total_digits = pos + 2u + (unsigned)(num_digits > total_digits ? num_digits : total_digits);
            if(C4_LIKELY(buf.len >= total_digits))
            {
                buf.str[pos + 0] = '0';
                buf.str[pos + 1] = 'o';
                write_oct(buf.sub(pos + 2), v, num_digits);
            }
            break;
        }
        return total_digits;
    }
    C4_SUPPRESS_WARNING_GCC_POP
    // when T is the min value (eg i8: -128), negating it
    // will overflow
    return detail::_itoa2buf<T>(buf, radix, num_digits);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** convert an integral unsigned decimal to a string.
 *
 * @note the resulting string is NOT zero-terminated.
 * @note it is ok to call this with an empty or too-small buffer;
 * no writes will occur, and the needed size will be returned
 * @return the number of characters required for the buffer. */
template<class T>
C4_ALWAYS_INLINE size_t utoa(substr buf, T v) noexcept
{
    C4_STATIC_ASSERT(std::is_unsigned<T>::value);
    // write_dec() does the buffer length check, so no need to check here
    return write_dec(buf, v);
}

/** convert an integral unsigned integer to a string, using a specific
 * radix. The radix must be 2, 8, 10 or 16.
 *
 * @note the resulting string is NOT zero-terminated.
 * @note it is ok to call this with an empty or too-small buffer;
 * no writes will occur, and the needed size will be returned
 * @return the number of characters required for the buffer. */
template<class T>
C4_ALWAYS_INLINE size_t utoa(substr buf, T v, T radix) noexcept
{
    C4_STATIC_ASSERT(std::is_unsigned<T>::value);
    C4_ASSERT(radix == 10 || radix == 16 || radix == 2 || radix == 8);
    unsigned digits = 0;
    switch(radix)
    {
    case T(10):
        digits = digits_dec(v);
        if(C4_LIKELY(buf.len >= digits))
            write_dec_unchecked(buf, v, digits);
        break;
    case T(16):
        digits = digits_hex(v);
        if(C4_LIKELY(buf.len >= digits+2u))
        {
            buf.str[0] = '0';
            buf.str[1] = 'x';
            write_hex_unchecked(buf.sub(2), v, digits);
        }
        digits += 2u;
        break;
    case T(2):
        digits = digits_bin(v);
        if(C4_LIKELY(buf.len >= digits+2u))
        {
            buf.str[0] = '0';
            buf.str[1] = 'b';
            write_bin_unchecked(buf.sub(2), v, digits);
        }
        digits += 2u;
        break;
    case T(8):
        digits = digits_oct(v);
        if(C4_LIKELY(buf.len >= digits+2u))
        {
            buf.str[0] = '0';
            buf.str[1] = 'o';
            write_oct_unchecked(buf.sub(2), v, digits);
        }
        digits += 2u;
        break;
    }
    return digits;
}

/** same as c4::utoa(), but pad with zeroes on the left such that the
 * resulting string is @p num_digits wide. The @p radix must be 2,
 * 8, 10 or 16.
 *
 * @note the resulting string is NOT zero-terminated.
 * @note it is ok to call this with an empty or too-small buffer;
 * no writes will occur, and the needed size will be returned
 * @return the number of characters required for the buffer. */
template<class T>
C4_ALWAYS_INLINE size_t utoa(substr buf, T v, T radix, size_t num_digits) noexcept
{
    C4_STATIC_ASSERT(std::is_unsigned<T>::value);
    C4_ASSERT(radix == 10 || radix == 16 || radix == 2 || radix == 8);
    unsigned total_digits = 0;
    switch(radix)
    {
    case T(10):
        total_digits = digits_dec(v);
        total_digits = (unsigned)(num_digits > total_digits ? num_digits : total_digits);
        if(C4_LIKELY(buf.len >= total_digits))
            write_dec(buf, v, num_digits);
        break;
    case T(16):
        total_digits = digits_hex(v);
        total_digits = 2u + (unsigned)(num_digits > total_digits ? num_digits : total_digits);
        if(C4_LIKELY(buf.len >= total_digits))
        {
            buf.str[0] = '0';
            buf.str[1] = 'x';
            write_hex(buf.sub(2), v, num_digits);
        }
        break;
    case T(2):
        total_digits = digits_bin(v);
        total_digits = 2u + (unsigned)(num_digits > total_digits ? num_digits : total_digits);
        if(C4_LIKELY(buf.len >= total_digits))
        {
            buf.str[0] = '0';
            buf.str[1] = 'b';
            write_bin(buf.sub(2), v, num_digits);
        }
        break;
    case T(8):
        total_digits = digits_oct(v);
        total_digits = 2u + (unsigned)(num_digits > total_digits ? num_digits : total_digits);
        if(C4_LIKELY(buf.len >= total_digits))
        {
            buf.str[0] = '0';
            buf.str[1] = 'o';
            write_oct(buf.sub(2), v, num_digits);
        }
        break;
    }
    return total_digits;
}
C4_SUPPRESS_WARNING_GCC_POP


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** Convert a trimmed string to a signed integral value. The input
 * string can be formatted as decimal, binary (prefix 0b or 0B), octal
 * (prefix 0o or 0O) or hexadecimal (prefix 0x or 0X). Strings with
 * leading zeroes are considered as decimal and not octal (unlike the
 * C/C++ convention). Every character in the input string is read for
 * the conversion; the input string must not contain any leading or
 * trailing whitespace.
 *
 * @return true if the conversion was successful.
 *
 * @note overflow is not detected: the return status is true even if
 * the conversion would return a value outside of the type's range, in
 * which case the result will wrap around the type's range.
 * This is similar to native behavior.
 *
 * @note a positive sign is not accepted. ie, the string must not
 * start with '+'
 *
 * @see atoi_first() if the string is not trimmed to the value to read. */
template<class T>
C4_NO_UBSAN_IOVRFLW
C4_ALWAYS_INLINE bool atoi(csubstr str, T * C4_RESTRICT v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    C4_STATIC_ASSERT(std::is_signed<T>::value);

    if(C4_UNLIKELY(str.len == 0))
        return false;

    C4_ASSERT(str.str[0] != '+');

    T sign = 1;
    size_t start = 0;
    if(str.str[0] == '-')
    {
        if(C4_UNLIKELY(str.len == ++start))
            return false;
        sign = -1;
    }

    bool parsed_ok = true;
    if(str.str[start] != '0') // this should be the common case, so put it first
    {
        parsed_ok = read_dec(str.sub(start), v);
    }
    else if(str.len > start + 1)
    {
        // starts with 0: is it 0x, 0o, 0b?
        const char pfx = str.str[start + 1];
        if(pfx == 'x' || pfx == 'X')
            parsed_ok = str.len > start + 2 && read_hex(str.sub(start + 2), v);
        else if(pfx == 'b' || pfx == 'B')
            parsed_ok = str.len > start + 2 && read_bin(str.sub(start + 2), v);
        else if(pfx == 'o' || pfx == 'O')
            parsed_ok = str.len > start + 2 && read_oct(str.sub(start + 2), v);
        else
            parsed_ok = read_dec(str.sub(start + 1), v);
    }
    else
    {
        parsed_ok = read_dec(str.sub(start), v);
    }
    if(C4_LIKELY(parsed_ok))
        *v *= sign;
    return parsed_ok;
}


/** Select the next range of characters in the string that can be parsed
 * as a signed integral value, and convert it using atoi(). Leading
 * whitespace (space, newline, tabs) is skipped.
 * @return the number of characters read for conversion, or csubstr::npos if the conversion failed
 * @see atoi() if the string is already trimmed to the value to read.
 * @see csubstr::first_int_span() */
template<class T>
C4_ALWAYS_INLINE size_t atoi_first(csubstr str, T * C4_RESTRICT v)
{
    csubstr trimmed = str.first_int_span();
    if(trimmed.len == 0)
        return csubstr::npos;
    if(atoi(trimmed, v))
        return static_cast<size_t>(trimmed.end() - str.begin());
    return csubstr::npos;
}


//-----------------------------------------------------------------------------

/** Convert a trimmed string to an unsigned integral value. The string can be
 * formatted as decimal, binary (prefix 0b or 0B), octal (prefix 0o or 0O)
 * or hexadecimal (prefix 0x or 0X). Every character in the input string is read
 * for the conversion; it must not contain any leading or trailing whitespace.
 *
 * @return true if the conversion was successful.
 *
 * @note overflow is not detected: the return status is true even if
 * the conversion would return a value outside of the type's range, in
 * which case the result will wrap around the type's range.
 *
 * @note If the string has a minus character, the return status
 * will be false.
 *
 * @see atou_first() if the string is not trimmed to the value to read. */
template<class T>
bool atou(csubstr str, T * C4_RESTRICT v) noexcept
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);

    if(C4_UNLIKELY(str.len == 0 || str.front() == '-'))
        return false;

    bool parsed_ok = true;
    if(str.str[0] != '0')
    {
        parsed_ok = read_dec(str, v);
    }
    else
    {
        if(str.len > 1)
        {
            const char pfx = str.str[1];
            if(pfx == 'x' || pfx == 'X')
                parsed_ok = str.len > 2 && read_hex(str.sub(2), v);
            else if(pfx == 'b' || pfx == 'B')
                parsed_ok = str.len > 2 && read_bin(str.sub(2), v);
            else if(pfx == 'o' || pfx == 'O')
                parsed_ok = str.len > 2 && read_oct(str.sub(2), v);
            else
                parsed_ok = read_dec(str, v);
        }
        else
        {
            *v = 0; // we know the first character is 0
        }
    }
    return parsed_ok;
}


/** Select the next range of characters in the string that can be parsed
 * as an unsigned integral value, and convert it using atou(). Leading
 * whitespace (space, newline, tabs) is skipped.
 * @return the number of characters read for conversion, or csubstr::npos if the conversion faileds
 * @see atou() if the string is already trimmed to the value to read.
 * @see csubstr::first_uint_span() */
template<class T>
C4_ALWAYS_INLINE size_t atou_first(csubstr str, T *v)
{
    csubstr trimmed = str.first_uint_span();
    if(trimmed.len == 0)
        return csubstr::npos;
    if(atou(trimmed, v))
        return static_cast<size_t>(trimmed.end() - str.begin());
    return csubstr::npos;
}


#ifdef _MSC_VER
#   pragma warning(pop)
#elif defined(__clang__)
#   pragma clang diagnostic pop
#elif defined(__GNUC__)
#   pragma GCC diagnostic pop
#endif


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
namespace detail {
inline bool check_overflow(csubstr str, csubstr limit) noexcept
{
    if(str.len == limit.len)
    {
        for(size_t i = 0; i < limit.len; ++i)
        {
            if(str[i] < limit[i])
                return false;
            else if(str[i] > limit[i])
                return true;
        }
        return false;
    }
    else
        return str.len > limit.len;
}
} // namespace detail


/** Test if the following string would overflow when converted to associated
 * types.
 * @return true if number will overflow, false if it fits (or doesn't parse)
 */
template<class T>
auto overflows(csubstr str) noexcept
    -> typename std::enable_if<std::is_unsigned<T>::value, bool>::type 
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);

    if(C4_UNLIKELY(str.len == 0))
    {
        return false;
    }
    else if(str.str[0] == '0')
    {
        if (str.len == 1)
            return false;
        switch (str.str[1])
        {
            case 'x':
            case 'X':
            {
                size_t fno = str.first_not_of('0', 2);
                if (fno == csubstr::npos)
                    return false;
                return !(str.len <= fno + (sizeof(T) * 2));
            }
            case 'b':
            case 'B':
            {
                size_t fno = str.first_not_of('0', 2);
                if (fno == csubstr::npos)
                    return false;
                return !(str.len <= fno +(sizeof(T) * 8));
            }
            case 'o':
            case 'O':
            {
                size_t fno = str.first_not_of('0', 2);
                if(fno == csubstr::npos)
                    return false;
                return detail::charconv_digits<T>::is_oct_overflow(str.sub(fno));
            }
            default:
            {
                size_t fno = str.first_not_of('0', 1);
                if(fno == csubstr::npos)
                    return false;
                return detail::check_overflow(str.sub(fno), detail::charconv_digits<T>::max_value_dec());
            }
        }
    }
    else if(C4_UNLIKELY(str[0] == '-'))
    {
        return true;
    }
    else
    {
        return detail::check_overflow(str, detail::charconv_digits<T>::max_value_dec());
    }
}


/** Test if the following string would overflow when converted to associated
 * types.
 * @return true if number will overflow, false if it fits (or doesn't parse)
 */
template<class T>
auto overflows(csubstr str)
    -> typename std::enable_if<std::is_signed<T>::value, bool>::type 
{
    C4_STATIC_ASSERT(std::is_integral<T>::value);
    if(C4_UNLIKELY(str.len == 0))
        return false;
    if(str.str[0] == '-')
    {
        if(str.str[1] == '0')
        {
            if(str.len == 2)
                return false;
            switch(str.str[2])
            {
                case 'x':
                case 'X':
                {
                    size_t fno = str.first_not_of('0', 3);
                    if (fno == csubstr::npos)
                        return false;
                    return detail::check_overflow(str.sub(fno), detail::charconv_digits<T>::min_value_hex());
                }
                case 'b':
                case 'B':
                {
                    size_t fno = str.first_not_of('0', 3);
                    if (fno == csubstr::npos)
                        return false;
                    return detail::check_overflow(str.sub(fno), detail::charconv_digits<T>::min_value_bin());
                }
                case 'o':
                case 'O':
                {
                    size_t fno = str.first_not_of('0', 3);
                    if(fno == csubstr::npos)
                        return false;
                    return detail::check_overflow(str.sub(fno), detail::charconv_digits<T>::min_value_oct());
                }
                default:
                {
                    size_t fno = str.first_not_of('0', 2);
                    if(fno == csubstr::npos)
                        return false;
                    return detail::check_overflow(str.sub(fno), detail::charconv_digits<T>::min_value_dec());
                }
            }
        }
        else
            return detail::check_overflow(str.sub(1), detail::charconv_digits<T>::min_value_dec());
    }
    else if(str.str[0] == '0')
    {
        if (str.len == 1)
            return false;
        switch(str.str[1])
        {
            case 'x':
            case 'X':
            {
                size_t fno = str.first_not_of('0', 2);
                if (fno == csubstr::npos)
                    return false;
                const size_t len = str.len - fno;
                return !((len < sizeof (T) * 2) || (len == sizeof(T) * 2 && str[fno] <= '7'));
            }
            case 'b':
            case 'B':
            {
                size_t fno = str.first_not_of('0', 2);
                if (fno == csubstr::npos)
                    return false;
                return !(str.len <= fno + (sizeof(T) * 8 - 1));
            }
            case 'o':
            case 'O':
            {
                size_t fno = str.first_not_of('0', 2);
                if(fno == csubstr::npos)
                    return false;
                return detail::charconv_digits<T>::is_oct_overflow(str.sub(fno));
            }
            default:
            {
                size_t fno = str.first_not_of('0', 1);
                if(fno == csubstr::npos)
                    return false;
                return detail::check_overflow(str.sub(fno), detail::charconv_digits<T>::max_value_dec());
            }
        }
    }
    else
        return detail::check_overflow(str, detail::charconv_digits<T>::max_value_dec());
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

namespace detail {


#if (!C4CORE_HAVE_STD_FROMCHARS)
/** @see http://www.exploringbinary.com/ for many good examples on float-str conversion */
template<size_t N>
void get_real_format_str(char (& C4_RESTRICT fmt)[N], int precision, RealFormat_e formatting, const char* length_modifier="")
{
    int iret;
    if(precision == -1)
        iret = snprintf(fmt, sizeof(fmt), "%%%s%c", length_modifier, formatting);
    else if(precision == 0)
        iret = snprintf(fmt, sizeof(fmt), "%%.%s%c", length_modifier, formatting);
    else
        iret = snprintf(fmt, sizeof(fmt), "%%.%d%s%c", precision, length_modifier, formatting);
    C4_ASSERT(iret >= 2 && size_t(iret) < sizeof(fmt));
    C4_UNUSED(iret);
}


/** @todo we're depending on snprintf()/sscanf() for converting to/from
 * floating point numbers. Apparently, this increases the binary size
 * by a considerable amount. There are some lightweight printf
 * implementations:
 *
 * @see http://www.sparetimelabs.com/tinyprintf/tinyprintf.php (BSD)
 * @see https://github.com/weiss/c99-snprintf
 * @see https://github.com/nothings/stb/blob/master/stb_sprintf.h
 * @see http://www.exploringbinary.com/
 * @see https://blog.benoitblanchon.fr/lightweight-float-to-string/
 * @see http://www.ryanjuckett.com/programming/printing-floating-point-numbers/
 */
template<class T>
size_t print_one(substr str, const char* full_fmt, T v)
{
#ifdef _MSC_VER
    /** use _snprintf() to prevent early termination of the output
     * for writing the null character at the last position
     * @see https://msdn.microsoft.com/en-us/library/2ts7cx93.aspx */
    int iret = _snprintf(str.str, str.len, full_fmt, v);
    if(iret < 0)
    {
        /* when buf.len is not enough, VS returns a negative value.
         * so call it again with a negative value for getting an
         * actual length of the string */
        iret = snprintf(nullptr, 0, full_fmt, v);
        C4_ASSERT(iret > 0);
    }
    size_t ret = (size_t) iret;
    return ret;
#else
    int iret = snprintf(str.str, str.len, full_fmt, v);
    C4_ASSERT(iret >= 0);
    size_t ret = (size_t) iret;
    if(ret >= str.len)
        ++ret; /* snprintf() reserves the last character to write \0 */
    return ret;
#endif
}
#endif // (!C4CORE_HAVE_STD_FROMCHARS)


#if (!C4CORE_HAVE_STD_FROMCHARS) && (!C4CORE_HAVE_FAST_FLOAT)
/** scans a string using the given type format, while at the same time
 * allowing non-null-terminated strings AND guaranteeing that the given
 * string length is strictly respected, so that no buffer overflows
 * might occur. */
template<typename T>
inline size_t scan_one(csubstr str, const char *type_fmt, T *v)
{
    /* snscanf() is absolutely needed here as we must be sure that
     * str.len is strictly respected, because substr is
     * generally not null-terminated.
     *
     * Alas, there is no snscanf().
     *
     * So we fake it by using a dynamic format with an explicit
     * field size set to the length of the given span.
     * This trick is taken from:
     * https://stackoverflow.com/a/18368910/5875572 */

    /* this is the actual format we'll use for scanning */
    char fmt[16];

    /* write the length into it. Eg "%12f".
     * Also, get the number of characters read from the string.
     * So the final format ends up as "%12f%n"*/
    int iret = std::snprintf(fmt, sizeof(fmt), "%%" "%zu" "%s" "%%n", str.len, type_fmt);
    /* no nasty surprises, please! */
    C4_ASSERT(iret >= 0 && size_t(iret) < C4_COUNTOF(fmt));

    /* now we scan with confidence that the span length is respected */
    int num_chars;
    iret = std::sscanf(str.str, fmt, v, &num_chars);
    /* scanf returns the number of successful conversions */
    if(iret != 1) return csubstr::npos;
    C4_ASSERT(num_chars >= 0);
    return (size_t)(num_chars);
}
#endif // (!C4CORE_HAVE_STD_FROMCHARS) && (!C4CORE_HAVE_FAST_FLOAT)


#if C4CORE_HAVE_STD_TOCHARS
template<class T>
C4_ALWAYS_INLINE size_t rtoa(substr buf, T v, int precision=-1, RealFormat_e formatting=FTOA_FLEX) noexcept
{
    std::to_chars_result result;
    size_t pos = 0;
    if(formatting == FTOA_HEXA)
    {
        if(buf.len > size_t(2))
        {
            buf.str[0] = '0';
            buf.str[1] = 'x';
        }
        pos += size_t(2);
    }
    if(precision == -1)
        result = std::to_chars(buf.str + pos, buf.str + buf.len, v, (std::chars_format)formatting);
    else
        result = std::to_chars(buf.str + pos, buf.str + buf.len, v, (std::chars_format)formatting, precision);
    if(result.ec == std::errc())
    {
        // all good, no errors.
        C4_ASSERT(result.ptr >= buf.str);
        ptrdiff_t delta = result.ptr - buf.str;
        return static_cast<size_t>(delta);
    }
    C4_ASSERT(result.ec == std::errc::value_too_large);
    // This is unfortunate.
    //
    // When the result can't fit in the given buffer,
    // std::to_chars() returns the end pointer it was originally
    // given, which is useless because here we would like to know
    // _exactly_ how many characters the buffer must have to fit
    // the result.
    //
    // So we take the pessimistic view, and assume as many digits
    // as could ever be required:
    size_t ret = static_cast<size_t>(std::numeric_limits<T>::max_digits10);
    return ret > buf.len ? ret : buf.len + 1;
}
#endif // C4CORE_HAVE_STD_TOCHARS


#if C4CORE_HAVE_FAST_FLOAT
template<class T>
C4_ALWAYS_INLINE bool scan_rhex(csubstr s, T *C4_RESTRICT val) noexcept
{
    C4_ASSERT(s.len > 0);
    C4_ASSERT(s.str[0] != '-');
    C4_ASSERT(s.str[0] != '+');
    C4_ASSERT(!s.begins_with("0x"));
    C4_ASSERT(!s.begins_with("0X"));
    size_t pos = 0;
    // integer part
    for( ; pos < s.len; ++pos)
    {
        const char c = s.str[pos];
        if(c >= '0' && c <= '9')
            *val = *val * T(16) + T(c - '0');
        else if(c >= 'a' && c <= 'f')
            *val = *val * T(16) + T(c - 'a');
        else if(c >= 'A' && c <= 'F')
            *val = *val * T(16) + T(c - 'A');
        else if(c == '.')
        {
            ++pos;
            break; // follow on to mantissa
        }
        else if(c == 'p' || c == 'P')
        {
            ++pos;
            goto power; // no mantissa given, jump to power
        }
        else
        {
            return false;
        }
    }
    // mantissa
    {
        // 0.0625 == 1/16 == value of first digit after the comma
        for(T digit = T(0.0625); pos < s.len; ++pos, digit /= T(16))
        {
            const char c = s.str[pos];
            if(c >= '0' && c <= '9')
                *val += digit * T(c - '0');
            else if(c >= 'a' && c <= 'f')
                *val += digit * T(c - 'a');
            else if(c >= 'A' && c <= 'F')
                *val += digit * T(c - 'A');
            else if(c == 'p' || c == 'P')
            {
                ++pos;
                goto power; // mantissa finished, jump to power
            }
            else
            {
                return false;
            }
        }
    }
    return true;
power:
    if(C4_LIKELY(pos < s.len))
    {
        if(s.str[pos] == '+') // atoi() cannot handle a leading '+'
            ++pos;
        if(C4_LIKELY(pos < s.len))
        {
            int16_t powval = {};
            if(C4_LIKELY(atoi(s.sub(pos), &powval)))
            {
                *val *= ipow<T, int16_t, 16>(powval);
                return true;
            }
        }
    }
    return false;
}
#endif

} // namespace detail


#undef _c4appendhex
#undef _c4append


/** Convert a single-precision real number to string.  The string will
 * in general be NOT null-terminated.  For FTOA_FLEX, \p precision is
 * the number of significand digits. Otherwise \p precision is the
 * number of decimals. It is safe to call this function with an empty
 * or too-small buffer.
 *
 * @return the size of the buffer needed to write the number
 */
C4_ALWAYS_INLINE size_t ftoa(substr str, float v, int precision=-1, RealFormat_e formatting=FTOA_FLEX) noexcept
{
#if C4CORE_HAVE_STD_TOCHARS
    return detail::rtoa(str, v, precision, formatting);
#else
    char fmt[16];
    detail::get_real_format_str(fmt, precision, formatting, /*length_modifier*/"");
    return detail::print_one(str, fmt, v);
#endif
}


/** Convert a double-precision real number to string.  The string will
 * in general be NOT null-terminated.  For FTOA_FLEX, \p precision is
 * the number of significand digits. Otherwise \p precision is the
 * number of decimals. It is safe to call this function with an empty
 * or too-small buffer.
 *
 * @return the size of the buffer needed to write the number
 */
C4_ALWAYS_INLINE size_t dtoa(substr str, double v, int precision=-1, RealFormat_e formatting=FTOA_FLEX) noexcept
{
#if C4CORE_HAVE_STD_TOCHARS
    return detail::rtoa(str, v, precision, formatting);
#else
    char fmt[16];
    detail::get_real_format_str(fmt, precision, formatting, /*length_modifier*/"l");
    return detail::print_one(str, fmt, v);
#endif
}


/** Convert a string to a single precision real number.
 * The input string must be trimmed to the value, ie
 * no leading or trailing whitespace can be present.
 * @return true iff the conversion succeeded
 * @see atof_first() if the string is not trimmed
 */
C4_ALWAYS_INLINE bool atof(csubstr str, float * C4_RESTRICT v) noexcept
{
    C4_ASSERT(str.len > 0);
    C4_ASSERT(str.triml(" \r\t\n").len == str.len);
#if C4CORE_HAVE_FAST_FLOAT
    // fastfloat cannot parse hexadecimal floats
    bool isneg = (str.str[0] == '-');
    csubstr rem = str.sub(isneg || str.str[0] == '+');
    if(!(rem.len >= 2 && (rem.str[0] == '0' && (rem.str[1] == 'x' || rem.str[1] == 'X'))))
    {
        fast_float::from_chars_result result;
        result = fast_float::from_chars(str.str, str.str + str.len, *v);
        return result.ec == std::errc();
    }
    else if(detail::scan_rhex(rem.sub(2), v))
    {
        *v *= isneg ? -1.f : 1.f;
        return true;
    }
    return false;
#elif C4CORE_HAVE_STD_FROMCHARS
    std::from_chars_result result;
    result = std::from_chars(str.str, str.str + str.len, *v);
    return result.ec == std::errc();
#else
    csubstr rem = str.sub(str.str[0] == '-' || str.str[0] == '+');
    if(!(rem.len >= 2 && (rem.str[0] == '0' && (rem.str[1] == 'x' || rem.str[1] == 'X'))))
        return detail::scan_one(str, "f", v) != csubstr::npos;
    else
        return detail::scan_one(str, "a", v) != csubstr::npos;
#endif
}


/** Convert a string to a double precision real number.
 * The input string must be trimmed to the value, ie
 * no leading or trailing whitespace can be present.
 * @return true iff the conversion succeeded
 * @see atod_first() if the string is not trimmed
 */
C4_ALWAYS_INLINE bool atod(csubstr str, double * C4_RESTRICT v) noexcept
{
    C4_ASSERT(str.triml(" \r\t\n").len == str.len);
#if C4CORE_HAVE_FAST_FLOAT
    // fastfloat cannot parse hexadecimal floats
    bool isneg = (str.str[0] == '-');
    csubstr rem = str.sub(isneg || str.str[0] == '+');
    if(!(rem.len >= 2 && (rem.str[0] == '0' && (rem.str[1] == 'x' || rem.str[1] == 'X'))))
    {
        fast_float::from_chars_result result;
        result = fast_float::from_chars(str.str, str.str + str.len, *v);
        return result.ec == std::errc();
    }
    else if(detail::scan_rhex(rem.sub(2), v))
    {
        *v *= isneg ? -1. : 1.;
        return true;
    }
    return false;
#elif C4CORE_HAVE_STD_FROMCHARS
    std::from_chars_result result;
    result = std::from_chars(str.str, str.str + str.len, *v);
    return result.ec == std::errc();
#else
    csubstr rem = str.sub(str.str[0] == '-' || str.str[0] == '+');
    if(!(rem.len >= 2 && (rem.str[0] == '0' && (rem.str[1] == 'x' || rem.str[1] == 'X'))))
        return detail::scan_one(str, "lf", v) != csubstr::npos;
    else
        return detail::scan_one(str, "la", v) != csubstr::npos;
#endif
}


/** Convert a string to a single precision real number.
 * Leading whitespace is skipped until valid characters are found.
 * @return the number of characters read from the string, or npos if
 * conversion was not successful or if the string was empty */
inline size_t atof_first(csubstr str, float * C4_RESTRICT v) noexcept
{
    csubstr trimmed = str.first_real_span();
    if(trimmed.len == 0)
        return csubstr::npos;
    if(atof(trimmed, v))
        return static_cast<size_t>(trimmed.end() - str.begin());
    return csubstr::npos;
}


/** Convert a string to a double precision real number.
 * Leading whitespace is skipped until valid characters are found.
 * @return the number of characters read from the string, or npos if
 * conversion was not successful or if the string was empty */
inline size_t atod_first(csubstr str, double * C4_RESTRICT v) noexcept
{
    csubstr trimmed = str.first_real_span();
    if(trimmed.len == 0)
        return csubstr::npos;
    if(atod(trimmed, v))
        return static_cast<size_t>(trimmed.end() - str.begin());
    return csubstr::npos;
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// generic versions

C4_ALWAYS_INLINE size_t xtoa(substr s,  uint8_t v) noexcept { return write_dec(s, v); }
C4_ALWAYS_INLINE size_t xtoa(substr s, uint16_t v) noexcept { return write_dec(s, v); }
C4_ALWAYS_INLINE size_t xtoa(substr s, uint32_t v) noexcept { return write_dec(s, v); }
C4_ALWAYS_INLINE size_t xtoa(substr s, uint64_t v) noexcept { return write_dec(s, v); }
C4_ALWAYS_INLINE size_t xtoa(substr s,   int8_t v) noexcept { return itoa(s, v); }
C4_ALWAYS_INLINE size_t xtoa(substr s,  int16_t v) noexcept { return itoa(s, v); }
C4_ALWAYS_INLINE size_t xtoa(substr s,  int32_t v) noexcept { return itoa(s, v); }
C4_ALWAYS_INLINE size_t xtoa(substr s,  int64_t v) noexcept { return itoa(s, v); }
C4_ALWAYS_INLINE size_t xtoa(substr s,    float v) noexcept { return ftoa(s, v); }
C4_ALWAYS_INLINE size_t xtoa(substr s,   double v) noexcept { return dtoa(s, v); }

C4_ALWAYS_INLINE size_t xtoa(substr s,  uint8_t v,  uint8_t radix) noexcept { return utoa(s, v, radix); }
C4_ALWAYS_INLINE size_t xtoa(substr s, uint16_t v, uint16_t radix) noexcept { return utoa(s, v, radix); }
C4_ALWAYS_INLINE size_t xtoa(substr s, uint32_t v, uint32_t radix) noexcept { return utoa(s, v, radix); }
C4_ALWAYS_INLINE size_t xtoa(substr s, uint64_t v, uint64_t radix) noexcept { return utoa(s, v, radix); }
C4_ALWAYS_INLINE size_t xtoa(substr s,   int8_t v,   int8_t radix) noexcept { return itoa(s, v, radix); }
C4_ALWAYS_INLINE size_t xtoa(substr s,  int16_t v,  int16_t radix) noexcept { return itoa(s, v, radix); }
C4_ALWAYS_INLINE size_t xtoa(substr s,  int32_t v,  int32_t radix) noexcept { return itoa(s, v, radix); }
C4_ALWAYS_INLINE size_t xtoa(substr s,  int64_t v,  int64_t radix) noexcept { return itoa(s, v, radix); }

C4_ALWAYS_INLINE size_t xtoa(substr s,  uint8_t v,  uint8_t radix, size_t num_digits) noexcept { return utoa(s, v, radix, num_digits); }
C4_ALWAYS_INLINE size_t xtoa(substr s, uint16_t v, uint16_t radix, size_t num_digits) noexcept { return utoa(s, v, radix, num_digits); }
C4_ALWAYS_INLINE size_t xtoa(substr s, uint32_t v, uint32_t radix, size_t num_digits) noexcept { return utoa(s, v, radix, num_digits); }
C4_ALWAYS_INLINE size_t xtoa(substr s, uint64_t v, uint64_t radix, size_t num_digits) noexcept { return utoa(s, v, radix, num_digits); }
C4_ALWAYS_INLINE size_t xtoa(substr s,   int8_t v,   int8_t radix, size_t num_digits) noexcept { return itoa(s, v, radix, num_digits); }
C4_ALWAYS_INLINE size_t xtoa(substr s,  int16_t v,  int16_t radix, size_t num_digits) noexcept { return itoa(s, v, radix, num_digits); }
C4_ALWAYS_INLINE size_t xtoa(substr s,  int32_t v,  int32_t radix, size_t num_digits) noexcept { return itoa(s, v, radix, num_digits); }
C4_ALWAYS_INLINE size_t xtoa(substr s,  int64_t v,  int64_t radix, size_t num_digits) noexcept { return itoa(s, v, radix, num_digits); }

C4_ALWAYS_INLINE size_t xtoa(substr s,  float v, int precision, RealFormat_e formatting=FTOA_FLEX) noexcept { return ftoa(s, v, precision, formatting); }
C4_ALWAYS_INLINE size_t xtoa(substr s, double v, int precision, RealFormat_e formatting=FTOA_FLEX) noexcept { return dtoa(s, v, precision, formatting); }

C4_ALWAYS_INLINE bool atox(csubstr s,  uint8_t *C4_RESTRICT v) noexcept { return atou(s, v); }
C4_ALWAYS_INLINE bool atox(csubstr s, uint16_t *C4_RESTRICT v) noexcept { return atou(s, v); }
C4_ALWAYS_INLINE bool atox(csubstr s, uint32_t *C4_RESTRICT v) noexcept { return atou(s, v); }
C4_ALWAYS_INLINE bool atox(csubstr s, uint64_t *C4_RESTRICT v) noexcept { return atou(s, v); }
C4_ALWAYS_INLINE bool atox(csubstr s,   int8_t *C4_RESTRICT v) noexcept { return atoi(s, v); }
C4_ALWAYS_INLINE bool atox(csubstr s,  int16_t *C4_RESTRICT v) noexcept { return atoi(s, v); }
C4_ALWAYS_INLINE bool atox(csubstr s,  int32_t *C4_RESTRICT v) noexcept { return atoi(s, v); }
C4_ALWAYS_INLINE bool atox(csubstr s,  int64_t *C4_RESTRICT v) noexcept { return atoi(s, v); }
C4_ALWAYS_INLINE bool atox(csubstr s,    float *C4_RESTRICT v) noexcept { return atof(s, v); }
C4_ALWAYS_INLINE bool atox(csubstr s,   double *C4_RESTRICT v) noexcept { return atod(s, v); }

C4_ALWAYS_INLINE size_t to_chars(substr buf,  uint8_t v) noexcept { return write_dec(buf, v); }
C4_ALWAYS_INLINE size_t to_chars(substr buf, uint16_t v) noexcept { return write_dec(buf, v); }
C4_ALWAYS_INLINE size_t to_chars(substr buf, uint32_t v) noexcept { return write_dec(buf, v); }
C4_ALWAYS_INLINE size_t to_chars(substr buf, uint64_t v) noexcept { return write_dec(buf, v); }
C4_ALWAYS_INLINE size_t to_chars(substr buf,   int8_t v) noexcept { return itoa(buf, v); }
C4_ALWAYS_INLINE size_t to_chars(substr buf,  int16_t v) noexcept { return itoa(buf, v); }
C4_ALWAYS_INLINE size_t to_chars(substr buf,  int32_t v) noexcept { return itoa(buf, v); }
C4_ALWAYS_INLINE size_t to_chars(substr buf,  int64_t v) noexcept { return itoa(buf, v); }
C4_ALWAYS_INLINE size_t to_chars(substr buf,    float v) noexcept { return ftoa(buf, v); }
C4_ALWAYS_INLINE size_t to_chars(substr buf,   double v) noexcept { return dtoa(buf, v); }

C4_ALWAYS_INLINE bool from_chars(csubstr buf,  uint8_t *C4_RESTRICT v) noexcept { return atou(buf, v); }
C4_ALWAYS_INLINE bool from_chars(csubstr buf, uint16_t *C4_RESTRICT v) noexcept { return atou(buf, v); }
C4_ALWAYS_INLINE bool from_chars(csubstr buf, uint32_t *C4_RESTRICT v) noexcept { return atou(buf, v); }
C4_ALWAYS_INLINE bool from_chars(csubstr buf, uint64_t *C4_RESTRICT v) noexcept { return atou(buf, v); }
C4_ALWAYS_INLINE bool from_chars(csubstr buf,   int8_t *C4_RESTRICT v) noexcept { return atoi(buf, v); }
C4_ALWAYS_INLINE bool from_chars(csubstr buf,  int16_t *C4_RESTRICT v) noexcept { return atoi(buf, v); }
C4_ALWAYS_INLINE bool from_chars(csubstr buf,  int32_t *C4_RESTRICT v) noexcept { return atoi(buf, v); }
C4_ALWAYS_INLINE bool from_chars(csubstr buf,  int64_t *C4_RESTRICT v) noexcept { return atoi(buf, v); }
C4_ALWAYS_INLINE bool from_chars(csubstr buf,    float *C4_RESTRICT v) noexcept { return atof(buf, v); }
C4_ALWAYS_INLINE bool from_chars(csubstr buf,   double *C4_RESTRICT v) noexcept { return atod(buf, v); }

C4_ALWAYS_INLINE size_t from_chars_first(csubstr buf,  uint8_t *C4_RESTRICT v) noexcept { return atou_first(buf, v); }
C4_ALWAYS_INLINE size_t from_chars_first(csubstr buf, uint16_t *C4_RESTRICT v) noexcept { return atou_first(buf, v); }
C4_ALWAYS_INLINE size_t from_chars_first(csubstr buf, uint32_t *C4_RESTRICT v) noexcept { return atou_first(buf, v); }
C4_ALWAYS_INLINE size_t from_chars_first(csubstr buf, uint64_t *C4_RESTRICT v) noexcept { return atou_first(buf, v); }
C4_ALWAYS_INLINE size_t from_chars_first(csubstr buf,   int8_t *C4_RESTRICT v) noexcept { return atoi_first(buf, v); }
C4_ALWAYS_INLINE size_t from_chars_first(csubstr buf,  int16_t *C4_RESTRICT v) noexcept { return atoi_first(buf, v); }
C4_ALWAYS_INLINE size_t from_chars_first(csubstr buf,  int32_t *C4_RESTRICT v) noexcept { return atoi_first(buf, v); }
C4_ALWAYS_INLINE size_t from_chars_first(csubstr buf,  int64_t *C4_RESTRICT v) noexcept { return atoi_first(buf, v); }
C4_ALWAYS_INLINE size_t from_chars_first(csubstr buf,    float *C4_RESTRICT v) noexcept { return atof_first(buf, v); }
C4_ALWAYS_INLINE size_t from_chars_first(csubstr buf,   double *C4_RESTRICT v) noexcept { return atod_first(buf, v); }


//-----------------------------------------------------------------------------
// on some platforms, (unsigned) int and (unsigned) long
// are not any of the fixed length types above

#define _C4_IF_NOT_FIXED_LENGTH_I(T, ty) C4_ALWAYS_INLINE typename std::enable_if<std::  is_signed<T>::value && !is_fixed_length<T>::value_i, ty>
#define _C4_IF_NOT_FIXED_LENGTH_U(T, ty) C4_ALWAYS_INLINE typename std::enable_if<std::is_unsigned<T>::value && !is_fixed_length<T>::value_u, ty>

template <class T> _C4_IF_NOT_FIXED_LENGTH_I(T, size_t)::type xtoa(substr buf, T v) noexcept { return itoa(buf, v); }
template <class T> _C4_IF_NOT_FIXED_LENGTH_U(T, size_t)::type xtoa(substr buf, T v) noexcept { return write_dec(buf, v); }

template <class T> _C4_IF_NOT_FIXED_LENGTH_I(T, bool  )::type atox(csubstr buf, T *C4_RESTRICT v) noexcept { return atoi(buf, v); }
template <class T> _C4_IF_NOT_FIXED_LENGTH_U(T, bool  )::type atox(csubstr buf, T *C4_RESTRICT v) noexcept { return atou(buf, v); }

template <class T> _C4_IF_NOT_FIXED_LENGTH_I(T, size_t)::type to_chars(substr buf, T v) noexcept { return itoa(buf, v); }
template <class T> _C4_IF_NOT_FIXED_LENGTH_U(T, size_t)::type to_chars(substr buf, T v) noexcept { return write_dec(buf, v); }

template <class T> _C4_IF_NOT_FIXED_LENGTH_I(T, bool  )::type from_chars(csubstr buf, T *C4_RESTRICT v) noexcept { return atoi(buf, v); }
template <class T> _C4_IF_NOT_FIXED_LENGTH_U(T, bool  )::type from_chars(csubstr buf, T *C4_RESTRICT v) noexcept { return atou(buf, v); }

template <class T> _C4_IF_NOT_FIXED_LENGTH_I(T, size_t)::type from_chars_first(csubstr buf, T *C4_RESTRICT v) noexcept { return atoi_first(buf, v); }
template <class T> _C4_IF_NOT_FIXED_LENGTH_U(T, size_t)::type from_chars_first(csubstr buf, T *C4_RESTRICT v) noexcept { return atou_first(buf, v); }

#undef _C4_IF_NOT_FIXED_LENGTH_I
#undef _C4_IF_NOT_FIXED_LENGTH_U


//-----------------------------------------------------------------------------
// for pointers

template <class T> C4_ALWAYS_INLINE size_t xtoa(substr s, T *v) noexcept { return itoa(s, (intptr_t)v, (intptr_t)16); }
template <class T> C4_ALWAYS_INLINE bool   atox(csubstr s, T **v) noexcept { intptr_t tmp; bool ret = atox(s, &tmp); if(ret) { *v = (T*)tmp; } return ret; }
template <class T> C4_ALWAYS_INLINE size_t to_chars(substr s, T *v) noexcept { return itoa(s, (intptr_t)v, (intptr_t)16); }
template <class T> C4_ALWAYS_INLINE bool   from_chars(csubstr buf, T **v) noexcept { intptr_t tmp; bool ret = from_chars(buf, &tmp); if(ret) { *v = (T*)tmp; } return ret; }
template <class T> C4_ALWAYS_INLINE size_t from_chars_first(csubstr buf, T **v) noexcept { intptr_t tmp; bool ret = from_chars_first(buf, &tmp); if(ret) { *v = (T*)tmp; } return ret; }


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/** call to_chars() and return a substr consisting of the
 * written portion of the input buffer. Ie, same as to_chars(),
 * but return a substr instead of a size_t.
 *
 * @see to_chars() */
template<class T>
C4_ALWAYS_INLINE substr to_chars_sub(substr buf, T const& C4_RESTRICT v) noexcept
{
    size_t sz = to_chars(buf, v);
    return buf.left_of(sz <= buf.len ? sz : buf.len);
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// bool implementation

C4_ALWAYS_INLINE size_t to_chars(substr buf, bool v) noexcept
{
    int val = v;
    return to_chars(buf, val);
}

inline bool from_chars(csubstr buf, bool * C4_RESTRICT v) noexcept
{
    if(buf == '0')
    {
        *v = false; return true;
    }
    else if(buf == '1')
    {
        *v = true; return true;
    }
    else if(buf == "false")
    {
        *v = false; return true;
    }
    else if(buf == "true")
    {
        *v = true; return true;
    }
    else if(buf == "False")
    {
        *v = false; return true;
    }
    else if(buf == "True")
    {
        *v = true; return true;
    }
    else if(buf == "FALSE")
    {
        *v = false; return true;
    }
    else if(buf == "TRUE")
    {
        *v = true; return true;
    }
    // fallback to c-style int bools
    int val = 0;
    bool ret = from_chars(buf, &val);
    if(C4_LIKELY(ret))
    {
        *v = (val != 0);
    }
    return ret;
}

inline size_t from_chars_first(csubstr buf, bool * C4_RESTRICT v) noexcept
{
    csubstr trimmed = buf.first_non_empty_span();
    if(trimmed.len == 0 || !from_chars(buf, v))
        return csubstr::npos;
    return trimmed.len;
}


//-----------------------------------------------------------------------------
// single-char implementation

inline size_t to_chars(substr buf, char v) noexcept
{
    if(buf.len > 0)
    {
        C4_XASSERT(buf.str);
        buf.str[0] = v;
    }
    return 1;
}

/** extract a single character from a substring
 * @note to extract a string instead and not just a single character, use the csubstr overload */
inline bool from_chars(csubstr buf, char * C4_RESTRICT v) noexcept
{
    if(buf.len != 1)
        return false;
    C4_XASSERT(buf.str);
    *v = buf.str[0];
    return true;
}

inline size_t from_chars_first(csubstr buf, char * C4_RESTRICT v) noexcept
{
    if(buf.len < 1)
        return csubstr::npos;
    *v = buf.str[0];
    return 1;
}


//-----------------------------------------------------------------------------
// csubstr implementation

inline size_t to_chars(substr buf, csubstr v) noexcept
{
    C4_ASSERT(!buf.overlaps(v));
    size_t len = buf.len < v.len ? buf.len : v.len;
    // calling memcpy with null strings is undefined behavior
    // and will wreak havoc in calling code's branches.
    // see https://github.com/biojppm/rapidyaml/pull/264#issuecomment-1262133637
    if(len)
    {
        C4_ASSERT(buf.str != nullptr);
        C4_ASSERT(v.str != nullptr);
        memcpy(buf.str, v.str, len);
    }
    return v.len;
}

inline bool from_chars(csubstr buf, csubstr *C4_RESTRICT v) noexcept
{
    *v = buf;
    return true;
}

inline size_t from_chars_first(substr buf, csubstr * C4_RESTRICT v) noexcept
{
    csubstr trimmed = buf.first_non_empty_span();
    if(trimmed.len == 0)
        return csubstr::npos;
    *v = trimmed;
    return static_cast<size_t>(trimmed.end() - buf.begin());
}


//-----------------------------------------------------------------------------
// substr

inline size_t to_chars(substr buf, substr v) noexcept
{
    C4_ASSERT(!buf.overlaps(v));
    size_t len = buf.len < v.len ? buf.len : v.len;
    // calling memcpy with null strings is undefined behavior
    // and will wreak havoc in calling code's branches.
    // see https://github.com/biojppm/rapidyaml/pull/264#issuecomment-1262133637
    if(len)
    {
        C4_ASSERT(buf.str != nullptr);
        C4_ASSERT(v.str != nullptr);
        memcpy(buf.str, v.str, len);
    }
    return v.len;
}

inline bool from_chars(csubstr buf, substr * C4_RESTRICT v) noexcept
{
    C4_ASSERT(!buf.overlaps(*v));
    // is the destination buffer wide enough?
    if(v->len >= buf.len)
    {
        // calling memcpy with null strings is undefined behavior
        // and will wreak havoc in calling code's branches.
        // see https://github.com/biojppm/rapidyaml/pull/264#issuecomment-1262133637
        if(buf.len)
        {
            C4_ASSERT(buf.str != nullptr);
            C4_ASSERT(v->str != nullptr);
            memcpy(v->str, buf.str, buf.len);
        }
        v->len = buf.len;
        return true;
    }
    return false;
}

inline size_t from_chars_first(csubstr buf, substr * C4_RESTRICT v) noexcept
{
    csubstr trimmed = buf.first_non_empty_span();
    C4_ASSERT(!trimmed.overlaps(*v));
    if(C4_UNLIKELY(trimmed.len == 0))
        return csubstr::npos;
    size_t len = trimmed.len > v->len ? v->len : trimmed.len;
    // calling memcpy with null strings is undefined behavior
    // and will wreak havoc in calling code's branches.
    // see https://github.com/biojppm/rapidyaml/pull/264#issuecomment-1262133637
    if(len)
    {
        C4_ASSERT(buf.str != nullptr);
        C4_ASSERT(v->str != nullptr);
        memcpy(v->str, trimmed.str, len);
    }
    if(C4_UNLIKELY(trimmed.len > v->len))
        return csubstr::npos;
    return static_cast<size_t>(trimmed.end() - buf.begin());
}


//-----------------------------------------------------------------------------

template<size_t N>
inline size_t to_chars(substr buf, const char (& C4_RESTRICT v)[N]) noexcept
{
    csubstr sp(v);
    return to_chars(buf, sp);
}

inline size_t to_chars(substr buf, const char * C4_RESTRICT v) noexcept
{
    return to_chars(buf, to_csubstr(v));
}

} // namespace c4

#ifdef _MSC_VER
#   pragma warning(pop)
#endif

#if defined(__clang__)
#   pragma clang diagnostic pop
#elif defined(__GNUC__)
#   pragma GCC diagnostic pop
#endif

#endif /* _C4_CHARCONV_HPP_ */
