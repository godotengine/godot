#ifndef _C4_PREPROCESSOR_HPP_
#define _C4_PREPROCESSOR_HPP_

/** @file preprocessor.hpp Contains basic macros and preprocessor utilities.
 * @ingroup basic_headers */

#ifdef __clang__
    /* NOTE: using , ## __VA_ARGS__ to deal with zero-args calls to
     * variadic macros is not portable, but works in clang, gcc, msvc, icc.
     * clang requires switching off compiler warnings for pedantic mode.
     * @see http://stackoverflow.com/questions/32047685/variadic-macro-without-arguments */
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments" // warning: token pasting of ',' and __VA_ARGS__ is a GNU extension
#elif defined(__GNUC__)
    /* GCC also issues a warning for zero-args calls to variadic macros.
     * This warning is switched on with -pedantic and apparently there is no
     * easy way to turn it off as with clang. But marking this as a system
     * header works.
     * @see https://gcc.gnu.org/onlinedocs/cpp/System-Headers.html
     * @see http://stackoverflow.com/questions/35587137/ */
#   pragma GCC system_header
#endif

#define C4_WIDEN(str) L"" str

#define C4_COUNTOF(arr) (sizeof(arr)/sizeof((arr)[0]))

#define C4_EXPAND(arg) arg

/** useful in some macro calls with template arguments */
#define C4_COMMA ,
/** useful in some macro calls with template arguments
 * @see C4_COMMA */
#define C4_COMMA_X C4_COMMA

/** expand and quote */
#define C4_XQUOTE(arg) _C4_XQUOTE(arg)
#define _C4_XQUOTE(arg) C4_QUOTE(arg)
#define C4_QUOTE(arg) #arg

/** expand and concatenate */
#define C4_XCAT(arg1, arg2) _C4_XCAT(arg1, arg2)
#define _C4_XCAT(arg1, arg2) C4_CAT(arg1, arg2)
#define C4_CAT(arg1, arg2) arg1##arg2

#define C4_VERSION_CAT(major, minor, patch) ((major)*10000 + (minor)*100 + (patch))

/** A preprocessor foreach. Spectacular trick taken from:
 * http://stackoverflow.com/a/1872506/5875572
 * The first argument is for a macro receiving a single argument,
 * which will be called with every subsequent argument. There is
 * currently a limit of 32 arguments, and at least 1 must be provided.
 *
Example:
@code{.cpp}
struct Example {
    int a;
    int b;
    int c;
};
// define a one-arg macro to be called
#define PRN_STRUCT_OFFSETS(field) PRN_STRUCT_OFFSETS_(Example, field)
#define PRN_STRUCT_OFFSETS_(structure, field) printf(C4_XQUOTE(structure) ":" C4_XQUOTE(field)" - offset=%zu\n", offsetof(structure, field));

// now call the macro for a, b and c
C4_FOR_EACH(PRN_STRUCT_OFFSETS, a, b, c);
@endcode */
#define C4_FOR_EACH(what, ...) C4_FOR_EACH_SEP(what, ;, __VA_ARGS__)

/** same as C4_FOR_EACH(), but use a custom separator between statements.
 * If a comma is needed as the separator, use the C4_COMMA macro.
 * @see C4_FOR_EACH
 * @see C4_COMMA
 */
#define C4_FOR_EACH_SEP(what, sep, ...) _C4_FOR_EACH_(_C4_FOR_EACH_NARG(__VA_ARGS__), what, sep, __VA_ARGS__)

/// @cond dev

#define _C4_FOR_EACH_01(what, sep, x) what(x) sep
#define _C4_FOR_EACH_02(what, sep, x, ...) what(x) sep _C4_FOR_EACH_01(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_03(what, sep, x, ...) what(x) sep _C4_FOR_EACH_02(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_04(what, sep, x, ...) what(x) sep _C4_FOR_EACH_03(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_05(what, sep, x, ...) what(x) sep _C4_FOR_EACH_04(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_06(what, sep, x, ...) what(x) sep _C4_FOR_EACH_05(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_07(what, sep, x, ...) what(x) sep _C4_FOR_EACH_06(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_08(what, sep, x, ...) what(x) sep _C4_FOR_EACH_07(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_09(what, sep, x, ...) what(x) sep _C4_FOR_EACH_08(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_10(what, sep, x, ...) what(x) sep _C4_FOR_EACH_09(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_11(what, sep, x, ...) what(x) sep _C4_FOR_EACH_10(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_12(what, sep, x, ...) what(x) sep _C4_FOR_EACH_11(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_13(what, sep, x, ...) what(x) sep _C4_FOR_EACH_12(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_14(what, sep, x, ...) what(x) sep _C4_FOR_EACH_13(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_15(what, sep, x, ...) what(x) sep _C4_FOR_EACH_14(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_16(what, sep, x, ...) what(x) sep _C4_FOR_EACH_15(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_17(what, sep, x, ...) what(x) sep _C4_FOR_EACH_16(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_18(what, sep, x, ...) what(x) sep _C4_FOR_EACH_17(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_19(what, sep, x, ...) what(x) sep _C4_FOR_EACH_18(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_20(what, sep, x, ...) what(x) sep _C4_FOR_EACH_19(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_21(what, sep, x, ...) what(x) sep _C4_FOR_EACH_20(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_22(what, sep, x, ...) what(x) sep _C4_FOR_EACH_21(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_23(what, sep, x, ...) what(x) sep _C4_FOR_EACH_22(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_24(what, sep, x, ...) what(x) sep _C4_FOR_EACH_23(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_25(what, sep, x, ...) what(x) sep _C4_FOR_EACH_24(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_26(what, sep, x, ...) what(x) sep _C4_FOR_EACH_25(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_27(what, sep, x, ...) what(x) sep _C4_FOR_EACH_26(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_28(what, sep, x, ...) what(x) sep _C4_FOR_EACH_27(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_29(what, sep, x, ...) what(x) sep _C4_FOR_EACH_28(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_30(what, sep, x, ...) what(x) sep _C4_FOR_EACH_29(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_31(what, sep, x, ...) what(x) sep _C4_FOR_EACH_30(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_32(what, sep, x, ...) what(x) sep _C4_FOR_EACH_31(what, sep, __VA_ARGS__)
#define _C4_FOR_EACH_NARG(...) _C4_FOR_EACH_NARG_(__VA_ARGS__, _C4_FOR_EACH_RSEQ_N())
#define _C4_FOR_EACH_NARG_(...) _C4_FOR_EACH_ARG_N(__VA_ARGS__)
#define _C4_FOR_EACH_ARG_N(_01, _02, _03, _04, _05, _06, _07, _08, _09, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, N, ...) N
#define _C4_FOR_EACH_RSEQ_N() 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 09, 08, 07, 06, 05, 04, 03, 02, 01
#define _C4_FOR_EACH_(N, what, sep, ...) C4_XCAT(_C4_FOR_EACH_, N)(what, sep, __VA_ARGS__)

/// @endcond

#ifdef __clang__
#   pragma clang diagnostic pop
#endif

#endif /* _C4_PREPROCESSOR_HPP_ */
