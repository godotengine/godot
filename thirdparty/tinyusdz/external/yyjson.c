/*==============================================================================
 Copyright (c) 2020 YaoYuan <ibireme@gmail.com>

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 *============================================================================*/

#include "yyjson.h"
#include <math.h> /* for `HUGE_VAL/INFINIY/NAN` macros, no libm required */



/*==============================================================================
 * Warning Suppress
 *============================================================================*/

#if defined(__clang__)
#   pragma clang diagnostic ignored "-Wunused-function"
#   pragma clang diagnostic ignored "-Wunused-parameter"
#   pragma clang diagnostic ignored "-Wunused-label"
#   pragma clang diagnostic ignored "-Wunused-macros"
#   pragma clang diagnostic ignored "-Wunused-variable"
#elif defined(__GNUC__)
#   pragma GCC diagnostic ignored "-Wunused-function"
#   pragma GCC diagnostic ignored "-Wunused-parameter"
#   pragma GCC diagnostic ignored "-Wunused-label"
#   pragma GCC diagnostic ignored "-Wunused-macros"
#   pragma GCC diagnostic ignored "-Wunused-variable"
#elif defined(_MSC_VER)
#   pragma warning(disable:4100) /* unreferenced formal parameter */
#   pragma warning(disable:4101) /* unreferenced variable */
#   pragma warning(disable:4102) /* unreferenced label */
#   pragma warning(disable:4127) /* conditional expression is constant */
#   pragma warning(disable:4706) /* assignment within conditional expression */
#endif



/*==============================================================================
 * Version
 *============================================================================*/

uint32_t yyjson_version(void) {
    return YYJSON_VERSION_HEX;
}



/*==============================================================================
 * Flags
 *============================================================================*/

/* msvc intrinsic */
#if YYJSON_MSC_VER >= 1400
#   include <intrin.h>
#   if defined(_M_AMD64) || defined(_M_ARM64)
#       define MSC_HAS_BIT_SCAN_64 1
#       pragma intrinsic(_BitScanForward64)
#       pragma intrinsic(_BitScanReverse64)
#   else
#       define MSC_HAS_BIT_SCAN_64 0
#   endif
#   if defined(_M_AMD64) || defined(_M_ARM64) || \
        defined(_M_IX86) || defined(_M_ARM)
#       define MSC_HAS_BIT_SCAN 1
#       pragma intrinsic(_BitScanForward)
#       pragma intrinsic(_BitScanReverse)
#   else
#       define MSC_HAS_BIT_SCAN 0
#   endif
#   if defined(_M_AMD64)
#       define MSC_HAS_UMUL128 1
#       pragma intrinsic(_umul128)
#   else
#       define MSC_HAS_UMUL128 0
#   endif
#else
#   define MSC_HAS_BIT_SCAN_64 0
#   define MSC_HAS_BIT_SCAN 0
#   define MSC_HAS_UMUL128 0
#endif

/* gcc builtin */
#if yyjson_has_builtin(__builtin_clzll) || yyjson_gcc_available(3, 4, 0)
#   define GCC_HAS_CLZLL 1
#else
#   define GCC_HAS_CLZLL 0
#endif

#if yyjson_has_builtin(__builtin_ctzll) || yyjson_gcc_available(3, 4, 0)
#   define GCC_HAS_CTZLL 1
#else
#   define GCC_HAS_CTZLL 0
#endif

/* int128 type */
#if defined(__SIZEOF_INT128__) && (__SIZEOF_INT128__ == 16) && \
    (defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER))
#    define YYJSON_HAS_INT128 1
#else
#    define YYJSON_HAS_INT128 0
#endif

/* IEEE 754 floating-point binary representation */
#if defined(__STDC_IEC_559__) || defined(__STDC_IEC_60559_BFP__)
#   define YYJSON_HAS_IEEE_754 1
#elif FLT_RADIX == 2 && \
    FLT_MANT_DIG == 24 && FLT_DIG == 6 && \
    FLT_MIN_EXP == -125 && FLT_MAX_EXP == 128 && \
    FLT_MIN_10_EXP == -37 && FLT_MAX_10_EXP == 38 && \
    DBL_MANT_DIG == 53 && DBL_DIG == 15 && \
    DBL_MIN_EXP == -1021 && DBL_MAX_EXP == 1024 && \
    DBL_MIN_10_EXP == -307 && DBL_MAX_10_EXP == 308
#   define YYJSON_HAS_IEEE_754 1
#else
#   define YYJSON_HAS_IEEE_754 0
#endif

/*
 Correct rounding in double number computations.

 On the x86 architecture, some compilers may use x87 FPU instructions for
 floating-point arithmetic. The x87 FPU loads all floating point number as
 80-bit double-extended precision internally, then rounds the result to original
 precision, which may produce inaccurate results. For a more detailed
 explanation, see the paper: https://arxiv.org/abs/cs/0701192

 Here are some examples of double precision calculation error:

     2877.0 / 1e6   == 0.002877,  but x87 returns 0.0028770000000000002
     43683.0 * 1e21 == 4.3683e25, but x87 returns 4.3683000000000004e25

 Here are some examples of compiler flags to generate x87 instructions on x86:

     clang -m32 -mno-sse
     gcc/icc -m32 -mfpmath=387
     msvc /arch:SSE or /arch:IA32

 If we are sure that there's no similar error described above, we can define the
 YYJSON_DOUBLE_MATH_CORRECT as 1 to enable the fast path calculation. This is
 not an accurate detection, it's just try to avoid the error at compile-time.
 An accurate detection can be done at run-time:

     bool is_double_math_correct(void) {
         volatile double r = 43683.0;
         r *= 1e21;
         return r == 4.3683e25;
     }

 See also: utils.h in https://github.com/google/double-conversion/
 */
#if !defined(FLT_EVAL_METHOD) && defined(__FLT_EVAL_METHOD__)
#    define FLT_EVAL_METHOD __FLT_EVAL_METHOD__
#endif

#if defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0 && FLT_EVAL_METHOD != 1
#    define YYJSON_DOUBLE_MATH_CORRECT 0
#elif defined(i386) || defined(__i386) || defined(__i386__) || \
    defined(_X86_) || defined(__X86__) || defined(_M_IX86) || \
    defined(__I86__) || defined(__IA32__) || defined(__THW_INTEL)
#   if (defined(_MSC_VER) && defined(_M_IX86_FP) && _M_IX86_FP == 2) || \
        (defined(__SSE2_MATH__) && __SSE2_MATH__)
#       define YYJSON_DOUBLE_MATH_CORRECT 1
#   else
#       define YYJSON_DOUBLE_MATH_CORRECT 0
#   endif
#elif defined(__mc68000__) || defined(__pnacl__) || defined(__native_client__)
#   define YYJSON_DOUBLE_MATH_CORRECT 0
#else
#   define YYJSON_DOUBLE_MATH_CORRECT 1
#endif

/* endian */
#if yyjson_has_include(<sys/types.h>)
#    include <sys/types.h> /* POSIX */
#endif
#if yyjson_has_include(<endian.h>)
#    include <endian.h> /* Linux */
#elif yyjson_has_include(<sys/endian.h>)
#    include <sys/endian.h> /* BSD, Android */
#elif yyjson_has_include(<machine/endian.h>)
#    include <machine/endian.h> /* BSD, Darwin */
#endif

#define YYJSON_BIG_ENDIAN       4321
#define YYJSON_LITTLE_ENDIAN    1234

#if defined(BYTE_ORDER) && BYTE_ORDER
#   if defined(BIG_ENDIAN) && (BYTE_ORDER == BIG_ENDIAN)
#       define YYJSON_ENDIAN YYJSON_BIG_ENDIAN
#   elif defined(LITTLE_ENDIAN) && (BYTE_ORDER == LITTLE_ENDIAN)
#       define YYJSON_ENDIAN YYJSON_LITTLE_ENDIAN
#   endif

#elif defined(__BYTE_ORDER) && __BYTE_ORDER
#   if defined(__BIG_ENDIAN) && (__BYTE_ORDER == __BIG_ENDIAN)
#       define YYJSON_ENDIAN YYJSON_BIG_ENDIAN
#   elif defined(__LITTLE_ENDIAN) && (__BYTE_ORDER == __LITTLE_ENDIAN)
#       define YYJSON_ENDIAN YYJSON_LITTLE_ENDIAN
#   endif

#elif defined(__BYTE_ORDER__) && __BYTE_ORDER__
#   if defined(__ORDER_BIG_ENDIAN__) && \
        (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#       define YYJSON_ENDIAN YYJSON_BIG_ENDIAN
#   elif defined(__ORDER_LITTLE_ENDIAN__) && \
        (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#       define YYJSON_ENDIAN YYJSON_LITTLE_ENDIAN
#   endif

#elif (defined(__LITTLE_ENDIAN__) && __LITTLE_ENDIAN__ == 1) || \
    defined(__i386) || defined(__i386__) || \
    defined(_X86_) || defined(__X86__) || \
    defined(_M_IX86) || defined(__THW_INTEL__) || \
    defined(__x86_64) || defined(__x86_64__) || \
    defined(__amd64) || defined(__amd64__) || \
    defined(_M_AMD64) || defined(_M_X64) || \
    defined(_M_ARM) || defined(_M_ARM64) || \
    defined(__ARMEL__) || defined(__THUMBEL__) || defined(__AARCH64EL__) || \
    defined(_MIPSEL) || defined(__MIPSEL) || defined(__MIPSEL__) || \
    defined(__EMSCRIPTEN__) || defined(__wasm__) || \
    defined(__loongarch__)
#   define YYJSON_ENDIAN YYJSON_LITTLE_ENDIAN

#elif (defined(__BIG_ENDIAN__) && __BIG_ENDIAN__ == 1) || \
    defined(__ARMEB__) || defined(__THUMBEB__) || defined(__AARCH64EB__) || \
    defined(_MIPSEB) || defined(__MIPSEB) || defined(__MIPSEB__) || \
    defined(__or1k__) || defined(__OR1K__)
#   define YYJSON_ENDIAN YYJSON_BIG_ENDIAN

#else
#   define YYJSON_ENDIAN 0 /* unknown endian, detect at run-time */
#endif

/*
 This macro controls how yyjson handles unaligned memory accesses.

 By default, yyjson uses `memcpy()` for memory copying. This allows the compiler
 to optimize the code and emit unaligned memory access instructions when
 supported by the target architecture.

 However, on some older compilers or architectures where `memcpy()` is not
 well-optimized and may result in unnecessary function calls, defining this
 macro as 1 may help. In such cases, yyjson switches to manual byte-by-byte
 access, which can potentially improve performance.

 An example of the generated assembly code for ARM can be found here:
 https://godbolt.org/z/334jjhxPT

 This flag is already enabled for common architectures in the following code,
 so manual configuration is usually unnecessary. If unsure, you can check the
 generated assembly or run benchmarks to make an informed decision.
 */
#ifndef YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
#   if defined(__ia64) || defined(_IA64) || defined(__IA64__) ||  \
        defined(__ia64__) || defined(_M_IA64) || defined(__itanium__)
#       define YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS 1 /* Itanium */
#   elif (defined(__arm__) || defined(__arm64__) || defined(__aarch64__)) && \
        (defined(__GNUC__) || defined(__clang__)) && \
        (!defined(__ARM_FEATURE_UNALIGNED) || !__ARM_FEATURE_UNALIGNED)
#       define YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS 1 /* ARM */
#   elif defined(__sparc) || defined(__sparc__)
#       define YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS 1 /* SPARC */
#   elif defined(__mips) || defined(__mips__) || defined(__MIPS__)
#       define YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS 1 /* MIPS */
#   elif defined(__m68k__) || defined(M68000)
#       define YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS 1 /* M68K */
#   else
#       define YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS 0
#   endif
#endif

/*
 Estimated initial ratio of the JSON data (data_size / value_count).
 For example:

    data:        {"id":12345678,"name":"Harry"}
    data_size:   30
    value_count: 5
    ratio:       6

 yyjson uses dynamic memory with a growth factor of 1.5 when reading and writing
 JSON, the ratios below are used to determine the initial memory size.

 A too large ratio will waste memory, and a too small ratio will cause multiple
 memory growths and degrade performance. Currently, these ratios are generated
 with some commonly used JSON datasets.
 */
#define YYJSON_READER_ESTIMATED_PRETTY_RATIO 16
#define YYJSON_READER_ESTIMATED_MINIFY_RATIO 6
#define YYJSON_WRITER_ESTIMATED_PRETTY_RATIO 32
#define YYJSON_WRITER_ESTIMATED_MINIFY_RATIO 18

/* The initial and maximum size of the memory pool's chunk in yyjson_mut_doc. */
#define YYJSON_MUT_DOC_STR_POOL_INIT_SIZE   0x100
#define YYJSON_MUT_DOC_STR_POOL_MAX_SIZE    0x10000000
#define YYJSON_MUT_DOC_VAL_POOL_INIT_SIZE   (0x10 * sizeof(yyjson_mut_val))
#define YYJSON_MUT_DOC_VAL_POOL_MAX_SIZE    (0x1000000 * sizeof(yyjson_mut_val))

/* The minimum size of the dynamic allocator's chunk. */
#define YYJSON_ALC_DYN_MIN_SIZE             0x1000

/* Default value for compile-time options. */
#ifndef YYJSON_DISABLE_READER
#define YYJSON_DISABLE_READER 0
#endif
#ifndef YYJSON_DISABLE_WRITER
#define YYJSON_DISABLE_WRITER 0
#endif
#ifndef YYJSON_DISABLE_INCR_READER
#define YYJSON_DISABLE_INCR_READER 0
#endif
#ifndef YYJSON_DISABLE_UTILS
#define YYJSON_DISABLE_UTILS 0
#endif
#ifndef YYJSON_DISABLE_FAST_FP_CONV
#define YYJSON_DISABLE_FAST_FP_CONV 0
#endif
#ifndef YYJSON_DISABLE_NON_STANDARD
#define YYJSON_DISABLE_NON_STANDARD 0
#endif
#ifndef YYJSON_DISABLE_UTF8_VALIDATION
#define YYJSON_DISABLE_UTF8_VALIDATION 0
#endif



/*==============================================================================
 * Macros
 *============================================================================*/

/* Macros used for loop unrolling and other purpose. */
#define repeat2(x)  { x x }
#define repeat3(x)  { x x x }
#define repeat4(x)  { x x x x }
#define repeat8(x)  { x x x x x x x x }
#define repeat16(x) { x x x x x x x x x x x x x x x x }

#define repeat2_incr(x)   { x(0)  x(1) }
#define repeat4_incr(x)   { x(0)  x(1)  x(2)  x(3) }
#define repeat8_incr(x)   { x(0)  x(1)  x(2)  x(3)  x(4)  x(5)  x(6)  x(7)  }
#define repeat16_incr(x)  { x(0)  x(1)  x(2)  x(3)  x(4)  x(5)  x(6)  x(7)  \
                            x(8)  x(9)  x(10) x(11) x(12) x(13) x(14) x(15) }
#define repeat_in_1_18(x) { x(1)  x(2)  x(3)  x(4)  x(5)  x(6)  x(7)  x(8)  \
                            x(9)  x(10) x(11) x(12) x(13) x(14) x(15) x(16) \
                            x(17) x(18) }

/* Macros used to provide branch prediction information for compiler. */
#undef  likely
#define likely(x)       yyjson_likely(x)
#undef  unlikely
#define unlikely(x)     yyjson_unlikely(x)

/* Macros used to provide inline information for compiler. */
#undef  static_inline
#define static_inline   static yyjson_inline
#undef  static_noinline
#define static_noinline static yyjson_noinline

/* Macros for min and max. */
#undef  yyjson_min
#define yyjson_min(x, y) ((x) < (y) ? (x) : (y))
#undef  yyjson_max
#define yyjson_max(x, y) ((x) > (y) ? (x) : (y))

/* Used to write u64 literal for C89 which doesn't support "ULL" suffix. */
#undef  U64
#define U64(hi, lo) ((((u64)hi##UL) << 32U) + lo##UL)
#undef  U32
#define U32(hi) ((u32)(hi##UL))

/* Used to cast away (remove) const qualifier. */
#define constcast(type) (type)(void *)(size_t)(const void *)

/* Common error messages. */
#define MSG_FOPEN       "failed to open file"
#define MSG_FREAD       "failed to read file"
#define MSG_FWRITE      "failed to write file"
#define MSG_FCLOSE      "failed to close file"
#define MSG_MALLOC      "failed to allocate memory"
#define MSG_CHAT_T      "invalid literal, expected 'true'"
#define MSG_CHAR_F      "invalid literal, expected 'false'"
#define MSG_CHAR_N      "invalid literal, expected 'null'"
#define MSG_CHAR        "unexpected character, expected a JSON value"
#define MSG_ARR_END     "unexpected character, expected ',' or ']'"
#define MSG_OBJ_KEY     "unexpected character, expected a string key"
#define MSG_OBJ_SEP     "unexpected character, expected ':' after key"
#define MSG_OBJ_END     "unexpected character, expected ',' or '}'"
#define MSG_GARBAGE     "unexpected content after document"
#define MSG_NOT_END     "unexpected end of data"
#define MSG_COMMENT     "unclosed multiline comment"
#define MSG_COMMA       "trailing comma is not allowed"
#define MSG_INF_NAN     "nan or inf number is not allowed"
#define MSG_ERR_TYPE    "invalid JSON value type"
#define MSG_ERR_UTF8    "invalid utf-8 encoding in string"
#define MSG_ERR_BOM     "UTF-8 byte order mark (BOM) is not supported"
#define MSG_ERR_UTF16   "UTF-16 encoding is not supported"
#define MSG_ERR_UTF32   "UTF-32 encoding is not supported"

/*
 Check flags using a function to avoid `always false` warnings.
 When non-standard features are disabled, unnecessary checks
 will be evaluated and optimized out at compile-time.
 */
static_inline bool read_flag_eq(yyjson_read_flag flg, yyjson_read_flag chk) {
#if YYJSON_DISABLE_NON_STANDARD
    if (chk == YYJSON_READ_ALLOW_INF_AND_NAN ||
        chk == YYJSON_READ_ALLOW_COMMENTS ||
        chk == YYJSON_READ_ALLOW_TRAILING_COMMAS ||
        chk == YYJSON_READ_ALLOW_INVALID_UNICODE ||
        chk == YYJSON_READ_ALLOW_BOM)
        return false;
#endif
    return (flg & chk) != 0;
}
static_inline bool write_flag_eq(yyjson_write_flag flg, yyjson_write_flag chk) {
#if YYJSON_DISABLE_NON_STANDARD
    if (chk == YYJSON_WRITE_ALLOW_INF_AND_NAN ||
        chk == YYJSON_WRITE_ALLOW_INVALID_UNICODE)
        return false;
#endif
    return (flg & chk) != 0;
}
#define has_read_flag(_flag) unlikely(read_flag_eq(flg, YYJSON_READ_##_flag))
#define has_write_flag(_flag) unlikely(write_flag_eq(flg, YYJSON_WRITE_##_flag))



/*==============================================================================
 * Number Constants
 *============================================================================*/

/* U64 constant values */
#undef  U64_MAX
#define U64_MAX         U64(0xFFFFFFFF, 0xFFFFFFFF)
#undef  I64_MAX
#define I64_MAX         U64(0x7FFFFFFF, 0xFFFFFFFF)
#undef  USIZE_MAX
#define USIZE_MAX       ((usize)(~(usize)0))

/* Maximum number of digits for reading u32/u64/usize safety (not overflow). */
#undef  U32_SAFE_DIG
#define U32_SAFE_DIG    9   /* u32 max is 4294967295, 10 digits */
#undef  U64_SAFE_DIG
#define U64_SAFE_DIG    19  /* u64 max is 18446744073709551615, 20 digits */
#undef  USIZE_SAFE_DIG
#define USIZE_SAFE_DIG  (sizeof(usize) == 8 ? U64_SAFE_DIG : U32_SAFE_DIG)

/* Inf raw value (positive) */
#define F64_RAW_INF U64(0x7FF00000, 0x00000000)

/* NaN raw value (quiet NaN, no payload, no sign) */
#if defined(__hppa__) || (defined(__mips__) && !defined(__mips_nan2008))
#define F64_RAW_NAN U64(0x7FF7FFFF, 0xFFFFFFFF)
#else
#define F64_RAW_NAN U64(0x7FF80000, 0x00000000)
#endif

/* maximum significant digits count in decimal when reading double number */
#define F64_MAX_DEC_DIG 768

/* maximum decimal power of double number (1.7976931348623157e308) */
#define F64_MAX_DEC_EXP 308

/* minimum decimal power of double number (4.9406564584124654e-324) */
#define F64_MIN_DEC_EXP (-324)

/* maximum binary power of double number */
#define F64_MAX_BIN_EXP 1024

/* minimum binary power of double number */
#define F64_MIN_BIN_EXP (-1021)

/* float/double number bits */
#define F32_BITS 32
#define F64_BITS 64

/* float/double number exponent part bits */
#define F32_EXP_BITS 8
#define F64_EXP_BITS 11

/* float/double number significand part bits */
#define F32_SIG_BITS 23
#define F64_SIG_BITS 52

/* float/double number significand part bits (with 1 hidden bit) */
#define F32_SIG_FULL_BITS 24
#define F64_SIG_FULL_BITS 53

/* float/double number significand bit mask */
#define F32_SIG_MASK U32(0x007FFFFF)
#define F64_SIG_MASK U64(0x000FFFFF, 0xFFFFFFFF)

/* float/double number exponent bit mask */
#define F32_EXP_MASK U32(0x7F800000)
#define F64_EXP_MASK U64(0x7FF00000, 0x00000000)

/* float/double number exponent bias */
#define F32_EXP_BIAS 127
#define F64_EXP_BIAS 1023

/* float/double number significant digits count in decimal */
#define F32_DEC_DIG 9
#define F64_DEC_DIG 17

/* buffer length required for float/double number writer */
#define FP_BUF_LEN 40

/* maximum length of a number in incremental parsing */
#define INCR_NUM_MAX_LEN 1024



/*==============================================================================
 * Types
 *============================================================================*/

/** Type define for primitive types. */
typedef float       f32;
typedef double      f64;
typedef int8_t      i8;
typedef uint8_t     u8;
typedef int16_t     i16;
typedef uint16_t    u16;
typedef int32_t     i32;
typedef uint32_t    u32;
typedef int64_t     i64;
typedef uint64_t    u64;
typedef size_t      usize;

/** 128-bit integer, used by floating-point number reader and writer. */
#if YYJSON_HAS_INT128
__extension__ typedef __int128          i128;
__extension__ typedef unsigned __int128 u128;
#endif

/** 16/32/64-bit vector */
typedef struct v16 { char c[2]; } v16;
typedef struct v32 { char c[4]; } v32;
typedef struct v64 { char c[8]; } v64;

/** 16/32/64-bit vector union */
typedef union v16_uni { v16 v; u16 u; } v16_uni;
typedef union v32_uni { v32 v; u32 u; } v32_uni;
typedef union v64_uni { v64 v; u64 u; } v64_uni;



/*==============================================================================
 * Load/Store Utils
 *============================================================================*/

#define byte_move_idx(x) ((char *)dst)[x] = ((const char *)src)[x];
#define byte_move_src(x) ((char *)tmp)[x] = ((const char *)src)[x];
#define byte_move_dst(x) ((char *)dst)[x] = ((const char *)tmp)[x];

static_inline void byte_copy_2(void *dst, const void *src) {
#if !YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
    memcpy(dst, src, 2);
#else
    repeat2_incr(byte_move_idx)
#endif
}

static_inline void byte_copy_4(void *dst, const void *src) {
#if !YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
    memcpy(dst, src, 4);
#else
    repeat4_incr(byte_move_idx)
#endif
}

static_inline void byte_copy_8(void *dst, const void *src) {
#if !YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
    memcpy(dst, src, 8);
#else
    repeat8_incr(byte_move_idx)
#endif
}

static_inline void byte_copy_16(void *dst, const void *src) {
#if !YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
    memcpy(dst, src, 16);
#else
    repeat16_incr(byte_move_idx)
#endif
}

static_inline void byte_move_2(void *dst, const void *src) {
#if !YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
    u16 tmp;
    memcpy(&tmp, src, 2);
    memcpy(dst, &tmp, 2);
#else
    char tmp[2];
    repeat2_incr(byte_move_src)
    repeat2_incr(byte_move_dst)
#endif
}

static_inline void byte_move_4(void *dst, const void *src) {
#if !YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
    u32 tmp;
    memcpy(&tmp, src, 4);
    memcpy(dst, &tmp, 4);
#else
    char tmp[4];
    repeat4_incr(byte_move_src)
    repeat4_incr(byte_move_dst)
#endif
}

static_inline void byte_move_8(void *dst, const void *src) {
#if !YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
    u64 tmp;
    memcpy(&tmp, src, 8);
    memcpy(dst, &tmp, 8);
#else
    char tmp[8];
    repeat8_incr(byte_move_src)
    repeat8_incr(byte_move_dst)
#endif
}

static_inline void byte_move_16(void *dst, const void *src) {
#if !YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
    char *pdst = (char *)dst;
    const char *psrc = (const char *)src;
    u64 tmp1, tmp2;
    memcpy(&tmp1, psrc, 8);
    memcpy(&tmp2, psrc + 8, 8);
    memcpy(pdst, &tmp1, 8);
    memcpy(pdst + 8, &tmp2, 8);
#else
    char tmp[16];
    repeat16_incr(byte_move_src)
    repeat16_incr(byte_move_dst)
#endif
}

static_inline bool byte_match_2(void *buf, const char *pat) {
#if !YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
    v16_uni u1, u2;
    memcpy(&u1, buf, 2);
    memcpy(&u2, pat, 2);
    return u1.u == u2.u;
#else
    return ((char *)buf)[0] == ((const char *)pat)[0] &&
           ((char *)buf)[1] == ((const char *)pat)[1];
#endif
}

static_inline bool byte_match_4(void *buf, const char *pat) {
#if !YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
    v32_uni u1, u2;
    memcpy(&u1, buf, 4);
    memcpy(&u2, pat, 4);
    return u1.u == u2.u;
#else
    return ((char *)buf)[0] == ((const char *)pat)[0] &&
           ((char *)buf)[1] == ((const char *)pat)[1] &&
           ((char *)buf)[2] == ((const char *)pat)[2] &&
           ((char *)buf)[3] == ((const char *)pat)[3];
#endif
}

static_inline u16 byte_load_2(const void *src) {
    v16_uni uni;
#if !YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
    memcpy(&uni, src, 2);
#else
    uni.v.c[0] = ((const char *)src)[0];
    uni.v.c[1] = ((const char *)src)[1];
#endif
    return uni.u;
}

static_inline u32 byte_load_3(const void *src) {
    v32_uni uni;
#if !YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
    memcpy(&uni, src, 2);
    uni.v.c[2] = ((const char *)src)[2];
    uni.v.c[3] = 0;
#else
    uni.v.c[0] = ((const char *)src)[0];
    uni.v.c[1] = ((const char *)src)[1];
    uni.v.c[2] = ((const char *)src)[2];
    uni.v.c[3] = 0;
#endif
    return uni.u;
}

static_inline u32 byte_load_4(const void *src) {
    v32_uni uni;
#if !YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS
    memcpy(&uni, src, 4);
#else
    uni.v.c[0] = ((const char *)src)[0];
    uni.v.c[1] = ((const char *)src)[1];
    uni.v.c[2] = ((const char *)src)[2];
    uni.v.c[3] = ((const char *)src)[3];
#endif
    return uni.u;
}



/*==============================================================================
 * Number Utils
 * These functions are used to detect and convert NaN and Inf numbers.
 * The `memcpy` is used to avoid violating the strict aliasing rule.
 *============================================================================*/

/** Convert raw binary to double. */
static_inline f64 f64_from_raw(u64 u) {
    f64 f;
    memcpy(&f, &u, sizeof(u));
    return f;
}

/** Convert raw binary to float. */
static_inline f32 f32_from_raw(u32 u) {
    f32 f;
    memcpy(&f, &u, sizeof(u));
    return f;
}

/** Convert double to raw binary. */
static_inline u64 f64_to_raw(f64 f) {
    u64 u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

/** Convert double to raw binary. */
static_inline u32 f32_to_raw(f32 f) {
    u32 u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

/** Get raw 'infinity' with sign. */
static_inline u64 f64_raw_get_inf(bool sign) {
#if YYJSON_HAS_IEEE_754
    return F64_RAW_INF | ((u64)sign << 63);
#elif defined(INFINITY)
    return f64_to_raw(sign ? -INFINITY : INFINITY);
#else
    return f64_to_raw(sign ? -HUGE_VAL : HUGE_VAL);
#endif
}

/** Get raw 'nan' with sign. */
static_inline u64 f64_raw_get_nan(bool sign) {
#if YYJSON_HAS_IEEE_754
    return F64_RAW_NAN | ((u64)sign << 63);
#elif defined(NAN)
    return f64_to_raw(sign ? (f64)-NAN : (f64)NAN);
#else
    return f64_to_raw((sign ? -0.0 : 0.0) / 0.0);
#endif
}

/** Casting double to float, allow overflow. */
#if yyjson_has_attribute(no_sanitize)
__attribute__((no_sanitize("undefined")))
#elif yyjson_gcc_available(4, 9, 0)
__attribute__((__no_sanitize_undefined__))
#endif
static_inline f32 f64_to_f32(f64 val) {
    return (f32)val;
}



/*==============================================================================
 * Size Utils
 * These functions are used for memory allocation.
 *============================================================================*/

/** Returns whether the size is overflow after increment. */
static_inline bool size_add_is_overflow(usize size, usize add) {
    return size > (size + add);
}

/** Returns whether the size is power of 2 (size should not be 0). */
static_inline bool size_is_pow2(usize size) {
    return (size & (size - 1)) == 0;
}

/** Align size upwards (may overflow). */
static_inline usize size_align_up(usize size, usize align) {
    if (size_is_pow2(align)) {
        return (size + (align - 1)) & ~(align - 1);
    } else {
        return size + align - (size + align - 1) % align - 1;
    }
}

/** Align size downwards. */
static_inline usize size_align_down(usize size, usize align) {
    if (size_is_pow2(align)) {
        return size & ~(align - 1);
    } else {
        return size - (size % align);
    }
}

/** Align address upwards (may overflow). */
static_inline void *mem_align_up(void *mem, usize align) {
    usize size;
    memcpy(&size, &mem, sizeof(usize));
    size = size_align_up(size, align);
    memcpy(&mem, &size, sizeof(usize));
    return mem;
}



/*==============================================================================
 * Bits Utils
 * These functions are used by the floating-point number reader and writer.
 *============================================================================*/

/** Returns the number of leading 0-bits in value (input should not be 0). */
static_inline u32 u64_lz_bits(u64 v) {
#if GCC_HAS_CLZLL
    return (u32)__builtin_clzll(v);
#elif MSC_HAS_BIT_SCAN_64
    unsigned long r;
    _BitScanReverse64(&r, v);
    return (u32)63 - (u32)r;
#elif MSC_HAS_BIT_SCAN
    unsigned long hi, lo;
    bool hi_set = _BitScanReverse(&hi, (u32)(v >> 32)) != 0;
    _BitScanReverse(&lo, (u32)v);
    hi |= 32;
    return (u32)63 - (u32)(hi_set ? hi : lo);
#else
    /* branchless, use De Bruijn sequence */
    /* see: https://www.chessprogramming.org/BitScan */
    const u8 table[64] = {
        63, 16, 62,  7, 15, 36, 61,  3,  6, 14, 22, 26, 35, 47, 60,  2,
         9,  5, 28, 11, 13, 21, 42, 19, 25, 31, 34, 40, 46, 52, 59,  1,
        17,  8, 37,  4, 23, 27, 48, 10, 29, 12, 43, 20, 32, 41, 53, 18,
        38, 24, 49, 30, 44, 33, 54, 39, 50, 45, 55, 51, 56, 57, 58,  0
    };
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return table[(v * U64(0x03F79D71, 0xB4CB0A89)) >> 58];
#endif
}

/** Returns the number of trailing 0-bits in value (input should not be 0). */
static_inline u32 u64_tz_bits(u64 v) {
#if GCC_HAS_CTZLL
    return (u32)__builtin_ctzll(v);
#elif MSC_HAS_BIT_SCAN_64
    unsigned long r;
    _BitScanForward64(&r, v);
    return (u32)r;
#elif MSC_HAS_BIT_SCAN
    unsigned long lo, hi;
    bool lo_set = _BitScanForward(&lo, (u32)(v)) != 0;
    _BitScanForward(&hi, (u32)(v >> 32));
    hi += 32;
    return lo_set ? lo : hi;
#else
    /* branchless, use De Bruijn sequence */
    /* see: https://www.chessprogramming.org/BitScan */
    const u8 table[64] = {
         0,  1,  2, 53,  3,  7, 54, 27,  4, 38, 41,  8, 34, 55, 48, 28,
        62,  5, 39, 46, 44, 42, 22,  9, 24, 35, 59, 56, 49, 18, 29, 11,
        63, 52,  6, 26, 37, 40, 33, 47, 61, 45, 43, 21, 23, 58, 17, 10,
        51, 25, 36, 32, 60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13, 12
    };
    return table[((v & (~v + 1)) * U64(0x022FDD63, 0xCC95386D)) >> 58];
#endif
}



/*==============================================================================
 * 128-bit Integer Utils
 * These functions are used by the floating-point number reader and writer.
 *============================================================================*/

/** Multiplies two 64-bit unsigned integers (a * b),
    returns the 128-bit result as 'hi' and 'lo'. */
static_inline void u128_mul(u64 a, u64 b, u64 *hi, u64 *lo) {
#if YYJSON_HAS_INT128
    u128 m = (u128)a * b;
    *hi = (u64)(m >> 64);
    *lo = (u64)(m);
#elif MSC_HAS_UMUL128
    *lo = _umul128(a, b, hi);
#else
    u32 a0 = (u32)(a), a1 = (u32)(a >> 32);
    u32 b0 = (u32)(b), b1 = (u32)(b >> 32);
    u64 p00 = (u64)a0 * b0, p01 = (u64)a0 * b1;
    u64 p10 = (u64)a1 * b0, p11 = (u64)a1 * b1;
    u64 m0 = p01 + (p00 >> 32);
    u32 m00 = (u32)(m0), m01 = (u32)(m0 >> 32);
    u64 m1 = p10 + m00;
    u32 m10 = (u32)(m1), m11 = (u32)(m1 >> 32);
    *hi = p11 + m01 + m11;
    *lo = ((u64)m10 << 32) | (u32)p00;
#endif
}

/** Multiplies two 64-bit unsigned integers and add a value (a * b + c),
    returns the 128-bit result as 'hi' and 'lo'. */
static_inline void u128_mul_add(u64 a, u64 b, u64 c, u64 *hi, u64 *lo) {
#if YYJSON_HAS_INT128
    u128 m = (u128)a * b + c;
    *hi = (u64)(m >> 64);
    *lo = (u64)(m);
#else
    u64 h, l, t;
    u128_mul(a, b, &h, &l);
    t = l + c;
    h += (u64)(((t < l) | (t < c)));
    *hi = h;
    *lo = t;
#endif
}



/*==============================================================================
 * File Utils
 * These functions are used to read and write JSON files.
 *============================================================================*/

#define YYJSON_FOPEN_EXT
#if !defined(_MSC_VER) && defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#   if __GLIBC_PREREQ(2, 7)
#       undef YYJSON_FOPEN_EXT
#       define YYJSON_FOPEN_EXT "e" /* glibc extension to enable O_CLOEXEC */
#   endif
#endif

static_inline FILE *fopen_safe(const char *path, const char *mode) {
#if YYJSON_MSC_VER >= 1400
    FILE *file = NULL;
    if (fopen_s(&file, path, mode) != 0) return NULL;
    return file;
#else
    return fopen(path, mode);
#endif
}

static_inline FILE *fopen_readonly(const char *path) {
    return fopen_safe(path, "rb" YYJSON_FOPEN_EXT);
}

static_inline FILE *fopen_writeonly(const char *path) {
    return fopen_safe(path, "wb" YYJSON_FOPEN_EXT);
}

static_inline usize fread_safe(void *buf, usize size, FILE *file) {
#if YYJSON_MSC_VER >= 1400
    return fread_s(buf, size, 1, size, file);
#else
    return fread(buf, 1, size, file);
#endif
}



/*==============================================================================
 * Default Memory Allocator
 *
 * This is a simple libc memory allocator wrapper.
 *============================================================================*/

static void *default_malloc(void *ctx, usize size) {
    return malloc(size);
}

static void *default_realloc(void *ctx, void *ptr, usize old_size, usize size) {
    return realloc(ptr, size);
}

static void default_free(void *ctx, void *ptr) {
    free(ptr);
}

static const yyjson_alc YYJSON_DEFAULT_ALC = {
    default_malloc, default_realloc, default_free, NULL
};



/*==============================================================================
 * Null Memory Allocator
 *
 * This allocator is just a placeholder to ensure that the internal
 * malloc/realloc/free function pointers are not null.
 *============================================================================*/

static void *null_malloc(void *ctx, usize size) {
    return NULL;
}

static void *null_realloc(void *ctx, void *ptr, usize old_size, usize size) {
    return NULL;
}

static void null_free(void *ctx, void *ptr) {
    return;
}

static const yyjson_alc YYJSON_NULL_ALC = {
    null_malloc, null_realloc, null_free, NULL
};



/*==============================================================================
 * Pool Memory Allocator
 *
 * This allocator is initialized with a fixed-size buffer.
 * The buffer is split into multiple memory chunks for memory allocation.
 *============================================================================*/

/** memory chunk header */
typedef struct pool_chunk {
    usize size; /* chunk memory size, include chunk header */
    struct pool_chunk *next; /* linked list, nullable */
    /* char mem[]; flexible array member */
} pool_chunk;

/** allocator ctx header */
typedef struct pool_ctx {
    usize size; /* total memory size, include ctx header */
    pool_chunk *free_list; /* linked list, nullable */
    /* pool_chunk chunks[]; flexible array member */
} pool_ctx;

/** align up the input size to chunk size */
static_inline void pool_size_align(usize *size) {
    *size = size_align_up(*size, sizeof(pool_chunk)) + sizeof(pool_chunk);
}

static void *pool_malloc(void *ctx_ptr, usize size) {
    /* assert(size != 0) */
    pool_ctx *ctx = (pool_ctx *)ctx_ptr;
    pool_chunk *next, *prev = NULL, *cur = ctx->free_list;

    if (unlikely(size >= ctx->size)) return NULL;
    pool_size_align(&size);

    while (cur) {
        if (cur->size < size) {
            /* not enough space, try next chunk */
            prev = cur;
            cur = cur->next;
            continue;
        }
        if (cur->size >= size + sizeof(pool_chunk) * 2) {
            /* too much space, split this chunk */
            next = (pool_chunk *)(void *)((u8 *)cur + size);
            next->size = cur->size - size;
            next->next = cur->next;
            cur->size = size;
        } else {
            /* just enough space, use whole chunk */
            next = cur->next;
        }
        if (prev) prev->next = next;
        else ctx->free_list = next;
        return (void *)(cur + 1);
    }
    return NULL;
}

static void pool_free(void *ctx_ptr, void *ptr) {
    /* assert(ptr != NULL) */
    pool_ctx *ctx = (pool_ctx *)ctx_ptr;
    pool_chunk *cur = ((pool_chunk *)ptr) - 1;
    pool_chunk *prev = NULL, *next = ctx->free_list;

    while (next && next < cur) {
        prev = next;
        next = next->next;
    }
    if (prev) prev->next = cur;
    else ctx->free_list = cur;
    cur->next = next;

    if (next && ((u8 *)cur + cur->size) == (u8 *)next) {
        /* merge cur to higher chunk */
        cur->size += next->size;
        cur->next = next->next;
    }
    if (prev && ((u8 *)prev + prev->size) == (u8 *)cur) {
        /* merge cur to lower chunk */
        prev->size += cur->size;
        prev->next = cur->next;
    }
}

static void *pool_realloc(void *ctx_ptr, void *ptr,
                          usize old_size, usize size) {
    /* assert(ptr != NULL && size != 0 && old_size < size) */
    pool_ctx *ctx = (pool_ctx *)ctx_ptr;
    pool_chunk *cur = ((pool_chunk *)ptr) - 1, *prev, *next, *tmp;

    /* check size */
    if (unlikely(size >= ctx->size)) return NULL;
    pool_size_align(&old_size);
    pool_size_align(&size);
    if (unlikely(old_size == size)) return ptr;

    /* find next and prev chunk */
    prev = NULL;
    next = ctx->free_list;
    while (next && next < cur) {
        prev = next;
        next = next->next;
    }

    if ((u8 *)cur + cur->size == (u8 *)next && cur->size + next->size >= size) {
        /* merge to higher chunk if they are contiguous */
        usize free_size = cur->size + next->size - size;
        if (free_size > sizeof(pool_chunk) * 2) {
            tmp = (pool_chunk *)(void *)((u8 *)cur + size);
            if (prev) prev->next = tmp;
            else ctx->free_list = tmp;
            tmp->next = next->next;
            tmp->size = free_size;
            cur->size = size;
        } else {
            if (prev) prev->next = next->next;
            else ctx->free_list = next->next;
            cur->size += next->size;
        }
        return ptr;
    } else {
        /* fallback to malloc and memcpy */
        void *new_ptr = pool_malloc(ctx_ptr, size - sizeof(pool_chunk));
        if (new_ptr) {
            memcpy(new_ptr, ptr, cur->size - sizeof(pool_chunk));
            pool_free(ctx_ptr, ptr);
        }
        return new_ptr;
    }
}

bool yyjson_alc_pool_init(yyjson_alc *alc, void *buf, usize size) {
    pool_chunk *chunk;
    pool_ctx *ctx;

    if (unlikely(!alc)) return false;
    *alc = YYJSON_NULL_ALC;
    if (size < sizeof(pool_ctx) * 4) return false;
    ctx = (pool_ctx *)mem_align_up(buf, sizeof(pool_ctx));
    if (unlikely(!ctx)) return false;
    size -= (usize)((u8 *)ctx - (u8 *)buf);
    size = size_align_down(size, sizeof(pool_ctx));

    chunk = (pool_chunk *)(ctx + 1);
    chunk->size = size - sizeof(pool_ctx);
    chunk->next = NULL;
    ctx->size = size;
    ctx->free_list = chunk;

    alc->malloc = pool_malloc;
    alc->realloc = pool_realloc;
    alc->free = pool_free;
    alc->ctx = (void *)ctx;
    return true;
}



/*==============================================================================
 * Dynamic Memory Allocator
 *
 * This allocator allocates memory on demand and does not immediately release
 * unused memory. Instead, it places the unused memory into a freelist for
 * potential reuse in the future. It is only when the entire allocator is
 * destroyed that all previously allocated memory is released at once.
 *============================================================================*/

/** memory chunk header */
typedef struct dyn_chunk {
    usize size; /* chunk size, include header */
    struct dyn_chunk *next;
    /* char mem[]; flexible array member */
} dyn_chunk;

/** allocator ctx header */
typedef struct {
    dyn_chunk free_list; /* dummy header, sorted from small to large */
    dyn_chunk used_list; /* dummy header */
} dyn_ctx;

/** align up the input size to chunk size */
static_inline bool dyn_size_align(usize *size) {
    usize alc_size = *size + sizeof(dyn_chunk);
    alc_size = size_align_up(alc_size, YYJSON_ALC_DYN_MIN_SIZE);
    if (unlikely(alc_size < *size)) return false; /* overflow */
    *size = alc_size;
    return true;
}

/** remove a chunk from list (the chunk must already be in the list) */
static_inline void dyn_chunk_list_remove(dyn_chunk *list, dyn_chunk *chunk) {
    dyn_chunk *prev = list, *cur;
    for (cur = prev->next; cur; cur = cur->next) {
        if (cur == chunk) {
            prev->next = cur->next;
            cur->next = NULL;
            return;
        }
        prev = cur;
    }
}

/** add a chunk to list header (the chunk must not be in the list) */
static_inline void dyn_chunk_list_add(dyn_chunk *list, dyn_chunk *chunk) {
    chunk->next = list->next;
    list->next = chunk;
}

static void *dyn_malloc(void *ctx_ptr, usize size) {
    /* assert(size != 0) */
    const yyjson_alc def = YYJSON_DEFAULT_ALC;
    dyn_ctx *ctx = (dyn_ctx *)ctx_ptr;
    dyn_chunk *chunk, *prev;
    if (unlikely(!dyn_size_align(&size))) return NULL;

    /* freelist is empty, create new chunk */
    if (!ctx->free_list.next) {
        chunk = (dyn_chunk *)def.malloc(def.ctx, size);
        if (unlikely(!chunk)) return NULL;
        chunk->size = size;
        chunk->next = NULL;
        dyn_chunk_list_add(&ctx->used_list, chunk);
        return (void *)(chunk + 1);
    }

    /* find a large enough chunk, or resize the largest chunk */
    prev = &ctx->free_list;
    while (true) {
        chunk = prev->next;
        if (chunk->size >= size) { /* enough size, reuse this chunk */
            prev->next = chunk->next;
            dyn_chunk_list_add(&ctx->used_list, chunk);
            return (void *)(chunk + 1);
        }
        if (!chunk->next) { /* resize the largest chunk */
            chunk = (dyn_chunk *)def.realloc(def.ctx, chunk, chunk->size, size);
            if (unlikely(!chunk)) return NULL;
            prev->next = NULL;
            chunk->size = size;
            dyn_chunk_list_add(&ctx->used_list, chunk);
            return (void *)(chunk + 1);
        }
        prev = chunk;
    }
}

static void *dyn_realloc(void *ctx_ptr, void *ptr,
                          usize old_size, usize size) {
    /* assert(ptr != NULL && size != 0 && old_size < size) */
    const yyjson_alc def = YYJSON_DEFAULT_ALC;
    dyn_ctx *ctx = (dyn_ctx *)ctx_ptr;
    dyn_chunk *new_chunk, *chunk = (dyn_chunk *)ptr - 1;
    if (unlikely(!dyn_size_align(&size))) return NULL;
    if (chunk->size >= size) return ptr;

    dyn_chunk_list_remove(&ctx->used_list, chunk);
    new_chunk = (dyn_chunk *)def.realloc(def.ctx, chunk, chunk->size, size);
    if (likely(new_chunk)) {
        new_chunk->size = size;
        chunk = new_chunk;
    }
    dyn_chunk_list_add(&ctx->used_list, chunk);
    return new_chunk ? (void *)(new_chunk + 1) : NULL;
}

static void dyn_free(void *ctx_ptr, void *ptr) {
    /* assert(ptr != NULL) */
    dyn_ctx *ctx = (dyn_ctx *)ctx_ptr;
    dyn_chunk *chunk = (dyn_chunk *)ptr - 1, *prev;

    dyn_chunk_list_remove(&ctx->used_list, chunk);
    for (prev = &ctx->free_list; prev; prev = prev->next) {
        if (!prev->next || prev->next->size >= chunk->size) {
            chunk->next = prev->next;
            prev->next = chunk;
            break;
        }
    }
}

yyjson_alc *yyjson_alc_dyn_new(void) {
    const yyjson_alc def = YYJSON_DEFAULT_ALC;
    usize hdr_len = sizeof(yyjson_alc) + sizeof(dyn_ctx);
    yyjson_alc *alc = (yyjson_alc *)def.malloc(def.ctx, hdr_len);
    dyn_ctx *ctx = (dyn_ctx *)(void *)(alc + 1);
    if (unlikely(!alc)) return NULL;
    alc->malloc = dyn_malloc;
    alc->realloc = dyn_realloc;
    alc->free = dyn_free;
    alc->ctx = alc + 1;
    memset(ctx, 0, sizeof(*ctx));
    return alc;
}

void yyjson_alc_dyn_free(yyjson_alc *alc) {
    const yyjson_alc def = YYJSON_DEFAULT_ALC;
    dyn_ctx *ctx = (dyn_ctx *)(void *)(alc + 1);
    dyn_chunk *chunk, *next;
    if (unlikely(!alc)) return;
    for (chunk = ctx->free_list.next; chunk; chunk = next) {
        next = chunk->next;
        def.free(def.ctx, chunk);
    }
    for (chunk = ctx->used_list.next; chunk; chunk = next) {
        next = chunk->next;
        def.free(def.ctx, chunk);
    }
    def.free(def.ctx, alc);
}



/*==============================================================================
 * JSON document and value
 *============================================================================*/

static_inline void unsafe_yyjson_str_pool_release(yyjson_str_pool *pool,
                                                  yyjson_alc *alc) {
    yyjson_str_chunk *chunk = pool->chunks, *next;
    while (chunk) {
        next = chunk->next;
        alc->free(alc->ctx, chunk);
        chunk = next;
    }
}

static_inline void unsafe_yyjson_val_pool_release(yyjson_val_pool *pool,
                                                  yyjson_alc *alc) {
    yyjson_val_chunk *chunk = pool->chunks, *next;
    while (chunk) {
        next = chunk->next;
        alc->free(alc->ctx, chunk);
        chunk = next;
    }
}

bool unsafe_yyjson_str_pool_grow(yyjson_str_pool *pool,
                                 const yyjson_alc *alc, usize len) {
    yyjson_str_chunk *chunk;
    usize size, max_len;

    /* create a new chunk */
    max_len = USIZE_MAX - sizeof(yyjson_str_chunk);
    if (unlikely(len > max_len)) return false;
    size = len + sizeof(yyjson_str_chunk);
    size = yyjson_max(pool->chunk_size, size);
    chunk = (yyjson_str_chunk *)alc->malloc(alc->ctx, size);
    if (unlikely(!chunk)) return false;

    /* insert the new chunk as the head of the linked list */
    chunk->next = pool->chunks;
    chunk->chunk_size = size;
    pool->chunks = chunk;
    pool->cur = (char *)chunk + sizeof(yyjson_str_chunk);
    pool->end = (char *)chunk + size;

    /* the next chunk is twice the size of the current one */
    size = yyjson_min(pool->chunk_size * 2, pool->chunk_size_max);
    if (size < pool->chunk_size) size = pool->chunk_size_max; /* overflow */
    pool->chunk_size = size;
    return true;
}

bool unsafe_yyjson_val_pool_grow(yyjson_val_pool *pool,
                                 const yyjson_alc *alc, usize count) {
    yyjson_val_chunk *chunk;
    usize size, max_count;

    /* create a new chunk */
    max_count = USIZE_MAX / sizeof(yyjson_mut_val) - 1;
    if (unlikely(count > max_count)) return false;
    size = (count + 1) * sizeof(yyjson_mut_val);
    size = yyjson_max(pool->chunk_size, size);
    chunk = (yyjson_val_chunk *)alc->malloc(alc->ctx, size);
    if (unlikely(!chunk)) return false;

    /* insert the new chunk as the head of the linked list */
    chunk->next = pool->chunks;
    chunk->chunk_size = size;
    pool->chunks = chunk;
    pool->cur = (yyjson_mut_val *)(void *)((u8 *)chunk) + 1;
    pool->end = (yyjson_mut_val *)(void *)((u8 *)chunk + size);

    /* the next chunk is twice the size of the current one */
    size = yyjson_min(pool->chunk_size * 2, pool->chunk_size_max);
    if (size < pool->chunk_size) size = pool->chunk_size_max; /* overflow */
    pool->chunk_size = size;
    return true;
}

bool yyjson_mut_doc_set_str_pool_size(yyjson_mut_doc *doc, size_t len) {
    usize max_size = USIZE_MAX - sizeof(yyjson_str_chunk);
    if (!doc || !len || len > max_size) return false;
    doc->str_pool.chunk_size = len + sizeof(yyjson_str_chunk);
    return true;
}

bool yyjson_mut_doc_set_val_pool_size(yyjson_mut_doc *doc, size_t count) {
    usize max_count = USIZE_MAX / sizeof(yyjson_mut_val) - 1;
    if (!doc || !count || count > max_count) return false;
    doc->val_pool.chunk_size = (count + 1) * sizeof(yyjson_mut_val);
    return true;
}

void yyjson_mut_doc_free(yyjson_mut_doc *doc) {
    if (doc) {
        yyjson_alc alc = doc->alc;
        memset(&doc->alc, 0, sizeof(alc));
        unsafe_yyjson_str_pool_release(&doc->str_pool, &alc);
        unsafe_yyjson_val_pool_release(&doc->val_pool, &alc);
        alc.free(alc.ctx, doc);
    }
}

yyjson_mut_doc *yyjson_mut_doc_new(const yyjson_alc *alc) {
    yyjson_mut_doc *doc;
    if (!alc) alc = &YYJSON_DEFAULT_ALC;
    doc = (yyjson_mut_doc *)alc->malloc(alc->ctx, sizeof(yyjson_mut_doc));
    if (!doc) return NULL;
    memset(doc, 0, sizeof(yyjson_mut_doc));

    doc->alc = *alc;
    doc->str_pool.chunk_size = YYJSON_MUT_DOC_STR_POOL_INIT_SIZE;
    doc->str_pool.chunk_size_max = YYJSON_MUT_DOC_STR_POOL_MAX_SIZE;
    doc->val_pool.chunk_size = YYJSON_MUT_DOC_VAL_POOL_INIT_SIZE;
    doc->val_pool.chunk_size_max = YYJSON_MUT_DOC_VAL_POOL_MAX_SIZE;
    return doc;
}

yyjson_mut_doc *yyjson_doc_mut_copy(yyjson_doc *doc, const yyjson_alc *alc) {
    yyjson_mut_doc *m_doc;
    yyjson_mut_val *m_val;

    if (!doc || !doc->root) return NULL;
    m_doc = yyjson_mut_doc_new(alc);
    if (!m_doc) return NULL;
    m_val = yyjson_val_mut_copy(m_doc, doc->root);
    if (!m_val) {
        yyjson_mut_doc_free(m_doc);
        return NULL;
    }
    yyjson_mut_doc_set_root(m_doc, m_val);
    return m_doc;
}

yyjson_mut_doc *yyjson_mut_doc_mut_copy(yyjson_mut_doc *doc,
                                        const yyjson_alc *alc) {
    yyjson_mut_doc *m_doc;
    yyjson_mut_val *m_val;

    if (!doc) return NULL;
    if (!doc->root) return yyjson_mut_doc_new(alc);

    m_doc = yyjson_mut_doc_new(alc);
    if (!m_doc) return NULL;
    m_val = yyjson_mut_val_mut_copy(m_doc, doc->root);
    if (!m_val) {
        yyjson_mut_doc_free(m_doc);
        return NULL;
    }
    yyjson_mut_doc_set_root(m_doc, m_val);
    return m_doc;
}

yyjson_mut_val *yyjson_val_mut_copy(yyjson_mut_doc *m_doc,
                                    yyjson_val *i_vals) {
    /*
     The immutable object or array stores all sub-values in a contiguous memory,
     We copy them to another contiguous memory as mutable values,
     then reconnect the mutable values with the original relationship.
     */
    usize i_vals_len;
    yyjson_mut_val *m_vals, *m_val;
    yyjson_val *i_val, *i_end;

    if (!m_doc || !i_vals) return NULL;
    i_end = unsafe_yyjson_get_next(i_vals);
    i_vals_len = (usize)(unsafe_yyjson_get_next(i_vals) - i_vals);
    m_vals = unsafe_yyjson_mut_val(m_doc, i_vals_len);
    if (!m_vals) return NULL;
    i_val = i_vals;
    m_val = m_vals;

    for (; i_val < i_end; i_val++, m_val++) {
        yyjson_type type = unsafe_yyjson_get_type(i_val);
        m_val->tag = i_val->tag;
        m_val->uni.u64 = i_val->uni.u64;
        if (type == YYJSON_TYPE_STR || type == YYJSON_TYPE_RAW) {
            const char *str = i_val->uni.str;
            usize str_len = unsafe_yyjson_get_len(i_val);
            m_val->uni.str = unsafe_yyjson_mut_strncpy(m_doc, str, str_len);
            if (!m_val->uni.str) return NULL;
        } else if (type == YYJSON_TYPE_ARR) {
            usize len = unsafe_yyjson_get_len(i_val);
            if (len > 0) {
                yyjson_val *ii_val = i_val + 1, *ii_next;
                yyjson_mut_val *mm_val = m_val + 1, *mm_ctn = m_val, *mm_next;
                while (len-- > 1) {
                    ii_next = unsafe_yyjson_get_next(ii_val);
                    mm_next = mm_val + (ii_next - ii_val);
                    mm_val->next = mm_next;
                    ii_val = ii_next;
                    mm_val = mm_next;
                }
                mm_val->next = mm_ctn + 1;
                mm_ctn->uni.ptr = mm_val;
            }
        } else if (type == YYJSON_TYPE_OBJ) {
            usize len = unsafe_yyjson_get_len(i_val);
            if (len > 0) {
                yyjson_val *ii_key = i_val + 1, *ii_nextkey;
                yyjson_mut_val *mm_key = m_val + 1, *mm_ctn = m_val;
                yyjson_mut_val *mm_nextkey;
                while (len-- > 1) {
                    ii_nextkey = unsafe_yyjson_get_next(ii_key + 1);
                    mm_nextkey = mm_key + (ii_nextkey - ii_key);
                    mm_key->next = mm_key + 1;
                    mm_key->next->next = mm_nextkey;
                    ii_key = ii_nextkey;
                    mm_key = mm_nextkey;
                }
                mm_key->next = mm_key + 1;
                mm_key->next->next = mm_ctn + 1;
                mm_ctn->uni.ptr = mm_key;
            }
        }
    }
    return m_vals;
}

static yyjson_mut_val *unsafe_yyjson_mut_val_mut_copy(yyjson_mut_doc *m_doc,
                                                      yyjson_mut_val *m_vals) {
    /*
     The mutable object or array stores all sub-values in a circular linked
     list, so we can traverse them in the same loop. The traversal starts from
     the last item, continues with the first item in a list, and ends with the
     second to last item, which needs to be linked to the last item to close the
     circle.
     */
    yyjson_mut_val *m_val = unsafe_yyjson_mut_val(m_doc, 1);
    if (unlikely(!m_val)) return NULL;
    m_val->tag = m_vals->tag;

    switch (unsafe_yyjson_get_type(m_vals)) {
        case YYJSON_TYPE_OBJ:
        case YYJSON_TYPE_ARR:
            if (unsafe_yyjson_get_len(m_vals) > 0) {
                yyjson_mut_val *last = (yyjson_mut_val *)m_vals->uni.ptr;
                yyjson_mut_val *next = last->next, *prev;
                prev = unsafe_yyjson_mut_val_mut_copy(m_doc, last);
                if (!prev) return NULL;
                m_val->uni.ptr = (void *)prev;
                while (next != last) {
                    prev->next = unsafe_yyjson_mut_val_mut_copy(m_doc, next);
                    if (!prev->next) return NULL;
                    prev = prev->next;
                    next = next->next;
                }
                prev->next = (yyjson_mut_val *)m_val->uni.ptr;
            }
            break;
        case YYJSON_TYPE_RAW:
        case YYJSON_TYPE_STR: {
            const char *str = m_vals->uni.str;
            usize str_len = unsafe_yyjson_get_len(m_vals);
            m_val->uni.str = unsafe_yyjson_mut_strncpy(m_doc, str, str_len);
            if (!m_val->uni.str) return NULL;
            break;
        }
        default:
            m_val->uni = m_vals->uni;
            break;
    }
    return m_val;
}

yyjson_mut_val *yyjson_mut_val_mut_copy(yyjson_mut_doc *doc,
                                        yyjson_mut_val *val) {
    if (doc && val) return unsafe_yyjson_mut_val_mut_copy(doc, val);
    return NULL;
}

/* Count the number of values and the total length of the strings. */
static void yyjson_mut_stat(yyjson_mut_val *val,
                            usize *val_sum, usize *str_sum) {
    yyjson_type type = unsafe_yyjson_get_type(val);
    *val_sum += 1;
    if (type == YYJSON_TYPE_ARR || type == YYJSON_TYPE_OBJ) {
        yyjson_mut_val *child = (yyjson_mut_val *)val->uni.ptr;
        usize len = unsafe_yyjson_get_len(val), i;
        len <<= (u8)(type == YYJSON_TYPE_OBJ);
        *val_sum += len;
        for (i = 0; i < len; i++) {
            yyjson_type stype = unsafe_yyjson_get_type(child);
            if (stype == YYJSON_TYPE_STR || stype == YYJSON_TYPE_RAW) {
                *str_sum += unsafe_yyjson_get_len(child) + 1;
            } else if (stype == YYJSON_TYPE_ARR || stype == YYJSON_TYPE_OBJ) {
                yyjson_mut_stat(child, val_sum, str_sum);
                *val_sum -= 1;
            }
            child = child->next;
        }
    } else if (type == YYJSON_TYPE_STR || type == YYJSON_TYPE_RAW) {
        *str_sum += unsafe_yyjson_get_len(val) + 1;
    }
}

/* Copy mutable values to immutable value pool. */
static usize yyjson_imut_copy(yyjson_val **val_ptr, char **buf_ptr,
                              yyjson_mut_val *mval) {
    yyjson_val *val = *val_ptr;
    yyjson_type type = unsafe_yyjson_get_type(mval);
    if (type == YYJSON_TYPE_ARR || type == YYJSON_TYPE_OBJ) {
        yyjson_mut_val *child = (yyjson_mut_val *)mval->uni.ptr;
        usize len = unsafe_yyjson_get_len(mval), i;
        usize val_sum = 1;
        if (type == YYJSON_TYPE_OBJ) {
            if (len) child = child->next->next;
            len <<= 1;
        } else {
            if (len) child = child->next;
        }
        *val_ptr = val + 1;
        for (i = 0; i < len; i++) {
            val_sum += yyjson_imut_copy(val_ptr, buf_ptr, child);
            child = child->next;
        }
        val->tag = mval->tag;
        val->uni.ofs = val_sum * sizeof(yyjson_val);
        return val_sum;
    } else if (type == YYJSON_TYPE_STR || type == YYJSON_TYPE_RAW) {
        char *buf = *buf_ptr;
        usize len = unsafe_yyjson_get_len(mval);
        memcpy((void *)buf, (const void *)mval->uni.str, len);
        buf[len] = '\0';
        val->tag = mval->tag;
        val->uni.str = buf;
        *val_ptr = val + 1;
        *buf_ptr = buf + len + 1;
        return 1;
    } else {
        val->tag = mval->tag;
        val->uni = mval->uni;
        *val_ptr = val + 1;
        return 1;
    }
}

yyjson_doc *yyjson_mut_doc_imut_copy(yyjson_mut_doc *mdoc,
                                     const yyjson_alc *alc) {
    if (!mdoc) return NULL;
    return yyjson_mut_val_imut_copy(mdoc->root, alc);
}

yyjson_doc *yyjson_mut_val_imut_copy(yyjson_mut_val *mval,
                                     const yyjson_alc *alc) {
    usize val_num = 0, str_sum = 0, hdr_size, buf_size;
    yyjson_doc *doc = NULL;
    yyjson_val *val_hdr = NULL;

    /* This value should be NULL here. Setting a non-null value suppresses
       warning from the clang analyzer. */
    char *str_hdr = (char *)(void *)&str_sum;
    if (!mval) return NULL;
    if (!alc) alc = &YYJSON_DEFAULT_ALC;

    /* traverse the input value to get pool size */
    yyjson_mut_stat(mval, &val_num, &str_sum);

    /* create doc and val pool */
    hdr_size = size_align_up(sizeof(yyjson_doc), sizeof(yyjson_val));
    buf_size = hdr_size + val_num * sizeof(yyjson_val);
    doc = (yyjson_doc *)alc->malloc(alc->ctx, buf_size);
    if (!doc) return NULL;
    memset(doc, 0, sizeof(yyjson_doc));
    val_hdr = (yyjson_val *)(void *)((char *)(void *)doc + hdr_size);
    doc->root = val_hdr;
    doc->alc = *alc;

    /* create str pool */
    if (str_sum > 0) {
        str_hdr = (char *)alc->malloc(alc->ctx, str_sum);
        doc->str_pool = str_hdr;
        if (!str_hdr) {
            alc->free(alc->ctx, (void *)doc);
            return NULL;
        }
    }

    /* copy vals and strs */
    doc->val_read = yyjson_imut_copy(&val_hdr, &str_hdr, mval);
    doc->dat_read = str_sum + 1;
    return doc;
}

static_inline bool unsafe_yyjson_num_equals(void *lhs, void *rhs) {
    yyjson_val_uni *luni = &((yyjson_val *)lhs)->uni;
    yyjson_val_uni *runi = &((yyjson_val *)rhs)->uni;
    yyjson_subtype lt = unsafe_yyjson_get_subtype(lhs);
    yyjson_subtype rt = unsafe_yyjson_get_subtype(rhs);
    if (lt == rt) return luni->u64 == runi->u64;
    if (lt == YYJSON_SUBTYPE_SINT && rt == YYJSON_SUBTYPE_UINT) {
        return luni->i64 >= 0 && luni->u64 == runi->u64;
    }
    if (lt == YYJSON_SUBTYPE_UINT && rt == YYJSON_SUBTYPE_SINT) {
        return runi->i64 >= 0 && luni->u64 == runi->u64;
    }
    return false;
}

static_inline bool unsafe_yyjson_str_equals(void *lhs, void *rhs) {
    usize len = unsafe_yyjson_get_len(lhs);
    if (len != unsafe_yyjson_get_len(rhs)) return false;
    return !memcmp(unsafe_yyjson_get_str(lhs),
                   unsafe_yyjson_get_str(rhs), len);
}

bool unsafe_yyjson_equals(yyjson_val *lhs, yyjson_val *rhs) {
    yyjson_type type = unsafe_yyjson_get_type(lhs);
    if (type != unsafe_yyjson_get_type(rhs)) return false;

    switch (type) {
        case YYJSON_TYPE_OBJ: {
            usize len = unsafe_yyjson_get_len(lhs);
            if (len != unsafe_yyjson_get_len(rhs)) return false;
            if (len > 0) {
                yyjson_obj_iter iter;
                yyjson_obj_iter_init(rhs, &iter);
                lhs = unsafe_yyjson_get_first(lhs);
                while (len-- > 0) {
                    rhs = yyjson_obj_iter_getn(&iter, lhs->uni.str,
                                               unsafe_yyjson_get_len(lhs));
                    if (!rhs) return false;
                    if (!unsafe_yyjson_equals(lhs + 1, rhs)) return false;
                    lhs = unsafe_yyjson_get_next(lhs + 1);
                }
            }
            /* yyjson allows duplicate keys, so the check may be inaccurate */
            return true;
        }

        case YYJSON_TYPE_ARR: {
            usize len = unsafe_yyjson_get_len(lhs);
            if (len != unsafe_yyjson_get_len(rhs)) return false;
            if (len > 0) {
                lhs = unsafe_yyjson_get_first(lhs);
                rhs = unsafe_yyjson_get_first(rhs);
                while (len-- > 0) {
                    if (!unsafe_yyjson_equals(lhs, rhs)) return false;
                    lhs = unsafe_yyjson_get_next(lhs);
                    rhs = unsafe_yyjson_get_next(rhs);
                }
            }
            return true;
        }

        case YYJSON_TYPE_NUM:
            return unsafe_yyjson_num_equals(lhs, rhs);

        case YYJSON_TYPE_RAW:
        case YYJSON_TYPE_STR:
            return unsafe_yyjson_str_equals(lhs, rhs);

        case YYJSON_TYPE_NULL:
        case YYJSON_TYPE_BOOL:
            return lhs->tag == rhs->tag;

        default:
            return false;
    }
}

bool unsafe_yyjson_mut_equals(yyjson_mut_val *lhs, yyjson_mut_val *rhs) {
    yyjson_type type = unsafe_yyjson_get_type(lhs);
    if (type != unsafe_yyjson_get_type(rhs)) return false;

    switch (type) {
        case YYJSON_TYPE_OBJ: {
            usize len = unsafe_yyjson_get_len(lhs);
            if (len != unsafe_yyjson_get_len(rhs)) return false;
            if (len > 0) {
                yyjson_mut_obj_iter iter;
                yyjson_mut_obj_iter_init(rhs, &iter);
                lhs = (yyjson_mut_val *)lhs->uni.ptr;
                while (len-- > 0) {
                    rhs = yyjson_mut_obj_iter_getn(&iter, lhs->uni.str,
                                                   unsafe_yyjson_get_len(lhs));
                    if (!rhs) return false;
                    if (!unsafe_yyjson_mut_equals(lhs->next, rhs)) return false;
                    lhs = lhs->next->next;
                }
            }
            /* yyjson allows duplicate keys, so the check may be inaccurate */
            return true;
        }

        case YYJSON_TYPE_ARR: {
            usize len = unsafe_yyjson_get_len(lhs);
            if (len != unsafe_yyjson_get_len(rhs)) return false;
            if (len > 0) {
                lhs = (yyjson_mut_val *)lhs->uni.ptr;
                rhs = (yyjson_mut_val *)rhs->uni.ptr;
                while (len-- > 0) {
                    if (!unsafe_yyjson_mut_equals(lhs, rhs)) return false;
                    lhs = lhs->next;
                    rhs = rhs->next;
                }
            }
            return true;
        }

        case YYJSON_TYPE_NUM:
            return unsafe_yyjson_num_equals(lhs, rhs);

        case YYJSON_TYPE_RAW:
        case YYJSON_TYPE_STR:
            return unsafe_yyjson_str_equals(lhs, rhs);

        case YYJSON_TYPE_NULL:
        case YYJSON_TYPE_BOOL:
            return lhs->tag == rhs->tag;

        default:
            return false;
    }
}

static_inline bool is_utf8_bom(const u8 *hdr) {
    return hdr[0] == 0xEF && hdr[1] == 0xBB && hdr[2] == 0xBF;
}

static_inline bool is_utf16_bom(const u8 *hdr) {
    return ((hdr[0] == 0xFE && hdr[1] == 0xFF) ||
            (hdr[0] == 0xFF && hdr[1] == 0xFE));
}

static_inline bool is_utf32_bom(const u8 *hdr) {
    /* need check length to avoid zero padding */
    return ((hdr[0] == 0x00 && hdr[1] == 0x00 &&
             hdr[2] == 0xFE && hdr[3] == 0xFF) ||
            (hdr[0] == 0xFF && hdr[1] == 0xFE &&
             hdr[2] == 0x00 && hdr[3] == 0x00));
}

bool yyjson_locate_pos(const char *str, size_t len, size_t pos,
                       size_t *line, size_t *col, size_t *chr) {
    usize line_sum = 0, line_pos = 0, chr_sum = 0;
    const u8 *cur = (const u8 *)str;
    const u8 *end = cur + pos;

    if (!str || pos > len) {
        if (line) *line = 0;
        if (col) *col = 0;
        if (chr) *chr = 0;
        return false;
    }

    if (pos >= 3 && is_utf8_bom(cur)) cur += 3; /* don't count BOM */
    while (cur < end) {
        u8 c = *cur;
        chr_sum += 1;
        if (likely(c < 0x80)) {         /* 0xxxxxxx (0x00-0x7F) ASCII */
            if (c == '\n') {
                line_sum += 1;
                line_pos = chr_sum;
            }
            cur += 1;
        }
        else if (c < 0xC0) cur += 1;    /* 10xxxxxx (0x80-0xBF) Invalid */
        else if (c < 0xE0) cur += 2;    /* 110xxxxx (0xC0-0xDF) 2-byte UTF-8 */
        else if (c < 0xF0) cur += 3;    /* 1110xxxx (0xE0-0xEF) 3-byte UTF-8 */
        else if (c < 0xF8) cur += 4;    /* 11110xxx (0xF0-0xF7) 4-byte UTF-8 */
        else               cur += 1;    /* 11111xxx (0xF8-0xFF) Invalid */
    }
    if (line) *line = line_sum + 1;
    if (col) *col = chr_sum - line_pos + 1;
    if (chr) *chr = chr_sum;
    return true;
}



#if !YYJSON_DISABLE_UTILS

/*==============================================================================
 * JSON Pointer API (RFC 6901)
 *============================================================================*/

/**
 Get a token from JSON pointer string.
 @param ptr [in]  string that points to current token prefix `/`
            [out] string that points to next token prefix `/`, or string end
 @param end [in] end of the entire JSON Pointer string
 @param len [out] unescaped token length
 @param esc [out] number of escaped characters in this token
 @return head of the token, or NULL if syntax error
 */
static_inline const char *ptr_next_token(const char **ptr, const char *end,
                                         usize *len, usize *esc) {
    const char *hdr = *ptr + 1;
    const char *cur = hdr;
    /* skip unescaped characters */
    while (cur < end && *cur != '/' && *cur != '~') cur++;
    if (likely(cur == end || *cur != '~')) {
        /* no escaped characters, return */
        *ptr = cur;
        *len = (usize)(cur - hdr);
        *esc = 0;
        return hdr;
    } else {
        /* handle escaped characters */
        usize esc_num = 0;
        while (cur < end && *cur != '/') {
            if (*cur++ == '~') {
                if (cur == end || (*cur != '0' && *cur != '1')) {
                    *ptr = cur - 1;
                    return NULL;
                }
                esc_num++;
            }
        }
        *ptr = cur;
        *len = (usize)(cur - hdr) - esc_num;
        *esc = esc_num;
        return hdr;
    }
}

/**
 Convert token string to index.
 @param cur [in]  token head
 @param len [in]  token length
 @param idx [out] the index number, or USIZE_MAX if token is '-'
 @return true if token is a valid array index
 */
static_inline bool ptr_token_to_idx(const char *cur, usize len, usize *idx) {
    const char *end = cur + len;
    usize num = 0, add;
    if (unlikely(len == 0 || len > USIZE_SAFE_DIG)) return false;
    if (*cur == '0') {
        if (unlikely(len > 1)) return false;
        *idx = 0;
        return true;
    }
    if (*cur == '-') {
        if (unlikely(len > 1)) return false;
        *idx = USIZE_MAX;
        return true;
    }
    for (; cur < end && (add = (usize)((u8)*cur - (u8)'0')) <= 9; cur++) {
        num = num * 10 + add;
    }
    if (unlikely(num == 0 || cur < end)) return false;
    *idx = num;
    return true;
}

/**
 Compare JSON key with token.
 @param key a string key (yyjson_val or yyjson_mut_val)
 @param token a JSON pointer token
 @param len unescaped token length
 @param esc number of escaped characters in this token
 @return true if `str` is equals to `token`
 */
static_inline bool ptr_token_eq(void *key,
                                const char *token, usize len, usize esc) {
    yyjson_val *val = (yyjson_val *)key;
    if (unsafe_yyjson_get_len(val) != len) return false;
    if (likely(!esc)) {
        return memcmp(val->uni.str, token, len) == 0;
    } else {
        const char *str = val->uni.str;
        for (; len-- > 0; token++, str++) {
            if (*token == '~') {
                if (*str != (*++token == '0' ? '~' : '/')) return false;
            } else {
                if (*str != *token) return false;
            }
        }
        return true;
    }
}

/**
 Get a value from array by token.
 @param arr   an array, should not be NULL or non-array type
 @param token a JSON pointer token
 @param len   unescaped token length
 @param esc   number of escaped characters in this token
 @return value at index, or NULL if token is not index or index is out of range
 */
static_inline yyjson_val *ptr_arr_get(yyjson_val *arr, const char *token,
                                      usize len, usize esc) {
    yyjson_val *val = unsafe_yyjson_get_first(arr);
    usize num = unsafe_yyjson_get_len(arr), idx = 0;
    if (unlikely(num == 0)) return NULL;
    if (unlikely(!ptr_token_to_idx(token, len, &idx))) return NULL;
    if (unlikely(idx >= num)) return NULL;
    if (unsafe_yyjson_arr_is_flat(arr)) {
        return val + idx;
    } else {
        while (idx-- > 0) val = unsafe_yyjson_get_next(val);
        return val;
    }
}

/**
 Get a value from object by token.
 @param obj   [in] an object, should not be NULL or non-object type
 @param token [in] a JSON pointer token
 @param len   [in] unescaped token length
 @param esc   [in] number of escaped characters in this token
 @return value associated with the token, or NULL if no value
 */
static_inline yyjson_val *ptr_obj_get(yyjson_val *obj, const char *token,
                                      usize len, usize esc) {
    yyjson_val *key = unsafe_yyjson_get_first(obj);
    usize num = unsafe_yyjson_get_len(obj);
    if (unlikely(num == 0)) return NULL;
    for (; num > 0; num--, key = unsafe_yyjson_get_next(key + 1)) {
        if (ptr_token_eq(key, token, len, esc)) return key + 1;
    }
    return NULL;
}

/**
 Get a value from array by token.
 @param arr   [in] an array, should not be NULL or non-array type
 @param token [in] a JSON pointer token
 @param len   [in] unescaped token length
 @param esc   [in] number of escaped characters in this token
 @param pre   [out] previous (sibling) value of the returned value
 @param last  [out] whether index is last
 @return value at index, or NULL if token is not index or index is out of range
 */
static_inline yyjson_mut_val *ptr_mut_arr_get(yyjson_mut_val *arr,
                                              const char *token,
                                              usize len, usize esc,
                                              yyjson_mut_val **pre,
                                              bool *last) {
    yyjson_mut_val *val = (yyjson_mut_val *)arr->uni.ptr; /* last (tail) */
    usize num = unsafe_yyjson_get_len(arr), idx;
    if (last) *last = false;
    if (pre) *pre = NULL;
    if (unlikely(num == 0)) {
        if (last && len == 1 && (*token == '0' || *token == '-')) *last = true;
        return NULL;
    }
    if (unlikely(!ptr_token_to_idx(token, len, &idx))) return NULL;
    if (last) *last = (idx == num || idx == USIZE_MAX);
    if (unlikely(idx >= num)) return NULL;
    while (idx-- > 0) val = val->next;
    if (pre) *pre = val;
    return val->next;
}

/**
 Get a value from object by token.
 @param obj   [in] an object, should not be NULL or non-object type
 @param token [in] a JSON pointer token
 @param len   [in] unescaped token length
 @param esc   [in] number of escaped characters in this token
 @param pre   [out] previous (sibling) key of the returned value's key
 @return value associated with the token, or NULL if no value
 */
static_inline yyjson_mut_val *ptr_mut_obj_get(yyjson_mut_val *obj,
                                              const char *token,
                                              usize len, usize esc,
                                              yyjson_mut_val **pre) {
    yyjson_mut_val *pre_key = (yyjson_mut_val *)obj->uni.ptr, *key;
    usize num = unsafe_yyjson_get_len(obj);
    if (pre) *pre = NULL;
    if (unlikely(num == 0)) return NULL;
    for (; num > 0; num--, pre_key = key) {
        key = pre_key->next->next;
        if (ptr_token_eq(key, token, len, esc)) {
            if (pre) *pre = pre_key;
            return key->next;
        }
    }
    return NULL;
}

/**
 Create a string value with JSON pointer token.
 @param token [in] a JSON pointer token
 @param len   [in] unescaped token length
 @param esc   [in] number of escaped characters in this token
 @param doc   [in] used for memory allocation when creating value
 @return new string value, or NULL if memory allocation failed
 */
static_inline yyjson_mut_val *ptr_new_key(const char *token,
                                          usize len, usize esc,
                                          yyjson_mut_doc *doc) {
    const char *src = token;
    if (likely(!esc)) {
        return yyjson_mut_strncpy(doc, src, len);
    } else {
        const char *end = src + len + esc;
        char *dst = unsafe_yyjson_mut_str_alc(doc, len + esc);
        char *str = dst;
        if (unlikely(!dst)) return NULL;
        for (; src < end; src++, dst++) {
            if (*src != '~') *dst = *src;
            else *dst = (*++src == '0' ? '~' : '/');
        }
        *dst = '\0';
        return yyjson_mut_strn(doc, str, len);
    }
}

/* macros for yyjson_ptr */
#define return_err(_ret, _code, _pos, _msg) do { \
    if (err) { \
        err->code = YYJSON_PTR_ERR_##_code; \
        err->msg = _msg; \
        err->pos = (usize)(_pos); \
    } \
    return _ret; \
} while (false)

#define return_err_resolve(_ret, _pos) \
    return_err(_ret, RESOLVE, _pos, "JSON pointer cannot be resolved")
#define return_err_syntax(_ret, _pos) \
    return_err(_ret, SYNTAX, _pos, "invalid escaped character")
#define return_err_alloc(_ret) \
    return_err(_ret, MEMORY_ALLOCATION, 0, "failed to create value")

yyjson_val *unsafe_yyjson_ptr_getx(yyjson_val *val,
                                   const char *ptr, size_t ptr_len,
                                   yyjson_ptr_err *err) {

    const char *hdr = ptr, *end = ptr + ptr_len, *token;
    usize len, esc;
    yyjson_type type;

    while (true) {
        token = ptr_next_token(&ptr, end, &len, &esc);
        if (unlikely(!token)) return_err_syntax(NULL, ptr - hdr);
        type = unsafe_yyjson_get_type(val);
        if (type == YYJSON_TYPE_OBJ) {
            val = ptr_obj_get(val, token, len, esc);
        } else if (type == YYJSON_TYPE_ARR) {
            val = ptr_arr_get(val, token, len, esc);
        } else {
            val = NULL;
        }
        if (!val) return_err_resolve(NULL, token - hdr);
        if (ptr == end) return val;
    }
}

yyjson_mut_val *unsafe_yyjson_mut_ptr_getx(
    yyjson_mut_val *val, const char *ptr, size_t ptr_len,
    yyjson_ptr_ctx *ctx, yyjson_ptr_err *err) {

    const char *hdr = ptr, *end = ptr + ptr_len, *token;
    usize len, esc;
    yyjson_mut_val *ctn, *pre = NULL;
    yyjson_type type;
    bool idx_is_last = false;

    while (true) {
        token = ptr_next_token(&ptr, end, &len, &esc);
        if (unlikely(!token)) return_err_syntax(NULL, ptr - hdr);
        ctn = val;
        type = unsafe_yyjson_get_type(val);
        if (type == YYJSON_TYPE_OBJ) {
            val = ptr_mut_obj_get(val, token, len, esc, &pre);
        } else if (type == YYJSON_TYPE_ARR) {
            val = ptr_mut_arr_get(val, token, len, esc, &pre, &idx_is_last);
        } else {
            val = NULL;
        }
        if (ctx && (ptr == end)) {
            if (type == YYJSON_TYPE_OBJ ||
                (type == YYJSON_TYPE_ARR && (val || idx_is_last))) {
                ctx->ctn = ctn;
                ctx->pre = pre;
            }
        }
        if (!val) return_err_resolve(NULL, token - hdr);
        if (ptr == end) return val;
    }
}

bool unsafe_yyjson_mut_ptr_putx(
    yyjson_mut_val *val, const char *ptr, size_t ptr_len,
    yyjson_mut_val *new_val, yyjson_mut_doc *doc, bool create_parent,
    bool insert_new, yyjson_ptr_ctx *ctx, yyjson_ptr_err *err) {

    const char *hdr = ptr, *end = ptr + ptr_len, *token;
    usize token_len, esc, ctn_len;
    yyjson_mut_val *ctn, *key, *pre = NULL;
    yyjson_mut_val *sep_ctn = NULL, *sep_key = NULL, *sep_val = NULL;
    yyjson_type ctn_type;
    bool idx_is_last = false;

    /* skip exist parent nodes */
    while (true) {
        token = ptr_next_token(&ptr, end, &token_len, &esc);
        if (unlikely(!token)) return_err_syntax(false, ptr - hdr);
        ctn = val;
        ctn_type = unsafe_yyjson_get_type(ctn);
        if (ctn_type == YYJSON_TYPE_OBJ) {
            val = ptr_mut_obj_get(ctn, token, token_len, esc, &pre);
        } else if (ctn_type == YYJSON_TYPE_ARR) {
            val = ptr_mut_arr_get(ctn, token, token_len, esc, &pre,
                                  &idx_is_last);
        } else return_err_resolve(false, token - hdr);
        if (!val) break;
        if (ptr == end) break; /* is last token */
    }

    /* create parent nodes if not exist */
    if (unlikely(ptr != end)) { /* not last token */
        if (!create_parent) return_err_resolve(false, token - hdr);

        /* add value at last index if container is array */
        if (ctn_type == YYJSON_TYPE_ARR) {
            if (!idx_is_last || !insert_new) {
                return_err_resolve(false, token - hdr);
            }
            val = yyjson_mut_obj(doc);
            if (!val) return_err_alloc(false);

            /* delay attaching until all operations are completed */
            sep_ctn = ctn;
            sep_key = NULL;
            sep_val = val;

            /* move to next token */
            ctn = val;
            val = NULL;
            ctn_type = YYJSON_TYPE_OBJ;
            token = ptr_next_token(&ptr, end, &token_len, &esc);
            if (unlikely(!token)) return_err_resolve(false, token - hdr);
        }

        /* container is object, create parent nodes */
        while (ptr != end) { /* not last token */
            key = ptr_new_key(token, token_len, esc, doc);
            if (!key) return_err_alloc(false);
            val = yyjson_mut_obj(doc);
            if (!val) return_err_alloc(false);

            /* delay attaching until all operations are completed */
            if (!sep_ctn) {
                sep_ctn = ctn;
                sep_key = key;
                sep_val = val;
            } else {
                yyjson_mut_obj_add(ctn, key, val);
            }

            /* move to next token */
            ctn = val;
            val = NULL;
            token = ptr_next_token(&ptr, end, &token_len, &esc);
            if (unlikely(!token)) return_err_syntax(false, ptr - hdr);
        }
    }

    /* JSON pointer is resolved, insert or replace target value */
    ctn_len = unsafe_yyjson_get_len(ctn);
    if (ctn_type == YYJSON_TYPE_OBJ) {
        if (ctx) ctx->ctn = ctn;
        if (!val || insert_new) {
            /* insert new key-value pair */
            key = ptr_new_key(token, token_len, esc, doc);
            if (unlikely(!key)) return_err_alloc(false);
            if (ctx) ctx->pre = ctn_len ? (yyjson_mut_val *)ctn->uni.ptr : key;
            unsafe_yyjson_mut_obj_add(ctn, key, new_val, ctn_len);
        } else {
            /* replace exist value */
            key = pre->next->next;
            if (ctx) ctx->pre = pre;
            if (ctx) ctx->old = val;
            yyjson_mut_obj_put(ctn, key, new_val);
        }
    } else {
        /* array */
        if (ctx && (val || idx_is_last)) ctx->ctn = ctn;
        if (insert_new) {
            /* append new value */
            if (val) {
                pre->next = new_val;
                new_val->next = val;
                if (ctx) ctx->pre = pre;
                unsafe_yyjson_set_len(ctn, ctn_len + 1);
            } else if (idx_is_last) {
                if (ctx) ctx->pre = ctn_len ?
                    (yyjson_mut_val *)ctn->uni.ptr : new_val;
                yyjson_mut_arr_append(ctn, new_val);
            } else {
                return_err_resolve(false, token - hdr);
            }
        } else {
            /* replace exist value */
            if (!val) return_err_resolve(false, token - hdr);
            if (ctn_len > 1) {
                new_val->next = val->next;
                pre->next = new_val;
                if (ctn->uni.ptr == val) ctn->uni.ptr = new_val;
            } else {
                new_val->next = new_val;
                ctn->uni.ptr = new_val;
                pre = new_val;
            }
            if (ctx) ctx->pre = pre;
            if (ctx) ctx->old = val;
        }
    }

    /* all operations are completed, attach the new components to the target */
    if (unlikely(sep_ctn)) {
        if (sep_key) yyjson_mut_obj_add(sep_ctn, sep_key, sep_val);
        else yyjson_mut_arr_append(sep_ctn, sep_val);
    }
    return true;
}

yyjson_mut_val *unsafe_yyjson_mut_ptr_replacex(
    yyjson_mut_val *val, const char *ptr, size_t len, yyjson_mut_val *new_val,
    yyjson_ptr_ctx *ctx, yyjson_ptr_err *err) {

    yyjson_mut_val *cur_val;
    yyjson_ptr_ctx cur_ctx;
    memset(&cur_ctx, 0, sizeof(cur_ctx));
    if (!ctx) ctx = &cur_ctx;
    cur_val = unsafe_yyjson_mut_ptr_getx(val, ptr, len, ctx, err);
    if (!cur_val) return NULL;

    if (yyjson_mut_is_obj(ctx->ctn)) {
        yyjson_mut_val *key = ctx->pre->next->next;
        yyjson_mut_obj_put(ctx->ctn, key, new_val);
    } else {
        yyjson_ptr_ctx_replace(ctx, new_val);
    }
    ctx->old = cur_val;
    return cur_val;
}

yyjson_mut_val *unsafe_yyjson_mut_ptr_removex(
    yyjson_mut_val *val, const char *ptr, size_t len,
    yyjson_ptr_ctx *ctx, yyjson_ptr_err *err) {

    yyjson_mut_val *cur_val;
    yyjson_ptr_ctx cur_ctx;
    memset(&cur_ctx, 0, sizeof(cur_ctx));
    if (!ctx) ctx = &cur_ctx;
    cur_val = unsafe_yyjson_mut_ptr_getx(val, ptr, len, ctx, err);
    if (cur_val) {
        if (yyjson_mut_is_obj(ctx->ctn)) {
            yyjson_mut_val *key = ctx->pre->next->next;
            yyjson_mut_obj_put(ctx->ctn, key, NULL);
        } else {
            yyjson_ptr_ctx_remove(ctx);
        }
        ctx->pre = NULL;
        ctx->old = cur_val;
    }
    return cur_val;
}

/* macros for yyjson_ptr */
#undef return_err
#undef return_err_resolve
#undef return_err_syntax
#undef return_err_alloc



/*==============================================================================
 * JSON Patch API (RFC 6902)
 *============================================================================*/

/* JSON Patch operation */
typedef enum patch_op {
    PATCH_OP_ADD,       /* path, value */
    PATCH_OP_REMOVE,    /* path */
    PATCH_OP_REPLACE,   /* path, value */
    PATCH_OP_MOVE,      /* from, path */
    PATCH_OP_COPY,      /* from, path */
    PATCH_OP_TEST,      /* path, value */
    PATCH_OP_NONE       /* invalid */
} patch_op;

static patch_op patch_op_get(yyjson_val *op) {
    const char *str = op->uni.str;
    switch (unsafe_yyjson_get_len(op)) {
        case 3:
            if (!memcmp(str, "add", 3)) return PATCH_OP_ADD;
            return PATCH_OP_NONE;
        case 4:
            if (!memcmp(str, "move", 4)) return PATCH_OP_MOVE;
            if (!memcmp(str, "copy", 4)) return PATCH_OP_COPY;
            if (!memcmp(str, "test", 4)) return PATCH_OP_TEST;
            return PATCH_OP_NONE;
        case 6:
            if (!memcmp(str, "remove", 6)) return PATCH_OP_REMOVE;
            return PATCH_OP_NONE;
        case 7:
            if (!memcmp(str, "replace", 7)) return PATCH_OP_REPLACE;
            return PATCH_OP_NONE;
        default:
            return PATCH_OP_NONE;
    }
}

/* macros for yyjson_patch */
#define return_err(_code, _msg) do { \
    if (err->ptr.code == YYJSON_PTR_ERR_MEMORY_ALLOCATION) { \
        err->code = YYJSON_PATCH_ERROR_MEMORY_ALLOCATION; \
        err->msg = _msg; \
        memset(&err->ptr, 0, sizeof(yyjson_ptr_err)); \
    } else { \
        err->code = YYJSON_PATCH_ERROR_##_code; \
        err->msg = _msg; \
        err->idx = iter.idx ? iter.idx - 1 : 0; \
    } \
    return NULL; \
} while (false)

#define return_err_copy() \
    return_err(MEMORY_ALLOCATION, "failed to copy value")
#define return_err_key(_key) \
    return_err(MISSING_KEY, "missing key " _key)
#define return_err_val(_key) \
    return_err(INVALID_MEMBER, "invalid member " _key)

#define ptr_get(_ptr) yyjson_mut_ptr_getx( \
    root, _ptr->uni.str, _ptr##_len, NULL, &err->ptr)
#define ptr_add(_ptr, _val) yyjson_mut_ptr_addx( \
    root, _ptr->uni.str, _ptr##_len, _val, doc, false, NULL, &err->ptr)
#define ptr_remove(_ptr) yyjson_mut_ptr_removex( \
    root, _ptr->uni.str, _ptr##_len, NULL, &err->ptr)
#define ptr_replace(_ptr, _val)yyjson_mut_ptr_replacex( \
    root, _ptr->uni.str, _ptr##_len, _val, NULL, &err->ptr)

yyjson_mut_val *yyjson_patch(yyjson_mut_doc *doc,
                             yyjson_val *orig,
                             yyjson_val *patch,
                             yyjson_patch_err *err) {

    yyjson_mut_val *root;
    yyjson_val *obj;
    yyjson_arr_iter iter;
    yyjson_patch_err err_tmp;
    if (!err) err = &err_tmp;
    memset(err, 0, sizeof(*err));
    memset(&iter, 0, sizeof(iter));

    if (unlikely(!doc || !orig || !patch)) {
        return_err(INVALID_PARAMETER, "input parameter is NULL");
    }
    if (unlikely(!yyjson_is_arr(patch))) {
        return_err(INVALID_PARAMETER, "input patch is not array");
    }
    root = yyjson_val_mut_copy(doc, orig);
    if (unlikely(!root)) return_err_copy();

    /* iterate through the patch array */
    yyjson_arr_iter_init(patch, &iter);
    while ((obj = yyjson_arr_iter_next(&iter))) {
        patch_op op_enum;
        yyjson_val *op, *path, *from = NULL, *value;
        yyjson_mut_val *val = NULL, *test;
        usize path_len, from_len = 0;
        if (unlikely(!unsafe_yyjson_is_obj(obj))) {
            return_err(INVALID_OPERATION, "JSON patch operation is not object");
        }

        /* get required member: op */
        op = yyjson_obj_get(obj, "op");
        if (unlikely(!op)) return_err_key("`op`");
        if (unlikely(!yyjson_is_str(op))) return_err_val("`op`");
        op_enum = patch_op_get(op);

        /* get required member: path */
        path = yyjson_obj_get(obj, "path");
        if (unlikely(!path)) return_err_key("`path`");
        if (unlikely(!yyjson_is_str(path))) return_err_val("`path`");
        path_len = unsafe_yyjson_get_len(path);

        /* get required member: value, from */
        switch ((int)op_enum) {
            case PATCH_OP_ADD: case PATCH_OP_REPLACE: case PATCH_OP_TEST:
                value = yyjson_obj_get(obj, "value");
                if (unlikely(!value)) return_err_key("`value`");
                val = yyjson_val_mut_copy(doc, value);
                if (unlikely(!val)) return_err_copy();
                break;
            case PATCH_OP_MOVE: case PATCH_OP_COPY:
                from = yyjson_obj_get(obj, "from");
                if (unlikely(!from)) return_err_key("`from`");
                if (unlikely(!yyjson_is_str(from))) return_err_val("`from`");
                from_len = unsafe_yyjson_get_len(from);
                break;
            default:
                break;
        }

        /* perform an operation */
        switch ((int)op_enum) {
            case PATCH_OP_ADD: /* add(path, val) */
                if (unlikely(path_len == 0)) { root = val; break; }
                if (unlikely(!ptr_add(path, val))) {
                    return_err(POINTER, "failed to add `path`");
                }
                break;
            case PATCH_OP_REMOVE: /* remove(path) */
                if (unlikely(!ptr_remove(path))) {
                    return_err(POINTER, "failed to remove `path`");
                }
                break;
            case PATCH_OP_REPLACE: /* replace(path, val) */
                if (unlikely(path_len == 0)) { root = val; break; }
                if (unlikely(!ptr_replace(path, val))) {
                    return_err(POINTER, "failed to replace `path`");
                }
                break;
            case PATCH_OP_MOVE: /* val = remove(from), add(path, val) */
                if (unlikely(from_len == 0 && path_len == 0)) break;
                val = ptr_remove(from);
                if (unlikely(!val)) {
                    return_err(POINTER, "failed to remove `from`");
                }
                if (unlikely(path_len == 0)) { root = val; break; }
                if (unlikely(!ptr_add(path, val))) {
                    return_err(POINTER, "failed to add `path`");
                }
                break;
            case PATCH_OP_COPY: /* val = get(from).copy, add(path, val) */
                val = ptr_get(from);
                if (unlikely(!val)) {
                    return_err(POINTER, "failed to get `from`");
                }
                if (unlikely(path_len == 0)) { root = val; break; }
                val = yyjson_mut_val_mut_copy(doc, val);
                if (unlikely(!val)) return_err_copy();
                if (unlikely(!ptr_add(path, val))) {
                    return_err(POINTER, "failed to add `path`");
                }
                break;
            case PATCH_OP_TEST: /* test = get(path), test.eq(val) */
                test = ptr_get(path);
                if (unlikely(!test)) {
                    return_err(POINTER, "failed to get `path`");
                }
                if (unlikely(!yyjson_mut_equals(val, test))) {
                    return_err(EQUAL, "failed to test equal");
                }
                break;
            default:
                return_err(INVALID_MEMBER, "unsupported `op`");
        }
    }
    return root;
}

yyjson_mut_val *yyjson_mut_patch(yyjson_mut_doc *doc,
                                 yyjson_mut_val *orig,
                                 yyjson_mut_val *patch,
                                 yyjson_patch_err *err) {
    yyjson_mut_val *root, *obj;
    yyjson_mut_arr_iter iter;
    yyjson_patch_err err_tmp;
    if (!err) err = &err_tmp;
    memset(err, 0, sizeof(*err));
    memset(&iter, 0, sizeof(iter));

    if (unlikely(!doc || !orig || !patch)) {
        return_err(INVALID_PARAMETER, "input parameter is NULL");
    }
    if (unlikely(!yyjson_mut_is_arr(patch))) {
        return_err(INVALID_PARAMETER, "input patch is not array");
    }
    root = yyjson_mut_val_mut_copy(doc, orig);
    if (unlikely(!root)) return_err_copy();

    /* iterate through the patch array */
    yyjson_mut_arr_iter_init(patch, &iter);
    while ((obj = yyjson_mut_arr_iter_next(&iter))) {
        patch_op op_enum;
        yyjson_mut_val *op, *path, *from = NULL, *value;
        yyjson_mut_val *val = NULL, *test;
        usize path_len, from_len = 0;
        if (!unsafe_yyjson_is_obj(obj)) {
            return_err(INVALID_OPERATION, "JSON patch operation is not object");
        }

        /* get required member: op */
        op = yyjson_mut_obj_get(obj, "op");
        if (unlikely(!op)) return_err_key("`op`");
        if (unlikely(!yyjson_mut_is_str(op))) return_err_val("`op`");
        op_enum = patch_op_get((yyjson_val *)(void *)op);

        /* get required member: path */
        path = yyjson_mut_obj_get(obj, "path");
        if (unlikely(!path)) return_err_key("`path`");
        if (unlikely(!yyjson_mut_is_str(path))) return_err_val("`path`");
        path_len = unsafe_yyjson_get_len(path);

        /* get required member: value, from */
        switch ((int)op_enum) {
            case PATCH_OP_ADD: case PATCH_OP_REPLACE: case PATCH_OP_TEST:
                value = yyjson_mut_obj_get(obj, "value");
                if (unlikely(!value)) return_err_key("`value`");
                val = yyjson_mut_val_mut_copy(doc, value);
                if (unlikely(!val)) return_err_copy();
                break;
            case PATCH_OP_MOVE: case PATCH_OP_COPY:
                from = yyjson_mut_obj_get(obj, "from");
                if (unlikely(!from)) return_err_key("`from`");
                if (unlikely(!yyjson_mut_is_str(from))) {
                    return_err_val("`from`");
                }
                from_len = unsafe_yyjson_get_len(from);
                break;
            default:
                break;
        }

        /* perform an operation */
        switch ((int)op_enum) {
            case PATCH_OP_ADD: /* add(path, val) */
                if (unlikely(path_len == 0)) { root = val; break; }
                if (unlikely(!ptr_add(path, val))) {
                    return_err(POINTER, "failed to add `path`");
                }
                break;
            case PATCH_OP_REMOVE: /* remove(path) */
                if (unlikely(!ptr_remove(path))) {
                    return_err(POINTER, "failed to remove `path`");
                }
                break;
            case PATCH_OP_REPLACE: /* replace(path, val) */
                if (unlikely(path_len == 0)) { root = val; break; }
                if (unlikely(!ptr_replace(path, val))) {
                    return_err(POINTER, "failed to replace `path`");
                }
                break;
            case PATCH_OP_MOVE: /* val = remove(from), add(path, val) */
                if (unlikely(from_len == 0 && path_len == 0)) break;
                val = ptr_remove(from);
                if (unlikely(!val)) {
                    return_err(POINTER, "failed to remove `from`");
                }
                if (unlikely(path_len == 0)) { root = val; break; }
                if (unlikely(!ptr_add(path, val))) {
                    return_err(POINTER, "failed to add `path`");
                }
                break;
            case PATCH_OP_COPY: /* val = get(from).copy, add(path, val) */
                val = ptr_get(from);
                if (unlikely(!val)) {
                    return_err(POINTER, "failed to get `from`");
                }
                if (unlikely(path_len == 0)) { root = val; break; }
                val = yyjson_mut_val_mut_copy(doc, val);
                if (unlikely(!val)) return_err_copy();
                if (unlikely(!ptr_add(path, val))) {
                    return_err(POINTER, "failed to add `path`");
                }
                break;
            case PATCH_OP_TEST: /* test = get(path), test.eq(val) */
                test = ptr_get(path);
                if (unlikely(!test)) {
                    return_err(POINTER, "failed to get `path`");
                }
                if (unlikely(!yyjson_mut_equals(val, test))) {
                    return_err(EQUAL, "failed to test equal");
                }
                break;
            default:
                return_err(INVALID_MEMBER, "unsupported `op`");
        }
    }
    return root;
}

/* macros for yyjson_patch */
#undef return_err
#undef return_err_copy
#undef return_err_key
#undef return_err_val
#undef ptr_get
#undef ptr_add
#undef ptr_remove
#undef ptr_replace



/*==============================================================================
 * JSON Merge-Patch API (RFC 7386)
 *============================================================================*/

yyjson_mut_val *yyjson_merge_patch(yyjson_mut_doc *doc,
                                   yyjson_val *orig,
                                   yyjson_val *patch) {
    usize idx, max;
    yyjson_val *key, *orig_val, *patch_val, local_orig;
    yyjson_mut_val *builder, *mut_key, *mut_val, *merged_val;

    if (unlikely(!yyjson_is_obj(patch))) {
        return yyjson_val_mut_copy(doc, patch);
    }

    builder = yyjson_mut_obj(doc);
    if (unlikely(!builder)) return NULL;

    memset(&local_orig, 0, sizeof(local_orig));
    if (!yyjson_is_obj(orig)) {
        orig = &local_orig;
        orig->tag = builder->tag;
        orig->uni = builder->uni;
    }

    /* If orig is contributing, copy any items not modified by the patch */
    if (orig != &local_orig) {
        yyjson_obj_foreach(orig, idx, max, key, orig_val) {
            patch_val = yyjson_obj_getn(patch,
                                        unsafe_yyjson_get_str(key),
                                        unsafe_yyjson_get_len(key));
            if (!patch_val) {
                mut_key = yyjson_val_mut_copy(doc, key);
                mut_val = yyjson_val_mut_copy(doc, orig_val);
                if (!yyjson_mut_obj_add(builder, mut_key, mut_val)) return NULL;
            }
        }
    }

    /* Merge items modified by the patch. */
    yyjson_obj_foreach(patch, idx, max, key, patch_val) {
        /* null indicates the field is removed. */
        if (unsafe_yyjson_is_null(patch_val)) {
            continue;
        }
        mut_key = yyjson_val_mut_copy(doc, key);
        orig_val = yyjson_obj_getn(orig,
                                   unsafe_yyjson_get_str(key),
                                   unsafe_yyjson_get_len(key));
        merged_val = yyjson_merge_patch(doc, orig_val, patch_val);
        if (!yyjson_mut_obj_add(builder, mut_key, merged_val)) return NULL;
    }

    return builder;
}

yyjson_mut_val *yyjson_mut_merge_patch(yyjson_mut_doc *doc,
                                       yyjson_mut_val *orig,
                                       yyjson_mut_val *patch) {
    usize idx, max;
    yyjson_mut_val *key, *orig_val, *patch_val, local_orig;
    yyjson_mut_val *builder, *mut_key, *mut_val, *merged_val;

    if (unlikely(!yyjson_mut_is_obj(patch))) {
        return yyjson_mut_val_mut_copy(doc, patch);
    }

    builder = yyjson_mut_obj(doc);
    if (unlikely(!builder)) return NULL;

    memset(&local_orig, 0, sizeof(local_orig));
    if (!yyjson_mut_is_obj(orig)) {
        orig = &local_orig;
        orig->tag = builder->tag;
        orig->uni = builder->uni;
    }

    /* If orig is contributing, copy any items not modified by the patch */
    if (orig != &local_orig) {
        yyjson_mut_obj_foreach(orig, idx, max, key, orig_val) {
            patch_val = yyjson_mut_obj_getn(patch,
                                            unsafe_yyjson_get_str(key),
                                            unsafe_yyjson_get_len(key));
            if (!patch_val) {
                mut_key = yyjson_mut_val_mut_copy(doc, key);
                mut_val = yyjson_mut_val_mut_copy(doc, orig_val);
                if (!yyjson_mut_obj_add(builder, mut_key, mut_val)) return NULL;
            }
        }
    }

    /* Merge items modified by the patch. */
    yyjson_mut_obj_foreach(patch, idx, max, key, patch_val) {
        /* null indicates the field is removed. */
        if (unsafe_yyjson_is_null(patch_val)) {
            continue;
        }
        mut_key = yyjson_mut_val_mut_copy(doc, key);
        orig_val = yyjson_mut_obj_getn(orig,
                                       unsafe_yyjson_get_str(key),
                                       unsafe_yyjson_get_len(key));
        merged_val = yyjson_mut_merge_patch(doc, orig_val, patch_val);
        if (!yyjson_mut_obj_add(builder, mut_key, merged_val)) return NULL;
    }

    return builder;
}

#endif /* YYJSON_DISABLE_UTILS */



/*==============================================================================
 * Power10 Lookup Table
 * These data are used by the floating-point number reader and writer.
 *============================================================================*/

#if (!YYJSON_DISABLE_READER || !YYJSON_DISABLE_WRITER) && \
    (!YYJSON_DISABLE_FAST_FP_CONV)

/** Minimum decimal exponent in pow10_sig_table. */
#define POW10_SIG_TABLE_MIN_EXP -343

/** Maximum decimal exponent in pow10_sig_table. */
#define POW10_SIG_TABLE_MAX_EXP 324

/** Minimum exact decimal exponent in pow10_sig_table */
#define POW10_SIG_TABLE_MIN_EXACT_EXP 0

/** Maximum exact decimal exponent in pow10_sig_table */
#define POW10_SIG_TABLE_MAX_EXACT_EXP 55

/** Normalized significant 128 bits of pow10, no rounded up (size: 10.4KB).
    This lookup table is used by both the double number reader and writer.
    (generate with misc/make_tables.c) */
static const u64 pow10_sig_table[] = {
    U64(0xBF29DCAB, 0xA82FDEAE), U64(0x7432EE87, 0x3880FC33), /* ~= 10^-343 */
    U64(0xEEF453D6, 0x923BD65A), U64(0x113FAA29, 0x06A13B3F), /* ~= 10^-342 */
    U64(0x9558B466, 0x1B6565F8), U64(0x4AC7CA59, 0xA424C507), /* ~= 10^-341 */
    U64(0xBAAEE17F, 0xA23EBF76), U64(0x5D79BCF0, 0x0D2DF649), /* ~= 10^-340 */
    U64(0xE95A99DF, 0x8ACE6F53), U64(0xF4D82C2C, 0x107973DC), /* ~= 10^-339 */
    U64(0x91D8A02B, 0xB6C10594), U64(0x79071B9B, 0x8A4BE869), /* ~= 10^-338 */
    U64(0xB64EC836, 0xA47146F9), U64(0x9748E282, 0x6CDEE284), /* ~= 10^-337 */
    U64(0xE3E27A44, 0x4D8D98B7), U64(0xFD1B1B23, 0x08169B25), /* ~= 10^-336 */
    U64(0x8E6D8C6A, 0xB0787F72), U64(0xFE30F0F5, 0xE50E20F7), /* ~= 10^-335 */
    U64(0xB208EF85, 0x5C969F4F), U64(0xBDBD2D33, 0x5E51A935), /* ~= 10^-334 */
    U64(0xDE8B2B66, 0xB3BC4723), U64(0xAD2C7880, 0x35E61382), /* ~= 10^-333 */
    U64(0x8B16FB20, 0x3055AC76), U64(0x4C3BCB50, 0x21AFCC31), /* ~= 10^-332 */
    U64(0xADDCB9E8, 0x3C6B1793), U64(0xDF4ABE24, 0x2A1BBF3D), /* ~= 10^-331 */
    U64(0xD953E862, 0x4B85DD78), U64(0xD71D6DAD, 0x34A2AF0D), /* ~= 10^-330 */
    U64(0x87D4713D, 0x6F33AA6B), U64(0x8672648C, 0x40E5AD68), /* ~= 10^-329 */
    U64(0xA9C98D8C, 0xCB009506), U64(0x680EFDAF, 0x511F18C2), /* ~= 10^-328 */
    U64(0xD43BF0EF, 0xFDC0BA48), U64(0x0212BD1B, 0x2566DEF2), /* ~= 10^-327 */
    U64(0x84A57695, 0xFE98746D), U64(0x014BB630, 0xF7604B57), /* ~= 10^-326 */
    U64(0xA5CED43B, 0x7E3E9188), U64(0x419EA3BD, 0x35385E2D), /* ~= 10^-325 */
    U64(0xCF42894A, 0x5DCE35EA), U64(0x52064CAC, 0x828675B9), /* ~= 10^-324 */
    U64(0x818995CE, 0x7AA0E1B2), U64(0x7343EFEB, 0xD1940993), /* ~= 10^-323 */
    U64(0xA1EBFB42, 0x19491A1F), U64(0x1014EBE6, 0xC5F90BF8), /* ~= 10^-322 */
    U64(0xCA66FA12, 0x9F9B60A6), U64(0xD41A26E0, 0x77774EF6), /* ~= 10^-321 */
    U64(0xFD00B897, 0x478238D0), U64(0x8920B098, 0x955522B4), /* ~= 10^-320 */
    U64(0x9E20735E, 0x8CB16382), U64(0x55B46E5F, 0x5D5535B0), /* ~= 10^-319 */
    U64(0xC5A89036, 0x2FDDBC62), U64(0xEB2189F7, 0x34AA831D), /* ~= 10^-318 */
    U64(0xF712B443, 0xBBD52B7B), U64(0xA5E9EC75, 0x01D523E4), /* ~= 10^-317 */
    U64(0x9A6BB0AA, 0x55653B2D), U64(0x47B233C9, 0x2125366E), /* ~= 10^-316 */
    U64(0xC1069CD4, 0xEABE89F8), U64(0x999EC0BB, 0x696E840A), /* ~= 10^-315 */
    U64(0xF148440A, 0x256E2C76), U64(0xC00670EA, 0x43CA250D), /* ~= 10^-314 */
    U64(0x96CD2A86, 0x5764DBCA), U64(0x38040692, 0x6A5E5728), /* ~= 10^-313 */
    U64(0xBC807527, 0xED3E12BC), U64(0xC6050837, 0x04F5ECF2), /* ~= 10^-312 */
    U64(0xEBA09271, 0xE88D976B), U64(0xF7864A44, 0xC633682E), /* ~= 10^-311 */
    U64(0x93445B87, 0x31587EA3), U64(0x7AB3EE6A, 0xFBE0211D), /* ~= 10^-310 */
    U64(0xB8157268, 0xFDAE9E4C), U64(0x5960EA05, 0xBAD82964), /* ~= 10^-309 */
    U64(0xE61ACF03, 0x3D1A45DF), U64(0x6FB92487, 0x298E33BD), /* ~= 10^-308 */
    U64(0x8FD0C162, 0x06306BAB), U64(0xA5D3B6D4, 0x79F8E056), /* ~= 10^-307 */
    U64(0xB3C4F1BA, 0x87BC8696), U64(0x8F48A489, 0x9877186C), /* ~= 10^-306 */
    U64(0xE0B62E29, 0x29ABA83C), U64(0x331ACDAB, 0xFE94DE87), /* ~= 10^-305 */
    U64(0x8C71DCD9, 0xBA0B4925), U64(0x9FF0C08B, 0x7F1D0B14), /* ~= 10^-304 */
    U64(0xAF8E5410, 0x288E1B6F), U64(0x07ECF0AE, 0x5EE44DD9), /* ~= 10^-303 */
    U64(0xDB71E914, 0x32B1A24A), U64(0xC9E82CD9, 0xF69D6150), /* ~= 10^-302 */
    U64(0x892731AC, 0x9FAF056E), U64(0xBE311C08, 0x3A225CD2), /* ~= 10^-301 */
    U64(0xAB70FE17, 0xC79AC6CA), U64(0x6DBD630A, 0x48AAF406), /* ~= 10^-300 */
    U64(0xD64D3D9D, 0xB981787D), U64(0x092CBBCC, 0xDAD5B108), /* ~= 10^-299 */
    U64(0x85F04682, 0x93F0EB4E), U64(0x25BBF560, 0x08C58EA5), /* ~= 10^-298 */
    U64(0xA76C5823, 0x38ED2621), U64(0xAF2AF2B8, 0x0AF6F24E), /* ~= 10^-297 */
    U64(0xD1476E2C, 0x07286FAA), U64(0x1AF5AF66, 0x0DB4AEE1), /* ~= 10^-296 */
    U64(0x82CCA4DB, 0x847945CA), U64(0x50D98D9F, 0xC890ED4D), /* ~= 10^-295 */
    U64(0xA37FCE12, 0x6597973C), U64(0xE50FF107, 0xBAB528A0), /* ~= 10^-294 */
    U64(0xCC5FC196, 0xFEFD7D0C), U64(0x1E53ED49, 0xA96272C8), /* ~= 10^-293 */
    U64(0xFF77B1FC, 0xBEBCDC4F), U64(0x25E8E89C, 0x13BB0F7A), /* ~= 10^-292 */
    U64(0x9FAACF3D, 0xF73609B1), U64(0x77B19161, 0x8C54E9AC), /* ~= 10^-291 */
    U64(0xC795830D, 0x75038C1D), U64(0xD59DF5B9, 0xEF6A2417), /* ~= 10^-290 */
    U64(0xF97AE3D0, 0xD2446F25), U64(0x4B057328, 0x6B44AD1D), /* ~= 10^-289 */
    U64(0x9BECCE62, 0x836AC577), U64(0x4EE367F9, 0x430AEC32), /* ~= 10^-288 */
    U64(0xC2E801FB, 0x244576D5), U64(0x229C41F7, 0x93CDA73F), /* ~= 10^-287 */
    U64(0xF3A20279, 0xED56D48A), U64(0x6B435275, 0x78C1110F), /* ~= 10^-286 */
    U64(0x9845418C, 0x345644D6), U64(0x830A1389, 0x6B78AAA9), /* ~= 10^-285 */
    U64(0xBE5691EF, 0x416BD60C), U64(0x23CC986B, 0xC656D553), /* ~= 10^-284 */
    U64(0xEDEC366B, 0x11C6CB8F), U64(0x2CBFBE86, 0xB7EC8AA8), /* ~= 10^-283 */
    U64(0x94B3A202, 0xEB1C3F39), U64(0x7BF7D714, 0x32F3D6A9), /* ~= 10^-282 */
    U64(0xB9E08A83, 0xA5E34F07), U64(0xDAF5CCD9, 0x3FB0CC53), /* ~= 10^-281 */
    U64(0xE858AD24, 0x8F5C22C9), U64(0xD1B3400F, 0x8F9CFF68), /* ~= 10^-280 */
    U64(0x91376C36, 0xD99995BE), U64(0x23100809, 0xB9C21FA1), /* ~= 10^-279 */
    U64(0xB5854744, 0x8FFFFB2D), U64(0xABD40A0C, 0x2832A78A), /* ~= 10^-278 */
    U64(0xE2E69915, 0xB3FFF9F9), U64(0x16C90C8F, 0x323F516C), /* ~= 10^-277 */
    U64(0x8DD01FAD, 0x907FFC3B), U64(0xAE3DA7D9, 0x7F6792E3), /* ~= 10^-276 */
    U64(0xB1442798, 0xF49FFB4A), U64(0x99CD11CF, 0xDF41779C), /* ~= 10^-275 */
    U64(0xDD95317F, 0x31C7FA1D), U64(0x40405643, 0xD711D583), /* ~= 10^-274 */
    U64(0x8A7D3EEF, 0x7F1CFC52), U64(0x482835EA, 0x666B2572), /* ~= 10^-273 */
    U64(0xAD1C8EAB, 0x5EE43B66), U64(0xDA324365, 0x0005EECF), /* ~= 10^-272 */
    U64(0xD863B256, 0x369D4A40), U64(0x90BED43E, 0x40076A82), /* ~= 10^-271 */
    U64(0x873E4F75, 0xE2224E68), U64(0x5A7744A6, 0xE804A291), /* ~= 10^-270 */
    U64(0xA90DE353, 0x5AAAE202), U64(0x711515D0, 0xA205CB36), /* ~= 10^-269 */
    U64(0xD3515C28, 0x31559A83), U64(0x0D5A5B44, 0xCA873E03), /* ~= 10^-268 */
    U64(0x8412D999, 0x1ED58091), U64(0xE858790A, 0xFE9486C2), /* ~= 10^-267 */
    U64(0xA5178FFF, 0x668AE0B6), U64(0x626E974D, 0xBE39A872), /* ~= 10^-266 */
    U64(0xCE5D73FF, 0x402D98E3), U64(0xFB0A3D21, 0x2DC8128F), /* ~= 10^-265 */
    U64(0x80FA687F, 0x881C7F8E), U64(0x7CE66634, 0xBC9D0B99), /* ~= 10^-264 */
    U64(0xA139029F, 0x6A239F72), U64(0x1C1FFFC1, 0xEBC44E80), /* ~= 10^-263 */
    U64(0xC9874347, 0x44AC874E), U64(0xA327FFB2, 0x66B56220), /* ~= 10^-262 */
    U64(0xFBE91419, 0x15D7A922), U64(0x4BF1FF9F, 0x0062BAA8), /* ~= 10^-261 */
    U64(0x9D71AC8F, 0xADA6C9B5), U64(0x6F773FC3, 0x603DB4A9), /* ~= 10^-260 */
    U64(0xC4CE17B3, 0x99107C22), U64(0xCB550FB4, 0x384D21D3), /* ~= 10^-259 */
    U64(0xF6019DA0, 0x7F549B2B), U64(0x7E2A53A1, 0x46606A48), /* ~= 10^-258 */
    U64(0x99C10284, 0x4F94E0FB), U64(0x2EDA7444, 0xCBFC426D), /* ~= 10^-257 */
    U64(0xC0314325, 0x637A1939), U64(0xFA911155, 0xFEFB5308), /* ~= 10^-256 */
    U64(0xF03D93EE, 0xBC589F88), U64(0x793555AB, 0x7EBA27CA), /* ~= 10^-255 */
    U64(0x96267C75, 0x35B763B5), U64(0x4BC1558B, 0x2F3458DE), /* ~= 10^-254 */
    U64(0xBBB01B92, 0x83253CA2), U64(0x9EB1AAED, 0xFB016F16), /* ~= 10^-253 */
    U64(0xEA9C2277, 0x23EE8BCB), U64(0x465E15A9, 0x79C1CADC), /* ~= 10^-252 */
    U64(0x92A1958A, 0x7675175F), U64(0x0BFACD89, 0xEC191EC9), /* ~= 10^-251 */
    U64(0xB749FAED, 0x14125D36), U64(0xCEF980EC, 0x671F667B), /* ~= 10^-250 */
    U64(0xE51C79A8, 0x5916F484), U64(0x82B7E127, 0x80E7401A), /* ~= 10^-249 */
    U64(0x8F31CC09, 0x37AE58D2), U64(0xD1B2ECB8, 0xB0908810), /* ~= 10^-248 */
    U64(0xB2FE3F0B, 0x8599EF07), U64(0x861FA7E6, 0xDCB4AA15), /* ~= 10^-247 */
    U64(0xDFBDCECE, 0x67006AC9), U64(0x67A791E0, 0x93E1D49A), /* ~= 10^-246 */
    U64(0x8BD6A141, 0x006042BD), U64(0xE0C8BB2C, 0x5C6D24E0), /* ~= 10^-245 */
    U64(0xAECC4991, 0x4078536D), U64(0x58FAE9F7, 0x73886E18), /* ~= 10^-244 */
    U64(0xDA7F5BF5, 0x90966848), U64(0xAF39A475, 0x506A899E), /* ~= 10^-243 */
    U64(0x888F9979, 0x7A5E012D), U64(0x6D8406C9, 0x52429603), /* ~= 10^-242 */
    U64(0xAAB37FD7, 0xD8F58178), U64(0xC8E5087B, 0xA6D33B83), /* ~= 10^-241 */
    U64(0xD5605FCD, 0xCF32E1D6), U64(0xFB1E4A9A, 0x90880A64), /* ~= 10^-240 */
    U64(0x855C3BE0, 0xA17FCD26), U64(0x5CF2EEA0, 0x9A55067F), /* ~= 10^-239 */
    U64(0xA6B34AD8, 0xC9DFC06F), U64(0xF42FAA48, 0xC0EA481E), /* ~= 10^-238 */
    U64(0xD0601D8E, 0xFC57B08B), U64(0xF13B94DA, 0xF124DA26), /* ~= 10^-237 */
    U64(0x823C1279, 0x5DB6CE57), U64(0x76C53D08, 0xD6B70858), /* ~= 10^-236 */
    U64(0xA2CB1717, 0xB52481ED), U64(0x54768C4B, 0x0C64CA6E), /* ~= 10^-235 */
    U64(0xCB7DDCDD, 0xA26DA268), U64(0xA9942F5D, 0xCF7DFD09), /* ~= 10^-234 */
    U64(0xFE5D5415, 0x0B090B02), U64(0xD3F93B35, 0x435D7C4C), /* ~= 10^-233 */
    U64(0x9EFA548D, 0x26E5A6E1), U64(0xC47BC501, 0x4A1A6DAF), /* ~= 10^-232 */
    U64(0xC6B8E9B0, 0x709F109A), U64(0x359AB641, 0x9CA1091B), /* ~= 10^-231 */
    U64(0xF867241C, 0x8CC6D4C0), U64(0xC30163D2, 0x03C94B62), /* ~= 10^-230 */
    U64(0x9B407691, 0xD7FC44F8), U64(0x79E0DE63, 0x425DCF1D), /* ~= 10^-229 */
    U64(0xC2109436, 0x4DFB5636), U64(0x985915FC, 0x12F542E4), /* ~= 10^-228 */
    U64(0xF294B943, 0xE17A2BC4), U64(0x3E6F5B7B, 0x17B2939D), /* ~= 10^-227 */
    U64(0x979CF3CA, 0x6CEC5B5A), U64(0xA705992C, 0xEECF9C42), /* ~= 10^-226 */
    U64(0xBD8430BD, 0x08277231), U64(0x50C6FF78, 0x2A838353), /* ~= 10^-225 */
    U64(0xECE53CEC, 0x4A314EBD), U64(0xA4F8BF56, 0x35246428), /* ~= 10^-224 */
    U64(0x940F4613, 0xAE5ED136), U64(0x871B7795, 0xE136BE99), /* ~= 10^-223 */
    U64(0xB9131798, 0x99F68584), U64(0x28E2557B, 0x59846E3F), /* ~= 10^-222 */
    U64(0xE757DD7E, 0xC07426E5), U64(0x331AEADA, 0x2FE589CF), /* ~= 10^-221 */
    U64(0x9096EA6F, 0x3848984F), U64(0x3FF0D2C8, 0x5DEF7621), /* ~= 10^-220 */
    U64(0xB4BCA50B, 0x065ABE63), U64(0x0FED077A, 0x756B53A9), /* ~= 10^-219 */
    U64(0xE1EBCE4D, 0xC7F16DFB), U64(0xD3E84959, 0x12C62894), /* ~= 10^-218 */
    U64(0x8D3360F0, 0x9CF6E4BD), U64(0x64712DD7, 0xABBBD95C), /* ~= 10^-217 */
    U64(0xB080392C, 0xC4349DEC), U64(0xBD8D794D, 0x96AACFB3), /* ~= 10^-216 */
    U64(0xDCA04777, 0xF541C567), U64(0xECF0D7A0, 0xFC5583A0), /* ~= 10^-215 */
    U64(0x89E42CAA, 0xF9491B60), U64(0xF41686C4, 0x9DB57244), /* ~= 10^-214 */
    U64(0xAC5D37D5, 0xB79B6239), U64(0x311C2875, 0xC522CED5), /* ~= 10^-213 */
    U64(0xD77485CB, 0x25823AC7), U64(0x7D633293, 0x366B828B), /* ~= 10^-212 */
    U64(0x86A8D39E, 0xF77164BC), U64(0xAE5DFF9C, 0x02033197), /* ~= 10^-211 */
    U64(0xA8530886, 0xB54DBDEB), U64(0xD9F57F83, 0x0283FDFC), /* ~= 10^-210 */
    U64(0xD267CAA8, 0x62A12D66), U64(0xD072DF63, 0xC324FD7B), /* ~= 10^-209 */
    U64(0x8380DEA9, 0x3DA4BC60), U64(0x4247CB9E, 0x59F71E6D), /* ~= 10^-208 */
    U64(0xA4611653, 0x8D0DEB78), U64(0x52D9BE85, 0xF074E608), /* ~= 10^-207 */
    U64(0xCD795BE8, 0x70516656), U64(0x67902E27, 0x6C921F8B), /* ~= 10^-206 */
    U64(0x806BD971, 0x4632DFF6), U64(0x00BA1CD8, 0xA3DB53B6), /* ~= 10^-205 */
    U64(0xA086CFCD, 0x97BF97F3), U64(0x80E8A40E, 0xCCD228A4), /* ~= 10^-204 */
    U64(0xC8A883C0, 0xFDAF7DF0), U64(0x6122CD12, 0x8006B2CD), /* ~= 10^-203 */
    U64(0xFAD2A4B1, 0x3D1B5D6C), U64(0x796B8057, 0x20085F81), /* ~= 10^-202 */
    U64(0x9CC3A6EE, 0xC6311A63), U64(0xCBE33036, 0x74053BB0), /* ~= 10^-201 */
    U64(0xC3F490AA, 0x77BD60FC), U64(0xBEDBFC44, 0x11068A9C), /* ~= 10^-200 */
    U64(0xF4F1B4D5, 0x15ACB93B), U64(0xEE92FB55, 0x15482D44), /* ~= 10^-199 */
    U64(0x99171105, 0x2D8BF3C5), U64(0x751BDD15, 0x2D4D1C4A), /* ~= 10^-198 */
    U64(0xBF5CD546, 0x78EEF0B6), U64(0xD262D45A, 0x78A0635D), /* ~= 10^-197 */
    U64(0xEF340A98, 0x172AACE4), U64(0x86FB8971, 0x16C87C34), /* ~= 10^-196 */
    U64(0x9580869F, 0x0E7AAC0E), U64(0xD45D35E6, 0xAE3D4DA0), /* ~= 10^-195 */
    U64(0xBAE0A846, 0xD2195712), U64(0x89748360, 0x59CCA109), /* ~= 10^-194 */
    U64(0xE998D258, 0x869FACD7), U64(0x2BD1A438, 0x703FC94B), /* ~= 10^-193 */
    U64(0x91FF8377, 0x5423CC06), U64(0x7B6306A3, 0x4627DDCF), /* ~= 10^-192 */
    U64(0xB67F6455, 0x292CBF08), U64(0x1A3BC84C, 0x17B1D542), /* ~= 10^-191 */
    U64(0xE41F3D6A, 0x7377EECA), U64(0x20CABA5F, 0x1D9E4A93), /* ~= 10^-190 */
    U64(0x8E938662, 0x882AF53E), U64(0x547EB47B, 0x7282EE9C), /* ~= 10^-189 */
    U64(0xB23867FB, 0x2A35B28D), U64(0xE99E619A, 0x4F23AA43), /* ~= 10^-188 */
    U64(0xDEC681F9, 0xF4C31F31), U64(0x6405FA00, 0xE2EC94D4), /* ~= 10^-187 */
    U64(0x8B3C113C, 0x38F9F37E), U64(0xDE83BC40, 0x8DD3DD04), /* ~= 10^-186 */
    U64(0xAE0B158B, 0x4738705E), U64(0x9624AB50, 0xB148D445), /* ~= 10^-185 */
    U64(0xD98DDAEE, 0x19068C76), U64(0x3BADD624, 0xDD9B0957), /* ~= 10^-184 */
    U64(0x87F8A8D4, 0xCFA417C9), U64(0xE54CA5D7, 0x0A80E5D6), /* ~= 10^-183 */
    U64(0xA9F6D30A, 0x038D1DBC), U64(0x5E9FCF4C, 0xCD211F4C), /* ~= 10^-182 */
    U64(0xD47487CC, 0x8470652B), U64(0x7647C320, 0x0069671F), /* ~= 10^-181 */
    U64(0x84C8D4DF, 0xD2C63F3B), U64(0x29ECD9F4, 0x0041E073), /* ~= 10^-180 */
    U64(0xA5FB0A17, 0xC777CF09), U64(0xF4681071, 0x00525890), /* ~= 10^-179 */
    U64(0xCF79CC9D, 0xB955C2CC), U64(0x7182148D, 0x4066EEB4), /* ~= 10^-178 */
    U64(0x81AC1FE2, 0x93D599BF), U64(0xC6F14CD8, 0x48405530), /* ~= 10^-177 */
    U64(0xA21727DB, 0x38CB002F), U64(0xB8ADA00E, 0x5A506A7C), /* ~= 10^-176 */
    U64(0xCA9CF1D2, 0x06FDC03B), U64(0xA6D90811, 0xF0E4851C), /* ~= 10^-175 */
    U64(0xFD442E46, 0x88BD304A), U64(0x908F4A16, 0x6D1DA663), /* ~= 10^-174 */
    U64(0x9E4A9CEC, 0x15763E2E), U64(0x9A598E4E, 0x043287FE), /* ~= 10^-173 */
    U64(0xC5DD4427, 0x1AD3CDBA), U64(0x40EFF1E1, 0x853F29FD), /* ~= 10^-172 */
    U64(0xF7549530, 0xE188C128), U64(0xD12BEE59, 0xE68EF47C), /* ~= 10^-171 */
    U64(0x9A94DD3E, 0x8CF578B9), U64(0x82BB74F8, 0x301958CE), /* ~= 10^-170 */
    U64(0xC13A148E, 0x3032D6E7), U64(0xE36A5236, 0x3C1FAF01), /* ~= 10^-169 */
    U64(0xF18899B1, 0xBC3F8CA1), U64(0xDC44E6C3, 0xCB279AC1), /* ~= 10^-168 */
    U64(0x96F5600F, 0x15A7B7E5), U64(0x29AB103A, 0x5EF8C0B9), /* ~= 10^-167 */
    U64(0xBCB2B812, 0xDB11A5DE), U64(0x7415D448, 0xF6B6F0E7), /* ~= 10^-166 */
    U64(0xEBDF6617, 0x91D60F56), U64(0x111B495B, 0x3464AD21), /* ~= 10^-165 */
    U64(0x936B9FCE, 0xBB25C995), U64(0xCAB10DD9, 0x00BEEC34), /* ~= 10^-164 */
    U64(0xB84687C2, 0x69EF3BFB), U64(0x3D5D514F, 0x40EEA742), /* ~= 10^-163 */
    U64(0xE65829B3, 0x046B0AFA), U64(0x0CB4A5A3, 0x112A5112), /* ~= 10^-162 */
    U64(0x8FF71A0F, 0xE2C2E6DC), U64(0x47F0E785, 0xEABA72AB), /* ~= 10^-161 */
    U64(0xB3F4E093, 0xDB73A093), U64(0x59ED2167, 0x65690F56), /* ~= 10^-160 */
    U64(0xE0F218B8, 0xD25088B8), U64(0x306869C1, 0x3EC3532C), /* ~= 10^-159 */
    U64(0x8C974F73, 0x83725573), U64(0x1E414218, 0xC73A13FB), /* ~= 10^-158 */
    U64(0xAFBD2350, 0x644EEACF), U64(0xE5D1929E, 0xF90898FA), /* ~= 10^-157 */
    U64(0xDBAC6C24, 0x7D62A583), U64(0xDF45F746, 0xB74ABF39), /* ~= 10^-156 */
    U64(0x894BC396, 0xCE5DA772), U64(0x6B8BBA8C, 0x328EB783), /* ~= 10^-155 */
    U64(0xAB9EB47C, 0x81F5114F), U64(0x066EA92F, 0x3F326564), /* ~= 10^-154 */
    U64(0xD686619B, 0xA27255A2), U64(0xC80A537B, 0x0EFEFEBD), /* ~= 10^-153 */
    U64(0x8613FD01, 0x45877585), U64(0xBD06742C, 0xE95F5F36), /* ~= 10^-152 */
    U64(0xA798FC41, 0x96E952E7), U64(0x2C481138, 0x23B73704), /* ~= 10^-151 */
    U64(0xD17F3B51, 0xFCA3A7A0), U64(0xF75A1586, 0x2CA504C5), /* ~= 10^-150 */
    U64(0x82EF8513, 0x3DE648C4), U64(0x9A984D73, 0xDBE722FB), /* ~= 10^-149 */
    U64(0xA3AB6658, 0x0D5FDAF5), U64(0xC13E60D0, 0xD2E0EBBA), /* ~= 10^-148 */
    U64(0xCC963FEE, 0x10B7D1B3), U64(0x318DF905, 0x079926A8), /* ~= 10^-147 */
    U64(0xFFBBCFE9, 0x94E5C61F), U64(0xFDF17746, 0x497F7052), /* ~= 10^-146 */
    U64(0x9FD561F1, 0xFD0F9BD3), U64(0xFEB6EA8B, 0xEDEFA633), /* ~= 10^-145 */
    U64(0xC7CABA6E, 0x7C5382C8), U64(0xFE64A52E, 0xE96B8FC0), /* ~= 10^-144 */
    U64(0xF9BD690A, 0x1B68637B), U64(0x3DFDCE7A, 0xA3C673B0), /* ~= 10^-143 */
    U64(0x9C1661A6, 0x51213E2D), U64(0x06BEA10C, 0xA65C084E), /* ~= 10^-142 */
    U64(0xC31BFA0F, 0xE5698DB8), U64(0x486E494F, 0xCFF30A62), /* ~= 10^-141 */
    U64(0xF3E2F893, 0xDEC3F126), U64(0x5A89DBA3, 0xC3EFCCFA), /* ~= 10^-140 */
    U64(0x986DDB5C, 0x6B3A76B7), U64(0xF8962946, 0x5A75E01C), /* ~= 10^-139 */
    U64(0xBE895233, 0x86091465), U64(0xF6BBB397, 0xF1135823), /* ~= 10^-138 */
    U64(0xEE2BA6C0, 0x678B597F), U64(0x746AA07D, 0xED582E2C), /* ~= 10^-137 */
    U64(0x94DB4838, 0x40B717EF), U64(0xA8C2A44E, 0xB4571CDC), /* ~= 10^-136 */
    U64(0xBA121A46, 0x50E4DDEB), U64(0x92F34D62, 0x616CE413), /* ~= 10^-135 */
    U64(0xE896A0D7, 0xE51E1566), U64(0x77B020BA, 0xF9C81D17), /* ~= 10^-134 */
    U64(0x915E2486, 0xEF32CD60), U64(0x0ACE1474, 0xDC1D122E), /* ~= 10^-133 */
    U64(0xB5B5ADA8, 0xAAFF80B8), U64(0x0D819992, 0x132456BA), /* ~= 10^-132 */
    U64(0xE3231912, 0xD5BF60E6), U64(0x10E1FFF6, 0x97ED6C69), /* ~= 10^-131 */
    U64(0x8DF5EFAB, 0xC5979C8F), U64(0xCA8D3FFA, 0x1EF463C1), /* ~= 10^-130 */
    U64(0xB1736B96, 0xB6FD83B3), U64(0xBD308FF8, 0xA6B17CB2), /* ~= 10^-129 */
    U64(0xDDD0467C, 0x64BCE4A0), U64(0xAC7CB3F6, 0xD05DDBDE), /* ~= 10^-128 */
    U64(0x8AA22C0D, 0xBEF60EE4), U64(0x6BCDF07A, 0x423AA96B), /* ~= 10^-127 */
    U64(0xAD4AB711, 0x2EB3929D), U64(0x86C16C98, 0xD2C953C6), /* ~= 10^-126 */
    U64(0xD89D64D5, 0x7A607744), U64(0xE871C7BF, 0x077BA8B7), /* ~= 10^-125 */
    U64(0x87625F05, 0x6C7C4A8B), U64(0x11471CD7, 0x64AD4972), /* ~= 10^-124 */
    U64(0xA93AF6C6, 0xC79B5D2D), U64(0xD598E40D, 0x3DD89BCF), /* ~= 10^-123 */
    U64(0xD389B478, 0x79823479), U64(0x4AFF1D10, 0x8D4EC2C3), /* ~= 10^-122 */
    U64(0x843610CB, 0x4BF160CB), U64(0xCEDF722A, 0x585139BA), /* ~= 10^-121 */
    U64(0xA54394FE, 0x1EEDB8FE), U64(0xC2974EB4, 0xEE658828), /* ~= 10^-120 */
    U64(0xCE947A3D, 0xA6A9273E), U64(0x733D2262, 0x29FEEA32), /* ~= 10^-119 */
    U64(0x811CCC66, 0x8829B887), U64(0x0806357D, 0x5A3F525F), /* ~= 10^-118 */
    U64(0xA163FF80, 0x2A3426A8), U64(0xCA07C2DC, 0xB0CF26F7), /* ~= 10^-117 */
    U64(0xC9BCFF60, 0x34C13052), U64(0xFC89B393, 0xDD02F0B5), /* ~= 10^-116 */
    U64(0xFC2C3F38, 0x41F17C67), U64(0xBBAC2078, 0xD443ACE2), /* ~= 10^-115 */
    U64(0x9D9BA783, 0x2936EDC0), U64(0xD54B944B, 0x84AA4C0D), /* ~= 10^-114 */
    U64(0xC5029163, 0xF384A931), U64(0x0A9E795E, 0x65D4DF11), /* ~= 10^-113 */
    U64(0xF64335BC, 0xF065D37D), U64(0x4D4617B5, 0xFF4A16D5), /* ~= 10^-112 */
    U64(0x99EA0196, 0x163FA42E), U64(0x504BCED1, 0xBF8E4E45), /* ~= 10^-111 */
    U64(0xC06481FB, 0x9BCF8D39), U64(0xE45EC286, 0x2F71E1D6), /* ~= 10^-110 */
    U64(0xF07DA27A, 0x82C37088), U64(0x5D767327, 0xBB4E5A4C), /* ~= 10^-109 */
    U64(0x964E858C, 0x91BA2655), U64(0x3A6A07F8, 0xD510F86F), /* ~= 10^-108 */
    U64(0xBBE226EF, 0xB628AFEA), U64(0x890489F7, 0x0A55368B), /* ~= 10^-107 */
    U64(0xEADAB0AB, 0xA3B2DBE5), U64(0x2B45AC74, 0xCCEA842E), /* ~= 10^-106 */
    U64(0x92C8AE6B, 0x464FC96F), U64(0x3B0B8BC9, 0x0012929D), /* ~= 10^-105 */
    U64(0xB77ADA06, 0x17E3BBCB), U64(0x09CE6EBB, 0x40173744), /* ~= 10^-104 */
    U64(0xE5599087, 0x9DDCAABD), U64(0xCC420A6A, 0x101D0515), /* ~= 10^-103 */
    U64(0x8F57FA54, 0xC2A9EAB6), U64(0x9FA94682, 0x4A12232D), /* ~= 10^-102 */
    U64(0xB32DF8E9, 0xF3546564), U64(0x47939822, 0xDC96ABF9), /* ~= 10^-101 */
    U64(0xDFF97724, 0x70297EBD), U64(0x59787E2B, 0x93BC56F7), /* ~= 10^-100 */
    U64(0x8BFBEA76, 0xC619EF36), U64(0x57EB4EDB, 0x3C55B65A), /* ~= 10^-99 */
    U64(0xAEFAE514, 0x77A06B03), U64(0xEDE62292, 0x0B6B23F1), /* ~= 10^-98 */
    U64(0xDAB99E59, 0x958885C4), U64(0xE95FAB36, 0x8E45ECED), /* ~= 10^-97 */
    U64(0x88B402F7, 0xFD75539B), U64(0x11DBCB02, 0x18EBB414), /* ~= 10^-96 */
    U64(0xAAE103B5, 0xFCD2A881), U64(0xD652BDC2, 0x9F26A119), /* ~= 10^-95 */
    U64(0xD59944A3, 0x7C0752A2), U64(0x4BE76D33, 0x46F0495F), /* ~= 10^-94 */
    U64(0x857FCAE6, 0x2D8493A5), U64(0x6F70A440, 0x0C562DDB), /* ~= 10^-93 */
    U64(0xA6DFBD9F, 0xB8E5B88E), U64(0xCB4CCD50, 0x0F6BB952), /* ~= 10^-92 */
    U64(0xD097AD07, 0xA71F26B2), U64(0x7E2000A4, 0x1346A7A7), /* ~= 10^-91 */
    U64(0x825ECC24, 0xC873782F), U64(0x8ED40066, 0x8C0C28C8), /* ~= 10^-90 */
    U64(0xA2F67F2D, 0xFA90563B), U64(0x72890080, 0x2F0F32FA), /* ~= 10^-89 */
    U64(0xCBB41EF9, 0x79346BCA), U64(0x4F2B40A0, 0x3AD2FFB9), /* ~= 10^-88 */
    U64(0xFEA126B7, 0xD78186BC), U64(0xE2F610C8, 0x4987BFA8), /* ~= 10^-87 */
    U64(0x9F24B832, 0xE6B0F436), U64(0x0DD9CA7D, 0x2DF4D7C9), /* ~= 10^-86 */
    U64(0xC6EDE63F, 0xA05D3143), U64(0x91503D1C, 0x79720DBB), /* ~= 10^-85 */
    U64(0xF8A95FCF, 0x88747D94), U64(0x75A44C63, 0x97CE912A), /* ~= 10^-84 */
    U64(0x9B69DBE1, 0xB548CE7C), U64(0xC986AFBE, 0x3EE11ABA), /* ~= 10^-83 */
    U64(0xC24452DA, 0x229B021B), U64(0xFBE85BAD, 0xCE996168), /* ~= 10^-82 */
    U64(0xF2D56790, 0xAB41C2A2), U64(0xFAE27299, 0x423FB9C3), /* ~= 10^-81 */
    U64(0x97C560BA, 0x6B0919A5), U64(0xDCCD879F, 0xC967D41A), /* ~= 10^-80 */
    U64(0xBDB6B8E9, 0x05CB600F), U64(0x5400E987, 0xBBC1C920), /* ~= 10^-79 */
    U64(0xED246723, 0x473E3813), U64(0x290123E9, 0xAAB23B68), /* ~= 10^-78 */
    U64(0x9436C076, 0x0C86E30B), U64(0xF9A0B672, 0x0AAF6521), /* ~= 10^-77 */
    U64(0xB9447093, 0x8FA89BCE), U64(0xF808E40E, 0x8D5B3E69), /* ~= 10^-76 */
    U64(0xE7958CB8, 0x7392C2C2), U64(0xB60B1D12, 0x30B20E04), /* ~= 10^-75 */
    U64(0x90BD77F3, 0x483BB9B9), U64(0xB1C6F22B, 0x5E6F48C2), /* ~= 10^-74 */
    U64(0xB4ECD5F0, 0x1A4AA828), U64(0x1E38AEB6, 0x360B1AF3), /* ~= 10^-73 */
    U64(0xE2280B6C, 0x20DD5232), U64(0x25C6DA63, 0xC38DE1B0), /* ~= 10^-72 */
    U64(0x8D590723, 0x948A535F), U64(0x579C487E, 0x5A38AD0E), /* ~= 10^-71 */
    U64(0xB0AF48EC, 0x79ACE837), U64(0x2D835A9D, 0xF0C6D851), /* ~= 10^-70 */
    U64(0xDCDB1B27, 0x98182244), U64(0xF8E43145, 0x6CF88E65), /* ~= 10^-69 */
    U64(0x8A08F0F8, 0xBF0F156B), U64(0x1B8E9ECB, 0x641B58FF), /* ~= 10^-68 */
    U64(0xAC8B2D36, 0xEED2DAC5), U64(0xE272467E, 0x3D222F3F), /* ~= 10^-67 */
    U64(0xD7ADF884, 0xAA879177), U64(0x5B0ED81D, 0xCC6ABB0F), /* ~= 10^-66 */
    U64(0x86CCBB52, 0xEA94BAEA), U64(0x98E94712, 0x9FC2B4E9), /* ~= 10^-65 */
    U64(0xA87FEA27, 0xA539E9A5), U64(0x3F2398D7, 0x47B36224), /* ~= 10^-64 */
    U64(0xD29FE4B1, 0x8E88640E), U64(0x8EEC7F0D, 0x19A03AAD), /* ~= 10^-63 */
    U64(0x83A3EEEE, 0xF9153E89), U64(0x1953CF68, 0x300424AC), /* ~= 10^-62 */
    U64(0xA48CEAAA, 0xB75A8E2B), U64(0x5FA8C342, 0x3C052DD7), /* ~= 10^-61 */
    U64(0xCDB02555, 0x653131B6), U64(0x3792F412, 0xCB06794D), /* ~= 10^-60 */
    U64(0x808E1755, 0x5F3EBF11), U64(0xE2BBD88B, 0xBEE40BD0), /* ~= 10^-59 */
    U64(0xA0B19D2A, 0xB70E6ED6), U64(0x5B6ACEAE, 0xAE9D0EC4), /* ~= 10^-58 */
    U64(0xC8DE0475, 0x64D20A8B), U64(0xF245825A, 0x5A445275), /* ~= 10^-57 */
    U64(0xFB158592, 0xBE068D2E), U64(0xEED6E2F0, 0xF0D56712), /* ~= 10^-56 */
    U64(0x9CED737B, 0xB6C4183D), U64(0x55464DD6, 0x9685606B), /* ~= 10^-55 */
    U64(0xC428D05A, 0xA4751E4C), U64(0xAA97E14C, 0x3C26B886), /* ~= 10^-54 */
    U64(0xF5330471, 0x4D9265DF), U64(0xD53DD99F, 0x4B3066A8), /* ~= 10^-53 */
    U64(0x993FE2C6, 0xD07B7FAB), U64(0xE546A803, 0x8EFE4029), /* ~= 10^-52 */
    U64(0xBF8FDB78, 0x849A5F96), U64(0xDE985204, 0x72BDD033), /* ~= 10^-51 */
    U64(0xEF73D256, 0xA5C0F77C), U64(0x963E6685, 0x8F6D4440), /* ~= 10^-50 */
    U64(0x95A86376, 0x27989AAD), U64(0xDDE70013, 0x79A44AA8), /* ~= 10^-49 */
    U64(0xBB127C53, 0xB17EC159), U64(0x5560C018, 0x580D5D52), /* ~= 10^-48 */
    U64(0xE9D71B68, 0x9DDE71AF), U64(0xAAB8F01E, 0x6E10B4A6), /* ~= 10^-47 */
    U64(0x92267121, 0x62AB070D), U64(0xCAB39613, 0x04CA70E8), /* ~= 10^-46 */
    U64(0xB6B00D69, 0xBB55C8D1), U64(0x3D607B97, 0xC5FD0D22), /* ~= 10^-45 */
    U64(0xE45C10C4, 0x2A2B3B05), U64(0x8CB89A7D, 0xB77C506A), /* ~= 10^-44 */
    U64(0x8EB98A7A, 0x9A5B04E3), U64(0x77F3608E, 0x92ADB242), /* ~= 10^-43 */
    U64(0xB267ED19, 0x40F1C61C), U64(0x55F038B2, 0x37591ED3), /* ~= 10^-42 */
    U64(0xDF01E85F, 0x912E37A3), U64(0x6B6C46DE, 0xC52F6688), /* ~= 10^-41 */
    U64(0x8B61313B, 0xBABCE2C6), U64(0x2323AC4B, 0x3B3DA015), /* ~= 10^-40 */
    U64(0xAE397D8A, 0xA96C1B77), U64(0xABEC975E, 0x0A0D081A), /* ~= 10^-39 */
    U64(0xD9C7DCED, 0x53C72255), U64(0x96E7BD35, 0x8C904A21), /* ~= 10^-38 */
    U64(0x881CEA14, 0x545C7575), U64(0x7E50D641, 0x77DA2E54), /* ~= 10^-37 */
    U64(0xAA242499, 0x697392D2), U64(0xDDE50BD1, 0xD5D0B9E9), /* ~= 10^-36 */
    U64(0xD4AD2DBF, 0xC3D07787), U64(0x955E4EC6, 0x4B44E864), /* ~= 10^-35 */
    U64(0x84EC3C97, 0xDA624AB4), U64(0xBD5AF13B, 0xEF0B113E), /* ~= 10^-34 */
    U64(0xA6274BBD, 0xD0FADD61), U64(0xECB1AD8A, 0xEACDD58E), /* ~= 10^-33 */
    U64(0xCFB11EAD, 0x453994BA), U64(0x67DE18ED, 0xA5814AF2), /* ~= 10^-32 */
    U64(0x81CEB32C, 0x4B43FCF4), U64(0x80EACF94, 0x8770CED7), /* ~= 10^-31 */
    U64(0xA2425FF7, 0x5E14FC31), U64(0xA1258379, 0xA94D028D), /* ~= 10^-30 */
    U64(0xCAD2F7F5, 0x359A3B3E), U64(0x096EE458, 0x13A04330), /* ~= 10^-29 */
    U64(0xFD87B5F2, 0x8300CA0D), U64(0x8BCA9D6E, 0x188853FC), /* ~= 10^-28 */
    U64(0x9E74D1B7, 0x91E07E48), U64(0x775EA264, 0xCF55347D), /* ~= 10^-27 */
    U64(0xC6120625, 0x76589DDA), U64(0x95364AFE, 0x032A819D), /* ~= 10^-26 */
    U64(0xF79687AE, 0xD3EEC551), U64(0x3A83DDBD, 0x83F52204), /* ~= 10^-25 */
    U64(0x9ABE14CD, 0x44753B52), U64(0xC4926A96, 0x72793542), /* ~= 10^-24 */
    U64(0xC16D9A00, 0x95928A27), U64(0x75B7053C, 0x0F178293), /* ~= 10^-23 */
    U64(0xF1C90080, 0xBAF72CB1), U64(0x5324C68B, 0x12DD6338), /* ~= 10^-22 */
    U64(0x971DA050, 0x74DA7BEE), U64(0xD3F6FC16, 0xEBCA5E03), /* ~= 10^-21 */
    U64(0xBCE50864, 0x92111AEA), U64(0x88F4BB1C, 0xA6BCF584), /* ~= 10^-20 */
    U64(0xEC1E4A7D, 0xB69561A5), U64(0x2B31E9E3, 0xD06C32E5), /* ~= 10^-19 */
    U64(0x9392EE8E, 0x921D5D07), U64(0x3AFF322E, 0x62439FCF), /* ~= 10^-18 */
    U64(0xB877AA32, 0x36A4B449), U64(0x09BEFEB9, 0xFAD487C2), /* ~= 10^-17 */
    U64(0xE69594BE, 0xC44DE15B), U64(0x4C2EBE68, 0x7989A9B3), /* ~= 10^-16 */
    U64(0x901D7CF7, 0x3AB0ACD9), U64(0x0F9D3701, 0x4BF60A10), /* ~= 10^-15 */
    U64(0xB424DC35, 0x095CD80F), U64(0x538484C1, 0x9EF38C94), /* ~= 10^-14 */
    U64(0xE12E1342, 0x4BB40E13), U64(0x2865A5F2, 0x06B06FB9), /* ~= 10^-13 */
    U64(0x8CBCCC09, 0x6F5088CB), U64(0xF93F87B7, 0x442E45D3), /* ~= 10^-12 */
    U64(0xAFEBFF0B, 0xCB24AAFE), U64(0xF78F69A5, 0x1539D748), /* ~= 10^-11 */
    U64(0xDBE6FECE, 0xBDEDD5BE), U64(0xB573440E, 0x5A884D1B), /* ~= 10^-10 */
    U64(0x89705F41, 0x36B4A597), U64(0x31680A88, 0xF8953030), /* ~= 10^-9 */
    U64(0xABCC7711, 0x8461CEFC), U64(0xFDC20D2B, 0x36BA7C3D), /* ~= 10^-8 */
    U64(0xD6BF94D5, 0xE57A42BC), U64(0x3D329076, 0x04691B4C), /* ~= 10^-7 */
    U64(0x8637BD05, 0xAF6C69B5), U64(0xA63F9A49, 0xC2C1B10F), /* ~= 10^-6 */
    U64(0xA7C5AC47, 0x1B478423), U64(0x0FCF80DC, 0x33721D53), /* ~= 10^-5 */
    U64(0xD1B71758, 0xE219652B), U64(0xD3C36113, 0x404EA4A8), /* ~= 10^-4 */
    U64(0x83126E97, 0x8D4FDF3B), U64(0x645A1CAC, 0x083126E9), /* ~= 10^-3 */
    U64(0xA3D70A3D, 0x70A3D70A), U64(0x3D70A3D7, 0x0A3D70A3), /* ~= 10^-2 */
    U64(0xCCCCCCCC, 0xCCCCCCCC), U64(0xCCCCCCCC, 0xCCCCCCCC), /* ~= 10^-1 */
    U64(0x80000000, 0x00000000), U64(0x00000000, 0x00000000), /* == 10^0 */
    U64(0xA0000000, 0x00000000), U64(0x00000000, 0x00000000), /* == 10^1 */
    U64(0xC8000000, 0x00000000), U64(0x00000000, 0x00000000), /* == 10^2 */
    U64(0xFA000000, 0x00000000), U64(0x00000000, 0x00000000), /* == 10^3 */
    U64(0x9C400000, 0x00000000), U64(0x00000000, 0x00000000), /* == 10^4 */
    U64(0xC3500000, 0x00000000), U64(0x00000000, 0x00000000), /* == 10^5 */
    U64(0xF4240000, 0x00000000), U64(0x00000000, 0x00000000), /* == 10^6 */
    U64(0x98968000, 0x00000000), U64(0x00000000, 0x00000000), /* == 10^7 */
    U64(0xBEBC2000, 0x00000000), U64(0x00000000, 0x00000000), /* == 10^8 */
    U64(0xEE6B2800, 0x00000000), U64(0x00000000, 0x00000000), /* == 10^9 */
    U64(0x9502F900, 0x00000000), U64(0x00000000, 0x00000000), /* == 10^10 */
    U64(0xBA43B740, 0x00000000), U64(0x00000000, 0x00000000), /* == 10^11 */
    U64(0xE8D4A510, 0x00000000), U64(0x00000000, 0x00000000), /* == 10^12 */
    U64(0x9184E72A, 0x00000000), U64(0x00000000, 0x00000000), /* == 10^13 */
    U64(0xB5E620F4, 0x80000000), U64(0x00000000, 0x00000000), /* == 10^14 */
    U64(0xE35FA931, 0xA0000000), U64(0x00000000, 0x00000000), /* == 10^15 */
    U64(0x8E1BC9BF, 0x04000000), U64(0x00000000, 0x00000000), /* == 10^16 */
    U64(0xB1A2BC2E, 0xC5000000), U64(0x00000000, 0x00000000), /* == 10^17 */
    U64(0xDE0B6B3A, 0x76400000), U64(0x00000000, 0x00000000), /* == 10^18 */
    U64(0x8AC72304, 0x89E80000), U64(0x00000000, 0x00000000), /* == 10^19 */
    U64(0xAD78EBC5, 0xAC620000), U64(0x00000000, 0x00000000), /* == 10^20 */
    U64(0xD8D726B7, 0x177A8000), U64(0x00000000, 0x00000000), /* == 10^21 */
    U64(0x87867832, 0x6EAC9000), U64(0x00000000, 0x00000000), /* == 10^22 */
    U64(0xA968163F, 0x0A57B400), U64(0x00000000, 0x00000000), /* == 10^23 */
    U64(0xD3C21BCE, 0xCCEDA100), U64(0x00000000, 0x00000000), /* == 10^24 */
    U64(0x84595161, 0x401484A0), U64(0x00000000, 0x00000000), /* == 10^25 */
    U64(0xA56FA5B9, 0x9019A5C8), U64(0x00000000, 0x00000000), /* == 10^26 */
    U64(0xCECB8F27, 0xF4200F3A), U64(0x00000000, 0x00000000), /* == 10^27 */
    U64(0x813F3978, 0xF8940984), U64(0x40000000, 0x00000000), /* == 10^28 */
    U64(0xA18F07D7, 0x36B90BE5), U64(0x50000000, 0x00000000), /* == 10^29 */
    U64(0xC9F2C9CD, 0x04674EDE), U64(0xA4000000, 0x00000000), /* == 10^30 */
    U64(0xFC6F7C40, 0x45812296), U64(0x4D000000, 0x00000000), /* == 10^31 */
    U64(0x9DC5ADA8, 0x2B70B59D), U64(0xF0200000, 0x00000000), /* == 10^32 */
    U64(0xC5371912, 0x364CE305), U64(0x6C280000, 0x00000000), /* == 10^33 */
    U64(0xF684DF56, 0xC3E01BC6), U64(0xC7320000, 0x00000000), /* == 10^34 */
    U64(0x9A130B96, 0x3A6C115C), U64(0x3C7F4000, 0x00000000), /* == 10^35 */
    U64(0xC097CE7B, 0xC90715B3), U64(0x4B9F1000, 0x00000000), /* == 10^36 */
    U64(0xF0BDC21A, 0xBB48DB20), U64(0x1E86D400, 0x00000000), /* == 10^37 */
    U64(0x96769950, 0xB50D88F4), U64(0x13144480, 0x00000000), /* == 10^38 */
    U64(0xBC143FA4, 0xE250EB31), U64(0x17D955A0, 0x00000000), /* == 10^39 */
    U64(0xEB194F8E, 0x1AE525FD), U64(0x5DCFAB08, 0x00000000), /* == 10^40 */
    U64(0x92EFD1B8, 0xD0CF37BE), U64(0x5AA1CAE5, 0x00000000), /* == 10^41 */
    U64(0xB7ABC627, 0x050305AD), U64(0xF14A3D9E, 0x40000000), /* == 10^42 */
    U64(0xE596B7B0, 0xC643C719), U64(0x6D9CCD05, 0xD0000000), /* == 10^43 */
    U64(0x8F7E32CE, 0x7BEA5C6F), U64(0xE4820023, 0xA2000000), /* == 10^44 */
    U64(0xB35DBF82, 0x1AE4F38B), U64(0xDDA2802C, 0x8A800000), /* == 10^45 */
    U64(0xE0352F62, 0xA19E306E), U64(0xD50B2037, 0xAD200000), /* == 10^46 */
    U64(0x8C213D9D, 0xA502DE45), U64(0x4526F422, 0xCC340000), /* == 10^47 */
    U64(0xAF298D05, 0x0E4395D6), U64(0x9670B12B, 0x7F410000), /* == 10^48 */
    U64(0xDAF3F046, 0x51D47B4C), U64(0x3C0CDD76, 0x5F114000), /* == 10^49 */
    U64(0x88D8762B, 0xF324CD0F), U64(0xA5880A69, 0xFB6AC800), /* == 10^50 */
    U64(0xAB0E93B6, 0xEFEE0053), U64(0x8EEA0D04, 0x7A457A00), /* == 10^51 */
    U64(0xD5D238A4, 0xABE98068), U64(0x72A49045, 0x98D6D880), /* == 10^52 */
    U64(0x85A36366, 0xEB71F041), U64(0x47A6DA2B, 0x7F864750), /* == 10^53 */
    U64(0xA70C3C40, 0xA64E6C51), U64(0x999090B6, 0x5F67D924), /* == 10^54 */
    U64(0xD0CF4B50, 0xCFE20765), U64(0xFFF4B4E3, 0xF741CF6D), /* == 10^55 */
    U64(0x82818F12, 0x81ED449F), U64(0xBFF8F10E, 0x7A8921A4), /* ~= 10^56 */
    U64(0xA321F2D7, 0x226895C7), U64(0xAFF72D52, 0x192B6A0D), /* ~= 10^57 */
    U64(0xCBEA6F8C, 0xEB02BB39), U64(0x9BF4F8A6, 0x9F764490), /* ~= 10^58 */
    U64(0xFEE50B70, 0x25C36A08), U64(0x02F236D0, 0x4753D5B4), /* ~= 10^59 */
    U64(0x9F4F2726, 0x179A2245), U64(0x01D76242, 0x2C946590), /* ~= 10^60 */
    U64(0xC722F0EF, 0x9D80AAD6), U64(0x424D3AD2, 0xB7B97EF5), /* ~= 10^61 */
    U64(0xF8EBAD2B, 0x84E0D58B), U64(0xD2E08987, 0x65A7DEB2), /* ~= 10^62 */
    U64(0x9B934C3B, 0x330C8577), U64(0x63CC55F4, 0x9F88EB2F), /* ~= 10^63 */
    U64(0xC2781F49, 0xFFCFA6D5), U64(0x3CBF6B71, 0xC76B25FB), /* ~= 10^64 */
    U64(0xF316271C, 0x7FC3908A), U64(0x8BEF464E, 0x3945EF7A), /* ~= 10^65 */
    U64(0x97EDD871, 0xCFDA3A56), U64(0x97758BF0, 0xE3CBB5AC), /* ~= 10^66 */
    U64(0xBDE94E8E, 0x43D0C8EC), U64(0x3D52EEED, 0x1CBEA317), /* ~= 10^67 */
    U64(0xED63A231, 0xD4C4FB27), U64(0x4CA7AAA8, 0x63EE4BDD), /* ~= 10^68 */
    U64(0x945E455F, 0x24FB1CF8), U64(0x8FE8CAA9, 0x3E74EF6A), /* ~= 10^69 */
    U64(0xB975D6B6, 0xEE39E436), U64(0xB3E2FD53, 0x8E122B44), /* ~= 10^70 */
    U64(0xE7D34C64, 0xA9C85D44), U64(0x60DBBCA8, 0x7196B616), /* ~= 10^71 */
    U64(0x90E40FBE, 0xEA1D3A4A), U64(0xBC8955E9, 0x46FE31CD), /* ~= 10^72 */
    U64(0xB51D13AE, 0xA4A488DD), U64(0x6BABAB63, 0x98BDBE41), /* ~= 10^73 */
    U64(0xE264589A, 0x4DCDAB14), U64(0xC696963C, 0x7EED2DD1), /* ~= 10^74 */
    U64(0x8D7EB760, 0x70A08AEC), U64(0xFC1E1DE5, 0xCF543CA2), /* ~= 10^75 */
    U64(0xB0DE6538, 0x8CC8ADA8), U64(0x3B25A55F, 0x43294BCB), /* ~= 10^76 */
    U64(0xDD15FE86, 0xAFFAD912), U64(0x49EF0EB7, 0x13F39EBE), /* ~= 10^77 */
    U64(0x8A2DBF14, 0x2DFCC7AB), U64(0x6E356932, 0x6C784337), /* ~= 10^78 */
    U64(0xACB92ED9, 0x397BF996), U64(0x49C2C37F, 0x07965404), /* ~= 10^79 */
    U64(0xD7E77A8F, 0x87DAF7FB), U64(0xDC33745E, 0xC97BE906), /* ~= 10^80 */
    U64(0x86F0AC99, 0xB4E8DAFD), U64(0x69A028BB, 0x3DED71A3), /* ~= 10^81 */
    U64(0xA8ACD7C0, 0x222311BC), U64(0xC40832EA, 0x0D68CE0C), /* ~= 10^82 */
    U64(0xD2D80DB0, 0x2AABD62B), U64(0xF50A3FA4, 0x90C30190), /* ~= 10^83 */
    U64(0x83C7088E, 0x1AAB65DB), U64(0x792667C6, 0xDA79E0FA), /* ~= 10^84 */
    U64(0xA4B8CAB1, 0xA1563F52), U64(0x577001B8, 0x91185938), /* ~= 10^85 */
    U64(0xCDE6FD5E, 0x09ABCF26), U64(0xED4C0226, 0xB55E6F86), /* ~= 10^86 */
    U64(0x80B05E5A, 0xC60B6178), U64(0x544F8158, 0x315B05B4), /* ~= 10^87 */
    U64(0xA0DC75F1, 0x778E39D6), U64(0x696361AE, 0x3DB1C721), /* ~= 10^88 */
    U64(0xC913936D, 0xD571C84C), U64(0x03BC3A19, 0xCD1E38E9), /* ~= 10^89 */
    U64(0xFB587849, 0x4ACE3A5F), U64(0x04AB48A0, 0x4065C723), /* ~= 10^90 */
    U64(0x9D174B2D, 0xCEC0E47B), U64(0x62EB0D64, 0x283F9C76), /* ~= 10^91 */
    U64(0xC45D1DF9, 0x42711D9A), U64(0x3BA5D0BD, 0x324F8394), /* ~= 10^92 */
    U64(0xF5746577, 0x930D6500), U64(0xCA8F44EC, 0x7EE36479), /* ~= 10^93 */
    U64(0x9968BF6A, 0xBBE85F20), U64(0x7E998B13, 0xCF4E1ECB), /* ~= 10^94 */
    U64(0xBFC2EF45, 0x6AE276E8), U64(0x9E3FEDD8, 0xC321A67E), /* ~= 10^95 */
    U64(0xEFB3AB16, 0xC59B14A2), U64(0xC5CFE94E, 0xF3EA101E), /* ~= 10^96 */
    U64(0x95D04AEE, 0x3B80ECE5), U64(0xBBA1F1D1, 0x58724A12), /* ~= 10^97 */
    U64(0xBB445DA9, 0xCA61281F), U64(0x2A8A6E45, 0xAE8EDC97), /* ~= 10^98 */
    U64(0xEA157514, 0x3CF97226), U64(0xF52D09D7, 0x1A3293BD), /* ~= 10^99 */
    U64(0x924D692C, 0xA61BE758), U64(0x593C2626, 0x705F9C56), /* ~= 10^100 */
    U64(0xB6E0C377, 0xCFA2E12E), U64(0x6F8B2FB0, 0x0C77836C), /* ~= 10^101 */
    U64(0xE498F455, 0xC38B997A), U64(0x0B6DFB9C, 0x0F956447), /* ~= 10^102 */
    U64(0x8EDF98B5, 0x9A373FEC), U64(0x4724BD41, 0x89BD5EAC), /* ~= 10^103 */
    U64(0xB2977EE3, 0x00C50FE7), U64(0x58EDEC91, 0xEC2CB657), /* ~= 10^104 */
    U64(0xDF3D5E9B, 0xC0F653E1), U64(0x2F2967B6, 0x6737E3ED), /* ~= 10^105 */
    U64(0x8B865B21, 0x5899F46C), U64(0xBD79E0D2, 0x0082EE74), /* ~= 10^106 */
    U64(0xAE67F1E9, 0xAEC07187), U64(0xECD85906, 0x80A3AA11), /* ~= 10^107 */
    U64(0xDA01EE64, 0x1A708DE9), U64(0xE80E6F48, 0x20CC9495), /* ~= 10^108 */
    U64(0x884134FE, 0x908658B2), U64(0x3109058D, 0x147FDCDD), /* ~= 10^109 */
    U64(0xAA51823E, 0x34A7EEDE), U64(0xBD4B46F0, 0x599FD415), /* ~= 10^110 */
    U64(0xD4E5E2CD, 0xC1D1EA96), U64(0x6C9E18AC, 0x7007C91A), /* ~= 10^111 */
    U64(0x850FADC0, 0x9923329E), U64(0x03E2CF6B, 0xC604DDB0), /* ~= 10^112 */
    U64(0xA6539930, 0xBF6BFF45), U64(0x84DB8346, 0xB786151C), /* ~= 10^113 */
    U64(0xCFE87F7C, 0xEF46FF16), U64(0xE6126418, 0x65679A63), /* ~= 10^114 */
    U64(0x81F14FAE, 0x158C5F6E), U64(0x4FCB7E8F, 0x3F60C07E), /* ~= 10^115 */
    U64(0xA26DA399, 0x9AEF7749), U64(0xE3BE5E33, 0x0F38F09D), /* ~= 10^116 */
    U64(0xCB090C80, 0x01AB551C), U64(0x5CADF5BF, 0xD3072CC5), /* ~= 10^117 */
    U64(0xFDCB4FA0, 0x02162A63), U64(0x73D9732F, 0xC7C8F7F6), /* ~= 10^118 */
    U64(0x9E9F11C4, 0x014DDA7E), U64(0x2867E7FD, 0xDCDD9AFA), /* ~= 10^119 */
    U64(0xC646D635, 0x01A1511D), U64(0xB281E1FD, 0x541501B8), /* ~= 10^120 */
    U64(0xF7D88BC2, 0x4209A565), U64(0x1F225A7C, 0xA91A4226), /* ~= 10^121 */
    U64(0x9AE75759, 0x6946075F), U64(0x3375788D, 0xE9B06958), /* ~= 10^122 */
    U64(0xC1A12D2F, 0xC3978937), U64(0x0052D6B1, 0x641C83AE), /* ~= 10^123 */
    U64(0xF209787B, 0xB47D6B84), U64(0xC0678C5D, 0xBD23A49A), /* ~= 10^124 */
    U64(0x9745EB4D, 0x50CE6332), U64(0xF840B7BA, 0x963646E0), /* ~= 10^125 */
    U64(0xBD176620, 0xA501FBFF), U64(0xB650E5A9, 0x3BC3D898), /* ~= 10^126 */
    U64(0xEC5D3FA8, 0xCE427AFF), U64(0xA3E51F13, 0x8AB4CEBE), /* ~= 10^127 */
    U64(0x93BA47C9, 0x80E98CDF), U64(0xC66F336C, 0x36B10137), /* ~= 10^128 */
    U64(0xB8A8D9BB, 0xE123F017), U64(0xB80B0047, 0x445D4184), /* ~= 10^129 */
    U64(0xE6D3102A, 0xD96CEC1D), U64(0xA60DC059, 0x157491E5), /* ~= 10^130 */
    U64(0x9043EA1A, 0xC7E41392), U64(0x87C89837, 0xAD68DB2F), /* ~= 10^131 */
    U64(0xB454E4A1, 0x79DD1877), U64(0x29BABE45, 0x98C311FB), /* ~= 10^132 */
    U64(0xE16A1DC9, 0xD8545E94), U64(0xF4296DD6, 0xFEF3D67A), /* ~= 10^133 */
    U64(0x8CE2529E, 0x2734BB1D), U64(0x1899E4A6, 0x5F58660C), /* ~= 10^134 */
    U64(0xB01AE745, 0xB101E9E4), U64(0x5EC05DCF, 0xF72E7F8F), /* ~= 10^135 */
    U64(0xDC21A117, 0x1D42645D), U64(0x76707543, 0xF4FA1F73), /* ~= 10^136 */
    U64(0x899504AE, 0x72497EBA), U64(0x6A06494A, 0x791C53A8), /* ~= 10^137 */
    U64(0xABFA45DA, 0x0EDBDE69), U64(0x0487DB9D, 0x17636892), /* ~= 10^138 */
    U64(0xD6F8D750, 0x9292D603), U64(0x45A9D284, 0x5D3C42B6), /* ~= 10^139 */
    U64(0x865B8692, 0x5B9BC5C2), U64(0x0B8A2392, 0xBA45A9B2), /* ~= 10^140 */
    U64(0xA7F26836, 0xF282B732), U64(0x8E6CAC77, 0x68D7141E), /* ~= 10^141 */
    U64(0xD1EF0244, 0xAF2364FF), U64(0x3207D795, 0x430CD926), /* ~= 10^142 */
    U64(0x8335616A, 0xED761F1F), U64(0x7F44E6BD, 0x49E807B8), /* ~= 10^143 */
    U64(0xA402B9C5, 0xA8D3A6E7), U64(0x5F16206C, 0x9C6209A6), /* ~= 10^144 */
    U64(0xCD036837, 0x130890A1), U64(0x36DBA887, 0xC37A8C0F), /* ~= 10^145 */
    U64(0x80222122, 0x6BE55A64), U64(0xC2494954, 0xDA2C9789), /* ~= 10^146 */
    U64(0xA02AA96B, 0x06DEB0FD), U64(0xF2DB9BAA, 0x10B7BD6C), /* ~= 10^147 */
    U64(0xC83553C5, 0xC8965D3D), U64(0x6F928294, 0x94E5ACC7), /* ~= 10^148 */
    U64(0xFA42A8B7, 0x3ABBF48C), U64(0xCB772339, 0xBA1F17F9), /* ~= 10^149 */
    U64(0x9C69A972, 0x84B578D7), U64(0xFF2A7604, 0x14536EFB), /* ~= 10^150 */
    U64(0xC38413CF, 0x25E2D70D), U64(0xFEF51385, 0x19684ABA), /* ~= 10^151 */
    U64(0xF46518C2, 0xEF5B8CD1), U64(0x7EB25866, 0x5FC25D69), /* ~= 10^152 */
    U64(0x98BF2F79, 0xD5993802), U64(0xEF2F773F, 0xFBD97A61), /* ~= 10^153 */
    U64(0xBEEEFB58, 0x4AFF8603), U64(0xAAFB550F, 0xFACFD8FA), /* ~= 10^154 */
    U64(0xEEAABA2E, 0x5DBF6784), U64(0x95BA2A53, 0xF983CF38), /* ~= 10^155 */
    U64(0x952AB45C, 0xFA97A0B2), U64(0xDD945A74, 0x7BF26183), /* ~= 10^156 */
    U64(0xBA756174, 0x393D88DF), U64(0x94F97111, 0x9AEEF9E4), /* ~= 10^157 */
    U64(0xE912B9D1, 0x478CEB17), U64(0x7A37CD56, 0x01AAB85D), /* ~= 10^158 */
    U64(0x91ABB422, 0xCCB812EE), U64(0xAC62E055, 0xC10AB33A), /* ~= 10^159 */
    U64(0xB616A12B, 0x7FE617AA), U64(0x577B986B, 0x314D6009), /* ~= 10^160 */
    U64(0xE39C4976, 0x5FDF9D94), U64(0xED5A7E85, 0xFDA0B80B), /* ~= 10^161 */
    U64(0x8E41ADE9, 0xFBEBC27D), U64(0x14588F13, 0xBE847307), /* ~= 10^162 */
    U64(0xB1D21964, 0x7AE6B31C), U64(0x596EB2D8, 0xAE258FC8), /* ~= 10^163 */
    U64(0xDE469FBD, 0x99A05FE3), U64(0x6FCA5F8E, 0xD9AEF3BB), /* ~= 10^164 */
    U64(0x8AEC23D6, 0x80043BEE), U64(0x25DE7BB9, 0x480D5854), /* ~= 10^165 */
    U64(0xADA72CCC, 0x20054AE9), U64(0xAF561AA7, 0x9A10AE6A), /* ~= 10^166 */
    U64(0xD910F7FF, 0x28069DA4), U64(0x1B2BA151, 0x8094DA04), /* ~= 10^167 */
    U64(0x87AA9AFF, 0x79042286), U64(0x90FB44D2, 0xF05D0842), /* ~= 10^168 */
    U64(0xA99541BF, 0x57452B28), U64(0x353A1607, 0xAC744A53), /* ~= 10^169 */
    U64(0xD3FA922F, 0x2D1675F2), U64(0x42889B89, 0x97915CE8), /* ~= 10^170 */
    U64(0x847C9B5D, 0x7C2E09B7), U64(0x69956135, 0xFEBADA11), /* ~= 10^171 */
    U64(0xA59BC234, 0xDB398C25), U64(0x43FAB983, 0x7E699095), /* ~= 10^172 */
    U64(0xCF02B2C2, 0x1207EF2E), U64(0x94F967E4, 0x5E03F4BB), /* ~= 10^173 */
    U64(0x8161AFB9, 0x4B44F57D), U64(0x1D1BE0EE, 0xBAC278F5), /* ~= 10^174 */
    U64(0xA1BA1BA7, 0x9E1632DC), U64(0x6462D92A, 0x69731732), /* ~= 10^175 */
    U64(0xCA28A291, 0x859BBF93), U64(0x7D7B8F75, 0x03CFDCFE), /* ~= 10^176 */
    U64(0xFCB2CB35, 0xE702AF78), U64(0x5CDA7352, 0x44C3D43E), /* ~= 10^177 */
    U64(0x9DEFBF01, 0xB061ADAB), U64(0x3A088813, 0x6AFA64A7), /* ~= 10^178 */
    U64(0xC56BAEC2, 0x1C7A1916), U64(0x088AAA18, 0x45B8FDD0), /* ~= 10^179 */
    U64(0xF6C69A72, 0xA3989F5B), U64(0x8AAD549E, 0x57273D45), /* ~= 10^180 */
    U64(0x9A3C2087, 0xA63F6399), U64(0x36AC54E2, 0xF678864B), /* ~= 10^181 */
    U64(0xC0CB28A9, 0x8FCF3C7F), U64(0x84576A1B, 0xB416A7DD), /* ~= 10^182 */
    U64(0xF0FDF2D3, 0xF3C30B9F), U64(0x656D44A2, 0xA11C51D5), /* ~= 10^183 */
    U64(0x969EB7C4, 0x7859E743), U64(0x9F644AE5, 0xA4B1B325), /* ~= 10^184 */
    U64(0xBC4665B5, 0x96706114), U64(0x873D5D9F, 0x0DDE1FEE), /* ~= 10^185 */
    U64(0xEB57FF22, 0xFC0C7959), U64(0xA90CB506, 0xD155A7EA), /* ~= 10^186 */
    U64(0x9316FF75, 0xDD87CBD8), U64(0x09A7F124, 0x42D588F2), /* ~= 10^187 */
    U64(0xB7DCBF53, 0x54E9BECE), U64(0x0C11ED6D, 0x538AEB2F), /* ~= 10^188 */
    U64(0xE5D3EF28, 0x2A242E81), U64(0x8F1668C8, 0xA86DA5FA), /* ~= 10^189 */
    U64(0x8FA47579, 0x1A569D10), U64(0xF96E017D, 0x694487BC), /* ~= 10^190 */
    U64(0xB38D92D7, 0x60EC4455), U64(0x37C981DC, 0xC395A9AC), /* ~= 10^191 */
    U64(0xE070F78D, 0x3927556A), U64(0x85BBE253, 0xF47B1417), /* ~= 10^192 */
    U64(0x8C469AB8, 0x43B89562), U64(0x93956D74, 0x78CCEC8E), /* ~= 10^193 */
    U64(0xAF584166, 0x54A6BABB), U64(0x387AC8D1, 0x970027B2), /* ~= 10^194 */
    U64(0xDB2E51BF, 0xE9D0696A), U64(0x06997B05, 0xFCC0319E), /* ~= 10^195 */
    U64(0x88FCF317, 0xF22241E2), U64(0x441FECE3, 0xBDF81F03), /* ~= 10^196 */
    U64(0xAB3C2FDD, 0xEEAAD25A), U64(0xD527E81C, 0xAD7626C3), /* ~= 10^197 */
    U64(0xD60B3BD5, 0x6A5586F1), U64(0x8A71E223, 0xD8D3B074), /* ~= 10^198 */
    U64(0x85C70565, 0x62757456), U64(0xF6872D56, 0x67844E49), /* ~= 10^199 */
    U64(0xA738C6BE, 0xBB12D16C), U64(0xB428F8AC, 0x016561DB), /* ~= 10^200 */
    U64(0xD106F86E, 0x69D785C7), U64(0xE13336D7, 0x01BEBA52), /* ~= 10^201 */
    U64(0x82A45B45, 0x0226B39C), U64(0xECC00246, 0x61173473), /* ~= 10^202 */
    U64(0xA34D7216, 0x42B06084), U64(0x27F002D7, 0xF95D0190), /* ~= 10^203 */
    U64(0xCC20CE9B, 0xD35C78A5), U64(0x31EC038D, 0xF7B441F4), /* ~= 10^204 */
    U64(0xFF290242, 0xC83396CE), U64(0x7E670471, 0x75A15271), /* ~= 10^205 */
    U64(0x9F79A169, 0xBD203E41), U64(0x0F0062C6, 0xE984D386), /* ~= 10^206 */
    U64(0xC75809C4, 0x2C684DD1), U64(0x52C07B78, 0xA3E60868), /* ~= 10^207 */
    U64(0xF92E0C35, 0x37826145), U64(0xA7709A56, 0xCCDF8A82), /* ~= 10^208 */
    U64(0x9BBCC7A1, 0x42B17CCB), U64(0x88A66076, 0x400BB691), /* ~= 10^209 */
    U64(0xC2ABF989, 0x935DDBFE), U64(0x6ACFF893, 0xD00EA435), /* ~= 10^210 */
    U64(0xF356F7EB, 0xF83552FE), U64(0x0583F6B8, 0xC4124D43), /* ~= 10^211 */
    U64(0x98165AF3, 0x7B2153DE), U64(0xC3727A33, 0x7A8B704A), /* ~= 10^212 */
    U64(0xBE1BF1B0, 0x59E9A8D6), U64(0x744F18C0, 0x592E4C5C), /* ~= 10^213 */
    U64(0xEDA2EE1C, 0x7064130C), U64(0x1162DEF0, 0x6F79DF73), /* ~= 10^214 */
    U64(0x9485D4D1, 0xC63E8BE7), U64(0x8ADDCB56, 0x45AC2BA8), /* ~= 10^215 */
    U64(0xB9A74A06, 0x37CE2EE1), U64(0x6D953E2B, 0xD7173692), /* ~= 10^216 */
    U64(0xE8111C87, 0xC5C1BA99), U64(0xC8FA8DB6, 0xCCDD0437), /* ~= 10^217 */
    U64(0x910AB1D4, 0xDB9914A0), U64(0x1D9C9892, 0x400A22A2), /* ~= 10^218 */
    U64(0xB54D5E4A, 0x127F59C8), U64(0x2503BEB6, 0xD00CAB4B), /* ~= 10^219 */
    U64(0xE2A0B5DC, 0x971F303A), U64(0x2E44AE64, 0x840FD61D), /* ~= 10^220 */
    U64(0x8DA471A9, 0xDE737E24), U64(0x5CEAECFE, 0xD289E5D2), /* ~= 10^221 */
    U64(0xB10D8E14, 0x56105DAD), U64(0x7425A83E, 0x872C5F47), /* ~= 10^222 */
    U64(0xDD50F199, 0x6B947518), U64(0xD12F124E, 0x28F77719), /* ~= 10^223 */
    U64(0x8A5296FF, 0xE33CC92F), U64(0x82BD6B70, 0xD99AAA6F), /* ~= 10^224 */
    U64(0xACE73CBF, 0xDC0BFB7B), U64(0x636CC64D, 0x1001550B), /* ~= 10^225 */
    U64(0xD8210BEF, 0xD30EFA5A), U64(0x3C47F7E0, 0x5401AA4E), /* ~= 10^226 */
    U64(0x8714A775, 0xE3E95C78), U64(0x65ACFAEC, 0x34810A71), /* ~= 10^227 */
    U64(0xA8D9D153, 0x5CE3B396), U64(0x7F1839A7, 0x41A14D0D), /* ~= 10^228 */
    U64(0xD31045A8, 0x341CA07C), U64(0x1EDE4811, 0x1209A050), /* ~= 10^229 */
    U64(0x83EA2B89, 0x2091E44D), U64(0x934AED0A, 0xAB460432), /* ~= 10^230 */
    U64(0xA4E4B66B, 0x68B65D60), U64(0xF81DA84D, 0x5617853F), /* ~= 10^231 */
    U64(0xCE1DE406, 0x42E3F4B9), U64(0x36251260, 0xAB9D668E), /* ~= 10^232 */
    U64(0x80D2AE83, 0xE9CE78F3), U64(0xC1D72B7C, 0x6B426019), /* ~= 10^233 */
    U64(0xA1075A24, 0xE4421730), U64(0xB24CF65B, 0x8612F81F), /* ~= 10^234 */
    U64(0xC94930AE, 0x1D529CFC), U64(0xDEE033F2, 0x6797B627), /* ~= 10^235 */
    U64(0xFB9B7CD9, 0xA4A7443C), U64(0x169840EF, 0x017DA3B1), /* ~= 10^236 */
    U64(0x9D412E08, 0x06E88AA5), U64(0x8E1F2895, 0x60EE864E), /* ~= 10^237 */
    U64(0xC491798A, 0x08A2AD4E), U64(0xF1A6F2BA, 0xB92A27E2), /* ~= 10^238 */
    U64(0xF5B5D7EC, 0x8ACB58A2), U64(0xAE10AF69, 0x6774B1DB), /* ~= 10^239 */
    U64(0x9991A6F3, 0xD6BF1765), U64(0xACCA6DA1, 0xE0A8EF29), /* ~= 10^240 */
    U64(0xBFF610B0, 0xCC6EDD3F), U64(0x17FD090A, 0x58D32AF3), /* ~= 10^241 */
    U64(0xEFF394DC, 0xFF8A948E), U64(0xDDFC4B4C, 0xEF07F5B0), /* ~= 10^242 */
    U64(0x95F83D0A, 0x1FB69CD9), U64(0x4ABDAF10, 0x1564F98E), /* ~= 10^243 */
    U64(0xBB764C4C, 0xA7A4440F), U64(0x9D6D1AD4, 0x1ABE37F1), /* ~= 10^244 */
    U64(0xEA53DF5F, 0xD18D5513), U64(0x84C86189, 0x216DC5ED), /* ~= 10^245 */
    U64(0x92746B9B, 0xE2F8552C), U64(0x32FD3CF5, 0xB4E49BB4), /* ~= 10^246 */
    U64(0xB7118682, 0xDBB66A77), U64(0x3FBC8C33, 0x221DC2A1), /* ~= 10^247 */
    U64(0xE4D5E823, 0x92A40515), U64(0x0FABAF3F, 0xEAA5334A), /* ~= 10^248 */
    U64(0x8F05B116, 0x3BA6832D), U64(0x29CB4D87, 0xF2A7400E), /* ~= 10^249 */
    U64(0xB2C71D5B, 0xCA9023F8), U64(0x743E20E9, 0xEF511012), /* ~= 10^250 */
    U64(0xDF78E4B2, 0xBD342CF6), U64(0x914DA924, 0x6B255416), /* ~= 10^251 */
    U64(0x8BAB8EEF, 0xB6409C1A), U64(0x1AD089B6, 0xC2F7548E), /* ~= 10^252 */
    U64(0xAE9672AB, 0xA3D0C320), U64(0xA184AC24, 0x73B529B1), /* ~= 10^253 */
    U64(0xDA3C0F56, 0x8CC4F3E8), U64(0xC9E5D72D, 0x90A2741E), /* ~= 10^254 */
    U64(0x88658996, 0x17FB1871), U64(0x7E2FA67C, 0x7A658892), /* ~= 10^255 */
    U64(0xAA7EEBFB, 0x9DF9DE8D), U64(0xDDBB901B, 0x98FEEAB7), /* ~= 10^256 */
    U64(0xD51EA6FA, 0x85785631), U64(0x552A7422, 0x7F3EA565), /* ~= 10^257 */
    U64(0x8533285C, 0x936B35DE), U64(0xD53A8895, 0x8F87275F), /* ~= 10^258 */
    U64(0xA67FF273, 0xB8460356), U64(0x8A892ABA, 0xF368F137), /* ~= 10^259 */
    U64(0xD01FEF10, 0xA657842C), U64(0x2D2B7569, 0xB0432D85), /* ~= 10^260 */
    U64(0x8213F56A, 0x67F6B29B), U64(0x9C3B2962, 0x0E29FC73), /* ~= 10^261 */
    U64(0xA298F2C5, 0x01F45F42), U64(0x8349F3BA, 0x91B47B8F), /* ~= 10^262 */
    U64(0xCB3F2F76, 0x42717713), U64(0x241C70A9, 0x36219A73), /* ~= 10^263 */
    U64(0xFE0EFB53, 0xD30DD4D7), U64(0xED238CD3, 0x83AA0110), /* ~= 10^264 */
    U64(0x9EC95D14, 0x63E8A506), U64(0xF4363804, 0x324A40AA), /* ~= 10^265 */
    U64(0xC67BB459, 0x7CE2CE48), U64(0xB143C605, 0x3EDCD0D5), /* ~= 10^266 */
    U64(0xF81AA16F, 0xDC1B81DA), U64(0xDD94B786, 0x8E94050A), /* ~= 10^267 */
    U64(0x9B10A4E5, 0xE9913128), U64(0xCA7CF2B4, 0x191C8326), /* ~= 10^268 */
    U64(0xC1D4CE1F, 0x63F57D72), U64(0xFD1C2F61, 0x1F63A3F0), /* ~= 10^269 */
    U64(0xF24A01A7, 0x3CF2DCCF), U64(0xBC633B39, 0x673C8CEC), /* ~= 10^270 */
    U64(0x976E4108, 0x8617CA01), U64(0xD5BE0503, 0xE085D813), /* ~= 10^271 */
    U64(0xBD49D14A, 0xA79DBC82), U64(0x4B2D8644, 0xD8A74E18), /* ~= 10^272 */
    U64(0xEC9C459D, 0x51852BA2), U64(0xDDF8E7D6, 0x0ED1219E), /* ~= 10^273 */
    U64(0x93E1AB82, 0x52F33B45), U64(0xCABB90E5, 0xC942B503), /* ~= 10^274 */
    U64(0xB8DA1662, 0xE7B00A17), U64(0x3D6A751F, 0x3B936243), /* ~= 10^275 */
    U64(0xE7109BFB, 0xA19C0C9D), U64(0x0CC51267, 0x0A783AD4), /* ~= 10^276 */
    U64(0x906A617D, 0x450187E2), U64(0x27FB2B80, 0x668B24C5), /* ~= 10^277 */
    U64(0xB484F9DC, 0x9641E9DA), U64(0xB1F9F660, 0x802DEDF6), /* ~= 10^278 */
    U64(0xE1A63853, 0xBBD26451), U64(0x5E7873F8, 0xA0396973), /* ~= 10^279 */
    U64(0x8D07E334, 0x55637EB2), U64(0xDB0B487B, 0x6423E1E8), /* ~= 10^280 */
    U64(0xB049DC01, 0x6ABC5E5F), U64(0x91CE1A9A, 0x3D2CDA62), /* ~= 10^281 */
    U64(0xDC5C5301, 0xC56B75F7), U64(0x7641A140, 0xCC7810FB), /* ~= 10^282 */
    U64(0x89B9B3E1, 0x1B6329BA), U64(0xA9E904C8, 0x7FCB0A9D), /* ~= 10^283 */
    U64(0xAC2820D9, 0x623BF429), U64(0x546345FA, 0x9FBDCD44), /* ~= 10^284 */
    U64(0xD732290F, 0xBACAF133), U64(0xA97C1779, 0x47AD4095), /* ~= 10^285 */
    U64(0x867F59A9, 0xD4BED6C0), U64(0x49ED8EAB, 0xCCCC485D), /* ~= 10^286 */
    U64(0xA81F3014, 0x49EE8C70), U64(0x5C68F256, 0xBFFF5A74), /* ~= 10^287 */
    U64(0xD226FC19, 0x5C6A2F8C), U64(0x73832EEC, 0x6FFF3111), /* ~= 10^288 */
    U64(0x83585D8F, 0xD9C25DB7), U64(0xC831FD53, 0xC5FF7EAB), /* ~= 10^289 */
    U64(0xA42E74F3, 0xD032F525), U64(0xBA3E7CA8, 0xB77F5E55), /* ~= 10^290 */
    U64(0xCD3A1230, 0xC43FB26F), U64(0x28CE1BD2, 0xE55F35EB), /* ~= 10^291 */
    U64(0x80444B5E, 0x7AA7CF85), U64(0x7980D163, 0xCF5B81B3), /* ~= 10^292 */
    U64(0xA0555E36, 0x1951C366), U64(0xD7E105BC, 0xC332621F), /* ~= 10^293 */
    U64(0xC86AB5C3, 0x9FA63440), U64(0x8DD9472B, 0xF3FEFAA7), /* ~= 10^294 */
    U64(0xFA856334, 0x878FC150), U64(0xB14F98F6, 0xF0FEB951), /* ~= 10^295 */
    U64(0x9C935E00, 0xD4B9D8D2), U64(0x6ED1BF9A, 0x569F33D3), /* ~= 10^296 */
    U64(0xC3B83581, 0x09E84F07), U64(0x0A862F80, 0xEC4700C8), /* ~= 10^297 */
    U64(0xF4A642E1, 0x4C6262C8), U64(0xCD27BB61, 0x2758C0FA), /* ~= 10^298 */
    U64(0x98E7E9CC, 0xCFBD7DBD), U64(0x8038D51C, 0xB897789C), /* ~= 10^299 */
    U64(0xBF21E440, 0x03ACDD2C), U64(0xE0470A63, 0xE6BD56C3), /* ~= 10^300 */
    U64(0xEEEA5D50, 0x04981478), U64(0x1858CCFC, 0xE06CAC74), /* ~= 10^301 */
    U64(0x95527A52, 0x02DF0CCB), U64(0x0F37801E, 0x0C43EBC8), /* ~= 10^302 */
    U64(0xBAA718E6, 0x8396CFFD), U64(0xD3056025, 0x8F54E6BA), /* ~= 10^303 */
    U64(0xE950DF20, 0x247C83FD), U64(0x47C6B82E, 0xF32A2069), /* ~= 10^304 */
    U64(0x91D28B74, 0x16CDD27E), U64(0x4CDC331D, 0x57FA5441), /* ~= 10^305 */
    U64(0xB6472E51, 0x1C81471D), U64(0xE0133FE4, 0xADF8E952), /* ~= 10^306 */
    U64(0xE3D8F9E5, 0x63A198E5), U64(0x58180FDD, 0xD97723A6), /* ~= 10^307 */
    U64(0x8E679C2F, 0x5E44FF8F), U64(0x570F09EA, 0xA7EA7648), /* ~= 10^308 */
    U64(0xB201833B, 0x35D63F73), U64(0x2CD2CC65, 0x51E513DA), /* ~= 10^309 */
    U64(0xDE81E40A, 0x034BCF4F), U64(0xF8077F7E, 0xA65E58D1), /* ~= 10^310 */
    U64(0x8B112E86, 0x420F6191), U64(0xFB04AFAF, 0x27FAF782), /* ~= 10^311 */
    U64(0xADD57A27, 0xD29339F6), U64(0x79C5DB9A, 0xF1F9B563), /* ~= 10^312 */
    U64(0xD94AD8B1, 0xC7380874), U64(0x18375281, 0xAE7822BC), /* ~= 10^313 */
    U64(0x87CEC76F, 0x1C830548), U64(0x8F229391, 0x0D0B15B5), /* ~= 10^314 */
    U64(0xA9C2794A, 0xE3A3C69A), U64(0xB2EB3875, 0x504DDB22), /* ~= 10^315 */
    U64(0xD433179D, 0x9C8CB841), U64(0x5FA60692, 0xA46151EB), /* ~= 10^316 */
    U64(0x849FEEC2, 0x81D7F328), U64(0xDBC7C41B, 0xA6BCD333), /* ~= 10^317 */
    U64(0xA5C7EA73, 0x224DEFF3), U64(0x12B9B522, 0x906C0800), /* ~= 10^318 */
    U64(0xCF39E50F, 0xEAE16BEF), U64(0xD768226B, 0x34870A00), /* ~= 10^319 */
    U64(0x81842F29, 0xF2CCE375), U64(0xE6A11583, 0x00D46640), /* ~= 10^320 */
    U64(0xA1E53AF4, 0x6F801C53), U64(0x60495AE3, 0xC1097FD0), /* ~= 10^321 */
    U64(0xCA5E89B1, 0x8B602368), U64(0x385BB19C, 0xB14BDFC4), /* ~= 10^322 */
    U64(0xFCF62C1D, 0xEE382C42), U64(0x46729E03, 0xDD9ED7B5), /* ~= 10^323 */
    U64(0x9E19DB92, 0xB4E31BA9), U64(0x6C07A2C2, 0x6A8346D1)  /* ~= 10^324 */
};

/**
 Get the cached pow10 value from `pow10_sig_table`.
 @param exp10 The exponent of pow(10, e). This value must in range
              `POW10_SIG_TABLE_MIN_EXP` to `POW10_SIG_TABLE_MAX_EXP`.
 @param hi    The highest 64 bits of pow(10, e).
 @param lo    The lower 64 bits after `hi`.
 */
static_inline void pow10_table_get_sig(i32 exp10, u64 *hi, u64 *lo) {
    i32 idx = exp10 - (POW10_SIG_TABLE_MIN_EXP);
    *hi = pow10_sig_table[idx * 2];
    *lo = pow10_sig_table[idx * 2 + 1];
}

/**
 Get the exponent (base 2) for highest 64 bits significand in `pow10_sig_table`.
 */
static_inline void pow10_table_get_exp(i32 exp10, i32 *exp2) {
    /* e2 = floor(log2(pow(10, e))) - 64 + 1 */
    /*    = floor(e * log2(10) - 63)         */
    *exp2 = (exp10 * 217706 - 4128768) >> 16;
}

#endif



/*==============================================================================
 * JSON Character Matcher
 *============================================================================*/

/** Character type */
typedef u8 char_type;

/** Whitespace character: ' ', '\\t', '\\n', '\\r'. */
static const char_type CHAR_TYPE_SPACE      = 1 << 0;

/** Number character: '-', [0-9]. */
static const char_type CHAR_TYPE_NUMBER     = 1 << 1;

/** JSON Escaped character: '"', '\', [0x00-0x1F]. */
static const char_type CHAR_TYPE_ESC_ASCII  = 1 << 2;

/** Non-ASCII character: [0x80-0xFF]. */
static const char_type CHAR_TYPE_NON_ASCII  = 1 << 3;

/** JSON container character: '{', '['. */
static const char_type CHAR_TYPE_CONTAINER  = 1 << 4;

/** Comment character: '/'. */
static const char_type CHAR_TYPE_COMMENT    = 1 << 5;

/** Line end character: '\\n', '\\r', '\0'. */
static const char_type CHAR_TYPE_LINE_END   = 1 << 6;

/** Hexadecimal numeric character: [0-9a-fA-F]. */
static const char_type CHAR_TYPE_HEX        = 1 << 7;

/** Character type table (generate with misc/make_tables.c) */
static const char_type char_table[256] = {
    0x44, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x05, 0x45, 0x04, 0x04, 0x45, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x01, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x20,
    0x82, 0x82, 0x82, 0x82, 0x82, 0x82, 0x82, 0x82,
    0x82, 0x82, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x10, 0x04, 0x00, 0x00, 0x00,
    0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08
};

/** Match a character with specified type. */
static_inline bool char_is_type(u8 c, char_type type) {
    return (char_table[c] & type) != 0;
}

/** Match a whitespace: ' ', '\\t', '\\n', '\\r'. */
static_inline bool char_is_space(u8 c) {
    return char_is_type(c, (char_type)CHAR_TYPE_SPACE);
}

/** Match a whitespace or comment: ' ', '\\t', '\\n', '\\r', '/'. */
static_inline bool char_is_space_or_comment(u8 c) {
    return char_is_type(c, (char_type)(CHAR_TYPE_SPACE | CHAR_TYPE_COMMENT));
}

/** Match a JSON number: '-', [0-9]. */
static_inline bool char_is_num(u8 c) {
    return char_is_type(c, (char_type)CHAR_TYPE_NUMBER);
}

/** Match a JSON container: '{', '['. */
static_inline bool char_is_container(u8 c) {
    return char_is_type(c, (char_type)CHAR_TYPE_CONTAINER);
}

/** Match a stop character in ASCII string: '"', '\', [0x00-0x1F,0x80-0xFF]. */
static_inline bool char_is_ascii_stop(u8 c) {
    return char_is_type(c, (char_type)(CHAR_TYPE_ESC_ASCII |
                                       CHAR_TYPE_NON_ASCII));
}

/** Match a line end character: '\\n', '\\r', '\0'. */
static_inline bool char_is_line_end(u8 c) {
    return char_is_type(c, (char_type)CHAR_TYPE_LINE_END);
}

/** Match a hexadecimal numeric character: [0-9a-fA-F]. */
static_inline bool char_is_hex(u8 c) {
    return char_is_type(c, (char_type)CHAR_TYPE_HEX);
}



/*==============================================================================
 * Digit Character Matcher
 *============================================================================*/

/** Digit type */
typedef u8 digi_type;

/** Digit: '0'. */
static const digi_type DIGI_TYPE_ZERO       = 1 << 0;

/** Digit: [1-9]. */
static const digi_type DIGI_TYPE_NONZERO    = 1 << 1;

/** Plus sign (positive): '+'. */
static const digi_type DIGI_TYPE_POS        = 1 << 2;

/** Minus sign (negative): '-'. */
static const digi_type DIGI_TYPE_NEG        = 1 << 3;

/** Decimal point: '.' */
static const digi_type DIGI_TYPE_DOT        = 1 << 4;

/** Exponent sign: 'e, 'E'. */
static const digi_type DIGI_TYPE_EXP        = 1 << 5;

/** Digit type table (generate with misc/make_tables.c) */
static const digi_type digi_table[256] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x10, 0x00,
    0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

/** Match a character with specified type. */
static_inline bool digi_is_type(u8 d, digi_type type) {
    return (digi_table[d] & type) != 0;
}

/** Match a sign: '+', '-' */
static_inline bool digi_is_sign(u8 d) {
    return digi_is_type(d, (digi_type)(DIGI_TYPE_POS | DIGI_TYPE_NEG));
}

/** Match a none zero digit: [1-9] */
static_inline bool digi_is_nonzero(u8 d) {
    return digi_is_type(d, (digi_type)DIGI_TYPE_NONZERO);
}

/** Match a digit: [0-9] */
static_inline bool digi_is_digit(u8 d) {
    return digi_is_type(d, (digi_type)(DIGI_TYPE_ZERO | DIGI_TYPE_NONZERO));
}

/** Match an exponent sign: 'e', 'E'. */
static_inline bool digi_is_exp(u8 d) {
    return digi_is_type(d, (digi_type)DIGI_TYPE_EXP);
}

/** Match a floating point indicator: '.', 'e', 'E'. */
static_inline bool digi_is_fp(u8 d) {
    return digi_is_type(d, (digi_type)(DIGI_TYPE_DOT | DIGI_TYPE_EXP));
}

/** Match a digit or floating point indicator: [0-9], '.', 'e', 'E'. */
static_inline bool digi_is_digit_or_fp(u8 d) {
    return digi_is_type(d, (digi_type)(DIGI_TYPE_ZERO | DIGI_TYPE_NONZERO |
                                       DIGI_TYPE_DOT | DIGI_TYPE_EXP));
}



#if !YYJSON_DISABLE_READER

/*==============================================================================
 * Hex Character Reader
 * This function is used by JSON reader to read escaped characters.
 *============================================================================*/

/**
 This table is used to convert 4 hex character sequence to a number.
 A valid hex character [0-9A-Fa-f] will mapped to it's raw number [0x00, 0x0F],
 an invalid hex character will mapped to [0xF0].
 (generate with misc/make_tables.c)
 */
static const u8 hex_conv_table[256] = {
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0
};

/**
 Scans an escaped character sequence as a UTF-16 code unit (branchless).
 e.g. "\\u005C" should pass "005C" as `cur`.

 This requires the string has 4-byte zero padding.
 */
static_inline bool read_hex_u16(const u8 *cur, u16 *val) {
    u16 c0, c1, c2, c3, t0, t1;
    c0 = hex_conv_table[cur[0]];
    c1 = hex_conv_table[cur[1]];
    c2 = hex_conv_table[cur[2]];
    c3 = hex_conv_table[cur[3]];
    t0 = (u16)((c0 << 8) | c2);
    t1 = (u16)((c1 << 8) | c3);
    *val = (u16)((t0 << 4) | t1);
    return ((t0 | t1) & (u16)0xF0F0) == 0;
}



/*==============================================================================
 * JSON Reader Utils
 * These functions are used by JSON reader to read literals and comments.
 *============================================================================*/

/** Read 'true' literal, '*cur' should be 't'. */
static_inline bool read_true(u8 **ptr, yyjson_val *val) {
    u8 *cur = *ptr;
    if (likely(byte_match_4(cur, "true"))) {
        val->tag = YYJSON_TYPE_BOOL | YYJSON_SUBTYPE_TRUE;
        *ptr = cur + 4;
        return true;
    }
    return false;
}

/** Read 'false' literal, '*cur' should be 'f'. */
static_inline bool read_false(u8 **ptr, yyjson_val *val) {
    u8 *cur = *ptr;
    if (likely(byte_match_4(cur + 1, "alse"))) {
        val->tag = YYJSON_TYPE_BOOL | YYJSON_SUBTYPE_FALSE;
        *ptr = cur + 5;
        return true;
    }
    return false;
}

/** Read 'null' literal, '*cur' should be 'n'. */
static_inline bool read_null(u8 **ptr, yyjson_val *val) {
    u8 *cur = *ptr;
    if (likely(byte_match_4(cur, "null"))) {
        val->tag = YYJSON_TYPE_NULL;
        *ptr = cur + 4;
        return true;
    }
    return false;
}

/** Read 'Inf' or 'Infinity' literal (ignoring case). */
static_inline bool read_inf(bool sign, u8 **ptr, u8 **pre,
                            yyjson_read_flag flg, yyjson_val *val) {
    u8 *hdr = *ptr - sign;
    u8 *cur = *ptr;
    u8 **end = ptr;
    if ((cur[0] == 'I' || cur[0] == 'i') &&
        (cur[1] == 'N' || cur[1] == 'n') &&
        (cur[2] == 'F' || cur[2] == 'f')) {
        if (cur[3] == 'I' || cur[3] == 'i') {
            if ((cur[4] == 'N' || cur[4] == 'n') &&
                (cur[5] == 'I' || cur[5] == 'i') &&
                (cur[6] == 'T' || cur[6] == 't') &&
                (cur[7] == 'Y' || cur[7] == 'y')) {
                cur += 8;
            } else {
                /* Don't accept INF as a complete value if it's followed by I.
                   This is to better support incremental parsing. */
                return false;
            }
        } else {
            cur += 3;
        }
        *end = cur;
        if (has_read_flag(NUMBER_AS_RAW)) {
            /* add null-terminator for previous raw string */
            if (*pre) **pre = '\0';
            *pre = cur;
            val->tag = ((u64)(cur - hdr) << YYJSON_TAG_BIT) | YYJSON_TYPE_RAW;
            val->uni.str = (const char *)hdr;
        } else {
            val->tag = YYJSON_TYPE_NUM | YYJSON_SUBTYPE_REAL;
            val->uni.u64 = f64_raw_get_inf(sign);
        }
        return true;
    }
    return false;
}

/** Read 'NaN' literal (ignoring case). */
static_inline bool read_nan(bool sign, u8 **ptr, u8 **pre,
                            yyjson_read_flag flg, yyjson_val *val) {
    u8 *hdr = *ptr - sign;
    u8 *cur = *ptr;
    u8 **end = ptr;
    if ((cur[0] == 'N' || cur[0] == 'n') &&
        (cur[1] == 'A' || cur[1] == 'a') &&
        (cur[2] == 'N' || cur[2] == 'n')) {
        cur += 3;
        *end = cur;
        if (has_read_flag(NUMBER_AS_RAW)) {
            /* add null-terminator for previous raw string */
            if (*pre) **pre = '\0';
            *pre = cur;
            val->tag = ((u64)(cur - hdr) << YYJSON_TAG_BIT) | YYJSON_TYPE_RAW;
            val->uni.str = (const char *)hdr;
        } else {
            val->tag = YYJSON_TYPE_NUM | YYJSON_SUBTYPE_REAL;
            val->uni.u64 = f64_raw_get_nan(sign);
        }
        return true;
    }
    return false;
}

/** Read 'Inf', 'Infinity' or 'NaN' literal (ignoring case). */
static_inline bool read_inf_or_nan(bool sign, u8 **ptr, u8 **pre,
                                   yyjson_read_flag flg, yyjson_val *val) {
    if (read_inf(sign, ptr, pre, flg, val)) return true;
    if (read_nan(sign, ptr, pre, flg, val)) return true;
    return false;
}

/** Read a JSON number as raw string. */
static_noinline bool read_num_raw(u8 **ptr, u8 **pre, yyjson_read_flag flg,
                                  yyjson_val *val, const char **msg) {

#define return_err(_pos, _msg) do { \
    *msg = _msg; \
    *end = _pos; \
    return false; \
} while (false)

#define return_raw() do { \
    val->tag = ((u64)(cur - hdr) << YYJSON_TAG_BIT) | YYJSON_TYPE_RAW; \
    val->uni.str = (const char *)hdr; \
    *pre = cur; *end = cur; return true; \
} while (false)

    u8 *hdr = *ptr;
    u8 *cur = *ptr;
    u8 **end = ptr;

    /* add null-terminator for previous raw string */
    if (*pre) **pre = '\0';

    /* skip sign */
    cur += (*cur == '-');

    /* read first digit, check leading zero */
    if (unlikely(!digi_is_digit(*cur))) {
        if (has_read_flag(ALLOW_INF_AND_NAN)) {
            if (read_inf_or_nan(*hdr == '-', &cur, pre, flg, val)) return_raw();
        }
        return_err(cur, "no digit after minus sign");
    }

    /* read integral part */
    if (*cur == '0') {
        cur++;
        if (unlikely(digi_is_digit(*cur))) {
            return_err(cur - 1, "number with leading zero is not allowed");
        }
        if (!digi_is_fp(*cur)) return_raw();
    } else {
        while (digi_is_digit(*cur)) cur++;
        if (!digi_is_fp(*cur)) return_raw();
    }

    /* read fraction part */
    if (*cur == '.') {
        cur++;
        if (!digi_is_digit(*cur++)) {
            return_err(cur, "no digit after decimal point");
        }
        while (digi_is_digit(*cur)) cur++;
    }

    /* read exponent part */
    if (digi_is_exp(*cur)) {
        cur += 1 + digi_is_sign(cur[1]);
        if (!digi_is_digit(*cur++)) {
            return_err(cur, "no digit after exponent sign");
        }
        while (digi_is_digit(*cur)) cur++;
    }

    return_raw();

#undef return_err
#undef return_raw
}

/**
 Skips spaces and comments as many as possible.

 It will return false in these cases:
    1. No character is skipped. The 'end' pointer is set as input cursor.
    2. A multiline comment is not closed. The 'end' pointer is set as the head
       of this comment block.
 */
static_noinline bool skip_spaces_and_comments(u8 **ptr) {
    u8 *hdr = *ptr;
    u8 *cur = *ptr;
    u8 **end = ptr;
    while (true) {
        if (byte_match_2(cur, "/*")) {
            hdr = cur;
            cur += 2;
            while (true) {
                if (byte_match_2(cur, "*/")) {
                    cur += 2;
                    break;
                }
                if (*cur == 0) {
                    *end = hdr;
                    return false;
                }
                cur++;
            }
            continue;
        }
        if (byte_match_2(cur, "//")) {
            cur += 2;
            while (!char_is_line_end(*cur)) cur++;
            continue;
        }
        if (char_is_space(*cur)) {
            cur += 1;
            while (char_is_space(*cur)) cur++;
            continue;
        }
        break;
    }
    *end = cur;
    return hdr != cur;
}

/**
 Check truncated string.
 Returns true if `cur` match `str` but is truncated.
 */
static_inline bool is_truncated_str(u8 *cur, u8 *end,
                                    const char *str,
                                    bool case_sensitive) {
    usize len = strlen(str);
    if (cur + len <= end || end <= cur) return false;
    if (case_sensitive) {
        return memcmp(cur, str, (usize)(end - cur)) == 0;
    }
    for (; cur < end; cur++, str++) {
        if ((*cur != (u8)*str) && (*cur != (u8)*str - 'a' + 'A')) {
            return false;
        }
    }
    return true;
}

/**
 Check truncated JSON on parsing errors.
 Returns true if the input is valid but truncated.
 */
static_noinline bool is_truncated_end(u8 *hdr, u8 *cur, u8 *end,
                                      yyjson_read_code code,
                                      yyjson_read_flag flg) {
    if (cur >= end) return true;
    if (code == YYJSON_READ_ERROR_LITERAL) {
        if (is_truncated_str(cur, end, "true", true) ||
            is_truncated_str(cur, end, "false", true) ||
            is_truncated_str(cur, end, "null", true)) {
            return true;
        }
    }
    if (code == YYJSON_READ_ERROR_UNEXPECTED_CHARACTER ||
        code == YYJSON_READ_ERROR_INVALID_NUMBER ||
        code == YYJSON_READ_ERROR_LITERAL) {
        if (has_read_flag(ALLOW_INF_AND_NAN)) {
            if (*cur == '-') cur++;
            if (is_truncated_str(cur, end, "infinity", false) ||
                is_truncated_str(cur, end, "nan", false)) {
                return true;
            }
        }
    }
    if (code == YYJSON_READ_ERROR_UNEXPECTED_CONTENT) {
        if (has_read_flag(ALLOW_INF_AND_NAN)) {
            if (hdr + 3 <= cur &&
                is_truncated_str(cur - 3, end, "infinity", false)) {
                return true; /* e.g. infin would be read as inf + in */
            }
        }
    }
    if (code == YYJSON_READ_ERROR_INVALID_STRING) {
        usize len = (usize)(end - cur);

        /* unicode escape sequence */
        if (*cur == '\\') {
            if (len == 1) return true;
            if (len <= 5) {
                if (*++cur != 'u') return false;
                for (++cur; cur < end; cur++) {
                    if (!char_is_hex(*cur)) return false;
                }
                return true;
            } else if (len <= 11) {
                /* incomplete surrogate pair? */
                u16 hi;
                if (*++cur != 'u') return false;
                if (!read_hex_u16(++cur, &hi)) return false;
                if ((hi & 0xF800) != 0xD800) return false;
                cur += 4;
                if (cur >= end) return true;
                /* valid low surrogate is DC00...DFFF */
                if (*cur != '\\') return false;
                if (++cur >= end) return true;
                if (*cur != 'u') return false;
                if (++cur >= end) return true;
                if (*cur != 'd' && *cur != 'D') return false;
                if (++cur >= end) return true;
                if ((*cur < 'c' || *cur > 'f') && (*cur < 'C' || *cur > 'F'))
                    return false;
                if (++cur >= end) return true;
                if (!char_is_hex(*cur)) return false;
                return true;
            }
            return false;
        }

        /* 2 to 4 bytes UTF-8, see `read_str()` for details. */
        if (*cur & 0x80) {
            u8 c0 = cur[0], c1 = cur[1], c2 = cur[2];
            if (len == 1) {
                /* 2 bytes UTF-8, truncated */
                if ((c0 & 0xE0) == 0xC0 && (c0 & 0x1E) != 0x00) return true;
                /* 3 bytes UTF-8, truncated */
                if ((c0 & 0xF0) == 0xE0) return true;
                /* 4 bytes UTF-8, truncated */
                if ((c0 & 0xF8) == 0xF0 && (c0 & 0x07) <= 0x04) return true;
            }
            if (len == 2) {
                /* 3 bytes UTF-8, truncated */
                if ((c0 & 0xF0) == 0xE0 &&
                    (c1 & 0xC0) == 0x80) {
                    u8 pat = (u8)(((c0 & 0x0F) << 1) | ((c1 & 0x20) >> 5));
                    return 0x01 <= pat && pat != 0x1B;
                }
                /* 4 bytes UTF-8, truncated */
                if ((c0 & 0xF8) == 0xF0 &&
                    (c1 & 0xC0) == 0x80) {
                    u8 pat = (u8)(((c0 & 0x07) << 2) | ((c1 & 0x30) >> 4));
                    return 0x01 <= pat && pat <= 0x10;
                }
            }
            if (len == 3) {
                /* 4 bytes UTF-8, truncated */
                if ((c0 & 0xF8) == 0xF0 &&
                    (c1 & 0xC0) == 0x80 &&
                    (c2 & 0xC0) == 0x80) {
                    u8 pat = (u8)(((c0 & 0x07) << 2) | ((c1 & 0x30) >> 4));
                    return 0x01 <= pat && pat <= 0x10;
                }
            }
        }
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (code == YYJSON_READ_ERROR_INVALID_COMMENT) {
            /* unclosed multiline comment */
            return true;
        }
        if (code == YYJSON_READ_ERROR_UNEXPECTED_CHARACTER &&
            *cur == '/' && cur + 1 == end) {
            /* truncated beginning of comment */
            return true;
        }
    }
    if (code == YYJSON_READ_ERROR_UNEXPECTED_CHARACTER &&
        has_read_flag(ALLOW_BOM)) {
        /* truncated UTF-8 BOM */
        usize len = (usize)(end - cur);
        if (cur == hdr && len < 3 && !memcmp(hdr, "\xEF\xBB\xBF", len)) {
            return true;
        }
    }
    return false;
}



#if YYJSON_HAS_IEEE_754 && !YYJSON_DISABLE_FAST_FP_CONV /* FP_READER */

/*==============================================================================
 * BigInt For Floating Point Number Reader
 *
 * The bigint algorithm is used by floating-point number reader to get correctly
 * rounded result for numbers with lots of digits. This part of code is rarely
 * used for common numbers.
 *============================================================================*/

/** Maximum exponent of exact pow10 */
#define U64_POW10_MAX_EXP 19

/** Table: [ 10^0, ..., 10^19 ] (generate with misc/make_tables.c) */
static const u64 u64_pow10_table[U64_POW10_MAX_EXP + 1] = {
    U64(0x00000000, 0x00000001), U64(0x00000000, 0x0000000A),
    U64(0x00000000, 0x00000064), U64(0x00000000, 0x000003E8),
    U64(0x00000000, 0x00002710), U64(0x00000000, 0x000186A0),
    U64(0x00000000, 0x000F4240), U64(0x00000000, 0x00989680),
    U64(0x00000000, 0x05F5E100), U64(0x00000000, 0x3B9ACA00),
    U64(0x00000002, 0x540BE400), U64(0x00000017, 0x4876E800),
    U64(0x000000E8, 0xD4A51000), U64(0x00000918, 0x4E72A000),
    U64(0x00005AF3, 0x107A4000), U64(0x00038D7E, 0xA4C68000),
    U64(0x002386F2, 0x6FC10000), U64(0x01634578, 0x5D8A0000),
    U64(0x0DE0B6B3, 0xA7640000), U64(0x8AC72304, 0x89E80000)
};

/** Maximum numbers of chunks used by a bigint (58 is enough here). */
#define BIGINT_MAX_CHUNKS 64

/** Unsigned arbitrarily large integer */
typedef struct bigint {
    u32 used; /* used chunks count, should not be 0 */
    u64 bits[BIGINT_MAX_CHUNKS]; /* chunks */
} bigint;

/**
 Evaluate 'big += val'.
 @param big A big number (can be 0).
 @param val An unsigned integer (can be 0).
 */
static_inline void bigint_add_u64(bigint *big, u64 val) {
    u32 idx, max;
    u64 num = big->bits[0];
    u64 add = num + val;
    big->bits[0] = add;
    if (likely((add >= num) || (add >= val))) return;
    for ((void)(idx = 1), max = big->used; idx < max; idx++) {
        if (likely(big->bits[idx] != U64_MAX)) {
            big->bits[idx] += 1;
            return;
        }
        big->bits[idx] = 0;
    }
    big->bits[big->used++] = 1;
}

/**
 Evaluate 'big *= val'.
 @param big A big number (can be 0).
 @param val An unsigned integer (cannot be 0).
 */
static_inline void bigint_mul_u64(bigint *big, u64 val) {
    u32 idx = 0, max = big->used;
    u64 hi, lo, carry = 0;
    for (; idx < max; idx++) {
        if (big->bits[idx]) break;
    }
    for (; idx < max; idx++) {
        u128_mul_add(big->bits[idx], val, carry, &hi, &lo);
        big->bits[idx] = lo;
        carry = hi;
    }
    if (carry) big->bits[big->used++] = carry;
}

/**
 Evaluate 'big *= 2^exp'.
 @param big A big number (can be 0).
 @param exp An exponent integer (can be 0).
 */
static_inline void bigint_mul_pow2(bigint *big, u32 exp) {
    u32 shft = exp % 64;
    u32 move = exp / 64;
    u32 idx = big->used;
    if (unlikely(shft == 0)) {
        for (; idx > 0; idx--) {
            big->bits[idx + move - 1] = big->bits[idx - 1];
        }
        big->used += move;
        while (move) big->bits[--move] = 0;
    } else {
        big->bits[idx] = 0;
        for (; idx > 0; idx--) {
            u64 num = big->bits[idx] << shft;
            num |= big->bits[idx - 1] >> (64 - shft);
            big->bits[idx + move] = num;
        }
        big->bits[move] = big->bits[0] << shft;
        big->used += move + (big->bits[big->used + move] > 0);
        while (move) big->bits[--move] = 0;
    }
}

/**
 Evaluate 'big *= 10^exp'.
 @param big A big number (can be 0).
 @param exp An exponent integer (cannot be 0).
 */
static_inline void bigint_mul_pow10(bigint *big, i32 exp) {
    for (; exp >= U64_POW10_MAX_EXP; exp -= U64_POW10_MAX_EXP) {
        bigint_mul_u64(big, u64_pow10_table[U64_POW10_MAX_EXP]);
    }
    if (exp) {
        bigint_mul_u64(big, u64_pow10_table[exp]);
    }
}

/**
 Compare two bigint.
 @return -1 if 'a < b', +1 if 'a > b', 0 if 'a == b'.
 */
static_inline i32 bigint_cmp(bigint *a, bigint *b) {
    u32 idx = a->used;
    if (a->used < b->used) return -1;
    if (a->used > b->used) return +1;
    while (idx-- > 0) {
        u64 av = a->bits[idx];
        u64 bv = b->bits[idx];
        if (av < bv) return -1;
        if (av > bv) return +1;
    }
    return 0;
}

/**
 Evaluate 'big = val'.
 @param big A big number (can be 0).
 @param val An unsigned integer (can be 0).
 */
static_inline void bigint_set_u64(bigint *big, u64 val) {
    big->used = 1;
    big->bits[0] = val;
}

/** Set a bigint with floating point number string. */
static_noinline void bigint_set_buf(bigint *big, u64 sig, i32 *exp,
                                    u8 *sig_cut, u8 *sig_end, u8 *dot_pos) {

    if (unlikely(!sig_cut)) {
        /* no digit cut, set significant part only */
        bigint_set_u64(big, sig);
        return;

    } else {
        /* some digits were cut, read them from 'sig_cut' to 'sig_end' */
        u8 *hdr = sig_cut;
        u8 *cur = hdr;
        u32 len = 0;
        u64 val = 0;
        bool dig_big_cut = false;
        bool has_dot = (hdr < dot_pos) & (dot_pos < sig_end);
        u32 dig_len_total = U64_SAFE_DIG + (u32)(sig_end - hdr) - has_dot;

        sig -= (*sig_cut >= '5'); /* sig was rounded before */
        if (dig_len_total > F64_MAX_DEC_DIG) {
            dig_big_cut = true;
            sig_end -= dig_len_total - (F64_MAX_DEC_DIG + 1);
            sig_end -= (dot_pos + 1 == sig_end);
            dig_len_total = (F64_MAX_DEC_DIG + 1);
        }
        *exp -= (i32)dig_len_total - U64_SAFE_DIG;

        big->used = 1;
        big->bits[0] = sig;
        while (cur < sig_end) {
            if (likely(cur != dot_pos)) {
                val = val * 10 + (u8)(*cur++ - '0');
                len++;
                if (unlikely(cur == sig_end && dig_big_cut)) {
                    /* The last digit must be non-zero,    */
                    /* set it to '1' for correct rounding. */
                    val = val - (val % 10) + 1;
                }
                if (len == U64_SAFE_DIG || cur == sig_end) {
                    bigint_mul_pow10(big, (i32)len);
                    bigint_add_u64(big, val);
                    val = 0;
                    len = 0;
                }
            } else {
                cur++;
            }
        }
    }
}



/*==============================================================================
 * Diy Floating Point
 *============================================================================*/

/** "Do It Yourself Floating Point" struct. */
typedef struct diy_fp {
    u64 sig; /* significand */
    i32 exp; /* exponent, base 2 */
    i32 pad; /* padding, useless */
} diy_fp;

/** Get cached rounded diy_fp with pow(10, e) The input value must in range
    [POW10_SIG_TABLE_MIN_EXP, POW10_SIG_TABLE_MAX_EXP]. */
static_inline diy_fp diy_fp_get_cached_pow10(i32 exp10) {
    diy_fp fp;
    u64 sig_ext;
    pow10_table_get_sig(exp10, &fp.sig, &sig_ext);
    pow10_table_get_exp(exp10, &fp.exp);
    fp.sig += (sig_ext >> 63);
    return fp;
}

/** Returns fp * fp2. */
static_inline diy_fp diy_fp_mul(diy_fp fp, diy_fp fp2) {
    u64 hi, lo;
    u128_mul(fp.sig, fp2.sig, &hi, &lo);
    fp.sig = hi + (lo >> 63);
    fp.exp += fp2.exp + 64;
    return fp;
}

/** Convert diy_fp to IEEE-754 raw value. */
static_inline u64 diy_fp_to_ieee_raw(diy_fp fp) {
    u64 sig = fp.sig;
    i32 exp = fp.exp;
    u32 lz_bits;
    if (unlikely(fp.sig == 0)) return 0;

    lz_bits = u64_lz_bits(sig);
    sig <<= lz_bits;
    sig >>= F64_BITS - F64_SIG_FULL_BITS;
    exp -= (i32)lz_bits;
    exp += F64_BITS - F64_SIG_FULL_BITS;
    exp += F64_SIG_BITS;

    if (unlikely(exp >= F64_MAX_BIN_EXP)) {
        /* overflow */
        return F64_RAW_INF;
    } else if (likely(exp >= F64_MIN_BIN_EXP - 1)) {
        /* normal */
        exp += F64_EXP_BIAS;
        return ((u64)exp << F64_SIG_BITS) | (sig & F64_SIG_MASK);
    } else if (likely(exp >= F64_MIN_BIN_EXP - F64_SIG_FULL_BITS)) {
        /* subnormal */
        return sig >> (F64_MIN_BIN_EXP - exp - 1);
    } else {
        /* underflow */
        return 0;
    }
}



/*==============================================================================
 * JSON Number Reader (IEEE-754)
 *============================================================================*/

/** Maximum exact pow10 exponent for double value. */
#define F64_POW10_EXP_MAX_EXACT 22

#if YYJSON_DOUBLE_MATH_CORRECT
/** Cached pow10 table. */
static const f64 f64_pow10_table[] = {
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12,
    1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22
};
#endif

/**
 Read a JSON number.

 1. This function assume that the floating-point number is in IEEE-754 format.
 2. This function support uint64/int64/double number. If an integer number
    cannot fit in uint64/int64, it will returns as a double number. If a double
    number is infinite, the return value is based on flag.
 3. This function (with inline attribute) may generate a lot of instructions.
 */
static_inline bool read_num(u8 **ptr, u8 **pre, yyjson_read_flag flg,
                            yyjson_val *val, const char **msg) {

#define return_err(_pos, _msg) do { \
    *msg = _msg; \
    *end = _pos; \
    return false; \
} while (false)

#define return_0() do { \
    val->tag = YYJSON_TYPE_NUM | (u8)((u8)sign << 3); \
    val->uni.u64 = 0; \
    *end = cur; return true; \
} while (false)

#define return_i64(_v) do { \
    val->tag = YYJSON_TYPE_NUM | (u8)((u8)sign << 3); \
    val->uni.u64 = (u64)(sign ? (u64)(~(_v) + 1) : (u64)(_v)); \
    *end = cur; return true; \
} while (false)

#define return_f64(_v) do { \
    val->tag = YYJSON_TYPE_NUM | YYJSON_SUBTYPE_REAL; \
    val->uni.f64 = sign ? -(f64)(_v) : (f64)(_v); \
    *end = cur; return true; \
} while (false)

#define return_f64_bin(_v) do { \
    val->tag = YYJSON_TYPE_NUM | YYJSON_SUBTYPE_REAL; \
    val->uni.u64 = ((u64)sign << 63) | (u64)(_v); \
    *end = cur; return true; \
} while (false)

#define return_inf() do { \
    if (has_read_flag(BIGNUM_AS_RAW)) return_raw(); \
    if (has_read_flag(ALLOW_INF_AND_NAN)) return_f64_bin(F64_RAW_INF); \
    else return_err(hdr, "number is infinity when parsed as double"); \
} while (false)

#define return_raw() do { \
    if (*pre) **pre = '\0'; /* add null-terminator for previous raw string */ \
    val->tag = ((u64)(cur - hdr) << YYJSON_TAG_BIT) | YYJSON_TYPE_RAW; \
    val->uni.str = (const char *)hdr; \
    *pre = cur; *end = cur; return true; \
} while (false)

    u8 *sig_cut = NULL; /* significant part cutting position for long number */
    u8 *sig_end = NULL; /* significant part ending position */
    u8 *dot_pos = NULL; /* decimal point position */

    u64 sig = 0; /* significant part of the number */
    i32 exp = 0; /* exponent part of the number */

    bool exp_sign; /* temporary exponent sign from literal part */
    i64 exp_sig = 0; /* temporary exponent number from significant part */
    i64 exp_lit = 0; /* temporary exponent number from exponent literal part */
    u64 num; /* temporary number for reading */
    u8 *tmp; /* temporary cursor for reading */

    u8 *hdr = *ptr;
    u8 *cur = *ptr;
    u8 **end = ptr;
    bool sign;

    /* read number as raw string if has `YYJSON_READ_NUMBER_AS_RAW` flag */
    if (has_read_flag(NUMBER_AS_RAW)) {
        return read_num_raw(ptr, pre, flg, val, msg);
    }

    sign = (*hdr == '-');
    cur += sign;

    /* begin with a leading zero or non-digit */
    if (unlikely(!digi_is_nonzero(*cur))) { /* 0 or non-digit char */
        if (unlikely(*cur != '0')) { /* non-digit char */
            if (has_read_flag(ALLOW_INF_AND_NAN)) {
                if (read_inf_or_nan(sign, &cur, pre, flg, val)) {
                    *end = cur;
                    return true;
                }
            }
            return_err(cur, "no digit after minus sign");
        }
        /* begin with 0 */
        if (likely(!digi_is_digit_or_fp(*++cur))) return_0();
        if (likely(*cur == '.')) {
            dot_pos = cur++;
            if (unlikely(!digi_is_digit(*cur))) {
                return_err(cur, "no digit after decimal point");
            }
            while (unlikely(*cur == '0')) cur++;
            if (likely(digi_is_digit(*cur))) {
                /* first non-zero digit after decimal point */
                sig = (u64)(*cur - '0'); /* read first digit */
                cur--;
                goto digi_frac_1; /* continue read fraction part */
            }
        }
        if (unlikely(digi_is_digit(*cur))) {
            return_err(cur - 1, "number with leading zero is not allowed");
        }
        if (unlikely(digi_is_exp(*cur))) { /* 0 with any exponent is still 0 */
            cur += (usize)1 + digi_is_sign(cur[1]);
            if (unlikely(!digi_is_digit(*cur))) {
                return_err(cur, "no digit after exponent sign");
            }
            while (digi_is_digit(*++cur));
        }
        return_f64_bin(0);
    }

    /* begin with non-zero digit */
    sig = (u64)(*cur - '0');

    /*
     Read integral part, same as the following code.

         for (int i = 1; i <= 18; i++) {
            num = cur[i] - '0';
            if (num <= 9) sig = num + sig * 10;
            else goto digi_sepr_i;
         }
     */
#define expr_intg(i) \
    if (likely((num = (u64)(cur[i] - (u8)'0')) <= 9)) sig = num + sig * 10; \
    else { goto digi_sepr_##i; }
    repeat_in_1_18(expr_intg)
#undef expr_intg


    cur += 19; /* skip continuous 19 digits */
    if (!digi_is_digit_or_fp(*cur)) {
        /* this number is an integer consisting of 19 digits */
        if (sign && (sig > ((u64)1 << 63))) { /* overflow */
            if (has_read_flag(BIGNUM_AS_RAW)) return_raw();
            return_f64(unsafe_yyjson_u64_to_f64(sig));
        }
        return_i64(sig);
    }
    goto digi_intg_more; /* read more digits in integral part */


    /* process first non-digit character */
#define expr_sepr(i) \
    digi_sepr_##i: \
    if (likely(!digi_is_fp(cur[i]))) { cur += i; return_i64(sig); } \
    dot_pos = cur + i; \
    if (likely(cur[i] == '.')) goto digi_frac_##i; \
    cur += i; sig_end = cur; goto digi_exp_more;
    repeat_in_1_18(expr_sepr)
#undef expr_sepr


    /* read fraction part */
#define expr_frac(i) \
    digi_frac_##i: \
    if (likely((num = (u64)(cur[i + 1] - (u8)'0')) <= 9)) \
        sig = num + sig * 10; \
    else { goto digi_stop_##i; }
    repeat_in_1_18(expr_frac)
#undef expr_frac

    cur += 20; /* skip 19 digits and 1 decimal point */
    if (!digi_is_digit(*cur)) goto digi_frac_end; /* fraction part end */
    goto digi_frac_more; /* read more digits in fraction part */


    /* significant part end */
#define expr_stop(i) \
    digi_stop_##i: \
    cur += i + 1; \
    goto digi_frac_end;
    repeat_in_1_18(expr_stop)
#undef expr_stop


    /* read more digits in integral part */
digi_intg_more:
    if (digi_is_digit(*cur)) {
        if (!digi_is_digit_or_fp(cur[1])) {
            /* this number is an integer consisting of 20 digits */
            num = (u64)(*cur - '0');
            if ((sig < (U64_MAX / 10)) ||
                (sig == (U64_MAX / 10) && num <= (U64_MAX % 10))) {
                sig = num + sig * 10;
                cur++;
                /* convert to double if overflow */
                if (sign) {
                    if (has_read_flag(BIGNUM_AS_RAW)) return_raw();
                    return_f64(unsafe_yyjson_u64_to_f64(sig));
                }
                return_i64(sig);
            }
        }
    }

    if (digi_is_exp(*cur)) {
        dot_pos = cur;
        goto digi_exp_more;
    }

    if (*cur == '.') {
        dot_pos = cur++;
        if (!digi_is_digit(*cur)) {
            return_err(cur, "no digit after decimal point");
        }
    }


    /* read more digits in fraction part */
digi_frac_more:
    sig_cut = cur; /* too large to fit in u64, excess digits need to be cut */
    sig += (*cur >= '5'); /* round */
    while (digi_is_digit(*++cur));
    if (!dot_pos) {
        if (!digi_is_fp(*cur) && has_read_flag(BIGNUM_AS_RAW)) {
            return_raw(); /* it's a large integer */
        }
        dot_pos = cur;
        if (*cur == '.') {
            if (!digi_is_digit(*++cur)) {
                return_err(cur, "no digit after decimal point");
            }
            while (digi_is_digit(*cur)) cur++;
        }
    }
    exp_sig = (i64)(dot_pos - sig_cut);
    exp_sig += (dot_pos < sig_cut);

    /* ignore trailing zeros */
    tmp = cur - 1;
    while (*tmp == '0' || *tmp == '.') tmp--;
    if (tmp < sig_cut) {
        sig_cut = NULL;
    } else {
        sig_end = cur;
    }

    if (digi_is_exp(*cur)) goto digi_exp_more;
    goto digi_exp_finish;


    /* fraction part end */
digi_frac_end:
    if (unlikely(dot_pos + 1 == cur)) {
        return_err(cur, "no digit after decimal point");
    }
    sig_end = cur;
    exp_sig = -(i64)((u64)(cur - dot_pos) - 1);
    if (likely(!digi_is_exp(*cur))) {
        if (unlikely(exp_sig < F64_MIN_DEC_EXP - 19)) {
            return_f64_bin(0); /* underflow */
        }
        exp = (i32)exp_sig;
        goto digi_finish;
    } else {
        goto digi_exp_more;
    }


    /* read exponent part */
digi_exp_more:
    exp_sign = (*++cur == '-');
    cur += digi_is_sign(*cur);
    if (unlikely(!digi_is_digit(*cur))) {
        return_err(cur, "no digit after exponent sign");
    }
    while (*cur == '0') cur++;

    /* read exponent literal */
    tmp = cur;
    while (digi_is_digit(*cur)) {
        exp_lit = (i64)((u8)(*cur++ - '0') + (u64)exp_lit * 10);
    }
    if (unlikely(cur - tmp >= U64_SAFE_DIG)) {
        if (exp_sign) {
            return_f64_bin(0); /* underflow */
        } else {
            return_inf(); /* overflow */
        }
    }
    exp_sig += exp_sign ? -exp_lit : exp_lit;


    /* validate exponent value */
digi_exp_finish:
    if (unlikely(exp_sig < F64_MIN_DEC_EXP - 19)) {
        return_f64_bin(0); /* underflow */
    }
    if (unlikely(exp_sig > F64_MAX_DEC_EXP)) {
        return_inf(); /* overflow */
    }
    exp = (i32)exp_sig;


    /* all digit read finished */
digi_finish:

    /*
     Fast path 1:

     1. The floating-point number calculation should be accurate, see the
        comments of macro `YYJSON_DOUBLE_MATH_CORRECT`.
     2. Correct rounding should be performed (fegetround() == FE_TONEAREST).
     3. The input of floating point number calculation does not lose precision,
        which means: 64 - leading_zero(input) - trailing_zero(input) < 53.

     We don't check all available inputs here, because that would make the code
     more complicated, and not friendly to branch predictor.
     */
#if YYJSON_DOUBLE_MATH_CORRECT
    if (sig < ((u64)1 << 53) &&
        exp >= -F64_POW10_EXP_MAX_EXACT &&
        exp <= +F64_POW10_EXP_MAX_EXACT) {
        f64 dbl = (f64)sig;
        if (exp < 0) {
            dbl /= f64_pow10_table[-exp];
        } else {
            dbl *= f64_pow10_table[+exp];
        }
        return_f64(dbl);
    }
#endif

    /*
     Fast path 2:

     To keep it simple, we only accept normal number here,
     let the slow path to handle subnormal and infinity number.
     */
    if (likely(!sig_cut &&
               exp > -F64_MAX_DEC_EXP + 1 &&
               exp < +F64_MAX_DEC_EXP - 20)) {
        /*
         The result value is exactly equal to (sig * 10^exp),
         the exponent part (10^exp) can be converted to (sig2 * 2^exp2).

         The sig2 can be an infinite length number, only the highest 128 bits
         is cached in the pow10_sig_table.

         Now we have these bits:
         sig1 (normalized 64bit)        : aaaaaaaa
         sig2 (higher 64bit)            : bbbbbbbb
         sig2_ext (lower 64bit)         : cccccccc
         sig2_cut (extra unknown bits)  : dddddddddddd....

         And the calculation process is:
         ----------------------------------------
                 aaaaaaaa *
                 bbbbbbbbccccccccdddddddddddd....
         ----------------------------------------
         abababababababab +
                 acacacacacacacac +
                         adadadadadadadadadad....
         ----------------------------------------
         [hi____][lo____] +
                 [hi2___][lo2___] +
                         [unknown___________....]
         ----------------------------------------

         The addition with carry may affect higher bits, but if there is a 0
         in higher bits, the bits higher than 0 will not be affected.

         `lo2` + `unknown` may get a carry bit and may affect `hi2`, the max
         value of `hi2` is 0xFFFFFFFFFFFFFFFE, so `hi2` will not overflow.

         `lo` + `hi2` may also get a carry bit and may affect `hi`, but only
         the highest significant 53 bits of `hi` is needed. If there is a 0
         in the lower bits of `hi`, then all the following bits can be dropped.

         To convert the result to IEEE-754 double number, we need to perform
         correct rounding:
         1. if bit 54 is 0, round down,
         2. if bit 54 is 1 and any bit beyond bit 54 is 1, round up,
         3. if bit 54 is 1 and all bits beyond bit 54 are 0, round to even,
            as the extra bits is unknown, this case will not be handled here.
         */

        u64 raw;
        u64 sig1, sig2, sig2_ext, hi, lo, hi2, lo2, add, bits;
        i32 exp2;
        u32 lz;
        bool exact = false, carry, round_up;

        /* convert (10^exp) to (sig2 * 2^exp2) */
        pow10_table_get_sig(exp, &sig2, &sig2_ext);
        pow10_table_get_exp(exp, &exp2);

        /* normalize and multiply */
        lz = u64_lz_bits(sig);
        sig1 = sig << lz;
        exp2 -= (i32)lz;
        u128_mul(sig1, sig2, &hi, &lo);

        /*
         The `hi` is in range [0x4000000000000000, 0xFFFFFFFFFFFFFFFE],
         To get normalized value, `hi` should be shifted to the left by 0 or 1.

         The highest significant 53 bits is used by IEEE-754 double number,
         and the bit 54 is used to detect rounding direction.

         The lowest (64 - 54 - 1) bits is used to check whether it contains 0.
         */
        bits = hi & (((u64)1 << (64 - 54 - 1)) - 1);
        if (bits - 1 < (((u64)1 << (64 - 54 - 1)) - 2)) {
            /*
             (bits != 0 && bits != 0x1FF) => (bits - 1 < 0x1FF - 1)
             The `bits` is not zero, so we don't need to check `round to even`
             case. The `bits` contains bit `0`, so we can drop the extra bits
             after `0`.
             */
            exact = true;

        } else {
            /*
             (bits == 0 || bits == 0x1FF)
             The `bits` is filled with all `0` or all `1`, so we need to check
             lower bits with another 64-bit multiplication.
             */
            u128_mul(sig1, sig2_ext, &hi2, &lo2);

            add = lo + hi2;
            if (add + 1 > (u64)1) {
                /*
                 (add != 0 && add != U64_MAX) => (add + 1 > 1)
                 The `add` is not zero, so we don't need to check `round to
                 even` case. The `add` contains bit `0`, so we can drop the
                 extra bits after `0`. The `hi` cannot be U64_MAX, so it will
                 not overflow.
                 */
                carry = add < lo || add < hi2;
                hi += carry;
                exact = true;
            }
        }

        if (exact) {
            /* normalize */
            lz = hi < ((u64)1 << 63);
            hi <<= lz;
            exp2 -= (i32)lz;
            exp2 += 64;

            /* test the bit 54 and get rounding direction */
            round_up = (hi & ((u64)1 << (64 - 54))) > (u64)0;
            hi += (round_up ? ((u64)1 << (64 - 54)) : (u64)0);

            /* test overflow */
            if (hi < ((u64)1 << (64 - 54))) {
                hi = ((u64)1 << 63);
                exp2 += 1;
            }

            /* This is a normal number, convert it to IEEE-754 format. */
            hi >>= F64_BITS - F64_SIG_FULL_BITS;
            exp2 += F64_BITS - F64_SIG_FULL_BITS + F64_SIG_BITS;
            exp2 += F64_EXP_BIAS;
            raw = ((u64)exp2 << F64_SIG_BITS) | (hi & F64_SIG_MASK);
            return_f64_bin(raw);
        }
    }

    /*
     Slow path: read double number exactly with diyfp.
     1. Use cached diyfp to get an approximation value.
     2. Use bigcomp to check the approximation value if needed.

     This algorithm refers to google's double-conversion project:
     https://github.com/google/double-conversion
     */
    {
        const i32 ERR_ULP_LOG = 3;
        const i32 ERR_ULP = 1 << ERR_ULP_LOG;
        const i32 ERR_CACHED_POW = ERR_ULP / 2;
        const i32 ERR_MUL_FIXED = ERR_ULP / 2;
        const i32 DIY_SIG_BITS = 64;
        const i32 EXP_BIAS = F64_EXP_BIAS + F64_SIG_BITS;
        const i32 EXP_SUBNORMAL = -EXP_BIAS + 1;

        u64 fp_err;
        u32 bits;
        i32 order_of_magnitude;
        i32 effective_significand_size;
        i32 precision_digits_count;
        u64 precision_bits;
        u64 half_way;

        u64 raw;
        diy_fp fp, fp_upper;
        bigint big_full, big_comp;
        i32 cmp;

        fp.sig = sig;
        fp.exp = 0;
        fp_err = sig_cut ? (u64)(ERR_ULP / 2) : (u64)0;

        /* normalize */
        bits = u64_lz_bits(fp.sig);
        fp.sig <<= bits;
        fp.exp -= (i32)bits;
        fp_err <<= bits;

        /* multiply and add error */
        fp = diy_fp_mul(fp, diy_fp_get_cached_pow10(exp));
        fp_err += (u64)ERR_CACHED_POW + (fp_err != 0) + (u64)ERR_MUL_FIXED;

        /* normalize */
        bits = u64_lz_bits(fp.sig);
        fp.sig <<= bits;
        fp.exp -= (i32)bits;
        fp_err <<= bits;

        /* effective significand */
        order_of_magnitude = DIY_SIG_BITS + fp.exp;
        if (likely(order_of_magnitude >= EXP_SUBNORMAL + F64_SIG_FULL_BITS)) {
            effective_significand_size = F64_SIG_FULL_BITS;
        } else if (order_of_magnitude <= EXP_SUBNORMAL) {
            effective_significand_size = 0;
        } else {
            effective_significand_size = order_of_magnitude - EXP_SUBNORMAL;
        }

        /* precision digits count */
        precision_digits_count = DIY_SIG_BITS - effective_significand_size;
        if (unlikely(precision_digits_count + ERR_ULP_LOG >= DIY_SIG_BITS)) {
            i32 shr = (precision_digits_count + ERR_ULP_LOG) - DIY_SIG_BITS + 1;
            fp.sig >>= shr;
            fp.exp += shr;
            fp_err = (fp_err >> shr) + 1 + (u32)ERR_ULP;
            precision_digits_count -= shr;
        }

        /* half way */
        precision_bits = fp.sig & (((u64)1 << precision_digits_count) - 1);
        precision_bits *= (u32)ERR_ULP;
        half_way = (u64)1 << (precision_digits_count - 1);
        half_way *= (u32)ERR_ULP;

        /* rounding */
        fp.sig >>= precision_digits_count;
        fp.sig += (precision_bits >= half_way + fp_err);
        fp.exp += precision_digits_count;

        /* get IEEE double raw value */
        raw = diy_fp_to_ieee_raw(fp);
        if (unlikely(raw == F64_RAW_INF)) return_inf();
        if (likely(precision_bits <= half_way - fp_err ||
                   precision_bits >= half_way + fp_err)) {
            return_f64_bin(raw); /* number is accurate */
        }
        /* now the number is the correct value, or the next lower value */

        /* upper boundary */
        if (raw & F64_EXP_MASK) {
            fp_upper.sig = (raw & F64_SIG_MASK) + ((u64)1 << F64_SIG_BITS);
            fp_upper.exp = (i32)((raw & F64_EXP_MASK) >> F64_SIG_BITS);
        } else {
            fp_upper.sig = (raw & F64_SIG_MASK);
            fp_upper.exp = 1;
        }
        fp_upper.exp -= F64_EXP_BIAS + F64_SIG_BITS;
        fp_upper.sig <<= 1;
        fp_upper.exp -= 1;
        fp_upper.sig += 1; /* add half ulp */

        /* compare with bigint */
        bigint_set_buf(&big_full, sig, &exp, sig_cut, sig_end, dot_pos);
        bigint_set_u64(&big_comp, fp_upper.sig);
        if (exp >= 0) {
            bigint_mul_pow10(&big_full, +exp);
        } else {
            bigint_mul_pow10(&big_comp, -exp);
        }
        if (fp_upper.exp > 0) {
            bigint_mul_pow2(&big_comp, (u32)+fp_upper.exp);
        } else {
            bigint_mul_pow2(&big_full, (u32)-fp_upper.exp);
        }
        cmp = bigint_cmp(&big_full, &big_comp);
        if (likely(cmp != 0)) {
            /* round down or round up */
            raw += (cmp > 0);
        } else {
            /* falls midway, round to even */
            raw += (raw & 1);
        }

        if (unlikely(raw == F64_RAW_INF)) return_inf();
        return_f64_bin(raw);
    }

#undef return_err
#undef return_inf
#undef return_0
#undef return_i64
#undef return_f64
#undef return_f64_bin
#undef return_raw
}



#else /* FP_READER */

/**
 Read a JSON number.
 This is a fallback function if the custom number reader is disabled.
 This function use libc's strtod() to read floating-point number.
 */
static_inline bool read_num(u8 **ptr, u8 **pre, yyjson_read_flag flg,
                            yyjson_val *val, const char **msg) {

#define return_err(_pos, _msg) do { \
    *msg = _msg; \
    *end = _pos; \
    return false; \
} while (false)

#define return_0() do { \
    val->tag = YYJSON_TYPE_NUM | (u64)((u8)sign << 3); \
    val->uni.u64 = 0; \
    *end = cur; return true; \
} while (false)

#define return_i64(_v) do { \
    val->tag = YYJSON_TYPE_NUM | (u64)((u8)sign << 3); \
    val->uni.u64 = (u64)(sign ? (u64)(~(_v) + 1) : (u64)(_v)); \
    *end = cur; return true; \
} while (false)

#define return_f64(_v) do { \
    val->tag = YYJSON_TYPE_NUM | YYJSON_SUBTYPE_REAL; \
    val->uni.f64 = sign ? -(f64)(_v) : (f64)(_v); \
    *end = cur; return true; \
} while (false)

#define return_f64_bin(_v) do { \
    val->tag = YYJSON_TYPE_NUM | YYJSON_SUBTYPE_REAL; \
    val->uni.u64 = ((u64)sign << 63) | (u64)(_v); \
    *end = cur; return true; \
} while (false)

#define return_inf() do { \
    if (has_read_flag(BIGNUM_AS_RAW)) return_raw(); \
    if (has_read_flag(ALLOW_INF_AND_NAN)) return_f64_bin(F64_RAW_INF); \
    else return_err(hdr, "number is infinity when parsed as double"); \
} while (false)

#define return_raw() do { \
    if (*pre) **pre = '\0'; /* add null-terminator for previous raw string */ \
    val->tag = ((u64)(cur - hdr) << YYJSON_TAG_BIT) | YYJSON_TYPE_RAW; \
    val->uni.str = (const char *)hdr; \
    *pre = cur; *end = cur; return true; \
} while (false)

    u64 sig, num;
    u8 *hdr = *ptr;
    u8 *cur = *ptr;
    u8 **end = ptr;
    u8 *dot = NULL;
    u8 *f64_end = NULL;
    bool sign;

    /* read number as raw string if has `YYJSON_READ_NUMBER_AS_RAW` flag */
    if (has_read_flag(NUMBER_AS_RAW)) {
        return read_num_raw(ptr, pre, flg, val, msg);
    }

    sign = (*hdr == '-');
    cur += sign;
    sig = (u8)(*cur - '0');

    /* read first digit, check leading zero */
    if (unlikely(!digi_is_digit(*cur))) {
        if (has_read_flag(ALLOW_INF_AND_NAN)) {
            if (read_inf_or_nan(sign, &cur, pre, flg, val)) {
                *end = cur;
                return true;
            }
        }
        return_err(cur, "no digit after minus sign");
    }
    if (*cur == '0') {
        cur++;
        if (unlikely(digi_is_digit(*cur))) {
            return_err(cur - 1, "number with leading zero is not allowed");
        }
        if (!digi_is_fp(*cur)) return_0();
        goto read_double;
    }

    /* read continuous digits, up to 19 characters */
#define expr_intg(i) \
    if (likely((num = (u64)(cur[i] - (u8)'0')) <= 9)) sig = num + sig * 10; \
    else { cur += i; goto intg_end; }
    repeat_in_1_18(expr_intg)
#undef expr_intg

    /* here are 19 continuous digits, skip them */
    cur += 19;
    if (digi_is_digit(cur[0]) && !digi_is_digit_or_fp(cur[1])) {
        /* this number is an integer consisting of 20 digits */
        num = (u8)(*cur - '0');
        if ((sig < (U64_MAX / 10)) ||
            (sig == (U64_MAX / 10) && num <= (U64_MAX % 10))) {
            sig = num + sig * 10;
            cur++;
            if (sign) {
                if (has_read_flag(BIGNUM_AS_RAW)) return_raw();
                return_f64(unsafe_yyjson_u64_to_f64(sig));
            }
            return_i64(sig);
        }
    }

intg_end:
    /* continuous digits ended */
    if (!digi_is_digit_or_fp(*cur)) {
        /* this number is an integer consisting of 1 to 19 digits */
        if (sign && (sig > ((u64)1 << 63))) {
            if (has_read_flag(BIGNUM_AS_RAW)) return_raw();
            return_f64(unsafe_yyjson_u64_to_f64(sig));
        }
        return_i64(sig);
    }

read_double:
    /* this number should be read as double */
    while (digi_is_digit(*cur)) cur++;
    if (!digi_is_fp(*cur) && has_read_flag(BIGNUM_AS_RAW)) {
        return_raw(); /* it's a large integer */
    }
    if (*cur == '.') {
        /* skip fraction part */
        dot = cur;
        cur++;
        if (!digi_is_digit(*cur)) {
            return_err(cur, "no digit after decimal point");
        }
        cur++;
        while (digi_is_digit(*cur)) cur++;
    }
    if (digi_is_exp(*cur)) {
        /* skip exponent part */
        cur += 1 + digi_is_sign(cur[1]);
        if (!digi_is_digit(*cur)) {
            return_err(cur, "no digit after exponent sign");
        }
        cur++;
        while (digi_is_digit(*cur)) cur++;
    }

    /*
     libc's strtod() is used to parse the floating-point number.

     Note that the decimal point character used by strtod() is locale-dependent,
     and the rounding direction may affected by fesetround().

     For currently known locales, (en, zh, ja, ko, am, he, hi) use '.' as the
     decimal point, while other locales use ',' as the decimal point.

     Here strtod() is called twice for different locales, but if another thread
     happens calls setlocale() between two strtod(), parsing may still fail.
     */
    val->uni.f64 = strtod((const char *)hdr, (char **)&f64_end);
    if (unlikely(f64_end != cur)) {
        /* replace '.' with ',' for locale */
        bool cut = (*cur == ',');
        if (cut) *cur = ' ';
        if (dot) *dot = ',';
        val->uni.f64 = strtod((const char *)hdr, (char **)&f64_end);
        /* restore ',' to '.' */
        if (cut) *cur = ',';
        if (dot) *dot = '.';
        if (unlikely(f64_end != cur)) {
            return_err(hdr, "strtod() failed to parse the number");
        }
    }
    if (unlikely(val->uni.f64 >= HUGE_VAL || val->uni.f64 <= -HUGE_VAL)) {
        return_inf();
    }
    val->tag = YYJSON_TYPE_NUM | YYJSON_SUBTYPE_REAL;
    *end = cur;
    return true;

#undef return_err
#undef return_0
#undef return_i64
#undef return_f64
#undef return_f64_bin
#undef return_inf
#undef return_raw
}

#endif /* FP_READER */



/*==============================================================================
 * JSON String Reader
 *============================================================================*/

/**
 Read a JSON string.
 @param ptr The head pointer of string before '"' prefix (inout).
 @param lst JSON last position.
 @param inv Allow invalid unicode.
 @param val The string value to be written.
 @param msg The error message pointer.
 @param con Continuation for incremental parsing.
 @return Whether success.
 */
static_inline bool read_str(u8 **ptr, u8 *lst, bool inv, yyjson_val *val,
                            const char **msg, u8 **con) {
    /*
     Each unicode code point is encoded as 1 to 4 bytes in UTF-8 encoding,
     we use 4-byte mask and pattern value to validate UTF-8 byte sequence,
     this requires the input data to have 4-byte zero padding.
     ---------------------------------------------------
     1 byte
     unicode range [U+0000, U+007F]
     unicode min   [.......0]
     unicode max   [.1111111]
     bit pattern   [0.......]
     ---------------------------------------------------
     2 byte
     unicode range [U+0080, U+07FF]
     unicode min   [......10 ..000000]
     unicode max   [...11111 ..111111]
     bit require   [...xxxx. ........] (1E 00)
     bit mask      [xxx..... xx......] (E0 C0)
     bit pattern   [110..... 10......] (C0 80)
     ---------------------------------------------------
     3 byte
     unicode range [U+0800, U+FFFF]
     unicode min   [........ ..100000 ..000000]
     unicode max   [....1111 ..111111 ..111111]
     bit require   [....xxxx ..x..... ........] (0F 20 00)
     bit mask      [xxxx.... xx...... xx......] (F0 C0 C0)
     bit pattern   [1110.... 10...... 10......] (E0 80 80)
     ---------------------------------------------------
     3 byte invalid (reserved for surrogate halves)
     unicode range [U+D800, U+DFFF]
     unicode min   [....1101 ..100000 ..000000]
     unicode max   [....1101 ..111111 ..111111]
     bit mask      [....xxxx ..x..... ........] (0F 20 00)
     bit pattern   [....1101 ..1..... ........] (0D 20 00)
     ---------------------------------------------------
     4 byte
     unicode range [U+10000, U+10FFFF]
     unicode min   [........ ...10000 ..000000 ..000000]
     unicode max   [.....100 ..001111 ..111111 ..111111]
     bit require   [.....xxx ..xx.... ........ ........] (07 30 00 00)
     bit mask      [xxxxx... xx...... xx...... xx......] (F8 C0 C0 C0)
     bit pattern   [11110... 10...... 10...... 10......] (F0 80 80 80)
     ---------------------------------------------------
     */
#if YYJSON_ENDIAN == YYJSON_BIG_ENDIAN
    const u32 b1_mask = 0x80000000UL;
    const u32 b1_patt = 0x00000000UL;
    const u32 b2_mask = 0xE0C00000UL;
    const u32 b2_patt = 0xC0800000UL;
    const u32 b2_requ = 0x1E000000UL;
    const u32 b3_mask = 0xF0C0C000UL;
    const u32 b3_patt = 0xE0808000UL;
    const u32 b3_requ = 0x0F200000UL;
    const u32 b3_erro = 0x0D200000UL;
    const u32 b4_mask = 0xF8C0C0C0UL;
    const u32 b4_patt = 0xF0808080UL;
    const u32 b4_requ = 0x07300000UL;
    const u32 b4_err0 = 0x04000000UL;
    const u32 b4_err1 = 0x03300000UL;
#elif YYJSON_ENDIAN == YYJSON_LITTLE_ENDIAN
    const u32 b1_mask = 0x00000080UL;
    const u32 b1_patt = 0x00000000UL;
    const u32 b2_mask = 0x0000C0E0UL;
    const u32 b2_patt = 0x000080C0UL;
    const u32 b2_requ = 0x0000001EUL;
    const u32 b3_mask = 0x00C0C0F0UL;
    const u32 b3_patt = 0x008080E0UL;
    const u32 b3_requ = 0x0000200FUL;
    const u32 b3_erro = 0x0000200DUL;
    const u32 b4_mask = 0xC0C0C0F8UL;
    const u32 b4_patt = 0x808080F0UL;
    const u32 b4_requ = 0x00003007UL;
    const u32 b4_err0 = 0x00000004UL;
    const u32 b4_err1 = 0x00003003UL;
#else
    /* this should be evaluated at compile-time */
    v32_uni b1_mask_uni = {{ 0x80, 0x00, 0x00, 0x00 }};
    v32_uni b1_patt_uni = {{ 0x00, 0x00, 0x00, 0x00 }};
    v32_uni b2_mask_uni = {{ 0xE0, 0xC0, 0x00, 0x00 }};
    v32_uni b2_patt_uni = {{ 0xC0, 0x80, 0x00, 0x00 }};
    v32_uni b2_requ_uni = {{ 0x1E, 0x00, 0x00, 0x00 }};
    v32_uni b3_mask_uni = {{ 0xF0, 0xC0, 0xC0, 0x00 }};
    v32_uni b3_patt_uni = {{ 0xE0, 0x80, 0x80, 0x00 }};
    v32_uni b3_requ_uni = {{ 0x0F, 0x20, 0x00, 0x00 }};
    v32_uni b3_erro_uni = {{ 0x0D, 0x20, 0x00, 0x00 }};
    v32_uni b4_mask_uni = {{ 0xF8, 0xC0, 0xC0, 0xC0 }};
    v32_uni b4_patt_uni = {{ 0xF0, 0x80, 0x80, 0x80 }};
    v32_uni b4_requ_uni = {{ 0x07, 0x30, 0x00, 0x00 }};
    v32_uni b4_err0_uni = {{ 0x04, 0x00, 0x00, 0x00 }};
    v32_uni b4_err1_uni = {{ 0x03, 0x30, 0x00, 0x00 }};
    u32 b1_mask = b1_mask_uni.u;
    u32 b1_patt = b1_patt_uni.u;
    u32 b2_mask = b2_mask_uni.u;
    u32 b2_patt = b2_patt_uni.u;
    u32 b2_requ = b2_requ_uni.u;
    u32 b3_mask = b3_mask_uni.u;
    u32 b3_patt = b3_patt_uni.u;
    u32 b3_requ = b3_requ_uni.u;
    u32 b3_erro = b3_erro_uni.u;
    u32 b4_mask = b4_mask_uni.u;
    u32 b4_patt = b4_patt_uni.u;
    u32 b4_requ = b4_requ_uni.u;
    u32 b4_err0 = b4_err0_uni.u;
    u32 b4_err1 = b4_err1_uni.u;
#endif

#define is_valid_seq_1(uni) ( \
    ((uni & b1_mask) == b1_patt) \
)

#define is_valid_seq_2(uni) ( \
    ((uni & b2_mask) == b2_patt) && \
    ((uni & b2_requ)) \
)

#define is_valid_seq_3(uni) ( \
    ((uni & b3_mask) == b3_patt) && \
    ((tmp = (uni & b3_requ))) && \
    ((tmp != b3_erro)) \
)

#define is_valid_seq_4(uni) ( \
    ((uni & b4_mask) == b4_patt) && \
    ((tmp = (uni & b4_requ))) && \
    ((tmp & b4_err0) == 0 || (tmp & b4_err1) == 0) \
)

#define return_err(_end, _msg) do { \
    *msg = _msg; \
    *end = _end; \
    if (con) { con[0] = _end; con[1] = dst; } \
    return false; \
} while (false)

    u8 *cur = *ptr;
    u8 **end = ptr;
    u8 *src = ++cur, *dst = NULL, *pos;
    u16 hi, lo;
    u32 uni, tmp;

    if (unlikely(con && con[0])) {
        /* Resume incremental parsing. */
        src = con[0];
        dst = con[1];
        if (dst) goto copy_ascii;
    }

skip_ascii:
    /* Most strings have no escaped characters, so we can jump them quickly. */

skip_ascii_begin:
    /*
     We want to make loop unrolling, as shown in the following code. Some
     compiler may not generate instructions as expected, so we rewrite it with
     explicit goto statements. We hope the compiler can generate instructions
     like this: https://godbolt.org/z/8vjsYq

         while (true) repeat16({
            if (likely(!(char_is_ascii_stop(*src)))) src++;
            else break;
         })
     */
#define expr_jump(i) \
    if (likely(!char_is_ascii_stop(src[i]))) {} \
    else goto skip_ascii_stop##i;

#define expr_stop(i) \
    skip_ascii_stop##i: \
    src += i; \
    goto skip_ascii_end;

    repeat16_incr(expr_jump)
    src += 16;
    goto skip_ascii_begin;
    repeat16_incr(expr_stop)

#undef expr_jump
#undef expr_stop

skip_ascii_end:

    /*
     GCC may store src[i] in a register at each line of expr_jump(i) above.
     These instructions are useless and will degrade performance.
     This inline asm is a hint for gcc: "the memory has been modified,
     do not cache it".

     MSVC, Clang, ICC can generate expected instructions without this hint.
     */
#if YYJSON_IS_REAL_GCC
    __asm__ volatile("":"=m"(*src));
#endif
    if (likely(*src == '"')) {
        val->tag = ((u64)(src - cur) << YYJSON_TAG_BIT) |
                    (u64)(YYJSON_TYPE_STR | YYJSON_SUBTYPE_NOESC);
        val->uni.str = (const char *)cur;
        *src = '\0';
        *end = src + 1;
        if (con) con[0] = con[1] = NULL;
        return true;
    }

skip_utf8:
    if (*src & 0x80) { /* non-ASCII character */
        /*
         Non-ASCII character appears here, which means that the text is likely
         to be written in non-English or emoticons. According to some common
         data set statistics, byte sequences of the same length may appear
         consecutively. We process the byte sequences of the same length in each
         loop, which is more friendly to branch prediction.
         */
        pos = src;
#if YYJSON_DISABLE_UTF8_VALIDATION
        while (true) repeat8({
            if (likely((*src & 0xF0) == 0xE0)) src += 3;
            else break;
        })
        if (*src < 0x80) goto skip_ascii;
        while (true) repeat8({
            if (likely((*src & 0xE0) == 0xC0)) src += 2;
            else break;
        })
        while (true) repeat8({
            if (likely((*src & 0xF8) == 0xF0)) src += 4;
            else break;
        })
#else
        uni = byte_load_4(src);
        while (is_valid_seq_3(uni)) {
            src += 3;
            uni = byte_load_4(src);
        }
        if (is_valid_seq_1(uni)) goto skip_ascii;
        while (is_valid_seq_2(uni)) {
            src += 2;
            uni = byte_load_4(src);
        }
        while (is_valid_seq_4(uni)) {
            src += 4;
            uni = byte_load_4(src);
        }
#endif
        if (unlikely(pos == src)) {
            if (!inv) return_err(src, "invalid UTF-8 encoding in string");
            ++src;
        }
        goto skip_ascii;
    }

    /* The escape character appears, we need to copy it. */
    dst = src;
copy_escape:
    if (likely(*src == '\\')) {
        switch (*++src) {
            case '"':  *dst++ = '"';  src++; break;
            case '\\': *dst++ = '\\'; src++; break;
            case '/':  *dst++ = '/';  src++; break;
            case 'b':  *dst++ = '\b'; src++; break;
            case 'f':  *dst++ = '\f'; src++; break;
            case 'n':  *dst++ = '\n'; src++; break;
            case 'r':  *dst++ = '\r'; src++; break;
            case 't':  *dst++ = '\t'; src++; break;
            case 'u':
                if (unlikely(!read_hex_u16(++src, &hi))) {
                    return_err(src - 2, "invalid escaped sequence in string");
                }
                src += 4;
                if (likely((hi & 0xF800) != 0xD800)) {
                    /* a BMP character */
                    if (hi >= 0x800) {
                        *dst++ = (u8)(0xE0 | (hi >> 12));
                        *dst++ = (u8)(0x80 | ((hi >> 6) & 0x3F));
                        *dst++ = (u8)(0x80 | (hi & 0x3F));
                    } else if (hi >= 0x80) {
                        *dst++ = (u8)(0xC0 | (hi >> 6));
                        *dst++ = (u8)(0x80 | (hi & 0x3F));
                    } else {
                        *dst++ = (u8)hi;
                    }
                } else {
                    /* a non-BMP character, represented as a surrogate pair */
                    if (unlikely((hi & 0xFC00) != 0xD800)) {
                        return_err(src - 6, "invalid high surrogate in string");
                    }
                    if (unlikely(!byte_match_2(src, "\\u"))) {
                        return_err(src - 6, "no low surrogate in string");
                    }
                    if (unlikely(!read_hex_u16(src + 2, &lo))) {
                        return_err(src - 6, "invalid escape in string");
                    }
                    if (unlikely((lo & 0xFC00) != 0xDC00)) {
                        return_err(src - 6, "invalid low surrogate in string");
                    }
                    uni = ((((u32)hi - 0xD800) << 10) |
                            ((u32)lo - 0xDC00)) + 0x10000;
                    *dst++ = (u8)(0xF0 | (uni >> 18));
                    *dst++ = (u8)(0x80 | ((uni >> 12) & 0x3F));
                    *dst++ = (u8)(0x80 | ((uni >> 6) & 0x3F));
                    *dst++ = (u8)(0x80 | (uni & 0x3F));
                    src += 6;
                }
                break;
            default: return_err(src - 1, "invalid escaped sequence in string");
        }
    } else if (likely(*src == '"')) {
        val->tag = ((u64)(dst - cur) << YYJSON_TAG_BIT) | YYJSON_TYPE_STR;
        val->uni.str = (const char *)cur;
        *dst = '\0';
        *end = src + 1;
        if (con) con[0] = con[1] = NULL;
        return true;
    } else {
        if (!inv) return_err(src, "unexpected control character in string");
        if (src >= lst) return_err(src, "unclosed string");
        *dst++ = *src++;
    }

copy_ascii:
    /*
     Copy continuous ASCII, loop unrolling, same as the following code:

         while (true) repeat16({
            if (unlikely(char_is_ascii_stop(*src))) break;
            *dst++ = *src++;
         })
     */
#if YYJSON_IS_REAL_GCC
#   define expr_jump(i) \
    if (likely(!(char_is_ascii_stop(src[i])))) {} \
    else { __asm__ volatile("":"=m"(src[i])); goto copy_ascii_stop_##i; }
#else
#   define expr_jump(i) \
    if (likely(!(char_is_ascii_stop(src[i])))) {} \
    else { goto copy_ascii_stop_##i; }
#endif
    repeat16_incr(expr_jump)
#undef expr_jump

    byte_move_16(dst, src);
    src += 16;
    dst += 16;
    goto copy_ascii;

    /*
     The memory will be moved forward by at least 1 byte. So the `byte_move`
     can be one byte more than needed to reduce the number of instructions.
     */
copy_ascii_stop_0:
    goto copy_utf8;
copy_ascii_stop_1:
    byte_move_2(dst, src);
    src += 1;
    dst += 1;
    goto copy_utf8;
copy_ascii_stop_2:
    byte_move_2(dst, src);
    src += 2;
    dst += 2;
    goto copy_utf8;
copy_ascii_stop_3:
    byte_move_4(dst, src);
    src += 3;
    dst += 3;
    goto copy_utf8;
copy_ascii_stop_4:
    byte_move_4(dst, src);
    src += 4;
    dst += 4;
    goto copy_utf8;
copy_ascii_stop_5:
    byte_move_4(dst, src);
    byte_move_2(dst + 4, src + 4);
    src += 5;
    dst += 5;
    goto copy_utf8;
copy_ascii_stop_6:
    byte_move_4(dst, src);
    byte_move_2(dst + 4, src + 4);
    src += 6;
    dst += 6;
    goto copy_utf8;
copy_ascii_stop_7:
    byte_move_8(dst, src);
    src += 7;
    dst += 7;
    goto copy_utf8;
copy_ascii_stop_8:
    byte_move_8(dst, src);
    src += 8;
    dst += 8;
    goto copy_utf8;
copy_ascii_stop_9:
    byte_move_8(dst, src);
    byte_move_2(dst + 8, src + 8);
    src += 9;
    dst += 9;
    goto copy_utf8;
copy_ascii_stop_10:
    byte_move_8(dst, src);
    byte_move_2(dst + 8, src + 8);
    src += 10;
    dst += 10;
    goto copy_utf8;
copy_ascii_stop_11:
    byte_move_8(dst, src);
    byte_move_4(dst + 8, src + 8);
    src += 11;
    dst += 11;
    goto copy_utf8;
copy_ascii_stop_12:
    byte_move_8(dst, src);
    byte_move_4(dst + 8, src + 8);
    src += 12;
    dst += 12;
    goto copy_utf8;
copy_ascii_stop_13:
    byte_move_8(dst, src);
    byte_move_4(dst + 8, src + 8);
    byte_move_2(dst + 12, src + 12);
    src += 13;
    dst += 13;
    goto copy_utf8;
copy_ascii_stop_14:
    byte_move_8(dst, src);
    byte_move_4(dst + 8, src + 8);
    byte_move_2(dst + 12, src + 12);
    src += 14;
    dst += 14;
    goto copy_utf8;
copy_ascii_stop_15:
    byte_move_16(dst, src);
    src += 15;
    dst += 15;
    goto copy_utf8;

copy_utf8:
    if (*src & 0x80) { /* non-ASCII character */
        pos = src;
        uni = byte_load_4(src);
#if YYJSON_DISABLE_UTF8_VALIDATION
        while (true) repeat4({
            if ((uni & b3_mask) == b3_patt) {
                byte_copy_4(dst, &uni);
                dst += 3;
                src += 3;
                uni = byte_load_4(src);
            } else break;
        })
        if ((uni & b1_mask) == b1_patt) goto copy_ascii;
        while (true) repeat4({
            if ((uni & b2_mask) == b2_patt) {
                byte_copy_2(dst, &uni);
                dst += 2;
                src += 2;
                uni = byte_load_4(src);
            } else break;
        })
        while (true) repeat4({
            if ((uni & b4_mask) == b4_patt) {
                byte_copy_4(dst, &uni);
                dst += 4;
                src += 4;
                uni = byte_load_4(src);
            } else break;
        })
#else
        while (is_valid_seq_3(uni)) {
            byte_copy_4(dst, &uni);
            dst += 3;
            src += 3;
            uni = byte_load_4(src);
        }
        if (is_valid_seq_1(uni)) goto copy_ascii;
        while (is_valid_seq_2(uni)) {
            byte_copy_2(dst, &uni);
            dst += 2;
            src += 2;
            uni = byte_load_4(src);
        }
        while (is_valid_seq_4(uni)) {
            byte_copy_4(dst, &uni);
            dst += 4;
            src += 4;
            uni = byte_load_4(src);
        }
#endif
        if (unlikely(pos == src)) {
            if (!inv) return_err(src, MSG_ERR_UTF8);
            goto copy_ascii_stop_1;
        }
        goto copy_ascii;
    }
    goto copy_escape;

#undef return_err
#undef is_valid_seq_1
#undef is_valid_seq_2
#undef is_valid_seq_3
#undef is_valid_seq_4
}



/*==============================================================================
 * JSON Reader Implementation
 *
 * We use goto statements to build the finite state machine (FSM).
 * The FSM's state was held by program counter (PC) and the 'goto' make the
 * state transitions.
 *============================================================================*/

/** Read single value JSON document. */
static_noinline yyjson_doc *read_root_single(u8 *hdr, u8 *cur, u8 *end,
                                             yyjson_alc alc,
                                             yyjson_read_flag flg,
                                             yyjson_read_err *err) {

#define return_err(_pos, _code, _msg) do { \
    if (is_truncated_end(hdr, _pos, end, YYJSON_READ_ERROR_##_code, flg)) { \
        err->pos = (usize)(end - hdr); \
        err->code = YYJSON_READ_ERROR_UNEXPECTED_END; \
        err->msg = MSG_NOT_END; \
    } else { \
        err->pos = (usize)(_pos - hdr); \
        err->code = YYJSON_READ_ERROR_##_code; \
        err->msg = _msg; \
    } \
    if (val_hdr) alc.free(alc.ctx, (void *)val_hdr); \
    return NULL; \
} while (false)

    usize hdr_len; /* value count used by doc */
    usize alc_num; /* value count capacity */
    yyjson_val *val_hdr; /* the head of allocated values */
    yyjson_val *val; /* current value */
    yyjson_doc *doc; /* the JSON document, equals to val_hdr */
    const char *msg; /* error message */

    bool raw; /* read number as raw */
    bool inv; /* allow invalid unicode */
    u8 *raw_end; /* raw end for null-terminator */
    u8 **pre; /* previous raw end pointer */

    hdr_len = sizeof(yyjson_doc) / sizeof(yyjson_val);
    hdr_len += (sizeof(yyjson_doc) % sizeof(yyjson_val)) > 0;
    alc_num = hdr_len + 1; /* single value */

    val_hdr = (yyjson_val *)alc.malloc(alc.ctx, alc_num * sizeof(yyjson_val));
    if (unlikely(!val_hdr)) goto fail_alloc;
    val = val_hdr + hdr_len;
    raw = has_read_flag(NUMBER_AS_RAW) || has_read_flag(BIGNUM_AS_RAW);
    inv = has_read_flag(ALLOW_INVALID_UNICODE) != 0;
    raw_end = NULL;
    pre = raw ? &raw_end : NULL;

    if (char_is_num(*cur)) {
        if (likely(read_num(&cur, pre, flg, val, &msg))) goto doc_end;
        goto fail_number;
    }
    if (*cur == '"') {
        if (likely(read_str(&cur, end, inv, val, &msg, NULL))) goto doc_end;
        goto fail_string;
    }
    if (*cur == 't') {
        if (likely(read_true(&cur, val))) goto doc_end;
        goto fail_literal_true;
    }
    if (*cur == 'f') {
        if (likely(read_false(&cur, val))) goto doc_end;
        goto fail_literal_false;
    }
    if (*cur == 'n') {
        if (likely(read_null(&cur, val))) goto doc_end;
        if (has_read_flag(ALLOW_INF_AND_NAN)) {
            if (read_nan(false, &cur, pre, flg, val)) goto doc_end;
        }
        goto fail_literal_null;
    }
    if (has_read_flag(ALLOW_INF_AND_NAN)) {
        if (read_inf_or_nan(false, &cur, pre, flg, val)) goto doc_end;
    }
    goto fail_character;

doc_end:
    /* check invalid contents after json document */
    if (unlikely(cur < end) && !has_read_flag(STOP_WHEN_DONE)) {
        if (has_read_flag(ALLOW_COMMENTS)) {
            if (!skip_spaces_and_comments(&cur)) {
                if (byte_match_2(cur, "/*")) goto fail_comment;
            }
        } else {
            while (char_is_space(*cur)) cur++;
        }
        if (unlikely(cur < end)) goto fail_garbage;
    }

    if (pre && *pre) **pre = '\0';
    doc = (yyjson_doc *)val_hdr;
    doc->root = val_hdr + hdr_len;
    doc->alc = alc;
    doc->dat_read = (usize)(cur - hdr);
    doc->val_read = 1;
    doc->str_pool = has_read_flag(INSITU) ? NULL : (char *)hdr;
    return doc;

fail_string:        return_err(cur, INVALID_STRING, msg);
fail_number:        return_err(cur, INVALID_NUMBER, msg);
fail_alloc:         return_err(cur, MEMORY_ALLOCATION, MSG_MALLOC);
fail_literal_true:  return_err(cur, LITERAL, MSG_CHAT_T);
fail_literal_false: return_err(cur, LITERAL, MSG_CHAR_F);
fail_literal_null:  return_err(cur, LITERAL, MSG_CHAR_N);
fail_character:     return_err(cur, UNEXPECTED_CHARACTER, MSG_CHAR);
fail_comment:       return_err(cur, INVALID_COMMENT, MSG_COMMENT);
fail_garbage:       return_err(cur, UNEXPECTED_CONTENT, MSG_GARBAGE);

#undef return_err
}

/** Read JSON document (accept all style, but optimized for minify). */
static_inline yyjson_doc *read_root_minify(u8 *hdr, u8 *cur, u8 *end,
                                           yyjson_alc alc,
                                           yyjson_read_flag flg,
                                           yyjson_read_err *err) {

#define return_err(_pos, _code, _msg) do { \
    if (is_truncated_end(hdr, _pos, end, YYJSON_READ_ERROR_##_code, flg)) { \
        err->pos = (usize)(end - hdr); \
        err->code = YYJSON_READ_ERROR_UNEXPECTED_END; \
        err->msg = MSG_NOT_END; \
    } else { \
        err->pos = (usize)(_pos - hdr); \
        err->code = YYJSON_READ_ERROR_##_code; \
        err->msg = _msg; \
    } \
    if (val_hdr) alc.free(alc.ctx, (void *)val_hdr); \
    return NULL; \
} while (false)

#define val_incr() do { \
    val++; \
    if (unlikely(val >= val_end)) { \
        usize alc_old = alc_len; \
        usize val_ofs = (usize)(val - val_hdr); \
        usize ctn_ofs = (usize)(ctn - val_hdr); \
        alc_len += alc_len / 2; \
        if ((sizeof(usize) < 8) && (alc_len >= alc_max)) goto fail_alloc; \
        val_tmp = (yyjson_val *)alc.realloc(alc.ctx, (void *)val_hdr, \
            alc_old * sizeof(yyjson_val), \
            alc_len * sizeof(yyjson_val)); \
        if ((!val_tmp)) goto fail_alloc; \
        val = val_tmp + val_ofs; \
        ctn = val_tmp + ctn_ofs; \
        val_hdr = val_tmp; \
        val_end = val_tmp + (alc_len - 2); \
    } \
} while (false)

    usize dat_len; /* data length in bytes, hint for allocator */
    usize hdr_len; /* value count used by yyjson_doc */
    usize alc_len; /* value count allocated */
    usize alc_max; /* maximum value count for allocator */
    usize ctn_len; /* the number of elements in current container */
    yyjson_val *val_hdr; /* the head of allocated values */
    yyjson_val *val_end; /* the end of allocated values */
    yyjson_val *val_tmp; /* temporary pointer for realloc */
    yyjson_val *val; /* current JSON value */
    yyjson_val *ctn; /* current container */
    yyjson_val *ctn_parent; /* parent of current container */
    yyjson_doc *doc; /* the JSON document, equals to val_hdr */
    const char *msg; /* error message */

    bool raw; /* read number as raw */
    bool inv; /* allow invalid unicode */
    u8 *raw_end; /* raw end for null-terminator */
    u8 **pre; /* previous raw end pointer */

    dat_len = has_read_flag(STOP_WHEN_DONE) ? 256 : (usize)(end - cur);
    hdr_len = sizeof(yyjson_doc) / sizeof(yyjson_val);
    hdr_len += (sizeof(yyjson_doc) % sizeof(yyjson_val)) > 0;
    alc_max = USIZE_MAX / sizeof(yyjson_val);
    alc_len = hdr_len + (dat_len / YYJSON_READER_ESTIMATED_MINIFY_RATIO) + 4;
    alc_len = yyjson_min(alc_len, alc_max);

    val_hdr = (yyjson_val *)alc.malloc(alc.ctx, alc_len * sizeof(yyjson_val));
    if (unlikely(!val_hdr)) goto fail_alloc;
    val_end = val_hdr + (alc_len - 2); /* padding for key-value pair reading */
    val = val_hdr + hdr_len;
    ctn = val;
    ctn_len = 0;
    raw = has_read_flag(NUMBER_AS_RAW) || has_read_flag(BIGNUM_AS_RAW);
    inv = has_read_flag(ALLOW_INVALID_UNICODE) != 0;
    raw_end = NULL;
    pre = raw ? &raw_end : NULL;

    if (*cur++ == '{') {
        ctn->tag = YYJSON_TYPE_OBJ;
        ctn->uni.ofs = 0;
        goto obj_key_begin;
    } else {
        ctn->tag = YYJSON_TYPE_ARR;
        ctn->uni.ofs = 0;
        goto arr_val_begin;
    }

arr_begin:
    /* save current container */
    ctn->tag = (((u64)ctn_len + 1) << YYJSON_TAG_BIT) |
               (ctn->tag & YYJSON_TAG_MASK);

    /* create a new array value, save parent container offset */
    val_incr();
    val->tag = YYJSON_TYPE_ARR;
    val->uni.ofs = (usize)((u8 *)val - (u8 *)ctn);

    /* push the new array value as current container */
    ctn = val;
    ctn_len = 0;

arr_val_begin:
    if (*cur == '{') {
        cur++;
        goto obj_begin;
    }
    if (*cur == '[') {
        cur++;
        goto arr_begin;
    }
    if (char_is_num(*cur)) {
        val_incr();
        ctn_len++;
        if (likely(read_num(&cur, pre, flg, val, &msg))) goto arr_val_end;
        goto fail_number;
    }
    if (*cur == '"') {
        val_incr();
        ctn_len++;
        if (likely(read_str(&cur, end, inv, val, &msg, NULL))) goto arr_val_end;
        goto fail_string;
    }
    if (*cur == 't') {
        val_incr();
        ctn_len++;
        if (likely(read_true(&cur, val))) goto arr_val_end;
        goto fail_literal_true;
    }
    if (*cur == 'f') {
        val_incr();
        ctn_len++;
        if (likely(read_false(&cur, val))) goto arr_val_end;
        goto fail_literal_false;
    }
    if (*cur == 'n') {
        val_incr();
        ctn_len++;
        if (likely(read_null(&cur, val))) goto arr_val_end;
        if (has_read_flag(ALLOW_INF_AND_NAN)) {
            if (read_nan(false, &cur, pre, flg, val)) goto arr_val_end;
        }
        goto fail_literal_null;
    }
    if (*cur == ']') {
        cur++;
        if (likely(ctn_len == 0)) goto arr_end;
        if (has_read_flag(ALLOW_TRAILING_COMMAS)) goto arr_end;
        while (*cur != ',') cur--;
        goto fail_trailing_comma;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto arr_val_begin;
    }
    if (has_read_flag(ALLOW_INF_AND_NAN) &&
        (*cur == 'i' || *cur == 'I' || *cur == 'N')) {
        val_incr();
        ctn_len++;
        if (read_inf_or_nan(false, &cur, pre, flg, val)) goto arr_val_end;
        goto fail_character_val;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto arr_val_begin;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_val;

arr_val_end:
    if (*cur == ',') {
        cur++;
        goto arr_val_begin;
    }
    if (*cur == ']') {
        cur++;
        goto arr_end;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto arr_val_end;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto arr_val_end;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_arr_end;

arr_end:
    /* get parent container */
    ctn_parent = (yyjson_val *)(void *)((u8 *)ctn - ctn->uni.ofs);

    /* save the next sibling value offset */
    ctn->uni.ofs = (usize)((u8 *)val - (u8 *)ctn) + sizeof(yyjson_val);
    ctn->tag = ((ctn_len) << YYJSON_TAG_BIT) | YYJSON_TYPE_ARR;
    if (unlikely(ctn == ctn_parent)) goto doc_end;

    /* pop parent as current container */
    ctn = ctn_parent;
    ctn_len = (usize)(ctn->tag >> YYJSON_TAG_BIT);
    if ((ctn->tag & YYJSON_TYPE_MASK) == YYJSON_TYPE_OBJ) {
        goto obj_val_end;
    } else {
        goto arr_val_end;
    }

obj_begin:
    /* push container */
    ctn->tag = (((u64)ctn_len + 1) << YYJSON_TAG_BIT) |
               (ctn->tag & YYJSON_TAG_MASK);
    val_incr();
    val->tag = YYJSON_TYPE_OBJ;
    /* offset to the parent */
    val->uni.ofs = (usize)((u8 *)val - (u8 *)ctn);
    ctn = val;
    ctn_len = 0;

obj_key_begin:
    if (likely(*cur == '"')) {
        val_incr();
        ctn_len++;
        if (likely(read_str(&cur, end, inv, val, &msg, NULL))) goto obj_key_end;
        goto fail_string;
    }
    if (likely(*cur == '}')) {
        cur++;
        if (likely(ctn_len == 0)) goto obj_end;
        if (has_read_flag(ALLOW_TRAILING_COMMAS)) goto obj_end;
        while (*cur != ',') cur--;
        goto fail_trailing_comma;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto obj_key_begin;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto obj_key_begin;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_obj_key;

obj_key_end:
    if (*cur == ':') {
        cur++;
        goto obj_val_begin;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto obj_key_end;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto obj_key_end;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_obj_sep;

obj_val_begin:
    if (*cur == '"') {
        val++;
        ctn_len++;
        if (likely(read_str(&cur, end, inv, val, &msg, NULL))) goto obj_val_end;
        goto fail_string;
    }
    if (char_is_num(*cur)) {
        val++;
        ctn_len++;
        if (likely(read_num(&cur, pre, flg, val, &msg))) goto obj_val_end;
        goto fail_number;
    }
    if (*cur == '{') {
        cur++;
        goto obj_begin;
    }
    if (*cur == '[') {
        cur++;
        goto arr_begin;
    }
    if (*cur == 't') {
        val++;
        ctn_len++;
        if (likely(read_true(&cur, val))) goto obj_val_end;
        goto fail_literal_true;
    }
    if (*cur == 'f') {
        val++;
        ctn_len++;
        if (likely(read_false(&cur, val))) goto obj_val_end;
        goto fail_literal_false;
    }
    if (*cur == 'n') {
        val++;
        ctn_len++;
        if (likely(read_null(&cur, val))) goto obj_val_end;
        if (has_read_flag(ALLOW_INF_AND_NAN)) {
            if (read_nan(false, &cur, pre, flg, val)) goto obj_val_end;
        }
        goto fail_literal_null;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto obj_val_begin;
    }
    if (has_read_flag(ALLOW_INF_AND_NAN) &&
        (*cur == 'i' || *cur == 'I' || *cur == 'N')) {
        val++;
        ctn_len++;
        if (read_inf_or_nan(false, &cur, pre, flg, val)) goto obj_val_end;
        goto fail_character_val;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto obj_val_begin;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_val;

obj_val_end:
    if (likely(*cur == ',')) {
        cur++;
        goto obj_key_begin;
    }
    if (likely(*cur == '}')) {
        cur++;
        goto obj_end;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto obj_val_end;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto obj_val_end;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_obj_end;

obj_end:
    /* pop container */
    ctn_parent = (yyjson_val *)(void *)((u8 *)ctn - ctn->uni.ofs);
    /* point to the next value */
    ctn->uni.ofs = (usize)((u8 *)val - (u8 *)ctn) + sizeof(yyjson_val);
    ctn->tag = (ctn_len << (YYJSON_TAG_BIT - 1)) | YYJSON_TYPE_OBJ;
    if (unlikely(ctn == ctn_parent)) goto doc_end;
    ctn = ctn_parent;
    ctn_len = (usize)(ctn->tag >> YYJSON_TAG_BIT);
    if ((ctn->tag & YYJSON_TYPE_MASK) == YYJSON_TYPE_OBJ) {
        goto obj_val_end;
    } else {
        goto arr_val_end;
    }

doc_end:
    /* check invalid contents after json document */
    if (unlikely(cur < end) && !has_read_flag(STOP_WHEN_DONE)) {
        if (has_read_flag(ALLOW_COMMENTS)) {
            skip_spaces_and_comments(&cur);
            if (byte_match_2(cur, "/*")) goto fail_comment;
        } else {
            while (char_is_space(*cur)) cur++;
        }
        if (unlikely(cur < end)) goto fail_garbage;
    }

    if (pre && *pre) **pre = '\0';
    doc = (yyjson_doc *)val_hdr;
    doc->root = val_hdr + hdr_len;
    doc->alc = alc;
    doc->dat_read = (usize)(cur - hdr);
    doc->val_read = (usize)((val - doc->root) + 1);
    doc->str_pool = has_read_flag(INSITU) ? NULL : (char *)hdr;
    return doc;

fail_string:            return_err(cur, INVALID_STRING, msg);
fail_number:            return_err(cur, INVALID_NUMBER, msg);
fail_alloc:             return_err(cur, MEMORY_ALLOCATION, MSG_MALLOC);
fail_trailing_comma:    return_err(cur, JSON_STRUCTURE, MSG_COMMA);
fail_literal_true:      return_err(cur, LITERAL, MSG_CHAT_T);
fail_literal_false:     return_err(cur, LITERAL, MSG_CHAR_F);
fail_literal_null:      return_err(cur, LITERAL, MSG_CHAR_N);
fail_character_val:     return_err(cur, UNEXPECTED_CHARACTER, MSG_CHAR);
fail_character_arr_end: return_err(cur, UNEXPECTED_CHARACTER, MSG_ARR_END);
fail_character_obj_key: return_err(cur, UNEXPECTED_CHARACTER, MSG_OBJ_KEY);
fail_character_obj_sep: return_err(cur, UNEXPECTED_CHARACTER, MSG_OBJ_SEP);
fail_character_obj_end: return_err(cur, UNEXPECTED_CHARACTER, MSG_OBJ_END);
fail_comment:           return_err(cur, INVALID_COMMENT, MSG_COMMENT);
fail_garbage:           return_err(cur, UNEXPECTED_CONTENT, MSG_GARBAGE);

#undef val_incr
#undef return_err
}

/** Read JSON document (accept all style, but optimized for pretty). */
static_inline yyjson_doc *read_root_pretty(u8 *hdr, u8 *cur, u8 *end,
                                           yyjson_alc alc,
                                           yyjson_read_flag flg,
                                           yyjson_read_err *err) {

#define return_err(_pos, _code, _msg) do { \
    if (is_truncated_end(hdr, _pos, end, YYJSON_READ_ERROR_##_code, flg)) { \
        err->pos = (usize)(end - hdr); \
        err->code = YYJSON_READ_ERROR_UNEXPECTED_END; \
        err->msg = MSG_NOT_END; \
    } else { \
        err->pos = (usize)(_pos - hdr); \
        err->code = YYJSON_READ_ERROR_##_code; \
        err->msg = _msg; \
    } \
    if (val_hdr) alc.free(alc.ctx, (void *)val_hdr); \
    return NULL; \
} while (false)

#define val_incr() do { \
    val++; \
    if (unlikely(val >= val_end)) { \
        usize alc_old = alc_len; \
        usize val_ofs = (usize)(val - val_hdr); \
        usize ctn_ofs = (usize)(ctn - val_hdr); \
        alc_len += alc_len / 2; \
        if ((sizeof(usize) < 8) && (alc_len >= alc_max)) goto fail_alloc; \
        val_tmp = (yyjson_val *)alc.realloc(alc.ctx, (void *)val_hdr, \
            alc_old * sizeof(yyjson_val), \
            alc_len * sizeof(yyjson_val)); \
        if ((!val_tmp)) goto fail_alloc; \
        val = val_tmp + val_ofs; \
        ctn = val_tmp + ctn_ofs; \
        val_hdr = val_tmp; \
        val_end = val_tmp + (alc_len - 2); \
    } \
} while (false)

    usize dat_len; /* data length in bytes, hint for allocator */
    usize hdr_len; /* value count used by yyjson_doc */
    usize alc_len; /* value count allocated */
    usize alc_max; /* maximum value count for allocator */
    usize ctn_len; /* the number of elements in current container */
    yyjson_val *val_hdr; /* the head of allocated values */
    yyjson_val *val_end; /* the end of allocated values */
    yyjson_val *val_tmp; /* temporary pointer for realloc */
    yyjson_val *val; /* current JSON value */
    yyjson_val *ctn; /* current container */
    yyjson_val *ctn_parent; /* parent of current container */
    yyjson_doc *doc; /* the JSON document, equals to val_hdr */
    const char *msg; /* error message */

    bool raw; /* read number as raw */
    bool inv; /* allow invalid unicode */
    u8 *raw_end; /* raw end for null-terminator */
    u8 **pre; /* previous raw end pointer */

    dat_len = has_read_flag(STOP_WHEN_DONE) ? 256 : (usize)(end - cur);
    hdr_len = sizeof(yyjson_doc) / sizeof(yyjson_val);
    hdr_len += (sizeof(yyjson_doc) % sizeof(yyjson_val)) > 0;
    alc_max = USIZE_MAX / sizeof(yyjson_val);
    alc_len = hdr_len + (dat_len / YYJSON_READER_ESTIMATED_PRETTY_RATIO) + 4;
    alc_len = yyjson_min(alc_len, alc_max);

    val_hdr = (yyjson_val *)alc.malloc(alc.ctx, alc_len * sizeof(yyjson_val));
    if (unlikely(!val_hdr)) goto fail_alloc;
    val_end = val_hdr + (alc_len - 2); /* padding for key-value pair reading */
    val = val_hdr + hdr_len;
    ctn = val;
    ctn_len = 0;
    raw = has_read_flag(NUMBER_AS_RAW) || has_read_flag(BIGNUM_AS_RAW);
    inv = has_read_flag(ALLOW_INVALID_UNICODE) != 0;
    raw_end = NULL;
    pre = raw ? &raw_end : NULL;

    if (*cur++ == '{') {
        ctn->tag = YYJSON_TYPE_OBJ;
        ctn->uni.ofs = 0;
        if (*cur == '\n') cur++;
        goto obj_key_begin;
    } else {
        ctn->tag = YYJSON_TYPE_ARR;
        ctn->uni.ofs = 0;
        if (*cur == '\n') cur++;
        goto arr_val_begin;
    }

arr_begin:
    /* save current container */
    ctn->tag = (((u64)ctn_len + 1) << YYJSON_TAG_BIT) |
               (ctn->tag & YYJSON_TAG_MASK);

    /* create a new array value, save parent container offset */
    val_incr();
    val->tag = YYJSON_TYPE_ARR;
    val->uni.ofs = (usize)((u8 *)val - (u8 *)ctn);

    /* push the new array value as current container */
    ctn = val;
    ctn_len = 0;
    if (*cur == '\n') cur++;

arr_val_begin:
#if YYJSON_IS_REAL_GCC
    while (true) repeat16({
        if (byte_match_2(cur, "  ")) cur += 2;
        else break;
    })
#else
    while (true) repeat16({
        if (likely(byte_match_2(cur, "  "))) cur += 2;
        else break;
    })
#endif

    if (*cur == '{') {
        cur++;
        goto obj_begin;
    }
    if (*cur == '[') {
        cur++;
        goto arr_begin;
    }
    if (char_is_num(*cur)) {
        val_incr();
        ctn_len++;
        if (likely(read_num(&cur, pre, flg, val, &msg))) goto arr_val_end;
        goto fail_number;
    }
    if (*cur == '"') {
        val_incr();
        ctn_len++;
        if (likely(read_str(&cur, end, inv, val, &msg, NULL))) goto arr_val_end;
        goto fail_string;
    }
    if (*cur == 't') {
        val_incr();
        ctn_len++;
        if (likely(read_true(&cur, val))) goto arr_val_end;
        goto fail_literal_true;
    }
    if (*cur == 'f') {
        val_incr();
        ctn_len++;
        if (likely(read_false(&cur, val))) goto arr_val_end;
        goto fail_literal_false;
    }
    if (*cur == 'n') {
        val_incr();
        ctn_len++;
        if (likely(read_null(&cur, val))) goto arr_val_end;
        if (has_read_flag(ALLOW_INF_AND_NAN)) {
            if (read_nan(false, &cur, pre, flg, val)) goto arr_val_end;
        }
        goto fail_literal_null;
    }
    if (*cur == ']') {
        cur++;
        if (likely(ctn_len == 0)) goto arr_end;
        if (has_read_flag(ALLOW_TRAILING_COMMAS)) goto arr_end;
        while (*cur != ',') cur--;
        goto fail_trailing_comma;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto arr_val_begin;
    }
    if (has_read_flag(ALLOW_INF_AND_NAN) &&
        (*cur == 'i' || *cur == 'I' || *cur == 'N')) {
        val_incr();
        ctn_len++;
        if (read_inf_or_nan(false, &cur, pre, flg, val)) goto arr_val_end;
        goto fail_character_val;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto arr_val_begin;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_val;

arr_val_end:
    if (byte_match_2(cur, ",\n")) {
        cur += 2;
        goto arr_val_begin;
    }
    if (*cur == ',') {
        cur++;
        goto arr_val_begin;
    }
    if (*cur == ']') {
        cur++;
        goto arr_end;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto arr_val_end;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto arr_val_end;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_arr_end;

arr_end:
    /* get parent container */
    ctn_parent = (yyjson_val *)(void *)((u8 *)ctn - ctn->uni.ofs);

    /* save the next sibling value offset */
    ctn->uni.ofs = (usize)((u8 *)val - (u8 *)ctn) + sizeof(yyjson_val);
    ctn->tag = ((ctn_len) << YYJSON_TAG_BIT) | YYJSON_TYPE_ARR;
    if (unlikely(ctn == ctn_parent)) goto doc_end;

    /* pop parent as current container */
    ctn = ctn_parent;
    ctn_len = (usize)(ctn->tag >> YYJSON_TAG_BIT);
    if (*cur == '\n') cur++;
    if ((ctn->tag & YYJSON_TYPE_MASK) == YYJSON_TYPE_OBJ) {
        goto obj_val_end;
    } else {
        goto arr_val_end;
    }

obj_begin:
    /* push container */
    ctn->tag = (((u64)ctn_len + 1) << YYJSON_TAG_BIT) |
               (ctn->tag & YYJSON_TAG_MASK);
    val_incr();
    val->tag = YYJSON_TYPE_OBJ;
    /* offset to the parent */
    val->uni.ofs = (usize)((u8 *)val - (u8 *)ctn);
    ctn = val;
    ctn_len = 0;
    if (*cur == '\n') cur++;

obj_key_begin:
#if YYJSON_IS_REAL_GCC
    while (true) repeat16({
        if (byte_match_2(cur, "  ")) cur += 2;
        else break;
    })
#else
    while (true) repeat16({
        if (likely(byte_match_2(cur, "  "))) cur += 2;
        else break;
    })
#endif
    if (likely(*cur == '"')) {
        val_incr();
        ctn_len++;
        if (likely(read_str(&cur, end, inv, val, &msg, NULL))) goto obj_key_end;
        goto fail_string;
    }
    if (likely(*cur == '}')) {
        cur++;
        if (likely(ctn_len == 0)) goto obj_end;
        if (has_read_flag(ALLOW_TRAILING_COMMAS)) goto obj_end;
        while (*cur != ',') cur--;
        goto fail_trailing_comma;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto obj_key_begin;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto obj_key_begin;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_obj_key;

obj_key_end:
    if (byte_match_2(cur, ": ")) {
        cur += 2;
        goto obj_val_begin;
    }
    if (*cur == ':') {
        cur++;
        goto obj_val_begin;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto obj_key_end;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto obj_key_end;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_obj_sep;

obj_val_begin:
    if (*cur == '"') {
        val++;
        ctn_len++;
        if (likely(read_str(&cur, end, inv, val, &msg, NULL))) goto obj_val_end;
        goto fail_string;
    }
    if (char_is_num(*cur)) {
        val++;
        ctn_len++;
        if (likely(read_num(&cur, pre, flg, val, &msg))) goto obj_val_end;
        goto fail_number;
    }
    if (*cur == '{') {
        cur++;
        goto obj_begin;
    }
    if (*cur == '[') {
        cur++;
        goto arr_begin;
    }
    if (*cur == 't') {
        val++;
        ctn_len++;
        if (likely(read_true(&cur, val))) goto obj_val_end;
        goto fail_literal_true;
    }
    if (*cur == 'f') {
        val++;
        ctn_len++;
        if (likely(read_false(&cur, val))) goto obj_val_end;
        goto fail_literal_false;
    }
    if (*cur == 'n') {
        val++;
        ctn_len++;
        if (likely(read_null(&cur, val))) goto obj_val_end;
        if (has_read_flag(ALLOW_INF_AND_NAN)) {
            if (read_nan(false, &cur, pre, flg, val)) goto obj_val_end;
        }
        goto fail_literal_null;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto obj_val_begin;
    }
    if (has_read_flag(ALLOW_INF_AND_NAN) &&
        (*cur == 'i' || *cur == 'I' || *cur == 'N')) {
        val++;
        ctn_len++;
        if (read_inf_or_nan(false, &cur, pre, flg, val)) goto obj_val_end;
        goto fail_character_val;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto obj_val_begin;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_val;

obj_val_end:
    if (byte_match_2(cur, ",\n")) {
        cur += 2;
        goto obj_key_begin;
    }
    if (likely(*cur == ',')) {
        cur++;
        goto obj_key_begin;
    }
    if (likely(*cur == '}')) {
        cur++;
        goto obj_end;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto obj_val_end;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto obj_val_end;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_obj_end;

obj_end:
    /* pop container */
    ctn_parent = (yyjson_val *)(void *)((u8 *)ctn - ctn->uni.ofs);
    /* point to the next value */
    ctn->uni.ofs = (usize)((u8 *)val - (u8 *)ctn) + sizeof(yyjson_val);
    ctn->tag = (ctn_len << (YYJSON_TAG_BIT - 1)) | YYJSON_TYPE_OBJ;
    if (unlikely(ctn == ctn_parent)) goto doc_end;
    ctn = ctn_parent;
    ctn_len = (usize)(ctn->tag >> YYJSON_TAG_BIT);
    if (*cur == '\n') cur++;
    if ((ctn->tag & YYJSON_TYPE_MASK) == YYJSON_TYPE_OBJ) {
        goto obj_val_end;
    } else {
        goto arr_val_end;
    }

doc_end:
    /* check invalid contents after json document */
    if (unlikely(cur < end) && !has_read_flag(STOP_WHEN_DONE)) {
        if (has_read_flag(ALLOW_COMMENTS)) {
            skip_spaces_and_comments(&cur);
            if (byte_match_2(cur, "/*")) goto fail_comment;
        } else {
            while (char_is_space(*cur)) cur++;
        }
        if (unlikely(cur < end)) goto fail_garbage;
    }

    if (pre && *pre) **pre = '\0';
    doc = (yyjson_doc *)val_hdr;
    doc->root = val_hdr + hdr_len;
    doc->alc = alc;
    doc->dat_read = (usize)(cur - hdr);
    doc->val_read = (usize)((val - doc->root) + 1);
    doc->str_pool = has_read_flag(INSITU) ? NULL : (char *)hdr;
    return doc;

fail_string:            return_err(cur, INVALID_STRING, msg);
fail_number:            return_err(cur, INVALID_NUMBER, msg);
fail_alloc:             return_err(cur, MEMORY_ALLOCATION, MSG_MALLOC);
fail_trailing_comma:    return_err(cur, JSON_STRUCTURE, MSG_COMMA);
fail_literal_true:      return_err(cur, LITERAL, MSG_CHAT_T);
fail_literal_false:     return_err(cur, LITERAL, MSG_CHAR_F);
fail_literal_null:      return_err(cur, LITERAL, MSG_CHAR_N);
fail_character_val:     return_err(cur, UNEXPECTED_CHARACTER, MSG_CHAR);
fail_character_arr_end: return_err(cur, UNEXPECTED_CHARACTER, MSG_ARR_END);
fail_character_obj_key: return_err(cur, UNEXPECTED_CHARACTER, MSG_OBJ_KEY);
fail_character_obj_sep: return_err(cur, UNEXPECTED_CHARACTER, MSG_OBJ_SEP);
fail_character_obj_end: return_err(cur, UNEXPECTED_CHARACTER, MSG_OBJ_END);
fail_comment:           return_err(cur, INVALID_COMMENT, MSG_COMMENT);
fail_garbage:           return_err(cur, UNEXPECTED_CONTENT, MSG_GARBAGE);

#undef val_incr
#undef return_err
}



/*==============================================================================
 * JSON Reader Entrance
 *============================================================================*/

yyjson_doc *yyjson_read_opts(char *dat, usize len,
                             yyjson_read_flag flg,
                             const yyjson_alc *alc_ptr,
                             yyjson_read_err *err) {

#define return_err(_pos, _code, _msg) do { \
    err->pos = (usize)(_pos); \
    err->msg = _msg; \
    err->code = YYJSON_READ_ERROR_##_code; \
    if (!has_read_flag(INSITU) && hdr) alc.free(alc.ctx, (void *)hdr); \
    return NULL; \
} while (false)

    yyjson_read_err dummy_err;
    yyjson_alc alc = alc_ptr ? *alc_ptr : YYJSON_DEFAULT_ALC;
    yyjson_doc *doc;
    u8 *hdr = NULL, *end, *cur;

    /* validate input parameters */
    if (!err) err = &dummy_err;
    if (unlikely(!dat)) return_err(0, INVALID_PARAMETER, "input data is NULL");
    if (unlikely(!len)) return_err(0, INVALID_PARAMETER, "input length is 0");

    /* add 4-byte zero padding for input data if necessary */
    if (has_read_flag(INSITU)) {
        hdr = (u8 *)dat;
        end = (u8 *)dat + len;
        cur = (u8 *)dat;
    } else {
        if (unlikely(len >= USIZE_MAX - YYJSON_PADDING_SIZE)) {
            return_err(0, MEMORY_ALLOCATION, MSG_MALLOC);
        }
        hdr = (u8 *)alc.malloc(alc.ctx, len + YYJSON_PADDING_SIZE);
        if (unlikely(!hdr)) {
            return_err(0, MEMORY_ALLOCATION, MSG_MALLOC);
        }
        end = hdr + len;
        cur = hdr;
        memcpy(hdr, dat, len);
        memset(end, 0, YYJSON_PADDING_SIZE);
    }

    if (has_read_flag(ALLOW_BOM)) {
        if (len >= 3 && is_utf8_bom(cur)) cur += 3;
    }

    /* skip empty contents before json document */
    if (unlikely(char_is_space_or_comment(*cur))) {
        if (has_read_flag(ALLOW_COMMENTS)) {
            if (!skip_spaces_and_comments(&cur)) {
                return_err(cur - hdr, INVALID_COMMENT, MSG_COMMENT);
            }
        } else {
            if (likely(char_is_space(*cur))) {
                while (char_is_space(*++cur));
            }
        }
        if (unlikely(cur >= end)) {
            return_err(0, EMPTY_CONTENT, "input data is empty");
        }
    }

    /* read json document */
    if (likely(char_is_container(*cur))) {
        if (char_is_space(cur[1]) && char_is_space(cur[2])) {
            doc = read_root_pretty(hdr, cur, end, alc, flg, err);
        } else {
            doc = read_root_minify(hdr, cur, end, alc, flg, err);
        }
    } else {
        doc = read_root_single(hdr, cur, end, alc, flg, err);
    }

    /* check result */
    if (likely(doc)) {
        memset(err, 0, sizeof(yyjson_read_err));
    } else {
        /* RFC 8259: JSON text MUST be encoded using UTF-8 */
        if (err->pos == 0 && err->code != YYJSON_READ_ERROR_MEMORY_ALLOCATION) {
            if (is_utf8_bom(hdr)) err->msg = MSG_ERR_BOM;
            else if (len >= 4 && is_utf32_bom(hdr)) err->msg = MSG_ERR_UTF32;
            else if (len >= 2 && is_utf16_bom(hdr)) err->msg = MSG_ERR_UTF16;
        }
        if (!has_read_flag(INSITU)) alc.free(alc.ctx, (void *)hdr);
    }
    return doc;

#undef return_err
}



#if !YYJSON_DISABLE_INCR_READER

/* labels within yyjson_incr_read() to resume incremental parsing */
#define YYJSON_READ_LABEL_doc_begin 0
#define YYJSON_READ_LABEL_arr_val_begin 1
#define YYJSON_READ_LABEL_arr_val_end 2
#define YYJSON_READ_LABEL_obj_key_begin 3
#define YYJSON_READ_LABEL_obj_key_end 4
#define YYJSON_READ_LABEL_obj_val_begin 5
#define YYJSON_READ_LABEL_obj_val_end 6
#define YYJSON_READ_LABEL_doc_end 7

/** State for incremental JSON reader, opaque in the API. */
struct yyjson_incr_state {
    u32 label; /* current parser goto label */
    const yyjson_alc *alc; /* allocator */
    yyjson_read_flag flg; /* read flags */
    u8 *hdr; /* JSON data */
    u8 *cur; /* current position in JSON data */
    usize len;
    usize hdr_len; /* value count used by yyjson_doc */
    usize alc_len; /* value count allocated */
    usize ctn_len; /* the number of elements in current container */
    yyjson_val *val_hdr; /* the head of allocated values */
    yyjson_val *val_end; /* the end of allocated values */
    yyjson_val *val; /* current JSON value */
    yyjson_val *ctn; /* current container */
    u8 *str_con[2]; /* string parser incremental state */
};

yyjson_incr_state *yyjson_incr_new(char *buf, size_t buf_len,
                                   yyjson_read_flag flg,
                                   const yyjson_alc *alc) {
    yyjson_incr_state *state = NULL;
    if (unlikely(!buf)) goto error;
    if (likely(!alc)) alc = &YYJSON_DEFAULT_ALC;
    state = (yyjson_incr_state *)alc->malloc(alc->ctx,
                                             sizeof(yyjson_incr_state));
    if (!state) goto error;
    memset(state, 0, sizeof(yyjson_incr_state));
    state->alc = alc;
    state->flg = flg;
    state->len = buf_len;

    /* add 4-byte zero padding for input data if necessary */
    if (has_read_flag(INSITU)) {
        state->hdr = (u8 *)buf;
        state->cur = (u8 *)buf;
    } else {
        if (unlikely(buf_len >= USIZE_MAX - YYJSON_PADDING_SIZE)) goto error;
        state->hdr = (u8 *)alc->malloc(alc->ctx, buf_len + YYJSON_PADDING_SIZE);
        if (unlikely(!state->hdr)) goto error;
        state->cur = state->hdr;
        memcpy(state->hdr, buf, buf_len);
        memset(state->hdr + buf_len, 0, YYJSON_PADDING_SIZE);
    }
    return state;

error:
    if (state) yyjson_incr_free(state);
    return NULL;
}

void yyjson_incr_free(yyjson_incr_state *state) {
    const yyjson_alc *alc = state->alc;
    if (state->val_hdr != NULL) {
        alc->free(alc->ctx, (void *)state->val_hdr);
    }
    if (state->hdr != NULL && !(state->flg & YYJSON_READ_INSITU)) {
        alc->free(alc->ctx, (void *)state->hdr);
    }
    alc->free(alc->ctx, (void *)state);
}

yyjson_doc *yyjson_incr_read(yyjson_incr_state *state, size_t len,
                             yyjson_read_err *err) {

#define return_err_inv_param(_msg) do { \
    err->pos = 0; \
    err->msg = _msg; \
    err->code = YYJSON_READ_ERROR_INVALID_PARAMETER; \
    return NULL; \
} while (false)

#define return_err(_pos, _code, _msg) do { \
    if (is_truncated_end(hdr, _pos, end, YYJSON_READ_ERROR_##_code, flg)) { \
        goto unexpected_end; \
    } else { \
        err->pos = (usize)(_pos - hdr); \
        err->code = YYJSON_READ_ERROR_##_code; \
        err->msg = _msg; \
    } \
    return NULL; \
} while (false)

#define val_incr() do { \
    val++; \
    if (unlikely(val >= val_end)) { \
        usize alc_old = alc_len; \
        alc_len += alc_len / 2; \
        if ((sizeof(usize) < 8) && (alc_len >= alc_max)) goto fail_alloc; \
        val_tmp = (yyjson_val *)alc.realloc(alc.ctx, (void *)val_hdr, \
                                            alc_old * sizeof(yyjson_val), \
                                            alc_len * sizeof(yyjson_val)); \
        if ((!val_tmp)) goto fail_alloc; \
        val = val_tmp + (usize)(val - val_hdr); \
        ctn = val_tmp + (usize)(ctn - val_hdr); \
        state->val = val_tmp + (usize)(state->val - val_hdr); \
        state->val_hdr = val_hdr = val_tmp; \
        val_end = val_tmp + (alc_len - 2); \
        state->val_end = val_end; \
    } \
} while (false)

#define save_incr_state(_label) do { \
    /* save position where it's possible to resume incremental parsing */ \
    state->label = YYJSON_READ_LABEL_##_label; \
    state->cur = cur; \
    state->val = val; \
    state->ctn_len = ctn_len; \
    state->hdr_len = hdr_len; \
    if (unlikely(cur >= end)) goto unexpected_end; \
} while (false)

#define check_maybe_truncated_number() do { \
    if (unlikely(cur >= end)) { \
        if (unlikely(cur > state->cur + INCR_NUM_MAX_LEN)) { \
            msg = "number too long"; \
            goto fail_number; \
        } \
        goto unexpected_end; \
    } \
} while (false)

    u8 *hdr = NULL, *end = NULL, *cur = NULL;
    yyjson_read_flag flg;
    yyjson_alc alc;
    usize dat_len; /* data length in bytes, hint for allocator */
    usize hdr_len; /* value count used by yyjson_doc */
    usize alc_len; /* value count allocated */
    usize alc_max; /* maximum value count for allocator */
    usize ctn_len; /* the number of elements in current container */
    yyjson_val *val_hdr; /* the head of allocated values */
    yyjson_val *val_end; /* the end of allocated values */
    yyjson_val *val_tmp; /* temporary pointer for realloc */
    yyjson_val *val; /* current JSON value */
    yyjson_val *ctn; /* current container */
    yyjson_val *ctn_parent; /* parent of current container */
    yyjson_doc *doc; /* the JSON document, equals to val_hdr */
    const char *msg; /* error message */

    bool raw; /* read number as raw */
    bool inv; /* allow invalid unicode */
    u8 *raw_end; /* raw end for null-terminator */
    u8 **pre; /* previous raw end pointer */
    u8 **con = NULL; /* for incremental string parsing */
    u8 saved_end = '\0'; /* saved end char */

    /* validate input parameters */
    if (unlikely(!err)) {
        return NULL;
    }
    if (unlikely(!state)) {
        return_err_inv_param("input state is NULL");
    }
    if (unlikely(!len)) {
        return_err_inv_param("input length is 0");
    }
    if (unlikely(len > state->len)) {
        return_err_inv_param("length is greater than total input length");
    }

    hdr = state->hdr;
    end = state->hdr + len;
    cur = state->cur;
    flg = state->flg;
    alc = *state->alc;
    ctn_len = state->ctn_len;
    hdr_len = state->hdr_len;
    alc_len = state->alc_len;
    val = state->val;
    val_hdr = state->val_hdr;
    val_end = state->val_end;
    ctn = state->ctn;
    con = state->str_con;

    alc_max = USIZE_MAX / sizeof(yyjson_val);
    raw = has_read_flag(NUMBER_AS_RAW) || has_read_flag(BIGNUM_AS_RAW);
    inv = has_read_flag(ALLOW_INVALID_UNICODE) != 0;
    raw_end = NULL;
    pre = raw ? &raw_end : NULL;

    /* insert null terminator to make us stop at the specified end, even if
       the data contains more valid JSON */
    saved_end = *end;
    *end = '\0';

    /* resume parsing from the last save point */
    switch (state->label) {
    case YYJSON_READ_LABEL_doc_begin: goto doc_begin;
    case YYJSON_READ_LABEL_arr_val_begin: goto arr_val_begin;
    case YYJSON_READ_LABEL_arr_val_end: goto arr_val_end;
    case YYJSON_READ_LABEL_obj_key_begin: goto obj_key_begin;
    case YYJSON_READ_LABEL_obj_key_end: goto obj_key_end;
    case YYJSON_READ_LABEL_obj_val_begin: goto obj_val_begin;
    case YYJSON_READ_LABEL_obj_val_end: goto obj_val_end;
    case YYJSON_READ_LABEL_doc_end: goto doc_end;
    default: return_err_inv_param("invalid incremental state");
    }

doc_begin:
    if (cur == hdr && has_read_flag(ALLOW_BOM)) {
        if (len >= 3 && is_utf8_bom(cur)) cur += 3;
    }

    /* skip empty contents before json document */
    if (unlikely(char_is_space_or_comment(*cur))) {
        if (has_read_flag(ALLOW_COMMENTS)) {
            if (!skip_spaces_and_comments(&cur)) {
                /* unclosed multiline comment */
                goto unexpected_end;
            }
        } else {
            if (likely(char_is_space(*cur))) {
                while (char_is_space(*++cur));
            }
        }
        if (unlikely(cur >= end)) {
            /* input data is empty */
            goto unexpected_end;
        }
    }

    /* allocate memory for document */
    if (!val_hdr) {
        hdr_len = sizeof(yyjson_doc) / sizeof(yyjson_val);
        hdr_len += (sizeof(yyjson_doc) % sizeof(yyjson_val)) > 0;
        if (likely(char_is_container(*cur))) {
            dat_len = has_read_flag(STOP_WHEN_DONE) ? 256 : state->len;
            alc_len = hdr_len +
                    (dat_len / YYJSON_READER_ESTIMATED_MINIFY_RATIO) + 4;
            alc_len = yyjson_min(alc_len, alc_max);
        } else {
            alc_len = hdr_len + 1; /* single value */
        }
        val_hdr = (yyjson_val *)alc.malloc(alc.ctx,
                                           alc_len * sizeof(yyjson_val));
        if (unlikely(!val_hdr)) goto fail_alloc;
        val_end = val_hdr + (alc_len - 2); /* padding for kv pair reading */
        val = val_hdr + hdr_len;
        ctn = val;
        ctn_len = 0;
        state->val_hdr = val_hdr;
        state->val_end = val_end;
        save_incr_state(doc_begin);
    }

    /* read json document */
    if (*cur == '{') {
        cur++;
        ctn->tag = YYJSON_TYPE_OBJ;
        ctn->uni.ofs = 0;
        goto obj_key_begin;
    }
    if (*cur == '[') {
        cur++;
        ctn->tag = YYJSON_TYPE_ARR;
        ctn->uni.ofs = 0;
        goto arr_val_begin;
    }
    if (char_is_num(*cur)) {
        if (likely(read_num(&cur, pre, flg, val, &msg))) goto doc_end;
        goto fail_number;
    }
    if (*cur == '"') {
        if (likely(read_str(&cur, end, inv, val, &msg, con))) goto doc_end;
        goto fail_string;
    }
    if (*cur == 't') {
        if (likely(read_true(&cur, val))) goto doc_end;
        goto fail_literal_true;
    }
    if (*cur == 'f') {
        if (likely(read_false(&cur, val))) goto doc_end;
        goto fail_literal_false;
    }
    if (*cur == 'n') {
        if (likely(read_null(&cur, val))) goto doc_end;
        if (has_read_flag(ALLOW_INF_AND_NAN)) {
            if (read_nan(false, &cur, pre, flg, val)) goto doc_end;
        }
        goto fail_literal_null;
    }
    if (has_read_flag(ALLOW_INF_AND_NAN)) {
        if (read_inf_or_nan(false, &cur, pre, flg, val)) goto doc_end;
    }

    msg = "unexpected character, expected a valid root value";
    if (cur == hdr) {
        /* RFC 8259: JSON text MUST be encoded using UTF-8 */
        if (is_utf8_bom(hdr)) msg = MSG_ERR_BOM;
        else if (len >= 4 && is_utf32_bom(hdr)) msg = MSG_ERR_UTF32;
        else if (len >= 2 && is_utf16_bom(hdr)) msg = MSG_ERR_UTF16;
    }
    return_err(cur, UNEXPECTED_CHARACTER, msg);

arr_begin:
    /* save current container */
    ctn->tag = (((u64)ctn_len + 1) << YYJSON_TAG_BIT) |
               (ctn->tag & YYJSON_TAG_MASK);

    /* create a new array value, save parent container offset */
    val_incr();
    val->tag = YYJSON_TYPE_ARR;
    val->uni.ofs = (usize)((u8 *)val - (u8 *)ctn);

    /* push the new array value as current container */
    ctn = val;
    ctn_len = 0;

arr_val_begin:
    save_incr_state(arr_val_begin);
arr_val_continue:
    if (*cur == '{') {
        cur++;
        goto obj_begin;
    }
    if (*cur == '[') {
        cur++;
        goto arr_begin;
    }
    if (char_is_num(*cur)) {
        val_incr();
        ctn_len++;
        if (likely(read_num(&cur, pre, flg, val, &msg))) goto arr_val_maybe_end;
        goto fail_number;
    }
    if (*cur == '"') {
        val_incr();
        ctn_len++;
        if (likely(read_str(&cur, end, inv, val, &msg, con))) goto arr_val_end;
        goto fail_string;
    }
    if (*cur == 't') {
        val_incr();
        ctn_len++;
        if (likely(read_true(&cur, val))) goto arr_val_end;
        goto fail_literal_true;
    }
    if (*cur == 'f') {
        val_incr();
        ctn_len++;
        if (likely(read_false(&cur, val))) goto arr_val_end;
        goto fail_literal_false;
    }
    if (*cur == 'n') {
        val_incr();
        ctn_len++;
        if (likely(read_null(&cur, val))) goto arr_val_end;
        if (has_read_flag(ALLOW_INF_AND_NAN)) {
            if (read_nan(false, &cur, pre, flg, val)) goto arr_val_end;
        }
        goto fail_literal_null;
    }
    if (*cur == ']') {
        cur++;
        if (likely(ctn_len == 0)) goto arr_end;
        if (has_read_flag(ALLOW_TRAILING_COMMAS)) goto arr_end;
        while (*cur != ',') cur--;
        goto fail_trailing_comma;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto arr_val_continue;
    }
    if (has_read_flag(ALLOW_INF_AND_NAN) &&
        (*cur == 'i' || *cur == 'I' || *cur == 'N')) {
        val_incr();
        ctn_len++;
        if (read_inf_or_nan(false, &cur, pre, flg, val)) goto arr_val_maybe_end;
        goto fail_character_val;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto arr_val_continue;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_val;

arr_val_maybe_end:
    /* if incremental parsing stops in the middle of a number, it may continue
       with more digits, so arr val maybe didn't end yet */
    check_maybe_truncated_number();

arr_val_end:
    save_incr_state(arr_val_end);
    if (*cur == ',') {
        cur++;
        goto arr_val_begin;
    }
    if (*cur == ']') {
        cur++;
        goto arr_end;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto arr_val_end;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto arr_val_end;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_arr_end;

arr_end:
    /* get parent container */
    ctn_parent = (yyjson_val *)(void *)((u8 *)ctn - ctn->uni.ofs);

    /* save the next sibling value offset */
    ctn->uni.ofs = (usize)((u8 *)val - (u8 *)ctn) + sizeof(yyjson_val);
    ctn->tag = ((ctn_len) << YYJSON_TAG_BIT) | YYJSON_TYPE_ARR;
    if (unlikely(ctn == ctn_parent)) goto doc_end;

    /* pop parent as current container */
    ctn = ctn_parent;
    ctn_len = (usize)(ctn->tag >> YYJSON_TAG_BIT);
    if ((ctn->tag & YYJSON_TYPE_MASK) == YYJSON_TYPE_OBJ) {
        goto obj_val_end;
    } else {
        goto arr_val_end;
    }

obj_begin:
    /* push container */
    ctn->tag = (((u64)ctn_len + 1) << YYJSON_TAG_BIT) |
               (ctn->tag & YYJSON_TAG_MASK);
    val_incr();
    val->tag = YYJSON_TYPE_OBJ;
    /* offset to the parent */
    val->uni.ofs = (usize)((u8 *)val - (u8 *)ctn);
    ctn = val;
    ctn_len = 0;

obj_key_begin:
    save_incr_state(obj_key_begin);
obj_key_continue:
    if (likely(*cur == '"')) {
        val_incr();
        ctn_len++;
        if (likely(read_str(&cur, end, inv, val, &msg, con))) goto obj_key_end;
        goto fail_string;
    }
    if (likely(*cur == '}')) {
        cur++;
        if (likely(ctn_len == 0)) goto obj_end;
        if (has_read_flag(ALLOW_TRAILING_COMMAS)) goto obj_end;
        while (*cur != ',') cur--;
        goto fail_trailing_comma;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto obj_key_continue;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto obj_key_continue;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_obj_key;

obj_key_end:
    save_incr_state(obj_key_end);
    if (*cur == ':') {
        cur++;
        goto obj_val_begin;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto obj_key_end;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto obj_key_end;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_obj_sep;

obj_val_begin:
    save_incr_state(obj_val_begin);
obj_val_continue:
    if (*cur == '"') {
        val++;
        ctn_len++;
        if (likely(read_str(&cur, end, inv, val, &msg, con))) goto obj_val_end;
        goto fail_string;
    }
    if (char_is_num(*cur)) {
        val++;
        ctn_len++;
        if (likely(read_num(&cur, pre, flg, val, &msg))) goto obj_val_maybe_end;
        goto fail_number;
    }
    if (*cur == '{') {
        cur++;
        goto obj_begin;
    }
    if (*cur == '[') {
        cur++;
        goto arr_begin;
    }
    if (*cur == 't') {
        val++;
        ctn_len++;
        if (likely(read_true(&cur, val))) goto obj_val_end;
        goto fail_literal_true;
    }
    if (*cur == 'f') {
        val++;
        ctn_len++;
        if (likely(read_false(&cur, val))) goto obj_val_end;
        goto fail_literal_false;
    }
    if (*cur == 'n') {
        val++;
        ctn_len++;
        if (likely(read_null(&cur, val))) goto obj_val_end;
        if (has_read_flag(ALLOW_INF_AND_NAN)) {
            if (read_nan(false, &cur, pre, flg, val)) goto obj_val_end;
        }
        goto fail_literal_null;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto obj_val_continue;
    }
    if (has_read_flag(ALLOW_INF_AND_NAN) &&
        (*cur == 'i' || *cur == 'I' || *cur == 'N')) {
        val++;
        ctn_len++;
        if (read_inf_or_nan(false, &cur, pre, flg, val)) goto obj_val_maybe_end;
        goto fail_character_val;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto obj_val_continue;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_val;

obj_val_maybe_end:
    /* if incremental parsing stops in the middle of a number, it may continue
       with more digits, so obj val maybe didn't end yet */
    check_maybe_truncated_number();

obj_val_end:
    save_incr_state(obj_val_end);
    if (likely(*cur == ',')) {
        cur++;
        goto obj_key_begin;
    }
    if (likely(*cur == '}')) {
        cur++;
        goto obj_end;
    }
    if (char_is_space(*cur)) {
        while (char_is_space(*++cur));
        goto obj_val_end;
    }
    if (has_read_flag(ALLOW_COMMENTS)) {
        if (skip_spaces_and_comments(&cur)) goto obj_val_end;
        if (byte_match_2(cur, "/*")) goto fail_comment;
    }
    goto fail_character_obj_end;

obj_end:
    /* pop container */
    ctn_parent = (yyjson_val *)(void *)((u8 *)ctn - ctn->uni.ofs);
    /* point to the next value */
    ctn->uni.ofs = (usize)((u8 *)val - (u8 *)ctn) + sizeof(yyjson_val);
    ctn->tag = (ctn_len << (YYJSON_TAG_BIT - 1)) | YYJSON_TYPE_OBJ;
    if (unlikely(ctn == ctn_parent)) goto doc_end;
    ctn = ctn_parent;
    ctn_len = (usize)(ctn->tag >> YYJSON_TAG_BIT);
    if ((ctn->tag & YYJSON_TYPE_MASK) == YYJSON_TYPE_OBJ) {
        goto obj_val_end;
    } else {
        goto arr_val_end;
    }

doc_end:
    /* check invalid contents after json document */
    if (unlikely(cur < end) && !has_read_flag(STOP_WHEN_DONE)) {
        save_incr_state(doc_end);
        if (has_read_flag(ALLOW_COMMENTS)) {
            skip_spaces_and_comments(&cur);
            if (byte_match_2(cur, "/*")) goto fail_comment;
            if (*cur == '/' && cur + 1 == end) {
                /* truncated beginning of comment */
                goto unexpected_end;
            }
        } else {
            while (char_is_space(*cur)) cur++;
        }
        if (unlikely(cur < end)) goto fail_garbage;
    }

    if (pre && *pre) **pre = '\0';
    doc = (yyjson_doc *)val_hdr;
    doc->root = val_hdr + hdr_len;
    doc->alc = alc;
    doc->dat_read = (usize)(cur - hdr);
    doc->val_read = (usize)((val - doc->root) + 1);
    doc->str_pool = has_read_flag(INSITU) ? NULL : (char *)hdr;
    state->hdr = NULL;
    state->val_hdr = NULL;
    memset(err, 0, sizeof(yyjson_read_err));
    return doc;

unexpected_end:
    err->pos = len;
    if (unlikely(len >= state->len)) {
        err->code = YYJSON_READ_ERROR_UNEXPECTED_END;
        err->msg = MSG_NOT_END;
        return NULL;
    }
    /* save parser state in extended error struct, in addition to what was
     * stored in the last save_incr_state */
    err->code = YYJSON_READ_ERROR_MORE;
    err->msg = "need more data";
    state->val_end = val_end;
    state->ctn = ctn;
    state->alc_len = alc_len;
    /* restore the end where we've inserted a null terminator */
    *end = saved_end;
    return NULL;

fail_string:            return_err(cur, INVALID_STRING, msg);
fail_number:            return_err(cur, INVALID_NUMBER, msg);
fail_alloc:             return_err(cur, MEMORY_ALLOCATION, MSG_MALLOC);
fail_trailing_comma:    return_err(cur, JSON_STRUCTURE, MSG_COMMA);
fail_literal_true:      return_err(cur, LITERAL, MSG_CHAT_T);
fail_literal_false:     return_err(cur, LITERAL, MSG_CHAR_F);
fail_literal_null:      return_err(cur, LITERAL, MSG_CHAR_N);
fail_character_val:     return_err(cur, UNEXPECTED_CHARACTER, MSG_CHAR);
fail_character_arr_end: return_err(cur, UNEXPECTED_CHARACTER, MSG_ARR_END);
fail_character_obj_key: return_err(cur, UNEXPECTED_CHARACTER, MSG_OBJ_KEY);
fail_character_obj_sep: return_err(cur, UNEXPECTED_CHARACTER, MSG_OBJ_SEP);
fail_character_obj_end: return_err(cur, UNEXPECTED_CHARACTER, MSG_OBJ_END);
fail_comment:           return_err(cur, INVALID_COMMENT, MSG_COMMENT);
fail_garbage:           return_err(cur, UNEXPECTED_CONTENT, MSG_GARBAGE);

#undef val_incr
#undef return_err
#undef return_err_inv_param
#undef save_incr_state
#undef check_maybe_truncated_number
}

#endif /* YYJSON_DISABLE_INCR_READER */



yyjson_doc *yyjson_read_file(const char *path,
                             yyjson_read_flag flg,
                             const yyjson_alc *alc_ptr,
                             yyjson_read_err *err) {
#define return_err(_code, _msg) do { \
    err->pos = 0; \
    err->msg = _msg; \
    err->code = YYJSON_READ_ERROR_##_code; \
    return NULL; \
} while (false)

    yyjson_read_err dummy_err;
    yyjson_doc *doc;
    FILE *file;

    if (!err) err = &dummy_err;
    if (unlikely(!path)) return_err(INVALID_PARAMETER, "input path is NULL");

    file = fopen_readonly(path);
    if (unlikely(!file)) return_err(FILE_OPEN, MSG_FREAD);

    doc = yyjson_read_fp(file, flg, alc_ptr, err);
    fclose(file);
    return doc;

#undef return_err
}

yyjson_doc *yyjson_read_fp(FILE *file,
                           yyjson_read_flag flg,
                           const yyjson_alc *alc_ptr,
                           yyjson_read_err *err) {
#define return_err(_code, _msg) do { \
    err->pos = 0; \
    err->msg = _msg; \
    err->code = YYJSON_READ_ERROR_##_code; \
    if (buf) alc.free(alc.ctx, buf); \
    return NULL; \
} while (false)

    yyjson_read_err dummy_err;
    yyjson_alc alc = alc_ptr ? *alc_ptr : YYJSON_DEFAULT_ALC;
    yyjson_doc *doc;

    long file_size = 0, file_pos;
    void *buf = NULL;
    usize buf_size = 0;

    /* validate input parameters */
    if (!err) err = &dummy_err;
    if (unlikely(!file)) return_err(INVALID_PARAMETER, "input file is NULL");

    /* get current position */
    file_pos = ftell(file);
    if (file_pos != -1) {
        /* get total file size, may fail */
        if (fseek(file, 0, SEEK_END) == 0) file_size = ftell(file);
        /* reset to original position, may fail */
        if (fseek(file, file_pos, SEEK_SET) != 0) file_size = 0;
        /* get file size from current postion to end */
        if (file_size > 0) file_size -= file_pos;
    }

    /* read file */
    if (file_size > 0) {
        /* read the entire file in one call */
        buf_size = (usize)file_size + YYJSON_PADDING_SIZE;
        buf = alc.malloc(alc.ctx, buf_size);
        if (buf == NULL) {
            return_err(MEMORY_ALLOCATION, MSG_MALLOC);
        }
        if (fread_safe(buf, (usize)file_size, file) != (usize)file_size) {
            return_err(FILE_READ, MSG_FREAD);
        }
    } else {
        /* failed to get file size, read it as a stream */
        usize chunk_min = (usize)64;
        usize chunk_max = (usize)512 * 1024 * 1024;
        usize chunk_now = chunk_min;
        usize read_size;
        void *tmp;

        buf_size = YYJSON_PADDING_SIZE;
        while (true) {
            if (buf_size + chunk_now < buf_size) { /* overflow */
                return_err(MEMORY_ALLOCATION, MSG_MALLOC);
            }
            buf_size += chunk_now;
            if (!buf) {
                buf = alc.malloc(alc.ctx, buf_size);
                if (!buf) return_err(MEMORY_ALLOCATION, MSG_MALLOC);
            } else {
                tmp = alc.realloc(alc.ctx, buf, buf_size - chunk_now, buf_size);
                if (!tmp) return_err(MEMORY_ALLOCATION, MSG_MALLOC);
                buf = tmp;
            }
            tmp = ((u8 *)buf) + buf_size - YYJSON_PADDING_SIZE - chunk_now;
            read_size = fread_safe(tmp, chunk_now, file);
            file_size += (long)read_size;
            if (read_size != chunk_now) break;

            chunk_now *= 2;
            if (chunk_now > chunk_max) chunk_now = chunk_max;
        }
    }

    /* read JSON */
    memset((u8 *)buf + file_size, 0, YYJSON_PADDING_SIZE);
    flg |= YYJSON_READ_INSITU;
    doc = yyjson_read_opts((char *)buf, (usize)file_size, flg, &alc, err);
    if (doc) {
        doc->str_pool = (char *)buf;
        return doc;
    } else {
        alc.free(alc.ctx, buf);
        return NULL;
    }

#undef return_err
}

const char *yyjson_read_number(const char *dat,
                               yyjson_val *val,
                               yyjson_read_flag flg,
                               const yyjson_alc *alc,
                               yyjson_read_err *err) {
#define return_err(_pos, _code, _msg) do { \
    err->pos = _pos > hdr ? (usize)(_pos - hdr) : 0; \
    err->msg = _msg; \
    err->code = YYJSON_READ_ERROR_##_code; \
    return NULL; \
} while (false)

    u8 *hdr = constcast(u8 *)dat, *cur = hdr;
    bool raw; /* read number as raw */
    u8 *raw_end; /* raw end for null-terminator */
    u8 **pre; /* previous raw end pointer */
    const char *msg;
    yyjson_read_err dummy_err;

#if !YYJSON_HAS_IEEE_754 || YYJSON_DISABLE_FAST_FP_CONV
    u8 buf[128];
    usize dat_len;
#endif

    if (!err) err = &dummy_err;
    if (unlikely(!dat)) {
        return_err(cur, INVALID_PARAMETER, "input data is NULL");
    }
    if (unlikely(!val)) {
        return_err(cur, INVALID_PARAMETER, "output value is NULL");
    }

#if !YYJSON_HAS_IEEE_754 || YYJSON_DISABLE_FAST_FP_CONV
    if (!alc) alc = &YYJSON_DEFAULT_ALC;
    dat_len = strlen(dat);
    if (dat_len < sizeof(buf)) {
        memcpy(buf, dat, dat_len + 1);
        hdr = buf;
        cur = hdr;
    } else {
        hdr = (u8 *)alc->malloc(alc->ctx, dat_len + 1);
        cur = hdr;
        if (unlikely(!hdr)) {
            return_err(cur, MEMORY_ALLOCATION, MSG_MALLOC);
        }
        memcpy(hdr, dat, dat_len + 1);
    }
    hdr[dat_len] = 0;
#endif

    raw = (flg & (YYJSON_READ_NUMBER_AS_RAW | YYJSON_READ_BIGNUM_AS_RAW)) != 0;
    raw_end = NULL;
    pre = raw ? &raw_end : NULL;

#if !YYJSON_HAS_IEEE_754 || YYJSON_DISABLE_FAST_FP_CONV
    if (!read_num(&cur, pre, flg, val, &msg)) {
        if (dat_len >= sizeof(buf)) alc->free(alc->ctx, hdr);
        return_err(cur, INVALID_NUMBER, msg);
    }
    if (dat_len >= sizeof(buf)) alc->free(alc->ctx, hdr);
    if (yyjson_is_raw(val)) val->uni.str = dat;
    return dat + (cur - hdr);
#else
    if (!read_num(&cur, pre, flg, val, &msg)) {
        return_err(cur, INVALID_NUMBER, msg);
    }
    return (const char *)cur;
#endif

#undef return_err
}

#endif /* YYJSON_DISABLE_READER */



#if !YYJSON_DISABLE_WRITER

/*==============================================================================
 * Integer Writer
 *
 * The maximum value of uint32_t is 4294967295 (10 digits),
 * these digits are named as 'aabbccddee' here.
 *
 * Although most compilers may convert the "division by constant value" into
 * "multiply and shift", manual conversion can still help some compilers
 * generate fewer and better instructions.
 *
 * Reference:
 * Division by Invariant Integers using Multiplication, 1994.
 * https://gmplib.org/~tege/divcnst-pldi94.pdf
 * Improved division by invariant integers, 2011.
 * https://gmplib.org/~tege/division-paper.pdf
 *============================================================================*/

/** Digit table from 00 to 99. */
yyjson_align(2)
static const char digit_table[200] = {
    '0', '0', '0', '1', '0', '2', '0', '3', '0', '4',
    '0', '5', '0', '6', '0', '7', '0', '8', '0', '9',
    '1', '0', '1', '1', '1', '2', '1', '3', '1', '4',
    '1', '5', '1', '6', '1', '7', '1', '8', '1', '9',
    '2', '0', '2', '1', '2', '2', '2', '3', '2', '4',
    '2', '5', '2', '6', '2', '7', '2', '8', '2', '9',
    '3', '0', '3', '1', '3', '2', '3', '3', '3', '4',
    '3', '5', '3', '6', '3', '7', '3', '8', '3', '9',
    '4', '0', '4', '1', '4', '2', '4', '3', '4', '4',
    '4', '5', '4', '6', '4', '7', '4', '8', '4', '9',
    '5', '0', '5', '1', '5', '2', '5', '3', '5', '4',
    '5', '5', '5', '6', '5', '7', '5', '8', '5', '9',
    '6', '0', '6', '1', '6', '2', '6', '3', '6', '4',
    '6', '5', '6', '6', '6', '7', '6', '8', '6', '9',
    '7', '0', '7', '1', '7', '2', '7', '3', '7', '4',
    '7', '5', '7', '6', '7', '7', '7', '8', '7', '9',
    '8', '0', '8', '1', '8', '2', '8', '3', '8', '4',
    '8', '5', '8', '6', '8', '7', '8', '8', '8', '9',
    '9', '0', '9', '1', '9', '2', '9', '3', '9', '4',
    '9', '5', '9', '6', '9', '7', '9', '8', '9', '9'
};

static_inline u8 *write_u32_len_8(u32 val, u8 *buf) {
    u32 aa, bb, cc, dd, aabb, ccdd;                 /* 8 digits: aabbccdd */
    aabb = (u32)(((u64)val * 109951163) >> 40);     /* (val / 10000) */
    ccdd = val - aabb * 10000;                      /* (val % 10000) */
    aa = (aabb * 5243) >> 19;                       /* (aabb / 100) */
    cc = (ccdd * 5243) >> 19;                       /* (ccdd / 100) */
    bb = aabb - aa * 100;                           /* (aabb % 100) */
    dd = ccdd - cc * 100;                           /* (ccdd % 100) */
    byte_copy_2(buf + 0, digit_table + aa * 2);
    byte_copy_2(buf + 2, digit_table + bb * 2);
    byte_copy_2(buf + 4, digit_table + cc * 2);
    byte_copy_2(buf + 6, digit_table + dd * 2);
    return buf + 8;
}

static_inline u8 *write_u32_len_4(u32 val, u8 *buf) {
    u32 aa, bb;                                     /* 4 digits: aabb */
    aa = (val * 5243) >> 19;                        /* (val / 100) */
    bb = val - aa * 100;                            /* (val % 100) */
    byte_copy_2(buf + 0, digit_table + aa * 2);
    byte_copy_2(buf + 2, digit_table + bb * 2);
    return buf + 4;
}

static_inline u8 *write_u32_len_1_to_8(u32 val, u8 *buf) {
    u32 aa, bb, cc, dd, aabb, bbcc, ccdd, lz;

    if (val < 100) {                                /* 1-2 digits: aa */
        lz = val < 10;                              /* leading zero: 0 or 1 */
        byte_copy_2(buf + 0, digit_table + val * 2 + lz);
        buf -= lz;
        return buf + 2;

    } else if (val < 10000) {                       /* 3-4 digits: aabb */
        aa = (val * 5243) >> 19;                    /* (val / 100) */
        bb = val - aa * 100;                        /* (val % 100) */
        lz = aa < 10;                               /* leading zero: 0 or 1 */
        byte_copy_2(buf + 0, digit_table + aa * 2 + lz);
        buf -= lz;
        byte_copy_2(buf + 2, digit_table + bb * 2);
        return buf + 4;

    } else if (val < 1000000) {                     /* 5-6 digits: aabbcc */
        aa = (u32)(((u64)val * 429497) >> 32);      /* (val / 10000) */
        bbcc = val - aa * 10000;                    /* (val % 10000) */
        bb = (bbcc * 5243) >> 19;                   /* (bbcc / 100) */
        cc = bbcc - bb * 100;                       /* (bbcc % 100) */
        lz = aa < 10;                               /* leading zero: 0 or 1 */
        byte_copy_2(buf + 0, digit_table + aa * 2 + lz);
        buf -= lz;
        byte_copy_2(buf + 2, digit_table + bb * 2);
        byte_copy_2(buf + 4, digit_table + cc * 2);
        return buf + 6;

    } else {                                        /* 7-8 digits: aabbccdd */
        aabb = (u32)(((u64)val * 109951163) >> 40); /* (val / 10000) */
        ccdd = val - aabb * 10000;                  /* (val % 10000) */
        aa = (aabb * 5243) >> 19;                   /* (aabb / 100) */
        cc = (ccdd * 5243) >> 19;                   /* (ccdd / 100) */
        bb = aabb - aa * 100;                       /* (aabb % 100) */
        dd = ccdd - cc * 100;                       /* (ccdd % 100) */
        lz = aa < 10;                               /* leading zero: 0 or 1 */
        byte_copy_2(buf + 0, digit_table + aa * 2 + lz);
        buf -= lz;
        byte_copy_2(buf + 2, digit_table + bb * 2);
        byte_copy_2(buf + 4, digit_table + cc * 2);
        byte_copy_2(buf + 6, digit_table + dd * 2);
        return buf + 8;
    }
}

static_inline u8 *write_u32_len_5_to_8(u32 val, u8 *buf) {
    u32 aa, bb, cc, dd, aabb, bbcc, ccdd, lz;

    if (val < 1000000) {                            /* 5-6 digits: aabbcc */
        aa = (u32)(((u64)val * 429497) >> 32);      /* (val / 10000) */
        bbcc = val - aa * 10000;                    /* (val % 10000) */
        bb = (bbcc * 5243) >> 19;                   /* (bbcc / 100) */
        cc = bbcc - bb * 100;                       /* (bbcc % 100) */
        lz = aa < 10;                               /* leading zero: 0 or 1 */
        byte_copy_2(buf + 0, digit_table + aa * 2 + lz);
        buf -= lz;
        byte_copy_2(buf + 2, digit_table + bb * 2);
        byte_copy_2(buf + 4, digit_table + cc * 2);
        return buf + 6;

    } else {                                        /* 7-8 digits: aabbccdd */
        aabb = (u32)(((u64)val * 109951163) >> 40); /* (val / 10000) */
        ccdd = val - aabb * 10000;                  /* (val % 10000) */
        aa = (aabb * 5243) >> 19;                   /* (aabb / 100) */
        cc = (ccdd * 5243) >> 19;                   /* (ccdd / 100) */
        bb = aabb - aa * 100;                       /* (aabb % 100) */
        dd = ccdd - cc * 100;                       /* (ccdd % 100) */
        lz = aa < 10;                               /* leading zero: 0 or 1 */
        byte_copy_2(buf + 0, digit_table + aa * 2 + lz);
        buf -= lz;
        byte_copy_2(buf + 2, digit_table + bb * 2);
        byte_copy_2(buf + 4, digit_table + cc * 2);
        byte_copy_2(buf + 6, digit_table + dd * 2);
        return buf + 8;
    }
}

static_inline u8 *write_u64(u64 val, u8 *buf) {
    u64 tmp, hgh;
    u32 mid, low;

    if (val < 100000000) {                          /* 1-8 digits */
        buf = write_u32_len_1_to_8((u32)val, buf);
        return buf;

    } else if (val < (u64)100000000 * 100000000) {  /* 9-16 digits */
        hgh = val / 100000000;                      /* (val / 100000000) */
        low = (u32)(val - hgh * 100000000);         /* (val % 100000000) */
        buf = write_u32_len_1_to_8((u32)hgh, buf);
        buf = write_u32_len_8(low, buf);
        return buf;

    } else {                                        /* 17-20 digits */
        tmp = val / 100000000;                      /* (val / 100000000) */
        low = (u32)(val - tmp * 100000000);         /* (val % 100000000) */
        hgh = (u32)(tmp / 10000);                   /* (tmp / 10000) */
        mid = (u32)(tmp - hgh * 10000);             /* (tmp % 10000) */
        buf = write_u32_len_5_to_8((u32)hgh, buf);
        buf = write_u32_len_4(mid, buf);
        buf = write_u32_len_8(low, buf);
        return buf;
    }
}



/*==============================================================================
 * Number Writer
 *============================================================================*/

#if YYJSON_HAS_IEEE_754 && !YYJSON_DISABLE_FAST_FP_CONV  /* FP_WRITER */

/** Trailing zero count table for number 0 to 99.
    (generate with misc/make_tables.c) */
static const u8 dec_trailing_zero_table[] = {
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

static_inline u8 *write_u32_len_1_to_9(u32 val, u8 *buf) {
    if (val >= 100000000) {
        u32 hi = val / 10000000;
        val = val - hi * 10000000;
        *buf++ = (u8)(hi + '0');
    }
    return write_u32_len_1_to_8((u32)val, buf);
}

static_inline u8 *write_u64_len_1_to_16(u64 val, u8 *buf) {
    u64 hgh;
    u32 low;
    if (val < 100000000) {                          /* 1-8 digits */
        buf = write_u32_len_1_to_8((u32)val, buf);
        return buf;
    } else {                                        /* 9-16 digits */
        hgh = val / 100000000;                      /* (val / 100000000) */
        low = (u32)(val - hgh * 100000000);         /* (val % 100000000) */
        buf = write_u32_len_1_to_8((u32)hgh, buf);
        buf = write_u32_len_8(low, buf);
        return buf;
    }
}

static_inline u8 *write_u64_len_1_to_17(u64 val, u8 *buf) {
    u64 hgh;
    u32 mid, low, one;
    if (val >= (u64)100000000 * 10000000) {         /* len: 16 to 17 */
        hgh = val / 100000000;                      /* (val / 100000000) */
        low = (u32)(val - hgh * 100000000);         /* (val % 100000000) */
        one = (u32)(hgh / 100000000);               /* (hgh / 100000000) */
        mid = (u32)(hgh - (u64)one * 100000000);    /* (hgh % 100000000) */
        *buf = (u8)((u8)one + (u8)'0');
        buf += one > 0;
        buf = write_u32_len_8(mid, buf);
        buf = write_u32_len_8(low, buf);
        return buf;
    } else if (val >= (u64)100000000){              /* len: 9 to 15 */
        hgh = val / 100000000;                      /* (val / 100000000) */
        low = (u32)(val - hgh * 100000000);         /* (val % 100000000) */
        buf = write_u32_len_1_to_8((u32)hgh, buf);
        buf = write_u32_len_8(low, buf);
        return buf;
    } else {                                        /* len: 1 to 8 */
        buf = write_u32_len_1_to_8((u32)val, buf);
        return buf;
    }
}

/**
 Write an unsigned integer with a length of 7 to 9 with trailing zero trimmed.
 These digits are named as "abbccddee" here.
 For example, input 123456000, output "123456".
 */
static_inline u8 *write_u32_len_7_to_9_trim(u32 val, u8 *buf) {
    bool lz;
    u32 tz, tz1, tz2;

    u32 abbcc = val / 10000;                        /* (abbccddee / 10000) */
    u32 ddee = val - abbcc * 10000;                 /* (abbccddee % 10000) */
    u32 abb = (u32)(((u64)abbcc * 167773) >> 24);   /* (abbcc / 100) */
    u32 a = (abb * 41) >> 12;                       /* (abb / 100) */
    u32 bb = abb - a * 100;                         /* (abb % 100) */
    u32 cc = abbcc - abb * 100;                     /* (abbcc % 100) */

    /* write abbcc */
    buf[0] = (u8)(a + '0');
    buf += a > 0;
    lz = bb < 10 && a == 0;
    byte_copy_2(buf + 0, digit_table + bb * 2 + lz);
    buf -= lz;
    byte_copy_2(buf + 2, digit_table + cc * 2);

    if (ddee) {
        u32 dd = (ddee * 5243) >> 19;               /* (ddee / 100) */
        u32 ee = ddee - dd * 100;                   /* (ddee % 100) */
        byte_copy_2(buf + 4, digit_table + dd * 2);
        byte_copy_2(buf + 6, digit_table + ee * 2);
        tz1 = dec_trailing_zero_table[dd];
        tz2 = dec_trailing_zero_table[ee];
        tz = ee ? tz2 : (tz1 + 2);
        buf += 8 - tz;
        return buf;
    } else {
        tz1 = dec_trailing_zero_table[bb];
        tz2 = dec_trailing_zero_table[cc];
        tz = cc ? tz2 : (tz1 + tz2);
        buf += 4 - tz;
        return buf;
    }
}

/**
 Write an unsigned integer with a length of 16 or 17 with trailing zero trimmed.
 These digits are named as "abbccddeeffgghhii" here.
 For example, input 1234567890123000, output "1234567890123".
 */
static_inline u8 *write_u64_len_16_to_17_trim(u64 val, u8 *buf) {
    u32 tz, tz1, tz2;

    u32 abbccddee = (u32)(val / 100000000);
    u32 ffgghhii = (u32)(val - (u64)abbccddee * 100000000);
    u32 abbcc = abbccddee / 10000;
    u32 ddee = abbccddee - abbcc * 10000;
    u32 abb = (u32)(((u64)abbcc * 167773) >> 24);   /* (abbcc / 100) */
    u32 a = (abb * 41) >> 12;                       /* (abb / 100) */
    u32 bb = abb - a * 100;                         /* (abb % 100) */
    u32 cc = abbcc - abb * 100;                     /* (abbcc % 100) */
    buf[0] = (u8)(a + '0');
    buf += a > 0;
    byte_copy_2(buf + 0, digit_table + bb * 2);
    byte_copy_2(buf + 2, digit_table + cc * 2);

    if (ffgghhii) {
        u32 dd = (ddee * 5243) >> 19;               /* (ddee / 100) */
        u32 ee = ddee - dd * 100;                   /* (ddee % 100) */
        u32 ffgg = (u32)(((u64)ffgghhii * 109951163) >> 40); /* (val / 10000) */
        u32 hhii = ffgghhii - ffgg * 10000;         /* (val % 10000) */
        u32 ff = (ffgg * 5243) >> 19;               /* (aabb / 100) */
        u32 gg = ffgg - ff * 100;                   /* (aabb % 100) */
        byte_copy_2(buf + 4, digit_table + dd * 2);
        byte_copy_2(buf + 6, digit_table + ee * 2);
        byte_copy_2(buf + 8, digit_table + ff * 2);
        byte_copy_2(buf + 10, digit_table + gg * 2);
        if (hhii) {
            u32 hh = (hhii * 5243) >> 19;           /* (ccdd / 100) */
            u32 ii = hhii - hh * 100;               /* (ccdd % 100) */
            byte_copy_2(buf + 12, digit_table + hh * 2);
            byte_copy_2(buf + 14, digit_table + ii * 2);
            tz1 = dec_trailing_zero_table[hh];
            tz2 = dec_trailing_zero_table[ii];
            tz = ii ? tz2 : (tz1 + 2);
            return buf + 16 - tz;
        } else {
            tz1 = dec_trailing_zero_table[ff];
            tz2 = dec_trailing_zero_table[gg];
            tz = gg ? tz2 : (tz1 + 2);
            return buf + 12 - tz;
        }
    } else {
        if (ddee) {
            u32 dd = (ddee * 5243) >> 19;           /* (ddee / 100) */
            u32 ee = ddee - dd * 100;               /* (ddee % 100) */
            byte_copy_2(buf + 4, digit_table + dd * 2);
            byte_copy_2(buf + 6, digit_table + ee * 2);
            tz1 = dec_trailing_zero_table[dd];
            tz2 = dec_trailing_zero_table[ee];
            tz = ee ? tz2 : (tz1 + 2);
            return buf + 8 - tz;
        } else {
            tz1 = dec_trailing_zero_table[bb];
            tz2 = dec_trailing_zero_table[cc];
            tz = cc ? tz2 : (tz1 + tz2);
            return buf + 4 - tz;
        }
    }
}

/** Write exponent part in range `e-45` to `e38`. */
static_inline u8 *write_f32_exp(i32 exp, u8 *buf) {
    bool lz;
    byte_copy_2(buf, "e-");
    buf += 2 - (exp >= 0);
    exp = exp < 0 ? -exp : exp;
    lz = exp < 10;
    byte_copy_2(buf + 0, digit_table + (u32)exp * 2 + lz);
    return buf + 2 - lz;
}

/** Write exponent part in range `e-324` to `e308`. */
static_inline u8 *write_f64_exp(i32 exp, u8 *buf) {
    byte_copy_2(buf, "e-");
    buf += 2 - (exp >= 0);
    exp = exp < 0 ? -exp : exp;
    if (exp < 100) {
        bool lz = exp < 10;
        byte_copy_2(buf + 0, digit_table + (u32)exp * 2 + lz);
        return buf + 2 - lz;
    } else {
        u32 hi = ((u32)exp * 656) >> 16;    /* exp / 100 */
        u32 lo = (u32)exp - hi * 100;       /* exp % 100 */
        buf[0] = (u8)((u8)hi + (u8)'0');
        byte_copy_2(buf + 1, digit_table + lo * 2);
        return buf + 3;
    }
}

/** Magic number for fast `divide by power of 10`. */
typedef struct {
    u64 p10, mul;
    u32 shr1, shr2;
} div_pow10_magic;

/** Generated with llvm, see https://github.com/llvm/llvm-project/
    blob/main/llvm/lib/Support/DivisionByConstantInfo.cpp */
static const div_pow10_magic div_pow10_table[] = {
    { U64(0x00000000, 0x00000001), U64(0x00000000, 0x00000000), 0,  0  },
    { U64(0x00000000, 0x0000000A), U64(0xCCCCCCCC, 0xCCCCCCCD), 0,  3  },
    { U64(0x00000000, 0x00000064), U64(0x28F5C28F, 0x5C28F5C3), 2,  2  },
    { U64(0x00000000, 0x000003E8), U64(0x20C49BA5, 0xE353F7CF), 3,  4  },
    { U64(0x00000000, 0x00002710), U64(0x346DC5D6, 0x3886594B), 0,  11 },
    { U64(0x00000000, 0x000186A0), U64(0x0A7C5AC4, 0x71B47843), 5,  7  },
    { U64(0x00000000, 0x000F4240), U64(0x431BDE82, 0xD7B634DB), 0,  18 },
    { U64(0x00000000, 0x00989680), U64(0xD6BF94D5, 0xE57A42BD), 0,  23 },
    { U64(0x00000000, 0x05F5E100), U64(0xABCC7711, 0x8461CEFD), 0,  26 },
    { U64(0x00000000, 0x3B9ACA00), U64(0x0044B82F, 0xA09B5A53), 9,  11 },
    { U64(0x00000002, 0x540BE400), U64(0xDBE6FECE, 0xBDEDD5BF), 0,  33 },
    { U64(0x00000017, 0x4876E800), U64(0xAFEBFF0B, 0xCB24AAFF), 0,  36 },
    { U64(0x000000E8, 0xD4A51000), U64(0x232F3302, 0x5BD42233), 0,  37 },
    { U64(0x00000918, 0x4E72A000), U64(0x384B84D0, 0x92ED0385), 0,  41 },
    { U64(0x00005AF3, 0x107A4000), U64(0x0B424DC3, 0x5095CD81), 0,  42 },
    { U64(0x00038D7E, 0xA4C68000), U64(0x00024075, 0xF3DCEAC3), 15, 20 },
    { U64(0x002386F2, 0x6FC10000), U64(0x39A5652F, 0xB1137857), 0,  51 },
    { U64(0x01634578, 0x5D8A0000), U64(0x00005C3B, 0xD5191B53), 17, 22 },
    { U64(0x0DE0B6B3, 0xA7640000), U64(0x000049C9, 0x7747490F), 18, 24 },
    { U64(0x8AC72304, 0x89E80000), U64(0x760F253E, 0xDB4AB0d3), 0,  62 },
};

/** Divide a number by power of 10. */
static_inline void div_pow10(u64 num, u32 exp, u64 *div, u64 *mod, u64 *p10) {
    u64 hi, lo;
    div_pow10_magic m = div_pow10_table[exp];
    u128_mul(num >> m.shr1, m.mul, &hi, &lo);
    *div = hi >> m.shr2;
    *mod = num - (*div * m.p10);
    *p10 = m.p10;
}

/** Multiplies 64-bit integer and returns highest 64-bit rounded value. */
static_inline u32 u64_round_to_odd(u64 u, u32 cp) {
    u64 hi, lo;
    u32 y_hi, y_lo;
    u128_mul(cp, u, &hi, &lo);
    y_hi = (u32)hi;
    y_lo = (u32)(lo >> 32);
    return y_hi | (y_lo > 1);
}

/** Multiplies 128-bit integer and returns highest 64-bit rounded value. */
static_inline u64 u128_round_to_odd(u64 hi, u64 lo, u64 cp) {
    u64 x_hi, x_lo, y_hi, y_lo;
    u128_mul(cp, lo, &x_hi, &x_lo);
    u128_mul_add(cp, hi, x_hi, &y_hi, &y_lo);
    return y_hi | (y_lo > 1);
}

/** Convert f32 from binary to decimal (shortest but may have trailing zeros).
    The input should not be 0, inf or nan. */
static_inline void f32_bin_to_dec(u32 sig_raw, u32 exp_raw,
                                  u32 sig_bin, i32 exp_bin,
                                  u32 *sig_dec, i32 *exp_dec) {

    bool is_even, irregular, round_up, trim;
    bool u0_inside, u1_inside, w0_inside, w1_inside;
    u64 p10_hi, p10_lo, hi, lo;
    u32 s, sp, cb, cbl, cbr, vb, vbl, vbr, upper, lower, mid;
    i32 k, h;

    /* Fast path, see f64_bin_to_dec(). */
    while (likely(sig_raw)) {
        u32 mod, dec, add_1, add_10, s_hi, s_lo;
        u32 c, half_ulp, t0, t1;

        /* k = floor(exp_bin * log10(2)); */
        /* h = exp_bin + floor(log2(10) * -k); (h = 0/1/2/3) */
        k = (i32)(exp_bin * 315653) >> 20;
        h = exp_bin + ((-k * 217707) >> 16);
        pow10_table_get_sig(-k, &p10_hi, &p10_lo);

        /* sig_bin << (1/2/3/4) */
        cb = sig_bin << (h + 1);
        u128_mul(cb, p10_hi, &hi, &lo);
        s_hi = (u32)(hi);
        s_lo = (u32)(lo >> 32);
        mod = s_hi % 10;
        dec = s_hi - mod;

        /* right shift 4 to fit in u32 */
        c = (mod << (32 - 4)) | (s_lo >> 4);
        half_ulp = (u32)(p10_hi >> (32 + 4 - h));

        /* check w1, u0, w0 range */
        w1_inside = (s_lo >= ((u32)1 << 31));
        if (unlikely(s_lo == ((u32)1 << 31))) break;
        u0_inside = (half_ulp >= c);
        if (unlikely(half_ulp == c)) break;
        t0 = (u32)10 << (32 - 4);
        t1 = c + half_ulp;
        w0_inside = (t1 >= t0);
        if (unlikely(t0 - t1 <= (u32)1)) break;

        trim = (u0_inside | w0_inside);
        add_10 = (w0_inside ? 10 : 0);
        add_1 = mod + w1_inside;
        *sig_dec = dec + (trim ? add_10 : add_1);
        *exp_dec = k;
        return;
    }

    /* Schubfach algorithm, see f64_bin_to_dec(). */
    irregular = (sig_raw == 0 && exp_raw > 1);
    is_even = !(sig_bin & 1);
    cbl = 4 * sig_bin - 2 + irregular;
    cb  = 4 * sig_bin;
    cbr = 4 * sig_bin + 2;

    /* k = floor(exp_bin * log10(2) + (irregular ? log10(3.0 / 4.0) : 0)); */
    /* h = exp_bin + floor(log2(10) * -k) + 1; (h = 1/2/3/4) */
    k = (i32)(exp_bin * 315653 - (irregular ? 131237 : 0)) >> 20;
    h = exp_bin + ((-k * 217707) >> 16) + 1;
    pow10_table_get_sig(-k, &p10_hi, &p10_lo);
    p10_hi += 1;

    vbl = u64_round_to_odd(p10_hi, cbl << h);
    vb  = u64_round_to_odd(p10_hi, cb  << h);
    vbr = u64_round_to_odd(p10_hi, cbr << h);
    lower = vbl + !is_even;
    upper = vbr - !is_even;

    s = vb / 4;
    if (s >= 10) {
        sp = s / 10;
        u0_inside = (lower <= 40 * sp);
        w0_inside = (upper >= 40 * sp + 40);
        if (u0_inside != w0_inside) {
            *sig_dec = sp * 10 + (w0_inside ? 10 : 0);
            *exp_dec = k;
            return;
        }
    }
    u1_inside = (lower <= 4 * s);
    w1_inside = (upper >= 4 * s + 4);
    mid = 4 * s + 2;
    round_up = (vb > mid) || (vb == mid && (s & 1) != 0);
    *sig_dec = s + ((u1_inside != w1_inside) ? w1_inside : round_up);
    *exp_dec = k;
}

/** Convert f64 from binary to decimal (shortest but may have trailing zeros).
    The input should not be 0, inf or nan. */
static_inline void f64_bin_to_dec(u64 sig_raw, u32 exp_raw,
                                  u64 sig_bin, i32 exp_bin,
                                  u64 *sig_dec, i32 *exp_dec) {

    bool is_even, irregular, round_up, trim;
    bool u0_inside, u1_inside, w0_inside, w1_inside;
    u64 s, sp, cb, cbl, cbr, vb, vbl, vbr, p10_hi, p10_lo, upper, lower, mid;
    i32 k, h;

    /*
     Fast path:
     For regular spacing significand 'c', there are 4 candidates:

             u0             u1 c  w1                            w0
     ----|----|----|----|----|-*--|----|----|----|----|----|----|----|----
         9    0    1    2    3    4    5    6    7    8    9    0    1
           |___________________|___________________|
                             1ulp

     The `1ulp` is in the range [1.0, 10.0).
     If (c - 0.5ulp < u0), trim the last digit and round down.
     If (c + 0.5ulp > w0), trim the last digit and round up.
     If (c - 0.5ulp < u1), round down.
     If (c + 0.5ulp > w1), round up.
     */
    while (likely(sig_raw)) {
        u64 mod, dec, add_1, add_10, s_hi, s_lo;
        u64 c, half_ulp, t0, t1;

        /* k = floor(exp_bin * log10(2)); */
        /* h = exp_bin + floor(log2(10) * -k); (h = 0/1/2/3) */
        k = (i32)(exp_bin * 315653) >> 20;
        h = exp_bin + ((-k * 217707) >> 16);
        pow10_table_get_sig(-k, &p10_hi, &p10_lo);

        /* sig_bin << (1/2/3/4) */
        cb = sig_bin << (h + 1);
        u128_mul(cb, p10_lo, &s_hi, &s_lo);
        u128_mul_add(cb, p10_hi, s_hi, &s_hi, &s_lo);
        mod = s_hi % 10;
        dec = s_hi - mod;

        /* right shift 4 to fit in u64 */
        c = (mod << (64 - 4)) | (s_lo >> 4);
        half_ulp = p10_hi >> (4 - h);

        /* check w1, u0, w0 range */
        w1_inside = (s_lo >= ((u64)1 << 63));
        if (unlikely(s_lo == ((u64)1 << 63))) break;
        u0_inside = (half_ulp >= c);
        if (unlikely(half_ulp == c)) break;
        t0 = ((u64)10 << (64 - 4));
        t1 = c + half_ulp;
        w0_inside = (t1 >= t0);
        if (unlikely(t0 - t1 <= (u64)1)) break;

        trim = (u0_inside | w0_inside);
        add_10 = (w0_inside ? 10 : 0);
        add_1 = mod + w1_inside;
        *sig_dec = dec + (trim ? add_10 : add_1);
        *exp_dec = k;
        return;
    }

    /*
     Schubfach algorithm:
     Raffaello Giulietti, The Schubfach way to render doubles, 2022.
     https://drive.google.com/file/d/1gp5xv4CAa78SVgCeWfGqqI4FfYYYuNFb (Paper)
     https://github.com/openjdk/jdk/pull/3402 (Java implementation)
     https://github.com/abolz/Drachennest (C++ implementation)
     */
    irregular = (sig_raw == 0 && exp_raw > 1);
    is_even = !(sig_bin & 1);
    cbl = 4 * sig_bin - 2 + irregular;
    cb  = 4 * sig_bin;
    cbr = 4 * sig_bin + 2;

    /* k = floor(exp_bin * log10(2) + (irregular ? log10(3.0 / 4.0) : 0)); */
    /* h = exp_bin + floor(log2(10) * -k) + 1; (h = 1/2/3/4) */
    k = (i32)(exp_bin * 315653 - (irregular ? 131237 : 0)) >> 20;
    h = exp_bin + ((-k * 217707) >> 16) + 1;
    pow10_table_get_sig(-k, &p10_hi, &p10_lo);
    p10_lo += 1;

    vbl = u128_round_to_odd(p10_hi, p10_lo, cbl << h);
    vb  = u128_round_to_odd(p10_hi, p10_lo, cb  << h);
    vbr = u128_round_to_odd(p10_hi, p10_lo, cbr << h);
    lower = vbl + !is_even;
    upper = vbr - !is_even;

    s = vb / 4;
    if (s >= 10) {
        sp = s / 10;
        u0_inside = (lower <= 40 * sp);
        w0_inside = (upper >= 40 * sp + 40);
        if (u0_inside != w0_inside) {
            *sig_dec = sp * 10 + (w0_inside ? 10 : 0);
            *exp_dec = k;
            return;
        }
    }
    u1_inside = (lower <= 4 * s);
    w1_inside = (upper >= 4 * s + 4);
    mid = 4 * s + 2;
    round_up = (vb > mid) || (vb == mid && (s & 1) != 0);
    *sig_dec = s + ((u1_inside != w1_inside) ? w1_inside : round_up);
    *exp_dec = k;
}

/** Convert f64 from binary to decimal (fast but not the shortest).
    The input should not be 0, inf, nan. */
static_inline void f64_bin_to_dec_fast(u64 sig_raw, u32 exp_raw,
                                       u64 sig_bin, i32 exp_bin,
                                       u64 *sig_dec, i32 *exp_dec,
                                       bool *round_up) {
    u64 cb, p10_hi, p10_lo, s_hi, s_lo;
    i32 k, h;
    bool irregular, u;

    irregular = (sig_raw == 0 && exp_raw > 1);

    /* k = floor(exp_bin * log10(2) + (irregular ? log10(3.0 / 4.0) : 0)); */
    /* h = exp_bin + floor(log2(10) * -k) + 1; (h = 1/2/3/4) */
    k = (i32)(exp_bin * 315653 - (irregular ? 131237 : 0)) >> 20;
    h = exp_bin + ((-k * 217707) >> 16);
    pow10_table_get_sig(-k, &p10_hi, &p10_lo);

    /* sig_bin << (1/2/3/4) */
    cb = sig_bin << (h + 1);
    u128_mul(cb, p10_lo, &s_hi, &s_lo);
    u128_mul_add(cb, p10_hi, s_hi, &s_hi, &s_lo);

    /* round up */
    u = s_lo >= (irregular ? U64(0x55555555, 0x55555555) : ((u64)1 << 63));

    *sig_dec = s_hi + u;
    *exp_dec = k;
    *round_up = u;
    return;
}

/** Write inf/nan if allowed. */
static_inline u8 *write_inf_or_nan(u8 *buf, yyjson_write_flag flg,
                                   u64 sig_raw, bool sign) {
    if (has_write_flag(INF_AND_NAN_AS_NULL)) {
        byte_copy_4(buf, "null");
        return buf + 4;
    }
    if (has_write_flag(ALLOW_INF_AND_NAN)) {
        if (sig_raw == 0) {
            buf[0] = '-';
            buf += sign;
            byte_copy_8(buf, "Infinity");
            return buf + 8;
        } else {
            byte_copy_4(buf, "NaN");
            return buf + 3;
        }
    }
    return NULL;
}

/**
 Write a float number (requires 40 bytes buffer).
 We follow the ECMAScript specification for printing floating-point numbers,
 similar to `Number.prototype.toString()`, but with the following changes:
 1. Keep the negative sign of `-0.0` to preserve input information.
 2. Keep decimal point to indicate the number is floating point.
 3. Remove positive sign in the exponent part.
 */
static_noinline u8 *write_f32_raw(u8 *buf, u64 raw_f64,
                                  yyjson_write_flag flg) {
    u32 sig_bin, sig_dec, sig_raw;
    i32 exp_bin, exp_dec, sig_len, dot_ofs;
    u32 exp_raw, raw;
    u8 *end;
    bool sign;

    /* cast double to float */
    raw = f32_to_raw(f64_to_f32(f64_from_raw(raw_f64)));

    /* decode raw bytes from IEEE-754 double format. */
    sign = (bool)(raw >> (F32_BITS - 1));
    sig_raw = raw & F32_SIG_MASK;
    exp_raw = (raw & F32_EXP_MASK) >> F32_SIG_BITS;

    /* return inf or nan */
    if (unlikely(exp_raw == ((u32)1 << F32_EXP_BITS) - 1)) {
        return write_inf_or_nan(buf, flg, sig_raw, sign);
    }

    /* add sign for all finite number */
    buf[0] = '-';
    buf += sign;

    /* return zero */
    if ((raw << 1) == 0) {
        byte_copy_4(buf, "0.0");
        return buf + 3;
    }

    if (likely(exp_raw != 0)) {
        /* normal number */
        sig_bin = sig_raw | ((u32)1 << F32_SIG_BITS);
        exp_bin = (i32)exp_raw - F32_EXP_BIAS - F32_SIG_BITS;

        /* fast path for small integer number without fraction */
        if ((-F32_SIG_BITS <= exp_bin && exp_bin <= 0) &&
            (u64_tz_bits(sig_bin) >= (u32)-exp_bin)) {
            sig_dec = sig_bin >> -exp_bin; /* range: [1, 0xFFFFFF] */
            buf = write_u32_len_1_to_8(sig_dec, buf);
            byte_copy_2(buf, ".0");
            return buf + 2;
        }

        /* binary to decimal */
        f32_bin_to_dec(sig_raw, exp_raw, sig_bin, exp_bin, &sig_dec, &exp_dec);

        /* the sig length is 7 or 9 */
        sig_len = 7 + (sig_dec >= (u32)10000000) + (sig_dec >= (u32)100000000);

        /* the decimal point offset relative to the first digit */
        dot_ofs = sig_len + exp_dec;

        if (-6 < dot_ofs && dot_ofs <= 21) {
            i32 num_sep_pos, dot_set_pos, pre_ofs;
            u8 *num_hdr, *num_end, *num_sep, *dot_end;
            bool no_pre_zero;

            /* fill zeros */
            memset(buf, '0', 32);

            /* not prefixed with zero, e.g. 1.234, 1234.0 */
            no_pre_zero = (dot_ofs > 0);

            /* write the number as digits */
            pre_ofs = no_pre_zero ? 0 : (2 - dot_ofs);
            num_hdr = buf + pre_ofs;
            num_end = write_u32_len_7_to_9_trim(sig_dec, num_hdr);

            /* seperate these digits to leave a space for dot */
            num_sep_pos = no_pre_zero ? dot_ofs : 0;
            num_sep = num_hdr + num_sep_pos;
            byte_move_8(num_sep + no_pre_zero, num_sep);
            num_end += no_pre_zero;

            /* write the dot */
            dot_set_pos = yyjson_max(dot_ofs, 1);
            buf[dot_set_pos] = '.';

            /* return the ending */
            dot_end = buf + dot_ofs + 2;
            return yyjson_max(dot_end, num_end);

        } else {
            /* write with scientific notation, e.g. 1.234e56 */
            end = write_u32_len_7_to_9_trim(sig_dec, buf + 1);
            end -= (end == buf + 2); /* remove '.0', e.g. 2.0e34 -> 2e34 */
            exp_dec += sig_len - 1;
            buf[0] = buf[1];
            buf[1] = '.';
            return write_f32_exp(exp_dec, end);
        }

    } else {
        /* subnormal number */
        sig_bin = sig_raw;
        exp_bin = 1 - F32_EXP_BIAS - F32_SIG_BITS;

        /* binary to decimal */
        f32_bin_to_dec(sig_raw, exp_raw, sig_bin, exp_bin, &sig_dec, &exp_dec);

        /* write significand part */
        end = write_u32_len_1_to_9(sig_dec, buf + 1);
        buf[0] = buf[1];
        buf[1] = '.';
        exp_dec += (i32)(end - buf) - 2;

        /* trim trailing zeros */
        end -= *(end - 1) == '0'; /* branchless for last zero */
        end -= *(end - 1) == '0'; /* branchless for second last zero */
        while (*(end - 1) == '0') end--; /* for unlikely more zeros */
        end -= *(end - 1) == '.'; /* remove dot, e.g. 2.e-321 -> 2e-321 */

        /* write exponent part */
        return write_f32_exp(exp_dec, end);
    }
}

/**
 Write a double number (requires 40 bytes buffer).
 We follow the ECMAScript specification for printing floating-point numbers,
 similar to `Number.prototype.toString()`, but with the following changes:
 1. Keep the negative sign of `-0.0` to preserve input information.
 2. Keep decimal point to indicate the number is floating point.
 3. Remove positive sign in the exponent part.
 */
static_noinline u8 *write_f64_raw(u8 *buf, u64 raw, yyjson_write_flag flg) {
    u64 sig_bin, sig_dec, sig_raw;
    i32 exp_bin, exp_dec, sig_len, dot_ofs;
    u32 exp_raw;
    u8 *end;
    bool sign;

    /* decode raw bytes from IEEE-754 double format. */
    sign = (bool)(raw >> (F64_BITS - 1));
    sig_raw = raw & F64_SIG_MASK;
    exp_raw = (u32)((raw & F64_EXP_MASK) >> F64_SIG_BITS);

    /* return inf or nan */
    if (unlikely(exp_raw == ((u32)1 << F64_EXP_BITS) - 1)) {
        return write_inf_or_nan(buf, flg, sig_raw, sign);
    }

    /* add sign for all finite number */
    buf[0] = '-';
    buf += sign;

    /* return zero */
    if ((raw << 1) == 0) {
        byte_copy_4(buf, "0.0");
        return buf + 3;
    }

    if (likely(exp_raw != 0)) {
        /* normal number */
        sig_bin = sig_raw | ((u64)1 << F64_SIG_BITS);
        exp_bin = (i32)exp_raw - F64_EXP_BIAS - F64_SIG_BITS;

        /* fast path for small integer number without fraction */
        if ((-F64_SIG_BITS <= exp_bin && exp_bin <= 0) &&
            (u64_tz_bits(sig_bin) >= (u32)-exp_bin)) {
            sig_dec = sig_bin >> -exp_bin; /* range: [1, 0x1FFFFFFFFFFFFF] */
            buf = write_u64_len_1_to_16(sig_dec, buf);
            byte_copy_2(buf, ".0");
            return buf + 2;
        }

        /* binary to decimal */
        f64_bin_to_dec(sig_raw, exp_raw, sig_bin, exp_bin, &sig_dec, &exp_dec);

        /* the sig length is 16 or 17 */
        sig_len = 16 + (sig_dec >= (u64)100000000 * 100000000);

        /* the decimal point offset relative to the first digit */
        dot_ofs = sig_len + exp_dec;

        if (-6 < dot_ofs && dot_ofs <= 21) {
            i32 num_sep_pos, dot_set_pos, pre_ofs;
            u8 *num_hdr, *num_end, *num_sep, *dot_end;
            bool no_pre_zero;

            /* fill zeros */
            memset(buf, '0', 32);

            /* not prefixed with zero, e.g. 1.234, 1234.0 */
            no_pre_zero = (dot_ofs > 0);

            /* write the number as digits */
            pre_ofs = no_pre_zero ? 0 : (2 - dot_ofs);
            num_hdr = buf + pre_ofs;
            num_end = write_u64_len_16_to_17_trim(sig_dec, num_hdr);

            /* seperate these digits to leave a space for dot */
            num_sep_pos = no_pre_zero ? dot_ofs : 0;
            num_sep = num_hdr + num_sep_pos;
            byte_move_16(num_sep + no_pre_zero, num_sep);
            num_end += no_pre_zero;

            /* write the dot */
            dot_set_pos = yyjson_max(dot_ofs, 1);
            buf[dot_set_pos] = '.';

            /* return the ending */
            dot_end = buf + dot_ofs + 2;
            return yyjson_max(dot_end, num_end);

        } else {
            /* write with scientific notation, e.g. 1.234e56 */
            end = write_u64_len_16_to_17_trim(sig_dec, buf + 1);
            end -= (end == buf + 2); /* remove '.0', e.g. 2.0e34 -> 2e34 */
            exp_dec += sig_len - 1;
            buf[0] = buf[1];
            buf[1] = '.';
            return write_f64_exp(exp_dec, end);
        }

    } else {
        /* subnormal number */
        sig_bin = sig_raw;
        exp_bin = 1 - F64_EXP_BIAS - F64_SIG_BITS;

        /* binary to decimal */
        f64_bin_to_dec(sig_raw, exp_raw, sig_bin, exp_bin, &sig_dec, &exp_dec);

        /* write significand part */
        end = write_u64_len_1_to_17(sig_dec, buf + 1);
        buf[0] = buf[1];
        buf[1] = '.';
        exp_dec += (i32)(end - buf) - 2;

        /* trim trailing zeros */
        end -= *(end - 1) == '0'; /* branchless for last zero */
        end -= *(end - 1) == '0'; /* branchless for second last zero */
        while (*(end - 1) == '0') end--; /* for unlikely more zeros */
        end -= *(end - 1) == '.'; /* remove dot, e.g. 2.e-321 -> 2e-321 */

        /* write exponent part */
        return write_f64_exp(exp_dec, end);
    }
}

/**
 Write a double number using fixed-point notation (requires 40 bytes buffer).

 We follow the ECMAScript specification for printing floating-point numbers,
 similar to `Number.prototype.toFixed(prec)`, but with the following changes:
 1. Keep the negative sign of `-0.0` to preserve input information.
 2. Keep decimal point to indicate the number is floating point.
 3. Remove positive sign in the exponent part.
 4. Remove trailing zeros and reduce unnecessary precision.
 */
static_noinline u8 *write_f64_raw_fixed(u8 *buf, u64 raw, yyjson_write_flag flg,
                                        u32 prec) {
    u64 sig_bin, sig_dec, sig_raw;
    i32 exp_bin, exp_dec, sig_len, dot_ofs;
    u32 exp_raw;
    u8 *end;
    bool sign;

    /* decode raw bytes from IEEE-754 double format. */
    sign = (bool)(raw >> (F64_BITS - 1));
    sig_raw = raw & F64_SIG_MASK;
    exp_raw = (u32)((raw & F64_EXP_MASK) >> F64_SIG_BITS);

    /* return inf or nan */
    if (unlikely(exp_raw == ((u32)1 << F64_EXP_BITS) - 1)) {
        return write_inf_or_nan(buf, flg, sig_raw, sign);
    }

    /* add sign for all finite number */
    buf[0] = '-';
    buf += sign;

    /* return zero */
    if ((raw << 1) == 0) {
        byte_copy_4(buf, "0.0");
        return buf + 3;
    }

    if (likely(exp_raw != 0)) {
        /* normal number */
        sig_bin = sig_raw | ((u64)1 << F64_SIG_BITS);
        exp_bin = (i32)exp_raw - F64_EXP_BIAS - F64_SIG_BITS;

        /* fast path for small integer number without fraction */
        if ((-F64_SIG_BITS <= exp_bin && exp_bin <= 0) &&
            (u64_tz_bits(sig_bin) >= (u32)-exp_bin)) {
            sig_dec = sig_bin >> -exp_bin; /* range: [1, 0x1FFFFFFFFFFFFF] */
            buf = write_u64_len_1_to_16(sig_dec, buf);
            byte_copy_2(buf, ".0");
            return buf + 2;
        }

        /* only `fabs(num) < 1e21` are processed here. */
        if ((raw << 1) < (U64(0x444B1AE4, 0xD6E2EF50) << 1)) {
            i32 num_sep_pos, dot_set_pos, pre_ofs;
            u8 *num_hdr, *num_end, *num_sep;
            bool round_up, no_pre_zero;

            /* binary to decimal */
            f64_bin_to_dec_fast(sig_raw, exp_raw, sig_bin, exp_bin,
                                &sig_dec, &exp_dec, &round_up);

            /* the sig length is 16 or 17 */
            sig_len = 16 + (sig_dec >= (u64)100000000 * 100000000);

            /* limit the length of digits after the decimal point */
            if (exp_dec < -1) {
                i32 sig_len_cut = -exp_dec - (i32)prec;
                if (sig_len_cut > sig_len) {
                    byte_copy_4(buf, "0.0");
                    return buf + 3;
                }
                if (sig_len_cut > 0) {
                    u64 div, mod, p10;

                    /* remove round up */
                    sig_dec -= round_up;
                    sig_len = 16 + (sig_dec >= (u64)100000000 * 100000000);

                    /* cut off some digits */
                    div_pow10(sig_dec, (u32)sig_len_cut, &div, &mod, &p10);

                    /* add round up */
                    sig_dec = div + (mod >= p10 / 2);

                    /* update exp and sig length */
                    exp_dec += sig_len_cut;
                    sig_len -= sig_len_cut;
                    sig_len += (sig_len >= 0) &&
                               (sig_dec >= div_pow10_table[sig_len].p10);
                }
                if (sig_len <= 0) {
                    byte_copy_4(buf, "0.0");
                    return buf + 3;
                }
            }

            /* fill zeros */
            memset(buf, '0', 32);

            /* the decimal point offset relative to the first digit */
            dot_ofs = sig_len + exp_dec;

            /* not prefixed with zero, e.g. 1.234, 1234.0 */
            no_pre_zero = (dot_ofs > 0);

            /* write the number as digits */
            pre_ofs = no_pre_zero ? 0 : (1 - dot_ofs);
            num_hdr = buf + pre_ofs;
            num_end = write_u64_len_1_to_17(sig_dec, num_hdr);

            /* seperate these digits to leave a space for dot */
            num_sep_pos = no_pre_zero ? dot_ofs : -dot_ofs;
            num_sep = buf + num_sep_pos;
            byte_move_16(num_sep + 1, num_sep);
            num_end += (exp_dec < 0);

            /* write the dot */
            dot_set_pos = yyjson_max(dot_ofs, 1);
            buf[dot_set_pos] = '.';

            /* remove trailing zeros */
            buf += dot_set_pos + 2;
            buf = yyjson_max(buf, num_end);
            buf -= *(buf - 1) == '0'; /* branchless for last zero */
            buf -= *(buf - 1) == '0'; /* branchless for second last zero */
            while (*(buf - 1) == '0') buf--; /* for unlikely more zeros */
            buf += *(buf - 1) == '.'; /* keep a zero after dot */
            return buf;

        } else {
            /* binary to decimal */
            f64_bin_to_dec(sig_raw, exp_raw, sig_bin, exp_bin,
                           &sig_dec, &exp_dec);

            /* the sig length is 16 or 17 */
            sig_len = 16 + (sig_dec >= (u64)100000000 * 100000000);

            /* write with scientific notation, e.g. 1.234e56 */
            end = write_u64_len_16_to_17_trim(sig_dec, buf + 1);
            end -= (end == buf + 2); /* remove '.0', e.g. 2.0e34 -> 2e34 */
            exp_dec += sig_len - 1;
            buf[0] = buf[1];
            buf[1] = '.';
            return write_f64_exp(exp_dec, end);
        }
    } else {
        /* subnormal number */
        byte_copy_4(buf, "0.0");
        return buf + 3;
    }
}

#else /* FP_WRITER */

#if YYJSON_MSC_VER >= 1400
#define snprintf_num(buf, len, fmt, dig, val) \
    sprintf_s((char *)buf, len, fmt, dig, val)
#elif defined(snprintf) || (YYJSON_STDC_VER >= 199901L)
#define snprintf_num(buf, len, fmt, dig, val) \
    snprintf((char *)buf, len, fmt, dig, val)
#else
#define snprintf_num(buf, len, fmt, dig, val) \
    sprintf((char *)buf, fmt, dig, val)
#endif

static_noinline u8 *write_fp_reformat(u8 *buf, int len,
                                    yyjson_write_flag flg, bool fixed) {
    u8 *cur = buf;
    if (unlikely(len < 1)) return NULL;
    cur += (*cur == '-');
    if (unlikely(!digi_is_digit(*cur))) {
        /* nan, inf, or bad output */
        if (has_write_flag(INF_AND_NAN_AS_NULL)) {
            byte_copy_4(buf, "null");
            return buf + 4;
        } else if (has_write_flag(ALLOW_INF_AND_NAN)) {
            if (*cur == 'i') {
                byte_copy_8(cur, "Infinity");
                return cur + 8;
            } else if (*cur == 'n') {
                byte_copy_4(buf, "NaN");
                return buf + 3;
            }
        }
        return NULL;
    } else {
        /* finite number */
        u8 *end = buf + len, *dot = NULL, *exp = NULL;

        /*
         The snprintf() function is locale-dependent. For currently known
         locales, (en, zh, ja, ko, am, he, hi) use '.' as the decimal point,
         while other locales use ',' as the decimal point. we need to replace
         ',' with '.' to avoid the locale setting.
         */
        for (; cur < end; cur++) {
            switch (*cur) {
                case ',': *cur = '.'; /* fallthrough */
                case '.': dot = cur; break;
                case 'e': exp = cur; break;
                default: break;
            }
        }
        if (fixed) {
            /* remove trailing zeros */
            while (*(end - 1) == '0') end--;
            end += *(end - 1) == '.';
        } else {
            if (!dot && !exp) {
                /* add decimal point, e.g. 123 -> 123.0 */
                byte_copy_2(end, ".0");
                end += 2;
            } else if (exp) {
                cur = exp + 1;
                /* remove positive sign in the exponent part */
                if (*cur == '+') {
                    memmove(cur, cur + 1, (usize)(end - cur - 1));
                    end--;
                }
                cur += (*cur == '-');
                /* remove leading zeros in the exponent part */
                if (*cur == '0') {
                    u8 *hdr = cur++;
                    while (*cur == '0') cur++;
                    memmove(hdr, cur, (usize)(end - cur));
                    end -= (usize)(cur - hdr);
                }
            }
        }
        return end;
    }
}

/** Write a double number (requires 40 bytes buffer). */
static_noinline u8 *write_f64_raw(u8 *buf, u64 raw, yyjson_write_flag flg) {
#if defined(DBL_DECIMAL_DIG) && DBL_DECIMAL_DIG < F64_DEC_DIG
    int dig = DBL_DECIMAL_DIG;
#else
    int dig = F64_DEC_DIG;
#endif
    f64 val = f64_from_raw(raw);
    int len = snprintf_num(buf, FP_BUF_LEN, "%.*g", dig, val);
    return write_fp_reformat(buf, len, flg, false);
}

/** Write a double number (requires 40 bytes buffer). */
static_noinline u8 *write_f32_raw(u8 *buf, u64 raw, yyjson_write_flag flg) {
#if defined(FLT_DECIMAL_DIG) && FLT_DECIMAL_DIG < F32_DEC_DIG
    int dig = FLT_DECIMAL_DIG;
#else
    int dig = F32_DEC_DIG;
#endif
    f64 val = (f64)f64_to_f32(f64_from_raw(raw));
    int len = snprintf_num(buf, FP_BUF_LEN, "%.*g", dig, val);
    return write_fp_reformat(buf, len, flg, false);
}

/** Write a double number (requires 40 bytes buffer). */
static_noinline u8 *write_f64_raw_fixed(u8 *buf, u64 raw,
                                        yyjson_write_flag flg, u32 prec) {
    f64 val = (f64)f64_from_raw(raw);
    if (-1e21 < val && val < 1e21) {
        int len = snprintf_num(buf, FP_BUF_LEN, "%.*f", (int)prec, val);
        return write_fp_reformat(buf, len, flg, true);
    } else {
        return write_f64_raw(buf, raw, flg);
    }
}

#endif /* FP_WRITER */

/** Write a JSON number (requires 40 bytes buffer). */
static_inline u8 *write_num(u8 *cur, yyjson_val *val, yyjson_write_flag flg) {
    if (!(val->tag & YYJSON_SUBTYPE_REAL)) {
        u64 pos = val->uni.u64;
        u64 neg = ~pos + 1;
        usize sign = ((val->tag & YYJSON_SUBTYPE_SINT) > 0) & ((i64)pos < 0);
        *cur = '-';
        return write_u64(sign ? neg : pos, cur + sign);
    } else {
        u64 raw = val->uni.u64;
        u32 val_fmt = (u32)(val->tag >> 32);
        u32 all_fmt = flg;
        u32 fmt = val_fmt | all_fmt;
        if (likely(!(fmt >> (32 - YYJSON_WRITE_FP_FLAG_BITS)))) {
            /* double to shortest */
            return write_f64_raw(cur, raw, flg);
        } else if (fmt >> (32 - YYJSON_WRITE_FP_PREC_BITS)) {
            /* double to fixed */
            u32 val_prec = val_fmt >> (32 - YYJSON_WRITE_FP_PREC_BITS);
            u32 all_prec = all_fmt >> (32 - YYJSON_WRITE_FP_PREC_BITS);
            u32 prec = val_prec ? val_prec : all_prec;
            return write_f64_raw_fixed(cur, raw, flg, prec);
        } else {
            if (fmt & YYJSON_WRITE_FP_TO_FLOAT) {
                /* float to shortest */
                return write_f32_raw(cur, raw, flg);
            } else {
                /* double to shortest */
                return write_f64_raw(cur, raw, flg);
            }
        }
    }
}

char *yyjson_write_number(const yyjson_val *val, char *buf) {
    if (unlikely(!val || !buf)) return NULL;
    switch (val->tag & YYJSON_TAG_MASK) {
        case YYJSON_TYPE_NUM | YYJSON_SUBTYPE_UINT: {
            buf = (char *)write_u64(val->uni.u64, (u8 *)buf);
            *buf = '\0';
            return buf;
        }
        case YYJSON_TYPE_NUM | YYJSON_SUBTYPE_SINT: {
            u64 pos = val->uni.u64;
            u64 neg = ~pos + 1;
            usize sign = ((i64)pos < 0);
            *buf = '-';
            buf = (char *)write_u64(sign ? neg : pos, (u8 *)buf + sign);
            *buf = '\0';
            return buf;
        }
        case YYJSON_TYPE_NUM | YYJSON_SUBTYPE_REAL: {
            u64 raw = val->uni.u64;
            u32 fmt = (u32)(val->tag >> 32);
            u32 flg = YYJSON_WRITE_ALLOW_INF_AND_NAN;
            if (likely(!(fmt >> (32 - YYJSON_WRITE_FP_FLAG_BITS)))) {
                buf = (char *)write_f64_raw((u8 *)buf, raw, flg);
            } else if (fmt >> (32 - YYJSON_WRITE_FP_PREC_BITS)) {
                u32 prec = fmt >> (32 - YYJSON_WRITE_FP_PREC_BITS);
                buf = (char *)write_f64_raw_fixed((u8 *)buf, raw, flg, prec);
            } else {
                if (fmt & YYJSON_WRITE_FP_TO_FLOAT) {
                    buf = (char *)write_f32_raw((u8 *)buf, raw, flg);
                } else {
                    buf = (char *)write_f64_raw((u8 *)buf, raw, flg);
                }
            }
            if (buf) *buf = '\0';
            return buf;
        }
        default: return NULL;
    }
}



/*==============================================================================
 * String Writer
 *============================================================================*/

/** Character encode type, if (type > CHAR_ENC_ERR_1) bytes = type / 2; */
typedef u8 char_enc_type;
#define CHAR_ENC_CPY_1  0 /* 1-byte UTF-8, copy. */
#define CHAR_ENC_ERR_1  1 /* 1-byte UTF-8, error. */
#define CHAR_ENC_ESC_A  2 /* 1-byte ASCII, escaped as '\x'. */
#define CHAR_ENC_ESC_1  3 /* 1-byte UTF-8, escaped as '\uXXXX'. */
#define CHAR_ENC_CPY_2  4 /* 2-byte UTF-8, copy. */
#define CHAR_ENC_ESC_2  5 /* 2-byte UTF-8, escaped as '\uXXXX'. */
#define CHAR_ENC_CPY_3  6 /* 3-byte UTF-8, copy. */
#define CHAR_ENC_ESC_3  7 /* 3-byte UTF-8, escaped as '\uXXXX'. */
#define CHAR_ENC_CPY_4  8 /* 4-byte UTF-8, copy. */
#define CHAR_ENC_ESC_4  9 /* 4-byte UTF-8, escaped as '\uXXXX\uXXXX'. */

/** Character encode type table: don't escape unicode, don't escape '/'.
    (generate with misc/make_tables.c) */
static const char_enc_type enc_table_cpy[256] = {
    3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1
};

/** Character encode type table: don't escape unicode, escape '/'.
    (generate with misc/make_tables.c) */
static const char_enc_type enc_table_cpy_slash[256] = {
    3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1
};

/** Character encode type table: escape unicode, don't escape '/'.
    (generate with misc/make_tables.c) */
static const char_enc_type enc_table_esc[256] = {
    3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1
};

/** Character encode type table: escape unicode, escape '/'.
    (generate with misc/make_tables.c) */
static const char_enc_type enc_table_esc_slash[256] = {
    3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1
};

/** Escaped hex character table: ["00" "01" "02" ... "FD" "FE" "FF"].
    (generate with misc/make_tables.c) */
yyjson_align(2)
static const u8 esc_hex_char_table[512] = {
    '0', '0', '0', '1', '0', '2', '0', '3',
    '0', '4', '0', '5', '0', '6', '0', '7',
    '0', '8', '0', '9', '0', 'A', '0', 'B',
    '0', 'C', '0', 'D', '0', 'E', '0', 'F',
    '1', '0', '1', '1', '1', '2', '1', '3',
    '1', '4', '1', '5', '1', '6', '1', '7',
    '1', '8', '1', '9', '1', 'A', '1', 'B',
    '1', 'C', '1', 'D', '1', 'E', '1', 'F',
    '2', '0', '2', '1', '2', '2', '2', '3',
    '2', '4', '2', '5', '2', '6', '2', '7',
    '2', '8', '2', '9', '2', 'A', '2', 'B',
    '2', 'C', '2', 'D', '2', 'E', '2', 'F',
    '3', '0', '3', '1', '3', '2', '3', '3',
    '3', '4', '3', '5', '3', '6', '3', '7',
    '3', '8', '3', '9', '3', 'A', '3', 'B',
    '3', 'C', '3', 'D', '3', 'E', '3', 'F',
    '4', '0', '4', '1', '4', '2', '4', '3',
    '4', '4', '4', '5', '4', '6', '4', '7',
    '4', '8', '4', '9', '4', 'A', '4', 'B',
    '4', 'C', '4', 'D', '4', 'E', '4', 'F',
    '5', '0', '5', '1', '5', '2', '5', '3',
    '5', '4', '5', '5', '5', '6', '5', '7',
    '5', '8', '5', '9', '5', 'A', '5', 'B',
    '5', 'C', '5', 'D', '5', 'E', '5', 'F',
    '6', '0', '6', '1', '6', '2', '6', '3',
    '6', '4', '6', '5', '6', '6', '6', '7',
    '6', '8', '6', '9', '6', 'A', '6', 'B',
    '6', 'C', '6', 'D', '6', 'E', '6', 'F',
    '7', '0', '7', '1', '7', '2', '7', '3',
    '7', '4', '7', '5', '7', '6', '7', '7',
    '7', '8', '7', '9', '7', 'A', '7', 'B',
    '7', 'C', '7', 'D', '7', 'E', '7', 'F',
    '8', '0', '8', '1', '8', '2', '8', '3',
    '8', '4', '8', '5', '8', '6', '8', '7',
    '8', '8', '8', '9', '8', 'A', '8', 'B',
    '8', 'C', '8', 'D', '8', 'E', '8', 'F',
    '9', '0', '9', '1', '9', '2', '9', '3',
    '9', '4', '9', '5', '9', '6', '9', '7',
    '9', '8', '9', '9', '9', 'A', '9', 'B',
    '9', 'C', '9', 'D', '9', 'E', '9', 'F',
    'A', '0', 'A', '1', 'A', '2', 'A', '3',
    'A', '4', 'A', '5', 'A', '6', 'A', '7',
    'A', '8', 'A', '9', 'A', 'A', 'A', 'B',
    'A', 'C', 'A', 'D', 'A', 'E', 'A', 'F',
    'B', '0', 'B', '1', 'B', '2', 'B', '3',
    'B', '4', 'B', '5', 'B', '6', 'B', '7',
    'B', '8', 'B', '9', 'B', 'A', 'B', 'B',
    'B', 'C', 'B', 'D', 'B', 'E', 'B', 'F',
    'C', '0', 'C', '1', 'C', '2', 'C', '3',
    'C', '4', 'C', '5', 'C', '6', 'C', '7',
    'C', '8', 'C', '9', 'C', 'A', 'C', 'B',
    'C', 'C', 'C', 'D', 'C', 'E', 'C', 'F',
    'D', '0', 'D', '1', 'D', '2', 'D', '3',
    'D', '4', 'D', '5', 'D', '6', 'D', '7',
    'D', '8', 'D', '9', 'D', 'A', 'D', 'B',
    'D', 'C', 'D', 'D', 'D', 'E', 'D', 'F',
    'E', '0', 'E', '1', 'E', '2', 'E', '3',
    'E', '4', 'E', '5', 'E', '6', 'E', '7',
    'E', '8', 'E', '9', 'E', 'A', 'E', 'B',
    'E', 'C', 'E', 'D', 'E', 'E', 'E', 'F',
    'F', '0', 'F', '1', 'F', '2', 'F', '3',
    'F', '4', 'F', '5', 'F', '6', 'F', '7',
    'F', '8', 'F', '9', 'F', 'A', 'F', 'B',
    'F', 'C', 'F', 'D', 'F', 'E', 'F', 'F'
};

/** Escaped single character table. (generate with misc/make_tables.c) */
yyjson_align(2)
static const u8 esc_single_char_table[512] = {
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    '\\', 'b', '\\', 't', '\\', 'n', ' ', ' ',
    '\\', 'f', '\\', 'r', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', '\\', '"', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', '\\', '/',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    '\\', '\\', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '
};

/** Returns the encode table with options. */
static_inline const char_enc_type *get_enc_table_with_flag(
    yyjson_write_flag flg) {
    if (has_write_flag(ESCAPE_UNICODE)) {
        if (has_write_flag(ESCAPE_SLASHES)) {
            return enc_table_esc_slash;
        } else {
            return enc_table_esc;
        }
    } else {
        if (has_write_flag(ESCAPE_SLASHES)) {
            return enc_table_cpy_slash;
        } else {
            return enc_table_cpy;
        }
    }
}

/** Write raw string. */
static_inline u8 *write_raw(u8 *cur, const u8 *raw, usize raw_len) {
    memcpy(cur, raw, raw_len);
    return cur + raw_len;
}

/**
 Write string no-escape.
 @param cur Buffer cursor.
 @param str A UTF-8 string, null-terminator is not required.
 @param str_len Length of string in bytes.
 @return The buffer cursor after string.
 */
static_inline u8 *write_str_noesc(u8 *cur, const u8 *str, usize str_len) {
    *cur++ = '"';
    while (str_len >= 16) {
        byte_copy_16(cur, str);
        cur += 16;
        str += 16;
        str_len -= 16;
    }
    while (str_len >= 4) {
        byte_copy_4(cur, str);
        cur += 4;
        str += 4;
        str_len -= 4;
    }
    while (str_len) {
        *cur++ = *str++;
        str_len -= 1;
    }
    *cur++ = '"';
    return cur;
}

/**
 Write UTF-8 string (requires len * 6 + 2 bytes buffer).
 @param cur Buffer cursor.
 @param esc Escape unicode.
 @param inv Allow invalid unicode.
 @param str A UTF-8 string, null-terminator is not required.
 @param str_len Length of string in bytes.
 @param enc_table Encode type table for character.
 @return The buffer cursor after string, or NULL on invalid unicode.
 */
static_inline u8 *write_str(u8 *cur, bool esc, bool inv,
                            const u8 *str, usize str_len,
                            const char_enc_type *enc_table) {

    /* UTF-8 character mask and pattern, see `read_str()` for details. */
#if YYJSON_ENDIAN == YYJSON_BIG_ENDIAN
    const u16 b2_mask = 0xE0C0UL;
    const u16 b2_patt = 0xC080UL;
    const u16 b2_requ = 0x1E00UL;
    const u32 b3_mask = 0xF0C0C000UL;
    const u32 b3_patt = 0xE0808000UL;
    const u32 b3_requ = 0x0F200000UL;
    const u32 b3_erro = 0x0D200000UL;
    const u32 b4_mask = 0xF8C0C0C0UL;
    const u32 b4_patt = 0xF0808080UL;
    const u32 b4_requ = 0x07300000UL;
    const u32 b4_err0 = 0x04000000UL;
    const u32 b4_err1 = 0x03300000UL;
#elif YYJSON_ENDIAN == YYJSON_LITTLE_ENDIAN
    const u16 b2_mask = 0xC0E0UL;
    const u16 b2_patt = 0x80C0UL;
    const u16 b2_requ = 0x001EUL;
    const u32 b3_mask = 0x00C0C0F0UL;
    const u32 b3_patt = 0x008080E0UL;
    const u32 b3_requ = 0x0000200FUL;
    const u32 b3_erro = 0x0000200DUL;
    const u32 b4_mask = 0xC0C0C0F8UL;
    const u32 b4_patt = 0x808080F0UL;
    const u32 b4_requ = 0x00003007UL;
    const u32 b4_err0 = 0x00000004UL;
    const u32 b4_err1 = 0x00003003UL;
#else
    /* this should be evaluated at compile-time */
    v16_uni b2_mask_uni = {{ 0xE0, 0xC0 }};
    v16_uni b2_patt_uni = {{ 0xC0, 0x80 }};
    v16_uni b2_requ_uni = {{ 0x1E, 0x00 }};
    v32_uni b3_mask_uni = {{ 0xF0, 0xC0, 0xC0, 0x00 }};
    v32_uni b3_patt_uni = {{ 0xE0, 0x80, 0x80, 0x00 }};
    v32_uni b3_requ_uni = {{ 0x0F, 0x20, 0x00, 0x00 }};
    v32_uni b3_erro_uni = {{ 0x0D, 0x20, 0x00, 0x00 }};
    v32_uni b4_mask_uni = {{ 0xF8, 0xC0, 0xC0, 0xC0 }};
    v32_uni b4_patt_uni = {{ 0xF0, 0x80, 0x80, 0x80 }};
    v32_uni b4_requ_uni = {{ 0x07, 0x30, 0x00, 0x00 }};
    v32_uni b4_err0_uni = {{ 0x04, 0x00, 0x00, 0x00 }};
    v32_uni b4_err1_uni = {{ 0x03, 0x30, 0x00, 0x00 }};
    u16 b2_mask = b2_mask_uni.u;
    u16 b2_patt = b2_patt_uni.u;
    u16 b2_requ = b2_requ_uni.u;
    u32 b3_mask = b3_mask_uni.u;
    u32 b3_patt = b3_patt_uni.u;
    u32 b3_requ = b3_requ_uni.u;
    u32 b3_erro = b3_erro_uni.u;
    u32 b4_mask = b4_mask_uni.u;
    u32 b4_patt = b4_patt_uni.u;
    u32 b4_requ = b4_requ_uni.u;
    u32 b4_err0 = b4_err0_uni.u;
    u32 b4_err1 = b4_err1_uni.u;
#endif

#define is_valid_seq_2(uni) ( \
    ((uni & b2_mask) == b2_patt) && \
    ((uni & b2_requ)) \
)

#define is_valid_seq_3(uni) ( \
    ((uni & b3_mask) == b3_patt) && \
    ((tmp = (uni & b3_requ))) && \
    ((tmp != b3_erro)) \
)

#define is_valid_seq_4(uni) ( \
    ((uni & b4_mask) == b4_patt) && \
    ((tmp = (uni & b4_requ))) && \
    ((tmp & b4_err0) == 0 || (tmp & b4_err1) == 0) \
)

    /* The replacement character U+FFFD, used to indicate invalid character. */
    const v32 rep = {{ 'F', 'F', 'F', 'D' }};
    const v32 pre = {{ '\\', 'u', '0', '0' }};

    const u8 *src = str;
    const u8 *end = str + str_len;
    *cur++ = '"';

copy_ascii:
    /*
     Copy continuous ASCII, loop unrolling, same as the following code:

         while (end > src) (
            if (unlikely(enc_table[*src])) break;
            *cur++ = *src++;
         );
     */
#define expr_jump(i) \
    if (unlikely(enc_table[src[i]])) goto stop_char_##i;

#define expr_stop(i) \
    stop_char_##i: \
    memcpy(cur, src, i); \
    cur += i; src += i; goto copy_utf8;

    while (end - src >= 16) {
        repeat16_incr(expr_jump)
        byte_copy_16(cur, src);
        cur += 16; src += 16;
    }

    while (end - src >= 4) {
        repeat4_incr(expr_jump)
        byte_copy_4(cur, src);
        cur += 4; src += 4;
    }

    while (end > src) {
        expr_jump(0)
        *cur++ = *src++;
    }

    *cur++ = '"';
    return cur;

    repeat16_incr(expr_stop)

#undef expr_jump
#undef expr_stop

copy_utf8:
    if (unlikely(src + 4 > end)) {
        if (end == src) goto copy_end;
        if (end - src < enc_table[*src] / 2) goto err_one;
    }
    switch (enc_table[*src]) {
        case CHAR_ENC_CPY_1: {
            *cur++ = *src++;
            goto copy_ascii;
        }
        case CHAR_ENC_CPY_2: {
            u16 v;
#if YYJSON_DISABLE_UTF8_VALIDATION
            byte_copy_2(cur, src);
#else
            v = byte_load_2(src);
            if (unlikely(!is_valid_seq_2(v))) goto err_cpy;
            byte_copy_2(cur, src);
#endif
            cur += 2;
            src += 2;
            goto copy_utf8;
        }
        case CHAR_ENC_CPY_3: {
            u32 v, tmp;
#if YYJSON_DISABLE_UTF8_VALIDATION
            if (likely(src + 4 <= end)) {
                byte_copy_4(cur, src);
            } else {
                byte_copy_2(cur, src);
                cur[2] = src[2];
            }
#else
            if (likely(src + 4 <= end)) {
                v = byte_load_4(src);
                if (unlikely(!is_valid_seq_3(v))) goto err_cpy;
                byte_copy_4(cur, src);
            } else {
                v = byte_load_3(src);
                if (unlikely(!is_valid_seq_3(v))) goto err_cpy;
                byte_copy_4(cur, &v);
            }
#endif
            cur += 3;
            src += 3;
            goto copy_utf8;
        }
        case CHAR_ENC_CPY_4: {
            u32 v, tmp;
#if YYJSON_DISABLE_UTF8_VALIDATION
            byte_copy_4(cur, src);
#else
            v = byte_load_4(src);
            if (unlikely(!is_valid_seq_4(v))) goto err_cpy;
            byte_copy_4(cur, src);
#endif
            cur += 4;
            src += 4;
            goto copy_utf8;
        }
        case CHAR_ENC_ESC_A: {
            byte_copy_2(cur, &esc_single_char_table[*src * 2]);
            cur += 2;
            src += 1;
            goto copy_utf8;
        }
        case CHAR_ENC_ESC_1: {
            byte_copy_4(cur + 0, &pre);
            byte_copy_2(cur + 4, &esc_hex_char_table[*src * 2]);
            cur += 6;
            src += 1;
            goto copy_utf8;
        }
        case CHAR_ENC_ESC_2: {
            u16 u, v;
#if !YYJSON_DISABLE_UTF8_VALIDATION
            v = byte_load_2(src);
            if (unlikely(!is_valid_seq_2(v))) goto err_esc;
#endif
            u = (u16)(((u16)(src[0] & 0x1F) << 6) |
                      ((u16)(src[1] & 0x3F) << 0));
            byte_copy_2(cur + 0, &pre);
            byte_copy_2(cur + 2, &esc_hex_char_table[(u >> 8) * 2]);
            byte_copy_2(cur + 4, &esc_hex_char_table[(u & 0xFF) * 2]);
            cur += 6;
            src += 2;
            goto copy_utf8;
        }
        case CHAR_ENC_ESC_3: {
            u16 u;
            u32 v, tmp;
#if !YYJSON_DISABLE_UTF8_VALIDATION
            v = byte_load_3(src);
            if (unlikely(!is_valid_seq_3(v))) goto err_esc;
#endif
            u = (u16)(((u16)(src[0] & 0x0F) << 12) |
                      ((u16)(src[1] & 0x3F) << 6) |
                      ((u16)(src[2] & 0x3F) << 0));
            byte_copy_2(cur + 0, &pre);
            byte_copy_2(cur + 2, &esc_hex_char_table[(u >> 8) * 2]);
            byte_copy_2(cur + 4, &esc_hex_char_table[(u & 0xFF) * 2]);
            cur += 6;
            src += 3;
            goto copy_utf8;
        }
        case CHAR_ENC_ESC_4: {
            u32 hi, lo, u, v, tmp;
#if !YYJSON_DISABLE_UTF8_VALIDATION
            v = byte_load_4(src);
            if (unlikely(!is_valid_seq_4(v))) goto err_esc;
#endif
            u = ((u32)(src[0] & 0x07) << 18) |
                ((u32)(src[1] & 0x3F) << 12) |
                ((u32)(src[2] & 0x3F) << 6) |
                ((u32)(src[3] & 0x3F) << 0);
            u -= 0x10000;
            hi = (u >> 10) + 0xD800;
            lo = (u & 0x3FF) + 0xDC00;
            byte_copy_2(cur + 0, &pre);
            byte_copy_2(cur + 2, &esc_hex_char_table[(hi >> 8) * 2]);
            byte_copy_2(cur + 4, &esc_hex_char_table[(hi & 0xFF) * 2]);
            byte_copy_2(cur + 6, &pre);
            byte_copy_2(cur + 8, &esc_hex_char_table[(lo >> 8) * 2]);
            byte_copy_2(cur + 10, &esc_hex_char_table[(lo & 0xFF) * 2]);
            cur += 12;
            src += 4;
            goto copy_utf8;
        }
        case CHAR_ENC_ERR_1: {
            goto err_one;
        }
        default: break;
    }

copy_end:
    *cur++ = '"';
    return cur;

err_one:
    if (esc) goto err_esc;
    else goto err_cpy;

err_cpy:
    if (!inv) return NULL;
    *cur++ = *src++;
    goto copy_utf8;

err_esc:
    if (!inv) return NULL;
    byte_copy_2(cur + 0, &pre);
    byte_copy_4(cur + 2, &rep);
    cur += 6;
    src += 1;
    goto copy_utf8;

#undef is_valid_seq_2
#undef is_valid_seq_3
#undef is_valid_seq_4
}



/*==============================================================================
 * Writer Utilities
 *============================================================================*/

/** Write null (requires 8 bytes buffer). */
static_inline u8 *write_null(u8 *cur) {
    v64 v = {{ 'n', 'u', 'l', 'l', ',', '\n', 0, 0 }};
    byte_copy_8(cur, &v);
    return cur + 4;
}

/** Write bool (requires 8 bytes buffer). */
static_inline u8 *write_bool(u8 *cur, bool val) {
    v64 v0 = {{ 'f', 'a', 'l', 's', 'e', ',', '\n', 0 }};
    v64 v1 = {{ 't', 'r', 'u', 'e', ',', '\n', 0, 0 }};
    if (val) {
        byte_copy_8(cur, &v1);
    } else {
        byte_copy_8(cur, &v0);
    }
    return cur + 5 - val;
}

/** Write indent (requires level x 4 bytes buffer).
    Param spaces should not larger than 4. */
static_inline u8 *write_indent(u8 *cur, usize level, usize spaces) {
    while (level-- > 0) {
        byte_copy_4(cur, "    ");
        cur += spaces;
    }
    return cur;
}

/** Write data to file pointer. */
static bool write_dat_to_fp(FILE *fp, u8 *dat, usize len,
                            yyjson_write_err *err) {
    if (fwrite(dat, len, 1, fp) != 1) {
        err->msg = "file writing failed";
        err->code = YYJSON_WRITE_ERROR_FILE_WRITE;
        return false;
    }
    return true;
}

/** Write data to file. */
static bool write_dat_to_file(const char *path, u8 *dat, usize len,
                              yyjson_write_err *err) {

#define return_err(_code, _msg) do { \
    err->msg = _msg; \
    err->code = YYJSON_WRITE_ERROR_##_code; \
    if (file) fclose(file); \
    return false; \
} while (false)

    FILE *file = fopen_writeonly(path);
    if (file == NULL) {
        return_err(FILE_OPEN, MSG_FOPEN);
    }
    if (fwrite(dat, len, 1, file) != 1) {
        return_err(FILE_WRITE, MSG_FWRITE);
    }
    if (fclose(file) != 0) {
        file = NULL;
        return_err(FILE_WRITE, MSG_FCLOSE);
    }
    return true;

#undef return_err
}



/*==============================================================================
 * JSON Writer Implementation
 *============================================================================*/

typedef struct yyjson_write_ctx {
    usize tag;
} yyjson_write_ctx;

static_inline void yyjson_write_ctx_set(yyjson_write_ctx *ctx,
                                        usize size, bool is_obj) {
    ctx->tag = (size << 1) | (usize)is_obj;
}

static_inline void yyjson_write_ctx_get(yyjson_write_ctx *ctx,
                                        usize *size, bool *is_obj) {
    usize tag = ctx->tag;
    *size = tag >> 1;
    *is_obj = (bool)(tag & 1);
}

/** Write single JSON value. */
static_inline u8 *yyjson_write_single(yyjson_val *val,
                                      yyjson_write_flag flg,
                                      yyjson_alc alc,
                                      usize *dat_len,
                                      yyjson_write_err *err) {

#define return_err(_code, _msg) do { \
    if (hdr) alc.free(alc.ctx, (void *)hdr); \
    *dat_len = 0; \
    err->code = YYJSON_WRITE_ERROR_##_code; \
    err->msg = _msg; \
    return NULL; \
} while (false)

#define incr_len(_len) do { \
    hdr = (u8 *)alc.malloc(alc.ctx, _len); \
    if (!hdr) goto fail_alloc; \
    cur = hdr; \
} while (false)

#define check_str_len(_len) do { \
    if ((sizeof(usize) < 8) && (_len >= (USIZE_MAX - 16) / 6)) \
        goto fail_alloc; \
} while (false)

    u8 *hdr = NULL, *cur;
    usize str_len;
    const u8 *str_ptr;
    const char_enc_type *enc_table = get_enc_table_with_flag(flg);
    bool cpy = (enc_table == enc_table_cpy);
    bool esc = has_write_flag(ESCAPE_UNICODE) != 0;
    bool inv = has_write_flag(ALLOW_INVALID_UNICODE) != 0;
    bool newline = has_write_flag(NEWLINE_AT_END) != 0;
    const usize end_len = 2; /* '\n' and '\0' */

    switch (unsafe_yyjson_get_type(val)) {
        case YYJSON_TYPE_RAW:
            str_len = unsafe_yyjson_get_len(val);
            str_ptr = (const u8 *)unsafe_yyjson_get_str(val);
            check_str_len(str_len);
            incr_len(str_len + end_len);
            cur = write_raw(cur, str_ptr, str_len);
            break;

        case YYJSON_TYPE_STR:
            str_len = unsafe_yyjson_get_len(val);
            str_ptr = (const u8 *)unsafe_yyjson_get_str(val);
            check_str_len(str_len);
            incr_len(str_len * 6 + 2 + end_len);
            if (likely(cpy) && unsafe_yyjson_get_subtype(val)) {
                cur = write_str_noesc(cur, str_ptr, str_len);
            } else {
                cur = write_str(cur, esc, inv, str_ptr, str_len, enc_table);
                if (unlikely(!cur)) goto fail_str;
            }
            break;

        case YYJSON_TYPE_NUM:
            incr_len(FP_BUF_LEN + end_len);
            cur = write_num(cur, val, flg);
            if (unlikely(!cur)) goto fail_num;
            break;

        case YYJSON_TYPE_BOOL:
            incr_len(8);
            cur = write_bool(cur, unsafe_yyjson_get_bool(val));
            break;

        case YYJSON_TYPE_NULL:
            incr_len(8);
            cur = write_null(cur);
            break;

        case YYJSON_TYPE_ARR:
            incr_len(2 + end_len);
            byte_copy_2(cur, "[]");
            cur += 2;
            break;

        case YYJSON_TYPE_OBJ:
            incr_len(2 + end_len);
            byte_copy_2(cur, "{}");
            cur += 2;
            break;

        default:
            goto fail_type;
    }

    if (newline) *cur++ = '\n';
    *cur = '\0';
    *dat_len = (usize)(cur - hdr);
    memset(err, 0, sizeof(yyjson_write_err));
    return hdr;

fail_alloc: return_err(MEMORY_ALLOCATION, MSG_MALLOC);
fail_type:  return_err(INVALID_VALUE_TYPE, MSG_ERR_TYPE);
fail_num:   return_err(NAN_OR_INF, MSG_INF_NAN);
fail_str:   return_err(INVALID_STRING, MSG_ERR_UTF8);

#undef return_err
#undef check_str_len
#undef incr_len
}

/** Write JSON document minify.
    The root of this document should be a non-empty container. */
static_inline u8 *yyjson_write_minify(const yyjson_val *root,
                                      const yyjson_write_flag flg,
                                      const yyjson_alc alc,
                                      usize *dat_len,
                                      yyjson_write_err *err) {

#define return_err(_code, _msg) do { \
    *dat_len = 0; \
    err->code = YYJSON_WRITE_ERROR_##_code; \
    err->msg = _msg; \
    if (hdr) alc.free(alc.ctx, hdr); \
    return NULL; \
} while (false)

#define incr_len(_len) do { \
    ext_len = (usize)(_len); \
    if (unlikely((u8 *)(cur + ext_len) >= (u8 *)ctx)) { \
        usize ctx_pos = (usize)((u8 *)ctx - hdr); \
        usize cur_pos = (usize)(cur - hdr); \
        ctx_len = (usize)(end - (u8 *)ctx); \
        alc_inc = yyjson_max(alc_len / 2, ext_len); \
        alc_inc = size_align_up(alc_inc, sizeof(yyjson_write_ctx)); \
        if ((sizeof(usize) < 8) && size_add_is_overflow(alc_len, alc_inc)) \
            goto fail_alloc; \
        alc_len += alc_inc; \
        tmp = (u8 *)alc.realloc(alc.ctx, hdr, alc_len - alc_inc, alc_len); \
        if (unlikely(!tmp)) goto fail_alloc; \
        ctx_tmp = (yyjson_write_ctx *)(void *)(tmp + (alc_len - ctx_len)); \
        memmove((void *)ctx_tmp, (void *)(tmp + ctx_pos), ctx_len); \
        ctx = ctx_tmp; \
        cur = tmp + cur_pos; \
        end = tmp + alc_len; \
        hdr = tmp; \
    } \
} while (false)

#define check_str_len(_len) do { \
    if ((sizeof(usize) < 8) && (_len >= (USIZE_MAX - 16) / 6)) \
        goto fail_alloc; \
} while (false)

    yyjson_val *val;
    yyjson_type val_type;
    usize ctn_len, ctn_len_tmp;
    bool ctn_obj, ctn_obj_tmp, is_key;
    u8 *hdr, *cur, *end, *tmp;
    yyjson_write_ctx *ctx, *ctx_tmp;
    usize alc_len, alc_inc, ctx_len, ext_len, str_len;
    const u8 *str_ptr;
    const char_enc_type *enc_table = get_enc_table_with_flag(flg);
    bool cpy = (enc_table == enc_table_cpy);
    bool esc = has_write_flag(ESCAPE_UNICODE) != 0;
    bool inv = has_write_flag(ALLOW_INVALID_UNICODE) != 0;
    bool newline = has_write_flag(NEWLINE_AT_END) != 0;

    alc_len = root->uni.ofs / sizeof(yyjson_val);
    alc_len = alc_len * YYJSON_WRITER_ESTIMATED_MINIFY_RATIO + 64;
    alc_len = size_align_up(alc_len, sizeof(yyjson_write_ctx));
    hdr = (u8 *)alc.malloc(alc.ctx, alc_len);
    if (!hdr) goto fail_alloc;
    cur = hdr;
    end = hdr + alc_len;
    ctx = (yyjson_write_ctx *)(void *)end;

doc_begin:
    val = constcast(yyjson_val *)root;
    val_type = unsafe_yyjson_get_type(val);
    ctn_obj = (val_type == YYJSON_TYPE_OBJ);
    ctn_len = unsafe_yyjson_get_len(val) << (u8)ctn_obj;
    *cur++ = (u8)('[' | ((u8)ctn_obj << 5));
    val++;

val_begin:
    val_type = unsafe_yyjson_get_type(val);
    if (val_type == YYJSON_TYPE_STR) {
        is_key = ((u8)ctn_obj & (u8)~ctn_len);
        str_len = unsafe_yyjson_get_len(val);
        str_ptr = (const u8 *)unsafe_yyjson_get_str(val);
        check_str_len(str_len);
        incr_len(str_len * 6 + 16);
        if (likely(cpy) && unsafe_yyjson_get_subtype(val)) {
            cur = write_str_noesc(cur, str_ptr, str_len);
        } else {
            cur = write_str(cur, esc, inv, str_ptr, str_len, enc_table);
            if (unlikely(!cur)) goto fail_str;
        }
        *cur++ = is_key ? ':' : ',';
        goto val_end;
    }
    if (val_type == YYJSON_TYPE_NUM) {
        incr_len(FP_BUF_LEN);
        cur = write_num(cur, val, flg);
        if (unlikely(!cur)) goto fail_num;
        *cur++ = ',';
        goto val_end;
    }
    if ((val_type & (YYJSON_TYPE_ARR & YYJSON_TYPE_OBJ)) ==
                    (YYJSON_TYPE_ARR & YYJSON_TYPE_OBJ)) {
        ctn_len_tmp = unsafe_yyjson_get_len(val);
        ctn_obj_tmp = (val_type == YYJSON_TYPE_OBJ);
        incr_len(16);
        if (unlikely(ctn_len_tmp == 0)) {
            /* write empty container */
            *cur++ = (u8)('[' | ((u8)ctn_obj_tmp << 5));
            *cur++ = (u8)(']' | ((u8)ctn_obj_tmp << 5));
            *cur++ = ',';
            goto val_end;
        } else {
            /* push context, setup new container */
            yyjson_write_ctx_set(--ctx, ctn_len, ctn_obj);
            ctn_len = ctn_len_tmp << (u8)ctn_obj_tmp;
            ctn_obj = ctn_obj_tmp;
            *cur++ = (u8)('[' | ((u8)ctn_obj << 5));
            val++;
            goto val_begin;
        }
    }
    if (val_type == YYJSON_TYPE_BOOL) {
        incr_len(16);
        cur = write_bool(cur, unsafe_yyjson_get_bool(val));
        cur++;
        goto val_end;
    }
    if (val_type == YYJSON_TYPE_NULL) {
        incr_len(16);
        cur = write_null(cur);
        cur++;
        goto val_end;
    }
    if (val_type == YYJSON_TYPE_RAW) {
        str_len = unsafe_yyjson_get_len(val);
        str_ptr = (const u8 *)unsafe_yyjson_get_str(val);
        check_str_len(str_len);
        incr_len(str_len + 2);
        cur = write_raw(cur, str_ptr, str_len);
        *cur++ = ',';
        goto val_end;
    }
    goto fail_type;

val_end:
    val++;
    ctn_len--;
    if (unlikely(ctn_len == 0)) goto ctn_end;
    goto val_begin;

ctn_end:
    cur--;
    *cur++ = (u8)(']' | ((u8)ctn_obj << 5));
    *cur++ = ',';
    if (unlikely((u8 *)ctx >= end)) goto doc_end;
    yyjson_write_ctx_get(ctx++, &ctn_len, &ctn_obj);
    ctn_len--;
    if (likely(ctn_len > 0)) {
        goto val_begin;
    } else {
        goto ctn_end;
    }

doc_end:
    if (newline) {
        incr_len(2);
        *(cur - 1) = '\n';
        cur++;
    }
    *--cur = '\0';
    *dat_len = (usize)(cur - hdr);
    memset(err, 0, sizeof(yyjson_write_err));
    return hdr;

fail_alloc: return_err(MEMORY_ALLOCATION, MSG_MALLOC);
fail_type:  return_err(INVALID_VALUE_TYPE, MSG_ERR_TYPE);
fail_num:   return_err(NAN_OR_INF, MSG_INF_NAN);
fail_str:   return_err(INVALID_STRING, MSG_ERR_UTF8);

#undef return_err
#undef incr_len
#undef check_str_len
}

/** Write JSON document pretty.
    The root of this document should be a non-empty container. */
static_inline u8 *yyjson_write_pretty(const yyjson_val *root,
                                      const yyjson_write_flag flg,
                                      const yyjson_alc alc,
                                      usize *dat_len,
                                      yyjson_write_err *err) {

#define return_err(_code, _msg) do { \
    *dat_len = 0; \
    err->code = YYJSON_WRITE_ERROR_##_code; \
    err->msg = _msg; \
    if (hdr) alc.free(alc.ctx, hdr); \
    return NULL; \
} while (false)

#define incr_len(_len) do { \
    ext_len = (usize)(_len); \
    if (unlikely((u8 *)(cur + ext_len) >= (u8 *)ctx)) { \
        usize ctx_pos = (usize)((u8 *)ctx - hdr); \
        usize cur_pos = (usize)(cur - hdr); \
        ctx_len = (usize)(end - (u8 *)ctx); \
        alc_inc = yyjson_max(alc_len / 2, ext_len); \
        alc_inc = size_align_up(alc_inc, sizeof(yyjson_write_ctx)); \
        if ((sizeof(usize) < 8) && size_add_is_overflow(alc_len, alc_inc)) \
            goto fail_alloc; \
        alc_len += alc_inc; \
        tmp = (u8 *)alc.realloc(alc.ctx, hdr, alc_len - alc_inc, alc_len); \
        if (unlikely(!tmp)) goto fail_alloc; \
        ctx_tmp = (yyjson_write_ctx *)(void *)(tmp + (alc_len - ctx_len)); \
        memmove((void *)ctx_tmp, (void *)(tmp + ctx_pos), ctx_len); \
        ctx = ctx_tmp; \
        cur = tmp + cur_pos; \
        end = tmp + alc_len; \
        hdr = tmp; \
    } \
} while (false)

#define check_str_len(_len) do { \
    if ((sizeof(usize) < 8) && (_len >= (USIZE_MAX - 16) / 6)) \
        goto fail_alloc; \
} while (false)

    yyjson_val *val;
    yyjson_type val_type;
    usize ctn_len, ctn_len_tmp;
    bool ctn_obj, ctn_obj_tmp, is_key, no_indent;
    u8 *hdr, *cur, *end, *tmp;
    yyjson_write_ctx *ctx, *ctx_tmp;
    usize alc_len, alc_inc, ctx_len, ext_len, str_len, level;
    const u8 *str_ptr;
    const char_enc_type *enc_table = get_enc_table_with_flag(flg);
    bool cpy = (enc_table == enc_table_cpy);
    bool esc = has_write_flag(ESCAPE_UNICODE) != 0;
    bool inv = has_write_flag(ALLOW_INVALID_UNICODE) != 0;
    usize spaces = has_write_flag(PRETTY_TWO_SPACES) ? 2 : 4;
    bool newline = has_write_flag(NEWLINE_AT_END) != 0;

    alc_len = root->uni.ofs / sizeof(yyjson_val);
    alc_len = alc_len * YYJSON_WRITER_ESTIMATED_PRETTY_RATIO + 64;
    alc_len = size_align_up(alc_len, sizeof(yyjson_write_ctx));
    hdr = (u8 *)alc.malloc(alc.ctx, alc_len);
    if (!hdr) goto fail_alloc;
    cur = hdr;
    end = hdr + alc_len;
    ctx = (yyjson_write_ctx *)(void *)end;

doc_begin:
    val = constcast(yyjson_val *)root;
    val_type = unsafe_yyjson_get_type(val);
    ctn_obj = (val_type == YYJSON_TYPE_OBJ);
    ctn_len = unsafe_yyjson_get_len(val) << (u8)ctn_obj;
    *cur++ = (u8)('[' | ((u8)ctn_obj << 5));
    *cur++ = '\n';
    val++;
    level = 1;

val_begin:
    val_type = unsafe_yyjson_get_type(val);
    if (val_type == YYJSON_TYPE_STR) {
        is_key = (bool)((u8)ctn_obj & (u8)~ctn_len);
        no_indent = (bool)((u8)ctn_obj & (u8)ctn_len);
        str_len = unsafe_yyjson_get_len(val);
        str_ptr = (const u8 *)unsafe_yyjson_get_str(val);
        check_str_len(str_len);
        incr_len(str_len * 6 + 16 + (no_indent ? 0 : level * 4));
        cur = write_indent(cur, no_indent ? 0 : level, spaces);
        if (likely(cpy) && unsafe_yyjson_get_subtype(val)) {
            cur = write_str_noesc(cur, str_ptr, str_len);
        } else {
            cur = write_str(cur, esc, inv, str_ptr, str_len, enc_table);
            if (unlikely(!cur)) goto fail_str;
        }
        *cur++ = is_key ? ':' : ',';
        *cur++ = is_key ? ' ' : '\n';
        goto val_end;
    }
    if (val_type == YYJSON_TYPE_NUM) {
        no_indent = (bool)((u8)ctn_obj & (u8)ctn_len);
        incr_len(FP_BUF_LEN + (no_indent ? 0 : level * 4));
        cur = write_indent(cur, no_indent ? 0 : level, spaces);
        cur = write_num(cur, val, flg);
        if (unlikely(!cur)) goto fail_num;
        *cur++ = ',';
        *cur++ = '\n';
        goto val_end;
    }
    if ((val_type & (YYJSON_TYPE_ARR & YYJSON_TYPE_OBJ)) ==
                    (YYJSON_TYPE_ARR & YYJSON_TYPE_OBJ)) {
        no_indent = (bool)((u8)ctn_obj & (u8)ctn_len);
        ctn_len_tmp = unsafe_yyjson_get_len(val);
        ctn_obj_tmp = (val_type == YYJSON_TYPE_OBJ);
        if (unlikely(ctn_len_tmp == 0)) {
            /* write empty container */
            incr_len(16 + (no_indent ? 0 : level * 4));
            cur = write_indent(cur, no_indent ? 0 : level, spaces);
            *cur++ = (u8)('[' | ((u8)ctn_obj_tmp << 5));
            *cur++ = (u8)(']' | ((u8)ctn_obj_tmp << 5));
            *cur++ = ',';
            *cur++ = '\n';
            goto val_end;
        } else {
            /* push context, setup new container */
            incr_len(32 + (no_indent ? 0 : level * 4));
            yyjson_write_ctx_set(--ctx, ctn_len, ctn_obj);
            ctn_len = ctn_len_tmp << (u8)ctn_obj_tmp;
            ctn_obj = ctn_obj_tmp;
            cur = write_indent(cur, no_indent ? 0 : level, spaces);
            level++;
            *cur++ = (u8)('[' | ((u8)ctn_obj << 5));
            *cur++ = '\n';
            val++;
            goto val_begin;
        }
    }
    if (val_type == YYJSON_TYPE_BOOL) {
        no_indent = (bool)((u8)ctn_obj & (u8)ctn_len);
        incr_len(16 + (no_indent ? 0 : level * 4));
        cur = write_indent(cur, no_indent ? 0 : level, spaces);
        cur = write_bool(cur, unsafe_yyjson_get_bool(val));
        cur += 2;
        goto val_end;
    }
    if (val_type == YYJSON_TYPE_NULL) {
        no_indent = (bool)((u8)ctn_obj & (u8)ctn_len);
        incr_len(16 + (no_indent ? 0 : level * 4));
        cur = write_indent(cur, no_indent ? 0 : level, spaces);
        cur = write_null(cur);
        cur += 2;
        goto val_end;
    }
    if (val_type == YYJSON_TYPE_RAW) {
        no_indent = (bool)((u8)ctn_obj & (u8)ctn_len);
        str_len = unsafe_yyjson_get_len(val);
        str_ptr = (const u8 *)unsafe_yyjson_get_str(val);
        check_str_len(str_len);
        incr_len(str_len + 3 + (no_indent ? 0 : level * 4));
        cur = write_indent(cur, no_indent ? 0 : level, spaces);
        cur = write_raw(cur, str_ptr, str_len);
        *cur++ = ',';
        *cur++ = '\n';
        goto val_end;
    }
    goto fail_type;

val_end:
    val++;
    ctn_len--;
    if (unlikely(ctn_len == 0)) goto ctn_end;
    goto val_begin;

ctn_end:
    cur -= 2;
    *cur++ = '\n';
    incr_len(level * 4);
    cur = write_indent(cur, --level, spaces);
    *cur++ = (u8)(']' | ((u8)ctn_obj << 5));
    if (unlikely((u8 *)ctx >= end)) goto doc_end;
    yyjson_write_ctx_get(ctx++, &ctn_len, &ctn_obj);
    ctn_len--;
    *cur++ = ',';
    *cur++ = '\n';
    if (likely(ctn_len > 0)) {
        goto val_begin;
    } else {
        goto ctn_end;
    }

doc_end:
    if (newline) {
        incr_len(2);
        *cur++ = '\n';
    }
    *cur = '\0';
    *dat_len = (usize)(cur - hdr);
    memset(err, 0, sizeof(yyjson_write_err));
    return hdr;

fail_alloc: return_err(MEMORY_ALLOCATION, MSG_MALLOC);
fail_type:  return_err(INVALID_VALUE_TYPE, MSG_ERR_TYPE);
fail_num:   return_err(NAN_OR_INF, MSG_INF_NAN);
fail_str:   return_err(INVALID_STRING, MSG_ERR_UTF8);

#undef return_err
#undef incr_len
#undef check_str_len
}

char *yyjson_val_write_opts(const yyjson_val *val,
                            yyjson_write_flag flg,
                            const yyjson_alc *alc_ptr,
                            usize *dat_len,
                            yyjson_write_err *err) {
    yyjson_write_err dummy_err;
    usize dummy_dat_len;
    yyjson_alc alc = alc_ptr ? *alc_ptr : YYJSON_DEFAULT_ALC;
    yyjson_val *root = constcast(yyjson_val *)val;

    err = err ? err : &dummy_err;
    dat_len = dat_len ? dat_len : &dummy_dat_len;

    if (unlikely(!root)) {
        *dat_len = 0;
        err->msg = "input JSON is NULL";
        err->code = YYJSON_READ_ERROR_INVALID_PARAMETER;
        return NULL;
    }

    if (!unsafe_yyjson_is_ctn(root) || unsafe_yyjson_get_len(root) == 0) {
        return (char *)yyjson_write_single(root, flg, alc, dat_len, err);
    } else if (flg & (YYJSON_WRITE_PRETTY | YYJSON_WRITE_PRETTY_TWO_SPACES)) {
        return (char *)yyjson_write_pretty(root, flg, alc, dat_len, err);
    } else {
        return (char *)yyjson_write_minify(root, flg, alc, dat_len, err);
    }
}

char *yyjson_write_opts(const yyjson_doc *doc,
                        yyjson_write_flag flg,
                        const yyjson_alc *alc_ptr,
                        usize *dat_len,
                        yyjson_write_err *err) {
    yyjson_val *root = doc ? doc->root : NULL;
    return yyjson_val_write_opts(root, flg, alc_ptr, dat_len, err);
}

bool yyjson_val_write_file(const char *path,
                           const yyjson_val *val,
                           yyjson_write_flag flg,
                           const yyjson_alc *alc_ptr,
                           yyjson_write_err *err) {
    yyjson_write_err dummy_err;
    yyjson_alc alc = alc_ptr ? *alc_ptr : YYJSON_DEFAULT_ALC;
    u8 *dat;
    usize dat_len = 0;
    yyjson_val *root = constcast(yyjson_val *)val;
    bool suc;

    err = err ? err : &dummy_err;
    if (unlikely(!path || !*path)) {
        err->msg = "input path is invalid";
        err->code = YYJSON_READ_ERROR_INVALID_PARAMETER;
        return false;
    }

    dat = (u8 *)yyjson_val_write_opts(root, flg, &alc, &dat_len, err);
    if (unlikely(!dat)) return false;
    suc = write_dat_to_file(path, dat, dat_len, err);
    alc.free(alc.ctx, dat);
    return suc;
}

bool yyjson_val_write_fp(FILE *fp,
                         const yyjson_val *val,
                         yyjson_write_flag flg,
                         const yyjson_alc *alc_ptr,
                         yyjson_write_err *err) {
    yyjson_write_err dummy_err;
    yyjson_alc alc = alc_ptr ? *alc_ptr : YYJSON_DEFAULT_ALC;
    u8 *dat;
    usize dat_len = 0;
    yyjson_val *root = constcast(yyjson_val *)val;
    bool suc;

    err = err ? err : &dummy_err;
    if (unlikely(!fp)) {
        err->msg = "input fp is invalid";
        err->code = YYJSON_READ_ERROR_INVALID_PARAMETER;
        return false;
    }

    dat = (u8 *)yyjson_val_write_opts(root, flg, &alc, &dat_len, err);
    if (unlikely(!dat)) return false;
    suc = write_dat_to_fp(fp, dat, dat_len, err);
    alc.free(alc.ctx, dat);
    return suc;
}

bool yyjson_write_file(const char *path,
                       const yyjson_doc *doc,
                       yyjson_write_flag flg,
                       const yyjson_alc *alc_ptr,
                       yyjson_write_err *err) {
    yyjson_val *root = doc ? doc->root : NULL;
    return yyjson_val_write_file(path, root, flg, alc_ptr, err);
}

bool yyjson_write_fp(FILE *fp,
                     const yyjson_doc *doc,
                     yyjson_write_flag flg,
                     const yyjson_alc *alc_ptr,
                     yyjson_write_err *err) {
    yyjson_val *root = doc ? doc->root : NULL;
    return yyjson_val_write_fp(fp, root, flg, alc_ptr, err);
}



/*==============================================================================
 * Mutable JSON Writer Implementation
 *============================================================================*/

typedef struct yyjson_mut_write_ctx {
    usize tag;
    yyjson_mut_val *ctn;
} yyjson_mut_write_ctx;

static_inline void yyjson_mut_write_ctx_set(yyjson_mut_write_ctx *ctx,
                                            yyjson_mut_val *ctn,
                                            usize size, bool is_obj) {
    ctx->tag = (size << 1) | (usize)is_obj;
    ctx->ctn = ctn;
}

static_inline void yyjson_mut_write_ctx_get(yyjson_mut_write_ctx *ctx,
                                            yyjson_mut_val **ctn,
                                            usize *size, bool *is_obj) {
    usize tag = ctx->tag;
    *size = tag >> 1;
    *is_obj = (bool)(tag & 1);
    *ctn = ctx->ctn;
}

/** Get the estimated number of values for the mutable JSON document. */
static_inline usize yyjson_mut_doc_estimated_val_num(
    const yyjson_mut_doc *doc) {
    usize sum = 0;
    yyjson_val_chunk *chunk = doc->val_pool.chunks;
    while (chunk) {
        sum += chunk->chunk_size / sizeof(yyjson_mut_val) - 1;
        if (chunk == doc->val_pool.chunks) {
            sum -= (usize)(doc->val_pool.end - doc->val_pool.cur);
        }
        chunk = chunk->next;
    }
    return sum;
}

/** Write single JSON value. */
static_inline u8 *yyjson_mut_write_single(yyjson_mut_val *val,
                                          yyjson_write_flag flg,
                                          yyjson_alc alc,
                                          usize *dat_len,
                                          yyjson_write_err *err) {
    return yyjson_write_single((yyjson_val *)val, flg, alc, dat_len, err);
}

/** Write JSON document minify.
    The root of this document should be a non-empty container. */
static_inline u8 *yyjson_mut_write_minify(const yyjson_mut_val *root,
                                          usize estimated_val_num,
                                          yyjson_write_flag flg,
                                          yyjson_alc alc,
                                          usize *dat_len,
                                          yyjson_write_err *err) {

#define return_err(_code, _msg) do { \
    *dat_len = 0; \
    err->code = YYJSON_WRITE_ERROR_##_code; \
    err->msg = _msg; \
    if (hdr) alc.free(alc.ctx, hdr); \
    return NULL; \
} while (false)

#define incr_len(_len) do { \
    ext_len = (usize)(_len); \
    if (unlikely((u8 *)(cur + ext_len) >= (u8 *)ctx)) { \
        usize ctx_pos = (usize)((u8 *)ctx - hdr); \
        usize cur_pos = (usize)(cur - hdr); \
        ctx_len = (usize)(end - (u8 *)ctx); \
        alc_inc = yyjson_max(alc_len / 2, ext_len); \
        alc_inc = size_align_up(alc_inc, sizeof(yyjson_mut_write_ctx)); \
        if ((sizeof(usize) < 8) && size_add_is_overflow(alc_len, alc_inc)) \
            goto fail_alloc; \
        alc_len += alc_inc; \
        tmp = (u8 *)alc.realloc(alc.ctx, hdr, alc_len - alc_inc, alc_len); \
        if (unlikely(!tmp)) goto fail_alloc; \
        ctx_tmp = (yyjson_mut_write_ctx *)(void *)(tmp + (alc_len - ctx_len)); \
        memmove((void *)ctx_tmp, (void *)(tmp + ctx_pos), ctx_len); \
        ctx = ctx_tmp; \
        cur = tmp + cur_pos; \
        end = tmp + alc_len; \
        hdr = tmp; \
    } \
} while (false)

#define check_str_len(_len) do { \
    if ((sizeof(usize) < 8) && (_len >= (USIZE_MAX - 16) / 6)) \
        goto fail_alloc; \
} while (false)

    yyjson_mut_val *val, *ctn;
    yyjson_type val_type;
    usize ctn_len, ctn_len_tmp;
    bool ctn_obj, ctn_obj_tmp, is_key;
    u8 *hdr, *cur, *end, *tmp;
    yyjson_mut_write_ctx *ctx, *ctx_tmp;
    usize alc_len, alc_inc, ctx_len, ext_len, str_len;
    const u8 *str_ptr;
    const char_enc_type *enc_table = get_enc_table_with_flag(flg);
    bool cpy = (enc_table == enc_table_cpy);
    bool esc = has_write_flag(ESCAPE_UNICODE) != 0;
    bool inv = has_write_flag(ALLOW_INVALID_UNICODE) != 0;
    bool newline = has_write_flag(NEWLINE_AT_END) != 0;

    alc_len = estimated_val_num * YYJSON_WRITER_ESTIMATED_MINIFY_RATIO + 64;
    alc_len = size_align_up(alc_len, sizeof(yyjson_mut_write_ctx));
    hdr = (u8 *)alc.malloc(alc.ctx, alc_len);
    if (!hdr) goto fail_alloc;
    cur = hdr;
    end = hdr + alc_len;
    ctx = (yyjson_mut_write_ctx *)(void *)end;

doc_begin:
    val = constcast(yyjson_mut_val *)root;
    val_type = unsafe_yyjson_get_type(val);
    ctn_obj = (val_type == YYJSON_TYPE_OBJ);
    ctn_len = unsafe_yyjson_get_len(val) << (u8)ctn_obj;
    *cur++ = (u8)('[' | ((u8)ctn_obj << 5));
    ctn = val;
    val = (yyjson_mut_val *)val->uni.ptr; /* tail */
    val = ctn_obj ? val->next->next : val->next;

val_begin:
    val_type = unsafe_yyjson_get_type(val);
    if (val_type == YYJSON_TYPE_STR) {
        is_key = ((u8)ctn_obj & (u8)~ctn_len);
        str_len = unsafe_yyjson_get_len(val);
        str_ptr = (const u8 *)unsafe_yyjson_get_str(val);
        check_str_len(str_len);
        incr_len(str_len * 6 + 16);
        if (likely(cpy) && unsafe_yyjson_get_subtype(val)) {
            cur = write_str_noesc(cur, str_ptr, str_len);
        } else {
            cur = write_str(cur, esc, inv, str_ptr, str_len, enc_table);
            if (unlikely(!cur)) goto fail_str;
        }
        *cur++ = is_key ? ':' : ',';
        goto val_end;
    }
    if (val_type == YYJSON_TYPE_NUM) {
        incr_len(FP_BUF_LEN);
        cur = write_num(cur, (yyjson_val *)val, flg);
        if (unlikely(!cur)) goto fail_num;
        *cur++ = ',';
        goto val_end;
    }
    if ((val_type & (YYJSON_TYPE_ARR & YYJSON_TYPE_OBJ)) ==
                    (YYJSON_TYPE_ARR & YYJSON_TYPE_OBJ)) {
        ctn_len_tmp = unsafe_yyjson_get_len(val);
        ctn_obj_tmp = (val_type == YYJSON_TYPE_OBJ);
        incr_len(16);
        if (unlikely(ctn_len_tmp == 0)) {
            /* write empty container */
            *cur++ = (u8)('[' | ((u8)ctn_obj_tmp << 5));
            *cur++ = (u8)(']' | ((u8)ctn_obj_tmp << 5));
            *cur++ = ',';
            goto val_end;
        } else {
            /* push context, setup new container */
            yyjson_mut_write_ctx_set(--ctx, ctn, ctn_len, ctn_obj);
            ctn_len = ctn_len_tmp << (u8)ctn_obj_tmp;
            ctn_obj = ctn_obj_tmp;
            *cur++ = (u8)('[' | ((u8)ctn_obj << 5));
            ctn = val;
            val = (yyjson_mut_val *)ctn->uni.ptr; /* tail */
            val = ctn_obj ? val->next->next : val->next;
            goto val_begin;
        }
    }
    if (val_type == YYJSON_TYPE_BOOL) {
        incr_len(16);
        cur = write_bool(cur, unsafe_yyjson_get_bool(val));
        cur++;
        goto val_end;
    }
    if (val_type == YYJSON_TYPE_NULL) {
        incr_len(16);
        cur = write_null(cur);
        cur++;
        goto val_end;
    }
    if (val_type == YYJSON_TYPE_RAW) {
        str_len = unsafe_yyjson_get_len(val);
        str_ptr = (const u8 *)unsafe_yyjson_get_str(val);
        check_str_len(str_len);
        incr_len(str_len + 2);
        cur = write_raw(cur, str_ptr, str_len);
        *cur++ = ',';
        goto val_end;
    }
    goto fail_type;

val_end:
    ctn_len--;
    if (unlikely(ctn_len == 0)) goto ctn_end;
    val = val->next;
    goto val_begin;

ctn_end:
    cur--;
    *cur++ = (u8)(']' | ((u8)ctn_obj << 5));
    *cur++ = ',';
    if (unlikely((u8 *)ctx >= end)) goto doc_end;
    val = ctn->next;
    yyjson_mut_write_ctx_get(ctx++, &ctn, &ctn_len, &ctn_obj);
    ctn_len--;
    if (likely(ctn_len > 0)) {
        goto val_begin;
    } else {
        goto ctn_end;
    }

doc_end:
    if (newline) {
        incr_len(2);
        *(cur - 1) = '\n';
        cur++;
    }
    *--cur = '\0';
    *dat_len = (usize)(cur - hdr);
    err->code = YYJSON_WRITE_SUCCESS;
    err->msg = NULL;
    return hdr;

fail_alloc: return_err(MEMORY_ALLOCATION, MSG_MALLOC);
fail_type:  return_err(INVALID_VALUE_TYPE, MSG_ERR_TYPE);
fail_num:   return_err(NAN_OR_INF, MSG_INF_NAN);
fail_str:   return_err(INVALID_STRING, MSG_ERR_UTF8);

#undef return_err
#undef incr_len
#undef check_str_len
}

/** Write JSON document pretty.
    The root of this document should be a non-empty container. */
static_inline u8 *yyjson_mut_write_pretty(const yyjson_mut_val *root,
                                          usize estimated_val_num,
                                          yyjson_write_flag flg,
                                          yyjson_alc alc,
                                          usize *dat_len,
                                          yyjson_write_err *err) {

#define return_err(_code, _msg) do { \
    *dat_len = 0; \
    err->code = YYJSON_WRITE_ERROR_##_code; \
    err->msg = _msg; \
    if (hdr) alc.free(alc.ctx, hdr); \
    return NULL; \
} while (false)

#define incr_len(_len) do { \
    ext_len = (usize)(_len); \
    if (unlikely((u8 *)(cur + ext_len) >= (u8 *)ctx)) { \
        usize ctx_pos = (usize)((u8 *)ctx - hdr); \
        usize cur_pos = (usize)(cur - hdr); \
        ctx_len = (usize)(end - (u8 *)ctx); \
        alc_inc = yyjson_max(alc_len / 2, ext_len); \
        alc_inc = size_align_up(alc_inc, sizeof(yyjson_mut_write_ctx)); \
        if ((sizeof(usize) < 8) && size_add_is_overflow(alc_len, alc_inc)) \
            goto fail_alloc; \
        alc_len += alc_inc; \
        tmp = (u8 *)alc.realloc(alc.ctx, hdr, alc_len - alc_inc, alc_len); \
        if (unlikely(!tmp)) goto fail_alloc; \
        ctx_tmp = (yyjson_mut_write_ctx *)(void *)(tmp + (alc_len - ctx_len)); \
        memmove((void *)ctx_tmp, (void *)(tmp + ctx_pos), ctx_len); \
        ctx = ctx_tmp; \
        cur = tmp + cur_pos; \
        end = tmp + alc_len; \
        hdr = tmp; \
    } \
} while (false)

#define check_str_len(_len) do { \
    if ((sizeof(usize) < 8) && (_len >= (USIZE_MAX - 16) / 6)) \
        goto fail_alloc; \
} while (false)

    yyjson_mut_val *val, *ctn;
    yyjson_type val_type;
    usize ctn_len, ctn_len_tmp;
    bool ctn_obj, ctn_obj_tmp, is_key, no_indent;
    u8 *hdr, *cur, *end, *tmp;
    yyjson_mut_write_ctx *ctx, *ctx_tmp;
    usize alc_len, alc_inc, ctx_len, ext_len, str_len, level;
    const u8 *str_ptr;
    const char_enc_type *enc_table = get_enc_table_with_flag(flg);
    bool cpy = (enc_table == enc_table_cpy);
    bool esc = has_write_flag(ESCAPE_UNICODE) != 0;
    bool inv = has_write_flag(ALLOW_INVALID_UNICODE) != 0;
    usize spaces = has_write_flag(PRETTY_TWO_SPACES) ? 2 : 4;
    bool newline = has_write_flag(NEWLINE_AT_END) != 0;

    alc_len = estimated_val_num * YYJSON_WRITER_ESTIMATED_PRETTY_RATIO + 64;
    alc_len = size_align_up(alc_len, sizeof(yyjson_mut_write_ctx));
    hdr = (u8 *)alc.malloc(alc.ctx, alc_len);
    if (!hdr) goto fail_alloc;
    cur = hdr;
    end = hdr + alc_len;
    ctx = (yyjson_mut_write_ctx *)(void *)end;

doc_begin:
    val = constcast(yyjson_mut_val *)root;
    val_type = unsafe_yyjson_get_type(val);
    ctn_obj = (val_type == YYJSON_TYPE_OBJ);
    ctn_len = unsafe_yyjson_get_len(val) << (u8)ctn_obj;
    *cur++ = (u8)('[' | ((u8)ctn_obj << 5));
    *cur++ = '\n';
    ctn = val;
    val = (yyjson_mut_val *)val->uni.ptr; /* tail */
    val = ctn_obj ? val->next->next : val->next;
    level = 1;

val_begin:
    val_type = unsafe_yyjson_get_type(val);
    if (val_type == YYJSON_TYPE_STR) {
        is_key = (bool)((u8)ctn_obj & (u8)~ctn_len);
        no_indent = (bool)((u8)ctn_obj & (u8)ctn_len);
        str_len = unsafe_yyjson_get_len(val);
        str_ptr = (const u8 *)unsafe_yyjson_get_str(val);
        check_str_len(str_len);
        incr_len(str_len * 6 + 16 + (no_indent ? 0 : level * 4));
        cur = write_indent(cur, no_indent ? 0 : level, spaces);
        if (likely(cpy) && unsafe_yyjson_get_subtype(val)) {
            cur = write_str_noesc(cur, str_ptr, str_len);
        } else {
            cur = write_str(cur, esc, inv, str_ptr, str_len, enc_table);
            if (unlikely(!cur)) goto fail_str;
        }
        *cur++ = is_key ? ':' : ',';
        *cur++ = is_key ? ' ' : '\n';
        goto val_end;
    }
    if (val_type == YYJSON_TYPE_NUM) {
        no_indent = (bool)((u8)ctn_obj & (u8)ctn_len);
        incr_len(FP_BUF_LEN + (no_indent ? 0 : level * 4));
        cur = write_indent(cur, no_indent ? 0 : level, spaces);
        cur = write_num(cur, (yyjson_val *)val, flg);
        if (unlikely(!cur)) goto fail_num;
        *cur++ = ',';
        *cur++ = '\n';
        goto val_end;
    }
    if ((val_type & (YYJSON_TYPE_ARR & YYJSON_TYPE_OBJ)) ==
                    (YYJSON_TYPE_ARR & YYJSON_TYPE_OBJ)) {
        no_indent = (bool)((u8)ctn_obj & (u8)ctn_len);
        ctn_len_tmp = unsafe_yyjson_get_len(val);
        ctn_obj_tmp = (val_type == YYJSON_TYPE_OBJ);
        if (unlikely(ctn_len_tmp == 0)) {
            /* write empty container */
            incr_len(16 + (no_indent ? 0 : level * 4));
            cur = write_indent(cur, no_indent ? 0 : level, spaces);
            *cur++ = (u8)('[' | ((u8)ctn_obj_tmp << 5));
            *cur++ = (u8)(']' | ((u8)ctn_obj_tmp << 5));
            *cur++ = ',';
            *cur++ = '\n';
            goto val_end;
        } else {
            /* push context, setup new container */
            incr_len(32 + (no_indent ? 0 : level * 4));
            yyjson_mut_write_ctx_set(--ctx, ctn, ctn_len, ctn_obj);
            ctn_len = ctn_len_tmp << (u8)ctn_obj_tmp;
            ctn_obj = ctn_obj_tmp;
            cur = write_indent(cur, no_indent ? 0 : level, spaces);
            level++;
            *cur++ = (u8)('[' | ((u8)ctn_obj << 5));
            *cur++ = '\n';
            ctn = val;
            val = (yyjson_mut_val *)ctn->uni.ptr; /* tail */
            val = ctn_obj ? val->next->next : val->next;
            goto val_begin;
        }
    }
    if (val_type == YYJSON_TYPE_BOOL) {
        no_indent = (bool)((u8)ctn_obj & (u8)ctn_len);
        incr_len(16 + (no_indent ? 0 : level * 4));
        cur = write_indent(cur, no_indent ? 0 : level, spaces);
        cur = write_bool(cur, unsafe_yyjson_get_bool(val));
        cur += 2;
        goto val_end;
    }
    if (val_type == YYJSON_TYPE_NULL) {
        no_indent = (bool)((u8)ctn_obj & (u8)ctn_len);
        incr_len(16 + (no_indent ? 0 : level * 4));
        cur = write_indent(cur, no_indent ? 0 : level, spaces);
        cur = write_null(cur);
        cur += 2;
        goto val_end;
    }
    if (val_type == YYJSON_TYPE_RAW) {
        no_indent = (bool)((u8)ctn_obj & (u8)ctn_len);
        str_len = unsafe_yyjson_get_len(val);
        str_ptr = (const u8 *)unsafe_yyjson_get_str(val);
        check_str_len(str_len);
        incr_len(str_len + 3 + (no_indent ? 0 : level * 4));
        cur = write_indent(cur, no_indent ? 0 : level, spaces);
        cur = write_raw(cur, str_ptr, str_len);
        *cur++ = ',';
        *cur++ = '\n';
        goto val_end;
    }
    goto fail_type;

val_end:
    ctn_len--;
    if (unlikely(ctn_len == 0)) goto ctn_end;
    val = val->next;
    goto val_begin;

ctn_end:
    cur -= 2;
    *cur++ = '\n';
    incr_len(level * 4);
    cur = write_indent(cur, --level, spaces);
    *cur++ = (u8)(']' | ((u8)ctn_obj << 5));
    if (unlikely((u8 *)ctx >= end)) goto doc_end;
    val = ctn->next;
    yyjson_mut_write_ctx_get(ctx++, &ctn, &ctn_len, &ctn_obj);
    ctn_len--;
    *cur++ = ',';
    *cur++ = '\n';
    if (likely(ctn_len > 0)) {
        goto val_begin;
    } else {
        goto ctn_end;
    }

doc_end:
    if (newline) {
        incr_len(2);
        *cur++ = '\n';
    }
    *cur = '\0';
    *dat_len = (usize)(cur - hdr);
    err->code = YYJSON_WRITE_SUCCESS;
    err->msg = NULL;
    return hdr;

fail_alloc: return_err(MEMORY_ALLOCATION, MSG_MALLOC);
fail_type:  return_err(INVALID_VALUE_TYPE, MSG_ERR_TYPE);
fail_num:   return_err(NAN_OR_INF, MSG_INF_NAN);
fail_str:   return_err(INVALID_STRING, MSG_ERR_UTF8);

#undef return_err
#undef incr_len
#undef check_str_len
}

static char *yyjson_mut_write_opts_impl(const yyjson_mut_val *val,
                                        usize estimated_val_num,
                                        yyjson_write_flag flg,
                                        const yyjson_alc *alc_ptr,
                                        usize *dat_len,
                                        yyjson_write_err *err) {
    yyjson_write_err dummy_err;
    usize dummy_dat_len;
    yyjson_alc alc = alc_ptr ? *alc_ptr : YYJSON_DEFAULT_ALC;
    yyjson_mut_val *root = constcast(yyjson_mut_val *)val;

    err = err ? err : &dummy_err;
    dat_len = dat_len ? dat_len : &dummy_dat_len;

    if (unlikely(!root)) {
        *dat_len = 0;
        err->msg = "input JSON is NULL";
        err->code = YYJSON_WRITE_ERROR_INVALID_PARAMETER;
        return NULL;
    }

    if (!unsafe_yyjson_is_ctn(root) || unsafe_yyjson_get_len(root) == 0) {
        return (char *)yyjson_mut_write_single(root, flg, alc, dat_len, err);
    } else if (flg & (YYJSON_WRITE_PRETTY | YYJSON_WRITE_PRETTY_TWO_SPACES)) {
        return (char *)yyjson_mut_write_pretty(root, estimated_val_num,
                                               flg, alc, dat_len, err);
    } else {
        return (char *)yyjson_mut_write_minify(root, estimated_val_num,
                                               flg, alc, dat_len, err);
    }
}

char *yyjson_mut_val_write_opts(const yyjson_mut_val *val,
                                yyjson_write_flag flg,
                                const yyjson_alc *alc_ptr,
                                usize *dat_len,
                                yyjson_write_err *err) {
    return yyjson_mut_write_opts_impl(val, 0, flg, alc_ptr, dat_len, err);
}

char *yyjson_mut_write_opts(const yyjson_mut_doc *doc,
                            yyjson_write_flag flg,
                            const yyjson_alc *alc_ptr,
                            usize *dat_len,
                            yyjson_write_err *err) {
    yyjson_mut_val *root;
    usize estimated_val_num;
    if (likely(doc)) {
        root = doc->root;
        estimated_val_num = yyjson_mut_doc_estimated_val_num(doc);
    } else {
        root = NULL;
        estimated_val_num = 0;
    }
    return yyjson_mut_write_opts_impl(root, estimated_val_num,
                                      flg, alc_ptr, dat_len, err);
}

bool yyjson_mut_val_write_file(const char *path,
                               const yyjson_mut_val *val,
                               yyjson_write_flag flg,
                               const yyjson_alc *alc_ptr,
                               yyjson_write_err *err) {
    yyjson_write_err dummy_err;
    yyjson_alc alc = alc_ptr ? *alc_ptr : YYJSON_DEFAULT_ALC;
    u8 *dat;
    usize dat_len = 0;
    yyjson_mut_val *root = constcast(yyjson_mut_val *)val;
    bool suc;

    err = err ? err : &dummy_err;
    if (unlikely(!path || !*path)) {
        err->msg = "input path is invalid";
        err->code = YYJSON_WRITE_ERROR_INVALID_PARAMETER;
        return false;
    }

    dat = (u8 *)yyjson_mut_val_write_opts(root, flg, &alc, &dat_len, err);
    if (unlikely(!dat)) return false;
    suc = write_dat_to_file(path, dat, dat_len, err);
    alc.free(alc.ctx, dat);
    return suc;
}

bool yyjson_mut_val_write_fp(FILE *fp,
                             const yyjson_mut_val *val,
                             yyjson_write_flag flg,
                             const yyjson_alc *alc_ptr,
                             yyjson_write_err *err) {
    yyjson_write_err dummy_err;
    yyjson_alc alc = alc_ptr ? *alc_ptr : YYJSON_DEFAULT_ALC;
    u8 *dat;
    usize dat_len = 0;
    yyjson_mut_val *root = constcast(yyjson_mut_val *)val;
    bool suc;

    err = err ? err : &dummy_err;
    if (unlikely(!fp)) {
        err->msg = "input fp is invalid";
        err->code = YYJSON_WRITE_ERROR_INVALID_PARAMETER;
        return false;
    }

    dat = (u8 *)yyjson_mut_val_write_opts(root, flg, &alc, &dat_len, err);
    if (unlikely(!dat)) return false;
    suc = write_dat_to_fp(fp, dat, dat_len, err);
    alc.free(alc.ctx, dat);
    return suc;
}

bool yyjson_mut_write_file(const char *path,
                           const yyjson_mut_doc *doc,
                           yyjson_write_flag flg,
                           const yyjson_alc *alc_ptr,
                           yyjson_write_err *err) {
    yyjson_mut_val *root = doc ? doc->root : NULL;
    return yyjson_mut_val_write_file(path, root, flg, alc_ptr, err);
}

bool yyjson_mut_write_fp(FILE *fp,
                         const yyjson_mut_doc *doc,
                         yyjson_write_flag flg,
                         const yyjson_alc *alc_ptr,
                         yyjson_write_err *err) {
    yyjson_mut_val *root = doc ? doc->root : NULL;
    return yyjson_mut_val_write_fp(fp, root, flg, alc_ptr, err);
}

#endif /* YYJSON_DISABLE_WRITER */
