/**************************************************************************
 *
 * Copyright 2008 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/


/**
 * Math utilities and approximations for common math functions.
 * Reduced precision is usually acceptable in shaders...
 *
 * "fast" is used in the names of functions which are low-precision,
 * or at least lower-precision than the normal C lib functions.
 */


#ifndef U_MATH_H
#define U_MATH_H


#include "c99_compat.h"
#include <assert.h>
#include <float.h>
#include <stdarg.h>
#include <math.h>

#include "bitscan.h"
#include "u_endian.h" /* for UTIL_ARCH_BIG_ENDIAN */
#include "util/detect_cc.h"
#include "util/detect_arch.h"

#ifdef __cplusplus
extern "C" {
#endif


#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif


/**
 * Initialize math module.  This should be called before using any
 * other functions in this module.
 */
extern void
util_init_math(void);


union fi {
   float f;
   int32_t i;
   uint32_t ui;
};


union di {
   double d;
   int64_t i;
   uint64_t ui;
};


/**
 * Extract the IEEE float32 exponent.
 */
static inline signed
util_get_float32_exponent(float x)
{
   union fi f;

   f.f = x;

   return ((f.ui >> 23) & 0xff) - 127;
}


#define LOG2_TABLE_SIZE_LOG2 8
#define LOG2_TABLE_SCALE (1 << LOG2_TABLE_SIZE_LOG2)
#define LOG2_TABLE_SIZE (LOG2_TABLE_SCALE + 1)
extern float log2_table[LOG2_TABLE_SIZE];


/**
 * Fast approximation to log2(x).
 */
static inline float
util_fast_log2(float x)
{
   union fi num;
   float epart, mpart;
   num.f = x;
   epart = (float)(((num.i & 0x7f800000) >> 23) - 127);
   /* mpart = log2_table[mantissa*LOG2_TABLE_SCALE + 0.5] */
   mpart = log2_table[((num.i & 0x007fffff) + (1 << (22 - LOG2_TABLE_SIZE_LOG2))) >> (23 - LOG2_TABLE_SIZE_LOG2)];
   return epart + mpart;
}


/**
 * Floor(x), returned as int.
 */
static inline int
util_ifloor(float f)
{
#if defined(USE_X86_ASM) && defined(__GNUC__) && defined(__i386__)
   /*
    * IEEE floor for computers that round to nearest or even.
    * 'f' must be between -4194304 and 4194303.
    * This floor operation is done by "(iround(f + .5) + iround(f - .5)) >> 1",
    * but uses some IEEE specific tricks for better speed.
    * Contributed by Josh Vanderhoof
    */
   int ai, bi;
   double af, bf;
   af = (3 << 22) + 0.5 + (double)f;
   bf = (3 << 22) + 0.5 - (double)f;
   /* GCC generates an extra fstp/fld without this. */
   __asm__ ("fstps %0" : "=m" (ai) : "t" (af) : "st");
   __asm__ ("fstps %0" : "=m" (bi) : "t" (bf) : "st");
   return (ai - bi) >> 1;
#else
   int ai, bi;
   double af, bf;
   union fi u;
   af = (3 << 22) + 0.5 + (double) f;
   bf = (3 << 22) + 0.5 - (double) f;
   u.f = (float) af;  ai = u.i;
   u.f = (float) bf;  bi = u.i;
   return (ai - bi) >> 1;
#endif
}


/**
 * Round float to nearest int.
 */
static inline int
util_iround(float f)
{
#if DETECT_CC_GCC && DETECT_ARCH_X86
   int r;
   __asm__ ("fistpl %0" : "=m" (r) : "t" (f) : "st");
   return r;
#elif DETECT_CC_MSVC && DETECT_ARCH_X86
   int r;
   _asm {
      fld f
      fistp r
   }
   return r;
#else
   if (f >= 0.0f)
      return (int) (f + 0.5f);
   else
      return (int) (f - 0.5f);
#endif
}


/**
 * Approximate floating point comparison
 */
static inline bool
util_is_approx(float a, float b, float tol)
{
   return fabsf(b - a) <= tol;
}


/**
 * util_is_X_inf_or_nan = test if x is NaN or +/- Inf
 * util_is_X_nan        = test if x is NaN
 * util_X_inf_sign      = return +1 for +Inf, -1 for -Inf, or 0 for not Inf
 *
 * NaN can be checked with x != x, however this fails with the fast math flag
 **/


/**
 * Single-float
 */
static inline bool
util_is_inf_or_nan(float x)
{
   union fi tmp;
   tmp.f = x;
   return (tmp.ui & 0x7f800000) == 0x7f800000;
}


static inline bool
util_is_nan(float x)
{
   union fi tmp;
   tmp.f = x;
   return (tmp.ui & 0x7fffffff) > 0x7f800000;
}


static inline int
util_inf_sign(float x)
{
   union fi tmp;
   tmp.f = x;
   if ((tmp.ui & 0x7fffffff) != 0x7f800000) {
      return 0;
   }

   return (x < 0) ? -1 : 1;
}


/**
 * Double-float
 */
static inline bool
util_is_double_inf_or_nan(double x)
{
   union di tmp;
   tmp.d = x;
   return (tmp.ui & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL;
}


static inline bool
util_is_double_nan(double x)
{
   union di tmp;
   tmp.d = x;
   return (tmp.ui & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL;
}


static inline int
util_double_inf_sign(double x)
{
   union di tmp;
   tmp.d = x;
   if ((tmp.ui & 0x7fffffffffffffffULL) != 0x7ff0000000000000ULL) {
      return 0;
   }

   return (x < 0) ? -1 : 1;
}


/**
 * Half-float
 */
static inline bool
util_is_half_inf_or_nan(int16_t x)
{
   return (x & 0x7c00) == 0x7c00;
}


static inline bool
util_is_half_nan(int16_t x)
{
   return (x & 0x7fff) > 0x7c00;
}


static inline int
util_half_inf_sign(int16_t x)
{
   if ((x & 0x7fff) != 0x7c00) {
      return 0;
   }

   return (x < 0) ? -1 : 1;
}


/**
 * Return float bits.
 */
static inline unsigned
fui( float f )
{
   union fi fi;
   fi.f = f;
   return fi.ui;
}

static inline float
uif(uint32_t ui)
{
   union fi fi;
   fi.ui = ui;
   return fi.f;
}


/**
 * Convert uint8_t to float in [0, 1].
 */
static inline float
ubyte_to_float(uint8_t ub)
{
   return (float) ub * (1.0f / 255.0f);
}


/**
 * Convert float in [0,1] to uint8_t in [0,255] with clamping.
 */
static inline uint8_t
float_to_ubyte(float f)
{
   /* return 0 for NaN too */
   if (!(f > 0.0f)) {
      return (uint8_t) 0;
   }
   else if (f >= 1.0f) {
      return (uint8_t) 255;
   }
   else {
      union fi tmp;
      tmp.f = f;
      tmp.f = tmp.f * (255.0f/256.0f) + 32768.0f;
      return (uint8_t) tmp.i;
   }
}

/**
 * Convert uint16_t to float in [0, 1].
 */
static inline float
ushort_to_float(uint16_t us)
{
   return (float) us * (1.0f / 65535.0f);
}


/**
 * Convert float in [0,1] to uint16_t in [0,65535] with clamping.
 */
static inline uint16_t
float_to_ushort(float f)
{
   /* return 0 for NaN too */
   if (!(f > 0.0f)) {
      return (uint16_t) 0;
   }
   else if (f >= 1.0f) {
      return (uint16_t) 65535;
   }
   else {
      union fi tmp;
      tmp.f = f;
      tmp.f = tmp.f * (65535.0f/65536.0f) + 128.0f;
      return (uint16_t) tmp.i;
   }
}

static inline float
byte_to_float_tex(int8_t b)
{
   return (b == -128) ? -1.0F : b * 1.0F / 127.0F;
}

static inline int8_t
float_to_byte_tex(float f)
{
   return (int8_t) (127.0F * f);
}

/**
 * Calc log base 2
 */
static inline unsigned
util_logbase2(unsigned n)
{
#if defined(HAVE___BUILTIN_CLZ)
   return ((sizeof(unsigned) * 8 - 1) - __builtin_clz(n | 1));
#else
   unsigned pos = 0;
   if (n >= 1<<16) { n >>= 16; pos += 16; }
   if (n >= 1<< 8) { n >>=  8; pos +=  8; }
   if (n >= 1<< 4) { n >>=  4; pos +=  4; }
   if (n >= 1<< 2) { n >>=  2; pos +=  2; }
   if (n >= 1<< 1) {           pos +=  1; }
   return pos;
#endif
}

static inline uint64_t
util_logbase2_64(uint64_t n)
{
#if defined(HAVE___BUILTIN_CLZLL)
   return ((sizeof(uint64_t) * 8 - 1) - __builtin_clzll(n | 1));
#else
   uint64_t pos = 0ull;
   if (n >= 1ull<<32) { n >>= 32; pos += 32; }
   if (n >= 1ull<<16) { n >>= 16; pos += 16; }
   if (n >= 1ull<< 8) { n >>=  8; pos +=  8; }
   if (n >= 1ull<< 4) { n >>=  4; pos +=  4; }
   if (n >= 1ull<< 2) { n >>=  2; pos +=  2; }
   if (n >= 1ull<< 1) {           pos +=  1; }
   return pos;
#endif
}

/**
 * Returns the ceiling of log n base 2, and 0 when n == 0. Equivalently,
 * returns the smallest x such that n <= 2**x.
 */
static inline unsigned
util_logbase2_ceil(unsigned n)
{
   if (n <= 1)
      return 0;

   return 1 + util_logbase2(n - 1);
}

static inline uint64_t
util_logbase2_ceil64(uint64_t n)
{
   if (n <= 1)
      return 0;

   return 1ull + util_logbase2_64(n - 1);
}

/**
 * Returns the smallest power of two >= x
 */
static inline unsigned
util_next_power_of_two(unsigned x)
{
#if defined(HAVE___BUILTIN_CLZ)
   if (x <= 1)
       return 1;

   return (1 << ((sizeof(unsigned) * 8) - __builtin_clz(x - 1)));
#else
   unsigned val = x;

   if (x <= 1)
      return 1;

   if (util_is_power_of_two_or_zero(x))
      return x;

   val--;
   val = (val >> 1) | val;
   val = (val >> 2) | val;
   val = (val >> 4) | val;
   val = (val >> 8) | val;
   val = (val >> 16) | val;
   val++;
   return val;
#endif
}

static inline uint64_t
util_next_power_of_two64(uint64_t x)
{
#if defined(HAVE___BUILTIN_CLZLL)
   if (x <= 1)
       return 1;

   return (1ull << ((sizeof(uint64_t) * 8) - __builtin_clzll(x - 1)));
#else
   uint64_t val = x;

   if (x <= 1)
      return 1;

   if (util_is_power_of_two_or_zero64(x))
      return x;

   val--;
   val = (val >> 1)  | val;
   val = (val >> 2)  | val;
   val = (val >> 4)  | val;
   val = (val >> 8)  | val;
   val = (val >> 16) | val;
   val = (val >> 32) | val;
   val++;
   return val;
#endif
}

/**
 * Reverse bits in n
 * Algorithm taken from:
 * http://stackoverflow.com/questions/9144800/c-reverse-bits-in-unsigned-integer
 */
static inline unsigned
util_bitreverse(unsigned n)
{
    n = ((n >> 1) & 0x55555555u) | ((n & 0x55555555u) << 1);
    n = ((n >> 2) & 0x33333333u) | ((n & 0x33333333u) << 2);
    n = ((n >> 4) & 0x0f0f0f0fu) | ((n & 0x0f0f0f0fu) << 4);
    n = ((n >> 8) & 0x00ff00ffu) | ((n & 0x00ff00ffu) << 8);
    n = ((n >> 16) & 0xffffu) | ((n & 0xffffu) << 16);
    return n;
}

/**
 * Convert from little endian to CPU byte order.
 */

#if UTIL_ARCH_BIG_ENDIAN
#define util_le64_to_cpu(x) util_bswap64(x)
#define util_le32_to_cpu(x) util_bswap32(x)
#define util_le16_to_cpu(x) util_bswap16(x)
#else
#define util_le64_to_cpu(x) (x)
#define util_le32_to_cpu(x) (x)
#define util_le16_to_cpu(x) (x)
#endif

#define util_cpu_to_le64(x) util_le64_to_cpu(x)
#define util_cpu_to_le32(x) util_le32_to_cpu(x)
#define util_cpu_to_le16(x) util_le16_to_cpu(x)

/**
 * Reverse byte order of a 32 bit word.
 */
static inline uint32_t
util_bswap32(uint32_t n)
{
#if defined(HAVE___BUILTIN_BSWAP32)
   return __builtin_bswap32(n);
#else
   return (n >> 24) |
          ((n >> 8) & 0x0000ff00) |
          ((n << 8) & 0x00ff0000) |
          (n << 24);
#endif
}

/**
 * Reverse byte order of a 64bit word.
 */
static inline uint64_t
util_bswap64(uint64_t n)
{
#if defined(HAVE___BUILTIN_BSWAP64)
   return __builtin_bswap64(n);
#else
   return ((uint64_t)util_bswap32((uint32_t)n) << 32) |
          util_bswap32((n >> 32));
#endif
}


/**
 * Reverse byte order of a 16 bit word.
 */
static inline uint16_t
util_bswap16(uint16_t n)
{
   return (n >> 8) |
          (n << 8);
}

/**
 * Mask and sign-extend a number
 *
 * The bit at position `width - 1` is replicated to all the higher bits.
 * This makes no assumptions about the high bits of the value and will
 * overwrite them with the sign bit.
 */
static inline int64_t
util_mask_sign_extend(uint64_t val, unsigned width)
{
   assert(width > 0 && width <= 64);
   unsigned shift = 64 - width;
   return (int64_t)(val << shift) >> shift;
}

/**
 * Sign-extend a number
 *
 * The bit at position `width - 1` is replicated to all the higher bits.
 * This assumes and asserts that the value fits into `width` bits.
 */
static inline int64_t
util_sign_extend(uint64_t val, unsigned width)
{
   assert(width == 64 || val < (UINT64_C(1) << width));
   return util_mask_sign_extend(val, width);
}

static inline void*
util_memcpy_cpu_to_le32(void * restrict dest, const void * restrict src, size_t n)
{
#if UTIL_ARCH_BIG_ENDIAN
   size_t i, e;
   assert(n % 4 == 0);

   for (i = 0, e = n / 4; i < e; i++) {
      uint32_t * restrict d = (uint32_t* restrict)dest;
      const uint32_t * restrict s = (const uint32_t* restrict)src;
      d[i] = util_bswap32(s[i]);
   }
   return dest;
#else
   return memcpy(dest, src, n);
#endif
}

/**
 * Clamp X to [MIN, MAX].
 * This is a macro to allow float, int, uint, etc. types.
 * We arbitrarily turn NaN into MIN.
 */
#define CLAMP( X, MIN, MAX )  ( (X)>(MIN) ? ((X)>(MAX) ? (MAX) : (X)) : (MIN) )

/* Syntax sugar occuring frequently in graphics code */
#define SATURATE( X ) CLAMP(X, 0.0f, 1.0f)

#define MIN2( A, B )   ( (A)<(B) ? (A) : (B) )
#define MAX2( A, B )   ( (A)>(B) ? (A) : (B) )

#define MIN3( A, B, C ) ((A) < (B) ? MIN2(A, C) : MIN2(B, C))
#define MAX3( A, B, C ) ((A) > (B) ? MAX2(A, C) : MAX2(B, C))

#define MIN4( A, B, C, D ) ((A) < (B) ? MIN3(A, C, D) : MIN3(B, C, D))
#define MAX4( A, B, C, D ) ((A) > (B) ? MAX3(A, C, D) : MAX3(B, C, D))


/**
 * Align a value up to an alignment value
 *
 * If \c value is not already aligned to the requested alignment value, it
 * will be rounded up.
 *
 * \param value  Value to be rounded
 * \param alignment  Alignment value to be used.  This must be a power of two.
 *
 * \sa ROUND_DOWN_TO()
 */

#if defined(ALIGN)
#undef ALIGN
#endif
static inline uintptr_t
ALIGN(uintptr_t value, int32_t alignment)
{
   assert(util_is_power_of_two_nonzero(alignment));
   return (((value) + (alignment) - 1) & ~((alignment) - 1));
}

/**
 * Like ALIGN(), but works with a non-power-of-two alignment.
 */
static inline uintptr_t
ALIGN_NPOT(uintptr_t value, int32_t alignment)
{
   assert(alignment > 0);
   return (value + alignment - 1) / alignment * alignment;
}

/**
 * Align a value down to an alignment value
 *
 * If \c value is not already aligned to the requested alignment value, it
 * will be rounded down.
 *
 * \param value  Value to be rounded
 * \param alignment  Alignment value to be used.  This must be a power of two.
 *
 * \sa ALIGN()
 */
static inline uint64_t
ROUND_DOWN_TO(uint64_t value, int32_t alignment)
{
   assert(util_is_power_of_two_nonzero(alignment));
   return ((value) & ~(alignment - 1));
}

/**
 * Align a value, only works pot alignemnts.
 */
static inline int
align(int value, int alignment)
{
   return (value + alignment - 1) & ~(alignment - 1);
}

static inline uint64_t
align64(uint64_t value, unsigned alignment)
{
   return (value + alignment - 1) & ~((uint64_t)alignment - 1);
}

/**
 * Works like align but on npot alignments.
 */
static inline size_t
util_align_npot(size_t value, size_t alignment)
{
   if (value % alignment)
      return value + (alignment - (value % alignment));
   return value;
}

static inline unsigned
u_minify(unsigned value, unsigned levels)
{
    return MAX2(1, value >> levels);
}

#ifndef COPY_4V
#define COPY_4V( DST, SRC )         \
do {                                \
   (DST)[0] = (SRC)[0];             \
   (DST)[1] = (SRC)[1];             \
   (DST)[2] = (SRC)[2];             \
   (DST)[3] = (SRC)[3];             \
} while (0)
#endif


#ifndef COPY_4FV
#define COPY_4FV( DST, SRC )  COPY_4V(DST, SRC)
#endif


#ifndef ASSIGN_4V
#define ASSIGN_4V( DST, V0, V1, V2, V3 ) \
do {                                     \
   (DST)[0] = (V0);                      \
   (DST)[1] = (V1);                      \
   (DST)[2] = (V2);                      \
   (DST)[3] = (V3);                      \
} while (0)
#endif


static inline uint32_t
util_unsigned_fixed(float value, unsigned frac_bits)
{
   return value < 0 ? 0 : (uint32_t)(value * (1<<frac_bits));
}

static inline int32_t
util_signed_fixed(float value, unsigned frac_bits)
{
   return (int32_t)(value * (1<<frac_bits));
}

unsigned
util_fpstate_get(void);
unsigned
util_fpstate_set_denorms_to_zero(unsigned current_fpstate);
void
util_fpstate_set(unsigned fpstate);

/**
 * For indexed draw calls, return true if the vertex count to be drawn is
 * much lower than the vertex count that has to be uploaded, meaning
 * that the driver should flatten indices instead of trying to upload
 * a too big range.
 *
 * This is used by vertex upload code in u_vbuf and glthread.
 */
static inline bool
util_is_vbo_upload_ratio_too_large(unsigned draw_vertex_count,
                                   unsigned upload_vertex_count)
{
   if (upload_vertex_count > 256)
      return upload_vertex_count > draw_vertex_count * 4;
   else if (upload_vertex_count > 64)
      return upload_vertex_count > draw_vertex_count * 8;
   else
      return upload_vertex_count > draw_vertex_count * 16;
}

bool util_invert_mat4x4(float *out, const float *m);

/* Quantize the lod bias value to reduce the number of sampler state
 * variants in gallium because apps use it for smooth mipmap transitions,
 * thrashing cso_cache and degrading performance.
 *
 * This quantization matches the AMD hw specification, so having more
 * precision would have no effect anyway.
 */
static inline float
util_quantize_lod_bias(float lod)
{
   lod = CLAMP(lod, -16, 16);
   return roundf(lod * 256) / 256;
}

/**
 * Adds two unsigned integers and if the addition
 * overflows then clamp it to ~0U.
 */
static inline unsigned
util_clamped_uadd(unsigned a, unsigned b)
{
   unsigned res = a + b;
   if (res < a) {
      res = ~0U;
   }
   return res;
}

#ifdef __cplusplus
}
#endif

#endif /* U_MATH_H */
