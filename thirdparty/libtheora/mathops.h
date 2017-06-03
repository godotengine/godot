#if !defined(_mathops_H)
# define _mathops_H (1)
# include <ogg/ogg.h>

# ifdef __GNUC_PREREQ
#  if __GNUC_PREREQ(3,4)
#   include <limits.h>
/*Note the casts to (int) below: this prevents OC_CLZ{32|64}_OFFS from
   "upgrading" the type of an entire expression to an (unsigned) size_t.*/
#   if INT_MAX>=2147483647
#    define OC_CLZ32_OFFS ((int)sizeof(unsigned)*CHAR_BIT)
#    define OC_CLZ32(_x) (__builtin_clz(_x))
#   elif LONG_MAX>=2147483647L
#    define OC_CLZ32_OFFS ((int)sizeof(unsigned long)*CHAR_BIT)
#    define OC_CLZ32(_x) (__builtin_clzl(_x))
#   endif
#   if INT_MAX>=9223372036854775807LL
#    define OC_CLZ64_OFFS ((int)sizeof(unsigned)*CHAR_BIT)
#    define OC_CLZ64(_x) (__builtin_clz(_x))
#   elif LONG_MAX>=9223372036854775807LL
#    define OC_CLZ64_OFFS ((int)sizeof(unsigned long)*CHAR_BIT)
#    define OC_CLZ64(_x) (__builtin_clzl(_x))
#   elif LLONG_MAX>=9223372036854775807LL|| \
     __LONG_LONG_MAX__>=9223372036854775807LL
#    define OC_CLZ64_OFFS ((int)sizeof(unsigned long long)*CHAR_BIT)
#    define OC_CLZ64(_x) (__builtin_clzll(_x))
#   endif
#  endif
# endif



/**
 * oc_ilog32 - Integer binary logarithm of a 32-bit value.
 * @_v: A 32-bit value.
 * Returns floor(log2(_v))+1, or 0 if _v==0.
 * This is the number of bits that would be required to represent _v in two's
 *  complement notation with all of the leading zeros stripped.
 * The OC_ILOG_32() or OC_ILOGNZ_32() macros may be able to use a builtin
 *  function instead, which should be faster.
 */
int oc_ilog32(ogg_uint32_t _v);
/**
 * oc_ilog64 - Integer binary logarithm of a 64-bit value.
 * @_v: A 64-bit value.
 * Returns floor(log2(_v))+1, or 0 if _v==0.
 * This is the number of bits that would be required to represent _v in two's
 *  complement notation with all of the leading zeros stripped.
 * The OC_ILOG_64() or OC_ILOGNZ_64() macros may be able to use a builtin
 *  function instead, which should be faster.
 */
int oc_ilog64(ogg_int64_t _v);


# if defined(OC_CLZ32)
/**
 * OC_ILOGNZ_32 - Integer binary logarithm of a non-zero 32-bit value.
 * @_v: A non-zero 32-bit value.
 * Returns floor(log2(_v))+1.
 * This is the number of bits that would be required to represent _v in two's
 *  complement notation with all of the leading zeros stripped.
 * If _v is zero, the return value is undefined; use OC_ILOG_32() instead.
 */
#  define OC_ILOGNZ_32(_v) (OC_CLZ32_OFFS-OC_CLZ32(_v))
/**
 * OC_ILOG_32 - Integer binary logarithm of a 32-bit value.
 * @_v: A 32-bit value.
 * Returns floor(log2(_v))+1, or 0 if _v==0.
 * This is the number of bits that would be required to represent _v in two's
 *  complement notation with all of the leading zeros stripped.
 */
#  define OC_ILOG_32(_v)   (OC_ILOGNZ_32(_v)&-!!(_v))
# else
#  define OC_ILOGNZ_32(_v) (oc_ilog32(_v))
#  define OC_ILOG_32(_v)   (oc_ilog32(_v))
# endif

# if defined(CLZ64)
/**
 * OC_ILOGNZ_64 - Integer binary logarithm of a non-zero 64-bit value.
 * @_v: A non-zero 64-bit value.
 * Returns floor(log2(_v))+1.
 * This is the number of bits that would be required to represent _v in two's
 *  complement notation with all of the leading zeros stripped.
 * If _v is zero, the return value is undefined; use OC_ILOG_64() instead.
 */
#  define OC_ILOGNZ_64(_v) (CLZ64_OFFS-CLZ64(_v))
/**
 * OC_ILOG_64 - Integer binary logarithm of a 64-bit value.
 * @_v: A 64-bit value.
 * Returns floor(log2(_v))+1, or 0 if _v==0.
 * This is the number of bits that would be required to represent _v in two's
 *  complement notation with all of the leading zeros stripped.
 */
#  define OC_ILOG_64(_v)   (OC_ILOGNZ_64(_v)&-!!(_v))
# else
#  define OC_ILOGNZ_64(_v) (oc_ilog64(_v))
#  define OC_ILOG_64(_v)   (oc_ilog64(_v))
# endif

# define OC_STATIC_ILOG0(_v) (!!(_v))
# define OC_STATIC_ILOG1(_v) (((_v)&0x2)?2:OC_STATIC_ILOG0(_v))
# define OC_STATIC_ILOG2(_v) \
 (((_v)&0xC)?2+OC_STATIC_ILOG1((_v)>>2):OC_STATIC_ILOG1(_v))
# define OC_STATIC_ILOG3(_v) \
 (((_v)&0xF0)?4+OC_STATIC_ILOG2((_v)>>4):OC_STATIC_ILOG2(_v))
# define OC_STATIC_ILOG4(_v) \
 (((_v)&0xFF00)?8+OC_STATIC_ILOG3((_v)>>8):OC_STATIC_ILOG3(_v))
# define OC_STATIC_ILOG5(_v) \
 (((_v)&0xFFFF0000)?16+OC_STATIC_ILOG4((_v)>>16):OC_STATIC_ILOG4(_v))
# define OC_STATIC_ILOG6(_v) \
 (((_v)&0xFFFFFFFF00000000ULL)?32+OC_STATIC_ILOG5((_v)>>32):OC_STATIC_ILOG5(_v))
/**
 * OC_STATIC_ILOG_32 - The integer logarithm of an (unsigned, 32-bit) constant.
 * @_v: A non-negative 32-bit constant.
 * Returns floor(log2(_v))+1, or 0 if _v==0.
 * This is the number of bits that would be required to represent _v in two's
 *  complement notation with all of the leading zeros stripped.
 * This macro is suitable for evaluation at compile time, but it should not be
 *  used on values that can change at runtime, as it operates via exhaustive
 *  search.
 */
# define OC_STATIC_ILOG_32(_v) (OC_STATIC_ILOG5((ogg_uint32_t)(_v)))
/**
 * OC_STATIC_ILOG_64 - The integer logarithm of an (unsigned, 64-bit) constant.
 * @_v: A non-negative 64-bit constant.
 * Returns floor(log2(_v))+1, or 0 if _v==0.
 * This is the number of bits that would be required to represent _v in two's
 *  complement notation with all of the leading zeros stripped.
 * This macro is suitable for evaluation at compile time, but it should not be
 *  used on values that can change at runtime, as it operates via exhaustive
 *  search.
 */
# define OC_STATIC_ILOG_64(_v) (OC_STATIC_ILOG6((ogg_int64_t)(_v)))

#define OC_Q57(_v) ((ogg_int64_t)(_v)<<57)

ogg_int64_t oc_bexp64(ogg_int64_t _z);
ogg_int64_t oc_blog64(ogg_int64_t _w);

#endif
