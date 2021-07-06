/*-----------------------------------------------------------------------------
 * MurmurHash3 was written by Austin Appleby, and is placed in the public
 * domain.
 *
 * This implementation was written by Shane Day, and is also public domain.
 *
 * This is a portable ANSI C implementation of MurmurHash3_x86_32 (Murmur3A)
 * with support for progressive processing.
 */

/* ------------------------------------------------------------------------- */
/* Determine what native type to use for uint32_t */

/* We can't use the name 'uint32_t' here because it will conflict with
 * any version provided by the system headers or application. */

/* First look for special cases */
#if defined(_MSC_VER)
#    define MH_UINT32 unsigned long
#endif

/* If the compiler says it's C99 then take its word for it */
#if !defined(MH_UINT32) && (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L)
#    include <stdint.h>
#    define MH_UINT32 uint32_t
#endif

/* Otherwise try testing against max value macros from limit.h */
#if !defined(MH_UINT32)
#    include <limits.h>
#    if (USHRT_MAX == 0xffffffffUL)
#        define MH_UINT32 unsigned short
#    elif (UINT_MAX == 0xffffffffUL)
#        define MH_UINT32 unsigned int
#    elif (ULONG_MAX == 0xffffffffUL)
#        define MH_UINT32 unsigned long
#    endif
#endif

#if !defined(MH_UINT32)
#    error Unable to determine type name for unsigned 32-bit int
#endif

/* I'm yet to work on a platform where 'unsigned char' is not 8 bits */
#define MH_UINT8 unsigned char

/* ------------------------------------------------------------------------- */
/* Prototypes */

namespace angle
{
void PMurHash32_Process(MH_UINT32 *ph1, MH_UINT32 *pcarry, const void *key, int len);
MH_UINT32 PMurHash32_Result(MH_UINT32 h1, MH_UINT32 carry, MH_UINT32 total_length);
MH_UINT32 PMurHash32(MH_UINT32 seed, const void *key, int len);

void PMurHash32_test(const void *key, int len, MH_UINT32 seed, void *out);
}  // namespace angle
