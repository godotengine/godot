#ifndef CSNAPPY_COMPAT_H

/* This file was added to Sereal to attempt some MSVC compatibility,
 * but is at best a band-aid. And done without a lot of experience
 * in whatever subset of C99 MSVC supports.
 */

#ifndef INLINE
#   if defined(_MSC_VER)
#     define INLINE __inline
#   else
#     define INLINE inline
#   endif
#endif

#endif
