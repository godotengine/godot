/***
 * libccd
 * ---------------------------------
 * Copyright (c)2010 Daniel Fiser <danfis@danfis.cz>
 *
 *
 *  This file is part of libccd.
 *
 *  Distributed under the OSI-approved BSD License (the "License");
 *  see accompanying file BDS-LICENSE for details or see
 *  <http://www.opensource.org/licenses/bsd-license.php>.
 *
 *  This software is distributed WITHOUT ANY WARRANTY; without even the
 *  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the License for more information.
 */

#ifndef __CCD_COMPILER_H__
#define __CCD_COMPILER_H__

#include <stddef.h>

#define ccd_offsetof(TYPE, MEMBER) offsetof(TYPE, MEMBER)

#define ccd_container_of(ptr, type, member) \
    (type *)( (char *)ptr - ccd_offsetof(type, member))

/**
 * Marks inline function.
 */
#ifdef __GNUC__
# define _ccd_inline static inline __attribute__((always_inline))
#else /* __GNUC__ */
# define _ccd_inline static __inline
#endif /* __GNUC__ */


/**
 * __prefetch(x)  - prefetches the cacheline at "x" for read
 * __prefetchw(x) - prefetches the cacheline at "x" for write
 */
#ifdef __GNUC__
# define _ccd_prefetch(x) __builtin_prefetch(x)
# define _ccd_prefetchw(x) __builtin_prefetch(x,1)
#else /* __GNUC__ */
# define _ccd_prefetch(x) ((void)0)
# define _ccd_prefetchw(x) ((void)0)
#endif /* __GNUC__ */


#ifdef __ICC
// disable unused parameter warning
# pragma warning(disable:869)
// disable annoying "operands are evaluated in unspecified order" warning
# pragma warning(disable:981)
#endif /* __ICC */

#ifdef _MSC_VER
// disable unsafe function warning
# define _CRT_SECURE_NO_WARNINGS
#endif /* _MSC_VER */

#endif /* __CCD_COMPILER_H__ */

