#ifndef foomemoryhfoo
#define foomemoryhfoo

/***
  This file is part of PulseAudio.

  Copyright 2004-2006 Lennart Poettering

  PulseAudio is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published
  by the Free Software Foundation; either version 2.1 of the License,
  or (at your option) any later version.

  PulseAudio is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with PulseAudio; if not, see <http://www.gnu.org/licenses/>.
***/

#include <sys/types.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>

#include <pulse/cdecl.h>
#include <pulse/gccmacro.h>
#include <pulse/version.h>

/** \file
 * Memory allocation functions.
 */

PA_C_DECL_BEGIN

/** Allocate the specified number of bytes, just like malloc() does. However, in case of OOM, terminate */
void* pa_xmalloc(size_t l) PA_GCC_MALLOC PA_GCC_ALLOC_SIZE(1);

/** Same as pa_xmalloc(), but initialize allocated memory to 0 */
void *pa_xmalloc0(size_t l) PA_GCC_MALLOC PA_GCC_ALLOC_SIZE(1);

/**  The combination of pa_xmalloc() and realloc() */
void *pa_xrealloc(void *ptr, size_t size) PA_GCC_ALLOC_SIZE(2);

/** Free allocated memory */
void pa_xfree(void *p);

/** Duplicate the specified string, allocating memory with pa_xmalloc() */
char *pa_xstrdup(const char *s) PA_GCC_MALLOC;

/** Duplicate the specified string, but truncate after l characters */
char *pa_xstrndup(const char *s, size_t l) PA_GCC_MALLOC;

/** Duplicate the specified memory block */
void* pa_xmemdup(const void *p, size_t l) PA_GCC_MALLOC PA_GCC_ALLOC_SIZE(2);

/** Internal helper for pa_xnew() */
static void* _pa_xnew_internal(size_t n, size_t k) PA_GCC_MALLOC PA_GCC_ALLOC_SIZE2(1,2);

static inline void* _pa_xnew_internal(size_t n, size_t k) {
    assert(n < INT_MAX/k);
    return pa_xmalloc(n*k);
}

/** Allocate n new structures of the specified type. */
#define pa_xnew(type, n) ((type*) _pa_xnew_internal((n), sizeof(type)))

/** Internal helper for pa_xnew0() */
static void* _pa_xnew0_internal(size_t n, size_t k) PA_GCC_MALLOC PA_GCC_ALLOC_SIZE2(1,2);

static inline void* _pa_xnew0_internal(size_t n, size_t k) {
    assert(n < INT_MAX/k);
    return pa_xmalloc0(n*k);
}

/** Same as pa_xnew() but set the memory to zero */
#define pa_xnew0(type, n) ((type*) _pa_xnew0_internal((n), sizeof(type)))

/** Internal helper for pa_xnew0() */
static void* _pa_xnewdup_internal(const void *p, size_t n, size_t k) PA_GCC_MALLOC PA_GCC_ALLOC_SIZE2(2,3);

static inline void* _pa_xnewdup_internal(const void *p, size_t n, size_t k) {
    assert(n < INT_MAX/k);
    return pa_xmemdup(p, n*k);
}

/** Same as pa_xnew() but duplicate the specified data */
#define pa_xnewdup(type, p, n) ((type*) _pa_xnewdup_internal((p), (n), sizeof(type)))

/** Internal helper for pa_xrenew() */
static void* _pa_xrenew_internal(void *p, size_t n, size_t k) PA_GCC_MALLOC PA_GCC_ALLOC_SIZE2(2,3);

static inline void* _pa_xrenew_internal(void *p, size_t n, size_t k) {
    assert(n < INT_MAX/k);
    return pa_xrealloc(p, n*k);
}

/** Reallocate n new structures of the specified type. */
#define pa_xrenew(type, p, n) ((type*) _pa_xrenew_internal(p, (n), sizeof(type)))

PA_C_DECL_END

#endif
