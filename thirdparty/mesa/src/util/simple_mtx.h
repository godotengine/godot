/*
 * Copyright © 2015 Intel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef _SIMPLE_MTX_H
#define _SIMPLE_MTX_H

#include "util/futex.h"
#include "util/macros.h"
#include "util/u_call_once.h"
#include "u_atomic.h"

#if UTIL_FUTEX_SUPPORTED
#if defined(HAVE_VALGRIND) && !defined(NDEBUG)
#  include <valgrind.h>
#  include <helgrind.h>
#  define HG(x) x
#else
#  define HG(x)
#endif
#else /* !UTIL_FUTEX_SUPPORTED */
#  include "c11/threads.h"
#endif /* UTIL_FUTEX_SUPPORTED */

#ifdef __cplusplus
extern "C" {
#endif

#if UTIL_FUTEX_SUPPORTED

/* mtx_t - Fast, simple mutex
 *
 * While modern pthread mutexes are very fast (implemented using futex), they
 * still incur a call to an external DSO and overhead of the generality and
 * features of pthread mutexes.  Most mutexes in mesa only needs lock/unlock,
 * and the idea here is that we can inline the atomic operation and make the
 * fast case just two intructions.  Mutexes are subtle and finicky to
 * implement, so we carefully copy the implementation from Ulrich Dreppers
 * well-written and well-reviewed paper:
 *
 *   "Futexes Are Tricky"
 *   http://www.akkadia.org/drepper/futex.pdf
 *
 * We implement "mutex3", which gives us a mutex that has no syscalls on
 * uncontended lock or unlock.  Further, the uncontended case boils down to a
 * locked cmpxchg and an untaken branch, the uncontended unlock is just a
 * locked decr and an untaken branch.  We use __builtin_expect() to indicate
 * that contention is unlikely so that gcc will put the contention code out of
 * the main code flow.
 *
 * A fast mutex only supports lock/unlock, can't be recursive or used with
 * condition variables.
 */

typedef struct {
   uint32_t val;
} simple_mtx_t;

#define SIMPLE_MTX_INITIALIZER { 0 }

#define _SIMPLE_MTX_INVALID_VALUE 0xd0d0d0d0

static inline void
simple_mtx_init(simple_mtx_t *mtx, ASSERTED int type)
{
   assert(type == mtx_plain);

   mtx->val = 0;

   HG(ANNOTATE_RWLOCK_CREATE(mtx));
}

static inline void
simple_mtx_destroy(ASSERTED simple_mtx_t *mtx)
{
   HG(ANNOTATE_RWLOCK_DESTROY(mtx));
#ifndef NDEBUG
   mtx->val = _SIMPLE_MTX_INVALID_VALUE;
#endif
}

static inline void
simple_mtx_lock(simple_mtx_t *mtx)
{
   uint32_t c;

   c = p_atomic_cmpxchg(&mtx->val, 0, 1);

   assert(c != _SIMPLE_MTX_INVALID_VALUE);

   if (__builtin_expect(c != 0, 0)) {
      if (c != 2)
         c = p_atomic_xchg(&mtx->val, 2);
      while (c != 0) {
         futex_wait(&mtx->val, 2, NULL);
         c = p_atomic_xchg(&mtx->val, 2);
      }
   }

   HG(ANNOTATE_RWLOCK_ACQUIRED(mtx, 1));
}

static inline void
simple_mtx_unlock(simple_mtx_t *mtx)
{
   uint32_t c;

   HG(ANNOTATE_RWLOCK_RELEASED(mtx, 1));

   c = p_atomic_fetch_add(&mtx->val, -1);

   assert(c != _SIMPLE_MTX_INVALID_VALUE);

   if (__builtin_expect(c != 1, 0)) {
      mtx->val = 0;
      futex_wake(&mtx->val, 1);
   }
}

static inline void
simple_mtx_assert_locked(simple_mtx_t *mtx)
{
   assert(mtx->val);
}

#else /* !UTIL_FUTEX_SUPPORTED */

typedef struct simple_mtx_t {
   util_once_flag flag;
   mtx_t mtx;
} simple_mtx_t;

#define SIMPLE_MTX_INITIALIZER { UTIL_ONCE_FLAG_INIT }

void _simple_mtx_plain_init_once(simple_mtx_t *mtx);

static inline void
_simple_mtx_init_with_once(simple_mtx_t *mtx)
{
   util_call_once_data(&mtx->flag,
      (util_call_once_data_func)_simple_mtx_plain_init_once, mtx);
}

void
simple_mtx_init(simple_mtx_t *mtx, int type);

void
simple_mtx_destroy(simple_mtx_t *mtx);

static inline void
simple_mtx_lock(simple_mtx_t *mtx)
{
   _simple_mtx_init_with_once(mtx);
   mtx_lock(&mtx->mtx);
}

static inline void
simple_mtx_unlock(simple_mtx_t *mtx)
{
   _simple_mtx_init_with_once(mtx);
   mtx_unlock(&mtx->mtx);
}

static inline void
simple_mtx_assert_locked(simple_mtx_t *mtx)
{
#ifndef NDEBUG
   _simple_mtx_init_with_once(mtx);
   /* NOTE: this would not work for recursive mutexes, but
    * mtx_t doesn't support those
    */
   int ret = mtx_trylock(&mtx->mtx);
   assert(ret == thrd_busy);
   if (ret == thrd_success)
      mtx_unlock(&mtx->mtx);
#else
   (void)mtx;
#endif
}

#endif /* UTIL_FUTEX_SUPPORTED */

#ifdef __cplusplus
}
#endif

#endif /* _SIMPLE_MTX_H */
