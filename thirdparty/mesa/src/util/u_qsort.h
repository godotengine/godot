/*
 * Copyright © 2020 Intel Corporation
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

#ifndef U_QSORT_H
#define U_QSORT_H

#include <stdlib.h>

#include "detect_os.h"

#ifdef __cplusplus
extern "C" {
#endif

void util_tls_qsort_r(void *base, size_t nmemb, size_t size,
                      int (*compar)(const void *, const void *, void *),
                      void *arg);


struct util_qsort_adapter_data {
   int (*compar)(const void*, const void*, void*);
   void *args;
};

/**
 * Converts comparison function arguments
 * from [MSVC, BSD, macOS]
 * (void *ctx, const void *elem1, const void *elem2)
 * to [GNU, C11]
 * (const void *elem1, const void *elem2, void *ctx);
 */
int util_qsort_adapter(void *ctx, const void *elem1, const void *elem2);

static inline void
util_qsort_r(void *base, size_t nmemb, size_t size,
             int (*compar)(const void *, const void *, void *),
             void *arg)
{
#if defined(HAVE_GNU_QSORT_R)
   /* GNU extension added in glibc 2.8 */
   qsort_r(base, nmemb, size, compar, arg);
#elif defined(HAVE_BSD_QSORT_R)
   /* BSD/macOS qsort_r takes "arg" before the comparison function and it
    * pass the "arg" before the elements.
    */
   struct util_qsort_adapter_data data = {
      compar,
      arg
   };
   qsort_r(base, nmemb, size, &data, util_qsort_adapter);
#elif HAVE_QSORT_S
#  ifdef _WIN32
   /* MSVC/MinGW qsort_s takes "arg" after the comparison function and it
    * pass the "arg" before the elements.
    */
   struct util_qsort_adapter_data data = {
      compar,
      arg
   };
   qsort_s(base, nmemb, size, util_qsort_adapter, &data);
#  else
   /* C11 added qsort_s */
   qsort_s(base, nmemb, size, compar, arg);
#  endif
#else
   /* Fall-back to using thread local storage */
   util_tls_qsort_r(base, nmemb, size, compar, arg);
#endif
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* U_QSORT_H */
