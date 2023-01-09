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

#include "u_qsort.h"
#include <thread>

thread_local int (*tl_qsort_r_compar)(const void *, const void *, void *);
thread_local void *tl_qsort_r_arg;

static int
qsort_r_compar(const void *a, const void *b)
{
   return tl_qsort_r_compar(a, b, tl_qsort_r_arg);
}

void
util_tls_qsort_r(void *base, size_t nmemb, size_t size,
                 int (*compar)(const void *, const void *, void *),
                 void *arg)
{
   tl_qsort_r_compar = compar;
   tl_qsort_r_arg = arg;
   return qsort(base, nmemb, size, qsort_r_compar);
}

int
util_qsort_adapter(void *ctx, const void *elem1, const void *elem2)
{
   util_qsort_adapter_data *data =
      reinterpret_cast<util_qsort_adapter_data*>(ctx);
   return data->compar(elem1, elem2, data->args);
}
