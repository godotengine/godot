/*
 * Copyright (c) 2015 Intel Corporation
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

#ifndef STRNDUP_H
#define STRNDUP_H

#if defined(_WIN32)

#include <stdlib.h> // size_t
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline char *
strndup(const char *str, size_t max)
{
   size_t n;
   char *ptr;

   if (!str)
      return NULL;

   n = strnlen(str, max);
   ptr = (char *) calloc(n + 1, sizeof(char));
   if (!ptr)
      return NULL;

   memcpy(ptr, str, n);
   return ptr;
}

#ifdef __cplusplus
}
#endif

#endif /* _WIN32 */

#endif /* STRNDUP_H */
