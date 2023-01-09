/**************************************************************************
 *
 * Copyright 2020 Lag Free Games, LLC
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

#ifndef MEMSTREAM_H
#define MEMSTREAM_H

#include <stdbool.h>
#include <stdio.h>
#include <limits.h> /* PATH_MAX */

#ifdef _MSC_VER
#include <stdlib.h>
#ifndef PATH_MAX
#define PATH_MAX _MAX_PATH /* Equivalent to MAX_PATH from minwindef.h */
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct u_memstream
{
   FILE *f;
#ifdef _WIN32
   char **bufp;
   size_t *sizep;
   char temp[PATH_MAX];
#endif
};

extern bool
u_memstream_open(struct u_memstream *mem, char **bufp, size_t *sizep);

extern void
u_memstream_close(struct u_memstream *mem);

static inline FILE *
u_memstream_get(const struct u_memstream *mem)
{
   return mem->f;
}

#ifdef __cplusplus
}
#endif

#endif
