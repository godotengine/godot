/* Copyright © 2014 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef MESA_SHA1_H
#define MESA_SHA1_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "sha1/sha1.h"

#ifdef __cplusplus
extern "C" {
#endif

#define mesa_sha1 _SHA1_CTX
#define SHA1_DIGEST_LENGTH32 (SHA1_DIGEST_LENGTH / 4)

static inline void
_mesa_sha1_init(struct mesa_sha1 *ctx)
{
   SHA1Init(ctx);
}

static inline void
_mesa_sha1_update(struct mesa_sha1 *ctx, const void *data, size_t size)
{
   SHA1Update(ctx, (const unsigned char *)data, size);
}

static inline void
_mesa_sha1_final(struct mesa_sha1 *ctx, unsigned char result[20])
{
   SHA1Final(result, ctx);
}

void
_mesa_sha1_format(char *buf, const unsigned char *sha1);

void
_mesa_sha1_hex_to_sha1(unsigned char *buf, const char *hex);

void
_mesa_sha1_compute(const void *data, size_t size, unsigned char result[20]);

void
_mesa_sha1_print(FILE *f, const uint8_t sha1[SHA1_DIGEST_LENGTH]);

bool
_mesa_printed_sha1_equal(const uint8_t sha1[SHA1_DIGEST_LENGTH],
                         const uint32_t printed_sha1[SHA1_DIGEST_LENGTH32]);

#ifdef __cplusplus
} /* extern C */
#endif

#endif
