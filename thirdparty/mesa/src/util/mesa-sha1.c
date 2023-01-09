/* Copyright © 2007 Carl Worth
 * Copyright © 2009 Jeremy Huddleston, Julien Cristau, and Matthieu Herrb
 * Copyright © 2009-2010 Mikhail Gusarov
 * Copyright © 2012 Yaakov Selkowitz and Keith Packard
 * Copyright © 2014 Intel Corporation
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

#include "sha1/sha1.h"
#include "mesa-sha1.h"
#include <string.h>
#include <inttypes.h>

void
_mesa_sha1_compute(const void *data, size_t size, unsigned char result[20])
{
   struct mesa_sha1 ctx;

   _mesa_sha1_init(&ctx);
   _mesa_sha1_update(&ctx, data, size);
   _mesa_sha1_final(&ctx, result);
}

void
_mesa_sha1_format(char *buf, const unsigned char *sha1)
{
   static const char hex_digits[] = "0123456789abcdef";
   int i;

   for (i = 0; i < 40; i += 2) {
      buf[i] = hex_digits[sha1[i >> 1] >> 4];
      buf[i + 1] = hex_digits[sha1[i >> 1] & 0x0f];
   }
   buf[i] = '\0';
}

/* Convert a hashs string hexidecimal representation into its more compact
 * form.
 */
void
_mesa_sha1_hex_to_sha1(unsigned char *buf, const char *hex)
{
   for (unsigned i = 0; i < 20; i++) {
      char tmp[3];
      tmp[0] = hex[i * 2];
      tmp[1] = hex[(i * 2) + 1];
      tmp[2] = '\0';
      buf[i] = strtol(tmp, NULL, 16);
   }
}

static void
sha1_to_uint32(const uint8_t sha1[SHA1_DIGEST_LENGTH],
               uint32_t out[SHA1_DIGEST_LENGTH32])
{
   memset(out, 0, SHA1_DIGEST_LENGTH);

   for (unsigned i = 0; i < SHA1_DIGEST_LENGTH; i++)
      out[i / 4] |= (uint32_t)sha1[i] << ((i % 4) * 8);
}

void
_mesa_sha1_print(FILE *f, const uint8_t sha1[SHA1_DIGEST_LENGTH])
{
   uint32_t u32[SHA1_DIGEST_LENGTH];
   sha1_to_uint32(sha1, u32);

   for (unsigned i = 0; i < SHA1_DIGEST_LENGTH32; i++) {
      fprintf(f, i ? ", 0x%08" PRIx32 : "0x%08" PRIx32, u32[i]);
   }
}

bool
_mesa_printed_sha1_equal(const uint8_t sha1[SHA1_DIGEST_LENGTH],
                         const uint32_t printed_sha1[SHA1_DIGEST_LENGTH32])
{
   uint32_t u32[SHA1_DIGEST_LENGTH32];
   sha1_to_uint32(sha1, u32);

   return memcmp(u32, printed_sha1, sizeof(u32)) == 0;
}
