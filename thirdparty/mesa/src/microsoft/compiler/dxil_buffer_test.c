/*
 * Copyright © Microsoft Corporation
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "dxil_buffer.h"
#include <assert.h>
#include <stdio.h>

static void
init()
{
   struct dxil_buffer buf;
   dxil_buffer_init(&buf, 2);
   assert(!buf.buf);
   assert(!buf.buf_bits);
}

static void
assert_blob_data(const struct dxil_buffer *m, const uint8_t *data,
                   size_t len)
{
   if (m->blob.size != len) {
      fprintf(stderr, "blob-size mismatch, expected %zd, got %zd",
                      len, m->blob.size);
      abort();
   }

   for (size_t i = 0; i < len; ++i) {
      if (m->blob.data[i] != data[i]) {
         fprintf(stderr, "blob-data mismatch at index %zd, "
                         "expected 0x%02x, got 0x%02x", i,
                         data[i], m->blob.data[i]);
         abort();
      }
   }
}

#define ASSERT_BLOB_DATA(m, data) \
   assert_blob_data(m, data, sizeof(data))

static void
align()
{
   struct dxil_buffer buf;
   dxil_buffer_init(&buf, 2);
   assert_blob_data(&buf, NULL, 0);

   dxil_buffer_init(&buf, 2);
   dxil_buffer_emit_bits(&buf, 0xbeef, 16);
   dxil_buffer_align(&buf);
   assert(!buf.buf);
   assert(!buf.buf_bits);
   uint8_t expected0[] = { 0xef, 0xbe, 0x00, 0x00 };
   ASSERT_BLOB_DATA(&buf, expected0);
   dxil_buffer_align(&buf);
   ASSERT_BLOB_DATA(&buf, expected0);
}

static void
emit_bits()
{
   struct dxil_buffer buf;
   dxil_buffer_init(&buf, 2);
   dxil_buffer_emit_bits(&buf, 0xbeef, 16);
   dxil_buffer_align(&buf);
   assert(!buf.buf);
   assert(!buf.buf_bits);
   uint8_t expected0[] = { 0xef, 0xbe, 0x00, 0x00 };
   ASSERT_BLOB_DATA(&buf, expected0);

   dxil_buffer_init(&buf, 2);
   dxil_buffer_emit_bits(&buf, 0xdead, 16);
   dxil_buffer_emit_bits(&buf, 0xbeef, 16);
   assert(!buf.buf);
   assert(!buf.buf_bits);
   uint8_t expected1[] = { 0xad, 0xde, 0xef, 0xbe };
   ASSERT_BLOB_DATA(&buf, expected1);

   dxil_buffer_init(&buf, 2);
   dxil_buffer_emit_bits(&buf, 0x1111111, 28);
   dxil_buffer_emit_bits(&buf, 0x22222222, 32);
   dxil_buffer_align(&buf);
   uint8_t expected2[] = { 0x11, 0x11, 0x11, 0x21, 0x22, 0x22, 0x22, 0x02 };
   ASSERT_BLOB_DATA(&buf, expected2);
}

static void
emit_vbr_bits()
{
   struct dxil_buffer buf;
   dxil_buffer_init(&buf, 2);
   dxil_buffer_emit_vbr_bits(&buf, 0x1a, 8);
   dxil_buffer_emit_vbr_bits(&buf, 0x1a, 6);
   dxil_buffer_emit_vbr_bits(&buf, 0x00, 2);
   dxil_buffer_emit_vbr_bits(&buf, 0x0a, 4);
   dxil_buffer_emit_vbr_bits(&buf, 0x04, 2);
   dxil_buffer_emit_vbr_bits(&buf, 0x00, 2);
   uint8_t expected[] = { 0x1a, 0x1a, 0x1a, 0x1a };
   ASSERT_BLOB_DATA(&buf, expected);
}

int
main()
{
   init();
   align();
   emit_bits();
   emit_vbr_bits();
   return 0;
}
