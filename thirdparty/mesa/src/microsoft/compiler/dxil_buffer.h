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

#ifndef DXIL_BUFFER_H
#define DXIL_BUFFER_H

#include "util/blob.h"

struct dxil_buffer {
   struct blob blob;
   uint64_t buf;
   unsigned buf_bits;

   unsigned abbrev_width;
};

void
dxil_buffer_init(struct dxil_buffer *b, unsigned abbrev_width);

void
dxil_buffer_finish(struct dxil_buffer *b);

bool
dxil_buffer_emit_bits(struct dxil_buffer *b, uint32_t data, unsigned width);

bool
dxil_buffer_emit_vbr_bits(struct dxil_buffer *b, uint64_t data,
                          unsigned width);

bool
dxil_buffer_align(struct dxil_buffer *b);

static inline bool
dxil_buffer_emit_abbrev_id(struct dxil_buffer *b, uint32_t id)
{
   return dxil_buffer_emit_bits(b, id, b->abbrev_width);
}


#endif
