// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Bit writing and boolean coder
//
// Author: Skal (pascal.massimino@gmail.com)
//         Vikas Arora (vikaas.arora@gmail.com)

#include <assert.h>
#include <stdlib.h>
#include <string.h>   // for memcpy()

#include "src/utils/bit_writer_utils.h"
#include "src/webp/types.h"
#include "src/utils/endian_inl_utils.h"
#include "src/utils/utils.h"

//------------------------------------------------------------------------------
// VP8BitWriter

static int BitWriterResize(VP8BitWriter* const bw, size_t extra_size) {
  uint8_t* new_buf;
  size_t new_size;
  const uint64_t needed_size_64b = (uint64_t)bw->pos + extra_size;
  const size_t needed_size = (size_t)needed_size_64b;
  if (needed_size_64b != needed_size) {
    bw->error = 1;
    return 0;
  }
  if (needed_size <= bw->max_pos) return 1;
  // If the following line wraps over 32bit, the test just after will catch it.
  new_size = 2 * bw->max_pos;
  if (new_size < needed_size) new_size = needed_size;
  if (new_size < 1024) new_size = 1024;
  new_buf = (uint8_t*)WebPSafeMalloc(1ULL, new_size);
  if (new_buf == NULL) {
    bw->error = 1;
    return 0;
  }
  if (bw->pos > 0) {
    assert(bw->buf != NULL);
    memcpy(new_buf, bw->buf, bw->pos);
  }
  WebPSafeFree(bw->buf);
  bw->buf = new_buf;
  bw->max_pos = new_size;
  return 1;
}

static void Flush(VP8BitWriter* const bw) {
  const int s = 8 + bw->nb_bits;
  const int32_t bits = bw->value >> s;
  assert(bw->nb_bits >= 0);
  bw->value -= bits << s;
  bw->nb_bits -= 8;
  if ((bits & 0xff) != 0xff) {
    size_t pos = bw->pos;
    if (!BitWriterResize(bw, bw->run + 1)) {
      return;
    }
    if (bits & 0x100) {  // overflow -> propagate carry over pending 0xff's
      if (pos > 0) bw->buf[pos - 1]++;
    }
    if (bw->run > 0) {
      const int value = (bits & 0x100) ? 0x00 : 0xff;
      for (; bw->run > 0; --bw->run) bw->buf[pos++] = value;
    }
    bw->buf[pos++] = bits & 0xff;
    bw->pos = pos;
  } else {
    bw->run++;   // delay writing of bytes 0xff, pending eventual carry.
  }
}

//------------------------------------------------------------------------------
// renormalization

static const uint8_t kNorm[128] = {  // renorm_sizes[i] = 8 - log2(i)
     7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  0
};

// range = ((range + 1) << kVP8Log2Range[range]) - 1
static const uint8_t kNewRange[128] = {
  127, 127, 191, 127, 159, 191, 223, 127, 143, 159, 175, 191, 207, 223, 239,
  127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239,
  247, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179,
  183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239,
  243, 247, 251, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149,
  151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179,
  181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209,
  211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239,
  241, 243, 245, 247, 249, 251, 253, 127
};

int VP8PutBit(VP8BitWriter* const bw, int bit, int prob) {
  const int split = (bw->range * prob) >> 8;
  if (bit) {
    bw->value += split + 1;
    bw->range -= split + 1;
  } else {
    bw->range = split;
  }
  if (bw->range < 127) {   // emit 'shift' bits out and renormalize
    const int shift = kNorm[bw->range];
    bw->range = kNewRange[bw->range];
    bw->value <<= shift;
    bw->nb_bits += shift;
    if (bw->nb_bits > 0) Flush(bw);
  }
  return bit;
}

int VP8PutBitUniform(VP8BitWriter* const bw, int bit) {
  const int split = bw->range >> 1;
  if (bit) {
    bw->value += split + 1;
    bw->range -= split + 1;
  } else {
    bw->range = split;
  }
  if (bw->range < 127) {
    bw->range = kNewRange[bw->range];
    bw->value <<= 1;
    bw->nb_bits += 1;
    if (bw->nb_bits > 0) Flush(bw);
  }
  return bit;
}

void VP8PutBits(VP8BitWriter* const bw, uint32_t value, int nb_bits) {
  uint32_t mask;
  assert(nb_bits > 0 && nb_bits < 32);
  for (mask = 1u << (nb_bits - 1); mask; mask >>= 1) {
    VP8PutBitUniform(bw, value & mask);
  }
}

void VP8PutSignedBits(VP8BitWriter* const bw, int value, int nb_bits) {
  if (!VP8PutBitUniform(bw, value != 0)) return;
  if (value < 0) {
    VP8PutBits(bw, ((-value) << 1) | 1, nb_bits + 1);
  } else {
    VP8PutBits(bw, value << 1, nb_bits + 1);
  }
}

//------------------------------------------------------------------------------

int VP8BitWriterInit(VP8BitWriter* const bw, size_t expected_size) {
  bw->range   = 255 - 1;
  bw->value   = 0;
  bw->run     = 0;
  bw->nb_bits = -8;
  bw->pos     = 0;
  bw->max_pos = 0;
  bw->error   = 0;
  bw->buf     = NULL;
  return (expected_size > 0) ? BitWriterResize(bw, expected_size) : 1;
}

uint8_t* VP8BitWriterFinish(VP8BitWriter* const bw) {
  VP8PutBits(bw, 0, 9 - bw->nb_bits);
  bw->nb_bits = 0;   // pad with zeroes
  Flush(bw);
  return bw->buf;
}

int VP8BitWriterAppend(VP8BitWriter* const bw,
                       const uint8_t* data, size_t size) {
  assert(data != NULL);
  if (bw->nb_bits != -8) return 0;   // Flush() must have been called
  if (!BitWriterResize(bw, size)) return 0;
  memcpy(bw->buf + bw->pos, data, size);
  bw->pos += size;
  return 1;
}

void VP8BitWriterWipeOut(VP8BitWriter* const bw) {
  if (bw != NULL) {
    WebPSafeFree(bw->buf);
    memset(bw, 0, sizeof(*bw));
  }
}

//------------------------------------------------------------------------------
// VP8LBitWriter

// This is the minimum amount of size the memory buffer is guaranteed to grow
// when extra space is needed.
#define MIN_EXTRA_SIZE  (32768ULL)

// Returns 1 on success.
static int VP8LBitWriterResize(VP8LBitWriter* const bw, size_t extra_size) {
  uint8_t* allocated_buf;
  size_t allocated_size;
  const size_t max_bytes = bw->end - bw->buf;
  const size_t current_size = bw->cur - bw->buf;
  const uint64_t size_required_64b = (uint64_t)current_size + extra_size;
  const size_t size_required = (size_t)size_required_64b;
  if (size_required != size_required_64b) {
    bw->error = 1;
    return 0;
  }
  if (max_bytes > 0 && size_required <= max_bytes) return 1;
  allocated_size = (3 * max_bytes) >> 1;
  if (allocated_size < size_required) allocated_size = size_required;
  // make allocated size multiple of 1k
  allocated_size = (((allocated_size >> 10) + 1) << 10);
  allocated_buf = (uint8_t*)WebPSafeMalloc(1ULL, allocated_size);
  if (allocated_buf == NULL) {
    bw->error = 1;
    return 0;
  }
  if (current_size > 0) {
    memcpy(allocated_buf, bw->buf, current_size);
  }
  WebPSafeFree(bw->buf);
  bw->buf = allocated_buf;
  bw->cur = bw->buf + current_size;
  bw->end = bw->buf + allocated_size;
  return 1;
}

int VP8LBitWriterInit(VP8LBitWriter* const bw, size_t expected_size) {
  memset(bw, 0, sizeof(*bw));
  return VP8LBitWriterResize(bw, expected_size);
}

int VP8LBitWriterClone(const VP8LBitWriter* const src,
                       VP8LBitWriter* const dst) {
  const size_t current_size = src->cur - src->buf;
  assert(src->cur >= src->buf && src->cur <= src->end);
  if (!VP8LBitWriterResize(dst, current_size)) return 0;
  memcpy(dst->buf, src->buf, current_size);
  dst->bits = src->bits;
  dst->used = src->used;
  dst->error = src->error;
  dst->cur = dst->buf + current_size;
  return 1;
}

void VP8LBitWriterWipeOut(VP8LBitWriter* const bw) {
  if (bw != NULL) {
    WebPSafeFree(bw->buf);
    memset(bw, 0, sizeof(*bw));
  }
}

void VP8LBitWriterReset(const VP8LBitWriter* const bw_init,
                        VP8LBitWriter* const bw) {
  bw->bits = bw_init->bits;
  bw->used = bw_init->used;
  bw->cur = bw->buf + (bw_init->cur - bw_init->buf);
  assert(bw->cur <= bw->end);
  bw->error = bw_init->error;
}

void VP8LBitWriterSwap(VP8LBitWriter* const src, VP8LBitWriter* const dst) {
  const VP8LBitWriter tmp = *src;
  *src = *dst;
  *dst = tmp;
}

void VP8LPutBitsFlushBits(VP8LBitWriter* const bw) {
  // If needed, make some room by flushing some bits out.
  if (bw->cur + VP8L_WRITER_BYTES > bw->end) {
    const uint64_t extra_size = (bw->end - bw->buf) + MIN_EXTRA_SIZE;
    if (!CheckSizeOverflow(extra_size) ||
        !VP8LBitWriterResize(bw, (size_t)extra_size)) {
      bw->cur = bw->buf;
      bw->error = 1;
      return;
    }
  }
  *(vp8l_wtype_t*)bw->cur = (vp8l_wtype_t)WSWAP((vp8l_wtype_t)bw->bits);
  bw->cur += VP8L_WRITER_BYTES;
  bw->bits >>= VP8L_WRITER_BITS;
  bw->used -= VP8L_WRITER_BITS;
}

void VP8LPutBitsInternal(VP8LBitWriter* const bw, uint32_t bits, int n_bits) {
  assert(n_bits <= 32);
  // That's the max we can handle:
  assert(sizeof(vp8l_wtype_t) == 2);
  if (n_bits > 0) {
    vp8l_atype_t lbits = bw->bits;
    int used = bw->used;
    // Special case of overflow handling for 32bit accumulator (2-steps flush).
#if VP8L_WRITER_BITS == 16
    if (used + n_bits >= VP8L_WRITER_MAX_BITS) {
      // Fill up all the VP8L_WRITER_MAX_BITS so it can be flushed out below.
      const int shift = VP8L_WRITER_MAX_BITS - used;
      lbits |= (vp8l_atype_t)bits << used;
      used = VP8L_WRITER_MAX_BITS;
      n_bits -= shift;
      bits >>= shift;
      assert(n_bits <= VP8L_WRITER_MAX_BITS);
    }
#endif
    // If needed, make some room by flushing some bits out.
    while (used >= VP8L_WRITER_BITS) {
      if (bw->cur + VP8L_WRITER_BYTES > bw->end) {
        const uint64_t extra_size = (bw->end - bw->buf) + MIN_EXTRA_SIZE;
        if (!CheckSizeOverflow(extra_size) ||
            !VP8LBitWriterResize(bw, (size_t)extra_size)) {
          bw->cur = bw->buf;
          bw->error = 1;
          return;
        }
      }
      *(vp8l_wtype_t*)bw->cur = (vp8l_wtype_t)WSWAP((vp8l_wtype_t)lbits);
      bw->cur += VP8L_WRITER_BYTES;
      lbits >>= VP8L_WRITER_BITS;
      used -= VP8L_WRITER_BITS;
    }
    bw->bits = lbits | ((vp8l_atype_t)bits << used);
    bw->used = used + n_bits;
  }
}

uint8_t* VP8LBitWriterFinish(VP8LBitWriter* const bw) {
  // flush leftover bits
  if (VP8LBitWriterResize(bw, (bw->used + 7) >> 3)) {
    while (bw->used > 0) {
      *bw->cur++ = (uint8_t)bw->bits;
      bw->bits >>= 8;
      bw->used -= 8;
    }
    bw->used = 0;
  }
  return bw->buf;
}

//------------------------------------------------------------------------------
