/**************************************************************************
 *
 * Copyright 2011 Christian König.
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

/*
 * Functions for fast bitwise access to multiple probably unaligned input buffers
 */

#ifndef vl_vlc_h
#define vl_vlc_h

#include "util/u_math.h"

struct vl_vlc
{
   uint64_t buffer;
   signed invalid_bits;
   const uint8_t *data;
   const uint8_t *end;

   const void *const *inputs;
   const unsigned    *sizes;
   unsigned          bytes_left;
};

struct vl_vlc_entry
{
   int8_t length;
   int8_t value;
};

struct vl_vlc_compressed
{
   uint16_t bitcode;
   struct vl_vlc_entry entry;
};

/**
 * initalize and decompress a lookup table
 */
static inline void
vl_vlc_init_table(struct vl_vlc_entry *dst, unsigned dst_size, const struct vl_vlc_compressed *src, unsigned src_size)
{
   unsigned i, bits = util_logbase2(dst_size);

   assert(dst && dst_size);
   assert(src && src_size);

   for (i=0;i<dst_size;++i) {
      dst[i].length = 0;
      dst[i].value = 0;
   }

   for(; src_size > 0; --src_size, ++src) {
      for(i = 0; i < (1u << (bits - src->entry.length)); ++i)
         dst[src->bitcode >> (16 - bits) | i] = src->entry;
   }
}

/**
 * switch over to next input buffer
 */
static inline void
vl_vlc_next_input(struct vl_vlc *vlc)
{
   unsigned len = vlc->sizes[0];

   assert(vlc);
   assert(vlc->bytes_left);

   if (len < vlc->bytes_left)
      vlc->bytes_left -= len;
   else {
      len = vlc->bytes_left;
      vlc->bytes_left = 0;
   }

   vlc->data = (const uint8_t *) vlc->inputs[0];
   vlc->end = vlc->data + len;

   ++vlc->inputs;
   ++vlc->sizes;
}

/**
 * align the data pointer to the next dword
 */
static inline void
vl_vlc_align_data_ptr(struct vl_vlc *vlc)
{
   /* align the data pointer */
   while (vlc->data != vlc->end && ((uintptr_t)vlc->data) & 3) {
      vlc->buffer |= (uint64_t)*vlc->data << (24 + vlc->invalid_bits);
      ++vlc->data;
      vlc->invalid_bits -= 8;
   }
}

/**
 * fill the bit buffer, so that at least 32 bits are valid
 */
static inline void
vl_vlc_fillbits(struct vl_vlc *vlc)
{
   assert(vlc);

   /* as long as the buffer needs to be filled */
   while (vlc->invalid_bits > 0) {
      unsigned bytes_left = vlc->end - vlc->data;

      /* if this input is depleted */
      if (bytes_left == 0) {

         if (vlc->bytes_left) {
            /* go on to next input */
            vl_vlc_next_input(vlc);
            vl_vlc_align_data_ptr(vlc);
         } else
            /* or give up since we don't have anymore inputs */
            return;

      } else if (bytes_left >= 4) {

         /* enough bytes in buffer, read in a whole dword */
         uint64_t value = *(const uint32_t*)vlc->data;

#if !UTIL_ARCH_BIG_ENDIAN
         value = util_bswap32(value);
#endif

         vlc->buffer |= value << vlc->invalid_bits;
         vlc->data += 4;
         vlc->invalid_bits -= 32;

         /* buffer is now definitely filled up avoid the loop test */
         break;

      } else while (vlc->data < vlc->end) {

         /* not enough bytes left in buffer, read single bytes */
         vlc->buffer |= (uint64_t)*vlc->data << (24 + vlc->invalid_bits);
         ++vlc->data;
         vlc->invalid_bits -= 8;
      }
   }
}

/**
 * initialize vlc structure and start reading from first input buffer
 */
static inline void
vl_vlc_init(struct vl_vlc *vlc, unsigned num_inputs,
            const void *const *inputs, const unsigned *sizes)
{
   unsigned i;

   assert(vlc);
   assert(num_inputs);

   vlc->buffer = 0;
   vlc->invalid_bits = 32;
   vlc->inputs = inputs;
   vlc->sizes = sizes;
   vlc->bytes_left = 0;

   for (i = 0; i < num_inputs; ++i)
      vlc->bytes_left += sizes[i];

   if (vlc->bytes_left) {
      vl_vlc_next_input(vlc);
      vl_vlc_align_data_ptr(vlc);
      vl_vlc_fillbits(vlc);
   }
}

/**
 * number of bits still valid in bit buffer
 */
static inline unsigned
vl_vlc_valid_bits(struct vl_vlc *vlc)
{
   return 32 - vlc->invalid_bits;
}

/**
 * number of bits left over all inbut buffers
 */
static inline unsigned
vl_vlc_bits_left(struct vl_vlc *vlc)
{
   signed bytes_left = vlc->end - vlc->data;
   bytes_left += vlc->bytes_left;
   return bytes_left * 8 + vl_vlc_valid_bits(vlc);
}

/**
 * get num_bits from bit buffer without removing them
 */
static inline unsigned
vl_vlc_peekbits(struct vl_vlc *vlc, unsigned num_bits)
{
   assert(vl_vlc_valid_bits(vlc) >= num_bits || vlc->data >= vlc->end);
   return vlc->buffer >> (64 - num_bits);
}

/**
 * remove num_bits from bit buffer
 */
static inline void
vl_vlc_eatbits(struct vl_vlc *vlc, unsigned num_bits)
{
   assert(vl_vlc_valid_bits(vlc) >= num_bits);

   vlc->buffer <<= num_bits;
   vlc->invalid_bits += num_bits;
}

/**
 * get num_bits from bit buffer with removing them
 */
static inline unsigned
vl_vlc_get_uimsbf(struct vl_vlc *vlc, unsigned num_bits)
{
   unsigned value;

   assert(vl_vlc_valid_bits(vlc) >= num_bits);

   value = vlc->buffer >> (64 - num_bits);
   vl_vlc_eatbits(vlc, num_bits);

   return value;
}

/**
 * treat num_bits as signed value and remove them from bit buffer
 */
static inline signed
vl_vlc_get_simsbf(struct vl_vlc *vlc, unsigned num_bits)
{
   signed value;

   assert(vl_vlc_valid_bits(vlc) >= num_bits);

   value = ((int64_t)vlc->buffer) >> (64 - num_bits);
   vl_vlc_eatbits(vlc, num_bits);

   return value;
}

/**
 * lookup a value and length in a decompressed table
 */
static inline int8_t
vl_vlc_get_vlclbf(struct vl_vlc *vlc, const struct vl_vlc_entry *tbl, unsigned num_bits)
{
   tbl += vl_vlc_peekbits(vlc, num_bits);
   vl_vlc_eatbits(vlc, tbl->length);
   return tbl->value;
}

/**
 * fast forward search for a specific byte value
 */
static inline bool
vl_vlc_search_byte(struct vl_vlc *vlc, unsigned num_bits, uint8_t value)
{
   /* make sure we are on a byte boundary */
   assert((vl_vlc_valid_bits(vlc) % 8) == 0);
   assert(num_bits == ~0u || (num_bits % 8) == 0);

   /* deplete the bit buffer */
   while (vl_vlc_valid_bits(vlc) > 0) {

      if (vl_vlc_peekbits(vlc, 8) == value) {
         vl_vlc_fillbits(vlc);
         return true;
      }

      vl_vlc_eatbits(vlc, 8);

      if (num_bits != ~0u) {
         num_bits -= 8;
         if (num_bits == 0)
            return false;
      }
   }

   /* deplete the byte buffers */
   while (1) {

      /* if this input is depleted */
      if (vlc->data == vlc->end) {
         if (vlc->bytes_left)
            /* go on to next input */
            vl_vlc_next_input(vlc);
         else
            /* or give up since we don't have anymore inputs */
            return false;
      }

      if (*vlc->data == value) {
         vl_vlc_align_data_ptr(vlc);
         vl_vlc_fillbits(vlc);
         return true;
      }

      ++vlc->data;
      if (num_bits != ~0u) {
         num_bits -= 8;
         if (num_bits == 0) {
            vl_vlc_align_data_ptr(vlc);
            return false;
         }
      }
   }
}

/**
 * remove num_bits bits starting at pos from the bitbuffer
 */
static inline void
vl_vlc_removebits(struct vl_vlc *vlc, unsigned pos, unsigned num_bits)
{
   uint64_t lo = (vlc->buffer & (~0UL >> (pos + num_bits))) << num_bits;
   uint64_t hi = (vlc->buffer & (~0UL << (64 - pos)));
   vlc->buffer = lo | hi;
   vlc->invalid_bits += num_bits;
}

/**
 * limit the number of bits left for fetching
 */
static inline void
vl_vlc_limit(struct vl_vlc *vlc, unsigned bits_left)
{
   assert(bits_left <= vl_vlc_bits_left(vlc));

   vl_vlc_fillbits(vlc);
   if (bits_left < vl_vlc_valid_bits(vlc)) {
      vlc->invalid_bits = 32 - bits_left;
      vlc->buffer &= ~0L << (vlc->invalid_bits + 32);
      vlc->end = vlc->data;
      vlc->bytes_left = 0;
   } else {
      assert((bits_left - vl_vlc_valid_bits(vlc)) % 8 == 0);
      vlc->bytes_left = (bits_left - vl_vlc_valid_bits(vlc)) / 8;
      if (vlc->bytes_left < (vlc->end - vlc->data)) {
         vlc->end = vlc->data + vlc->bytes_left;
         vlc->bytes_left = 0;
      } else
         vlc->bytes_left -= vlc->end - vlc->data;
   }
}

#endif /* vl_vlc_h */
