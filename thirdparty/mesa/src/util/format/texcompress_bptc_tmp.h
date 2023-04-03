/*
 * Copyright (C) 2014 Intel Corporation
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

/*
 * Included by texcompress_bptc and gallium to define BPTC decoding routines.
 */

#ifndef TEXCOMPRESS_BPTC_TMP_H
#define TEXCOMPRESS_BPTC_TMP_H

#include "util/bitscan.h"
#include "util/format_srgb.h"
#include "util/half_float.h"
#include "util/u_math.h"

#define BLOCK_SIZE 4
#define N_PARTITIONS 64
#define BLOCK_BYTES 16

struct bptc_unorm_mode {
   int n_subsets;
   int n_partition_bits;
   bool has_rotation_bits;
   bool has_index_selection_bit;
   int n_color_bits;
   int n_alpha_bits;
   bool has_endpoint_pbits;
   bool has_shared_pbits;
   int n_index_bits;
   int n_secondary_index_bits;
};

struct bptc_float_bitfield {
   int8_t endpoint;
   uint8_t component;
   uint8_t offset;
   uint8_t n_bits;
   bool reverse;
};

struct bptc_float_mode {
   bool reserved;
   bool transformed_endpoints;
   int n_partition_bits;
   int n_endpoint_bits;
   int n_index_bits;
   int n_delta_bits[3];
   struct bptc_float_bitfield bitfields[24];
};

struct bit_writer {
   uint8_t buf;
   int pos;
   uint8_t *dst;
};

static const struct bptc_unorm_mode
bptc_unorm_modes[] = {
   /* 0 */ { 3, 4, false, false, 4, 0, true,  false, 3, 0 },
   /* 1 */ { 2, 6, false, false, 6, 0, false, true,  3, 0 },
   /* 2 */ { 3, 6, false, false, 5, 0, false, false, 2, 0 },
   /* 3 */ { 2, 6, false, false, 7, 0, true,  false, 2, 0 },
   /* 4 */ { 1, 0, true,  true,  5, 6, false, false, 2, 3 },
   /* 5 */ { 1, 0, true,  false, 7, 8, false, false, 2, 2 },
   /* 6 */ { 1, 0, false, false, 7, 7, true,  false, 4, 0 },
   /* 7 */ { 2, 6, false, false, 5, 5, true,  false, 2, 0 }
};

static const struct bptc_float_mode
bptc_float_modes[] = {
   /* 00 */
   { false, true, 5, 10, 3, { 5, 5, 5 },
     { { 2, 1, 4, 1, false }, { 2, 2, 4, 1, false }, { 3, 2, 4, 1, false },
       { 0, 0, 0, 10, false }, { 0, 1, 0, 10, false }, { 0, 2, 0, 10, false },
       { 1, 0, 0, 5, false }, { 3, 1, 4, 1, false }, { 2, 1, 0, 4, false },
       { 1, 1, 0, 5, false }, { 3, 2, 0, 1, false }, { 3, 1, 0, 4, false },
       { 1, 2, 0, 5, false }, { 3, 2, 1, 1, false }, { 2, 2, 0, 4, false },
       { 2, 0, 0, 5, false }, { 3, 2, 2, 1, false }, { 3, 0, 0, 5, false },
       { 3, 2, 3, 1, false },
       { -1 } }
   },
   /* 01 */
   { false, true, 5, 7, 3, { 6, 6, 6 },
     { { 2, 1, 5, 1, false }, { 3, 1, 4, 1, false }, { 3, 1, 5, 1, false },
       { 0, 0, 0, 7, false }, { 3, 2, 0, 1, false }, { 3, 2, 1, 1, false },
       { 2, 2, 4, 1, false }, { 0, 1, 0, 7, false }, { 2, 2, 5, 1, false },
       { 3, 2, 2, 1, false }, { 2, 1, 4, 1, false }, { 0, 2, 0, 7, false },
       { 3, 2, 3, 1, false }, { 3, 2, 5, 1, false }, { 3, 2, 4, 1, false },
       { 1, 0, 0, 6, false }, { 2, 1, 0, 4, false }, { 1, 1, 0, 6, false },
       { 3, 1, 0, 4, false }, { 1, 2, 0, 6, false }, { 2, 2, 0, 4, false },
       { 2, 0, 0, 6, false },
       { 3, 0, 0, 6, false },
       { -1 } }
   },
   /* 00010 */
   { false, true, 5, 11, 3, { 5, 4, 4 },
     { { 0, 0, 0, 10, false }, { 0, 1, 0, 10, false }, { 0, 2, 0, 10, false },
       { 1, 0, 0, 5, false }, { 0, 0, 10, 1, false }, { 2, 1, 0, 4, false },
       { 1, 1, 0, 4, false }, { 0, 1, 10, 1, false }, { 3, 2, 0, 1, false },
       { 3, 1, 0, 4, false }, { 1, 2, 0, 4, false }, { 0, 2, 10, 1, false },
       { 3, 2, 1, 1, false }, { 2, 2, 0, 4, false }, { 2, 0, 0, 5, false },
       { 3, 2, 2, 1, false }, { 3, 0, 0, 5, false }, { 3, 2, 3, 1, false },
       { -1 } }
   },
   /* 00011 */
   { false, false, 0, 10, 4, { 10, 10, 10 },
     { { 0, 0, 0, 10, false }, { 0, 1, 0, 10, false }, { 0, 2, 0, 10, false },
       { 1, 0, 0, 10, false }, { 1, 1, 0, 10, false }, { 1, 2, 0, 10, false },
       { -1 } }
   },
   /* 00110 */
   { false, true, 5, 11, 3, { 4, 5, 4 },
     { { 0, 0, 0, 10, false }, { 0, 1, 0, 10, false }, { 0, 2, 0, 10, false },
       { 1, 0, 0, 4, false }, { 0, 0, 10, 1, false }, { 3, 1, 4, 1, false },
       { 2, 1, 0, 4, false }, { 1, 1, 0, 5, false }, { 0, 1, 10, 1, false },
       { 3, 1, 0, 4, false }, { 1, 2, 0, 4, false }, { 0, 2, 10, 1, false },
       { 3, 2, 1, 1, false }, { 2, 2, 0, 4, false }, { 2, 0, 0, 4, false },
       { 3, 2, 0, 1, false }, { 3, 2, 2, 1, false }, { 3, 0, 0, 4, false },
       { 2, 1, 4, 1, false }, { 3, 2, 3, 1, false },
       { -1 } }
   },
   /* 00111 */
   { false, true, 0, 11, 4, { 9, 9, 9 },
     { { 0, 0, 0, 10, false }, { 0, 1, 0, 10, false }, { 0, 2, 0, 10, false },
       { 1, 0, 0, 9, false }, { 0, 0, 10, 1, false }, { 1, 1, 0, 9, false },
       { 0, 1, 10, 1, false }, { 1, 2, 0, 9, false }, { 0, 2, 10, 1, false },
       { -1 } }
   },
   /* 01010 */
   { false, true, 5, 11, 3, { 4, 4, 5 },
     { { 0, 0, 0, 10, false }, { 0, 1, 0, 10, false }, { 0, 2, 0, 10, false },
       { 1, 0, 0, 4, false }, { 0, 0, 10, 1, false }, { 2, 2, 4, 1, false },
       { 2, 1, 0, 4, false }, { 1, 1, 0, 4, false }, { 0, 1, 10, 1, false },
       { 3, 2, 0, 1, false }, { 3, 1, 0, 4, false }, { 1, 2, 0, 5, false },
       { 0, 2, 10, 1, false }, { 2, 2, 0, 4, false }, { 2, 0, 0, 4, false },
       { 3, 2, 1, 1, false }, { 3, 2, 2, 1, false }, { 3, 0, 0, 4, false },
       { 3, 2, 4, 1, false }, { 3, 2, 3, 1, false },
       { -1 } }
   },
   /* 01011 */
   { false, true, 0, 12, 4, { 8, 8, 8 },
     { { 0, 0, 0, 10, false }, { 0, 1, 0, 10, false }, { 0, 2, 0, 10, false },
       { 1, 0, 0, 8, false }, { 0, 0, 10, 2, true }, { 1, 1, 0, 8, false },
       { 0, 1, 10, 2, true }, { 1, 2, 0, 8, false }, { 0, 2, 10, 2, true },
       { -1 } }
   },
   /* 01110 */
   { false, true, 5, 9, 3, { 5, 5, 5 },
     { { 0, 0, 0, 9, false }, { 2, 2, 4, 1, false }, { 0, 1, 0, 9, false },
       { 2, 1, 4, 1, false }, { 0, 2, 0, 9, false }, { 3, 2, 4, 1, false },
       { 1, 0, 0, 5, false }, { 3, 1, 4, 1, false }, { 2, 1, 0, 4, false },
       { 1, 1, 0, 5, false }, { 3, 2, 0, 1, false }, { 3, 1, 0, 4, false },
       { 1, 2, 0, 5, false }, { 3, 2, 1, 1, false }, { 2, 2, 0, 4, false },
       { 2, 0, 0, 5, false }, { 3, 2, 2, 1, false }, { 3, 0, 0, 5, false },
       { 3, 2, 3, 1, false },
       { -1 } }
   },
   /* 01111 */
   { false, true, 0, 16, 4, { 4, 4, 4 },
     { { 0, 0, 0, 10, false }, { 0, 1, 0, 10, false }, { 0, 2, 0, 10, false },
       { 1, 0, 0, 4, false }, { 0, 0, 10, 6, true }, { 1, 1, 0, 4, false },
       { 0, 1, 10, 6, true }, { 1, 2, 0, 4, false }, { 0, 2, 10, 6, true },
       { -1 } }
   },
   /* 10010 */
   { false, true, 5, 8, 3, { 6, 5, 5 },
     { { 0, 0, 0, 8, false }, { 3, 1, 4, 1, false }, { 2, 2, 4, 1, false },
       { 0, 1, 0, 8, false }, { 3, 2, 2, 1, false }, { 2, 1, 4, 1, false },
       { 0, 2, 0, 8, false }, { 3, 2, 3, 1, false }, { 3, 2, 4, 1, false },
       { 1, 0, 0, 6, false }, { 2, 1, 0, 4, false }, { 1, 1, 0, 5, false },
       { 3, 2, 0, 1, false }, { 3, 1, 0, 4, false }, { 1, 2, 0, 5, false },
       { 3, 2, 1, 1, false }, { 2, 2, 0, 4, false }, { 2, 0, 0, 6, false },
       { 3, 0, 0, 6, false },
       { -1 } }
   },
   /* 10011 */
   { true /* reserved */ },
   /* 10110 */
   { false, true, 5, 8, 3, { 5, 6, 5 },
     { { 0, 0, 0, 8, false }, { 3, 2, 0, 1, false }, { 2, 2, 4, 1, false },
       { 0, 1, 0, 8, false }, { 2, 1, 5, 1, false }, { 2, 1, 4, 1, false },
       { 0, 2, 0, 8, false }, { 3, 1, 5, 1, false }, { 3, 2, 4, 1, false },
       { 1, 0, 0, 5, false }, { 3, 1, 4, 1, false }, { 2, 1, 0, 4, false },
       { 1, 1, 0, 6, false }, { 3, 1, 0, 4, false }, { 1, 2, 0, 5, false },
       { 3, 2, 1, 1, false }, { 2, 2, 0, 4, false }, { 2, 0, 0, 5, false },
       { 3, 2, 2, 1, false }, { 3, 0, 0, 5, false }, { 3, 2, 3, 1, false },
       { -1 } }
   },
   /* 10111 */
   { true /* reserved */ },
   /* 11010 */
   { false, true, 5, 8, 3, { 5, 5, 6 },
     { { 0, 0, 0, 8, false }, { 3, 2, 1, 1, false }, { 2, 2, 4, 1, false },
       { 0, 1, 0, 8, false }, { 2, 2, 5, 1, false }, { 2, 1, 4, 1, false },
       { 0, 2, 0, 8, false }, { 3, 2, 5, 1, false }, { 3, 2, 4, 1, false },
       { 1, 0, 0, 5, false }, { 3, 1, 4, 1, false }, { 2, 1, 0, 4, false },
       { 1, 1, 0, 5, false }, { 3, 2, 0, 1, false }, { 3, 1, 0, 4, false },
       { 1, 2, 0, 6, false }, { 2, 2, 0, 4, false }, { 2, 0, 0, 5, false },
       { 3, 2, 2, 1, false }, { 3, 0, 0, 5, false }, { 3, 2, 3, 1, false },
       { -1 } }
   },
   /* 11011 */
   { true /* reserved */ },
   /* 11110 */
   { false, false, 5, 6, 3, { 6, 6, 6 },
     { { 0, 0, 0, 6, false }, { 3, 1, 4, 1, false }, { 3, 2, 0, 1, false },
       { 3, 2, 1, 1, false }, { 2, 2, 4, 1, false }, { 0, 1, 0, 6, false },
       { 2, 1, 5, 1, false }, { 2, 2, 5, 1, false }, { 3, 2, 2, 1, false },
       { 2, 1, 4, 1, false }, { 0, 2, 0, 6, false }, { 3, 1, 5, 1, false },
       { 3, 2, 3, 1, false }, { 3, 2, 5, 1, false }, { 3, 2, 4, 1, false },
       { 1, 0, 0, 6, false }, { 2, 1, 0, 4, false }, { 1, 1, 0, 6, false },
       { 3, 1, 0, 4, false }, { 1, 2, 0, 6, false }, { 2, 2, 0, 4, false },
       { 2, 0, 0, 6, false }, { 3, 0, 0, 6, false },
       { -1 } }
   },
   /* 11111 */
   { true /* reserved */ },
};

/* This partition table is used when the mode has two subsets. Each
 * partition is represented by a 32-bit value which gives 2 bits per texel
 * within the block. The value of the two bits represents which subset to use
 * (0 or 1).
 */
static const uint32_t
partition_table1[N_PARTITIONS] = {
   0x50505050U, 0x40404040U, 0x54545454U, 0x54505040U,
   0x50404000U, 0x55545450U, 0x55545040U, 0x54504000U,
   0x50400000U, 0x55555450U, 0x55544000U, 0x54400000U,
   0x55555440U, 0x55550000U, 0x55555500U, 0x55000000U,
   0x55150100U, 0x00004054U, 0x15010000U, 0x00405054U,
   0x00004050U, 0x15050100U, 0x05010000U, 0x40505054U,
   0x00404050U, 0x05010100U, 0x14141414U, 0x05141450U,
   0x01155440U, 0x00555500U, 0x15014054U, 0x05414150U,
   0x44444444U, 0x55005500U, 0x11441144U, 0x05055050U,
   0x05500550U, 0x11114444U, 0x41144114U, 0x44111144U,
   0x15055054U, 0x01055040U, 0x05041050U, 0x05455150U,
   0x14414114U, 0x50050550U, 0x41411414U, 0x00141400U,
   0x00041504U, 0x00105410U, 0x10541000U, 0x04150400U,
   0x50410514U, 0x41051450U, 0x05415014U, 0x14054150U,
   0x41050514U, 0x41505014U, 0x40011554U, 0x54150140U,
   0x50505500U, 0x00555050U, 0x15151010U, 0x54540404U,
};

/* This partition table is used when the mode has three subsets. In this case
 * the values can be 0, 1 or 2.
 */
static const uint32_t
partition_table2[N_PARTITIONS] = {
   0xaa685050U, 0x6a5a5040U, 0x5a5a4200U, 0x5450a0a8U,
   0xa5a50000U, 0xa0a05050U, 0x5555a0a0U, 0x5a5a5050U,
   0xaa550000U, 0xaa555500U, 0xaaaa5500U, 0x90909090U,
   0x94949494U, 0xa4a4a4a4U, 0xa9a59450U, 0x2a0a4250U,
   0xa5945040U, 0x0a425054U, 0xa5a5a500U, 0x55a0a0a0U,
   0xa8a85454U, 0x6a6a4040U, 0xa4a45000U, 0x1a1a0500U,
   0x0050a4a4U, 0xaaa59090U, 0x14696914U, 0x69691400U,
   0xa08585a0U, 0xaa821414U, 0x50a4a450U, 0x6a5a0200U,
   0xa9a58000U, 0x5090a0a8U, 0xa8a09050U, 0x24242424U,
   0x00aa5500U, 0x24924924U, 0x24499224U, 0x50a50a50U,
   0x500aa550U, 0xaaaa4444U, 0x66660000U, 0xa5a0a5a0U,
   0x50a050a0U, 0x69286928U, 0x44aaaa44U, 0x66666600U,
   0xaa444444U, 0x54a854a8U, 0x95809580U, 0x96969600U,
   0xa85454a8U, 0x80959580U, 0xaa141414U, 0x96960000U,
   0xaaaa1414U, 0xa05050a0U, 0xa0a5a5a0U, 0x96000000U,
   0x40804080U, 0xa9a8a9a8U, 0xaaaaaa44U, 0x2a4a5254U
};

static const uint8_t
anchor_indices[][N_PARTITIONS] = {
   /* Anchor index values for the second subset of two-subset partitioning */
   {
      0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,
      0xf,0x2,0x8,0x2,0x2,0x8,0x8,0xf,0x2,0x8,0x2,0x2,0x8,0x8,0x2,0x2,
      0xf,0xf,0x6,0x8,0x2,0x8,0xf,0xf,0x2,0x8,0x2,0x2,0x2,0xf,0xf,0x6,
      0x6,0x2,0x6,0x8,0xf,0xf,0x2,0x2,0xf,0xf,0xf,0xf,0xf,0x2,0x2,0xf
   },

   /* Anchor index values for the second subset of three-subset partitioning */
   {
      0x3,0x3,0xf,0xf,0x8,0x3,0xf,0xf,0x8,0x8,0x6,0x6,0x6,0x5,0x3,0x3,
      0x3,0x3,0x8,0xf,0x3,0x3,0x6,0xa,0x5,0x8,0x8,0x6,0x8,0x5,0xf,0xf,
      0x8,0xf,0x3,0x5,0x6,0xa,0x8,0xf,0xf,0x3,0xf,0x5,0xf,0xf,0xf,0xf,
      0x3,0xf,0x5,0x5,0x5,0x8,0x5,0xa,0x5,0xa,0x8,0xd,0xf,0xc,0x3,0x3
   },

   /* Anchor index values for the third subset of three-subset
    * partitioning
    */
   {
      0xf,0x8,0x8,0x3,0xf,0xf,0x3,0x8,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0x8,
      0xf,0x8,0xf,0x3,0xf,0x8,0xf,0x8,0x3,0xf,0x6,0xa,0xf,0xf,0xa,0x8,
      0xf,0x3,0xf,0xa,0xa,0x8,0x9,0xa,0x6,0xf,0x8,0xf,0x3,0x6,0x6,0x8,
      0xf,0x3,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0x3,0xf,0xf,0x8
   }
};

static int
extract_bits(const uint8_t *block,
             int offset,
             int n_bits)
{
   int byte_index = offset / 8;
   int bit_index = offset % 8;
   int n_bits_in_byte = MIN2(n_bits, 8 - bit_index);
   int result = 0;
   int bit = 0;

   while (true) {
      result |= ((block[byte_index] >> bit_index) &
                 ((1 << n_bits_in_byte) - 1)) << bit;

      n_bits -= n_bits_in_byte;

      if (n_bits <= 0)
         return result;

      bit += n_bits_in_byte;
      byte_index++;
      bit_index = 0;
      n_bits_in_byte = MIN2(n_bits, 8);
   }
}

static uint8_t
expand_component(uint8_t byte,
                 int n_bits)
{
   /* Expands a n-bit quantity into a byte by copying the most-significant
    * bits into the unused least-significant bits.
    */
   return byte << (8 - n_bits) | (byte >> (2 * n_bits - 8));
}

static int
extract_unorm_endpoints(const struct bptc_unorm_mode *mode,
                        const uint8_t *block,
                        int bit_offset,
                        uint8_t endpoints[][4])
{
   int component;
   int subset;
   int endpoint;
   int pbit;
   int n_components;

   /* Extract each color component */
   for (component = 0; component < 3; component++) {
      for (subset = 0; subset < mode->n_subsets; subset++) {
         for (endpoint = 0; endpoint < 2; endpoint++) {
            endpoints[subset * 2 + endpoint][component] =
               extract_bits(block, bit_offset, mode->n_color_bits);
            bit_offset += mode->n_color_bits;
         }
      }
   }

   /* Extract the alpha values */
   if (mode->n_alpha_bits > 0) {
      for (subset = 0; subset < mode->n_subsets; subset++) {
         for (endpoint = 0; endpoint < 2; endpoint++) {
            endpoints[subset * 2 + endpoint][3] =
               extract_bits(block, bit_offset, mode->n_alpha_bits);
            bit_offset += mode->n_alpha_bits;
         }
      }

      n_components = 4;
   } else {
      for (subset = 0; subset < mode->n_subsets; subset++)
         for (endpoint = 0; endpoint < 2; endpoint++)
            endpoints[subset * 2 + endpoint][3] = 255;

      n_components = 3;
   }

   /* Add in the p-bits */
   if (mode->has_endpoint_pbits) {
      for (subset = 0; subset < mode->n_subsets; subset++) {
         for (endpoint = 0; endpoint < 2; endpoint++) {
            pbit = extract_bits(block, bit_offset, 1);
            bit_offset += 1;

            for (component = 0; component < n_components; component++) {
               endpoints[subset * 2 + endpoint][component] <<= 1;
               endpoints[subset * 2 + endpoint][component] |= pbit;
            }
         }
      }
   } else if (mode->has_shared_pbits) {
      for (subset = 0; subset < mode->n_subsets; subset++) {
         pbit = extract_bits(block, bit_offset, 1);
         bit_offset += 1;

         for (endpoint = 0; endpoint < 2; endpoint++) {
            for (component = 0; component < n_components; component++) {
               endpoints[subset * 2 + endpoint][component] <<= 1;
               endpoints[subset * 2 + endpoint][component] |= pbit;
            }
         }
      }
   }

   /* Expand the n-bit values to a byte */
   for (subset = 0; subset < mode->n_subsets; subset++) {
      for (endpoint = 0; endpoint < 2; endpoint++) {
         for (component = 0; component < 3; component++) {
            endpoints[subset * 2 + endpoint][component] =
               expand_component(endpoints[subset * 2 + endpoint][component],
                                mode->n_color_bits +
                                mode->has_endpoint_pbits +
                                mode->has_shared_pbits);
         }

         if (mode->n_alpha_bits > 0) {
            endpoints[subset * 2 + endpoint][3] =
               expand_component(endpoints[subset * 2 + endpoint][3],
                                mode->n_alpha_bits +
                                mode->has_endpoint_pbits +
                                mode->has_shared_pbits);
         }
      }
   }

   return bit_offset;
}

static bool
is_anchor(int n_subsets,
          int partition_num,
          int texel)
{
   if (texel == 0)
      return true;

   switch (n_subsets) {
   case 1:
      return false;
   case 2:
      return anchor_indices[0][partition_num] == texel;
   case 3:
      return (anchor_indices[1][partition_num] == texel ||
              anchor_indices[2][partition_num] == texel);
   default:
      assert(false);
      return false;
   }
}

static int
count_anchors_before_texel(int n_subsets,
                           int partition_num,
                           int texel)
{
   int count = 1;

   if (texel == 0)
      return 0;

   switch (n_subsets) {
   case 1:
      break;
   case 2:
      if (texel > anchor_indices[0][partition_num])
         count++;
      break;
   case 3:
      if (texel > anchor_indices[1][partition_num])
         count++;
      if (texel > anchor_indices[2][partition_num])
         count++;
      break;
   default:
      assert(false);
      return 0;
   }

   return count;
}

static int32_t
interpolate(int32_t a, int32_t b,
            int index,
            int index_bits)
{
   static const uint8_t weights2[] = { 0, 21, 43, 64 };
   static const uint8_t weights3[] = { 0, 9, 18, 27, 37, 46, 55, 64 };
   static const uint8_t weights4[] =
      { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };
   static const uint8_t *weights[] = {
      NULL, NULL, weights2, weights3, weights4
   };
   int weight;

   weight = weights[index_bits][index];

   return ((64 - weight) * a + weight * b + 32) >> 6;
}

static void
apply_rotation(int rotation,
               uint8_t *result)
{
   uint8_t t;

   if (rotation == 0)
      return;

   rotation--;

   t = result[rotation];
   result[rotation] = result[3];
   result[3] = t;
}

static void
fetch_rgba_unorm_from_block(const uint8_t *block,
                            uint8_t *result,
                            int texel)
{
   int mode_num = ffs(block[0]);
   const struct bptc_unorm_mode *mode;
   int bit_offset, secondary_bit_offset;
   int partition_num;
   int subset_num;
   int rotation;
   int index_selection;
   int index_bits;
   int indices[2];
   int index;
   int anchors_before_texel;
   bool anchor;
   uint8_t endpoints[3 * 2][4];
   uint32_t subsets;
   int component;

   if (mode_num == 0) {
      /* According to the spec this mode is reserved and shouldn't be used. */
      memset(result, 0, 4);
      return;
   }

   mode = bptc_unorm_modes + mode_num - 1;
   bit_offset = mode_num;

   partition_num = extract_bits(block, bit_offset, mode->n_partition_bits);
   bit_offset += mode->n_partition_bits;

   switch (mode->n_subsets) {
   case 1:
      subsets = 0;
      break;
   case 2:
      subsets = partition_table1[partition_num];
      break;
   case 3:
      subsets = partition_table2[partition_num];
      break;
   default:
      assert(false);
      return;
   }

   if (mode->has_rotation_bits) {
      rotation = extract_bits(block, bit_offset, 2);
      bit_offset += 2;
   } else {
      rotation = 0;
   }

   if (mode->has_index_selection_bit) {
      index_selection = extract_bits(block, bit_offset, 1);
      bit_offset++;
   } else {
      index_selection = 0;
   }

   bit_offset = extract_unorm_endpoints(mode, block, bit_offset, endpoints);

   anchors_before_texel = count_anchors_before_texel(mode->n_subsets,
                                                     partition_num, texel);

   /* Calculate the offset to the secondary index */
   secondary_bit_offset = (bit_offset +
                           BLOCK_SIZE * BLOCK_SIZE * mode->n_index_bits -
                           mode->n_subsets +
                           mode->n_secondary_index_bits * texel -
                           anchors_before_texel);

   /* Calculate the offset to the primary index for this texel */
   bit_offset += mode->n_index_bits * texel - anchors_before_texel;

   subset_num = (subsets >> (texel * 2)) & 3;

   anchor = is_anchor(mode->n_subsets, partition_num, texel);

   index_bits = mode->n_index_bits;
   if (anchor)
      index_bits--;
   indices[0] = extract_bits(block, bit_offset, index_bits);

   if (mode->n_secondary_index_bits) {
      index_bits = mode->n_secondary_index_bits;
      if (anchor)
         index_bits--;
      indices[1] = extract_bits(block, secondary_bit_offset, index_bits);
   }

   index = indices[index_selection];
   index_bits = (index_selection ?
                 mode->n_secondary_index_bits :
                 mode->n_index_bits);

   for (component = 0; component < 3; component++)
      result[component] = interpolate(endpoints[subset_num * 2][component],
                                      endpoints[subset_num * 2 + 1][component],
                                      index,
                                      index_bits);

   /* Alpha uses the opposite index from the color components */
   if (mode->n_secondary_index_bits && !index_selection) {
      index = indices[1];
      index_bits = mode->n_secondary_index_bits;
   } else {
      index = indices[0];
      index_bits = mode->n_index_bits;
   }

   result[3] = interpolate(endpoints[subset_num * 2][3],
                           endpoints[subset_num * 2 + 1][3],
                           index,
                           index_bits);

   apply_rotation(rotation, result);
}

static void
decompress_rgba_unorm_block(int src_width, int src_height,
                            const uint8_t *block,
                            uint8_t *dst_row, int dst_rowstride)
{
   int mode_num = ffs(block[0]);
   const struct bptc_unorm_mode *mode;
   int bit_offset_head, bit_offset, secondary_bit_offset;
   int partition_num;
   int subset_num;
   int rotation;
   int index_selection;
   int index_bits;
   int indices[2];
   int index;
   int anchors_before_texel;
   bool anchor;
   uint8_t endpoints[3 * 2][4];
   uint32_t subsets;
   int component;
   unsigned x, y;

   if (mode_num == 0) {
      /* According to the spec this mode is reserved and shouldn't be used. */
      for(y = 0; y < src_height; y += 1) {
         uint8_t *result = dst_row;
         memset(result, 0, 4 * src_width);
         dst_row += dst_rowstride;
      }
      return;
   }

   mode = bptc_unorm_modes + mode_num - 1;
   bit_offset_head = mode_num;

   partition_num = extract_bits(block, bit_offset_head, mode->n_partition_bits);
   bit_offset_head += mode->n_partition_bits;

   switch (mode->n_subsets) {
   case 1:
      subsets = 0;
      break;
   case 2:
      subsets = partition_table1[partition_num];
      break;
   case 3:
      subsets = partition_table2[partition_num];
      break;
   default:
      assert(false);
      return;
   }

   if (mode->has_rotation_bits) {
      rotation = extract_bits(block, bit_offset_head, 2);
      bit_offset_head += 2;
   } else {
      rotation = 0;
   }

   if (mode->has_index_selection_bit) {
      index_selection = extract_bits(block, bit_offset_head, 1);
      bit_offset_head++;
   } else {
      index_selection = 0;
   }

   bit_offset_head = extract_unorm_endpoints(mode, block, bit_offset_head, endpoints);

   for(y = 0; y < src_height; y += 1) {
      uint8_t *result = dst_row;
      for(x = 0; x < src_width; x += 1) {
         int texel;
         texel = x + y * 4;
         bit_offset = bit_offset_head;

         anchors_before_texel = count_anchors_before_texel(mode->n_subsets,
                                                           partition_num,
                                                           texel);

         /* Calculate the offset to the secondary index */
         secondary_bit_offset = (bit_offset +
                                 BLOCK_SIZE * BLOCK_SIZE * mode->n_index_bits -
                                 mode->n_subsets +
                                 mode->n_secondary_index_bits * texel -
                                 anchors_before_texel);

         /* Calculate the offset to the primary index for this texel */
         bit_offset += mode->n_index_bits * texel - anchors_before_texel;

         subset_num = (subsets >> (texel * 2)) & 3;

         anchor = is_anchor(mode->n_subsets, partition_num, texel);

         index_bits = mode->n_index_bits;
         if (anchor)
            index_bits--;
         indices[0] = extract_bits(block, bit_offset, index_bits);

         if (mode->n_secondary_index_bits) {
            index_bits = mode->n_secondary_index_bits;
            if (anchor)
               index_bits--;
            indices[1] = extract_bits(block, secondary_bit_offset, index_bits);
         }

         index = indices[index_selection];
         index_bits = (index_selection ?
                       mode->n_secondary_index_bits :
                       mode->n_index_bits);

         for (component = 0; component < 3; component++)
            result[component] = interpolate(endpoints[subset_num * 2][component],
                                            endpoints[subset_num * 2 + 1][component],
                                            index,
                                            index_bits);

         /* Alpha uses the opposite index from the color components */
         if (mode->n_secondary_index_bits && !index_selection) {
            index = indices[1];
            index_bits = mode->n_secondary_index_bits;
         } else {
            index = indices[0];
            index_bits = mode->n_index_bits;
         }

         result[3] = interpolate(endpoints[subset_num * 2][3],
                                 endpoints[subset_num * 2 + 1][3],
                                 index,
                                 index_bits);

         apply_rotation(rotation, result);
         result += 4;
      }
      dst_row += dst_rowstride;
   }
}

static void
decompress_rgba_unorm(int width, int height,
                      const uint8_t *src, int src_rowstride,
                      uint8_t *dst, int dst_rowstride)
{
   int src_row_diff;
   int y, x;

   if (src_rowstride >= width * 4)
      src_row_diff = src_rowstride - ((width + 3) & ~3) * 4;
   else
      src_row_diff = 0;

   for (y = 0; y < height; y += BLOCK_SIZE) {
      for (x = 0; x < width; x += BLOCK_SIZE) {
         decompress_rgba_unorm_block(MIN2(width - x, BLOCK_SIZE),
                                     MIN2(height - y, BLOCK_SIZE),
                                     src,
                                     dst + x * 4 + y * dst_rowstride,
                                     dst_rowstride);
         src += BLOCK_BYTES;
      }
      src += src_row_diff;
   }
}

static int
signed_unquantize(int value, int n_endpoint_bits)
{
   bool sign;

   if (n_endpoint_bits >= 16)
      return value;

   if (value == 0)
      return 0;

   sign = false;

   if (value < 0) {
      sign = true;
      value = -value;
   }

   if (value >= (1 << (n_endpoint_bits - 1)) - 1)
      value = 0x7fff;
   else
      value = ((value << 15) + 0x4000) >> (n_endpoint_bits - 1);

   if (sign)
      value = -value;

   return value;
}

static int
unsigned_unquantize(int value, int n_endpoint_bits)
{
   if (n_endpoint_bits >= 15)
      return value;

   if (value == 0)
      return 0;

   if (value == (1 << n_endpoint_bits) - 1)
      return 0xffff;

   return ((value << 15) + 0x4000) >> (n_endpoint_bits - 1);
}

static int
extract_float_endpoints(const struct bptc_float_mode *mode,
                        const uint8_t *block,
                        int bit_offset,
                        int32_t endpoints[][3],
                        bool is_signed)
{
   const struct bptc_float_bitfield *bitfield;
   int endpoint, component;
   int n_endpoints;
   int value;
   int i;

   if (mode->n_partition_bits)
      n_endpoints = 4;
   else
      n_endpoints = 2;

   memset(endpoints, 0, sizeof endpoints[0][0] * n_endpoints * 3);

   for (bitfield = mode->bitfields; bitfield->endpoint != -1; bitfield++) {
      value = extract_bits(block, bit_offset, bitfield->n_bits);
      bit_offset += bitfield->n_bits;

      if (bitfield->reverse) {
         for (i = 0; i < bitfield->n_bits; i++) {
            if (value & (1 << i))
               endpoints[bitfield->endpoint][bitfield->component] |=
                  1 << ((bitfield->n_bits - 1 - i) + bitfield->offset);
         }
      } else {
         endpoints[bitfield->endpoint][bitfield->component] |=
            value << bitfield->offset;
      }
   }

   if (mode->transformed_endpoints) {
      /* The endpoints are specified as signed offsets from e0 */
      for (endpoint = 1; endpoint < n_endpoints; endpoint++) {
         for (component = 0; component < 3; component++) {
            value = util_sign_extend(endpoints[endpoint][component],
                                     mode->n_delta_bits[component]);
            endpoints[endpoint][component] =
               ((endpoints[0][component] + value) &
                ((1 << mode->n_endpoint_bits) - 1));
         }
      }
   }

   if (is_signed) {
      for (endpoint = 0; endpoint < n_endpoints; endpoint++) {
         for (component = 0; component < 3; component++) {
            value = util_sign_extend(endpoints[endpoint][component],
                                     mode->n_endpoint_bits);
            endpoints[endpoint][component] =
               signed_unquantize(value, mode->n_endpoint_bits);
         }
      }
   } else {
      for (endpoint = 0; endpoint < n_endpoints; endpoint++) {
         for (component = 0; component < 3; component++) {
            endpoints[endpoint][component] =
               unsigned_unquantize(endpoints[endpoint][component],
                                   mode->n_endpoint_bits);
         }
      }
   }

   return bit_offset;
}

static int32_t
finish_unsigned_unquantize(int32_t value)
{
   return value * 31 / 64;
}

static int32_t
finish_signed_unquantize(int32_t value)
{
   if (value < 0)
      return (-value * 31 / 32) | 0x8000;
   else
      return value * 31 / 32;
}

static void
fetch_rgb_float_from_block(const uint8_t *block,
                           float *result,
                           int texel,
                           bool is_signed)
{
   int mode_num;
   const struct bptc_float_mode *mode;
   int bit_offset;
   int partition_num;
   int subset_num;
   int index_bits;
   int index;
   int anchors_before_texel;
   int32_t endpoints[2 * 2][3];
   uint32_t subsets;
   int n_subsets;
   int component;
   int32_t value;

   if (block[0] & 0x2) {
      mode_num = (((block[0] >> 1) & 0xe) | (block[0] & 1)) + 2;
      bit_offset = 5;
   } else {
      mode_num = block[0] & 3;
      bit_offset = 2;
   }

   mode = bptc_float_modes + mode_num;

   if (mode->reserved) {
      memset(result, 0, sizeof result[0] * 3);
      result[3] = 1.0f;
      return;
   }

   bit_offset = extract_float_endpoints(mode, block, bit_offset,
                                        endpoints, is_signed);

   if (mode->n_partition_bits) {
      partition_num = extract_bits(block, bit_offset, mode->n_partition_bits);
      bit_offset += mode->n_partition_bits;

      subsets = partition_table1[partition_num];
      n_subsets = 2;
   } else {
      partition_num = 0;
      subsets = 0;
      n_subsets = 1;
   }

   anchors_before_texel =
      count_anchors_before_texel(n_subsets, partition_num, texel);

   /* Calculate the offset to the primary index for this texel */
   bit_offset += mode->n_index_bits * texel - anchors_before_texel;

   subset_num = (subsets >> (texel * 2)) & 3;

   index_bits = mode->n_index_bits;
   if (is_anchor(n_subsets, partition_num, texel))
      index_bits--;
   index = extract_bits(block, bit_offset, index_bits);

   for (component = 0; component < 3; component++) {
      value = interpolate(endpoints[subset_num * 2][component],
                          endpoints[subset_num * 2 + 1][component],
                          index,
                          mode->n_index_bits);

      if (is_signed)
         value = finish_signed_unquantize(value);
      else
         value = finish_unsigned_unquantize(value);

      result[component] = _mesa_half_to_float(value);
   }

   result[3] = 1.0f;
}

static void
decompress_rgb_float_block(unsigned src_width, unsigned src_height,
                           const uint8_t *block,
                           float *dst_row, unsigned dst_rowstride,
                           bool is_signed)
{
   int mode_num;
   const struct bptc_float_mode *mode;
   int bit_offset_head, bit_offset;
   int partition_num;
   int subset_num;
   int index_bits;
   int index;
   int anchors_before_texel;
   int32_t endpoints[2 * 2][3];
   uint32_t subsets;
   int n_subsets;
   int component;
   int32_t value;
   unsigned x, y;

   if (block[0] & 0x2) {
      mode_num = (((block[0] >> 1) & 0xe) | (block[0] & 1)) + 2;
      bit_offset_head = 5;
   } else {
      mode_num = block[0] & 3;
      bit_offset_head = 2;
   }

   mode = bptc_float_modes + mode_num;

   if (mode->reserved) {
      for(y = 0; y < src_height; y += 1) {
         float *result = dst_row;
         memset(result, 0, sizeof result[0] * 4 * src_width);
         for(x = 0; x < src_width; x += 1) {
            result[3] = 1.0f;
            result += 4;
         }
         dst_row += dst_rowstride / sizeof dst_row[0];
      }
      return;
   }

   bit_offset_head = extract_float_endpoints(mode, block, bit_offset_head,
                                        endpoints, is_signed);

   if (mode->n_partition_bits) {
      partition_num = extract_bits(block, bit_offset_head, mode->n_partition_bits);
      bit_offset_head += mode->n_partition_bits;

      subsets = partition_table1[partition_num];
      n_subsets = 2;
   } else {
      partition_num = 0;
      subsets = 0;
      n_subsets = 1;
   }

   for(y = 0; y < src_height; y += 1) {
      float *result = dst_row;
      for(x = 0; x < src_width; x += 1) {
         int texel;

         bit_offset = bit_offset_head;

         texel = x + y * 4;

         anchors_before_texel =
            count_anchors_before_texel(n_subsets, partition_num, texel);

         /* Calculate the offset to the primary index for this texel */
         bit_offset += mode->n_index_bits * texel - anchors_before_texel;

         subset_num = (subsets >> (texel * 2)) & 3;

         index_bits = mode->n_index_bits;
         if (is_anchor(n_subsets, partition_num, texel))
            index_bits--;
         index = extract_bits(block, bit_offset, index_bits);

         for (component = 0; component < 3; component++) {
            value = interpolate(endpoints[subset_num * 2][component],
                                endpoints[subset_num * 2 + 1][component],
                                index,
                                mode->n_index_bits);

            if (is_signed)
               value = finish_signed_unquantize(value);
            else
               value = finish_unsigned_unquantize(value);

            result[component] = _mesa_half_to_float(value);
         }

         result[3] = 1.0f;
         result += 4;
      }
      dst_row += dst_rowstride / sizeof dst_row[0];
   }
}

static void
decompress_rgb_float(int width, int height,
                      const uint8_t *src, int src_rowstride,
                      float *dst, int dst_rowstride, bool is_signed)
{
   int src_row_diff;
   int y, x;

   if (src_rowstride >= width * 4)
      src_row_diff = src_rowstride - ((width + 3) & ~3) * 4;
   else
      src_row_diff = 0;

   for (y = 0; y < height; y += BLOCK_SIZE) {
      for (x = 0; x < width; x += BLOCK_SIZE) {
         decompress_rgb_float_block(MIN2(width - x, BLOCK_SIZE),
                                    MIN2(height - y, BLOCK_SIZE),
                                    src,
                                    (dst + x * 4 +
                                     (y * dst_rowstride / sizeof dst[0])),
                                    dst_rowstride, is_signed);
         src += BLOCK_BYTES;
      }
      src += src_row_diff;
   }
}

static void
decompress_rgb_fp16_block(unsigned src_width, unsigned src_height,
                          const uint8_t *block,
                          uint16_t *dst_row, unsigned dst_rowstride,
                          bool is_signed)
{
   int mode_num;
   const struct bptc_float_mode *mode;
   int bit_offset_head, bit_offset;
   int partition_num;
   int subset_num;
   int index_bits;
   int index;
   int anchors_before_texel;
   int32_t endpoints[2 * 2][3];
   uint32_t subsets;
   int n_subsets;
   int component;
   int32_t value;
   unsigned x, y;

   if (block[0] & 0x2) {
      mode_num = (((block[0] >> 1) & 0xe) | (block[0] & 1)) + 2;
      bit_offset_head = 5;
   } else {
      mode_num = block[0] & 3;
      bit_offset_head = 2;
   }

   mode = bptc_float_modes + mode_num;

   if (mode->reserved) {
      for(y = 0; y < src_height; y += 1) {
         uint16_t *result = dst_row;
         memset(result, 0, sizeof result[0] * 4 * src_width);
         for(x = 0; x < src_width; x += 1) {
            result[3] = 1.0f;
            result += 4;
         }
         dst_row += dst_rowstride / sizeof dst_row[0];
      }
      return;
   }

   bit_offset_head = extract_float_endpoints(mode, block, bit_offset_head,
                                        endpoints, is_signed);

   if (mode->n_partition_bits) {
      partition_num = extract_bits(block, bit_offset_head, mode->n_partition_bits);
      bit_offset_head += mode->n_partition_bits;

      subsets = partition_table1[partition_num];
      n_subsets = 2;
   } else {
      partition_num = 0;
      subsets = 0;
      n_subsets = 1;
   }

   for(y = 0; y < src_height; y += 1) {
      uint16_t *result = dst_row;
      for(x = 0; x < src_width; x += 1) {
         int texel;

         bit_offset = bit_offset_head;

         texel = x + y * 4;

         anchors_before_texel =
            count_anchors_before_texel(n_subsets, partition_num, texel);

         /* Calculate the offset to the primary index for this texel */
         bit_offset += mode->n_index_bits * texel - anchors_before_texel;

         subset_num = (subsets >> (texel * 2)) & 3;

         index_bits = mode->n_index_bits;
         if (is_anchor(n_subsets, partition_num, texel))
            index_bits--;
         index = extract_bits(block, bit_offset, index_bits);

         for (component = 0; component < 3; component++) {
            value = interpolate(endpoints[subset_num * 2][component],
                                endpoints[subset_num * 2 + 1][component],
                                index,
                                mode->n_index_bits);

            if (is_signed)
               value = finish_signed_unquantize(value);
            else
               value = finish_unsigned_unquantize(value);

            result[component] = (uint16_t)value;
         }

         result[3] = FP16_ONE;
         result += 4;
      }
      dst_row += dst_rowstride / sizeof dst_row[0];
   }
}

static void
decompress_rgb_fp16(int width, int height,
                    const uint8_t *src, int src_rowstride,
                    uint16_t *dst, int dst_rowstride, bool is_signed)
{
   int src_row_diff;
   int y, x;

   if (src_rowstride >= width * 4)
      src_row_diff = src_rowstride - ((width + 3) & ~3) * 4;
   else
      src_row_diff = 0;

   for (y = 0; y < height; y += BLOCK_SIZE) {
      for (x = 0; x < width; x += BLOCK_SIZE) {
         decompress_rgb_fp16_block(MIN2(width - x, BLOCK_SIZE),
                                   MIN2(height - y, BLOCK_SIZE),
                                   src,
                                   (dst + x * 4 +
                                    (y * dst_rowstride / sizeof dst[0])),
                                   dst_rowstride, is_signed);
         src += BLOCK_BYTES;
      }
      src += src_row_diff;
   }
}

static void
write_bits(struct bit_writer *writer, int n_bits, int value)
{
   do {
      if (n_bits + writer->pos >= 8) {
         *(writer->dst++) = writer->buf | (value << writer->pos);
         writer->buf = 0;
         value >>= (8 - writer->pos);
         n_bits -= (8 - writer->pos);
         writer->pos = 0;
      } else {
         writer->buf |= value << writer->pos;
         writer->pos += n_bits;
         break;
      }
   } while (n_bits > 0);
}

static void
get_average_luminance_alpha_unorm(int width, int height,
                                  const uint8_t *src, int src_rowstride,
                                  int *average_luminance, int *average_alpha)
{
   int luminance_sum = 0, alpha_sum = 0;
   int y, x;

   for (y = 0; y < height; y++) {
      for (x = 0; x < width; x++) {
         luminance_sum += src[0] + src[1] + src[2];
         alpha_sum += src[3];
         src += 4;
      }
      src += src_rowstride - width * 4;
   }

   *average_luminance = luminance_sum / (width * height);
   *average_alpha = alpha_sum / (width * height);
}

static void
get_rgba_endpoints_unorm(int width, int height,
                         const uint8_t *src, int src_rowstride,
                         int average_luminance, int average_alpha,
                         uint8_t endpoints[][4])
{
   int endpoint_luminances[2];
   int midpoint;
   int sums[2][4];
   int endpoint;
   int luminance;
   uint8_t temp[3];
   const uint8_t *p = src;
   int rgb_left_endpoint_count = 0;
   int alpha_left_endpoint_count = 0;
   int y, x, i;

   memset(sums, 0, sizeof sums);

   for (y = 0; y < height; y++) {
      for (x = 0; x < width; x++) {
         luminance = p[0] + p[1] + p[2];
         if (luminance < average_luminance) {
            endpoint = 0;
            rgb_left_endpoint_count++;
         } else {
            endpoint = 1;
         }
         for (i = 0; i < 3; i++)
            sums[endpoint][i] += p[i];

         if (p[2] < average_alpha) {
            endpoint = 0;
            alpha_left_endpoint_count++;
         } else {
            endpoint = 1;
         }
         sums[endpoint][3] += p[3];

         p += 4;
      }

      p += src_rowstride - width * 4;
   }

   if (rgb_left_endpoint_count == 0 ||
       rgb_left_endpoint_count == width * height) {
      for (i = 0; i < 3; i++)
         endpoints[0][i] = endpoints[1][i] =
            (sums[0][i] + sums[1][i]) / (width * height);
   } else {
      for (i = 0; i < 3; i++) {
         endpoints[0][i] = sums[0][i] / rgb_left_endpoint_count;
         endpoints[1][i] = (sums[1][i] /
                            (width * height - rgb_left_endpoint_count));
      }
   }

   if (alpha_left_endpoint_count == 0 ||
       alpha_left_endpoint_count == width * height) {
      endpoints[0][3] = endpoints[1][3] =
         (sums[0][3] + sums[1][3]) / (width * height);
   } else {
         endpoints[0][3] = sums[0][3] / alpha_left_endpoint_count;
         endpoints[1][3] = (sums[1][3] /
                            (width * height - alpha_left_endpoint_count));
   }

   /* We may need to swap the endpoints to ensure the most-significant bit of
    * the first index is zero */

   for (endpoint = 0; endpoint < 2; endpoint++) {
      endpoint_luminances[endpoint] =
         endpoints[endpoint][0] +
         endpoints[endpoint][1] +
         endpoints[endpoint][2];
   }
   midpoint = (endpoint_luminances[0] + endpoint_luminances[1]) / 2;

   if ((src[0] + src[1] + src[2] <= midpoint) !=
       (endpoint_luminances[0] <= midpoint)) {
      memcpy(temp, endpoints[0], 3);
      memcpy(endpoints[0], endpoints[1], 3);
      memcpy(endpoints[1], temp, 3);
   }

   /* Same for the alpha endpoints */

   midpoint = (endpoints[0][3] + endpoints[1][3]) / 2;

   if ((src[3] <= midpoint) != (endpoints[0][3] <= midpoint)) {
      temp[0] = endpoints[0][3];
      endpoints[0][3] = endpoints[1][3];
      endpoints[1][3] = temp[0];
   }
}

static void
write_rgb_indices_unorm(struct bit_writer *writer,
                        int src_width, int src_height,
                        const uint8_t *src, int src_rowstride,
                        uint8_t endpoints[][4])
{
   int luminance;
   int endpoint_luminances[2];
   int endpoint;
   int index;
   int y, x;

   for (endpoint = 0; endpoint < 2; endpoint++) {
      endpoint_luminances[endpoint] =
         endpoints[endpoint][0] +
         endpoints[endpoint][1] +
         endpoints[endpoint][2];
   }

   /* If the endpoints have the same luminance then we'll just use index 0 for
    * all of the texels */
   if (endpoint_luminances[0] == endpoint_luminances[1]) {
      write_bits(writer, BLOCK_SIZE * BLOCK_SIZE * 2 - 1, 0);
      return;
   }

   for (y = 0; y < src_height; y++) {
      for (x = 0; x < src_width; x++) {
         luminance = src[0] + src[1] + src[2];

         index = ((luminance - endpoint_luminances[0]) * 3 /
                  (endpoint_luminances[1] - endpoint_luminances[0]));
         if (index < 0)
            index = 0;
         else if (index > 3)
            index = 3;

         assert(x != 0 || y != 0 || index < 2);

         write_bits(writer, (x == 0 && y == 0) ? 1 : 2, index);

         src += 4;
      }

      /* Pad the indices out to the block size */
      if (src_width < BLOCK_SIZE)
         write_bits(writer, 2 * (BLOCK_SIZE - src_width), 0);

      src += src_rowstride - src_width * 4;
   }

   /* Pad the indices out to the block size */
   if (src_height < BLOCK_SIZE)
      write_bits(writer, 2 * BLOCK_SIZE * (BLOCK_SIZE - src_height), 0);
}

static void
write_alpha_indices_unorm(struct bit_writer *writer,
                          int src_width, int src_height,
                          const uint8_t *src, int src_rowstride,
                          uint8_t endpoints[][4])
{
   int index;
   int y, x;

   /* If the endpoints have the same alpha then we'll just use index 0 for
    * all of the texels */
   if (endpoints[0][3] == endpoints[1][3]) {
      write_bits(writer, BLOCK_SIZE * BLOCK_SIZE * 3 - 1, 0);
      return;
   }

   for (y = 0; y < src_height; y++) {
      for (x = 0; x < src_width; x++) {
         index = (((int) src[3] - (int) endpoints[0][3]) * 7 /
                  ((int) endpoints[1][3] - endpoints[0][3]));
         if (index < 0)
            index = 0;
         else if (index > 7)
            index = 7;

         assert(x != 0 || y != 0 || index < 4);

         /* The first index has one less bit */
         write_bits(writer, (x == 0 && y == 0) ? 2 : 3, index);

         src += 4;
      }

      /* Pad the indices out to the block size */
      if (src_width < BLOCK_SIZE)
         write_bits(writer, 3 * (BLOCK_SIZE - src_width), 0);

      src += src_rowstride - src_width * 4;
   }

   /* Pad the indices out to the block size */
   if (src_height < BLOCK_SIZE)
      write_bits(writer, 3 * BLOCK_SIZE * (BLOCK_SIZE - src_height), 0);
}

static void
compress_rgba_unorm_block(int src_width, int src_height,
                          const uint8_t *src, int src_rowstride,
                          uint8_t *dst)
{
   int average_luminance, average_alpha;
   uint8_t endpoints[2][4];
   struct bit_writer writer;
   int component, endpoint;

   get_average_luminance_alpha_unorm(src_width, src_height, src, src_rowstride,
                                     &average_luminance, &average_alpha);
   get_rgba_endpoints_unorm(src_width, src_height, src, src_rowstride,
                            average_luminance, average_alpha,
                            endpoints);

   writer.dst = dst;
   writer.pos = 0;
   writer.buf = 0;

   write_bits(&writer, 5, 0x10); /* mode 4 */
   write_bits(&writer, 2, 0); /* rotation 0 */
   write_bits(&writer, 1, 0); /* index selection bit */

   /* Write the color endpoints */
   for (component = 0; component < 3; component++)
      for (endpoint = 0; endpoint < 2; endpoint++)
         write_bits(&writer, 5, endpoints[endpoint][component] >> 3);

   /* Write the alpha endpoints */
   for (endpoint = 0; endpoint < 2; endpoint++)
      write_bits(&writer, 6, endpoints[endpoint][3] >> 2);

   write_rgb_indices_unorm(&writer,
                           src_width, src_height,
                           src, src_rowstride,
                           endpoints);
   write_alpha_indices_unorm(&writer,
                             src_width, src_height,
                             src, src_rowstride,
                             endpoints);
}

static void
compress_rgba_unorm(int width, int height,
                    const uint8_t *src, int src_rowstride,
                    uint8_t *dst, int dst_rowstride)
{
   int dst_row_diff;
   int y, x;

   if (dst_rowstride >= width * 4)
      dst_row_diff = dst_rowstride - ((width + 3) & ~3) * 4;
   else
      dst_row_diff = 0;

   for (y = 0; y < height; y += BLOCK_SIZE) {
      for (x = 0; x < width; x += BLOCK_SIZE) {
         compress_rgba_unorm_block(MIN2(width - x, BLOCK_SIZE),
                                   MIN2(height - y, BLOCK_SIZE),
                                   src + x * 4 + y * src_rowstride,
                                   src_rowstride,
                                   dst);
         dst += BLOCK_BYTES;
      }
      dst += dst_row_diff;
   }
}

static float
get_average_luminance_float(int width, int height,
                            const float *src, int src_rowstride)
{
   float luminance_sum = 0;
   int y, x;

   for (y = 0; y < height; y++) {
      for (x = 0; x < width; x++) {
         luminance_sum += src[0] + src[1] + src[2];
         src += 3;
      }
      src += (src_rowstride - width * 3 * sizeof (float)) / sizeof (float);
   }

   return luminance_sum / (width * height);
}

static float
clamp_value(float value, bool is_signed)
{
   if (value > 65504.0f)
      return 65504.0f;

   if (is_signed) {
      if (value < -65504.0f)
         return -65504.0f;
      else
         return value;
   }

   if (value < 0.0f)
      return 0.0f;

   return value;
}

static void
get_endpoints_float(int width, int height,
                    const float *src, int src_rowstride,
                    float average_luminance, float endpoints[][3],
                    bool is_signed)
{
   float endpoint_luminances[2];
   float midpoint;
   float sums[2][3];
   int endpoint, component;
   float luminance;
   float temp[3];
   const float *p = src;
   int left_endpoint_count = 0;
   int y, x, i;

   memset(sums, 0, sizeof sums);

   for (y = 0; y < height; y++) {
      for (x = 0; x < width; x++) {
         luminance = p[0] + p[1] + p[2];
         if (luminance < average_luminance) {
            endpoint = 0;
            left_endpoint_count++;
         } else {
            endpoint = 1;
         }
         for (i = 0; i < 3; i++)
            sums[endpoint][i] += p[i];

         p += 3;
      }

      p += (src_rowstride - width * 3 * sizeof (float)) / sizeof (float);
   }

   if (left_endpoint_count == 0 ||
       left_endpoint_count == width * height) {
      for (i = 0; i < 3; i++)
         endpoints[0][i] = endpoints[1][i] =
            (sums[0][i] + sums[1][i]) / (width * height);
   } else {
      for (i = 0; i < 3; i++) {
         endpoints[0][i] = sums[0][i] / left_endpoint_count;
         endpoints[1][i] = sums[1][i] / (width * height - left_endpoint_count);
      }
   }

   /* Clamp the endpoints to the range of a half float and strip out
    * infinities */
   for (endpoint = 0; endpoint < 2; endpoint++) {
      for (component = 0; component < 3; component++) {
         endpoints[endpoint][component] =
            clamp_value(endpoints[endpoint][component], is_signed);
      }
   }

   /* We may need to swap the endpoints to ensure the most-significant bit of
    * the first index is zero */

   for (endpoint = 0; endpoint < 2; endpoint++) {
      endpoint_luminances[endpoint] =
         endpoints[endpoint][0] +
         endpoints[endpoint][1] +
         endpoints[endpoint][2];
   }
   midpoint = (endpoint_luminances[0] + endpoint_luminances[1]) / 2.0f;

   if ((src[0] + src[1] + src[2] <= midpoint) !=
       (endpoint_luminances[0] <= midpoint)) {
      memcpy(temp, endpoints[0], sizeof temp);
      memcpy(endpoints[0], endpoints[1], sizeof temp);
      memcpy(endpoints[1], temp, sizeof temp);
   }
}

static void
write_rgb_indices_float(struct bit_writer *writer,
                        int src_width, int src_height,
                        const float *src, int src_rowstride,
                        float endpoints[][3])
{
   float luminance;
   float endpoint_luminances[2];
   int endpoint;
   int index;
   int y, x;

   for (endpoint = 0; endpoint < 2; endpoint++) {
      endpoint_luminances[endpoint] =
         endpoints[endpoint][0] +
         endpoints[endpoint][1] +
         endpoints[endpoint][2];
   }

   /* If the endpoints have the same luminance then we'll just use index 0 for
    * all of the texels */
   if (endpoint_luminances[0] == endpoint_luminances[1]) {
      write_bits(writer, BLOCK_SIZE * BLOCK_SIZE * 4 - 1, 0);
      return;
   }

   for (y = 0; y < src_height; y++) {
      for (x = 0; x < src_width; x++) {
         luminance = src[0] + src[1] + src[2];

         index = ((luminance - endpoint_luminances[0]) * 15 /
                  (endpoint_luminances[1] - endpoint_luminances[0]));
         if (index < 0)
            index = 0;
         else if (index > 15)
            index = 15;

         assert(x != 0 || y != 0 || index < 8);

         write_bits(writer, (x == 0 && y == 0) ? 3 : 4, index);

         src += 3;
      }

      /* Pad the indices out to the block size */
      if (src_width < BLOCK_SIZE)
         write_bits(writer, 4 * (BLOCK_SIZE - src_width), 0);

      src += (src_rowstride - src_width * 3 * sizeof (float)) / sizeof (float);
   }

   /* Pad the indices out to the block size */
   if (src_height < BLOCK_SIZE)
      write_bits(writer, 4 * BLOCK_SIZE * (BLOCK_SIZE - src_height), 0);
}

static int
get_endpoint_value(float value, bool is_signed)
{
   bool sign = false;
   int half;

   if (is_signed) {
      half = _mesa_float_to_half(value);

      if (half & 0x8000) {
         half &= 0x7fff;
         sign = true;
      }

      half = (32 * half / 31) >> 6;

      if (sign)
         half = -half & ((1 << 10) - 1);

      return half;
   } else {
      if (value <= 0.0f)
         return 0;

      half = _mesa_float_to_half(value);

      return (64 * half / 31) >> 6;
   }
}

static void
compress_rgb_float_block(int src_width, int src_height,
                         const float *src, int src_rowstride,
                         uint8_t *dst,
                         bool is_signed)
{
   float average_luminance;
   float endpoints[2][3];
   struct bit_writer writer;
   int component, endpoint;
   int endpoint_value;

   average_luminance =
      get_average_luminance_float(src_width, src_height, src, src_rowstride);
   get_endpoints_float(src_width, src_height, src, src_rowstride,
                       average_luminance, endpoints, is_signed);

   writer.dst = dst;
   writer.pos = 0;
   writer.buf = 0;

   write_bits(&writer, 5, 3); /* mode 3 */

   /* Write the endpoints */
   for (endpoint = 0; endpoint < 2; endpoint++) {
      for (component = 0; component < 3; component++) {
         endpoint_value =
            get_endpoint_value(endpoints[endpoint][component], is_signed);
         write_bits(&writer, 10, endpoint_value);
      }
   }

   write_rgb_indices_float(&writer,
                           src_width, src_height,
                           src, src_rowstride,
                           endpoints);
}

static void
compress_rgb_float(int width, int height,
                   const float *src, int src_rowstride,
                   uint8_t *dst, int dst_rowstride,
                   bool is_signed)
{
   int dst_row_diff;
   int y, x;

   if (dst_rowstride >= width * 4)
      dst_row_diff = dst_rowstride - ((width + 3) & ~3) * 4;
   else
      dst_row_diff = 0;

   for (y = 0; y < height; y += BLOCK_SIZE) {
      for (x = 0; x < width; x += BLOCK_SIZE) {
         compress_rgb_float_block(MIN2(width - x, BLOCK_SIZE),
                                  MIN2(height - y, BLOCK_SIZE),
                                  src + x * 3 +
                                  y * src_rowstride / sizeof (float),
                                  src_rowstride,
                                  dst,
                                  is_signed);
         dst += BLOCK_BYTES;
      }
      dst += dst_row_diff;
   }
}

#endif
