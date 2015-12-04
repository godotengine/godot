// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Spatial prediction using various filters
//
// Author(s): Branimir Vasic (branimir.vasic@imgtec.com)
//            Djordje Pesut (djordje.pesut@imgtec.com)

#include "./dsp.h"

#if defined(WEBP_USE_MIPS_DSP_R2)

#include "../dsp/dsp.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

//------------------------------------------------------------------------------
// Helpful macro.

# define SANITY_CHECK(in, out)                                                 \
  assert(in != NULL);                                                          \
  assert(out != NULL);                                                         \
  assert(width > 0);                                                           \
  assert(height > 0);                                                          \
  assert(stride >= width);                                                     \
  assert(row >= 0 && num_rows > 0 && row + num_rows <= height);                \
  (void)height;  // Silence unused warning.

// if INVERSE
//   preds == &dst[-1] == &src[-1]
// else
//   preds == &src[-1] != &dst[-1]
#define DO_PREDICT_LINE(SRC, DST, LENGTH, INVERSE) do {                        \
    const uint8_t* psrc = (uint8_t*)(SRC);                                     \
    uint8_t* pdst = (uint8_t*)(DST);                                           \
    const int ilength = (int)(LENGTH);                                         \
    int temp0, temp1, temp2, temp3, temp4, temp5, temp6;                       \
    __asm__ volatile (                                                         \
      ".set      push                                   \n\t"                  \
      ".set      noreorder                              \n\t"                  \
      "srl       %[temp0],    %[length],    0x2         \n\t"                  \
      "beqz      %[temp0],    4f                        \n\t"                  \
      " andi     %[temp6],    %[length],    0x3         \n\t"                  \
    ".if " #INVERSE "                                   \n\t"                  \
      "lbu       %[temp1],    -1(%[src])                \n\t"                  \
    "1:                                                 \n\t"                  \
      "lbu       %[temp2],    0(%[src])                 \n\t"                  \
      "lbu       %[temp3],    1(%[src])                 \n\t"                  \
      "lbu       %[temp4],    2(%[src])                 \n\t"                  \
      "lbu       %[temp5],    3(%[src])                 \n\t"                  \
      "addiu     %[src],      %[src],       4           \n\t"                  \
      "addiu     %[temp0],    %[temp0],     -1          \n\t"                  \
      "addu      %[temp2],    %[temp2],     %[temp1]    \n\t"                  \
      "addu      %[temp3],    %[temp3],     %[temp2]    \n\t"                  \
      "addu      %[temp4],    %[temp4],     %[temp3]    \n\t"                  \
      "addu      %[temp1],    %[temp5],     %[temp4]    \n\t"                  \
      "sb        %[temp2],    -4(%[src])                \n\t"                  \
      "sb        %[temp3],    -3(%[src])                \n\t"                  \
      "sb        %[temp4],    -2(%[src])                \n\t"                  \
      "bnez      %[temp0],    1b                        \n\t"                  \
      " sb       %[temp1],    -1(%[src])                \n\t"                  \
    ".else                                              \n\t"                  \
    "1:                                                 \n\t"                  \
      "ulw       %[temp1],    -1(%[src])                \n\t"                  \
      "ulw       %[temp2],    0(%[src])                 \n\t"                  \
      "addiu     %[src],      %[src],       4           \n\t"                  \
      "addiu     %[temp0],    %[temp0],     -1          \n\t"                  \
      "subu.qb   %[temp3],    %[temp2],     %[temp1]    \n\t"                  \
      "usw       %[temp3],    0(%[dst])                 \n\t"                  \
      "bnez      %[temp0],    1b                        \n\t"                  \
      " addiu    %[dst],      %[dst],       4           \n\t"                  \
    ".endif                                             \n\t"                  \
    "4:                                                 \n\t"                  \
      "beqz      %[temp6],    3f                        \n\t"                  \
      " nop                                             \n\t"                  \
    "2:                                                 \n\t"                  \
      "lbu       %[temp1],    -1(%[src])                \n\t"                  \
      "lbu       %[temp2],    0(%[src])                 \n\t"                  \
      "addiu     %[src],      %[src],       1           \n\t"                  \
    ".if " #INVERSE "                                   \n\t"                  \
      "addu      %[temp3],    %[temp1],     %[temp2]    \n\t"                  \
      "sb        %[temp3],    -1(%[src])                \n\t"                  \
    ".else                                              \n\t"                  \
      "subu      %[temp3],    %[temp1],     %[temp2]    \n\t"                  \
      "sb        %[temp3],    0(%[dst])                 \n\t"                  \
    ".endif                                             \n\t"                  \
      "addiu     %[temp6],    %[temp6],     -1          \n\t"                  \
      "bnez      %[temp6],    2b                        \n\t"                  \
      " addiu    %[dst],      %[dst],       1           \n\t"                  \
    "3:                                                 \n\t"                  \
      ".set      pop                                    \n\t"                  \
      : [temp0]"=&r"(temp0), [temp1]"=&r"(temp1), [temp2]"=&r"(temp2),         \
        [temp3]"=&r"(temp3), [temp4]"=&r"(temp4), [temp5]"=&r"(temp5),         \
        [temp6]"=&r"(temp6), [dst]"+&r"(pdst), [src]"+&r"(psrc)                \
      : [length]"r"(ilength)                                                   \
      : "memory"                                                               \
    );                                                                         \
  } while (0)

static WEBP_INLINE void PredictLine(const uint8_t* src, uint8_t* dst,
                                    int length, int inverse) {
  if (inverse) {
    DO_PREDICT_LINE(src, dst, length, 1);
  } else {
    DO_PREDICT_LINE(src, dst, length, 0);
  }
}

#define DO_PREDICT_LINE_VERTICAL(SRC, PRED, DST, LENGTH, INVERSE) do {         \
    const uint8_t* psrc = (uint8_t*)(SRC);                                     \
    const uint8_t* ppred = (uint8_t*)(PRED);                                   \
    uint8_t* pdst = (uint8_t*)(DST);                                           \
    const int ilength = (int)(LENGTH);                                         \
    int temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;                \
    __asm__ volatile (                                                         \
      ".set      push                                   \n\t"                  \
      ".set      noreorder                              \n\t"                  \
      "srl       %[temp0],    %[length],    0x3         \n\t"                  \
      "beqz      %[temp0],    4f                        \n\t"                  \
      " andi     %[temp7],    %[length],    0x7         \n\t"                  \
    "1:                                                 \n\t"                  \
      "ulw       %[temp1],    0(%[src])                 \n\t"                  \
      "ulw       %[temp2],    0(%[pred])                \n\t"                  \
      "ulw       %[temp3],    4(%[src])                 \n\t"                  \
      "ulw       %[temp4],    4(%[pred])                \n\t"                  \
      "addiu     %[src],      %[src],       8           \n\t"                  \
    ".if " #INVERSE "                                   \n\t"                  \
      "addu.qb   %[temp5],    %[temp1],     %[temp2]    \n\t"                  \
      "addu.qb   %[temp6],    %[temp3],     %[temp4]    \n\t"                  \
    ".else                                              \n\t"                  \
      "subu.qb   %[temp5],    %[temp1],     %[temp2]    \n\t"                  \
      "subu.qb   %[temp6],    %[temp3],     %[temp4]    \n\t"                  \
    ".endif                                             \n\t"                  \
      "addiu     %[pred],     %[pred],      8           \n\t"                  \
      "usw       %[temp5],    0(%[dst])                 \n\t"                  \
      "usw       %[temp6],    4(%[dst])                 \n\t"                  \
      "addiu     %[temp0],    %[temp0],     -1          \n\t"                  \
      "bnez      %[temp0],    1b                        \n\t"                  \
      " addiu    %[dst],      %[dst],       8           \n\t"                  \
    "4:                                                 \n\t"                  \
      "beqz      %[temp7],    3f                        \n\t"                  \
      " nop                                             \n\t"                  \
    "2:                                                 \n\t"                  \
      "lbu       %[temp1],    0(%[src])                 \n\t"                  \
      "lbu       %[temp2],    0(%[pred])                \n\t"                  \
      "addiu     %[src],      %[src],       1           \n\t"                  \
      "addiu     %[pred],     %[pred],      1           \n\t"                  \
    ".if " #INVERSE "                                   \n\t"                  \
      "addu      %[temp3],    %[temp1],     %[temp2]    \n\t"                  \
    ".else                                              \n\t"                  \
      "subu      %[temp3],    %[temp1],     %[temp2]    \n\t"                  \
    ".endif                                             \n\t"                  \
      "sb        %[temp3],    0(%[dst])                 \n\t"                  \
      "addiu     %[temp7],    %[temp7],     -1          \n\t"                  \
      "bnez      %[temp7],    2b                        \n\t"                  \
      " addiu    %[dst],      %[dst],       1           \n\t"                  \
    "3:                                                 \n\t"                  \
      ".set      pop                                    \n\t"                  \
      : [temp0]"=&r"(temp0), [temp1]"=&r"(temp1), [temp2]"=&r"(temp2),         \
        [temp3]"=&r"(temp3), [temp4]"=&r"(temp4), [temp5]"=&r"(temp5),         \
        [temp6]"=&r"(temp6), [temp7]"=&r"(temp7), [pred]"+&r"(ppred),          \
        [dst]"+&r"(pdst), [src]"+&r"(psrc)                                     \
      : [length]"r"(ilength)                                                   \
      : "memory"                                                               \
    );                                                                         \
  } while (0)

#define PREDICT_LINE_ONE_PASS(SRC, PRED, DST, INVERSE) do {                    \
    int temp1, temp2, temp3;                                                   \
    __asm__ volatile (                                                         \
      "lbu       %[temp1],   0(%[src])               \n\t"                     \
      "lbu       %[temp2],   0(%[pred])              \n\t"                     \
    ".if " #INVERSE "                                \n\t"                     \
      "addu      %[temp3],   %[temp1],   %[temp2]    \n\t"                     \
    ".else                                           \n\t"                     \
      "subu      %[temp3],   %[temp1],   %[temp2]    \n\t"                     \
    ".endif                                          \n\t"                     \
      "sb        %[temp3],   0(%[dst])               \n\t"                     \
      : [temp1]"=&r"(temp1), [temp2]"=&r"(temp2), [temp3]"=&r"(temp3)          \
      : [pred]"r"((PRED)), [dst]"r"((DST)), [src]"r"((SRC))                    \
      : "memory"                                                               \
    );                                                                         \
  } while (0)

//------------------------------------------------------------------------------
// Horizontal filter.

#define FILTER_LINE_BY_LINE(INVERSE) do {                                      \
    while (row < last_row) {                                                   \
      PREDICT_LINE_ONE_PASS(in, preds - stride, out, INVERSE);                 \
      DO_PREDICT_LINE(in + 1, out + 1, width - 1, INVERSE);                    \
      ++row;                                                                   \
      preds += stride;                                                         \
      in += stride;                                                            \
      out += stride;                                                           \
    }                                                                          \
  } while (0)

static WEBP_INLINE void DoHorizontalFilter(const uint8_t* in,
                                           int width, int height, int stride,
                                           int row, int num_rows,
                                           int inverse, uint8_t* out) {
  const uint8_t* preds;
  const size_t start_offset = row * stride;
  const int last_row = row + num_rows;
  SANITY_CHECK(in, out);
  in += start_offset;
  out += start_offset;
  preds = inverse ? out : in;

  if (row == 0) {
    // Leftmost pixel is the same as input for topmost scanline.
    out[0] = in[0];
    PredictLine(in + 1, out + 1, width - 1, inverse);
    row = 1;
    preds += stride;
    in += stride;
    out += stride;
  }

  // Filter line-by-line.
  if (inverse) {
    FILTER_LINE_BY_LINE(1);
  } else {
    FILTER_LINE_BY_LINE(0);
  }
}

#undef FILTER_LINE_BY_LINE

static void HorizontalFilter(const uint8_t* data, int width, int height,
                             int stride, uint8_t* filtered_data) {
  DoHorizontalFilter(data, width, height, stride, 0, height, 0, filtered_data);
}

static void HorizontalUnfilter(int width, int height, int stride, int row,
                               int num_rows, uint8_t* data) {
  DoHorizontalFilter(data, width, height, stride, row, num_rows, 1, data);
}

//------------------------------------------------------------------------------
// Vertical filter.

#define FILTER_LINE_BY_LINE(INVERSE) do {                                      \
    while (row < last_row) {                                                   \
      DO_PREDICT_LINE_VERTICAL(in, preds, out, width, INVERSE);                \
      ++row;                                                                   \
      preds += stride;                                                         \
      in += stride;                                                            \
      out += stride;                                                           \
    }                                                                          \
  } while (0)

static WEBP_INLINE void DoVerticalFilter(const uint8_t* in,
                                         int width, int height, int stride,
                                         int row, int num_rows,
                                         int inverse, uint8_t* out) {
  const uint8_t* preds;
  const size_t start_offset = row * stride;
  const int last_row = row + num_rows;
  SANITY_CHECK(in, out);
  in += start_offset;
  out += start_offset;
  preds = inverse ? out : in;

  if (row == 0) {
    // Very first top-left pixel is copied.
    out[0] = in[0];
    // Rest of top scan-line is left-predicted.
    PredictLine(in + 1, out + 1, width - 1, inverse);
    row = 1;
    in += stride;
    out += stride;
  } else {
    // We are starting from in-between. Make sure 'preds' points to prev row.
    preds -= stride;
  }

  // Filter line-by-line.
  if (inverse) {
    FILTER_LINE_BY_LINE(1);
  } else {
    FILTER_LINE_BY_LINE(0);
  }
}

#undef FILTER_LINE_BY_LINE
#undef DO_PREDICT_LINE_VERTICAL

static void VerticalFilter(const uint8_t* data, int width, int height,
                           int stride, uint8_t* filtered_data) {
  DoVerticalFilter(data, width, height, stride, 0, height, 0, filtered_data);
}

static void VerticalUnfilter(int width, int height, int stride, int row,
                             int num_rows, uint8_t* data) {
  DoVerticalFilter(data, width, height, stride, row, num_rows, 1, data);
}

//------------------------------------------------------------------------------
// Gradient filter.

static WEBP_INLINE int GradientPredictor(uint8_t a, uint8_t b, uint8_t c) {
  int temp0;
  __asm__ volatile (
    "addu             %[temp0],   %[a],       %[b]        \n\t"
    "subu             %[temp0],   %[temp0],   %[c]        \n\t"
    "shll_s.w         %[temp0],   %[temp0],   23          \n\t"
    "precrqu_s.qb.ph  %[temp0],   %[temp0],   $zero       \n\t"
    "srl              %[temp0],   %[temp0],   24          \n\t"
    : [temp0]"=&r"(temp0)
    : [a]"r"(a),[b]"r"(b),[c]"r"(c)
  );
  return temp0;
}

#define FILTER_LINE_BY_LINE(INVERSE, PREDS, OPERATION) do {                    \
    while (row < last_row) {                                                   \
      int w;                                                                   \
      PREDICT_LINE_ONE_PASS(in, PREDS - stride, out, INVERSE);                 \
      for (w = 1; w < width; ++w) {                                            \
        const int pred = GradientPredictor(PREDS[w - 1],                       \
                                           PREDS[w - stride],                  \
                                           PREDS[w - stride - 1]);             \
        out[w] = in[w] OPERATION pred;                                         \
      }                                                                        \
      ++row;                                                                   \
      in += stride;                                                            \
      out += stride;                                                           \
    }                                                                          \
  } while (0)

static WEBP_INLINE void DoGradientFilter(const uint8_t* in,
                                         int width, int height, int stride,
                                         int row, int num_rows,
                                         int inverse, uint8_t* out) {
  const uint8_t* preds;
  const size_t start_offset = row * stride;
  const int last_row = row + num_rows;
  SANITY_CHECK(in, out);
  in += start_offset;
  out += start_offset;
  preds = inverse ? out : in;

  // left prediction for top scan-line
  if (row == 0) {
    out[0] = in[0];
    PredictLine(in + 1, out + 1, width - 1, inverse);
    row = 1;
    preds += stride;
    in += stride;
    out += stride;
  }

  // Filter line-by-line.
  if (inverse) {
    FILTER_LINE_BY_LINE(1, out, +);
  } else {
    FILTER_LINE_BY_LINE(0, in, -);
  }
}

#undef FILTER_LINE_BY_LINE

static void GradientFilter(const uint8_t* data, int width, int height,
                           int stride, uint8_t* filtered_data) {
  DoGradientFilter(data, width, height, stride, 0, height, 0, filtered_data);
}

static void GradientUnfilter(int width, int height, int stride, int row,
                             int num_rows, uint8_t* data) {
  DoGradientFilter(data, width, height, stride, row, num_rows, 1, data);
}

#undef PREDICT_LINE_ONE_PASS
#undef DO_PREDICT_LINE
#undef SANITY_CHECK

//------------------------------------------------------------------------------
// Entry point

extern void VP8FiltersInitMIPSdspR2(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8FiltersInitMIPSdspR2(void) {
  WebPFilters[WEBP_FILTER_HORIZONTAL] = HorizontalFilter;
  WebPFilters[WEBP_FILTER_VERTICAL] = VerticalFilter;
  WebPFilters[WEBP_FILTER_GRADIENT] = GradientFilter;

  WebPUnfilters[WEBP_FILTER_HORIZONTAL] = HorizontalUnfilter;
  WebPUnfilters[WEBP_FILTER_VERTICAL] = VerticalUnfilter;
  WebPUnfilters[WEBP_FILTER_GRADIENT] = GradientUnfilter;
}

#else  // !WEBP_USE_MIPS_DSP_R2

WEBP_DSP_INIT_STUB(VP8FiltersInitMIPSdspR2)

#endif  // WEBP_USE_MIPS_DSP_R2
