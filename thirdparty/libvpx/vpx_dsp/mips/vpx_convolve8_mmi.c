/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <string.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/vpx_filter.h"
#include "vpx_ports/asmdefs_mmi.h"
#include "vpx_ports/mem.h"

#define GET_DATA_H_MMI                                     \
  "pmaddhw    %[ftmp4],    %[ftmp4],   %[filter1]    \n\t" \
  "pmaddhw    %[ftmp5],    %[ftmp5],   %[filter2]    \n\t" \
  "paddw      %[ftmp4],    %[ftmp4],   %[ftmp5]      \n\t" \
  "punpckhwd  %[ftmp5],    %[ftmp4],   %[ftmp0]      \n\t" \
  "paddw      %[ftmp4],    %[ftmp4],   %[ftmp5]      \n\t" \
  "pmaddhw    %[ftmp6],    %[ftmp6],   %[filter1]    \n\t" \
  "pmaddhw    %[ftmp7],    %[ftmp7],   %[filter2]    \n\t" \
  "paddw      %[ftmp6],    %[ftmp6],   %[ftmp7]      \n\t" \
  "punpckhwd  %[ftmp7],    %[ftmp6],   %[ftmp0]      \n\t" \
  "paddw      %[ftmp6],    %[ftmp6],   %[ftmp7]      \n\t" \
  "punpcklwd  %[srcl],     %[ftmp4],   %[ftmp6]      \n\t" \
  "pmaddhw    %[ftmp8],    %[ftmp8],   %[filter1]    \n\t" \
  "pmaddhw    %[ftmp9],    %[ftmp9],   %[filter2]    \n\t" \
  "paddw      %[ftmp8],    %[ftmp8],   %[ftmp9]      \n\t" \
  "punpckhwd  %[ftmp9],    %[ftmp8],   %[ftmp0]      \n\t" \
  "paddw      %[ftmp8],    %[ftmp8],   %[ftmp9]      \n\t" \
  "pmaddhw    %[ftmp10],   %[ftmp10],  %[filter1]    \n\t" \
  "pmaddhw    %[ftmp11],   %[ftmp11],  %[filter2]    \n\t" \
  "paddw      %[ftmp10],   %[ftmp10],  %[ftmp11]     \n\t" \
  "punpckhwd  %[ftmp11],   %[ftmp10],  %[ftmp0]      \n\t" \
  "paddw      %[ftmp10],   %[ftmp10],  %[ftmp11]     \n\t" \
  "punpcklwd  %[srch],     %[ftmp8],   %[ftmp10]     \n\t"

#define GET_DATA_V_MMI                                     \
  "punpcklhw  %[srcl],     %[ftmp4],   %[ftmp5]      \n\t" \
  "pmaddhw    %[srcl],     %[srcl],    %[filter10]   \n\t" \
  "punpcklhw  %[ftmp12],   %[ftmp6],   %[ftmp7]      \n\t" \
  "pmaddhw    %[ftmp12],   %[ftmp12],  %[filter32]   \n\t" \
  "paddw      %[srcl],     %[srcl],    %[ftmp12]     \n\t" \
  "punpcklhw  %[ftmp12],   %[ftmp8],   %[ftmp9]      \n\t" \
  "pmaddhw    %[ftmp12],   %[ftmp12],  %[filter54]   \n\t" \
  "paddw      %[srcl],     %[srcl],    %[ftmp12]     \n\t" \
  "punpcklhw  %[ftmp12],   %[ftmp10],  %[ftmp11]     \n\t" \
  "pmaddhw    %[ftmp12],   %[ftmp12],  %[filter76]   \n\t" \
  "paddw      %[srcl],     %[srcl],    %[ftmp12]     \n\t" \
  "punpckhhw  %[srch],     %[ftmp4],   %[ftmp5]      \n\t" \
  "pmaddhw    %[srch],     %[srch],    %[filter10]   \n\t" \
  "punpckhhw  %[ftmp12],   %[ftmp6],   %[ftmp7]      \n\t" \
  "pmaddhw    %[ftmp12],   %[ftmp12],  %[filter32]   \n\t" \
  "paddw      %[srch],     %[srch],    %[ftmp12]     \n\t" \
  "punpckhhw  %[ftmp12],   %[ftmp8],   %[ftmp9]      \n\t" \
  "pmaddhw    %[ftmp12],   %[ftmp12],  %[filter54]   \n\t" \
  "paddw      %[srch],     %[srch],    %[ftmp12]     \n\t" \
  "punpckhhw  %[ftmp12],   %[ftmp10],  %[ftmp11]     \n\t" \
  "pmaddhw    %[ftmp12],   %[ftmp12],  %[filter76]   \n\t" \
  "paddw      %[srch],     %[srch],    %[ftmp12]     \n\t"

/* clang-format off */
#define ROUND_POWER_OF_TWO_MMI                             \
  /* Add para[0] */                                        \
  "lw         %[tmp0],     0x00(%[para])             \n\t" \
  MMI_MTC1(%[tmp0],     %[ftmp6])                          \
  "punpcklwd  %[ftmp6],    %[ftmp6],    %[ftmp6]     \n\t" \
  "paddw      %[srcl],     %[srcl],     %[ftmp6]     \n\t" \
  "paddw      %[srch],     %[srch],     %[ftmp6]     \n\t" \
  /* Arithmetic right shift para[1] bits */                \
  "lw         %[tmp0],     0x04(%[para])             \n\t" \
  MMI_MTC1(%[tmp0],     %[ftmp5])                          \
  "psraw      %[srcl],     %[srcl],     %[ftmp5]     \n\t" \
  "psraw      %[srch],     %[srch],     %[ftmp5]     \n\t"
/* clang-format on */

#define CLIP_PIXEL_MMI                                     \
  /* Staturated operation */                               \
  "packsswh   %[srcl],     %[srcl],     %[srch]      \n\t" \
  "packushb   %[ftmp12],   %[srcl],     %[ftmp0]     \n\t"

static void convolve_horiz_mmi(const uint8_t *src, ptrdiff_t src_stride,
                               uint8_t *dst, ptrdiff_t dst_stride,
                               const InterpKernel *filter, int x0_q4,
                               int x_step_q4, int32_t w, int32_t h) {
  const int16_t *filter_x = filter[x0_q4];
  double ftmp[14];
  uint32_t tmp[2];
  uint32_t para[5];
  para[0] = (1 << ((FILTER_BITS)-1));
  para[1] = FILTER_BITS;
  src -= SUBPEL_TAPS / 2 - 1;
  src_stride -= w;
  dst_stride -= w;
  (void)x_step_q4;

  /* clang-format off */
  __asm__ volatile(
    "move       %[tmp1],    %[width]                   \n\t"
    "pxor       %[ftmp0],   %[ftmp0],    %[ftmp0]      \n\t"
    "gsldlc1    %[filter1], 0x03(%[filter])            \n\t"
    "gsldrc1    %[filter1], 0x00(%[filter])            \n\t"
    "gsldlc1    %[filter2], 0x0b(%[filter])            \n\t"
    "gsldrc1    %[filter2], 0x08(%[filter])            \n\t"
    "1:                                                \n\t"
    /* Get 8 data per row */
    "gsldlc1    %[ftmp5],   0x07(%[src])               \n\t"
    "gsldrc1    %[ftmp5],   0x00(%[src])               \n\t"
    "gsldlc1    %[ftmp7],   0x08(%[src])               \n\t"
    "gsldrc1    %[ftmp7],   0x01(%[src])               \n\t"
    "gsldlc1    %[ftmp9],   0x09(%[src])               \n\t"
    "gsldrc1    %[ftmp9],   0x02(%[src])               \n\t"
    "gsldlc1    %[ftmp11],  0x0A(%[src])               \n\t"
    "gsldrc1    %[ftmp11],  0x03(%[src])               \n\t"
    "punpcklbh  %[ftmp4],   %[ftmp5],    %[ftmp0]      \n\t"
    "punpckhbh  %[ftmp5],   %[ftmp5],    %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp6],   %[ftmp7],    %[ftmp0]      \n\t"
    "punpckhbh  %[ftmp7],   %[ftmp7],    %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp8],   %[ftmp9],    %[ftmp0]      \n\t"
    "punpckhbh  %[ftmp9],   %[ftmp9],    %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp10],  %[ftmp11],   %[ftmp0]      \n\t"
    "punpckhbh  %[ftmp11],  %[ftmp11],   %[ftmp0]      \n\t"
    MMI_ADDIU(%[width],   %[width],    -0x04)
    /* Get raw data */
    GET_DATA_H_MMI
    ROUND_POWER_OF_TWO_MMI
    CLIP_PIXEL_MMI
    "swc1       %[ftmp12],  0x00(%[dst])               \n\t"
    MMI_ADDIU(%[dst],     %[dst],      0x04)
    MMI_ADDIU(%[src],     %[src],      0x04)
    /* Loop count */
    "bnez       %[width],   1b                         \n\t"
    "move       %[width],   %[tmp1]                    \n\t"
    MMI_ADDU(%[src],      %[src],      %[src_stride])
    MMI_ADDU(%[dst],      %[dst],      %[dst_stride])
    MMI_ADDIU(%[height],  %[height],   -0x01)
    "bnez       %[height],  1b                         \n\t"
    : [srcl]"=&f"(ftmp[0]),     [srch]"=&f"(ftmp[1]),
      [filter1]"=&f"(ftmp[2]),  [filter2]"=&f"(ftmp[3]),
      [ftmp0]"=&f"(ftmp[4]),    [ftmp4]"=&f"(ftmp[5]),
      [ftmp5]"=&f"(ftmp[6]),    [ftmp6]"=&f"(ftmp[7]),
      [ftmp7]"=&f"(ftmp[8]),    [ftmp8]"=&f"(ftmp[9]),
      [ftmp9]"=&f"(ftmp[10]),   [ftmp10]"=&f"(ftmp[11]),
      [ftmp11]"=&f"(ftmp[12]),  [ftmp12]"=&f"(ftmp[13]),
      [tmp0]"=&r"(tmp[0]),      [tmp1]"=&r"(tmp[1]),
      [src]"+&r"(src),          [width]"+&r"(w),
      [dst]"+&r"(dst),          [height]"+&r"(h)
    : [filter]"r"(filter_x),    [para]"r"(para),
      [src_stride]"r"((mips_reg)src_stride),
      [dst_stride]"r"((mips_reg)dst_stride)
    : "memory"
  );
  /* clang-format on */
}

static void convolve_vert_mmi(const uint8_t *src, ptrdiff_t src_stride,
                              uint8_t *dst, ptrdiff_t dst_stride,
                              const InterpKernel *filter, int y0_q4,
                              int y_step_q4, int32_t w, int32_t h) {
  const int16_t *filter_y = filter[y0_q4];
  double ftmp[16];
  uint32_t tmp[1];
  uint32_t para[2];
  ptrdiff_t addr = src_stride;
  para[0] = (1 << ((FILTER_BITS)-1));
  para[1] = FILTER_BITS;
  src -= src_stride * (SUBPEL_TAPS / 2 - 1);
  src_stride -= w;
  dst_stride -= w;
  (void)y_step_q4;

  __asm__ volatile(
    "pxor       %[ftmp0],    %[ftmp0],   %[ftmp0]      \n\t"
    "gsldlc1    %[ftmp4],    0x03(%[filter])           \n\t"
    "gsldrc1    %[ftmp4],    0x00(%[filter])           \n\t"
    "gsldlc1    %[ftmp5],    0x0b(%[filter])           \n\t"
    "gsldrc1    %[ftmp5],    0x08(%[filter])           \n\t"
    "punpcklwd  %[filter10], %[ftmp4],   %[ftmp4]      \n\t"
    "punpckhwd  %[filter32], %[ftmp4],   %[ftmp4]      \n\t"
    "punpcklwd  %[filter54], %[ftmp5],   %[ftmp5]      \n\t"
    "punpckhwd  %[filter76], %[ftmp5],   %[ftmp5]      \n\t"
    "1:                                                \n\t"
    /* Get 8 data per column */
    "gsldlc1    %[ftmp4],    0x07(%[src])              \n\t"
    "gsldrc1    %[ftmp4],    0x00(%[src])              \n\t"
    MMI_ADDU(%[tmp0],     %[src],     %[addr])
    "gsldlc1    %[ftmp5],    0x07(%[tmp0])             \n\t"
    "gsldrc1    %[ftmp5],    0x00(%[tmp0])             \n\t"
    MMI_ADDU(%[tmp0],     %[tmp0],    %[addr])
    "gsldlc1    %[ftmp6],    0x07(%[tmp0])             \n\t"
    "gsldrc1    %[ftmp6],    0x00(%[tmp0])             \n\t"
    MMI_ADDU(%[tmp0],     %[tmp0],    %[addr])
    "gsldlc1    %[ftmp7],    0x07(%[tmp0])             \n\t"
    "gsldrc1    %[ftmp7],    0x00(%[tmp0])             \n\t"
    MMI_ADDU(%[tmp0],     %[tmp0],    %[addr])
    "gsldlc1    %[ftmp8],    0x07(%[tmp0])             \n\t"
    "gsldrc1    %[ftmp8],    0x00(%[tmp0])             \n\t"
    MMI_ADDU(%[tmp0],     %[tmp0],    %[addr])
    "gsldlc1    %[ftmp9],    0x07(%[tmp0])             \n\t"
    "gsldrc1    %[ftmp9],    0x00(%[tmp0])             \n\t"
    MMI_ADDU(%[tmp0],     %[tmp0],    %[addr])
    "gsldlc1    %[ftmp10],   0x07(%[tmp0])             \n\t"
    "gsldrc1    %[ftmp10],   0x00(%[tmp0])             \n\t"
    MMI_ADDU(%[tmp0],     %[tmp0],    %[addr])
    "gsldlc1    %[ftmp11],   0x07(%[tmp0])             \n\t"
    "gsldrc1    %[ftmp11],   0x00(%[tmp0])             \n\t"
    "punpcklbh  %[ftmp4],    %[ftmp4],   %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp5],    %[ftmp5],   %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp6],    %[ftmp6],   %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp7],    %[ftmp7],   %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp8],    %[ftmp8],   %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp9],    %[ftmp9],   %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp10],   %[ftmp10],  %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp11],   %[ftmp11],  %[ftmp0]      \n\t"
    MMI_ADDIU(%[width],   %[width],   -0x04)
    /* Get raw data */
    GET_DATA_V_MMI
    ROUND_POWER_OF_TWO_MMI
    CLIP_PIXEL_MMI
    "swc1       %[ftmp12],   0x00(%[dst])              \n\t"
    MMI_ADDIU(%[dst],     %[dst],      0x04)
    MMI_ADDIU(%[src],     %[src],      0x04)
    /* Loop count */
    "bnez       %[width],    1b                        \n\t"
    MMI_SUBU(%[width],    %[addr],     %[src_stride])
    MMI_ADDU(%[src],      %[src],      %[src_stride])
    MMI_ADDU(%[dst],      %[dst],      %[dst_stride])
    MMI_ADDIU(%[height],  %[height],   -0x01)
    "bnez       %[height],   1b                        \n\t"
    : [srcl]"=&f"(ftmp[0]),     [srch]"=&f"(ftmp[1]),
      [filter10]"=&f"(ftmp[2]), [filter32]"=&f"(ftmp[3]),
      [filter54]"=&f"(ftmp[4]), [filter76]"=&f"(ftmp[5]),
      [ftmp0]"=&f"(ftmp[6]),    [ftmp4]"=&f"(ftmp[7]),
      [ftmp5]"=&f"(ftmp[8]),    [ftmp6]"=&f"(ftmp[9]),
      [ftmp7]"=&f"(ftmp[10]),   [ftmp8]"=&f"(ftmp[11]),
      [ftmp9]"=&f"(ftmp[12]),   [ftmp10]"=&f"(ftmp[13]),
      [ftmp11]"=&f"(ftmp[14]),  [ftmp12]"=&f"(ftmp[15]),
      [src]"+&r"(src),          [dst]"+&r"(dst),
      [width]"+&r"(w),          [height]"+&r"(h),
      [tmp0]"=&r"(tmp[0])
    : [filter]"r"(filter_y),    [para]"r"(para),
      [src_stride]"r"((mips_reg)src_stride),
      [dst_stride]"r"((mips_reg)dst_stride),
      [addr]"r"((mips_reg)addr)
    : "memory"
  );
}

static void convolve_avg_horiz_mmi(const uint8_t *src, ptrdiff_t src_stride,
                                   uint8_t *dst, ptrdiff_t dst_stride,
                                   const InterpKernel *filter, int x0_q4,
                                   int x_step_q4, int32_t w, int32_t h) {
  const int16_t *filter_x = filter[x0_q4];
  double ftmp[14];
  uint32_t tmp[2];
  uint32_t para[2];
  para[0] = (1 << ((FILTER_BITS)-1));
  para[1] = FILTER_BITS;
  src -= SUBPEL_TAPS / 2 - 1;
  src_stride -= w;
  dst_stride -= w;
  (void)x_step_q4;

  __asm__ volatile(
    "move       %[tmp1],    %[width]                   \n\t"
    "pxor       %[ftmp0],   %[ftmp0],    %[ftmp0]      \n\t"
    "gsldlc1    %[filter1], 0x03(%[filter])            \n\t"
    "gsldrc1    %[filter1], 0x00(%[filter])            \n\t"
    "gsldlc1    %[filter2], 0x0b(%[filter])            \n\t"
    "gsldrc1    %[filter2], 0x08(%[filter])            \n\t"
    "1:                                                \n\t"
    /* Get 8 data per row */
    "gsldlc1    %[ftmp5],   0x07(%[src])               \n\t"
    "gsldrc1    %[ftmp5],   0x00(%[src])               \n\t"
    "gsldlc1    %[ftmp7],   0x08(%[src])               \n\t"
    "gsldrc1    %[ftmp7],   0x01(%[src])               \n\t"
    "gsldlc1    %[ftmp9],   0x09(%[src])               \n\t"
    "gsldrc1    %[ftmp9],   0x02(%[src])               \n\t"
    "gsldlc1    %[ftmp11],  0x0A(%[src])               \n\t"
    "gsldrc1    %[ftmp11],  0x03(%[src])               \n\t"
    "punpcklbh  %[ftmp4],   %[ftmp5],    %[ftmp0]      \n\t"
    "punpckhbh  %[ftmp5],   %[ftmp5],    %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp6],   %[ftmp7],    %[ftmp0]      \n\t"
    "punpckhbh  %[ftmp7],   %[ftmp7],    %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp8],   %[ftmp9],    %[ftmp0]      \n\t"
    "punpckhbh  %[ftmp9],   %[ftmp9],    %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp10],  %[ftmp11],   %[ftmp0]      \n\t"
    "punpckhbh  %[ftmp11],  %[ftmp11],   %[ftmp0]      \n\t"
    MMI_ADDIU(%[width],   %[width],    -0x04)
    /* Get raw data */
    GET_DATA_H_MMI
    ROUND_POWER_OF_TWO_MMI
    CLIP_PIXEL_MMI
    "punpcklbh  %[ftmp12],  %[ftmp12],   %[ftmp0]      \n\t"
    "gsldlc1    %[ftmp4],   0x07(%[dst])               \n\t"
    "gsldrc1    %[ftmp4],   0x00(%[dst])               \n\t"
    "punpcklbh  %[ftmp4],   %[ftmp4],    %[ftmp0]      \n\t"
    "paddh      %[ftmp12],  %[ftmp12],   %[ftmp4]      \n\t"
    "li         %[tmp0],    0x10001                    \n\t"
    MMI_MTC1(%[tmp0],     %[ftmp5])
    "punpcklhw  %[ftmp5],   %[ftmp5],    %[ftmp5]      \n\t"
    "paddh      %[ftmp12],  %[ftmp12],   %[ftmp5]      \n\t"
    "psrah      %[ftmp12],  %[ftmp12],   %[ftmp5]      \n\t"
    "packushb   %[ftmp12],  %[ftmp12],   %[ftmp0]      \n\t"
    "swc1       %[ftmp12],  0x00(%[dst])               \n\t"
    MMI_ADDIU(%[dst],     %[dst],      0x04)
    MMI_ADDIU(%[src],     %[src],      0x04)
    /* Loop count */
    "bnez       %[width],   1b                         \n\t"
    "move       %[width],   %[tmp1]                    \n\t"
    MMI_ADDU(%[src],      %[src],      %[src_stride])
    MMI_ADDU(%[dst],      %[dst],      %[dst_stride])
    MMI_ADDIU(%[height],  %[height],   -0x01)
    "bnez       %[height],  1b                         \n\t"
    : [srcl]"=&f"(ftmp[0]),     [srch]"=&f"(ftmp[1]),
      [filter1]"=&f"(ftmp[2]),  [filter2]"=&f"(ftmp[3]),
      [ftmp0]"=&f"(ftmp[4]),    [ftmp4]"=&f"(ftmp[5]),
      [ftmp5]"=&f"(ftmp[6]),    [ftmp6]"=&f"(ftmp[7]),
      [ftmp7]"=&f"(ftmp[8]),    [ftmp8]"=&f"(ftmp[9]),
      [ftmp9]"=&f"(ftmp[10]),   [ftmp10]"=&f"(ftmp[11]),
      [ftmp11]"=&f"(ftmp[12]),  [ftmp12]"=&f"(ftmp[13]),
      [tmp0]"=&r"(tmp[0]),      [tmp1]"=&r"(tmp[1]),
      [src]"+&r"(src),          [width]"+&r"(w),
      [dst]"+&r"(dst),          [height]"+&r"(h)
    : [filter]"r"(filter_x),    [para]"r"(para),
      [src_stride]"r"((mips_reg)src_stride),
      [dst_stride]"r"((mips_reg)dst_stride)
    : "memory"
  );
}

static void convolve_avg_vert_mmi(const uint8_t *src, ptrdiff_t src_stride,
                                  uint8_t *dst, ptrdiff_t dst_stride,
                                  const InterpKernel *filter, int y0_q4,
                                  int y_step_q4, int32_t w, int32_t h) {
  const int16_t *filter_y = filter[y0_q4];
  double ftmp[16];
  uint32_t tmp[1];
  uint32_t para[2];
  ptrdiff_t addr = src_stride;
  para[0] = (1 << ((FILTER_BITS)-1));
  para[1] = FILTER_BITS;
  src -= src_stride * (SUBPEL_TAPS / 2 - 1);
  src_stride -= w;
  dst_stride -= w;
  (void)y_step_q4;

  __asm__ volatile(
    "pxor       %[ftmp0],    %[ftmp0],   %[ftmp0]      \n\t"
    "gsldlc1    %[ftmp4],    0x03(%[filter])           \n\t"
    "gsldrc1    %[ftmp4],    0x00(%[filter])           \n\t"
    "gsldlc1    %[ftmp5],    0x0b(%[filter])           \n\t"
    "gsldrc1    %[ftmp5],    0x08(%[filter])           \n\t"
    "punpcklwd  %[filter10], %[ftmp4],   %[ftmp4]      \n\t"
    "punpckhwd  %[filter32], %[ftmp4],   %[ftmp4]      \n\t"
    "punpcklwd  %[filter54], %[ftmp5],   %[ftmp5]      \n\t"
    "punpckhwd  %[filter76], %[ftmp5],   %[ftmp5]      \n\t"
    "1:                                                \n\t"
    /* Get 8 data per column */
    "gsldlc1    %[ftmp4],    0x07(%[src])              \n\t"
    "gsldrc1    %[ftmp4],    0x00(%[src])              \n\t"
    MMI_ADDU(%[tmp0],     %[src],     %[addr])
    "gsldlc1    %[ftmp5],    0x07(%[tmp0])             \n\t"
    "gsldrc1    %[ftmp5],    0x00(%[tmp0])             \n\t"
    MMI_ADDU(%[tmp0],     %[tmp0],    %[addr])
    "gsldlc1    %[ftmp6],    0x07(%[tmp0])             \n\t"
    "gsldrc1    %[ftmp6],    0x00(%[tmp0])             \n\t"
    MMI_ADDU(%[tmp0],     %[tmp0],    %[addr])
    "gsldlc1    %[ftmp7],    0x07(%[tmp0])             \n\t"
    "gsldrc1    %[ftmp7],    0x00(%[tmp0])             \n\t"
    MMI_ADDU(%[tmp0],     %[tmp0],    %[addr])
    "gsldlc1    %[ftmp8],    0x07(%[tmp0])             \n\t"
    "gsldrc1    %[ftmp8],    0x00(%[tmp0])             \n\t"
    MMI_ADDU(%[tmp0],     %[tmp0],    %[addr])
    "gsldlc1    %[ftmp9],    0x07(%[tmp0])             \n\t"
    "gsldrc1    %[ftmp9],    0x00(%[tmp0])             \n\t"
    MMI_ADDU(%[tmp0],     %[tmp0],    %[addr])
    "gsldlc1    %[ftmp10],   0x07(%[tmp0])             \n\t"
    "gsldrc1    %[ftmp10],   0x00(%[tmp0])             \n\t"
    MMI_ADDU(%[tmp0],     %[tmp0],    %[addr])
    "gsldlc1    %[ftmp11],   0x07(%[tmp0])             \n\t"
    "gsldrc1    %[ftmp11],   0x00(%[tmp0])             \n\t"
    "punpcklbh  %[ftmp4],    %[ftmp4],   %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp5],    %[ftmp5],   %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp6],    %[ftmp6],   %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp7],    %[ftmp7],   %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp8],    %[ftmp8],   %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp9],    %[ftmp9],   %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp10],   %[ftmp10],  %[ftmp0]      \n\t"
    "punpcklbh  %[ftmp11],   %[ftmp11],  %[ftmp0]      \n\t"
    MMI_ADDIU(%[width],   %[width],   -0x04)
    /* Get raw data */
    GET_DATA_V_MMI
    ROUND_POWER_OF_TWO_MMI
    CLIP_PIXEL_MMI
    "punpcklbh  %[ftmp12],   %[ftmp12],  %[ftmp0]      \n\t"
    "gsldlc1    %[ftmp4],    0x07(%[dst])              \n\t"
    "gsldrc1    %[ftmp4],    0x00(%[dst])              \n\t"
    "punpcklbh  %[ftmp4],    %[ftmp4],   %[ftmp0]      \n\t"
    "paddh      %[ftmp12],   %[ftmp12],  %[ftmp4]      \n\t"
    "li         %[tmp0],     0x10001                   \n\t"
    MMI_MTC1(%[tmp0],     %[ftmp5])
    "punpcklhw  %[ftmp5],    %[ftmp5],   %[ftmp5]      \n\t"
    "paddh      %[ftmp12],   %[ftmp12],  %[ftmp5]      \n\t"
    "psrah      %[ftmp12],   %[ftmp12],  %[ftmp5]      \n\t"
    "packushb   %[ftmp12],   %[ftmp12],  %[ftmp0]      \n\t"
    "swc1       %[ftmp12],   0x00(%[dst])              \n\t"
    MMI_ADDIU(%[dst],     %[dst],      0x04)
    MMI_ADDIU(%[src],     %[src],      0x04)
    /* Loop count */
    "bnez       %[width],    1b                        \n\t"
    MMI_SUBU(%[width],    %[addr],     %[src_stride])
    MMI_ADDU(%[src],      %[src],      %[src_stride])
    MMI_ADDU(%[dst],      %[dst],      %[dst_stride])
    MMI_ADDIU(%[height],  %[height],   -0x01)
    "bnez       %[height],   1b                        \n\t"
    : [srcl]"=&f"(ftmp[0]),     [srch]"=&f"(ftmp[1]),
      [filter10]"=&f"(ftmp[2]), [filter32]"=&f"(ftmp[3]),
      [filter54]"=&f"(ftmp[4]), [filter76]"=&f"(ftmp[5]),
      [ftmp0]"=&f"(ftmp[6]),    [ftmp4]"=&f"(ftmp[7]),
      [ftmp5]"=&f"(ftmp[8]),    [ftmp6]"=&f"(ftmp[9]),
      [ftmp7]"=&f"(ftmp[10]),   [ftmp8]"=&f"(ftmp[11]),
      [ftmp9]"=&f"(ftmp[12]),   [ftmp10]"=&f"(ftmp[13]),
      [ftmp11]"=&f"(ftmp[14]),  [ftmp12]"=&f"(ftmp[15]),
      [src]"+&r"(src),          [dst]"+&r"(dst),
      [width]"+&r"(w),          [height]"+&r"(h),
      [tmp0]"=&r"(tmp[0])
    : [filter]"r"(filter_y),    [para]"r"(para),
      [src_stride]"r"((mips_reg)src_stride),
      [dst_stride]"r"((mips_reg)dst_stride),
      [addr]"r"((mips_reg)addr)
    : "memory"
  );
}

void vpx_convolve_avg_mmi(const uint8_t *src, ptrdiff_t src_stride,
                          uint8_t *dst, ptrdiff_t dst_stride,
                          const InterpKernel *filter, int x0_q4, int x_step_q4,
                          int y0_q4, int y_step_q4, int w, int h) {
  int x, y;

  (void)filter;
  (void)x0_q4;
  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;

  if (w & 0x03) {
    for (y = 0; y < h; ++y) {
      for (x = 0; x < w; ++x) dst[x] = ROUND_POWER_OF_TWO(dst[x] + src[x], 1);
      src += src_stride;
      dst += dst_stride;
    }
  } else {
    double ftmp[4];
    uint32_t tmp[2];
    src_stride -= w;
    dst_stride -= w;

    __asm__ volatile(
      "move       %[tmp1],    %[width]                  \n\t"
      "pxor       %[ftmp0],   %[ftmp0],   %[ftmp0]      \n\t"
      "li         %[tmp0],    0x10001                   \n\t"
      MMI_MTC1(%[tmp0],    %[ftmp3])
      "punpcklhw  %[ftmp3],   %[ftmp3],   %[ftmp3]      \n\t"
      "1:                                               \n\t"
      "gsldlc1    %[ftmp1],   0x07(%[src])              \n\t"
      "gsldrc1    %[ftmp1],   0x00(%[src])              \n\t"
      "gsldlc1    %[ftmp2],   0x07(%[dst])              \n\t"
      "gsldrc1    %[ftmp2],   0x00(%[dst])              \n\t"
      "punpcklbh  %[ftmp1],   %[ftmp1],   %[ftmp0]      \n\t"
      "punpcklbh  %[ftmp2],   %[ftmp2],   %[ftmp0]      \n\t"
      "paddh      %[ftmp1],   %[ftmp1],   %[ftmp2]      \n\t"
      "paddh      %[ftmp1],   %[ftmp1],   %[ftmp3]      \n\t"
      "psrah      %[ftmp1],   %[ftmp1],   %[ftmp3]      \n\t"
      "packushb   %[ftmp1],   %[ftmp1],   %[ftmp0]      \n\t"
      "swc1       %[ftmp1],   0x00(%[dst])              \n\t"
      MMI_ADDIU(%[width],  %[width],   -0x04)
      MMI_ADDIU(%[dst],    %[dst],     0x04)
      MMI_ADDIU(%[src],    %[src],     0x04)
      "bnez       %[width],   1b                        \n\t"
      "move       %[width],   %[tmp1]                   \n\t"
      MMI_ADDU(%[dst],     %[dst],     %[dst_stride])
      MMI_ADDU(%[src],     %[src],     %[src_stride])
      MMI_ADDIU(%[height], %[height],  -0x01)
      "bnez       %[height],  1b                        \n\t"
      : [ftmp0]"=&f"(ftmp[0]),  [ftmp1]"=&f"(ftmp[1]),
        [ftmp2]"=&f"(ftmp[2]),  [ftmp3]"=&f"(ftmp[3]),
        [tmp0]"=&r"(tmp[0]),    [tmp1]"=&r"(tmp[1]),
        [src]"+&r"(src),        [dst]"+&r"(dst),
        [width]"+&r"(w),        [height]"+&r"(h)
      : [src_stride]"r"((mips_reg)src_stride),
        [dst_stride]"r"((mips_reg)dst_stride)
      : "memory"
    );
  }
}

static void convolve_horiz(const uint8_t *src, ptrdiff_t src_stride,
                           uint8_t *dst, ptrdiff_t dst_stride,
                           const InterpKernel *x_filters, int x0_q4,
                           int x_step_q4, int w, int h) {
  int x, y;
  src -= SUBPEL_TAPS / 2 - 1;

  for (y = 0; y < h; ++y) {
    int x_q4 = x0_q4;
    for (x = 0; x < w; ++x) {
      const uint8_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
      const int16_t *const x_filter = x_filters[x_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < SUBPEL_TAPS; ++k) sum += src_x[k] * x_filter[k];
      dst[x] = clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
      x_q4 += x_step_q4;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

static void convolve_vert(const uint8_t *src, ptrdiff_t src_stride,
                          uint8_t *dst, ptrdiff_t dst_stride,
                          const InterpKernel *y_filters, int y0_q4,
                          int y_step_q4, int w, int h) {
  int x, y;
  src -= src_stride * (SUBPEL_TAPS / 2 - 1);

  for (x = 0; x < w; ++x) {
    int y_q4 = y0_q4;
    for (y = 0; y < h; ++y) {
      const uint8_t *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
      const int16_t *const y_filter = y_filters[y_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < SUBPEL_TAPS; ++k)
        sum += src_y[k * src_stride] * y_filter[k];
      dst[y * dst_stride] = clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
      y_q4 += y_step_q4;
    }
    ++src;
    ++dst;
  }
}

static void convolve_avg_vert(const uint8_t *src, ptrdiff_t src_stride,
                              uint8_t *dst, ptrdiff_t dst_stride,
                              const InterpKernel *y_filters, int y0_q4,
                              int y_step_q4, int w, int h) {
  int x, y;
  src -= src_stride * (SUBPEL_TAPS / 2 - 1);

  for (x = 0; x < w; ++x) {
    int y_q4 = y0_q4;
    for (y = 0; y < h; ++y) {
      const uint8_t *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
      const int16_t *const y_filter = y_filters[y_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < SUBPEL_TAPS; ++k)
        sum += src_y[k * src_stride] * y_filter[k];
      dst[y * dst_stride] = ROUND_POWER_OF_TWO(
          dst[y * dst_stride] +
              clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS)),
          1);
      y_q4 += y_step_q4;
    }
    ++src;
    ++dst;
  }
}

static void convolve_avg_horiz(const uint8_t *src, ptrdiff_t src_stride,
                               uint8_t *dst, ptrdiff_t dst_stride,
                               const InterpKernel *x_filters, int x0_q4,
                               int x_step_q4, int w, int h) {
  int x, y;
  src -= SUBPEL_TAPS / 2 - 1;

  for (y = 0; y < h; ++y) {
    int x_q4 = x0_q4;
    for (x = 0; x < w; ++x) {
      const uint8_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
      const int16_t *const x_filter = x_filters[x_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < SUBPEL_TAPS; ++k) sum += src_x[k] * x_filter[k];
      dst[x] = ROUND_POWER_OF_TWO(
          dst[x] + clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS)), 1);
      x_q4 += x_step_q4;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

void vpx_convolve8_mmi(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                       ptrdiff_t dst_stride, const InterpKernel *filter,
                       int x0_q4, int32_t x_step_q4, int y0_q4,
                       int32_t y_step_q4, int32_t w, int32_t h) {
  // Note: Fixed size intermediate buffer, temp, places limits on parameters.
  // 2d filtering proceeds in 2 steps:
  //   (1) Interpolate horizontally into an intermediate buffer, temp.
  //   (2) Interpolate temp vertically to derive the sub-pixel result.
  // Deriving the maximum number of rows in the temp buffer (135):
  // --Smallest scaling factor is x1/2 ==> y_step_q4 = 32 (Normative).
  // --Largest block size is 64x64 pixels.
  // --64 rows in the downscaled frame span a distance of (64 - 1) * 32 in the
  //   original frame (in 1/16th pixel units).
  // --Must round-up because block may be located at sub-pixel position.
  // --Require an additional SUBPEL_TAPS rows for the 8-tap filter tails.
  // --((64 - 1) * 32 + 15) >> 4 + 8 = 135.
  // When calling in frame scaling function, the smallest scaling factor is x1/4
  // ==> y_step_q4 = 64. Since w and h are at most 16, the temp buffer is still
  // big enough.
  uint8_t temp[64 * 135];
  const int intermediate_height =
      (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + SUBPEL_TAPS;

  assert(w <= 64);
  assert(h <= 64);
  assert(y_step_q4 <= 32 || (y_step_q4 <= 64 && h <= 32));
  assert(x_step_q4 <= 64);

  if (w & 0x03) {
    convolve_horiz(src - src_stride * (SUBPEL_TAPS / 2 - 1), src_stride, temp,
                   64, filter, x0_q4, x_step_q4, w, intermediate_height);
    convolve_vert(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst, dst_stride,
                  filter, y0_q4, y_step_q4, w, h);
  } else {
    convolve_horiz_mmi(src - src_stride * (SUBPEL_TAPS / 2 - 1), src_stride,
                       temp, 64, filter, x0_q4, x_step_q4, w,
                       intermediate_height);
    convolve_vert_mmi(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst, dst_stride,
                      filter, y0_q4, y_step_q4, w, h);
  }
}

void vpx_convolve8_horiz_mmi(const uint8_t *src, ptrdiff_t src_stride,
                             uint8_t *dst, ptrdiff_t dst_stride,
                             const InterpKernel *filter, int x0_q4,
                             int32_t x_step_q4, int y0_q4, int32_t y_step_q4,
                             int32_t w, int32_t h) {
  (void)y0_q4;
  (void)y_step_q4;
  if (w & 0x03)
    convolve_horiz(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4,
                   w, h);
  else
    convolve_horiz_mmi(src, src_stride, dst, dst_stride, filter, x0_q4,
                       x_step_q4, w, h);
}

void vpx_convolve8_vert_mmi(const uint8_t *src, ptrdiff_t src_stride,
                            uint8_t *dst, ptrdiff_t dst_stride,
                            const InterpKernel *filter, int x0_q4,
                            int32_t x_step_q4, int y0_q4, int y_step_q4, int w,
                            int h) {
  (void)x0_q4;
  (void)x_step_q4;
  if (w & 0x03)
    convolve_vert(src, src_stride, dst, dst_stride, filter, y0_q4, y_step_q4, w,
                  h);
  else
    convolve_vert_mmi(src, src_stride, dst, dst_stride, filter, y0_q4,
                      y_step_q4, w, h);
}

void vpx_convolve8_avg_horiz_mmi(const uint8_t *src, ptrdiff_t src_stride,
                                 uint8_t *dst, ptrdiff_t dst_stride,
                                 const InterpKernel *filter, int x0_q4,
                                 int32_t x_step_q4, int y0_q4, int y_step_q4,
                                 int w, int h) {
  (void)y0_q4;
  (void)y_step_q4;
  if (w & 0x03)
    convolve_avg_horiz(src, src_stride, dst, dst_stride, filter, x0_q4,
                       x_step_q4, w, h);
  else
    convolve_avg_horiz_mmi(src, src_stride, dst, dst_stride, filter, x0_q4,
                           x_step_q4, w, h);
}

void vpx_convolve8_avg_vert_mmi(const uint8_t *src, ptrdiff_t src_stride,
                                uint8_t *dst, ptrdiff_t dst_stride,
                                const InterpKernel *filter, int x0_q4,
                                int32_t x_step_q4, int y0_q4, int y_step_q4,
                                int w, int h) {
  (void)x0_q4;
  (void)x_step_q4;
  if (w & 0x03)
    convolve_avg_vert(src, src_stride, dst, dst_stride, filter, y0_q4,
                      y_step_q4, w, h);
  else
    convolve_avg_vert_mmi(src, src_stride, dst, dst_stride, filter, y0_q4,
                          y_step_q4, w, h);
}

void vpx_convolve8_avg_mmi(const uint8_t *src, ptrdiff_t src_stride,
                           uint8_t *dst, ptrdiff_t dst_stride,
                           const InterpKernel *filter, int x0_q4,
                           int32_t x_step_q4, int y0_q4, int32_t y_step_q4,
                           int32_t w, int32_t h) {
  // Fixed size intermediate buffer places limits on parameters.
  DECLARE_ALIGNED(16, uint8_t, temp[64 * 64]);
  assert(w <= 64);
  assert(h <= 64);

  vpx_convolve8_mmi(src, src_stride, temp, 64, filter, x0_q4, x_step_q4, y0_q4,
                    y_step_q4, w, h);
  vpx_convolve_avg_mmi(temp, 64, dst, dst_stride, NULL, 0, 0, 0, 0, w, h);
}
