/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"
#include "vpx_ports/asmdefs_mmi.h"

void vpx_subtract_block_mmi(int rows, int cols, int16_t *diff,
                            ptrdiff_t diff_stride, const uint8_t *src,
                            ptrdiff_t src_stride, const uint8_t *pred,
                            ptrdiff_t pred_stride) {
  double ftmp[13];
  uint32_t tmp[1];

  if (rows == cols) {
    switch (rows) {
      case 4:
        __asm__ volatile(
            "pxor       %[ftmp0],   %[ftmp0],           %[ftmp0]        \n\t"
#if _MIPS_SIM == _ABIO32
            "ulw        %[tmp0],    0x00(%[src])                        \n\t"
            "mtc1       %[tmp0],    %[ftmp1]                            \n\t"
            "ulw        %[tmp0],    0x00(%[pred])                       \n\t"
            "mtc1       %[tmp0],    %[ftmp2]                            \n\t"
#else
            "gslwlc1    %[ftmp1],   0x03(%[src])                        \n\t"
            "gslwrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gslwlc1    %[ftmp2],   0x03(%[pred])                       \n\t"
            "gslwrc1    %[ftmp2],   0x00(%[pred])                       \n\t"
#endif
            MMI_ADDU(%[src], %[src], %[src_stride])
            MMI_ADDU(%[pred], %[pred], %[pred_stride])

#if _MIPS_SIM == _ABIO32
            "ulw        %[tmp0],    0x00(%[src])                        \n\t"
            "mtc1       %[tmp0],    %[ftmp3]                            \n\t"
            "ulw        %[tmp0],    0x00(%[pred])                       \n\t"
            "mtc1       %[tmp0],    %[ftmp4]                            \n\t"
#else
            "gslwlc1    %[ftmp3],   0x03(%[src])                        \n\t"
            "gslwrc1    %[ftmp3],   0x00(%[src])                        \n\t"
            "gslwlc1    %[ftmp4],   0x03(%[pred])                       \n\t"
            "gslwrc1    %[ftmp4],   0x00(%[pred])                       \n\t"
#endif
            MMI_ADDU(%[src], %[src], %[src_stride])
            MMI_ADDU(%[pred], %[pred], %[pred_stride])

#if _MIPS_SIM == _ABIO32
            "ulw        %[tmp0],    0x00(%[src])                        \n\t"
            "mtc1       %[tmp0],    %[ftmp5]                            \n\t"
            "ulw        %[tmp0],    0x00(%[pred])                       \n\t"
            "mtc1       %[tmp0],    %[ftmp6]                            \n\t"
#else
            "gslwlc1    %[ftmp5],   0x03(%[src])                        \n\t"
            "gslwrc1    %[ftmp5],   0x00(%[src])                        \n\t"
            "gslwlc1    %[ftmp6],   0x03(%[pred])                       \n\t"
            "gslwrc1    %[ftmp6],   0x00(%[pred])                       \n\t"
#endif
            MMI_ADDU(%[src], %[src], %[src_stride])
            MMI_ADDU(%[pred], %[pred], %[pred_stride])

#if _MIPS_SIM == _ABIO32
            "ulw        %[tmp0],    0x00(%[src])                        \n\t"
            "mtc1       %[tmp0],    %[ftmp7]                            \n\t"
            "ulw        %[tmp0],    0x00(%[pred])                       \n\t"
            "mtc1       %[tmp0],    %[ftmp8]                            \n\t"
#else
            "gslwlc1    %[ftmp7],   0x03(%[src])                        \n\t"
            "gslwrc1    %[ftmp7],   0x00(%[src])                        \n\t"
            "gslwlc1    %[ftmp8],   0x03(%[pred])                       \n\t"
            "gslwrc1    %[ftmp8],   0x00(%[pred])                       \n\t"
#endif
            "punpcklbh  %[ftmp9],   %[ftmp1],           %[ftmp0]        \n\t"
            "punpcklbh  %[ftmp10],  %[ftmp2],           %[ftmp0]        \n\t"
            "psubh      %[ftmp11],  %[ftmp9],           %[ftmp10]       \n\t"
            "gssdlc1    %[ftmp11],  0x07(%[diff])                       \n\t"
            "gssdrc1    %[ftmp11],  0x00(%[diff])                       \n\t"
            MMI_ADDU(%[diff], %[diff], %[diff_stride])
            "punpcklbh  %[ftmp9],   %[ftmp3],           %[ftmp0]        \n\t"
            "punpcklbh  %[ftmp10],  %[ftmp4],           %[ftmp0]        \n\t"
            "psubh      %[ftmp11],  %[ftmp9],           %[ftmp10]       \n\t"
            "gssdlc1    %[ftmp11],  0x07(%[diff])                       \n\t"
            "gssdrc1    %[ftmp11],  0x00(%[diff])                       \n\t"
            MMI_ADDU(%[diff], %[diff], %[diff_stride])
            "punpcklbh  %[ftmp9],   %[ftmp5],           %[ftmp0]        \n\t"
            "punpcklbh  %[ftmp10],  %[ftmp6],           %[ftmp0]        \n\t"
            "psubh      %[ftmp11],  %[ftmp9],           %[ftmp10]       \n\t"
            "gssdlc1    %[ftmp11],  0x07(%[diff])                       \n\t"
            "gssdrc1    %[ftmp11],  0x00(%[diff])                       \n\t"
            MMI_ADDU(%[diff], %[diff], %[diff_stride])
            "punpcklbh  %[ftmp9],   %[ftmp7],           %[ftmp0]        \n\t"
            "punpcklbh  %[ftmp10],  %[ftmp8],           %[ftmp0]        \n\t"
            "psubh      %[ftmp11],  %[ftmp9],           %[ftmp10]       \n\t"
            "gssdlc1    %[ftmp11],  0x07(%[diff])                       \n\t"
            "gssdrc1    %[ftmp11],  0x00(%[diff])                       \n\t"
            : [ftmp0] "=&f"(ftmp[0]), [ftmp1] "=&f"(ftmp[1]),
              [ftmp2] "=&f"(ftmp[2]), [ftmp3] "=&f"(ftmp[3]),
              [ftmp4] "=&f"(ftmp[4]), [ftmp5] "=&f"(ftmp[5]),
              [ftmp6] "=&f"(ftmp[6]), [ftmp7] "=&f"(ftmp[7]),
              [ftmp8] "=&f"(ftmp[8]), [ftmp9] "=&f"(ftmp[9]),
              [ftmp10] "=&f"(ftmp[10]), [ftmp11] "=&f"(ftmp[11]),
#if _MIPS_SIM == _ABIO32
              [tmp0] "=&r"(tmp[0]),
#endif
              [src] "+&r"(src), [pred] "+&r"(pred), [diff] "+&r"(diff)
            : [src_stride] "r"((mips_reg)src_stride),
              [pred_stride] "r"((mips_reg)pred_stride),
              [diff_stride] "r"((mips_reg)(diff_stride * 2))
            : "memory");
        break;
      case 8:
        __asm__ volatile(
            "pxor       %[ftmp0],   %[ftmp0],           %[ftmp0]        \n\t"
            "li         %[tmp0],    0x02                                \n\t"
            "1:                                                         \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp2],   0x07(%[pred])                       \n\t"
            "gsldrc1    %[ftmp2],   0x00(%[pred])                       \n\t"
            MMI_ADDU(%[src], %[src], %[src_stride])
            MMI_ADDU(%[pred], %[pred], %[pred_stride])
            "gsldlc1    %[ftmp3],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp3],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp4],   0x07(%[pred])                       \n\t"
            "gsldrc1    %[ftmp4],   0x00(%[pred])                       \n\t"
            MMI_ADDU(%[src], %[src], %[src_stride])
            MMI_ADDU(%[pred], %[pred], %[pred_stride])
            "gsldlc1    %[ftmp5],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp5],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp6],   0x07(%[pred])                       \n\t"
            "gsldrc1    %[ftmp6],   0x00(%[pred])                       \n\t"
            MMI_ADDU(%[src], %[src], %[src_stride])
            MMI_ADDU(%[pred], %[pred], %[pred_stride])
            "gsldlc1    %[ftmp7],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp7],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp8],   0x07(%[pred])                       \n\t"
            "gsldrc1    %[ftmp8],   0x00(%[pred])                       \n\t"
            MMI_ADDU(%[src], %[src], %[src_stride])
            MMI_ADDU(%[pred], %[pred], %[pred_stride])
            "punpcklbh  %[ftmp9],   %[ftmp1],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp10],  %[ftmp1],           %[ftmp0]        \n\t"
            "punpcklbh  %[ftmp11],  %[ftmp2],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp12],  %[ftmp2],           %[ftmp0]        \n\t"
            "psubsh     %[ftmp9],   %[ftmp9],           %[ftmp11]       \n\t"
            "psubsh     %[ftmp10],  %[ftmp10],          %[ftmp12]       \n\t"
            "gssdlc1    %[ftmp9],   0x07(%[diff])                       \n\t"
            "gssdrc1    %[ftmp9],   0x00(%[diff])                       \n\t"
            "gssdlc1    %[ftmp10],  0x0f(%[diff])                       \n\t"
            "gssdrc1    %[ftmp10],  0x08(%[diff])                       \n\t"
            MMI_ADDU(%[diff], %[diff], %[diff_stride])
            "punpcklbh  %[ftmp9],   %[ftmp3],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp10],  %[ftmp3],           %[ftmp0]        \n\t"
            "punpcklbh  %[ftmp11],  %[ftmp4],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp12],  %[ftmp4],           %[ftmp0]        \n\t"
            "psubsh     %[ftmp9],   %[ftmp9],           %[ftmp11]       \n\t"
            "psubsh     %[ftmp10],  %[ftmp10],          %[ftmp12]       \n\t"
            "gssdlc1    %[ftmp9],   0x07(%[diff])                       \n\t"
            "gssdrc1    %[ftmp9],   0x00(%[diff])                       \n\t"
            "gssdlc1    %[ftmp10],  0x0f(%[diff])                       \n\t"
            "gssdrc1    %[ftmp10],  0x08(%[diff])                       \n\t"
            MMI_ADDU(%[diff], %[diff], %[diff_stride])
            "punpcklbh  %[ftmp9],   %[ftmp5],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp10],  %[ftmp5],           %[ftmp0]        \n\t"
            "punpcklbh  %[ftmp11],  %[ftmp6],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp12],  %[ftmp6],           %[ftmp0]        \n\t"
            "psubsh     %[ftmp9],   %[ftmp9],           %[ftmp11]       \n\t"
            "psubsh     %[ftmp10],  %[ftmp10],          %[ftmp12]       \n\t"
            "gssdlc1    %[ftmp9],   0x07(%[diff])                       \n\t"
            "gssdrc1    %[ftmp9],   0x00(%[diff])                       \n\t"
            "gssdlc1    %[ftmp10],  0x0f(%[diff])                       \n\t"
            "gssdrc1    %[ftmp10],  0x08(%[diff])                       \n\t"
            MMI_ADDU(%[diff], %[diff], %[diff_stride])
            "punpcklbh  %[ftmp9],   %[ftmp7],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp10],  %[ftmp7],           %[ftmp0]        \n\t"
            "punpcklbh  %[ftmp11],  %[ftmp8],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp12],  %[ftmp8],           %[ftmp0]        \n\t"
            "psubsh     %[ftmp9],   %[ftmp9],           %[ftmp11]       \n\t"
            "psubsh     %[ftmp10],  %[ftmp10],          %[ftmp12]       \n\t"
            "gssdlc1    %[ftmp9],   0x07(%[diff])                       \n\t"
            "gssdrc1    %[ftmp9],   0x00(%[diff])                       \n\t"
            "gssdlc1    %[ftmp10],  0x0f(%[diff])                       \n\t"
            "gssdrc1    %[ftmp10],  0x08(%[diff])                       \n\t"
            MMI_ADDU(%[diff], %[diff], %[diff_stride])
            "addiu      %[tmp0],    %[tmp0],            -0x01           \n\t"
            "bnez       %[tmp0],    1b                                  \n\t"
            : [ftmp0] "=&f"(ftmp[0]), [ftmp1] "=&f"(ftmp[1]),
              [ftmp2] "=&f"(ftmp[2]), [ftmp3] "=&f"(ftmp[3]),
              [ftmp4] "=&f"(ftmp[4]), [ftmp5] "=&f"(ftmp[5]),
              [ftmp6] "=&f"(ftmp[6]), [ftmp7] "=&f"(ftmp[7]),
              [ftmp8] "=&f"(ftmp[8]), [ftmp9] "=&f"(ftmp[9]),
              [ftmp10] "=&f"(ftmp[10]), [ftmp11] "=&f"(ftmp[11]),
              [ftmp12] "=&f"(ftmp[12]), [tmp0] "=&r"(tmp[0]), [src] "+&r"(src),
              [pred] "+&r"(pred), [diff] "+&r"(diff)
            : [pred_stride] "r"((mips_reg)pred_stride),
              [src_stride] "r"((mips_reg)src_stride),
              [diff_stride] "r"((mips_reg)(diff_stride * 2))
            : "memory");
        break;
      case 16:
        __asm__ volatile(
            "pxor       %[ftmp0],   %[ftmp0],           %[ftmp0]        \n\t"
            "li         %[tmp0],    0x08                                \n\t"
            "1:                                                         \n\t"
            "gsldlc1    %[ftmp1],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp1],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp2],   0x07(%[pred])                       \n\t"
            "gsldrc1    %[ftmp2],   0x00(%[pred])                       \n\t"
            "gsldlc1    %[ftmp3],   0x0f(%[src])                        \n\t"
            "gsldrc1    %[ftmp3],   0x08(%[src])                        \n\t"
            "gsldlc1    %[ftmp4],   0x0f(%[pred])                       \n\t"
            "gsldrc1    %[ftmp4],   0x08(%[pred])                       \n\t"
            MMI_ADDU(%[src], %[src], %[src_stride])
            MMI_ADDU(%[pred], %[pred], %[pred_stride])
            "gsldlc1    %[ftmp5],   0x07(%[src])                        \n\t"
            "gsldrc1    %[ftmp5],   0x00(%[src])                        \n\t"
            "gsldlc1    %[ftmp6],   0x07(%[pred])                       \n\t"
            "gsldrc1    %[ftmp6],   0x00(%[pred])                       \n\t"
            "gsldlc1    %[ftmp7],   0x0f(%[src])                        \n\t"
            "gsldrc1    %[ftmp7],   0x08(%[src])                        \n\t"
            "gsldlc1    %[ftmp8],   0x0f(%[pred])                       \n\t"
            "gsldrc1    %[ftmp8],   0x08(%[pred])                       \n\t"
            MMI_ADDU(%[src], %[src], %[src_stride])
            MMI_ADDU(%[pred], %[pred], %[pred_stride])
            "punpcklbh  %[ftmp9],   %[ftmp1],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp10],  %[ftmp1],           %[ftmp0]        \n\t"
            "punpcklbh  %[ftmp11],  %[ftmp2],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp12],  %[ftmp2],           %[ftmp0]        \n\t"
            "psubsh     %[ftmp9],   %[ftmp9],           %[ftmp11]       \n\t"
            "psubsh     %[ftmp10],  %[ftmp10],          %[ftmp12]       \n\t"
            "gssdlc1    %[ftmp9],   0x07(%[diff])                       \n\t"
            "gssdrc1    %[ftmp9],   0x00(%[diff])                       \n\t"
            "gssdlc1    %[ftmp10],  0x0f(%[diff])                       \n\t"
            "gssdrc1    %[ftmp10],  0x08(%[diff])                       \n\t"
            "punpcklbh  %[ftmp9],   %[ftmp3],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp10],  %[ftmp3],           %[ftmp0]        \n\t"
            "punpcklbh  %[ftmp11],  %[ftmp4],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp12],  %[ftmp4],           %[ftmp0]        \n\t"
            "psubsh     %[ftmp9],   %[ftmp9],           %[ftmp11]       \n\t"
            "psubsh     %[ftmp10],  %[ftmp10],          %[ftmp12]       \n\t"
            "gssdlc1    %[ftmp9],   0x17(%[diff])                       \n\t"
            "gssdrc1    %[ftmp9],   0x10(%[diff])                       \n\t"
            "gssdlc1    %[ftmp10],  0x1f(%[diff])                       \n\t"
            "gssdrc1    %[ftmp10],  0x18(%[diff])                       \n\t"
            MMI_ADDU(%[diff], %[diff], %[diff_stride])
            "punpcklbh  %[ftmp9],   %[ftmp5],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp10],  %[ftmp5],           %[ftmp0]        \n\t"
            "punpcklbh  %[ftmp11],  %[ftmp6],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp12],  %[ftmp6],           %[ftmp0]        \n\t"
            "psubsh     %[ftmp9],   %[ftmp9],           %[ftmp11]       \n\t"
            "psubsh     %[ftmp10],  %[ftmp10],          %[ftmp12]       \n\t"
            "gssdlc1    %[ftmp9],   0x07(%[diff])                       \n\t"
            "gssdrc1    %[ftmp9],   0x00(%[diff])                       \n\t"
            "gssdlc1    %[ftmp10],  0x0f(%[diff])                       \n\t"
            "gssdrc1    %[ftmp10],  0x08(%[diff])                       \n\t"
            "punpcklbh  %[ftmp9],   %[ftmp7],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp10],  %[ftmp7],           %[ftmp0]        \n\t"
            "punpcklbh  %[ftmp11],  %[ftmp8],           %[ftmp0]        \n\t"
            "punpckhbh  %[ftmp12],  %[ftmp8],           %[ftmp0]        \n\t"
            "psubsh     %[ftmp9],   %[ftmp9],           %[ftmp11]       \n\t"
            "psubsh     %[ftmp10],  %[ftmp10],          %[ftmp12]       \n\t"
            "gssdlc1    %[ftmp9],   0x17(%[diff])                       \n\t"
            "gssdrc1    %[ftmp9],   0x10(%[diff])                       \n\t"
            "gssdlc1    %[ftmp10],  0x1f(%[diff])                       \n\t"
            "gssdrc1    %[ftmp10],  0x18(%[diff])                       \n\t"
            MMI_ADDU(%[diff], %[diff], %[diff_stride])
            "addiu      %[tmp0],    %[tmp0],            -0x01           \n\t"
            "bnez       %[tmp0],    1b                                  \n\t"
            : [ftmp0] "=&f"(ftmp[0]), [ftmp1] "=&f"(ftmp[1]),
              [ftmp2] "=&f"(ftmp[2]), [ftmp3] "=&f"(ftmp[3]),
              [ftmp4] "=&f"(ftmp[4]), [ftmp5] "=&f"(ftmp[5]),
              [ftmp6] "=&f"(ftmp[6]), [ftmp7] "=&f"(ftmp[7]),
              [ftmp8] "=&f"(ftmp[8]), [ftmp9] "=&f"(ftmp[9]),
              [ftmp10] "=&f"(ftmp[10]), [ftmp11] "=&f"(ftmp[11]),
              [ftmp12] "=&f"(ftmp[12]), [tmp0] "=&r"(tmp[0]), [src] "+&r"(src),
              [pred] "+&r"(pred), [diff] "+&r"(diff)
            : [pred_stride] "r"((mips_reg)pred_stride),
              [src_stride] "r"((mips_reg)src_stride),
              [diff_stride] "r"((mips_reg)(diff_stride * 2))
            : "memory");
        break;
      case 32:
        vpx_subtract_block_c(rows, cols, diff, diff_stride, src, src_stride,
                             pred, pred_stride);
        break;
      case 64:
        vpx_subtract_block_c(rows, cols, diff, diff_stride, src, src_stride,
                             pred, pred_stride);
        break;
      default:
        vpx_subtract_block_c(rows, cols, diff, diff_stride, src, src_stride,
                             pred, pred_stride);
        break;
    }
  } else {
    vpx_subtract_block_c(rows, cols, diff, diff_stride, src, src_stride, pred,
                         pred_stride);
  }
}
