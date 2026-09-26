/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp8_rtcd.h"
#include "vpx_ports/mem.h"
#include "vpx_ports/asmdefs_mmi.h"

#define TRANSPOSE_4H \
  "pxor          %[ftmp0],    %[ftmp0],    %[ftmp0]          \n\t" \
  MMI_LI(%[tmp0], 0x93)                                            \
  "mtc1          %[tmp0],     %[ftmp10]                      \n\t" \
  "punpcklhw     %[ftmp5],    %[ftmp1],    %[ftmp0]          \n\t" \
  "punpcklhw     %[ftmp9],    %[ftmp2],    %[ftmp0]          \n\t" \
  "pshufh        %[ftmp9],    %[ftmp9],    %[ftmp10]         \n\t" \
  "por           %[ftmp5],    %[ftmp5],    %[ftmp9]          \n\t" \
  "punpckhhw     %[ftmp6],    %[ftmp1],    %[ftmp0]          \n\t" \
  "punpckhhw     %[ftmp9],    %[ftmp2],    %[ftmp0]          \n\t" \
  "pshufh        %[ftmp9],    %[ftmp9],    %[ftmp10]         \n\t" \
  "por           %[ftmp6],    %[ftmp6],    %[ftmp9]          \n\t" \
  "punpcklhw     %[ftmp7],    %[ftmp3],    %[ftmp0]          \n\t" \
  "punpcklhw     %[ftmp9],    %[ftmp4],    %[ftmp0]          \n\t" \
  "pshufh        %[ftmp9],    %[ftmp9],    %[ftmp10]         \n\t" \
  "por           %[ftmp7],    %[ftmp7],    %[ftmp9]          \n\t" \
  "punpckhhw     %[ftmp8],    %[ftmp3],    %[ftmp0]          \n\t" \
  "punpckhhw     %[ftmp9],    %[ftmp4],    %[ftmp0]          \n\t" \
  "pshufh        %[ftmp9],    %[ftmp9],    %[ftmp10]         \n\t" \
  "por           %[ftmp8],    %[ftmp8],    %[ftmp9]          \n\t" \
  "punpcklwd     %[ftmp1],    %[ftmp5],    %[ftmp7]          \n\t" \
  "punpckhwd     %[ftmp2],    %[ftmp5],    %[ftmp7]          \n\t" \
  "punpcklwd     %[ftmp3],    %[ftmp6],    %[ftmp8]          \n\t" \
  "punpckhwd     %[ftmp4],    %[ftmp6],    %[ftmp8]          \n\t"

void vp8_short_idct4x4llm_mmi(int16_t *input, unsigned char *pred_ptr,
                              int pred_stride, unsigned char *dst_ptr,
                              int dst_stride) {
  double ftmp[12];
  uint64_t tmp[1];
  double ff_ph_04, ff_ph_4e7b, ff_ph_22a3;

  __asm__ volatile (
    "dli        %[tmp0],    0x0004000400040004                  \n\t"
    "dmtc1      %[tmp0],    %[ff_ph_04]                         \n\t"
    "dli        %[tmp0],    0x4e7b4e7b4e7b4e7b                  \n\t"
    "dmtc1      %[tmp0],    %[ff_ph_4e7b]                       \n\t"
    "dli        %[tmp0],    0x22a322a322a322a3                  \n\t"
    "dmtc1      %[tmp0],    %[ff_ph_22a3]                       \n\t"
    MMI_LI(%[tmp0], 0x02)
    "dmtc1      %[tmp0],    %[ftmp11]                           \n\t"
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"

    "gsldlc1    %[ftmp1],   0x07(%[ip])                         \n\t"
    "gsldrc1    %[ftmp1],   0x00(%[ip])                         \n\t"
    "gsldlc1    %[ftmp2],   0x0f(%[ip])                         \n\t"
    "gsldrc1    %[ftmp2],   0x08(%[ip])                         \n\t"
    "gsldlc1    %[ftmp3],   0x17(%[ip])                         \n\t"
    "gsldrc1    %[ftmp3],   0x10(%[ip])                         \n\t"
    "gsldlc1    %[ftmp4],   0x1f(%[ip])                         \n\t"
    "gsldrc1    %[ftmp4],   0x18(%[ip])                         \n\t"

    // ip[0...3] + ip[8...11]
    "paddh      %[ftmp5],   %[ftmp1],       %[ftmp3]            \n\t"
    // ip[0...3] - ip[8...11]
    "psubh      %[ftmp6],   %[ftmp1],       %[ftmp3]            \n\t"
    // (ip[12...15] * sinpi8sqrt2) >> 16
    "psllh      %[ftmp9],   %[ftmp4],       %[ftmp11]           \n\t"
    "pmulhh     %[ftmp7],   %[ftmp9],       %[ff_ph_22a3]       \n\t"
    // (ip[ 4... 7] * sinpi8sqrt2) >> 16
    "psllh      %[ftmp9],   %[ftmp2],       %[ftmp11]           \n\t"
    "pmulhh     %[ftmp8],   %[ftmp9],       %[ff_ph_22a3]       \n\t"
    // ip[ 4... 7] + ((ip[ 4... 7] * cospi8sqrt2minus1) >> 16)
    "pmulhh     %[ftmp9],   %[ftmp2],       %[ff_ph_4e7b]       \n\t"
    "paddh      %[ftmp9],   %[ftmp9],       %[ftmp2]            \n\t"
    // ip[12...15] + ((ip[12...15] * cospi8sqrt2minus1) >> 16)
    "pmulhh     %[ftmp10],  %[ftmp4],       %[ff_ph_4e7b]       \n\t"
    "paddh      %[ftmp10],  %[ftmp10],      %[ftmp4]            \n\t"

    "paddh      %[ftmp1],   %[ftmp5],       %[ftmp7]            \n\t"
    "paddh      %[ftmp1],   %[ftmp1],       %[ftmp9]            \n\t"
    "paddh      %[ftmp2],   %[ftmp6],       %[ftmp8]            \n\t"
    "psubh      %[ftmp2],   %[ftmp2],       %[ftmp10]           \n\t"
    "psubh      %[ftmp3],   %[ftmp6],       %[ftmp8]            \n\t"
    "paddh      %[ftmp3],   %[ftmp3],       %[ftmp10]           \n\t"
    "psubh      %[ftmp4],   %[ftmp5],       %[ftmp7]            \n\t"
    "psubh      %[ftmp4],   %[ftmp4],       %[ftmp9]            \n\t"

    TRANSPOSE_4H
    // a
    "paddh      %[ftmp5],   %[ftmp1],       %[ftmp3]            \n\t"
    // b
    "psubh      %[ftmp6],   %[ftmp1],       %[ftmp3]            \n\t"
    // c
    "psllh      %[ftmp9],   %[ftmp2],       %[ftmp11]           \n\t"
    "pmulhh     %[ftmp9],   %[ftmp9],       %[ff_ph_22a3]       \n\t"
    "psubh      %[ftmp7],   %[ftmp9],       %[ftmp4]            \n\t"
    "pmulhh     %[ftmp10],  %[ftmp4],       %[ff_ph_4e7b]       \n\t"
    "psubh      %[ftmp7],   %[ftmp7],       %[ftmp10]           \n\t"
    // d
    "psllh      %[ftmp9],   %[ftmp4],       %[ftmp11]           \n\t"
    "pmulhh     %[ftmp9],   %[ftmp9],       %[ff_ph_22a3]       \n\t"
    "paddh      %[ftmp8],   %[ftmp9],       %[ftmp2]            \n\t"
    "pmulhh     %[ftmp10],  %[ftmp2],       %[ff_ph_4e7b]       \n\t"
    "paddh      %[ftmp8],   %[ftmp8],       %[ftmp10]           \n\t"

    MMI_LI(%[tmp0], 0x03)
    "mtc1       %[tmp0],    %[ftmp11]                           \n\t"
    // a + d
    "paddh      %[ftmp1],   %[ftmp5],       %[ftmp8]            \n\t"
    "paddh      %[ftmp1],   %[ftmp1],       %[ff_ph_04]         \n\t"
    "psrah      %[ftmp1],   %[ftmp1],       %[ftmp11]           \n\t"
    // b + c
    "paddh      %[ftmp2],   %[ftmp6],       %[ftmp7]            \n\t"
    "paddh      %[ftmp2],   %[ftmp2],       %[ff_ph_04]         \n\t"
    "psrah      %[ftmp2],   %[ftmp2],       %[ftmp11]           \n\t"
    // b - c
    "psubh      %[ftmp3],   %[ftmp6],       %[ftmp7]            \n\t"
    "paddh      %[ftmp3],   %[ftmp3],       %[ff_ph_04]         \n\t"
    "psrah      %[ftmp3],   %[ftmp3],       %[ftmp11]           \n\t"
    // a - d
    "psubh      %[ftmp4],   %[ftmp5],       %[ftmp8]            \n\t"
    "paddh      %[ftmp4],   %[ftmp4],       %[ff_ph_04]         \n\t"
    "psrah      %[ftmp4],   %[ftmp4],       %[ftmp11]           \n\t"

    TRANSPOSE_4H
#if _MIPS_SIM == _ABIO32
    "ulw        %[tmp0],    0x00(%[pred_prt])                   \n\t"
    "mtc1       %[tmp0],    %[ftmp5]                            \n\t"
#else
    "gslwlc1    %[ftmp5],   0x03(%[pred_ptr])                   \n\t"
    "gslwrc1    %[ftmp5],   0x00(%[pred_ptr])                   \n\t"
#endif
    "punpcklbh  %[ftmp5],   %[ftmp5],       %[ftmp0]            \n\t"
    "paddh      %[ftmp1],   %[ftmp1],       %[ftmp5]            \n\t"
    "packushb   %[ftmp1],   %[ftmp1],       %[ftmp0]            \n\t"
    "gsswlc1    %[ftmp1],   0x03(%[dst_ptr])                    \n\t"
    "gsswrc1    %[ftmp1],   0x00(%[dst_ptr])                    \n\t"
    MMI_ADDU(%[pred_ptr], %[pred_ptr], %[pred_stride])
    MMI_ADDU(%[dst_ptr], %[dst_ptr], %[dst_stride])

#if _MIPS_SIM == _ABIO32
    "ulw        %[tmp0],    0x00(%[pred_prt])                   \n\t"
    "mtc1       %[tmp0],    %[ftmp6]                            \n\t"
#else
    "gslwlc1    %[ftmp6],   0x03(%[pred_ptr])                   \n\t"
    "gslwrc1    %[ftmp6],   0x00(%[pred_ptr])                   \n\t"
#endif
    "punpcklbh  %[ftmp6],   %[ftmp6],       %[ftmp0]            \n\t"
    "paddh      %[ftmp2],   %[ftmp2],       %[ftmp6]            \n\t"
    "packushb   %[ftmp2],   %[ftmp2],       %[ftmp0]            \n\t"
    "gsswlc1    %[ftmp2],   0x03(%[dst_ptr])                    \n\t"
    "gsswrc1    %[ftmp2],   0x00(%[dst_ptr])                    \n\t"
    MMI_ADDU(%[pred_ptr], %[pred_ptr], %[pred_stride])
    MMI_ADDU(%[dst_ptr], %[dst_ptr], %[dst_stride])

#if _MIPS_SIM == _ABIO32
    "ulw        %[tmp0],    0x00(%[pred_prt])                   \n\t"
    "mtc1       %[tmp0],    %[ftmp7]                            \n\t"
#else
    "gslwlc1    %[ftmp7],   0x03(%[pred_ptr])                   \n\t"
    "gslwrc1    %[ftmp7],   0x00(%[pred_ptr])                   \n\t"
#endif
    "punpcklbh  %[ftmp7],   %[ftmp7],       %[ftmp0]            \n\t"
    "paddh      %[ftmp3],   %[ftmp3],       %[ftmp7]            \n\t"
    "packushb   %[ftmp3],   %[ftmp3],       %[ftmp0]            \n\t"
    "gsswlc1    %[ftmp3],   0x03(%[dst_ptr])                    \n\t"
    "gsswrc1    %[ftmp3],   0x00(%[dst_ptr])                    \n\t"
    MMI_ADDU(%[pred_ptr], %[pred_ptr], %[pred_stride])
    MMI_ADDU(%[dst_ptr], %[dst_ptr], %[dst_stride])

#if _MIPS_SIM == _ABIO32
    "ulw        %[tmp0],    0x00(%[pred_prt])                   \n\t"
    "mtc1       %[tmp0],    %[ftmp8]                            \n\t"
#else
    "gslwlc1    %[ftmp8],   0x03(%[pred_ptr])                   \n\t"
    "gslwrc1    %[ftmp8],   0x00(%[pred_ptr])                   \n\t"
#endif
    "punpcklbh  %[ftmp8],   %[ftmp8],       %[ftmp0]            \n\t"
    "paddh      %[ftmp4],   %[ftmp4],       %[ftmp8]            \n\t"
    "packushb   %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t"
    "gsswlc1    %[ftmp4],   0x03(%[dst_ptr])                    \n\t"
    "gsswrc1    %[ftmp4],   0x00(%[dst_ptr])                    \n\t"
    : [ftmp0]"=&f"(ftmp[0]), [ftmp1]"=&f"(ftmp[1]), [ftmp2]"=&f"(ftmp[2]),
      [ftmp3]"=&f"(ftmp[3]), [ftmp4]"=&f"(ftmp[4]), [ftmp5]"=&f"(ftmp[5]),
      [ftmp6]"=&f"(ftmp[6]), [ftmp7]"=&f"(ftmp[7]), [ftmp8]"=&f"(ftmp[8]),
      [ftmp9]"=&f"(ftmp[9]), [ftmp10]"=&f"(ftmp[10]),
      [ftmp11]"=&f"(ftmp[11]), [tmp0]"=&r"(tmp[0]),
      [pred_ptr]"+&r"(pred_ptr), [dst_ptr]"+&r"(dst_ptr),
      [ff_ph_4e7b]"=&f"(ff_ph_4e7b), [ff_ph_04]"=&f"(ff_ph_04),
      [ff_ph_22a3]"=&f"(ff_ph_22a3)
    : [ip]"r"(input),
      [pred_stride]"r"((mips_reg)pred_stride),
      [dst_stride]"r"((mips_reg)dst_stride)
    : "memory"
  );
}

void vp8_dc_only_idct_add_mmi(int16_t input_dc, unsigned char *pred_ptr,
                              int pred_stride, unsigned char *dst_ptr,
                              int dst_stride) {
  int a0 = ((input_dc + 4) >> 3);
  double a1, ftmp[5];
  int low32;

  __asm__ volatile (
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]        \n\t"
    "dmtc1      %[a0],      %[a1]                           \n\t"
    "pshufh     %[a1],      %[a1],          %[ftmp0]        \n\t"
    "ulw        %[low32],   0x00(%[pred_ptr])               \n\t"
    "mtc1       %[low32],   %[ftmp1]                        \n\t"
    "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]        \n\t"
    "paddsh     %[ftmp2],   %[ftmp2],       %[a1]           \n\t"
    "packushb   %[ftmp1],   %[ftmp2],       %[ftmp0]        \n\t"
    "gsswlc1    %[ftmp1],   0x03(%[dst_ptr])                \n\t"
    "gsswrc1    %[ftmp1],   0x00(%[dst_ptr])                \n\t"

    MMI_ADDU(%[pred_ptr], %[pred_ptr], %[pred_stride])
    MMI_ADDU(%[dst_ptr], %[dst_ptr], %[dst_stride])
    "ulw        %[low32],   0x00(%[pred_ptr])               \n\t"
    "mtc1       %[low32],   %[ftmp1]                        \n\t"
    "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]        \n\t"
    "paddsh     %[ftmp2],   %[ftmp2],       %[a1]           \n\t"
    "packushb   %[ftmp1],   %[ftmp2],       %[ftmp0]        \n\t"
    "gsswlc1    %[ftmp1],   0x03(%[dst_ptr])                \n\t"
    "gsswrc1    %[ftmp1],   0x00(%[dst_ptr])                \n\t"

    MMI_ADDU(%[pred_ptr], %[pred_ptr], %[pred_stride])
    MMI_ADDU(%[dst_ptr], %[dst_ptr], %[dst_stride])
    "ulw        %[low32],   0x00(%[pred_ptr])               \n\t"
    "mtc1       %[low32],   %[ftmp1]                        \n\t"
    "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]        \n\t"
    "paddsh     %[ftmp2],   %[ftmp2],       %[a1]           \n\t"
    "packushb   %[ftmp1],   %[ftmp2],       %[ftmp0]        \n\t"
    "gsswlc1    %[ftmp1],   0x03(%[dst_ptr])                \n\t"
    "gsswrc1    %[ftmp1],   0x00(%[dst_ptr])                \n\t"

    MMI_ADDU(%[pred_ptr], %[pred_ptr], %[pred_stride])
    MMI_ADDU(%[dst_ptr], %[dst_ptr], %[dst_stride])
    "ulw        %[low32],   0x00(%[pred_ptr])               \n\t"
    "mtc1       %[low32],   %[ftmp1]                        \n\t"
    "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]        \n\t"
    "paddsh     %[ftmp2],   %[ftmp2],       %[a1]           \n\t"
    "packushb   %[ftmp1],   %[ftmp2],       %[ftmp0]        \n\t"
    "gsswlc1    %[ftmp1],   0x03(%[dst_ptr])                \n\t"
    "gsswrc1    %[ftmp1],   0x00(%[dst_ptr])                \n\t"
    : [ftmp0]"=&f"(ftmp[0]), [ftmp1]"=&f"(ftmp[1]), [ftmp2]"=&f"(ftmp[2]),
      [ftmp3]"=&f"(ftmp[3]), [ftmp4]"=&f"(ftmp[4]), [low32]"=&r"(low32),
      [dst_ptr]"+&r"(dst_ptr), [pred_ptr]"+&r"(pred_ptr), [a1]"=&f"(a1)
    : [dst_stride]"r"((mips_reg)dst_stride),
      [pred_stride]"r"((mips_reg)pred_stride), [a0]"r"(a0)
    : "memory"
  );
}

void vp8_short_inv_walsh4x4_mmi(int16_t *input, int16_t *mb_dqcoeff) {
  int i;
  int16_t output[16];
  double ff_ph_03, ftmp[12];
  uint64_t tmp[1];

  __asm__ volatile (
    "dli        %[tmp0],    0x0003000300030003                  \n\t"
    "dmtc1      %[tmp0],    %[ff_ph_03]                         \n\t"
    MMI_LI(%[tmp0], 0x03)
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
    "dmtc1      %[tmp0],    %[ftmp11]                           \n\t"
    "gsldlc1    %[ftmp1],   0x07(%[ip])                         \n\t"
    "gsldrc1    %[ftmp1],   0x00(%[ip])                         \n\t"
    "gsldlc1    %[ftmp2],   0x0f(%[ip])                         \n\t"
    "gsldrc1    %[ftmp2],   0x08(%[ip])                         \n\t"
    "gsldlc1    %[ftmp3],   0x17(%[ip])                         \n\t"
    "gsldrc1    %[ftmp3],   0x10(%[ip])                         \n\t"
    "gsldlc1    %[ftmp4],   0x1f(%[ip])                         \n\t"
    "gsldrc1    %[ftmp4],   0x18(%[ip])                         \n\t"
    "paddh      %[ftmp5],   %[ftmp1],       %[ftmp2]            \n\t"
    "psubh      %[ftmp6],   %[ftmp1],       %[ftmp2]            \n\t"
    "paddh      %[ftmp7],   %[ftmp3],       %[ftmp4]            \n\t"
    "psubh      %[ftmp8],   %[ftmp3],       %[ftmp4]            \n\t"

    "paddh      %[ftmp1],   %[ftmp5],       %[ftmp7]            \n\t"
    "psubh      %[ftmp2],   %[ftmp5],       %[ftmp7]            \n\t"
    "psubh      %[ftmp3],   %[ftmp6],       %[ftmp8]            \n\t"
    "paddh      %[ftmp4],   %[ftmp6],       %[ftmp8]            \n\t"

    TRANSPOSE_4H
    // a
    "paddh      %[ftmp5],   %[ftmp1],       %[ftmp4]            \n\t"
    // d
    "psubh      %[ftmp6],   %[ftmp1],       %[ftmp4]            \n\t"
    // b
    "paddh      %[ftmp7],   %[ftmp2],       %[ftmp3]            \n\t"
    // c
    "psubh      %[ftmp8],   %[ftmp2],       %[ftmp3]            \n\t"

    "paddh      %[ftmp1],   %[ftmp5],       %[ftmp7]            \n\t"
    "paddh      %[ftmp2],   %[ftmp6],       %[ftmp8]            \n\t"
    "psubh      %[ftmp3],   %[ftmp5],       %[ftmp7]            \n\t"
    "psubh      %[ftmp4],   %[ftmp6],       %[ftmp8]            \n\t"

    "paddh      %[ftmp1],   %[ftmp1],       %[ff_ph_03]         \n\t"
    "psrah      %[ftmp1],   %[ftmp1],       %[ftmp11]           \n\t"
    "paddh      %[ftmp2],   %[ftmp2],       %[ff_ph_03]         \n\t"
    "psrah      %[ftmp2],   %[ftmp2],       %[ftmp11]           \n\t"
    "paddh      %[ftmp3],   %[ftmp3],       %[ff_ph_03]         \n\t"
    "psrah      %[ftmp3],   %[ftmp3],       %[ftmp11]           \n\t"
    "paddh      %[ftmp4],   %[ftmp4],       %[ff_ph_03]         \n\t"
    "psrah      %[ftmp4],   %[ftmp4],       %[ftmp11]           \n\t"

    TRANSPOSE_4H
    "gssdlc1    %[ftmp1],   0x07(%[op])                         \n\t"
    "gssdrc1    %[ftmp1],   0x00(%[op])                         \n\t"
    "gssdlc1    %[ftmp2],   0x0f(%[op])                         \n\t"
    "gssdrc1    %[ftmp2],   0x08(%[op])                         \n\t"
    "gssdlc1    %[ftmp3],   0x17(%[op])                         \n\t"
    "gssdrc1    %[ftmp3],   0x10(%[op])                         \n\t"
    "gssdlc1    %[ftmp4],   0x1f(%[op])                         \n\t"
    "gssdrc1    %[ftmp4],   0x18(%[op])                         \n\t"
    : [ftmp0]"=&f"(ftmp[0]), [ftmp1]"=&f"(ftmp[1]), [ftmp2]"=&f"(ftmp[2]),
      [ftmp3]"=&f"(ftmp[3]), [ftmp4]"=&f"(ftmp[4]), [ftmp5]"=&f"(ftmp[5]),
      [ftmp6]"=&f"(ftmp[6]), [ftmp7]"=&f"(ftmp[7]), [ftmp8]"=&f"(ftmp[8]),
      [ftmp9]"=&f"(ftmp[9]), [ftmp10]"=&f"(ftmp[10]),
      [ftmp11]"=&f"(ftmp[11]), [tmp0]"=&r"(tmp[0]), [ff_ph_03]"=&f"(ff_ph_03)
    : [ip]"r"(input), [op]"r"(output)
    : "memory"
  );

  for (i = 0; i < 16; i++) {
    mb_dqcoeff[i * 16] = output[i];
  }
}
