/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/asmdefs_mmi.h"
#include "vp8/encoder/onyx_int.h"
#include "vp8/encoder/quantize.h"
#include "vp8/common/quant_common.h"

#define REGULAR_SELECT_EOB(i, rc)                                        \
  z = coeff_ptr[rc];                                                     \
  sz = (z >> 31);                                                        \
  x = (z ^ sz) - sz;                                                     \
  zbin = zbin_ptr[rc] + *(zbin_boost_ptr++) + zbin_oq_value;             \
  if (x >= zbin) {                                                       \
    x += round_ptr[rc];                                                  \
    y = ((((x * quant_ptr[rc]) >> 16) + x) * quant_shift_ptr[rc]) >> 16; \
    if (y) {                                                             \
      x = (y ^ sz) - sz;                                                 \
      qcoeff_ptr[rc] = x;                                                \
      dqcoeff_ptr[rc] = x * dequant_ptr[rc];                             \
      eob = i;                                                           \
      zbin_boost_ptr = b->zrun_zbin_boost;                               \
    }                                                                    \
  }

void vp8_fast_quantize_b_mmi(BLOCK *b, BLOCKD *d) {
  const int16_t *coeff_ptr = b->coeff;
  const int16_t *round_ptr = b->round;
  const int16_t *quant_ptr = b->quant_fast;
  int16_t *qcoeff_ptr = d->qcoeff;
  int16_t *dqcoeff_ptr = d->dqcoeff;
  const int16_t *dequant_ptr = d->dequant;
  const int16_t *inv_zig_zag = vp8_default_inv_zig_zag;

  double ftmp[13];
  uint64_t tmp[1];
  int64_t eob = 0;
  double ones;

  __asm__ volatile(
      // loop 0 ~ 7
      "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]        \n\t"
      "pcmpeqh    %[ones],    %[ones],        %[ones]         \n\t"
      "gsldlc1    %[ftmp1],   0x07(%[coeff_ptr])              \n\t"
      "gsldrc1    %[ftmp1],   0x00(%[coeff_ptr])              \n\t"
      "dli        %[tmp0],    0x0f                            \n\t"
      "dmtc1      %[tmp0],    %[ftmp9]                        \n\t"
      "gsldlc1    %[ftmp2],   0x0f(%[coeff_ptr])              \n\t"
      "gsldrc1    %[ftmp2],   0x08(%[coeff_ptr])              \n\t"

      "psrah      %[ftmp3],   %[ftmp1],       %[ftmp9]        \n\t"
      "pxor       %[ftmp1],   %[ftmp3],       %[ftmp1]        \n\t"
      "psubh      %[ftmp1],   %[ftmp1],       %[ftmp3]        \n\t"
      "psrah      %[ftmp4],   %[ftmp2],       %[ftmp9]        \n\t"
      "pxor       %[ftmp2],   %[ftmp4],       %[ftmp2]        \n\t"
      "psubh      %[ftmp2],   %[ftmp2],       %[ftmp4]        \n\t"

      "gsldlc1    %[ftmp5],   0x07(%[round_ptr])              \n\t"
      "gsldrc1    %[ftmp5],   0x00(%[round_ptr])              \n\t"
      "gsldlc1    %[ftmp6],   0x0f(%[round_ptr])              \n\t"
      "gsldrc1    %[ftmp6],   0x08(%[round_ptr])              \n\t"
      "paddh      %[ftmp5],   %[ftmp5],       %[ftmp1]        \n\t"
      "paddh      %[ftmp6],   %[ftmp6],       %[ftmp2]        \n\t"
      "gsldlc1    %[ftmp7],   0x07(%[quant_ptr])              \n\t"
      "gsldrc1    %[ftmp7],   0x00(%[quant_ptr])              \n\t"
      "gsldlc1    %[ftmp8],   0x0f(%[quant_ptr])              \n\t"
      "gsldrc1    %[ftmp8],   0x08(%[quant_ptr])              \n\t"
      "pmulhuh    %[ftmp5],   %[ftmp5],       %[ftmp7]        \n\t"
      "pmulhuh    %[ftmp6],   %[ftmp6],       %[ftmp8]        \n\t"

      "pxor       %[ftmp7],   %[ftmp5],       %[ftmp3]        \n\t"
      "pxor       %[ftmp8],   %[ftmp6],       %[ftmp4]        \n\t"
      "psubh      %[ftmp7],   %[ftmp7],       %[ftmp3]        \n\t"
      "psubh      %[ftmp8],   %[ftmp8],       %[ftmp4]        \n\t"
      "gssdlc1    %[ftmp7],   0x07(%[qcoeff_ptr])             \n\t"
      "gssdrc1    %[ftmp7],   0x00(%[qcoeff_ptr])             \n\t"
      "gssdlc1    %[ftmp8],   0x0f(%[qcoeff_ptr])             \n\t"
      "gssdrc1    %[ftmp8],   0x08(%[qcoeff_ptr])             \n\t"

      "gsldlc1    %[ftmp1],   0x07(%[inv_zig_zag])            \n\t"
      "gsldrc1    %[ftmp1],   0x00(%[inv_zig_zag])            \n\t"
      "gsldlc1    %[ftmp2],   0x0f(%[inv_zig_zag])            \n\t"
      "gsldrc1    %[ftmp2],   0x08(%[inv_zig_zag])            \n\t"
      "pcmpeqh    %[ftmp5],   %[ftmp5],       %[ftmp0]        \n\t"
      "pcmpeqh    %[ftmp6],   %[ftmp6],       %[ftmp0]        \n\t"
      "pxor       %[ftmp5],   %[ftmp5],       %[ones]         \n\t"
      "pxor       %[ftmp6],   %[ftmp6],       %[ones]         \n\t"
      "pand       %[ftmp5],   %[ftmp5],       %[ftmp1]        \n\t"
      "pand       %[ftmp6],   %[ftmp6],       %[ftmp2]        \n\t"
      "pmaxsh     %[ftmp10],  %[ftmp5],       %[ftmp6]        \n\t"

      "gsldlc1    %[ftmp5],   0x07(%[dequant_ptr])            \n\t"
      "gsldrc1    %[ftmp5],   0x00(%[dequant_ptr])            \n\t"
      "gsldlc1    %[ftmp6],   0x0f(%[dequant_ptr])            \n\t"
      "gsldrc1    %[ftmp6],   0x08(%[dequant_ptr])            \n\t"
      "pmullh     %[ftmp5],   %[ftmp5],       %[ftmp7]        \n\t"
      "pmullh     %[ftmp6],   %[ftmp6],       %[ftmp8]        \n\t"
      "gssdlc1    %[ftmp5],   0x07(%[dqcoeff_ptr])            \n\t"
      "gssdrc1    %[ftmp5],   0x00(%[dqcoeff_ptr])            \n\t"
      "gssdlc1    %[ftmp6],   0x0f(%[dqcoeff_ptr])            \n\t"
      "gssdrc1    %[ftmp6],   0x08(%[dqcoeff_ptr])            \n\t"

      // loop 8 ~ 15
      "gsldlc1    %[ftmp1],   0x17(%[coeff_ptr])              \n\t"
      "gsldrc1    %[ftmp1],   0x10(%[coeff_ptr])              \n\t"
      "gsldlc1    %[ftmp2],   0x1f(%[coeff_ptr])              \n\t"
      "gsldrc1    %[ftmp2],   0x18(%[coeff_ptr])              \n\t"

      "psrah      %[ftmp3],   %[ftmp1],       %[ftmp9]        \n\t"
      "pxor       %[ftmp1],   %[ftmp3],       %[ftmp1]        \n\t"
      "psubh      %[ftmp1],   %[ftmp1],       %[ftmp3]        \n\t"
      "psrah      %[ftmp4],   %[ftmp2],       %[ftmp9]        \n\t"
      "pxor       %[ftmp2],   %[ftmp4],       %[ftmp2]        \n\t"
      "psubh      %[ftmp2],   %[ftmp2],       %[ftmp4]        \n\t"

      "gsldlc1    %[ftmp5],   0x17(%[round_ptr])              \n\t"
      "gsldrc1    %[ftmp5],   0x10(%[round_ptr])              \n\t"
      "gsldlc1    %[ftmp6],   0x1f(%[round_ptr])              \n\t"
      "gsldrc1    %[ftmp6],   0x18(%[round_ptr])              \n\t"
      "paddh      %[ftmp5],   %[ftmp5],       %[ftmp1]        \n\t"
      "paddh      %[ftmp6],   %[ftmp6],       %[ftmp2]        \n\t"
      "gsldlc1    %[ftmp7],   0x17(%[quant_ptr])              \n\t"
      "gsldrc1    %[ftmp7],   0x10(%[quant_ptr])              \n\t"
      "gsldlc1    %[ftmp8],   0x1f(%[quant_ptr])              \n\t"
      "gsldrc1    %[ftmp8],   0x18(%[quant_ptr])              \n\t"
      "pmulhuh    %[ftmp5],   %[ftmp5],       %[ftmp7]        \n\t"
      "pmulhuh    %[ftmp6],   %[ftmp6],       %[ftmp8]        \n\t"

      "pxor       %[ftmp7],   %[ftmp5],       %[ftmp3]        \n\t"
      "pxor       %[ftmp8],   %[ftmp6],       %[ftmp4]        \n\t"
      "psubh      %[ftmp7],   %[ftmp7],       %[ftmp3]        \n\t"
      "psubh      %[ftmp8],   %[ftmp8],       %[ftmp4]        \n\t"
      "gssdlc1    %[ftmp7],   0x17(%[qcoeff_ptr])             \n\t"
      "gssdrc1    %[ftmp7],   0x10(%[qcoeff_ptr])             \n\t"
      "gssdlc1    %[ftmp8],   0x1f(%[qcoeff_ptr])             \n\t"
      "gssdrc1    %[ftmp8],   0x18(%[qcoeff_ptr])             \n\t"

      "gsldlc1    %[ftmp1],   0x17(%[inv_zig_zag])            \n\t"
      "gsldrc1    %[ftmp1],   0x10(%[inv_zig_zag])            \n\t"
      "gsldlc1    %[ftmp2],   0x1f(%[inv_zig_zag])            \n\t"
      "gsldrc1    %[ftmp2],   0x18(%[inv_zig_zag])            \n\t"
      "pcmpeqh    %[ftmp5],   %[ftmp5],       %[ftmp0]        \n\t"
      "pcmpeqh    %[ftmp6],   %[ftmp6],       %[ftmp0]        \n\t"
      "pxor       %[ftmp5],   %[ftmp5],       %[ones]         \n\t"
      "pxor       %[ftmp6],   %[ftmp6],       %[ones]         \n\t"
      "pand       %[ftmp5],   %[ftmp5],       %[ftmp1]        \n\t"
      "pand       %[ftmp6],   %[ftmp6],       %[ftmp2]        \n\t"
      "pmaxsh     %[ftmp11],  %[ftmp5],       %[ftmp6]        \n\t"

      "gsldlc1    %[ftmp5],   0x17(%[dequant_ptr])            \n\t"
      "gsldrc1    %[ftmp5],   0x10(%[dequant_ptr])            \n\t"
      "gsldlc1    %[ftmp6],   0x1f(%[dequant_ptr])            \n\t"
      "gsldrc1    %[ftmp6],   0x18(%[dequant_ptr])            \n\t"
      "pmullh     %[ftmp5],   %[ftmp5],       %[ftmp7]        \n\t"
      "pmullh     %[ftmp6],   %[ftmp6],       %[ftmp8]        \n\t"
      "gssdlc1    %[ftmp5],   0x17(%[dqcoeff_ptr])            \n\t"
      "gssdrc1    %[ftmp5],   0x10(%[dqcoeff_ptr])            \n\t"
      "gssdlc1    %[ftmp6],   0x1f(%[dqcoeff_ptr])            \n\t"
      "gssdrc1    %[ftmp6],   0x18(%[dqcoeff_ptr])            \n\t"

      "dli        %[tmp0],    0x10                            \n\t"
      "dmtc1      %[tmp0],    %[ftmp9]                        \n\t"

      "pmaxsh     %[ftmp10],  %[ftmp10],       %[ftmp11]      \n\t"
      "psrlw      %[ftmp11],  %[ftmp10],       %[ftmp9]       \n\t"
      "pmaxsh     %[ftmp10],  %[ftmp10],       %[ftmp11]      \n\t"
      "dli        %[tmp0],    0xaa                            \n\t"
      "dmtc1      %[tmp0],    %[ftmp9]                        \n\t"
      "pshufh     %[ftmp11],  %[ftmp10],       %[ftmp9]       \n\t"
      "pmaxsh     %[ftmp10],  %[ftmp10],       %[ftmp11]      \n\t"
      "dli        %[tmp0],    0xffff                          \n\t"
      "dmtc1      %[tmp0],    %[ftmp9]                        \n\t"
      "pand       %[ftmp10],  %[ftmp10],       %[ftmp9]       \n\t"
      "gssdlc1    %[ftmp10],  0x07(%[eob])                    \n\t"
      "gssdrc1    %[ftmp10],  0x00(%[eob])                    \n\t"
      : [ftmp0] "=&f"(ftmp[0]), [ftmp1] "=&f"(ftmp[1]), [ftmp2] "=&f"(ftmp[2]),
        [ftmp3] "=&f"(ftmp[3]), [ftmp4] "=&f"(ftmp[4]), [ftmp5] "=&f"(ftmp[5]),
        [ftmp6] "=&f"(ftmp[6]), [ftmp7] "=&f"(ftmp[7]), [ftmp8] "=&f"(ftmp[8]),
        [ftmp9] "=&f"(ftmp[9]), [ftmp10] "=&f"(ftmp[10]),
        [ftmp11] "=&f"(ftmp[11]), [ftmp12] "=&f"(ftmp[12]),
        [tmp0] "=&r"(tmp[0]), [ones] "=&f"(ones)
      : [coeff_ptr] "r"((mips_reg)coeff_ptr),
        [qcoeff_ptr] "r"((mips_reg)qcoeff_ptr),
        [dequant_ptr] "r"((mips_reg)dequant_ptr),
        [round_ptr] "r"((mips_reg)round_ptr),
        [quant_ptr] "r"((mips_reg)quant_ptr),
        [dqcoeff_ptr] "r"((mips_reg)dqcoeff_ptr),
        [inv_zig_zag] "r"((mips_reg)inv_zig_zag), [eob] "r"((mips_reg)&eob)
      : "memory");

  *d->eob = eob;
}

void vp8_regular_quantize_b_mmi(BLOCK *b, BLOCKD *d) {
  int eob = 0;
  int x, y, z, sz, zbin;
  const int16_t *zbin_boost_ptr = b->zrun_zbin_boost;
  const int16_t *coeff_ptr = b->coeff;
  const int16_t *zbin_ptr = b->zbin;
  const int16_t *round_ptr = b->round;
  const int16_t *quant_ptr = b->quant;
  const int16_t *quant_shift_ptr = b->quant_shift;
  int16_t *qcoeff_ptr = d->qcoeff;
  int16_t *dqcoeff_ptr = d->dqcoeff;
  const int16_t *dequant_ptr = d->dequant;
  const int16_t zbin_oq_value = b->zbin_extra;
  register double ftmp0 asm("$f0");

  //  memset(qcoeff_ptr, 0, 32);
  //  memset(dqcoeff_ptr, 0, 32);
  /* clang-format off */
  __asm__ volatile (
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]        \n\t"
    "gssdlc1    %[ftmp0],   0x07(%[qcoeff_ptr])             \n\t"
    "gssdrc1    %[ftmp0],   0x00(%[qcoeff_ptr])             \n\t"
    "gssdlc1    %[ftmp0],   0x0f(%[qcoeff_ptr])             \n\t"
    "gssdrc1    %[ftmp0],   0x08(%[qcoeff_ptr])             \n\t"
    "gssdlc1    %[ftmp0],   0x17(%[qcoeff_ptr])             \n\t"
    "gssdrc1    %[ftmp0],   0x10(%[qcoeff_ptr])             \n\t"
    "gssdlc1    %[ftmp0],   0x1f(%[qcoeff_ptr])             \n\t"
    "gssdrc1    %[ftmp0],   0x18(%[qcoeff_ptr])             \n\t"

    "gssdlc1    %[ftmp0],   0x07(%[dqcoeff_ptr])            \n\t"
    "gssdrc1    %[ftmp0],   0x00(%[dqcoeff_ptr])            \n\t"
    "gssdlc1    %[ftmp0],   0x0f(%[dqcoeff_ptr])            \n\t"
    "gssdrc1    %[ftmp0],   0x08(%[dqcoeff_ptr])            \n\t"
    "gssdlc1    %[ftmp0],   0x17(%[dqcoeff_ptr])            \n\t"
    "gssdrc1    %[ftmp0],   0x10(%[dqcoeff_ptr])            \n\t"
    "gssdlc1    %[ftmp0],   0x1f(%[dqcoeff_ptr])            \n\t"
    "gssdrc1    %[ftmp0],   0x18(%[dqcoeff_ptr])            \n\t"
    : [ftmp0]"=&f"(ftmp0)
    : [qcoeff_ptr]"r"(qcoeff_ptr), [dqcoeff_ptr]"r"(dqcoeff_ptr)
    : "memory"
  );
  /* clang-format on */

  REGULAR_SELECT_EOB(1, 0);
  REGULAR_SELECT_EOB(2, 1);
  REGULAR_SELECT_EOB(3, 4);
  REGULAR_SELECT_EOB(4, 8);
  REGULAR_SELECT_EOB(5, 5);
  REGULAR_SELECT_EOB(6, 2);
  REGULAR_SELECT_EOB(7, 3);
  REGULAR_SELECT_EOB(8, 6);
  REGULAR_SELECT_EOB(9, 9);
  REGULAR_SELECT_EOB(10, 12);
  REGULAR_SELECT_EOB(11, 13);
  REGULAR_SELECT_EOB(12, 10);
  REGULAR_SELECT_EOB(13, 7);
  REGULAR_SELECT_EOB(14, 11);
  REGULAR_SELECT_EOB(15, 14);
  REGULAR_SELECT_EOB(16, 15);

  *d->eob = (char)eob;
}
