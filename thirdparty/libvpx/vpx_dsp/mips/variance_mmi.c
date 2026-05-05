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
#include "vpx_dsp/variance.h"
#include "vpx_ports/mem.h"
#include "vpx/vpx_integer.h"
#include "vpx_ports/asmdefs_mmi.h"

static const uint8_t bilinear_filters[8][2] = {
  { 128, 0 }, { 112, 16 }, { 96, 32 }, { 80, 48 },
  { 64, 64 }, { 48, 80 },  { 32, 96 }, { 16, 112 },
};

/* Use VARIANCE_SSE_SUM_8_FOR_W64 in vpx_variance64x64,vpx_variance64x32,
   vpx_variance32x64. VARIANCE_SSE_SUM_8 will lead to sum overflow. */
#define VARIANCE_SSE_SUM_8_FOR_W64                                  \
  /* sse */                                                         \
  "pasubub    %[ftmp3],   %[ftmp1],       %[ftmp2]            \n\t" \
  "punpcklbh  %[ftmp4],   %[ftmp3],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp5],   %[ftmp3],       %[ftmp0]            \n\t" \
  "pmaddhw    %[ftmp6],   %[ftmp4],       %[ftmp4]            \n\t" \
  "pmaddhw    %[ftmp7],   %[ftmp5],       %[ftmp5]            \n\t" \
  "paddw      %[ftmp10],  %[ftmp10],      %[ftmp6]            \n\t" \
  "paddw      %[ftmp10],  %[ftmp10],      %[ftmp7]            \n\t" \
                                                                    \
  /* sum */                                                         \
  "punpcklbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp4],   %[ftmp1],       %[ftmp0]            \n\t" \
  "punpcklbh  %[ftmp5],   %[ftmp2],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp6],   %[ftmp2],       %[ftmp0]            \n\t" \
  "punpcklhw  %[ftmp1],   %[ftmp3],       %[ftmp0]            \n\t" \
  "punpckhhw  %[ftmp2],   %[ftmp3],       %[ftmp0]            \n\t" \
  "punpcklhw  %[ftmp7],   %[ftmp5],       %[ftmp0]            \n\t" \
  "punpckhhw  %[ftmp8],   %[ftmp5],       %[ftmp0]            \n\t" \
  "psubw      %[ftmp3],   %[ftmp1],       %[ftmp7]            \n\t" \
  "psubw      %[ftmp5],   %[ftmp2],       %[ftmp8]            \n\t" \
  "punpcklhw  %[ftmp1],   %[ftmp4],       %[ftmp0]            \n\t" \
  "punpckhhw  %[ftmp2],   %[ftmp4],       %[ftmp0]            \n\t" \
  "punpcklhw  %[ftmp7],   %[ftmp6],       %[ftmp0]            \n\t" \
  "punpckhhw  %[ftmp8],   %[ftmp6],       %[ftmp0]            \n\t" \
  "psubw      %[ftmp4],   %[ftmp1],       %[ftmp7]            \n\t" \
  "psubw      %[ftmp6],   %[ftmp2],       %[ftmp8]            \n\t" \
  "paddw      %[ftmp9],   %[ftmp9],       %[ftmp3]            \n\t" \
  "paddw      %[ftmp9],   %[ftmp9],       %[ftmp4]            \n\t" \
  "paddw      %[ftmp9],   %[ftmp9],       %[ftmp5]            \n\t" \
  "paddw      %[ftmp9],   %[ftmp9],       %[ftmp6]            \n\t"

#define VARIANCE_SSE_SUM_4                                          \
  /* sse */                                                         \
  "pasubub    %[ftmp3],   %[ftmp1],       %[ftmp2]            \n\t" \
  "punpcklbh  %[ftmp4],   %[ftmp3],       %[ftmp0]            \n\t" \
  "pmaddhw    %[ftmp5],   %[ftmp4],       %[ftmp4]            \n\t" \
  "paddw      %[ftmp6],   %[ftmp6],       %[ftmp5]            \n\t" \
                                                                    \
  /* sum */                                                         \
  "punpcklbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t" \
  "punpcklbh  %[ftmp4],   %[ftmp2],       %[ftmp0]            \n\t" \
  "paddh      %[ftmp7],   %[ftmp7],       %[ftmp3]            \n\t" \
  "paddh      %[ftmp8],   %[ftmp8],       %[ftmp4]            \n\t"

#define VARIANCE_SSE_SUM_8                                          \
  /* sse */                                                         \
  "pasubub    %[ftmp3],   %[ftmp1],       %[ftmp2]            \n\t" \
  "punpcklbh  %[ftmp4],   %[ftmp3],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp5],   %[ftmp3],       %[ftmp0]            \n\t" \
  "pmaddhw    %[ftmp6],   %[ftmp4],       %[ftmp4]            \n\t" \
  "pmaddhw    %[ftmp7],   %[ftmp5],       %[ftmp5]            \n\t" \
  "paddw      %[ftmp8],   %[ftmp8],       %[ftmp6]            \n\t" \
  "paddw      %[ftmp8],   %[ftmp8],       %[ftmp7]            \n\t" \
                                                                    \
  /* sum */                                                         \
  "punpcklbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp4],   %[ftmp1],       %[ftmp0]            \n\t" \
  "punpcklbh  %[ftmp5],   %[ftmp2],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp6],   %[ftmp2],       %[ftmp0]            \n\t" \
  "paddh      %[ftmp10],  %[ftmp10],      %[ftmp3]            \n\t" \
  "paddh      %[ftmp10],  %[ftmp10],      %[ftmp4]            \n\t" \
  "paddh      %[ftmp12],  %[ftmp12],      %[ftmp5]            \n\t" \
  "paddh      %[ftmp12],  %[ftmp12],      %[ftmp6]            \n\t"

#define VARIANCE_SSE_8                                              \
  "gsldlc1    %[ftmp1],   0x07(%[src_ptr])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src_ptr])                    \n\t" \
  "gsldlc1    %[ftmp2],   0x07(%[ref_ptr])                    \n\t" \
  "gsldrc1    %[ftmp2],   0x00(%[ref_ptr])                    \n\t" \
  "pasubub    %[ftmp3],   %[ftmp1],       %[ftmp2]            \n\t" \
  "punpcklbh  %[ftmp4],   %[ftmp3],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp5],   %[ftmp3],       %[ftmp0]            \n\t" \
  "pmaddhw    %[ftmp6],   %[ftmp4],       %[ftmp4]            \n\t" \
  "pmaddhw    %[ftmp7],   %[ftmp5],       %[ftmp5]            \n\t" \
  "paddw      %[ftmp8],   %[ftmp8],       %[ftmp6]            \n\t" \
  "paddw      %[ftmp8],   %[ftmp8],       %[ftmp7]            \n\t"

#define VARIANCE_SSE_16                                             \
  VARIANCE_SSE_8                                                    \
  "gsldlc1    %[ftmp1],   0x0f(%[src_ptr])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x08(%[src_ptr])                    \n\t" \
  "gsldlc1    %[ftmp2],   0x0f(%[ref_ptr])                    \n\t" \
  "gsldrc1    %[ftmp2],   0x08(%[ref_ptr])                    \n\t" \
  "pasubub    %[ftmp3],   %[ftmp1],       %[ftmp2]            \n\t" \
  "punpcklbh  %[ftmp4],   %[ftmp3],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp5],   %[ftmp3],       %[ftmp0]            \n\t" \
  "pmaddhw    %[ftmp6],   %[ftmp4],       %[ftmp4]            \n\t" \
  "pmaddhw    %[ftmp7],   %[ftmp5],       %[ftmp5]            \n\t" \
  "paddw      %[ftmp8],   %[ftmp8],       %[ftmp6]            \n\t" \
  "paddw      %[ftmp8],   %[ftmp8],       %[ftmp7]            \n\t"

#define VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_4_A                       \
  /* calculate fdata3[0]~fdata3[3], store at ftmp2*/                \
  "gsldlc1    %[ftmp1],   0x07(%[src_ptr])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src_ptr])                    \n\t" \
  "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t" \
  "gsldlc1    %[ftmp1],   0x08(%[src_ptr])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x01(%[src_ptr])                    \n\t" \
  "punpcklbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t" \
  "pmullh     %[ftmp2],   %[ftmp2],       %[filter_x0]        \n\t" \
  "paddh      %[ftmp2],   %[ftmp2],       %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp3],   %[ftmp3],       %[filter_x1]        \n\t" \
  "paddh      %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t" \
  "psrlh      %[ftmp2],   %[ftmp2],       %[ftmp6]            \n\t"

#define VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_4_B                       \
  /* calculate fdata3[0]~fdata3[3], store at ftmp4*/                \
  "gsldlc1    %[ftmp1],   0x07(%[src_ptr])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src_ptr])                    \n\t" \
  "punpcklbh  %[ftmp4],   %[ftmp1],       %[ftmp0]            \n\t" \
  "gsldlc1    %[ftmp1],   0x08(%[src_ptr])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x01(%[src_ptr])                    \n\t" \
  "punpcklbh  %[ftmp5],   %[ftmp1],       %[ftmp0]            \n\t" \
  "pmullh     %[ftmp4],   %[ftmp4],       %[filter_x0]        \n\t" \
  "paddh      %[ftmp4],   %[ftmp4],       %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp5],   %[ftmp5],       %[filter_x1]        \n\t" \
  "paddh      %[ftmp4],   %[ftmp4],       %[ftmp5]            \n\t" \
  "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp6]            \n\t"

#define VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_4_A                      \
  /* calculate: temp2[0] ~ temp2[3] */                              \
  "pmullh     %[ftmp2],   %[ftmp2],       %[filter_y0]        \n\t" \
  "paddh      %[ftmp2],   %[ftmp2],       %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp1],   %[ftmp4],       %[filter_y1]        \n\t" \
  "paddh      %[ftmp2],   %[ftmp2],       %[ftmp1]            \n\t" \
  "psrlh      %[ftmp2],   %[ftmp2],       %[ftmp6]            \n\t" \
                                                                    \
  /* store: temp2[0] ~ temp2[3] */                                  \
  "pand       %[ftmp2],   %[ftmp2],       %[mask]             \n\t" \
  "packushb   %[ftmp2],   %[ftmp2],       %[ftmp0]            \n\t" \
  "gssdrc1    %[ftmp2],   0x00(%[temp2_ptr])                  \n\t"

#define VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_4_B                      \
  /* calculate: temp2[0] ~ temp2[3] */                              \
  "pmullh     %[ftmp4],   %[ftmp4],       %[filter_y0]        \n\t" \
  "paddh      %[ftmp4],   %[ftmp4],       %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp1],   %[ftmp2],       %[filter_y1]        \n\t" \
  "paddh      %[ftmp4],   %[ftmp4],       %[ftmp1]            \n\t" \
  "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp6]            \n\t" \
                                                                    \
  /* store: temp2[0] ~ temp2[3] */                                  \
  "pand       %[ftmp4],   %[ftmp4],       %[mask]             \n\t" \
  "packushb   %[ftmp4],   %[ftmp4],       %[ftmp0]            \n\t" \
  "gssdrc1    %[ftmp4],   0x00(%[temp2_ptr])                  \n\t"

#define VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_8_A                       \
  /* calculate fdata3[0]~fdata3[7], store at ftmp2 and ftmp3*/      \
  "gsldlc1    %[ftmp1],   0x07(%[src_ptr])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src_ptr])                    \n\t" \
  "punpcklbh  %[ftmp2],   %[ftmp1],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp3],   %[ftmp1],       %[ftmp0]            \n\t" \
  "gsldlc1    %[ftmp1],   0x08(%[src_ptr])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x01(%[src_ptr])                    \n\t" \
  "punpcklbh  %[ftmp4],   %[ftmp1],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp5],   %[ftmp1],       %[ftmp0]            \n\t" \
  "pmullh     %[ftmp2],   %[ftmp2],       %[filter_x0]        \n\t" \
  "pmullh     %[ftmp3],   %[ftmp3],       %[filter_x0]        \n\t" \
  "paddh      %[ftmp2],   %[ftmp2],       %[ff_ph_40]         \n\t" \
  "paddh      %[ftmp3],   %[ftmp3],       %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp4],   %[ftmp4],       %[filter_x1]        \n\t" \
  "pmullh     %[ftmp5],   %[ftmp5],       %[filter_x1]        \n\t" \
  "paddh      %[ftmp2],   %[ftmp2],       %[ftmp4]            \n\t" \
  "paddh      %[ftmp3],   %[ftmp3],       %[ftmp5]            \n\t" \
  "psrlh      %[ftmp2],   %[ftmp2],       %[ftmp14]           \n\t" \
  "psrlh      %[ftmp3],   %[ftmp3],       %[ftmp14]           \n\t"

#define VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_8_B                       \
  /* calculate fdata3[0]~fdata3[7], store at ftmp8 and ftmp9*/      \
  "gsldlc1    %[ftmp1],   0x07(%[src_ptr])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x00(%[src_ptr])                    \n\t" \
  "punpcklbh  %[ftmp8],   %[ftmp1],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp9],   %[ftmp1],       %[ftmp0]            \n\t" \
  "gsldlc1    %[ftmp1],   0x08(%[src_ptr])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x01(%[src_ptr])                    \n\t" \
  "punpcklbh  %[ftmp10],  %[ftmp1],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp11],  %[ftmp1],       %[ftmp0]            \n\t" \
  "pmullh     %[ftmp8],   %[ftmp8],       %[filter_x0]        \n\t" \
  "pmullh     %[ftmp9],   %[ftmp9],       %[filter_x0]        \n\t" \
  "paddh      %[ftmp8],   %[ftmp8],       %[ff_ph_40]         \n\t" \
  "paddh      %[ftmp9],   %[ftmp9],       %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp10],  %[ftmp10],      %[filter_x1]        \n\t" \
  "pmullh     %[ftmp11],  %[ftmp11],      %[filter_x1]        \n\t" \
  "paddh      %[ftmp8],   %[ftmp8],       %[ftmp10]           \n\t" \
  "paddh      %[ftmp9],   %[ftmp9],       %[ftmp11]           \n\t" \
  "psrlh      %[ftmp8],   %[ftmp8],       %[ftmp14]           \n\t" \
  "psrlh      %[ftmp9],   %[ftmp9],       %[ftmp14]           \n\t"

#define VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_8_A                      \
  /* calculate: temp2[0] ~ temp2[3] */                              \
  "pmullh     %[ftmp2],   %[ftmp2],       %[filter_y0]        \n\t" \
  "paddh      %[ftmp2],   %[ftmp2],       %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp1],   %[ftmp8],       %[filter_y1]        \n\t" \
  "paddh      %[ftmp2],   %[ftmp2],       %[ftmp1]            \n\t" \
  "psrlh      %[ftmp2],   %[ftmp2],       %[ftmp14]           \n\t" \
                                                                    \
  /* calculate: temp2[4] ~ temp2[7] */                              \
  "pmullh     %[ftmp3],   %[ftmp3],       %[filter_y0]        \n\t" \
  "paddh      %[ftmp3],   %[ftmp3],       %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp1],   %[ftmp9],       %[filter_y1]        \n\t" \
  "paddh      %[ftmp3],   %[ftmp3],       %[ftmp1]            \n\t" \
  "psrlh      %[ftmp3],   %[ftmp3],       %[ftmp14]           \n\t" \
                                                                    \
  /* store: temp2[0] ~ temp2[7] */                                  \
  "pand       %[ftmp2],   %[ftmp2],       %[mask]             \n\t" \
  "pand       %[ftmp3],   %[ftmp3],       %[mask]             \n\t" \
  "packushb   %[ftmp2],   %[ftmp2],       %[ftmp3]            \n\t" \
  "gssdlc1    %[ftmp2],   0x07(%[temp2_ptr])                  \n\t" \
  "gssdrc1    %[ftmp2],   0x00(%[temp2_ptr])                  \n\t"

#define VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_8_B                      \
  /* calculate: temp2[0] ~ temp2[3] */                              \
  "pmullh     %[ftmp8],   %[ftmp8],       %[filter_y0]        \n\t" \
  "paddh      %[ftmp8],   %[ftmp8],       %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp1],   %[ftmp2],       %[filter_y1]        \n\t" \
  "paddh      %[ftmp8],   %[ftmp8],       %[ftmp1]            \n\t" \
  "psrlh      %[ftmp8],   %[ftmp8],       %[ftmp14]           \n\t" \
                                                                    \
  /* calculate: temp2[4] ~ temp2[7] */                              \
  "pmullh     %[ftmp9],   %[ftmp9],       %[filter_y0]        \n\t" \
  "paddh      %[ftmp9],   %[ftmp9],       %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp1],   %[ftmp3],       %[filter_y1]        \n\t" \
  "paddh      %[ftmp9],   %[ftmp9],       %[ftmp1]            \n\t" \
  "psrlh      %[ftmp9],   %[ftmp9],       %[ftmp14]           \n\t" \
                                                                    \
  /* store: temp2[0] ~ temp2[7] */                                  \
  "pand       %[ftmp8],   %[ftmp8],       %[mask]             \n\t" \
  "pand       %[ftmp9],   %[ftmp9],       %[mask]             \n\t" \
  "packushb   %[ftmp8],   %[ftmp8],       %[ftmp9]            \n\t" \
  "gssdlc1    %[ftmp8],   0x07(%[temp2_ptr])                  \n\t" \
  "gssdrc1    %[ftmp8],   0x00(%[temp2_ptr])                  \n\t"

#define VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_16_A                      \
  /* calculate fdata3[0]~fdata3[7], store at ftmp2 and ftmp3*/      \
  VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_8_A                             \
                                                                    \
  /* calculate fdata3[8]~fdata3[15], store at ftmp4 and ftmp5*/     \
  "gsldlc1    %[ftmp1],   0x0f(%[src_ptr])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x08(%[src_ptr])                    \n\t" \
  "punpcklbh  %[ftmp4],   %[ftmp1],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp5],   %[ftmp1],       %[ftmp0]            \n\t" \
  "gsldlc1    %[ftmp1],   0x10(%[src_ptr])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x09(%[src_ptr])                    \n\t" \
  "punpcklbh  %[ftmp6],   %[ftmp1],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp7],   %[ftmp1],       %[ftmp0]            \n\t" \
  "pmullh     %[ftmp4],   %[ftmp4],       %[filter_x0]        \n\t" \
  "pmullh     %[ftmp5],   %[ftmp5],       %[filter_x0]        \n\t" \
  "paddh      %[ftmp4],   %[ftmp4],       %[ff_ph_40]         \n\t" \
  "paddh      %[ftmp5],   %[ftmp5],       %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp6],   %[ftmp6],       %[filter_x1]        \n\t" \
  "pmullh     %[ftmp7],   %[ftmp7],       %[filter_x1]        \n\t" \
  "paddh      %[ftmp4],   %[ftmp4],       %[ftmp6]            \n\t" \
  "paddh      %[ftmp5],   %[ftmp5],       %[ftmp7]            \n\t" \
  "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp14]           \n\t" \
  "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp14]           \n\t"

#define VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_16_B                      \
  /* calculate fdata3[0]~fdata3[7], store at ftmp8 and ftmp9*/      \
  VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_8_B                             \
                                                                    \
  /* calculate fdata3[8]~fdata3[15], store at ftmp10 and ftmp11*/   \
  "gsldlc1    %[ftmp1],   0x0f(%[src_ptr])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x08(%[src_ptr])                    \n\t" \
  "punpcklbh  %[ftmp10],  %[ftmp1],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp11],  %[ftmp1],       %[ftmp0]            \n\t" \
  "gsldlc1    %[ftmp1],   0x10(%[src_ptr])                    \n\t" \
  "gsldrc1    %[ftmp1],   0x09(%[src_ptr])                    \n\t" \
  "punpcklbh  %[ftmp12],  %[ftmp1],       %[ftmp0]            \n\t" \
  "punpckhbh  %[ftmp13],  %[ftmp1],       %[ftmp0]            \n\t" \
  "pmullh     %[ftmp10],  %[ftmp10],      %[filter_x0]        \n\t" \
  "pmullh     %[ftmp11],  %[ftmp11],      %[filter_x0]        \n\t" \
  "paddh      %[ftmp10],  %[ftmp10],      %[ff_ph_40]         \n\t" \
  "paddh      %[ftmp11],  %[ftmp11],      %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp12],  %[ftmp12],      %[filter_x1]        \n\t" \
  "pmullh     %[ftmp13],  %[ftmp13],      %[filter_x1]        \n\t" \
  "paddh      %[ftmp10],  %[ftmp10],      %[ftmp12]           \n\t" \
  "paddh      %[ftmp11],  %[ftmp11],      %[ftmp13]           \n\t" \
  "psrlh      %[ftmp10],  %[ftmp10],      %[ftmp14]           \n\t" \
  "psrlh      %[ftmp11],  %[ftmp11],      %[ftmp14]           \n\t"

#define VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_16_A                     \
  VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_8_A                            \
                                                                    \
  /* calculate: temp2[8] ~ temp2[11] */                             \
  "pmullh     %[ftmp4],   %[ftmp4],       %[filter_y0]        \n\t" \
  "paddh      %[ftmp4],   %[ftmp4],       %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp1],   %[ftmp10],      %[filter_y1]        \n\t" \
  "paddh      %[ftmp4],   %[ftmp4],       %[ftmp1]            \n\t" \
  "psrlh      %[ftmp4],   %[ftmp4],       %[ftmp14]           \n\t" \
                                                                    \
  /* calculate: temp2[12] ~ temp2[15] */                            \
  "pmullh     %[ftmp5],   %[ftmp5],       %[filter_y0]        \n\t" \
  "paddh      %[ftmp5],   %[ftmp5],       %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp1],   %[ftmp11],       %[filter_y1]       \n\t" \
  "paddh      %[ftmp5],   %[ftmp5],       %[ftmp1]            \n\t" \
  "psrlh      %[ftmp5],   %[ftmp5],       %[ftmp14]           \n\t" \
                                                                    \
  /* store: temp2[8] ~ temp2[15] */                                 \
  "pand       %[ftmp4],   %[ftmp4],       %[mask]             \n\t" \
  "pand       %[ftmp5],   %[ftmp5],       %[mask]             \n\t" \
  "packushb   %[ftmp4],   %[ftmp4],       %[ftmp5]            \n\t" \
  "gssdlc1    %[ftmp4],   0x0f(%[temp2_ptr])                  \n\t" \
  "gssdrc1    %[ftmp4],   0x08(%[temp2_ptr])                  \n\t"

#define VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_16_B                     \
  VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_8_B                            \
                                                                    \
  /* calculate: temp2[8] ~ temp2[11] */                             \
  "pmullh     %[ftmp10],  %[ftmp10],      %[filter_y0]        \n\t" \
  "paddh      %[ftmp10],  %[ftmp10],      %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp1],   %[ftmp4],       %[filter_y1]        \n\t" \
  "paddh      %[ftmp10],  %[ftmp10],      %[ftmp1]            \n\t" \
  "psrlh      %[ftmp10],  %[ftmp10],      %[ftmp14]           \n\t" \
                                                                    \
  /* calculate: temp2[12] ~ temp2[15] */                            \
  "pmullh     %[ftmp11],  %[ftmp11],      %[filter_y0]        \n\t" \
  "paddh      %[ftmp11],  %[ftmp11],      %[ff_ph_40]         \n\t" \
  "pmullh     %[ftmp1],   %[ftmp5],       %[filter_y1]        \n\t" \
  "paddh      %[ftmp11],  %[ftmp11],      %[ftmp1]            \n\t" \
  "psrlh      %[ftmp11],  %[ftmp11],      %[ftmp14]           \n\t" \
                                                                    \
  /* store: temp2[8] ~ temp2[15] */                                 \
  "pand       %[ftmp10],  %[ftmp10],      %[mask]             \n\t" \
  "pand       %[ftmp11],  %[ftmp11],      %[mask]             \n\t" \
  "packushb   %[ftmp10],  %[ftmp10],      %[ftmp11]           \n\t" \
  "gssdlc1    %[ftmp10],  0x0f(%[temp2_ptr])                  \n\t" \
  "gssdrc1    %[ftmp10],  0x08(%[temp2_ptr])                  \n\t"

// Applies a 1-D 2-tap bilinear filter to the source block in either horizontal
// or vertical direction to produce the filtered output block. Used to implement
// the first-pass of 2-D separable filter.
//
// Produces int16_t output to retain precision for the next pass. Two filter
// taps should sum to FILTER_WEIGHT. pixel_step defines whether the filter is
// applied horizontally (pixel_step = 1) or vertically (pixel_step = stride).
// It defines the offset required to move from one input to the next.
static void var_filter_block2d_bil_first_pass(
    const uint8_t *src_ptr, uint16_t *ref_ptr, unsigned int src_pixels_per_line,
    int pixel_step, unsigned int output_height, unsigned int output_width,
    const uint8_t *filter) {
  unsigned int i, j;

  for (i = 0; i < output_height; ++i) {
    for (j = 0; j < output_width; ++j) {
      ref_ptr[j] = ROUND_POWER_OF_TWO(
          (int)src_ptr[0] * filter[0] + (int)src_ptr[pixel_step] * filter[1],
          FILTER_BITS);

      ++src_ptr;
    }

    src_ptr += src_pixels_per_line - output_width;
    ref_ptr += output_width;
  }
}

// Applies a 1-D 2-tap bilinear filter to the source block in either horizontal
// or vertical direction to produce the filtered output block. Used to implement
// the second-pass of 2-D separable filter.
//
// Requires 16-bit input as produced by filter_block2d_bil_first_pass. Two
// filter taps should sum to FILTER_WEIGHT. pixel_step defines whether the
// filter is applied horizontally (pixel_step = 1) or vertically
// (pixel_step = stride). It defines the offset required to move from one input
// to the next. Output is 8-bit.
static void var_filter_block2d_bil_second_pass(
    const uint16_t *src_ptr, uint8_t *ref_ptr, unsigned int src_pixels_per_line,
    unsigned int pixel_step, unsigned int output_height,
    unsigned int output_width, const uint8_t *filter) {
  unsigned int i, j;

  for (i = 0; i < output_height; ++i) {
    for (j = 0; j < output_width; ++j) {
      ref_ptr[j] = ROUND_POWER_OF_TWO(
          (int)src_ptr[0] * filter[0] + (int)src_ptr[pixel_step] * filter[1],
          FILTER_BITS);
      ++src_ptr;
    }

    src_ptr += src_pixels_per_line - output_width;
    ref_ptr += output_width;
  }
}

static inline uint32_t vpx_variance64x(const uint8_t *src_ptr, int src_stride,
                                       const uint8_t *ref_ptr, int ref_stride,
                                       uint32_t *sse, int high) {
  int sum;
  double ftmp[12];
  uint32_t tmp[3];

  *sse = 0;

  /* clang-format off */
  __asm__ volatile (
    "li         %[tmp0],    0x20                                \n\t"
    "mtc1       %[tmp0],    %[ftmp11]                           \n\t"
    MMI_L(%[tmp0], %[high], 0x00)
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
    "pxor       %[ftmp9],   %[ftmp9],       %[ftmp9]            \n\t"
    "pxor       %[ftmp10],  %[ftmp10],      %[ftmp10]           \n\t"
    "1:                                                         \n\t"
    "gsldlc1    %[ftmp1],   0x07(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x00(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x07(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x00(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8_FOR_W64

    "gsldlc1    %[ftmp1],   0x0f(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x08(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x0f(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x08(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8_FOR_W64

    "gsldlc1    %[ftmp1],   0x17(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x10(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x17(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x10(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8_FOR_W64

    "gsldlc1    %[ftmp1],   0x1f(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x18(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x1f(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x18(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8_FOR_W64

    "gsldlc1    %[ftmp1],   0x27(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x20(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x27(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x20(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8_FOR_W64

    "gsldlc1    %[ftmp1],   0x2f(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x28(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x2f(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x28(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8_FOR_W64

    "gsldlc1    %[ftmp1],   0x37(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x30(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x37(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x30(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8_FOR_W64

    "gsldlc1    %[ftmp1],   0x3f(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x38(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x3f(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x38(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8_FOR_W64

    "addiu      %[tmp0],    %[tmp0],        -0x01               \n\t"
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    MMI_ADDU(%[ref_ptr], %[ref_ptr], %[ref_stride])
    "bnez       %[tmp0],    1b                                  \n\t"

    "mfc1       %[tmp1],    %[ftmp9]                            \n\t"
    "mfhc1      %[tmp2],    %[ftmp9]                            \n\t"
    "addu       %[sum],     %[tmp1],        %[tmp2]             \n\t"
    "ssrld      %[ftmp1],   %[ftmp10],      %[ftmp11]           \n\t"
    "paddw      %[ftmp1],   %[ftmp1],       %[ftmp10]           \n\t"
    "swc1       %[ftmp1],   0x00(%[sse])                        \n\t"
    : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
      [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
      [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
      [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
      [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
      [ftmp10]"=&f"(ftmp[10]),          [ftmp11]"=&f"(ftmp[11]),
      [tmp0]"=&r"(tmp[0]),              [tmp1]"=&r"(tmp[1]),
      [tmp2]"=&r"(tmp[2]),
      [src_ptr]"+&r"(src_ptr),          [ref_ptr]"+&r"(ref_ptr),
      [sum]"=&r"(sum)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride),
      [high]"r"(&high), [sse]"r"(sse)
    : "memory"
  );
  /* clang-format on */

  return *sse - (((int64_t)sum * sum) / (64 * high));
}

#define VPX_VARIANCE64XN(n)                                                   \
  uint32_t vpx_variance64x##n##_mmi(const uint8_t *src_ptr, int src_stride,   \
                                    const uint8_t *ref_ptr, int ref_stride,   \
                                    uint32_t *sse) {                          \
    return vpx_variance64x(src_ptr, src_stride, ref_ptr, ref_stride, sse, n); \
  }

VPX_VARIANCE64XN(64)
VPX_VARIANCE64XN(32)

uint32_t vpx_variance32x64_mmi(const uint8_t *src_ptr, int src_stride,
                               const uint8_t *ref_ptr, int ref_stride,
                               uint32_t *sse) {
  int sum;
  double ftmp[12];
  uint32_t tmp[3];

  *sse = 0;

  /* clang-format off */
  __asm__ volatile (
    "li         %[tmp0],    0x20                                \n\t"
    "mtc1       %[tmp0],    %[ftmp11]                           \n\t"
    "li         %[tmp0],    0x40                                \n\t"
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
    "pxor       %[ftmp9],   %[ftmp9],       %[ftmp9]            \n\t"
    "pxor       %[ftmp10],  %[ftmp10],      %[ftmp10]           \n\t"
    "1:                                                         \n\t"
    "gsldlc1    %[ftmp1],   0x07(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x00(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x07(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x00(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8_FOR_W64

    "gsldlc1    %[ftmp1],   0x0f(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x08(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x0f(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x08(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8_FOR_W64

    "gsldlc1    %[ftmp1],   0x17(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x10(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x17(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x10(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8_FOR_W64

    "gsldlc1    %[ftmp1],   0x1f(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x18(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x1f(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x18(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8_FOR_W64

    "addiu      %[tmp0],    %[tmp0],        -0x01               \n\t"
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    MMI_ADDU(%[ref_ptr], %[ref_ptr], %[ref_stride])
    "bnez       %[tmp0],    1b                                  \n\t"

    "mfc1       %[tmp1],    %[ftmp9]                            \n\t"
    "mfhc1      %[tmp2],    %[ftmp9]                            \n\t"
    "addu       %[sum],     %[tmp1],        %[tmp2]             \n\t"
    "ssrld      %[ftmp1],   %[ftmp10],      %[ftmp11]           \n\t"
    "paddw      %[ftmp1],   %[ftmp1],       %[ftmp10]           \n\t"
    "swc1       %[ftmp1],   0x00(%[sse])                        \n\t"
    : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
      [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
      [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
      [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
      [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
      [ftmp10]"=&f"(ftmp[10]),          [ftmp11]"=&f"(ftmp[11]),
      [tmp0]"=&r"(tmp[0]),              [tmp1]"=&r"(tmp[1]),
      [tmp2]"=&r"(tmp[2]),
      [src_ptr]"+&r"(src_ptr),          [ref_ptr]"+&r"(ref_ptr),
      [sum]"=&r"(sum)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride),
      [sse]"r"(sse)
    : "memory"
  );
  /* clang-format on */

  return *sse - (((int64_t)sum * sum) / 2048);
}

static inline uint32_t vpx_variance32x(const uint8_t *src_ptr, int src_stride,
                                       const uint8_t *ref_ptr, int ref_stride,
                                       uint32_t *sse, int high) {
  int sum;
  double ftmp[13];
  uint32_t tmp[3];

  *sse = 0;

  /* clang-format off */
  __asm__ volatile (
    "li         %[tmp0],    0x20                                \n\t"
    "mtc1       %[tmp0],    %[ftmp11]                           \n\t"
    MMI_L(%[tmp0], %[high], 0x00)
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
    "pxor       %[ftmp8],   %[ftmp8],       %[ftmp8]            \n\t"
    "pxor       %[ftmp10],  %[ftmp10],      %[ftmp10]           \n\t"
    "pxor       %[ftmp12],  %[ftmp12],      %[ftmp12]           \n\t"
    "1:                                                         \n\t"
    "gsldlc1    %[ftmp1],   0x07(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x00(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x07(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x00(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8
    "gsldlc1    %[ftmp1],   0x0f(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x08(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x0f(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x08(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8
    "gsldlc1    %[ftmp1],   0x17(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x10(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x17(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x10(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8
    "gsldlc1    %[ftmp1],   0x1f(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x18(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x1f(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x18(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8

    "addiu      %[tmp0],    %[tmp0],        -0x01               \n\t"
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    MMI_ADDU(%[ref_ptr], %[ref_ptr], %[ref_stride])
    "bnez       %[tmp0],    1b                                  \n\t"

    "ssrld      %[ftmp9],   %[ftmp8],       %[ftmp11]           \n\t"
    "paddw      %[ftmp9],   %[ftmp9],       %[ftmp8]            \n\t"
    "swc1       %[ftmp9],   0x00(%[sse])                        \n\t"

    "punpcklhw  %[ftmp3],   %[ftmp10],      %[ftmp0]            \n\t"
    "punpckhhw  %[ftmp4],   %[ftmp10],      %[ftmp0]            \n\t"
    "punpcklhw  %[ftmp5],   %[ftmp12],      %[ftmp0]            \n\t"
    "punpckhhw  %[ftmp6],   %[ftmp12],      %[ftmp0]            \n\t"
    "paddw      %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
    "psubw      %[ftmp3],   %[ftmp3],       %[ftmp5]            \n\t"
    "psubw      %[ftmp3],   %[ftmp3],       %[ftmp6]            \n\t"
    "ssrld      %[ftmp0],   %[ftmp3],       %[ftmp11]           \n\t"
    "paddw      %[ftmp0],   %[ftmp0],       %[ftmp3]            \n\t"
    "swc1       %[ftmp0],   0x00(%[sum])                        \n\t"

    : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
      [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
      [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
      [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
      [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
      [ftmp10]"=&f"(ftmp[10]),          [ftmp11]"=&f"(ftmp[11]),
      [ftmp12]"=&f"(ftmp[12]),          [tmp0]"=&r"(tmp[0]),
      [src_ptr]"+&r"(src_ptr),          [ref_ptr]"+&r"(ref_ptr)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride),
      [high]"r"(&high), [sse]"r"(sse), [sum]"r"(&sum)
    : "memory"
  );
  /* clang-format on */

  return *sse - (((int64_t)sum * sum) / (32 * high));
}

#define VPX_VARIANCE32XN(n)                                                   \
  uint32_t vpx_variance32x##n##_mmi(const uint8_t *src_ptr, int src_stride,   \
                                    const uint8_t *ref_ptr, int ref_stride,   \
                                    uint32_t *sse) {                          \
    return vpx_variance32x(src_ptr, src_stride, ref_ptr, ref_stride, sse, n); \
  }

VPX_VARIANCE32XN(32)
VPX_VARIANCE32XN(16)

static inline uint32_t vpx_variance16x(const uint8_t *src_ptr, int src_stride,
                                       const uint8_t *ref_ptr, int ref_stride,
                                       uint32_t *sse, int high) {
  int sum;
  double ftmp[13];
  uint32_t tmp[3];

  *sse = 0;

  /* clang-format off */
  __asm__ volatile (
    "li         %[tmp0],    0x20                                \n\t"
    "mtc1       %[tmp0],    %[ftmp11]                           \n\t"
    MMI_L(%[tmp0], %[high], 0x00)
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
    "pxor       %[ftmp8],   %[ftmp8],       %[ftmp8]            \n\t"
    "pxor       %[ftmp10],  %[ftmp10],      %[ftmp10]           \n\t"
    "pxor       %[ftmp12],  %[ftmp12],      %[ftmp12]           \n\t"
    "1:                                                         \n\t"
    "gsldlc1    %[ftmp1],   0x07(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x00(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x07(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x00(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8
    "gsldlc1    %[ftmp1],   0x0f(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x08(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x0f(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x08(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8

    "addiu      %[tmp0],    %[tmp0],        -0x01               \n\t"
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    MMI_ADDU(%[ref_ptr], %[ref_ptr], %[ref_stride])
    "bnez       %[tmp0],    1b                                  \n\t"

    "ssrld      %[ftmp9],   %[ftmp8],       %[ftmp11]           \n\t"
    "paddw      %[ftmp9],   %[ftmp9],       %[ftmp8]            \n\t"
    "swc1       %[ftmp9],   0x00(%[sse])                        \n\t"

    "punpcklhw  %[ftmp3],   %[ftmp10],      %[ftmp0]            \n\t"
    "punpckhhw  %[ftmp4],   %[ftmp10],      %[ftmp0]            \n\t"
    "punpcklhw  %[ftmp5],   %[ftmp12],      %[ftmp0]            \n\t"
    "punpckhhw  %[ftmp6],   %[ftmp12],      %[ftmp0]            \n\t"
    "paddw      %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
    "psubw      %[ftmp3],   %[ftmp3],       %[ftmp5]            \n\t"
    "psubw      %[ftmp3],   %[ftmp3],       %[ftmp6]            \n\t"
    "ssrld      %[ftmp0],   %[ftmp3],       %[ftmp11]           \n\t"
    "paddw      %[ftmp0],   %[ftmp0],       %[ftmp3]            \n\t"
    "swc1       %[ftmp0],   0x00(%[sum])                        \n\t"

    : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
      [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
      [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
      [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
      [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
      [ftmp10]"=&f"(ftmp[10]),          [ftmp11]"=&f"(ftmp[11]),
      [ftmp12]"=&f"(ftmp[12]),          [tmp0]"=&r"(tmp[0]),
      [src_ptr]"+&r"(src_ptr),          [ref_ptr]"+&r"(ref_ptr)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride),
      [high]"r"(&high), [sse]"r"(sse), [sum]"r"(&sum)
    : "memory"
  );
  /* clang-format on */

  return *sse - (((int64_t)sum * sum) / (16 * high));
}

#define VPX_VARIANCE16XN(n)                                                   \
  uint32_t vpx_variance16x##n##_mmi(const uint8_t *src_ptr, int src_stride,   \
                                    const uint8_t *ref_ptr, int ref_stride,   \
                                    uint32_t *sse) {                          \
    return vpx_variance16x(src_ptr, src_stride, ref_ptr, ref_stride, sse, n); \
  }

VPX_VARIANCE16XN(32)
VPX_VARIANCE16XN(16)
VPX_VARIANCE16XN(8)

static inline uint32_t vpx_variance8x(const uint8_t *src_ptr, int src_stride,
                                      const uint8_t *ref_ptr, int ref_stride,
                                      uint32_t *sse, int high) {
  int sum;
  double ftmp[13];
  uint32_t tmp[3];

  *sse = 0;

  /* clang-format off */
  __asm__ volatile (
    "li         %[tmp0],    0x20                                \n\t"
    "mtc1       %[tmp0],    %[ftmp11]                           \n\t"
    MMI_L(%[tmp0], %[high], 0x00)
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
    "pxor       %[ftmp8],   %[ftmp8],       %[ftmp8]            \n\t"
    "pxor       %[ftmp10],  %[ftmp10],      %[ftmp10]           \n\t"
    "pxor       %[ftmp12],  %[ftmp12],      %[ftmp12]           \n\t"
    "1:                                                         \n\t"
    "gsldlc1    %[ftmp1],   0x07(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x00(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x07(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x00(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_8

    "addiu      %[tmp0],    %[tmp0],        -0x01               \n\t"
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    MMI_ADDU(%[ref_ptr], %[ref_ptr], %[ref_stride])
    "bnez       %[tmp0],    1b                                  \n\t"

    "ssrld      %[ftmp9],   %[ftmp8],       %[ftmp11]           \n\t"
    "paddw      %[ftmp9],   %[ftmp9],       %[ftmp8]            \n\t"
    "swc1       %[ftmp9],   0x00(%[sse])                        \n\t"

    "punpcklhw  %[ftmp3],   %[ftmp10],      %[ftmp0]            \n\t"
    "punpckhhw  %[ftmp4],   %[ftmp10],      %[ftmp0]            \n\t"
    "punpcklhw  %[ftmp5],   %[ftmp12],      %[ftmp0]            \n\t"
    "punpckhhw  %[ftmp6],   %[ftmp12],      %[ftmp0]            \n\t"
    "paddw      %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
    "psubw      %[ftmp3],   %[ftmp3],       %[ftmp5]            \n\t"
    "psubw      %[ftmp3],   %[ftmp3],       %[ftmp6]            \n\t"
    "ssrld      %[ftmp0],   %[ftmp3],       %[ftmp11]           \n\t"
    "paddw      %[ftmp0],   %[ftmp0],       %[ftmp3]            \n\t"
    "swc1       %[ftmp0],   0x00(%[sum])                        \n\t"

    : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
      [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
      [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
      [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
      [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
      [ftmp10]"=&f"(ftmp[10]),          [ftmp11]"=&f"(ftmp[11]),
      [ftmp12]"=&f"(ftmp[12]),          [tmp0]"=&r"(tmp[0]),
      [src_ptr]"+&r"(src_ptr),          [ref_ptr]"+&r"(ref_ptr)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride),
      [high]"r"(&high), [sse]"r"(sse), [sum]"r"(&sum)
    : "memory"
  );
  /* clang-format on */

  return *sse - (((int64_t)sum * sum) / (8 * high));
}

#define VPX_VARIANCE8XN(n)                                                   \
  uint32_t vpx_variance8x##n##_mmi(const uint8_t *src_ptr, int src_stride,   \
                                   const uint8_t *ref_ptr, int ref_stride,   \
                                   uint32_t *sse) {                          \
    return vpx_variance8x(src_ptr, src_stride, ref_ptr, ref_stride, sse, n); \
  }

VPX_VARIANCE8XN(16)
VPX_VARIANCE8XN(8)
VPX_VARIANCE8XN(4)

static inline uint32_t vpx_variance4x(const uint8_t *src_ptr, int src_stride,
                                      const uint8_t *ref_ptr, int ref_stride,
                                      uint32_t *sse, int high) {
  int sum;
  double ftmp[12];
  uint32_t tmp[3];

  *sse = 0;

  /* clang-format off */
  __asm__ volatile (
    "li         %[tmp0],    0x20                                \n\t"
    "mtc1       %[tmp0],    %[ftmp10]                           \n\t"
    MMI_L(%[tmp0], %[high], 0x00)
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
    "pxor       %[ftmp6],   %[ftmp6],       %[ftmp6]            \n\t"
    "pxor       %[ftmp7],   %[ftmp7],       %[ftmp7]            \n\t"
    "pxor       %[ftmp8],   %[ftmp8],       %[ftmp8]            \n\t"
    "1:                                                         \n\t"
    "gsldlc1    %[ftmp1],   0x07(%[src_ptr])                    \n\t"
    "gsldrc1    %[ftmp1],   0x00(%[src_ptr])                    \n\t"
    "gsldlc1    %[ftmp2],   0x07(%[ref_ptr])                    \n\t"
    "gsldrc1    %[ftmp2],   0x00(%[ref_ptr])                    \n\t"
    VARIANCE_SSE_SUM_4

    "addiu      %[tmp0],    %[tmp0],        -0x01               \n\t"
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    MMI_ADDU(%[ref_ptr], %[ref_ptr], %[ref_stride])
    "bnez       %[tmp0],    1b                                  \n\t"

    "ssrld      %[ftmp9],   %[ftmp6],       %[ftmp10]           \n\t"
    "paddw      %[ftmp9],   %[ftmp9],       %[ftmp6]            \n\t"
    "swc1       %[ftmp9],   0x00(%[sse])                        \n\t"

    "punpcklhw  %[ftmp3],   %[ftmp7],       %[ftmp0]            \n\t"
    "punpckhhw  %[ftmp4],   %[ftmp7],       %[ftmp0]            \n\t"
    "punpcklhw  %[ftmp5],   %[ftmp8],       %[ftmp0]            \n\t"
    "punpckhhw  %[ftmp6],   %[ftmp8],       %[ftmp0]            \n\t"
    "paddw      %[ftmp3],   %[ftmp3],       %[ftmp4]            \n\t"
    "psubw      %[ftmp3],   %[ftmp3],       %[ftmp5]            \n\t"
    "psubw      %[ftmp3],   %[ftmp3],       %[ftmp6]            \n\t"
    "ssrld      %[ftmp0],   %[ftmp3],       %[ftmp10]           \n\t"
    "paddw      %[ftmp0],   %[ftmp0],       %[ftmp3]            \n\t"
    "swc1       %[ftmp0],   0x00(%[sum])                        \n\t"
    : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
      [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
      [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
      [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
      [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
      [ftmp10]"=&f"(ftmp[10]),
      [tmp0]"=&r"(tmp[0]),
      [src_ptr]"+&r"(src_ptr),          [ref_ptr]"+&r"(ref_ptr)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride),
      [high]"r"(&high), [sse]"r"(sse), [sum]"r"(&sum)
    : "memory"
  );
  /* clang-format on */

  return *sse - (((int64_t)sum * sum) / (4 * high));
}

#define VPX_VARIANCE4XN(n)                                                   \
  uint32_t vpx_variance4x##n##_mmi(const uint8_t *src_ptr, int src_stride,   \
                                   const uint8_t *ref_ptr, int ref_stride,   \
                                   uint32_t *sse) {                          \
    return vpx_variance4x(src_ptr, src_stride, ref_ptr, ref_stride, sse, n); \
  }

VPX_VARIANCE4XN(8)
VPX_VARIANCE4XN(4)

static inline uint32_t vpx_mse16x(const uint8_t *src_ptr, int src_stride,
                                  const uint8_t *ref_ptr, int ref_stride,
                                  uint32_t *sse, uint64_t high) {
  double ftmp[12];
  uint32_t tmp[1];

  *sse = 0;

  /* clang-format off */
  __asm__ volatile (
    "li         %[tmp0],    0x20                                \n\t"
    "mtc1       %[tmp0],    %[ftmp11]                           \n\t"
    MMI_L(%[tmp0], %[high], 0x00)
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
    "pxor       %[ftmp8],   %[ftmp8],       %[ftmp8]            \n\t"

    "1:                                                         \n\t"
    VARIANCE_SSE_16

    "addiu      %[tmp0],    %[tmp0],        -0x01               \n\t"
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    MMI_ADDU(%[ref_ptr], %[ref_ptr], %[ref_stride])
    "bnez       %[tmp0],    1b                                  \n\t"

    "ssrld      %[ftmp9],   %[ftmp8],       %[ftmp11]           \n\t"
    "paddw      %[ftmp9],   %[ftmp9],       %[ftmp8]            \n\t"
    "swc1       %[ftmp9],   0x00(%[sse])                        \n\t"
    : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
      [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
      [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
      [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
      [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
      [ftmp10]"=&f"(ftmp[10]),          [ftmp11]"=&f"(ftmp[11]),
      [tmp0]"=&r"(tmp[0]),
      [src_ptr]"+&r"(src_ptr),          [ref_ptr]"+&r"(ref_ptr)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride),
      [high]"r"(&high), [sse]"r"(sse)
    : "memory"
  );
  /* clang-format on */

  return *sse;
}

#define vpx_mse16xN(n)                                                   \
  uint32_t vpx_mse16x##n##_mmi(const uint8_t *src_ptr, int src_stride,   \
                               const uint8_t *ref_ptr, int ref_stride,   \
                               uint32_t *sse) {                          \
    return vpx_mse16x(src_ptr, src_stride, ref_ptr, ref_stride, sse, n); \
  }

vpx_mse16xN(16);
vpx_mse16xN(8);

static inline uint32_t vpx_mse8x(const uint8_t *src_ptr, int src_stride,
                                 const uint8_t *ref_ptr, int ref_stride,
                                 uint32_t *sse, uint64_t high) {
  double ftmp[12];
  uint32_t tmp[1];

  *sse = 0;

  /* clang-format off */
  __asm__ volatile (
    "li         %[tmp0],    0x20                                \n\t"
    "mtc1       %[tmp0],    %[ftmp11]                           \n\t"
    MMI_L(%[tmp0], %[high], 0x00)
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
    "pxor       %[ftmp8],   %[ftmp8],       %[ftmp8]            \n\t"

    "1:                                                         \n\t"
    VARIANCE_SSE_8

    "addiu      %[tmp0],    %[tmp0],        -0x01               \n\t"
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    MMI_ADDU(%[ref_ptr], %[ref_ptr], %[ref_stride])
    "bnez       %[tmp0],    1b                                  \n\t"

    "ssrld      %[ftmp9],   %[ftmp8],       %[ftmp11]           \n\t"
    "paddw      %[ftmp9],   %[ftmp9],       %[ftmp8]            \n\t"
    "swc1       %[ftmp9],   0x00(%[sse])                        \n\t"
    : [ftmp0]"=&f"(ftmp[0]),            [ftmp1]"=&f"(ftmp[1]),
      [ftmp2]"=&f"(ftmp[2]),            [ftmp3]"=&f"(ftmp[3]),
      [ftmp4]"=&f"(ftmp[4]),            [ftmp5]"=&f"(ftmp[5]),
      [ftmp6]"=&f"(ftmp[6]),            [ftmp7]"=&f"(ftmp[7]),
      [ftmp8]"=&f"(ftmp[8]),            [ftmp9]"=&f"(ftmp[9]),
      [ftmp10]"=&f"(ftmp[10]),          [ftmp11]"=&f"(ftmp[11]),
      [tmp0]"=&r"(tmp[0]),
      [src_ptr]"+&r"(src_ptr),          [ref_ptr]"+&r"(ref_ptr)
    : [src_stride]"r"((mips_reg)src_stride),
      [ref_stride]"r"((mips_reg)ref_stride),
      [high]"r"(&high), [sse]"r"(sse)
    : "memory"
  );
  /* clang-format on */

  return *sse;
}

#define vpx_mse8xN(n)                                                   \
  uint32_t vpx_mse8x##n##_mmi(const uint8_t *src_ptr, int src_stride,   \
                              const uint8_t *ref_ptr, int ref_stride,   \
                              uint32_t *sse) {                          \
    return vpx_mse8x(src_ptr, src_stride, ref_ptr, ref_stride, sse, n); \
  }

vpx_mse8xN(16);
vpx_mse8xN(8);

#define SUBPIX_VAR(W, H)                                                       \
  uint32_t vpx_sub_pixel_variance##W##x##H##_mmi(                              \
      const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset,      \
      const uint8_t *ref_ptr, int ref_stride, uint32_t *sse) {                 \
    uint16_t fdata3[((H) + 1) * (W)];                                          \
    uint8_t temp2[(H) * (W)];                                                  \
                                                                               \
    var_filter_block2d_bil_first_pass(src_ptr, fdata3, src_stride, 1, (H) + 1, \
                                      W, bilinear_filters[x_offset]);          \
    var_filter_block2d_bil_second_pass(fdata3, temp2, W, W, H, W,              \
                                       bilinear_filters[y_offset]);            \
                                                                               \
    return vpx_variance##W##x##H##_mmi(temp2, W, ref_ptr, ref_stride, sse);    \
  }

SUBPIX_VAR(64, 64)
SUBPIX_VAR(64, 32)
SUBPIX_VAR(32, 64)
SUBPIX_VAR(32, 32)
SUBPIX_VAR(32, 16)
SUBPIX_VAR(16, 32)

static inline void var_filter_block2d_bil_16x(const uint8_t *src_ptr,
                                              int src_stride, int x_offset,
                                              int y_offset, uint8_t *temp2,
                                              int counter) {
  uint8_t *temp2_ptr = temp2;
  mips_reg l_counter = counter;
  double ftmp[15];
  double ff_ph_40, mask;
  double filter_x0, filter_x1, filter_y0, filter_y1;
  mips_reg tmp[2];
  uint64_t x0, x1, y0, y1, all;

  const uint8_t *filter_x = bilinear_filters[x_offset];
  const uint8_t *filter_y = bilinear_filters[y_offset];
  x0 = (uint64_t)filter_x[0];
  x1 = (uint64_t)filter_x[1];
  y0 = (uint64_t)filter_y[0];
  y1 = (uint64_t)filter_y[1];
  all = x0 | x1 << 8 | y0 << 16 | y1 << 24;

  /* clang-format off */
  __asm__ volatile (
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
    MMI_MTC1(%[all], %[ftmp14])
    "punpcklbh  %[ftmp14],  %[ftmp14],      %[ftmp0]            \n\t"
    "pshufh     %[filter_x0], %[ftmp14],    %[ftmp0]            \n\t"
    MMI_LI(%[tmp0], 0x10)
    MMI_MTC1(%[tmp0], %[mask])
    "ssrld      %[ftmp14],  %[ftmp14],      %[mask]             \n\t"
    "pshufh     %[filter_x1], %[ftmp14],    %[ftmp0]            \n\t"
    "ssrld      %[ftmp14],  %[ftmp14],      %[mask]             \n\t"
    "pshufh     %[filter_y0], %[ftmp14],    %[ftmp0]            \n\t"
    "ssrld      %[ftmp14],  %[ftmp14],      %[mask]             \n\t"
    "pshufh     %[filter_y1], %[ftmp14],    %[ftmp0]            \n\t"
    MMI_LI(%[tmp0], 0x07)
    MMI_MTC1(%[tmp0], %[ftmp14])
    MMI_LI(%[tmp0], 0x0040004000400040)
    MMI_MTC1(%[tmp0], %[ff_ph_40])
    MMI_LI(%[tmp0], 0x00ff00ff00ff00ff)
    MMI_MTC1(%[tmp0], %[mask])
    // fdata3: fdata3[0] ~ fdata3[15]
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_16_A

    // fdata3 +src_stride*1: fdata3[0] ~ fdata3[15]
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_16_B
    // temp2: temp2[0] ~ temp2[15]
    VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_16_A

    // fdata3 +src_stride*2: fdata3[0] ~ fdata3[15]
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_16_A
    // temp2+16*1: temp2[0] ~ temp2[15]
    MMI_ADDIU(%[temp2_ptr], %[temp2_ptr], 0x10)
    VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_16_B

    "1:                                                         \n\t"
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_16_B
    MMI_ADDIU(%[temp2_ptr], %[temp2_ptr], 0x10)
    VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_16_A

    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_16_A
    MMI_ADDIU(%[temp2_ptr], %[temp2_ptr], 0x10)
    VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_16_B
    "addiu      %[counter], %[counter],     -0x01               \n\t"
    "bnez       %[counter], 1b                                  \n\t"
    : [ftmp0] "=&f"(ftmp[0]), [ftmp1] "=&f"(ftmp[1]), [ftmp2] "=&f"(ftmp[2]),
      [ftmp3] "=&f"(ftmp[3]), [ftmp4] "=&f"(ftmp[4]), [ftmp5] "=&f"(ftmp[5]),
      [ftmp6] "=&f"(ftmp[6]), [ftmp7] "=&f"(ftmp[7]), [ftmp8] "=&f"(ftmp[8]),
      [ftmp9] "=&f"(ftmp[9]), [ftmp10] "=&f"(ftmp[10]),
      [ftmp11] "=&f"(ftmp[11]), [ftmp12] "=&f"(ftmp[12]),
      [ftmp13] "=&f"(ftmp[13]), [ftmp14] "=&f"(ftmp[14]),
      [tmp0] "=&r"(tmp[0]), [src_ptr] "+&r"(src_ptr), [temp2_ptr] "+&r"(temp2_ptr),
      [counter]"+&r"(l_counter), [ff_ph_40] "=&f"(ff_ph_40), [mask] "=&f"(mask),
      [filter_x0] "=&f"(filter_x0), [filter_x1] "=&f"(filter_x1),
      [filter_y0] "=&f"(filter_y0), [filter_y1] "=&f"(filter_y1)
    : [src_stride] "r"((mips_reg)src_stride), [all] "r"(all)
    : "memory"
  );
  /* clang-format on */
}

#define SUBPIX_VAR16XN(H)                                                      \
  uint32_t vpx_sub_pixel_variance16x##H##_mmi(                                 \
      const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset,      \
      const uint8_t *ref_ptr, int ref_stride, uint32_t *sse) {                 \
    uint8_t temp2[16 * (H)];                                                   \
    var_filter_block2d_bil_16x(src_ptr, src_stride, x_offset, y_offset, temp2, \
                               ((H) - 2) / 2);                                 \
                                                                               \
    return vpx_variance16x##H##_mmi(temp2, 16, ref_ptr, ref_stride, sse);      \
  }

SUBPIX_VAR16XN(16)
SUBPIX_VAR16XN(8)

static inline void var_filter_block2d_bil_8x(const uint8_t *src_ptr,
                                             int src_stride, int x_offset,
                                             int y_offset, uint8_t *temp2,
                                             int counter) {
  uint8_t *temp2_ptr = temp2;
  mips_reg l_counter = counter;
  double ftmp[15];
  mips_reg tmp[2];
  double ff_ph_40, mask;
  uint64_t x0, x1, y0, y1, all;
  double filter_x0, filter_x1, filter_y0, filter_y1;
  const uint8_t *filter_x = bilinear_filters[x_offset];
  const uint8_t *filter_y = bilinear_filters[y_offset];
  x0 = (uint64_t)filter_x[0];
  x1 = (uint64_t)filter_x[1];
  y0 = (uint64_t)filter_y[0];
  y1 = (uint64_t)filter_y[1];
  all = x0 | x1 << 8 | y0 << 16 | y1 << 24;

  /* clang-format off */
  __asm__ volatile (
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
    MMI_MTC1(%[all], %[ftmp14])
    "punpcklbh  %[ftmp14],  %[ftmp14],      %[ftmp0]            \n\t"
    "pshufh     %[filter_x0], %[ftmp14],    %[ftmp0]            \n\t"
    MMI_LI(%[tmp0], 0x10)
    MMI_MTC1(%[tmp0], %[mask])
    "ssrld      %[ftmp14],  %[ftmp14],      %[mask]             \n\t"
    "pshufh     %[filter_x1], %[ftmp14],    %[ftmp0]            \n\t"
    "ssrld      %[ftmp14],  %[ftmp14],      %[mask]             \n\t"
    "pshufh     %[filter_y0], %[ftmp14],    %[ftmp0]            \n\t"
    "ssrld      %[ftmp14],  %[ftmp14],      %[mask]             \n\t"
    "pshufh     %[filter_y1], %[ftmp14],    %[ftmp0]            \n\t"
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
    MMI_LI(%[tmp0], 0x07)
    MMI_MTC1(%[tmp0], %[ftmp14])
    MMI_LI(%[tmp0], 0x0040004000400040)
    MMI_MTC1(%[tmp0], %[ff_ph_40])
    MMI_LI(%[tmp0], 0x00ff00ff00ff00ff)
    MMI_MTC1(%[tmp0], %[mask])

    // fdata3: fdata3[0] ~ fdata3[7]
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_8_A

    // fdata3 +src_stride*1: fdata3[0] ~ fdata3[7]
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_8_B
    // temp2: temp2[0] ~ temp2[7]
    VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_8_A

    // fdata3 +src_stride*2: fdata3[0] ~ fdata3[7]
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_8_A
    // temp2+8*1: temp2[0] ~ temp2[7]
    MMI_ADDIU(%[temp2_ptr], %[temp2_ptr], 0x08)
    VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_8_B

    "1:                                                         \n\t"
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_8_B
    MMI_ADDIU(%[temp2_ptr], %[temp2_ptr], 0x08)
    VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_8_A

    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_8_A
    MMI_ADDIU(%[temp2_ptr], %[temp2_ptr], 0x08)
    VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_8_B
    "addiu      %[counter], %[counter],     -0x01               \n\t"
    "bnez       %[counter], 1b                                  \n\t"
    : [ftmp0] "=&f"(ftmp[0]), [ftmp1] "=&f"(ftmp[1]), [ftmp2] "=&f"(ftmp[2]),
      [ftmp3] "=&f"(ftmp[3]), [ftmp4] "=&f"(ftmp[4]), [ftmp5] "=&f"(ftmp[5]),
      [ftmp6] "=&f"(ftmp[6]), [ftmp7] "=&f"(ftmp[7]), [ftmp8] "=&f"(ftmp[8]),
      [ftmp9] "=&f"(ftmp[9]), [ftmp10] "=&f"(ftmp[10]),
      [ftmp11] "=&f"(ftmp[11]), [ftmp12] "=&f"(ftmp[12]),
      [ftmp13] "=&f"(ftmp[13]), [ftmp14] "=&f"(ftmp[14]),
      [tmp0] "=&r"(tmp[0]), [src_ptr] "+&r"(src_ptr), [temp2_ptr] "+&r"(temp2_ptr),
      [counter]"+&r"(l_counter), [ff_ph_40] "=&f"(ff_ph_40), [mask] "=&f"(mask),
      [filter_x0] "=&f"(filter_x0), [filter_x1] "=&f"(filter_x1),
      [filter_y0] "=&f"(filter_y0), [filter_y1] "=&f"(filter_y1)
    : [src_stride] "r"((mips_reg)src_stride), [all] "r"(all)
    : "memory"
  );
  /* clang-format on */
}

#define SUBPIX_VAR8XN(H)                                                      \
  uint32_t vpx_sub_pixel_variance8x##H##_mmi(                                 \
      const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset,     \
      const uint8_t *ref_ptr, int ref_stride, uint32_t *sse) {                \
    uint8_t temp2[8 * (H)];                                                   \
    var_filter_block2d_bil_8x(src_ptr, src_stride, x_offset, y_offset, temp2, \
                              ((H) - 2) / 2);                                 \
                                                                              \
    return vpx_variance8x##H##_mmi(temp2, 8, ref_ptr, ref_stride, sse);       \
  }

SUBPIX_VAR8XN(16)
SUBPIX_VAR8XN(8)
SUBPIX_VAR8XN(4)

static inline void var_filter_block2d_bil_4x(const uint8_t *src_ptr,
                                             int src_stride, int x_offset,
                                             int y_offset, uint8_t *temp2,
                                             int counter) {
  uint8_t *temp2_ptr = temp2;
  mips_reg l_counter = counter;
  double ftmp[7];
  mips_reg tmp[2];
  double ff_ph_40, mask;
  uint64_t x0, x1, y0, y1, all;
  double filter_x0, filter_x1, filter_y0, filter_y1;
  const uint8_t *filter_x = bilinear_filters[x_offset];
  const uint8_t *filter_y = bilinear_filters[y_offset];
  x0 = (uint64_t)filter_x[0];
  x1 = (uint64_t)filter_x[1];
  y0 = (uint64_t)filter_y[0];
  y1 = (uint64_t)filter_y[1];
  all = x0 | x1 << 8 | y0 << 16 | y1 << 24;

  /* clang-format off */
  __asm__ volatile (
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
    MMI_MTC1(%[all], %[ftmp6])
    "punpcklbh  %[ftmp6],   %[ftmp6],       %[ftmp0]            \n\t"
    "pshufh     %[filter_x0], %[ftmp6],     %[ftmp0]            \n\t"
    MMI_LI(%[tmp0], 0x10)
    MMI_MTC1(%[tmp0], %[mask])
    "ssrld      %[ftmp6],   %[ftmp6],       %[mask]             \n\t"
    "pshufh     %[filter_x1], %[ftmp6],     %[ftmp0]            \n\t"
    "ssrld      %[ftmp6],   %[ftmp6],       %[mask]             \n\t"
    "pshufh     %[filter_y0], %[ftmp6],     %[ftmp0]            \n\t"
    "ssrld      %[ftmp6],   %[ftmp6],       %[mask]             \n\t"
    "pshufh     %[filter_y1], %[ftmp6],     %[ftmp0]            \n\t"
    "pxor       %[ftmp0],   %[ftmp0],       %[ftmp0]            \n\t"
    MMI_LI(%[tmp0], 0x07)
    MMI_MTC1(%[tmp0], %[ftmp6])
    MMI_LI(%[tmp0], 0x0040004000400040)
    MMI_MTC1(%[tmp0], %[ff_ph_40])
    MMI_LI(%[tmp0], 0x00ff00ff00ff00ff)
    MMI_MTC1(%[tmp0], %[mask])
    // fdata3: fdata3[0] ~ fdata3[3]
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_4_A

    // fdata3 +src_stride*1: fdata3[0] ~ fdata3[3]
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_4_B
    // temp2: temp2[0] ~ temp2[7]
    VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_4_A

    // fdata3 +src_stride*2: fdata3[0] ~ fdata3[3]
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_4_A
    // temp2+4*1: temp2[0] ~ temp2[7]
    MMI_ADDIU(%[temp2_ptr], %[temp2_ptr], 0x04)
    VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_4_B

    "1:                                                         \n\t"
    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_4_B
    MMI_ADDIU(%[temp2_ptr], %[temp2_ptr], 0x04)
    VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_4_A

    MMI_ADDU(%[src_ptr], %[src_ptr], %[src_stride])
    VAR_FILTER_BLOCK2D_BIL_FIRST_PASS_4_A
    MMI_ADDIU(%[temp2_ptr], %[temp2_ptr], 0x04)
    VAR_FILTER_BLOCK2D_BIL_SECOND_PASS_4_B
    "addiu      %[counter], %[counter],     -0x01               \n\t"
    "bnez       %[counter], 1b                                  \n\t"
    : [ftmp0] "=&f"(ftmp[0]), [ftmp1] "=&f"(ftmp[1]), [ftmp2] "=&f"(ftmp[2]),
      [ftmp3] "=&f"(ftmp[3]), [ftmp4] "=&f"(ftmp[4]), [ftmp5] "=&f"(ftmp[5]),
      [ftmp6] "=&f"(ftmp[6]), [tmp0] "=&r"(tmp[0]), [src_ptr] "+&r"(src_ptr),
      [temp2_ptr] "+&r"(temp2_ptr), [counter]"+&r"(l_counter),
      [ff_ph_40] "=&f"(ff_ph_40), [mask] "=&f"(mask),
      [filter_x0] "=&f"(filter_x0), [filter_x1] "=&f"(filter_x1),
      [filter_y0] "=&f"(filter_y0), [filter_y1] "=&f"(filter_y1)
    : [src_stride] "r"((mips_reg)src_stride), [all] "r"(all)
    : "memory"
  );
  /* clang-format on */
}

#define SUBPIX_VAR4XN(H)                                                      \
  uint32_t vpx_sub_pixel_variance4x##H##_mmi(                                 \
      const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset,     \
      const uint8_t *ref_ptr, int ref_stride, uint32_t *sse) {                \
    uint8_t temp2[4 * (H)];                                                   \
    var_filter_block2d_bil_4x(src_ptr, src_stride, x_offset, y_offset, temp2, \
                              ((H) - 2) / 2);                                 \
                                                                              \
    return vpx_variance4x##H##_mmi(temp2, 4, ref_ptr, ref_stride, sse);       \
  }

SUBPIX_VAR4XN(8)
SUBPIX_VAR4XN(4)

#define SUBPIX_AVG_VAR(W, H)                                                   \
  uint32_t vpx_sub_pixel_avg_variance##W##x##H##_mmi(                          \
      const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset,      \
      const uint8_t *ref_ptr, int ref_stride, uint32_t *sse,                   \
      const uint8_t *second_pred) {                                            \
    uint16_t fdata3[((H) + 1) * (W)];                                          \
    uint8_t temp2[(H) * (W)];                                                  \
    DECLARE_ALIGNED(16, uint8_t, temp3[(H) * (W)]);                            \
                                                                               \
    var_filter_block2d_bil_first_pass(src_ptr, fdata3, src_stride, 1, (H) + 1, \
                                      W, bilinear_filters[x_offset]);          \
    var_filter_block2d_bil_second_pass(fdata3, temp2, W, W, H, W,              \
                                       bilinear_filters[y_offset]);            \
                                                                               \
    vpx_comp_avg_pred_c(temp3, second_pred, W, H, temp2, W);                   \
                                                                               \
    return vpx_variance##W##x##H##_mmi(temp3, W, ref_ptr, ref_stride, sse);    \
  }

SUBPIX_AVG_VAR(64, 64)
SUBPIX_AVG_VAR(64, 32)
SUBPIX_AVG_VAR(32, 64)
SUBPIX_AVG_VAR(32, 32)
SUBPIX_AVG_VAR(32, 16)
SUBPIX_AVG_VAR(16, 32)
SUBPIX_AVG_VAR(16, 16)
SUBPIX_AVG_VAR(16, 8)
SUBPIX_AVG_VAR(8, 16)
SUBPIX_AVG_VAR(8, 8)
SUBPIX_AVG_VAR(8, 4)
SUBPIX_AVG_VAR(4, 8)
SUBPIX_AVG_VAR(4, 4)
