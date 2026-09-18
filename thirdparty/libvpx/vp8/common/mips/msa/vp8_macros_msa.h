/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_COMMON_MIPS_MSA_VP8_MACROS_MSA_H_
#define VPX_VP8_COMMON_MIPS_MSA_VP8_MACROS_MSA_H_

#include <msa.h>

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"

#define LD_B(RTYPE, psrc) *((const RTYPE *)(psrc))
#define LD_UB(...) LD_B(v16u8, __VA_ARGS__)
#define LD_SB(...) LD_B(v16i8, __VA_ARGS__)

#define LD_H(RTYPE, psrc) *((const RTYPE *)(psrc))
#define LD_UH(...) LD_H(v8u16, __VA_ARGS__)
#define LD_SH(...) LD_H(v8i16, __VA_ARGS__)

#define LD_W(RTYPE, psrc) *((const RTYPE *)(psrc))
#define LD_UW(...) LD_W(v4u32, __VA_ARGS__)
#define LD_SW(...) LD_W(v4i32, __VA_ARGS__)

#define ST_B(RTYPE, in, pdst) *((RTYPE *)(pdst)) = (in)
#define ST_UB(...) ST_B(v16u8, __VA_ARGS__)
#define ST_SB(...) ST_B(v16i8, __VA_ARGS__)

#define ST_H(RTYPE, in, pdst) *((RTYPE *)(pdst)) = (in)
#define ST_UH(...) ST_H(v8u16, __VA_ARGS__)
#define ST_SH(...) ST_H(v8i16, __VA_ARGS__)

#define ST_W(RTYPE, in, pdst) *((RTYPE *)(pdst)) = (in)
#define ST_SW(...) ST_W(v4i32, __VA_ARGS__)

#if (__mips_isa_rev >= 6)
#define LW(psrc)                                        \
  ({                                                    \
    const uint8_t *lw_psrc_m = (const uint8_t *)(psrc); \
    uint32_t lw_val_m;                                  \
                                                        \
    asm volatile("lw  %[lw_val_m],  %[lw_psrc_m]  \n\t" \
                                                        \
                 : [lw_val_m] "=r"(lw_val_m)            \
                 : [lw_psrc_m] "m"(*lw_psrc_m));        \
                                                        \
    lw_val_m;                                           \
  })

#if (__mips == 64)
#define LD(psrc)                                        \
  ({                                                    \
    const uint8_t *ld_psrc_m = (const uint8_t *)(psrc); \
    uint64_t ld_val_m = 0;                              \
                                                        \
    asm volatile("ld  %[ld_val_m],  %[ld_psrc_m]  \n\t" \
                                                        \
                 : [ld_val_m] "=r"(ld_val_m)            \
                 : [ld_psrc_m] "m"(*ld_psrc_m));        \
                                                        \
    ld_val_m;                                           \
  })
#else  // !(__mips == 64)
#define LD(psrc)                                                  \
  ({                                                              \
    const uint8_t *ld_psrc_m = (const uint8_t *)(psrc);           \
    uint32_t ld_val0_m, ld_val1_m;                                \
    uint64_t ld_val_m = 0;                                        \
                                                                  \
    ld_val0_m = LW(ld_psrc_m);                                    \
    ld_val1_m = LW(ld_psrc_m + 4);                                \
                                                                  \
    ld_val_m = (uint64_t)(ld_val1_m);                             \
    ld_val_m = (uint64_t)((ld_val_m << 32) & 0xFFFFFFFF00000000); \
    ld_val_m = (uint64_t)(ld_val_m | (uint64_t)ld_val0_m);        \
                                                                  \
    ld_val_m;                                                     \
  })
#endif  // (__mips == 64)

#define SH(val, pdst)                                   \
  {                                                     \
    uint8_t *sh_pdst_m = (uint8_t *)(pdst);             \
    const uint16_t sh_val_m = (val);                    \
                                                        \
    asm volatile("sh  %[sh_val_m],  %[sh_pdst_m]  \n\t" \
                                                        \
                 : [sh_pdst_m] "=m"(*sh_pdst_m)         \
                 : [sh_val_m] "r"(sh_val_m));           \
  }

#define SW(val, pdst)                                   \
  {                                                     \
    uint8_t *sw_pdst_m = (uint8_t *)(pdst);             \
    const uint32_t sw_val_m = (val);                    \
                                                        \
    asm volatile("sw  %[sw_val_m],  %[sw_pdst_m]  \n\t" \
                                                        \
                 : [sw_pdst_m] "=m"(*sw_pdst_m)         \
                 : [sw_val_m] "r"(sw_val_m));           \
  }

#define SD(val, pdst)                                   \
  {                                                     \
    uint8_t *sd_pdst_m = (uint8_t *)(pdst);             \
    const uint64_t sd_val_m = (val);                    \
                                                        \
    asm volatile("sd  %[sd_val_m],  %[sd_pdst_m]  \n\t" \
                                                        \
                 : [sd_pdst_m] "=m"(*sd_pdst_m)         \
                 : [sd_val_m] "r"(sd_val_m));           \
  }
#else  // !(__mips_isa_rev >= 6)
#define LW(psrc)                                        \
  ({                                                    \
    const uint8_t *lw_psrc_m = (const uint8_t *)(psrc); \
    uint32_t lw_val_m;                                  \
                                                        \
    asm volatile(                                       \
        "lwr %[lw_val_m], 0(%[lw_psrc_m]) \n\t"         \
        "lwl %[lw_val_m], 3(%[lw_psrc_m]) \n\t"         \
        : [lw_val_m] "=&r"(lw_val_m)                    \
        : [lw_psrc_m] "r"(lw_psrc_m));                  \
                                                        \
    lw_val_m;                                           \
  })

#if (__mips == 64)
#define LD(psrc)                                        \
  ({                                                    \
    const uint8_t *ld_psrc_m = (const uint8_t *)(psrc); \
    uint64_t ld_val_m = 0;                              \
                                                        \
    asm volatile(                                       \
        "ldr %[ld_val_m], 0(%[ld_psrc_m]) \n\t"         \
        "ldl %[ld_val_m], 7(%[ld_psrc_m]) \n\t"         \
        : [ld_val_m] "=&r"(ld_val_m)                    \
        : [ld_psrc_m] "r"(ld_psrc_m));                  \
                                                        \
    ld_val_m;                                           \
  })
#else  // !(__mips == 64)
#define LD(psrc)                                                  \
  ({                                                              \
    const uint8_t *ld_psrc_m1 = (const uint8_t *)(psrc);          \
    uint32_t ld_val0_m, ld_val1_m;                                \
    uint64_t ld_val_m = 0;                                        \
                                                                  \
    ld_val0_m = LW(ld_psrc_m1);                                   \
    ld_val1_m = LW(ld_psrc_m1 + 4);                               \
                                                                  \
    ld_val_m = (uint64_t)(ld_val1_m);                             \
    ld_val_m = (uint64_t)((ld_val_m << 32) & 0xFFFFFFFF00000000); \
    ld_val_m = (uint64_t)(ld_val_m | (uint64_t)ld_val0_m);        \
                                                                  \
    ld_val_m;                                                     \
  })
#endif  // (__mips == 64)
#define SH(val, pdst)                                    \
  {                                                      \
    uint8_t *sh_pdst_m = (uint8_t *)(pdst);              \
    const uint16_t sh_val_m = (val);                     \
                                                         \
    asm volatile("ush  %[sh_val_m],  %[sh_pdst_m]  \n\t" \
                                                         \
                 : [sh_pdst_m] "=m"(*sh_pdst_m)          \
                 : [sh_val_m] "r"(sh_val_m));            \
  }

#define SW(val, pdst)                                    \
  {                                                      \
    uint8_t *sw_pdst_m = (uint8_t *)(pdst);              \
    const uint32_t sw_val_m = (val);                     \
                                                         \
    asm volatile("usw  %[sw_val_m],  %[sw_pdst_m]  \n\t" \
                                                         \
                 : [sw_pdst_m] "=m"(*sw_pdst_m)          \
                 : [sw_val_m] "r"(sw_val_m));            \
  }

#define SD(val, pdst)                                           \
  {                                                             \
    uint8_t *sd_pdst_m1 = (uint8_t *)(pdst);                    \
    uint32_t sd_val0_m, sd_val1_m;                              \
                                                                \
    sd_val0_m = (uint32_t)((val) & 0x00000000FFFFFFFF);         \
    sd_val1_m = (uint32_t)(((val) >> 32) & 0x00000000FFFFFFFF); \
                                                                \
    SW(sd_val0_m, sd_pdst_m1);                                  \
    SW(sd_val1_m, sd_pdst_m1 + 4);                              \
  }
#endif  // (__mips_isa_rev >= 6)

/* Description : Load 4 words with stride
   Arguments   : Inputs  - psrc, stride
                 Outputs - out0, out1, out2, out3
   Details     : Load word in 'out0' from (psrc)
                 Load word in 'out1' from (psrc + stride)
                 Load word in 'out2' from (psrc + 2 * stride)
                 Load word in 'out3' from (psrc + 3 * stride)
*/
#define LW4(psrc, stride, out0, out1, out2, out3) \
  {                                               \
    out0 = LW((psrc));                            \
    out1 = LW((psrc) + stride);                   \
    out2 = LW((psrc) + 2 * stride);               \
    out3 = LW((psrc) + 3 * stride);               \
  }

/* Description : Load double words with stride
   Arguments   : Inputs  - psrc, stride
                 Outputs - out0, out1
   Details     : Load double word in 'out0' from (psrc)
                 Load double word in 'out1' from (psrc + stride)
*/
#define LD2(psrc, stride, out0, out1) \
  {                                   \
    out0 = LD((psrc));                \
    out1 = LD((psrc) + stride);       \
  }
#define LD4(psrc, stride, out0, out1, out2, out3) \
  {                                               \
    LD2((psrc), stride, out0, out1);              \
    LD2((psrc) + 2 * stride, stride, out2, out3); \
  }

/* Description : Store 4 words with stride
   Arguments   : Inputs - in0, in1, in2, in3, pdst, stride
   Details     : Store word from 'in0' to (pdst)
                 Store word from 'in1' to (pdst + stride)
                 Store word from 'in2' to (pdst + 2 * stride)
                 Store word from 'in3' to (pdst + 3 * stride)
*/
#define SW4(in0, in1, in2, in3, pdst, stride) \
  {                                           \
    SW(in0, (pdst));                          \
    SW(in1, (pdst) + stride);                 \
    SW(in2, (pdst) + 2 * stride);             \
    SW(in3, (pdst) + 3 * stride);             \
  }

/* Description : Store 4 double words with stride
   Arguments   : Inputs - in0, in1, in2, in3, pdst, stride
   Details     : Store double word from 'in0' to (pdst)
                 Store double word from 'in1' to (pdst + stride)
                 Store double word from 'in2' to (pdst + 2 * stride)
                 Store double word from 'in3' to (pdst + 3 * stride)
*/
#define SD4(in0, in1, in2, in3, pdst, stride) \
  {                                           \
    SD(in0, (pdst));                          \
    SD(in1, (pdst) + stride);                 \
    SD(in2, (pdst) + 2 * stride);             \
    SD(in3, (pdst) + 3 * stride);             \
  }

/* Description : Load vectors with 16 byte elements with stride
   Arguments   : Inputs  - psrc, stride
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Load 16 byte elements in 'out0' from (psrc)
                 Load 16 byte elements in 'out1' from (psrc + stride)
*/
#define LD_B2(RTYPE, psrc, stride, out0, out1) \
  {                                            \
    out0 = LD_B(RTYPE, (psrc));                \
    out1 = LD_B(RTYPE, (psrc) + stride);       \
  }
#define LD_UB2(...) LD_B2(v16u8, __VA_ARGS__)
#define LD_SB2(...) LD_B2(v16i8, __VA_ARGS__)

#define LD_B3(RTYPE, psrc, stride, out0, out1, out2) \
  {                                                  \
    LD_B2(RTYPE, (psrc), stride, out0, out1);        \
    out2 = LD_B(RTYPE, (psrc) + 2 * stride);         \
  }
#define LD_UB3(...) LD_B3(v16u8, __VA_ARGS__)
#define LD_SB3(...) LD_B3(v16i8, __VA_ARGS__)

#define LD_B4(RTYPE, psrc, stride, out0, out1, out2, out3) \
  {                                                        \
    LD_B2(RTYPE, (psrc), stride, out0, out1);              \
    LD_B2(RTYPE, (psrc) + 2 * stride, stride, out2, out3); \
  }
#define LD_UB4(...) LD_B4(v16u8, __VA_ARGS__)
#define LD_SB4(...) LD_B4(v16i8, __VA_ARGS__)

#define LD_B5(RTYPE, psrc, stride, out0, out1, out2, out3, out4) \
  {                                                              \
    LD_B4(RTYPE, (psrc), stride, out0, out1, out2, out3);        \
    out4 = LD_B(RTYPE, (psrc) + 4 * stride);                     \
  }
#define LD_UB5(...) LD_B5(v16u8, __VA_ARGS__)
#define LD_SB5(...) LD_B5(v16i8, __VA_ARGS__)

#define LD_B8(RTYPE, psrc, stride, out0, out1, out2, out3, out4, out5, out6, \
              out7)                                                          \
  {                                                                          \
    LD_B4(RTYPE, (psrc), stride, out0, out1, out2, out3);                    \
    LD_B4(RTYPE, (psrc) + 4 * stride, stride, out4, out5, out6, out7);       \
  }
#define LD_UB8(...) LD_B8(v16u8, __VA_ARGS__)
#define LD_SB8(...) LD_B8(v16i8, __VA_ARGS__)

/* Description : Load vectors with 8 halfword elements with stride
   Arguments   : Inputs  - psrc, stride
                 Outputs - out0, out1
   Details     : Load 8 halfword elements in 'out0' from (psrc)
                 Load 8 halfword elements in 'out1' from (psrc + stride)
*/
#define LD_H2(RTYPE, psrc, stride, out0, out1) \
  {                                            \
    out0 = LD_H(RTYPE, (psrc));                \
    out1 = LD_H(RTYPE, (psrc) + (stride));     \
  }
#define LD_SH2(...) LD_H2(v8i16, __VA_ARGS__)

#define LD_H4(RTYPE, psrc, stride, out0, out1, out2, out3) \
  {                                                        \
    LD_H2(RTYPE, (psrc), stride, out0, out1);              \
    LD_H2(RTYPE, (psrc) + 2 * stride, stride, out2, out3); \
  }
#define LD_SH4(...) LD_H4(v8i16, __VA_ARGS__)

/* Description : Load 2 vectors of signed word elements with stride
   Arguments   : Inputs  - psrc, stride
                 Outputs - out0, out1
                 Return Type - signed word
*/
#define LD_SW2(psrc, stride, out0, out1) \
  {                                      \
    out0 = LD_SW((psrc));                \
    out1 = LD_SW((psrc) + stride);       \
  }

/* Description : Store vectors of 16 byte elements with stride
   Arguments   : Inputs - in0, in1, pdst, stride
   Details     : Store 16 byte elements from 'in0' to (pdst)
                 Store 16 byte elements from 'in1' to (pdst + stride)
*/
#define ST_B2(RTYPE, in0, in1, pdst, stride) \
  {                                          \
    ST_B(RTYPE, in0, (pdst));                \
    ST_B(RTYPE, in1, (pdst) + stride);       \
  }
#define ST_UB2(...) ST_B2(v16u8, __VA_ARGS__)

#define ST_B4(RTYPE, in0, in1, in2, in3, pdst, stride)   \
  {                                                      \
    ST_B2(RTYPE, in0, in1, (pdst), stride);              \
    ST_B2(RTYPE, in2, in3, (pdst) + 2 * stride, stride); \
  }
#define ST_UB4(...) ST_B4(v16u8, __VA_ARGS__)
#define ST_SB4(...) ST_B4(v16i8, __VA_ARGS__)

#define ST_B8(RTYPE, in0, in1, in2, in3, in4, in5, in6, in7, pdst, stride) \
  {                                                                        \
    ST_B4(RTYPE, in0, in1, in2, in3, pdst, stride);                        \
    ST_B4(RTYPE, in4, in5, in6, in7, (pdst) + 4 * stride, stride);         \
  }
#define ST_UB8(...) ST_B8(v16u8, __VA_ARGS__)

/* Description : Store vectors of 8 halfword elements with stride
   Arguments   : Inputs - in0, in1, pdst, stride
   Details     : Store 8 halfword elements from 'in0' to (pdst)
                 Store 8 halfword elements from 'in1' to (pdst + stride)
*/
#define ST_H2(RTYPE, in0, in1, pdst, stride) \
  {                                          \
    ST_H(RTYPE, in0, (pdst));                \
    ST_H(RTYPE, in1, (pdst) + stride);       \
  }
#define ST_SH2(...) ST_H2(v8i16, __VA_ARGS__)

/* Description : Store vectors of word elements with stride
   Arguments   : Inputs - in0, in1, pdst, stride
   Details     : Store 4 word elements from 'in0' to (pdst)
                 Store 4 word elements from 'in1' to (pdst + stride)
*/
#define ST_SW2(in0, in1, pdst, stride) \
  {                                    \
    ST_SW(in0, (pdst));                \
    ST_SW(in1, (pdst) + stride);       \
  }

/* Description : Store 2x4 byte block to destination memory from input vector
   Arguments   : Inputs - in, stidx, pdst, stride
   Details     : Index 'stidx' halfword element from 'in' vector is copied to
                 the GP register and stored to (pdst)
                 Index 'stidx+1' halfword element from 'in' vector is copied to
                 the GP register and stored to (pdst + stride)
                 Index 'stidx+2' halfword element from 'in' vector is copied to
                 the GP register and stored to (pdst + 2 * stride)
                 Index 'stidx+3' halfword element from 'in' vector is copied to
                 the GP register and stored to (pdst + 3 * stride)
*/
#define ST2x4_UB(in, stidx, pdst, stride)            \
  {                                                  \
    uint16_t out0_m, out1_m, out2_m, out3_m;         \
    uint8_t *pblk_2x4_m = (uint8_t *)(pdst);         \
                                                     \
    out0_m = __msa_copy_u_h((v8i16)in, (stidx));     \
    out1_m = __msa_copy_u_h((v8i16)in, (stidx + 1)); \
    out2_m = __msa_copy_u_h((v8i16)in, (stidx + 2)); \
    out3_m = __msa_copy_u_h((v8i16)in, (stidx + 3)); \
                                                     \
    SH(out0_m, pblk_2x4_m);                          \
    SH(out1_m, pblk_2x4_m + stride);                 \
    SH(out2_m, pblk_2x4_m + 2 * stride);             \
    SH(out3_m, pblk_2x4_m + 3 * stride);             \
  }

/* Description : Store 4x4 byte block to destination memory from input vector
   Arguments   : Inputs - in0, in1, pdst, stride
   Details     : 'Idx0' word element from input vector 'in0' is copied to the
                 GP register and stored to (pdst)
                 'Idx1' word element from input vector 'in0' is copied to the
                 GP register and stored to (pdst + stride)
                 'Idx2' word element from input vector 'in0' is copied to the
                 GP register and stored to (pdst + 2 * stride)
                 'Idx3' word element from input vector 'in0' is copied to the
                 GP register and stored to (pdst + 3 * stride)
*/
#define ST4x4_UB(in0, in1, idx0, idx1, idx2, idx3, pdst, stride) \
  {                                                              \
    uint32_t out0_m, out1_m, out2_m, out3_m;                     \
    uint8_t *pblk_4x4_m = (uint8_t *)(pdst);                     \
                                                                 \
    out0_m = __msa_copy_u_w((v4i32)in0, idx0);                   \
    out1_m = __msa_copy_u_w((v4i32)in0, idx1);                   \
    out2_m = __msa_copy_u_w((v4i32)in1, idx2);                   \
    out3_m = __msa_copy_u_w((v4i32)in1, idx3);                   \
                                                                 \
    SW4(out0_m, out1_m, out2_m, out3_m, pblk_4x4_m, stride);     \
  }
#define ST4x8_UB(in0, in1, pdst, stride)                           \
  {                                                                \
    uint8_t *pblk_4x8 = (uint8_t *)(pdst);                         \
                                                                   \
    ST4x4_UB(in0, in0, 0, 1, 2, 3, pblk_4x8, stride);              \
    ST4x4_UB(in1, in1, 0, 1, 2, 3, pblk_4x8 + 4 * stride, stride); \
  }

/* Description : Store 8x1 byte block to destination memory from input vector
   Arguments   : Inputs - in, pdst
   Details     : Index 0 double word element from 'in' vector is copied to the
                 GP register and stored to (pdst)
*/
#define ST8x1_UB(in, pdst)                 \
  {                                        \
    uint64_t out0_m;                       \
                                           \
    out0_m = __msa_copy_u_d((v2i64)in, 0); \
    SD(out0_m, pdst);                      \
  }

/* Description : Store 8x2 byte block to destination memory from input vector
   Arguments   : Inputs - in, pdst, stride
   Details     : Index 0 double word element from 'in' vector is copied to the
                 GP register and stored to (pdst)
                 Index 1 double word element from 'in' vector is copied to the
                 GP register and stored to (pdst + stride)
*/
#define ST8x2_UB(in, pdst, stride)           \
  {                                          \
    uint64_t out0_m, out1_m;                 \
    uint8_t *pblk_8x2_m = (uint8_t *)(pdst); \
                                             \
    out0_m = __msa_copy_u_d((v2i64)in, 0);   \
    out1_m = __msa_copy_u_d((v2i64)in, 1);   \
                                             \
    SD(out0_m, pblk_8x2_m);                  \
    SD(out1_m, pblk_8x2_m + stride);         \
  }

/* Description : Store 8x4 byte block to destination memory from input
                 vectors
   Arguments   : Inputs - in0, in1, pdst, stride
   Details     : Index 0 double word element from 'in0' vector is copied to the
                 GP register and stored to (pdst)
                 Index 1 double word element from 'in0' vector is copied to the
                 GP register and stored to (pdst + stride)
                 Index 0 double word element from 'in1' vector is copied to the
                 GP register and stored to (pdst + 2 * stride)
                 Index 1 double word element from 'in1' vector is copied to the
                 GP register and stored to (pdst + 3 * stride)
*/
#define ST8x4_UB(in0, in1, pdst, stride)                     \
  {                                                          \
    uint64_t out0_m, out1_m, out2_m, out3_m;                 \
    uint8_t *pblk_8x4_m = (uint8_t *)(pdst);                 \
                                                             \
    out0_m = __msa_copy_u_d((v2i64)in0, 0);                  \
    out1_m = __msa_copy_u_d((v2i64)in0, 1);                  \
    out2_m = __msa_copy_u_d((v2i64)in1, 0);                  \
    out3_m = __msa_copy_u_d((v2i64)in1, 1);                  \
                                                             \
    SD4(out0_m, out1_m, out2_m, out3_m, pblk_8x4_m, stride); \
  }

/* Description : Immediate number of elements to slide with zero
   Arguments   : Inputs  - in0, in1, slide_val
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Byte elements from 'zero_m' vector are slid into 'in0' by
                 value specified in the 'slide_val'
*/
#define SLDI_B2_0(RTYPE, in0, in1, out0, out1, slide_val)             \
  {                                                                   \
    v16i8 zero_m = { 0 };                                             \
                                                                      \
    out0 = (RTYPE)__msa_sldi_b((v16i8)zero_m, (v16i8)in0, slide_val); \
    out1 = (RTYPE)__msa_sldi_b((v16i8)zero_m, (v16i8)in1, slide_val); \
  }
#define SLDI_B2_0_UB(...) SLDI_B2_0(v16u8, __VA_ARGS__)

/* Description : Immediate number of elements to slide
   Arguments   : Inputs  - in0_0, in0_1, in1_0, in1_1, slide_val
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Byte elements from 'in0_0' vector are slid into 'in1_0' by
                 value specified in the 'slide_val'
*/
#define SLDI_B2(RTYPE, in0_0, in0_1, in1_0, in1_1, out0, out1, slide_val) \
  {                                                                       \
    out0 = (RTYPE)__msa_sldi_b((v16i8)in0_0, (v16i8)in1_0, slide_val);    \
    out1 = (RTYPE)__msa_sldi_b((v16i8)in0_1, (v16i8)in1_1, slide_val);    \
  }

#define SLDI_B3(RTYPE, in0_0, in0_1, in0_2, in1_0, in1_1, in1_2, out0, out1, \
                out2, slide_val)                                             \
  {                                                                          \
    SLDI_B2(RTYPE, in0_0, in0_1, in1_0, in1_1, out0, out1, slide_val);       \
    out2 = (RTYPE)__msa_sldi_b((v16i8)in0_2, (v16i8)in1_2, slide_val);       \
  }
#define SLDI_B3_UH(...) SLDI_B3(v8u16, __VA_ARGS__)

/* Description : Shuffle byte vector elements as per mask vector
   Arguments   : Inputs  - in0, in1, in2, in3, mask0, mask1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Byte elements from 'in0' & 'in1' are copied selectively to
                 'out0' as per control vector 'mask0'
*/
#define VSHF_B2(RTYPE, in0, in1, in2, in3, mask0, mask1, out0, out1)  \
  {                                                                   \
    out0 = (RTYPE)__msa_vshf_b((v16i8)mask0, (v16i8)in1, (v16i8)in0); \
    out1 = (RTYPE)__msa_vshf_b((v16i8)mask1, (v16i8)in3, (v16i8)in2); \
  }
#define VSHF_B2_UB(...) VSHF_B2(v16u8, __VA_ARGS__)
#define VSHF_B2_SB(...) VSHF_B2(v16i8, __VA_ARGS__)
#define VSHF_B2_UH(...) VSHF_B2(v8u16, __VA_ARGS__)

#define VSHF_B3(RTYPE, in0, in1, in2, in3, in4, in5, mask0, mask1, mask2, \
                out0, out1, out2)                                         \
  {                                                                       \
    VSHF_B2(RTYPE, in0, in1, in2, in3, mask0, mask1, out0, out1);         \
    out2 = (RTYPE)__msa_vshf_b((v16i8)mask2, (v16i8)in5, (v16i8)in4);     \
  }
#define VSHF_B3_SB(...) VSHF_B3(v16i8, __VA_ARGS__)

/* Description : Shuffle halfword vector elements as per mask vector
   Arguments   : Inputs  - in0, in1, in2, in3, mask0, mask1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : halfword elements from 'in0' & 'in1' are copied selectively to
                 'out0' as per control vector 'mask0'
*/
#define VSHF_H2(RTYPE, in0, in1, in2, in3, mask0, mask1, out0, out1)  \
  {                                                                   \
    out0 = (RTYPE)__msa_vshf_h((v8i16)mask0, (v8i16)in1, (v8i16)in0); \
    out1 = (RTYPE)__msa_vshf_h((v8i16)mask1, (v8i16)in3, (v8i16)in2); \
  }
#define VSHF_H2_SH(...) VSHF_H2(v8i16, __VA_ARGS__)

/* Description : Dot product of byte vector elements
   Arguments   : Inputs  - mult0, mult1, cnst0, cnst1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Unsigned byte elements from 'mult0' are multiplied with
                 unsigned byte elements from 'cnst0' producing a result
                 twice the size of input i.e. unsigned halfword.
                 The multiplication result of adjacent odd-even elements
                 are added together and written to the 'out0' vector
*/
#define DOTP_UB2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1) \
  {                                                             \
    out0 = (RTYPE)__msa_dotp_u_h((v16u8)mult0, (v16u8)cnst0);   \
    out1 = (RTYPE)__msa_dotp_u_h((v16u8)mult1, (v16u8)cnst1);   \
  }
#define DOTP_UB2_UH(...) DOTP_UB2(v8u16, __VA_ARGS__)

#define DOTP_UB4(RTYPE, mult0, mult1, mult2, mult3, cnst0, cnst1, cnst2, \
                 cnst3, out0, out1, out2, out3)                          \
  {                                                                      \
    DOTP_UB2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1);             \
    DOTP_UB2(RTYPE, mult2, mult3, cnst2, cnst3, out2, out3);             \
  }
#define DOTP_UB4_UH(...) DOTP_UB4(v8u16, __VA_ARGS__)

/* Description : Dot product of byte vector elements
   Arguments   : Inputs  - mult0, mult1, cnst0, cnst1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Signed byte elements from 'mult0' are multiplied with
                 signed byte elements from 'cnst0' producing a result
                 twice the size of input i.e. signed halfword.
                 The multiplication result of adjacent odd-even elements
                 are added together and written to the 'out0' vector
*/
#define DOTP_SB2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1) \
  {                                                             \
    out0 = (RTYPE)__msa_dotp_s_h((v16i8)mult0, (v16i8)cnst0);   \
    out1 = (RTYPE)__msa_dotp_s_h((v16i8)mult1, (v16i8)cnst1);   \
  }
#define DOTP_SB2_SH(...) DOTP_SB2(v8i16, __VA_ARGS__)

#define DOTP_SB4(RTYPE, mult0, mult1, mult2, mult3, cnst0, cnst1, cnst2, \
                 cnst3, out0, out1, out2, out3)                          \
  {                                                                      \
    DOTP_SB2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1);             \
    DOTP_SB2(RTYPE, mult2, mult3, cnst2, cnst3, out2, out3);             \
  }
#define DOTP_SB4_SH(...) DOTP_SB4(v8i16, __VA_ARGS__)

/* Description : Dot product of halfword vector elements
   Arguments   : Inputs  - mult0, mult1, cnst0, cnst1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Signed halfword elements from 'mult0' are multiplied with
                 signed halfword elements from 'cnst0' producing a result
                 twice the size of input i.e. signed word.
                 The multiplication result of adjacent odd-even elements
                 are added together and written to the 'out0' vector
*/
#define DOTP_SH2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1) \
  {                                                             \
    out0 = (RTYPE)__msa_dotp_s_w((v8i16)mult0, (v8i16)cnst0);   \
    out1 = (RTYPE)__msa_dotp_s_w((v8i16)mult1, (v8i16)cnst1);   \
  }

#define DOTP_SH4(RTYPE, mult0, mult1, mult2, mult3, cnst0, cnst1, cnst2, \
                 cnst3, out0, out1, out2, out3)                          \
  {                                                                      \
    DOTP_SH2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1);             \
    DOTP_SH2(RTYPE, mult2, mult3, cnst2, cnst3, out2, out3);             \
  }
#define DOTP_SH4_SW(...) DOTP_SH4(v4i32, __VA_ARGS__)

/* Description : Dot product of word vector elements
   Arguments   : Inputs  - mult0, mult1, cnst0, cnst1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Signed word elements from 'mult0' are multiplied with
                 signed word elements from 'cnst0' producing a result
                 twice the size of input i.e. signed double word.
                 The multiplication result of adjacent odd-even elements
                 are added together and written to the 'out0' vector
*/
#define DOTP_SW2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1) \
  {                                                             \
    out0 = (RTYPE)__msa_dotp_s_d((v4i32)mult0, (v4i32)cnst0);   \
    out1 = (RTYPE)__msa_dotp_s_d((v4i32)mult1, (v4i32)cnst1);   \
  }
#define DOTP_SW2_SD(...) DOTP_SW2(v2i64, __VA_ARGS__)

/* Description : Dot product & addition of byte vector elements
   Arguments   : Inputs  - mult0, mult1, cnst0, cnst1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Signed byte elements from 'mult0' are multiplied with
                 signed byte elements from 'cnst0' producing a result
                 twice the size of input i.e. signed halfword.
                 The multiplication result of adjacent odd-even elements
                 are added to the 'out0' vector
*/
#define DPADD_SB2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1)            \
  {                                                                         \
    out0 = (RTYPE)__msa_dpadd_s_h((v8i16)out0, (v16i8)mult0, (v16i8)cnst0); \
    out1 = (RTYPE)__msa_dpadd_s_h((v8i16)out1, (v16i8)mult1, (v16i8)cnst1); \
  }
#define DPADD_SB2_SH(...) DPADD_SB2(v8i16, __VA_ARGS__)

#define DPADD_SB4(RTYPE, mult0, mult1, mult2, mult3, cnst0, cnst1, cnst2, \
                  cnst3, out0, out1, out2, out3)                          \
  {                                                                       \
    DPADD_SB2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1);             \
    DPADD_SB2(RTYPE, mult2, mult3, cnst2, cnst3, out2, out3);             \
  }
#define DPADD_SB4_SH(...) DPADD_SB4(v8i16, __VA_ARGS__)

/* Description : Dot product & addition of halfword vector elements
   Arguments   : Inputs  - mult0, mult1, cnst0, cnst1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Signed halfword elements from 'mult0' are multiplied with
                 signed halfword elements from 'cnst0' producing a result
                 twice the size of input i.e. signed word.
                 The multiplication result of adjacent odd-even elements
                 are added to the 'out0' vector
*/
#define DPADD_SH2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1)            \
  {                                                                         \
    out0 = (RTYPE)__msa_dpadd_s_w((v4i32)out0, (v8i16)mult0, (v8i16)cnst0); \
    out1 = (RTYPE)__msa_dpadd_s_w((v4i32)out1, (v8i16)mult1, (v8i16)cnst1); \
  }
#define DPADD_SH2_SW(...) DPADD_SH2(v4i32, __VA_ARGS__)

#define DPADD_SH4(RTYPE, mult0, mult1, mult2, mult3, cnst0, cnst1, cnst2, \
                  cnst3, out0, out1, out2, out3)                          \
  {                                                                       \
    DPADD_SH2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1);             \
    DPADD_SH2(RTYPE, mult2, mult3, cnst2, cnst3, out2, out3);             \
  }
#define DPADD_SH4_SW(...) DPADD_SH4(v4i32, __VA_ARGS__)

/* Description : Dot product & addition of double word vector elements
   Arguments   : Inputs  - mult0, mult1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Each signed word element from 'mult0' is multiplied with itself
                 producing an intermediate result twice the size of it
                 i.e. signed double word
                 The multiplication result of adjacent odd-even elements
                 are added to the 'out0' vector
*/
#define DPADD_SD2(RTYPE, mult0, mult1, out0, out1)                          \
  {                                                                         \
    out0 = (RTYPE)__msa_dpadd_s_d((v2i64)out0, (v4i32)mult0, (v4i32)mult0); \
    out1 = (RTYPE)__msa_dpadd_s_d((v2i64)out1, (v4i32)mult1, (v4i32)mult1); \
  }
#define DPADD_SD2_SD(...) DPADD_SD2(v2i64, __VA_ARGS__)

/* Description : Clips all signed halfword elements of input vector
                 between 0 & 255
   Arguments   : Input  - in
                 Output - out_m
                 Return Type - signed halfword
*/
#define CLIP_SH_0_255(in)                              \
  ({                                                   \
    v8i16 max_m = __msa_ldi_h(255);                    \
    v8i16 out_m;                                       \
                                                       \
    out_m = __msa_maxi_s_h((v8i16)in, 0);              \
    out_m = __msa_min_s_h((v8i16)max_m, (v8i16)out_m); \
    out_m;                                             \
  })
#define CLIP_SH2_0_255(in0, in1) \
  {                              \
    in0 = CLIP_SH_0_255(in0);    \
    in1 = CLIP_SH_0_255(in1);    \
  }
#define CLIP_SH4_0_255(in0, in1, in2, in3) \
  {                                        \
    CLIP_SH2_0_255(in0, in1);              \
    CLIP_SH2_0_255(in2, in3);              \
  }

/* Description : Clips all signed word elements of input vector
                 between 0 & 255
   Arguments   : Input  - in
                 Output - out_m
                 Return Type - signed word
*/
#define CLIP_SW_0_255(in)                              \
  ({                                                   \
    v4i32 max_m = __msa_ldi_w(255);                    \
    v4i32 out_m;                                       \
                                                       \
    out_m = __msa_maxi_s_w((v4i32)in, 0);              \
    out_m = __msa_min_s_w((v4i32)max_m, (v4i32)out_m); \
    out_m;                                             \
  })

/* Description : Horizontal addition of 4 signed word elements of input vector
   Arguments   : Input  - in       (signed word vector)
                 Output - sum_m    (i32 sum)
                 Return Type - signed word (GP)
   Details     : 4 signed word elements of 'in' vector are added together and
                 the resulting integer sum is returned
*/
#define HADD_SW_S32(in)                            \
  ({                                               \
    v2i64 res0_m, res1_m;                          \
    int32_t sum_m;                                 \
                                                   \
    res0_m = __msa_hadd_s_d((v4i32)in, (v4i32)in); \
    res1_m = __msa_splati_d(res0_m, 1);            \
    res0_m = res0_m + res1_m;                      \
    sum_m = __msa_copy_s_w((v4i32)res0_m, 0);      \
    sum_m;                                         \
  })

/* Description : Horizontal addition of 8 unsigned halfword elements
   Arguments   : Inputs  - in       (unsigned halfword vector)
                 Outputs - sum_m    (u32 sum)
                 Return Type - unsigned word
   Details     : 8 unsigned halfword elements of input vector are added
                 together and the resulting integer sum is returned
*/
#define HADD_UH_U32(in)                               \
  ({                                                  \
    v4u32 res_m;                                      \
    v2u64 res0_m, res1_m;                             \
    uint32_t sum_m;                                   \
                                                      \
    res_m = __msa_hadd_u_w((v8u16)in, (v8u16)in);     \
    res0_m = __msa_hadd_u_d(res_m, res_m);            \
    res1_m = (v2u64)__msa_splati_d((v2i64)res0_m, 1); \
    res0_m = res0_m + res1_m;                         \
    sum_m = __msa_copy_u_w((v4i32)res0_m, 0);         \
    sum_m;                                            \
  })

/* Description : Horizontal addition of unsigned byte vector elements
   Arguments   : Inputs  - in0, in1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Each unsigned odd byte element from 'in0' is added to
                 even unsigned byte element from 'in0' (pairwise) and the
                 halfword result is written to 'out0'
*/
#define HADD_UB2(RTYPE, in0, in1, out0, out1)             \
  {                                                       \
    out0 = (RTYPE)__msa_hadd_u_h((v16u8)in0, (v16u8)in0); \
    out1 = (RTYPE)__msa_hadd_u_h((v16u8)in1, (v16u8)in1); \
  }
#define HADD_UB2_UH(...) HADD_UB2(v8u16, __VA_ARGS__)

/* Description : Horizontal subtraction of unsigned byte vector elements
   Arguments   : Inputs  - in0, in1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Each unsigned odd byte element from 'in0' is subtracted from
                 even unsigned byte element from 'in0' (pairwise) and the
                 halfword result is written to 'out0'
*/
#define HSUB_UB2(RTYPE, in0, in1, out0, out1)             \
  {                                                       \
    out0 = (RTYPE)__msa_hsub_u_h((v16u8)in0, (v16u8)in0); \
    out1 = (RTYPE)__msa_hsub_u_h((v16u8)in1, (v16u8)in1); \
  }
#define HSUB_UB2_SH(...) HSUB_UB2(v8i16, __VA_ARGS__)

/* Description : Horizontal subtraction of signed halfword vector elements
   Arguments   : Inputs  - in0, in1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Each signed odd halfword element from 'in0' is subtracted from
                 even signed halfword element from 'in0' (pairwise) and the
                 word result is written to 'out0'
*/
#define HSUB_UH2(RTYPE, in0, in1, out0, out1)             \
  {                                                       \
    out0 = (RTYPE)__msa_hsub_s_w((v8i16)in0, (v8i16)in0); \
    out1 = (RTYPE)__msa_hsub_s_w((v8i16)in1, (v8i16)in1); \
  }
#define HSUB_UH2_SW(...) HSUB_UH2(v4i32, __VA_ARGS__)

/* Description : Set element n input vector to GPR value
   Arguments   : Inputs - in0, in1, in2, in3
                 Output - out
                 Return Type - as per RTYPE
   Details     : Set element 0 in vector 'out' to value specified in 'in0'
*/
#define INSERT_D2(RTYPE, in0, in1, out)              \
  {                                                  \
    out = (RTYPE)__msa_insert_d((v2i64)out, 0, in0); \
    out = (RTYPE)__msa_insert_d((v2i64)out, 1, in1); \
  }
#define INSERT_D2_SB(...) INSERT_D2(v16i8, __VA_ARGS__)

/* Description : Interleave even byte elements from vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Even byte elements of 'in0' and 'in1' are interleaved
                 and written to 'out0'
*/
#define ILVEV_B2(RTYPE, in0, in1, in2, in3, out0, out1)  \
  {                                                      \
    out0 = (RTYPE)__msa_ilvev_b((v16i8)in1, (v16i8)in0); \
    out1 = (RTYPE)__msa_ilvev_b((v16i8)in3, (v16i8)in2); \
  }
#define ILVEV_B2_UB(...) ILVEV_B2(v16u8, __VA_ARGS__)
#define ILVEV_B2_SH(...) ILVEV_B2(v8i16, __VA_ARGS__)
#define ILVEV_B2_SD(...) ILVEV_B2(v2i64, __VA_ARGS__)

/* Description : Interleave even halfword elements from vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Even halfword elements of 'in0' and 'in1' are interleaved
                 and written to 'out0'
*/
#define ILVEV_H2(RTYPE, in0, in1, in2, in3, out0, out1)  \
  {                                                      \
    out0 = (RTYPE)__msa_ilvev_h((v8i16)in1, (v8i16)in0); \
    out1 = (RTYPE)__msa_ilvev_h((v8i16)in3, (v8i16)in2); \
  }
#define ILVEV_H2_UB(...) ILVEV_H2(v16u8, __VA_ARGS__)
#define ILVEV_H2_SH(...) ILVEV_H2(v8i16, __VA_ARGS__)

/* Description : Interleave even word elements from vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Even word elements of 'in0' and 'in1' are interleaved
                 and written to 'out0'
*/
#define ILVEV_W2(RTYPE, in0, in1, in2, in3, out0, out1)  \
  {                                                      \
    out0 = (RTYPE)__msa_ilvev_w((v4i32)in1, (v4i32)in0); \
    out1 = (RTYPE)__msa_ilvev_w((v4i32)in3, (v4i32)in2); \
  }
#define ILVEV_W2_SD(...) ILVEV_W2(v2i64, __VA_ARGS__)

/* Description : Interleave even double word elements from vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Even double word elements of 'in0' and 'in1' are interleaved
                 and written to 'out0'
*/
#define ILVEV_D2(RTYPE, in0, in1, in2, in3, out0, out1)  \
  {                                                      \
    out0 = (RTYPE)__msa_ilvev_d((v2i64)in1, (v2i64)in0); \
    out1 = (RTYPE)__msa_ilvev_d((v2i64)in3, (v2i64)in2); \
  }
#define ILVEV_D2_UB(...) ILVEV_D2(v16u8, __VA_ARGS__)

/* Description : Interleave left half of byte elements from vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Left half of byte elements of 'in0' and 'in1' are interleaved
                 and written to 'out0'.
*/
#define ILVL_B2(RTYPE, in0, in1, in2, in3, out0, out1)  \
  {                                                     \
    out0 = (RTYPE)__msa_ilvl_b((v16i8)in0, (v16i8)in1); \
    out1 = (RTYPE)__msa_ilvl_b((v16i8)in2, (v16i8)in3); \
  }
#define ILVL_B2_UB(...) ILVL_B2(v16u8, __VA_ARGS__)
#define ILVL_B2_SB(...) ILVL_B2(v16i8, __VA_ARGS__)
#define ILVL_B2_SH(...) ILVL_B2(v8i16, __VA_ARGS__)

#define ILVL_B4(RTYPE, in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, \
                out2, out3)                                                \
  {                                                                        \
    ILVL_B2(RTYPE, in0, in1, in2, in3, out0, out1);                        \
    ILVL_B2(RTYPE, in4, in5, in6, in7, out2, out3);                        \
  }
#define ILVL_B4_SB(...) ILVL_B4(v16i8, __VA_ARGS__)
#define ILVL_B4_SH(...) ILVL_B4(v8i16, __VA_ARGS__)

/* Description : Interleave left half of halfword elements from vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Left half of halfword elements of 'in0' and 'in1' are
                 interleaved and written to 'out0'.
*/
#define ILVL_H2(RTYPE, in0, in1, in2, in3, out0, out1)  \
  {                                                     \
    out0 = (RTYPE)__msa_ilvl_h((v8i16)in0, (v8i16)in1); \
    out1 = (RTYPE)__msa_ilvl_h((v8i16)in2, (v8i16)in3); \
  }
#define ILVL_H2_SH(...) ILVL_H2(v8i16, __VA_ARGS__)
#define ILVL_H2_SW(...) ILVL_H2(v4i32, __VA_ARGS__)

/* Description : Interleave left half of word elements from vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Left half of word elements of 'in0' and 'in1' are interleaved
                 and written to 'out0'.
*/
#define ILVL_W2(RTYPE, in0, in1, in2, in3, out0, out1)  \
  {                                                     \
    out0 = (RTYPE)__msa_ilvl_w((v4i32)in0, (v4i32)in1); \
    out1 = (RTYPE)__msa_ilvl_w((v4i32)in2, (v4i32)in3); \
  }
#define ILVL_W2_SH(...) ILVL_W2(v8i16, __VA_ARGS__)

/* Description : Interleave right half of byte elements from vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Right half of byte elements of 'in0' and 'in1' are interleaved
                 and written to out0.
*/
#define ILVR_B2(RTYPE, in0, in1, in2, in3, out0, out1)  \
  {                                                     \
    out0 = (RTYPE)__msa_ilvr_b((v16i8)in0, (v16i8)in1); \
    out1 = (RTYPE)__msa_ilvr_b((v16i8)in2, (v16i8)in3); \
  }
#define ILVR_B2_UB(...) ILVR_B2(v16u8, __VA_ARGS__)
#define ILVR_B2_SB(...) ILVR_B2(v16i8, __VA_ARGS__)
#define ILVR_B2_SH(...) ILVR_B2(v8i16, __VA_ARGS__)
#define ILVR_B2_SW(...) ILVR_B2(v4i32, __VA_ARGS__)

#define ILVR_B4(RTYPE, in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, \
                out2, out3)                                                \
  {                                                                        \
    ILVR_B2(RTYPE, in0, in1, in2, in3, out0, out1);                        \
    ILVR_B2(RTYPE, in4, in5, in6, in7, out2, out3);                        \
  }
#define ILVR_B4_UB(...) ILVR_B4(v16u8, __VA_ARGS__)
#define ILVR_B4_SB(...) ILVR_B4(v16i8, __VA_ARGS__)
#define ILVR_B4_UH(...) ILVR_B4(v8u16, __VA_ARGS__)
#define ILVR_B4_SH(...) ILVR_B4(v8i16, __VA_ARGS__)
#define ILVR_B4_SW(...) ILVR_B4(v4i32, __VA_ARGS__)

/* Description : Interleave right half of halfword elements from vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Right half of halfword elements of 'in0' and 'in1' are
                 interleaved and written to 'out0'.
*/
#define ILVR_H2(RTYPE, in0, in1, in2, in3, out0, out1)  \
  {                                                     \
    out0 = (RTYPE)__msa_ilvr_h((v8i16)in0, (v8i16)in1); \
    out1 = (RTYPE)__msa_ilvr_h((v8i16)in2, (v8i16)in3); \
  }
#define ILVR_H2_SH(...) ILVR_H2(v8i16, __VA_ARGS__)
#define ILVR_H2_SW(...) ILVR_H2(v4i32, __VA_ARGS__)

#define ILVR_H4(RTYPE, in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, \
                out2, out3)                                                \
  {                                                                        \
    ILVR_H2(RTYPE, in0, in1, in2, in3, out0, out1);                        \
    ILVR_H2(RTYPE, in4, in5, in6, in7, out2, out3);                        \
  }
#define ILVR_H4_SH(...) ILVR_H4(v8i16, __VA_ARGS__)
#define ILVR_H4_SW(...) ILVR_H4(v4i32, __VA_ARGS__)

#define ILVR_W2(RTYPE, in0, in1, in2, in3, out0, out1)  \
  {                                                     \
    out0 = (RTYPE)__msa_ilvr_w((v4i32)in0, (v4i32)in1); \
    out1 = (RTYPE)__msa_ilvr_w((v4i32)in2, (v4i32)in3); \
  }
#define ILVR_W2_SH(...) ILVR_W2(v8i16, __VA_ARGS__)

/* Description : Interleave right half of double word elements from vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Right half of double word elements of 'in0' and 'in1' are
                 interleaved and written to 'out0'.
*/
#define ILVR_D2(RTYPE, in0, in1, in2, in3, out0, out1)      \
  {                                                         \
    out0 = (RTYPE)__msa_ilvr_d((v2i64)(in0), (v2i64)(in1)); \
    out1 = (RTYPE)__msa_ilvr_d((v2i64)(in2), (v2i64)(in3)); \
  }
#define ILVR_D2_UB(...) ILVR_D2(v16u8, __VA_ARGS__)
#define ILVR_D2_SB(...) ILVR_D2(v16i8, __VA_ARGS__)
#define ILVR_D2_SH(...) ILVR_D2(v8i16, __VA_ARGS__)

#define ILVR_D4(RTYPE, in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, \
                out2, out3)                                                \
  {                                                                        \
    ILVR_D2(RTYPE, in0, in1, in2, in3, out0, out1);                        \
    ILVR_D2(RTYPE, in4, in5, in6, in7, out2, out3);                        \
  }
#define ILVR_D4_SB(...) ILVR_D4(v16i8, __VA_ARGS__)
#define ILVR_D4_UB(...) ILVR_D4(v16u8, __VA_ARGS__)

/* Description : Interleave both left and right half of input vectors
   Arguments   : Inputs  - in0, in1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Right half of byte elements from 'in0' and 'in1' are
                 interleaved and written to 'out0'
*/
#define ILVRL_B2(RTYPE, in0, in1, out0, out1)           \
  {                                                     \
    out0 = (RTYPE)__msa_ilvr_b((v16i8)in0, (v16i8)in1); \
    out1 = (RTYPE)__msa_ilvl_b((v16i8)in0, (v16i8)in1); \
  }
#define ILVRL_B2_UB(...) ILVRL_B2(v16u8, __VA_ARGS__)
#define ILVRL_B2_SB(...) ILVRL_B2(v16i8, __VA_ARGS__)
#define ILVRL_B2_UH(...) ILVRL_B2(v8u16, __VA_ARGS__)
#define ILVRL_B2_SH(...) ILVRL_B2(v8i16, __VA_ARGS__)

#define ILVRL_H2(RTYPE, in0, in1, out0, out1)           \
  {                                                     \
    out0 = (RTYPE)__msa_ilvr_h((v8i16)in0, (v8i16)in1); \
    out1 = (RTYPE)__msa_ilvl_h((v8i16)in0, (v8i16)in1); \
  }
#define ILVRL_H2_SH(...) ILVRL_H2(v8i16, __VA_ARGS__)
#define ILVRL_H2_SW(...) ILVRL_H2(v4i32, __VA_ARGS__)

#define ILVRL_W2(RTYPE, in0, in1, out0, out1)           \
  {                                                     \
    out0 = (RTYPE)__msa_ilvr_w((v4i32)in0, (v4i32)in1); \
    out1 = (RTYPE)__msa_ilvl_w((v4i32)in0, (v4i32)in1); \
  }
#define ILVRL_W2_UB(...) ILVRL_W2(v16u8, __VA_ARGS__)
#define ILVRL_W2_SH(...) ILVRL_W2(v8i16, __VA_ARGS__)
#define ILVRL_W2_SW(...) ILVRL_W2(v4i32, __VA_ARGS__)

/* Description : Maximum values between signed elements of vector and
                 5-bit signed immediate value are copied to the output vector
   Arguments   : Inputs  - in0, in1, in2, in3, max_val
                 Outputs - in place operation
                 Return Type - unsigned halfword
   Details     : Maximum of signed halfword element values from 'in0' and
                 'max_val' are written in place
*/
#define MAXI_SH2(RTYPE, in0, in1, max_val)              \
  {                                                     \
    in0 = (RTYPE)__msa_maxi_s_h((v8i16)in0, (max_val)); \
    in1 = (RTYPE)__msa_maxi_s_h((v8i16)in1, (max_val)); \
  }
#define MAXI_SH2_SH(...) MAXI_SH2(v8i16, __VA_ARGS__)

/* Description : Saturate the halfword element values to the max
                 unsigned value of (sat_val + 1) bits
                 The element data width remains unchanged
   Arguments   : Inputs  - in0, in1, sat_val
                 Outputs - in place operation
                 Return Type - as per RTYPE
   Details     : Each unsigned halfword element from 'in0' is saturated to the
                 value generated with (sat_val + 1) bit range.
                 The results are written in place
*/
#define SAT_UH2(RTYPE, in0, in1, sat_val)            \
  {                                                  \
    in0 = (RTYPE)__msa_sat_u_h((v8u16)in0, sat_val); \
    in1 = (RTYPE)__msa_sat_u_h((v8u16)in1, sat_val); \
  }
#define SAT_UH2_SH(...) SAT_UH2(v8i16, __VA_ARGS__)

/* Description : Saturate the halfword element values to the max
                 unsigned value of (sat_val + 1) bits
                 The element data width remains unchanged
   Arguments   : Inputs  - in0, in1, sat_val
                 Outputs - in place operation
                 Return Type - as per RTYPE
   Details     : Each unsigned halfword element from 'in0' is saturated to the
                 value generated with (sat_val + 1) bit range
                 The results are written in place
*/
#define SAT_SH2(RTYPE, in0, in1, sat_val)            \
  {                                                  \
    in0 = (RTYPE)__msa_sat_s_h((v8i16)in0, sat_val); \
    in1 = (RTYPE)__msa_sat_s_h((v8i16)in1, sat_val); \
  }
#define SAT_SH2_SH(...) SAT_SH2(v8i16, __VA_ARGS__)

#define SAT_SH4(RTYPE, in0, in1, in2, in3, sat_val) \
  {                                                 \
    SAT_SH2(RTYPE, in0, in1, sat_val);              \
    SAT_SH2(RTYPE, in2, in3, sat_val);              \
  }
#define SAT_SH4_SH(...) SAT_SH4(v8i16, __VA_ARGS__)

/* Description : Indexed halfword element values are replicated to all
                 elements in output vector
   Arguments   : Inputs  - in, idx0, idx1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : 'idx0' element value from 'in' vector is replicated to all
                  elements in 'out0' vector
                  Valid index range for halfword operation is 0-7
*/
#define SPLATI_H2(RTYPE, in, idx0, idx1, out0, out1) \
  {                                                  \
    out0 = (RTYPE)__msa_splati_h((v8i16)in, idx0);   \
    out1 = (RTYPE)__msa_splati_h((v8i16)in, idx1);   \
  }
#define SPLATI_H2_SB(...) SPLATI_H2(v16i8, __VA_ARGS__)
#define SPLATI_H2_SH(...) SPLATI_H2(v8i16, __VA_ARGS__)

#define SPLATI_H3(RTYPE, in, idx0, idx1, idx2, out0, out1, out2) \
  {                                                              \
    SPLATI_H2(RTYPE, in, idx0, idx1, out0, out1);                \
    out2 = (RTYPE)__msa_splati_h((v8i16)in, idx2);               \
  }
#define SPLATI_H3_SB(...) SPLATI_H3(v16i8, __VA_ARGS__)
#define SPLATI_H3_SH(...) SPLATI_H3(v8i16, __VA_ARGS__)

/* Description : Indexed word element values are replicated to all
                 elements in output vector
   Arguments   : Inputs  - in, stidx
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : 'stidx' element value from 'in' vector is replicated to all
                 elements in 'out0' vector
                 'stidx + 1' element value from 'in' vector is replicated to all
                 elements in 'out1' vector
                 Valid index range for word operation is 0-3
*/
#define SPLATI_W2(RTYPE, in, stidx, out0, out1)           \
  {                                                       \
    out0 = (RTYPE)__msa_splati_w((v4i32)in, stidx);       \
    out1 = (RTYPE)__msa_splati_w((v4i32)in, (stidx + 1)); \
  }
#define SPLATI_W2_SW(...) SPLATI_W2(v4i32, __VA_ARGS__)

/* Description : Pack even byte elements of vector pairs
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Even byte elements of 'in0' are copied to the left half of
                 'out0' & even byte elements of 'in1' are copied to the right
                 half of 'out0'.
*/
#define PCKEV_B2(RTYPE, in0, in1, in2, in3, out0, out1)  \
  {                                                      \
    out0 = (RTYPE)__msa_pckev_b((v16i8)in0, (v16i8)in1); \
    out1 = (RTYPE)__msa_pckev_b((v16i8)in2, (v16i8)in3); \
  }
#define PCKEV_B2_SB(...) PCKEV_B2(v16i8, __VA_ARGS__)
#define PCKEV_B2_UB(...) PCKEV_B2(v16u8, __VA_ARGS__)
#define PCKEV_B2_SH(...) PCKEV_B2(v8i16, __VA_ARGS__)
#define PCKEV_B2_SW(...) PCKEV_B2(v4i32, __VA_ARGS__)

#define PCKEV_B4(RTYPE, in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, \
                 out2, out3)                                                \
  {                                                                         \
    PCKEV_B2(RTYPE, in0, in1, in2, in3, out0, out1);                        \
    PCKEV_B2(RTYPE, in4, in5, in6, in7, out2, out3);                        \
  }
#define PCKEV_B4_SB(...) PCKEV_B4(v16i8, __VA_ARGS__)
#define PCKEV_B4_UB(...) PCKEV_B4(v16u8, __VA_ARGS__)
#define PCKEV_B4_SH(...) PCKEV_B4(v8i16, __VA_ARGS__)

/* Description : Pack even halfword elements of vector pairs
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Even halfword elements of 'in0' are copied to the left half of
                 'out0' & even halfword elements of 'in1' are copied to the
                 right half of 'out0'.
*/
#define PCKEV_H2(RTYPE, in0, in1, in2, in3, out0, out1)  \
  {                                                      \
    out0 = (RTYPE)__msa_pckev_h((v8i16)in0, (v8i16)in1); \
    out1 = (RTYPE)__msa_pckev_h((v8i16)in2, (v8i16)in3); \
  }
#define PCKEV_H2_SH(...) PCKEV_H2(v8i16, __VA_ARGS__)

#define PCKEV_H4(RTYPE, in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, \
                 out2, out3)                                                \
  {                                                                         \
    PCKEV_H2(RTYPE, in0, in1, in2, in3, out0, out1);                        \
    PCKEV_H2(RTYPE, in4, in5, in6, in7, out2, out3);                        \
  }
#define PCKEV_H4_SH(...) PCKEV_H4(v8i16, __VA_ARGS__)

/* Description : Pack even double word elements of vector pairs
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Even double elements of 'in0' are copied to the left half of
                 'out0' & even double elements of 'in1' are copied to the right
                 half of 'out0'.
*/
#define PCKEV_D2(RTYPE, in0, in1, in2, in3, out0, out1)  \
  {                                                      \
    out0 = (RTYPE)__msa_pckev_d((v2i64)in0, (v2i64)in1); \
    out1 = (RTYPE)__msa_pckev_d((v2i64)in2, (v2i64)in3); \
  }
#define PCKEV_D2_UB(...) PCKEV_D2(v16u8, __VA_ARGS__)
#define PCKEV_D2_SH(...) PCKEV_D2(v8i16, __VA_ARGS__)

/* Description : Pack odd double word elements of vector pairs
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Odd double word elements of 'in0' are copied to the left half
                 of 'out0' & odd double word elements of 'in1' are copied to
                 the right half of 'out0'.
*/
#define PCKOD_D2(RTYPE, in0, in1, in2, in3, out0, out1)  \
  {                                                      \
    out0 = (RTYPE)__msa_pckod_d((v2i64)in0, (v2i64)in1); \
    out1 = (RTYPE)__msa_pckod_d((v2i64)in2, (v2i64)in3); \
  }
#define PCKOD_D2_UB(...) PCKOD_D2(v16u8, __VA_ARGS__)
#define PCKOD_D2_SH(...) PCKOD_D2(v8i16, __VA_ARGS__)

/* Description : Each byte element is logically xor'ed with immediate 128
   Arguments   : Inputs  - in0, in1
                 Outputs - in place operation
                 Return Type - as per RTYPE
   Details     : Each unsigned byte element from input vector 'in0' is
                 logically xor'ed with 128 and the result is stored in-place.
*/
#define XORI_B2_128(RTYPE, in0, in1)            \
  {                                             \
    in0 = (RTYPE)__msa_xori_b((v16u8)in0, 128); \
    in1 = (RTYPE)__msa_xori_b((v16u8)in1, 128); \
  }
#define XORI_B2_128_UB(...) XORI_B2_128(v16u8, __VA_ARGS__)
#define XORI_B2_128_SB(...) XORI_B2_128(v16i8, __VA_ARGS__)

#define XORI_B3_128(RTYPE, in0, in1, in2)       \
  {                                             \
    XORI_B2_128(RTYPE, in0, in1);               \
    in2 = (RTYPE)__msa_xori_b((v16u8)in2, 128); \
  }
#define XORI_B3_128_SB(...) XORI_B3_128(v16i8, __VA_ARGS__)

#define XORI_B4_128(RTYPE, in0, in1, in2, in3) \
  {                                            \
    XORI_B2_128(RTYPE, in0, in1);              \
    XORI_B2_128(RTYPE, in2, in3);              \
  }
#define XORI_B4_128_UB(...) XORI_B4_128(v16u8, __VA_ARGS__)
#define XORI_B4_128_SB(...) XORI_B4_128(v16i8, __VA_ARGS__)

#define XORI_B5_128(RTYPE, in0, in1, in2, in3, in4) \
  {                                                 \
    XORI_B3_128(RTYPE, in0, in1, in2);              \
    XORI_B2_128(RTYPE, in3, in4);                   \
  }
#define XORI_B5_128_SB(...) XORI_B5_128(v16i8, __VA_ARGS__)

#define XORI_B8_128(RTYPE, in0, in1, in2, in3, in4, in5, in6, in7) \
  {                                                                \
    XORI_B4_128(RTYPE, in0, in1, in2, in3);                        \
    XORI_B4_128(RTYPE, in4, in5, in6, in7);                        \
  }
#define XORI_B8_128_SB(...) XORI_B8_128(v16i8, __VA_ARGS__)

/* Description : Shift left all elements of vector (generic for all data types)
   Arguments   : Inputs  - in0, in1, in2, in3, shift
                 Outputs - in place operation
                 Return Type - as per input vector RTYPE
   Details     : Each element of vector 'in0' is left shifted by 'shift' and
                 the result is written in-place.
*/
#define SLLI_4V(in0, in1, in2, in3, shift) \
  {                                        \
    in0 = in0 << shift;                    \
    in1 = in1 << shift;                    \
    in2 = in2 << shift;                    \
    in3 = in3 << shift;                    \
  }

/* Description : Arithmetic shift right all elements of vector
                 (generic for all data types)
   Arguments   : Inputs  - in0, in1, in2, in3, shift
                 Outputs - in place operation
                 Return Type - as per input vector RTYPE
   Details     : Each element of vector 'in0' is right shifted by 'shift' and
                 the result is written in-place. 'shift' is a GP variable.
*/
#define SRA_4V(in0, in1, in2, in3, shift) \
  {                                       \
    in0 = in0 >> shift;                   \
    in1 = in1 >> shift;                   \
    in2 = in2 >> shift;                   \
    in3 = in3 >> shift;                   \
  }

/* Description : Shift right arithmetic rounded words
   Arguments   : Inputs  - in0, in1, shift
                 Outputs - in place operation
                 Return Type - as per RTYPE
   Details     : Each element of vector 'in0' is shifted right arithmetically by
                 the number of bits in the corresponding element in the vector
                 'shift'. The last discarded bit is added to shifted value for
                 rounding and the result is written in-place.
                 'shift' is a vector.
*/
#define SRAR_W2(RTYPE, in0, in1, shift)                  \
  {                                                      \
    in0 = (RTYPE)__msa_srar_w((v4i32)in0, (v4i32)shift); \
    in1 = (RTYPE)__msa_srar_w((v4i32)in1, (v4i32)shift); \
  }

#define SRAR_W4(RTYPE, in0, in1, in2, in3, shift) \
  {                                               \
    SRAR_W2(RTYPE, in0, in1, shift);              \
    SRAR_W2(RTYPE, in2, in3, shift);              \
  }
#define SRAR_W4_SW(...) SRAR_W4(v4i32, __VA_ARGS__)

/* Description : Shift right arithmetic rounded (immediate)
   Arguments   : Inputs  - in0, in1, shift
                 Outputs - in place operation
                 Return Type - as per RTYPE
   Details     : Each element of vector 'in0' is shifted right arithmetically by
                 the value in 'shift'. The last discarded bit is added to the
                 shifted value for rounding and the result is written in-place.
                 'shift' is an immediate value.
*/
#define SRARI_H2(RTYPE, in0, in1, shift)           \
  {                                                \
    in0 = (RTYPE)__msa_srari_h((v8i16)in0, shift); \
    in1 = (RTYPE)__msa_srari_h((v8i16)in1, shift); \
  }
#define SRARI_H2_UH(...) SRARI_H2(v8u16, __VA_ARGS__)
#define SRARI_H2_SH(...) SRARI_H2(v8i16, __VA_ARGS__)

#define SRARI_H4(RTYPE, in0, in1, in2, in3, shift) \
  {                                                \
    SRARI_H2(RTYPE, in0, in1, shift);              \
    SRARI_H2(RTYPE, in2, in3, shift);              \
  }
#define SRARI_H4_UH(...) SRARI_H4(v8u16, __VA_ARGS__)
#define SRARI_H4_SH(...) SRARI_H4(v8i16, __VA_ARGS__)

#define SRARI_W2(RTYPE, in0, in1, shift)           \
  {                                                \
    in0 = (RTYPE)__msa_srari_w((v4i32)in0, shift); \
    in1 = (RTYPE)__msa_srari_w((v4i32)in1, shift); \
  }

#define SRARI_W4(RTYPE, in0, in1, in2, in3, shift) \
  {                                                \
    SRARI_W2(RTYPE, in0, in1, shift);              \
    SRARI_W2(RTYPE, in2, in3, shift);              \
  }
#define SRARI_W4_SW(...) SRARI_W4(v4i32, __VA_ARGS__)

/* Description : Multiplication of pairs of vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
   Details     : Each element from 'in0' is multiplied with elements from 'in1'
                 and the result is written to 'out0'
*/
#define MUL2(in0, in1, in2, in3, out0, out1) \
  {                                          \
    out0 = in0 * in1;                        \
    out1 = in2 * in3;                        \
  }
#define MUL4(in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, out2, out3) \
  {                                                                          \
    MUL2(in0, in1, in2, in3, out0, out1);                                    \
    MUL2(in4, in5, in6, in7, out2, out3);                                    \
  }

/* Description : Addition of 2 pairs of vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
   Details     : Each element in 'in0' is added to 'in1' and result is written
                 to 'out0'.
*/
#define ADD2(in0, in1, in2, in3, out0, out1) \
  {                                          \
    out0 = in0 + in1;                        \
    out1 = in2 + in3;                        \
  }
#define ADD4(in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, out2, out3) \
  {                                                                          \
    ADD2(in0, in1, in2, in3, out0, out1);                                    \
    ADD2(in4, in5, in6, in7, out2, out3);                                    \
  }

/* Description : Subtraction of 2 pairs of vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1
   Details     : Each element in 'in1' is subtracted from 'in0' and result is
                 written to 'out0'.
*/
#define SUB2(in0, in1, in2, in3, out0, out1) \
  {                                          \
    out0 = in0 - in1;                        \
    out1 = in2 - in3;                        \
  }
#define SUB4(in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, out2, out3) \
  {                                                                          \
    out0 = in0 - in1;                                                        \
    out1 = in2 - in3;                                                        \
    out2 = in4 - in5;                                                        \
    out3 = in6 - in7;                                                        \
  }

/* Description : Sign extend halfword elements from right half of the vector
   Arguments   : Input  - in    (halfword vector)
                 Output - out   (sign extended word vector)
                 Return Type - signed word
   Details     : Sign bit of halfword elements from input vector 'in' is
                 extracted and interleaved with same vector 'in0' to generate
                 4 word elements keeping sign intact
*/
#define UNPCK_R_SH_SW(in, out)                    \
  {                                               \
    v8i16 sign_m;                                 \
                                                  \
    sign_m = __msa_clti_s_h((v8i16)in, 0);        \
    out = (v4i32)__msa_ilvr_h(sign_m, (v8i16)in); \
  }

/* Description : Zero extend unsigned byte elements to halfword elements
   Arguments   : Input   - in          (unsigned byte vector)
                 Outputs - out0, out1  (unsigned  halfword vectors)
                 Return Type - signed halfword
   Details     : Zero extended right half of vector is returned in 'out0'
                 Zero extended left half of vector is returned in 'out1'
*/
#define UNPCK_UB_SH(in, out0, out1)      \
  {                                      \
    v16i8 zero_m = { 0 };                \
                                         \
    ILVRL_B2_SH(zero_m, in, out0, out1); \
  }

/* Description : Sign extend halfword elements from input vector and return
                 the result in pair of vectors
   Arguments   : Input   - in            (halfword vector)
                 Outputs - out0, out1   (sign extended word vectors)
                 Return Type - signed word
   Details     : Sign bit of halfword elements from input vector 'in' is
                 extracted and interleaved right with same vector 'in0' to
                 generate 4 signed word elements in 'out0'
                 Then interleaved left with same vector 'in0' to
                 generate 4 signed word elements in 'out1'
*/
#define UNPCK_SH_SW(in, out0, out1)       \
  {                                       \
    v8i16 tmp_m;                          \
                                          \
    tmp_m = __msa_clti_s_h((v8i16)in, 0); \
    ILVRL_H2_SW(tmp_m, in, out0, out1);   \
  }

/* Description : Butterfly of 4 input vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1, out2, out3
   Details     : Butterfly operation
*/
#define BUTTERFLY_4(in0, in1, in2, in3, out0, out1, out2, out3) \
  {                                                             \
    out0 = in0 + in3;                                           \
    out1 = in1 + in2;                                           \
                                                                \
    out2 = in1 - in2;                                           \
    out3 = in0 - in3;                                           \
  }

/* Description : Transpose input 8x8 byte block
   Arguments   : Inputs  - in0, in1, in2, in3, in4, in5, in6, in7
                 Outputs - out0, out1, out2, out3, out4, out5, out6, out7
                 Return Type - as per RTYPE
*/
#define TRANSPOSE8x8_UB(RTYPE, in0, in1, in2, in3, in4, in5, in6, in7, out0,   \
                        out1, out2, out3, out4, out5, out6, out7)              \
  {                                                                            \
    v16i8 tmp0_m, tmp1_m, tmp2_m, tmp3_m;                                      \
    v16i8 tmp4_m, tmp5_m, tmp6_m, tmp7_m;                                      \
                                                                               \
    ILVR_B4_SB(in2, in0, in3, in1, in6, in4, in7, in5, tmp0_m, tmp1_m, tmp2_m, \
               tmp3_m);                                                        \
    ILVRL_B2_SB(tmp1_m, tmp0_m, tmp4_m, tmp5_m);                               \
    ILVRL_B2_SB(tmp3_m, tmp2_m, tmp6_m, tmp7_m);                               \
    ILVRL_W2(RTYPE, tmp6_m, tmp4_m, out0, out2);                               \
    ILVRL_W2(RTYPE, tmp7_m, tmp5_m, out4, out6);                               \
    SLDI_B2_0(RTYPE, out0, out2, out1, out3, 8);                               \
    SLDI_B2_0(RTYPE, out4, out6, out5, out7, 8);                               \
  }
#define TRANSPOSE8x8_UB_UB(...) TRANSPOSE8x8_UB(v16u8, __VA_ARGS__)

/* Description : Transpose 16x4 block into 4x16 with byte elements in vectors
   Arguments   : Inputs  - in0, in1, in2, in3, in4, in5, in6, in7,
                           in8, in9, in10, in11, in12, in13, in14, in15
                 Outputs - out0, out1, out2, out3
                 Return Type - unsigned byte
*/
#define TRANSPOSE16x4_UB_UB(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, \
                            in10, in11, in12, in13, in14, in15, out0, out1,   \
                            out2, out3)                                       \
  {                                                                           \
    v2i64 tmp0_m, tmp1_m, tmp2_m, tmp3_m;                                     \
                                                                              \
    ILVEV_W2_SD(in0, in4, in8, in12, tmp0_m, tmp1_m);                         \
    out1 = (v16u8)__msa_ilvev_d(tmp1_m, tmp0_m);                              \
                                                                              \
    ILVEV_W2_SD(in1, in5, in9, in13, tmp0_m, tmp1_m);                         \
    out3 = (v16u8)__msa_ilvev_d(tmp1_m, tmp0_m);                              \
                                                                              \
    ILVEV_W2_SD(in2, in6, in10, in14, tmp0_m, tmp1_m);                        \
                                                                              \
    tmp2_m = __msa_ilvev_d(tmp1_m, tmp0_m);                                   \
    ILVEV_W2_SD(in3, in7, in11, in15, tmp0_m, tmp1_m);                        \
                                                                              \
    tmp3_m = __msa_ilvev_d(tmp1_m, tmp0_m);                                   \
    ILVEV_B2_SD(out1, out3, tmp2_m, tmp3_m, tmp0_m, tmp1_m);                  \
    out0 = (v16u8)__msa_ilvev_h((v8i16)tmp1_m, (v8i16)tmp0_m);                \
    out2 = (v16u8)__msa_ilvod_h((v8i16)tmp1_m, (v8i16)tmp0_m);                \
                                                                              \
    tmp0_m = (v2i64)__msa_ilvod_b((v16i8)out3, (v16i8)out1);                  \
    tmp1_m = (v2i64)__msa_ilvod_b((v16i8)tmp3_m, (v16i8)tmp2_m);              \
    out1 = (v16u8)__msa_ilvev_h((v8i16)tmp1_m, (v8i16)tmp0_m);                \
    out3 = (v16u8)__msa_ilvod_h((v8i16)tmp1_m, (v8i16)tmp0_m);                \
  }

/* Description : Transpose 16x8 block into 8x16 with byte elements in vectors
   Arguments   : Inputs  - in0, in1, in2, in3, in4, in5, in6, in7,
                           in8, in9, in10, in11, in12, in13, in14, in15
                 Outputs - out0, out1, out2, out3, out4, out5, out6, out7
                 Return Type - unsigned byte
*/
#define TRANSPOSE16x8_UB_UB(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, \
                            in10, in11, in12, in13, in14, in15, out0, out1,   \
                            out2, out3, out4, out5, out6, out7)               \
  {                                                                           \
    v16u8 tmp0_m, tmp1_m, tmp2_m, tmp3_m;                                     \
    v16u8 tmp4_m, tmp5_m, tmp6_m, tmp7_m;                                     \
                                                                              \
    ILVEV_D2_UB(in0, in8, in1, in9, out7, out6);                              \
    ILVEV_D2_UB(in2, in10, in3, in11, out5, out4);                            \
    ILVEV_D2_UB(in4, in12, in5, in13, out3, out2);                            \
    ILVEV_D2_UB(in6, in14, in7, in15, out1, out0);                            \
                                                                              \
    tmp0_m = (v16u8)__msa_ilvev_b((v16i8)out6, (v16i8)out7);                  \
    tmp4_m = (v16u8)__msa_ilvod_b((v16i8)out6, (v16i8)out7);                  \
    tmp1_m = (v16u8)__msa_ilvev_b((v16i8)out4, (v16i8)out5);                  \
    tmp5_m = (v16u8)__msa_ilvod_b((v16i8)out4, (v16i8)out5);                  \
    out5 = (v16u8)__msa_ilvev_b((v16i8)out2, (v16i8)out3);                    \
    tmp6_m = (v16u8)__msa_ilvod_b((v16i8)out2, (v16i8)out3);                  \
    out7 = (v16u8)__msa_ilvev_b((v16i8)out0, (v16i8)out1);                    \
    tmp7_m = (v16u8)__msa_ilvod_b((v16i8)out0, (v16i8)out1);                  \
                                                                              \
    ILVEV_H2_UB(tmp0_m, tmp1_m, out5, out7, tmp2_m, tmp3_m);                  \
    out0 = (v16u8)__msa_ilvev_w((v4i32)tmp3_m, (v4i32)tmp2_m);                \
    out4 = (v16u8)__msa_ilvod_w((v4i32)tmp3_m, (v4i32)tmp2_m);                \
                                                                              \
    tmp2_m = (v16u8)__msa_ilvod_h((v8i16)tmp1_m, (v8i16)tmp0_m);              \
    tmp3_m = (v16u8)__msa_ilvod_h((v8i16)out7, (v8i16)out5);                  \
    out2 = (v16u8)__msa_ilvev_w((v4i32)tmp3_m, (v4i32)tmp2_m);                \
    out6 = (v16u8)__msa_ilvod_w((v4i32)tmp3_m, (v4i32)tmp2_m);                \
                                                                              \
    ILVEV_H2_UB(tmp4_m, tmp5_m, tmp6_m, tmp7_m, tmp2_m, tmp3_m);              \
    out1 = (v16u8)__msa_ilvev_w((v4i32)tmp3_m, (v4i32)tmp2_m);                \
    out5 = (v16u8)__msa_ilvod_w((v4i32)tmp3_m, (v4i32)tmp2_m);                \
                                                                              \
    tmp2_m = (v16u8)__msa_ilvod_h((v8i16)tmp5_m, (v8i16)tmp4_m);              \
    tmp2_m = (v16u8)__msa_ilvod_h((v8i16)tmp5_m, (v8i16)tmp4_m);              \
    tmp3_m = (v16u8)__msa_ilvod_h((v8i16)tmp7_m, (v8i16)tmp6_m);              \
    tmp3_m = (v16u8)__msa_ilvod_h((v8i16)tmp7_m, (v8i16)tmp6_m);              \
    out3 = (v16u8)__msa_ilvev_w((v4i32)tmp3_m, (v4i32)tmp2_m);                \
    out7 = (v16u8)__msa_ilvod_w((v4i32)tmp3_m, (v4i32)tmp2_m);                \
  }

/* Description : Transpose 4x4 block with half word elements in vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1, out2, out3
                 Return Type - signed halfword
*/
#define TRANSPOSE4x4_SH_SH(in0, in1, in2, in3, out0, out1, out2, out3) \
  {                                                                    \
    v8i16 s0_m, s1_m;                                                  \
                                                                       \
    ILVR_H2_SH(in1, in0, in3, in2, s0_m, s1_m);                        \
    ILVRL_W2_SH(s1_m, s0_m, out0, out2);                               \
    out1 = (v8i16)__msa_ilvl_d((v2i64)out0, (v2i64)out0);              \
    out3 = (v8i16)__msa_ilvl_d((v2i64)out0, (v2i64)out2);              \
  }

/* Description : Transpose 8x4 block with half word elements in vectors
   Arguments   : Inputs  - in0, in1, in2, in3, in4, in5, in6, in7
                 Outputs - out0, out1, out2, out3, out4, out5, out6, out7
                 Return Type - signed halfword
*/
#define TRANSPOSE8X4_SH_SH(in0, in1, in2, in3, out0, out1, out2, out3) \
  {                                                                    \
    v8i16 tmp0_m, tmp1_m, tmp2_m, tmp3_m;                              \
                                                                       \
    ILVR_H2_SH(in1, in0, in3, in2, tmp0_m, tmp1_m);                    \
    ILVL_H2_SH(in1, in0, in3, in2, tmp2_m, tmp3_m);                    \
    ILVR_W2_SH(tmp1_m, tmp0_m, tmp3_m, tmp2_m, out0, out2);            \
    ILVL_W2_SH(tmp1_m, tmp0_m, tmp3_m, tmp2_m, out1, out3);            \
  }

/* Description : Transpose 4x4 block with word elements in vectors
   Arguments   : Inputs  - in0, in1, in2, in3
                 Outputs - out0, out1, out2, out3
                 Return Type - signed word
*/
#define TRANSPOSE4x4_SW_SW(in0, in1, in2, in3, out0, out1, out2, out3) \
  {                                                                    \
    v4i32 s0_m, s1_m, s2_m, s3_m;                                      \
                                                                       \
    ILVRL_W2_SW(in1, in0, s0_m, s1_m);                                 \
    ILVRL_W2_SW(in3, in2, s2_m, s3_m);                                 \
                                                                       \
    out0 = (v4i32)__msa_ilvr_d((v2i64)s2_m, (v2i64)s0_m);              \
    out1 = (v4i32)__msa_ilvl_d((v2i64)s2_m, (v2i64)s0_m);              \
    out2 = (v4i32)__msa_ilvr_d((v2i64)s3_m, (v2i64)s1_m);              \
    out3 = (v4i32)__msa_ilvl_d((v2i64)s3_m, (v2i64)s1_m);              \
  }

/* Description : Dot product and addition of 3 signed halfword input vectors
   Arguments   : Inputs - in0, in1, in2, coeff0, coeff1, coeff2
                 Output - out0_m
                 Return Type - signed halfword
   Details     : Dot product of 'in0' with 'coeff0'
                 Dot product of 'in1' with 'coeff1'
                 Dot product of 'in2' with 'coeff2'
                 Addition of all the 3 vector results
                 out0_m = (in0 * coeff0) + (in1 * coeff1) + (in2 * coeff2)
*/
#define DPADD_SH3_SH(in0, in1, in2, coeff0, coeff1, coeff2)      \
  ({                                                             \
    v8i16 tmp1_m;                                                \
    v8i16 out0_m;                                                \
                                                                 \
    out0_m = __msa_dotp_s_h((v16i8)in0, (v16i8)coeff0);          \
    out0_m = __msa_dpadd_s_h(out0_m, (v16i8)in1, (v16i8)coeff1); \
    tmp1_m = __msa_dotp_s_h((v16i8)in2, (v16i8)coeff2);          \
    out0_m = __msa_adds_s_h(out0_m, tmp1_m);                     \
                                                                 \
    out0_m;                                                      \
  })

/* Description : Pack even elements of input vectors & xor with 128
   Arguments   : Inputs - in0, in1
                 Output - out_m
                 Return Type - unsigned byte
   Details     : Signed byte even elements from 'in0' and 'in1' are packed
                 together in one vector and the resulting vector is xor'ed with
                 128 to shift the range from signed to unsigned byte
*/
#define PCKEV_XORI128_UB(in0, in1)                        \
  ({                                                      \
    v16u8 out_m;                                          \
    out_m = (v16u8)__msa_pckev_b((v16i8)in1, (v16i8)in0); \
    out_m = (v16u8)__msa_xori_b((v16u8)out_m, 128);       \
    out_m;                                                \
  })

/* Description : Pack even byte elements and store byte vector in destination
                 memory
   Arguments   : Inputs - in0, in1, pdst
*/
#define PCKEV_ST_SB(in0, in1, pdst)                \
  {                                                \
    v16i8 tmp_m;                                   \
    tmp_m = __msa_pckev_b((v16i8)in1, (v16i8)in0); \
    ST_SB(tmp_m, (pdst));                          \
  }

/* Description : Horizontal 2 tap filter kernel code
   Arguments   : Inputs - in0, in1, mask, coeff, shift
*/
#define HORIZ_2TAP_FILT_UH(in0, in1, mask, coeff, shift)        \
  ({                                                            \
    v16i8 tmp0_m;                                               \
    v8u16 tmp1_m;                                               \
                                                                \
    tmp0_m = __msa_vshf_b((v16i8)mask, (v16i8)in1, (v16i8)in0); \
    tmp1_m = __msa_dotp_u_h((v16u8)tmp0_m, (v16u8)coeff);       \
    tmp1_m = (v8u16)__msa_srari_h((v8i16)tmp1_m, shift);        \
                                                                \
    tmp1_m;                                                     \
  })
#endif  // VPX_VP8_COMMON_MIPS_MSA_VP8_MACROS_MSA_H_
