// Copyright 2016 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// MSA common macros
//
// Author(s):  Prashant Patil   (prashant.patil@imgtec.com)

#ifndef WEBP_DSP_MSA_MACRO_H_
#define WEBP_DSP_MSA_MACRO_H_

#include <stdint.h>
#include <msa.h>

#if defined(__clang__)
  #define CLANG_BUILD
#endif

#ifdef CLANG_BUILD
  #define ADDVI_H(a, b)  __msa_addvi_h((v8i16)a, b)
  #define ADDVI_W(a, b)  __msa_addvi_w((v4i32)a, b)
  #define SRAI_B(a, b)  __msa_srai_b((v16i8)a, b)
  #define SRAI_H(a, b)  __msa_srai_h((v8i16)a, b)
  #define SRAI_W(a, b)  __msa_srai_w((v4i32)a, b)
  #define SRLI_H(a, b)  __msa_srli_h((v8i16)a, b)
  #define SLLI_B(a, b)  __msa_slli_b((v4i32)a, b)
  #define ANDI_B(a, b)  __msa_andi_b((v16u8)a, b)
  #define ORI_B(a, b)   __msa_ori_b((v16u8)a, b)
#else
  #define ADDVI_H(a, b)  (a + b)
  #define ADDVI_W(a, b)  (a + b)
  #define SRAI_B(a, b)  (a >> b)
  #define SRAI_H(a, b)  (a >> b)
  #define SRAI_W(a, b)  (a >> b)
  #define SRLI_H(a, b)  (a << b)
  #define SLLI_B(a, b)  (a << b)
  #define ANDI_B(a, b)  (a & b)
  #define ORI_B(a, b)   (a | b)
#endif

#define LD_B(RTYPE, psrc) *((RTYPE*)(psrc))
#define LD_UB(...) LD_B(v16u8, __VA_ARGS__)
#define LD_SB(...) LD_B(v16i8, __VA_ARGS__)

#define LD_H(RTYPE, psrc) *((RTYPE*)(psrc))
#define LD_UH(...) LD_H(v8u16, __VA_ARGS__)
#define LD_SH(...) LD_H(v8i16, __VA_ARGS__)

#define LD_W(RTYPE, psrc) *((RTYPE*)(psrc))
#define LD_UW(...) LD_W(v4u32, __VA_ARGS__)
#define LD_SW(...) LD_W(v4i32, __VA_ARGS__)

#define ST_B(RTYPE, in, pdst) *((RTYPE*)(pdst)) = in
#define ST_UB(...) ST_B(v16u8, __VA_ARGS__)
#define ST_SB(...) ST_B(v16i8, __VA_ARGS__)

#define ST_H(RTYPE, in, pdst) *((RTYPE*)(pdst)) = in
#define ST_UH(...) ST_H(v8u16, __VA_ARGS__)
#define ST_SH(...) ST_H(v8i16, __VA_ARGS__)

#define ST_W(RTYPE, in, pdst) *((RTYPE*)(pdst)) = in
#define ST_UW(...) ST_W(v4u32, __VA_ARGS__)
#define ST_SW(...) ST_W(v4i32, __VA_ARGS__)

#define MSA_LOAD_FUNC(TYPE, INSTR, FUNC_NAME)             \
  static inline TYPE FUNC_NAME(const void* const psrc) {  \
    const uint8_t* const psrc_m = (const uint8_t*)psrc;   \
    TYPE val_m;                                           \
    asm volatile (                                        \
      "" #INSTR " %[val_m], %[psrc_m]  \n\t"              \
      : [val_m] "=r" (val_m)                              \
      : [psrc_m] "m" (*psrc_m));                          \
    return val_m;                                         \
  }

#define MSA_LOAD(psrc, FUNC_NAME)  FUNC_NAME(psrc)

#define MSA_STORE_FUNC(TYPE, INSTR, FUNC_NAME)               \
  static inline void FUNC_NAME(TYPE val, void* const pdst) { \
    uint8_t* const pdst_m = (uint8_t*)pdst;                  \
    TYPE val_m = val;                                        \
    asm volatile (                                           \
      " " #INSTR "  %[val_m],  %[pdst_m]  \n\t"              \
      : [pdst_m] "=m" (*pdst_m)                              \
      : [val_m] "r" (val_m));                                \
  }

#define MSA_STORE(val, pdst, FUNC_NAME)  FUNC_NAME(val, pdst)

#if (__mips_isa_rev >= 6)
  MSA_LOAD_FUNC(uint16_t, lh, msa_lh);
  #define LH(psrc)  MSA_LOAD(psrc, msa_lh)
  MSA_LOAD_FUNC(uint32_t, lw, msa_lw);
  #define LW(psrc)  MSA_LOAD(psrc, msa_lw)
  #if (__mips == 64)
    MSA_LOAD_FUNC(uint64_t, ld, msa_ld);
    #define LD(psrc)  MSA_LOAD(psrc, msa_ld)
  #else  // !(__mips == 64)
    #define LD(psrc)  ((((uint64_t)MSA_LOAD(psrc + 4, msa_lw)) << 32) | \
                       MSA_LOAD(psrc, msa_lw))
  #endif  // (__mips == 64)

  MSA_STORE_FUNC(uint16_t, sh, msa_sh);
  #define SH(val, pdst)  MSA_STORE(val, pdst, msa_sh)
  MSA_STORE_FUNC(uint32_t, sw, msa_sw);
  #define SW(val, pdst)  MSA_STORE(val, pdst, msa_sw)
  MSA_STORE_FUNC(uint64_t, sd, msa_sd);
  #define SD(val, pdst)  MSA_STORE(val, pdst, msa_sd)
#else  // !(__mips_isa_rev >= 6)
  MSA_LOAD_FUNC(uint16_t, ulh, msa_ulh);
  #define LH(psrc)  MSA_LOAD(psrc, msa_ulh)
  MSA_LOAD_FUNC(uint32_t, ulw, msa_ulw);
  #define LW(psrc)  MSA_LOAD(psrc, msa_ulw)
  #if (__mips == 64)
    MSA_LOAD_FUNC(uint64_t, uld, msa_uld);
    #define LD(psrc)  MSA_LOAD(psrc, msa_uld)
  #else  // !(__mips == 64)
    #define LD(psrc)  ((((uint64_t)MSA_LOAD(psrc + 4, msa_ulw)) << 32) | \
                        MSA_LOAD(psrc, msa_ulw))
  #endif  // (__mips == 64)

  MSA_STORE_FUNC(uint16_t, ush, msa_ush);
  #define SH(val, pdst)  MSA_STORE(val, pdst, msa_ush)
  MSA_STORE_FUNC(uint32_t, usw, msa_usw);
  #define SW(val, pdst)  MSA_STORE(val, pdst, msa_usw)
  #define SD(val, pdst) do {                                               \
    uint8_t* const pdst_sd_m = (uint8_t*)(pdst);                           \
    const uint32_t val0_m = (uint32_t)(val & 0x00000000FFFFFFFF);          \
    const uint32_t val1_m = (uint32_t)((val >> 32) & 0x00000000FFFFFFFF);  \
    SW(val0_m, pdst_sd_m);                                                 \
    SW(val1_m, pdst_sd_m + 4);                                             \
  } while (0)
#endif  // (__mips_isa_rev >= 6)

/* Description : Load 4 words with stride
 * Arguments   : Inputs  - psrc, stride
 *               Outputs - out0, out1, out2, out3
 * Details     : Load word in 'out0' from (psrc)
 *               Load word in 'out1' from (psrc + stride)
 *               Load word in 'out2' from (psrc + 2 * stride)
 *               Load word in 'out3' from (psrc + 3 * stride)
 */
#define LW4(psrc, stride, out0, out1, out2, out3) do {  \
  const uint8_t* ptmp = (const uint8_t*)psrc;           \
  out0 = LW(ptmp);                                      \
  ptmp += stride;                                       \
  out1 = LW(ptmp);                                      \
  ptmp += stride;                                       \
  out2 = LW(ptmp);                                      \
  ptmp += stride;                                       \
  out3 = LW(ptmp);                                      \
} while (0)

/* Description : Store words with stride
 * Arguments   : Inputs - in0, in1, in2, in3, pdst, stride
 * Details     : Store word from 'in0' to (pdst)
 *               Store word from 'in1' to (pdst + stride)
 *               Store word from 'in2' to (pdst + 2 * stride)
 *               Store word from 'in3' to (pdst + 3 * stride)
 */
#define SW4(in0, in1, in2, in3, pdst, stride) do {  \
  uint8_t* ptmp = (uint8_t*)pdst;                   \
  SW(in0, ptmp);                                    \
  ptmp += stride;                                   \
  SW(in1, ptmp);                                    \
  ptmp += stride;                                   \
  SW(in2, ptmp);                                    \
  ptmp += stride;                                   \
  SW(in3, ptmp);                                    \
} while (0)

#define SW3(in0, in1, in2, pdst, stride) do {  \
  uint8_t* ptmp = (uint8_t*)pdst;              \
  SW(in0, ptmp);                               \
  ptmp += stride;                              \
  SW(in1, ptmp);                               \
  ptmp += stride;                              \
  SW(in2, ptmp);                               \
} while (0)

#define SW2(in0, in1, pdst, stride) do {  \
  uint8_t* ptmp = (uint8_t*)pdst;         \
  SW(in0, ptmp);                          \
  ptmp += stride;                         \
  SW(in1, ptmp);                          \
} while (0)

/* Description : Store 4 double words with stride
 * Arguments   : Inputs - in0, in1, in2, in3, pdst, stride
 * Details     : Store double word from 'in0' to (pdst)
 *               Store double word from 'in1' to (pdst + stride)
 *               Store double word from 'in2' to (pdst + 2 * stride)
 *               Store double word from 'in3' to (pdst + 3 * stride)
 */
#define SD4(in0, in1, in2, in3, pdst, stride) do {  \
  uint8_t* ptmp = (uint8_t*)pdst;                   \
  SD(in0, ptmp);                                    \
  ptmp += stride;                                   \
  SD(in1, ptmp);                                    \
  ptmp += stride;                                   \
  SD(in2, ptmp);                                    \
  ptmp += stride;                                   \
  SD(in3, ptmp);                                    \
} while (0)

/* Description : Load vectors with 16 byte elements with stride
 * Arguments   : Inputs  - psrc, stride
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Load 16 byte elements in 'out0' from (psrc)
 *               Load 16 byte elements in 'out1' from (psrc + stride)
 */
#define LD_B2(RTYPE, psrc, stride, out0, out1) do {  \
  out0 = LD_B(RTYPE, psrc);                          \
  out1 = LD_B(RTYPE, psrc + stride);                 \
} while (0)
#define LD_UB2(...) LD_B2(v16u8, __VA_ARGS__)
#define LD_SB2(...) LD_B2(v16i8, __VA_ARGS__)

#define LD_B3(RTYPE, psrc, stride, out0, out1, out2) do {  \
  LD_B2(RTYPE, psrc, stride, out0, out1);                  \
  out2 = LD_B(RTYPE, psrc + 2 * stride);                   \
} while (0)
#define LD_UB3(...) LD_B3(v16u8, __VA_ARGS__)
#define LD_SB3(...) LD_B3(v16i8, __VA_ARGS__)

#define LD_B4(RTYPE, psrc, stride, out0, out1, out2, out3) do {  \
  LD_B2(RTYPE, psrc, stride, out0, out1);                        \
  LD_B2(RTYPE, psrc + 2 * stride , stride, out2, out3);          \
} while (0)
#define LD_UB4(...) LD_B4(v16u8, __VA_ARGS__)
#define LD_SB4(...) LD_B4(v16i8, __VA_ARGS__)

#define LD_B8(RTYPE, psrc, stride,                                  \
              out0, out1, out2, out3, out4, out5, out6, out7) do {  \
  LD_B4(RTYPE, psrc, stride, out0, out1, out2, out3);               \
  LD_B4(RTYPE, psrc + 4 * stride, stride, out4, out5, out6, out7);  \
} while (0)
#define LD_UB8(...) LD_B8(v16u8, __VA_ARGS__)
#define LD_SB8(...) LD_B8(v16i8, __VA_ARGS__)

/* Description : Load vectors with 8 halfword elements with stride
 * Arguments   : Inputs  - psrc, stride
 *               Outputs - out0, out1
 * Details     : Load 8 halfword elements in 'out0' from (psrc)
 *               Load 8 halfword elements in 'out1' from (psrc + stride)
 */
#define LD_H2(RTYPE, psrc, stride, out0, out1) do {  \
  out0 = LD_H(RTYPE, psrc);                          \
  out1 = LD_H(RTYPE, psrc + stride);                 \
} while (0)
#define LD_UH2(...) LD_H2(v8u16, __VA_ARGS__)
#define LD_SH2(...) LD_H2(v8i16, __VA_ARGS__)

/* Description : Load vectors with 4 word elements with stride
 * Arguments   : Inputs  - psrc, stride
 *               Outputs - out0, out1, out2, out3
 * Details     : Load 4 word elements in 'out0' from (psrc + 0 * stride)
 *               Load 4 word elements in 'out1' from (psrc + 1 * stride)
 *               Load 4 word elements in 'out2' from (psrc + 2 * stride)
 *               Load 4 word elements in 'out3' from (psrc + 3 * stride)
 */
#define LD_W2(RTYPE, psrc, stride, out0, out1) do {  \
  out0 = LD_W(RTYPE, psrc);                          \
  out1 = LD_W(RTYPE, psrc + stride);                 \
} while (0)
#define LD_UW2(...) LD_W2(v4u32, __VA_ARGS__)
#define LD_SW2(...) LD_W2(v4i32, __VA_ARGS__)

#define LD_W3(RTYPE, psrc, stride, out0, out1, out2) do {  \
  LD_W2(RTYPE, psrc, stride, out0, out1);                  \
  out2 = LD_W(RTYPE, psrc + 2 * stride);                   \
} while (0)
#define LD_UW3(...) LD_W3(v4u32, __VA_ARGS__)
#define LD_SW3(...) LD_W3(v4i32, __VA_ARGS__)

#define LD_W4(RTYPE, psrc, stride, out0, out1, out2, out3) do {  \
  LD_W2(RTYPE, psrc, stride, out0, out1);                        \
  LD_W2(RTYPE, psrc + 2 * stride, stride, out2, out3);           \
} while (0)
#define LD_UW4(...) LD_W4(v4u32, __VA_ARGS__)
#define LD_SW4(...) LD_W4(v4i32, __VA_ARGS__)

/* Description : Store vectors of 16 byte elements with stride
 * Arguments   : Inputs - in0, in1, pdst, stride
 * Details     : Store 16 byte elements from 'in0' to (pdst)
 *               Store 16 byte elements from 'in1' to (pdst + stride)
 */
#define ST_B2(RTYPE, in0, in1, pdst, stride) do {  \
  ST_B(RTYPE, in0, pdst);                          \
  ST_B(RTYPE, in1, pdst + stride);                 \
} while (0)
#define ST_UB2(...) ST_B2(v16u8, __VA_ARGS__)
#define ST_SB2(...) ST_B2(v16i8, __VA_ARGS__)

#define ST_B4(RTYPE, in0, in1, in2, in3, pdst, stride) do {  \
  ST_B2(RTYPE, in0, in1, pdst, stride);                      \
  ST_B2(RTYPE, in2, in3, pdst + 2 * stride, stride);         \
} while (0)
#define ST_UB4(...) ST_B4(v16u8, __VA_ARGS__)
#define ST_SB4(...) ST_B4(v16i8, __VA_ARGS__)

#define ST_B8(RTYPE, in0, in1, in2, in3, in4, in5, in6, in7,    \
              pdst, stride) do {                                \
  ST_B4(RTYPE, in0, in1, in2, in3, pdst, stride);               \
  ST_B4(RTYPE, in4, in5, in6, in7, pdst + 4 * stride, stride);  \
} while (0)
#define ST_UB8(...) ST_B8(v16u8, __VA_ARGS__)

/* Description : Store vectors of 4 word elements with stride
 * Arguments   : Inputs - in0, in1, in2, in3, pdst, stride
 * Details     : Store 4 word elements from 'in0' to (pdst + 0 * stride)
 *               Store 4 word elements from 'in1' to (pdst + 1 * stride)
 *               Store 4 word elements from 'in2' to (pdst + 2 * stride)
 *               Store 4 word elements from 'in3' to (pdst + 3 * stride)
 */
#define ST_W2(RTYPE, in0, in1, pdst, stride) do {  \
  ST_W(RTYPE, in0, pdst);                          \
  ST_W(RTYPE, in1, pdst + stride);                 \
} while (0)
#define ST_UW2(...) ST_W2(v4u32, __VA_ARGS__)
#define ST_SW2(...) ST_W2(v4i32, __VA_ARGS__)

#define ST_W3(RTYPE, in0, in1, in2, pdst, stride) do {  \
  ST_W2(RTYPE, in0, in1, pdst, stride);                 \
  ST_W(RTYPE, in2, pdst + 2 * stride);                  \
} while (0)
#define ST_UW3(...) ST_W3(v4u32, __VA_ARGS__)
#define ST_SW3(...) ST_W3(v4i32, __VA_ARGS__)

#define ST_W4(RTYPE, in0, in1, in2, in3, pdst, stride) do {  \
  ST_W2(RTYPE, in0, in1, pdst, stride);                      \
  ST_W2(RTYPE, in2, in3, pdst + 2 * stride, stride);         \
} while (0)
#define ST_UW4(...) ST_W4(v4u32, __VA_ARGS__)
#define ST_SW4(...) ST_W4(v4i32, __VA_ARGS__)

/* Description : Store vectors of 8 halfword elements with stride
 * Arguments   : Inputs - in0, in1, pdst, stride
 * Details     : Store 8 halfword elements from 'in0' to (pdst)
 *               Store 8 halfword elements from 'in1' to (pdst + stride)
 */
#define ST_H2(RTYPE, in0, in1, pdst, stride) do {  \
  ST_H(RTYPE, in0, pdst);                          \
  ST_H(RTYPE, in1, pdst + stride);                 \
} while (0)
#define ST_UH2(...) ST_H2(v8u16, __VA_ARGS__)
#define ST_SH2(...) ST_H2(v8i16, __VA_ARGS__)

/* Description : Store 2x4 byte block to destination memory from input vector
 * Arguments   : Inputs - in, stidx, pdst, stride
 * Details     : Index 'stidx' halfword element from 'in' vector is copied to
 *               the GP register and stored to (pdst)
 *               Index 'stidx+1' halfword element from 'in' vector is copied to
 *               the GP register and stored to (pdst + stride)
 *               Index 'stidx+2' halfword element from 'in' vector is copied to
 *               the GP register and stored to (pdst + 2 * stride)
 *               Index 'stidx+3' halfword element from 'in' vector is copied to
 *               the GP register and stored to (pdst + 3 * stride)
 */
#define ST2x4_UB(in, stidx, pdst, stride) do {                   \
  uint8_t* pblk_2x4_m = (uint8_t*)pdst;                          \
  const uint16_t out0_m = __msa_copy_s_h((v8i16)in, stidx);      \
  const uint16_t out1_m = __msa_copy_s_h((v8i16)in, stidx + 1);  \
  const uint16_t out2_m = __msa_copy_s_h((v8i16)in, stidx + 2);  \
  const uint16_t out3_m = __msa_copy_s_h((v8i16)in, stidx + 3);  \
  SH(out0_m, pblk_2x4_m);                                        \
  pblk_2x4_m += stride;                                          \
  SH(out1_m, pblk_2x4_m);                                        \
  pblk_2x4_m += stride;                                          \
  SH(out2_m, pblk_2x4_m);                                        \
  pblk_2x4_m += stride;                                          \
  SH(out3_m, pblk_2x4_m);                                        \
} while (0)

/* Description : Store 4x4 byte block to destination memory from input vector
 * Arguments   : Inputs - in0, in1, pdst, stride
 * Details     : 'Idx0' word element from input vector 'in0' is copied to the
 *               GP register and stored to (pdst)
 *               'Idx1' word element from input vector 'in0' is copied to the
 *               GP register and stored to (pdst + stride)
 *               'Idx2' word element from input vector 'in0' is copied to the
 *               GP register and stored to (pdst + 2 * stride)
 *               'Idx3' word element from input vector 'in0' is copied to the
 *               GP register and stored to (pdst + 3 * stride)
 */
#define ST4x4_UB(in0, in1, idx0, idx1, idx2, idx3, pdst, stride) do {  \
  uint8_t* const pblk_4x4_m = (uint8_t*)pdst;                          \
  const uint32_t out0_m = __msa_copy_s_w((v4i32)in0, idx0);            \
  const uint32_t out1_m = __msa_copy_s_w((v4i32)in0, idx1);            \
  const uint32_t out2_m = __msa_copy_s_w((v4i32)in1, idx2);            \
  const uint32_t out3_m = __msa_copy_s_w((v4i32)in1, idx3);            \
  SW4(out0_m, out1_m, out2_m, out3_m, pblk_4x4_m, stride);             \
} while (0)

#define ST4x8_UB(in0, in1, pdst, stride) do {                     \
  uint8_t* const pblk_4x8 = (uint8_t*)pdst;                       \
  ST4x4_UB(in0, in0, 0, 1, 2, 3, pblk_4x8, stride);               \
  ST4x4_UB(in1, in1, 0, 1, 2, 3, pblk_4x8 + 4 * stride, stride);  \
} while (0)

/* Description : Immediate number of elements to slide
 * Arguments   : Inputs  - in0, in1, slide_val
 *               Outputs - out
 *               Return Type - as per RTYPE
 * Details     : Byte elements from 'in1' vector are slid into 'in0' by
 *               value specified in the 'slide_val'
 */
#define SLDI_B(RTYPE, in0, in1, slide_val)                      \
        (RTYPE)__msa_sldi_b((v16i8)in0, (v16i8)in1, slide_val)  \

#define SLDI_UB(...) SLDI_B(v16u8, __VA_ARGS__)
#define SLDI_SB(...) SLDI_B(v16i8, __VA_ARGS__)
#define SLDI_SH(...) SLDI_B(v8i16, __VA_ARGS__)

/* Description : Shuffle byte vector elements as per mask vector
 * Arguments   : Inputs  - in0, in1, in2, in3, mask0, mask1
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Byte elements from 'in0' & 'in1' are copied selectively to
 *               'out0' as per control vector 'mask0'
 */
#define VSHF_B(RTYPE, in0, in1, mask)                              \
        (RTYPE)__msa_vshf_b((v16i8)mask, (v16i8)in1, (v16i8)in0)

#define VSHF_UB(...) VSHF_B(v16u8, __VA_ARGS__)
#define VSHF_SB(...) VSHF_B(v16i8, __VA_ARGS__)
#define VSHF_UH(...) VSHF_B(v8u16, __VA_ARGS__)
#define VSHF_SH(...) VSHF_B(v8i16, __VA_ARGS__)

#define VSHF_B2(RTYPE, in0, in1, in2, in3, mask0, mask1, out0, out1) do {  \
  out0 = VSHF_B(RTYPE, in0, in1, mask0);                                   \
  out1 = VSHF_B(RTYPE, in2, in3, mask1);                                   \
} while (0)
#define VSHF_B2_UB(...) VSHF_B2(v16u8, __VA_ARGS__)
#define VSHF_B2_SB(...) VSHF_B2(v16i8, __VA_ARGS__)
#define VSHF_B2_UH(...) VSHF_B2(v8u16, __VA_ARGS__)
#define VSHF_B2_SH(...) VSHF_B2(v8i16, __VA_ARGS__)

/* Description : Shuffle halfword vector elements as per mask vector
 * Arguments   : Inputs  - in0, in1, in2, in3, mask0, mask1
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : halfword elements from 'in0' & 'in1' are copied selectively to
 *               'out0' as per control vector 'mask0'
 */
#define VSHF_H2(RTYPE, in0, in1, in2, in3, mask0, mask1, out0, out1) do {  \
  out0 = (RTYPE)__msa_vshf_h((v8i16)mask0, (v8i16)in1, (v8i16)in0);        \
  out1 = (RTYPE)__msa_vshf_h((v8i16)mask1, (v8i16)in3, (v8i16)in2);        \
} while (0)
#define VSHF_H2_UH(...) VSHF_H2(v8u16, __VA_ARGS__)
#define VSHF_H2_SH(...) VSHF_H2(v8i16, __VA_ARGS__)

/* Description : Dot product of byte vector elements
 * Arguments   : Inputs  - mult0, mult1, cnst0, cnst1
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Signed byte elements from 'mult0' are multiplied with
 *               signed byte elements from 'cnst0' producing a result
 *               twice the size of input i.e. signed halfword.
 *               The multiplication result of adjacent odd-even elements
 *               are added together and written to the 'out0' vector
*/
#define DOTP_SB2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1) do {  \
  out0 = (RTYPE)__msa_dotp_s_h((v16i8)mult0, (v16i8)cnst0);           \
  out1 = (RTYPE)__msa_dotp_s_h((v16i8)mult1, (v16i8)cnst1);           \
} while (0)
#define DOTP_SB2_SH(...) DOTP_SB2(v8i16, __VA_ARGS__)

/* Description : Dot product of halfword vector elements
 * Arguments   : Inputs  - mult0, mult1, cnst0, cnst1
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Signed halfword elements from 'mult0' are multiplied with
 *               signed halfword elements from 'cnst0' producing a result
 *               twice the size of input i.e. signed word.
 *               The multiplication result of adjacent odd-even elements
 *               are added together and written to the 'out0' vector
 */
#define DOTP_SH2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1) do {  \
  out0 = (RTYPE)__msa_dotp_s_w((v8i16)mult0, (v8i16)cnst0);           \
  out1 = (RTYPE)__msa_dotp_s_w((v8i16)mult1, (v8i16)cnst1);           \
} while (0)
#define DOTP_SH2_SW(...) DOTP_SH2(v4i32, __VA_ARGS__)

/* Description : Dot product of unsigned word vector elements
 * Arguments   : Inputs  - mult0, mult1, cnst0, cnst1
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Unsigned word elements from 'mult0' are multiplied with
 *               unsigned word elements from 'cnst0' producing a result
 *               twice the size of input i.e. unsigned double word.
 *               The multiplication result of adjacent odd-even elements
 *               are added together and written to the 'out0' vector
 */
#define DOTP_UW2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1) do {  \
  out0 = (RTYPE)__msa_dotp_u_d((v4u32)mult0, (v4u32)cnst0);           \
  out1 = (RTYPE)__msa_dotp_u_d((v4u32)mult1, (v4u32)cnst1);           \
} while (0)
#define DOTP_UW2_UD(...) DOTP_UW2(v2u64, __VA_ARGS__)

/* Description : Dot product & addition of halfword vector elements
 * Arguments   : Inputs  - mult0, mult1, cnst0, cnst1
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Signed halfword elements from 'mult0' are multiplied with
 *               signed halfword elements from 'cnst0' producing a result
 *               twice the size of input i.e. signed word.
 *               The multiplication result of adjacent odd-even elements
 *               are added to the 'out0' vector
 */
#define DPADD_SH2(RTYPE, mult0, mult1, cnst0, cnst1, out0, out1) do {      \
  out0 = (RTYPE)__msa_dpadd_s_w((v4i32)out0, (v8i16)mult0, (v8i16)cnst0);  \
  out1 = (RTYPE)__msa_dpadd_s_w((v4i32)out1, (v8i16)mult1, (v8i16)cnst1);  \
} while (0)
#define DPADD_SH2_SW(...) DPADD_SH2(v4i32, __VA_ARGS__)

/* Description : Clips all signed halfword elements of input vector
 *               between 0 & 255
 * Arguments   : Input/output  - val
 *               Return Type - signed halfword
 */
#define CLIP_SH_0_255(val) do {                   \
  const v8i16 max_m = __msa_ldi_h(255);           \
  val = __msa_maxi_s_h((v8i16)val, 0);            \
  val = __msa_min_s_h(max_m, (v8i16)val);         \
} while (0)

#define CLIP_SH2_0_255(in0, in1) do {  \
  CLIP_SH_0_255(in0);                  \
  CLIP_SH_0_255(in1);                  \
} while (0)

#define CLIP_SH4_0_255(in0, in1, in2, in3) do {  \
  CLIP_SH2_0_255(in0, in1);                      \
  CLIP_SH2_0_255(in2, in3);                      \
} while (0)

/* Description : Clips all unsigned halfword elements of input vector
 *               between 0 & 255
 * Arguments   : Input  - in
 *               Output - out_m
 *               Return Type - unsigned halfword
 */
#define CLIP_UH_0_255(in) do {                    \
  const v8u16 max_m = (v8u16)__msa_ldi_h(255);    \
  in = __msa_maxi_u_h((v8u16) in, 0);             \
  in = __msa_min_u_h((v8u16) max_m, (v8u16) in);  \
} while (0)

#define CLIP_UH2_0_255(in0, in1) do {  \
  CLIP_UH_0_255(in0);                  \
  CLIP_UH_0_255(in1);                  \
} while (0)

/* Description : Clips all signed word elements of input vector
 *               between 0 & 255
 * Arguments   : Input/output  - val
 *               Return Type - signed word
 */
#define CLIP_SW_0_255(val) do {                   \
  const v4i32 max_m = __msa_ldi_w(255);           \
  val = __msa_maxi_s_w((v4i32)val, 0);            \
  val = __msa_min_s_w(max_m, (v4i32)val);         \
} while (0)

#define CLIP_SW4_0_255(in0, in1, in2, in3) do {   \
  CLIP_SW_0_255(in0);                             \
  CLIP_SW_0_255(in1);                             \
  CLIP_SW_0_255(in2);                             \
  CLIP_SW_0_255(in3);                             \
} while (0)

/* Description : Horizontal addition of 4 signed word elements of input vector
 * Arguments   : Input  - in       (signed word vector)
 *               Output - sum_m    (i32 sum)
 *               Return Type - signed word (GP)
 * Details     : 4 signed word elements of 'in' vector are added together and
 *               the resulting integer sum is returned
 */
static WEBP_INLINE int32_t func_hadd_sw_s32(v4i32 in) {
  const v2i64 res0_m = __msa_hadd_s_d((v4i32)in, (v4i32)in);
  const v2i64 res1_m = __msa_splati_d(res0_m, 1);
  const v2i64 out = res0_m + res1_m;
  int32_t sum_m = __msa_copy_s_w((v4i32)out, 0);
  return sum_m;
}
#define HADD_SW_S32(in) func_hadd_sw_s32(in)

/* Description : Horizontal addition of 8 signed halfword elements
 * Arguments   : Input  - in       (signed halfword vector)
 *               Output - sum_m    (s32 sum)
 *               Return Type - signed word
 * Details     : 8 signed halfword elements of input vector are added
 *               together and the resulting integer sum is returned
 */
static WEBP_INLINE int32_t func_hadd_sh_s32(v8i16 in) {
  const v4i32 res = __msa_hadd_s_w(in, in);
  const v2i64 res0 = __msa_hadd_s_d(res, res);
  const v2i64 res1 = __msa_splati_d(res0, 1);
  const v2i64 res2 = res0 + res1;
  const int32_t sum_m = __msa_copy_s_w((v4i32)res2, 0);
  return sum_m;
}
#define HADD_SH_S32(in) func_hadd_sh_s32(in)

/* Description : Horizontal addition of 8 unsigned halfword elements
 * Arguments   : Input  - in       (unsigned halfword vector)
 *               Output - sum_m    (u32 sum)
 *               Return Type - unsigned word
 * Details     : 8 unsigned halfword elements of input vector are added
 *               together and the resulting integer sum is returned
 */
static WEBP_INLINE uint32_t func_hadd_uh_u32(v8u16 in) {
  uint32_t sum_m;
  const v4u32 res_m = __msa_hadd_u_w(in, in);
  v2u64 res0_m = __msa_hadd_u_d(res_m, res_m);
  v2u64 res1_m = (v2u64)__msa_splati_d((v2i64)res0_m, 1);
  res0_m = res0_m + res1_m;
  sum_m = __msa_copy_s_w((v4i32)res0_m, 0);
  return sum_m;
}
#define HADD_UH_U32(in) func_hadd_uh_u32(in)

/* Description : Horizontal addition of signed half word vector elements
   Arguments   : Inputs  - in0, in1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Each signed odd half word element from 'in0' is added to
                 even signed half word element from 'in0' (pairwise) and the
                 halfword result is written in 'out0'
*/
#define HADD_SH2(RTYPE, in0, in1, out0, out1) do {       \
  out0 = (RTYPE)__msa_hadd_s_w((v8i16)in0, (v8i16)in0);  \
  out1 = (RTYPE)__msa_hadd_s_w((v8i16)in1, (v8i16)in1);  \
} while (0)
#define HADD_SH2_SW(...) HADD_SH2(v4i32, __VA_ARGS__)

#define HADD_SH4(RTYPE, in0, in1, in2, in3, out0, out1, out2, out3) do {  \
  HADD_SH2(RTYPE, in0, in1, out0, out1);                                  \
  HADD_SH2(RTYPE, in2, in3, out2, out3);                                  \
} while (0)
#define HADD_SH4_SW(...) HADD_SH4(v4i32, __VA_ARGS__)

/* Description : Horizontal subtraction of unsigned byte vector elements
 * Arguments   : Inputs  - in0, in1
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Each unsigned odd byte element from 'in0' is subtracted from
 *               even unsigned byte element from 'in0' (pairwise) and the
 *               halfword result is written to 'out0'
 */
#define HSUB_UB2(RTYPE, in0, in1, out0, out1) do {       \
  out0 = (RTYPE)__msa_hsub_u_h((v16u8)in0, (v16u8)in0);  \
  out1 = (RTYPE)__msa_hsub_u_h((v16u8)in1, (v16u8)in1);  \
} while (0)
#define HSUB_UB2_UH(...) HSUB_UB2(v8u16, __VA_ARGS__)
#define HSUB_UB2_SH(...) HSUB_UB2(v8i16, __VA_ARGS__)
#define HSUB_UB2_SW(...) HSUB_UB2(v4i32, __VA_ARGS__)

/* Description : Set element n input vector to GPR value
 * Arguments   : Inputs - in0, in1, in2, in3
 *               Output - out
 *               Return Type - as per RTYPE
 * Details     : Set element 0 in vector 'out' to value specified in 'in0'
 */
#define INSERT_W2(RTYPE, in0, in1, out) do {        \
  out = (RTYPE)__msa_insert_w((v4i32)out, 0, in0);  \
  out = (RTYPE)__msa_insert_w((v4i32)out, 1, in1);  \
} while (0)
#define INSERT_W2_UB(...) INSERT_W2(v16u8, __VA_ARGS__)
#define INSERT_W2_SB(...) INSERT_W2(v16i8, __VA_ARGS__)

#define INSERT_W4(RTYPE, in0, in1, in2, in3, out) do {  \
  out = (RTYPE)__msa_insert_w((v4i32)out, 0, in0);      \
  out = (RTYPE)__msa_insert_w((v4i32)out, 1, in1);      \
  out = (RTYPE)__msa_insert_w((v4i32)out, 2, in2);      \
  out = (RTYPE)__msa_insert_w((v4i32)out, 3, in3);      \
} while (0)
#define INSERT_W4_UB(...) INSERT_W4(v16u8, __VA_ARGS__)
#define INSERT_W4_SB(...) INSERT_W4(v16i8, __VA_ARGS__)
#define INSERT_W4_SW(...) INSERT_W4(v4i32, __VA_ARGS__)

/* Description : Set element n of double word input vector to GPR value
 * Arguments   : Inputs - in0, in1
 *               Output - out
 *               Return Type - as per RTYPE
 * Details     : Set element 0 in vector 'out' to GPR value specified in 'in0'
 *               Set element 1 in vector 'out' to GPR value specified in 'in1'
 */
#define INSERT_D2(RTYPE, in0, in1, out) do {        \
  out = (RTYPE)__msa_insert_d((v2i64)out, 0, in0);  \
  out = (RTYPE)__msa_insert_d((v2i64)out, 1, in1);  \
} while (0)
#define INSERT_D2_UB(...) INSERT_D2(v16u8, __VA_ARGS__)
#define INSERT_D2_SB(...) INSERT_D2(v16i8, __VA_ARGS__)

/* Description : Interleave even byte elements from vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Even byte elements of 'in0' and 'in1' are interleaved
 *               and written to 'out0'
 */
#define ILVEV_B2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_ilvev_b((v16i8)in1, (v16i8)in0);        \
  out1 = (RTYPE)__msa_ilvev_b((v16i8)in3, (v16i8)in2);        \
} while (0)
#define ILVEV_B2_UB(...) ILVEV_B2(v16u8, __VA_ARGS__)
#define ILVEV_B2_SB(...) ILVEV_B2(v16i8, __VA_ARGS__)
#define ILVEV_B2_UH(...) ILVEV_B2(v8u16, __VA_ARGS__)
#define ILVEV_B2_SH(...) ILVEV_B2(v8i16, __VA_ARGS__)
#define ILVEV_B2_SD(...) ILVEV_B2(v2i64, __VA_ARGS__)

/* Description : Interleave odd byte elements from vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Odd byte elements of 'in0' and 'in1' are interleaved
 *               and written to 'out0'
 */
#define ILVOD_B2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_ilvod_b((v16i8)in1, (v16i8)in0);        \
  out1 = (RTYPE)__msa_ilvod_b((v16i8)in3, (v16i8)in2);        \
} while (0)
#define ILVOD_B2_UB(...) ILVOD_B2(v16u8, __VA_ARGS__)
#define ILVOD_B2_SB(...) ILVOD_B2(v16i8, __VA_ARGS__)
#define ILVOD_B2_UH(...) ILVOD_B2(v8u16, __VA_ARGS__)
#define ILVOD_B2_SH(...) ILVOD_B2(v8i16, __VA_ARGS__)
#define ILVOD_B2_SD(...) ILVOD_B2(v2i64, __VA_ARGS__)

/* Description : Interleave even halfword elements from vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Even halfword elements of 'in0' and 'in1' are interleaved
 *               and written to 'out0'
 */
#define ILVEV_H2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_ilvev_h((v8i16)in1, (v8i16)in0);        \
  out1 = (RTYPE)__msa_ilvev_h((v8i16)in3, (v8i16)in2);        \
} while (0)
#define ILVEV_H2_UB(...) ILVEV_H2(v16u8, __VA_ARGS__)
#define ILVEV_H2_UH(...) ILVEV_H2(v8u16, __VA_ARGS__)
#define ILVEV_H2_SH(...) ILVEV_H2(v8i16, __VA_ARGS__)
#define ILVEV_H2_SW(...) ILVEV_H2(v4i32, __VA_ARGS__)

/* Description : Interleave odd halfword elements from vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Odd halfword elements of 'in0' and 'in1' are interleaved
 *               and written to 'out0'
 */
#define ILVOD_H2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_ilvod_h((v8i16)in1, (v8i16)in0);        \
  out1 = (RTYPE)__msa_ilvod_h((v8i16)in3, (v8i16)in2);        \
} while (0)
#define ILVOD_H2_UB(...) ILVOD_H2(v16u8, __VA_ARGS__)
#define ILVOD_H2_UH(...) ILVOD_H2(v8u16, __VA_ARGS__)
#define ILVOD_H2_SH(...) ILVOD_H2(v8i16, __VA_ARGS__)
#define ILVOD_H2_SW(...) ILVOD_H2(v4i32, __VA_ARGS__)

/* Description : Interleave even word elements from vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Even word elements of 'in0' and 'in1' are interleaved
 *               and written to 'out0'
 */
#define ILVEV_W2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_ilvev_w((v4i32)in1, (v4i32)in0);        \
  out1 = (RTYPE)__msa_ilvev_w((v4i32)in3, (v4i32)in2);        \
} while (0)
#define ILVEV_W2_UB(...) ILVEV_W2(v16u8, __VA_ARGS__)
#define ILVEV_W2_SB(...) ILVEV_W2(v16i8, __VA_ARGS__)
#define ILVEV_W2_UH(...) ILVEV_W2(v8u16, __VA_ARGS__)
#define ILVEV_W2_SD(...) ILVEV_W2(v2i64, __VA_ARGS__)

/* Description : Interleave even-odd word elements from vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Even word elements of 'in0' and 'in1' are interleaved
 *               and written to 'out0'
 *               Odd word elements of 'in2' and 'in3' are interleaved
 *               and written to 'out1'
 */
#define ILVEVOD_W2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_ilvev_w((v4i32)in1, (v4i32)in0);          \
  out1 = (RTYPE)__msa_ilvod_w((v4i32)in3, (v4i32)in2);          \
} while (0)
#define ILVEVOD_W2_UB(...) ILVEVOD_W2(v16u8, __VA_ARGS__)
#define ILVEVOD_W2_UH(...) ILVEVOD_W2(v8u16, __VA_ARGS__)
#define ILVEVOD_W2_SH(...) ILVEVOD_W2(v8i16, __VA_ARGS__)
#define ILVEVOD_W2_SW(...) ILVEVOD_W2(v4i32, __VA_ARGS__)

/* Description : Interleave even-odd half-word elements from vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Even half-word elements of 'in0' and 'in1' are interleaved
 *               and written to 'out0'
 *               Odd half-word elements of 'in2' and 'in3' are interleaved
 *               and written to 'out1'
 */
#define ILVEVOD_H2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_ilvev_h((v8i16)in1, (v8i16)in0);          \
  out1 = (RTYPE)__msa_ilvod_h((v8i16)in3, (v8i16)in2);          \
} while (0)
#define ILVEVOD_H2_UB(...) ILVEVOD_H2(v16u8, __VA_ARGS__)
#define ILVEVOD_H2_UH(...) ILVEVOD_H2(v8u16, __VA_ARGS__)
#define ILVEVOD_H2_SH(...) ILVEVOD_H2(v8i16, __VA_ARGS__)
#define ILVEVOD_H2_SW(...) ILVEVOD_H2(v4i32, __VA_ARGS__)

/* Description : Interleave even double word elements from vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Even double word elements of 'in0' and 'in1' are interleaved
 *               and written to 'out0'
 */
#define ILVEV_D2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_ilvev_d((v2i64)in1, (v2i64)in0);        \
  out1 = (RTYPE)__msa_ilvev_d((v2i64)in3, (v2i64)in2);        \
} while (0)
#define ILVEV_D2_UB(...) ILVEV_D2(v16u8, __VA_ARGS__)
#define ILVEV_D2_SB(...) ILVEV_D2(v16i8, __VA_ARGS__)
#define ILVEV_D2_SW(...) ILVEV_D2(v4i32, __VA_ARGS__)
#define ILVEV_D2_SD(...) ILVEV_D2(v2i64, __VA_ARGS__)

/* Description : Interleave left half of byte elements from vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Left half of byte elements of 'in0' and 'in1' are interleaved
 *               and written to 'out0'.
 */
#define ILVL_B2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_ilvl_b((v16i8)in0, (v16i8)in1);        \
  out1 = (RTYPE)__msa_ilvl_b((v16i8)in2, (v16i8)in3);        \
} while (0)
#define ILVL_B2_UB(...) ILVL_B2(v16u8, __VA_ARGS__)
#define ILVL_B2_SB(...) ILVL_B2(v16i8, __VA_ARGS__)
#define ILVL_B2_UH(...) ILVL_B2(v8u16, __VA_ARGS__)
#define ILVL_B2_SH(...) ILVL_B2(v8i16, __VA_ARGS__)
#define ILVL_B2_SW(...) ILVL_B2(v4i32, __VA_ARGS__)

/* Description : Interleave right half of byte elements from vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Right half of byte elements of 'in0' and 'in1' are interleaved
 *               and written to out0.
 */
#define ILVR_B2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_ilvr_b((v16i8)in0, (v16i8)in1);        \
  out1 = (RTYPE)__msa_ilvr_b((v16i8)in2, (v16i8)in3);        \
} while (0)
#define ILVR_B2_UB(...) ILVR_B2(v16u8, __VA_ARGS__)
#define ILVR_B2_SB(...) ILVR_B2(v16i8, __VA_ARGS__)
#define ILVR_B2_UH(...) ILVR_B2(v8u16, __VA_ARGS__)
#define ILVR_B2_SH(...) ILVR_B2(v8i16, __VA_ARGS__)
#define ILVR_B2_SW(...) ILVR_B2(v4i32, __VA_ARGS__)

#define ILVR_B4(RTYPE, in0, in1, in2, in3, in4, in5, in6, in7,  \
                out0, out1, out2, out3) do {                    \
  ILVR_B2(RTYPE, in0, in1, in2, in3, out0, out1);               \
  ILVR_B2(RTYPE, in4, in5, in6, in7, out2, out3);               \
} while (0)
#define ILVR_B4_UB(...) ILVR_B4(v16u8, __VA_ARGS__)
#define ILVR_B4_SB(...) ILVR_B4(v16i8, __VA_ARGS__)
#define ILVR_B4_UH(...) ILVR_B4(v8u16, __VA_ARGS__)
#define ILVR_B4_SH(...) ILVR_B4(v8i16, __VA_ARGS__)
#define ILVR_B4_SW(...) ILVR_B4(v4i32, __VA_ARGS__)

/* Description : Interleave right half of halfword elements from vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Right half of halfword elements of 'in0' and 'in1' are
 *               interleaved and written to 'out0'.
 */
#define ILVR_H2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_ilvr_h((v8i16)in0, (v8i16)in1);        \
  out1 = (RTYPE)__msa_ilvr_h((v8i16)in2, (v8i16)in3);        \
} while (0)
#define ILVR_H2_UB(...) ILVR_H2(v16u8, __VA_ARGS__)
#define ILVR_H2_SH(...) ILVR_H2(v8i16, __VA_ARGS__)
#define ILVR_H2_SW(...) ILVR_H2(v4i32, __VA_ARGS__)

#define ILVR_H4(RTYPE, in0, in1, in2, in3, in4, in5, in6, in7,  \
                out0, out1, out2, out3) do {                    \
  ILVR_H2(RTYPE, in0, in1, in2, in3, out0, out1);               \
  ILVR_H2(RTYPE, in4, in5, in6, in7, out2, out3);               \
} while (0)
#define ILVR_H4_UB(...) ILVR_H4(v16u8, __VA_ARGS__)
#define ILVR_H4_SH(...) ILVR_H4(v8i16, __VA_ARGS__)
#define ILVR_H4_SW(...) ILVR_H4(v4i32, __VA_ARGS__)

/* Description : Interleave right half of double word elements from vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Right half of double word elements of 'in0' and 'in1' are
 *               interleaved and written to 'out0'.
 */
#define ILVR_D2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_ilvr_d((v2i64)in0, (v2i64)in1);        \
  out1 = (RTYPE)__msa_ilvr_d((v2i64)in2, (v2i64)in3);        \
} while (0)
#define ILVR_D2_UB(...) ILVR_D2(v16u8, __VA_ARGS__)
#define ILVR_D2_SB(...) ILVR_D2(v16i8, __VA_ARGS__)
#define ILVR_D2_SH(...) ILVR_D2(v8i16, __VA_ARGS__)

#define ILVR_D4(RTYPE, in0, in1, in2, in3, in4, in5, in6, in7,  \
                out0, out1, out2, out3) do {                    \
  ILVR_D2(RTYPE, in0, in1, in2, in3, out0, out1);               \
  ILVR_D2(RTYPE, in4, in5, in6, in7, out2, out3);               \
} while (0)
#define ILVR_D4_SB(...) ILVR_D4(v16i8, __VA_ARGS__)
#define ILVR_D4_UB(...) ILVR_D4(v16u8, __VA_ARGS__)

/* Description : Interleave both left and right half of input vectors
 * Arguments   : Inputs  - in0, in1
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Right half of byte elements from 'in0' and 'in1' are
 *               interleaved and written to 'out0'
 */
#define ILVRL_B2(RTYPE, in0, in1, out0, out1) do {     \
  out0 = (RTYPE)__msa_ilvr_b((v16i8)in0, (v16i8)in1);  \
  out1 = (RTYPE)__msa_ilvl_b((v16i8)in0, (v16i8)in1);  \
} while (0)
#define ILVRL_B2_UB(...) ILVRL_B2(v16u8, __VA_ARGS__)
#define ILVRL_B2_SB(...) ILVRL_B2(v16i8, __VA_ARGS__)
#define ILVRL_B2_UH(...) ILVRL_B2(v8u16, __VA_ARGS__)
#define ILVRL_B2_SH(...) ILVRL_B2(v8i16, __VA_ARGS__)
#define ILVRL_B2_SW(...) ILVRL_B2(v4i32, __VA_ARGS__)

#define ILVRL_H2(RTYPE, in0, in1, out0, out1) do {     \
  out0 = (RTYPE)__msa_ilvr_h((v8i16)in0, (v8i16)in1);  \
  out1 = (RTYPE)__msa_ilvl_h((v8i16)in0, (v8i16)in1);  \
} while (0)
#define ILVRL_H2_UB(...) ILVRL_H2(v16u8, __VA_ARGS__)
#define ILVRL_H2_SB(...) ILVRL_H2(v16i8, __VA_ARGS__)
#define ILVRL_H2_SH(...) ILVRL_H2(v8i16, __VA_ARGS__)
#define ILVRL_H2_SW(...) ILVRL_H2(v4i32, __VA_ARGS__)
#define ILVRL_H2_UW(...) ILVRL_H2(v4u32, __VA_ARGS__)

#define ILVRL_W2(RTYPE, in0, in1, out0, out1) do {     \
  out0 = (RTYPE)__msa_ilvr_w((v4i32)in0, (v4i32)in1);  \
  out1 = (RTYPE)__msa_ilvl_w((v4i32)in0, (v4i32)in1);  \
} while (0)
#define ILVRL_W2_UB(...) ILVRL_W2(v16u8, __VA_ARGS__)
#define ILVRL_W2_SH(...) ILVRL_W2(v8i16, __VA_ARGS__)
#define ILVRL_W2_SW(...) ILVRL_W2(v4i32, __VA_ARGS__)
#define ILVRL_W2_UW(...) ILVRL_W2(v4u32, __VA_ARGS__)

/* Description : Pack even byte elements of vector pairs
 *  Arguments   : Inputs  - in0, in1, in2, in3
 *                Outputs - out0, out1
 *                Return Type - as per RTYPE
 *  Details     : Even byte elements of 'in0' are copied to the left half of
 *                'out0' & even byte elements of 'in1' are copied to the right
 *                half of 'out0'.
 */
#define PCKEV_B2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_pckev_b((v16i8)in0, (v16i8)in1);        \
  out1 = (RTYPE)__msa_pckev_b((v16i8)in2, (v16i8)in3);        \
} while (0)
#define PCKEV_B2_SB(...) PCKEV_B2(v16i8, __VA_ARGS__)
#define PCKEV_B2_UB(...) PCKEV_B2(v16u8, __VA_ARGS__)
#define PCKEV_B2_SH(...) PCKEV_B2(v8i16, __VA_ARGS__)
#define PCKEV_B2_SW(...) PCKEV_B2(v4i32, __VA_ARGS__)

#define PCKEV_B4(RTYPE, in0, in1, in2, in3, in4, in5, in6, in7,  \
                 out0, out1, out2, out3) do {                    \
  PCKEV_B2(RTYPE, in0, in1, in2, in3, out0, out1);               \
  PCKEV_B2(RTYPE, in4, in5, in6, in7, out2, out3);               \
} while (0)
#define PCKEV_B4_SB(...) PCKEV_B4(v16i8, __VA_ARGS__)
#define PCKEV_B4_UB(...) PCKEV_B4(v16u8, __VA_ARGS__)
#define PCKEV_B4_SH(...) PCKEV_B4(v8i16, __VA_ARGS__)
#define PCKEV_B4_SW(...) PCKEV_B4(v4i32, __VA_ARGS__)

/* Description : Pack even halfword elements of vector pairs
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Even halfword elements of 'in0' are copied to the left half of
 *               'out0' & even halfword elements of 'in1' are copied to the
 *               right half of 'out0'.
 */
#define PCKEV_H2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_pckev_h((v8i16)in0, (v8i16)in1);        \
  out1 = (RTYPE)__msa_pckev_h((v8i16)in2, (v8i16)in3);        \
} while (0)
#define PCKEV_H2_UH(...) PCKEV_H2(v8u16, __VA_ARGS__)
#define PCKEV_H2_SH(...) PCKEV_H2(v8i16, __VA_ARGS__)
#define PCKEV_H2_SW(...) PCKEV_H2(v4i32, __VA_ARGS__)
#define PCKEV_H2_UW(...) PCKEV_H2(v4u32, __VA_ARGS__)

/* Description : Pack even word elements of vector pairs
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Even word elements of 'in0' are copied to the left half of
 *               'out0' & even word elements of 'in1' are copied to the
 *               right half of 'out0'.
 */
#define PCKEV_W2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_pckev_w((v4i32)in0, (v4i32)in1);        \
  out1 = (RTYPE)__msa_pckev_w((v4i32)in2, (v4i32)in3);        \
} while (0)
#define PCKEV_W2_UH(...) PCKEV_W2(v8u16, __VA_ARGS__)
#define PCKEV_W2_SH(...) PCKEV_W2(v8i16, __VA_ARGS__)
#define PCKEV_W2_SW(...) PCKEV_W2(v4i32, __VA_ARGS__)
#define PCKEV_W2_UW(...) PCKEV_W2(v4u32, __VA_ARGS__)

/* Description : Pack odd halfword elements of vector pairs
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Odd halfword elements of 'in0' are copied to the left half of
 *               'out0' & odd halfword elements of 'in1' are copied to the
 *               right half of 'out0'.
 */
#define PCKOD_H2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_pckod_h((v8i16)in0, (v8i16)in1);        \
  out1 = (RTYPE)__msa_pckod_h((v8i16)in2, (v8i16)in3);        \
} while (0)
#define PCKOD_H2_UH(...) PCKOD_H2(v8u16, __VA_ARGS__)
#define PCKOD_H2_SH(...) PCKOD_H2(v8i16, __VA_ARGS__)
#define PCKOD_H2_SW(...) PCKOD_H2(v4i32, __VA_ARGS__)
#define PCKOD_H2_UW(...) PCKOD_H2(v4u32, __VA_ARGS__)

/* Description : Arithmetic immediate shift right all elements of word vector
 * Arguments   : Inputs  - in0, in1, shift
 *               Outputs - in place operation
 *               Return Type - as per input vector RTYPE
 * Details     : Each element of vector 'in0' is right shifted by 'shift' and
 *               the result is written in-place. 'shift' is a GP variable.
 */
#define SRAI_W2(RTYPE, in0, in1, shift_val) do {  \
  in0 = (RTYPE)SRAI_W(in0, shift_val);            \
  in1 = (RTYPE)SRAI_W(in1, shift_val);            \
} while (0)
#define SRAI_W2_SW(...) SRAI_W2(v4i32, __VA_ARGS__)
#define SRAI_W2_UW(...) SRAI_W2(v4u32, __VA_ARGS__)

#define SRAI_W4(RTYPE, in0, in1, in2, in3, shift_val) do {  \
  SRAI_W2(RTYPE, in0, in1, shift_val);                      \
  SRAI_W2(RTYPE, in2, in3, shift_val);                      \
} while (0)
#define SRAI_W4_SW(...) SRAI_W4(v4i32, __VA_ARGS__)
#define SRAI_W4_UW(...) SRAI_W4(v4u32, __VA_ARGS__)

/* Description : Arithmetic shift right all elements of half-word vector
 * Arguments   : Inputs  - in0, in1, shift
 *               Outputs - in place operation
 *               Return Type - as per input vector RTYPE
 * Details     : Each element of vector 'in0' is right shifted by 'shift' and
 *               the result is written in-place. 'shift' is a GP variable.
 */
#define SRAI_H2(RTYPE, in0, in1, shift_val) do {  \
  in0 = (RTYPE)SRAI_H(in0, shift_val);            \
  in1 = (RTYPE)SRAI_H(in1, shift_val);            \
} while (0)
#define SRAI_H2_SH(...) SRAI_H2(v8i16, __VA_ARGS__)
#define SRAI_H2_UH(...) SRAI_H2(v8u16, __VA_ARGS__)

/* Description : Arithmetic rounded shift right all elements of word vector
 * Arguments   : Inputs  - in0, in1, shift
 *               Outputs - in place operation
 *               Return Type - as per input vector RTYPE
 * Details     : Each element of vector 'in0' is right shifted by 'shift' and
 *               the result is written in-place. 'shift' is a GP variable.
 */
#define SRARI_W2(RTYPE, in0, in1, shift) do {     \
  in0 = (RTYPE)__msa_srari_w((v4i32)in0, shift);  \
  in1 = (RTYPE)__msa_srari_w((v4i32)in1, shift);  \
} while (0)
#define SRARI_W2_SW(...) SRARI_W2(v4i32, __VA_ARGS__)

#define SRARI_W4(RTYPE, in0, in1, in2, in3, shift) do {  \
  SRARI_W2(RTYPE, in0, in1, shift);                      \
  SRARI_W2(RTYPE, in2, in3, shift);                      \
} while (0)
#define SRARI_W4_SH(...) SRARI_W4(v8i16, __VA_ARGS__)
#define SRARI_W4_UW(...) SRARI_W4(v4u32, __VA_ARGS__)
#define SRARI_W4_SW(...) SRARI_W4(v4i32, __VA_ARGS__)

/* Description : Shift right arithmetic rounded double words
 * Arguments   : Inputs  - in0, in1, shift
 *               Outputs - in place operation
 *               Return Type - as per RTYPE
 * Details     : Each element of vector 'in0' is shifted right arithmetically by
 *               the number of bits in the corresponding element in the vector
 *               'shift'. The last discarded bit is added to shifted value for
 *               rounding and the result is written in-place.
 *               'shift' is a vector.
 */
#define SRAR_D2(RTYPE, in0, in1, shift) do {            \
  in0 = (RTYPE)__msa_srar_d((v2i64)in0, (v2i64)shift);  \
  in1 = (RTYPE)__msa_srar_d((v2i64)in1, (v2i64)shift);  \
} while (0)
#define SRAR_D2_SW(...) SRAR_D2(v4i32, __VA_ARGS__)
#define SRAR_D2_SD(...) SRAR_D2(v2i64, __VA_ARGS__)
#define SRAR_D2_UD(...) SRAR_D2(v2u64, __VA_ARGS__)

#define SRAR_D4(RTYPE, in0, in1, in2, in3, shift) do {  \
  SRAR_D2(RTYPE, in0, in1, shift);                      \
  SRAR_D2(RTYPE, in2, in3, shift);                      \
} while (0)
#define SRAR_D4_SD(...) SRAR_D4(v2i64, __VA_ARGS__)
#define SRAR_D4_UD(...) SRAR_D4(v2u64, __VA_ARGS__)

/* Description : Addition of 2 pairs of half-word vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 * Details     : Each element in 'in0' is added to 'in1' and result is written
 *               to 'out0'.
 */
#define ADDVI_H2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)ADDVI_H(in0, in1);                            \
  out1 = (RTYPE)ADDVI_H(in2, in3);                            \
} while (0)
#define ADDVI_H2_SH(...) ADDVI_H2(v8i16, __VA_ARGS__)
#define ADDVI_H2_UH(...) ADDVI_H2(v8u16, __VA_ARGS__)

/* Description : Addition of 2 pairs of word vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 * Details     : Each element in 'in0' is added to 'in1' and result is written
 *               to 'out0'.
 */
#define ADDVI_W2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)ADDVI_W(in0, in1);                            \
  out1 = (RTYPE)ADDVI_W(in2, in3);                            \
} while (0)
#define ADDVI_W2_SW(...) ADDVI_W2(v4i32, __VA_ARGS__)

/* Description : Fill 2 pairs of word vectors with GP registers
 * Arguments   : Inputs  - in0, in1
 *               Outputs - out0, out1
 * Details     : GP register in0 is replicated in each word element of out0
 *               GP register in1 is replicated in each word element of out1
 */
#define FILL_W2(RTYPE, in0, in1, out0, out1) do {  \
  out0 = (RTYPE)__msa_fill_w(in0);                 \
  out1 = (RTYPE)__msa_fill_w(in1);                 \
} while (0)
#define FILL_W2_SW(...) FILL_W2(v4i32, __VA_ARGS__)

/* Description : Addition of 2 pairs of vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 * Details     : Each element in 'in0' is added to 'in1' and result is written
 *               to 'out0'.
 */
#define ADD2(in0, in1, in2, in3, out0, out1) do {  \
  out0 = in0 + in1;                                \
  out1 = in2 + in3;                                \
} while (0)

#define ADD4(in0, in1, in2, in3, in4, in5, in6, in7,  \
             out0, out1, out2, out3) do {             \
  ADD2(in0, in1, in2, in3, out0, out1);               \
  ADD2(in4, in5, in6, in7, out2, out3);               \
} while (0)

/* Description : Subtraction of 2 pairs of vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 * Details     : Each element in 'in1' is subtracted from 'in0' and result is
 *               written to 'out0'.
 */
#define SUB2(in0, in1, in2, in3, out0, out1) do {  \
  out0 = in0 - in1;                                \
  out1 = in2 - in3;                                \
} while (0)

#define SUB3(in0, in1, in2, in3, in4, in5, out0, out1, out2) do {  \
  out0 = in0 - in1;                                                \
  out1 = in2 - in3;                                                \
  out2 = in4 - in5;                                                \
} while (0)

#define SUB4(in0, in1, in2, in3, in4, in5, in6, in7,  \
             out0, out1, out2, out3) do {             \
  out0 = in0 - in1;                                   \
  out1 = in2 - in3;                                   \
  out2 = in4 - in5;                                   \
  out3 = in6 - in7;                                   \
} while (0)

/* Description : Addition - Subtraction of input vectors
 * Arguments   : Inputs  - in0, in1
 *               Outputs - out0, out1
 * Details     : Each element in 'in1' is added to 'in0' and result is
 *               written to 'out0'.
 *               Each element in 'in1' is subtracted from 'in0' and result is
 *               written to 'out1'.
 */
#define ADDSUB2(in0, in1, out0, out1) do {  \
  out0 = in0 + in1;                         \
  out1 = in0 - in1;                         \
} while (0)

/* Description : Multiplication of pairs of vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1
 * Details     : Each element from 'in0' is multiplied with elements from 'in1'
 *               and the result is written to 'out0'
 */
#define MUL2(in0, in1, in2, in3, out0, out1) do {  \
  out0 = in0 * in1;                                \
  out1 = in2 * in3;                                \
} while (0)

#define MUL4(in0, in1, in2, in3, in4, in5, in6, in7,  \
             out0, out1, out2, out3) do {             \
  MUL2(in0, in1, in2, in3, out0, out1);               \
  MUL2(in4, in5, in6, in7, out2, out3);               \
} while (0)

/* Description : Sign extend halfword elements from right half of the vector
 * Arguments   : Input  - in    (halfword vector)
 *               Output - out   (sign extended word vector)
 *               Return Type - signed word
 * Details     : Sign bit of halfword elements from input vector 'in' is
 *               extracted and interleaved with same vector 'in0' to generate
 *               4 word elements keeping sign intact
 */
#define UNPCK_R_SH_SW(in, out) do {                   \
  const v8i16 sign_m = __msa_clti_s_h((v8i16)in, 0);  \
  out = (v4i32)__msa_ilvr_h(sign_m, (v8i16)in);       \
} while (0)

/* Description : Sign extend halfword elements from input vector and return
 *               the result in pair of vectors
 * Arguments   : Input   - in            (halfword vector)
 *               Outputs - out0, out1   (sign extended word vectors)
 *               Return Type - signed word
 * Details     : Sign bit of halfword elements from input vector 'in' is
 *               extracted and interleaved right with same vector 'in0' to
 *               generate 4 signed word elements in 'out0'
 *               Then interleaved left with same vector 'in0' to
 *               generate 4 signed word elements in 'out1'
 */
#define UNPCK_SH_SW(in, out0, out1) do {              \
  const v8i16 tmp_m = __msa_clti_s_h((v8i16)in, 0);   \
  ILVRL_H2_SW(tmp_m, in, out0, out1);                 \
} while (0)

/* Description : Butterfly of 4 input vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *               Outputs - out0, out1, out2, out3
 * Details     : Butterfly operation
 */
#define BUTTERFLY_4(in0, in1, in2, in3, out0, out1, out2, out3) do {  \
  out0 = in0 + in3;                                                   \
  out1 = in1 + in2;                                                   \
  out2 = in1 - in2;                                                   \
  out3 = in0 - in3;                                                   \
} while (0)

/* Description : Transpose 16x4 block into 4x16 with byte elements in vectors
 * Arguments   : Inputs  - in0, in1, in2, in3, in4, in5, in6, in7,
 *                         in8, in9, in10, in11, in12, in13, in14, in15
 *               Outputs - out0, out1, out2, out3
 *               Return Type - unsigned byte
 */
#define TRANSPOSE16x4_UB_UB(in0, in1, in2, in3, in4, in5, in6, in7,        \
                            in8, in9, in10, in11, in12, in13, in14, in15,  \
                            out0, out1, out2, out3) do {                   \
  v2i64 tmp0_m, tmp1_m, tmp2_m, tmp3_m, tmp4_m, tmp5_m;                    \
  ILVEV_W2_SD(in0, in4, in8, in12, tmp2_m, tmp3_m);                        \
  ILVEV_W2_SD(in1, in5, in9, in13, tmp0_m, tmp1_m);                        \
  ILVEV_D2_UB(tmp2_m, tmp3_m, tmp0_m, tmp1_m, out1, out3);                 \
  ILVEV_W2_SD(in2, in6, in10, in14, tmp4_m, tmp5_m);                       \
  ILVEV_W2_SD(in3, in7, in11, in15, tmp0_m, tmp1_m);                       \
  ILVEV_D2_SD(tmp4_m, tmp5_m, tmp0_m, tmp1_m, tmp2_m, tmp3_m);             \
  ILVEV_B2_SD(out1, out3, tmp2_m, tmp3_m, tmp0_m, tmp1_m);                 \
  ILVEVOD_H2_UB(tmp0_m, tmp1_m, tmp0_m, tmp1_m, out0, out2);               \
  ILVOD_B2_SD(out1, out3, tmp2_m, tmp3_m, tmp0_m, tmp1_m);                 \
  ILVEVOD_H2_UB(tmp0_m, tmp1_m, tmp0_m, tmp1_m, out1, out3);               \
} while (0)

/* Description : Transpose 16x8 block into 8x16 with byte elements in vectors
 * Arguments   : Inputs  - in0, in1, in2, in3, in4, in5, in6, in7,
 *                         in8, in9, in10, in11, in12, in13, in14, in15
 *               Outputs - out0, out1, out2, out3, out4, out5, out6, out7
 *               Return Type - unsigned byte
 */
#define TRANSPOSE16x8_UB_UB(in0, in1, in2, in3, in4, in5, in6, in7,        \
                            in8, in9, in10, in11, in12, in13, in14, in15,  \
                            out0, out1, out2, out3, out4, out5,            \
                            out6, out7) do {                               \
  v8i16 tmp0_m, tmp1_m, tmp4_m, tmp5_m, tmp6_m, tmp7_m;                    \
  v4i32 tmp2_m, tmp3_m;                                                    \
  ILVEV_D2_UB(in0, in8, in1, in9, out7, out6);                             \
  ILVEV_D2_UB(in2, in10, in3, in11, out5, out4);                           \
  ILVEV_D2_UB(in4, in12, in5, in13, out3, out2);                           \
  ILVEV_D2_UB(in6, in14, in7, in15, out1, out0);                           \
  ILVEV_B2_SH(out7, out6, out5, out4, tmp0_m, tmp1_m);                     \
  ILVOD_B2_SH(out7, out6, out5, out4, tmp4_m, tmp5_m);                     \
  ILVEV_B2_UB(out3, out2, out1, out0, out5, out7);                         \
  ILVOD_B2_SH(out3, out2, out1, out0, tmp6_m, tmp7_m);                     \
  ILVEV_H2_SW(tmp0_m, tmp1_m, out5, out7, tmp2_m, tmp3_m);                 \
  ILVEVOD_W2_UB(tmp2_m, tmp3_m, tmp2_m, tmp3_m, out0, out4);               \
  ILVOD_H2_SW(tmp0_m, tmp1_m, out5, out7, tmp2_m, tmp3_m);                 \
  ILVEVOD_W2_UB(tmp2_m, tmp3_m, tmp2_m, tmp3_m, out2, out6);               \
  ILVEV_H2_SW(tmp4_m, tmp5_m, tmp6_m, tmp7_m, tmp2_m, tmp3_m);             \
  ILVEVOD_W2_UB(tmp2_m, tmp3_m, tmp2_m, tmp3_m, out1, out5);               \
  ILVOD_H2_SW(tmp4_m, tmp5_m, tmp6_m, tmp7_m, tmp2_m, tmp3_m);             \
  ILVEVOD_W2_UB(tmp2_m, tmp3_m, tmp2_m, tmp3_m, out3, out7);               \
} while (0)

/* Description : Transpose 4x4 block with word elements in vectors
 * Arguments   : Inputs  - in0, in1, in2, in3
 *                Outputs - out0, out1, out2, out3
 *                Return Type - as per RTYPE
 */
#define TRANSPOSE4x4_W(RTYPE, in0, in1, in2, in3,                            \
                       out0, out1, out2, out3) do {                          \
  v4i32 s0_m, s1_m, s2_m, s3_m;                                              \
  ILVRL_W2_SW(in1, in0, s0_m, s1_m);                                         \
  ILVRL_W2_SW(in3, in2, s2_m, s3_m);                                         \
  out0 = (RTYPE)__msa_ilvr_d((v2i64)s2_m, (v2i64)s0_m);                      \
  out1 = (RTYPE)__msa_ilvl_d((v2i64)s2_m, (v2i64)s0_m);                      \
  out2 = (RTYPE)__msa_ilvr_d((v2i64)s3_m, (v2i64)s1_m);                      \
  out3 = (RTYPE)__msa_ilvl_d((v2i64)s3_m, (v2i64)s1_m);                      \
} while (0)
#define TRANSPOSE4x4_SW_SW(...) TRANSPOSE4x4_W(v4i32, __VA_ARGS__)

/* Description : Add block 4x4
 * Arguments   : Inputs - in0, in1, in2, in3, pdst, stride
 * Details     : Least significant 4 bytes from each input vector are added to
 *               the destination bytes, clipped between 0-255 and stored.
 */
#define ADDBLK_ST4x4_UB(in0, in1, in2, in3, pdst, stride) do {  \
  uint32_t src0_m, src1_m, src2_m, src3_m;                      \
  v8i16 inp0_m, inp1_m, res0_m, res1_m;                         \
  v16i8 dst0_m = { 0 };                                         \
  v16i8 dst1_m = { 0 };                                         \
  const v16i8 zero_m = { 0 };                                   \
  ILVR_D2_SH(in1, in0, in3, in2, inp0_m, inp1_m);               \
  LW4(pdst, stride, src0_m, src1_m, src2_m, src3_m);            \
  INSERT_W2_SB(src0_m, src1_m, dst0_m);                         \
  INSERT_W2_SB(src2_m, src3_m, dst1_m);                         \
  ILVR_B2_SH(zero_m, dst0_m, zero_m, dst1_m, res0_m, res1_m);   \
  ADD2(res0_m, inp0_m, res1_m, inp1_m, res0_m, res1_m);         \
  CLIP_SH2_0_255(res0_m, res1_m);                               \
  PCKEV_B2_SB(res0_m, res0_m, res1_m, res1_m, dst0_m, dst1_m);  \
  ST4x4_UB(dst0_m, dst1_m, 0, 1, 0, 1, pdst, stride);           \
} while (0)

/* Description : Pack even byte elements, extract 0 & 2 index words from pair
 *               of results and store 4 words in destination memory as per
 *               stride
 * Arguments   : Inputs - in0, in1, in2, in3, pdst, stride
 */
#define PCKEV_ST4x4_UB(in0, in1, in2, in3, pdst, stride) do {  \
  v16i8 tmp0_m, tmp1_m;                                        \
  PCKEV_B2_SB(in1, in0, in3, in2, tmp0_m, tmp1_m);             \
  ST4x4_UB(tmp0_m, tmp1_m, 0, 2, 0, 2, pdst, stride);          \
} while (0)

/* Description : average with rounding (in0 + in1 + 1) / 2.
 * Arguments   : Inputs  - in0, in1, in2, in3,
 *               Outputs - out0, out1
 *               Return Type - as per RTYPE
 * Details     : Each unsigned byte element from 'in0' vector is added with
 *               each unsigned byte element from 'in1' vector. Then the average
 *               with rounding is calculated and written to 'out0'
 */
#define AVER_UB2(RTYPE, in0, in1, in2, in3, out0, out1) do {  \
  out0 = (RTYPE)__msa_aver_u_b((v16u8)in0, (v16u8)in1);       \
  out1 = (RTYPE)__msa_aver_u_b((v16u8)in2, (v16u8)in3);       \
} while (0)
#define AVER_UB2_UB(...) AVER_UB2(v16u8, __VA_ARGS__)

#endif  /* WEBP_DSP_MSA_MACRO_H_ */
