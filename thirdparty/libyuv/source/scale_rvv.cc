/*
 *  Copyright 2023 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/*
 * Copyright (c) 2023 SiFive, Inc. All rights reserved.
 *
 * Contributed by Darren Hsieh <darren.hsieh@sifive.com>
 * Contributed by Bruce Lai <bruce.lai@sifive.com>
 */

#include "libyuv/row.h"
#include "libyuv/scale_row.h"

// This module is for clang rvv. GCC hasn't supported segment load & store.
#if !defined(LIBYUV_DISABLE_RVV) && defined(__riscv_vector) && \
    defined(__clang__)
#include <assert.h>
#include <riscv_vector.h>
#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#ifdef HAS_SCALEARGBFILTERCOLS_RVV
void ScaleARGBFilterCols_RVV(uint8_t* dst_argb,
                             const uint8_t* src_argb,
                             int dst_width,
                             int x,
                             int dx) {
  assert(x >= 0);

  size_t vl = __riscv_vsetvl_e32m4(dst_width);
  vuint32m4_t vx = __riscv_vmv_v_x_u32m4(x, vl);
  vx = __riscv_vmacc_vx_u32m4(vx, dx, __riscv_vid_v_u32m4(vl), vl);
  do {
    vuint32m4_t v0_argb, v1_argb;
    vuint32m4_t v_xf0_u32, v_xf1_u32;
    vuint8m4_t v0_argb_u8, v1_argb_u8, v_xf0_u8, v_xf1_u8;
    vuint16m8_t _v0_argb_u16, v_row_u16;
    // idx is x >> 16
    vuint32m4_t v_xi_bindex = __riscv_vsrl_vx_u32m4(vx, 14, vl);
    v_xi_bindex = __riscv_vand_vx_u32m4(v_xi_bindex, ~3u, vl);
    // Read Packed ARGB w/ byte index.
    __riscv_vluxseg2ei32_v_u32m4(&v0_argb, &v1_argb, (const uint32_t*)src_argb,
                                 v_xi_bindex, vl);
    // xf = (x >> 9) & 0x7f;
    v_xf0_u32 = __riscv_vsrl_vx_u32m4(vx, 9, vl);
    v_xf0_u32 = __riscv_vand_vx_u32m4(v_xf0_u32, 0x7f, vl);
    vx = __riscv_vadd_vx_u32m4(vx, vl * dx, vl);
    // duplicate v_xf0_u32[i] from {0,0,0,f[i]} to {f[i],f[i],f[i],f[i]}
    v_xf0_u32 = __riscv_vmul_vx_u32m4(v_xf0_u32, 0x01010101, vl);
    // TODO(fbarchard): Replace 0x7f ^ f with 128-f.  bug=607.
    v_xf1_u32 = __riscv_vxor_vx_u32m4(v_xf0_u32, 0x7f7f7f7f, vl);

    v0_argb_u8 = __riscv_vreinterpret_v_u32m4_u8m4(v0_argb);
    v1_argb_u8 = __riscv_vreinterpret_v_u32m4_u8m4(v1_argb);
    v_xf0_u8 = __riscv_vreinterpret_v_u32m4_u8m4(v_xf0_u32);
    v_xf1_u8 = __riscv_vreinterpret_v_u32m4_u8m4(v_xf1_u32);
    // ((a) * (0x7f ^ f) + (b)*f) >> 7
    _v0_argb_u16 = __riscv_vwmulu_vv_u16m8(v0_argb_u8, v_xf1_u8, 4 * vl);
    v_row_u16 =
        __riscv_vwmaccu_vv_u16m8(_v0_argb_u16, v1_argb_u8, v_xf0_u8, 4 * vl);

    __riscv_vse8_v_u8m4(dst_argb, __riscv_vnsrl_wx_u8m4(v_row_u16, 7, 4 * vl),
                        4 * vl);
    dst_width -= vl;
    dst_argb += 4 * vl;
    vl = __riscv_vsetvl_e32m4(dst_width);
  } while (dst_width > 0);
}
#endif

#ifdef HAS_SCALEADDROW_RVV
void ScaleAddRow_RVV(const uint8_t* src_ptr, uint16_t* dst_ptr, int src_width) {
  size_t w = (size_t)src_width;
  do {
    size_t vl = __riscv_vsetvl_e8m4(w);
    vuint8m4_t v_src = __riscv_vle8_v_u8m4(src_ptr, vl);
    vuint16m8_t v_dst = __riscv_vle16_v_u16m8(dst_ptr, vl);
    // Use widening multiply-add instead of widening + add
    v_dst = __riscv_vwmaccu_vx_u16m8(v_dst, 1, v_src, vl);
    __riscv_vse16_v_u16m8(dst_ptr, v_dst, vl);
    w -= vl;
    src_ptr += vl;
    dst_ptr += vl;
  } while (w > 0);
}
#endif

#ifdef HAS_SCALEARGBROWDOWN2_RVV
// TODO: Reimplement similar to linear with vlseg2 so u64 is not required
void ScaleARGBRowDown2_RVV(const uint8_t* src_argb,
                           ptrdiff_t src_stride,
                           uint8_t* dst_argb,
                           int dst_width) {
  (void)src_stride;
  size_t w = (size_t)dst_width;
  const uint64_t* src = (const uint64_t*)(src_argb);
  uint32_t* dst = (uint32_t*)(dst_argb);
  do {
    size_t vl = __riscv_vsetvl_e64m8(w);
    vuint64m8_t v_data = __riscv_vle64_v_u64m8(src, vl);
    vuint32m4_t v_dst = __riscv_vnsrl_wx_u32m4(v_data, 32, vl);
    __riscv_vse32_v_u32m4(dst, v_dst, vl);
    w -= vl;
    src += vl;
    dst += vl;
  } while (w > 0);
}
#endif

#ifdef HAS_SCALEARGBROWDOWN2LINEAR_RVV
#if defined(LIBYUV_RVV_HAS_TUPLE_TYPE) && defined(LIBYUV_RVV_HAS_VXRM_ARG)
void ScaleARGBRowDown2Linear_RVV(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_argb,
                                 int dst_width) {
  (void)src_stride;
  size_t w = (size_t)dst_width;
  const uint32_t* src = (const uint32_t*)(src_argb);
  do {
    size_t vl = __riscv_vsetvl_e32m4(w);
    vuint32m4x2_t v_src = __riscv_vlseg2e32_v_u32m4x2(src, vl);
    vuint32m4_t v_even_32 = __riscv_vget_v_u32m4x2_u32m4(v_src, 0);
    vuint32m4_t v_odd_32 = __riscv_vget_v_u32m4x2_u32m4(v_src, 1);
    vuint8m4_t v_even = __riscv_vreinterpret_v_u32m4_u8m4(v_even_32);
    vuint8m4_t v_odd = __riscv_vreinterpret_v_u32m4_u8m4(v_odd_32);
    vuint8m4_t v_dst =
        __riscv_vaaddu_vv_u8m4(v_even, v_odd, __RISCV_VXRM_RNU, vl * 4);
    __riscv_vse8_v_u8m4(dst_argb, v_dst, vl * 4);
    w -= vl;
    src += vl * 2;
    dst_argb += vl * 4;
  } while (w > 0);
}
#else
void ScaleARGBRowDown2Linear_RVV(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_argb,
                                 int dst_width) {
  (void)src_stride;
  size_t w = (size_t)dst_width;
  const uint32_t* src = (const uint32_t*)(src_argb);
  // NOTE: To match behavior on other platforms, vxrm (fixed-point rounding mode
  // register) is set to round-to-nearest-up mode(0).
  asm volatile("csrwi vxrm, 0");
  do {
    vuint8m4_t v_odd, v_even, v_dst;
    vuint32m4_t v_odd_32, v_even_32;
    size_t vl = __riscv_vsetvl_e32m4(w);
    __riscv_vlseg2e32_v_u32m4(&v_even_32, &v_odd_32, src, vl);
    v_even = __riscv_vreinterpret_v_u32m4_u8m4(v_even_32);
    v_odd = __riscv_vreinterpret_v_u32m4_u8m4(v_odd_32);
    // Use round-to-nearest-up mode for averaging add
    v_dst = __riscv_vaaddu_vv_u8m4(v_even, v_odd, vl * 4);
    __riscv_vse8_v_u8m4(dst_argb, v_dst, vl * 4);
    w -= vl;
    src += vl * 2;
    dst_argb += vl * 4;
  } while (w > 0);
}
#endif
#endif

#ifdef HAS_SCALEARGBROWDOWN2BOX_RVV
#if defined(LIBYUV_RVV_HAS_TUPLE_TYPE) && defined(LIBYUV_RVV_HAS_VXRM_ARG)
void ScaleARGBRowDown2Box_RVV(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              uint8_t* dst_argb,
                              int dst_width) {
  size_t w = (size_t)dst_width;
  const uint32_t* src0 = (const uint32_t*)(src_argb);
  const uint32_t* src1 = (const uint32_t*)(src_argb + src_stride);
  do {
    size_t vl = __riscv_vsetvl_e32m4(w);
    vuint32m4x2_t v_src0 = __riscv_vlseg2e32_v_u32m4x2(src0, vl);
    vuint32m4x2_t v_src1 = __riscv_vlseg2e32_v_u32m4x2(src1, vl);
    vuint32m4_t v_row0_even_32 = __riscv_vget_v_u32m4x2_u32m4(v_src0, 0);
    vuint32m4_t v_row0_odd_32 = __riscv_vget_v_u32m4x2_u32m4(v_src0, 1);
    vuint32m4_t v_row1_even_32 = __riscv_vget_v_u32m4x2_u32m4(v_src1, 0);
    vuint32m4_t v_row1_odd_32 = __riscv_vget_v_u32m4x2_u32m4(v_src1, 1);
    vuint8m4_t v_row0_even = __riscv_vreinterpret_v_u32m4_u8m4(v_row0_even_32);
    vuint8m4_t v_row0_odd = __riscv_vreinterpret_v_u32m4_u8m4(v_row0_odd_32);
    vuint8m4_t v_row1_even = __riscv_vreinterpret_v_u32m4_u8m4(v_row1_even_32);
    vuint8m4_t v_row1_odd = __riscv_vreinterpret_v_u32m4_u8m4(v_row1_odd_32);
    vuint16m8_t v_row0_sum =
        __riscv_vwaddu_vv_u16m8(v_row0_even, v_row0_odd, vl * 4);
    vuint16m8_t v_row1_sum =
        __riscv_vwaddu_vv_u16m8(v_row1_even, v_row1_odd, vl * 4);
    vuint16m8_t v_dst_16 =
        __riscv_vadd_vv_u16m8(v_row0_sum, v_row1_sum, vl * 4);
    vuint8m4_t v_dst =
        __riscv_vnclipu_wx_u8m4(v_dst_16, 2, __RISCV_VXRM_RNU, vl * 4);
    __riscv_vse8_v_u8m4(dst_argb, v_dst, vl * 4);
    w -= vl;
    src0 += vl * 2;
    src1 += vl * 2;
    dst_argb += vl * 4;
  } while (w > 0);
}
#else
void ScaleARGBRowDown2Box_RVV(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              uint8_t* dst_argb,
                              int dst_width) {
  size_t w = (size_t)dst_width;
  const uint32_t* src0 = (const uint32_t*)(src_argb);
  const uint32_t* src1 = (const uint32_t*)(src_argb + src_stride);
  // NOTE: To match behavior on other platforms, vxrm (fixed-point rounding mode
  // register) is set to round-to-nearest-up mode(0).
  asm volatile("csrwi vxrm, 0");
  do {
    vuint8m4_t v_row0_odd, v_row0_even, v_row1_odd, v_row1_even, v_dst;
    vuint16m8_t v_row0_sum, v_row1_sum, v_dst_16;
    vuint32m4_t v_row0_odd_32, v_row0_even_32, v_row1_odd_32, v_row1_even_32;
    size_t vl = __riscv_vsetvl_e32m4(w);
    __riscv_vlseg2e32_v_u32m4(&v_row0_even_32, &v_row0_odd_32, src0, vl);
    __riscv_vlseg2e32_v_u32m4(&v_row1_even_32, &v_row1_odd_32, src1, vl);
    v_row0_even = __riscv_vreinterpret_v_u32m4_u8m4(v_row0_even_32);
    v_row0_odd = __riscv_vreinterpret_v_u32m4_u8m4(v_row0_odd_32);
    v_row1_even = __riscv_vreinterpret_v_u32m4_u8m4(v_row1_even_32);
    v_row1_odd = __riscv_vreinterpret_v_u32m4_u8m4(v_row1_odd_32);
    v_row0_sum = __riscv_vwaddu_vv_u16m8(v_row0_even, v_row0_odd, vl * 4);
    v_row1_sum = __riscv_vwaddu_vv_u16m8(v_row1_even, v_row1_odd, vl * 4);
    v_dst_16 = __riscv_vadd_vv_u16m8(v_row0_sum, v_row1_sum, vl * 4);
    // Use round-to-nearest-up mode for vnclip
    v_dst = __riscv_vnclipu_wx_u8m4(v_dst_16, 2, vl * 4);
    __riscv_vse8_v_u8m4(dst_argb, v_dst, vl * 4);
    w -= vl;
    src0 += vl * 2;
    src1 += vl * 2;
    dst_argb += vl * 4;
  } while (w > 0);
}
#endif
#endif

#ifdef HAS_SCALEARGBROWDOWNEVEN_RVV
void ScaleARGBRowDownEven_RVV(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              int src_stepx,
                              uint8_t* dst_argb,
                              int dst_width) {
  size_t w = (size_t)dst_width;
  const uint32_t* src = (const uint32_t*)(src_argb);
  uint32_t* dst = (uint32_t*)(dst_argb);
  const int stride_byte = src_stepx * 4;
  do {
    size_t vl = __riscv_vsetvl_e32m8(w);
    vuint32m8_t v_row = __riscv_vlse32_v_u32m8(src, stride_byte, vl);
    __riscv_vse32_v_u32m8(dst, v_row, vl);
    w -= vl;
    src += vl * src_stepx;
    dst += vl;
  } while (w > 0);
}
#endif

#ifdef HAS_SCALEARGBROWDOWNEVENBOX_RVV
#if defined(LIBYUV_RVV_HAS_TUPLE_TYPE) && defined(LIBYUV_RVV_HAS_VXRM_ARG)
void ScaleARGBRowDownEvenBox_RVV(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 int src_stepx,
                                 uint8_t* dst_argb,
                                 int dst_width) {
  size_t w = (size_t)dst_width;
  const uint32_t* src0 = (const uint32_t*)(src_argb);
  const uint32_t* src1 = (const uint32_t*)(src_argb + src_stride);
  const int stride_byte = src_stepx * 4;
  do {
    size_t vl = __riscv_vsetvl_e32m4(w);
    vuint32m4x2_t v_src0 = __riscv_vlsseg2e32_v_u32m4x2(src0, stride_byte, vl);
    vuint32m4x2_t v_src1 = __riscv_vlsseg2e32_v_u32m4x2(src1, stride_byte, vl);
    vuint32m4_t v_row0_low_32 = __riscv_vget_v_u32m4x2_u32m4(v_src0, 0);
    vuint32m4_t v_row0_high_32 = __riscv_vget_v_u32m4x2_u32m4(v_src0, 1);
    vuint32m4_t v_row1_low_32 = __riscv_vget_v_u32m4x2_u32m4(v_src1, 0);
    vuint32m4_t v_row1_high_32 = __riscv_vget_v_u32m4x2_u32m4(v_src1, 1);
    vuint8m4_t v_row0_low = __riscv_vreinterpret_v_u32m4_u8m4(v_row0_low_32);
    vuint8m4_t v_row0_high = __riscv_vreinterpret_v_u32m4_u8m4(v_row0_high_32);
    vuint8m4_t v_row1_low = __riscv_vreinterpret_v_u32m4_u8m4(v_row1_low_32);
    vuint8m4_t v_row1_high = __riscv_vreinterpret_v_u32m4_u8m4(v_row1_high_32);
    vuint16m8_t v_row0_sum =
        __riscv_vwaddu_vv_u16m8(v_row0_low, v_row0_high, vl * 4);
    vuint16m8_t v_row1_sum =
        __riscv_vwaddu_vv_u16m8(v_row1_low, v_row1_high, vl * 4);
    vuint16m8_t v_sum = __riscv_vadd_vv_u16m8(v_row0_sum, v_row1_sum, vl * 4);
    vuint8m4_t v_dst =
        __riscv_vnclipu_wx_u8m4(v_sum, 2, __RISCV_VXRM_RNU, vl * 4);
    __riscv_vse8_v_u8m4(dst_argb, v_dst, vl * 4);
    w -= vl;
    src0 += vl * src_stepx;
    src1 += vl * src_stepx;
    dst_argb += vl * 4;
  } while (w > 0);
}
#else
void ScaleARGBRowDownEvenBox_RVV(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 int src_stepx,
                                 uint8_t* dst_argb,
                                 int dst_width) {
  size_t w = (size_t)dst_width;
  const uint32_t* src0 = (const uint32_t*)(src_argb);
  const uint32_t* src1 = (const uint32_t*)(src_argb + src_stride);
  const int stride_byte = src_stepx * 4;
  // NOTE: To match behavior on other platforms, vxrm (fixed-point rounding mode
  // register) is set to round-to-nearest-up mode(0).
  asm volatile("csrwi vxrm, 0");
  do {
    vuint8m4_t v_row0_low, v_row0_high, v_row1_low, v_row1_high, v_dst;
    vuint16m8_t v_row0_sum, v_row1_sum, v_sum;
    vuint32m4_t v_row0_low_32, v_row0_high_32, v_row1_low_32, v_row1_high_32;
    size_t vl = __riscv_vsetvl_e32m4(w);
    __riscv_vlsseg2e32_v_u32m4(&v_row0_low_32, &v_row0_high_32, src0,
                               stride_byte, vl);
    __riscv_vlsseg2e32_v_u32m4(&v_row1_low_32, &v_row1_high_32, src1,
                               stride_byte, vl);
    v_row0_low = __riscv_vreinterpret_v_u32m4_u8m4(v_row0_low_32);
    v_row0_high = __riscv_vreinterpret_v_u32m4_u8m4(v_row0_high_32);
    v_row1_low = __riscv_vreinterpret_v_u32m4_u8m4(v_row1_low_32);
    v_row1_high = __riscv_vreinterpret_v_u32m4_u8m4(v_row1_high_32);
    v_row0_sum = __riscv_vwaddu_vv_u16m8(v_row0_low, v_row0_high, vl * 4);
    v_row1_sum = __riscv_vwaddu_vv_u16m8(v_row1_low, v_row1_high, vl * 4);
    v_sum = __riscv_vadd_vv_u16m8(v_row0_sum, v_row1_sum, vl * 4);
    // Use round-to-nearest-up mode for vnclip
    v_dst = __riscv_vnclipu_wx_u8m4(v_sum, 2, vl * 4);
    __riscv_vse8_v_u8m4(dst_argb, v_dst, vl * 4);
    w -= vl;
    src0 += vl * src_stepx;
    src1 += vl * src_stepx;
    dst_argb += vl * 4;
  } while (w > 0);
}
#endif
#endif

#ifdef HAS_SCALEROWDOWN2_RVV
void ScaleRowDown2_RVV(const uint8_t* src_ptr,
                       ptrdiff_t src_stride,
                       uint8_t* dst,
                       int dst_width) {
  size_t w = (size_t)dst_width;
  const uint16_t* src = (const uint16_t*)src_ptr;
  (void)src_stride;
  do {
    size_t vl = __riscv_vsetvl_e16m8(w);
    vuint16m8_t v_src = __riscv_vle16_v_u16m8(src, vl);
    vuint8m4_t v_dst = __riscv_vnsrl_wx_u8m4(v_src, 8, vl);
    __riscv_vse8_v_u8m4(dst, v_dst, vl);
    w -= vl;
    src += vl;
    dst += vl;
  } while (w > 0);
}
#endif

#ifdef HAS_SCALEROWDOWN2LINEAR_RVV
#if defined(LIBYUV_RVV_HAS_TUPLE_TYPE) && defined(LIBYUV_RVV_HAS_VXRM_ARG)
void ScaleRowDown2Linear_RVV(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst,
                             int dst_width) {
  size_t w = (size_t)dst_width;
  (void)src_stride;
  do {
    size_t vl = __riscv_vsetvl_e8m4(w);
    vuint8m4x2_t v_src = __riscv_vlseg2e8_v_u8m4x2(src_ptr, vl);
    vuint8m4_t v_s0 = __riscv_vget_v_u8m4x2_u8m4(v_src, 0);
    vuint8m4_t v_s1 = __riscv_vget_v_u8m4x2_u8m4(v_src, 1);
    vuint8m4_t v_dst = __riscv_vaaddu_vv_u8m4(v_s0, v_s1, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_u8m4(dst, v_dst, vl);
    w -= vl;
    src_ptr += 2 * vl;
    dst += vl;
  } while (w > 0);
}
#else
void ScaleRowDown2Linear_RVV(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst,
                             int dst_width) {
  size_t w = (size_t)dst_width;
  (void)src_stride;
  // NOTE: To match behavior on other platforms, vxrm (fixed-point rounding mode
  // register) is set to round-to-nearest-up mode(0).
  asm volatile("csrwi vxrm, 0");
  do {
    vuint8m4_t v_s0, v_s1, v_dst;
    size_t vl = __riscv_vsetvl_e8m4(w);
    __riscv_vlseg2e8_v_u8m4(&v_s0, &v_s1, src_ptr, vl);
    // Use round-to-nearest-up mode for averaging add
    v_dst = __riscv_vaaddu_vv_u8m4(v_s0, v_s1, vl);
    __riscv_vse8_v_u8m4(dst, v_dst, vl);
    w -= vl;
    src_ptr += 2 * vl;
    dst += vl;
  } while (w > 0);
}
#endif
#endif

#ifdef HAS_SCALEROWDOWN2BOX_RVV
#if defined(LIBYUV_RVV_HAS_TUPLE_TYPE) && defined(LIBYUV_RVV_HAS_VXRM_ARG)
void ScaleRowDown2Box_RVV(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst,
                          int dst_width) {
  const uint8_t* s = src_ptr;
  const uint8_t* t = src_ptr + src_stride;
  size_t w = (size_t)dst_width;
  do {
    size_t vl = __riscv_vsetvl_e8m4(w);
    vuint8m4x2_t v_s = __riscv_vlseg2e8_v_u8m4x2(s, vl);
    vuint8m4x2_t v_t = __riscv_vlseg2e8_v_u8m4x2(t, vl);
    vuint8m4_t v_s0 = __riscv_vget_v_u8m4x2_u8m4(v_s, 0);
    vuint8m4_t v_s1 = __riscv_vget_v_u8m4x2_u8m4(v_s, 1);
    vuint8m4_t v_t0 = __riscv_vget_v_u8m4x2_u8m4(v_t, 0);
    vuint8m4_t v_t1 = __riscv_vget_v_u8m4x2_u8m4(v_t, 1);
    vuint16m8_t v_s01 = __riscv_vwaddu_vv_u16m8(v_s0, v_s1, vl);
    vuint16m8_t v_t01 = __riscv_vwaddu_vv_u16m8(v_t0, v_t1, vl);
    vuint16m8_t v_st01 = __riscv_vadd_vv_u16m8(v_s01, v_t01, vl);
    // Use round-to-nearest-up mode for vnclip
    vuint8m4_t v_dst = __riscv_vnclipu_wx_u8m4(v_st01, 2, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_u8m4(dst, v_dst, vl);
    w -= vl;
    s += 2 * vl;
    t += 2 * vl;
    dst += vl;
  } while (w > 0);
}
#else
void ScaleRowDown2Box_RVV(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst,
                          int dst_width) {
  const uint8_t* s = src_ptr;
  const uint8_t* t = src_ptr + src_stride;
  size_t w = (size_t)dst_width;
  // NOTE: To match behavior on other platforms, vxrm (fixed-point rounding mode
  // register) is set to round-to-nearest-up mode(0).
  asm volatile("csrwi vxrm, 0");
  do {
    size_t vl = __riscv_vsetvl_e8m4(w);
    vuint8m4_t v_s0, v_s1, v_t0, v_t1;
    vuint16m8_t v_s01, v_t01, v_st01;
    vuint8m4_t v_dst;
    __riscv_vlseg2e8_v_u8m4(&v_s0, &v_s1, s, vl);
    __riscv_vlseg2e8_v_u8m4(&v_t0, &v_t1, t, vl);
    v_s01 = __riscv_vwaddu_vv_u16m8(v_s0, v_s1, vl);
    v_t01 = __riscv_vwaddu_vv_u16m8(v_t0, v_t1, vl);
    v_st01 = __riscv_vadd_vv_u16m8(v_s01, v_t01, vl);
    // Use round-to-nearest-up mode for vnclip
    v_dst = __riscv_vnclipu_wx_u8m4(v_st01, 2, vl);
    __riscv_vse8_v_u8m4(dst, v_dst, vl);
    w -= vl;
    s += 2 * vl;
    t += 2 * vl;
    dst += vl;
  } while (w > 0);
}
#endif
#endif

#ifdef HAS_SCALEROWDOWN4_RVV
#ifdef LIBYUV_RVV_HAS_TUPLE_TYPE
void ScaleRowDown4_RVV(const uint8_t* src_ptr,
                       ptrdiff_t src_stride,
                       uint8_t* dst_ptr,
                       int dst_width) {
  size_t w = (size_t)dst_width;
  (void)src_stride;
  do {
    size_t vl = __riscv_vsetvl_e8m2(w);
    vuint8m2x4_t v_s = __riscv_vlseg4e8_v_u8m2x4(src_ptr, vl);
    vuint8m2_t v_s2 = __riscv_vget_v_u8m2x4_u8m2(v_s, 2);
    __riscv_vse8_v_u8m2(dst_ptr, v_s2, vl);
    w -= vl;
    src_ptr += (4 * vl);
    dst_ptr += vl;
  } while (w > 0);
}
#else
void ScaleRowDown4_RVV(const uint8_t* src_ptr,
                       ptrdiff_t src_stride,
                       uint8_t* dst_ptr,
                       int dst_width) {
  size_t w = (size_t)dst_width;
  (void)src_stride;
  do {
    size_t vl = __riscv_vsetvl_e8m2(w);
    vuint8m2_t v_s0, v_s1, v_s2, v_s3;
    __riscv_vlseg4e8_v_u8m2(&v_s0, &v_s1, &v_s2, &v_s3, src_ptr, vl);
    __riscv_vse8_v_u8m2(dst_ptr, v_s2, vl);
    w -= vl;
    src_ptr += (4 * vl);
    dst_ptr += vl;
  } while (w > 0);
}
#endif
#endif

#ifdef HAS_SCALEROWDOWN4BOX_RVV
#if defined(LIBYUV_RVV_HAS_TUPLE_TYPE) && defined(LIBYUV_RVV_HAS_VXRM_ARG)
void ScaleRowDown4Box_RVV(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst_ptr,
                          int dst_width) {
  const uint8_t* src_ptr1 = src_ptr + src_stride;
  const uint8_t* src_ptr2 = src_ptr + src_stride * 2;
  const uint8_t* src_ptr3 = src_ptr + src_stride * 3;
  size_t w = (size_t)dst_width;
  do {
    size_t vl = __riscv_vsetvl_e8m2(w);
    vuint8m2x4_t v_s = __riscv_vlseg4e8_v_u8m2x4(src_ptr, vl);
    vuint8m2_t v_s0 = __riscv_vget_v_u8m2x4_u8m2(v_s, 0);
    vuint8m2_t v_s1 = __riscv_vget_v_u8m2x4_u8m2(v_s, 1);
    vuint8m2_t v_s2 = __riscv_vget_v_u8m2x4_u8m2(v_s, 2);
    vuint8m2_t v_s3 = __riscv_vget_v_u8m2x4_u8m2(v_s, 3);
    vuint16m4_t v_s01 = __riscv_vwaddu_vv_u16m4(v_s0, v_s1, vl);
    vuint8m2x4_t v_t = __riscv_vlseg4e8_v_u8m2x4(src_ptr1, vl);
    vuint8m2_t v_t0 = __riscv_vget_v_u8m2x4_u8m2(v_t, 0);
    vuint8m2_t v_t1 = __riscv_vget_v_u8m2x4_u8m2(v_t, 1);
    vuint8m2_t v_t2 = __riscv_vget_v_u8m2x4_u8m2(v_t, 2);
    vuint8m2_t v_t3 = __riscv_vget_v_u8m2x4_u8m2(v_t, 3);
    vuint16m4_t v_t01 = __riscv_vwaddu_vv_u16m4(v_t0, v_t1, vl);
    vuint8m2x4_t v_u = __riscv_vlseg4e8_v_u8m2x4(src_ptr2, vl);
    vuint8m2_t v_u0 = __riscv_vget_v_u8m2x4_u8m2(v_u, 0);
    vuint8m2_t v_u1 = __riscv_vget_v_u8m2x4_u8m2(v_u, 1);
    vuint8m2_t v_u2 = __riscv_vget_v_u8m2x4_u8m2(v_u, 2);
    vuint8m2_t v_u3 = __riscv_vget_v_u8m2x4_u8m2(v_u, 3);
    vuint16m4_t v_u01 = __riscv_vwaddu_vv_u16m4(v_u0, v_u1, vl);
    vuint16m4_t v_u23 = __riscv_vwaddu_vv_u16m4(v_u2, v_u3, vl);
    vuint16m4_t v_s23 = __riscv_vwaddu_vv_u16m4(v_s2, v_s3, vl);
    vuint16m4_t v_t23 = __riscv_vwaddu_vv_u16m4(v_t2, v_t3, vl);
    vuint16m4_t v_st01 = __riscv_vadd_vv_u16m4(v_s01, v_t01, vl);
    vuint16m4_t v_st23 = __riscv_vadd_vv_u16m4(v_s23, v_t23, vl);
    vuint8m2x4_t v_v = __riscv_vlseg4e8_v_u8m2x4(src_ptr3, vl);
    vuint8m2_t v_v0 = __riscv_vget_v_u8m2x4_u8m2(v_v, 0);
    vuint8m2_t v_v1 = __riscv_vget_v_u8m2x4_u8m2(v_v, 1);
    vuint8m2_t v_v2 = __riscv_vget_v_u8m2x4_u8m2(v_v, 2);
    vuint8m2_t v_v3 = __riscv_vget_v_u8m2x4_u8m2(v_v, 3);

    vuint16m4_t v_v01 = __riscv_vwaddu_vv_u16m4(v_v0, v_v1, vl);
    vuint16m4_t v_v23 = __riscv_vwaddu_vv_u16m4(v_v2, v_v3, vl);

    vuint16m4_t v_uv01 = __riscv_vadd_vv_u16m4(v_u01, v_v01, vl);
    vuint16m4_t v_uv23 = __riscv_vadd_vv_u16m4(v_u23, v_v23, vl);

    vuint16m4_t v_st0123 = __riscv_vadd_vv_u16m4(v_st01, v_st23, vl);
    vuint16m4_t v_uv0123 = __riscv_vadd_vv_u16m4(v_uv01, v_uv23, vl);
    vuint16m4_t v_stuv0123 = __riscv_vadd_vv_u16m4(v_st0123, v_uv0123, vl);
    vuint8m2_t v_dst =
        __riscv_vnclipu_wx_u8m2(v_stuv0123, 4, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_u8m2(dst_ptr, v_dst, vl);
    w -= vl;
    src_ptr += 4 * vl;
    src_ptr1 += 4 * vl;
    src_ptr2 += 4 * vl;
    src_ptr3 += 4 * vl;
    dst_ptr += vl;
  } while (w > 0);
}
#else
void ScaleRowDown4Box_RVV(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst_ptr,
                          int dst_width) {
  const uint8_t* src_ptr1 = src_ptr + src_stride;
  const uint8_t* src_ptr2 = src_ptr + src_stride * 2;
  const uint8_t* src_ptr3 = src_ptr + src_stride * 3;
  size_t w = (size_t)dst_width;
  // NOTE: To match behavior on other platforms, vxrm (fixed-point rounding mode
  // register) is set to round-to-nearest-up mode(0).
  asm volatile("csrwi vxrm, 0");
  do {
    vuint8m2_t v_s0, v_s1, v_s2, v_s3;
    vuint8m2_t v_t0, v_t1, v_t2, v_t3;
    vuint8m2_t v_u0, v_u1, v_u2, v_u3;
    vuint8m2_t v_v0, v_v1, v_v2, v_v3;
    vuint16m4_t v_s01, v_s23, v_t01, v_t23;
    vuint16m4_t v_u01, v_u23, v_v01, v_v23;
    vuint16m4_t v_st01, v_st23, v_uv01, v_uv23;
    vuint16m4_t v_st0123, v_uv0123, v_stuv0123;
    vuint8m2_t v_dst;
    size_t vl = __riscv_vsetvl_e8m2(w);

    __riscv_vlseg4e8_v_u8m2(&v_s0, &v_s1, &v_s2, &v_s3, src_ptr, vl);
    v_s01 = __riscv_vwaddu_vv_u16m4(v_s0, v_s1, vl);

    __riscv_vlseg4e8_v_u8m2(&v_t0, &v_t1, &v_t2, &v_t3, src_ptr1, vl);
    v_t01 = __riscv_vwaddu_vv_u16m4(v_t0, v_t1, vl);

    __riscv_vlseg4e8_v_u8m2(&v_u0, &v_u1, &v_u2, &v_u3, src_ptr2, vl);
    v_u01 = __riscv_vwaddu_vv_u16m4(v_u0, v_u1, vl);
    v_u23 = __riscv_vwaddu_vv_u16m4(v_u2, v_u3, vl);

    v_s23 = __riscv_vwaddu_vv_u16m4(v_s2, v_s3, vl);
    v_t23 = __riscv_vwaddu_vv_u16m4(v_t2, v_t3, vl);
    v_st01 = __riscv_vadd_vv_u16m4(v_s01, v_t01, vl);
    v_st23 = __riscv_vadd_vv_u16m4(v_s23, v_t23, vl);

    __riscv_vlseg4e8_v_u8m2(&v_v0, &v_v1, &v_v2, &v_v3, src_ptr3, vl);

    v_v01 = __riscv_vwaddu_vv_u16m4(v_v0, v_v1, vl);
    v_v23 = __riscv_vwaddu_vv_u16m4(v_v2, v_v3, vl);

    v_uv01 = __riscv_vadd_vv_u16m4(v_u01, v_v01, vl);
    v_uv23 = __riscv_vadd_vv_u16m4(v_u23, v_v23, vl);

    v_st0123 = __riscv_vadd_vv_u16m4(v_st01, v_st23, vl);
    v_uv0123 = __riscv_vadd_vv_u16m4(v_uv01, v_uv23, vl);
    v_stuv0123 = __riscv_vadd_vv_u16m4(v_st0123, v_uv0123, vl);
    // Use round-to-nearest-up mode for vnclip
    v_dst = __riscv_vnclipu_wx_u8m2(v_stuv0123, 4, vl);
    __riscv_vse8_v_u8m2(dst_ptr, v_dst, vl);
    w -= vl;
    src_ptr += 4 * vl;
    src_ptr1 += 4 * vl;
    src_ptr2 += 4 * vl;
    src_ptr3 += 4 * vl;
    dst_ptr += vl;
  } while (w > 0);
}
#endif
#endif

#ifdef HAS_SCALEROWDOWN34_RVV
#ifdef LIBYUV_RVV_HAS_TUPLE_TYPE
void ScaleRowDown34_RVV(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst_ptr,
                        int dst_width) {
  size_t w = (size_t)dst_width / 3u;
  do {
    size_t vl = __riscv_vsetvl_e8m2(w);
    vuint8m2x4_t v_src = __riscv_vlseg4e8_v_u8m2x4(src_ptr, vl);
    vuint8m2_t v_0 = __riscv_vget_v_u8m2x4_u8m2(v_src, 0);
    vuint8m2_t v_1 = __riscv_vget_v_u8m2x4_u8m2(v_src, 1);
    vuint8m2_t v_3 = __riscv_vget_v_u8m2x4_u8m2(v_src, 3);
    vuint8m2x3_t v_dst = __riscv_vcreate_v_u8m2x3(v_0, v_1, v_3);
    __riscv_vsseg3e8_v_u8m2x3(dst_ptr, v_dst, vl);
    w -= vl;
    src_ptr += 4 * vl;
    dst_ptr += 3 * vl;
  } while (w > 0);
}
#else
void ScaleRowDown34_RVV(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst_ptr,
                        int dst_width) {
  size_t w = (size_t)dst_width / 3u;
  do {
    size_t vl = __riscv_vsetvl_e8m2(w);
    vuint8m2_t v_s0, v_s1, v_s2, v_s3;
    __riscv_vlseg4e8_v_u8m2(&v_s0, &v_s1, &v_s2, &v_s3, src_ptr, vl);
    __riscv_vsseg3e8_v_u8m2(dst_ptr, v_s0, v_s1, v_s3, vl);
    w -= vl;
    src_ptr += 4 * vl;
    dst_ptr += 3 * vl;
  } while (w > 0);
}
#endif
#endif

#ifdef HAS_SCALEROWDOWN34_0_BOX_RVV
#if defined(LIBYUV_RVV_HAS_TUPLE_TYPE) && defined(LIBYUV_RVV_HAS_VXRM_ARG)
void ScaleRowDown34_0_Box_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  size_t w = (size_t)dst_width / 3u;
  const uint8_t* s = src_ptr;
  const uint8_t* t = src_ptr + src_stride;
  do {
    vuint16m4_t v_t0_u16, v_t1_u16, v_t2_u16, v_t3_u16;
    vuint8m2_t v_u0, v_u1, v_u2, v_u3;
    vuint16m4_t v_u1_u16;
    vuint8m2_t v_a0, v_a1, v_a2;
    vuint8m2x3_t v_dst;
    size_t vl = __riscv_vsetvl_e8m2(w);
    vuint8m2x4_t v_s = __riscv_vlseg4e8_v_u8m2x4(s, vl);
    vuint8m2_t v_s0 = __riscv_vget_v_u8m2x4_u8m2(v_s, 0);
    vuint8m2_t v_s1 = __riscv_vget_v_u8m2x4_u8m2(v_s, 1);
    vuint8m2_t v_s2 = __riscv_vget_v_u8m2x4_u8m2(v_s, 2);
    vuint8m2_t v_s3 = __riscv_vget_v_u8m2x4_u8m2(v_s, 3);

    if (src_stride == 0) {
      v_t0_u16 = __riscv_vwaddu_vx_u16m4(v_s0, 2, vl);
      v_t1_u16 = __riscv_vwaddu_vx_u16m4(v_s1, 2, vl);
      v_t2_u16 = __riscv_vwaddu_vx_u16m4(v_s2, 2, vl);
      v_t3_u16 = __riscv_vwaddu_vx_u16m4(v_s3, 2, vl);
    } else {
      vuint8m2x4_t v_t = __riscv_vlseg4e8_v_u8m2x4(t, vl);
      vuint8m2_t v_t0 = __riscv_vget_v_u8m2x4_u8m2(v_t, 0);
      vuint8m2_t v_t1 = __riscv_vget_v_u8m2x4_u8m2(v_t, 1);
      vuint8m2_t v_t2 = __riscv_vget_v_u8m2x4_u8m2(v_t, 2);
      vuint8m2_t v_t3 = __riscv_vget_v_u8m2x4_u8m2(v_t, 3);
      v_t0_u16 = __riscv_vwaddu_vx_u16m4(v_t0, 0, vl);
      v_t1_u16 = __riscv_vwaddu_vx_u16m4(v_t1, 0, vl);
      v_t2_u16 = __riscv_vwaddu_vx_u16m4(v_t2, 0, vl);
      v_t3_u16 = __riscv_vwaddu_vx_u16m4(v_t3, 0, vl);
      t += 4 * vl;
    }

    v_t0_u16 = __riscv_vwmaccu_vx_u16m4(v_t0_u16, 3, v_s0, vl);
    v_t1_u16 = __riscv_vwmaccu_vx_u16m4(v_t1_u16, 3, v_s1, vl);
    v_t2_u16 = __riscv_vwmaccu_vx_u16m4(v_t2_u16, 3, v_s2, vl);
    v_t3_u16 = __riscv_vwmaccu_vx_u16m4(v_t3_u16, 3, v_s3, vl);

    v_u0 = __riscv_vnclipu_wx_u8m2(v_t0_u16, 2, __RISCV_VXRM_RNU, vl);
    v_u1 = __riscv_vnclipu_wx_u8m2(v_t1_u16, 2, __RISCV_VXRM_RNU, vl);
    v_u2 = __riscv_vnclipu_wx_u8m2(v_t2_u16, 2, __RISCV_VXRM_RNU, vl);
    v_u3 = __riscv_vnclipu_wx_u8m2(v_t3_u16, 2, __RISCV_VXRM_RNU, vl);
    // a0 = (src[0] * 3 + s[1] * 1 + 2) >> 2
    v_u1_u16 = __riscv_vwaddu_vx_u16m4(v_u1, 0, vl);
    v_u1_u16 = __riscv_vwmaccu_vx_u16m4(v_u1_u16, 3, v_u0, vl);
    v_a0 = __riscv_vnclipu_wx_u8m2(v_u1_u16, 2, __RISCV_VXRM_RNU, vl);
    // a1 = (src[1] * 1 + s[2] * 1 + 1) >> 1
    v_a1 = __riscv_vaaddu_vv_u8m2(v_u1, v_u2, __RISCV_VXRM_RNU, vl);
    // a2 = (src[2] * 1 + s[3] * 3 + 2) >> 2
    v_u1_u16 = __riscv_vwaddu_vx_u16m4(v_u2, 0, vl);
    v_u1_u16 = __riscv_vwmaccu_vx_u16m4(v_u1_u16, 3, v_u3, vl);
    v_a2 = __riscv_vnclipu_wx_u8m2(v_u1_u16, 2, __RISCV_VXRM_RNU, vl);

    v_dst = __riscv_vcreate_v_u8m2x3(v_a0, v_a1, v_a2);
    __riscv_vsseg3e8_v_u8m2x3(dst_ptr, v_dst, vl);

    w -= vl;
    s += 4 * vl;
    dst_ptr += 3 * vl;
  } while (w > 0);
}
#else
void ScaleRowDown34_0_Box_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  size_t w = (size_t)dst_width / 3u;
  const uint8_t* s = src_ptr;
  const uint8_t* t = src_ptr + src_stride;
  // NOTE: To match behavior on other platforms, vxrm (fixed-point rounding mode
  // register) is set to round-to-nearest-up mode(0).
  asm volatile("csrwi vxrm, 0");
  do {
    vuint8m2_t v_s0, v_s1, v_s2, v_s3;
    vuint16m4_t v_t0_u16, v_t1_u16, v_t2_u16, v_t3_u16;
    vuint8m2_t v_u0, v_u1, v_u2, v_u3;
    vuint16m4_t v_u1_u16;
    vuint8m2_t v_a0, v_a1, v_a2;
    size_t vl = __riscv_vsetvl_e8m2(w);
    __riscv_vlseg4e8_v_u8m2(&v_s0, &v_s1, &v_s2, &v_s3, s, vl);

    if (src_stride == 0) {
      v_t0_u16 = __riscv_vwaddu_vx_u16m4(v_s0, 2, vl);
      v_t1_u16 = __riscv_vwaddu_vx_u16m4(v_s1, 2, vl);
      v_t2_u16 = __riscv_vwaddu_vx_u16m4(v_s2, 2, vl);
      v_t3_u16 = __riscv_vwaddu_vx_u16m4(v_s3, 2, vl);
    } else {
      vuint8m2_t v_t0, v_t1, v_t2, v_t3;
      __riscv_vlseg4e8_v_u8m2(&v_t0, &v_t1, &v_t2, &v_t3, t, vl);
      v_t0_u16 = __riscv_vwaddu_vx_u16m4(v_t0, 0, vl);
      v_t1_u16 = __riscv_vwaddu_vx_u16m4(v_t1, 0, vl);
      v_t2_u16 = __riscv_vwaddu_vx_u16m4(v_t2, 0, vl);
      v_t3_u16 = __riscv_vwaddu_vx_u16m4(v_t3, 0, vl);
      t += 4 * vl;
    }

    v_t0_u16 = __riscv_vwmaccu_vx_u16m4(v_t0_u16, 3, v_s0, vl);
    v_t1_u16 = __riscv_vwmaccu_vx_u16m4(v_t1_u16, 3, v_s1, vl);
    v_t2_u16 = __riscv_vwmaccu_vx_u16m4(v_t2_u16, 3, v_s2, vl);
    v_t3_u16 = __riscv_vwmaccu_vx_u16m4(v_t3_u16, 3, v_s3, vl);

    // Use round-to-nearest-up mode for vnclip & averaging add
    v_u0 = __riscv_vnclipu_wx_u8m2(v_t0_u16, 2, vl);
    v_u1 = __riscv_vnclipu_wx_u8m2(v_t1_u16, 2, vl);
    v_u2 = __riscv_vnclipu_wx_u8m2(v_t2_u16, 2, vl);
    v_u3 = __riscv_vnclipu_wx_u8m2(v_t3_u16, 2, vl);

    // a0 = (src[0] * 3 + s[1] * 1 + 2) >> 2
    v_u1_u16 = __riscv_vwaddu_vx_u16m4(v_u1, 0, vl);
    v_u1_u16 = __riscv_vwmaccu_vx_u16m4(v_u1_u16, 3, v_u0, vl);
    v_a0 = __riscv_vnclipu_wx_u8m2(v_u1_u16, 2, vl);

    // a1 = (src[1] * 1 + s[2] * 1 + 1) >> 1
    v_a1 = __riscv_vaaddu_vv_u8m2(v_u1, v_u2, vl);

    // a2 = (src[2] * 1 + s[3] * 3 + 2) >> 2
    v_u1_u16 = __riscv_vwaddu_vx_u16m4(v_u2, 0, vl);
    v_u1_u16 = __riscv_vwmaccu_vx_u16m4(v_u1_u16, 3, v_u3, vl);
    v_a2 = __riscv_vnclipu_wx_u8m2(v_u1_u16, 2, vl);

    __riscv_vsseg3e8_v_u8m2(dst_ptr, v_a0, v_a1, v_a2, vl);

    w -= vl;
    s += 4 * vl;
    dst_ptr += 3 * vl;
  } while (w > 0);
}
#endif
#endif

#ifdef HAS_SCALEROWDOWN34_1_BOX_RVV
#if defined(LIBYUV_RVV_HAS_TUPLE_TYPE) && defined(LIBYUV_RVV_HAS_VXRM_ARG)
void ScaleRowDown34_1_Box_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  size_t w = (size_t)dst_width / 3u;
  const uint8_t* s = src_ptr;
  const uint8_t* t = src_ptr + src_stride;
  do {
    vuint8m2_t v_ave0, v_ave1, v_ave2, v_ave3;
    vuint16m4_t v_u1_u16;
    vuint8m2_t v_a0, v_a1, v_a2;
    vuint8m2x3_t v_dst;
    size_t vl = __riscv_vsetvl_e8m2(w);
    vuint8m2x4_t v_s = __riscv_vlseg4e8_v_u8m2x4(s, vl);
    vuint8m2_t v_s0 = __riscv_vget_v_u8m2x4_u8m2(v_s, 0);
    vuint8m2_t v_s1 = __riscv_vget_v_u8m2x4_u8m2(v_s, 1);
    vuint8m2_t v_s2 = __riscv_vget_v_u8m2x4_u8m2(v_s, 2);
    vuint8m2_t v_s3 = __riscv_vget_v_u8m2x4_u8m2(v_s, 3);

    // Use round-to-nearest-up mode for vnclip & averaging add
    if (src_stride == 0) {
      v_ave0 = __riscv_vaaddu_vv_u8m2(v_s0, v_s0, __RISCV_VXRM_RNU, vl);
      v_ave1 = __riscv_vaaddu_vv_u8m2(v_s1, v_s1, __RISCV_VXRM_RNU, vl);
      v_ave2 = __riscv_vaaddu_vv_u8m2(v_s2, v_s2, __RISCV_VXRM_RNU, vl);
      v_ave3 = __riscv_vaaddu_vv_u8m2(v_s3, v_s3, __RISCV_VXRM_RNU, vl);
    } else {
      vuint8m2x4_t v_t = __riscv_vlseg4e8_v_u8m2x4(t, vl);
      vuint8m2_t v_t0 = __riscv_vget_v_u8m2x4_u8m2(v_t, 0);
      vuint8m2_t v_t1 = __riscv_vget_v_u8m2x4_u8m2(v_t, 1);
      vuint8m2_t v_t2 = __riscv_vget_v_u8m2x4_u8m2(v_t, 2);
      vuint8m2_t v_t3 = __riscv_vget_v_u8m2x4_u8m2(v_t, 3);
      v_ave0 = __riscv_vaaddu_vv_u8m2(v_s0, v_t0, __RISCV_VXRM_RNU, vl);
      v_ave1 = __riscv_vaaddu_vv_u8m2(v_s1, v_t1, __RISCV_VXRM_RNU, vl);
      v_ave2 = __riscv_vaaddu_vv_u8m2(v_s2, v_t2, __RISCV_VXRM_RNU, vl);
      v_ave3 = __riscv_vaaddu_vv_u8m2(v_s3, v_t3, __RISCV_VXRM_RNU, vl);
      t += 4 * vl;
    }
    // a0 = (src[0] * 3 + s[1] * 1 + 2) >> 2
    v_u1_u16 = __riscv_vwaddu_vx_u16m4(v_ave1, 0, vl);
    v_u1_u16 = __riscv_vwmaccu_vx_u16m4(v_u1_u16, 3, v_ave0, vl);
    v_a0 = __riscv_vnclipu_wx_u8m2(v_u1_u16, 2, __RISCV_VXRM_RNU, vl);

    // a1 = (src[1] * 1 + s[2] * 1 + 1) >> 1
    v_a1 = __riscv_vaaddu_vv_u8m2(v_ave1, v_ave2, __RISCV_VXRM_RNU, vl);

    // a2 = (src[2] * 1 + s[3] * 3 + 2) >> 2
    v_u1_u16 = __riscv_vwaddu_vx_u16m4(v_ave2, 0, vl);
    v_u1_u16 = __riscv_vwmaccu_vx_u16m4(v_u1_u16, 3, v_ave3, vl);
    v_a2 = __riscv_vnclipu_wx_u8m2(v_u1_u16, 2, __RISCV_VXRM_RNU, vl);

    v_dst = __riscv_vcreate_v_u8m2x3(v_a0, v_a1, v_a2);
    __riscv_vsseg3e8_v_u8m2x3(dst_ptr, v_dst, vl);

    w -= vl;
    s += 4 * vl;
    dst_ptr += 3 * vl;
  } while (w > 0);
}
#else
void ScaleRowDown34_1_Box_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  size_t w = (size_t)dst_width / 3u;
  const uint8_t* s = src_ptr;
  const uint8_t* t = src_ptr + src_stride;
  // NOTE: To match behavior on other platforms, vxrm (fixed-point rounding mode
  // register) is set to round-to-nearest-up mode(0).
  asm volatile("csrwi vxrm, 0");
  do {
    vuint8m2_t v_s0, v_s1, v_s2, v_s3;
    vuint8m2_t v_ave0, v_ave1, v_ave2, v_ave3;
    vuint16m4_t v_u1_u16;
    vuint8m2_t v_a0, v_a1, v_a2;
    size_t vl = __riscv_vsetvl_e8m2(w);
    __riscv_vlseg4e8_v_u8m2(&v_s0, &v_s1, &v_s2, &v_s3, s, vl);

    // Use round-to-nearest-up mode for vnclip & averaging add
    if (src_stride == 0) {
      v_ave0 = __riscv_vaaddu_vv_u8m2(v_s0, v_s0, vl);
      v_ave1 = __riscv_vaaddu_vv_u8m2(v_s1, v_s1, vl);
      v_ave2 = __riscv_vaaddu_vv_u8m2(v_s2, v_s2, vl);
      v_ave3 = __riscv_vaaddu_vv_u8m2(v_s3, v_s3, vl);
    } else {
      vuint8m2_t v_t0, v_t1, v_t2, v_t3;
      __riscv_vlseg4e8_v_u8m2(&v_t0, &v_t1, &v_t2, &v_t3, t, vl);
      v_ave0 = __riscv_vaaddu_vv_u8m2(v_s0, v_t0, vl);
      v_ave1 = __riscv_vaaddu_vv_u8m2(v_s1, v_t1, vl);
      v_ave2 = __riscv_vaaddu_vv_u8m2(v_s2, v_t2, vl);
      v_ave3 = __riscv_vaaddu_vv_u8m2(v_s3, v_t3, vl);
      t += 4 * vl;
    }
    // a0 = (src[0] * 3 + s[1] * 1 + 2) >> 2
    v_u1_u16 = __riscv_vwaddu_vx_u16m4(v_ave1, 0, vl);
    v_u1_u16 = __riscv_vwmaccu_vx_u16m4(v_u1_u16, 3, v_ave0, vl);
    v_a0 = __riscv_vnclipu_wx_u8m2(v_u1_u16, 2, vl);

    // a1 = (src[1] * 1 + s[2] * 1 + 1) >> 1
    v_a1 = __riscv_vaaddu_vv_u8m2(v_ave1, v_ave2, vl);

    // a2 = (src[2] * 1 + s[3] * 3 + 2) >> 2
    v_u1_u16 = __riscv_vwaddu_vx_u16m4(v_ave2, 0, vl);
    v_u1_u16 = __riscv_vwmaccu_vx_u16m4(v_u1_u16, 3, v_ave3, vl);
    v_a2 = __riscv_vnclipu_wx_u8m2(v_u1_u16, 2, vl);

    __riscv_vsseg3e8_v_u8m2(dst_ptr, v_a0, v_a1, v_a2, vl);

    w -= vl;
    s += 4 * vl;
    dst_ptr += 3 * vl;
  } while (w > 0);
}
#endif
#endif

#ifdef HAS_SCALEROWDOWN38_RVV
#ifdef LIBYUV_RVV_HAS_TUPLE_TYPE
void ScaleRowDown38_RVV(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst_ptr,
                        int dst_width) {
  size_t w = (size_t)dst_width / 3u;
  (void)src_stride;
  assert(dst_width % 3 == 0);
  do {
    size_t vl = __riscv_vsetvl_e8m1(w);
    vuint8m1x8_t v_src = __riscv_vlseg8e8_v_u8m1x8(src_ptr, vl);
    vuint8m1_t v_s0 = __riscv_vget_v_u8m1x8_u8m1(v_src, 0);
    vuint8m1_t v_s3 = __riscv_vget_v_u8m1x8_u8m1(v_src, 3);
    vuint8m1_t v_s6 = __riscv_vget_v_u8m1x8_u8m1(v_src, 6);
    vuint8m1x3_t v_dst = __riscv_vcreate_v_u8m1x3(v_s0, v_s3, v_s6);
    __riscv_vsseg3e8_v_u8m1x3(dst_ptr, v_dst, vl);
    w -= vl;
    src_ptr += 8 * vl;
    dst_ptr += 3 * vl;
  } while (w > 0);
}
#else
void ScaleRowDown38_RVV(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst_ptr,
                        int dst_width) {
  size_t w = (size_t)dst_width / 3u;
  (void)src_stride;
  assert(dst_width % 3 == 0);
  do {
    vuint8m1_t v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
    size_t vl = __riscv_vsetvl_e8m1(w);
    __riscv_vlseg8e8_v_u8m1(&v_s0, &v_s1, &v_s2, &v_s3, &v_s4, &v_s5, &v_s6,
                            &v_s7, src_ptr, vl);
    __riscv_vsseg3e8_v_u8m1(dst_ptr, v_s0, v_s3, v_s6, vl);
    w -= vl;
    src_ptr += 8 * vl;
    dst_ptr += 3 * vl;
  } while (w > 0);
}
#endif
#endif

#ifdef HAS_SCALEROWDOWN38_2_BOX_RVV
#ifdef LIBYUV_RVV_HAS_TUPLE_TYPE
void ScaleRowDown38_2_Box_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  size_t w = (size_t)dst_width / 3u;
  const uint16_t coeff_a = (65536u / 6u);
  const uint16_t coeff_b = (65536u / 4u);
  assert((dst_width % 3 == 0) && (dst_width > 0));
  do {
    vuint16m2_t v_e, v_f, v_g;
    vuint8m1_t v_dst_e, v_dst_f, v_dst_g;
    vuint8m1x3_t v_dst;
    size_t vl = __riscv_vsetvl_e8m1(w);
    // s: e00, e10, e20, f00, f10, f20, g00, g10
    vuint8m1x8_t v_s = __riscv_vlseg8e8_v_u8m1x8(src_ptr, vl);
    vuint8m1_t v_s0 = __riscv_vget_v_u8m1x8_u8m1(v_s, 0);
    vuint8m1_t v_s1 = __riscv_vget_v_u8m1x8_u8m1(v_s, 1);
    vuint8m1_t v_s2 = __riscv_vget_v_u8m1x8_u8m1(v_s, 2);
    vuint8m1_t v_s3 = __riscv_vget_v_u8m1x8_u8m1(v_s, 3);
    vuint8m1_t v_s4 = __riscv_vget_v_u8m1x8_u8m1(v_s, 4);
    vuint8m1_t v_s5 = __riscv_vget_v_u8m1x8_u8m1(v_s, 5);
    vuint8m1_t v_s6 = __riscv_vget_v_u8m1x8_u8m1(v_s, 6);
    vuint8m1_t v_s7 = __riscv_vget_v_u8m1x8_u8m1(v_s, 7);
    // t: e01, e11, e21, f01, f11, f21, g01, g11
    vuint8m1x8_t v_t = __riscv_vlseg8e8_v_u8m1x8(src_ptr + src_stride, vl);
    vuint8m1_t v_t0 = __riscv_vget_v_u8m1x8_u8m1(v_t, 0);
    vuint8m1_t v_t1 = __riscv_vget_v_u8m1x8_u8m1(v_t, 1);
    vuint8m1_t v_t2 = __riscv_vget_v_u8m1x8_u8m1(v_t, 2);
    vuint8m1_t v_t3 = __riscv_vget_v_u8m1x8_u8m1(v_t, 3);
    vuint8m1_t v_t4 = __riscv_vget_v_u8m1x8_u8m1(v_t, 4);
    vuint8m1_t v_t5 = __riscv_vget_v_u8m1x8_u8m1(v_t, 5);
    vuint8m1_t v_t6 = __riscv_vget_v_u8m1x8_u8m1(v_t, 6);
    vuint8m1_t v_t7 = __riscv_vget_v_u8m1x8_u8m1(v_t, 7);
    // Calculate sum of [e00, e21] to v_e
    // Calculate sum of [f00, f21] to v_f
    // Calculate sum of [g00, g11] to v_g
    vuint16m2_t v_e0 = __riscv_vwaddu_vv_u16m2(v_s0, v_t0, vl);
    vuint16m2_t v_e1 = __riscv_vwaddu_vv_u16m2(v_s1, v_t1, vl);
    vuint16m2_t v_e2 = __riscv_vwaddu_vv_u16m2(v_s2, v_t2, vl);
    vuint16m2_t v_f0 = __riscv_vwaddu_vv_u16m2(v_s3, v_t3, vl);
    vuint16m2_t v_f1 = __riscv_vwaddu_vv_u16m2(v_s4, v_t4, vl);
    vuint16m2_t v_f2 = __riscv_vwaddu_vv_u16m2(v_s5, v_t5, vl);
    vuint16m2_t v_g0 = __riscv_vwaddu_vv_u16m2(v_s6, v_t6, vl);
    vuint16m2_t v_g1 = __riscv_vwaddu_vv_u16m2(v_s7, v_t7, vl);

    v_e0 = __riscv_vadd_vv_u16m2(v_e0, v_e1, vl);
    v_f0 = __riscv_vadd_vv_u16m2(v_f0, v_f1, vl);
    v_e = __riscv_vadd_vv_u16m2(v_e0, v_e2, vl);
    v_f = __riscv_vadd_vv_u16m2(v_f0, v_f2, vl);
    v_g = __riscv_vadd_vv_u16m2(v_g0, v_g1, vl);

    // Average in 16-bit fixed-point
    v_e = __riscv_vmulhu_vx_u16m2(v_e, coeff_a, vl);
    v_f = __riscv_vmulhu_vx_u16m2(v_f, coeff_a, vl);
    v_g = __riscv_vmulhu_vx_u16m2(v_g, coeff_b, vl);
    v_dst_e = __riscv_vnsrl_wx_u8m1(v_e, 0, vl);
    v_dst_f = __riscv_vnsrl_wx_u8m1(v_f, 0, vl);
    v_dst_g = __riscv_vnsrl_wx_u8m1(v_g, 0, vl);

    v_dst = __riscv_vcreate_v_u8m1x3(v_dst_e, v_dst_f, v_dst_g);
    __riscv_vsseg3e8_v_u8m1x3(dst_ptr, v_dst, vl);
    w -= vl;
    src_ptr += 8 * vl;
    dst_ptr += 3 * vl;
  } while (w > 0);
}
#else
void ScaleRowDown38_2_Box_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  size_t w = (size_t)dst_width / 3u;
  const uint16_t coeff_a = (65536u / 6u);
  const uint16_t coeff_b = (65536u / 4u);
  assert((dst_width % 3 == 0) && (dst_width > 0));
  do {
    vuint8m1_t v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
    vuint8m1_t v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7;
    vuint16m2_t v_e0, v_e1, v_e2, v_e;
    vuint16m2_t v_f0, v_f1, v_f2, v_f;
    vuint16m2_t v_g0, v_g1, v_g;
    vuint8m1_t v_dst_e, v_dst_f, v_dst_g;
    size_t vl = __riscv_vsetvl_e8m1(w);
    // s: e00, e10, e20, f00, f10, f20, g00, g10
    // t: e01, e11, e21, f01, f11, f21, g01, g11
    __riscv_vlseg8e8_v_u8m1(&v_s0, &v_s1, &v_s2, &v_s3, &v_s4, &v_s5, &v_s6,
                            &v_s7, src_ptr, vl);
    __riscv_vlseg8e8_v_u8m1(&v_t0, &v_t1, &v_t2, &v_t3, &v_t4, &v_t5, &v_t6,
                            &v_t7, src_ptr + src_stride, vl);
    // Calculate sum of [e00, e21] to v_e
    // Calculate sum of [f00, f21] to v_f
    // Calculate sum of [g00, g11] to v_g
    v_e0 = __riscv_vwaddu_vv_u16m2(v_s0, v_t0, vl);
    v_e1 = __riscv_vwaddu_vv_u16m2(v_s1, v_t1, vl);
    v_e2 = __riscv_vwaddu_vv_u16m2(v_s2, v_t2, vl);
    v_f0 = __riscv_vwaddu_vv_u16m2(v_s3, v_t3, vl);
    v_f1 = __riscv_vwaddu_vv_u16m2(v_s4, v_t4, vl);
    v_f2 = __riscv_vwaddu_vv_u16m2(v_s5, v_t5, vl);
    v_g0 = __riscv_vwaddu_vv_u16m2(v_s6, v_t6, vl);
    v_g1 = __riscv_vwaddu_vv_u16m2(v_s7, v_t7, vl);

    v_e0 = __riscv_vadd_vv_u16m2(v_e0, v_e1, vl);
    v_f0 = __riscv_vadd_vv_u16m2(v_f0, v_f1, vl);
    v_e = __riscv_vadd_vv_u16m2(v_e0, v_e2, vl);
    v_f = __riscv_vadd_vv_u16m2(v_f0, v_f2, vl);
    v_g = __riscv_vadd_vv_u16m2(v_g0, v_g1, vl);

    // Average in 16-bit fixed-point
    v_e = __riscv_vmulhu_vx_u16m2(v_e, coeff_a, vl);
    v_f = __riscv_vmulhu_vx_u16m2(v_f, coeff_a, vl);
    v_g = __riscv_vmulhu_vx_u16m2(v_g, coeff_b, vl);

    v_dst_e = __riscv_vnsrl_wx_u8m1(v_e, 0, vl);
    v_dst_f = __riscv_vnsrl_wx_u8m1(v_f, 0, vl);
    v_dst_g = __riscv_vnsrl_wx_u8m1(v_g, 0, vl);

    __riscv_vsseg3e8_v_u8m1(dst_ptr, v_dst_e, v_dst_f, v_dst_g, vl);
    w -= vl;
    src_ptr += 8 * vl;
    dst_ptr += 3 * vl;
  } while (w > 0);
}
#endif
#endif

#ifdef HAS_SCALEROWDOWN38_3_BOX_RVV
#ifdef LIBYUV_RVV_HAS_TUPLE_TYPE
void ScaleRowDown38_3_Box_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  size_t w = (size_t)dst_width / 3u;
  const uint16_t coeff_a = (65536u / 9u);
  const uint16_t coeff_b = (65536u / 6u);
  assert((dst_width % 3 == 0) && (dst_width > 0));
  do {
    vuint16m2_t v_e0, v_e1, v_e2, v_e3, v_e4, v_e;
    vuint16m2_t v_f0, v_f1, v_f2, v_f3, v_f4, v_f;
    vuint16m2_t v_g0, v_g1, v_g2, v_g;
    vuint8m1_t v_dst_e, v_dst_f, v_dst_g;
    vuint8m1x3_t v_dst;
    size_t vl = __riscv_vsetvl_e8m1(w);
    // s: e00, e10, e20, f00, f10, f20, g00, g10
    vuint8m1x8_t v_s = __riscv_vlseg8e8_v_u8m1x8(src_ptr, vl);
    vuint8m1_t v_s0 = __riscv_vget_v_u8m1x8_u8m1(v_s, 0);
    vuint8m1_t v_s1 = __riscv_vget_v_u8m1x8_u8m1(v_s, 1);
    vuint8m1_t v_s2 = __riscv_vget_v_u8m1x8_u8m1(v_s, 2);
    vuint8m1_t v_s3 = __riscv_vget_v_u8m1x8_u8m1(v_s, 3);
    vuint8m1_t v_s4 = __riscv_vget_v_u8m1x8_u8m1(v_s, 4);
    vuint8m1_t v_s5 = __riscv_vget_v_u8m1x8_u8m1(v_s, 5);
    vuint8m1_t v_s6 = __riscv_vget_v_u8m1x8_u8m1(v_s, 6);
    vuint8m1_t v_s7 = __riscv_vget_v_u8m1x8_u8m1(v_s, 7);
    // t: e01, e11, e21, f01, f11, f21, g01, g11
    vuint8m1x8_t v_t = __riscv_vlseg8e8_v_u8m1x8(src_ptr + src_stride, vl);
    vuint8m1_t v_t0 = __riscv_vget_v_u8m1x8_u8m1(v_t, 0);
    vuint8m1_t v_t1 = __riscv_vget_v_u8m1x8_u8m1(v_t, 1);
    vuint8m1_t v_t2 = __riscv_vget_v_u8m1x8_u8m1(v_t, 2);
    vuint8m1_t v_t3 = __riscv_vget_v_u8m1x8_u8m1(v_t, 3);
    vuint8m1_t v_t4 = __riscv_vget_v_u8m1x8_u8m1(v_t, 4);
    vuint8m1_t v_t5 = __riscv_vget_v_u8m1x8_u8m1(v_t, 5);
    vuint8m1_t v_t6 = __riscv_vget_v_u8m1x8_u8m1(v_t, 6);
    vuint8m1_t v_t7 = __riscv_vget_v_u8m1x8_u8m1(v_t, 7);
    // u: e02, e12, e22, f02, f12, f22, g02, g12
    vuint8m1x8_t v_u = __riscv_vlseg8e8_v_u8m1x8(src_ptr + 2 * src_stride, vl);
    vuint8m1_t v_u0 = __riscv_vget_v_u8m1x8_u8m1(v_u, 0);
    vuint8m1_t v_u1 = __riscv_vget_v_u8m1x8_u8m1(v_u, 1);
    vuint8m1_t v_u2 = __riscv_vget_v_u8m1x8_u8m1(v_u, 2);
    vuint8m1_t v_u3 = __riscv_vget_v_u8m1x8_u8m1(v_u, 3);
    vuint8m1_t v_u4 = __riscv_vget_v_u8m1x8_u8m1(v_u, 4);
    vuint8m1_t v_u5 = __riscv_vget_v_u8m1x8_u8m1(v_u, 5);
    vuint8m1_t v_u6 = __riscv_vget_v_u8m1x8_u8m1(v_u, 6);
    vuint8m1_t v_u7 = __riscv_vget_v_u8m1x8_u8m1(v_u, 7);
    // Calculate sum of [e00, e22]
    v_e0 = __riscv_vwaddu_vv_u16m2(v_s0, v_t0, vl);
    v_e1 = __riscv_vwaddu_vv_u16m2(v_s1, v_t1, vl);
    v_e2 = __riscv_vwaddu_vv_u16m2(v_s2, v_t2, vl);
    v_e3 = __riscv_vwaddu_vv_u16m2(v_u0, v_u1, vl);
    v_e4 = __riscv_vwaddu_vx_u16m2(v_u2, 0, vl);

    v_e0 = __riscv_vadd_vv_u16m2(v_e0, v_e1, vl);
    v_e2 = __riscv_vadd_vv_u16m2(v_e2, v_e3, vl);
    v_e0 = __riscv_vadd_vv_u16m2(v_e0, v_e4, vl);
    v_e = __riscv_vadd_vv_u16m2(v_e0, v_e2, vl);
    // Calculate sum of [f00, f22]
    v_f0 = __riscv_vwaddu_vv_u16m2(v_s3, v_t3, vl);
    v_f1 = __riscv_vwaddu_vv_u16m2(v_s4, v_t4, vl);
    v_f2 = __riscv_vwaddu_vv_u16m2(v_s5, v_t5, vl);
    v_f3 = __riscv_vwaddu_vv_u16m2(v_u3, v_u4, vl);
    v_f4 = __riscv_vwaddu_vx_u16m2(v_u5, 0, vl);

    v_f0 = __riscv_vadd_vv_u16m2(v_f0, v_f1, vl);
    v_f2 = __riscv_vadd_vv_u16m2(v_f2, v_f3, vl);
    v_f0 = __riscv_vadd_vv_u16m2(v_f0, v_f4, vl);
    v_f = __riscv_vadd_vv_u16m2(v_f0, v_f2, vl);
    // Calculate sum of [g00, g12]
    v_g0 = __riscv_vwaddu_vv_u16m2(v_s6, v_t6, vl);
    v_g1 = __riscv_vwaddu_vv_u16m2(v_s7, v_t7, vl);
    v_g2 = __riscv_vwaddu_vv_u16m2(v_u6, v_u7, vl);

    v_g = __riscv_vadd_vv_u16m2(v_g0, v_g1, vl);
    v_g = __riscv_vadd_vv_u16m2(v_g, v_g2, vl);

    // Average in 16-bit fixed-point
    v_e = __riscv_vmulhu_vx_u16m2(v_e, coeff_a, vl);
    v_f = __riscv_vmulhu_vx_u16m2(v_f, coeff_a, vl);
    v_g = __riscv_vmulhu_vx_u16m2(v_g, coeff_b, vl);
    v_dst_e = __riscv_vnsrl_wx_u8m1(v_e, 0, vl);
    v_dst_f = __riscv_vnsrl_wx_u8m1(v_f, 0, vl);
    v_dst_g = __riscv_vnsrl_wx_u8m1(v_g, 0, vl);

    v_dst = __riscv_vcreate_v_u8m1x3(v_dst_e, v_dst_f, v_dst_g);
    __riscv_vsseg3e8_v_u8m1x3(dst_ptr, v_dst, vl);
    w -= vl;
    src_ptr += 8 * vl;
    dst_ptr += 3 * vl;
  } while (w > 0);
}
#else
void ScaleRowDown38_3_Box_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width) {
  size_t w = (size_t)dst_width / 3u;
  const uint16_t coeff_a = (65536u / 9u);
  const uint16_t coeff_b = (65536u / 6u);
  assert((dst_width % 3 == 0) && (dst_width > 0));
  do {
    vuint8m1_t v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7;
    vuint8m1_t v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7;
    vuint8m1_t v_u0, v_u1, v_u2, v_u3, v_u4, v_u5, v_u6, v_u7;
    vuint16m2_t v_e0, v_e1, v_e2, v_e3, v_e4, v_e;
    vuint16m2_t v_f0, v_f1, v_f2, v_f3, v_f4, v_f;
    vuint16m2_t v_g0, v_g1, v_g2, v_g;
    vuint8m1_t v_dst_e, v_dst_f, v_dst_g;
    size_t vl = __riscv_vsetvl_e8m1(w);
    // s: e00, e10, e20, f00, f10, f20, g00, g10
    // t: e01, e11, e21, f01, f11, f21, g01, g11
    // u: e02, e12, e22, f02, f12, f22, g02, g12
    __riscv_vlseg8e8_v_u8m1(&v_s0, &v_s1, &v_s2, &v_s3, &v_s4, &v_s5, &v_s6,
                            &v_s7, src_ptr, vl);
    __riscv_vlseg8e8_v_u8m1(&v_t0, &v_t1, &v_t2, &v_t3, &v_t4, &v_t5, &v_t6,
                            &v_t7, src_ptr + src_stride, vl);
    __riscv_vlseg8e8_v_u8m1(&v_u0, &v_u1, &v_u2, &v_u3, &v_u4, &v_u5, &v_u6,
                            &v_u7, src_ptr + 2 * src_stride, vl);
    // Calculate sum of [e00, e22]
    v_e0 = __riscv_vwaddu_vv_u16m2(v_s0, v_t0, vl);
    v_e1 = __riscv_vwaddu_vv_u16m2(v_s1, v_t1, vl);
    v_e2 = __riscv_vwaddu_vv_u16m2(v_s2, v_t2, vl);
    v_e3 = __riscv_vwaddu_vv_u16m2(v_u0, v_u1, vl);
    v_e4 = __riscv_vwaddu_vx_u16m2(v_u2, 0, vl);

    v_e0 = __riscv_vadd_vv_u16m2(v_e0, v_e1, vl);
    v_e2 = __riscv_vadd_vv_u16m2(v_e2, v_e3, vl);
    v_e0 = __riscv_vadd_vv_u16m2(v_e0, v_e4, vl);
    v_e = __riscv_vadd_vv_u16m2(v_e0, v_e2, vl);
    // Calculate sum of [f00, f22]
    v_f0 = __riscv_vwaddu_vv_u16m2(v_s3, v_t3, vl);
    v_f1 = __riscv_vwaddu_vv_u16m2(v_s4, v_t4, vl);
    v_f2 = __riscv_vwaddu_vv_u16m2(v_s5, v_t5, vl);
    v_f3 = __riscv_vwaddu_vv_u16m2(v_u3, v_u4, vl);
    v_f4 = __riscv_vwaddu_vx_u16m2(v_u5, 0, vl);

    v_f0 = __riscv_vadd_vv_u16m2(v_f0, v_f1, vl);
    v_f2 = __riscv_vadd_vv_u16m2(v_f2, v_f3, vl);
    v_f0 = __riscv_vadd_vv_u16m2(v_f0, v_f4, vl);
    v_f = __riscv_vadd_vv_u16m2(v_f0, v_f2, vl);
    // Calculate sum of [g00, g12]
    v_g0 = __riscv_vwaddu_vv_u16m2(v_s6, v_t6, vl);
    v_g1 = __riscv_vwaddu_vv_u16m2(v_s7, v_t7, vl);
    v_g2 = __riscv_vwaddu_vv_u16m2(v_u6, v_u7, vl);

    v_g = __riscv_vadd_vv_u16m2(v_g0, v_g1, vl);
    v_g = __riscv_vadd_vv_u16m2(v_g, v_g2, vl);

    // Average in 16-bit fixed-point
    v_e = __riscv_vmulhu_vx_u16m2(v_e, coeff_a, vl);
    v_f = __riscv_vmulhu_vx_u16m2(v_f, coeff_a, vl);
    v_g = __riscv_vmulhu_vx_u16m2(v_g, coeff_b, vl);

    v_dst_e = __riscv_vnsrl_wx_u8m1(v_e, 0, vl);
    v_dst_f = __riscv_vnsrl_wx_u8m1(v_f, 0, vl);
    v_dst_g = __riscv_vnsrl_wx_u8m1(v_g, 0, vl);
    __riscv_vsseg3e8_v_u8m1(dst_ptr, v_dst_e, v_dst_f, v_dst_g, vl);
    w -= vl;
    src_ptr += 8 * vl;
    dst_ptr += 3 * vl;
  } while (w > 0);
}
#endif
#endif

// ScaleUVRowUp2_(Bi)linear_RVV function is equal to other platforms'
// ScaleRowUp2_(Bi)linear_Any_XXX. We process entire row in this function. Other
// platforms only implement non-edge part of image and process edge with scalar.

#ifdef HAS_SCALEROWUP2_LINEAR_RVV
#ifdef LIBYUV_RVV_HAS_TUPLE_TYPE
void ScaleRowUp2_Linear_RVV(const uint8_t* src_ptr,
                            uint8_t* dst_ptr,
                            int dst_width) {
  size_t work_width = (size_t)dst_width - 1u;
  size_t src_width = work_width >> 1u;
  const uint8_t* work_src_ptr = src_ptr;
  uint8_t* work_dst_ptr = dst_ptr + 1;
  size_t vl = __riscv_vsetvlmax_e8m4();
  vuint8m4_t v_3 = __riscv_vmv_v_x_u8m4(3, vl);
  dst_ptr[0] = src_ptr[0];
  while (src_width > 0) {
    vuint8m4_t v_src0, v_src1, v_dst_odd, v_dst_even;
    vuint16m8_t v_src0_u16, v_src1_u16;
    vuint8m4x2_t v_dst;
    size_t vl = __riscv_vsetvl_e8m4(src_width);
    v_src0 = __riscv_vle8_v_u8m4(work_src_ptr, vl);
    v_src1 = __riscv_vle8_v_u8m4(work_src_ptr + 1, vl);

    v_src0_u16 = __riscv_vwaddu_vx_u16m8(v_src0, 2, vl);
    v_src1_u16 = __riscv_vwaddu_vx_u16m8(v_src1, 2, vl);
    v_src0_u16 = __riscv_vwmaccu_vv_u16m8(v_src0_u16, v_3, v_src1, vl);
    v_src1_u16 = __riscv_vwmaccu_vv_u16m8(v_src1_u16, v_3, v_src0, vl);

    v_dst_odd = __riscv_vnsrl_wx_u8m4(v_src0_u16, 2, vl);
    v_dst_even = __riscv_vnsrl_wx_u8m4(v_src1_u16, 2, vl);

    v_dst = __riscv_vcreate_v_u8m4x2(v_dst_even, v_dst_odd);
    __riscv_vsseg2e8_v_u8m4x2(work_dst_ptr, v_dst, vl);

    src_width -= vl;
    work_src_ptr += vl;
    work_dst_ptr += 2 * vl;
  }
  dst_ptr[dst_width - 1] = src_ptr[(dst_width - 1) / 2];
}
#else
void ScaleRowUp2_Linear_RVV(const uint8_t* src_ptr,
                            uint8_t* dst_ptr,
                            int dst_width) {
  size_t work_width = (size_t)dst_width - 1u;
  size_t src_width = work_width >> 1u;
  const uint8_t* work_src_ptr = src_ptr;
  uint8_t* work_dst_ptr = dst_ptr + 1;
  size_t vl = __riscv_vsetvlmax_e8m4();
  vuint8m4_t v_3 = __riscv_vmv_v_x_u8m4(3, vl);
  dst_ptr[0] = src_ptr[0];
  while (src_width > 0) {
    vuint8m4_t v_src0, v_src1, v_dst_odd, v_dst_even;
    vuint16m8_t v_src0_u16, v_src1_u16;
    size_t vl = __riscv_vsetvl_e8m4(src_width);
    v_src0 = __riscv_vle8_v_u8m4(work_src_ptr, vl);
    v_src1 = __riscv_vle8_v_u8m4(work_src_ptr + 1, vl);

    v_src0_u16 = __riscv_vwaddu_vx_u16m8(v_src0, 2, vl);
    v_src1_u16 = __riscv_vwaddu_vx_u16m8(v_src1, 2, vl);
    v_src0_u16 = __riscv_vwmaccu_vv_u16m8(v_src0_u16, v_3, v_src1, vl);
    v_src1_u16 = __riscv_vwmaccu_vv_u16m8(v_src1_u16, v_3, v_src0, vl);

    v_dst_odd = __riscv_vnsrl_wx_u8m4(v_src0_u16, 2, vl);
    v_dst_even = __riscv_vnsrl_wx_u8m4(v_src1_u16, 2, vl);

    __riscv_vsseg2e8_v_u8m4(work_dst_ptr, v_dst_even, v_dst_odd, vl);

    src_width -= vl;
    work_src_ptr += vl;
    work_dst_ptr += 2 * vl;
  }
  dst_ptr[dst_width - 1] = src_ptr[(dst_width - 1) / 2];
}
#endif
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_RVV
#ifdef LIBYUV_RVV_HAS_TUPLE_TYPE
void ScaleRowUp2_Bilinear_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              ptrdiff_t dst_stride,
                              int dst_width) {
  size_t work_width = ((size_t)dst_width - 1u) & ~1u;
  size_t src_width = work_width >> 1u;
  const uint8_t* work_s = src_ptr;
  const uint8_t* work_t = src_ptr + src_stride;
  const uint8_t* s = work_s;
  const uint8_t* t = work_t;
  uint8_t* d = dst_ptr;
  uint8_t* e = dst_ptr + dst_stride;
  uint8_t* work_d = d + 1;
  uint8_t* work_e = e + 1;
  size_t vl = __riscv_vsetvlmax_e16m4();
  vuint16m4_t v_3_u16 = __riscv_vmv_v_x_u16m4(3, vl);
  vuint8m2_t v_3_u8 = __riscv_vmv_v_x_u8m2(3, vl);
  d[0] = (3 * s[0] + t[0] + 2) >> 2;
  e[0] = (s[0] + 3 * t[0] + 2) >> 2;
  while (src_width > 0) {
    vuint8m2_t v_s0, v_s1, v_t0, v_t1;
    vuint16m4_t v_s0_u16, v_s1_u16, v_t0_u16, v_t1_u16;
    vuint16m4_t v_t0_u16_, v_t1_u16_;
    vuint8m2_t v_dst0_even, v_dst0_odd, v_dst1_even, v_dst1_odd;
    vuint8m2x2_t v_dst0, v_dst1;
    size_t vl = __riscv_vsetvl_e8m2(src_width);
    v_s0 = __riscv_vle8_v_u8m2(work_s, vl);
    v_s1 = __riscv_vle8_v_u8m2(work_s + 1, vl);

    v_s0_u16 = __riscv_vwaddu_vx_u16m4(v_s0, 2, vl);
    v_s1_u16 = __riscv_vwaddu_vx_u16m4(v_s1, 2, vl);
    v_s0_u16 = __riscv_vwmaccu_vv_u16m4(v_s0_u16, v_3_u8, v_s1, vl);
    v_s1_u16 = __riscv_vwmaccu_vv_u16m4(v_s1_u16, v_3_u8, v_s0, vl);

    v_t0 = __riscv_vle8_v_u8m2(work_t, vl);
    v_t1 = __riscv_vle8_v_u8m2(work_t + 1, vl);

    v_t0_u16 = __riscv_vwaddu_vx_u16m4(v_t0, 2, vl);
    v_t1_u16 = __riscv_vwaddu_vx_u16m4(v_t1, 2, vl);
    v_t0_u16 = __riscv_vwmaccu_vv_u16m4(v_t0_u16, v_3_u8, v_t1, vl);
    v_t1_u16 = __riscv_vwmaccu_vv_u16m4(v_t1_u16, v_3_u8, v_t0, vl);

    v_t0_u16_ = __riscv_vmv_v_v_u16m4(v_t0_u16, vl);
    v_t1_u16_ = __riscv_vmv_v_v_u16m4(v_t1_u16, vl);

    v_t0_u16 = __riscv_vmacc_vv_u16m4(v_t0_u16, v_3_u16, v_s0_u16, vl);
    v_t1_u16 = __riscv_vmacc_vv_u16m4(v_t1_u16, v_3_u16, v_s1_u16, vl);
    v_s0_u16 = __riscv_vmacc_vv_u16m4(v_s0_u16, v_3_u16, v_t0_u16_, vl);
    v_s1_u16 = __riscv_vmacc_vv_u16m4(v_s1_u16, v_3_u16, v_t1_u16_, vl);

    v_dst0_odd = __riscv_vnsrl_wx_u8m2(v_t0_u16, 4, vl);
    v_dst0_even = __riscv_vnsrl_wx_u8m2(v_t1_u16, 4, vl);
    v_dst1_odd = __riscv_vnsrl_wx_u8m2(v_s0_u16, 4, vl);
    v_dst1_even = __riscv_vnsrl_wx_u8m2(v_s1_u16, 4, vl);

    v_dst0 = __riscv_vcreate_v_u8m2x2(v_dst0_even, v_dst0_odd);
    __riscv_vsseg2e8_v_u8m2x2(work_d, v_dst0, vl);
    v_dst1 = __riscv_vcreate_v_u8m2x2(v_dst1_even, v_dst1_odd);
    __riscv_vsseg2e8_v_u8m2x2(work_e, v_dst1, vl);
    src_width -= vl;
    work_s += vl;
    work_t += vl;
    work_d += 2 * vl;
    work_e += 2 * vl;
  }
  d[dst_width - 1] =
      (3 * s[(dst_width - 1) / 2] + t[(dst_width - 1) / 2] + 2) >> 2;
  e[dst_width - 1] =
      (s[(dst_width - 1) / 2] + 3 * t[(dst_width - 1) / 2] + 2) >> 2;
}
#else
void ScaleRowUp2_Bilinear_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              ptrdiff_t dst_stride,
                              int dst_width) {
  size_t work_width = ((size_t)dst_width - 1u) & ~1u;
  size_t src_width = work_width >> 1u;
  const uint8_t* work_s = src_ptr;
  const uint8_t* work_t = src_ptr + src_stride;
  const uint8_t* s = work_s;
  const uint8_t* t = work_t;
  uint8_t* d = dst_ptr;
  uint8_t* e = dst_ptr + dst_stride;
  uint8_t* work_d = d + 1;
  uint8_t* work_e = e + 1;
  size_t vl = __riscv_vsetvlmax_e16m4();
  vuint16m4_t v_3_u16 = __riscv_vmv_v_x_u16m4(3, vl);
  vuint8m2_t v_3_u8 = __riscv_vmv_v_x_u8m2(3, vl);
  d[0] = (3 * s[0] + t[0] + 2) >> 2;
  e[0] = (s[0] + 3 * t[0] + 2) >> 2;
  while (src_width > 0) {
    vuint8m2_t v_s0, v_s1, v_t0, v_t1;
    vuint16m4_t v_s0_u16, v_s1_u16, v_t0_u16, v_t1_u16;
    vuint16m4_t v_t0_u16_, v_t1_u16_;
    vuint8m2_t v_dst0_even, v_dst0_odd, v_dst1_even, v_dst1_odd;
    size_t vl = __riscv_vsetvl_e8m2(src_width);
    v_s0 = __riscv_vle8_v_u8m2(work_s, vl);
    v_s1 = __riscv_vle8_v_u8m2(work_s + 1, vl);

    v_s0_u16 = __riscv_vwaddu_vx_u16m4(v_s0, 2, vl);
    v_s1_u16 = __riscv_vwaddu_vx_u16m4(v_s1, 2, vl);
    v_s0_u16 = __riscv_vwmaccu_vv_u16m4(v_s0_u16, v_3_u8, v_s1, vl);
    v_s1_u16 = __riscv_vwmaccu_vv_u16m4(v_s1_u16, v_3_u8, v_s0, vl);

    v_t0 = __riscv_vle8_v_u8m2(work_t, vl);
    v_t1 = __riscv_vle8_v_u8m2(work_t + 1, vl);

    v_t0_u16 = __riscv_vwaddu_vx_u16m4(v_t0, 2, vl);
    v_t1_u16 = __riscv_vwaddu_vx_u16m4(v_t1, 2, vl);
    v_t0_u16 = __riscv_vwmaccu_vv_u16m4(v_t0_u16, v_3_u8, v_t1, vl);
    v_t1_u16 = __riscv_vwmaccu_vv_u16m4(v_t1_u16, v_3_u8, v_t0, vl);

    v_t0_u16_ = __riscv_vmv_v_v_u16m4(v_t0_u16, vl);
    v_t1_u16_ = __riscv_vmv_v_v_u16m4(v_t1_u16, vl);

    v_t0_u16 = __riscv_vmacc_vv_u16m4(v_t0_u16, v_3_u16, v_s0_u16, vl);
    v_t1_u16 = __riscv_vmacc_vv_u16m4(v_t1_u16, v_3_u16, v_s1_u16, vl);
    v_s0_u16 = __riscv_vmacc_vv_u16m4(v_s0_u16, v_3_u16, v_t0_u16_, vl);
    v_s1_u16 = __riscv_vmacc_vv_u16m4(v_s1_u16, v_3_u16, v_t1_u16_, vl);

    v_dst0_odd = __riscv_vnsrl_wx_u8m2(v_t0_u16, 4, vl);
    v_dst0_even = __riscv_vnsrl_wx_u8m2(v_t1_u16, 4, vl);
    v_dst1_odd = __riscv_vnsrl_wx_u8m2(v_s0_u16, 4, vl);
    v_dst1_even = __riscv_vnsrl_wx_u8m2(v_s1_u16, 4, vl);

    __riscv_vsseg2e8_v_u8m2(work_d, v_dst0_even, v_dst0_odd, vl);
    __riscv_vsseg2e8_v_u8m2(work_e, v_dst1_even, v_dst1_odd, vl);

    src_width -= vl;
    work_s += vl;
    work_t += vl;
    work_d += 2 * vl;
    work_e += 2 * vl;
  }
  d[dst_width - 1] =
      (3 * s[(dst_width - 1) / 2] + t[(dst_width - 1) / 2] + 2) >> 2;
  e[dst_width - 1] =
      (s[(dst_width - 1) / 2] + 3 * t[(dst_width - 1) / 2] + 2) >> 2;
}
#endif
#endif

#ifdef HAS_SCALEUVROWDOWN2_RVV
void ScaleUVRowDown2_RVV(const uint8_t* src_uv,
                         ptrdiff_t src_stride,
                         uint8_t* dst_uv,
                         int dst_width) {
  size_t w = (size_t)dst_width;
  const uint32_t* src = (const uint32_t*)src_uv;
  uint16_t* dst = (uint16_t*)dst_uv;
  (void)src_stride;
  do {
    size_t vl = __riscv_vsetvl_e32m8(w);
    vuint32m8_t v_data = __riscv_vle32_v_u32m8(src, vl);
    vuint16m4_t v_u1v1 = __riscv_vnsrl_wx_u16m4(v_data, 16, vl);
    __riscv_vse16_v_u16m4(dst, v_u1v1, vl);
    w -= vl;
    src += vl;
    dst += vl;
  } while (w > 0);
}
#endif

#ifdef HAS_SCALEUVROWDOWN2LINEAR_RVV
#if defined(LIBYUV_RVV_HAS_TUPLE_TYPE) && defined(LIBYUV_RVV_HAS_VXRM_ARG)
void ScaleUVRowDown2Linear_RVV(const uint8_t* src_uv,
                               ptrdiff_t src_stride,
                               uint8_t* dst_uv,
                               int dst_width) {
  size_t w = (size_t)dst_width;
  const uint16_t* src = (const uint16_t*)src_uv;
  (void)src_stride;
  do {
    size_t vl = __riscv_vsetvl_e16m4(w);
    vuint16m4x2_t v_src = __riscv_vlseg2e16_v_u16m4x2(src, vl);
    vuint16m4_t v_u0v0_16 = __riscv_vget_v_u16m4x2_u16m4(v_src, 0);
    vuint16m4_t v_u1v1_16 = __riscv_vget_v_u16m4x2_u16m4(v_src, 1);
    vuint8m4_t v_u0v0 = __riscv_vreinterpret_v_u16m4_u8m4(v_u0v0_16);
    vuint8m4_t v_u1v1 = __riscv_vreinterpret_v_u16m4_u8m4(v_u1v1_16);
    vuint8m4_t v_avg =
        __riscv_vaaddu_vv_u8m4(v_u0v0, v_u1v1, __RISCV_VXRM_RNU, vl * 2);
    __riscv_vse8_v_u8m4(dst_uv, v_avg, vl * 2);
    w -= vl;
    src += vl * 2;
    dst_uv += vl * 2;
  } while (w > 0);
}
#else
void ScaleUVRowDown2Linear_RVV(const uint8_t* src_uv,
                               ptrdiff_t src_stride,
                               uint8_t* dst_uv,
                               int dst_width) {
  size_t w = (size_t)dst_width;
  const uint16_t* src = (const uint16_t*)src_uv;
  (void)src_stride;
  // NOTE: To match behavior on other platforms, vxrm (fixed-point rounding mode
  // register) is set to round-to-nearest-up mode(0).
  asm volatile("csrwi vxrm, 0");
  do {
    vuint8m4_t v_u0v0, v_u1v1, v_avg;
    vuint16m4_t v_u0v0_16, v_u1v1_16;
    size_t vl = __riscv_vsetvl_e16m4(w);
    __riscv_vlseg2e16_v_u16m4(&v_u0v0_16, &v_u1v1_16, src, vl);
    v_u0v0 = __riscv_vreinterpret_v_u16m4_u8m4(v_u0v0_16);
    v_u1v1 = __riscv_vreinterpret_v_u16m4_u8m4(v_u1v1_16);
    // Use round-to-nearest-up mode for averaging add
    v_avg = __riscv_vaaddu_vv_u8m4(v_u0v0, v_u1v1, vl * 2);
    __riscv_vse8_v_u8m4(dst_uv, v_avg, vl * 2);
    w -= vl;
    src += vl * 2;
    dst_uv += vl * 2;
  } while (w > 0);
}
#endif
#endif

#ifdef HAS_SCALEUVROWDOWN2BOX_RVV
#if defined(LIBYUV_RVV_HAS_TUPLE_TYPE) && defined(LIBYUV_RVV_HAS_VXRM_ARG)
void ScaleUVRowDown2Box_RVV(const uint8_t* src_uv,
                            ptrdiff_t src_stride,
                            uint8_t* dst_uv,
                            int dst_width) {
  const uint8_t* src_uv_row1 = src_uv + src_stride;
  size_t w = (size_t)dst_width;
  do {
    size_t vl = __riscv_vsetvl_e8m2(w);
    vuint8m2x4_t v_s = __riscv_vlseg4e8_v_u8m2x4(src_uv, vl);
    vuint8m2_t v_u0_row0 = __riscv_vget_v_u8m2x4_u8m2(v_s, 0);
    vuint8m2_t v_v0_row0 = __riscv_vget_v_u8m2x4_u8m2(v_s, 1);
    vuint8m2_t v_u1_row0 = __riscv_vget_v_u8m2x4_u8m2(v_s, 2);
    vuint8m2_t v_v1_row0 = __riscv_vget_v_u8m2x4_u8m2(v_s, 3);
    vuint8m2x4_t v_t = __riscv_vlseg4e8_v_u8m2x4(src_uv_row1, vl);
    vuint8m2_t v_u0_row1 = __riscv_vget_v_u8m2x4_u8m2(v_t, 0);
    vuint8m2_t v_v0_row1 = __riscv_vget_v_u8m2x4_u8m2(v_t, 1);
    vuint8m2_t v_u1_row1 = __riscv_vget_v_u8m2x4_u8m2(v_t, 2);
    vuint8m2_t v_v1_row1 = __riscv_vget_v_u8m2x4_u8m2(v_t, 3);

    vuint16m4_t v_u0u1_row0 = __riscv_vwaddu_vv_u16m4(v_u0_row0, v_u1_row0, vl);
    vuint16m4_t v_u0u1_row1 = __riscv_vwaddu_vv_u16m4(v_u0_row1, v_u1_row1, vl);
    vuint16m4_t v_v0v1_row0 = __riscv_vwaddu_vv_u16m4(v_v0_row0, v_v1_row0, vl);
    vuint16m4_t v_v0v1_row1 = __riscv_vwaddu_vv_u16m4(v_v0_row1, v_v1_row1, vl);
    vuint16m4_t v_sum0 = __riscv_vadd_vv_u16m4(v_u0u1_row0, v_u0u1_row1, vl);
    vuint16m4_t v_sum1 = __riscv_vadd_vv_u16m4(v_v0v1_row0, v_v0v1_row1, vl);
    vuint8m2_t v_dst_u =
        __riscv_vnclipu_wx_u8m2(v_sum0, 2, __RISCV_VXRM_RNU, vl);
    vuint8m2_t v_dst_v =
        __riscv_vnclipu_wx_u8m2(v_sum1, 2, __RISCV_VXRM_RNU, vl);

    vuint8m2x2_t v_dst_uv = __riscv_vcreate_v_u8m2x2(v_dst_u, v_dst_v);
    __riscv_vsseg2e8_v_u8m2x2(dst_uv, v_dst_uv, vl);

    dst_uv += 2 * vl;
    src_uv += 4 * vl;
    w -= vl;
    src_uv_row1 += 4 * vl;
  } while (w > 0);
}
#else
void ScaleUVRowDown2Box_RVV(const uint8_t* src_uv,
                            ptrdiff_t src_stride,
                            uint8_t* dst_uv,
                            int dst_width) {
  const uint8_t* src_uv_row1 = src_uv + src_stride;
  size_t w = (size_t)dst_width;
  // NOTE: To match behavior on other platforms, vxrm (fixed-point rounding mode
  // register) is set to round-to-nearest-up mode(0).
  asm volatile("csrwi vxrm, 0");
  do {
    vuint8m2_t v_u0_row0, v_v0_row0, v_u1_row0, v_v1_row0;
    vuint8m2_t v_u0_row1, v_v0_row1, v_u1_row1, v_v1_row1;
    vuint16m4_t v_u0u1_row0, v_u0u1_row1, v_v0v1_row0, v_v0v1_row1;
    vuint16m4_t v_sum0, v_sum1;
    vuint8m2_t v_dst_u, v_dst_v;
    size_t vl = __riscv_vsetvl_e8m2(w);

    __riscv_vlseg4e8_v_u8m2(&v_u0_row0, &v_v0_row0, &v_u1_row0, &v_v1_row0,
                            src_uv, vl);
    __riscv_vlseg4e8_v_u8m2(&v_u0_row1, &v_v0_row1, &v_u1_row1, &v_v1_row1,
                            src_uv_row1, vl);

    v_u0u1_row0 = __riscv_vwaddu_vv_u16m4(v_u0_row0, v_u1_row0, vl);
    v_u0u1_row1 = __riscv_vwaddu_vv_u16m4(v_u0_row1, v_u1_row1, vl);
    v_v0v1_row0 = __riscv_vwaddu_vv_u16m4(v_v0_row0, v_v1_row0, vl);
    v_v0v1_row1 = __riscv_vwaddu_vv_u16m4(v_v0_row1, v_v1_row1, vl);

    v_sum0 = __riscv_vadd_vv_u16m4(v_u0u1_row0, v_u0u1_row1, vl);
    v_sum1 = __riscv_vadd_vv_u16m4(v_v0v1_row0, v_v0v1_row1, vl);
    // Use round-to-nearest-up mode for vnclip
    v_dst_u = __riscv_vnclipu_wx_u8m2(v_sum0, 2, vl);
    v_dst_v = __riscv_vnclipu_wx_u8m2(v_sum1, 2, vl);

    __riscv_vsseg2e8_v_u8m2(dst_uv, v_dst_u, v_dst_v, vl);

    dst_uv += 2 * vl;
    src_uv += 4 * vl;
    w -= vl;
    src_uv_row1 += 4 * vl;
  } while (w > 0);
}
#endif
#endif

#ifdef HAS_SCALEUVROWDOWN4_RVV
void ScaleUVRowDown4_RVV(const uint8_t* src_uv,
                         ptrdiff_t src_stride,
                         int src_stepx,
                         uint8_t* dst_uv,
                         int dst_width) {
  // Overflow will never happen here, since sizeof(size_t)/sizeof(int)=2.
  // dst_width = src_width / 4 and src_width is also int.
  size_t w = (size_t)dst_width * 8;
  (void)src_stride;
  (void)src_stepx;
  do {
    size_t vl = __riscv_vsetvl_e8m8(w);
    vuint8m8_t v_row = __riscv_vle8_v_u8m8(src_uv, vl);
    vuint64m8_t v_row_64 = __riscv_vreinterpret_v_u8m8_u64m8(v_row);
    // Narrowing without clipping
    vuint32m4_t v_tmp = __riscv_vncvt_x_x_w_u32m4(v_row_64, vl / 8);
    vuint16m2_t v_dst_16 = __riscv_vncvt_x_x_w_u16m2(v_tmp, vl / 8);
    vuint8m2_t v_dst = __riscv_vreinterpret_v_u16m2_u8m2(v_dst_16);
    __riscv_vse8_v_u8m2(dst_uv, v_dst, vl / 4);
    w -= vl;
    src_uv += vl;
    dst_uv += vl / 4;
  } while (w > 0);
}
#endif

#ifdef HAS_SCALEUVROWDOWNEVEN_RVV
void ScaleUVRowDownEven_RVV(const uint8_t* src_uv,
                            ptrdiff_t src_stride,
                            int src_stepx,
                            uint8_t* dst_uv,
                            int dst_width) {
  size_t w = (size_t)dst_width;
  const ptrdiff_t stride_byte = (ptrdiff_t)src_stepx * 2;
  const uint16_t* src = (const uint16_t*)(src_uv);
  uint16_t* dst = (uint16_t*)(dst_uv);
  (void)src_stride;
  do {
    size_t vl = __riscv_vsetvl_e16m8(w);
    vuint16m8_t v_row = __riscv_vlse16_v_u16m8(src, stride_byte, vl);
    __riscv_vse16_v_u16m8(dst, v_row, vl);
    w -= vl;
    src += vl * src_stepx;
    dst += vl;
  } while (w > 0);
}
#endif

// ScaleUVRowUp2_(Bi)linear_RVV function is equal to other platforms'
// ScaleUVRowUp2_(Bi)linear_Any_XXX. We process entire row in this function.
// Other platforms only implement non-edge part of image and process edge with
// scalar.

#ifdef HAS_SCALEUVROWUP2_LINEAR_RVV
#ifdef LIBYUV_RVV_HAS_TUPLE_TYPE
void ScaleUVRowUp2_Linear_RVV(const uint8_t* src_ptr,
                              uint8_t* dst_ptr,
                              int dst_width) {
  size_t work_width = ((size_t)dst_width - 1u) & ~1u;
  uint16_t* work_dst_ptr = (uint16_t*)dst_ptr + 1;
  const uint8_t* work_src_ptr = src_ptr;
  size_t vl = __riscv_vsetvlmax_e8m4();
  vuint8m4_t v_3_u8 = __riscv_vmv_v_x_u8m4(3, vl);
  dst_ptr[0] = src_ptr[0];
  dst_ptr[1] = src_ptr[1];
  while (work_width > 0) {
    vuint8m4_t v_uv0, v_uv1, v_dst_odd_u8, v_dst_even_u8;
    vuint16m4_t v_dst_odd, v_dst_even;
    vuint16m8_t v_uv0_u16, v_uv1_u16;
    vuint16m4x2_t v_dst;
    size_t vl = __riscv_vsetvl_e8m4(work_width);
    v_uv0 = __riscv_vle8_v_u8m4(work_src_ptr, vl);
    v_uv1 = __riscv_vle8_v_u8m4(work_src_ptr + 2, vl);

    v_uv0_u16 = __riscv_vwaddu_vx_u16m8(v_uv0, 2, vl);
    v_uv1_u16 = __riscv_vwaddu_vx_u16m8(v_uv1, 2, vl);

    v_uv0_u16 = __riscv_vwmaccu_vv_u16m8(v_uv0_u16, v_3_u8, v_uv1, vl);
    v_uv1_u16 = __riscv_vwmaccu_vv_u16m8(v_uv1_u16, v_3_u8, v_uv0, vl);

    v_dst_odd_u8 = __riscv_vnsrl_wx_u8m4(v_uv0_u16, 2, vl);
    v_dst_even_u8 = __riscv_vnsrl_wx_u8m4(v_uv1_u16, 2, vl);

    v_dst_even = __riscv_vreinterpret_v_u8m4_u16m4(v_dst_even_u8);
    v_dst_odd = __riscv_vreinterpret_v_u8m4_u16m4(v_dst_odd_u8);

    v_dst = __riscv_vcreate_v_u16m4x2(v_dst_even, v_dst_odd);
    __riscv_vsseg2e16_v_u16m4x2(work_dst_ptr, v_dst, vl / 2);

    work_width -= vl;
    work_src_ptr += vl;
    work_dst_ptr += vl;
  }
  dst_ptr[2 * dst_width - 2] = src_ptr[((dst_width + 1) & ~1) - 2];
  dst_ptr[2 * dst_width - 1] = src_ptr[((dst_width + 1) & ~1) - 1];
}
#else
void ScaleUVRowUp2_Linear_RVV(const uint8_t* src_ptr,
                              uint8_t* dst_ptr,
                              int dst_width) {
  size_t work_width = ((size_t)dst_width - 1u) & ~1u;
  uint16_t* work_dst_ptr = (uint16_t*)dst_ptr + 1;
  const uint8_t* work_src_ptr = src_ptr;
  size_t vl = __riscv_vsetvlmax_e8m4();
  vuint8m4_t v_3_u8 = __riscv_vmv_v_x_u8m4(3, vl);
  dst_ptr[0] = src_ptr[0];
  dst_ptr[1] = src_ptr[1];
  while (work_width > 0) {
    vuint8m4_t v_uv0, v_uv1, v_dst_odd_u8, v_dst_even_u8;
    vuint16m4_t v_dst_odd, v_dst_even;
    vuint16m8_t v_uv0_u16, v_uv1_u16;
    size_t vl = __riscv_vsetvl_e8m4(work_width);
    v_uv0 = __riscv_vle8_v_u8m4(work_src_ptr, vl);
    v_uv1 = __riscv_vle8_v_u8m4(work_src_ptr + 2, vl);

    v_uv0_u16 = __riscv_vwaddu_vx_u16m8(v_uv0, 2, vl);
    v_uv1_u16 = __riscv_vwaddu_vx_u16m8(v_uv1, 2, vl);

    v_uv0_u16 = __riscv_vwmaccu_vv_u16m8(v_uv0_u16, v_3_u8, v_uv1, vl);
    v_uv1_u16 = __riscv_vwmaccu_vv_u16m8(v_uv1_u16, v_3_u8, v_uv0, vl);

    v_dst_odd_u8 = __riscv_vnsrl_wx_u8m4(v_uv0_u16, 2, vl);
    v_dst_even_u8 = __riscv_vnsrl_wx_u8m4(v_uv1_u16, 2, vl);

    v_dst_even = __riscv_vreinterpret_v_u8m4_u16m4(v_dst_even_u8);
    v_dst_odd = __riscv_vreinterpret_v_u8m4_u16m4(v_dst_odd_u8);

    __riscv_vsseg2e16_v_u16m4(work_dst_ptr, v_dst_even, v_dst_odd, vl / 2);

    work_width -= vl;
    work_src_ptr += vl;
    work_dst_ptr += vl;
  }
  dst_ptr[2 * dst_width - 2] = src_ptr[((dst_width + 1) & ~1) - 2];
  dst_ptr[2 * dst_width - 1] = src_ptr[((dst_width + 1) & ~1) - 1];
}
#endif
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_RVV
#ifdef LIBYUV_RVV_HAS_TUPLE_TYPE
void ScaleUVRowUp2_Bilinear_RVV(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                ptrdiff_t dst_stride,
                                int dst_width) {
  size_t work_width = ((size_t)dst_width - 1u) & ~1u;
  const uint8_t* work_s = src_ptr;
  const uint8_t* work_t = src_ptr + src_stride;
  const uint8_t* s = work_s;
  const uint8_t* t = work_t;
  uint8_t* d = dst_ptr;
  uint8_t* e = dst_ptr + dst_stride;
  uint16_t* work_d = (uint16_t*)d + 1;
  uint16_t* work_e = (uint16_t*)e + 1;
  size_t vl = __riscv_vsetvlmax_e16m4();
  vuint16m4_t v_3_u16 = __riscv_vmv_v_x_u16m4(3, vl);
  vuint8m2_t v_3_u8 = __riscv_vmv_v_x_u8m2(3, vl);
  d[0] = (3 * s[0] + t[0] + 2) >> 2;
  e[0] = (s[0] + 3 * t[0] + 2) >> 2;
  d[1] = (3 * s[1] + t[1] + 2) >> 2;
  e[1] = (s[1] + 3 * t[1] + 2) >> 2;
  while (work_width > 0) {
    vuint8m2_t v_s0, v_s1, v_t0, v_t1;
    vuint16m4_t v_s0_u16, v_s1_u16, v_t0_u16, v_t1_u16;
    vuint16m4_t v_t0_u16_, v_t1_u16_;
    vuint8m2_t v_dst0_odd_u8, v_dst0_even_u8, v_dst1_odd_u8, v_dst1_even_u8;
    vuint16m2_t v_dst0_even, v_dst0_odd, v_dst1_even, v_dst1_odd;
    vuint16m2x2_t v_dst0, v_dst1;
    size_t vl = __riscv_vsetvl_e8m2(work_width);
    v_s0 = __riscv_vle8_v_u8m2(work_s, vl);
    v_s1 = __riscv_vle8_v_u8m2(work_s + 2, vl);

    v_s0_u16 = __riscv_vwaddu_vx_u16m4(v_s0, 2, vl);
    v_s1_u16 = __riscv_vwaddu_vx_u16m4(v_s1, 2, vl);
    v_s0_u16 = __riscv_vwmaccu_vv_u16m4(v_s0_u16, v_3_u8, v_s1, vl);
    v_s1_u16 = __riscv_vwmaccu_vv_u16m4(v_s1_u16, v_3_u8, v_s0, vl);

    v_t0 = __riscv_vle8_v_u8m2(work_t, vl);
    v_t1 = __riscv_vle8_v_u8m2(work_t + 2, vl);

    v_t0_u16 = __riscv_vwaddu_vx_u16m4(v_t0, 2, vl);
    v_t1_u16 = __riscv_vwaddu_vx_u16m4(v_t1, 2, vl);
    v_t0_u16 = __riscv_vwmaccu_vv_u16m4(v_t0_u16, v_3_u8, v_t1, vl);
    v_t1_u16 = __riscv_vwmaccu_vv_u16m4(v_t1_u16, v_3_u8, v_t0, vl);

    v_t0_u16_ = __riscv_vmv_v_v_u16m4(v_t0_u16, vl);
    v_t1_u16_ = __riscv_vmv_v_v_u16m4(v_t1_u16, vl);

    v_t0_u16 = __riscv_vmacc_vv_u16m4(v_t0_u16, v_3_u16, v_s0_u16, vl);
    v_t1_u16 = __riscv_vmacc_vv_u16m4(v_t1_u16, v_3_u16, v_s1_u16, vl);
    v_s0_u16 = __riscv_vmacc_vv_u16m4(v_s0_u16, v_3_u16, v_t0_u16_, vl);
    v_s1_u16 = __riscv_vmacc_vv_u16m4(v_s1_u16, v_3_u16, v_t1_u16_, vl);

    v_dst0_odd_u8 = __riscv_vnsrl_wx_u8m2(v_t0_u16, 4, vl);
    v_dst0_even_u8 = __riscv_vnsrl_wx_u8m2(v_t1_u16, 4, vl);
    v_dst1_odd_u8 = __riscv_vnsrl_wx_u8m2(v_s0_u16, 4, vl);
    v_dst1_even_u8 = __riscv_vnsrl_wx_u8m2(v_s1_u16, 4, vl);

    v_dst0_even = __riscv_vreinterpret_v_u8m2_u16m2(v_dst0_even_u8);
    v_dst0_odd = __riscv_vreinterpret_v_u8m2_u16m2(v_dst0_odd_u8);
    v_dst1_even = __riscv_vreinterpret_v_u8m2_u16m2(v_dst1_even_u8);
    v_dst1_odd = __riscv_vreinterpret_v_u8m2_u16m2(v_dst1_odd_u8);

    v_dst0 = __riscv_vcreate_v_u16m2x2(v_dst0_even, v_dst0_odd);
    __riscv_vsseg2e16_v_u16m2x2(work_d, v_dst0, vl / 2);
    v_dst1 = __riscv_vcreate_v_u16m2x2(v_dst1_even, v_dst1_odd);
    __riscv_vsseg2e16_v_u16m2x2(work_e, v_dst1, vl / 2);

    work_width -= vl;
    work_s += vl;
    work_t += vl;
    work_d += vl;
    work_e += vl;
  }
  d[2 * dst_width - 2] =
      (3 * s[((dst_width + 1) & ~1) - 2] + t[((dst_width + 1) & ~1) - 2] + 2) >>
      2;
  e[2 * dst_width - 2] =
      (s[((dst_width + 1) & ~1) - 2] + 3 * t[((dst_width + 1) & ~1) - 2] + 2) >>
      2;
  d[2 * dst_width - 1] =
      (3 * s[((dst_width + 1) & ~1) - 1] + t[((dst_width + 1) & ~1) - 1] + 2) >>
      2;
  e[2 * dst_width - 1] =
      (s[((dst_width + 1) & ~1) - 1] + 3 * t[((dst_width + 1) & ~1) - 1] + 2) >>
      2;
}
#else
void ScaleUVRowUp2_Bilinear_RVV(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                ptrdiff_t dst_stride,
                                int dst_width) {
  size_t work_width = ((size_t)dst_width - 1u) & ~1u;
  const uint8_t* work_s = src_ptr;
  const uint8_t* work_t = src_ptr + src_stride;
  const uint8_t* s = work_s;
  const uint8_t* t = work_t;
  uint8_t* d = dst_ptr;
  uint8_t* e = dst_ptr + dst_stride;
  uint16_t* work_d = (uint16_t*)d + 1;
  uint16_t* work_e = (uint16_t*)e + 1;
  size_t vl = __riscv_vsetvlmax_e16m4();
  vuint16m4_t v_3_u16 = __riscv_vmv_v_x_u16m4(3, vl);
  vuint8m2_t v_3_u8 = __riscv_vmv_v_x_u8m2(3, vl);
  d[0] = (3 * s[0] + t[0] + 2) >> 2;
  e[0] = (s[0] + 3 * t[0] + 2) >> 2;
  d[1] = (3 * s[1] + t[1] + 2) >> 2;
  e[1] = (s[1] + 3 * t[1] + 2) >> 2;
  while (work_width > 0) {
    vuint8m2_t v_s0, v_s1, v_t0, v_t1;
    vuint16m4_t v_s0_u16, v_s1_u16, v_t0_u16, v_t1_u16;
    vuint16m4_t v_t0_u16_, v_t1_u16_;
    vuint8m2_t v_dst0_odd_u8, v_dst0_even_u8, v_dst1_odd_u8, v_dst1_even_u8;
    vuint16m2_t v_dst0_even, v_dst0_odd, v_dst1_even, v_dst1_odd;
    size_t vl = __riscv_vsetvl_e8m2(work_width);
    v_s0 = __riscv_vle8_v_u8m2(work_s, vl);
    v_s1 = __riscv_vle8_v_u8m2(work_s + 2, vl);

    v_s0_u16 = __riscv_vwaddu_vx_u16m4(v_s0, 2, vl);
    v_s1_u16 = __riscv_vwaddu_vx_u16m4(v_s1, 2, vl);
    v_s0_u16 = __riscv_vwmaccu_vv_u16m4(v_s0_u16, v_3_u8, v_s1, vl);
    v_s1_u16 = __riscv_vwmaccu_vv_u16m4(v_s1_u16, v_3_u8, v_s0, vl);

    v_t0 = __riscv_vle8_v_u8m2(work_t, vl);
    v_t1 = __riscv_vle8_v_u8m2(work_t + 2, vl);

    v_t0_u16 = __riscv_vwaddu_vx_u16m4(v_t0, 2, vl);
    v_t1_u16 = __riscv_vwaddu_vx_u16m4(v_t1, 2, vl);
    v_t0_u16 = __riscv_vwmaccu_vv_u16m4(v_t0_u16, v_3_u8, v_t1, vl);
    v_t1_u16 = __riscv_vwmaccu_vv_u16m4(v_t1_u16, v_3_u8, v_t0, vl);

    v_t0_u16_ = __riscv_vmv_v_v_u16m4(v_t0_u16, vl);
    v_t1_u16_ = __riscv_vmv_v_v_u16m4(v_t1_u16, vl);

    v_t0_u16 = __riscv_vmacc_vv_u16m4(v_t0_u16, v_3_u16, v_s0_u16, vl);
    v_t1_u16 = __riscv_vmacc_vv_u16m4(v_t1_u16, v_3_u16, v_s1_u16, vl);
    v_s0_u16 = __riscv_vmacc_vv_u16m4(v_s0_u16, v_3_u16, v_t0_u16_, vl);
    v_s1_u16 = __riscv_vmacc_vv_u16m4(v_s1_u16, v_3_u16, v_t1_u16_, vl);

    v_dst0_odd_u8 = __riscv_vnsrl_wx_u8m2(v_t0_u16, 4, vl);
    v_dst0_even_u8 = __riscv_vnsrl_wx_u8m2(v_t1_u16, 4, vl);
    v_dst1_odd_u8 = __riscv_vnsrl_wx_u8m2(v_s0_u16, 4, vl);
    v_dst1_even_u8 = __riscv_vnsrl_wx_u8m2(v_s1_u16, 4, vl);

    v_dst0_even = __riscv_vreinterpret_v_u8m2_u16m2(v_dst0_even_u8);
    v_dst0_odd = __riscv_vreinterpret_v_u8m2_u16m2(v_dst0_odd_u8);
    v_dst1_even = __riscv_vreinterpret_v_u8m2_u16m2(v_dst1_even_u8);
    v_dst1_odd = __riscv_vreinterpret_v_u8m2_u16m2(v_dst1_odd_u8);

    __riscv_vsseg2e16_v_u16m2(work_d, v_dst0_even, v_dst0_odd, vl / 2);
    __riscv_vsseg2e16_v_u16m2(work_e, v_dst1_even, v_dst1_odd, vl / 2);

    work_width -= vl;
    work_s += vl;
    work_t += vl;
    work_d += vl;
    work_e += vl;
  }
  d[2 * dst_width - 2] =
      (3 * s[((dst_width + 1) & ~1) - 2] + t[((dst_width + 1) & ~1) - 2] + 2) >>
      2;
  e[2 * dst_width - 2] =
      (s[((dst_width + 1) & ~1) - 2] + 3 * t[((dst_width + 1) & ~1) - 2] + 2) >>
      2;
  d[2 * dst_width - 1] =
      (3 * s[((dst_width + 1) & ~1) - 1] + t[((dst_width + 1) & ~1) - 1] + 2) >>
      2;
  e[2 * dst_width - 1] =
      (s[((dst_width + 1) & ~1) - 1] + 3 * t[((dst_width + 1) & ~1) - 1] + 2) >>
      2;
}
#endif
#endif

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_RVV) && defined(__riscv_vector) &&
        // defined(__clang__)
