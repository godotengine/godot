// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// ARM NEON version of speed-critical encoding functions.
//
// adapted from libvpx (http://www.webmproject.org/code/)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_NEON)

#include <assert.h>

#include "src/dsp/neon.h"
#include "src/enc/vp8i_enc.h"

//------------------------------------------------------------------------------
// Transforms (Paragraph 14.4)

// Inverse transform.
// This code is pretty much the same as TransformOne in the dec_neon.c, except
// for subtraction to *ref. See the comments there for algorithmic explanations.

static const int16_t kC1 = 20091;
static const int16_t kC2 = 17734;  // half of kC2, actually. See comment above.

// This code works but is *slower* than the inlined-asm version below
// (with gcc-4.6). So we disable it for now. Later, it'll be conditional to
// WEBP_USE_INTRINSICS define.
// With gcc-4.8, it's a little faster speed than inlined-assembly.
#if defined(WEBP_USE_INTRINSICS)

// Treats 'v' as an uint8x8_t and zero extends to an int16x8_t.
static WEBP_INLINE int16x8_t ConvertU8ToS16_NEON(uint32x2_t v) {
  return vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(v)));
}

// Performs unsigned 8b saturation on 'dst01' and 'dst23' storing the result
// to the corresponding rows of 'dst'.
static WEBP_INLINE void SaturateAndStore4x4_NEON(uint8_t* const dst,
                                                 const int16x8_t dst01,
                                                 const int16x8_t dst23) {
  // Unsigned saturate to 8b.
  const uint8x8_t dst01_u8 = vqmovun_s16(dst01);
  const uint8x8_t dst23_u8 = vqmovun_s16(dst23);

  // Store the results.
  vst1_lane_u32((uint32_t*)(dst + 0 * BPS), vreinterpret_u32_u8(dst01_u8), 0);
  vst1_lane_u32((uint32_t*)(dst + 1 * BPS), vreinterpret_u32_u8(dst01_u8), 1);
  vst1_lane_u32((uint32_t*)(dst + 2 * BPS), vreinterpret_u32_u8(dst23_u8), 0);
  vst1_lane_u32((uint32_t*)(dst + 3 * BPS), vreinterpret_u32_u8(dst23_u8), 1);
}

static WEBP_INLINE void Add4x4_NEON(const int16x8_t row01,
                                    const int16x8_t row23,
                                    const uint8_t* const ref,
                                    uint8_t* const dst) {
  uint32x2_t dst01 = vdup_n_u32(0);
  uint32x2_t dst23 = vdup_n_u32(0);

  // Load the source pixels.
  dst01 = vld1_lane_u32((uint32_t*)(ref + 0 * BPS), dst01, 0);
  dst23 = vld1_lane_u32((uint32_t*)(ref + 2 * BPS), dst23, 0);
  dst01 = vld1_lane_u32((uint32_t*)(ref + 1 * BPS), dst01, 1);
  dst23 = vld1_lane_u32((uint32_t*)(ref + 3 * BPS), dst23, 1);

  {
    // Convert to 16b.
    const int16x8_t dst01_s16 = ConvertU8ToS16_NEON(dst01);
    const int16x8_t dst23_s16 = ConvertU8ToS16_NEON(dst23);

    // Descale with rounding.
    const int16x8_t out01 = vrsraq_n_s16(dst01_s16, row01, 3);
    const int16x8_t out23 = vrsraq_n_s16(dst23_s16, row23, 3);
    // Add the inverse transform.
    SaturateAndStore4x4_NEON(dst, out01, out23);
  }
}

static WEBP_INLINE void Transpose8x2_NEON(const int16x8_t in0,
                                          const int16x8_t in1,
                                          int16x8x2_t* const out) {
  // a0 a1 a2 a3 | b0 b1 b2 b3   => a0 b0 c0 d0 | a1 b1 c1 d1
  // c0 c1 c2 c3 | d0 d1 d2 d3      a2 b2 c2 d2 | a3 b3 c3 d3
  const int16x8x2_t tmp0 = vzipq_s16(in0, in1);   // a0 c0 a1 c1 a2 c2 ...
                                                  // b0 d0 b1 d1 b2 d2 ...
  *out = vzipq_s16(tmp0.val[0], tmp0.val[1]);
}

static WEBP_INLINE void TransformPass_NEON(int16x8x2_t* const rows) {
  // {rows} = in0 | in4
  //          in8 | in12
  // B1 = in4 | in12
  const int16x8_t B1 =
      vcombine_s16(vget_high_s16(rows->val[0]), vget_high_s16(rows->val[1]));
  // C0 = kC1 * in4 | kC1 * in12
  // C1 = kC2 * in4 | kC2 * in12
  const int16x8_t C0 = vsraq_n_s16(B1, vqdmulhq_n_s16(B1, kC1), 1);
  const int16x8_t C1 = vqdmulhq_n_s16(B1, kC2);
  const int16x4_t a = vqadd_s16(vget_low_s16(rows->val[0]),
                                vget_low_s16(rows->val[1]));   // in0 + in8
  const int16x4_t b = vqsub_s16(vget_low_s16(rows->val[0]),
                                vget_low_s16(rows->val[1]));   // in0 - in8
  // c = kC2 * in4 - kC1 * in12
  // d = kC1 * in4 + kC2 * in12
  const int16x4_t c = vqsub_s16(vget_low_s16(C1), vget_high_s16(C0));
  const int16x4_t d = vqadd_s16(vget_low_s16(C0), vget_high_s16(C1));
  const int16x8_t D0 = vcombine_s16(a, b);      // D0 = a | b
  const int16x8_t D1 = vcombine_s16(d, c);      // D1 = d | c
  const int16x8_t E0 = vqaddq_s16(D0, D1);      // a+d | b+c
  const int16x8_t E_tmp = vqsubq_s16(D0, D1);   // a-d | b-c
  const int16x8_t E1 = vcombine_s16(vget_high_s16(E_tmp), vget_low_s16(E_tmp));
  Transpose8x2_NEON(E0, E1, rows);
}

static void ITransformOne_NEON(const uint8_t* ref,
                               const int16_t* in, uint8_t* dst) {
  int16x8x2_t rows;
  INIT_VECTOR2(rows, vld1q_s16(in + 0), vld1q_s16(in + 8));
  TransformPass_NEON(&rows);
  TransformPass_NEON(&rows);
  Add4x4_NEON(rows.val[0], rows.val[1], ref, dst);
}

#else

static void ITransformOne_NEON(const uint8_t* ref,
                               const int16_t* in, uint8_t* dst) {
  const int kBPS = BPS;
  const int16_t kC1C2[] = { kC1, kC2, 0, 0 };

  __asm__ volatile (
    "vld1.16         {q1, q2}, [%[in]]           \n"
    "vld1.16         {d0}, [%[kC1C2]]            \n"

    // d2: in[0]
    // d3: in[8]
    // d4: in[4]
    // d5: in[12]
    "vswp            d3, d4                      \n"

    // q8 = {in[4], in[12]} * kC1 * 2 >> 16
    // q9 = {in[4], in[12]} * kC2 >> 16
    "vqdmulh.s16     q8, q2, d0[0]               \n"
    "vqdmulh.s16     q9, q2, d0[1]               \n"

    // d22 = a = in[0] + in[8]
    // d23 = b = in[0] - in[8]
    "vqadd.s16       d22, d2, d3                 \n"
    "vqsub.s16       d23, d2, d3                 \n"

    //  q8 = in[4]/[12] * kC1 >> 16
    "vshr.s16        q8, q8, #1                  \n"

    // Add {in[4], in[12]} back after the multiplication.
    "vqadd.s16       q8, q2, q8                  \n"

    // d20 = c = in[4]*kC2 - in[12]*kC1
    // d21 = d = in[4]*kC1 + in[12]*kC2
    "vqsub.s16       d20, d18, d17               \n"
    "vqadd.s16       d21, d19, d16               \n"

    // d2 = tmp[0] = a + d
    // d3 = tmp[1] = b + c
    // d4 = tmp[2] = b - c
    // d5 = tmp[3] = a - d
    "vqadd.s16       d2, d22, d21                \n"
    "vqadd.s16       d3, d23, d20                \n"
    "vqsub.s16       d4, d23, d20                \n"
    "vqsub.s16       d5, d22, d21                \n"

    "vzip.16         q1, q2                      \n"
    "vzip.16         q1, q2                      \n"

    "vswp            d3, d4                      \n"

    // q8 = {tmp[4], tmp[12]} * kC1 * 2 >> 16
    // q9 = {tmp[4], tmp[12]} * kC2 >> 16
    "vqdmulh.s16     q8, q2, d0[0]               \n"
    "vqdmulh.s16     q9, q2, d0[1]               \n"

    // d22 = a = tmp[0] + tmp[8]
    // d23 = b = tmp[0] - tmp[8]
    "vqadd.s16       d22, d2, d3                 \n"
    "vqsub.s16       d23, d2, d3                 \n"

    "vshr.s16        q8, q8, #1                  \n"
    "vqadd.s16       q8, q2, q8                  \n"

    // d20 = c = in[4]*kC2 - in[12]*kC1
    // d21 = d = in[4]*kC1 + in[12]*kC2
    "vqsub.s16       d20, d18, d17               \n"
    "vqadd.s16       d21, d19, d16               \n"

    // d2 = tmp[0] = a + d
    // d3 = tmp[1] = b + c
    // d4 = tmp[2] = b - c
    // d5 = tmp[3] = a - d
    "vqadd.s16       d2, d22, d21                \n"
    "vqadd.s16       d3, d23, d20                \n"
    "vqsub.s16       d4, d23, d20                \n"
    "vqsub.s16       d5, d22, d21                \n"

    "vld1.32         d6[0], [%[ref]], %[kBPS]    \n"
    "vld1.32         d6[1], [%[ref]], %[kBPS]    \n"
    "vld1.32         d7[0], [%[ref]], %[kBPS]    \n"
    "vld1.32         d7[1], [%[ref]], %[kBPS]    \n"

    "sub         %[ref], %[ref], %[kBPS], lsl #2 \n"

    // (val) + 4 >> 3
    "vrshr.s16       d2, d2, #3                  \n"
    "vrshr.s16       d3, d3, #3                  \n"
    "vrshr.s16       d4, d4, #3                  \n"
    "vrshr.s16       d5, d5, #3                  \n"

    "vzip.16         q1, q2                      \n"
    "vzip.16         q1, q2                      \n"

    // Must accumulate before saturating
    "vmovl.u8        q8, d6                      \n"
    "vmovl.u8        q9, d7                      \n"

    "vqadd.s16       q1, q1, q8                  \n"
    "vqadd.s16       q2, q2, q9                  \n"

    "vqmovun.s16     d0, q1                      \n"
    "vqmovun.s16     d1, q2                      \n"

    "vst1.32         d0[0], [%[dst]], %[kBPS]    \n"
    "vst1.32         d0[1], [%[dst]], %[kBPS]    \n"
    "vst1.32         d1[0], [%[dst]], %[kBPS]    \n"
    "vst1.32         d1[1], [%[dst]]             \n"

    : [in] "+r"(in), [dst] "+r"(dst)               // modified registers
    : [kBPS] "r"(kBPS), [kC1C2] "r"(kC1C2), [ref] "r"(ref)  // constants
    : "memory", "q0", "q1", "q2", "q8", "q9", "q10", "q11"  // clobbered
  );
}

#endif    // WEBP_USE_INTRINSICS

static void ITransform_NEON(const uint8_t* ref,
                            const int16_t* in, uint8_t* dst, int do_two) {
  ITransformOne_NEON(ref, in, dst);
  if (do_two) {
    ITransformOne_NEON(ref + 4, in + 16, dst + 4);
  }
}

// Load all 4x4 pixels into a single uint8x16_t variable.
static uint8x16_t Load4x4_NEON(const uint8_t* src) {
  uint32x4_t out = vdupq_n_u32(0);
  out = vld1q_lane_u32((const uint32_t*)(src + 0 * BPS), out, 0);
  out = vld1q_lane_u32((const uint32_t*)(src + 1 * BPS), out, 1);
  out = vld1q_lane_u32((const uint32_t*)(src + 2 * BPS), out, 2);
  out = vld1q_lane_u32((const uint32_t*)(src + 3 * BPS), out, 3);
  return vreinterpretq_u8_u32(out);
}

// Forward transform.

#if defined(WEBP_USE_INTRINSICS)

static WEBP_INLINE void Transpose4x4_S16_NEON(const int16x4_t A,
                                              const int16x4_t B,
                                              const int16x4_t C,
                                              const int16x4_t D,
                                              int16x8_t* const out01,
                                              int16x8_t* const out32) {
  const int16x4x2_t AB = vtrn_s16(A, B);
  const int16x4x2_t CD = vtrn_s16(C, D);
  const int32x2x2_t tmp02 = vtrn_s32(vreinterpret_s32_s16(AB.val[0]),
                                     vreinterpret_s32_s16(CD.val[0]));
  const int32x2x2_t tmp13 = vtrn_s32(vreinterpret_s32_s16(AB.val[1]),
                                     vreinterpret_s32_s16(CD.val[1]));
  *out01 = vreinterpretq_s16_s64(
      vcombine_s64(vreinterpret_s64_s32(tmp02.val[0]),
                   vreinterpret_s64_s32(tmp13.val[0])));
  *out32 = vreinterpretq_s16_s64(
      vcombine_s64(vreinterpret_s64_s32(tmp13.val[1]),
                   vreinterpret_s64_s32(tmp02.val[1])));
}

static WEBP_INLINE int16x8_t DiffU8ToS16_NEON(const uint8x8_t a,
                                              const uint8x8_t b) {
  return vreinterpretq_s16_u16(vsubl_u8(a, b));
}

static void FTransform_NEON(const uint8_t* src, const uint8_t* ref,
                            int16_t* out) {
  int16x8_t d0d1, d3d2;   // working 4x4 int16 variables
  {
    const uint8x16_t S0 = Load4x4_NEON(src);
    const uint8x16_t R0 = Load4x4_NEON(ref);
    const int16x8_t D0D1 = DiffU8ToS16_NEON(vget_low_u8(S0), vget_low_u8(R0));
    const int16x8_t D2D3 = DiffU8ToS16_NEON(vget_high_u8(S0), vget_high_u8(R0));
    const int16x4_t D0 = vget_low_s16(D0D1);
    const int16x4_t D1 = vget_high_s16(D0D1);
    const int16x4_t D2 = vget_low_s16(D2D3);
    const int16x4_t D3 = vget_high_s16(D2D3);
    Transpose4x4_S16_NEON(D0, D1, D2, D3, &d0d1, &d3d2);
  }
  {    // 1rst pass
    const int32x4_t kCst937 = vdupq_n_s32(937);
    const int32x4_t kCst1812 = vdupq_n_s32(1812);
    const int16x8_t a0a1 = vaddq_s16(d0d1, d3d2);   // d0+d3 | d1+d2   (=a0|a1)
    const int16x8_t a3a2 = vsubq_s16(d0d1, d3d2);   // d0-d3 | d1-d2   (=a3|a2)
    const int16x8_t a0a1_2 = vshlq_n_s16(a0a1, 3);
    const int16x4_t tmp0 = vadd_s16(vget_low_s16(a0a1_2),
                                    vget_high_s16(a0a1_2));
    const int16x4_t tmp2 = vsub_s16(vget_low_s16(a0a1_2),
                                    vget_high_s16(a0a1_2));
    const int32x4_t a3_2217 = vmull_n_s16(vget_low_s16(a3a2), 2217);
    const int32x4_t a2_2217 = vmull_n_s16(vget_high_s16(a3a2), 2217);
    const int32x4_t a2_p_a3 = vmlal_n_s16(a2_2217, vget_low_s16(a3a2), 5352);
    const int32x4_t a3_m_a2 = vmlsl_n_s16(a3_2217, vget_high_s16(a3a2), 5352);
    const int16x4_t tmp1 = vshrn_n_s32(vaddq_s32(a2_p_a3, kCst1812), 9);
    const int16x4_t tmp3 = vshrn_n_s32(vaddq_s32(a3_m_a2, kCst937), 9);
    Transpose4x4_S16_NEON(tmp0, tmp1, tmp2, tmp3, &d0d1, &d3d2);
  }
  {    // 2nd pass
    // the (1<<16) addition is for the replacement: a3!=0  <-> 1-(a3==0)
    const int32x4_t kCst12000 = vdupq_n_s32(12000 + (1 << 16));
    const int32x4_t kCst51000 = vdupq_n_s32(51000);
    const int16x8_t a0a1 = vaddq_s16(d0d1, d3d2);   // d0+d3 | d1+d2   (=a0|a1)
    const int16x8_t a3a2 = vsubq_s16(d0d1, d3d2);   // d0-d3 | d1-d2   (=a3|a2)
    const int16x4_t a0_k7 = vadd_s16(vget_low_s16(a0a1), vdup_n_s16(7));
    const int16x4_t out0 = vshr_n_s16(vadd_s16(a0_k7, vget_high_s16(a0a1)), 4);
    const int16x4_t out2 = vshr_n_s16(vsub_s16(a0_k7, vget_high_s16(a0a1)), 4);
    const int32x4_t a3_2217 = vmull_n_s16(vget_low_s16(a3a2), 2217);
    const int32x4_t a2_2217 = vmull_n_s16(vget_high_s16(a3a2), 2217);
    const int32x4_t a2_p_a3 = vmlal_n_s16(a2_2217, vget_low_s16(a3a2), 5352);
    const int32x4_t a3_m_a2 = vmlsl_n_s16(a3_2217, vget_high_s16(a3a2), 5352);
    const int16x4_t tmp1 = vaddhn_s32(a2_p_a3, kCst12000);
    const int16x4_t out3 = vaddhn_s32(a3_m_a2, kCst51000);
    const int16x4_t a3_eq_0 =
        vreinterpret_s16_u16(vceq_s16(vget_low_s16(a3a2), vdup_n_s16(0)));
    const int16x4_t out1 = vadd_s16(tmp1, a3_eq_0);
    vst1_s16(out +  0, out0);
    vst1_s16(out +  4, out1);
    vst1_s16(out +  8, out2);
    vst1_s16(out + 12, out3);
  }
}

#else

// adapted from vp8/encoder/arm/neon/shortfdct_neon.asm
static const int16_t kCoeff16[] = {
  5352,  5352,  5352, 5352, 2217,  2217,  2217, 2217
};
static const int32_t kCoeff32[] = {
   1812,  1812,  1812,  1812,
    937,   937,   937,   937,
  12000, 12000, 12000, 12000,
  51000, 51000, 51000, 51000
};

static void FTransform_NEON(const uint8_t* src, const uint8_t* ref,
                            int16_t* out) {
  const int kBPS = BPS;
  const uint8_t* src_ptr = src;
  const uint8_t* ref_ptr = ref;
  const int16_t* coeff16 = kCoeff16;
  const int32_t* coeff32 = kCoeff32;

  __asm__ volatile (
    // load src into q4, q5 in high half
    "vld1.8 {d8},  [%[src_ptr]], %[kBPS]      \n"
    "vld1.8 {d10}, [%[src_ptr]], %[kBPS]      \n"
    "vld1.8 {d9},  [%[src_ptr]], %[kBPS]      \n"
    "vld1.8 {d11}, [%[src_ptr]]               \n"

    // load ref into q6, q7 in high half
    "vld1.8 {d12}, [%[ref_ptr]], %[kBPS]      \n"
    "vld1.8 {d14}, [%[ref_ptr]], %[kBPS]      \n"
    "vld1.8 {d13}, [%[ref_ptr]], %[kBPS]      \n"
    "vld1.8 {d15}, [%[ref_ptr]]               \n"

    // Pack the high values in to q4 and q6
    "vtrn.32     q4, q5                       \n"
    "vtrn.32     q6, q7                       \n"

    // d[0-3] = src - ref
    "vsubl.u8    q0, d8, d12                  \n"
    "vsubl.u8    q1, d9, d13                  \n"

    // load coeff16 into q8(d16=5352, d17=2217)
    "vld1.16     {q8}, [%[coeff16]]           \n"

    // load coeff32 high half into q9 = 1812, q10 = 937
    "vld1.32     {q9, q10}, [%[coeff32]]!     \n"

    // load coeff32 low half into q11=12000, q12=51000
    "vld1.32     {q11,q12}, [%[coeff32]]      \n"

    // part 1
    // Transpose. Register dN is the same as dN in C
    "vtrn.32         d0, d2                   \n"
    "vtrn.32         d1, d3                   \n"
    "vtrn.16         d0, d1                   \n"
    "vtrn.16         d2, d3                   \n"

    "vadd.s16        d4, d0, d3               \n" // a0 = d0 + d3
    "vadd.s16        d5, d1, d2               \n" // a1 = d1 + d2
    "vsub.s16        d6, d1, d2               \n" // a2 = d1 - d2
    "vsub.s16        d7, d0, d3               \n" // a3 = d0 - d3

    "vadd.s16        d0, d4, d5               \n" // a0 + a1
    "vshl.s16        d0, d0, #3               \n" // temp[0+i*4] = (a0+a1) << 3
    "vsub.s16        d2, d4, d5               \n" // a0 - a1
    "vshl.s16        d2, d2, #3               \n" // (temp[2+i*4] = (a0-a1) << 3

    "vmlal.s16       q9, d7, d16              \n" // a3*5352 + 1812
    "vmlal.s16       q10, d7, d17             \n" // a3*2217 + 937
    "vmlal.s16       q9, d6, d17              \n" // a2*2217 + a3*5352 + 1812
    "vmlsl.s16       q10, d6, d16             \n" // a3*2217 + 937 - a2*5352

    // temp[1+i*4] = (d2*2217 + d3*5352 + 1812) >> 9
    // temp[3+i*4] = (d3*2217 + 937 - d2*5352) >> 9
    "vshrn.s32       d1, q9, #9               \n"
    "vshrn.s32       d3, q10, #9              \n"

    // part 2
    // transpose d0=ip[0], d1=ip[4], d2=ip[8], d3=ip[12]
    "vtrn.32         d0, d2                   \n"
    "vtrn.32         d1, d3                   \n"
    "vtrn.16         d0, d1                   \n"
    "vtrn.16         d2, d3                   \n"

    "vmov.s16        d26, #7                  \n"

    "vadd.s16        d4, d0, d3               \n" // a1 = ip[0] + ip[12]
    "vadd.s16        d5, d1, d2               \n" // b1 = ip[4] + ip[8]
    "vsub.s16        d6, d1, d2               \n" // c1 = ip[4] - ip[8]
    "vadd.s16        d4, d4, d26              \n" // a1 + 7
    "vsub.s16        d7, d0, d3               \n" // d1 = ip[0] - ip[12]

    "vadd.s16        d0, d4, d5               \n" // op[0] = a1 + b1 + 7
    "vsub.s16        d2, d4, d5               \n" // op[8] = a1 - b1 + 7

    "vmlal.s16       q11, d7, d16             \n" // d1*5352 + 12000
    "vmlal.s16       q12, d7, d17             \n" // d1*2217 + 51000

    "vceq.s16        d4, d7, #0               \n"

    "vshr.s16        d0, d0, #4               \n"
    "vshr.s16        d2, d2, #4               \n"

    "vmlal.s16       q11, d6, d17             \n" // c1*2217 + d1*5352 + 12000
    "vmlsl.s16       q12, d6, d16             \n" // d1*2217 - c1*5352 + 51000

    "vmvn            d4, d4                   \n" // !(d1 == 0)
    // op[4] = (c1*2217 + d1*5352 + 12000)>>16
    "vshrn.s32       d1, q11, #16             \n"
    // op[4] += (d1!=0)
    "vsub.s16        d1, d1, d4               \n"
    // op[12]= (d1*2217 - c1*5352 + 51000)>>16
    "vshrn.s32       d3, q12, #16             \n"

    // set result to out array
    "vst1.16         {q0, q1}, [%[out]]   \n"
    : [src_ptr] "+r"(src_ptr), [ref_ptr] "+r"(ref_ptr),
      [coeff32] "+r"(coeff32)          // modified registers
    : [kBPS] "r"(kBPS), [coeff16] "r"(coeff16),
      [out] "r"(out)                   // constants
    : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
      "q10", "q11", "q12", "q13"       // clobbered
  );
}

#endif

#define LOAD_LANE_16b(VALUE, LANE) do {             \
  (VALUE) = vld1_lane_s16(src, (VALUE), (LANE));    \
  src += stride;                                    \
} while (0)

static void FTransformWHT_NEON(const int16_t* src, int16_t* out) {
  const int stride = 16;
  const int16x4_t zero = vdup_n_s16(0);
  int32x4x4_t tmp0;
  int16x4x4_t in;
  INIT_VECTOR4(in, zero, zero, zero, zero);
  LOAD_LANE_16b(in.val[0], 0);
  LOAD_LANE_16b(in.val[1], 0);
  LOAD_LANE_16b(in.val[2], 0);
  LOAD_LANE_16b(in.val[3], 0);
  LOAD_LANE_16b(in.val[0], 1);
  LOAD_LANE_16b(in.val[1], 1);
  LOAD_LANE_16b(in.val[2], 1);
  LOAD_LANE_16b(in.val[3], 1);
  LOAD_LANE_16b(in.val[0], 2);
  LOAD_LANE_16b(in.val[1], 2);
  LOAD_LANE_16b(in.val[2], 2);
  LOAD_LANE_16b(in.val[3], 2);
  LOAD_LANE_16b(in.val[0], 3);
  LOAD_LANE_16b(in.val[1], 3);
  LOAD_LANE_16b(in.val[2], 3);
  LOAD_LANE_16b(in.val[3], 3);

  {
    // a0 = in[0 * 16] + in[2 * 16]
    // a1 = in[1 * 16] + in[3 * 16]
    // a2 = in[1 * 16] - in[3 * 16]
    // a3 = in[0 * 16] - in[2 * 16]
    const int32x4_t a0 = vaddl_s16(in.val[0], in.val[2]);
    const int32x4_t a1 = vaddl_s16(in.val[1], in.val[3]);
    const int32x4_t a2 = vsubl_s16(in.val[1], in.val[3]);
    const int32x4_t a3 = vsubl_s16(in.val[0], in.val[2]);
    tmp0.val[0] = vaddq_s32(a0, a1);
    tmp0.val[1] = vaddq_s32(a3, a2);
    tmp0.val[2] = vsubq_s32(a3, a2);
    tmp0.val[3] = vsubq_s32(a0, a1);
  }
  {
    const int32x4x4_t tmp1 = Transpose4x4_NEON(tmp0);
    // a0 = tmp[0 + i] + tmp[ 8 + i]
    // a1 = tmp[4 + i] + tmp[12 + i]
    // a2 = tmp[4 + i] - tmp[12 + i]
    // a3 = tmp[0 + i] - tmp[ 8 + i]
    const int32x4_t a0 = vaddq_s32(tmp1.val[0], tmp1.val[2]);
    const int32x4_t a1 = vaddq_s32(tmp1.val[1], tmp1.val[3]);
    const int32x4_t a2 = vsubq_s32(tmp1.val[1], tmp1.val[3]);
    const int32x4_t a3 = vsubq_s32(tmp1.val[0], tmp1.val[2]);
    const int32x4_t b0 = vhaddq_s32(a0, a1);  // (a0 + a1) >> 1
    const int32x4_t b1 = vhaddq_s32(a3, a2);  // (a3 + a2) >> 1
    const int32x4_t b2 = vhsubq_s32(a3, a2);  // (a3 - a2) >> 1
    const int32x4_t b3 = vhsubq_s32(a0, a1);  // (a0 - a1) >> 1
    const int16x4_t out0 = vmovn_s32(b0);
    const int16x4_t out1 = vmovn_s32(b1);
    const int16x4_t out2 = vmovn_s32(b2);
    const int16x4_t out3 = vmovn_s32(b3);

    vst1_s16(out +  0, out0);
    vst1_s16(out +  4, out1);
    vst1_s16(out +  8, out2);
    vst1_s16(out + 12, out3);
  }
}
#undef LOAD_LANE_16b

//------------------------------------------------------------------------------
// Texture distortion
//
// We try to match the spectral content (weighted) between source and
// reconstructed samples.

// a 0123, b 0123
// a 4567, b 4567
// a 89ab, b 89ab
// a cdef, b cdef
//
// transpose
//
// a 048c, b 048c
// a 159d, b 159d
// a 26ae, b 26ae
// a 37bf, b 37bf
//
static WEBP_INLINE int16x8x4_t DistoTranspose4x4S16_NEON(int16x8x4_t q4_in) {
  const int16x8x2_t q2_tmp0 = vtrnq_s16(q4_in.val[0], q4_in.val[1]);
  const int16x8x2_t q2_tmp1 = vtrnq_s16(q4_in.val[2], q4_in.val[3]);
  const int32x4x2_t q2_tmp2 = vtrnq_s32(vreinterpretq_s32_s16(q2_tmp0.val[0]),
                                        vreinterpretq_s32_s16(q2_tmp1.val[0]));
  const int32x4x2_t q2_tmp3 = vtrnq_s32(vreinterpretq_s32_s16(q2_tmp0.val[1]),
                                        vreinterpretq_s32_s16(q2_tmp1.val[1]));
  q4_in.val[0] = vreinterpretq_s16_s32(q2_tmp2.val[0]);
  q4_in.val[2] = vreinterpretq_s16_s32(q2_tmp2.val[1]);
  q4_in.val[1] = vreinterpretq_s16_s32(q2_tmp3.val[0]);
  q4_in.val[3] = vreinterpretq_s16_s32(q2_tmp3.val[1]);
  return q4_in;
}

static WEBP_INLINE int16x8x4_t DistoHorizontalPass_NEON(
    const int16x8x4_t q4_in) {
  // {a0, a1} = {in[0] + in[2], in[1] + in[3]}
  // {a3, a2} = {in[0] - in[2], in[1] - in[3]}
  const int16x8_t q_a0 = vaddq_s16(q4_in.val[0], q4_in.val[2]);
  const int16x8_t q_a1 = vaddq_s16(q4_in.val[1], q4_in.val[3]);
  const int16x8_t q_a3 = vsubq_s16(q4_in.val[0], q4_in.val[2]);
  const int16x8_t q_a2 = vsubq_s16(q4_in.val[1], q4_in.val[3]);
  int16x8x4_t q4_out;
  // tmp[0] = a0 + a1
  // tmp[1] = a3 + a2
  // tmp[2] = a3 - a2
  // tmp[3] = a0 - a1
  INIT_VECTOR4(q4_out,
               vabsq_s16(vaddq_s16(q_a0, q_a1)),
               vabsq_s16(vaddq_s16(q_a3, q_a2)),
               vabdq_s16(q_a3, q_a2), vabdq_s16(q_a0, q_a1));
  return q4_out;
}

static WEBP_INLINE int16x8x4_t DistoVerticalPass_NEON(const uint8x8x4_t q4_in) {
  const int16x8_t q_a0 = vreinterpretq_s16_u16(vaddl_u8(q4_in.val[0],
                                                        q4_in.val[2]));
  const int16x8_t q_a1 = vreinterpretq_s16_u16(vaddl_u8(q4_in.val[1],
                                                        q4_in.val[3]));
  const int16x8_t q_a2 = vreinterpretq_s16_u16(vsubl_u8(q4_in.val[1],
                                                        q4_in.val[3]));
  const int16x8_t q_a3 = vreinterpretq_s16_u16(vsubl_u8(q4_in.val[0],
                                                        q4_in.val[2]));
  int16x8x4_t q4_out;

  INIT_VECTOR4(q4_out,
               vaddq_s16(q_a0, q_a1), vaddq_s16(q_a3, q_a2),
               vsubq_s16(q_a3, q_a2), vsubq_s16(q_a0, q_a1));
  return q4_out;
}

static WEBP_INLINE int16x4x4_t DistoLoadW_NEON(const uint16_t* w) {
  const uint16x8_t q_w07 = vld1q_u16(&w[0]);
  const uint16x8_t q_w8f = vld1q_u16(&w[8]);
  int16x4x4_t d4_w;
  INIT_VECTOR4(d4_w,
               vget_low_s16(vreinterpretq_s16_u16(q_w07)),
               vget_high_s16(vreinterpretq_s16_u16(q_w07)),
               vget_low_s16(vreinterpretq_s16_u16(q_w8f)),
               vget_high_s16(vreinterpretq_s16_u16(q_w8f)));
  return d4_w;
}

static WEBP_INLINE int32x2_t DistoSum_NEON(const int16x8x4_t q4_in,
                                           const int16x4x4_t d4_w) {
  int32x2_t d_sum;
  // sum += w[ 0] * abs(b0);
  // sum += w[ 4] * abs(b1);
  // sum += w[ 8] * abs(b2);
  // sum += w[12] * abs(b3);
  int32x4_t q_sum0 = vmull_s16(d4_w.val[0], vget_low_s16(q4_in.val[0]));
  int32x4_t q_sum1 = vmull_s16(d4_w.val[1], vget_low_s16(q4_in.val[1]));
  int32x4_t q_sum2 = vmull_s16(d4_w.val[2], vget_low_s16(q4_in.val[2]));
  int32x4_t q_sum3 = vmull_s16(d4_w.val[3], vget_low_s16(q4_in.val[3]));
  q_sum0 = vmlsl_s16(q_sum0, d4_w.val[0], vget_high_s16(q4_in.val[0]));
  q_sum1 = vmlsl_s16(q_sum1, d4_w.val[1], vget_high_s16(q4_in.val[1]));
  q_sum2 = vmlsl_s16(q_sum2, d4_w.val[2], vget_high_s16(q4_in.val[2]));
  q_sum3 = vmlsl_s16(q_sum3, d4_w.val[3], vget_high_s16(q4_in.val[3]));

  q_sum0 = vaddq_s32(q_sum0, q_sum1);
  q_sum2 = vaddq_s32(q_sum2, q_sum3);
  q_sum2 = vaddq_s32(q_sum0, q_sum2);
  d_sum = vpadd_s32(vget_low_s32(q_sum2), vget_high_s32(q_sum2));
  d_sum = vpadd_s32(d_sum, d_sum);
  return d_sum;
}

#define LOAD_LANE_32b(src, VALUE, LANE) \
    (VALUE) = vld1_lane_u32((const uint32_t*)(src), (VALUE), (LANE))

// Hadamard transform
// Returns the weighted sum of the absolute value of transformed coefficients.
// w[] contains a row-major 4 by 4 symmetric matrix.
static int Disto4x4_NEON(const uint8_t* const a, const uint8_t* const b,
                         const uint16_t* const w) {
  uint32x2_t d_in_ab_0123 = vdup_n_u32(0);
  uint32x2_t d_in_ab_4567 = vdup_n_u32(0);
  uint32x2_t d_in_ab_89ab = vdup_n_u32(0);
  uint32x2_t d_in_ab_cdef = vdup_n_u32(0);
  uint8x8x4_t d4_in;

  // load data a, b
  LOAD_LANE_32b(a + 0 * BPS, d_in_ab_0123, 0);
  LOAD_LANE_32b(a + 1 * BPS, d_in_ab_4567, 0);
  LOAD_LANE_32b(a + 2 * BPS, d_in_ab_89ab, 0);
  LOAD_LANE_32b(a + 3 * BPS, d_in_ab_cdef, 0);
  LOAD_LANE_32b(b + 0 * BPS, d_in_ab_0123, 1);
  LOAD_LANE_32b(b + 1 * BPS, d_in_ab_4567, 1);
  LOAD_LANE_32b(b + 2 * BPS, d_in_ab_89ab, 1);
  LOAD_LANE_32b(b + 3 * BPS, d_in_ab_cdef, 1);
  INIT_VECTOR4(d4_in,
               vreinterpret_u8_u32(d_in_ab_0123),
               vreinterpret_u8_u32(d_in_ab_4567),
               vreinterpret_u8_u32(d_in_ab_89ab),
               vreinterpret_u8_u32(d_in_ab_cdef));

  {
    // Vertical pass first to avoid a transpose (vertical and horizontal passes
    // are commutative because w/kWeightY is symmetric) and subsequent
    // transpose.
    const int16x8x4_t q4_v = DistoVerticalPass_NEON(d4_in);
    const int16x4x4_t d4_w = DistoLoadW_NEON(w);
    // horizontal pass
    const int16x8x4_t q4_t = DistoTranspose4x4S16_NEON(q4_v);
    const int16x8x4_t q4_h = DistoHorizontalPass_NEON(q4_t);
    int32x2_t d_sum = DistoSum_NEON(q4_h, d4_w);

    // abs(sum2 - sum1) >> 5
    d_sum = vabs_s32(d_sum);
    d_sum = vshr_n_s32(d_sum, 5);
    return vget_lane_s32(d_sum, 0);
  }
}
#undef LOAD_LANE_32b

static int Disto16x16_NEON(const uint8_t* const a, const uint8_t* const b,
                           const uint16_t* const w) {
  int D = 0;
  int x, y;
  for (y = 0; y < 16 * BPS; y += 4 * BPS) {
    for (x = 0; x < 16; x += 4) {
      D += Disto4x4_NEON(a + x + y, b + x + y, w);
    }
  }
  return D;
}

//------------------------------------------------------------------------------

static void CollectHistogram_NEON(const uint8_t* ref, const uint8_t* pred,
                                  int start_block, int end_block,
                                  VP8Histogram* const histo) {
  const uint16x8_t max_coeff_thresh = vdupq_n_u16(MAX_COEFF_THRESH);
  int j;
  int distribution[MAX_COEFF_THRESH + 1] = { 0 };
  for (j = start_block; j < end_block; ++j) {
    int16_t out[16];
    FTransform_NEON(ref + VP8DspScan[j], pred + VP8DspScan[j], out);
    {
      int k;
      const int16x8_t a0 = vld1q_s16(out + 0);
      const int16x8_t b0 = vld1q_s16(out + 8);
      const uint16x8_t a1 = vreinterpretq_u16_s16(vabsq_s16(a0));
      const uint16x8_t b1 = vreinterpretq_u16_s16(vabsq_s16(b0));
      const uint16x8_t a2 = vshrq_n_u16(a1, 3);
      const uint16x8_t b2 = vshrq_n_u16(b1, 3);
      const uint16x8_t a3 = vminq_u16(a2, max_coeff_thresh);
      const uint16x8_t b3 = vminq_u16(b2, max_coeff_thresh);
      vst1q_s16(out + 0, vreinterpretq_s16_u16(a3));
      vst1q_s16(out + 8, vreinterpretq_s16_u16(b3));
      // Convert coefficients to bin.
      for (k = 0; k < 16; ++k) {
        ++distribution[out[k]];
      }
    }
  }
  VP8SetHistogramData(distribution, histo);
}

//------------------------------------------------------------------------------

static WEBP_INLINE void AccumulateSSE16_NEON(const uint8_t* const a,
                                             const uint8_t* const b,
                                             uint32x4_t* const sum) {
  const uint8x16_t a0 = vld1q_u8(a);
  const uint8x16_t b0 = vld1q_u8(b);
  const uint8x16_t abs_diff = vabdq_u8(a0, b0);
  const uint16x8_t prod1 = vmull_u8(vget_low_u8(abs_diff),
                                    vget_low_u8(abs_diff));
  const uint16x8_t prod2 = vmull_u8(vget_high_u8(abs_diff),
                                    vget_high_u8(abs_diff));
  /* pair-wise adds and widen */
  const uint32x4_t sum1 = vpaddlq_u16(prod1);
  const uint32x4_t sum2 = vpaddlq_u16(prod2);
  *sum = vaddq_u32(*sum, vaddq_u32(sum1, sum2));
}

// Horizontal sum of all four uint32_t values in 'sum'.
static int SumToInt_NEON(uint32x4_t sum) {
  const uint64x2_t sum2 = vpaddlq_u32(sum);
  const uint64_t sum3 = vgetq_lane_u64(sum2, 0) + vgetq_lane_u64(sum2, 1);
  return (int)sum3;
}

static int SSE16x16_NEON(const uint8_t* a, const uint8_t* b) {
  uint32x4_t sum = vdupq_n_u32(0);
  int y;
  for (y = 0; y < 16; ++y) {
    AccumulateSSE16_NEON(a + y * BPS, b + y * BPS, &sum);
  }
  return SumToInt_NEON(sum);
}

static int SSE16x8_NEON(const uint8_t* a, const uint8_t* b) {
  uint32x4_t sum = vdupq_n_u32(0);
  int y;
  for (y = 0; y < 8; ++y) {
    AccumulateSSE16_NEON(a + y * BPS, b + y * BPS, &sum);
  }
  return SumToInt_NEON(sum);
}

static int SSE8x8_NEON(const uint8_t* a, const uint8_t* b) {
  uint32x4_t sum = vdupq_n_u32(0);
  int y;
  for (y = 0; y < 8; ++y) {
    const uint8x8_t a0 = vld1_u8(a + y * BPS);
    const uint8x8_t b0 = vld1_u8(b + y * BPS);
    const uint8x8_t abs_diff = vabd_u8(a0, b0);
    const uint16x8_t prod = vmull_u8(abs_diff, abs_diff);
    sum = vpadalq_u16(sum, prod);
  }
  return SumToInt_NEON(sum);
}

static int SSE4x4_NEON(const uint8_t* a, const uint8_t* b) {
  const uint8x16_t a0 = Load4x4_NEON(a);
  const uint8x16_t b0 = Load4x4_NEON(b);
  const uint8x16_t abs_diff = vabdq_u8(a0, b0);
  const uint16x8_t prod1 = vmull_u8(vget_low_u8(abs_diff),
                                    vget_low_u8(abs_diff));
  const uint16x8_t prod2 = vmull_u8(vget_high_u8(abs_diff),
                                    vget_high_u8(abs_diff));
  /* pair-wise adds and widen */
  const uint32x4_t sum1 = vpaddlq_u16(prod1);
  const uint32x4_t sum2 = vpaddlq_u16(prod2);
  return SumToInt_NEON(vaddq_u32(sum1, sum2));
}

//------------------------------------------------------------------------------

// Compilation with gcc-4.6.x is problematic for now.
#if !defined(WORK_AROUND_GCC)

static int16x8_t Quantize_NEON(int16_t* const in,
                               const VP8Matrix* const mtx, int offset) {
  const uint16x8_t sharp = vld1q_u16(&mtx->sharpen_[offset]);
  const uint16x8_t q = vld1q_u16(&mtx->q_[offset]);
  const uint16x8_t iq = vld1q_u16(&mtx->iq_[offset]);
  const uint32x4_t bias0 = vld1q_u32(&mtx->bias_[offset + 0]);
  const uint32x4_t bias1 = vld1q_u32(&mtx->bias_[offset + 4]);

  const int16x8_t a = vld1q_s16(in + offset);                // in
  const uint16x8_t b = vreinterpretq_u16_s16(vabsq_s16(a));  // coeff = abs(in)
  const int16x8_t sign = vshrq_n_s16(a, 15);                 // sign
  const uint16x8_t c = vaddq_u16(b, sharp);                  // + sharpen
  const uint32x4_t m0 = vmull_u16(vget_low_u16(c), vget_low_u16(iq));
  const uint32x4_t m1 = vmull_u16(vget_high_u16(c), vget_high_u16(iq));
  const uint32x4_t m2 = vhaddq_u32(m0, bias0);
  const uint32x4_t m3 = vhaddq_u32(m1, bias1);     // (coeff * iQ + bias) >> 1
  const uint16x8_t c0 = vcombine_u16(vshrn_n_u32(m2, 16),
                                     vshrn_n_u32(m3, 16));   // QFIX=17 = 16+1
  const uint16x8_t c1 = vminq_u16(c0, vdupq_n_u16(MAX_LEVEL));
  const int16x8_t c2 = veorq_s16(vreinterpretq_s16_u16(c1), sign);
  const int16x8_t c3 = vsubq_s16(c2, sign);                  // restore sign
  const int16x8_t c4 = vmulq_s16(c3, vreinterpretq_s16_u16(q));
  vst1q_s16(in + offset, c4);
  assert(QFIX == 17);  // this function can't work as is if QFIX != 16+1
  return c3;
}

static const uint8_t kShuffles[4][8] = {
  { 0,   1,  2,  3,  8,  9, 16, 17 },
  { 10, 11,  4,  5,  6,  7, 12, 13 },
  { 18, 19, 24, 25, 26, 27, 20, 21 },
  { 14, 15, 22, 23, 28, 29, 30, 31 }
};

static int QuantizeBlock_NEON(int16_t in[16], int16_t out[16],
                              const VP8Matrix* const mtx) {
  const int16x8_t out0 = Quantize_NEON(in, mtx, 0);
  const int16x8_t out1 = Quantize_NEON(in, mtx, 8);
  uint8x8x4_t shuffles;
  // vtbl?_u8 are marked unavailable for iOS arm64 with Xcode < 6.3, use
  // non-standard versions there.
#if defined(__APPLE__) && defined(__aarch64__) && \
    defined(__apple_build_version__) && (__apple_build_version__< 6020037)
  uint8x16x2_t all_out;
  INIT_VECTOR2(all_out, vreinterpretq_u8_s16(out0), vreinterpretq_u8_s16(out1));
  INIT_VECTOR4(shuffles,
               vtbl2q_u8(all_out, vld1_u8(kShuffles[0])),
               vtbl2q_u8(all_out, vld1_u8(kShuffles[1])),
               vtbl2q_u8(all_out, vld1_u8(kShuffles[2])),
               vtbl2q_u8(all_out, vld1_u8(kShuffles[3])));
#else
  uint8x8x4_t all_out;
  INIT_VECTOR4(all_out,
               vreinterpret_u8_s16(vget_low_s16(out0)),
               vreinterpret_u8_s16(vget_high_s16(out0)),
               vreinterpret_u8_s16(vget_low_s16(out1)),
               vreinterpret_u8_s16(vget_high_s16(out1)));
  INIT_VECTOR4(shuffles,
               vtbl4_u8(all_out, vld1_u8(kShuffles[0])),
               vtbl4_u8(all_out, vld1_u8(kShuffles[1])),
               vtbl4_u8(all_out, vld1_u8(kShuffles[2])),
               vtbl4_u8(all_out, vld1_u8(kShuffles[3])));
#endif
  // Zigzag reordering
  vst1_u8((uint8_t*)(out +  0), shuffles.val[0]);
  vst1_u8((uint8_t*)(out +  4), shuffles.val[1]);
  vst1_u8((uint8_t*)(out +  8), shuffles.val[2]);
  vst1_u8((uint8_t*)(out + 12), shuffles.val[3]);
  // test zeros
  if (*(uint64_t*)(out +  0) != 0) return 1;
  if (*(uint64_t*)(out +  4) != 0) return 1;
  if (*(uint64_t*)(out +  8) != 0) return 1;
  if (*(uint64_t*)(out + 12) != 0) return 1;
  return 0;
}

static int Quantize2Blocks_NEON(int16_t in[32], int16_t out[32],
                                const VP8Matrix* const mtx) {
  int nz;
  nz  = QuantizeBlock_NEON(in + 0 * 16, out + 0 * 16, mtx) << 0;
  nz |= QuantizeBlock_NEON(in + 1 * 16, out + 1 * 16, mtx) << 1;
  return nz;
}

#endif   // !WORK_AROUND_GCC

//------------------------------------------------------------------------------
// Entry point

extern void VP8EncDspInitNEON(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8EncDspInitNEON(void) {
  VP8ITransform = ITransform_NEON;
  VP8FTransform = FTransform_NEON;

  VP8FTransformWHT = FTransformWHT_NEON;

  VP8TDisto4x4 = Disto4x4_NEON;
  VP8TDisto16x16 = Disto16x16_NEON;
  VP8CollectHistogram = CollectHistogram_NEON;

  VP8SSE16x16 = SSE16x16_NEON;
  VP8SSE16x8 = SSE16x8_NEON;
  VP8SSE8x8 = SSE8x8_NEON;
  VP8SSE4x4 = SSE4x4_NEON;

#if !defined(WORK_AROUND_GCC)
  VP8EncQuantizeBlock = QuantizeBlock_NEON;
  VP8EncQuantize2Blocks = Quantize2Blocks_NEON;
#endif
}

#else  // !WEBP_USE_NEON

WEBP_DSP_INIT_STUB(VP8EncDspInitNEON)

#endif  // WEBP_USE_NEON
