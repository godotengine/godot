/*
 *  Copyright 2024 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_ROW_SVE_H_
#define INCLUDE_LIBYUV_ROW_SVE_H_

#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#if !defined(LIBYUV_DISABLE_SVE) && defined(__aarch64__)

#if !defined(LIBYUV_DISABLE_SME) && defined(CLANG_HAS_SME) && \
    defined(__aarch64__)
#define STREAMING_COMPATIBLE __arm_streaming_compatible
#else  // defined(LIBYUV_DISABLE_SME) || !defined(CLANG_HAS_SME) ||
       // !defined(__aarch64__)
#define STREAMING_COMPATIBLE
#endif  // !defined(LIBYUV_DISABLE_SME) && defined(CLANG_HAS_SME) &&
        // defined(__aarch64__)

#define YUVTORGB_SVE_SETUP                          \
  "ld1rb  {z28.b}, p0/z, [%[kUVCoeff], #0]      \n" \
  "ld1rb  {z29.b}, p0/z, [%[kUVCoeff], #1]      \n" \
  "ld1rb  {z30.b}, p0/z, [%[kUVCoeff], #2]      \n" \
  "ld1rb  {z31.b}, p0/z, [%[kUVCoeff], #3]      \n" \
  "ld1rh  {z24.h}, p0/z, [%[kRGBCoeffBias], #0] \n" \
  "ld1rh  {z25.h}, p0/z, [%[kRGBCoeffBias], #2] \n" \
  "ld1rh  {z26.h}, p0/z, [%[kRGBCoeffBias], #4] \n" \
  "ld1rh  {z27.h}, p0/z, [%[kRGBCoeffBias], #6] \n"

#define READYUV444_SVE                           \
  "ld1b       {z0.h}, p1/z, [%[src_y]]       \n" \
  "ld1b       {z1.h}, p1/z, [%[src_u]]       \n" \
  "ld1b       {z2.h}, p1/z, [%[src_v]]       \n" \
  "add        %[src_y], %[src_y], %[vl]      \n" \
  "add        %[src_u], %[src_u], %[vl]      \n" \
  "add        %[src_v], %[src_v], %[vl]      \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "prfm       pldl1keep, [%[src_u], 448]     \n" \
  "trn1       z0.b, z0.b, z0.b               \n" \
  "prfm       pldl1keep, [%[src_v], 448]     \n"

// Read twice as much data from YUV, putting the even elements from the Y data
// in z0.h and odd elements in z1.h.
#define READYUV444_SVE_2X                        \
  "ld1b       {z0.b}, p1/z, [%[src_y]]       \n" \
  "ld1b       {z2.b}, p1/z, [%[src_u]]       \n" \
  "ld1b       {z3.b}, p1/z, [%[src_v]]       \n" \
  "incb       %[src_y]                       \n" \
  "incb       %[src_u]                       \n" \
  "incb       %[src_v]                       \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "prfm       pldl1keep, [%[src_u], 128]     \n" \
  "prfm       pldl1keep, [%[src_v], 128]     \n" \
  "trn2       z1.b, z0.b, z0.b               \n" \
  "trn1       z0.b, z0.b, z0.b               \n"

#define READYUV400_SVE                           \
  "ld1b       {z0.h}, p1/z, [%[src_y]]       \n" \
  "inch       %[src_y]                       \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "trn1       z0.b, z0.b, z0.b               \n"

#define READYUV422_SVE                           \
  "ld1b       {z0.h}, p1/z, [%[src_y]]       \n" \
  "ld1b       {z1.s}, p1/z, [%[src_u]]       \n" \
  "ld1b       {z2.s}, p1/z, [%[src_v]]       \n" \
  "inch       %[src_y]                       \n" \
  "incw       %[src_u]                       \n" \
  "incw       %[src_v]                       \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "prfm       pldl1keep, [%[src_u], 128]     \n" \
  "prfm       pldl1keep, [%[src_v], 128]     \n" \
  "trn1       z0.b, z0.b, z0.b               \n" \
  "trn1       z1.h, z1.h, z1.h               \n" \
  "trn1       z2.h, z2.h, z2.h               \n"

// Read twice as much data from YUV, putting the even elements from the Y data
// in z0.h and odd elements in z1.h. U/V data is not duplicated, stored in
// z2.h/z3.h.
#define READYUV422_SVE_2X                        \
  "ld1b       {z0.b}, p1/z, [%[src_y]]       \n" \
  "ld1b       {z2.h}, p1/z, [%[src_u]]       \n" \
  "ld1b       {z3.h}, p1/z, [%[src_v]]       \n" \
  "incb       %[src_y]                       \n" \
  "inch       %[src_u]                       \n" \
  "inch       %[src_v]                       \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "prfm       pldl1keep, [%[src_u], 128]     \n" \
  "prfm       pldl1keep, [%[src_v], 128]     \n" \
  "trn2       z1.b, z0.b, z0.b               \n" \
  "trn1       z0.b, z0.b, z0.b               \n"

#define READI210_SVE                             \
  "ld1h       {z3.h}, p1/z, [%[src_y]]       \n" \
  "ld1h       {z1.s}, p1/z, [%[src_u]]       \n" \
  "ld1h       {z2.s}, p1/z, [%[src_v]]       \n" \
  "incb       %[src_y]                       \n" \
  "inch       %[src_u]                       \n" \
  "inch       %[src_v]                       \n" \
  "lsl        z0.h, z3.h, #6                 \n" \
  "trn1       z1.h, z1.h, z1.h               \n" \
  "trn1       z2.h, z2.h, z2.h               \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "prfm       pldl1keep, [%[src_u], 128]     \n" \
  "prfm       pldl1keep, [%[src_v], 128]     \n" \
  "usra       z0.h, z3.h, #4                 \n" \
  "uqshrnb    z1.b, z1.h, #2                 \n" \
  "uqshrnb    z2.b, z2.h, #2                 \n"

#define READI210_SVE_2X                                      \
  "ld1h       {z4.h}, p2/z, [%[src_y]]                   \n" \
  "ld1h       {z5.h}, p3/z, [%[src_y], #1, mul vl]       \n" \
  "ld1h       {z2.h}, p1/z, [%[src_u]]                   \n" \
  "ld1h       {z3.h}, p1/z, [%[src_v]]                   \n" \
  "incb       %[src_y], all, mul #2                      \n" \
  "uzp1       z6.h, z4.h, z5.h                           \n" \
  "uzp2       z5.h, z4.h, z5.h                           \n" \
  "incb       %[src_u]                                   \n" \
  "incb       %[src_v]                                   \n" \
  "lsl        z0.h, z6.h, #6                             \n" \
  "lsl        z1.h, z5.h, #6                             \n" \
  "prfm       pldl1keep, [%[src_y], 448]                 \n" \
  "prfm       pldl1keep, [%[src_u], 128]                 \n" \
  "prfm       pldl1keep, [%[src_v], 128]                 \n" \
  "usra       z0.h, z6.h, #4                             \n" \
  "usra       z1.h, z5.h, #4                             \n" \
  "uqshrnb    z2.b, z2.h, #2                             \n" \
  "uqshrnb    z3.b, z3.h, #2                             \n"

#define READP210_SVE                             \
  "ld1h       {z0.h}, p1/z, [%[src_y]]       \n" \
  "ld1h       {z1.h}, p2/z, [%[src_uv]]      \n" \
  "incb       %[src_y]                       \n" \
  "incb       %[src_uv]                      \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "prfm       pldl1keep, [%[src_uv], 256]    \n" \
  "tbl        z1.b, {z1.b}, z22.b            \n"

#define READI410_SVE                             \
  "ld1h       {z3.h}, p1/z, [%[src_y]]       \n" \
  "lsl        z0.h, z3.h, #6                 \n" \
  "usra       z0.h, z3.h, #4                 \n" \
  "ld1h       {z1.h}, p1/z, [%[src_u]]       \n" \
  "ld1h       {z2.h}, p1/z, [%[src_v]]       \n" \
  "incb       %[src_y]                       \n" \
  "incb       %[src_u]                       \n" \
  "incb       %[src_v]                       \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "prfm       pldl1keep, [%[src_u], 128]     \n" \
  "prfm       pldl1keep, [%[src_v], 128]     \n" \
  "uqshrnb    z1.b, z1.h, #2                 \n" \
  "uqshrnb    z2.b, z2.h, #2                 \n"

// We need different predicates for the UV components since we are reading
// 32-bit (pairs of UV) elements rather than 16-bit Y elements.
#define READP410_SVE                                    \
  "ld1h       {z0.h}, p1/z, [%[src_y]]              \n" \
  "ld1w       {z1.s}, p2/z, [%[src_uv]]             \n" \
  "ld1w       {z2.s}, p3/z, [%[src_uv], #1, mul vl] \n" \
  "incb       %[src_y]                              \n" \
  "incb       %[src_uv], all, mul #2                \n" \
  "prfm       pldl1keep, [%[src_y], 448]            \n" \
  "prfm       pldl1keep, [%[src_uv], 256]           \n" \
  "uzp2       z1.b, z1.b, z2.b                      \n"

#define READI212_SVE                             \
  "ld1h       {z3.h}, p1/z, [%[src_y]]       \n" \
  "ld1h       {z1.s}, p1/z, [%[src_u]]       \n" \
  "ld1h       {z2.s}, p1/z, [%[src_v]]       \n" \
  "incb       %[src_y]                       \n" \
  "inch       %[src_u]                       \n" \
  "inch       %[src_v]                       \n" \
  "lsl        z0.h, z3.h, #4                 \n" \
  "trn1       z1.h, z1.h, z1.h               \n" \
  "trn1       z2.h, z2.h, z2.h               \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "prfm       pldl1keep, [%[src_u], 128]     \n" \
  "prfm       pldl1keep, [%[src_v], 128]     \n" \
  "usra       z0.h, z3.h, #8                 \n" \
  "uqshrnb    z1.b, z1.h, #4                 \n" \
  "uqshrnb    z2.b, z2.h, #4                 \n"

#define I400TORGB_SVE                                    \
  "umulh      z18.h, z24.h, z0.h             \n" /* Y */ \
  "movprfx    z16, z18                       \n"         \
  "usqadd     z16.h, p0/m, z16.h, z4.h       \n" /* B */ \
  "movprfx    z17, z18                       \n"         \
  "usqadd     z17.h, p0/m, z17.h, z6.h       \n" /* G */ \
  "usqadd     z18.h, p0/m, z18.h, z5.h       \n" /* R */

// We need a different predicate for the UV component to handle the tail.
// If there is a single element remaining then we want to load one Y element
// but two UV elements.
#define READNV_SVE_2X                                               \
  "ld1b       {z0.b}, p1/z, [%[src_y]]       \n" /* Y0Y0 */         \
  "ld1b       {z2.b}, p2/z, [%[src_uv]]      \n" /* U0V0 or V0U0 */ \
  "incb       %[src_y]                       \n"                    \
  "incb       %[src_uv]                      \n"                    \
  "prfm       pldl1keep, [%[src_y], 448]     \n"                    \
  "prfm       pldl1keep, [%[src_uv], 256]    \n"                    \
  "trn2       z1.b, z0.b, z0.b               \n" /* YYYY */         \
  "trn1       z0.b, z0.b, z0.b               \n" /* YYYY */

// Like NVTORGB_SVE but U/V components are stored in widened .h elements of
// z1/z2 rather than even/odd .b lanes of z1.
#define I4XXTORGB_SVE                                     \
  "umulh      z0.h, z24.h, z0.h              \n" /* Y */  \
  "umullb     z6.h, z30.b, z1.b              \n"          \
  "umullb     z4.h, z28.b, z1.b              \n" /* DB */ \
  "umullb     z5.h, z29.b, z2.b              \n" /* DR */ \
  "umlalb     z6.h, z31.b, z2.b              \n" /* DG */ \
  "add        z17.h, z0.h, z26.h             \n" /* G */  \
  "add        z16.h, z0.h, z4.h              \n" /* B */  \
  "add        z18.h, z0.h, z5.h              \n" /* R */  \
  "uqsub      z17.h, z17.h, z6.h             \n" /* G */  \
  "uqsub      z16.h, z16.h, z25.h            \n" /* B */  \
  "uqsub      z18.h, z18.h, z27.h            \n" /* R */

#define I444TORGB_SVE_2X                                  \
  "umulh      z0.h, z24.h, z0.h              \n" /* Y0 */ \
  "umulh      z1.h, z24.h, z1.h              \n" /* Y1 */ \
  "umullb     z6.h, z30.b, z2.b              \n"          \
  "umullt     z7.h, z30.b, z2.b              \n"          \
  "umullb     z4.h, z28.b, z2.b              \n" /* DB */ \
  "umullt     z2.h, z28.b, z2.b              \n" /* DB */ \
  "umlalb     z6.h, z31.b, z3.b              \n" /* DG */ \
  "umlalt     z7.h, z31.b, z3.b              \n" /* DG */ \
  "umullb     z5.h, z29.b, z3.b              \n" /* DR */ \
  "umullt     z3.h, z29.b, z3.b              \n" /* DR */ \
  "add        z17.h, z0.h, z26.h             \n" /* G */  \
  "add        z21.h, z1.h, z26.h             \n" /* G */  \
  "add        z16.h, z0.h, z4.h              \n" /* B */  \
  "add        z20.h, z1.h, z2.h              \n" /* B */  \
  "add        z18.h, z0.h, z5.h              \n" /* R */  \
  "add        z22.h, z1.h, z3.h              \n" /* R */  \
  "uqsub      z17.h, z17.h, z6.h             \n" /* G */  \
  "uqsub      z21.h, z21.h, z7.h             \n" /* G */  \
  "uqsub      z16.h, z16.h, z25.h            \n" /* B */  \
  "uqsub      z20.h, z20.h, z25.h            \n" /* B */  \
  "uqsub      z18.h, z18.h, z27.h            \n" /* R */  \
  "uqsub      z22.h, z22.h, z27.h            \n" /* R */

// Like I4XXTORGB_SVE but U/V components are stored in even/odd .b lanes of z1
// rather than widened .h elements of z1/z2.
#define NVTORGB_SVE                                       \
  "umulh      z0.h, z24.h, z0.h              \n" /* Y */  \
  "umullb     z6.h, z30.b, z1.b              \n"          \
  "umullb     z4.h, z28.b, z1.b              \n" /* DB */ \
  "umullt     z5.h, z29.b, z1.b              \n" /* DR */ \
  "umlalt     z6.h, z31.b, z1.b              \n" /* DG */ \
  "add        z17.h, z0.h, z26.h             \n" /* G */  \
  "add        z16.h, z0.h, z4.h              \n" /* B */  \
  "add        z18.h, z0.h, z5.h              \n" /* R */  \
  "uqsub      z17.h, z17.h, z6.h             \n" /* G */  \
  "uqsub      z16.h, z16.h, z25.h            \n" /* B */  \
  "uqsub      z18.h, z18.h, z27.h            \n" /* R */

// The U/V component multiplies do not need to be duplicated in I422, we just
// need to combine them with Y0/Y1 correctly.
#define I422TORGB_SVE_2X                                  \
  "umulh      z0.h, z24.h, z0.h              \n" /* Y0 */ \
  "umulh      z1.h, z24.h, z1.h              \n" /* Y1 */ \
  "umullb     z6.h, z30.b, z2.b              \n"          \
  "umullb     z4.h, z28.b, z2.b              \n" /* DB */ \
  "umullb     z5.h, z29.b, z3.b              \n" /* DR */ \
  "umlalb     z6.h, z31.b, z3.b              \n" /* DG */ \
                                                          \
  "add        z17.h, z0.h, z26.h             \n" /* G0 */ \
  "add        z21.h, z1.h, z26.h             \n" /* G1 */ \
  "add        z16.h, z0.h, z4.h              \n" /* B0 */ \
  "add        z20.h, z1.h, z4.h              \n" /* B1 */ \
  "add        z18.h, z0.h, z5.h              \n" /* R0 */ \
  "add        z22.h, z1.h, z5.h              \n" /* R1 */ \
  "uqsub      z17.h, z17.h, z6.h             \n" /* G0 */ \
  "uqsub      z21.h, z21.h, z6.h             \n" /* G1 */ \
  "uqsub      z16.h, z16.h, z25.h            \n" /* B0 */ \
  "uqsub      z20.h, z20.h, z25.h            \n" /* B1 */ \
  "uqsub      z18.h, z18.h, z27.h            \n" /* R0 */ \
  "uqsub      z22.h, z22.h, z27.h            \n" /* R1 */

// clang-format off
#define NVTORGB_SVE_2X(bt_u, bt_v)                        \
  "umulh      z0.h, z24.h, z0.h              \n" /* Y0 */ \
  "umulh      z1.h, z24.h, z1.h              \n" /* Y1 */ \
  "umull" #bt_u " z6.h, z30.b, z2.b          \n"          \
  "umull" #bt_u " z4.h, z28.b, z2.b          \n" /* DB */ \
  "umull" #bt_v " z5.h, z29.b, z2.b          \n" /* DR */ \
  "umlal" #bt_v " z6.h, z31.b, z2.b          \n" /* DG */ \
                                                          \
  "add        z17.h, z0.h, z26.h             \n" /* G0 */ \
  "add        z21.h, z1.h, z26.h             \n" /* G1 */ \
  "add        z16.h, z0.h, z4.h              \n" /* B0 */ \
  "add        z20.h, z1.h, z4.h              \n" /* B1 */ \
  "add        z18.h, z0.h, z5.h              \n" /* R0 */ \
  "add        z22.h, z1.h, z5.h              \n" /* R1 */ \
  "uqsub      z17.h, z17.h, z6.h             \n" /* G0 */ \
  "uqsub      z21.h, z21.h, z6.h             \n" /* G1 */ \
  "uqsub      z16.h, z16.h, z25.h            \n" /* B0 */ \
  "uqsub      z20.h, z20.h, z25.h            \n" /* B1 */ \
  "uqsub      z18.h, z18.h, z27.h            \n" /* R0 */ \
  "uqsub      z22.h, z22.h, z27.h            \n" /* R1 */
// clang-format on

#define RGBTOARGB8_SVE_TOP_2X                        \
  /* Inputs: B: z16.h,  G: z17.h,  R: z18.h */       \
  "uqshl     z16.h, p0/m, z16.h, #2     \n" /* B0 */ \
  "uqshl     z17.h, p0/m, z17.h, #2     \n" /* G0 */ \
  "uqshl     z18.h, p0/m, z18.h, #2     \n" /* R0 */ \
  "uqshl     z20.h, p0/m, z20.h, #2     \n" /* B1 */ \
  "uqshl     z21.h, p0/m, z21.h, #2     \n" /* G1 */ \
  "uqshl     z22.h, p0/m, z22.h, #2     \n" /* R1 */

// Convert from 2.14 fixed point RGB to 8 bit ARGB, interleaving as BG and RA
// pairs to allow us to use ST2 for storing rather than ST4.
#define RGBTOARGB8_SVE                                    \
  /* Inputs: B: z16.h,  G: z17.h,  R: z18.h,  A: z19.b */ \
  "uqshrnb     z16.b, z16.h, #6     \n" /* B0 */          \
  "uqshrnb     z18.b, z18.h, #6     \n" /* R0 */          \
  "uqshrnt     z16.b, z17.h, #6     \n" /* BG */          \
  "trn1        z17.b, z18.b, z19.b  \n" /* RA */

// Convert from 2.14 fixed point RGBA to 8 bit ARGB, interleaving as BG and RA
// pairs to allow us to use ST2 for storing rather than ST4.
#define RGBATOARGB8_SVE                                   \
  /* Inputs: B: z16.h,  G: z17.h,  R: z18.h,  A: z19.h */ \
  "uqshrnb     z16.b, z16.h, #6     \n" /* B0 */          \
  "uqshrnt     z16.b, z17.h, #6     \n" /* BG */          \
  "uqshrnb     z17.b, z18.h, #6     \n" /* R0 */          \
  "uqshrnt     z17.b, z19.h, #2     \n" /* RA */

// Convert from 2.14 fixed point RGB to 8 bit RGBA, interleaving as AB and GR
// pairs to allow us to use ST2 for storing rather than ST4.
#define RGBTORGBA8_SVE                                    \
  /* Inputs: B: z16.h,  G: z17.h,  R: z18.h,  A: z19.b */ \
  "uqshrnt     z19.b, z16.h, #6     \n" /* AB */          \
  "uqshrnb     z20.b, z17.h, #6     \n" /* G0 */          \
  "uqshrnt     z20.b, z18.h, #6     \n" /* GR */

#define RGBTOARGB8_SVE_2X                                 \
  /* Inputs: B: z16.h,  G: z17.h,  R: z18.h,  A: z19.b */ \
  "uqshrnb     z16.b, z16.h, #6     \n" /* B0 */          \
  "uqshrnb     z17.b, z17.h, #6     \n" /* G0 */          \
  "uqshrnb     z18.b, z18.h, #6     \n" /* R0 */          \
  "uqshrnt     z16.b, z20.h, #6     \n" /* B1 */          \
  "uqshrnt     z17.b, z21.h, #6     \n" /* G1 */          \
  "uqshrnt     z18.b, z22.h, #6     \n" /* R1 */

// Store AR30 elements. Inputs are 2.14 fixed point RGB. We expect z23 to be
// populated with 0x3ff0 (0x3fff would also work) to saturate the R input
// rather than needing a pair of shifts to saturate and then insert into the
// correct position in the lane.
#define STOREAR30_SVE                                                    \
  /* Inputs: B: z16.h,  G: z17.h,  R: z18.h */                           \
  "uqshl    z16.h, p0/m, z16.h, #2            \n" /* bbbbbbbbbbxxxxxx */ \
  "uqshl    z17.h, p0/m, z17.h, #2            \n" /* ggggggggggxxxxxx */ \
  "umin     z18.h, p0/m, z18.h, z23.h         \n" /* 00rrrrrrrrrrxxxx */ \
  "orr      z18.h, z18.h, #0xc000             \n" /* 11rrrrrrrrrrxxxx */ \
  "sri      z18.h, z17.h, #12                 \n" /* 11rrrrrrrrrrgggg */ \
  "lsl      z17.h, z17.h, #4                  \n" /* ggggggxxxxxx0000 */ \
  "sri      z17.h, z16.h, #6                  \n" /* ggggggbbbbbbbbbb */ \
  "st2h     {z17.h, z18.h}, p1, [%[dst_ar30]] \n"                        \
  "incb     %[dst_ar30], all, mul #2          \n"

#define STOREAR30_SVE_2X                                                 \
  /* Inputs: B: z16.h,  G: z17.h,  R: z18.h */                           \
  /*         B: z20.h,  G: z21.h,  R: z22.h */                           \
  "uqshl    z16.h, p0/m, z16.h, #2            \n" /* bbbbbbbbbbxxxxxx */ \
  "uqshl    z20.h, p0/m, z20.h, #2            \n" /* bbbbbbbbbbxxxxxx */ \
  "uqshl    z17.h, p0/m, z17.h, #2            \n" /* ggggggggggxxxxxx */ \
  "uqshl    z21.h, p0/m, z21.h, #2            \n" /* ggggggggggxxxxxx */ \
  "umin     z18.h, p0/m, z18.h, z23.h         \n" /* 00rrrrrrrrrrxxxx */ \
  "umin     z22.h, p0/m, z22.h, z23.h         \n" /* 00rrrrrrrrrrxxxx */ \
  "orr      z18.h, z18.h, #0xc000             \n" /* 11rrrrrrrrrrxxxx */ \
  "orr      z22.h, z22.h, #0xc000             \n" /* 11rrrrrrrrrrxxxx */ \
  "sri      z18.h, z17.h, #12                 \n" /* 11rrrrrrrrrrgggg */ \
  "sri      z22.h, z21.h, #12                 \n" /* 11rrrrrrrrrrgggg */ \
  "lsl      z17.h, z17.h, #4                  \n" /* ggggggxxxxxx0000 */ \
  "lsl      z19.h, z21.h, #4                  \n" /* ggggggxxxxxx0000 */ \
  "sri      z17.h, z16.h, #6                  \n" /* ggggggbbbbbbbbbb */ \
  "sri      z19.h, z20.h, #6                  \n" /* ggggggbbbbbbbbbb */ \
  "zip2     z16.h, z17.h, z19.h               \n"                        \
  "zip1     z21.h, z17.h, z19.h               \n"                        \
  "zip2     z17.h, z18.h, z22.h               \n"                        \
  "zip1     z22.h, z18.h, z22.h               \n"                        \
  "st2h     {z21.h, z22.h}, p2, [%[dst_ar30]] \n"                        \
  "st2h     {z16.h, z17.h}, p3, [%[dst_ar30], #2, mul vl] \n"            \
  "incb     %[dst_ar30], all, mul #4          \n"

#define YUVTORGB_SVE_REGS                                                     \
  "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z16", "z17", "z18", "z19", \
      "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",   \
      "z30", "z31", "p0", "p1", "p2", "p3"

static inline void I444ToRGB24Row_SVE_SC(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    uint8_t* dst_rgb24,
    const struct YuvConstants* yuvconstants,
    int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm volatile(
      "cntb     %[vl]                                     \n"
      "ptrue    p0.b                                      \n"  //
      YUVTORGB_SVE_SETUP
      "subs     %w[width], %w[width], %w[vl]              \n"
      "b.lt     2f                                        \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.b                                      \n"
      "1:                                                 \n"  //
      READYUV444_SVE_2X I444TORGB_SVE_2X RGBTOARGB8_SVE_2X
      "subs     %w[width], %w[width], %w[vl]              \n"
      "st3b     {z16.b, z17.b, z18.b}, p1, [%[dst_rgb24]] \n"
      "incb     %[dst_rgb24], all, mul #3                 \n"
      "b.ge     1b                                        \n"

      "2:                                                 \n"
      "adds     %w[width], %w[width], %w[vl]              \n"
      "b.eq     99f                                       \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "cnth     %[vl]                                     \n"
      "whilelt  p1.b, wzr, %w[width]                      \n"  //
      READYUV444_SVE_2X I444TORGB_SVE_2X RGBTOARGB8_SVE_2X
      "st3b     {z16.b, z17.b, z18.b}, p1, [%[dst_rgb24]] \n"

      "99:                                                \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_rgb24] "+r"(dst_rgb24),                       // %[dst_argb]
        [width] "+r"(width),                               // %[width]
        [vl] "=&r"(vl)                                     // %[vl]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void I400ToARGBRow_SVE_SC(const uint8_t* src_y,
                                        uint8_t* dst_argb,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm volatile(
      "cnth     %[vl]                                   \n"
      "ptrue    p0.b                                    \n"
      "dup      z19.b, #255                             \n"  // Alpha
      YUVTORGB_SVE_SETUP
      "cmp      %w[width], %w[vl]                       \n"
      "mov      z1.h, #128                              \n"  // U/V
      "umullb   z6.h, z30.b, z1.b                       \n"
      "umullb   z4.h, z28.b, z1.b                       \n"  // DB
      "umullb   z5.h, z29.b, z1.b                       \n"  // DR
      "mla      z6.h, p0/m, z31.h, z1.h                 \n"  // DG
      "sub      z4.h, z4.h, z25.h                       \n"
      "sub      z5.h, z5.h, z27.h                       \n"
      "sub      z6.h, z26.h, z6.h                       \n"
      "b.le     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "sub      %w[width], %w[width], %w[vl]            \n"
      "1:                                               \n"  //
      READYUV400_SVE I400TORGB_SVE RGBTOARGB8_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2 \n"
      "b.gt     1b                                      \n"
      "add      %w[width], %w[width], %w[vl]            \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "2:                                               \n"
      "whilelt  p1.h, wzr, %w[width]                    \n"  //
      READYUV400_SVE I400TORGB_SVE RGBTOARGB8_SVE
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width),                               // %[width]
        [vl] "=&r"(vl)                                     // %[vl]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void I422ToARGBRow_SVE_SC(const uint8_t* src_y,
                                        const uint8_t* src_u,
                                        const uint8_t* src_v,
                                        uint8_t* dst_argb,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm volatile(
      "cntb     %[vl]                                   \n"
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z19.b, #255                             \n"  // Alpha
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.b                                    \n"
      "1:                                               \n"  //
      READYUV422_SVE_2X I422TORGB_SVE_2X RGBTOARGB8_SVE_2X
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st4b     {z16.b, z17.b, z18.b, z19.b}, p1, [%[dst_argb]] \n"
      "incb     %[dst_argb], all, mul #4                \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "cnth     %[vl]                                   \n"
      "whilelt  p1.b, wzr, %w[width]                    \n"  //
      READYUV422_SVE_2X I422TORGB_SVE_2X RGBTOARGB8_SVE_2X
      "st4b     {z16.b, z17.b, z18.b, z19.b}, p1, [%[dst_argb]] \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width),                               // %[width]
        [vl] "=&r"(vl)                                     // %[vl]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void I422ToRGB24Row_SVE_SC(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm volatile(
      "cntb     %[vl]                                   \n"
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.b                                    \n"
      "1:                                               \n"  //
      READYUV422_SVE_2X I422TORGB_SVE_2X RGBTOARGB8_SVE_2X
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st3b     {z16.b, z17.b, z18.b}, p1, [%[dst_argb]] \n"
      "incb     %[dst_argb], all, mul #3                \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "cnth     %[vl]                                   \n"
      "whilelt  p1.b, wzr, %w[width]                    \n"  //
      READYUV422_SVE_2X I422TORGB_SVE_2X RGBTOARGB8_SVE_2X
      "st3b     {z16.b, z17.b, z18.b}, p1, [%[dst_argb]] \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width),                               // %[width]
        [vl] "=&r"(vl)                                     // %[vl]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

#define RGB8TORGB565_SVE_FROM_TOP_2X                        \
  "sri      z18.h, z17.h, #5     \n" /* rrrrrgggggg00000 */ \
  "sri      z22.h, z21.h, #5     \n" /* rrrrrgggggg00000 */ \
  "sri      z18.h, z16.h, #11    \n" /* rrrrrggggggbbbbb */ \
  "sri      z22.h, z20.h, #11    \n" /* rrrrrggggggbbbbb */ \
  "mov      z19.d, z22.d         \n"

static inline void I422ToRGB565Row_SVE_SC(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    uint8_t* dst_rgb565,
    const struct YuvConstants* yuvconstants,
    int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm volatile(
      "cntb     %[vl]                                   \n"
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.b                                    \n"
      "1:                                               \n"  //
      READYUV422_SVE_2X I422TORGB_SVE_2X RGBTOARGB8_SVE_TOP_2X
      "subs     %w[width], %w[width], %w[vl]            \n"  //
      RGB8TORGB565_SVE_FROM_TOP_2X
      "st2h     {z18.h, z19.h}, p1, [%[dst]] \n"
      "incb     %[dst], all, mul #2                     \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "cnth     %[vl]                                   \n"
      "whilelt  p1.b, wzr, %w[width]                    \n"  //
      READYUV422_SVE_2X I422TORGB_SVE_2X RGBTOARGB8_SVE_TOP_2X
          RGB8TORGB565_SVE_FROM_TOP_2X
      // Need to permute the data on the final iteration such that the
      // predicates (.b) line up with the 16-bit element data.
      "trn1     z20.b, z18.b, z19.b                     \n"
      "trn2     z21.b, z18.b, z19.b                     \n"
      "st2b     {z20.b, z21.b}, p1, [%[dst]]            \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst] "+r"(dst_rgb565),                            // %[dst]
        [width] "+r"(width),                               // %[width]
        [vl] "=&r"(vl)                                     // %[vl]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

#define RGB8TOARGB1555_SVE_FROM_TOP_2X                      \
  "dup      z0.h, #0x8000        \n" /* 1000000000000000 */ \
  "dup      z1.h, #0x8000        \n" /* 1000000000000000 */ \
  "sri      z0.h, z18.h, #1      \n" /* 1rrrrrxxxxxxxxxx */ \
  "sri      z1.h, z22.h, #1      \n" /* 1rrrrrxxxxxxxxxx */ \
  "sri      z0.h, z17.h, #6      \n" /* 1rrrrrgggggxxxxx */ \
  "sri      z1.h, z21.h, #6      \n" /* 1rrrrrgggggxxxxx */ \
  "sri      z0.h, z16.h, #11     \n" /* 1rrrrrgggggbbbbb */ \
  "sri      z1.h, z20.h, #11     \n" /* 1rrrrrgggggbbbbb */

static inline void I422ToARGB1555Row_SVE_SC(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    uint8_t* dst_argb1555,
    const struct YuvConstants* yuvconstants,
    int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm volatile(
      "cntb     %[vl]                                   \n"
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.b                                    \n"
      "1:                                               \n"  //
      READYUV422_SVE_2X I422TORGB_SVE_2X RGBTOARGB8_SVE_TOP_2X
      "subs     %w[width], %w[width], %w[vl]            \n"  //
      RGB8TOARGB1555_SVE_FROM_TOP_2X
      "st2h     {z0.h, z1.h}, p1, [%[dst]] \n"
      "incb     %[dst], all, mul #2                     \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "cnth     %[vl]                                   \n"
      "whilelt  p1.b, wzr, %w[width]                    \n"  //
      READYUV422_SVE_2X I422TORGB_SVE_2X RGBTOARGB8_SVE_TOP_2X
          RGB8TOARGB1555_SVE_FROM_TOP_2X
      "st2h     {z0.h, z1.h}, p1, [%[dst]] \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst] "+r"(dst_argb1555),                          // %[dst]
        [width] "+r"(width),                               // %[width]
        [vl] "=&r"(vl)                                     // %[vl]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

#define RGB8TOARGB4444_SVE_FROM_TOP_2X                      \
  "dup      z0.h, #0xf000        \n" /* 1111000000000000 */ \
  "dup      z1.h, #0xf000        \n" /* 1111000000000000 */ \
  "sri      z0.h, z18.h, #4      \n" /* 1111rrrrxxxxxxxx */ \
  "sri      z1.h, z22.h, #4      \n" /* 1111rrrrxxxxxxxx */ \
  "sri      z0.h, z17.h, #8      \n" /* 1111rrrrggggxxxx */ \
  "sri      z1.h, z21.h, #8      \n" /* 1111rrrrggggxxxx */ \
  "sri      z0.h, z16.h, #12     \n" /* 1111rrrrggggbbbb */ \
  "sri      z1.h, z20.h, #12     \n" /* 1111rrrrggggbbbb */

static inline void I422ToARGB4444Row_SVE_SC(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    uint8_t* dst_argb4444,
    const struct YuvConstants* yuvconstants,
    int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm volatile(
      "cntb     %[vl]                                   \n"
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.b                                    \n"
      "1:                                               \n"  //
      READYUV422_SVE_2X I422TORGB_SVE_2X RGBTOARGB8_SVE_TOP_2X
      "subs     %w[width], %w[width], %w[vl]            \n"  //
      RGB8TOARGB4444_SVE_FROM_TOP_2X
      "st2h     {z0.h, z1.h}, p1, [%[dst]] \n"
      "incb     %[dst], all, mul #2                     \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "cnth     %[vl]                                   \n"
      "whilelt  p1.b, wzr, %w[width]                    \n"  //
      READYUV422_SVE_2X I422TORGB_SVE_2X RGBTOARGB8_SVE_TOP_2X
          RGB8TOARGB4444_SVE_FROM_TOP_2X
      "st2h     {z0.h, z1.h}, p1, [%[dst]] \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst] "+r"(dst_argb4444),                          // %[dst]
        [width] "+r"(width),                               // %[width]
        [vl] "=&r"(vl)                                     // %[vl]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void I422ToRGBARow_SVE_SC(const uint8_t* src_y,
                                        const uint8_t* src_u,
                                        const uint8_t* src_v,
                                        uint8_t* dst_argb,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm volatile(
      "cnth     %[vl]                                   \n"
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z19.b, #255                             \n"  // Alpha
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.le     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "1:                                               \n"  //
      READYUV422_SVE I4XXTORGB_SVE RGBTORGBA8_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st2h     {z19.h, z20.h}, p1, [%[dst_argb]]       \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2 \n"
      "b.gt     1b                                      \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "2:                                               \n"
      "adds    %w[width], %w[width], %w[vl]             \n"
      "b.eq    99f                                      \n"

      "whilelt  p1.h, wzr, %w[width]                    \n"  //
      READYUV422_SVE I4XXTORGB_SVE RGBTORGBA8_SVE
      "st2h     {z19.h, z20.h}, p1, [%[dst_argb]]       \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width),                               // %[width]
        [vl] "=&r"(vl)                                     // %[vl]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void I422ToAR30Row_SVE_SC(const uint8_t* src_y,
                                        const uint8_t* src_u,
                                        const uint8_t* src_v,
                                        uint8_t* dst_ar30,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  // The limit is used for saturating the 2.14 red channel in STOREAR30_SVE_2X.
  const uint16_t limit = 0x3ff0;
  asm volatile(
      "cnth     %[vl]                                   \n"
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z23.h, %w[limit]                        \n"
      "subs     %w[width], %w[width], %w[vl], lsl #1    \n"
      "b.le     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.b                                    \n"
      "ptrue    p2.b                                    \n"
      "ptrue    p3.b                                    \n"
      "1:                                               \n"  //
      READYUV422_SVE_2X I422TORGB_SVE_2X STOREAR30_SVE_2X
      "subs     %w[width], %w[width], %w[vl], lsl #1    \n"
      "b.gt     1b                                      \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "2:                                               \n"
      "adds    %w[width], %w[width], %w[vl], lsl #1     \n"
      "b.eq    99f                                      \n"

      "whilelt  p1.b, wzr, %w[width]                    \n"
      "whilelt  p2.h, wzr, %w[width]                    \n"
      "whilelt  p3.h, %w[vl], %w[width]                 \n"  //
      READYUV422_SVE_2X I422TORGB_SVE_2X STOREAR30_SVE_2X

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_u] "+r"(src_u),                                // %[src_u]
        [src_v] "+r"(src_v),                                // %[src_v]
        [dst_ar30] "+r"(dst_ar30),                          // %[dst_ar30]
        [width] "+r"(width),                                // %[width]
        [vl] "=&r"(vl)                                      // %[vl]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [limit] "r"(limit)                                  // %[limit]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void I422AlphaToARGBRow_SVE_SC(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    const uint8_t* src_a,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm volatile(
      "cntb     %[vl]                                   \n"
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.b                                    \n"
      "1:                                               \n"  //
      READYUV422_SVE_2X
      "ld1b     {z19.b}, p1/z, [%[src_a]]               \n"
      "add      %[src_a], %[src_a], %[vl]               \n"  // Alpha
      I422TORGB_SVE_2X RGBTOARGB8_SVE_2X
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st4b     {z16.b, z17.b, z18.b, z19.b}, p1, [%[dst_argb]] \n"
      "incb     %[dst_argb], all, mul #4                \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "cnth     %[vl]                                   \n"
      "whilelt  p1.b, wzr, %w[width]                    \n"  //
      READYUV422_SVE_2X
      "ld1b     {z19.b}, p1/z, [%[src_a]]               \n"  // Alpha
      I422TORGB_SVE_2X RGBTOARGB8_SVE_2X
      "st4b     {z16.b, z17.b, z18.b, z19.b}, p1, [%[dst_argb]] \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [src_a] "+r"(src_a),                               // %[src_a]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width),                               // %[width]
        [vl] "=&r"(vl)                                     // %[vl]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void I444AlphaToARGBRow_SVE_SC(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    const uint8_t* src_a,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm volatile(
      "cnth     %[vl]                                   \n"
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "1:                                               \n"  //
      READYUV444_SVE
      "ld1b     {z19.h}, p1/z, [%[src_a]]               \n"
      "add      %[src_a], %[src_a], %[vl]               \n"  // Alpha
      I4XXTORGB_SVE RGBTOARGB8_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2 \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width]                    \n"  //
      READYUV444_SVE
      "ld1b     {z19.h}, p1/z, [%[src_a]]               \n"  // Alpha
      I4XXTORGB_SVE RGBTOARGB8_SVE
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [src_a] "+r"(src_a),                               // %[src_a]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width),                               // %[width]
        [vl] "=&r"(vl)                                     // %[vl]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void NV12ToARGBRow_SVE_SC(const uint8_t* src_y,
                                        const uint8_t* src_uv,
                                        uint8_t* dst_argb,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cntb %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  int width_last_uv = width_last_y + (width_last_y & 1);
  asm volatile(
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z19.b, #255                             \n"  // Alpha
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.b                                    \n"
      "ptrue    p2.b                                    \n"
      "1:                                               \n"  //
      READNV_SVE_2X NVTORGB_SVE_2X(b, t) RGBTOARGB8_SVE_2X
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st4b     {z16.b, z17.b, z18.b, z19.b}, p1, [%[dst_argb]]       \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2 \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.b, wzr, %w[width_last_y]             \n"
      "whilelt  p2.b, wzr, %w[width_last_uv]            \n"  //
      READNV_SVE_2X NVTORGB_SVE_2X(b, t) RGBTOARGB8_SVE_2X
      "st4b     {z16.b, z17.b, z18.b, z19.b}, p1, [%[dst_argb]]       \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_uv] "+r"(src_uv),                              // %[src_uv]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [width_last_y] "r"(width_last_y),                   // %[width_last_y]
        [width_last_uv] "r"(width_last_uv)                  // %[width_last_uv]
      : "cc", "memory", YUVTORGB_SVE_REGS, "p2");
}

static inline void NV21ToARGBRow_SVE_SC(const uint8_t* src_y,
                                        const uint8_t* src_vu,
                                        uint8_t* dst_argb,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cntb %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  int width_last_uv = width_last_y + (width_last_y & 1);
  asm volatile(
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z19.b, #255                             \n"  // Alpha
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.b                                    \n"
      "ptrue    p2.b                                    \n"
      "1:                                               \n"  //
      READNV_SVE_2X NVTORGB_SVE_2X(t, b) RGBTOARGB8_SVE_2X
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st4b     {z16.b, z17.b, z18.b, z19.b}, p1, [%[dst_argb]]       \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2 \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.b, wzr, %w[width_last_y]             \n"
      "whilelt  p2.b, wzr, %w[width_last_uv]            \n"  //
      READNV_SVE_2X NVTORGB_SVE_2X(t, b) RGBTOARGB8_SVE_2X
      "st4b     {z16.b, z17.b, z18.b, z19.b}, p1, [%[dst_argb]]       \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_uv] "+r"(src_vu),                              // %[src_vu]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [width_last_y] "r"(width_last_y),                   // %[width_last_y]
        [width_last_uv] "r"(width_last_uv)                  // %[width_last_uv]
      : "cc", "memory", YUVTORGB_SVE_REGS, "p2");
}

static inline void NV12ToRGB24Row_SVE_SC(
    const uint8_t* src_y,
    const uint8_t* src_uv,
    uint8_t* dst_rgb24,
    const struct YuvConstants* yuvconstants,
    int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cntb %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  int width_last_uv = width_last_y + (width_last_y & 1);
  asm volatile(
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z19.b, #255                             \n"  // Alpha
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.b                                    \n"
      "ptrue    p2.b                                    \n"
      "1:                                               \n"  //
      READNV_SVE_2X NVTORGB_SVE_2X(b, t) RGBTOARGB8_SVE_2X
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st3b     {z16.b, z17.b, z18.b}, p1, [%[dst_rgb24]]       \n"
      "incb     %[dst_rgb24], all, mul #3               \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.b, wzr, %w[width_last_y]             \n"
      "whilelt  p2.b, wzr, %w[width_last_uv]            \n"  //
      READNV_SVE_2X NVTORGB_SVE_2X(b, t) RGBTOARGB8_SVE_2X
      "st3b     {z16.b, z17.b, z18.b}, p1, [%[dst_rgb24]]       \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_uv] "+r"(src_uv),                              // %[src_uv]
        [dst_rgb24] "+r"(dst_rgb24),                        // %[dst_rgb24]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [width_last_y] "r"(width_last_y),                   // %[width_last_y]
        [width_last_uv] "r"(width_last_uv)                  // %[width_last_uv]
      : "cc", "memory", YUVTORGB_SVE_REGS, "p2");
}

static inline void NV21ToRGB24Row_SVE_SC(
    const uint8_t* src_y,
    const uint8_t* src_vu,
    uint8_t* dst_rgb24,
    const struct YuvConstants* yuvconstants,
    int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cntb %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  int width_last_uv = width_last_y + (width_last_y & 1);
  asm volatile(
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z19.b, #255                             \n"  // Alpha
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.b                                    \n"
      "ptrue    p2.b                                    \n"
      "1:                                               \n"  //
      READNV_SVE_2X NVTORGB_SVE_2X(t, b) RGBTOARGB8_SVE_2X
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st3b     {z16.b, z17.b, z18.b}, p1, [%[dst_rgb24]]       \n"
      "incb     %[dst_rgb24], all, mul #3               \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.b, wzr, %w[width_last_y]             \n"
      "whilelt  p2.b, wzr, %w[width_last_uv]            \n"  //
      READNV_SVE_2X NVTORGB_SVE_2X(t, b) RGBTOARGB8_SVE_2X
      "st3b     {z16.b, z17.b, z18.b}, p1, [%[dst_rgb24]]       \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_uv] "+r"(src_vu),                              // %[src_vu]
        [dst_rgb24] "+r"(dst_rgb24),                        // %[dst_rgb24]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [width_last_y] "r"(width_last_y),                   // %[width_last_y]
        [width_last_uv] "r"(width_last_uv)                  // %[width_last_uv]
      : "cc", "memory", YUVTORGB_SVE_REGS, "p2");
}

#define READYUY2_SVE                                        \
  "ld1w       {z0.s}, p2/z, [%[src_yuy2]]    \n" /* YUYV */ \
  "incb       %[src_yuy2]                    \n"            \
  "prfm       pldl1keep, [%[src_yuy2], 448]  \n"            \
  "tbl        z1.b, {z0.b}, z22.b            \n" /* UVUV */ \
  "trn1       z0.b, z0.b, z0.b               \n" /* YYYY */

static inline void YUY2ToARGBRow_SVE_SC(const uint8_t* src_yuy2,
                                        uint8_t* dst_argb,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint32_t nv_uv_start = 0x03010301U;
  uint32_t nv_uv_step = 0x04040404U;
  uint64_t vl;
  asm("cnth %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  int width_last_uv = width_last_y + (width_last_y & 1);
  asm volatile(
      "ptrue    p0.b                                    \n"
      "index    z22.s, %w[nv_uv_start], %w[nv_uv_step]  \n"
      "dup      z19.b, #255                             \n"  // Alpha
      YUVTORGB_SVE_SETUP
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "ptrue    p2.h                                    \n"
      "1:                                               \n"  //
      READYUY2_SVE NVTORGB_SVE RGBTOARGB8_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2 \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width_last_y]             \n"
      "whilelt  p2.h, wzr, %w[width_last_uv]            \n"  //
      READYUY2_SVE NVTORGB_SVE RGBTOARGB8_SVE
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"

      "99:                                              \n"
      : [src_yuy2] "+r"(src_yuy2),                          // %[src_yuy2]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [nv_uv_start] "r"(nv_uv_start),                     // %[nv_uv_start]
        [nv_uv_step] "r"(nv_uv_step),                       // %[nv_uv_step]
        [width_last_y] "r"(width_last_y),                   // %[width_last_y]
        [width_last_uv] "r"(width_last_uv)                  // %[width_last_uv]
      : "cc", "memory", YUVTORGB_SVE_REGS, "p2");
}

#define READUYVY_SVE                                        \
  "ld1w       {z0.s}, p2/z, [%[src_uyvy]]    \n" /* UYVY */ \
  "incb       %[src_uyvy]                    \n"            \
  "prfm       pldl1keep, [%[src_uyvy], 448]  \n"            \
  "tbl        z1.b, {z0.b}, z22.b            \n" /* UVUV */ \
  "trn2       z0.b, z0.b, z0.b               \n" /* YYYY */

static inline void UYVYToARGBRow_SVE_SC(const uint8_t* src_uyvy,
                                        uint8_t* dst_argb,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint32_t nv_uv_start = 0x02000200U;
  uint32_t nv_uv_step = 0x04040404U;
  uint64_t vl;
  asm("cnth %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  int width_last_uv = width_last_y + (width_last_y & 1);
  asm volatile(
      "ptrue    p0.b                                    \n"
      "index    z22.s, %w[nv_uv_start], %w[nv_uv_step]  \n"
      "dup      z19.b, #255                             \n"  // Alpha
      YUVTORGB_SVE_SETUP
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "ptrue    p2.h                                    \n"
      "1:                                               \n"  //
      READUYVY_SVE NVTORGB_SVE RGBTOARGB8_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2 \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "2:                                               \n"
      "whilelt  p1.h, wzr, %w[width_last_y]             \n"
      "whilelt  p2.h, wzr, %w[width_last_uv]            \n"  //
      READUYVY_SVE NVTORGB_SVE RGBTOARGB8_SVE
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"

      "99:                                              \n"
      : [src_uyvy] "+r"(src_uyvy),                          // %[src_yuy2]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [nv_uv_start] "r"(nv_uv_start),                     // %[nv_uv_start]
        [nv_uv_step] "r"(nv_uv_step),                       // %[nv_uv_step]
        [width_last_y] "r"(width_last_y),                   // %[width_last_y]
        [width_last_uv] "r"(width_last_uv)                  // %[width_last_uv]
      : "cc", "memory", YUVTORGB_SVE_REGS, "p2");
}

static inline void I210ToARGBRow_SVE_SC(const uint16_t* src_y,
                                        const uint16_t* src_u,
                                        const uint16_t* src_v,
                                        uint8_t* dst_argb,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cnth %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  asm volatile(
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z19.b, #255                             \n"  // Alpha
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "1:                                               \n"  //
      READI210_SVE I4XXTORGB_SVE RGBTOARGB8_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2 \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width_last_y]             \n"  //
      READI210_SVE I4XXTORGB_SVE RGBTOARGB8_SVE
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_u] "+r"(src_u),                                // %[src_u]
        [src_v] "+r"(src_v),                                // %[src_v]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [width_last_y] "r"(width_last_y)                    // %[width_last_y]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void I210AlphaToARGBRow_SVE_SC(
    const uint16_t* src_y,
    const uint16_t* src_u,
    const uint16_t* src_v,
    const uint16_t* src_a,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cnth %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  asm volatile(
      "ptrue    p0.b                                               \n"  //
      YUVTORGB_SVE_SETUP
      "subs     %w[width], %w[width], %w[vl]                       \n"
      "b.lt     2f                                                 \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                               \n"
      "1:                                                          \n"  //
      READI210_SVE
      "ld1h     {z19.h}, p1/z, [%[src_a]]                          \n"  //
      I4XXTORGB_SVE
      "incb     %[src_a]                                           \n"  //
      RGBATOARGB8_SVE
      "subs     %w[width], %w[width], %w[vl]                       \n"
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]                  \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2            \n"
      "b.ge     1b                                                 \n"

      "2:                                                          \n"
      "adds     %w[width], %w[width], %w[vl]                       \n"
      "b.eq     99f                                                \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width_last_y]                        \n"  //
      READI210_SVE
      "ld1h     {z19.h}, p1/z, [%[src_a]]                          \n"  //
      I4XXTORGB_SVE RGBATOARGB8_SVE
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]                  \n"

      "99:                                                         \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_u] "+r"(src_u),                                // %[src_u]
        [src_v] "+r"(src_v),                                // %[src_v]
        [src_a] "+r"(src_a),                                // %[src_a]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [width_last_y] "r"(width_last_y)                    // %[width_last_y]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void I210ToAR30Row_SVE_SC(const uint16_t* src_y,
                                        const uint16_t* src_u,
                                        const uint16_t* src_v,
                                        uint8_t* dst_ar30,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cnth %0" : "=r"(vl));
  int width_last_y = width & (2 * vl - 1);
  int width_last_uv = (width_last_y + 1) / 2;
  // The limit is used for saturating the 2.14 red channel in STOREAR30_SVE_2X.
  uint16_t limit = 0x3ff0;
  asm volatile(
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z23.h, %w[limit]                        \n"
      "subs     %w[width], %w[width], %w[vl], lsl #1    \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "ptrue    p2.h                                    \n"
      "ptrue    p3.h                                    \n"
      "1:                                               \n"  //
      READI210_SVE_2X I422TORGB_SVE_2X STOREAR30_SVE_2X
      "subs     %w[width], %w[width], %w[vl], lsl #1    \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl], lsl #1    \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width_last_uv]            \n"
      "whilelt  p2.h, wzr, %w[width_last_y]             \n"
      "whilelt  p3.h, %w[vl], %w[width_last_y]          \n"  //
      READI210_SVE_2X I422TORGB_SVE_2X STOREAR30_SVE_2X

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_u] "+r"(src_u),                                // %[src_u]
        [src_v] "+r"(src_v),                                // %[src_v]
        [dst_ar30] "+r"(dst_ar30),                          // %[dst_ar30]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [width_last_y] "r"(width_last_y),                   // %[width_last_y]
        [width_last_uv] "r"(width_last_uv),                 // %[width_last_uv]
        [limit] "r"(limit)                                  // %[limit]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

// P210 has 10 bits in msb of 16 bit NV12 style layout.
static inline void P210ToARGBRow_SVE_SC(const uint16_t* src_y,
                                        const uint16_t* src_uv,
                                        uint8_t* dst_argb,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cnth %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  int width_last_uv = width_last_y + (width_last_y & 1);
  uint32_t nv_uv_start = 0x03010301U;
  uint32_t nv_uv_step = 0x04040404U;
  asm volatile(
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "index    z22.s, %w[nv_uv_start], %w[nv_uv_step]  \n"
      "dup      z19.b, #255                             \n"  // Alpha
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "ptrue    p2.h                                    \n"
      "1:                                               \n"  //
      READP210_SVE NVTORGB_SVE RGBTOARGB8_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2 \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width_last_y]             \n"
      "whilelt  p2.h, wzr, %w[width_last_uv]            \n"  //
      READP210_SVE NVTORGB_SVE RGBTOARGB8_SVE
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_uv] "+r"(src_uv),                              // %[src_uv]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [nv_uv_start] "r"(nv_uv_start),                     // %[nv_uv_start]
        [nv_uv_step] "r"(nv_uv_step),                       // %[nv_uv_step]
        [width_last_y] "r"(width_last_y),                   // %[width_last_y]
        [width_last_uv] "r"(width_last_uv)                  // %[width_last_uv]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void P210ToAR30Row_SVE_SC(const uint16_t* src_y,
                                        const uint16_t* src_uv,
                                        uint8_t* dst_ar30,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cnth %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  int width_last_uv = width_last_y + (width_last_y & 1);
  uint32_t nv_uv_start = 0x03010301U;
  uint32_t nv_uv_step = 0x04040404U;
  // The limit is used for saturating the 2.14 red channel in STOREAR30_SVE.
  uint16_t limit = 0x3ff0;
  asm volatile(
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "index    z22.s, %w[nv_uv_start], %w[nv_uv_step]  \n"
      "dup      z23.h, %w[limit]                        \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "ptrue    p2.h                                    \n"
      "1:                                               \n"  //
      READP210_SVE NVTORGB_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"  //
      STOREAR30_SVE
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width_last_y]             \n"
      "whilelt  p2.h, wzr, %w[width_last_uv]            \n"  //
      READP210_SVE NVTORGB_SVE STOREAR30_SVE

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_uv] "+r"(src_uv),                              // %[src_uv]
        [dst_ar30] "+r"(dst_ar30),                          // %[dst_ar30]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [nv_uv_start] "r"(nv_uv_start),                     // %[nv_uv_start]
        [nv_uv_step] "r"(nv_uv_step),                       // %[nv_uv_step]
        [width_last_y] "r"(width_last_y),                   // %[width_last_y]
        [width_last_uv] "r"(width_last_uv),                 // %[width_last_uv]
        [limit] "r"(limit)                                  // %[limit]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void I410ToARGBRow_SVE_SC(const uint16_t* src_y,
                                        const uint16_t* src_u,
                                        const uint16_t* src_v,
                                        uint8_t* dst_argb,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cnth %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  asm volatile(
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z19.b, #255                             \n"  // Alpha
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "1:                                               \n"  //
      READI410_SVE I4XXTORGB_SVE RGBTOARGB8_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2 \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width_last_y]             \n"  //
      READI410_SVE I4XXTORGB_SVE RGBTOARGB8_SVE
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_u] "+r"(src_u),                                // %[src_u]
        [src_v] "+r"(src_v),                                // %[src_v]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [width_last_y] "r"(width_last_y)                    // %[width_last_y]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void I410AlphaToARGBRow_SVE_SC(
    const uint16_t* src_y,
    const uint16_t* src_u,
    const uint16_t* src_v,
    const uint16_t* src_a,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cnth %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  asm volatile(
      "ptrue    p0.b                                             \n"  //
      YUVTORGB_SVE_SETUP
      "cmp      %w[width], %w[vl]                                \n"
      "subs     %w[width], %w[width], %w[vl]                     \n"
      "b.lt     2f                                               \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                             \n"
      "1:                                                        \n"  //
      READI410_SVE
      "ld1h     {z19.h}, p1/z, [%[src_a]]                        \n"  //
      I4XXTORGB_SVE
      "incb     %[src_a]                                         \n"  //
      RGBATOARGB8_SVE
      "subs     %w[width], %w[width], %w[vl]                     \n"
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]                \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2          \n"
      "b.ge     1b                                               \n"

      "2:                                                        \n"
      "adds     %w[width], %w[width], %w[vl]                     \n"
      "b.eq     99f                                              \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width_last_y]                      \n"  //
      READI410_SVE
      "ld1h     {z19.h}, p1/z, [%[src_a]]                        \n"  //
      I4XXTORGB_SVE RGBATOARGB8_SVE
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]                \n"

      "99:                                                       \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_u] "+r"(src_u),                                // %[src_u]
        [src_v] "+r"(src_v),                                // %[src_v]
        [src_a] "+r"(src_a),                                // %[src_a]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [width_last_y] "r"(width_last_y)                    // %[width_last_y]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void I410ToAR30Row_SVE_SC(const uint16_t* src_y,
                                        const uint16_t* src_u,
                                        const uint16_t* src_v,
                                        uint8_t* dst_ar30,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cnth %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  // The limit is used for saturating the 2.14 red channel in STOREAR30_SVE.
  uint16_t limit = 0x3ff0;
  asm volatile(
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z23.h, %w[limit]                        \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "1:                                               \n"  //
      READI410_SVE I4XXTORGB_SVE STOREAR30_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width_last_y]             \n"  //
      READI410_SVE I4XXTORGB_SVE STOREAR30_SVE

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_u] "+r"(src_u),                                // %[src_u]
        [src_v] "+r"(src_v),                                // %[src_v]
        [dst_ar30] "+r"(dst_ar30),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [width_last_y] "r"(width_last_y),                   // %[width_last_y]
        [limit] "r"(limit)                                  // %[limit]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void P410ToARGBRow_SVE_SC(const uint16_t* src_y,
                                        const uint16_t* src_uv,
                                        uint8_t* dst_argb,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cnth %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  asm volatile(
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z19.b, #255                             \n"  // Alpha
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "ptrue    p2.s                                    \n"
      "ptrue    p3.s                                    \n"
      "1:                                               \n"  //
      READP410_SVE NVTORGB_SVE RGBTOARGB8_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2 \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width_last_y]             \n"
      "whilelt  p2.s, wzr, %w[width_last_y]             \n"
      "cntw     %x[vl]                                  \n"
      "whilelt  p3.s, %w[vl], %w[width_last_y]          \n"  //
      READP410_SVE NVTORGB_SVE RGBTOARGB8_SVE
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_uv] "+r"(src_uv),                              // %[src_uv]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [width_last_y] "r"(width_last_y)                    // %[width_last_y]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void P410ToAR30Row_SVE_SC(const uint16_t* src_y,
                                        const uint16_t* src_uv,
                                        uint8_t* dst_ar30,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cnth %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  // The limit is used for saturating the 2.14 red channel in STOREAR30_SVE.
  uint16_t limit = 0x3ff0;
  asm volatile(
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z23.h, %w[limit]                        \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "ptrue    p2.s                                    \n"
      "ptrue    p3.s                                    \n"
      "1:                                               \n"  //
      READP410_SVE NVTORGB_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"  //
      STOREAR30_SVE
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width_last_y]             \n"
      "whilelt  p2.s, wzr, %w[width_last_y]             \n"
      "cntw     %x[vl]                                  \n"
      "whilelt  p3.s, %w[vl], %w[width_last_y]          \n"  //
      READP410_SVE NVTORGB_SVE STOREAR30_SVE

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_uv] "+r"(src_uv),                              // %[src_uv]
        [dst_ar30] "+r"(dst_ar30),                          // %[dst_ar30]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [width_last_y] "r"(width_last_y),                   // %[width_last_y]
        [limit] "r"(limit)                                  // %[limit]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void I212ToAR30Row_SVE_SC(const uint16_t* src_y,
                                        const uint16_t* src_u,
                                        const uint16_t* src_v,
                                        uint8_t* dst_ar30,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cnth %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  // The limit is used for saturating the 2.14 red channel in STOREAR30_SVE.
  uint16_t limit = 0x3ff0;
  asm volatile(
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z23.h, %w[limit]                        \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "1:                                               \n"  //
      READI212_SVE I4XXTORGB_SVE STOREAR30_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width_last_y]             \n"  //
      READI212_SVE I4XXTORGB_SVE STOREAR30_SVE

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_u] "+r"(src_u),                                // %[src_u]
        [src_v] "+r"(src_v),                                // %[src_v]
        [dst_ar30] "+r"(dst_ar30),                          // %[dst_ar30]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [width_last_y] "r"(width_last_y),                   // %[width_last_y]
        [limit] "r"(limit)                                  // %[limit]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

static inline void I212ToARGBRow_SVE_SC(const uint16_t* src_y,
                                        const uint16_t* src_u,
                                        const uint16_t* src_v,
                                        uint8_t* dst_argb,
                                        const struct YuvConstants* yuvconstants,
                                        int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm("cnth %0" : "=r"(vl));
  int width_last_y = width & (vl - 1);
  asm volatile(
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z19.b, #255                             \n"  // Alpha
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "1:                                               \n"  //
      READI212_SVE I4XXTORGB_SVE RGBTOARGB8_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2 \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width_last_y]             \n"  //
      READI212_SVE I4XXTORGB_SVE RGBTOARGB8_SVE
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_u] "+r"(src_u),                                // %[src_u]
        [src_v] "+r"(src_v),                                // %[src_v]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [vl] "r"(vl),                                       // %[vl]
        [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [width_last_y] "r"(width_last_y)                    // %[width_last_y]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

#define CONVERT8TO8_SVE                                  \
  "ld1b        {z0.b}, p0/z, [%[src]]                \n" \
  "ld1b        {z1.b}, p1/z, [%[src], #1, mul vl]    \n" \
  "incb        %[src], all, mul #2                   \n" \
  "subs        %w[width], %w[width], %w[vl], lsl #1  \n" \
  "umulh       z0.b, z0.b, z2.b                      \n" \
  "umulh       z1.b, z1.b, z2.b                      \n" \
  "prfm        pldl1keep, [%[src], 448]              \n" \
  "add         z0.b, z0.b, z3.b                      \n" \
  "add         z1.b, z1.b, z3.b                      \n" \
  "st1b        {z0.b}, p0, [%[dst]]                  \n" \
  "st1b        {z1.b}, p1, [%[dst], #1, mul vl]      \n" \
  "incb        %[dst], all, mul #2                   \n"

static inline void Convert8To8Row_SVE_SC(const uint8_t* src_y,
                                         uint8_t* dst_y,
                                         int scale,
                                         int bias,
                                         int width) STREAMING_COMPATIBLE {
  uint64_t vl;
  asm volatile(
      "dup      z2.b, %w[scale]                         \n"
      "dup      z3.b, %w[bias]                          \n"
      "cntb     %[vl]                                   \n"
      "subs     %w[width], %w[width], %w[vl], lsl #1    \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with all-true predicates to avoid predicate
      // generation overhead.
      "ptrue    p0.b                                    \n"
      "ptrue    p1.b                                    \n"
      "1:                                               \n"  //
      CONVERT8TO8_SVE
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl], lsl #1    \n"
      "b.eq     99f                                     \n"

      // Calculate predicates for the final iteration to deal with the tail.
      "whilelt     p0.b, wzr, %w[width]                 \n"
      "whilelt     p1.b, %w[vl], %w[width]              \n"  //
      CONVERT8TO8_SVE

      "99:                                              \n"
      : [src] "+r"(src_y),    // %[src]
        [dst] "+r"(dst_y),    // %[dst]
        [width] "+r"(width),  // %[width]
        [vl] "=&r"(vl)        // %[vl]
      : [scale] "r"(scale),   // %[scale]
        [bias] "r"(bias)      // %[bias]
      : "cc", "memory", "z0", "z1", "z2", "z3", "p0", "p1");
}

// SVE constants are stored negated such that we can store 128 in int8_t.

// RGB to BT601 coefficients
// UB   0.875 coefficient = 112
// UG -0.5781 coefficient = -74
// UR -0.2969 coefficient = -38
// VB -0.1406 coefficient = -18
// VG -0.7344 coefficient = -94
// VR   0.875 coefficient = 112

static const int8_t kARGBToUVCoefficients[] = {
    // -UB, -UG, -UR, 0, -VB, -VG, -VR, 0
    -112, 74, 38, 0, 18, 94, -112, 0,
};

static const int8_t kABGRToUVCoefficients[] = {
    // -UR, -UG, -UB, 0, -VR, -VG, -VB, 0
    38, 74, -112, 0, -112, 94, 18, 0,
};

static const int8_t kBGRAToUVCoefficients[] = {
    // 0, -UR, -UG, -UB, 0, -VR, -VG, -VB
    0, 38, 74, -112, 0, -112, 94, 18,
};

static const int8_t kRGBAToUVCoefficients[] = {
    // 0, -UB, -UG, -UR, 0, -VB, -VG, -VR
    0, -112, 74, 38, 0, 18, 94, -112,
};

// RGB to JPEG coefficients
// UB  0.500    coefficient = 128
// UG -0.33126  coefficient = -85
// UR -0.16874  coefficient = -43
// VB -0.08131  coefficient = -21
// VG -0.41869  coefficient = -107
// VR 0.500     coefficient = 128

static const int8_t kARGBToUVJCoefficients[] = {
    // -UB, -UG, -UR, 0, -VB, -VG, -VR, 0
    -128, 85, 43, 0, 21, 107, -128, 0,
};

static const int8_t kABGRToUVJCoefficients[] = {
    // -UR, -UG, -UB, 0, -VR, -VG, -VB, 0
    43, 85, -128, 0, -128, 107, 21, 0,
};

#define ABCDTOUVMATRIX_SVE                                                  \
  "ld1d     {z0.d}, p1/z, [%[src0]]               \n" /* ABCD(bgra) */      \
  "ld1d     {z1.d}, p2/z, [%[src0], #1, mul vl]   \n" /* EFGH(bgra) */      \
  "ld1d     {z2.d}, p3/z, [%[src0], #2, mul vl]   \n" /* IJKL(bgra) */      \
  "ld1d     {z3.d}, p4/z, [%[src0], #3, mul vl]   \n" /* MNOP(bgra) */      \
  "ld1d     {z4.d}, p1/z, [%[src1]]               \n" /* ABCD(bgra) */      \
  "ld1d     {z5.d}, p2/z, [%[src1], #1, mul vl]   \n" /* EFGH(bgra) */      \
  "ld1d     {z6.d}, p3/z, [%[src1], #2, mul vl]   \n" /* IJKL(bgra) */      \
  "ld1d     {z7.d}, p4/z, [%[src1], #3, mul vl]   \n" /* MNOP(bgra) */      \
  "incb     %[src0], all, mul #4                  \n"                       \
  "incb     %[src1], all, mul #4                  \n"                       \
                                                                            \
  "uaddlb   z16.h, z0.b, z4.b                     \n" /* ABCD(br) */        \
  "uaddlb   z18.h, z1.b, z5.b                     \n" /* EFGH(br) */        \
  "uaddlb   z20.h, z2.b, z6.b                     \n" /* IJKL(br) */        \
  "uaddlb   z22.h, z3.b, z7.b                     \n" /* MNOP(br) */        \
  "uaddlt   z17.h, z0.b, z4.b                     \n" /* ABCD(ga) */        \
  "uaddlt   z19.h, z1.b, z5.b                     \n" /* EFGH(ga) */        \
  "uaddlt   z21.h, z2.b, z6.b                     \n" /* IJKL(ga) */        \
  "uaddlt   z23.h, z3.b, z7.b                     \n" /* MNOP(ga) */        \
                                                                            \
  /* Use ADDP on 32-bit elements to add adjacent pairs of 9-bit unsigned */ \
  "addp     z16.s, p0/m, z16.s, z18.s             \n" /* ABEFCDGH(br) */    \
  "addp     z17.s, p0/m, z17.s, z19.s             \n" /* ABEFCDGH(ga) */    \
  "addp     z20.s, p0/m, z20.s, z22.s             \n" /* IJMNKLOP(br) */    \
  "addp     z21.s, p0/m, z21.s, z23.s             \n" /* IJMNKLOP(ga) */    \
                                                                            \
  "rshrnb    z0.b, z16.h, #2                      \n" /* ABEFCDGH(b0r0) */  \
  "rshrnb    z1.b, z20.h, #2                      \n" /* IJMNKLOP(b0r0) */  \
  "rshrnt    z0.b, z17.h, #2                      \n" /* ABEFCDGH(bgra) */  \
  "rshrnt    z1.b, z21.h, #2                      \n" /* IJMNKLOP(bgra) */  \
                                                                            \
  "tbl       z0.s, {z0.s}, z27.s                  \n" /* ABCDEFGH */        \
  "tbl       z1.s, {z1.s}, z27.s                  \n" /* IJKLMNOP */        \
                                                                            \
  "subs     %w[width], %w[width], %w[vl], lsl #2  \n" /* VL per loop */     \
                                                                            \
  "fmov     s16, wzr                              \n"                       \
  "fmov     s17, wzr                              \n"                       \
  "fmov     s20, wzr                              \n"                       \
  "fmov     s21, wzr                              \n"                       \
                                                                            \
  "usdot    z16.s, z0.b, z24.b                    \n"                       \
  "usdot    z17.s, z1.b, z24.b                    \n"                       \
  "usdot    z20.s, z0.b, z25.b                    \n"                       \
  "usdot    z21.s, z1.b, z25.b                    \n"                       \
                                                                            \
  "subhnb   z16.b, z26.h, z16.h                   \n" /* U */               \
  "subhnb   z20.b, z26.h, z20.h                   \n" /* V */               \
  "subhnb   z17.b, z26.h, z17.h                   \n" /* U */               \
  "subhnb   z21.b, z26.h, z21.h                   \n" /* V */               \
                                                                            \
  "uzp1     z16.h, z16.h, z17.h                   \n"                       \
  "uzp1     z20.h, z20.h, z21.h                   \n"                       \
                                                                            \
  "st1b     {z16.h}, p5, [%[dst_u]]               \n" /* U */               \
  "st1b     {z20.h}, p5, [%[dst_v]]               \n" /* V */               \
  "inch     %[dst_u]                              \n"                       \
  "inch     %[dst_v]                              \n"

static inline void ARGBToUVMatrixRow_SVE_SC(const uint8_t* src_argb,
                                            int src_stride_argb,
                                            uint8_t* dst_u,
                                            uint8_t* dst_v,
                                            int width,
                                            const int8_t* uvconstants)
    STREAMING_COMPATIBLE {
  const uint8_t* src_argb_1 = src_argb + src_stride_argb;
  uint64_t vl;
  asm("cntd %x0" : "=r"(vl));

  // Width is a multiple of two here, so halve it.
  width >>= 1;

  asm volatile(
      "ptrue    p0.b                                 \n"
      "ld1rw    {z24.s}, p0/z, [%[uvconstants]]      \n"
      "ld1rw    {z25.s}, p0/z, [%[uvconstants], #4]  \n"
      "mov      z26.h, #0x8000                       \n"  // 128.0 (0x8000)

      // Generate some TBL indices to undo the interleaving from ADDP.
      "index    z0.s, #0, #1                         \n"
      "index    z1.s, #1, #1                         \n"
      "uzp1     z27.s, z0.s, z1.s                    \n"

      "subs     %w[width], %w[width], %w[vl], lsl #2 \n"
      "b.lt    2f                                    \n"

      "ptrue  p1.d                                   \n"
      "ptrue  p2.d                                   \n"
      "ptrue  p3.d                                   \n"
      "ptrue  p4.d                                   \n"
      "ptrue  p5.h                                   \n"
      "1:                                            \n"  //
      ABCDTOUVMATRIX_SVE
      "b.gt     1b                                   \n"

      "2:                                            \n"
      "adds    %w[width], %w[width], %w[vl], lsl #2  \n"
      "b.eq    99f                                   \n"

      "3:                                            \n"
      "whilelt  p1.d, wzr, %w[width]                 \n"
      "whilelt  p2.d, %w[vl], %w[width]              \n"
      "whilelt  p3.d, %w[vl2], %w[width]             \n"
      "whilelt  p4.d, %w[vl3], %w[width]             \n"
      "whilelt  p5.h, wzr, %w[width]                 \n"  //
      ABCDTOUVMATRIX_SVE
      "b.gt     3b                                   \n"

      "99:                                           \n"
      : [src0] "+r"(src_argb),           // %[src0]
        [src1] "+r"(src_argb_1),         // %[src1]
        [dst_u] "+r"(dst_u),             // %[dst_u]
        [dst_v] "+r"(dst_v),             // %[dst_v]
        [width] "+r"(width)              // %[width]
      : [uvconstants] "r"(uvconstants),  // %[uvconstants]
        [vl] "r"(vl),                    // %[vl]
        [vl2] "r"(vl * 2),               // %[vl2]
        [vl3] "r"(vl * 3)                // %[vl3]
      : "cc", "memory", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z16",
        "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26",
        "z27", "p0", "p1", "p2", "p3", "p4", "p5");
}

#endif  // !defined(LIBYUV_DISABLE_SVE) && defined(__aarch64__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_ROW_SVE_H_
