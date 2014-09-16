/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// This module is for Visual C x86.
#if !defined(LIBYUV_DISABLE_X86) && defined(_M_IX86) && defined(_MSC_VER)

#ifdef HAS_ARGBTOYROW_SSSE3

// Constants for ARGB.
static const vec8 kARGBToY = {
  13, 65, 33, 0, 13, 65, 33, 0, 13, 65, 33, 0, 13, 65, 33, 0
};

// JPeg full range.
static const vec8 kARGBToYJ = {
  15, 75, 38, 0, 15, 75, 38, 0, 15, 75, 38, 0, 15, 75, 38, 0
};

static const vec8 kARGBToU = {
  112, -74, -38, 0, 112, -74, -38, 0, 112, -74, -38, 0, 112, -74, -38, 0
};

static const vec8 kARGBToUJ = {
  127, -84, -43, 0, 127, -84, -43, 0, 127, -84, -43, 0, 127, -84, -43, 0
};

static const vec8 kARGBToV = {
  -18, -94, 112, 0, -18, -94, 112, 0, -18, -94, 112, 0, -18, -94, 112, 0,
};

static const vec8 kARGBToVJ = {
  -20, -107, 127, 0, -20, -107, 127, 0, -20, -107, 127, 0, -20, -107, 127, 0
};

// vpermd for vphaddw + vpackuswb vpermd.
static const lvec32 kPermdARGBToY_AVX = {
  0, 4, 1, 5, 2, 6, 3, 7
};

// vpshufb for vphaddw + vpackuswb packed to shorts.
static const lvec8 kShufARGBToUV_AVX = {
  0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
  0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
};

// Constants for BGRA.
static const vec8 kBGRAToY = {
  0, 33, 65, 13, 0, 33, 65, 13, 0, 33, 65, 13, 0, 33, 65, 13
};

static const vec8 kBGRAToU = {
  0, -38, -74, 112, 0, -38, -74, 112, 0, -38, -74, 112, 0, -38, -74, 112
};

static const vec8 kBGRAToV = {
  0, 112, -94, -18, 0, 112, -94, -18, 0, 112, -94, -18, 0, 112, -94, -18
};

// Constants for ABGR.
static const vec8 kABGRToY = {
  33, 65, 13, 0, 33, 65, 13, 0, 33, 65, 13, 0, 33, 65, 13, 0
};

static const vec8 kABGRToU = {
  -38, -74, 112, 0, -38, -74, 112, 0, -38, -74, 112, 0, -38, -74, 112, 0
};

static const vec8 kABGRToV = {
  112, -94, -18, 0, 112, -94, -18, 0, 112, -94, -18, 0, 112, -94, -18, 0
};

// Constants for RGBA.
static const vec8 kRGBAToY = {
  0, 13, 65, 33, 0, 13, 65, 33, 0, 13, 65, 33, 0, 13, 65, 33
};

static const vec8 kRGBAToU = {
  0, 112, -74, -38, 0, 112, -74, -38, 0, 112, -74, -38, 0, 112, -74, -38
};

static const vec8 kRGBAToV = {
  0, -18, -94, 112, 0, -18, -94, 112, 0, -18, -94, 112, 0, -18, -94, 112
};

static const uvec8 kAddY16 = {
  16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u
};

static const vec16 kAddYJ64 = {
  64, 64, 64, 64, 64, 64, 64, 64
};

static const uvec8 kAddUV128 = {
  128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u,
  128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u
};

static const uvec16 kAddUVJ128 = {
  0x8080u, 0x8080u, 0x8080u, 0x8080u, 0x8080u, 0x8080u, 0x8080u, 0x8080u
};

// Shuffle table for converting RGB24 to ARGB.
static const uvec8 kShuffleMaskRGB24ToARGB = {
  0u, 1u, 2u, 12u, 3u, 4u, 5u, 13u, 6u, 7u, 8u, 14u, 9u, 10u, 11u, 15u
};

// Shuffle table for converting RAW to ARGB.
static const uvec8 kShuffleMaskRAWToARGB = {
  2u, 1u, 0u, 12u, 5u, 4u, 3u, 13u, 8u, 7u, 6u, 14u, 11u, 10u, 9u, 15u
};

// Shuffle table for converting ARGB to RGB24.
static const uvec8 kShuffleMaskARGBToRGB24 = {
  0u, 1u, 2u, 4u, 5u, 6u, 8u, 9u, 10u, 12u, 13u, 14u, 128u, 128u, 128u, 128u
};

// Shuffle table for converting ARGB to RAW.
static const uvec8 kShuffleMaskARGBToRAW = {
  2u, 1u, 0u, 6u, 5u, 4u, 10u, 9u, 8u, 14u, 13u, 12u, 128u, 128u, 128u, 128u
};

// Shuffle table for converting ARGBToRGB24 for I422ToRGB24.  First 8 + next 4
static const uvec8 kShuffleMaskARGBToRGB24_0 = {
  0u, 1u, 2u, 4u, 5u, 6u, 8u, 9u, 128u, 128u, 128u, 128u, 10u, 12u, 13u, 14u
};

// Shuffle table for converting ARGB to RAW.
static const uvec8 kShuffleMaskARGBToRAW_0 = {
  2u, 1u, 0u, 6u, 5u, 4u, 10u, 9u, 128u, 128u, 128u, 128u, 8u, 14u, 13u, 12u
};

// Duplicates gray value 3 times and fills in alpha opaque.
__declspec(naked) __declspec(align(16))
void I400ToARGBRow_SSE2(const uint8* src_y, uint8* dst_argb, int pix) {
  __asm {
    mov        eax, [esp + 4]        // src_y
    mov        edx, [esp + 8]        // dst_argb
    mov        ecx, [esp + 12]       // pix
    pcmpeqb    xmm5, xmm5            // generate mask 0xff000000
    pslld      xmm5, 24

    align      4
  convertloop:
    movq       xmm0, qword ptr [eax]
    lea        eax,  [eax + 8]
    punpcklbw  xmm0, xmm0
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm0
    punpckhwd  xmm1, xmm1
    por        xmm0, xmm5
    por        xmm1, xmm5
    movdqa     [edx], xmm0
    movdqa     [edx + 16], xmm1
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void I400ToARGBRow_Unaligned_SSE2(const uint8* src_y, uint8* dst_argb,
                                  int pix) {
  __asm {
    mov        eax, [esp + 4]        // src_y
    mov        edx, [esp + 8]        // dst_argb
    mov        ecx, [esp + 12]       // pix
    pcmpeqb    xmm5, xmm5            // generate mask 0xff000000
    pslld      xmm5, 24

    align      4
  convertloop:
    movq       xmm0, qword ptr [eax]
    lea        eax,  [eax + 8]
    punpcklbw  xmm0, xmm0
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm0
    punpckhwd  xmm1, xmm1
    por        xmm0, xmm5
    por        xmm1, xmm5
    movdqu     [edx], xmm0
    movdqu     [edx + 16], xmm1
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void RGB24ToARGBRow_SSSE3(const uint8* src_rgb24, uint8* dst_argb, int pix) {
  __asm {
    mov       eax, [esp + 4]   // src_rgb24
    mov       edx, [esp + 8]   // dst_argb
    mov       ecx, [esp + 12]  // pix
    pcmpeqb   xmm5, xmm5       // generate mask 0xff000000
    pslld     xmm5, 24
    movdqa    xmm4, kShuffleMaskRGB24ToARGB

    align      4
 convertloop:
    movdqu    xmm0, [eax]
    movdqu    xmm1, [eax + 16]
    movdqu    xmm3, [eax + 32]
    lea       eax, [eax + 48]
    movdqa    xmm2, xmm3
    palignr   xmm2, xmm1, 8    // xmm2 = { xmm3[0:3] xmm1[8:15]}
    pshufb    xmm2, xmm4
    por       xmm2, xmm5
    palignr   xmm1, xmm0, 12   // xmm1 = { xmm3[0:7] xmm0[12:15]}
    pshufb    xmm0, xmm4
    movdqa    [edx + 32], xmm2
    por       xmm0, xmm5
    pshufb    xmm1, xmm4
    movdqa    [edx], xmm0
    por       xmm1, xmm5
    palignr   xmm3, xmm3, 4    // xmm3 = { xmm3[4:15]}
    pshufb    xmm3, xmm4
    movdqa    [edx + 16], xmm1
    por       xmm3, xmm5
    sub       ecx, 16
    movdqa    [edx + 48], xmm3
    lea       edx, [edx + 64]
    jg        convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void RAWToARGBRow_SSSE3(const uint8* src_raw, uint8* dst_argb,
                        int pix) {
  __asm {
    mov       eax, [esp + 4]   // src_raw
    mov       edx, [esp + 8]   // dst_argb
    mov       ecx, [esp + 12]  // pix
    pcmpeqb   xmm5, xmm5       // generate mask 0xff000000
    pslld     xmm5, 24
    movdqa    xmm4, kShuffleMaskRAWToARGB

    align      4
 convertloop:
    movdqu    xmm0, [eax]
    movdqu    xmm1, [eax + 16]
    movdqu    xmm3, [eax + 32]
    lea       eax, [eax + 48]
    movdqa    xmm2, xmm3
    palignr   xmm2, xmm1, 8    // xmm2 = { xmm3[0:3] xmm1[8:15]}
    pshufb    xmm2, xmm4
    por       xmm2, xmm5
    palignr   xmm1, xmm0, 12   // xmm1 = { xmm3[0:7] xmm0[12:15]}
    pshufb    xmm0, xmm4
    movdqa    [edx + 32], xmm2
    por       xmm0, xmm5
    pshufb    xmm1, xmm4
    movdqa    [edx], xmm0
    por       xmm1, xmm5
    palignr   xmm3, xmm3, 4    // xmm3 = { xmm3[4:15]}
    pshufb    xmm3, xmm4
    movdqa    [edx + 16], xmm1
    por       xmm3, xmm5
    sub       ecx, 16
    movdqa    [edx + 48], xmm3
    lea       edx, [edx + 64]
    jg        convertloop
    ret
  }
}

// pmul method to replicate bits.
// Math to replicate bits:
// (v << 8) | (v << 3)
// v * 256 + v * 8
// v * (256 + 8)
// G shift of 5 is incorporated, so shift is 5 + 8 and 5 + 3
// 20 instructions.
__declspec(naked) __declspec(align(16))
void RGB565ToARGBRow_SSE2(const uint8* src_rgb565, uint8* dst_argb,
                          int pix) {
  __asm {
    mov       eax, 0x01080108  // generate multiplier to repeat 5 bits
    movd      xmm5, eax
    pshufd    xmm5, xmm5, 0
    mov       eax, 0x20802080  // multiplier shift by 5 and then repeat 6 bits
    movd      xmm6, eax
    pshufd    xmm6, xmm6, 0
    pcmpeqb   xmm3, xmm3       // generate mask 0xf800f800 for Red
    psllw     xmm3, 11
    pcmpeqb   xmm4, xmm4       // generate mask 0x07e007e0 for Green
    psllw     xmm4, 10
    psrlw     xmm4, 5
    pcmpeqb   xmm7, xmm7       // generate mask 0xff00ff00 for Alpha
    psllw     xmm7, 8

    mov       eax, [esp + 4]   // src_rgb565
    mov       edx, [esp + 8]   // dst_argb
    mov       ecx, [esp + 12]  // pix
    sub       edx, eax
    sub       edx, eax

    align      4
 convertloop:
    movdqu    xmm0, [eax]   // fetch 8 pixels of bgr565
    movdqa    xmm1, xmm0
    movdqa    xmm2, xmm0
    pand      xmm1, xmm3    // R in upper 5 bits
    psllw     xmm2, 11      // B in upper 5 bits
    pmulhuw   xmm1, xmm5    // * (256 + 8)
    pmulhuw   xmm2, xmm5    // * (256 + 8)
    psllw     xmm1, 8
    por       xmm1, xmm2    // RB
    pand      xmm0, xmm4    // G in middle 6 bits
    pmulhuw   xmm0, xmm6    // << 5 * (256 + 4)
    por       xmm0, xmm7    // AG
    movdqa    xmm2, xmm1
    punpcklbw xmm1, xmm0
    punpckhbw xmm2, xmm0
    movdqa    [eax * 2 + edx], xmm1  // store 4 pixels of ARGB
    movdqa    [eax * 2 + edx + 16], xmm2  // store next 4 pixels of ARGB
    lea       eax, [eax + 16]
    sub       ecx, 8
    jg        convertloop
    ret
  }
}

// 24 instructions
__declspec(naked) __declspec(align(16))
void ARGB1555ToARGBRow_SSE2(const uint8* src_argb1555, uint8* dst_argb,
                            int pix) {
  __asm {
    mov       eax, 0x01080108  // generate multiplier to repeat 5 bits
    movd      xmm5, eax
    pshufd    xmm5, xmm5, 0
    mov       eax, 0x42004200  // multiplier shift by 6 and then repeat 5 bits
    movd      xmm6, eax
    pshufd    xmm6, xmm6, 0
    pcmpeqb   xmm3, xmm3       // generate mask 0xf800f800 for Red
    psllw     xmm3, 11
    movdqa    xmm4, xmm3       // generate mask 0x03e003e0 for Green
    psrlw     xmm4, 6
    pcmpeqb   xmm7, xmm7       // generate mask 0xff00ff00 for Alpha
    psllw     xmm7, 8

    mov       eax, [esp + 4]   // src_argb1555
    mov       edx, [esp + 8]   // dst_argb
    mov       ecx, [esp + 12]  // pix
    sub       edx, eax
    sub       edx, eax

    align      4
 convertloop:
    movdqu    xmm0, [eax]   // fetch 8 pixels of 1555
    movdqa    xmm1, xmm0
    movdqa    xmm2, xmm0
    psllw     xmm1, 1       // R in upper 5 bits
    psllw     xmm2, 11      // B in upper 5 bits
    pand      xmm1, xmm3
    pmulhuw   xmm2, xmm5    // * (256 + 8)
    pmulhuw   xmm1, xmm5    // * (256 + 8)
    psllw     xmm1, 8
    por       xmm1, xmm2    // RB
    movdqa    xmm2, xmm0
    pand      xmm0, xmm4    // G in middle 5 bits
    psraw     xmm2, 8       // A
    pmulhuw   xmm0, xmm6    // << 6 * (256 + 8)
    pand      xmm2, xmm7
    por       xmm0, xmm2    // AG
    movdqa    xmm2, xmm1
    punpcklbw xmm1, xmm0
    punpckhbw xmm2, xmm0
    movdqa    [eax * 2 + edx], xmm1  // store 4 pixels of ARGB
    movdqa    [eax * 2 + edx + 16], xmm2  // store next 4 pixels of ARGB
    lea       eax, [eax + 16]
    sub       ecx, 8
    jg        convertloop
    ret
  }
}

// 18 instructions.
__declspec(naked) __declspec(align(16))
void ARGB4444ToARGBRow_SSE2(const uint8* src_argb4444, uint8* dst_argb,
                            int pix) {
  __asm {
    mov       eax, 0x0f0f0f0f  // generate mask 0x0f0f0f0f
    movd      xmm4, eax
    pshufd    xmm4, xmm4, 0
    movdqa    xmm5, xmm4       // 0xf0f0f0f0 for high nibbles
    pslld     xmm5, 4
    mov       eax, [esp + 4]   // src_argb4444
    mov       edx, [esp + 8]   // dst_argb
    mov       ecx, [esp + 12]  // pix
    sub       edx, eax
    sub       edx, eax

    align      4
 convertloop:
    movdqu    xmm0, [eax]   // fetch 8 pixels of bgra4444
    movdqa    xmm2, xmm0
    pand      xmm0, xmm4    // mask low nibbles
    pand      xmm2, xmm5    // mask high nibbles
    movdqa    xmm1, xmm0
    movdqa    xmm3, xmm2
    psllw     xmm1, 4
    psrlw     xmm3, 4
    por       xmm0, xmm1
    por       xmm2, xmm3
    movdqa    xmm1, xmm0
    punpcklbw xmm0, xmm2
    punpckhbw xmm1, xmm2
    movdqa    [eax * 2 + edx], xmm0  // store 4 pixels of ARGB
    movdqa    [eax * 2 + edx + 16], xmm1  // store next 4 pixels of ARGB
    lea       eax, [eax + 16]
    sub       ecx, 8
    jg        convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ARGBToRGB24Row_SSSE3(const uint8* src_argb, uint8* dst_rgb, int pix) {
  __asm {
    mov       eax, [esp + 4]   // src_argb
    mov       edx, [esp + 8]   // dst_rgb
    mov       ecx, [esp + 12]  // pix
    movdqa    xmm6, kShuffleMaskARGBToRGB24

    align      4
 convertloop:
    movdqu    xmm0, [eax]   // fetch 16 pixels of argb
    movdqu    xmm1, [eax + 16]
    movdqu    xmm2, [eax + 32]
    movdqu    xmm3, [eax + 48]
    lea       eax, [eax + 64]
    pshufb    xmm0, xmm6    // pack 16 bytes of ARGB to 12 bytes of RGB
    pshufb    xmm1, xmm6
    pshufb    xmm2, xmm6
    pshufb    xmm3, xmm6
    movdqa    xmm4, xmm1   // 4 bytes from 1 for 0
    psrldq    xmm1, 4      // 8 bytes from 1
    pslldq    xmm4, 12     // 4 bytes from 1 for 0
    movdqa    xmm5, xmm2   // 8 bytes from 2 for 1
    por       xmm0, xmm4   // 4 bytes from 1 for 0
    pslldq    xmm5, 8      // 8 bytes from 2 for 1
    movdqu    [edx], xmm0  // store 0
    por       xmm1, xmm5   // 8 bytes from 2 for 1
    psrldq    xmm2, 8      // 4 bytes from 2
    pslldq    xmm3, 4      // 12 bytes from 3 for 2
    por       xmm2, xmm3   // 12 bytes from 3 for 2
    movdqu    [edx + 16], xmm1   // store 1
    movdqu    [edx + 32], xmm2   // store 2
    lea       edx, [edx + 48]
    sub       ecx, 16
    jg        convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ARGBToRAWRow_SSSE3(const uint8* src_argb, uint8* dst_rgb, int pix) {
  __asm {
    mov       eax, [esp + 4]   // src_argb
    mov       edx, [esp + 8]   // dst_rgb
    mov       ecx, [esp + 12]  // pix
    movdqa    xmm6, kShuffleMaskARGBToRAW

    align      4
 convertloop:
    movdqu    xmm0, [eax]   // fetch 16 pixels of argb
    movdqu    xmm1, [eax + 16]
    movdqu    xmm2, [eax + 32]
    movdqu    xmm3, [eax + 48]
    lea       eax, [eax + 64]
    pshufb    xmm0, xmm6    // pack 16 bytes of ARGB to 12 bytes of RGB
    pshufb    xmm1, xmm6
    pshufb    xmm2, xmm6
    pshufb    xmm3, xmm6
    movdqa    xmm4, xmm1   // 4 bytes from 1 for 0
    psrldq    xmm1, 4      // 8 bytes from 1
    pslldq    xmm4, 12     // 4 bytes from 1 for 0
    movdqa    xmm5, xmm2   // 8 bytes from 2 for 1
    por       xmm0, xmm4   // 4 bytes from 1 for 0
    pslldq    xmm5, 8      // 8 bytes from 2 for 1
    movdqu    [edx], xmm0  // store 0
    por       xmm1, xmm5   // 8 bytes from 2 for 1
    psrldq    xmm2, 8      // 4 bytes from 2
    pslldq    xmm3, 4      // 12 bytes from 3 for 2
    por       xmm2, xmm3   // 12 bytes from 3 for 2
    movdqu    [edx + 16], xmm1   // store 1
    movdqu    [edx + 32], xmm2   // store 2
    lea       edx, [edx + 48]
    sub       ecx, 16
    jg        convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ARGBToRGB565Row_SSE2(const uint8* src_argb, uint8* dst_rgb, int pix) {
  __asm {
    mov       eax, [esp + 4]   // src_argb
    mov       edx, [esp + 8]   // dst_rgb
    mov       ecx, [esp + 12]  // pix
    pcmpeqb   xmm3, xmm3       // generate mask 0x0000001f
    psrld     xmm3, 27
    pcmpeqb   xmm4, xmm4       // generate mask 0x000007e0
    psrld     xmm4, 26
    pslld     xmm4, 5
    pcmpeqb   xmm5, xmm5       // generate mask 0xfffff800
    pslld     xmm5, 11

    align      4
 convertloop:
    movdqa    xmm0, [eax]   // fetch 4 pixels of argb
    movdqa    xmm1, xmm0    // B
    movdqa    xmm2, xmm0    // G
    pslld     xmm0, 8       // R
    psrld     xmm1, 3       // B
    psrld     xmm2, 5       // G
    psrad     xmm0, 16      // R
    pand      xmm1, xmm3    // B
    pand      xmm2, xmm4    // G
    pand      xmm0, xmm5    // R
    por       xmm1, xmm2    // BG
    por       xmm0, xmm1    // BGR
    packssdw  xmm0, xmm0
    lea       eax, [eax + 16]
    movq      qword ptr [edx], xmm0  // store 4 pixels of RGB565
    lea       edx, [edx + 8]
    sub       ecx, 4
    jg        convertloop
    ret
  }
}

// TODO(fbarchard): Improve sign extension/packing.
__declspec(naked) __declspec(align(16))
void ARGBToARGB1555Row_SSE2(const uint8* src_argb, uint8* dst_rgb, int pix) {
  __asm {
    mov       eax, [esp + 4]   // src_argb
    mov       edx, [esp + 8]   // dst_rgb
    mov       ecx, [esp + 12]  // pix
    pcmpeqb   xmm4, xmm4       // generate mask 0x0000001f
    psrld     xmm4, 27
    movdqa    xmm5, xmm4       // generate mask 0x000003e0
    pslld     xmm5, 5
    movdqa    xmm6, xmm4       // generate mask 0x00007c00
    pslld     xmm6, 10
    pcmpeqb   xmm7, xmm7       // generate mask 0xffff8000
    pslld     xmm7, 15

    align      4
 convertloop:
    movdqa    xmm0, [eax]   // fetch 4 pixels of argb
    movdqa    xmm1, xmm0    // B
    movdqa    xmm2, xmm0    // G
    movdqa    xmm3, xmm0    // R
    psrad     xmm0, 16      // A
    psrld     xmm1, 3       // B
    psrld     xmm2, 6       // G
    psrld     xmm3, 9       // R
    pand      xmm0, xmm7    // A
    pand      xmm1, xmm4    // B
    pand      xmm2, xmm5    // G
    pand      xmm3, xmm6    // R
    por       xmm0, xmm1    // BA
    por       xmm2, xmm3    // GR
    por       xmm0, xmm2    // BGRA
    packssdw  xmm0, xmm0
    lea       eax, [eax + 16]
    movq      qword ptr [edx], xmm0  // store 4 pixels of ARGB1555
    lea       edx, [edx + 8]
    sub       ecx, 4
    jg        convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ARGBToARGB4444Row_SSE2(const uint8* src_argb, uint8* dst_rgb, int pix) {
  __asm {
    mov       eax, [esp + 4]   // src_argb
    mov       edx, [esp + 8]   // dst_rgb
    mov       ecx, [esp + 12]  // pix
    pcmpeqb   xmm4, xmm4       // generate mask 0xf000f000
    psllw     xmm4, 12
    movdqa    xmm3, xmm4       // generate mask 0x00f000f0
    psrlw     xmm3, 8

    align      4
 convertloop:
    movdqa    xmm0, [eax]   // fetch 4 pixels of argb
    movdqa    xmm1, xmm0
    pand      xmm0, xmm3    // low nibble
    pand      xmm1, xmm4    // high nibble
    psrl      xmm0, 4
    psrl      xmm1, 8
    por       xmm0, xmm1
    packuswb  xmm0, xmm0
    lea       eax, [eax + 16]
    movq      qword ptr [edx], xmm0  // store 4 pixels of ARGB4444
    lea       edx, [edx + 8]
    sub       ecx, 4
    jg        convertloop
    ret
  }
}

// Convert 16 ARGB pixels (64 bytes) to 16 Y values.
__declspec(naked) __declspec(align(16))
void ARGBToYRow_SSSE3(const uint8* src_argb, uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_y */
    mov        ecx, [esp + 12]  /* pix */
    movdqa     xmm5, kAddY16
    movdqa     xmm4, kARGBToY

    align      4
 convertloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    paddb      xmm0, xmm5
    sub        ecx, 16
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    ret
  }
}

// Convert 16 ARGB pixels (64 bytes) to 16 Y values.
__declspec(naked) __declspec(align(16))
void ARGBToYJRow_SSSE3(const uint8* src_argb, uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_y */
    mov        ecx, [esp + 12]  /* pix */
    movdqa     xmm4, kARGBToYJ
    movdqa     xmm5, kAddYJ64

    align      4
 convertloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    paddw      xmm0, xmm5  // Add .5 for rounding.
    paddw      xmm2, xmm5
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    sub        ecx, 16
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    ret
  }
}

#ifdef HAS_ARGBTOYROW_AVX2
// Convert 32 ARGB pixels (128 bytes) to 32 Y values.
__declspec(naked) __declspec(align(32))
void ARGBToYRow_AVX2(const uint8* src_argb, uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_y */
    mov        ecx, [esp + 12]  /* pix */
    vbroadcastf128 ymm4, kARGBToY
    vbroadcastf128 ymm5, kAddY16
    vmovdqa    ymm6, kPermdARGBToY_AVX

    align      4
 convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    vmovdqu    ymm2, [eax + 64]
    vmovdqu    ymm3, [eax + 96]
    vpmaddubsw ymm0, ymm0, ymm4
    vpmaddubsw ymm1, ymm1, ymm4
    vpmaddubsw ymm2, ymm2, ymm4
    vpmaddubsw ymm3, ymm3, ymm4
    lea        eax, [eax + 128]
    vphaddw    ymm0, ymm0, ymm1  // mutates.
    vphaddw    ymm2, ymm2, ymm3
    vpsrlw     ymm0, ymm0, 7
    vpsrlw     ymm2, ymm2, 7
    vpackuswb  ymm0, ymm0, ymm2  // mutates.
    vpermd     ymm0, ymm6, ymm0  // For vphaddw + vpackuswb mutation.
    vpaddb     ymm0, ymm0, ymm5
    sub        ecx, 32
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    jg         convertloop
    vzeroupper
    ret
  }
}
#endif  //  HAS_ARGBTOYROW_AVX2

#ifdef HAS_ARGBTOYROW_AVX2
// Convert 32 ARGB pixels (128 bytes) to 32 Y values.
__declspec(naked) __declspec(align(32))
void ARGBToYJRow_AVX2(const uint8* src_argb, uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_y */
    mov        ecx, [esp + 12]  /* pix */
    vbroadcastf128 ymm4, kARGBToYJ
    vbroadcastf128 ymm5, kAddYJ64
    vmovdqa    ymm6, kPermdARGBToY_AVX

    align      4
 convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    vmovdqu    ymm2, [eax + 64]
    vmovdqu    ymm3, [eax + 96]
    vpmaddubsw ymm0, ymm0, ymm4
    vpmaddubsw ymm1, ymm1, ymm4
    vpmaddubsw ymm2, ymm2, ymm4
    vpmaddubsw ymm3, ymm3, ymm4
    lea        eax, [eax + 128]
    vphaddw    ymm0, ymm0, ymm1  // mutates.
    vphaddw    ymm2, ymm2, ymm3
    vpaddw     ymm0, ymm0, ymm5  // Add .5 for rounding.
    vpaddw     ymm2, ymm2, ymm5
    vpsrlw     ymm0, ymm0, 7
    vpsrlw     ymm2, ymm2, 7
    vpackuswb  ymm0, ymm0, ymm2  // mutates.
    vpermd     ymm0, ymm6, ymm0  // For vphaddw + vpackuswb mutation.
    sub        ecx, 32
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    jg         convertloop

    vzeroupper
    ret
  }
}
#endif  //  HAS_ARGBTOYJROW_AVX2

__declspec(naked) __declspec(align(16))
void ARGBToYRow_Unaligned_SSSE3(const uint8* src_argb, uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_y */
    mov        ecx, [esp + 12]  /* pix */
    movdqa     xmm5, kAddY16
    movdqa     xmm4, kARGBToY

    align      4
 convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    paddb      xmm0, xmm5
    sub        ecx, 16
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ARGBToYJRow_Unaligned_SSSE3(const uint8* src_argb, uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_y */
    mov        ecx, [esp + 12]  /* pix */
    movdqa     xmm4, kARGBToYJ
    movdqa     xmm5, kAddYJ64

    align      4
 convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    paddw      xmm0, xmm5
    paddw      xmm2, xmm5
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    sub        ecx, 16
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void BGRAToYRow_SSSE3(const uint8* src_argb, uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_y */
    mov        ecx, [esp + 12]  /* pix */
    movdqa     xmm5, kAddY16
    movdqa     xmm4, kBGRAToY

    align      4
 convertloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    paddb      xmm0, xmm5
    sub        ecx, 16
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void BGRAToYRow_Unaligned_SSSE3(const uint8* src_argb, uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_y */
    mov        ecx, [esp + 12]  /* pix */
    movdqa     xmm5, kAddY16
    movdqa     xmm4, kBGRAToY

    align      4
 convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    paddb      xmm0, xmm5
    sub        ecx, 16
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ABGRToYRow_SSSE3(const uint8* src_argb, uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_y */
    mov        ecx, [esp + 12]  /* pix */
    movdqa     xmm5, kAddY16
    movdqa     xmm4, kABGRToY

    align      4
 convertloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    paddb      xmm0, xmm5
    sub        ecx, 16
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ABGRToYRow_Unaligned_SSSE3(const uint8* src_argb, uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_y */
    mov        ecx, [esp + 12]  /* pix */
    movdqa     xmm5, kAddY16
    movdqa     xmm4, kABGRToY

    align      4
 convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    paddb      xmm0, xmm5
    sub        ecx, 16
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void RGBAToYRow_SSSE3(const uint8* src_argb, uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_y */
    mov        ecx, [esp + 12]  /* pix */
    movdqa     xmm5, kAddY16
    movdqa     xmm4, kRGBAToY

    align      4
 convertloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    paddb      xmm0, xmm5
    sub        ecx, 16
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void RGBAToYRow_Unaligned_SSSE3(const uint8* src_argb, uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_y */
    mov        ecx, [esp + 12]  /* pix */
    movdqa     xmm5, kAddY16
    movdqa     xmm4, kRGBAToY

    align      4
 convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    paddb      xmm0, xmm5
    sub        ecx, 16
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ARGBToUVRow_SSSE3(const uint8* src_argb0, int src_stride_argb,
                       uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // src_argb
    mov        esi, [esp + 8 + 8]   // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // pix
    movdqa     xmm7, kARGBToU
    movdqa     xmm6, kARGBToV
    movdqa     xmm5, kAddUV128
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]
    pavgb      xmm0, [eax + esi]
    pavgb      xmm1, [eax + esi + 16]
    pavgb      xmm2, [eax + esi + 32]
    pavgb      xmm3, [eax + esi + 48]
    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

    // step 2 - convert to U and V
    // from here down is very similar to Y code except
    // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1
    paddb      xmm0, xmm5            // -> unsigned

    // step 3 - store 8 U and 8 V values
    sub        ecx, 16
    movlps     qword ptr [edx], xmm0 // U
    movhps     qword ptr [edx + edi], xmm0 // V
    lea        edx, [edx + 8]
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ARGBToUVJRow_SSSE3(const uint8* src_argb0, int src_stride_argb,
                        uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // src_argb
    mov        esi, [esp + 8 + 8]   // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // pix
    movdqa     xmm7, kARGBToUJ
    movdqa     xmm6, kARGBToVJ
    movdqa     xmm5, kAddUVJ128
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]
    pavgb      xmm0, [eax + esi]
    pavgb      xmm1, [eax + esi + 16]
    pavgb      xmm2, [eax + esi + 32]
    pavgb      xmm3, [eax + esi + 48]
    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

    // step 2 - convert to U and V
    // from here down is very similar to Y code except
    // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    paddw      xmm0, xmm5            // +.5 rounding -> unsigned
    paddw      xmm1, xmm5
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1

    // step 3 - store 8 U and 8 V values
    sub        ecx, 16
    movlps     qword ptr [edx], xmm0 // U
    movhps     qword ptr [edx + edi], xmm0 // V
    lea        edx, [edx + 8]
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

#ifdef HAS_ARGBTOUVROW_AVX2
__declspec(naked) __declspec(align(32))
void ARGBToUVRow_AVX2(const uint8* src_argb0, int src_stride_argb,
                      uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // src_argb
    mov        esi, [esp + 8 + 8]   // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // pix
    vbroadcastf128 ymm5, kAddUV128
    vbroadcastf128 ymm6, kARGBToV
    vbroadcastf128 ymm7, kARGBToU
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* step 1 - subsample 32x2 argb pixels to 16x1 */
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    vmovdqu    ymm2, [eax + 64]
    vmovdqu    ymm3, [eax + 96]
    vpavgb     ymm0, ymm0, [eax + esi]
    vpavgb     ymm1, ymm1, [eax + esi + 32]
    vpavgb     ymm2, ymm2, [eax + esi + 64]
    vpavgb     ymm3, ymm3, [eax + esi + 96]
    lea        eax,  [eax + 128]
    vshufps    ymm4, ymm0, ymm1, 0x88
    vshufps    ymm0, ymm0, ymm1, 0xdd
    vpavgb     ymm0, ymm0, ymm4  // mutated by vshufps
    vshufps    ymm4, ymm2, ymm3, 0x88
    vshufps    ymm2, ymm2, ymm3, 0xdd
    vpavgb     ymm2, ymm2, ymm4  // mutated by vshufps

    // step 2 - convert to U and V
    // from here down is very similar to Y code except
    // instead of 32 different pixels, its 16 pixels of U and 16 of V
    vpmaddubsw ymm1, ymm0, ymm7  // U
    vpmaddubsw ymm3, ymm2, ymm7
    vpmaddubsw ymm0, ymm0, ymm6  // V
    vpmaddubsw ymm2, ymm2, ymm6
    vphaddw    ymm1, ymm1, ymm3  // mutates
    vphaddw    ymm0, ymm0, ymm2
    vpsraw     ymm1, ymm1, 8
    vpsraw     ymm0, ymm0, 8
    vpacksswb  ymm0, ymm1, ymm0  // mutates
    vpermq     ymm0, ymm0, 0xd8  // For vpacksswb
    vpshufb    ymm0, ymm0, kShufARGBToUV_AVX  // For vshufps + vphaddw
    vpaddb     ymm0, ymm0, ymm5  // -> unsigned

    // step 3 - store 16 U and 16 V values
    sub         ecx, 32
    vextractf128 [edx], ymm0, 0 // U
    vextractf128 [edx + edi], ymm0, 1 // V
    lea        edx, [edx + 16]
    jg         convertloop

    pop        edi
    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBTOUVROW_AVX2

__declspec(naked) __declspec(align(16))
void ARGBToUVRow_Unaligned_SSSE3(const uint8* src_argb0, int src_stride_argb,
                                 uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // src_argb
    mov        esi, [esp + 8 + 8]   // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // pix
    movdqa     xmm7, kARGBToU
    movdqa     xmm6, kARGBToV
    movdqa     xmm5, kAddUV128
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    movdqu     xmm4, [eax + esi]
    pavgb      xmm0, xmm4
    movdqu     xmm4, [eax + esi + 16]
    pavgb      xmm1, xmm4
    movdqu     xmm4, [eax + esi + 32]
    pavgb      xmm2, xmm4
    movdqu     xmm4, [eax + esi + 48]
    pavgb      xmm3, xmm4
    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

    // step 2 - convert to U and V
    // from here down is very similar to Y code except
    // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1
    paddb      xmm0, xmm5            // -> unsigned

    // step 3 - store 8 U and 8 V values
    sub        ecx, 16
    movlps     qword ptr [edx], xmm0 // U
    movhps     qword ptr [edx + edi], xmm0 // V
    lea        edx, [edx + 8]
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ARGBToUVJRow_Unaligned_SSSE3(const uint8* src_argb0, int src_stride_argb,
                                 uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // src_argb
    mov        esi, [esp + 8 + 8]   // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // pix
    movdqa     xmm7, kARGBToUJ
    movdqa     xmm6, kARGBToVJ
    movdqa     xmm5, kAddUVJ128
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    movdqu     xmm4, [eax + esi]
    pavgb      xmm0, xmm4
    movdqu     xmm4, [eax + esi + 16]
    pavgb      xmm1, xmm4
    movdqu     xmm4, [eax + esi + 32]
    pavgb      xmm2, xmm4
    movdqu     xmm4, [eax + esi + 48]
    pavgb      xmm3, xmm4
    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

    // step 2 - convert to U and V
    // from here down is very similar to Y code except
    // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    paddw      xmm0, xmm5            // +.5 rounding -> unsigned
    paddw      xmm1, xmm5
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1

    // step 3 - store 8 U and 8 V values
    sub        ecx, 16
    movlps     qword ptr [edx], xmm0 // U
    movhps     qword ptr [edx + edi], xmm0 // V
    lea        edx, [edx + 8]
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ARGBToUV444Row_SSSE3(const uint8* src_argb0,
                          uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]   // src_argb
    mov        edx, [esp + 4 + 8]   // dst_u
    mov        edi, [esp + 4 + 12]  // dst_v
    mov        ecx, [esp + 4 + 16]  // pix
    movdqa     xmm7, kARGBToU
    movdqa     xmm6, kARGBToV
    movdqa     xmm5, kAddUV128
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* convert to U and V */
    movdqa     xmm0, [eax]          // U
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm7
    pmaddubsw  xmm1, xmm7
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm3, xmm7
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psraw      xmm0, 8
    psraw      xmm2, 8
    packsswb   xmm0, xmm2
    paddb      xmm0, xmm5
    sub        ecx,  16
    movdqa     [edx], xmm0

    movdqa     xmm0, [eax]          // V
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm6
    pmaddubsw  xmm1, xmm6
    pmaddubsw  xmm2, xmm6
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psraw      xmm0, 8
    psraw      xmm2, 8
    packsswb   xmm0, xmm2
    paddb      xmm0, xmm5
    lea        eax,  [eax + 64]
    movdqa     [edx + edi], xmm0
    lea        edx,  [edx + 16]
    jg         convertloop

    pop        edi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ARGBToUV444Row_Unaligned_SSSE3(const uint8* src_argb0,
                                    uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]   // src_argb
    mov        edx, [esp + 4 + 8]   // dst_u
    mov        edi, [esp + 4 + 12]  // dst_v
    mov        ecx, [esp + 4 + 16]  // pix
    movdqa     xmm7, kARGBToU
    movdqa     xmm6, kARGBToV
    movdqa     xmm5, kAddUV128
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* convert to U and V */
    movdqu     xmm0, [eax]          // U
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm7
    pmaddubsw  xmm1, xmm7
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm3, xmm7
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psraw      xmm0, 8
    psraw      xmm2, 8
    packsswb   xmm0, xmm2
    paddb      xmm0, xmm5
    sub        ecx,  16
    movdqu     [edx], xmm0

    movdqu     xmm0, [eax]          // V
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm6
    pmaddubsw  xmm1, xmm6
    pmaddubsw  xmm2, xmm6
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psraw      xmm0, 8
    psraw      xmm2, 8
    packsswb   xmm0, xmm2
    paddb      xmm0, xmm5
    lea        eax,  [eax + 64]
    movdqu     [edx + edi], xmm0
    lea        edx,  [edx + 16]
    jg         convertloop

    pop        edi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ARGBToUV422Row_SSSE3(const uint8* src_argb0,
                          uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]   // src_argb
    mov        edx, [esp + 4 + 8]   // dst_u
    mov        edi, [esp + 4 + 12]  // dst_v
    mov        ecx, [esp + 4 + 16]  // pix
    movdqa     xmm7, kARGBToU
    movdqa     xmm6, kARGBToV
    movdqa     xmm5, kAddUV128
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]
    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

    // step 2 - convert to U and V
    // from here down is very similar to Y code except
    // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1
    paddb      xmm0, xmm5            // -> unsigned

    // step 3 - store 8 U and 8 V values
    sub        ecx, 16
    movlps     qword ptr [edx], xmm0 // U
    movhps     qword ptr [edx + edi], xmm0 // V
    lea        edx, [edx + 8]
    jg         convertloop

    pop        edi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ARGBToUV422Row_Unaligned_SSSE3(const uint8* src_argb0,
                                    uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]   // src_argb
    mov        edx, [esp + 4 + 8]   // dst_u
    mov        edi, [esp + 4 + 12]  // dst_v
    mov        ecx, [esp + 4 + 16]  // pix
    movdqa     xmm7, kARGBToU
    movdqa     xmm6, kARGBToV
    movdqa     xmm5, kAddUV128
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

    // step 2 - convert to U and V
    // from here down is very similar to Y code except
    // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1
    paddb      xmm0, xmm5            // -> unsigned

    // step 3 - store 8 U and 8 V values
    sub        ecx, 16
    movlps     qword ptr [edx], xmm0 // U
    movhps     qword ptr [edx + edi], xmm0 // V
    lea        edx, [edx + 8]
    jg         convertloop

    pop        edi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void BGRAToUVRow_SSSE3(const uint8* src_argb0, int src_stride_argb,
                       uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // src_argb
    mov        esi, [esp + 8 + 8]   // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // pix
    movdqa     xmm7, kBGRAToU
    movdqa     xmm6, kBGRAToV
    movdqa     xmm5, kAddUV128
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]
    pavgb      xmm0, [eax + esi]
    pavgb      xmm1, [eax + esi + 16]
    pavgb      xmm2, [eax + esi + 32]
    pavgb      xmm3, [eax + esi + 48]
    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

    // step 2 - convert to U and V
    // from here down is very similar to Y code except
    // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1
    paddb      xmm0, xmm5            // -> unsigned

    // step 3 - store 8 U and 8 V values
    sub        ecx, 16
    movlps     qword ptr [edx], xmm0 // U
    movhps     qword ptr [edx + edi], xmm0 // V
    lea        edx, [edx + 8]
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void BGRAToUVRow_Unaligned_SSSE3(const uint8* src_argb0, int src_stride_argb,
                                 uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // src_argb
    mov        esi, [esp + 8 + 8]   // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // pix
    movdqa     xmm7, kBGRAToU
    movdqa     xmm6, kBGRAToV
    movdqa     xmm5, kAddUV128
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    movdqu     xmm4, [eax + esi]
    pavgb      xmm0, xmm4
    movdqu     xmm4, [eax + esi + 16]
    pavgb      xmm1, xmm4
    movdqu     xmm4, [eax + esi + 32]
    pavgb      xmm2, xmm4
    movdqu     xmm4, [eax + esi + 48]
    pavgb      xmm3, xmm4
    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

    // step 2 - convert to U and V
    // from here down is very similar to Y code except
    // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1
    paddb      xmm0, xmm5            // -> unsigned

    // step 3 - store 8 U and 8 V values
    sub        ecx, 16
    movlps     qword ptr [edx], xmm0 // U
    movhps     qword ptr [edx + edi], xmm0 // V
    lea        edx, [edx + 8]
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ABGRToUVRow_SSSE3(const uint8* src_argb0, int src_stride_argb,
                       uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // src_argb
    mov        esi, [esp + 8 + 8]   // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // pix
    movdqa     xmm7, kABGRToU
    movdqa     xmm6, kABGRToV
    movdqa     xmm5, kAddUV128
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]
    pavgb      xmm0, [eax + esi]
    pavgb      xmm1, [eax + esi + 16]
    pavgb      xmm2, [eax + esi + 32]
    pavgb      xmm3, [eax + esi + 48]
    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

    // step 2 - convert to U and V
    // from here down is very similar to Y code except
    // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1
    paddb      xmm0, xmm5            // -> unsigned

    // step 3 - store 8 U and 8 V values
    sub        ecx, 16
    movlps     qword ptr [edx], xmm0 // U
    movhps     qword ptr [edx + edi], xmm0 // V
    lea        edx, [edx + 8]
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ABGRToUVRow_Unaligned_SSSE3(const uint8* src_argb0, int src_stride_argb,
                                 uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // src_argb
    mov        esi, [esp + 8 + 8]   // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // pix
    movdqa     xmm7, kABGRToU
    movdqa     xmm6, kABGRToV
    movdqa     xmm5, kAddUV128
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    movdqu     xmm4, [eax + esi]
    pavgb      xmm0, xmm4
    movdqu     xmm4, [eax + esi + 16]
    pavgb      xmm1, xmm4
    movdqu     xmm4, [eax + esi + 32]
    pavgb      xmm2, xmm4
    movdqu     xmm4, [eax + esi + 48]
    pavgb      xmm3, xmm4
    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

    // step 2 - convert to U and V
    // from here down is very similar to Y code except
    // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1
    paddb      xmm0, xmm5            // -> unsigned

    // step 3 - store 8 U and 8 V values
    sub        ecx, 16
    movlps     qword ptr [edx], xmm0 // U
    movhps     qword ptr [edx + edi], xmm0 // V
    lea        edx, [edx + 8]
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void RGBAToUVRow_SSSE3(const uint8* src_argb0, int src_stride_argb,
                       uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // src_argb
    mov        esi, [esp + 8 + 8]   // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // pix
    movdqa     xmm7, kRGBAToU
    movdqa     xmm6, kRGBAToV
    movdqa     xmm5, kAddUV128
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]
    pavgb      xmm0, [eax + esi]
    pavgb      xmm1, [eax + esi + 16]
    pavgb      xmm2, [eax + esi + 32]
    pavgb      xmm3, [eax + esi + 48]
    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

    // step 2 - convert to U and V
    // from here down is very similar to Y code except
    // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1
    paddb      xmm0, xmm5            // -> unsigned

    // step 3 - store 8 U and 8 V values
    sub        ecx, 16
    movlps     qword ptr [edx], xmm0 // U
    movhps     qword ptr [edx + edi], xmm0 // V
    lea        edx, [edx + 8]
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void RGBAToUVRow_Unaligned_SSSE3(const uint8* src_argb0, int src_stride_argb,
                                 uint8* dst_u, uint8* dst_v, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // src_argb
    mov        esi, [esp + 8 + 8]   // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // pix
    movdqa     xmm7, kRGBAToU
    movdqa     xmm6, kRGBAToV
    movdqa     xmm5, kAddUV128
    sub        edi, edx             // stride from u to v

    align      4
 convertloop:
    /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    movdqu     xmm4, [eax + esi]
    pavgb      xmm0, xmm4
    movdqu     xmm4, [eax + esi + 16]
    pavgb      xmm1, xmm4
    movdqu     xmm4, [eax + esi + 32]
    pavgb      xmm2, xmm4
    movdqu     xmm4, [eax + esi + 48]
    pavgb      xmm3, xmm4
    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

    // step 2 - convert to U and V
    // from here down is very similar to Y code except
    // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1
    paddb      xmm0, xmm5            // -> unsigned

    // step 3 - store 8 U and 8 V values
    sub        ecx, 16
    movlps     qword ptr [edx], xmm0 // U
    movhps     qword ptr [edx + edi], xmm0 // V
    lea        edx, [edx + 8]
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBTOYROW_SSSE3

#define YG 74 /* (int8)(1.164 * 64 + 0.5) */

#define UB 127 /* min(63,(int8)(2.018 * 64)) */
#define UG -25 /* (int8)(-0.391 * 64 - 0.5) */
#define UR 0

#define VB 0
#define VG -52 /* (int8)(-0.813 * 64 - 0.5) */
#define VR 102 /* (int8)(1.596 * 64 + 0.5) */

// Bias
#define BB UB * 128 + VB * 128
#define BG UG * 128 + VG * 128
#define BR UR * 128 + VR * 128

#ifdef HAS_I422TOARGBROW_AVX2

static const lvec8 kUVToB_AVX = {
  UB, VB, UB, VB, UB, VB, UB, VB, UB, VB, UB, VB, UB, VB, UB, VB,
  UB, VB, UB, VB, UB, VB, UB, VB, UB, VB, UB, VB, UB, VB, UB, VB
};
static const lvec8 kUVToR_AVX = {
  UR, VR, UR, VR, UR, VR, UR, VR, UR, VR, UR, VR, UR, VR, UR, VR,
  UR, VR, UR, VR, UR, VR, UR, VR, UR, VR, UR, VR, UR, VR, UR, VR
};
static const lvec8 kUVToG_AVX = {
  UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG,
  UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG
};
static const lvec16 kYToRgb_AVX = {
  YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG
};
static const lvec16 kYSub16_AVX = {
  16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
};
static const lvec16 kUVBiasB_AVX = {
  BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB
};
static const lvec16 kUVBiasG_AVX = {
  BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG
};
static const lvec16 kUVBiasR_AVX = {
  BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR
};

// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
__declspec(naked) __declspec(align(16))
void I422ToARGBRow_AVX2(const uint8* y_buf,
                         const uint8* u_buf,
                         const uint8* v_buf,
                         uint8* dst_argb,
                         int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // Y
    mov        esi, [esp + 8 + 8]   // U
    mov        edi, [esp + 8 + 12]  // V
    mov        edx, [esp + 8 + 16]  // argb
    mov        ecx, [esp + 8 + 20]  // width
    sub        edi, esi
    vpcmpeqb   ymm5, ymm5, ymm5     // generate 0xffffffffffffffff for alpha
    vpxor      ymm4, ymm4, ymm4

    align      4
 convertloop:
    vmovq      xmm0, qword ptr [esi]          //  U
    vmovq      xmm1, qword ptr [esi + edi]    //  V
    lea        esi,  [esi + 8]
    vpunpcklbw ymm0, ymm0, ymm1               // UV
    vpermq     ymm0, ymm0, 0xd8
    vpunpcklwd ymm0, ymm0, ymm0              // UVUV
    vpmaddubsw ymm2, ymm0, kUVToB_AVX        // scale B UV
    vpmaddubsw ymm1, ymm0, kUVToG_AVX        // scale G UV
    vpmaddubsw ymm0, ymm0, kUVToR_AVX        // scale R UV
    vpsubw     ymm2, ymm2, kUVBiasB_AVX      // unbias back to signed
    vpsubw     ymm1, ymm1, kUVBiasG_AVX
    vpsubw     ymm0, ymm0, kUVBiasR_AVX

    // Step 2: Find Y contribution to 16 R,G,B values
    vmovdqu    xmm3, [eax]                  // NOLINT
    lea        eax, [eax + 16]
    vpermq     ymm3, ymm3, 0xd8
    vpunpcklbw ymm3, ymm3, ymm4
    vpsubsw    ymm3, ymm3, kYSub16_AVX
    vpmullw    ymm3, ymm3, kYToRgb_AVX
    vpaddsw    ymm2, ymm2, ymm3           // B += Y
    vpaddsw    ymm1, ymm1, ymm3           // G += Y
    vpaddsw    ymm0, ymm0, ymm3           // R += Y
    vpsraw     ymm2, ymm2, 6
    vpsraw     ymm1, ymm1, 6
    vpsraw     ymm0, ymm0, 6
    vpackuswb  ymm2, ymm2, ymm2           // B
    vpackuswb  ymm1, ymm1, ymm1           // G
    vpackuswb  ymm0, ymm0, ymm0           // R

    // Step 3: Weave into ARGB
    vpunpcklbw ymm2, ymm2, ymm1           // BG
    vpermq     ymm2, ymm2, 0xd8
    vpunpcklbw ymm0, ymm0, ymm5           // RA
    vpermq     ymm0, ymm0, 0xd8
    vpunpcklwd ymm1, ymm2, ymm0           // BGRA first 8 pixels
    vpunpckhwd ymm2, ymm2, ymm0           // BGRA next 8 pixels
    vmovdqu    [edx], ymm1
    vmovdqu    [edx + 32], ymm2
    lea        edx,  [edx + 64]
    sub        ecx, 16
    jg         convertloop
    vzeroupper

    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_I422TOARGBROW_AVX2

#ifdef HAS_I422TOARGBROW_SSSE3

static const vec8 kUVToB = {
  UB, VB, UB, VB, UB, VB, UB, VB, UB, VB, UB, VB, UB, VB, UB, VB
};

static const vec8 kUVToR = {
  UR, VR, UR, VR, UR, VR, UR, VR, UR, VR, UR, VR, UR, VR, UR, VR
};

static const vec8 kUVToG = {
  UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG
};

static const vec8 kVUToB = {
  VB, UB, VB, UB, VB, UB, VB, UB, VB, UB, VB, UB, VB, UB, VB, UB,
};

static const vec8 kVUToR = {
  VR, UR, VR, UR, VR, UR, VR, UR, VR, UR, VR, UR, VR, UR, VR, UR,
};

static const vec8 kVUToG = {
  VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG,
};

static const vec16 kYToRgb = { YG, YG, YG, YG, YG, YG, YG, YG };
static const vec16 kYSub16 = { 16, 16, 16, 16, 16, 16, 16, 16 };
static const vec16 kUVBiasB = { BB, BB, BB, BB, BB, BB, BB, BB };
static const vec16 kUVBiasG = { BG, BG, BG, BG, BG, BG, BG, BG };
static const vec16 kUVBiasR = { BR, BR, BR, BR, BR, BR, BR, BR };

// TODO(fbarchard): Read that does half size on Y and treats 420 as 444.

// Read 8 UV from 444.
#define READYUV444 __asm {                                                     \
    __asm movq       xmm0, qword ptr [esi] /* U */                /* NOLINT */ \
    __asm movq       xmm1, qword ptr [esi + edi] /* V */          /* NOLINT */ \
    __asm lea        esi,  [esi + 8]                                           \
    __asm punpcklbw  xmm0, xmm1           /* UV */                             \
  }

// Read 4 UV from 422, upsample to 8 UV.
#define READYUV422 __asm {                                                     \
    __asm movd       xmm0, [esi]          /* U */                              \
    __asm movd       xmm1, [esi + edi]    /* V */                              \
    __asm lea        esi,  [esi + 4]                                           \
    __asm punpcklbw  xmm0, xmm1           /* UV */                             \
    __asm punpcklwd  xmm0, xmm0           /* UVUV (upsample) */                \
  }

// Read 2 UV from 411, upsample to 8 UV.
#define READYUV411 __asm {                                                     \
    __asm movzx      ebx, word ptr [esi]        /* U */           /* NOLINT */ \
    __asm movd       xmm0, ebx                                                 \
    __asm movzx      ebx, word ptr [esi + edi]  /* V */           /* NOLINT */ \
    __asm movd       xmm1, ebx                                                 \
    __asm lea        esi,  [esi + 2]                                           \
    __asm punpcklbw  xmm0, xmm1           /* UV */                             \
    __asm punpcklwd  xmm0, xmm0           /* UVUV (upsample) */                \
    __asm punpckldq  xmm0, xmm0           /* UVUV (upsample) */                \
  }

// Read 4 UV from NV12, upsample to 8 UV.
#define READNV12 __asm {                                                       \
    __asm movq       xmm0, qword ptr [esi] /* UV */               /* NOLINT */ \
    __asm lea        esi,  [esi + 8]                                           \
    __asm punpcklwd  xmm0, xmm0           /* UVUV (upsample) */                \
  }

// Convert 8 pixels: 8 UV and 8 Y.
#define YUVTORGB __asm {                                                       \
    /* Step 1: Find 4 UV contributions to 8 R,G,B values */                    \
    __asm movdqa     xmm1, xmm0                                                \
    __asm movdqa     xmm2, xmm0                                                \
    __asm pmaddubsw  xmm0, kUVToB        /* scale B UV */                      \
    __asm pmaddubsw  xmm1, kUVToG        /* scale G UV */                      \
    __asm pmaddubsw  xmm2, kUVToR        /* scale R UV */                      \
    __asm psubw      xmm0, kUVBiasB      /* unbias back to signed */           \
    __asm psubw      xmm1, kUVBiasG                                            \
    __asm psubw      xmm2, kUVBiasR                                            \
    /* Step 2: Find Y contribution to 8 R,G,B values */                        \
    __asm movq       xmm3, qword ptr [eax]                        /* NOLINT */ \
    __asm lea        eax, [eax + 8]                                            \
    __asm punpcklbw  xmm3, xmm4                                                \
    __asm psubsw     xmm3, kYSub16                                             \
    __asm pmullw     xmm3, kYToRgb                                             \
    __asm paddsw     xmm0, xmm3           /* B += Y */                         \
    __asm paddsw     xmm1, xmm3           /* G += Y */                         \
    __asm paddsw     xmm2, xmm3           /* R += Y */                         \
    __asm psraw      xmm0, 6                                                   \
    __asm psraw      xmm1, 6                                                   \
    __asm psraw      xmm2, 6                                                   \
    __asm packuswb   xmm0, xmm0           /* B */                              \
    __asm packuswb   xmm1, xmm1           /* G */                              \
    __asm packuswb   xmm2, xmm2           /* R */                              \
  }

// Convert 8 pixels: 8 VU and 8 Y.
#define YVUTORGB __asm {                                                       \
    /* Step 1: Find 4 UV contributions to 8 R,G,B values */                    \
    __asm movdqa     xmm1, xmm0                                                \
    __asm movdqa     xmm2, xmm0                                                \
    __asm pmaddubsw  xmm0, kVUToB        /* scale B UV */                      \
    __asm pmaddubsw  xmm1, kVUToG        /* scale G UV */                      \
    __asm pmaddubsw  xmm2, kVUToR        /* scale R UV */                      \
    __asm psubw      xmm0, kUVBiasB      /* unbias back to signed */           \
    __asm psubw      xmm1, kUVBiasG                                            \
    __asm psubw      xmm2, kUVBiasR                                            \
    /* Step 2: Find Y contribution to 8 R,G,B values */                        \
    __asm movq       xmm3, qword ptr [eax]                        /* NOLINT */ \
    __asm lea        eax, [eax + 8]                                            \
    __asm punpcklbw  xmm3, xmm4                                                \
    __asm psubsw     xmm3, kYSub16                                             \
    __asm pmullw     xmm3, kYToRgb                                             \
    __asm paddsw     xmm0, xmm3           /* B += Y */                         \
    __asm paddsw     xmm1, xmm3           /* G += Y */                         \
    __asm paddsw     xmm2, xmm3           /* R += Y */                         \
    __asm psraw      xmm0, 6                                                   \
    __asm psraw      xmm1, 6                                                   \
    __asm psraw      xmm2, 6                                                   \
    __asm packuswb   xmm0, xmm0           /* B */                              \
    __asm packuswb   xmm1, xmm1           /* G */                              \
    __asm packuswb   xmm2, xmm2           /* R */                              \
  }

// 8 pixels, dest aligned 16.
// 8 UV values, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) __declspec(align(16))
void I444ToARGBRow_SSSE3(const uint8* y_buf,
                         const uint8* u_buf,
                         const uint8* v_buf,
                         uint8* dst_argb,
                         int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // Y
    mov        esi, [esp + 8 + 8]   // U
    mov        edi, [esp + 8 + 12]  // V
    mov        edx, [esp + 8 + 16]  // argb
    mov        ecx, [esp + 8 + 20]  // width
    sub        edi, esi
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READYUV444
    YUVTORGB

    // Step 3: Weave into ARGB
    punpcklbw  xmm0, xmm1           // BG
    punpcklbw  xmm2, xmm5           // RA
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm2           // BGRA first 4 pixels
    punpckhwd  xmm1, xmm2           // BGRA next 4 pixels
    movdqa     [edx], xmm0
    movdqa     [edx + 16], xmm1
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

// 8 pixels, dest aligned 16.
// 4 UV values upsampled to 8 UV, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) __declspec(align(16))
void I422ToRGB24Row_SSSE3(const uint8* y_buf,
                          const uint8* u_buf,
                          const uint8* v_buf,
                          uint8* dst_rgb24,
                          int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // Y
    mov        esi, [esp + 8 + 8]   // U
    mov        edi, [esp + 8 + 12]  // V
    mov        edx, [esp + 8 + 16]  // rgb24
    mov        ecx, [esp + 8 + 20]  // width
    sub        edi, esi
    pxor       xmm4, xmm4
    movdqa     xmm5, kShuffleMaskARGBToRGB24_0
    movdqa     xmm6, kShuffleMaskARGBToRGB24

    align      4
 convertloop:
    READYUV422
    YUVTORGB

    // Step 3: Weave into RRGB
    punpcklbw  xmm0, xmm1           // BG
    punpcklbw  xmm2, xmm2           // RR
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm2           // BGRR first 4 pixels
    punpckhwd  xmm1, xmm2           // BGRR next 4 pixels
    pshufb     xmm0, xmm5           // Pack into first 8 and last 4 bytes.
    pshufb     xmm1, xmm6           // Pack into first 12 bytes.
    palignr    xmm1, xmm0, 12       // last 4 bytes of xmm0 + 12 from xmm1
    movq       qword ptr [edx], xmm0  // First 8 bytes
    movdqu     [edx + 8], xmm1      // Last 16 bytes. = 24 bytes, 8 RGB pixels.
    lea        edx,  [edx + 24]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

// 8 pixels, dest aligned 16.
// 4 UV values upsampled to 8 UV, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) __declspec(align(16))
void I422ToRAWRow_SSSE3(const uint8* y_buf,
                        const uint8* u_buf,
                        const uint8* v_buf,
                        uint8* dst_raw,
                        int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // Y
    mov        esi, [esp + 8 + 8]   // U
    mov        edi, [esp + 8 + 12]  // V
    mov        edx, [esp + 8 + 16]  // raw
    mov        ecx, [esp + 8 + 20]  // width
    sub        edi, esi
    pxor       xmm4, xmm4
    movdqa     xmm5, kShuffleMaskARGBToRAW_0
    movdqa     xmm6, kShuffleMaskARGBToRAW

    align      4
 convertloop:
    READYUV422
    YUVTORGB

    // Step 3: Weave into RRGB
    punpcklbw  xmm0, xmm1           // BG
    punpcklbw  xmm2, xmm2           // RR
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm2           // BGRR first 4 pixels
    punpckhwd  xmm1, xmm2           // BGRR next 4 pixels
    pshufb     xmm0, xmm5           // Pack into first 8 and last 4 bytes.
    pshufb     xmm1, xmm6           // Pack into first 12 bytes.
    palignr    xmm1, xmm0, 12       // last 4 bytes of xmm0 + 12 from xmm1
    movq       qword ptr [edx], xmm0  // First 8 bytes
    movdqu     [edx + 8], xmm1      // Last 16 bytes. = 24 bytes, 8 RGB pixels.
    lea        edx,  [edx + 24]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

// 8 pixels, dest unaligned.
// 4 UV values upsampled to 8 UV, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) __declspec(align(16))
void I422ToRGB565Row_SSSE3(const uint8* y_buf,
                           const uint8* u_buf,
                           const uint8* v_buf,
                           uint8* rgb565_buf,
                           int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // Y
    mov        esi, [esp + 8 + 8]   // U
    mov        edi, [esp + 8 + 12]  // V
    mov        edx, [esp + 8 + 16]  // rgb565
    mov        ecx, [esp + 8 + 20]  // width
    sub        edi, esi
    pxor       xmm4, xmm4
    pcmpeqb    xmm5, xmm5       // generate mask 0x0000001f
    psrld      xmm5, 27
    pcmpeqb    xmm6, xmm6       // generate mask 0x000007e0
    psrld      xmm6, 26
    pslld      xmm6, 5
    pcmpeqb    xmm7, xmm7       // generate mask 0xfffff800
    pslld      xmm7, 11

    align      4
 convertloop:
    READYUV422
    YUVTORGB

    // Step 3: Weave into RRGB
    punpcklbw  xmm0, xmm1           // BG
    punpcklbw  xmm2, xmm2           // RR
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm2           // BGRR first 4 pixels
    punpckhwd  xmm1, xmm2           // BGRR next 4 pixels

    // Step 3b: RRGB -> RGB565
    movdqa     xmm3, xmm0    // B  first 4 pixels of argb
    movdqa     xmm2, xmm0    // G
    pslld      xmm0, 8       // R
    psrld      xmm3, 3       // B
    psrld      xmm2, 5       // G
    psrad      xmm0, 16      // R
    pand       xmm3, xmm5    // B
    pand       xmm2, xmm6    // G
    pand       xmm0, xmm7    // R
    por        xmm3, xmm2    // BG
    por        xmm0, xmm3    // BGR
    movdqa     xmm3, xmm1    // B  next 4 pixels of argb
    movdqa     xmm2, xmm1    // G
    pslld      xmm1, 8       // R
    psrld      xmm3, 3       // B
    psrld      xmm2, 5       // G
    psrad      xmm1, 16      // R
    pand       xmm3, xmm5    // B
    pand       xmm2, xmm6    // G
    pand       xmm1, xmm7    // R
    por        xmm3, xmm2    // BG
    por        xmm1, xmm3    // BGR
    packssdw   xmm0, xmm1
    sub        ecx, 8
    movdqu     [edx], xmm0   // store 8 pixels of RGB565
    lea        edx, [edx + 16]
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

// 8 pixels, dest aligned 16.
// 4 UV values upsampled to 8 UV, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) __declspec(align(16))
void I422ToARGBRow_SSSE3(const uint8* y_buf,
                         const uint8* u_buf,
                         const uint8* v_buf,
                         uint8* dst_argb,
                         int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // Y
    mov        esi, [esp + 8 + 8]   // U
    mov        edi, [esp + 8 + 12]  // V
    mov        edx, [esp + 8 + 16]  // argb
    mov        ecx, [esp + 8 + 20]  // width
    sub        edi, esi
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READYUV422
    YUVTORGB

    // Step 3: Weave into ARGB
    punpcklbw  xmm0, xmm1           // BG
    punpcklbw  xmm2, xmm5           // RA
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm2           // BGRA first 4 pixels
    punpckhwd  xmm1, xmm2           // BGRA next 4 pixels
    movdqa     [edx], xmm0
    movdqa     [edx + 16], xmm1
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

// 8 pixels, dest aligned 16.
// 2 UV values upsampled to 8 UV, mixed with 8 Y producing 8 ARGB (32 bytes).
// Similar to I420 but duplicate UV once more.
__declspec(naked) __declspec(align(16))
void I411ToARGBRow_SSSE3(const uint8* y_buf,
                         const uint8* u_buf,
                         const uint8* v_buf,
                         uint8* dst_argb,
                         int width) {
  __asm {
    push       ebx
    push       esi
    push       edi
    mov        eax, [esp + 12 + 4]   // Y
    mov        esi, [esp + 12 + 8]   // U
    mov        edi, [esp + 12 + 12]  // V
    mov        edx, [esp + 12 + 16]  // argb
    mov        ecx, [esp + 12 + 20]  // width
    sub        edi, esi
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READYUV411  // modifies EBX
    YUVTORGB

    // Step 3: Weave into ARGB
    punpcklbw  xmm0, xmm1           // BG
    punpcklbw  xmm2, xmm5           // RA
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm2           // BGRA first 4 pixels
    punpckhwd  xmm1, xmm2           // BGRA next 4 pixels
    movdqa     [edx], xmm0
    movdqa     [edx + 16], xmm1
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    pop        ebx
    ret
  }
}

// 8 pixels, dest aligned 16.
// 4 UV values upsampled to 8 UV, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) __declspec(align(16))
void NV12ToARGBRow_SSSE3(const uint8* y_buf,
                         const uint8* uv_buf,
                         uint8* dst_argb,
                         int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // Y
    mov        esi, [esp + 4 + 8]   // UV
    mov        edx, [esp + 4 + 12]  // argb
    mov        ecx, [esp + 4 + 16]  // width
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READNV12
    YUVTORGB

    // Step 3: Weave into ARGB
    punpcklbw  xmm0, xmm1           // BG
    punpcklbw  xmm2, xmm5           // RA
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm2           // BGRA first 4 pixels
    punpckhwd  xmm1, xmm2           // BGRA next 4 pixels
    movdqa     [edx], xmm0
    movdqa     [edx + 16], xmm1
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        esi
    ret
  }
}

// 8 pixels, dest aligned 16.
// 4 UV values upsampled to 8 UV, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) __declspec(align(16))
void NV21ToARGBRow_SSSE3(const uint8* y_buf,
                         const uint8* uv_buf,
                         uint8* dst_argb,
                         int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // Y
    mov        esi, [esp + 4 + 8]   // VU
    mov        edx, [esp + 4 + 12]  // argb
    mov        ecx, [esp + 4 + 16]  // width
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READNV12
    YVUTORGB

    // Step 3: Weave into ARGB
    punpcklbw  xmm0, xmm1           // BG
    punpcklbw  xmm2, xmm5           // RA
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm2           // BGRA first 4 pixels
    punpckhwd  xmm1, xmm2           // BGRA next 4 pixels
    movdqa     [edx], xmm0
    movdqa     [edx + 16], xmm1
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        esi
    ret
  }
}

// 8 pixels, unaligned.
// 8 UV values, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) __declspec(align(16))
void I444ToARGBRow_Unaligned_SSSE3(const uint8* y_buf,
                                   const uint8* u_buf,
                                   const uint8* v_buf,
                                   uint8* dst_argb,
                                   int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // Y
    mov        esi, [esp + 8 + 8]   // U
    mov        edi, [esp + 8 + 12]  // V
    mov        edx, [esp + 8 + 16]  // argb
    mov        ecx, [esp + 8 + 20]  // width
    sub        edi, esi
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READYUV444
    YUVTORGB

    // Step 3: Weave into ARGB
    punpcklbw  xmm0, xmm1           // BG
    punpcklbw  xmm2, xmm5           // RA
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm2           // BGRA first 4 pixels
    punpckhwd  xmm1, xmm2           // BGRA next 4 pixels
    movdqu     [edx], xmm0
    movdqu     [edx + 16], xmm1
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

// 8 pixels, unaligned.
// 4 UV values upsampled to 8 UV, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) __declspec(align(16))
void I422ToARGBRow_Unaligned_SSSE3(const uint8* y_buf,
                                   const uint8* u_buf,
                                   const uint8* v_buf,
                                   uint8* dst_argb,
                                   int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // Y
    mov        esi, [esp + 8 + 8]   // U
    mov        edi, [esp + 8 + 12]  // V
    mov        edx, [esp + 8 + 16]  // argb
    mov        ecx, [esp + 8 + 20]  // width
    sub        edi, esi
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READYUV422
    YUVTORGB

    // Step 3: Weave into ARGB
    punpcklbw  xmm0, xmm1           // BG
    punpcklbw  xmm2, xmm5           // RA
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm2           // BGRA first 4 pixels
    punpckhwd  xmm1, xmm2           // BGRA next 4 pixels
    movdqu     [edx], xmm0
    movdqu     [edx + 16], xmm1
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

// 8 pixels, unaligned.
// 2 UV values upsampled to 8 UV, mixed with 8 Y producing 8 ARGB (32 bytes).
// Similar to I420 but duplicate UV once more.
__declspec(naked) __declspec(align(16))
void I411ToARGBRow_Unaligned_SSSE3(const uint8* y_buf,
                                   const uint8* u_buf,
                                   const uint8* v_buf,
                                   uint8* dst_argb,
                                   int width) {
  __asm {
    push       ebx
    push       esi
    push       edi
    mov        eax, [esp + 12 + 4]   // Y
    mov        esi, [esp + 12 + 8]   // U
    mov        edi, [esp + 12 + 12]  // V
    mov        edx, [esp + 12 + 16]  // argb
    mov        ecx, [esp + 12 + 20]  // width
    sub        edi, esi
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READYUV411  // modifies EBX
    YUVTORGB

    // Step 3: Weave into ARGB
    punpcklbw  xmm0, xmm1           // BG
    punpcklbw  xmm2, xmm5           // RA
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm2           // BGRA first 4 pixels
    punpckhwd  xmm1, xmm2           // BGRA next 4 pixels
    movdqu     [edx], xmm0
    movdqu     [edx + 16], xmm1
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    pop        ebx
    ret
  }
}

// 8 pixels, dest aligned 16.
// 4 UV values upsampled to 8 UV, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) __declspec(align(16))
void NV12ToARGBRow_Unaligned_SSSE3(const uint8* y_buf,
                                   const uint8* uv_buf,
                                   uint8* dst_argb,
                                   int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // Y
    mov        esi, [esp + 4 + 8]   // UV
    mov        edx, [esp + 4 + 12]  // argb
    mov        ecx, [esp + 4 + 16]  // width
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READNV12
    YUVTORGB

    // Step 3: Weave into ARGB
    punpcklbw  xmm0, xmm1           // BG
    punpcklbw  xmm2, xmm5           // RA
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm2           // BGRA first 4 pixels
    punpckhwd  xmm1, xmm2           // BGRA next 4 pixels
    movdqu     [edx], xmm0
    movdqu     [edx + 16], xmm1
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        esi
    ret
  }
}

// 8 pixels, dest aligned 16.
// 4 UV values upsampled to 8 UV, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) __declspec(align(16))
void NV21ToARGBRow_Unaligned_SSSE3(const uint8* y_buf,
                                   const uint8* uv_buf,
                                   uint8* dst_argb,
                                   int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // Y
    mov        esi, [esp + 4 + 8]   // VU
    mov        edx, [esp + 4 + 12]  // argb
    mov        ecx, [esp + 4 + 16]  // width
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READNV12
    YVUTORGB

    // Step 3: Weave into ARGB
    punpcklbw  xmm0, xmm1           // BG
    punpcklbw  xmm2, xmm5           // RA
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm2           // BGRA first 4 pixels
    punpckhwd  xmm1, xmm2           // BGRA next 4 pixels
    movdqu     [edx], xmm0
    movdqu     [edx + 16], xmm1
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void I422ToBGRARow_SSSE3(const uint8* y_buf,
                         const uint8* u_buf,
                         const uint8* v_buf,
                         uint8* dst_bgra,
                         int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // Y
    mov        esi, [esp + 8 + 8]   // U
    mov        edi, [esp + 8 + 12]  // V
    mov        edx, [esp + 8 + 16]  // bgra
    mov        ecx, [esp + 8 + 20]  // width
    sub        edi, esi
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READYUV422
    YUVTORGB

    // Step 3: Weave into BGRA
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    punpcklbw  xmm1, xmm0           // GB
    punpcklbw  xmm5, xmm2           // AR
    movdqa     xmm0, xmm5
    punpcklwd  xmm5, xmm1           // BGRA first 4 pixels
    punpckhwd  xmm0, xmm1           // BGRA next 4 pixels
    movdqa     [edx], xmm5
    movdqa     [edx + 16], xmm0
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void I422ToBGRARow_Unaligned_SSSE3(const uint8* y_buf,
                                   const uint8* u_buf,
                                   const uint8* v_buf,
                                   uint8* dst_bgra,
                                   int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // Y
    mov        esi, [esp + 8 + 8]   // U
    mov        edi, [esp + 8 + 12]  // V
    mov        edx, [esp + 8 + 16]  // bgra
    mov        ecx, [esp + 8 + 20]  // width
    sub        edi, esi
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READYUV422
    YUVTORGB

    // Step 3: Weave into BGRA
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    punpcklbw  xmm1, xmm0           // GB
    punpcklbw  xmm5, xmm2           // AR
    movdqa     xmm0, xmm5
    punpcklwd  xmm5, xmm1           // BGRA first 4 pixels
    punpckhwd  xmm0, xmm1           // BGRA next 4 pixels
    movdqu     [edx], xmm5
    movdqu     [edx + 16], xmm0
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void I422ToABGRRow_SSSE3(const uint8* y_buf,
                         const uint8* u_buf,
                         const uint8* v_buf,
                         uint8* dst_abgr,
                         int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // Y
    mov        esi, [esp + 8 + 8]   // U
    mov        edi, [esp + 8 + 12]  // V
    mov        edx, [esp + 8 + 16]  // abgr
    mov        ecx, [esp + 8 + 20]  // width
    sub        edi, esi
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READYUV422
    YUVTORGB

    // Step 3: Weave into ARGB
    punpcklbw  xmm2, xmm1           // RG
    punpcklbw  xmm0, xmm5           // BA
    movdqa     xmm1, xmm2
    punpcklwd  xmm2, xmm0           // RGBA first 4 pixels
    punpckhwd  xmm1, xmm0           // RGBA next 4 pixels
    movdqa     [edx], xmm2
    movdqa     [edx + 16], xmm1
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void I422ToABGRRow_Unaligned_SSSE3(const uint8* y_buf,
                                   const uint8* u_buf,
                                   const uint8* v_buf,
                                   uint8* dst_abgr,
                                   int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // Y
    mov        esi, [esp + 8 + 8]   // U
    mov        edi, [esp + 8 + 12]  // V
    mov        edx, [esp + 8 + 16]  // abgr
    mov        ecx, [esp + 8 + 20]  // width
    sub        edi, esi
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READYUV422
    YUVTORGB

    // Step 3: Weave into ARGB
    punpcklbw  xmm2, xmm1           // RG
    punpcklbw  xmm0, xmm5           // BA
    movdqa     xmm1, xmm2
    punpcklwd  xmm2, xmm0           // RGBA first 4 pixels
    punpckhwd  xmm1, xmm0           // RGBA next 4 pixels
    movdqu     [edx], xmm2
    movdqu     [edx + 16], xmm1
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void I422ToRGBARow_SSSE3(const uint8* y_buf,
                         const uint8* u_buf,
                         const uint8* v_buf,
                         uint8* dst_rgba,
                         int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // Y
    mov        esi, [esp + 8 + 8]   // U
    mov        edi, [esp + 8 + 12]  // V
    mov        edx, [esp + 8 + 16]  // rgba
    mov        ecx, [esp + 8 + 20]  // width
    sub        edi, esi
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READYUV422
    YUVTORGB

    // Step 3: Weave into RGBA
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    punpcklbw  xmm1, xmm2           // GR
    punpcklbw  xmm5, xmm0           // AB
    movdqa     xmm0, xmm5
    punpcklwd  xmm5, xmm1           // RGBA first 4 pixels
    punpckhwd  xmm0, xmm1           // RGBA next 4 pixels
    movdqa     [edx], xmm5
    movdqa     [edx + 16], xmm0
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void I422ToRGBARow_Unaligned_SSSE3(const uint8* y_buf,
                                   const uint8* u_buf,
                                   const uint8* v_buf,
                                   uint8* dst_rgba,
                                   int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // Y
    mov        esi, [esp + 8 + 8]   // U
    mov        edi, [esp + 8 + 12]  // V
    mov        edx, [esp + 8 + 16]  // rgba
    mov        ecx, [esp + 8 + 20]  // width
    sub        edi, esi
    pxor       xmm4, xmm4

    align      4
 convertloop:
    READYUV422
    YUVTORGB

    // Step 3: Weave into RGBA
    pcmpeqb    xmm5, xmm5           // generate 0xffffffff for alpha
    punpcklbw  xmm1, xmm2           // GR
    punpcklbw  xmm5, xmm0           // AB
    movdqa     xmm0, xmm5
    punpcklwd  xmm5, xmm1           // RGBA first 4 pixels
    punpckhwd  xmm0, xmm1           // RGBA next 4 pixels
    movdqu     [edx], xmm5
    movdqu     [edx + 16], xmm0
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

#endif  // HAS_I422TOARGBROW_SSSE3

#ifdef HAS_YTOARGBROW_SSE2
__declspec(naked) __declspec(align(16))
void YToARGBRow_SSE2(const uint8* y_buf,
                     uint8* rgb_buf,
                     int width) {
  __asm {
    pxor       xmm5, xmm5
    pcmpeqb    xmm4, xmm4           // generate mask 0xff000000
    pslld      xmm4, 24
    mov        eax, 0x00100010
    movd       xmm3, eax
    pshufd     xmm3, xmm3, 0
    mov        eax, 0x004a004a       // 74
    movd       xmm2, eax
    pshufd     xmm2, xmm2,0
    mov        eax, [esp + 4]       // Y
    mov        edx, [esp + 8]       // rgb
    mov        ecx, [esp + 12]      // width

    align      4
 convertloop:
    // Step 1: Scale Y contribution to 8 G values. G = (y - 16) * 1.164
    movq       xmm0, qword ptr [eax]
    lea        eax, [eax + 8]
    punpcklbw  xmm0, xmm5           // 0.Y
    psubusw    xmm0, xmm3
    pmullw     xmm0, xmm2
    psrlw      xmm0, 6
    packuswb   xmm0, xmm0           // G

    // Step 2: Weave into ARGB
    punpcklbw  xmm0, xmm0           // GG
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm0           // BGRA first 4 pixels
    punpckhwd  xmm1, xmm1           // BGRA next 4 pixels
    por        xmm0, xmm4
    por        xmm1, xmm4
    movdqa     [edx], xmm0
    movdqa     [edx + 16], xmm1
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop

    ret
  }
}
#endif  // HAS_YTOARGBROW_SSE2

#ifdef HAS_MIRRORROW_SSSE3
// Shuffle table for reversing the bytes.
static const uvec8 kShuffleMirror = {
  15u, 14u, 13u, 12u, 11u, 10u, 9u, 8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u
};

__declspec(naked) __declspec(align(16))
void MirrorRow_SSSE3(const uint8* src, uint8* dst, int width) {
  __asm {
    mov       eax, [esp + 4]   // src
    mov       edx, [esp + 8]   // dst
    mov       ecx, [esp + 12]  // width
    movdqa    xmm5, kShuffleMirror
    lea       eax, [eax - 16]

    align      4
 convertloop:
    movdqa    xmm0, [eax + ecx]
    pshufb    xmm0, xmm5
    sub       ecx, 16
    movdqa    [edx], xmm0
    lea       edx, [edx + 16]
    jg        convertloop
    ret
  }
}
#endif  // HAS_MIRRORROW_SSSE3

#ifdef HAS_MIRRORROW_AVX2
// Shuffle table for reversing the bytes.
static const ulvec8 kShuffleMirror_AVX2 = {
  15u, 14u, 13u, 12u, 11u, 10u, 9u, 8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u,
  15u, 14u, 13u, 12u, 11u, 10u, 9u, 8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u
};

__declspec(naked) __declspec(align(16))
void MirrorRow_AVX2(const uint8* src, uint8* dst, int width) {
  __asm {
    mov       eax, [esp + 4]   // src
    mov       edx, [esp + 8]   // dst
    mov       ecx, [esp + 12]  // width
    vmovdqa   ymm5, kShuffleMirror_AVX2
    lea       eax, [eax - 32]

    align      4
 convertloop:
    vmovdqu   ymm0, [eax + ecx]
    vpshufb   ymm0, ymm0, ymm5
    vpermq    ymm0, ymm0, 0x4e  // swap high and low halfs
    sub       ecx, 32
    vmovdqu   [edx], ymm0
    lea       edx, [edx + 32]
    jg        convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_MIRRORROW_AVX2

#ifdef HAS_MIRRORROW_SSE2
// SSE2 version has movdqu so it can be used on unaligned buffers when SSSE3
// version can not.
__declspec(naked) __declspec(align(16))
void MirrorRow_SSE2(const uint8* src, uint8* dst, int width) {
  __asm {
    mov       eax, [esp + 4]   // src
    mov       edx, [esp + 8]   // dst
    mov       ecx, [esp + 12]  // width
    lea       eax, [eax - 16]

    align      4
 convertloop:
    movdqu    xmm0, [eax + ecx]
    movdqa    xmm1, xmm0        // swap bytes
    psllw     xmm0, 8
    psrlw     xmm1, 8
    por       xmm0, xmm1
    pshuflw   xmm0, xmm0, 0x1b  // swap words
    pshufhw   xmm0, xmm0, 0x1b
    pshufd    xmm0, xmm0, 0x4e  // swap qwords
    sub       ecx, 16
    movdqu    [edx], xmm0
    lea       edx, [edx + 16]
    jg        convertloop
    ret
  }
}
#endif  // HAS_MIRRORROW_SSE2

#ifdef HAS_MIRRORROW_UV_SSSE3
// Shuffle table for reversing the bytes of UV channels.
static const uvec8 kShuffleMirrorUV = {
  14u, 12u, 10u, 8u, 6u, 4u, 2u, 0u, 15u, 13u, 11u, 9u, 7u, 5u, 3u, 1u
};

__declspec(naked) __declspec(align(16))
void MirrorUVRow_SSSE3(const uint8* src, uint8* dst_u, uint8* dst_v,
                       int width) {
  __asm {
    push      edi
    mov       eax, [esp + 4 + 4]   // src
    mov       edx, [esp + 4 + 8]   // dst_u
    mov       edi, [esp + 4 + 12]  // dst_v
    mov       ecx, [esp + 4 + 16]  // width
    movdqa    xmm1, kShuffleMirrorUV
    lea       eax, [eax + ecx * 2 - 16]
    sub       edi, edx

    align      4
 convertloop:
    movdqa    xmm0, [eax]
    lea       eax, [eax - 16]
    pshufb    xmm0, xmm1
    sub       ecx, 8
    movlpd    qword ptr [edx], xmm0
    movhpd    qword ptr [edx + edi], xmm0
    lea       edx, [edx + 8]
    jg        convertloop

    pop       edi
    ret
  }
}
#endif  // HAS_MIRRORROW_UV_SSSE3

#ifdef HAS_ARGBMIRRORROW_SSSE3
// Shuffle table for reversing the bytes.
static const uvec8 kARGBShuffleMirror = {
  12u, 13u, 14u, 15u, 8u, 9u, 10u, 11u, 4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u
};

__declspec(naked) __declspec(align(16))
void ARGBMirrorRow_SSSE3(const uint8* src, uint8* dst, int width) {
  __asm {
    mov       eax, [esp + 4]   // src
    mov       edx, [esp + 8]   // dst
    mov       ecx, [esp + 12]  // width
    lea       eax, [eax - 16 + ecx * 4]  // last 4 pixels.
    movdqa    xmm5, kARGBShuffleMirror

    align      4
 convertloop:
    movdqa    xmm0, [eax]
    lea       eax, [eax - 16]
    pshufb    xmm0, xmm5
    sub       ecx, 4
    movdqa    [edx], xmm0
    lea       edx, [edx + 16]
    jg        convertloop
    ret
  }
}
#endif  // HAS_ARGBMIRRORROW_SSSE3

#ifdef HAS_ARGBMIRRORROW_AVX2
// Shuffle table for reversing the bytes.
static const ulvec32 kARGBShuffleMirror_AVX2 = {
  7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u
};

__declspec(naked) __declspec(align(16))
void ARGBMirrorRow_AVX2(const uint8* src, uint8* dst, int width) {
  __asm {
    mov       eax, [esp + 4]   // src
    mov       edx, [esp + 8]   // dst
    mov       ecx, [esp + 12]  // width
    lea       eax, [eax - 32]
    vmovdqa   ymm5, kARGBShuffleMirror_AVX2

    align      4
 convertloop:
    vpermd    ymm0, ymm5, [eax + ecx * 4]  // permute dword order
    sub       ecx, 8
    vmovdqu   [edx], ymm0
    lea       edx, [edx + 32]
    jg        convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBMIRRORROW_AVX2

#ifdef HAS_SPLITUVROW_SSE2
__declspec(naked) __declspec(align(16))
void SplitUVRow_SSE2(const uint8* src_uv, uint8* dst_u, uint8* dst_v, int pix) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]    // src_uv
    mov        edx, [esp + 4 + 8]    // dst_u
    mov        edi, [esp + 4 + 12]   // dst_v
    mov        ecx, [esp + 4 + 16]   // pix
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

    align      4
  convertloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    movdqa     xmm2, xmm0
    movdqa     xmm3, xmm1
    pand       xmm0, xmm5   // even bytes
    pand       xmm1, xmm5
    packuswb   xmm0, xmm1
    psrlw      xmm2, 8      // odd bytes
    psrlw      xmm3, 8
    packuswb   xmm2, xmm3
    movdqa     [edx], xmm0
    movdqa     [edx + edi], xmm2
    lea        edx, [edx + 16]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void SplitUVRow_Unaligned_SSE2(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                               int pix) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]    // src_uv
    mov        edx, [esp + 4 + 8]    // dst_u
    mov        edi, [esp + 4 + 12]   // dst_v
    mov        ecx, [esp + 4 + 16]   // pix
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

    align      4
  convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    movdqa     xmm2, xmm0
    movdqa     xmm3, xmm1
    pand       xmm0, xmm5   // even bytes
    pand       xmm1, xmm5
    packuswb   xmm0, xmm1
    psrlw      xmm2, 8      // odd bytes
    psrlw      xmm3, 8
    packuswb   xmm2, xmm3
    movdqu     [edx], xmm0
    movdqu     [edx + edi], xmm2
    lea        edx, [edx + 16]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    ret
  }
}
#endif  // HAS_SPLITUVROW_SSE2

#ifdef HAS_SPLITUVROW_AVX2
__declspec(naked) __declspec(align(16))
void SplitUVRow_AVX2(const uint8* src_uv, uint8* dst_u, uint8* dst_v, int pix) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]    // src_uv
    mov        edx, [esp + 4 + 8]    // dst_u
    mov        edi, [esp + 4 + 12]   // dst_v
    mov        ecx, [esp + 4 + 16]   // pix
    vpcmpeqb   ymm5, ymm5, ymm5      // generate mask 0x00ff00ff
    vpsrlw     ymm5, ymm5, 8
    sub        edi, edx

    align      4
  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    lea        eax,  [eax + 64]
    vpsrlw     ymm2, ymm0, 8      // odd bytes
    vpsrlw     ymm3, ymm1, 8
    vpand      ymm0, ymm0, ymm5   // even bytes
    vpand      ymm1, ymm1, ymm5
    vpackuswb  ymm0, ymm0, ymm1
    vpackuswb  ymm2, ymm2, ymm3
    vpermq     ymm0, ymm0, 0xd8
    vpermq     ymm2, ymm2, 0xd8
    vmovdqu    [edx], ymm0
    vmovdqu    [edx + edi], ymm2
    lea        edx, [edx + 32]
    sub        ecx, 32
    jg         convertloop

    pop        edi
    vzeroupper
    ret
  }
}
#endif  // HAS_SPLITUVROW_AVX2

#ifdef HAS_MERGEUVROW_SSE2
__declspec(naked) __declspec(align(16))
void MergeUVRow_SSE2(const uint8* src_u, const uint8* src_v, uint8* dst_uv,
                     int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]    // src_u
    mov        edx, [esp + 4 + 8]    // src_v
    mov        edi, [esp + 4 + 12]   // dst_uv
    mov        ecx, [esp + 4 + 16]   // width
    sub        edx, eax

    align      4
  convertloop:
    movdqa     xmm0, [eax]      // read 16 U's
    movdqa     xmm1, [eax + edx]  // and 16 V's
    lea        eax,  [eax + 16]
    movdqa     xmm2, xmm0
    punpcklbw  xmm0, xmm1       // first 8 UV pairs
    punpckhbw  xmm2, xmm1       // next 8 UV pairs
    movdqa     [edi], xmm0
    movdqa     [edi + 16], xmm2
    lea        edi, [edi + 32]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void MergeUVRow_Unaligned_SSE2(const uint8* src_u, const uint8* src_v,
                               uint8* dst_uv, int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]    // src_u
    mov        edx, [esp + 4 + 8]    // src_v
    mov        edi, [esp + 4 + 12]   // dst_uv
    mov        ecx, [esp + 4 + 16]   // width
    sub        edx, eax

    align      4
  convertloop:
    movdqu     xmm0, [eax]      // read 16 U's
    movdqu     xmm1, [eax + edx]  // and 16 V's
    lea        eax,  [eax + 16]
    movdqa     xmm2, xmm0
    punpcklbw  xmm0, xmm1       // first 8 UV pairs
    punpckhbw  xmm2, xmm1       // next 8 UV pairs
    movdqu     [edi], xmm0
    movdqu     [edi + 16], xmm2
    lea        edi, [edi + 32]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    ret
  }
}
#endif  //  HAS_MERGEUVROW_SSE2

#ifdef HAS_MERGEUVROW_AVX2
__declspec(naked) __declspec(align(16))
void MergeUVRow_AVX2(const uint8* src_u, const uint8* src_v, uint8* dst_uv,
                     int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]    // src_u
    mov        edx, [esp + 4 + 8]    // src_v
    mov        edi, [esp + 4 + 12]   // dst_uv
    mov        ecx, [esp + 4 + 16]   // width
    sub        edx, eax

    align      4
  convertloop:
    vmovdqu    ymm0, [eax]           // read 32 U's
    vmovdqu    ymm1, [eax + edx]     // and 32 V's
    lea        eax,  [eax + 32]
    vpunpcklbw ymm2, ymm0, ymm1      // low 16 UV pairs. mutated qqword 0,2
    vpunpckhbw ymm0, ymm0, ymm1      // high 16 UV pairs. mutated qqword 1,3
    vperm2i128 ymm1, ymm2, ymm0, 0x20  // low 128 of ymm2 and low 128 of ymm0
    vperm2i128 ymm2, ymm2, ymm0, 0x31  // high 128 of ymm2 and high 128 of ymm0
    vmovdqu    [edi], ymm1
    vmovdqu    [edi + 32], ymm2
    lea        edi, [edi + 64]
    sub        ecx, 32
    jg         convertloop

    pop        edi
    vzeroupper
    ret
  }
}
#endif  //  HAS_MERGEUVROW_AVX2

#ifdef HAS_COPYROW_SSE2
// CopyRow copys 'count' bytes using a 16 byte load/store, 32 bytes at time.
__declspec(naked) __declspec(align(16))
void CopyRow_SSE2(const uint8* src, uint8* dst, int count) {
  __asm {
    mov        eax, [esp + 4]   // src
    mov        edx, [esp + 8]   // dst
    mov        ecx, [esp + 12]  // count

    align      4
  convertloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax, [eax + 32]
    movdqa     [edx], xmm0
    movdqa     [edx + 16], xmm1
    lea        edx, [edx + 32]
    sub        ecx, 32
    jg         convertloop
    ret
  }
}
#endif  // HAS_COPYROW_SSE2

// Unaligned Multiple of 1.
__declspec(naked) __declspec(align(16))
void CopyRow_ERMS(const uint8* src, uint8* dst, int count) {
  __asm {
    mov        eax, esi
    mov        edx, edi
    mov        esi, [esp + 4]   // src
    mov        edi, [esp + 8]   // dst
    mov        ecx, [esp + 12]  // count
    rep movsb
    mov        edi, edx
    mov        esi, eax
    ret
  }
}

#ifdef HAS_COPYROW_X86
__declspec(naked) __declspec(align(16))
void CopyRow_X86(const uint8* src, uint8* dst, int count) {
  __asm {
    mov        eax, esi
    mov        edx, edi
    mov        esi, [esp + 4]   // src
    mov        edi, [esp + 8]   // dst
    mov        ecx, [esp + 12]  // count
    shr        ecx, 2
    rep movsd
    mov        edi, edx
    mov        esi, eax
    ret
  }
}
#endif  // HAS_COPYROW_X86

#ifdef HAS_ARGBCOPYALPHAROW_SSE2
// width in pixels
__declspec(naked) __declspec(align(16))
void ARGBCopyAlphaRow_SSE2(const uint8* src, uint8* dst, int width) {
  __asm {
    mov        eax, [esp + 4]   // src
    mov        edx, [esp + 8]   // dst
    mov        ecx, [esp + 12]  // count
    pcmpeqb    xmm0, xmm0       // generate mask 0xff000000
    pslld      xmm0, 24
    pcmpeqb    xmm1, xmm1       // generate mask 0x00ffffff
    psrld      xmm1, 8

    align      4
  convertloop:
    movdqa     xmm2, [eax]
    movdqa     xmm3, [eax + 16]
    lea        eax, [eax + 32]
    movdqa     xmm4, [edx]
    movdqa     xmm5, [edx + 16]
    pand       xmm2, xmm0
    pand       xmm3, xmm0
    pand       xmm4, xmm1
    pand       xmm5, xmm1
    por        xmm2, xmm4
    por        xmm3, xmm5
    movdqa     [edx], xmm2
    movdqa     [edx + 16], xmm3
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop

    ret
  }
}
#endif  // HAS_ARGBCOPYALPHAROW_SSE2

#ifdef HAS_ARGBCOPYALPHAROW_AVX2
// width in pixels
__declspec(naked) __declspec(align(16))
void ARGBCopyAlphaRow_AVX2(const uint8* src, uint8* dst, int width) {
  __asm {
    mov        eax, [esp + 4]   // src
    mov        edx, [esp + 8]   // dst
    mov        ecx, [esp + 12]  // count
    vpcmpeqb   ymm0, ymm0, ymm0
    vpsrld     ymm0, ymm0, 8    // generate mask 0x00ffffff

    align      4
  convertloop:
    vmovdqu    ymm1, [eax]
    vmovdqu    ymm2, [eax + 32]
    lea        eax, [eax + 64]
    vpblendvb  ymm1, ymm1, [edx], ymm0
    vpblendvb  ymm2, ymm2, [edx + 32], ymm0
    vmovdqu    [edx], ymm1
    vmovdqu    [edx + 32], ymm2
    lea        edx, [edx + 64]
    sub        ecx, 16
    jg         convertloop

    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBCOPYALPHAROW_AVX2

#ifdef HAS_ARGBCOPYYTOALPHAROW_SSE2
// width in pixels
__declspec(naked) __declspec(align(16))
void ARGBCopyYToAlphaRow_SSE2(const uint8* src, uint8* dst, int width) {
  __asm {
    mov        eax, [esp + 4]   // src
    mov        edx, [esp + 8]   // dst
    mov        ecx, [esp + 12]  // count
    pcmpeqb    xmm0, xmm0       // generate mask 0xff000000
    pslld      xmm0, 24
    pcmpeqb    xmm1, xmm1       // generate mask 0x00ffffff
    psrld      xmm1, 8

    align      4
  convertloop:
    movq       xmm2, qword ptr [eax]  // 8 Y's
    lea        eax, [eax + 8]
    punpcklbw  xmm2, xmm2
    punpckhwd  xmm3, xmm2
    punpcklwd  xmm2, xmm2
    movdqa     xmm4, [edx]
    movdqa     xmm5, [edx + 16]
    pand       xmm2, xmm0
    pand       xmm3, xmm0
    pand       xmm4, xmm1
    pand       xmm5, xmm1
    por        xmm2, xmm4
    por        xmm3, xmm5
    movdqa     [edx], xmm2
    movdqa     [edx + 16], xmm3
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop

    ret
  }
}
#endif  // HAS_ARGBCOPYYTOALPHAROW_SSE2

#ifdef HAS_ARGBCOPYYTOALPHAROW_AVX2
// width in pixels
__declspec(naked) __declspec(align(16))
void ARGBCopyYToAlphaRow_AVX2(const uint8* src, uint8* dst, int width) {
  __asm {
    mov        eax, [esp + 4]   // src
    mov        edx, [esp + 8]   // dst
    mov        ecx, [esp + 12]  // count
    vpcmpeqb   ymm0, ymm0, ymm0
    vpsrld     ymm0, ymm0, 8    // generate mask 0x00ffffff

    align      4
  convertloop:
    vpmovzxbd  ymm1, qword ptr [eax]
    vpmovzxbd  ymm2, qword ptr [eax + 8]
    lea        eax, [eax + 16]
    vpslld     ymm1, ymm1, 24
    vpslld     ymm2, ymm2, 24
    vpblendvb  ymm1, ymm1, [edx], ymm0
    vpblendvb  ymm2, ymm2, [edx + 32], ymm0
    vmovdqu    [edx], ymm1
    vmovdqu    [edx + 32], ymm2
    lea        edx, [edx + 64]
    sub        ecx, 16
    jg         convertloop

    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBCOPYYTOALPHAROW_AVX2

#ifdef HAS_SETROW_X86
// SetRow8 writes 'count' bytes using a 32 bit value repeated.
__declspec(naked) __declspec(align(16))
void SetRow_X86(uint8* dst, uint32 v32, int count) {
  __asm {
    mov        edx, edi
    mov        edi, [esp + 4]   // dst
    mov        eax, [esp + 8]   // v32
    mov        ecx, [esp + 12]  // count
    shr        ecx, 2
    rep stosd
    mov        edi, edx
    ret
  }
}

// SetRow32 writes 'count' words using a 32 bit value repeated.
__declspec(naked) __declspec(align(16))
void ARGBSetRows_X86(uint8* dst, uint32 v32, int width,
                   int dst_stride, int height) {
  __asm {
    push       esi
    push       edi
    push       ebp
    mov        edi, [esp + 12 + 4]   // dst
    mov        eax, [esp + 12 + 8]   // v32
    mov        ebp, [esp + 12 + 12]  // width
    mov        edx, [esp + 12 + 16]  // dst_stride
    mov        esi, [esp + 12 + 20]  // height
    lea        ecx, [ebp * 4]
    sub        edx, ecx             // stride - width * 4

    align      4
  convertloop:
    mov        ecx, ebp
    rep stosd
    add        edi, edx
    sub        esi, 1
    jg         convertloop

    pop        ebp
    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_SETROW_X86

#ifdef HAS_YUY2TOYROW_AVX2
__declspec(naked) __declspec(align(16))
void YUY2ToYRow_AVX2(const uint8* src_yuy2,
                     uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]    // src_yuy2
    mov        edx, [esp + 8]    // dst_y
    mov        ecx, [esp + 12]   // pix
    vpcmpeqb   ymm5, ymm5, ymm5  // generate mask 0x00ff00ff
    vpsrlw     ymm5, ymm5, 8

    align      4
  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    lea        eax,  [eax + 64]
    vpand      ymm0, ymm0, ymm5   // even bytes are Y
    vpand      ymm1, ymm1, ymm5
    vpackuswb  ymm0, ymm0, ymm1   // mutates.
    vpermq     ymm0, ymm0, 0xd8
    sub        ecx, 32
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    jg         convertloop
    vzeroupper
    ret
  }
}

__declspec(naked) __declspec(align(16))
void YUY2ToUVRow_AVX2(const uint8* src_yuy2, int stride_yuy2,
                      uint8* dst_u, uint8* dst_v, int pix) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]    // src_yuy2
    mov        esi, [esp + 8 + 8]    // stride_yuy2
    mov        edx, [esp + 8 + 12]   // dst_u
    mov        edi, [esp + 8 + 16]   // dst_v
    mov        ecx, [esp + 8 + 20]   // pix
    vpcmpeqb   ymm5, ymm5, ymm5      // generate mask 0x00ff00ff
    vpsrlw     ymm5, ymm5, 8
    sub        edi, edx

    align      4
  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    vpavgb     ymm0, ymm0, [eax + esi]
    vpavgb     ymm1, ymm1, [eax + esi + 32]
    lea        eax,  [eax + 64]
    vpsrlw     ymm0, ymm0, 8      // YUYV -> UVUV
    vpsrlw     ymm1, ymm1, 8
    vpackuswb  ymm0, ymm0, ymm1   // mutates.
    vpermq     ymm0, ymm0, 0xd8
    vpand      ymm1, ymm0, ymm5  // U
    vpsrlw     ymm0, ymm0, 8     // V
    vpackuswb  ymm1, ymm1, ymm1  // mutates.
    vpackuswb  ymm0, ymm0, ymm0  // mutates.
    vpermq     ymm1, ymm1, 0xd8
    vpermq     ymm0, ymm0, 0xd8
    vextractf128 [edx], ymm1, 0  // U
    vextractf128 [edx + edi], ymm0, 0 // V
    lea        edx, [edx + 16]
    sub        ecx, 32
    jg         convertloop

    pop        edi
    pop        esi
    vzeroupper
    ret
  }
}

__declspec(naked) __declspec(align(16))
void YUY2ToUV422Row_AVX2(const uint8* src_yuy2,
                         uint8* dst_u, uint8* dst_v, int pix) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]    // src_yuy2
    mov        edx, [esp + 4 + 8]    // dst_u
    mov        edi, [esp + 4 + 12]   // dst_v
    mov        ecx, [esp + 4 + 16]   // pix
    vpcmpeqb   ymm5, ymm5, ymm5      // generate mask 0x00ff00ff
    vpsrlw     ymm5, ymm5, 8
    sub        edi, edx

    align      4
  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    lea        eax,  [eax + 64]
    vpsrlw     ymm0, ymm0, 8      // YUYV -> UVUV
    vpsrlw     ymm1, ymm1, 8
    vpackuswb  ymm0, ymm0, ymm1   // mutates.
    vpermq     ymm0, ymm0, 0xd8
    vpand      ymm1, ymm0, ymm5  // U
    vpsrlw     ymm0, ymm0, 8     // V
    vpackuswb  ymm1, ymm1, ymm1  // mutates.
    vpackuswb  ymm0, ymm0, ymm0  // mutates.
    vpermq     ymm1, ymm1, 0xd8
    vpermq     ymm0, ymm0, 0xd8
    vextractf128 [edx], ymm1, 0  // U
    vextractf128 [edx + edi], ymm0, 0 // V
    lea        edx, [edx + 16]
    sub        ecx, 32
    jg         convertloop

    pop        edi
    vzeroupper
    ret
  }
}

__declspec(naked) __declspec(align(16))
void UYVYToYRow_AVX2(const uint8* src_uyvy,
                     uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]    // src_uyvy
    mov        edx, [esp + 8]    // dst_y
    mov        ecx, [esp + 12]   // pix

    align      4
  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    lea        eax,  [eax + 64]
    vpsrlw     ymm0, ymm0, 8      // odd bytes are Y
    vpsrlw     ymm1, ymm1, 8
    vpackuswb  ymm0, ymm0, ymm1   // mutates.
    vpermq     ymm0, ymm0, 0xd8
    sub        ecx, 32
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    jg         convertloop
    ret
    vzeroupper
  }
}

__declspec(naked) __declspec(align(16))
void UYVYToUVRow_AVX2(const uint8* src_uyvy, int stride_uyvy,
                      uint8* dst_u, uint8* dst_v, int pix) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]    // src_yuy2
    mov        esi, [esp + 8 + 8]    // stride_yuy2
    mov        edx, [esp + 8 + 12]   // dst_u
    mov        edi, [esp + 8 + 16]   // dst_v
    mov        ecx, [esp + 8 + 20]   // pix
    vpcmpeqb   ymm5, ymm5, ymm5      // generate mask 0x00ff00ff
    vpsrlw     ymm5, ymm5, 8
    sub        edi, edx

    align      4
  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    vpavgb     ymm0, ymm0, [eax + esi]
    vpavgb     ymm1, ymm1, [eax + esi + 32]
    lea        eax,  [eax + 64]
    vpand      ymm0, ymm0, ymm5   // UYVY -> UVUV
    vpand      ymm1, ymm1, ymm5
    vpackuswb  ymm0, ymm0, ymm1   // mutates.
    vpermq     ymm0, ymm0, 0xd8
    vpand      ymm1, ymm0, ymm5  // U
    vpsrlw     ymm0, ymm0, 8     // V
    vpackuswb  ymm1, ymm1, ymm1  // mutates.
    vpackuswb  ymm0, ymm0, ymm0  // mutates.
    vpermq     ymm1, ymm1, 0xd8
    vpermq     ymm0, ymm0, 0xd8
    vextractf128 [edx], ymm1, 0  // U
    vextractf128 [edx + edi], ymm0, 0 // V
    lea        edx, [edx + 16]
    sub        ecx, 32
    jg         convertloop

    pop        edi
    pop        esi
    vzeroupper
    ret
  }
}

__declspec(naked) __declspec(align(16))
void UYVYToUV422Row_AVX2(const uint8* src_uyvy,
                         uint8* dst_u, uint8* dst_v, int pix) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]    // src_yuy2
    mov        edx, [esp + 4 + 8]    // dst_u
    mov        edi, [esp + 4 + 12]   // dst_v
    mov        ecx, [esp + 4 + 16]   // pix
    vpcmpeqb   ymm5, ymm5, ymm5      // generate mask 0x00ff00ff
    vpsrlw     ymm5, ymm5, 8
    sub        edi, edx

    align      4
  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    lea        eax,  [eax + 64]
    vpand      ymm0, ymm0, ymm5   // UYVY -> UVUV
    vpand      ymm1, ymm1, ymm5
    vpackuswb  ymm0, ymm0, ymm1   // mutates.
    vpermq     ymm0, ymm0, 0xd8
    vpand      ymm1, ymm0, ymm5  // U
    vpsrlw     ymm0, ymm0, 8     // V
    vpackuswb  ymm1, ymm1, ymm1  // mutates.
    vpackuswb  ymm0, ymm0, ymm0  // mutates.
    vpermq     ymm1, ymm1, 0xd8
    vpermq     ymm0, ymm0, 0xd8
    vextractf128 [edx], ymm1, 0  // U
    vextractf128 [edx + edi], ymm0, 0 // V
    lea        edx, [edx + 16]
    sub        ecx, 32
    jg         convertloop

    pop        edi
    vzeroupper
    ret
  }
}
#endif  // HAS_YUY2TOYROW_AVX2

#ifdef HAS_YUY2TOYROW_SSE2
__declspec(naked) __declspec(align(16))
void YUY2ToYRow_SSE2(const uint8* src_yuy2,
                     uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]    // src_yuy2
    mov        edx, [esp + 8]    // dst_y
    mov        ecx, [esp + 12]   // pix
    pcmpeqb    xmm5, xmm5        // generate mask 0x00ff00ff
    psrlw      xmm5, 8

    align      4
  convertloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    pand       xmm0, xmm5   // even bytes are Y
    pand       xmm1, xmm5
    packuswb   xmm0, xmm1
    sub        ecx, 16
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void YUY2ToUVRow_SSE2(const uint8* src_yuy2, int stride_yuy2,
                      uint8* dst_u, uint8* dst_v, int pix) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]    // src_yuy2
    mov        esi, [esp + 8 + 8]    // stride_yuy2
    mov        edx, [esp + 8 + 12]   // dst_u
    mov        edi, [esp + 8 + 16]   // dst_v
    mov        ecx, [esp + 8 + 20]   // pix
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

    align      4
  convertloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + esi]
    movdqa     xmm3, [eax + esi + 16]
    lea        eax,  [eax + 32]
    pavgb      xmm0, xmm2
    pavgb      xmm1, xmm3
    psrlw      xmm0, 8      // YUYV -> UVUV
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    movdqa     xmm1, xmm0
    pand       xmm0, xmm5  // U
    packuswb   xmm0, xmm0
    psrlw      xmm1, 8     // V
    packuswb   xmm1, xmm1
    movq       qword ptr [edx], xmm0
    movq       qword ptr [edx + edi], xmm1
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void YUY2ToUV422Row_SSE2(const uint8* src_yuy2,
                         uint8* dst_u, uint8* dst_v, int pix) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]    // src_yuy2
    mov        edx, [esp + 4 + 8]    // dst_u
    mov        edi, [esp + 4 + 12]   // dst_v
    mov        ecx, [esp + 4 + 16]   // pix
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

    align      4
  convertloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    psrlw      xmm0, 8      // YUYV -> UVUV
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    movdqa     xmm1, xmm0
    pand       xmm0, xmm5  // U
    packuswb   xmm0, xmm0
    psrlw      xmm1, 8     // V
    packuswb   xmm1, xmm1
    movq       qword ptr [edx], xmm0
    movq       qword ptr [edx + edi], xmm1
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void YUY2ToYRow_Unaligned_SSE2(const uint8* src_yuy2,
                               uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]    // src_yuy2
    mov        edx, [esp + 8]    // dst_y
    mov        ecx, [esp + 12]   // pix
    pcmpeqb    xmm5, xmm5        // generate mask 0x00ff00ff
    psrlw      xmm5, 8

    align      4
  convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    pand       xmm0, xmm5   // even bytes are Y
    pand       xmm1, xmm5
    packuswb   xmm0, xmm1
    sub        ecx, 16
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void YUY2ToUVRow_Unaligned_SSE2(const uint8* src_yuy2, int stride_yuy2,
                                uint8* dst_u, uint8* dst_v, int pix) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]    // src_yuy2
    mov        esi, [esp + 8 + 8]    // stride_yuy2
    mov        edx, [esp + 8 + 12]   // dst_u
    mov        edi, [esp + 8 + 16]   // dst_v
    mov        ecx, [esp + 8 + 20]   // pix
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

    align      4
  convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + esi]
    movdqu     xmm3, [eax + esi + 16]
    lea        eax,  [eax + 32]
    pavgb      xmm0, xmm2
    pavgb      xmm1, xmm3
    psrlw      xmm0, 8      // YUYV -> UVUV
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    movdqa     xmm1, xmm0
    pand       xmm0, xmm5  // U
    packuswb   xmm0, xmm0
    psrlw      xmm1, 8     // V
    packuswb   xmm1, xmm1
    movq       qword ptr [edx], xmm0
    movq       qword ptr [edx + edi], xmm1
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void YUY2ToUV422Row_Unaligned_SSE2(const uint8* src_yuy2,
                                   uint8* dst_u, uint8* dst_v, int pix) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]    // src_yuy2
    mov        edx, [esp + 4 + 8]    // dst_u
    mov        edi, [esp + 4 + 12]   // dst_v
    mov        ecx, [esp + 4 + 16]   // pix
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

    align      4
  convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    psrlw      xmm0, 8      // YUYV -> UVUV
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    movdqa     xmm1, xmm0
    pand       xmm0, xmm5  // U
    packuswb   xmm0, xmm0
    psrlw      xmm1, 8     // V
    packuswb   xmm1, xmm1
    movq       qword ptr [edx], xmm0
    movq       qword ptr [edx + edi], xmm1
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void UYVYToYRow_SSE2(const uint8* src_uyvy,
                     uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]    // src_uyvy
    mov        edx, [esp + 8]    // dst_y
    mov        ecx, [esp + 12]   // pix

    align      4
  convertloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    psrlw      xmm0, 8    // odd bytes are Y
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    sub        ecx, 16
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void UYVYToUVRow_SSE2(const uint8* src_uyvy, int stride_uyvy,
                      uint8* dst_u, uint8* dst_v, int pix) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]    // src_yuy2
    mov        esi, [esp + 8 + 8]    // stride_yuy2
    mov        edx, [esp + 8 + 12]   // dst_u
    mov        edi, [esp + 8 + 16]   // dst_v
    mov        ecx, [esp + 8 + 20]   // pix
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

    align      4
  convertloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + esi]
    movdqa     xmm3, [eax + esi + 16]
    lea        eax,  [eax + 32]
    pavgb      xmm0, xmm2
    pavgb      xmm1, xmm3
    pand       xmm0, xmm5   // UYVY -> UVUV
    pand       xmm1, xmm5
    packuswb   xmm0, xmm1
    movdqa     xmm1, xmm0
    pand       xmm0, xmm5  // U
    packuswb   xmm0, xmm0
    psrlw      xmm1, 8     // V
    packuswb   xmm1, xmm1
    movq       qword ptr [edx], xmm0
    movq       qword ptr [edx + edi], xmm1
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void UYVYToUV422Row_SSE2(const uint8* src_uyvy,
                         uint8* dst_u, uint8* dst_v, int pix) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]    // src_yuy2
    mov        edx, [esp + 4 + 8]    // dst_u
    mov        edi, [esp + 4 + 12]   // dst_v
    mov        ecx, [esp + 4 + 16]   // pix
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

    align      4
  convertloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    pand       xmm0, xmm5   // UYVY -> UVUV
    pand       xmm1, xmm5
    packuswb   xmm0, xmm1
    movdqa     xmm1, xmm0
    pand       xmm0, xmm5  // U
    packuswb   xmm0, xmm0
    psrlw      xmm1, 8     // V
    packuswb   xmm1, xmm1
    movq       qword ptr [edx], xmm0
    movq       qword ptr [edx + edi], xmm1
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void UYVYToYRow_Unaligned_SSE2(const uint8* src_uyvy,
                               uint8* dst_y, int pix) {
  __asm {
    mov        eax, [esp + 4]    // src_uyvy
    mov        edx, [esp + 8]    // dst_y
    mov        ecx, [esp + 12]   // pix

    align      4
  convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    psrlw      xmm0, 8    // odd bytes are Y
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    sub        ecx, 16
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void UYVYToUVRow_Unaligned_SSE2(const uint8* src_uyvy, int stride_uyvy,
                                uint8* dst_u, uint8* dst_v, int pix) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]    // src_yuy2
    mov        esi, [esp + 8 + 8]    // stride_yuy2
    mov        edx, [esp + 8 + 12]   // dst_u
    mov        edi, [esp + 8 + 16]   // dst_v
    mov        ecx, [esp + 8 + 20]   // pix
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

    align      4
  convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + esi]
    movdqu     xmm3, [eax + esi + 16]
    lea        eax,  [eax + 32]
    pavgb      xmm0, xmm2
    pavgb      xmm1, xmm3
    pand       xmm0, xmm5   // UYVY -> UVUV
    pand       xmm1, xmm5
    packuswb   xmm0, xmm1
    movdqa     xmm1, xmm0
    pand       xmm0, xmm5  // U
    packuswb   xmm0, xmm0
    psrlw      xmm1, 8     // V
    packuswb   xmm1, xmm1
    movq       qword ptr [edx], xmm0
    movq       qword ptr [edx + edi], xmm1
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void UYVYToUV422Row_Unaligned_SSE2(const uint8* src_uyvy,
                                   uint8* dst_u, uint8* dst_v, int pix) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]    // src_yuy2
    mov        edx, [esp + 4 + 8]    // dst_u
    mov        edi, [esp + 4 + 12]   // dst_v
    mov        ecx, [esp + 4 + 16]   // pix
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

    align      4
  convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    pand       xmm0, xmm5   // UYVY -> UVUV
    pand       xmm1, xmm5
    packuswb   xmm0, xmm1
    movdqa     xmm1, xmm0
    pand       xmm0, xmm5  // U
    packuswb   xmm0, xmm0
    psrlw      xmm1, 8     // V
    packuswb   xmm1, xmm1
    movq       qword ptr [edx], xmm0
    movq       qword ptr [edx + edi], xmm1
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    ret
  }
}
#endif  // HAS_YUY2TOYROW_SSE2

#ifdef HAS_ARGBBLENDROW_SSE2
// Blend 8 pixels at a time.
__declspec(naked) __declspec(align(16))
void ARGBBlendRow_SSE2(const uint8* src_argb0, const uint8* src_argb1,
                       uint8* dst_argb, int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // src_argb0
    mov        esi, [esp + 4 + 8]   // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width
    pcmpeqb    xmm7, xmm7       // generate constant 1
    psrlw      xmm7, 15
    pcmpeqb    xmm6, xmm6       // generate mask 0x00ff00ff
    psrlw      xmm6, 8
    pcmpeqb    xmm5, xmm5       // generate mask 0xff00ff00
    psllw      xmm5, 8
    pcmpeqb    xmm4, xmm4       // generate mask 0xff000000
    pslld      xmm4, 24

    sub        ecx, 1
    je         convertloop1     // only 1 pixel?
    jl         convertloop1b

    // 1 pixel loop until destination pointer is aligned.
  alignloop1:
    test       edx, 15          // aligned?
    je         alignloop1b
    movd       xmm3, [eax]
    lea        eax, [eax + 4]
    movdqa     xmm0, xmm3       // src argb
    pxor       xmm3, xmm4       // ~alpha
    movd       xmm2, [esi]      // _r_b
    psrlw      xmm3, 8          // alpha
    pshufhw    xmm3, xmm3, 0F5h // 8 alpha words
    pshuflw    xmm3, xmm3, 0F5h
    pand       xmm2, xmm6       // _r_b
    paddw      xmm3, xmm7       // 256 - alpha
    pmullw     xmm2, xmm3       // _r_b * alpha
    movd       xmm1, [esi]      // _a_g
    lea        esi, [esi + 4]
    psrlw      xmm1, 8          // _a_g
    por        xmm0, xmm4       // set alpha to 255
    pmullw     xmm1, xmm3       // _a_g * alpha
    psrlw      xmm2, 8          // _r_b convert to 8 bits again
    paddusb    xmm0, xmm2       // + src argb
    pand       xmm1, xmm5       // a_g_ convert to 8 bits again
    paddusb    xmm0, xmm1       // + src argb
    sub        ecx, 1
    movd       [edx], xmm0
    lea        edx, [edx + 4]
    jge        alignloop1

  alignloop1b:
    add        ecx, 1 - 4
    jl         convertloop4b

    // 4 pixel loop.
  convertloop4:
    movdqu     xmm3, [eax]      // src argb
    lea        eax, [eax + 16]
    movdqa     xmm0, xmm3       // src argb
    pxor       xmm3, xmm4       // ~alpha
    movdqu     xmm2, [esi]      // _r_b
    psrlw      xmm3, 8          // alpha
    pshufhw    xmm3, xmm3, 0F5h // 8 alpha words
    pshuflw    xmm3, xmm3, 0F5h
    pand       xmm2, xmm6       // _r_b
    paddw      xmm3, xmm7       // 256 - alpha
    pmullw     xmm2, xmm3       // _r_b * alpha
    movdqu     xmm1, [esi]      // _a_g
    lea        esi, [esi + 16]
    psrlw      xmm1, 8          // _a_g
    por        xmm0, xmm4       // set alpha to 255
    pmullw     xmm1, xmm3       // _a_g * alpha
    psrlw      xmm2, 8          // _r_b convert to 8 bits again
    paddusb    xmm0, xmm2       // + src argb
    pand       xmm1, xmm5       // a_g_ convert to 8 bits again
    paddusb    xmm0, xmm1       // + src argb
    sub        ecx, 4
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jge        convertloop4

  convertloop4b:
    add        ecx, 4 - 1
    jl         convertloop1b

    // 1 pixel loop.
  convertloop1:
    movd       xmm3, [eax]      // src argb
    lea        eax, [eax + 4]
    movdqa     xmm0, xmm3       // src argb
    pxor       xmm3, xmm4       // ~alpha
    movd       xmm2, [esi]      // _r_b
    psrlw      xmm3, 8          // alpha
    pshufhw    xmm3, xmm3, 0F5h // 8 alpha words
    pshuflw    xmm3, xmm3, 0F5h
    pand       xmm2, xmm6       // _r_b
    paddw      xmm3, xmm7       // 256 - alpha
    pmullw     xmm2, xmm3       // _r_b * alpha
    movd       xmm1, [esi]      // _a_g
    lea        esi, [esi + 4]
    psrlw      xmm1, 8          // _a_g
    por        xmm0, xmm4       // set alpha to 255
    pmullw     xmm1, xmm3       // _a_g * alpha
    psrlw      xmm2, 8          // _r_b convert to 8 bits again
    paddusb    xmm0, xmm2       // + src argb
    pand       xmm1, xmm5       // a_g_ convert to 8 bits again
    paddusb    xmm0, xmm1       // + src argb
    sub        ecx, 1
    movd       [edx], xmm0
    lea        edx, [edx + 4]
    jge        convertloop1

  convertloop1b:
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBBLENDROW_SSE2

#ifdef HAS_ARGBBLENDROW_SSSE3
// Shuffle table for isolating alpha.
static const uvec8 kShuffleAlpha = {
  3u, 0x80, 3u, 0x80, 7u, 0x80, 7u, 0x80,
  11u, 0x80, 11u, 0x80, 15u, 0x80, 15u, 0x80
};
// Same as SSE2, but replaces:
//    psrlw      xmm3, 8          // alpha
//    pshufhw    xmm3, xmm3, 0F5h // 8 alpha words
//    pshuflw    xmm3, xmm3, 0F5h
// with..
//    pshufb     xmm3, kShuffleAlpha // alpha
// Blend 8 pixels at a time.

__declspec(naked) __declspec(align(16))
void ARGBBlendRow_SSSE3(const uint8* src_argb0, const uint8* src_argb1,
                        uint8* dst_argb, int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // src_argb0
    mov        esi, [esp + 4 + 8]   // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width
    pcmpeqb    xmm7, xmm7       // generate constant 0x0001
    psrlw      xmm7, 15
    pcmpeqb    xmm6, xmm6       // generate mask 0x00ff00ff
    psrlw      xmm6, 8
    pcmpeqb    xmm5, xmm5       // generate mask 0xff00ff00
    psllw      xmm5, 8
    pcmpeqb    xmm4, xmm4       // generate mask 0xff000000
    pslld      xmm4, 24

    sub        ecx, 1
    je         convertloop1     // only 1 pixel?
    jl         convertloop1b

    // 1 pixel loop until destination pointer is aligned.
  alignloop1:
    test       edx, 15          // aligned?
    je         alignloop1b
    movd       xmm3, [eax]
    lea        eax, [eax + 4]
    movdqa     xmm0, xmm3       // src argb
    pxor       xmm3, xmm4       // ~alpha
    movd       xmm2, [esi]      // _r_b
    pshufb     xmm3, kShuffleAlpha // alpha
    pand       xmm2, xmm6       // _r_b
    paddw      xmm3, xmm7       // 256 - alpha
    pmullw     xmm2, xmm3       // _r_b * alpha
    movd       xmm1, [esi]      // _a_g
    lea        esi, [esi + 4]
    psrlw      xmm1, 8          // _a_g
    por        xmm0, xmm4       // set alpha to 255
    pmullw     xmm1, xmm3       // _a_g * alpha
    psrlw      xmm2, 8          // _r_b convert to 8 bits again
    paddusb    xmm0, xmm2       // + src argb
    pand       xmm1, xmm5       // a_g_ convert to 8 bits again
    paddusb    xmm0, xmm1       // + src argb
    sub        ecx, 1
    movd       [edx], xmm0
    lea        edx, [edx + 4]
    jge        alignloop1

  alignloop1b:
    add        ecx, 1 - 4
    jl         convertloop4b

    test       eax, 15          // unaligned?
    jne        convertuloop4
    test       esi, 15          // unaligned?
    jne        convertuloop4

    // 4 pixel loop.
  convertloop4:
    movdqa     xmm3, [eax]      // src argb
    lea        eax, [eax + 16]
    movdqa     xmm0, xmm3       // src argb
    pxor       xmm3, xmm4       // ~alpha
    movdqa     xmm2, [esi]      // _r_b
    pshufb     xmm3, kShuffleAlpha // alpha
    pand       xmm2, xmm6       // _r_b
    paddw      xmm3, xmm7       // 256 - alpha
    pmullw     xmm2, xmm3       // _r_b * alpha
    movdqa     xmm1, [esi]      // _a_g
    lea        esi, [esi + 16]
    psrlw      xmm1, 8          // _a_g
    por        xmm0, xmm4       // set alpha to 255
    pmullw     xmm1, xmm3       // _a_g * alpha
    psrlw      xmm2, 8          // _r_b convert to 8 bits again
    paddusb    xmm0, xmm2       // + src argb
    pand       xmm1, xmm5       // a_g_ convert to 8 bits again
    paddusb    xmm0, xmm1       // + src argb
    sub        ecx, 4
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jge        convertloop4
    jmp        convertloop4b

    // 4 pixel unaligned loop.
  convertuloop4:
    movdqu     xmm3, [eax]      // src argb
    lea        eax, [eax + 16]
    movdqa     xmm0, xmm3       // src argb
    pxor       xmm3, xmm4       // ~alpha
    movdqu     xmm2, [esi]      // _r_b
    pshufb     xmm3, kShuffleAlpha // alpha
    pand       xmm2, xmm6       // _r_b
    paddw      xmm3, xmm7       // 256 - alpha
    pmullw     xmm2, xmm3       // _r_b * alpha
    movdqu     xmm1, [esi]      // _a_g
    lea        esi, [esi + 16]
    psrlw      xmm1, 8          // _a_g
    por        xmm0, xmm4       // set alpha to 255
    pmullw     xmm1, xmm3       // _a_g * alpha
    psrlw      xmm2, 8          // _r_b convert to 8 bits again
    paddusb    xmm0, xmm2       // + src argb
    pand       xmm1, xmm5       // a_g_ convert to 8 bits again
    paddusb    xmm0, xmm1       // + src argb
    sub        ecx, 4
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jge        convertuloop4

  convertloop4b:
    add        ecx, 4 - 1
    jl         convertloop1b

    // 1 pixel loop.
  convertloop1:
    movd       xmm3, [eax]      // src argb
    lea        eax, [eax + 4]
    movdqa     xmm0, xmm3       // src argb
    pxor       xmm3, xmm4       // ~alpha
    movd       xmm2, [esi]      // _r_b
    pshufb     xmm3, kShuffleAlpha // alpha
    pand       xmm2, xmm6       // _r_b
    paddw      xmm3, xmm7       // 256 - alpha
    pmullw     xmm2, xmm3       // _r_b * alpha
    movd       xmm1, [esi]      // _a_g
    lea        esi, [esi + 4]
    psrlw      xmm1, 8          // _a_g
    por        xmm0, xmm4       // set alpha to 255
    pmullw     xmm1, xmm3       // _a_g * alpha
    psrlw      xmm2, 8          // _r_b convert to 8 bits again
    paddusb    xmm0, xmm2       // + src argb
    pand       xmm1, xmm5       // a_g_ convert to 8 bits again
    paddusb    xmm0, xmm1       // + src argb
    sub        ecx, 1
    movd       [edx], xmm0
    lea        edx, [edx + 4]
    jge        convertloop1

  convertloop1b:
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBBLENDROW_SSSE3

#ifdef HAS_ARGBATTENUATEROW_SSE2
// Attenuate 4 pixels at a time.
// Aligned to 16 bytes.
__declspec(naked) __declspec(align(16))
void ARGBAttenuateRow_SSE2(const uint8* src_argb, uint8* dst_argb, int width) {
  __asm {
    mov        eax, [esp + 4]   // src_argb0
    mov        edx, [esp + 8]   // dst_argb
    mov        ecx, [esp + 12]  // width
    pcmpeqb    xmm4, xmm4       // generate mask 0xff000000
    pslld      xmm4, 24
    pcmpeqb    xmm5, xmm5       // generate mask 0x00ffffff
    psrld      xmm5, 8

    align      4
 convertloop:
    movdqa     xmm0, [eax]      // read 4 pixels
    punpcklbw  xmm0, xmm0       // first 2
    pshufhw    xmm2, xmm0, 0FFh // 8 alpha words
    pshuflw    xmm2, xmm2, 0FFh
    pmulhuw    xmm0, xmm2       // rgb * a
    movdqa     xmm1, [eax]      // read 4 pixels
    punpckhbw  xmm1, xmm1       // next 2 pixels
    pshufhw    xmm2, xmm1, 0FFh // 8 alpha words
    pshuflw    xmm2, xmm2, 0FFh
    pmulhuw    xmm1, xmm2       // rgb * a
    movdqa     xmm2, [eax]      // alphas
    lea        eax, [eax + 16]
    psrlw      xmm0, 8
    pand       xmm2, xmm4
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    pand       xmm0, xmm5       // keep original alphas
    por        xmm0, xmm2
    sub        ecx, 4
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop

    ret
  }
}
#endif  // HAS_ARGBATTENUATEROW_SSE2

#ifdef HAS_ARGBATTENUATEROW_SSSE3
// Shuffle table duplicating alpha.
static const uvec8 kShuffleAlpha0 = {
  3u, 3u, 3u, 3u, 3u, 3u, 128u, 128u, 7u, 7u, 7u, 7u, 7u, 7u, 128u, 128u,
};
static const uvec8 kShuffleAlpha1 = {
  11u, 11u, 11u, 11u, 11u, 11u, 128u, 128u,
  15u, 15u, 15u, 15u, 15u, 15u, 128u, 128u,
};
__declspec(naked) __declspec(align(16))
void ARGBAttenuateRow_SSSE3(const uint8* src_argb, uint8* dst_argb, int width) {
  __asm {
    mov        eax, [esp + 4]   // src_argb0
    mov        edx, [esp + 8]   // dst_argb
    mov        ecx, [esp + 12]  // width
    pcmpeqb    xmm3, xmm3       // generate mask 0xff000000
    pslld      xmm3, 24
    movdqa     xmm4, kShuffleAlpha0
    movdqa     xmm5, kShuffleAlpha1

    align      4
 convertloop:
    movdqu     xmm0, [eax]      // read 4 pixels
    pshufb     xmm0, xmm4       // isolate first 2 alphas
    movdqu     xmm1, [eax]      // read 4 pixels
    punpcklbw  xmm1, xmm1       // first 2 pixel rgbs
    pmulhuw    xmm0, xmm1       // rgb * a
    movdqu     xmm1, [eax]      // read 4 pixels
    pshufb     xmm1, xmm5       // isolate next 2 alphas
    movdqu     xmm2, [eax]      // read 4 pixels
    punpckhbw  xmm2, xmm2       // next 2 pixel rgbs
    pmulhuw    xmm1, xmm2       // rgb * a
    movdqu     xmm2, [eax]      // mask original alpha
    lea        eax, [eax + 16]
    pand       xmm2, xmm3
    psrlw      xmm0, 8
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    por        xmm0, xmm2       // copy original alpha
    sub        ecx, 4
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop

    ret
  }
}
#endif  // HAS_ARGBATTENUATEROW_SSSE3

#ifdef HAS_ARGBATTENUATEROW_AVX2
// Shuffle table duplicating alpha.
static const ulvec8 kShuffleAlpha_AVX2 = {
  6u, 7u, 6u, 7u, 6u, 7u, 128u, 128u,
  14u, 15u, 14u, 15u, 14u, 15u, 128u, 128u,
  6u, 7u, 6u, 7u, 6u, 7u, 128u, 128u,
  14u, 15u, 14u, 15u, 14u, 15u, 128u, 128u,
};
__declspec(naked) __declspec(align(16))
void ARGBAttenuateRow_AVX2(const uint8* src_argb, uint8* dst_argb, int width) {
  __asm {
    mov        eax, [esp + 4]   // src_argb0
    mov        edx, [esp + 8]   // dst_argb
    mov        ecx, [esp + 12]  // width
    sub        edx, eax
    vmovdqa    ymm4, kShuffleAlpha_AVX2
    vpcmpeqb   ymm5, ymm5, ymm5 // generate mask 0xff000000
    vpslld     ymm5, ymm5, 24

    align      4
 convertloop:
    vmovdqu    ymm6, [eax]       // read 8 pixels.
    vpunpcklbw ymm0, ymm6, ymm6  // low 4 pixels. mutated.
    vpunpckhbw ymm1, ymm6, ymm6  // high 4 pixels. mutated.
    vpshufb    ymm2, ymm0, ymm4  // low 4 alphas
    vpshufb    ymm3, ymm1, ymm4  // high 4 alphas
    vpmulhuw   ymm0, ymm0, ymm2  // rgb * a
    vpmulhuw   ymm1, ymm1, ymm3  // rgb * a
    vpand      ymm6, ymm6, ymm5  // isolate alpha
    vpsrlw     ymm0, ymm0, 8
    vpsrlw     ymm1, ymm1, 8
    vpackuswb  ymm0, ymm0, ymm1  // unmutated.
    vpor       ymm0, ymm0, ymm6  // copy original alpha
    sub        ecx, 8
    vmovdqu    [eax + edx], ymm0
    lea        eax, [eax + 32]
    jg         convertloop

    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBATTENUATEROW_AVX2

#ifdef HAS_ARGBUNATTENUATEROW_SSE2
// Unattenuate 4 pixels at a time.
// Aligned to 16 bytes.
__declspec(naked) __declspec(align(16))
void ARGBUnattenuateRow_SSE2(const uint8* src_argb, uint8* dst_argb,
                             int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // src_argb0
    mov        edx, [esp + 8 + 8]   // dst_argb
    mov        ecx, [esp + 8 + 12]  // width

    align      4
 convertloop:
    movdqu     xmm0, [eax]      // read 4 pixels
    movzx      esi, byte ptr [eax + 3]  // first alpha
    movzx      edi, byte ptr [eax + 7]  // second alpha
    punpcklbw  xmm0, xmm0       // first 2
    movd       xmm2, dword ptr fixed_invtbl8[esi * 4]
    movd       xmm3, dword ptr fixed_invtbl8[edi * 4]
    pshuflw    xmm2, xmm2, 040h // first 4 inv_alpha words.  1, a, a, a
    pshuflw    xmm3, xmm3, 040h // next 4 inv_alpha words
    movlhps    xmm2, xmm3
    pmulhuw    xmm0, xmm2       // rgb * a

    movdqu     xmm1, [eax]      // read 4 pixels
    movzx      esi, byte ptr [eax + 11]  // third alpha
    movzx      edi, byte ptr [eax + 15]  // forth alpha
    punpckhbw  xmm1, xmm1       // next 2
    movd       xmm2, dword ptr fixed_invtbl8[esi * 4]
    movd       xmm3, dword ptr fixed_invtbl8[edi * 4]
    pshuflw    xmm2, xmm2, 040h // first 4 inv_alpha words
    pshuflw    xmm3, xmm3, 040h // next 4 inv_alpha words
    movlhps    xmm2, xmm3
    pmulhuw    xmm1, xmm2       // rgb * a
    lea        eax, [eax + 16]

    packuswb   xmm0, xmm1
    sub        ecx, 4
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop
    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBUNATTENUATEROW_SSE2

#ifdef HAS_ARGBUNATTENUATEROW_AVX2
// Shuffle table duplicating alpha.
static const ulvec8 kUnattenShuffleAlpha_AVX2 = {
  0u, 1u, 0u, 1u, 0u, 1u, 6u, 7u, 8u, 9u, 8u, 9u, 8u, 9u, 14u, 15,
  0u, 1u, 0u, 1u, 0u, 1u, 6u, 7u, 8u, 9u, 8u, 9u, 8u, 9u, 14u, 15,
};
// TODO(fbarchard): Enable USE_GATHER for future hardware if faster.
// USE_GATHER is not on by default, due to being a slow instruction.
#ifdef USE_GATHER
__declspec(naked) __declspec(align(16))
void ARGBUnattenuateRow_AVX2(const uint8* src_argb, uint8* dst_argb,
                             int width) {
  __asm {
    mov        eax, [esp + 4]   // src_argb0
    mov        edx, [esp + 8]   // dst_argb
    mov        ecx, [esp + 12]  // width
    sub        edx, eax
    vmovdqa    ymm4, kUnattenShuffleAlpha_AVX2

    align      4
 convertloop:
    vmovdqu    ymm6, [eax]       // read 8 pixels.
    vpcmpeqb   ymm5, ymm5, ymm5  // generate mask 0xffffffff for gather.
    vpsrld     ymm2, ymm6, 24    // alpha in low 8 bits.
    vpunpcklbw ymm0, ymm6, ymm6  // low 4 pixels. mutated.
    vpunpckhbw ymm1, ymm6, ymm6  // high 4 pixels. mutated.
    vpgatherdd ymm3, [ymm2 * 4 + fixed_invtbl8], ymm5  // ymm5 cleared.  1, a
    vpunpcklwd ymm2, ymm3, ymm3  // low 4 inverted alphas. mutated. 1, 1, a, a
    vpunpckhwd ymm3, ymm3, ymm3  // high 4 inverted alphas. mutated.
    vpshufb    ymm2, ymm2, ymm4  // replicate low 4 alphas. 1, a, a, a
    vpshufb    ymm3, ymm3, ymm4  // replicate high 4 alphas
    vpmulhuw   ymm0, ymm0, ymm2  // rgb * ia
    vpmulhuw   ymm1, ymm1, ymm3  // rgb * ia
    vpackuswb  ymm0, ymm0, ymm1  // unmutated.
    sub        ecx, 8
    vmovdqu    [eax + edx], ymm0
    lea        eax, [eax + 32]
    jg         convertloop

    vzeroupper
    ret
  }
}
#else  // USE_GATHER
__declspec(naked) __declspec(align(16))
void ARGBUnattenuateRow_AVX2(const uint8* src_argb, uint8* dst_argb,
                             int width) {
  __asm {

    mov        eax, [esp + 4]   // src_argb0
    mov        edx, [esp + 8]   // dst_argb
    mov        ecx, [esp + 12]  // width
    sub        edx, eax
    vmovdqa    ymm5, kUnattenShuffleAlpha_AVX2

    push       esi
    push       edi

    align      4
 convertloop:
    // replace VPGATHER
    movzx      esi, byte ptr [eax + 3]                 // alpha0
    movzx      edi, byte ptr [eax + 7]                 // alpha1
    vmovd      xmm0, dword ptr fixed_invtbl8[esi * 4]  // [1,a0]
    vmovd      xmm1, dword ptr fixed_invtbl8[edi * 4]  // [1,a1]
    movzx      esi, byte ptr [eax + 11]                // alpha2
    movzx      edi, byte ptr [eax + 15]                // alpha3
    vpunpckldq xmm6, xmm0, xmm1                        // [1,a1,1,a0]
    vmovd      xmm2, dword ptr fixed_invtbl8[esi * 4]  // [1,a2]
    vmovd      xmm3, dword ptr fixed_invtbl8[edi * 4]  // [1,a3]
    movzx      esi, byte ptr [eax + 19]                // alpha4
    movzx      edi, byte ptr [eax + 23]                // alpha5
    vpunpckldq xmm7, xmm2, xmm3                        // [1,a3,1,a2]
    vmovd      xmm0, dword ptr fixed_invtbl8[esi * 4]  // [1,a4]
    vmovd      xmm1, dword ptr fixed_invtbl8[edi * 4]  // [1,a5]
    movzx      esi, byte ptr [eax + 27]                // alpha6
    movzx      edi, byte ptr [eax + 31]                // alpha7
    vpunpckldq xmm0, xmm0, xmm1                        // [1,a5,1,a4]
    vmovd      xmm2, dword ptr fixed_invtbl8[esi * 4]  // [1,a6]
    vmovd      xmm3, dword ptr fixed_invtbl8[edi * 4]  // [1,a7]
    vpunpckldq xmm2, xmm2, xmm3                        // [1,a7,1,a6]
    vpunpcklqdq xmm3, xmm6, xmm7                       // [1,a3,1,a2,1,a1,1,a0]
    vpunpcklqdq xmm0, xmm0, xmm2                       // [1,a7,1,a6,1,a5,1,a4]
    vinserti128 ymm3, ymm3, xmm0, 1 // [1,a7,1,a6,1,a5,1,a4,1,a3,1,a2,1,a1,1,a0]
    // end of VPGATHER

    vmovdqu    ymm6, [eax]       // read 8 pixels.
    vpunpcklbw ymm0, ymm6, ymm6  // low 4 pixels. mutated.
    vpunpckhbw ymm1, ymm6, ymm6  // high 4 pixels. mutated.
    vpunpcklwd ymm2, ymm3, ymm3  // low 4 inverted alphas. mutated. 1, 1, a, a
    vpunpckhwd ymm3, ymm3, ymm3  // high 4 inverted alphas. mutated.
    vpshufb    ymm2, ymm2, ymm5  // replicate low 4 alphas. 1, a, a, a
    vpshufb    ymm3, ymm3, ymm5  // replicate high 4 alphas
    vpmulhuw   ymm0, ymm0, ymm2  // rgb * ia
    vpmulhuw   ymm1, ymm1, ymm3  // rgb * ia
    vpackuswb  ymm0, ymm0, ymm1  // unmutated.
    sub        ecx, 8
    vmovdqu    [eax + edx], ymm0
    lea        eax, [eax + 32]
    jg         convertloop

    pop        edi
    pop        esi
    vzeroupper
    ret
  }
}
#endif  // USE_GATHER
#endif  // HAS_ARGBATTENUATEROW_AVX2

#ifdef HAS_ARGBGRAYROW_SSSE3
// Convert 8 ARGB pixels (64 bytes) to 8 Gray ARGB pixels.
__declspec(naked) __declspec(align(16))
void ARGBGrayRow_SSSE3(const uint8* src_argb, uint8* dst_argb, int width) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_argb */
    mov        ecx, [esp + 12]  /* width */
    movdqa     xmm4, kARGBToYJ
    movdqa     xmm5, kAddYJ64

    align      4
 convertloop:
    movdqa     xmm0, [eax]  // G
    movdqa     xmm1, [eax + 16]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    phaddw     xmm0, xmm1
    paddw      xmm0, xmm5  // Add .5 for rounding.
    psrlw      xmm0, 7
    packuswb   xmm0, xmm0   // 8 G bytes
    movdqa     xmm2, [eax]  // A
    movdqa     xmm3, [eax + 16]
    lea        eax, [eax + 32]
    psrld      xmm2, 24
    psrld      xmm3, 24
    packuswb   xmm2, xmm3
    packuswb   xmm2, xmm2   // 8 A bytes
    movdqa     xmm3, xmm0   // Weave into GG, GA, then GGGA
    punpcklbw  xmm0, xmm0   // 8 GG words
    punpcklbw  xmm3, xmm2   // 8 GA words
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm3   // GGGA first 4
    punpckhwd  xmm1, xmm3   // GGGA next 4
    sub        ecx, 8
    movdqa     [edx], xmm0
    movdqa     [edx + 16], xmm1
    lea        edx, [edx + 32]
    jg         convertloop
    ret
  }
}
#endif  // HAS_ARGBGRAYROW_SSSE3

#ifdef HAS_ARGBSEPIAROW_SSSE3
//    b = (r * 35 + g * 68 + b * 17) >> 7
//    g = (r * 45 + g * 88 + b * 22) >> 7
//    r = (r * 50 + g * 98 + b * 24) >> 7
// Constant for ARGB color to sepia tone.
static const vec8 kARGBToSepiaB = {
  17, 68, 35, 0, 17, 68, 35, 0, 17, 68, 35, 0, 17, 68, 35, 0
};

static const vec8 kARGBToSepiaG = {
  22, 88, 45, 0, 22, 88, 45, 0, 22, 88, 45, 0, 22, 88, 45, 0
};

static const vec8 kARGBToSepiaR = {
  24, 98, 50, 0, 24, 98, 50, 0, 24, 98, 50, 0, 24, 98, 50, 0
};

// Convert 8 ARGB pixels (32 bytes) to 8 Sepia ARGB pixels.
__declspec(naked) __declspec(align(16))
void ARGBSepiaRow_SSSE3(uint8* dst_argb, int width) {
  __asm {
    mov        eax, [esp + 4]   /* dst_argb */
    mov        ecx, [esp + 8]   /* width */
    movdqa     xmm2, kARGBToSepiaB
    movdqa     xmm3, kARGBToSepiaG
    movdqa     xmm4, kARGBToSepiaR

    align      4
 convertloop:
    movdqa     xmm0, [eax]  // B
    movdqa     xmm6, [eax + 16]
    pmaddubsw  xmm0, xmm2
    pmaddubsw  xmm6, xmm2
    phaddw     xmm0, xmm6
    psrlw      xmm0, 7
    packuswb   xmm0, xmm0   // 8 B values
    movdqa     xmm5, [eax]  // G
    movdqa     xmm1, [eax + 16]
    pmaddubsw  xmm5, xmm3
    pmaddubsw  xmm1, xmm3
    phaddw     xmm5, xmm1
    psrlw      xmm5, 7
    packuswb   xmm5, xmm5   // 8 G values
    punpcklbw  xmm0, xmm5   // 8 BG values
    movdqa     xmm5, [eax]  // R
    movdqa     xmm1, [eax + 16]
    pmaddubsw  xmm5, xmm4
    pmaddubsw  xmm1, xmm4
    phaddw     xmm5, xmm1
    psrlw      xmm5, 7
    packuswb   xmm5, xmm5   // 8 R values
    movdqa     xmm6, [eax]  // A
    movdqa     xmm1, [eax + 16]
    psrld      xmm6, 24
    psrld      xmm1, 24
    packuswb   xmm6, xmm1
    packuswb   xmm6, xmm6   // 8 A values
    punpcklbw  xmm5, xmm6   // 8 RA values
    movdqa     xmm1, xmm0   // Weave BG, RA together
    punpcklwd  xmm0, xmm5   // BGRA first 4
    punpckhwd  xmm1, xmm5   // BGRA next 4
    sub        ecx, 8
    movdqa     [eax], xmm0
    movdqa     [eax + 16], xmm1
    lea        eax, [eax + 32]
    jg         convertloop
    ret
  }
}
#endif  // HAS_ARGBSEPIAROW_SSSE3

#ifdef HAS_ARGBCOLORMATRIXROW_SSSE3
// Tranform 8 ARGB pixels (32 bytes) with color matrix.
// Same as Sepia except matrix is provided.
// TODO(fbarchard): packuswbs only use half of the reg. To make RGBA, combine R
// and B into a high and low, then G/A, unpackl/hbw and then unpckl/hwd.
__declspec(naked) __declspec(align(16))
void ARGBColorMatrixRow_SSSE3(const uint8* src_argb, uint8* dst_argb,
                              const int8* matrix_argb, int width) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_argb */
    mov        ecx, [esp + 12]  /* matrix_argb */
    movdqu     xmm5, [ecx]
    pshufd     xmm2, xmm5, 0x00
    pshufd     xmm3, xmm5, 0x55
    pshufd     xmm4, xmm5, 0xaa
    pshufd     xmm5, xmm5, 0xff
    mov        ecx, [esp + 16]  /* width */

    align      4
 convertloop:
    movdqa     xmm0, [eax]  // B
    movdqa     xmm7, [eax + 16]
    pmaddubsw  xmm0, xmm2
    pmaddubsw  xmm7, xmm2
    movdqa     xmm6, [eax]  // G
    movdqa     xmm1, [eax + 16]
    pmaddubsw  xmm6, xmm3
    pmaddubsw  xmm1, xmm3
    phaddsw    xmm0, xmm7   // B
    phaddsw    xmm6, xmm1   // G
    psraw      xmm0, 6      // B
    psraw      xmm6, 6      // G
    packuswb   xmm0, xmm0   // 8 B values
    packuswb   xmm6, xmm6   // 8 G values
    punpcklbw  xmm0, xmm6   // 8 BG values
    movdqa     xmm1, [eax]  // R
    movdqa     xmm7, [eax + 16]
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm7, xmm4
    phaddsw    xmm1, xmm7   // R
    movdqa     xmm6, [eax]  // A
    movdqa     xmm7, [eax + 16]
    pmaddubsw  xmm6, xmm5
    pmaddubsw  xmm7, xmm5
    phaddsw    xmm6, xmm7   // A
    psraw      xmm1, 6      // R
    psraw      xmm6, 6      // A
    packuswb   xmm1, xmm1   // 8 R values
    packuswb   xmm6, xmm6   // 8 A values
    punpcklbw  xmm1, xmm6   // 8 RA values
    movdqa     xmm6, xmm0   // Weave BG, RA together
    punpcklwd  xmm0, xmm1   // BGRA first 4
    punpckhwd  xmm6, xmm1   // BGRA next 4
    sub        ecx, 8
    movdqa     [edx], xmm0
    movdqa     [edx + 16], xmm6
    lea        eax, [eax + 32]
    lea        edx, [edx + 32]
    jg         convertloop
    ret
  }
}
#endif  // HAS_ARGBCOLORMATRIXROW_SSSE3

#ifdef HAS_ARGBQUANTIZEROW_SSE2
// Quantize 4 ARGB pixels (16 bytes).
// Aligned to 16 bytes.
__declspec(naked) __declspec(align(16))
void ARGBQuantizeRow_SSE2(uint8* dst_argb, int scale, int interval_size,
                          int interval_offset, int width) {
  __asm {
    mov        eax, [esp + 4]    /* dst_argb */
    movd       xmm2, [esp + 8]   /* scale */
    movd       xmm3, [esp + 12]  /* interval_size */
    movd       xmm4, [esp + 16]  /* interval_offset */
    mov        ecx, [esp + 20]   /* width */
    pshuflw    xmm2, xmm2, 040h
    pshufd     xmm2, xmm2, 044h
    pshuflw    xmm3, xmm3, 040h
    pshufd     xmm3, xmm3, 044h
    pshuflw    xmm4, xmm4, 040h
    pshufd     xmm4, xmm4, 044h
    pxor       xmm5, xmm5  // constant 0
    pcmpeqb    xmm6, xmm6  // generate mask 0xff000000
    pslld      xmm6, 24

    align      4
 convertloop:
    movdqa     xmm0, [eax]  // read 4 pixels
    punpcklbw  xmm0, xmm5   // first 2 pixels
    pmulhuw    xmm0, xmm2   // pixel * scale >> 16
    movdqa     xmm1, [eax]  // read 4 pixels
    punpckhbw  xmm1, xmm5   // next 2 pixels
    pmulhuw    xmm1, xmm2
    pmullw     xmm0, xmm3   // * interval_size
    movdqa     xmm7, [eax]  // read 4 pixels
    pmullw     xmm1, xmm3
    pand       xmm7, xmm6   // mask alpha
    paddw      xmm0, xmm4   // + interval_size / 2
    paddw      xmm1, xmm4
    packuswb   xmm0, xmm1
    por        xmm0, xmm7
    sub        ecx, 4
    movdqa     [eax], xmm0
    lea        eax, [eax + 16]
    jg         convertloop
    ret
  }
}
#endif  // HAS_ARGBQUANTIZEROW_SSE2

#ifdef HAS_ARGBSHADEROW_SSE2
// Shade 4 pixels at a time by specified value.
// Aligned to 16 bytes.
__declspec(naked) __declspec(align(16))
void ARGBShadeRow_SSE2(const uint8* src_argb, uint8* dst_argb, int width,
                       uint32 value) {
  __asm {
    mov        eax, [esp + 4]   // src_argb
    mov        edx, [esp + 8]   // dst_argb
    mov        ecx, [esp + 12]  // width
    movd       xmm2, [esp + 16]  // value
    punpcklbw  xmm2, xmm2
    punpcklqdq xmm2, xmm2

    align      4
 convertloop:
    movdqa     xmm0, [eax]      // read 4 pixels
    lea        eax, [eax + 16]
    movdqa     xmm1, xmm0
    punpcklbw  xmm0, xmm0       // first 2
    punpckhbw  xmm1, xmm1       // next 2
    pmulhuw    xmm0, xmm2       // argb * value
    pmulhuw    xmm1, xmm2       // argb * value
    psrlw      xmm0, 8
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    sub        ecx, 4
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop

    ret
  }
}
#endif  // HAS_ARGBSHADEROW_SSE2

#ifdef HAS_ARGBMULTIPLYROW_SSE2
// Multiply 2 rows of ARGB pixels together, 4 pixels at a time.
__declspec(naked) __declspec(align(16))
void ARGBMultiplyRow_SSE2(const uint8* src_argb0, const uint8* src_argb1,
                          uint8* dst_argb, int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // src_argb0
    mov        esi, [esp + 4 + 8]   // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width
    pxor       xmm5, xmm5  // constant 0

    align      4
 convertloop:
    movdqu     xmm0, [eax]        // read 4 pixels from src_argb0
    movdqu     xmm2, [esi]        // read 4 pixels from src_argb1
    movdqu     xmm1, xmm0
    movdqu     xmm3, xmm2
    punpcklbw  xmm0, xmm0         // first 2
    punpckhbw  xmm1, xmm1         // next 2
    punpcklbw  xmm2, xmm5         // first 2
    punpckhbw  xmm3, xmm5         // next 2
    pmulhuw    xmm0, xmm2         // src_argb0 * src_argb1 first 2
    pmulhuw    xmm1, xmm3         // src_argb0 * src_argb1 next 2
    lea        eax, [eax + 16]
    lea        esi, [esi + 16]
    packuswb   xmm0, xmm1
    sub        ecx, 4
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop

    pop        esi
    ret
  }
}
#endif  // HAS_ARGBMULTIPLYROW_SSE2

#ifdef HAS_ARGBADDROW_SSE2
// Add 2 rows of ARGB pixels together, 4 pixels at a time.
// TODO(fbarchard): Port this to posix, neon and other math functions.
__declspec(naked) __declspec(align(16))
void ARGBAddRow_SSE2(const uint8* src_argb0, const uint8* src_argb1,
                     uint8* dst_argb, int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // src_argb0
    mov        esi, [esp + 4 + 8]   // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width

    sub        ecx, 4
    jl         convertloop49

    align      4
 convertloop4:
    movdqu     xmm0, [eax]        // read 4 pixels from src_argb0
    lea        eax, [eax + 16]
    movdqu     xmm1, [esi]        // read 4 pixels from src_argb1
    lea        esi, [esi + 16]
    paddusb    xmm0, xmm1         // src_argb0 + src_argb1
    sub        ecx, 4
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jge        convertloop4

 convertloop49:
    add        ecx, 4 - 1
    jl         convertloop19

 convertloop1:
    movd       xmm0, [eax]        // read 1 pixels from src_argb0
    lea        eax, [eax + 4]
    movd       xmm1, [esi]        // read 1 pixels from src_argb1
    lea        esi, [esi + 4]
    paddusb    xmm0, xmm1         // src_argb0 + src_argb1
    sub        ecx, 1
    movd       [edx], xmm0
    lea        edx, [edx + 4]
    jge        convertloop1

 convertloop19:
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBADDROW_SSE2

#ifdef HAS_ARGBSUBTRACTROW_SSE2
// Subtract 2 rows of ARGB pixels together, 4 pixels at a time.
__declspec(naked) __declspec(align(16))
void ARGBSubtractRow_SSE2(const uint8* src_argb0, const uint8* src_argb1,
                          uint8* dst_argb, int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // src_argb0
    mov        esi, [esp + 4 + 8]   // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width

    align      4
 convertloop:
    movdqu     xmm0, [eax]        // read 4 pixels from src_argb0
    lea        eax, [eax + 16]
    movdqu     xmm1, [esi]        // read 4 pixels from src_argb1
    lea        esi, [esi + 16]
    psubusb    xmm0, xmm1         // src_argb0 - src_argb1
    sub        ecx, 4
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop

    pop        esi
    ret
  }
}
#endif  // HAS_ARGBSUBTRACTROW_SSE2

#ifdef HAS_ARGBMULTIPLYROW_AVX2
// Multiply 2 rows of ARGB pixels together, 8 pixels at a time.
__declspec(naked) __declspec(align(16))
void ARGBMultiplyRow_AVX2(const uint8* src_argb0, const uint8* src_argb1,
                          uint8* dst_argb, int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // src_argb0
    mov        esi, [esp + 4 + 8]   // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width
    vpxor      ymm5, ymm5, ymm5     // constant 0

    align      4
 convertloop:
    vmovdqu    ymm1, [eax]        // read 8 pixels from src_argb0
    lea        eax, [eax + 32]
    vmovdqu    ymm3, [esi]        // read 8 pixels from src_argb1
    lea        esi, [esi + 32]
    vpunpcklbw ymm0, ymm1, ymm1   // low 4
    vpunpckhbw ymm1, ymm1, ymm1   // high 4
    vpunpcklbw ymm2, ymm3, ymm5   // low 4
    vpunpckhbw ymm3, ymm3, ymm5   // high 4
    vpmulhuw   ymm0, ymm0, ymm2   // src_argb0 * src_argb1 low 4
    vpmulhuw   ymm1, ymm1, ymm3   // src_argb0 * src_argb1 high 4
    vpackuswb  ymm0, ymm0, ymm1
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBMULTIPLYROW_AVX2

#ifdef HAS_ARGBADDROW_AVX2
// Add 2 rows of ARGB pixels together, 8 pixels at a time.
__declspec(naked) __declspec(align(16))
void ARGBAddRow_AVX2(const uint8* src_argb0, const uint8* src_argb1,
                     uint8* dst_argb, int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // src_argb0
    mov        esi, [esp + 4 + 8]   // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width

    align      4
 convertloop:
    vmovdqu    ymm0, [eax]              // read 8 pixels from src_argb0
    lea        eax, [eax + 32]
    vpaddusb   ymm0, ymm0, [esi]        // add 8 pixels from src_argb1
    lea        esi, [esi + 32]
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBADDROW_AVX2

#ifdef HAS_ARGBSUBTRACTROW_AVX2
// Subtract 2 rows of ARGB pixels together, 8 pixels at a time.
__declspec(naked) __declspec(align(16))
void ARGBSubtractRow_AVX2(const uint8* src_argb0, const uint8* src_argb1,
                          uint8* dst_argb, int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // src_argb0
    mov        esi, [esp + 4 + 8]   // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width

    align      4
 convertloop:
    vmovdqu    ymm0, [eax]              // read 8 pixels from src_argb0
    lea        eax, [eax + 32]
    vpsubusb   ymm0, ymm0, [esi]        // src_argb0 - src_argb1
    lea        esi, [esi + 32]
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBSUBTRACTROW_AVX2

#ifdef HAS_SOBELXROW_SSE2
// SobelX as a matrix is
// -1  0  1
// -2  0  2
// -1  0  1
__declspec(naked) __declspec(align(16))
void SobelXRow_SSE2(const uint8* src_y0, const uint8* src_y1,
                    const uint8* src_y2, uint8* dst_sobelx, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   // src_y0
    mov        esi, [esp + 8 + 8]   // src_y1
    mov        edi, [esp + 8 + 12]  // src_y2
    mov        edx, [esp + 8 + 16]  // dst_sobelx
    mov        ecx, [esp + 8 + 20]  // width
    sub        esi, eax
    sub        edi, eax
    sub        edx, eax
    pxor       xmm5, xmm5  // constant 0

    align      4
 convertloop:
    movq       xmm0, qword ptr [eax]            // read 8 pixels from src_y0[0]
    movq       xmm1, qword ptr [eax + 2]        // read 8 pixels from src_y0[2]
    punpcklbw  xmm0, xmm5
    punpcklbw  xmm1, xmm5
    psubw      xmm0, xmm1
    movq       xmm1, qword ptr [eax + esi]      // read 8 pixels from src_y1[0]
    movq       xmm2, qword ptr [eax + esi + 2]  // read 8 pixels from src_y1[2]
    punpcklbw  xmm1, xmm5
    punpcklbw  xmm2, xmm5
    psubw      xmm1, xmm2
    movq       xmm2, qword ptr [eax + edi]      // read 8 pixels from src_y2[0]
    movq       xmm3, qword ptr [eax + edi + 2]  // read 8 pixels from src_y2[2]
    punpcklbw  xmm2, xmm5
    punpcklbw  xmm3, xmm5
    psubw      xmm2, xmm3
    paddw      xmm0, xmm2
    paddw      xmm0, xmm1
    paddw      xmm0, xmm1
    pxor       xmm1, xmm1   // abs = max(xmm0, -xmm0).  SSSE3 could use pabsw
    psubw      xmm1, xmm0
    pmaxsw     xmm0, xmm1
    packuswb   xmm0, xmm0
    sub        ecx, 8
    movq       qword ptr [eax + edx], xmm0
    lea        eax, [eax + 8]
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_SOBELXROW_SSE2

#ifdef HAS_SOBELYROW_SSE2
// SobelY as a matrix is
// -1 -2 -1
//  0  0  0
//  1  2  1
__declspec(naked) __declspec(align(16))
void SobelYRow_SSE2(const uint8* src_y0, const uint8* src_y1,
                    uint8* dst_sobely, int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // src_y0
    mov        esi, [esp + 4 + 8]   // src_y1
    mov        edx, [esp + 4 + 12]  // dst_sobely
    mov        ecx, [esp + 4 + 16]  // width
    sub        esi, eax
    sub        edx, eax
    pxor       xmm5, xmm5  // constant 0

    align      4
 convertloop:
    movq       xmm0, qword ptr [eax]            // read 8 pixels from src_y0[0]
    movq       xmm1, qword ptr [eax + esi]      // read 8 pixels from src_y1[0]
    punpcklbw  xmm0, xmm5
    punpcklbw  xmm1, xmm5
    psubw      xmm0, xmm1
    movq       xmm1, qword ptr [eax + 1]        // read 8 pixels from src_y0[1]
    movq       xmm2, qword ptr [eax + esi + 1]  // read 8 pixels from src_y1[1]
    punpcklbw  xmm1, xmm5
    punpcklbw  xmm2, xmm5
    psubw      xmm1, xmm2
    movq       xmm2, qword ptr [eax + 2]        // read 8 pixels from src_y0[2]
    movq       xmm3, qword ptr [eax + esi + 2]  // read 8 pixels from src_y1[2]
    punpcklbw  xmm2, xmm5
    punpcklbw  xmm3, xmm5
    psubw      xmm2, xmm3
    paddw      xmm0, xmm2
    paddw      xmm0, xmm1
    paddw      xmm0, xmm1
    pxor       xmm1, xmm1   // abs = max(xmm0, -xmm0).  SSSE3 could use pabsw
    psubw      xmm1, xmm0
    pmaxsw     xmm0, xmm1
    packuswb   xmm0, xmm0
    sub        ecx, 8
    movq       qword ptr [eax + edx], xmm0
    lea        eax, [eax + 8]
    jg         convertloop

    pop        esi
    ret
  }
}
#endif  // HAS_SOBELYROW_SSE2

#ifdef HAS_SOBELROW_SSE2
// Adds Sobel X and Sobel Y and stores Sobel into ARGB.
// A = 255
// R = Sobel
// G = Sobel
// B = Sobel
__declspec(naked) __declspec(align(16))
void SobelRow_SSE2(const uint8* src_sobelx, const uint8* src_sobely,
                   uint8* dst_argb, int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // src_sobelx
    mov        esi, [esp + 4 + 8]   // src_sobely
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width
    sub        esi, eax
    pcmpeqb    xmm5, xmm5           // alpha 255
    pslld      xmm5, 24             // 0xff000000

    align      4
 convertloop:
    movdqa     xmm0, [eax]            // read 16 pixels src_sobelx
    movdqa     xmm1, [eax + esi]      // read 16 pixels src_sobely
    lea        eax, [eax + 16]
    paddusb    xmm0, xmm1             // sobel = sobelx + sobely
    movdqa     xmm2, xmm0             // GG
    punpcklbw  xmm2, xmm0             // First 8
    punpckhbw  xmm0, xmm0             // Next 8
    movdqa     xmm1, xmm2             // GGGG
    punpcklwd  xmm1, xmm2             // First 4
    punpckhwd  xmm2, xmm2             // Next 4
    por        xmm1, xmm5             // GGGA
    por        xmm2, xmm5
    movdqa     xmm3, xmm0             // GGGG
    punpcklwd  xmm3, xmm0             // Next 4
    punpckhwd  xmm0, xmm0             // Last 4
    por        xmm3, xmm5             // GGGA
    por        xmm0, xmm5
    sub        ecx, 16
    movdqa     [edx], xmm1
    movdqa     [edx + 16], xmm2
    movdqa     [edx + 32], xmm3
    movdqa     [edx + 48], xmm0
    lea        edx, [edx + 64]
    jg         convertloop

    pop        esi
    ret
  }
}
#endif  // HAS_SOBELROW_SSE2

#ifdef HAS_SOBELTOPLANEROW_SSE2
// Adds Sobel X and Sobel Y and stores Sobel into a plane.
__declspec(naked) __declspec(align(16))
void SobelToPlaneRow_SSE2(const uint8* src_sobelx, const uint8* src_sobely,
                          uint8* dst_y, int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // src_sobelx
    mov        esi, [esp + 4 + 8]   // src_sobely
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width
    sub        esi, eax

    align      4
 convertloop:
    movdqa     xmm0, [eax]            // read 16 pixels src_sobelx
    movdqa     xmm1, [eax + esi]      // read 16 pixels src_sobely
    lea        eax, [eax + 16]
    paddusb    xmm0, xmm1             // sobel = sobelx + sobely
    sub        ecx, 16
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         convertloop

    pop        esi
    ret
  }
}
#endif  // HAS_SOBELTOPLANEROW_SSE2

#ifdef HAS_SOBELXYROW_SSE2
// Mixes Sobel X, Sobel Y and Sobel into ARGB.
// A = 255
// R = Sobel X
// G = Sobel
// B = Sobel Y
__declspec(naked) __declspec(align(16))
void SobelXYRow_SSE2(const uint8* src_sobelx, const uint8* src_sobely,
                     uint8* dst_argb, int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   // src_sobelx
    mov        esi, [esp + 4 + 8]   // src_sobely
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width
    sub        esi, eax
    pcmpeqb    xmm5, xmm5           // alpha 255

    align      4
 convertloop:
    movdqa     xmm0, [eax]            // read 16 pixels src_sobelx
    movdqa     xmm1, [eax + esi]      // read 16 pixels src_sobely
    lea        eax, [eax + 16]
    movdqa     xmm2, xmm0
    paddusb    xmm2, xmm1             // sobel = sobelx + sobely
    movdqa     xmm3, xmm0             // XA
    punpcklbw  xmm3, xmm5
    punpckhbw  xmm0, xmm5
    movdqa     xmm4, xmm1             // YS
    punpcklbw  xmm4, xmm2
    punpckhbw  xmm1, xmm2
    movdqa     xmm6, xmm4             // YSXA
    punpcklwd  xmm6, xmm3             // First 4
    punpckhwd  xmm4, xmm3             // Next 4
    movdqa     xmm7, xmm1             // YSXA
    punpcklwd  xmm7, xmm0             // Next 4
    punpckhwd  xmm1, xmm0             // Last 4
    sub        ecx, 16
    movdqa     [edx], xmm6
    movdqa     [edx + 16], xmm4
    movdqa     [edx + 32], xmm7
    movdqa     [edx + 48], xmm1
    lea        edx, [edx + 64]
    jg         convertloop

    pop        esi
    ret
  }
}
#endif  // HAS_SOBELXYROW_SSE2

#ifdef HAS_CUMULATIVESUMTOAVERAGEROW_SSE2
// Consider float CumulativeSum.
// Consider calling CumulativeSum one row at time as needed.
// Consider circular CumulativeSum buffer of radius * 2 + 1 height.
// Convert cumulative sum for an area to an average for 1 pixel.
// topleft is pointer to top left of CumulativeSum buffer for area.
// botleft is pointer to bottom left of CumulativeSum buffer.
// width is offset from left to right of area in CumulativeSum buffer measured
//   in number of ints.
// area is the number of pixels in the area being averaged.
// dst points to pixel to store result to.
// count is number of averaged pixels to produce.
// Does 4 pixels at a time, requires CumulativeSum pointers to be 16 byte
// aligned.
void CumulativeSumToAverageRow_SSE2(const int32* topleft, const int32* botleft,
                                    int width, int area, uint8* dst,
                                    int count) {
  __asm {
    mov        eax, topleft  // eax topleft
    mov        esi, botleft  // esi botleft
    mov        edx, width
    movd       xmm5, area
    mov        edi, dst
    mov        ecx, count
    cvtdq2ps   xmm5, xmm5
    rcpss      xmm4, xmm5  // 1.0f / area
    pshufd     xmm4, xmm4, 0
    sub        ecx, 4
    jl         l4b

    cmp        area, 128  // 128 pixels will not overflow 15 bits.
    ja         l4

    pshufd     xmm5, xmm5, 0        // area
    pcmpeqb    xmm6, xmm6           // constant of 65536.0 - 1 = 65535.0
    psrld      xmm6, 16
    cvtdq2ps   xmm6, xmm6
    addps      xmm5, xmm6           // (65536.0 + area - 1)
    mulps      xmm5, xmm4           // (65536.0 + area - 1) * 1 / area
    cvtps2dq   xmm5, xmm5           // 0.16 fixed point
    packssdw   xmm5, xmm5           // 16 bit shorts

    // 4 pixel loop small blocks.
    align      4
  s4:
    // top left
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]

    // - top right
    psubd      xmm0, [eax + edx * 4]
    psubd      xmm1, [eax + edx * 4 + 16]
    psubd      xmm2, [eax + edx * 4 + 32]
    psubd      xmm3, [eax + edx * 4 + 48]
    lea        eax, [eax + 64]

    // - bottom left
    psubd      xmm0, [esi]
    psubd      xmm1, [esi + 16]
    psubd      xmm2, [esi + 32]
    psubd      xmm3, [esi + 48]

    // + bottom right
    paddd      xmm0, [esi + edx * 4]
    paddd      xmm1, [esi + edx * 4 + 16]
    paddd      xmm2, [esi + edx * 4 + 32]
    paddd      xmm3, [esi + edx * 4 + 48]
    lea        esi, [esi + 64]

    packssdw   xmm0, xmm1  // pack 4 pixels into 2 registers
    packssdw   xmm2, xmm3

    pmulhuw    xmm0, xmm5
    pmulhuw    xmm2, xmm5

    packuswb   xmm0, xmm2
    movdqu     [edi], xmm0
    lea        edi, [edi + 16]
    sub        ecx, 4
    jge        s4

    jmp        l4b

    // 4 pixel loop
    align      4
  l4:
    // top left
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + 32]
    movdqa     xmm3, [eax + 48]

    // - top right
    psubd      xmm0, [eax + edx * 4]
    psubd      xmm1, [eax + edx * 4 + 16]
    psubd      xmm2, [eax + edx * 4 + 32]
    psubd      xmm3, [eax + edx * 4 + 48]
    lea        eax, [eax + 64]

    // - bottom left
    psubd      xmm0, [esi]
    psubd      xmm1, [esi + 16]
    psubd      xmm2, [esi + 32]
    psubd      xmm3, [esi + 48]

    // + bottom right
    paddd      xmm0, [esi + edx * 4]
    paddd      xmm1, [esi + edx * 4 + 16]
    paddd      xmm2, [esi + edx * 4 + 32]
    paddd      xmm3, [esi + edx * 4 + 48]
    lea        esi, [esi + 64]

    cvtdq2ps   xmm0, xmm0   // Average = Sum * 1 / Area
    cvtdq2ps   xmm1, xmm1
    mulps      xmm0, xmm4
    mulps      xmm1, xmm4
    cvtdq2ps   xmm2, xmm2
    cvtdq2ps   xmm3, xmm3
    mulps      xmm2, xmm4
    mulps      xmm3, xmm4
    cvtps2dq   xmm0, xmm0
    cvtps2dq   xmm1, xmm1
    cvtps2dq   xmm2, xmm2
    cvtps2dq   xmm3, xmm3
    packssdw   xmm0, xmm1
    packssdw   xmm2, xmm3
    packuswb   xmm0, xmm2
    movdqu     [edi], xmm0
    lea        edi, [edi + 16]
    sub        ecx, 4
    jge        l4

  l4b:
    add        ecx, 4 - 1
    jl         l1b

    // 1 pixel loop
    align      4
  l1:
    movdqa     xmm0, [eax]
    psubd      xmm0, [eax + edx * 4]
    lea        eax, [eax + 16]
    psubd      xmm0, [esi]
    paddd      xmm0, [esi + edx * 4]
    lea        esi, [esi + 16]
    cvtdq2ps   xmm0, xmm0
    mulps      xmm0, xmm4
    cvtps2dq   xmm0, xmm0
    packssdw   xmm0, xmm0
    packuswb   xmm0, xmm0
    movd       dword ptr [edi], xmm0
    lea        edi, [edi + 4]
    sub        ecx, 1
    jge        l1
  l1b:
  }
}
#endif  // HAS_CUMULATIVESUMTOAVERAGEROW_SSE2

#ifdef HAS_COMPUTECUMULATIVESUMROW_SSE2
// Creates a table of cumulative sums where each value is a sum of all values
// above and to the left of the value.
void ComputeCumulativeSumRow_SSE2(const uint8* row, int32* cumsum,
                                  const int32* previous_cumsum, int width) {
  __asm {
    mov        eax, row
    mov        edx, cumsum
    mov        esi, previous_cumsum
    mov        ecx, width
    pxor       xmm0, xmm0
    pxor       xmm1, xmm1

    sub        ecx, 4
    jl         l4b
    test       edx, 15
    jne        l4b

    // 4 pixel loop
    align      4
  l4:
    movdqu     xmm2, [eax]  // 4 argb pixels 16 bytes.
    lea        eax, [eax + 16]
    movdqa     xmm4, xmm2

    punpcklbw  xmm2, xmm1
    movdqa     xmm3, xmm2
    punpcklwd  xmm2, xmm1
    punpckhwd  xmm3, xmm1

    punpckhbw  xmm4, xmm1
    movdqa     xmm5, xmm4
    punpcklwd  xmm4, xmm1
    punpckhwd  xmm5, xmm1

    paddd      xmm0, xmm2
    movdqa     xmm2, [esi]  // previous row above.
    paddd      xmm2, xmm0

    paddd      xmm0, xmm3
    movdqa     xmm3, [esi + 16]
    paddd      xmm3, xmm0

    paddd      xmm0, xmm4
    movdqa     xmm4, [esi + 32]
    paddd      xmm4, xmm0

    paddd      xmm0, xmm5
    movdqa     xmm5, [esi + 48]
    lea        esi, [esi + 64]
    paddd      xmm5, xmm0

    movdqa     [edx], xmm2
    movdqa     [edx + 16], xmm3
    movdqa     [edx + 32], xmm4
    movdqa     [edx + 48], xmm5

    lea        edx, [edx + 64]
    sub        ecx, 4
    jge        l4

  l4b:
    add        ecx, 4 - 1
    jl         l1b

    // 1 pixel loop
    align      4
  l1:
    movd       xmm2, dword ptr [eax]  // 1 argb pixel 4 bytes.
    lea        eax, [eax + 4]
    punpcklbw  xmm2, xmm1
    punpcklwd  xmm2, xmm1
    paddd      xmm0, xmm2
    movdqu     xmm2, [esi]
    lea        esi, [esi + 16]
    paddd      xmm2, xmm0
    movdqu     [edx], xmm2
    lea        edx, [edx + 16]
    sub        ecx, 1
    jge        l1

 l1b:
  }
}
#endif  // HAS_COMPUTECUMULATIVESUMROW_SSE2

#ifdef HAS_ARGBAFFINEROW_SSE2
// Copy ARGB pixels from source image with slope to a row of destination.
__declspec(naked) __declspec(align(16))
LIBYUV_API
void ARGBAffineRow_SSE2(const uint8* src_argb, int src_argb_stride,
                        uint8* dst_argb, const float* uv_dudv, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 12]  // src_argb
    mov        esi, [esp + 16]  // stride
    mov        edx, [esp + 20]  // dst_argb
    mov        ecx, [esp + 24]  // pointer to uv_dudv
    movq       xmm2, qword ptr [ecx]  // uv
    movq       xmm7, qword ptr [ecx + 8]  // dudv
    mov        ecx, [esp + 28]  // width
    shl        esi, 16          // 4, stride
    add        esi, 4
    movd       xmm5, esi
    sub        ecx, 4
    jl         l4b

    // setup for 4 pixel loop
    pshufd     xmm7, xmm7, 0x44  // dup dudv
    pshufd     xmm5, xmm5, 0  // dup 4, stride
    movdqa     xmm0, xmm2    // x0, y0, x1, y1
    addps      xmm0, xmm7
    movlhps    xmm2, xmm0
    movdqa     xmm4, xmm7
    addps      xmm4, xmm4    // dudv *= 2
    movdqa     xmm3, xmm2    // x2, y2, x3, y3
    addps      xmm3, xmm4
    addps      xmm4, xmm4    // dudv *= 4

    // 4 pixel loop
    align      4
  l4:
    cvttps2dq  xmm0, xmm2    // x, y float to int first 2
    cvttps2dq  xmm1, xmm3    // x, y float to int next 2
    packssdw   xmm0, xmm1    // x, y as 8 shorts
    pmaddwd    xmm0, xmm5    // offsets = x * 4 + y * stride.
    movd       esi, xmm0
    pshufd     xmm0, xmm0, 0x39  // shift right
    movd       edi, xmm0
    pshufd     xmm0, xmm0, 0x39  // shift right
    movd       xmm1, [eax + esi]  // read pixel 0
    movd       xmm6, [eax + edi]  // read pixel 1
    punpckldq  xmm1, xmm6     // combine pixel 0 and 1
    addps      xmm2, xmm4    // x, y += dx, dy first 2
    movq       qword ptr [edx], xmm1
    movd       esi, xmm0
    pshufd     xmm0, xmm0, 0x39  // shift right
    movd       edi, xmm0
    movd       xmm6, [eax + esi]  // read pixel 2
    movd       xmm0, [eax + edi]  // read pixel 3
    punpckldq  xmm6, xmm0     // combine pixel 2 and 3
    addps      xmm3, xmm4    // x, y += dx, dy next 2
    sub        ecx, 4
    movq       qword ptr 8[edx], xmm6
    lea        edx, [edx + 16]
    jge        l4

  l4b:
    add        ecx, 4 - 1
    jl         l1b

    // 1 pixel loop
    align      4
  l1:
    cvttps2dq  xmm0, xmm2    // x, y float to int
    packssdw   xmm0, xmm0    // x, y as shorts
    pmaddwd    xmm0, xmm5    // offset = x * 4 + y * stride
    addps      xmm2, xmm7    // x, y += dx, dy
    movd       esi, xmm0
    movd       xmm0, [eax + esi]  // copy a pixel
    sub        ecx, 1
    movd       [edx], xmm0
    lea        edx, [edx + 4]
    jge        l1
  l1b:
    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBAFFINEROW_SSE2

#ifdef HAS_INTERPOLATEROW_AVX2
// Bilinear filter 16x2 -> 16x1
__declspec(naked) __declspec(align(16))
void InterpolateRow_AVX2(uint8* dst_ptr, const uint8* src_ptr,
                          ptrdiff_t src_stride, int dst_width,
                          int source_y_fraction) {
  __asm {
    push       esi
    push       edi
    mov        edi, [esp + 8 + 4]   // dst_ptr
    mov        esi, [esp + 8 + 8]   // src_ptr
    mov        edx, [esp + 8 + 12]  // src_stride
    mov        ecx, [esp + 8 + 16]  // dst_width
    mov        eax, [esp + 8 + 20]  // source_y_fraction (0..255)
    shr        eax, 1
    // Dispatch to specialized filters if applicable.
    cmp        eax, 0
    je         xloop100  // 0 / 128.  Blend 100 / 0.
    sub        edi, esi
    cmp        eax, 32
    je         xloop75   // 32 / 128 is 0.25.  Blend 75 / 25.
    cmp        eax, 64
    je         xloop50   // 64 / 128 is 0.50.  Blend 50 / 50.
    cmp        eax, 96
    je         xloop25   // 96 / 128 is 0.75.  Blend 25 / 75.

    vmovd      xmm0, eax  // high fraction 0..127
    neg        eax
    add        eax, 128
    vmovd      xmm5, eax  // low fraction 128..1
    vpunpcklbw xmm5, xmm5, xmm0
    vpunpcklwd xmm5, xmm5, xmm5
    vpxor      ymm0, ymm0, ymm0
    vpermd     ymm5, ymm0, ymm5

    align      4
  xloop:
    vmovdqu    ymm0, [esi]
    vmovdqu    ymm2, [esi + edx]
    vpunpckhbw ymm1, ymm0, ymm2  // mutates
    vpunpcklbw ymm0, ymm0, ymm2  // mutates
    vpmaddubsw ymm0, ymm0, ymm5
    vpmaddubsw ymm1, ymm1, ymm5
    vpsrlw     ymm0, ymm0, 7
    vpsrlw     ymm1, ymm1, 7
    vpackuswb  ymm0, ymm0, ymm1  // unmutates
    sub        ecx, 32
    vmovdqu    [esi + edi], ymm0
    lea        esi, [esi + 32]
    jg         xloop
    jmp        xloop99

    // Blend 25 / 75.
    align      4
  xloop25:
    vmovdqu    ymm0, [esi]
    vpavgb     ymm0, ymm0, [esi + edx]
    vpavgb     ymm0, ymm0, [esi + edx]
    sub        ecx, 32
    vmovdqu    [esi + edi], ymm0
    lea        esi, [esi + 32]
    jg         xloop25
    jmp        xloop99

    // Blend 50 / 50.
    align      4
  xloop50:
    vmovdqu    ymm0, [esi]
    vpavgb     ymm0, ymm0, [esi + edx]
    sub        ecx, 32
    vmovdqu    [esi + edi], ymm0
    lea        esi, [esi + 32]
    jg         xloop50
    jmp        xloop99

    // Blend 75 / 25.
    align      4
  xloop75:
    vmovdqu    ymm0, [esi + edx]
    vpavgb     ymm0, ymm0, [esi]
    vpavgb     ymm0, ymm0, [esi]
    sub        ecx, 32
    vmovdqu     [esi + edi], ymm0
    lea        esi, [esi + 32]
    jg         xloop75
    jmp        xloop99

    // Blend 100 / 0 - Copy row unchanged.
    align      4
  xloop100:
    rep movsb

  xloop99:
    pop        edi
    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_INTERPOLATEROW_AVX2

#ifdef HAS_INTERPOLATEROW_SSSE3
// Bilinear filter 16x2 -> 16x1
__declspec(naked) __declspec(align(16))
void InterpolateRow_SSSE3(uint8* dst_ptr, const uint8* src_ptr,
                          ptrdiff_t src_stride, int dst_width,
                          int source_y_fraction) {
  __asm {
    push       esi
    push       edi
    mov        edi, [esp + 8 + 4]   // dst_ptr
    mov        esi, [esp + 8 + 8]   // src_ptr
    mov        edx, [esp + 8 + 12]  // src_stride
    mov        ecx, [esp + 8 + 16]  // dst_width
    mov        eax, [esp + 8 + 20]  // source_y_fraction (0..255)
    sub        edi, esi
    shr        eax, 1
    // Dispatch to specialized filters if applicable.
    cmp        eax, 0
    je         xloop100  // 0 / 128.  Blend 100 / 0.
    cmp        eax, 32
    je         xloop75   // 32 / 128 is 0.25.  Blend 75 / 25.
    cmp        eax, 64
    je         xloop50   // 64 / 128 is 0.50.  Blend 50 / 50.
    cmp        eax, 96
    je         xloop25   // 96 / 128 is 0.75.  Blend 25 / 75.

    movd       xmm0, eax  // high fraction 0..127
    neg        eax
    add        eax, 128
    movd       xmm5, eax  // low fraction 128..1
    punpcklbw  xmm5, xmm0
    punpcklwd  xmm5, xmm5
    pshufd     xmm5, xmm5, 0

    align      4
  xloop:
    movdqa     xmm0, [esi]
    movdqa     xmm2, [esi + edx]
    movdqa     xmm1, xmm0
    punpcklbw  xmm0, xmm2
    punpckhbw  xmm1, xmm2
    pmaddubsw  xmm0, xmm5
    pmaddubsw  xmm1, xmm5
    psrlw      xmm0, 7
    psrlw      xmm1, 7
    packuswb   xmm0, xmm1
    sub        ecx, 16
    movdqa     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop
    jmp        xloop99

    // Blend 25 / 75.
    align      4
  xloop25:
    movdqa     xmm0, [esi]
    movdqa     xmm1, [esi + edx]
    pavgb      xmm0, xmm1
    pavgb      xmm0, xmm1
    sub        ecx, 16
    movdqa     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop25
    jmp        xloop99

    // Blend 50 / 50.
    align      4
  xloop50:
    movdqa     xmm0, [esi]
    movdqa     xmm1, [esi + edx]
    pavgb      xmm0, xmm1
    sub        ecx, 16
    movdqa     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop50
    jmp        xloop99

    // Blend 75 / 25.
    align      4
  xloop75:
    movdqa     xmm1, [esi]
    movdqa     xmm0, [esi + edx]
    pavgb      xmm0, xmm1
    pavgb      xmm0, xmm1
    sub        ecx, 16
    movdqa     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop75
    jmp        xloop99

    // Blend 100 / 0 - Copy row unchanged.
    align      4
  xloop100:
    movdqa     xmm0, [esi]
    sub        ecx, 16
    movdqa     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop100

  xloop99:
    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_INTERPOLATEROW_SSSE3

#ifdef HAS_INTERPOLATEROW_SSE2
// Bilinear filter 16x2 -> 16x1
__declspec(naked) __declspec(align(16))
void InterpolateRow_SSE2(uint8* dst_ptr, const uint8* src_ptr,
                         ptrdiff_t src_stride, int dst_width,
                         int source_y_fraction) {
  __asm {
    push       esi
    push       edi
    mov        edi, [esp + 8 + 4]   // dst_ptr
    mov        esi, [esp + 8 + 8]   // src_ptr
    mov        edx, [esp + 8 + 12]  // src_stride
    mov        ecx, [esp + 8 + 16]  // dst_width
    mov        eax, [esp + 8 + 20]  // source_y_fraction (0..255)
    sub        edi, esi
    // Dispatch to specialized filters if applicable.
    cmp        eax, 0
    je         xloop100  // 0 / 256.  Blend 100 / 0.
    cmp        eax, 64
    je         xloop75   // 64 / 256 is 0.25.  Blend 75 / 25.
    cmp        eax, 128
    je         xloop50   // 128 / 256 is 0.50.  Blend 50 / 50.
    cmp        eax, 192
    je         xloop25   // 192 / 256 is 0.75.  Blend 25 / 75.

    movd       xmm5, eax            // xmm5 = y fraction
    punpcklbw  xmm5, xmm5
    psrlw      xmm5, 1
    punpcklwd  xmm5, xmm5
    punpckldq  xmm5, xmm5
    punpcklqdq xmm5, xmm5
    pxor       xmm4, xmm4

    align      4
  xloop:
    movdqa     xmm0, [esi]  // row0
    movdqa     xmm2, [esi + edx]  // row1
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    punpcklbw  xmm2, xmm4
    punpckhbw  xmm3, xmm4
    punpcklbw  xmm0, xmm4
    punpckhbw  xmm1, xmm4
    psubw      xmm2, xmm0  // row1 - row0
    psubw      xmm3, xmm1
    paddw      xmm2, xmm2  // 9 bits * 15 bits = 8.16
    paddw      xmm3, xmm3
    pmulhw     xmm2, xmm5  // scale diff
    pmulhw     xmm3, xmm5
    paddw      xmm0, xmm2  // sum rows
    paddw      xmm1, xmm3
    packuswb   xmm0, xmm1
    sub        ecx, 16
    movdqa     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop
    jmp        xloop99

    // Blend 25 / 75.
    align      4
  xloop25:
    movdqa     xmm0, [esi]
    movdqa     xmm1, [esi + edx]
    pavgb      xmm0, xmm1
    pavgb      xmm0, xmm1
    sub        ecx, 16
    movdqa     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop25
    jmp        xloop99

    // Blend 50 / 50.
    align      4
  xloop50:
    movdqa     xmm0, [esi]
    movdqa     xmm1, [esi + edx]
    pavgb      xmm0, xmm1
    sub        ecx, 16
    movdqa     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop50
    jmp        xloop99

    // Blend 75 / 25.
    align      4
  xloop75:
    movdqa     xmm1, [esi]
    movdqa     xmm0, [esi + edx]
    pavgb      xmm0, xmm1
    pavgb      xmm0, xmm1
    sub        ecx, 16
    movdqa     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop75
    jmp        xloop99

    // Blend 100 / 0 - Copy row unchanged.
    align      4
  xloop100:
    movdqa     xmm0, [esi]
    sub        ecx, 16
    movdqa     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop100

  xloop99:
    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_INTERPOLATEROW_SSE2

// Bilinear filter 16x2 -> 16x1
__declspec(naked) __declspec(align(16))
void InterpolateRow_Unaligned_SSSE3(uint8* dst_ptr, const uint8* src_ptr,
                                    ptrdiff_t src_stride, int dst_width,
                                    int source_y_fraction) {
  __asm {
    push       esi
    push       edi
    mov        edi, [esp + 8 + 4]   // dst_ptr
    mov        esi, [esp + 8 + 8]   // src_ptr
    mov        edx, [esp + 8 + 12]  // src_stride
    mov        ecx, [esp + 8 + 16]  // dst_width
    mov        eax, [esp + 8 + 20]  // source_y_fraction (0..255)
    sub        edi, esi
    shr        eax, 1
    // Dispatch to specialized filters if applicable.
    cmp        eax, 0
    je         xloop100  // 0 / 128.  Blend 100 / 0.
    cmp        eax, 32
    je         xloop75   // 32 / 128 is 0.25.  Blend 75 / 25.
    cmp        eax, 64
    je         xloop50   // 64 / 128 is 0.50.  Blend 50 / 50.
    cmp        eax, 96
    je         xloop25   // 96 / 128 is 0.75.  Blend 25 / 75.

    movd       xmm0, eax  // high fraction 0..127
    neg        eax
    add        eax, 128
    movd       xmm5, eax  // low fraction 128..1
    punpcklbw  xmm5, xmm0
    punpcklwd  xmm5, xmm5
    pshufd     xmm5, xmm5, 0

    align      4
  xloop:
    movdqu     xmm0, [esi]
    movdqu     xmm2, [esi + edx]
    movdqu     xmm1, xmm0
    punpcklbw  xmm0, xmm2
    punpckhbw  xmm1, xmm2
    pmaddubsw  xmm0, xmm5
    pmaddubsw  xmm1, xmm5
    psrlw      xmm0, 7
    psrlw      xmm1, 7
    packuswb   xmm0, xmm1
    sub        ecx, 16
    movdqu     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop
    jmp        xloop99

    // Blend 25 / 75.
    align      4
  xloop25:
    movdqu     xmm0, [esi]
    movdqu     xmm1, [esi + edx]
    pavgb      xmm0, xmm1
    pavgb      xmm0, xmm1
    sub        ecx, 16
    movdqu     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop25
    jmp        xloop99

    // Blend 50 / 50.
    align      4
  xloop50:
    movdqu     xmm0, [esi]
    movdqu     xmm1, [esi + edx]
    pavgb      xmm0, xmm1
    sub        ecx, 16
    movdqu     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop50
    jmp        xloop99

    // Blend 75 / 25.
    align      4
  xloop75:
    movdqu     xmm1, [esi]
    movdqu     xmm0, [esi + edx]
    pavgb      xmm0, xmm1
    pavgb      xmm0, xmm1
    sub        ecx, 16
    movdqu     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop75
    jmp        xloop99

    // Blend 100 / 0 - Copy row unchanged.
    align      4
  xloop100:
    movdqu     xmm0, [esi]
    sub        ecx, 16
    movdqu     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop100

  xloop99:
    pop        edi
    pop        esi
    ret
  }
}

#ifdef HAS_INTERPOLATEROW_SSE2
// Bilinear filter 16x2 -> 16x1
__declspec(naked) __declspec(align(16))
void InterpolateRow_Unaligned_SSE2(uint8* dst_ptr, const uint8* src_ptr,
                                   ptrdiff_t src_stride, int dst_width,
                                   int source_y_fraction) {
  __asm {
    push       esi
    push       edi
    mov        edi, [esp + 8 + 4]   // dst_ptr
    mov        esi, [esp + 8 + 8]   // src_ptr
    mov        edx, [esp + 8 + 12]  // src_stride
    mov        ecx, [esp + 8 + 16]  // dst_width
    mov        eax, [esp + 8 + 20]  // source_y_fraction (0..255)
    sub        edi, esi
    // Dispatch to specialized filters if applicable.
    cmp        eax, 0
    je         xloop100  // 0 / 256.  Blend 100 / 0.
    cmp        eax, 64
    je         xloop75   // 64 / 256 is 0.25.  Blend 75 / 25.
    cmp        eax, 128
    je         xloop50   // 128 / 256 is 0.50.  Blend 50 / 50.
    cmp        eax, 192
    je         xloop25   // 192 / 256 is 0.75.  Blend 25 / 75.

    movd       xmm5, eax            // xmm5 = y fraction
    punpcklbw  xmm5, xmm5
    psrlw      xmm5, 1
    punpcklwd  xmm5, xmm5
    punpckldq  xmm5, xmm5
    punpcklqdq xmm5, xmm5
    pxor       xmm4, xmm4

    align      4
  xloop:
    movdqu     xmm0, [esi]  // row0
    movdqu     xmm2, [esi + edx]  // row1
    movdqu     xmm1, xmm0
    movdqu     xmm3, xmm2
    punpcklbw  xmm2, xmm4
    punpckhbw  xmm3, xmm4
    punpcklbw  xmm0, xmm4
    punpckhbw  xmm1, xmm4
    psubw      xmm2, xmm0  // row1 - row0
    psubw      xmm3, xmm1
    paddw      xmm2, xmm2  // 9 bits * 15 bits = 8.16
    paddw      xmm3, xmm3
    pmulhw     xmm2, xmm5  // scale diff
    pmulhw     xmm3, xmm5
    paddw      xmm0, xmm2  // sum rows
    paddw      xmm1, xmm3
    packuswb   xmm0, xmm1
    sub        ecx, 16
    movdqu     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop
    jmp        xloop99

    // Blend 25 / 75.
    align      4
  xloop25:
    movdqu     xmm0, [esi]
    movdqu     xmm1, [esi + edx]
    pavgb      xmm0, xmm1
    pavgb      xmm0, xmm1
    sub        ecx, 16
    movdqu     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop25
    jmp        xloop99

    // Blend 50 / 50.
    align      4
  xloop50:
    movdqu     xmm0, [esi]
    movdqu     xmm1, [esi + edx]
    pavgb      xmm0, xmm1
    sub        ecx, 16
    movdqu     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop50
    jmp        xloop99

    // Blend 75 / 25.
    align      4
  xloop75:
    movdqu     xmm1, [esi]
    movdqu     xmm0, [esi + edx]
    pavgb      xmm0, xmm1
    pavgb      xmm0, xmm1
    sub        ecx, 16
    movdqu     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop75
    jmp        xloop99

    // Blend 100 / 0 - Copy row unchanged.
    align      4
  xloop100:
    movdqu     xmm0, [esi]
    sub        ecx, 16
    movdqu     [esi + edi], xmm0
    lea        esi, [esi + 16]
    jg         xloop100

  xloop99:
    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_INTERPOLATEROW_SSE2

__declspec(naked) __declspec(align(16))
void HalfRow_SSE2(const uint8* src_uv, int src_uv_stride,
                  uint8* dst_uv, int pix) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]    // src_uv
    mov        edx, [esp + 4 + 8]    // src_uv_stride
    mov        edi, [esp + 4 + 12]   // dst_v
    mov        ecx, [esp + 4 + 16]   // pix
    sub        edi, eax

    align      4
  convertloop:
    movdqa     xmm0, [eax]
    pavgb      xmm0, [eax + edx]
    sub        ecx, 16
    movdqa     [eax + edi], xmm0
    lea        eax,  [eax + 16]
    jg         convertloop
    pop        edi
    ret
  }
}

#ifdef HAS_HALFROW_AVX2
__declspec(naked) __declspec(align(16))
void HalfRow_AVX2(const uint8* src_uv, int src_uv_stride,
                  uint8* dst_uv, int pix) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]    // src_uv
    mov        edx, [esp + 4 + 8]    // src_uv_stride
    mov        edi, [esp + 4 + 12]   // dst_v
    mov        ecx, [esp + 4 + 16]   // pix
    sub        edi, eax

    align      4
  convertloop:
    vmovdqu    ymm0, [eax]
    vpavgb     ymm0, ymm0, [eax + edx]
    sub        ecx, 32
    vmovdqu    [eax + edi], ymm0
    lea        eax,  [eax + 32]
    jg         convertloop

    pop        edi
    vzeroupper
    ret
  }
}
#endif  // HAS_HALFROW_AVX2

__declspec(naked) __declspec(align(16))
void ARGBToBayerRow_SSSE3(const uint8* src_argb, uint8* dst_bayer,
                          uint32 selector, int pix) {
  __asm {
    mov        eax, [esp + 4]    // src_argb
    mov        edx, [esp + 8]    // dst_bayer
    movd       xmm5, [esp + 12]  // selector
    mov        ecx, [esp + 16]   // pix
    pshufd     xmm5, xmm5, 0

    align      4
  wloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax, [eax + 32]
    pshufb     xmm0, xmm5
    pshufb     xmm1, xmm5
    punpckldq  xmm0, xmm1
    sub        ecx, 8
    movq       qword ptr [edx], xmm0
    lea        edx, [edx + 8]
    jg         wloop
    ret
  }
}

// Specialized ARGB to Bayer that just isolates G channel.
__declspec(naked) __declspec(align(16))
void ARGBToBayerGGRow_SSE2(const uint8* src_argb, uint8* dst_bayer,
                           uint32 selector, int pix) {
  __asm {
    mov        eax, [esp + 4]    // src_argb
    mov        edx, [esp + 8]    // dst_bayer
                                 // selector
    mov        ecx, [esp + 16]   // pix
    pcmpeqb    xmm5, xmm5        // generate mask 0x000000ff
    psrld      xmm5, 24

    align      4
  wloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax, [eax + 32]
    psrld      xmm0, 8  // Move green to bottom.
    psrld      xmm1, 8
    pand       xmm0, xmm5
    pand       xmm1, xmm5
    packssdw   xmm0, xmm1
    packuswb   xmm0, xmm1
    sub        ecx, 8
    movq       qword ptr [edx], xmm0
    lea        edx, [edx + 8]
    jg         wloop
    ret
  }
}

// For BGRAToARGB, ABGRToARGB, RGBAToARGB, and ARGBToRGBA.
__declspec(naked) __declspec(align(16))
void ARGBShuffleRow_SSSE3(const uint8* src_argb, uint8* dst_argb,
                          const uint8* shuffler, int pix) {
  __asm {
    mov        eax, [esp + 4]    // src_argb
    mov        edx, [esp + 8]    // dst_argb
    mov        ecx, [esp + 12]   // shuffler
    movdqa     xmm5, [ecx]
    mov        ecx, [esp + 16]   // pix

    align      4
  wloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax, [eax + 32]
    pshufb     xmm0, xmm5
    pshufb     xmm1, xmm5
    sub        ecx, 8
    movdqa     [edx], xmm0
    movdqa     [edx + 16], xmm1
    lea        edx, [edx + 32]
    jg         wloop
    ret
  }
}

__declspec(naked) __declspec(align(16))
void ARGBShuffleRow_Unaligned_SSSE3(const uint8* src_argb, uint8* dst_argb,
                                    const uint8* shuffler, int pix) {
  __asm {
    mov        eax, [esp + 4]    // src_argb
    mov        edx, [esp + 8]    // dst_argb
    mov        ecx, [esp + 12]   // shuffler
    movdqa     xmm5, [ecx]
    mov        ecx, [esp + 16]   // pix

    align      4
  wloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax, [eax + 32]
    pshufb     xmm0, xmm5
    pshufb     xmm1, xmm5
    sub        ecx, 8
    movdqu     [edx], xmm0
    movdqu     [edx + 16], xmm1
    lea        edx, [edx + 32]
    jg         wloop
    ret
  }
}

#ifdef HAS_ARGBSHUFFLEROW_AVX2
__declspec(naked) __declspec(align(16))
void ARGBShuffleRow_AVX2(const uint8* src_argb, uint8* dst_argb,
                         const uint8* shuffler, int pix) {
  __asm {
    mov        eax, [esp + 4]     // src_argb
    mov        edx, [esp + 8]     // dst_argb
    mov        ecx, [esp + 12]    // shuffler
    vbroadcastf128 ymm5, [ecx]    // same shuffle in high as low.
    mov        ecx, [esp + 16]    // pix

    align      4
  wloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    lea        eax, [eax + 64]
    vpshufb    ymm0, ymm0, ymm5
    vpshufb    ymm1, ymm1, ymm5
    sub        ecx, 16
    vmovdqu    [edx], ymm0
    vmovdqu    [edx + 32], ymm1
    lea        edx, [edx + 64]
    jg         wloop

    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBSHUFFLEROW_AVX2

__declspec(naked) __declspec(align(16))
void ARGBShuffleRow_SSE2(const uint8* src_argb, uint8* dst_argb,
                         const uint8* shuffler, int pix) {
  __asm {
    push       ebx
    push       esi
    mov        eax, [esp + 8 + 4]    // src_argb
    mov        edx, [esp + 8 + 8]    // dst_argb
    mov        esi, [esp + 8 + 12]   // shuffler
    mov        ecx, [esp + 8 + 16]   // pix
    pxor       xmm5, xmm5

    mov        ebx, [esi]   // shuffler
    cmp        ebx, 0x03000102
    je         shuf_3012
    cmp        ebx, 0x00010203
    je         shuf_0123
    cmp        ebx, 0x00030201
    je         shuf_0321
    cmp        ebx, 0x02010003
    je         shuf_2103

  // TODO(fbarchard): Use one source pointer and 3 offsets.
  shuf_any1:
    movzx      ebx, byte ptr [esi]
    movzx      ebx, byte ptr [eax + ebx]
    mov        [edx], bl
    movzx      ebx, byte ptr [esi + 1]
    movzx      ebx, byte ptr [eax + ebx]
    mov        [edx + 1], bl
    movzx      ebx, byte ptr [esi + 2]
    movzx      ebx, byte ptr [eax + ebx]
    mov        [edx + 2], bl
    movzx      ebx, byte ptr [esi + 3]
    movzx      ebx, byte ptr [eax + ebx]
    mov        [edx + 3], bl
    lea        eax, [eax + 4]
    lea        edx, [edx + 4]
    sub        ecx, 1
    jg         shuf_any1
    jmp        shuf99

    align      4
  shuf_0123:
    movdqu     xmm0, [eax]
    lea        eax, [eax + 16]
    movdqa     xmm1, xmm0
    punpcklbw  xmm0, xmm5
    punpckhbw  xmm1, xmm5
    pshufhw    xmm0, xmm0, 01Bh   // 1B = 00011011 = 0x0123 = BGRAToARGB
    pshuflw    xmm0, xmm0, 01Bh
    pshufhw    xmm1, xmm1, 01Bh
    pshuflw    xmm1, xmm1, 01Bh
    packuswb   xmm0, xmm1
    sub        ecx, 4
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         shuf_0123
    jmp        shuf99

    align      4
  shuf_0321:
    movdqu     xmm0, [eax]
    lea        eax, [eax + 16]
    movdqa     xmm1, xmm0
    punpcklbw  xmm0, xmm5
    punpckhbw  xmm1, xmm5
    pshufhw    xmm0, xmm0, 039h   // 39 = 00111001 = 0x0321 = RGBAToARGB
    pshuflw    xmm0, xmm0, 039h
    pshufhw    xmm1, xmm1, 039h
    pshuflw    xmm1, xmm1, 039h
    packuswb   xmm0, xmm1
    sub        ecx, 4
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         shuf_0321
    jmp        shuf99

    align      4
  shuf_2103:
    movdqu     xmm0, [eax]
    lea        eax, [eax + 16]
    movdqa     xmm1, xmm0
    punpcklbw  xmm0, xmm5
    punpckhbw  xmm1, xmm5
    pshufhw    xmm0, xmm0, 093h   // 93 = 10010011 = 0x2103 = ARGBToRGBA
    pshuflw    xmm0, xmm0, 093h
    pshufhw    xmm1, xmm1, 093h
    pshuflw    xmm1, xmm1, 093h
    packuswb   xmm0, xmm1
    sub        ecx, 4
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         shuf_2103
    jmp        shuf99

    align      4
  shuf_3012:
    movdqu     xmm0, [eax]
    lea        eax, [eax + 16]
    movdqa     xmm1, xmm0
    punpcklbw  xmm0, xmm5
    punpckhbw  xmm1, xmm5
    pshufhw    xmm0, xmm0, 0C6h   // C6 = 11000110 = 0x3012 = ABGRToARGB
    pshuflw    xmm0, xmm0, 0C6h
    pshufhw    xmm1, xmm1, 0C6h
    pshuflw    xmm1, xmm1, 0C6h
    packuswb   xmm0, xmm1
    sub        ecx, 4
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         shuf_3012

  shuf99:
    pop        esi
    pop        ebx
    ret
  }
}

// YUY2 - Macro-pixel = 2 image pixels
// Y0U0Y1V0....Y2U2Y3V2...Y4U4Y5V4....

// UYVY - Macro-pixel = 2 image pixels
// U0Y0V0Y1

__declspec(naked) __declspec(align(16))
void I422ToYUY2Row_SSE2(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_frame, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]    // src_y
    mov        esi, [esp + 8 + 8]    // src_u
    mov        edx, [esp + 8 + 12]   // src_v
    mov        edi, [esp + 8 + 16]   // dst_frame
    mov        ecx, [esp + 8 + 20]   // width
    sub        edx, esi

    align      4
  convertloop:
    movq       xmm2, qword ptr [esi] // U
    movq       xmm3, qword ptr [esi + edx] // V
    lea        esi, [esi + 8]
    punpcklbw  xmm2, xmm3 // UV
    movdqu     xmm0, [eax] // Y
    lea        eax, [eax + 16]
    movdqa     xmm1, xmm0
    punpcklbw  xmm0, xmm2 // YUYV
    punpckhbw  xmm1, xmm2
    movdqu     [edi], xmm0
    movdqu     [edi + 16], xmm1
    lea        edi, [edi + 32]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) __declspec(align(16))
void I422ToUYVYRow_SSE2(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_frame, int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]    // src_y
    mov        esi, [esp + 8 + 8]    // src_u
    mov        edx, [esp + 8 + 12]   // src_v
    mov        edi, [esp + 8 + 16]   // dst_frame
    mov        ecx, [esp + 8 + 20]   // width
    sub        edx, esi

    align      4
  convertloop:
    movq       xmm2, qword ptr [esi] // U
    movq       xmm3, qword ptr [esi + edx] // V
    lea        esi, [esi + 8]
    punpcklbw  xmm2, xmm3 // UV
    movdqu     xmm0, [eax] // Y
    movdqa     xmm1, xmm2
    lea        eax, [eax + 16]
    punpcklbw  xmm1, xmm0 // UYVY
    punpckhbw  xmm2, xmm0
    movdqu     [edi], xmm1
    movdqu     [edi + 16], xmm2
    lea        edi, [edi + 32]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

#ifdef HAS_ARGBPOLYNOMIALROW_SSE2
__declspec(naked) __declspec(align(16))
void ARGBPolynomialRow_SSE2(const uint8* src_argb,
                            uint8* dst_argb, const float* poly,
                            int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   /* src_argb */
    mov        edx, [esp + 4 + 8]   /* dst_argb */
    mov        esi, [esp + 4 + 12]  /* poly */
    mov        ecx, [esp + 4 + 16]  /* width */
    pxor       xmm3, xmm3  // 0 constant for zero extending bytes to ints.

    // 2 pixel loop.
    align      4
 convertloop:
//    pmovzxbd  xmm0, dword ptr [eax]  // BGRA pixel
//    pmovzxbd  xmm4, dword ptr [eax + 4]  // BGRA pixel
    movq       xmm0, qword ptr [eax]  // BGRABGRA
    lea        eax, [eax + 8]
    punpcklbw  xmm0, xmm3
    movdqa     xmm4, xmm0
    punpcklwd  xmm0, xmm3  // pixel 0
    punpckhwd  xmm4, xmm3  // pixel 1
    cvtdq2ps   xmm0, xmm0  // 4 floats
    cvtdq2ps   xmm4, xmm4
    movdqa     xmm1, xmm0  // X
    movdqa     xmm5, xmm4
    mulps      xmm0, [esi + 16]  // C1 * X
    mulps      xmm4, [esi + 16]
    addps      xmm0, [esi]  // result = C0 + C1 * X
    addps      xmm4, [esi]
    movdqa     xmm2, xmm1
    movdqa     xmm6, xmm5
    mulps      xmm2, xmm1  // X * X
    mulps      xmm6, xmm5
    mulps      xmm1, xmm2  // X * X * X
    mulps      xmm5, xmm6
    mulps      xmm2, [esi + 32]  // C2 * X * X
    mulps      xmm6, [esi + 32]
    mulps      xmm1, [esi + 48]  // C3 * X * X * X
    mulps      xmm5, [esi + 48]
    addps      xmm0, xmm2  // result += C2 * X * X
    addps      xmm4, xmm6
    addps      xmm0, xmm1  // result += C3 * X * X * X
    addps      xmm4, xmm5
    cvttps2dq  xmm0, xmm0
    cvttps2dq  xmm4, xmm4
    packuswb   xmm0, xmm4
    packuswb   xmm0, xmm0
    sub        ecx, 2
    movq       qword ptr [edx], xmm0
    lea        edx, [edx + 8]
    jg         convertloop
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBPOLYNOMIALROW_SSE2

#ifdef HAS_ARGBPOLYNOMIALROW_AVX2
__declspec(naked) __declspec(align(16))
void ARGBPolynomialRow_AVX2(const uint8* src_argb,
                            uint8* dst_argb, const float* poly,
                            int width) {
  __asm {
    mov        eax, [esp + 4]   /* src_argb */
    mov        edx, [esp + 8]   /* dst_argb */
    mov        ecx, [esp + 12]   /* poly */
    vbroadcastf128 ymm4, [ecx]       // C0
    vbroadcastf128 ymm5, [ecx + 16]  // C1
    vbroadcastf128 ymm6, [ecx + 32]  // C2
    vbroadcastf128 ymm7, [ecx + 48]  // C3
    mov        ecx, [esp + 16]  /* width */

    // 2 pixel loop.
    align      4
 convertloop:
    vpmovzxbd   ymm0, qword ptr [eax]  // 2 BGRA pixels
    lea         eax, [eax + 8]
    vcvtdq2ps   ymm0, ymm0        // X 8 floats
    vmulps      ymm2, ymm0, ymm0  // X * X
    vmulps      ymm3, ymm0, ymm7  // C3 * X
    vfmadd132ps ymm0, ymm4, ymm5  // result = C0 + C1 * X
    vfmadd231ps ymm0, ymm2, ymm6  // result += C2 * X * X
    vfmadd231ps ymm0, ymm2, ymm3  // result += C3 * X * X * X
    vcvttps2dq  ymm0, ymm0
    vpackusdw   ymm0, ymm0, ymm0  // b0g0r0a0_00000000_b0g0r0a0_00000000
    vpermq      ymm0, ymm0, 0xd8  // b0g0r0a0_b0g0r0a0_00000000_00000000
    vpackuswb   xmm0, xmm0, xmm0  // bgrabgra_00000000_00000000_00000000
    sub         ecx, 2
    vmovq       qword ptr [edx], xmm0
    lea         edx, [edx + 8]
    jg          convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBPOLYNOMIALROW_AVX2

#ifdef HAS_ARGBCOLORTABLEROW_X86
// Tranform ARGB pixels with color table.
__declspec(naked) __declspec(align(16))
void ARGBColorTableRow_X86(uint8* dst_argb, const uint8* table_argb,
                           int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   /* dst_argb */
    mov        esi, [esp + 4 + 8]   /* table_argb */
    mov        ecx, [esp + 4 + 12]  /* width */

    // 1 pixel loop.
    align      4
  convertloop:
    movzx      edx, byte ptr [eax]
    lea        eax, [eax + 4]
    movzx      edx, byte ptr [esi + edx * 4]
    mov        byte ptr [eax - 4], dl
    movzx      edx, byte ptr [eax - 4 + 1]
    movzx      edx, byte ptr [esi + edx * 4 + 1]
    mov        byte ptr [eax - 4 + 1], dl
    movzx      edx, byte ptr [eax - 4 + 2]
    movzx      edx, byte ptr [esi + edx * 4 + 2]
    mov        byte ptr [eax - 4 + 2], dl
    movzx      edx, byte ptr [eax - 4 + 3]
    movzx      edx, byte ptr [esi + edx * 4 + 3]
    mov        byte ptr [eax - 4 + 3], dl
    dec        ecx
    jg         convertloop
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBCOLORTABLEROW_X86

#ifdef HAS_RGBCOLORTABLEROW_X86
// Tranform RGB pixels with color table.
__declspec(naked) __declspec(align(16))
void RGBColorTableRow_X86(uint8* dst_argb, const uint8* table_argb, int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]   /* dst_argb */
    mov        esi, [esp + 4 + 8]   /* table_argb */
    mov        ecx, [esp + 4 + 12]  /* width */

    // 1 pixel loop.
    align      4
  convertloop:
    movzx      edx, byte ptr [eax]
    lea        eax, [eax + 4]
    movzx      edx, byte ptr [esi + edx * 4]
    mov        byte ptr [eax - 4], dl
    movzx      edx, byte ptr [eax - 4 + 1]
    movzx      edx, byte ptr [esi + edx * 4 + 1]
    mov        byte ptr [eax - 4 + 1], dl
    movzx      edx, byte ptr [eax - 4 + 2]
    movzx      edx, byte ptr [esi + edx * 4 + 2]
    mov        byte ptr [eax - 4 + 2], dl
    dec        ecx
    jg         convertloop

    pop        esi
    ret
  }
}
#endif  // HAS_RGBCOLORTABLEROW_X86

#ifdef HAS_ARGBLUMACOLORTABLEROW_SSSE3
// Tranform RGB pixels with luma table.
__declspec(naked) __declspec(align(16))
void ARGBLumaColorTableRow_SSSE3(const uint8* src_argb, uint8* dst_argb,
                                 int width,
                                 const uint8* luma, uint32 lumacoeff) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]   /* src_argb */
    mov        edi, [esp + 8 + 8]   /* dst_argb */
    mov        ecx, [esp + 8 + 12]  /* width */
    movd       xmm2, dword ptr [esp + 8 + 16]  // luma table
    movd       xmm3, dword ptr [esp + 8 + 20]  // lumacoeff
    pshufd     xmm2, xmm2, 0
    pshufd     xmm3, xmm3, 0
    pcmpeqb    xmm4, xmm4        // generate mask 0xff00ff00
    psllw      xmm4, 8
    pxor       xmm5, xmm5

    // 4 pixel loop.
    align      4
  convertloop:
    movdqu     xmm0, qword ptr [eax]      // generate luma ptr
    pmaddubsw  xmm0, xmm3
    phaddw     xmm0, xmm0
    pand       xmm0, xmm4  // mask out low bits
    punpcklwd  xmm0, xmm5
    paddd      xmm0, xmm2  // add table base
    movd       esi, xmm0
    pshufd     xmm0, xmm0, 0x39  // 00111001 to rotate right 32

    movzx      edx, byte ptr [eax]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi], dl
    movzx      edx, byte ptr [eax + 1]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 1], dl
    movzx      edx, byte ptr [eax + 2]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 2], dl
    movzx      edx, byte ptr [eax + 3]  // copy alpha.
    mov        byte ptr [edi + 3], dl

    movd       esi, xmm0
    pshufd     xmm0, xmm0, 0x39  // 00111001 to rotate right 32

    movzx      edx, byte ptr [eax + 4]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 4], dl
    movzx      edx, byte ptr [eax + 5]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 5], dl
    movzx      edx, byte ptr [eax + 6]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 6], dl
    movzx      edx, byte ptr [eax + 7]  // copy alpha.
    mov        byte ptr [edi + 7], dl

    movd       esi, xmm0
    pshufd     xmm0, xmm0, 0x39  // 00111001 to rotate right 32

    movzx      edx, byte ptr [eax + 8]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 8], dl
    movzx      edx, byte ptr [eax + 9]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 9], dl
    movzx      edx, byte ptr [eax + 10]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 10], dl
    movzx      edx, byte ptr [eax + 11]  // copy alpha.
    mov        byte ptr [edi + 11], dl

    movd       esi, xmm0

    movzx      edx, byte ptr [eax + 12]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 12], dl
    movzx      edx, byte ptr [eax + 13]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 13], dl
    movzx      edx, byte ptr [eax + 14]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 14], dl
    movzx      edx, byte ptr [eax + 15]  // copy alpha.
    mov        byte ptr [edi + 15], dl

    sub        ecx, 4
    lea        eax, [eax + 16]
    lea        edi, [edi + 16]
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBLUMACOLORTABLEROW_SSSE3

#endif  // !defined(LIBYUV_DISABLE_X86) && defined(_M_IX86) && defined(_MSC_VER)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
