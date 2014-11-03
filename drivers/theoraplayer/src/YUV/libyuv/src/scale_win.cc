/*
 *  Copyright 2013 The LibYuv Project Authors. All rights reserved.
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

// Offsets for source bytes 0 to 9
static uvec8 kShuf0 =
  { 0, 1, 3, 4, 5, 7, 8, 9, 128, 128, 128, 128, 128, 128, 128, 128 };

// Offsets for source bytes 11 to 20 with 8 subtracted = 3 to 12.
static uvec8 kShuf1 =
  { 3, 4, 5, 7, 8, 9, 11, 12, 128, 128, 128, 128, 128, 128, 128, 128 };

// Offsets for source bytes 21 to 31 with 16 subtracted = 5 to 31.
static uvec8 kShuf2 =
  { 5, 7, 8, 9, 11, 12, 13, 15, 128, 128, 128, 128, 128, 128, 128, 128 };

// Offsets for source bytes 0 to 10
static uvec8 kShuf01 =
  { 0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9, 9, 10 };

// Offsets for source bytes 10 to 21 with 8 subtracted = 3 to 13.
static uvec8 kShuf11 =
  { 2, 3, 4, 5, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 12, 13 };

// Offsets for source bytes 21 to 31 with 16 subtracted = 5 to 31.
static uvec8 kShuf21 =
  { 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 12, 13, 13, 14, 14, 15 };

// Coefficients for source bytes 0 to 10
static uvec8 kMadd01 =
  { 3, 1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3, 1, 2, 2 };

// Coefficients for source bytes 10 to 21
static uvec8 kMadd11 =
  { 1, 3, 3, 1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3, 1 };

// Coefficients for source bytes 21 to 31
static uvec8 kMadd21 =
  { 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3 };

// Coefficients for source bytes 21 to 31
static vec16 kRound34 =
  { 2, 2, 2, 2, 2, 2, 2, 2 };

static uvec8 kShuf38a =
  { 0, 3, 6, 8, 11, 14, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 };

static uvec8 kShuf38b =
  { 128, 128, 128, 128, 128, 128, 0, 3, 6, 8, 11, 14, 128, 128, 128, 128 };

// Arrange words 0,3,6 into 0,1,2
static uvec8 kShufAc =
  { 0, 1, 6, 7, 12, 13, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 };

// Arrange words 0,3,6 into 3,4,5
static uvec8 kShufAc3 =
  { 128, 128, 128, 128, 128, 128, 0, 1, 6, 7, 12, 13, 128, 128, 128, 128 };

// Scaling values for boxes of 3x3 and 2x3
static uvec16 kScaleAc33 =
  { 65536 / 9, 65536 / 9, 65536 / 6, 65536 / 9, 65536 / 9, 65536 / 6, 0, 0 };

// Arrange first value for pixels 0,1,2,3,4,5
static uvec8 kShufAb0 =
  { 0, 128, 3, 128, 6, 128, 8, 128, 11, 128, 14, 128, 128, 128, 128, 128 };

// Arrange second value for pixels 0,1,2,3,4,5
static uvec8 kShufAb1 =
  { 1, 128, 4, 128, 7, 128, 9, 128, 12, 128, 15, 128, 128, 128, 128, 128 };

// Arrange third value for pixels 0,1,2,3,4,5
static uvec8 kShufAb2 =
  { 2, 128, 5, 128, 128, 128, 10, 128, 13, 128, 128, 128, 128, 128, 128, 128 };

// Scaling values for boxes of 3x2 and 2x2
static uvec16 kScaleAb2 =
  { 65536 / 3, 65536 / 3, 65536 / 2, 65536 / 3, 65536 / 3, 65536 / 2, 0, 0 };

// Reads 32 pixels, throws half away and writes 16 pixels.
// Alignment requirement: src_ptr 16 byte aligned, dst_ptr 16 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleRowDown2_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst_ptr, int dst_width) {
  __asm {
    mov        eax, [esp + 4]        // src_ptr
                                     // src_stride ignored
    mov        edx, [esp + 12]       // dst_ptr
    mov        ecx, [esp + 16]       // dst_width

    align      4
  wloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    psrlw      xmm0, 8               // isolate odd pixels.
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    sub        ecx, 16
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         wloop

    ret
  }
}

// Blends 32x1 rectangle to 16x1.
// Alignment requirement: src_ptr 16 byte aligned, dst_ptr 16 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleRowDown2Linear_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                              uint8* dst_ptr, int dst_width) {
  __asm {
    mov        eax, [esp + 4]        // src_ptr
                                     // src_stride
    mov        edx, [esp + 12]       // dst_ptr
    mov        ecx, [esp + 16]       // dst_width
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff00ff
    psrlw      xmm5, 8

    align      4
  wloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax,  [eax + 32]

    movdqa     xmm2, xmm0            // average columns (32 to 16 pixels)
    psrlw      xmm0, 8
    movdqa     xmm3, xmm1
    psrlw      xmm1, 8
    pand       xmm2, xmm5
    pand       xmm3, xmm5
    pavgw      xmm0, xmm2
    pavgw      xmm1, xmm3
    packuswb   xmm0, xmm1

    sub        ecx, 16
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         wloop

    ret
  }
}

// Blends 32x2 rectangle to 16x1.
// Alignment requirement: src_ptr 16 byte aligned, dst_ptr 16 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleRowDown2Box_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst_ptr, int dst_width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]    // src_ptr
    mov        esi, [esp + 4 + 8]    // src_stride
    mov        edx, [esp + 4 + 12]   // dst_ptr
    mov        ecx, [esp + 4 + 16]   // dst_width
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff00ff
    psrlw      xmm5, 8

    align      4
  wloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + esi]
    movdqa     xmm3, [eax + esi + 16]
    lea        eax,  [eax + 32]
    pavgb      xmm0, xmm2            // average rows
    pavgb      xmm1, xmm3

    movdqa     xmm2, xmm0            // average columns (32 to 16 pixels)
    psrlw      xmm0, 8
    movdqa     xmm3, xmm1
    psrlw      xmm1, 8
    pand       xmm2, xmm5
    pand       xmm3, xmm5
    pavgw      xmm0, xmm2
    pavgw      xmm1, xmm3
    packuswb   xmm0, xmm1

    sub        ecx, 16
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         wloop

    pop        esi
    ret
  }
}

// Reads 32 pixels, throws half away and writes 16 pixels.
// Alignment requirement: src_ptr 16 byte aligned, dst_ptr 16 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleRowDown2_Unaligned_SSE2(const uint8* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8* dst_ptr, int dst_width) {
  __asm {
    mov        eax, [esp + 4]        // src_ptr
                                     // src_stride ignored
    mov        edx, [esp + 12]       // dst_ptr
    mov        ecx, [esp + 16]       // dst_width

    align      4
  wloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    psrlw      xmm0, 8               // isolate odd pixels.
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    sub        ecx, 16
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         wloop

    ret
  }
}

// Blends 32x1 rectangle to 16x1.
// Alignment requirement: src_ptr 16 byte aligned, dst_ptr 16 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleRowDown2Linear_Unaligned_SSE2(const uint8* src_ptr,
                                        ptrdiff_t src_stride,
                                        uint8* dst_ptr, int dst_width) {
  __asm {
    mov        eax, [esp + 4]        // src_ptr
                                     // src_stride
    mov        edx, [esp + 12]       // dst_ptr
    mov        ecx, [esp + 16]       // dst_width
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff00ff
    psrlw      xmm5, 8

    align      4
  wloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax,  [eax + 32]

    movdqa     xmm2, xmm0            // average columns (32 to 16 pixels)
    psrlw      xmm0, 8
    movdqa     xmm3, xmm1
    psrlw      xmm1, 8
    pand       xmm2, xmm5
    pand       xmm3, xmm5
    pavgw      xmm0, xmm2
    pavgw      xmm1, xmm3
    packuswb   xmm0, xmm1

    sub        ecx, 16
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         wloop

    ret
  }
}

// Blends 32x2 rectangle to 16x1.
// Alignment requirement: src_ptr 16 byte aligned, dst_ptr 16 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleRowDown2Box_Unaligned_SSE2(const uint8* src_ptr,
                                     ptrdiff_t src_stride,
                                     uint8* dst_ptr, int dst_width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]    // src_ptr
    mov        esi, [esp + 4 + 8]    // src_stride
    mov        edx, [esp + 4 + 12]   // dst_ptr
    mov        ecx, [esp + 4 + 16]   // dst_width
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff00ff
    psrlw      xmm5, 8

    align      4
  wloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + esi]
    movdqu     xmm3, [eax + esi + 16]
    lea        eax,  [eax + 32]
    pavgb      xmm0, xmm2            // average rows
    pavgb      xmm1, xmm3

    movdqa     xmm2, xmm0            // average columns (32 to 16 pixels)
    psrlw      xmm0, 8
    movdqa     xmm3, xmm1
    psrlw      xmm1, 8
    pand       xmm2, xmm5
    pand       xmm3, xmm5
    pavgw      xmm0, xmm2
    pavgw      xmm1, xmm3
    packuswb   xmm0, xmm1

    sub        ecx, 16
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    jg         wloop

    pop        esi
    ret
  }
}

// Point samples 32 pixels to 8 pixels.
// Alignment requirement: src_ptr 16 byte aligned, dst_ptr 8 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleRowDown4_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst_ptr, int dst_width) {
  __asm {
    mov        eax, [esp + 4]        // src_ptr
                                     // src_stride ignored
    mov        edx, [esp + 12]       // dst_ptr
    mov        ecx, [esp + 16]       // dst_width
    pcmpeqb    xmm5, xmm5            // generate mask 0x00ff0000
    psrld      xmm5, 24
    pslld      xmm5, 16

    align      4
  wloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    pand       xmm0, xmm5
    pand       xmm1, xmm5
    packuswb   xmm0, xmm1
    psrlw      xmm0, 8
    packuswb   xmm0, xmm0
    sub        ecx, 8
    movq       qword ptr [edx], xmm0
    lea        edx, [edx + 8]
    jg         wloop

    ret
  }
}

// Blends 32x4 rectangle to 8x1.
// Alignment requirement: src_ptr 16 byte aligned, dst_ptr 8 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleRowDown4Box_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst_ptr, int dst_width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]    // src_ptr
    mov        esi, [esp + 8 + 8]    // src_stride
    mov        edx, [esp + 8 + 12]   // dst_ptr
    mov        ecx, [esp + 8 + 16]   // dst_width
    lea        edi, [esi + esi * 2]  // src_stride * 3
    pcmpeqb    xmm7, xmm7            // generate mask 0x00ff00ff
    psrlw      xmm7, 8

    align      4
  wloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + esi]
    movdqa     xmm3, [eax + esi + 16]
    pavgb      xmm0, xmm2            // average rows
    pavgb      xmm1, xmm3
    movdqa     xmm2, [eax + esi * 2]
    movdqa     xmm3, [eax + esi * 2 + 16]
    movdqa     xmm4, [eax + edi]
    movdqa     xmm5, [eax + edi + 16]
    lea        eax, [eax + 32]
    pavgb      xmm2, xmm4
    pavgb      xmm3, xmm5
    pavgb      xmm0, xmm2
    pavgb      xmm1, xmm3

    movdqa     xmm2, xmm0            // average columns (32 to 16 pixels)
    psrlw      xmm0, 8
    movdqa     xmm3, xmm1
    psrlw      xmm1, 8
    pand       xmm2, xmm7
    pand       xmm3, xmm7
    pavgw      xmm0, xmm2
    pavgw      xmm1, xmm3
    packuswb   xmm0, xmm1

    movdqa     xmm2, xmm0            // average columns (16 to 8 pixels)
    psrlw      xmm0, 8
    pand       xmm2, xmm7
    pavgw      xmm0, xmm2
    packuswb   xmm0, xmm0

    sub        ecx, 8
    movq       qword ptr [edx], xmm0
    lea        edx, [edx + 8]
    jg         wloop

    pop        edi
    pop        esi
    ret
  }
}

// Point samples 32 pixels to 24 pixels.
// Produces three 8 byte values. For each 8 bytes, 16 bytes are read.
// Then shuffled to do the scaling.

// Note that movdqa+palign may be better than movdqu.
// Alignment requirement: src_ptr 16 byte aligned, dst_ptr 8 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleRowDown34_SSSE3(const uint8* src_ptr, ptrdiff_t src_stride,
                          uint8* dst_ptr, int dst_width) {
  __asm {
    mov        eax, [esp + 4]        // src_ptr
                                     // src_stride ignored
    mov        edx, [esp + 12]       // dst_ptr
    mov        ecx, [esp + 16]       // dst_width
    movdqa     xmm3, kShuf0
    movdqa     xmm4, kShuf1
    movdqa     xmm5, kShuf2

    align      4
  wloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    movdqa     xmm2, xmm1
    palignr    xmm1, xmm0, 8
    pshufb     xmm0, xmm3
    pshufb     xmm1, xmm4
    pshufb     xmm2, xmm5
    movq       qword ptr [edx], xmm0
    movq       qword ptr [edx + 8], xmm1
    movq       qword ptr [edx + 16], xmm2
    lea        edx, [edx + 24]
    sub        ecx, 24
    jg         wloop

    ret
  }
}

// Blends 32x2 rectangle to 24x1
// Produces three 8 byte values. For each 8 bytes, 16 bytes are read.
// Then shuffled to do the scaling.

// Register usage:
// xmm0 src_row 0
// xmm1 src_row 1
// xmm2 shuf 0
// xmm3 shuf 1
// xmm4 shuf 2
// xmm5 madd 0
// xmm6 madd 1
// xmm7 kRound34

// Note that movdqa+palign may be better than movdqu.
// Alignment requirement: src_ptr 16 byte aligned, dst_ptr 8 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleRowDown34_1_Box_SSSE3(const uint8* src_ptr,
                                ptrdiff_t src_stride,
                                uint8* dst_ptr, int dst_width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]    // src_ptr
    mov        esi, [esp + 4 + 8]    // src_stride
    mov        edx, [esp + 4 + 12]   // dst_ptr
    mov        ecx, [esp + 4 + 16]   // dst_width
    movdqa     xmm2, kShuf01
    movdqa     xmm3, kShuf11
    movdqa     xmm4, kShuf21
    movdqa     xmm5, kMadd01
    movdqa     xmm6, kMadd11
    movdqa     xmm7, kRound34

    align      4
  wloop:
    movdqa     xmm0, [eax]           // pixels 0..7
    movdqa     xmm1, [eax + esi]
    pavgb      xmm0, xmm1
    pshufb     xmm0, xmm2
    pmaddubsw  xmm0, xmm5
    paddsw     xmm0, xmm7
    psrlw      xmm0, 2
    packuswb   xmm0, xmm0
    movq       qword ptr [edx], xmm0
    movdqu     xmm0, [eax + 8]       // pixels 8..15
    movdqu     xmm1, [eax + esi + 8]
    pavgb      xmm0, xmm1
    pshufb     xmm0, xmm3
    pmaddubsw  xmm0, xmm6
    paddsw     xmm0, xmm7
    psrlw      xmm0, 2
    packuswb   xmm0, xmm0
    movq       qword ptr [edx + 8], xmm0
    movdqa     xmm0, [eax + 16]      // pixels 16..23
    movdqa     xmm1, [eax + esi + 16]
    lea        eax, [eax + 32]
    pavgb      xmm0, xmm1
    pshufb     xmm0, xmm4
    movdqa     xmm1, kMadd21
    pmaddubsw  xmm0, xmm1
    paddsw     xmm0, xmm7
    psrlw      xmm0, 2
    packuswb   xmm0, xmm0
    sub        ecx, 24
    movq       qword ptr [edx + 16], xmm0
    lea        edx, [edx + 24]
    jg         wloop

    pop        esi
    ret
  }
}

// Note that movdqa+palign may be better than movdqu.
// Alignment requirement: src_ptr 16 byte aligned, dst_ptr 8 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleRowDown34_0_Box_SSSE3(const uint8* src_ptr,
                                ptrdiff_t src_stride,
                                uint8* dst_ptr, int dst_width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]    // src_ptr
    mov        esi, [esp + 4 + 8]    // src_stride
    mov        edx, [esp + 4 + 12]   // dst_ptr
    mov        ecx, [esp + 4 + 16]   // dst_width
    movdqa     xmm2, kShuf01
    movdqa     xmm3, kShuf11
    movdqa     xmm4, kShuf21
    movdqa     xmm5, kMadd01
    movdqa     xmm6, kMadd11
    movdqa     xmm7, kRound34

    align      4
  wloop:
    movdqa     xmm0, [eax]           // pixels 0..7
    movdqa     xmm1, [eax + esi]
    pavgb      xmm1, xmm0
    pavgb      xmm0, xmm1
    pshufb     xmm0, xmm2
    pmaddubsw  xmm0, xmm5
    paddsw     xmm0, xmm7
    psrlw      xmm0, 2
    packuswb   xmm0, xmm0
    movq       qword ptr [edx], xmm0
    movdqu     xmm0, [eax + 8]       // pixels 8..15
    movdqu     xmm1, [eax + esi + 8]
    pavgb      xmm1, xmm0
    pavgb      xmm0, xmm1
    pshufb     xmm0, xmm3
    pmaddubsw  xmm0, xmm6
    paddsw     xmm0, xmm7
    psrlw      xmm0, 2
    packuswb   xmm0, xmm0
    movq       qword ptr [edx + 8], xmm0
    movdqa     xmm0, [eax + 16]      // pixels 16..23
    movdqa     xmm1, [eax + esi + 16]
    lea        eax, [eax + 32]
    pavgb      xmm1, xmm0
    pavgb      xmm0, xmm1
    pshufb     xmm0, xmm4
    movdqa     xmm1, kMadd21
    pmaddubsw  xmm0, xmm1
    paddsw     xmm0, xmm7
    psrlw      xmm0, 2
    packuswb   xmm0, xmm0
    sub        ecx, 24
    movq       qword ptr [edx + 16], xmm0
    lea        edx, [edx+24]
    jg         wloop

    pop        esi
    ret
  }
}

// 3/8 point sampler

// Scale 32 pixels to 12
__declspec(naked) __declspec(align(16))
void ScaleRowDown38_SSSE3(const uint8* src_ptr, ptrdiff_t src_stride,
                          uint8* dst_ptr, int dst_width) {
  __asm {
    mov        eax, [esp + 4]        // src_ptr
                                     // src_stride ignored
    mov        edx, [esp + 12]       // dst_ptr
    mov        ecx, [esp + 16]       // dst_width
    movdqa     xmm4, kShuf38a
    movdqa     xmm5, kShuf38b

    align      4
  xloop:
    movdqa     xmm0, [eax]           // 16 pixels -> 0,1,2,3,4,5
    movdqa     xmm1, [eax + 16]      // 16 pixels -> 6,7,8,9,10,11
    lea        eax, [eax + 32]
    pshufb     xmm0, xmm4
    pshufb     xmm1, xmm5
    paddusb    xmm0, xmm1

    sub        ecx, 12
    movq       qword ptr [edx], xmm0  // write 12 pixels
    movhlps    xmm1, xmm0
    movd       [edx + 8], xmm1
    lea        edx, [edx + 12]
    jg         xloop

    ret
  }
}

// Scale 16x3 pixels to 6x1 with interpolation
__declspec(naked) __declspec(align(16))
void ScaleRowDown38_3_Box_SSSE3(const uint8* src_ptr,
                                ptrdiff_t src_stride,
                                uint8* dst_ptr, int dst_width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]    // src_ptr
    mov        esi, [esp + 4 + 8]    // src_stride
    mov        edx, [esp + 4 + 12]   // dst_ptr
    mov        ecx, [esp + 4 + 16]   // dst_width
    movdqa     xmm2, kShufAc
    movdqa     xmm3, kShufAc3
    movdqa     xmm4, kScaleAc33
    pxor       xmm5, xmm5

    align      4
  xloop:
    movdqa     xmm0, [eax]           // sum up 3 rows into xmm0/1
    movdqa     xmm6, [eax + esi]
    movhlps    xmm1, xmm0
    movhlps    xmm7, xmm6
    punpcklbw  xmm0, xmm5
    punpcklbw  xmm1, xmm5
    punpcklbw  xmm6, xmm5
    punpcklbw  xmm7, xmm5
    paddusw    xmm0, xmm6
    paddusw    xmm1, xmm7
    movdqa     xmm6, [eax + esi * 2]
    lea        eax, [eax + 16]
    movhlps    xmm7, xmm6
    punpcklbw  xmm6, xmm5
    punpcklbw  xmm7, xmm5
    paddusw    xmm0, xmm6
    paddusw    xmm1, xmm7

    movdqa     xmm6, xmm0            // 8 pixels -> 0,1,2 of xmm6
    psrldq     xmm0, 2
    paddusw    xmm6, xmm0
    psrldq     xmm0, 2
    paddusw    xmm6, xmm0
    pshufb     xmm6, xmm2

    movdqa     xmm7, xmm1            // 8 pixels -> 3,4,5 of xmm6
    psrldq     xmm1, 2
    paddusw    xmm7, xmm1
    psrldq     xmm1, 2
    paddusw    xmm7, xmm1
    pshufb     xmm7, xmm3
    paddusw    xmm6, xmm7

    pmulhuw    xmm6, xmm4            // divide by 9,9,6, 9,9,6
    packuswb   xmm6, xmm6

    sub        ecx, 6
    movd       [edx], xmm6           // write 6 pixels
    psrlq      xmm6, 16
    movd       [edx + 2], xmm6
    lea        edx, [edx + 6]
    jg         xloop

    pop        esi
    ret
  }
}

// Scale 16x2 pixels to 6x1 with interpolation
__declspec(naked) __declspec(align(16))
void ScaleRowDown38_2_Box_SSSE3(const uint8* src_ptr,
                                ptrdiff_t src_stride,
                                uint8* dst_ptr, int dst_width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]    // src_ptr
    mov        esi, [esp + 4 + 8]    // src_stride
    mov        edx, [esp + 4 + 12]   // dst_ptr
    mov        ecx, [esp + 4 + 16]   // dst_width
    movdqa     xmm2, kShufAb0
    movdqa     xmm3, kShufAb1
    movdqa     xmm4, kShufAb2
    movdqa     xmm5, kScaleAb2

    align      4
  xloop:
    movdqa     xmm0, [eax]           // average 2 rows into xmm0
    pavgb      xmm0, [eax + esi]
    lea        eax, [eax + 16]

    movdqa     xmm1, xmm0            // 16 pixels -> 0,1,2,3,4,5 of xmm1
    pshufb     xmm1, xmm2
    movdqa     xmm6, xmm0
    pshufb     xmm6, xmm3
    paddusw    xmm1, xmm6
    pshufb     xmm0, xmm4
    paddusw    xmm1, xmm0

    pmulhuw    xmm1, xmm5            // divide by 3,3,2, 3,3,2
    packuswb   xmm1, xmm1

    sub        ecx, 6
    movd       [edx], xmm1           // write 6 pixels
    psrlq      xmm1, 16
    movd       [edx + 2], xmm1
    lea        edx, [edx + 6]
    jg         xloop

    pop        esi
    ret
  }
}

// Reads 16xN bytes and produces 16 shorts at a time.
// TODO(fbarchard): Make this handle 4xN bytes for any width ARGB.
__declspec(naked) __declspec(align(16))
void ScaleAddRows_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                       uint16* dst_ptr, int src_width,
                       int src_height) {
  __asm {
    push       esi
    push       edi
    push       ebx
    push       ebp
    mov        esi, [esp + 16 + 4]   // src_ptr
    mov        edx, [esp + 16 + 8]   // src_stride
    mov        edi, [esp + 16 + 12]  // dst_ptr
    mov        ecx, [esp + 16 + 16]  // dst_width
    mov        ebx, [esp + 16 + 20]  // height
    pxor       xmm4, xmm4
    dec        ebx

    align      4
  xloop:
    // first row
    movdqa     xmm0, [esi]
    lea        eax, [esi + edx]
    movdqa     xmm1, xmm0
    punpcklbw  xmm0, xmm4
    punpckhbw  xmm1, xmm4
    lea        esi, [esi + 16]
    mov        ebp, ebx
    test       ebp, ebp
    je         ydone

    // sum remaining rows
    align      4
  yloop:
    movdqa     xmm2, [eax]       // read 16 pixels
    lea        eax, [eax + edx]  // advance to next row
    movdqa     xmm3, xmm2
    punpcklbw  xmm2, xmm4
    punpckhbw  xmm3, xmm4
    paddusw    xmm0, xmm2        // sum 16 words
    paddusw    xmm1, xmm3
    sub        ebp, 1
    jg         yloop

    align      4
  ydone:
    movdqa     [edi], xmm0
    movdqa     [edi + 16], xmm1
    lea        edi, [edi + 32]

    sub        ecx, 16
    jg         xloop

    pop        ebp
    pop        ebx
    pop        edi
    pop        esi
    ret
  }
}

// Bilinear column filtering. SSSE3 version.
// TODO(fbarchard): Port to Neon
// TODO(fbarchard): Switch the following:
//    xor        ebx, ebx
//    mov        bx, word ptr [esi + eax]  // 2 source x0 pixels
// To
//    movzx      ebx, word ptr [esi + eax]  // 2 source x0 pixels
// when drmemory bug fixed.
// https://code.google.com/p/drmemory/issues/detail?id=1396

__declspec(naked) __declspec(align(16))
void ScaleFilterCols_SSSE3(uint8* dst_ptr, const uint8* src_ptr,
                           int dst_width, int x, int dx) {
  __asm {
    push       ebx
    push       esi
    push       edi
    mov        edi, [esp + 12 + 4]    // dst_ptr
    mov        esi, [esp + 12 + 8]    // src_ptr
    mov        ecx, [esp + 12 + 12]   // dst_width
    movd       xmm2, [esp + 12 + 16]  // x
    movd       xmm3, [esp + 12 + 20]  // dx
    mov        eax, 0x04040000      // shuffle to line up fractions with pixel.
    movd       xmm5, eax
    pcmpeqb    xmm6, xmm6           // generate 0x007f for inverting fraction.
    psrlw      xmm6, 9
    pextrw     eax, xmm2, 1         // get x0 integer. preroll
    sub        ecx, 2
    jl         xloop29

    movdqa     xmm0, xmm2           // x1 = x0 + dx
    paddd      xmm0, xmm3
    punpckldq  xmm2, xmm0           // x0 x1
    punpckldq  xmm3, xmm3           // dx dx
    paddd      xmm3, xmm3           // dx * 2, dx * 2
    pextrw     edx, xmm2, 3         // get x1 integer. preroll

    // 2 Pixel loop.
    align      4
  xloop2:
    movdqa     xmm1, xmm2           // x0, x1 fractions.
    paddd      xmm2, xmm3           // x += dx
    movzx      ebx, word ptr [esi + eax]  // 2 source x0 pixels
    movd       xmm0, ebx
    psrlw      xmm1, 9              // 7 bit fractions.
    movzx      ebx, word ptr [esi + edx]  // 2 source x1 pixels
    movd       xmm4, ebx
    pshufb     xmm1, xmm5           // 0011
    punpcklwd  xmm0, xmm4
    pxor       xmm1, xmm6           // 0..7f and 7f..0
    pmaddubsw  xmm0, xmm1           // 16 bit, 2 pixels.
    pextrw     eax, xmm2, 1         // get x0 integer. next iteration.
    pextrw     edx, xmm2, 3         // get x1 integer. next iteration.
    psrlw      xmm0, 7              // 8.7 fixed point to low 8 bits.
    packuswb   xmm0, xmm0           // 8 bits, 2 pixels.
    movd       ebx, xmm0
    mov        [edi], bx
    lea        edi, [edi + 2]
    sub        ecx, 2               // 2 pixels
    jge        xloop2

    align      4
 xloop29:

    add        ecx, 2 - 1
    jl         xloop99

    // 1 pixel remainder
    movzx      ebx, word ptr [esi + eax]  // 2 source x0 pixels
    movd       xmm0, ebx
    psrlw      xmm2, 9              // 7 bit fractions.
    pshufb     xmm2, xmm5           // 0011
    pxor       xmm2, xmm6           // 0..7f and 7f..0
    pmaddubsw  xmm0, xmm2           // 16 bit
    psrlw      xmm0, 7              // 8.7 fixed point to low 8 bits.
    packuswb   xmm0, xmm0           // 8 bits
    movd       ebx, xmm0
    mov        [edi], bl

    align      4
 xloop99:

    pop        edi
    pop        esi
    pop        ebx
    ret
  }
}

// Reads 16 pixels, duplicates them and writes 32 pixels.
// Alignment requirement: src_argb 16 byte aligned, dst_argb 16 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleColsUp2_SSE2(uint8* dst_ptr, const uint8* src_ptr,
                       int dst_width, int x, int dx) {
  __asm {
    mov        edx, [esp + 4]    // dst_ptr
    mov        eax, [esp + 8]    // src_ptr
    mov        ecx, [esp + 12]   // dst_width

    align      4
  wloop:
    movdqa     xmm0, [eax]
    lea        eax,  [eax + 16]
    movdqa     xmm1, xmm0
    punpcklbw  xmm0, xmm0
    punpckhbw  xmm1, xmm1
    sub        ecx, 32
    movdqa     [edx], xmm0
    movdqa     [edx + 16], xmm1
    lea        edx, [edx + 32]
    jg         wloop

    ret
  }
}

// Reads 8 pixels, throws half away and writes 4 even pixels (0, 2, 4, 6)
// Alignment requirement: src_argb 16 byte aligned, dst_argb 16 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleARGBRowDown2_SSE2(const uint8* src_argb,
                            ptrdiff_t src_stride,
                            uint8* dst_argb, int dst_width) {
  __asm {
    mov        eax, [esp + 4]        // src_argb
                                     // src_stride ignored
    mov        edx, [esp + 12]       // dst_argb
    mov        ecx, [esp + 16]       // dst_width

    align      4
  wloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    shufps     xmm0, xmm1, 0xdd
    sub        ecx, 4
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         wloop

    ret
  }
}

// Blends 8x1 rectangle to 4x1.
// Alignment requirement: src_argb 16 byte aligned, dst_argb 16 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleARGBRowDown2Linear_SSE2(const uint8* src_argb,
                                  ptrdiff_t src_stride,
                                  uint8* dst_argb, int dst_width) {
  __asm {
    mov        eax, [esp + 4]        // src_argb
                                     // src_stride ignored
    mov        edx, [esp + 12]       // dst_argb
    mov        ecx, [esp + 16]       // dst_width

    align      4
  wloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    movdqa     xmm2, xmm0
    shufps     xmm0, xmm1, 0x88      // even pixels
    shufps     xmm2, xmm1, 0xdd      // odd pixels
    pavgb      xmm0, xmm2
    sub        ecx, 4
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         wloop

    ret
  }
}

// Blends 8x2 rectangle to 4x1.
// Alignment requirement: src_argb 16 byte aligned, dst_argb 16 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleARGBRowDown2Box_SSE2(const uint8* src_argb,
                               ptrdiff_t src_stride,
                               uint8* dst_argb, int dst_width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]    // src_argb
    mov        esi, [esp + 4 + 8]    // src_stride
    mov        edx, [esp + 4 + 12]   // dst_argb
    mov        ecx, [esp + 4 + 16]   // dst_width

    align      4
  wloop:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    movdqa     xmm2, [eax + esi]
    movdqa     xmm3, [eax + esi + 16]
    lea        eax,  [eax + 32]
    pavgb      xmm0, xmm2            // average rows
    pavgb      xmm1, xmm3
    movdqa     xmm2, xmm0            // average columns (8 to 4 pixels)
    shufps     xmm0, xmm1, 0x88      // even pixels
    shufps     xmm2, xmm1, 0xdd      // odd pixels
    pavgb      xmm0, xmm2
    sub        ecx, 4
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         wloop

    pop        esi
    ret
  }
}

// Reads 4 pixels at a time.
// Alignment requirement: dst_argb 16 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleARGBRowDownEven_SSE2(const uint8* src_argb, ptrdiff_t src_stride,
                               int src_stepx,
                               uint8* dst_argb, int dst_width) {
  __asm {
    push       ebx
    push       edi
    mov        eax, [esp + 8 + 4]    // src_argb
                                     // src_stride ignored
    mov        ebx, [esp + 8 + 12]   // src_stepx
    mov        edx, [esp + 8 + 16]   // dst_argb
    mov        ecx, [esp + 8 + 20]   // dst_width
    lea        ebx, [ebx * 4]
    lea        edi, [ebx + ebx * 2]

    align      4
  wloop:
    movd       xmm0, [eax]
    movd       xmm1, [eax + ebx]
    punpckldq  xmm0, xmm1
    movd       xmm2, [eax + ebx * 2]
    movd       xmm3, [eax + edi]
    lea        eax,  [eax + ebx * 4]
    punpckldq  xmm2, xmm3
    punpcklqdq xmm0, xmm2
    sub        ecx, 4
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         wloop

    pop        edi
    pop        ebx
    ret
  }
}

// Blends four 2x2 to 4x1.
// Alignment requirement: dst_argb 16 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleARGBRowDownEvenBox_SSE2(const uint8* src_argb,
                                  ptrdiff_t src_stride,
                                  int src_stepx,
                                  uint8* dst_argb, int dst_width) {
  __asm {
    push       ebx
    push       esi
    push       edi
    mov        eax, [esp + 12 + 4]    // src_argb
    mov        esi, [esp + 12 + 8]    // src_stride
    mov        ebx, [esp + 12 + 12]   // src_stepx
    mov        edx, [esp + 12 + 16]   // dst_argb
    mov        ecx, [esp + 12 + 20]   // dst_width
    lea        esi, [eax + esi]       // row1 pointer
    lea        ebx, [ebx * 4]
    lea        edi, [ebx + ebx * 2]

    align      4
  wloop:
    movq       xmm0, qword ptr [eax]  // row0 4 pairs
    movhps     xmm0, qword ptr [eax + ebx]
    movq       xmm1, qword ptr [eax + ebx * 2]
    movhps     xmm1, qword ptr [eax + edi]
    lea        eax,  [eax + ebx * 4]
    movq       xmm2, qword ptr [esi]  // row1 4 pairs
    movhps     xmm2, qword ptr [esi + ebx]
    movq       xmm3, qword ptr [esi + ebx * 2]
    movhps     xmm3, qword ptr [esi + edi]
    lea        esi,  [esi + ebx * 4]
    pavgb      xmm0, xmm2            // average rows
    pavgb      xmm1, xmm3
    movdqa     xmm2, xmm0            // average columns (8 to 4 pixels)
    shufps     xmm0, xmm1, 0x88      // even pixels
    shufps     xmm2, xmm1, 0xdd      // odd pixels
    pavgb      xmm0, xmm2
    sub        ecx, 4
    movdqa     [edx], xmm0
    lea        edx, [edx + 16]
    jg         wloop

    pop        edi
    pop        esi
    pop        ebx
    ret
  }
}

// Column scaling unfiltered. SSE2 version.
__declspec(naked) __declspec(align(16))
void ScaleARGBCols_SSE2(uint8* dst_argb, const uint8* src_argb,
                        int dst_width, int x, int dx) {
  __asm {
    push       edi
    push       esi
    mov        edi, [esp + 8 + 4]    // dst_argb
    mov        esi, [esp + 8 + 8]    // src_argb
    mov        ecx, [esp + 8 + 12]   // dst_width
    movd       xmm2, [esp + 8 + 16]  // x
    movd       xmm3, [esp + 8 + 20]  // dx

    pshufd     xmm2, xmm2, 0         // x0 x0 x0 x0
    pshufd     xmm0, xmm3, 0x11      // dx  0 dx  0
    paddd      xmm2, xmm0
    paddd      xmm3, xmm3            // 0, 0, 0,  dx * 2
    pshufd     xmm0, xmm3, 0x05      // dx * 2, dx * 2, 0, 0
    paddd      xmm2, xmm0            // x3 x2 x1 x0
    paddd      xmm3, xmm3            // 0, 0, 0,  dx * 4
    pshufd     xmm3, xmm3, 0         // dx * 4, dx * 4, dx * 4, dx * 4

    pextrw     eax, xmm2, 1          // get x0 integer.
    pextrw     edx, xmm2, 3          // get x1 integer.

    cmp        ecx, 0
    jle        xloop99
    sub        ecx, 4
    jl         xloop49

    // 4 Pixel loop.
    align      4
 xloop4:
    movd       xmm0, [esi + eax * 4]  // 1 source x0 pixels
    movd       xmm1, [esi + edx * 4]  // 1 source x1 pixels
    pextrw     eax, xmm2, 5           // get x2 integer.
    pextrw     edx, xmm2, 7           // get x3 integer.
    paddd      xmm2, xmm3             // x += dx
    punpckldq  xmm0, xmm1             // x0 x1

    movd       xmm1, [esi + eax * 4]  // 1 source x2 pixels
    movd       xmm4, [esi + edx * 4]  // 1 source x3 pixels
    pextrw     eax, xmm2, 1           // get x0 integer. next iteration.
    pextrw     edx, xmm2, 3           // get x1 integer. next iteration.
    punpckldq  xmm1, xmm4             // x2 x3
    punpcklqdq xmm0, xmm1             // x0 x1 x2 x3
    sub        ecx, 4                 // 4 pixels
    movdqu     [edi], xmm0
    lea        edi, [edi + 16]
    jge        xloop4

    align      4
 xloop49:
    test       ecx, 2
    je         xloop29

    // 2 Pixels.
    movd       xmm0, [esi + eax * 4]  // 1 source x0 pixels
    movd       xmm1, [esi + edx * 4]  // 1 source x1 pixels
    pextrw     eax, xmm2, 5           // get x2 integer.
    punpckldq  xmm0, xmm1             // x0 x1

    movq       qword ptr [edi], xmm0
    lea        edi, [edi + 8]

 xloop29:
    test       ecx, 1
    je         xloop99

    // 1 Pixels.
    movd       xmm0, [esi + eax * 4]  // 1 source x2 pixels
    movd       dword ptr [edi], xmm0
    align      4
 xloop99:

    pop        esi
    pop        edi
    ret
  }
}

// Bilinear row filtering combines 2x1 -> 1x1. SSSE3 version.
// TODO(fbarchard): Port to Neon

// Shuffle table for arranging 2 pixels into pairs for pmaddubsw
static uvec8 kShuffleColARGB = {
  0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u,  // bbggrraa 1st pixel
  8u, 12u, 9u, 13u, 10u, 14u, 11u, 15u  // bbggrraa 2nd pixel
};

// Shuffle table for duplicating 2 fractions into 8 bytes each
static uvec8 kShuffleFractions = {
  0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u,
};

__declspec(naked) __declspec(align(16))
void ScaleARGBFilterCols_SSSE3(uint8* dst_argb, const uint8* src_argb,
                               int dst_width, int x, int dx) {
  __asm {
    push       esi
    push       edi
    mov        edi, [esp + 8 + 4]    // dst_argb
    mov        esi, [esp + 8 + 8]    // src_argb
    mov        ecx, [esp + 8 + 12]   // dst_width
    movd       xmm2, [esp + 8 + 16]  // x
    movd       xmm3, [esp + 8 + 20]  // dx
    movdqa     xmm4, kShuffleColARGB
    movdqa     xmm5, kShuffleFractions
    pcmpeqb    xmm6, xmm6           // generate 0x007f for inverting fraction.
    psrlw      xmm6, 9
    pextrw     eax, xmm2, 1         // get x0 integer. preroll
    sub        ecx, 2
    jl         xloop29

    movdqa     xmm0, xmm2           // x1 = x0 + dx
    paddd      xmm0, xmm3
    punpckldq  xmm2, xmm0           // x0 x1
    punpckldq  xmm3, xmm3           // dx dx
    paddd      xmm3, xmm3           // dx * 2, dx * 2
    pextrw     edx, xmm2, 3         // get x1 integer. preroll

    // 2 Pixel loop.
    align      4
  xloop2:
    movdqa     xmm1, xmm2           // x0, x1 fractions.
    paddd      xmm2, xmm3           // x += dx
    movq       xmm0, qword ptr [esi + eax * 4]  // 2 source x0 pixels
    psrlw      xmm1, 9              // 7 bit fractions.
    movhps     xmm0, qword ptr [esi + edx * 4]  // 2 source x1 pixels
    pshufb     xmm1, xmm5           // 0000000011111111
    pshufb     xmm0, xmm4           // arrange pixels into pairs
    pxor       xmm1, xmm6           // 0..7f and 7f..0
    pmaddubsw  xmm0, xmm1           // argb_argb 16 bit, 2 pixels.
    pextrw     eax, xmm2, 1         // get x0 integer. next iteration.
    pextrw     edx, xmm2, 3         // get x1 integer. next iteration.
    psrlw      xmm0, 7              // argb 8.7 fixed point to low 8 bits.
    packuswb   xmm0, xmm0           // argb_argb 8 bits, 2 pixels.
    movq       qword ptr [edi], xmm0
    lea        edi, [edi + 8]
    sub        ecx, 2               // 2 pixels
    jge        xloop2

    align      4
 xloop29:

    add        ecx, 2 - 1
    jl         xloop99

    // 1 pixel remainder
    psrlw      xmm2, 9              // 7 bit fractions.
    movq       xmm0, qword ptr [esi + eax * 4]  // 2 source x0 pixels
    pshufb     xmm2, xmm5           // 00000000
    pshufb     xmm0, xmm4           // arrange pixels into pairs
    pxor       xmm2, xmm6           // 0..7f and 7f..0
    pmaddubsw  xmm0, xmm2           // argb 16 bit, 1 pixel.
    psrlw      xmm0, 7
    packuswb   xmm0, xmm0           // argb 8 bits, 1 pixel.
    movd       [edi], xmm0

    align      4
 xloop99:

    pop        edi
    pop        esi
    ret
  }
}

// Reads 4 pixels, duplicates them and writes 8 pixels.
// Alignment requirement: src_argb 16 byte aligned, dst_argb 16 byte aligned.
__declspec(naked) __declspec(align(16))
void ScaleARGBColsUp2_SSE2(uint8* dst_argb, const uint8* src_argb,
                           int dst_width, int x, int dx) {
  __asm {
    mov        edx, [esp + 4]    // dst_argb
    mov        eax, [esp + 8]    // src_argb
    mov        ecx, [esp + 12]   // dst_width

    align      4
  wloop:
    movdqa     xmm0, [eax]
    lea        eax,  [eax + 16]
    movdqa     xmm1, xmm0
    punpckldq  xmm0, xmm0
    punpckhdq  xmm1, xmm1
    sub        ecx, 8
    movdqa     [edx], xmm0
    movdqa     [edx + 16], xmm1
    lea        edx, [edx + 32]
    jg         wloop

    ret
  }
}

// Divide num by div and return as 16.16 fixed point result.
__declspec(naked) __declspec(align(16))
int FixedDiv_X86(int num, int div) {
  __asm {
    mov        eax, [esp + 4]    // num
    cdq                          // extend num to 64 bits
    shld       edx, eax, 16      // 32.16
    shl        eax, 16
    idiv       dword ptr [esp + 8]
    ret
  }
}

// Divide num by div and return as 16.16 fixed point result.
__declspec(naked) __declspec(align(16))
int FixedDiv1_X86(int num, int div) {
  __asm {
    mov        eax, [esp + 4]    // num
    mov        ecx, [esp + 8]    // denom
    cdq                          // extend num to 64 bits
    shld       edx, eax, 16      // 32.16
    shl        eax, 16
    sub        eax, 0x00010001
    sbb        edx, 0
    sub        ecx, 1
    idiv       ecx
    ret
  }
}

#endif  // !defined(LIBYUV_DISABLE_X86) && defined(_M_IX86) && defined(_MSC_VER)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
