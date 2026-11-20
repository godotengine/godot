/*
 *  Copyright 2013 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/rotate_row.h"
#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// This module is for 32 bit Visual C x86 and clangcl
#if !defined(LIBYUV_DISABLE_X86) && defined(_M_IX86) && defined(_MSC_VER)

__declspec(naked) void TransposeWx8_SSSE3(const uint8_t* src,
                                          int src_stride,
                                          uint8_t* dst,
                                          int dst_stride,
                                          int width) {
  __asm {
    push      edi
    push      esi
    push      ebp
    mov       eax, [esp + 12 + 4]  // src
    mov       edi, [esp + 12 + 8]  // src_stride
    mov       edx, [esp + 12 + 12]  // dst
    mov       esi, [esp + 12 + 16]  // dst_stride
    mov       ecx, [esp + 12 + 20]  // width

    // Read in the data from the source pointer.
    // First round of bit swap.
    align      4
 convertloop:
    movq      xmm0, qword ptr [eax]
    lea       ebp, [eax + 8]
    movq      xmm1, qword ptr [eax + edi]
    lea       eax, [eax + 2 * edi]
    punpcklbw xmm0, xmm1
    movq      xmm2, qword ptr [eax]
    movdqa    xmm1, xmm0
    palignr   xmm1, xmm1, 8
    movq      xmm3, qword ptr [eax + edi]
    lea       eax, [eax + 2 * edi]
    punpcklbw xmm2, xmm3
    movdqa    xmm3, xmm2
    movq      xmm4, qword ptr [eax]
    palignr   xmm3, xmm3, 8
    movq      xmm5, qword ptr [eax + edi]
    punpcklbw xmm4, xmm5
    lea       eax, [eax + 2 * edi]
    movdqa    xmm5, xmm4
    movq      xmm6, qword ptr [eax]
    palignr   xmm5, xmm5, 8
    movq      xmm7, qword ptr [eax + edi]
    punpcklbw xmm6, xmm7
    mov       eax, ebp
    movdqa    xmm7, xmm6
    palignr   xmm7, xmm7, 8
    // Second round of bit swap.
    punpcklwd xmm0, xmm2
    punpcklwd xmm1, xmm3
    movdqa    xmm2, xmm0
    movdqa    xmm3, xmm1
    palignr   xmm2, xmm2, 8
    palignr   xmm3, xmm3, 8
    punpcklwd xmm4, xmm6
    punpcklwd xmm5, xmm7
    movdqa    xmm6, xmm4
    movdqa    xmm7, xmm5
    palignr   xmm6, xmm6, 8
    palignr   xmm7, xmm7, 8
    // Third round of bit swap.
    // Write to the destination pointer.
    punpckldq xmm0, xmm4
    movq      qword ptr [edx], xmm0
    movdqa    xmm4, xmm0
    palignr   xmm4, xmm4, 8
    movq      qword ptr [edx + esi], xmm4
    lea       edx, [edx + 2 * esi]
    punpckldq xmm2, xmm6
    movdqa    xmm6, xmm2
    palignr   xmm6, xmm6, 8
    movq      qword ptr [edx], xmm2
    punpckldq xmm1, xmm5
    movq      qword ptr [edx + esi], xmm6
    lea       edx, [edx + 2 * esi]
    movdqa    xmm5, xmm1
    movq      qword ptr [edx], xmm1
    palignr   xmm5, xmm5, 8
    punpckldq xmm3, xmm7
    movq      qword ptr [edx + esi], xmm5
    lea       edx, [edx + 2 * esi]
    movq      qword ptr [edx], xmm3
    movdqa    xmm7, xmm3
    palignr   xmm7, xmm7, 8
    sub       ecx, 8
    movq      qword ptr [edx + esi], xmm7
    lea       edx, [edx + 2 * esi]
    jg        convertloop

    pop       ebp
    pop       esi
    pop       edi
    ret
  }
}

__declspec(naked) void TransposeUVWx8_SSE2(const uint8_t* src,
                                           int src_stride,
                                           uint8_t* dst_a,
                                           int dst_stride_a,
                                           uint8_t* dst_b,
                                           int dst_stride_b,
                                           int w) {
  __asm {
    push      ebx
    push      esi
    push      edi
    push      ebp
    mov       eax, [esp + 16 + 4]  // src
    mov       edi, [esp + 16 + 8]  // src_stride
    mov       edx, [esp + 16 + 12]  // dst_a
    mov       esi, [esp + 16 + 16]  // dst_stride_a
    mov       ebx, [esp + 16 + 20]  // dst_b
    mov       ebp, [esp + 16 + 24]  // dst_stride_b
    mov       ecx, esp
    sub       esp, 4 + 16
    and       esp, ~15
    mov       [esp + 16], ecx
    mov       ecx, [ecx + 16 + 28]  // w

    align      4
    // Read in the data from the source pointer.
    // First round of bit swap.
  convertloop:
    movdqu    xmm0, [eax]
    movdqu    xmm1, [eax + edi]
    lea       eax, [eax + 2 * edi]
    movdqa    xmm7, xmm0  // use xmm7 as temp register.
    punpcklbw xmm0, xmm1
    punpckhbw xmm7, xmm1
    movdqa    xmm1, xmm7
    movdqu    xmm2, [eax]
    movdqu    xmm3, [eax + edi]
    lea       eax, [eax + 2 * edi]
    movdqa    xmm7, xmm2
    punpcklbw xmm2, xmm3
    punpckhbw xmm7, xmm3
    movdqa    xmm3, xmm7
    movdqu    xmm4, [eax]
    movdqu    xmm5, [eax + edi]
    lea       eax, [eax + 2 * edi]
    movdqa    xmm7, xmm4
    punpcklbw xmm4, xmm5
    punpckhbw xmm7, xmm5
    movdqa    xmm5, xmm7
    movdqu    xmm6, [eax]
    movdqu    xmm7, [eax + edi]
    lea       eax, [eax + 2 * edi]
    movdqu    [esp], xmm5  // backup xmm5
    neg       edi
    movdqa    xmm5, xmm6  // use xmm5 as temp register.
    punpcklbw xmm6, xmm7
    punpckhbw xmm5, xmm7
    movdqa    xmm7, xmm5
    lea       eax, [eax + 8 * edi + 16]
    neg       edi
        // Second round of bit swap.
    movdqa    xmm5, xmm0
    punpcklwd xmm0, xmm2
    punpckhwd xmm5, xmm2
    movdqa    xmm2, xmm5
    movdqa    xmm5, xmm1
    punpcklwd xmm1, xmm3
    punpckhwd xmm5, xmm3
    movdqa    xmm3, xmm5
    movdqa    xmm5, xmm4
    punpcklwd xmm4, xmm6
    punpckhwd xmm5, xmm6
    movdqa    xmm6, xmm5
    movdqu    xmm5, [esp]  // restore xmm5
    movdqu    [esp], xmm6  // backup xmm6
    movdqa    xmm6, xmm5  // use xmm6 as temp register.
    punpcklwd xmm5, xmm7
    punpckhwd xmm6, xmm7
    movdqa    xmm7, xmm6

        // Third round of bit swap.
        // Write to the destination pointer.
    movdqa    xmm6, xmm0
    punpckldq xmm0, xmm4
    punpckhdq xmm6, xmm4
    movdqa    xmm4, xmm6
    movdqu    xmm6, [esp]  // restore xmm6
    movlpd    qword ptr [edx], xmm0
    movhpd    qword ptr [ebx], xmm0
    movlpd    qword ptr [edx + esi], xmm4
    lea       edx, [edx + 2 * esi]
    movhpd    qword ptr [ebx + ebp], xmm4
    lea       ebx, [ebx + 2 * ebp]
    movdqa    xmm0, xmm2  // use xmm0 as the temp register.
    punpckldq xmm2, xmm6
    movlpd    qword ptr [edx], xmm2
    movhpd    qword ptr [ebx], xmm2
    punpckhdq xmm0, xmm6
    movlpd    qword ptr [edx + esi], xmm0
    lea       edx, [edx + 2 * esi]
    movhpd    qword ptr [ebx + ebp], xmm0
    lea       ebx, [ebx + 2 * ebp]
    movdqa    xmm0, xmm1  // use xmm0 as the temp register.
    punpckldq xmm1, xmm5
    movlpd    qword ptr [edx], xmm1
    movhpd    qword ptr [ebx], xmm1
    punpckhdq xmm0, xmm5
    movlpd    qword ptr [edx + esi], xmm0
    lea       edx, [edx + 2 * esi]
    movhpd    qword ptr [ebx + ebp], xmm0
    lea       ebx, [ebx + 2 * ebp]
    movdqa    xmm0, xmm3  // use xmm0 as the temp register.
    punpckldq xmm3, xmm7
    movlpd    qword ptr [edx], xmm3
    movhpd    qword ptr [ebx], xmm3
    punpckhdq xmm0, xmm7
    sub       ecx, 8
    movlpd    qword ptr [edx + esi], xmm0
    lea       edx, [edx + 2 * esi]
    movhpd    qword ptr [ebx + ebp], xmm0
    lea       ebx, [ebx + 2 * ebp]
    jg        convertloop

    mov       esp, [esp + 16]
    pop       ebp
    pop       edi
    pop       esi
    pop       ebx
    ret
  }
}

#endif  // !defined(LIBYUV_DISABLE_X86) && defined(_M_IX86)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
