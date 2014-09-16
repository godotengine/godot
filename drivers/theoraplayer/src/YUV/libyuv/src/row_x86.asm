;
; Copyright 2012 The LibYuv Project Authors. All rights reserved.
;
; Use of this source code is governed by a BSD-style license
; that can be found in the LICENSE file in the root of the source
; tree. An additional intellectual property rights grant can be found
; in the file PATENTS. All contributing project authors may
; be found in the AUTHORS file in the root of the source tree.
;

%ifdef __YASM_VERSION_ID__
%if __YASM_VERSION_ID__ < 01020000h
%error AVX2 is supported only by yasm 1.2.0 or later.
%endif
%endif
%include "x86inc.asm"

SECTION .text

; cglobal numeric constants are parameters, gpr regs, mm regs

; void YUY2ToYRow_SSE2(const uint8* src_yuy2, uint8* dst_y, int pix)

%macro YUY2TOYROW 2-3
cglobal %1ToYRow%3, 3, 3, 3, src_yuy2, dst_y, pix
%ifidn %1,YUY2
    pcmpeqb    m2, m2, m2        ; generate mask 0x00ff00ff
    psrlw      m2, m2, 8
%endif

    ALIGN      4
.convertloop:
    mov%2      m0, [src_yuy2q]
    mov%2      m1, [src_yuy2q + mmsize]
    lea        src_yuy2q, [src_yuy2q + mmsize * 2]
%ifidn %1,YUY2
    pand       m0, m0, m2   ; YUY2 even bytes are Y
    pand       m1, m1, m2
%else
    psrlw      m0, m0, 8    ; UYVY odd bytes are Y
    psrlw      m1, m1, 8
%endif
    packuswb   m0, m0, m1
%if cpuflag(AVX2)
    vpermq     m0, m0, 0xd8
%endif
    sub        pixd, mmsize
    mov%2      [dst_yq], m0
    lea        dst_yq, [dst_yq + mmsize]
    jg         .convertloop
    REP_RET
%endmacro

; TODO(fbarchard): Remove MMX.  Add SSSE3 pshufb version.
INIT_MMX MMX
YUY2TOYROW YUY2,a,
YUY2TOYROW YUY2,u,_Unaligned
YUY2TOYROW UYVY,a,
YUY2TOYROW UYVY,u,_Unaligned
INIT_XMM SSE2
YUY2TOYROW YUY2,a,
YUY2TOYROW YUY2,u,_Unaligned
YUY2TOYROW UYVY,a,
YUY2TOYROW UYVY,u,_Unaligned
INIT_YMM AVX2
YUY2TOYROW YUY2,a,
YUY2TOYROW UYVY,a,

; void SplitUVRow_SSE2(const uint8* src_uv, uint8* dst_u, uint8* dst_v, int pix)

%macro SplitUVRow 1-2
cglobal SplitUVRow%2, 4, 4, 5, src_uv, dst_u, dst_v, pix
    pcmpeqb    m4, m4, m4        ; generate mask 0x00ff00ff
    psrlw      m4, m4, 8
    sub        dst_vq, dst_uq

    ALIGN      4
.convertloop:
    mov%1      m0, [src_uvq]
    mov%1      m1, [src_uvq + mmsize]
    lea        src_uvq, [src_uvq + mmsize * 2]
    psrlw      m2, m0, 8         ; odd bytes
    psrlw      m3, m1, 8
    pand       m0, m0, m4        ; even bytes
    pand       m1, m1, m4
    packuswb   m0, m0, m1
    packuswb   m2, m2, m3
%if cpuflag(AVX2)
    vpermq     m0, m0, 0xd8
    vpermq     m2, m2, 0xd8
%endif
    mov%1      [dst_uq], m0
    mov%1      [dst_uq + dst_vq], m2
    lea        dst_uq, [dst_uq + mmsize]
    sub        pixd, mmsize
    jg         .convertloop
    REP_RET
%endmacro

INIT_MMX MMX
SplitUVRow a,
SplitUVRow u,_Unaligned
INIT_XMM SSE2
SplitUVRow a,
SplitUVRow u,_Unaligned
INIT_YMM AVX2
SplitUVRow a,

; void MergeUVRow_SSE2(const uint8* src_u, const uint8* src_v, uint8* dst_uv,
;                      int width);

%macro MergeUVRow_ 1-2
cglobal MergeUVRow_%2, 4, 4, 3, src_u, src_v, dst_uv, pix
    sub        src_vq, src_uq

    ALIGN      4
.convertloop:
    mov%1      m0, [src_uq]
    mov%1      m1, [src_vq]
    lea        src_uq, [src_uq + mmsize]
    punpcklbw  m2, m0, m1       // first 8 UV pairs
    punpckhbw  m0, m0, m1       // next 8 UV pairs
%if cpuflag(AVX2)
    vperm2i128 m1, m2, m0, 0x20  // low 128 of ymm2 and low 128 of ymm0
    vperm2i128 m2, m2, m0, 0x31  // high 128 of ymm2 and high 128 of ymm0
    mov%1      [dst_uvq], m1
    mov%1      [dst_uvq + mmsize], m2
%else
    mov%1      [dst_uvq], m2
    mov%1      [dst_uvq + mmsize], m0
%endif
    lea        dst_uvq, [dst_uvq + mmsize * 2]
    sub        pixd, mmsize
    jg         .convertloop
    REP_RET
%endmacro

INIT_MMX MMX
MergeUVRow_ a,
MergeUVRow_ u,_Unaligned
INIT_XMM SSE2
MergeUVRow_ a,
MergeUVRow_ u,_Unaligned
INIT_YMM AVX2
MergeUVRow_ a,

