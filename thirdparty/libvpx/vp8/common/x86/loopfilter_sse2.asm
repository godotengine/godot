;
;  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;


%include "vpx_ports/x86_abi_support.asm"
%define _t0 0
%define _t1 _t0 + 16
%define _p3 _t1 + 16
%define _p2 _p3 + 16
%define _p1 _p2 + 16
%define _p0 _p1 + 16
%define _q0 _p0 + 16
%define _q1 _q0 + 16
%define _q2 _q1 + 16
%define _q3 _q2 + 16
%define lf_var_size 160

; Use of pmaxub instead of psubusb to compute filter mask was seen
; in ffvp8

%macro LFH_FILTER_AND_HEV_MASK 1
%if %1
        movdqa      xmm2,                   [rdi+2*rax]       ; q3
        movdqa      xmm1,                   [rsi+2*rax]       ; q2
        movdqa      xmm4,                   [rsi+rax]         ; q1
        movdqa      xmm5,                   [rsi]             ; q0
        neg         rax                     ; negate pitch to deal with above border
%else
        movlps      xmm2,                   [rsi + rcx*2]     ; q3
        movlps      xmm1,                   [rsi + rcx]       ; q2
        movlps      xmm4,                   [rsi]             ; q1
        movlps      xmm5,                   [rsi + rax]       ; q0

        movhps      xmm2,                   [rdi + rcx*2]
        movhps      xmm1,                   [rdi + rcx]
        movhps      xmm4,                   [rdi]
        movhps      xmm5,                   [rdi + rax]

        lea         rsi,                    [rsi + rax*4]
        lea         rdi,                    [rdi + rax*4]

        movdqa      [rsp+_q2],              xmm1              ; store q2
        movdqa      [rsp+_q1],              xmm4              ; store q1
%endif
        movdqa      xmm7,                   [rdx]             ;limit

        movdqa      xmm6,                   xmm1              ; q2
        movdqa      xmm3,                   xmm4              ; q1

        psubusb     xmm1,                   xmm2              ; q2-=q3
        psubusb     xmm2,                   xmm6              ; q3-=q2

        psubusb     xmm4,                   xmm6              ; q1-=q2
        psubusb     xmm6,                   xmm3              ; q2-=q1

        por         xmm4,                   xmm6              ; abs(q2-q1)
        por         xmm1,                   xmm2              ; abs(q3-q2)

        movdqa      xmm0,                   xmm5              ; q0
        pmaxub      xmm1,                   xmm4

        psubusb     xmm5,                   xmm3              ; q0-=q1
        psubusb     xmm3,                   xmm0              ; q1-=q0

        por         xmm5,                   xmm3              ; abs(q0-q1)
        movdqa      [rsp+_t0],              xmm5              ; save to t0

        pmaxub      xmm1,                   xmm5

%if %1
        movdqa      xmm2,                   [rsi+4*rax]       ; p3
        movdqa      xmm4,                   [rdi+4*rax]       ; p2
        movdqa      xmm6,                   [rsi+2*rax]       ; p1
%else
        movlps      xmm2,                   [rsi + rax]       ; p3
        movlps      xmm4,                   [rsi]             ; p2
        movlps      xmm6,                   [rsi + rcx]       ; p1

        movhps      xmm2,                   [rdi + rax]
        movhps      xmm4,                   [rdi]
        movhps      xmm6,                   [rdi + rcx]

        movdqa      [rsp+_p2],              xmm4              ; store p2
        movdqa      [rsp+_p1],              xmm6              ; store p1
%endif

        movdqa      xmm5,                   xmm4              ; p2
        movdqa      xmm3,                   xmm6              ; p1

        psubusb     xmm4,                   xmm2              ; p2-=p3
        psubusb     xmm2,                   xmm5              ; p3-=p2

        psubusb     xmm3,                   xmm5              ; p1-=p2
        pmaxub      xmm1,                   xmm4              ; abs(p3 - p2)

        psubusb     xmm5,                   xmm6              ; p2-=p1
        pmaxub      xmm1,                   xmm2              ; abs(p3 - p2)

        pmaxub      xmm1,                   xmm5              ; abs(p2 - p1)
        movdqa      xmm2,                   xmm6              ; p1

        pmaxub      xmm1,                   xmm3              ; abs(p2 - p1)
%if %1
        movdqa      xmm4,                   [rsi+rax]         ; p0
        movdqa      xmm3,                   [rdi]             ; q1
%else
        movlps      xmm4,                   [rsi + rcx*2]     ; p0
        movhps      xmm4,                   [rdi + rcx*2]
        movdqa      xmm3,                   [rsp+_q1]                ; q1
%endif

        movdqa      xmm5,                   xmm4              ; p0
        psubusb     xmm4,                   xmm6              ; p0-=p1

        psubusb     xmm6,                   xmm5              ; p1-=p0

        por         xmm6,                   xmm4              ; abs(p1 - p0)
        mov         rdx,                    arg(2)            ; get blimit

        movdqa     [rsp+_t1],               xmm6              ; save to t1

        movdqa      xmm4,                   xmm3              ; q1
        pmaxub      xmm1,                   xmm6

        psubusb     xmm3,                   xmm2              ; q1-=p1
        psubusb     xmm2,                   xmm4              ; p1-=q1

        psubusb     xmm1,                   xmm7
        por         xmm2,                   xmm3              ; abs(p1-q1)

        movdqa      xmm7,                   [rdx]             ; blimit
        mov         rdx,                    arg(4)            ; hev get thresh

        movdqa      xmm3,                   xmm0              ; q0
        pand        xmm2,                   [GLOBAL(tfe)]     ; set lsb of each byte to zero

        movdqa      xmm6,                   xmm5              ; p0
        psrlw       xmm2,                   1                 ; abs(p1-q1)/2

        psubusb     xmm5,                   xmm3              ; p0-=q0
        psubusb     xmm3,                   xmm6              ; q0-=p0
        por         xmm5,                   xmm3              ; abs(p0 - q0)

        paddusb     xmm5,                   xmm5              ; abs(p0-q0)*2

        movdqa      xmm4,                   [rsp+_t0]                ; hev get abs (q1 - q0)
        movdqa      xmm3,                   [rsp+_t1]                ; get abs (p1 - p0)

        paddusb     xmm5,                   xmm2              ; abs (p0 - q0) *2 + abs(p1-q1)/2

        movdqa      xmm2,                   [rdx]             ; hev

        psubusb     xmm5,                   xmm7              ; abs (p0 - q0) *2 + abs(p1-q1)/2  > blimit
        psubusb     xmm4,                   xmm2              ; hev

        psubusb     xmm3,                   xmm2              ; hev
        por         xmm1,                   xmm5

        pxor        xmm7,                   xmm7
        paddb       xmm4,                   xmm3              ; hev abs(q1 - q0) > thresh || abs(p1 - p0) > thresh

        pcmpeqb     xmm4,                   xmm5              ; hev
        pcmpeqb     xmm3,                   xmm3              ; hev

        pcmpeqb     xmm1,                   xmm7              ; mask xmm1
        pxor        xmm4,                   xmm3              ; hev
%endmacro

%macro B_FILTER 1
        movdqa      xmm3,                   [GLOBAL(t80)]
%if %1 == 0
        movdqa      xmm2,                   [rsp+_p1]                ; p1
        movdqa      xmm7,                   [rsp+_q1]                ; q1
%elif %1 == 1
        movdqa      xmm2,                   [rsi+2*rax]       ; p1
        movdqa      xmm7,                   [rdi]             ; q1
%elif %1 == 2
        movdqa      xmm2,                   [rsp+_p1]         ; p1
        movdqa      xmm6,                   [rsp+_p0]         ; p0
        movdqa      xmm0,                   [rsp+_q0]         ; q0
        movdqa      xmm7,                   [rsp+_q1]         ; q1
%endif

        pxor        xmm2,                   xmm3              ; p1 offset to convert to signed values
        pxor        xmm7,                   xmm3              ; q1 offset to convert to signed values

        psubsb      xmm2,                   xmm7              ; p1 - q1
        pxor        xmm6,                   xmm3              ; offset to convert to signed values

        pand        xmm2,                   xmm4              ; high var mask (hvm)(p1 - q1)
        pxor        xmm0,                   xmm3              ; offset to convert to signed values

        movdqa      xmm3,                   xmm0              ; q0
        psubsb      xmm0,                   xmm6              ; q0 - p0
        paddsb      xmm2,                   xmm0              ; 1 * (q0 - p0) + hvm(p1 - q1)
        paddsb      xmm2,                   xmm0              ; 2 * (q0 - p0) + hvm(p1 - q1)
        paddsb      xmm2,                   xmm0              ; 3 * (q0 - p0) + hvm(p1 - q1)
        pand        xmm1,                   xmm2              ; mask filter values we don't care about

        movdqa      xmm2,                   xmm1
        paddsb      xmm1,                   [GLOBAL(t4)]      ; 3* (q0 - p0) + hvm(p1 - q1) + 4
        paddsb      xmm2,                   [GLOBAL(t3)]      ; 3* (q0 - p0) + hvm(p1 - q1) + 3

        punpckhbw   xmm5,                   xmm2              ; axbxcxdx
        punpcklbw   xmm2,                   xmm2              ; exfxgxhx

        punpcklbw   xmm0,                   xmm1              ; exfxgxhx
        psraw       xmm5,                   11                ; sign extended shift right by 3

        punpckhbw   xmm1,                   xmm1              ; axbxcxdx
        psraw       xmm2,                   11                ; sign extended shift right by 3

        packsswb    xmm2,                   xmm5              ; (3* (q0 - p0) + hvm(p1 - q1) + 3) >> 3;
        psraw       xmm0,                   11                ; sign extended shift right by 3

        psraw       xmm1,                   11                ; sign extended shift right by 3
        movdqa      xmm5,                   xmm0              ; save results

        packsswb    xmm0,                   xmm1              ; (3* (q0 - p0) + hvm(p1 - q1) + 4) >>3

        paddsb      xmm6,                   xmm2              ; p0+= p0 add

        movdqa      xmm2,                   [GLOBAL(ones)]
        paddsw      xmm5,                   xmm2
        paddsw      xmm1,                   xmm2
        psraw       xmm5,                   1                 ; partial shifted one more time for 2nd tap
        psraw       xmm1,                   1                 ; partial shifted one more time for 2nd tap
        packsswb    xmm5,                   xmm1              ; (3* (q0 - p0) + hvm(p1 - q1) + 4) >>4
        movdqa      xmm2,                   [GLOBAL(t80)]

%if %1 == 0
        movdqa      xmm1,                   [rsp+_p1]         ; p1
        lea         rsi,                    [rsi + rcx*2]
        lea         rdi,                    [rdi + rcx*2]
%elif %1 == 1
        movdqa      xmm1,                   [rsi+2*rax]       ; p1
%elif %1 == 2
        movdqa      xmm1,                   [rsp+_p1]         ; p1
%endif

        pandn       xmm4,                   xmm5              ; high edge variance additive
        pxor        xmm6,                   xmm2              ; unoffset

        pxor        xmm1,                   xmm2              ; reoffset
        psubsb      xmm3,                   xmm0              ; q0-= q0 add

        paddsb      xmm1,                   xmm4              ; p1+= p1 add
        pxor        xmm3,                   xmm2              ; unoffset

        pxor        xmm1,                   xmm2              ; unoffset
        psubsb      xmm7,                   xmm4              ; q1-= q1 add

        pxor        xmm7,                   xmm2              ; unoffset
%if %1 == 0
        movq        [rsi],                  xmm6              ; p0
        movhps      [rdi],                  xmm6
        movq        [rsi + rax],            xmm1              ; p1
        movhps      [rdi + rax],            xmm1
        movq        [rsi + rcx],            xmm3              ; q0
        movhps      [rdi + rcx],            xmm3
        movq        [rsi + rcx*2],          xmm7              ; q1
        movhps      [rdi + rcx*2],          xmm7
%elif %1 == 1
        movdqa      [rsi+rax],              xmm6              ; write back
        movdqa      [rsi+2*rax],            xmm1              ; write back
        movdqa      [rsi],                  xmm3              ; write back
        movdqa      [rdi],                  xmm7              ; write back
%endif

%endmacro

SECTION .text

%if ABI_IS_32BIT

;void vp8_loop_filter_horizontal_edge_sse2
;(
;    unsigned char *src_ptr,
;    int            src_pixel_step,
;    const char    *blimit,
;    const char    *limit,
;    const char    *thresh,
;)
globalsym(vp8_loop_filter_horizontal_edge_sse2)
sym(vp8_loop_filter_horizontal_edge_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 5
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, lf_var_size

        mov         rsi,                    arg(0)           ;src_ptr
        movsxd      rax,                    dword ptr arg(1) ;src_pixel_step

        mov         rdx,                    arg(3)           ;limit

        lea         rdi,                    [rsi+rax]        ; rdi points to row +1 for indirect addressing

        ; calculate breakout conditions and high edge variance
        LFH_FILTER_AND_HEV_MASK 1
        ; filter and write back the result
        B_FILTER 1

    add rsp, lf_var_size
    pop rsp
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

%endif

;void vp8_loop_filter_horizontal_edge_uv_sse2
;(
;    unsigned char *src_ptr,
;    int            src_pixel_step,
;    const char    *blimit,
;    const char    *limit,
;    const char    *thresh,
;    int            count
;)
globalsym(vp8_loop_filter_horizontal_edge_uv_sse2)
sym(vp8_loop_filter_horizontal_edge_uv_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, lf_var_size

        mov         rsi,                    arg(0)             ; u
        mov         rdi,                    arg(5)             ; v
        movsxd      rax,                    dword ptr arg(1)   ; src_pixel_step
        mov         rcx,                    rax
        neg         rax                     ; negate pitch to deal with above border

        mov         rdx,                    arg(3)             ;limit

        lea         rsi,                    [rsi + rcx]
        lea         rdi,                    [rdi + rcx]

        ; calculate breakout conditions and high edge variance
        LFH_FILTER_AND_HEV_MASK 0
        ; filter and write back the result
        B_FILTER 0

    add rsp, lf_var_size
    pop rsp
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


%macro MB_FILTER_AND_WRITEBACK 1
        movdqa      xmm3,                   [GLOBAL(t80)]
%if %1 == 0
        movdqa      xmm2,                   [rsp+_p1]              ; p1
        movdqa      xmm7,                   [rsp+_q1]              ; q1
%elif %1 == 1
        movdqa      xmm2,                   [rsi+2*rax]     ; p1
        movdqa      xmm7,                   [rdi]           ; q1

        mov         rcx,                    rax
        neg         rcx
%elif %1 == 2
        movdqa      xmm2,                   [rsp+_p1]       ; p1
        movdqa      xmm6,                   [rsp+_p0]       ; p0
        movdqa      xmm0,                   [rsp+_q0]       ; q0
        movdqa      xmm7,                   [rsp+_q1]       ; q1
%endif

        pxor        xmm2,                   xmm3            ; p1 offset to convert to signed values
        pxor        xmm7,                   xmm3            ; q1 offset to convert to signed values
        pxor        xmm6,                   xmm3            ; offset to convert to signed values
        pxor        xmm0,                   xmm3            ; offset to convert to signed values

        psubsb      xmm2,                   xmm7            ; p1 - q1

        movdqa      xmm3,                   xmm0            ; q0
        psubsb      xmm0,                   xmm6            ; q0 - p0
        paddsb      xmm2,                   xmm0            ; 1 * (q0 - p0) + (p1 - q1)
        paddsb      xmm2,                   xmm0            ; 2 * (q0 - p0)
        paddsb      xmm2,                   xmm0            ; 3 * (q0 - p0) + (p1 - q1)
        pand        xmm1,                   xmm2            ; mask filter values we don't care about

        movdqa      xmm2,                   xmm1            ; vp8_filter

        pand        xmm2,                   xmm4            ; Filter2 = vp8_filter & hev
        pxor        xmm0,                   xmm0

        pandn       xmm4,                   xmm1            ; vp8_filter&=~hev
        pxor        xmm1,                   xmm1

        punpcklbw   xmm0,                   xmm4            ; Filter 2 (hi)
        punpckhbw   xmm1,                   xmm4            ; Filter 2 (lo)

        movdqa      xmm5,                   xmm2

        movdqa      xmm4,                   [GLOBAL(s9)]
        paddsb      xmm5,                   [GLOBAL(t3)]    ; vp8_signed_char_clamp(Filter2 + 3)
        paddsb      xmm2,                   [GLOBAL(t4)]    ; vp8_signed_char_clamp(Filter2 + 4)

        pmulhw      xmm1,                   xmm4            ; Filter 2 (lo) * 9
        pmulhw      xmm0,                   xmm4            ; Filter 2 (hi) * 9

        punpckhbw   xmm7,                   xmm5            ; axbxcxdx
        punpcklbw   xmm5,                   xmm5            ; exfxgxhx

        psraw       xmm7,                   11              ; sign extended shift right by 3

        psraw       xmm5,                   11              ; sign extended shift right by 3
        punpckhbw   xmm4,                   xmm2            ; axbxcxdx

        punpcklbw   xmm2,                   xmm2            ; exfxgxhx
        psraw       xmm4,                   11              ; sign extended shift right by 3

        packsswb    xmm5,                   xmm7            ; Filter2 >>=3;
        psraw       xmm2,                   11              ; sign extended shift right by 3

        packsswb    xmm2,                   xmm4            ; Filter1 >>=3;

        paddsb      xmm6,                   xmm5            ; ps0 =ps0 + Fitler2

        psubsb      xmm3,                   xmm2            ; qs0 =qs0 - Filter1
        movdqa      xmm7,                   xmm1

        movdqa      xmm4,                   [GLOBAL(s63)]
        movdqa      xmm5,                   xmm0
        movdqa      xmm2,                   xmm5
        paddw       xmm0,                   xmm4            ; Filter 2 (hi) * 9 + 63
        paddw       xmm1,                   xmm4            ; Filter 2 (lo) * 9 + 63
        movdqa      xmm4,                   xmm7

        paddw       xmm5,                   xmm5            ; Filter 2 (hi) * 18

        paddw       xmm7,                   xmm7            ; Filter 2 (lo) * 18
        paddw       xmm5,                   xmm0            ; Filter 2 (hi) * 27 + 63

        paddw       xmm7,                   xmm1            ; Filter 2 (lo) * 27 + 63
        paddw       xmm2,                   xmm0            ; Filter 2 (hi) * 18 + 63
        psraw       xmm0,                   7               ; (Filter 2 (hi) * 9 + 63) >> 7

        paddw       xmm4,                   xmm1            ; Filter 2 (lo) * 18 + 63
        psraw       xmm1,                   7               ; (Filter 2 (lo) * 9 + 63) >> 7
        psraw       xmm2,                   7               ; (Filter 2 (hi) * 18 + 63) >> 7

        packsswb    xmm0,                   xmm1            ; u1 = vp8_signed_char_clamp((63 + Filter2 * 9)>>7)

        psraw       xmm4,                   7               ; (Filter 2 (lo) * 18 + 63) >> 7
        psraw       xmm5,                   7               ; (Filter 2 (hi) * 27 + 63) >> 7
        psraw       xmm7,                   7               ; (Filter 2 (lo) * 27 + 63) >> 7

        packsswb    xmm5,                   xmm7            ; u3 = vp8_signed_char_clamp((63 + Filter2 * 27)>>7)
        packsswb    xmm2,                   xmm4            ; u2 = vp8_signed_char_clamp((63 + Filter2 * 18)>>7)
        movdqa      xmm7,                   [GLOBAL(t80)]

%if %1 == 0
        movdqa      xmm1,                   [rsp+_q1]       ; q1
        movdqa      xmm4,                   [rsp+_p1]       ; p1
        lea         rsi,                    [rsi+rcx*2]
        lea         rdi,                    [rdi+rcx*2]

%elif %1 == 1
        movdqa      xmm1,                   [rdi]           ; q1
        movdqa      xmm4,                   [rsi+rax*2]     ; p1
%elif %1 == 2
        movdqa      xmm4,                   [rsp+_p1]       ; p1
        movdqa      xmm1,                   [rsp+_q1]       ; q1
%endif

        pxor        xmm1,                   xmm7
        pxor        xmm4,                   xmm7

        psubsb      xmm3,                   xmm5            ; sq = vp8_signed_char_clamp(qs0 - u3)
        paddsb      xmm6,                   xmm5            ; sp = vp8_signed_char_clamp(ps0 - u3)
        psubsb      xmm1,                   xmm2            ; sq = vp8_signed_char_clamp(qs1 - u2)
        paddsb      xmm4,                   xmm2            ; sp = vp8_signed_char_clamp(ps1 - u2)

%if %1 == 1
        movdqa      xmm2,                   [rdi+rax*4]     ; p2
        movdqa      xmm5,                   [rdi+rcx]       ; q2
%else
        movdqa      xmm2,                   [rsp+_p2]       ; p2
        movdqa      xmm5,                   [rsp+_q2]       ; q2
%endif

        pxor        xmm1,                   xmm7            ; *oq1 = sq^0x80;
        pxor        xmm4,                   xmm7            ; *op1 = sp^0x80;
        pxor        xmm2,                   xmm7
        pxor        xmm5,                   xmm7
        paddsb      xmm2,                   xmm0            ; sp = vp8_signed_char_clamp(ps2 - u)
        psubsb      xmm5,                   xmm0            ; sq = vp8_signed_char_clamp(qs2 - u)
        pxor        xmm2,                   xmm7            ; *op2 = sp^0x80;
        pxor        xmm5,                   xmm7            ; *oq2 = sq^0x80;
        pxor        xmm3,                   xmm7            ; *oq0 = sq^0x80
        pxor        xmm6,                   xmm7            ; *oq0 = sp^0x80
%if %1 == 0
        movq        [rsi],                  xmm6            ; p0
        movhps      [rdi],                  xmm6
        movq        [rsi + rcx],            xmm3            ; q0
        movhps      [rdi + rcx],            xmm3
        lea         rdx,                    [rcx + rcx*2]
        movq        [rsi+rcx*2],            xmm1            ; q1
        movhps      [rdi+rcx*2],            xmm1

        movq        [rsi + rax],            xmm4            ; p1
        movhps      [rdi + rax],            xmm4

        movq        [rsi+rax*2],            xmm2            ; p2
        movhps      [rdi+rax*2],            xmm2

        movq        [rsi+rdx],              xmm5            ; q2
        movhps      [rdi+rdx],              xmm5
%elif %1 == 1
        movdqa      [rdi+rcx],              xmm5            ; q2
        movdqa      [rdi],                  xmm1            ; q1
        movdqa      [rsi],                  xmm3            ; q0
        movdqa      [rsi+rax  ],            xmm6            ; p0
        movdqa      [rsi+rax*2],            xmm4            ; p1
        movdqa      [rdi+rax*4],            xmm2            ; p2
%elif %1 == 2
        movdqa      [rsp+_p1],              xmm4            ; p1
        movdqa      [rsp+_p0],              xmm6            ; p0
        movdqa      [rsp+_q0],              xmm3            ; q0
        movdqa      [rsp+_q1],              xmm1            ; q1
%endif

%endmacro


;void vp8_mbloop_filter_horizontal_edge_sse2
;(
;    unsigned char *src_ptr,
;    int            src_pixel_step,
;    const char    *blimit,
;    const char    *limit,
;    const char    *thresh,
;)
globalsym(vp8_mbloop_filter_horizontal_edge_sse2)
sym(vp8_mbloop_filter_horizontal_edge_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 5
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, lf_var_size

        mov         rsi,                    arg(0)            ;src_ptr
        movsxd      rax,                    dword ptr arg(1)  ;src_pixel_step
        mov         rdx,                    arg(3)            ;limit

        lea         rdi,                    [rsi+rax]         ; rdi points to row +1 for indirect addressing

        ; calculate breakout conditions and high edge variance
        LFH_FILTER_AND_HEV_MASK 1
        ; filter and write back the results
        MB_FILTER_AND_WRITEBACK 1

    add rsp, lf_var_size
    pop rsp
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


;void vp8_mbloop_filter_horizontal_edge_uv_sse2
;(
;    unsigned char *u,
;    int            src_pixel_step,
;    const char    *blimit,
;    const char    *limit,
;    const char    *thresh,
;    unsigned char *v
;)
globalsym(vp8_mbloop_filter_horizontal_edge_uv_sse2)
sym(vp8_mbloop_filter_horizontal_edge_uv_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, lf_var_size

        mov         rsi,                    arg(0)             ; u
        mov         rdi,                    arg(5)             ; v
        movsxd      rax,                    dword ptr arg(1)   ; src_pixel_step
        mov         rcx,                    rax
        neg         rax                     ; negate pitch to deal with above border
        mov         rdx,                    arg(3)             ;limit

        lea         rsi,                    [rsi + rcx]
        lea         rdi,                    [rdi + rcx]

        ; calculate breakout conditions and high edge variance
        LFH_FILTER_AND_HEV_MASK 0
        ; filter and write back the results
        MB_FILTER_AND_WRITEBACK 0

    add rsp, lf_var_size
    pop rsp
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


%macro TRANSPOSE_16X8 2
        movq        xmm4,               [rsi]           ; xx xx xx xx xx xx xx xx 07 06 05 04 03 02 01 00
        movq        xmm1,               [rdi]           ; xx xx xx xx xx xx xx xx 17 16 15 14 13 12 11 10
        movq        xmm0,               [rsi+2*rax]     ; xx xx xx xx xx xx xx xx 27 26 25 24 23 22 21 20
        movq        xmm7,               [rdi+2*rax]     ; xx xx xx xx xx xx xx xx 37 36 35 34 33 32 31 30
        movq        xmm5,               [rsi+4*rax]     ; xx xx xx xx xx xx xx xx 47 46 45 44 43 42 41 40
        movq        xmm2,               [rdi+4*rax]     ; xx xx xx xx xx xx xx xx 57 56 55 54 53 52 51 50

        punpcklbw   xmm4,               xmm1            ; 17 07 16 06 15 05 14 04 13 03 12 02 11 01 10 00

        movq        xmm1,               [rdi+2*rcx]     ; xx xx xx xx xx xx xx xx 77 76 75 74 73 72 71 70

        movdqa      xmm3,               xmm4            ; 17 07 16 06 15 05 14 04 13 03 12 02 11 01 10 00
        punpcklbw   xmm0,               xmm7            ; 37 27 36 36 35 25 34 24 33 23 32 22 31 21 30 20

        movq        xmm7,               [rsi+2*rcx]     ; xx xx xx xx xx xx xx xx 67 66 65 64 63 62 61 60

        punpcklbw   xmm5,               xmm2            ; 57 47 56 46 55 45 54 44 53 43 52 42 51 41 50 40
%if %1
        lea         rsi,                [rsi+rax*8]
        lea         rdi,                [rdi+rax*8]
%else
        mov         rsi,                arg(5)          ; v_ptr
%endif

        movdqa      xmm6,               xmm5            ; 57 47 56 46 55 45 54 44 53 43 52 42 51 41 50 40
        punpcklbw   xmm7,               xmm1            ; 77 67 76 66 75 65 74 64 73 63 72 62 71 61 70 60
        punpcklwd   xmm5,               xmm7            ; 73 63 53 43 72 62 52 42 71 61 51 41 70 60 50 40
        punpckhwd   xmm6,               xmm7            ; 77 67 57 47 76 66 56 46 75 65 55 45 74 64 54 44
        punpcklwd   xmm3,               xmm0            ; 33 23 13 03 32 22 12 02 31 21 11 01 30 20 10 00

%if %1 == 0
        lea         rdi,                [rsi + rax - 4] ; rdi points to row +1 for indirect addressing
        lea         rsi,                [rsi - 4]
%endif

        movdqa      xmm2,               xmm3            ; 33 23 13 03 32 22 12 02 31 21 11 01 30 20 10 00
        punpckhwd   xmm4,               xmm0            ; 37 27 17 07 36 26 16 06 35 25 15 05 34 24 14 04

        movdqa      xmm7,               xmm4            ; 37 27 17 07 36 26 16 06 35 25 15 05 34 24 14 04
        punpckhdq   xmm3,               xmm5            ; 73 63 53 43 33 23 13 03 72 62 52 42 32 22 12 02

        punpckhdq   xmm7,               xmm6            ; 77 67 57 47 37 27 17 07 76 66 56 46 36 26 16 06

        punpckldq   xmm4,               xmm6            ; 75 65 55 45 35 25 15 05 74 64 54 44 34 24 14 04

        punpckldq   xmm2,               xmm5            ; 71 61 51 41 31 21 11 01 70 60 50 40 30 20 10 00

        movdqa      [rsp+_t0],          xmm2            ; save to free XMM2

        movq        xmm2,               [rsi]           ; xx xx xx xx xx xx xx xx 87 86 85 84 83 82 81 80
        movq        xmm6,               [rdi]           ; xx xx xx xx xx xx xx xx 97 96 95 94 93 92 91 90
        movq        xmm0,               [rsi+2*rax]     ; xx xx xx xx xx xx xx xx a7 a6 a5 a4 a3 a2 a1 a0
        movq        xmm5,               [rdi+2*rax]     ; xx xx xx xx xx xx xx xx b7 b6 b5 b4 b3 b2 b1 b0
        movq        xmm1,               [rsi+4*rax]     ; xx xx xx xx xx xx xx xx c7 c6 c5 c4 c3 c2 c1 c0

        punpcklbw   xmm2,               xmm6            ; 97 87 96 86 95 85 94 84 93 83 92 82 91 81 90 80

        movq        xmm6,               [rdi+4*rax]     ; xx xx xx xx xx xx xx xx d7 d6 d5 d4 d3 d2 d1 d0

        punpcklbw   xmm0,               xmm5            ; b7 a7 b6 a6 b5 a5 b4 a4 b3 a3 b2 a2 b1 a1 b0 a0

        movq        xmm5,               [rsi+2*rcx]     ; xx xx xx xx xx xx xx xx e7 e6 e5 e4 e3 e2 e1 e0

        punpcklbw   xmm1,               xmm6            ; d7 c7 d6 c6 d5 c5 d4 c4 d3 c3 d2 c2 d1 e1 d0 c0

        movq        xmm6,               [rdi+2*rcx]     ; xx xx xx xx xx xx xx xx f7 f6 f5 f4 f3 f2 f1 f0

        punpcklbw   xmm5,               xmm6            ; f7 e7 f6 e6 f5 e5 f4 e4 f3 e3 f2 e2 f1 e1 f0 e0

        movdqa      xmm6,               xmm1            ;
        punpckhwd   xmm6,               xmm5            ; f7 e7 d7 c7 f6 e6 d6 c6 f5 e5 d5 c5 f4 e4 d4 c4

        punpcklwd   xmm1,               xmm5            ; f3 e3 d3 c3 f2 e2 d2 c2 f1 e1 d1 c1 f0 e0 d0 c0
        movdqa      xmm5,               xmm2            ; 97 87 96 86 95 85 94 84 93 83 92 82 91 81 90 80

        punpcklwd   xmm5,               xmm0            ; b3 a3 93 83 b2 a2 92 82 b1 a1 91 81 b0 a0 90 80

        punpckhwd   xmm2,               xmm0            ; b7 a7 97 87 b6 a6 96 86 b5 a5 95 85 b4 a4 94 84

        movdqa      xmm0,               xmm5
        punpckldq   xmm0,               xmm1            ; f1 e1 d1 c1 b1 a1 91 81 f0 e0 d0 c0 b0 a0 90 80

        punpckhdq   xmm5,               xmm1            ; f3 e3 d3 c3 b3 a3 93 83 f2 e2 d2 c2 b2 a2 92 82
        movdqa      xmm1,               xmm2            ; b7 a7 97 87 b6 a6 96 86 b5 a5 95 85 b4 a4 94 84

        punpckldq   xmm1,               xmm6            ; f5 e5 d5 c5 b5 a5 95 85 f4 e4 d4 c4 b4 a4 94 84

        punpckhdq   xmm2,               xmm6            ; f7 e7 d7 c7 b7 a7 97 87 f6 e6 d6 c6 b6 a6 96 86
        movdqa      xmm6,               xmm7            ; 77 67 57 47 37 27 17 07 76 66 56 46 36 26 16 06

        punpcklqdq  xmm6,               xmm2            ; f6 e6 d6 c6 b6 a6 96 86 76 66 56 46 36 26 16 06

        punpckhqdq  xmm7,               xmm2            ; f7 e7 d7 c7 b7 a7 97 87 77 67 57 47 37 27 17 07

%if %2 == 0
        movdqa      [rsp+_q3],          xmm7            ; save 7
        movdqa      [rsp+_q2],          xmm6            ; save 6
%endif
        movdqa      xmm2,               xmm3            ; 73 63 53 43 33 23 13 03 72 62 52 42 32 22 12 02
        punpckhqdq  xmm3,               xmm5            ; f3 e3 d3 c3 b3 a3 93 83 73 63 53 43 33 23 13 03
        punpcklqdq  xmm2,               xmm5            ; f2 e2 d2 c2 b2 a2 92 82 72 62 52 42 32 22 12 02
        movdqa      [rsp+_p1],          xmm2            ; save 2

        movdqa      xmm5,               xmm4            ; 75 65 55 45 35 25 15 05 74 64 54 44 34 24 14 04
        punpcklqdq  xmm4,               xmm1            ; f4 e4 d4 c4 b4 a4 94 84 74 64 54 44 34 24 14 04
        movdqa      [rsp+_p0],          xmm3            ; save 3

        punpckhqdq  xmm5,               xmm1            ; f5 e5 d5 c5 b5 a5 95 85 75 65 55 45 35 25 15 05

        movdqa      [rsp+_q0],          xmm4            ; save 4
        movdqa      [rsp+_q1],          xmm5            ; save 5
        movdqa      xmm1,               [rsp+_t0]

        movdqa      xmm2,               xmm1            ;
        punpckhqdq  xmm1,               xmm0            ; f1 e1 d1 c1 b1 a1 91 81 71 61 51 41 31 21 11 01
        punpcklqdq  xmm2,               xmm0            ; f0 e0 d0 c0 b0 a0 90 80 70 60 50 40 30 20 10 00

%if %2 == 0
        movdqa      [rsp+_p2],          xmm1
        movdqa      [rsp+_p3],          xmm2
%endif

%endmacro

%macro LFV_FILTER_MASK_HEV_MASK 0
        movdqa      xmm0,               xmm6            ; q2
        psubusb     xmm0,               xmm7            ; q2-q3

        psubusb     xmm7,               xmm6            ; q3-q2
        movdqa      xmm4,               xmm5            ; q1

        por         xmm7,               xmm0            ; abs (q3-q2)
        psubusb     xmm4,               xmm6            ; q1-q2

        movdqa      xmm0,               xmm1
        psubusb     xmm6,               xmm5            ; q2-q1

        por         xmm6,               xmm4            ; abs (q2-q1)
        psubusb     xmm0,               xmm2            ; p2 - p3;

        psubusb     xmm2,               xmm1            ; p3 - p2;
        por         xmm0,               xmm2            ; abs(p2-p3)

        movdqa      xmm5,               [rsp+_p1]       ; p1
        pmaxub      xmm0,               xmm7

        movdqa      xmm2,               xmm5            ; p1
        psubusb     xmm5,               xmm1            ; p1-p2
        psubusb     xmm1,               xmm2            ; p2-p1

        movdqa      xmm7,               xmm3            ; p0
        psubusb     xmm7,               xmm2            ; p0-p1

        por         xmm1,               xmm5            ; abs(p2-p1)
        pmaxub      xmm0,               xmm6

        pmaxub      xmm0,               xmm1
        movdqa      xmm1,               xmm2            ; p1

        psubusb     xmm2,               xmm3            ; p1-p0

        por         xmm2,               xmm7            ; abs(p1-p0)

        pmaxub      xmm0,               xmm2

        movdqa      xmm5,               [rsp+_q0]       ; q0
        movdqa      xmm7,               [rsp+_q1]       ; q1

        mov         rdx,                arg(3)          ; limit

        movdqa      xmm6,               xmm5            ; q0
        movdqa      xmm4,               xmm7            ; q1

        psubusb     xmm5,               xmm7            ; q0-q1
        psubusb     xmm7,               xmm6            ; q1-q0

        por         xmm7,               xmm5            ; abs(q1-q0)

        pmaxub      xmm0,               xmm7

        psubusb     xmm0,               [rdx]           ; limit

        mov         rdx,                arg(2)          ; blimit
        movdqa      xmm5,               xmm4            ; q1

        psubusb     xmm5,               xmm1            ; q1-=p1
        psubusb     xmm1,               xmm4            ; p1-=q1

        por         xmm5,               xmm1            ; abs(p1-q1)
        movdqa      xmm1,               xmm3            ; p0

        pand        xmm5,               [GLOBAL(tfe)]   ; set lsb of each byte to zero
        psubusb     xmm1,               xmm6            ; p0-q0

        movdqa      xmm4,               [rdx]           ; blimit
        mov         rdx,                arg(4)          ; get thresh

        psrlw       xmm5,               1               ; abs(p1-q1)/2
        psubusb     xmm6,               xmm3            ; q0-p0

        por         xmm1,               xmm6            ; abs(q0-p0)
        paddusb     xmm1,               xmm1            ; abs(q0-p0)*2
        movdqa      xmm3,               [rdx]

        paddusb     xmm1,               xmm5            ; abs (p0 - q0) *2 + abs(p1-q1)/2
        psubusb     xmm2,               xmm3            ; abs(q1 - q0) > thresh

        psubusb     xmm7,               xmm3            ; abs(p1 - p0)> thresh

        psubusb     xmm1,               xmm4            ; abs (p0 - q0) *2 + abs(p1-q1)/2  > blimit
        por         xmm2,               xmm7            ; abs(q1 - q0) > thresh || abs(p1 - p0) > thresh

        por         xmm1,               xmm0            ; mask
        pcmpeqb     xmm2,               xmm0

        pxor        xmm0,               xmm0
        pcmpeqb     xmm4,               xmm4

        pcmpeqb     xmm1,               xmm0
        pxor        xmm4,               xmm2
%endmacro

%macro BV_TRANSPOSE 0
        ; xmm1 =    f2 e2 d2 c2 b2 a2 92 82 72 62 52 42 32 22 12 02
        ; xmm6 =    f3 e3 d3 c3 b3 a3 93 83 73 63 53 43 33 23 13 03
        ; xmm3 =    f4 e4 d4 c4 b4 a4 94 84 74 64 54 44 34 24 14 04
        ; xmm7 =    f5 e5 d5 c5 b5 a5 95 85 75 65 55 45 35 25 15 05
        movdqa      xmm2,               xmm1            ; f2 e2 d2 c2 b2 a2 92 82 72 62 52 42 32 22 12 02
        punpcklbw   xmm2,               xmm6            ; 73 72 63 62 53 52 43 42 33 32 23 22 13 12 03 02

        movdqa      xmm4,               xmm3            ; f4 e4 d4 c4 b4 a4 94 84 74 64 54 44 34 24 14 04
        punpckhbw   xmm1,               xmm6            ; f3 f2 e3 e2 d3 d2 c3 c2 b3 b2 a3 a2 93 92 83 82

        punpcklbw   xmm4,               xmm7            ; 75 74 65 64 55 54 45 44 35 34 25 24 15 14 05 04

        punpckhbw   xmm3,               xmm7            ; f5 f4 e5 e4 d5 d4 c5 c4 b5 b4 a5 a4 95 94 85 84

        movdqa      xmm6,               xmm2            ; 73 72 63 62 53 52 43 42 33 32 23 22 13 12 03 02
        punpcklwd   xmm2,               xmm4            ; 35 34 33 32 25 24 23 22 15 14 13 12 05 04 03 02

        punpckhwd   xmm6,               xmm4            ; 75 74 73 72 65 64 63 62 55 54 53 52 45 44 43 42
        movdqa      xmm5,               xmm1            ; f3 f2 e3 e2 d3 d2 c3 c2 b3 b2 a3 a2 93 92 83 82

        punpcklwd   xmm1,               xmm3            ; b5 b4 b3 b2 a5 a4 a3 a2 95 94 93 92 85 84 83 82

        punpckhwd   xmm5,               xmm3            ; f5 f4 f3 f2 e5 e4 e3 e2 d5 d4 d3 d2 c5 c4 c3 c2
        ; xmm2 = 35 34 33 32 25 24 23 22 15 14 13 12 05 04 03 02
        ; xmm6 = 75 74 73 72 65 64 63 62 55 54 53 52 45 44 43 42
        ; xmm1 = b5 b4 b3 b2 a5 a4 a3 a2 95 94 93 92 85 84 83 82
        ; xmm5 = f5 f4 f3 f2 e5 e4 e3 e2 d5 d4 d3 d2 c5 c4 c3 c2
%endmacro

%macro BV_WRITEBACK 2
        movd        [rsi+2],            %1
        movd        [rsi+4*rax+2],      %2
        psrldq      %1,                 4
        psrldq      %2,                 4
        movd        [rdi+2],            %1
        movd        [rdi+4*rax+2],      %2
        psrldq      %1,                 4
        psrldq      %2,                 4
        movd        [rsi+2*rax+2],      %1
        movd        [rsi+2*rcx+2],      %2
        psrldq      %1,                 4
        psrldq      %2,                 4
        movd        [rdi+2*rax+2],      %1
        movd        [rdi+2*rcx+2],      %2
%endmacro

%if ABI_IS_32BIT

;void vp8_loop_filter_vertical_edge_sse2
;(
;    unsigned char *src_ptr,
;    int            src_pixel_step,
;    const char    *blimit,
;    const char    *limit,
;    const char    *thresh,
;)
globalsym(vp8_loop_filter_vertical_edge_sse2)
sym(vp8_loop_filter_vertical_edge_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 5
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub             rsp, lf_var_size

        mov         rsi,        arg(0)                  ; src_ptr
        movsxd      rax,        dword ptr arg(1)        ; src_pixel_step

        lea         rsi,        [rsi - 4]
        lea         rdi,        [rsi + rax]             ; rdi points to row +1 for indirect addressing
        lea         rcx,        [rax*2+rax]

        ;transpose 16x8 to 8x16, and store the 8-line result on stack.
        TRANSPOSE_16X8 1, 1

        ; calculate filter mask and high edge variance
        LFV_FILTER_MASK_HEV_MASK

        ; start work on filters
        B_FILTER 2

        ; transpose and write back - only work on q1, q0, p0, p1
        BV_TRANSPOSE
        ; store 16-line result

        lea         rdx,        [rax]
        neg         rdx

        BV_WRITEBACK xmm1, xmm5

        lea         rsi,        [rsi+rdx*8]
        lea         rdi,        [rdi+rdx*8]
        BV_WRITEBACK xmm2, xmm6

    add rsp, lf_var_size
    pop rsp
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

%endif

;void vp8_loop_filter_vertical_edge_uv_sse2
;(
;    unsigned char *u,
;    int            src_pixel_step,
;    const char    *blimit,
;    const char    *limit,
;    const char    *thresh,
;    unsigned char *v
;)
globalsym(vp8_loop_filter_vertical_edge_uv_sse2)
sym(vp8_loop_filter_vertical_edge_uv_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub             rsp, lf_var_size

        mov         rsi,        arg(0)                  ; u_ptr
        movsxd      rax,        dword ptr arg(1)        ; src_pixel_step

        lea         rsi,        [rsi - 4]
        lea         rdi,        [rsi + rax]             ; rdi points to row +1 for indirect addressing
        lea         rcx,        [rax+2*rax]

        ;transpose 16x8 to 8x16, and store the 8-line result on stack.
        TRANSPOSE_16X8 0, 1

        ; calculate filter mask and high edge variance
        LFV_FILTER_MASK_HEV_MASK

        ; start work on filters
        B_FILTER 2

        ; transpose and write back - only work on q1, q0, p0, p1
        BV_TRANSPOSE

        lea         rdi,        [rsi + rax]             ; rdi points to row +1 for indirect addressing

        ; store 16-line result
        BV_WRITEBACK xmm1, xmm5

        mov         rsi,        arg(0)                  ; u_ptr
        lea         rsi,        [rsi - 4]
        lea         rdi,        [rsi + rax]             ; rdi points to row +1 for indirect addressing
        BV_WRITEBACK xmm2, xmm6

    add rsp, lf_var_size
    pop rsp
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

%macro MBV_TRANSPOSE 0
        movdqa      xmm0,               [rsp+_p3]           ; f0 e0 d0 c0 b0 a0 90 80 70 60 50 40 30 20 10 00
        movdqa      xmm1,               xmm0                ; f0 e0 d0 c0 b0 a0 90 80 70 60 50 40 30 20 10 00

        punpcklbw   xmm0,               xmm2                ; 71 70 61 60 51 50 41 40 31 30 21 20 11 10 01 00
        punpckhbw   xmm1,               xmm2                ; f1 f0 e1 e0 d1 d0 c1 c0 b1 b0 a1 a0 91 90 81 80

        movdqa      xmm7,               [rsp+_p1]           ; f2 e2 d2 c2 b2 a2 92 82 72 62 52 42 32 22 12 02
        movdqa      xmm6,               xmm7                ; f2 e2 d2 c2 b2 a2 92 82 72 62 52 42 32 22 12 02

        punpcklbw   xmm7,               [rsp+_p0]           ; 73 72 63 62 53 52 43 42 33 32 23 22 13 12 03 02
        punpckhbw   xmm6,               [rsp+_p0]           ; f3 f2 e3 e2 d3 d2 c3 c2 b3 b2 a3 a2 93 92 83 82

        movdqa      xmm3,               xmm0                ; 71 70 61 60 51 50 41 40 31 30 21 20 11 10 01 00
        punpcklwd   xmm0,               xmm7                ; 33 32 31 30 23 22 21 20 13 12 11 10 03 02 01 00

        punpckhwd   xmm3,               xmm7                ; 73 72 71 70 63 62 61 60 53 52 51 50 43 42 41 40
        movdqa      xmm4,               xmm1                ; f1 f0 e1 e0 d1 d0 c1 c0 b1 b0 a1 a0 91 90 81 80

        punpcklwd   xmm1,               xmm6                ; b3 b2 b1 b0 a3 a2 a1 a0 93 92 91 90 83 82 81 80
        punpckhwd   xmm4,               xmm6                ; f3 f2 f1 f0 e3 e2 e1 e0 d3 d2 d1 d0 c3 c2 c1 c0

        movdqa      xmm7,               [rsp+_q0]           ; f4 e4 d4 c4 b4 a4 94 84 74 64 54 44 34 24 14 04
        punpcklbw   xmm7,               [rsp+_q1]           ; 75 74 65 64 55 54 45 44 35 34 25 24 15 14 05 04

        movdqa      xmm6,               xmm5                ; f6 e6 d6 c6 b6 a6 96 86 76 66 56 46 36 26 16 06
        punpcklbw   xmm6,               [rsp+_q3]           ; 77 76 67 66 57 56 47 46 37 36 27 26 17 16 07 06

        movdqa      xmm2,               xmm7                ; 75 74 65 64 55 54 45 44 35 34 25 24 15 14 05 04
        punpcklwd   xmm7,               xmm6                ; 37 36 35 34 27 26 25 24 17 16 15 14 07 06 05 04

        punpckhwd   xmm2,               xmm6                ; 77 76 75 74 67 66 65 64 57 56 55 54 47 46 45 44
        movdqa      xmm6,               xmm0                ; 33 32 31 30 23 22 21 20 13 12 11 10 03 02 01 00

        punpckldq   xmm0,               xmm7                ; 17 16 15 14 13 12 11 10 07 06 05 04 03 02 01 00
        punpckhdq   xmm6,               xmm7                ; 37 36 35 34 33 32 31 30 27 26 25 24 23 22 21 20
%endmacro

%macro MBV_WRITEBACK_1 0
        movq        [rsi],              xmm0
        movhps      [rdi],              xmm0

        movq        [rsi+2*rax],        xmm6
        movhps      [rdi+2*rax],        xmm6

        movdqa      xmm0,               xmm3                ; 73 72 71 70 63 62 61 60 53 52 51 50 43 42 41 40
        punpckldq   xmm0,               xmm2                ; 57 56 55 54 53 52 51 50 47 46 45 44 43 42 41 40
        punpckhdq   xmm3,               xmm2                ; 77 76 75 74 73 72 71 70 67 66 65 64 63 62 61 60

        movq        [rsi+4*rax],        xmm0
        movhps      [rdi+4*rax],        xmm0

        movq        [rsi+2*rcx],        xmm3
        movhps      [rdi+2*rcx],        xmm3

        movdqa      xmm7,               [rsp+_q0]           ; f4 e4 d4 c4 b4 a4 94 84 74 64 54 44 34 24 14 04
        punpckhbw   xmm7,               [rsp+_q1]           ; f5 f4 e5 e4 d5 d4 c5 c4 b5 b4 a5 a4 95 94 85 84
        punpckhbw   xmm5,               [rsp+_q3]           ; f7 f6 e7 e6 d7 d6 c7 c6 b7 b6 a7 a6 97 96 87 86

        movdqa      xmm0,               xmm7
        punpcklwd   xmm0,               xmm5                ; b7 b6 b4 b4 a7 a6 a5 a4 97 96 95 94 87 86 85 84
        punpckhwd   xmm7,               xmm5                ; f7 f6 f5 f4 e7 e6 e5 e4 d7 d6 d5 d4 c7 c6 c5 c4

        movdqa      xmm5,               xmm1                ; b3 b2 b1 b0 a3 a2 a1 a0 93 92 91 90 83 82 81 80
        punpckldq   xmm1,               xmm0                ; 97 96 95 94 93 92 91 90 87 86 85 83 84 82 81 80
        punpckhdq   xmm5,               xmm0                ; b7 b6 b5 b4 b3 b2 b1 b0 a7 a6 a5 a4 a3 a2 a1 a0
%endmacro

%macro MBV_WRITEBACK_2 0
        movq        [rsi],              xmm1
        movhps      [rdi],              xmm1

        movq        [rsi+2*rax],        xmm5
        movhps      [rdi+2*rax],        xmm5

        movdqa      xmm1,               xmm4                ; f3 f2 f1 f0 e3 e2 e1 e0 d3 d2 d1 d0 c3 c2 c1 c0
        punpckldq   xmm1,               xmm7                ; d7 d6 d5 d4 d3 d2 d1 d0 c7 c6 c5 c4 c3 c2 c1 c0
        punpckhdq   xmm4,               xmm7                ; f7 f6 f4 f4 f3 f2 f1 f0 e7 e6 e5 e4 e3 e2 e1 e0

        movq        [rsi+4*rax],        xmm1
        movhps      [rdi+4*rax],        xmm1

        movq        [rsi+2*rcx],        xmm4
        movhps      [rdi+2*rcx],        xmm4
%endmacro


;void vp8_mbloop_filter_vertical_edge_sse2
;(
;    unsigned char *src_ptr,
;    int            src_pixel_step,
;    const char    *blimit,
;    const char    *limit,
;    const char    *thresh,
;)
globalsym(vp8_mbloop_filter_vertical_edge_sse2)
sym(vp8_mbloop_filter_vertical_edge_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 5
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub          rsp, lf_var_size

        mov         rsi,                arg(0)              ; src_ptr
        movsxd      rax,                dword ptr arg(1)    ; src_pixel_step

        lea         rsi,                [rsi - 4]
        lea         rdi,                [rsi + rax]         ; rdi points to row +1 for indirect addressing
        lea         rcx,                [rax*2+rax]

        ; Transpose
        TRANSPOSE_16X8 1, 0

        ; calculate filter mask and high edge variance
        LFV_FILTER_MASK_HEV_MASK

        neg         rax
        ; start work on filters
        MB_FILTER_AND_WRITEBACK 2

        lea         rsi,                [rsi+rax*8]
        lea         rdi,                [rdi+rax*8]

        ; transpose and write back
        MBV_TRANSPOSE

        neg         rax

        MBV_WRITEBACK_1


        lea         rsi,                [rsi+rax*8]
        lea         rdi,                [rdi+rax*8]
        MBV_WRITEBACK_2

    add rsp, lf_var_size
    pop rsp
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


;void vp8_mbloop_filter_vertical_edge_uv_sse2
;(
;    unsigned char *u,
;    int            src_pixel_step,
;    const char    *blimit,
;    const char    *limit,
;    const char    *thresh,
;    unsigned char *v
;)
globalsym(vp8_mbloop_filter_vertical_edge_uv_sse2)
sym(vp8_mbloop_filter_vertical_edge_uv_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub          rsp, lf_var_size

        mov         rsi,                arg(0)              ; u_ptr
        movsxd      rax,                dword ptr arg(1)    ; src_pixel_step

        lea         rsi,                [rsi - 4]
        lea         rdi,                [rsi + rax]         ; rdi points to row +1 for indirect addressing
        lea         rcx,                [rax+2*rax]

        ; Transpose
        TRANSPOSE_16X8 0, 0

        ; calculate filter mask and high edge variance
        LFV_FILTER_MASK_HEV_MASK

        ; start work on filters
        MB_FILTER_AND_WRITEBACK 2

        ; transpose and write back
        MBV_TRANSPOSE

        mov         rsi,                arg(0)             ;u_ptr
        lea         rsi,                [rsi - 4]
        lea         rdi,                [rsi + rax]
        MBV_WRITEBACK_1
        mov         rsi,                arg(5)             ;v_ptr
        lea         rsi,                [rsi - 4]
        lea         rdi,                [rsi + rax]
        MBV_WRITEBACK_2

    add rsp, lf_var_size
    pop rsp
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


;void vp8_loop_filter_simple_horizontal_edge_sse2
;(
;    unsigned char *src_ptr,
;    int  src_pixel_step,
;    const char *blimit,
;)
globalsym(vp8_loop_filter_simple_horizontal_edge_sse2)
sym(vp8_loop_filter_simple_horizontal_edge_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 3
    SAVE_XMM 7
    GET_GOT     rbx
    ; end prolog

        mov         rcx, arg(0)             ;src_ptr
        movsxd      rax, dword ptr arg(1)   ;src_pixel_step     ; destination pitch?
        movdqa      xmm6, [GLOBAL(tfe)]
        lea         rdx, [rcx + rax]
        neg         rax

        ; calculate mask
        movdqa      xmm0, [rdx]             ; q1
        mov         rdx, arg(2)             ;blimit
        movdqa      xmm1, [rcx+2*rax]       ; p1

        movdqa      xmm2, xmm1
        movdqa      xmm3, xmm0

        psubusb     xmm0, xmm1              ; q1-=p1
        psubusb     xmm1, xmm3              ; p1-=q1
        por         xmm1, xmm0              ; abs(p1-q1)
        pand        xmm1, xmm6              ; set lsb of each byte to zero
        psrlw       xmm1, 1                 ; abs(p1-q1)/2

        movdqa      xmm7, XMMWORD PTR [rdx]

        movdqa      xmm5, [rcx+rax]         ; p0
        movdqa      xmm4, [rcx]             ; q0
        movdqa      xmm0, xmm4              ; q0
        movdqa      xmm6, xmm5              ; p0
        psubusb     xmm5, xmm4              ; p0-=q0
        psubusb     xmm4, xmm6              ; q0-=p0
        por         xmm5, xmm4              ; abs(p0 - q0)

        movdqa      xmm4, [GLOBAL(t80)]

        paddusb     xmm5, xmm5              ; abs(p0-q0)*2
        paddusb     xmm5, xmm1              ; abs (p0 - q0) *2 + abs(p1-q1)/2
        psubusb     xmm5, xmm7              ; abs(p0 - q0) *2 + abs(p1-q1)/2  > blimit
        pxor        xmm7, xmm7
        pcmpeqb     xmm5, xmm7


        ; start work on filters
        pxor        xmm2, xmm4     ; p1 offset to convert to signed values
        pxor        xmm3, xmm4     ; q1 offset to convert to signed values
        psubsb      xmm2, xmm3              ; p1 - q1

        pxor        xmm6, xmm4     ; offset to convert to signed values
        pxor        xmm0, xmm4     ; offset to convert to signed values
        movdqa      xmm3, xmm0              ; q0
        psubsb      xmm0, xmm6              ; q0 - p0
        paddsb      xmm2, xmm0              ; p1 - q1 + 1 * (q0 - p0)
        paddsb      xmm2, xmm0              ; p1 - q1 + 2 * (q0 - p0)
        paddsb      xmm2, xmm0              ; p1 - q1 + 3 * (q0 - p0)
        pand        xmm5, xmm2              ; mask filter values we don't care about

        movdqa      xmm0, xmm5
        paddsb      xmm5,        [GLOBAL(t3)]                  ;  3* (q0 - p0) + (p1 - q1) + 4
        paddsb      xmm0,        [GLOBAL(t4)]                  ; +3 instead of +4

        movdqa      xmm1, [GLOBAL(te0)]
        movdqa      xmm2, [GLOBAL(t1f)]

;        pxor        xmm7, xmm7
        pcmpgtb     xmm7, xmm0              ;save sign
        pand        xmm7, xmm1              ;preserve the upper 3 bits
        psrlw       xmm0, 3
        pand        xmm0, xmm2              ;clear out upper 3 bits
        por         xmm0, xmm7              ;add sign
        psubsb      xmm3, xmm0              ; q0-= q0sz add

        pxor        xmm7, xmm7
        pcmpgtb     xmm7, xmm5              ;save sign
        pand        xmm7, xmm1              ;preserve the upper 3 bits
        psrlw       xmm5, 3
        pand        xmm5, xmm2              ;clear out upper 3 bits
        por         xmm5, xmm7              ;add sign
        paddsb      xmm6, xmm5              ; p0+= p0 add

        pxor        xmm3, xmm4     ; unoffset
        movdqa      [rcx], xmm3             ; write back

        pxor        xmm6, xmm4     ; unoffset
        movdqa      [rcx+rax], xmm6         ; write back

    ; begin epilog
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


;void vp8_loop_filter_simple_vertical_edge_sse2
;(
;    unsigned char *src_ptr,
;    int  src_pixel_step,
;    const char *blimit,
;)
globalsym(vp8_loop_filter_simple_vertical_edge_sse2)
sym(vp8_loop_filter_simple_vertical_edge_sse2):
    push        rbp         ; save old base pointer value.
    mov         rbp, rsp    ; set new base pointer value.
    SHADOW_ARGS_TO_STACK 3
    SAVE_XMM 7
    GET_GOT     rbx         ; save callee-saved reg
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 32                         ; reserve 32 bytes
    %define t0  [rsp + 0]    ;__declspec(align(16)) char t0[16];
    %define t1  [rsp + 16]   ;__declspec(align(16)) char t1[16];

        mov         rsi, arg(0) ;src_ptr
        movsxd      rax, dword ptr arg(1) ;src_pixel_step     ; destination pitch?

        lea         rsi,        [rsi - 2 ]
        lea         rdi,        [rsi + rax]
        lea         rdx,        [rsi + rax*4]
        lea         rcx,        [rdx + rax]

        movd        xmm0,       [rsi]                   ; (high 96 bits unused) 03 02 01 00
        movd        xmm1,       [rdx]                   ; (high 96 bits unused) 43 42 41 40
        movd        xmm2,       [rdi]                   ; 13 12 11 10
        movd        xmm3,       [rcx]                   ; 53 52 51 50
        punpckldq   xmm0,       xmm1                    ; (high 64 bits unused) 43 42 41 40 03 02 01 00
        punpckldq   xmm2,       xmm3                    ; 53 52 51 50 13 12 11 10

        movd        xmm4,       [rsi + rax*2]           ; 23 22 21 20
        movd        xmm5,       [rdx + rax*2]           ; 63 62 61 60
        movd        xmm6,       [rdi + rax*2]           ; 33 32 31 30
        movd        xmm7,       [rcx + rax*2]           ; 73 72 71 70
        punpckldq   xmm4,       xmm5                    ; 63 62 61 60 23 22 21 20
        punpckldq   xmm6,       xmm7                    ; 73 72 71 70 33 32 31 30

        punpcklbw   xmm0,       xmm2                    ; 53 43 52 42 51 41 50 40 13 03 12 02 11 01 10 00
        punpcklbw   xmm4,       xmm6                    ; 73 63 72 62 71 61 70 60 33 23 32 22 31 21 30 20

        movdqa      xmm1,       xmm0
        punpcklwd   xmm0,       xmm4                    ; 33 23 13 03 32 22 12 02 31 21 11 01 30 20 10 00
        punpckhwd   xmm1,       xmm4                    ; 73 63 53 43 72 62 52 42 71 61 51 41 70 60 50 40

        movdqa      xmm2,       xmm0
        punpckldq   xmm0,       xmm1                    ; 71 61 51 41 31 21 11 01 70 60 50 40 30 20 10 00
        punpckhdq   xmm2,       xmm1                    ; 73 63 53 43 33 23 13 03 72 62 52 42 32 22 12 02

        lea         rsi,        [rsi + rax*8]
        lea         rdi,        [rsi + rax]
        lea         rdx,        [rsi + rax*4]
        lea         rcx,        [rdx + rax]

        movd        xmm4,       [rsi]                   ; 83 82 81 80
        movd        xmm1,       [rdx]                   ; c3 c2 c1 c0
        movd        xmm6,       [rdi]                   ; 93 92 91 90
        movd        xmm3,       [rcx]                   ; d3 d2 d1 d0
        punpckldq   xmm4,       xmm1                    ; c3 c2 c1 c0 83 82 81 80
        punpckldq   xmm6,       xmm3                    ; d3 d2 d1 d0 93 92 91 90

        movd        xmm1,       [rsi + rax*2]           ; a3 a2 a1 a0
        movd        xmm5,       [rdx + rax*2]           ; e3 e2 e1 e0
        movd        xmm3,       [rdi + rax*2]           ; b3 b2 b1 b0
        movd        xmm7,       [rcx + rax*2]           ; f3 f2 f1 f0
        punpckldq   xmm1,       xmm5                    ; e3 e2 e1 e0 a3 a2 a1 a0
        punpckldq   xmm3,       xmm7                    ; f3 f2 f1 f0 b3 b2 b1 b0

        punpcklbw   xmm4,       xmm6                    ; d3 c3 d2 c2 d1 c1 d0 c0 93 83 92 82 91 81 90 80
        punpcklbw   xmm1,       xmm3                    ; f3 e3 f2 e2 f1 e1 f0 e0 b3 a3 b2 a2 b1 a1 b0 a0

        movdqa      xmm7,       xmm4
        punpcklwd   xmm4,       xmm1                    ; b3 a3 93 83 b2 a2 92 82 b1 a1 91 81 b0 a0 90 80
        punpckhwd   xmm7,       xmm1                    ; f3 e3 d3 c3 f2 e2 d2 c2 f1 e1 d1 c1 f0 e0 d0 c0

        movdqa      xmm6,       xmm4
        punpckldq   xmm4,       xmm7                    ; f1 e1 d1 c1 b1 a1 91 81 f0 e0 d0 c0 b0 a0 90 80
        punpckhdq   xmm6,       xmm7                    ; f3 e3 d3 c3 b3 a3 93 83 f2 e2 d2 c2 b2 a2 92 82

        movdqa      xmm1,       xmm0
        movdqa      xmm3,       xmm2

        punpcklqdq  xmm0,       xmm4                    ; p1  f0 e0 d0 c0 b0 a0 90 80 70 60 50 40 30 20 10 00
        punpckhqdq  xmm1,       xmm4                    ; p0  f1 e1 d1 c1 b1 a1 91 81 71 61 51 41 31 21 11 01
        punpcklqdq  xmm2,       xmm6                    ; q0  f2 e2 d2 c2 b2 a2 92 82 72 62 52 42 32 22 12 02
        punpckhqdq  xmm3,       xmm6                    ; q1  f3 e3 d3 c3 b3 a3 93 83 73 63 53 43 33 23 13 03

        mov         rdx,        arg(2)                          ;blimit

        ; calculate mask
        movdqa      xmm6,       xmm0                            ; p1
        movdqa      xmm7,       xmm3                            ; q1
        psubusb     xmm7,       xmm0                            ; q1-=p1
        psubusb     xmm6,       xmm3                            ; p1-=q1
        por         xmm6,       xmm7                            ; abs(p1-q1)
        pand        xmm6,       [GLOBAL(tfe)]                   ; set lsb of each byte to zero
        psrlw       xmm6,       1                               ; abs(p1-q1)/2

        movdqa      xmm7, [rdx]

        movdqa      xmm5,       xmm1                            ; p0
        movdqa      xmm4,       xmm2                            ; q0
        psubusb     xmm5,       xmm2                            ; p0-=q0
        psubusb     xmm4,       xmm1                            ; q0-=p0
        por         xmm5,       xmm4                            ; abs(p0 - q0)
        paddusb     xmm5,       xmm5                            ; abs(p0-q0)*2
        paddusb     xmm5,       xmm6                            ; abs (p0 - q0) *2 + abs(p1-q1)/2

        movdqa      xmm4, [GLOBAL(t80)]

        psubusb     xmm5,        xmm7                           ; abs(p0 - q0) *2 + abs(p1-q1)/2  > blimit
        pxor        xmm7,        xmm7
        pcmpeqb     xmm5,        xmm7                           ; mm5 = mask

        ; start work on filters
        movdqa        t0,        xmm0
        movdqa        t1,        xmm3

        pxor        xmm0,        xmm4                  ; p1 offset to convert to signed values
        pxor        xmm3,        xmm4                  ; q1 offset to convert to signed values
        psubsb      xmm0,        xmm3                           ; p1 - q1

        pxor        xmm1,        xmm4                  ; offset to convert to signed values
        pxor        xmm2,        xmm4                  ; offset to convert to signed values

        movdqa      xmm3,        xmm2                           ; offseted ; q0
        psubsb      xmm2,        xmm1                           ; q0 - p0
        paddsb      xmm0,        xmm2                           ; p1 - q1 + 1 * (q0 - p0)
        paddsb      xmm0,        xmm2                           ; p1 - q1 + 2 * (q0 - p0)
        paddsb      xmm0,        xmm2                           ; p1 - q1 + 3 * (q0 - p0)
        pand        xmm5,        xmm0                           ; mask filter values we don't care about

        movdqa      xmm0, xmm5
        paddsb      xmm5,        [GLOBAL(t3)]                  ;  3* (q0 - p0) + (p1 - q1) + 4
        paddsb      xmm0,        [GLOBAL(t4)]                  ; +3 instead of +4

        movdqa  xmm6, [GLOBAL(te0)]
        movdqa  xmm2, [GLOBAL(t1f)]

;        pxor        xmm7, xmm7
        pcmpgtb     xmm7, xmm0              ;save sign
        pand        xmm7, xmm6              ;preserve the upper 3 bits
        psrlw       xmm0, 3
        pand        xmm0, xmm2              ;clear out upper 3 bits
        por         xmm0, xmm7              ;add sign
        psubsb      xmm3, xmm0              ; q0-= q0sz add

        pxor        xmm7, xmm7
        pcmpgtb     xmm7, xmm5              ;save sign
        pand        xmm7, xmm6              ;preserve the upper 3 bits
        psrlw       xmm5, 3
        pand        xmm5, xmm2              ;clear out upper 3 bits
        por         xmm5, xmm7              ;add sign
        paddsb      xmm1, xmm5              ; p0+= p0 add

        pxor        xmm3,        xmm4                  ; unoffset   q0
        pxor        xmm1,        xmm4                  ; unoffset   p0

        movdqa      xmm0,        t0                             ; p1
        movdqa      xmm4,        t1                             ; q1

        ; write out order: xmm0 xmm2 xmm1 xmm3
        lea         rdx,        [rsi + rax*4]

        ; transpose back to write out
        ; p1  f0 e0 d0 c0 b0 a0 90 80 70 60 50 40 30 20 10 00
        ; p0  f1 e1 d1 c1 b1 a1 91 81 71 61 51 41 31 21 11 01
        ; q0  f2 e2 d2 c2 b2 a2 92 82 72 62 52 42 32 22 12 02
        ; q1  f3 e3 d3 c3 b3 a3 93 83 73 63 53 43 33 23 13 03
        movdqa      xmm6,       xmm0
        punpcklbw   xmm0,       xmm1                               ; 71 70 61 60 51 50 41 40 31 30 21 20 11 10 01 00
        punpckhbw   xmm6,       xmm1                               ; f1 f0 e1 e0 d1 d0 c1 c0 b1 b0 a1 a0 91 90 81 80

        movdqa      xmm5,       xmm3
        punpcklbw   xmm3,       xmm4                               ; 73 72 63 62 53 52 43 42 33 32 23 22 13 12 03 02
        punpckhbw   xmm5,       xmm4                               ; f3 f2 e3 e2 d3 d2 c3 c2 b3 b2 a3 a2 93 92 83 82

        movdqa      xmm2,       xmm0
        punpcklwd   xmm0,       xmm3                               ; 33 32 31 30 23 22 21 20 13 12 11 10 03 02 01 00
        punpckhwd   xmm2,       xmm3                               ; 73 72 71 70 63 62 61 60 53 52 51 50 43 42 41 40

        movdqa      xmm3,       xmm6
        punpcklwd   xmm6,       xmm5                               ; b3 b2 b1 b0 a3 a2 a1 a0 93 92 91 90 83 82 81 80
        punpckhwd   xmm3,       xmm5                               ; f3 f2 f1 f0 e3 e2 e1 e0 d3 d2 d1 d0 c3 c2 c1 c0

        movd        [rsi],      xmm6                               ; write the second 8-line result
        movd        [rdx],      xmm3
        psrldq      xmm6,       4
        psrldq      xmm3,       4
        movd        [rdi],      xmm6
        movd        [rcx],      xmm3
        psrldq      xmm6,       4
        psrldq      xmm3,       4
        movd        [rsi + rax*2], xmm6
        movd        [rdx + rax*2], xmm3
        psrldq      xmm6,       4
        psrldq      xmm3,       4
        movd        [rdi + rax*2], xmm6
        movd        [rcx + rax*2], xmm3

        neg         rax
        lea         rsi,        [rsi + rax*8]
        neg         rax
        lea         rdi,        [rsi + rax]
        lea         rdx,        [rsi + rax*4]
        lea         rcx,        [rdx + rax]

        movd        [rsi],      xmm0                                ; write the first 8-line result
        movd        [rdx],      xmm2
        psrldq      xmm0,       4
        psrldq      xmm2,       4
        movd        [rdi],      xmm0
        movd        [rcx],      xmm2
        psrldq      xmm0,       4
        psrldq      xmm2,       4
        movd        [rsi + rax*2], xmm0
        movd        [rdx + rax*2], xmm2
        psrldq      xmm0,       4
        psrldq      xmm2,       4
        movd        [rdi + rax*2], xmm0
        movd        [rcx + rax*2], xmm2

    add rsp, 32
    pop rsp
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

SECTION_RODATA
align 16
tfe:
    times 16 db 0xfe
align 16
t80:
    times 16 db 0x80
align 16
t1s:
    times 16 db 0x01
align 16
t3:
    times 16 db 0x03
align 16
t4:
    times 16 db 0x04
align 16
ones:
    times 8 dw 0x0001
align 16
s9:
    times 8 dw 0x0900
align 16
s63:
    times 8 dw 0x003f
align 16
te0:
    times 16 db 0xe0
align 16
t1f:
    times 16 db 0x1f
