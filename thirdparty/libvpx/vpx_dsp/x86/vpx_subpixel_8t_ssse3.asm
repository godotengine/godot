;
;  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

%include "third_party/x86inc/x86inc.asm"

SECTION_RODATA
pw_64:    times 8 dw 64

; %define USE_PMULHRSW
; NOTE: pmulhrsw has a latency of 5 cycles.  Tests showed a performance loss
; when using this instruction.
;
; The add order below (based on ffvp9) must be followed to prevent outranges.
; x = k0k1 + k4k5
; y = k2k3 + k6k7
; z = signed SAT(x + y)

SECTION .text
%if ARCH_X86_64
  %define LOCAL_VARS_SIZE 16*4
%else
  %define LOCAL_VARS_SIZE 16*6
%endif

%macro SETUP_LOCAL_VARS 0
    ; TODO(slavarnway): using xmm registers for these on ARCH_X86_64 +
    ; pmaddubsw has a higher latency on some platforms, this might be eased by
    ; interleaving the instructions.
    %define    k0k1  [rsp + 16*0]
    %define    k2k3  [rsp + 16*1]
    %define    k4k5  [rsp + 16*2]
    %define    k6k7  [rsp + 16*3]
    packsswb     m4, m4
    ; TODO(slavarnway): multiple pshufb instructions had a higher latency on
    ; some platforms.
    pshuflw      m0, m4, 0b              ;k0_k1
    pshuflw      m1, m4, 01010101b       ;k2_k3
    pshuflw      m2, m4, 10101010b       ;k4_k5
    pshuflw      m3, m4, 11111111b       ;k6_k7
    punpcklqdq   m0, m0
    punpcklqdq   m1, m1
    punpcklqdq   m2, m2
    punpcklqdq   m3, m3
    mova       k0k1, m0
    mova       k2k3, m1
    mova       k4k5, m2
    mova       k6k7, m3
%if ARCH_X86_64
    %define     krd  m12
    %define     tmp  m13
    mova        krd, [GLOBAL(pw_64)]
%else
    %define     tmp  [rsp + 16*4]
    %define     krd  [rsp + 16*5]
%if CONFIG_PIC=0
    mova         m6, [GLOBAL(pw_64)]
%else
    ; build constants without accessing global memory
    pcmpeqb      m6, m6                  ;all ones
    psrlw        m6, 15
    psllw        m6, 6                   ;aka pw_64
%endif
    mova        krd, m6
%endif
%endm

%macro HORIZx4_ROW 2
    mova      %2, %1
    punpcklbw %1, %1
    punpckhbw %2, %2

    mova      m3, %2
    palignr   %2, %1, 1
    palignr   m3, %1, 5

    pmaddubsw %2, k0k1k4k5
    pmaddubsw m3, k2k3k6k7
    mova      m4, %2        ;k0k1
    mova      m5, m3        ;k2k3
    psrldq    %2, 8         ;k4k5
    psrldq    m3, 8         ;k6k7
    paddsw    %2, m4
    paddsw    m5, m3
    paddsw    %2, m5
    paddsw    %2, krd
    psraw     %2, 7
    packuswb  %2, %2
%endm

;-------------------------------------------------------------------------------
%macro SUBPIX_HFILTER4 1
cglobal filter_block1d4_%1, 6, 6+(ARCH_X86_64*2), 11, LOCAL_VARS_SIZE, \
                            src, sstride, dst, dstride, height, filter
    mova                m4, [filterq]
    packsswb            m4, m4
%if ARCH_X86_64
    %define       k0k1k4k5 m8
    %define       k2k3k6k7 m9
    %define            krd m10
    %define    orig_height r7d
    mova               krd, [GLOBAL(pw_64)]
    pshuflw       k0k1k4k5, m4, 0b              ;k0_k1
    pshufhw       k0k1k4k5, k0k1k4k5, 10101010b ;k0_k1_k4_k5
    pshuflw       k2k3k6k7, m4, 01010101b       ;k2_k3
    pshufhw       k2k3k6k7, k2k3k6k7, 11111111b ;k2_k3_k6_k7
%else
    %define       k0k1k4k5 [rsp + 16*0]
    %define       k2k3k6k7 [rsp + 16*1]
    %define            krd [rsp + 16*2]
    %define    orig_height [rsp + 16*3]
    pshuflw             m6, m4, 0b              ;k0_k1
    pshufhw             m6, m6, 10101010b       ;k0_k1_k4_k5
    pshuflw             m7, m4, 01010101b       ;k2_k3
    pshufhw             m7, m7, 11111111b       ;k2_k3_k6_k7
%if CONFIG_PIC=0
    mova                m1, [GLOBAL(pw_64)]
%else
    ; build constants without accessing global memory
    pcmpeqb             m1, m1                  ;all ones
    psrlw               m1, 15
    psllw               m1, 6                   ;aka pw_64
%endif
    mova          k0k1k4k5, m6
    mova          k2k3k6k7, m7
    mova               krd, m1
%endif
    mov        orig_height, heightd
    shr            heightd, 1
.loop:
    ;Do two rows at once
    movh                m0, [srcq - 3]
    movh                m1, [srcq + 5]
    punpcklqdq          m0, m1
    mova                m1, m0
    movh                m2, [srcq + sstrideq - 3]
    movh                m3, [srcq + sstrideq + 5]
    punpcklqdq          m2, m3
    mova                m3, m2
    punpcklbw           m0, m0
    punpckhbw           m1, m1
    punpcklbw           m2, m2
    punpckhbw           m3, m3
    mova                m4, m1
    palignr             m4, m0,  1
    pmaddubsw           m4, k0k1k4k5
    palignr             m1, m0,  5
    pmaddubsw           m1, k2k3k6k7
    mova                m7, m3
    palignr             m7, m2,  1
    pmaddubsw           m7, k0k1k4k5
    palignr             m3, m2,  5
    pmaddubsw           m3, k2k3k6k7
    mova                m0, m4                  ;k0k1
    mova                m5, m1                  ;k2k3
    mova                m2, m7                  ;k0k1 upper
    psrldq              m4, 8                   ;k4k5
    psrldq              m1, 8                   ;k6k7
    paddsw              m4, m0
    paddsw              m5, m1
    mova                m1, m3                  ;k2k3 upper
    psrldq              m7, 8                   ;k4k5 upper
    psrldq              m3, 8                   ;k6k7 upper
    paddsw              m7, m2
    paddsw              m4, m5
    paddsw              m1, m3
    paddsw              m7, m1
    paddsw              m4, krd
    psraw               m4, 7
    packuswb            m4, m4
    paddsw              m7, krd
    psraw               m7, 7
    packuswb            m7, m7

%ifidn %1, h8_avg
    movd                m0, [dstq]
    pavgb               m4, m0
    movd                m2, [dstq + dstrideq]
    pavgb               m7, m2
%endif
    movd            [dstq], m4
    movd [dstq + dstrideq], m7

    lea               srcq, [srcq + sstrideq        ]
    prefetcht0              [srcq + 4 * sstrideq - 3]
    lea               srcq, [srcq + sstrideq        ]
    lea               dstq, [dstq + 2 * dstrideq    ]
    prefetcht0              [srcq + 2 * sstrideq - 3]

    dec            heightd
    jnz              .loop

    ; Do last row if output_height is odd
    mov            heightd, orig_height
    and            heightd, 1
    je               .done

    movh                m0, [srcq - 3]    ; load src
    movh                m1, [srcq + 5]
    punpcklqdq          m0, m1

    HORIZx4_ROW         m0, m1
%ifidn %1, h8_avg
    movd                m0, [dstq]
    pavgb               m1, m0
%endif
    movd            [dstq], m1
.done
    RET
%endm

%macro HORIZx8_ROW 5
    mova        %2, %1
    punpcklbw   %1, %1
    punpckhbw   %2, %2

    mova        %3, %2
    mova        %4, %2
    mova        %5, %2

    palignr     %2, %1, 1
    palignr     %3, %1, 5
    palignr     %4, %1, 9
    palignr     %5, %1, 13

    pmaddubsw   %2, k0k1
    pmaddubsw   %3, k2k3
    pmaddubsw   %4, k4k5
    pmaddubsw   %5, k6k7
    paddsw      %2, %4
    paddsw      %5, %3
    paddsw      %2, %5
    paddsw      %2, krd
    psraw       %2, 7
    packuswb    %2, %2
    SWAP        %1, %2
%endm

;-------------------------------------------------------------------------------
%macro SUBPIX_HFILTER8 1
cglobal filter_block1d8_%1, 6, 6+(ARCH_X86_64*1), 14, LOCAL_VARS_SIZE, \
                            src, sstride, dst, dstride, height, filter
    mova                 m4, [filterq]
    SETUP_LOCAL_VARS
%if ARCH_X86_64
    %define     orig_height r7d
%else
    %define     orig_height heightmp
%endif
    mov         orig_height, heightd
    shr             heightd, 1

.loop:
    movh                 m0, [srcq - 3]
    movh                 m3, [srcq + 5]
    movh                 m4, [srcq + sstrideq - 3]
    movh                 m7, [srcq + sstrideq + 5]
    punpcklqdq           m0, m3
    mova                 m1, m0
    punpcklbw            m0, m0
    punpckhbw            m1, m1
    mova                 m5, m1
    palignr              m5, m0, 13
    pmaddubsw            m5, k6k7
    mova                 m2, m1
    mova                 m3, m1
    palignr              m1, m0, 1
    pmaddubsw            m1, k0k1
    punpcklqdq           m4, m7
    mova                 m6, m4
    punpcklbw            m4, m4
    palignr              m2, m0, 5
    punpckhbw            m6, m6
    palignr              m3, m0, 9
    mova                 m7, m6
    pmaddubsw            m2, k2k3
    pmaddubsw            m3, k4k5

    palignr              m7, m4, 13
    mova                 m0, m6
    palignr              m0, m4, 5
    pmaddubsw            m7, k6k7
    paddsw               m1, m3
    paddsw               m2, m5
    paddsw               m1, m2
    mova                 m5, m6
    palignr              m6, m4, 1
    pmaddubsw            m0, k2k3
    pmaddubsw            m6, k0k1
    palignr              m5, m4, 9
    paddsw               m1, krd
    pmaddubsw            m5, k4k5
    psraw                m1, 7
    paddsw               m0, m7
%ifidn %1, h8_avg
    movh                 m7, [dstq]
    movh                 m2, [dstq + dstrideq]
%endif
    packuswb             m1, m1
    paddsw               m6, m5
    paddsw               m6, m0
    paddsw               m6, krd
    psraw                m6, 7
    packuswb             m6, m6
%ifidn %1, h8_avg
    pavgb                m1, m7
    pavgb                m6, m2
%endif
    movh             [dstq], m1
    movh  [dstq + dstrideq], m6

    lea                srcq, [srcq + sstrideq        ]
    prefetcht0               [srcq + 4 * sstrideq - 3]
    lea                srcq, [srcq + sstrideq        ]
    lea                dstq, [dstq + 2 * dstrideq    ]
    prefetcht0               [srcq + 2 * sstrideq - 3]
    dec             heightd
    jnz             .loop

    ;Do last row if output_height is odd
    mov             heightd, orig_height
    and             heightd, 1
    je                .done

    movh                 m0, [srcq - 3]
    movh                 m3, [srcq + 5]
    punpcklqdq           m0, m3

    HORIZx8_ROW          m0, m1, m2, m3, m4

%ifidn %1, h8_avg
    movh                 m1, [dstq]
    pavgb                m0, m1
%endif
    movh             [dstq], m0
.done:
    RET
%endm

;-------------------------------------------------------------------------------
%macro SUBPIX_HFILTER16 1
cglobal filter_block1d16_%1, 6, 6+(ARCH_X86_64*0), 14, LOCAL_VARS_SIZE, \
                             src, sstride, dst, dstride, height, filter
    mova          m4, [filterq]
    SETUP_LOCAL_VARS
.loop:
    prefetcht0        [srcq + 2 * sstrideq -3]

    movh          m0, [srcq -  3]
    movh          m4, [srcq +  5]
    movh          m6, [srcq + 13]
    punpcklqdq    m0, m4
    mova          m7, m0
    punpckhbw     m0, m0
    mova          m1, m0
    punpcklqdq    m4, m6
    mova          m3, m0
    punpcklbw     m7, m7

    palignr       m3, m7, 13
    mova          m2, m0
    pmaddubsw     m3, k6k7
    palignr       m0, m7, 1
    pmaddubsw     m0, k0k1
    palignr       m1, m7, 5
    pmaddubsw     m1, k2k3
    palignr       m2, m7, 9
    pmaddubsw     m2, k4k5
    paddsw        m1, m3
    mova          m3, m4
    punpckhbw     m4, m4
    mova          m5, m4
    punpcklbw     m3, m3
    mova          m7, m4
    palignr       m5, m3, 5
    mova          m6, m4
    palignr       m4, m3, 1
    pmaddubsw     m4, k0k1
    pmaddubsw     m5, k2k3
    palignr       m6, m3, 9
    pmaddubsw     m6, k4k5
    palignr       m7, m3, 13
    pmaddubsw     m7, k6k7
    paddsw        m0, m2
    paddsw        m0, m1
%ifidn %1, h8_avg
    mova          m1, [dstq]
%endif
    paddsw        m4, m6
    paddsw        m5, m7
    paddsw        m4, m5
    paddsw        m0, krd
    paddsw        m4, krd
    psraw         m0, 7
    psraw         m4, 7
    packuswb      m0, m4
%ifidn %1, h8_avg
    pavgb         m0, m1
%endif
    lea         srcq, [srcq + sstrideq]
    mova      [dstq], m0
    lea         dstq, [dstq + dstrideq]
    dec      heightd
    jnz        .loop
    RET
%endm

INIT_XMM ssse3
SUBPIX_HFILTER16 h8
SUBPIX_HFILTER16 h8_avg
SUBPIX_HFILTER8  h8
SUBPIX_HFILTER8  h8_avg
SUBPIX_HFILTER4  h8
SUBPIX_HFILTER4  h8_avg

;-------------------------------------------------------------------------------
%macro SUBPIX_VFILTER 2
cglobal filter_block1d%2_%1, 6, 6+(ARCH_X86_64*3), 14, LOCAL_VARS_SIZE, \
                             src, sstride, dst, dstride, height, filter
    mova          m4, [filterq]
    SETUP_LOCAL_VARS
%if ARCH_X86_64
    %define      src1q r7
    %define  sstride6q r8
    %define dst_stride dstrideq
%else
    %define      src1q filterq
    %define  sstride6q dstrideq
    %define dst_stride dstridemp
%endif
    mov       src1q, srcq
    add       src1q, sstrideq
    lea   sstride6q, [sstrideq + sstrideq * 4]
    add   sstride6q, sstrideq                   ;pitch * 6

%ifidn %2, 8
    %define movx movh
%else
    %define movx movd
%endif
.loop:
    movx         m0, [srcq                ]     ;A
    movx         m1, [srcq + sstrideq     ]     ;B
    punpcklbw    m0, m1                         ;A B
    movx         m2, [srcq + sstrideq * 2 ]     ;C
    pmaddubsw    m0, k0k1
    mova         m6, m2
    movx         m3, [src1q + sstrideq * 2]     ;D
    punpcklbw    m2, m3                         ;C D
    pmaddubsw    m2, k2k3
    movx         m4, [srcq + sstrideq * 4 ]     ;E
    mova         m7, m4
    movx         m5, [src1q + sstrideq * 4]     ;F
    punpcklbw    m4, m5                         ;E F
    pmaddubsw    m4, k4k5
    punpcklbw    m1, m6                         ;A B next iter
    movx         m6, [srcq + sstride6q    ]     ;G
    punpcklbw    m5, m6                         ;E F next iter
    punpcklbw    m3, m7                         ;C D next iter
    pmaddubsw    m5, k4k5
    movx         m7, [src1q + sstride6q   ]     ;H
    punpcklbw    m6, m7                         ;G H
    pmaddubsw    m6, k6k7
    pmaddubsw    m3, k2k3
    pmaddubsw    m1, k0k1
    paddsw       m0, m4
    paddsw       m2, m6
    movx         m6, [srcq + sstrideq * 8 ]     ;H next iter
    punpcklbw    m7, m6
    pmaddubsw    m7, k6k7
    paddsw       m0, m2
    paddsw       m0, krd
    psraw        m0, 7
    paddsw       m1, m5
    packuswb     m0, m0

    paddsw       m3, m7
    paddsw       m1, m3
    paddsw       m1, krd
    psraw        m1, 7
    lea        srcq, [srcq + sstrideq * 2 ]
    lea       src1q, [src1q + sstrideq * 2]
    packuswb     m1, m1

%ifidn %1, v8_avg
    movx         m2, [dstq]
    pavgb        m0, m2
%endif
    movx     [dstq], m0
    add        dstq, dst_stride
%ifidn %1, v8_avg
    movx         m3, [dstq]
    pavgb        m1, m3
%endif
    movx     [dstq], m1
    add        dstq, dst_stride
    sub     heightd, 2
    cmp     heightd, 1
    jg        .loop

    cmp     heightd, 0
    je        .done

    movx         m0, [srcq                ]     ;A
    movx         m1, [srcq + sstrideq     ]     ;B
    movx         m6, [srcq + sstride6q    ]     ;G
    punpcklbw    m0, m1                         ;A B
    movx         m7, [src1q + sstride6q   ]     ;H
    pmaddubsw    m0, k0k1
    movx         m2, [srcq + sstrideq * 2 ]     ;C
    punpcklbw    m6, m7                         ;G H
    movx         m3, [src1q + sstrideq * 2]     ;D
    pmaddubsw    m6, k6k7
    movx         m4, [srcq + sstrideq * 4 ]     ;E
    punpcklbw    m2, m3                         ;C D
    movx         m5, [src1q + sstrideq * 4]     ;F
    punpcklbw    m4, m5                         ;E F
    pmaddubsw    m2, k2k3
    pmaddubsw    m4, k4k5
    paddsw       m2, m6
    paddsw       m0, m4
    paddsw       m0, m2
    paddsw       m0, krd
    psraw        m0, 7
    packuswb     m0, m0
%ifidn %1, v8_avg
    movx         m1, [dstq]
    pavgb        m0, m1
%endif
    movx     [dstq], m0
.done:
    RET
%endm

;-------------------------------------------------------------------------------
%macro SUBPIX_VFILTER16 1
cglobal filter_block1d16_%1, 6, 6+(ARCH_X86_64*3), 14, LOCAL_VARS_SIZE, \
                             src, sstride, dst, dstride, height, filter
    mova          m4, [filterq]
    SETUP_LOCAL_VARS
%if ARCH_X86_64
    %define      src1q r7
    %define  sstride6q r8
    %define dst_stride dstrideq
%else
    %define      src1q filterq
    %define  sstride6q dstrideq
    %define dst_stride dstridemp
%endif
    mov        src1q, srcq
    add        src1q, sstrideq
    lea    sstride6q, [sstrideq + sstrideq * 4]
    add    sstride6q, sstrideq                   ;pitch * 6

.loop:
    movh          m0, [srcq                ]     ;A
    movh          m1, [srcq + sstrideq     ]     ;B
    movh          m2, [srcq + sstrideq * 2 ]     ;C
    movh          m3, [src1q + sstrideq * 2]     ;D
    movh          m4, [srcq + sstrideq * 4 ]     ;E
    movh          m5, [src1q + sstrideq * 4]     ;F

    punpcklbw     m0, m1                         ;A B
    movh          m6, [srcq + sstride6q]         ;G
    punpcklbw     m2, m3                         ;C D
    movh          m7, [src1q + sstride6q]        ;H
    punpcklbw     m4, m5                         ;E F
    pmaddubsw     m0, k0k1
    movh          m3, [srcq + 8]                 ;A
    pmaddubsw     m2, k2k3
    punpcklbw     m6, m7                         ;G H
    movh          m5, [srcq + sstrideq + 8]      ;B
    pmaddubsw     m4, k4k5
    punpcklbw     m3, m5                         ;A B
    movh          m7, [srcq + sstrideq * 2 + 8]  ;C
    pmaddubsw     m6, k6k7
    movh          m5, [src1q + sstrideq * 2 + 8] ;D
    punpcklbw     m7, m5                         ;C D
    paddsw        m2, m6
    pmaddubsw     m3, k0k1
    movh          m1, [srcq + sstrideq * 4 + 8]  ;E
    paddsw        m0, m4
    pmaddubsw     m7, k2k3
    movh          m6, [src1q + sstrideq * 4 + 8] ;F
    punpcklbw     m1, m6                         ;E F
    paddsw        m0, m2
    paddsw        m0, krd
    movh          m2, [srcq + sstride6q + 8]     ;G
    pmaddubsw     m1, k4k5
    movh          m5, [src1q + sstride6q + 8]    ;H
    psraw         m0, 7
    punpcklbw     m2, m5                         ;G H
    pmaddubsw     m2, k6k7
%ifidn %1, v8_avg
    mova          m4, [dstq]
%endif
    movh      [dstq], m0
    paddsw        m7, m2
    paddsw        m3, m1
    paddsw        m3, m7
    paddsw        m3, krd
    psraw         m3, 7
    packuswb      m0, m3

    add         srcq, sstrideq
    add        src1q, sstrideq
%ifidn %1, v8_avg
    pavgb         m0, m4
%endif
    mova      [dstq], m0
    add         dstq, dst_stride
    dec      heightd
    jnz        .loop
    RET
%endm

INIT_XMM ssse3
SUBPIX_VFILTER16     v8
SUBPIX_VFILTER16 v8_avg
SUBPIX_VFILTER       v8, 8
SUBPIX_VFILTER   v8_avg, 8
SUBPIX_VFILTER       v8, 4
SUBPIX_VFILTER   v8_avg, 4
