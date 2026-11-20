;
;  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

%include "third_party/x86inc/x86inc.asm"

SECTION_RODATA
pw_8: times  8 dw  8
bilin_filter_m_sse2: times  8 dw 16
                     times  8 dw  0
                     times  8 dw 14
                     times  8 dw  2
                     times  8 dw 12
                     times  8 dw  4
                     times  8 dw 10
                     times  8 dw  6
                     times 16 dw  8
                     times  8 dw  6
                     times  8 dw 10
                     times  8 dw  4
                     times  8 dw 12
                     times  8 dw  2
                     times  8 dw 14

bilin_filter_m_ssse3: times  8 db 16,  0
                      times  8 db 14,  2
                      times  8 db 12,  4
                      times  8 db 10,  6
                      times 16 db  8
                      times  8 db  6, 10
                      times  8 db  4, 12
                      times  8 db  2, 14

SECTION .text

; int vpx_sub_pixel_varianceNxh(const uint8_t *src, ptrdiff_t src_stride,
;                               int x_offset, int y_offset,
;                               const uint8_t *ref, ptrdiff_t ref_stride,
;                               int height, unsigned int *sse);
;
; This function returns the SE and stores SSE in the given pointer.

%macro SUM_SSE 6 ; src1, ref1, src2, ref2, sum, sse
  psubw                %3, %4
  psubw                %1, %2
  paddw                %5, %3
  pmaddwd              %3, %3
  paddw                %5, %1
  pmaddwd              %1, %1
  paddd                %6, %3
  paddd                %6, %1
%endmacro

%macro STORE_AND_RET 1
%if %1 > 4
  ; if H=64 and W=16, we have 8 words of each 2(1bit)x64(6bit)x9bit=16bit
  ; in m6, i.e. it _exactly_ fits in a signed word per word in the xmm reg.
  ; We have to sign-extend it before adding the words within the register
  ; and outputing to a dword.
  pcmpgtw              m5, m6           ; mask for 0 > x
  movhlps              m3, m7
  punpcklwd            m4, m6, m5
  punpckhwd            m6, m5           ; sign-extend m6 word->dword
  paddd                m7, m3
  paddd                m6, m4
  pshufd               m3, m7, 0x1
  movhlps              m4, m6
  paddd                m7, m3
  paddd                m6, m4
  mov                  r1, ssem         ; r1 = unsigned int *sse
  pshufd               m4, m6, 0x1
  movd               [r1], m7           ; store sse
  paddd                m6, m4
  movd               raxd, m6           ; store sum as return value
%else ; 4xh
  pshuflw              m4, m6, 0xe
  pshuflw              m3, m7, 0xe
  paddw                m6, m4
  paddd                m7, m3
  pcmpgtw              m5, m6           ; mask for 0 > x
  mov                  r1, ssem         ; r1 = unsigned int *sse
  punpcklwd            m6, m5           ; sign-extend m6 word->dword
  movd               [r1], m7           ; store sse
  pshuflw              m4, m6, 0xe
  paddd                m6, m4
  movd               raxd, m6           ; store sum as return value
%endif
  RET
%endmacro

%macro INC_SRC_BY_SRC_STRIDE  0
%if VPX_ARCH_X86=1 && CONFIG_PIC=1
  add                srcq, src_stridemp
%else
  add                srcq, src_strideq
%endif
%endmacro

%macro SUBPEL_VARIANCE 1-2 0 ; W
%if cpuflag(ssse3)
%define bilin_filter_m bilin_filter_m_ssse3
%define filter_idx_shift 4
%else
%define bilin_filter_m bilin_filter_m_sse2
%define filter_idx_shift 5
%endif
; FIXME(rbultje) only bilinear filters use >8 registers, and ssse3 only uses
; 11, not 13, if the registers are ordered correctly. May make a minor speed
; difference on Win64

%if VPX_ARCH_X86_64
  %if %2 == 1 ; avg
    cglobal sub_pixel_avg_variance%1xh, 9, 10, 13, src, src_stride, \
                                        x_offset, y_offset, ref, ref_stride, \
                                        second_pred, second_stride, height, sse
    %define second_str second_strideq
  %else
    cglobal sub_pixel_variance%1xh, 7, 8, 13, src, src_stride, \
                                    x_offset, y_offset, ref, ref_stride, \
                                    height, sse
  %endif
  %define block_height heightd
  %define bilin_filter sseq
%else
  %if CONFIG_PIC=1
    %if %2 == 1 ; avg
      cglobal sub_pixel_avg_variance%1xh, 7, 7, 13, src, src_stride, \
                                          x_offset, y_offset, ref, ref_stride, \
                                          second_pred, second_stride, height, sse
      %define block_height dword heightm
      %define second_str second_stridemp
    %else
      cglobal sub_pixel_variance%1xh, 7, 7, 13, src, src_stride, \
                                      x_offset, y_offset, ref, ref_stride, \
                                      height, sse
      %define block_height heightd
    %endif

    ; reuse argument stack space
    %define g_bilin_filterm x_offsetm
    %define g_pw_8m y_offsetm

    ;Store bilin_filter and pw_8 location in stack
    %if GET_GOT_DEFINED == 1
      GET_GOT eax
      add esp, 4                ; restore esp
    %endif

    lea ecx, [GLOBAL(bilin_filter_m)]
    mov g_bilin_filterm, ecx

    lea ecx, [GLOBAL(pw_8)]
    mov g_pw_8m, ecx

    LOAD_IF_USED 0, 1         ; load eax, ecx back
  %else
    %if %2 == 1 ; avg
      cglobal sub_pixel_avg_variance%1xh, 7, 7, 13, src, src_stride, \
                                          x_offset, y_offset, \
                                          ref, ref_stride, second_pred, second_stride, \
                                          height, sse
      %define block_height dword heightm
      %define second_str second_stridemp
    %else
      cglobal sub_pixel_variance%1xh, 7, 7, 13, src, src_stride, \
                                      x_offset, y_offset, ref, ref_stride, \
                                      height, sse
      %define block_height heightd
    %endif
    %define bilin_filter bilin_filter_m
  %endif
%endif

%if %1 == 4
  %define movx movd
%else
  %define movx movh
%endif

  ASSERT               %1 <= 16         ; m6 overflows if w > 16
  pxor                 m6, m6           ; sum
  pxor                 m7, m7           ; sse
  ; FIXME(rbultje) if both filters are bilinear, we don't actually use m5; we
  ; could perhaps use it for something more productive then
  pxor                 m5, m5           ; dedicated zero register
%if %1 < 16
  sar                   block_height, 1
%if %2 == 1 ; avg
  shl             second_str, 1
%endif
%endif

  ; FIXME(rbultje) replace by jumptable?
  test          x_offsetd, x_offsetd
  jnz .x_nonzero
  ; x_offset == 0
  test          y_offsetd, y_offsetd
  jnz .x_zero_y_nonzero

  ; x_offset == 0 && y_offset == 0
.x_zero_y_zero_loop:
%if %1 == 16
  movu                 m0, [srcq]
  mova                 m1, [refq]
%if %2 == 1 ; avg
  pavgb                m0, [second_predq]
  punpckhbw            m3, m1, m5
  punpcklbw            m1, m5
%endif
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5

%if %2 == 0 ; !avg
  punpckhbw            m3, m1, m5
  punpcklbw            m1, m5
%endif
  SUM_SSE              m0, m1, m2, m3, m6, m7

  add                srcq, src_strideq
  add                refq, ref_strideq
%else ; %1 < 16
  movx                 m0, [srcq]
%if %2 == 1 ; avg
%if %1 > 4
  movhps               m0, [srcq+src_strideq]
%else ; 4xh
  movx                 m1, [srcq+src_strideq]
  punpckldq            m0, m1
%endif
%else ; !avg
  movx                 m2, [srcq+src_strideq]
%endif

  movx                 m1, [refq]
  movx                 m3, [refq+ref_strideq]

%if %2 == 1 ; avg
%if %1 > 4
  pavgb                m0, [second_predq]
%else
  movh                 m2, [second_predq]
  pavgb                m0, m2
%endif
  punpcklbw            m3, m5
  punpcklbw            m1, m5
%if %1 > 4
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
%else ; 4xh
  punpcklbw            m0, m5
  movhlps              m2, m0
%endif
%else ; !avg
  punpcklbw            m0, m5
  punpcklbw            m2, m5
  punpcklbw            m3, m5
  punpcklbw            m1, m5
%endif
  SUM_SSE              m0, m1, m2, m3, m6, m7

  lea                srcq, [srcq+src_strideq*2]
  lea                refq, [refq+ref_strideq*2]
%endif
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
  dec                   block_height
  jg .x_zero_y_zero_loop
  STORE_AND_RET %1

.x_zero_y_nonzero:
  cmp           y_offsetd, 4
  jne .x_zero_y_nonhalf

  ; x_offset == 0 && y_offset == 0.5
.x_zero_y_half_loop:
%if %1 == 16
  movu                 m0, [srcq]
  movu                 m4, [srcq+src_strideq]
  mova                 m1, [refq]
  pavgb                m0, m4
  punpckhbw            m3, m1, m5
%if %2 == 1 ; avg
  pavgb                m0, [second_predq]
%endif
  punpcklbw            m1, m5
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
  SUM_SSE              m0, m1, m2, m3, m6, m7

  add                srcq, src_strideq
  add                refq, ref_strideq
%else ; %1 < 16
  movx                 m0, [srcq]
  movx                 m2, [srcq+src_strideq]
%if %2 == 1 ; avg
%if %1 > 4
  movhps               m2, [srcq+src_strideq*2]
%else ; 4xh
  movx                 m1, [srcq+src_strideq*2]
  punpckldq            m2, m1
%endif
  movx                 m1, [refq]
%if %1 > 4
  movlhps              m0, m2
%else ; 4xh
  punpckldq            m0, m2
%endif
  movx                 m3, [refq+ref_strideq]
  pavgb                m0, m2
  punpcklbw            m1, m5
%if %1 > 4
  pavgb                m0, [second_predq]
  punpcklbw            m3, m5
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
%else ; 4xh
  movh                 m4, [second_predq]
  pavgb                m0, m4
  punpcklbw            m3, m5
  punpcklbw            m0, m5
  movhlps              m2, m0
%endif
%else ; !avg
  movx                 m4, [srcq+src_strideq*2]
  movx                 m1, [refq]
  pavgb                m0, m2
  movx                 m3, [refq+ref_strideq]
  pavgb                m2, m4
  punpcklbw            m0, m5
  punpcklbw            m2, m5
  punpcklbw            m3, m5
  punpcklbw            m1, m5
%endif
  SUM_SSE              m0, m1, m2, m3, m6, m7

  lea                srcq, [srcq+src_strideq*2]
  lea                refq, [refq+ref_strideq*2]
%endif
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
  dec                   block_height
  jg .x_zero_y_half_loop
  STORE_AND_RET %1

.x_zero_y_nonhalf:
  ; x_offset == 0 && y_offset == bilin interpolation
%if VPX_ARCH_X86_64
  lea        bilin_filter, [GLOBAL(bilin_filter_m)]
%endif
  shl           y_offsetd, filter_idx_shift
%if VPX_ARCH_X86_64 && %1 > 4
  mova                 m8, [bilin_filter+y_offsetq]
%if notcpuflag(ssse3) ; FIXME(rbultje) don't scatter registers on x86-64
  mova                 m9, [bilin_filter+y_offsetq+16]
%endif
  mova                m10, [GLOBAL(pw_8)]
%define filter_y_a m8
%define filter_y_b m9
%define filter_rnd m10
%else ; x86-32 or mmx
%if VPX_ARCH_X86=1 && CONFIG_PIC=1
; x_offset == 0, reuse x_offset reg
%define tempq x_offsetq
  add y_offsetq, g_bilin_filterm
%define filter_y_a [y_offsetq]
%define filter_y_b [y_offsetq+16]
  mov tempq, g_pw_8m
%define filter_rnd [tempq]
%else
  add           y_offsetq, bilin_filter
%define filter_y_a [y_offsetq]
%define filter_y_b [y_offsetq+16]
%define filter_rnd [GLOBAL(pw_8)]
%endif
%endif

.x_zero_y_other_loop:
%if %1 == 16
  movu                 m0, [srcq]
  movu                 m4, [srcq+src_strideq]
  mova                 m1, [refq]
%if cpuflag(ssse3)
  punpckhbw            m2, m0, m4
  punpcklbw            m0, m4
  pmaddubsw            m2, filter_y_a
  pmaddubsw            m0, filter_y_a
  paddw                m2, filter_rnd
  paddw                m0, filter_rnd
%else
  punpckhbw            m2, m0, m5
  punpckhbw            m3, m4, m5
  punpcklbw            m0, m5
  punpcklbw            m4, m5
  ; FIXME(rbultje) instead of out=((num-x)*in1+x*in2+rnd)>>log2(num), we can
  ; also do out=in1+(((num-x)*(in2-in1)+rnd)>>log2(num)). Total number of
  ; instructions is the same (5), but it is 1 mul instead of 2, so might be
  ; slightly faster because of pmullw latency. It would also cut our rodata
  ; tables in half for this function, and save 1-2 registers on x86-64.
  pmullw               m2, filter_y_a
  pmullw               m3, filter_y_b
  paddw                m2, filter_rnd
  pmullw               m0, filter_y_a
  pmullw               m4, filter_y_b
  paddw                m0, filter_rnd
  paddw                m2, m3
  paddw                m0, m4
%endif
  psraw                m2, 4
  psraw                m0, 4
%if %2 == 1 ; avg
  ; FIXME(rbultje) pipeline
  packuswb             m0, m2
  pavgb                m0, [second_predq]
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
%endif
  punpckhbw            m3, m1, m5
  punpcklbw            m1, m5
  SUM_SSE              m0, m1, m2, m3, m6, m7

  add                srcq, src_strideq
  add                refq, ref_strideq
%else ; %1 < 16
  movx                 m0, [srcq]
  movx                 m2, [srcq+src_strideq]
  movx                 m4, [srcq+src_strideq*2]
  movx                 m3, [refq+ref_strideq]
%if cpuflag(ssse3)
  movx                 m1, [refq]
  punpcklbw            m0, m2
  punpcklbw            m2, m4
  pmaddubsw            m0, filter_y_a
  pmaddubsw            m2, filter_y_a
  punpcklbw            m3, m5
  paddw                m2, filter_rnd
  paddw                m0, filter_rnd
%else
  punpcklbw            m0, m5
  punpcklbw            m2, m5
  punpcklbw            m4, m5
  pmullw               m0, filter_y_a
  pmullw               m1, m2, filter_y_b
  punpcklbw            m3, m5
  paddw                m0, filter_rnd
  pmullw               m2, filter_y_a
  pmullw               m4, filter_y_b
  paddw                m0, m1
  paddw                m2, filter_rnd
  movx                 m1, [refq]
  paddw                m2, m4
%endif
  psraw                m0, 4
  psraw                m2, 4
%if %2 == 1 ; avg
  ; FIXME(rbultje) pipeline
%if %1 == 4
  movlhps              m0, m2
%endif
  packuswb             m0, m2
%if %1 > 4
  pavgb                m0, [second_predq]
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
%else ; 4xh
  movh                 m2, [second_predq]
  pavgb                m0, m2
  punpcklbw            m0, m5
  movhlps              m2, m0
%endif
%endif
  punpcklbw            m1, m5
  SUM_SSE              m0, m1, m2, m3, m6, m7

  lea                srcq, [srcq+src_strideq*2]
  lea                refq, [refq+ref_strideq*2]
%endif
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
  dec                   block_height
  jg .x_zero_y_other_loop
%undef filter_y_a
%undef filter_y_b
%undef filter_rnd
  STORE_AND_RET %1

.x_nonzero:
  cmp           x_offsetd, 4
  jne .x_nonhalf
  ; x_offset == 0.5
  test          y_offsetd, y_offsetd
  jnz .x_half_y_nonzero

  ; x_offset == 0.5 && y_offset == 0
.x_half_y_zero_loop:
%if %1 == 16
  movu                 m0, [srcq]
  movu                 m4, [srcq+1]
  mova                 m1, [refq]
  pavgb                m0, m4
  punpckhbw            m3, m1, m5
%if %2 == 1 ; avg
  pavgb                m0, [second_predq]
%endif
  punpcklbw            m1, m5
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
  SUM_SSE              m0, m1, m2, m3, m6, m7

  add                srcq, src_strideq
  add                refq, ref_strideq
%else ; %1 < 16
  movx                 m0, [srcq]
  movx                 m4, [srcq+1]
%if %2 == 1 ; avg
%if %1 > 4
  movhps               m0, [srcq+src_strideq]
  movhps               m4, [srcq+src_strideq+1]
%else ; 4xh
  movx                 m1, [srcq+src_strideq]
  punpckldq            m0, m1
  movx                 m2, [srcq+src_strideq+1]
  punpckldq            m4, m2
%endif
  movx                 m1, [refq]
  movx                 m3, [refq+ref_strideq]
  pavgb                m0, m4
  punpcklbw            m3, m5
%if %1 > 4
  pavgb                m0, [second_predq]
  punpcklbw            m1, m5
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
%else ; 4xh
  movh                 m2, [second_predq]
  pavgb                m0, m2
  punpcklbw            m1, m5
  punpcklbw            m0, m5
  movhlps              m2, m0
%endif
%else ; !avg
  movx                 m2, [srcq+src_strideq]
  movx                 m1, [refq]
  pavgb                m0, m4
  movx                 m4, [srcq+src_strideq+1]
  movx                 m3, [refq+ref_strideq]
  pavgb                m2, m4
  punpcklbw            m0, m5
  punpcklbw            m2, m5
  punpcklbw            m3, m5
  punpcklbw            m1, m5
%endif
  SUM_SSE              m0, m1, m2, m3, m6, m7

  lea                srcq, [srcq+src_strideq*2]
  lea                refq, [refq+ref_strideq*2]
%endif
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
  dec                   block_height
  jg .x_half_y_zero_loop
  STORE_AND_RET %1

.x_half_y_nonzero:
  cmp           y_offsetd, 4
  jne .x_half_y_nonhalf

  ; x_offset == 0.5 && y_offset == 0.5
%if %1 == 16
  movu                 m0, [srcq]
  movu                 m3, [srcq+1]
  add                srcq, src_strideq
  pavgb                m0, m3
.x_half_y_half_loop:
  movu                 m4, [srcq]
  movu                 m3, [srcq+1]
  mova                 m1, [refq]
  pavgb                m4, m3
  punpckhbw            m3, m1, m5
  pavgb                m0, m4
%if %2 == 1 ; avg
  punpcklbw            m1, m5
  pavgb                m0, [second_predq]
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
%else
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
  punpcklbw            m1, m5
%endif
  SUM_SSE              m0, m1, m2, m3, m6, m7
  mova                 m0, m4

  add                srcq, src_strideq
  add                refq, ref_strideq
%else ; %1 < 16
  movx                 m0, [srcq]
  movx                 m3, [srcq+1]
  add                srcq, src_strideq
  pavgb                m0, m3
.x_half_y_half_loop:
  movx                 m2, [srcq]
  movx                 m3, [srcq+1]
%if %2 == 1 ; avg
%if %1 > 4
  movhps               m2, [srcq+src_strideq]
  movhps               m3, [srcq+src_strideq+1]
%else
  movx                 m1, [srcq+src_strideq]
  punpckldq            m2, m1
  movx                 m1, [srcq+src_strideq+1]
  punpckldq            m3, m1
%endif
  pavgb                m2, m3
%if %1 > 4
  movlhps              m0, m2
  movhlps              m4, m2
%else ; 4xh
  punpckldq            m0, m2
  pshuflw              m4, m2, 0xe
%endif
  movx                 m1, [refq]
  pavgb                m0, m2
  movx                 m3, [refq+ref_strideq]
%if %1 > 4
  pavgb                m0, [second_predq]
%else
  movh                 m2, [second_predq]
  pavgb                m0, m2
%endif
  punpcklbw            m3, m5
  punpcklbw            m1, m5
%if %1 > 4
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
%else
  punpcklbw            m0, m5
  movhlps              m2, m0
%endif
%else ; !avg
  movx                 m4, [srcq+src_strideq]
  movx                 m1, [srcq+src_strideq+1]
  pavgb                m2, m3
  pavgb                m4, m1
  pavgb                m0, m2
  pavgb                m2, m4
  movx                 m1, [refq]
  movx                 m3, [refq+ref_strideq]
  punpcklbw            m0, m5
  punpcklbw            m2, m5
  punpcklbw            m3, m5
  punpcklbw            m1, m5
%endif
  SUM_SSE              m0, m1, m2, m3, m6, m7
  mova                 m0, m4

  lea                srcq, [srcq+src_strideq*2]
  lea                refq, [refq+ref_strideq*2]
%endif
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
  dec                   block_height
  jg .x_half_y_half_loop
  STORE_AND_RET %1

.x_half_y_nonhalf:
  ; x_offset == 0.5 && y_offset == bilin interpolation
%if VPX_ARCH_X86_64
  lea        bilin_filter, [GLOBAL(bilin_filter_m)]
%endif
  shl           y_offsetd, filter_idx_shift
%if VPX_ARCH_X86_64 && %1 > 4
  mova                 m8, [bilin_filter+y_offsetq]
%if notcpuflag(ssse3) ; FIXME(rbultje) don't scatter registers on x86-64
  mova                 m9, [bilin_filter+y_offsetq+16]
%endif
  mova                m10, [GLOBAL(pw_8)]
%define filter_y_a m8
%define filter_y_b m9
%define filter_rnd m10
%else  ;x86_32
%if VPX_ARCH_X86=1 && CONFIG_PIC=1
; x_offset == 0.5. We can reuse x_offset reg
%define tempq x_offsetq
  add y_offsetq, g_bilin_filterm
%define filter_y_a [y_offsetq]
%define filter_y_b [y_offsetq+16]
  mov tempq, g_pw_8m
%define filter_rnd [tempq]
%else
  add           y_offsetq, bilin_filter
%define filter_y_a [y_offsetq]
%define filter_y_b [y_offsetq+16]
%define filter_rnd [GLOBAL(pw_8)]
%endif
%endif

%if %1 == 16
  movu                 m0, [srcq]
  movu                 m3, [srcq+1]
  add                srcq, src_strideq
  pavgb                m0, m3
.x_half_y_other_loop:
  movu                 m4, [srcq]
  movu                 m2, [srcq+1]
  mova                 m1, [refq]
  pavgb                m4, m2
%if cpuflag(ssse3)
  punpckhbw            m2, m0, m4
  punpcklbw            m0, m4
  pmaddubsw            m2, filter_y_a
  pmaddubsw            m0, filter_y_a
  paddw                m2, filter_rnd
  paddw                m0, filter_rnd
  psraw                m2, 4
%else
  punpckhbw            m2, m0, m5
  punpckhbw            m3, m4, m5
  pmullw               m2, filter_y_a
  pmullw               m3, filter_y_b
  paddw                m2, filter_rnd
  punpcklbw            m0, m5
  paddw                m2, m3
  punpcklbw            m3, m4, m5
  pmullw               m0, filter_y_a
  pmullw               m3, filter_y_b
  paddw                m0, filter_rnd
  psraw                m2, 4
  paddw                m0, m3
%endif
  punpckhbw            m3, m1, m5
  psraw                m0, 4
%if %2 == 1 ; avg
  ; FIXME(rbultje) pipeline
  packuswb             m0, m2
  pavgb                m0, [second_predq]
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
%endif
  punpcklbw            m1, m5
  SUM_SSE              m0, m1, m2, m3, m6, m7
  mova                 m0, m4

  add                srcq, src_strideq
  add                refq, ref_strideq
%else ; %1 < 16
  movx                 m0, [srcq]
  movx                 m3, [srcq+1]
  add                srcq, src_strideq
  pavgb                m0, m3
%if notcpuflag(ssse3)
  punpcklbw            m0, m5
%endif
.x_half_y_other_loop:
  movx                 m2, [srcq]
  movx                 m1, [srcq+1]
  movx                 m4, [srcq+src_strideq]
  movx                 m3, [srcq+src_strideq+1]
  pavgb                m2, m1
  pavgb                m4, m3
  movx                 m3, [refq+ref_strideq]
%if cpuflag(ssse3)
  movx                 m1, [refq]
  punpcklbw            m0, m2
  punpcklbw            m2, m4
  pmaddubsw            m0, filter_y_a
  pmaddubsw            m2, filter_y_a
  punpcklbw            m3, m5
  paddw                m0, filter_rnd
  paddw                m2, filter_rnd
%else
  punpcklbw            m2, m5
  punpcklbw            m4, m5
  pmullw               m0, filter_y_a
  pmullw               m1, m2, filter_y_b
  punpcklbw            m3, m5
  paddw                m0, filter_rnd
  pmullw               m2, filter_y_a
  paddw                m0, m1
  pmullw               m1, m4, filter_y_b
  paddw                m2, filter_rnd
  paddw                m2, m1
  movx                 m1, [refq]
%endif
  psraw                m0, 4
  psraw                m2, 4
%if %2 == 1 ; avg
  ; FIXME(rbultje) pipeline
%if %1 == 4
  movlhps              m0, m2
%endif
  packuswb             m0, m2
%if %1 > 4
  pavgb                m0, [second_predq]
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
%else
  movh                 m2, [second_predq]
  pavgb                m0, m2
  punpcklbw            m0, m5
  movhlps              m2, m0
%endif
%endif
  punpcklbw            m1, m5
  SUM_SSE              m0, m1, m2, m3, m6, m7
  mova                 m0, m4

  lea                srcq, [srcq+src_strideq*2]
  lea                refq, [refq+ref_strideq*2]
%endif
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
  dec                   block_height
  jg .x_half_y_other_loop
%undef filter_y_a
%undef filter_y_b
%undef filter_rnd
  STORE_AND_RET %1

.x_nonhalf:
  test          y_offsetd, y_offsetd
  jnz .x_nonhalf_y_nonzero

  ; x_offset == bilin interpolation && y_offset == 0
%if VPX_ARCH_X86_64
  lea        bilin_filter, [GLOBAL(bilin_filter_m)]
%endif
  shl           x_offsetd, filter_idx_shift
%if VPX_ARCH_X86_64 && %1 > 4
  mova                 m8, [bilin_filter+x_offsetq]
%if notcpuflag(ssse3) ; FIXME(rbultje) don't scatter registers on x86-64
  mova                 m9, [bilin_filter+x_offsetq+16]
%endif
  mova                m10, [GLOBAL(pw_8)]
%define filter_x_a m8
%define filter_x_b m9
%define filter_rnd m10
%else    ; x86-32
%if VPX_ARCH_X86=1 && CONFIG_PIC=1
;y_offset == 0. We can reuse y_offset reg.
%define tempq y_offsetq
  add x_offsetq, g_bilin_filterm
%define filter_x_a [x_offsetq]
%define filter_x_b [x_offsetq+16]
  mov tempq, g_pw_8m
%define filter_rnd [tempq]
%else
  add           x_offsetq, bilin_filter
%define filter_x_a [x_offsetq]
%define filter_x_b [x_offsetq+16]
%define filter_rnd [GLOBAL(pw_8)]
%endif
%endif

.x_other_y_zero_loop:
%if %1 == 16
  movu                 m0, [srcq]
  movu                 m4, [srcq+1]
  mova                 m1, [refq]
%if cpuflag(ssse3)
  punpckhbw            m2, m0, m4
  punpcklbw            m0, m4
  pmaddubsw            m2, filter_x_a
  pmaddubsw            m0, filter_x_a
  paddw                m2, filter_rnd
  paddw                m0, filter_rnd
%else
  punpckhbw            m2, m0, m5
  punpckhbw            m3, m4, m5
  punpcklbw            m0, m5
  punpcklbw            m4, m5
  pmullw               m2, filter_x_a
  pmullw               m3, filter_x_b
  paddw                m2, filter_rnd
  pmullw               m0, filter_x_a
  pmullw               m4, filter_x_b
  paddw                m0, filter_rnd
  paddw                m2, m3
  paddw                m0, m4
%endif
  psraw                m2, 4
  psraw                m0, 4
%if %2 == 1 ; avg
  ; FIXME(rbultje) pipeline
  packuswb             m0, m2
  pavgb                m0, [second_predq]
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
%endif
  punpckhbw            m3, m1, m5
  punpcklbw            m1, m5
  SUM_SSE              m0, m1, m2, m3, m6, m7

  add                srcq, src_strideq
  add                refq, ref_strideq
%else ; %1 < 16
  movx                 m0, [srcq]
  movx                 m1, [srcq+1]
  movx                 m2, [srcq+src_strideq]
  movx                 m4, [srcq+src_strideq+1]
  movx                 m3, [refq+ref_strideq]
%if cpuflag(ssse3)
  punpcklbw            m0, m1
  movx                 m1, [refq]
  punpcklbw            m2, m4
  pmaddubsw            m0, filter_x_a
  pmaddubsw            m2, filter_x_a
  punpcklbw            m3, m5
  paddw                m0, filter_rnd
  paddw                m2, filter_rnd
%else
  punpcklbw            m0, m5
  punpcklbw            m1, m5
  punpcklbw            m2, m5
  punpcklbw            m4, m5
  pmullw               m0, filter_x_a
  pmullw               m1, filter_x_b
  punpcklbw            m3, m5
  paddw                m0, filter_rnd
  pmullw               m2, filter_x_a
  pmullw               m4, filter_x_b
  paddw                m0, m1
  paddw                m2, filter_rnd
  movx                 m1, [refq]
  paddw                m2, m4
%endif
  psraw                m0, 4
  psraw                m2, 4
%if %2 == 1 ; avg
  ; FIXME(rbultje) pipeline
%if %1 == 4
  movlhps              m0, m2
%endif
  packuswb             m0, m2
%if %1 > 4
  pavgb                m0, [second_predq]
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
%else
  movh                 m2, [second_predq]
  pavgb                m0, m2
  punpcklbw            m0, m5
  movhlps              m2, m0
%endif
%endif
  punpcklbw            m1, m5
  SUM_SSE              m0, m1, m2, m3, m6, m7

  lea                srcq, [srcq+src_strideq*2]
  lea                refq, [refq+ref_strideq*2]
%endif
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
  dec                   block_height
  jg .x_other_y_zero_loop
%undef filter_x_a
%undef filter_x_b
%undef filter_rnd
  STORE_AND_RET %1

.x_nonhalf_y_nonzero:
  cmp           y_offsetd, 4
  jne .x_nonhalf_y_nonhalf

  ; x_offset == bilin interpolation && y_offset == 0.5
%if VPX_ARCH_X86_64
  lea        bilin_filter, [GLOBAL(bilin_filter_m)]
%endif
  shl           x_offsetd, filter_idx_shift
%if VPX_ARCH_X86_64 && %1 > 4
  mova                 m8, [bilin_filter+x_offsetq]
%if notcpuflag(ssse3) ; FIXME(rbultje) don't scatter registers on x86-64
  mova                 m9, [bilin_filter+x_offsetq+16]
%endif
  mova                m10, [GLOBAL(pw_8)]
%define filter_x_a m8
%define filter_x_b m9
%define filter_rnd m10
%else    ; x86-32
%if VPX_ARCH_X86=1 && CONFIG_PIC=1
; y_offset == 0.5. We can reuse y_offset reg.
%define tempq y_offsetq
  add x_offsetq, g_bilin_filterm
%define filter_x_a [x_offsetq]
%define filter_x_b [x_offsetq+16]
  mov tempq, g_pw_8m
%define filter_rnd [tempq]
%else
  add           x_offsetq, bilin_filter
%define filter_x_a [x_offsetq]
%define filter_x_b [x_offsetq+16]
%define filter_rnd [GLOBAL(pw_8)]
%endif
%endif

%if %1 == 16
  movu                 m0, [srcq]
  movu                 m1, [srcq+1]
%if cpuflag(ssse3)
  punpckhbw            m2, m0, m1
  punpcklbw            m0, m1
  pmaddubsw            m2, filter_x_a
  pmaddubsw            m0, filter_x_a
  paddw                m2, filter_rnd
  paddw                m0, filter_rnd
%else
  punpckhbw            m2, m0, m5
  punpckhbw            m3, m1, m5
  punpcklbw            m0, m5
  punpcklbw            m1, m5
  pmullw               m0, filter_x_a
  pmullw               m1, filter_x_b
  paddw                m0, filter_rnd
  pmullw               m2, filter_x_a
  pmullw               m3, filter_x_b
  paddw                m2, filter_rnd
  paddw                m0, m1
  paddw                m2, m3
%endif
  psraw                m0, 4
  psraw                m2, 4
  add                srcq, src_strideq
  packuswb             m0, m2
.x_other_y_half_loop:
  movu                 m4, [srcq]
  movu                 m3, [srcq+1]
%if cpuflag(ssse3)
  mova                 m1, [refq]
  punpckhbw            m2, m4, m3
  punpcklbw            m4, m3
  pmaddubsw            m2, filter_x_a
  pmaddubsw            m4, filter_x_a
  paddw                m2, filter_rnd
  paddw                m4, filter_rnd
  psraw                m2, 4
  psraw                m4, 4
  packuswb             m4, m2
  pavgb                m0, m4
  punpckhbw            m3, m1, m5
  punpcklbw            m1, m5
%else
  punpckhbw            m2, m4, m5
  punpckhbw            m1, m3, m5
  punpcklbw            m4, m5
  punpcklbw            m3, m5
  pmullw               m4, filter_x_a
  pmullw               m3, filter_x_b
  paddw                m4, filter_rnd
  pmullw               m2, filter_x_a
  pmullw               m1, filter_x_b
  paddw                m2, filter_rnd
  paddw                m4, m3
  paddw                m2, m1
  mova                 m1, [refq]
  psraw                m4, 4
  psraw                m2, 4
  punpckhbw            m3, m1, m5
  ; FIXME(rbultje) the repeated pack/unpack here around m0/m2 is because we
  ; have a 1-register shortage to be able to store the backup of the bilin
  ; filtered second line as words as cache for the next line. Packing into
  ; a byte costs 1 pack and 2 unpacks, but saves a register.
  packuswb             m4, m2
  punpcklbw            m1, m5
  pavgb                m0, m4
%endif
%if %2 == 1 ; avg
  ; FIXME(rbultje) pipeline
  pavgb                m0, [second_predq]
%endif
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
  SUM_SSE              m0, m1, m2, m3, m6, m7
  mova                 m0, m4

  add                srcq, src_strideq
  add                refq, ref_strideq
%else ; %1 < 16
  movx                 m0, [srcq]
  movx                 m1, [srcq+1]
%if cpuflag(ssse3)
  punpcklbw            m0, m1
  pmaddubsw            m0, filter_x_a
  paddw                m0, filter_rnd
%else
  punpcklbw            m0, m5
  punpcklbw            m1, m5
  pmullw               m0, filter_x_a
  pmullw               m1, filter_x_b
  paddw                m0, filter_rnd
  paddw                m0, m1
%endif
  add                srcq, src_strideq
  psraw                m0, 4
.x_other_y_half_loop:
  movx                 m2, [srcq]
  movx                 m1, [srcq+1]
  movx                 m4, [srcq+src_strideq]
  movx                 m3, [srcq+src_strideq+1]
%if cpuflag(ssse3)
  punpcklbw            m2, m1
  punpcklbw            m4, m3
  pmaddubsw            m2, filter_x_a
  pmaddubsw            m4, filter_x_a
  movx                 m1, [refq]
  movx                 m3, [refq+ref_strideq]
  paddw                m2, filter_rnd
  paddw                m4, filter_rnd
%else
  punpcklbw            m2, m5
  punpcklbw            m1, m5
  punpcklbw            m4, m5
  punpcklbw            m3, m5
  pmullw               m2, filter_x_a
  pmullw               m1, filter_x_b
  paddw                m2, filter_rnd
  pmullw               m4, filter_x_a
  pmullw               m3, filter_x_b
  paddw                m4, filter_rnd
  paddw                m2, m1
  movx                 m1, [refq]
  paddw                m4, m3
  movx                 m3, [refq+ref_strideq]
%endif
  psraw                m2, 4
  psraw                m4, 4
  pavgw                m0, m2
  pavgw                m2, m4
%if %2 == 1 ; avg
  ; FIXME(rbultje) pipeline - also consider going to bytes here
%if %1 == 4
  movlhps              m0, m2
%endif
  packuswb             m0, m2
%if %1 > 4
  pavgb                m0, [second_predq]
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
%else
  movh                 m2, [second_predq]
  pavgb                m0, m2
  punpcklbw            m0, m5
  movhlps              m2, m0
%endif
%endif
  punpcklbw            m3, m5
  punpcklbw            m1, m5
  SUM_SSE              m0, m1, m2, m3, m6, m7
  mova                 m0, m4

  lea                srcq, [srcq+src_strideq*2]
  lea                refq, [refq+ref_strideq*2]
%endif
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
  dec                   block_height
  jg .x_other_y_half_loop
%undef filter_x_a
%undef filter_x_b
%undef filter_rnd
  STORE_AND_RET %1

.x_nonhalf_y_nonhalf:
%if VPX_ARCH_X86_64
  lea        bilin_filter, [GLOBAL(bilin_filter_m)]
%endif
  shl           x_offsetd, filter_idx_shift
  shl           y_offsetd, filter_idx_shift
%if VPX_ARCH_X86_64 && %1 > 4
  mova                 m8, [bilin_filter+x_offsetq]
%if notcpuflag(ssse3) ; FIXME(rbultje) don't scatter registers on x86-64
  mova                 m9, [bilin_filter+x_offsetq+16]
%endif
  mova                m10, [bilin_filter+y_offsetq]
%if notcpuflag(ssse3) ; FIXME(rbultje) don't scatter registers on x86-64
  mova                m11, [bilin_filter+y_offsetq+16]
%endif
  mova                m12, [GLOBAL(pw_8)]
%define filter_x_a m8
%define filter_x_b m9
%define filter_y_a m10
%define filter_y_b m11
%define filter_rnd m12
%else   ; x86-32
%if VPX_ARCH_X86=1 && CONFIG_PIC=1
; In this case, there is NO unused register. Used src_stride register. Later,
; src_stride has to be loaded from stack when it is needed.
%define tempq src_strideq
  mov tempq, g_bilin_filterm
  add           x_offsetq, tempq
  add           y_offsetq, tempq
%define filter_x_a [x_offsetq]
%define filter_x_b [x_offsetq+16]
%define filter_y_a [y_offsetq]
%define filter_y_b [y_offsetq+16]

  mov tempq, g_pw_8m
%define filter_rnd [tempq]
%else
  add           x_offsetq, bilin_filter
  add           y_offsetq, bilin_filter
%define filter_x_a [x_offsetq]
%define filter_x_b [x_offsetq+16]
%define filter_y_a [y_offsetq]
%define filter_y_b [y_offsetq+16]
%define filter_rnd [GLOBAL(pw_8)]
%endif
%endif

  ; x_offset == bilin interpolation && y_offset == bilin interpolation
%if %1 == 16
  movu                 m0, [srcq]
  movu                 m1, [srcq+1]
%if cpuflag(ssse3)
  punpckhbw            m2, m0, m1
  punpcklbw            m0, m1
  pmaddubsw            m2, filter_x_a
  pmaddubsw            m0, filter_x_a
  paddw                m2, filter_rnd
  paddw                m0, filter_rnd
%else
  punpckhbw            m2, m0, m5
  punpckhbw            m3, m1, m5
  punpcklbw            m0, m5
  punpcklbw            m1, m5
  pmullw               m0, filter_x_a
  pmullw               m1, filter_x_b
  paddw                m0, filter_rnd
  pmullw               m2, filter_x_a
  pmullw               m3, filter_x_b
  paddw                m2, filter_rnd
  paddw                m0, m1
  paddw                m2, m3
%endif
  psraw                m0, 4
  psraw                m2, 4

  INC_SRC_BY_SRC_STRIDE

  packuswb             m0, m2
.x_other_y_other_loop:
%if cpuflag(ssse3)
  movu                 m4, [srcq]
  movu                 m3, [srcq+1]
  mova                 m1, [refq]
  punpckhbw            m2, m4, m3
  punpcklbw            m4, m3
  pmaddubsw            m2, filter_x_a
  pmaddubsw            m4, filter_x_a
  punpckhbw            m3, m1, m5
  paddw                m2, filter_rnd
  paddw                m4, filter_rnd
  psraw                m2, 4
  psraw                m4, 4
  packuswb             m4, m2
  punpckhbw            m2, m0, m4
  punpcklbw            m0, m4
  pmaddubsw            m2, filter_y_a
  pmaddubsw            m0, filter_y_a
  punpcklbw            m1, m5
  paddw                m2, filter_rnd
  paddw                m0, filter_rnd
  psraw                m2, 4
  psraw                m0, 4
%else
  movu                 m3, [srcq]
  movu                 m4, [srcq+1]
  punpckhbw            m1, m3, m5
  punpckhbw            m2, m4, m5
  punpcklbw            m3, m5
  punpcklbw            m4, m5
  pmullw               m3, filter_x_a
  pmullw               m4, filter_x_b
  paddw                m3, filter_rnd
  pmullw               m1, filter_x_a
  pmullw               m2, filter_x_b
  paddw                m1, filter_rnd
  paddw                m3, m4
  paddw                m1, m2
  psraw                m3, 4
  psraw                m1, 4
  packuswb             m4, m3, m1
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
  pmullw               m2, filter_y_a
  pmullw               m1, filter_y_b
  paddw                m2, filter_rnd
  pmullw               m0, filter_y_a
  pmullw               m3, filter_y_b
  paddw                m2, m1
  mova                 m1, [refq]
  paddw                m0, filter_rnd
  psraw                m2, 4
  paddw                m0, m3
  punpckhbw            m3, m1, m5
  psraw                m0, 4
  punpcklbw            m1, m5
%endif
%if %2 == 1 ; avg
  ; FIXME(rbultje) pipeline
  packuswb             m0, m2
  pavgb                m0, [second_predq]
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
%endif
  SUM_SSE              m0, m1, m2, m3, m6, m7
  mova                 m0, m4

  INC_SRC_BY_SRC_STRIDE
  add                refq, ref_strideq
%else ; %1 < 16
  movx                 m0, [srcq]
  movx                 m1, [srcq+1]
%if cpuflag(ssse3)
  punpcklbw            m0, m1
  pmaddubsw            m0, filter_x_a
  paddw                m0, filter_rnd
%else
  punpcklbw            m0, m5
  punpcklbw            m1, m5
  pmullw               m0, filter_x_a
  pmullw               m1, filter_x_b
  paddw                m0, filter_rnd
  paddw                m0, m1
%endif
  psraw                m0, 4
%if cpuflag(ssse3)
  packuswb             m0, m0
%endif

  INC_SRC_BY_SRC_STRIDE

.x_other_y_other_loop:
  movx                 m2, [srcq]
  movx                 m1, [srcq+1]

  INC_SRC_BY_SRC_STRIDE
  movx                 m4, [srcq]
  movx                 m3, [srcq+1]

%if cpuflag(ssse3)
  punpcklbw            m2, m1
  punpcklbw            m4, m3
  pmaddubsw            m2, filter_x_a
  pmaddubsw            m4, filter_x_a
  movx                 m3, [refq+ref_strideq]
  movx                 m1, [refq]
  paddw                m2, filter_rnd
  paddw                m4, filter_rnd
  psraw                m2, 4
  psraw                m4, 4
  packuswb             m2, m2
  packuswb             m4, m4
  punpcklbw            m0, m2
  punpcklbw            m2, m4
  pmaddubsw            m0, filter_y_a
  pmaddubsw            m2, filter_y_a
  punpcklbw            m3, m5
  paddw                m0, filter_rnd
  paddw                m2, filter_rnd
  psraw                m0, 4
  psraw                m2, 4
  punpcklbw            m1, m5
%else
  punpcklbw            m2, m5
  punpcklbw            m1, m5
  punpcklbw            m4, m5
  punpcklbw            m3, m5
  pmullw               m2, filter_x_a
  pmullw               m1, filter_x_b
  paddw                m2, filter_rnd
  pmullw               m4, filter_x_a
  pmullw               m3, filter_x_b
  paddw                m4, filter_rnd
  paddw                m2, m1
  paddw                m4, m3
  psraw                m2, 4
  psraw                m4, 4
  pmullw               m0, filter_y_a
  pmullw               m3, m2, filter_y_b
  paddw                m0, filter_rnd
  pmullw               m2, filter_y_a
  pmullw               m1, m4, filter_y_b
  paddw                m2, filter_rnd
  paddw                m0, m3
  movx                 m3, [refq+ref_strideq]
  paddw                m2, m1
  movx                 m1, [refq]
  psraw                m0, 4
  psraw                m2, 4
  punpcklbw            m3, m5
  punpcklbw            m1, m5
%endif
%if %2 == 1 ; avg
  ; FIXME(rbultje) pipeline
%if %1 == 4
  movlhps              m0, m2
%endif
  packuswb             m0, m2
%if %1 > 4
  pavgb                m0, [second_predq]
  punpckhbw            m2, m0, m5
  punpcklbw            m0, m5
%else
  movh                 m2, [second_predq]
  pavgb                m0, m2
  punpcklbw            m0, m5
  movhlps              m2, m0
%endif
%endif
  SUM_SSE              m0, m1, m2, m3, m6, m7
  mova                 m0, m4

  INC_SRC_BY_SRC_STRIDE
  lea                refq, [refq+ref_strideq*2]
%endif
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
  dec                   block_height
  jg .x_other_y_other_loop
%undef filter_x_a
%undef filter_x_b
%undef filter_y_a
%undef filter_y_b
%undef filter_rnd
%undef movx
  STORE_AND_RET %1
%endmacro

; FIXME(rbultje) the non-bilinear versions (i.e. x=0,8&&y=0,8) are identical
; between the ssse3 and non-ssse3 version. It may make sense to merge their
; code in the sense that the ssse3 version would jump to the appropriate
; location in the sse/2 version, rather than duplicating that code in the
; binary.

INIT_XMM sse2
SUBPEL_VARIANCE  4
SUBPEL_VARIANCE  8
SUBPEL_VARIANCE 16

INIT_XMM ssse3
SUBPEL_VARIANCE  4
SUBPEL_VARIANCE  8
SUBPEL_VARIANCE 16

INIT_XMM sse2
SUBPEL_VARIANCE  4, 1
SUBPEL_VARIANCE  8, 1
SUBPEL_VARIANCE 16, 1

INIT_XMM ssse3
SUBPEL_VARIANCE  4, 1
SUBPEL_VARIANCE  8, 1
SUBPEL_VARIANCE 16, 1
