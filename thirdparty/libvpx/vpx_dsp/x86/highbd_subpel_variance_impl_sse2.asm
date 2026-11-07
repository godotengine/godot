;
;  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
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
  mova                 %4, %3       ; make copies to manipulate to calc sum
  mova                 %2, %1       ; use originals for calc sse
  pmaddwd              %3, %3
  paddw                %4, %2
  pmaddwd              %1, %1
  movhlps              %2, %4
  paddd                %6, %3
  paddw                %4, %2
  pxor                 %2, %2
  pcmpgtw              %2, %4       ; mask for 0 > %4 (sum)
  punpcklwd            %4, %2       ; sign-extend word to dword
  paddd                %6, %1
  paddd                %5, %4

%endmacro

%macro STORE_AND_RET 0
%if mmsize == 16
  ; if H=64 and W=16, we have 8 words of each 2(1bit)x64(6bit)x9bit=16bit
  ; in m6, i.e. it _exactly_ fits in a signed word per word in the xmm reg.
  ; We have to sign-extend it before adding the words within the register
  ; and outputing to a dword.
  movhlps              m3, m7
  movhlps              m4, m6
  paddd                m7, m3
  paddd                m6, m4
  pshufd               m3, m7, 0x1
  pshufd               m4, m6, 0x1
  paddd                m7, m3
  paddd                m6, m4
  mov                  r1, ssem         ; r1 = unsigned int *sse
  movd               [r1], m7           ; store sse
  movd                eax, m6           ; store sum as return value
%endif
  RET
%endmacro

%macro INC_SRC_BY_SRC_STRIDE  0
%if VPX_ARCH_X86=1 && CONFIG_PIC=1
  add                srcq, src_stridemp
  add                srcq, src_stridemp
%else
  lea                srcq, [srcq + src_strideq*2]
%endif
%endmacro

%macro SUBPEL_VARIANCE 1-2 0 ; W
%define bilin_filter_m bilin_filter_m_sse2
%define filter_idx_shift 5


%if VPX_ARCH_X86_64
  %if %2 == 1 ; avg
    cglobal highbd_sub_pixel_avg_variance%1xh, 9, 10, 13, src, src_stride, \
                                      x_offset, y_offset, \
                                      ref, ref_stride, \
                                      second_pred, second_stride, height, sse
    %define second_str second_strideq
  %else
    cglobal highbd_sub_pixel_variance%1xh, 7, 8, 13, src, src_stride, \
                                  x_offset, y_offset, \
                                  ref, ref_stride, height, sse
  %endif
  %define block_height heightd
  %define bilin_filter sseq
%else
  %if CONFIG_PIC=1
    %if %2 == 1 ; avg
      cglobal highbd_sub_pixel_avg_variance%1xh, 7, 7, 13, src, src_stride, \
                                        x_offset, y_offset, \
                                        ref, ref_stride, \
                                        second_pred, second_stride, height, sse
      %define block_height dword heightm
      %define second_str second_stridemp
    %else
      cglobal highbd_sub_pixel_variance%1xh, 7, 7, 13, src, src_stride, \
                                    x_offset, y_offset, \
                                    ref, ref_stride, height, sse
      %define block_height heightd
    %endif

    ; reuse argument stack space
    %define g_bilin_filterm x_offsetm
    %define g_pw_8m y_offsetm

    ; Store bilin_filter and pw_8 location in stack
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
      cglobal highbd_sub_pixel_avg_variance%1xh, 7, 7, 13, src, src_stride, \
                                        x_offset, y_offset, \
                                        ref, ref_stride, \
                                        second_pred, second_stride, height, sse
      %define block_height dword heightm
      %define second_str second_stridemp
    %else
      cglobal highbd_sub_pixel_variance%1xh, 7, 7, 13, src, src_stride, \
                                    x_offset, y_offset, \
                                    ref, ref_stride, height, sse
      %define block_height heightd
    %endif

    %define bilin_filter bilin_filter_m
  %endif
%endif

  ASSERT               %1 <= 16         ; m6 overflows if w > 16
  pxor                 m6, m6           ; sum
  pxor                 m7, m7           ; sse

%if %1 < 16
  sar                   block_height, 1
%endif
%if %2 == 1 ; avg
  shl             second_str, 1
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
  movu                 m2, [srcq + 16]
  mova                 m1, [refq]
  mova                 m3, [refq + 16]
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  pavgw                m2, [second_predq+16]
%endif
  SUM_SSE              m0, m1, m2, m3, m6, m7

  lea                srcq, [srcq + src_strideq*2]
  lea                refq, [refq + ref_strideq*2]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%else ; %1 < 16
  movu                 m0, [srcq]
  movu                 m2, [srcq + src_strideq*2]
  mova                 m1, [refq]
  mova                 m3, [refq + ref_strideq*2]
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  add                second_predq, second_str
  pavgw                m2, [second_predq]
%endif
  SUM_SSE              m0, m1, m2, m3, m6, m7

  lea                srcq, [srcq + src_strideq*4]
  lea                refq, [refq + ref_strideq*4]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%endif
  dec                   block_height
  jg .x_zero_y_zero_loop
  STORE_AND_RET

.x_zero_y_nonzero:
  cmp           y_offsetd, 8
  jne .x_zero_y_nonhalf

  ; x_offset == 0 && y_offset == 0.5
.x_zero_y_half_loop:
%if %1 == 16
  movu                 m0, [srcq]
  movu                 m1, [srcq+16]
  movu                 m4, [srcq+src_strideq*2]
  movu                 m5, [srcq+src_strideq*2+16]
  mova                 m2, [refq]
  mova                 m3, [refq+16]
  pavgw                m0, m4
  pavgw                m1, m5
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  pavgw                m1, [second_predq+16]
%endif
  SUM_SSE              m0, m2, m1, m3, m6, m7

  lea                srcq, [srcq + src_strideq*2]
  lea                refq, [refq + ref_strideq*2]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%else ; %1 < 16
  movu                 m0, [srcq]
  movu                 m1, [srcq+src_strideq*2]
  movu                 m5, [srcq+src_strideq*4]
  mova                 m2, [refq]
  mova                 m3, [refq+ref_strideq*2]
  pavgw                m0, m1
  pavgw                m1, m5
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  add                second_predq, second_str
  pavgw                m1, [second_predq]
%endif
  SUM_SSE              m0, m2, m1, m3, m6, m7

  lea                srcq, [srcq + src_strideq*4]
  lea                refq, [refq + ref_strideq*4]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%endif
  dec                   block_height
  jg .x_zero_y_half_loop
  STORE_AND_RET

.x_zero_y_nonhalf:
  ; x_offset == 0 && y_offset == bilin interpolation
%if VPX_ARCH_X86_64
  lea        bilin_filter, [GLOBAL(bilin_filter_m)]
%endif
  shl           y_offsetd, filter_idx_shift
%if VPX_ARCH_X86_64 && mmsize == 16
  mova                 m8, [bilin_filter+y_offsetq]
  mova                 m9, [bilin_filter+y_offsetq+16]
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
  movu                 m1, [srcq + 16]
  movu                 m4, [srcq+src_strideq*2]
  movu                 m5, [srcq+src_strideq*2+16]
  mova                 m2, [refq]
  mova                 m3, [refq+16]
  ; FIXME(rbultje) instead of out=((num-x)*in1+x*in2+rnd)>>log2(num), we can
  ; also do out=in1+(((num-x)*(in2-in1)+rnd)>>log2(num)). Total number of
  ; instructions is the same (5), but it is 1 mul instead of 2, so might be
  ; slightly faster because of pmullw latency. It would also cut our rodata
  ; tables in half for this function, and save 1-2 registers on x86-64.
  pmullw               m1, filter_y_a
  pmullw               m5, filter_y_b
  paddw                m1, filter_rnd
  pmullw               m0, filter_y_a
  pmullw               m4, filter_y_b
  paddw                m0, filter_rnd
  paddw                m1, m5
  paddw                m0, m4
  psrlw                m1, 4
  psrlw                m0, 4
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  pavgw                m1, [second_predq+16]
%endif
  SUM_SSE              m0, m2, m1, m3, m6, m7

  lea                srcq, [srcq + src_strideq*2]
  lea                refq, [refq + ref_strideq*2]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%else ; %1 < 16
  movu                 m0, [srcq]
  movu                 m1, [srcq+src_strideq*2]
  movu                 m5, [srcq+src_strideq*4]
  mova                 m4, m1
  mova                 m2, [refq]
  mova                 m3, [refq+ref_strideq*2]
  pmullw               m1, filter_y_a
  pmullw               m5, filter_y_b
  paddw                m1, filter_rnd
  pmullw               m0, filter_y_a
  pmullw               m4, filter_y_b
  paddw                m0, filter_rnd
  paddw                m1, m5
  paddw                m0, m4
  psrlw                m1, 4
  psrlw                m0, 4
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  add                second_predq, second_str
  pavgw                m1, [second_predq]
%endif
  SUM_SSE              m0, m2, m1, m3, m6, m7

  lea                srcq, [srcq + src_strideq*4]
  lea                refq, [refq + ref_strideq*4]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%endif
  dec                   block_height
  jg .x_zero_y_other_loop
%undef filter_y_a
%undef filter_y_b
%undef filter_rnd
  STORE_AND_RET

.x_nonzero:
  cmp           x_offsetd, 8
  jne .x_nonhalf
  ; x_offset == 0.5
  test          y_offsetd, y_offsetd
  jnz .x_half_y_nonzero

  ; x_offset == 0.5 && y_offset == 0
.x_half_y_zero_loop:
%if %1 == 16
  movu                 m0, [srcq]
  movu                 m1, [srcq + 16]
  movu                 m4, [srcq + 2]
  movu                 m5, [srcq + 18]
  mova                 m2, [refq]
  mova                 m3, [refq + 16]
  pavgw                m0, m4
  pavgw                m1, m5
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  pavgw                m1, [second_predq+16]
%endif
  SUM_SSE              m0, m2, m1, m3, m6, m7

  lea                srcq, [srcq + src_strideq*2]
  lea                refq, [refq + ref_strideq*2]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%else ; %1 < 16
  movu                 m0, [srcq]
  movu                 m1, [srcq + src_strideq*2]
  movu                 m4, [srcq + 2]
  movu                 m5, [srcq + src_strideq*2 + 2]
  mova                 m2, [refq]
  mova                 m3, [refq + ref_strideq*2]
  pavgw                m0, m4
  pavgw                m1, m5
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  add                second_predq, second_str
  pavgw                m1, [second_predq]
%endif
  SUM_SSE              m0, m2, m1, m3, m6, m7

  lea                srcq, [srcq + src_strideq*4]
  lea                refq, [refq + ref_strideq*4]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%endif
  dec                   block_height
  jg .x_half_y_zero_loop
  STORE_AND_RET

.x_half_y_nonzero:
  cmp           y_offsetd, 8
  jne .x_half_y_nonhalf

  ; x_offset == 0.5 && y_offset == 0.5
%if %1 == 16
  movu                 m0, [srcq]
  movu                 m1, [srcq+16]
  movu                 m2, [srcq+2]
  movu                 m3, [srcq+18]
  lea                srcq, [srcq + src_strideq*2]
  pavgw                m0, m2
  pavgw                m1, m3
.x_half_y_half_loop:
  movu                 m2, [srcq]
  movu                 m3, [srcq + 16]
  movu                 m4, [srcq + 2]
  movu                 m5, [srcq + 18]
  pavgw                m2, m4
  pavgw                m3, m5
  pavgw                m0, m2
  pavgw                m1, m3
  mova                 m4, [refq]
  mova                 m5, [refq + 16]
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  pavgw                m1, [second_predq+16]
%endif
  SUM_SSE              m0, m4, m1, m5, m6, m7
  mova                 m0, m2
  mova                 m1, m3

  lea                srcq, [srcq + src_strideq*2]
  lea                refq, [refq + ref_strideq*2]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%else ; %1 < 16
  movu                 m0, [srcq]
  movu                 m2, [srcq+2]
  lea                srcq, [srcq + src_strideq*2]
  pavgw                m0, m2
.x_half_y_half_loop:
  movu                 m2, [srcq]
  movu                 m3, [srcq + src_strideq*2]
  movu                 m4, [srcq + 2]
  movu                 m5, [srcq + src_strideq*2 + 2]
  pavgw                m2, m4
  pavgw                m3, m5
  pavgw                m0, m2
  pavgw                m2, m3
  mova                 m4, [refq]
  mova                 m5, [refq + ref_strideq*2]
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  add                second_predq, second_str
  pavgw                m2, [second_predq]
%endif
  SUM_SSE              m0, m4, m2, m5, m6, m7
  mova                 m0, m3

  lea                srcq, [srcq + src_strideq*4]
  lea                refq, [refq + ref_strideq*4]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%endif
  dec                   block_height
  jg .x_half_y_half_loop
  STORE_AND_RET

.x_half_y_nonhalf:
  ; x_offset == 0.5 && y_offset == bilin interpolation
%if VPX_ARCH_X86_64
  lea        bilin_filter, [GLOBAL(bilin_filter_m)]
%endif
  shl           y_offsetd, filter_idx_shift
%if VPX_ARCH_X86_64 && mmsize == 16
  mova                 m8, [bilin_filter+y_offsetq]
  mova                 m9, [bilin_filter+y_offsetq+16]
  mova                m10, [GLOBAL(pw_8)]
%define filter_y_a m8
%define filter_y_b m9
%define filter_rnd m10
%else  ; x86_32
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
  movu                 m1, [srcq+16]
  movu                 m2, [srcq+2]
  movu                 m3, [srcq+18]
  lea                srcq, [srcq + src_strideq*2]
  pavgw                m0, m2
  pavgw                m1, m3
.x_half_y_other_loop:
  movu                 m2, [srcq]
  movu                 m3, [srcq+16]
  movu                 m4, [srcq+2]
  movu                 m5, [srcq+18]
  pavgw                m2, m4
  pavgw                m3, m5
  mova                 m4, m2
  mova                 m5, m3
  pmullw               m1, filter_y_a
  pmullw               m3, filter_y_b
  paddw                m1, filter_rnd
  paddw                m1, m3
  pmullw               m0, filter_y_a
  pmullw               m2, filter_y_b
  paddw                m0, filter_rnd
  psrlw                m1, 4
  paddw                m0, m2
  mova                 m2, [refq]
  psrlw                m0, 4
  mova                 m3, [refq+16]
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  pavgw                m1, [second_predq+16]
%endif
  SUM_SSE              m0, m2, m1, m3, m6, m7
  mova                 m0, m4
  mova                 m1, m5

  lea                srcq, [srcq + src_strideq*2]
  lea                refq, [refq + ref_strideq*2]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%else ; %1 < 16
  movu                 m0, [srcq]
  movu                 m2, [srcq+2]
  lea                srcq, [srcq + src_strideq*2]
  pavgw                m0, m2
.x_half_y_other_loop:
  movu                 m2, [srcq]
  movu                 m3, [srcq+src_strideq*2]
  movu                 m4, [srcq+2]
  movu                 m5, [srcq+src_strideq*2+2]
  pavgw                m2, m4
  pavgw                m3, m5
  mova                 m4, m2
  mova                 m5, m3
  pmullw               m4, filter_y_a
  pmullw               m3, filter_y_b
  paddw                m4, filter_rnd
  paddw                m4, m3
  pmullw               m0, filter_y_a
  pmullw               m2, filter_y_b
  paddw                m0, filter_rnd
  psrlw                m4, 4
  paddw                m0, m2
  mova                 m2, [refq]
  psrlw                m0, 4
  mova                 m3, [refq+ref_strideq*2]
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  add                second_predq, second_str
  pavgw                m4, [second_predq]
%endif
  SUM_SSE              m0, m2, m4, m3, m6, m7
  mova                 m0, m5

  lea                srcq, [srcq + src_strideq*4]
  lea                refq, [refq + ref_strideq*4]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%endif
  dec                   block_height
  jg .x_half_y_other_loop
%undef filter_y_a
%undef filter_y_b
%undef filter_rnd
  STORE_AND_RET

.x_nonhalf:
  test          y_offsetd, y_offsetd
  jnz .x_nonhalf_y_nonzero

  ; x_offset == bilin interpolation && y_offset == 0
%if VPX_ARCH_X86_64
  lea        bilin_filter, [GLOBAL(bilin_filter_m)]
%endif
  shl           x_offsetd, filter_idx_shift
%if VPX_ARCH_X86_64 && mmsize == 16
  mova                 m8, [bilin_filter+x_offsetq]
  mova                 m9, [bilin_filter+x_offsetq+16]
  mova                m10, [GLOBAL(pw_8)]
%define filter_x_a m8
%define filter_x_b m9
%define filter_rnd m10
%else    ; x86-32
%if VPX_ARCH_X86=1 && CONFIG_PIC=1
; y_offset == 0. We can reuse y_offset reg.
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
  movu                 m1, [srcq+16]
  movu                 m2, [srcq+2]
  movu                 m3, [srcq+18]
  mova                 m4, [refq]
  mova                 m5, [refq+16]
  pmullw               m1, filter_x_a
  pmullw               m3, filter_x_b
  paddw                m1, filter_rnd
  pmullw               m0, filter_x_a
  pmullw               m2, filter_x_b
  paddw                m0, filter_rnd
  paddw                m1, m3
  paddw                m0, m2
  psrlw                m1, 4
  psrlw                m0, 4
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  pavgw                m1, [second_predq+16]
%endif
  SUM_SSE              m0, m4, m1, m5, m6, m7

  lea                srcq, [srcq+src_strideq*2]
  lea                refq, [refq+ref_strideq*2]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%else ; %1 < 16
  movu                 m0, [srcq]
  movu                 m1, [srcq+src_strideq*2]
  movu                 m2, [srcq+2]
  movu                 m3, [srcq+src_strideq*2+2]
  mova                 m4, [refq]
  mova                 m5, [refq+ref_strideq*2]
  pmullw               m1, filter_x_a
  pmullw               m3, filter_x_b
  paddw                m1, filter_rnd
  pmullw               m0, filter_x_a
  pmullw               m2, filter_x_b
  paddw                m0, filter_rnd
  paddw                m1, m3
  paddw                m0, m2
  psrlw                m1, 4
  psrlw                m0, 4
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  add                second_predq, second_str
  pavgw                m1, [second_predq]
%endif
  SUM_SSE              m0, m4, m1, m5, m6, m7

  lea                srcq, [srcq+src_strideq*4]
  lea                refq, [refq+ref_strideq*4]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%endif
  dec                   block_height
  jg .x_other_y_zero_loop
%undef filter_x_a
%undef filter_x_b
%undef filter_rnd
  STORE_AND_RET

.x_nonhalf_y_nonzero:
  cmp           y_offsetd, 8
  jne .x_nonhalf_y_nonhalf

  ; x_offset == bilin interpolation && y_offset == 0.5
%if VPX_ARCH_X86_64
  lea        bilin_filter, [GLOBAL(bilin_filter_m)]
%endif
  shl           x_offsetd, filter_idx_shift
%if VPX_ARCH_X86_64 && mmsize == 16
  mova                 m8, [bilin_filter+x_offsetq]
  mova                 m9, [bilin_filter+x_offsetq+16]
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
  movu                 m1, [srcq+16]
  movu                 m2, [srcq+2]
  movu                 m3, [srcq+18]
  pmullw               m0, filter_x_a
  pmullw               m2, filter_x_b
  paddw                m0, filter_rnd
  pmullw               m1, filter_x_a
  pmullw               m3, filter_x_b
  paddw                m1, filter_rnd
  paddw                m0, m2
  paddw                m1, m3
  psrlw                m0, 4
  psrlw                m1, 4
  lea                srcq, [srcq+src_strideq*2]
.x_other_y_half_loop:
  movu                 m2, [srcq]
  movu                 m3, [srcq+16]
  movu                 m4, [srcq+2]
  movu                 m5, [srcq+18]
  pmullw               m2, filter_x_a
  pmullw               m4, filter_x_b
  paddw                m2, filter_rnd
  pmullw               m3, filter_x_a
  pmullw               m5, filter_x_b
  paddw                m3, filter_rnd
  paddw                m2, m4
  paddw                m3, m5
  mova                 m4, [refq]
  mova                 m5, [refq+16]
  psrlw                m2, 4
  psrlw                m3, 4
  pavgw                m0, m2
  pavgw                m1, m3
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  pavgw                m1, [second_predq+16]
%endif
  SUM_SSE              m0, m4, m1, m5, m6, m7
  mova                 m0, m2
  mova                 m1, m3

  lea                srcq, [srcq+src_strideq*2]
  lea                refq, [refq+ref_strideq*2]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%else ; %1 < 16
  movu                 m0, [srcq]
  movu                 m2, [srcq+2]
  pmullw               m0, filter_x_a
  pmullw               m2, filter_x_b
  paddw                m0, filter_rnd
  paddw                m0, m2
  psrlw                m0, 4
  lea                srcq, [srcq+src_strideq*2]
.x_other_y_half_loop:
  movu                 m2, [srcq]
  movu                 m3, [srcq+src_strideq*2]
  movu                 m4, [srcq+2]
  movu                 m5, [srcq+src_strideq*2+2]
  pmullw               m2, filter_x_a
  pmullw               m4, filter_x_b
  paddw                m2, filter_rnd
  pmullw               m3, filter_x_a
  pmullw               m5, filter_x_b
  paddw                m3, filter_rnd
  paddw                m2, m4
  paddw                m3, m5
  mova                 m4, [refq]
  mova                 m5, [refq+ref_strideq*2]
  psrlw                m2, 4
  psrlw                m3, 4
  pavgw                m0, m2
  pavgw                m2, m3
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  add                second_predq, second_str
  pavgw                m2, [second_predq]
%endif
  SUM_SSE              m0, m4, m2, m5, m6, m7
  mova                 m0, m3

  lea                srcq, [srcq+src_strideq*4]
  lea                refq, [refq+ref_strideq*4]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%endif
  dec                   block_height
  jg .x_other_y_half_loop
%undef filter_x_a
%undef filter_x_b
%undef filter_rnd
  STORE_AND_RET

.x_nonhalf_y_nonhalf:
; loading filter - this is same as in 8-bit depth
%if VPX_ARCH_X86_64
  lea        bilin_filter, [GLOBAL(bilin_filter_m)]
%endif
  shl           x_offsetd, filter_idx_shift ; filter_idx_shift = 5
  shl           y_offsetd, filter_idx_shift
%if VPX_ARCH_X86_64 && mmsize == 16
  mova                 m8, [bilin_filter+x_offsetq]
  mova                 m9, [bilin_filter+x_offsetq+16]
  mova                m10, [bilin_filter+y_offsetq]
  mova                m11, [bilin_filter+y_offsetq+16]
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
; end of load filter

  ; x_offset == bilin interpolation && y_offset == bilin interpolation
%if %1 == 16
  movu                 m0, [srcq]
  movu                 m2, [srcq+2]
  movu                 m1, [srcq+16]
  movu                 m3, [srcq+18]
  pmullw               m0, filter_x_a
  pmullw               m2, filter_x_b
  paddw                m0, filter_rnd
  pmullw               m1, filter_x_a
  pmullw               m3, filter_x_b
  paddw                m1, filter_rnd
  paddw                m0, m2
  paddw                m1, m3
  psrlw                m0, 4
  psrlw                m1, 4

  INC_SRC_BY_SRC_STRIDE

.x_other_y_other_loop:
  movu                 m2, [srcq]
  movu                 m4, [srcq+2]
  movu                 m3, [srcq+16]
  movu                 m5, [srcq+18]
  pmullw               m2, filter_x_a
  pmullw               m4, filter_x_b
  paddw                m2, filter_rnd
  pmullw               m3, filter_x_a
  pmullw               m5, filter_x_b
  paddw                m3, filter_rnd
  paddw                m2, m4
  paddw                m3, m5
  psrlw                m2, 4
  psrlw                m3, 4
  mova                 m4, m2
  mova                 m5, m3
  pmullw               m0, filter_y_a
  pmullw               m2, filter_y_b
  paddw                m0, filter_rnd
  pmullw               m1, filter_y_a
  pmullw               m3, filter_y_b
  paddw                m0, m2
  paddw                m1, filter_rnd
  mova                 m2, [refq]
  paddw                m1, m3
  psrlw                m0, 4
  psrlw                m1, 4
  mova                 m3, [refq+16]
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  pavgw                m1, [second_predq+16]
%endif
  SUM_SSE              m0, m2, m1, m3, m6, m7
  mova                 m0, m4
  mova                 m1, m5

  INC_SRC_BY_SRC_STRIDE
  lea                refq, [refq + ref_strideq * 2]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%else ; %1 < 16
  movu                 m0, [srcq]
  movu                 m2, [srcq+2]
  pmullw               m0, filter_x_a
  pmullw               m2, filter_x_b
  paddw                m0, filter_rnd
  paddw                m0, m2
  psrlw                m0, 4

  INC_SRC_BY_SRC_STRIDE

.x_other_y_other_loop:
  movu                 m2, [srcq]
  movu                 m4, [srcq+2]
  INC_SRC_BY_SRC_STRIDE
  movu                 m3, [srcq]
  movu                 m5, [srcq+2]
  pmullw               m2, filter_x_a
  pmullw               m4, filter_x_b
  paddw                m2, filter_rnd
  pmullw               m3, filter_x_a
  pmullw               m5, filter_x_b
  paddw                m3, filter_rnd
  paddw                m2, m4
  paddw                m3, m5
  psrlw                m2, 4
  psrlw                m3, 4
  mova                 m4, m2
  mova                 m5, m3
  pmullw               m0, filter_y_a
  pmullw               m2, filter_y_b
  paddw                m0, filter_rnd
  pmullw               m4, filter_y_a
  pmullw               m3, filter_y_b
  paddw                m0, m2
  paddw                m4, filter_rnd
  mova                 m2, [refq]
  paddw                m4, m3
  psrlw                m0, 4
  psrlw                m4, 4
  mova                 m3, [refq+ref_strideq*2]
%if %2 == 1 ; avg
  pavgw                m0, [second_predq]
  add                second_predq, second_str
  pavgw                m4, [second_predq]
%endif
  SUM_SSE              m0, m2, m4, m3, m6, m7
  mova                 m0, m5

  INC_SRC_BY_SRC_STRIDE
  lea                refq, [refq + ref_strideq * 4]
%if %2 == 1 ; avg
  add                second_predq, second_str
%endif
%endif
  dec                   block_height
  jg .x_other_y_other_loop
%undef filter_x_a
%undef filter_x_b
%undef filter_y_a
%undef filter_y_b
%undef filter_rnd
  STORE_AND_RET
%endmacro

INIT_XMM sse2
SUBPEL_VARIANCE  8
SUBPEL_VARIANCE 16

INIT_XMM sse2
SUBPEL_VARIANCE  8, 1
SUBPEL_VARIANCE 16, 1
