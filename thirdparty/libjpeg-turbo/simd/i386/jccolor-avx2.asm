;
; jccolor.asm - colorspace conversion (AVX2)
;
; Copyright (C) 2009, 2016, 2024, D. R. Commander.
; Copyright (C) 2015, Intel Corporation.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.

%include "jsimdext.inc"

; --------------------------------------------------------------------------

%define SCALEBITS  16

F_0_081 equ  5329                ; FIX(0.08131)
F_0_114 equ  7471                ; FIX(0.11400)
F_0_168 equ 11059                ; FIX(0.16874)
F_0_250 equ 16384                ; FIX(0.25000)
F_0_299 equ 19595                ; FIX(0.29900)
F_0_331 equ 21709                ; FIX(0.33126)
F_0_418 equ 27439                ; FIX(0.41869)
F_0_587 equ 38470                ; FIX(0.58700)
F_0_337 equ (F_0_587 - F_0_250)  ; FIX(0.58700) - FIX(0.25000)

; --------------------------------------------------------------------------
    SECTION     SEG_CONST

    ALIGNZ      32
    GLOBAL_DATA(jconst_rgb_ycc_convert_avx2)

EXTN(jconst_rgb_ycc_convert_avx2):

PW_F0299_F0337  times 8 dw  F_0_299,  F_0_337
PW_F0114_F0250  times 8 dw  F_0_114,  F_0_250
PW_MF016_MF033  times 8 dw -F_0_168, -F_0_331
PW_MF008_MF041  times 8 dw -F_0_081, -F_0_418
PD_ONEHALFM1_CJ times 8 dd  (1 << (SCALEBITS - 1)) - 1 + \
                            (CENTERJSAMPLE << SCALEBITS)
PD_ONEHALF      times 8 dd  (1 << (SCALEBITS - 1))

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32

%include "jccolext-avx2.asm"

%undef RGB_RED
%undef RGB_GREEN
%undef RGB_BLUE
%undef RGB_PIXELSIZE
%define RGB_RED  EXT_RGB_RED
%define RGB_GREEN  EXT_RGB_GREEN
%define RGB_BLUE  EXT_RGB_BLUE
%define RGB_PIXELSIZE  EXT_RGB_PIXELSIZE
%define jsimd_rgb_ycc_convert_avx2  jsimd_extrgb_ycc_convert_avx2
%include "jccolext-avx2.asm"

%undef RGB_RED
%undef RGB_GREEN
%undef RGB_BLUE
%undef RGB_PIXELSIZE
%define RGB_RED  EXT_RGBX_RED
%define RGB_GREEN  EXT_RGBX_GREEN
%define RGB_BLUE  EXT_RGBX_BLUE
%define RGB_PIXELSIZE  EXT_RGBX_PIXELSIZE
%define jsimd_rgb_ycc_convert_avx2  jsimd_extrgbx_ycc_convert_avx2
%include "jccolext-avx2.asm"

%undef RGB_RED
%undef RGB_GREEN
%undef RGB_BLUE
%undef RGB_PIXELSIZE
%define RGB_RED  EXT_BGR_RED
%define RGB_GREEN  EXT_BGR_GREEN
%define RGB_BLUE  EXT_BGR_BLUE
%define RGB_PIXELSIZE  EXT_BGR_PIXELSIZE
%define jsimd_rgb_ycc_convert_avx2  jsimd_extbgr_ycc_convert_avx2
%include "jccolext-avx2.asm"

%undef RGB_RED
%undef RGB_GREEN
%undef RGB_BLUE
%undef RGB_PIXELSIZE
%define RGB_RED  EXT_BGRX_RED
%define RGB_GREEN  EXT_BGRX_GREEN
%define RGB_BLUE  EXT_BGRX_BLUE
%define RGB_PIXELSIZE  EXT_BGRX_PIXELSIZE
%define jsimd_rgb_ycc_convert_avx2  jsimd_extbgrx_ycc_convert_avx2
%include "jccolext-avx2.asm"

%undef RGB_RED
%undef RGB_GREEN
%undef RGB_BLUE
%undef RGB_PIXELSIZE
%define RGB_RED  EXT_XBGR_RED
%define RGB_GREEN  EXT_XBGR_GREEN
%define RGB_BLUE  EXT_XBGR_BLUE
%define RGB_PIXELSIZE  EXT_XBGR_PIXELSIZE
%define jsimd_rgb_ycc_convert_avx2  jsimd_extxbgr_ycc_convert_avx2
%include "jccolext-avx2.asm"

%undef RGB_RED
%undef RGB_GREEN
%undef RGB_BLUE
%undef RGB_PIXELSIZE
%define RGB_RED  EXT_XRGB_RED
%define RGB_GREEN  EXT_XRGB_GREEN
%define RGB_BLUE  EXT_XRGB_BLUE
%define RGB_PIXELSIZE  EXT_XRGB_PIXELSIZE
%define jsimd_rgb_ycc_convert_avx2  jsimd_extxrgb_ycc_convert_avx2
%include "jccolext-avx2.asm"
