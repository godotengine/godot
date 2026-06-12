/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp8/common/filter.h"
#include "vpx_ports/asmdefs_mmi.h"

DECLARE_ALIGNED(8, static const int16_t, vp8_six_tap_mmi[8][6 * 8]) = {
  { 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000 },
  { 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa,
    0x007b, 0x007b, 0x007b, 0x007b, 0x007b, 0x007b, 0x007b, 0x007b,
    0x000c, 0x000c, 0x000c, 0x000c, 0x000c, 0x000c, 0x000c, 0x000c,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000 },
  { 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002,
    0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5,
    0x006c, 0x006c, 0x006c, 0x006c, 0x006c, 0x006c, 0x006c, 0x006c,
    0x0024, 0x0024, 0x0024, 0x0024, 0x0024, 0x0024, 0x0024, 0x0024,
    0xfff8, 0xfff8, 0xfff8, 0xfff8, 0xfff8, 0xfff8, 0xfff8, 0xfff8,
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001 },
  { 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0xfff7, 0xfff7, 0xfff7, 0xfff7, 0xfff7, 0xfff7, 0xfff7, 0xfff7,
    0x005d, 0x005d, 0x005d, 0x005d, 0x005d, 0x005d, 0x005d, 0x005d,
    0x0032, 0x0032, 0x0032, 0x0032, 0x0032, 0x0032, 0x0032, 0x0032,
    0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000 },
  { 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003,
    0xfff0, 0xfff0, 0xfff0, 0xfff0, 0xfff0, 0xfff0, 0xfff0, 0xfff0,
    0x004d, 0x004d, 0x004d, 0x004d, 0x004d, 0x004d, 0x004d, 0x004d,
    0x004d, 0x004d, 0x004d, 0x004d, 0x004d, 0x004d, 0x004d, 0x004d,
    0xfff0, 0xfff0, 0xfff0, 0xfff0, 0xfff0, 0xfff0, 0xfff0, 0xfff0,
    0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003, 0x0003 },
  { 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa,
    0x0032, 0x0032, 0x0032, 0x0032, 0x0032, 0x0032, 0x0032, 0x0032,
    0x005d, 0x005d, 0x005d, 0x005d, 0x005d, 0x005d, 0x005d, 0x005d,
    0xfff7, 0xfff7, 0xfff7, 0xfff7, 0xfff7, 0xfff7, 0xfff7, 0xfff7,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000 },
  { 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001,
    0xfff8, 0xfff8, 0xfff8, 0xfff8, 0xfff8, 0xfff8, 0xfff8, 0xfff8,
    0x0024, 0x0024, 0x0024, 0x0024, 0x0024, 0x0024, 0x0024, 0x0024,
    0x006c, 0x006c, 0x006c, 0x006c, 0x006c, 0x006c, 0x006c, 0x006c,
    0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5, 0xfff5,
    0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002 },
  { 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0x000c, 0x000c, 0x000c, 0x000c, 0x000c, 0x000c, 0x000c, 0x000c,
    0x007b, 0x007b, 0x007b, 0x007b, 0x007b, 0x007b, 0x007b, 0x007b,
    0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa, 0xfffa,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000 }
};

/* Horizontal filter:  pixel_step is 1, output_height and output_width are
   the size of horizontal filtering output, output_height is always H + 5 */
static INLINE void vp8_filter_block1d_h6_mmi(unsigned char *src_ptr,
                                             uint16_t *output_ptr,
                                             unsigned int src_pixels_per_line,
                                             unsigned int output_height,
                                             unsigned int output_width,
                                             const int16_t *vp8_filter) {
  uint64_t tmp[1];
  double ff_ph_40;
#if _MIPS_SIM == _ABIO32
  register double fzero asm("$f0");
  register double ftmp0 asm("$f2");
  register double ftmp1 asm("$f4");
  register double ftmp2 asm("$f6");
  register double ftmp3 asm("$f8");
  register double ftmp4 asm("$f10");
  register double ftmp5 asm("$f12");
  register double ftmp6 asm("$f14");
  register double ftmp7 asm("$f16");
  register double ftmp8 asm("$f18");
  register double ftmp9 asm("$f20");
  register double ftmp10 asm("$f22");
  register double ftmp11 asm("$f24");
#else
  register double fzero asm("$f0");
  register double ftmp0 asm("$f1");
  register double ftmp1 asm("$f2");
  register double ftmp2 asm("$f3");
  register double ftmp3 asm("$f4");
  register double ftmp4 asm("$f5");
  register double ftmp5 asm("$f6");
  register double ftmp6 asm("$f7");
  register double ftmp7 asm("$f8");
  register double ftmp8 asm("$f9");
  register double ftmp9 asm("$f10");
  register double ftmp10 asm("$f11");
  register double ftmp11 asm("$f12");
#endif  // _MIPS_SIM == _ABIO32

  /* clang-format off */
  __asm__ volatile (
    "dli        %[tmp0],        0x0040004000400040                    \n\t"
    "dmtc1      %[tmp0],        %[ff_ph_40]                           \n\t"
    "ldc1       %[ftmp0],       0x00(%[vp8_filter])                   \n\t"
    "ldc1       %[ftmp1],       0x10(%[vp8_filter])                   \n\t"
    "ldc1       %[ftmp2],       0x20(%[vp8_filter])                   \n\t"
    "ldc1       %[ftmp3],       0x30(%[vp8_filter])                   \n\t"
    "ldc1       %[ftmp4],       0x40(%[vp8_filter])                   \n\t"
    "ldc1       %[ftmp5],       0x50(%[vp8_filter])                   \n\t"
    "pxor       %[fzero],       %[fzero],           %[fzero]          \n\t"
    "dli        %[tmp0],        0x07                                  \n\t"
    "dmtc1      %[tmp0],        %[ftmp7]                              \n\t"
    "dli        %[tmp0],        0x08                                  \n\t"
    "dmtc1      %[tmp0],        %[ftmp11]                             \n\t"

    "1:                                                               \n\t"
    "gsldlc1    %[ftmp9],       0x05(%[src_ptr])                      \n\t"
    "gsldrc1    %[ftmp9],       -0x02(%[src_ptr])                     \n\t"
    "gsldlc1    %[ftmp10],      0x06(%[src_ptr])                      \n\t"
    "gsldrc1    %[ftmp10],      -0x01(%[src_ptr])                     \n\t"

    "punpcklbh  %[ftmp6],       %[ftmp9],          %[fzero]           \n\t"
    "pmullh     %[ftmp8],       %[ftmp6],          %[ftmp0]           \n\t"

    "punpckhbh  %[ftmp6],       %[ftmp9],          %[fzero]           \n\t"
    "pmullh     %[ftmp6],       %[ftmp6],          %[ftmp4]           \n\t"
    "paddsh     %[ftmp8],       %[ftmp8],          %[ftmp6]           \n\t"

    "punpcklbh  %[ftmp6],       %[ftmp10],         %[fzero]           \n\t"
    "pmullh     %[ftmp6],       %[ftmp6],          %[ftmp1]           \n\t"
    "paddsh     %[ftmp8],       %[ftmp8],          %[ftmp6]           \n\t"

    "punpckhbh  %[ftmp6],       %[ftmp10],         %[fzero]           \n\t"
    "pmullh     %[ftmp6],       %[ftmp6],          %[ftmp5]           \n\t"
    "paddsh     %[ftmp8],       %[ftmp8],          %[ftmp6]           \n\t"

    "ssrld      %[ftmp10],      %[ftmp10],         %[ftmp11]          \n\t"
    "punpcklbh  %[ftmp6],       %[ftmp10],         %[fzero]           \n\t"
    "pmullh     %[ftmp6],       %[ftmp6],          %[ftmp2]           \n\t"
    "paddsh     %[ftmp8],       %[ftmp8],          %[ftmp6]           \n\t"

    "ssrld      %[ftmp10],      %[ftmp10],         %[ftmp11]          \n\t"
    "punpcklbh  %[ftmp6],       %[ftmp10],         %[fzero]           \n\t"
    "pmullh     %[ftmp6],       %[ftmp6],          %[ftmp3]           \n\t"
    "paddsh     %[ftmp8],       %[ftmp8],          %[ftmp6]           \n\t"

    "paddsh     %[ftmp8],       %[ftmp8],          %[ff_ph_40]        \n\t"
    "psrah      %[ftmp8],       %[ftmp8],          %[ftmp7]           \n\t"
    "packushb   %[ftmp8],       %[ftmp8],          %[fzero]           \n\t"
    "punpcklbh  %[ftmp8],       %[ftmp8],          %[fzero]           \n\t"
    "gssdlc1    %[ftmp8],       0x07(%[output_ptr])                   \n\t"
    "gssdrc1    %[ftmp8],       0x00(%[output_ptr])                   \n\t"

    "addiu      %[output_height], %[output_height], -0x01             \n\t"
    MMI_ADDU(%[output_ptr],  %[output_ptr],    %[output_width])
    MMI_ADDU(%[src_ptr],  %[src_ptr], %[src_pixels_per_line])
    "bnez       %[output_height],               1b                    \n\t"
    : [fzero]"=&f"(fzero),              [ftmp0]"=&f"(ftmp0),
      [ftmp1]"=&f"(ftmp1),              [ftmp2]"=&f"(ftmp2),
      [ftmp3]"=&f"(ftmp3),              [ftmp4]"=&f"(ftmp4),
      [ftmp5]"=&f"(ftmp5),              [ftmp6]"=&f"(ftmp6),
      [ftmp7]"=&f"(ftmp7),              [ftmp8]"=&f"(ftmp8),
      [ftmp9]"=&f"(ftmp9),              [ftmp10]"=&f"(ftmp10),
      [ftmp11]"=&f"(ftmp11),            [tmp0]"=&r"(tmp[0]),
      [output_ptr]"+&r"(output_ptr),    [output_height]"+&r"(output_height),
      [src_ptr]"+&r"(src_ptr),          [ff_ph_40]"=&f"(ff_ph_40)
    : [src_pixels_per_line]"r"((mips_reg)src_pixels_per_line),
      [vp8_filter]"r"(vp8_filter),      [output_width]"r"(output_width)
    : "memory"
    );
  /* clang-format on */
}

/* Horizontal filter:  pixel_step is always W */
static INLINE void vp8_filter_block1dc_v6_mmi(
    uint16_t *src_ptr, unsigned char *output_ptr, unsigned int output_height,
    int output_pitch, unsigned int pixels_per_line, const int16_t *vp8_filter) {
  double ff_ph_40;
  uint64_t tmp[1];
  mips_reg addr[1];

#if _MIPS_SIM == _ABIO32
  register double fzero asm("$f0");
  register double ftmp0 asm("$f2");
  register double ftmp1 asm("$f4");
  register double ftmp2 asm("$f6");
  register double ftmp3 asm("$f8");
  register double ftmp4 asm("$f10");
  register double ftmp5 asm("$f12");
  register double ftmp6 asm("$f14");
  register double ftmp7 asm("$f16");
  register double ftmp8 asm("$f18");
  register double ftmp9 asm("$f20");
  register double ftmp10 asm("$f22");
  register double ftmp11 asm("$f24");
  register double ftmp12 asm("$f26");
  register double ftmp13 asm("$f28");
#else
  register double fzero asm("$f0");
  register double ftmp0 asm("$f1");
  register double ftmp1 asm("$f2");
  register double ftmp2 asm("$f3");
  register double ftmp3 asm("$f4");
  register double ftmp4 asm("$f5");
  register double ftmp5 asm("$f6");
  register double ftmp6 asm("$f7");
  register double ftmp7 asm("$f8");
  register double ftmp8 asm("$f9");
  register double ftmp9 asm("$f10");
  register double ftmp10 asm("$f11");
  register double ftmp11 asm("$f12");
  register double ftmp12 asm("$f13");
  register double ftmp13 asm("$f14");
#endif  // _MIPS_SIM == _ABIO32

  /* clang-format off */
  __asm__ volatile (
    "dli        %[tmp0],      0x0040004000400040                      \n\t"
    "dmtc1      %[tmp0],      %[ff_ph_40]                             \n\t"
    "ldc1       %[ftmp0],     0x00(%[vp8_filter])                     \n\t"
    "ldc1       %[ftmp1],     0x10(%[vp8_filter])                     \n\t"
    "ldc1       %[ftmp2],     0x20(%[vp8_filter])                     \n\t"
    "ldc1       %[ftmp3],     0x30(%[vp8_filter])                     \n\t"
    "ldc1       %[ftmp4],     0x40(%[vp8_filter])                     \n\t"
    "ldc1       %[ftmp5],     0x50(%[vp8_filter])                     \n\t"
    "pxor       %[fzero],     %[fzero],        %[fzero]               \n\t"
    "dli        %[tmp0],      0x07                                    \n\t"
    "dmtc1      %[tmp0],      %[ftmp13]                               \n\t"

    /* In order to make full use of memory load delay slot,
     * Operation of memory loading and calculating has been rearranged.
     */
    "1:                                                               \n\t"
    "gsldlc1    %[ftmp6],     0x07(%[src_ptr])                        \n\t"
    "gsldrc1    %[ftmp6],     0x00(%[src_ptr])                        \n\t"
    MMI_ADDU(%[addr0],     %[src_ptr],      %[pixels_per_line])
    "gsldlc1    %[ftmp7],     0x07(%[addr0])                          \n\t"
    "gsldrc1    %[ftmp7],     0x00(%[addr0])                          \n\t"
    MMI_ADDU(%[addr0],     %[src_ptr],      %[pixels_per_line_x2])
    "gsldlc1    %[ftmp8],     0x07(%[addr0])                          \n\t"
    "gsldrc1    %[ftmp8],     0x00(%[addr0])                          \n\t"

    MMI_ADDU(%[addr0],     %[src_ptr],      %[pixels_per_line_x4])
    "gsldlc1    %[ftmp9],     0x07(%[addr0])                          \n\t"
    "gsldrc1    %[ftmp9],     0x00(%[addr0])                          \n\t"
    MMI_ADDU(%[src_ptr],   %[src_ptr],      %[pixels_per_line])
    MMI_ADDU(%[addr0],     %[src_ptr],      %[pixels_per_line_x2])
    "gsldlc1    %[ftmp10],    0x07(%[addr0])                          \n\t"
    "gsldrc1    %[ftmp10],    0x00(%[addr0])                          \n\t"
    MMI_ADDU(%[addr0],     %[src_ptr],      %[pixels_per_line_x4])
    "gsldlc1    %[ftmp11],    0x07(%[addr0])                          \n\t"
    "gsldrc1    %[ftmp11],    0x00(%[addr0])                          \n\t"

    "pmullh     %[ftmp12],    %[ftmp6],        %[ftmp0]               \n\t"

    "pmullh     %[ftmp7],     %[ftmp7],        %[ftmp1]               \n\t"
    "paddsh     %[ftmp12],    %[ftmp12],       %[ftmp7]               \n\t"

    "pmullh     %[ftmp8],     %[ftmp8],        %[ftmp2]               \n\t"
    "paddsh     %[ftmp12],    %[ftmp12],       %[ftmp8]               \n\t"

    "pmullh     %[ftmp9],     %[ftmp9],        %[ftmp4]               \n\t"
    "paddsh     %[ftmp12],    %[ftmp12],       %[ftmp9]               \n\t"

    "pmullh     %[ftmp10],    %[ftmp10],       %[ftmp3]               \n\t"
    "paddsh     %[ftmp12],    %[ftmp12],       %[ftmp10]              \n\t"

    "pmullh     %[ftmp11],    %[ftmp11],       %[ftmp5]               \n\t"
    "paddsh     %[ftmp12],    %[ftmp12],       %[ftmp11]              \n\t"

    "paddsh     %[ftmp12],    %[ftmp12],       %[ff_ph_40]            \n\t"
    "psrah      %[ftmp12],    %[ftmp12],       %[ftmp13]              \n\t"
    "packushb   %[ftmp12],    %[ftmp12],       %[fzero]               \n\t"
    "gsswlc1    %[ftmp12],    0x03(%[output_ptr])                     \n\t"
    "gsswrc1    %[ftmp12],    0x00(%[output_ptr])                     \n\t"

    MMI_ADDIU(%[output_height], %[output_height], -0x01)
    MMI_ADDU(%[output_ptr], %[output_ptr], %[output_pitch])
    "bnez       %[output_height], 1b                                  \n\t"
    : [fzero]"=&f"(fzero),              [ftmp0]"=&f"(ftmp0),
      [ftmp1]"=&f"(ftmp1),              [ftmp2]"=&f"(ftmp2),
      [ftmp3]"=&f"(ftmp3),              [ftmp4]"=&f"(ftmp4),
      [ftmp5]"=&f"(ftmp5),              [ftmp6]"=&f"(ftmp6),
      [ftmp7]"=&f"(ftmp7),              [ftmp8]"=&f"(ftmp8),
      [ftmp9]"=&f"(ftmp9),              [ftmp10]"=&f"(ftmp10),
      [ftmp11]"=&f"(ftmp11),            [ftmp12]"=&f"(ftmp12),
      [ftmp13]"=&f"(ftmp13),            [tmp0]"=&r"(tmp[0]),
      [addr0]"=&r"(addr[0]),            [src_ptr]"+&r"(src_ptr),
      [output_ptr]"+&r"(output_ptr),    [output_height]"+&r"(output_height),
      [ff_ph_40]"=&f"(ff_ph_40)
    : [pixels_per_line]"r"((mips_reg)pixels_per_line),
      [pixels_per_line_x2]"r"((mips_reg)(pixels_per_line<<1)),
      [pixels_per_line_x4]"r"((mips_reg)(pixels_per_line<<2)),
      [vp8_filter]"r"(vp8_filter),
      [output_pitch]"r"((mips_reg)output_pitch)
    : "memory"
    );
  /* clang-format on */
}

/* When xoffset == 0, vp8_filter= {0,0,128,0,0,0},
   function vp8_filter_block1d_h6_mmi and vp8_filter_block1d_v6_mmi can
   be simplified */
static INLINE void vp8_filter_block1d_h6_filter0_mmi(
    unsigned char *src_ptr, uint16_t *output_ptr,
    unsigned int src_pixels_per_line, unsigned int output_height,
    unsigned int output_width) {
#if _MIPS_SIM == _ABIO32
  register double fzero asm("$f0");
  register double ftmp0 asm("$f2");
  register double ftmp1 asm("$f4");
#else
  register double fzero asm("$f0");
  register double ftmp0 asm("$f1");
  register double ftmp1 asm("$f2");
#endif  // _MIPS_SIM == _ABIO32

  /* clang-format off */
  __asm__ volatile (
    "pxor       %[fzero],       %[fzero],           %[fzero]          \n\t"

    "1:                                                               \n\t"
    "gsldlc1    %[ftmp0],       0x07(%[src_ptr])                      \n\t"
    "gsldrc1    %[ftmp0],       0x00(%[src_ptr])                      \n\t"
    MMI_ADDU(%[src_ptr],  %[src_ptr], %[src_pixels_per_line])

    "punpcklbh  %[ftmp1],       %[ftmp0],          %[fzero]           \n\t"
    "gssdlc1    %[ftmp1],       0x07(%[output_ptr])                   \n\t"
    "gssdrc1    %[ftmp1],       0x00(%[output_ptr])                   \n\t"

    "addiu      %[output_height], %[output_height], -0x01             \n\t"
    MMI_ADDU(%[output_ptr],  %[output_ptr],    %[output_width])
    "bnez       %[output_height],               1b                    \n\t"
    : [fzero]"=&f"(fzero),              [ftmp0]"=&f"(ftmp0),
      [ftmp1]"=&f"(ftmp1),              [src_ptr]"+&r"(src_ptr),
      [output_ptr]"+&r"(output_ptr),    [output_height]"+&r"(output_height)
    : [src_pixels_per_line]"r"((mips_reg)src_pixels_per_line),
      [output_width]"r"(output_width)
    : "memory"
    );
  /* clang-format on */
}

static INLINE void vp8_filter_block1dc_v6_filter0_mmi(
    uint16_t *src_ptr, unsigned char *output_ptr, unsigned int output_height,
    int output_pitch, unsigned int pixels_per_line) {
#if _MIPS_SIM == _ABIO32
  register double fzero asm("$f0");
  register double ftmp0 asm("$f2");
  register double ftmp1 asm("$f4");
#else
  register double fzero asm("$f0");
  register double ftmp0 asm("$f1");
  register double ftmp1 asm("$f2");
#endif  // _MIPS_SIM == _ABIO32

  /* clang-format on */
  __asm__ volatile (
    "pxor       %[fzero],     %[fzero],        %[fzero]               \n\t"

    "1:                                                               \n\t"
    "gsldlc1    %[ftmp0],     0x07(%[src_ptr])                        \n\t"
    "gsldrc1    %[ftmp0],     0x00(%[src_ptr])                        \n\t"
    MMI_ADDU(%[src_ptr],   %[src_ptr],      %[pixels_per_line])
    MMI_ADDIU(%[output_height], %[output_height], -0x01)
    "packushb   %[ftmp1],     %[ftmp0],        %[fzero]               \n\t"
    "gsswlc1    %[ftmp1],     0x03(%[output_ptr])                     \n\t"
    "gsswrc1    %[ftmp1],     0x00(%[output_ptr])                     \n\t"

    MMI_ADDU(%[output_ptr], %[output_ptr], %[output_pitch])
    "bnez       %[output_height], 1b                                  \n\t"
    : [fzero]"=&f"(fzero),              [ftmp0]"=&f"(ftmp0),
      [ftmp1]"=&f"(ftmp1),              [src_ptr]"+&r"(src_ptr),
      [output_ptr]"+&r"(output_ptr),    [output_height]"+&r"(output_height)
    : [pixels_per_line]"r"((mips_reg)pixels_per_line),
      [output_pitch]"r"((mips_reg)output_pitch)
    : "memory"
    );
  /* clang-format on */
}

#define sixtapNxM(n, m)                                                        \
  void vp8_sixtap_predict##n##x##m##_mmi(                                      \
      unsigned char *src_ptr, int src_pixels_per_line, int xoffset,            \
      int yoffset, unsigned char *dst_ptr, int dst_pitch) {                    \
    DECLARE_ALIGNED(16, uint16_t,                                              \
                    FData2[(n + 5) * (n == 16 ? 24 : (n == 8 ? 16 : n))]);     \
    const int16_t *HFilter, *VFilter;                                          \
    int i, loop = n / 4;                                                       \
    HFilter = vp8_six_tap_mmi[xoffset];                                        \
    VFilter = vp8_six_tap_mmi[yoffset];                                        \
                                                                               \
    if (xoffset == 0) {                                                        \
      for (i = 0; i < loop; ++i) {                                             \
        vp8_filter_block1d_h6_filter0_mmi(                                     \
            src_ptr - (2 * src_pixels_per_line) + i * 4, FData2 + i * 4,       \
            src_pixels_per_line, m + 5, n * 2);                                \
      }                                                                        \
    } else {                                                                   \
      for (i = 0; i < loop; ++i) {                                             \
        vp8_filter_block1d_h6_mmi(src_ptr - (2 * src_pixels_per_line) + i * 4, \
                                  FData2 + i * 4, src_pixels_per_line, m + 5,  \
                                  n * 2, HFilter);                             \
      }                                                                        \
    }                                                                          \
    if (yoffset == 0) {                                                        \
      for (i = 0; i < loop; ++i) {                                             \
        vp8_filter_block1dc_v6_filter0_mmi(                                    \
            FData2 + n * 2 + i * 4, dst_ptr + i * 4, m, dst_pitch, n * 2);     \
      }                                                                        \
    } else {                                                                   \
      for (i = 0; i < loop; ++i) {                                             \
        vp8_filter_block1dc_v6_mmi(FData2 + i * 4, dst_ptr + i * 4, m,         \
                                   dst_pitch, n * 2, VFilter);                 \
      }                                                                        \
    }                                                                          \
  }

sixtapNxM(4, 4);
sixtapNxM(8, 8);
sixtapNxM(8, 4);
sixtapNxM(16, 16);
