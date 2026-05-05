##
## Copyright (c) 2015 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

DSP_SRCS-yes += vpx_dsp.mk
DSP_SRCS-yes += vpx_dsp_common.h

DSP_SRCS-$(HAVE_MSA)    += mips/macros_msa.h

DSP_SRCS-$(HAVE_AVX2)   += x86/bitdepth_conversion_avx2.h
DSP_SRCS-$(HAVE_SSE2)   += x86/bitdepth_conversion_sse2.h
# This file is included in libs.mk. Including it here would cause it to be
# compiled into an object. Even as an empty file, this would create an
# executable section on the stack.
#DSP_SRCS-$(HAVE_SSE2)   += x86/bitdepth_conversion_sse2$(ASM)

# bit reader
DSP_SRCS-yes += prob.h
DSP_SRCS-yes += prob.c

ifeq ($(CONFIG_ENCODERS),yes)
DSP_SRCS-yes += bitwriter.h
DSP_SRCS-yes += bitwriter.c
DSP_SRCS-yes += bitwriter_buffer.c
DSP_SRCS-yes += bitwriter_buffer.h
DSP_SRCS-yes += psnr.c
DSP_SRCS-yes += psnr.h
DSP_SRCS-yes += sse.c
DSP_SRCS-$(CONFIG_INTERNAL_STATS) += ssim.c
DSP_SRCS-$(CONFIG_INTERNAL_STATS) += ssim.h
DSP_SRCS-$(CONFIG_INTERNAL_STATS) += psnrhvs.c
DSP_SRCS-$(CONFIG_INTERNAL_STATS) += fastssim.c
DSP_SRCS-$(HAVE_NEON) += arm/sse_neon.c
DSP_SRCS-$(HAVE_NEON_DOTPROD) += arm/sse_neon_dotprod.c
DSP_SRCS-$(HAVE_SSE4_1) += x86/sse_sse4.c
DSP_SRCS-$(HAVE_AVX2) += x86/sse_avx2.c
endif

ifeq ($(CONFIG_DECODERS),yes)
DSP_SRCS-yes += bitreader.h
DSP_SRCS-yes += bitreader.c
DSP_SRCS-yes += bitreader_buffer.c
DSP_SRCS-yes += bitreader_buffer.h
endif

# intra predictions
DSP_SRCS-yes += intrapred.c

DSP_SRCS-$(HAVE_SSE2) += x86/intrapred_sse2.asm
DSP_SRCS-$(HAVE_SSSE3) += x86/intrapred_ssse3.asm
DSP_SRCS-$(HAVE_VSX) += ppc/intrapred_vsx.c

ifeq ($(CONFIG_VP9_HIGHBITDEPTH),yes)
DSP_SRCS-$(HAVE_SSE2) += x86/highbd_intrapred_sse2.asm
DSP_SRCS-$(HAVE_SSE2) += x86/highbd_intrapred_intrin_sse2.c
DSP_SRCS-$(HAVE_SSSE3) += x86/highbd_intrapred_intrin_ssse3.c
DSP_SRCS-$(HAVE_NEON) += arm/highbd_intrapred_neon.c
endif  # CONFIG_VP9_HIGHBITDEPTH

ifneq ($(filter yes,$(CONFIG_POSTPROC) $(CONFIG_VP9_POSTPROC)),)
DSP_SRCS-yes += add_noise.c
DSP_SRCS-yes += deblock.c
DSP_SRCS-yes += postproc.h
DSP_SRCS-$(HAVE_MSA) += mips/add_noise_msa.c
DSP_SRCS-$(HAVE_MSA) += mips/deblock_msa.c
DSP_SRCS-$(HAVE_NEON) += arm/deblock_neon.c
DSP_SRCS-$(HAVE_SSE2) += x86/add_noise_sse2.asm
DSP_SRCS-$(HAVE_SSE2) += x86/deblock_sse2.asm
DSP_SRCS-$(HAVE_SSE2) += x86/post_proc_sse2.c
DSP_SRCS-$(HAVE_VSX) += ppc/deblock_vsx.c
endif # CONFIG_POSTPROC

DSP_SRCS-$(HAVE_NEON_ASM) += arm/intrapred_neon_asm$(ASM)
DSP_SRCS-$(HAVE_NEON) += arm/intrapred_neon.c
DSP_SRCS-$(HAVE_MSA) += mips/intrapred_msa.c
DSP_SRCS-$(HAVE_LSX) += loongarch/intrapred_lsx.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/intrapred4_dspr2.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/intrapred8_dspr2.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/intrapred16_dspr2.c

DSP_SRCS-$(HAVE_DSPR2)  += mips/common_dspr2.h
DSP_SRCS-$(HAVE_DSPR2)  += mips/common_dspr2.c

DSP_SRCS-yes += vpx_filter.h
ifeq ($(CONFIG_VP9),yes)
# interpolation filters
DSP_SRCS-yes += vpx_convolve.c
DSP_SRCS-yes += vpx_convolve.h

DSP_SRCS-$(VPX_ARCH_X86)$(VPX_ARCH_X86_64) += x86/convolve.h

DSP_SRCS-$(HAVE_SSE2) += x86/convolve_sse2.h
DSP_SRCS-$(HAVE_SSSE3) += x86/convolve_ssse3.h
DSP_SRCS-$(HAVE_AVX2) += x86/convolve_avx2.h
DSP_SRCS-$(HAVE_SSE2)  += x86/vpx_subpixel_8t_sse2.asm
DSP_SRCS-$(HAVE_SSE2)  += x86/vpx_subpixel_4t_intrin_sse2.c
DSP_SRCS-$(HAVE_SSE2)  += x86/vpx_subpixel_bilinear_sse2.asm
DSP_SRCS-$(HAVE_SSSE3) += x86/vpx_subpixel_8t_ssse3.asm
DSP_SRCS-$(HAVE_SSSE3) += x86/vpx_subpixel_bilinear_ssse3.asm
DSP_SRCS-$(HAVE_AVX2)  += x86/vpx_subpixel_8t_intrin_avx2.c
DSP_SRCS-$(HAVE_SSSE3) += x86/vpx_subpixel_8t_intrin_ssse3.c
ifeq ($(CONFIG_VP9_HIGHBITDEPTH),yes)
DSP_SRCS-$(HAVE_SSE2)  += x86/vpx_high_subpixel_8t_sse2.asm
DSP_SRCS-$(HAVE_SSE2)  += x86/vpx_high_subpixel_bilinear_sse2.asm
DSP_SRCS-$(HAVE_AVX2)  += x86/highbd_convolve_avx2.c
DSP_SRCS-$(HAVE_NEON)  += arm/highbd_vpx_convolve_copy_neon.c
DSP_SRCS-$(HAVE_NEON)  += arm/highbd_vpx_convolve_avg_neon.c
DSP_SRCS-$(HAVE_NEON)  += arm/highbd_vpx_convolve8_neon.c
DSP_SRCS-$(HAVE_SVE)   += arm/highbd_vpx_convolve8_sve.c
DSP_SRCS-$(HAVE_SVE2)  += arm/highbd_vpx_convolve8_sve2.c
endif

DSP_SRCS-$(HAVE_SSE2)  += x86/vpx_convolve_copy_sse2.asm
DSP_SRCS-$(HAVE_NEON)  += arm/vpx_scaled_convolve8_neon.c

ifeq ($(HAVE_NEON_ASM),yes)
DSP_SRCS-yes += arm/vpx_convolve_copy_neon_asm$(ASM)
DSP_SRCS-yes += arm/vpx_convolve8_horiz_filter_type2_neon$(ASM)
DSP_SRCS-yes += arm/vpx_convolve8_vert_filter_type2_neon$(ASM)
DSP_SRCS-yes += arm/vpx_convolve8_horiz_filter_type1_neon$(ASM)
DSP_SRCS-yes += arm/vpx_convolve8_vert_filter_type1_neon$(ASM)
DSP_SRCS-yes += arm/vpx_convolve8_avg_horiz_filter_type2_neon$(ASM)
DSP_SRCS-yes += arm/vpx_convolve8_avg_vert_filter_type2_neon$(ASM)
DSP_SRCS-yes += arm/vpx_convolve8_avg_horiz_filter_type1_neon$(ASM)
DSP_SRCS-yes += arm/vpx_convolve8_avg_vert_filter_type1_neon$(ASM)
DSP_SRCS-yes += arm/vpx_convolve_avg_neon_asm$(ASM)
DSP_SRCS-yes += arm/vpx_convolve8_neon_asm.c
DSP_SRCS-yes += arm/vpx_convolve8_neon_asm.h
DSP_SRCS-yes += arm/vpx_convolve_neon.c
else
ifeq ($(HAVE_NEON),yes)
DSP_SRCS-yes += arm/vpx_convolve_copy_neon.c
DSP_SRCS-yes += arm/vpx_convolve8_neon.c
DSP_SRCS-yes += arm/vpx_convolve_avg_neon.c
DSP_SRCS-yes += arm/vpx_convolve_neon.c
DSP_SRCS-$(HAVE_NEON_DOTPROD) += arm/vpx_convolve8_neon_dotprod.c
DSP_SRCS-$(HAVE_NEON_I8MM) += arm/vpx_convolve8_neon_i8mm.c
endif  # HAVE_NEON
endif  # HAVE_NEON_ASM

# common (msa)
DSP_SRCS-$(HAVE_MSA) += mips/vpx_convolve8_avg_horiz_msa.c
DSP_SRCS-$(HAVE_MSA) += mips/vpx_convolve8_avg_msa.c
DSP_SRCS-$(HAVE_MSA) += mips/vpx_convolve8_avg_vert_msa.c
DSP_SRCS-$(HAVE_MSA) += mips/vpx_convolve8_horiz_msa.c
DSP_SRCS-$(HAVE_MSA) += mips/vpx_convolve8_msa.c
DSP_SRCS-$(HAVE_MSA) += mips/vpx_convolve8_vert_msa.c
DSP_SRCS-$(HAVE_MSA) += mips/vpx_convolve_avg_msa.c
DSP_SRCS-$(HAVE_MSA) += mips/vpx_convolve_copy_msa.c
DSP_SRCS-$(HAVE_MSA) += mips/vpx_convolve_msa.h
DSP_SRCS-$(HAVE_MMI) += mips/vpx_convolve8_mmi.c

# common (dspr2)
DSP_SRCS-$(HAVE_DSPR2)  += mips/convolve_common_dspr2.h
DSP_SRCS-$(HAVE_DSPR2)  += mips/convolve2_avg_dspr2.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/convolve2_avg_horiz_dspr2.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/convolve2_dspr2.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/convolve2_horiz_dspr2.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/convolve2_vert_dspr2.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/convolve8_avg_dspr2.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/convolve8_avg_horiz_dspr2.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/convolve8_dspr2.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/convolve8_horiz_dspr2.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/convolve8_vert_dspr2.c

DSP_SRCS-$(HAVE_VSX)  += ppc/vpx_convolve_vsx.c

# common (lsx)
DSP_SRCS-$(HAVE_LSX) += loongarch/vpx_convolve8_avg_horiz_lsx.c
DSP_SRCS-$(HAVE_LSX) += loongarch/vpx_convolve8_avg_vert_lsx.c
DSP_SRCS-$(HAVE_LSX) += loongarch/vpx_convolve8_horiz_lsx.c
DSP_SRCS-$(HAVE_LSX) += loongarch/vpx_convolve8_vert_lsx.c
DSP_SRCS-$(HAVE_LSX) += loongarch/vpx_convolve8_lsx.c
DSP_SRCS-$(HAVE_LSX) += loongarch/vpx_convolve8_avg_lsx.c
DSP_SRCS-$(HAVE_LSX) += loongarch/vpx_convolve_avg_lsx.c
DSP_SRCS-$(HAVE_LSX) += loongarch/vpx_convolve_copy_lsx.c
DSP_SRCS-$(HAVE_LSX) += loongarch/vpx_convolve_lsx.h

# loop filters
DSP_SRCS-yes += loopfilter.c

DSP_SRCS-$(HAVE_SSE2)  += x86/loopfilter_sse2.c
DSP_SRCS-$(HAVE_AVX2)  += x86/loopfilter_avx2.c

ifeq ($(HAVE_NEON_ASM),yes)
DSP_SRCS-yes  += arm/loopfilter_16_neon$(ASM)
DSP_SRCS-yes  += arm/loopfilter_8_neon$(ASM)
DSP_SRCS-yes  += arm/loopfilter_4_neon$(ASM)
else
DSP_SRCS-$(HAVE_NEON)   += arm/loopfilter_neon.c
endif  # HAVE_NEON_ASM

DSP_SRCS-$(HAVE_MSA)    += mips/loopfilter_msa.h
DSP_SRCS-$(HAVE_MSA)    += mips/loopfilter_16_msa.c
DSP_SRCS-$(HAVE_MSA)    += mips/loopfilter_8_msa.c
DSP_SRCS-$(HAVE_MSA)    += mips/loopfilter_4_msa.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/loopfilter_filters_dspr2.h
DSP_SRCS-$(HAVE_DSPR2)  += mips/loopfilter_filters_dspr2.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/loopfilter_macros_dspr2.h
DSP_SRCS-$(HAVE_DSPR2)  += mips/loopfilter_masks_dspr2.h
DSP_SRCS-$(HAVE_DSPR2)  += mips/loopfilter_mb_dspr2.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/loopfilter_mb_horiz_dspr2.c
DSP_SRCS-$(HAVE_DSPR2)  += mips/loopfilter_mb_vert_dspr2.c

DSP_SRCS-$(HAVE_LSX)    += loongarch/loopfilter_lsx.h
DSP_SRCS-$(HAVE_LSX)    += loongarch/loopfilter_16_lsx.c
DSP_SRCS-$(HAVE_LSX)    += loongarch/loopfilter_8_lsx.c
DSP_SRCS-$(HAVE_LSX)    += loongarch/loopfilter_4_lsx.c

ifeq ($(CONFIG_VP9_HIGHBITDEPTH),yes)
DSP_SRCS-$(HAVE_NEON)   += arm/highbd_loopfilter_neon.c
DSP_SRCS-$(HAVE_SSE2)   += x86/highbd_loopfilter_sse2.c
endif  # CONFIG_VP9_HIGHBITDEPTH
endif # CONFIG_VP9

DSP_SRCS-yes            += txfm_common.h
DSP_SRCS-$(HAVE_SSE2)   += x86/txfm_common_sse2.h
DSP_SRCS-$(HAVE_MSA)    += mips/txfm_macros_msa.h
DSP_SRCS-$(HAVE_LSX)    += loongarch/txfm_macros_lsx.h
# forward transform
ifeq ($(CONFIG_VP9_ENCODER),yes)
DSP_SRCS-yes            += fwd_txfm.c
DSP_SRCS-yes            += fwd_txfm.h
DSP_SRCS-$(HAVE_SSE2)   += x86/fwd_txfm_sse2.h
DSP_SRCS-$(HAVE_SSE2)   += x86/fwd_txfm_sse2.c
DSP_SRCS-$(HAVE_SSE2)   += x86/fwd_txfm_impl_sse2.h
DSP_SRCS-$(HAVE_SSE2)   += x86/fwd_dct32x32_impl_sse2.h
ifeq ($(VPX_ARCH_X86_64),yes)
DSP_SRCS-$(HAVE_SSSE3)  += x86/fwd_txfm_ssse3_x86_64.asm
endif
DSP_SRCS-$(HAVE_AVX2)   += x86/fwd_dct32x32_impl_avx2.h
DSP_SRCS-$(HAVE_NEON)   += arm/fdct4x4_neon.c
DSP_SRCS-$(HAVE_NEON)   += arm/fdct8x8_neon.c
DSP_SRCS-$(HAVE_NEON)   += arm/fdct16x16_neon.c
DSP_SRCS-$(HAVE_NEON)   += arm/fdct32x32_neon.c
DSP_SRCS-$(HAVE_NEON)   += arm/fdct_partial_neon.c
DSP_SRCS-$(HAVE_MSA)    += mips/fwd_txfm_msa.h
DSP_SRCS-$(HAVE_MSA)    += mips/fwd_txfm_msa.c
DSP_SRCS-$(HAVE_LSX)    += loongarch/fwd_txfm_lsx.h
DSP_SRCS-$(HAVE_LSX)    += loongarch/fwd_txfm_lsx.c

ifneq ($(CONFIG_VP9_HIGHBITDEPTH),yes)
DSP_SRCS-$(HAVE_AVX2)   += x86/fwd_txfm_avx2.c
DSP_SRCS-$(HAVE_MSA)    += mips/fwd_dct32x32_msa.c
DSP_SRCS-$(HAVE_LSX)    += loongarch/fwd_dct32x32_lsx.c
endif  # !CONFIG_VP9_HIGHBITDEPTH

DSP_SRCS-$(HAVE_VSX)    += ppc/fdct32x32_vsx.c
endif  # CONFIG_VP9_ENCODER

# inverse transform
ifeq ($(CONFIG_VP9),yes)
DSP_SRCS-yes            += inv_txfm.h
DSP_SRCS-yes            += inv_txfm.c
DSP_SRCS-$(HAVE_SSE2)   += x86/inv_txfm_sse2.h
DSP_SRCS-$(HAVE_SSE2)   += x86/inv_txfm_sse2.c
DSP_SRCS-$(HAVE_AVX2)   += x86/inv_txfm_avx2.c
DSP_SRCS-$(HAVE_SSE2)   += x86/inv_wht_sse2.asm
DSP_SRCS-$(HAVE_SSSE3)  += x86/inv_txfm_ssse3.h
DSP_SRCS-$(HAVE_SSSE3)  += x86/inv_txfm_ssse3.c

DSP_SRCS-$(HAVE_NEON_ASM) += arm/save_reg_neon$(ASM)

DSP_SRCS-$(HAVE_VSX) += ppc/inv_txfm_vsx.c

ifneq ($(CONFIG_VP9_HIGHBITDEPTH),yes)
DSP_SRCS-$(HAVE_MSA)   += mips/inv_txfm_msa.h
DSP_SRCS-$(HAVE_MSA)   += mips/idct4x4_msa.c
DSP_SRCS-$(HAVE_MSA)   += mips/idct8x8_msa.c
DSP_SRCS-$(HAVE_MSA)   += mips/idct16x16_msa.c
DSP_SRCS-$(HAVE_MSA)   += mips/idct32x32_msa.c

DSP_SRCS-$(HAVE_DSPR2) += mips/inv_txfm_dspr2.h
DSP_SRCS-$(HAVE_DSPR2) += mips/itrans4_dspr2.c
DSP_SRCS-$(HAVE_DSPR2) += mips/itrans8_dspr2.c
DSP_SRCS-$(HAVE_DSPR2) += mips/itrans16_dspr2.c
DSP_SRCS-$(HAVE_DSPR2) += mips/itrans32_dspr2.c
DSP_SRCS-$(HAVE_DSPR2) += mips/itrans32_cols_dspr2.c

DSP_SRCS-$(HAVE_LSX)   += loongarch/idct32x32_lsx.c
else  # CONFIG_VP9_HIGHBITDEPTH
DSP_SRCS-$(HAVE_NEON)  += arm/highbd_idct4x4_add_neon.c
DSP_SRCS-$(HAVE_NEON)  += arm/highbd_idct8x8_add_neon.c
DSP_SRCS-$(HAVE_NEON)  += arm/highbd_idct16x16_add_neon.c
DSP_SRCS-$(HAVE_NEON)  += arm/highbd_idct32x32_add_neon.c
DSP_SRCS-$(HAVE_NEON)  += arm/highbd_idct32x32_34_add_neon.c
DSP_SRCS-$(HAVE_NEON)  += arm/highbd_idct32x32_135_add_neon.c
DSP_SRCS-$(HAVE_NEON)  += arm/highbd_idct32x32_1024_add_neon.c
DSP_SRCS-$(HAVE_NEON)  += arm/highbd_idct_neon.h
DSP_SRCS-$(HAVE_SSE2)  += x86/highbd_inv_txfm_sse2.h
DSP_SRCS-$(HAVE_SSE2)  += x86/highbd_idct4x4_add_sse2.c
DSP_SRCS-$(HAVE_SSE2)  += x86/highbd_idct8x8_add_sse2.c
DSP_SRCS-$(HAVE_SSE2)  += x86/highbd_idct16x16_add_sse2.c
DSP_SRCS-$(HAVE_SSE2)  += x86/highbd_idct32x32_add_sse2.c
DSP_SRCS-$(HAVE_SSE4_1) += x86/highbd_inv_txfm_sse4.h
DSP_SRCS-$(HAVE_SSE4_1) += x86/highbd_idct4x4_add_sse4.c
DSP_SRCS-$(HAVE_SSE4_1) += x86/highbd_idct8x8_add_sse4.c
DSP_SRCS-$(HAVE_SSE4_1) += x86/highbd_idct16x16_add_sse4.c
DSP_SRCS-$(HAVE_SSE4_1) += x86/highbd_idct32x32_add_sse4.c
endif  # !CONFIG_VP9_HIGHBITDEPTH

ifeq ($(HAVE_NEON_ASM),yes)
DSP_SRCS-yes += arm/idct_neon$(ASM)
DSP_SRCS-yes += arm/idct4x4_1_add_neon$(ASM)
DSP_SRCS-yes += arm/idct4x4_add_neon$(ASM)
else
DSP_SRCS-$(HAVE_NEON) += arm/idct4x4_1_add_neon.c
DSP_SRCS-$(HAVE_NEON) += arm/idct4x4_add_neon.c
endif  # HAVE_NEON_ASM
DSP_SRCS-$(HAVE_NEON) += arm/idct_neon.h
DSP_SRCS-$(HAVE_NEON) += arm/idct8x8_1_add_neon.c
DSP_SRCS-$(HAVE_NEON) += arm/idct8x8_add_neon.c
DSP_SRCS-$(HAVE_NEON) += arm/idct16x16_1_add_neon.c
DSP_SRCS-$(HAVE_NEON) += arm/idct16x16_add_neon.c
DSP_SRCS-$(HAVE_NEON) += arm/idct32x32_1_add_neon.c
DSP_SRCS-$(HAVE_NEON) += arm/idct32x32_34_add_neon.c
DSP_SRCS-$(HAVE_NEON) += arm/idct32x32_135_add_neon.c
DSP_SRCS-$(HAVE_NEON) += arm/idct32x32_add_neon.c

endif  # CONFIG_VP9

# quantization
ifeq ($(CONFIG_VP9_ENCODER),yes)
DSP_SRCS-yes            += quantize.c
DSP_SRCS-yes            += quantize.h

DSP_SRCS-$(HAVE_SSE2)   += x86/quantize_sse2.c
DSP_SRCS-$(HAVE_SSE2)   += x86/quantize_sse2.h
DSP_SRCS-$(HAVE_SSSE3)  += x86/quantize_ssse3.c
DSP_SRCS-$(HAVE_SSSE3)  += x86/quantize_ssse3.h
DSP_SRCS-$(HAVE_AVX)    += x86/quantize_avx.c
DSP_SRCS-$(HAVE_AVX2)   += x86/quantize_avx2.c
DSP_SRCS-$(HAVE_NEON)   += arm/quantize_neon.c
DSP_SRCS-$(HAVE_VSX)    += ppc/quantize_vsx.c
DSP_SRCS-$(HAVE_LSX)    += loongarch/quantize_lsx.c
ifeq ($(CONFIG_VP9_HIGHBITDEPTH),yes)
DSP_SRCS-$(HAVE_SSE2)   += x86/highbd_quantize_intrin_sse2.c
DSP_SRCS-$(HAVE_AVX2)   += x86/highbd_quantize_intrin_avx2.c
DSP_SRCS-$(HAVE_NEON)   += arm/highbd_quantize_neon.c
endif

# avg
DSP_SRCS-yes           += avg.c
DSP_SRCS-$(HAVE_SSE2)  += x86/avg_intrin_sse2.c
DSP_SRCS-$(HAVE_AVX2)  += x86/avg_intrin_avx2.c
DSP_SRCS-$(HAVE_NEON)  += arm/avg_neon.c
DSP_SRCS-$(HAVE_NEON)  += arm/hadamard_neon.c
ifeq ($(CONFIG_VP9_HIGHBITDEPTH),yes)
DSP_SRCS-$(HAVE_NEON)  += arm/highbd_hadamard_neon.c
DSP_SRCS-$(HAVE_NEON)  += arm/highbd_avg_neon.c
endif
DSP_SRCS-$(HAVE_MSA)   += mips/avg_msa.c
DSP_SRCS-$(HAVE_LSX)   += loongarch/avg_lsx.c
ifeq ($(VPX_ARCH_X86_64),yes)
DSP_SRCS-$(HAVE_SSSE3) += x86/avg_ssse3_x86_64.asm
endif
DSP_SRCS-$(HAVE_VSX)   += ppc/hadamard_vsx.c

endif  # CONFIG_VP9_ENCODER

# skin detection
DSP_SRCS-yes            += skin_detection.h
DSP_SRCS-yes            += skin_detection.c

ifeq ($(CONFIG_ENCODERS),yes)
DSP_SRCS-yes            += sad.c
DSP_SRCS-yes            += subtract.c
DSP_SRCS-yes            += sum_squares.c
DSP_SRCS-$(HAVE_NEON)   += arm/sum_squares_neon.c
DSP_SRCS-$(HAVE_SVE)    += arm/sum_squares_sve.c
DSP_SRCS-$(HAVE_SSE2)   += x86/sum_squares_sse2.c
DSP_SRCS-$(HAVE_MSA)    += mips/sum_squares_msa.c

DSP_SRCS-$(HAVE_NEON)   += arm/sad4d_neon.c
DSP_SRCS-$(HAVE_NEON_DOTPROD) += arm/sad4d_neon_dotprod.c
DSP_SRCS-$(HAVE_NEON)   += arm/sad_neon.c
DSP_SRCS-$(HAVE_NEON_DOTPROD) += arm/sad_neon_dotprod.c
DSP_SRCS-$(HAVE_NEON)   += arm/subtract_neon.c

DSP_SRCS-$(HAVE_MSA)    += mips/sad_msa.c
DSP_SRCS-$(HAVE_MSA)    += mips/subtract_msa.c

DSP_SRCS-$(HAVE_LSX)    += loongarch/sad_lsx.c

DSP_SRCS-$(HAVE_MMI)    += mips/sad_mmi.c
DSP_SRCS-$(HAVE_MMI)    += mips/subtract_mmi.c

DSP_SRCS-$(HAVE_AVX2)   += x86/sad4d_avx2.c
DSP_SRCS-$(HAVE_AVX2)   += x86/sad_avx2.c
DSP_SRCS-$(HAVE_AVX2)   += x86/subtract_avx2.c
DSP_SRCS-$(HAVE_AVX512) += x86/sad4d_avx512.c
DSP_SRCS-$(HAVE_AVX512) += x86/sad_avx512.c

DSP_SRCS-$(HAVE_SSE2)   += x86/sad4d_sse2.asm
DSP_SRCS-$(HAVE_SSE2)   += x86/sad_sse2.asm
DSP_SRCS-$(HAVE_SSE2)   += x86/subtract_sse2.asm

DSP_SRCS-$(HAVE_VSX) += ppc/sad_vsx.c
DSP_SRCS-$(HAVE_VSX) += ppc/subtract_vsx.c

DSP_SRCS-$(HAVE_LSX)    += loongarch/subtract_lsx.c

ifeq ($(CONFIG_VP9_HIGHBITDEPTH),yes)
DSP_SRCS-$(HAVE_SSE2) += x86/highbd_sad4d_sse2.asm
DSP_SRCS-$(HAVE_SSE2) += x86/highbd_sad_sse2.asm
DSP_SRCS-$(HAVE_NEON) += arm/highbd_sad4d_neon.c
DSP_SRCS-$(HAVE_NEON) += arm/highbd_sad_neon.c
DSP_SRCS-$(HAVE_AVX2) += x86/highbd_sad4d_avx2.c
DSP_SRCS-$(HAVE_AVX2) += x86/highbd_sad_avx2.c
endif  # CONFIG_VP9_HIGHBITDEPTH

endif  # CONFIG_ENCODERS

ifneq ($(filter yes,$(CONFIG_ENCODERS) $(CONFIG_POSTPROC) $(CONFIG_VP9_POSTPROC)),)
DSP_SRCS-yes            += variance.c
DSP_SRCS-yes            += variance.h

DSP_SRCS-$(HAVE_NEON)   += arm/avg_pred_neon.c
DSP_SRCS-$(HAVE_NEON)   += arm/subpel_variance_neon.c
DSP_SRCS-$(HAVE_NEON)   += arm/variance_neon.c
DSP_SRCS-$(HAVE_NEON_DOTPROD)   += arm/variance_neon_dotprod.c

DSP_SRCS-$(HAVE_MSA)    += mips/variance_msa.c
DSP_SRCS-$(HAVE_MSA)    += mips/sub_pixel_variance_msa.c

DSP_SRCS-$(HAVE_LSX)    += loongarch/variance_lsx.h
DSP_SRCS-$(HAVE_LSX)    += loongarch/variance_lsx.c
DSP_SRCS-$(HAVE_LSX)    += loongarch/sub_pixel_variance_lsx.c
DSP_SRCS-$(HAVE_LSX)    += loongarch/avg_pred_lsx.c

DSP_SRCS-$(HAVE_MMI)    += mips/variance_mmi.c

DSP_SRCS-$(HAVE_SSE2)   += x86/avg_pred_sse2.c
DSP_SRCS-$(HAVE_AVX2)   += x86/avg_pred_avx2.c
DSP_SRCS-$(HAVE_SSE2)   += x86/variance_sse2.c  # Contains SSE2 and SSSE3
DSP_SRCS-$(HAVE_AVX2)   += x86/variance_avx2.c
DSP_SRCS-$(HAVE_VSX)    += ppc/variance_vsx.c

ifeq ($(VPX_ARCH_X86_64),yes)
DSP_SRCS-$(HAVE_SSE2)   += x86/ssim_opt_x86_64.asm
endif  # VPX_ARCH_X86_64

DSP_SRCS-$(HAVE_SSE2)   += x86/subpel_variance_sse2.asm  # Contains SSE2 and SSSE3

ifeq ($(CONFIG_VP9_HIGHBITDEPTH),yes)
DSP_SRCS-$(HAVE_SSE2)   += x86/highbd_variance_sse2.c
DSP_SRCS-$(HAVE_SSE2)   += x86/highbd_variance_impl_sse2.asm
DSP_SRCS-$(HAVE_SSE2)   += x86/highbd_subpel_variance_impl_sse2.asm
DSP_SRCS-$(HAVE_NEON)   += arm/highbd_avg_pred_neon.c
DSP_SRCS-$(HAVE_NEON)   += arm/highbd_sse_neon.c
DSP_SRCS-$(HAVE_NEON)   += arm/highbd_variance_neon.c
DSP_SRCS-$(HAVE_NEON_DOTPROD)   += arm/highbd_variance_neon_dotprod.c
DSP_SRCS-$(HAVE_SVE)    += arm/highbd_variance_sve.c
DSP_SRCS-$(HAVE_NEON)   += arm/highbd_subpel_variance_neon.c
endif  # CONFIG_VP9_HIGHBITDEPTH
endif  # CONFIG_ENCODERS || CONFIG_POSTPROC || CONFIG_VP9_POSTPROC

# Neon utilities
DSP_SRCS-$(HAVE_NEON) += arm/mem_neon.h
DSP_SRCS-$(HAVE_NEON) += arm/sum_neon.h
DSP_SRCS-$(HAVE_NEON) += arm/transpose_neon.h
DSP_SRCS-$(HAVE_NEON) += arm/vpx_convolve8_neon.h

# PPC VSX utilities
DSP_SRCS-$(HAVE_VSX)  += ppc/types_vsx.h
DSP_SRCS-$(HAVE_VSX)  += ppc/txfm_common_vsx.h
DSP_SRCS-$(HAVE_VSX)  += ppc/transpose_vsx.h
DSP_SRCS-$(HAVE_VSX)  += ppc/bitdepth_conversion_vsx.h

# X86 utilities
DSP_SRCS-$(HAVE_SSE2) += x86/mem_sse2.h
DSP_SRCS-$(HAVE_SSE2) += x86/transpose_sse2.h

# LSX utilities
DSP_SRCS-$(HAVE_LSX)  += loongarch/bitdepth_conversion_lsx.h

DSP_SRCS-no += $(DSP_SRCS_REMOVE-yes)

DSP_SRCS-yes += vpx_dsp_rtcd.c
DSP_SRCS-yes += vpx_dsp_rtcd_defs.pl

$(eval $(call rtcd_h_template,vpx_dsp_rtcd,vpx_dsp/vpx_dsp_rtcd_defs.pl))
