##
##  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

VP9_CX_EXPORTS += exports_enc

VP9_CX_SRCS-yes += $(VP9_COMMON_SRCS-yes)
VP9_CX_SRCS-no  += $(VP9_COMMON_SRCS-no)
VP9_CX_SRCS_REMOVE-yes += $(VP9_COMMON_SRCS_REMOVE-yes)
VP9_CX_SRCS_REMOVE-no  += $(VP9_COMMON_SRCS_REMOVE-no)

VP9_CX_SRCS-yes += vp9_cx_iface.c
VP9_CX_SRCS-yes += vp9_cx_iface.h

VP9_CX_SRCS-yes += encoder/vp9_bitstream.c
VP9_CX_SRCS-yes += encoder/vp9_context_tree.c
VP9_CX_SRCS-yes += encoder/vp9_context_tree.h
VP9_CX_SRCS-yes += encoder/vp9_cost.h
VP9_CX_SRCS-yes += encoder/vp9_cost.c
VP9_CX_SRCS-yes += encoder/vp9_dct.c
VP9_CX_SRCS-$(CONFIG_VP9_TEMPORAL_DENOISING) += encoder/vp9_denoiser.c
VP9_CX_SRCS-$(CONFIG_VP9_TEMPORAL_DENOISING) += encoder/vp9_denoiser.h
VP9_CX_SRCS-yes += encoder/vp9_encodeframe.c
VP9_CX_SRCS-yes += encoder/vp9_encodeframe.h
VP9_CX_SRCS-yes += encoder/vp9_encodemb.c
VP9_CX_SRCS-yes += encoder/vp9_encodemv.c
VP9_CX_SRCS-yes += encoder/vp9_ethread.h
VP9_CX_SRCS-yes += encoder/vp9_ethread.c
VP9_CX_SRCS-yes += encoder/vp9_extend.c
VP9_CX_SRCS-yes += encoder/vp9_firstpass.c
VP9_CX_SRCS-yes += encoder/vp9_block.h
VP9_CX_SRCS-yes += encoder/vp9_bitstream.h
VP9_CX_SRCS-yes += encoder/vp9_encodemb.h
VP9_CX_SRCS-yes += encoder/vp9_encodemv.h
VP9_CX_SRCS-yes += encoder/vp9_extend.h
VP9_CX_SRCS-yes += encoder/vp9_firstpass.h
VP9_CX_SRCS-yes += encoder/vp9_firstpass_stats.h
VP9_CX_SRCS-yes += encoder/vp9_frame_scale.c
VP9_CX_SRCS-yes += encoder/vp9_job_queue.h
VP9_CX_SRCS-yes += encoder/vp9_lookahead.c
VP9_CX_SRCS-yes += encoder/vp9_lookahead.h
VP9_CX_SRCS-yes += encoder/vp9_mcomp.h
VP9_CX_SRCS-yes += encoder/vp9_multi_thread.c
VP9_CX_SRCS-yes += encoder/vp9_multi_thread.h
VP9_CX_SRCS-yes += encoder/vp9_encoder.h
VP9_CX_SRCS-yes += encoder/vp9_quantize.h
VP9_CX_SRCS-yes += encoder/vp9_ratectrl.h
VP9_CX_SRCS-yes += encoder/vp9_rd.h
VP9_CX_SRCS-yes += encoder/vp9_rdopt.h
VP9_CX_SRCS-yes += encoder/vp9_pickmode.h
VP9_CX_SRCS-yes += encoder/vp9_svc_layercontext.h
VP9_CX_SRCS-yes += encoder/vp9_tokenize.h
VP9_CX_SRCS-yes += encoder/vp9_treewriter.h
VP9_CX_SRCS-yes += encoder/vp9_mcomp.c
VP9_CX_SRCS-yes += encoder/vp9_encoder.c
VP9_CX_SRCS-yes += encoder/vp9_picklpf.c
VP9_CX_SRCS-yes += encoder/vp9_picklpf.h
VP9_CX_SRCS-yes += encoder/vp9_quantize.c
VP9_CX_SRCS-yes += encoder/vp9_ratectrl.c
VP9_CX_SRCS-yes += encoder/vp9_rd.c
VP9_CX_SRCS-yes += encoder/vp9_rdopt.c
VP9_CX_SRCS-yes += encoder/vp9_pickmode.c
VP9_CX_SRCS-yes += encoder/vp9_partition_models.h
VP9_CX_SRCS-yes += encoder/vp9_segmentation.c
VP9_CX_SRCS-yes += encoder/vp9_segmentation.h
VP9_CX_SRCS-yes += encoder/vp9_speed_features.c
VP9_CX_SRCS-yes += encoder/vp9_speed_features.h
VP9_CX_SRCS-yes += encoder/vp9_subexp.c
VP9_CX_SRCS-yes += encoder/vp9_subexp.h
VP9_CX_SRCS-yes += encoder/vp9_svc_layercontext.c
VP9_CX_SRCS-yes += encoder/vp9_resize.c
VP9_CX_SRCS-yes += encoder/vp9_resize.h
VP9_CX_SRCS-$(CONFIG_INTERNAL_STATS) += encoder/vp9_blockiness.c
VP9_CX_SRCS-$(CONFIG_INTERNAL_STATS) += encoder/vp9_blockiness.h
VP9_CX_SRCS-$(CONFIG_NON_GREEDY_MV) += encoder/vp9_non_greedy_mv.c
VP9_CX_SRCS-$(CONFIG_NON_GREEDY_MV) += encoder/vp9_non_greedy_mv.h

VP9_CX_SRCS-yes += encoder/vp9_tokenize.c
VP9_CX_SRCS-yes += encoder/vp9_treewriter.c
VP9_CX_SRCS-yes += encoder/vp9_aq_variance.c
VP9_CX_SRCS-yes += encoder/vp9_aq_variance.h
VP9_CX_SRCS-yes += encoder/vp9_aq_360.c
VP9_CX_SRCS-yes += encoder/vp9_aq_360.h
VP9_CX_SRCS-yes += encoder/vp9_aq_cyclicrefresh.c
VP9_CX_SRCS-yes += encoder/vp9_aq_cyclicrefresh.h
VP9_CX_SRCS-yes += encoder/vp9_aq_complexity.c
VP9_CX_SRCS-yes += encoder/vp9_aq_complexity.h
VP9_CX_SRCS-yes += encoder/vp9_alt_ref_aq.h
VP9_CX_SRCS-yes += encoder/vp9_alt_ref_aq.c
VP9_CX_SRCS-yes += encoder/vp9_skin_detection.c
VP9_CX_SRCS-yes += encoder/vp9_skin_detection.h
VP9_CX_SRCS-yes += encoder/vp9_noise_estimate.c
VP9_CX_SRCS-yes += encoder/vp9_noise_estimate.h
VP9_CX_SRCS-yes += encoder/vp9_ext_ratectrl.c
VP9_CX_SRCS-yes += encoder/vp9_ext_ratectrl.h
ifeq ($(CONFIG_VP9_POSTPROC),yes)
VP9_CX_SRCS-$(CONFIG_INTERNAL_STATS) += common/vp9_postproc.h
VP9_CX_SRCS-$(CONFIG_INTERNAL_STATS) += common/vp9_postproc.c
endif
VP9_CX_SRCS-yes += encoder/vp9_temporal_filter.c
VP9_CX_SRCS-yes += encoder/vp9_temporal_filter.h
VP9_CX_SRCS-yes += encoder/vp9_tpl_model.c
VP9_CX_SRCS-yes += encoder/vp9_tpl_model.h
VP9_CX_SRCS-yes += encoder/vp9_mbgraph.c
VP9_CX_SRCS-yes += encoder/vp9_mbgraph.h

VP9_CX_SRCS-$(HAVE_SSSE3) += encoder/x86/temporal_filter_ssse3.c
VP9_CX_SRCS-$(HAVE_SSE4_1) += encoder/x86/temporal_filter_sse4.c
VP9_CX_SRCS-$(HAVE_AVX2) += encoder/x86/temporal_filter_avx2.c
VP9_CX_SRCS-$(HAVE_SSE4_1) += encoder/vp9_temporal_filter_constants.h
VP9_CX_SRCS-$(HAVE_NEON) += encoder/arm/neon/vp9_temporal_filter_neon.c
VP9_CX_SRCS-$(HAVE_NEON) += encoder/vp9_temporal_filter_constants.h
VP9_CX_SRCS-$(HAVE_NEON_DOTPROD) += encoder/arm/neon/vp9_temporal_filter_neon_dotprod.c
VP9_CX_SRCS-$(HAVE_NEON_I8MM) += encoder/arm/neon/vp9_temporal_filter_neon_i8mm.c

VP9_CX_SRCS-$(HAVE_SSE2) += encoder/x86/vp9_quantize_sse2.c
VP9_CX_SRCS-$(HAVE_SSSE3) += encoder/x86/vp9_quantize_ssse3.c
VP9_CX_SRCS-$(HAVE_AVX2) += encoder/x86/vp9_quantize_avx2.c
VP9_CX_SRCS-$(HAVE_NEON) += encoder/arm/neon/vp9_diamond_search_sad_neon.c
ifeq ($(CONFIG_VP9_HIGHBITDEPTH),yes)
VP9_CX_SRCS-$(HAVE_SSE2) += encoder/x86/vp9_highbd_block_error_intrin_sse2.c
VP9_CX_SRCS-$(HAVE_SSSE3) += encoder/x86/highbd_temporal_filter_ssse3.c
VP9_CX_SRCS-$(HAVE_SSE4_1) += encoder/x86/highbd_temporal_filter_sse4.c
VP9_CX_SRCS-$(HAVE_AVX2) += encoder/x86/highbd_temporal_filter_avx2.c
VP9_CX_SRCS-$(HAVE_NEON) += encoder/arm/neon/vp9_highbd_temporal_filter_neon.c
VP9_CX_SRCS-$(HAVE_SVE2) += encoder/arm/neon/vp9_highbd_temporal_filter_sve2.c
endif

VP9_CX_SRCS-$(HAVE_SSE2) += encoder/x86/vp9_dct_sse2.asm
VP9_CX_SRCS-$(HAVE_SSE2) += encoder/x86/vp9_error_sse2.asm

VP9_CX_SRCS-$(HAVE_SSE2) += encoder/x86/vp9_dct_intrin_sse2.c
VP9_CX_SRCS-$(HAVE_SSSE3) += encoder/x86/vp9_frame_scale_ssse3.c
VP9_CX_SRCS-$(HAVE_NEON) += encoder/arm/neon/vp9_dct_neon.c

ifeq ($(CONFIG_VP9_TEMPORAL_DENOISING),yes)
VP9_CX_SRCS-$(HAVE_SSE2) += encoder/x86/vp9_denoiser_sse2.c
VP9_CX_SRCS-$(HAVE_NEON) += encoder/arm/neon/vp9_denoiser_neon.c
endif

VP9_CX_SRCS-$(HAVE_AVX2) += encoder/x86/vp9_error_avx2.c

VP9_CX_SRCS-$(HAVE_NEON) += encoder/arm/neon/vp9_error_neon.c
VP9_CX_SRCS-$(HAVE_SVE)  += encoder/arm/neon/vp9_error_sve.c
VP9_CX_SRCS-$(HAVE_NEON) += encoder/arm/neon/vp9_frame_scale_neon.c
VP9_CX_SRCS-$(HAVE_NEON) += encoder/arm/neon/vp9_quantize_neon.c
ifeq ($(CONFIG_VP9_HIGHBITDEPTH),yes)
VP9_CX_SRCS-$(HAVE_NEON) += encoder/arm/neon/vp9_highbd_error_neon.c
endif

VP9_CX_SRCS-$(HAVE_MSA) += encoder/mips/msa/vp9_error_msa.c

ifneq ($(CONFIG_VP9_HIGHBITDEPTH),yes)
VP9_CX_SRCS-$(HAVE_MSA) += encoder/mips/msa/vp9_fdct4x4_msa.c
VP9_CX_SRCS-$(HAVE_MSA) += encoder/mips/msa/vp9_fdct8x8_msa.c
VP9_CX_SRCS-$(HAVE_MSA) += encoder/mips/msa/vp9_fdct16x16_msa.c
VP9_CX_SRCS-$(HAVE_MSA) += encoder/mips/msa/vp9_fdct_msa.h
endif  # !CONFIG_VP9_HIGHBITDEPTH

VP9_CX_SRCS-$(HAVE_VSX) += encoder/ppc/vp9_quantize_vsx.c

# Strip unnecessary files with CONFIG_REALTIME_ONLY
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/vp9_firstpass.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/vp9_mbgraph.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/vp9_temporal_filter.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/x86/temporal_filter_ssse3.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/x86/temporal_filter_sse4.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/x86/temporal_filter_avx2.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/vp9_temporal_filter_constants.h
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/x86/highbd_temporal_filter_ssse3.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/x86/highbd_temporal_filter_sse4.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/x86/highbd_temporal_filter_avx2.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/arm/neon/vp9_temporal_filter_neon.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/arm/neon/vp9_temporal_filter_neon_dotprod.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/arm/neon/vp9_temporal_filter_neon_i8mm.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/arm/neon/vp9_highbd_temporal_filter_neon.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/arm/neon/vp9_highbd_temporal_filter_sve2.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/vp9_alt_ref_aq.h
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/vp9_alt_ref_aq.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/vp9_aq_variance.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/vp9_aq_variance.h
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/vp9_aq_360.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/vp9_aq_360.h
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/vp9_aq_complexity.c
VP9_CX_SRCS_REMOVE-$(CONFIG_REALTIME_ONLY) += encoder/vp9_aq_complexity.h

VP9_CX_SRCS-yes := $(filter-out $(VP9_CX_SRCS_REMOVE-yes),$(VP9_CX_SRCS-yes))
