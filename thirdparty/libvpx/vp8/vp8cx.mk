##
##  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##


VP8_CX_EXPORTS += exports_enc

VP8_CX_SRCS-yes += $(VP8_COMMON_SRCS-yes)
VP8_CX_SRCS-no  += $(VP8_COMMON_SRCS-no)
VP8_CX_SRCS_REMOVE-yes += $(VP8_COMMON_SRCS_REMOVE-yes)
VP8_CX_SRCS_REMOVE-no  += $(VP8_COMMON_SRCS_REMOVE-no)

VP8_CX_SRCS-yes += vp8cx.mk

VP8_CX_SRCS-yes += vp8_cx_iface.c

VP8_CX_SRCS-yes += encoder/defaultcoefcounts.h
VP8_CX_SRCS-yes += encoder/bitstream.c
VP8_CX_SRCS-yes += encoder/boolhuff.c
VP8_CX_SRCS-yes += encoder/copy_c.c
VP8_CX_SRCS-yes += encoder/dct.c
VP8_CX_SRCS-yes += encoder/encodeframe.c
VP8_CX_SRCS-yes += encoder/encodeframe.h
VP8_CX_SRCS-yes += encoder/encodeintra.c
VP8_CX_SRCS-yes += encoder/encodemb.c
VP8_CX_SRCS-yes += encoder/encodemv.c
VP8_CX_SRCS-$(CONFIG_MULTITHREAD) += encoder/ethreading.c
VP8_CX_SRCS-$(CONFIG_MULTITHREAD) += encoder/ethreading.h
VP8_CX_SRCS-yes += encoder/firstpass.c
VP8_CX_SRCS-yes += encoder/block.h
VP8_CX_SRCS-yes += encoder/boolhuff.h
VP8_CX_SRCS-yes += encoder/bitstream.h
VP8_CX_SRCS-$(CONFIG_TEMPORAL_DENOISING) += encoder/denoising.h
VP8_CX_SRCS-$(CONFIG_TEMPORAL_DENOISING) += encoder/denoising.c
VP8_CX_SRCS-yes += encoder/encodeintra.h
VP8_CX_SRCS-yes += encoder/encodemb.h
VP8_CX_SRCS-yes += encoder/encodemv.h
VP8_CX_SRCS-yes += encoder/firstpass.h
VP8_CX_SRCS-yes += encoder/lookahead.c
VP8_CX_SRCS-yes += encoder/lookahead.h
VP8_CX_SRCS-yes += encoder/mcomp.h
VP8_CX_SRCS-yes += encoder/modecosts.h
VP8_CX_SRCS-yes += encoder/onyx_int.h
VP8_CX_SRCS-yes += encoder/pickinter.h
VP8_CX_SRCS-yes += encoder/quantize.h
VP8_CX_SRCS-yes += encoder/ratectrl.h
VP8_CX_SRCS-yes += encoder/rdopt.h
VP8_CX_SRCS-yes += encoder/tokenize.h
VP8_CX_SRCS-yes += encoder/treewriter.h
VP8_CX_SRCS-yes += encoder/mcomp.c
VP8_CX_SRCS-yes += encoder/modecosts.c
VP8_CX_SRCS-yes += encoder/onyx_if.c
VP8_CX_SRCS-yes += encoder/pickinter.c
VP8_CX_SRCS-yes += encoder/picklpf.c
VP8_CX_SRCS-yes += encoder/picklpf.h
VP8_CX_SRCS-yes += encoder/vp8_quantize.c
VP8_CX_SRCS-yes += encoder/ratectrl.c
VP8_CX_SRCS-yes += encoder/rdopt.c
VP8_CX_SRCS-yes += encoder/segmentation.c
VP8_CX_SRCS-yes += encoder/segmentation.h
VP8_CX_SRCS-yes += common/vp8_skin_detection.c
VP8_CX_SRCS-yes += common/vp8_skin_detection.h
VP8_CX_SRCS-yes += encoder/tokenize.c
VP8_CX_SRCS-yes += encoder/dct_value_cost.h
VP8_CX_SRCS-yes += encoder/dct_value_tokens.h
VP8_CX_SRCS-yes += encoder/treewriter.c
VP8_CX_SRCS-$(CONFIG_INTERNAL_STATS) += common/postproc.h
VP8_CX_SRCS-$(CONFIG_INTERNAL_STATS) += common/postproc.c
VP8_CX_SRCS-yes += encoder/temporal_filter.c
VP8_CX_SRCS-yes += encoder/temporal_filter.h
VP8_CX_SRCS-$(CONFIG_MULTI_RES_ENCODING) += encoder/mr_dissim.c
VP8_CX_SRCS-$(CONFIG_MULTI_RES_ENCODING) += encoder/mr_dissim.h

ifeq ($(CONFIG_REALTIME_ONLY),yes)
VP8_CX_SRCS_REMOVE-yes += encoder/firstpass.c
VP8_CX_SRCS_REMOVE-yes += encoder/temporal_filter.c
VP8_CX_SRCS_REMOVE-yes += encoder/temporal_filter.h
endif

VP8_CX_SRCS-$(HAVE_SSE2) += encoder/x86/copy_sse2.asm
VP8_CX_SRCS-$(HAVE_SSE2) += encoder/x86/copy_sse3.asm
VP8_CX_SRCS-$(HAVE_SSE2) += encoder/x86/dct_sse2.asm
VP8_CX_SRCS-$(HAVE_SSE2) += encoder/x86/fwalsh_sse2.asm
VP8_CX_SRCS-$(HAVE_SSE2) += encoder/x86/vp8_quantize_sse2.c
VP8_CX_SRCS-$(HAVE_SSSE3) += encoder/x86/vp8_quantize_ssse3.c
VP8_CX_SRCS-$(HAVE_SSE4_1) += encoder/x86/quantize_sse4.c

ifeq ($(CONFIG_TEMPORAL_DENOISING),yes)
VP8_CX_SRCS-$(HAVE_SSE2) += encoder/x86/denoising_sse2.c
endif

VP8_CX_SRCS-$(HAVE_SSE2) += encoder/x86/block_error_sse2.asm
VP8_CX_SRCS-$(HAVE_SSE2) += encoder/x86/temporal_filter_apply_sse2.asm
VP8_CX_SRCS-$(HAVE_SSE2) += encoder/x86/vp8_enc_stubs_sse2.c

ifeq ($(CONFIG_REALTIME_ONLY),yes)
VP8_CX_SRCS_REMOVE-$(HAVE_SSE2) += encoder/x86/temporal_filter_apply_sse2.asm
endif

VP8_CX_SRCS-$(HAVE_NEON) += encoder/arm/neon/denoising_neon.c
VP8_CX_SRCS-$(HAVE_NEON) += encoder/arm/neon/fastquantizeb_neon.c
VP8_CX_SRCS-$(HAVE_NEON) += encoder/arm/neon/shortfdct_neon.c
VP8_CX_SRCS-$(HAVE_NEON) += encoder/arm/neon/vp8_shortwalsh4x4_neon.c

VP8_CX_SRCS-$(HAVE_MSA) += encoder/mips/msa/dct_msa.c
VP8_CX_SRCS-$(HAVE_MSA) += encoder/mips/msa/encodeopt_msa.c
VP8_CX_SRCS-$(HAVE_MSA) += encoder/mips/msa/quantize_msa.c
VP8_CX_SRCS-$(HAVE_MSA) += encoder/mips/msa/temporal_filter_msa.c

VP8_CX_SRCS-$(HAVE_MMI) += encoder/mips/mmi/vp8_quantize_mmi.c
VP8_CX_SRCS-$(HAVE_MMI) += encoder/mips/mmi/dct_mmi.c

ifeq ($(CONFIG_TEMPORAL_DENOISING),yes)
VP8_CX_SRCS-$(HAVE_MSA) += encoder/mips/msa/denoising_msa.c
endif

ifeq ($(CONFIG_REALTIME_ONLY),yes)
VP8_CX_SRCS_REMOVE-$(HAVE_MSA) += encoder/mips/msa/temporal_filter_msa.c
endif

# common (loongarch LSX intrinsics)
VP8_CX_SRCS-$(HAVE_LSX) += encoder/loongarch/dct_lsx.c
VP8_CX_SRCS-$(HAVE_LSX) += encoder/loongarch/encodeopt_lsx.c
VP8_CX_SRCS-$(HAVE_LSX) += encoder/loongarch/vp8_quantize_lsx.c

VP8_CX_SRCS-yes := $(filter-out $(VP8_CX_SRCS_REMOVE-yes),$(VP8_CX_SRCS-yes))
