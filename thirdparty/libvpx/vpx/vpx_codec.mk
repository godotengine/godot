##
##  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##


API_EXPORTS += exports

API_SRCS-$(CONFIG_VP8_ENCODER) += vp8.h
API_SRCS-$(CONFIG_VP8_ENCODER) += vp8cx.h
API_DOC_SRCS-$(CONFIG_VP8_ENCODER) += vp8.h
API_DOC_SRCS-$(CONFIG_VP8_ENCODER) += vp8cx.h

API_SRCS-$(CONFIG_VP8_DECODER) += vp8.h
API_SRCS-$(CONFIG_VP8_DECODER) += vp8dx.h
API_DOC_SRCS-$(CONFIG_VP8_DECODER) += vp8.h
API_DOC_SRCS-$(CONFIG_VP8_DECODER) += vp8dx.h

API_DOC_SRCS-yes += vpx_codec.h
API_DOC_SRCS-yes += vpx_decoder.h
API_DOC_SRCS-yes += vpx_encoder.h
API_DOC_SRCS-$(CONFIG_ENCODERS) += vpx_ext_ratectrl.h
API_DOC_SRCS-yes += vpx_frame_buffer.h
API_DOC_SRCS-yes += vpx_image.h
API_DOC_SRCS-$(CONFIG_ENCODERS) += vpx_tpl.h

API_SRCS-yes += src/vpx_decoder.c
API_SRCS-yes += vpx_decoder.h
API_SRCS-yes += src/vpx_encoder.c
API_SRCS-yes += vpx_encoder.h
API_SRCS-yes += internal/vpx_codec_internal.h
API_SRCS-yes += internal/vpx_ratectrl_rtc.h
API_SRCS-yes += src/vpx_codec.c
API_SRCS-yes += src/vpx_image.c
API_SRCS-yes += vpx_codec.h
API_SRCS-yes += vpx_codec.mk
API_SRCS-yes += vpx_frame_buffer.h
API_SRCS-yes += vpx_image.h
API_SRCS-yes += vpx_integer.h
API_SRCS-yes += vpx_ext_ratectrl.h
API_SRCS-yes += vpx_tpl.h
