##
##  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

VP9_DX_EXPORTS += exports_dec

VP9_DX_SRCS-yes += $(VP9_COMMON_SRCS-yes)
VP9_DX_SRCS-no  += $(VP9_COMMON_SRCS-no)
VP9_DX_SRCS_REMOVE-yes += $(VP9_COMMON_SRCS_REMOVE-yes)
VP9_DX_SRCS_REMOVE-no  += $(VP9_COMMON_SRCS_REMOVE-no)

VP9_DX_SRCS-yes += vp9_dx_iface.c
VP9_DX_SRCS-yes += vp9_dx_iface.h

VP9_DX_SRCS-yes += decoder/vp9_decodemv.c
VP9_DX_SRCS-yes += decoder/vp9_decodeframe.c
VP9_DX_SRCS-yes += decoder/vp9_decodeframe.h
VP9_DX_SRCS-yes += decoder/vp9_detokenize.c
VP9_DX_SRCS-yes += decoder/vp9_decodemv.h
VP9_DX_SRCS-yes += decoder/vp9_detokenize.h
VP9_DX_SRCS-yes += decoder/vp9_decoder.c
VP9_DX_SRCS-yes += decoder/vp9_decoder.h
VP9_DX_SRCS-yes += decoder/vp9_dsubexp.c
VP9_DX_SRCS-yes += decoder/vp9_dsubexp.h
VP9_DX_SRCS-yes += decoder/vp9_job_queue.c
VP9_DX_SRCS-yes += decoder/vp9_job_queue.h

VP9_DX_SRCS-yes := $(filter-out $(VP9_DX_SRCS_REMOVE-yes),$(VP9_DX_SRCS-yes))
