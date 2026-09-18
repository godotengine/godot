##
## Copyright (c) 2015 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

UTIL_SRCS-yes += vpx_atomics.h
UTIL_SRCS-yes += vpx_util.mk
UTIL_SRCS-yes += vpx_pthread.h
UTIL_SRCS-yes += vpx_thread.c
UTIL_SRCS-yes += vpx_thread.h
UTIL_SRCS-yes += endian_inl.h
UTIL_SRCS-yes += vpx_write_yuv_frame.h
UTIL_SRCS-yes += vpx_write_yuv_frame.c
UTIL_SRCS-yes += vpx_timestamp.h
UTIL_SRCS-$(or $(CONFIG_BITSTREAM_DEBUG),$(CONFIG_MISMATCH_DEBUG)) += vpx_debug_util.h
UTIL_SRCS-$(or $(CONFIG_BITSTREAM_DEBUG),$(CONFIG_MISMATCH_DEBUG)) += vpx_debug_util.c
