# Copyright (c) 2013 The WebM project authors. All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the LICENSE file in the root of the source
# tree. An additional intellectual property rights grant can be found
# in the file PATENTS.  All contributing project authors may
# be found in the AUTHORS file in the root of the source tree.
#
# This make file builds vpx_test app for android.
# The test app itself runs on the command line through adb shell
# The paths are really messed up as the libvpx make file
# expects to be made from a parent directory.

# Ignore this file during non-NDK builds.
ifdef NDK_ROOT
CUR_WD := $(call my-dir)
BINDINGS_DIR := $(CUR_WD)/../../..
LOCAL_PATH := $(CUR_WD)/../../..

#libwebm
include $(CLEAR_VARS)
include $(BINDINGS_DIR)/libvpx/third_party/libwebm/Android.mk
LOCAL_PATH := $(CUR_WD)/../../..

#libvpx
include $(CLEAR_VARS)
LOCAL_STATIC_LIBRARIES := libwebm
include $(BINDINGS_DIR)/libvpx/build/make/Android.mk
LOCAL_PATH := $(CUR_WD)/../..

#libgtest
include $(CLEAR_VARS)
LOCAL_ARM_MODE := arm
LOCAL_CPP_EXTENSION := .cc
LOCAL_MODULE := gtest
LOCAL_C_INCLUDES := $(LOCAL_PATH)/third_party/googletest/src/
LOCAL_C_INCLUDES += $(LOCAL_PATH)/third_party/googletest/src/include/
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/third_party/googletest/src/include/
LOCAL_SRC_FILES := ./third_party/googletest/src/src/gtest-all.cc
LOCAL_LICENSE_KINDS := SPDX-license-identifier-BSD
LOCAL_LICENSE_CONDITIONS := notice
LOCAL_NOTICE_FILE := $(LOCAL_PATH)/../../LICENSE $(LOCAL_PATH)/../../PATENTS
include $(BUILD_STATIC_LIBRARY)

#libvpx_test
include $(CLEAR_VARS)
LOCAL_ARM_MODE := arm
LOCAL_MODULE := libvpx_test
LOCAL_STATIC_LIBRARIES := gtest libwebm

ifeq ($(ENABLE_SHARED),1)
  LOCAL_SHARED_LIBRARIES := vpx
else
  LOCAL_STATIC_LIBRARIES += vpx
endif

LOCAL_LICENSE_KINDS := SPDX-license-identifier-BSD
LOCAL_LICENSE_CONDITIONS := notice
LOCAL_NOTICE_FILE := $(LOCAL_PATH)/../../LICENSE $(LOCAL_PATH)/../../PATENTS
include $(LOCAL_PATH)/test/test.mk
LOCAL_C_INCLUDES := $(BINDINGS_DIR)
FILTERED_SRC := $(sort $(filter %.cc %.c, $(LIBVPX_TEST_SRCS-yes)))
LOCAL_SRC_FILES := $(addprefix ./test/, $(FILTERED_SRC))
# some test files depend on *_rtcd.h, ensure they're generated first.
$(eval $(call rtcd_dep_template))
include $(BUILD_EXECUTABLE)
endif  # NDK_ROOT
