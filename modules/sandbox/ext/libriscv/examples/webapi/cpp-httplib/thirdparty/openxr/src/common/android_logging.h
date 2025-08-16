// Copyright (c) 2018-2025 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Utility macros for Android logging.

#pragma once

#ifdef XR_USE_PLATFORM_ANDROID
#include <android/log.h>

#if !defined(LOG_TAG)
#error "LOG_TAG must be defined before including android_logging.h"
#endif

#define ALOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGF(...) __android_log_print(ANDROID_LOG_FATAL, LOG_TAG, __VA_ARGS__)

#endif  // XR_USE_PLATFORM_ANDROID
