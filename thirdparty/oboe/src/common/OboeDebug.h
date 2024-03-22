/*
 * Copyright (C) 2015 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef OBOE_DEBUG_H
#define OBOE_DEBUG_H

#include <android/log.h>

#ifndef MODULE_NAME
#define MODULE_NAME  "OboeAudio"
#endif

// Always log INFO and errors.
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, MODULE_NAME, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, MODULE_NAME, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, MODULE_NAME, __VA_ARGS__)
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL, MODULE_NAME, __VA_ARGS__)

#if OBOE_ENABLE_LOGGING
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, MODULE_NAME, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, MODULE_NAME, __VA_ARGS__)
#else
#define LOGV(...)
#define LOGD(...)
#endif

#endif //OBOE_DEBUG_H
