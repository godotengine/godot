// Copyright 2016 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef DRACO_CORE_MACROS_H_
#define DRACO_CORE_MACROS_H_

#include <cassert>

#include "draco/draco_features.h"

#ifdef ANDROID_LOGGING
#include <android/log.h>
#define LOG_TAG "draco"
#define DRACO_LOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define DRACO_LOGE(...) \
  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define DRACO_LOGI printf
#define DRACO_LOGE printf
#endif

#include <iostream>
namespace draco {

#ifndef FALLTHROUGH_INTENDED
#if defined(__clang__) && defined(__has_warning)
#if __has_feature(cxx_attributes) && __has_warning("-Wimplicit-fallthrough")
#define FALLTHROUGH_INTENDED [[clang::fallthrough]]
#endif
#elif defined(__GNUC__) && __GNUC__ >= 7
#define FALLTHROUGH_INTENDED [[gnu::fallthrough]]
#endif  // FALLTHROUGH_INTENDED

// If FALLTHROUGH_INTENDED is still not defined, define it.
#ifndef FALLTHROUGH_INTENDED
#define FALLTHROUGH_INTENDED \
  do {                       \
  } while (0)
#endif
#endif  // FALLTHROUGH_INTENDED

#ifndef LOG
#define LOG(...) std::cout
#endif

#ifndef VLOG
#define VLOG(...) std::cout
#endif

}  // namespace draco

#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName &) = delete;     \
  void operator=(const TypeName &) = delete;
#endif  // DISALLOW_COPY_AND_ASSIGN

#ifdef DRACO_DEBUG
#define DRACO_DCHECK(x) (assert(x));
#define DRACO_DCHECK_EQ(a, b) assert((a) == (b));
#define DRACO_DCHECK_NE(a, b) assert((a) != (b));
#define DRACO_DCHECK_GE(a, b) assert((a) >= (b));
#define DRACO_DCHECK_GT(a, b) assert((a) > (b));
#define DRACO_DCHECK_LE(a, b) assert((a) <= (b));
#define DRACO_DCHECK_LT(a, b) assert((a) < (b));
#define DRACO_DCHECK_NOTNULL(x) assert((x) != NULL);
#else
#define DRACO_DCHECK(x)
#define DRACO_DCHECK_EQ(a, b)
#define DRACO_DCHECK_NE(a, b)
#define DRACO_DCHECK_GE(a, b)
#define DRACO_DCHECK_GT(a, b)
#define DRACO_DCHECK_LE(a, b)
#define DRACO_DCHECK_LT(a, b)
#define DRACO_DCHECK_NOTNULL(x)
#endif  // DRACO_DEBUG

// Helper macros for concatenating macro values.
#define DRACO_MACROS_IMPL_CONCAT_INNER_(x, y) x##y
#define DRACO_MACROS_IMPL_CONCAT_(x, y) DRACO_MACROS_IMPL_CONCAT_INNER_(x, y)

#define DRACO_MACROS_IMPL_CONCAT_INNER_3_(x, y, z) x##y##z
#define DRACO_MACROS_IMPL_CONCAT_3_(x, y, z) \
  DRACO_MACROS_IMPL_CONCAT_INNER_3_(x, y, z)

// Expand the n-th argument of the macro. Used to select an argument based on
// the number of entries in a variadic macro argument. Example usage:
//
// #define FUNC_1(x) x
// #define FUNC_2(x, y) x + y
// #define FUNC_3(x, y, z) x + y + z
//
// #define VARIADIC_MACRO(...)
//   DRACO_SELECT_NTH_FROM_3(__VA_ARGS__, FUNC_3, FUNC_2, FUNC_1) __VA_ARGS__
//
#define DRACO_SELECT_NTH_FROM_2(_1, _2, NAME, ...) NAME
#define DRACO_SELECT_NTH_FROM_3(_1, _2, _3, NAME, ...) NAME
#define DRACO_SELECT_NTH_FROM_4(_1, _2, _3, _4, NAME, ...) NAME

// Macro that converts the Draco bit-stream into one uint16_t number.
// Useful mostly when checking version numbers.
#define DRACO_BITSTREAM_VERSION(MAJOR, MINOR) \
  ((static_cast<uint16_t>(MAJOR) << 8) | MINOR)

// Macro that converts the uint16_t Draco bit-stream number into the major
// and minor components respectively.
#define DRACO_BISTREAM_VERSION_MAJOR(VERSION) \
  (static_cast<uint8_t>(VERSION >> 8))
#define DRACO_BISTREAM_VERSION_MINOR(VERSION) \
  (static_cast<uint8_t>(VERSION & 0xFF))

#endif  // DRACO_CORE_MACROS_H_
