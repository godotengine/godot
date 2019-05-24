// Copyright (c) 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOURCE_OPT_LOG_H_
#define SOURCE_OPT_LOG_H_

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "spirv-tools/libspirv.hpp"

// Asserts the given condition is true. Otherwise, sends a message to the
// consumer and exits the problem with failure code. Accepts the following
// formats:
//
// SPIRV_ASSERT(<message-consumer>, <condition-expression>);
// SPIRV_ASSERT(<message-consumer>, <condition-expression>, <message>);
// SPIRV_ASSERT(<message-consumer>, <condition-expression>,
//              <message-format>,   <variable-arguments>);
//
// In the third format, the number of <variable-arguments> cannot exceed (5 -
// 2). If more arguments are wanted, grow PP_ARG_N and PP_NARGS in the below.
#if !defined(NDEBUG)
#define SPIRV_ASSERT(consumer, ...) SPIRV_ASSERT_IMPL(consumer, __VA_ARGS__)
#else
#define SPIRV_ASSERT(consumer, ...)
#endif

// Logs a debug message to the consumer. Accepts the following formats:
//
// SPIRV_DEBUG(<message-consumer>, <message>);
// SPIRV_DEBUG(<message-consumer>, <message-format>, <variable-arguments>);
//
// In the second format, the number of <variable-arguments> cannot exceed (5 -
// 1). If more arguments are wanted, grow PP_ARG_N and PP_NARGS in the below.
#if !defined(NDEBUG) && defined(SPIRV_LOG_DEBUG)
#define SPIRV_DEBUG(consumer, ...) SPIRV_DEBUG_IMPL(consumer, __VA_ARGS__)
#else
#define SPIRV_DEBUG(consumer, ...)
#endif

// Logs an error message to the consumer saying the given feature is
// unimplemented.
#define SPIRV_UNIMPLEMENTED(consumer, feature)                  \
  do {                                                          \
    spvtools::Log(consumer, SPV_MSG_INTERNAL_ERROR, __FILE__,   \
                  {__LINE__, 0, 0}, "unimplemented: " feature); \
  } while (0)

// Logs an error message to the consumer saying the code location
// should be unreachable.
#define SPIRV_UNREACHABLE(consumer)                           \
  do {                                                        \
    spvtools::Log(consumer, SPV_MSG_INTERNAL_ERROR, __FILE__, \
                  {__LINE__, 0, 0}, "unreachable");           \
  } while (0)

// Helper macros for concatenating arguments.
#define SPIRV_CONCATENATE(a, b) SPIRV_CONCATENATE_(a, b)
#define SPIRV_CONCATENATE_(a, b) a##b

// Helper macro to force expanding __VA_ARGS__ to satisfy MSVC compiler.
#define PP_EXPAND(x) x

namespace spvtools {

// Calls the given |consumer| by supplying  the |message|. The |message| is from
// the given |source| and |location| and of the given severity |level|.
inline void Log(const MessageConsumer& consumer, spv_message_level_t level,
                const char* source, const spv_position_t& position,
                const char* message) {
  if (consumer != nullptr) consumer(level, source, position, message);
}

// Calls the given |consumer| by supplying the message composed according to the
// given |format|. The |message| is from the given |source| and |location| and
// of the given severity |level|.
template <typename... Args>
void Logf(const MessageConsumer& consumer, spv_message_level_t level,
          const char* source, const spv_position_t& position,
          const char* format, Args&&... args) {
#if defined(_MSC_VER) && _MSC_VER < 1900
// Sadly, snprintf() is not supported until Visual Studio 2015!
#define snprintf _snprintf
#endif

  enum { kInitBufferSize = 256 };

  char message[kInitBufferSize];
  const int size =
      snprintf(message, kInitBufferSize, format, std::forward<Args>(args)...);

  if (size >= 0 && size < kInitBufferSize) {
    Log(consumer, level, source, position, message);
    return;
  }

  if (size >= 0) {
    // The initial buffer is insufficient.  Allocate a buffer of a larger size,
    // and write to it instead.  Force the size to be unsigned to avoid a
    // warning in GCC 7.1.
    std::vector<char> longer_message(size + 1u);
    snprintf(longer_message.data(), longer_message.size(), format,
             std::forward<Args>(args)...);
    Log(consumer, level, source, position, longer_message.data());
    return;
  }

  Log(consumer, level, source, position, "cannot compose log message");

#if defined(_MSC_VER) && _MSC_VER < 1900
#undef snprintf
#endif
}

// Calls the given |consumer| by supplying  the given error |message|. The
// |message| is from the given |source| and |location|.
inline void Error(const MessageConsumer& consumer, const char* source,
                  const spv_position_t& position, const char* message) {
  Log(consumer, SPV_MSG_ERROR, source, position, message);
}

// Calls the given |consumer| by supplying the error message composed according
// to the given |format|. The |message| is from the given |source| and
// |location|.
template <typename... Args>
inline void Errorf(const MessageConsumer& consumer, const char* source,
                   const spv_position_t& position, const char* format,
                   Args&&... args) {
  Logf(consumer, SPV_MSG_ERROR, source, position, format,
       std::forward<Args>(args)...);
}

}  // namespace spvtools

#define SPIRV_ASSERT_IMPL(consumer, ...)                             \
  PP_EXPAND(SPIRV_CONCATENATE(SPIRV_ASSERT_, PP_NARGS(__VA_ARGS__))( \
      consumer, __VA_ARGS__))

#define SPIRV_DEBUG_IMPL(consumer, ...)                             \
  PP_EXPAND(SPIRV_CONCATENATE(SPIRV_DEBUG_, PP_NARGS(__VA_ARGS__))( \
      consumer, __VA_ARGS__))

#define SPIRV_ASSERT_1(consumer, condition)                             \
  do {                                                                  \
    if (!(condition)) {                                                 \
      spvtools::Log(consumer, SPV_MSG_INTERNAL_ERROR, __FILE__,         \
                    {__LINE__, 0, 0}, "assertion failed: " #condition); \
      std::exit(EXIT_FAILURE);                                          \
    }                                                                   \
  } while (0)

#define SPIRV_ASSERT_2(consumer, condition, message)                 \
  do {                                                               \
    if (!(condition)) {                                              \
      spvtools::Log(consumer, SPV_MSG_INTERNAL_ERROR, __FILE__,      \
                    {__LINE__, 0, 0}, "assertion failed: " message); \
      std::exit(EXIT_FAILURE);                                       \
    }                                                                \
  } while (0)

#define SPIRV_ASSERT_more(consumer, condition, format, ...)         \
  do {                                                              \
    if (!(condition)) {                                             \
      spvtools::Logf(consumer, SPV_MSG_INTERNAL_ERROR, __FILE__,    \
                     {__LINE__, 0, 0}, "assertion failed: " format, \
                     __VA_ARGS__);                                  \
      std::exit(EXIT_FAILURE);                                      \
    }                                                               \
  } while (0)

#define SPIRV_ASSERT_3(consumer, condition, format, ...) \
  SPIRV_ASSERT_more(consumer, condition, format, __VA_ARGS__)

#define SPIRV_ASSERT_4(consumer, condition, format, ...) \
  SPIRV_ASSERT_more(consumer, condition, format, __VA_ARGS__)

#define SPIRV_ASSERT_5(consumer, condition, format, ...) \
  SPIRV_ASSERT_more(consumer, condition, format, __VA_ARGS__)

#define SPIRV_DEBUG_1(consumer, message)                               \
  do {                                                                 \
    spvtools::Log(consumer, SPV_MSG_DEBUG, __FILE__, {__LINE__, 0, 0}, \
                  message);                                            \
  } while (0)

#define SPIRV_DEBUG_more(consumer, format, ...)                         \
  do {                                                                  \
    spvtools::Logf(consumer, SPV_MSG_DEBUG, __FILE__, {__LINE__, 0, 0}, \
                   format, __VA_ARGS__);                                \
  } while (0)

#define SPIRV_DEBUG_2(consumer, format, ...) \
  SPIRV_DEBUG_more(consumer, format, __VA_ARGS__)

#define SPIRV_DEBUG_3(consumer, format, ...) \
  SPIRV_DEBUG_more(consumer, format, __VA_ARGS__)

#define SPIRV_DEBUG_4(consumer, format, ...) \
  SPIRV_DEBUG_more(consumer, format, __VA_ARGS__)

#define SPIRV_DEBUG_5(consumer, format, ...) \
  SPIRV_DEBUG_more(consumer, format, __VA_ARGS__)

// Macros for counting the number of arguments passed in.
#define PP_NARGS(...) PP_EXPAND(PP_ARG_N(__VA_ARGS__, 5, 4, 3, 2, 1, 0))
#define PP_ARG_N(_1, _2, _3, _4, _5, N, ...) N

// Tests for making sure that PP_NARGS() behaves as expected.
static_assert(PP_NARGS(0) == 1, "PP_NARGS macro error");
static_assert(PP_NARGS(0, 0) == 2, "PP_NARGS macro error");
static_assert(PP_NARGS(0, 0, 0) == 3, "PP_NARGS macro error");
static_assert(PP_NARGS(0, 0, 0, 0) == 4, "PP_NARGS macro error");
static_assert(PP_NARGS(0, 0, 0, 0, 0) == 5, "PP_NARGS macro error");
static_assert(PP_NARGS(1 + 1, 2, 3 / 3) == 3, "PP_NARGS macro error");
static_assert(PP_NARGS((1, 1), 2, (3, 3)) == 3, "PP_NARGS macro error");

#endif  // SOURCE_OPT_LOG_H_
