//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// debug.h: Debugging utilities. A lot of the logging code is adapted from Chromium's
// base/logging.h.

#ifndef COMMON_DEBUG_H_
#define COMMON_DEBUG_H_

#include <assert.h>
#include <stdio.h>

#include <iomanip>
#include <ios>
#include <sstream>
#include <string>

#include "common/angleutils.h"
#include "common/platform.h"

#if !defined(TRACE_OUTPUT_FILE)
#    define TRACE_OUTPUT_FILE "angle_debug.txt"
#endif

namespace gl
{

// Pairs a D3D begin event with an end event.
class ScopedPerfEventHelper : angle::NonCopyable
{
  public:
    ANGLE_FORMAT_PRINTF(2, 3)
    ScopedPerfEventHelper(const char *format, ...);
    ~ScopedPerfEventHelper();

  private:
    const char *mFunctionName;
};

using LogSeverity = int;
// Note: the log severities are used to index into the array of names,
// see g_logSeverityNames.
constexpr LogSeverity LOG_EVENT          = 0;
constexpr LogSeverity LOG_INFO           = 1;
constexpr LogSeverity LOG_WARN           = 2;
constexpr LogSeverity LOG_ERR            = 3;
constexpr LogSeverity LOG_FATAL          = 4;
constexpr LogSeverity LOG_NUM_SEVERITIES = 5;

void Trace(LogSeverity severity, const char *message);

// This class more or less represents a particular log message.  You
// create an instance of LogMessage and then stream stuff to it.
// When you finish streaming to it, ~LogMessage is called and the
// full message gets streamed to the appropriate destination.
//
// You shouldn't actually use LogMessage's constructor to log things,
// though.  You should use the ERR() and WARN() macros.
class LogMessage : angle::NonCopyable
{
  public:
    // Used for ANGLE_LOG(severity).
    LogMessage(const char *function, int line, LogSeverity severity);
    ~LogMessage();
    std::ostream &stream() { return mStream; }

    LogSeverity getSeverity() const;
    std::string getMessage() const;

  private:
    const char *mFunction;
    const int mLine;
    const LogSeverity mSeverity;

    std::ostringstream mStream;
};

// Wraps the API/Platform-specific debug annotation functions.
// Also handles redirecting logging destination.
class DebugAnnotator : angle::NonCopyable
{
  public:
    DebugAnnotator() {}
    virtual ~DebugAnnotator() {}
    virtual void beginEvent(const char *eventName, const char *eventMessage) = 0;
    virtual void endEvent(const char *eventName)                             = 0;
    virtual void setMarker(const char *markerName)                           = 0;
    virtual bool getStatus()                                                 = 0;
    // Log Message Handler that gets passed every log message,
    // when debug annotations are initialized,
    // replacing default handling by LogMessage.
    virtual void logMessage(const LogMessage &msg) const = 0;
};

void InitializeDebugAnnotations(DebugAnnotator *debugAnnotator);
void UninitializeDebugAnnotations();
bool DebugAnnotationsActive();
bool DebugAnnotationsInitialized();

void InitializeDebugMutexIfNeeded();

namespace priv
{
// This class is used to explicitly ignore values in the conditional logging macros. This avoids
// compiler warnings like "value computed is not used" and "statement has no effect".
class LogMessageVoidify
{
  public:
    LogMessageVoidify() {}
    // This has to be an operator with a precedence lower than << but higher than ?:
    void operator&(std::ostream &) {}
};

extern std::ostream *gSwallowStream;

// Used by ANGLE_LOG_IS_ON to lazy-evaluate stream arguments.
bool ShouldCreatePlatformLogMessage(LogSeverity severity);

template <int N, typename T>
std::ostream &FmtHex(std::ostream &os, T value)
{
    os << "0x";

    std::ios_base::fmtflags oldFlags = os.flags();
    std::streamsize oldWidth         = os.width();
    std::ostream::char_type oldFill  = os.fill();

    os << std::hex << std::uppercase << std::setw(N) << std::setfill('0') << value;

    os.flags(oldFlags);
    os.width(oldWidth);
    os.fill(oldFill);

    return os;
}

template <typename T>
std::ostream &FmtHexAutoSized(std::ostream &os, T value)
{
    constexpr int N = sizeof(T) * 2;
    return priv::FmtHex<N>(os, value);
}

template <typename T>
class FmtHexHelper
{
  public:
    FmtHexHelper(const char *prefix, T value) : mPrefix(prefix), mValue(value) {}
    explicit FmtHexHelper(T value) : mPrefix(nullptr), mValue(value) {}

  private:
    const char *mPrefix;
    T mValue;

    friend std::ostream &operator<<(std::ostream &os, const FmtHexHelper &fmt)
    {
        if (fmt.mPrefix)
        {
            os << fmt.mPrefix;
        }
        return FmtHexAutoSized(os, fmt.mValue);
    }
};

}  // namespace priv

template <typename T>
priv::FmtHexHelper<T> FmtHex(T value)
{
    return priv::FmtHexHelper<T>(value);
}

#if defined(ANGLE_PLATFORM_WINDOWS)
priv::FmtHexHelper<HRESULT> FmtHR(HRESULT value);
priv::FmtHexHelper<DWORD> FmtErr(DWORD value);
#endif  // defined(ANGLE_PLATFORM_WINDOWS)

template <typename T>
std::ostream &FmtHex(std::ostream &os, T value)
{
    return priv::FmtHexAutoSized(os, value);
}

// A few definitions of macros that don't generate much code. These are used
// by ANGLE_LOG(). Since these are used all over our code, it's
// better to have compact code for these operations.
#define COMPACT_ANGLE_LOG_EX_EVENT(ClassName, ...) \
    ::gl::ClassName(__FUNCTION__, __LINE__, ::gl::LOG_EVENT, ##__VA_ARGS__)
#define COMPACT_ANGLE_LOG_EX_INFO(ClassName, ...) \
    ::gl::ClassName(__FUNCTION__, __LINE__, ::gl::LOG_INFO, ##__VA_ARGS__)
#define COMPACT_ANGLE_LOG_EX_WARN(ClassName, ...) \
    ::gl::ClassName(__FUNCTION__, __LINE__, ::gl::LOG_WARN, ##__VA_ARGS__)
#define COMPACT_ANGLE_LOG_EX_ERR(ClassName, ...) \
    ::gl::ClassName(__FUNCTION__, __LINE__, ::gl::LOG_ERR, ##__VA_ARGS__)
#define COMPACT_ANGLE_LOG_EX_FATAL(ClassName, ...) \
    ::gl::ClassName(__FUNCTION__, __LINE__, ::gl::LOG_FATAL, ##__VA_ARGS__)

#define COMPACT_ANGLE_LOG_EVENT COMPACT_ANGLE_LOG_EX_EVENT(LogMessage)
#define COMPACT_ANGLE_LOG_INFO COMPACT_ANGLE_LOG_EX_INFO(LogMessage)
#define COMPACT_ANGLE_LOG_WARN COMPACT_ANGLE_LOG_EX_WARN(LogMessage)
#define COMPACT_ANGLE_LOG_ERR COMPACT_ANGLE_LOG_EX_ERR(LogMessage)
#define COMPACT_ANGLE_LOG_FATAL COMPACT_ANGLE_LOG_EX_FATAL(LogMessage)

#define ANGLE_LOG_IS_ON(severity) (::gl::priv::ShouldCreatePlatformLogMessage(::gl::LOG_##severity))

// Helper macro which avoids evaluating the arguments to a stream if the condition doesn't hold.
// Condition is evaluated once and only once.
#define ANGLE_LAZY_STREAM(stream, condition) \
    !(condition) ? static_cast<void>(0) : ::gl::priv::LogMessageVoidify() & (stream)

// We use the preprocessor's merging operator, "##", so that, e.g.,
// ANGLE_LOG(EVENT) becomes the token COMPACT_ANGLE_LOG_EVENT.  There's some funny
// subtle difference between ostream member streaming functions (e.g.,
// ostream::operator<<(int) and ostream non-member streaming functions
// (e.g., ::operator<<(ostream&, string&): it turns out that it's
// impossible to stream something like a string directly to an unnamed
// ostream. We employ a neat hack by calling the stream() member
// function of LogMessage which seems to avoid the problem.
#define ANGLE_LOG_STREAM(severity) COMPACT_ANGLE_LOG_##severity.stream()

#define ANGLE_LOG(severity) ANGLE_LAZY_STREAM(ANGLE_LOG_STREAM(severity), ANGLE_LOG_IS_ON(severity))

}  // namespace gl

#if defined(ANGLE_ENABLE_DEBUG_TRACE) || defined(ANGLE_ENABLE_DEBUG_ANNOTATIONS)
#    define ANGLE_TRACE_ENABLED
#endif

#if !defined(NDEBUG) || defined(ANGLE_ENABLE_RELEASE_ASSERTS)
#    define ANGLE_ENABLE_ASSERTS
#endif

#define INFO() ANGLE_LOG(INFO)
#define WARN() ANGLE_LOG(WARN)
#define ERR() ANGLE_LOG(ERR)
#define FATAL() ANGLE_LOG(FATAL)

// A macro to log a performance event around a scope.
#if defined(ANGLE_TRACE_ENABLED)
#    if defined(_MSC_VER)
#        define EVENT(function, message, ...)                                                      \
            gl::ScopedPerfEventHelper scopedPerfEventHelper##__LINE__("%s(" message ")", function, \
                                                                      __VA_ARGS__)
#    else
#        define EVENT(function, message, ...)                                            \
            gl::ScopedPerfEventHelper scopedPerfEventHelper("%s(" message ")", function, \
                                                            ##__VA_ARGS__)
#    endif  // _MSC_VER
#else
#    define EVENT(message, ...) (void(0))
#endif

// The state tracked by ANGLE will be validated with the driver state before each call
#if defined(ANGLE_ENABLE_ASSERTS)
#    define ANGLE_STATE_VALIDATION_ENABLED
#endif

#if defined(__GNUC__)
#    define ANGLE_CRASH() __builtin_trap()
#else
#    define ANGLE_CRASH() ((void)(*(volatile char *)0 = 0)), __assume(0)
#endif

#if !defined(NDEBUG)
#    define ANGLE_ASSERT_IMPL(expression) assert(expression)
#else
// TODO(jmadill): Detect if debugger is attached and break.
#    define ANGLE_ASSERT_IMPL(expression) ANGLE_CRASH()
#endif  // !defined(NDEBUG)

// Note that gSwallowStream is used instead of an arbitrary LOG() stream to avoid the creation of an
// object with a non-trivial destructor (LogMessage). On MSVC x86 (checked on 2015 Update 3), this
// causes a few additional pointless instructions to be emitted even at full optimization level,
// even though the : arm of the ternary operator is clearly never executed. Using a simpler object
// to be &'d with Voidify() avoids these extra instructions. Using a simpler POD object with a
// templated operator<< also works to avoid these instructions. However, this causes warnings on
// statically defined implementations of operator<<(std::ostream, ...) in some .cpp files, because
// they become defined-but-unreferenced functions. A reinterpret_cast of 0 to an ostream* also is
// not suitable, because some compilers warn of undefined behavior.
#define ANGLE_EAT_STREAM_PARAMETERS \
    true ? static_cast<void>(0) : ::gl::priv::LogMessageVoidify() & (*::gl::priv::gSwallowStream)

// A macro asserting a condition and outputting failures to the debug log
#if defined(ANGLE_ENABLE_ASSERTS)
#    define ASSERT(expression)                                                                \
        (expression ? static_cast<void>(0)                                                    \
                    : (FATAL() << "\t! Assert failed in " << __FUNCTION__ << " (" << __FILE__ \
                               << ":" << __LINE__ << "): " << #expression))
#else
#    define ASSERT(condition) ANGLE_EAT_STREAM_PARAMETERS << !(condition)
#endif  // defined(ANGLE_ENABLE_ASSERTS)

#define UNREACHABLE_IS_NORETURN 0

#define ANGLE_UNUSED_VARIABLE(variable) (static_cast<void>(variable))

// A macro to indicate unimplemented functionality
#ifndef NOASSERT_UNIMPLEMENTED
#    define NOASSERT_UNIMPLEMENTED 1
#endif

#if defined(ANGLE_TRACE_ENABLED) || defined(ANGLE_ENABLE_ASSERTS)
#    define UNIMPLEMENTED()                                                                       \
        do                                                                                        \
        {                                                                                         \
            WARN() << "\t! Unimplemented: " << __FUNCTION__ << "(" << __FILE__ << ":" << __LINE__ \
                   << ")";                                                                        \
            ASSERT(NOASSERT_UNIMPLEMENTED);                                                       \
        } while (0)

// A macro for code which is not expected to be reached under valid assumptions
#    define UNREACHABLE()                                                                    \
        do                                                                                   \
        {                                                                                    \
            FATAL() << "\t! Unreachable reached: " << __FUNCTION__ << "(" << __FILE__ << ":" \
                    << __LINE__ << ")";                                                      \
        } while (0)
#else
#    define UNIMPLEMENTED()                 \
        do                                  \
        {                                   \
            ASSERT(NOASSERT_UNIMPLEMENTED); \
        } while (0)

// A macro for code which is not expected to be reached under valid assumptions
#    define UNREACHABLE()  \
        do                 \
        {                  \
            ASSERT(false); \
        } while (0)
#endif  // defined(ANGLE_TRACE_ENABLED) || defined(ANGLE_ENABLE_ASSERTS)

#if defined(ANGLE_PLATFORM_WINDOWS)
#    define ANGLE_FUNCTION __FUNCTION__
#else
#    define ANGLE_FUNCTION __func__
#endif

// Defining ANGLE_ENABLE_STRUCT_PADDING_WARNINGS will enable warnings when members are added to
// structs to enforce packing. This is helpful for diagnosing unexpected struct sizes when making
// fast cache variables.
#if defined(__clang__)
#    define ANGLE_ENABLE_STRUCT_PADDING_WARNINGS \
        _Pragma("clang diagnostic push") _Pragma("clang diagnostic error \"-Wpadded\"")
#    define ANGLE_DISABLE_STRUCT_PADDING_WARNINGS _Pragma("clang diagnostic pop")
#elif defined(__GNUC__)
#    define ANGLE_ENABLE_STRUCT_PADDING_WARNINGS \
        _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic error \"-Wpadded\"")
#    define ANGLE_DISABLE_STRUCT_PADDING_WARNINGS _Pragma("GCC diagnostic pop")
#elif defined(_MSC_VER)
#    define ANGLE_ENABLE_STRUCT_PADDING_WARNINGS \
        __pragma(warning(push)) __pragma(warning(error : 4820))
#    define ANGLE_DISABLE_STRUCT_PADDING_WARNINGS __pragma(warning(pop))
#else
#    define ANGLE_ENABLE_STRUCT_PADDING_WARNINGS
#    define ANGLE_DISABLE_STRUCT_PADDING_WARNINGS
#endif

#if defined(__clang__)
#    define ANGLE_DISABLE_EXTRA_SEMI_WARNING \
        _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wextra-semi\"")
#    define ANGLE_REENABLE_EXTRA_SEMI_WARNING _Pragma("clang diagnostic pop")
#else
#    define ANGLE_DISABLE_EXTRA_SEMI_WARNING
#    define ANGLE_REENABLE_EXTRA_SEMI_WARNING
#endif

#endif  // COMMON_DEBUG_H_
