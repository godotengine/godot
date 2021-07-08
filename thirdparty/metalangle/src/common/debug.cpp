//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// debug.cpp: Debugging utilities.

#include "common/debug.h"

#include <stdarg.h>

#include <array>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <ostream>
#include <vector>

#if defined(ANGLE_PLATFORM_ANDROID)
#    include <android/log.h>
#endif

#include "anglebase/no_destructor.h"
#include "common/Optional.h"
#include "common/angleutils.h"
#include "common/system_utils.h"

namespace gl
{

namespace
{

DebugAnnotator *g_debugAnnotator = nullptr;

std::mutex *g_debugMutex = nullptr;

constexpr std::array<const char *, LOG_NUM_SEVERITIES> g_logSeverityNames = {
    {"EVENT", "INFO", "WARN", "ERR", "FATAL"}};

constexpr const char *LogSeverityName(int severity)
{
    return (severity >= 0 && severity < LOG_NUM_SEVERITIES) ? g_logSeverityNames[severity]
                                                            : "UNKNOWN";
}

bool ShouldCreateLogMessage(LogSeverity severity)
{
#if defined(ANGLE_TRACE_ENABLED)
    return true;
#elif defined(ANGLE_ENABLE_ASSERTS)
    return severity != LOG_EVENT;
#else
    return false;
#endif
}

}  // namespace

namespace priv
{

bool ShouldCreatePlatformLogMessage(LogSeverity severity)
{
#if defined(ANGLE_TRACE_ENABLED)
    return true;
#else
    return severity != LOG_EVENT;
#endif
}

// This is never instantiated, it's just used for EAT_STREAM_PARAMETERS to an object of the correct
// type on the LHS of the unused part of the ternary operator.
std::ostream *gSwallowStream;
}  // namespace priv

bool DebugAnnotationsActive()
{
#if defined(ANGLE_ENABLE_DEBUG_ANNOTATIONS)
    return g_debugAnnotator != nullptr && g_debugAnnotator->getStatus();
#else
    return false;
#endif
}

bool DebugAnnotationsInitialized()
{
    return g_debugAnnotator != nullptr;
}

void InitializeDebugAnnotations(DebugAnnotator *debugAnnotator)
{
    UninitializeDebugAnnotations();
    g_debugAnnotator = debugAnnotator;
}

void UninitializeDebugAnnotations()
{
    // Pointer is not managed.
    g_debugAnnotator = nullptr;
}

void InitializeDebugMutexIfNeeded()
{
    if (g_debugMutex == nullptr)
    {
        g_debugMutex = new std::mutex();
    }
}

ScopedPerfEventHelper::ScopedPerfEventHelper(const char *format, ...) : mFunctionName(nullptr)
{
    bool dbgTrace = DebugAnnotationsActive();
#if !defined(ANGLE_ENABLE_DEBUG_TRACE)
    if (!dbgTrace)
    {
        return;
    }
#endif  // !ANGLE_ENABLE_DEBUG_TRACE

    va_list vararg;
    va_start(vararg, format);
    std::vector<char> buffer(512);
    size_t len = FormatStringIntoVector(format, vararg, buffer);
    ANGLE_LOG(EVENT) << std::string(&buffer[0], len);
    // Pull function name from variable args
    mFunctionName = va_arg(vararg, const char *);
    va_end(vararg);
    if (dbgTrace)
    {
        g_debugAnnotator->beginEvent(mFunctionName, buffer.data());
    }
}

ScopedPerfEventHelper::~ScopedPerfEventHelper()
{
    if (DebugAnnotationsActive())
    {
        g_debugAnnotator->endEvent(mFunctionName);
    }
}

LogMessage::LogMessage(const char *function, int line, LogSeverity severity)
    : mFunction(function), mLine(line), mSeverity(severity)
{
    // EVENT() does not require additional function(line) info.
    if (mSeverity != LOG_EVENT)
    {
        mStream << mFunction << "(" << mLine << "): ";
    }
}

LogMessage::~LogMessage()
{
    std::unique_lock<std::mutex> lock;
    if (g_debugMutex != nullptr)
    {
        lock = std::unique_lock<std::mutex>(*g_debugMutex);
    }

    if (DebugAnnotationsInitialized() && (mSeverity >= LOG_INFO))
    {
        g_debugAnnotator->logMessage(*this);
    }
    else
    {
        Trace(getSeverity(), getMessage().c_str());
    }

    if (mSeverity == LOG_FATAL)
    {
        if (angle::IsDebuggerAttached())
        {
            angle::BreakDebugger();
        }
        else
        {
            ANGLE_CRASH();
        }
    }
}

void Trace(LogSeverity severity, const char *message)
{
    if (!ShouldCreateLogMessage(severity))
    {
        return;
    }

    std::string str(message);

    if (DebugAnnotationsActive())
    {

        switch (severity)
        {
            case LOG_EVENT:
                // Debugging logging done in ScopedPerfEventHelper
                break;
            default:
                g_debugAnnotator->setMarker(message);
                break;
        }
    }

    if (severity == LOG_FATAL || severity == LOG_ERR || severity == LOG_WARN ||
        severity == LOG_INFO)
    {
#if defined(ANGLE_PLATFORM_ANDROID)
        android_LogPriority android_priority = ANDROID_LOG_ERROR;
        switch (severity)
        {
            case LOG_INFO:
                android_priority = ANDROID_LOG_INFO;
                break;
            case LOG_WARN:
                android_priority = ANDROID_LOG_WARN;
                break;
            case LOG_ERR:
                android_priority = ANDROID_LOG_ERROR;
                break;
            case LOG_FATAL:
                android_priority = ANDROID_LOG_FATAL;
                break;
            default:
                UNREACHABLE();
        }
        __android_log_print(android_priority, "ANGLE", "%s: %s\n", LogSeverityName(severity),
                            str.c_str());
#else
        // Note: we use fprintf because <iostream> includes static initializers.
        fprintf((severity >= LOG_ERR) ? stderr : stdout, "%s: %s\n", LogSeverityName(severity),
                str.c_str());
#endif
    }

#if defined(ANGLE_PLATFORM_WINDOWS) && \
    (defined(ANGLE_ENABLE_DEBUG_TRACE_TO_DEBUGGER) || !defined(NDEBUG))
#    if !defined(ANGLE_ENABLE_DEBUG_TRACE_TO_DEBUGGER)
    if (severity >= LOG_ERR)
#    endif  // !defined(ANGLE_ENABLE_DEBUG_TRACE_TO_DEBUGGER)
    {
        OutputDebugStringA(str.c_str());
    }
#endif

#if defined(ANGLE_ENABLE_DEBUG_TRACE)
#    if defined(NDEBUG)
    if (severity == LOG_EVENT || severity == LOG_WARN || severity == LOG_INFO)
    {
        return;
    }
#    endif  // defined(NDEBUG)
    static angle::base::NoDestructor<std::ofstream> file(TRACE_OUTPUT_FILE, std::ofstream::app);
    if (file->good())
    {
        if (severity > LOG_EVENT)
        {
            *file << LogSeverityName(severity) << ": ";
        }
        *file << str << "\n";
        file->flush();
    }
#endif  // defined(ANGLE_ENABLE_DEBUG_TRACE)
}

LogSeverity LogMessage::getSeverity() const
{
    return mSeverity;
}

std::string LogMessage::getMessage() const
{
    return mStream.str();
}

#if defined(ANGLE_PLATFORM_WINDOWS)
priv::FmtHexHelper<HRESULT> FmtHR(HRESULT value)
{
    return priv::FmtHexHelper<HRESULT>("HRESULT: ", value);
}

priv::FmtHexHelper<DWORD> FmtErr(DWORD value)
{
    return priv::FmtHexHelper<DWORD>("error: ", value);
}
#endif  // defined(ANGLE_PLATFORM_WINDOWS)

}  // namespace gl
