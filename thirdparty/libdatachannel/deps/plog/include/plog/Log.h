//////////////////////////////////////////////////////////////////////////
//  Plog - portable and simple log for C++
//  Documentation and sources: https://github.com/SergiusTheBest/plog
//  License: MIT, https://choosealicense.com/licenses/mit

#pragma once

#if defined(GODOT_PLOG_DISABLE_LOG)

#include <iostream>
#include <sstream>
#include <plog/Severity.h>

#define PLOG_GET_FUNC() ""
#define PLOG_GET_THIS() reinterpret_cast<void*>(0)
#define PLOG_GET_FILE() ""

class NullLog {
public:
	template <class T> NullLog &operator<<(const T&) { return *this; }
};

namespace plog {
	class Logger_ {
	public:
		Severity getMaxSeverity() const { return none; }
	};
	inline Logger_ *get() { return nullptr; }
}

#define PLOG_DEFAULT_INSTANCE_ID 0

#define PLOG_(inst, sev) NullLog()
#define PLOG(sev) NullLog()

#ifdef _MSC_VER
#    define IF_PLOG_(instanceId, severity)  __pragma(warning(push)) __pragma(warning(disable:4127)) if (true) {;} else __pragma(warning(pop)) // conditional expression is constant
#else
#    define IF_PLOG_(instanceId, severity)   if (true) {;} else
#endif
#define IF_PLOG(severity)                IF_PLOG_(PLOG_DEFAULT_INSTANCE_ID, severity)

#else // defined(GODOT_PLOG_DISABLE_LOG)

#include <plog/Logger.h>

//////////////////////////////////////////////////////////////////////////
// Helper macros that get context info

#if defined(PLOG_ENABLE_GET_THIS) && defined(_MSC_VER) && _MSC_VER >= 1600 && !defined(__INTELLISENSE__) && !defined(__INTEL_COMPILER) && !defined(__llvm__) && !defined(__RESHARPER__) // >= Visual Studio 2010, skip IntelliSense, Intel Compiler, Clang Code Model and ReSharper
#   define PLOG_GET_THIS()      __if_exists(this) { this } __if_not_exists(this) { 0 }
#else
#   define PLOG_GET_THIS()      reinterpret_cast<void*>(0)
#endif

#ifdef _MSC_VER
#   define PLOG_GET_FUNC()      __FUNCTION__
#elif defined(__BORLANDC__)
#   define PLOG_GET_FUNC()      __FUNC__
#else
#   define PLOG_GET_FUNC()      __PRETTY_FUNCTION__
#endif

#ifdef PLOG_CAPTURE_FILE
#   define PLOG_GET_FILE()      __FILE__
#else
#   define PLOG_GET_FILE()      ""
#endif

//////////////////////////////////////////////////////////////////////////
// Log severity level checker

#ifdef PLOG_DISABLE_LOGGING
#   ifdef _MSC_VER
#       define IF_PLOG_(instanceId, severity)  __pragma(warning(push)) __pragma(warning(disable:4127)) if (true) {;} else __pragma(warning(pop)) // conditional expression is constant
#   else
#       define IF_PLOG_(instanceId, severity)   if (true) {;} else
#   endif
#else
#   define IF_PLOG_(instanceId, severity)   if (!plog::get<instanceId>() || !plog::get<instanceId>()->checkSeverity(severity)) {;} else
#endif

#define IF_PLOG(severity)                IF_PLOG_(PLOG_DEFAULT_INSTANCE_ID, severity)

//////////////////////////////////////////////////////////////////////////
// Main logging macros

#define PLOG_(instanceId, severity)      IF_PLOG_(instanceId, severity) (*plog::get<instanceId>()) += plog::Record(severity, PLOG_GET_FUNC(), __LINE__, PLOG_GET_FILE(), PLOG_GET_THIS(), instanceId).ref()
#define PLOG(severity)                   PLOG_(PLOG_DEFAULT_INSTANCE_ID, severity)

#endif // !defined(GODOT_PLOG_DISABLE_LOG)

#define PLOG_VERBOSE                     PLOG(plog::verbose)
#define PLOG_DEBUG                       PLOG(plog::debug)
#define PLOG_INFO                        PLOG(plog::info)
#define PLOG_WARNING                     PLOG(plog::warning)
#define PLOG_ERROR                       PLOG(plog::error)
#define PLOG_FATAL                       PLOG(plog::fatal)
#define PLOG_NONE                        PLOG(plog::none)

#define PLOG_VERBOSE_(instanceId)        PLOG_(instanceId, plog::verbose)
#define PLOG_DEBUG_(instanceId)          PLOG_(instanceId, plog::debug)
#define PLOG_INFO_(instanceId)           PLOG_(instanceId, plog::info)
#define PLOG_WARNING_(instanceId)        PLOG_(instanceId, plog::warning)
#define PLOG_ERROR_(instanceId)          PLOG_(instanceId, plog::error)
#define PLOG_FATAL_(instanceId)          PLOG_(instanceId, plog::fatal)
#define PLOG_NONE_(instanceId)           PLOG_(instanceId, plog::none)

#define PLOGV                            PLOG_VERBOSE
#define PLOGD                            PLOG_DEBUG
#define PLOGI                            PLOG_INFO
#define PLOGW                            PLOG_WARNING
#define PLOGE                            PLOG_ERROR
#define PLOGF                            PLOG_FATAL
#define PLOGN                            PLOG_NONE

#define PLOGV_(instanceId)               PLOG_VERBOSE_(instanceId)
#define PLOGD_(instanceId)               PLOG_DEBUG_(instanceId)
#define PLOGI_(instanceId)               PLOG_INFO_(instanceId)
#define PLOGW_(instanceId)               PLOG_WARNING_(instanceId)
#define PLOGE_(instanceId)               PLOG_ERROR_(instanceId)
#define PLOGF_(instanceId)               PLOG_FATAL_(instanceId)
#define PLOGN_(instanceId)               PLOG_NONE_(instanceId)

//////////////////////////////////////////////////////////////////////////
// Conditional logging macros

#define PLOG_IF_(instanceId, severity, condition)  if (!(condition)) {;} else PLOG_(instanceId, severity)
#define PLOG_IF(severity, condition)               PLOG_IF_(PLOG_DEFAULT_INSTANCE_ID, severity, condition)

#define PLOG_VERBOSE_IF(condition)               PLOG_IF(plog::verbose, condition)
#define PLOG_DEBUG_IF(condition)                 PLOG_IF(plog::debug, condition)
#define PLOG_INFO_IF(condition)                  PLOG_IF(plog::info, condition)
#define PLOG_WARNING_IF(condition)               PLOG_IF(plog::warning, condition)
#define PLOG_ERROR_IF(condition)                 PLOG_IF(plog::error, condition)
#define PLOG_FATAL_IF(condition)                 PLOG_IF(plog::fatal, condition)
#define PLOG_NONE_IF(condition)                  PLOG_IF(plog::none, condition)

#define PLOG_VERBOSE_IF_(instanceId, condition)  PLOG_IF_(instanceId, plog::verbose, condition)
#define PLOG_DEBUG_IF_(instanceId, condition)    PLOG_IF_(instanceId, plog::debug, condition)
#define PLOG_INFO_IF_(instanceId, condition)     PLOG_IF_(instanceId, plog::info, condition)
#define PLOG_WARNING_IF_(instanceId, condition)  PLOG_IF_(instanceId, plog::warning, condition)
#define PLOG_ERROR_IF_(instanceId, condition)    PLOG_IF_(instanceId, plog::error, condition)
#define PLOG_FATAL_IF_(instanceId, condition)    PLOG_IF_(instanceId, plog::fatal, condition)
#define PLOG_NONE_IF_(instanceId, condition)     PLOG_IF_(instanceId, plog::none, condition)

#define PLOGV_IF(condition)                      PLOG_VERBOSE_IF(condition)
#define PLOGD_IF(condition)                      PLOG_DEBUG_IF(condition)
#define PLOGI_IF(condition)                      PLOG_INFO_IF(condition)
#define PLOGW_IF(condition)                      PLOG_WARNING_IF(condition)
#define PLOGE_IF(condition)                      PLOG_ERROR_IF(condition)
#define PLOGF_IF(condition)                      PLOG_FATAL_IF(condition)
#define PLOGN_IF(condition)                      PLOG_NONE_IF(condition)

#define PLOGV_IF_(instanceId, condition)         PLOG_VERBOSE_IF_(instanceId, condition)
#define PLOGD_IF_(instanceId, condition)         PLOG_DEBUG_IF_(instanceId, condition)
#define PLOGI_IF_(instanceId, condition)         PLOG_INFO_IF_(instanceId, condition)
#define PLOGW_IF_(instanceId, condition)         PLOG_WARNING_IF_(instanceId, condition)
#define PLOGE_IF_(instanceId, condition)         PLOG_ERROR_IF_(instanceId, condition)
#define PLOGF_IF_(instanceId, condition)         PLOG_FATAL_IF_(instanceId, condition)
#define PLOGN_IF_(instanceId, condition)         PLOG_NONE_IF_(instanceId, condition)

// Old macro names for downward compatibility. To bypass including these macro names, add
// #define PLOG_OMIT_LOG_DEFINES before #include <plog/Log.h>
#ifndef PLOG_OMIT_LOG_DEFINES

//////////////////////////////////////////////////////////////////////////
// Main logging macros - can be changed later to point at macros for a different logging package

#define LOG_(instanceId, severity)      IF_PLOG_(instanceId, severity) (*plog::get<instanceId>()) += plog::Record(severity, PLOG_GET_FUNC(), __LINE__, PLOG_GET_FILE(), PLOG_GET_THIS(), instanceId).ref()
#define LOG(severity)                   PLOG_(PLOG_DEFAULT_INSTANCE_ID, severity)

#define LOG_VERBOSE                     PLOG(plog::verbose)
#define LOG_DEBUG                       PLOG(plog::debug)
#define LOG_INFO                        PLOG(plog::info)
#define LOG_WARNING                     PLOG(plog::warning)
#define LOG_ERROR                       PLOG(plog::error)
#define LOG_FATAL                       PLOG(plog::fatal)
#define LOG_NONE                        PLOG(plog::none)

#define LOG_VERBOSE_(instanceId)        PLOG_(instanceId, plog::verbose)
#define LOG_DEBUG_(instanceId)          PLOG_(instanceId, plog::debug)
#define LOG_INFO_(instanceId)           PLOG_(instanceId, plog::info)
#define LOG_WARNING_(instanceId)        PLOG_(instanceId, plog::warning)
#define LOG_ERROR_(instanceId)          PLOG_(instanceId, plog::error)
#define LOG_FATAL_(instanceId)          PLOG_(instanceId, plog::fatal)
#define LOG_NONE_(instanceId)           PLOG_(instanceId, plog::none)

#define LOGV                            PLOG_VERBOSE
#define LOGD                            PLOG_DEBUG
#define LOGI                            PLOG_INFO
#define LOGW                            PLOG_WARNING
#define LOGE                            PLOG_ERROR
#define LOGF                            PLOG_FATAL
#define LOGN                            PLOG_NONE

#define LOGV_(instanceId)               PLOG_VERBOSE_(instanceId)
#define LOGD_(instanceId)               PLOG_DEBUG_(instanceId)
#define LOGI_(instanceId)               PLOG_INFO_(instanceId)
#define LOGW_(instanceId)               PLOG_WARNING_(instanceId)
#define LOGE_(instanceId)               PLOG_ERROR_(instanceId)
#define LOGF_(instanceId)               PLOG_FATAL_(instanceId)
#define LOGN_(instanceId)               PLOG_NONE_(instanceId)

//////////////////////////////////////////////////////////////////////////
// Conditional logging macros

#define LOG_IF_(instanceId, severity, condition)  if (!(condition)) {;} else PLOG_(instanceId, severity)
#define LOG_IF(severity, condition)               PLOG_IF_(PLOG_DEFAULT_INSTANCE_ID, severity, condition)

#define LOG_VERBOSE_IF(condition)               PLOG_IF(plog::verbose, condition)
#define LOG_DEBUG_IF(condition)                 PLOG_IF(plog::debug, condition)
#define LOG_INFO_IF(condition)                  PLOG_IF(plog::info, condition)
#define LOG_WARNING_IF(condition)               PLOG_IF(plog::warning, condition)
#define LOG_ERROR_IF(condition)                 PLOG_IF(plog::error, condition)
#define LOG_FATAL_IF(condition)                 PLOG_IF(plog::fatal, condition)
#define LOG_NONE_IF(condition)                  PLOG_IF(plog::none, condition)

#define LOG_VERBOSE_IF_(instanceId, condition)  PLOG_IF_(instanceId, plog::verbose, condition)
#define LOG_DEBUG_IF_(instanceId, condition)    PLOG_IF_(instanceId, plog::debug, condition)
#define LOG_INFO_IF_(instanceId, condition)     PLOG_IF_(instanceId, plog::info, condition)
#define LOG_WARNING_IF_(instanceId, condition)  PLOG_IF_(instanceId, plog::warning, condition)
#define LOG_ERROR_IF_(instanceId, condition)    PLOG_IF_(instanceId, plog::error, condition)
#define LOG_FATAL_IF_(instanceId, condition)    PLOG_IF_(instanceId, plog::fatal, condition)
#define LOG_NONE_IF_(instanceId, condition)     PLOG_IF_(instanceId, plog::none, condition)

#define LOGV_IF(condition)                      PLOG_VERBOSE_IF(condition)
#define LOGD_IF(condition)                      PLOG_DEBUG_IF(condition)
#define LOGI_IF(condition)                      PLOG_INFO_IF(condition)
#define LOGW_IF(condition)                      PLOG_WARNING_IF(condition)
#define LOGE_IF(condition)                      PLOG_ERROR_IF(condition)
#define LOGF_IF(condition)                      PLOG_FATAL_IF(condition)
#define LOGN_IF(condition)                      PLOG_NONE_IF(condition)

#define LOGV_IF_(instanceId, condition)         PLOG_VERBOSE_IF_(instanceId, condition)
#define LOGD_IF_(instanceId, condition)         PLOG_DEBUG_IF_(instanceId, condition)
#define LOGI_IF_(instanceId, condition)         PLOG_INFO_IF_(instanceId, condition)
#define LOGW_IF_(instanceId, condition)         PLOG_WARNING_IF_(instanceId, condition)
#define LOGE_IF_(instanceId, condition)         PLOG_ERROR_IF_(instanceId, condition)
#define LOGF_IF_(instanceId, condition)         PLOG_FATAL_IF_(instanceId, condition)
#define LOGN_IF_(instanceId, condition)         PLOG_NONE_IF_(instanceId, condition)
#endif
