#ifndef _C4_PLATFORM_HPP_
#define _C4_PLATFORM_HPP_

/** @file platform.hpp Provides platform information macros
 * @ingroup basic_headers */

// see also https://sourceforge.net/p/predef/wiki/OperatingSystems/

#if defined(_WIN64)
#   define C4_WIN
#   define C4_WIN64
#elif defined(_WIN32)
#   define C4_WIN
#   define C4_WIN32
#elif defined(__ANDROID__)
#   define C4_ANDROID
#elif defined(__APPLE__)
#   include "TargetConditionals.h"
#   if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
#       define C4_IOS
#   elif TARGET_OS_MAC || TARGET_OS_OSX
#       define C4_MACOS
#   else
#       error "Unknown Apple platform"
#   endif
#elif defined(__linux__) || defined(__linux)
#   define C4_UNIX
#   define C4_LINUX
#elif defined(__unix__) || defined(__unix)
#   define C4_UNIX
#elif defined(__arm__) || defined(__aarch64__)
#   define C4_ARM
#elif defined(__xtensa__) || defined(__XTENSA__)
#   define C4_XTENSA
#elif defined(SWIG)
#   define C4_SWIG
#else
#   error "unknown platform"
#endif

#if defined(__posix) || defined(C4_UNIX) || defined(C4_LINUX)
#   define C4_POSIX
#endif


#endif /* _C4_PLATFORM_HPP_ */
