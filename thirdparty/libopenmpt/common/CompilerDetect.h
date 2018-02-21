/*
 * CompilerDetect.h
 * ----------------
 * Purpose: Detect current compiler and provide readable version test macros.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once



#define MPT_COMPILER_MAKE_VERSION2(version,sp)              ((version) * 100 + (sp))
#define MPT_COMPILER_MAKE_VERSION3(major,minor,patch)       ((major) * 10000 + (minor) * 100 + (patch))
#define MPT_COMPILER_MAKE_VERSION3_BUILD(major,minor,build) ((major) * 10000000 + (minor) * 100000 + (patch))



#if defined(MPT_COMPILER_GENERIC)

#undef MPT_COMPILER_GENERIC
#define MPT_COMPILER_GENERIC                         1

#elif defined(__clang__) && defined(_MSC_VER) && defined(__c2__)

#define MPT_COMPILER_MSVCCLANGC2                     1
#define MPT_COMPILER_MSVCCLANGC2_VERSION             (__c2_version__)
#define MPT_MSVCCLANGC2_AT_LEAST(major,minor,build)  (MPT_COMPILER_MSVCCLANGC2_VERSION >= MPT_COMPILER_MAKE_VERSION3_BUILD((major),(minor),(build)))
#define MPT_MSVCCLANGC2_BEFORE(major,minor,build)    (MPT_COMPILER_MSVCCLANGC2_VERSION <  MPT_COMPILER_MAKE_VERSION3_BUILD((major),(minor),(build)))

#elif defined(__clang__)

#define MPT_COMPILER_CLANG                           1
#define MPT_COMPILER_CLANG_VERSION                   MPT_COMPILER_MAKE_VERSION3(__clang_major__,__clang_minor__,__clang_patchlevel__)
#define MPT_CLANG_AT_LEAST(major,minor,patch)        (MPT_COMPILER_CLANG_VERSION >= MPT_COMPILER_MAKE_VERSION3((major),(minor),(patch)))
#define MPT_CLANG_BEFORE(major,minor,patch)          (MPT_COMPILER_CLANG_VERSION <  MPT_COMPILER_MAKE_VERSION3((major),(minor),(patch)))

#if MPT_CLANG_BEFORE(3,4,0)
#error "clang version 3.4 required"
#endif

#if defined(__clang_analyzer__) 
#ifndef MPT_BUILD_ANALYZED
#define MPT_BUILD_ANALYZED
#endif
#endif

#elif defined(__GNUC__)

#define MPT_COMPILER_GCC                             1
#define MPT_COMPILER_GCC_VERSION                     MPT_COMPILER_MAKE_VERSION3(__GNUC__,__GNUC_MINOR__,__GNUC_PATCHLEVEL__)
#define MPT_GCC_AT_LEAST(major,minor,patch)          (MPT_COMPILER_GCC_VERSION >= MPT_COMPILER_MAKE_VERSION3((major),(minor),(patch)))
#define MPT_GCC_BEFORE(major,minor,patch)            (MPT_COMPILER_GCC_VERSION <  MPT_COMPILER_MAKE_VERSION3((major),(minor),(patch)))

#if MPT_GCC_BEFORE(4,8,0)
#error "GCC version 4.8 required"
#endif

#elif defined(_MSC_VER)

#define MPT_COMPILER_MSVC                            1
#if (_MSC_VER >= 1912)
#define MPT_COMPILER_MSVC_VERSION                    MPT_COMPILER_MAKE_VERSION2(2017,5)
#elif (_MSC_VER >= 1911)
#define MPT_COMPILER_MSVC_VERSION                    MPT_COMPILER_MAKE_VERSION2(2017,3)
#elif (_MSC_VER >= 1910)
#define MPT_COMPILER_MSVC_VERSION                    MPT_COMPILER_MAKE_VERSION2(2017,0)
#elif (_MSC_VER >= 1900)
#define MPT_COMPILER_MSVC_VERSION                    MPT_COMPILER_MAKE_VERSION2(2015,0)
#elif (_MSC_VER >= 1800)
#define MPT_COMPILER_MSVC_VERSION                    MPT_COMPILER_MAKE_VERSION2(2013,0)
#elif (_MSC_VER >= 1700)
#define MPT_COMPILER_MSVC_VERSION                    MPT_COMPILER_MAKE_VERSION2(2012,0)
#elif (_MSC_VER >= 1600)
#define MPT_COMPILER_MSVC_VERSION                    MPT_COMPILER_MAKE_VERSION2(2010,0)
#elif (_MSC_VER >= 1500)
#define MPT_COMPILER_MSVC_VERSION                    MPT_COMPILER_MAKE_VERSION2(2008,0)
#else
#define MPT_COMPILER_MSVC_VERSION                    MPT_COMPILER_MAKE_VERSION2(2005,0)
#endif
#define MPT_MSVC_AT_LEAST(version,sp)                (MPT_COMPILER_MSVC_VERSION >= MPT_COMPILER_MAKE_VERSION2((version),(sp)))
#define MPT_MSVC_BEFORE(version,sp)                  (MPT_COMPILER_MSVC_VERSION <  MPT_COMPILER_MAKE_VERSION2((version),(sp)))

#if MPT_MSVC_BEFORE(2015,0)
#error "MSVC version 2015 required"
#endif

#if defined(_PREFAST_)
#ifndef MPT_BUILD_ANALYZED
#define MPT_BUILD_ANALYZED
#endif
#endif

#else

#define MPT_COMPILER_GENERIC                         1

#endif



#ifndef MPT_COMPILER_GENERIC
#define MPT_COMPILER_GENERIC                  0
#endif
#ifndef MPT_COMPILER_MSVCCLANGC2
#define MPT_COMPILER_MSVCCLANGC2                    0
#define MPT_MSVCCLANGC2_AT_LEAST(major,minor,build) 0
#define MPT_MSVCCLANGC2_BEFORE(major,minor,build)   0
#endif
#ifndef MPT_COMPILER_CLANG
#define MPT_COMPILER_CLANG                    0
#define MPT_CLANG_AT_LEAST(major,minor,patch) 0
#define MPT_CLANG_BEFORE(major,minor,patch)   0
#endif
#ifndef MPT_COMPILER_GCC
#define MPT_COMPILER_GCC                      0
#define MPT_GCC_AT_LEAST(major,minor,patch)   0
#define MPT_GCC_BEFORE(major,minor,patch)     0
#endif
#ifndef MPT_COMPILER_MSVC
#define MPT_COMPILER_MSVC                     0
#define MPT_MSVC_AT_LEAST(version,sp)         0
#define MPT_MSVC_BEFORE(version,sp)           0
#endif



#if MPT_COMPILER_GENERIC || MPT_COMPILER_GCC || MPT_COMPILER_CLANG

#if (__cplusplus >= 201703)
#define MPT_CXX 17
#elif (__cplusplus >= 201402)
#define MPT_CXX 14
#else
#define MPT_CXX 11
#endif

#elif MPT_COMPILER_MSVC

#if MPT_MSVC_AT_LEAST(2017,0)
#if (_MSVC_LANG >= 201402)
#define MPT_CXX 14
#else
#define MPT_CXX 11
#endif
#else
#define MPT_CXX 11
#endif

#else

#define MPT_CXX 11

#endif

// MPT_CXX is stricter than just using __cplusplus directly.
// We will only claim a language version as supported IFF all core language and
// library fatures that we need are actually supported AND working correctly
// (to our needs).

#define MPT_CXX_AT_LEAST(version) (MPT_CXX >= (version))
#define MPT_CXX_BEFORE(version)   (MPT_CXX <  (version))



#if MPT_COMPILER_MSVC
	#define MPT_PLATFORM_LITTLE_ENDIAN
#elif MPT_COMPILER_GCC
	#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
		#define MPT_PLATFORM_BIG_ENDIAN
	#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
		#define MPT_PLATFORM_LITTLE_ENDIAN
	#endif
#elif MPT_COMPILER_CLANG || MPT_COMPILER_MSVCCLANGC2
	#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
		#define MPT_PLATFORM_BIG_ENDIAN
	#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
		#define MPT_PLATFORM_LITTLE_ENDIAN
	#endif
#endif

// fallback:
#if !defined(MPT_PLATFORM_BIG_ENDIAN) && !defined(MPT_PLATFORM_LITTLE_ENDIAN)
	// taken from boost/detail/endian.hpp
	#if (defined(_BIG_ENDIAN) && !defined(_LITTLE_ENDIAN)) \
		|| (defined(__BIG_ENDIAN__) && !defined(__LITTLE_ENDIAN__)) \
		|| (defined(_STLP_BIG_ENDIAN) && !defined(_STLP_LITTLE_ENDIAN))
			#define MPT_PLATFORM_BIG_ENDIAN
	#elif (defined(_LITTLE_ENDIAN) && !defined(_BIG_ENDIAN)) \
		|| (defined(__LITTLE_ENDIAN__) && !defined(__BIG_ENDIAN__)) \
		|| (defined(_STLP_LITTLE_ENDIAN) && !defined(_STLP_BIG_ENDIAN))
			#define MPT_PLATFORM_LITTLE_ENDIAN
	#elif defined(__sparc) || defined(__sparc__) \
		|| defined(_POWER) || defined(__powerpc__) \
		|| defined(__ppc__) || defined(__hpux) || defined(__hppa) \
		|| defined(_MIPSEB) || defined(_POWER) \
		|| defined(__s390__)
			#define MPT_PLATFORM_BIG_ENDIAN
	#elif defined(__i386__) || defined(__alpha__) \
		|| defined(__ia64) || defined(__ia64__) \
		|| defined(_M_IX86) || defined(_M_IA64) \
		|| defined(_M_ALPHA) || defined(__amd64) \
		|| defined(__amd64__) || defined(_M_AMD64) \
		|| defined(__x86_64) || defined(__x86_64__) \
		|| defined(_M_X64) || defined(__bfin__)
			#define MPT_PLATFORM_LITTLE_ENDIAN
	#endif
#endif

#if defined(MPT_PLATFORM_BIG_ENDIAN) || defined(MPT_PLATFORM_LITTLE_ENDIAN)
#define MPT_PLATFORM_ENDIAN_KNOWN 1
#else
#define MPT_PLATFORM_ENDIAN_KNOWN 0
#endif



// This should really be based on __STDCPP_THREADS__, but that is not defined by
// GCC or clang. Stupid.
// Just assume multithreaded and disable for platforms we know are
// singlethreaded later on.
#define MPT_PLATFORM_MULTITHREADED 1



// specific C++ features



// C++11 constexpr

#if MPT_COMPILER_MSVC
#define MPT_COMPILER_QUIRK_CONSTEXPR_NO_STRING_LITERALS
#endif



#if MPT_COMPILER_MSVC
// Compiler has multiplication/division semantics when shifting signed integers.
#define MPT_COMPILER_SHIFT_SIGNED 1
#endif

#ifndef MPT_COMPILER_SHIFT_SIGNED
#define MPT_COMPILER_SHIFT_SIGNED 0
#endif



#if MPT_COMPILER_GCC || MPT_COMPILER_MSVC
// Compiler supports type-punning through unions. This is not stricly standard-conforming.
// For GCC, this is documented, for MSVC this is apparently not documented, but we assume it.
#define MPT_COMPILER_UNION_TYPE_ALIASES 1
#endif

#ifndef MPT_COMPILER_UNION_TYPE_ALIASES
// Compiler does not support type-punning through unions. std::memcpy is used instead.
// This is the safe fallback and strictly standard-conforming.
// Another standard-compliant alternative would be casting pointers to a character type pointer.
// This results in rather unreadable code and,
// in most cases, compilers generate better code by just inlining the memcpy anyway.
// (see <http://blog.regehr.org/archives/959>).
#define MPT_COMPILER_UNION_TYPE_ALIASES 0
#endif



// The order of the checks matters!
#if defined(__EMSCRIPTEN__)
	#define MPT_OS_EMSCRIPTEN 1
	#if defined(__EMSCRIPTEN_major__) && defined(__EMSCRIPTEN_minor__)
		#if (__EMSCRIPTEN_major__ > 1)
			#define MPT_OS_EMSCRIPTEN_ANCIENT 0
		#elif (__EMSCRIPTEN_major__ == 1) && (__EMSCRIPTEN_minor__ >= 36)
			#define MPT_OS_EMSCRIPTEN_ANCIENT 0
		#else
			#define MPT_OS_EMSCRIPTEN_ANCIENT 1
		#endif
	#else
		#define MPT_OS_EMSCRIPTEN_ANCIENT 1
	#endif
#elif defined(_WIN32)
	#define MPT_OS_WINDOWS 1
	#if defined(WINAPI_FAMILY)
		#include <winapifamily.h>
		#if (WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP)
			#define MPT_OS_WINDOWS_WINRT 0
		#else
			#define MPT_OS_WINDOWS_WINRT 1
		#endif
	#else // !WINAPI_FAMILY
		#define MPT_OS_WINDOWS_WINRT 0
	#endif // WINAPI_FAMILY
#elif defined(__APPLE__)
	#define MPT_OS_MACOSX_OR_IOS 1
	//#include "TargetConditionals.h"
	//#if TARGET_IPHONE_SIMULATOR
	//#elif TARGET_OS_IPHONE
	//#elif TARGET_OS_MAC
	//#else
	//#endif
#elif defined(__ANDROID__) || defined(ANDROID)
	#define MPT_OS_ANDROID 1
#elif defined(__linux__)
	#define MPT_OS_LINUX 1
#elif defined(__DragonFly__)
	#define MPT_OS_DRAGONFLYBSD 1
#elif defined(__FreeBSD__)
	#define MPT_OS_FREEBSD 1
#elif defined(__OpenBSD__)
	#define MPT_OS_OPENBSD 1
#elif defined(__NetBSD__)
	#define MPT_OS_NETBSD 1
#elif defined(__unix__)
	#define MPT_OS_GENERIC_UNIX 1
#else
	#define MPT_OS_UNKNOWN 1
#endif

#ifndef MPT_OS_EMSCRIPTEN
#define MPT_OS_EMSCRIPTEN 0
#endif
#ifndef MPT_OS_WINDOWS
#define MPT_OS_WINDOWS 0
#endif
#ifndef MPT_OS_WINDOWS_WINRT
#define MPT_OS_WINDOWS_WINRT 0
#endif
#ifndef MPT_OS_MACOSX_OR_IOS
#define MPT_OS_MACOSX_OR_IOS 0
#endif
#ifndef MPT_OS_ANDROID
#define MPT_OS_ANDROID 0
#endif
#ifndef MPT_OS_LINUX
#define MPT_OS_LINUX 0
#endif
#ifndef MPT_OS_DRAGONFLYBSD
#define MPT_OS_DRAGONFLYBSD 0
#endif
#ifndef MPT_OS_FREEBSD
#define MPT_OS_FREEBSD 0
#endif
#ifndef MPT_OS_OPENBSD
#define MPT_OS_OPENBSD 0
#endif
#ifndef MPT_OS_NETBSD
#define MPT_OS_NETBSD 0
#endif
#ifndef MPT_OS_GENERIC_UNIX
#define MPT_OS_GENERIC_UNIX 0
#endif
#ifndef MPT_OS_UNKNOWN
#define MPT_OS_UNKNOWN 0
#endif

#ifndef MPT_OS_EMSCRIPTEN_ANCIENT
#define MPT_OS_EMSCRIPTEN_ANCIENT 0
#endif



#if MPT_OS_EMSCRIPTEN
#undef MPT_PLATFORM_MULTITHREADED
#define MPT_PLATFORM_MULTITHREADED 0
#endif
