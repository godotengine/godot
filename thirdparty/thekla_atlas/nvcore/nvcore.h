// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_CORE_H
#define NV_CORE_H

// Function linkage
#if NVCORE_SHARED
#ifdef NVCORE_EXPORTS
#define NVCORE_API DLL_EXPORT
#define NVCORE_CLASS DLL_EXPORT_CLASS
#else
#define NVCORE_API DLL_IMPORT
#define NVCORE_CLASS DLL_IMPORT
#endif
#else // NVCORE_SHARED
#define NVCORE_API
#define NVCORE_CLASS
#endif // NVCORE_SHARED


// Platform definitions
#include <posh.h>

// OS:
// NV_OS_WIN32
// NV_OS_WIN64
// NV_OS_MINGW
// NV_OS_CYGWIN
// NV_OS_LINUX
// NV_OS_UNIX
// NV_OS_DARWIN
// NV_OS_XBOX
// NV_OS_ORBIS
// NV_OS_IOS

#define NV_OS_STRING POSH_OS_STRING

#if defined POSH_OS_LINUX
#   define NV_OS_LINUX 1
#   define NV_OS_UNIX 1
#elif defined POSH_OS_ORBIS
#   define NV_OS_ORBIS 1
#elif defined POSH_OS_FREEBSD
#   define NV_OS_FREEBSD 1
#   define NV_OS_UNIX 1
#elif defined POSH_OS_OPENBSD
#   define NV_OS_OPENBSD 1
#   define NV_OS_UNIX 1
#elif defined POSH_OS_CYGWIN32
#   define NV_OS_CYGWIN 1
#elif defined POSH_OS_MINGW
#   define NV_OS_MINGW 1
#   define NV_OS_WIN32 1
#elif defined POSH_OS_OSX
#   define NV_OS_OSX 1      // IC: Adding this, because iOS defines NV_OS_DARWIN too.
#   define NV_OS_DARWIN 1
#   define NV_OS_UNIX 1
#elif defined POSH_OS_IOS
#   define NV_OS_DARWIN 1 //ACS should we keep this on IOS?
#   define NV_OS_UNIX 1
#   define NV_OS_IOS 1
#elif defined POSH_OS_UNIX
#   define NV_OS_UNIX 1
#elif defined POSH_OS_WIN64
#   define NV_OS_WIN32 1
#   define NV_OS_WIN64 1
#elif defined POSH_OS_WIN32
#   define NV_OS_WIN32 1
#elif defined POSH_OS_XBOX
#   define NV_OS_XBOX 1
#elif defined POSH_OS_DURANGO
#   define NV_OS_DURANGO 1
#else
#   error "Unsupported OS"
#endif


// Is this a console OS? (i.e. connected to a TV)
#if NV_OS_ORBIS || NV_OS_XBOX || NV_OS_DURANGO
#   define NV_OS_CONSOLE 1
#endif 


// Threading:
// some platforms don't implement __thread or similar for thread-local-storage
#if NV_OS_UNIX || NV_OS_ORBIS || NV_OS_IOS //ACStodoIOS darwin instead of ios?
#   define NV_OS_USE_PTHREAD 1
#   if NV_OS_IOS
#       define NV_OS_HAS_TLS_QUALIFIER 0
#   else
#       define NV_OS_HAS_TLS_QUALIFIER 1
#   endif
#else
#   define NV_OS_USE_PTHREAD 0
#   define NV_OS_HAS_TLS_QUALIFIER 1
#endif


// CPUs:
// NV_CPU_X86
// NV_CPU_X86_64
// NV_CPU_PPC
// NV_CPU_ARM

#define NV_CPU_STRING   POSH_CPU_STRING

#if defined POSH_CPU_X86_64
//#   define NV_CPU_X86 1
#   define NV_CPU_X86_64 1
#elif defined POSH_CPU_X86
#   define NV_CPU_X86 1
#elif defined POSH_CPU_PPC
#   define NV_CPU_PPC 1
#elif defined POSH_CPU_STRONGARM
#   define NV_CPU_ARM 1
#else
#   error "Unsupported CPU"
#endif


// Compiler:
// NV_CC_GNUC
// NV_CC_MSVC
// NV_CC_CLANG

#if defined POSH_COMPILER_CLANG
#   define NV_CC_CLANG  1
#   define NV_CC_GNUC   1    // Clang is compatible with GCC.
#   define NV_CC_STRING "clang"
#elif defined POSH_COMPILER_GCC
#   define NV_CC_GNUC   1
#   define NV_CC_STRING "gcc"
#elif defined POSH_COMPILER_MSVC
#   define NV_CC_MSVC   1
#   define NV_CC_STRING "msvc"
#else
#   error "Unsupported compiler"
#endif

#if NV_CC_MSVC
#define NV_CC_CPP11 (__cplusplus > 199711L || _MSC_VER >= 1800) // Visual Studio 2013 has all the features we use, but doesn't advertise full C++11 support yet.
#else
// @@ IC: This works in CLANG, about GCC?
// @@ ES: Doesn't work in gcc. These 3 features are available in GCC >= 4.4.
#ifdef __clang__
#define NV_CC_CPP11 (__has_feature(cxx_deleted_functions) && __has_feature(cxx_rvalue_references) && __has_feature(cxx_static_assert))
#elif defined __GNUC__ 
#define NV_CC_CPP11 ( __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 4))
#endif
#endif

// Endiannes:
#define NV_LITTLE_ENDIAN    POSH_LITTLE_ENDIAN
#define NV_BIG_ENDIAN       POSH_BIG_ENDIAN
#define NV_ENDIAN_STRING    POSH_ENDIAN_STRING


// Define the right printf prefix for size_t arguments:
#if POSH_64BIT_POINTER
#  define NV_SIZET_PRINTF_PREFIX POSH_I64_PRINTF_PREFIX
#else
#  define NV_SIZET_PRINTF_PREFIX
#endif


// cmake config
#include "nvconfig.h"

#if NV_OS_DARWIN
#include <stdint.h>
//#include <inttypes.h>

// Type definitions:
typedef uint8_t     uint8;
typedef int8_t      int8;

typedef uint16_t    uint16;
typedef int16_t     int16;

typedef uint32_t    uint32;
typedef int32_t     int32;

typedef uint64_t    uint64;
typedef int64_t     int64;

// POSH gets this wrong due to __LP64__
#undef POSH_I64_PRINTF_PREFIX
#define POSH_I64_PRINTF_PREFIX "ll"

#else

// Type definitions:
typedef posh_u8_t   uint8;
typedef posh_i8_t   int8;

typedef posh_u16_t  uint16;
typedef posh_i16_t  int16;

typedef posh_u32_t  uint32;
typedef posh_i32_t  int32;

//#if NV_OS_DARWIN
// OSX-64 is supposed to be LP64 (longs and pointers are 64 bits), thus uint64 is defined as 
// unsigned long. However, some OSX headers define it as unsigned long long, producing errors,
// even though both types are 64 bit. Ideally posh should handle that, but it has not been
// updated in ages, so here I'm just falling back to the standard C99 types defined in inttypes.h
//#include <inttypes.h>
//typedef posh_u64_t  uint64_t;
//typedef posh_i64_t  int64_t;
//#else
typedef posh_u64_t  uint64;
typedef posh_i64_t  int64;
//#endif
#if NV_OS_DARWIN
// To avoid duplicate definitions.
#define _UINT64
#endif
#endif

// Aliases
typedef uint32      uint;


// Version string:
#define NV_VERSION_STRING \
    NV_OS_STRING "/" NV_CC_STRING "/" NV_CPU_STRING"/" \
    NV_ENDIAN_STRING"-endian - " __DATE__ "-" __TIME__


// Disable copy constructor and assignment operator. 
#if NV_CC_CPP11
#define NV_FORBID_COPY(C) \
    C( const C & ) = delete; \
    C &operator=( const C & ) = delete
#else
#define NV_FORBID_COPY(C) \
    private: \
    C( const C & ); \
    C &operator=( const C & )
#endif

// Disable dynamic allocation on the heap. 
// See Prohibiting Heap-Based Objects in More Effective C++.
#define NV_FORBID_HEAPALLOC() \
    private: \
    void *operator new(size_t size); \
    void *operator new[](size_t size)
    //static void *operator new(size_t size); \
    //static void *operator new[](size_t size);

// String concatenation macros.
#define NV_STRING_JOIN2(arg1, arg2) NV_DO_STRING_JOIN2(arg1, arg2)
#define NV_DO_STRING_JOIN2(arg1, arg2) arg1 ## arg2
#define NV_STRING_JOIN3(arg1, arg2, arg3) NV_DO_STRING_JOIN3(arg1, arg2, arg3)
#define NV_DO_STRING_JOIN3(arg1, arg2, arg3) arg1 ## arg2 ## arg3
#define NV_STRING2(x) #x
#define NV_STRING(x) NV_STRING2(x)

#if NV_CC_MSVC
#define NV_MULTI_LINE_MACRO_BEGIN do {  
#define NV_MULTI_LINE_MACRO_END \
    __pragma(warning(push)) \
    __pragma(warning(disable:4127)) \
    } while(false) \
    __pragma(warning(pop))  
#else
#define NV_MULTI_LINE_MACRO_BEGIN do {
#define NV_MULTI_LINE_MACRO_END } while(false)
#endif

#if NV_CC_CPP11
#define nvStaticCheck(x) static_assert((x), "Static assert "#x" failed")
#else
#define nvStaticCheck(x) typedef char NV_STRING_JOIN2(__static_assert_,__LINE__)[(x)]
#endif
#define NV_COMPILER_CHECK(x) nvStaticCheck(x)   // I like this name best.

// Make sure type definitions are fine.
NV_COMPILER_CHECK(sizeof(int8) == 1);
NV_COMPILER_CHECK(sizeof(uint8) == 1);
NV_COMPILER_CHECK(sizeof(int16) == 2);
NV_COMPILER_CHECK(sizeof(uint16) == 2);
NV_COMPILER_CHECK(sizeof(int32) == 4);
NV_COMPILER_CHECK(sizeof(uint32) == 4);
NV_COMPILER_CHECK(sizeof(int32) == 4);
NV_COMPILER_CHECK(sizeof(uint32) == 4);

#include <stddef.h> // for size_t
template <typename T, size_t N> char (&ArraySizeHelper(T (&array)[N]))[N];
#define NV_ARRAY_SIZE(x) sizeof(ArraySizeHelper(x))
//#define NV_ARRAY_SIZE(x) (sizeof(x)/sizeof((x)[0]))

#if 0 // Disabled in The Witness.
#if NV_CC_MSVC
#define NV_MESSAGE(x) message(__FILE__ "(" NV_STRING(__LINE__) ") : " x)
#else
#define NV_MESSAGE(x) message(x)
#endif
#else
#define NV_MESSAGE(x) 
#endif


// Startup initialization macro.
#define NV_AT_STARTUP(some_code) \
    namespace { \
        static struct NV_STRING_JOIN2(AtStartup_, __LINE__) { \
            NV_STRING_JOIN2(AtStartup_, __LINE__)() { some_code; } \
        } \
        NV_STRING_JOIN3(AtStartup_, __LINE__, Instance); \
    }

// Indicate the compiler that the parameter is not used to suppress compier warnings.
#if NV_CC_MSVC
#define NV_UNUSED(a) ((a)=(a))
#else
#define NV_UNUSED(a) _Pragma(NV_STRING(unused(a)))
#endif

// Null index. @@ Move this somewhere else... it's only used by nvmesh.
//const unsigned int NIL = unsigned int(~0);
#define NIL uint(~0)

// Null pointer.
#ifndef NULL
#define NULL 0
#endif

// Platform includes
#if NV_CC_MSVC
#   if NV_OS_WIN32
#       include "DefsVcWin32.h"
#   elif NV_OS_XBOX
#       include "DefsVcXBox.h"
#   elif NV_OS_DURANGO
#       include "DefsVcDurango.h"
#   else
#       error "MSVC: Platform not supported"
#   endif
#elif NV_CC_GNUC
#   if NV_OS_LINUX
#       include "DefsGnucLinux.h"
#   elif NV_OS_DARWIN || NV_OS_FREEBSD || NV_OS_OPENBSD
#       include "DefsGnucDarwin.h"
#   elif NV_OS_ORBIS
#       include "DefsOrbis.h"
#   elif NV_OS_MINGW
#       include "DefsGnucWin32.h"
#   elif NV_OS_CYGWIN
#       error "GCC: Cygwin not supported"
#   else
#       error "GCC: Platform not supported"
#   endif
#endif

#endif // NV_CORE_H
