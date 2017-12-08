// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#ifndef NV_CORE_H
#error "Do not include this file directly."
#endif

// Function linkage
#define DLL_IMPORT __declspec(dllimport)
#define DLL_EXPORT __declspec(dllexport)
#define DLL_EXPORT_CLASS DLL_EXPORT

// Function calling modes
#define NV_CDECL        __cdecl
#define NV_STDCALL      __stdcall
#define NV_FASTCALL     __fastcall
#define NV_DEPRECATED

#define NV_PURE
#define NV_CONST

// Set standard function names.
#if _MSC_VER < 1900
#	define snprintf _snprintf
#endif
#if _MSC_VER < 1500
#   define vsnprintf _vsnprintf
#endif
#if _MSC_VER < 1700
#   define strtoll _strtoi64
#   define strtoull _strtoui64
#endif
//#define chdir _chdir
#define getcwd _getcwd 

#if _MSC_VER <= 1600
#define va_copy(a, b) (a) = (b)
#endif

#if !defined restrict
#define restrict
#endif

// Ignore gcc attributes.
#define __attribute__(X)

#if !defined __FUNC__
#define __FUNC__ __FUNCTION__ 
#endif

#define NV_NOINLINE __declspec(noinline)
#define NV_FORCEINLINE __forceinline

#define NV_THREAD_LOCAL __declspec(thread)

/*
// Type definitions
typedef unsigned char       uint8;
typedef signed char         int8;

typedef unsigned short      uint16;
typedef signed short        int16;

typedef unsigned int        uint32;
typedef signed int          int32;

typedef unsigned __int64    uint64;
typedef signed __int64      int64;

// Aliases
typedef uint32              uint;
*/

// Unwanted VC++ warnings to disable.
/*
#pragma warning(disable : 4244)     // conversion to float, possible loss of data
#pragma warning(disable : 4245)     // conversion from 'enum ' to 'unsigned long', signed/unsigned mismatch
#pragma warning(disable : 4100)     // unreferenced formal parameter
#pragma warning(disable : 4514)     // unreferenced inline function has been removed
#pragma warning(disable : 4710)     // inline function not expanded
#pragma warning(disable : 4127)     // Conditional expression is constant
#pragma warning(disable : 4305)     // truncation from 'const double' to 'float'
#pragma warning(disable : 4505)     // unreferenced local function has been removed

#pragma warning(disable : 4702)     // unreachable code in inline expanded function
#pragma warning(disable : 4711)     // function selected for automatic inlining
#pragma warning(disable : 4725)     // Pentium fdiv bug

#pragma warning(disable : 4786)     // Identifier was truncated and cannot be debugged.

#pragma warning(disable : 4675)     // resolved overload was found by argument-dependent lookup
*/

#pragma warning(1 : 4705)     // Report unused local variables.
#pragma warning(1 : 4555)     // Expression has no effect.
