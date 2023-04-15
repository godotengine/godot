// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <cstddef>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstring>
#include <stdint.h>
#include <functional>

////////////////////////////////////////////////////////////////////////////////
/// detect platform
////////////////////////////////////////////////////////////////////////////////

/* detect 32 or 64 Intel platform */
#if defined(__x86_64__) || defined(__ia64__) || defined(_M_X64)
#define __X86_64__
#define __X86_ASM__
#elif defined(__i386__) || defined(_M_IX86)
#define __X86_ASM__
#endif

/* detect 64 bit platform */
#if defined(__X86_64__) || defined(__aarch64__)
#define __64BIT__
#endif

/* detect Linux platform */
#if defined(linux) || defined(__linux__) || defined(__LINUX__)
#  if !defined(__LINUX__)
#     define __LINUX__
#  endif
#  if !defined(__UNIX__)
#     define __UNIX__
#  endif
#endif

/* detect FreeBSD platform */
#if defined(__FreeBSD__) || defined(__FREEBSD__)
#  if !defined(__FREEBSD__)
#     define __FREEBSD__
#  endif
#  if !defined(__UNIX__)
#     define __UNIX__
#  endif
#endif

/* detect Windows 95/98/NT/2000/XP/Vista/7/8/10 platform */
#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)) && !defined(__CYGWIN__)
#  if !defined(__WIN32__)
#     define __WIN32__
#  endif
#endif

/* detect Cygwin platform */
#if defined(__CYGWIN__)
#  if !defined(__UNIX__)
#     define __UNIX__
#  endif
#endif

/* detect MAC OS X platform */
#if defined(__APPLE__) || defined(MACOSX) || defined(__MACOSX__)
#  if !defined(__MACOSX__)
#     define __MACOSX__
#  endif
#  if !defined(__UNIX__)
#     define __UNIX__
#  endif
#endif

/* try to detect other Unix systems */
#if defined(__unix__) || defined (unix) || defined(__unix) || defined(_unix)
#  if !defined(__UNIX__)
#     define __UNIX__
#  endif
#endif

////////////////////////////////////////////////////////////////////////////////
/// Macros
////////////////////////////////////////////////////////////////////////////////

#ifdef __WIN32__
#  if defined(EMBREE_STATIC_LIB)
#    define dll_export
#    define dll_import
#  else
#    define dll_export __declspec(dllexport)
#    define dll_import __declspec(dllimport)
#  endif
#else
#  define dll_export __attribute__ ((visibility ("default")))
#  define dll_import
#endif

#if defined(__WIN32__) && !defined(__MINGW32__)
#if !defined(__noinline)
#define __noinline             __declspec(noinline)
#endif
//#define __forceinline        __forceinline
//#define __restrict           __restrict
#if defined(__INTEL_COMPILER)
#define __restrict__           __restrict
#else
#define __restrict__           //__restrict // causes issues with MSVC
#endif
#if !defined(__thread)
#define __thread               __declspec(thread)
#endif
#if !defined(__aligned)
#define __aligned(...)           __declspec(align(__VA_ARGS__))
#endif
//#define __FUNCTION__           __FUNCTION__
#define debugbreak()           __debugbreak()

#else
#if !defined(__noinline)
#define __noinline             __attribute__((noinline))
#endif
#if !defined(__forceinline)
#define __forceinline          inline __attribute__((always_inline))
#endif
//#define __restrict             __restrict
//#define __thread               __thread
#if !defined(__aligned)
#define __aligned(...)           __attribute__((aligned(__VA_ARGS__)))
#endif
#if !defined(__FUNCTION__)
#define __FUNCTION__           __PRETTY_FUNCTION__
#endif
#define debugbreak()           asm ("int $3")
#endif

#if defined(__clang__) || defined(__GNUC__)
  #define MAYBE_UNUSED __attribute__((unused))
#else
  #define MAYBE_UNUSED
#endif

#if defined(_MSC_VER) && (_MSC_VER < 1900) // before VS2015 deleted functions are not supported properly
  #define DELETED
#else
  #define DELETED  = delete
#endif

#if !defined(likely)
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#define   likely(expr) (expr)
#define unlikely(expr) (expr)
#else
#define   likely(expr) __builtin_expect((bool)(expr),true )
#define unlikely(expr) __builtin_expect((bool)(expr),false)
#endif
#endif

////////////////////////////////////////////////////////////////////////////////
/// Error handling and debugging
////////////////////////////////////////////////////////////////////////////////

/* debug printing macros */
#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define PING embree_cout << __FILE__ << " (" << __LINE__ << "): " << __FUNCTION__ << embree_endl
#define PRINT(x) embree_cout << STRING(x) << " = " << (x) << embree_endl
#define PRINT2(x,y) embree_cout << STRING(x) << " = " << (x) << ", " << STRING(y) << " = " << (y) << embree_endl
#define PRINT3(x,y,z) embree_cout << STRING(x) << " = " << (x) << ", " << STRING(y) << " = " << (y) << ", " << STRING(z) << " = " << (z) << embree_endl
#define PRINT4(x,y,z,w) embree_cout << STRING(x) << " = " << (x) << ", " << STRING(y) << " = " << (y) << ", " << STRING(z) << " = " << (z) << ", " << STRING(w) << " = " << (w) << embree_endl

#if defined(DEBUG) // only report file and line in debug mode
  // -- GODOT start --
  // #define THROW_RUNTIME_ERROR(str)
  //   throw std::runtime_error(std::string(__FILE__) + " (" + toString(__LINE__) + "): " + std::string(str));
  #define THROW_RUNTIME_ERROR(str) \
    printf("%s (%d): %s", __FILE__, __LINE__, std::string(str).c_str()), abort();
  // -- GODOT end --
#else
  // -- GODOT start --
  // #define THROW_RUNTIME_ERROR(str)
  //   throw std::runtime_error(str);
  #define THROW_RUNTIME_ERROR(str) \
    abort();
  // -- GODOT end --
#endif

#define FATAL(x)   THROW_RUNTIME_ERROR(x)
#define WARNING(x) { std::cerr << "Warning: " << x << embree_endl << std::flush; }

#define NOT_IMPLEMENTED FATAL(std::string(__FUNCTION__) + " not implemented")

////////////////////////////////////////////////////////////////////////////////
/// Basic types
////////////////////////////////////////////////////////////////////////////////

/* default floating-point type */
namespace embree {
  typedef float real;
}

/* windows does not have ssize_t */
#if defined(__WIN32__)
#if defined(__64BIT__)
typedef int64_t ssize_t;
#else
typedef int32_t ssize_t;
#endif
#endif

////////////////////////////////////////////////////////////////////////////////
/// Basic utility functions
////////////////////////////////////////////////////////////////////////////////

__forceinline std::string toString(long long value) {
  return std::to_string(value);
}

////////////////////////////////////////////////////////////////////////////////
/// Disable some compiler warnings
////////////////////////////////////////////////////////////////////////////////

#if defined(__INTEL_COMPILER)
//#pragma warning(disable:265 ) // floating-point operation result is out of range
//#pragma warning(disable:383 ) // value copied to temporary, reference to temporary used
//#pragma warning(disable:869 ) // parameter was never referenced
//#pragma warning(disable:981 ) // operands are evaluated in unspecified order
//#pragma warning(disable:1418) // external function definition with no prior declaration
//#pragma warning(disable:1419) // external declaration in primary source file
//#pragma warning(disable:1572) // floating-point equality and inequality comparisons are unreliable
//#pragma warning(disable:94  ) // the size of an array must be greater than zero
//#pragma warning(disable:1599) // declaration hides parameter
//#pragma warning(disable:424 ) // extra ";" ignored
#pragma warning(disable:2196) // routine is both "inline" and "noinline"
//#pragma warning(disable:177 ) // label was declared but never referenced
//#pragma warning(disable:114 ) // function was referenced but not defined
//#pragma warning(disable:819 ) // template nesting depth does not match the previous declaration of function
#pragma warning(disable:15335)  // was not vectorized: vectorization possible but seems inefficient
#endif

#if defined(_MSC_VER)
//#pragma warning(disable:4200) // nonstandard extension used : zero-sized array in struct/union
#pragma warning(disable:4800) // forcing value to bool 'true' or 'false' (performance warning)
//#pragma warning(disable:4267) // '=' : conversion from 'size_t' to 'unsigned long', possible loss of data
#pragma warning(disable:4244) // 'argument' : conversion from 'ssize_t' to 'unsigned int', possible loss of data
#pragma warning(disable:4267) // conversion from 'size_t' to 'const int', possible loss of data
//#pragma warning(disable:4355) // 'this' : used in base member initializer list
//#pragma warning(disable:391 ) // '<=' : signed / unsigned mismatch
//#pragma warning(disable:4018) // '<' : signed / unsigned mismatch
//#pragma warning(disable:4305) // 'initializing' : truncation from 'double' to 'float'
//#pragma warning(disable:4068) // unknown pragma
//#pragma warning(disable:4146) // unary minus operator applied to unsigned type, result still unsigned
//#pragma warning(disable:4838) // conversion from 'unsigned int' to 'const int' requires a narrowing conversion)
//#pragma warning(disable:4227) // anachronism used : qualifiers on reference are ignored
#pragma warning(disable:4503) // decorated name length exceeded, name was truncated
#pragma warning(disable:4180) // qualifier applied to function type has no meaning; ignored
#pragma warning(disable:4258) // definition from the for loop is ignored; the definition from the enclosing scope is used

#  if _MSC_VER < 1910 // prior to Visual studio 2017 (V141)
#    pragma warning(disable:4101) // warning C4101: 'x': unreferenced local variable // a compiler bug issues wrong warnings
#    pragma warning(disable:4789) // buffer '' of size 8 bytes will be overrun; 32 bytes will be written starting at offset 0
#  endif

#endif

#if defined(__clang__) && !defined(__INTEL_COMPILER)
//#pragma clang diagnostic ignored "-Wunknown-pragmas"
//#pragma clang diagnostic ignored "-Wunused-variable"
//#pragma clang diagnostic ignored "-Wreorder"
//#pragma clang diagnostic ignored "-Wmicrosoft"
//#pragma clang diagnostic ignored "-Wunused-private-field"
//#pragma clang diagnostic ignored "-Wunused-local-typedef"
//#pragma clang diagnostic ignored "-Wunused-function"
//#pragma clang diagnostic ignored "-Wnarrowing"
//#pragma clang diagnostic ignored "-Wc++11-narrowing"
//#pragma clang diagnostic ignored "-Wdeprecated-register"
//#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wpragmas"
//#pragma GCC diagnostic ignored "-Wnarrowing"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
//#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
//#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wattributes"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wparentheses"
#endif

#if defined(__clang__) && defined(__WIN32__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wmicrosoft-cast"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"
#pragma clang diagnostic ignored "-Wmicrosoft-include"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#endif

/* disabling deprecated warning, please use only where use of deprecated Embree API functions is desired */
#if defined(__WIN32__) && defined(__INTEL_COMPILER)
#define DISABLE_DEPRECATED_WARNING __pragma(warning (disable: 1478)) // warning: function was declared deprecated
#define ENABLE_DEPRECATED_WARNING  __pragma(warning (enable:  1478)) // warning: function was declared deprecated
#elif defined(__INTEL_COMPILER)
#define DISABLE_DEPRECATED_WARNING _Pragma("warning (disable: 1478)") // warning: function was declared deprecated
#define ENABLE_DEPRECATED_WARNING  _Pragma("warning (enable : 1478)") // warning: function was declared deprecated
#elif defined(__clang__)
#define DISABLE_DEPRECATED_WARNING _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"") // warning: xxx is deprecated
#define ENABLE_DEPRECATED_WARNING  _Pragma("clang diagnostic warning \"-Wdeprecated-declarations\"") // warning: xxx is deprecated
#elif defined(__GNUC__)
#define DISABLE_DEPRECATED_WARNING _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"") // warning: xxx is deprecated
#define ENABLE_DEPRECATED_WARNING  _Pragma("GCC diagnostic warning \"-Wdeprecated-declarations\"") // warning: xxx is deprecated
#elif defined(_MSC_VER)
#define DISABLE_DEPRECATED_WARNING __pragma(warning (disable: 4996)) // warning: function was declared deprecated
#define ENABLE_DEPRECATED_WARNING  __pragma(warning (enable : 4996)) // warning: function was declared deprecated
#endif

/* embree output stream */
#define embree_ostream std::ostream&
#define embree_cout std::cout
#define embree_cout_uniform std::cout
#define embree_endl std::endl
  
////////////////////////////////////////////////////////////////////////////////
/// Some macros for static profiling
////////////////////////////////////////////////////////////////////////////////

#if defined (__GNUC__) 
#define IACA_SSC_MARK( MARK_ID )						\
__asm__ __volatile__ (									\
					  "\n\t  movl $"#MARK_ID", %%ebx"	\
					  "\n\t  .byte 0x64, 0x67, 0x90"	\
					  : : : "memory" );

#define IACA_UD_BYTES __asm__ __volatile__ ("\n\t .byte 0x0F, 0x0B");

#else
#define IACA_UD_BYTES {__asm _emit 0x0F \
	__asm _emit 0x0B}

#define IACA_SSC_MARK(x) {__asm  mov ebx, x\
	__asm  _emit 0x64 \
	__asm  _emit 0x67 \
	__asm  _emit 0x90 }

#define IACA_VC64_START __writegsbyte(111, 111);
#define IACA_VC64_END   __writegsbyte(222, 222);

#endif

#define IACA_START {IACA_UD_BYTES \
					IACA_SSC_MARK(111)}
#define IACA_END {IACA_SSC_MARK(222) \
					IACA_UD_BYTES}

namespace embree
{
  template<typename Closure>
    struct OnScopeExitHelper
  {
    OnScopeExitHelper (const Closure f) : active(true), f(f) {}
    ~OnScopeExitHelper() { if (active) f(); }
    void deactivate() { active = false; }
    bool active;
    const Closure f;
  };
  
  template <typename Closure>
    OnScopeExitHelper<Closure> OnScopeExit(const Closure f) {
    return OnScopeExitHelper<Closure>(f);
  }

#define STRING_JOIN2(arg1, arg2) DO_STRING_JOIN2(arg1, arg2)
#define DO_STRING_JOIN2(arg1, arg2) arg1 ## arg2
#define ON_SCOPE_EXIT(code)                                             \
  auto STRING_JOIN2(on_scope_exit_, __LINE__) = OnScopeExit([&](){code;})

  template<typename Ty>
    std::unique_ptr<Ty> make_unique(Ty* ptr) {
    return std::unique_ptr<Ty>(ptr);
  }

}
