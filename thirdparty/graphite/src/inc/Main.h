/*  GRAPHITE2 LICENSING

    Copyright 2010, SIL International
    All rights reserved.

    This library is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation; either version 2.1 of License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should also have received a copy of the GNU Lesser General Public
    License along with this library in the file named "LICENSE".
    If not, write to the Free Software Foundation, 51 Franklin Street,
    Suite 500, Boston, MA 02110-1335, USA or visit their web page on the
    internet at http://www.fsf.org/licenses/lgpl.html.

Alternatively, the contents of this file may be used under the terms of the
Mozilla Public License (http://mozilla.org/MPL) or the GNU General Public
License, as published by the Free Software Foundation, either version 2
of the License or (at your option) any later version.
*/
#pragma once

#include <cstdlib>
#include "graphite2/Types.h"

#ifdef GRAPHITE2_CUSTOM_HEADER
#include GRAPHITE2_CUSTOM_HEADER
#endif

namespace graphite2 {

typedef gr_uint8        uint8;
typedef gr_uint8        byte;
typedef gr_uint16       uint16;
typedef gr_uint32       uint32;
typedef gr_int8         int8;
typedef gr_int16        int16;
typedef gr_int32        int32;
typedef size_t          uintptr;

#ifdef GRAPHITE2_TELEMETRY
struct telemetry
{
    class category;

    static size_t   * _category;
    static void set_category(size_t & t) throw()    { _category = &t; }
    static void stop() throw()                      { _category = 0; }
    static void count_bytes(size_t n) throw()       { if (_category) *_category += n; }

    size_t  misc,
            silf,
            glyph,
            code,
            states,
            starts,
            transitions;

    telemetry() : misc(0), silf(0), glyph(0), code(0), states(0), starts(0), transitions(0) {}
};

class telemetry::category
{
    size_t * _prev;
public:
    category(size_t & t) : _prev(_category) { _category = &t; }
    ~category() { _category = _prev; }
};

#else
struct telemetry  {};
#endif

// Checked multiplaction to catch overflow or underflow when allocating memory
#if defined(__has_builtin)
  #if __has_builtin(__builtin_mul_overflow)
    #define HAVE_BUILTIN_OVERFLOW
  #endif
#elif defined(__GNUC__) && (__GNUC__ >= 5) && !defined(__INTEL_COMPILER)
  #define HAVE_BUILTIN_OVERFLOW
#endif
#if defined(__has_include)
  #if __has_include(<intsafe.h>) && !defined(__CYGWIN__)
    #define HAVE_INTSAFE_H
  #endif
#elif defined(_WIN32)
  #define HAVE_INTSAFE_H
#endif

// Need to import intsafe into the top level namespace
#if defined(HAVE_INTSAFE_H)
} // namespace graphite2

#include <intsafe.h>

namespace graphite2 {
#endif

#if defined(HAVE_BUILTIN_OVERFLOW)
inline
bool checked_mul(const size_t a, const size_t b, size_t & t) {
    return __builtin_mul_overflow(a, b, &t);
}
#elif defined(HAVE_INTSAFE_H)
inline
bool checked_mul(const size_t a, const size_t b, size_t & t) {
    return SizeTMult(a, b, &t) == INTSAFE_E_ARITHMETIC_OVERFLOW;
}
#else
inline
bool checked_mul(const size_t a, const size_t b, size_t & t) {
  t = a*b;
  return (((a | b) & (~size_t(0) << (sizeof(size_t) << 2))) && (t / a != b));
}
#endif

// typesafe wrapper around malloc for simple types
// use free(pointer) to deallocate

template <typename T> T * gralloc(size_t n)
{
    size_t total;
    if (checked_mul(n, sizeof(T), total))
      return 0;
#ifdef GRAPHITE2_TELEMETRY
    telemetry::count_bytes(total);
#endif
    return static_cast<T*>(malloc(total));
}

template <typename T> T * grzeroalloc(size_t n)
{
#ifdef GRAPHITE2_TELEMETRY
    telemetry::count_bytes(sizeof(T) * n);
#endif
    return static_cast<T*>(calloc(n, sizeof(T)));
}

template <typename T>
inline T min(const T a, const T b)
{
    return a < b ? a : b;
}

template <typename T>
inline T max(const T a, const T b)
{
    return a > b ? a : b;
}

} // namespace graphite2

#define CLASS_NEW_DELETE \
    void * operator new   (size_t size){ return gralloc<byte>(size);} \
    void * operator new   (size_t, void * p) throw() { return p; } \
    void * operator new[] (size_t size) {return gralloc<byte>(size);} \
    void * operator new[] (size_t, void * p) throw() { return p; } \
    void operator delete   (void * p) throw() { free(p);} \
    void operator delete   (void *, void *) throw() {} \
    void operator delete[] (void * p)throw() { free(p); } \
    void operator delete[] (void *, void *) throw() {}

#if defined(__GNUC__)  || defined(__clang__)
#define GR_MAYBE_UNUSED __attribute__((unused))
#else
#define GR_MAYBE_UNUSED
#endif

#ifndef __has_cpp_attribute
#  define __has_cpp_attribute(x) 0
#endif

#if __has_cpp_attribute(clang::fallthrough)
#  define GR_FALLTHROUGH [[clang::fallthrough]]
#elif __has_cpp_attribute(gnu::fallthrough)
#  define GR_FALLTHROUGH [[gnu::fallthrough]]
#elif defined(_MSC_VER)
   /*
    * MSVC's __fallthrough annotations are checked by /analyze (Code Analysis):
    * https://msdn.microsoft.com/en-us/library/ms235402%28VS.80%29.aspx
    */
    #include <sal.h>
    #define GR_FALLTHROUGH __fallthrough
#elif __GNUC__ >= 7
    #define GR_FALLTHROUGH __attribute__ ((fallthrough))
#else
    #define GR_FALLTHROUGH /* fallthrough */
#endif

#ifdef _MSC_VER
#pragma warning(disable: 4800)
#pragma warning(disable: 4355)
#endif
