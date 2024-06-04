// Copyright (c) 2006, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// string_utils-inl.h: Safer string manipulation on Windows, supporting
// pre-MSVC8 environments.

#ifndef COMMON_WINDOWS_STRING_UTILS_INL_H_
#define COMMON_WINDOWS_STRING_UTILS_INL_H_

#include <stdarg.h>
#include <wchar.h>

#include <string>

// The "ll" printf format size specifier corresponding to |long long| was
// intrudced in MSVC8.  Earlier versions did not provide this size specifier,
// but "I64" can be used to print 64-bit types.  Don't use "I64" where "ll"
// is available, in the event of oddball systems where |long long| is not
// 64 bits wide.
#if _MSC_VER >= 1400  // MSVC 2005/8
#define WIN_STRING_FORMAT_LL "ll"
#else  // MSC_VER >= 1400
#define WIN_STRING_FORMAT_LL "I64"
#endif  // MSC_VER >= 1400

// A nonconforming version of swprintf, without the length argument, was
// included with the CRT prior to MSVC8.  Although a conforming version was
// also available via an overload, it is not reliably chosen.  _snwprintf
// behaves as a standards-confirming swprintf should, so force the use of
// _snwprintf when using older CRTs.
#if _MSC_VER < 1400  // MSVC 2005/8
#define swprintf _snwprintf
#else
// For MSVC8 and newer, swprintf_s is the recommended method. Conveniently,
// it takes the same argument list as swprintf.
#define swprintf swprintf_s
#endif  // MSC_VER < 1400

namespace google_breakpad {

using std::string;
using std::wstring;

class WindowsStringUtils {
 public:
  // Roughly equivalent to MSVC8's wcscpy_s, except pre-MSVC8, this does
  // not fail if source is longer than destination_size.  The destination
  // buffer is always 0-terminated.
  static void safe_wcscpy(wchar_t* destination, size_t destination_size,
                          const wchar_t* source);

  // Roughly equivalent to MSVC8's wcsncpy_s, except that _TRUNCATE cannot
  // be passed directly, and pre-MSVC8, this will not fail if source or count
  // are longer than destination_size.  The destination buffer is always
  // 0-terminated.
  static void safe_wcsncpy(wchar_t* destination, size_t destination_size,
                           const wchar_t* source, size_t count);

  // Performs multi-byte to wide character conversion on C++ strings, using
  // mbstowcs_s (MSVC8) or mbstowcs (pre-MSVC8).  Returns false on failure,
  // without setting wcs.
  static bool safe_mbstowcs(const string& mbs, wstring* wcs);

  // The inverse of safe_mbstowcs.
  static bool safe_wcstombs(const wstring& wcs, string* mbs);

  // Returns the base name of a file, e.g. strips off the path.
  static wstring GetBaseName(const wstring& filename);

 private:
  // Disallow instantiation and other object-based operations.
  WindowsStringUtils();
  WindowsStringUtils(const WindowsStringUtils&);
  ~WindowsStringUtils();
  void operator=(const WindowsStringUtils&);
};

// static
inline void WindowsStringUtils::safe_wcscpy(wchar_t* destination,
                                            size_t destination_size,
                                            const wchar_t* source) {
#if _MSC_VER >= 1400  // MSVC 2005/8
  wcscpy_s(destination, destination_size, source);
#else  // _MSC_VER >= 1400
  // Pre-MSVC 2005/8 doesn't have wcscpy_s.  Simulate it with wcsncpy.
  // wcsncpy doesn't 0-terminate the destination buffer if the source string
  // is longer than size.  Ensure that the destination is 0-terminated.
  wcsncpy(destination, source, destination_size);
  if (destination && destination_size)
    destination[destination_size - 1] = 0;
#endif  // _MSC_VER >= 1400
}

// static
inline void WindowsStringUtils::safe_wcsncpy(wchar_t* destination,
                                             size_t destination_size,
                                             const wchar_t* source,
                                             size_t count) {
#if _MSC_VER >= 1400  // MSVC 2005/8
  wcsncpy_s(destination, destination_size, source, count);
#else  // _MSC_VER >= 1400
  // Pre-MSVC 2005/8 doesn't have wcsncpy_s.  Simulate it with wcsncpy.
  // wcsncpy doesn't 0-terminate the destination buffer if the source string
  // is longer than size.  Ensure that the destination is 0-terminated.
  if (destination_size < count)
    count = destination_size;

  wcsncpy(destination, source, count);
  if (destination && count)
    destination[count - 1] = 0;
#endif  // _MSC_VER >= 1400
}

}  // namespace google_breakpad

#endif  // COMMON_WINDOWS_STRING_UTILS_INL_H_
