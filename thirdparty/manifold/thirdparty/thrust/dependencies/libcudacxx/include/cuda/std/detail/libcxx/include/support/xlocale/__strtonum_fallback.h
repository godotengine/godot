// -*- C++ -*-
//===-------------- support/xlocale/__strtonum_fallback.h -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// These are reimplementations of some extended locale functions ( *_l ) that
// aren't part of POSIX.  They are widely available though (GLIBC, BSD, maybe
// others).  The unifying aspect in this case is that all of these functions
// convert strings to some numeric type.
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_SUPPORT_XLOCALE_STRTONUM_FALLBACK_H
#define _LIBCUDACXX_SUPPORT_XLOCALE_STRTONUM_FALLBACK_H

#ifdef __cplusplus
extern "C" {
#endif

inline _LIBCUDACXX_INLINE_VISIBILITY float strtof_l(const char *nptr,
                                                char **endptr, locale_t) {
  return ::strtof(nptr, endptr);
}

inline _LIBCUDACXX_INLINE_VISIBILITY double strtod_l(const char *nptr,
                                                 char **endptr, locale_t) {
  return ::strtod(nptr, endptr);
}

inline _LIBCUDACXX_INLINE_VISIBILITY long double strtold_l(const char *nptr,
                                                       char **endptr, locale_t) {
  return ::strtold(nptr, endptr);
}

inline _LIBCUDACXX_INLINE_VISIBILITY long long
strtoll_l(const char *nptr, char **endptr, int base, locale_t) {
  return ::strtoll(nptr, endptr, base);
}

inline _LIBCUDACXX_INLINE_VISIBILITY unsigned long long
strtoull_l(const char *nptr, char **endptr, int base, locale_t) {
  return ::strtoull(nptr, endptr, base);
}

inline _LIBCUDACXX_INLINE_VISIBILITY long long
wcstoll_l(const wchar_t *nptr, wchar_t **endptr, int base, locale_t) {
  return ::wcstoll(nptr, endptr, base);
}

inline _LIBCUDACXX_INLINE_VISIBILITY unsigned long long
wcstoull_l(const wchar_t *nptr, wchar_t **endptr, int base, locale_t) {
  return ::wcstoull(nptr, endptr, base);
}

inline _LIBCUDACXX_INLINE_VISIBILITY long double wcstold_l(const wchar_t *nptr,
                                                       wchar_t **endptr, locale_t) {
  return ::wcstold(nptr, endptr);
}

#ifdef __cplusplus
}
#endif

#endif // _LIBCUDACXX_SUPPORT_XLOCALE_STRTONUM_FALLBACK_H
