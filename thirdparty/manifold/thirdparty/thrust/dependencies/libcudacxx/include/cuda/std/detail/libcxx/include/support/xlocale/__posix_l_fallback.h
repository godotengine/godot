// -*- C++ -*-
//===--------------- support/xlocale/__posix_l_fallback.h -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// These are reimplementations of some extended locale functions ( *_l ) that
// are normally part of POSIX.  This shared implementation provides parts of the
// extended locale support for libc's that normally don't have any (like
// Android's bionic and Newlib).
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_SUPPORT_XLOCALE_POSIX_L_FALLBACK_H
#define _LIBCUDACXX_SUPPORT_XLOCALE_POSIX_L_FALLBACK_H

#ifdef __cplusplus
extern "C" {
#endif

inline _LIBCUDACXX_INLINE_VISIBILITY int isalnum_l(int c, locale_t) {
  return ::isalnum(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int isalpha_l(int c, locale_t) {
  return ::isalpha(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int isblank_l(int c, locale_t) {
  return ::isblank(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int iscntrl_l(int c, locale_t) {
  return ::iscntrl(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int isdigit_l(int c, locale_t) {
  return ::isdigit(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int isgraph_l(int c, locale_t) {
  return ::isgraph(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int islower_l(int c, locale_t) {
  return ::islower(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int isprint_l(int c, locale_t) {
  return ::isprint(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int ispunct_l(int c, locale_t) {
  return ::ispunct(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int isspace_l(int c, locale_t) {
  return ::isspace(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int isupper_l(int c, locale_t) {
  return ::isupper(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int isxdigit_l(int c, locale_t) {
  return ::isxdigit(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int iswalnum_l(wint_t c, locale_t) {
  return ::iswalnum(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int iswalpha_l(wint_t c, locale_t) {
  return ::iswalpha(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int iswblank_l(wint_t c, locale_t) {
  return ::iswblank(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int iswcntrl_l(wint_t c, locale_t) {
  return ::iswcntrl(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int iswdigit_l(wint_t c, locale_t) {
  return ::iswdigit(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int iswgraph_l(wint_t c, locale_t) {
  return ::iswgraph(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int iswlower_l(wint_t c, locale_t) {
  return ::iswlower(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int iswprint_l(wint_t c, locale_t) {
  return ::iswprint(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int iswpunct_l(wint_t c, locale_t) {
  return ::iswpunct(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int iswspace_l(wint_t c, locale_t) {
  return ::iswspace(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int iswupper_l(wint_t c, locale_t) {
  return ::iswupper(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int iswxdigit_l(wint_t c, locale_t) {
  return ::iswxdigit(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int toupper_l(int c, locale_t) {
  return ::toupper(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int tolower_l(int c, locale_t) {
  return ::tolower(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY wint_t towupper_l(wint_t c, locale_t) {
  return ::towupper(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY wint_t towlower_l(wint_t c, locale_t) {
  return ::towlower(c);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int strcoll_l(const char *s1, const char *s2,
                                               locale_t) {
  return ::strcoll(s1, s2);
}

inline _LIBCUDACXX_INLINE_VISIBILITY size_t strxfrm_l(char *dest, const char *src,
                                                  size_t n, locale_t) {
  return ::strxfrm(dest, src, n);
}

inline _LIBCUDACXX_INLINE_VISIBILITY size_t strftime_l(char *s, size_t max,
                                                   const char *format,
                                                   const struct tm *tm, locale_t) {
  return ::strftime(s, max, format, tm);
}

inline _LIBCUDACXX_INLINE_VISIBILITY int wcscoll_l(const wchar_t *ws1,
                                               const wchar_t *ws2, locale_t) {
  return ::wcscoll(ws1, ws2);
}

inline _LIBCUDACXX_INLINE_VISIBILITY size_t wcsxfrm_l(wchar_t *dest, const wchar_t *src,
                                                  size_t n, locale_t) {
  return ::wcsxfrm(dest, src, n);
}

#ifdef __cplusplus
}
#endif

#endif // _LIBCUDACXX_SUPPORT_XLOCALE_POSIX_L_FALLBACK_H
