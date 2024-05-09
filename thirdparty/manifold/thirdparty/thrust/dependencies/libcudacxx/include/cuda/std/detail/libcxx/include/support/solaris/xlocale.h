//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

////////////////////////////////////////////////////////////////////////////////
// Minimal xlocale implementation for Solaris.  This implements the subset of
// the xlocale APIs that libc++ depends on.
////////////////////////////////////////////////////////////////////////////////
#ifndef __XLOCALE_H_INCLUDED
#define __XLOCALE_H_INCLUDED

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


int snprintf_l(char *__s, size_t __n, locale_t __l, const char *__format, ...);
int asprintf_l(char **__s, locale_t __l, const char *__format, ...);

int sscanf_l(const char *__s, locale_t __l, const char *__format, ...);

int toupper_l(int __c, locale_t __l);
int tolower_l(int __c, locale_t __l);

struct lconv *localeconv(void);
struct lconv *localeconv_l(locale_t __l);

// FIXME: These are quick-and-dirty hacks to make things pretend to work
static inline
long long strtoll_l(const char *__nptr, char **__endptr,
    int __base, locale_t __loc) {
  return strtoll(__nptr, __endptr, __base);
}
static inline
long strtol_l(const char *__nptr, char **__endptr,
    int __base, locale_t __loc) {
  return strtol(__nptr, __endptr, __base);
}
static inline
unsigned long long strtoull_l(const char *__nptr, char **__endptr,
    int __base, locale_t __loc) {
  return strtoull(__nptr, __endptr, __base);
}
static inline
unsigned long strtoul_l(const char *__nptr, char **__endptr,
    int __base, locale_t __loc) {
  return strtoul(__nptr, __endptr, __base);
}
static inline
float strtof_l(const char *__nptr, char **__endptr,
    locale_t __loc) {
  return strtof(__nptr, __endptr);
}
static inline
double strtod_l(const char *__nptr, char **__endptr,
    locale_t __loc) {
  return strtod(__nptr, __endptr);
}
static inline
long double strtold_l(const char *__nptr, char **__endptr,
    locale_t __loc) {
  return strtold(__nptr, __endptr);
}


#ifdef __cplusplus
}
#endif

#endif
