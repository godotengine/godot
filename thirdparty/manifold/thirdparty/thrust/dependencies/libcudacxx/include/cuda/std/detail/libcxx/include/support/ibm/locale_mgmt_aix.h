// -*- C++ -*-
//===------------------- support/ibm/locale_mgmt_aix.h --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_SUPPORT_IBM_LOCALE_MGMT_AIX_H
#define _LIBCUDACXX_SUPPORT_IBM_LOCALE_MGMT_AIX_H

#if defined(_AIX)
#include "cstdlib"

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(_AIX71)
// AIX 7.1 and higher has these definitions.  Definitions and stubs
// are provied here as a temporary workaround on AIX 6.1.

#define LC_COLLATE_MASK         1
#define LC_CTYPE_MASK           2
#define LC_MESSAGES_MASK        4
#define LC_MONETARY_MASK        8
#define LC_NUMERIC_MASK         16
#define LC_TIME_MASK            32
#define LC_ALL_MASK             (LC_COLLATE_MASK | LC_CTYPE_MASK | \
                                 LC_MESSAGES_MASK | LC_MONETARY_MASK |\
                                 LC_NUMERIC_MASK | LC_TIME_MASK)

typedef void* locale_t;

// The following are stubs.  They are not supported on AIX 6.1.
static inline
locale_t newlocale(int category_mask, const char *locale, locale_t base)
{
  _LC_locale_t *newloc, *loc;
  if ((loc = (_LC_locale_t *)__xopen_locale(locale)) == NULL)
  {
    errno = EINVAL;
    return (locale_t)0;
  }
  if ((newloc = (_LC_locale_t *)calloc(1, sizeof(_LC_locale_t))) == NULL)
  {
    errno = ENOMEM;
    return (locale_t)0;
  }
  if (!base)
    base = (_LC_locale_t *)__xopen_locale("C");
  memcpy(newloc, base, sizeof (_LC_locale_t));
  if (category_mask & LC_COLLATE_MASK)
    newloc->lc_collate = loc->lc_collate;
  if (category_mask & LC_CTYPE_MASK)
    newloc->lc_ctype = loc->lc_ctype;
  //if (category_mask & LC_MESSAGES_MASK)
  //  newloc->lc_messages = loc->lc_messages;
  if (category_mask & LC_MONETARY_MASK)
    newloc->lc_monetary = loc->lc_monetary;
  if (category_mask & LC_TIME_MASK)
    newloc->lc_time = loc->lc_time;
  if (category_mask & LC_NUMERIC_MASK)
    newloc->lc_numeric = loc->lc_numeric;
  return (locale_t)newloc;
}
static inline
void freelocale(locale_t locobj)
{
  free(locobj);
}
static inline
locale_t uselocale(locale_t newloc)
{
  return (locale_t)0;
}
#endif // !defined(_AIX71)

#ifdef __cplusplus
}
#endif
#endif // defined(_AIX)
#endif // _LIBCUDACXX_SUPPORT_IBM_LOCALE_MGMT_AIX_H
