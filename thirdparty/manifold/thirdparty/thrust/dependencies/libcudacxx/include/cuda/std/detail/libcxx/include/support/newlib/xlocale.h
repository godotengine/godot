//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_SUPPORT_NEWLIB_XLOCALE_H
#define _LIBCUDACXX_SUPPORT_NEWLIB_XLOCALE_H

#if defined(_NEWLIB_VERSION)

#include <cstdlib>
#include <clocale>
#include <cwctype>
#include <ctype.h>
#if !defined(__NEWLIB__) || __NEWLIB__ < 2 || \
    __NEWLIB__ == 2 && __NEWLIB_MINOR__ < 5
#include <support/xlocale/__nop_locale_mgmt.h>
#include <support/xlocale/__posix_l_fallback.h>
#include <support/xlocale/__strtonum_fallback.h>
#endif

#endif // _NEWLIB_VERSION

#endif
