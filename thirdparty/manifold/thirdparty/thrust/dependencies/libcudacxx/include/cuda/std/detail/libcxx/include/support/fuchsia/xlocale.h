// -*- C++ -*-
//===------------------- support/fuchsia/xlocale.h ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_SUPPORT_FUCHSIA_XLOCALE_H
#define _LIBCUDACXX_SUPPORT_FUCHSIA_XLOCALE_H

#if defined(__Fuchsia__)

#include <cstdlib>
#include <cwchar>
#include <support/xlocale/__posix_l_fallback.h>
#include <support/xlocale/__strtonum_fallback.h>

#endif // defined(__Fuchsia__)

#endif // _LIBCUDACXX_SUPPORT_FUCHSIA_XLOCALE_H
