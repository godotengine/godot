// -*- C++ -*-
//===--------------------------- stdbool.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX_STDBOOL_H
#define _LIBCUDACXX_STDBOOL_H


/*
    stdbool.h synopsis

Macros:

    __bool_true_false_are_defined

*/

#include <__config>

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#include_next <stdbool.h>

#ifdef __cplusplus
#undef bool
#undef true
#undef false
#undef __bool_true_false_are_defined
#define __bool_true_false_are_defined 1
#endif

#endif  // _LIBCUDACXX_STDBOOL_H
