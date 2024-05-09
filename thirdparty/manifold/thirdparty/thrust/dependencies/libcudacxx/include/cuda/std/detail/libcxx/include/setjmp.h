// -*- C++ -*-
//===--------------------------- setjmp.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_SETJMP_H
#define _LIBCUDACXX_SETJMP_H

/*
    setjmp.h synopsis

Macros:

    setjmp

Types:

    jmp_buf

void longjmp(jmp_buf env, int val);

*/

#include <__config>

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#include_next <setjmp.h>

#ifdef __cplusplus

#ifndef setjmp
#define setjmp(env) setjmp(env)
#endif

#endif // __cplusplus

#endif  // _LIBCUDACXX_SETJMP_H
