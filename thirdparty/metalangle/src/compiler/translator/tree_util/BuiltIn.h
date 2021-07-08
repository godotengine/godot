// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// BuiltIn.h:
//   Chooses BuiltIn_complete_autogen.h or BuiltIn_ESSL_autogen.h
//   depending on whether platform is android

#ifndef COMPILER_TRANSLATOR_TREEUTIL_BUILTIN_H_
#define COMPILER_TRANSLATOR_TREEUTIL_BUILTIN_H_

#if defined(ANDROID)
#    include "BuiltIn_ESSL_autogen.h"
#else
#    include "BuiltIn_complete_autogen.h"
#endif

#endif  // COMPILER_TRANSLATOR_TREEUTIL_BUILTIN_H_
