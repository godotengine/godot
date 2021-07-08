// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ParseContext_interm.h:
//   Chooses ParseContext_complete_autogen.h or ParseContext_ESSL_autogen.h
//   depending on whether platform is android

#ifndef COMPILER_TRANSLATOR_PARSECONTEXT_INTERM_H_
#define COMPILER_TRANSLATOR_PARSECONTEXT_INTERM_H_

#if defined(ANDROID)
#    include "ParseContext_ESSL_autogen.h"
#else
#    include "ParseContext_complete_autogen.h"
#endif

#endif  // COMPILER_TRANSLATOR_PARSECONTEXT_INTERM_H_
