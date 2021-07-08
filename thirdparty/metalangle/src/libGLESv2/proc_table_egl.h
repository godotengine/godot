//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// getProcAddress loader table:
//   Mapping from a string entry point name to function address.
//

#ifndef LIBGLESV2_PROC_TABLE_H_
#define LIBGLESV2_PROC_TABLE_H_

#include <EGL/egl.h>
#include <stddef.h>
#include <utility>

namespace egl
{
using ProcEntry = std::pair<const char *, __eglMustCastToProperFunctionPointerType>;

extern ProcEntry g_procTable[];
extern size_t g_numProcs;
}  // namespace egl

#endif  // LIBGLESV2_PROC_TABLE_H_
