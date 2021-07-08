//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// glext_angle.h: ANGLE modifications to the glext.h header file.
//   Currently we don't include this file directly, we patch glext.h
//   to include it implicitly so it is visible throughout our code.

#ifndef INCLUDE_GLES_GLEXT_ANGLE_H_
#define INCLUDE_GLES_GLEXT_ANGLE_H_

// clang-format off

#ifndef GL_ANGLE_explicit_context_gles1
#define GL_ANGLE_explicit_context_gles1
typedef void *GLeglContext;
#include "glext_explicit_context_autogen.inc"
#endif /* GL_ANGLE_explicit_context_gles1 */

// clang-format on

#endif  // INCLUDE_GLES_GLEXT_ANGLE_H_
