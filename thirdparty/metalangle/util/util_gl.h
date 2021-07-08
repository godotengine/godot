//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// util_gl.h: Includes the right GL/EGL headers for static/shared link.

#ifndef UTIL_GL_H_
#define UTIL_GL_H_

#include "common/platform.h"

#if defined(ANGLE_USE_UTIL_LOADER)
#    include "util/egl_loader_autogen.h"
#    include "util/gles_loader_autogen.h"
#    if defined(ANGLE_PLATFORM_WINDOWS) && !defined(ANGLE_ENABLE_WINDOWS_UWP)
#        include "util/windows/wgl_loader_autogen.h"
#    endif  // defined(ANGLE_PLATFORM_WINDOWS) && !defined(ANGLE_ENABLE_WINDOWS_UWP)
#else

#    if !defined(GL_GLES_PROTOTYPES)
#        error Config error. Should either be using the ANGLE GL loader or header prototypes.
#    endif  // !defined(GL_GLES_PROTOTYPES)

#    include <EGL/egl.h>
#    include <EGL/eglext.h>
#    include "angle_gl.h"
#endif  // defined(ANGLE_USE_UTIL_LOADER)

#endif  // UTIL_GL_H_
