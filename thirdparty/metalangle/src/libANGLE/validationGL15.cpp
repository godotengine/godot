//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// validationGL15.cpp: Validation functions for OpenGL 1.5 entry point parameters

#include "libANGLE/validationGL15_autogen.h"

namespace gl
{

bool ValidateGetBufferSubData(Context *context,
                              GLenum target,
                              GLintptr offset,
                              GLsizeiptr size,
                              void *data)
{
    return true;
}

bool ValidateGetQueryObjectiv(Context *context, QueryID id, GLenum pname, GLint *params)
{
    return true;
}

bool ValidateMapBuffer(Context *context, BufferBinding targetPacked, GLenum access)
{
    return true;
}

}  // namespace gl
