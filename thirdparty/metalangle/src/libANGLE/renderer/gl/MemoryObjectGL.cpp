// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "libANGLE/renderer/gl/MemoryObjectGL.h"

#include "libANGLE/Context.h"
#include "libANGLE/renderer/gl/ContextGL.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"
#include "libANGLE/renderer/gl/renderergl_utils.h"

namespace rx
{
MemoryObjectGL::MemoryObjectGL(GLuint memoryObject) : mMemoryObject(memoryObject)
{
    ASSERT(mMemoryObject != 0);
}

MemoryObjectGL::~MemoryObjectGL()
{
    ASSERT(mMemoryObject == 0);
}

void MemoryObjectGL::onDestroy(const gl::Context *context)
{
    const FunctionsGL *functions = GetFunctionsGL(context);
    functions->deleteMemoryObjectsEXT(1, &mMemoryObject);
    mMemoryObject = 0;
}

angle::Result MemoryObjectGL::importFd(gl::Context *context,
                                       GLuint64 size,
                                       gl::HandleType handleType,
                                       GLint fd)
{
    const FunctionsGL *functions = GetFunctionsGL(context);
    functions->importMemoryFdEXT(mMemoryObject, size, ToGLenum(handleType), fd);
    return angle::Result::Continue;
}

GLuint MemoryObjectGL::getMemoryObjectID() const
{
    return mMemoryObject;
}
}  // namespace rx
