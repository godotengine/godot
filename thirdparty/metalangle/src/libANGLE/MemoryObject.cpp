//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// MemoryObject.h: Implements the gl::MemoryObject class [EXT_external_objects]

#include "libANGLE/MemoryObject.h"

#include "common/angleutils.h"
#include "libANGLE/renderer/GLImplFactory.h"
#include "libANGLE/renderer/MemoryObjectImpl.h"

namespace gl
{

MemoryObject::MemoryObject(rx::GLImplFactory *factory, MemoryObjectID id)
    : RefCountObject(id), mImplementation(factory->createMemoryObject())
{}

MemoryObject::~MemoryObject() {}

void MemoryObject::onDestroy(const Context *context)
{
    mImplementation->onDestroy(context);
}

angle::Result MemoryObject::importFd(Context *context,
                                     GLuint64 size,
                                     HandleType handleType,
                                     GLint fd)
{
    return mImplementation->importFd(context, size, handleType, fd);
}

}  // namespace gl
