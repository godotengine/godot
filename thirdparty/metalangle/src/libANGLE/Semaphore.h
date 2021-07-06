//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Semaphore.h: Defines the gl::Semaphore class [EXT_external_objects]

#ifndef LIBANGLE_SEMAPHORE_H_
#define LIBANGLE_SEMAPHORE_H_

#include <memory>

#include "angle_gl.h"
#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/RefCountObject.h"
#include "libANGLE/angletypes.h"

namespace rx
{
class GLImplFactory;
class SemaphoreImpl;
}  // namespace rx

namespace gl
{
class Context;

class Semaphore final : public RefCountObject<SemaphoreID>
{
  public:
    Semaphore(rx::GLImplFactory *factory, SemaphoreID id);
    ~Semaphore() override;

    void onDestroy(const Context *context) override;

    rx::SemaphoreImpl *getImplementation() const { return mImplementation.get(); }

    angle::Result importFd(Context *context, HandleType handleType, GLint fd);

    angle::Result wait(Context *context,
                       const BufferBarrierVector &bufferBarriers,
                       const TextureBarrierVector &textureBarriers);

    angle::Result signal(Context *context,
                         const BufferBarrierVector &bufferBarriers,
                         const TextureBarrierVector &textureBarriers);

  private:
    std::unique_ptr<rx::SemaphoreImpl> mImplementation;
};

}  // namespace gl

#endif  // LIBANGLE_SEMAPHORE_H_
