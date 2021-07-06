//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// EGLSync.h: Defines the egl::Sync classes, which support the EGL_KHR_fence_sync,
// EGL_KHR_wait_sync and EGL 1.5 sync objects.

#ifndef LIBANGLE_EGLSYNC_H_
#define LIBANGLE_EGLSYNC_H_

#include "libANGLE/Debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/RefCountObject.h"

#include "common/angleutils.h"

namespace rx
{
class EGLImplFactory;
class EGLSyncImpl;
}  // namespace rx

namespace gl
{
class Context;
}  // namespace gl

namespace egl
{
class Sync final : public angle::RefCountObject<Display, angle::Result>, public LabeledObject
{
  public:
    Sync(rx::EGLImplFactory *factory, EGLenum type, const AttributeMap &attribs);
    ~Sync() override;

    void setLabel(EGLLabelKHR label) override;
    EGLLabelKHR getLabel() const override;

    void onDestroy(const Display *display) override;

    Error initialize(const Display *display, const gl::Context *context);
    Error clientWait(const Display *display,
                     const gl::Context *context,
                     EGLint flags,
                     EGLTime timeout,
                     EGLint *outResult);
    Error serverWait(const Display *display, const gl::Context *context, EGLint flags);
    Error getStatus(const Display *display, EGLint *outStatus) const;

    Error dupNativeFenceFD(const Display *display, EGLint *result) const;

    EGLenum getType() const { return mType; }
    EGLint getCondition() const { return mCondition; }
    EGLint getNativeFenceFD() const { return mNativeFenceFD; }

  private:
    std::unique_ptr<rx::EGLSyncImpl> mFence;

    EGLLabelKHR mLabel;

    EGLenum mType;
    static constexpr EGLint mCondition = EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR;
    EGLint mNativeFenceFD;
};

}  // namespace egl

#endif  // LIBANGLE_FENCE_H_
