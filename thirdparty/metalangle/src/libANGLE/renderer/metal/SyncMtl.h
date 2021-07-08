//
// Copyright (c) 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SyncMtl:
//    Defines the class interface for SyncMtl, implementing SyncImpl.
//

#ifndef LIBANGLE_RENDERER_METAL_SYNCMTL_H_
#define LIBANGLE_RENDERER_METAL_SYNCMTL_H_

#include <condition_variable>
#include <mutex>

#include "libANGLE/renderer/EGLSyncImpl.h"
#include "libANGLE/renderer/FenceNVImpl.h"
#include "libANGLE/renderer/SyncImpl.h"
#include "libANGLE/renderer/metal/mtl_common.h"

namespace egl
{
class AttributeMap;
}

namespace rx
{

class ContextMtl;

namespace mtl
{

// Common class to be used by both SyncImpl and EGLSyncImpl.
// NOTE: SharedEvent is only declared on iOS 12.0+ or mac 10.14+
#if defined(__IPHONE_12_0) || defined(__MAC_10_14)
class Sync
{
  public:
    Sync();
    ~Sync();

    void onDestroy();

    angle::Result initialize(ContextMtl *contextMtl);

    angle::Result set(ContextMtl *contextMtl, GLenum condition, GLbitfield flags);
    angle::Result clientWait(ContextMtl *contextMtl,
                             bool flushCommands,
                             uint64_t timeout,
                             GLenum *outResult);
    void serverWait(ContextMtl *contextMtl);
    angle::Result getStatus(bool *signaled);

  private:
    SharedEventRef mMetalSharedEvent;
    uint64_t mSetCounter = 0;

    std::shared_ptr<std::condition_variable> mCv;
    std::shared_ptr<std::mutex> mLock;
};
#else   // #if defined(__IPHONE_12_0) || defined(__MAC_10_14)
class Sync
{
  public:
    void onDestroy() { UNREACHABLE(); }

    angle::Result initialize(ContextMtl *context)
    {
        UNREACHABLE();
        return angle::Result::Stop;
    }
    angle::Result set(ContextMtl *contextMtl, GLenum condition, GLbitfield flags)
    {
        UNREACHABLE();
        return angle::Result::Stop;
    }
    angle::Result clientWait(ContextMtl *context,
                             bool flushCommands,
                             uint64_t timeout,
                             GLenum *outResult)
    {
        UNREACHABLE();
        return angle::Result::Stop;
    }
    void serverWait(ContextMtl *contextMtl) { UNREACHABLE(); }
    angle::Result getStatus(bool *signaled)
    {
        UNREACHABLE();
        return angle::Result::Stop;
    }
};
#endif  // #if defined(__IPHONE_12_0) || defined(__MAC_10_14)
}  // namespace mtl

class FenceNVMtl : public FenceNVImpl
{
  public:
    FenceNVMtl();
    ~FenceNVMtl() override;

    angle::Result set(const gl::Context *context, GLenum condition) override;
    angle::Result test(const gl::Context *context, GLboolean *outFinished) override;
    angle::Result finish(const gl::Context *context) override;

  private:
    mtl::Sync mSync;
};

class SyncMtl : public SyncImpl
{
  public:
    SyncMtl();
    ~SyncMtl() override;

    void onDestroy(const gl::Context *context) override;

    angle::Result set(const gl::Context *context, GLenum condition, GLbitfield flags) override;
    angle::Result clientWait(const gl::Context *context,
                             GLbitfield flags,
                             GLuint64 timeout,
                             GLenum *outResult) override;
    angle::Result serverWait(const gl::Context *context,
                             GLbitfield flags,
                             GLuint64 timeout) override;
    angle::Result getStatus(const gl::Context *context, GLint *outResult) override;

  private:
    mtl::Sync mSync;
};

class EGLSyncMtl final : public EGLSyncImpl
{
  public:
    EGLSyncMtl(const egl::AttributeMap &attribs);
    ~EGLSyncMtl() override;

    void onDestroy(const egl::Display *display) override;

    egl::Error initialize(const egl::Display *display,
                          const gl::Context *context,
                          EGLenum type) override;
    egl::Error clientWait(const egl::Display *display,
                          const gl::Context *context,
                          EGLint flags,
                          EGLTime timeout,
                          EGLint *outResult) override;
    egl::Error serverWait(const egl::Display *display,
                          const gl::Context *context,
                          EGLint flags) override;
    egl::Error getStatus(const egl::Display *display, EGLint *outStatus) override;

    egl::Error dupNativeFenceFD(const egl::Display *display, EGLint *result) const override;

  private:
    mtl::Sync mSync;
};

}  // namespace rx

#endif