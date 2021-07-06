//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// RenderbufferD3d.h: Defines the RenderbufferD3D class which implements RenderbufferImpl.

#ifndef LIBANGLE_RENDERER_D3D_RENDERBUFFERD3D_H_
#define LIBANGLE_RENDERER_D3D_RENDERBUFFERD3D_H_

#include "angle_gl.h"

#include "common/angleutils.h"
#include "libANGLE/renderer/RenderbufferImpl.h"

namespace rx
{
class EGLImageD3D;
class RendererD3D;
class RenderTargetD3D;
class SwapChainD3D;

class RenderbufferD3D : public RenderbufferImpl
{
  public:
    RenderbufferD3D(const gl::RenderbufferState &state, RendererD3D *renderer);
    ~RenderbufferD3D() override;

    void onDestroy(const gl::Context *context) override;

    angle::Result setStorage(const gl::Context *context,
                             GLenum internalformat,
                             size_t width,
                             size_t height) override;
    angle::Result setStorageMultisample(const gl::Context *context,
                                        size_t samples,
                                        GLenum internalformat,
                                        size_t width,
                                        size_t height) override;
    angle::Result setStorageEGLImageTarget(const gl::Context *context, egl::Image *image) override;

    angle::Result getRenderTarget(const gl::Context *context, RenderTargetD3D **outRenderTarget);
    angle::Result getAttachmentRenderTarget(const gl::Context *context,
                                            GLenum binding,
                                            const gl::ImageIndex &imageIndex,
                                            GLsizei samples,
                                            FramebufferAttachmentRenderTarget **rtOut) override;

    angle::Result initializeContents(const gl::Context *context,
                                     const gl::ImageIndex &imageIndex) override;

  private:
    RendererD3D *mRenderer;
    RenderTargetD3D *mRenderTarget;
    EGLImageD3D *mImage;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_RENDERBUFFERD3D_H_
