//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// DisplayCGL.h: CGL implementation of egl::Display

#ifndef LIBANGLE_RENDERER_GL_CGL_DISPLAYCGL_H_
#define LIBANGLE_RENDERER_GL_CGL_DISPLAYCGL_H_

#include "libANGLE/renderer/gl/DisplayGL.h"

struct _CGLContextObject;
typedef _CGLContextObject *CGLContextObj;

struct _CGLPixelFormatObject;
typedef _CGLPixelFormatObject *CGLPixelFormatObj;

namespace rx
{

class WorkerContext;

class DisplayCGL : public DisplayGL
{
  public:
    DisplayCGL(const egl::DisplayState &state);
    ~DisplayCGL() override;

    egl::Error initialize(egl::Display *display) override;
    void terminate() override;

    SurfaceImpl *createWindowSurface(const egl::SurfaceState &state,
                                     EGLNativeWindowType window,
                                     const egl::AttributeMap &attribs) override;
    SurfaceImpl *createPbufferSurface(const egl::SurfaceState &state,
                                      const egl::AttributeMap &attribs) override;
    SurfaceImpl *createPbufferFromClientBuffer(const egl::SurfaceState &state,
                                               EGLenum buftype,
                                               EGLClientBuffer clientBuffer,
                                               const egl::AttributeMap &attribs) override;
    SurfaceImpl *createPixmapSurface(const egl::SurfaceState &state,
                                     NativePixmapType nativePixmap,
                                     const egl::AttributeMap &attribs) override;

    ContextImpl *createContext(const gl::State &state,
                               gl::ErrorSet *errorSet,
                               const egl::Config *configuration,
                               const gl::Context *shareContext,
                               const egl::AttributeMap &attribs) override;

    egl::ConfigSet generateConfigs() override;

    bool testDeviceLost() override;
    egl::Error restoreLostDevice(const egl::Display *display) override;

    bool isValidNativeWindow(EGLNativeWindowType window) const override;
    egl::Error validateClientBuffer(const egl::Config *configuration,
                                    EGLenum buftype,
                                    EGLClientBuffer clientBuffer,
                                    const egl::AttributeMap &attribs) const override;

    DeviceImpl *createDevice() override;

    std::string getVendorString() const override;

    egl::Error waitClient(const gl::Context *context) override;
    egl::Error waitNative(const gl::Context *context, EGLint engine) override;

    gl::Version getMaxSupportedESVersion() const override;

    CGLContextObj getCGLContext() const;
    CGLPixelFormatObj getCGLPixelFormat() const;

    WorkerContext *createWorkerContext(std::string *infoLog);

    void initializeFrontendFeatures(angle::FrontendFeatures *features) const override;

    void populateFeatureList(angle::FeatureList *features) override;

    // Support for dual-GPU MacBook Pros. If the context was created
    // preferring the high-power GPU, unreference that GPU during
    // context destruction.
    void unreferenceDiscreteGPU();

  private:
    egl::Error makeCurrentSurfaceless(gl::Context *context) override;

    void generateExtensions(egl::DisplayExtensions *outExtensions) const override;
    void generateCaps(egl::Caps *outCaps) const override;

    std::shared_ptr<RendererGL> mRenderer;

    egl::Display *mEGLDisplay;
    CGLContextObj mContext;
    CGLPixelFormatObj mPixelFormat;
    bool mSupportsGPUSwitching;
    CGLPixelFormatObj mDiscreteGPUPixelFormat;
    int mDiscreteGPURefs;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_CGL_DISPLAYCGL_H_
