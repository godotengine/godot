//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// DisplayCGL.mm: CGL implementation of egl::Display

#if __has_include(<Cocoa/Cocoa.h>)

#    include "libANGLE/renderer/gl/cgl/DisplayCGL.h"

#    import <Cocoa/Cocoa.h>
#    include <EGL/eglext.h>
#    include <dlfcn.h>

#    include "common/debug.h"
#    include "gpu_info_util/SystemInfo.h"
#    include "libANGLE/Display.h"
#    include "libANGLE/renderer/gl/cgl/ContextCGL.h"
#    include "libANGLE/renderer/gl/cgl/DeviceCGL.h"
#    include "libANGLE/renderer/gl/cgl/IOSurfaceSurfaceCGL.h"
#    include "libANGLE/renderer/gl/cgl/PbufferSurfaceCGL.h"
#    include "libANGLE/renderer/gl/cgl/RendererCGL.h"
#    include "libANGLE/renderer/gl/cgl/WindowSurfaceCGL.h"

namespace
{

const char *kDefaultOpenGLDylibName =
    "/System/Library/Frameworks/OpenGL.framework/Libraries/libGL.dylib";
const char *kFallbackOpenGLDylibName = "GL";

}

namespace rx
{

class FunctionsGLCGL : public FunctionsGL
{
  public:
    FunctionsGLCGL(void *dylibHandle) : mDylibHandle(dylibHandle) {}

    ~FunctionsGLCGL() override { dlclose(mDylibHandle); }

  private:
    void *loadProcAddress(const std::string &function) const override
    {
        return dlsym(mDylibHandle, function.c_str());
    }

    void *mDylibHandle;
};

DisplayCGL::DisplayCGL(const egl::DisplayState &state)
    : DisplayGL(state),
      mEGLDisplay(nullptr),
      mContext(nullptr),
      mPixelFormat(nullptr),
      mSupportsGPUSwitching(false),
      mDiscreteGPUPixelFormat(nullptr),
      mDiscreteGPURefs(0)
{}

DisplayCGL::~DisplayCGL() {}

egl::Error DisplayCGL::initialize(egl::Display *display)
{
    mEGLDisplay = display;

    angle::SystemInfo info;
    if (!angle::GetSystemInfo(&info))
    {
        return egl::EglNotInitialized() << "Unable to query ANGLE's SystemInfo.";
    }
    mSupportsGPUSwitching = info.isMacSwitchable;

    {
        // TODO(cwallez) investigate which pixel format we want
        std::vector<CGLPixelFormatAttribute> attribs;
        attribs.push_back(kCGLPFAOpenGLProfile);
        attribs.push_back(static_cast<CGLPixelFormatAttribute>(kCGLOGLPVersion_3_2_Core));
        if (mSupportsGPUSwitching)
        {
            attribs.push_back(kCGLPFAAllowOfflineRenderers);
        }
        attribs.push_back(static_cast<CGLPixelFormatAttribute>(0));
        GLint nVirtualScreens = 0;
        CGLChoosePixelFormat(attribs.data(), &mPixelFormat, &nVirtualScreens);

        if (mPixelFormat == nullptr)
        {
            return egl::EglNotInitialized() << "Could not create the context's pixel format.";
        }
    }

    CGLCreateContext(mPixelFormat, nullptr, &mContext);
    if (mContext == nullptr)
    {
        return egl::EglNotInitialized() << "Could not create the CGL context.";
    }
    CGLSetCurrentContext(mContext);

    // There is no equivalent getProcAddress in CGL so we open the dylib directly
    void *handle = dlopen(kDefaultOpenGLDylibName, RTLD_NOW);
    if (!handle)
    {
        handle = dlopen(kFallbackOpenGLDylibName, RTLD_NOW);
    }
    if (!handle)
    {
        return egl::EglNotInitialized() << "Could not open the OpenGL Framework.";
    }

    std::unique_ptr<FunctionsGL> functionsGL(new FunctionsGLCGL(handle));
    functionsGL->initialize(display->getAttributeMap());

    mRenderer.reset(new RendererCGL(std::move(functionsGL), display->getAttributeMap(), this));

    const gl::Version &maxVersion = mRenderer->getMaxSupportedESVersion();
    if (maxVersion < gl::Version(2, 0))
    {
        return egl::EglNotInitialized() << "OpenGL ES 2.0 is not supportable.";
    }

    return DisplayGL::initialize(display);
}

void DisplayCGL::terminate()
{
    DisplayGL::terminate();

    mRenderer.reset();
    if (mPixelFormat != nullptr)
    {
        CGLDestroyPixelFormat(mPixelFormat);
        mPixelFormat = nullptr;
    }
    if (mContext != nullptr)
    {
        CGLSetCurrentContext(nullptr);
        CGLReleaseContext(mContext);
        mContext = nullptr;
    }
}

SurfaceImpl *DisplayCGL::createWindowSurface(const egl::SurfaceState &state,
                                             EGLNativeWindowType window,
                                             const egl::AttributeMap &attribs)
{
    return new WindowSurfaceCGL(state, mRenderer.get(), window, mContext);
}

SurfaceImpl *DisplayCGL::createPbufferSurface(const egl::SurfaceState &state,
                                              const egl::AttributeMap &attribs)
{
    EGLint width  = static_cast<EGLint>(attribs.get(EGL_WIDTH, 0));
    EGLint height = static_cast<EGLint>(attribs.get(EGL_HEIGHT, 0));
    return new PbufferSurfaceCGL(state, mRenderer.get(), width, height);
}

SurfaceImpl *DisplayCGL::createPbufferFromClientBuffer(const egl::SurfaceState &state,
                                                       EGLenum buftype,
                                                       EGLClientBuffer clientBuffer,
                                                       const egl::AttributeMap &attribs)
{
    ASSERT(buftype == EGL_IOSURFACE_ANGLE);

    return new IOSurfaceSurfaceCGL(state, mContext, clientBuffer, attribs);
}

SurfaceImpl *DisplayCGL::createPixmapSurface(const egl::SurfaceState &state,
                                             NativePixmapType nativePixmap,
                                             const egl::AttributeMap &attribs)
{
    UNIMPLEMENTED();
    return nullptr;
}

ContextImpl *DisplayCGL::createContext(const gl::State &state,
                                       gl::ErrorSet *errorSet,
                                       const egl::Config *configuration,
                                       const gl::Context *shareContext,
                                       const egl::AttributeMap &attribs)
{
    bool usesDiscreteGPU = false;

    if (attribs.get(EGL_POWER_PREFERENCE_ANGLE, EGL_LOW_POWER_ANGLE) == EGL_HIGH_POWER_ANGLE)
    {
        // Should have been rejected by validation if not supported.
        ASSERT(mSupportsGPUSwitching);
        // Create discrete pixel format if necessary.
        if (!mDiscreteGPUPixelFormat)
        {
            CGLPixelFormatAttribute discreteAttribs[] = {static_cast<CGLPixelFormatAttribute>(0)};
            GLint numPixelFormats                     = 0;
            if (CGLChoosePixelFormat(discreteAttribs, &mDiscreteGPUPixelFormat, &numPixelFormats) !=
                kCGLNoError)
            {
                ERR() << "Error choosing discrete pixel format.";
                return nullptr;
            }
        }
        ++mDiscreteGPURefs;
        usesDiscreteGPU = true;
    }

    return new ContextCGL(state, errorSet, mRenderer, usesDiscreteGPU);
}

DeviceImpl *DisplayCGL::createDevice()
{
    return new DeviceCGL();
}

egl::ConfigSet DisplayCGL::generateConfigs()
{
    // TODO(cwallez): generate more config permutations
    egl::ConfigSet configs;

    const gl::Version &maxVersion = getMaxSupportedESVersion();
    ASSERT(maxVersion >= gl::Version(2, 0));
    bool supportsES3 = maxVersion >= gl::Version(3, 0);

    egl::Config config;

    // Native stuff
    config.nativeVisualID   = 0;
    config.nativeVisualType = 0;
    config.nativeRenderable = EGL_TRUE;

    // Buffer sizes
    config.redSize     = 8;
    config.greenSize   = 8;
    config.blueSize    = 8;
    config.alphaSize   = 8;
    config.depthSize   = 24;
    config.stencilSize = 8;

    config.colorBufferType = EGL_RGB_BUFFER;
    config.luminanceSize   = 0;
    config.alphaMaskSize   = 0;

    config.bufferSize = config.redSize + config.greenSize + config.blueSize + config.alphaSize;

    config.transparentType = EGL_NONE;

    // Pbuffer
    config.maxPBufferWidth  = 4096;
    config.maxPBufferHeight = 4096;
    config.maxPBufferPixels = 4096 * 4096;

    // Caveat
    config.configCaveat = EGL_NONE;

    // Misc
    config.sampleBuffers     = 0;
    config.samples           = 0;
    config.level             = 0;
    config.bindToTextureRGB  = EGL_FALSE;
    config.bindToTextureRGBA = EGL_FALSE;

    config.surfaceType = EGL_WINDOW_BIT | EGL_PBUFFER_BIT;

    config.minSwapInterval = 1;
    config.maxSwapInterval = 1;

    config.renderTargetFormat = GL_RGBA8;
    config.depthStencilFormat = GL_DEPTH24_STENCIL8;

    config.conformant     = EGL_OPENGL_ES2_BIT | (supportsES3 ? EGL_OPENGL_ES3_BIT_KHR : 0);
    config.renderableType = config.conformant;

    config.matchNativePixmap = EGL_NONE;

    config.colorComponentType = EGL_COLOR_COMPONENT_TYPE_FIXED_EXT;

    configs.add(config);
    return configs;
}

bool DisplayCGL::testDeviceLost()
{
    // TODO(cwallez) investigate implementing this
    return false;
}

egl::Error DisplayCGL::restoreLostDevice(const egl::Display *display)
{
    UNIMPLEMENTED();
    return egl::EglBadDisplay();
}

bool DisplayCGL::isValidNativeWindow(EGLNativeWindowType window) const
{
    NSObject *layer = reinterpret_cast<NSObject *>(window);
    return [layer isKindOfClass:[CALayer class]];
}

egl::Error DisplayCGL::validateClientBuffer(const egl::Config *configuration,
                                            EGLenum buftype,
                                            EGLClientBuffer clientBuffer,
                                            const egl::AttributeMap &attribs) const
{
    ASSERT(buftype == EGL_IOSURFACE_ANGLE);

    if (!IOSurfaceSurfaceCGL::validateAttributes(clientBuffer, attribs))
    {
        return egl::EglBadAttribute();
    }

    return egl::NoError();
}

std::string DisplayCGL::getVendorString() const
{
    // TODO(cwallez) find a useful vendor string
    return "";
}

CGLContextObj DisplayCGL::getCGLContext() const
{
    return mContext;
}

CGLPixelFormatObj DisplayCGL::getCGLPixelFormat() const
{
    return mPixelFormat;
}

void DisplayCGL::generateExtensions(egl::DisplayExtensions *outExtensions) const
{
    outExtensions->iosurfaceClientBuffer = true;
    outExtensions->surfacelessContext    = true;
    outExtensions->deviceQuery           = true;

    // Contexts are virtualized so textures can be shared globally
    outExtensions->displayTextureShareGroup = true;

    // Support any surface & context combination
    outExtensions->flexibleSurfaceCompatibility = true;

    if (mSupportsGPUSwitching)
    {
        outExtensions->powerPreference = true;
    }

    DisplayGL::generateExtensions(outExtensions);
}

void DisplayCGL::generateCaps(egl::Caps *outCaps) const
{
    outCaps->textureNPOT = true;
}

egl::Error DisplayCGL::waitClient(const gl::Context *context)
{
    // TODO(cwallez) UNIMPLEMENTED()
    return egl::NoError();
}

egl::Error DisplayCGL::waitNative(const gl::Context *context, EGLint engine)
{
    // TODO(cwallez) UNIMPLEMENTED()
    return egl::NoError();
}

gl::Version DisplayCGL::getMaxSupportedESVersion() const
{
    return mRenderer->getMaxSupportedESVersion();
}

egl::Error DisplayCGL::makeCurrentSurfaceless(gl::Context *context)
{
    // We have nothing to do as mContext is always current, and that CGL is surfaceless by
    // default.
    return egl::NoError();
}

class WorkerContextCGL final : public WorkerContext
{
  public:
    WorkerContextCGL(CGLContextObj context);
    ~WorkerContextCGL() override;

    bool makeCurrent() override;
    void unmakeCurrent() override;

  private:
    CGLContextObj mContext;
};

WorkerContextCGL::WorkerContextCGL(CGLContextObj context) : mContext(context) {}

WorkerContextCGL::~WorkerContextCGL()
{
    CGLSetCurrentContext(nullptr);
    CGLReleaseContext(mContext);
    mContext = nullptr;
}

bool WorkerContextCGL::makeCurrent()
{
    CGLError error = CGLSetCurrentContext(mContext);
    if (error != kCGLNoError)
    {
        ERR() << "Unable to make gl context current.";
        return false;
    }
    return true;
}

void WorkerContextCGL::unmakeCurrent()
{
    CGLSetCurrentContext(nullptr);
}

WorkerContext *DisplayCGL::createWorkerContext(std::string *infoLog)
{
    CGLContextObj context = nullptr;
    CGLCreateContext(mPixelFormat, mContext, &context);
    if (context == nullptr)
    {
        *infoLog += "Could not create the CGL context.";
        return nullptr;
    }

    return new WorkerContextCGL(context);
}

void DisplayCGL::unreferenceDiscreteGPU()
{
    ASSERT(mDiscreteGPURefs > 0);
    if (--mDiscreteGPURefs == 0)
    {
        CGLDestroyPixelFormat(mDiscreteGPUPixelFormat);
        mDiscreteGPUPixelFormat = nullptr;
    }
}

void DisplayCGL::initializeFrontendFeatures(angle::FrontendFeatures *features) const
{
    mRenderer->initializeFrontendFeatures(features);
}

void DisplayCGL::populateFeatureList(angle::FeatureList *features)
{
    mRenderer->getFeatures().populateFeatureList(features);
}
}

#endif  // __has_include(<Cocoa/Cocoa.h>)
