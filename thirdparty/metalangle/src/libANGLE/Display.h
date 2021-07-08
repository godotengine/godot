//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Display.h: Defines the egl::Display class, representing the abstract
// display on which graphics are drawn. Implements EGLDisplay.
// [EGL 1.4] section 2.1.2 page 3.

#ifndef LIBANGLE_DISPLAY_H_
#define LIBANGLE_DISPLAY_H_

#include <set>
#include <vector>

#include "libANGLE/AttributeMap.h"
#include "libANGLE/BlobCache.h"
#include "libANGLE/Caps.h"
#include "libANGLE/Config.h"
#include "libANGLE/Debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/LoggingAnnotator.h"
#include "libANGLE/MemoryProgramCache.h"
#include "libANGLE/Version.h"
#include "platform/Feature.h"
#include "platform/FrontendFeatures.h"

namespace gl
{
class Context;
class TextureManager;
}  // namespace gl

namespace rx
{
class DisplayImpl;
}

namespace egl
{
class Device;
class Image;
class Stream;
class Surface;
class Sync;
class Thread;

using SurfaceSet = std::set<Surface *>;

struct DisplayState final : private angle::NonCopyable
{
    DisplayState();
    ~DisplayState();

    EGLLabelKHR label;
    SurfaceSet surfaceSet;
    std::vector<std::string> featureOverridesEnabled;
    std::vector<std::string> featureOverridesDisabled;
};

// Constant coded here as a sanity limit.
constexpr EGLAttrib kProgramCacheSizeAbsoluteMax = 0x4000000;

class Display final : public LabeledObject, angle::NonCopyable
{
  public:
    ~Display() override;

    void setLabel(EGLLabelKHR label) override;
    EGLLabelKHR getLabel() const override;

    Error initialize();
    Error terminate(const Thread *thread);

    static Display *GetDisplayFromDevice(Device *device, const AttributeMap &attribMap);
    static Display *GetDisplayFromNativeDisplay(EGLNativeDisplayType nativeDisplay,
                                                const AttributeMap &attribMap);
    static Display *GetExistingDisplayFromNativeDisplay(EGLNativeDisplayType nativeDisplay);

    static const ClientExtensions &GetClientExtensions();
    static const std::string &GetClientExtensionString();

    std::vector<const Config *> getConfigs(const AttributeMap &attribs) const;
    std::vector<const Config *> chooseConfig(const AttributeMap &attribs) const;

    Error createWindowSurface(const Config *configuration,
                              EGLNativeWindowType window,
                              const AttributeMap &attribs,
                              Surface **outSurface);
    Error createPbufferSurface(const Config *configuration,
                               const AttributeMap &attribs,
                               Surface **outSurface);
    Error createPbufferFromClientBuffer(const Config *configuration,
                                        EGLenum buftype,
                                        EGLClientBuffer clientBuffer,
                                        const AttributeMap &attribs,
                                        Surface **outSurface);
    Error createPixmapSurface(const Config *configuration,
                              NativePixmapType nativePixmap,
                              const AttributeMap &attribs,
                              Surface **outSurface);

    Error createImage(const gl::Context *context,
                      EGLenum target,
                      EGLClientBuffer buffer,
                      const AttributeMap &attribs,
                      Image **outImage);

    Error createStream(const AttributeMap &attribs, Stream **outStream);

    Error createContext(const Config *configuration,
                        gl::Context *shareContext,
                        const EGLenum clientType,
                        const AttributeMap &attribs,
                        gl::Context **outContext);

    Error createSync(const gl::Context *currentContext,
                     EGLenum type,
                     const AttributeMap &attribs,
                     Sync **outSync);

    Error makeCurrent(const Thread *thread,
                      Surface *drawSurface,
                      Surface *readSurface,
                      gl::Context *context);

    Error destroySurface(Surface *surface);
    void destroyImage(Image *image);
    void destroyStream(Stream *stream);
    Error destroyContext(const Thread *thread, gl::Context *context);
    void destroySync(Sync *sync);

    bool isInitialized() const;
    bool isValidConfig(const Config *config) const;
    bool isValidContext(const gl::Context *context) const;
    bool isValidSurface(const Surface *surface) const;
    bool isValidImage(const Image *image) const;
    bool isValidStream(const Stream *stream) const;
    bool isValidSync(const Sync *sync) const;
    bool isValidNativeWindow(EGLNativeWindowType window) const;

    Error validateClientBuffer(const Config *configuration,
                               EGLenum buftype,
                               EGLClientBuffer clientBuffer,
                               const AttributeMap &attribs) const;
    Error validateImageClientBuffer(const gl::Context *context,
                                    EGLenum target,
                                    EGLClientBuffer clientBuffer,
                                    const egl::AttributeMap &attribs) const;

    static bool isValidDisplay(const Display *display);
    static bool isValidNativeDisplay(EGLNativeDisplayType display);
    static bool hasExistingWindowSurface(EGLNativeWindowType window);

    bool isDeviceLost() const;
    bool testDeviceLost();
    void notifyDeviceLost();

    void setBlobCacheFuncs(EGLSetBlobFuncANDROID set, EGLGetBlobFuncANDROID get);
    bool areBlobCacheFuncsSet() const { return mBlobCache.areBlobCacheFuncsSet(); }
    BlobCache &getBlobCache() { return mBlobCache; }

    static EGLClientBuffer GetNativeClientBuffer(const struct AHardwareBuffer *buffer);

    Error waitClient(const gl::Context *context);
    Error waitNative(const gl::Context *context, EGLint engine);

    const Caps &getCaps() const;

    const DisplayExtensions &getExtensions() const;
    const std::string &getExtensionString() const;
    const std::string &getVendorString() const;

    EGLint programCacheGetAttrib(EGLenum attrib) const;
    Error programCacheQuery(EGLint index,
                            void *key,
                            EGLint *keysize,
                            void *binary,
                            EGLint *binarysize);
    Error programCachePopulate(const void *key,
                               EGLint keysize,
                               const void *binary,
                               EGLint binarysize);
    EGLint programCacheResize(EGLint limit, EGLenum mode);

    const AttributeMap &getAttributeMap() const { return mAttributeMap; }
    EGLNativeDisplayType getNativeDisplayId() const { return mDisplayId; }

    rx::DisplayImpl *getImplementation() const { return mImplementation; }
    Device *getDevice() const;
    Surface *getWGLSurface() const;
    EGLenum getPlatform() const { return mPlatform; }

    gl::Version getMaxSupportedESVersion() const;

    const DisplayState &getState() const { return mState; }

    typedef std::set<gl::Context *> ContextSet;
    const ContextSet &getContextSet() { return mContextSet; }

    const angle::FrontendFeatures &getFrontendFeatures() { return mFrontendFeatures; }

    const angle::FeatureList &getFeatures() const { return mFeatures; }

    const char *queryStringi(const EGLint name, const EGLint index);

    EGLAttrib queryAttrib(const EGLint attribute);

  private:
    Display(EGLenum platform, EGLNativeDisplayType displayId, Device *eglDevice);

    void setAttributes(rx::DisplayImpl *impl, const AttributeMap &attribMap);

    Error restoreLostDevice();

    void initDisplayExtensions();
    void initVendorString();
    void initializeFrontendFeatures();

    DisplayState mState;
    rx::DisplayImpl *mImplementation;

    EGLNativeDisplayType mDisplayId;
    AttributeMap mAttributeMap;

    ConfigSet mConfigSet;

    ContextSet mContextSet;

    typedef std::set<Image *> ImageSet;
    ImageSet mImageSet;

    typedef std::set<Stream *> StreamSet;
    StreamSet mStreamSet;

    typedef std::set<Sync *> SyncSet;
    SyncSet mSyncSet;

    bool mInitialized;
    bool mDeviceLost;

    Caps mCaps;

    DisplayExtensions mDisplayExtensions;
    std::string mDisplayExtensionString;

    std::string mVendorString;

    Device *mDevice;
    Surface *mSurface;
    EGLenum mPlatform;
    angle::LoggingAnnotator mAnnotator;

    gl::TextureManager *mTextureManager;
    BlobCache mBlobCache;
    gl::MemoryProgramCache mMemoryProgramCache;
    size_t mGlobalTextureShareGroupUsers;

    angle::FrontendFeatures mFrontendFeatures;

    angle::FeatureList mFeatures;
};

}  // namespace egl

#endif  // LIBANGLE_DISPLAY_H_
