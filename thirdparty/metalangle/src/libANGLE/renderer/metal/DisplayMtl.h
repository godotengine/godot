//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DisplayMtl.h:
//    Defines the class interface for DisplayMtl, implementing DisplayImpl.
//

#ifndef LIBANGLE_RENDERER_METAL_DISPLAYMTL_H_
#define LIBANGLE_RENDERER_METAL_DISPLAYMTL_H_

#include "common/PackedEnums.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/DisplayImpl.h"
#include "libANGLE/renderer/metal/mtl_command_buffer.h"
#include "libANGLE/renderer/metal/mtl_format_utils.h"
#include "libANGLE/renderer/metal/mtl_render_utils.h"
#include "libANGLE/renderer/metal/mtl_state_cache.h"
#include "libANGLE/renderer/metal/mtl_utils.h"
#include "platform/FeaturesMtl.h"

namespace egl
{
class Surface;
}

namespace rx
{
class ContextMtl;

class DisplayMtl : public DisplayImpl
{
  public:
    DisplayMtl(const egl::DisplayState &state);
    ~DisplayMtl() override;

    egl::Error initialize(egl::Display *display) override;
    void terminate() override;

    bool testDeviceLost() override;
    egl::Error restoreLostDevice(const egl::Display *display) override;

    std::string getVendorString() const override;

    DeviceImpl *createDevice() override;

    egl::Error waitClient(const gl::Context *context) override;
    egl::Error waitNative(const gl::Context *context, EGLint engine) override;

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

    ImageImpl *createImage(const egl::ImageState &state,
                           const gl::Context *context,
                           EGLenum target,
                           const egl::AttributeMap &attribs) override;

    ContextImpl *createContext(const gl::State &state,
                               gl::ErrorSet *errorSet,
                               const egl::Config *configuration,
                               const gl::Context *shareContext,
                               const egl::AttributeMap &attribs) override;

    StreamProducerImpl *createStreamProducerD3DTexture(egl::Stream::ConsumerType consumerType,
                                                       const egl::AttributeMap &attribs) override;

    ExternalImageSiblingImpl *createExternalImageSibling(const gl::Context *context,
                                                         EGLenum target,
                                                         EGLClientBuffer buffer,
                                                         const egl::AttributeMap &attribs) override;
    gl::Version getMaxSupportedESVersion() const override;
    gl::Version getMaxConformantESVersion() const override;

    EGLSyncImpl *createSync(const egl::AttributeMap &attribs) override;

    egl::Error makeCurrent(egl::Surface *drawSurface,
                           egl::Surface *readSurface,
                           gl::Context *context) override;

    void populateFeatureList(angle::FeatureList *features) override;

    bool isValidNativeWindow(EGLNativeWindowType window) const override;

    egl::Error validateImageClientBuffer(const gl::Context *context,
                                         EGLenum target,
                                         EGLClientBuffer clientBuffer,
                                         const egl::AttributeMap &attribs) const override;
    egl::ConfigSet generateConfigs() override;

    std::string getRendererDescription() const;
    gl::Caps getNativeCaps() const;
    const gl::TextureCapsMap &getNativeTextureCaps() const;
    const gl::Extensions &getNativeExtensions() const;
    const gl::Limitations &getNativeLimitations() const { return mNativeLimitations; }
    const angle::FeaturesMtl &getFeatures() const { return mFeatures; }

    // Check whether either of the specified iOS or Mac GPU family is supported
    bool supportEitherGPUFamily(uint8_t iOSFamily, uint8_t macFamily) const;
    bool supportiOSGPUFamily(uint8_t iOSFamily) const;
    bool supportMacGPUFamily(uint8_t macFamily) const;

    id<MTLDevice> getMetalDevice() const { return mMetalDevice; }

    mtl::CommandQueue &cmdQueue() { return mCmdQueue; }
    const mtl::FormatTable &getFormatTable() const { return mFormatTable; }
    mtl::RenderUtils &getUtils() { return mUtils; }
    mtl::StateCache &getStateCache() { return mStateCache; }

    id<MTLLibrary> getDefaultShadersLib();

    id<MTLDepthStencilState> getDepthStencilState(const mtl::DepthStencilDesc &desc)
    {
        return mStateCache.getDepthStencilState(getMetalDevice(), desc);
    }
    id<MTLSamplerState> getSamplerState(const mtl::SamplerDesc &desc)
    {
        return mStateCache.getSamplerState(getMetalDevice(), desc);
    }

    const mtl::Format &getPixelFormat(angle::FormatID angleFormatId) const
    {
        return mFormatTable.getPixelFormat(angleFormatId);
    }
    const mtl::FormatCaps &getNativeFormatCaps(MTLPixelFormat mtlFormat) const
    {
        return mFormatTable.getNativeFormatCaps(mtlFormat);
    }

    // See mtl::FormatTable::getVertexFormat()
    const mtl::VertexFormat &getVertexFormat(angle::FormatID angleFormatId,
                                             bool tightlyPacked) const
    {
        return mFormatTable.getVertexFormat(angleFormatId, tightlyPacked);
    }
#if defined(__IPHONE_12_0) || defined(__MAC_10_14)
    mtl::AutoObjCObj<MTLSharedEventListener> getOrCreateSharedEventListener();
#endif
  protected:
    void generateExtensions(egl::DisplayExtensions *outExtensions) const override;
    void generateCaps(egl::Caps *outCaps) const override;

  private:
    angle::Result initializeImpl(egl::Display *display);
    void ensureCapsInitialized() const;
    void initializeCaps() const;
    void initializeExtensions() const;
    void initializeTextureCaps() const;
    void initializeFeatures();
    angle::Result initializeShaderLibrary();

    mtl::AutoObjCPtr<id<MTLDevice>> mMetalDevice = nil;

    mtl::CommandQueue mCmdQueue;

    mutable mtl::FormatTable mFormatTable;
    mtl::StateCache mStateCache;
    mtl::RenderUtils mUtils;

    // Built-in Shaders
    mtl::AutoObjCPtr<id<MTLLibrary>> mDefaultShaders = nil;
#if defined(__IPHONE_12_0) || defined(__MAC_10_14)
    mtl::AutoObjCObj<MTLSharedEventListener> mSharedEventListener;
#endif

    mutable bool mCapsInitialized;
    mutable gl::TextureCapsMap mNativeTextureCaps;
    mutable gl::Extensions mNativeExtensions;
    mutable gl::Caps mNativeCaps;
    mutable gl::Limitations mNativeLimitations;

    angle::FeaturesMtl mFeatures;
};

}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_DISPLAYMTL_H_ */
