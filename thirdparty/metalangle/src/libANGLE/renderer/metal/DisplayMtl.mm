//
// Copyright (c) 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// DisplayMtl.mm: Metal implementation of DisplayImpl

#include "libANGLE/renderer/metal/DisplayMtl.h"

#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/Surface.h"
#include "libANGLE/renderer/DeviceImpl.h"
#include "libANGLE/renderer/glslang_wrapper_utils.h"
#include "libANGLE/renderer/metal/ContextMtl.h"
#include "libANGLE/renderer/metal/ImageMtl.h"
#include "libANGLE/renderer/metal/SurfaceMtl.h"
#include "libANGLE/renderer/metal/SyncMtl.h"
#include "libANGLE/renderer/metal/mtl_common.h"
#include "libANGLE/renderer/metal/shaders/compiled/mtl_default_shaders.inc"
#include "platform/Platform.h"

#include "EGL/eglext.h"

namespace rx
{

bool IsMetalDisplayAvailable()
{
    // We only support macos 10.13+ and 11 for now. Since they are requirements for Metal 2.0.
#if TARGET_OS_SIMULATOR
    if (ANGLE_APPLE_AVAILABLE_XCI(10.13, 13.0, 13))
#else
    if (ANGLE_APPLE_AVAILABLE_XCI(10.13, 13.0, 11))
#endif
    {
        return true;
    }
    return false;
}

DisplayImpl *CreateMetalDisplay(const egl::DisplayState &state)
{
    return new DisplayMtl(state);
}

// DeviceMTL implementation, implements DeviceImpl
class DeviceMTL : public DeviceImpl
{
  public:
    DeviceMTL() {}
    ~DeviceMTL() override {}

    egl::Error initialize() override { return egl::NoError(); }
    egl::Error getAttribute(const egl::Display *display, EGLint attribute, void **outValue) override
    {
        DisplayMtl *displayImpl = mtl::GetImpl(display);

        switch (attribute)
        {
            case EGL_MTL_DEVICE_ANGLE:
                *outValue = displayImpl->getMetalDevice();
                break;
            default:
                return egl::EglBadAttribute();
        }

        return egl::NoError();
    }
    EGLint getType() override { return 0; }
    void generateExtensions(egl::DeviceExtensions *outExtensions) const override
    {
        outExtensions->deviceMTL = true;
    }
};

// DisplayMtl implementation
DisplayMtl::DisplayMtl(const egl::DisplayState &state)
    : DisplayImpl(state), mStateCache(mFeatures), mUtils(this)
{}

DisplayMtl::~DisplayMtl() {}

egl::Error DisplayMtl::initialize(egl::Display *display)
{
    ASSERT(IsMetalDisplayAvailable());

    angle::Result result = initializeImpl(display);
    if (result != angle::Result::Continue)
    {
        return egl::EglNotInitialized();
    }
    return egl::NoError();
}

angle::Result DisplayMtl::initializeImpl(egl::Display *display)
{
    ANGLE_MTL_OBJC_SCOPE
    {
        mMetalDevice = [MTLCreateSystemDefaultDevice() ANGLE_MTL_AUTORELEASE];
        if (!mMetalDevice)
        {
            return angle::Result::Stop;
        }

        mCmdQueue.set([[mMetalDevice.get() newCommandQueue] ANGLE_MTL_AUTORELEASE]);

        mCapsInitialized = false;

        GlslangInitialize();
        initializeFeatures();

        ANGLE_TRY(mFormatTable.initialize(this));

        return mUtils.initialize();
    }
}

void DisplayMtl::terminate()
{
    mUtils.onDestroy();
    mCmdQueue.reset();
    mDefaultShaders = nil;
    mMetalDevice    = nil;
#if defined(__IPHONE_12_0) || defined(__MAC_10_14)
    mSharedEventListener = nil;
#endif
    mCapsInitialized = false;

    GlslangRelease();
}

bool DisplayMtl::testDeviceLost()
{
    return false;
}

egl::Error DisplayMtl::restoreLostDevice(const egl::Display *display)
{
    return egl::NoError();
}

std::string DisplayMtl::getVendorString() const
{
    ANGLE_MTL_OBJC_SCOPE
    {
        std::string vendorString = "Google Inc.";
        if (mMetalDevice)
        {
            vendorString += " Metal Renderer: ";
            vendorString += mMetalDevice.get().name.UTF8String;
        }

        return vendorString;
    }
}

DeviceImpl *DisplayMtl::createDevice()
{
    return new DeviceMTL();
}

egl::Error DisplayMtl::waitClient(const gl::Context *context)
{
    auto contextMtl      = GetImplAs<ContextMtl>(context);
    angle::Result result = contextMtl->finishCommandBuffer();

    if (result != angle::Result::Continue)
    {
        return egl::EglBadAccess();
    }
    return egl::NoError();
}

egl::Error DisplayMtl::waitNative(const gl::Context *context, EGLint engine)
{
    UNIMPLEMENTED();
    return egl::EglBadAccess();
}

SurfaceImpl *DisplayMtl::createWindowSurface(const egl::SurfaceState &state,
                                             EGLNativeWindowType window,
                                             const egl::AttributeMap &attribs)
{
    return new WindowSurfaceMtl(this, state, window, attribs);
}

SurfaceImpl *DisplayMtl::createPbufferSurface(const egl::SurfaceState &state,
                                              const egl::AttributeMap &attribs)
{
    return new PBufferSurfaceMtl(this, state, attribs);
}

SurfaceImpl *DisplayMtl::createPbufferFromClientBuffer(const egl::SurfaceState &state,
                                                       EGLenum buftype,
                                                       EGLClientBuffer clientBuffer,
                                                       const egl::AttributeMap &attribs)
{
    UNIMPLEMENTED();
    return static_cast<SurfaceImpl *>(0);
}

SurfaceImpl *DisplayMtl::createPixmapSurface(const egl::SurfaceState &state,
                                             NativePixmapType nativePixmap,
                                             const egl::AttributeMap &attribs)
{
    UNIMPLEMENTED();
    return static_cast<SurfaceImpl *>(0);
}

ImageImpl *DisplayMtl::createImage(const egl::ImageState &state,
                                   const gl::Context *context,
                                   EGLenum target,
                                   const egl::AttributeMap &attribs)
{
    return new ImageMtl(state, context);
}

rx::ContextImpl *DisplayMtl::createContext(const gl::State &state,
                                           gl::ErrorSet *errorSet,
                                           const egl::Config *configuration,
                                           const gl::Context *shareContext,
                                           const egl::AttributeMap &attribs)
{
    return new ContextMtl(state, errorSet, this);
}

StreamProducerImpl *DisplayMtl::createStreamProducerD3DTexture(
    egl::Stream::ConsumerType consumerType,
    const egl::AttributeMap &attribs)
{
    UNIMPLEMENTED();
    return nullptr;
}

ExternalImageSiblingImpl *DisplayMtl::createExternalImageSibling(const gl::Context *context,
                                                                 EGLenum target,
                                                                 EGLClientBuffer buffer,
                                                                 const egl::AttributeMap &attribs)
{
    switch (target)
    {
        case EGL_MTL_TEXTURE_MGL:
            return new TextureImageSiblingMtl(buffer);

        default:
            UNREACHABLE();
            return nullptr;
    }
}

gl::Version DisplayMtl::getMaxSupportedESVersion() const
{
    return gl::Version(3, 0);
}

gl::Version DisplayMtl::getMaxConformantESVersion() const
{
    return std::min(getMaxSupportedESVersion(), gl::Version(2, 0));
}

EGLSyncImpl *DisplayMtl::createSync(const egl::AttributeMap &attribs)
{
    return new EGLSyncMtl(attribs);
}

egl::Error DisplayMtl::makeCurrent(egl::Surface *drawSurface,
                                   egl::Surface *readSurface,
                                   gl::Context *context)
{
    if (!context)
    {
        return egl::NoError();
    }

    return egl::NoError();
}

void DisplayMtl::generateExtensions(egl::DisplayExtensions *outExtensions) const
{
    outExtensions->flexibleSurfaceCompatibility = true;
    outExtensions->surfacelessContext           = true;
    outExtensions->glColorspace                 = true;
    outExtensions->mtlTextureClientBuffer       = true;
    outExtensions->deviceQuery                  = true;

    if (ANGLE_APPLE_AVAILABLE_XCI(10.14, 13.0, 12.0))
    {
        // MTLSharedEvent is only available since Metal 2.1
        outExtensions->fenceSync = true;
        outExtensions->waitSync  = true;
    }

    // EGL_KHR_image
    outExtensions->image     = true;
    outExtensions->imageBase = true;
}

void DisplayMtl::generateCaps(egl::Caps *outCaps) const {}

void DisplayMtl::populateFeatureList(angle::FeatureList *features) {}

egl::ConfigSet DisplayMtl::generateConfigs()
{
    // NOTE(hqle): generate more config permutations
    egl::ConfigSet configs;

    const gl::Version &maxVersion = getMaxSupportedESVersion();
    ASSERT(maxVersion >= gl::Version(2, 0));
    bool supportsES3 = maxVersion >= gl::Version(3, 0);

    egl::Config config;

    // Native stuff
    config.nativeVisualID   = 0;
    config.nativeVisualType = 0;
    config.nativeRenderable = EGL_TRUE;

    config.colorBufferType = EGL_RGB_BUFFER;
    config.luminanceSize   = 0;
    config.alphaMaskSize   = 0;

    config.transparentType = EGL_NONE;

    // Pbuffer
    config.maxPBufferWidth  = 4096;
    config.maxPBufferHeight = 4096;
    config.maxPBufferPixels = 4096 * 4096;

    // Caveat
    config.configCaveat = EGL_NONE;

    // Misc
    config.sampleBuffers     = 0;
    config.level             = 0;
    config.bindToTextureRGB  = EGL_FALSE;
    config.bindToTextureRGBA = EGL_TRUE;

    config.surfaceType = EGL_WINDOW_BIT | EGL_PBUFFER_BIT | EGL_SWAP_BEHAVIOR_PRESERVED_BIT;

#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
    config.minSwapInterval = 0;
    config.maxSwapInterval = 1;
#else
    config.minSwapInterval = 1;
    config.maxSwapInterval = 1;
#endif

    config.renderTargetFormat = GL_RGBA8;
    config.depthStencilFormat = GL_DEPTH24_STENCIL8;

    config.conformant     = EGL_OPENGL_ES2_BIT | (supportsES3 ? EGL_OPENGL_ES3_BIT_KHR : 0);
    config.renderableType = config.conformant;

    config.matchNativePixmap = EGL_NONE;

    config.colorComponentType = EGL_COLOR_COMPONENT_TYPE_FIXED_EXT;

    constexpr int samplesSupported[] = {0, 4};

    for (int samples : samplesSupported)
    {
        config.samples       = samples;
        config.sampleBuffers = (samples == 0) ? 0 : 1;

        // Buffer sizes
        config.redSize    = 8;
        config.greenSize  = 8;
        config.blueSize   = 8;
        config.alphaSize  = 8;
        config.bufferSize = config.redSize + config.greenSize + config.blueSize + config.alphaSize;

        // With DS
        config.depthSize   = 24;
        config.stencilSize = 8;

        configs.add(config);

        // With D
        config.depthSize   = 24;
        config.stencilSize = 0;
        configs.add(config);

        // With S
        config.depthSize   = 0;
        config.stencilSize = 8;
        configs.add(config);

        // No DS
        config.depthSize   = 0;
        config.stencilSize = 0;
        configs.add(config);
    }

    return configs;
}

bool DisplayMtl::isValidNativeWindow(EGLNativeWindowType window) const
{
    NSObject *layer = (__bridge NSObject *)(window);
    return [layer isKindOfClass:[CALayer class]];
}

egl::Error DisplayMtl::validateImageClientBuffer(const gl::Context *context,
                                                 EGLenum target,
                                                 EGLClientBuffer clientBuffer,
                                                 const egl::AttributeMap &attribs) const
{
    switch (target)
    {
        case EGL_MTL_TEXTURE_MGL:
            if (!TextureImageSiblingMtl::ValidateClientBuffer(this, clientBuffer))
            {
                return egl::EglBadAttribute();
            }
            break;
        default:
            UNREACHABLE();
            return egl::EglBadAttribute();
    }
    return egl::NoError();
}

std::string DisplayMtl::getRendererDescription() const
{
    ANGLE_MTL_OBJC_SCOPE
    {
        std::string desc = "Metal Renderer";

        if (mMetalDevice)
        {
            desc += ": ";
            desc += mMetalDevice.get().name.UTF8String;
        }

        return desc;
    }
}

gl::Caps DisplayMtl::getNativeCaps() const
{
    ensureCapsInitialized();
    return mNativeCaps;
}
const gl::TextureCapsMap &DisplayMtl::getNativeTextureCaps() const
{
    ensureCapsInitialized();
    return mNativeTextureCaps;
}
const gl::Extensions &DisplayMtl::getNativeExtensions() const
{
    ensureCapsInitialized();
    return mNativeExtensions;
}

void DisplayMtl::ensureCapsInitialized() const
{
    if (mCapsInitialized)
    {
        return;
    }

    mCapsInitialized = true;

    // Reset
    mNativeCaps = gl::Caps();

    // Fill extension and texture caps
    initializeExtensions();
    initializeTextureCaps();

    // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
    mNativeCaps.maxElementIndex  = std::numeric_limits<GLuint>::max() - 1;
    mNativeCaps.max3DTextureSize = 2048;
#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
    mNativeCaps.max2DTextureSize = 16384;
    // On macOS exclude [[position]] from maxVaryingVectors.
    mNativeCaps.maxVaryingVectors         = 31 - 1;
    mNativeCaps.maxVertexOutputComponents = mNativeCaps.maxFragmentInputComponents = 124 - 4;
#else
    if (supportiOSGPUFamily(3))
    {
        mNativeCaps.max2DTextureSize          = 16384;
        mNativeCaps.maxVertexOutputComponents = mNativeCaps.maxFragmentInputComponents = 124;
        mNativeCaps.maxVaryingVectors = mNativeCaps.maxVertexOutputComponents / 4;
    }
    else
    {
        mNativeCaps.max2DTextureSize          = 8192;
        mNativeCaps.maxVertexOutputComponents = mNativeCaps.maxFragmentInputComponents = 60;
        mNativeCaps.maxVaryingVectors = mNativeCaps.maxVertexOutputComponents / 4;
    }
#endif

    mNativeCaps.maxArrayTextureLayers = 2048;
    mNativeCaps.maxLODBias            = 2.0;
    mNativeCaps.maxCubeMapTextureSize = mNativeCaps.max2DTextureSize;
    mNativeCaps.maxRenderbufferSize   = mNativeCaps.max2DTextureSize;
    mNativeCaps.minAliasedPointSize   = 1;
    // NOTE(hqle): Metal has some problems drawing big point size even though
    // Metal-Feature-Set-Tables.pdf says that max supported point size is 511. We limit it to 255
    // on Intel and 64 on AMD for now.
    if ([mMetalDevice.get().name rangeOfString:@"Intel"].location != NSNotFound)
    {
        mNativeCaps.maxAliasedPointSize = 255;
    }
    else
    {
        mNativeCaps.maxAliasedPointSize = 64;
    }

    mNativeCaps.minAliasedLineWidth = 1.0f;
    mNativeCaps.maxAliasedLineWidth = 1.0f;

    mNativeCaps.maxDrawBuffers       = mtl::kMaxRenderTargets;
    mNativeCaps.maxFramebufferWidth  = mNativeCaps.max2DTextureSize;
    mNativeCaps.maxFramebufferHeight = mNativeCaps.max2DTextureSize;
    mNativeCaps.maxColorAttachments  = mtl::kMaxRenderTargets;
    mNativeCaps.maxViewportWidth     = mNativeCaps.max2DTextureSize;
    mNativeCaps.maxViewportHeight    = mNativeCaps.max2DTextureSize;

    // MSAA
    mNativeCaps.maxSamples             = mFormatTable.getMaxSamples();
    mNativeCaps.maxSampleMaskWords     = 0;
    mNativeCaps.maxColorTextureSamples = mNativeCaps.maxSamples;
    mNativeCaps.maxDepthTextureSamples = mNativeCaps.maxSamples;
    mNativeCaps.maxIntegerSamples      = 1;

    mNativeCaps.maxVertexAttributes           = mtl::kMaxVertexAttribs;
    mNativeCaps.maxVertexAttribBindings       = mtl::kMaxVertexAttribs;
    mNativeCaps.maxVertexAttribRelativeOffset = std::numeric_limits<GLint>::max();
    mNativeCaps.maxVertexAttribStride         = std::numeric_limits<GLint>::max();

    mNativeCaps.maxElementsIndices  = std::numeric_limits<GLint>::max();
    mNativeCaps.maxElementsVertices = std::numeric_limits<GLint>::max();

    // Looks like all floats are IEEE according to the docs here:
    mNativeCaps.vertexHighpFloat.setIEEEFloat();
    mNativeCaps.vertexMediumpFloat.setIEEEFloat();
    mNativeCaps.vertexLowpFloat.setIEEEFloat();
    mNativeCaps.fragmentHighpFloat.setIEEEFloat();
    mNativeCaps.fragmentMediumpFloat.setIEEEFloat();
    mNativeCaps.fragmentLowpFloat.setIEEEFloat();

    mNativeCaps.vertexHighpInt.setTwosComplementInt(32);
    mNativeCaps.vertexMediumpInt.setTwosComplementInt(32);
    mNativeCaps.vertexLowpInt.setTwosComplementInt(32);
    mNativeCaps.fragmentHighpInt.setTwosComplementInt(32);
    mNativeCaps.fragmentMediumpInt.setTwosComplementInt(32);
    mNativeCaps.fragmentLowpInt.setTwosComplementInt(32);

    // Uniform limits
    GLuint maxDefaultUniformVectors = mtl::kDefaultUniformsMaxSize / (sizeof(GLfloat) * 4);

    const GLuint maxDefaultUniformComponents = maxDefaultUniformVectors * 4;

    mNativeCaps.maxVertexUniformVectors                              = maxDefaultUniformVectors;
    mNativeCaps.maxShaderUniformComponents[gl::ShaderType::Vertex]   = maxDefaultUniformComponents;
    mNativeCaps.maxFragmentUniformVectors                            = maxDefaultUniformVectors;
    mNativeCaps.maxShaderUniformComponents[gl::ShaderType::Fragment] = maxDefaultUniformComponents;

    // NOTE(hqle): support UBO (ES 3.0 feature)
    mNativeCaps.maxShaderUniformBlocks[gl::ShaderType::Vertex]   = mtl::kMaxShaderUBOs;
    mNativeCaps.maxShaderUniformBlocks[gl::ShaderType::Fragment] = mtl::kMaxShaderUBOs;
    mNativeCaps.maxCombinedUniformBlocks                         = mtl::kMaxGLUBOBindings;

    // Note that we currently implement textures as combined image+samplers, so the limit is
    // the minimum of supported samplers and sampled images.
    mNativeCaps.maxCombinedTextureImageUnits                         = mtl::kMaxGLSamplerBindings;
    mNativeCaps.maxShaderTextureImageUnits[gl::ShaderType::Fragment] = mtl::kMaxShaderSamplers;
    mNativeCaps.maxShaderTextureImageUnits[gl::ShaderType::Vertex]   = mtl::kMaxShaderSamplers;

    // No info from Metal given, use default spec values:
    mNativeCaps.minProgramTexelOffset = -8;
    mNativeCaps.maxProgramTexelOffset = 7;

    // NOTE(hqle): support storage buffer.
    const uint32_t maxPerStageStorageBuffers                     = 0;
    mNativeCaps.maxShaderStorageBlocks[gl::ShaderType::Vertex]   = maxPerStageStorageBuffers;
    mNativeCaps.maxShaderStorageBlocks[gl::ShaderType::Fragment] = maxPerStageStorageBuffers;
    mNativeCaps.maxCombinedShaderStorageBlocks                   = maxPerStageStorageBuffers;

    // Fill in additional limits for UBOs and SSBOs.
    mNativeCaps.maxUniformBufferBindings = mNativeCaps.maxCombinedUniformBlocks;
    mNativeCaps.maxUniformBlockSize      = mtl::kMaxUBOSize;  // Default according to GLES 3.0 spec.
    mNativeCaps.uniformBufferOffsetAlignment = 1;

    mNativeCaps.maxShaderStorageBufferBindings     = 0;
    mNativeCaps.maxShaderStorageBlockSize          = 0;
    mNativeCaps.shaderStorageBufferOffsetAlignment = 0;

    // UBO plus default uniform limits
    const uint32_t maxCombinedUniformComponents =
        maxDefaultUniformComponents + mtl::kMaxUBOSize * mtl::kMaxShaderUBOs / 4;
    for (gl::ShaderType shaderType : gl::kAllGraphicsShaderTypes)
    {
        mNativeCaps.maxCombinedShaderUniformComponents[shaderType] = maxCombinedUniformComponents;
    }

    mNativeCaps.maxCombinedShaderOutputResources = 0;

    mNativeCaps.maxTransformFeedbackInterleavedComponents =
        gl::IMPLEMENTATION_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS;
    mNativeCaps.maxTransformFeedbackSeparateAttributes =
        gl::IMPLEMENTATION_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS;
    mNativeCaps.maxTransformFeedbackSeparateComponents =
        gl::IMPLEMENTATION_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS;

    // GL_OES_get_program_binary
    mNativeCaps.programBinaryFormats.push_back(GL_PROGRAM_BINARY_ANGLE);

    // GL_APPLE_clip_distance
    mNativeCaps.maxClipDistances = 8;
}

void DisplayMtl::initializeExtensions() const
{
    // Reset
    mNativeExtensions = gl::Extensions();

    // Enable this for simple buffer readback testing, but some functionality is missing.
    mNativeExtensions.mapBuffer              = true;
    mNativeExtensions.mapBufferRange         = true;
    mNativeExtensions.textureStorage         = true;
    mNativeExtensions.drawBuffers            = true;
    mNativeExtensions.fragDepth              = true;
    mNativeExtensions.framebufferBlit        = true;
    mNativeExtensions.framebufferMultisample = true;
    mNativeExtensions.copyTexture            = false;
    mNativeExtensions.copyCompressedTexture  = false;
    mNativeExtensions.debugMarker            = true;
    mNativeExtensions.robustness             = true;
    mNativeExtensions.textureBorderClamp     = false;  // not implemented yet
    mNativeExtensions.translatedShaderSource = true;
    mNativeExtensions.discardFramebuffer     = true;

    // EXT_multisampled_render_to_texture
    if (mFeatures.allowMultisampleStoreAndResolve.enabled &&
        mFeatures.hasDepthAutoResolve.enabled && mFeatures.hasStencilAutoResolve.enabled)
    {
        mNativeExtensions.multisampledRenderToTexture = true;
    }

    // Enable EXT_blend_minmax
    mNativeExtensions.blendMinMax = true;

    mNativeExtensions.eglImage         = true;
    mNativeExtensions.eglImageExternal = false;
    // NOTE(hqle): Support GL_OES_EGL_image_external_essl3.
    mNativeExtensions.eglImageExternalEssl3 = false;

    // MGL_EGL_image_cube
    mNativeExtensions.eglImageCubeMGL = true;

    mNativeExtensions.memoryObject   = false;
    mNativeExtensions.memoryObjectFd = false;

    mNativeExtensions.semaphore   = false;
    mNativeExtensions.semaphoreFd = false;

    mNativeExtensions.instancedArraysANGLE = true;
    mNativeExtensions.instancedArraysEXT   = true;

    mNativeExtensions.robustBufferAccessBehavior = false;

    mNativeExtensions.eglSync = false;

    mNativeExtensions.occlusionQueryBoolean = true;

    mNativeExtensions.disjointTimerQuery          = false;
    mNativeExtensions.queryCounterBitsTimeElapsed = false;
    mNativeExtensions.queryCounterBitsTimestamp   = false;

    mNativeExtensions.textureFilterAnisotropic = true;
    mNativeExtensions.maxTextureAnisotropy     = 16;

    mNativeExtensions.textureNPOT = true;

    mNativeExtensions.texture3DOES = true;

    mNativeExtensions.standardDerivatives = true;

    mNativeExtensions.elementIndexUint = true;

    // GL_OES_get_program_binary
    mNativeExtensions.getProgramBinary = true;

    // GL_APPLE_clip_distance
    mNativeExtensions.clipDistanceAPPLE = true;

    // GL_NV_pixel_buffer_object
    mNativeExtensions.pixelBufferObject = true;

    if (ANGLE_APPLE_AVAILABLE_XCI(10.14, 13.0, 12.0))
    {
        // MTLSharedEvent is only available since Metal 2.1

        // GL_NV_fence
        mNativeExtensions.fence = true;

        // GL_OES_EGL_sync
        mNativeExtensions.eglSync = true;
    }
}

void DisplayMtl::initializeTextureCaps() const
{
    mNativeTextureCaps.clear();

    mFormatTable.generateTextureCaps(this, &mNativeTextureCaps,
                                     &mNativeCaps.compressedTextureFormats);

    // Re-verify texture extensions.
    mNativeExtensions.setTextureExtensionSupport(mNativeTextureCaps);
}

void DisplayMtl::initializeFeatures()
{
    // default values:
    mFeatures.hasBaseVertexInstancedDraw.enabled        = true;
    mFeatures.hasDepthTextureFiltering.enabled          = false;
    mFeatures.hasExplicitMemBarrier.enabled             = false;
    mFeatures.hasNonUniformDispatch.enabled             = true;
    mFeatures.hasStencilOutput.enabled                  = false;
    mFeatures.hasTextureSwizzle.enabled                 = false;
    mFeatures.allowInlineConstVertexData.enabled        = true;
    mFeatures.allowSeparatedDepthStencilBuffers.enabled = false;
    mFeatures.forceBufferGPUStorage.enabled             = false;
    mFeatures.emulateDepthRangeMappingInShader.enabled  = true;

    ANGLE_FEATURE_CONDITION((&mFeatures), hasDepthAutoResolve, supportEitherGPUFamily(3, 2));
    ANGLE_FEATURE_CONDITION((&mFeatures), hasStencilAutoResolve, supportEitherGPUFamily(5, 2));
    ANGLE_FEATURE_CONDITION((&mFeatures), allowBufferReadWrite, supportEitherGPUFamily(3, 1));
    ANGLE_FEATURE_CONDITION((&mFeatures), allowRuntimeSamplerCompareMode,
                            supportEitherGPUFamily(3, 1));
    ANGLE_FEATURE_CONDITION((&mFeatures), allowMultisampleStoreAndResolve,
                            supportEitherGPUFamily(3, 1));

    if (ANGLE_APPLE_AVAILABLE_XCI(10.15, 13.0, 13.0))
    {
        // Disable Compute Shader based mipmap generation on iOS GPU family 3 and below.
        ANGLE_FEATURE_CONDITION((&mFeatures), forceNonCSBaseMipmapGeneration,
                                !supportEitherGPUFamily(4, 1));
    }
    else
    {
        // NOTE(hqle): Disable Compute Shader based mipmap generation on macOS 10.14, iOS 12.0 and
        // below also. Until CS based mipmap generation can be tested on those platform.
        mFeatures.forceNonCSBaseMipmapGeneration.enabled = true;
    }

    if (ANGLE_APPLE_AVAILABLE_XCI(10.14, 13.0, 12.0))
    {
        mFeatures.hasStencilOutput.enabled                 = true;
        mFeatures.emulateDepthRangeMappingInShader.enabled = false;
        ANGLE_FEATURE_CONDITION((&mFeatures), hasExplicitMemBarrier,
                                (TARGET_OS_OSX || TARGET_OS_MACCATALYST) && !ANGLE_MTL_ARM);
    }
    if (ANGLE_APPLE_AVAILABLE_XCI(10.15, 13.0, 13.0))
    {
        ANGLE_FEATURE_CONDITION((&mFeatures), hasTextureSwizzle, supportEitherGPUFamily(1, 2));
    }

#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
    mFeatures.hasDepthTextureFiltering.enabled = !ANGLE_MTL_ARM;
    mFeatures.breakRenderPassIsCheap.enabled   = !ANGLE_MTL_ARM;

#elif TARGET_OS_IOS || TARGET_OS_TV
    mFeatures.breakRenderPassIsCheap.enabled = false;

    // Base Vertex drawing is only supported since GPU family 3.
    ANGLE_FEATURE_CONDITION((&mFeatures), hasBaseVertexInstancedDraw, supportiOSGPUFamily(3));

    ANGLE_FEATURE_CONDITION((&mFeatures), hasNonUniformDispatch,
                            TARGET_OS_IOS && supportiOSGPUFamily(4));

    ANGLE_FEATURE_CONDITION((&mFeatures), allowSeparatedDepthStencilBuffers, !TARGET_OS_SIMULATOR);

#endif

    angle::PlatformMethods *platform = ANGLEPlatformCurrent();
    platform->overrideFeaturesMtl(platform, &mFeatures);
}

angle::Result DisplayMtl::initializeShaderLibrary()
{
    mtl::AutoObjCObj<NSError> err = nil;

    const uint8_t *compiled_shader_binary;
    size_t compiled_shader_binary_len;

#if !defined(NDEBUG)
    if (getFeatures().hasStencilOutput.enabled)
    {
        compiled_shader_binary     = compiled_default_metallib_2_1_debug;
        compiled_shader_binary_len = compiled_default_metallib_2_1_debug_len;
    }
    else
    {
        compiled_shader_binary     = compiled_default_metallib_debug;
        compiled_shader_binary_len = compiled_default_metallib_debug_len;
    }
#else
    if (getFeatures().hasStencilOutput.enabled)
    {
        compiled_shader_binary     = compiled_default_metallib_2_1;
        compiled_shader_binary_len = compiled_default_metallib_2_1_len;
    }
    else
    {
        compiled_shader_binary     = compiled_default_metallib;
        compiled_shader_binary_len = compiled_default_metallib_len;
    }
#endif

    mDefaultShaders = CreateShaderLibraryFromBinary(getMetalDevice(), compiled_shader_binary,
                                                    compiled_shader_binary_len, &err);

    if (err && !mDefaultShaders)
    {
        ANGLE_MTL_OBJC_SCOPE
        {
            ERR() << "Internal error: " << err.get().localizedDescription.UTF8String;
        }
        return angle::Result::Stop;
    }

    return angle::Result::Continue;
}

id<MTLLibrary> DisplayMtl::getDefaultShadersLib()
{
    if (!mDefaultShaders)
    {
        if (initializeShaderLibrary() != angle::Result::Continue)
        {
            return nil;
        }
    }
    return mDefaultShaders;
}

bool DisplayMtl::supportiOSGPUFamily(uint8_t iOSFamily) const
{
#if (!TARGET_OS_IOS && !TARGET_OS_TV) || TARGET_OS_MACCATALYST
    return false;
#else
#    if (__IPHONE_OS_VERSION_MAX_ALLOWED >= 130000) || (__TV_OS_VERSION_MAX_ALLOWED >= 130000)
    // If device supports [MTLDevice supportsFamily:], then use it.
    if (ANGLE_APPLE_AVAILABLE_I(13.0))
    {
        MTLGPUFamily family;
        switch (iOSFamily)
        {
            case 1:
                family = MTLGPUFamilyApple1;
                break;
            case 2:
                family = MTLGPUFamilyApple2;
                break;
            case 3:
                family = MTLGPUFamilyApple3;
                break;
            case 4:
                family = MTLGPUFamilyApple4;
                break;
            case 5:
                family = MTLGPUFamilyApple5;
                break;
#        if TARGET_OS_IOS
            case 6:
                family = MTLGPUFamilyApple6;
                break;
#        endif
            default:
                return false;
        }
        return [getMetalDevice() supportsFamily:family];
    }  // Metal 2.2
#    endif  // __IPHONE_OS_VERSION_MAX_ALLOWED

    // If device doesn't support [MTLDevice supportsFamily:], then use
    // [MTLDevice supportsFeatureSet:].
    MTLFeatureSet featureSet;
    switch (iOSFamily)
    {
#    if TARGET_OS_IOS
        case 1:
            featureSet = MTLFeatureSet_iOS_GPUFamily1_v1;
            break;
        case 2:
            featureSet = MTLFeatureSet_iOS_GPUFamily2_v1;
            break;
        case 3:
            featureSet = MTLFeatureSet_iOS_GPUFamily3_v1;
            break;
        case 4:
            featureSet = MTLFeatureSet_iOS_GPUFamily4_v1;
            break;
#        if __IPHONE_OS_VERSION_MAX_ALLOWED >= 120000
        case 5:
            featureSet = MTLFeatureSet_iOS_GPUFamily5_v1;
            break;
#        endif  // __IPHONE_OS_VERSION_MAX_ALLOWED
#    elif TARGET_OS_TV
        case 1:
        case 2:
            featureSet = MTLFeatureSet_tvOS_GPUFamily1_v1;
            break;
        case 3:
            featureSet = MTLFeatureSet_tvOS_GPUFamily2_v1;
            break;
#    endif  // TARGET_OS_IOS
        default:
            return false;
    }

    return [getMetalDevice() supportsFeatureSet:featureSet];
#endif      // TARGET_OS_IOS || TARGET_OS_TV
}

bool DisplayMtl::supportMacGPUFamily(uint8_t macFamily) const
{
#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
#    if defined(__MAC_10_15)
    // If device supports [MTLDevice supportsFamily:], then use it.
    if (ANGLE_APPLE_AVAILABLE_XC(10.15, 13.0))
    {
        MTLGPUFamily family;

        switch (macFamily)
        {
#        if TARGET_OS_MACCATALYST
            case 1:
                family = MTLGPUFamilyMacCatalyst1;
                break;
            case 2:
                family = MTLGPUFamilyMacCatalyst2;
                break;
#        else   // TARGET_OS_MACCATALYST
            case 1:
                family = MTLGPUFamilyMac1;
                break;
            case 2:
                family = MTLGPUFamilyMac2;
                break;
#        endif  // TARGET_OS_MACCATALYST
            default:
                return false;
        }

        return [getMetalDevice() supportsFamily:family];
    }  // Metal 2.2
#    endif

    // If device doesn't support [MTLDevice supportsFamily:], then use
    // [MTLDevice supportsFeatureSet:].
#    if TARGET_OS_MACCATALYST
    UNREACHABLE();
    return false;
#    else
    MTLFeatureSet featureSet;
    switch (macFamily)
    {
        case 1:
            featureSet = MTLFeatureSet_macOS_GPUFamily1_v1;
            break;
#        if defined(__MAC_10_14)
        case 2:
            featureSet = MTLFeatureSet_macOS_GPUFamily2_v1;
            break;
#        endif
        default:
            return false;
    }
    return [getMetalDevice() supportsFeatureSet:featureSet];
#    endif  // TARGET_OS_MACCATALYST
#else       // #if TARGET_OS_OSX || TARGET_OS_MACCATALYST

    return false;

#endif
}

bool DisplayMtl::supportEitherGPUFamily(uint8_t iOSFamily, uint8_t macFamily) const
{
    return supportiOSGPUFamily(iOSFamily) || supportMacGPUFamily(macFamily);
}

#if defined(__IPHONE_12_0) || defined(__MAC_10_14)
mtl::AutoObjCObj<MTLSharedEventListener> DisplayMtl::getOrCreateSharedEventListener()
{
    if (!mSharedEventListener)
    {
        ANGLE_MTL_OBJC_SCOPE
        {
            mSharedEventListener = [[[MTLSharedEventListener alloc] init] ANGLE_MTL_AUTORELEASE];
        }
    }
    return mSharedEventListener;
}
#endif

}  // namespace rx
