//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_state_cache.mm:
//    Implements StateCache, RenderPipelineCache and various
//    C struct versions of Metal sampler, depth stencil, render pass, render pipeline descriptors.
//

#include "libANGLE/renderer/metal/mtl_state_cache.h"

#include <sstream>

#include "common/debug.h"
#include "common/hash_utils.h"
#include "libANGLE/renderer/metal/ContextMtl.h"
#include "libANGLE/renderer/metal/mtl_resources.h"
#include "libANGLE/renderer/metal/mtl_utils.h"
#include "platform/FeaturesMtl.h"

#define ANGLE_OBJC_CP_PROPERTY(DST, SRC, PROPERTY) \
    (DST).PROPERTY = static_cast<__typeof__((DST).PROPERTY)>(ToObjC((SRC).PROPERTY))

#define ANGLE_PROP_EQ(LHS, RHS, PROP) ((LHS).PROP == (RHS).PROP)

namespace rx
{
namespace mtl
{

namespace
{

template <class T>
inline T ToObjC(const T p)
{
    return p;
}

inline MTLStencilDescriptor *ToObjC(const StencilDesc &desc)
{
    MTLStencilDescriptor *objCDesc = [[MTLStencilDescriptor alloc] init];

    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, stencilFailureOperation);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, depthFailureOperation);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, depthStencilPassOperation);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, stencilCompareFunction);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, readMask);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, writeMask);

    return [objCDesc ANGLE_MTL_AUTORELEASE];
}

MTLDepthStencilDescriptor *ToObjC(const DepthStencilDesc &desc)
{
    MTLDepthStencilDescriptor *objCDesc = [[MTLDepthStencilDescriptor alloc] init];

    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, backFaceStencil);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, frontFaceStencil);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, depthCompareFunction);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, depthWriteEnabled);

    return [objCDesc ANGLE_MTL_AUTORELEASE];
}

MTLSamplerDescriptor *ToObjC(const SamplerDesc &desc)
{
    MTLSamplerDescriptor *objCDesc = [[MTLSamplerDescriptor alloc] init];

    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, rAddressMode);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, sAddressMode);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, tAddressMode);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, minFilter);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, magFilter);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, mipFilter);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, maxAnisotropy);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, compareFunction);

    return [objCDesc ANGLE_MTL_AUTORELEASE];
}

MTLVertexAttributeDescriptor *ToObjC(const VertexAttributeDesc &desc)
{
    MTLVertexAttributeDescriptor *objCDesc = [[MTLVertexAttributeDescriptor alloc] init];

    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, format);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, offset);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, bufferIndex);

    ASSERT(desc.bufferIndex >= kVboBindingIndexStart);

    return [objCDesc ANGLE_MTL_AUTORELEASE];
}

MTLVertexBufferLayoutDescriptor *ToObjC(const VertexBufferLayoutDesc &desc)
{
    MTLVertexBufferLayoutDescriptor *objCDesc = [[MTLVertexBufferLayoutDescriptor alloc] init];

    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, stepFunction);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, stepRate);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, stride);

    return [objCDesc ANGLE_MTL_AUTORELEASE];
}

MTLVertexDescriptor *ToObjC(const VertexDesc &desc)
{
    MTLVertexDescriptor *objCDesc = [[MTLVertexDescriptor alloc] init];
    [objCDesc reset];

    for (uint8_t i = 0; i < desc.numAttribs; ++i)
    {
        [objCDesc.attributes setObject:ToObjC(desc.attributes[i]) atIndexedSubscript:i];
    }

    for (uint8_t i = 0; i < desc.numBufferLayouts; ++i)
    {
        // Ignore if stepFunction is kVertexStepFunctionInvalid.
        if (desc.layouts[i].stepFunction != kVertexStepFunctionInvalid)
        {
            [objCDesc.layouts setObject:ToObjC(desc.layouts[i]) atIndexedSubscript:i];
        }
    }

    return [objCDesc ANGLE_MTL_AUTORELEASE];
}

MTLRenderPipelineColorAttachmentDescriptor *ToObjC(const RenderPipelineColorAttachmentDesc &desc)
{
    MTLRenderPipelineColorAttachmentDescriptor *objCDesc =
        [[MTLRenderPipelineColorAttachmentDescriptor alloc] init];

    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, pixelFormat);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, writeMask);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, alphaBlendOperation);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, rgbBlendOperation);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, destinationAlphaBlendFactor);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, destinationRGBBlendFactor);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, sourceAlphaBlendFactor);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, sourceRGBBlendFactor);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, blendingEnabled);

    return [objCDesc ANGLE_MTL_AUTORELEASE];
}

MTLRenderPipelineDescriptor *ToObjC(id<MTLFunction> vertexShader,
                                    id<MTLFunction> fragmentShader,
                                    const RenderPipelineDesc &desc)
{
    MTLRenderPipelineDescriptor *objCDesc = [[MTLRenderPipelineDescriptor alloc] init];
    [objCDesc reset];

    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, vertexDescriptor);

    for (uint8_t i = 0; i < desc.outputDescriptor.numColorAttachments; ++i)
    {
        [objCDesc.colorAttachments setObject:ToObjC(desc.outputDescriptor.colorAttachments[i])
                          atIndexedSubscript:i];
    }
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc.outputDescriptor, depthAttachmentPixelFormat);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc.outputDescriptor, stencilAttachmentPixelFormat);
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc.outputDescriptor, sampleCount);

#if ANGLE_MTL_PRIMITIVE_TOPOLOGY_CLASS_AVAILABLE
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, inputPrimitiveTopology);
#endif
    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, alphaToCoverageEnabled);

    // rasterizationEnabled will be true for both EmulatedDiscard & Enabled.
    objCDesc.rasterizationEnabled = desc.rasterizationEnabled();

    objCDesc.vertexFunction   = vertexShader;
    objCDesc.fragmentFunction = objCDesc.rasterizationEnabled ? fragmentShader : nil;

    return [objCDesc ANGLE_MTL_AUTORELEASE];
}

id<MTLTexture> ToObjC(const TextureRef &texture)
{
    auto textureRef = texture;
    return textureRef ? textureRef->get() : nil;
}

void BaseRenderPassAttachmentDescToObjC(const RenderPassAttachmentDesc &src,
                                        MTLRenderPassAttachmentDescriptor *dst)
{
    auto implicitMsTexture = src.implicitMSTexture();

    if (implicitMsTexture)
    {
        dst.texture        = ToObjC(implicitMsTexture);
        dst.level          = 0;
        dst.slice          = 0;
        dst.depthPlane     = 0;
        dst.resolveTexture = ToObjC(src.texture());
        dst.resolveLevel   = src.level();
        if (dst.resolveTexture.textureType == MTLTextureType3D)
        {
            dst.resolveDepthPlane = src.sliceOrDepth();
            dst.resolveSlice      = 0;
        }
        else
        {
            dst.resolveSlice      = src.sliceOrDepth();
            dst.resolveDepthPlane = 0;
        }
    }
    else
    {
        dst.texture = ToObjC(src.texture());
        dst.level   = src.level();
        if (dst.texture.textureType == MTLTextureType3D)
        {
            dst.depthPlane = src.sliceOrDepth();
            dst.slice      = 0;
        }
        else
        {
            dst.slice      = src.sliceOrDepth();
            dst.depthPlane = 0;
        }
        dst.resolveTexture    = nil;
        dst.resolveLevel      = 0;
        dst.resolveSlice      = 0;
        dst.resolveDepthPlane = 0;
    }

    if (!dst.texture)
    {
        dst.resolveTexture = nil;
        dst.loadAction     = MTLLoadActionDontCare;
        dst.storeAction    = MTLStoreActionDontCare;
    }
    else
    {
        ANGLE_OBJC_CP_PROPERTY(dst, src, loadAction);
        ANGLE_OBJC_CP_PROPERTY(dst, src, storeAction);
        ANGLE_OBJC_CP_PROPERTY(dst, src, storeActionOptions);
    }
}

void ToObjC(const RenderPassColorAttachmentDesc &desc,
            MTLRenderPassColorAttachmentDescriptor *objCDesc)
{
    BaseRenderPassAttachmentDescToObjC(desc, objCDesc);

    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, clearColor);
}

void ToObjC(const RenderPassDepthAttachmentDesc &desc,
            MTLRenderPassDepthAttachmentDescriptor *objCDesc)
{
    BaseRenderPassAttachmentDescToObjC(desc, objCDesc);

    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, clearDepth);
}

void ToObjC(const RenderPassStencilAttachmentDesc &desc,
            MTLRenderPassStencilAttachmentDescriptor *objCDesc)
{
    BaseRenderPassAttachmentDescToObjC(desc, objCDesc);

    ANGLE_OBJC_CP_PROPERTY(objCDesc, desc, clearStencil);
}

}  // namespace

// StencilDesc implementation
bool StencilDesc::operator==(const StencilDesc &rhs) const
{
    return ANGLE_PROP_EQ(*this, rhs, stencilFailureOperation) &&
           ANGLE_PROP_EQ(*this, rhs, depthFailureOperation) &&
           ANGLE_PROP_EQ(*this, rhs, depthStencilPassOperation) &&

           ANGLE_PROP_EQ(*this, rhs, stencilCompareFunction) &&

           ANGLE_PROP_EQ(*this, rhs, readMask) && ANGLE_PROP_EQ(*this, rhs, writeMask);
}

void StencilDesc::reset()
{
    stencilFailureOperation = depthFailureOperation = depthStencilPassOperation =
        MTLStencilOperationKeep;

    stencilCompareFunction = MTLCompareFunctionAlways;
    readMask = writeMask = std::numeric_limits<uint32_t>::max() & mtl::kStencilMaskAll;
}

// DepthStencilDesc implementation
DepthStencilDesc::DepthStencilDesc()
{
    memset(this, 0, sizeof(*this));
}
DepthStencilDesc::DepthStencilDesc(const DepthStencilDesc &src)
{
    memcpy(this, &src, sizeof(*this));
}
DepthStencilDesc::DepthStencilDesc(DepthStencilDesc &&src)
{
    memcpy(this, &src, sizeof(*this));
}

DepthStencilDesc &DepthStencilDesc::operator=(const DepthStencilDesc &src)
{
    memcpy(this, &src, sizeof(*this));
    return *this;
}

bool DepthStencilDesc::operator==(const DepthStencilDesc &rhs) const
{
    return ANGLE_PROP_EQ(*this, rhs, backFaceStencil) &&
           ANGLE_PROP_EQ(*this, rhs, frontFaceStencil) &&

           ANGLE_PROP_EQ(*this, rhs, depthCompareFunction) &&

           ANGLE_PROP_EQ(*this, rhs, depthWriteEnabled);
}

void DepthStencilDesc::reset()
{
    frontFaceStencil.reset();
    backFaceStencil.reset();

    depthCompareFunction = MTLCompareFunctionAlways;
    depthWriteEnabled    = true;
}

void DepthStencilDesc::updateDepthTestEnabled(const gl::DepthStencilState &dsState)
{
    if (!dsState.depthTest)
    {
        depthCompareFunction = MTLCompareFunctionAlways;
        depthWriteEnabled    = false;
    }
    else
    {
        updateDepthCompareFunc(dsState);
        updateDepthWriteEnabled(dsState);
    }
}

void DepthStencilDesc::updateDepthWriteEnabled(const gl::DepthStencilState &dsState)
{
    depthWriteEnabled = dsState.depthTest && dsState.depthMask;
}

void DepthStencilDesc::updateDepthCompareFunc(const gl::DepthStencilState &dsState)
{
    if (!dsState.depthTest)
    {
        return;
    }
    depthCompareFunction = GetCompareFunc(dsState.depthFunc);
}

void DepthStencilDesc::updateStencilTestEnabled(const gl::DepthStencilState &dsState)
{
    if (!dsState.stencilTest)
    {
        frontFaceStencil.stencilCompareFunction    = MTLCompareFunctionAlways;
        frontFaceStencil.depthFailureOperation     = MTLStencilOperationKeep;
        frontFaceStencil.depthStencilPassOperation = MTLStencilOperationKeep;
        frontFaceStencil.writeMask                 = 0;

        backFaceStencil.stencilCompareFunction    = MTLCompareFunctionAlways;
        backFaceStencil.depthFailureOperation     = MTLStencilOperationKeep;
        backFaceStencil.depthStencilPassOperation = MTLStencilOperationKeep;
        backFaceStencil.writeMask                 = 0;
    }
    else
    {
        updateStencilFrontFuncs(dsState);
        updateStencilFrontOps(dsState);
        updateStencilFrontWriteMask(dsState);
        updateStencilBackFuncs(dsState);
        updateStencilBackOps(dsState);
        updateStencilBackWriteMask(dsState);
    }
}

void DepthStencilDesc::updateStencilFrontOps(const gl::DepthStencilState &dsState)
{
    if (!dsState.stencilTest)
    {
        return;
    }
    frontFaceStencil.stencilFailureOperation   = GetStencilOp(dsState.stencilFail);
    frontFaceStencil.depthFailureOperation     = GetStencilOp(dsState.stencilPassDepthFail);
    frontFaceStencil.depthStencilPassOperation = GetStencilOp(dsState.stencilPassDepthPass);
}

void DepthStencilDesc::updateStencilBackOps(const gl::DepthStencilState &dsState)
{
    if (!dsState.stencilTest)
    {
        return;
    }
    backFaceStencil.stencilFailureOperation   = GetStencilOp(dsState.stencilBackFail);
    backFaceStencil.depthFailureOperation     = GetStencilOp(dsState.stencilBackPassDepthFail);
    backFaceStencil.depthStencilPassOperation = GetStencilOp(dsState.stencilBackPassDepthPass);
}

void DepthStencilDesc::updateStencilFrontFuncs(const gl::DepthStencilState &dsState)
{
    if (!dsState.stencilTest)
    {
        return;
    }
    frontFaceStencil.stencilCompareFunction = GetCompareFunc(dsState.stencilFunc);
    frontFaceStencil.readMask               = dsState.stencilMask & mtl::kStencilMaskAll;
}

void DepthStencilDesc::updateStencilBackFuncs(const gl::DepthStencilState &dsState)
{
    if (!dsState.stencilTest)
    {
        return;
    }
    backFaceStencil.stencilCompareFunction = GetCompareFunc(dsState.stencilBackFunc);
    backFaceStencil.readMask               = dsState.stencilBackMask & mtl::kStencilMaskAll;
}

void DepthStencilDesc::updateStencilFrontWriteMask(const gl::DepthStencilState &dsState)
{
    if (!dsState.stencilTest)
    {
        return;
    }
    frontFaceStencil.writeMask = dsState.stencilWritemask & mtl::kStencilMaskAll;
}

void DepthStencilDesc::updateStencilBackWriteMask(const gl::DepthStencilState &dsState)
{
    if (!dsState.stencilTest)
    {
        return;
    }
    backFaceStencil.writeMask = dsState.stencilBackWritemask & mtl::kStencilMaskAll;
}

size_t DepthStencilDesc::hash() const
{
    return angle::ComputeGenericHash(*this);
}

// SamplerDesc implementation
SamplerDesc::SamplerDesc()
{
    memset(this, 0, sizeof(*this));
}
SamplerDesc::SamplerDesc(const SamplerDesc &src)
{
    memcpy(this, &src, sizeof(*this));
}
SamplerDesc::SamplerDesc(SamplerDesc &&src)
{
    memcpy(this, &src, sizeof(*this));
}

SamplerDesc::SamplerDesc(const gl::SamplerState &glState) : SamplerDesc()
{
    rAddressMode = GetSamplerAddressMode(glState.getWrapR());
    sAddressMode = GetSamplerAddressMode(glState.getWrapS());
    tAddressMode = GetSamplerAddressMode(glState.getWrapT());

    minFilter = GetFilter(glState.getMinFilter());
    magFilter = GetFilter(glState.getMagFilter());
    mipFilter = GetMipmapFilter(glState.getMinFilter());

    maxAnisotropy = static_cast<uint32_t>(glState.getMaxAnisotropy());

    compareFunction = GetCompareFunc(glState.getCompareFunc());
}

SamplerDesc &SamplerDesc::operator=(const SamplerDesc &src)
{
    memcpy(this, &src, sizeof(*this));
    return *this;
}

void SamplerDesc::reset()
{
    rAddressMode = MTLSamplerAddressModeClampToEdge;
    sAddressMode = MTLSamplerAddressModeClampToEdge;
    tAddressMode = MTLSamplerAddressModeClampToEdge;

    minFilter = MTLSamplerMinMagFilterNearest;
    magFilter = MTLSamplerMinMagFilterNearest;
    mipFilter = MTLSamplerMipFilterNearest;

    maxAnisotropy = 1;

    compareFunction = MTLCompareFunctionNever;
}

bool SamplerDesc::operator==(const SamplerDesc &rhs) const
{
    return ANGLE_PROP_EQ(*this, rhs, rAddressMode) && ANGLE_PROP_EQ(*this, rhs, sAddressMode) &&
           ANGLE_PROP_EQ(*this, rhs, tAddressMode) &&

           ANGLE_PROP_EQ(*this, rhs, minFilter) && ANGLE_PROP_EQ(*this, rhs, magFilter) &&
           ANGLE_PROP_EQ(*this, rhs, mipFilter) &&

           ANGLE_PROP_EQ(*this, rhs, maxAnisotropy) &&

           ANGLE_PROP_EQ(*this, rhs, compareFunction);
    ;
}

size_t SamplerDesc::hash() const
{
    return angle::ComputeGenericHash(*this);
}

// BlendDesc implementation
bool BlendDesc::operator==(const BlendDesc &rhs) const
{
    return ANGLE_PROP_EQ(*this, rhs, writeMask) &&

           ANGLE_PROP_EQ(*this, rhs, alphaBlendOperation) &&
           ANGLE_PROP_EQ(*this, rhs, rgbBlendOperation) &&

           ANGLE_PROP_EQ(*this, rhs, destinationAlphaBlendFactor) &&
           ANGLE_PROP_EQ(*this, rhs, destinationRGBBlendFactor) &&
           ANGLE_PROP_EQ(*this, rhs, sourceAlphaBlendFactor) &&
           ANGLE_PROP_EQ(*this, rhs, sourceRGBBlendFactor) &&

           ANGLE_PROP_EQ(*this, rhs, blendingEnabled);
}

void BlendDesc::reset()
{
    reset(MTLColorWriteMaskAll);
}

void BlendDesc::reset(MTLColorWriteMask _writeMask)
{
    writeMask = _writeMask;

    blendingEnabled     = false;
    alphaBlendOperation = rgbBlendOperation = MTLBlendOperationAdd;

    destinationAlphaBlendFactor = destinationRGBBlendFactor = MTLBlendFactorZero;
    sourceAlphaBlendFactor = sourceRGBBlendFactor = MTLBlendFactorOne;
}

void BlendDesc::updateWriteMask(const gl::BlendState &blendState)
{
    writeMask = MTLColorWriteMaskNone;
    if (blendState.colorMaskRed)
    {
        writeMask |= MTLColorWriteMaskRed;
    }
    if (blendState.colorMaskGreen)
    {
        writeMask |= MTLColorWriteMaskGreen;
    }
    if (blendState.colorMaskBlue)
    {
        writeMask |= MTLColorWriteMaskBlue;
    }
    if (blendState.colorMaskAlpha)
    {
        writeMask |= MTLColorWriteMaskAlpha;
    }
}

void BlendDesc::updateBlendFactors(const gl::BlendState &blendState)
{
    sourceRGBBlendFactor        = GetBlendFactor(blendState.sourceBlendRGB);
    sourceAlphaBlendFactor      = GetBlendFactor(blendState.sourceBlendAlpha);
    destinationRGBBlendFactor   = GetBlendFactor(blendState.destBlendRGB);
    destinationAlphaBlendFactor = GetBlendFactor(blendState.destBlendAlpha);
}

void BlendDesc::updateBlendOps(const gl::BlendState &blendState)
{
    rgbBlendOperation   = GetBlendOp(blendState.blendEquationRGB);
    alphaBlendOperation = GetBlendOp(blendState.blendEquationAlpha);
}

void BlendDesc::updateBlendEnabled(const gl::BlendState &blendState)
{
    blendingEnabled = blendState.blend;
}

// RenderPipelineColorAttachmentDesc implementation
bool RenderPipelineColorAttachmentDesc::operator==(
    const RenderPipelineColorAttachmentDesc &rhs) const
{
    if (!BlendDesc::operator==(rhs))
    {
        return false;
    }
    return ANGLE_PROP_EQ(*this, rhs, pixelFormat);
}

void RenderPipelineColorAttachmentDesc::reset()
{
    reset(MTLPixelFormatInvalid);
}

void RenderPipelineColorAttachmentDesc::reset(MTLPixelFormat format)
{
    reset(format, MTLColorWriteMaskAll);
}

void RenderPipelineColorAttachmentDesc::reset(MTLPixelFormat format, MTLColorWriteMask _writeMask)
{
    this->pixelFormat = format;

    BlendDesc::reset(_writeMask);
}

void RenderPipelineColorAttachmentDesc::reset(MTLPixelFormat format, const BlendDesc &blendState)
{
    this->pixelFormat = format;

    BlendDesc::operator=(blendState);
}

void RenderPipelineColorAttachmentDesc::update(const BlendDesc &blendState)
{
    BlendDesc::operator=(blendState);
}

// RenderPipelineOutputDesc implementation
bool RenderPipelineOutputDesc::operator==(const RenderPipelineOutputDesc &rhs) const
{
    if (numColorAttachments != rhs.numColorAttachments)
    {
        return false;
    }

    for (uint8_t i = 0; i < numColorAttachments; ++i)
    {
        if (colorAttachments[i] != rhs.colorAttachments[i])
        {
            return false;
        }
    }

    return ANGLE_PROP_EQ(*this, rhs, depthAttachmentPixelFormat) &&
           ANGLE_PROP_EQ(*this, rhs, stencilAttachmentPixelFormat);
}

void RenderPipelineOutputDesc::updateEnabledDrawBuffers(gl::DrawBufferMask enabledBuffers)
{
    for (uint32_t colorIndex = 0; colorIndex < this->numColorAttachments; ++colorIndex)
    {
        if (!enabledBuffers.test(colorIndex))
        {
            this->colorAttachments[colorIndex].writeMask = MTLColorWriteMaskNone;
        }
    }
}

// RenderPipelineDesc implementation
RenderPipelineDesc::RenderPipelineDesc()
{
    memset(this, 0, sizeof(*this));
    outputDescriptor.sampleCount = 1;
    rasterizationType            = RenderPipelineRasterization::Enabled;
}

RenderPipelineDesc::RenderPipelineDesc(const RenderPipelineDesc &src)
{
    memcpy(this, &src, sizeof(*this));
}

RenderPipelineDesc::RenderPipelineDesc(RenderPipelineDesc &&src)
{
    memcpy(this, &src, sizeof(*this));
}

RenderPipelineDesc &RenderPipelineDesc::operator=(const RenderPipelineDesc &src)
{
    memcpy(this, &src, sizeof(*this));
    return *this;
}

bool RenderPipelineDesc::operator==(const RenderPipelineDesc &rhs) const
{
    // NOTE(hqle): Use a faster way to compare, i.e take into account
    // the number of active vertex attributes & render targets.
    // If that way is used, hash() method must be changed also.
    return memcmp(this, &rhs, sizeof(*this)) == 0;
}

size_t RenderPipelineDesc::hash() const
{
    return angle::ComputeGenericHash(*this);
}

bool RenderPipelineDesc::rasterizationEnabled() const
{
    return rasterizationType != RenderPipelineRasterization::Disabled;
}

// RenderPassAttachmentTextureTargetDesc implementation
void RenderPassAttachmentTextureTargetDesc::reset()
{
    targetTexture.reset();
    targetImplicitMSTexture.reset();
    targetLevel        = 0;
    targetSliceOrDepth = 0;
}

// RenderPassDesc implementation
RenderPassAttachmentDesc::RenderPassAttachmentDesc()
{
    reset();
}

void RenderPassAttachmentDesc::reset()
{
    RenderPassAttachmentTextureTargetDesc::reset();

    loadAction         = MTLLoadActionLoad;
    storeAction        = MTLStoreActionStore;
    storeActionOptions = MTLStoreActionOptionNone;
}

bool RenderPassAttachmentDesc::equalIgnoreLoadStoreOptions(
    const RenderPassAttachmentDesc &other) const
{
    return texture() == other.texture() && implicitMSTexture() == other.implicitMSTexture() &&
           level() == other.level() && sliceOrDepth() == other.sliceOrDepth();
}

bool RenderPassAttachmentDesc::operator==(const RenderPassAttachmentDesc &other) const
{
    if (!equalIgnoreLoadStoreOptions(other))
    {
        return false;
    }

    return loadAction == other.loadAction && storeAction == other.storeAction &&
           storeActionOptions == other.storeActionOptions;
}

void RenderPassDesc::populateRenderPipelineOutputDesc(RenderPipelineOutputDesc *outDesc) const
{
    populateRenderPipelineOutputDesc(MTLColorWriteMaskAll, outDesc);
}

void RenderPassDesc::populateRenderPipelineOutputDesc(MTLColorWriteMask colorWriteMask,
                                                      RenderPipelineOutputDesc *outDesc) const
{
    // Default blend state.
    BlendDesc blendState;
    blendState.reset(colorWriteMask);

    populateRenderPipelineOutputDesc(blendState, outDesc);
}

void RenderPassDesc::populateRenderPipelineOutputDesc(const BlendDesc &blendState,
                                                      RenderPipelineOutputDesc *outDesc) const
{
    auto &outputDescriptor               = *outDesc;
    outputDescriptor.numColorAttachments = this->numColorAttachments;
    outputDescriptor.sampleCount         = this->sampleCount;
    for (uint32_t i = 0; i < this->numColorAttachments; ++i)
    {
        auto &renderPassColorAttachment = this->colorAttachments[i];
        auto texture                    = renderPassColorAttachment.texture();

        if (texture)
        {
            // Copy parameters from blend state
            outputDescriptor.colorAttachments[i].update(blendState);

            outputDescriptor.colorAttachments[i].pixelFormat = texture->pixelFormat();

            // Combine the masks. This is useful when the texture is not supposed to have alpha
            // channel such as GL_RGB8, however, Metal doesn't natively support 24 bit RGB, so
            // we need to use RGBA texture, and then disable alpha write to this texture
            outputDescriptor.colorAttachments[i].writeMask &= texture->getColorWritableMask();
        }
        else
        {
            outputDescriptor.colorAttachments[i].blendingEnabled = false;
            outputDescriptor.colorAttachments[i].pixelFormat     = MTLPixelFormatInvalid;
        }
    }

    // Reset the unused output slots to ensure consistent hash value
    for (uint32_t i = this->numColorAttachments; i < kMaxRenderTargets; ++i)
    {
        outputDescriptor.colorAttachments[i].reset();
    }

    auto depthTexture = this->depthAttachment.texture();
    outputDescriptor.depthAttachmentPixelFormat =
        depthTexture ? depthTexture->pixelFormat() : MTLPixelFormatInvalid;

    auto stencilTexture = this->stencilAttachment.texture();
    outputDescriptor.stencilAttachmentPixelFormat =
        stencilTexture ? stencilTexture->pixelFormat() : MTLPixelFormatInvalid;
}

bool RenderPassDesc::equalIgnoreLoadStoreOptions(const RenderPassDesc &other) const
{
    if (numColorAttachments != other.numColorAttachments)
    {
        return false;
    }

    for (uint32_t i = 0; i < numColorAttachments; ++i)
    {
        auto &renderPassColorAttachment = colorAttachments[i];
        auto &otherRPAttachment         = other.colorAttachments[i];
        if (!renderPassColorAttachment.equalIgnoreLoadStoreOptions(otherRPAttachment))
        {
            return false;
        }
    }

    return depthAttachment.equalIgnoreLoadStoreOptions(other.depthAttachment) &&
           stencilAttachment.equalIgnoreLoadStoreOptions(other.stencilAttachment);
}

bool RenderPassDesc::operator==(const RenderPassDesc &other) const
{
    if (numColorAttachments != other.numColorAttachments)
    {
        return false;
    }

    for (uint32_t i = 0; i < numColorAttachments; ++i)
    {
        auto &renderPassColorAttachment = colorAttachments[i];
        auto &otherRPAttachment         = other.colorAttachments[i];
        if (renderPassColorAttachment != (otherRPAttachment))
        {
            return false;
        }
    }

    return depthAttachment == other.depthAttachment && stencilAttachment == other.stencilAttachment;
}

// Convert to Metal object
void RenderPassDesc::convertToMetalDesc(MTLRenderPassDescriptor *objCDesc) const
{
    ANGLE_MTL_OBJC_SCOPE
    {
        for (uint32_t i = 0; i < numColorAttachments; ++i)
        {
            ToObjC(colorAttachments[i], objCDesc.colorAttachments[i]);
        }
        for (uint32_t i = numColorAttachments; i < kMaxRenderTargets; ++i)
        {
            // Inactive render target
            objCDesc.colorAttachments[i].texture     = nil;
            objCDesc.colorAttachments[i].level       = 0;
            objCDesc.colorAttachments[i].slice       = 0;
            objCDesc.colorAttachments[i].depthPlane  = 0;
            objCDesc.colorAttachments[i].loadAction  = MTLLoadActionDontCare;
            objCDesc.colorAttachments[i].storeAction = MTLStoreActionDontCare;
        }

        ToObjC(depthAttachment, objCDesc.depthAttachment);
        ToObjC(stencilAttachment, objCDesc.stencilAttachment);
    }
}

// RenderPipelineCache implementation
RenderPipelineCache::RenderPipelineCache() : RenderPipelineCache(nullptr) {}

RenderPipelineCache::RenderPipelineCache(
    RenderPipelineCacheSpecializeShaderFactory *specializedShaderFactory)
    : mSpecializedShaderFactory(specializedShaderFactory)
{}

RenderPipelineCache::~RenderPipelineCache() {}

void RenderPipelineCache::setVertexShader(Context *context, id<MTLFunction> shader)
{
    mVertexShader.retainAssign(shader);

    if (!shader)
    {
        clearPipelineStates();
        return;
    }

    recreatePipelineStates(context);
}

void RenderPipelineCache::setFragmentShader(Context *context, id<MTLFunction> shader)
{
    mFragmentShader.retainAssign(shader);

    if (!shader)
    {
        clearPipelineStates();
        return;
    }

    recreatePipelineStates(context);
}

bool RenderPipelineCache::hasDefaultAttribs(const RenderPipelineDesc &rpdesc) const
{
    const VertexDesc &desc = rpdesc.vertexDescriptor;
    for (uint8_t i = 0; i < desc.numAttribs; ++i)
    {
        if (desc.attributes[i].bufferIndex == kDefaultAttribsBindingIndex)
        {
            return true;
        }
    }

    return false;
}

AutoObjCPtr<id<MTLRenderPipelineState>> RenderPipelineCache::getRenderPipelineState(
    ContextMtl *context,
    const RenderPipelineDesc &desc)
{
    auto insertDefaultAttribLayout = hasDefaultAttribs(desc);
    int tableIdx                   = insertDefaultAttribLayout ? 1 : 0;
    auto &table                    = mRenderPipelineStates[tableIdx];
    auto ite                       = table.find(desc);
    if (ite == table.end())
    {
        return insertRenderPipelineState(context, desc, insertDefaultAttribLayout);
    }

    return ite->second;
}

AutoObjCPtr<id<MTLRenderPipelineState>> RenderPipelineCache::insertRenderPipelineState(
    Context *context,
    const RenderPipelineDesc &desc,
    bool insertDefaultAttribLayout)
{
    AutoObjCPtr<id<MTLRenderPipelineState>> newState =
        createRenderPipelineState(context, desc, insertDefaultAttribLayout);
    if (!newState)
    {
        return nil;
    }

    int tableIdx = insertDefaultAttribLayout ? 1 : 0;
    auto re      = mRenderPipelineStates[tableIdx].insert(std::make_pair(desc, newState));
    if (!re.second)
    {
        return nil;
    }

    return re.first->second;
}

AutoObjCPtr<id<MTLRenderPipelineState>> RenderPipelineCache::createRenderPipelineState(
    Context *context,
    const RenderPipelineDesc &originalDesc,
    bool insertDefaultAttribLayout)
{
    ANGLE_MTL_OBJC_SCOPE
    {
        // Disable coverage if the render pipeline's sample count is only 1.
        RenderPipelineDesc desc = originalDesc;
        if (desc.outputDescriptor.sampleCount == 1)
        {
            // Disable sample coverage if the output is not multisample
            desc.emulateCoverageMask    = false;
            desc.alphaToCoverageEnabled = false;
        }

        // Choose shader variant
        id<MTLFunction> vertShader = nil;
        id<MTLFunction> fragShader = nil;
        if (mSpecializedShaderFactory &&
            mSpecializedShaderFactory->hasSpecializedShader(gl::ShaderType::Vertex, desc))
        {
            if (IsError(mSpecializedShaderFactory->getSpecializedShader(
                    context, gl::ShaderType::Vertex, desc, &vertShader)))
            {
                return nil;
            }
        }
        else
        {
            // Non-specialized version
            vertShader = mVertexShader;
        }

        if (mSpecializedShaderFactory &&
            mSpecializedShaderFactory->hasSpecializedShader(gl::ShaderType::Fragment, desc))
        {
            if (IsError(mSpecializedShaderFactory->getSpecializedShader(
                    context, gl::ShaderType::Fragment, desc, &fragShader)))
            {
                return nil;
            }
        }
        else
        {
            // Non-specialized version
            fragShader = mFragmentShader;
        }

        if (!vertShader)
        {
            // Render pipeline without vertex shader is invalid.
            context->handleError(GL_INVALID_OPERATION, __FILE__, ANGLE_FUNCTION, __LINE__);
            return nil;
        }

        id<MTLDevice> metalDevice = context->getMetalDevice();

        // Convert to Objective-C desc:
        AutoObjCObj<MTLRenderPipelineDescriptor> objCDesc = ToObjC(vertShader, fragShader, desc);

        // Special attribute slot for default attribute
        if (insertDefaultAttribLayout)
        {
            MTLVertexBufferLayoutDescriptor *defaultAttribLayoutObjCDesc =
                [[MTLVertexBufferLayoutDescriptor alloc] init];
            defaultAttribLayoutObjCDesc.stepFunction = MTLVertexStepFunctionConstant;
            defaultAttribLayoutObjCDesc.stepRate     = 0;
            defaultAttribLayoutObjCDesc.stride       = kDefaultAttributeSize * kMaxVertexAttribs;

            [objCDesc.get().vertexDescriptor.layouts
                         setObject:[defaultAttribLayoutObjCDesc ANGLE_MTL_AUTORELEASE]
                atIndexedSubscript:kDefaultAttribsBindingIndex];
        }
        // Create pipeline state
        NSError *err = nil;
        id<MTLRenderPipelineState> newState =
            [metalDevice newRenderPipelineStateWithDescriptor:objCDesc error:&err];
        if (err)
        {
            context->handleError(err, __FILE__, ANGLE_FUNCTION, __LINE__);
            return nil;
        }

        return [newState ANGLE_MTL_AUTORELEASE];
    }
}

void RenderPipelineCache::recreatePipelineStates(Context *context)
{
    for (int hasDefaultAttrib = 0; hasDefaultAttrib <= 1; ++hasDefaultAttrib)
    {
        for (auto &ite : mRenderPipelineStates[hasDefaultAttrib])
        {
            if (ite.second == nil)
            {
                continue;
            }

            ite.second = createRenderPipelineState(context, ite.first, hasDefaultAttrib);
        }
    }
}

void RenderPipelineCache::clear()
{
    mVertexShader   = nil;
    mFragmentShader = nil;
    clearPipelineStates();
}

void RenderPipelineCache::clearPipelineStates()
{
    mRenderPipelineStates[0].clear();
    mRenderPipelineStates[1].clear();
}

// ClientIndexArrayKey implementation
ClientIndexArrayKey::ClientIndexArrayKey() = default;

ClientIndexArrayKey::ClientIndexArrayKey(ClientIndexArrayKey &&rhs)
{
    (*this) = std::move(rhs);
}

ClientIndexArrayKey::ClientIndexArrayKey(const ClientIndexArrayKey &rhs)
{
    (*this) = rhs;
}

void ClientIndexArrayKey::assign(const void *data, gl::DrawElementsType type, size_t count)
{
    mtl::SmallVector tmp(static_cast<const uint8_t *>(data),
                         count * gl::GetDrawElementsTypeSize(type));

    mBytes = std::move(tmp);

    mWrappedBytes = nullptr;
    mWrappedSize  = 0;

    mType = type;

    mIsHashCached = false;
}

void ClientIndexArrayKey::wrap(const void *data, gl::DrawElementsType type, size_t count)
{
    mBytes.clear();

    mWrappedBytes = data;
    mWrappedSize  = count * gl::GetDrawElementsTypeSize(type);

    mType = type;

    mIsHashCached = false;
}

ClientIndexArrayKey &ClientIndexArrayKey::operator=(ClientIndexArrayKey &&rhs)
{
    if (rhs.isWrapping())
    {
        // Make a copy. We want to store it in the cache.
        assign(rhs.data(), rhs.type(), rhs.elementsCount());
    }
    else
    {
        mWrappedBytes = nullptr;
        mWrappedSize  = 0;

        mBytes = std::move(rhs.mBytes);
    }

    mType         = rhs.mType;
    mIsHashCached = false;

    return *this;
}
ClientIndexArrayKey &ClientIndexArrayKey::operator=(const ClientIndexArrayKey &rhs)
{
    assign(rhs.data(), rhs.type(), rhs.elementsCount());
    return *this;
}

const void *ClientIndexArrayKey::data() const
{
    return mWrappedBytes ? mWrappedBytes : mBytes.data();
}

size_t ClientIndexArrayKey::size() const
{
    return mWrappedBytes ? mWrappedSize : mBytes.size();
}

size_t ClientIndexArrayKey::elementsCount() const
{
    return size() / gl::GetDrawElementsTypeSize(mType);
}

bool ClientIndexArrayKey::isWrapping() const
{
    return mWrappedBytes;
}

void ClientIndexArrayKey::clear()
{
    mBytes.clear();
    mWrappedBytes = nullptr;
    mWrappedSize  = 0;
    mIsHashCached = false;
}

size_t ClientIndexArrayKey::hash() const
{
    if (mIsHashCached)
    {
        return mCachedHash;
    }

    auto bytePtr       = static_cast<const uint8_t *>(data());
    size_t totalBytes  = size();
    size_t multipleOf4 = (totalBytes >> 2) << 2;
    mCachedHash        = angle::ComputeGenericHash(bytePtr, multipleOf4);

    // Combine hash with the remaining bytes
    for (size_t i = multipleOf4, shift = 0; i < totalBytes; ++i, shift += 8)
    {
        mCachedHash ^= (*(bytePtr + i)) << shift;
    }

    mCachedHash ^= static_cast<uint8_t>(mType);

    mIsHashCached = true;

    return mCachedHash;
}

bool ClientIndexArrayKey::operator==(const ClientIndexArrayKey &rhs) const
{
    if (mType != rhs.mType || size() != rhs.size())
    {
        return false;
    }
    return memcmp(data(), rhs.data(), size()) == 0;
}

// StateCache implementation
StateCache::StateCache(const angle::FeaturesMtl &features) : mFeatures(features) {}

StateCache::~StateCache() {}

AutoObjCPtr<id<MTLDepthStencilState>> StateCache::getNullDepthStencilState(id<MTLDevice> device)
{
    if (!mNullDepthStencilState)
    {
        DepthStencilDesc desc;
        desc.reset();
        ASSERT(desc.frontFaceStencil.stencilCompareFunction == MTLCompareFunctionAlways);
        desc.depthWriteEnabled = false;
        mNullDepthStencilState = getDepthStencilState(device, desc);
    }
    return mNullDepthStencilState;
}

AutoObjCPtr<id<MTLDepthStencilState>> StateCache::getDepthStencilState(id<MTLDevice> metalDevice,
                                                                       const DepthStencilDesc &desc)
{
    ANGLE_MTL_OBJC_SCOPE
    {
        auto ite = mDepthStencilStates.find(desc);
        if (ite == mDepthStencilStates.end())
        {
            AutoObjCObj<MTLDepthStencilDescriptor> objCDesc = ToObjC(desc);
            AutoObjCPtr<id<MTLDepthStencilState>> newState =
                [[metalDevice newDepthStencilStateWithDescriptor:objCDesc] ANGLE_MTL_AUTORELEASE];

            auto re = mDepthStencilStates.insert(std::make_pair(desc, newState));
            if (!re.second)
            {
                return nil;
            }

            ite = re.first;
        }

        return ite->second;
    }
}

AutoObjCPtr<id<MTLSamplerState>> StateCache::getSamplerState(id<MTLDevice> metalDevice,
                                                             const SamplerDesc &desc)
{
    ANGLE_MTL_OBJC_SCOPE
    {
        auto ite = mSamplerStates.find(desc);
        if (ite == mSamplerStates.end())
        {
            AutoObjCObj<MTLSamplerDescriptor> objCDesc = ToObjC(desc);
            if (!mFeatures.allowRuntimeSamplerCompareMode.enabled)
            {
                // Runtime sampler compare mode is not supported, fallback to never.
                objCDesc.get().compareFunction = MTLCompareFunctionNever;
            }
            AutoObjCPtr<id<MTLSamplerState>> newState =
                [[metalDevice newSamplerStateWithDescriptor:objCDesc] ANGLE_MTL_AUTORELEASE];

            auto re = mSamplerStates.insert(std::make_pair(desc, newState));
            if (!re.second)
                return nil;

            ite = re.first;
        }

        return ite->second;
    }
}

AutoObjCPtr<id<MTLSamplerState>> StateCache::getNullSamplerState(Context *context)
{
    return getNullSamplerState(context->getMetalDevice());
}

AutoObjCPtr<id<MTLSamplerState>> StateCache::getNullSamplerState(id<MTLDevice> device)
{
    SamplerDesc desc;
    desc.reset();

    return getSamplerState(device, desc);
}

void StateCache::clear()
{
    mNullDepthStencilState = nil;
    mDepthStencilStates.clear();
    mSamplerStates.clear();
}
}  // namespace mtl
}  // namespace rx
