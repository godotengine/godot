//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_format_utils.mm:
//      Implements Format conversion utilities classes that convert from angle formats
//      to respective MTLPixelFormat and MTLVertexFormat.
//

#include "libANGLE/renderer/metal/mtl_format_utils.h"

#include "common/debug.h"
#include "libANGLE/renderer/Format.h"
#include "libANGLE/renderer/load_functions_table.h"
#include "libANGLE/renderer/metal/DisplayMtl.h"

namespace rx
{
namespace mtl
{

namespace
{

bool OverrideTextureCaps(const DisplayMtl *display, angle::FormatID formatId, gl::TextureCaps *caps)
{
    // NOTE(hqle): Auto generate this.
    switch (formatId)
    {
        // NOTE: even though iOS devices don't support filtering depth textures, we still report as
        // supported here in order for the OES_depth_texture extension to be enabled.
        // During draw call, the filter modes will be converted to nearest.
        case angle::FormatID::D16_UNORM:
        case angle::FormatID::D24_UNORM_S8_UINT:
        case angle::FormatID::D32_FLOAT_S8X24_UINT:
        case angle::FormatID::D32_FLOAT:
        case angle::FormatID::D32_UNORM:
            caps->texturable = caps->filterable = caps->textureAttachment = caps->renderbuffer =
                true;
            return true;
        default:
            // NOTE(hqle): Handle more cases
            return false;
    }
}

void GenerateTextureCapsMap(const FormatTable &formatTable,
                            const DisplayMtl *display,
                            gl::TextureCapsMap *capsMapOut,
                            std::vector<GLenum> *compressedFormatsOut,
                            uint32_t *maxSamplesOut)
{
    auto &textureCapsMap    = *capsMapOut;
    auto &compressedFormats = *compressedFormatsOut;

    compressedFormats.clear();

    auto formatVerifier = [&](const gl::InternalFormat &internalFormatInfo) {
        angle::FormatID angleFormatId =
            angle::Format::InternalFormatToID(internalFormatInfo.sizedInternalFormat);
        const Format &mtlFormat = formatTable.getPixelFormat(angleFormatId);
        if (!mtlFormat.valid())
        {
            return;
        }
        const FormatCaps &formatCaps = mtlFormat.getCaps();

        const angle::Format &intendedAngleFormat = mtlFormat.intendedAngleFormat();
        gl::TextureCaps textureCaps;

        // First let check whether we can override certain special cases.
        if (!OverrideTextureCaps(display, mtlFormat.intendedFormatId, &textureCaps))
        {
            // Fill the texture caps using pixel format's caps
            textureCaps.filterable = mtlFormat.getCaps().filterable;
            textureCaps.renderbuffer =
                mtlFormat.getCaps().colorRenderable || mtlFormat.getCaps().depthRenderable;
            textureCaps.texturable        = true;
            textureCaps.textureAttachment = textureCaps.renderbuffer;
        }

        if (formatCaps.multisample)
        {
            constexpr uint32_t sampleCounts[] = {2, 4, 8};
            for (auto sampleCount : sampleCounts)
            {
                if ([display->getMetalDevice() supportsTextureSampleCount:sampleCount])
                {
                    textureCaps.sampleCounts.insert(sampleCount);
                    *maxSamplesOut = std::max(*maxSamplesOut, sampleCount);
                }
            }
        }

        textureCapsMap.set(mtlFormat.intendedFormatId, textureCaps);

        if (intendedAngleFormat.isBlock)
        {
            compressedFormats.push_back(intendedAngleFormat.glInternalFormat);
        }
    };

    // Texture caps map.
    const gl::FormatSet &internalFormats = gl::GetAllSizedInternalFormats();
    for (const auto internalFormat : internalFormats)
    {
        const gl::InternalFormat &internalFormatInfo =
            gl::GetSizedInternalFormatInfo(internalFormat);

        formatVerifier(internalFormatInfo);
    }
}

}  // namespace

// FormatBase implementation
const angle::Format &FormatBase::actualAngleFormat() const
{
    return angle::Format::Get(actualFormatId);
}

const angle::Format &FormatBase::intendedAngleFormat() const
{
    return angle::Format::Get(intendedFormatId);
}

// Format implementation
const gl::InternalFormat &Format::intendedInternalFormat() const
{
    return gl::GetSizedInternalFormatInfo(intendedAngleFormat().glInternalFormat);
}

bool Format::needConversion(angle::FormatID srcFormatId) const
{
    if ((srcFormatId == angle::FormatID::BC1_RGB_UNORM_BLOCK &&
         actualFormatId == angle::FormatID::BC1_RGBA_UNORM_BLOCK) ||
        (srcFormatId == angle::FormatID::BC1_RGB_UNORM_SRGB_BLOCK &&
         actualFormatId == angle::FormatID::BC1_RGBA_UNORM_SRGB_BLOCK))
    {
        // DXT1 RGB format already swizzled with alpha=1, so no need to convert
        ASSERT(swizzled);
        return false;
    }
    if (srcFormatId == angle::FormatID::ETC1_R8G8B8_UNORM_BLOCK &&
        actualFormatId == angle::FormatID::ETC2_R8G8B8_UNORM_BLOCK)
    {
        // ETC1 RGB & ETC2 RGB are technically the same.
        return false;
    }
    return srcFormatId != actualFormatId;
}

bool Format::isPVRTC() const
{
    switch (metalFormat)
    {
#if TARGET_OS_IOS && !TARGET_OS_MACCATALYST
        case MTLPixelFormatPVRTC_RGB_2BPP:
        case MTLPixelFormatPVRTC_RGB_2BPP_sRGB:
        case MTLPixelFormatPVRTC_RGB_4BPP:
        case MTLPixelFormatPVRTC_RGB_4BPP_sRGB:
        case MTLPixelFormatPVRTC_RGBA_2BPP:
        case MTLPixelFormatPVRTC_RGBA_2BPP_sRGB:
        case MTLPixelFormatPVRTC_RGBA_4BPP:
        case MTLPixelFormatPVRTC_RGBA_4BPP_sRGB:
            return true;
#endif
        default:
            return false;
    }
}

// FormatTable implementation
angle::Result FormatTable::initialize(const DisplayMtl *display)
{
    mMaxSamples = 0;

    // Initialize native format caps
    initNativeFormatCaps(display);

    for (size_t i = 0; i < angle::kNumANGLEFormats; ++i)
    {
        const auto formatId = static_cast<angle::FormatID>(i);

        mPixelFormatTable[i].init(display, formatId);
        mPixelFormatTable[i].caps = &mNativePixelFormatCapsTable[mPixelFormatTable[i].metalFormat];

        if (!mPixelFormatTable[i].caps->depthRenderable &&
            mPixelFormatTable[i].actualFormatId != mPixelFormatTable[i].intendedFormatId)
        {
            mPixelFormatTable[i].textureLoadFunctions = angle::GetLoadFunctionsMap(
                mPixelFormatTable[i].intendedAngleFormat().glInternalFormat,
                mPixelFormatTable[i].actualFormatId);
        }

        mVertexFormatTables[0][i].init(formatId, false);
        mVertexFormatTables[1][i].init(formatId, true);
    }

    // NOTE(hqle): Work-around AMD's issue that D24S8 format sometimes returns zero during sampling:
    if (display->getRendererDescription().find("AMD") != std::string::npos)
    {
        // Fallback to D32_FLOAT_S8X24_UINT.
        Format &format =
            mPixelFormatTable[static_cast<uint32_t>(angle::FormatID::D24_UNORM_S8_UINT)];
        format.actualFormatId       = angle::FormatID::D32_FLOAT_S8X24_UINT;
        format.metalFormat          = MTLPixelFormatDepth32Float_Stencil8;
        format.initFunction         = nullptr;
        format.textureLoadFunctions = nullptr;
        format.caps = &mNativePixelFormatCapsTable[MTLPixelFormatDepth32Float_Stencil8];
    }

    return angle::Result::Continue;
}

void FormatTable::generateTextureCaps(const DisplayMtl *display,
                                      gl::TextureCapsMap *capsMapOut,
                                      std::vector<GLenum> *compressedFormatsOut)
{
    GenerateTextureCapsMap(*this, display, capsMapOut, compressedFormatsOut, &mMaxSamples);
}

const Format &FormatTable::getPixelFormat(angle::FormatID angleFormatId) const
{
    return mPixelFormatTable[static_cast<size_t>(angleFormatId)];
}
const FormatCaps &FormatTable::getNativeFormatCaps(MTLPixelFormat mtlFormat) const
{
    ASSERT(mNativePixelFormatCapsTable.count(mtlFormat));
    return mNativePixelFormatCapsTable.at(mtlFormat);
}
const VertexFormat &FormatTable::getVertexFormat(angle::FormatID angleFormatId,
                                                 bool tightlyPacked) const
{
    auto tableIdx = tightlyPacked ? 1 : 0;
    return mVertexFormatTables[tableIdx][static_cast<size_t>(angleFormatId)];
}

void FormatTable::setFormatCaps(MTLPixelFormat formatId,
                                bool filterable,
                                bool writable,
                                bool blendable,
                                bool multisample,
                                bool resolve,
                                bool colorRenderable)
{
    setFormatCaps(formatId, filterable, writable, blendable, multisample, resolve, colorRenderable,
                  false);
}

void FormatTable::setFormatCaps(MTLPixelFormat id,
                                bool filterable,
                                bool writable,
                                bool blendable,
                                bool multisample,
                                bool resolve,
                                bool colorRenderable,
                                bool depthRenderable)
{
    mNativePixelFormatCapsTable[id].filterable      = filterable;
    mNativePixelFormatCapsTable[id].writable        = writable;
    mNativePixelFormatCapsTable[id].colorRenderable = colorRenderable;
    mNativePixelFormatCapsTable[id].depthRenderable = depthRenderable;
    mNativePixelFormatCapsTable[id].blendable       = blendable;
    mNativePixelFormatCapsTable[id].multisample     = multisample;
    mNativePixelFormatCapsTable[id].resolve         = resolve;
}

void FormatTable::setCompressedFormatCaps(MTLPixelFormat formatId, bool filterable)
{
    setFormatCaps(formatId, filterable, false, false, false, false, false, false);
}

void FormatTable::initNativeFormatCaps(const DisplayMtl *display)
{
    const angle::FeaturesMtl &featuresMtl = display->getFeatures();
    // Skip auto resolve if either hasDepth/StencilAutoResolve or allowMultisampleStoreAndResolve
    // feature are disabled.
    bool supportDepthAutoResolve = featuresMtl.hasDepthAutoResolve.enabled &&
                                   featuresMtl.allowMultisampleStoreAndResolve.enabled;
    bool supportStencilAutoResolve = featuresMtl.hasStencilAutoResolve.enabled &&
                                     featuresMtl.allowMultisampleStoreAndResolve.enabled;
    bool supportDepthStencilAutoResolve = supportDepthAutoResolve && supportStencilAutoResolve;

    // Source: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
    // clang-format off

    //            |  formatId                  | filterable    |  writable  |  blendable |  multisample |  resolve                              | colorRenderable |
    setFormatCaps(MTLPixelFormatA8Unorm,        true,            false,       false,          false,       false,                                false);
    setFormatCaps(MTLPixelFormatR8Unorm,        true,            true,        true,           true,        true,                                 true);
    setFormatCaps(MTLPixelFormatR8Snorm,        true,            true,        true,           true,      display->supportEitherGPUFamily(2, 1),  true);
    setFormatCaps(MTLPixelFormatR16Unorm,       true,            true,        true,           true,      display->supportMacGPUFamily(1),        true);
    setFormatCaps(MTLPixelFormatR16Snorm,       true,            true,        true,           true,      display->supportMacGPUFamily(1),        true);
    setFormatCaps(MTLPixelFormatRG8Unorm,       true,            true,        true,           true,        true,                                 true);
    setFormatCaps(MTLPixelFormatRG8Snorm,       true,            true,        true,           true,      display->supportEitherGPUFamily(2, 1),  true);
    setFormatCaps(MTLPixelFormatRG16Unorm,      true,            true,        true,           true,      display->supportMacGPUFamily(1),        true);
    setFormatCaps(MTLPixelFormatRG16Snorm,      true,            true,        true,           true,      display->supportMacGPUFamily(1),        true);
    setFormatCaps(MTLPixelFormatRGBA16Unorm,    true,            true,        true,           true,      display->supportMacGPUFamily(1),        true);
    setFormatCaps(MTLPixelFormatRGBA16Snorm,    true,            true,        true,           true,      display->supportMacGPUFamily(1),        true);
    setFormatCaps(MTLPixelFormatRGBA16Float,    true,            true,        true,           true,        true,                                 true);

    //            |  formatId                      | filterable    |  writable                         |  blendable |  multisample |  resolve                              | colorRenderable |
    setFormatCaps(MTLPixelFormatRGBA8Unorm,          true,            true,                               true,           true,        true,                                    true);
    setFormatCaps(MTLPixelFormatRGBA8Unorm_sRGB,     true,          display->supportiOSGPUFamily(2),      true,           true,        true,                                    true);
    setFormatCaps(MTLPixelFormatRGBA8Snorm,          true,            true,                               true,           true,     display->supportEitherGPUFamily(2, 1),      true);
    setFormatCaps(MTLPixelFormatBGRA8Unorm,          true,            true,                               true,           true,        true,                                    true);
    setFormatCaps(MTLPixelFormatBGRA8Unorm_sRGB,     true,          display->supportiOSGPUFamily(2),      true,           true,        true,                                    true);

    //            |  formatId              | filterable                    |  writable  |  blendable |  multisample |  resolve                              | colorRenderable |
    setFormatCaps(MTLPixelFormatR16Float,       true,                          true,        true,           true,        true,                                 true);
    setFormatCaps(MTLPixelFormatRG16Float,      true,                          true,        true,           true,        true,                                 true);
    setFormatCaps(MTLPixelFormatR32Float,    display->supportMacGPUFamily(1),  true,        true,           true,      display->supportMacGPUFamily(1),        true);

#if TARGET_OS_IOS && !TARGET_OS_MACCATALYST
    //            |  formatId                  | filterable    |  writable  |  blendable |  multisample |  resolve   | colorRenderable |
    setFormatCaps(MTLPixelFormatB5G6R5Unorm,      true,            false,        true,           true,        true,      true);
    setFormatCaps(MTLPixelFormatABGR4Unorm,       true,            false,        true,           true,        true,      true);
    setFormatCaps(MTLPixelFormatBGR5A1Unorm,      true,            false,        true,           true,        true,      true);
    setFormatCaps(MTLPixelFormatA1BGR5Unorm,      true,            false,        true,           true,        true,      true);
#endif

    //            |  formatId                  | filterable    |  writable                                 |  blendable |  multisample |  resolve   | colorRenderable |
    setFormatCaps(MTLPixelFormatBGR10A2Unorm,     true,         display->supportEitherGPUFamily(3, 1),       true,           true,        true,      true);
    setFormatCaps(MTLPixelFormatRGB10A2Unorm,     true,         display->supportEitherGPUFamily(3, 1),       true,           true,        true,      true);
    setFormatCaps(MTLPixelFormatRGB10A2Uint,      false,        display->supportEitherGPUFamily(3, 1),       false,          true,        false,     true);
    setFormatCaps(MTLPixelFormatRG11B10Float,     true,         display->supportEitherGPUFamily(3, 1),       true,           true,        true,      true);

    //            |  formatId                  | filterable    |  writable                         |  blendable                     |  multisample                    |  resolve                       | colorRenderable                 |
    setFormatCaps(MTLPixelFormatRGB9E5Float,       true,          display->supportiOSGPUFamily(3),  display->supportiOSGPUFamily(1),  display->supportiOSGPUFamily(1), display->supportiOSGPUFamily(1), display->supportiOSGPUFamily(1));

    //            |  formatId               | filterable    |  writable  |  blendable  |  multisample                        |  resolve      | colorRenderable |
    setFormatCaps(MTLPixelFormatR8Uint,        false,           true,        false,          true,                             false,         true);
    setFormatCaps(MTLPixelFormatR8Sint,        false,           true,        false,          true,                             false,         true);
    setFormatCaps(MTLPixelFormatR16Uint,       false,           true,        false,          true,                             false,         true);
    setFormatCaps(MTLPixelFormatR16Sint,       false,           true,        false,          true,                             false,         true);
    setFormatCaps(MTLPixelFormatRG8Uint,       false,           true,        false,          true,                             false,         true);
    setFormatCaps(MTLPixelFormatRG8Sint,       false,           true,        false,          true,                             false,         true);
    setFormatCaps(MTLPixelFormatR32Uint,       false,           true,        false,          display->supportMacGPUFamily(1),  false,         true);
    setFormatCaps(MTLPixelFormatR32Sint,       false,           true,        false,          display->supportMacGPUFamily(1),  false,         true);
    setFormatCaps(MTLPixelFormatRG16Uint,      false,           true,        false,          true,                             false,         true);
    setFormatCaps(MTLPixelFormatRG16Sint,      false,           true,        false,          true,                             false,         true);
    setFormatCaps(MTLPixelFormatRGBA8Uint,     false,           true,        false,          true,                             false,         true);
    setFormatCaps(MTLPixelFormatRGBA8Sint,     false,           true,        false,          true,                             false,         true);
    setFormatCaps(MTLPixelFormatRG32Uint,      false,           true,        false,          display->supportMacGPUFamily(1),  false,         true);
    setFormatCaps(MTLPixelFormatRG32Sint,      false,           true,        false,          display->supportMacGPUFamily(1),  false,         true);
    setFormatCaps(MTLPixelFormatRGBA16Uint,    false,           true,        false,          true,                             false,         true);
    setFormatCaps(MTLPixelFormatRGBA16Sint,    false,           true,        false,          true,                             false,         true);
    setFormatCaps(MTLPixelFormatRGBA32Uint,    false,           true,        false,          display->supportMacGPUFamily(1),  false,         true);
    setFormatCaps(MTLPixelFormatRGBA32Sint,    false,           true,        false,          display->supportMacGPUFamily(1),  false,         true);

    //            |  formatId                   | filterable                      |  writable  |  blendable                     |  multisample                     |  resolve                         | colorRenderable |
    setFormatCaps(MTLPixelFormatRG32Float,       display->supportMacGPUFamily(1),   true,        true,                            display->supportMacGPUFamily(1),  display->supportMacGPUFamily(1),         true);
    setFormatCaps(MTLPixelFormatRGBA32Float,     display->supportMacGPUFamily(1),   true,        display->supportMacGPUFamily(1), display->supportMacGPUFamily(1),  display->supportMacGPUFamily(1),         true);

    //            |  formatId                           | filterable                       |  writable  |  blendable |  multisample |  resolve                                | colorRenderable | depthRenderable                    |
    setFormatCaps(MTLPixelFormatDepth32Float,               display->supportMacGPUFamily(1),   false,        false,           true,    supportDepthAutoResolve,                    false,            true);
    setFormatCaps(MTLPixelFormatStencil8,                   false,                             false,        false,           true,    false,                                      false,            true);
    setFormatCaps(MTLPixelFormatDepth32Float_Stencil8,      display->supportMacGPUFamily(1),   false,        false,           true,    supportDepthStencilAutoResolve,             false,            true);
#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
    setFormatCaps(MTLPixelFormatDepth16Unorm,               true,                              false,        false,           true,    supportDepthAutoResolve,                    false,            true);
    setFormatCaps(MTLPixelFormatDepth24Unorm_Stencil8,      display->supportMacGPUFamily(1),   false,        false,           true,    supportDepthStencilAutoResolve,             false,            display->supportMacGPUFamily(1));

    setCompressedFormatCaps(MTLPixelFormatBC1_RGBA, true);
    setCompressedFormatCaps(MTLPixelFormatBC1_RGBA_sRGB, true);
    setCompressedFormatCaps(MTLPixelFormatBC2_RGBA, true);
    setCompressedFormatCaps(MTLPixelFormatBC2_RGBA_sRGB, true);
    setCompressedFormatCaps(MTLPixelFormatBC3_RGBA, true);
    setCompressedFormatCaps(MTLPixelFormatBC3_RGBA_sRGB, true);
#else
    setCompressedFormatCaps(MTLPixelFormatPVRTC_RGB_2BPP, true);
    setCompressedFormatCaps(MTLPixelFormatPVRTC_RGB_2BPP_sRGB, true);
    setCompressedFormatCaps(MTLPixelFormatPVRTC_RGB_4BPP, true);
    setCompressedFormatCaps(MTLPixelFormatPVRTC_RGB_4BPP_sRGB, true);
    setCompressedFormatCaps(MTLPixelFormatPVRTC_RGBA_2BPP, true);
    setCompressedFormatCaps(MTLPixelFormatPVRTC_RGBA_2BPP_sRGB, true);
    setCompressedFormatCaps(MTLPixelFormatPVRTC_RGBA_4BPP, true);
    setCompressedFormatCaps(MTLPixelFormatPVRTC_RGBA_4BPP_sRGB, true);
    setCompressedFormatCaps(MTLPixelFormatEAC_R11Unorm, true);
    setCompressedFormatCaps(MTLPixelFormatEAC_R11Snorm, true);
    setCompressedFormatCaps(MTLPixelFormatEAC_RG11Unorm, true);
    setCompressedFormatCaps(MTLPixelFormatEAC_RG11Snorm, true);
    setCompressedFormatCaps(MTLPixelFormatEAC_RGBA8, true);
    setCompressedFormatCaps(MTLPixelFormatEAC_RGBA8_sRGB, true);
    setCompressedFormatCaps(MTLPixelFormatETC2_RGB8, true);
    setCompressedFormatCaps(MTLPixelFormatETC2_RGB8_sRGB, true);
    setCompressedFormatCaps(MTLPixelFormatETC2_RGB8A1, true);
    setCompressedFormatCaps(MTLPixelFormatETC2_RGB8A1_sRGB, true);
#endif
    // clang-format on
}

}  // namespace mtl
}  // namespace rx
