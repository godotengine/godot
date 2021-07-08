//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "libANGLE/Caps.h"

#include "anglebase/no_destructor.h"
#include "common/angleutils.h"
#include "common/debug.h"

#include "libANGLE/formatutils.h"

#include "angle_gl.h"

#include <algorithm>
#include <sstream>

static void InsertExtensionString(const std::string &extension,
                                  bool supported,
                                  std::vector<std::string> *extensionVector)
{
    if (supported)
    {
        extensionVector->push_back(extension);
    }
}

namespace gl
{

TextureCaps::TextureCaps() = default;

TextureCaps::TextureCaps(const TextureCaps &other) = default;

TextureCaps::~TextureCaps() = default;

GLuint TextureCaps::getMaxSamples() const
{
    return !sampleCounts.empty() ? *sampleCounts.rbegin() : 0;
}

GLuint TextureCaps::getNearestSamples(GLuint requestedSamples) const
{
    if (requestedSamples == 0)
    {
        return 0;
    }

    for (SupportedSampleSet::const_iterator i = sampleCounts.begin(); i != sampleCounts.end(); i++)
    {
        GLuint samples = *i;
        if (samples >= requestedSamples)
        {
            return samples;
        }
    }

    return 0;
}

TextureCaps GenerateMinimumTextureCaps(GLenum sizedInternalFormat,
                                       const Version &clientVersion,
                                       const Extensions &extensions)
{
    TextureCaps caps;

    const InternalFormat &internalFormatInfo = GetSizedInternalFormatInfo(sizedInternalFormat);
    caps.texturable        = internalFormatInfo.textureSupport(clientVersion, extensions);
    caps.filterable        = internalFormatInfo.filterSupport(clientVersion, extensions);
    caps.textureAttachment = internalFormatInfo.textureAttachmentSupport(clientVersion, extensions);
    caps.renderbuffer      = internalFormatInfo.renderbufferSupport(clientVersion, extensions);

    caps.sampleCounts.insert(0);
    if (internalFormatInfo.isRequiredRenderbufferFormat(clientVersion))
    {
        if ((clientVersion.major >= 3 && clientVersion.minor >= 1) ||
            (clientVersion.major >= 3 && !internalFormatInfo.isInt()))
        {
            caps.sampleCounts.insert(4);
        }
    }

    return caps;
}

TextureCapsMap::TextureCapsMap() {}

TextureCapsMap::~TextureCapsMap() {}

void TextureCapsMap::insert(GLenum internalFormat, const TextureCaps &caps)
{
    angle::FormatID formatID = angle::Format::InternalFormatToID(internalFormat);
    get(formatID)            = caps;
}

void TextureCapsMap::clear()
{
    mFormatData.fill(TextureCaps());
}

const TextureCaps &TextureCapsMap::get(GLenum internalFormat) const
{
    angle::FormatID formatID = angle::Format::InternalFormatToID(internalFormat);
    return get(formatID);
}

const TextureCaps &TextureCapsMap::get(angle::FormatID formatID) const
{
    return mFormatData[static_cast<size_t>(formatID)];
}

TextureCaps &TextureCapsMap::get(angle::FormatID formatID)
{
    return mFormatData[static_cast<size_t>(formatID)];
}

void TextureCapsMap::set(angle::FormatID formatID, const TextureCaps &caps)
{
    get(formatID) = caps;
}

void InitMinimumTextureCapsMap(const Version &clientVersion,
                               const Extensions &extensions,
                               TextureCapsMap *capsMap)
{
    for (GLenum internalFormat : GetAllSizedInternalFormats())
    {
        capsMap->insert(internalFormat,
                        GenerateMinimumTextureCaps(internalFormat, clientVersion, extensions));
    }
}

Extensions::Extensions() = default;

Extensions::Extensions(const Extensions &other) = default;

std::vector<std::string> Extensions::getStrings() const
{
    std::vector<std::string> extensionStrings;

    for (const auto &extensionInfo : GetExtensionInfoMap())
    {
        if (this->*(extensionInfo.second.ExtensionsMember))
        {
            extensionStrings.push_back(extensionInfo.first);
        }
    }

    return extensionStrings;
}

Limitations::Limitations() = default;

static bool GetFormatSupportBase(const TextureCapsMap &textureCaps,
                                 const GLenum *requiredFormats,
                                 size_t requiredFormatsSize,
                                 bool requiresTexturing,
                                 bool requiresFiltering,
                                 bool requiresAttachingTexture,
                                 bool requiresRenderbufferSupport)
{
    for (size_t i = 0; i < requiredFormatsSize; i++)
    {
        const TextureCaps &cap = textureCaps.get(requiredFormats[i]);

        if (requiresTexturing && !cap.texturable)
        {
            return false;
        }

        if (requiresFiltering && !cap.filterable)
        {
            return false;
        }

        if (requiresAttachingTexture && !cap.textureAttachment)
        {
            return false;
        }

        if (requiresRenderbufferSupport && !cap.renderbuffer)
        {
            return false;
        }
    }

    return true;
}

template <size_t N>
static bool GetFormatSupport(const TextureCapsMap &textureCaps,
                             const GLenum (&requiredFormats)[N],
                             bool requiresTexturing,
                             bool requiresFiltering,
                             bool requiresAttachingTexture,
                             bool requiresRenderbufferSupport)
{
    return GetFormatSupportBase(textureCaps, requiredFormats, N, requiresTexturing,
                                requiresFiltering, requiresAttachingTexture,
                                requiresRenderbufferSupport);
}

// Check for GL_OES_packed_depth_stencil
static bool DeterminePackedDepthStencilSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_DEPTH24_STENCIL8,
    };

    return GetFormatSupport(textureCaps, requiredFormats, false, false, true, true);
}

// Checks for GL_OES_rgb8_rgba8 support
static bool DetermineRGB8AndRGBA8TextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_RGB8,
        GL_RGBA8,
    };

    return GetFormatSupport(textureCaps, requiredFormats, false, false, false, true);
}

// Checks for GL_EXT_texture_format_BGRA8888 support
static bool DetermineBGRA8TextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_BGRA8_EXT,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, true, true);
}

// Checks for GL_OES_color_buffer_half_float support
static bool DetermineColorBufferHalfFloatSupport(const TextureCapsMap &textureCaps)
{
    // EXT_color_buffer_half_float issue #2 states that an implementation doesn't need to support
    // rendering to any of the formats but is expected to be able to render to at least one. WebGL
    // requires that at least RGBA16F is renderable so we make the same requirement.
    constexpr GLenum requiredFormats[] = {
        GL_RGBA16F,
    };

    return GetFormatSupport(textureCaps, requiredFormats, false, false, true, true);
}

// Checks for GL_OES_texture_half_float support
static bool DetermineHalfFloatTextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_RGBA16F, GL_RGB16F, GL_LUMINANCE_ALPHA16F_EXT, GL_LUMINANCE16F_EXT, GL_ALPHA16F_EXT,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, false, false, false);
}

// Checks for GL_OES_texture_half_float_linear support
static bool DetermineHalfFloatTextureFilteringSupport(const TextureCapsMap &textureCaps,
                                                      bool checkLegacyFormats)
{
    constexpr GLenum requiredFormats[] = {GL_RGBA16F, GL_RGB16F};
    // If GL_OES_texture_half_float is present, this extension must also support legacy formats
    // introduced by that extension
    constexpr GLenum requiredFormatsES2[] = {GL_LUMINANCE_ALPHA16F_EXT, GL_LUMINANCE16F_EXT,
                                             GL_ALPHA16F_EXT};

    if (checkLegacyFormats &&
        !GetFormatSupport(textureCaps, requiredFormatsES2, false, true, false, false))
    {
        return false;
    }

    return GetFormatSupport(textureCaps, requiredFormats, false, true, false, false);
}

// Checks for GL_OES_texture_float support
static bool DetermineFloatTextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_RGBA32F, GL_RGB32F, GL_LUMINANCE_ALPHA32F_EXT, GL_LUMINANCE32F_EXT, GL_ALPHA32F_EXT,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, false, false, false);
}

// Checks for GL_OES_texture_float_linear support
static bool DetermineFloatTextureFilteringSupport(const TextureCapsMap &textureCaps,
                                                  bool checkLegacyFormats)
{
    constexpr GLenum requiredFormats[] = {
        GL_RGBA32F,
        GL_RGB32F,
    };
    // If GL_OES_texture_float is present, this extension must also support legacy formats
    // introduced by that extension
    constexpr GLenum requiredFormatsES2[] = {
        GL_LUMINANCE_ALPHA32F_EXT,
        GL_LUMINANCE32F_EXT,
        GL_ALPHA32F_EXT,
    };

    if (checkLegacyFormats &&
        !GetFormatSupport(textureCaps, requiredFormatsES2, false, true, false, false))
    {
        return false;
    }

    return GetFormatSupport(textureCaps, requiredFormats, false, true, false, false);
}

// Checks for GL_EXT_texture_rg support
static bool DetermineRGTextureSupport(const TextureCapsMap &textureCaps,
                                      bool checkHalfFloatFormats,
                                      bool checkFloatFormats)
{
    constexpr GLenum requiredFormats[] = {
        GL_R8,
        GL_RG8,
    };
    constexpr GLenum requiredHalfFloatFormats[] = {
        GL_R16F,
        GL_RG16F,
    };
    constexpr GLenum requiredFloatFormats[] = {
        GL_R32F,
        GL_RG32F,
    };

    if (checkHalfFloatFormats &&
        !GetFormatSupport(textureCaps, requiredHalfFloatFormats, true, false, false, false))
    {
        return false;
    }

    if (checkFloatFormats &&
        !GetFormatSupport(textureCaps, requiredFloatFormats, true, false, false, false))
    {
        return false;
    }

    return GetFormatSupport(textureCaps, requiredFormats, true, true, true, true);
}

// Check for GL_EXT_texture_compression_dxt1
static bool DetermineDXT1TextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_RGB_S3TC_DXT1_EXT,
        GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for GL_ANGLE_texture_compression_dxt3
static bool DetermineDXT3TextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_RGBA_S3TC_DXT3_ANGLE,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for GL_ANGLE_texture_compression_dxt5
static bool DetermineDXT5TextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_RGBA_S3TC_DXT5_ANGLE,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for GL_EXT_texture_compression_s3tc_srgb
static bool DetermineS3TCsRGBTextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_SRGB_S3TC_DXT1_EXT,
        GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT,
        GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT,
        GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for GL_KHR_texture_compression_astc_ldr
static bool DetermineASTCLDRTextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_RGBA_ASTC_4x4_KHR,           GL_COMPRESSED_RGBA_ASTC_5x4_KHR,
        GL_COMPRESSED_RGBA_ASTC_5x5_KHR,           GL_COMPRESSED_RGBA_ASTC_6x5_KHR,
        GL_COMPRESSED_RGBA_ASTC_6x6_KHR,           GL_COMPRESSED_RGBA_ASTC_8x5_KHR,
        GL_COMPRESSED_RGBA_ASTC_8x6_KHR,           GL_COMPRESSED_RGBA_ASTC_8x8_KHR,
        GL_COMPRESSED_RGBA_ASTC_10x5_KHR,          GL_COMPRESSED_RGBA_ASTC_10x6_KHR,
        GL_COMPRESSED_RGBA_ASTC_10x8_KHR,          GL_COMPRESSED_RGBA_ASTC_10x10_KHR,
        GL_COMPRESSED_RGBA_ASTC_12x10_KHR,         GL_COMPRESSED_RGBA_ASTC_12x12_KHR,
        GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR,   GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR,
        GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR,   GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR,
        GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR,   GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR,
        GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR,   GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR,
        GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR,  GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR,
        GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR,  GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR,
        GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for GL_OES_texture_compression_astc
static bool DetermineASTCOESTExtureSupport(const TextureCapsMap &textureCaps)
{
    if (!DetermineASTCLDRTextureSupport(textureCaps))
    {
        return false;
    }

    // The OES version of the extension also requires the 3D ASTC formats
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_RGBA_ASTC_3x3x3_OES,         GL_COMPRESSED_RGBA_ASTC_4x3x3_OES,
        GL_COMPRESSED_RGBA_ASTC_4x4x3_OES,         GL_COMPRESSED_RGBA_ASTC_4x4x4_OES,
        GL_COMPRESSED_RGBA_ASTC_5x4x4_OES,         GL_COMPRESSED_RGBA_ASTC_5x5x4_OES,
        GL_COMPRESSED_RGBA_ASTC_5x5x5_OES,         GL_COMPRESSED_RGBA_ASTC_6x5x5_OES,
        GL_COMPRESSED_RGBA_ASTC_6x6x5_OES,         GL_COMPRESSED_RGBA_ASTC_6x6x6_OES,
        GL_COMPRESSED_SRGB8_ALPHA8_ASTC_3x3x3_OES, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x3x3_OES,
        GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4x3_OES, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4x4_OES,
        GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4x4_OES, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5x4_OES,
        GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5x5_OES, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5x5_OES,
        GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6x5_OES, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6x6_OES,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for GL_ETC1_RGB8_OES
static bool DetermineETC1RGB8TextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_ETC1_RGB8_OES,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for OES_compressed_ETC2_RGB8_texture
static bool DetermineETC2RGB8TextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_RGB8_ETC2,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for OES_compressed_ETC2_sRGB8_texture
static bool DetermineETC2sRGB8TextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_SRGB8_ETC2,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for OES_compressed_ETC2_punchthroughA_RGBA8_texture
static bool DetermineETC2PunchthroughARGB8TextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for OES_compressed_ETC2_punchthroughA_sRGB8_alpha_texture
static bool DetermineETC2PunchthroughAsRGB8AlphaTextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for OES_compressed_ETC2_RGBA8_texture
static bool DetermineETC2RGBA8TextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_RGBA8_ETC2_EAC,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for OES_compressed_ETC2_sRGB8_alpha8_texture
static bool DetermineETC2sRGB8Alpha8TextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for OES_compressed_EAC_R11_unsigned_texture
static bool DetermineEACR11UnsignedTextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_R11_EAC,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for OES_compressed_EAC_R11_signed_texture
static bool DetermineEACR11SignedTextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_SIGNED_R11_EAC,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for OES_compressed_EAC_RG11_unsigned_texture
static bool DetermineEACRG11UnsignedTextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_RG11_EAC,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for OES_compressed_EAC_RG11_signed_texture
static bool DetermineEACRG11SignedTextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_SIGNED_RG11_EAC,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for GL_EXT_sRGB
static bool DetermineSRGBTextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFilterFormats[] = {
        GL_SRGB8,
        GL_SRGB8_ALPHA8,
    };

    constexpr GLenum requiredRenderFormats[] = {
        GL_SRGB8_ALPHA8,
    };

    return GetFormatSupport(textureCaps, requiredFilterFormats, true, true, false, false) &&
           GetFormatSupport(textureCaps, requiredRenderFormats, true, false, true, true);
}

// Check for GL_ANGLE_depth_texture
static bool DetermineDepthTextureANGLESupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_DEPTH_COMPONENT16,
        GL_DEPTH_COMPONENT32_OES,
        GL_DEPTH24_STENCIL8_OES,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, true, true);
}

// Check for GL_OES_depth_texture
static bool DetermineDepthTextureOESSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_DEPTH_COMPONENT16,
        GL_DEPTH_COMPONENT32_OES,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, true, true, true);
}

// Check for GL_OES_depth24
static bool DetermineDepth24OESSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_DEPTH_COMPONENT24_OES,
    };

    return GetFormatSupport(textureCaps, requiredFormats, false, false, false, true);
}

// Check for GL_OES_depth32
static bool DetermineDepth32Support(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_DEPTH_COMPONENT32_OES,
    };

    return GetFormatSupport(textureCaps, requiredFormats, false, false, true, true);
}

// Check for GL_CHROMIUM_color_buffer_float_rgb
static bool DetermineColorBufferFloatRGBSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_RGB32F,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, false, true, false);
}

// Check for GL_CHROMIUM_color_buffer_float_rgba
static bool DetermineColorBufferFloatRGBASupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_RGBA32F,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, false, true, true);
}

// Check for GL_EXT_color_buffer_float
static bool DetermineColorBufferFloatSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_R16F, GL_RG16F, GL_RGBA16F, GL_R32F, GL_RG32F, GL_RGBA32F, GL_R11F_G11F_B10F,
    };

    return GetFormatSupport(textureCaps, requiredFormats, true, false, true, true);
}

// Check for GL_EXT_texture_norm16
static bool DetermineTextureNorm16Support(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFilterFormats[] = {
        GL_R16_EXT,       GL_RG16_EXT,       GL_RGB16_EXT,       GL_RGBA16_EXT,
        GL_R16_SNORM_EXT, GL_RG16_SNORM_EXT, GL_RGB16_SNORM_EXT, GL_RGBA16_SNORM_EXT,
    };

    constexpr GLenum requiredRenderFormats[] = {
        GL_R16_EXT,
        GL_RG16_EXT,
        GL_RGBA16_EXT,
    };

    return GetFormatSupport(textureCaps, requiredFilterFormats, true, true, false, false) &&
           GetFormatSupport(textureCaps, requiredRenderFormats, true, false, true, true);
}

// Check for EXT_texture_compression_bptc
static bool DetermineBPTCTextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_RGBA_BPTC_UNORM_EXT, GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT,
        GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT, GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT};

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for GL_IMG_texture_compression_pvrtc
static bool DeterminePVRTCTextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG, GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG,
        GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG, GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG};

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

// Check for GL_EXT_pvrtc_sRGB
static bool DeterminePVRTCsRGBTextureSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {
        GL_COMPRESSED_SRGB_PVRTC_2BPPV1_EXT, GL_COMPRESSED_SRGB_PVRTC_4BPPV1_EXT,
        GL_COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT, GL_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT};

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

bool DetermineCompressedTextureETCSupport(const TextureCapsMap &textureCaps)
{
    constexpr GLenum requiredFormats[] = {GL_COMPRESSED_R11_EAC,
                                          GL_COMPRESSED_SIGNED_R11_EAC,
                                          GL_COMPRESSED_RG11_EAC,
                                          GL_COMPRESSED_SIGNED_RG11_EAC,
                                          GL_COMPRESSED_RGB8_ETC2,
                                          GL_COMPRESSED_SRGB8_ETC2,
                                          GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2,
                                          GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2,
                                          GL_COMPRESSED_RGBA8_ETC2_EAC,
                                          GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC};

    return GetFormatSupport(textureCaps, requiredFormats, true, true, false, false);
}

void Extensions::setTextureExtensionSupport(const TextureCapsMap &textureCaps)
{
    // TODO(ynovikov): rgb8rgba8, colorBufferHalfFloat, textureHalfFloat, textureHalfFloatLinear,
    // textureFloat, textureFloatLinear, textureRG, sRGB, colorBufferFloatRGB, colorBufferFloatRGBA
    // and colorBufferFloat were verified. Verify the rest.
    packedDepthStencil    = DeterminePackedDepthStencilSupport(textureCaps);
    rgb8rgba8             = DetermineRGB8AndRGBA8TextureSupport(textureCaps);
    textureFormatBGRA8888 = DetermineBGRA8TextureSupport(textureCaps);
    textureHalfFloat      = DetermineHalfFloatTextureSupport(textureCaps);
    textureHalfFloatLinear =
        DetermineHalfFloatTextureFilteringSupport(textureCaps, textureHalfFloat);
    textureFloat           = DetermineFloatTextureSupport(textureCaps);
    textureFloatLinear     = DetermineFloatTextureFilteringSupport(textureCaps, textureFloat);
    textureRG              = DetermineRGTextureSupport(textureCaps, textureHalfFloat, textureFloat);
    colorBufferHalfFloat   = textureHalfFloat && DetermineColorBufferHalfFloatSupport(textureCaps);
    textureCompressionDXT1 = DetermineDXT1TextureSupport(textureCaps);
    textureCompressionDXT3 = DetermineDXT3TextureSupport(textureCaps);
    textureCompressionDXT5 = DetermineDXT5TextureSupport(textureCaps);
    textureCompressionS3TCsRGB   = DetermineS3TCsRGBTextureSupport(textureCaps);
    textureCompressionASTCLDRKHR = DetermineASTCLDRTextureSupport(textureCaps);
    textureCompressionASTCOES    = DetermineASTCOESTExtureSupport(textureCaps);
    compressedETC1RGB8Texture    = DetermineETC1RGB8TextureSupport(textureCaps);
    compressedETC2RGB8Texture    = DetermineETC2RGB8TextureSupport(textureCaps);
    compressedETC2sRGB8Texture   = DetermineETC2sRGB8TextureSupport(textureCaps);
    compressedETC2PunchthroughARGB8Texture =
        DetermineETC2PunchthroughARGB8TextureSupport(textureCaps);
    compressedETC2PunchthroughAsRGB8AlphaTexture =
        DetermineETC2PunchthroughAsRGB8AlphaTextureSupport(textureCaps);
    compressedETC2RGBA8Texture       = DetermineETC2RGBA8TextureSupport(textureCaps);
    compressedETC2sRGB8Alpha8Texture = DetermineETC2sRGB8Alpha8TextureSupport(textureCaps);
    compressedEACR11UnsignedTexture  = DetermineEACR11UnsignedTextureSupport(textureCaps);
    compressedEACR11SignedTexture    = DetermineEACR11SignedTextureSupport(textureCaps);
    compressedEACRG11UnsignedTexture = DetermineEACRG11UnsignedTextureSupport(textureCaps);
    compressedEACRG11SignedTexture   = DetermineEACRG11SignedTextureSupport(textureCaps);
    sRGB                             = DetermineSRGBTextureSupport(textureCaps);
    depthTextureANGLE                = DetermineDepthTextureANGLESupport(textureCaps);
    depthTextureOES                  = DetermineDepthTextureOESSupport(textureCaps);
    depth24OES                       = DetermineDepth24OESSupport(textureCaps);
    depth32                          = DetermineDepth32Support(textureCaps);
    colorBufferFloatRGB              = DetermineColorBufferFloatRGBSupport(textureCaps);
    colorBufferFloatRGBA             = DetermineColorBufferFloatRGBASupport(textureCaps);
    colorBufferFloat                 = DetermineColorBufferFloatSupport(textureCaps);
    textureNorm16                    = DetermineTextureNorm16Support(textureCaps);
    textureCompressionBPTC           = DetermineBPTCTextureSupport(textureCaps);
    compressedTexturePVRTC           = DeterminePVRTCTextureSupport(textureCaps);
    compressedTexturePVRTCsRGB       = DeterminePVRTCsRGBTextureSupport(textureCaps);
}

const ExtensionInfoMap &GetExtensionInfoMap()
{
    auto buildExtensionInfoMap = []() {
        auto enableableExtension = [](ExtensionInfo::ExtensionBool member) {
            ExtensionInfo info;
            info.Requestable      = true;
            info.ExtensionsMember = member;
            return info;
        };

        auto esOnlyExtension = [](ExtensionInfo::ExtensionBool member) {
            ExtensionInfo info;
            info.ExtensionsMember = member;
            return info;
        };

        // clang-format off
        ExtensionInfoMap map;
        map["GL_OES_element_index_uint"] = enableableExtension(&Extensions::elementIndexUint);
        map["GL_OES_packed_depth_stencil"] = esOnlyExtension(&Extensions::packedDepthStencil);
        map["GL_OES_get_program_binary"] = enableableExtension(&Extensions::getProgramBinary);
        map["GL_OES_rgb8_rgba8"] = enableableExtension(&Extensions::rgb8rgba8);
        map["GL_EXT_texture_format_BGRA8888"] = enableableExtension(&Extensions::textureFormatBGRA8888);
        map["GL_EXT_texture_type_2_10_10_10_REV"] = enableableExtension(&Extensions::textureFormat2101010REV);
        map["GL_EXT_read_format_bgra"] = esOnlyExtension(&Extensions::readFormatBGRA);
        map["GL_NV_pixel_buffer_object"] = enableableExtension(&Extensions::pixelBufferObject);
        map["GL_OES_mapbuffer"] = enableableExtension(&Extensions::mapBuffer);
        map["GL_EXT_map_buffer_range"] = enableableExtension(&Extensions::mapBufferRange);
        map["GL_EXT_color_buffer_half_float"] = enableableExtension(&Extensions::colorBufferHalfFloat);
        map["GL_OES_texture_half_float"] = enableableExtension(&Extensions::textureHalfFloat);
        map["GL_OES_texture_half_float_linear"] = enableableExtension(&Extensions::textureHalfFloatLinear);
        map["GL_OES_texture_float"] = enableableExtension(&Extensions::textureFloat);
        map["GL_OES_texture_float_linear"] = enableableExtension(&Extensions::textureFloatLinear);
        map["GL_EXT_texture_rg"] = enableableExtension(&Extensions::textureRG);
        map["GL_EXT_texture_compression_dxt1"] = enableableExtension(&Extensions::textureCompressionDXT1);
        map["GL_ANGLE_texture_compression_dxt3"] = enableableExtension(&Extensions::textureCompressionDXT3);
        map["GL_ANGLE_texture_compression_dxt5"] = enableableExtension(&Extensions::textureCompressionDXT5);
        map["GL_EXT_texture_compression_s3tc_srgb"] = enableableExtension(&Extensions::textureCompressionS3TCsRGB);
        map["GL_KHR_texture_compression_astc_ldr"] = enableableExtension(&Extensions::textureCompressionASTCLDRKHR);
        map["GL_KHR_texture_compression_astc_hdr"] = enableableExtension(&Extensions::textureCompressionASTCHDRKHR);
        map["GL_OES_texture_compression_astc"] = enableableExtension(&Extensions::textureCompressionASTCOES);
        map["GL_EXT_texture_compression_bptc"] = enableableExtension(&Extensions::textureCompressionBPTC);
        map["GL_OES_compressed_ETC1_RGB8_texture"] = enableableExtension(&Extensions::compressedETC1RGB8Texture);
        map["GL_OES_compressed_ETC2_RGB8_texture"] = enableableExtension(&Extensions::compressedETC2RGB8Texture);
        map["GL_OES_compressed_ETC2_sRGB8_texture"] = enableableExtension(&Extensions::compressedETC2sRGB8Texture);
        map["GL_OES_compressed_ETC2_punchthroughA_RGBA8_texture"] = enableableExtension(&Extensions::compressedETC2PunchthroughARGB8Texture);
        map["GL_OES_compressed_ETC2_punchthroughA_sRGB8_alpha_texture"] = enableableExtension(&Extensions::compressedETC2PunchthroughAsRGB8AlphaTexture);
        map["GL_OES_compressed_ETC2_RGBA8_texture"] = enableableExtension(&Extensions::compressedETC2RGBA8Texture);
        map["GL_OES_compressed_ETC2_sRGB8_alpha8_texture"] = enableableExtension(&Extensions::compressedETC2sRGB8Alpha8Texture);
        map["GL_OES_compressed_EAC_R11_unsigned_texture"] = enableableExtension(&Extensions::compressedEACR11UnsignedTexture);
        map["GL_OES_compressed_EAC_R11_signed_texture"] = enableableExtension(&Extensions::compressedEACR11SignedTexture);
        map["GL_OES_compressed_EAC_RG11_unsigned_texture"] = enableableExtension(&Extensions::compressedEACRG11UnsignedTexture);
        map["GL_OES_compressed_EAC_RG11_signed_texture"] = enableableExtension(&Extensions::compressedEACRG11SignedTexture);
        map["GL_ANGLE_compressed_texture_etc"] = enableableExtension(&Extensions::compressedTextureETC);
        map["GL_IMG_texture_compression_pvrtc"] = enableableExtension(&Extensions::compressedTexturePVRTC);
        map["GL_EXT_pvrtc_sRGB"] = enableableExtension(&Extensions::compressedTexturePVRTCsRGB);
        map["GL_EXT_sRGB"] = enableableExtension(&Extensions::sRGB);
        map["GL_ANGLE_depth_texture"] = esOnlyExtension(&Extensions::depthTextureANGLE);
        map["GL_OES_depth_texture"] = esOnlyExtension(&Extensions::depthTextureOES);
        map["GL_OES_depth24"] = esOnlyExtension(&Extensions::depth24OES);
        map["GL_OES_depth32"] = esOnlyExtension(&Extensions::depth32);
        map["GL_OES_texture_3D"] = enableableExtension(&Extensions::texture3DOES);
        map["GL_EXT_texture_storage"] = enableableExtension(&Extensions::textureStorage);
        map["GL_OES_texture_npot"] = enableableExtension(&Extensions::textureNPOT);
        map["GL_EXT_draw_buffers"] = enableableExtension(&Extensions::drawBuffers);
        map["GL_EXT_texture_filter_anisotropic"] = enableableExtension(&Extensions::textureFilterAnisotropic);
        map["GL_EXT_occlusion_query_boolean"] = enableableExtension(&Extensions::occlusionQueryBoolean);
        map["GL_NV_fence"] = esOnlyExtension(&Extensions::fence);
        map["GL_EXT_disjoint_timer_query"] = enableableExtension(&Extensions::disjointTimerQuery);
        map["GL_EXT_robustness"] = esOnlyExtension(&Extensions::robustness);
        map["GL_KHR_robust_buffer_access_behavior"] = esOnlyExtension(&Extensions::robustBufferAccessBehavior);
        map["GL_EXT_blend_minmax"] = enableableExtension(&Extensions::blendMinMax);
        map["GL_ANGLE_framebuffer_blit"] = enableableExtension(&Extensions::framebufferBlit);
        map["GL_ANGLE_framebuffer_multisample"] = enableableExtension(&Extensions::framebufferMultisample);
        map["GL_EXT_multisampled_render_to_texture"] = enableableExtension(&Extensions::multisampledRenderToTexture);
        map["GL_ANGLE_instanced_arrays"] = enableableExtension(&Extensions::instancedArraysANGLE);
        map["GL_EXT_instanced_arrays"] = enableableExtension(&Extensions::instancedArraysEXT);
        map["GL_ANGLE_pack_reverse_row_order"] = enableableExtension(&Extensions::packReverseRowOrder);
        map["GL_OES_standard_derivatives"] = enableableExtension(&Extensions::standardDerivatives);
        map["GL_EXT_shader_texture_lod"] = enableableExtension(&Extensions::shaderTextureLOD);
        map["GL_EXT_frag_depth"] = enableableExtension(&Extensions::fragDepth);
        map["GL_OVR_multiview"] = enableableExtension(&Extensions::multiview);
        map["GL_OVR_multiview2"] = enableableExtension(&Extensions::multiview2);
        map["GL_ANGLE_texture_usage"] = enableableExtension(&Extensions::textureUsage);
        map["GL_ANGLE_translated_shader_source"] = esOnlyExtension(&Extensions::translatedShaderSource);
        map["GL_OES_fbo_render_mipmap"] = enableableExtension(&Extensions::fboRenderMipmap);
        map["GL_EXT_discard_framebuffer"] = esOnlyExtension(&Extensions::discardFramebuffer);
        map["GL_EXT_debug_marker"] = esOnlyExtension(&Extensions::debugMarker);
        map["GL_OES_EGL_image"] = enableableExtension(&Extensions::eglImage);
        map["GL_OES_EGL_image_external"] = enableableExtension(&Extensions::eglImageExternal);
        map["GL_OES_EGL_image_external_essl3"] = enableableExtension(&Extensions::eglImageExternalEssl3);
        map["GL_MGL_EGL_image_cube"] = enableableExtension(&Extensions::eglImageCubeMGL);
        map["GL_OES_EGL_sync"] = esOnlyExtension(&Extensions::eglSync);
        map["GL_EXT_memory_object"] = enableableExtension(&Extensions::memoryObject);
        map["GL_EXT_memory_object_fd"] = enableableExtension(&Extensions::memoryObjectFd);
        map["GL_EXT_semaphore"] = enableableExtension(&Extensions::semaphore);
        map["GL_EXT_semaphore_fd"] = enableableExtension(&Extensions::semaphoreFd);
        map["GL_NV_EGL_stream_consumer_external"] = enableableExtension(&Extensions::eglStreamConsumerExternal);
        map["GL_EXT_unpack_subimage"] = enableableExtension(&Extensions::unpackSubimage);
        map["GL_NV_pack_subimage"] = enableableExtension(&Extensions::packSubimage);
        map["GL_EXT_color_buffer_float"] = enableableExtension(&Extensions::colorBufferFloat);
        map["GL_OES_vertex_half_float"] = enableableExtension(&Extensions::vertexHalfFloat);
        map["GL_OES_vertex_array_object"] = enableableExtension(&Extensions::vertexArrayObject);
        map["GL_OES_vertex_type_10_10_10_2"] = enableableExtension(&Extensions::vertexAttribType1010102);
        map["GL_KHR_debug"] = esOnlyExtension(&Extensions::debug);
        map["GL_OES_texture_border_clamp"] = enableableExtension(&Extensions::textureBorderClamp);
        // TODO(jmadill): Enable this when complete.
        //map["GL_KHR_no_error"] = esOnlyExtension(&Extensions::noError);
        map["GL_ANGLE_lossy_etc_decode"] = enableableExtension(&Extensions::lossyETCDecode);
        map["GL_CHROMIUM_bind_uniform_location"] = esOnlyExtension(&Extensions::bindUniformLocation);
        map["GL_CHROMIUM_sync_query"] = enableableExtension(&Extensions::syncQuery);
        map["GL_CHROMIUM_copy_texture"] = esOnlyExtension(&Extensions::copyTexture);
        map["GL_CHROMIUM_copy_compressed_texture"] = esOnlyExtension(&Extensions::copyCompressedTexture);
        map["GL_ANGLE_copy_texture_3d"] = enableableExtension(&Extensions::copyTexture3d);
        map["GL_ANGLE_webgl_compatibility"] = esOnlyExtension(&Extensions::webglCompatibility);
        map["GL_ANGLE_request_extension"] = esOnlyExtension(&Extensions::requestExtension);
        map["GL_CHROMIUM_bind_generates_resource"] = esOnlyExtension(&Extensions::bindGeneratesResource);
        map["GL_ANGLE_robust_client_memory"] = esOnlyExtension(&Extensions::robustClientMemory);
        map["GL_EXT_texture_sRGB_decode"] = esOnlyExtension(&Extensions::textureSRGBDecode);
        map["GL_EXT_sRGB_write_control"] = esOnlyExtension(&Extensions::sRGBWriteControl);
        map["GL_CHROMIUM_color_buffer_float_rgb"] = enableableExtension(&Extensions::colorBufferFloatRGB);
        map["GL_CHROMIUM_color_buffer_float_rgba"] = enableableExtension(&Extensions::colorBufferFloatRGBA);
        map["GL_EXT_multisample_compatibility"] = esOnlyExtension(&Extensions::multisampleCompatibility);
        map["GL_CHROMIUM_framebuffer_mixed_samples"] = esOnlyExtension(&Extensions::framebufferMixedSamples);
        map["GL_EXT_texture_norm16"] = esOnlyExtension(&Extensions::textureNorm16);
        map["GL_CHROMIUM_path_rendering"] = esOnlyExtension(&Extensions::pathRendering);
        map["GL_OES_surfaceless_context"] = esOnlyExtension(&Extensions::surfacelessContext);
        map["GL_ANGLE_client_arrays"] = esOnlyExtension(&Extensions::clientArrays);
        map["GL_ANGLE_robust_resource_initialization"] = esOnlyExtension(&Extensions::robustResourceInitialization);
        map["GL_ANGLE_program_cache_control"] = esOnlyExtension(&Extensions::programCacheControl);
        map["GL_ANGLE_texture_rectangle"] = enableableExtension(&Extensions::textureRectangle);
        map["GL_EXT_geometry_shader"] = enableableExtension(&Extensions::geometryShader);
        map["GL_ANGLE_explicit_context_gles1"] = enableableExtension(&Extensions::explicitContextGles1);
        map["GL_ANGLE_explicit_context"] = enableableExtension(&Extensions::explicitContext);
        map["GL_KHR_parallel_shader_compile"] = enableableExtension(&Extensions::parallelShaderCompile);
        map["GL_OES_texture_storage_multisample_2d_array"] = enableableExtension(&Extensions::textureStorageMultisample2DArray);
        map["GL_ANGLE_multiview_multisample"] = enableableExtension(&Extensions::multiviewMultisample);
        map["GL_EXT_blend_func_extended"] = enableableExtension(&Extensions::blendFuncExtended);
        map["GL_EXT_float_blend"] = enableableExtension(&Extensions::floatBlend);
        map["GL_ANGLE_texture_multisample"] = enableableExtension(&Extensions::textureMultisample);
        map["GL_ANGLE_multi_draw"] = enableableExtension(&Extensions::multiDraw);
        map["GL_ANGLE_provoking_vertex"] = enableableExtension(&Extensions::provokingVertex);
        map["GL_CHROMIUM_lose_context"] = enableableExtension(&Extensions::loseContextCHROMIUM);
        map["GL_ANGLE_texture_external_update"] = enableableExtension(&Extensions::textureExternalUpdateANGLE);
        map["GL_ANGLE_base_vertex_base_instance"] = enableableExtension(&Extensions::baseVertexBaseInstance);
        map["GL_APPLE_clip_distance"] = enableableExtension(&Extensions::clipDistanceAPPLE);
        // GLES1 extensinos
        map["GL_OES_point_size_array"] = enableableExtension(&Extensions::pointSizeArray);
        map["GL_OES_texture_cube_map"] = enableableExtension(&Extensions::textureCubeMap);
        map["GL_OES_point_sprite"] = enableableExtension(&Extensions::pointSprite);
        map["GL_OES_draw_texture"] = enableableExtension(&Extensions::drawTexture);
        map["GL_ANGLE_memory_size"] = enableableExtension(&Extensions::memorySize);
        // clang-format on

#if defined(ANGLE_ENABLE_ASSERTS)
        // Verify all extension strings start with GL_
        for (const auto &extension : map)
        {
            ASSERT(extension.first.rfind("GL_", 0) == 0);
        }
#endif

        return map;
    };

    static const angle::base::NoDestructor<ExtensionInfoMap> extensionInfo(buildExtensionInfoMap());
    return *extensionInfo;
}

TypePrecision::TypePrecision() = default;

TypePrecision::TypePrecision(const TypePrecision &other) = default;

void TypePrecision::setIEEEFloat()
{
    range     = {{127, 127}};
    precision = 23;
}

void TypePrecision::setTwosComplementInt(unsigned int bits)
{
    range     = {{static_cast<GLint>(bits) - 1, static_cast<GLint>(bits) - 2}};
    precision = 0;
}

void TypePrecision::setSimulatedFloat(unsigned int r, unsigned int p)
{
    range     = {{static_cast<GLint>(r), static_cast<GLint>(r)}};
    precision = static_cast<GLint>(p);
}

void TypePrecision::setSimulatedInt(unsigned int r)
{
    range     = {{static_cast<GLint>(r), static_cast<GLint>(r)}};
    precision = 0;
}

void TypePrecision::get(GLint *returnRange, GLint *returnPrecision) const
{
    std::copy(range.begin(), range.end(), returnRange);
    *returnPrecision = precision;
}

Caps::Caps()                  = default;
Caps::Caps(const Caps &other) = default;
Caps::~Caps()                 = default;

Caps GenerateMinimumCaps(const Version &clientVersion, const Extensions &extensions)
{
    Caps caps;

    // GLES1 emulation (Minimums taken from Table 6.20 / 6.22 (ES 1.1 spec))
    if (clientVersion < Version(2, 0))
    {
        caps.maxMultitextureUnits = 2;
        caps.maxLights            = 8;
        caps.maxClipPlanes        = 1;

        caps.maxModelviewMatrixStackDepth  = 16;
        caps.maxProjectionMatrixStackDepth = 2;
        caps.maxTextureMatrixStackDepth    = 2;

        caps.minSmoothPointSize = 1.0f;
        caps.maxSmoothPointSize = 1.0f;
    }

    if (clientVersion >= Version(2, 0))
    {
        // Table 6.18
        caps.max2DTextureSize      = 64;
        caps.maxCubeMapTextureSize = 16;
        caps.maxViewportWidth      = caps.max2DTextureSize;
        caps.maxViewportHeight     = caps.max2DTextureSize;
        caps.minAliasedPointSize   = 1;
        caps.maxAliasedPointSize   = 1;
        caps.minAliasedLineWidth   = 1;
        caps.maxAliasedLineWidth   = 1;

        // Table 6.19
        caps.vertexHighpFloat.setSimulatedFloat(62, 16);
        caps.vertexMediumpFloat.setSimulatedFloat(14, 10);
        caps.vertexLowpFloat.setSimulatedFloat(1, 8);
        caps.vertexHighpInt.setSimulatedInt(16);
        caps.vertexMediumpInt.setSimulatedInt(10);
        caps.vertexLowpInt.setSimulatedInt(8);
        caps.fragmentHighpFloat.setSimulatedFloat(62, 16);
        caps.fragmentMediumpFloat.setSimulatedFloat(14, 10);
        caps.fragmentLowpFloat.setSimulatedFloat(1, 8);
        caps.fragmentHighpInt.setSimulatedInt(16);
        caps.fragmentMediumpInt.setSimulatedInt(10);
        caps.fragmentLowpInt.setSimulatedInt(8);

        // Table 6.20
        caps.maxVertexAttributes                              = 8;
        caps.maxVertexUniformVectors                          = 128;
        caps.maxVaryingVectors                                = 8;
        caps.maxCombinedTextureImageUnits                     = 8;
        caps.maxShaderTextureImageUnits[ShaderType::Fragment] = 8;
        caps.maxFragmentUniformVectors                        = 16;
        caps.maxRenderbufferSize                              = 1;

        // Table 3.35
        caps.maxSamples = 4;
    }

    if (clientVersion >= Version(3, 0))
    {
        // Table 6.28
        caps.maxElementIndex       = (1 << 24) - 1;
        caps.max3DTextureSize      = 256;
        caps.max2DTextureSize      = 2048;
        caps.maxArrayTextureLayers = 256;
        caps.maxLODBias            = 2.0f;
        caps.maxCubeMapTextureSize = 2048;
        caps.maxRenderbufferSize   = 2048;
        caps.maxDrawBuffers        = 4;
        caps.maxColorAttachments   = 4;
        caps.maxViewportWidth      = caps.max2DTextureSize;
        caps.maxViewportHeight     = caps.max2DTextureSize;

        // Table 6.29
        caps.compressedTextureFormats.push_back(GL_COMPRESSED_R11_EAC);
        caps.compressedTextureFormats.push_back(GL_COMPRESSED_SIGNED_R11_EAC);
        caps.compressedTextureFormats.push_back(GL_COMPRESSED_RG11_EAC);
        caps.compressedTextureFormats.push_back(GL_COMPRESSED_SIGNED_RG11_EAC);
        caps.compressedTextureFormats.push_back(GL_COMPRESSED_RGB8_ETC2);
        caps.compressedTextureFormats.push_back(GL_COMPRESSED_SRGB8_ETC2);
        caps.compressedTextureFormats.push_back(GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2);
        caps.compressedTextureFormats.push_back(GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2);
        caps.compressedTextureFormats.push_back(GL_COMPRESSED_RGBA8_ETC2_EAC);
        caps.compressedTextureFormats.push_back(GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC);
        caps.vertexHighpFloat.setIEEEFloat();
        caps.vertexHighpInt.setTwosComplementInt(32);
        caps.vertexMediumpInt.setTwosComplementInt(16);
        caps.vertexLowpInt.setTwosComplementInt(8);
        caps.fragmentHighpFloat.setIEEEFloat();
        caps.fragmentHighpInt.setSimulatedInt(32);
        caps.fragmentMediumpInt.setTwosComplementInt(16);
        caps.fragmentLowpInt.setTwosComplementInt(8);
        caps.maxServerWaitTimeout = 0;

        // Table 6.31
        caps.maxVertexAttributes                            = 16;
        caps.maxShaderUniformComponents[ShaderType::Vertex] = 1024;
        caps.maxVertexUniformVectors                        = 256;
        caps.maxShaderUniformBlocks[ShaderType::Vertex]     = 12;
        caps.maxVertexOutputComponents                      = 64;
        caps.maxShaderTextureImageUnits[ShaderType::Vertex] = 16;

        // Table 6.32
        caps.maxShaderUniformComponents[ShaderType::Fragment] = 896;
        caps.maxFragmentUniformVectors                        = 224;
        caps.maxShaderUniformBlocks[ShaderType::Fragment]     = 12;
        caps.maxFragmentInputComponents                       = 60;
        caps.maxShaderTextureImageUnits[ShaderType::Fragment] = 16;
        caps.minProgramTexelOffset                            = -8;
        caps.maxProgramTexelOffset                            = 7;

        // Table 6.33
        caps.maxUniformBufferBindings     = 24;
        caps.maxUniformBlockSize          = 16384;
        caps.uniformBufferOffsetAlignment = 256;
        caps.maxCombinedUniformBlocks     = 24;
        caps.maxVaryingComponents         = 60;
        caps.maxVaryingVectors            = 15;
        caps.maxCombinedTextureImageUnits = 32;

        // Table 6.34
        caps.maxTransformFeedbackInterleavedComponents = 64;
        caps.maxTransformFeedbackSeparateAttributes    = 4;
        caps.maxTransformFeedbackSeparateComponents    = 4;
    }

    if (clientVersion >= Version(3, 1))
    {
        // Table 20.40
        caps.maxFramebufferWidth    = 2048;
        caps.maxFramebufferHeight   = 2048;
        caps.maxFramebufferSamples  = 4;
        caps.maxSampleMaskWords     = 1;
        caps.maxColorTextureSamples = 1;
        caps.maxDepthTextureSamples = 1;
        caps.maxIntegerSamples      = 1;

        // Table 20.41
        caps.maxVertexAttribRelativeOffset = 2047;
        caps.maxVertexAttribBindings       = 16;
        caps.maxVertexAttribStride         = 2048;

        // Table 20.43
        caps.maxShaderAtomicCounterBuffers[ShaderType::Vertex] = 0;
        caps.maxShaderAtomicCounters[ShaderType::Vertex]       = 0;
        caps.maxShaderImageUniforms[ShaderType::Vertex]        = 0;
        caps.maxShaderStorageBlocks[ShaderType::Vertex]        = 0;

        // Table 20.44
        caps.maxShaderUniformComponents[ShaderType::Fragment]    = 1024;
        caps.maxFragmentUniformVectors                           = 256;
        caps.maxShaderAtomicCounterBuffers[ShaderType::Fragment] = 0;
        caps.maxShaderAtomicCounters[ShaderType::Fragment]       = 0;
        caps.maxShaderImageUniforms[ShaderType::Fragment]        = 0;
        caps.maxShaderStorageBlocks[ShaderType::Fragment]        = 0;
        caps.minProgramTextureGatherOffset                       = 0;
        caps.maxProgramTextureGatherOffset                       = 0;

        // Table 20.45
        caps.maxComputeWorkGroupCount                           = {{65535, 65535, 65535}};
        caps.maxComputeWorkGroupSize                            = {{128, 128, 64}};
        caps.maxComputeWorkGroupInvocations                     = 12;
        caps.maxShaderUniformBlocks[ShaderType::Compute]        = 12;
        caps.maxShaderTextureImageUnits[ShaderType::Compute]    = 16;
        caps.maxComputeSharedMemorySize                         = 16384;
        caps.maxShaderUniformComponents[ShaderType::Compute]    = 1024;
        caps.maxShaderAtomicCounterBuffers[ShaderType::Compute] = 1;
        caps.maxShaderAtomicCounters[ShaderType::Compute]       = 8;
        caps.maxShaderImageUniforms[ShaderType::Compute]        = 4;
        caps.maxShaderStorageBlocks[ShaderType::Compute]        = 4;

        // Table 20.46
        caps.maxUniformBufferBindings         = 36;
        caps.maxCombinedTextureImageUnits     = 48;
        caps.maxCombinedShaderOutputResources = 4;

        // Table 20.47
        caps.maxUniformLocations                = 1024;
        caps.maxAtomicCounterBufferBindings     = 1;
        caps.maxAtomicCounterBufferSize         = 32;
        caps.maxCombinedAtomicCounterBuffers    = 1;
        caps.maxCombinedAtomicCounters          = 8;
        caps.maxImageUnits                      = 4;
        caps.maxCombinedImageUniforms           = 4;
        caps.maxShaderStorageBufferBindings     = 4;
        caps.maxShaderStorageBlockSize          = 1 << 27;
        caps.maxCombinedShaderStorageBlocks     = 4;
        caps.shaderStorageBufferOffsetAlignment = 256;
    }

    if (extensions.textureRectangle)
    {
        caps.maxRectangleTextureSize = 64;
    }

    if (extensions.geometryShader)
    {
        // Table 20.40 (GL_EXT_geometry_shader)
        caps.maxFramebufferLayers = 256;
        caps.layerProvokingVertex = GL_LAST_VERTEX_CONVENTION_EXT;

        // Table 20.43gs (GL_EXT_geometry_shader)
        caps.maxShaderUniformComponents[ShaderType::Geometry]    = 1024;
        caps.maxShaderUniformBlocks[ShaderType::Geometry]        = 12;
        caps.maxGeometryInputComponents                          = 64;
        caps.maxGeometryOutputComponents                         = 64;
        caps.maxGeometryOutputVertices                           = 256;
        caps.maxGeometryTotalOutputComponents                    = 1024;
        caps.maxShaderTextureImageUnits[ShaderType::Geometry]    = 16;
        caps.maxShaderAtomicCounterBuffers[ShaderType::Geometry] = 0;
        caps.maxShaderAtomicCounters[ShaderType::Geometry]       = 0;
        caps.maxShaderStorageBlocks[ShaderType::Geometry]        = 0;
        caps.maxGeometryShaderInvocations                        = 32;

        // Table 20.46 (GL_EXT_geometry_shader)
        caps.maxShaderImageUniforms[ShaderType::Geometry] = 0;

        // Table 20.46 (GL_EXT_geometry_shader)
        caps.maxUniformBufferBindings     = 48;
        caps.maxCombinedUniformBlocks     = 36;
        caps.maxCombinedTextureImageUnits = 64;
    }

    for (ShaderType shaderType : AllShaderTypes())
    {
        caps.maxCombinedShaderUniformComponents[shaderType] =
            caps.maxShaderUniformBlocks[shaderType] *
                static_cast<GLuint>(caps.maxUniformBlockSize / 4) +
            caps.maxShaderUniformComponents[shaderType];
    }

    return caps;
}
}  // namespace gl

namespace egl
{

Caps::Caps() = default;

DisplayExtensions::DisplayExtensions() = default;

std::vector<std::string> DisplayExtensions::getStrings() const
{
    std::vector<std::string> extensionStrings;

    // clang-format off
    //                   | Extension name                                       | Supported flag                    | Output vector   |
    InsertExtensionString("EGL_EXT_create_context_robustness",                   createContextRobustness,            &extensionStrings);
    InsertExtensionString("EGL_ANGLE_d3d_share_handle_client_buffer",            d3dShareHandleClientBuffer,         &extensionStrings);
    InsertExtensionString("EGL_ANGLE_d3d_texture_client_buffer",                 d3dTextureClientBuffer,             &extensionStrings);
    InsertExtensionString("EGL_ANGLE_surface_d3d_texture_2d_share_handle",       surfaceD3DTexture2DShareHandle,     &extensionStrings);
    InsertExtensionString("EGL_ANGLE_query_surface_pointer",                     querySurfacePointer,                &extensionStrings);
    InsertExtensionString("EGL_ANGLE_window_fixed_size",                         windowFixedSize,                    &extensionStrings);
    InsertExtensionString("EGL_ANGLE_keyed_mutex",                               keyedMutex,                         &extensionStrings);
    InsertExtensionString("EGL_ANGLE_surface_orientation",                       surfaceOrientation,                 &extensionStrings);
    InsertExtensionString("EGL_ANGLE_direct_composition",                        directComposition,                  &extensionStrings);
    InsertExtensionString("EGL_ANGLE_windows_ui_composition",                    windowsUIComposition,               &extensionStrings);
    InsertExtensionString("EGL_NV_post_sub_buffer",                              postSubBuffer,                      &extensionStrings);
    InsertExtensionString("EGL_KHR_create_context",                              createContext,                      &extensionStrings);
    InsertExtensionString("EGL_EXT_device_query",                                deviceQuery,                        &extensionStrings);
    InsertExtensionString("EGL_KHR_image",                                       image,                              &extensionStrings);
    InsertExtensionString("EGL_KHR_image_base",                                  imageBase,                          &extensionStrings);
    InsertExtensionString("EGL_KHR_image_pixmap",                                imagePixmap,                        &extensionStrings);
    InsertExtensionString("EGL_KHR_gl_colorspace",                               glColorspace,                       &extensionStrings);
    InsertExtensionString("EGL_EXT_gl_colorspace_scrgb",                         glColorspaceScrgb,                  &extensionStrings);
    InsertExtensionString("EGL_EXT_gl_colorspace_scrgb_linear",                  glColorspaceScrgbLinear,            &extensionStrings);
    InsertExtensionString("EGL_EXT_gl_colorspace_display_p3",                    glColorspaceDisplayP3,              &extensionStrings);
    InsertExtensionString("EGL_EXT_gl_colorspace_display_p3_linear",             glColorspaceDisplayP3Linear,        &extensionStrings);
    InsertExtensionString("EGL_EXT_gl_colorspace_display_p3_passthrough",        glColorspaceDisplayP3Passthrough,   &extensionStrings);
    InsertExtensionString("EGL_KHR_gl_texture_2D_image",                         glTexture2DImage,                   &extensionStrings);
    InsertExtensionString("EGL_KHR_gl_texture_cubemap_image",                    glTextureCubemapImage,              &extensionStrings);
    InsertExtensionString("EGL_KHR_gl_texture_3D_image",                         glTexture3DImage,                   &extensionStrings);
    InsertExtensionString("EGL_KHR_gl_renderbuffer_image",                       glRenderbufferImage,                &extensionStrings);
    InsertExtensionString("EGL_KHR_get_all_proc_addresses",                      getAllProcAddresses,                &extensionStrings);
    InsertExtensionString("EGL_KHR_stream",                                      stream,                             &extensionStrings);
    InsertExtensionString("EGL_KHR_stream_consumer_gltexture",                   streamConsumerGLTexture,            &extensionStrings);
    InsertExtensionString("EGL_NV_stream_consumer_gltexture_yuv",                streamConsumerGLTextureYUV,         &extensionStrings);
    InsertExtensionString("EGL_KHR_fence_sync",                                  fenceSync,                          &extensionStrings);
    InsertExtensionString("EGL_KHR_wait_sync",                                   waitSync,                           &extensionStrings);
    InsertExtensionString("EGL_ANGLE_flexible_surface_compatibility",            flexibleSurfaceCompatibility,       &extensionStrings);
    InsertExtensionString("EGL_ANGLE_stream_producer_d3d_texture",               streamProducerD3DTexture,           &extensionStrings);
    InsertExtensionString("EGL_ANGLE_create_context_webgl_compatibility",        createContextWebGLCompatibility,    &extensionStrings);
    InsertExtensionString("EGL_CHROMIUM_create_context_bind_generates_resource", createContextBindGeneratesResource, &extensionStrings);
    InsertExtensionString("EGL_CHROMIUM_sync_control",                           getSyncValues,                      &extensionStrings);
    InsertExtensionString("EGL_KHR_swap_buffers_with_damage",                    swapBuffersWithDamage,              &extensionStrings);
    InsertExtensionString("EGL_EXT_pixel_format_float",                          pixelFormatFloat,                   &extensionStrings);
    InsertExtensionString("EGL_KHR_surfaceless_context",                         surfacelessContext,                 &extensionStrings);
    InsertExtensionString("EGL_ANGLE_display_texture_share_group",               displayTextureShareGroup,           &extensionStrings);
    InsertExtensionString("EGL_ANGLE_create_context_client_arrays",              createContextClientArrays,          &extensionStrings);
    InsertExtensionString("EGL_ANGLE_program_cache_control",                     programCacheControl,                &extensionStrings);
    InsertExtensionString("EGL_ANGLE_robust_resource_initialization",            robustResourceInitialization,       &extensionStrings);
    InsertExtensionString("EGL_ANGLE_iosurface_client_buffer",                   iosurfaceClientBuffer,              &extensionStrings);
    InsertExtensionString("EGL_MGL_mtl_texture_client_buffer",                   mtlTextureClientBuffer,             &extensionStrings);
    InsertExtensionString("EGL_ANGLE_create_context_extensions_enabled",         createContextExtensionsEnabled,     &extensionStrings);
    InsertExtensionString("EGL_ANDROID_presentation_time",                       presentationTime,                   &extensionStrings);
    InsertExtensionString("EGL_ANDROID_blob_cache",                              blobCache,                          &extensionStrings);
    InsertExtensionString("EGL_ANDROID_image_native_buffer",                     imageNativeBuffer,                  &extensionStrings);
    InsertExtensionString("EGL_ANDROID_get_frame_timestamps",                    getFrameTimestamps,                 &extensionStrings);
    InsertExtensionString("EGL_ANDROID_recordable",                              recordable,                         &extensionStrings);
    InsertExtensionString("EGL_ANGLE_power_preference",                          powerPreference,                    &extensionStrings);
    InsertExtensionString("EGL_ANGLE_image_d3d11_texture",                       imageD3D11Texture,                  &extensionStrings);
    InsertExtensionString("EGL_ANDROID_get_native_client_buffer",                getNativeClientBufferANDROID,       &extensionStrings);
    InsertExtensionString("EGL_ANDROID_native_fence_sync",                       nativeFenceSyncANDROID,             &extensionStrings);
    InsertExtensionString("EGL_ANGLE_create_context_backwards_compatible",       createContextBackwardsCompatible,   &extensionStrings);
    InsertExtensionString("EGL_KHR_no_config_context",                           noConfigContext,                    &extensionStrings);
    // TODO(jmadill): Enable this when complete.
    //InsertExtensionString("KHR_create_context_no_error",                       createContextNoError,               &extensionStrings);
    // clang-format on

    return extensionStrings;
}

DeviceExtensions::DeviceExtensions() = default;

std::vector<std::string> DeviceExtensions::getStrings() const
{
    std::vector<std::string> extensionStrings;

    // clang-format off
    //                   | Extension name                                 | Supported flag                | Output vector   |
    InsertExtensionString("EGL_ANGLE_device_d3d",                          deviceD3D,                      &extensionStrings);
    InsertExtensionString("EGL_ANGLE_device_cgl",                          deviceCGL,                      &extensionStrings);
    InsertExtensionString("EGL_ANGLE_device_mtl",                          deviceMTL,                      &extensionStrings);
    // clang-format on

    return extensionStrings;
}

ClientExtensions::ClientExtensions()                              = default;
ClientExtensions::ClientExtensions(const ClientExtensions &other) = default;

std::vector<std::string> ClientExtensions::getStrings() const
{
    std::vector<std::string> extensionStrings;

    // clang-format off
    //                   | Extension name                                    | Supported flag                   | Output vector   |
    InsertExtensionString("EGL_EXT_client_extensions",                        clientExtensions,                   &extensionStrings);
    InsertExtensionString("EGL_EXT_platform_base",                            platformBase,                       &extensionStrings);
    InsertExtensionString("EGL_EXT_platform_device",                          platformDevice,                     &extensionStrings);
    InsertExtensionString("EGL_ANGLE_platform_angle",                         platformANGLE,                      &extensionStrings);
    InsertExtensionString("EGL_ANGLE_platform_angle_d3d",                     platformANGLED3D,                   &extensionStrings);
    InsertExtensionString("EGL_ANGLE_platform_angle_device_type_swiftshader", platformANGLEDeviceTypeSwiftShader, &extensionStrings);
    InsertExtensionString("EGL_ANGLE_platform_angle_opengl",                  platformANGLEOpenGL,                &extensionStrings);
    InsertExtensionString("EGL_ANGLE_platform_angle_null",                    platformANGLENULL,                  &extensionStrings);
    InsertExtensionString("EGL_ANGLE_platform_angle_vulkan",                  platformANGLEVulkan,                &extensionStrings);
    InsertExtensionString("EGL_ANGLE_platform_angle_metal",                   platformANGLEMetal,                 &extensionStrings);
    InsertExtensionString("EGL_ANGLE_platform_angle_context_virtualization",  platformANGLEContextVirtualization, &extensionStrings);
    InsertExtensionString("EGL_ANGLE_device_creation",                        deviceCreation,                     &extensionStrings);
    InsertExtensionString("EGL_ANGLE_device_creation_d3d11",                  deviceCreationD3D11,                &extensionStrings);
    InsertExtensionString("EGL_ANGLE_x11_visual",                             x11Visual,                          &extensionStrings);
    InsertExtensionString("EGL_ANGLE_experimental_present_path",              experimentalPresentPath,            &extensionStrings);
    InsertExtensionString("EGL_KHR_client_get_all_proc_addresses",            clientGetAllProcAddresses,          &extensionStrings);
    InsertExtensionString("EGL_KHR_debug",                                    debug,                              &extensionStrings);
    InsertExtensionString("EGL_ANGLE_explicit_context",                       explicitContext,                    &extensionStrings);
    InsertExtensionString("EGL_ANGLE_feature_control",                        featureControlANGLE,                &extensionStrings);
    // clang-format on

    return extensionStrings;
}

}  // namespace egl
