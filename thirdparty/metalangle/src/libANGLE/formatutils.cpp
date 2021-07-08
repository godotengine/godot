//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// formatutils.cpp: Queries for GL image formats.

#include "libANGLE/formatutils.h"

#include "anglebase/no_destructor.h"
#include "common/mathutil.h"
#include "libANGLE/Context.h"
#include "libANGLE/Framebuffer.h"

using namespace angle;

namespace gl
{

// ES2 requires that format is equal to internal format at all glTex*Image2D entry points and the
// implementation can decide the true, sized, internal format. The ES2FormatMap determines the
// internal format for all valid format and type combinations.
GLenum GetSizedFormatInternal(GLenum format, GLenum type);

namespace
{
using InternalFormatInfoMap =
    std::unordered_map<GLenum, std::unordered_map<GLenum, InternalFormat>>;

bool CheckedMathResult(const CheckedNumeric<GLuint> &value, GLuint *resultOut)
{
    if (!value.IsValid())
    {
        return false;
    }
    else
    {
        *resultOut = value.ValueOrDie();
        return true;
    }
}

constexpr uint32_t PackTypeInfo(GLuint bytes, bool specialized)
{
    // static_assert within constexpr requires c++17
    // static_assert(isPow2(bytes));
    return bytes | (rx::Log2(bytes) << 8) | (specialized << 16);
}

}  // anonymous namespace

FormatType::FormatType() : format(GL_NONE), type(GL_NONE) {}

FormatType::FormatType(GLenum format_, GLenum type_) : format(format_), type(type_) {}

bool FormatType::operator<(const FormatType &other) const
{
    if (format != other.format)
        return format < other.format;
    return type < other.type;
}

bool operator<(const Type &a, const Type &b)
{
    return memcmp(&a, &b, sizeof(Type)) < 0;
}

// Information about internal formats
static bool AlwaysSupported(const Version &, const Extensions &)
{
    return true;
}

static bool NeverSupported(const Version &, const Extensions &)
{
    return false;
}

template <GLuint minCoreGLMajorVersion, GLuint minCoreGLMinorVersion>
static bool RequireES(const Version &clientVersion, const Extensions &)
{
    return clientVersion >= Version(minCoreGLMajorVersion, minCoreGLMinorVersion);
}

// Pointer to a boolean memeber of the Extensions struct
typedef bool(Extensions::*ExtensionBool);

// Check support for a single extension
template <ExtensionBool bool1>
static bool RequireExt(const Version &, const Extensions &extensions)
{
    return extensions.*bool1;
}

// Check for a minimum client version or a single extension
template <GLuint minCoreGLMajorVersion, GLuint minCoreGLMinorVersion, ExtensionBool bool1>
static bool RequireESOrExt(const Version &clientVersion, const Extensions &extensions)
{
    return clientVersion >= Version(minCoreGLMajorVersion, minCoreGLMinorVersion) ||
           extensions.*bool1;
}

// Check for a minimum client version or two extensions
template <GLuint minCoreGLMajorVersion,
          GLuint minCoreGLMinorVersion,
          ExtensionBool bool1,
          ExtensionBool bool2>
static bool RequireESOrExtAndExt(const Version &clientVersion, const Extensions &extensions)
{
    return clientVersion >= Version(minCoreGLMajorVersion, minCoreGLMinorVersion) ||
           (extensions.*bool1 && extensions.*bool2);
}

// Check for a minimum client version or at least one of two extensions
template <GLuint minCoreGLMajorVersion,
          GLuint minCoreGLMinorVersion,
          ExtensionBool bool1,
          ExtensionBool bool2>
static bool RequireESOrExtOrExt(const Version &clientVersion, const Extensions &extensions)
{
    return clientVersion >= Version(minCoreGLMajorVersion, minCoreGLMinorVersion) ||
           extensions.*bool1 || extensions.*bool2;
}

// Check support for two extensions
template <ExtensionBool bool1, ExtensionBool bool2>
static bool RequireExtAndExt(const Version &, const Extensions &extensions)
{
    return extensions.*bool1 && extensions.*bool2;
}

// Check support for either of two extensions
template <ExtensionBool bool1, ExtensionBool bool2>
static bool RequireExtOrExt(const Version &, const Extensions &extensions)
{
    return extensions.*bool1 || extensions.*bool2;
}

// Check support for any of three extensions
template <ExtensionBool bool1, ExtensionBool bool2, ExtensionBool bool3>
static bool RequireExtOrExtOrExt(const Version &, const Extensions &extensions)
{
    return extensions.*bool1 || extensions.*bool2 || extensions.*bool3;
}

static bool UnsizedHalfFloatOESRGBATextureAttachmentSupport(const Version &clientVersion,
                                                            const Extensions &extensions)
{
    // dEQP requires ES3 + EXT_color_buffer_half_float for rendering to RGB[A] + HALF_FLOAT_OES
    // textures but WebGL allows it with just ES 2.0
    return (clientVersion.major >= 3 || extensions.webglCompatibility) &&
           extensions.colorBufferHalfFloat;
}

// R8, RG8
static bool SizedRGSupport(const Version &clientVersion, const Extensions &extensions)
{
    return clientVersion >= Version(3, 0) || (extensions.textureStorage && extensions.textureRG);
}

// R16F, RG16F with HALF_FLOAT_OES type
static bool SizedHalfFloatOESRGSupport(const Version &clientVersion, const Extensions &extensions)
{
    return extensions.textureStorage && extensions.textureHalfFloat && extensions.textureRG;
}

static bool SizedHalfFloatOESRGTextureAttachmentSupport(const Version &clientVersion,
                                                        const Extensions &extensions)
{
    return SizedHalfFloatOESRGSupport(clientVersion, extensions) && extensions.colorBufferHalfFloat;
}

// R16F, RG16F with either HALF_FLOAT_OES or HALF_FLOAT types
static bool SizedHalfFloatRGSupport(const Version &clientVersion, const Extensions &extensions)
{
    // HALF_FLOAT
    if (clientVersion >= Version(3, 0))
    {
        return true;
    }
    // HALF_FLOAT_OES
    else
    {
        return SizedHalfFloatOESRGSupport(clientVersion, extensions);
    }
}

static bool SizedHalfFloatRGTextureAttachmentSupport(const Version &clientVersion,
                                                     const Extensions &extensions)
{
    // HALF_FLOAT
    if (clientVersion >= Version(3, 0))
    {
        return extensions.colorBufferFloat;
    }
    // HALF_FLOAT_OES
    else
    {
        return SizedHalfFloatOESRGTextureAttachmentSupport(clientVersion, extensions);
    }
}

static bool SizedHalfFloatRGRenderbufferSupport(const Version &clientVersion,
                                                const Extensions &extensions)
{
    return (clientVersion >= Version(3, 0) ||
            (extensions.textureHalfFloat && extensions.textureRG)) &&
           (extensions.colorBufferFloat || extensions.colorBufferHalfFloat);
}

// RGB16F, RGBA16F with HALF_FLOAT_OES type
static bool SizedHalfFloatOESSupport(const Version &clientVersion, const Extensions &extensions)
{
    return extensions.textureStorage && extensions.textureHalfFloat;
}

static bool SizedHalfFloatOESTextureAttachmentSupport(const Version &clientVersion,
                                                      const Extensions &extensions)
{
    return SizedHalfFloatOESSupport(clientVersion, extensions) && extensions.colorBufferHalfFloat;
}

// RGB16F, RGBA16F with either HALF_FLOAT_OES or HALF_FLOAT types
static bool SizedHalfFloatSupport(const Version &clientVersion, const Extensions &extensions)
{
    // HALF_FLOAT
    if (clientVersion >= Version(3, 0))
    {
        return true;
    }
    // HALF_FLOAT_OES
    else
    {
        return SizedHalfFloatOESSupport(clientVersion, extensions);
    }
}

static bool SizedHalfFloatFilterSupport(const Version &clientVersion, const Extensions &extensions)
{
    // HALF_FLOAT
    if (clientVersion >= Version(3, 0))
    {
        return true;
    }
    // HALF_FLOAT_OES
    else
    {
        return extensions.textureHalfFloatLinear;
    }
}

static bool SizedHalfFloatRGBTextureAttachmentSupport(const Version &clientVersion,
                                                      const Extensions &extensions)
{
    // HALF_FLOAT
    if (clientVersion >= Version(3, 0))
    {
        // It is unclear how EXT_color_buffer_half_float applies to ES3.0 and above, however,
        // dEQP GLES3 es3fFboColorbufferTests.cpp verifies that texture attachment of GL_RGB16F
        // is possible, so assume that all GLES implementations support it.
        return extensions.colorBufferHalfFloat;
    }
    // HALF_FLOAT_OES
    else
    {
        return SizedHalfFloatOESTextureAttachmentSupport(clientVersion, extensions);
    }
}

static bool SizedHalfFloatRGBRenderbufferSupport(const Version &clientVersion,
                                                 const Extensions &extensions)
{
    return (clientVersion >= Version(3, 0) || extensions.textureHalfFloat) &&
           extensions.colorBufferHalfFloat;
}

static bool SizedHalfFloatRGBATextureAttachmentSupport(const Version &clientVersion,
                                                       const Extensions &extensions)
{
    // HALF_FLOAT
    if (clientVersion >= Version(3, 0))
    {
        return extensions.colorBufferFloat;
    }
    // HALF_FLOAT_OES
    else
    {
        return SizedHalfFloatOESTextureAttachmentSupport(clientVersion, extensions);
    }
}

static bool SizedHalfFloatRGBARenderbufferSupport(const Version &clientVersion,
                                                  const Extensions &extensions)
{
    return (clientVersion >= Version(3, 0) || extensions.textureHalfFloat) &&
           (extensions.colorBufferFloat || extensions.colorBufferHalfFloat);
}

// R32F, RG32F
static bool SizedFloatRGSupport(const Version &clientVersion, const Extensions &extensions)
{
    return clientVersion >= Version(3, 0) ||
           (extensions.textureStorage && extensions.textureFloat && extensions.textureRG);
}

// RGB32F
static bool SizedFloatRGBSupport(const Version &clientVersion, const Extensions &extensions)
{
    return clientVersion >= Version(3, 0) ||
           (extensions.textureStorage && extensions.textureFloat) || extensions.colorBufferFloatRGB;
}

// RGBA32F
static bool SizedFloatRGBASupport(const Version &clientVersion, const Extensions &extensions)
{
    return clientVersion >= Version(3, 0) ||
           (extensions.textureStorage && extensions.textureFloat) ||
           extensions.colorBufferFloatRGBA;
}

static bool SizedFloatRGBARenderableSupport(const Version &clientVersion,
                                            const Extensions &extensions)
{
    // This logic is the same for both Renderbuffers and TextureAttachment.
    return extensions.colorBufferFloatRGBA ||  // ES2
           extensions.colorBufferFloat;        // ES3
}

InternalFormat::InternalFormat()
    : internalFormat(GL_NONE),
      sized(false),
      sizedInternalFormat(GL_NONE),
      redBits(0),
      greenBits(0),
      blueBits(0),
      luminanceBits(0),
      alphaBits(0),
      sharedBits(0),
      depthBits(0),
      stencilBits(0),
      pixelBytes(0),
      componentCount(0),
      compressed(false),
      compressedBlockWidth(0),
      compressedBlockHeight(0),
      compressedBlockDepth(0),
      format(GL_NONE),
      type(GL_NONE),
      componentType(GL_NONE),
      colorEncoding(GL_NONE),
      textureSupport(NeverSupported),
      filterSupport(NeverSupported),
      textureAttachmentSupport(NeverSupported),
      renderbufferSupport(NeverSupported)
{}

InternalFormat::InternalFormat(const InternalFormat &other) = default;

bool InternalFormat::isLUMA() const
{
    return ((redBits + greenBits + blueBits + depthBits + stencilBits) == 0 &&
            (luminanceBits + alphaBits) > 0);
}

GLenum InternalFormat::getReadPixelsFormat() const
{
    return format;
}

GLenum InternalFormat::getReadPixelsType(const Version &version) const
{
    switch (type)
    {
        case GL_HALF_FLOAT:
        case GL_HALF_FLOAT_OES:
            if (version < Version(3, 0))
            {
                // The internal format may have a type of GL_HALF_FLOAT but when exposing this type
                // as the IMPLEMENTATION_READ_TYPE, only HALF_FLOAT_OES is allowed by
                // OES_texture_half_float.  HALF_FLOAT becomes core in ES3 and is acceptable to use
                // as an IMPLEMENTATION_READ_TYPE.
                return GL_HALF_FLOAT_OES;
            }
            else
            {
                return GL_HALF_FLOAT;
            }

        default:
            return type;
    }
}

bool InternalFormat::supportSubImage() const
{
    return !CompressedFormatRequiresWholeImage(internalFormat);
}

bool InternalFormat::isRequiredRenderbufferFormat(const Version &version) const
{
    // GLES 3.0.5 section 4.4.2.2:
    // "Implementations are required to support the same internal formats for renderbuffers as the
    // required formats for textures enumerated in section 3.8.3.1, with the exception of the color
    // formats labelled "texture-only"."
    if (!sized || compressed)
    {
        return false;
    }

    // Luma formats.
    if (isLUMA())
    {
        return false;
    }

    // Depth/stencil formats.
    if (depthBits > 0 || stencilBits > 0)
    {
        // GLES 2.0.25 table 4.5.
        // GLES 3.0.5 section 3.8.3.1.
        // GLES 3.1 table 8.14.

        // Required formats in all versions.
        switch (internalFormat)
        {
            case GL_DEPTH_COMPONENT16:
            case GL_STENCIL_INDEX8:
                // Note that STENCIL_INDEX8 is not mentioned in GLES 3.0.5 section 3.8.3.1, but it
                // is in section 4.4.2.2.
                return true;
            default:
                break;
        }
        if (version.major < 3)
        {
            return false;
        }
        // Required formats in GLES 3.0 and up.
        switch (internalFormat)
        {
            case GL_DEPTH_COMPONENT32F:
            case GL_DEPTH_COMPONENT24:
            case GL_DEPTH32F_STENCIL8:
            case GL_DEPTH24_STENCIL8:
                return true;
            default:
                return false;
        }
    }

    // RGBA formats.
    // GLES 2.0.25 table 4.5.
    // GLES 3.0.5 section 3.8.3.1.
    // GLES 3.1 table 8.13.

    // Required formats in all versions.
    switch (internalFormat)
    {
        case GL_RGBA4:
        case GL_RGB5_A1:
        case GL_RGB565:
            return true;
        default:
            break;
    }
    if (version.major < 3)
    {
        return false;
    }

    if (format == GL_BGRA_EXT)
    {
        return false;
    }

    switch (componentType)
    {
        case GL_SIGNED_NORMALIZED:
        case GL_FLOAT:
            return false;
        case GL_UNSIGNED_INT:
        case GL_INT:
            // Integer RGB formats are not required renderbuffer formats.
            if (alphaBits == 0 && blueBits != 0)
            {
                return false;
            }
            // All integer R and RG formats are required.
            // Integer RGBA formats including RGB10_A2_UI are required.
            return true;
        case GL_UNSIGNED_NORMALIZED:
            if (internalFormat == GL_SRGB8)
            {
                return false;
            }
            return true;
        default:
            UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
            return false;
#endif
    }
}

bool InternalFormat::isInt() const
{
    return componentType == GL_INT || componentType == GL_UNSIGNED_INT;
}

bool InternalFormat::isDepthOrStencil() const
{
    return depthBits != 0 || stencilBits != 0;
}

Format::Format(GLenum internalFormat) : Format(GetSizedInternalFormatInfo(internalFormat)) {}

Format::Format(const InternalFormat &internalFormat) : info(&internalFormat) {}

Format::Format(GLenum internalFormat, GLenum type)
    : info(&GetInternalFormatInfo(internalFormat, type))
{}

Format::Format(const Format &other) = default;
Format &Format::operator=(const Format &other) = default;

bool Format::valid() const
{
    return info->internalFormat != GL_NONE;
}

// static
bool Format::SameSized(const Format &a, const Format &b)
{
    return a.info->sizedInternalFormat == b.info->sizedInternalFormat;
}

static GLenum EquivalentBlitInternalFormat(GLenum internalformat)
{
    // BlitFramebuffer works if the color channels are identically
    // sized, even if there is a swizzle (for example, blitting from a
    // multisampled RGBA8 renderbuffer to a BGRA8 texture). This could
    // be expanded and/or autogenerated if that is found necessary.
    if (internalformat == GL_BGRA8_EXT)
        return GL_RGBA8;
    return internalformat;
}

// static
bool Format::EquivalentForBlit(const Format &a, const Format &b)
{
    return (EquivalentBlitInternalFormat(a.info->sizedInternalFormat) ==
            EquivalentBlitInternalFormat(b.info->sizedInternalFormat));
}

// static
Format Format::Invalid()
{
    static Format invalid(GL_NONE, GL_NONE);
    return invalid;
}

std::ostream &operator<<(std::ostream &os, const Format &fmt)
{
    // TODO(ynovikov): return string representation when available
    return FmtHex(os, fmt.info->sizedInternalFormat);
}

bool InternalFormat::operator==(const InternalFormat &other) const
{
    // We assume all internal formats are unique if they have the same internal format and type
    return internalFormat == other.internalFormat && type == other.type;
}

bool InternalFormat::operator!=(const InternalFormat &other) const
{
    return !(*this == other);
}

void InsertFormatInfo(InternalFormatInfoMap *map, const InternalFormat &formatInfo)
{
    ASSERT(!formatInfo.sized || (*map).count(formatInfo.internalFormat) == 0);
    ASSERT((*map)[formatInfo.internalFormat].count(formatInfo.type) == 0);
    (*map)[formatInfo.internalFormat][formatInfo.type] = formatInfo;
}

void AddRGBAFormat(InternalFormatInfoMap *map,
                   GLenum internalFormat,
                   bool sized,
                   GLuint red,
                   GLuint green,
                   GLuint blue,
                   GLuint alpha,
                   GLuint shared,
                   GLenum format,
                   GLenum type,
                   GLenum componentType,
                   bool srgb,
                   InternalFormat::SupportCheckFunction textureSupport,
                   InternalFormat::SupportCheckFunction filterSupport,
                   InternalFormat::SupportCheckFunction textureAttachmentSupport,
                   InternalFormat::SupportCheckFunction renderbufferSupport)
{
    InternalFormat formatInfo;
    formatInfo.internalFormat = internalFormat;
    formatInfo.sized          = sized;
    formatInfo.sizedInternalFormat =
        sized ? internalFormat : GetSizedFormatInternal(internalFormat, type);
    formatInfo.redBits    = red;
    formatInfo.greenBits  = green;
    formatInfo.blueBits   = blue;
    formatInfo.alphaBits  = alpha;
    formatInfo.sharedBits = shared;
    formatInfo.pixelBytes = (red + green + blue + alpha + shared) / 8;
    formatInfo.componentCount =
        ((red > 0) ? 1 : 0) + ((green > 0) ? 1 : 0) + ((blue > 0) ? 1 : 0) + ((alpha > 0) ? 1 : 0);
    formatInfo.format                   = format;
    formatInfo.type                     = type;
    formatInfo.componentType            = componentType;
    formatInfo.colorEncoding            = (srgb ? GL_SRGB : GL_LINEAR);
    formatInfo.textureSupport           = textureSupport;
    formatInfo.filterSupport            = filterSupport;
    formatInfo.textureAttachmentSupport = textureAttachmentSupport;
    formatInfo.renderbufferSupport      = renderbufferSupport;

    InsertFormatInfo(map, formatInfo);
}

static void AddLUMAFormat(InternalFormatInfoMap *map,
                          GLenum internalFormat,
                          bool sized,
                          GLuint luminance,
                          GLuint alpha,
                          GLenum format,
                          GLenum type,
                          GLenum componentType,
                          InternalFormat::SupportCheckFunction textureSupport,
                          InternalFormat::SupportCheckFunction filterSupport,
                          InternalFormat::SupportCheckFunction textureAttachmentSupport,
                          InternalFormat::SupportCheckFunction renderbufferSupport)
{
    InternalFormat formatInfo;
    formatInfo.internalFormat = internalFormat;
    formatInfo.sized          = sized;
    formatInfo.sizedInternalFormat =
        sized ? internalFormat : GetSizedFormatInternal(internalFormat, type);
    formatInfo.luminanceBits            = luminance;
    formatInfo.alphaBits                = alpha;
    formatInfo.pixelBytes               = (luminance + alpha) / 8;
    formatInfo.componentCount           = ((luminance > 0) ? 1 : 0) + ((alpha > 0) ? 1 : 0);
    formatInfo.format                   = format;
    formatInfo.type                     = type;
    formatInfo.componentType            = componentType;
    formatInfo.colorEncoding            = GL_LINEAR;
    formatInfo.textureSupport           = textureSupport;
    formatInfo.filterSupport            = filterSupport;
    formatInfo.textureAttachmentSupport = textureAttachmentSupport;
    formatInfo.renderbufferSupport      = renderbufferSupport;

    InsertFormatInfo(map, formatInfo);
}

void AddDepthStencilFormat(InternalFormatInfoMap *map,
                           GLenum internalFormat,
                           bool sized,
                           GLuint depthBits,
                           GLuint stencilBits,
                           GLuint unusedBits,
                           GLenum format,
                           GLenum type,
                           GLenum componentType,
                           InternalFormat::SupportCheckFunction textureSupport,
                           InternalFormat::SupportCheckFunction filterSupport,
                           InternalFormat::SupportCheckFunction textureAttachmentSupport,
                           InternalFormat::SupportCheckFunction renderbufferSupport)
{
    InternalFormat formatInfo;
    formatInfo.internalFormat = internalFormat;
    formatInfo.sized          = sized;
    formatInfo.sizedInternalFormat =
        sized ? internalFormat : GetSizedFormatInternal(internalFormat, type);
    formatInfo.depthBits                = depthBits;
    formatInfo.stencilBits              = stencilBits;
    formatInfo.pixelBytes               = (depthBits + stencilBits + unusedBits) / 8;
    formatInfo.componentCount           = ((depthBits > 0) ? 1 : 0) + ((stencilBits > 0) ? 1 : 0);
    formatInfo.format                   = format;
    formatInfo.type                     = type;
    formatInfo.componentType            = componentType;
    formatInfo.colorEncoding            = GL_LINEAR;
    formatInfo.textureSupport           = textureSupport;
    formatInfo.filterSupport            = filterSupport;
    formatInfo.textureAttachmentSupport = textureAttachmentSupport;
    formatInfo.renderbufferSupport      = renderbufferSupport;

    InsertFormatInfo(map, formatInfo);
}

void AddCompressedFormat(InternalFormatInfoMap *map,
                         GLenum internalFormat,
                         GLuint compressedBlockWidth,
                         GLuint compressedBlockHeight,
                         GLuint compressedBlockDepth,
                         GLuint compressedBlockSize,
                         GLuint componentCount,
                         bool srgb,
                         InternalFormat::SupportCheckFunction textureSupport,
                         InternalFormat::SupportCheckFunction filterSupport,
                         InternalFormat::SupportCheckFunction textureAttachmentSupport,
                         InternalFormat::SupportCheckFunction renderbufferSupport)
{
    InternalFormat formatInfo;
    formatInfo.internalFormat           = internalFormat;
    formatInfo.sized                    = true;
    formatInfo.sizedInternalFormat      = internalFormat;
    formatInfo.compressedBlockWidth     = compressedBlockWidth;
    formatInfo.compressedBlockHeight    = compressedBlockHeight;
    formatInfo.compressedBlockDepth     = compressedBlockDepth;
    formatInfo.pixelBytes               = compressedBlockSize / 8;
    formatInfo.componentCount           = componentCount;
    formatInfo.format                   = internalFormat;
    formatInfo.type                     = GL_UNSIGNED_BYTE;
    formatInfo.componentType            = GL_UNSIGNED_NORMALIZED;
    formatInfo.colorEncoding            = (srgb ? GL_SRGB : GL_LINEAR);
    formatInfo.compressed               = true;
    formatInfo.textureSupport           = textureSupport;
    formatInfo.filterSupport            = filterSupport;
    formatInfo.textureAttachmentSupport = textureAttachmentSupport;
    formatInfo.renderbufferSupport      = renderbufferSupport;

    InsertFormatInfo(map, formatInfo);
}

// Notes:
// 1. "Texture supported" includes all the means by which texture can be created, however,
//    GL_EXT_texture_storage in ES2 is a special case, when only glTexStorage* is allowed.
//    The assumption is that ES2 validation will not check textureSupport for sized formats.
//
// 2. Sized half float types are a combination of GL_HALF_FLOAT and GL_HALF_FLOAT_OES support,
//    due to a limitation that only one type for sized formats is allowed.
//
// TODO(ynovikov): http://anglebug.com/2846 Verify support fields of BGRA, depth, stencil
// and compressed formats. Perform texturable check as part of filterable and attachment checks.
static InternalFormatInfoMap BuildInternalFormatInfoMap()
{
    InternalFormatInfoMap map;

    // From ES 3.0.1 spec, table 3.12
    map[GL_NONE][GL_NONE] = InternalFormat();

    // clang-format off

    //                 | Internal format     |sized| R | G | B | A |S | Format         | Type                             | Component type        | SRGB | Texture supported                                | Filterable     | Texture attachment                               | Renderbuffer                                 |
    AddRGBAFormat(&map, GL_R8,                true,  8,  0,  0,  0, 0, GL_RED,          GL_UNSIGNED_BYTE,                  GL_UNSIGNED_NORMALIZED, false, SizedRGSupport,                                    AlwaysSupported, SizedRGSupport,                                    RequireESOrExt<3, 0, &Extensions::textureRG>  );
    AddRGBAFormat(&map, GL_R8_SNORM,          true,  8,  0,  0,  0, 0, GL_RED,          GL_BYTE,                           GL_SIGNED_NORMALIZED,   false, RequireES<3, 0>,                                   AlwaysSupported, NeverSupported,                                    NeverSupported                                );
    AddRGBAFormat(&map, GL_RG8,               true,  8,  8,  0,  0, 0, GL_RG,           GL_UNSIGNED_BYTE,                  GL_UNSIGNED_NORMALIZED, false, SizedRGSupport,                                    AlwaysSupported, SizedRGSupport,                                    RequireESOrExt<3, 0, &Extensions::textureRG>  );
    AddRGBAFormat(&map, GL_RG8_SNORM,         true,  8,  8,  0,  0, 0, GL_RG,           GL_BYTE,                           GL_SIGNED_NORMALIZED,   false, RequireES<3, 0>,                                   AlwaysSupported, NeverSupported,                                    NeverSupported                                );
    AddRGBAFormat(&map, GL_RGB8,              true,  8,  8,  8,  0, 0, GL_RGB,          GL_UNSIGNED_BYTE,                  GL_UNSIGNED_NORMALIZED, false, RequireESOrExt<3, 0, &Extensions::textureStorage>, AlwaysSupported, RequireESOrExt<3, 0, &Extensions::textureStorage>, RequireESOrExt<3, 0, &Extensions::rgb8rgba8>  );
    AddRGBAFormat(&map, GL_RGB8_SNORM,        true,  8,  8,  8,  0, 0, GL_RGB,          GL_BYTE,                           GL_SIGNED_NORMALIZED,   false, RequireES<3, 0>,                                   AlwaysSupported, NeverSupported,                                    NeverSupported                                );
    AddRGBAFormat(&map, GL_RGB565,            true,  5,  6,  5,  0, 0, GL_RGB,          GL_UNSIGNED_SHORT_5_6_5,           GL_UNSIGNED_NORMALIZED, false, RequireESOrExt<3, 0, &Extensions::textureStorage>, AlwaysSupported, RequireESOrExt<3, 0, &Extensions::textureStorage>, RequireES<2, 0>                               );
    AddRGBAFormat(&map, GL_RGBA4,             true,  4,  4,  4,  4, 0, GL_RGBA,         GL_UNSIGNED_SHORT_4_4_4_4,         GL_UNSIGNED_NORMALIZED, false, RequireESOrExt<3, 0, &Extensions::textureStorage>, AlwaysSupported, RequireESOrExt<3, 0, &Extensions::textureStorage>, RequireES<2, 0>                               );
    AddRGBAFormat(&map, GL_RGB5_A1,           true,  5,  5,  5,  1, 0, GL_RGBA,         GL_UNSIGNED_SHORT_5_5_5_1,         GL_UNSIGNED_NORMALIZED, false, RequireESOrExt<3, 0, &Extensions::textureStorage>, AlwaysSupported, RequireESOrExt<3, 0, &Extensions::textureStorage>, RequireES<2, 0>                               );
    AddRGBAFormat(&map, GL_RGBA8,             true,  8,  8,  8,  8, 0, GL_RGBA,         GL_UNSIGNED_BYTE,                  GL_UNSIGNED_NORMALIZED, false, RequireESOrExt<3, 0, &Extensions::textureStorage>, AlwaysSupported, RequireESOrExt<3, 0, &Extensions::textureStorage>, RequireESOrExt<3, 0, &Extensions::rgb8rgba8>  );
    AddRGBAFormat(&map, GL_RGBA8_SNORM,       true,  8,  8,  8,  8, 0, GL_RGBA,         GL_BYTE,                           GL_SIGNED_NORMALIZED,   false, RequireES<3, 0>,                                   AlwaysSupported, NeverSupported,                                    NeverSupported                                );
    AddRGBAFormat(&map, GL_RGB10_A2,          true, 10, 10, 10,  2, 0, GL_RGBA,         GL_UNSIGNED_INT_2_10_10_10_REV,    GL_UNSIGNED_NORMALIZED, false, RequireES<3, 0>,                                   AlwaysSupported, RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_RGB10_A2UI,        true, 10, 10, 10,  2, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT_2_10_10_10_REV,    GL_UNSIGNED_INT,        false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_SRGB8,             true,  8,  8,  8,  0, 0, GL_RGB,          GL_UNSIGNED_BYTE,                  GL_UNSIGNED_NORMALIZED, true,  RequireES<3, 0>,                                   AlwaysSupported, NeverSupported,                                    NeverSupported                                );
    AddRGBAFormat(&map, GL_SRGB8_ALPHA8,      true,  8,  8,  8,  8, 0, GL_RGBA,         GL_UNSIGNED_BYTE,                  GL_UNSIGNED_NORMALIZED, true,  RequireES<3, 0>,                                   AlwaysSupported, RequireES<3, 0>,                                   RequireESOrExt<3, 0, &Extensions::sRGB>       );
    AddRGBAFormat(&map, GL_R11F_G11F_B10F,    true, 11, 11, 10,  0, 0, GL_RGB,          GL_UNSIGNED_INT_10F_11F_11F_REV,   GL_FLOAT,               false, RequireES<3, 0>,                                   AlwaysSupported, RequireExt<&Extensions::colorBufferFloat>,         RequireExt<&Extensions::colorBufferFloat>     );
    AddRGBAFormat(&map, GL_RGB9_E5,           true,  9,  9,  9,  0, 5, GL_RGB,          GL_UNSIGNED_INT_5_9_9_9_REV,       GL_FLOAT,               false, RequireES<3, 0>,                                   AlwaysSupported, NeverSupported,                                    NeverSupported                                );
    AddRGBAFormat(&map, GL_R8I,               true,  8,  0,  0,  0, 0, GL_RED_INTEGER,  GL_BYTE,                           GL_INT,                 false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_R8UI,              true,  8,  0,  0,  0, 0, GL_RED_INTEGER,  GL_UNSIGNED_BYTE,                  GL_UNSIGNED_INT,        false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_R16I,              true, 16,  0,  0,  0, 0, GL_RED_INTEGER,  GL_SHORT,                          GL_INT,                 false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_R16UI,             true, 16,  0,  0,  0, 0, GL_RED_INTEGER,  GL_UNSIGNED_SHORT,                 GL_UNSIGNED_INT,        false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_R32I,              true, 32,  0,  0,  0, 0, GL_RED_INTEGER,  GL_INT,                            GL_INT,                 false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_R32UI,             true, 32,  0,  0,  0, 0, GL_RED_INTEGER,  GL_UNSIGNED_INT,                   GL_UNSIGNED_INT,        false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_RG8I,              true,  8,  8,  0,  0, 0, GL_RG_INTEGER,   GL_BYTE,                           GL_INT,                 false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_RG8UI,             true,  8,  8,  0,  0, 0, GL_RG_INTEGER,   GL_UNSIGNED_BYTE,                  GL_UNSIGNED_INT,        false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_RG16I,             true, 16, 16,  0,  0, 0, GL_RG_INTEGER,   GL_SHORT,                          GL_INT,                 false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_RG16UI,            true, 16, 16,  0,  0, 0, GL_RG_INTEGER,   GL_UNSIGNED_SHORT,                 GL_UNSIGNED_INT,        false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_RG32I,             true, 32, 32,  0,  0, 0, GL_RG_INTEGER,   GL_INT,                            GL_INT,                 false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_RG32UI,            true, 32, 32,  0,  0, 0, GL_RG_INTEGER,   GL_UNSIGNED_INT,                   GL_UNSIGNED_INT,        false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_RGB8I,             true,  8,  8,  8,  0, 0, GL_RGB_INTEGER,  GL_BYTE,                           GL_INT,                 false, RequireES<3, 0>,                                   NeverSupported,  NeverSupported,                                    NeverSupported                                );
    AddRGBAFormat(&map, GL_RGB8UI,            true,  8,  8,  8,  0, 0, GL_RGB_INTEGER,  GL_UNSIGNED_BYTE,                  GL_UNSIGNED_INT,        false, RequireES<3, 0>,                                   NeverSupported,  NeverSupported,                                    NeverSupported                                );
    AddRGBAFormat(&map, GL_RGB16I,            true, 16, 16, 16,  0, 0, GL_RGB_INTEGER,  GL_SHORT,                          GL_INT,                 false, RequireES<3, 0>,                                   NeverSupported,  NeverSupported,                                    NeverSupported                                );
    AddRGBAFormat(&map, GL_RGB16UI,           true, 16, 16, 16,  0, 0, GL_RGB_INTEGER,  GL_UNSIGNED_SHORT,                 GL_UNSIGNED_INT,        false, RequireES<3, 0>,                                   NeverSupported,  NeverSupported,                                    NeverSupported                                );
    AddRGBAFormat(&map, GL_RGB32I,            true, 32, 32, 32,  0, 0, GL_RGB_INTEGER,  GL_INT,                            GL_INT,                 false, RequireES<3, 0>,                                   NeverSupported,  NeverSupported,                                    NeverSupported                                );
    AddRGBAFormat(&map, GL_RGB32UI,           true, 32, 32, 32,  0, 0, GL_RGB_INTEGER,  GL_UNSIGNED_INT,                   GL_UNSIGNED_INT,        false, RequireES<3, 0>,                                   NeverSupported,  NeverSupported,                                    NeverSupported                                );
    AddRGBAFormat(&map, GL_RGBA8I,            true,  8,  8,  8,  8, 0, GL_RGBA_INTEGER, GL_BYTE,                           GL_INT,                 false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_RGBA8UI,           true,  8,  8,  8,  8, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE,                  GL_UNSIGNED_INT,        false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_RGBA16I,           true, 16, 16, 16, 16, 0, GL_RGBA_INTEGER, GL_SHORT,                          GL_INT,                 false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_RGBA16UI,          true, 16, 16, 16, 16, 0, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT,                 GL_UNSIGNED_INT,        false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_RGBA32I,           true, 32, 32, 32, 32, 0, GL_RGBA_INTEGER, GL_INT,                            GL_INT,                 false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );
    AddRGBAFormat(&map, GL_RGBA32UI,          true, 32, 32, 32, 32, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT,                   GL_UNSIGNED_INT,        false, RequireES<3, 0>,                                   NeverSupported,  RequireES<3, 0>,                                   RequireES<3, 0>                               );

    AddRGBAFormat(&map, GL_BGRA8_EXT,         true,  8,  8,  8,  8, 0, GL_BGRA_EXT,     GL_UNSIGNED_BYTE,                  GL_UNSIGNED_NORMALIZED, false, RequireExt<&Extensions::textureFormatBGRA8888>,    AlwaysSupported, RequireExt<&Extensions::textureFormatBGRA8888>,    RequireExt<&Extensions::textureFormatBGRA8888>);
    AddRGBAFormat(&map, GL_BGRA4_ANGLEX,      true,  4,  4,  4,  4, 0, GL_BGRA_EXT,     GL_UNSIGNED_SHORT_4_4_4_4_REV_EXT, GL_UNSIGNED_NORMALIZED, false, RequireExt<&Extensions::textureFormatBGRA8888>,    AlwaysSupported, RequireExt<&Extensions::textureFormatBGRA8888>,    RequireExt<&Extensions::textureFormatBGRA8888>);
    AddRGBAFormat(&map, GL_BGR5_A1_ANGLEX,    true,  5,  5,  5,  1, 0, GL_BGRA_EXT,     GL_UNSIGNED_SHORT_1_5_5_5_REV_EXT, GL_UNSIGNED_NORMALIZED, false, RequireExt<&Extensions::textureFormatBGRA8888>,    AlwaysSupported, RequireExt<&Extensions::textureFormatBGRA8888>,    RequireExt<&Extensions::textureFormatBGRA8888>);

    // Special format that is used for D3D textures that are used within ANGLE via the
    // EGL_ANGLE_d3d_texture_client_buffer extension. We don't allow uploading texture images with
    // this format, but textures in this format can be created from D3D textures, and filtering them
    // and rendering to them is allowed.
    AddRGBAFormat(&map, GL_BGRA8_SRGB_ANGLEX, true,  8,  8,  8,  8, 0, GL_BGRA_EXT,     GL_UNSIGNED_BYTE,                  GL_UNSIGNED_NORMALIZED, true,  NeverSupported,                                    AlwaysSupported, AlwaysSupported,                                   AlwaysSupported                               );

    // Special format which is not really supported, so always false for all supports.
    AddRGBAFormat(&map, GL_BGRX8_ANGLEX,      true,  8,  8,  8,  0, 0, GL_BGRA_EXT,     GL_UNSIGNED_BYTE,                  GL_UNSIGNED_NORMALIZED, false, NeverSupported,                                    NeverSupported,  NeverSupported,                                    NeverSupported                                );
    AddRGBAFormat(&map, GL_BGR565_ANGLEX,     true,  5,  6,  5,  1, 0, GL_BGRA_EXT,     GL_UNSIGNED_SHORT_5_6_5,           GL_UNSIGNED_NORMALIZED, false, NeverSupported,                                    NeverSupported,  NeverSupported,                                    NeverSupported                                );

    // Floating point formats
    //                 | Internal format |sized| R | G | B | A |S | Format | Type             | Component type | SRGB | Texture supported         | Filterable                                 | Texture attachment                          | Renderbuffer                            |
    // It's not possible to have two entries per sized format.
    // E.g. for GL_RG16F, one with GL_HALF_FLOAT type and the other with GL_HALF_FLOAT_OES type.
    // So, GL_HALF_FLOAT type formats conditions are merged with GL_HALF_FLOAT_OES type conditions.
    AddRGBAFormat(&map, GL_R16F,          true, 16,  0,  0,  0, 0, GL_RED,  GL_HALF_FLOAT,     GL_FLOAT,        false, SizedHalfFloatRGSupport,    SizedHalfFloatFilterSupport,                 SizedHalfFloatRGTextureAttachmentSupport,     SizedHalfFloatRGRenderbufferSupport      );
    AddRGBAFormat(&map, GL_RG16F,         true, 16, 16,  0,  0, 0, GL_RG,   GL_HALF_FLOAT,     GL_FLOAT,        false, SizedHalfFloatRGSupport,    SizedHalfFloatFilterSupport,                 SizedHalfFloatRGTextureAttachmentSupport,     SizedHalfFloatRGRenderbufferSupport      );
    AddRGBAFormat(&map, GL_RGB16F,        true, 16, 16, 16,  0, 0, GL_RGB,  GL_HALF_FLOAT,     GL_FLOAT,        false, SizedHalfFloatSupport,      SizedHalfFloatFilterSupport,                 SizedHalfFloatRGBTextureAttachmentSupport,    SizedHalfFloatRGBRenderbufferSupport     );
    AddRGBAFormat(&map, GL_RGBA16F,       true, 16, 16, 16, 16, 0, GL_RGBA, GL_HALF_FLOAT,     GL_FLOAT,        false, SizedHalfFloatSupport,      SizedHalfFloatFilterSupport,                 SizedHalfFloatRGBATextureAttachmentSupport,   SizedHalfFloatRGBARenderbufferSupport    );
    AddRGBAFormat(&map, GL_R32F,          true, 32,  0,  0,  0, 0, GL_RED,  GL_FLOAT,          GL_FLOAT,        false, SizedFloatRGSupport,        RequireExt<&Extensions::textureFloatLinear>, RequireExt<&Extensions::colorBufferFloat>,    RequireExt<&Extensions::colorBufferFloat>);
    AddRGBAFormat(&map, GL_RG32F,         true, 32, 32,  0,  0, 0, GL_RG,   GL_FLOAT,          GL_FLOAT,        false, SizedFloatRGSupport,        RequireExt<&Extensions::textureFloatLinear>, RequireExt<&Extensions::colorBufferFloat>,    RequireExt<&Extensions::colorBufferFloat>);
    AddRGBAFormat(&map, GL_RGB32F,        true, 32, 32, 32,  0, 0, GL_RGB,  GL_FLOAT,          GL_FLOAT,        false, SizedFloatRGBSupport,       RequireExt<&Extensions::textureFloatLinear>, RequireExt<&Extensions::colorBufferFloatRGB>, NeverSupported                           );
    AddRGBAFormat(&map, GL_RGBA32F,       true, 32, 32, 32, 32, 0, GL_RGBA, GL_FLOAT,          GL_FLOAT,        false, SizedFloatRGBASupport,      RequireExt<&Extensions::textureFloatLinear>, SizedFloatRGBARenderableSupport,              SizedFloatRGBARenderableSupport          );

    // ANGLE Depth stencil formats
    //                         | Internal format         |sized| D |S | X | Format            | Type                             | Component type        | Texture supported                                                | Filterable                                      | Texture attachment                                                                   | Renderbuffer                                                                         |
    AddDepthStencilFormat(&map, GL_DEPTH_COMPONENT16,     true, 16, 0,  0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT,                 GL_UNSIGNED_NORMALIZED, RequireES<1, 0>,                                                   RequireESOrExtOrExt<3, 0, &Extensions::depthTextureANGLE, &Extensions::depthTextureOES>, RequireES<1, 0>,                                RequireES<1, 0>                                                                       );
    AddDepthStencilFormat(&map, GL_DEPTH_COMPONENT24,     true, 24, 0,  0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT,                   GL_UNSIGNED_NORMALIZED, RequireES<3, 0>,                                                   RequireESOrExt<3, 0, &Extensions::depthTextureANGLE>, RequireES<3, 0>,                                                                   RequireESOrExt<3, 0, &Extensions::depth24OES>                                            );
    AddDepthStencilFormat(&map, GL_DEPTH_COMPONENT32F,    true, 32, 0,  0, GL_DEPTH_COMPONENT, GL_FLOAT,                          GL_FLOAT,               RequireES<3, 0>,                                                   RequireESOrExt<3, 0, &Extensions::depthTextureANGLE>, RequireES<3, 0>,                                                                   RequireES<3, 0>                                                                       );
    AddDepthStencilFormat(&map, GL_DEPTH_COMPONENT32_OES, true, 32, 0,  0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT,                   GL_UNSIGNED_NORMALIZED, RequireExtOrExtOrExt<&Extensions::depthTextureANGLE, &Extensions::depthTextureOES, &Extensions::depth32>, AlwaysSupported, RequireExtOrExtOrExt<&Extensions::depthTextureANGLE, &Extensions::depthTextureOES, &Extensions::depth32>, RequireExtOrExtOrExt<&Extensions::depthTextureANGLE, &Extensions::depthTextureOES, &Extensions::depth32>);
    AddDepthStencilFormat(&map, GL_DEPTH24_STENCIL8,      true, 24, 8,  0, GL_DEPTH_STENCIL,   GL_UNSIGNED_INT_24_8,              GL_UNSIGNED_NORMALIZED, RequireESOrExt<3, 0, &Extensions::depthTextureANGLE>,              AlwaysSupported,                                  RequireESOrExtOrExt<3, 0, &Extensions::depthTextureANGLE, &Extensions::packedDepthStencil>, RequireESOrExtOrExt<3, 0, &Extensions::depthTextureANGLE, &Extensions::packedDepthStencil>);
    AddDepthStencilFormat(&map, GL_DEPTH32F_STENCIL8,     true, 32, 8, 24, GL_DEPTH_STENCIL,   GL_FLOAT_32_UNSIGNED_INT_24_8_REV, GL_FLOAT,               RequireES<3, 0>,                                                   AlwaysSupported,                                  RequireES<3, 0>,                                                                       RequireES<3, 0>                                                                       );
    // STENCIL_INDEX8 is special-cased, see around the bottom of the list.

    // Luminance alpha formats
    //                | Internal format           |sized| L | A | Format            | Type             | Component type        | Texture supported                                                           | Filterable                                     | Texture attachment | Renderbuffer |
    AddLUMAFormat(&map, GL_ALPHA8_EXT,             true,  0,  8, GL_ALPHA,           GL_UNSIGNED_BYTE,  GL_UNSIGNED_NORMALIZED, RequireExt<&Extensions::textureStorage>,                                      AlwaysSupported,                                 NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_LUMINANCE8_EXT,         true,  8,  0, GL_LUMINANCE,       GL_UNSIGNED_BYTE,  GL_UNSIGNED_NORMALIZED, RequireExt<&Extensions::textureStorage>,                                      AlwaysSupported,                                 NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_LUMINANCE8_ALPHA8_EXT,  true,  8,  8, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE,  GL_UNSIGNED_NORMALIZED, RequireExt<&Extensions::textureStorage>,                                      AlwaysSupported,                                 NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_ALPHA16F_EXT,           true,  0, 16, GL_ALPHA,           GL_HALF_FLOAT_OES, GL_FLOAT,               RequireExtAndExt<&Extensions::textureStorage, &Extensions::textureHalfFloat>, RequireExt<&Extensions::textureHalfFloatLinear>, NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_LUMINANCE16F_EXT,       true, 16,  0, GL_LUMINANCE,       GL_HALF_FLOAT_OES, GL_FLOAT,               RequireExtAndExt<&Extensions::textureStorage, &Extensions::textureHalfFloat>, RequireExt<&Extensions::textureHalfFloatLinear>, NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_LUMINANCE_ALPHA16F_EXT, true, 16, 16, GL_LUMINANCE_ALPHA, GL_HALF_FLOAT_OES, GL_FLOAT,               RequireExtAndExt<&Extensions::textureStorage, &Extensions::textureHalfFloat>, RequireExt<&Extensions::textureHalfFloatLinear>, NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_ALPHA32F_EXT,           true,  0, 32, GL_ALPHA,           GL_FLOAT,          GL_FLOAT,               RequireExtAndExt<&Extensions::textureStorage, &Extensions::textureFloat>,     RequireExt<&Extensions::textureFloatLinear>,     NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_LUMINANCE32F_EXT,       true, 32,  0, GL_LUMINANCE,       GL_FLOAT,          GL_FLOAT,               RequireExtAndExt<&Extensions::textureStorage, &Extensions::textureFloat>,     RequireExt<&Extensions::textureFloatLinear>,     NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_LUMINANCE_ALPHA32F_EXT, true, 32, 32, GL_LUMINANCE_ALPHA, GL_FLOAT,          GL_FLOAT,               RequireExtAndExt<&Extensions::textureStorage, &Extensions::textureFloat>,     RequireExt<&Extensions::textureFloatLinear>,     NeverSupported,      NeverSupported);

    // Compressed formats, From ES 3.0.1 spec, table 3.16
    //                       | Internal format                             |W |H |D | BS |CC| SRGB | Texture supported                                                                                                      | Filterable     | Texture attachment | Renderbuffer |
    AddCompressedFormat(&map, GL_COMPRESSED_R11_EAC,                        4, 4, 1,  64, 1, false, RequireESOrExtOrExt<3, 0, &Extensions::compressedTextureETC, &Extensions::compressedEACR11UnsignedTexture>,              AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SIGNED_R11_EAC,                 4, 4, 1,  64, 1, false, RequireESOrExtOrExt<3, 0, &Extensions::compressedTextureETC, &Extensions::compressedEACR11SignedTexture>,                AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RG11_EAC,                       4, 4, 1, 128, 2, false, RequireESOrExtOrExt<3, 0, &Extensions::compressedTextureETC, &Extensions::compressedEACRG11UnsignedTexture>,             AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SIGNED_RG11_EAC,                4, 4, 1, 128, 2, false, RequireESOrExtOrExt<3, 0, &Extensions::compressedTextureETC, &Extensions::compressedEACRG11SignedTexture>,               AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGB8_ETC2,                      4, 4, 1,  64, 3, false, RequireESOrExtOrExt<3, 0, &Extensions::compressedTextureETC, &Extensions::compressedETC2RGB8Texture>,                    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ETC2,                     4, 4, 1,  64, 3, true,  RequireESOrExtOrExt<3, 0, &Extensions::compressedTextureETC, &Extensions::compressedETC2sRGB8Texture>,                   AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2,  4, 4, 1,  64, 3, false, RequireESOrExtOrExt<3, 0, &Extensions::compressedTextureETC, &Extensions::compressedETC2PunchthroughARGB8Texture>,       AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2, 4, 4, 1,  64, 3, true,  RequireESOrExtOrExt<3, 0, &Extensions::compressedTextureETC, &Extensions::compressedETC2PunchthroughAsRGB8AlphaTexture>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA8_ETC2_EAC,                 4, 4, 1, 128, 4, false, RequireESOrExtOrExt<3, 0, &Extensions::compressedTextureETC, &Extensions::compressedETC2RGBA8Texture>,                   AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,          4, 4, 1, 128, 4, true,  RequireESOrExtOrExt<3, 0, &Extensions::compressedTextureETC, &Extensions::compressedETC2sRGB8Alpha8Texture>,             AlwaysSupported, NeverSupported,      NeverSupported);

    // From GL_EXT_texture_compression_dxt1
    //                       | Internal format                   |W |H |D | BS |CC| SRGB | Texture supported                                 | Filterable     | Texture attachment | Renderbuffer |
    AddCompressedFormat(&map, GL_COMPRESSED_RGB_S3TC_DXT1_EXT,    4, 4, 1,  64, 3, false, RequireExt<&Extensions::textureCompressionDXT1>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,   4, 4, 1,  64, 4, false, RequireExt<&Extensions::textureCompressionDXT1>,    AlwaysSupported, NeverSupported,      NeverSupported);

    // From GL_ANGLE_texture_compression_dxt3
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_S3TC_DXT3_ANGLE, 4, 4, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionDXT3>,    AlwaysSupported, NeverSupported,      NeverSupported);

    // From GL_ANGLE_texture_compression_dxt5
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_S3TC_DXT5_ANGLE, 4, 4, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionDXT5>,    AlwaysSupported, NeverSupported,      NeverSupported);

    // From GL_OES_compressed_ETC1_RGB8_texture
    AddCompressedFormat(&map, GL_ETC1_RGB8_OES,                   4, 4, 1,  64, 3, false, RequireExt<&Extensions::compressedETC1RGB8Texture>, AlwaysSupported, NeverSupported,      NeverSupported);

    // From GL_EXT_texture_compression_s3tc_srgb
    //                       | Internal format                       |W |H |D | BS |CC|SRGB | Texture supported                                 | Filterable     | Texture attachment | Renderbuffer |
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB_S3TC_DXT1_EXT,       4, 4, 1,  64, 3, true, RequireExt<&Extensions::textureCompressionS3TCsRGB>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, 4, 4, 1,  64, 4, true, RequireExt<&Extensions::textureCompressionS3TCsRGB>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, 4, 4, 1, 128, 4, true, RequireExt<&Extensions::textureCompressionS3TCsRGB>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT, 4, 4, 1, 128, 4, true, RequireExt<&Extensions::textureCompressionS3TCsRGB>, AlwaysSupported, NeverSupported,      NeverSupported);

    // From GL_KHR_texture_compression_astc_ldr and KHR_texture_compression_astc_hdr and GL_OES_texture_compression_astc
    //                       | Internal format                          | W | H |D | BS |CC| SRGB | Texture supported                                    | Filterable     | Texture attachment | Renderbuffer |
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_4x4_KHR,            4,  4, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_5x4_KHR,            5,  4, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_5x5_KHR,            5,  5, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_6x5_KHR,            6,  5, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_6x6_KHR,            6,  6, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_8x5_KHR,            8,  5, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_8x6_KHR,            8,  6, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_8x8_KHR,            8,  8, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_10x5_KHR,          10,  5, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_10x6_KHR,          10,  6, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_10x8_KHR,          10,  8, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_10x10_KHR,         10, 10, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_12x10_KHR,         12, 10, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_12x12_KHR,         12, 12, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);

    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR,    4,  4, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR,    5,  4, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR,    5,  5, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR,    6,  5, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR,    6,  6, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR,    8,  5, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR,    8,  6, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR,    8,  8, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR,  10,  5, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR,  10,  6, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR,  10,  8, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR, 10, 10, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR, 12, 10, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR, 12, 12, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCLDRKHR>, AlwaysSupported, NeverSupported,      NeverSupported);

    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_3x3x3_OES,          3,  3, 3, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_4x3x3_OES,          4,  3, 3, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_4x4x3_OES,          4,  4, 3, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_4x4x4_OES,          4,  4, 4, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_5x4x4_OES,          5,  4, 4, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_5x5x4_OES,          5,  5, 4, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_5x5x5_OES,          5,  5, 5, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_6x5x5_OES,          6,  5, 5, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_6x6x5_OES,          6,  6, 5, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_ASTC_6x6x6_OES,          6,  6, 6, 128, 4, false, RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);

    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_3x3x3_OES,  3,  3, 3, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x3x3_OES,  4,  3, 3, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4x3_OES,  4,  4, 3, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4x4_OES,  4,  4, 4, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4x4_OES,  5,  4, 4, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5x4_OES,  5,  5, 4, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5x5_OES,  5,  5, 5, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5x5_OES,  6,  5, 5, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6x5_OES,  6,  6, 5, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6x6_OES,  6,  6, 6, 128, 4, true,  RequireExt<&Extensions::textureCompressionASTCOES>,    AlwaysSupported, NeverSupported,      NeverSupported);

    // From EXT_texture_compression_bptc
    //                       | Internal format                         | W | H |D | BS |CC| SRGB | Texture supported                              | Filterable     | Texture attachment | Renderbuffer |
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_BPTC_UNORM_EXT,         4,  4, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionBPTC>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT,   4,  4, 1, 128, 4, true,  RequireExt<&Extensions::textureCompressionBPTC>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT,   4,  4, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionBPTC>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT, 4,  4, 1, 128, 4, false, RequireExt<&Extensions::textureCompressionBPTC>, AlwaysSupported, NeverSupported,      NeverSupported);

    // From GL_IMG_texture_compression_pvrtc
    //                       | Internal format                       | W | H | D | BS |CC| SRGB | Texture supported                                 | Filterable     | Texture attachment | Renderbuffer |
    AddCompressedFormat(&map, GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG,      4,  4,  1,  64, 3, false, RequireExt<&Extensions::compressedTexturePVRTC>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG,      8,  4,  1,  64, 3, false, RequireExt<&Extensions::compressedTexturePVRTC>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG,     4,  4,  1,  64, 4, false, RequireExt<&Extensions::compressedTexturePVRTC>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG,     8,  4,  1,  64, 4, false, RequireExt<&Extensions::compressedTexturePVRTC>,    AlwaysSupported, NeverSupported,      NeverSupported);

    // From GL_EXT_pvrtc_sRGB
    //                       | Internal format                             | W | H | D | BS |CC| SRGB | Texture supported                                     | Filterable     | Texture attachment | Renderbuffer |
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB_PVRTC_2BPPV1_EXT,           8,  4,  1,  64, 3,  true, RequireExt<&Extensions::compressedTexturePVRTCsRGB>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB_PVRTC_4BPPV1_EXT,           4,  4,  1,  64, 3,  true, RequireExt<&Extensions::compressedTexturePVRTCsRGB>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT,     8,  4,  1,  64, 4,  true, RequireExt<&Extensions::compressedTexturePVRTCsRGB>,    AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT,     4,  4,  1,  64, 4,  true, RequireExt<&Extensions::compressedTexturePVRTCsRGB>,    AlwaysSupported, NeverSupported,      NeverSupported);

    // For STENCIL_INDEX8 we chose a normalized component type for the following reasons:
    // - Multisampled buffer are disallowed for non-normalized integer component types and we want to support it for STENCIL_INDEX8
    // - All other stencil formats (all depth-stencil) are either float or normalized
    // - It affects only validation of internalformat in RenderbufferStorageMultisample.
    //                         | Internal format  |sized|D |S |X | Format    | Type            | Component type        | Texture supported | Filterable    | Texture attachment | Renderbuffer  |
    AddDepthStencilFormat(&map, GL_STENCIL_INDEX8, true, 0, 8, 0, GL_STENCIL, GL_UNSIGNED_BYTE, GL_UNSIGNED_NORMALIZED, RequireES<1, 0>,    NeverSupported, RequireES<1, 0>,     RequireES<1, 0>);

    // From GL_ANGLE_lossy_etc_decode
    //                       | Internal format                                                |W |H |D |BS |CC| SRGB | Texture supported                      | Filterable     | Texture attachment | Renderbuffer |
    AddCompressedFormat(&map, GL_ETC1_RGB8_LOSSY_DECODE_ANGLE,                                 4, 4, 1, 64, 3, false, RequireExt<&Extensions::lossyETCDecode>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGB8_LOSSY_DECODE_ETC2_ANGLE,                      4, 4, 1, 64, 3, false, RequireExt<&Extensions::lossyETCDecode>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_LOSSY_DECODE_ETC2_ANGLE,                     4, 4, 1, 64, 3, true,  RequireExt<&Extensions::lossyETCDecode>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_LOSSY_DECODE_ETC2_ANGLE,  4, 4, 1, 64, 3, false, RequireExt<&Extensions::lossyETCDecode>, AlwaysSupported, NeverSupported,      NeverSupported);
    AddCompressedFormat(&map, GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_LOSSY_DECODE_ETC2_ANGLE, 4, 4, 1, 64, 3, true,  RequireExt<&Extensions::lossyETCDecode>, AlwaysSupported, NeverSupported,      NeverSupported);

    // From GL_EXT_texture_norm16
    //                 | Internal format    |sized| R | G | B | A |S | Format | Type             | Component type        | SRGB | Texture supported                     | Filterable     | Texture attachment                    | Renderbuffer                         |
    AddRGBAFormat(&map, GL_R16_EXT,          true, 16,  0,  0,  0, 0, GL_RED,  GL_UNSIGNED_SHORT, GL_UNSIGNED_NORMALIZED, false, RequireExt<&Extensions::textureNorm16>, AlwaysSupported, RequireExt<&Extensions::textureNorm16>, RequireExt<&Extensions::textureNorm16>);
    AddRGBAFormat(&map, GL_R16_SNORM_EXT,    true, 16,  0,  0,  0, 0, GL_RED,  GL_SHORT,          GL_SIGNED_NORMALIZED,   false, RequireExt<&Extensions::textureNorm16>, AlwaysSupported, NeverSupported,                         NeverSupported                        );
    AddRGBAFormat(&map, GL_RG16_EXT,         true, 16, 16,  0,  0, 0, GL_RG,   GL_UNSIGNED_SHORT, GL_UNSIGNED_NORMALIZED, false, RequireExt<&Extensions::textureNorm16>, AlwaysSupported, RequireExt<&Extensions::textureNorm16>, RequireExt<&Extensions::textureNorm16>);
    AddRGBAFormat(&map, GL_RG16_SNORM_EXT,   true, 16, 16,  0,  0, 0, GL_RG,   GL_SHORT,          GL_SIGNED_NORMALIZED,   false, RequireExt<&Extensions::textureNorm16>, AlwaysSupported, NeverSupported,                         NeverSupported                        );
    AddRGBAFormat(&map, GL_RGB16_EXT,        true, 16, 16, 16,  0, 0, GL_RGB,  GL_UNSIGNED_SHORT, GL_UNSIGNED_NORMALIZED, false, RequireExt<&Extensions::textureNorm16>, AlwaysSupported, NeverSupported,                         NeverSupported                        );
    AddRGBAFormat(&map, GL_RGB16_SNORM_EXT,  true, 16, 16, 16,  0, 0, GL_RGB,  GL_SHORT,          GL_SIGNED_NORMALIZED,   false, RequireExt<&Extensions::textureNorm16>, AlwaysSupported, NeverSupported,                         NeverSupported                        );
    AddRGBAFormat(&map, GL_RGBA16_EXT,       true, 16, 16, 16, 16, 0, GL_RGBA, GL_UNSIGNED_SHORT, GL_UNSIGNED_NORMALIZED, false, RequireExt<&Extensions::textureNorm16>, AlwaysSupported, RequireExt<&Extensions::textureNorm16>, RequireExt<&Extensions::textureNorm16>);
    AddRGBAFormat(&map, GL_RGBA16_SNORM_EXT, true, 16, 16, 16, 16, 0, GL_RGBA, GL_SHORT,          GL_SIGNED_NORMALIZED,   false, RequireExt<&Extensions::textureNorm16>, AlwaysSupported, NeverSupported,                         NeverSupported                        );

    // Unsized formats
    //                 | Internal format  |sized | R | G | B | A |S | Format           | Type                          | Component type        | SRGB | Texture supported                               | Filterable     | Texture attachment                            | Renderbuffer |
    AddRGBAFormat(&map, GL_RED,            false,  8,  0,  0,  0, 0, GL_RED,            GL_UNSIGNED_BYTE,               GL_UNSIGNED_NORMALIZED, false, RequireExt<&Extensions::textureRG>,               AlwaysSupported, RequireExt<&Extensions::textureRG>,             NeverSupported);
    AddRGBAFormat(&map, GL_RED,            false,  8,  0,  0,  0, 0, GL_RED,            GL_BYTE,                        GL_SIGNED_NORMALIZED,   false, NeverSupported,                                   NeverSupported,  NeverSupported,                                 NeverSupported);
    AddRGBAFormat(&map, GL_RG,             false,  8,  8,  0,  0, 0, GL_RG,             GL_UNSIGNED_BYTE,               GL_UNSIGNED_NORMALIZED, false, RequireExt<&Extensions::textureRG>,               AlwaysSupported, RequireExt<&Extensions::textureRG>,             NeverSupported);
    AddRGBAFormat(&map, GL_RG,             false,  8,  8,  0,  0, 0, GL_RG,             GL_BYTE,                        GL_SIGNED_NORMALIZED,   false, NeverSupported,                                   NeverSupported,  NeverSupported,                                 NeverSupported);
    AddRGBAFormat(&map, GL_RGB,            false,  8,  8,  8,  0, 0, GL_RGB,            GL_UNSIGNED_BYTE,               GL_UNSIGNED_NORMALIZED, false, AlwaysSupported,                                  AlwaysSupported, RequireES<2, 0>,                                NeverSupported);
    AddRGBAFormat(&map, GL_RGB,            false,  5,  6,  5,  0, 0, GL_RGB,            GL_UNSIGNED_SHORT_5_6_5,        GL_UNSIGNED_NORMALIZED, false, AlwaysSupported,                                  AlwaysSupported, RequireES<2, 0>,                                NeverSupported);
    AddRGBAFormat(&map, GL_RGB,            false,  8,  8,  8,  0, 0, GL_RGB,            GL_BYTE,                        GL_SIGNED_NORMALIZED,   false, NeverSupported,                                   NeverSupported,  NeverSupported,                                 NeverSupported);
    AddRGBAFormat(&map, GL_RGB,            false, 10, 10, 10,  0, 0, GL_RGB,            GL_UNSIGNED_INT_2_10_10_10_REV, GL_UNSIGNED_NORMALIZED, false, RequireExt<&Extensions::textureFormat2101010REV>, AlwaysSupported, NeverSupported,                                 NeverSupported);
    AddRGBAFormat(&map, GL_RGBA,           false,  4,  4,  4,  4, 0, GL_RGBA,           GL_UNSIGNED_SHORT_4_4_4_4,      GL_UNSIGNED_NORMALIZED, false, AlwaysSupported,                                  AlwaysSupported, RequireES<2, 0>,                                NeverSupported);
    AddRGBAFormat(&map, GL_RGBA,           false,  5,  5,  5,  1, 0, GL_RGBA,           GL_UNSIGNED_SHORT_5_5_5_1,      GL_UNSIGNED_NORMALIZED, false, AlwaysSupported,                                  AlwaysSupported, RequireES<2, 0>,                                NeverSupported);
    AddRGBAFormat(&map, GL_RGBA,           false,  8,  8,  8,  8, 0, GL_RGBA,           GL_UNSIGNED_BYTE,               GL_UNSIGNED_NORMALIZED, false, AlwaysSupported,                                  AlwaysSupported, RequireES<2, 0>,                                NeverSupported);
    AddRGBAFormat(&map, GL_RGBA,           false, 10, 10, 10,  2, 0, GL_RGBA,           GL_UNSIGNED_INT_2_10_10_10_REV, GL_UNSIGNED_NORMALIZED, false, RequireExt<&Extensions::textureFormat2101010REV>, AlwaysSupported, NeverSupported,                                 NeverSupported);
    AddRGBAFormat(&map, GL_RGBA,           false,  8,  8,  8,  8, 0, GL_RGBA,           GL_BYTE,                        GL_SIGNED_NORMALIZED,   false, NeverSupported,                                   NeverSupported,  NeverSupported,                                 NeverSupported);
    AddRGBAFormat(&map, GL_SRGB,           false,  8,  8,  8,  0, 0, GL_SRGB,           GL_UNSIGNED_BYTE,               GL_UNSIGNED_NORMALIZED, true,  RequireExt<&Extensions::sRGB>,                    AlwaysSupported, NeverSupported,                                 NeverSupported);
    AddRGBAFormat(&map, GL_SRGB_ALPHA_EXT, false,  8,  8,  8,  8, 0, GL_SRGB_ALPHA_EXT, GL_UNSIGNED_BYTE,               GL_UNSIGNED_NORMALIZED, true,  RequireExt<&Extensions::sRGB>,                    AlwaysSupported, RequireExt<&Extensions::sRGB>,                  NeverSupported);
#if defined(ANGLE_PLATFORM_IOS) && !defined(ANGLE_PLATFORM_MACCATALYST)
    AddRGBAFormat(&map, GL_BGRA_EXT,       false,  8,  8,  8,  8, 0, GL_BGRA_EXT,       GL_UNSIGNED_BYTE,               GL_UNSIGNED_NORMALIZED, false, RequireES<2, 0>,                                  AlwaysSupported, RequireES<2, 0>,                                NeverSupported);
#else
    AddRGBAFormat(&map, GL_BGRA_EXT,       false,  8,  8,  8,  8, 0, GL_BGRA_EXT,       GL_UNSIGNED_BYTE,               GL_UNSIGNED_NORMALIZED, false, RequireExt<&Extensions::textureFormatBGRA8888>,   AlwaysSupported, RequireExt<&Extensions::textureFormatBGRA8888>, NeverSupported);
#endif

    // Unsized integer formats
    //                 |Internal format |sized | R | G | B | A |S | Format         | Type                          | Component type | SRGB | Texture supported | Filterable    | Texture attachment | Renderbuffer |
    AddRGBAFormat(&map, GL_RED_INTEGER,  false,  8,  0,  0,  0, 0, GL_RED_INTEGER,  GL_BYTE,                        GL_INT,          false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RED_INTEGER,  false,  8,  0,  0,  0, 0, GL_RED_INTEGER,  GL_UNSIGNED_BYTE,               GL_UNSIGNED_INT, false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RED_INTEGER,  false, 16,  0,  0,  0, 0, GL_RED_INTEGER,  GL_SHORT,                       GL_INT,          false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RED_INTEGER,  false, 16,  0,  0,  0, 0, GL_RED_INTEGER,  GL_UNSIGNED_SHORT,              GL_UNSIGNED_INT, false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RED_INTEGER,  false, 32,  0,  0,  0, 0, GL_RED_INTEGER,  GL_INT,                         GL_INT,          false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RED_INTEGER,  false, 32,  0,  0,  0, 0, GL_RED_INTEGER,  GL_UNSIGNED_INT,                GL_UNSIGNED_INT, false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RG_INTEGER,   false,  8,  8,  0,  0, 0, GL_RG_INTEGER,   GL_BYTE,                        GL_INT,          false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RG_INTEGER,   false,  8,  8,  0,  0, 0, GL_RG_INTEGER,   GL_UNSIGNED_BYTE,               GL_UNSIGNED_INT, false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RG_INTEGER,   false, 16, 16,  0,  0, 0, GL_RG_INTEGER,   GL_SHORT,                       GL_INT,          false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RG_INTEGER,   false, 16, 16,  0,  0, 0, GL_RG_INTEGER,   GL_UNSIGNED_SHORT,              GL_UNSIGNED_INT, false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RG_INTEGER,   false, 32, 32,  0,  0, 0, GL_RG_INTEGER,   GL_INT,                         GL_INT,          false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RG_INTEGER,   false, 32, 32,  0,  0, 0, GL_RG_INTEGER,   GL_UNSIGNED_INT,                GL_UNSIGNED_INT, false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RGB_INTEGER,  false,  8,  8,  8,  0, 0, GL_RGB_INTEGER,  GL_BYTE,                        GL_INT,          false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RGB_INTEGER,  false,  8,  8,  8,  0, 0, GL_RGB_INTEGER,  GL_UNSIGNED_BYTE,               GL_UNSIGNED_INT, false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RGB_INTEGER,  false, 16, 16, 16,  0, 0, GL_RGB_INTEGER,  GL_SHORT,                       GL_INT,          false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RGB_INTEGER,  false, 16, 16, 16,  0, 0, GL_RGB_INTEGER,  GL_UNSIGNED_SHORT,              GL_UNSIGNED_INT, false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RGB_INTEGER,  false, 32, 32, 32,  0, 0, GL_RGB_INTEGER,  GL_INT,                         GL_INT,          false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RGB_INTEGER,  false, 32, 32, 32,  0, 0, GL_RGB_INTEGER,  GL_UNSIGNED_INT,                GL_UNSIGNED_INT, false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RGBA_INTEGER, false,  8,  8,  8,  8, 0, GL_RGBA_INTEGER, GL_BYTE,                        GL_INT,          false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RGBA_INTEGER, false,  8,  8,  8,  8, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE,               GL_UNSIGNED_INT, false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RGBA_INTEGER, false, 16, 16, 16, 16, 0, GL_RGBA_INTEGER, GL_SHORT,                       GL_INT,          false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RGBA_INTEGER, false, 16, 16, 16, 16, 0, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT,              GL_UNSIGNED_INT, false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RGBA_INTEGER, false, 32, 32, 32, 32, 0, GL_RGBA_INTEGER, GL_INT,                         GL_INT,          false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RGBA_INTEGER, false, 32, 32, 32, 32, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT,                GL_UNSIGNED_INT, false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);
    AddRGBAFormat(&map, GL_RGBA_INTEGER, false, 10, 10, 10,  2, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT_2_10_10_10_REV, GL_UNSIGNED_INT, false, RequireES<3, 0>,    NeverSupported, NeverSupported,      NeverSupported);

    // Unsized floating point formats
    //                 |Internal format |sized | R | G | B | A |S | Format | Type                           | Comp    | SRGB | Texture supported                                                      | Filterable                                     | Texture attachment                            | Renderbuffer |
    AddRGBAFormat(&map, GL_RED,          false, 16,  0,  0,  0, 0, GL_RED,  GL_HALF_FLOAT,                   GL_FLOAT, false, NeverSupported,                                                          NeverSupported,                                  NeverSupported,                                 NeverSupported);
    AddRGBAFormat(&map, GL_RG,           false, 16, 16,  0,  0, 0, GL_RG,   GL_HALF_FLOAT,                   GL_FLOAT, false, NeverSupported,                                                          NeverSupported,                                  NeverSupported,                                 NeverSupported);
    AddRGBAFormat(&map, GL_RGB,          false, 16, 16, 16,  0, 0, GL_RGB,  GL_HALF_FLOAT,                   GL_FLOAT, false, NeverSupported,                                                          NeverSupported,                                  NeverSupported,                                 NeverSupported);
    AddRGBAFormat(&map, GL_RGBA,         false, 16, 16, 16, 16, 0, GL_RGBA, GL_HALF_FLOAT,                   GL_FLOAT, false, NeverSupported,                                                          NeverSupported,                                  NeverSupported,                                 NeverSupported);
    AddRGBAFormat(&map, GL_RED,          false, 16,  0,  0,  0, 0, GL_RED,  GL_HALF_FLOAT_OES,               GL_FLOAT, false, RequireExtAndExt<&Extensions::textureHalfFloat, &Extensions::textureRG>, RequireExt<&Extensions::textureHalfFloatLinear>, AlwaysSupported,                                NeverSupported);
    AddRGBAFormat(&map, GL_RG,           false, 16, 16,  0,  0, 0, GL_RG,   GL_HALF_FLOAT_OES,               GL_FLOAT, false, RequireExtAndExt<&Extensions::textureHalfFloat, &Extensions::textureRG>, RequireExt<&Extensions::textureHalfFloatLinear>, AlwaysSupported,                                NeverSupported);
    AddRGBAFormat(&map, GL_RGB,          false, 16, 16, 16,  0, 0, GL_RGB,  GL_HALF_FLOAT_OES,               GL_FLOAT, false, RequireExt<&Extensions::textureHalfFloat>,                               RequireExt<&Extensions::textureHalfFloatLinear>, UnsizedHalfFloatOESRGBATextureAttachmentSupport, NeverSupported);
    AddRGBAFormat(&map, GL_RGBA,         false, 16, 16, 16, 16, 0, GL_RGBA, GL_HALF_FLOAT_OES,               GL_FLOAT, false, RequireExt<&Extensions::textureHalfFloat>,                               RequireExt<&Extensions::textureHalfFloatLinear>, UnsizedHalfFloatOESRGBATextureAttachmentSupport, NeverSupported);
    AddRGBAFormat(&map, GL_RED,          false, 32,  0,  0,  0, 0, GL_RED,  GL_FLOAT,                        GL_FLOAT, false, RequireExtAndExt<&Extensions::textureFloat, &Extensions::textureRG>,     RequireExt<&Extensions::textureFloatLinear>,     AlwaysSupported,                                NeverSupported);
    AddRGBAFormat(&map, GL_RG,           false, 32, 32,  0,  0, 0, GL_RG,   GL_FLOAT,                        GL_FLOAT, false, RequireExtAndExt<&Extensions::textureFloat, &Extensions::textureRG>,     RequireExt<&Extensions::textureFloatLinear>,     AlwaysSupported,                                NeverSupported);
    AddRGBAFormat(&map, GL_RGB,          false, 32, 32, 32,  0, 0, GL_RGB,  GL_FLOAT,                        GL_FLOAT, false, RequireExt<&Extensions::textureFloat>,                                   RequireExt<&Extensions::textureFloatLinear>,     NeverSupported,                                 NeverSupported);
    AddRGBAFormat(&map, GL_RGB,          false,  9,  9,  9,  0, 5, GL_RGB,  GL_UNSIGNED_INT_5_9_9_9_REV,     GL_FLOAT, false, NeverSupported,                                                          NeverSupported,                                  NeverSupported,                                 NeverSupported);
    AddRGBAFormat(&map, GL_RGB,          false, 11, 11, 10,  0, 0, GL_RGB,  GL_UNSIGNED_INT_10F_11F_11F_REV, GL_FLOAT, false, NeverSupported,                                                          NeverSupported,                                  NeverSupported,                                 NeverSupported);
    AddRGBAFormat(&map, GL_RGBA,         false, 32, 32, 32, 32, 0, GL_RGBA, GL_FLOAT,                        GL_FLOAT, false, RequireExt<&Extensions::textureFloat>,                                   RequireExt<&Extensions::textureFloatLinear>,     NeverSupported,                                 NeverSupported);

    // Unsized luminance alpha formats
    //                 | Internal format   |sized | L | A | Format            | Type             | Component type        | Texture supported                        | Filterable                                     | Texture attachment | Renderbuffer |
    AddLUMAFormat(&map, GL_ALPHA,           false,  0,  8, GL_ALPHA,           GL_UNSIGNED_BYTE,  GL_UNSIGNED_NORMALIZED, AlwaysSupported,                           AlwaysSupported,                                 NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_LUMINANCE,       false,  8,  0, GL_LUMINANCE,       GL_UNSIGNED_BYTE,  GL_UNSIGNED_NORMALIZED, AlwaysSupported,                           AlwaysSupported,                                 NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_LUMINANCE_ALPHA, false,  8,  8, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE,  GL_UNSIGNED_NORMALIZED, AlwaysSupported,                           AlwaysSupported,                                 NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_ALPHA,           false,  0, 16, GL_ALPHA,           GL_HALF_FLOAT_OES, GL_FLOAT,               RequireExt<&Extensions::textureHalfFloat>, RequireExt<&Extensions::textureHalfFloatLinear>, NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_LUMINANCE,       false, 16,  0, GL_LUMINANCE,       GL_HALF_FLOAT_OES, GL_FLOAT,               RequireExt<&Extensions::textureHalfFloat>, RequireExt<&Extensions::textureHalfFloatLinear>, NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_LUMINANCE_ALPHA, false, 16, 16, GL_LUMINANCE_ALPHA, GL_HALF_FLOAT_OES, GL_FLOAT,               RequireExt<&Extensions::textureHalfFloat>, RequireExt<&Extensions::textureHalfFloatLinear>, NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_ALPHA,           false,  0, 32, GL_ALPHA,           GL_FLOAT,          GL_FLOAT,               RequireExt<&Extensions::textureFloat>,     RequireExt<&Extensions::textureFloatLinear>,     NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_LUMINANCE,       false, 32,  0, GL_LUMINANCE,       GL_FLOAT,          GL_FLOAT,               RequireExt<&Extensions::textureFloat>,     RequireExt<&Extensions::textureFloatLinear>,     NeverSupported,      NeverSupported);
    AddLUMAFormat(&map, GL_LUMINANCE_ALPHA, false, 32, 32, GL_LUMINANCE_ALPHA, GL_FLOAT,          GL_FLOAT,               RequireExt<&Extensions::textureFloat>,     RequireExt<&Extensions::textureFloatLinear>,     NeverSupported,      NeverSupported);

    // Unsized depth stencil formats
    //                         | Internal format   |sized | D |S | X | Format            | Type                             | Component type        | Texture supported                                    | Filterable     | Texture attachment                                   | Renderbuffer                                        |
    AddDepthStencilFormat(&map, GL_DEPTH_COMPONENT, false, 16, 0,  0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT,                 GL_UNSIGNED_NORMALIZED, RequireES<1, 0>,                                       AlwaysSupported, RequireES<1, 0>,                                       RequireES<1, 0>                                      );
    AddDepthStencilFormat(&map, GL_DEPTH_COMPONENT, false, 24, 0,  0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT,                   GL_UNSIGNED_NORMALIZED, RequireES<1, 0>,                                       AlwaysSupported, RequireES<1, 0>,                                       RequireES<1, 0>                                      );
    AddDepthStencilFormat(&map, GL_DEPTH_COMPONENT, false, 32, 0,  0, GL_DEPTH_COMPONENT, GL_FLOAT,                          GL_FLOAT,               RequireES<1, 0>,                                       AlwaysSupported, RequireES<1, 0>,                                       RequireES<1, 0>                                      );
    AddDepthStencilFormat(&map, GL_DEPTH_STENCIL,   false, 24, 8,  0, GL_DEPTH_STENCIL,   GL_UNSIGNED_INT_24_8,              GL_UNSIGNED_NORMALIZED, RequireESOrExt<3, 0, &Extensions::packedDepthStencil>, AlwaysSupported, RequireESOrExt<3, 0, &Extensions::packedDepthStencil>, RequireESOrExt<3, 0, &Extensions::packedDepthStencil>);
    AddDepthStencilFormat(&map, GL_DEPTH_STENCIL,   false, 32, 8, 24, GL_DEPTH_STENCIL,   GL_FLOAT_32_UNSIGNED_INT_24_8_REV, GL_FLOAT,               RequireESOrExt<3, 0, &Extensions::packedDepthStencil>, AlwaysSupported, RequireESOrExt<3, 0, &Extensions::packedDepthStencil>, RequireESOrExt<3, 0, &Extensions::packedDepthStencil>);
    AddDepthStencilFormat(&map, GL_STENCIL,         false,  0, 8,  0, GL_STENCIL,         GL_UNSIGNED_BYTE,                  GL_UNSIGNED_NORMALIZED, RequireES<1, 0>,                                       NeverSupported , RequireES<1, 0>,                                       RequireES<1, 0>                                      );
    // clang-format on

    return map;
}

static const InternalFormatInfoMap &GetInternalFormatMap()
{
    static const angle::base::NoDestructor<InternalFormatInfoMap> formatMap(
        BuildInternalFormatInfoMap());
    return *formatMap;
}

static FormatSet BuildAllSizedInternalFormatSet()
{
    FormatSet result;

    for (const auto &internalFormat : GetInternalFormatMap())
    {
        for (const auto &type : internalFormat.second)
        {
            if (type.second.sized)
            {
                // TODO(jmadill): Fix this hack.
                if (internalFormat.first == GL_BGR565_ANGLEX)
                    continue;

                result.insert(internalFormat.first);
            }
        }
    }

    return result;
}

uint32_t GetPackedTypeInfo(GLenum type)
{
    switch (type)
    {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        {
            static constexpr uint32_t kPacked = PackTypeInfo(1, false);
            return kPacked;
        }
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_HALF_FLOAT:
        case GL_HALF_FLOAT_OES:
        {
            static constexpr uint32_t kPacked = PackTypeInfo(2, false);
            return kPacked;
        }
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        {
            static constexpr uint32_t kPacked = PackTypeInfo(4, false);
            return kPacked;
        }
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV_EXT:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV_EXT:
        {
            static constexpr uint32_t kPacked = PackTypeInfo(2, true);
            return kPacked;
        }
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_24_8:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        {
            ASSERT(GL_UNSIGNED_INT_24_8_OES == GL_UNSIGNED_INT_24_8);
            static constexpr uint32_t kPacked = PackTypeInfo(4, true);
            return kPacked;
        }
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
        {
            static constexpr uint32_t kPacked = PackTypeInfo(8, true);
            return kPacked;
        }
        default:
        {
            return 0;
        }
    }
}

const InternalFormat &GetSizedInternalFormatInfo(GLenum internalFormat)
{
    static const InternalFormat defaultInternalFormat;
    const InternalFormatInfoMap &formatMap = GetInternalFormatMap();
    auto iter                              = formatMap.find(internalFormat);

    // Sized internal formats only have one type per entry
    if (iter == formatMap.end() || iter->second.size() != 1)
    {
        return defaultInternalFormat;
    }

    const InternalFormat &internalFormatInfo = iter->second.begin()->second;
    if (!internalFormatInfo.sized)
    {
        return defaultInternalFormat;
    }

    return internalFormatInfo;
}

const InternalFormat &GetInternalFormatInfo(GLenum internalFormat, GLenum type)
{
    static const InternalFormat defaultInternalFormat;
    const InternalFormatInfoMap &formatMap = GetInternalFormatMap();

    auto internalFormatIter = formatMap.find(internalFormat);
    if (internalFormatIter == formatMap.end())
    {
        return defaultInternalFormat;
    }

    // If the internal format is sized, simply return it without the type check.
    if (internalFormatIter->second.size() == 1 && internalFormatIter->second.begin()->second.sized)
    {
        return internalFormatIter->second.begin()->second;
    }

    auto typeIter = internalFormatIter->second.find(type);
    if (typeIter == internalFormatIter->second.end())
    {
        return defaultInternalFormat;
    }

    return typeIter->second;
}

GLuint InternalFormat::computePixelBytes(GLenum formatType) const
{
    const auto &typeInfo = GetTypeInfo(formatType);
    GLuint components    = typeInfo.specialInterpretation ? 1u : componentCount;
    return components * typeInfo.bytes;
}

bool InternalFormat::computeRowPitch(GLenum formatType,
                                     GLsizei width,
                                     GLint alignment,
                                     GLint rowLength,
                                     GLuint *resultOut) const
{
    // Compressed images do not use pack/unpack parameters.
    if (compressed)
    {
        ASSERT(rowLength == 0);
        return computeCompressedImageSize(Extents(width, 1, 1), resultOut);
    }

    CheckedNumeric<GLuint> checkedWidth(rowLength > 0 ? rowLength : width);
    CheckedNumeric<GLuint> checkedRowBytes = checkedWidth * computePixelBytes(formatType);

    ASSERT(alignment > 0 && isPow2(alignment));
    CheckedNumeric<GLuint> checkedAlignment(alignment);
    auto aligned = rx::roundUp(checkedRowBytes, checkedAlignment);
    return CheckedMathResult(aligned, resultOut);
}

bool InternalFormat::computeDepthPitch(GLsizei height,
                                       GLint imageHeight,
                                       GLuint rowPitch,
                                       GLuint *resultOut) const
{
    CheckedNumeric<GLuint> pixelsHeight(imageHeight > 0 ? static_cast<GLuint>(imageHeight)
                                                        : static_cast<GLuint>(height));

    CheckedNumeric<GLuint> rowCount;
    if (compressed)
    {
        CheckedNumeric<GLuint> checkedBlockHeight(compressedBlockHeight);
        rowCount = (pixelsHeight + checkedBlockHeight - 1u) / checkedBlockHeight;
    }
    else
    {
        rowCount = pixelsHeight;
    }

    CheckedNumeric<GLuint> checkedRowPitch(rowPitch);

    return CheckedMathResult(checkedRowPitch * rowCount, resultOut);
}

bool InternalFormat::computeDepthPitch(GLenum formatType,
                                       GLsizei width,
                                       GLsizei height,
                                       GLint alignment,
                                       GLint rowLength,
                                       GLint imageHeight,
                                       GLuint *resultOut) const
{
    GLuint rowPitch = 0;
    if (!computeRowPitch(formatType, width, alignment, rowLength, &rowPitch))
    {
        return false;
    }
    return computeDepthPitch(height, imageHeight, rowPitch, resultOut);
}

bool InternalFormat::computeCompressedImageSize(const Extents &size, GLuint *resultOut) const
{
    CheckedNumeric<GLuint> checkedWidth(size.width);
    CheckedNumeric<GLuint> checkedHeight(size.height);
    CheckedNumeric<GLuint> checkedDepth(size.depth);
    CheckedNumeric<GLuint> checkedBlockWidth(compressedBlockWidth);
    CheckedNumeric<GLuint> checkedBlockHeight(compressedBlockHeight);

    ASSERT(compressed);
    auto numBlocksWide = (checkedWidth + checkedBlockWidth - 1u) / checkedBlockWidth;
    auto numBlocksHigh = (checkedHeight + checkedBlockHeight - 1u) / checkedBlockHeight;
    auto bytes         = numBlocksWide * numBlocksHigh * pixelBytes * checkedDepth;
    return CheckedMathResult(bytes, resultOut);
}

bool InternalFormat::computeSkipBytes(GLenum formatType,
                                      GLuint rowPitch,
                                      GLuint depthPitch,
                                      const PixelStoreStateBase &state,
                                      bool is3D,
                                      GLuint *resultOut) const
{
    CheckedNumeric<GLuint> checkedRowPitch(rowPitch);
    CheckedNumeric<GLuint> checkedDepthPitch(depthPitch);
    CheckedNumeric<GLuint> checkedSkipImages(static_cast<GLuint>(state.skipImages));
    CheckedNumeric<GLuint> checkedSkipRows(static_cast<GLuint>(state.skipRows));
    CheckedNumeric<GLuint> checkedSkipPixels(static_cast<GLuint>(state.skipPixels));
    CheckedNumeric<GLuint> checkedPixelBytes(computePixelBytes(formatType));
    auto checkedSkipImagesBytes = checkedSkipImages * checkedDepthPitch;
    if (!is3D)
    {
        checkedSkipImagesBytes = 0;
    }
    auto skipBytes = checkedSkipImagesBytes + checkedSkipRows * checkedRowPitch +
                     checkedSkipPixels * checkedPixelBytes;
    return CheckedMathResult(skipBytes, resultOut);
}

bool InternalFormat::computePackUnpackEndByte(GLenum formatType,
                                              const Extents &size,
                                              const PixelStoreStateBase &state,
                                              bool is3D,
                                              GLuint *resultOut) const
{
    GLuint rowPitch = 0;
    if (!computeRowPitch(formatType, size.width, state.alignment, state.rowLength, &rowPitch))
    {
        return false;
    }

    GLuint depthPitch = 0;
    if (is3D && !computeDepthPitch(size.height, state.imageHeight, rowPitch, &depthPitch))
    {
        return false;
    }

    CheckedNumeric<GLuint> checkedCopyBytes(0);
    if (compressed)
    {
        GLuint copyBytes = 0;
        if (!computeCompressedImageSize(size, &copyBytes))
        {
            return false;
        }
        checkedCopyBytes = copyBytes;
    }
    else if (size.height != 0 && (!is3D || size.depth != 0))
    {
        CheckedNumeric<GLuint> bytes = computePixelBytes(formatType);
        checkedCopyBytes += size.width * bytes;

        CheckedNumeric<GLuint> heightMinusOne = size.height - 1;
        checkedCopyBytes += heightMinusOne * rowPitch;

        if (is3D)
        {
            CheckedNumeric<GLuint> depthMinusOne = size.depth - 1;
            checkedCopyBytes += depthMinusOne * depthPitch;
        }
    }

    GLuint skipBytes = 0;
    if (!computeSkipBytes(formatType, rowPitch, depthPitch, state, is3D, &skipBytes))
    {
        return false;
    }

    CheckedNumeric<GLuint> endByte = checkedCopyBytes + CheckedNumeric<GLuint>(skipBytes);

    return CheckedMathResult(endByte, resultOut);
}

GLenum GetUnsizedFormat(GLenum internalFormat)
{
    auto sizedFormatInfo = GetSizedInternalFormatInfo(internalFormat);
    if (sizedFormatInfo.internalFormat != GL_NONE)
    {
        return sizedFormatInfo.format;
    }

    return internalFormat;
}

bool CompressedFormatRequiresWholeImage(GLenum internalFormat)
{
    // List of compressed texture format that require that the sub-image size is equal to texture's
    // respective mip level's size
    switch (internalFormat)
    {
        case GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG:
        case GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG:
        case GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG:
        case GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG:
        case GL_COMPRESSED_SRGB_PVRTC_2BPPV1_EXT:
        case GL_COMPRESSED_SRGB_PVRTC_4BPPV1_EXT:
        case GL_COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT:
        case GL_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT:
            return true;

        default:
            return false;
    }
}

const FormatSet &GetAllSizedInternalFormats()
{
    static angle::base::NoDestructor<FormatSet> formatSet(BuildAllSizedInternalFormatSet());
    return *formatSet;
}

AttributeType GetAttributeType(GLenum enumValue)
{
    switch (enumValue)
    {
        case GL_FLOAT:
            return ATTRIBUTE_FLOAT;
        case GL_FLOAT_VEC2:
            return ATTRIBUTE_VEC2;
        case GL_FLOAT_VEC3:
            return ATTRIBUTE_VEC3;
        case GL_FLOAT_VEC4:
            return ATTRIBUTE_VEC4;
        case GL_INT:
            return ATTRIBUTE_INT;
        case GL_INT_VEC2:
            return ATTRIBUTE_IVEC2;
        case GL_INT_VEC3:
            return ATTRIBUTE_IVEC3;
        case GL_INT_VEC4:
            return ATTRIBUTE_IVEC4;
        case GL_UNSIGNED_INT:
            return ATTRIBUTE_UINT;
        case GL_UNSIGNED_INT_VEC2:
            return ATTRIBUTE_UVEC2;
        case GL_UNSIGNED_INT_VEC3:
            return ATTRIBUTE_UVEC3;
        case GL_UNSIGNED_INT_VEC4:
            return ATTRIBUTE_UVEC4;
        case GL_FLOAT_MAT2:
            return ATTRIBUTE_MAT2;
        case GL_FLOAT_MAT3:
            return ATTRIBUTE_MAT3;
        case GL_FLOAT_MAT4:
            return ATTRIBUTE_MAT4;
        case GL_FLOAT_MAT2x3:
            return ATTRIBUTE_MAT2x3;
        case GL_FLOAT_MAT2x4:
            return ATTRIBUTE_MAT2x4;
        case GL_FLOAT_MAT3x2:
            return ATTRIBUTE_MAT3x2;
        case GL_FLOAT_MAT3x4:
            return ATTRIBUTE_MAT3x4;
        case GL_FLOAT_MAT4x2:
            return ATTRIBUTE_MAT4x2;
        case GL_FLOAT_MAT4x3:
            return ATTRIBUTE_MAT4x3;
        default:
            UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
            return ATTRIBUTE_FLOAT;
#endif
    }
}

angle::FormatID GetVertexFormatID(VertexAttribType type,
                                  GLboolean normalized,
                                  GLuint components,
                                  bool pureInteger)
{
    switch (type)
    {
        case VertexAttribType::Byte:
            switch (components)
            {
                case 1:
                    if (pureInteger)
                        return angle::FormatID::R8_SINT;
                    if (normalized)
                        return angle::FormatID::R8_SNORM;
                    return angle::FormatID::R8_SSCALED;
                case 2:
                    if (pureInteger)
                        return angle::FormatID::R8G8_SINT;
                    if (normalized)
                        return angle::FormatID::R8G8_SNORM;
                    return angle::FormatID::R8G8_SSCALED;
                case 3:
                    if (pureInteger)
                        return angle::FormatID::R8G8B8_SINT;
                    if (normalized)
                        return angle::FormatID::R8G8B8_SNORM;
                    return angle::FormatID::R8G8B8_SSCALED;
                case 4:
                    if (pureInteger)
                        return angle::FormatID::R8G8B8A8_SINT;
                    if (normalized)
                        return angle::FormatID::R8G8B8A8_SNORM;
                    return angle::FormatID::R8G8B8A8_SSCALED;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return angle::FormatID::NONE;
#endif
            }
        case VertexAttribType::UnsignedByte:
            switch (components)
            {
                case 1:
                    if (pureInteger)
                        return angle::FormatID::R8_UINT;
                    if (normalized)
                        return angle::FormatID::R8_UNORM;
                    return angle::FormatID::R8_USCALED;
                case 2:
                    if (pureInteger)
                        return angle::FormatID::R8G8_UINT;
                    if (normalized)
                        return angle::FormatID::R8G8_UNORM;
                    return angle::FormatID::R8G8_USCALED;
                case 3:
                    if (pureInteger)
                        return angle::FormatID::R8G8B8_UINT;
                    if (normalized)
                        return angle::FormatID::R8G8B8_UNORM;
                    return angle::FormatID::R8G8B8_USCALED;
                case 4:
                    if (pureInteger)
                        return angle::FormatID::R8G8B8A8_UINT;
                    if (normalized)
                        return angle::FormatID::R8G8B8A8_UNORM;
                    return angle::FormatID::R8G8B8A8_USCALED;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return angle::FormatID::NONE;
#endif
            }
        case VertexAttribType::Short:
            switch (components)
            {
                case 1:
                    if (pureInteger)
                        return angle::FormatID::R16_SINT;
                    if (normalized)
                        return angle::FormatID::R16_SNORM;
                    return angle::FormatID::R16_SSCALED;
                case 2:
                    if (pureInteger)
                        return angle::FormatID::R16G16_SINT;
                    if (normalized)
                        return angle::FormatID::R16G16_SNORM;
                    return angle::FormatID::R16G16_SSCALED;
                case 3:
                    if (pureInteger)
                        return angle::FormatID::R16G16B16_SINT;
                    if (normalized)
                        return angle::FormatID::R16G16B16_SNORM;
                    return angle::FormatID::R16G16B16_SSCALED;
                case 4:
                    if (pureInteger)
                        return angle::FormatID::R16G16B16A16_SINT;
                    if (normalized)
                        return angle::FormatID::R16G16B16A16_SNORM;
                    return angle::FormatID::R16G16B16A16_SSCALED;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return angle::FormatID::NONE;
#endif
            }
        case VertexAttribType::UnsignedShort:
            switch (components)
            {
                case 1:
                    if (pureInteger)
                        return angle::FormatID::R16_UINT;
                    if (normalized)
                        return angle::FormatID::R16_UNORM;
                    return angle::FormatID::R16_USCALED;
                case 2:
                    if (pureInteger)
                        return angle::FormatID::R16G16_UINT;
                    if (normalized)
                        return angle::FormatID::R16G16_UNORM;
                    return angle::FormatID::R16G16_USCALED;
                case 3:
                    if (pureInteger)
                        return angle::FormatID::R16G16B16_UINT;
                    if (normalized)
                        return angle::FormatID::R16G16B16_UNORM;
                    return angle::FormatID::R16G16B16_USCALED;
                case 4:
                    if (pureInteger)
                        return angle::FormatID::R16G16B16A16_UINT;
                    if (normalized)
                        return angle::FormatID::R16G16B16A16_UNORM;
                    return angle::FormatID::R16G16B16A16_USCALED;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return angle::FormatID::NONE;
#endif
            }
        case VertexAttribType::Int:
            switch (components)
            {
                case 1:
                    if (pureInteger)
                        return angle::FormatID::R32_SINT;
                    if (normalized)
                        return angle::FormatID::R32_SNORM;
                    return angle::FormatID::R32_SSCALED;
                case 2:
                    if (pureInteger)
                        return angle::FormatID::R32G32_SINT;
                    if (normalized)
                        return angle::FormatID::R32G32_SNORM;
                    return angle::FormatID::R32G32_SSCALED;
                case 3:
                    if (pureInteger)
                        return angle::FormatID::R32G32B32_SINT;
                    if (normalized)
                        return angle::FormatID::R32G32B32_SNORM;
                    return angle::FormatID::R32G32B32_SSCALED;
                case 4:
                    if (pureInteger)
                        return angle::FormatID::R32G32B32A32_SINT;
                    if (normalized)
                        return angle::FormatID::R32G32B32A32_SNORM;
                    return angle::FormatID::R32G32B32A32_SSCALED;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return angle::FormatID::NONE;
#endif
            }
        case VertexAttribType::UnsignedInt:
            switch (components)
            {
                case 1:
                    if (pureInteger)
                        return angle::FormatID::R32_UINT;
                    if (normalized)
                        return angle::FormatID::R32_UNORM;
                    return angle::FormatID::R32_USCALED;
                case 2:
                    if (pureInteger)
                        return angle::FormatID::R32G32_UINT;
                    if (normalized)
                        return angle::FormatID::R32G32_UNORM;
                    return angle::FormatID::R32G32_USCALED;
                case 3:
                    if (pureInteger)
                        return angle::FormatID::R32G32B32_UINT;
                    if (normalized)
                        return angle::FormatID::R32G32B32_UNORM;
                    return angle::FormatID::R32G32B32_USCALED;
                case 4:
                    if (pureInteger)
                        return angle::FormatID::R32G32B32A32_UINT;
                    if (normalized)
                        return angle::FormatID::R32G32B32A32_UNORM;
                    return angle::FormatID::R32G32B32A32_USCALED;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return angle::FormatID::NONE;
#endif
            }
        case VertexAttribType::Float:
            switch (components)
            {
                case 1:
                    return angle::FormatID::R32_FLOAT;
                case 2:
                    return angle::FormatID::R32G32_FLOAT;
                case 3:
                    return angle::FormatID::R32G32B32_FLOAT;
                case 4:
                    return angle::FormatID::R32G32B32A32_FLOAT;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return angle::FormatID::NONE;
#endif
            }
        case VertexAttribType::HalfFloat:
        case VertexAttribType::HalfFloatOES:
            switch (components)
            {
                case 1:
                    return angle::FormatID::R16_FLOAT;
                case 2:
                    return angle::FormatID::R16G16_FLOAT;
                case 3:
                    return angle::FormatID::R16G16B16_FLOAT;
                case 4:
                    return angle::FormatID::R16G16B16A16_FLOAT;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return angle::FormatID::NONE;
#endif
            }
        case VertexAttribType::Fixed:
            switch (components)
            {
                case 1:
                    return angle::FormatID::R32_FIXED;
                case 2:
                    return angle::FormatID::R32G32_FIXED;
                case 3:
                    return angle::FormatID::R32G32B32_FIXED;
                case 4:
                    return angle::FormatID::R32G32B32A32_FIXED;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return angle::FormatID::NONE;
#endif
            }
        case VertexAttribType::Int2101010:
            if (pureInteger)
                return angle::FormatID::R10G10B10A2_SINT;
            if (normalized)
                return angle::FormatID::R10G10B10A2_SNORM;
            return angle::FormatID::R10G10B10A2_SSCALED;
        case VertexAttribType::UnsignedInt2101010:
            if (pureInteger)
                return angle::FormatID::R10G10B10A2_UINT;
            if (normalized)
                return angle::FormatID::R10G10B10A2_UNORM;
            return angle::FormatID::R10G10B10A2_USCALED;
        case VertexAttribType::Int1010102:
            switch (components)
            {
                case 3:
                    if (pureInteger)
                        return angle::FormatID::X2R10G10B10_SINT_VERTEX;
                    if (normalized)
                        return angle::FormatID::X2R10G10B10_SNORM_VERTEX;
                    return angle::FormatID::X2R10G10B10_SSCALED_VERTEX;
                case 4:
                    if (pureInteger)
                        return angle::FormatID::A2R10G10B10_SINT_VERTEX;
                    if (normalized)
                        return angle::FormatID::A2R10G10B10_SNORM_VERTEX;
                    return angle::FormatID::A2R10G10B10_SSCALED_VERTEX;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return angle::FormatID::NONE;
#endif
            }
        case VertexAttribType::UnsignedInt1010102:
            switch (components)
            {
                case 3:
                    if (pureInteger)
                        return angle::FormatID::X2R10G10B10_UINT_VERTEX;
                    if (normalized)
                        return angle::FormatID::X2R10G10B10_UNORM_VERTEX;
                    return angle::FormatID::X2R10G10B10_USCALED_VERTEX;

                case 4:
                    if (pureInteger)
                        return angle::FormatID::A2R10G10B10_UINT_VERTEX;
                    if (normalized)
                        return angle::FormatID::A2R10G10B10_UNORM_VERTEX;
                    return angle::FormatID::A2R10G10B10_USCALED_VERTEX;
                default:
                    UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
                    return angle::FormatID::NONE;
#endif
            }
        default:
            UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
            return angle::FormatID::NONE;
#endif
    }
}

angle::FormatID GetVertexFormatID(const VertexAttribute &attrib, VertexAttribType currentValueType)
{
    if (!attrib.enabled)
    {
        return GetCurrentValueFormatID(currentValueType);
    }
    return attrib.format->id;
}

angle::FormatID GetCurrentValueFormatID(VertexAttribType currentValueType)
{
    switch (currentValueType)
    {
        case VertexAttribType::Float:
            return angle::FormatID::R32G32B32A32_FLOAT;
        case VertexAttribType::Int:
            return angle::FormatID::R32G32B32A32_SINT;
        case VertexAttribType::UnsignedInt:
            return angle::FormatID::R32G32B32A32_UINT;
        default:
            UNREACHABLE();
            return angle::FormatID::NONE;
    }
}

const VertexFormat &GetVertexFormatFromID(angle::FormatID vertexFormatID)
{
    switch (vertexFormatID)
    {
        case angle::FormatID::R8_SSCALED:
        {
            static const VertexFormat format(GL_BYTE, GL_FALSE, 1, false);
            return format;
        }
        case angle::FormatID::R8_SNORM:
        {
            static const VertexFormat format(GL_BYTE, GL_TRUE, 1, false);
            return format;
        }
        case angle::FormatID::R8G8_SSCALED:
        {
            static const VertexFormat format(GL_BYTE, GL_FALSE, 2, false);
            return format;
        }
        case angle::FormatID::R8G8_SNORM:
        {
            static const VertexFormat format(GL_BYTE, GL_TRUE, 2, false);
            return format;
        }
        case angle::FormatID::R8G8B8_SSCALED:
        {
            static const VertexFormat format(GL_BYTE, GL_FALSE, 3, false);
            return format;
        }
        case angle::FormatID::R8G8B8_SNORM:
        {
            static const VertexFormat format(GL_BYTE, GL_TRUE, 3, false);
            return format;
        }
        case angle::FormatID::R8G8B8A8_SSCALED:
        {
            static const VertexFormat format(GL_BYTE, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::R8G8B8A8_SNORM:
        {
            static const VertexFormat format(GL_BYTE, GL_TRUE, 4, false);
            return format;
        }
        case angle::FormatID::R8_USCALED:
        {
            static const VertexFormat format(GL_UNSIGNED_BYTE, GL_FALSE, 1, false);
            return format;
        }
        case angle::FormatID::R8_UNORM:
        {
            static const VertexFormat format(GL_UNSIGNED_BYTE, GL_TRUE, 1, false);
            return format;
        }
        case angle::FormatID::R8G8_USCALED:
        {
            static const VertexFormat format(GL_UNSIGNED_BYTE, GL_FALSE, 2, false);
            return format;
        }
        case angle::FormatID::R8G8_UNORM:
        {
            static const VertexFormat format(GL_UNSIGNED_BYTE, GL_TRUE, 2, false);
            return format;
        }
        case angle::FormatID::R8G8B8_USCALED:
        {
            static const VertexFormat format(GL_UNSIGNED_BYTE, GL_FALSE, 3, false);
            return format;
        }
        case angle::FormatID::R8G8B8_UNORM:
        {
            static const VertexFormat format(GL_UNSIGNED_BYTE, GL_TRUE, 3, false);
            return format;
        }
        case angle::FormatID::R8G8B8A8_USCALED:
        {
            static const VertexFormat format(GL_UNSIGNED_BYTE, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::R8G8B8A8_UNORM:
        {
            static const VertexFormat format(GL_UNSIGNED_BYTE, GL_TRUE, 4, false);
            return format;
        }
        case angle::FormatID::R16_SSCALED:
        {
            static const VertexFormat format(GL_SHORT, GL_FALSE, 1, false);
            return format;
        }
        case angle::FormatID::R16_SNORM:
        {
            static const VertexFormat format(GL_SHORT, GL_TRUE, 1, false);
            return format;
        }
        case angle::FormatID::R16G16_SSCALED:
        {
            static const VertexFormat format(GL_SHORT, GL_FALSE, 2, false);
            return format;
        }
        case angle::FormatID::R16G16_SNORM:
        {
            static const VertexFormat format(GL_SHORT, GL_TRUE, 2, false);
            return format;
        }
        case angle::FormatID::R16G16B16_SSCALED:
        {
            static const VertexFormat format(GL_SHORT, GL_FALSE, 3, false);
            return format;
        }
        case angle::FormatID::R16G16B16_SNORM:
        {
            static const VertexFormat format(GL_SHORT, GL_TRUE, 3, false);
            return format;
        }
        case angle::FormatID::R16G16B16A16_SSCALED:
        {
            static const VertexFormat format(GL_SHORT, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::R16G16B16A16_SNORM:
        {
            static const VertexFormat format(GL_SHORT, GL_TRUE, 4, false);
            return format;
        }
        case angle::FormatID::R16_USCALED:
        {
            static const VertexFormat format(GL_UNSIGNED_SHORT, GL_FALSE, 1, false);
            return format;
        }
        case angle::FormatID::R16_UNORM:
        {
            static const VertexFormat format(GL_UNSIGNED_SHORT, GL_TRUE, 1, false);
            return format;
        }
        case angle::FormatID::R16G16_USCALED:
        {
            static const VertexFormat format(GL_UNSIGNED_SHORT, GL_FALSE, 2, false);
            return format;
        }
        case angle::FormatID::R16G16_UNORM:
        {
            static const VertexFormat format(GL_UNSIGNED_SHORT, GL_TRUE, 2, false);
            return format;
        }
        case angle::FormatID::R16G16B16_USCALED:
        {
            static const VertexFormat format(GL_UNSIGNED_SHORT, GL_FALSE, 3, false);
            return format;
        }
        case angle::FormatID::R16G16B16_UNORM:
        {
            static const VertexFormat format(GL_UNSIGNED_SHORT, GL_TRUE, 3, false);
            return format;
        }
        case angle::FormatID::R16G16B16A16_USCALED:
        {
            static const VertexFormat format(GL_UNSIGNED_SHORT, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::R16G16B16A16_UNORM:
        {
            static const VertexFormat format(GL_UNSIGNED_SHORT, GL_TRUE, 4, false);
            return format;
        }
        case angle::FormatID::R32_SSCALED:
        {
            static const VertexFormat format(GL_INT, GL_FALSE, 1, false);
            return format;
        }
        case angle::FormatID::R32_SNORM:
        {
            static const VertexFormat format(GL_INT, GL_TRUE, 1, false);
            return format;
        }
        case angle::FormatID::R32G32_SSCALED:
        {
            static const VertexFormat format(GL_INT, GL_FALSE, 2, false);
            return format;
        }
        case angle::FormatID::R32G32_SNORM:
        {
            static const VertexFormat format(GL_INT, GL_TRUE, 2, false);
            return format;
        }
        case angle::FormatID::R32G32B32_SSCALED:
        {
            static const VertexFormat format(GL_INT, GL_FALSE, 3, false);
            return format;
        }
        case angle::FormatID::R32G32B32_SNORM:
        {
            static const VertexFormat format(GL_INT, GL_TRUE, 3, false);
            return format;
        }
        case angle::FormatID::R32G32B32A32_SSCALED:
        {
            static const VertexFormat format(GL_INT, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::R32G32B32A32_SNORM:
        {
            static const VertexFormat format(GL_INT, GL_TRUE, 4, false);
            return format;
        }
        case angle::FormatID::R32_USCALED:
        {
            static const VertexFormat format(GL_UNSIGNED_INT, GL_FALSE, 1, false);
            return format;
        }
        case angle::FormatID::R32_UNORM:
        {
            static const VertexFormat format(GL_UNSIGNED_INT, GL_TRUE, 1, false);
            return format;
        }
        case angle::FormatID::R32G32_USCALED:
        {
            static const VertexFormat format(GL_UNSIGNED_INT, GL_FALSE, 2, false);
            return format;
        }
        case angle::FormatID::R32G32_UNORM:
        {
            static const VertexFormat format(GL_UNSIGNED_INT, GL_TRUE, 2, false);
            return format;
        }
        case angle::FormatID::R32G32B32_USCALED:
        {
            static const VertexFormat format(GL_UNSIGNED_INT, GL_FALSE, 3, false);
            return format;
        }
        case angle::FormatID::R32G32B32_UNORM:
        {
            static const VertexFormat format(GL_UNSIGNED_INT, GL_TRUE, 3, false);
            return format;
        }
        case angle::FormatID::R32G32B32A32_USCALED:
        {
            static const VertexFormat format(GL_UNSIGNED_INT, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::R32G32B32A32_UNORM:
        {
            static const VertexFormat format(GL_UNSIGNED_INT, GL_TRUE, 4, false);
            return format;
        }
        case angle::FormatID::R8_SINT:
        {
            static const VertexFormat format(GL_BYTE, GL_FALSE, 1, true);
            return format;
        }
        case angle::FormatID::R8G8_SINT:
        {
            static const VertexFormat format(GL_BYTE, GL_FALSE, 2, true);
            return format;
        }
        case angle::FormatID::R8G8B8_SINT:
        {
            static const VertexFormat format(GL_BYTE, GL_FALSE, 3, true);
            return format;
        }
        case angle::FormatID::R8G8B8A8_SINT:
        {
            static const VertexFormat format(GL_BYTE, GL_FALSE, 4, true);
            return format;
        }
        case angle::FormatID::R8_UINT:
        {
            static const VertexFormat format(GL_UNSIGNED_BYTE, GL_FALSE, 1, true);
            return format;
        }
        case angle::FormatID::R8G8_UINT:
        {
            static const VertexFormat format(GL_UNSIGNED_BYTE, GL_FALSE, 2, true);
            return format;
        }
        case angle::FormatID::R8G8B8_UINT:
        {
            static const VertexFormat format(GL_UNSIGNED_BYTE, GL_FALSE, 3, true);
            return format;
        }
        case angle::FormatID::R8G8B8A8_UINT:
        {
            static const VertexFormat format(GL_UNSIGNED_BYTE, GL_FALSE, 4, true);
            return format;
        }
        case angle::FormatID::R16_SINT:
        {
            static const VertexFormat format(GL_SHORT, GL_FALSE, 1, true);
            return format;
        }
        case angle::FormatID::R16G16_SINT:
        {
            static const VertexFormat format(GL_SHORT, GL_FALSE, 2, true);
            return format;
        }
        case angle::FormatID::R16G16B16_SINT:
        {
            static const VertexFormat format(GL_SHORT, GL_FALSE, 3, true);
            return format;
        }
        case angle::FormatID::R16G16B16A16_SINT:
        {
            static const VertexFormat format(GL_SHORT, GL_FALSE, 4, true);
            return format;
        }
        case angle::FormatID::R16_UINT:
        {
            static const VertexFormat format(GL_UNSIGNED_SHORT, GL_FALSE, 1, true);
            return format;
        }
        case angle::FormatID::R16G16_UINT:
        {
            static const VertexFormat format(GL_UNSIGNED_SHORT, GL_FALSE, 2, true);
            return format;
        }
        case angle::FormatID::R16G16B16_UINT:
        {
            static const VertexFormat format(GL_UNSIGNED_SHORT, GL_FALSE, 3, true);
            return format;
        }
        case angle::FormatID::R16G16B16A16_UINT:
        {
            static const VertexFormat format(GL_UNSIGNED_SHORT, GL_FALSE, 4, true);
            return format;
        }
        case angle::FormatID::R32_SINT:
        {
            static const VertexFormat format(GL_INT, GL_FALSE, 1, true);
            return format;
        }
        case angle::FormatID::R32G32_SINT:
        {
            static const VertexFormat format(GL_INT, GL_FALSE, 2, true);
            return format;
        }
        case angle::FormatID::R32G32B32_SINT:
        {
            static const VertexFormat format(GL_INT, GL_FALSE, 3, true);
            return format;
        }
        case angle::FormatID::R32G32B32A32_SINT:
        {
            static const VertexFormat format(GL_INT, GL_FALSE, 4, true);
            return format;
        }
        case angle::FormatID::R32_UINT:
        {
            static const VertexFormat format(GL_UNSIGNED_INT, GL_FALSE, 1, true);
            return format;
        }
        case angle::FormatID::R32G32_UINT:
        {
            static const VertexFormat format(GL_UNSIGNED_INT, GL_FALSE, 2, true);
            return format;
        }
        case angle::FormatID::R32G32B32_UINT:
        {
            static const VertexFormat format(GL_UNSIGNED_INT, GL_FALSE, 3, true);
            return format;
        }
        case angle::FormatID::R32G32B32A32_UINT:
        {
            static const VertexFormat format(GL_UNSIGNED_INT, GL_FALSE, 4, true);
            return format;
        }
        case angle::FormatID::R32_FIXED:
        {
            static const VertexFormat format(GL_FIXED, GL_FALSE, 1, false);
            return format;
        }
        case angle::FormatID::R32G32_FIXED:
        {
            static const VertexFormat format(GL_FIXED, GL_FALSE, 2, false);
            return format;
        }
        case angle::FormatID::R32G32B32_FIXED:
        {
            static const VertexFormat format(GL_FIXED, GL_FALSE, 3, false);
            return format;
        }
        case angle::FormatID::R32G32B32A32_FIXED:
        {
            static const VertexFormat format(GL_FIXED, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::R16_FLOAT:
        {
            static const VertexFormat format(GL_HALF_FLOAT, GL_FALSE, 1, false);
            return format;
        }
        case angle::FormatID::R16G16_FLOAT:
        {
            static const VertexFormat format(GL_HALF_FLOAT, GL_FALSE, 2, false);
            return format;
        }
        case angle::FormatID::R16G16B16_FLOAT:
        {
            static const VertexFormat format(GL_HALF_FLOAT, GL_FALSE, 3, false);
            return format;
        }
        case angle::FormatID::R16G16B16A16_FLOAT:
        {
            static const VertexFormat format(GL_HALF_FLOAT, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::R32_FLOAT:
        {
            static const VertexFormat format(GL_FLOAT, GL_FALSE, 1, false);
            return format;
        }
        case angle::FormatID::R32G32_FLOAT:
        {
            static const VertexFormat format(GL_FLOAT, GL_FALSE, 2, false);
            return format;
        }
        case angle::FormatID::R32G32B32_FLOAT:
        {
            static const VertexFormat format(GL_FLOAT, GL_FALSE, 3, false);
            return format;
        }
        case angle::FormatID::R32G32B32A32_FLOAT:
        {
            static const VertexFormat format(GL_FLOAT, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::R10G10B10A2_SSCALED:
        {
            static const VertexFormat format(GL_INT_2_10_10_10_REV, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::R10G10B10A2_USCALED:
        {
            static const VertexFormat format(GL_UNSIGNED_INT_2_10_10_10_REV, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::R10G10B10A2_SNORM:
        {
            static const VertexFormat format(GL_INT_2_10_10_10_REV, GL_TRUE, 4, false);
            return format;
        }
        case angle::FormatID::R10G10B10A2_UNORM:
        {
            static const VertexFormat format(GL_UNSIGNED_INT_2_10_10_10_REV, GL_TRUE, 4, false);
            return format;
        }
        case angle::FormatID::R10G10B10A2_SINT:
        {
            static const VertexFormat format(GL_INT_2_10_10_10_REV, GL_FALSE, 4, true);
            return format;
        }
        case angle::FormatID::R10G10B10A2_UINT:
        {
            static const VertexFormat format(GL_UNSIGNED_INT_2_10_10_10_REV, GL_FALSE, 4, true);
            return format;
        }
        case angle::FormatID::A2R10G10B10_SSCALED_VERTEX:
        {
            static const VertexFormat format(GL_INT_10_10_10_2_OES, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::A2R10G10B10_USCALED_VERTEX:
        {
            static const VertexFormat format(GL_UNSIGNED_INT_10_10_10_2_OES, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::A2R10G10B10_SNORM_VERTEX:
        {
            static const VertexFormat format(GL_INT_10_10_10_2_OES, GL_TRUE, 4, false);
            return format;
        }
        case angle::FormatID::A2R10G10B10_UNORM_VERTEX:
        {
            static const VertexFormat format(GL_UNSIGNED_INT_10_10_10_2_OES, GL_TRUE, 4, false);
            return format;
        }
        case angle::FormatID::A2R10G10B10_SINT_VERTEX:
        {
            static const VertexFormat format(GL_INT_10_10_10_2_OES, GL_FALSE, 4, true);
            return format;
        }
        case angle::FormatID::A2R10G10B10_UINT_VERTEX:
        {
            static const VertexFormat format(GL_UNSIGNED_INT_10_10_10_2_OES, GL_FALSE, 4, true);
            return format;
        }
        case angle::FormatID::X2R10G10B10_SSCALED_VERTEX:
        {
            static const VertexFormat format(GL_INT_10_10_10_2_OES, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::X2R10G10B10_USCALED_VERTEX:
        {
            static const VertexFormat format(GL_UNSIGNED_INT_10_10_10_2_OES, GL_FALSE, 4, false);
            return format;
        }
        case angle::FormatID::X2R10G10B10_SNORM_VERTEX:
        {
            static const VertexFormat format(GL_INT_10_10_10_2_OES, GL_TRUE, 4, false);
            return format;
        }
        case angle::FormatID::X2R10G10B10_UNORM_VERTEX:
        {
            static const VertexFormat format(GL_UNSIGNED_INT_10_10_10_2_OES, GL_TRUE, 4, false);
            return format;
        }
        case angle::FormatID::X2R10G10B10_SINT_VERTEX:
        {
            static const VertexFormat format(GL_INT_10_10_10_2_OES, GL_FALSE, 4, true);
            return format;
        }
        default:
        {
            static const VertexFormat format(GL_NONE, GL_FALSE, 0, false);
            return format;
        }
    }
}

size_t GetVertexFormatSize(angle::FormatID vertexFormatID)
{
    switch (vertexFormatID)
    {
        case angle::FormatID::R8_SSCALED:
        case angle::FormatID::R8_SNORM:
        case angle::FormatID::R8_USCALED:
        case angle::FormatID::R8_UNORM:
        case angle::FormatID::R8_SINT:
        case angle::FormatID::R8_UINT:
            return 1;

        case angle::FormatID::R8G8_SSCALED:
        case angle::FormatID::R8G8_SNORM:
        case angle::FormatID::R8G8_USCALED:
        case angle::FormatID::R8G8_UNORM:
        case angle::FormatID::R8G8_SINT:
        case angle::FormatID::R8G8_UINT:
        case angle::FormatID::R16_SSCALED:
        case angle::FormatID::R16_SNORM:
        case angle::FormatID::R16_USCALED:
        case angle::FormatID::R16_UNORM:
        case angle::FormatID::R16_SINT:
        case angle::FormatID::R16_UINT:
        case angle::FormatID::R16_FLOAT:
            return 2;

        case angle::FormatID::R8G8B8_SSCALED:
        case angle::FormatID::R8G8B8_SNORM:
        case angle::FormatID::R8G8B8_USCALED:
        case angle::FormatID::R8G8B8_UNORM:
        case angle::FormatID::R8G8B8_SINT:
        case angle::FormatID::R8G8B8_UINT:
            return 3;

        case angle::FormatID::R8G8B8A8_SSCALED:
        case angle::FormatID::R8G8B8A8_SNORM:
        case angle::FormatID::R8G8B8A8_USCALED:
        case angle::FormatID::R8G8B8A8_UNORM:
        case angle::FormatID::R8G8B8A8_SINT:
        case angle::FormatID::R8G8B8A8_UINT:
        case angle::FormatID::R16G16_SSCALED:
        case angle::FormatID::R16G16_SNORM:
        case angle::FormatID::R16G16_USCALED:
        case angle::FormatID::R16G16_UNORM:
        case angle::FormatID::R16G16_SINT:
        case angle::FormatID::R16G16_UINT:
        case angle::FormatID::R32_SSCALED:
        case angle::FormatID::R32_SNORM:
        case angle::FormatID::R32_USCALED:
        case angle::FormatID::R32_UNORM:
        case angle::FormatID::R32_SINT:
        case angle::FormatID::R32_UINT:
        case angle::FormatID::R16G16_FLOAT:
        case angle::FormatID::R32_FIXED:
        case angle::FormatID::R32_FLOAT:
        case angle::FormatID::R10G10B10A2_SSCALED:
        case angle::FormatID::R10G10B10A2_USCALED:
        case angle::FormatID::R10G10B10A2_SNORM:
        case angle::FormatID::R10G10B10A2_UNORM:
        case angle::FormatID::R10G10B10A2_SINT:
        case angle::FormatID::R10G10B10A2_UINT:
        case angle::FormatID::A2R10G10B10_SSCALED_VERTEX:
        case angle::FormatID::A2R10G10B10_USCALED_VERTEX:
        case angle::FormatID::A2R10G10B10_SINT_VERTEX:
        case angle::FormatID::A2R10G10B10_UINT_VERTEX:
        case angle::FormatID::A2R10G10B10_SNORM_VERTEX:
        case angle::FormatID::A2R10G10B10_UNORM_VERTEX:
        case angle::FormatID::X2R10G10B10_SSCALED_VERTEX:
        case angle::FormatID::X2R10G10B10_USCALED_VERTEX:
        case angle::FormatID::X2R10G10B10_SINT_VERTEX:
        case angle::FormatID::X2R10G10B10_UINT_VERTEX:
        case angle::FormatID::X2R10G10B10_SNORM_VERTEX:
        case angle::FormatID::X2R10G10B10_UNORM_VERTEX:
            return 4;

        case angle::FormatID::R16G16B16_SSCALED:
        case angle::FormatID::R16G16B16_SNORM:
        case angle::FormatID::R16G16B16_USCALED:
        case angle::FormatID::R16G16B16_UNORM:
        case angle::FormatID::R16G16B16_SINT:
        case angle::FormatID::R16G16B16_UINT:
        case angle::FormatID::R16G16B16_FLOAT:
            return 6;

        case angle::FormatID::R16G16B16A16_SSCALED:
        case angle::FormatID::R16G16B16A16_SNORM:
        case angle::FormatID::R16G16B16A16_USCALED:
        case angle::FormatID::R16G16B16A16_UNORM:
        case angle::FormatID::R16G16B16A16_SINT:
        case angle::FormatID::R16G16B16A16_UINT:
        case angle::FormatID::R32G32_SSCALED:
        case angle::FormatID::R32G32_SNORM:
        case angle::FormatID::R32G32_USCALED:
        case angle::FormatID::R32G32_UNORM:
        case angle::FormatID::R32G32_SINT:
        case angle::FormatID::R32G32_UINT:
        case angle::FormatID::R16G16B16A16_FLOAT:
        case angle::FormatID::R32G32_FIXED:
        case angle::FormatID::R32G32_FLOAT:
            return 8;

        case angle::FormatID::R32G32B32_SSCALED:
        case angle::FormatID::R32G32B32_SNORM:
        case angle::FormatID::R32G32B32_USCALED:
        case angle::FormatID::R32G32B32_UNORM:
        case angle::FormatID::R32G32B32_SINT:
        case angle::FormatID::R32G32B32_UINT:
        case angle::FormatID::R32G32B32_FIXED:
        case angle::FormatID::R32G32B32_FLOAT:
            return 12;

        case angle::FormatID::R32G32B32A32_SSCALED:
        case angle::FormatID::R32G32B32A32_SNORM:
        case angle::FormatID::R32G32B32A32_USCALED:
        case angle::FormatID::R32G32B32A32_UNORM:
        case angle::FormatID::R32G32B32A32_SINT:
        case angle::FormatID::R32G32B32A32_UINT:
        case angle::FormatID::R32G32B32A32_FIXED:
        case angle::FormatID::R32G32B32A32_FLOAT:
            return 16;

        case angle::FormatID::NONE:
        default:
            UNREACHABLE();
#if !UNREACHABLE_IS_NORETURN
            return 0;
#endif
    }
}

bool ValidES3InternalFormat(GLenum internalFormat)
{
    const InternalFormatInfoMap &formatMap = GetInternalFormatMap();
    return internalFormat != GL_NONE && formatMap.find(internalFormat) != formatMap.end();
}

VertexFormat::VertexFormat(GLenum typeIn,
                           GLboolean normalizedIn,
                           GLuint componentsIn,
                           bool pureIntegerIn)
    : type(typeIn), normalized(normalizedIn), components(componentsIn), pureInteger(pureIntegerIn)
{
    // float -> !normalized
    ASSERT(!(type == GL_FLOAT || type == GL_HALF_FLOAT || type == GL_FIXED) ||
           normalized == GL_FALSE);
}
}  // namespace gl
