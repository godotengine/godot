//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// renderergl_utils.cpp: Conversion functions and other utility routines
// specific to the OpenGL renderer.

#include "libANGLE/renderer/gl/renderergl_utils.h"

#include <limits>

#include "common/mathutil.h"
#include "common/platform.h"
#include "libANGLE/Buffer.h"
#include "libANGLE/Caps.h"
#include "libANGLE/Context.h"
#include "libANGLE/formatutils.h"
#include "libANGLE/queryconversions.h"
#include "libANGLE/renderer/gl/ContextGL.h"
#include "libANGLE/renderer/gl/FenceNVGL.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"
#include "libANGLE/renderer/gl/QueryGL.h"
#include "libANGLE/renderer/gl/formatutilsgl.h"
#include "platform/FeaturesGL.h"
#include "platform/FrontendFeatures.h"

#include <EGL/eglext.h>
#include <algorithm>
#include <sstream>

using angle::CheckedNumeric;

namespace rx
{

VendorID GetVendorID(const FunctionsGL *functions)
{
    std::string nativeVendorString(reinterpret_cast<const char *>(functions->getString(GL_VENDOR)));
    if (nativeVendorString.find("Intel") != std::string::npos)
    {
        return VENDOR_ID_INTEL;
    }
    else if (nativeVendorString.find("NVIDIA") != std::string::npos)
    {
        return VENDOR_ID_NVIDIA;
    }
    else if (nativeVendorString.find("ATI") != std::string::npos ||
             nativeVendorString.find("AMD") != std::string::npos)
    {
        return VENDOR_ID_AMD;
    }
    else if (nativeVendorString.find("Qualcomm") != std::string::npos)
    {
        return VENDOR_ID_QUALCOMM;
    }
    else
    {
        return VENDOR_ID_UNKNOWN;
    }
}

uint32_t GetDeviceID(const FunctionsGL *functions)
{
    std::string nativeRendererString(
        reinterpret_cast<const char *>(functions->getString(GL_RENDERER)));
    constexpr std::pair<const char *, uint32_t> kKnownDeviceIDs[] = {
        {"Adreno (TM) 418", ANDROID_DEVICE_ID_NEXUS5X},
        {"Adreno (TM) 530", ANDROID_DEVICE_ID_PIXEL1XL},
        {"Adreno (TM) 540", ANDROID_DEVICE_ID_PIXEL2},
    };

    for (const auto &knownDeviceID : kKnownDeviceIDs)
    {
        if (nativeRendererString.find(knownDeviceID.first) != std::string::npos)
        {
            return knownDeviceID.second;
        }
    }

    return 0;
}

namespace nativegl_gl
{

static bool MeetsRequirements(const FunctionsGL *functions,
                              const nativegl::SupportRequirement &requirements)
{
    bool hasRequiredExtensions = false;
    for (const std::vector<std::string> &exts : requirements.requiredExtensions)
    {
        bool hasAllExtensionsInSet = true;
        for (const std::string &extension : exts)
        {
            if (!functions->hasExtension(extension))
            {
                hasAllExtensionsInSet = false;
                break;
            }
        }
        if (hasAllExtensionsInSet)
        {
            hasRequiredExtensions = true;
            break;
        }
    }
    if (!requirements.requiredExtensions.empty() && !hasRequiredExtensions)
    {
        return false;
    }

    if (functions->version >= requirements.version)
    {
        return true;
    }
    else if (!requirements.versionExtensions.empty())
    {
        for (const std::string &extension : requirements.versionExtensions)
        {
            if (!functions->hasExtension(extension))
            {
                return false;
            }
        }
        return true;
    }
    else
    {
        return false;
    }
}

static bool CheckSizedInternalFormatTextureRenderability(const FunctionsGL *functions,
                                                         const angle::FeaturesGL &features,
                                                         GLenum internalFormat)
{
    const gl::InternalFormat &formatInfo = gl::GetSizedInternalFormatInfo(internalFormat);
    ASSERT(formatInfo.sized);

    // Query the current texture so it can be rebound afterwards
    GLint oldTextureBinding = 0;
    functions->getIntegerv(GL_TEXTURE_BINDING_2D, &oldTextureBinding);

    // Create a small texture with the same format and type that gl::Texture would use
    GLuint texture = 0;
    functions->genTextures(1, &texture);
    functions->bindTexture(GL_TEXTURE_2D, texture);

    // Nearest filter needed for framebuffer completeness on some drivers.
    functions->texParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    nativegl::TexImageFormat texImageFormat = nativegl::GetTexImageFormat(
        functions, features, formatInfo.internalFormat, formatInfo.format, formatInfo.type);
    constexpr GLsizei kTextureSize = 16;
    functions->texImage2D(GL_TEXTURE_2D, 0, texImageFormat.internalFormat, kTextureSize,
                          kTextureSize, 0, texImageFormat.format, texImageFormat.type, nullptr);

    // Query the current framebuffer so it can be rebound afterwards
    GLint oldFramebufferBinding = 0;
    functions->getIntegerv(GL_FRAMEBUFFER_BINDING, &oldFramebufferBinding);

    // Bind the texture to the framebuffer and check renderability
    GLuint fbo = 0;
    functions->genFramebuffers(1, &fbo);
    functions->bindFramebuffer(GL_FRAMEBUFFER, fbo);
    functions->framebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture,
                                    0);

    bool supported = functions->checkFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;

    // Delete the framebuffer and restore the previous binding
    functions->deleteFramebuffers(1, &fbo);
    functions->bindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(oldFramebufferBinding));

    // Delete the texture and restore the previous binding
    functions->deleteTextures(1, &texture);
    functions->bindTexture(GL_TEXTURE_2D, static_cast<GLuint>(oldTextureBinding));

    // Clear error
    if (!supported)
    {
        (void)functions->getError();
    }

    return supported;
}

static bool CheckInternalFormatRenderbufferRenderability(const FunctionsGL *functions,
                                                         const angle::FeaturesGL &features,
                                                         GLenum internalFormat)
{
    const gl::InternalFormat &formatInfo = gl::GetSizedInternalFormatInfo(internalFormat);
    ASSERT(formatInfo.sized);

    // Query the current renderbuffer so it can be rebound afterwards
    GLint oldRenderbufferBinding = 0;
    functions->getIntegerv(GL_RENDERBUFFER_BINDING, &oldRenderbufferBinding);

    // Create a small renderbuffer with the same format and type that gl::Renderbuffer would use
    GLuint renderbuffer = 0;
    functions->genRenderbuffers(1, &renderbuffer);
    functions->bindRenderbuffer(GL_RENDERBUFFER, renderbuffer);

    nativegl::RenderbufferFormat renderbufferFormat =
        nativegl::GetRenderbufferFormat(functions, features, formatInfo.internalFormat);
    constexpr GLsizei kRenderbufferSize = 16;
    functions->renderbufferStorage(GL_RENDERBUFFER, renderbufferFormat.internalFormat,
                                   kRenderbufferSize, kRenderbufferSize);

    // Query the current framebuffer so it can be rebound afterwards
    GLint oldFramebufferBinding = 0;
    functions->getIntegerv(GL_FRAMEBUFFER_BINDING, &oldFramebufferBinding);

    // Bind the texture to the framebuffer and check renderability
    GLuint fbo = 0;
    functions->genFramebuffers(1, &fbo);
    functions->bindFramebuffer(GL_FRAMEBUFFER, fbo);
    functions->framebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
                                       renderbuffer);

    bool supported = functions->checkFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;

    // Delete the framebuffer and restore the previous binding
    functions->deleteFramebuffers(1, &fbo);
    functions->bindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(oldFramebufferBinding));

    // Delete the renderbuffer and restore the previous binding
    functions->deleteRenderbuffers(1, &renderbuffer);
    functions->bindRenderbuffer(GL_RENDERBUFFER, static_cast<GLuint>(oldRenderbufferBinding));

    return supported;
}

static gl::TextureCaps GenerateTextureFormatCaps(const FunctionsGL *functions,
                                                 const angle::FeaturesGL &features,
                                                 GLenum internalFormat)
{
    ASSERT(functions->getError() == GL_NO_ERROR);

    gl::TextureCaps textureCaps;

    const nativegl::InternalFormat &formatInfo =
        nativegl::GetInternalFormatInfo(internalFormat, functions->standard);
    textureCaps.texturable = MeetsRequirements(functions, formatInfo.texture);
    textureCaps.filterable =
        textureCaps.texturable && MeetsRequirements(functions, formatInfo.filter);
    textureCaps.textureAttachment = MeetsRequirements(functions, formatInfo.textureAttachment);
    textureCaps.renderbuffer      = MeetsRequirements(functions, formatInfo.renderbuffer);

    // Do extra renderability validation for some formats.
    // We require GL_RGBA16F is renderable to expose EXT_color_buffer_half_float but we can't know
    // if the format is supported unless we try to create a framebuffer.
    if (internalFormat == GL_RGBA16F)
    {
        if (textureCaps.textureAttachment)
        {
            textureCaps.textureAttachment =
                CheckSizedInternalFormatTextureRenderability(functions, features, internalFormat);
        }
        if (textureCaps.renderbuffer)
        {
            textureCaps.renderbuffer =
                CheckInternalFormatRenderbufferRenderability(functions, features, internalFormat);
        }
    }

    // glGetInternalformativ is not available until version 4.2 but may be available through the 3.0
    // extension GL_ARB_internalformat_query
    if (textureCaps.renderbuffer && functions->getInternalformativ)
    {
        GLenum queryInternalFormat = internalFormat;

        if (internalFormat == GL_BGRA8_EXT)
        {
            // Querying GL_NUM_SAMPLE_COUNTS for GL_BGRA8_EXT generates an INVALID_ENUM on some
            // drivers.  It seems however that allocating a multisampled renderbuffer of this format
            // succeeds. To avoid breaking multisampling for this format, query the supported sample
            // counts for GL_RGBA8 instead.
            queryInternalFormat = GL_RGBA8;
        }

        GLint numSamples = 0;
        functions->getInternalformativ(GL_RENDERBUFFER, queryInternalFormat, GL_NUM_SAMPLE_COUNTS,
                                       1, &numSamples);

        if (numSamples > 0)
        {
            std::vector<GLint> samples(numSamples);
            functions->getInternalformativ(GL_RENDERBUFFER, queryInternalFormat, GL_SAMPLES,
                                           static_cast<GLsizei>(samples.size()), &samples[0]);

            if (internalFormat == GL_STENCIL_INDEX8)
            {
                // The query below does generates an error with STENCIL_INDEX8 on NVIDIA driver
                // 382.33. So for now we assume that the same sampling modes are conformant for
                // STENCIL_INDEX8 as for DEPTH24_STENCIL8. Clean this up once the driver is fixed.
                // http://anglebug.com/2059
                queryInternalFormat = GL_DEPTH24_STENCIL8;
            }
            for (size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++)
            {
                if (features.limitMaxMSAASamplesTo4.enabled && samples[sampleIndex] > 4)
                {
                    continue;
                }

                // Some NVIDIA drivers expose multisampling modes implemented as a combination of
                // multisampling and supersampling. These are non-conformant and should not be
                // exposed through ANGLE. Query which formats are conformant from the driver if
                // supported.
                GLint conformant = GL_TRUE;
                if (functions->getInternalformatSampleivNV)
                {
                    ASSERT(functions->getError() == GL_NO_ERROR);
                    functions->getInternalformatSampleivNV(GL_RENDERBUFFER, queryInternalFormat,
                                                           samples[sampleIndex], GL_CONFORMANT_NV,
                                                           1, &conformant);
                    // getInternalFormatSampleivNV does not work for all formats on NVIDIA Shield TV
                    // drivers. Assume that formats with large sample counts are non-conformant in
                    // case the query generates an error.
                    if (functions->getError() != GL_NO_ERROR)
                    {
                        conformant = (samples[sampleIndex] <= 8) ? GL_TRUE : GL_FALSE;
                    }
                }
                if (conformant == GL_TRUE)
                {
                    textureCaps.sampleCounts.insert(samples[sampleIndex]);
                }
            }
        }
    }

    ASSERT(functions->getError() == GL_NO_ERROR);
    return textureCaps;
}

static GLint QuerySingleGLInt(const FunctionsGL *functions, GLenum name)
{
    GLint result = 0;
    functions->getIntegerv(name, &result);
    return result;
}

static GLint QuerySingleIndexGLInt(const FunctionsGL *functions, GLenum name, GLuint index)
{
    GLint result;
    functions->getIntegeri_v(name, index, &result);
    return result;
}

static GLint QueryGLIntRange(const FunctionsGL *functions, GLenum name, size_t index)
{
    GLint result[2] = {};
    functions->getIntegerv(name, result);
    return result[index];
}

static GLint64 QuerySingleGLInt64(const FunctionsGL *functions, GLenum name)
{
    // Fall back to 32-bit int if 64-bit query is not available. This can become relevant for some
    // caps that are defined as 64-bit values in core spec, but were introduced earlier in
    // extensions as 32-bit. Triggered in some cases by RenderDoc's emulated OpenGL driver.
    if (!functions->getInteger64v)
    {
        GLint result = 0;
        functions->getIntegerv(name, &result);
        return static_cast<GLint64>(result);
    }
    else
    {
        GLint64 result = 0;
        functions->getInteger64v(name, &result);
        return result;
    }
}

static GLfloat QuerySingleGLFloat(const FunctionsGL *functions, GLenum name)
{
    GLfloat result = 0.0f;
    functions->getFloatv(name, &result);
    return result;
}

static GLfloat QueryGLFloatRange(const FunctionsGL *functions, GLenum name, size_t index)
{
    GLfloat result[2] = {};
    functions->getFloatv(name, result);
    return result[index];
}

static gl::TypePrecision QueryTypePrecision(const FunctionsGL *functions,
                                            GLenum shaderType,
                                            GLenum precisionType)
{
    gl::TypePrecision precision;
    functions->getShaderPrecisionFormat(shaderType, precisionType, precision.range.data(),
                                        &precision.precision);
    return precision;
}

static GLint QueryQueryValue(const FunctionsGL *functions, GLenum target, GLenum name)
{
    GLint result;
    functions->getQueryiv(target, name, &result);
    return result;
}

static void LimitVersion(gl::Version *curVersion, const gl::Version &maxVersion)
{
    if (*curVersion >= maxVersion)
    {
        *curVersion = maxVersion;
    }
}

void CapCombinedLimitToESShaders(GLuint *combinedLimit, gl::ShaderMap<GLuint> &perShaderLimit)
{
    GLuint combinedESLimit = 0;
    for (gl::ShaderType shaderType : gl::kAllGraphicsShaderTypes)
    {
        combinedESLimit += perShaderLimit[shaderType];
    }

    *combinedLimit = std::min(*combinedLimit, combinedESLimit);
}

void GenerateCaps(const FunctionsGL *functions,
                  const angle::FeaturesGL &features,
                  gl::Caps *caps,
                  gl::TextureCapsMap *textureCapsMap,
                  gl::Extensions *extensions,
                  gl::Version *maxSupportedESVersion,
                  MultiviewImplementationTypeGL *multiviewImplementationType)
{
    // Texture format support checks
    const gl::FormatSet &allFormats = gl::GetAllSizedInternalFormats();
    for (GLenum internalFormat : allFormats)
    {
        gl::TextureCaps textureCaps =
            GenerateTextureFormatCaps(functions, features, internalFormat);
        textureCapsMap->insert(internalFormat, textureCaps);

        if (gl::GetSizedInternalFormatInfo(internalFormat).compressed)
        {
            caps->compressedTextureFormats.push_back(internalFormat);
        }
    }

    // Start by assuming ES3.1 support and work down
    *maxSupportedESVersion = gl::Version(3, 1);

    // Table 6.28, implementation dependent values
    if (functions->isAtLeastGL(gl::Version(4, 3)) ||
        functions->hasGLExtension("GL_ARB_ES3_compatibility") ||
        functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        caps->maxElementIndex = QuerySingleGLInt64(functions, GL_MAX_ELEMENT_INDEX);

        // Work around the null driver limitations.
        if (caps->maxElementIndex == 0)
        {
            caps->maxElementIndex = 0xFFFF;
        }
    }
    else
    {
        // Doesn't affect ES3 support, can use a pre-defined limit
        caps->maxElementIndex = static_cast<GLint64>(std::numeric_limits<unsigned int>::max());
    }

    GLint textureSizeLimit = std::numeric_limits<GLint>::max();
    if (features.limitMaxTextureSizeTo4096.enabled)
    {
        textureSizeLimit = 4096;
    }

    GLint max3dArrayTextureSizeLimit = std::numeric_limits<GLint>::max();
    if (features.limitMax3dArrayTextureSizeTo1024.enabled)
    {
        max3dArrayTextureSizeLimit = 1024;
    }

    if (functions->isAtLeastGL(gl::Version(1, 2)) || functions->isAtLeastGLES(gl::Version(3, 0)) ||
        functions->hasGLESExtension("GL_OES_texture_3D"))
    {
        caps->max3DTextureSize = std::min({QuerySingleGLInt(functions, GL_MAX_3D_TEXTURE_SIZE),
                                           textureSizeLimit, max3dArrayTextureSizeLimit});
    }
    else
    {
        // Can't support ES3 without 3D textures
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    caps->max2DTextureSize = std::min(QuerySingleGLInt(functions, GL_MAX_TEXTURE_SIZE),
                                      textureSizeLimit);  // GL 1.0 / ES 2.0
    caps->maxCubeMapTextureSize =
        std::min(QuerySingleGLInt(functions, GL_MAX_CUBE_MAP_TEXTURE_SIZE),
                 textureSizeLimit);  // GL 1.3 / ES 2.0

    if (functions->isAtLeastGL(gl::Version(3, 0)) ||
        functions->hasGLExtension("GL_EXT_texture_array") ||
        functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        caps->maxArrayTextureLayers =
            std::min({QuerySingleGLInt(functions, GL_MAX_ARRAY_TEXTURE_LAYERS), textureSizeLimit,
                      max3dArrayTextureSizeLimit});
    }
    else
    {
        // Can't support ES3 without array textures
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    if (functions->isAtLeastGL(gl::Version(1, 5)) ||
        functions->hasGLExtension("GL_EXT_texture_lod_bias") ||
        functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        caps->maxLODBias = QuerySingleGLFloat(functions, GL_MAX_TEXTURE_LOD_BIAS);
    }
    else
    {
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    if (functions->isAtLeastGL(gl::Version(3, 0)) ||
        functions->hasGLExtension("GL_EXT_framebuffer_object") ||
        functions->isAtLeastGLES(gl::Version(2, 0)))
    {
        caps->maxRenderbufferSize = QuerySingleGLInt(functions, GL_MAX_RENDERBUFFER_SIZE);
        caps->maxColorAttachments = QuerySingleGLInt(functions, GL_MAX_COLOR_ATTACHMENTS);
    }
    else
    {
        // Can't support ES2 without framebuffers and renderbuffers
        LimitVersion(maxSupportedESVersion, gl::Version(0, 0));
    }

    if (functions->isAtLeastGL(gl::Version(2, 0)) ||
        functions->hasGLExtension("ARB_draw_buffers") ||
        functions->isAtLeastGLES(gl::Version(3, 0)) ||
        functions->hasGLESExtension("GL_EXT_draw_buffers"))
    {
        caps->maxDrawBuffers = QuerySingleGLInt(functions, GL_MAX_DRAW_BUFFERS);
    }
    else
    {
        // Framebuffer is required to have at least one drawbuffer even if the extension is not
        // supported
        caps->maxDrawBuffers = 1;
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    caps->maxViewportWidth =
        QueryGLIntRange(functions, GL_MAX_VIEWPORT_DIMS, 0);  // GL 1.0 / ES 2.0
    caps->maxViewportHeight =
        QueryGLIntRange(functions, GL_MAX_VIEWPORT_DIMS, 1);  // GL 1.0 / ES 2.0

    if (functions->standard == STANDARD_GL_DESKTOP &&
        (functions->profile & GL_CONTEXT_CORE_PROFILE_BIT) != 0)
    {
        // Desktop GL core profile deprecated the GL_ALIASED_POINT_SIZE_RANGE query.  Use
        // GL_POINT_SIZE_RANGE instead.
        caps->minAliasedPointSize =
            std::max(1.0f, QueryGLFloatRange(functions, GL_POINT_SIZE_RANGE, 0));
        caps->maxAliasedPointSize = QueryGLFloatRange(functions, GL_POINT_SIZE_RANGE, 1);
    }
    else
    {
        caps->minAliasedPointSize =
            std::max(1.0f, QueryGLFloatRange(functions, GL_ALIASED_POINT_SIZE_RANGE, 0));
        caps->maxAliasedPointSize = QueryGLFloatRange(functions, GL_ALIASED_POINT_SIZE_RANGE, 1);
    }

    caps->minAliasedLineWidth =
        QueryGLFloatRange(functions, GL_ALIASED_LINE_WIDTH_RANGE, 0);  // GL 1.2 / ES 2.0
    caps->maxAliasedLineWidth =
        QueryGLFloatRange(functions, GL_ALIASED_LINE_WIDTH_RANGE, 1);  // GL 1.2 / ES 2.0

    // Table 6.29, implementation dependent values (cont.)
    if (functions->isAtLeastGL(gl::Version(1, 2)) || functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        caps->maxElementsIndices  = QuerySingleGLInt(functions, GL_MAX_ELEMENTS_INDICES);
        caps->maxElementsVertices = QuerySingleGLInt(functions, GL_MAX_ELEMENTS_VERTICES);
    }
    else
    {
        // Doesn't impact supported version
    }

    if (functions->isAtLeastGL(gl::Version(4, 1)) ||
        functions->hasGLExtension("GL_ARB_get_program_binary") ||
        functions->isAtLeastGLES(gl::Version(3, 0)) ||
        functions->hasGLESExtension("GL_OES_get_program_binary"))
    {
        // Able to support the GL_PROGRAM_BINARY_ANGLE format as long as another program binary
        // format is available.
        GLint numBinaryFormats = QuerySingleGLInt(functions, GL_NUM_PROGRAM_BINARY_FORMATS_OES);
        if (numBinaryFormats > 0)
        {
            caps->programBinaryFormats.push_back(GL_PROGRAM_BINARY_ANGLE);
        }
    }
    else
    {
        // Doesn't impact supported version
    }

    // glGetShaderPrecisionFormat is not available until desktop GL version 4.1 or
    // GL_ARB_ES2_compatibility exists
    if (functions->isAtLeastGL(gl::Version(4, 1)) ||
        functions->hasGLExtension("GL_ARB_ES2_compatibility") ||
        functions->isAtLeastGLES(gl::Version(2, 0)))
    {
        caps->vertexHighpFloat   = QueryTypePrecision(functions, GL_VERTEX_SHADER, GL_HIGH_FLOAT);
        caps->vertexMediumpFloat = QueryTypePrecision(functions, GL_VERTEX_SHADER, GL_MEDIUM_FLOAT);
        caps->vertexLowpFloat    = QueryTypePrecision(functions, GL_VERTEX_SHADER, GL_LOW_FLOAT);
        caps->fragmentHighpFloat = QueryTypePrecision(functions, GL_FRAGMENT_SHADER, GL_HIGH_FLOAT);
        caps->fragmentMediumpFloat =
            QueryTypePrecision(functions, GL_FRAGMENT_SHADER, GL_MEDIUM_FLOAT);
        caps->fragmentLowpFloat  = QueryTypePrecision(functions, GL_FRAGMENT_SHADER, GL_LOW_FLOAT);
        caps->vertexHighpInt     = QueryTypePrecision(functions, GL_VERTEX_SHADER, GL_HIGH_INT);
        caps->vertexMediumpInt   = QueryTypePrecision(functions, GL_VERTEX_SHADER, GL_MEDIUM_INT);
        caps->vertexLowpInt      = QueryTypePrecision(functions, GL_VERTEX_SHADER, GL_LOW_INT);
        caps->fragmentHighpInt   = QueryTypePrecision(functions, GL_FRAGMENT_SHADER, GL_HIGH_INT);
        caps->fragmentMediumpInt = QueryTypePrecision(functions, GL_FRAGMENT_SHADER, GL_MEDIUM_INT);
        caps->fragmentLowpInt    = QueryTypePrecision(functions, GL_FRAGMENT_SHADER, GL_LOW_INT);
    }
    else
    {
        // Doesn't impact supported version, set some default values
        caps->vertexHighpFloat.setIEEEFloat();
        caps->vertexMediumpFloat.setIEEEFloat();
        caps->vertexLowpFloat.setIEEEFloat();
        caps->fragmentHighpFloat.setIEEEFloat();
        caps->fragmentMediumpFloat.setIEEEFloat();
        caps->fragmentLowpFloat.setIEEEFloat();
        caps->vertexHighpInt.setTwosComplementInt(32);
        caps->vertexMediumpInt.setTwosComplementInt(32);
        caps->vertexLowpInt.setTwosComplementInt(32);
        caps->fragmentHighpInt.setTwosComplementInt(32);
        caps->fragmentMediumpInt.setTwosComplementInt(32);
        caps->fragmentLowpInt.setTwosComplementInt(32);
    }

    if (functions->isAtLeastGL(gl::Version(3, 2)) || functions->hasGLExtension("GL_ARB_sync") ||
        functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        // Work around Linux NVIDIA driver bug where GL_TIMEOUT_IGNORED is returned.
        caps->maxServerWaitTimeout =
            std::max<GLint64>(QuerySingleGLInt64(functions, GL_MAX_SERVER_WAIT_TIMEOUT), 0);
    }
    else
    {
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    // Table 6.31, implementation dependent vertex shader limits
    if (functions->isAtLeastGL(gl::Version(2, 0)) || functions->isAtLeastGLES(gl::Version(2, 0)))
    {
        caps->maxVertexAttributes = QuerySingleGLInt(functions, GL_MAX_VERTEX_ATTRIBS);
        caps->maxShaderUniformComponents[gl::ShaderType::Vertex] =
            QuerySingleGLInt(functions, GL_MAX_VERTEX_UNIFORM_COMPONENTS);
        caps->maxShaderTextureImageUnits[gl::ShaderType::Vertex] =
            QuerySingleGLInt(functions, GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS);
    }
    else
    {
        // Can't support ES2 version without these caps
        LimitVersion(maxSupportedESVersion, gl::Version(0, 0));
    }

    if (functions->isAtLeastGL(gl::Version(4, 1)) ||
        functions->hasGLExtension("GL_ARB_ES2_compatibility") ||
        functions->isAtLeastGLES(gl::Version(2, 0)))
    {
        caps->maxVertexUniformVectors = QuerySingleGLInt(functions, GL_MAX_VERTEX_UNIFORM_VECTORS);
        caps->maxFragmentUniformVectors =
            QuerySingleGLInt(functions, GL_MAX_FRAGMENT_UNIFORM_VECTORS);
    }
    else
    {
        // Doesn't limit ES version, GL_MAX_VERTEX_UNIFORM_COMPONENTS / 4 is acceptable.
        caps->maxVertexUniformVectors =
            caps->maxShaderUniformComponents[gl::ShaderType::Vertex] / 4;
        // Doesn't limit ES version, GL_MAX_FRAGMENT_UNIFORM_COMPONENTS / 4 is acceptable.
        caps->maxFragmentUniformVectors =
            caps->maxShaderUniformComponents[gl::ShaderType::Fragment] / 4;
    }

    if (functions->isAtLeastGL(gl::Version(3, 2)) || functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        caps->maxVertexOutputComponents =
            QuerySingleGLInt(functions, GL_MAX_VERTEX_OUTPUT_COMPONENTS);
    }
    else
    {
        // There doesn't seem, to be a desktop extension to add this cap, maybe it could be given a
        // safe limit instead of limiting the supported ES version.
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    // Table 6.32, implementation dependent fragment shader limits
    if (functions->isAtLeastGL(gl::Version(2, 0)) || functions->isAtLeastGLES(gl::Version(2, 0)))
    {
        caps->maxShaderUniformComponents[gl::ShaderType::Fragment] =
            QuerySingleGLInt(functions, GL_MAX_FRAGMENT_UNIFORM_COMPONENTS);
        caps->maxShaderTextureImageUnits[gl::ShaderType::Fragment] =
            QuerySingleGLInt(functions, GL_MAX_TEXTURE_IMAGE_UNITS);
    }
    else
    {
        // Can't support ES2 version without these caps
        LimitVersion(maxSupportedESVersion, gl::Version(0, 0));
    }

    if (functions->isAtLeastGL(gl::Version(3, 2)) || functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        caps->maxFragmentInputComponents =
            QuerySingleGLInt(functions, GL_MAX_FRAGMENT_INPUT_COMPONENTS);
    }
    else
    {
        // There doesn't seem, to be a desktop extension to add this cap, maybe it could be given a
        // safe limit instead of limiting the supported ES version.
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    if (functions->isAtLeastGL(gl::Version(3, 0)) || functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        caps->minProgramTexelOffset = QuerySingleGLInt(functions, GL_MIN_PROGRAM_TEXEL_OFFSET);
        caps->maxProgramTexelOffset = QuerySingleGLInt(functions, GL_MAX_PROGRAM_TEXEL_OFFSET);
    }
    else
    {
        // Can't support ES3 without texel offset, could possibly be emulated in the shader
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    // Table 6.33, implementation dependent aggregate shader limits
    if (functions->isAtLeastGL(gl::Version(3, 1)) ||
        functions->hasGLExtension("GL_ARB_uniform_buffer_object") ||
        functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        caps->maxShaderUniformBlocks[gl::ShaderType::Vertex] =
            QuerySingleGLInt(functions, GL_MAX_VERTEX_UNIFORM_BLOCKS);
        caps->maxShaderUniformBlocks[gl::ShaderType::Fragment] =
            QuerySingleGLInt(functions, GL_MAX_FRAGMENT_UNIFORM_BLOCKS);
        caps->maxUniformBufferBindings =
            QuerySingleGLInt(functions, GL_MAX_UNIFORM_BUFFER_BINDINGS);
        caps->maxUniformBlockSize = QuerySingleGLInt64(functions, GL_MAX_UNIFORM_BLOCK_SIZE);
        caps->uniformBufferOffsetAlignment =
            QuerySingleGLInt(functions, GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT);
        caps->maxCombinedUniformBlocks =
            QuerySingleGLInt(functions, GL_MAX_COMBINED_UNIFORM_BLOCKS);
        caps->maxCombinedShaderUniformComponents[gl::ShaderType::Vertex] =
            QuerySingleGLInt64(functions, GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS);
        caps->maxCombinedShaderUniformComponents[gl::ShaderType::Fragment] =
            QuerySingleGLInt64(functions, GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS);
    }
    else
    {
        // Can't support ES3 without uniform blocks
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    if (functions->isAtLeastGL(gl::Version(3, 2)) &&
        (functions->profile & GL_CONTEXT_CORE_PROFILE_BIT) != 0)
    {
        caps->maxVaryingComponents = QuerySingleGLInt(functions, GL_MAX_VERTEX_OUTPUT_COMPONENTS);
    }
    else if (functions->isAtLeastGL(gl::Version(3, 0)) ||
             functions->hasGLExtension("GL_ARB_ES2_compatibility") ||
             functions->isAtLeastGLES(gl::Version(2, 0)))
    {
        caps->maxVaryingComponents = QuerySingleGLInt(functions, GL_MAX_VARYING_COMPONENTS);
    }
    else if (functions->isAtLeastGL(gl::Version(2, 0)))
    {
        caps->maxVaryingComponents = QuerySingleGLInt(functions, GL_MAX_VARYING_FLOATS);
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }
    else
    {
        LimitVersion(maxSupportedESVersion, gl::Version(0, 0));
    }

    if (functions->isAtLeastGL(gl::Version(4, 1)) ||
        functions->hasGLExtension("GL_ARB_ES2_compatibility") ||
        functions->isAtLeastGLES(gl::Version(2, 0)))
    {
        caps->maxVaryingVectors = QuerySingleGLInt(functions, GL_MAX_VARYING_VECTORS);
    }
    else
    {
        // Doesn't limit ES version, GL_MAX_VARYING_COMPONENTS / 4 is acceptable.
        caps->maxVaryingVectors = caps->maxVaryingComponents / 4;
    }

    // Determine the max combined texture image units by adding the vertex and fragment limits.  If
    // the real cap is queried, it would contain the limits for shader types that are not available
    // to ES.
    caps->maxCombinedTextureImageUnits =
        QuerySingleGLInt(functions, GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS);

    // Table 6.34, implementation dependent transform feedback limits
    if (functions->isAtLeastGL(gl::Version(4, 0)) ||
        functions->hasGLExtension("GL_ARB_transform_feedback2") ||
        functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        caps->maxTransformFeedbackInterleavedComponents =
            QuerySingleGLInt(functions, GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS);
        caps->maxTransformFeedbackSeparateAttributes =
            QuerySingleGLInt(functions, GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS);
        caps->maxTransformFeedbackSeparateComponents =
            QuerySingleGLInt(functions, GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS);
    }
    else
    {
        // Can't support ES3 without transform feedback
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    GLint sampleCountLimit = std::numeric_limits<GLint>::max();
    if (features.limitMaxMSAASamplesTo4.enabled)
    {
        sampleCountLimit = 4;
    }

    // Table 6.35, Framebuffer Dependent Values
    if (functions->isAtLeastGL(gl::Version(3, 0)) ||
        functions->hasGLExtension("GL_EXT_framebuffer_multisample") ||
        functions->isAtLeastGLES(gl::Version(3, 0)) ||
        functions->hasGLESExtension("GL_EXT_multisampled_render_to_texture"))
    {
        caps->maxSamples = std::min(QuerySingleGLInt(functions, GL_MAX_SAMPLES), sampleCountLimit);
    }
    else
    {
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    // Non-constant sampler array indexing is required for OpenGL ES 2 and OpenGL ES after 3.2.
    // However having it available on OpenGL ES 2 is a specification bug, and using this
    // indexing in WebGL is undefined. Requiring this feature would break WebGL 1 for some users
    // so we don't check for it. (it is present with ESSL 100, ESSL >= 320, GLSL >= 400 and
    // GL_ARB_gpu_shader5)

    // Check if sampler objects are supported
    if (!functions->isAtLeastGL(gl::Version(3, 3)) &&
        !functions->hasGLExtension("GL_ARB_sampler_objects") &&
        !functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        // Can't support ES3 without sampler objects
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    // Can't support ES3 without texture swizzling
    if (!functions->isAtLeastGL(gl::Version(3, 3)) &&
        !functions->hasGLExtension("GL_ARB_texture_swizzle") &&
        !functions->hasGLExtension("GL_EXT_texture_swizzle") &&
        !functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));

        // Texture swizzling is required to work around the luminance texture format not being
        // present in the core profile
        if (functions->profile & GL_CONTEXT_CORE_PROFILE_BIT)
        {
            LimitVersion(maxSupportedESVersion, gl::Version(0, 0));
        }
    }

    // Can't support ES3 without the GLSL packing builtins. We have a workaround for all
    // desktop OpenGL versions starting from 3.3 with the bit packing extension.
    if (!functions->isAtLeastGL(gl::Version(4, 2)) &&
        !(functions->isAtLeastGL(gl::Version(3, 2)) &&
          functions->hasGLExtension("GL_ARB_shader_bit_encoding")) &&
        !functions->hasGLExtension("GL_ARB_shading_language_packing") &&
        !functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    // ES3 needs to support explicit layout location qualifiers, while it might be possible to
    // fake them in our side, we currently don't support that.
    if (!functions->isAtLeastGL(gl::Version(3, 3)) &&
        !functions->hasGLExtension("GL_ARB_explicit_attrib_location") &&
        !functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    if (functions->isAtLeastGL(gl::Version(4, 3)) || functions->isAtLeastGLES(gl::Version(3, 1)) ||
        functions->hasGLExtension("GL_ARB_framebuffer_no_attachments"))
    {
        caps->maxFramebufferWidth  = QuerySingleGLInt(functions, GL_MAX_FRAMEBUFFER_WIDTH);
        caps->maxFramebufferHeight = QuerySingleGLInt(functions, GL_MAX_FRAMEBUFFER_HEIGHT);
        caps->maxFramebufferSamples =
            std::min(QuerySingleGLInt(functions, GL_MAX_FRAMEBUFFER_SAMPLES), sampleCountLimit);
    }
    else
    {
        LimitVersion(maxSupportedESVersion, gl::Version(3, 0));
    }

    if (functions->isAtLeastGL(gl::Version(3, 2)) || functions->isAtLeastGLES(gl::Version(3, 1)) ||
        functions->hasGLExtension("GL_ARB_texture_multisample"))
    {
        caps->maxSampleMaskWords = QuerySingleGLInt(functions, GL_MAX_SAMPLE_MASK_WORDS);
        caps->maxColorTextureSamples =
            std::min(QuerySingleGLInt(functions, GL_MAX_COLOR_TEXTURE_SAMPLES), sampleCountLimit);
        caps->maxDepthTextureSamples =
            std::min(QuerySingleGLInt(functions, GL_MAX_DEPTH_TEXTURE_SAMPLES), sampleCountLimit);
        caps->maxIntegerSamples =
            std::min(QuerySingleGLInt(functions, GL_MAX_INTEGER_SAMPLES), sampleCountLimit);
    }
    else
    {
        LimitVersion(maxSupportedESVersion, gl::Version(3, 0));
    }

    if (functions->isAtLeastGL(gl::Version(4, 3)) || functions->isAtLeastGLES(gl::Version(3, 1)) ||
        functions->hasGLExtension("GL_ARB_vertex_attrib_binding"))
    {
        caps->maxVertexAttribRelativeOffset =
            QuerySingleGLInt(functions, GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET);
        caps->maxVertexAttribBindings = QuerySingleGLInt(functions, GL_MAX_VERTEX_ATTRIB_BINDINGS);

        // OpenGL 4.3 has no limit on maximum value of stride.
        // [OpenGL 4.3 (Core Profile) - February 14, 2013] Chapter 10.3.1 Page 298
        if (features.emulateMaxVertexAttribStride.enabled ||
            (functions->standard == STANDARD_GL_DESKTOP && functions->version == gl::Version(4, 3)))
        {
            caps->maxVertexAttribStride = 2048;
        }
        else
        {
            caps->maxVertexAttribStride = QuerySingleGLInt(functions, GL_MAX_VERTEX_ATTRIB_STRIDE);
        }
    }
    else
    {
        LimitVersion(maxSupportedESVersion, gl::Version(3, 0));
    }

    if (functions->isAtLeastGL(gl::Version(4, 3)) || functions->isAtLeastGLES(gl::Version(3, 1)) ||
        functions->hasGLExtension("GL_ARB_shader_storage_buffer_object"))
    {
        caps->maxCombinedShaderOutputResources =
            QuerySingleGLInt(functions, GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES);
        caps->maxShaderStorageBlocks[gl::ShaderType::Fragment] =
            QuerySingleGLInt(functions, GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS);
        caps->maxShaderStorageBlocks[gl::ShaderType::Vertex] =
            QuerySingleGLInt(functions, GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS);
        caps->maxShaderStorageBufferBindings =
            QuerySingleGLInt(functions, GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS);
        caps->maxShaderStorageBlockSize =
            QuerySingleGLInt64(functions, GL_MAX_SHADER_STORAGE_BLOCK_SIZE);
        caps->maxCombinedShaderStorageBlocks =
            QuerySingleGLInt(functions, GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS);
        caps->shaderStorageBufferOffsetAlignment =
            QuerySingleGLInt(functions, GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT);
    }
    else
    {
        LimitVersion(maxSupportedESVersion, gl::Version(3, 0));
    }

    if (nativegl::SupportsCompute(functions))
    {
        for (GLuint index = 0u; index < 3u; ++index)
        {
            caps->maxComputeWorkGroupCount[index] =
                QuerySingleIndexGLInt(functions, GL_MAX_COMPUTE_WORK_GROUP_COUNT, index);

            caps->maxComputeWorkGroupSize[index] =
                QuerySingleIndexGLInt(functions, GL_MAX_COMPUTE_WORK_GROUP_SIZE, index);
        }
        caps->maxComputeWorkGroupInvocations =
            QuerySingleGLInt(functions, GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS);
        caps->maxShaderUniformBlocks[gl::ShaderType::Compute] =
            QuerySingleGLInt(functions, GL_MAX_COMPUTE_UNIFORM_BLOCKS);
        caps->maxShaderTextureImageUnits[gl::ShaderType::Compute] =
            QuerySingleGLInt(functions, GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS);
        caps->maxComputeSharedMemorySize =
            QuerySingleGLInt(functions, GL_MAX_COMPUTE_SHARED_MEMORY_SIZE);
        caps->maxShaderUniformComponents[gl::ShaderType::Compute] =
            QuerySingleGLInt(functions, GL_MAX_COMPUTE_UNIFORM_COMPONENTS);
        caps->maxShaderAtomicCounterBuffers[gl::ShaderType::Compute] =
            QuerySingleGLInt(functions, GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS);
        caps->maxShaderAtomicCounters[gl::ShaderType::Compute] =
            QuerySingleGLInt(functions, GL_MAX_COMPUTE_ATOMIC_COUNTERS);
        caps->maxShaderImageUniforms[gl::ShaderType::Compute] =
            QuerySingleGLInt(functions, GL_MAX_COMPUTE_IMAGE_UNIFORMS);
        caps->maxCombinedShaderUniformComponents[gl::ShaderType::Compute] =
            QuerySingleGLInt(functions, GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS);
        caps->maxShaderStorageBlocks[gl::ShaderType::Compute] =
            QuerySingleGLInt(functions, GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS);
    }
    else
    {
        LimitVersion(maxSupportedESVersion, gl::Version(3, 0));
    }

    if (functions->isAtLeastGL(gl::Version(4, 3)) || functions->isAtLeastGLES(gl::Version(3, 1)) ||
        functions->hasGLExtension("GL_ARB_explicit_uniform_location"))
    {
        caps->maxUniformLocations = QuerySingleGLInt(functions, GL_MAX_UNIFORM_LOCATIONS);
    }
    else
    {
        LimitVersion(maxSupportedESVersion, gl::Version(3, 0));
    }

    if (functions->isAtLeastGL(gl::Version(4, 0)) || functions->isAtLeastGLES(gl::Version(3, 1)) ||
        functions->hasGLExtension("GL_ARB_texture_gather"))
    {
        caps->minProgramTextureGatherOffset =
            QuerySingleGLInt(functions, GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET);
        caps->maxProgramTextureGatherOffset =
            QuerySingleGLInt(functions, GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET);
    }
    else
    {
        LimitVersion(maxSupportedESVersion, gl::Version(3, 0));
    }

    if (functions->isAtLeastGL(gl::Version(4, 2)) || functions->isAtLeastGLES(gl::Version(3, 1)) ||
        functions->hasGLExtension("GL_ARB_shader_image_load_store"))
    {
        caps->maxShaderImageUniforms[gl::ShaderType::Vertex] =
            QuerySingleGLInt(functions, GL_MAX_VERTEX_IMAGE_UNIFORMS);
        caps->maxShaderImageUniforms[gl::ShaderType::Fragment] =
            QuerySingleGLInt(functions, GL_MAX_FRAGMENT_IMAGE_UNIFORMS);
        caps->maxImageUnits = QuerySingleGLInt(functions, GL_MAX_IMAGE_UNITS);
        caps->maxCombinedImageUniforms =
            QuerySingleGLInt(functions, GL_MAX_COMBINED_IMAGE_UNIFORMS);
    }
    else
    {
        LimitVersion(maxSupportedESVersion, gl::Version(3, 0));
    }

    if (functions->isAtLeastGL(gl::Version(4, 2)) || functions->isAtLeastGLES(gl::Version(3, 1)) ||
        functions->hasGLExtension("GL_ARB_shader_atomic_counters"))
    {
        caps->maxShaderAtomicCounterBuffers[gl::ShaderType::Vertex] =
            QuerySingleGLInt(functions, GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS);
        caps->maxShaderAtomicCounters[gl::ShaderType::Vertex] =
            QuerySingleGLInt(functions, GL_MAX_VERTEX_ATOMIC_COUNTERS);
        caps->maxShaderAtomicCounterBuffers[gl::ShaderType::Fragment] =
            QuerySingleGLInt(functions, GL_MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS);
        caps->maxShaderAtomicCounters[gl::ShaderType::Fragment] =
            QuerySingleGLInt(functions, GL_MAX_FRAGMENT_ATOMIC_COUNTERS);
        caps->maxAtomicCounterBufferBindings =
            QuerySingleGLInt(functions, GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS);
        caps->maxAtomicCounterBufferSize =
            QuerySingleGLInt(functions, GL_MAX_ATOMIC_COUNTER_BUFFER_SIZE);
        caps->maxCombinedAtomicCounterBuffers =
            QuerySingleGLInt(functions, GL_MAX_COMBINED_ATOMIC_COUNTER_BUFFERS);
        caps->maxCombinedAtomicCounters =
            QuerySingleGLInt(functions, GL_MAX_COMBINED_ATOMIC_COUNTERS);
    }
    else
    {
        LimitVersion(maxSupportedESVersion, gl::Version(3, 0));
    }

    // TODO(geofflang): The gl-uniform-arrays WebGL conformance test struggles to complete on time
    // if the max uniform vectors is too large.  Artificially limit the maximum until the test is
    // updated.
    caps->maxVertexUniformVectors = std::min(1024u, caps->maxVertexUniformVectors);
    caps->maxShaderUniformComponents[gl::ShaderType::Vertex] =
        std::min(caps->maxVertexUniformVectors * 4,
                 caps->maxShaderUniformComponents[gl::ShaderType::Vertex]);
    caps->maxFragmentUniformVectors = std::min(1024u, caps->maxFragmentUniformVectors);
    caps->maxShaderUniformComponents[gl::ShaderType::Fragment] =
        std::min(caps->maxFragmentUniformVectors * 4,
                 caps->maxShaderUniformComponents[gl::ShaderType::Fragment]);

    // If it is not possible to support reading buffer data back, a shadow copy of the buffers must
    // be held. This disallows writing to buffers indirectly through transform feedback, thus
    // disallowing ES3.
    if (!CanMapBufferForRead(functions))
    {
        LimitVersion(maxSupportedESVersion, gl::Version(2, 0));
    }

    // Extension support
    extensions->setTextureExtensionSupport(*textureCapsMap);
    extensions->textureCompressionASTCHDRKHR =
        extensions->textureCompressionASTCLDRKHR &&
        functions->hasGLESExtension("GL_KHR_texture_compression_astc_hdr");
    extensions->elementIndexUint = functions->standard == STANDARD_GL_DESKTOP ||
                                   functions->isAtLeastGLES(gl::Version(3, 0)) ||
                                   functions->hasGLESExtension("GL_OES_element_index_uint");
    extensions->getProgramBinary = caps->programBinaryFormats.size() > 0;
    extensions->readFormatBGRA   = functions->isAtLeastGL(gl::Version(1, 2)) ||
                                 functions->hasGLExtension("GL_EXT_bgra") ||
                                 functions->hasGLESExtension("GL_EXT_read_format_bgra");
    extensions->mapBuffer = functions->isAtLeastGL(gl::Version(1, 5)) ||
                            functions->isAtLeastGLES(gl::Version(3, 0)) ||
                            functions->hasGLESExtension("GL_OES_mapbuffer");
    extensions->mapBufferRange = functions->isAtLeastGL(gl::Version(3, 0)) ||
                                 functions->hasGLExtension("GL_ARB_map_buffer_range") ||
                                 functions->isAtLeastGLES(gl::Version(3, 0)) ||
                                 functions->hasGLESExtension("GL_EXT_map_buffer_range");
    extensions->textureNPOT = functions->standard == STANDARD_GL_DESKTOP ||
                              functions->isAtLeastGLES(gl::Version(3, 0)) ||
                              functions->hasGLESExtension("GL_OES_texture_npot");
    // TODO(jmadill): Investigate emulating EXT_draw_buffers on ES 3.0's core functionality.
    extensions->drawBuffers = functions->isAtLeastGL(gl::Version(2, 0)) ||
                              functions->hasGLExtension("ARB_draw_buffers") ||
                              functions->hasGLESExtension("GL_EXT_draw_buffers");
    extensions->textureStorage = functions->standard == STANDARD_GL_DESKTOP ||
                                 functions->hasGLESExtension("GL_EXT_texture_storage");
    extensions->textureFilterAnisotropic =
        functions->hasGLExtension("GL_EXT_texture_filter_anisotropic") ||
        functions->hasGLESExtension("GL_EXT_texture_filter_anisotropic");
    extensions->occlusionQueryBoolean = nativegl::SupportsOcclusionQueries(functions);
    extensions->maxTextureAnisotropy =
        extensions->textureFilterAnisotropic
            ? QuerySingleGLFloat(functions, GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
            : 0.0f;
    extensions->fence = FenceNVGL::Supported(functions) || FenceNVSyncGL::Supported(functions);
    extensions->blendMinMax = functions->isAtLeastGL(gl::Version(1, 5)) ||
                              functions->hasGLExtension("GL_EXT_blend_minmax") ||
                              functions->isAtLeastGLES(gl::Version(3, 0)) ||
                              functions->hasGLESExtension("GL_EXT_blend_minmax");
    extensions->framebufferBlit        = (functions->blitFramebuffer != nullptr);
    extensions->framebufferMultisample = caps->maxSamples > 0;
    extensions->standardDerivatives    = functions->isAtLeastGL(gl::Version(2, 0)) ||
                                      functions->hasGLExtension("GL_ARB_fragment_shader") ||
                                      functions->hasGLESExtension("GL_OES_standard_derivatives");
    extensions->shaderTextureLOD = functions->isAtLeastGL(gl::Version(3, 0)) ||
                                   functions->hasGLExtension("GL_ARB_shader_texture_lod") ||
                                   functions->hasGLESExtension("GL_EXT_shader_texture_lod");
    extensions->fragDepth = functions->standard == STANDARD_GL_DESKTOP ||
                            functions->hasGLESExtension("GL_EXT_frag_depth");

    if (functions->hasGLExtension("GL_ARB_shader_viewport_layer_array") ||
        functions->hasGLExtension("GL_NV_viewport_array2"))
    {
        extensions->multiview  = true;
        extensions->multiview2 = true;
        // GL_MAX_ARRAY_TEXTURE_LAYERS is guaranteed to be at least 256.
        const int maxLayers = QuerySingleGLInt(functions, GL_MAX_ARRAY_TEXTURE_LAYERS);
        // GL_MAX_VIEWPORTS is guaranteed to be at least 16.
        const int maxViewports = QuerySingleGLInt(functions, GL_MAX_VIEWPORTS);
        extensions->maxViews   = static_cast<GLuint>(
            std::min(static_cast<int>(gl::IMPLEMENTATION_ANGLE_MULTIVIEW_MAX_VIEWS),
                     std::min(maxLayers, maxViewports)));
        *multiviewImplementationType = MultiviewImplementationTypeGL::NV_VIEWPORT_ARRAY2;
    }

    extensions->fboRenderMipmap = functions->isAtLeastGL(gl::Version(3, 0)) ||
                                  functions->hasGLExtension("GL_EXT_framebuffer_object") ||
                                  functions->isAtLeastGLES(gl::Version(3, 0)) ||
                                  functions->hasGLESExtension("GL_OES_fbo_render_mipmap");
    extensions->textureBorderClamp = functions->standard == STANDARD_GL_DESKTOP ||
                                     functions->hasGLESExtension("GL_OES_texture_border_clamp") ||
                                     functions->hasGLESExtension("GL_EXT_texture_border_clamp") ||
                                     functions->hasGLESExtension("GL_NV_texture_border_clamp");
    extensions->instancedArraysANGLE = functions->isAtLeastGL(gl::Version(3, 1)) ||
                                       (functions->hasGLExtension("GL_ARB_instanced_arrays") &&
                                        (functions->hasGLExtension("GL_ARB_draw_instanced") ||
                                         functions->hasGLExtension("GL_EXT_draw_instanced"))) ||
                                       functions->isAtLeastGLES(gl::Version(3, 0)) ||
                                       functions->hasGLESExtension("GL_EXT_instanced_arrays");
    extensions->instancedArraysEXT = extensions->instancedArraysANGLE;
    extensions->unpackSubimage     = functions->standard == STANDARD_GL_DESKTOP ||
                                 functions->isAtLeastGLES(gl::Version(3, 0)) ||
                                 functions->hasGLESExtension("GL_EXT_unpack_subimage");
    extensions->packSubimage = functions->standard == STANDARD_GL_DESKTOP ||
                               functions->isAtLeastGLES(gl::Version(3, 0)) ||
                               functions->hasGLESExtension("GL_NV_pack_subimage");
    extensions->vertexArrayObject = functions->isAtLeastGL(gl::Version(3, 0)) ||
                                    functions->hasGLExtension("GL_ARB_vertex_array_object") ||
                                    functions->isAtLeastGLES(gl::Version(3, 0)) ||
                                    functions->hasGLESExtension("GL_OES_vertex_array_object");
    extensions->debugMarker = functions->isAtLeastGL(gl::Version(4, 3)) ||
                              functions->hasGLExtension("GL_KHR_debug") ||
                              functions->hasGLExtension("GL_EXT_debug_marker") ||
                              functions->isAtLeastGLES(gl::Version(3, 2)) ||
                              functions->hasGLESExtension("GL_KHR_debug") ||
                              functions->hasGLESExtension("GL_EXT_debug_marker");
    extensions->eglImage         = functions->hasGLESExtension("GL_OES_EGL_image");
    extensions->eglImageExternal = functions->hasGLESExtension("GL_OES_EGL_image_external");
    extensions->eglImageExternalEssl3 =
        functions->hasGLESExtension("GL_OES_EGL_image_external_essl3");

    extensions->eglSync = functions->hasGLESExtension("GL_OES_EGL_sync");

    if (functions->isAtLeastGL(gl::Version(3, 3)) ||
        functions->hasGLExtension("GL_ARB_timer_query") ||
        functions->hasGLESExtension("GL_EXT_disjoint_timer_query"))
    {
        extensions->disjointTimerQuery = true;

        // If we can't query the counter bits, leave them at 0.
        if (!features.queryCounterBitsGeneratesErrors.enabled)
        {
            extensions->queryCounterBitsTimeElapsed =
                QueryQueryValue(functions, GL_TIME_ELAPSED, GL_QUERY_COUNTER_BITS);
            extensions->queryCounterBitsTimestamp =
                QueryQueryValue(functions, GL_TIMESTAMP, GL_QUERY_COUNTER_BITS);
        }
    }

    // the EXT_multisample_compatibility is written against ES3.1 but can apply
    // to earlier versions so therefore we're only checking for the extension string
    // and not the specific GLES version.
    extensions->multisampleCompatibility =
        functions->isAtLeastGL(gl::Version(1, 3)) ||
        functions->hasGLESExtension("GL_EXT_multisample_compatibility");

    extensions->framebufferMixedSamples =
        functions->hasGLExtension("GL_NV_framebuffer_mixed_samples") ||
        functions->hasGLESExtension("GL_NV_framebuffer_mixed_samples");

    extensions->robustness = functions->isAtLeastGL(gl::Version(4, 5)) ||
                             functions->hasGLExtension("GL_KHR_robustness") ||
                             functions->hasGLExtension("GL_ARB_robustness") ||
                             functions->isAtLeastGLES(gl::Version(3, 2)) ||
                             functions->hasGLESExtension("GL_KHR_robustness") ||
                             functions->hasGLESExtension("GL_EXT_robustness");

    extensions->robustBufferAccessBehavior =
        extensions->robustness &&
        (functions->hasGLExtension("GL_ARB_robust_buffer_access_behavior") ||
         functions->hasGLESExtension("GL_KHR_robust_buffer_access_behavior"));

    extensions->copyTexture = true;
    extensions->syncQuery   = SyncQueryGL::IsSupported(functions);

    // Note that OES_texture_storage_multisample_2d_array support could be extended down to GL 3.2
    // if we emulated texStorage* API on top of texImage*.
    extensions->textureStorageMultisample2DArray =
        functions->isAtLeastGL(gl::Version(4, 2)) || functions->isAtLeastGLES(gl::Version(3, 2));

    extensions->multiviewMultisample = extensions->textureStorageMultisample2DArray &&
                                       (extensions->multiview || extensions->multiview2);

    extensions->textureMultisample = functions->isAtLeastGL(gl::Version(3, 2)) ||
                                     functions->hasGLExtension("GL_ARB_texture_multisample");

    // NV_path_rendering
    // We also need interface query which is available in
    // >= 4.3 core or ARB_interface_query or >= GLES 3.1
    const bool canEnableGLPathRendering =
        functions->hasGLExtension("GL_NV_path_rendering") &&
        (functions->hasGLExtension("GL_ARB_program_interface_query") ||
         functions->isAtLeastGL(gl::Version(4, 3)));

    const bool canEnableESPathRendering = functions->hasGLESExtension("GL_NV_path_rendering") &&
                                          functions->isAtLeastGLES(gl::Version(3, 1));

    extensions->pathRendering = canEnableGLPathRendering || canEnableESPathRendering;

    extensions->textureSRGBDecode = functions->hasGLExtension("GL_EXT_texture_sRGB_decode") ||
                                    functions->hasGLESExtension("GL_EXT_texture_sRGB_decode");

#if defined(ANGLE_PLATFORM_MACOS) || defined(ANGLE_PLATFORM_MACCATALYST)
    VendorID vendor = GetVendorID(functions);
    if ((IsAMD(vendor) || IsIntel(vendor)) && *maxSupportedESVersion >= gl::Version(3, 0))
    {
        // Apple Intel/AMD drivers do not correctly use the TEXTURE_SRGB_DECODE property of sampler
        // states.  Disable this extension when we would advertise any ES version that has samplers.
        extensions->textureSRGBDecode = false;
    }
#endif

    extensions->sRGBWriteControl = functions->isAtLeastGL(gl::Version(3, 0)) ||
                                   functions->hasGLExtension("GL_EXT_framebuffer_sRGB") ||
                                   functions->hasGLExtension("GL_ARB_framebuffer_sRGB") ||
                                   functions->hasGLESExtension("GL_EXT_sRGB_write_control");

#if defined(ANGLE_PLATFORM_ANDROID)
    // SRGB blending does not appear to work correctly on the Nexus 5. Writing to an SRGB
    // framebuffer with GL_FRAMEBUFFER_SRGB enabled and then reading back returns the same value.
    // Disabling GL_FRAMEBUFFER_SRGB will then convert in the wrong direction.
    extensions->sRGBWriteControl = false;

    // BGRA formats do not appear to be accepted by the Nexus 5X driver dispite the extension being
    // exposed.
    extensions->textureFormatBGRA8888 = false;
#endif

    // EXT_discard_framebuffer can be implemented as long as glDiscardFramebufferEXT or
    // glInvalidateFramebuffer is available
    extensions->discardFramebuffer = functions->isAtLeastGL(gl::Version(4, 3)) ||
                                     functions->hasGLExtension("GL_ARB_invalidate_subdata") ||
                                     functions->isAtLeastGLES(gl::Version(3, 0)) ||
                                     functions->hasGLESExtension("GL_EXT_discard_framebuffer") ||
                                     functions->hasGLESExtension("GL_ARB_invalidate_subdata");

    extensions->translatedShaderSource = true;

    if (functions->isAtLeastGL(gl::Version(3, 1)) ||
        functions->hasGLExtension("GL_ARB_texture_rectangle"))
    {
        extensions->textureRectangle  = true;
        caps->maxRectangleTextureSize = std::min(
            QuerySingleGLInt(functions, GL_MAX_RECTANGLE_TEXTURE_SIZE_ANGLE), textureSizeLimit);
    }

    // OpenGL 4.3 (and above) can support all features and constants defined in
    // GL_EXT_geometry_shader.
    if (functions->isAtLeastGL(gl::Version(4, 3)) || functions->isAtLeastGLES(gl::Version(3, 2)) ||
        functions->hasGLESExtension("GL_OES_geometry_shader") ||
        functions->hasGLESExtension("GL_EXT_geometry_shader") ||
        // OpenGL 4.0 adds the support for instanced geometry shader
        // GL_ARB_shader_atomic_counters adds atomic counters to geometry shader
        // GL_ARB_shader_storage_buffer_object adds shader storage buffers to geometry shader
        // GL_ARB_shader_image_load_store adds images to geometry shader
        (functions->isAtLeastGL(gl::Version(4, 0)) &&
         functions->hasGLExtension("GL_ARB_shader_atomic_counters") &&
         functions->hasGLExtension("GL_ARB_shader_storage_buffer_object") &&
         functions->hasGLExtension("GL_ARB_shader_image_load_store")))
    {
        extensions->geometryShader = true;

        caps->maxFramebufferLayers = QuerySingleGLInt(functions, GL_MAX_FRAMEBUFFER_LAYERS_EXT);

        // GL_PROVOKING_VERTEX isn't a valid return value of GL_LAYER_PROVOKING_VERTEX_EXT in
        // GL_EXT_geometry_shader SPEC, however it is legal in desktop OpenGL, which means the value
        // follows the one set by glProvokingVertex.
        // [OpenGL 4.3] Chapter 11.3.4.6
        // The vertex conventions followed for gl_Layer and gl_ViewportIndex may be determined by
        // calling GetIntegerv with the symbolic constants LAYER_PROVOKING_VERTEX and
        // VIEWPORT_INDEX_PROVOKING_VERTEX, respectively. For either query, if the value returned is
        // PROVOKING_VERTEX, then vertex selection follows the convention specified by
        // ProvokingVertex.
        caps->layerProvokingVertex = QuerySingleGLInt(functions, GL_LAYER_PROVOKING_VERTEX_EXT);
        if (caps->layerProvokingVertex == GL_PROVOKING_VERTEX)
        {
            // We should use GL_LAST_VERTEX_CONVENTION_EXT instead because desktop OpenGL SPEC
            // requires the initial value of provoking vertex mode is LAST_VERTEX_CONVENTION.
            // [OpenGL 4.3] Chapter 13.4
            // The initial value of the provoking vertex mode is LAST_VERTEX_CONVENTION.
            caps->layerProvokingVertex = GL_LAST_VERTEX_CONVENTION_EXT;
        }

        caps->maxShaderUniformComponents[gl::ShaderType::Geometry] =
            QuerySingleGLInt(functions, GL_MAX_GEOMETRY_UNIFORM_COMPONENTS_EXT);
        caps->maxShaderUniformBlocks[gl::ShaderType::Geometry] =
            QuerySingleGLInt(functions, GL_MAX_GEOMETRY_UNIFORM_BLOCKS_EXT);
        caps->maxCombinedShaderUniformComponents[gl::ShaderType::Geometry] =
            QuerySingleGLInt(functions, GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS_EXT);
        caps->maxGeometryInputComponents =
            QuerySingleGLInt(functions, GL_MAX_GEOMETRY_INPUT_COMPONENTS_EXT);
        caps->maxGeometryOutputComponents =
            QuerySingleGLInt(functions, GL_MAX_GEOMETRY_OUTPUT_COMPONENTS_EXT);
        caps->maxGeometryOutputVertices =
            QuerySingleGLInt(functions, GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT);
        caps->maxGeometryTotalOutputComponents =
            QuerySingleGLInt(functions, GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS_EXT);
        caps->maxGeometryShaderInvocations =
            QuerySingleGLInt(functions, GL_MAX_GEOMETRY_SHADER_INVOCATIONS_EXT);
        caps->maxShaderTextureImageUnits[gl::ShaderType::Geometry] =
            QuerySingleGLInt(functions, GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS_EXT);
        caps->maxShaderAtomicCounterBuffers[gl::ShaderType::Geometry] =
            QuerySingleGLInt(functions, GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS_EXT);
        caps->maxShaderAtomicCounters[gl::ShaderType::Geometry] =
            QuerySingleGLInt(functions, GL_MAX_GEOMETRY_ATOMIC_COUNTERS_EXT);
        caps->maxShaderImageUniforms[gl::ShaderType::Geometry] =
            QuerySingleGLInt(functions, GL_MAX_GEOMETRY_IMAGE_UNIFORMS_EXT);
        caps->maxShaderStorageBlocks[gl::ShaderType::Geometry] =
            QuerySingleGLInt(functions, GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS_EXT);
    }

    // The real combined caps contain limits for shader types that are not available to ES, so limit
    // the caps to the sum of vertex+fragment+geometry shader caps.
    CapCombinedLimitToESShaders(&caps->maxCombinedUniformBlocks, caps->maxShaderUniformBlocks);
    CapCombinedLimitToESShaders(&caps->maxCombinedTextureImageUnits,
                                caps->maxShaderTextureImageUnits);
    CapCombinedLimitToESShaders(&caps->maxCombinedShaderStorageBlocks,
                                caps->maxShaderStorageBlocks);
    CapCombinedLimitToESShaders(&caps->maxCombinedImageUniforms, caps->maxShaderImageUniforms);
    CapCombinedLimitToESShaders(&caps->maxCombinedAtomicCounterBuffers,
                                caps->maxShaderAtomicCounterBuffers);
    CapCombinedLimitToESShaders(&caps->maxCombinedAtomicCounters, caps->maxShaderAtomicCounters);

    // EXT_blend_func_extended.
    // Note that this could be implemented also on top of native EXT_blend_func_extended, but it's
    // currently not fully implemented.
    extensions->blendFuncExtended = !features.disableBlendFuncExtended.enabled &&
                                    functions->standard == STANDARD_GL_DESKTOP &&
                                    functions->hasGLExtension("GL_ARB_blend_func_extended");
    if (extensions->blendFuncExtended)
    {
        // TODO(http://anglebug.com/1085): Support greater values of
        // MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT queried from the driver. See comments in ProgramGL.cpp
        // for more information about this limitation.
        extensions->maxDualSourceDrawBuffers = 1;
    }

    // EXT_float_blend
    // Assume all desktop driver supports this by default.
    extensions->floatBlend = functions->standard == STANDARD_GL_DESKTOP ||
                             functions->hasGLESExtension("GL_EXT_float_blend");

    // ANGLE_base_vertex_base_instance
    extensions->baseVertexBaseInstance =
        functions->isAtLeastGL(gl::Version(3, 2)) || functions->isAtLeastGLES(gl::Version(3, 2)) ||
        functions->hasGLESExtension("GL_OES_draw_elements_base_vertex") ||
        functions->hasGLESExtension("GL_EXT_draw_elements_base_vertex");

    // ANGLE_compressed_texture_etc
    // Expose this extension only when we support the formats or we're running on top of a native
    // ES driver.
    extensions->compressedTextureETC = functions->standard == STANDARD_GL_ES &&
                                       gl::DetermineCompressedTextureETCSupport(*textureCapsMap);

    // To work around broken unsized sRGB textures, sized sRGB textures are used. Disable EXT_sRGB
    // if those formats are not available.
    if (features.unsizedsRGBReadPixelsDoesntTransform.enabled &&
        !functions->isAtLeastGLES(gl::Version(3, 0)))
    {
        extensions->sRGB = false;
    }

    extensions->provokingVertex = functions->hasGLExtension("GL_ARB_provoking_vertex") ||
                                  functions->hasGLExtension("GL_EXT_provoking_vertex") ||
                                  functions->isAtLeastGL(gl::Version(3, 2));

    extensions->textureExternalUpdateANGLE = true;
    extensions->texture3DOES               = functions->isAtLeastGL(gl::Version(1, 2)) ||
                               functions->isAtLeastGLES(gl::Version(3, 0)) ||
                               functions->hasGLESExtension("GL_OES_texture_3D");

    extensions->memoryObject = functions->hasGLExtension("GL_EXT_memory_object") ||
                               functions->hasGLESExtension("GL_EXT_memory_object");
    extensions->semaphore = functions->hasGLExtension("GL_EXT_semaphore") ||
                            functions->hasGLESExtension("GL_EXT_semaphore");
    extensions->memoryObjectFd = functions->hasGLExtension("GL_EXT_memory_object_fd") ||
                                 functions->hasGLESExtension("GL_EXT_memory_object_fd");
    extensions->semaphoreFd = functions->hasGLExtension("GL_EXT_semaphore_fd") ||
                              functions->hasGLESExtension("GL_EXT_semaphore_fd");

    // GL_APPLE_clip_distance
    extensions->clipDistanceAPPLE = functions->isAtLeastGL(gl::Version(3, 0));
    if (extensions->clipDistanceAPPLE)
    {
        caps->maxClipDistances = QuerySingleGLInt(functions, GL_MAX_CLIP_DISTANCES_EXT);
    }
    else
    {
        caps->maxClipDistances = 0;
    }
}

void InitializeFeatures(const FunctionsGL *functions, angle::FeaturesGL *features)
{
    VendorID vendor = GetVendorID(functions);
    uint32_t device = GetDeviceID(functions);
    bool isAMD      = IsAMD(vendor);
    bool isIntel    = IsIntel(vendor);
    bool isNvidia   = IsNvidia(vendor);
    bool isQualcomm = IsQualcomm(vendor);

    // Don't use 1-bit alpha formats on desktop GL with AMD drivers.
    ANGLE_FEATURE_CONDITION(features, avoid1BitAlphaTextureFormats,
                            functions->standard == STANDARD_GL_DESKTOP && isAMD)

    ANGLE_FEATURE_CONDITION(features, rgba4IsNotSupportedForColorRendering,
                            functions->standard == STANDARD_GL_DESKTOP && isIntel)

    ANGLE_FEATURE_CONDITION(features, emulateAbsIntFunction, isIntel)

    ANGLE_FEATURE_CONDITION(features, addAndTrueToLoopCondition, isIntel)

    ANGLE_FEATURE_CONDITION(features, emulateIsnanFloat, isIntel)

    ANGLE_FEATURE_CONDITION(features, doesSRGBClearsOnLinearFramebufferAttachments,
                            functions->standard == STANDARD_GL_DESKTOP && (isIntel || isAMD))

    ANGLE_FEATURE_CONDITION(features, emulateMaxVertexAttribStride,
                            IsLinux() && functions->standard == STANDARD_GL_DESKTOP && isAMD)
    ANGLE_FEATURE_CONDITION(
        features, useUnusedBlocksWithStandardOrSharedLayout,
        (IsApple() && functions->standard == STANDARD_GL_DESKTOP) || (IsLinux() && isAMD))

    ANGLE_FEATURE_CONDITION(features, doWhileGLSLCausesGPUHang, IsApple())
    ANGLE_FEATURE_CONDITION(features, rewriteFloatUnaryMinusOperator, IsApple() && isIntel)

    ANGLE_FEATURE_CONDITION(features, addBaseVertexToVertexID, IsApple() && isAMD)

    // Triggers a bug on Marshmallow Adreno (4xx?) driver.
    // http://anglebug.com/2046
    ANGLE_FEATURE_CONDITION(features, dontInitializeUninitializedLocals, IsAndroid() && isQualcomm)

    ANGLE_FEATURE_CONDITION(features, finishDoesNotCauseQueriesToBeAvailable,
                            functions->standard == STANDARD_GL_DESKTOP && isNvidia)

    // TODO(cwallez): Disable this workaround for MacOSX versions 10.9 or later.
    ANGLE_FEATURE_CONDITION(features, alwaysCallUseProgramAfterLink, true)

    ANGLE_FEATURE_CONDITION(features, unpackOverlappingRowsSeparatelyUnpackBuffer, isNvidia)
    ANGLE_FEATURE_CONDITION(features, packOverlappingRowsSeparatelyPackBuffer, isNvidia)

    ANGLE_FEATURE_CONDITION(features, initializeCurrentVertexAttributes, isNvidia)

    ANGLE_FEATURE_CONDITION(features, unpackLastRowSeparatelyForPaddingInclusion,
                            IsApple() || isNvidia)
    ANGLE_FEATURE_CONDITION(features, packLastRowSeparatelyForPaddingInclusion,
                            IsApple() || isNvidia)

    ANGLE_FEATURE_CONDITION(features, removeInvariantAndCentroidForESSL3,
                            functions->isAtMostGL(gl::Version(4, 1)) ||
                                (functions->standard == STANDARD_GL_DESKTOP && isAMD))

    // TODO(oetuaho): Make this specific to the affected driver versions. Versions that came after
    // 364 are known to be affected, at least up to 375.
    ANGLE_FEATURE_CONDITION(features, emulateAtan2Float, isNvidia)

    ANGLE_FEATURE_CONDITION(features, reapplyUBOBindingsAfterUsingBinaryProgram,
                            isAMD || IsAndroid())

    ANGLE_FEATURE_CONDITION(features, rewriteVectorScalarArithmetic, isNvidia)

    // TODO(oetuaho): Make this specific to the affected driver versions. Versions at least up to
    // 390 are known to be affected. Versions after that are expected not to be affected.
    ANGLE_FEATURE_CONDITION(features, clampFragDepth, isNvidia)

    // TODO(oetuaho): Make this specific to the affected driver versions. Versions since 397.31 are
    // not affected.
    ANGLE_FEATURE_CONDITION(features, rewriteRepeatedAssignToSwizzled, isNvidia)

    // TODO(jmadill): Narrow workaround range for specific devices.

    ANGLE_FEATURE_CONDITION(features, clampPointSize, IsAndroid() || isNvidia)

    ANGLE_FEATURE_CONDITION(features, dontUseLoopsToInitializeVariables, IsAndroid() && !isNvidia)

    ANGLE_FEATURE_CONDITION(features, disableBlendFuncExtended, isAMD || isIntel)

    ANGLE_FEATURE_CONDITION(features, unsizedsRGBReadPixelsDoesntTransform,
                            IsAndroid() && isQualcomm)

    ANGLE_FEATURE_CONDITION(features, queryCounterBitsGeneratesErrors, IsNexus5X(vendor, device))

    ANGLE_FEATURE_CONDITION(features, dontRelinkProgramsInParallel,
                            IsAndroid() || (IsWindows() && isIntel))

    // TODO(jie.a.chen@intel.com): Clean up the bugs.
    // anglebug.com/3031
    // crbug.com/922936
    ANGLE_FEATURE_CONDITION(features, disableWorkerContexts,
                            (IsWindows() && (isIntel || isAMD)) || (IsLinux() && isNvidia))

    ANGLE_FEATURE_CONDITION(features, limitMaxTextureSizeTo4096,
                            IsAndroid() || (isIntel && IsLinux()))
    ANGLE_FEATURE_CONDITION(features, limitMaxMSAASamplesTo4, IsAndroid())
    ANGLE_FEATURE_CONDITION(features, limitMax3dArrayTextureSizeTo1024, isIntel && IsLinux())

    ANGLE_FEATURE_CONDITION(features, allowClearForRobustResourceInit, IsApple())

    // The WebGL conformance/uniforms/out-of-bounds-uniform-array-access test has been seen to fail
    // on AMD and Android devices.
    ANGLE_FEATURE_CONDITION(features, clampArrayAccess, IsAndroid() || isAMD)

#if defined(ANGLE_PLATFORM_MACOS)
    ANGLE_FEATURE_CONDITION(features, resetTexImage2DBaseLevel,
                            IsApple() && isIntel && GetMacOSVersion() >= OSVersion(10, 12, 4))

    ANGLE_FEATURE_CONDITION(features, clearToZeroOrOneBroken,
                            IsApple() && isIntel && GetMacOSVersion() < OSVersion(10, 12, 6))
#endif

    ANGLE_FEATURE_CONDITION(features, adjustSrcDstRegionBlitFramebuffer,
                            IsLinux() || (IsAndroid() && isNvidia) || (IsWindows() && isNvidia))

    ANGLE_FEATURE_CONDITION(features, clipSrcRegionBlitFramebuffer, IsApple())

    ANGLE_FEATURE_CONDITION(features, resettingTexturesGeneratesErrors,
                            IsApple() || (IsWindows() && isAMD))

    ANGLE_FEATURE_CONDITION(features, rgbDXT1TexturesSampleZeroAlpha, IsApple())

    ANGLE_FEATURE_CONDITION(features, unfoldShortCircuits, IsApple())
}

void InitializeFrontendFeatures(const FunctionsGL *functions, angle::FrontendFeatures *features)
{
    VendorID vendor = GetVendorID(functions);
    bool isIntel    = IsIntel(vendor);
    bool isQualcomm = IsQualcomm(vendor);

    ANGLE_FEATURE_CONDITION(features, disableProgramCachingForTransformFeedback,
                            IsAndroid() && isQualcomm)
    ANGLE_FEATURE_CONDITION(features, syncFramebufferBindingsOnTexImage, IsWindows() && isIntel)
}

}  // namespace nativegl_gl

namespace nativegl
{
bool SupportsDrawIndirect(const FunctionsGL *functions)
{
    return functions->isAtLeastGL(gl::Version(4, 0)) ||
           functions->hasGLExtension("GL_ARB_draw_indirect") ||
           functions->isAtLeastGLES(gl::Version(3, 1));
}
bool SupportsCompute(const FunctionsGL *functions)
{
    // OpenGL 4.2 is required for GL_ARB_compute_shader, some platform drivers have the extension,
    // but their maximum supported GL versions are less than 4.2. Explicitly limit the minimum
    // GL version to 4.2.
    return (functions->isAtLeastGL(gl::Version(4, 3)) ||
            functions->isAtLeastGLES(gl::Version(3, 1)) ||
            (functions->isAtLeastGL(gl::Version(4, 2)) &&
             functions->hasGLExtension("GL_ARB_compute_shader") &&
             functions->hasGLExtension("GL_ARB_shader_storage_buffer_object")));
}

bool SupportsFenceSync(const FunctionsGL *functions)
{
    return functions->isAtLeastGL(gl::Version(3, 2)) || functions->hasGLExtension("GL_ARB_sync") ||
           functions->isAtLeastGLES(gl::Version(3, 0));
}

bool SupportsOcclusionQueries(const FunctionsGL *functions)
{
    return functions->isAtLeastGL(gl::Version(1, 5)) ||
           functions->hasGLExtension("GL_ARB_occlusion_query2") ||
           functions->isAtLeastGLES(gl::Version(3, 0)) ||
           functions->hasGLESExtension("GL_EXT_occlusion_query_boolean");
}

bool SupportsNativeRendering(const FunctionsGL *functions,
                             gl::TextureType type,
                             GLenum internalFormat)
{
    // Some desktop drivers allow rendering to formats that are not required by the spec, this is
    // exposed through the GL_FRAMEBUFFER_RENDERABLE query.
    bool hasInternalFormatQuery = functions->isAtLeastGL(gl::Version(4, 3)) ||
                                  functions->hasGLExtension("GL_ARB_internalformat_query2");

    // Some Intel drivers have a bug that returns GL_FULL_SUPPORT when asked if they support
    // rendering to compressed texture formats yet return framebuffer incomplete when attempting to
    // render to the format.  Skip any native queries for compressed formats.
    const gl::InternalFormat &internalFormatInfo = gl::GetSizedInternalFormatInfo(internalFormat);

    if (hasInternalFormatQuery && !internalFormatInfo.compressed)
    {
        GLint framebufferRenderable = GL_NONE;
        functions->getInternalformativ(ToGLenum(type), internalFormat, GL_FRAMEBUFFER_RENDERABLE, 1,
                                       &framebufferRenderable);
        return framebufferRenderable != GL_NONE;
    }
    else
    {
        const nativegl::InternalFormat &nativeInfo =
            nativegl::GetInternalFormatInfo(internalFormat, functions->standard);
        return nativegl_gl::MeetsRequirements(functions, nativeInfo.textureAttachment);
    }
}

bool SupportsTexImage(gl::TextureType type)
{
    switch (type)
    {
            // Multi-sample texture types only support TexStorage data upload
        case gl::TextureType::_2DMultisample:
        case gl::TextureType::_2DMultisampleArray:
            return false;

        default:
            return true;
    }
}

bool UseTexImage2D(gl::TextureType textureType)
{
    return textureType == gl::TextureType::_2D || textureType == gl::TextureType::CubeMap ||
           textureType == gl::TextureType::Rectangle ||
           textureType == gl::TextureType::_2DMultisample ||
           textureType == gl::TextureType::External;
}

bool UseTexImage3D(gl::TextureType textureType)
{
    return textureType == gl::TextureType::_2DArray || textureType == gl::TextureType::_3D ||
           textureType == gl::TextureType::_2DMultisampleArray;
}

GLenum GetTextureBindingQuery(gl::TextureType textureType)
{
    switch (textureType)
    {
        case gl::TextureType::_2D:
            return GL_TEXTURE_BINDING_2D;
        case gl::TextureType::_2DArray:
            return GL_TEXTURE_BINDING_2D_ARRAY;
        case gl::TextureType::_2DMultisample:
            return GL_TEXTURE_BINDING_2D_MULTISAMPLE;
        case gl::TextureType::_2DMultisampleArray:
            return GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY;
        case gl::TextureType::_3D:
            return GL_TEXTURE_BINDING_3D;
        case gl::TextureType::External:
            return GL_TEXTURE_BINDING_EXTERNAL_OES;
        case gl::TextureType::Rectangle:
            return GL_TEXTURE_BINDING_RECTANGLE;
        case gl::TextureType::CubeMap:
            return GL_TEXTURE_BINDING_CUBE_MAP;
        default:
            UNREACHABLE();
            return 0;
    }
}

GLenum GetBufferBindingQuery(gl::BufferBinding bufferBinding)
{
    switch (bufferBinding)
    {
        case gl::BufferBinding::Array:
            return GL_ARRAY_BUFFER_BINDING;
        case gl::BufferBinding::AtomicCounter:
            return GL_ATOMIC_COUNTER_BUFFER_BINDING;
        case gl::BufferBinding::CopyRead:
            return GL_COPY_READ_BUFFER_BINDING;
        case gl::BufferBinding::CopyWrite:
            return GL_COPY_WRITE_BUFFER_BINDING;
        case gl::BufferBinding::DispatchIndirect:
            return GL_DISPATCH_INDIRECT_BUFFER_BINDING;
        case gl::BufferBinding::DrawIndirect:
            return GL_DRAW_INDIRECT_BUFFER_BINDING;
        case gl::BufferBinding::ElementArray:
            return GL_ELEMENT_ARRAY_BUFFER_BINDING;
        case gl::BufferBinding::PixelPack:
            return GL_PIXEL_PACK_BUFFER_BINDING;
        case gl::BufferBinding::PixelUnpack:
            return GL_PIXEL_UNPACK_BUFFER_BINDING;
        case gl::BufferBinding::ShaderStorage:
            return GL_SHADER_STORAGE_BUFFER_BINDING;
        case gl::BufferBinding::TransformFeedback:
            return GL_TRANSFORM_FEEDBACK_BUFFER_BINDING;
        case gl::BufferBinding::Uniform:
            return GL_UNIFORM_BUFFER_BINDING;
        default:
            UNREACHABLE();
            return 0;
    }
}

std::string GetBufferBindingString(gl::BufferBinding bufferBinding)
{
    std::ostringstream os;
    os << bufferBinding << "_BINDING";
    return os.str();
}

}  // namespace nativegl

const FunctionsGL *GetFunctionsGL(const gl::Context *context)
{
    return GetImplAs<ContextGL>(context)->getFunctions();
}

StateManagerGL *GetStateManagerGL(const gl::Context *context)
{
    return GetImplAs<ContextGL>(context)->getStateManager();
}

BlitGL *GetBlitGL(const gl::Context *context)
{
    return GetImplAs<ContextGL>(context)->getBlitter();
}

ClearMultiviewGL *GetMultiviewClearer(const gl::Context *context)
{
    return GetImplAs<ContextGL>(context)->getMultiviewClearer();
}

const angle::FeaturesGL &GetFeaturesGL(const gl::Context *context)
{
    return GetImplAs<ContextGL>(context)->getFeaturesGL();
}

void ClearErrors(const gl::Context *context,
                 const char *file,
                 const char *function,
                 unsigned int line)
{
    const FunctionsGL *functions = GetFunctionsGL(context);

    GLenum error = functions->getError();
    while (error != GL_NO_ERROR)
    {
        ERR() << "Preexisting GL error " << gl::FmtHex(error) << " as of " << file << ", "
              << function << ":" << line << ". ";
        error = functions->getError();
    }
}

angle::Result CheckError(const gl::Context *context,
                         const char *call,
                         const char *file,
                         const char *function,
                         unsigned int line)
{
    const FunctionsGL *functions = GetFunctionsGL(context);

    GLenum error = functions->getError();
    if (ANGLE_UNLIKELY(error != GL_NO_ERROR))
    {
        ContextGL *contextGL = GetImplAs<ContextGL>(context);
        contextGL->handleError(error, "Unexpected driver error.", file, function, line);
        ERR() << "GL call " << call << " generated error " << gl::FmtHex(error) << " in " << file
              << ", " << function << ":" << line << ". ";

        // Check that only one GL error was generated, ClearErrors should have been called first
        GLenum nextError = functions->getError();
        while (nextError != GL_NO_ERROR)
        {
            ERR() << "Additional GL error " << gl::FmtHex(nextError) << " generated.";
            nextError = functions->getError();
        }

        return angle::Result::Stop;
    }

    return angle::Result::Continue;
}

bool CanMapBufferForRead(const FunctionsGL *functions)
{
    return (functions->mapBufferRange != nullptr) ||
           (functions->mapBuffer != nullptr && functions->standard == STANDARD_GL_DESKTOP);
}

uint8_t *MapBufferRangeWithFallback(const FunctionsGL *functions,
                                    GLenum target,
                                    size_t offset,
                                    size_t length,
                                    GLbitfield access)
{
    if (functions->mapBufferRange != nullptr)
    {
        return static_cast<uint8_t *>(functions->mapBufferRange(target, offset, length, access));
    }
    else if (functions->mapBuffer != nullptr &&
             (functions->standard == STANDARD_GL_DESKTOP || access == GL_MAP_WRITE_BIT))
    {
        // Only the read and write bits are supported
        ASSERT((access & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT)) != 0);

        GLenum accessEnum = 0;
        if (access == (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT))
        {
            accessEnum = GL_READ_WRITE;
        }
        else if (access == GL_MAP_READ_BIT)
        {
            accessEnum = GL_READ_ONLY;
        }
        else if (access == GL_MAP_WRITE_BIT)
        {
            accessEnum = GL_WRITE_ONLY;
        }
        else
        {
            UNREACHABLE();
            return nullptr;
        }

        return static_cast<uint8_t *>(functions->mapBuffer(target, accessEnum)) + offset;
    }
    else
    {
        // No options available
        UNREACHABLE();
        return nullptr;
    }
}

angle::Result ShouldApplyLastRowPaddingWorkaround(ContextGL *contextGL,
                                                  const gl::Extents &size,
                                                  const gl::PixelStoreStateBase &state,
                                                  const gl::Buffer *pixelBuffer,
                                                  GLenum format,
                                                  GLenum type,
                                                  bool is3D,
                                                  const void *pixels,
                                                  bool *shouldApplyOut)
{
    if (pixelBuffer == nullptr)
    {
        *shouldApplyOut = false;
        return angle::Result::Continue;
    }

    // We are using an pack or unpack buffer, compute what the driver thinks is going to be the
    // last byte read or written. If it is past the end of the buffer, we will need to use the
    // workaround otherwise the driver will generate INVALID_OPERATION and not do the operation.

    const gl::InternalFormat &glFormat = gl::GetInternalFormatInfo(format, type);
    GLuint endByte                     = 0;
    ANGLE_CHECK_GL_MATH(contextGL,
                        glFormat.computePackUnpackEndByte(type, size, state, is3D, &endByte));
    GLuint rowPitch = 0;
    ANGLE_CHECK_GL_MATH(contextGL, glFormat.computeRowPitch(type, size.width, state.alignment,
                                                            state.rowLength, &rowPitch));

    CheckedNumeric<size_t> checkedPixelBytes = glFormat.computePixelBytes(type);
    CheckedNumeric<size_t> checkedEndByte =
        angle::CheckedNumeric<size_t>(endByte) + reinterpret_cast<intptr_t>(pixels);

    // At this point checkedEndByte is the actual last byte read.
    // The driver adds an extra row padding (if any), mimic it.
    ANGLE_CHECK_GL_MATH(contextGL, checkedPixelBytes.IsValid());
    if (checkedPixelBytes.ValueOrDie() * size.width < rowPitch)
    {
        checkedEndByte += rowPitch - checkedPixelBytes * size.width;
    }

    ANGLE_CHECK_GL_MATH(contextGL, checkedEndByte.IsValid());

    *shouldApplyOut = checkedEndByte.ValueOrDie() > static_cast<size_t>(pixelBuffer->getSize());
    return angle::Result::Continue;
}

std::vector<ContextCreationTry> GenerateContextCreationToTry(EGLint requestedType, bool isMesaGLX)
{
    using Type                         = ContextCreationTry::Type;
    constexpr EGLint kPlatformOpenGL   = EGL_PLATFORM_ANGLE_TYPE_OPENGL_ANGLE;
    constexpr EGLint kPlatformOpenGLES = EGL_PLATFORM_ANGLE_TYPE_OPENGLES_ANGLE;

    std::vector<ContextCreationTry> contextsToTry;

    if (requestedType == EGL_PLATFORM_ANGLE_TYPE_DEFAULT_ANGLE || requestedType == kPlatformOpenGL)
    {
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_CORE, gl::Version(4, 5));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_CORE, gl::Version(4, 4));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_CORE, gl::Version(4, 3));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_CORE, gl::Version(4, 2));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_CORE, gl::Version(4, 1));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_CORE, gl::Version(4, 0));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_CORE, gl::Version(3, 3));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_CORE, gl::Version(3, 2));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_LEGACY, gl::Version(3, 3));

        // On Mesa, do not try to create OpenGL context versions between 3.0 and
        // 3.2 because of compatibility problems. See crbug.com/659030
        if (!isMesaGLX)
        {
            contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_LEGACY, gl::Version(3, 2));
            contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_LEGACY, gl::Version(3, 1));
            contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_LEGACY, gl::Version(3, 0));
        }

        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_LEGACY, gl::Version(2, 1));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_LEGACY, gl::Version(2, 0));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_LEGACY, gl::Version(1, 5));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_LEGACY, gl::Version(1, 4));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_LEGACY, gl::Version(1, 3));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_LEGACY, gl::Version(1, 2));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_LEGACY, gl::Version(1, 1));
        contextsToTry.emplace_back(kPlatformOpenGL, Type::DESKTOP_LEGACY, gl::Version(1, 0));
    }

    if (requestedType == EGL_PLATFORM_ANGLE_TYPE_DEFAULT_ANGLE ||
        requestedType == kPlatformOpenGLES)
    {
        contextsToTry.emplace_back(kPlatformOpenGLES, Type::ES, gl::Version(3, 2));
        contextsToTry.emplace_back(kPlatformOpenGLES, Type::ES, gl::Version(3, 1));
        contextsToTry.emplace_back(kPlatformOpenGLES, Type::ES, gl::Version(3, 0));
        contextsToTry.emplace_back(kPlatformOpenGLES, Type::ES, gl::Version(2, 0));
    }

    return contextsToTry;
}
}  // namespace rx
