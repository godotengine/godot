//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// validationES.h: Validation functions for generic OpenGL ES entry point parameters

#include "libANGLE/validationES.h"

#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/ErrorStrings.h"
#include "libANGLE/Framebuffer.h"
#include "libANGLE/FramebufferAttachment.h"
#include "libANGLE/Image.h"
#include "libANGLE/Program.h"
#include "libANGLE/Query.h"
#include "libANGLE/Texture.h"
#include "libANGLE/TransformFeedback.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/formatutils.h"
#include "libANGLE/queryconversions.h"
#include "libANGLE/queryutils.h"
#include "libANGLE/validationES2.h"
#include "libANGLE/validationES3.h"

#include "common/mathutil.h"
#include "common/utilities.h"

using namespace angle;

namespace gl
{
using namespace err;

namespace
{
bool CompressedTextureFormatRequiresExactSize(GLenum internalFormat)
{
    // List of compressed format that require that the texture size is smaller than or a multiple of
    // the compressed block size.
    switch (internalFormat)
    {
        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT3_ANGLE:
        case GL_COMPRESSED_RGBA_S3TC_DXT5_ANGLE:
        case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT:
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT:
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:
        case GL_ETC1_RGB8_LOSSY_DECODE_ANGLE:
        case GL_COMPRESSED_RGB8_LOSSY_DECODE_ETC2_ANGLE:
        case GL_COMPRESSED_SRGB8_LOSSY_DECODE_ETC2_ANGLE:
        case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_LOSSY_DECODE_ETC2_ANGLE:
        case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_LOSSY_DECODE_ETC2_ANGLE:
        case GL_COMPRESSED_RGBA8_LOSSY_DECODE_ETC2_EAC_ANGLE:
        case GL_COMPRESSED_SRGB8_ALPHA8_LOSSY_DECODE_ETC2_EAC_ANGLE:
        case GL_COMPRESSED_RGBA_BPTC_UNORM_EXT:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT:
            return true;

        default:
            return false;
    }
}
bool CompressedSubTextureFormatRequiresExactSize(GLenum internalFormat)
{
    // Compressed sub textures have additional formats that requires exact size.
    // ES 3.1, Section 8.7, Page 171
    return CompressedTextureFormatRequiresExactSize(internalFormat) ||
           IsETC2EACFormat(internalFormat);
}

bool DifferenceCanOverflow(GLint a, GLint b)
{
    CheckedNumeric<GLint> checkedA(a);
    checkedA -= b;
    // Use negation to make sure that the difference can't overflow regardless of the order.
    checkedA = -checkedA;
    return !checkedA.IsValid();
}

bool ValidReadPixelsTypeEnum(Context *context, GLenum type)
{
    switch (type)
    {
        // Types referenced in Table 3.4 of the ES 2.0.25 spec
        case GL_UNSIGNED_BYTE:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_5_6_5:
            return context->getClientVersion() >= ES_2_0;

        // Types referenced in Table 3.2 of the ES 3.0.5 spec (Except depth stencil)
        case GL_BYTE:
        case GL_INT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_24_8:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_SHORT:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV_EXT:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV_EXT:
            return context->getClientVersion() >= ES_3_0;

        case GL_FLOAT:
            return context->getClientVersion() >= ES_3_0 || context->getExtensions().textureFloat ||
                   context->getExtensions().colorBufferHalfFloat;

        case GL_HALF_FLOAT:
            return context->getClientVersion() >= ES_3_0 ||
                   context->getExtensions().textureHalfFloat;

        case GL_HALF_FLOAT_OES:
            return context->getExtensions().colorBufferHalfFloat;

        default:
            return false;
    }
}

bool ValidReadPixelsFormatEnum(Context *context, GLenum format)
{
    switch (format)
    {
        // Formats referenced in Table 3.4 of the ES 2.0.25 spec (Except luminance)
        case GL_RGBA:
        case GL_RGB:
        case GL_ALPHA:
            return context->getClientVersion() >= ES_2_0;

        // Formats referenced in Table 3.2 of the ES 3.0.5 spec
        case GL_RG:
        case GL_RED:
        case GL_RGBA_INTEGER:
        case GL_RGB_INTEGER:
        case GL_RG_INTEGER:
        case GL_RED_INTEGER:
            return context->getClientVersion() >= ES_3_0;

        case GL_SRGB_ALPHA_EXT:
        case GL_SRGB_EXT:
            return context->getExtensions().sRGB;

        case GL_BGRA_EXT:
            return context->getExtensions().readFormatBGRA;

        default:
            return false;
    }
}

bool ValidReadPixelsFormatType(Context *context,
                               GLenum framebufferComponentType,
                               GLenum format,
                               GLenum type)
{
    switch (framebufferComponentType)
    {
        case GL_UNSIGNED_NORMALIZED:
            // TODO(geofflang): Don't accept BGRA here.  Some chrome internals appear to try to use
            // ReadPixels with BGRA even if the extension is not present
            return (format == GL_RGBA && type == GL_UNSIGNED_BYTE) ||
                   (context->getExtensions().readFormatBGRA && format == GL_BGRA_EXT &&
                    type == GL_UNSIGNED_BYTE);

        case GL_SIGNED_NORMALIZED:
            return (format == GL_RGBA && type == GL_UNSIGNED_BYTE);

        case GL_INT:
            return (format == GL_RGBA_INTEGER && type == GL_INT);

        case GL_UNSIGNED_INT:
            return (format == GL_RGBA_INTEGER && type == GL_UNSIGNED_INT);

        case GL_FLOAT:
            return (format == GL_RGBA && type == GL_FLOAT);

        default:
            UNREACHABLE();
            return false;
    }
}

template <typename ParamType>
bool ValidateTextureWrapModeValue(Context *context,
                                  const ParamType *params,
                                  bool restrictedWrapModes)
{
    switch (ConvertToGLenum(params[0]))
    {
        case GL_CLAMP_TO_EDGE:
            break;

        case GL_CLAMP_TO_BORDER:
            if (!context->getExtensions().textureBorderClamp)
            {
                context->validationError(GL_INVALID_ENUM, kExtensionNotEnabled);
                return false;
            }
            break;

        case GL_REPEAT:
        case GL_MIRRORED_REPEAT:
            if (restrictedWrapModes)
            {
                // OES_EGL_image_external and ANGLE_texture_rectangle specifies this error.
                context->validationError(GL_INVALID_ENUM, kInvalidWrapModeTexture);
                return false;
            }
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kInvalidTextureWrap);
            return false;
    }

    return true;
}

template <typename ParamType>
bool ValidateTextureMinFilterValue(Context *context,
                                   const ParamType *params,
                                   bool restrictedMinFilter)
{
    switch (ConvertToGLenum(params[0]))
    {
        case GL_NEAREST:
        case GL_LINEAR:
            break;

        case GL_NEAREST_MIPMAP_NEAREST:
        case GL_LINEAR_MIPMAP_NEAREST:
        case GL_NEAREST_MIPMAP_LINEAR:
        case GL_LINEAR_MIPMAP_LINEAR:
            if (restrictedMinFilter)
            {
                // OES_EGL_image_external specifies this error.
                context->validationError(GL_INVALID_ENUM, kInvalidFilterTexture);
                return false;
            }
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kInvalidTextureFilterParam);
            return false;
    }

    return true;
}

template <typename ParamType>
bool ValidateTextureMagFilterValue(Context *context, const ParamType *params)
{
    switch (ConvertToGLenum(params[0]))
    {
        case GL_NEAREST:
        case GL_LINEAR:
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kInvalidTextureFilterParam);
            return false;
    }

    return true;
}

template <typename ParamType>
bool ValidateTextureCompareModeValue(Context *context, const ParamType *params)
{
    // Acceptable mode parameters from GLES 3.0.2 spec, table 3.17
    switch (ConvertToGLenum(params[0]))
    {
        case GL_NONE:
        case GL_COMPARE_REF_TO_TEXTURE:
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kUnknownParameter);
            return false;
    }

    return true;
}

template <typename ParamType>
bool ValidateTextureCompareFuncValue(Context *context, const ParamType *params)
{
    // Acceptable function parameters from GLES 3.0.2 spec, table 3.17
    switch (ConvertToGLenum(params[0]))
    {
        case GL_LEQUAL:
        case GL_GEQUAL:
        case GL_LESS:
        case GL_GREATER:
        case GL_EQUAL:
        case GL_NOTEQUAL:
        case GL_ALWAYS:
        case GL_NEVER:
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kUnknownParameter);
            return false;
    }

    return true;
}

template <typename ParamType>
bool ValidateTextureSRGBDecodeValue(Context *context, const ParamType *params)
{
    if (!context->getExtensions().textureSRGBDecode)
    {
        context->validationError(GL_INVALID_ENUM, kExtensionNotEnabled);
        return false;
    }

    switch (ConvertToGLenum(params[0]))
    {
        case GL_DECODE_EXT:
        case GL_SKIP_DECODE_EXT:
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kUnknownParameter);
            return false;
    }

    return true;
}

bool ValidateTextureMaxAnisotropyExtensionEnabled(Context *context)
{
    if (!context->getExtensions().textureFilterAnisotropic)
    {
        context->validationError(GL_INVALID_ENUM, kExtensionNotEnabled);
        return false;
    }

    return true;
}

bool ValidateTextureMaxAnisotropyValue(Context *context, GLfloat paramValue)
{
    if (!ValidateTextureMaxAnisotropyExtensionEnabled(context))
    {
        return false;
    }

    GLfloat largest = context->getExtensions().maxTextureAnisotropy;

    if (paramValue < 1 || paramValue > largest)
    {
        context->validationError(GL_INVALID_VALUE, kOutsideOfBounds);
        return false;
    }

    return true;
}

bool ValidateFragmentShaderColorBufferMaskMatch(Context *context)
{
    const Program *program         = context->getState().getLinkedProgram(context);
    const Framebuffer *framebuffer = context->getState().getDrawFramebuffer();

    auto drawBufferMask     = framebuffer->getDrawBufferMask().to_ulong();
    auto fragmentOutputMask = program->getActiveOutputVariables().to_ulong();

    return drawBufferMask == (drawBufferMask & fragmentOutputMask);
}

bool ValidateFragmentShaderColorBufferTypeMatch(Context *context)
{
    const Program *program         = context->getState().getLinkedProgram(context);
    const Framebuffer *framebuffer = context->getState().getDrawFramebuffer();

    return ValidateComponentTypeMasks(program->getDrawBufferTypeMask().to_ulong(),
                                      framebuffer->getDrawBufferTypeMask().to_ulong(),
                                      program->getActiveOutputVariables().to_ulong(),
                                      framebuffer->getDrawBufferMask().to_ulong());
}

bool ValidateVertexShaderAttributeTypeMatch(Context *context)
{
    const auto &glState    = context->getState();
    const Program *program = context->getState().getLinkedProgram(context);
    const VertexArray *vao = context->getState().getVertexArray();

    unsigned long stateCurrentValuesTypeBits = glState.getCurrentValuesTypeMask().to_ulong();
    unsigned long vaoAttribTypeBits          = vao->getAttributesTypeMask().to_ulong();
    unsigned long vaoAttribEnabledMask       = vao->getAttributesMask().to_ulong();

    vaoAttribEnabledMask |= vaoAttribEnabledMask << kMaxComponentTypeMaskIndex;
    vaoAttribTypeBits = (vaoAttribEnabledMask & vaoAttribTypeBits);
    vaoAttribTypeBits |= (~vaoAttribEnabledMask & stateCurrentValuesTypeBits);

    return ValidateComponentTypeMasks(program->getAttributesTypeMask().to_ulong(),
                                      vaoAttribTypeBits, program->getAttributesMask().to_ulong(),
                                      0xFFFF);
}

bool IsCompatibleDrawModeWithGeometryShader(PrimitiveMode drawMode,
                                            PrimitiveMode geometryShaderInputPrimitiveType)
{
    // [EXT_geometry_shader] Section 11.1gs.1, Geometry Shader Input Primitives
    switch (drawMode)
    {
        case PrimitiveMode::Points:
            return geometryShaderInputPrimitiveType == PrimitiveMode::Points;
        case PrimitiveMode::Lines:
        case PrimitiveMode::LineStrip:
        case PrimitiveMode::LineLoop:
            return geometryShaderInputPrimitiveType == PrimitiveMode::Lines;
        case PrimitiveMode::LinesAdjacency:
        case PrimitiveMode::LineStripAdjacency:
            return geometryShaderInputPrimitiveType == PrimitiveMode::LinesAdjacency;
        case PrimitiveMode::Triangles:
        case PrimitiveMode::TriangleFan:
        case PrimitiveMode::TriangleStrip:
            return geometryShaderInputPrimitiveType == PrimitiveMode::Triangles;
        case PrimitiveMode::TrianglesAdjacency:
        case PrimitiveMode::TriangleStripAdjacency:
            return geometryShaderInputPrimitiveType == PrimitiveMode::TrianglesAdjacency;
        default:
            UNREACHABLE();
            return false;
    }
}

// GLES1 texture parameters are a small subset of the others
bool IsValidGLES1TextureParameter(GLenum pname)
{
    switch (pname)
    {
        case GL_TEXTURE_MAG_FILTER:
        case GL_TEXTURE_MIN_FILTER:
        case GL_TEXTURE_WRAP_S:
        case GL_TEXTURE_WRAP_T:
        case GL_TEXTURE_WRAP_R:
        case GL_GENERATE_MIPMAP:
        case GL_TEXTURE_CROP_RECT_OES:
            return true;
        default:
            return false;
    }
}

unsigned int GetSamplerParameterCount(GLenum pname)
{
    return pname == GL_TEXTURE_BORDER_COLOR ? 4 : 1;
}

}  // anonymous namespace

void SetRobustLengthParam(GLsizei *length, GLsizei value)
{
    if (length)
    {
        *length = value;
    }
}

bool IsETC2EACFormat(const GLenum format)
{
    // ES 3.1, Table 8.19
    switch (format)
    {
        case GL_COMPRESSED_R11_EAC:
        case GL_COMPRESSED_SIGNED_R11_EAC:
        case GL_COMPRESSED_RG11_EAC:
        case GL_COMPRESSED_SIGNED_RG11_EAC:
        case GL_COMPRESSED_RGB8_ETC2:
        case GL_COMPRESSED_SRGB8_ETC2:
        case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_RGBA8_ETC2_EAC:
        case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
            return true;

        default:
            return false;
    }
}

bool ValidTextureTarget(const Context *context, TextureType type)
{
    switch (type)
    {
        case TextureType::_2D:
        case TextureType::CubeMap:
            return true;

        case TextureType::Rectangle:
            return context->getExtensions().textureRectangle;

        case TextureType::_3D:
            return ((context->getClientMajorVersion() >= 3) ||
                    context->getExtensions().texture3DOES);

        case TextureType::_2DArray:
            return (context->getClientMajorVersion() >= 3);

        case TextureType::_2DMultisample:
            return (context->getClientVersion() >= Version(3, 1) ||
                    context->getExtensions().textureMultisample);
        case TextureType::_2DMultisampleArray:
            return context->getExtensions().textureStorageMultisample2DArray;

        default:
            return false;
    }
}

bool ValidTexture2DTarget(const Context *context, TextureType type)
{
    switch (type)
    {
        case TextureType::_2D:
        case TextureType::CubeMap:
            return true;

        case TextureType::Rectangle:
            return context->getExtensions().textureRectangle;

        default:
            return false;
    }
}

bool ValidTexture3DTarget(const Context *context, TextureType target)
{
    switch (target)
    {
        case TextureType::_3D:
        case TextureType::_2DArray:
            return (context->getClientMajorVersion() >= 3);

        default:
            return false;
    }
}

// Most texture GL calls are not compatible with external textures, so we have a separate validation
// function for use in the GL calls that do
bool ValidTextureExternalTarget(const Context *context, TextureType target)
{
    return (target == TextureType::External) &&
           (context->getExtensions().eglImageExternal ||
            context->getExtensions().eglStreamConsumerExternal);
}

bool ValidTextureExternalTarget(const Context *context, TextureTarget target)
{
    return (target == TextureTarget::External) &&
           ValidTextureExternalTarget(context, TextureType::External);
}

// This function differs from ValidTextureTarget in that the target must be
// usable as the destination of a 2D operation-- so a cube face is valid, but
// GL_TEXTURE_CUBE_MAP is not.
// Note: duplicate of IsInternalTextureTarget
bool ValidTexture2DDestinationTarget(const Context *context, TextureTarget target)
{
    switch (target)
    {
        case TextureTarget::_2D:
        case TextureTarget::CubeMapNegativeX:
        case TextureTarget::CubeMapNegativeY:
        case TextureTarget::CubeMapNegativeZ:
        case TextureTarget::CubeMapPositiveX:
        case TextureTarget::CubeMapPositiveY:
        case TextureTarget::CubeMapPositiveZ:
            return true;
        case TextureTarget::Rectangle:
            return context->getExtensions().textureRectangle;
        default:
            return false;
    }
}

bool ValidateTransformFeedbackPrimitiveMode(const Context *context,
                                            PrimitiveMode transformFeedbackPrimitiveMode,
                                            PrimitiveMode renderPrimitiveMode)
{
    ASSERT(context);

    if (!context->getExtensions().geometryShader)
    {
        // It is an invalid operation to call DrawArrays or DrawArraysInstanced with a draw mode
        // that does not match the current transform feedback object's draw mode (if transform
        // feedback is active), (3.0.2, section 2.14, pg 86)
        return transformFeedbackPrimitiveMode == renderPrimitiveMode;
    }

    // [GL_EXT_geometry_shader] Table 12.1gs
    switch (renderPrimitiveMode)
    {
        case PrimitiveMode::Points:
            return transformFeedbackPrimitiveMode == PrimitiveMode::Points;
        case PrimitiveMode::Lines:
        case PrimitiveMode::LineStrip:
        case PrimitiveMode::LineLoop:
            return transformFeedbackPrimitiveMode == PrimitiveMode::Lines;
        case PrimitiveMode::Triangles:
        case PrimitiveMode::TriangleFan:
        case PrimitiveMode::TriangleStrip:
            return transformFeedbackPrimitiveMode == PrimitiveMode::Triangles;
        default:
            UNREACHABLE();
            return false;
    }
}

bool ValidateDrawElementsInstancedBase(Context *context,
                                       PrimitiveMode mode,
                                       GLsizei count,
                                       DrawElementsType type,
                                       const GLvoid *indices,
                                       GLsizei primcount)
{
    if (primcount <= 0)
    {
        if (primcount < 0)
        {
            context->validationError(GL_INVALID_VALUE, kNegativePrimcount);
            return false;
        }

        // Early exit.
        return ValidateDrawElementsCommon(context, mode, count, type, indices, primcount);
    }

    if (!ValidateDrawElementsCommon(context, mode, count, type, indices, primcount))
    {
        return false;
    }

    if (count == 0)
    {
        // Early exit.
        return true;
    }

    return ValidateDrawInstancedAttribs(context, primcount);
}

bool ValidateDrawArraysInstancedBase(Context *context,
                                     PrimitiveMode mode,
                                     GLint first,
                                     GLsizei count,
                                     GLsizei primcount)
{
    if (primcount <= 0)
    {
        if (primcount < 0)
        {
            context->validationError(GL_INVALID_VALUE, kNegativePrimcount);
            return false;
        }

        // Early exit.
        return ValidateDrawArraysCommon(context, mode, first, count, primcount);
    }

    if (!ValidateDrawArraysCommon(context, mode, first, count, primcount))
    {
        return false;
    }

    if (count == 0)
    {
        // Early exit.
        return true;
    }

    return ValidateDrawInstancedAttribs(context, primcount);
}

bool ValidateDrawInstancedANGLE(Context *context)
{
    // Verify there is at least one active attribute with a divisor of zero
    const State &state = context->getState();

    Program *program = state.getLinkedProgram(context);

    const auto &attribs  = state.getVertexArray()->getVertexAttributes();
    const auto &bindings = state.getVertexArray()->getVertexBindings();
    for (size_t attributeIndex = 0; attributeIndex < MAX_VERTEX_ATTRIBS; attributeIndex++)
    {
        const VertexAttribute &attrib = attribs[attributeIndex];
        const VertexBinding &binding  = bindings[attrib.bindingIndex];
        if (program->isAttribLocationActive(attributeIndex) && binding.getDivisor() == 0)
        {
            return true;
        }
    }

    context->validationError(GL_INVALID_OPERATION, kNoZeroDivisor);
    return false;
}

bool ValidTexture3DDestinationTarget(const Context *context, TextureTarget target)
{
    switch (target)
    {
        case TextureTarget::_3D:
        case TextureTarget::_2DArray:
            return true;
        default:
            return false;
    }
}

bool ValidTexLevelDestinationTarget(const Context *context, TextureType type)
{
    switch (type)
    {
        case TextureType::_2D:
        case TextureType::_2DArray:
        case TextureType::_2DMultisample:
        case TextureType::CubeMap:
        case TextureType::_3D:
            return true;
        case TextureType::Rectangle:
            return context->getExtensions().textureRectangle;
        case TextureType::_2DMultisampleArray:
            return context->getExtensions().textureStorageMultisample2DArray;
        default:
            return false;
    }
}

bool ValidFramebufferTarget(const Context *context, GLenum target)
{
    static_assert(GL_DRAW_FRAMEBUFFER_ANGLE == GL_DRAW_FRAMEBUFFER &&
                      GL_READ_FRAMEBUFFER_ANGLE == GL_READ_FRAMEBUFFER,
                  "ANGLE framebuffer enums must equal the ES3 framebuffer enums.");

    switch (target)
    {
        case GL_FRAMEBUFFER:
            return true;

        case GL_READ_FRAMEBUFFER:
        case GL_DRAW_FRAMEBUFFER:
            return (context->getExtensions().framebufferBlit ||
                    context->getClientMajorVersion() >= 3);

        default:
            return false;
    }
}

bool ValidMipLevel(const Context *context, TextureType type, GLint level)
{
    const auto &caps    = context->getCaps();
    size_t maxDimension = 0;
    switch (type)
    {
        case TextureType::_2D:
        case TextureType::_2DArray:
        case TextureType::_2DMultisample:
        case TextureType::_2DMultisampleArray:
            // TODO(http://anglebug.com/2775): It's a bit unclear what the "maximum allowable
            // level-of-detail" for multisample textures should be. Could maybe make it zero.
            maxDimension = caps.max2DTextureSize;
            break;

        case TextureType::CubeMap:
            maxDimension = caps.maxCubeMapTextureSize;
            break;

        case TextureType::External:
        case TextureType::Rectangle:
            return level == 0;

        case TextureType::_3D:
            maxDimension = caps.max3DTextureSize;
            break;

        default:
            UNREACHABLE();
    }

    return level <= log2(static_cast<int>(maxDimension)) && level >= 0;
}

bool ValidImageSizeParameters(Context *context,
                              TextureType target,
                              GLint level,
                              GLsizei width,
                              GLsizei height,
                              GLsizei depth,
                              bool isSubImage)
{
    if (width < 0 || height < 0 || depth < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeSize);
        return false;
    }
    // TexSubImage parameters can be NPOT without textureNPOT extension,
    // as long as the destination texture is POT.
    bool hasNPOTSupport =
        context->getExtensions().textureNPOT || context->getClientVersion() >= Version(3, 0);
    if (!isSubImage && !hasNPOTSupport &&
        (level != 0 && (!isPow2(width) || !isPow2(height) || !isPow2(depth))))
    {
        context->validationError(GL_INVALID_VALUE, kTextureNotPow2);
        return false;
    }

    if (!ValidMipLevel(context, target, level))
    {
        context->validationError(GL_INVALID_VALUE, kInvalidMipLevel);
        return false;
    }

    return true;
}

bool ValidCompressedDimension(GLsizei size, GLuint blockSize, bool smallerThanBlockSizeAllowed)
{
    return (smallerThanBlockSizeAllowed && (size > 0) && (blockSize % size == 0)) ||
           (size % blockSize == 0);
}

bool ValidCompressedImageSize(const Context *context,
                              GLenum internalFormat,
                              GLint level,
                              GLsizei width,
                              GLsizei height,
                              GLsizei depth)
{
    const InternalFormat &formatInfo = GetSizedInternalFormatInfo(internalFormat);
    if (!formatInfo.compressed)
    {
        return false;
    }

    if (width < 0 || height < 0)
    {
        return false;
    }

    if (CompressedTextureFormatRequiresExactSize(internalFormat))
    {
        // The ANGLE extensions allow specifying compressed textures with sizes smaller than the
        // block size for level 0 but WebGL disallows this.
        bool smallerThanBlockSizeAllowed =
            level > 0 || !context->getExtensions().webglCompatibility;

        if (!ValidCompressedDimension(width, formatInfo.compressedBlockWidth,
                                      smallerThanBlockSizeAllowed) ||
            !ValidCompressedDimension(height, formatInfo.compressedBlockHeight,
                                      smallerThanBlockSizeAllowed) ||
            !ValidCompressedDimension(depth, formatInfo.compressedBlockDepth,
                                      smallerThanBlockSizeAllowed))
        {
            return false;
        }
    }

    return true;
}

bool ValidCompressedSubImageSize(const Context *context,
                                 GLenum internalFormat,
                                 GLint xoffset,
                                 GLint yoffset,
                                 GLint zoffset,
                                 GLsizei width,
                                 GLsizei height,
                                 GLsizei depth,
                                 size_t textureWidth,
                                 size_t textureHeight,
                                 size_t textureDepth)
{
    const InternalFormat &formatInfo = GetSizedInternalFormatInfo(internalFormat);
    if (!formatInfo.compressed)
    {
        return false;
    }

    if (xoffset < 0 || yoffset < 0 || zoffset < 0 || width < 0 || height < 0 || depth < 0)
    {
        return false;
    }

    bool fillsEntireMip =
        xoffset == 0 && yoffset == 0 && static_cast<size_t>(width) == textureWidth &&
        static_cast<size_t>(height) == textureHeight && static_cast<size_t>(depth) == textureDepth;

    if (CompressedFormatRequiresWholeImage(internalFormat))
    {
        return fillsEntireMip;
    }

    if (CompressedSubTextureFormatRequiresExactSize(internalFormat))
    {
        if (xoffset % formatInfo.compressedBlockWidth != 0 ||
            yoffset % formatInfo.compressedBlockHeight != 0 ||
            zoffset % formatInfo.compressedBlockDepth != 0)
        {
            return false;
        }

        // Allowed to either have data that is a multiple of block size or is smaller than the block
        // size but fills the entire mip
        bool sizeMultipleOfBlockSize = (width % formatInfo.compressedBlockWidth) == 0 &&
                                       (height % formatInfo.compressedBlockHeight) == 0 &&
                                       (depth % formatInfo.compressedBlockDepth) == 0;
        if (!sizeMultipleOfBlockSize && !fillsEntireMip)
        {
            return false;
        }
    }

    return true;
}

bool ValidImageDataSize(Context *context,
                        TextureType texType,
                        GLsizei width,
                        GLsizei height,
                        GLsizei depth,
                        GLenum format,
                        GLenum type,
                        const void *pixels,
                        GLsizei imageSize)
{
    Buffer *pixelUnpackBuffer = context->getState().getTargetBuffer(BufferBinding::PixelUnpack);
    if (pixelUnpackBuffer == nullptr && imageSize < 0)
    {
        // Checks are not required
        return true;
    }

    // ...the data would be unpacked from the buffer object such that the memory reads required
    // would exceed the data store size.
    const InternalFormat &formatInfo = GetInternalFormatInfo(format, type);
    ASSERT(formatInfo.internalFormat != GL_NONE);
    const Extents size(width, height, depth);
    const auto &unpack = context->getState().getUnpackState();

    bool targetIs3D = texType == TextureType::_3D || texType == TextureType::_2DArray;
    GLuint endByte  = 0;
    if (!formatInfo.computePackUnpackEndByte(type, size, unpack, targetIs3D, &endByte))
    {
        context->validationError(GL_INVALID_OPERATION, kIntegerOverflow);
        return false;
    }

    if (pixelUnpackBuffer)
    {
        CheckedNumeric<size_t> checkedEndByte(endByte);
        CheckedNumeric<size_t> checkedOffset(reinterpret_cast<size_t>(pixels));
        checkedEndByte += checkedOffset;

        if (!checkedEndByte.IsValid() ||
            (checkedEndByte.ValueOrDie() > static_cast<size_t>(pixelUnpackBuffer->getSize())))
        {
            // Overflow past the end of the buffer
            context->validationError(GL_INVALID_OPERATION, kIntegerOverflow);
            return false;
        }
        if (context->getExtensions().webglCompatibility &&
            pixelUnpackBuffer->isBoundForTransformFeedbackAndOtherUse())
        {
            context->validationError(GL_INVALID_OPERATION,
                                     kPixelUnpackBufferBoundForTransformFeedback);
            return false;
        }
    }
    else
    {
        ASSERT(imageSize >= 0);
        if (pixels == nullptr && imageSize != 0)
        {
            context->validationError(GL_INVALID_OPERATION, kImageSizeMustBeZero);
            return false;
        }

        if (pixels != nullptr && endByte > static_cast<GLuint>(imageSize))
        {
            context->validationError(GL_INVALID_OPERATION, kImageSizeTooSmall);
            return false;
        }
    }

    return true;
}

bool ValidQueryType(const Context *context, QueryType queryType)
{
    switch (queryType)
    {
        case QueryType::AnySamples:
        case QueryType::AnySamplesConservative:
            return context->getClientMajorVersion() >= 3 ||
                   context->getExtensions().occlusionQueryBoolean;
        case QueryType::TransformFeedbackPrimitivesWritten:
            return (context->getClientMajorVersion() >= 3);
        case QueryType::TimeElapsed:
            return context->getExtensions().disjointTimerQuery;
        case QueryType::CommandsCompleted:
            return context->getExtensions().syncQuery;
        case QueryType::PrimitivesGenerated:
            return context->getExtensions().geometryShader;
        default:
            return false;
    }
}

bool ValidateWebGLVertexAttribPointer(Context *context,
                                      VertexAttribType type,
                                      GLboolean normalized,
                                      GLsizei stride,
                                      const void *ptr,
                                      bool pureInteger)
{
    ASSERT(context->getExtensions().webglCompatibility);
    // WebGL 1.0 [Section 6.11] Vertex Attribute Data Stride
    // The WebGL API supports vertex attribute data strides up to 255 bytes. A call to
    // vertexAttribPointer will generate an INVALID_VALUE error if the value for the stride
    // parameter exceeds 255.
    constexpr GLsizei kMaxWebGLStride = 255;
    if (stride > kMaxWebGLStride)
    {
        context->validationError(GL_INVALID_VALUE, kStrideExceedsWebGLLimit);
        return false;
    }

    // WebGL 1.0 [Section 6.4] Buffer Offset and Stride Requirements
    // The offset arguments to drawElements and vertexAttribPointer, and the stride argument to
    // vertexAttribPointer, must be a multiple of the size of the data type passed to the call,
    // or an INVALID_OPERATION error is generated.
    angle::FormatID internalType = GetVertexFormatID(type, normalized, 1, pureInteger);
    size_t typeSize              = GetVertexFormatSize(internalType);

    ASSERT(isPow2(typeSize) && typeSize > 0);
    size_t sizeMask = (typeSize - 1);
    if ((reinterpret_cast<intptr_t>(ptr) & sizeMask) != 0)
    {
        context->validationError(GL_INVALID_OPERATION, kOffsetMustBeMultipleOfType);
        return false;
    }

    if ((stride & sizeMask) != 0)
    {
        context->validationError(GL_INVALID_OPERATION, kStrideMustBeMultipleOfType);
        return false;
    }

    return true;
}

Program *GetValidProgramNoResolve(Context *context, ShaderProgramID id)
{
    // ES3 spec (section 2.11.1) -- "Commands that accept shader or program object names will
    // generate the error INVALID_VALUE if the provided name is not the name of either a shader
    // or program object and INVALID_OPERATION if the provided name identifies an object
    // that is not the expected type."

    Program *validProgram = context->getProgramNoResolveLink(id);

    if (!validProgram)
    {
        if (context->getShader(id))
        {
            context->validationError(GL_INVALID_OPERATION, kExpectedProgramName);
        }
        else
        {
            context->validationError(GL_INVALID_VALUE, kInvalidProgramName);
        }
    }

    return validProgram;
}

Program *GetValidProgram(Context *context, ShaderProgramID id)
{
    Program *program = GetValidProgramNoResolve(context, id);
    if (program)
    {
        program->resolveLink(context);
    }
    return program;
}

Shader *GetValidShader(Context *context, ShaderProgramID id)
{
    // See ValidProgram for spec details.

    Shader *validShader = context->getShader(id);

    if (!validShader)
    {
        if (context->getProgramNoResolveLink(id))
        {
            context->validationError(GL_INVALID_OPERATION, kExpectedShaderName);
        }
        else
        {
            context->validationError(GL_INVALID_VALUE, kInvalidShaderName);
        }
    }

    return validShader;
}

bool ValidateAttachmentTarget(Context *context, GLenum attachment)
{
    if (attachment >= GL_COLOR_ATTACHMENT1_EXT && attachment <= GL_COLOR_ATTACHMENT15_EXT)
    {
        if (context->getClientMajorVersion() < 3 && !context->getExtensions().drawBuffers)
        {
            context->validationError(GL_INVALID_ENUM, kInvalidAttachment);
            return false;
        }

        // Color attachment 0 is validated below because it is always valid
        const unsigned int colorAttachment = (attachment - GL_COLOR_ATTACHMENT0_EXT);
        if (colorAttachment >= context->getCaps().maxColorAttachments)
        {
            context->validationError(GL_INVALID_OPERATION, kInvalidAttachment);
            return false;
        }
    }
    else
    {
        switch (attachment)
        {
            case GL_COLOR_ATTACHMENT0:
            case GL_DEPTH_ATTACHMENT:
            case GL_STENCIL_ATTACHMENT:
                break;

            case GL_DEPTH_STENCIL_ATTACHMENT:
                if (!context->getExtensions().webglCompatibility &&
                    context->getClientMajorVersion() < 3)
                {
                    context->validationError(GL_INVALID_ENUM, kInvalidAttachment);
                    return false;
                }
                break;

            default:
                context->validationError(GL_INVALID_ENUM, kInvalidAttachment);
                return false;
        }
    }

    return true;
}

bool ValidateRenderbufferStorageParametersBase(Context *context,
                                               GLenum target,
                                               GLsizei samples,
                                               GLenum internalformat,
                                               GLsizei width,
                                               GLsizei height)
{
    switch (target)
    {
        case GL_RENDERBUFFER:
            break;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidRenderbufferTarget);
            return false;
    }

    if (width < 0 || height < 0 || samples < 0)
    {
        context->validationError(GL_INVALID_VALUE, kInvalidRenderbufferWidthHeight);
        return false;
    }

    // Hack for the special WebGL 1 "DEPTH_STENCIL" internal format.
    GLenum convertedInternalFormat = context->getConvertedRenderbufferFormat(internalformat);

    const TextureCaps &formatCaps = context->getTextureCaps().get(convertedInternalFormat);
    if (!formatCaps.renderbuffer)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidRenderbufferInternalFormat);
        return false;
    }

    // ANGLE_framebuffer_multisample does not explicitly state that the internal format must be
    // sized but it does state that the format must be in the ES2.0 spec table 4.5 which contains
    // only sized internal formats.
    const InternalFormat &formatInfo = GetSizedInternalFormatInfo(convertedInternalFormat);
    if (formatInfo.internalFormat == GL_NONE)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidRenderbufferInternalFormat);
        return false;
    }

    if (static_cast<GLuint>(std::max(width, height)) > context->getCaps().maxRenderbufferSize)
    {
        context->validationError(GL_INVALID_VALUE, kResourceMaxRenderbufferSize);
        return false;
    }

    RenderbufferID id = context->getState().getRenderbufferId();
    if (id.value == 0)
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidRenderbufferTarget);
        return false;
    }

    return true;
}

bool ValidateFramebufferRenderbufferParameters(Context *context,
                                               GLenum target,
                                               GLenum attachment,
                                               GLenum renderbuffertarget,
                                               RenderbufferID renderbuffer)
{
    if (!ValidFramebufferTarget(context, target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidFramebufferTarget);
        return false;
    }

    Framebuffer *framebuffer = context->getState().getTargetFramebuffer(target);

    ASSERT(framebuffer);
    if (framebuffer->isDefault())
    {
        context->validationError(GL_INVALID_OPERATION, kDefaultFramebufferTarget);
        return false;
    }

    if (!ValidateAttachmentTarget(context, attachment))
    {
        return false;
    }

    // [OpenGL ES 2.0.25] Section 4.4.3 page 112
    // [OpenGL ES 3.0.2] Section 4.4.2 page 201
    // 'renderbuffer' must be either zero or the name of an existing renderbuffer object of
    // type 'renderbuffertarget', otherwise an INVALID_OPERATION error is generated.
    if (renderbuffer.value != 0)
    {
        if (!context->getRenderbuffer(renderbuffer))
        {
            context->validationError(GL_INVALID_OPERATION, kInvalidRenderbufferTarget);
            return false;
        }
    }

    return true;
}

bool ValidateBlitFramebufferParameters(Context *context,
                                       GLint srcX0,
                                       GLint srcY0,
                                       GLint srcX1,
                                       GLint srcY1,
                                       GLint dstX0,
                                       GLint dstY0,
                                       GLint dstX1,
                                       GLint dstY1,
                                       GLbitfield mask,
                                       GLenum filter)
{
    switch (filter)
    {
        case GL_NEAREST:
            break;
        case GL_LINEAR:
            break;
        default:
            context->validationError(GL_INVALID_ENUM, kBlitInvalidFilter);
            return false;
    }

    if ((mask & ~(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)) != 0)
    {
        context->validationError(GL_INVALID_VALUE, kBlitInvalidMask);
        return false;
    }

    // ES3.0 spec, section 4.3.2 states that linear filtering is only available for the
    // color buffer, leaving only nearest being unfiltered from above
    if ((mask & ~GL_COLOR_BUFFER_BIT) != 0 && filter != GL_NEAREST)
    {
        context->validationError(GL_INVALID_OPERATION, kBlitOnlyNearestForNonColor);
        return false;
    }

    const auto &glState          = context->getState();
    Framebuffer *readFramebuffer = glState.getReadFramebuffer();
    Framebuffer *drawFramebuffer = glState.getDrawFramebuffer();

    if (!readFramebuffer || !drawFramebuffer)
    {
        context->validationError(GL_INVALID_FRAMEBUFFER_OPERATION, kBlitFramebufferMissing);
        return false;
    }

    if (!ValidateFramebufferComplete(context, readFramebuffer))
    {
        return false;
    }

    if (!ValidateFramebufferComplete(context, drawFramebuffer))
    {
        return false;
    }

    // The draw and read framebuffers can only match if:
    // - They are the default framebuffer AND
    // - The read/draw surfaces are different
    if ((readFramebuffer->id() == drawFramebuffer->id()) &&
        ((drawFramebuffer->id() != Framebuffer::kDefaultDrawFramebufferHandle) ||
         (context->getCurrentDrawSurface() == context->getCurrentReadSurface())))
    {
        context->validationError(GL_INVALID_OPERATION, kBlitFeedbackLoop);
        return false;
    }

    // Not allow blitting to MS buffers, therefore if renderToTextureSamples exist,
    // consider it MS. needResourceSamples = false
    if (!ValidateFramebufferNotMultisampled(context, drawFramebuffer, false))
    {
        return false;
    }

    // This validation is specified in the WebGL 2.0 spec and not in the GLES 3.0.5 spec, but we
    // always run it in order to avoid triggering driver bugs.
    if (DifferenceCanOverflow(srcX0, srcX1) || DifferenceCanOverflow(srcY0, srcY1) ||
        DifferenceCanOverflow(dstX0, dstX1) || DifferenceCanOverflow(dstY0, dstY1))
    {
        context->validationError(GL_INVALID_VALUE, kBlitDimensionsOutOfRange);
        return false;
    }

    bool sameBounds = srcX0 == dstX0 && srcY0 == dstY0 && srcX1 == dstX1 && srcY1 == dstY1;

    if (mask & GL_COLOR_BUFFER_BIT)
    {
        const FramebufferAttachment *readColorBuffer = readFramebuffer->getReadColorAttachment();
        const Extensions &extensions                 = context->getExtensions();

        if (readColorBuffer)
        {
            const Format &readFormat = readColorBuffer->getFormat();

            for (size_t drawbufferIdx = 0;
                 drawbufferIdx < drawFramebuffer->getDrawbufferStateCount(); ++drawbufferIdx)
            {
                const FramebufferAttachment *attachment =
                    drawFramebuffer->getDrawBuffer(drawbufferIdx);
                if (attachment)
                {
                    const Format &drawFormat = attachment->getFormat();

                    // The GL ES 3.0.2 spec (pg 193) states that:
                    // 1) If the read buffer is fixed point format, the draw buffer must be as well
                    // 2) If the read buffer is an unsigned integer format, the draw buffer must be
                    // as well
                    // 3) If the read buffer is a signed integer format, the draw buffer must be as
                    // well
                    // Changes with EXT_color_buffer_float:
                    // Case 1) is changed to fixed point OR floating point
                    GLenum readComponentType = readFormat.info->componentType;
                    GLenum drawComponentType = drawFormat.info->componentType;
                    bool readFixedPoint      = (readComponentType == GL_UNSIGNED_NORMALIZED ||
                                           readComponentType == GL_SIGNED_NORMALIZED);
                    bool drawFixedPoint      = (drawComponentType == GL_UNSIGNED_NORMALIZED ||
                                           drawComponentType == GL_SIGNED_NORMALIZED);

                    if (extensions.colorBufferFloat)
                    {
                        bool readFixedOrFloat = (readFixedPoint || readComponentType == GL_FLOAT);
                        bool drawFixedOrFloat = (drawFixedPoint || drawComponentType == GL_FLOAT);

                        if (readFixedOrFloat != drawFixedOrFloat)
                        {
                            context->validationError(GL_INVALID_OPERATION,
                                                     kBlitTypeMismatchFixedOrFloat);
                            return false;
                        }
                    }
                    else if (readFixedPoint != drawFixedPoint)
                    {
                        context->validationError(GL_INVALID_OPERATION, kBlitTypeMismatchFixedPoint);
                        return false;
                    }

                    if (readComponentType == GL_UNSIGNED_INT &&
                        drawComponentType != GL_UNSIGNED_INT)
                    {
                        context->validationError(GL_INVALID_OPERATION,
                                                 kBlitTypeMismatchUnsignedInteger);
                        return false;
                    }

                    if (readComponentType == GL_INT && drawComponentType != GL_INT)
                    {
                        context->validationError(GL_INVALID_OPERATION,
                                                 kBlitTypeMismatchSignedInteger);
                        return false;
                    }

                    if (readColorBuffer->getSamples() > 0 &&
                        (!Format::EquivalentForBlit(readFormat, drawFormat) || !sameBounds))
                    {
                        context->validationError(GL_INVALID_OPERATION,
                                                 kBlitMultisampledFormatOrBoundsMismatch);
                        return false;
                    }

                    if (context->getExtensions().webglCompatibility &&
                        *readColorBuffer == *attachment)
                    {
                        context->validationError(GL_INVALID_OPERATION, kBlitSameImageColor);
                        return false;
                    }
                }
            }

            if (readFormat.info->isInt() && filter == GL_LINEAR)
            {
                context->validationError(GL_INVALID_OPERATION, kBlitIntegerWithLinearFilter);
                return false;
            }
        }
        // WebGL 2.0 BlitFramebuffer when blitting from a missing attachment
        // In OpenGL ES it is undefined what happens when an operation tries to blit from a missing
        // attachment and WebGL defines it to be an error. We do the check unconditionally as the
        // situation is an application error that would lead to a crash in ANGLE.
        else if (drawFramebuffer->hasEnabledDrawBuffer())
        {
            context->validationError(GL_INVALID_OPERATION, kBlitMissingColor);
            return false;
        }
    }

    GLenum masks[]       = {GL_DEPTH_BUFFER_BIT, GL_STENCIL_BUFFER_BIT};
    GLenum attachments[] = {GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT};
    for (size_t i = 0; i < 2; i++)
    {
        if (mask & masks[i])
        {
            const FramebufferAttachment *readBuffer =
                readFramebuffer->getAttachment(context, attachments[i]);
            const FramebufferAttachment *drawBuffer =
                drawFramebuffer->getAttachment(context, attachments[i]);

            if (readBuffer && drawBuffer)
            {
                if (!Format::EquivalentForBlit(readBuffer->getFormat(), drawBuffer->getFormat()))
                {
                    context->validationError(GL_INVALID_OPERATION,
                                             kBlitDepthOrStencilFormatMismatch);
                    return false;
                }

                if (readBuffer->getSamples() > 0 && !sameBounds)
                {
                    context->validationError(GL_INVALID_OPERATION, kBlitMultisampledBoundsMismatch);
                    return false;
                }

                if (context->getExtensions().webglCompatibility && *readBuffer == *drawBuffer)
                {
                    context->validationError(GL_INVALID_OPERATION, kBlitSameImageDepthOrStencil);
                    return false;
                }
            }
            // WebGL 2.0 BlitFramebuffer when blitting from a missing attachment
            else if (drawBuffer)
            {
                context->validationError(GL_INVALID_OPERATION, kBlitMissingDepthOrStencil);
                return false;
            }
        }
    }

    // OVR_multiview2:
    // Calling BlitFramebuffer will result in an INVALID_FRAMEBUFFER_OPERATION error if the
    // current draw framebuffer isMultiview() or the number of
    // views in the current read framebuffer is more than one.
    if (readFramebuffer->readDisallowedByMultiview())
    {
        context->validationError(GL_INVALID_FRAMEBUFFER_OPERATION, kBlitFromMultiview);
        return false;
    }
    if (drawFramebuffer->isMultiview())
    {
        context->validationError(GL_INVALID_FRAMEBUFFER_OPERATION, kBlitToMultiview);
        return false;
    }

    return true;
}

bool ValidateReadPixelsRobustANGLE(Context *context,
                                   GLint x,
                                   GLint y,
                                   GLsizei width,
                                   GLsizei height,
                                   GLenum format,
                                   GLenum type,
                                   GLsizei bufSize,
                                   GLsizei *length,
                                   GLsizei *columns,
                                   GLsizei *rows,
                                   void *pixels)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei writeLength  = 0;
    GLsizei writeColumns = 0;
    GLsizei writeRows    = 0;

    if (!ValidateReadPixelsBase(context, x, y, width, height, format, type, bufSize, &writeLength,
                                &writeColumns, &writeRows, pixels))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, writeLength))
    {
        return false;
    }

    SetRobustLengthParam(length, writeLength);
    SetRobustLengthParam(columns, writeColumns);
    SetRobustLengthParam(rows, writeRows);

    return true;
}

bool ValidateReadnPixelsEXT(Context *context,
                            GLint x,
                            GLint y,
                            GLsizei width,
                            GLsizei height,
                            GLenum format,
                            GLenum type,
                            GLsizei bufSize,
                            void *pixels)
{
    if (bufSize < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeBufferSize);
        return false;
    }

    return ValidateReadPixelsBase(context, x, y, width, height, format, type, bufSize, nullptr,
                                  nullptr, nullptr, pixels);
}

bool ValidateReadnPixelsRobustANGLE(Context *context,
                                    GLint x,
                                    GLint y,
                                    GLsizei width,
                                    GLsizei height,
                                    GLenum format,
                                    GLenum type,
                                    GLsizei bufSize,
                                    GLsizei *length,
                                    GLsizei *columns,
                                    GLsizei *rows,
                                    void *data)
{
    GLsizei writeLength  = 0;
    GLsizei writeColumns = 0;
    GLsizei writeRows    = 0;

    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    if (!ValidateReadPixelsBase(context, x, y, width, height, format, type, bufSize, &writeLength,
                                &writeColumns, &writeRows, data))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, writeLength))
    {
        return false;
    }

    SetRobustLengthParam(length, writeLength);
    SetRobustLengthParam(columns, writeColumns);
    SetRobustLengthParam(rows, writeRows);

    return true;
}

bool ValidateGenQueriesEXT(Context *context, GLsizei n, QueryID *ids)
{
    if (!context->getExtensions().occlusionQueryBoolean &&
        !context->getExtensions().disjointTimerQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kQueryExtensionNotEnabled);
        return false;
    }

    return ValidateGenOrDelete(context, n);
}

bool ValidateDeleteQueriesEXT(Context *context, GLsizei n, const QueryID *ids)
{
    if (!context->getExtensions().occlusionQueryBoolean &&
        !context->getExtensions().disjointTimerQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kQueryExtensionNotEnabled);
        return false;
    }

    return ValidateGenOrDelete(context, n);
}

bool ValidateIsQueryEXT(Context *context, QueryID id)
{
    if (!context->getExtensions().occlusionQueryBoolean &&
        !context->getExtensions().disjointTimerQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kQueryExtensionNotEnabled);
        return false;
    }

    return true;
}

bool ValidateBeginQueryBase(Context *context, QueryType target, QueryID id)
{
    if (!ValidQueryType(context, target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidQueryType);
        return false;
    }

    if (id.value == 0)
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidQueryId);
        return false;
    }

    // From EXT_occlusion_query_boolean: If BeginQueryEXT is called with an <id>
    // of zero, if the active query object name for <target> is non-zero (for the
    // targets ANY_SAMPLES_PASSED_EXT and ANY_SAMPLES_PASSED_CONSERVATIVE_EXT, if
    // the active query for either target is non-zero), if <id> is the name of an
    // existing query object whose type does not match <target>, or if <id> is the
    // active query object name for any query type, the error INVALID_OPERATION is
    // generated.

    // Ensure no other queries are active
    // NOTE: If other queries than occlusion are supported, we will need to check
    // separately that:
    //    a) The query ID passed is not the current active query for any target/type
    //    b) There are no active queries for the requested target (and in the case
    //       of GL_ANY_SAMPLES_PASSED_EXT and GL_ANY_SAMPLES_PASSED_CONSERVATIVE_EXT,
    //       no query may be active for either if glBeginQuery targets either.

    if (context->getState().isQueryActive(target))
    {
        context->validationError(GL_INVALID_OPERATION, kOtherQueryActive);
        return false;
    }

    Query *queryObject = context->getQuery(id, true, target);

    // check that name was obtained with glGenQueries
    if (!queryObject)
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidQueryId);
        return false;
    }

    // check for type mismatch
    if (queryObject->getType() != target)
    {
        context->validationError(GL_INVALID_OPERATION, kQueryTargetMismatch);
        return false;
    }

    return true;
}

bool ValidateBeginQueryEXT(Context *context, QueryType target, QueryID id)
{
    if (!context->getExtensions().occlusionQueryBoolean &&
        !context->getExtensions().disjointTimerQuery && !context->getExtensions().syncQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kQueryExtensionNotEnabled);
        return false;
    }

    return ValidateBeginQueryBase(context, target, id);
}

bool ValidateEndQueryBase(Context *context, QueryType target)
{
    if (!ValidQueryType(context, target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidQueryType);
        return false;
    }

    const Query *queryObject = context->getState().getActiveQuery(target);

    if (queryObject == nullptr)
    {
        context->validationError(GL_INVALID_OPERATION, kQueryInactive);
        return false;
    }

    return true;
}

bool ValidateEndQueryEXT(Context *context, QueryType target)
{
    if (!context->getExtensions().occlusionQueryBoolean &&
        !context->getExtensions().disjointTimerQuery && !context->getExtensions().syncQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kQueryExtensionNotEnabled);
        return false;
    }

    return ValidateEndQueryBase(context, target);
}

bool ValidateQueryCounterEXT(Context *context, QueryID id, QueryType target)
{
    if (!context->getExtensions().disjointTimerQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    if (target != QueryType::Timestamp)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidQueryTarget);
        return false;
    }

    Query *queryObject = context->getQuery(id, true, target);
    if (queryObject == nullptr)
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidQueryId);
        return false;
    }

    if (context->getState().isQueryActive(queryObject))
    {
        context->validationError(GL_INVALID_OPERATION, kQueryActive);
        return false;
    }

    return true;
}

bool ValidateGetQueryivBase(Context *context, QueryType target, GLenum pname, GLsizei *numParams)
{
    if (numParams)
    {
        *numParams = 0;
    }

    if (!ValidQueryType(context, target) && target != QueryType::Timestamp)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidQueryType);
        return false;
    }

    switch (pname)
    {
        case GL_CURRENT_QUERY_EXT:
            if (target == QueryType::Timestamp)
            {
                context->validationError(GL_INVALID_ENUM, kInvalidQueryTarget);
                return false;
            }
            break;
        case GL_QUERY_COUNTER_BITS_EXT:
            if (!context->getExtensions().disjointTimerQuery ||
                (target != QueryType::Timestamp && target != QueryType::TimeElapsed))
            {
                context->validationError(GL_INVALID_ENUM, kInvalidPname);
                return false;
            }
            break;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidPname);
            return false;
    }

    if (numParams)
    {
        // All queries return only one value
        *numParams = 1;
    }

    return true;
}

bool ValidateGetQueryivEXT(Context *context, QueryType target, GLenum pname, GLint *params)
{
    if (!context->getExtensions().occlusionQueryBoolean &&
        !context->getExtensions().disjointTimerQuery && !context->getExtensions().syncQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    return ValidateGetQueryivBase(context, target, pname, nullptr);
}

bool ValidateGetQueryivRobustANGLE(Context *context,
                                   QueryType target,
                                   GLenum pname,
                                   GLsizei bufSize,
                                   GLsizei *length,
                                   GLint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;

    if (!ValidateGetQueryivBase(context, target, pname, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

bool ValidateGetQueryObjectValueBase(Context *context, QueryID id, GLenum pname, GLsizei *numParams)
{
    if (numParams)
    {
        *numParams = 1;
    }

    if (context->isContextLost())
    {
        context->validationError(GL_CONTEXT_LOST, kContextLost);

        if (pname == GL_QUERY_RESULT_AVAILABLE_EXT)
        {
            // Generate an error but still return true, the context still needs to return a
            // value in this case.
            return true;
        }
        else
        {
            return false;
        }
    }

    Query *queryObject = context->getQuery(id, false, QueryType::InvalidEnum);

    if (!queryObject)
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidQueryId);
        return false;
    }

    if (context->getState().isQueryActive(queryObject))
    {
        context->validationError(GL_INVALID_OPERATION, kQueryActive);
        return false;
    }

    switch (pname)
    {
        case GL_QUERY_RESULT_EXT:
        case GL_QUERY_RESULT_AVAILABLE_EXT:
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
    }

    return true;
}

bool ValidateGetQueryObjectivEXT(Context *context, QueryID id, GLenum pname, GLint *params)
{
    if (!context->getExtensions().disjointTimerQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }
    return ValidateGetQueryObjectValueBase(context, id, pname, nullptr);
}

bool ValidateGetQueryObjectivRobustANGLE(Context *context,
                                         QueryID id,
                                         GLenum pname,
                                         GLsizei bufSize,
                                         GLsizei *length,
                                         GLint *params)
{
    if (!context->getExtensions().disjointTimerQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;

    if (!ValidateGetQueryObjectValueBase(context, id, pname, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

bool ValidateGetQueryObjectuivEXT(Context *context, QueryID id, GLenum pname, GLuint *params)
{
    if (!context->getExtensions().disjointTimerQuery &&
        !context->getExtensions().occlusionQueryBoolean && !context->getExtensions().syncQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }
    return ValidateGetQueryObjectValueBase(context, id, pname, nullptr);
}

bool ValidateGetQueryObjectuivRobustANGLE(Context *context,
                                          QueryID id,
                                          GLenum pname,
                                          GLsizei bufSize,
                                          GLsizei *length,
                                          GLuint *params)
{
    if (!context->getExtensions().disjointTimerQuery &&
        !context->getExtensions().occlusionQueryBoolean && !context->getExtensions().syncQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;

    if (!ValidateGetQueryObjectValueBase(context, id, pname, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

bool ValidateGetQueryObjecti64vEXT(Context *context, QueryID id, GLenum pname, GLint64 *params)
{
    if (!context->getExtensions().disjointTimerQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }
    return ValidateGetQueryObjectValueBase(context, id, pname, nullptr);
}

bool ValidateGetQueryObjecti64vRobustANGLE(Context *context,
                                           QueryID id,
                                           GLenum pname,
                                           GLsizei bufSize,
                                           GLsizei *length,
                                           GLint64 *params)
{
    if (!context->getExtensions().disjointTimerQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;

    if (!ValidateGetQueryObjectValueBase(context, id, pname, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

bool ValidateGetQueryObjectui64vEXT(Context *context, QueryID id, GLenum pname, GLuint64 *params)
{
    if (!context->getExtensions().disjointTimerQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }
    return ValidateGetQueryObjectValueBase(context, id, pname, nullptr);
}

bool ValidateGetQueryObjectui64vRobustANGLE(Context *context,
                                            QueryID id,
                                            GLenum pname,
                                            GLsizei bufSize,
                                            GLsizei *length,
                                            GLuint64 *params)
{
    if (!context->getExtensions().disjointTimerQuery)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;

    if (!ValidateGetQueryObjectValueBase(context, id, pname, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

bool ValidateUniformCommonBase(Context *context,
                               Program *program,
                               GLint location,
                               GLsizei count,
                               const LinkedUniform **uniformOut)
{
    // TODO(Jiajia): Add image uniform check in future.
    if (count < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeCount);
        return false;
    }

    if (!program)
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidProgramName);
        return false;
    }

    if (!program->isLinked())
    {
        context->validationError(GL_INVALID_OPERATION, kProgramNotLinked);
        return false;
    }

    if (location == -1)
    {
        // Silently ignore the uniform command
        return false;
    }

    const auto &uniformLocations = program->getUniformLocations();
    size_t castedLocation        = static_cast<size_t>(location);
    if (castedLocation >= uniformLocations.size())
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidUniformLocation);
        return false;
    }

    const auto &uniformLocation = uniformLocations[castedLocation];
    if (uniformLocation.ignored)
    {
        // Silently ignore the uniform command
        return false;
    }

    if (!uniformLocation.used())
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidUniformLocation);
        return false;
    }

    const auto &uniform = program->getUniformByIndex(uniformLocation.index);

    // attempting to write an array to a non-array uniform is an INVALID_OPERATION
    if (count > 1 && !uniform.isArray())
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidUniformCount);
        return false;
    }

    *uniformOut = &uniform;
    return true;
}

bool ValidateUniform1ivValue(Context *context,
                             GLenum uniformType,
                             GLsizei count,
                             const GLint *value)
{
    // Value type is GL_INT, because we only get here from glUniform1i{v}.
    // It is compatible with INT or BOOL.
    // Do these cheap tests first, for a little extra speed.
    if (GL_INT == uniformType || GL_BOOL == uniformType)
    {
        return true;
    }

    if (IsSamplerType(uniformType))
    {
        // Check that the values are in range.
        const GLint max = context->getCaps().maxCombinedTextureImageUnits;
        for (GLsizei i = 0; i < count; ++i)
        {
            if (value[i] < 0 || value[i] >= max)
            {
                context->validationError(GL_INVALID_VALUE, kSamplerUniformValueOutOfRange);
                return false;
            }
        }
        return true;
    }

    context->validationError(GL_INVALID_OPERATION, kUniformTypeMismatch);
    return false;
}

bool ValidateUniformMatrixValue(Context *context, GLenum valueType, GLenum uniformType)
{
    // Check that the value type is compatible with uniform type.
    if (valueType == uniformType)
    {
        return true;
    }

    context->validationError(GL_INVALID_OPERATION, kUniformTypeMismatch);
    return false;
}

bool ValidateUniform(Context *context, GLenum valueType, GLint location, GLsizei count)
{
    const LinkedUniform *uniform = nullptr;
    Program *programObject       = context->getState().getLinkedProgram(context);
    return ValidateUniformCommonBase(context, programObject, location, count, &uniform) &&
           ValidateUniformValue(context, valueType, uniform->type);
}

bool ValidateUniform1iv(Context *context, GLint location, GLsizei count, const GLint *value)
{
    const LinkedUniform *uniform = nullptr;
    Program *programObject       = context->getState().getLinkedProgram(context);
    return ValidateUniformCommonBase(context, programObject, location, count, &uniform) &&
           ValidateUniform1ivValue(context, uniform->type, count, value);
}

bool ValidateUniformMatrix(Context *context,
                           GLenum valueType,
                           GLint location,
                           GLsizei count,
                           GLboolean transpose)
{
    if (ConvertToBool(transpose) && context->getClientMajorVersion() < 3)
    {
        context->validationError(GL_INVALID_VALUE, kES3Required);
        return false;
    }

    const LinkedUniform *uniform = nullptr;
    Program *programObject       = context->getState().getLinkedProgram(context);
    return ValidateUniformCommonBase(context, programObject, location, count, &uniform) &&
           ValidateUniformMatrixValue(context, valueType, uniform->type);
}

bool ValidateStateQuery(Context *context, GLenum pname, GLenum *nativeType, unsigned int *numParams)
{
    if (!context->getQueryParameterInfo(pname, nativeType, numParams))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidPname);
        return false;
    }

    const Caps &caps = context->getCaps();

    if (pname >= GL_DRAW_BUFFER0 && pname <= GL_DRAW_BUFFER15)
    {
        unsigned int colorAttachment = (pname - GL_DRAW_BUFFER0);

        if (colorAttachment >= caps.maxDrawBuffers)
        {
            context->validationError(GL_INVALID_OPERATION, kIndexExceedsMaxDrawBuffer);
            return false;
        }
    }

    switch (pname)
    {
        case GL_TEXTURE_BINDING_2D:
        case GL_TEXTURE_BINDING_CUBE_MAP:
        case GL_TEXTURE_BINDING_3D:
        case GL_TEXTURE_BINDING_2D_ARRAY:
        case GL_TEXTURE_BINDING_2D_MULTISAMPLE:
            break;
        case GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY:
            if (!context->getExtensions().textureStorageMultisample2DArray)
            {
                context->validationError(GL_INVALID_ENUM, kMultisampleArrayExtensionRequired);
                return false;
            }
            break;
        case GL_TEXTURE_BINDING_RECTANGLE_ANGLE:
            if (!context->getExtensions().textureRectangle)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;
        case GL_TEXTURE_BINDING_EXTERNAL_OES:
            if (!context->getExtensions().eglStreamConsumerExternal &&
                !context->getExtensions().eglImageExternal)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;

        case GL_IMPLEMENTATION_COLOR_READ_TYPE:
        case GL_IMPLEMENTATION_COLOR_READ_FORMAT:
        {
            Framebuffer *readFramebuffer = context->getState().getReadFramebuffer();
            ASSERT(readFramebuffer);

            if (!ValidateFramebufferComplete<GL_INVALID_OPERATION>(context, readFramebuffer))
            {
                return false;
            }

            if (readFramebuffer->getReadBufferState() == GL_NONE)
            {
                context->validationError(GL_INVALID_OPERATION, kReadBufferNone);
                return false;
            }

            const FramebufferAttachment *attachment = readFramebuffer->getReadColorAttachment();
            if (!attachment)
            {
                context->validationError(GL_INVALID_OPERATION, kReadBufferNotAttached);
                return false;
            }
        }
        break;

        default:
            break;
    }

    // pname is valid, but there are no parameters to return
    if (*numParams == 0)
    {
        return false;
    }

    return true;
}

bool ValidateGetBooleanvRobustANGLE(Context *context,
                                    GLenum pname,
                                    GLsizei bufSize,
                                    GLsizei *length,
                                    GLboolean *params)
{
    GLenum nativeType;
    unsigned int numParams = 0;

    if (!ValidateRobustStateQuery(context, pname, bufSize, &nativeType, &numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

bool ValidateGetFloatvRobustANGLE(Context *context,
                                  GLenum pname,
                                  GLsizei bufSize,
                                  GLsizei *length,
                                  GLfloat *params)
{
    GLenum nativeType;
    unsigned int numParams = 0;

    if (!ValidateRobustStateQuery(context, pname, bufSize, &nativeType, &numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

bool ValidateGetIntegervRobustANGLE(Context *context,
                                    GLenum pname,
                                    GLsizei bufSize,
                                    GLsizei *length,
                                    GLint *data)
{
    GLenum nativeType;
    unsigned int numParams = 0;

    if (!ValidateRobustStateQuery(context, pname, bufSize, &nativeType, &numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

bool ValidateGetInteger64vRobustANGLE(Context *context,
                                      GLenum pname,
                                      GLsizei bufSize,
                                      GLsizei *length,
                                      GLint64 *data)
{
    GLenum nativeType;
    unsigned int numParams = 0;

    if (!ValidateRobustStateQuery(context, pname, bufSize, &nativeType, &numParams))
    {
        return false;
    }

    if (nativeType == GL_INT_64_ANGLEX)
    {
        CastStateValues(context, nativeType, pname, numParams, data);
        return false;
    }

    SetRobustLengthParam(length, numParams);
    return true;
}

bool ValidateRobustStateQuery(Context *context,
                              GLenum pname,
                              GLsizei bufSize,
                              GLenum *nativeType,
                              unsigned int *numParams)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    if (!ValidateStateQuery(context, pname, nativeType, numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, *numParams))
    {
        return false;
    }

    return true;
}

bool ValidateCopyTexImageParametersBase(Context *context,
                                        TextureTarget target,
                                        GLint level,
                                        GLenum internalformat,
                                        bool isSubImage,
                                        GLint xoffset,
                                        GLint yoffset,
                                        GLint zoffset,
                                        GLint x,
                                        GLint y,
                                        GLsizei width,
                                        GLsizei height,
                                        GLint border,
                                        Format *textureFormatOut)
{
    TextureType texType = TextureTargetToType(target);

    if (xoffset < 0 || yoffset < 0 || zoffset < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeOffset);
        return false;
    }

    if (width < 0 || height < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeSize);
        return false;
    }

    if (std::numeric_limits<GLsizei>::max() - xoffset < width ||
        std::numeric_limits<GLsizei>::max() - yoffset < height)
    {
        context->validationError(GL_INVALID_VALUE, kOffsetOverflow);
        return false;
    }

    if (border != 0)
    {
        context->validationError(GL_INVALID_VALUE, kInvalidBorder);
        return false;
    }

    if (!ValidMipLevel(context, texType, level))
    {
        context->validationError(GL_INVALID_VALUE, kInvalidMipLevel);
        return false;
    }

    const State &state           = context->getState();
    Framebuffer *readFramebuffer = state.getReadFramebuffer();
    if (!ValidateFramebufferComplete(context, readFramebuffer))
    {
        return false;
    }

    // needResourceSamples = true. Treat renderToTexture textures as single sample since they will
    // be resolved before copying
    if (!readFramebuffer->isDefault() &&
        !ValidateFramebufferNotMultisampled(context, readFramebuffer, true))
    {
        return false;
    }

    if (readFramebuffer->getReadBufferState() == GL_NONE)
    {
        context->validationError(GL_INVALID_OPERATION, kReadBufferNone);
        return false;
    }

    // WebGL 1.0 [Section 6.26] Reading From a Missing Attachment
    // In OpenGL ES it is undefined what happens when an operation tries to read from a missing
    // attachment and WebGL defines it to be an error. We do the check unconditionally as the
    // situation is an application error that would lead to a crash in ANGLE.
    const FramebufferAttachment *source = readFramebuffer->getReadColorAttachment();
    if (source == nullptr)
    {
        context->validationError(GL_INVALID_OPERATION, kMissingReadAttachment);
        return false;
    }

    // ANGLE_multiview spec, Revision 1:
    // Calling CopyTexSubImage3D, CopyTexImage2D, or CopyTexSubImage2D will result in an
    // INVALID_FRAMEBUFFER_OPERATION error if the multi-view layout of the current read framebuffer
    // is FRAMEBUFFER_MULTIVIEW_SIDE_BY_SIDE_ANGLE or the number of views in the current read
    // framebuffer is more than one.
    if (readFramebuffer->readDisallowedByMultiview())
    {
        context->validationError(GL_INVALID_FRAMEBUFFER_OPERATION, kMultiviewReadFramebuffer);
        return false;
    }

    const Caps &caps = context->getCaps();

    GLuint maxDimension = 0;
    switch (texType)
    {
        case TextureType::_2D:
            maxDimension = caps.max2DTextureSize;
            break;

        case TextureType::CubeMap:
            maxDimension = caps.maxCubeMapTextureSize;
            break;

        case TextureType::Rectangle:
            maxDimension = caps.maxRectangleTextureSize;
            break;

        case TextureType::_2DArray:
            maxDimension = caps.max2DTextureSize;
            break;

        case TextureType::_3D:
            maxDimension = caps.max3DTextureSize;
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kInvalidTextureTarget);
            return false;
    }

    Texture *texture = state.getTargetTexture(texType);
    if (!texture)
    {
        context->validationError(GL_INVALID_OPERATION, kTextureNotBound);
        return false;
    }

    if (texture->getImmutableFormat() && !isSubImage)
    {
        context->validationError(GL_INVALID_OPERATION, kTextureIsImmutable);
        return false;
    }

    const InternalFormat &formatInfo =
        isSubImage ? *texture->getFormat(target, level).info
                   : GetInternalFormatInfo(internalformat, GL_UNSIGNED_BYTE);

    if (formatInfo.depthBits > 0 || formatInfo.compressed)
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidFormat);
        return false;
    }

    if (isSubImage)
    {
        if (static_cast<size_t>(xoffset + width) > texture->getWidth(target, level) ||
            static_cast<size_t>(yoffset + height) > texture->getHeight(target, level) ||
            static_cast<size_t>(zoffset) >= texture->getDepth(target, level))
        {
            context->validationError(GL_INVALID_VALUE, kOffsetOverflow);
            return false;
        }
    }
    else
    {
        if (texType == TextureType::CubeMap && width != height)
        {
            context->validationError(GL_INVALID_VALUE, kCubemapIncomplete);
            return false;
        }

        if (!formatInfo.textureSupport(context->getClientVersion(), context->getExtensions()))
        {
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
        }

        int maxLevelDimension = (maxDimension >> level);
        if (static_cast<int>(width) > maxLevelDimension ||
            static_cast<int>(height) > maxLevelDimension)
        {
            context->validationError(GL_INVALID_VALUE, kResourceMaxTextureSize);
            return false;
        }
    }

    if (textureFormatOut)
    {
        *textureFormatOut = texture->getFormat(target, level);
    }

    // Detect texture copying feedback loops for WebGL.
    if (context->getExtensions().webglCompatibility)
    {
        if (readFramebuffer->formsCopyingFeedbackLoopWith(texture->id(), level, zoffset))
        {
            context->validationError(GL_INVALID_OPERATION, kFeedbackLoop);
            return false;
        }
    }

    return true;
}

// Note all errors returned from this function are INVALID_OPERATION except for the draw framebuffer
// completeness check.
const char *ValidateDrawStates(Context *context)
{
    const Extensions &extensions = context->getExtensions();
    const State &state           = context->getState();

    // WebGL buffers cannot be mapped/unmapped because the MapBufferRange, FlushMappedBufferRange,
    // and UnmapBuffer entry points are removed from the WebGL 2.0 API.
    // https://www.khronos.org/registry/webgl/specs/latest/2.0/#5.14
    VertexArray *vertexArray = state.getVertexArray();
    ASSERT(vertexArray);

    if (!extensions.webglCompatibility && vertexArray->hasMappedEnabledArrayBuffer())
    {
        return kBufferMapped;
    }

    // Note: these separate values are not supported in WebGL, due to D3D's limitations. See
    // Section 6.10 of the WebGL 1.0 spec.
    Framebuffer *framebuffer = state.getDrawFramebuffer();
    ASSERT(framebuffer);

    if (context->getLimitations().noSeparateStencilRefsAndMasks || extensions.webglCompatibility)
    {
        ASSERT(framebuffer);
        const FramebufferAttachment *dsAttachment =
            framebuffer->getStencilOrDepthStencilAttachment();
        const GLuint stencilBits = dsAttachment ? dsAttachment->getStencilSize() : 0;
        ASSERT(stencilBits <= 8);

        const DepthStencilState &depthStencilState = state.getDepthStencilState();
        if (depthStencilState.stencilTest && stencilBits > 0)
        {
            GLuint maxStencilValue = (1 << stencilBits) - 1;

            bool differentRefs =
                clamp(state.getStencilRef(), 0, static_cast<GLint>(maxStencilValue)) !=
                clamp(state.getStencilBackRef(), 0, static_cast<GLint>(maxStencilValue));
            bool differentWritemasks = (depthStencilState.stencilWritemask & maxStencilValue) !=
                                       (depthStencilState.stencilBackWritemask & maxStencilValue);
            bool differentMasks = (depthStencilState.stencilMask & maxStencilValue) !=
                                  (depthStencilState.stencilBackMask & maxStencilValue);

            if (differentRefs || differentWritemasks || differentMasks)
            {
                if (!extensions.webglCompatibility)
                {
                    WARN() << "This ANGLE implementation does not support separate front/back "
                              "stencil writemasks, reference values, or stencil mask values.";
                }
                return kStencilReferenceMaskOrMismatch;
            }
        }
    }

    if (!extensions.floatBlend && context->getState().isBlendEnabled() &&
        framebuffer->hasActiveFloat32ColorAttachment())
    {
        return kUnsupportedFloatBlending;
    }

    if (!framebuffer->isComplete(context))
    {
        // Note: this error should be generated as INVALID_FRAMEBUFFER_OPERATION.
        return kDrawFramebufferIncomplete;
    }

    if (context->getStateCache().hasAnyEnabledClientAttrib())
    {
        if (context->getExtensions().webglCompatibility || !state.areClientArraysEnabled())
        {
            // [WebGL 1.0] Section 6.5 Enabled Vertex Attributes and Range Checking
            // If a vertex attribute is enabled as an array via enableVertexAttribArray but no
            // buffer is bound to that attribute via bindBuffer and vertexAttribPointer, then calls
            // to drawArrays or drawElements will generate an INVALID_OPERATION error.
            return kVertexArrayNoBuffer;
        }

        if (state.getVertexArray()->hasEnabledNullPointerClientArray())
        {
            // This is an application error that would normally result in a crash, but we catch it
            // and return an error
            return kVertexArrayNoBufferPointer;
        }
    }

    // If we are running GLES1, there is no current program.
    if (context->getClientVersion() >= Version(2, 0))
    {
        Program *program = state.getLinkedProgram(context);
        if (!program)
        {
            return kProgramNotBound;
        }

        // In OpenGL ES spec for UseProgram at section 7.3, trying to render without
        // vertex shader stage or fragment shader stage is a undefined behaviour.
        // But ANGLE should clearly generate an INVALID_OPERATION error instead of
        // produce undefined result.
        if (!program->hasLinkedShaderStage(ShaderType::Vertex) ||
            !program->hasLinkedShaderStage(ShaderType::Fragment))
        {
            return kNoActiveGraphicsShaderStage;
        }

        if (!program->validateSamplers(nullptr, context->getCaps()))
        {
            return kTextureTypeConflict;
        }

        if (extensions.multiview || extensions.multiview2)
        {
            const int programNumViews     = program->usesMultiview() ? program->getNumViews() : 1;
            const int framebufferNumViews = framebuffer->getNumViews();
            if (framebufferNumViews != programNumViews)
            {
                return kMultiviewMismatch;
            }

            if (state.isTransformFeedbackActiveUnpaused() && framebufferNumViews > 1)
            {
                return kMultiviewTransformFeedback;
            }

            if (extensions.disjointTimerQuery && framebufferNumViews > 1 &&
                state.isQueryActive(QueryType::TimeElapsed))
            {
                return kMultiviewTimerQuery;
            }
        }

        // Uniform buffer validation
        for (unsigned int uniformBlockIndex = 0;
             uniformBlockIndex < program->getActiveUniformBlockCount(); uniformBlockIndex++)
        {
            const InterfaceBlock &uniformBlock = program->getUniformBlockByIndex(uniformBlockIndex);
            GLuint blockBinding                = program->getUniformBlockBinding(uniformBlockIndex);
            const OffsetBindingPointer<Buffer> &uniformBuffer =
                state.getIndexedUniformBuffer(blockBinding);

            if (uniformBuffer.get() == nullptr)
            {
                // undefined behaviour
                return kUniformBufferUnbound;
            }

            size_t uniformBufferSize = GetBoundBufferAvailableSize(uniformBuffer);
            if (uniformBufferSize < uniformBlock.dataSize)
            {
                // undefined behaviour
                return kUniformBufferTooSmall;
            }

            if (extensions.webglCompatibility &&
                uniformBuffer->isBoundForTransformFeedbackAndOtherUse())
            {
                return kUniformBufferBoundForTransformFeedback;
            }
        }

        // Do some additonal WebGL-specific validation
        if (extensions.webglCompatibility)
        {
            if (!state.validateSamplerFormats())
            {
                return kSamplerFormatMismatch;
            }

            const TransformFeedback *transformFeedbackObject = state.getCurrentTransformFeedback();
            if (state.isTransformFeedbackActive() &&
                transformFeedbackObject->buffersBoundForOtherUse())
            {
                return kTransformFeedbackBufferDoubleBound;
            }

            // Detect rendering feedback loops for WebGL.
            if (framebuffer->formsRenderingFeedbackLoopWith(context))
            {
                return kFeedbackLoop;
            }

            // Detect that the vertex shader input types match the attribute types
            if (!ValidateVertexShaderAttributeTypeMatch(context))
            {
                return kVertexShaderTypeMismatch;
            }

            if (!context->getState().getRasterizerState().rasterizerDiscard &&
                !context->getState().getBlendState().allChannelsMasked())
            {
                // Detect that if there's active color buffer without fragment shader output
                if (!ValidateFragmentShaderColorBufferMaskMatch(context))
                {
                    return kDrawBufferMaskMismatch;
                }

                // Detect that the color buffer types match the fragment shader output types
                if (!ValidateFragmentShaderColorBufferTypeMatch(context))
                {
                    return kDrawBufferTypeMismatch;
                }
            }

            const VertexArray *vao = context->getState().getVertexArray();
            if (vao->hasTransformFeedbackBindingConflict(context))
            {
                return kVertexBufferBoundForTransformFeedback;
            }
        }
    }

    return nullptr;
}

void RecordDrawModeError(Context *context, PrimitiveMode mode)
{
    const State &state                      = context->getState();
    TransformFeedback *curTransformFeedback = state.getCurrentTransformFeedback();
    if (state.isTransformFeedbackActiveUnpaused())
    {
        if (!ValidateTransformFeedbackPrimitiveMode(context,
                                                    curTransformFeedback->getPrimitiveMode(), mode))
        {
            context->validationError(GL_INVALID_OPERATION, kInvalidDrawModeTransformFeedback);
            return;
        }
    }

    const Extensions &extensions = context->getExtensions();

    switch (mode)
    {
        case PrimitiveMode::Points:
        case PrimitiveMode::Lines:
        case PrimitiveMode::LineLoop:
        case PrimitiveMode::LineStrip:
        case PrimitiveMode::Triangles:
        case PrimitiveMode::TriangleStrip:
        case PrimitiveMode::TriangleFan:
            break;

        case PrimitiveMode::LinesAdjacency:
        case PrimitiveMode::LineStripAdjacency:
        case PrimitiveMode::TrianglesAdjacency:
        case PrimitiveMode::TriangleStripAdjacency:
            if (!extensions.geometryShader)
            {
                context->validationError(GL_INVALID_ENUM, kGeometryShaderExtensionNotEnabled);
                return;
            }
            break;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidDrawMode);
            return;
    }

    // If we are running GLES1, there is no current program.
    if (context->getClientVersion() >= Version(2, 0))
    {
        Program *program = state.getLinkedProgram(context);
        ASSERT(program);

        // Do geometry shader specific validations
        if (program->hasLinkedShaderStage(ShaderType::Geometry))
        {
            if (!IsCompatibleDrawModeWithGeometryShader(
                    mode, program->getGeometryShaderInputPrimitiveType()))
            {
                context->validationError(GL_INVALID_OPERATION,
                                         kIncompatibleDrawModeAgainstGeometryShader);
                return;
            }
        }
    }

    // An error should be recorded.
    UNREACHABLE();
}

bool ValidateDrawArraysInstancedANGLE(Context *context,
                                      PrimitiveMode mode,
                                      GLint first,
                                      GLsizei count,
                                      GLsizei primcount)
{
    if (!context->getExtensions().instancedArraysANGLE)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    if (!ValidateDrawArraysInstancedBase(context, mode, first, count, primcount))
    {
        return false;
    }

    return ValidateDrawInstancedANGLE(context);
}

bool ValidateDrawArraysInstancedEXT(Context *context,
                                    PrimitiveMode mode,
                                    GLint first,
                                    GLsizei count,
                                    GLsizei primcount)
{
    if (!context->getExtensions().instancedArraysEXT)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    if (!ValidateDrawArraysInstancedBase(context, mode, first, count, primcount))
    {
        return false;
    }

    return true;
}

const char *ValidateDrawElementsStates(Context *context)
{
    const State &state = context->getState();

    if (context->getStateCache().isTransformFeedbackActiveUnpaused())
    {
        // EXT_geometry_shader allows transform feedback to work with all draw commands.
        // [EXT_geometry_shader] Section 12.1, "Transform Feedback"
        if (!context->getExtensions().geometryShader)
        {
            // It is an invalid operation to call DrawElements, DrawRangeElements or
            // DrawElementsInstanced while transform feedback is active, (3.0.2, section 2.14, pg
            // 86)
            return kUnsupportedDrawModeForTransformFeedback;
        }
    }

    const VertexArray *vao     = state.getVertexArray();
    Buffer *elementArrayBuffer = vao->getElementArrayBuffer();

    if (elementArrayBuffer)
    {
        if (context->getExtensions().webglCompatibility)
        {
            if (elementArrayBuffer->isBoundForTransformFeedbackAndOtherUse())
            {
                return kElementArrayBufferBoundForTransformFeedback;
            }
        }
        else if (elementArrayBuffer->isMapped())
        {
            // WebGL buffers cannot be mapped/unmapped because the MapBufferRange,
            // FlushMappedBufferRange, and UnmapBuffer entry points are removed from the
            // WebGL 2.0 API. https://www.khronos.org/registry/webgl/specs/latest/2.0/#5.14
            return kBufferMapped;
        }
    }
    else
    {
        // [WebGL 1.0] Section 6.2 No Client Side Arrays
        // If an indexed draw command (drawElements) is called and no WebGLBuffer is bound to
        // the ELEMENT_ARRAY_BUFFER binding point, an INVALID_OPERATION error is generated.
        if (!context->getState().areClientArraysEnabled() ||
            context->getExtensions().webglCompatibility)
        {
            return kMustHaveElementArrayBinding;
        }
    }

    return nullptr;
}

bool ValidateDrawElementsInstancedANGLE(Context *context,
                                        PrimitiveMode mode,
                                        GLsizei count,
                                        DrawElementsType type,
                                        const void *indices,
                                        GLsizei primcount)
{
    if (!context->getExtensions().instancedArraysANGLE)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    if (!ValidateDrawElementsInstancedBase(context, mode, count, type, indices, primcount))
    {
        return false;
    }

    return ValidateDrawInstancedANGLE(context);
}

bool ValidateDrawElementsInstancedEXT(Context *context,
                                      PrimitiveMode mode,
                                      GLsizei count,
                                      DrawElementsType type,
                                      const void *indices,
                                      GLsizei primcount)
{
    if (!context->getExtensions().instancedArraysEXT)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    if (!ValidateDrawElementsInstancedBase(context, mode, count, type, indices, primcount))
    {
        return false;
    }

    return true;
}

bool ValidateFramebufferTextureBase(Context *context,
                                    GLenum target,
                                    GLenum attachment,
                                    TextureID texture,
                                    GLint level)
{
    if (!ValidFramebufferTarget(context, target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidFramebufferTarget);
        return false;
    }

    if (!ValidateAttachmentTarget(context, attachment))
    {
        return false;
    }

    if (texture.value != 0)
    {
        Texture *tex = context->getTexture(texture);

        if (tex == nullptr)
        {
            context->validationError(GL_INVALID_OPERATION, kMissingTexture);
            return false;
        }

        if (level < 0)
        {
            context->validationError(GL_INVALID_VALUE, kInvalidMipLevel);
            return false;
        }
    }

    const Framebuffer *framebuffer = context->getState().getTargetFramebuffer(target);
    ASSERT(framebuffer);

    if (framebuffer->isDefault())
    {
        context->validationError(GL_INVALID_OPERATION, kDefaultFramebufferTarget);
        return false;
    }

    return true;
}

bool ValidateGetUniformBase(Context *context, ShaderProgramID program, GLint location)
{
    if (program.value == 0)
    {
        context->validationError(GL_INVALID_VALUE, kProgramDoesNotExist);
        return false;
    }

    Program *programObject = GetValidProgram(context, program);
    if (!programObject)
    {
        return false;
    }

    if (!programObject || !programObject->isLinked())
    {
        context->validationError(GL_INVALID_OPERATION, kProgramNotLinked);
        return false;
    }

    if (!programObject->isValidUniformLocation(location))
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidUniformLocation);
        return false;
    }

    return true;
}

static bool ValidateSizedGetUniform(Context *context,
                                    ShaderProgramID program,
                                    GLint location,
                                    GLsizei bufSize,
                                    GLsizei *length)
{
    if (length)
    {
        *length = 0;
    }

    if (!ValidateGetUniformBase(context, program, location))
    {
        return false;
    }

    if (bufSize < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeBufferSize);
        return false;
    }

    Program *programObject = context->getProgramResolveLink(program);
    ASSERT(programObject);

    // sized queries -- ensure the provided buffer is large enough
    const LinkedUniform &uniform = programObject->getUniformByLocation(location);
    size_t requiredBytes         = VariableExternalSize(uniform.type);
    if (static_cast<size_t>(bufSize) < requiredBytes)
    {
        context->validationError(GL_INVALID_OPERATION, kInsufficientBufferSize);
        return false;
    }

    if (length)
    {
        *length = VariableComponentCount(uniform.type);
    }

    return true;
}

bool ValidateGetnUniformfvEXT(Context *context,
                              ShaderProgramID program,
                              GLint location,
                              GLsizei bufSize,
                              GLfloat *params)
{
    return ValidateSizedGetUniform(context, program, location, bufSize, nullptr);
}

bool ValidateGetnUniformfvRobustANGLE(Context *context,
                                      ShaderProgramID program,
                                      GLint location,
                                      GLsizei bufSize,
                                      GLsizei *length,
                                      GLfloat *params)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateGetnUniformivEXT(Context *context,
                              ShaderProgramID program,
                              GLint location,
                              GLsizei bufSize,
                              GLint *params)
{
    return ValidateSizedGetUniform(context, program, location, bufSize, nullptr);
}

bool ValidateGetnUniformivRobustANGLE(Context *context,
                                      ShaderProgramID program,
                                      GLint location,
                                      GLsizei bufSize,
                                      GLsizei *length,
                                      GLint *params)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateGetnUniformuivRobustANGLE(Context *context,
                                       ShaderProgramID program,
                                       GLint location,
                                       GLsizei bufSize,
                                       GLsizei *length,
                                       GLuint *params)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateGetUniformfvRobustANGLE(Context *context,
                                     ShaderProgramID program,
                                     GLint location,
                                     GLsizei bufSize,
                                     GLsizei *length,
                                     GLfloat *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei writeLength = 0;

    // bufSize is validated in ValidateSizedGetUniform
    if (!ValidateSizedGetUniform(context, program, location, bufSize, &writeLength))
    {
        return false;
    }

    SetRobustLengthParam(length, writeLength);

    return true;
}

bool ValidateGetUniformivRobustANGLE(Context *context,
                                     ShaderProgramID program,
                                     GLint location,
                                     GLsizei bufSize,
                                     GLsizei *length,
                                     GLint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei writeLength = 0;

    // bufSize is validated in ValidateSizedGetUniform
    if (!ValidateSizedGetUniform(context, program, location, bufSize, &writeLength))
    {
        return false;
    }

    SetRobustLengthParam(length, writeLength);

    return true;
}

bool ValidateGetUniformuivRobustANGLE(Context *context,
                                      ShaderProgramID program,
                                      GLint location,
                                      GLsizei bufSize,
                                      GLsizei *length,
                                      GLuint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    if (context->getClientMajorVersion() < 3)
    {
        context->validationError(GL_INVALID_OPERATION, kES3Required);
        return false;
    }

    GLsizei writeLength = 0;

    // bufSize is validated in ValidateSizedGetUniform
    if (!ValidateSizedGetUniform(context, program, location, bufSize, &writeLength))
    {
        return false;
    }

    SetRobustLengthParam(length, writeLength);

    return true;
}

bool ValidateDiscardFramebufferBase(Context *context,
                                    GLenum target,
                                    GLsizei numAttachments,
                                    const GLenum *attachments,
                                    bool defaultFramebuffer)
{
    if (numAttachments < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeAttachments);
        return false;
    }

    for (GLsizei i = 0; i < numAttachments; ++i)
    {
        if (attachments[i] >= GL_COLOR_ATTACHMENT0 && attachments[i] <= GL_COLOR_ATTACHMENT31)
        {
            if (defaultFramebuffer)
            {
                context->validationError(GL_INVALID_ENUM, kDefaultFramebufferInvalidAttachment);
                return false;
            }

            if (attachments[i] >= GL_COLOR_ATTACHMENT0 + context->getCaps().maxColorAttachments)
            {
                context->validationError(GL_INVALID_OPERATION, kExceedsMaxColorAttachments);
                return false;
            }
        }
        else
        {
            switch (attachments[i])
            {
                case GL_DEPTH_ATTACHMENT:
                case GL_STENCIL_ATTACHMENT:
                case GL_DEPTH_STENCIL_ATTACHMENT:
                    if (defaultFramebuffer)
                    {
                        context->validationError(GL_INVALID_ENUM,
                                                 kDefaultFramebufferInvalidAttachment);
                        return false;
                    }
                    break;
                case GL_COLOR:
                case GL_DEPTH:
                case GL_STENCIL:
                    if (!defaultFramebuffer)
                    {
                        context->validationError(GL_INVALID_ENUM,
                                                 kDefaultFramebufferInvalidAttachment);
                        return false;
                    }
                    break;
                default:
                    context->validationError(GL_INVALID_ENUM, kInvalidAttachment);
                    return false;
            }
        }
    }

    return true;
}

bool ValidateInsertEventMarkerEXT(Context *context, GLsizei length, const char *marker)
{
    if (!context->getExtensions().debugMarker)
    {
        // The debug marker calls should not set error state
        // However, it seems reasonable to set an error state if the extension is not enabled
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    // Note that debug marker calls must not set error state
    if (length < 0)
    {
        return false;
    }

    if (marker == nullptr)
    {
        return false;
    }

    return true;
}

bool ValidatePushGroupMarkerEXT(Context *context, GLsizei length, const char *marker)
{
    if (!context->getExtensions().debugMarker)
    {
        // The debug marker calls should not set error state
        // However, it seems reasonable to set an error state if the extension is not enabled
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    // Note that debug marker calls must not set error state
    if (length < 0)
    {
        return false;
    }

    if (length > 0 && marker == nullptr)
    {
        return false;
    }

    return true;
}

bool ValidateEGLImageTargetTexture2DOES(Context *context, TextureType type, GLeglImageOES image)
{
    if (!context->getExtensions().eglImage && !context->getExtensions().eglImageExternal)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    switch (type)
    {
        case TextureType::_2D:
            if (!context->getExtensions().eglImage)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            }
            break;

        case TextureType::External:
            if (!context->getExtensions().eglImageExternal)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            }
            break;

        case TextureType::CubeMap:
            if (!context->getExtensions().eglImageCubeMGL)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            }
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kInvalidTextureTarget);
            return false;
    }

    egl::Image *imageObject = static_cast<egl::Image *>(image);

    ASSERT(context->getDisplay());
    if (!context->getDisplay()->isValidImage(imageObject))
    {
        context->validationError(GL_INVALID_VALUE, kInvalidEGLImage);
        return false;
    }

    if (imageObject->getSamples() > 0)
    {
        context->validationError(GL_INVALID_OPERATION, kEGLImageCannotCreate2DMultisampled);
        return false;
    }

    if (!imageObject->isTexturable(context))
    {
        context->validationError(GL_INVALID_OPERATION, kEGLImageTextureFormatNotSupported);
        return false;
    }

    return true;
}

bool ValidateEGLImageTargetRenderbufferStorageOES(Context *context,
                                                  GLenum target,
                                                  GLeglImageOES image)
{
    if (!context->getExtensions().eglImage)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    switch (target)
    {
        case GL_RENDERBUFFER:
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kInvalidRenderbufferTarget);
            return false;
    }

    egl::Image *imageObject = static_cast<egl::Image *>(image);

    ASSERT(context->getDisplay());
    if (!context->getDisplay()->isValidImage(imageObject))
    {
        context->validationError(GL_INVALID_VALUE, kInvalidEGLImage);
        return false;
    }

    if (!imageObject->isRenderable(context))
    {
        context->validationError(GL_INVALID_OPERATION, kEGLImageRenderbufferFormatNotSupported);
        return false;
    }

    return true;
}

bool ValidateBindVertexArrayBase(Context *context, VertexArrayID array)
{
    if (!context->isVertexArrayGenerated(array))
    {
        // The default VAO should always exist
        ASSERT(array.value != 0);
        context->validationError(GL_INVALID_OPERATION, kInvalidVertexArray);
        return false;
    }

    return true;
}

bool ValidateProgramBinaryBase(Context *context,
                               ShaderProgramID program,
                               GLenum binaryFormat,
                               const void *binary,
                               GLint length)
{
    Program *programObject = GetValidProgram(context, program);
    if (programObject == nullptr)
    {
        return false;
    }

    const std::vector<GLenum> &programBinaryFormats = context->getCaps().programBinaryFormats;
    if (std::find(programBinaryFormats.begin(), programBinaryFormats.end(), binaryFormat) ==
        programBinaryFormats.end())
    {
        context->validationError(GL_INVALID_ENUM, kInvalidProgramBinaryFormat);
        return false;
    }

    if (context->hasActiveTransformFeedback(program))
    {
        // ES 3.0.4 section 2.15 page 91
        context->validationError(GL_INVALID_OPERATION, kTransformFeedbackProgramBinary);
        return false;
    }

    return true;
}

bool ValidateGetProgramBinaryBase(Context *context,
                                  ShaderProgramID program,
                                  GLsizei bufSize,
                                  GLsizei *length,
                                  GLenum *binaryFormat,
                                  void *binary)
{
    Program *programObject = GetValidProgram(context, program);
    if (programObject == nullptr)
    {
        return false;
    }

    if (!programObject->isLinked())
    {
        context->validationError(GL_INVALID_OPERATION, kProgramNotLinked);
        return false;
    }

    if (context->getCaps().programBinaryFormats.empty())
    {
        context->validationError(GL_INVALID_OPERATION, kNoProgramBinaryFormats);
        return false;
    }

    return true;
}

bool ValidateDrawBuffersBase(Context *context, GLsizei n, const GLenum *bufs)
{
    // INVALID_VALUE is generated if n is negative or greater than value of MAX_DRAW_BUFFERS
    if (n < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeCount);
        return false;
    }
    if (static_cast<GLuint>(n) > context->getCaps().maxDrawBuffers)
    {
        context->validationError(GL_INVALID_VALUE, kIndexExceedsMaxDrawBuffer);
        return false;
    }

    ASSERT(context->getState().getDrawFramebuffer());
    FramebufferID frameBufferId = context->getState().getDrawFramebuffer()->id();
    GLuint maxColorAttachment   = GL_COLOR_ATTACHMENT0_EXT + context->getCaps().maxColorAttachments;

    // This should come first before the check for the default frame buffer
    // because when we switch to ES3.1+, invalid enums will return INVALID_ENUM
    // rather than INVALID_OPERATION
    for (int colorAttachment = 0; colorAttachment < n; colorAttachment++)
    {
        const GLenum attachment = GL_COLOR_ATTACHMENT0_EXT + colorAttachment;

        if (bufs[colorAttachment] != GL_NONE && bufs[colorAttachment] != GL_BACK &&
            (bufs[colorAttachment] < GL_COLOR_ATTACHMENT0 ||
             bufs[colorAttachment] > GL_COLOR_ATTACHMENT31))
        {
            // Value in bufs is not NONE, BACK, or GL_COLOR_ATTACHMENTi
            // The 3.0.4 spec says to generate GL_INVALID_OPERATION here, but this
            // was changed to GL_INVALID_ENUM in 3.1, which dEQP also expects.
            // 3.1 is still a bit ambiguous about the error, but future specs are
            // expected to clarify that GL_INVALID_ENUM is the correct error.
            context->validationError(GL_INVALID_ENUM, kInvalidDrawBuffer);
            return false;
        }
        else if (bufs[colorAttachment] >= maxColorAttachment)
        {
            context->validationError(GL_INVALID_OPERATION, kExceedsMaxColorAttachments);
            return false;
        }
        else if (bufs[colorAttachment] != GL_NONE && bufs[colorAttachment] != attachment &&
                 frameBufferId.value != 0)
        {
            // INVALID_OPERATION-GL is bound to buffer and ith argument
            // is not COLOR_ATTACHMENTi or NONE
            context->validationError(GL_INVALID_OPERATION, kInvalidDrawBufferValue);
            return false;
        }
    }

    // INVALID_OPERATION is generated if GL is bound to the default framebuffer
    // and n is not 1 or bufs is bound to value other than BACK and NONE
    if (frameBufferId.value == 0)
    {
        if (n != 1)
        {
            context->validationError(GL_INVALID_OPERATION, kInvalidDrawBufferCountForDefault);
            return false;
        }

        if (bufs[0] != GL_NONE && bufs[0] != GL_BACK)
        {
            context->validationError(GL_INVALID_OPERATION, kDefaultFramebufferInvalidDrawBuffer);
            return false;
        }
    }

    return true;
}

bool ValidateGetBufferPointervBase(Context *context,
                                   BufferBinding target,
                                   GLenum pname,
                                   GLsizei *length,
                                   void **params)
{
    if (length)
    {
        *length = 0;
    }

    if (!context->isValidBufferBinding(target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidBufferTypes);
        return false;
    }

    switch (pname)
    {
        case GL_BUFFER_MAP_POINTER:
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
    }

    // GLES 3.0 section 2.10.1: "Attempts to attempts to modify or query buffer object state for a
    // target bound to zero generate an INVALID_OPERATION error."
    // GLES 3.1 section 6.6 explicitly specifies this error.
    if (context->getState().getTargetBuffer(target) == nullptr)
    {
        context->validationError(GL_INVALID_OPERATION, kBufferPointerNotAvailable);
        return false;
    }

    if (length)
    {
        *length = 1;
    }

    return true;
}

bool ValidateUnmapBufferBase(Context *context, BufferBinding target)
{
    if (!context->isValidBufferBinding(target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidBufferTypes);
        return false;
    }

    Buffer *buffer = context->getState().getTargetBuffer(target);

    if (buffer == nullptr || !buffer->isMapped())
    {
        context->validationError(GL_INVALID_OPERATION, kBufferNotMapped);
        return false;
    }

    return true;
}

bool ValidateMapBufferRangeBase(Context *context,
                                BufferBinding target,
                                GLintptr offset,
                                GLsizeiptr length,
                                GLbitfield access)
{
    if (!context->isValidBufferBinding(target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidBufferTypes);
        return false;
    }

    if (offset < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeOffset);
        return false;
    }

    if (length < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeLength);
        return false;
    }

    Buffer *buffer = context->getState().getTargetBuffer(target);

    if (!buffer)
    {
        context->validationError(GL_INVALID_OPERATION, kBufferNotMappable);
        return false;
    }

    // Check for buffer overflow
    CheckedNumeric<size_t> checkedOffset(offset);
    auto checkedSize = checkedOffset + length;

    if (!checkedSize.IsValid() || checkedSize.ValueOrDie() > static_cast<size_t>(buffer->getSize()))
    {
        context->validationError(GL_INVALID_VALUE, kMapOutOfRange);
        return false;
    }

    // Check for invalid bits in the mask
    GLbitfield allAccessBits = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT |
                               GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_FLUSH_EXPLICIT_BIT |
                               GL_MAP_UNSYNCHRONIZED_BIT;

    if (access & ~(allAccessBits))
    {
        context->validationError(GL_INVALID_VALUE, kInvalidAccessBits);
        return false;
    }

    if (length == 0)
    {
        context->validationError(GL_INVALID_OPERATION, kLengthZero);
        return false;
    }

    if (buffer->isMapped())
    {
        context->validationError(GL_INVALID_OPERATION, kBufferAlreadyMapped);
        return false;
    }

    // Check for invalid bit combinations
    if ((access & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT)) == 0)
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidAccessBitsReadWrite);
        return false;
    }

    GLbitfield writeOnlyBits =
        GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT;

    if ((access & GL_MAP_READ_BIT) != 0 && (access & writeOnlyBits) != 0)
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidAccessBitsRead);
        return false;
    }

    if ((access & GL_MAP_WRITE_BIT) == 0 && (access & GL_MAP_FLUSH_EXPLICIT_BIT) != 0)
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidAccessBitsFlush);
        return false;
    }

    return ValidateMapBufferBase(context, target);
}

bool ValidateFlushMappedBufferRangeBase(Context *context,
                                        BufferBinding target,
                                        GLintptr offset,
                                        GLsizeiptr length)
{
    if (offset < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeOffset);
        return false;
    }

    if (length < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeLength);
        return false;
    }

    if (!context->isValidBufferBinding(target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidBufferTypes);
        return false;
    }

    Buffer *buffer = context->getState().getTargetBuffer(target);

    if (buffer == nullptr)
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidFlushZero);
        return false;
    }

    if (!buffer->isMapped() || (buffer->getAccessFlags() & GL_MAP_FLUSH_EXPLICIT_BIT) == 0)
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidFlushTarget);
        return false;
    }

    // Check for buffer overflow
    CheckedNumeric<size_t> checkedOffset(offset);
    auto checkedSize = checkedOffset + length;

    if (!checkedSize.IsValid() ||
        checkedSize.ValueOrDie() > static_cast<size_t>(buffer->getMapLength()))
    {
        context->validationError(GL_INVALID_VALUE, kInvalidFlushOutOfRange);
        return false;
    }

    return true;
}

bool ValidateGenOrDelete(Context *context, GLint n)
{
    if (n < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeCount);
        return false;
    }
    return true;
}

bool ValidateRobustEntryPoint(Context *context, GLsizei bufSize)
{
    if (!context->getExtensions().robustClientMemory)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    if (bufSize < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeBufferSize);
        return false;
    }

    return true;
}

bool ValidateRobustBufferSize(Context *context, GLsizei bufSize, GLsizei numParams)
{
    if (bufSize < numParams)
    {
        context->validationError(GL_INVALID_OPERATION, kInsufficientParams);
        return false;
    }

    return true;
}

bool ValidateGetFramebufferAttachmentParameterivBase(Context *context,
                                                     GLenum target,
                                                     GLenum attachment,
                                                     GLenum pname,
                                                     GLsizei *numParams)
{
    if (!ValidFramebufferTarget(context, target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidFramebufferTarget);
        return false;
    }

    int clientVersion = context->getClientMajorVersion();

    switch (pname)
    {
        case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE:
        case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME:
        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL:
        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE:
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR:
        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_BASE_VIEW_INDEX_OVR:
            if (clientVersion < 3 ||
                !(context->getExtensions().multiview || context->getExtensions().multiview2))
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_SAMPLES_EXT:
            if (!context->getExtensions().multisampledRenderToTexture)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING:
            if (clientVersion < 3 && !context->getExtensions().sRGB)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE:
        case GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE:
        case GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE:
        case GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE:
        case GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE:
        case GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE:
        case GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE:
        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER:
            if (clientVersion < 3)
            {
                context->validationError(GL_INVALID_ENUM, kES3Required);
                return false;
            }
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_LAYERED_EXT:
            if (!context->getExtensions().geometryShader)
            {
                context->validationError(GL_INVALID_ENUM, kGeometryShaderExtensionNotEnabled);
                return false;
            }
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kInvalidPname);
            return false;
    }

    // Determine if the attachment is a valid enum
    switch (attachment)
    {
        case GL_BACK:
        case GL_DEPTH:
        case GL_STENCIL:
            if (clientVersion < 3)
            {
                context->validationError(GL_INVALID_ENUM, kInvalidAttachment);
                return false;
            }
            break;

        case GL_DEPTH_STENCIL_ATTACHMENT:
            if (clientVersion < 3 && !context->isWebGL1())
            {
                context->validationError(GL_INVALID_ENUM, kInvalidAttachment);
                return false;
            }
            break;

        case GL_COLOR_ATTACHMENT0:
        case GL_DEPTH_ATTACHMENT:
        case GL_STENCIL_ATTACHMENT:
            break;

        default:
            if ((clientVersion < 3 && !context->getExtensions().drawBuffers) ||
                attachment < GL_COLOR_ATTACHMENT0_EXT ||
                (attachment - GL_COLOR_ATTACHMENT0_EXT) >= context->getCaps().maxColorAttachments)
            {
                context->validationError(GL_INVALID_ENUM, kInvalidAttachment);
                return false;
            }
            break;
    }

    const Framebuffer *framebuffer = context->getState().getTargetFramebuffer(target);
    ASSERT(framebuffer);

    if (framebuffer->isDefault())
    {
        if (clientVersion < 3)
        {
            context->validationError(GL_INVALID_OPERATION, kDefaultFramebufferTarget);
            return false;
        }

        switch (attachment)
        {
            case GL_BACK:
            case GL_DEPTH:
            case GL_STENCIL:
                break;

            default:
                context->validationError(GL_INVALID_OPERATION, kInvalidAttachment);
                return false;
        }
    }
    else
    {
        if (attachment >= GL_COLOR_ATTACHMENT0_EXT && attachment <= GL_COLOR_ATTACHMENT15_EXT)
        {
            // Valid attachment query
        }
        else
        {
            switch (attachment)
            {
                case GL_DEPTH_ATTACHMENT:
                case GL_STENCIL_ATTACHMENT:
                    break;

                case GL_DEPTH_STENCIL_ATTACHMENT:
                    if (!framebuffer->hasValidDepthStencil() && !context->isWebGL1())
                    {
                        context->validationError(GL_INVALID_OPERATION, kInvalidAttachment);
                        return false;
                    }
                    break;

                default:
                    context->validationError(GL_INVALID_OPERATION, kInvalidAttachment);
                    return false;
            }
        }
    }

    const FramebufferAttachment *attachmentObject = framebuffer->getAttachment(context, attachment);
    if (attachmentObject)
    {
        ASSERT(attachmentObject->type() == GL_RENDERBUFFER ||
               attachmentObject->type() == GL_TEXTURE ||
               attachmentObject->type() == GL_FRAMEBUFFER_DEFAULT);

        switch (pname)
        {
            case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME:
                if (attachmentObject->type() != GL_RENDERBUFFER &&
                    attachmentObject->type() != GL_TEXTURE)
                {
                    context->validationError(GL_INVALID_ENUM, kFramebufferIncompleteAttachment);
                    return false;
                }
                break;

            case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL:
                if (attachmentObject->type() != GL_TEXTURE)
                {
                    context->validationError(GL_INVALID_ENUM, kFramebufferIncompleteAttachment);
                    return false;
                }
                break;

            case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE:
                if (attachmentObject->type() != GL_TEXTURE)
                {
                    context->validationError(GL_INVALID_ENUM, kFramebufferIncompleteAttachment);
                    return false;
                }
                break;

            case GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE:
                if (attachment == GL_DEPTH_STENCIL_ATTACHMENT)
                {
                    context->validationError(GL_INVALID_OPERATION, kInvalidAttachment);
                    return false;
                }
                break;

            case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER:
                if (attachmentObject->type() != GL_TEXTURE)
                {
                    context->validationError(GL_INVALID_ENUM, kFramebufferIncompleteAttachment);
                    return false;
                }
                break;

            default:
                break;
        }
    }
    else
    {
        // ES 2.0.25 spec pg 127 states that if the value of FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE
        // is NONE, then querying any other pname will generate INVALID_ENUM.

        // ES 3.0.2 spec pg 235 states that if the attachment type is none,
        // GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME will return zero and be an
        // INVALID_OPERATION for all other pnames

        switch (pname)
        {
            case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE:
                break;

            case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME:
                if (clientVersion < 3)
                {
                    context->validationError(GL_INVALID_ENUM,
                                             kInvalidFramebufferAttachmentParameter);
                    return false;
                }
                break;

            default:
                if (clientVersion < 3)
                {
                    context->validationError(GL_INVALID_ENUM,
                                             kInvalidFramebufferAttachmentParameter);
                    return false;
                }
                else
                {
                    context->validationError(GL_INVALID_OPERATION,
                                             kInvalidFramebufferAttachmentParameter);
                    return false;
                }
        }
    }

    if (numParams)
    {
        *numParams = 1;
    }

    return true;
}

bool ValidateGetFramebufferAttachmentParameterivRobustANGLE(Context *context,
                                                            GLenum target,
                                                            GLenum attachment,
                                                            GLenum pname,
                                                            GLsizei bufSize,
                                                            GLsizei *length,
                                                            GLint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;
    if (!ValidateGetFramebufferAttachmentParameterivBase(context, target, attachment, pname,
                                                         &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

bool ValidateGetBufferParameterivRobustANGLE(Context *context,
                                             BufferBinding target,
                                             GLenum pname,
                                             GLsizei bufSize,
                                             GLsizei *length,
                                             GLint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;

    if (!ValidateGetBufferParameterBase(context, target, pname, false, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);
    return true;
}

bool ValidateGetBufferParameteri64vRobustANGLE(Context *context,
                                               BufferBinding target,
                                               GLenum pname,
                                               GLsizei bufSize,
                                               GLsizei *length,
                                               GLint64 *params)
{
    GLsizei numParams = 0;

    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    if (!ValidateGetBufferParameterBase(context, target, pname, false, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

bool ValidateGetProgramivBase(Context *context,
                              ShaderProgramID program,
                              GLenum pname,
                              GLsizei *numParams)
{
    // Currently, all GetProgramiv queries return 1 parameter
    if (numParams)
    {
        *numParams = 1;
    }

    if (context->isContextLost())
    {
        context->validationError(GL_CONTEXT_LOST, kContextLost);

        if (context->getExtensions().parallelShaderCompile && pname == GL_COMPLETION_STATUS_KHR)
        {
            // Generate an error but still return true, the context still needs to return a
            // value in this case.
            return true;
        }
        else
        {
            return false;
        }
    }

    // Special case for GL_COMPLETION_STATUS_KHR: don't resolve the link. Otherwise resolve it now.
    Program *programObject = (pname == GL_COMPLETION_STATUS_KHR)
                                 ? GetValidProgramNoResolve(context, program)
                                 : GetValidProgram(context, program);
    if (!programObject)
    {
        return false;
    }

    switch (pname)
    {
        case GL_DELETE_STATUS:
        case GL_LINK_STATUS:
        case GL_VALIDATE_STATUS:
        case GL_INFO_LOG_LENGTH:
        case GL_ATTACHED_SHADERS:
        case GL_ACTIVE_ATTRIBUTES:
        case GL_ACTIVE_ATTRIBUTE_MAX_LENGTH:
        case GL_ACTIVE_UNIFORMS:
        case GL_ACTIVE_UNIFORM_MAX_LENGTH:
            break;

        case GL_PROGRAM_BINARY_LENGTH:
            if (context->getClientMajorVersion() < 3 && !context->getExtensions().getProgramBinary)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;

        case GL_ACTIVE_UNIFORM_BLOCKS:
        case GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH:
        case GL_TRANSFORM_FEEDBACK_BUFFER_MODE:
        case GL_TRANSFORM_FEEDBACK_VARYINGS:
        case GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH:
        case GL_PROGRAM_BINARY_RETRIEVABLE_HINT:
            if (context->getClientMajorVersion() < 3)
            {
                context->validationError(GL_INVALID_ENUM, kEnumRequiresGLES30);
                return false;
            }
            break;

        case GL_PROGRAM_SEPARABLE:
        case GL_ACTIVE_ATOMIC_COUNTER_BUFFERS:
            if (context->getClientVersion() < Version(3, 1))
            {
                context->validationError(GL_INVALID_ENUM, kEnumRequiresGLES31);
                return false;
            }
            break;

        case GL_COMPUTE_WORK_GROUP_SIZE:
            if (context->getClientVersion() < Version(3, 1))
            {
                context->validationError(GL_INVALID_ENUM, kEnumRequiresGLES31);
                return false;
            }

            // [OpenGL ES 3.1] Chapter 7.12 Page 122
            // An INVALID_OPERATION error is generated if COMPUTE_WORK_GROUP_SIZE is queried for a
            // program which has not been linked successfully, or which does not contain objects to
            // form a compute shader.
            if (!programObject->isLinked())
            {
                context->validationError(GL_INVALID_OPERATION, kProgramNotLinked);
                return false;
            }
            if (!programObject->hasLinkedShaderStage(ShaderType::Compute))
            {
                context->validationError(GL_INVALID_OPERATION, kNoActiveComputeShaderStage);
                return false;
            }
            break;

        case GL_GEOMETRY_LINKED_INPUT_TYPE_EXT:
        case GL_GEOMETRY_LINKED_OUTPUT_TYPE_EXT:
        case GL_GEOMETRY_LINKED_VERTICES_OUT_EXT:
        case GL_GEOMETRY_SHADER_INVOCATIONS_EXT:
            if (!context->getExtensions().geometryShader)
            {
                context->validationError(GL_INVALID_ENUM, kGeometryShaderExtensionNotEnabled);
                return false;
            }

            // [EXT_geometry_shader] Chapter 7.12
            // An INVALID_OPERATION error is generated if GEOMETRY_LINKED_VERTICES_OUT_EXT,
            // GEOMETRY_LINKED_INPUT_TYPE_EXT, GEOMETRY_LINKED_OUTPUT_TYPE_EXT, or
            // GEOMETRY_SHADER_INVOCATIONS_EXT are queried for a program which has not been linked
            // successfully, or which does not contain objects to form a geometry shader.
            if (!programObject->isLinked())
            {
                context->validationError(GL_INVALID_OPERATION, kProgramNotLinked);
                return false;
            }
            if (!programObject->hasLinkedShaderStage(ShaderType::Geometry))
            {
                context->validationError(GL_INVALID_OPERATION, kNoActiveGeometryShaderStage);
                return false;
            }
            break;

        case GL_COMPLETION_STATUS_KHR:
            if (!context->getExtensions().parallelShaderCompile)
            {
                context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
                return false;
            }
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
    }

    return true;
}

bool ValidateGetProgramivRobustANGLE(Context *context,
                                     ShaderProgramID program,
                                     GLenum pname,
                                     GLsizei bufSize,
                                     GLsizei *length,
                                     GLint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;

    if (!ValidateGetProgramivBase(context, program, pname, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

bool ValidateGetRenderbufferParameterivRobustANGLE(Context *context,
                                                   GLenum target,
                                                   GLenum pname,
                                                   GLsizei bufSize,
                                                   GLsizei *length,
                                                   GLint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;

    if (!ValidateGetRenderbufferParameterivBase(context, target, pname, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

bool ValidateGetShaderivRobustANGLE(Context *context,
                                    ShaderProgramID shader,
                                    GLenum pname,
                                    GLsizei bufSize,
                                    GLsizei *length,
                                    GLint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;

    if (!ValidateGetShaderivBase(context, shader, pname, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

bool ValidateGetTexParameterfvRobustANGLE(Context *context,
                                          TextureType target,
                                          GLenum pname,
                                          GLsizei bufSize,
                                          GLsizei *length,
                                          GLfloat *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;

    if (!ValidateGetTexParameterBase(context, target, pname, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

bool ValidateGetTexParameterivRobustANGLE(Context *context,
                                          TextureType target,
                                          GLenum pname,
                                          GLsizei bufSize,
                                          GLsizei *length,
                                          GLint *params)
{

    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }
    GLsizei numParams = 0;
    if (!ValidateGetTexParameterBase(context, target, pname, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);
    return true;
}

bool ValidateGetTexParameterIivRobustANGLE(Context *context,
                                           TextureType target,
                                           GLenum pname,
                                           GLsizei bufSize,
                                           GLsizei *length,
                                           GLint *params)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateGetTexParameterIuivRobustANGLE(Context *context,
                                            TextureType target,
                                            GLenum pname,
                                            GLsizei bufSize,
                                            GLsizei *length,
                                            GLuint *params)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateTexParameterfvRobustANGLE(Context *context,
                                       TextureType target,
                                       GLenum pname,
                                       GLsizei bufSize,
                                       const GLfloat *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    return ValidateTexParameterBase(context, target, pname, bufSize, true, params);
}

bool ValidateTexParameterivRobustANGLE(Context *context,
                                       TextureType target,
                                       GLenum pname,
                                       GLsizei bufSize,
                                       const GLint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    return ValidateTexParameterBase(context, target, pname, bufSize, true, params);
}

bool ValidateTexParameterIivRobustANGLE(Context *context,
                                        TextureType target,
                                        GLenum pname,
                                        GLsizei bufSize,
                                        const GLint *params)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateTexParameterIuivRobustANGLE(Context *context,
                                         TextureType target,
                                         GLenum pname,
                                         GLsizei bufSize,
                                         const GLuint *params)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateGetSamplerParameterfvRobustANGLE(Context *context,
                                              SamplerID sampler,
                                              GLenum pname,
                                              GLsizei bufSize,
                                              GLsizei *length,
                                              GLfloat *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;

    if (!ValidateGetSamplerParameterBase(context, sampler, pname, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);
    return true;
}

bool ValidateGetSamplerParameterivRobustANGLE(Context *context,
                                              SamplerID sampler,
                                              GLenum pname,
                                              GLsizei bufSize,
                                              GLsizei *length,
                                              GLint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;

    if (!ValidateGetSamplerParameterBase(context, sampler, pname, &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);
    return true;
}

bool ValidateGetSamplerParameterIivRobustANGLE(Context *context,
                                               SamplerID sampler,
                                               GLenum pname,
                                               GLsizei bufSize,
                                               GLsizei *length,
                                               GLint *params)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateGetSamplerParameterIuivRobustANGLE(Context *context,
                                                SamplerID sampler,
                                                GLenum pname,
                                                GLsizei bufSize,
                                                GLsizei *length,
                                                GLuint *params)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateSamplerParameterfvRobustANGLE(Context *context,
                                           SamplerID sampler,
                                           GLenum pname,
                                           GLsizei bufSize,
                                           const GLfloat *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    return ValidateSamplerParameterBase(context, sampler, pname, bufSize, true, params);
}

bool ValidateSamplerParameterivRobustANGLE(Context *context,
                                           SamplerID sampler,
                                           GLenum pname,
                                           GLsizei bufSize,
                                           const GLint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    return ValidateSamplerParameterBase(context, sampler, pname, bufSize, true, params);
}

bool ValidateSamplerParameterIivRobustANGLE(Context *context,
                                            SamplerID sampler,
                                            GLenum pname,
                                            GLsizei bufSize,
                                            const GLint *param)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateSamplerParameterIuivRobustANGLE(Context *context,
                                             SamplerID sampler,
                                             GLenum pname,
                                             GLsizei bufSize,
                                             const GLuint *param)
{
    UNIMPLEMENTED();
    return false;
}

bool ValidateGetVertexAttribfvRobustANGLE(Context *context,
                                          GLuint index,
                                          GLenum pname,
                                          GLsizei bufSize,
                                          GLsizei *length,
                                          GLfloat *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei writeLength = 0;

    if (!ValidateGetVertexAttribBase(context, index, pname, &writeLength, false, false))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, writeLength))
    {
        return false;
    }

    SetRobustLengthParam(length, writeLength);
    return true;
}

bool ValidateGetVertexAttribivRobustANGLE(Context *context,
                                          GLuint index,
                                          GLenum pname,
                                          GLsizei bufSize,
                                          GLsizei *length,
                                          GLint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei writeLength = 0;

    if (!ValidateGetVertexAttribBase(context, index, pname, &writeLength, false, false))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, writeLength))
    {
        return false;
    }

    SetRobustLengthParam(length, writeLength);

    return true;
}

bool ValidateGetVertexAttribPointervRobustANGLE(Context *context,
                                                GLuint index,
                                                GLenum pname,
                                                GLsizei bufSize,
                                                GLsizei *length,
                                                void **pointer)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei writeLength = 0;

    if (!ValidateGetVertexAttribBase(context, index, pname, &writeLength, true, false))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, writeLength))
    {
        return false;
    }

    SetRobustLengthParam(length, writeLength);

    return true;
}

bool ValidateGetVertexAttribIivRobustANGLE(Context *context,
                                           GLuint index,
                                           GLenum pname,
                                           GLsizei bufSize,
                                           GLsizei *length,
                                           GLint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei writeLength = 0;

    if (!ValidateGetVertexAttribBase(context, index, pname, &writeLength, false, true))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, writeLength))
    {
        return false;
    }

    SetRobustLengthParam(length, writeLength);

    return true;
}

bool ValidateGetVertexAttribIuivRobustANGLE(Context *context,
                                            GLuint index,
                                            GLenum pname,
                                            GLsizei bufSize,
                                            GLsizei *length,
                                            GLuint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei writeLength = 0;

    if (!ValidateGetVertexAttribBase(context, index, pname, &writeLength, false, true))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, writeLength))
    {
        return false;
    }

    SetRobustLengthParam(length, writeLength);

    return true;
}

bool ValidateGetActiveUniformBlockivRobustANGLE(Context *context,
                                                ShaderProgramID program,
                                                GLuint uniformBlockIndex,
                                                GLenum pname,
                                                GLsizei bufSize,
                                                GLsizei *length,
                                                GLint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei writeLength = 0;

    if (!ValidateGetActiveUniformBlockivBase(context, program, uniformBlockIndex, pname,
                                             &writeLength))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, writeLength))
    {
        return false;
    }

    SetRobustLengthParam(length, writeLength);

    return true;
}

bool ValidateGetInternalformativRobustANGLE(Context *context,
                                            GLenum target,
                                            GLenum internalformat,
                                            GLenum pname,
                                            GLsizei bufSize,
                                            GLsizei *length,
                                            GLint *params)
{
    if (!ValidateRobustEntryPoint(context, bufSize))
    {
        return false;
    }

    GLsizei numParams = 0;

    if (!ValidateGetInternalFormativBase(context, target, internalformat, pname, bufSize,
                                         &numParams))
    {
        return false;
    }

    if (!ValidateRobustBufferSize(context, bufSize, numParams))
    {
        return false;
    }

    SetRobustLengthParam(length, numParams);

    return true;
}

// Perform validation from WebGL 2 section 5.10 "Invalid Clears":
// In the WebGL 2 API, trying to perform a clear when there is a mismatch between the type of the
// specified clear value and the type of a buffer that is being cleared generates an
// INVALID_OPERATION error instead of producing undefined results
bool ValidateWebGLFramebufferAttachmentClearType(Context *context,
                                                 GLint drawbuffer,
                                                 const GLenum *validComponentTypes,
                                                 size_t validComponentTypeCount)
{
    const FramebufferAttachment *attachment =
        context->getState().getDrawFramebuffer()->getDrawBuffer(drawbuffer);
    if (attachment)
    {
        GLenum componentType = attachment->getFormat().info->componentType;
        const GLenum *end    = validComponentTypes + validComponentTypeCount;
        if (std::find(validComponentTypes, end, componentType) == end)
        {
            context->validationError(GL_INVALID_OPERATION, kNoDefinedClearConversion);
            return false;
        }
    }

    return true;
}

bool ValidateRobustCompressedTexImageBase(Context *context, GLsizei imageSize, GLsizei dataSize)
{
    if (!ValidateRobustEntryPoint(context, dataSize))
    {
        return false;
    }

    Buffer *pixelUnpackBuffer = context->getState().getTargetBuffer(BufferBinding::PixelUnpack);
    if (pixelUnpackBuffer == nullptr)
    {
        if (dataSize < imageSize)
        {
            context->validationError(GL_INVALID_OPERATION, kCompressedDataSizeTooSmall);
        }
    }
    return true;
}

bool ValidateGetBufferParameterBase(Context *context,
                                    BufferBinding target,
                                    GLenum pname,
                                    bool pointerVersion,
                                    GLsizei *numParams)
{
    if (numParams)
    {
        *numParams = 0;
    }

    if (!context->isValidBufferBinding(target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidBufferTypes);
        return false;
    }

    const Buffer *buffer = context->getState().getTargetBuffer(target);
    if (!buffer)
    {
        // A null buffer means that "0" is bound to the requested buffer target
        context->validationError(GL_INVALID_OPERATION, kBufferNotBound);
        return false;
    }

    const Extensions &extensions = context->getExtensions();

    switch (pname)
    {
        case GL_BUFFER_USAGE:
        case GL_BUFFER_SIZE:
            break;

        case GL_BUFFER_ACCESS_OES:
            if (!extensions.mapBuffer)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;

        case GL_BUFFER_MAPPED:
            static_assert(GL_BUFFER_MAPPED == GL_BUFFER_MAPPED_OES, "GL enums should be equal.");
            if (context->getClientMajorVersion() < 3 && !extensions.mapBuffer &&
                !extensions.mapBufferRange)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;

        case GL_BUFFER_MAP_POINTER:
            if (!pointerVersion)
            {
                context->validationError(GL_INVALID_ENUM, kInvalidMapPointerQuery);
                return false;
            }
            break;

        case GL_BUFFER_ACCESS_FLAGS:
        case GL_BUFFER_MAP_OFFSET:
        case GL_BUFFER_MAP_LENGTH:
            if (context->getClientMajorVersion() < 3 && !extensions.mapBufferRange)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;

        case GL_MEMORY_SIZE_ANGLE:
            if (!context->getExtensions().memorySize)
            {
                context->validationError(GL_INVALID_ENUM, kExtensionNotEnabled);
                return false;
            }
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
    }

    // All buffer parameter queries return one value.
    if (numParams)
    {
        *numParams = 1;
    }

    return true;
}

bool ValidateGetRenderbufferParameterivBase(Context *context,
                                            GLenum target,
                                            GLenum pname,
                                            GLsizei *length)
{
    if (length)
    {
        *length = 0;
    }

    if (target != GL_RENDERBUFFER)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidRenderbufferTarget);
        return false;
    }

    Renderbuffer *renderbuffer = context->getState().getCurrentRenderbuffer();
    if (renderbuffer == nullptr)
    {
        context->validationError(GL_INVALID_OPERATION, kRenderbufferNotBound);
        return false;
    }

    switch (pname)
    {
        case GL_RENDERBUFFER_WIDTH:
        case GL_RENDERBUFFER_HEIGHT:
        case GL_RENDERBUFFER_INTERNAL_FORMAT:
        case GL_RENDERBUFFER_RED_SIZE:
        case GL_RENDERBUFFER_GREEN_SIZE:
        case GL_RENDERBUFFER_BLUE_SIZE:
        case GL_RENDERBUFFER_ALPHA_SIZE:
        case GL_RENDERBUFFER_DEPTH_SIZE:
        case GL_RENDERBUFFER_STENCIL_SIZE:
            break;

        case GL_RENDERBUFFER_SAMPLES_ANGLE:
            if (!context->getExtensions().framebufferMultisample)
            {
                context->validationError(GL_INVALID_ENUM, kExtensionNotEnabled);
                return false;
            }
            break;

        case GL_MEMORY_SIZE_ANGLE:
            if (!context->getExtensions().memorySize)
            {
                context->validationError(GL_INVALID_ENUM, kExtensionNotEnabled);
                return false;
            }
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
    }

    if (length)
    {
        *length = 1;
    }
    return true;
}

bool ValidateGetShaderivBase(Context *context,
                             ShaderProgramID shader,
                             GLenum pname,
                             GLsizei *length)
{
    if (length)
    {
        *length = 0;
    }

    if (context->isContextLost())
    {
        context->validationError(GL_CONTEXT_LOST, kContextLost);

        if (context->getExtensions().parallelShaderCompile && pname == GL_COMPLETION_STATUS_KHR)
        {
            // Generate an error but still return true, the context still needs to return a
            // value in this case.
            return true;
        }
        else
        {
            return false;
        }
    }

    if (GetValidShader(context, shader) == nullptr)
    {
        return false;
    }

    switch (pname)
    {
        case GL_SHADER_TYPE:
        case GL_DELETE_STATUS:
        case GL_COMPILE_STATUS:
        case GL_INFO_LOG_LENGTH:
        case GL_SHADER_SOURCE_LENGTH:
            break;

        case GL_TRANSLATED_SHADER_SOURCE_LENGTH_ANGLE:
            if (!context->getExtensions().translatedShaderSource)
            {
                context->validationError(GL_INVALID_ENUM, kExtensionNotEnabled);
                return false;
            }
            break;

        case GL_COMPLETION_STATUS_KHR:
            if (!context->getExtensions().parallelShaderCompile)
            {
                context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
                return false;
            }
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
    }

    if (length)
    {
        *length = 1;
    }
    return true;
}

bool ValidateGetTexParameterBase(Context *context,
                                 TextureType target,
                                 GLenum pname,
                                 GLsizei *length)
{
    if (length)
    {
        *length = 0;
    }

    if (!ValidTextureTarget(context, target) && !ValidTextureExternalTarget(context, target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidTextureTarget);
        return false;
    }

    if (context->getTextureByType(target) == nullptr)
    {
        // Should only be possible for external textures
        context->validationError(GL_INVALID_ENUM, kTextureNotBound);
        return false;
    }

    if (context->getClientMajorVersion() == 1 && !IsValidGLES1TextureParameter(pname))
    {
        context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
        return false;
    }

    switch (pname)
    {
        case GL_TEXTURE_MAG_FILTER:
        case GL_TEXTURE_MIN_FILTER:
        case GL_TEXTURE_WRAP_S:
        case GL_TEXTURE_WRAP_T:
            break;

        case GL_TEXTURE_USAGE_ANGLE:
            if (!context->getExtensions().textureUsage)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;

        case GL_TEXTURE_MAX_ANISOTROPY_EXT:
            if (!ValidateTextureMaxAnisotropyExtensionEnabled(context))
            {
                return false;
            }
            break;

        case GL_TEXTURE_IMMUTABLE_FORMAT:
            if (context->getClientMajorVersion() < 3 && !context->getExtensions().textureStorage)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;

        case GL_TEXTURE_WRAP_R:
        case GL_TEXTURE_IMMUTABLE_LEVELS:
        case GL_TEXTURE_SWIZZLE_R:
        case GL_TEXTURE_SWIZZLE_G:
        case GL_TEXTURE_SWIZZLE_B:
        case GL_TEXTURE_SWIZZLE_A:
        case GL_TEXTURE_BASE_LEVEL:
        case GL_TEXTURE_MAX_LEVEL:
        case GL_TEXTURE_MIN_LOD:
        case GL_TEXTURE_MAX_LOD:
        case GL_TEXTURE_COMPARE_MODE:
        case GL_TEXTURE_COMPARE_FUNC:
            if (context->getClientMajorVersion() < 3)
            {
                context->validationError(GL_INVALID_ENUM, kEnumRequiresGLES30);
                return false;
            }
            break;

        case GL_TEXTURE_SRGB_DECODE_EXT:
            if (!context->getExtensions().textureSRGBDecode)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;

        case GL_DEPTH_STENCIL_TEXTURE_MODE:
            if (context->getClientVersion() < Version(3, 1))
            {
                context->validationError(GL_INVALID_ENUM, kEnumRequiresGLES31);
                return false;
            }
            break;

        case GL_GENERATE_MIPMAP:
        case GL_TEXTURE_CROP_RECT_OES:
            // TODO(lfy@google.com): Restrict to GL_OES_draw_texture
            // after GL_OES_draw_texture functionality implemented
            if (context->getClientMajorVersion() > 1)
            {
                context->validationError(GL_INVALID_ENUM, kGLES1Only);
                return false;
            }
            break;

        case GL_MEMORY_SIZE_ANGLE:
            if (!context->getExtensions().memorySize)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;

        case GL_TEXTURE_BORDER_COLOR:
            if (!context->getExtensions().textureBorderClamp)
            {
                context->validationError(GL_INVALID_ENUM, kExtensionNotEnabled);
                return false;
            }
            break;

        case GL_TEXTURE_NATIVE_ID_ANGLE:
            if (!context->getExtensions().textureExternalUpdateANGLE)
            {
                context->validationError(GL_INVALID_ENUM, kExtensionNotEnabled);
                return false;
            }
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
    }

    if (length)
    {
        *length = GetTexParameterCount(pname);
    }
    return true;
}

bool ValidateGetVertexAttribBase(Context *context,
                                 GLuint index,
                                 GLenum pname,
                                 GLsizei *length,
                                 bool pointer,
                                 bool pureIntegerEntryPoint)
{
    if (length)
    {
        *length = 0;
    }

    if (pureIntegerEntryPoint && context->getClientMajorVersion() < 3)
    {
        context->validationError(GL_INVALID_OPERATION, kES3Required);
        return false;
    }

    if (index >= context->getCaps().maxVertexAttributes)
    {
        context->validationError(GL_INVALID_VALUE, kIndexExceedsMaxVertexAttribute);
        return false;
    }

    if (pointer)
    {
        if (pname != GL_VERTEX_ATTRIB_ARRAY_POINTER)
        {
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
        }
    }
    else
    {
        switch (pname)
        {
            case GL_VERTEX_ATTRIB_ARRAY_ENABLED:
            case GL_VERTEX_ATTRIB_ARRAY_SIZE:
            case GL_VERTEX_ATTRIB_ARRAY_STRIDE:
            case GL_VERTEX_ATTRIB_ARRAY_TYPE:
            case GL_VERTEX_ATTRIB_ARRAY_NORMALIZED:
            case GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING:
            case GL_CURRENT_VERTEX_ATTRIB:
                break;

            case GL_VERTEX_ATTRIB_ARRAY_DIVISOR:
                static_assert(
                    GL_VERTEX_ATTRIB_ARRAY_DIVISOR == GL_VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE,
                    "ANGLE extension enums not equal to GL enums.");
                if (context->getClientMajorVersion() < 3 &&
                    !context->getExtensions().instancedArraysAny())
                {
                    context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                    return false;
                }
                break;

            case GL_VERTEX_ATTRIB_ARRAY_INTEGER:
                if (context->getClientMajorVersion() < 3)
                {
                    context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                    return false;
                }
                break;

            case GL_VERTEX_ATTRIB_BINDING:
            case GL_VERTEX_ATTRIB_RELATIVE_OFFSET:
                if (context->getClientVersion() < ES_3_1)
                {
                    context->validationError(GL_INVALID_ENUM, kEnumRequiresGLES31);
                    return false;
                }
                break;

            default:
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
        }
    }

    if (length)
    {
        if (pname == GL_CURRENT_VERTEX_ATTRIB)
        {
            *length = 4;
        }
        else
        {
            *length = 1;
        }
    }

    return true;
}

bool ValidateReadPixelsBase(Context *context,
                            GLint x,
                            GLint y,
                            GLsizei width,
                            GLsizei height,
                            GLenum format,
                            GLenum type,
                            GLsizei bufSize,
                            GLsizei *length,
                            GLsizei *columns,
                            GLsizei *rows,
                            void *pixels)
{
    if (length != nullptr)
    {
        *length = 0;
    }
    if (rows != nullptr)
    {
        *rows = 0;
    }
    if (columns != nullptr)
    {
        *columns = 0;
    }

    if (width < 0 || height < 0)
    {
        context->validationError(GL_INVALID_VALUE, kNegativeSize);
        return false;
    }

    Framebuffer *readFramebuffer = context->getState().getReadFramebuffer();
    ASSERT(readFramebuffer);

    if (!ValidateFramebufferComplete(context, readFramebuffer))
    {
        return false;
    }

    // needIntrinsic = true. Treat renderToTexture textures as single sample since they will be
    // resolved before reading.
    if (!readFramebuffer->isDefault() &&
        !ValidateFramebufferNotMultisampled(context, readFramebuffer, true))
    {
        return false;
    }

    if (readFramebuffer->getReadBufferState() == GL_NONE)
    {
        context->validationError(GL_INVALID_OPERATION, kReadBufferNone);
        return false;
    }

    const FramebufferAttachment *readBuffer = readFramebuffer->getReadColorAttachment();
    // WebGL 1.0 [Section 6.26] Reading From a Missing Attachment
    // In OpenGL ES it is undefined what happens when an operation tries to read from a missing
    // attachment and WebGL defines it to be an error. We do the check unconditionnaly as the
    // situation is an application error that would lead to a crash in ANGLE.
    if (readBuffer == nullptr)
    {
        context->validationError(GL_INVALID_OPERATION, kMissingReadAttachment);
        return false;
    }

    // OVR_multiview2, Revision 1:
    // ReadPixels generates an INVALID_FRAMEBUFFER_OPERATION error if
    // the number of views in the current read framebuffer is more than one.
    if (readFramebuffer->readDisallowedByMultiview())
    {
        context->validationError(GL_INVALID_FRAMEBUFFER_OPERATION, kMultiviewReadFramebuffer);
        return false;
    }

    if (context->getExtensions().webglCompatibility)
    {
        // The ES 2.0 spec states that the format must be "among those defined in table 3.4,
        // excluding formats LUMINANCE and LUMINANCE_ALPHA.".  This requires validating the format
        // and type before validating the combination of format and type.  However, the
        // dEQP-GLES3.functional.negative_api.buffer.read_pixels passes GL_LUMINANCE as a format and
        // verifies that GL_INVALID_OPERATION is generated.
        // TODO(geofflang): Update this check to be done in all/no cases once this is resolved in
        // dEQP/WebGL.
        if (!ValidReadPixelsFormatEnum(context, format))
        {
            context->validationError(GL_INVALID_ENUM, kInvalidFormat);
            return false;
        }

        if (!ValidReadPixelsTypeEnum(context, type))
        {
            context->validationError(GL_INVALID_ENUM, kInvalidType);
            return false;
        }
    }

    GLenum currentFormat = GL_NONE;
    ANGLE_VALIDATION_TRY(
        readFramebuffer->getImplementationColorReadFormat(context, &currentFormat));

    GLenum currentType = GL_NONE;
    ANGLE_VALIDATION_TRY(readFramebuffer->getImplementationColorReadType(context, &currentType));

    GLenum currentComponentType = readBuffer->getFormat().info->componentType;

    bool validFormatTypeCombination =
        ValidReadPixelsFormatType(context, currentComponentType, format, type);

    if (!(currentFormat == format && currentType == type) && !validFormatTypeCombination)
    {
        context->validationError(GL_INVALID_OPERATION, kMismatchedTypeAndFormat);
        return false;
    }

    // Check for pixel pack buffer related API errors
    Buffer *pixelPackBuffer = context->getState().getTargetBuffer(BufferBinding::PixelPack);
    if (pixelPackBuffer != nullptr && pixelPackBuffer->isMapped())
    {
        // ...the buffer object's data store is currently mapped.
        context->validationError(GL_INVALID_OPERATION, kBufferMapped);
        return false;
    }
    if (context->getExtensions().webglCompatibility && pixelPackBuffer != nullptr &&
        pixelPackBuffer->isBoundForTransformFeedbackAndOtherUse())
    {
        context->validationError(GL_INVALID_OPERATION, kPixelPackBufferBoundForTransformFeedback);
        return false;
    }

    // ..  the data would be packed to the buffer object such that the memory writes required
    // would exceed the data store size.
    const InternalFormat &formatInfo = GetInternalFormatInfo(format, type);
    const Extents size(width, height, 1);
    const auto &pack = context->getState().getPackState();

    GLuint endByte = 0;
    if (!formatInfo.computePackUnpackEndByte(type, size, pack, false, &endByte))
    {
        context->validationError(GL_INVALID_OPERATION, kIntegerOverflow);
        return false;
    }

    if (bufSize >= 0)
    {
        if (pixelPackBuffer == nullptr && static_cast<size_t>(bufSize) < endByte)
        {
            context->validationError(GL_INVALID_OPERATION, kInsufficientBufferSize);
            return false;
        }
    }

    if (pixelPackBuffer != nullptr)
    {
        CheckedNumeric<size_t> checkedEndByte(endByte);
        CheckedNumeric<size_t> checkedOffset(reinterpret_cast<size_t>(pixels));
        checkedEndByte += checkedOffset;

        if (checkedEndByte.ValueOrDie() > static_cast<size_t>(pixelPackBuffer->getSize()))
        {
            // Overflow past the end of the buffer
            context->validationError(GL_INVALID_OPERATION, kParamOverflow);
            return false;
        }
    }

    if (pixelPackBuffer == nullptr && length != nullptr)
    {
        if (endByte > static_cast<size_t>(std::numeric_limits<GLsizei>::max()))
        {
            context->validationError(GL_INVALID_OPERATION, kIntegerOverflow);
            return false;
        }

        *length = static_cast<GLsizei>(endByte);
    }

    auto getClippedExtent = [](GLint start, GLsizei length, int bufferSize, GLsizei *outExtent) {
        angle::CheckedNumeric<int> clippedExtent(length);
        if (start < 0)
        {
            // "subtract" the area that is less than 0
            clippedExtent += start;
        }

        angle::CheckedNumeric<int> readExtent = start;
        readExtent += length;
        if (!readExtent.IsValid())
        {
            return false;
        }

        if (readExtent.ValueOrDie() > bufferSize)
        {
            // Subtract the region to the right of the read buffer
            clippedExtent -= (readExtent - bufferSize);
        }

        if (!clippedExtent.IsValid())
        {
            return false;
        }

        *outExtent = std::max(clippedExtent.ValueOrDie(), 0);
        return true;
    };

    GLsizei writtenColumns = 0;
    if (!getClippedExtent(x, width, readBuffer->getSize().width, &writtenColumns))
    {
        context->validationError(GL_INVALID_OPERATION, kIntegerOverflow);
        return false;
    }

    GLsizei writtenRows = 0;
    if (!getClippedExtent(y, height, readBuffer->getSize().height, &writtenRows))
    {
        context->validationError(GL_INVALID_OPERATION, kIntegerOverflow);
        return false;
    }

    if (columns != nullptr)
    {
        *columns = writtenColumns;
    }

    if (rows != nullptr)
    {
        *rows = writtenRows;
    }

    return true;
}

template <typename ParamType>
bool ValidateTexParameterBase(Context *context,
                              TextureType target,
                              GLenum pname,
                              GLsizei bufSize,
                              bool vectorParams,
                              const ParamType *params)
{
    if (!ValidTextureTarget(context, target) && !ValidTextureExternalTarget(context, target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidTextureTarget);
        return false;
    }

    if (context->getTextureByType(target) == nullptr)
    {
        // Should only be possible for external textures
        context->validationError(GL_INVALID_ENUM, kTextureNotBound);
        return false;
    }

    const GLsizei minBufSize = GetTexParameterCount(pname);
    if (bufSize >= 0 && bufSize < minBufSize)
    {
        context->validationError(GL_INVALID_OPERATION, kInsufficientBufferSize);
        return false;
    }

    if (context->getClientMajorVersion() == 1 && !IsValidGLES1TextureParameter(pname))
    {
        context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
        return false;
    }

    switch (pname)
    {
        case GL_TEXTURE_WRAP_R:
        case GL_TEXTURE_SWIZZLE_R:
        case GL_TEXTURE_SWIZZLE_G:
        case GL_TEXTURE_SWIZZLE_B:
        case GL_TEXTURE_SWIZZLE_A:
        case GL_TEXTURE_BASE_LEVEL:
        case GL_TEXTURE_MAX_LEVEL:
        case GL_TEXTURE_COMPARE_MODE:
        case GL_TEXTURE_COMPARE_FUNC:
        case GL_TEXTURE_MIN_LOD:
        case GL_TEXTURE_MAX_LOD:
            if (context->getClientMajorVersion() < 3 &&
                !(pname == GL_TEXTURE_WRAP_R && context->getExtensions().texture3DOES))
            {
                context->validationError(GL_INVALID_ENUM, kES3Required);
                return false;
            }
            if (target == TextureType::External && !context->getExtensions().eglImageExternalEssl3)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;

        case GL_GENERATE_MIPMAP:
        case GL_TEXTURE_CROP_RECT_OES:
            if (context->getClientMajorVersion() > 1)
            {
                context->validationError(GL_INVALID_ENUM, kGLES1Only);
                return false;
            }
            break;
        default:
            break;
    }

    if (target == TextureType::_2DMultisample || target == TextureType::_2DMultisampleArray)
    {
        switch (pname)
        {
            case GL_TEXTURE_MIN_FILTER:
            case GL_TEXTURE_MAG_FILTER:
            case GL_TEXTURE_WRAP_S:
            case GL_TEXTURE_WRAP_T:
            case GL_TEXTURE_WRAP_R:
            case GL_TEXTURE_MIN_LOD:
            case GL_TEXTURE_MAX_LOD:
            case GL_TEXTURE_COMPARE_MODE:
            case GL_TEXTURE_COMPARE_FUNC:
            case GL_TEXTURE_BORDER_COLOR:
                context->validationError(GL_INVALID_ENUM, kInvalidPname);
                return false;
        }
    }

    switch (pname)
    {
        case GL_TEXTURE_WRAP_S:
        case GL_TEXTURE_WRAP_T:
        case GL_TEXTURE_WRAP_R:
        {
            bool restrictedWrapModes =
                target == TextureType::External || target == TextureType::Rectangle;
            if (!ValidateTextureWrapModeValue(context, params, restrictedWrapModes))
            {
                return false;
            }
        }
        break;

        case GL_TEXTURE_MIN_FILTER:
        {
            bool restrictedMinFilter =
                target == TextureType::External || target == TextureType::Rectangle;
            if (!ValidateTextureMinFilterValue(context, params, restrictedMinFilter))
            {
                return false;
            }
        }
        break;

        case GL_TEXTURE_MAG_FILTER:
            if (!ValidateTextureMagFilterValue(context, params))
            {
                return false;
            }
            break;

        case GL_TEXTURE_USAGE_ANGLE:
            if (!context->getExtensions().textureUsage)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }

            switch (ConvertToGLenum(params[0]))
            {
                case GL_NONE:
                case GL_FRAMEBUFFER_ATTACHMENT_ANGLE:
                    break;

                default:
                    context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                    return false;
            }
            break;

        case GL_TEXTURE_MAX_ANISOTROPY_EXT:
        {
            GLfloat paramValue = static_cast<GLfloat>(params[0]);
            if (!ValidateTextureMaxAnisotropyValue(context, paramValue))
            {
                return false;
            }
            ASSERT(static_cast<ParamType>(paramValue) == params[0]);
        }
        break;

        case GL_TEXTURE_MIN_LOD:
        case GL_TEXTURE_MAX_LOD:
            // any value is permissible
            break;

        case GL_TEXTURE_COMPARE_MODE:
            if (!ValidateTextureCompareModeValue(context, params))
            {
                return false;
            }
            break;

        case GL_TEXTURE_COMPARE_FUNC:
            if (!ValidateTextureCompareFuncValue(context, params))
            {
                return false;
            }
            break;

        case GL_TEXTURE_SWIZZLE_R:
        case GL_TEXTURE_SWIZZLE_G:
        case GL_TEXTURE_SWIZZLE_B:
        case GL_TEXTURE_SWIZZLE_A:
            switch (ConvertToGLenum(params[0]))
            {
                case GL_RED:
                case GL_GREEN:
                case GL_BLUE:
                case GL_ALPHA:
                case GL_ZERO:
                case GL_ONE:
                    break;

                default:
                    context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                    return false;
            }
            break;

        case GL_TEXTURE_BASE_LEVEL:
            if (ConvertToGLint(params[0]) < 0)
            {
                context->validationError(GL_INVALID_VALUE, kBaseLevelNegative);
                return false;
            }
            if (target == TextureType::External && static_cast<GLuint>(params[0]) != 0)
            {
                context->validationError(GL_INVALID_OPERATION, kBaseLevelNonZero);
                return false;
            }
            if ((target == TextureType::_2DMultisample ||
                 target == TextureType::_2DMultisampleArray) &&
                static_cast<GLuint>(params[0]) != 0)
            {
                context->validationError(GL_INVALID_OPERATION, kBaseLevelNonZero);
                return false;
            }
            if (target == TextureType::Rectangle && static_cast<GLuint>(params[0]) != 0)
            {
                context->validationError(GL_INVALID_OPERATION, kBaseLevelNonZero);
                return false;
            }
            break;

        case GL_TEXTURE_MAX_LEVEL:
            if (ConvertToGLint(params[0]) < 0)
            {
                context->validationError(GL_INVALID_VALUE, kInvalidMipLevel);
                return false;
            }
            break;

        case GL_DEPTH_STENCIL_TEXTURE_MODE:
            if (context->getClientVersion() < Version(3, 1))
            {
                context->validationError(GL_INVALID_ENUM, kEnumRequiresGLES31);
                return false;
            }
            switch (ConvertToGLenum(params[0]))
            {
                case GL_DEPTH_COMPONENT:
                case GL_STENCIL_INDEX:
                    break;

                default:
                    context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                    return false;
            }
            break;

        case GL_TEXTURE_SRGB_DECODE_EXT:
            if (!ValidateTextureSRGBDecodeValue(context, params))
            {
                return false;
            }
            break;

        case GL_GENERATE_MIPMAP:
            if (context->getClientMajorVersion() > 1)
            {
                context->validationError(GL_INVALID_ENUM, kGLES1Only);
                return false;
            }
            break;

        case GL_TEXTURE_CROP_RECT_OES:
            if (context->getClientMajorVersion() > 1)
            {
                context->validationError(GL_INVALID_ENUM, kGLES1Only);
                return false;
            }
            if (!vectorParams)
            {
                context->validationError(GL_INVALID_OPERATION, kInsufficientBufferSize);
                return false;
            }
            break;

        case GL_TEXTURE_BORDER_COLOR:
            if (!context->getExtensions().textureBorderClamp)
            {
                context->validationError(GL_INVALID_ENUM, kExtensionNotEnabled);
                return false;
            }
            if (!vectorParams)
            {
                context->validationError(GL_INVALID_ENUM, kInsufficientBufferSize);
                return false;
            }
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
    }

    return true;
}

template bool ValidateTexParameterBase(Context *,
                                       TextureType,
                                       GLenum,
                                       GLsizei,
                                       bool,
                                       const GLfloat *);
template bool ValidateTexParameterBase(Context *,
                                       TextureType,
                                       GLenum,
                                       GLsizei,
                                       bool,
                                       const GLint *);
template bool ValidateTexParameterBase(Context *,
                                       TextureType,
                                       GLenum,
                                       GLsizei,
                                       bool,
                                       const GLuint *);

bool ValidateVertexAttribIndex(Context *context, GLuint index)
{
    if (index >= MAX_VERTEX_ATTRIBS)
    {
        context->validationError(GL_INVALID_VALUE, kIndexExceedsMaxVertexAttribute);
        return false;
    }

    return true;
}

bool ValidateGetActiveUniformBlockivBase(Context *context,
                                         ShaderProgramID program,
                                         GLuint uniformBlockIndex,
                                         GLenum pname,
                                         GLsizei *length)
{
    if (length)
    {
        *length = 0;
    }

    if (context->getClientMajorVersion() < 3)
    {
        context->validationError(GL_INVALID_OPERATION, kES3Required);
        return false;
    }

    Program *programObject = GetValidProgram(context, program);
    if (!programObject)
    {
        return false;
    }

    if (uniformBlockIndex >= programObject->getActiveUniformBlockCount())
    {
        context->validationError(GL_INVALID_VALUE, kIndexExceedsActiveUniformBlockCount);
        return false;
    }

    switch (pname)
    {
        case GL_UNIFORM_BLOCK_BINDING:
        case GL_UNIFORM_BLOCK_DATA_SIZE:
        case GL_UNIFORM_BLOCK_NAME_LENGTH:
        case GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS:
        case GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES:
        case GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER:
        case GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER:
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
    }

    if (length)
    {
        if (pname == GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES)
        {
            const InterfaceBlock &uniformBlock =
                programObject->getUniformBlockByIndex(uniformBlockIndex);
            *length = static_cast<GLsizei>(uniformBlock.memberIndexes.size());
        }
        else
        {
            *length = 1;
        }
    }

    return true;
}

template <typename ParamType>
bool ValidateSamplerParameterBase(Context *context,
                                  SamplerID sampler,
                                  GLenum pname,
                                  GLsizei bufSize,
                                  bool vectorParams,
                                  const ParamType *params)
{
    if (context->getClientMajorVersion() < 3)
    {
        context->validationError(GL_INVALID_OPERATION, kES3Required);
        return false;
    }

    if (!context->isSampler(sampler))
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidSampler);
        return false;
    }

    const GLsizei minBufSize = GetSamplerParameterCount(pname);
    if (bufSize >= 0 && bufSize < minBufSize)
    {
        context->validationError(GL_INVALID_OPERATION, kInsufficientBufferSize);
        return false;
    }

    switch (pname)
    {
        case GL_TEXTURE_WRAP_S:
        case GL_TEXTURE_WRAP_T:
        case GL_TEXTURE_WRAP_R:
            if (!ValidateTextureWrapModeValue(context, params, false))
            {
                return false;
            }
            break;

        case GL_TEXTURE_MIN_FILTER:
            if (!ValidateTextureMinFilterValue(context, params, false))
            {
                return false;
            }
            break;

        case GL_TEXTURE_MAG_FILTER:
            if (!ValidateTextureMagFilterValue(context, params))
            {
                return false;
            }
            break;

        case GL_TEXTURE_MIN_LOD:
        case GL_TEXTURE_MAX_LOD:
            // any value is permissible
            break;

        case GL_TEXTURE_COMPARE_MODE:
            if (!ValidateTextureCompareModeValue(context, params))
            {
                return false;
            }
            break;

        case GL_TEXTURE_COMPARE_FUNC:
            if (!ValidateTextureCompareFuncValue(context, params))
            {
                return false;
            }
            break;

        case GL_TEXTURE_SRGB_DECODE_EXT:
            if (!ValidateTextureSRGBDecodeValue(context, params))
            {
                return false;
            }
            break;

        case GL_TEXTURE_MAX_ANISOTROPY_EXT:
        {
            GLfloat paramValue = static_cast<GLfloat>(params[0]);
            if (!ValidateTextureMaxAnisotropyValue(context, paramValue))
            {
                return false;
            }
        }
        break;

        case GL_TEXTURE_BORDER_COLOR:
            if (!context->getExtensions().textureBorderClamp)
            {
                context->validationError(GL_INVALID_ENUM, kExtensionNotEnabled);
                return false;
            }
            if (!vectorParams)
            {
                context->validationError(GL_INVALID_ENUM, kInsufficientBufferSize);
                return false;
            }
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
    }

    return true;
}

template bool ValidateSamplerParameterBase(Context *,
                                           SamplerID,
                                           GLenum,
                                           GLsizei,
                                           bool,
                                           const GLfloat *);
template bool ValidateSamplerParameterBase(Context *,
                                           SamplerID,
                                           GLenum,
                                           GLsizei,
                                           bool,
                                           const GLint *);
template bool ValidateSamplerParameterBase(Context *,
                                           SamplerID,
                                           GLenum,
                                           GLsizei,
                                           bool,
                                           const GLuint *);

bool ValidateGetSamplerParameterBase(Context *context,
                                     SamplerID sampler,
                                     GLenum pname,
                                     GLsizei *length)
{
    if (length)
    {
        *length = 0;
    }

    if (context->getClientMajorVersion() < 3)
    {
        context->validationError(GL_INVALID_OPERATION, kES3Required);
        return false;
    }

    if (!context->isSampler(sampler))
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidSampler);
        return false;
    }

    switch (pname)
    {
        case GL_TEXTURE_WRAP_S:
        case GL_TEXTURE_WRAP_T:
        case GL_TEXTURE_WRAP_R:
        case GL_TEXTURE_MIN_FILTER:
        case GL_TEXTURE_MAG_FILTER:
        case GL_TEXTURE_MIN_LOD:
        case GL_TEXTURE_MAX_LOD:
        case GL_TEXTURE_COMPARE_MODE:
        case GL_TEXTURE_COMPARE_FUNC:
            break;

        case GL_TEXTURE_MAX_ANISOTROPY_EXT:
            if (!ValidateTextureMaxAnisotropyExtensionEnabled(context))
            {
                return false;
            }
            break;

        case GL_TEXTURE_SRGB_DECODE_EXT:
            if (!context->getExtensions().textureSRGBDecode)
            {
                context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
                return false;
            }
            break;

        case GL_TEXTURE_BORDER_COLOR:
            if (!context->getExtensions().textureBorderClamp)
            {
                context->validationError(GL_INVALID_ENUM, kExtensionNotEnabled);
                return false;
            }
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
    }

    if (length)
    {
        *length = GetSamplerParameterCount(pname);
    }
    return true;
}

bool ValidateGetInternalFormativBase(Context *context,
                                     GLenum target,
                                     GLenum internalformat,
                                     GLenum pname,
                                     GLsizei bufSize,
                                     GLsizei *numParams)
{
    if (numParams)
    {
        *numParams = 0;
    }

    if (context->getClientMajorVersion() < 3)
    {
        context->validationError(GL_INVALID_OPERATION, kES3Required);
        return false;
    }

    const TextureCaps &formatCaps = context->getTextureCaps().get(internalformat);
    if (!formatCaps.renderbuffer)
    {
        context->validationError(GL_INVALID_ENUM, kFormatNotRenderable);
        return false;
    }

    switch (target)
    {
        case GL_RENDERBUFFER:
            break;

        case GL_TEXTURE_2D_MULTISAMPLE:
            if (context->getClientVersion() < ES_3_1 &&
                !context->getExtensions().textureMultisample)
            {
                context->validationError(GL_INVALID_ENUM,
                                         kMultisampleTextureExtensionOrES31Required);
                return false;
            }
            break;
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY_OES:
            if (!context->getExtensions().textureStorageMultisample2DArray)
            {
                context->validationError(GL_INVALID_ENUM, kMultisampleArrayExtensionRequired);
                return false;
            }
            break;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidTarget);
            return false;
    }

    if (bufSize < 0)
    {
        context->validationError(GL_INVALID_VALUE, kInsufficientBufferSize);
        return false;
    }

    GLsizei maxWriteParams = 0;
    switch (pname)
    {
        case GL_NUM_SAMPLE_COUNTS:
            maxWriteParams = 1;
            break;

        case GL_SAMPLES:
            maxWriteParams = static_cast<GLsizei>(formatCaps.sampleCounts.size());
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
    }

    if (numParams)
    {
        // glGetInternalFormativ will not overflow bufSize
        *numParams = std::min(bufSize, maxWriteParams);
    }

    return true;
}

bool ValidateFramebufferNotMultisampled(Context *context,
                                        Framebuffer *framebuffer,
                                        bool needResourceSamples)
{
    int samples = needResourceSamples ? framebuffer->getResourceSamples(context)
                                      : framebuffer->getSamples(context);
    if (samples != 0)
    {
        context->validationError(GL_INVALID_OPERATION, kInvalidMultisampledFramebufferOperation);
        return false;
    }
    return true;
}

bool ValidateMultitextureUnit(Context *context, GLenum texture)
{
    if (texture < GL_TEXTURE0 || texture >= GL_TEXTURE0 + context->getCaps().maxMultitextureUnits)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidMultitextureUnit);
        return false;
    }
    return true;
}

bool ValidateTexStorageMultisample(Context *context,
                                   TextureType target,
                                   GLsizei samples,
                                   GLint internalFormat,
                                   GLsizei width,
                                   GLsizei height)
{
    const Caps &caps = context->getCaps();
    if (static_cast<GLuint>(width) > caps.max2DTextureSize ||
        static_cast<GLuint>(height) > caps.max2DTextureSize)
    {
        context->validationError(GL_INVALID_VALUE, kTextureWidthOrHeightOutOfRange);
        return false;
    }

    if (samples == 0)
    {
        context->validationError(GL_INVALID_VALUE, kSamplesZero);
        return false;
    }

    const TextureCaps &formatCaps = context->getTextureCaps().get(internalFormat);
    if (!formatCaps.textureAttachment)
    {
        context->validationError(GL_INVALID_ENUM, kRenderableInternalFormat);
        return false;
    }

    // The ES3.1 spec(section 8.8) states that an INVALID_ENUM error is generated if internalformat
    // is one of the unsized base internalformats listed in table 8.11.
    const InternalFormat &formatInfo = GetSizedInternalFormatInfo(internalFormat);
    if (formatInfo.internalFormat == GL_NONE)
    {
        context->validationError(GL_INVALID_ENUM, kUnsizedInternalFormatUnsupported);
        return false;
    }

    if (static_cast<GLuint>(samples) > formatCaps.getMaxSamples())
    {
        context->validationError(GL_INVALID_OPERATION, kSamplesOutOfRange);
        return false;
    }

    Texture *texture = context->getTextureByType(target);
    if (!texture || texture->id().value == 0)
    {
        context->validationError(GL_INVALID_OPERATION, kZeroBoundToTarget);
        return false;
    }

    if (texture->getImmutableFormat())
    {
        context->validationError(GL_INVALID_OPERATION, kImmutableTextureBound);
        return false;
    }
    return true;
}

bool ValidateTexStorage2DMultisampleBase(Context *context,
                                         TextureType target,
                                         GLsizei samples,
                                         GLint internalFormat,
                                         GLsizei width,
                                         GLsizei height)
{
    if (target != TextureType::_2DMultisample)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidTarget);
        return false;
    }

    if (width < 1 || height < 1)
    {
        context->validationError(GL_INVALID_VALUE, kTextureSizeTooSmall);
        return false;
    }

    return ValidateTexStorageMultisample(context, target, samples, internalFormat, width, height);
}

bool ValidateGetTexLevelParameterBase(Context *context,
                                      TextureTarget target,
                                      GLint level,
                                      GLenum pname,
                                      GLsizei *length)
{

    if (length)
    {
        *length = 0;
    }

    TextureType type = TextureTargetToType(target);

    if (!ValidTexLevelDestinationTarget(context, type))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidTextureTarget);
        return false;
    }

    if (context->getTextureByType(type) == nullptr)
    {
        context->validationError(GL_INVALID_ENUM, kTextureNotBound);
        return false;
    }

    if (!ValidMipLevel(context, type, level))
    {
        context->validationError(GL_INVALID_VALUE, kInvalidMipLevel);
        return false;
    }

    switch (pname)
    {
        case GL_TEXTURE_RED_TYPE:
        case GL_TEXTURE_GREEN_TYPE:
        case GL_TEXTURE_BLUE_TYPE:
        case GL_TEXTURE_ALPHA_TYPE:
        case GL_TEXTURE_DEPTH_TYPE:
            break;
        case GL_TEXTURE_RED_SIZE:
        case GL_TEXTURE_GREEN_SIZE:
        case GL_TEXTURE_BLUE_SIZE:
        case GL_TEXTURE_ALPHA_SIZE:
        case GL_TEXTURE_DEPTH_SIZE:
        case GL_TEXTURE_STENCIL_SIZE:
        case GL_TEXTURE_SHARED_SIZE:
            break;
        case GL_TEXTURE_INTERNAL_FORMAT:
        case GL_TEXTURE_WIDTH:
        case GL_TEXTURE_HEIGHT:
        case GL_TEXTURE_DEPTH:
            break;
        case GL_TEXTURE_SAMPLES:
        case GL_TEXTURE_FIXED_SAMPLE_LOCATIONS:
            break;
        case GL_TEXTURE_COMPRESSED:
            break;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidPname);
            return false;
    }

    if (length)
    {
        *length = 1;
    }
    return true;
}

bool ValidateGetMultisamplefvBase(Context *context, GLenum pname, GLuint index, GLfloat *val)
{
    if (pname != GL_SAMPLE_POSITION)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidPname);
        return false;
    }

    Framebuffer *framebuffer = context->getState().getDrawFramebuffer();
    GLint samples            = framebuffer->getSamples(context);

    if (index >= static_cast<GLuint>(samples))
    {
        context->validationError(GL_INVALID_VALUE, kIndexExceedsSamples);
        return false;
    }

    return true;
}

bool ValidateSampleMaskiBase(Context *context, GLuint maskNumber, GLbitfield mask)
{
    if (maskNumber >= context->getCaps().maxSampleMaskWords)
    {
        context->validationError(GL_INVALID_VALUE, kInvalidSampleMaskNumber);
        return false;
    }

    return true;
}

void RecordDrawAttribsError(Context *context)
{
    // An overflow can happen when adding the offset. Check against a special constant.
    if (context->getStateCache().getNonInstancedVertexElementLimit() ==
            VertexAttribute::kIntegerOverflow ||
        context->getStateCache().getInstancedVertexElementLimit() ==
            VertexAttribute::kIntegerOverflow)
    {
        context->validationError(GL_INVALID_OPERATION, kIntegerOverflow);
    }
    else
    {
        // [OpenGL ES 3.0.2] section 2.9.4 page 40:
        // We can return INVALID_OPERATION if our buffer does not have enough backing data.
        context->validationError(GL_INVALID_OPERATION, kInsufficientVertexBufferSize);
    }
}

bool ValidateLoseContextCHROMIUM(Context *context,
                                 GraphicsResetStatus current,
                                 GraphicsResetStatus other)
{
    if (!context->getExtensions().loseContextCHROMIUM)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    switch (current)
    {
        case GraphicsResetStatus::GuiltyContextReset:
        case GraphicsResetStatus::InnocentContextReset:
        case GraphicsResetStatus::UnknownContextReset:
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kInvalidResetStatus);
    }

    switch (other)
    {
        case GraphicsResetStatus::GuiltyContextReset:
        case GraphicsResetStatus::InnocentContextReset:
        case GraphicsResetStatus::UnknownContextReset:
            break;

        default:
            context->validationError(GL_INVALID_ENUM, kInvalidResetStatus);
    }

    return true;
}

// GL_ANGLE_texture_storage_external
bool ValidateTexImage2DExternalANGLE(Context *context,
                                     TextureTarget target,
                                     GLint level,
                                     GLint internalformat,
                                     GLsizei width,
                                     GLsizei height,
                                     GLint border,
                                     GLenum format,
                                     GLenum type)
{
    if (!context->getExtensions().textureExternalUpdateANGLE)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    if (!ValidTexture2DDestinationTarget(context, target) &&
        !ValidTextureExternalTarget(context, target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidTextureTarget);
        return false;
    }

    if (context->getClientMajorVersion() <= 2)
    {
        if (!ValidateES2TexImageParametersBase(context, target, level, internalformat, false, false,
                                               0, 0, width, height, border, format, type, -1,
                                               nullptr))
        {
            return false;
        }
    }
    else
    {
        if (!ValidateES3TexImageParametersBase(context, target, level, internalformat, false, false,
                                               0, 0, 0, width, height, 1, border, format, type, -1,
                                               nullptr))
        {
            return false;
        }
    }

    return true;
}

bool ValidateInvalidateTextureANGLE(Context *context, TextureType target)
{
    if (!context->getExtensions().textureExternalUpdateANGLE)
    {
        context->validationError(GL_INVALID_OPERATION, kExtensionNotEnabled);
        return false;
    }

    if (!ValidTextureTarget(context, target) && !ValidTextureExternalTarget(context, target))
    {
        context->validationError(GL_INVALID_ENUM, kInvalidTextureTarget);
        return false;
    }

    return true;
}

}  // namespace gl
