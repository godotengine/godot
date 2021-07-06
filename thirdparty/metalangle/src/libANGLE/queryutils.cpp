//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// queryutils.cpp: Utilities for querying values from GL objects

#include "libANGLE/queryutils.h"

#include "common/utilities.h"

#include "libANGLE/Buffer.h"
#include "libANGLE/Config.h"
#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/EGLSync.h"
#include "libANGLE/Fence.h"
#include "libANGLE/Framebuffer.h"
#include "libANGLE/GLES1State.h"
#include "libANGLE/Program.h"
#include "libANGLE/Renderbuffer.h"
#include "libANGLE/Sampler.h"
#include "libANGLE/Shader.h"
#include "libANGLE/Surface.h"
#include "libANGLE/Texture.h"
#include "libANGLE/Uniform.h"
#include "libANGLE/VertexAttribute.h"
#include "libANGLE/queryconversions.h"

namespace gl
{

namespace
{

template <bool isPureInteger>
ColorGeneric ConvertToColor(const GLfloat *params)
{
    if (isPureInteger)
    {
        UNREACHABLE();
        return ColorGeneric(ColorI());
    }
    else
    {
        return ColorGeneric(ColorF::fromData(params));
    }
}

template <bool isPureInteger>
ColorGeneric ConvertToColor(const GLint *params)
{
    if (isPureInteger)
    {
        return ColorGeneric(ColorI(params[0], params[1], params[2], params[3]));
    }
    else
    {
        return ColorGeneric(ColorF(normalizedToFloat(params[0]), normalizedToFloat(params[1]),
                                   normalizedToFloat(params[2]), normalizedToFloat(params[3])));
    }
}

template <bool isPureInteger>
ColorGeneric ConvertToColor(const GLuint *params)
{
    if (isPureInteger)
    {
        return ColorGeneric(ColorUI(params[0], params[1], params[2], params[3]));
    }
    else
    {
        UNREACHABLE();
        return ColorGeneric(ColorF());
    }
}

template <bool isPureInteger>
void ConvertFromColor(const ColorGeneric &color, GLfloat *outParams)
{
    if (isPureInteger)
    {
        UNREACHABLE();
    }
    else
    {
        ASSERT(color.type == ColorGeneric::Type::Float);
        color.colorF.writeData(outParams);
    }
}

template <bool isPureInteger>
void ConvertFromColor(const ColorGeneric &color, GLint *outParams)
{
    if (isPureInteger)
    {
        ASSERT(color.type == ColorGeneric::Type::Int);
        outParams[0] = color.colorI.red;
        outParams[1] = color.colorI.green;
        outParams[2] = color.colorI.blue;
        outParams[3] = color.colorI.alpha;
    }
    else
    {
        ASSERT(color.type == ColorGeneric::Type::Float);
        outParams[0] = floatToNormalized<GLint>(color.colorF.red);
        outParams[1] = floatToNormalized<GLint>(color.colorF.green);
        outParams[2] = floatToNormalized<GLint>(color.colorF.blue);
        outParams[3] = floatToNormalized<GLint>(color.colorF.alpha);
    }
}

template <bool isPureInteger>
void ConvertFromColor(const ColorGeneric &color, GLuint *outParams)
{
    if (isPureInteger)
    {
        ASSERT(color.type == ColorGeneric::Type::UInt);
        outParams[0] = color.colorUI.red;
        outParams[1] = color.colorUI.green;
        outParams[2] = color.colorUI.blue;
        outParams[3] = color.colorUI.alpha;
    }
    else
    {
        UNREACHABLE();
    }
}

template <typename ParamType>
void QueryTexLevelParameterBase(const Texture *texture,
                                TextureTarget target,
                                GLint level,
                                GLenum pname,
                                ParamType *params)
{
    ASSERT(texture != nullptr);
    const InternalFormat *info = texture->getTextureState().getImageDesc(target, level).format.info;

    switch (pname)
    {
        case GL_TEXTURE_RED_TYPE:
            *params = CastFromGLintStateValue<ParamType>(
                pname, info->redBits ? info->componentType : GL_NONE);
            break;
        case GL_TEXTURE_GREEN_TYPE:
            *params = CastFromGLintStateValue<ParamType>(
                pname, info->greenBits ? info->componentType : GL_NONE);
            break;
        case GL_TEXTURE_BLUE_TYPE:
            *params = CastFromGLintStateValue<ParamType>(
                pname, info->blueBits ? info->componentType : GL_NONE);
            break;
        case GL_TEXTURE_ALPHA_TYPE:
            *params = CastFromGLintStateValue<ParamType>(
                pname, info->alphaBits ? info->componentType : GL_NONE);
            break;
        case GL_TEXTURE_DEPTH_TYPE:
            *params = CastFromGLintStateValue<ParamType>(
                pname, info->depthBits ? info->componentType : GL_NONE);
            break;
        case GL_TEXTURE_RED_SIZE:
            *params = CastFromGLintStateValue<ParamType>(pname, info->redBits);
            break;
        case GL_TEXTURE_GREEN_SIZE:
            *params = CastFromGLintStateValue<ParamType>(pname, info->greenBits);
            break;
        case GL_TEXTURE_BLUE_SIZE:
            *params = CastFromGLintStateValue<ParamType>(pname, info->blueBits);
            break;
        case GL_TEXTURE_ALPHA_SIZE:
            *params = CastFromGLintStateValue<ParamType>(pname, info->alphaBits);
            break;
        case GL_TEXTURE_DEPTH_SIZE:
            *params = CastFromGLintStateValue<ParamType>(pname, info->depthBits);
            break;
        case GL_TEXTURE_STENCIL_SIZE:
            *params = CastFromGLintStateValue<ParamType>(pname, info->stencilBits);
            break;
        case GL_TEXTURE_SHARED_SIZE:
            *params = CastFromGLintStateValue<ParamType>(pname, info->sharedBits);
            break;
        case GL_TEXTURE_INTERNAL_FORMAT:
            *params = CastFromGLintStateValue<ParamType>(
                pname, info->internalFormat ? info->internalFormat : GL_RGBA);
            break;
        case GL_TEXTURE_WIDTH:
            *params = CastFromGLintStateValue<ParamType>(
                pname, static_cast<uint32_t>(texture->getWidth(target, level)));
            break;
        case GL_TEXTURE_HEIGHT:
            *params = CastFromGLintStateValue<ParamType>(
                pname, static_cast<uint32_t>(texture->getHeight(target, level)));
            break;
        case GL_TEXTURE_DEPTH:
            *params = CastFromGLintStateValue<ParamType>(
                pname, static_cast<uint32_t>(texture->getDepth(target, level)));
            break;
        case GL_TEXTURE_SAMPLES:
            *params = CastFromStateValue<ParamType>(pname, texture->getSamples(target, level));
            break;
        case GL_TEXTURE_FIXED_SAMPLE_LOCATIONS:
            *params = CastFromStateValue<ParamType>(
                pname, static_cast<GLint>(texture->getFixedSampleLocations(target, level)));
            break;
        case GL_TEXTURE_COMPRESSED:
            *params = CastFromStateValue<ParamType>(pname, static_cast<GLint>(info->compressed));
            break;
        case GL_MEMORY_SIZE_ANGLE:
            *params =
                CastFromStateValue<ParamType>(pname, texture->getLevelMemorySize(target, level));
            break;
        default:
            UNREACHABLE();
            break;
    }
}

template <bool isPureInteger, typename ParamType>
void QueryTexParameterBase(const Texture *texture, GLenum pname, ParamType *params)
{
    ASSERT(texture != nullptr);

    switch (pname)
    {
        case GL_TEXTURE_MAG_FILTER:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getMagFilter());
            break;
        case GL_TEXTURE_MIN_FILTER:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getMinFilter());
            break;
        case GL_TEXTURE_WRAP_S:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getWrapS());
            break;
        case GL_TEXTURE_WRAP_T:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getWrapT());
            break;
        case GL_TEXTURE_WRAP_R:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getWrapR());
            break;
        case GL_TEXTURE_IMMUTABLE_FORMAT:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getImmutableFormat());
            break;
        case GL_TEXTURE_IMMUTABLE_LEVELS:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getImmutableLevels());
            break;
        case GL_TEXTURE_USAGE_ANGLE:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getUsage());
            break;
        case GL_TEXTURE_MAX_ANISOTROPY_EXT:
            *params = CastFromStateValue<ParamType>(pname, texture->getMaxAnisotropy());
            break;
        case GL_TEXTURE_SWIZZLE_R:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getSwizzleRed());
            break;
        case GL_TEXTURE_SWIZZLE_G:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getSwizzleGreen());
            break;
        case GL_TEXTURE_SWIZZLE_B:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getSwizzleBlue());
            break;
        case GL_TEXTURE_SWIZZLE_A:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getSwizzleAlpha());
            break;
        case GL_TEXTURE_BASE_LEVEL:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getBaseLevel());
            break;
        case GL_TEXTURE_MAX_LEVEL:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getMaxLevel());
            break;
        case GL_TEXTURE_MIN_LOD:
            *params = CastFromStateValue<ParamType>(pname, texture->getMinLod());
            break;
        case GL_TEXTURE_MAX_LOD:
            *params = CastFromStateValue<ParamType>(pname, texture->getMaxLod());
            break;
        case GL_TEXTURE_COMPARE_MODE:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getCompareMode());
            break;
        case GL_TEXTURE_COMPARE_FUNC:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getCompareFunc());
            break;
        case GL_TEXTURE_SRGB_DECODE_EXT:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getSRGBDecode());
            break;
        case GL_DEPTH_STENCIL_TEXTURE_MODE:
            *params =
                CastFromGLintStateValue<ParamType>(pname, texture->getDepthStencilTextureMode());
            break;
        case GL_TEXTURE_CROP_RECT_OES:
        {
            const gl::Rectangle &crop = texture->getCrop();
            params[0]                 = CastFromGLintStateValue<ParamType>(pname, crop.x);
            params[1]                 = CastFromGLintStateValue<ParamType>(pname, crop.y);
            params[2]                 = CastFromGLintStateValue<ParamType>(pname, crop.width);
            params[3]                 = CastFromGLintStateValue<ParamType>(pname, crop.height);
            break;
        }
        case GL_GENERATE_MIPMAP:
            *params = CastFromGLintStateValue<ParamType>(pname, texture->getGenerateMipmapHint());
            break;
        case GL_MEMORY_SIZE_ANGLE:
            *params = CastFromStateValue<ParamType>(pname, texture->getMemorySize());
            break;
        case GL_TEXTURE_BORDER_COLOR:
            ConvertFromColor<isPureInteger>(texture->getBorderColor(), params);
            break;
        case GL_TEXTURE_NATIVE_ID_ANGLE:
            *params = CastFromStateValue<ParamType>(pname, texture->getNativeID());
            break;
        default:
            UNREACHABLE();
            break;
    }
}

template <bool isPureInteger, typename ParamType>
void SetTexParameterBase(Context *context, Texture *texture, GLenum pname, const ParamType *params)
{
    ASSERT(texture != nullptr);

    switch (pname)
    {
        case GL_TEXTURE_WRAP_S:
            texture->setWrapS(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_WRAP_T:
            texture->setWrapT(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_WRAP_R:
            texture->setWrapR(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_MIN_FILTER:
            texture->setMinFilter(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_MAG_FILTER:
            texture->setMagFilter(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_USAGE_ANGLE:
            texture->setUsage(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_MAX_ANISOTROPY_EXT:
            texture->setMaxAnisotropy(context, CastQueryValueTo<GLfloat>(pname, params[0]));
            break;
        case GL_TEXTURE_COMPARE_MODE:
            texture->setCompareMode(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_COMPARE_FUNC:
            texture->setCompareFunc(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_SWIZZLE_R:
            texture->setSwizzleRed(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_SWIZZLE_G:
            texture->setSwizzleGreen(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_SWIZZLE_B:
            texture->setSwizzleBlue(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_SWIZZLE_A:
            texture->setSwizzleAlpha(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_BASE_LEVEL:
        {
            (void)(texture->setBaseLevel(
                context, clampCast<GLuint>(CastQueryValueTo<GLint>(pname, params[0]))));
            break;
        }
        case GL_TEXTURE_MAX_LEVEL:
            texture->setMaxLevel(context,
                                 clampCast<GLuint>(CastQueryValueTo<GLint>(pname, params[0])));
            break;
        case GL_TEXTURE_MIN_LOD:
            texture->setMinLod(context, CastQueryValueTo<GLfloat>(pname, params[0]));
            break;
        case GL_TEXTURE_MAX_LOD:
            texture->setMaxLod(context, CastQueryValueTo<GLfloat>(pname, params[0]));
            break;
        case GL_DEPTH_STENCIL_TEXTURE_MODE:
            texture->setDepthStencilTextureMode(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_SRGB_DECODE_EXT:
            texture->setSRGBDecode(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_CROP_RECT_OES:
            texture->setCrop(gl::Rectangle(CastQueryValueTo<GLint>(pname, params[0]),
                                           CastQueryValueTo<GLint>(pname, params[1]),
                                           CastQueryValueTo<GLint>(pname, params[2]),
                                           CastQueryValueTo<GLint>(pname, params[3])));
            break;
        case GL_GENERATE_MIPMAP:
            texture->setGenerateMipmapHint(ConvertToGLenum(params[0]));
            break;
        case GL_TEXTURE_BORDER_COLOR:
            texture->setBorderColor(context, ConvertToColor<isPureInteger>(params));
            break;
        default:
            UNREACHABLE();
            break;
    }
}

template <bool isPureInteger, typename ParamType>
void QuerySamplerParameterBase(const Sampler *sampler, GLenum pname, ParamType *params)
{
    switch (pname)
    {
        case GL_TEXTURE_MIN_FILTER:
            *params = CastFromGLintStateValue<ParamType>(pname, sampler->getMinFilter());
            break;
        case GL_TEXTURE_MAG_FILTER:
            *params = CastFromGLintStateValue<ParamType>(pname, sampler->getMagFilter());
            break;
        case GL_TEXTURE_WRAP_S:
            *params = CastFromGLintStateValue<ParamType>(pname, sampler->getWrapS());
            break;
        case GL_TEXTURE_WRAP_T:
            *params = CastFromGLintStateValue<ParamType>(pname, sampler->getWrapT());
            break;
        case GL_TEXTURE_WRAP_R:
            *params = CastFromGLintStateValue<ParamType>(pname, sampler->getWrapR());
            break;
        case GL_TEXTURE_MAX_ANISOTROPY_EXT:
            *params = CastFromStateValue<ParamType>(pname, sampler->getMaxAnisotropy());
            break;
        case GL_TEXTURE_MIN_LOD:
            *params = CastFromStateValue<ParamType>(pname, sampler->getMinLod());
            break;
        case GL_TEXTURE_MAX_LOD:
            *params = CastFromStateValue<ParamType>(pname, sampler->getMaxLod());
            break;
        case GL_TEXTURE_COMPARE_MODE:
            *params = CastFromGLintStateValue<ParamType>(pname, sampler->getCompareMode());
            break;
        case GL_TEXTURE_COMPARE_FUNC:
            *params = CastFromGLintStateValue<ParamType>(pname, sampler->getCompareFunc());
            break;
        case GL_TEXTURE_SRGB_DECODE_EXT:
            *params = CastFromGLintStateValue<ParamType>(pname, sampler->getSRGBDecode());
            break;
        case GL_TEXTURE_BORDER_COLOR:
            ConvertFromColor<isPureInteger>(sampler->getBorderColor(), params);
            break;
        default:
            UNREACHABLE();
            break;
    }
}

template <bool isPureInteger, typename ParamType>
void SetSamplerParameterBase(Context *context,
                             Sampler *sampler,
                             GLenum pname,
                             const ParamType *params)
{
    switch (pname)
    {
        case GL_TEXTURE_WRAP_S:
            sampler->setWrapS(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_WRAP_T:
            sampler->setWrapT(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_WRAP_R:
            sampler->setWrapR(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_MIN_FILTER:
            sampler->setMinFilter(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_MAG_FILTER:
            sampler->setMagFilter(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_MAX_ANISOTROPY_EXT:
            sampler->setMaxAnisotropy(context, CastQueryValueTo<GLfloat>(pname, params[0]));
            break;
        case GL_TEXTURE_COMPARE_MODE:
            sampler->setCompareMode(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_COMPARE_FUNC:
            sampler->setCompareFunc(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_MIN_LOD:
            sampler->setMinLod(context, CastQueryValueTo<GLfloat>(pname, params[0]));
            break;
        case GL_TEXTURE_MAX_LOD:
            sampler->setMaxLod(context, CastQueryValueTo<GLfloat>(pname, params[0]));
            break;
        case GL_TEXTURE_SRGB_DECODE_EXT:
            sampler->setSRGBDecode(context, ConvertToGLenum(pname, params[0]));
            break;
        case GL_TEXTURE_BORDER_COLOR:
            sampler->setBorderColor(context, ConvertToColor<isPureInteger>(params));
            break;
        default:
            UNREACHABLE();
            break;
    }

    sampler->onStateChange(angle::SubjectMessage::ContentsChanged);
}

// Warning: you should ensure binding really matches attrib.bindingIndex before using this function.
template <typename ParamType, typename CurrentDataType, size_t CurrentValueCount>
void QueryVertexAttribBase(const VertexAttribute &attrib,
                           const VertexBinding &binding,
                           const CurrentDataType (&currentValueData)[CurrentValueCount],
                           GLenum pname,
                           ParamType *params)
{
    switch (pname)
    {
        case GL_CURRENT_VERTEX_ATTRIB:
            for (size_t i = 0; i < CurrentValueCount; ++i)
            {
                params[i] = CastFromStateValue<ParamType>(pname, currentValueData[i]);
            }
            break;
        case GL_VERTEX_ATTRIB_ARRAY_ENABLED:
            *params = CastFromStateValue<ParamType>(pname, static_cast<GLint>(attrib.enabled));
            break;
        case GL_VERTEX_ATTRIB_ARRAY_SIZE:
            *params = CastFromGLintStateValue<ParamType>(pname, attrib.format->channelCount);
            break;
        case GL_VERTEX_ATTRIB_ARRAY_STRIDE:
            *params = CastFromGLintStateValue<ParamType>(pname, attrib.vertexAttribArrayStride);
            break;
        case GL_VERTEX_ATTRIB_ARRAY_TYPE:
            *params = CastFromGLintStateValue<ParamType>(
                pname, gl::ToGLenum(attrib.format->vertexAttribType));
            break;
        case GL_VERTEX_ATTRIB_ARRAY_NORMALIZED:
            *params =
                CastFromStateValue<ParamType>(pname, static_cast<GLint>(attrib.format->isNorm()));
            break;
        case GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING:
            *params = CastFromGLintStateValue<ParamType>(pname, binding.getBuffer().id().value);
            break;
        case GL_VERTEX_ATTRIB_ARRAY_DIVISOR:
            *params = CastFromStateValue<ParamType>(pname, binding.getDivisor());
            break;
        case GL_VERTEX_ATTRIB_ARRAY_INTEGER:
            *params = CastFromGLintStateValue<ParamType>(pname, attrib.format->isPureInt());
            break;
        case GL_VERTEX_ATTRIB_BINDING:
            *params = CastFromGLintStateValue<ParamType>(pname, attrib.bindingIndex);
            break;
        case GL_VERTEX_ATTRIB_RELATIVE_OFFSET:
            *params = CastFromGLintStateValue<ParamType>(pname, attrib.relativeOffset);
            break;
        default:
            UNREACHABLE();
            break;
    }
}

template <typename ParamType>
void QueryBufferParameterBase(const Buffer *buffer, GLenum pname, ParamType *params)
{
    ASSERT(buffer != nullptr);

    switch (pname)
    {
        case GL_BUFFER_USAGE:
            *params = CastFromGLintStateValue<ParamType>(pname, ToGLenum(buffer->getUsage()));
            break;
        case GL_BUFFER_SIZE:
            *params = CastFromStateValue<ParamType>(pname, buffer->getSize());
            break;
        case GL_BUFFER_ACCESS_FLAGS:
            *params = CastFromGLintStateValue<ParamType>(pname, buffer->getAccessFlags());
            break;
        case GL_BUFFER_ACCESS_OES:
            *params = CastFromGLintStateValue<ParamType>(pname, buffer->getAccess());
            break;
        case GL_BUFFER_MAPPED:
            *params = CastFromStateValue<ParamType>(pname, buffer->isMapped());
            break;
        case GL_BUFFER_MAP_OFFSET:
            *params = CastFromStateValue<ParamType>(pname, buffer->getMapOffset());
            break;
        case GL_BUFFER_MAP_LENGTH:
            *params = CastFromStateValue<ParamType>(pname, buffer->getMapLength());
            break;
        case GL_MEMORY_SIZE_ANGLE:
            *params = CastFromStateValue<ParamType>(pname, buffer->getMemorySize());
            break;
        default:
            UNREACHABLE();
            break;
    }
}

GLint GetCommonVariableProperty(const sh::ShaderVariable &var, GLenum prop)
{
    switch (prop)
    {
        case GL_TYPE:
            return clampCast<GLint>(var.type);

        case GL_ARRAY_SIZE:
            // Queryable variables are guaranteed not to be arrays of arrays or arrays of structs,
            // see GLES 3.1 spec section 7.3.1.1 page 77.
            return clampCast<GLint>(var.getBasicTypeElementCount());

        case GL_NAME_LENGTH:
            // ES31 spec p84: This counts the terminating null char.
            return clampCast<GLint>(var.name.size() + 1u);

        default:
            UNREACHABLE();
            return GL_INVALID_VALUE;
    }
}

GLint GetInputResourceProperty(const Program *program, GLuint index, GLenum prop)
{
    const sh::ShaderVariable &variable = program->getInputResource(index);

    switch (prop)
    {
        case GL_TYPE:
        case GL_ARRAY_SIZE:
            return GetCommonVariableProperty(variable, prop);

        case GL_NAME_LENGTH:
            return clampCast<GLint>(program->getInputResourceName(index).size() + 1u);

        case GL_LOCATION:
            return variable.location;

        // The query is targeted at the set of active input variables used by the first shader stage
        // of program. If program contains multiple shader stages then input variables from any
        // stage other than the first will not be enumerated. Since we found the variable to get
        // this far, we know it exists in the first attached shader stage.
        case GL_REFERENCED_BY_VERTEX_SHADER:
            return program->getState().getFirstAttachedShaderStageType() == ShaderType::Vertex;
        case GL_REFERENCED_BY_FRAGMENT_SHADER:
            return program->getState().getFirstAttachedShaderStageType() == ShaderType::Fragment;
        case GL_REFERENCED_BY_COMPUTE_SHADER:
            return program->getState().getFirstAttachedShaderStageType() == ShaderType::Compute;
        case GL_REFERENCED_BY_GEOMETRY_SHADER_EXT:
            return program->getState().getFirstAttachedShaderStageType() == ShaderType::Geometry;

        default:
            UNREACHABLE();
            return GL_INVALID_VALUE;
    }
}

GLint GetOutputResourceProperty(const Program *program, GLuint index, const GLenum prop)
{
    const sh::ShaderVariable &outputVariable = program->getOutputResource(index);

    switch (prop)
    {
        case GL_TYPE:
        case GL_ARRAY_SIZE:
            return GetCommonVariableProperty(outputVariable, prop);

        case GL_NAME_LENGTH:
            return clampCast<GLint>(program->getOutputResourceName(index).size() + 1u);

        case GL_LOCATION:
            return outputVariable.location;

        case GL_LOCATION_INDEX_EXT:
            // EXT_blend_func_extended
            if (program->getState().getLastAttachedShaderStageType() == gl::ShaderType::Fragment)
            {
                return program->getFragDataIndex(outputVariable.name);
            }
            return GL_INVALID_INDEX;

        // The set of active user-defined outputs from the final shader stage in this program. If
        // the final stage is a Fragment Shader, then this represents the fragment outputs that get
        // written to individual color buffers. If the program only contains a Compute Shader, then
        // there are no user-defined outputs.
        case GL_REFERENCED_BY_VERTEX_SHADER:
            return program->getState().getLastAttachedShaderStageType() == ShaderType::Vertex;
        case GL_REFERENCED_BY_FRAGMENT_SHADER:
            return program->getState().getLastAttachedShaderStageType() == ShaderType::Fragment;
        case GL_REFERENCED_BY_COMPUTE_SHADER:
            return program->getState().getLastAttachedShaderStageType() == ShaderType::Compute;
        case GL_REFERENCED_BY_GEOMETRY_SHADER_EXT:
            return program->getState().getLastAttachedShaderStageType() == ShaderType::Geometry;

        default:
            UNREACHABLE();
            return GL_INVALID_VALUE;
    }
}

GLint GetTransformFeedbackVaryingResourceProperty(const Program *program,
                                                  GLuint index,
                                                  const GLenum prop)
{
    const auto &tfVariable = program->getTransformFeedbackVaryingResource(index);
    switch (prop)
    {
        case GL_TYPE:
            return clampCast<GLint>(tfVariable.type);

        case GL_ARRAY_SIZE:
            return clampCast<GLint>(tfVariable.size());

        case GL_NAME_LENGTH:
            return clampCast<GLint>(tfVariable.nameWithArrayIndex().size() + 1);

        default:
            UNREACHABLE();
            return GL_INVALID_VALUE;
    }
}

GLint QueryProgramInterfaceActiveResources(const Program *program, GLenum programInterface)
{
    switch (programInterface)
    {
        case GL_PROGRAM_INPUT:
            return clampCast<GLint>(program->getState().getProgramInputs().size());

        case GL_PROGRAM_OUTPUT:
            return clampCast<GLint>(program->getState().getOutputVariables().size());

        case GL_UNIFORM:
            return clampCast<GLint>(program->getState().getUniforms().size());

        case GL_UNIFORM_BLOCK:
            return clampCast<GLint>(program->getState().getUniformBlocks().size());

        case GL_ATOMIC_COUNTER_BUFFER:
            return clampCast<GLint>(program->getState().getAtomicCounterBuffers().size());

        case GL_BUFFER_VARIABLE:
            return clampCast<GLint>(program->getState().getBufferVariables().size());

        case GL_SHADER_STORAGE_BLOCK:
            return clampCast<GLint>(program->getState().getShaderStorageBlocks().size());

        case GL_TRANSFORM_FEEDBACK_VARYING:
            return clampCast<GLint>(program->getTransformFeedbackVaryingCount());

        default:
            UNREACHABLE();
            return 0;
    }
}

template <typename T, typename M>
GLint FindMaxSize(const std::vector<T> &resources, M member)
{
    GLint max = 0;
    for (const T &resource : resources)
    {
        max = std::max(max, clampCast<GLint>((resource.*member).size()));
    }
    return max;
}

GLint QueryProgramInterfaceMaxNameLength(const Program *program, GLenum programInterface)
{
    GLint maxNameLength = 0;
    switch (programInterface)
    {
        case GL_PROGRAM_INPUT:
            maxNameLength = program->getInputResourceMaxNameSize();
            break;

        case GL_PROGRAM_OUTPUT:
            maxNameLength = program->getOutputResourceMaxNameSize();
            break;

        case GL_UNIFORM:
            maxNameLength = FindMaxSize(program->getState().getUniforms(), &LinkedUniform::name);
            break;

        case GL_UNIFORM_BLOCK:
            return program->getActiveUniformBlockMaxNameLength();

        case GL_BUFFER_VARIABLE:
            maxNameLength =
                FindMaxSize(program->getState().getBufferVariables(), &BufferVariable::name);
            break;

        case GL_SHADER_STORAGE_BLOCK:
            return program->getActiveShaderStorageBlockMaxNameLength();

        case GL_TRANSFORM_FEEDBACK_VARYING:
            return clampCast<GLint>(program->getTransformFeedbackVaryingMaxLength());

        default:
            UNREACHABLE();
            return 0;
    }
    // This length includes an extra character for the null terminator.
    return (maxNameLength == 0 ? 0 : maxNameLength + 1);
}

GLint QueryProgramInterfaceMaxNumActiveVariables(const Program *program, GLenum programInterface)
{
    switch (programInterface)
    {
        case GL_UNIFORM_BLOCK:
            return FindMaxSize(program->getState().getUniformBlocks(),
                               &InterfaceBlock::memberIndexes);
        case GL_ATOMIC_COUNTER_BUFFER:
            return FindMaxSize(program->getState().getAtomicCounterBuffers(),
                               &AtomicCounterBuffer::memberIndexes);

        case GL_SHADER_STORAGE_BLOCK:
            return FindMaxSize(program->getState().getShaderStorageBlocks(),
                               &InterfaceBlock::memberIndexes);

        default:
            UNREACHABLE();
            return 0;
    }
}

GLenum GetUniformPropertyEnum(GLenum prop)
{
    switch (prop)
    {
        case GL_UNIFORM_TYPE:
            return GL_TYPE;
        case GL_UNIFORM_SIZE:
            return GL_ARRAY_SIZE;
        case GL_UNIFORM_NAME_LENGTH:
            return GL_NAME_LENGTH;
        case GL_UNIFORM_BLOCK_INDEX:
            return GL_BLOCK_INDEX;
        case GL_UNIFORM_OFFSET:
            return GL_OFFSET;
        case GL_UNIFORM_ARRAY_STRIDE:
            return GL_ARRAY_STRIDE;
        case GL_UNIFORM_MATRIX_STRIDE:
            return GL_MATRIX_STRIDE;
        case GL_UNIFORM_IS_ROW_MAJOR:
            return GL_IS_ROW_MAJOR;

        default:
            return prop;
    }
}

GLenum GetUniformBlockPropertyEnum(GLenum prop)
{
    switch (prop)
    {
        case GL_UNIFORM_BLOCK_BINDING:
            return GL_BUFFER_BINDING;

        case GL_UNIFORM_BLOCK_DATA_SIZE:
            return GL_BUFFER_DATA_SIZE;

        case GL_UNIFORM_BLOCK_NAME_LENGTH:
            return GL_NAME_LENGTH;

        case GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS:
            return GL_NUM_ACTIVE_VARIABLES;

        case GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES:
            return GL_ACTIVE_VARIABLES;

        case GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER:
            return GL_REFERENCED_BY_VERTEX_SHADER;

        case GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER:
            return GL_REFERENCED_BY_FRAGMENT_SHADER;

        default:
            return prop;
    }
}

void GetShaderVariableBufferResourceProperty(const ShaderVariableBuffer &buffer,
                                             GLenum pname,
                                             GLint *params,
                                             GLsizei bufSize,
                                             GLsizei *outputPosition)

{
    switch (pname)
    {
        case GL_BUFFER_BINDING:
            params[(*outputPosition)++] = buffer.binding;
            break;
        case GL_BUFFER_DATA_SIZE:
            params[(*outputPosition)++] = clampCast<GLint>(buffer.dataSize);
            break;
        case GL_NUM_ACTIVE_VARIABLES:
            params[(*outputPosition)++] = buffer.numActiveVariables();
            break;
        case GL_ACTIVE_VARIABLES:
            for (size_t memberIndex = 0;
                 memberIndex < buffer.memberIndexes.size() && *outputPosition < bufSize;
                 ++memberIndex)
            {
                params[(*outputPosition)++] = clampCast<GLint>(buffer.memberIndexes[memberIndex]);
            }
            break;
        case GL_REFERENCED_BY_VERTEX_SHADER:
            params[(*outputPosition)++] = static_cast<GLint>(buffer.isActive(ShaderType::Vertex));
            break;
        case GL_REFERENCED_BY_FRAGMENT_SHADER:
            params[(*outputPosition)++] = static_cast<GLint>(buffer.isActive(ShaderType::Fragment));
            break;
        case GL_REFERENCED_BY_COMPUTE_SHADER:
            params[(*outputPosition)++] = static_cast<GLint>(buffer.isActive(ShaderType::Compute));
            break;
        case GL_REFERENCED_BY_GEOMETRY_SHADER_EXT:
            params[(*outputPosition)++] = static_cast<GLint>(buffer.isActive(ShaderType::Geometry));
            break;
        default:
            UNREACHABLE();
            break;
    }
}

void GetInterfaceBlockResourceProperty(const InterfaceBlock &block,
                                       GLenum pname,
                                       GLint *params,
                                       GLsizei bufSize,
                                       GLsizei *outputPosition)
{
    if (pname == GL_NAME_LENGTH)
    {
        params[(*outputPosition)++] = clampCast<GLint>(block.nameWithArrayIndex().size() + 1);
        return;
    }
    GetShaderVariableBufferResourceProperty(block, pname, params, bufSize, outputPosition);
}

void GetUniformBlockResourceProperty(const Program *program,
                                     GLuint blockIndex,
                                     GLenum pname,
                                     GLint *params,
                                     GLsizei bufSize,
                                     GLsizei *outputPosition)

{
    ASSERT(*outputPosition < bufSize);
    const auto &block = program->getUniformBlockByIndex(blockIndex);
    GetInterfaceBlockResourceProperty(block, pname, params, bufSize, outputPosition);
}

void GetShaderStorageBlockResourceProperty(const Program *program,
                                           GLuint blockIndex,
                                           GLenum pname,
                                           GLint *params,
                                           GLsizei bufSize,
                                           GLsizei *outputPosition)

{
    ASSERT(*outputPosition < bufSize);
    const auto &block = program->getShaderStorageBlockByIndex(blockIndex);
    GetInterfaceBlockResourceProperty(block, pname, params, bufSize, outputPosition);
}

void GetAtomicCounterBufferResourceProperty(const Program *program,
                                            GLuint index,
                                            GLenum pname,
                                            GLint *params,
                                            GLsizei bufSize,
                                            GLsizei *outputPosition)

{
    ASSERT(*outputPosition < bufSize);
    const auto &buffer = program->getState().getAtomicCounterBuffers()[index];
    GetShaderVariableBufferResourceProperty(buffer, pname, params, bufSize, outputPosition);
}

bool IsTextureEnvEnumParameter(TextureEnvParameter pname)
{
    switch (pname)
    {
        case TextureEnvParameter::Mode:
        case TextureEnvParameter::CombineRgb:
        case TextureEnvParameter::CombineAlpha:
        case TextureEnvParameter::Src0Rgb:
        case TextureEnvParameter::Src1Rgb:
        case TextureEnvParameter::Src2Rgb:
        case TextureEnvParameter::Src0Alpha:
        case TextureEnvParameter::Src1Alpha:
        case TextureEnvParameter::Src2Alpha:
        case TextureEnvParameter::Op0Rgb:
        case TextureEnvParameter::Op1Rgb:
        case TextureEnvParameter::Op2Rgb:
        case TextureEnvParameter::Op0Alpha:
        case TextureEnvParameter::Op1Alpha:
        case TextureEnvParameter::Op2Alpha:
        case TextureEnvParameter::PointCoordReplace:
            return true;
        default:
            return false;
    }
}

}  // anonymous namespace

void QueryFramebufferAttachmentParameteriv(const Context *context,
                                           const Framebuffer *framebuffer,
                                           GLenum attachment,
                                           GLenum pname,
                                           GLint *params)
{
    ASSERT(framebuffer);

    const FramebufferAttachment *attachmentObject = framebuffer->getAttachment(context, attachment);

    if (attachmentObject == nullptr)
    {
        // ES 2.0.25 spec pg 127 states that if the value of FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE
        // is NONE, then querying any other pname will generate INVALID_ENUM.

        // ES 3.0.2 spec pg 235 states that if the attachment type is none,
        // GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME will return zero and be an
        // INVALID_OPERATION for all other pnames

        switch (pname)
        {
            case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE:
                *params = GL_NONE;
                break;

            case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME:
                *params = 0;
                break;

            default:
                UNREACHABLE();
                break;
        }

        return;
    }

    switch (pname)
    {
        case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE:
            *params = attachmentObject->type();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME:
            *params = attachmentObject->id();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL:
            *params = attachmentObject->mipLevel();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE:
        {
            TextureTarget face = attachmentObject->cubeMapFace();
            if (face != TextureTarget::InvalidEnum)
            {
                *params = ToGLenum(attachmentObject->cubeMapFace());
            }
            else
            {
                // This happens when the attachment isn't a texture cube map face
                *params = GL_NONE;
            }
        }
        break;

        case GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE:
            *params = attachmentObject->getRedSize();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE:
            *params = attachmentObject->getGreenSize();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE:
            *params = attachmentObject->getBlueSize();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE:
            *params = attachmentObject->getAlphaSize();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE:
            *params = attachmentObject->getDepthSize();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE:
            *params = attachmentObject->getStencilSize();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE:
            *params = attachmentObject->getComponentType();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING:
            *params = attachmentObject->getColorEncoding();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER:
            *params = attachmentObject->layer();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR:
            *params = attachmentObject->getNumViews();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_BASE_VIEW_INDEX_OVR:
            *params = attachmentObject->getBaseViewIndex();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_LAYERED_EXT:
            *params = attachmentObject->isLayered();
            break;

        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_SAMPLES_EXT:
            if (attachmentObject->type() == GL_TEXTURE)
            {
                *params = attachmentObject->getSamples();
            }
            else
            {
                *params = 0;
            }
            break;

        default:
            UNREACHABLE();
            break;
    }
}

void QueryBufferParameteriv(const Buffer *buffer, GLenum pname, GLint *params)
{
    QueryBufferParameterBase(buffer, pname, params);
}

void QueryBufferParameteri64v(const Buffer *buffer, GLenum pname, GLint64 *params)
{
    QueryBufferParameterBase(buffer, pname, params);
}

void QueryBufferPointerv(const Buffer *buffer, GLenum pname, void **params)
{
    switch (pname)
    {
        case GL_BUFFER_MAP_POINTER:
            *params = buffer->getMapPointer();
            break;

        default:
            UNREACHABLE();
            break;
    }
}

void QueryProgramiv(Context *context, const Program *program, GLenum pname, GLint *params)
{
    ASSERT(program != nullptr || pname == GL_COMPLETION_STATUS_KHR);

    switch (pname)
    {
        case GL_DELETE_STATUS:
            *params = program->isFlaggedForDeletion();
            return;
        case GL_LINK_STATUS:
            *params = program->isLinked();
            return;
        case GL_COMPLETION_STATUS_KHR:
            if (context->isContextLost())
            {
                *params = GL_TRUE;
            }
            else
            {
                *params = program->isLinking() ? GL_FALSE : GL_TRUE;
            }
            return;
        case GL_VALIDATE_STATUS:
            *params = program->isValidated();
            return;
        case GL_INFO_LOG_LENGTH:
            *params = program->getInfoLogLength();
            return;
        case GL_ATTACHED_SHADERS:
            *params = program->getAttachedShadersCount();
            return;
        case GL_ACTIVE_ATTRIBUTES:
            *params = program->getActiveAttributeCount();
            return;
        case GL_ACTIVE_ATTRIBUTE_MAX_LENGTH:
            *params = program->getActiveAttributeMaxLength();
            return;
        case GL_ACTIVE_UNIFORMS:
            *params = program->getActiveUniformCount();
            return;
        case GL_ACTIVE_UNIFORM_MAX_LENGTH:
            *params = program->getActiveUniformMaxLength();
            return;
        case GL_PROGRAM_BINARY_LENGTH_OES:
            *params = context->getCaps().programBinaryFormats.empty()
                          ? 0
                          : program->getBinaryLength(context);
            return;
        case GL_ACTIVE_UNIFORM_BLOCKS:
            *params = program->getActiveUniformBlockCount();
            return;
        case GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH:
            *params = program->getActiveUniformBlockMaxNameLength();
            break;
        case GL_TRANSFORM_FEEDBACK_BUFFER_MODE:
            *params = program->getTransformFeedbackBufferMode();
            break;
        case GL_TRANSFORM_FEEDBACK_VARYINGS:
            *params = clampCast<GLint>(program->getTransformFeedbackVaryingCount());
            break;
        case GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH:
            *params = program->getTransformFeedbackVaryingMaxLength();
            break;
        case GL_PROGRAM_BINARY_RETRIEVABLE_HINT:
            *params = program->getBinaryRetrievableHint();
            break;
        case GL_PROGRAM_SEPARABLE:
            *params = program->isSeparable();
            break;
        case GL_COMPUTE_WORK_GROUP_SIZE:
        {
            const sh::WorkGroupSize &localSize = program->getComputeShaderLocalSize();
            params[0]                          = localSize[0];
            params[1]                          = localSize[1];
            params[2]                          = localSize[2];
        }
        break;
        case GL_ACTIVE_ATOMIC_COUNTER_BUFFERS:
            *params = program->getActiveAtomicCounterBufferCount();
            break;
        case GL_GEOMETRY_LINKED_INPUT_TYPE_EXT:
            *params = ToGLenum(program->getGeometryShaderInputPrimitiveType());
            break;
        case GL_GEOMETRY_LINKED_OUTPUT_TYPE_EXT:
            *params = ToGLenum(program->getGeometryShaderOutputPrimitiveType());
            break;
        case GL_GEOMETRY_LINKED_VERTICES_OUT_EXT:
            *params = program->getGeometryShaderMaxVertices();
            break;
        case GL_GEOMETRY_SHADER_INVOCATIONS_EXT:
            *params = program->getGeometryShaderInvocations();
            break;
        default:
            UNREACHABLE();
            break;
    }
}

void QueryRenderbufferiv(const Context *context,
                         const Renderbuffer *renderbuffer,
                         GLenum pname,
                         GLint *params)
{
    ASSERT(renderbuffer != nullptr);

    switch (pname)
    {
        case GL_RENDERBUFFER_WIDTH:
            *params = renderbuffer->getWidth();
            break;
        case GL_RENDERBUFFER_HEIGHT:
            *params = renderbuffer->getHeight();
            break;
        case GL_RENDERBUFFER_INTERNAL_FORMAT:
            // Special case the WebGL 1 DEPTH_STENCIL format.
            if (context->isWebGL1() &&
                renderbuffer->getFormat().info->internalFormat == GL_DEPTH24_STENCIL8)
            {
                *params = GL_DEPTH_STENCIL;
            }
            else
            {
                *params = renderbuffer->getFormat().info->internalFormat;
            }
            break;
        case GL_RENDERBUFFER_RED_SIZE:
            *params = renderbuffer->getRedSize();
            break;
        case GL_RENDERBUFFER_GREEN_SIZE:
            *params = renderbuffer->getGreenSize();
            break;
        case GL_RENDERBUFFER_BLUE_SIZE:
            *params = renderbuffer->getBlueSize();
            break;
        case GL_RENDERBUFFER_ALPHA_SIZE:
            *params = renderbuffer->getAlphaSize();
            break;
        case GL_RENDERBUFFER_DEPTH_SIZE:
            *params = renderbuffer->getDepthSize();
            break;
        case GL_RENDERBUFFER_STENCIL_SIZE:
            *params = renderbuffer->getStencilSize();
            break;
        case GL_RENDERBUFFER_SAMPLES_ANGLE:
            *params = renderbuffer->getSamples();
            break;
        case GL_MEMORY_SIZE_ANGLE:
            *params = renderbuffer->getMemorySize();
            break;
        default:
            UNREACHABLE();
            break;
    }
}

void QueryShaderiv(const Context *context, Shader *shader, GLenum pname, GLint *params)
{
    ASSERT(shader != nullptr || pname == GL_COMPLETION_STATUS_KHR);

    switch (pname)
    {
        case GL_SHADER_TYPE:
            *params = static_cast<GLint>(ToGLenum(shader->getType()));
            return;
        case GL_DELETE_STATUS:
            *params = shader->isFlaggedForDeletion();
            return;
        case GL_COMPILE_STATUS:
            *params = shader->isCompiled() ? GL_TRUE : GL_FALSE;
            return;
        case GL_COMPLETION_STATUS_KHR:
            if (context->isContextLost())
            {
                *params = GL_TRUE;
            }
            else
            {
                *params = shader->isCompleted() ? GL_TRUE : GL_FALSE;
            }
            return;
        case GL_INFO_LOG_LENGTH:
            *params = shader->getInfoLogLength();
            return;
        case GL_SHADER_SOURCE_LENGTH:
            *params = shader->getSourceLength();
            return;
        case GL_TRANSLATED_SHADER_SOURCE_LENGTH_ANGLE:
            *params = shader->getTranslatedSourceWithDebugInfoLength();
            return;
        default:
            UNREACHABLE();
            break;
    }
}

void QueryTexLevelParameterfv(const Texture *texture,
                              TextureTarget target,
                              GLint level,
                              GLenum pname,
                              GLfloat *params)
{
    QueryTexLevelParameterBase(texture, target, level, pname, params);
}

void QueryTexLevelParameteriv(const Texture *texture,
                              TextureTarget target,
                              GLint level,
                              GLenum pname,
                              GLint *params)
{
    QueryTexLevelParameterBase(texture, target, level, pname, params);
}

void QueryTexParameterfv(const Texture *texture, GLenum pname, GLfloat *params)
{
    QueryTexParameterBase<false>(texture, pname, params);
}

void QueryTexParameteriv(const Texture *texture, GLenum pname, GLint *params)
{
    QueryTexParameterBase<false>(texture, pname, params);
}

void QueryTexParameterIiv(const Texture *texture, GLenum pname, GLint *params)
{
    QueryTexParameterBase<true>(texture, pname, params);
}

void QueryTexParameterIuiv(const Texture *texture, GLenum pname, GLuint *params)
{
    QueryTexParameterBase<true>(texture, pname, params);
}

void QuerySamplerParameterfv(const Sampler *sampler, GLenum pname, GLfloat *params)
{
    QuerySamplerParameterBase<false>(sampler, pname, params);
}

void QuerySamplerParameteriv(const Sampler *sampler, GLenum pname, GLint *params)
{
    QuerySamplerParameterBase<false>(sampler, pname, params);
}

void QuerySamplerParameterIiv(const Sampler *sampler, GLenum pname, GLint *params)
{
    QuerySamplerParameterBase<true>(sampler, pname, params);
}

void QuerySamplerParameterIuiv(const Sampler *sampler, GLenum pname, GLuint *params)
{
    QuerySamplerParameterBase<true>(sampler, pname, params);
}

void QueryVertexAttribfv(const VertexAttribute &attrib,
                         const VertexBinding &binding,
                         const VertexAttribCurrentValueData &currentValueData,
                         GLenum pname,
                         GLfloat *params)
{
    QueryVertexAttribBase(attrib, binding, currentValueData.Values.FloatValues, pname, params);
}

void QueryVertexAttribiv(const VertexAttribute &attrib,
                         const VertexBinding &binding,
                         const VertexAttribCurrentValueData &currentValueData,
                         GLenum pname,
                         GLint *params)
{
    QueryVertexAttribBase(attrib, binding, currentValueData.Values.FloatValues, pname, params);
}

void QueryVertexAttribPointerv(const VertexAttribute &attrib, GLenum pname, void **pointer)
{
    switch (pname)
    {
        case GL_VERTEX_ATTRIB_ARRAY_POINTER:
            *pointer = const_cast<void *>(attrib.pointer);
            break;

        default:
            UNREACHABLE();
            break;
    }
}

void QueryVertexAttribIiv(const VertexAttribute &attrib,
                          const VertexBinding &binding,
                          const VertexAttribCurrentValueData &currentValueData,
                          GLenum pname,
                          GLint *params)
{
    QueryVertexAttribBase(attrib, binding, currentValueData.Values.IntValues, pname, params);
}

void QueryVertexAttribIuiv(const VertexAttribute &attrib,
                           const VertexBinding &binding,
                           const VertexAttribCurrentValueData &currentValueData,
                           GLenum pname,
                           GLuint *params)
{
    QueryVertexAttribBase(attrib, binding, currentValueData.Values.UnsignedIntValues, pname,
                          params);
}

void QueryActiveUniformBlockiv(const Program *program,
                               GLuint uniformBlockIndex,
                               GLenum pname,
                               GLint *params)
{
    GLenum prop = GetUniformBlockPropertyEnum(pname);
    QueryProgramResourceiv(program, GL_UNIFORM_BLOCK, uniformBlockIndex, 1, &prop,
                           std::numeric_limits<GLsizei>::max(), nullptr, params);
}

void QueryInternalFormativ(const TextureCaps &format, GLenum pname, GLsizei bufSize, GLint *params)
{
    switch (pname)
    {
        case GL_NUM_SAMPLE_COUNTS:
            if (bufSize != 0)
            {
                *params = clampCast<GLint>(format.sampleCounts.size());
            }
            break;

        case GL_SAMPLES:
        {
            size_t returnCount   = std::min<size_t>(bufSize, format.sampleCounts.size());
            auto sampleReverseIt = format.sampleCounts.rbegin();
            for (size_t sampleIndex = 0; sampleIndex < returnCount; ++sampleIndex)
            {
                params[sampleIndex] = *sampleReverseIt++;
            }
        }
        break;

        default:
            UNREACHABLE();
            break;
    }
}

void QueryFramebufferParameteriv(const Framebuffer *framebuffer, GLenum pname, GLint *params)
{
    ASSERT(framebuffer);

    switch (pname)
    {
        case GL_FRAMEBUFFER_DEFAULT_WIDTH:
            *params = framebuffer->getDefaultWidth();
            break;
        case GL_FRAMEBUFFER_DEFAULT_HEIGHT:
            *params = framebuffer->getDefaultHeight();
            break;
        case GL_FRAMEBUFFER_DEFAULT_SAMPLES:
            *params = framebuffer->getDefaultSamples();
            break;
        case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS:
            *params = ConvertToGLBoolean(framebuffer->getDefaultFixedSampleLocations());
            break;
        case GL_FRAMEBUFFER_DEFAULT_LAYERS_EXT:
            *params = framebuffer->getDefaultLayers();
            break;
        default:
            UNREACHABLE();
            break;
    }
}

angle::Result QuerySynciv(const Context *context,
                          const Sync *sync,
                          GLenum pname,
                          GLsizei bufSize,
                          GLsizei *length,
                          GLint *values)
{
    ASSERT(sync != nullptr || pname == GL_SYNC_STATUS);

    // All queries return one value, exit early if the buffer can't fit anything.
    if (bufSize < 1)
    {
        if (length != nullptr)
        {
            *length = 0;
        }
        return angle::Result::Continue;
    }

    switch (pname)
    {
        case GL_OBJECT_TYPE:
            *values = clampCast<GLint>(GL_SYNC_FENCE);
            break;
        case GL_SYNC_CONDITION:
            *values = clampCast<GLint>(sync->getCondition());
            break;
        case GL_SYNC_FLAGS:
            *values = clampCast<GLint>(sync->getFlags());
            break;
        case GL_SYNC_STATUS:
            if (context->isContextLost())
            {
                *values = GL_SIGNALED;
            }
            else
            {
                ANGLE_TRY(sync->getStatus(context, values));
            }
            break;

        default:
            UNREACHABLE();
            break;
    }

    if (length != nullptr)
    {
        *length = 1;
    }

    return angle::Result::Continue;
}

void SetTexParameterf(Context *context, Texture *texture, GLenum pname, GLfloat param)
{
    SetTexParameterBase<false>(context, texture, pname, &param);
}

void SetTexParameterfv(Context *context, Texture *texture, GLenum pname, const GLfloat *params)
{
    SetTexParameterBase<false>(context, texture, pname, params);
}

void SetTexParameteri(Context *context, Texture *texture, GLenum pname, GLint param)
{
    SetTexParameterBase<false>(context, texture, pname, &param);
}

void SetTexParameteriv(Context *context, Texture *texture, GLenum pname, const GLint *params)
{
    SetTexParameterBase<false>(context, texture, pname, params);
}

void SetTexParameterIiv(Context *context, Texture *texture, GLenum pname, const GLint *params)
{
    SetTexParameterBase<true>(context, texture, pname, params);
}

void SetTexParameterIuiv(Context *context, Texture *texture, GLenum pname, const GLuint *params)
{
    SetTexParameterBase<true>(context, texture, pname, params);
}

void SetSamplerParameterf(Context *context, Sampler *sampler, GLenum pname, GLfloat param)
{
    SetSamplerParameterBase<false>(context, sampler, pname, &param);
}

void SetSamplerParameterfv(Context *context, Sampler *sampler, GLenum pname, const GLfloat *params)
{
    SetSamplerParameterBase<false>(context, sampler, pname, params);
}

void SetSamplerParameteri(Context *context, Sampler *sampler, GLenum pname, GLint param)
{
    SetSamplerParameterBase<false>(context, sampler, pname, &param);
}

void SetSamplerParameteriv(Context *context, Sampler *sampler, GLenum pname, const GLint *params)
{
    SetSamplerParameterBase<false>(context, sampler, pname, params);
}

void SetSamplerParameterIiv(Context *context, Sampler *sampler, GLenum pname, const GLint *params)
{
    SetSamplerParameterBase<true>(context, sampler, pname, params);
}

void SetSamplerParameterIuiv(Context *context, Sampler *sampler, GLenum pname, const GLuint *params)
{
    SetSamplerParameterBase<true>(context, sampler, pname, params);
}

void SetFramebufferParameteri(const Context *context,
                              Framebuffer *framebuffer,
                              GLenum pname,
                              GLint param)
{
    ASSERT(framebuffer);

    switch (pname)
    {
        case GL_FRAMEBUFFER_DEFAULT_WIDTH:
            framebuffer->setDefaultWidth(context, param);
            break;
        case GL_FRAMEBUFFER_DEFAULT_HEIGHT:
            framebuffer->setDefaultHeight(context, param);
            break;
        case GL_FRAMEBUFFER_DEFAULT_SAMPLES:
            framebuffer->setDefaultSamples(context, param);
            break;
        case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS:
            framebuffer->setDefaultFixedSampleLocations(context, ConvertToBool(param));
            break;
        case GL_FRAMEBUFFER_DEFAULT_LAYERS_EXT:
            framebuffer->setDefaultLayers(param);
            break;
        default:
            UNREACHABLE();
            break;
    }
}

void SetProgramParameteri(Program *program, GLenum pname, GLint value)
{
    ASSERT(program);

    switch (pname)
    {
        case GL_PROGRAM_BINARY_RETRIEVABLE_HINT:
            program->setBinaryRetrievableHint(ConvertToBool(value));
            break;
        case GL_PROGRAM_SEPARABLE:
            program->setSeparable(ConvertToBool(value));
            break;
        default:
            UNREACHABLE();
            break;
    }
}

GLint GetUniformResourceProperty(const Program *program, GLuint index, const GLenum prop)
{
    const auto &uniform = program->getUniformByIndex(index);
    GLenum resourceProp = GetUniformPropertyEnum(prop);
    switch (resourceProp)
    {
        case GL_TYPE:
        case GL_ARRAY_SIZE:
        case GL_NAME_LENGTH:
            return GetCommonVariableProperty(uniform, resourceProp);

        case GL_LOCATION:
            return program->getUniformLocation(uniform.name);

        case GL_BLOCK_INDEX:
            return (uniform.isAtomicCounter() ? -1 : uniform.bufferIndex);

        case GL_OFFSET:
            return uniform.blockInfo.offset;

        case GL_ARRAY_STRIDE:
            return uniform.blockInfo.arrayStride;

        case GL_MATRIX_STRIDE:
            return uniform.blockInfo.matrixStride;

        case GL_IS_ROW_MAJOR:
            return static_cast<GLint>(uniform.blockInfo.isRowMajorMatrix);

        case GL_REFERENCED_BY_VERTEX_SHADER:
            return uniform.isActive(ShaderType::Vertex);

        case GL_REFERENCED_BY_FRAGMENT_SHADER:
            return uniform.isActive(ShaderType::Fragment);

        case GL_REFERENCED_BY_COMPUTE_SHADER:
            return uniform.isActive(ShaderType::Compute);

        case GL_REFERENCED_BY_GEOMETRY_SHADER_EXT:
            return uniform.isActive(ShaderType::Geometry);

        case GL_ATOMIC_COUNTER_BUFFER_INDEX:
            return (uniform.isAtomicCounter() ? uniform.bufferIndex : -1);

        default:
            UNREACHABLE();
            return 0;
    }
}

GLint GetBufferVariableResourceProperty(const Program *program, GLuint index, const GLenum prop)
{
    const BufferVariable &bufferVariable = program->getBufferVariableByIndex(index);
    switch (prop)
    {
        case GL_TYPE:
        case GL_ARRAY_SIZE:
        case GL_NAME_LENGTH:
            return GetCommonVariableProperty(bufferVariable, prop);

        case GL_BLOCK_INDEX:
            return bufferVariable.bufferIndex;

        case GL_OFFSET:
            return bufferVariable.blockInfo.offset;

        case GL_ARRAY_STRIDE:
            return bufferVariable.blockInfo.arrayStride;

        case GL_MATRIX_STRIDE:
            return bufferVariable.blockInfo.matrixStride;

        case GL_IS_ROW_MAJOR:
            return static_cast<GLint>(bufferVariable.blockInfo.isRowMajorMatrix);

        case GL_REFERENCED_BY_VERTEX_SHADER:
            return bufferVariable.isActive(ShaderType::Vertex);

        case GL_REFERENCED_BY_FRAGMENT_SHADER:
            return bufferVariable.isActive(ShaderType::Fragment);

        case GL_REFERENCED_BY_COMPUTE_SHADER:
            return bufferVariable.isActive(ShaderType::Compute);

        case GL_REFERENCED_BY_GEOMETRY_SHADER_EXT:
            return bufferVariable.isActive(ShaderType::Geometry);

        case GL_TOP_LEVEL_ARRAY_SIZE:
            return bufferVariable.topLevelArraySize;

        case GL_TOP_LEVEL_ARRAY_STRIDE:
            return bufferVariable.blockInfo.topLevelArrayStride;

        default:
            UNREACHABLE();
            return 0;
    }
}

GLuint QueryProgramResourceIndex(const Program *program,
                                 GLenum programInterface,
                                 const GLchar *name)
{
    switch (programInterface)
    {
        case GL_PROGRAM_INPUT:
            return program->getInputResourceIndex(name);

        case GL_PROGRAM_OUTPUT:
            return program->getOutputResourceIndex(name);

        case GL_UNIFORM:
            return program->getState().getUniformIndexFromName(name);

        case GL_BUFFER_VARIABLE:
            return program->getState().getBufferVariableIndexFromName(name);

        case GL_SHADER_STORAGE_BLOCK:
            return program->getShaderStorageBlockIndex(name);

        case GL_UNIFORM_BLOCK:
            return program->getUniformBlockIndex(name);

        case GL_TRANSFORM_FEEDBACK_VARYING:
            return program->getTransformFeedbackVaryingResourceIndex(name);

        default:
            UNREACHABLE();
            return GL_INVALID_INDEX;
    }
}

void QueryProgramResourceName(const Program *program,
                              GLenum programInterface,
                              GLuint index,
                              GLsizei bufSize,
                              GLsizei *length,
                              GLchar *name)
{
    switch (programInterface)
    {
        case GL_PROGRAM_INPUT:
            program->getInputResourceName(index, bufSize, length, name);
            break;

        case GL_PROGRAM_OUTPUT:
            program->getOutputResourceName(index, bufSize, length, name);
            break;

        case GL_UNIFORM:
            program->getUniformResourceName(index, bufSize, length, name);
            break;

        case GL_BUFFER_VARIABLE:
            program->getBufferVariableResourceName(index, bufSize, length, name);
            break;

        case GL_SHADER_STORAGE_BLOCK:
            program->getActiveShaderStorageBlockName(index, bufSize, length, name);
            break;

        case GL_UNIFORM_BLOCK:
            program->getActiveUniformBlockName(index, bufSize, length, name);
            break;

        case GL_TRANSFORM_FEEDBACK_VARYING:
            program->getTransformFeedbackVarying(index, bufSize, length, nullptr, nullptr, name);
            break;

        default:
            UNREACHABLE();
    }
}

GLint QueryProgramResourceLocation(const Program *program,
                                   GLenum programInterface,
                                   const GLchar *name)
{
    switch (programInterface)
    {
        case GL_PROGRAM_INPUT:
            return program->getInputResourceLocation(name);

        case GL_PROGRAM_OUTPUT:
            return program->getOutputResourceLocation(name);

        case GL_UNIFORM:
            return program->getUniformLocation(name);

        default:
            UNREACHABLE();
            return -1;
    }
}

void QueryProgramResourceiv(const Program *program,
                            GLenum programInterface,
                            GLuint index,
                            GLsizei propCount,
                            const GLenum *props,
                            GLsizei bufSize,
                            GLsizei *length,
                            GLint *params)
{
    if (!program->isLinked())
    {
        return;
    }

    if (length != nullptr)
    {
        *length = 0;
    }

    if (bufSize == 0)
    {
        // No room to write the results
        return;
    }

    GLsizei pos = 0;
    for (GLsizei i = 0; i < propCount; i++)
    {
        switch (programInterface)
        {
            case GL_PROGRAM_INPUT:
                params[i] = GetInputResourceProperty(program, index, props[i]);
                ++pos;
                break;

            case GL_PROGRAM_OUTPUT:
                params[i] = GetOutputResourceProperty(program, index, props[i]);
                ++pos;
                break;

            case GL_UNIFORM:
                params[i] = GetUniformResourceProperty(program, index, props[i]);
                ++pos;
                break;

            case GL_BUFFER_VARIABLE:
                params[i] = GetBufferVariableResourceProperty(program, index, props[i]);
                ++pos;
                break;

            case GL_UNIFORM_BLOCK:
                GetUniformBlockResourceProperty(program, index, props[i], params, bufSize, &pos);
                break;

            case GL_SHADER_STORAGE_BLOCK:
                GetShaderStorageBlockResourceProperty(program, index, props[i], params, bufSize,
                                                      &pos);
                break;

            case GL_ATOMIC_COUNTER_BUFFER:
                GetAtomicCounterBufferResourceProperty(program, index, props[i], params, bufSize,
                                                       &pos);
                break;

            case GL_TRANSFORM_FEEDBACK_VARYING:
                params[i] = GetTransformFeedbackVaryingResourceProperty(program, index, props[i]);
                ++pos;
                break;

            default:
                UNREACHABLE();
                params[i] = GL_INVALID_VALUE;
        }
        if (pos == bufSize)
        {
            // Most properties return one value, but GL_ACTIVE_VARIABLES returns an array of values.
            // This checks not to break buffer bounds for such case.
            break;
        }
    }

    if (length != nullptr)
    {
        *length = pos;
    }
}

void QueryProgramInterfaceiv(const Program *program,
                             GLenum programInterface,
                             GLenum pname,
                             GLint *params)
{
    switch (pname)
    {
        case GL_ACTIVE_RESOURCES:
            *params = QueryProgramInterfaceActiveResources(program, programInterface);
            break;

        case GL_MAX_NAME_LENGTH:
            *params = QueryProgramInterfaceMaxNameLength(program, programInterface);
            break;

        case GL_MAX_NUM_ACTIVE_VARIABLES:
            *params = QueryProgramInterfaceMaxNumActiveVariables(program, programInterface);
            break;

        default:
            UNREACHABLE();
    }
}

ClientVertexArrayType ParamToVertexArrayType(GLenum param)
{
    switch (param)
    {
        case GL_VERTEX_ARRAY:
        case GL_VERTEX_ARRAY_BUFFER_BINDING:
        case GL_VERTEX_ARRAY_STRIDE:
        case GL_VERTEX_ARRAY_SIZE:
        case GL_VERTEX_ARRAY_TYPE:
        case GL_VERTEX_ARRAY_POINTER:
            return ClientVertexArrayType::Vertex;
        case GL_NORMAL_ARRAY:
        case GL_NORMAL_ARRAY_BUFFER_BINDING:
        case GL_NORMAL_ARRAY_STRIDE:
        case GL_NORMAL_ARRAY_TYPE:
        case GL_NORMAL_ARRAY_POINTER:
            return ClientVertexArrayType::Normal;
        case GL_COLOR_ARRAY:
        case GL_COLOR_ARRAY_BUFFER_BINDING:
        case GL_COLOR_ARRAY_STRIDE:
        case GL_COLOR_ARRAY_SIZE:
        case GL_COLOR_ARRAY_TYPE:
        case GL_COLOR_ARRAY_POINTER:
            return ClientVertexArrayType::Color;
        case GL_POINT_SIZE_ARRAY_OES:
        case GL_POINT_SIZE_ARRAY_BUFFER_BINDING_OES:
        case GL_POINT_SIZE_ARRAY_STRIDE_OES:
        case GL_POINT_SIZE_ARRAY_TYPE_OES:
        case GL_POINT_SIZE_ARRAY_POINTER_OES:
            return ClientVertexArrayType::PointSize;
        case GL_TEXTURE_COORD_ARRAY:
        case GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING:
        case GL_TEXTURE_COORD_ARRAY_STRIDE:
        case GL_TEXTURE_COORD_ARRAY_SIZE:
        case GL_TEXTURE_COORD_ARRAY_TYPE:
        case GL_TEXTURE_COORD_ARRAY_POINTER:
            return ClientVertexArrayType::TextureCoord;
        default:
            UNREACHABLE();
            return ClientVertexArrayType::InvalidEnum;
    }
}

void SetLightModelParameters(GLES1State *state, GLenum pname, const GLfloat *params)
{
    LightModelParameters &lightModel = state->lightModelParameters();

    switch (pname)
    {
        case GL_LIGHT_MODEL_AMBIENT:
            lightModel.color = ColorF::fromData(params);
            break;
        case GL_LIGHT_MODEL_TWO_SIDE:
            lightModel.twoSided = *params == 1.0f ? true : false;
            break;
        default:
            break;
    }
}

void GetLightModelParameters(const GLES1State *state, GLenum pname, GLfloat *params)
{
    const LightModelParameters &lightModel = state->lightModelParameters();

    switch (pname)
    {
        case GL_LIGHT_MODEL_TWO_SIDE:
            *params = lightModel.twoSided ? 1.0f : 0.0f;
            break;
        case GL_LIGHT_MODEL_AMBIENT:
            lightModel.color.writeData(params);
            break;
        default:
            break;
    }
}

bool IsLightModelTwoSided(const GLES1State *state)
{
    return state->lightModelParameters().twoSided;
}

void SetLightParameters(GLES1State *state,
                        GLenum light,
                        LightParameter pname,
                        const GLfloat *params)
{
    uint32_t lightIndex = light - GL_LIGHT0;

    LightParameters &lightParams = state->lightParameters(lightIndex);

    switch (pname)
    {
        case LightParameter::Ambient:
            lightParams.ambient = ColorF::fromData(params);
            break;
        case LightParameter::Diffuse:
            lightParams.diffuse = ColorF::fromData(params);
            break;
        case LightParameter::Specular:
            lightParams.specular = ColorF::fromData(params);
            break;
        case LightParameter::Position:
        {
            angle::Mat4 mv = state->getModelviewMatrix();
            angle::Vector4 transformedPos =
                mv.product(angle::Vector4(params[0], params[1], params[2], params[3]));
            lightParams.position[0] = transformedPos[0];
            lightParams.position[1] = transformedPos[1];
            lightParams.position[2] = transformedPos[2];
            lightParams.position[3] = transformedPos[3];
        }
        break;
        case LightParameter::SpotDirection:
        {
            angle::Mat4 mv = state->getModelviewMatrix();
            angle::Vector4 transformedPos =
                mv.product(angle::Vector4(params[0], params[1], params[2], 0.0f));
            lightParams.direction[0] = transformedPos[0];
            lightParams.direction[1] = transformedPos[1];
            lightParams.direction[2] = transformedPos[2];
        }
        break;
        case LightParameter::SpotExponent:
            lightParams.spotlightExponent = *params;
            break;
        case LightParameter::SpotCutoff:
            lightParams.spotlightCutoffAngle = *params;
            break;
        case LightParameter::ConstantAttenuation:
            lightParams.attenuationConst = *params;
            break;
        case LightParameter::LinearAttenuation:
            lightParams.attenuationLinear = *params;
            break;
        case LightParameter::QuadraticAttenuation:
            lightParams.attenuationQuadratic = *params;
            break;
        default:
            return;
    }
}

void GetLightParameters(const GLES1State *state,
                        GLenum light,
                        LightParameter pname,
                        GLfloat *params)
{
    uint32_t lightIndex                = light - GL_LIGHT0;
    const LightParameters &lightParams = state->lightParameters(lightIndex);

    switch (pname)
    {
        case LightParameter::Ambient:
            lightParams.ambient.writeData(params);
            break;
        case LightParameter::Diffuse:
            lightParams.diffuse.writeData(params);
            break;
        case LightParameter::Specular:
            lightParams.specular.writeData(params);
            break;
        case LightParameter::Position:
            memcpy(params, lightParams.position.data(), 4 * sizeof(GLfloat));
            break;
        case LightParameter::SpotDirection:
            memcpy(params, lightParams.direction.data(), 3 * sizeof(GLfloat));
            break;
        case LightParameter::SpotExponent:
            *params = lightParams.spotlightExponent;
            break;
        case LightParameter::SpotCutoff:
            *params = lightParams.spotlightCutoffAngle;
            break;
        case LightParameter::ConstantAttenuation:
            *params = lightParams.attenuationConst;
            break;
        case LightParameter::LinearAttenuation:
            *params = lightParams.attenuationLinear;
            break;
        case LightParameter::QuadraticAttenuation:
            *params = lightParams.attenuationQuadratic;
            break;
        default:
            break;
    }
}

void SetMaterialParameters(GLES1State *state,
                           GLenum face,
                           MaterialParameter pname,
                           const GLfloat *params)
{
    MaterialParameters &material = state->materialParameters();
    switch (pname)
    {
        case MaterialParameter::Ambient:
            material.ambient = ColorF::fromData(params);
            break;
        case MaterialParameter::Diffuse:
            material.diffuse = ColorF::fromData(params);
            break;
        case MaterialParameter::AmbientAndDiffuse:
            material.ambient = ColorF::fromData(params);
            material.diffuse = ColorF::fromData(params);
            break;
        case MaterialParameter::Specular:
            material.specular = ColorF::fromData(params);
            break;
        case MaterialParameter::Emission:
            material.emissive = ColorF::fromData(params);
            break;
        case MaterialParameter::Shininess:
            material.specularExponent = *params;
            break;
        default:
            return;
    }
}

void GetMaterialParameters(const GLES1State *state,
                           GLenum face,
                           MaterialParameter pname,
                           GLfloat *params)
{
    const ColorF &currentColor         = state->getCurrentColor();
    const MaterialParameters &material = state->materialParameters();
    const bool colorMaterialEnabled    = state->isColorMaterialEnabled();

    switch (pname)
    {
        case MaterialParameter::Ambient:
            if (colorMaterialEnabled)
            {
                currentColor.writeData(params);
            }
            else
            {
                material.ambient.writeData(params);
            }
            break;
        case MaterialParameter::Diffuse:
            if (colorMaterialEnabled)
            {
                currentColor.writeData(params);
            }
            else
            {
                material.diffuse.writeData(params);
            }
            break;
        case MaterialParameter::Specular:
            material.specular.writeData(params);
            break;
        case MaterialParameter::Emission:
            material.emissive.writeData(params);
            break;
        case MaterialParameter::Shininess:
            *params = material.specularExponent;
            break;
        default:
            return;
    }
}

unsigned int GetLightModelParameterCount(GLenum pname)
{
    switch (pname)
    {
        case GL_LIGHT_MODEL_AMBIENT:
            return 4;
        case GL_LIGHT_MODEL_TWO_SIDE:
            return 1;
        default:
            return 0;
    }
}

unsigned int GetLightParameterCount(LightParameter pname)
{
    switch (pname)
    {
        case LightParameter::Ambient:
        case LightParameter::Diffuse:
        case LightParameter::Specular:
        case LightParameter::Position:
            return 4;
        case LightParameter::SpotDirection:
            return 3;
        case LightParameter::SpotExponent:
        case LightParameter::SpotCutoff:
        case LightParameter::ConstantAttenuation:
        case LightParameter::LinearAttenuation:
        case LightParameter::QuadraticAttenuation:
            return 1;
        default:
            return 0;
    }
}

unsigned int GetMaterialParameterCount(MaterialParameter pname)
{
    switch (pname)
    {
        case MaterialParameter::Ambient:
        case MaterialParameter::Diffuse:
        case MaterialParameter::Specular:
        case MaterialParameter::Emission:
            return 4;
        case MaterialParameter::Shininess:
            return 1;
        default:
            return 0;
    }
}

void SetFogParameters(GLES1State *state, GLenum pname, const GLfloat *params)
{
    FogParameters &fog = state->fogParameters();
    switch (pname)
    {
        case GL_FOG_MODE:
            fog.mode = FromGLenum<FogMode>(static_cast<GLenum>(params[0]));
            break;
        case GL_FOG_DENSITY:
            fog.density = params[0];
            break;
        case GL_FOG_START:
            fog.start = params[0];
            break;
        case GL_FOG_END:
            fog.end = params[0];
            break;
        case GL_FOG_COLOR:
            fog.color = ColorF::fromData(params);
            break;
        default:
            return;
    }
}

void GetFogParameters(const GLES1State *state, GLenum pname, GLfloat *params)
{
    const FogParameters &fog = state->fogParameters();
    switch (pname)
    {
        case GL_FOG_MODE:
            params[0] = static_cast<GLfloat>(ToGLenum(fog.mode));
            break;
        case GL_FOG_DENSITY:
            params[0] = fog.density;
            break;
        case GL_FOG_START:
            params[0] = fog.start;
            break;
        case GL_FOG_END:
            params[0] = fog.end;
            break;
        case GL_FOG_COLOR:
            fog.color.writeData(params);
            break;
        default:
            return;
    }
}

unsigned int GetFogParameterCount(GLenum pname)
{
    switch (pname)
    {
        case GL_FOG_MODE:
        case GL_FOG_DENSITY:
        case GL_FOG_START:
        case GL_FOG_END:
            return 1;
        case GL_FOG_COLOR:
            return 4;
        default:
            return 0;
    }
}

unsigned int GetTextureEnvParameterCount(TextureEnvParameter pname)
{
    switch (pname)
    {
        case TextureEnvParameter::Mode:
        case TextureEnvParameter::CombineRgb:
        case TextureEnvParameter::CombineAlpha:
        case TextureEnvParameter::Src0Rgb:
        case TextureEnvParameter::Src1Rgb:
        case TextureEnvParameter::Src2Rgb:
        case TextureEnvParameter::Src0Alpha:
        case TextureEnvParameter::Src1Alpha:
        case TextureEnvParameter::Src2Alpha:
        case TextureEnvParameter::Op0Rgb:
        case TextureEnvParameter::Op1Rgb:
        case TextureEnvParameter::Op2Rgb:
        case TextureEnvParameter::Op0Alpha:
        case TextureEnvParameter::Op1Alpha:
        case TextureEnvParameter::Op2Alpha:
        case TextureEnvParameter::RgbScale:
        case TextureEnvParameter::AlphaScale:
        case TextureEnvParameter::PointCoordReplace:
            return 1;
        case TextureEnvParameter::Color:
            return 4;
        default:
            return 0;
    }
}

void ConvertTextureEnvFromInt(TextureEnvParameter pname, const GLint *input, GLfloat *output)
{
    if (IsTextureEnvEnumParameter(pname))
    {
        ConvertGLenumValue(input[0], output);
        return;
    }

    switch (pname)
    {
        case TextureEnvParameter::RgbScale:
        case TextureEnvParameter::AlphaScale:
            output[0] = static_cast<GLfloat>(input[0]);
            break;
        case TextureEnvParameter::Color:
            for (int i = 0; i < 4; i++)
            {
                output[i] = input[i] / 255.0f;
            }
            break;
        default:
            UNREACHABLE();
            break;
    }
}

void ConvertTextureEnvFromFixed(TextureEnvParameter pname, const GLfixed *input, GLfloat *output)
{
    if (IsTextureEnvEnumParameter(pname))
    {
        ConvertGLenumValue(input[0], output);
        return;
    }

    switch (pname)
    {
        case TextureEnvParameter::RgbScale:
        case TextureEnvParameter::AlphaScale:
            output[0] = ConvertFixedToFloat(input[0]);
            break;
        case TextureEnvParameter::Color:
            for (int i = 0; i < 4; i++)
            {
                output[i] = ConvertFixedToFloat(input[i]);
            }
            break;
        default:
            UNREACHABLE();
            break;
    }
}

void ConvertTextureEnvToInt(TextureEnvParameter pname, const GLfloat *input, GLint *output)
{
    if (IsTextureEnvEnumParameter(pname))
    {
        ConvertGLenumValue(input[0], output);
        return;
    }

    switch (pname)
    {
        case TextureEnvParameter::RgbScale:
        case TextureEnvParameter::AlphaScale:
            output[0] = static_cast<GLint>(input[0]);
            break;
        case TextureEnvParameter::Color:
            for (int i = 0; i < 4; i++)
            {
                output[i] = static_cast<GLint>(input[i] * 255.0f);
            }
            break;
        default:
            UNREACHABLE();
            break;
    }
}

void ConvertTextureEnvToFixed(TextureEnvParameter pname, const GLfloat *input, GLfixed *output)
{
    if (IsTextureEnvEnumParameter(pname))
    {
        ConvertGLenumValue(input[0], output);
        return;
    }

    switch (pname)
    {
        case TextureEnvParameter::RgbScale:
        case TextureEnvParameter::AlphaScale:
            output[0] = ConvertFloatToFixed(input[0]);
            break;
        case TextureEnvParameter::Color:
            for (int i = 0; i < 4; i++)
            {
                output[i] = ConvertFloatToFixed(input[i]);
            }
            break;
        default:
            UNREACHABLE();
            break;
    }
}

void SetTextureEnv(unsigned int unit,
                   GLES1State *state,
                   TextureEnvTarget target,
                   TextureEnvParameter pname,
                   const GLfloat *params)
{
    TextureEnvironmentParameters &env = state->textureEnvironment(unit);
    GLenum asEnum                     = ConvertToGLenum(params[0]);

    switch (target)
    {
        case TextureEnvTarget::Env:
            switch (pname)
            {
                case TextureEnvParameter::Mode:
                    env.mode = FromGLenum<TextureEnvMode>(asEnum);
                    break;
                case TextureEnvParameter::CombineRgb:
                    env.combineRgb = FromGLenum<TextureCombine>(asEnum);
                    break;
                case TextureEnvParameter::CombineAlpha:
                    env.combineAlpha = FromGLenum<TextureCombine>(asEnum);
                    break;
                case TextureEnvParameter::Src0Rgb:
                    env.src0Rgb = FromGLenum<TextureSrc>(asEnum);
                    break;
                case TextureEnvParameter::Src1Rgb:
                    env.src1Rgb = FromGLenum<TextureSrc>(asEnum);
                    break;
                case TextureEnvParameter::Src2Rgb:
                    env.src2Rgb = FromGLenum<TextureSrc>(asEnum);
                    break;
                case TextureEnvParameter::Src0Alpha:
                    env.src0Alpha = FromGLenum<TextureSrc>(asEnum);
                    break;
                case TextureEnvParameter::Src1Alpha:
                    env.src1Alpha = FromGLenum<TextureSrc>(asEnum);
                    break;
                case TextureEnvParameter::Src2Alpha:
                    env.src2Alpha = FromGLenum<TextureSrc>(asEnum);
                    break;
                case TextureEnvParameter::Op0Rgb:
                    env.op0Rgb = FromGLenum<TextureOp>(asEnum);
                    break;
                case TextureEnvParameter::Op1Rgb:
                    env.op1Rgb = FromGLenum<TextureOp>(asEnum);
                    break;
                case TextureEnvParameter::Op2Rgb:
                    env.op2Rgb = FromGLenum<TextureOp>(asEnum);
                    break;
                case TextureEnvParameter::Op0Alpha:
                    env.op0Alpha = FromGLenum<TextureOp>(asEnum);
                    break;
                case TextureEnvParameter::Op1Alpha:
                    env.op1Alpha = FromGLenum<TextureOp>(asEnum);
                    break;
                case TextureEnvParameter::Op2Alpha:
                    env.op2Alpha = FromGLenum<TextureOp>(asEnum);
                    break;
                case TextureEnvParameter::Color:
                    env.color = ColorF::fromData(params);
                    break;
                case TextureEnvParameter::RgbScale:
                    env.rgbScale = params[0];
                    break;
                case TextureEnvParameter::AlphaScale:
                    env.alphaScale = params[0];
                    break;
                default:
                    UNREACHABLE();
                    break;
            }
            break;
        case TextureEnvTarget::PointSprite:
            switch (pname)
            {
                case TextureEnvParameter::PointCoordReplace:
                    env.pointSpriteCoordReplace = static_cast<bool>(params[0]);
                    break;
                default:
                    UNREACHABLE();
                    break;
            }
            break;
        default:
            UNREACHABLE();
            break;
    }
}

void GetTextureEnv(unsigned int unit,
                   const GLES1State *state,
                   TextureEnvTarget target,
                   TextureEnvParameter pname,
                   GLfloat *params)
{
    const TextureEnvironmentParameters &env = state->textureEnvironment(unit);

    switch (target)
    {
        case TextureEnvTarget::Env:
            switch (pname)
            {
                case TextureEnvParameter::Mode:
                    ConvertPackedEnum(env.mode, params);
                    break;
                case TextureEnvParameter::CombineRgb:
                    ConvertPackedEnum(env.combineRgb, params);
                    break;
                case TextureEnvParameter::CombineAlpha:
                    ConvertPackedEnum(env.combineAlpha, params);
                    break;
                case TextureEnvParameter::Src0Rgb:
                    ConvertPackedEnum(env.src0Rgb, params);
                    break;
                case TextureEnvParameter::Src1Rgb:
                    ConvertPackedEnum(env.src1Rgb, params);
                    break;
                case TextureEnvParameter::Src2Rgb:
                    ConvertPackedEnum(env.src2Rgb, params);
                    break;
                case TextureEnvParameter::Src0Alpha:
                    ConvertPackedEnum(env.src0Alpha, params);
                    break;
                case TextureEnvParameter::Src1Alpha:
                    ConvertPackedEnum(env.src1Alpha, params);
                    break;
                case TextureEnvParameter::Src2Alpha:
                    ConvertPackedEnum(env.src2Alpha, params);
                    break;
                case TextureEnvParameter::Op0Rgb:
                    ConvertPackedEnum(env.op0Rgb, params);
                    break;
                case TextureEnvParameter::Op1Rgb:
                    ConvertPackedEnum(env.op1Rgb, params);
                    break;
                case TextureEnvParameter::Op2Rgb:
                    ConvertPackedEnum(env.op2Rgb, params);
                    break;
                case TextureEnvParameter::Op0Alpha:
                    ConvertPackedEnum(env.op0Alpha, params);
                    break;
                case TextureEnvParameter::Op1Alpha:
                    ConvertPackedEnum(env.op1Alpha, params);
                    break;
                case TextureEnvParameter::Op2Alpha:
                    ConvertPackedEnum(env.op2Alpha, params);
                    break;
                case TextureEnvParameter::Color:
                    env.color.writeData(params);
                    break;
                case TextureEnvParameter::RgbScale:
                    *params = env.rgbScale;
                    break;
                case TextureEnvParameter::AlphaScale:
                    *params = env.alphaScale;
                    break;
                default:
                    UNREACHABLE();
                    break;
            }
            break;
        case TextureEnvTarget::PointSprite:
            switch (pname)
            {
                case TextureEnvParameter::PointCoordReplace:
                    *params = static_cast<GLfloat>(env.pointSpriteCoordReplace);
                    break;
                default:
                    UNREACHABLE();
                    break;
            }
            break;
        default:
            UNREACHABLE();
            break;
    }
}

unsigned int GetPointParameterCount(PointParameter pname)
{
    switch (pname)
    {
        case PointParameter::PointSizeMin:
        case PointParameter::PointSizeMax:
        case PointParameter::PointFadeThresholdSize:
            return 1;
        case PointParameter::PointDistanceAttenuation:
            return 3;
        default:
            return 0;
    }
}

void SetPointParameter(GLES1State *state, PointParameter pname, const GLfloat *params)
{

    PointParameters &pointParams = state->pointParameters();

    switch (pname)
    {
        case PointParameter::PointSizeMin:
            pointParams.pointSizeMin = params[0];
            break;
        case PointParameter::PointSizeMax:
            pointParams.pointSizeMax = params[0];
            break;
        case PointParameter::PointFadeThresholdSize:
            pointParams.pointFadeThresholdSize = params[0];
            break;
        case PointParameter::PointDistanceAttenuation:
            for (unsigned int i = 0; i < 3; i++)
            {
                pointParams.pointDistanceAttenuation[i] = params[i];
            }
            break;
        default:
            UNREACHABLE();
    }
}

void GetPointParameter(const GLES1State *state, PointParameter pname, GLfloat *params)
{
    const PointParameters &pointParams = state->pointParameters();

    switch (pname)
    {
        case PointParameter::PointSizeMin:
            params[0] = pointParams.pointSizeMin;
            break;
        case PointParameter::PointSizeMax:
            params[0] = pointParams.pointSizeMax;
            break;
        case PointParameter::PointFadeThresholdSize:
            params[0] = pointParams.pointFadeThresholdSize;
            break;
        case PointParameter::PointDistanceAttenuation:
            for (unsigned int i = 0; i < 3; i++)
            {
                params[i] = pointParams.pointDistanceAttenuation[i];
            }
            break;
        default:
            UNREACHABLE();
    }
}

void SetPointSize(GLES1State *state, GLfloat size)
{
    PointParameters &params = state->pointParameters();
    params.pointSize        = size;
}

void GetPointSize(GLES1State *state, GLfloat *sizeOut)
{
    const PointParameters &params = state->pointParameters();
    *sizeOut                      = params.pointSize;
}

unsigned int GetTexParameterCount(GLenum pname)
{
    switch (pname)
    {
        case GL_TEXTURE_CROP_RECT_OES:
        case GL_TEXTURE_BORDER_COLOR:
            return 4;
        case GL_TEXTURE_MAG_FILTER:
        case GL_TEXTURE_MIN_FILTER:
        case GL_TEXTURE_WRAP_S:
        case GL_TEXTURE_WRAP_T:
        case GL_TEXTURE_USAGE_ANGLE:
        case GL_TEXTURE_MAX_ANISOTROPY_EXT:
        case GL_TEXTURE_IMMUTABLE_FORMAT:
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
        case GL_TEXTURE_SRGB_DECODE_EXT:
        case GL_DEPTH_STENCIL_TEXTURE_MODE:
        case GL_TEXTURE_NATIVE_ID_ANGLE:
            return 1;
        default:
            return 0;
    }
}

}  // namespace gl

namespace egl
{

void QueryConfigAttrib(const Config *config, EGLint attribute, EGLint *value)
{
    ASSERT(config != nullptr);
    switch (attribute)
    {
        case EGL_BUFFER_SIZE:
            *value = config->bufferSize;
            break;
        case EGL_ALPHA_SIZE:
            *value = config->alphaSize;
            break;
        case EGL_BLUE_SIZE:
            *value = config->blueSize;
            break;
        case EGL_GREEN_SIZE:
            *value = config->greenSize;
            break;
        case EGL_RED_SIZE:
            *value = config->redSize;
            break;
        case EGL_DEPTH_SIZE:
            *value = config->depthSize;
            break;
        case EGL_STENCIL_SIZE:
            *value = config->stencilSize;
            break;
        case EGL_CONFIG_CAVEAT:
            *value = config->configCaveat;
            break;
        case EGL_CONFIG_ID:
            *value = config->configID;
            break;
        case EGL_LEVEL:
            *value = config->level;
            break;
        case EGL_NATIVE_RENDERABLE:
            *value = config->nativeRenderable;
            break;
        case EGL_NATIVE_VISUAL_ID:
            *value = config->nativeVisualID;
            break;
        case EGL_NATIVE_VISUAL_TYPE:
            *value = config->nativeVisualType;
            break;
        case EGL_SAMPLES:
            *value = config->samples;
            break;
        case EGL_SAMPLE_BUFFERS:
            *value = config->sampleBuffers;
            break;
        case EGL_SURFACE_TYPE:
            *value = config->surfaceType;
            break;
        case EGL_TRANSPARENT_TYPE:
            *value = config->transparentType;
            break;
        case EGL_TRANSPARENT_BLUE_VALUE:
            *value = config->transparentBlueValue;
            break;
        case EGL_TRANSPARENT_GREEN_VALUE:
            *value = config->transparentGreenValue;
            break;
        case EGL_TRANSPARENT_RED_VALUE:
            *value = config->transparentRedValue;
            break;
        case EGL_BIND_TO_TEXTURE_RGB:
            *value = config->bindToTextureRGB;
            break;
        case EGL_BIND_TO_TEXTURE_RGBA:
            *value = config->bindToTextureRGBA;
            break;
        case EGL_MIN_SWAP_INTERVAL:
            *value = config->minSwapInterval;
            break;
        case EGL_MAX_SWAP_INTERVAL:
            *value = config->maxSwapInterval;
            break;
        case EGL_LUMINANCE_SIZE:
            *value = config->luminanceSize;
            break;
        case EGL_ALPHA_MASK_SIZE:
            *value = config->alphaMaskSize;
            break;
        case EGL_COLOR_BUFFER_TYPE:
            *value = config->colorBufferType;
            break;
        case EGL_RENDERABLE_TYPE:
            *value = config->renderableType;
            break;
        case EGL_MATCH_NATIVE_PIXMAP:
            *value = false;
            UNIMPLEMENTED();
            break;
        case EGL_CONFORMANT:
            *value = config->conformant;
            break;
        case EGL_MAX_PBUFFER_WIDTH:
            *value = config->maxPBufferWidth;
            break;
        case EGL_MAX_PBUFFER_HEIGHT:
            *value = config->maxPBufferHeight;
            break;
        case EGL_MAX_PBUFFER_PIXELS:
            *value = config->maxPBufferPixels;
            break;
        case EGL_OPTIMAL_SURFACE_ORIENTATION_ANGLE:
            *value = config->optimalOrientation;
            break;
        case EGL_COLOR_COMPONENT_TYPE_EXT:
            *value = config->colorComponentType;
            break;
        case EGL_RECORDABLE_ANDROID:
            *value = config->recordable;
            break;
        default:
            UNREACHABLE();
            break;
    }
}

void QueryContextAttrib(const gl::Context *context, EGLint attribute, EGLint *value)
{
    switch (attribute)
    {
        case EGL_CONFIG_ID:
            if (context->getConfig() != EGL_NO_CONFIG_KHR)
            {
                *value = context->getConfig()->configID;
            }
            else
            {
                *value = 0;
            }
            break;
        case EGL_CONTEXT_CLIENT_TYPE:
            *value = context->getClientType();
            break;
        case EGL_CONTEXT_CLIENT_VERSION:
            *value = context->getClientMajorVersion();
            break;
        case EGL_RENDER_BUFFER:
            *value = context->getRenderBuffer();
            break;
        case EGL_ROBUST_RESOURCE_INITIALIZATION_ANGLE:
            *value = context->isRobustResourceInitEnabled();
            break;
        default:
            UNREACHABLE();
            break;
    }
}

void QuerySurfaceAttrib(const Surface *surface, EGLint attribute, EGLint *value)
{
    switch (attribute)
    {
        case EGL_GL_COLORSPACE:
            *value = surface->getGLColorspace();
            break;
        case EGL_VG_ALPHA_FORMAT:
            *value = surface->getVGAlphaFormat();
            break;
        case EGL_VG_COLORSPACE:
            *value = surface->getVGColorspace();
            break;
        case EGL_CONFIG_ID:
            *value = surface->getConfig()->configID;
            break;
        case EGL_HEIGHT:
            *value = surface->getHeight();
            break;
        case EGL_HORIZONTAL_RESOLUTION:
            *value = surface->getHorizontalResolution();
            break;
        case EGL_LARGEST_PBUFFER:
            // The EGL spec states that value is not written if the surface is not a pbuffer
            if (surface->getType() == EGL_PBUFFER_BIT)
            {
                *value = surface->getLargestPbuffer();
            }
            break;
        case EGL_MIPMAP_TEXTURE:
            // The EGL spec states that value is not written if the surface is not a pbuffer
            if (surface->getType() == EGL_PBUFFER_BIT)
            {
                *value = surface->getMipmapTexture();
            }
            break;
        case EGL_MIPMAP_LEVEL:
            // The EGL spec states that value is not written if the surface is not a pbuffer
            if (surface->getType() == EGL_PBUFFER_BIT)
            {
                *value = surface->getMipmapLevel();
            }
            break;
        case EGL_MULTISAMPLE_RESOLVE:
            *value = surface->getMultisampleResolve();
            break;
        case EGL_PIXEL_ASPECT_RATIO:
            *value = surface->getPixelAspectRatio();
            break;
        case EGL_RENDER_BUFFER:
            *value = surface->getRenderBuffer();
            break;
        case EGL_SWAP_BEHAVIOR:
            *value = surface->getSwapBehavior();
            break;
        case EGL_TEXTURE_FORMAT:
            // The EGL spec states that value is not written if the surface is not a pbuffer
            if (surface->getType() == EGL_PBUFFER_BIT)
            {
                *value = ToEGLenum(surface->getTextureFormat());
            }
            break;
        case EGL_TEXTURE_TARGET:
            // The EGL spec states that value is not written if the surface is not a pbuffer
            if (surface->getType() == EGL_PBUFFER_BIT)
            {
                *value = surface->getTextureTarget();
            }
            break;
        case EGL_VERTICAL_RESOLUTION:
            *value = surface->getVerticalResolution();
            break;
        case EGL_WIDTH:
            *value = surface->getWidth();
            break;
        case EGL_POST_SUB_BUFFER_SUPPORTED_NV:
            *value = surface->isPostSubBufferSupported();
            break;
        case EGL_FIXED_SIZE_ANGLE:
            *value = surface->isFixedSize();
            break;
        case EGL_FLEXIBLE_SURFACE_COMPATIBILITY_SUPPORTED_ANGLE:
            *value = surface->flexibleSurfaceCompatibilityRequested();
            break;
        case EGL_SURFACE_ORIENTATION_ANGLE:
            *value = surface->getOrientation();
            break;
        case EGL_DIRECT_COMPOSITION_ANGLE:
            *value = surface->directComposition();
            break;
        case EGL_ROBUST_RESOURCE_INITIALIZATION_ANGLE:
            *value = surface->isRobustResourceInitEnabled();
            break;
        case EGL_TIMESTAMPS_ANDROID:
            *value = surface->isTimestampsEnabled();
            break;
        default:
            UNREACHABLE();
            break;
    }
}

void SetSurfaceAttrib(Surface *surface, EGLint attribute, EGLint value)
{
    switch (attribute)
    {
        case EGL_MIPMAP_LEVEL:
            surface->setMipmapLevel(value);
            break;
        case EGL_MULTISAMPLE_RESOLVE:
            surface->setMultisampleResolve(value);
            break;
        case EGL_SWAP_BEHAVIOR:
            surface->setSwapBehavior(value);
            break;
        case EGL_WIDTH:
            surface->setFixedWidth(value);
            break;
        case EGL_HEIGHT:
            surface->setFixedHeight(value);
            break;
        case EGL_TIMESTAMPS_ANDROID:
            surface->setTimestampsEnabled(value != EGL_FALSE);
            break;
        default:
            UNREACHABLE();
            break;
    }
}

Error GetSyncAttrib(Display *display, Sync *sync, EGLint attribute, EGLint *value)
{
    switch (attribute)
    {
        case EGL_SYNC_TYPE_KHR:
            *value = sync->getType();
            return NoError();

        case EGL_SYNC_STATUS_KHR:
            return sync->getStatus(display, value);

        case EGL_SYNC_CONDITION_KHR:
            *value = sync->getCondition();
            return NoError();

        default:
            break;
    }

    UNREACHABLE();
    return NoError();
}

}  // namespace egl
