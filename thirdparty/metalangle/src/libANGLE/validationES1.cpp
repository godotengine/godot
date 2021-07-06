//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// validationES1.cpp: Validation functions for OpenGL ES 1.0 entry point parameters

#include "libANGLE/validationES1_autogen.h"

#include "common/debug.h"
#include "libANGLE/Context.h"
#include "libANGLE/ErrorStrings.h"
#include "libANGLE/GLES1State.h"
#include "libANGLE/queryconversions.h"
#include "libANGLE/queryutils.h"
#include "libANGLE/validationES.h"

#define ANGLE_VALIDATE_IS_GLES1(context)                                                        \
    do                                                                                          \
    {                                                                                           \
        if (context->getClientType() != EGL_OPENGL_API && context->getClientMajorVersion() > 1) \
        {                                                                                       \
            context->validationError(GL_INVALID_OPERATION, kGLES1Only);                         \
            return false;                                                                       \
        }                                                                                       \
    } while (0)

namespace gl
{
using namespace err;

bool ValidateAlphaFuncCommon(Context *context, AlphaTestFunc func)
{
    switch (func)
    {
        case AlphaTestFunc::AlwaysPass:
        case AlphaTestFunc::Equal:
        case AlphaTestFunc::Gequal:
        case AlphaTestFunc::Greater:
        case AlphaTestFunc::Lequal:
        case AlphaTestFunc::Less:
        case AlphaTestFunc::Never:
        case AlphaTestFunc::NotEqual:
            return true;
        default:
            context->validationError(GL_INVALID_ENUM, kEnumNotSupported);
            return false;
    }
}

bool ValidateClientStateCommon(Context *context, ClientVertexArrayType arrayType)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    switch (arrayType)
    {
        case ClientVertexArrayType::Vertex:
        case ClientVertexArrayType::Normal:
        case ClientVertexArrayType::Color:
        case ClientVertexArrayType::TextureCoord:
            return true;
        case ClientVertexArrayType::PointSize:
            if (!context->getExtensions().pointSizeArray)
            {
                context->validationError(GL_INVALID_ENUM, kPointSizeArrayExtensionNotEnabled);
                return false;
            }
            return true;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidClientState);
            return false;
    }
}

bool ValidateBuiltinVertexAttributeCommon(Context *context,
                                          ClientVertexArrayType arrayType,
                                          GLint size,
                                          VertexAttribType type,
                                          GLsizei stride,
                                          const void *pointer)
{
    ANGLE_VALIDATE_IS_GLES1(context);

    if (stride < 0)
    {
        context->validationError(GL_INVALID_VALUE, kInvalidVertexPointerStride);
        return false;
    }

    int minSize = 1;
    int maxSize = 4;

    switch (arrayType)
    {
        case ClientVertexArrayType::Vertex:
        case ClientVertexArrayType::TextureCoord:
            minSize = 2;
            maxSize = 4;
            break;
        case ClientVertexArrayType::Normal:
            minSize = 3;
            maxSize = 3;
            break;
        case ClientVertexArrayType::Color:
            minSize = 4;
            maxSize = 4;
            break;
        case ClientVertexArrayType::PointSize:
            if (!context->getExtensions().pointSizeArray)
            {
                context->validationError(GL_INVALID_ENUM, kPointSizeArrayExtensionNotEnabled);
                return false;
            }

            minSize = 1;
            maxSize = 1;
            break;
        default:
            UNREACHABLE();
            return false;
    }

    if (size < minSize || size > maxSize)
    {
        context->validationError(GL_INVALID_VALUE, kInvalidVertexPointerSize);
        return false;
    }

    switch (type)
    {
        case VertexAttribType::Byte:
        case VertexAttribType::UnsignedByte:
            if (arrayType == ClientVertexArrayType::PointSize)
            {
                context->validationError(GL_INVALID_ENUM, kInvalidVertexPointerType);
                return false;
            }
            break;
        case VertexAttribType::Short:
        case VertexAttribType::UnsignedShort:
            if (arrayType == ClientVertexArrayType::PointSize ||
                arrayType == ClientVertexArrayType::Color)
            {
                context->validationError(GL_INVALID_ENUM, kInvalidVertexPointerType);
                return false;
            }
            break;
        case VertexAttribType::Fixed:
        case VertexAttribType::Float:
            break;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidVertexPointerType);
            return false;
    }

    return true;
}

bool ValidateLightCaps(Context *context, GLenum light)
{
    if (light < GL_LIGHT0 || light >= GL_LIGHT0 + context->getCaps().maxLights)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidLight);
        return false;
    }

    return true;
}

bool ValidateLightCommon(Context *context,
                         GLenum light,
                         LightParameter pname,
                         const GLfloat *params)
{

    ANGLE_VALIDATE_IS_GLES1(context);

    if (!ValidateLightCaps(context, light))
    {
        return false;
    }

    switch (pname)
    {
        case LightParameter::Ambient:
        case LightParameter::Diffuse:
        case LightParameter::Specular:
        case LightParameter::Position:
        case LightParameter::SpotDirection:
            return true;
        case LightParameter::SpotExponent:
            if (params[0] < 0.0f || params[0] > 128.0f)
            {
                context->validationError(GL_INVALID_VALUE, kLightParameterOutOfRange);
                return false;
            }
            return true;
        case LightParameter::SpotCutoff:
            if (params[0] == 180.0f)
            {
                return true;
            }
            if (params[0] < 0.0f || params[0] > 90.0f)
            {
                context->validationError(GL_INVALID_VALUE, kLightParameterOutOfRange);
                return false;
            }
            return true;
        case LightParameter::ConstantAttenuation:
        case LightParameter::LinearAttenuation:
        case LightParameter::QuadraticAttenuation:
            if (params[0] < 0.0f)
            {
                context->validationError(GL_INVALID_VALUE, kLightParameterOutOfRange);
                return false;
            }
            return true;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidLightParameter);
            return false;
    }
}

bool ValidateLightSingleComponent(Context *context,
                                  GLenum light,
                                  LightParameter pname,
                                  GLfloat param)
{
    if (!ValidateLightCommon(context, light, pname, &param))
    {
        return false;
    }

    if (GetLightParameterCount(pname) > 1)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidLightParameter);
        return false;
    }

    return true;
}

bool ValidateMaterialCommon(Context *context,
                            GLenum face,
                            MaterialParameter pname,
                            const GLfloat *params)
{
    switch (pname)
    {
        case MaterialParameter::Ambient:
        case MaterialParameter::Diffuse:
        case MaterialParameter::Specular:
        case MaterialParameter::Emission:
            return true;
        case MaterialParameter::Shininess:
            if (params[0] < 0.0f || params[0] > 128.0f)
            {
                context->validationError(GL_INVALID_VALUE, kMaterialParameterOutOfRange);
                return false;
            }
            return true;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidMaterialParameter);
            return false;
    }
}

bool ValidateMaterialSetting(Context *context,
                             GLenum face,
                             MaterialParameter pname,
                             const GLfloat *params)
{
    ANGLE_VALIDATE_IS_GLES1(context);

    if (face != GL_FRONT_AND_BACK)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidMaterialFace);
        return false;
    }

    return ValidateMaterialCommon(context, face, pname, params);
}

bool ValidateMaterialQuery(Context *context, GLenum face, MaterialParameter pname)
{
    ANGLE_VALIDATE_IS_GLES1(context);

    if (face != GL_FRONT && face != GL_BACK)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidMaterialFace);
        return false;
    }

    GLfloat dummyParams[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    return ValidateMaterialCommon(context, face, pname, dummyParams);
}

bool ValidateMaterialSingleComponent(Context *context,
                                     GLenum face,
                                     MaterialParameter pname,
                                     GLfloat param)
{
    if (!ValidateMaterialSetting(context, face, pname, &param))
    {
        return false;
    }

    if (GetMaterialParameterCount(pname) > 1)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidMaterialParameter);
        return false;
    }

    return true;
}

bool ValidateLightModelCommon(Context *context, GLenum pname)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    switch (pname)
    {
        case GL_LIGHT_MODEL_AMBIENT:
        case GL_LIGHT_MODEL_TWO_SIDE:
            return true;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidLightModelParameter);
            return false;
    }
}

bool ValidateLightModelSingleComponent(Context *context, GLenum pname)
{
    if (!ValidateLightModelCommon(context, pname))
    {
        return false;
    }

    switch (pname)
    {
        case GL_LIGHT_MODEL_TWO_SIDE:
            return true;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidLightModelParameter);
            return false;
    }
}

bool ValidateClipPlaneCommon(Context *context, GLenum plane)
{
    ANGLE_VALIDATE_IS_GLES1(context);

    if (plane < GL_CLIP_PLANE0 || plane >= GL_CLIP_PLANE0 + context->getCaps().maxClipPlanes)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidClipPlane);
        return false;
    }

    return true;
}

bool ValidateFogCommon(Context *context, GLenum pname, const GLfloat *params)
{
    ANGLE_VALIDATE_IS_GLES1(context);

    switch (pname)
    {
        case GL_FOG_MODE:
        {
            GLenum modeParam = static_cast<GLenum>(params[0]);
            switch (modeParam)
            {
                case GL_EXP:
                case GL_EXP2:
                case GL_LINEAR:
                    return true;
                default:
                    context->validationError(GL_INVALID_VALUE, kInvalidFogMode);
                    return false;
            }
        }
        break;
        case GL_FOG_START:
        case GL_FOG_END:
        case GL_FOG_COLOR:
            break;
        case GL_FOG_DENSITY:
            if (params[0] < 0.0f)
            {
                context->validationError(GL_INVALID_VALUE, kInvalidFogDensity);
                return false;
            }
            break;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidFogParameter);
            return false;
    }
    return true;
}

bool ValidateTexEnvCommon(Context *context,
                          TextureEnvTarget target,
                          TextureEnvParameter pname,
                          const GLfloat *params)
{
    ANGLE_VALIDATE_IS_GLES1(context);

    switch (target)
    {
        case TextureEnvTarget::Env:
            switch (pname)
            {
                case TextureEnvParameter::Mode:
                {
                    TextureEnvMode mode = FromGLenum<TextureEnvMode>(ConvertToGLenum(params[0]));
                    switch (mode)
                    {
                        case TextureEnvMode::Add:
                        case TextureEnvMode::Blend:
                        case TextureEnvMode::Combine:
                        case TextureEnvMode::Decal:
                        case TextureEnvMode::Modulate:
                        case TextureEnvMode::Replace:
                            break;
                        default:
                            context->validationError(GL_INVALID_VALUE, kInvalidTextureEnvMode);
                            return false;
                    }
                    break;
                }
                case TextureEnvParameter::CombineRgb:
                case TextureEnvParameter::CombineAlpha:
                {
                    TextureCombine combine = FromGLenum<TextureCombine>(ConvertToGLenum(params[0]));
                    switch (combine)
                    {
                        case TextureCombine::Add:
                        case TextureCombine::AddSigned:
                        case TextureCombine::Interpolate:
                        case TextureCombine::Modulate:
                        case TextureCombine::Replace:
                        case TextureCombine::Subtract:
                            break;
                        case TextureCombine::Dot3Rgb:
                        case TextureCombine::Dot3Rgba:
                            if (pname == TextureEnvParameter::CombineAlpha)
                            {
                                context->validationError(GL_INVALID_VALUE, kInvalidTextureCombine);
                                return false;
                            }
                            break;
                        default:
                            context->validationError(GL_INVALID_VALUE, kInvalidTextureCombine);
                            return false;
                    }
                    break;
                }
                case TextureEnvParameter::Src0Rgb:
                case TextureEnvParameter::Src1Rgb:
                case TextureEnvParameter::Src2Rgb:
                case TextureEnvParameter::Src0Alpha:
                case TextureEnvParameter::Src1Alpha:
                case TextureEnvParameter::Src2Alpha:
                {
                    TextureSrc combine = FromGLenum<TextureSrc>(ConvertToGLenum(params[0]));
                    switch (combine)
                    {
                        case TextureSrc::Constant:
                        case TextureSrc::Previous:
                        case TextureSrc::PrimaryColor:
                        case TextureSrc::Texture:
                            break;
                        default:
                            context->validationError(GL_INVALID_VALUE, kInvalidTextureCombineSrc);
                            return false;
                    }
                    break;
                }
                case TextureEnvParameter::Op0Rgb:
                case TextureEnvParameter::Op1Rgb:
                case TextureEnvParameter::Op2Rgb:
                case TextureEnvParameter::Op0Alpha:
                case TextureEnvParameter::Op1Alpha:
                case TextureEnvParameter::Op2Alpha:
                {
                    TextureOp operand = FromGLenum<TextureOp>(ConvertToGLenum(params[0]));
                    switch (operand)
                    {
                        case TextureOp::SrcAlpha:
                        case TextureOp::OneMinusSrcAlpha:
                            break;
                        case TextureOp::SrcColor:
                        case TextureOp::OneMinusSrcColor:
                            if (pname == TextureEnvParameter::Op0Alpha ||
                                pname == TextureEnvParameter::Op1Alpha ||
                                pname == TextureEnvParameter::Op2Alpha)
                            {
                                context->validationError(GL_INVALID_VALUE, kInvalidTextureCombine);
                                return false;
                            }
                            break;
                        default:
                            context->validationError(GL_INVALID_VALUE, kInvalidTextureCombineOp);
                            return false;
                    }
                    break;
                }
                case TextureEnvParameter::RgbScale:
                case TextureEnvParameter::AlphaScale:
                    if (params[0] != 1.0f && params[0] != 2.0f && params[0] != 4.0f)
                    {
                        context->validationError(GL_INVALID_VALUE, kInvalidTextureEnvScale);
                        return false;
                    }
                    break;
                case TextureEnvParameter::Color:
                    break;
                default:
                    context->validationError(GL_INVALID_ENUM, kInvalidTextureEnvParameter);
                    return false;
            }
            break;
        case TextureEnvTarget::PointSprite:
            if (!context->getExtensions().pointSprite)
            {
                context->validationError(GL_INVALID_ENUM, kInvalidTextureEnvTarget);
                return false;
            }
            switch (pname)
            {
                case TextureEnvParameter::PointCoordReplace:
                    break;
                default:
                    context->validationError(GL_INVALID_ENUM, kInvalidTextureEnvParameter);
                    return false;
            }
            break;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidTextureEnvTarget);
            return false;
    }
    return true;
}

bool ValidateGetTexEnvCommon(Context *context, TextureEnvTarget target, TextureEnvParameter pname)
{
    GLfloat dummy[4] = {};
    switch (pname)
    {
        case TextureEnvParameter::Mode:
            ConvertPackedEnum(TextureEnvMode::Add, dummy);
            break;
        case TextureEnvParameter::CombineRgb:
        case TextureEnvParameter::CombineAlpha:
            ConvertPackedEnum(TextureCombine::Add, dummy);
            break;
        case TextureEnvParameter::Src0Rgb:
        case TextureEnvParameter::Src1Rgb:
        case TextureEnvParameter::Src2Rgb:
        case TextureEnvParameter::Src0Alpha:
        case TextureEnvParameter::Src1Alpha:
        case TextureEnvParameter::Src2Alpha:
            ConvertPackedEnum(TextureSrc::Constant, dummy);
            break;
        case TextureEnvParameter::Op0Rgb:
        case TextureEnvParameter::Op1Rgb:
        case TextureEnvParameter::Op2Rgb:
        case TextureEnvParameter::Op0Alpha:
        case TextureEnvParameter::Op1Alpha:
        case TextureEnvParameter::Op2Alpha:
            ConvertPackedEnum(TextureOp::SrcAlpha, dummy);
            break;
        case TextureEnvParameter::RgbScale:
        case TextureEnvParameter::AlphaScale:
        case TextureEnvParameter::PointCoordReplace:
            dummy[0] = 1.0f;
            break;
        default:
            break;
    }

    return ValidateTexEnvCommon(context, target, pname, dummy);
}

bool ValidatePointParameterCommon(Context *context, PointParameter pname, const GLfloat *params)
{
    ANGLE_VALIDATE_IS_GLES1(context);

    switch (pname)
    {
        case PointParameter::PointSizeMin:
        case PointParameter::PointSizeMax:
        case PointParameter::PointFadeThresholdSize:
        case PointParameter::PointDistanceAttenuation:
            for (unsigned int i = 0; i < GetPointParameterCount(pname); i++)
            {
                if (params[i] < 0.0f)
                {
                    context->validationError(GL_INVALID_VALUE, kInvalidPointParameterValue);
                    return false;
                }
            }
            break;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidPointParameter);
            return false;
    }

    return true;
}

bool ValidatePointSizeCommon(Context *context, GLfloat size)
{
    ANGLE_VALIDATE_IS_GLES1(context);

    if (size <= 0.0f)
    {
        context->validationError(GL_INVALID_VALUE, kInvalidPointSizeValue);
        return false;
    }

    return true;
}

bool ValidateDrawTexCommon(Context *context, float width, float height)
{
    ANGLE_VALIDATE_IS_GLES1(context);

    if (width <= 0.0f || height <= 0.0f)
    {
        context->validationError(GL_INVALID_VALUE, kNonPositiveDrawTextureDimension);
        return false;
    }

    return true;
}

}  // namespace gl

namespace gl
{

bool ValidateAlphaFunc(Context *context, AlphaTestFunc func, GLfloat ref)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return ValidateAlphaFuncCommon(context, func);
}

bool ValidateAlphaFuncx(Context *context, AlphaTestFunc func, GLfixed ref)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return ValidateAlphaFuncCommon(context, func);
}

bool ValidateClearColorx(Context *context, GLfixed red, GLfixed green, GLfixed blue, GLfixed alpha)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateClearDepthx(Context *context, GLfixed depth)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateClientActiveTexture(Context *context, GLenum texture)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return ValidateMultitextureUnit(context, texture);
}

bool ValidateClipPlanef(Context *context, GLenum plane, const GLfloat *eqn)
{
    return ValidateClipPlaneCommon(context, plane);
}

bool ValidateClipPlanex(Context *context, GLenum plane, const GLfixed *equation)
{
    return ValidateClipPlaneCommon(context, plane);
}

bool ValidateColor4f(Context *context, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateColor4ub(Context *context, GLubyte red, GLubyte green, GLubyte blue, GLubyte alpha)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateColor4x(Context *context, GLfixed red, GLfixed green, GLfixed blue, GLfixed alpha)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateColorPointer(Context *context,
                          GLint size,
                          VertexAttribType type,
                          GLsizei stride,
                          const void *pointer)
{
    return ValidateBuiltinVertexAttributeCommon(context, ClientVertexArrayType::Color, size, type,
                                                stride, pointer);
}

bool ValidateCullFace(Context *context, GLenum mode)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateDepthRangex(Context *context, GLfixed n, GLfixed f)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateDisableClientState(Context *context, ClientVertexArrayType arrayType)
{
    return ValidateClientStateCommon(context, arrayType);
}

bool ValidateEnableClientState(Context *context, ClientVertexArrayType arrayType)
{
    return ValidateClientStateCommon(context, arrayType);
}

bool ValidateFogf(Context *context, GLenum pname, GLfloat param)
{
    return ValidateFogCommon(context, pname, &param);
}

bool ValidateFogfv(Context *context, GLenum pname, const GLfloat *params)
{
    return ValidateFogCommon(context, pname, params);
}

bool ValidateFogx(Context *context, GLenum pname, GLfixed param)
{
    GLfloat asFloat = ConvertFixedToFloat(param);
    return ValidateFogCommon(context, pname, &asFloat);
}

bool ValidateFogxv(Context *context, GLenum pname, const GLfixed *params)
{
    unsigned int paramCount = GetFogParameterCount(pname);
    GLfloat paramsf[4]      = {};

    for (unsigned int i = 0; i < paramCount; i++)
    {
        paramsf[i] = ConvertFixedToFloat(params[i]);
    }

    return ValidateFogCommon(context, pname, paramsf);
}

bool ValidateFrustumf(Context *context,
                      GLfloat l,
                      GLfloat r,
                      GLfloat b,
                      GLfloat t,
                      GLfloat n,
                      GLfloat f)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    if (l == r || b == t || n == f || n <= 0.0f || f <= 0.0f)
    {
        context->validationError(GL_INVALID_VALUE, kInvalidProjectionMatrix);
    }
    return true;
}

bool ValidateFrustumx(Context *context,
                      GLfixed l,
                      GLfixed r,
                      GLfixed b,
                      GLfixed t,
                      GLfixed n,
                      GLfixed f)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    if (l == r || b == t || n == f || n <= 0 || f <= 0)
    {
        context->validationError(GL_INVALID_VALUE, kInvalidProjectionMatrix);
    }
    return true;
}

bool ValidateGetBufferParameteriv(Context *context, GLenum target, GLenum pname, GLint *params)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateGetClipPlanef(Context *context, GLenum plane, GLfloat *equation)
{
    return ValidateClipPlaneCommon(context, plane);
}

bool ValidateGetClipPlanex(Context *context, GLenum plane, GLfixed *equation)
{
    return ValidateClipPlaneCommon(context, plane);
}

bool ValidateGetFixedv(Context *context, GLenum pname, GLfixed *params)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateGetLightfv(Context *context, GLenum light, LightParameter pname, GLfloat *params)
{
    GLfloat dummyParams[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    return ValidateLightCommon(context, light, pname, dummyParams);
}

bool ValidateGetLightxv(Context *context, GLenum light, LightParameter pname, GLfixed *params)
{
    GLfloat dummyParams[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    return ValidateLightCommon(context, light, pname, dummyParams);
}

bool ValidateGetMaterialfv(Context *context, GLenum face, MaterialParameter pname, GLfloat *params)
{
    return ValidateMaterialQuery(context, face, pname);
}

bool ValidateGetMaterialxv(Context *context, GLenum face, MaterialParameter pname, GLfixed *params)
{
    return ValidateMaterialQuery(context, face, pname);
}

bool ValidateGetPointerv(Context *context, GLenum pname, void **params)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    switch (pname)
    {
        case GL_VERTEX_ARRAY_POINTER:
        case GL_NORMAL_ARRAY_POINTER:
        case GL_COLOR_ARRAY_POINTER:
        case GL_TEXTURE_COORD_ARRAY_POINTER:
        case GL_POINT_SIZE_ARRAY_POINTER_OES:
            return true;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidPointerQuery);
            return false;
    }
}

bool ValidateGetTexEnvfv(Context *context,
                         TextureEnvTarget target,
                         TextureEnvParameter pname,
                         GLfloat *params)
{
    return ValidateGetTexEnvCommon(context, target, pname);
}

bool ValidateGetTexEnviv(Context *context,
                         TextureEnvTarget target,
                         TextureEnvParameter pname,
                         GLint *params)
{
    return ValidateGetTexEnvCommon(context, target, pname);
}

bool ValidateGetTexEnvxv(Context *context,
                         TextureEnvTarget target,
                         TextureEnvParameter pname,
                         GLfixed *params)
{
    return ValidateGetTexEnvCommon(context, target, pname);
}

bool ValidateGetTexParameterxv(Context *context, TextureType target, GLenum pname, GLfixed *params)
{
    ANGLE_VALIDATE_IS_GLES1(context);

    if (!ValidateGetTexParameterBase(context, target, pname, nullptr))
    {
        return false;
    }

    return true;
}

bool ValidateLightModelf(Context *context, GLenum pname, GLfloat param)
{
    return ValidateLightModelSingleComponent(context, pname);
}

bool ValidateLightModelfv(Context *context, GLenum pname, const GLfloat *params)
{
    return ValidateLightModelCommon(context, pname);
}

bool ValidateLightModelx(Context *context, GLenum pname, GLfixed param)
{
    return ValidateLightModelSingleComponent(context, pname);
}

bool ValidateLightModelxv(Context *context, GLenum pname, const GLfixed *param)
{
    return ValidateLightModelCommon(context, pname);
}

bool ValidateLightf(Context *context, GLenum light, LightParameter pname, GLfloat param)
{
    return ValidateLightSingleComponent(context, light, pname, param);
}

bool ValidateLightfv(Context *context, GLenum light, LightParameter pname, const GLfloat *params)
{
    return ValidateLightCommon(context, light, pname, params);
}

bool ValidateLightx(Context *context, GLenum light, LightParameter pname, GLfixed param)
{
    return ValidateLightSingleComponent(context, light, pname, ConvertFixedToFloat(param));
}

bool ValidateLightxv(Context *context, GLenum light, LightParameter pname, const GLfixed *params)
{
    GLfloat paramsf[4];
    for (unsigned int i = 0; i < GetLightParameterCount(pname); i++)
    {
        paramsf[i] = ConvertFixedToFloat(params[i]);
    }

    return ValidateLightCommon(context, light, pname, paramsf);
}

bool ValidateLineWidthx(Context *context, GLfixed width)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateLoadIdentity(Context *context)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateLoadMatrixf(Context *context, const GLfloat *m)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateLoadMatrixx(Context *context, const GLfixed *m)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateLogicOp(Context *context, LogicalOperation opcode)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    switch (opcode)
    {
        case LogicalOperation::And:
        case LogicalOperation::AndInverted:
        case LogicalOperation::AndReverse:
        case LogicalOperation::Clear:
        case LogicalOperation::Copy:
        case LogicalOperation::CopyInverted:
        case LogicalOperation::Equiv:
        case LogicalOperation::Invert:
        case LogicalOperation::Nand:
        case LogicalOperation::Noop:
        case LogicalOperation::Nor:
        case LogicalOperation::Or:
        case LogicalOperation::OrInverted:
        case LogicalOperation::OrReverse:
        case LogicalOperation::Set:
        case LogicalOperation::Xor:
            return true;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidLogicOp);
            return false;
    }
}

bool ValidateMaterialf(Context *context, GLenum face, MaterialParameter pname, GLfloat param)
{
    return ValidateMaterialSingleComponent(context, face, pname, param);
}

bool ValidateMaterialfv(Context *context,
                        GLenum face,
                        MaterialParameter pname,
                        const GLfloat *params)
{
    return ValidateMaterialSetting(context, face, pname, params);
}

bool ValidateMaterialx(Context *context, GLenum face, MaterialParameter pname, GLfixed param)
{
    return ValidateMaterialSingleComponent(context, face, pname, ConvertFixedToFloat(param));
}

bool ValidateMaterialxv(Context *context,
                        GLenum face,
                        MaterialParameter pname,
                        const GLfixed *params)
{
    GLfloat paramsf[4];

    for (unsigned int i = 0; i < GetMaterialParameterCount(pname); i++)
    {
        paramsf[i] = ConvertFixedToFloat(params[i]);
    }

    return ValidateMaterialSetting(context, face, pname, paramsf);
}

bool ValidateMatrixMode(Context *context, MatrixType mode)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    switch (mode)
    {
        case MatrixType::Projection:
        case MatrixType::Modelview:
        case MatrixType::Texture:
            return true;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidMatrixMode);
            return false;
    }
}

bool ValidateMultMatrixf(Context *context, const GLfloat *m)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateMultMatrixx(Context *context, const GLfixed *m)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateMultiTexCoord4f(Context *context,
                             GLenum target,
                             GLfloat s,
                             GLfloat t,
                             GLfloat r,
                             GLfloat q)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return ValidateMultitextureUnit(context, target);
}

bool ValidateMultiTexCoord4x(Context *context,
                             GLenum target,
                             GLfixed s,
                             GLfixed t,
                             GLfixed r,
                             GLfixed q)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return ValidateMultitextureUnit(context, target);
}

bool ValidateNormal3f(Context *context, GLfloat nx, GLfloat ny, GLfloat nz)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateNormal3x(Context *context, GLfixed nx, GLfixed ny, GLfixed nz)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateNormalPointer(Context *context,
                           VertexAttribType type,
                           GLsizei stride,
                           const void *pointer)
{
    return ValidateBuiltinVertexAttributeCommon(context, ClientVertexArrayType::Normal, 3, type,
                                                stride, pointer);
}

bool ValidateOrthof(Context *context,
                    GLfloat l,
                    GLfloat r,
                    GLfloat b,
                    GLfloat t,
                    GLfloat n,
                    GLfloat f)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    if (l == r || b == t || n == f)
    {
        context->validationError(GL_INVALID_VALUE, kInvalidProjectionMatrix);
    }
    return true;
}

bool ValidateOrthox(Context *context,
                    GLfixed l,
                    GLfixed r,
                    GLfixed b,
                    GLfixed t,
                    GLfixed n,
                    GLfixed f)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    if (l == r || b == t || n == f || n <= 0 || f <= 0)
    {
        context->validationError(GL_INVALID_VALUE, kInvalidProjectionMatrix);
    }
    return true;
}

bool ValidatePointParameterf(Context *context, PointParameter pname, GLfloat param)
{
    unsigned int paramCount = GetPointParameterCount(pname);
    if (paramCount != 1)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidPointParameter);
        return false;
    }

    return ValidatePointParameterCommon(context, pname, &param);
}

bool ValidatePointParameterfv(Context *context, PointParameter pname, const GLfloat *params)
{
    return ValidatePointParameterCommon(context, pname, params);
}

bool ValidatePointParameterx(Context *context, PointParameter pname, GLfixed param)
{
    unsigned int paramCount = GetPointParameterCount(pname);
    if (paramCount != 1)
    {
        context->validationError(GL_INVALID_ENUM, kInvalidPointParameter);
        return false;
    }

    GLfloat paramf = ConvertFixedToFloat(param);
    return ValidatePointParameterCommon(context, pname, &paramf);
}

bool ValidatePointParameterxv(Context *context, PointParameter pname, const GLfixed *params)
{
    GLfloat paramsf[4] = {};
    for (unsigned int i = 0; i < GetPointParameterCount(pname); i++)
    {
        paramsf[i] = ConvertFixedToFloat(params[i]);
    }
    return ValidatePointParameterCommon(context, pname, paramsf);
}

bool ValidatePointSize(Context *context, GLfloat size)
{
    return ValidatePointSizeCommon(context, size);
}

bool ValidatePointSizex(Context *context, GLfixed size)
{
    return ValidatePointSizeCommon(context, ConvertFixedToFloat(size));
}

bool ValidatePolygonOffsetx(Context *context, GLfixed factor, GLfixed units)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidatePopMatrix(Context *context)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    const auto &stack = context->getState().gles1().currentMatrixStack();
    if (stack.size() == 1)
    {
        context->validationError(GL_STACK_UNDERFLOW, kMatrixStackUnderflow);
        return false;
    }
    return true;
}

bool ValidatePushMatrix(Context *context)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    const auto &stack = context->getState().gles1().currentMatrixStack();
    if (stack.size() == stack.max_size())
    {
        context->validationError(GL_STACK_OVERFLOW, kMatrixStackOverflow);
        return false;
    }
    return true;
}

bool ValidateRotatef(Context *context, GLfloat angle, GLfloat x, GLfloat y, GLfloat z)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateRotatex(Context *context, GLfixed angle, GLfixed x, GLfixed y, GLfixed z)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateSampleCoveragex(Context *context, GLclampx value, GLboolean invert)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateScalef(Context *context, GLfloat x, GLfloat y, GLfloat z)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateScalex(Context *context, GLfixed x, GLfixed y, GLfixed z)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateShadeModel(Context *context, ShadingModel mode)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    switch (mode)
    {
        case ShadingModel::Flat:
        case ShadingModel::Smooth:
            return true;
        default:
            context->validationError(GL_INVALID_ENUM, kInvalidShadingModel);
            return false;
    }
}

bool ValidateTexCoordPointer(Context *context,
                             GLint size,
                             VertexAttribType type,
                             GLsizei stride,
                             const void *pointer)
{
    return ValidateBuiltinVertexAttributeCommon(context, ClientVertexArrayType::TextureCoord, size,
                                                type, stride, pointer);
}

bool ValidateTexEnvf(Context *context,
                     TextureEnvTarget target,
                     TextureEnvParameter pname,
                     GLfloat param)
{
    return ValidateTexEnvCommon(context, target, pname, &param);
}

bool ValidateTexEnvfv(Context *context,
                      TextureEnvTarget target,
                      TextureEnvParameter pname,
                      const GLfloat *params)
{
    return ValidateTexEnvCommon(context, target, pname, params);
}

bool ValidateTexEnvi(Context *context,
                     TextureEnvTarget target,
                     TextureEnvParameter pname,
                     GLint param)
{
    GLfloat paramf = static_cast<GLfloat>(param);
    return ValidateTexEnvCommon(context, target, pname, &paramf);
}

bool ValidateTexEnviv(Context *context,
                      TextureEnvTarget target,
                      TextureEnvParameter pname,
                      const GLint *params)
{
    GLfloat paramsf[4];
    for (unsigned int i = 0; i < GetTextureEnvParameterCount(pname); i++)
    {
        paramsf[i] = static_cast<GLfloat>(params[i]);
    }
    return ValidateTexEnvCommon(context, target, pname, paramsf);
}

bool ValidateTexEnvx(Context *context,
                     TextureEnvTarget target,
                     TextureEnvParameter pname,
                     GLfixed param)
{
    GLfloat paramf = static_cast<GLfloat>(param);
    return ValidateTexEnvCommon(context, target, pname, &paramf);
}

bool ValidateTexEnvxv(Context *context,
                      TextureEnvTarget target,
                      TextureEnvParameter pname,
                      const GLfixed *params)
{
    GLfloat paramsf[4];
    for (unsigned int i = 0; i < GetTextureEnvParameterCount(pname); i++)
    {
        paramsf[i] = static_cast<GLfloat>(params[i]);
    }
    return ValidateTexEnvCommon(context, target, pname, paramsf);
}

bool ValidateTexParameterx(Context *context, TextureType target, GLenum pname, GLfixed param)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    GLfloat paramf = ConvertFixedToFloat(param);
    return ValidateTexParameterBase(context, target, pname, -1, false, &paramf);
}

bool ValidateTexParameterxv(Context *context,
                            TextureType target,
                            GLenum pname,
                            const GLfixed *params)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    GLfloat paramsf[4] = {};
    for (unsigned int i = 0; i < GetTexParameterCount(pname); i++)
    {
        paramsf[i] = ConvertFixedToFloat(params[i]);
    }
    return ValidateTexParameterBase(context, target, pname, -1, true, paramsf);
}

bool ValidateTranslatef(Context *context, GLfloat x, GLfloat y, GLfloat z)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateTranslatex(Context *context, GLfixed x, GLfixed y, GLfixed z)
{
    ANGLE_VALIDATE_IS_GLES1(context);
    return true;
}

bool ValidateVertexPointer(Context *context,
                           GLint size,
                           VertexAttribType type,
                           GLsizei stride,
                           const void *pointer)
{
    return ValidateBuiltinVertexAttributeCommon(context, ClientVertexArrayType::Vertex, size, type,
                                                stride, pointer);
}

bool ValidateDrawTexfOES(Context *context,
                         GLfloat x,
                         GLfloat y,
                         GLfloat z,
                         GLfloat width,
                         GLfloat height)
{
    return ValidateDrawTexCommon(context, width, height);
}

bool ValidateDrawTexfvOES(Context *context, const GLfloat *coords)
{
    return ValidateDrawTexCommon(context, coords[3], coords[4]);
}

bool ValidateDrawTexiOES(Context *context, GLint x, GLint y, GLint z, GLint width, GLint height)
{
    return ValidateDrawTexCommon(context, static_cast<GLfloat>(width),
                                 static_cast<GLfloat>(height));
}

bool ValidateDrawTexivOES(Context *context, const GLint *coords)
{
    return ValidateDrawTexCommon(context, static_cast<GLfloat>(coords[3]),
                                 static_cast<GLfloat>(coords[4]));
}

bool ValidateDrawTexsOES(Context *context,
                         GLshort x,
                         GLshort y,
                         GLshort z,
                         GLshort width,
                         GLshort height)
{
    return ValidateDrawTexCommon(context, static_cast<GLfloat>(width),
                                 static_cast<GLfloat>(height));
}

bool ValidateDrawTexsvOES(Context *context, const GLshort *coords)
{
    return ValidateDrawTexCommon(context, static_cast<GLfloat>(coords[3]),
                                 static_cast<GLfloat>(coords[4]));
}

bool ValidateDrawTexxOES(Context *context,
                         GLfixed x,
                         GLfixed y,
                         GLfixed z,
                         GLfixed width,
                         GLfixed height)
{
    return ValidateDrawTexCommon(context, ConvertFixedToFloat(width), ConvertFixedToFloat(height));
}

bool ValidateDrawTexxvOES(Context *context, const GLfixed *coords)
{
    return ValidateDrawTexCommon(context, ConvertFixedToFloat(coords[3]),
                                 ConvertFixedToFloat(coords[4]));
}

bool ValidateCurrentPaletteMatrixOES(Context *context, GLuint matrixpaletteindex)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateLoadPaletteFromModelViewMatrixOES(Context *context)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateMatrixIndexPointerOES(Context *context,
                                   GLint size,
                                   GLenum type,
                                   GLsizei stride,
                                   const void *pointer)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateWeightPointerOES(Context *context,
                              GLint size,
                              GLenum type,
                              GLsizei stride,
                              const void *pointer)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidatePointSizePointerOES(Context *context,
                                 VertexAttribType type,
                                 GLsizei stride,
                                 const void *pointer)
{
    return ValidateBuiltinVertexAttributeCommon(context, ClientVertexArrayType::PointSize, 1, type,
                                                stride, pointer);
}

bool ValidateQueryMatrixxOES(Context *context, GLfixed *mantissa, GLint *exponent)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateGenFramebuffersOES(Context *context, GLsizei n, FramebufferID *framebuffers)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateDeleteFramebuffersOES(Context *context, GLsizei n, const FramebufferID *framebuffers)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateGenRenderbuffersOES(Context *context, GLsizei n, RenderbufferID *renderbuffers)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateDeleteRenderbuffersOES(Context *context,
                                    GLsizei n,
                                    const RenderbufferID *renderbuffers)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateBindFramebufferOES(Context *context, GLenum target, FramebufferID framebuffer)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateBindRenderbufferOES(Context *context, GLenum target, RenderbufferID renderbuffer)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateCheckFramebufferStatusOES(Context *context, GLenum target)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateFramebufferRenderbufferOES(Context *context,
                                        GLenum target,
                                        GLenum attachment,
                                        GLenum rbtarget,
                                        RenderbufferID renderbuffer)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateFramebufferTexture2DOES(Context *context,
                                     GLenum target,
                                     GLenum attachment,
                                     TextureTarget textarget,
                                     TextureID texture,
                                     GLint level)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateGenerateMipmapOES(Context *context, TextureType target)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateGetFramebufferAttachmentParameterivOES(Context *context,
                                                    GLenum target,
                                                    GLenum attachment,
                                                    GLenum pname,
                                                    GLint *params)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateGetRenderbufferParameterivOES(Context *context,
                                           GLenum target,
                                           GLenum pname,
                                           GLint *params)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateIsFramebufferOES(Context *context, FramebufferID framebuffer)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateIsRenderbufferOES(Context *context, RenderbufferID renderbuffer)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateRenderbufferStorageOES(Context *context,
                                    GLenum target,
                                    GLenum internalformat,
                                    GLsizei width,
                                    GLsizei height)
{
    UNIMPLEMENTED();
    return true;
}

// GL_OES_texture_cube_map

bool ValidateGetTexGenfvOES(Context *context, GLenum coord, GLenum pname, GLfloat *params)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateGetTexGenivOES(Context *context, GLenum coord, GLenum pname, int *params)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateGetTexGenxvOES(Context *context, GLenum coord, GLenum pname, GLfixed *params)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateTexGenfvOES(Context *context, GLenum coord, GLenum pname, const GLfloat *params)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateTexGenivOES(Context *context, GLenum coord, GLenum pname, const GLint *param)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateTexGenxvOES(Context *context, GLenum coord, GLenum pname, const GLint *param)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateTexGenfOES(Context *context, GLenum coord, GLenum pname, GLfloat param)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateTexGeniOES(Context *context, GLenum coord, GLenum pname, GLint param)
{
    UNIMPLEMENTED();
    return true;
}

bool ValidateTexGenxOES(Context *context, GLenum coord, GLenum pname, GLfixed param)
{
    UNIMPLEMENTED();
    return true;
}

}  // namespace gl
