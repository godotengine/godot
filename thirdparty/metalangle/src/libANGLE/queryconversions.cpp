//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// queryconversions.cpp: Implementation of state query cast conversions

#include "libANGLE/queryconversions.h"

#include <vector>

#include "common/utilities.h"
#include "libANGLE/Context.h"

namespace gl
{

namespace
{

GLint64 ExpandFloatToInteger(GLfloat value)
{
    return static_cast<GLint64>((static_cast<double>(0xFFFFFFFFULL) * value - 1.0) / 2.0);
}

template <typename QueryT, typename NativeT>
QueryT CastFromStateValueToInt(GLenum pname, NativeT value)
{
    GLenum nativeType = GLTypeToGLenum<NativeT>::value;

    if (nativeType == GL_FLOAT)
    {
        // RGBA color values and DepthRangeF values are converted to integer using Equation 2.4 from
        // Table 4.5
        switch (pname)
        {
            case GL_DEPTH_RANGE:
            case GL_COLOR_CLEAR_VALUE:
            case GL_DEPTH_CLEAR_VALUE:
            case GL_BLEND_COLOR:
            // GLES1 emulation:
            // Also, several GLES1.x values need to be converted to integer with
            // ExpandFloatToInteger rather than rounding. See GLES 1.1 spec 6.1.2 "Data
            // Conversions".
            case GL_ALPHA_TEST_REF:
            case GL_CURRENT_COLOR:
                return clampCast<QueryT>(ExpandFloatToInteger(static_cast<GLfloat>(value)));
            default:
                return clampCast<QueryT>(std::round(value));
        }
    }

    return clampCast<QueryT>(value);
}

template <typename NativeT, typename QueryT>
NativeT CastQueryValueToInt(GLenum pname, QueryT value)
{
    GLenum queryType = GLTypeToGLenum<QueryT>::value;

    if (queryType == GL_FLOAT)
    {
        return static_cast<NativeT>(std::round(value));
    }

    return static_cast<NativeT>(value);
}

}  // anonymous namespace

GLint CastMaskValue(GLuint value)
{
    return clampCast<GLint>(value);
}

template <typename QueryT, typename InternalT>
QueryT CastFromGLintStateValue(GLenum pname, InternalT value)
{
    return CastFromStateValue<QueryT, GLint>(pname, clampCast<GLint, InternalT>(value));
}

template GLfloat CastFromGLintStateValue<GLfloat, GLenum>(GLenum pname, GLenum value);
template GLint CastFromGLintStateValue<GLint, GLenum>(GLenum pname, GLenum value);
template GLint64 CastFromGLintStateValue<GLint64, GLenum>(GLenum pname, GLenum value);
template GLuint CastFromGLintStateValue<GLuint, GLenum>(GLenum pname, GLenum value);
template GLuint CastFromGLintStateValue<GLuint, GLint>(GLenum pname, GLint value);
template GLfloat CastFromGLintStateValue<GLfloat, GLint>(GLenum pname, GLint value);
template GLint CastFromGLintStateValue<GLint, GLint>(GLenum pname, GLint value);
template GLfloat CastFromGLintStateValue<GLfloat, bool>(GLenum pname, bool value);
template GLuint CastFromGLintStateValue<GLuint, bool>(GLenum pname, bool value);
template GLint CastFromGLintStateValue<GLint, bool>(GLenum pname, bool value);

template <typename QueryT, typename NativeT>
QueryT CastFromStateValue(GLenum pname, NativeT value)
{
    GLenum queryType = GLTypeToGLenum<QueryT>::value;

    switch (queryType)
    {
        case GL_INT:
        case GL_INT_64_ANGLEX:
        case GL_UNSIGNED_INT:
        case GL_UINT_64_ANGLEX:
            return CastFromStateValueToInt<QueryT, NativeT>(pname, value);
        case GL_FLOAT:
            return static_cast<QueryT>(value);
        case GL_BOOL:
            return static_cast<QueryT>(value == static_cast<NativeT>(0) ? GL_FALSE : GL_TRUE);
        default:
            UNREACHABLE();
            return 0;
    }
}
template GLint CastFromStateValue<GLint, GLint>(GLenum pname, GLint value);
template GLint CastFromStateValue<GLint, GLint64>(GLenum pname, GLint64 value);
template GLint64 CastFromStateValue<GLint64, GLint>(GLenum pname, GLint value);
template GLint64 CastFromStateValue<GLint64, GLint64>(GLenum pname, GLint64 value);
template GLfloat CastFromStateValue<GLfloat, GLint>(GLenum pname, GLint value);
template GLfloat CastFromStateValue<GLfloat, GLuint>(GLenum pname, GLuint value);
template GLfloat CastFromStateValue<GLfloat, GLfloat>(GLenum pname, GLfloat value);
template GLint CastFromStateValue<GLint, GLfloat>(GLenum pname, GLfloat value);
template GLuint CastFromStateValue<GLuint, GLfloat>(GLenum pname, GLfloat value);
template GLuint CastFromStateValue<GLuint, GLint>(GLenum pname, GLint value);
template GLuint CastFromStateValue<GLuint, GLuint>(GLenum pname, GLuint value);
template GLint CastFromStateValue<GLint, GLboolean>(GLenum pname, GLboolean value);
template GLint64 CastFromStateValue<GLint64, GLboolean>(GLenum pname, GLboolean value);
template GLint CastFromStateValue<GLint, GLuint>(GLenum pname, GLuint value);
template GLint64 CastFromStateValue<GLint64, GLuint>(GLenum pname, GLuint value);
template GLuint64 CastFromStateValue<GLuint64, GLuint>(GLenum pname, GLuint value);

template <typename NativeT, typename QueryT>
NativeT CastQueryValueTo(GLenum pname, QueryT value)
{
    GLenum nativeType = GLTypeToGLenum<NativeT>::value;

    switch (nativeType)
    {
        case GL_INT:
        case GL_INT_64_ANGLEX:
        case GL_UNSIGNED_INT:
        case GL_UINT_64_ANGLEX:
            return CastQueryValueToInt<NativeT, QueryT>(pname, value);
        case GL_FLOAT:
            return static_cast<NativeT>(value);
        case GL_BOOL:
            return static_cast<NativeT>(value == static_cast<QueryT>(0) ? GL_FALSE : GL_TRUE);
        default:
            UNREACHABLE();
            return 0;
    }
}

template GLint CastQueryValueTo<GLint, GLfloat>(GLenum pname, GLfloat value);
template GLboolean CastQueryValueTo<GLboolean, GLint>(GLenum pname, GLint value);
template GLint CastQueryValueTo<GLint, GLint>(GLenum pname, GLint value);
template GLint CastQueryValueTo<GLint, GLuint>(GLenum pname, GLuint value);
template GLfloat CastQueryValueTo<GLfloat, GLint>(GLenum pname, GLint value);
template GLfloat CastQueryValueTo<GLfloat, GLuint>(GLenum pname, GLuint value);
template GLfloat CastQueryValueTo<GLfloat, GLfloat>(GLenum pname, GLfloat value);
template GLuint CastQueryValueTo<GLuint, GLint>(GLenum pname, GLint value);
template GLuint CastQueryValueTo<GLuint, GLuint>(GLenum pname, GLuint value);
template GLuint CastQueryValueTo<GLuint, GLfloat>(GLenum pname, GLfloat value);

template <typename QueryT>
void CastStateValues(Context *context,
                     GLenum nativeType,
                     GLenum pname,
                     unsigned int numParams,
                     QueryT *outParams)
{
    if (nativeType == GL_INT)
    {
        std::vector<GLint> intParams(numParams, 0);
        context->getIntegervImpl(pname, intParams.data());

        for (unsigned int i = 0; i < numParams; ++i)
        {
            outParams[i] = CastFromStateValue<QueryT>(pname, intParams[i]);
        }
    }
    else if (nativeType == GL_BOOL)
    {
        std::vector<GLboolean> boolParams(numParams, GL_FALSE);
        context->getBooleanvImpl(pname, boolParams.data());

        for (unsigned int i = 0; i < numParams; ++i)
        {
            outParams[i] =
                (boolParams[i] == GL_FALSE ? static_cast<QueryT>(0) : static_cast<QueryT>(1));
        }
    }
    else if (nativeType == GL_FLOAT)
    {
        std::vector<GLfloat> floatParams(numParams, 0.0f);
        context->getFloatvImpl(pname, floatParams.data());

        for (unsigned int i = 0; i < numParams; ++i)
        {
            outParams[i] = CastFromStateValue<QueryT>(pname, floatParams[i]);
        }
    }
    else if (nativeType == GL_INT_64_ANGLEX)
    {
        std::vector<GLint64> int64Params(numParams, 0);
        context->getInteger64v(pname, int64Params.data());

        for (unsigned int i = 0; i < numParams; ++i)
        {
            outParams[i] = CastFromStateValue<QueryT>(pname, int64Params[i]);
        }
    }
    else
        UNREACHABLE();
}

// Explicit template instantiation (how we export template functions in different files)
// The calls below will make CastStateValues successfully link with the GL state query types
// The GL state query API types are: bool, int, uint, float, int64, uint64

template void CastStateValues<GLboolean>(Context *, GLenum, GLenum, unsigned int, GLboolean *);
template void CastStateValues<GLint>(Context *, GLenum, GLenum, unsigned int, GLint *);
template void CastStateValues<GLuint>(Context *, GLenum, GLenum, unsigned int, GLuint *);
template void CastStateValues<GLfloat>(Context *, GLenum, GLenum, unsigned int, GLfloat *);
template void CastStateValues<GLint64>(Context *, GLenum, GLenum, unsigned int, GLint64 *);

template <typename QueryT>
void CastIndexedStateValues(Context *context,
                            GLenum nativeType,
                            GLenum pname,
                            GLuint index,
                            unsigned int numParams,
                            QueryT *outParams)
{
    if (nativeType == GL_INT)
    {
        std::vector<GLint> intParams(numParams, 0);
        context->getIntegeri_v(pname, index, intParams.data());

        for (unsigned int i = 0; i < numParams; ++i)
        {
            outParams[i] = CastFromStateValue<QueryT>(pname, intParams[i]);
        }
    }
    else if (nativeType == GL_BOOL)
    {
        std::vector<GLboolean> boolParams(numParams, GL_FALSE);
        context->getBooleani_v(pname, index, boolParams.data());

        for (unsigned int i = 0; i < numParams; ++i)
        {
            outParams[i] =
                (boolParams[i] == GL_FALSE ? static_cast<QueryT>(0) : static_cast<QueryT>(1));
        }
    }
    else if (nativeType == GL_INT_64_ANGLEX)
    {
        std::vector<GLint64> int64Params(numParams, 0);
        context->getInteger64i_v(pname, index, int64Params.data());

        for (unsigned int i = 0; i < numParams; ++i)
        {
            outParams[i] = CastFromStateValue<QueryT>(pname, int64Params[i]);
        }
    }
    else
        UNREACHABLE();
}

template void CastIndexedStateValues<GLboolean>(Context *,
                                                GLenum,
                                                GLenum,
                                                GLuint index,
                                                unsigned int,
                                                GLboolean *);
template void CastIndexedStateValues<GLint>(Context *,
                                            GLenum,
                                            GLenum,
                                            GLuint index,
                                            unsigned int,
                                            GLint *);
template void CastIndexedStateValues<GLuint>(Context *,
                                             GLenum,
                                             GLenum,
                                             GLuint index,
                                             unsigned int,
                                             GLuint *);
template void CastIndexedStateValues<GLfloat>(Context *,
                                              GLenum,
                                              GLenum,
                                              GLuint index,
                                              unsigned int,
                                              GLfloat *);
template void CastIndexedStateValues<GLint64>(Context *,
                                              GLenum,
                                              GLenum,
                                              GLuint index,
                                              unsigned int,
                                              GLint64 *);
}  // namespace gl
