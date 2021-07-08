//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// queryconversions.h: Declaration of state query cast conversions

#ifndef LIBANGLE_QUERY_CONVERSIONS_H_
#define LIBANGLE_QUERY_CONVERSIONS_H_

#include "angle_gl.h"
#include "common/angleutils.h"

namespace gl
{
class Context;

// Helper class for converting a GL type to a GLenum:
// We can't use CastStateValueEnum generally, because of GLboolean + GLubyte overlap.
// We restrict our use to CastFromStateValue and CastQueryValueTo, where it eliminates
// duplicate parameters.

template <typename GLType>
struct GLTypeToGLenum
{
    // static constexpr GLenum value;
};

template <>
struct GLTypeToGLenum<GLint>
{
    static constexpr GLenum value = GL_INT;
};
template <>
struct GLTypeToGLenum<GLuint>
{
    static constexpr GLenum value = GL_UNSIGNED_INT;
};
template <>
struct GLTypeToGLenum<GLboolean>
{
    static constexpr GLenum value = GL_BOOL;
};
template <>
struct GLTypeToGLenum<GLint64>
{
    static constexpr GLenum value = GL_INT_64_ANGLEX;
};
template <>
struct GLTypeToGLenum<GLuint64>
{
    static constexpr GLenum value = GL_UINT_64_ANGLEX;
};
template <>
struct GLTypeToGLenum<GLfloat>
{
    static constexpr GLenum value = GL_FLOAT;
};

GLint CastMaskValue(GLuint value);

template <typename QueryT, typename InternalT>
QueryT CastFromGLintStateValue(GLenum pname, InternalT value);

template <typename QueryT, typename NativeT>
QueryT CastFromStateValue(GLenum pname, NativeT value);

template <typename NativeT, typename QueryT>
NativeT CastQueryValueTo(GLenum pname, QueryT value);

template <typename ParamType>
GLenum ConvertToGLenum(GLenum pname, ParamType param)
{
    return static_cast<GLenum>(CastQueryValueTo<GLuint>(pname, param));
}

template <typename ParamType>
GLenum ConvertToGLenum(ParamType param)
{
    return ConvertToGLenum(GL_NONE, param);
}

template <typename OutType>
OutType ConvertGLenum(GLenum param)
{
    return static_cast<OutType>(param);
}

template <typename InType, typename OutType>
void ConvertGLenumValue(InType param, OutType *out)
{
    *out = ConvertGLenum<OutType>(static_cast<GLenum>(param));
}

template <typename PackedEnumType, typename OutType>
void ConvertPackedEnum(PackedEnumType param, OutType *out)
{
    *out = static_cast<OutType>(ToGLenum(param));
}

template <typename ParamType>
GLint ConvertToGLint(ParamType param)
{
    return CastQueryValueTo<GLint>(GL_NONE, param);
}

template <typename ParamType>
bool ConvertToBool(ParamType param)
{
    return param != GL_FALSE;
}

template <typename ParamType>
GLboolean ConvertToGLBoolean(ParamType param)
{
    return param ? GL_TRUE : GL_FALSE;
}

// The GL state query API types are: bool, int, uint, float, int64, uint64
template <typename QueryT>
void CastStateValues(Context *context,
                     GLenum nativeType,
                     GLenum pname,
                     unsigned int numParams,
                     QueryT *outParams);

// The GL state query API types are: bool, int, uint, float, int64, uint64
template <typename QueryT>
void CastIndexedStateValues(Context *context,
                            GLenum nativeType,
                            GLenum pname,
                            GLuint index,
                            unsigned int numParams,
                            QueryT *outParams);
}  // namespace gl

#endif  // LIBANGLE_QUERY_CONVERSIONS_H_
