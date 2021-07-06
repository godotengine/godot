//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// entry_point_utils:
//   These helpers are used in GL/GLES entry point routines.

#ifndef LIBANGLE_ENTRY_POINT_UTILS_H_
#define LIBANGLE_ENTRY_POINT_UTILS_H_

#include "angle_gl.h"
#include "common/Optional.h"
#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "common/mathutil.h"
#include "libANGLE/Display.h"
#include "libANGLE/entry_points_enum_autogen.h"

namespace gl
{
// A template struct for determining the default value to return for each entry point.
template <EntryPoint EP, typename ReturnType>
struct DefaultReturnValue;

// Default return values for each basic return type.
template <EntryPoint EP>
struct DefaultReturnValue<EP, GLint>
{
    static constexpr GLint kValue = -1;
};

// This doubles as the GLenum return value.
template <EntryPoint EP>
struct DefaultReturnValue<EP, GLuint>
{
    static constexpr GLuint kValue = 0;
};

template <EntryPoint EP>
struct DefaultReturnValue<EP, GLboolean>
{
    static constexpr GLboolean kValue = GL_FALSE;
};

template <EntryPoint EP>
struct DefaultReturnValue<EP, ShaderProgramID>
{
    static constexpr ShaderProgramID kValue = {0};
};

// Catch-all rules for pointer types.
template <EntryPoint EP, typename PointerType>
struct DefaultReturnValue<EP, const PointerType *>
{
    static constexpr const PointerType *kValue = nullptr;
};

template <EntryPoint EP, typename PointerType>
struct DefaultReturnValue<EP, PointerType *>
{
    static constexpr PointerType *kValue = nullptr;
};

// Overloaded to return invalid index
template <>
struct DefaultReturnValue<EntryPoint::GetUniformBlockIndex, GLuint>
{
    static constexpr GLuint kValue = GL_INVALID_INDEX;
};

// Specialized enum error value.
template <>
struct DefaultReturnValue<EntryPoint::ClientWaitSync, GLenum>
{
    static constexpr GLenum kValue = GL_WAIT_FAILED;
};

// glTestFenceNV should still return TRUE for an invalid fence.
template <>
struct DefaultReturnValue<EntryPoint::TestFenceNV, GLboolean>
{
    static constexpr GLboolean kValue = GL_TRUE;
};

template <EntryPoint EP, typename ReturnType>
constexpr ANGLE_INLINE ReturnType GetDefaultReturnValue()
{
    return DefaultReturnValue<EP, ReturnType>::kValue;
}

#if ANGLE_CAPTURE_ENABLED
#    define ANGLE_CAPTURE(Func, ...) CaptureCallToFrameCapture(Capture##Func, __VA_ARGS__)
#else
#    define ANGLE_CAPTURE(...)
#endif  // ANGLE_CAPTURE_ENABLED

#define FUNC_EVENT(format, ...) EVENT(__FUNCTION__, format, __VA_ARGS__)

inline int CID(const Context *context)
{
    return context != nullptr ? context->id() : 0;
}
}  // namespace gl

namespace egl
{
inline int CID(EGLDisplay display, EGLContext context)
{
    auto *displayPtr = reinterpret_cast<const egl::Display *>(display);
    if (!Display::isValidDisplay(displayPtr))
    {
        return -1;
    }
    auto *contextPtr = reinterpret_cast<const gl::Context *>(context);
    if (!displayPtr->isValidContext(contextPtr))
    {
        return -1;
    }
    return gl::CID(contextPtr);
}
}  // namespace egl

#endif  // LIBANGLE_ENTRY_POINT_UTILS_H_
