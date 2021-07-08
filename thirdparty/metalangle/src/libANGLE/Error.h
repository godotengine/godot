//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Error.h: Defines the egl::Error and gl::Error classes which encapsulate API errors
// and optional error messages.

#ifndef LIBANGLE_ERROR_H_
#define LIBANGLE_ERROR_H_

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "angle_gl.h"
#include "common/angleutils.h"
#include "common/debug.h"

#include <memory>
#include <ostream>
#include <string>

namespace angle
{
template <typename ErrorT, typename ErrorBaseT, ErrorBaseT NoErrorVal, typename CodeT, CodeT EnumT>
class ErrorStreamBase : angle::NonCopyable
{
  public:
    ErrorStreamBase() : mID(EnumT) {}
    ErrorStreamBase(GLuint id) : mID(id) {}

    template <typename T>
    ErrorStreamBase &operator<<(T value)
    {
        mErrorStream << value;
        return *this;
    }

    operator ErrorT() { return ErrorT(EnumT, mID, mErrorStream.str()); }

  private:
    GLuint mID;
    std::ostringstream mErrorStream;
};
}  // namespace angle

namespace egl
{
class Error;
}  // namespace egl

namespace egl
{

class ANGLE_NO_DISCARD Error final
{
  public:
    explicit inline Error(EGLint errorCode);
    Error(EGLint errorCode, std::string &&message);
    Error(EGLint errorCode, EGLint id, std::string &&message);
    inline Error(const Error &other);
    inline Error(Error &&other);
    inline ~Error() = default;

    inline Error &operator=(const Error &other);
    inline Error &operator=(Error &&other);

    inline EGLint getCode() const;
    inline EGLint getID() const;
    inline bool isError() const;

    const std::string &getMessage() const;

    static inline Error NoError();

  private:
    void createMessageString() const;

    friend std::ostream &operator<<(std::ostream &os, const Error &err);

    EGLint mCode;
    EGLint mID;
    mutable std::unique_ptr<std::string> mMessage;
};

namespace priv
{

template <EGLint EnumT>
using ErrorStream = angle::ErrorStreamBase<Error, EGLint, EGL_SUCCESS, EGLint, EnumT>;

}  // namespace priv

using EglBadAccess         = priv::ErrorStream<EGL_BAD_ACCESS>;
using EglBadAlloc          = priv::ErrorStream<EGL_BAD_ALLOC>;
using EglBadAttribute      = priv::ErrorStream<EGL_BAD_ATTRIBUTE>;
using EglBadConfig         = priv::ErrorStream<EGL_BAD_CONFIG>;
using EglBadContext        = priv::ErrorStream<EGL_BAD_CONTEXT>;
using EglBadCurrentSurface = priv::ErrorStream<EGL_BAD_CURRENT_SURFACE>;
using EglBadDevice         = priv::ErrorStream<EGL_BAD_DEVICE_EXT>;
using EglBadDisplay        = priv::ErrorStream<EGL_BAD_DISPLAY>;
using EglBadMatch          = priv::ErrorStream<EGL_BAD_MATCH>;
using EglBadNativeWindow   = priv::ErrorStream<EGL_BAD_NATIVE_WINDOW>;
using EglBadParameter      = priv::ErrorStream<EGL_BAD_PARAMETER>;
using EglBadState          = priv::ErrorStream<EGL_BAD_STATE_KHR>;
using EglBadStream         = priv::ErrorStream<EGL_BAD_STREAM_KHR>;
using EglBadSurface        = priv::ErrorStream<EGL_BAD_SURFACE>;
using EglContextLost       = priv::ErrorStream<EGL_CONTEXT_LOST>;
using EglNotInitialized    = priv::ErrorStream<EGL_NOT_INITIALIZED>;

inline Error NoError()
{
    return Error::NoError();
}

}  // namespace egl

#define ANGLE_CONCAT1(x, y) x##y
#define ANGLE_CONCAT2(x, y) ANGLE_CONCAT1(x, y)
#define ANGLE_LOCAL_VAR ANGLE_CONCAT2(_localVar, __LINE__)

#define ANGLE_TRY_TEMPLATE(EXPR, FUNC)                \
    do                                                \
    {                                                 \
        auto ANGLE_LOCAL_VAR = EXPR;                  \
        if (ANGLE_UNLIKELY(IsError(ANGLE_LOCAL_VAR))) \
        {                                             \
            FUNC(ANGLE_LOCAL_VAR);                    \
        }                                             \
    } while (0)

#define ANGLE_RETURN(X) return X;
#define ANGLE_TRY(EXPR) ANGLE_TRY_TEMPLATE(EXPR, ANGLE_RETURN)

// TODO(jmadill): Remove after EGL error refactor. http://anglebug.com/3041
#define ANGLE_SWALLOW_ERR(EXPR)                                       \
    do                                                                \
    {                                                                 \
        auto ANGLE_LOCAL_VAR = EXPR;                                  \
        if (ANGLE_UNLIKELY(IsError(ANGLE_LOCAL_VAR)))                 \
        {                                                             \
            ERR() << "Unhandled internal error: " << ANGLE_LOCAL_VAR; \
        }                                                             \
    } while (0)

#undef ANGLE_LOCAL_VAR
#undef ANGLE_CONCAT2
#undef ANGLE_CONCAT1

#define ANGLE_CHECK(CONTEXT, EXPR, MESSAGE, ERROR)                                    \
    do                                                                                \
    {                                                                                 \
        if (ANGLE_UNLIKELY(!(EXPR)))                                                  \
        {                                                                             \
            CONTEXT->handleError(ERROR, MESSAGE, __FILE__, ANGLE_FUNCTION, __LINE__); \
            return angle::Result::Stop;                                               \
        }                                                                             \
    } while (0)

namespace angle
{
// Result signals if calling code should continue running or early exit. A value of Stop can
// either indicate an Error or a non-Error early exit condition such as a detected no-op.
// Incomplete signals special cases that are neither success nor failure but require
// special attention.
enum class ANGLE_NO_DISCARD Result
{
    Continue,
    Stop,
    Incomplete,
};

// TODO(jmadill): Remove this when refactor is complete. http://anglebug.com/3041
egl::Error ResultToEGL(Result result);
}  // namespace angle

// TODO(jmadill): Remove this when refactor is complete. http://anglebug.com/3041
inline bool IsError(angle::Result result)
{
    return result == angle::Result::Stop;
}

// TODO(jmadill): Remove this when refactor is complete. http://anglebug.com/3041
inline bool IsError(const egl::Error &err)
{
    return err.isError();
}

#include "Error.inc"

#endif  // LIBANGLE_ERROR_H_
