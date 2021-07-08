//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Thread.h : Defines the Thread class which represents a global EGL thread.

#ifndef LIBANGLE_THREAD_H_
#define LIBANGLE_THREAD_H_

#include <EGL/egl.h>

#include "libANGLE/Debug.h"

namespace gl
{
class Context;
}  // namespace gl

namespace egl
{
class Error;
class Debug;
class Display;
class Surface;

class Thread : public LabeledObject
{
  public:
    Thread();

    void setLabel(EGLLabelKHR label) override;
    EGLLabelKHR getLabel() const override;

    void setSuccess();
    void setError(const Error &error,
                  const Debug *debug,
                  const char *command,
                  const LabeledObject *object);
    EGLint getError() const;

    void setAPI(EGLenum api);
    EGLenum getAPI() const;

    void setCurrent(gl::Context *context);
    Surface *getCurrentDrawSurface() const;
    Surface *getCurrentReadSurface() const;
    gl::Context *getContext() const;
    gl::Context *getValidContext() const;
    Display *getDisplay() const;

  private:
    EGLLabelKHR mLabel;
    EGLint mError;
    EGLenum mAPI;
    gl::Context *mContext;
};

}  // namespace egl

#endif  // LIBANGLE_THREAD_H_
