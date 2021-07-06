//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ClearMultiviewGL:
//   A helper for clearing multiview side-by-side and layered framebuffers.
//

#ifndef LIBANGLE_RENDERER_GL_CLEARMULTIVIEWGL_H_
#define LIBANGLE_RENDERER_GL_CLEARMULTIVIEWGL_H_

#include "angle_gl.h"
#include "libANGLE/Error.h"
#include "libANGLE/angletypes.h"

namespace gl
{
class FramebufferState;
}  // namespace gl

namespace rx
{
class FunctionsGL;
class StateManagerGL;

class ClearMultiviewGL : angle::NonCopyable
{
  public:
    // Enum containing the different types of Clear* commands.
    enum class ClearCommandType
    {
        Clear,
        ClearBufferfv,
        ClearBufferuiv,
        ClearBufferiv,
        ClearBufferfi
    };

  public:
    ClearMultiviewGL(const FunctionsGL *functions, StateManagerGL *stateManager);
    ~ClearMultiviewGL();

    ClearMultiviewGL(const ClearMultiviewGL &rht) = delete;
    ClearMultiviewGL &operator=(const ClearMultiviewGL &rht) = delete;
    ClearMultiviewGL(ClearMultiviewGL &&rht)                 = delete;
    ClearMultiviewGL &operator=(ClearMultiviewGL &&rht) = delete;

    void clearMultiviewFBO(const gl::FramebufferState &state,
                           const gl::Rectangle &scissorBase,
                           ClearCommandType clearCommandType,
                           GLbitfield mask,
                           GLenum buffer,
                           GLint drawbuffer,
                           const uint8_t *values,
                           GLfloat depth,
                           GLint stencil);
    void initializeResources();

  private:
    void attachTextures(const gl::FramebufferState &state, int layer);
    void detachTextures(const gl::FramebufferState &state);
    void clearLayeredFBO(const gl::FramebufferState &state,
                         ClearCommandType clearCommandType,
                         GLbitfield mask,
                         GLenum buffer,
                         GLint drawbuffer,
                         const uint8_t *values,
                         GLfloat depth,
                         GLint stencil);
    void genericClear(ClearCommandType clearCommandType,
                      GLbitfield mask,
                      GLenum buffer,
                      GLint drawbuffer,
                      const uint8_t *values,
                      GLfloat depth,
                      GLint stencil);

    const FunctionsGL *mFunctions;
    StateManagerGL *mStateManager;

    GLuint mFramebuffer;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_CLEARMULTIVIEWGL_H_