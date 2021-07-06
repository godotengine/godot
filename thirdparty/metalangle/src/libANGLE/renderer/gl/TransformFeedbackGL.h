//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// TransformFeedbackGL.h: Defines the class interface for TransformFeedbackGL.

#ifndef LIBANGLE_RENDERER_GL_TRANSFORMFEEDBACKGL_H_
#define LIBANGLE_RENDERER_GL_TRANSFORMFEEDBACKGL_H_

#include "libANGLE/renderer/TransformFeedbackImpl.h"

namespace rx
{

class FunctionsGL;
class StateManagerGL;

class TransformFeedbackGL : public TransformFeedbackImpl
{
  public:
    TransformFeedbackGL(const gl::TransformFeedbackState &state,
                        const FunctionsGL *functions,
                        StateManagerGL *stateManager);
    ~TransformFeedbackGL() override;

    angle::Result begin(const gl::Context *context, gl::PrimitiveMode primitiveMode) override;
    angle::Result end(const gl::Context *context) override;
    angle::Result pause(const gl::Context *context) override;
    angle::Result resume(const gl::Context *context) override;

    angle::Result bindIndexedBuffer(const gl::Context *context,
                                    size_t index,
                                    const gl::OffsetBindingPointer<gl::Buffer> &binding) override;

    GLuint getTransformFeedbackID() const;

    void syncActiveState(const gl::Context *context,
                         bool active,
                         gl::PrimitiveMode primitiveMode) const;
    void syncPausedState(bool paused) const;

  private:
    const FunctionsGL *mFunctions;
    StateManagerGL *mStateManager;

    GLuint mTransformFeedbackID;

    mutable bool mIsActive;
    mutable bool mIsPaused;
    mutable GLuint mActiveProgram;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_TRANSFORMFEEDBACKGL_H_
