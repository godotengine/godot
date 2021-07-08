//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// TransformFeedbackImpl.h: Defines the abstract rx::TransformFeedbackImpl class.

#ifndef LIBANGLE_RENDERER_TRANSFORMFEEDBACKIMPL_H_
#define LIBANGLE_RENDERER_TRANSFORMFEEDBACKIMPL_H_

#include "common/angleutils.h"
#include "libANGLE/TransformFeedback.h"

namespace rx
{

class TransformFeedbackImpl : angle::NonCopyable
{
  public:
    TransformFeedbackImpl(const gl::TransformFeedbackState &state) : mState(state) {}
    virtual ~TransformFeedbackImpl() {}

    virtual angle::Result begin(const gl::Context *context, gl::PrimitiveMode primitiveMode) = 0;
    virtual angle::Result end(const gl::Context *context)                                    = 0;
    virtual angle::Result pause(const gl::Context *context)                                  = 0;
    virtual angle::Result resume(const gl::Context *context)                                 = 0;

    virtual angle::Result bindIndexedBuffer(
        const gl::Context *context,
        size_t index,
        const gl::OffsetBindingPointer<gl::Buffer> &binding) = 0;

  protected:
    const gl::TransformFeedbackState &mState;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_TRANSFORMFEEDBACKIMPL_H_
