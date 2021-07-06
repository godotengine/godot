//
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// TransformFeedbackMtl.h:
//    Defines the class interface for TransformFeedbackMtl, implementing TransformFeedbackImpl.
//

#ifndef LIBANGLE_RENDERER_METAL_TRANSFORMFEEDBACKMTL_H_
#define LIBANGLE_RENDERER_METAL_TRANSFORMFEEDBACKMTL_H_

#include "libANGLE/renderer/TransformFeedbackImpl.h"

namespace gl
{
class ProgramState;
}  // namespace gl

namespace rx
{

class ContextMtl;

class TransformFeedbackMtl : public TransformFeedbackImpl
{
  public:
    TransformFeedbackMtl(const gl::TransformFeedbackState &state);
    ~TransformFeedbackMtl() override;

    angle::Result begin(const gl::Context *context, gl::PrimitiveMode primitiveMode) override;
    angle::Result end(const gl::Context *context) override;
    angle::Result pause(const gl::Context *context) override;
    angle::Result resume(const gl::Context *context) override;

    angle::Result bindIndexedBuffer(const gl::Context *context,
                                    size_t index,
                                    const gl::OffsetBindingPointer<gl::Buffer> &binding) override;

    // Params:
    // - drawCallFirstVertex is first vertex used by glDrawArrays*. This is important because
    // gl_VertexIndex is starting from this.
    // - skippedVertices is number of skipped vertices (useful for multiple metal draws per GL draw
    // call).
    angle::Result getBufferOffsets(ContextMtl *contextMtl,
                                   const gl::ProgramState &programState,
                                   GLint drawCallFirstVertex,
                                   uint32_t skippedVertices,
                                   int32_t *offsetsOut);

  private:
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_METAL_TRANSFORMFEEDBACKMTL_H_