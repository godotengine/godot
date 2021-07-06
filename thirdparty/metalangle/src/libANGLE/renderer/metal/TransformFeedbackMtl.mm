//
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// TransformFeedbackMtl.mm:
//    Defines the class interface for TransformFeedbackMtl, implementing TransformFeedbackImpl.
//

#include "libANGLE/renderer/metal/TransformFeedbackMtl.h"

#include "common/debug.h"
#include "libANGLE/Context.h"
#include "libANGLE/Query.h"
#include "libANGLE/renderer/metal/ContextMtl.h"
#include "libANGLE/renderer/metal/QueryMtl.h"

namespace rx
{

TransformFeedbackMtl::TransformFeedbackMtl(const gl::TransformFeedbackState &state)
    : TransformFeedbackImpl(state)
{}

TransformFeedbackMtl::~TransformFeedbackMtl() {}

angle::Result TransformFeedbackMtl::begin(const gl::Context *context,
                                          gl::PrimitiveMode primitiveMode)
{
    mtl::GetImpl(context)->onTransformFeedbackActive(context, this);

    return angle::Result::Continue;
}

angle::Result TransformFeedbackMtl::end(const gl::Context *context)
{
    const gl::State &glState = context->getState();
    gl::Query *transformFeedbackQuery =
        glState.getActiveQuery(gl::QueryType::TransformFeedbackPrimitivesWritten);
    if (transformFeedbackQuery)
    {
        mtl::GetImpl(transformFeedbackQuery)->onTransformFeedbackEnd(context);
    }

    mtl::GetImpl(context)->onTransformFeedbackInactive(context, this);

    return angle::Result::Continue;
}

angle::Result TransformFeedbackMtl::pause(const gl::Context *context)
{
    // When XFB is paused, OpenGL allows XFB buffers to be bound for other purposes. We need to call
    // onTransformFeedbackInactive() to issue a sync.
    mtl::GetImpl(context)->onTransformFeedbackInactive(context, this);

    return angle::Result::Continue;
}

angle::Result TransformFeedbackMtl::resume(const gl::Context *context)
{
    mtl::GetImpl(context)->onTransformFeedbackActive(context, this);

    return angle::Result::Continue;
}

angle::Result TransformFeedbackMtl::bindIndexedBuffer(
    const gl::Context *context,
    size_t index,
    const gl::OffsetBindingPointer<gl::Buffer> &binding)
{
    // Do nothing for now

    return angle::Result::Continue;
}

angle::Result TransformFeedbackMtl::getBufferOffsets(ContextMtl *contextMtl,
                                                     const gl::ProgramState &programState,
                                                     GLint drawCallFirstVertex,
                                                     uint32_t skippedVertices,
                                                     int32_t *offsetsOut)
{
    int64_t verticesDrawn = static_cast<int64_t>(mState.getVerticesDrawn()) + skippedVertices;
    const std::vector<GLsizei> &bufferStrides =
        mState.getBoundProgram()->getTransformFeedbackStrides();
    size_t xfbBufferCount = programState.getTransformFeedbackBufferCount();

    ASSERT(xfbBufferCount > 0);

    for (size_t bufferIndex = 0; bufferIndex < xfbBufferCount; ++bufferIndex)
    {
        const gl::OffsetBindingPointer<gl::Buffer> &bufferBinding =
            mState.getIndexedBuffer(bufferIndex);

        ASSERT((bufferBinding.getOffset() % 4) == 0);

        // Offset the gl_VertexIndex by drawCallFirstVertex
        int64_t drawCallVertexOffset = static_cast<int64_t>(verticesDrawn) - drawCallFirstVertex;

        int64_t writeOffset =
            (bufferBinding.getOffset() + drawCallVertexOffset * bufferStrides[bufferIndex]) /
            static_cast<int64_t>(sizeof(uint32_t));

        offsetsOut[bufferIndex] = static_cast<int32_t>(writeOffset);

        // Check for overflow.
        ANGLE_CHECK_GL_ALLOC(contextMtl, offsetsOut[bufferIndex] == writeOffset);
    }

    return angle::Result::Continue;
}

}  // namespace rx