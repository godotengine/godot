//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Context.inl.h: Defines inline functions of gl::Context class
// Has to be included after libANGLE/Context.h when using one
// of the defined functions

#ifndef LIBANGLE_CONTEXT_INL_H_
#define LIBANGLE_CONTEXT_INL_H_

#include "libANGLE/GLES1Renderer.h"
#include "libANGLE/renderer/ContextImpl.h"

#define ANGLE_HANDLE_ERR(X) \
    (void)(X);              \
    return;
#define ANGLE_CONTEXT_TRY(EXPR) ANGLE_TRY_TEMPLATE(EXPR, ANGLE_HANDLE_ERR)

namespace gl
{
constexpr angle::PackedEnumMap<PrimitiveMode, GLsizei> kMinimumPrimitiveCounts = {{
    {PrimitiveMode::Points, 1},
    {PrimitiveMode::Lines, 2},
    {PrimitiveMode::LineLoop, 2},
    {PrimitiveMode::LineStrip, 2},
    {PrimitiveMode::Triangles, 3},
    {PrimitiveMode::TriangleStrip, 3},
    {PrimitiveMode::TriangleFan, 3},
    {PrimitiveMode::LinesAdjacency, 2},
    {PrimitiveMode::LineStripAdjacency, 2},
    {PrimitiveMode::TrianglesAdjacency, 3},
    {PrimitiveMode::TriangleStripAdjacency, 3},
}};

ANGLE_INLINE void MarkTransformFeedbackBufferUsage(const Context *context,
                                                   GLsizei count,
                                                   GLsizei instanceCount)
{
    if (context->getStateCache().isTransformFeedbackActiveUnpaused())
    {
        TransformFeedback *transformFeedback = context->getState().getCurrentTransformFeedback();
        transformFeedback->onVerticesDrawn(context, count, instanceCount);
    }
}

ANGLE_INLINE void MarkShaderStorageBufferUsage(const Context *context)
{
    for (size_t index : context->getStateCache().getActiveShaderStorageBufferIndices())
    {
        gl::Buffer *buffer = context->getState().getIndexedShaderStorageBuffer(index).get();
        if (buffer)
        {
            buffer->onDataChanged();
        }
    }
}

// Return true if the draw is a no-op, else return false.
//  A no-op draw occurs if the count of vertices is less than the minimum required to
//  have a valid primitive for this mode (0 for points, 0-1 for lines, 0-2 for tris).
ANGLE_INLINE bool Context::noopDraw(PrimitiveMode mode, GLsizei count)
{
    return count < kMinimumPrimitiveCounts[mode];
}

ANGLE_INLINE angle::Result Context::syncDirtyBits()
{
    const State::DirtyBits &dirtyBits = mState.getDirtyBits();
    ANGLE_TRY(mImplementation->syncState(this, dirtyBits, mAllDirtyBits));
    mState.clearDirtyBits();
    return angle::Result::Continue;
}

ANGLE_INLINE angle::Result Context::syncDirtyBits(const State::DirtyBits &bitMask)
{
    const State::DirtyBits &dirtyBits = (mState.getDirtyBits() & bitMask);
    ANGLE_TRY(mImplementation->syncState(this, dirtyBits, bitMask));
    mState.clearDirtyBits(dirtyBits);
    return angle::Result::Continue;
}

ANGLE_INLINE angle::Result Context::syncDirtyObjects(const State::DirtyObjects &objectMask)
{
    return mState.syncDirtyObjects(this, objectMask);
}

ANGLE_INLINE angle::Result Context::prepareForDraw(PrimitiveMode mode)
{
    if (mGLES1Renderer)
    {
        ANGLE_TRY(mGLES1Renderer->prepareForDraw(mode, this, &mState));
    }

    ANGLE_TRY(syncDirtyObjects(mDrawDirtyObjects));
    ASSERT(!isRobustResourceInitEnabled() ||
           !mState.getDrawFramebuffer()->hasResourceThatNeedsInit());
    return syncDirtyBits();
}

ANGLE_INLINE void Context::drawArrays(PrimitiveMode mode, GLint first, GLsizei count)
{
    // No-op if count draws no primitives for given mode
    if (noopDraw(mode, count))
    {
        return;
    }

    ANGLE_CONTEXT_TRY(prepareForDraw(mode));
    ANGLE_CONTEXT_TRY(mImplementation->drawArrays(this, mode, first, count));
    MarkTransformFeedbackBufferUsage(this, count, 1);
}

ANGLE_INLINE void Context::drawElements(PrimitiveMode mode,
                                        GLsizei count,
                                        DrawElementsType type,
                                        const void *indices)
{
    // No-op if count draws no primitives for given mode
    if (noopDraw(mode, count))
    {
        return;
    }

    ANGLE_CONTEXT_TRY(prepareForDraw(mode));
    ANGLE_CONTEXT_TRY(mImplementation->drawElements(this, mode, count, type, indices));
}

ANGLE_INLINE void StateCache::onBufferBindingChange(Context *context)
{
    updateBasicDrawStatesError();
    updateBasicDrawElementsError();
}

ANGLE_INLINE void Context::bindBuffer(BufferBinding target, BufferID buffer)
{
    Buffer *bufferObject =
        mState.mBufferManager->checkBufferAllocation(mImplementation.get(), buffer);
    mState.setBufferBinding(this, target, bufferObject);
    mStateCache.onBufferBindingChange(this);
}

}  // namespace gl

#endif  // LIBANGLE_CONTEXT_INL_H_
