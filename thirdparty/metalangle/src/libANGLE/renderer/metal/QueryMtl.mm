//
// Copyright (c) 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// QueryMtl.mm:
//    Defines the class interface for QueryMtl, implementing QueryImpl.
//

#include "libANGLE/renderer/metal/QueryMtl.h"

#include "libANGLE/renderer/metal/ContextMtl.h"

namespace rx
{
QueryMtl::QueryMtl(gl::QueryType type) : QueryImpl(type) {}

QueryMtl::~QueryMtl() {}

void QueryMtl::onDestroy(const gl::Context *context)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);
    if (!getAllocatedVisibilityOffsets().empty())
    {
        contextMtl->onOcclusionQueryDestroyed(context, this);
    }
    mVisibilityResultBuffer = nullptr;
}

angle::Result QueryMtl::begin(const gl::Context *context)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);
    switch (getType())
    {
        case gl::QueryType::AnySamples:
        case gl::QueryType::AnySamplesConservative:
            if (!mVisibilityResultBuffer)
            {
                // Allocate buffer
                ANGLE_TRY(mtl::Buffer::MakeBuffer(contextMtl, mtl::kOcclusionQueryResultSize,
                                                  nullptr, &mVisibilityResultBuffer));

                ANGLE_MTL_OBJC_SCOPE
                {
                    mVisibilityResultBuffer->get().label =
                        [NSString stringWithFormat:@"QueryMtl=%p", this];
                }
            }

            ANGLE_TRY(contextMtl->onOcclusionQueryBegan(context, this));
            break;
        case gl::QueryType::TransformFeedbackPrimitivesWritten:
            mTransformFeedbackPrimitivesDrawn = 0;
            break;
        default:
            UNIMPLEMENTED();
            break;
    }

    return angle::Result::Continue;
}
angle::Result QueryMtl::end(const gl::Context *context)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);
    switch (getType())
    {
        case gl::QueryType::AnySamples:
        case gl::QueryType::AnySamplesConservative:
            contextMtl->onOcclusionQueryEnded(context, this);
            break;
        case gl::QueryType::TransformFeedbackPrimitivesWritten:
            // There could be transform feedback in progress, so add the primitives drawn so far
            // from the current transform feedback object.
            onTransformFeedbackEnd(context);
            break;
        default:
            UNIMPLEMENTED();
            break;
    }
    return angle::Result::Continue;
}
angle::Result QueryMtl::queryCounter(const gl::Context *context)
{
    UNIMPLEMENTED();
    return angle::Result::Continue;
}

template <typename T>
angle::Result QueryMtl::waitAndGetResult(const gl::Context *context, T *params)
{
    ASSERT(params);
    ContextMtl *contextMtl = mtl::GetImpl(context);
    switch (getType())
    {
        case gl::QueryType::AnySamples:
        case gl::QueryType::AnySamplesConservative:
        {
            ASSERT(mVisibilityResultBuffer);
            if (mVisibilityResultBuffer->hasPendingWorks(contextMtl))
            {
                contextMtl->flushCommandBufer();
            }
            // map() will wait for the pending GPU works to finish
            const uint8_t *visibilityResultBytes = mVisibilityResultBuffer->mapReadOnly(contextMtl);
            uint64_t queryResult;
            memcpy(&queryResult, visibilityResultBytes, sizeof(queryResult));
            mVisibilityResultBuffer->unmap(contextMtl);

            *params = queryResult ? GL_TRUE : GL_FALSE;
        }
        break;
        case gl::QueryType::TransformFeedbackPrimitivesWritten:
            *params = static_cast<T>(mTransformFeedbackPrimitivesDrawn);
            break;
        default:
            UNIMPLEMENTED();
            break;
    }
    return angle::Result::Continue;
}

angle::Result QueryMtl::isResultAvailable(const gl::Context *context, bool *available)
{
    ASSERT(available);
    ContextMtl *contextMtl = mtl::GetImpl(context);
    // glGetQueryObjectuiv implicitly flush any pending works related to the query
    switch (getType())
    {
        case gl::QueryType::AnySamples:
        case gl::QueryType::AnySamplesConservative:
            ASSERT(mVisibilityResultBuffer);
            if (mVisibilityResultBuffer->hasPendingWorks(contextMtl))
            {
                contextMtl->flushCommandBufer();
            }

            *available = !mVisibilityResultBuffer->isBeingUsedByGPU(contextMtl);
            break;
        case gl::QueryType::TransformFeedbackPrimitivesWritten:
            *available = true;
            break;
        default:
            UNIMPLEMENTED();
            break;
    }
    return angle::Result::Continue;
}

angle::Result QueryMtl::getResult(const gl::Context *context, GLint *params)
{
    return waitAndGetResult(context, params);
}
angle::Result QueryMtl::getResult(const gl::Context *context, GLuint *params)
{
    return waitAndGetResult(context, params);
}
angle::Result QueryMtl::getResult(const gl::Context *context, GLint64 *params)
{
    return waitAndGetResult(context, params);
}
angle::Result QueryMtl::getResult(const gl::Context *context, GLuint64 *params)
{
    return waitAndGetResult(context, params);
}

void QueryMtl::resetVisibilityResult(ContextMtl *contextMtl)
{
    // Occlusion query buffer must be allocated in QueryMtl::begin
    ASSERT(mVisibilityResultBuffer);

    // Fill the query's buffer with zeros
    auto blitEncoder = contextMtl->getBlitCommandEncoder();
    blitEncoder->fillBuffer(mVisibilityResultBuffer, NSMakeRange(0, mtl::kOcclusionQueryResultSize),
                            0);
    mVisibilityResultBuffer->syncContent(contextMtl, blitEncoder);
}

void QueryMtl::onTransformFeedbackEnd(const gl::Context *context)
{
    gl::TransformFeedback *transformFeedback = context->getState().getCurrentTransformFeedback();
    if (transformFeedback)
    {
        mTransformFeedbackPrimitivesDrawn += transformFeedback->getPrimitivesDrawn();
    }
}

}
