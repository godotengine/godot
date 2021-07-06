//
// Copyright (c) 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_occlusion_query_pool: A visibility pool for allocating visibility query within
// one render pass.
//

#include "libANGLE/renderer/metal/mtl_occlusion_query_pool.h"

#include "libANGLE/renderer/metal/ContextMtl.h"
#include "libANGLE/renderer/metal/DisplayMtl.h"
#include "libANGLE/renderer/metal/QueryMtl.h"

namespace rx
{
namespace mtl
{

// OcclusionQueryPool implementation
OcclusionQueryPool::OcclusionQueryPool() {}
OcclusionQueryPool::~OcclusionQueryPool() {}

angle::Result OcclusionQueryPool::initialize(ContextMtl *contextMtl)
{
    // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
    if (contextMtl->getDisplay()->supportEitherGPUFamily(7, 1))
    {
        mMaxRenderPassResultsPoolSize = 256 * 1024;
    }
    else
    {
        mMaxRenderPassResultsPoolSize = 65528;
    }

    return angle::Result::Continue;
}

void OcclusionQueryPool::destroy(ContextMtl *contextMtl)
{
    mRenderPassResultsPool = nullptr;
    for (QueryMtl *allocatedQuery : mAllocatedQueries)
    {
        if (!allocatedQuery)
        {
            continue;
        }
        allocatedQuery->clearAllocatedVisibilityOffsets();
    }
    mAllocatedQueries.clear();
}

bool OcclusionQueryPool::canAllocateQueryOffset() const
{
    if (!mRenderPassResultsPool)
    {
        return false;
    }
    uint32_t currentOffset =
        static_cast<uint32_t>(mAllocatedQueries.size()) * kOcclusionQueryResultSize;
    return currentOffset + kOcclusionQueryResultSize <= mMaxRenderPassResultsPoolSize;
}

angle::Result OcclusionQueryPool::reserveForMaxPossibleQueryOffsets(ContextMtl *contextMtl)
{
    if (mRenderPassResultsPool && mRenderPassResultsPool->size() >= mMaxRenderPassResultsPoolSize)
    {
        return angle::Result::Continue;
    }

    ANGLE_TRY(Buffer::MakeBuffer(contextMtl, MTLResourceStorageModePrivate,
                                 mMaxRenderPassResultsPoolSize, nullptr, &mRenderPassResultsPool));
    mRenderPassResultsPool->get().label = @"OcclusionQueryPool";
    return angle::Result::Continue;
}

angle::Result OcclusionQueryPool::allocateQueryOffset(ContextMtl *contextMtl,
                                                      QueryMtl *query,
                                                      bool clearOldValue)
{
    // Only query that already has allocated offset or first query of the render pass is allowed to
    // keep old value. Other queries must be reset to zero before counting the samples visibility in
    // draw calls.
    ASSERT(clearOldValue || mAllocatedQueries.empty() ||
           !query->getAllocatedVisibilityOffsets().empty());

    ASSERT(canAllocateQueryOffset());

    uint32_t currentOffset =
        static_cast<uint32_t>(mAllocatedQueries.size()) * kOcclusionQueryResultSize;

    if (clearOldValue)
    {
        // If old value is not needed, deallocate any offset previously allocated for this query.
        deallocateQueryOffset(contextMtl, query);
    }
    if (query->getAllocatedVisibilityOffsets().empty())
    {
        mAllocatedQueries.push_back(query);
        query->setFirstAllocatedVisibilityOffset(currentOffset);
    }
    else
    {
        // Additional offset allocated for a query is only allowed if it is a continuous region.
        ASSERT(currentOffset ==
               query->getAllocatedVisibilityOffsets().back() + kOcclusionQueryResultSize);
        // Just reserve an empty slot in the allocated query array
        mAllocatedQueries.push_back(nullptr);
        query->addAllocatedVisibilityOffset();
    }

    if (currentOffset == 0)
    {
        mResetFirstQuery = clearOldValue;
        if (!clearOldValue && !contextMtl->getDisplay()->getFeatures().allowBufferReadWrite.enabled)
        {
            // If old value of first query needs to be retained and device doesn't support buffer
            // read-write, we need an additional offset to store the old value of the query.
            return allocateQueryOffset(contextMtl, query, false);
        }
    }

    return angle::Result::Continue;
}

void OcclusionQueryPool::deallocateQueryOffset(ContextMtl *contextMtl, QueryMtl *query)
{
    if (query->getAllocatedVisibilityOffsets().empty())
    {
        return;
    }

    mAllocatedQueries[query->getAllocatedVisibilityOffsets().front() / kOcclusionQueryResultSize] =
        nullptr;
    query->clearAllocatedVisibilityOffsets();
}

void OcclusionQueryPool::resolveVisibilityResults(ContextMtl *contextMtl)
{
    if (mAllocatedQueries.empty())
    {
        return;
    }

    RenderUtils &utils              = contextMtl->getDisplay()->getUtils();
    BlitCommandEncoder *blitEncoder = nullptr;
    // Combine the values stored in the offsets allocated for first query
    if (mAllocatedQueries[0])
    {
        const BufferRef &dstBuf = mAllocatedQueries[0]->getVisibilityResultBuffer();
        const VisibilityBufferOffsetsMtl &allocatedOffsets =
            mAllocatedQueries[0]->getAllocatedVisibilityOffsets();
        if (!mResetFirstQuery &&
            !contextMtl->getDisplay()->getFeatures().allowBufferReadWrite.enabled)
        {
            // If we cannot read and write to the same buffer in shader. We need to copy the old
            // value of first query to first offset allocated for it.
            blitEncoder = contextMtl->getBlitCommandEncoder();
            blitEncoder->copyBuffer(dstBuf, 0, mRenderPassResultsPool, allocatedOffsets.front(),
                                    kOcclusionQueryResultSize);
            utils.combineVisibilityResult(contextMtl, false, allocatedOffsets,
                                          mRenderPassResultsPool, dstBuf);
        }
        else
        {
            utils.combineVisibilityResult(contextMtl, !mResetFirstQuery, allocatedOffsets,
                                          mRenderPassResultsPool, dstBuf);
        }
    }

    // Combine the values stored in the offsets allocated for each of the remaining queries
    for (size_t i = 1; i < mAllocatedQueries.size(); ++i)
    {
        QueryMtl *query = mAllocatedQueries[i];
        if (!query)
        {
            continue;
        }

        const BufferRef &dstBuf = mAllocatedQueries[i]->getVisibilityResultBuffer();
        const VisibilityBufferOffsetsMtl &allocatedOffsets =
            mAllocatedQueries[i]->getAllocatedVisibilityOffsets();
        utils.combineVisibilityResult(contextMtl, false, allocatedOffsets, mRenderPassResultsPool,
                                      dstBuf);
    }

    // Request synchronization and cleanup
    blitEncoder = contextMtl->getBlitCommandEncoder();
    for (size_t i = 0; i < mAllocatedQueries.size(); ++i)
    {
        QueryMtl *query = mAllocatedQueries[i];
        if (!query)
        {
            continue;
        }

        const BufferRef &dstBuf = mAllocatedQueries[i]->getVisibilityResultBuffer();

        dstBuf->syncContent(contextMtl, blitEncoder);

        query->clearAllocatedVisibilityOffsets();
    }

    mAllocatedQueries.clear();
}

}
}
