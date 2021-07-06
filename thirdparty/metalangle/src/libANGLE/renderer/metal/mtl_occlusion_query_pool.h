//
// Copyright (c) 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_occlusion_query_pool: A pool for allocating visibility query within
// one render pass.
//

#ifndef LIBANGLE_RENDERER_METAL_MTL_OCCLUSION_QUERY_POOL_H_
#define LIBANGLE_RENDERER_METAL_MTL_OCCLUSION_QUERY_POOL_H_

#include <vector>

#include "libANGLE/Context.h"
#include "libANGLE/renderer/metal/mtl_common.h"
#include "libANGLE/renderer/metal/mtl_resources.h"

namespace rx
{

class ContextMtl;
class QueryMtl;

namespace mtl
{

class OcclusionQueryPool
{
  public:
    OcclusionQueryPool();
    ~OcclusionQueryPool();

    angle::Result initialize(ContextMtl *contextMtl);
    void destroy(ContextMtl *contextMtl);

    bool canAllocateQueryOffset() const;

    angle::Result reserveForMaxPossibleQueryOffsets(ContextMtl *contextMtl);

    // Allocate an offset in visibility buffer for a query in a render pass.
    // - clearOldValue = true, if the old value of query will be cleared before combining in the
    // visibility resolve pass. This flag is only allowed to be false for the first allocation of
    // the render pass or the query that already has an allocated offset.
    // Note: a query might have more than one allocated offset. They will be combined in the final
    // step.
    angle::Result allocateQueryOffset(ContextMtl *contextMtl, QueryMtl *query, bool clearOldValue);
    // Deallocate all offsets used for a query.
    void deallocateQueryOffset(ContextMtl *contextMtl, QueryMtl *query);
    // Retrieve a buffer that will contain the visibility results of all allocated queries for
    // a render pass
    const BufferRef &getRenderPassVisibilityPoolBuffer() const { return mRenderPassResultsPool; }
    size_t getNumRenderPassAllocatedQueries() const { return mAllocatedQueries.size(); }
    // This function is called at the end of render pass
    void resolveVisibilityResults(ContextMtl *contextMtl);

  private:
    // Buffer to hold the visibility results for current render pass
    BufferRef mRenderPassResultsPool;

    size_t mMaxRenderPassResultsPoolSize = 0;

    // List of allocated queries per render pass
    std::vector<QueryMtl *> mAllocatedQueries;

    bool mResetFirstQuery = false;
};

}  // namespace mtl
}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_MTL_OCCLUSION_QUERY_POOL_H_ */
