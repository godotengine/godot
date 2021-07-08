//
// Copyright (c) 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// QueryMtl.h:
//    Defines the class interface for QueryMtl, implementing QueryImpl.
//

#ifndef LIBANGLE_RENDERER_METAL_QUERYMTL_H_
#define LIBANGLE_RENDERER_METAL_QUERYMTL_H_

#include "libANGLE/renderer/QueryImpl.h"
#include "libANGLE/renderer/metal/mtl_common.h"
#include "libANGLE/renderer/metal/mtl_resources.h"

namespace rx
{

class ContextMtl;

// The class represents offset(s) allocated in the visiblity buffer for an occlusion query.
// See doc/OcclusionQuery.md.
// An occlusion query might have more than one offsets allocated, but all of them must be adjacent
// to each other. Multiple offsets typically allocated when the query is paused and resumed during
// viewport clear emulation with draw operations. In such case, Metal doesn't allow an offset to
// be reused in a render pass, hence multiple offsets will be allocated, and their values will
// be accumulated.
class VisibilityBufferOffsetsMtl
{
  public:
    void clear() { mStartOffset = mNumOffsets = 0; }
    bool empty() const { return mNumOffsets == 0; }
    uint32_t size() const { return mNumOffsets; }

    // Return last offset
    uint32_t back() const
    {
        ASSERT(!empty());
        return mStartOffset + (mNumOffsets - 1) * mtl::kOcclusionQueryResultSize;
    }

    uint32_t front() const
    {
        ASSERT(!empty());
        return mStartOffset;
    }

    void setStartOffset(uint32_t offset)
    {
        ASSERT(empty());
        mStartOffset = offset;
        mNumOffsets  = 1;
    }
    void addOffset()
    {
        ASSERT(!empty());
        mNumOffsets++;
    }

  private:
    uint32_t mStartOffset = 0;
    uint32_t mNumOffsets  = 0;
};

class QueryMtl : public QueryImpl
{
  public:
    QueryMtl(gl::QueryType type);
    ~QueryMtl() override;

    void onDestroy(const gl::Context *context) override;

    angle::Result begin(const gl::Context *context) override;
    angle::Result end(const gl::Context *context) override;
    angle::Result queryCounter(const gl::Context *context) override;
    angle::Result getResult(const gl::Context *context, GLint *params) override;
    angle::Result getResult(const gl::Context *context, GLuint *params) override;
    angle::Result getResult(const gl::Context *context, GLint64 *params) override;
    angle::Result getResult(const gl::Context *context, GLuint64 *params) override;
    angle::Result isResultAvailable(const gl::Context *context, bool *available) override;

    // Get allocated offsets in the render pass's occlusion query pool.
    const VisibilityBufferOffsetsMtl &getAllocatedVisibilityOffsets() const
    {
        return mVisibilityBufferOffsets;
    }
    // Set first allocated offset in the render pass's occlusion query pool.
    void setFirstAllocatedVisibilityOffset(uint32_t off)
    {
        mVisibilityBufferOffsets.setStartOffset(off);
    }
    // Add more offset allocated for the occlusion query
    void addAllocatedVisibilityOffset() { mVisibilityBufferOffsets.addOffset(); }

    void clearAllocatedVisibilityOffsets() { mVisibilityBufferOffsets.clear(); }
    // Returns the buffer containing the final occlusion query result.
    const mtl::BufferRef &getVisibilityResultBuffer() const { return mVisibilityResultBuffer; }
    // Reset the occlusion query result stored in buffer to zero
    void resetVisibilityResult(ContextMtl *contextMtl);

    void onTransformFeedbackEnd(const gl::Context *context);

  private:
    template <typename T>
    angle::Result waitAndGetResult(const gl::Context *context, T *params);

    // List of offsets in the render pass's occlusion query pool buffer allocated for this query
    VisibilityBufferOffsetsMtl mVisibilityBufferOffsets;
    mtl::BufferRef mVisibilityResultBuffer;

    size_t mTransformFeedbackPrimitivesDrawn = 0;
};

}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_QUERYMTL_H_ */
