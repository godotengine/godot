//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_buffer_pool.h:
//    Defines class interface for BufferPool, managing a pool of mtl::Buffer
//

#ifndef LIBANGLE_RENDERER_METAL_MTL_BUFFER_POOL_H_
#define LIBANGLE_RENDERER_METAL_MTL_BUFFER_POOL_H_

#include "libANGLE/renderer/metal/mtl_resources.h"

#include <deque>

namespace rx
{

class ContextMtl;

namespace mtl
{

enum class BufferPoolMemPolicy
{
    AlwaysSharedMem,  // Always allocate buffer in shared memory, useful for dynamic small buffer
    AlwaysGPUMem,     // Always allocate buffer in GPU dedicated memory.
    Auto,             // Auto allocate buffer in shared memory if it is small. GPU otherwise.
};

// A buffer pool is conceptually an infinitely long buffer. Each time you write to the buffer,
// you will always write to a previously unused portion. After a series of writes, you must flush
// the buffer data to the device. Buffer lifetime currently assumes that each new allocation will
// last as long or longer than each prior allocation.
//
// Buffer pool is used to implement a variety of data streaming operations in Metal, such
// as for immediate vertex array and element array data, and other dynamic data.
//
// Internally buffer pool keeps a collection of mtl::Buffer. When we write past the end of a
// currently active mtl::Buffer we keep it until it is no longer in use. We then mark it available
// for future allocations in a free list.
class BufferPool
{
  public:
    // - alwaysAllocNewBuffer=true will always allocate new buffer or reuse free buffer on
    // allocate(), regardless of whether current buffer still has unused portion or not.
    // - alwaysUseSharedMem: indicate the allocated buffers should be in shared memory or not.
    // If this flag is false. Buffer pool will automatically use shared mem if buffer size is small.
    BufferPool(bool alwaysAllocNewBuffer = false);
    BufferPool(bool alwaysAllocNewBuffer, BufferPoolMemPolicy memPolicy);
    ~BufferPool();

    // Init is called after the buffer creation so that the alignment can be specified later.
    void initialize(Context *context, size_t initialSize, size_t alignment, size_t maxBuffers = 0);
    // Calling this without initialize() will have same effect as calling initialize().
    // If called after initialize(), the old pending buffers will be flushed and might be re-used if
    // their size are big enough for the requested initialSize parameter.
    void reset(ContextMtl *contextMtl, size_t initialSize, size_t alignment, size_t maxBuffers = 0);

    // This call will allocate a new region at the end of the buffer. It internally may trigger
    // a new buffer to be created (which is returned in the optional parameter
    // `newBufferAllocatedOut`).  The new region will be in the returned buffer at given offset. If
    // a memory pointer is given, the buffer will be automatically map()ed.
    angle::Result allocate(ContextMtl *contextMtl,
                           size_t sizeInBytes,
                           uint8_t **ptrOut            = nullptr,
                           BufferRef *bufferOut        = nullptr,
                           size_t *offsetOut           = nullptr,
                           bool *newBufferAllocatedOut = nullptr);

    // After a sequence of CPU writes, call commit to ensure the data is visible to the device.
    angle::Result commit(ContextMtl *contextMtl);

    // This releases all the buffers that have been allocated since this was last called.
    void releaseInFlightBuffers(ContextMtl *contextMtl);

    // This frees resources immediately.
    void destroy(ContextMtl *contextMtl);

    const BufferRef &getCurrentBuffer() { return mBuffer; }

    size_t getAlignment() { return mAlignment; }
    void updateAlignment(Context *context, size_t alignment);

    size_t getMaxBuffers() const { return mMaxBuffers; }

    // Set whether allocate() will always allocate new buffer or attempting to append to previous
    // buffer or not. Default is false.
    void setAlwaysAllocateNewBuffer(bool e) { mAlwaysAllocateNewBuffer = e; }

    void setMemoryPolicy(BufferPoolMemPolicy policy) { mMemPolicy = policy; }

    // Set all subsequent allocated buffers should always use shared memory
    void setAlwaysUseSharedMem() { setMemoryPolicy(BufferPoolMemPolicy::AlwaysSharedMem); }

    // Set all subsequent allocated buffers should always use GPU memory
    void setAlwaysUseGPUMem() { setMemoryPolicy(BufferPoolMemPolicy::AlwaysGPUMem); }

  private:
    bool shouldAllocateInSharedMem(ContextMtl *contextMtl) const;
    void reset();
    angle::Result allocateNewBuffer(ContextMtl *contextMtl);
    void destroyBufferList(ContextMtl *contextMtl, std::deque<BufferRef> *buffers);
    angle::Result finalizePendingBuffer(ContextMtl *contextMtl);

    size_t mInitialSize;
    BufferRef mBuffer;
    uint32_t mNextAllocationOffset;
    uint32_t mLastFlushOffset;
    size_t mSize;
    size_t mAlignment;

    std::deque<BufferRef> mInFlightBuffers;
    std::deque<BufferRef> mBufferFreeList;

    size_t mBuffersAllocated;
    size_t mMaxBuffers;
    BufferPoolMemPolicy mMemPolicy = BufferPoolMemPolicy::Auto;
    bool mAlwaysAllocateNewBuffer;
};

}  // namespace mtl
}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_MTL_BUFFER_POOL_H_ */
