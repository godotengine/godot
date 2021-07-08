//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// PoolAlloc.h:
//    Defines the class interface for PoolAllocator and the Allocation
//    class that it uses internally.
//

#ifndef COMMON_POOLALLOC_H_
#define COMMON_POOLALLOC_H_

#if !defined(NDEBUG)
#    define ANGLE_POOL_ALLOC_GUARD_BLOCKS  // define to enable guard block sanity checking
#endif

//
// This header defines an allocator that can be used to efficiently
// allocate a large number of small requests for heap memory, with the
// intention that they are not individually deallocated, but rather
// collectively deallocated at one time.
//
// This simultaneously
//
// * Makes each individual allocation much more efficient; the
//     typical allocation is trivial.
// * Completely avoids the cost of doing individual deallocation.
// * Saves the trouble of tracking down and plugging a large class of leaks.
//
// Individual classes can use this allocator by supplying their own
// new and delete methods.
//

#include <stddef.h>
#include <string.h>
#include <memory>
#include <vector>

#include "angleutils.h"
#include "common/debug.h"

namespace angle
{
// If we are using guard blocks, we must track each individual
// allocation.  If we aren't using guard blocks, these
// never get instantiated, so won't have any impact.
//

class Allocation
{
  public:
    Allocation(size_t size, unsigned char *mem, Allocation *prev = 0)
        : mSize(size), mMem(mem), mPrevAlloc(prev)
    {
// Allocations are bracketed:
//    [allocationHeader][initialGuardBlock][userData][finalGuardBlock]
// This would be cleaner with if (kGuardBlockSize)..., but that
// makes the compiler print warnings about 0 length memsets,
// even with the if() protecting them.
#if defined(ANGLE_POOL_ALLOC_GUARD_BLOCKS)
        memset(preGuard(), kGuardBlockBeginVal, kGuardBlockSize);
        memset(data(), kUserDataFill, mSize);
        memset(postGuard(), kGuardBlockEndVal, kGuardBlockSize);
#endif
    }

    void check() const
    {
        checkGuardBlock(preGuard(), kGuardBlockBeginVal, "before");
        checkGuardBlock(postGuard(), kGuardBlockEndVal, "after");
    }

    void checkAllocList() const;

    // Return total size needed to accommodate user buffer of 'size',
    // plus our tracking data.
    static size_t AllocationSize(size_t size) { return size + 2 * kGuardBlockSize + HeaderSize(); }

    // Offset from surrounding buffer to get to user data buffer.
    static unsigned char *OffsetAllocation(unsigned char *m)
    {
        return m + kGuardBlockSize + HeaderSize();
    }

  private:
    void checkGuardBlock(unsigned char *blockMem, unsigned char val, const char *locText) const;

    // Find offsets to pre and post guard blocks, and user data buffer
    unsigned char *preGuard() const { return mMem + HeaderSize(); }
    unsigned char *data() const { return preGuard() + kGuardBlockSize; }
    unsigned char *postGuard() const { return data() + mSize; }
    size_t mSize;            // size of the user data area
    unsigned char *mMem;     // beginning of our allocation (pts to header)
    Allocation *mPrevAlloc;  // prior allocation in the chain

    static constexpr unsigned char kGuardBlockBeginVal = 0xfb;
    static constexpr unsigned char kGuardBlockEndVal   = 0xfe;
    static constexpr unsigned char kUserDataFill       = 0xcd;
#if defined(ANGLE_POOL_ALLOC_GUARD_BLOCKS)
    static constexpr size_t kGuardBlockSize = 16;
    static constexpr size_t HeaderSize() { return sizeof(Allocation); }
#else
    static constexpr size_t kGuardBlockSize = 0;
    static constexpr size_t HeaderSize() { return 0; }
#endif
};

//
// There are several stacks.  One is to track the pushing and popping
// of the user, and not yet implemented.  The others are simply a
// repositories of free pages or used pages.
//
// Page stacks are linked together with a simple header at the beginning
// of each allocation obtained from the underlying OS.  Multi-page allocations
// are returned to the OS.  Individual page allocations are kept for future
// re-use.
//
// The "page size" used is not, nor must it match, the underlying OS
// page size.  But, having it be about that size or equal to a set of
// pages is likely most optimal.
//
class PoolAllocator : angle::NonCopyable
{
  public:
    static const int kDefaultAlignment = 16;
    //
    // Create PoolAllocator. If alignment is be set to 1 byte then fastAllocate()
    //  function can be used to make allocations with less overhead.
    //
    PoolAllocator(int growthIncrement = 8 * 1024, int allocationAlignment = kDefaultAlignment);

    //
    // Don't call the destructor just to free up the memory, call pop()
    //
    ~PoolAllocator();

    //
    // Call push() to establish a new place to pop memory to.  Does not
    // have to be called to get things started.
    //
    void push();

    //
    // Call pop() to free all memory allocated since the last call to push(),
    // or if no last call to push, frees all memory since first allocation.
    //
    void pop();

    //
    // Call popAll() to free all memory allocated.
    //
    void popAll();

    //
    // Call allocate() to actually acquire memory.  Returns 0 if no memory
    // available, otherwise a properly aligned pointer to 'numBytes' of memory.
    //
    void *allocate(size_t numBytes);

    //
    // Call fastAllocate() for a faster allocate function that does minimal bookkeeping
    // preCondition: Allocator must have been created w/ alignment of 1
    ANGLE_INLINE uint8_t *fastAllocate(size_t numBytes)
    {
#if defined(ANGLE_DISABLE_POOL_ALLOC)
        return reinterpret_cast<uint8_t *>(allocate(numBytes));
#else
        ASSERT(mAlignment == 1);
        // No multi-page allocations
        ASSERT(numBytes <= (mPageSize - mHeaderSkip));
        //
        // Do the allocation, most likely case inline first, for efficiency.
        //
        if (numBytes <= mPageSize - mCurrentPageOffset)
        {
            //
            // Safe to allocate from mCurrentPageOffset.
            //
            uint8_t *memory = reinterpret_cast<uint8_t *>(mInUseList) + mCurrentPageOffset;
            mCurrentPageOffset += numBytes;
            return memory;
        }
        return reinterpret_cast<uint8_t *>(allocateNewPage(numBytes, numBytes));
#endif
    }

    //
    // There is no deallocate.  The point of this class is that
    // deallocation can be skipped by the user of it, as the model
    // of use is to simultaneously deallocate everything at once
    // by calling pop(), and to not have to solve memory leak problems.
    //

    // Catch unwanted allocations.
    // TODO(jmadill): Remove this when we remove the global allocator.
    void lock();
    void unlock();

  private:
    size_t mAlignment;  // all returned allocations will be aligned at
                        // this granularity, which will be a power of 2
    size_t mAlignmentMask;
#if !defined(ANGLE_DISABLE_POOL_ALLOC)
    friend struct Header;

    struct Header
    {
        Header(Header *nextPage, size_t pageCount)
            : nextPage(nextPage),
              pageCount(pageCount)
#    if defined(ANGLE_POOL_ALLOC_GUARD_BLOCKS)
              ,
              lastAllocation(0)
#    endif
        {}

        ~Header()
        {
#    if defined(ANGLE_POOL_ALLOC_GUARD_BLOCKS)
            if (lastAllocation)
                lastAllocation->checkAllocList();
#    endif
        }

        Header *nextPage;
        size_t pageCount;
#    if defined(ANGLE_POOL_ALLOC_GUARD_BLOCKS)
        Allocation *lastAllocation;
#    endif
    };

    struct AllocState
    {
        size_t offset;
        Header *page;
    };
    using AllocStack = std::vector<AllocState>;

    // Slow path of allocation when we have to get a new page.
    void *allocateNewPage(size_t numBytes, size_t allocationSize);
    // Track allocations if and only if we're using guard blocks
    void *initializeAllocation(Header *block, unsigned char *memory, size_t numBytes)
    {
#    if defined(ANGLE_POOL_ALLOC_GUARD_BLOCKS)
        new (memory) Allocation(numBytes + mAlignment, memory, block->lastAllocation);
        block->lastAllocation = reinterpret_cast<Allocation *>(memory);
#    endif
        // The OffsetAllocation() call is optimized away if !defined(ANGLE_POOL_ALLOC_GUARD_BLOCKS)
        void *unalignedPtr  = Allocation::OffsetAllocation(memory);
        size_t alignedBytes = numBytes + mAlignment;
        return std::align(mAlignment, numBytes, unalignedPtr, alignedBytes);
    }

    size_t mPageSize;           // granularity of allocation from the OS
    size_t mHeaderSkip;         // amount of memory to skip to make room for the
                                //      header (basically, size of header, rounded
                                //      up to make it aligned
    size_t mCurrentPageOffset;  // next offset in top of inUseList to allocate from
    Header *mFreeList;          // list of popped memory
    Header *mInUseList;         // list of all memory currently being used
    AllocStack mStack;          // stack of where to allocate from, to partition pool

    int mNumCalls;       // just an interesting statistic
    size_t mTotalBytes;  // just an interesting statistic

#else  // !defined(ANGLE_DISABLE_POOL_ALLOC)
    std::vector<std::vector<void *>> mStack;
#endif

    bool mLocked;
};

}  // namespace angle

#endif  // COMMON_POOLALLOC_H_
