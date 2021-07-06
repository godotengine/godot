//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// PoolAlloc.cpp:
//    Implements the class methods for PoolAllocator and Allocation classes.
//

#include "common/PoolAlloc.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "common/angleutils.h"
#include "common/debug.h"
#include "common/mathutil.h"
#include "common/platform.h"
#include "common/tls.h"

namespace angle
{

//
// Implement the functionality of the PoolAllocator class, which
// is documented in PoolAlloc.h.
//
PoolAllocator::PoolAllocator(int growthIncrement, int allocationAlignment)
    : mAlignment(allocationAlignment),
#if !defined(ANGLE_DISABLE_POOL_ALLOC)
      mPageSize(growthIncrement),
      mFreeList(0),
      mInUseList(0),
      mNumCalls(0),
      mTotalBytes(0),
#endif
      mLocked(false)
{
#if !defined(ANGLE_DISABLE_POOL_ALLOC)
    if (mAlignment == 1)
    {
        // This is a special fast-path where fastAllocation() is enabled
        mAlignmentMask = 0;
        mHeaderSkip    = sizeof(Header);
    }
    else
    {
#endif
        //
        // Adjust mAlignment to be at least pointer aligned and
        // power of 2.
        //
        size_t minAlign = sizeof(void *);
        mAlignment &= ~(minAlign - 1);
        if (mAlignment < minAlign)
            mAlignment = minAlign;
        mAlignment     = gl::ceilPow2(static_cast<unsigned int>(mAlignment));
        mAlignmentMask = mAlignment - 1;

#if !defined(ANGLE_DISABLE_POOL_ALLOC)
        //
        // Align header skip
        //
        mHeaderSkip = minAlign;
        if (mHeaderSkip < sizeof(Header))
        {
            mHeaderSkip = rx::roundUpPow2(sizeof(Header), mAlignment);
        }
    }
    //
    // Don't allow page sizes we know are smaller than all common
    // OS page sizes.
    //
    if (mPageSize < 4 * 1024)
        mPageSize = 4 * 1024;
    //
    // A large mCurrentPageOffset indicates a new page needs to
    // be obtained to allocate memory.
    //
    mCurrentPageOffset = mPageSize;
#else  // !defined(ANGLE_DISABLE_POOL_ALLOC)
    mStack.push_back({});
#endif
}

PoolAllocator::~PoolAllocator()
{
#if !defined(ANGLE_DISABLE_POOL_ALLOC)
    while (mInUseList)
    {
        Header *next = mInUseList->nextPage;
        mInUseList->~Header();
        delete[] reinterpret_cast<char *>(mInUseList);
        mInUseList = next;
    }
    // We should not check the guard blocks
    // here, because we did it already when the block was
    // placed into the free list.
    //
    while (mFreeList)
    {
        Header *next = mFreeList->nextPage;
        delete[] reinterpret_cast<char *>(mFreeList);
        mFreeList = next;
    }
#else  // !defined(ANGLE_DISABLE_POOL_ALLOC)
    for (auto &allocs : mStack)
    {
        for (auto alloc : allocs)
        {
            free(alloc);
        }
    }
    mStack.clear();
#endif
}

//
// Check a single guard block for damage
//
void Allocation::checkGuardBlock(unsigned char *blockMem,
                                 unsigned char val,
                                 const char *locText) const
{
#if defined(ANGLE_POOL_ALLOC_GUARD_BLOCKS)
    for (size_t x = 0; x < kGuardBlockSize; x++)
    {
        if (blockMem[x] != val)
        {
            char assertMsg[80];
            // We don't print the assert message.  It's here just to be helpful.
            snprintf(assertMsg, sizeof(assertMsg),
                     "PoolAlloc: Damage %s %zu byte allocation at 0x%p\n", locText, mSize, data());
            assert(0 && "PoolAlloc: Damage in guard block");
        }
    }
#endif
}

void PoolAllocator::push()
{
#if !defined(ANGLE_DISABLE_POOL_ALLOC)
    AllocState state = {mCurrentPageOffset, mInUseList};

    mStack.push_back(state);

    //
    // Indicate there is no current page to allocate from.
    //
    mCurrentPageOffset = mPageSize;
#else  // !defined(ANGLE_DISABLE_POOL_ALLOC)
    mStack.push_back({});
#endif
}

//
// Do a mass-deallocation of all the individual allocations
// that have occurred since the last push(), or since the
// last pop(), or since the object's creation.
//
// The deallocated pages are saved for future allocations.
//
void PoolAllocator::pop()
{
    if (mStack.size() < 1)
        return;

#if !defined(ANGLE_DISABLE_POOL_ALLOC)
    Header *page       = mStack.back().page;
    mCurrentPageOffset = mStack.back().offset;

    while (mInUseList != page)
    {
        // invoke destructor to free allocation list
        mInUseList->~Header();

        Header *nextInUse = mInUseList->nextPage;
        if (mInUseList->pageCount > 1)
            delete[] reinterpret_cast<char *>(mInUseList);
        else
        {
            mInUseList->nextPage = mFreeList;
            mFreeList            = mInUseList;
        }
        mInUseList = nextInUse;
    }

    mStack.pop_back();
#else  // !defined(ANGLE_DISABLE_POOL_ALLOC)
    for (auto &alloc : mStack.back())
    {
        free(alloc);
    }
    mStack.pop_back();
#endif
}

//
// Do a mass-deallocation of all the individual allocations
// that have occurred.
//
void PoolAllocator::popAll()
{
    while (mStack.size() > 0)
        pop();
}

void *PoolAllocator::allocate(size_t numBytes)
{
    ASSERT(!mLocked);

#if !defined(ANGLE_DISABLE_POOL_ALLOC)
    //
    // Just keep some interesting statistics.
    //
    ++mNumCalls;
    mTotalBytes += numBytes;

    // If we are using guard blocks, all allocations are bracketed by
    // them: [guardblock][allocation][guardblock].  numBytes is how
    // much memory the caller asked for.  allocationSize is the total
    // size including guard blocks.  In release build,
    // kGuardBlockSize=0 and this all gets optimized away.
    size_t allocationSize = Allocation::AllocationSize(numBytes) + mAlignment;
    // Detect integer overflow.
    if (allocationSize < numBytes)
        return 0;

    //
    // Do the allocation, most likely case first, for efficiency.
    // This step could be moved to be inline sometime.
    //
    if (allocationSize <= mPageSize - mCurrentPageOffset)
    {
        //
        // Safe to allocate from mCurrentPageOffset.
        //
        unsigned char *memory = reinterpret_cast<unsigned char *>(mInUseList) + mCurrentPageOffset;
        mCurrentPageOffset += allocationSize;
        mCurrentPageOffset = (mCurrentPageOffset + mAlignmentMask) & ~mAlignmentMask;

        return initializeAllocation(mInUseList, memory, numBytes);
    }

    if (allocationSize > mPageSize - mHeaderSkip)
    {
        //
        // Do a multi-page allocation.  Don't mix these with the others.
        // The OS is efficient in allocating and freeing multiple pages.
        //
        size_t numBytesToAlloc = allocationSize + mHeaderSkip;
        // Detect integer overflow.
        if (numBytesToAlloc < allocationSize)
            return 0;

        Header *memory = reinterpret_cast<Header *>(::new char[numBytesToAlloc]);
        if (memory == 0)
            return 0;

        // Use placement-new to initialize header
        new (memory) Header(mInUseList, (numBytesToAlloc + mPageSize - 1) / mPageSize);
        mInUseList = memory;

        mCurrentPageOffset = mPageSize;  // make next allocation come from a new page

        // No guard blocks for multi-page allocations (yet)
        void *unalignedPtr =
            reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(memory) + mHeaderSkip);
        return std::align(mAlignment, numBytes, unalignedPtr, allocationSize);
    }
    unsigned char *newPageAddr =
        static_cast<unsigned char *>(allocateNewPage(numBytes, allocationSize));
    return initializeAllocation(mInUseList, newPageAddr, numBytes);
#else  // !defined(ANGLE_DISABLE_POOL_ALLOC)
    void *alloc = malloc(numBytes + mAlignmentMask);
    mStack.back().push_back(alloc);

    intptr_t intAlloc = reinterpret_cast<intptr_t>(alloc);
    intAlloc          = (intAlloc + mAlignmentMask) & ~mAlignmentMask;
    return reinterpret_cast<void *>(intAlloc);
#endif
}

#if !defined(ANGLE_DISABLE_POOL_ALLOC)
void *PoolAllocator::allocateNewPage(size_t numBytes, size_t allocationSize)
{
    //
    // Need a simple page to allocate from.
    //
    Header *memory;
    if (mFreeList)
    {
        memory    = mFreeList;
        mFreeList = mFreeList->nextPage;
    }
    else
    {
        memory = reinterpret_cast<Header *>(::new char[mPageSize]);
        if (memory == 0)
            return 0;
    }
    // Use placement-new to initialize header
    new (memory) Header(mInUseList, 1);
    mInUseList = memory;

    unsigned char *ret = reinterpret_cast<unsigned char *>(mInUseList) + mHeaderSkip;
    mCurrentPageOffset = (mHeaderSkip + allocationSize + mAlignmentMask) & ~mAlignmentMask;
    return ret;
}
#endif

void PoolAllocator::lock()
{
    ASSERT(!mLocked);
    mLocked = true;
}

void PoolAllocator::unlock()
{
    ASSERT(mLocked);
    mLocked = false;
}

//
// Check all allocations in a list for damage by calling check on each.
//
void Allocation::checkAllocList() const
{
    for (const Allocation *alloc = this; alloc != 0; alloc = alloc->mPrevAlloc)
        alloc->check();
}

}  // namespace angle
