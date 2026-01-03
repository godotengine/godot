//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include "../Include/Common.h"
#include "../Include/PoolAlloc.h"

// Mostly here for target that do not support threads such as WASI.
#ifdef DISABLE_THREAD_SUPPORT
#define THREAD_LOCAL 
#else
#define THREAD_LOCAL thread_local
#endif

namespace glslang {

namespace {
THREAD_LOCAL TPoolAllocator* threadPoolAllocator = nullptr;

TPoolAllocator* GetDefaultThreadPoolAllocator()
{
    THREAD_LOCAL TPoolAllocator defaultAllocator;
    return &defaultAllocator;
}
} // anonymous namespace

// Return the thread-specific current pool.
TPoolAllocator& GetThreadPoolAllocator()
{
    return *(threadPoolAllocator ? threadPoolAllocator : GetDefaultThreadPoolAllocator());
}

// Set the thread-specific current pool.
void SetThreadPoolAllocator(TPoolAllocator* poolAllocator)
{
    threadPoolAllocator = poolAllocator;
}

//
// Implement the functionality of the TPoolAllocator class, which
// is documented in PoolAlloc.h.
//
TPoolAllocator::TPoolAllocator(int growthIncrement, int allocationAlignment) :
    pageSize(growthIncrement),
    alignment(allocationAlignment),
    freeList(nullptr),
    inUseList(nullptr),
    numCalls(0)
{
    //
    // Don't allow page sizes we know are smaller than all common
    // OS page sizes.
    //
    if (pageSize < 4*1024)
        pageSize = 4*1024;

    //
    // A large currentPageOffset indicates a new page needs to
    // be obtained to allocate memory.
    //
    currentPageOffset = pageSize;

    //
    // Adjust alignment to be at least pointer aligned and
    // power of 2.
    //
    size_t minAlign = sizeof(void*);
    alignment &= ~(minAlign - 1);
    if (alignment < minAlign)
        alignment = minAlign;
    size_t a = 1;
    while (a < alignment)
        a <<= 1;
    alignment = a;
    alignmentMask = a - 1;

    //
    // Align header skip
    //
    headerSkip = minAlign;
    if (headerSkip < sizeof(tHeader)) {
        headerSkip = (sizeof(tHeader) + alignmentMask) & ~alignmentMask;
    }

    push();
}

TPoolAllocator::~TPoolAllocator()
{
    while (inUseList) {
        tHeader* next = inUseList->nextPage;
        inUseList->~tHeader();
        delete [] reinterpret_cast<char*>(inUseList);
        inUseList = next;
    }

    //
    // Always delete the free list memory - it can't be being
    // (correctly) referenced, whether the pool allocator was
    // global or not.  We should not check the guard blocks
    // here, because we did it already when the block was
    // placed into the free list.
    //
    while (freeList) {
        tHeader* next = freeList->nextPage;
        delete [] reinterpret_cast<char*>(freeList);
        freeList = next;
    }
}

//
// Check a single guard block for damage
//
#ifdef GUARD_BLOCKS
void TAllocation::checkGuardBlock(unsigned char* blockMem, unsigned char val, const char* locText) const
#else
void TAllocation::checkGuardBlock(unsigned char*, unsigned char, const char*) const
#endif
{
#ifdef GUARD_BLOCKS
    for (size_t x = 0; x < guardBlockSize; x++) {
        if (blockMem[x] != val) {
            const int maxSize = 80;
            char assertMsg[maxSize];

            // We don't print the assert message.  It's here just to be helpful.
            snprintf(assertMsg, maxSize, "PoolAlloc: Damage %s %zu byte allocation at 0x%p\n",
                      locText, size, data());
            assert(0 && "PoolAlloc: Damage in guard block");
        }
    }
#else
    assert(guardBlockSize == 0);
#endif
}

void TPoolAllocator::push()
{
    tAllocState state = { currentPageOffset, inUseList };

    stack.push_back(state);

    //
    // Indicate there is no current page to allocate from.
    //
    currentPageOffset = pageSize;
}

//
// Do a mass-deallocation of all the individual allocations
// that have occurred since the last push(), or since the
// last pop(), or since the object's creation.
//
// The deallocated pages are saved for future allocations.
//
void TPoolAllocator::pop()
{
    if (stack.size() < 1)
        return;

    tHeader* page = stack.back().page;
    currentPageOffset = stack.back().offset;

    while (inUseList != page) {
        tHeader* nextInUse = inUseList->nextPage;
        size_t pageCount = inUseList->pageCount;

        // This technically ends the lifetime of the header as C++ object,
        // but we will still control the memory and reuse it.
        inUseList->~tHeader(); // currently, just a debug allocation checker

        if (pageCount > 1) {
            delete [] reinterpret_cast<char*>(inUseList);
        } else {
            inUseList->nextPage = freeList;
            freeList = inUseList;
        }
        inUseList = nextInUse;
    }

    stack.pop_back();
}

//
// Do a mass-deallocation of all the individual allocations
// that have occurred.
//
void TPoolAllocator::popAll()
{
    while (stack.size() > 0)
        pop();
}

void* TPoolAllocator::allocate(size_t numBytes)
{
    // If we are using guard blocks, all allocations are bracketed by
    // them: [guardblock][allocation][guardblock].  numBytes is how
    // much memory the caller asked for.  allocationSize is the total
    // size including guard blocks.  In release build,
    // guardBlockSize=0 and this all gets optimized away.
    size_t allocationSize = TAllocation::allocationSize(numBytes);

    //
    // Just keep some interesting statistics.
    //
    ++numCalls;
    totalBytes += numBytes;

    //
    // Do the allocation, most likely case first, for efficiency.
    // This step could be moved to be inline sometime.
    //
    if (currentPageOffset + allocationSize <= pageSize) {
        //
        // Safe to allocate from currentPageOffset.
        //
        unsigned char* memory = reinterpret_cast<unsigned char*>(inUseList) + currentPageOffset;
        currentPageOffset += allocationSize;
        currentPageOffset = (currentPageOffset + alignmentMask) & ~alignmentMask;

        return initializeAllocation(inUseList, memory, numBytes);
    }

    if (allocationSize + headerSkip > pageSize) {
        //
        // Do a multi-page allocation.  Don't mix these with the others.
        // The OS is efficient and allocating and free-ing multiple pages.
        //
        size_t numBytesToAlloc = allocationSize + headerSkip;
        tHeader* memory = reinterpret_cast<tHeader*>(::new char[numBytesToAlloc]);
        if (memory == nullptr)
            return nullptr;

        // Use placement-new to initialize header
        new(memory) tHeader(inUseList, (numBytesToAlloc + pageSize - 1) / pageSize);
        inUseList = memory;

        currentPageOffset = pageSize;  // make next allocation come from a new page

        // No guard blocks for multi-page allocations (yet)
        return reinterpret_cast<void*>(reinterpret_cast<UINT_PTR>(memory) + headerSkip);
    }

    //
    // Need a simple page to allocate from.
    //
    tHeader* memory;
    if (freeList) {
        memory = freeList;
        freeList = freeList->nextPage;
    } else {
        memory = reinterpret_cast<tHeader*>(::new char[pageSize]);
        if (memory == nullptr)
            return nullptr;
    }

    // Use placement-new to initialize header
    new(memory) tHeader(inUseList, 1);
    inUseList = memory;

    unsigned char* ret = reinterpret_cast<unsigned char*>(inUseList) + headerSkip;
    currentPageOffset = (headerSkip + allocationSize + alignmentMask) & ~alignmentMask;

    return initializeAllocation(inUseList, ret, numBytes);
}

//
// Check all allocations in a list for damage by calling check on each.
//
void TAllocation::checkAllocList() const
{
    for (const TAllocation* alloc = this; alloc != nullptr; alloc = alloc->prevAlloc)
        alloc->check();
}

} // end namespace glslang
