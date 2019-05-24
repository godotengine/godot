//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2013 LunarG, Inc.
//
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

#ifndef _POOLALLOC_INCLUDED_
#define _POOLALLOC_INCLUDED_

#ifdef _DEBUG
#  define GUARD_BLOCKS  // define to enable guard block sanity checking
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
// STL containers can use this allocator by using the pool_allocator
// class as the allocator (second) template argument.
//

#include <cstddef>
#include <cstring>
#include <vector>

namespace glslang {

// If we are using guard blocks, we must track each individual
// allocation.  If we aren't using guard blocks, these
// never get instantiated, so won't have any impact.
//

class TAllocation {
public:
    TAllocation(size_t size, unsigned char* mem, TAllocation* prev = 0) :
        size(size), mem(mem), prevAlloc(prev) {
        // Allocations are bracketed:
        //    [allocationHeader][initialGuardBlock][userData][finalGuardBlock]
        // This would be cleaner with if (guardBlockSize)..., but that
        // makes the compiler print warnings about 0 length memsets,
        // even with the if() protecting them.
#       ifdef GUARD_BLOCKS
            memset(preGuard(),  guardBlockBeginVal, guardBlockSize);
            memset(data(),      userDataFill,       size);
            memset(postGuard(), guardBlockEndVal,   guardBlockSize);
#       endif
    }

    void check() const {
        checkGuardBlock(preGuard(),  guardBlockBeginVal, "before");
        checkGuardBlock(postGuard(), guardBlockEndVal,   "after");
    }

    void checkAllocList() const;

    // Return total size needed to accommodate user buffer of 'size',
    // plus our tracking data.
    inline static size_t allocationSize(size_t size) {
        return size + 2 * guardBlockSize + headerSize();
    }

    // Offset from surrounding buffer to get to user data buffer.
    inline static unsigned char* offsetAllocation(unsigned char* m) {
        return m + guardBlockSize + headerSize();
    }

private:
    void checkGuardBlock(unsigned char* blockMem, unsigned char val, const char* locText) const;

    // Find offsets to pre and post guard blocks, and user data buffer
    unsigned char* preGuard()  const { return mem + headerSize(); }
    unsigned char* data()      const { return preGuard() + guardBlockSize; }
    unsigned char* postGuard() const { return data() + size; }

    size_t size;                  // size of the user data area
    unsigned char* mem;           // beginning of our allocation (pts to header)
    TAllocation* prevAlloc;       // prior allocation in the chain

    const static unsigned char guardBlockBeginVal;
    const static unsigned char guardBlockEndVal;
    const static unsigned char userDataFill;

    const static size_t guardBlockSize;
#   ifdef GUARD_BLOCKS
    inline static size_t headerSize() { return sizeof(TAllocation); }
#   else
    inline static size_t headerSize() { return 0; }
#   endif
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
class TPoolAllocator {
public:
    TPoolAllocator(int growthIncrement = 8*1024, int allocationAlignment = 16);

    //
    // Don't call the destructor just to free up the memory, call pop()
    //
    ~TPoolAllocator();

    //
    // Call push() to establish a new place to pop memory too.  Does not
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
    void* allocate(size_t numBytes);

    //
    // There is no deallocate.  The point of this class is that
    // deallocation can be skipped by the user of it, as the model
    // of use is to simultaneously deallocate everything at once
    // by calling pop(), and to not have to solve memory leak problems.
    //

protected:
    friend struct tHeader;

    struct tHeader {
        tHeader(tHeader* nextPage, size_t pageCount) :
#ifdef GUARD_BLOCKS
        lastAllocation(0),
#endif
        nextPage(nextPage), pageCount(pageCount) { }

        ~tHeader() {
#ifdef GUARD_BLOCKS
            if (lastAllocation)
                lastAllocation->checkAllocList();
#endif
        }

#ifdef GUARD_BLOCKS
        TAllocation* lastAllocation;
#endif
        tHeader* nextPage;
        size_t pageCount;
    };

    struct tAllocState {
        size_t offset;
        tHeader* page;
    };
    typedef std::vector<tAllocState> tAllocStack;

    // Track allocations if and only if we're using guard blocks
#ifndef GUARD_BLOCKS
    void* initializeAllocation(tHeader*, unsigned char* memory, size_t) {
#else
    void* initializeAllocation(tHeader* block, unsigned char* memory, size_t numBytes) {
        new(memory) TAllocation(numBytes, memory, block->lastAllocation);
        block->lastAllocation = reinterpret_cast<TAllocation*>(memory);
#endif

        // This is optimized entirely away if GUARD_BLOCKS is not defined.
        return TAllocation::offsetAllocation(memory);
    }

    size_t pageSize;        // granularity of allocation from the OS
    size_t alignment;       // all returned allocations will be aligned at
                            //      this granularity, which will be a power of 2
    size_t alignmentMask;
    size_t headerSkip;      // amount of memory to skip to make room for the
                            //      header (basically, size of header, rounded
                            //      up to make it aligned
    size_t currentPageOffset;  // next offset in top of inUseList to allocate from
    tHeader* freeList;      // list of popped memory
    tHeader* inUseList;     // list of all memory currently being used
    tAllocStack stack;      // stack of where to allocate from, to partition pool

    int numCalls;           // just an interesting statistic
    size_t totalBytes;      // just an interesting statistic
private:
    TPoolAllocator& operator=(const TPoolAllocator&);  // don't allow assignment operator
    TPoolAllocator(const TPoolAllocator&);  // don't allow default copy constructor
};

//
// There could potentially be many pools with pops happening at
// different times.  But a simple use is to have a global pop
// with everyone using the same global allocator.
//
extern TPoolAllocator& GetThreadPoolAllocator();
void SetThreadPoolAllocator(TPoolAllocator* poolAllocator);

//
// This STL compatible allocator is intended to be used as the allocator
// parameter to templatized STL containers, like vector and map.
//
// It will use the pools for allocation, and not
// do any deallocation, but will still do destruction.
//
template<class T>
class pool_allocator {
public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;
    template<class Other>
        struct rebind {
            typedef pool_allocator<Other> other;
        };
    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }

    pool_allocator() : allocator(GetThreadPoolAllocator()) { }
    pool_allocator(TPoolAllocator& a) : allocator(a) { }
    pool_allocator(const pool_allocator<T>& p) : allocator(p.allocator) { }

    template<class Other>
        pool_allocator(const pool_allocator<Other>& p) : allocator(p.getAllocator()) { }

    pointer allocate(size_type n) {
        return reinterpret_cast<pointer>(getAllocator().allocate(n * sizeof(T))); }
    pointer allocate(size_type n, const void*) {
        return reinterpret_cast<pointer>(getAllocator().allocate(n * sizeof(T))); }

    void deallocate(void*, size_type) { }
    void deallocate(pointer, size_type) { }

    pointer _Charalloc(size_t n) {
        return reinterpret_cast<pointer>(getAllocator().allocate(n)); }

    void construct(pointer p, const T& val) { new ((void *)p) T(val); }
    void destroy(pointer p) { p->T::~T(); }

    bool operator==(const pool_allocator& rhs) const { return &getAllocator() == &rhs.getAllocator(); }
    bool operator!=(const pool_allocator& rhs) const { return &getAllocator() != &rhs.getAllocator(); }

    size_type max_size() const { return static_cast<size_type>(-1) / sizeof(T); }
    size_type max_size(int size) const { return static_cast<size_type>(-1) / size; }

    void setAllocator(TPoolAllocator* a) { allocator = *a; }
    TPoolAllocator& getAllocator() const { return allocator; }

protected:
    pool_allocator& operator=(const pool_allocator&) { return *this; }
    TPoolAllocator& allocator;
};

} // end namespace glslang

#endif // _POOLALLOC_INCLUDED_
