//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_POOLALLOC_H_
#define COMPILER_TRANSLATOR_POOLALLOC_H_

//
// This header defines the pool_allocator class that allows STL containers
// to use the angle::PoolAllocator class by using the pool_allocator
// class as the allocator (second) template argument.
//
// It also defines functions for managing the GlobalPoolAllocator used by the compiler.
//

#include <stddef.h>
#include <string.h>
#include <vector>

#include "common/PoolAlloc.h"

//
// There could potentially be many pools with pops happening at
// different times.  But a simple use is to have a global pop
// with everyone using the same global allocator.
//
extern angle::PoolAllocator *GetGlobalPoolAllocator();
extern void SetGlobalPoolAllocator(angle::PoolAllocator *poolAllocator);

//
// This STL compatible allocator is intended to be used as the allocator
// parameter to templatized STL containers, like vector and map.
//
// It will use the pools for allocation, and not
// do any deallocation, but will still do destruction.
//
template <class T>
class pool_allocator
{
  public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef T &reference;
    typedef const T &const_reference;
    typedef T value_type;

    template <class Other>
    struct rebind
    {
        typedef pool_allocator<Other> other;
    };
    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }

    pool_allocator() {}

    template <class Other>
    pool_allocator(const pool_allocator<Other> &p)
    {}

    template <class Other>
    pool_allocator<T> &operator=(const pool_allocator<Other> &p)
    {
        return *this;
    }

#if defined(__SUNPRO_CC) && !defined(_RWSTD_ALLOCATOR)
    // libCStd on some platforms have a different allocate/deallocate interface.
    // Caller pre-bakes sizeof(T) into 'n' which is the number of bytes to be
    // allocated, not the number of elements.
    void *allocate(size_type n) { return getAllocator().allocate(n); }
    void *allocate(size_type n, const void *) { return getAllocator().allocate(n); }
    void deallocate(void *, size_type) {}
#else
    pointer allocate(size_type n)
    {
        return static_cast<pointer>(getAllocator().allocate(n * sizeof(T)));
    }
    pointer allocate(size_type n, const void *)
    {
        return static_cast<pointer>(getAllocator().allocate(n * sizeof(T)));
    }
    void deallocate(pointer, size_type) {}
#endif  // _RWSTD_ALLOCATOR

    void construct(pointer p, const T &val) { new ((void *)p) T(val); }
    void destroy(pointer p) { p->T::~T(); }

    bool operator==(const pool_allocator &rhs) const { return true; }
    bool operator!=(const pool_allocator &rhs) const { return false; }

    size_type max_size() const { return static_cast<size_type>(-1) / sizeof(T); }
    size_type max_size(int size) const { return static_cast<size_type>(-1) / size; }

    angle::PoolAllocator &getAllocator() const { return *GetGlobalPoolAllocator(); }
};

#endif  // COMPILER_TRANSLATOR_POOLALLOC_H_
