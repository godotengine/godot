// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "platform.h"
#include <vector>
#include <set>

namespace embree
{
#define ALIGNED_STRUCT_(align)                                           \
  void* operator new(size_t size) { return alignedMalloc(size,align); } \
  void operator delete(void* ptr) { alignedFree(ptr); }                 \
  void* operator new[](size_t size) { return alignedMalloc(size,align); } \
  void operator delete[](void* ptr) { alignedFree(ptr); }

#define ALIGNED_CLASS_(align)                                           \
 public:                                                               \
    ALIGNED_STRUCT_(align)                                              \
 private:
  
  /*! aligned allocation */
  void* alignedMalloc(size_t size, size_t align);
  void alignedFree(void* ptr);
  
  /*! allocator that performs aligned allocations */
  template<typename T, size_t alignment>
    struct aligned_allocator
    {
      typedef T value_type;
      typedef T* pointer;
      typedef const T* const_pointer;
      typedef T& reference;
      typedef const T& const_reference;
      typedef std::size_t size_type;
      typedef std::ptrdiff_t difference_type;

      __forceinline pointer allocate( size_type n ) {
        return (pointer) alignedMalloc(n*sizeof(value_type),alignment);
      }

      __forceinline void deallocate( pointer p, size_type n ) {
        return alignedFree(p);
      }

      __forceinline void construct( pointer p, const_reference val ) {
        new (p) T(val);
      }

      __forceinline void destroy( pointer p ) {
        p->~T();
      }
    };

  /*! allocates pages directly from OS */
  bool win_enable_selockmemoryprivilege(bool verbose);
  bool os_init(bool hugepages, bool verbose);
  void* os_malloc (size_t bytes, bool& hugepages);
  size_t os_shrink (void* ptr, size_t bytesNew, size_t bytesOld, bool hugepages);
  void  os_free   (void* ptr, size_t bytes, bool hugepages);
  void  os_advise (void* ptr, size_t bytes);

  /*! allocator that performs OS allocations */
  template<typename T>
    struct os_allocator
    {
      typedef T value_type;
      typedef T* pointer;
      typedef const T* const_pointer;
      typedef T& reference;
      typedef const T& const_reference;
      typedef std::size_t size_type;
      typedef std::ptrdiff_t difference_type;

      __forceinline os_allocator () 
        : hugepages(false) {}

      __forceinline pointer allocate( size_type n ) {
        return (pointer) os_malloc(n*sizeof(value_type),hugepages);
      }

      __forceinline void deallocate( pointer p, size_type n ) {
        return os_free(p,n*sizeof(value_type),hugepages);
      }

      __forceinline void construct( pointer p, const_reference val ) {
        new (p) T(val);
      }

      __forceinline void destroy( pointer p ) {
        p->~T();
      }

      bool hugepages;
    };

  /*! allocator for IDs */
  template<typename T, size_t max_id>
    struct IDPool
    {
      typedef T value_type;

      IDPool ()
      : nextID(0) {}

      T allocate() 
      {
        /* return ID from list */
        if (!IDs.empty()) 
        {
          T id = *IDs.begin();
          IDs.erase(IDs.begin());
          return id;
        } 

        /* allocate new ID */
        else
        {
          if (size_t(nextID)+1 > max_id)
            return -1;
          
          return nextID++;
        }
      }

      /* adds an ID provided by the user */
      bool add(T id)
      {
        if (id > max_id)
          return false;
        
        /* check if ID should be in IDs set */
        if (id < nextID) {
          auto p = IDs.find(id);
          if (p == IDs.end()) return false;
          IDs.erase(p);
          return true;
        }

        /* otherwise increase ID set */
        else
        {
          for (T i=nextID; i<id; i++) {
            IDs.insert(i);
          }
          nextID = id+1;
          return true;
        }
      }

      void deallocate( T id ) 
      {
        assert(id < nextID);
        MAYBE_UNUSED auto done = IDs.insert(id).second;
        assert(done);
      }

    private:
      std::set<T> IDs;   //!< stores deallocated IDs to be reused
      T nextID;          //!< next ID to use when IDs vector is empty
    };
}

