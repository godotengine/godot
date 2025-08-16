// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "default.h"

namespace embree
{
  /*! invokes the memory monitor callback */
  struct MemoryMonitorInterface {
    virtual void memoryMonitor(ssize_t bytes, bool post) = 0;
  };

  /*! allocator that performs aligned monitored allocations */
  template<typename T, size_t alignment = 64>
    struct aligned_monitored_allocator
    {
      typedef T value_type;
      typedef T* pointer;
      typedef const T* const_pointer;
      typedef T& reference;
      typedef const T& const_reference;
      typedef std::size_t size_type;
      typedef std::ptrdiff_t difference_type;
      
      __forceinline aligned_monitored_allocator(MemoryMonitorInterface* device) 
        : device(device), hugepages(false) {}

      __forceinline pointer allocate( size_type n ) 
      {
        if (n) {
          assert(device);
          device->memoryMonitor(n*sizeof(T),false);
        }
        if (n*sizeof(value_type) >= 14 * PAGE_SIZE_2M)
        {
          pointer p =  (pointer) os_malloc(n*sizeof(value_type),hugepages);
          assert(p);
          return p;
        }
        return (pointer) alignedMalloc(n*sizeof(value_type),alignment);
      }

      __forceinline void deallocate( pointer p, size_type n ) 
      {
        if (p)
        {
          if (n*sizeof(value_type) >= 14 * PAGE_SIZE_2M)
            os_free(p,n*sizeof(value_type),hugepages); 
          else
            alignedFree(p);
        }
        else assert(n == 0);

        if (n) {
          assert(device);
          device->memoryMonitor(-ssize_t(n)*sizeof(T),true);
        }
      }

      __forceinline void construct( pointer p, const_reference val ) {
        new (p) T(val);
      }

      __forceinline void destroy( pointer p ) {
        p->~T();
      }

    private:
      MemoryMonitorInterface* device;
      bool hugepages;
    };

  /*! monitored vector */
  template<typename T>
    using mvector = vector_t<T,aligned_monitored_allocator<T,std::alignment_of<T>::value> >;
}
