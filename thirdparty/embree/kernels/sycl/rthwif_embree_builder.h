// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../common/sys/platform.h"
#include "../../common/sys/sycl.h"
#include "../../common/sys/vector.h"
#include "../../common/math/bbox.h"
#include "../../include/embree4/rtcore.h"
 
namespace embree
{
  class Scene;
  
  void* rthwifAllocAccelBuffer(Device* embree_device, size_t bytes, sycl::device device, sycl::context context);

  void rthwifFreeAccelBuffer(Device* embree_device, void* ptr, size_t bytes, sycl::context context);

  /*! allocator that performs BVH memory allocations */
  template<typename T>
    struct AccelAllocator
    {
      typedef T value_type;
      typedef T* pointer;
      typedef const T* const_pointer;
      typedef T& reference;
      typedef const T& const_reference;
      typedef std::size_t size_type;
      typedef std::ptrdiff_t difference_type;
      
      AccelAllocator()
        : device(nullptr), context(nullptr) {}
      
      AccelAllocator(Device* embree_device, const sycl::device& device, const sycl::context& context)
        : embree_device(embree_device), device(&device), context(&context) {}
      
      __forceinline pointer allocate( size_type n ) {
        if (context && device)
          return (pointer) rthwifAllocAccelBuffer(embree_device,n*sizeof(T),*device,*context);
        else
          return nullptr;
      }
      
      __forceinline void deallocate( pointer p, size_type n ) {
        if (context)
          rthwifFreeAccelBuffer(embree_device,p,n*sizeof(T),*context);
      }
      
      __forceinline void construct( pointer p, const_reference val ) {
        new (p) T(val);
      }
      
      __forceinline void destroy( pointer p ) {
        p->~T();
      }

      private:

      Device* embree_device;
      const sycl::device* device;
      const sycl::context* context;
    };

  typedef vector_t<char,AccelAllocator<char>> AccelBuffer;
    
  void* zeRTASInitExp(sycl::device device, sycl::context context);
  
  void rthwifCleanup(Device* embree_device, void* dispatchGlobalsPtr, sycl::context context);

  int rthwifIsSYCLDeviceSupported(const sycl::device& sycl_device);
  
  BBox3f rthwifBuild(Scene* scene, AccelBuffer& buffer_o);
}
