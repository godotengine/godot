// Copyright (c) 2006, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ProtectedMemoryAllocator
//
// A very simple allocator class which allows allocation, but not deallocation.
// The allocations can be made read-only with the Protect() method.
// This class is NOT useful as a general-purpose memory allocation system,
// since it does not allow deallocation.  It is useful to use for a group
// of allocations which are created in the same time-frame and destroyed
// in the same time-frame.  It is useful for making allocations of memory
// which will not need to change often once initialized.  This memory can then
// be protected from memory smashers by calling the Protect() method.

#ifndef PROTECTED_MEMORY_ALLOCATOR_H__
#define PROTECTED_MEMORY_ALLOCATOR_H__

#include <mach/mach.h>

//
class ProtectedMemoryAllocator {
 public:
  ProtectedMemoryAllocator(vm_size_t pool_size);  
  ~ProtectedMemoryAllocator();
  
  // Returns a pointer to an allocation of size n within the pool.
  // Fails by returning NULL is no more space is available.
  // Please note that the pointers returned from this method should not
  // be freed in any way (for example by calling free() on them ).
  char *         Allocate(vm_size_t n);
  
  // Returns the base address of the allocation pool.
  char *         GetBaseAddress() { return (char*)base_address_; }

  // Returns the size of the allocation pool, including allocated
  // plus free space.
  vm_size_t      GetTotalSize() { return pool_size_; }

  // Returns the number of bytes already allocated in the pool.
  vm_size_t      GetAllocatedSize() { return next_alloc_offset_; }

  // Returns the number of bytes available for allocation.
  vm_size_t      GetFreeSize() { return pool_size_ - next_alloc_offset_; }
  
  // Makes the entire allocation pool read-only including, of course,
  // all allocations made from the pool.
  kern_return_t  Protect();  

  // Makes the entire allocation pool read/write.
  kern_return_t  Unprotect();  
  
 private:
  vm_size_t      pool_size_;
  vm_address_t   base_address_;
  vm_size_t      next_alloc_offset_;
  bool           valid_;
};

#endif // PROTECTED_MEMORY_ALLOCATOR_H__
