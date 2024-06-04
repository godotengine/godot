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
// See the header file for documentation

#include "protected_memory_allocator.h"
#include <assert.h>

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ProtectedMemoryAllocator::ProtectedMemoryAllocator(vm_size_t pool_size) 
  : pool_size_(pool_size),
    next_alloc_offset_(0),
    valid_(false) {
  
  kern_return_t result = vm_allocate(mach_task_self(),
                                     &base_address_,
                                     pool_size,
                                     TRUE
                                     );
  
  valid_ = (result == KERN_SUCCESS);
  assert(valid_);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ProtectedMemoryAllocator::~ProtectedMemoryAllocator() {
  vm_deallocate(mach_task_self(),
                base_address_,
                pool_size_
                );
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
char *ProtectedMemoryAllocator::Allocate(vm_size_t bytes) {
  if (valid_ && next_alloc_offset_ + bytes <= pool_size_) {
    char *p = (char*)base_address_ + next_alloc_offset_;
    next_alloc_offset_ += bytes;
    return p;
  }
  
  return NULL;  // ran out of memory in our allocation block
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
kern_return_t  ProtectedMemoryAllocator::Protect() {
  kern_return_t result = vm_protect(mach_task_self(),
                                    base_address_,
                                    pool_size_,
                                    FALSE,
                                    VM_PROT_READ);
  
  return result;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
kern_return_t  ProtectedMemoryAllocator::Unprotect() {
  kern_return_t result = vm_protect(mach_task_self(),
                                    base_address_,
                                    pool_size_,
                                    FALSE,
                                    VM_PROT_READ | VM_PROT_WRITE);
  
  return result;
}
