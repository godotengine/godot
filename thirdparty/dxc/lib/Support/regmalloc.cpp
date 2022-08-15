//===-- regmalloc.cpp - Memory allocation for regex implementation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Support operator new/delete overriding for regex memory allocations.
//===----------------------------------------------------------------------===//

#include "regutils.h"
#include <algorithm>
#include <new>
#include <cstring>

extern "C" {
  void *regex_malloc(size_t size) {
    return ::operator new(size, std::nothrow);
  }
  void *regex_calloc(size_t num, size_t size) {
    void* ptr = regex_malloc(num * size);
    if (ptr) std::memset(ptr, 0, num * size);
    return ptr;
  }
  void* regex_realloc(void* ptr, size_t oldsize, size_t newsize) {
    void* newptr = regex_malloc(newsize);
    if (newptr == nullptr) return nullptr;
    std::memcpy(newptr, ptr, std::min(oldsize, newsize));
    regex_free(ptr);
    return newptr;
  }
  void regex_free(void *ptr) {
    return ::operator delete(ptr);
  }
}
