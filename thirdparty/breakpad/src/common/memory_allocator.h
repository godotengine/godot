// Copyright 2009 Google LLC
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
//     * Neither the name of Google LLC nor the names of its
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

#ifndef GOOGLE_BREAKPAD_COMMON_MEMORY_ALLOCATOR_H_
#define GOOGLE_BREAKPAD_COMMON_MEMORY_ALLOCATOR_H_

#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>

#include <memory>
#include <vector>

#if defined(MEMORY_SANITIZER)
#include <sanitizer/msan_interface.h>
#endif

#ifdef __APPLE__
#define sys_mmap mmap
#define sys_munmap munmap
#define MAP_ANONYMOUS MAP_ANON
#else
#include "third_party/lss/linux_syscall_support.h"
#endif

namespace google_breakpad {

// This is very simple allocator which fetches pages from the kernel directly.
// Thus, it can be used even when the heap may be corrupted.
//
// There is no free operation. The pages are only freed when the object is
// destroyed.
class PageAllocator {
 public:
  PageAllocator()
      : page_size_(getpagesize()),
        last_(NULL),
        current_page_(NULL),
        page_offset_(0),
        pages_allocated_(0) {
  }

  ~PageAllocator() {
    FreeAll();
  }

  void* Alloc(size_t bytes) {
    if (!bytes)
      return NULL;

    if (current_page_ && page_size_ - page_offset_ >= bytes) {
      uint8_t* const ret = current_page_ + page_offset_;
      page_offset_ += bytes;
      if (page_offset_ == page_size_) {
        page_offset_ = 0;
        current_page_ = NULL;
      }

      return ret;
    }

    const size_t pages =
        (bytes + sizeof(PageHeader) + page_size_ - 1) / page_size_;
    uint8_t* const ret = GetNPages(pages);
    if (!ret)
      return NULL;

    page_offset_ =
        (page_size_ - (page_size_ * pages - (bytes + sizeof(PageHeader)))) %
        page_size_;
    current_page_ = page_offset_ ? ret + page_size_ * (pages - 1) : NULL;

    return ret + sizeof(PageHeader);
  }

  // Checks whether the page allocator owns the passed-in pointer.
  // This method exists for testing pursposes only.
  bool OwnsPointer(const void* p) {
    for (PageHeader* header = last_; header; header = header->next) {
      const char* current = reinterpret_cast<char*>(header);
      if ((p >= current) && (p < current + header->num_pages * page_size_))
        return true;
    }

    return false;
  }

  unsigned long pages_allocated() { return pages_allocated_; }

 private:
  uint8_t* GetNPages(size_t num_pages) {
    void* a = sys_mmap(NULL, page_size_ * num_pages, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (a == MAP_FAILED)
      return NULL;

#if defined(MEMORY_SANITIZER)
    // We need to indicate to MSan that memory allocated through sys_mmap is
    // initialized, since linux_syscall_support.h doesn't have MSan hooks.
    __msan_unpoison(a, page_size_ * num_pages);
#endif

    struct PageHeader* header = reinterpret_cast<PageHeader*>(a);
    header->next = last_;
    header->num_pages = num_pages;
    last_ = header;

    pages_allocated_ += num_pages;

    return reinterpret_cast<uint8_t*>(a);
  }

  void FreeAll() {
    PageHeader* next;

    for (PageHeader* cur = last_; cur; cur = next) {
      next = cur->next;
      sys_munmap(cur, cur->num_pages * page_size_);
    }
  }

  struct PageHeader {
    PageHeader* next;  // pointer to the start of the next set of pages.
    size_t num_pages;  // the number of pages in this set.
  };

  const size_t page_size_;
  PageHeader* last_;
  uint8_t* current_page_;
  size_t page_offset_;
  unsigned long pages_allocated_;
};

// Wrapper to use with STL containers
template <typename T>
struct PageStdAllocator {
  using AllocatorTraits = std::allocator_traits<std::allocator<T>>;
  using value_type = typename AllocatorTraits::value_type;
  using pointer = typename AllocatorTraits::pointer;
  using difference_type = typename AllocatorTraits::difference_type;
  using size_type = typename AllocatorTraits::size_type;

  explicit PageStdAllocator(PageAllocator& allocator) : allocator_(allocator),
                                                        stackdata_(NULL),
                                                        stackdata_size_(0)
  {}

  template <class Other> PageStdAllocator(const PageStdAllocator<Other>& other)
      : allocator_(other.allocator_),
        stackdata_(nullptr),
        stackdata_size_(0)
  {}

  explicit PageStdAllocator(PageAllocator& allocator,
                            pointer stackdata,
                            size_type stackdata_size) : allocator_(allocator),
      stackdata_(stackdata),
      stackdata_size_(stackdata_size)
  {}

  inline pointer allocate(size_type n, const void* = 0) {
    const size_type size = sizeof(T) * n;
    if (size <= stackdata_size_) {
      return stackdata_;
    }
    return static_cast<pointer>(allocator_.Alloc(size));
  }

  inline void deallocate(pointer, size_type) {
    // The PageAllocator doesn't free.
  }

  template <typename U> struct rebind {
    typedef PageStdAllocator<U> other;
  };

 private:
  // Silly workaround for the gcc from Android's ndk (gcc 4.6), which will
  // otherwise complain that `other.allocator_` is private in the constructor
  // code.
  template<typename Other> friend struct PageStdAllocator;

  PageAllocator& allocator_;
  pointer stackdata_;
  size_type stackdata_size_;
};

// A wasteful vector is a std::vector, except that it allocates memory from a
// PageAllocator. It's wasteful because, when resizing, it always allocates a
// whole new array since the PageAllocator doesn't support realloc.
template<class T>
class wasteful_vector : public std::vector<T, PageStdAllocator<T> > {
 public:
  wasteful_vector(PageAllocator* allocator, unsigned size_hint = 16)
      : std::vector<T, PageStdAllocator<T> >(PageStdAllocator<T>(*allocator)) {
    std::vector<T, PageStdAllocator<T> >::reserve(size_hint);
  }
 protected:
  wasteful_vector(PageStdAllocator<T> allocator)
      : std::vector<T, PageStdAllocator<T> >(allocator) {}
};

// auto_wasteful_vector allocates space on the stack for N entries to avoid
// using the PageAllocator for small data, while still allowing for larger data.
template<class T, unsigned int N>
class auto_wasteful_vector : public wasteful_vector<T> {
 T stackdata_[N];
 public:
  auto_wasteful_vector(PageAllocator* allocator)
      : wasteful_vector<T>(
            PageStdAllocator<T>(*allocator,
                                &stackdata_[0],
                                sizeof(stackdata_))) {
    std::vector<T, PageStdAllocator<T> >::reserve(N);
  }
};

}  // namespace google_breakpad

inline void* operator new(size_t nbytes,
                          google_breakpad::PageAllocator& allocator) {
  return allocator.Alloc(nbytes);
}

#endif  // GOOGLE_BREAKPAD_COMMON_MEMORY_ALLOCATOR_H_
