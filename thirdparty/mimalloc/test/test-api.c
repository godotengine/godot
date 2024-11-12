/* ----------------------------------------------------------------------------
Copyright (c) 2018-2020, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Walloc-size-larger-than="
#endif

/*
Testing allocators is difficult as bugs may only surface after particular
allocation patterns. The main approach to testing _mimalloc_ is therefore
to have extensive internal invariant checking (see `page_is_valid` in `page.c`
for example), which is enabled in debug mode with `-DMI_DEBUG_FULL=ON`.
The main testing is then to run `mimalloc-bench` [1] using full invariant checking
to catch any potential problems over a wide range of intensive allocation bench
marks.

However, this does not test well for the entire API surface. In this test file
we therefore test the API over various inputs. Please add more tests :-)

[1] https://github.com/daanx/mimalloc-bench
*/

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <errno.h>

#ifdef __cplusplus
#include <vector>
#endif

#include "mimalloc.h"
// #include "mimalloc/internal.h"
#include "mimalloc/types.h" // for MI_DEBUG and MI_BLOCK_ALIGNMENT_MAX

#include "testhelper.h"

// ---------------------------------------------------------------------------
// Test functions
// ---------------------------------------------------------------------------
bool test_heap1(void);
bool test_heap2(void);
bool test_stl_allocator1(void);
bool test_stl_allocator2(void);

bool test_stl_heap_allocator1(void);
bool test_stl_heap_allocator2(void);
bool test_stl_heap_allocator3(void);
bool test_stl_heap_allocator4(void);

bool mem_is_zero(uint8_t* p, size_t size) {
  if (p==NULL) return false;
  for (size_t i = 0; i < size; ++i) {
    if (p[i] != 0) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Main testing
// ---------------------------------------------------------------------------
int main(void) {
  mi_option_disable(mi_option_verbose);

  // ---------------------------------------------------
  // Malloc
  // ---------------------------------------------------

  CHECK_BODY("malloc-zero") {
    void* p = mi_malloc(0);
    result = (p != NULL);
    mi_free(p);
  };
  CHECK_BODY("malloc-nomem1") {
    result = (mi_malloc((size_t)PTRDIFF_MAX + (size_t)1) == NULL);
  };
  CHECK_BODY("malloc-null") {
    mi_free(NULL);
  };
  CHECK_BODY("calloc-overflow") {
    // use (size_t)&mi_calloc to get some number without triggering compiler warnings
    result = (mi_calloc((size_t)&mi_calloc,SIZE_MAX/1000) == NULL);
  };
  CHECK_BODY("calloc0") {
    void* p = mi_calloc(0,1000);
    result = (mi_usable_size(p) <= 16);
    mi_free(p);
  };
  CHECK_BODY("malloc-large") {   // see PR #544.
    void* p = mi_malloc(67108872);
    mi_free(p);
  };

  // ---------------------------------------------------
  // Extended
  // ---------------------------------------------------
  CHECK_BODY("posix_memalign1") {
    void* p = &p;
    int err = mi_posix_memalign(&p, sizeof(void*), 32);
    result = ((err==0 && (uintptr_t)p % sizeof(void*) == 0) || p==&p);
    mi_free(p);
  };
  CHECK_BODY("posix_memalign_no_align") {
    void* p = &p;
    int err = mi_posix_memalign(&p, 3, 32);
    result = (err==EINVAL && p==&p);
  };
  CHECK_BODY("posix_memalign_zero") {
    void* p = &p;
    int err = mi_posix_memalign(&p, sizeof(void*), 0);
    mi_free(p);
    result = (err==0);
  };
  CHECK_BODY("posix_memalign_nopow2") {
    void* p = &p;
    int err = mi_posix_memalign(&p, 3*sizeof(void*), 32);
    result = (err==EINVAL && p==&p);
  };
  CHECK_BODY("posix_memalign_nomem") {
    void* p = &p;
    int err = mi_posix_memalign(&p, sizeof(void*), SIZE_MAX);
    result = (err==ENOMEM && p==&p);
  };

  // ---------------------------------------------------
  // Aligned API
  // ---------------------------------------------------
  CHECK_BODY("malloc-aligned1") {
    void* p = mi_malloc_aligned(32,32); result = (p != NULL && (uintptr_t)(p) % 32 == 0); mi_free(p);
  };
  CHECK_BODY("malloc-aligned2") {
    void* p = mi_malloc_aligned(48,32); result = (p != NULL && (uintptr_t)(p) % 32 == 0); mi_free(p);
  };
  CHECK_BODY("malloc-aligned3") {
    void* p1 = mi_malloc_aligned(48,32); bool result1 = (p1 != NULL && (uintptr_t)(p1) % 32 == 0);
    void* p2 = mi_malloc_aligned(48,32); bool result2 = (p2 != NULL && (uintptr_t)(p2) % 32 == 0);
    mi_free(p2);
    mi_free(p1);
    result = (result1&&result2);
  };
  CHECK_BODY("malloc-aligned4") {
    void* p;
    bool ok = true;
    for (int i = 0; i < 8 && ok; i++) {
      p = mi_malloc_aligned(8, 16);
      ok = (p != NULL && (uintptr_t)(p) % 16 == 0); mi_free(p);
    }
    result = ok;
  };
  CHECK_BODY("malloc-aligned5") {
    void* p = mi_malloc_aligned(4097,4096);
    size_t usable = mi_usable_size(p);
    result = (usable >= 4097 && usable < 16000);
    printf("malloc_aligned5: usable size: %zi\n", usable);
    mi_free(p);
  };
  CHECK_BODY("malloc-aligned6") {
    bool ok = true;
    for (size_t align = 1; align <= MI_BLOCK_ALIGNMENT_MAX && ok; align *= 2) {
      void* ps[8];
      for (int i = 0; i < 8 && ok; i++) {
        ps[i] = mi_malloc_aligned(align*13  // size
                                 , align);
        if (ps[i] == NULL || (uintptr_t)(ps[i]) % align != 0) {
          ok = false;
        }
      }
      for (int i = 0; i < 8 && ok; i++) {
        mi_free(ps[i]);
      }
    }
    result = ok;
  };
  CHECK_BODY("malloc-aligned7") {
    void* p = mi_malloc_aligned(1024,MI_BLOCK_ALIGNMENT_MAX);
    mi_free(p);
    result = ((uintptr_t)p % MI_BLOCK_ALIGNMENT_MAX) == 0;
  };
  CHECK_BODY("malloc-aligned8") {
    bool ok = true;
    for (int i = 0; i < 5 && ok; i++) {
      int n = (1 << i);
      void* p = mi_malloc_aligned(1024, n * MI_BLOCK_ALIGNMENT_MAX);
      ok = ((uintptr_t)p % (n*MI_BLOCK_ALIGNMENT_MAX)) == 0;
      mi_free(p);
    }
    result = ok;
  };
  CHECK_BODY("malloc-aligned9") {
    bool ok = true;
    void* p[8];
    size_t sizes[8] = { 8, 512, 1024 * 1024, MI_BLOCK_ALIGNMENT_MAX, MI_BLOCK_ALIGNMENT_MAX + 1, 2 * MI_BLOCK_ALIGNMENT_MAX, 8 * MI_BLOCK_ALIGNMENT_MAX, 0 };
    for (int i = 0; i < 28 && ok; i++) {
      int align = (1 << i);
      for (int j = 0; j < 8 && ok; j++) {
        p[j] = mi_zalloc_aligned(sizes[j], align);
        ok = ((uintptr_t)p[j] % align) == 0;
      }
      for (int j = 0; j < 8; j++) {
        mi_free(p[j]);
      }
    }
    result = ok;
  };
  CHECK_BODY("malloc-aligned10") {
    bool ok = true;
    void* p[10+1];
    int align;
    int j;
    for(j = 0, align = 1; j <= 10 && ok; align *= 2, j++ ) {
      p[j] = mi_malloc_aligned(43 + align, align);
      ok = ((uintptr_t)p[j] % align) == 0;
    }
    for ( ; j > 0; j--) {
      mi_free(p[j-1]);
    }
    result = ok;
  }
  CHECK_BODY("malloc_aligned11") {
    mi_heap_t* heap = mi_heap_new();
    void* p = mi_heap_malloc_aligned(heap, 33554426, 8);
    result = mi_heap_contains_block(heap, p);
    mi_heap_destroy(heap);
  }
  CHECK_BODY("mimalloc-aligned12") {
    void* p = mi_malloc_aligned(0x100, 0x100);
    result = (((uintptr_t)p % 0x100) == 0); // #602
    mi_free(p);
  }
  CHECK_BODY("mimalloc-aligned13") {
    bool ok = true;
    for( size_t size = 1; size <= (MI_SMALL_SIZE_MAX * 2) && ok; size++ ) {
      for(size_t align = 1; align <= size && ok; align *= 2 ) {
        void* p[10];
        for(int i = 0; i < 10 && ok; i++) {
          p[i] = mi_malloc_aligned(size,align);;
          ok = (p[i] != NULL && ((uintptr_t)(p[i]) % align) == 0);
        }
        for(int i = 0; i < 10 && ok; i++) {
          mi_free(p[i]);
        }       
        /*
        if (ok && align <= size && ((size + MI_PADDING_SIZE) & (align-1)) == 0) {
          size_t bsize = mi_good_size(size);
          ok = (align <= bsize && (bsize & (align-1)) == 0);
        }
        */
      }
    }
    result = ok;
  }
  CHECK_BODY("malloc-aligned-at1") {
    void* p = mi_malloc_aligned_at(48,32,0); result = (p != NULL && ((uintptr_t)(p) + 0) % 32 == 0); mi_free(p);
  };
  CHECK_BODY("malloc-aligned-at2") {
    void* p = mi_malloc_aligned_at(50,32,8); result = (p != NULL && ((uintptr_t)(p) + 8) % 32 == 0); mi_free(p);
  };
  CHECK_BODY("memalign1") {
    void* p;
    bool ok = true;
    for (int i = 0; i < 8 && ok; i++) {
      p = mi_memalign(16,8);
      ok = (p != NULL && (uintptr_t)(p) % 16 == 0); mi_free(p);
    }
    result = ok;
  };
  CHECK_BODY("zalloc-aligned-small1") {
    size_t zalloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_zalloc_aligned(zalloc_size, MI_MAX_ALIGN_SIZE * 2);
    result = mem_is_zero(p, zalloc_size);
    mi_free(p);
  };
  CHECK_BODY("rezalloc_aligned-small1") {
    size_t zalloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_zalloc_aligned(zalloc_size, MI_MAX_ALIGN_SIZE * 2);
    result = mem_is_zero(p, zalloc_size);
    zalloc_size *= 3;
    p = (uint8_t*)mi_rezalloc_aligned(p, zalloc_size, MI_MAX_ALIGN_SIZE * 2);
    result = result && mem_is_zero(p, zalloc_size);
    mi_free(p);
  };

  // ---------------------------------------------------
  // Reallocation
  // ---------------------------------------------------
  CHECK_BODY("realloc-null") {
    void* p = mi_realloc(NULL,4);
    result = (p != NULL);
    mi_free(p);
  };

  CHECK_BODY("realloc-null-sizezero") {
    void* p = mi_realloc(NULL,0);  // <https://en.cppreference.com/w/c/memory/realloc> "If ptr is NULL, the behavior is the same as calling malloc(new_size)."
    result = (p != NULL);
    mi_free(p);
  };

  CHECK_BODY("realloc-sizezero") {
    void* p = mi_malloc(4);
    void* q = mi_realloc(p, 0);
    result = (q != NULL);
    mi_free(q);
  };

  CHECK_BODY("reallocarray-null-sizezero") {
    void* p = mi_reallocarray(NULL,0,16);  // issue #574
    result = (p != NULL && errno == 0);
    mi_free(p);
  };

  // ---------------------------------------------------
  // Heaps
  // ---------------------------------------------------
  CHECK("heap_destroy", test_heap1());
  CHECK("heap_delete", test_heap2());

  //mi_stats_print(NULL);

  // ---------------------------------------------------
  // various
  // ---------------------------------------------------
  #if !defined(MI_TRACK_ASAN)   // realpath may leak with ASAN enabled (as the ASAN allocator intercepts it)
  CHECK_BODY("realpath") {
    char* s = mi_realpath( ".", NULL );
    // printf("realpath: %s\n",s);
    mi_free(s);
  };
  #endif

  CHECK("stl_allocator1", test_stl_allocator1());
  CHECK("stl_allocator2", test_stl_allocator2());

	CHECK("stl_heap_allocator1", test_stl_heap_allocator1());
	CHECK("stl_heap_allocator2", test_stl_heap_allocator2());
	CHECK("stl_heap_allocator3", test_stl_heap_allocator3());
	CHECK("stl_heap_allocator4", test_stl_heap_allocator4());

  // ---------------------------------------------------
  // Done
  // ---------------------------------------------------[]
  return print_test_summary();
}

// ---------------------------------------------------
// Larger test functions
// ---------------------------------------------------

bool test_heap1(void) {
  mi_heap_t* heap = mi_heap_new();
  int* p1 = mi_heap_malloc_tp(heap,int);
  int* p2 = mi_heap_malloc_tp(heap,int);
  *p1 = *p2 = 43;
  mi_heap_destroy(heap);
  return true;
}

bool test_heap2(void) {
  mi_heap_t* heap = mi_heap_new();
  int* p1 = mi_heap_malloc_tp(heap,int);
  int* p2 = mi_heap_malloc_tp(heap,int);
  mi_heap_delete(heap);
  *p1 = 42;
  mi_free(p1);
  mi_free(p2);
  return true;
}

bool test_stl_allocator1(void) {
#ifdef __cplusplus
  std::vector<int, mi_stl_allocator<int> > vec;
  vec.push_back(1);
  vec.pop_back();
  return vec.size() == 0;
#else
  return true;
#endif
}

struct some_struct  { int i; int j; double z; };

bool test_stl_allocator2(void) {
#ifdef __cplusplus
  std::vector<some_struct, mi_stl_allocator<some_struct> > vec;
  vec.push_back(some_struct());
  vec.pop_back();
  return vec.size() == 0;
#else
  return true;
#endif
}

bool test_stl_heap_allocator1(void) {
#ifdef __cplusplus
  std::vector<some_struct, mi_heap_stl_allocator<some_struct> > vec;
  vec.push_back(some_struct());
  vec.pop_back();
  return vec.size() == 0;
#else
  return true;
#endif
}

bool test_stl_heap_allocator2(void) {
#ifdef __cplusplus
  std::vector<some_struct, mi_heap_destroy_stl_allocator<some_struct> > vec;
  vec.push_back(some_struct());
  vec.pop_back();
  return vec.size() == 0;
#else
  return true;
#endif
}

bool test_stl_heap_allocator3(void) {
#ifdef __cplusplus
	mi_heap_t* heap = mi_heap_new();
	bool good = false;
	{
		mi_heap_stl_allocator<some_struct> myAlloc(heap);
		std::vector<some_struct, mi_heap_stl_allocator<some_struct> > vec(myAlloc);
		vec.push_back(some_struct());
		vec.pop_back();
		good = vec.size() == 0;
	}
	mi_heap_delete(heap);
  return good;
#else
  return true;
#endif
}

bool test_stl_heap_allocator4(void) {
#ifdef __cplusplus
	mi_heap_t* heap = mi_heap_new();
	bool good = false;
	{
		mi_heap_destroy_stl_allocator<some_struct> myAlloc(heap);
		std::vector<some_struct, mi_heap_destroy_stl_allocator<some_struct> > vec(myAlloc);
		vec.push_back(some_struct());
		vec.pop_back();
		good = vec.size() == 0;
	}
	mi_heap_destroy(heap);
  return good;
#else
  return true;
#endif
}
