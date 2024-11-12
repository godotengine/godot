/* ----------------------------------------------------------------------------
Copyright (c) 2018-2020, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/
#include "mimalloc.h"
#include "mimalloc/types.h"

#include "testhelper.h"

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------
bool check_zero_init(uint8_t* p, size_t size);
#if MI_DEBUG >= 2
bool check_debug_fill_uninit(uint8_t* p, size_t size);
bool check_debug_fill_freed(uint8_t* p, size_t size);
#endif

// ---------------------------------------------------------------------------
// Main testing
// ---------------------------------------------------------------------------
int main(void) {
  mi_option_disable(mi_option_verbose);

  // ---------------------------------------------------
  // Zeroing allocation
  // ---------------------------------------------------
  CHECK_BODY("zeroinit-zalloc-small") {
    size_t zalloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_zalloc(zalloc_size);
    result = check_zero_init(p, zalloc_size);
    mi_free(p);
  };
  CHECK_BODY("zeroinit-zalloc-large") {
    size_t zalloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_zalloc(zalloc_size);
    result = check_zero_init(p, zalloc_size);
    mi_free(p);
  };
  CHECK_BODY("zeroinit-zalloc_small") {
    size_t zalloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_zalloc_small(zalloc_size);
    result = check_zero_init(p, zalloc_size);
    mi_free(p);
  };

  CHECK_BODY("zeroinit-calloc-small") {
    size_t calloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_calloc(calloc_size, 1);
    result = check_zero_init(p, calloc_size);
    mi_free(p);
  };
  CHECK_BODY("zeroinit-calloc-large") {
    size_t calloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_calloc(calloc_size, 1);
    result = check_zero_init(p, calloc_size);
    mi_free(p);
  };

  CHECK_BODY("zeroinit-rezalloc-small") {
    size_t zalloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_zalloc(zalloc_size);
    result = check_zero_init(p, zalloc_size);
    zalloc_size *= 3;
    p = (uint8_t*)mi_rezalloc(p, zalloc_size);
    result &= check_zero_init(p, zalloc_size);
    mi_free(p);
  };
  CHECK_BODY("zeroinit-rezalloc-large") {
    size_t zalloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_zalloc(zalloc_size);
    result = check_zero_init(p, zalloc_size);
    zalloc_size *= 3;
    p = (uint8_t*)mi_rezalloc(p, zalloc_size);
    result &= check_zero_init(p, zalloc_size);
    mi_free(p);
  };

  CHECK_BODY("zeroinit-recalloc-small") {
    size_t calloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_calloc(calloc_size, 1);
    result = check_zero_init(p, calloc_size);
    calloc_size *= 3;
    p = (uint8_t*)mi_recalloc(p, calloc_size, 1);
    result &= check_zero_init(p, calloc_size);
    mi_free(p);
  };
  CHECK_BODY("zeroinit-recalloc-large") {
    size_t calloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_calloc(calloc_size, 1);
    result = check_zero_init(p, calloc_size);
    calloc_size *= 3;
    p = (uint8_t*)mi_recalloc(p, calloc_size, 1);
    result &= check_zero_init(p, calloc_size);
    mi_free(p);
  };

  // ---------------------------------------------------
  // Zeroing in aligned API
  // ---------------------------------------------------
  CHECK_BODY("zeroinit-zalloc_aligned-small") {
    size_t zalloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_zalloc_aligned(zalloc_size, MI_MAX_ALIGN_SIZE * 2);
    result = check_zero_init(p, zalloc_size);
    mi_free(p);
  };
  CHECK_BODY("zeroinit-zalloc_aligned-large") {
    size_t zalloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_zalloc_aligned(zalloc_size, MI_MAX_ALIGN_SIZE * 2);
    result = check_zero_init(p, zalloc_size);
    mi_free(p);
  };

  CHECK_BODY("zeroinit-calloc_aligned-small") {
    size_t calloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_calloc_aligned(calloc_size, 1, MI_MAX_ALIGN_SIZE * 2);
    result = check_zero_init(p, calloc_size);
    mi_free(p);
  };
  CHECK_BODY("zeroinit-calloc_aligned-large") {
    size_t calloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_calloc_aligned(calloc_size, 1, MI_MAX_ALIGN_SIZE * 2);
    result = check_zero_init(p, calloc_size);
    mi_free(p);
  };

  CHECK_BODY("zeroinit-rezalloc_aligned-small") {
    size_t zalloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_zalloc_aligned(zalloc_size, MI_MAX_ALIGN_SIZE * 2);
    result = check_zero_init(p, zalloc_size);
    zalloc_size *= 3;
    p = (uint8_t*)mi_rezalloc_aligned(p, zalloc_size, MI_MAX_ALIGN_SIZE * 2);
    result &= check_zero_init(p, zalloc_size);
    mi_free(p);
  };
  CHECK_BODY("zeroinit-rezalloc_aligned-large") {
    size_t zalloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_zalloc_aligned(zalloc_size, MI_MAX_ALIGN_SIZE * 2);
    result = check_zero_init(p, zalloc_size);
    zalloc_size *= 3;
    p = (uint8_t*)mi_rezalloc_aligned(p, zalloc_size, MI_MAX_ALIGN_SIZE * 2);
    result &= check_zero_init(p, zalloc_size);
    mi_free(p);
  };

  CHECK_BODY("zeroinit-recalloc_aligned-small") {
    size_t calloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_calloc_aligned(calloc_size, 1, MI_MAX_ALIGN_SIZE * 2);
    result = check_zero_init(p, calloc_size);
    calloc_size *= 3;
    p = (uint8_t*)mi_recalloc_aligned(p, calloc_size, 1, MI_MAX_ALIGN_SIZE * 2);
    result &= check_zero_init(p, calloc_size);
    mi_free(p);
  };
  CHECK_BODY("zeroinit-recalloc_aligned-large") {
    size_t calloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_calloc_aligned(calloc_size, 1, MI_MAX_ALIGN_SIZE * 2);
    result = check_zero_init(p, calloc_size);
    calloc_size *= 3;
    p = (uint8_t*)mi_recalloc_aligned(p, calloc_size, 1, MI_MAX_ALIGN_SIZE * 2);
    result &= check_zero_init(p, calloc_size);
    mi_free(p);
  };

#if (MI_DEBUG >= 2) && !MI_TSAN
  // ---------------------------------------------------
  // Debug filling
  // ---------------------------------------------------
  CHECK_BODY("uninit-malloc-small") {
    size_t malloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_malloc(malloc_size);
    result = check_debug_fill_uninit(p, malloc_size);
    mi_free(p);
  };
  CHECK_BODY("uninit-malloc-large") {
    size_t malloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_malloc(malloc_size);
    result = check_debug_fill_uninit(p, malloc_size);
    mi_free(p);
  };

  CHECK_BODY("uninit-malloc_small") {
    size_t malloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_malloc_small(malloc_size);
    result = check_debug_fill_uninit(p, malloc_size);
    mi_free(p);
  };

  CHECK_BODY("uninit-realloc-small") {
    size_t malloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_malloc(malloc_size);
    result = check_debug_fill_uninit(p, malloc_size);
    malloc_size *= 3;
    p = (uint8_t*)mi_realloc(p, malloc_size);
    result &= check_debug_fill_uninit(p, malloc_size);
    mi_free(p);
  };
  CHECK_BODY("uninit-realloc-large") {
    size_t malloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_malloc(malloc_size);
    result = check_debug_fill_uninit(p, malloc_size);
    malloc_size *= 3;
    p = (uint8_t*)mi_realloc(p, malloc_size);
    result &= check_debug_fill_uninit(p, malloc_size);
    mi_free(p);
  };

  CHECK_BODY("uninit-mallocn-small") {
    size_t malloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_mallocn(malloc_size, 1);
    result = check_debug_fill_uninit(p, malloc_size);
    mi_free(p);
  };
  CHECK_BODY("uninit-mallocn-large") {
    size_t malloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_mallocn(malloc_size, 1);
    result = check_debug_fill_uninit(p, malloc_size);
    mi_free(p);
  };

  CHECK_BODY("uninit-reallocn-small") {
    size_t malloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_mallocn(malloc_size, 1);
    result = check_debug_fill_uninit(p, malloc_size);
    malloc_size *= 3;
    p = (uint8_t*)mi_reallocn(p, malloc_size, 1);
    result &= check_debug_fill_uninit(p, malloc_size);
    mi_free(p);
  };
  CHECK_BODY("uninit-reallocn-large") {
    size_t malloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_mallocn(malloc_size, 1);
    result = check_debug_fill_uninit(p, malloc_size);
    malloc_size *= 3;
    p = (uint8_t*)mi_reallocn(p, malloc_size, 1);
    result &= check_debug_fill_uninit(p, malloc_size);
    mi_free(p);
  };

  CHECK_BODY("uninit-malloc_aligned-small") {
    size_t malloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_malloc_aligned(malloc_size, MI_MAX_ALIGN_SIZE * 2);
    result = check_debug_fill_uninit(p, malloc_size);
    mi_free(p);
  };
  CHECK_BODY("uninit-malloc_aligned-large") {
    size_t malloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_malloc_aligned(malloc_size, MI_MAX_ALIGN_SIZE * 2);
    result = check_debug_fill_uninit(p, malloc_size);
    mi_free(p);
  };

  CHECK_BODY("uninit-realloc_aligned-small") {
    size_t malloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_malloc_aligned(malloc_size, MI_MAX_ALIGN_SIZE * 2);
    result = check_debug_fill_uninit(p, malloc_size);
    malloc_size *= 3;
    p = (uint8_t*)mi_realloc_aligned(p, malloc_size, MI_MAX_ALIGN_SIZE * 2);
    result &= check_debug_fill_uninit(p, malloc_size);
    mi_free(p);
  };
  CHECK_BODY("uninit-realloc_aligned-large") {
    size_t malloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_malloc_aligned(malloc_size, MI_MAX_ALIGN_SIZE * 2);
    result = check_debug_fill_uninit(p, malloc_size);
    malloc_size *= 3;
    p = (uint8_t*)mi_realloc_aligned(p, malloc_size, MI_MAX_ALIGN_SIZE * 2);
    result &= check_debug_fill_uninit(p, malloc_size);
    mi_free(p);
  };

  #if !(MI_TRACK_VALGRIND || MI_TRACK_ASAN)
  CHECK_BODY("fill-freed-small") {
    size_t malloc_size = MI_SMALL_SIZE_MAX / 2;
    uint8_t* p = (uint8_t*)mi_malloc(malloc_size);
    mi_free(p);
    // First sizeof(void*) bytes will contain housekeeping data, skip these
    result = check_debug_fill_freed(p + sizeof(void*), malloc_size - sizeof(void*));
  };
  CHECK_BODY("fill-freed-large") {
    size_t malloc_size = MI_SMALL_SIZE_MAX * 2;
    uint8_t* p = (uint8_t*)mi_malloc(malloc_size);
    mi_free(p);
    // First sizeof(void*) bytes will contain housekeeping data, skip these
    result = check_debug_fill_freed(p + sizeof(void*), malloc_size - sizeof(void*));
  };
  #endif
#endif

  // ---------------------------------------------------
  // Done
  // ---------------------------------------------------[]
  return print_test_summary();
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------
bool check_zero_init(uint8_t* p, size_t size) {
  if(!p)
    return false;
  bool result = true;
  for (size_t i = 0; i < size; ++i) {
    result &= p[i] == 0;
  }
  return result;
}

#if MI_DEBUG >= 2
bool check_debug_fill_uninit(uint8_t* p, size_t size) {
#if MI_TRACK_VALGRIND || MI_TRACK_ASAN
  (void)p; (void)size;
  return true; // when compiled with valgrind we don't init on purpose
#else
  if(!p)
    return false;

  bool result = true;
  for (size_t i = 0; i < size; ++i) {
    result &= p[i] == MI_DEBUG_UNINIT;
  }
  return result;
#endif
}

bool check_debug_fill_freed(uint8_t* p, size_t size) {
#if MI_TRACK_VALGRIND
  (void)p; (void)size;
  return true; // when compiled with valgrind we don't fill on purpose
#else
  if(!p)
    return false;

  bool result = true;
  for (size_t i = 0; i < size; ++i) {
    result &= p[i] == MI_DEBUG_FREED;
  }
  return result;
#endif
}
#endif
