// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Misc. common utility functions
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdlib.h>
#include <string.h>  // for memcpy()
#include "src/webp/decode.h"
#include "src/webp/encode.h"
#include "src/webp/format_constants.h"  // for MAX_PALETTE_SIZE
#include "src/utils/color_cache_utils.h"
#include "src/utils/utils.h"

// If PRINT_MEM_INFO is defined, extra info (like total memory used, number of
// alloc/free etc) is printed. For debugging/tuning purpose only (it's slow,
// and not multi-thread safe!).
// An interesting alternative is valgrind's 'massif' tool:
//    https://valgrind.org/docs/manual/ms-manual.html
// Here is an example command line:
/*    valgrind --tool=massif --massif-out-file=massif.out \
               --stacks=yes --alloc-fn=WebPSafeMalloc --alloc-fn=WebPSafeCalloc
      ms_print massif.out
*/
// In addition:
// * if PRINT_MEM_TRAFFIC is defined, all the details of the malloc/free cycles
//   are printed.
// * if MALLOC_FAIL_AT is defined, the global environment variable
//   $MALLOC_FAIL_AT is used to simulate a memory error when calloc or malloc
//   is called for the nth time. Example usage:
//   export MALLOC_FAIL_AT=50 && ./examples/cwebp input.png
// * if MALLOC_LIMIT is defined, the global environment variable $MALLOC_LIMIT
//   sets the maximum amount of memory (in bytes) made available to libwebp.
//   This can be used to emulate environment with very limited memory.
//   Example: export MALLOC_LIMIT=64000000 && ./examples/dwebp picture.webp

// #define PRINT_MEM_INFO
// #define PRINT_MEM_TRAFFIC
// #define MALLOC_FAIL_AT
// #define MALLOC_LIMIT

//------------------------------------------------------------------------------
// Checked memory allocation

#if defined(PRINT_MEM_INFO)

#include <stdio.h>

static int num_malloc_calls = 0;
static int num_calloc_calls = 0;
static int num_free_calls = 0;
static int countdown_to_fail = 0;     // 0 = off

typedef struct MemBlock MemBlock;
struct MemBlock {
  void* ptr_;
  size_t size_;
  MemBlock* next_;
};

static MemBlock* all_blocks = NULL;
static size_t total_mem = 0;
static size_t total_mem_allocated = 0;
static size_t high_water_mark = 0;
static size_t mem_limit = 0;

static int exit_registered = 0;

static void PrintMemInfo(void) {
  fprintf(stderr, "\nMEMORY INFO:\n");
  fprintf(stderr, "num calls to: malloc = %4d\n", num_malloc_calls);
  fprintf(stderr, "              calloc = %4d\n", num_calloc_calls);
  fprintf(stderr, "              free   = %4d\n", num_free_calls);
  fprintf(stderr, "total_mem: %u\n", (uint32_t)total_mem);
  fprintf(stderr, "total_mem allocated: %u\n", (uint32_t)total_mem_allocated);
  fprintf(stderr, "high-water mark: %u\n", (uint32_t)high_water_mark);
  while (all_blocks != NULL) {
    MemBlock* b = all_blocks;
    all_blocks = b->next_;
    free(b);
  }
}

static void Increment(int* const v) {
  if (!exit_registered) {
#if defined(MALLOC_FAIL_AT)
    {
      const char* const malloc_fail_at_str = getenv("MALLOC_FAIL_AT");
      if (malloc_fail_at_str != NULL) {
        countdown_to_fail = atoi(malloc_fail_at_str);
      }
    }
#endif
#if defined(MALLOC_LIMIT)
    {
      const char* const malloc_limit_str = getenv("MALLOC_LIMIT");
#if MALLOC_LIMIT > 1
      mem_limit = (size_t)MALLOC_LIMIT;
#endif
      if (malloc_limit_str != NULL) {
        mem_limit = atoi(malloc_limit_str);
      }
    }
#endif
    (void)countdown_to_fail;
    (void)mem_limit;
    atexit(PrintMemInfo);
    exit_registered = 1;
  }
  ++*v;
}

static void AddMem(void* ptr, size_t size) {
  if (ptr != NULL) {
    MemBlock* const b = (MemBlock*)malloc(sizeof(*b));
    if (b == NULL) abort();
    b->next_ = all_blocks;
    all_blocks = b;
    b->ptr_ = ptr;
    b->size_ = size;
    total_mem += size;
    total_mem_allocated += size;
#if defined(PRINT_MEM_TRAFFIC)
#if defined(MALLOC_FAIL_AT)
    fprintf(stderr, "fail-count: %5d [mem=%u]\n",
            num_malloc_calls + num_calloc_calls, (uint32_t)total_mem);
#else
    fprintf(stderr, "Mem: %u (+%u)\n", (uint32_t)total_mem, (uint32_t)size);
#endif
#endif
    if (total_mem > high_water_mark) high_water_mark = total_mem;
  }
}

static void SubMem(void* ptr) {
  if (ptr != NULL) {
    MemBlock** b = &all_blocks;
    // Inefficient search, but that's just for debugging.
    while (*b != NULL && (*b)->ptr_ != ptr) b = &(*b)->next_;
    if (*b == NULL) {
      fprintf(stderr, "Invalid pointer free! (%p)\n", ptr);
      abort();
    }
    {
      MemBlock* const block = *b;
      *b = block->next_;
      total_mem -= block->size_;
#if defined(PRINT_MEM_TRAFFIC)
      fprintf(stderr, "Mem: %u (-%u)\n",
              (uint32_t)total_mem, (uint32_t)block->size_);
#endif
      free(block);
    }
  }
}

#else
#define Increment(v) do {} while (0)
#define AddMem(p, s) do {} while (0)
#define SubMem(p)    do {} while (0)
#endif

// Returns 0 in case of overflow of nmemb * size.
static int CheckSizeArgumentsOverflow(uint64_t nmemb, size_t size) {
  const uint64_t total_size = nmemb * size;
  if (nmemb == 0) return 1;
  if ((uint64_t)size > WEBP_MAX_ALLOCABLE_MEMORY / nmemb) return 0;
  if (!CheckSizeOverflow(total_size)) return 0;
#if defined(PRINT_MEM_INFO) && defined(MALLOC_FAIL_AT)
  if (countdown_to_fail > 0 && --countdown_to_fail == 0) {
    return 0;    // fake fail!
  }
#endif
#if defined(PRINT_MEM_INFO) && defined(MALLOC_LIMIT)
  if (mem_limit > 0) {
    const uint64_t new_total_mem = (uint64_t)total_mem + total_size;
    if (!CheckSizeOverflow(new_total_mem) ||
        new_total_mem > mem_limit) {
      return 0;   // fake fail!
    }
  }
#endif

  return 1;
}

void* WebPSafeMalloc(uint64_t nmemb, size_t size) {
  void* ptr;
  Increment(&num_malloc_calls);
  if (!CheckSizeArgumentsOverflow(nmemb, size)) return NULL;
  assert(nmemb * size > 0);
  ptr = malloc((size_t)(nmemb * size));
  AddMem(ptr, (size_t)(nmemb * size));
  return ptr;
}

void* WebPSafeCalloc(uint64_t nmemb, size_t size) {
  void* ptr;
  Increment(&num_calloc_calls);
  if (!CheckSizeArgumentsOverflow(nmemb, size)) return NULL;
  assert(nmemb * size > 0);
  ptr = calloc((size_t)nmemb, size);
  AddMem(ptr, (size_t)(nmemb * size));
  return ptr;
}

void WebPSafeFree(void* const ptr) {
  if (ptr != NULL) {
    Increment(&num_free_calls);
    SubMem(ptr);
  }
  free(ptr);
}

// Public API functions.

void* WebPMalloc(size_t size) {
  return WebPSafeMalloc(1, size);
}

void WebPFree(void* ptr) {
  WebPSafeFree(ptr);
}

//------------------------------------------------------------------------------

void WebPCopyPlane(const uint8_t* src, int src_stride,
                   uint8_t* dst, int dst_stride, int width, int height) {
  assert(src != NULL && dst != NULL);
  assert(abs(src_stride) >= width && abs(dst_stride) >= width);
  while (height-- > 0) {
    memcpy(dst, src, width);
    src += src_stride;
    dst += dst_stride;
  }
}

void WebPCopyPixels(const WebPPicture* const src, WebPPicture* const dst) {
  assert(src != NULL && dst != NULL);
  assert(src->width == dst->width && src->height == dst->height);
  assert(src->use_argb && dst->use_argb);
  WebPCopyPlane((uint8_t*)src->argb, 4 * src->argb_stride, (uint8_t*)dst->argb,
                4 * dst->argb_stride, 4 * src->width, src->height);
}

//------------------------------------------------------------------------------

#define COLOR_HASH_SIZE         (MAX_PALETTE_SIZE * 4)
#define COLOR_HASH_RIGHT_SHIFT  22  // 32 - log2(COLOR_HASH_SIZE).

int WebPGetColorPalette(const WebPPicture* const pic, uint32_t* const palette) {
  int i;
  int x, y;
  int num_colors = 0;
  uint8_t in_use[COLOR_HASH_SIZE] = { 0 };
  uint32_t colors[COLOR_HASH_SIZE];
  const uint32_t* argb = pic->argb;
  const int width = pic->width;
  const int height = pic->height;
  uint32_t last_pix = ~argb[0];   // so we're sure that last_pix != argb[0]
  assert(pic != NULL);
  assert(pic->use_argb);

  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      int key;
      if (argb[x] == last_pix) {
        continue;
      }
      last_pix = argb[x];
      key = VP8LHashPix(last_pix, COLOR_HASH_RIGHT_SHIFT);
      while (1) {
        if (!in_use[key]) {
          colors[key] = last_pix;
          in_use[key] = 1;
          ++num_colors;
          if (num_colors > MAX_PALETTE_SIZE) {
            return MAX_PALETTE_SIZE + 1;  // Exact count not needed.
          }
          break;
        } else if (colors[key] == last_pix) {
          break;  // The color is already there.
        } else {
          // Some other color sits here, so do linear conflict resolution.
          ++key;
          key &= (COLOR_HASH_SIZE - 1);  // Key mask.
        }
      }
    }
    argb += pic->argb_stride;
  }

  if (palette != NULL) {  // Fill the colors into palette.
    num_colors = 0;
    for (i = 0; i < COLOR_HASH_SIZE; ++i) {
      if (in_use[i]) {
        palette[num_colors] = colors[i];
        ++num_colors;
      }
    }
  }
  return num_colors;
}

#undef COLOR_HASH_SIZE
#undef COLOR_HASH_RIGHT_SHIFT

//------------------------------------------------------------------------------

#if defined(WEBP_NEED_LOG_TABLE_8BIT)
const uint8_t WebPLogTable8bit[256] = {   // 31 ^ clz(i)
  0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
};
#endif

//------------------------------------------------------------------------------
