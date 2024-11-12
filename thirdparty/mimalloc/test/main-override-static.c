#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>

#include <mimalloc.h>
#include <mimalloc-override.h>  // redefines malloc etc.


static void double_free1();
static void double_free2();
static void corrupt_free();
static void block_overflow1();
static void invalid_free();
static void test_aslr(void);
static void test_process_info(void);
static void test_reserved(void);
static void negative_stat(void);
static void alloc_huge(void);
static void test_heap_walk(void);
static void test_heap_arena(void);
static void test_align(void);

int main() {
  mi_version();
  mi_stats_reset();
  // detect double frees and heap corruption
  // double_free1();
  // double_free2();
  // corrupt_free();
  // block_overflow1();
  // test_aslr();
  // invalid_free();
  // test_reserved();
  // negative_stat();
  // test_heap_walk();
  // alloc_huge();
  // test_heap_walk();
  // test_heap_arena();
  // test_align();
  
  void* p1 = malloc(78);
  void* p2 = malloc(24);
  free(p1);
  p1 = mi_malloc(8);
  char* s = strdup("hello\n");
  free(p2);

  mi_heap_t* h = mi_heap_new();
  mi_heap_set_default(h);

  p2 = malloc(16);
  p1 = realloc(p1, 32);
  free(p1);
  free(p2);
  free(s);
  
  /* now test if override worked by allocating/freeing across the api's*/
  //p1 = mi_malloc(32);
  //free(p1);
  //p2 = malloc(32);
  //mi_free(p2);

  //mi_collect(true);
  //mi_stats_print(NULL);

  // test_process_info();
  
  return 0;
}

static void test_align() {
  void* p = mi_malloc_aligned(256, 256);
  if (((uintptr_t)p % 256) != 0) {
    fprintf(stderr, "%p is not 256 alignend!\n", p);
  }
}

static void invalid_free() {
  free((void*)0xBADBEEF);
  realloc((void*)0xBADBEEF,10);
}

static void block_overflow1() {
  uint8_t* p = (uint8_t*)mi_malloc(17);
  p[18] = 0;
  free(p);
}

// The double free samples come ArcHeap [1] by Insu Yun (issue #161)
// [1]: https://arxiv.org/pdf/1903.00503.pdf

static void double_free1() {
  void* p[256];
  //uintptr_t buf[256];

  p[0] = mi_malloc(622616);
  p[1] = mi_malloc(655362);
  p[2] = mi_malloc(786432);
  mi_free(p[2]);
  // [VULN] Double free
  mi_free(p[2]);
  p[3] = mi_malloc(786456);
  // [BUG] Found overlap
  // p[3]=0x429b2ea2000 (size=917504), p[1]=0x429b2e42000 (size=786432)
  fprintf(stderr, "p3: %p-%p, p1: %p-%p, p2: %p\n", p[3], (uint8_t*)(p[3]) + 786456, p[1], (uint8_t*)(p[1]) + 655362, p[2]);
}

static void double_free2() {
  void* p[256];
  //uintptr_t buf[256];
  // [INFO] Command buffer: 0x327b2000
  // [INFO] Input size: 182
  p[0] = malloc(712352);
  p[1] = malloc(786432);
  free(p[0]);
  // [VULN] Double free
  free(p[0]);
  p[2] = malloc(786440);
  p[3] = malloc(917504);
  p[4] = malloc(786440);
  // [BUG] Found overlap
  // p[4]=0x433f1402000 (size=917504), p[1]=0x433f14c2000 (size=786432)
  fprintf(stderr, "p1: %p-%p, p2: %p-%p\n", p[4], (uint8_t*)(p[4]) + 917504, p[1], (uint8_t*)(p[1]) + 786432);
}


// Try to corrupt the heap through buffer overflow
#define N   256
#define SZ  64

static void corrupt_free() {
  void* p[N];
  // allocate
  for (int i = 0; i < N; i++) {
    p[i] = malloc(SZ);
  }
  // free some
  for (int i = 0; i < N; i += (N/10)) {
    free(p[i]);
    p[i] = NULL;
  }
  // try to corrupt the free list
  for (int i = 0; i < N; i++) {
    if (p[i] != NULL) {
      memset(p[i], 0, SZ+8);
    }
  }
  // allocate more.. trying to trigger an allocation from a corrupted entry
  // this may need many allocations to get there (if at all)
  for (int i = 0; i < 4096; i++) {
    malloc(SZ);
  }
}

static void test_aslr(void) {
  void* p[256];
  p[0] = malloc(378200);
  p[1] = malloc(1134626);
  printf("p1: %p, p2: %p\n", p[0], p[1]);
}

static void test_process_info(void) {
  size_t elapsed = 0;
  size_t user_msecs = 0;
  size_t system_msecs = 0;
  size_t current_rss = 0;
  size_t peak_rss = 0;
  size_t current_commit = 0;
  size_t peak_commit = 0;
  size_t page_faults = 0;
  for (int i = 0; i < 100000; i++) {
    void* p = calloc(100,10);
    free(p);
  }
  mi_process_info(&elapsed, &user_msecs, &system_msecs, &current_rss, &peak_rss, &current_commit, &peak_commit, &page_faults);
  printf("\n\n*** process info: elapsed %3zd.%03zd s, user: %3zd.%03zd s, rss: %zd b, commit: %zd b\n\n", elapsed/1000, elapsed%1000, user_msecs/1000, user_msecs%1000, peak_rss, peak_commit);
}

static void test_reserved(void) {
#define KiB 1024ULL
#define MiB (KiB*KiB)
#define GiB (MiB*KiB)
  mi_reserve_os_memory(4*GiB, false, true);
  void* p1 = malloc(100);
  void* p2 = malloc(100000);
  void* p3 = malloc(2*GiB);
  void* p4 = malloc(1*GiB + 100000);
  free(p1);
  free(p2);
  free(p3);
  p3 = malloc(1*GiB);
  free(p4);
}



static void negative_stat(void) {
  int* p = mi_malloc(60000);
  mi_stats_print_out(NULL, NULL);
  *p = 100;
  mi_free(p);
  mi_stats_print_out(NULL, NULL);
}

static void alloc_huge(void) {
  void* p = mi_malloc(67108872);
  mi_free(p);
}

static bool test_visit(const mi_heap_t* heap, const mi_heap_area_t* area, void* block, size_t block_size, void* arg) {
  if (block == NULL) {
    printf("visiting an area with blocks of size %zu (including padding)\n", area->full_block_size);
  }
  else {
    printf("  block of size %zu (allocated size is %zu)\n", block_size, mi_usable_size(block));
  }
  return true;
}

static void test_heap_walk(void) {
  mi_heap_t* heap = mi_heap_new();
  mi_heap_malloc(heap, 16*2097152);
  mi_heap_malloc(heap, 2067152);
  mi_heap_malloc(heap, 2097160);
  mi_heap_malloc(heap, 24576);
  mi_heap_visit_blocks(heap, true, &test_visit, NULL);
}

static void test_heap_arena(void) {
  mi_arena_id_t arena_id;
  int err = mi_reserve_os_memory_ex(100 * 1024 * 1024, false /* commit */, false /* allow large */, true /* exclusive */, &arena_id);
  if (err) abort();
  mi_heap_t* heap = mi_heap_new_in_arena(arena_id);
  for (int i = 0; i < 500000; i++) {
    void* p = mi_heap_malloc(heap, 1024);
    if (p == NULL) {
      printf("out of memory after %d kb (expecting about 100_000kb)\n", i);
      break;
    }
  }
}

// ----------------------------
// bin size experiments
// ------------------------------

#if 0
#include <stdint.h>
#include <stdbool.h>

#define MI_INTPTR_SIZE 8
#define MI_LARGE_WSIZE_MAX (4*1024*1024 / MI_INTPTR_SIZE)

#define MI_BIN_HUGE 100
//#define MI_ALIGN2W

// Bit scan reverse: return the index of the highest bit.
static inline uint8_t mi_bsr32(uint32_t x);

#if defined(_MSC_VER)
#include <windows.h>
#include <intrin.h>
static inline uint8_t mi_bsr32(uint32_t x) {
  uint32_t idx;
  _BitScanReverse((DWORD*)&idx, x);
  return idx;
}
#elif defined(__GNUC__) || defined(__clang__)
static inline uint8_t mi_bsr32(uint32_t x) {
  return (31 - __builtin_clz(x));
}
#else
static inline uint8_t mi_bsr32(uint32_t x) {
  // de Bruijn multiplication, see <http://supertech.csail.mit.edu/papers/debruijn.pdf>
  static const uint8_t debruijn[32] = {
     31,  0, 22,  1, 28, 23, 18,  2, 29, 26, 24, 10, 19,  7,  3, 12,
     30, 21, 27, 17, 25,  9,  6, 11, 20, 16,  8,  5, 15,  4, 14, 13,
  };
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return debruijn[(x*0x076be629) >> 27];
}
#endif

/*
// Bit scan reverse: return the index of the highest bit.
uint8_t _mi_bsr(uintptr_t x) {
  if (x == 0) return 0;
  #if MI_INTPTR_SIZE==8
  uint32_t hi = (x >> 32);
  return (hi == 0 ? mi_bsr32((uint32_t)x) : 32 + mi_bsr32(hi));
  #elif MI_INTPTR_SIZE==4
  return mi_bsr32(x);
  #else
  # error "define bsr for non-32 or 64-bit platforms"
  #endif
}
*/


static inline size_t _mi_wsize_from_size(size_t size) {
  return (size + sizeof(uintptr_t) - 1) / sizeof(uintptr_t);
}

// Return the bin for a given field size.
// Returns MI_BIN_HUGE if the size is too large.
// We use `wsize` for the size in "machine word sizes",
// i.e. byte size == `wsize*sizeof(void*)`.
extern inline uint8_t _mi_bin8(size_t size) {
  size_t wsize = _mi_wsize_from_size(size);
  uint8_t bin;
  if (wsize <= 1) {
    bin = 1;
  }
#if defined(MI_ALIGN4W)
  else if (wsize <= 4) {
    bin = (uint8_t)((wsize+1)&~1); // round to double word sizes
  }
#elif defined(MI_ALIGN2W)
  else if (wsize <= 8) {
    bin = (uint8_t)((wsize+1)&~1); // round to double word sizes
  }
#else
  else if (wsize <= 8) {
    bin = (uint8_t)wsize;
  }
#endif
  else if (wsize > MI_LARGE_WSIZE_MAX) {
    bin = MI_BIN_HUGE;
  }
  else {
#if defined(MI_ALIGN4W)
    if (wsize <= 16) { wsize = (wsize+3)&~3; } // round to 4x word sizes
#endif
    wsize--;
    // find the highest bit
    uint8_t b = mi_bsr32((uint32_t)wsize);
    // and use the top 3 bits to determine the bin (~12.5% worst internal fragmentation).
    // - adjust with 3 because we use do not round the first 8 sizes
    //   which each get an exact bin
    bin = ((b << 2) + (uint8_t)((wsize >> (b - 2)) & 0x03)) - 3;
  }
  return bin;
}

static inline uint8_t _mi_bin4(size_t size) {
  size_t wsize = _mi_wsize_from_size(size);
  uint8_t bin;
  if (wsize <= 1) {
    bin = 1;
  }
#if defined(MI_ALIGN4W)
  else if (wsize <= 4) {
    bin = (uint8_t)((wsize+1)&~1); // round to double word sizes
  }
#elif defined(MI_ALIGN2W)
  else if (wsize <= 8) {
    bin = (uint8_t)((wsize+1)&~1); // round to double word sizes
  }
#else
  else if (wsize <= 8) {
    bin = (uint8_t)wsize;
  }
#endif
  else if (wsize > MI_LARGE_WSIZE_MAX) {
    bin = MI_BIN_HUGE;
  }
  else {
    uint8_t b = mi_bsr32((uint32_t)wsize);
    bin = ((b << 1) + (uint8_t)((wsize >> (b - 1)) & 0x01)) + 3;
  }
  return bin;
}

static size_t _mi_binx4(size_t bsize) {
  if (bsize==0) return 0;
  uint8_t b = mi_bsr32((uint32_t)bsize);
  if (b <= 1) return bsize;
  size_t bin = ((b << 1) | (bsize >> (b - 1))&0x01);
  return bin;
}

static size_t _mi_binx8(size_t bsize) {
  if (bsize<=1) return bsize;
  uint8_t b = mi_bsr32((uint32_t)bsize);
  if (b <= 2) return bsize;
  size_t bin = ((b << 2) | (bsize >> (b - 2))&0x03) - 5;
  return bin;
}

static void mi_bins(void) {
  //printf("  QNULL(1), /* 0 */ \\\n  ");
  size_t last_bin = 0;
  size_t min_bsize = 0;
  size_t last_bsize = 0;
  for (size_t bsize = 1; bsize < 2*1024; bsize++) {
    size_t size = bsize * 64 * 1024;
    size_t bin = _mi_binx8(bsize);
    if (bin != last_bin) {
      printf("min bsize: %6zd, max bsize: %6zd, bin: %6zd\n", min_bsize, last_bsize, last_bin);
      //printf("QNULL(%6zd), ", wsize);
      //if (last_bin%8 == 0) printf("/* %i */ \\\n  ", last_bin);
      last_bin = bin;
      min_bsize = bsize;
    }
    last_bsize = bsize;
  }
}
#endif
