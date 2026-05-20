/* Sort.c -- Sort functions
: Igor Pavlov : Public domain */

#include "Precomp.h"

#include "Sort.h"
#include "CpuArch.h"

#if (  (defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))) \
    || (defined(__clang__) && Z7_has_builtin(__builtin_prefetch)) \
    )
// the code with prefetch is slow for small arrays on x86.
// So we disable prefetch for x86.
#ifndef MY_CPU_X86
  // #pragma message("Z7_PREFETCH : __builtin_prefetch")
  #define Z7_PREFETCH(a)  __builtin_prefetch((a))
#endif

#elif defined(_WIN32) // || defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "7zWindows.h"

// NOTE: CLANG/GCC/MSVC can define different values for _MM_HINT_T0 / PF_TEMPORAL_LEVEL_1.
// For example, clang-cl can generate "prefetcht2" instruction for
// PreFetchCacheLine(PF_TEMPORAL_LEVEL_1) call.
// But we want to generate "prefetcht0" instruction.
// So for CLANG/GCC we must use __builtin_prefetch() in code branch above
// instead of PreFetchCacheLine() / _mm_prefetch().

// New msvc-x86 compiler generates "prefetcht0" instruction for PreFetchCacheLine() call.
// But old x86 cpus don't support "prefetcht0".
// So we will use PreFetchCacheLine(), only if we are sure that
// generated instruction is supported by all cpus of that isa.
#if defined(MY_CPU_AMD64) \
    || defined(MY_CPU_ARM64) \
    || defined(MY_CPU_IA64)
// we need to use additional braces for (a) in PreFetchCacheLine call, because
// PreFetchCacheLine macro doesn't use braces:
//   #define PreFetchCacheLine(l, a)  _mm_prefetch((CHAR CONST *) a, l)
  // #pragma message("Z7_PREFETCH : PreFetchCacheLine")
  #define Z7_PREFETCH(a)  PreFetchCacheLine(PF_TEMPORAL_LEVEL_1, (a))
#endif

#endif // _WIN32


#define PREFETCH_NO(p,k,s,size)

#ifndef Z7_PREFETCH
  #define SORT_PREFETCH(p,k,s,size)
#else

// #define PREFETCH_LEVEL 2  // use it if cache line is 32-bytes
#define PREFETCH_LEVEL 3  // it is fast for most cases (64-bytes cache line prefetch)
// #define PREFETCH_LEVEL 4  // it can be faster for big array (128-bytes prefetch)

#if PREFETCH_LEVEL == 0

  #define SORT_PREFETCH(p,k,s,size)

#else // PREFETCH_LEVEL != 0

/*
if  defined(USE_PREFETCH_FOR_ALIGNED_ARRAY)
    we prefetch one value per cache line.
    Use it if array is aligned for cache line size (64 bytes)
    or if array is small (less than L1 cache size).

if !defined(USE_PREFETCH_FOR_ALIGNED_ARRAY)
    we perfetch all cache lines that can be required.
    it can be faster for big unaligned arrays.
*/
  #define USE_PREFETCH_FOR_ALIGNED_ARRAY

// s == k * 2
#if 0 && PREFETCH_LEVEL <= 3 && defined(MY_CPU_X86_OR_AMD64)
  // x86 supports (lea r1*8+offset)
  #define PREFETCH_OFFSET(k,s)  ((s) << PREFETCH_LEVEL)
#else
  #define PREFETCH_OFFSET(k,s)  ((k) << (PREFETCH_LEVEL + 1))
#endif

#if 1 && PREFETCH_LEVEL <= 3 && defined(USE_PREFETCH_FOR_ALIGNED_ARRAY)
  #define PREFETCH_ADD_OFFSET   0
#else
  // last offset that can be reqiured in PREFETCH_LEVEL step:
  #define PREFETCH_RANGE        ((2 << PREFETCH_LEVEL) - 1)
  #define PREFETCH_ADD_OFFSET   PREFETCH_RANGE / 2
#endif

#if PREFETCH_LEVEL <= 3

#ifdef USE_PREFETCH_FOR_ALIGNED_ARRAY
  #define SORT_PREFETCH(p,k,s,size) \
  { const size_t s2 = PREFETCH_OFFSET(k,s) + PREFETCH_ADD_OFFSET; \
    if (s2 <= size) { \
      Z7_PREFETCH((p + s2)); \
  }}
#else /* for unaligned array */
  #define SORT_PREFETCH(p,k,s,size) \
  { const size_t s2 = PREFETCH_OFFSET(k,s) + PREFETCH_RANGE; \
    if (s2 <= size) { \
      Z7_PREFETCH((p + s2 - PREFETCH_RANGE)); \
      Z7_PREFETCH((p + s2)); \
  }}
#endif

#else // PREFETCH_LEVEL > 3

#ifdef USE_PREFETCH_FOR_ALIGNED_ARRAY
  #define SORT_PREFETCH(p,k,s,size) \
  { const size_t s2 = PREFETCH_OFFSET(k,s) + PREFETCH_RANGE - 16 / 2; \
    if (s2 <= size) { \
      Z7_PREFETCH((p + s2 - 16)); \
      Z7_PREFETCH((p + s2)); \
  }}
#else /* for unaligned array */
  #define SORT_PREFETCH(p,k,s,size) \
  { const size_t s2 = PREFETCH_OFFSET(k,s) + PREFETCH_RANGE; \
    if (s2 <= size) { \
      Z7_PREFETCH((p + s2 - PREFETCH_RANGE)); \
      Z7_PREFETCH((p + s2 - PREFETCH_RANGE / 2)); \
      Z7_PREFETCH((p + s2)); \
  }}
#endif

#endif // PREFETCH_LEVEL > 3
#endif // PREFETCH_LEVEL != 0
#endif // Z7_PREFETCH


#if defined(MY_CPU_ARM64) \
    /* || defined(MY_CPU_AMD64) */ \
    /* || defined(MY_CPU_ARM) && !defined(_MSC_VER) */
  // we want to use cmov, if cmov is very fast:
  // - this cmov version is slower for clang-x64.
  // - this cmov version is faster for gcc-arm64 for some fast arm64 cpus.
  #define Z7_FAST_CMOV_SUPPORTED
#endif
 
#ifdef Z7_FAST_CMOV_SUPPORTED
  // we want to use cmov here, if cmov is fast: new arm64 cpus.
  // we want the compiler to use conditional move for this branch
  #define GET_MAX_VAL(n0, n1, max_val_slow)  if (n0 < n1) n0 = n1;
#else
  // use this branch, if cpu doesn't support fast conditional move.
  // it uses slow array access reading:
  #define GET_MAX_VAL(n0, n1, max_val_slow)  n0 = max_val_slow;
#endif

#define HeapSortDown(p, k, size, temp, macro_prefetch) \
{ \
  for (;;) { \
    UInt32 n0, n1; \
    size_t s = k * 2; \
    if (s >= size) { \
      if (s == size) { \
        n0 = p[s]; \
        p[k] = n0; \
        if (temp < n0) k = s; \
      } \
      break; \
    } \
    n0 = p[k * 2]; \
    n1 = p[k * 2 + 1]; \
    s += n0 < n1; \
    GET_MAX_VAL(n0, n1, p[s]) \
    if (temp >= n0) break; \
    macro_prefetch(p, k, s, size) \
    p[k] = n0; \
    k = s; \
  } \
  p[k] = temp; \
}


/*
stage-1 : O(n) :
  we generate intermediate partially sorted binary tree:
  p[0]  : it's additional item for better alignment of tree structure in memory.
  p[1]
  p[2]       p[3]
  p[4] p[5]  p[6] p[7]
  ...
  p[x] >= p[x * 2]
  p[x] >= p[x * 2 + 1]
  
stage-2 : O(n)*log2(N):
  we move largest item p[0] from head of tree to the end of array
  and insert last item to sorted binary tree.
*/

// (p) must be aligned for cache line size (64-bytes) for best performance

void Z7_FASTCALL HeapSort(UInt32 *p, size_t size)
{
  if (size < 2)
    return;
  if (size == 2)
  {
    const UInt32 a0 = p[0];
    const UInt32 a1 = p[1];
    const unsigned k = a1 < a0;
    p[k] = a0;
    p[k ^ 1] = a1;
    return;
  }
  {
    // stage-1 : O(n)
    // we transform array to partially sorted binary tree.
    size_t i = --size / 2;
    // (size) now is the index of the last item in tree,
    // if (i)
    {
      do
      {
        const UInt32 temp = p[i];
        size_t k = i;
        HeapSortDown(p, k, size, temp, PREFETCH_NO)
      }
      while (--i);
    }
    {
      const UInt32 temp = p[0];
      const UInt32 a1 = p[1];
      if (temp < a1)
      {
        size_t k = 1;
        p[0] = a1;
        HeapSortDown(p, k, size, temp, PREFETCH_NO)
      }
    }
  }

  if (size < 3)
  {
    // size == 2
    const UInt32 a0 = p[0];
    p[0] = p[2];
    p[2] = a0;
    return;
  }
  if (size != 3)
  {
    // stage-2 : O(size) * log2(size):
    // we move largest item p[0] from head to the end of array,
    // and insert last item to sorted binary tree.
    do
    {
      const UInt32 temp = p[size];
      size_t k = p[2] < p[3] ? 3 : 2;
      p[size--] = p[0];
      p[0] = p[1];
      p[1] = p[k];
      HeapSortDown(p, k, size, temp, SORT_PREFETCH) // PREFETCH_NO
    }
    while (size != 3);
  }
  {
    const UInt32 a2 = p[2];
    const UInt32 a3 = p[3];
    const size_t k = a2 < a3;
    p[2] = p[1];
    p[3] = p[0];
    p[k] = a3;
    p[k ^ 1] = a2;
  }
}
