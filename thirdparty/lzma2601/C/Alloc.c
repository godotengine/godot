/* Alloc.c -- Memory allocation functions
: Igor Pavlov : Public domain */

#include "Precomp.h"

#ifdef _WIN32
#include "7zWindows.h"
#endif
#include <stdlib.h>

#include "Alloc.h"

#if defined(Z7_LARGE_PAGES) && defined(_WIN32) && \
    (!defined(Z7_WIN32_WINNT_MIN) || Z7_WIN32_WINNT_MIN < 0x0502)  // < Win2003 (xp-64)
  #define Z7_USE_DYN_GetLargePageMinimum
#endif

// for debug:
#if 0
#if defined(__CHERI__) && defined(__SIZEOF_POINTER__) && (__SIZEOF_POINTER__ == 16)
// #pragma message("=== Z7_ALLOC_NO_OFFSET_ALLOCATOR === ")
#define Z7_ALLOC_NO_OFFSET_ALLOCATOR
#endif
#endif

// #define SZ_ALLOC_DEBUG
/* use SZ_ALLOC_DEBUG to debug alloc/free operations */
#ifdef SZ_ALLOC_DEBUG

#include <string.h>
#include <stdio.h>
static int g_allocCount = 0;
#ifdef _WIN32
static int g_allocCountMid = 0;
#ifdef Z7_LARGE_PAGES
static int g_allocCountBig = 0;
#endif
#endif

#define CONVERT_INT_TO_STR(charType, tempSize) \
  char temp[tempSize]; unsigned i = 0; \
  while (val >= 10) { temp[i++] = (char)('0' + (unsigned)(val % 10)); val /= 10; } \
  *s++ = (charType)('0' + (unsigned)val); \
  while (i != 0) { i--; *s++ = temp[i]; } \
  *s = 0;

static void ConvertUInt64ToString(UInt64 val, char *s)
{
  CONVERT_INT_TO_STR(char, 24)
}

#define GET_HEX_CHAR(t) ((char)(((t < 10) ? ('0' + t) : ('A' + (t - 10)))))

static void ConvertUInt64ToHex(UInt64 val, char *s)
{
  UInt64 v = val;
  unsigned i;
  for (i = 1;; i++)
  {
    v >>= 4;
    if (v == 0)
      break;
  }
  s[i] = 0;
  do
  {
    unsigned t = (unsigned)(val & 0xF);
    val >>= 4;
    s[--i] = GET_HEX_CHAR(t);
  }
  while (i);
}

#define DEBUG_OUT_STREAM stderr

static void Print(const char *s)
{
  fputs(s, DEBUG_OUT_STREAM);
}

static void PrintAligned(const char *s, size_t align)
{
  size_t len = strlen(s);
  for(;;)
  {
    fputc(' ', DEBUG_OUT_STREAM);
    if (len >= align)
      break;
    ++len;
  }
  Print(s);
}

static void PrintLn(void)
{
  Print("\n");
}

static void PrintHex(UInt64 v, size_t align)
{
  char s[32];
  ConvertUInt64ToHex(v, s);
  PrintAligned(s, align);
}

static void PrintDec(int v, size_t align)
{
  char s[32];
  ConvertUInt64ToString((unsigned)v, s);
  PrintAligned(s, align);
}

static void PrintAddr(void *p)
{
  PrintHex((UInt64)(size_t)(ptrdiff_t)p, 12);
}


#define PRINT_REALLOC(name, cnt, size, ptr) { \
    Print(name " "); \
    if (!ptr) PrintDec(cnt++, 10); \
    PrintHex(size, 10); \
    PrintAddr(ptr); \
    PrintLn(); }

#define PRINT_ALLOC(name, cnt, size, ptr) { \
    Print(name " "); \
    PrintDec(cnt++, 10); \
    PrintHex(size, 10); \
    PrintAddr(ptr); \
    PrintLn(); }
 
#define PRINT_FREE(name, cnt, ptr) if (ptr) { \
    Print(name " "); \
    PrintDec(--cnt, 10); \
    PrintAddr(ptr); \
    PrintLn(); }
 
#else

#ifdef _WIN32
#ifdef Z7_LARGE_PAGES
#define PRINT_ALLOC(name, cnt, size, ptr)
#endif
#endif
#define PRINT_FREE(name, cnt, ptr)
#define Print(s)
#define PrintLn()
#ifndef Z7_ALLOC_NO_OFFSET_ALLOCATOR
#define PrintHex(v, align)
#endif
#define PrintAddr(p)

#endif


/*
by specification:
  malloc(non_NULL, 0)   : returns NULL or a unique pointer value that can later be successfully passed to free()
  realloc(NULL, size)   : the call is equivalent to malloc(size)
  realloc(non_NULL, 0)  : the call is equivalent to free(ptr)

in main compilers:
  malloc(0)             : returns non_NULL
  realloc(NULL,     0)  : returns non_NULL
  realloc(non_NULL, 0)  : returns NULL
*/


void *MyAlloc(size_t size)
{
  if (size == 0)
    return NULL;
  // PRINT_ALLOC("Alloc    ", g_allocCount, size, NULL)
  #ifdef SZ_ALLOC_DEBUG
  {
    void *p = malloc(size);
    if (p)
    {
      PRINT_ALLOC("Alloc    ", g_allocCount, size, p)
    }
    return p;
  }
  #else
  return malloc(size);
  #endif
}

void MyFree(void *address)
{
  PRINT_FREE("Free    ", g_allocCount, address)
  
  free(address);
}

void *MyRealloc(void *address, size_t size)
{
  if (size == 0)
  {
    MyFree(address);
    return NULL;
  }
  // PRINT_REALLOC("Realloc  ", g_allocCount, size, address)
  #ifdef SZ_ALLOC_DEBUG
  {
    void *p = realloc(address, size);
    if (p)
    {
      PRINT_REALLOC("Realloc    ", g_allocCount, size, address)
    }
    return p;
  }
  #else
  return realloc(address, size);
  #endif
}


#ifdef _WIN32

void *MidAlloc(size_t size)
{
  if (size == 0)
    return NULL;
  #ifdef SZ_ALLOC_DEBUG
  {
    void *p = VirtualAlloc(NULL, size, MEM_COMMIT, PAGE_READWRITE);
    if (p)
    {
      PRINT_ALLOC("Alloc-Mid", g_allocCountMid, size, p)
    }
    return p;
  }
  #else
  return VirtualAlloc(NULL, size, MEM_COMMIT, PAGE_READWRITE);
  #endif
}

void MidFree(void *address)
{
  PRINT_FREE("Free-Mid", g_allocCountMid, address)

  if (!address)
    return;
  VirtualFree(address, 0, MEM_RELEASE);
}

#ifdef Z7_LARGE_PAGES
// #pragma message("Z7_LARGE_PAGES")

#ifdef MEM_LARGE_PAGES
  #define MY_MEM_LARGE_PAGES  MEM_LARGE_PAGES
#else
  #define MY_MEM_LARGE_PAGES  0x20000000
#endif

extern
size_t g_LargePageSize;
size_t g_LargePageSize = 0;
extern
size_t g_LargePageThresholdMin;
size_t g_LargePageThresholdMin = 0;
extern
UInt32 g_LargePageFlags;
UInt32 g_LargePageFlags = 0;

void *BigAlloc(size_t size)
{
  if (size == 0)
    return NULL;

  PRINT_ALLOC("Alloc-Big", g_allocCountBig, size, NULL)

  #ifdef Z7_LARGE_PAGES
  {
    const size_t ps = g_LargePageSize - 1;
    if (ps < (1u << 30) && size > g_LargePageThresholdMin)
    {
      const size_t size2 = (size + ps) & ~ps;
      if (size2 >= size)
      {
        void *p = VirtualAlloc(NULL, size2, MEM_COMMIT | MY_MEM_LARGE_PAGES, PAGE_READWRITE);
        if (p)
        {
          PRINT_ALLOC("Alloc-BM ", g_allocCountMid, size2, p)
          return p;
        }
        if (g_LargePageFlags & Z7_LARGE_PAGES_FLAG_FAIL_STOP)
          return p;
      }
    }
  }
  #endif

  return MidAlloc(size);
}

void BigFree(void *address)
{
  PRINT_FREE("Free-Big", g_allocCountBig, address)
  MidFree(address);
}

#endif // Z7_LARGE_PAGES
#endif // _WIN32


static void *SzAlloc(ISzAllocPtr p, size_t size) { UNUSED_VAR(p)  return MyAlloc(size); }
static void SzFree(ISzAllocPtr p, void *address) { UNUSED_VAR(p)  MyFree(address); }
const ISzAlloc g_Alloc = { SzAlloc, SzFree };

#ifdef _WIN32
static void *SzMidAlloc(ISzAllocPtr p, size_t size) { UNUSED_VAR(p)  return MidAlloc(size); }
static void SzMidFree(ISzAllocPtr p, void *address) { UNUSED_VAR(p)  MidFree(address); }
const ISzAlloc g_MidAlloc = { SzMidAlloc, SzMidFree };
#endif

#if defined(Z7_LARGE_PAGES)
static void *SzBigAlloc(ISzAllocPtr p, size_t size) { UNUSED_VAR(p)  return BigAlloc(size); }
static void SzBigFree(ISzAllocPtr p, void *address) { UNUSED_VAR(p)  BigFree(address); }
const ISzAlloc g_BigAlloc = { SzBigAlloc, SzBigFree };
#endif

#ifndef Z7_ALLOC_NO_OFFSET_ALLOCATOR

#define ADJUST_ALLOC_SIZE 0
/*
#define ADJUST_ALLOC_SIZE (sizeof(void *) - 1)
*/
/*
  Use (ADJUST_ALLOC_SIZE = (sizeof(void *) - 1)), if
     MyAlloc() can return address that is NOT multiple of sizeof(void *).
*/

/*
  uintptr_t : <stdint.h> C99 (optional)
            : unsupported in VS6
*/
typedef
  #ifdef _WIN32
    UINT_PTR
  #elif 1
    uintptr_t
  #else
    ptrdiff_t
  #endif
    MY_uintptr_t;

#if 0 \
    || (defined(__CHERI__) \
    || defined(__SIZEOF_POINTER__) && (__SIZEOF_POINTER__ > 8))
// for 128-bit pointers (cheri):
#define MY_ALIGN_PTR_DOWN(p, align)  \
    ((void *)((char *)(p) - ((size_t)(MY_uintptr_t)(p) & ((align) - 1))))
#else
#define MY_ALIGN_PTR_DOWN(p, align) \
    ((void *)((((MY_uintptr_t)(p)) & ~((MY_uintptr_t)(align) - 1))))
#endif

#endif

#ifndef _WIN32
#include <unistd.h> // for _POSIX_ADVISORY_INFO : for some linux
#if (defined(Z7_ALLOC_NO_OFFSET_ALLOCATOR) \
        || defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L) \
        || defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L) \
        || defined(__APPLE__) \
        /* || defined(__linux__) */)
  #define USE_posix_memalign
  // #pragma message("USE_posix_memalign")
#endif
#endif

#ifndef USE_posix_memalign
#define MY_ALIGN_PTR_UP_PLUS(p, align) MY_ALIGN_PTR_DOWN(((char *)(p) + (align) + ADJUST_ALLOC_SIZE), align)
#endif

/*
  This posix_memalign() is for test purposes only.
  We also need special Free() function instead of free(),
  if this posix_memalign() is used.
*/

/*
static int posix_memalign(void **ptr, size_t align, size_t size)
{
  size_t newSize = size + align;
  void *p;
  void *pAligned;
  *ptr = NULL;
  if (newSize < size)
    return 12; // ENOMEM
  p = MyAlloc(newSize);
  if (!p)
    return 12; // ENOMEM
  pAligned = MY_ALIGN_PTR_UP_PLUS(p, align);
  ((void **)pAligned)[-1] = p;
  *ptr = pAligned;
  return 0;
}
*/

/*
  ALLOC_ALIGN_SIZE >= sizeof(void *)
  ALLOC_ALIGN_SIZE >= cache_line_size
*/

#define ALLOC_ALIGN_SIZE ((size_t)1 << 7)

void *z7_AlignedAlloc(size_t size)
{
#ifndef USE_posix_memalign
  
  void *p;
  void *pAligned;
  size_t newSize;

  /* also we can allocate additional dummy ALLOC_ALIGN_SIZE bytes after aligned
     block to prevent cache line sharing with another allocated blocks */

  newSize = size + ALLOC_ALIGN_SIZE * 1 + ADJUST_ALLOC_SIZE;
  if (newSize < size)
    return NULL;

  p = MyAlloc(newSize);
  
  if (!p)
    return NULL;
  pAligned = MY_ALIGN_PTR_UP_PLUS(p, ALLOC_ALIGN_SIZE);

  Print(" size="); PrintHex(size, 8);
  Print(" a_size="); PrintHex(newSize, 8);
  Print(" ptr="); PrintAddr(p);
  Print(" a_ptr="); PrintAddr(pAligned);
  PrintLn();

  ((void **)pAligned)[-1] = p;

  return pAligned;

#else

  void *p;
  if (posix_memalign(&p, ALLOC_ALIGN_SIZE, size))
    return NULL;

  Print(" posix_memalign="); PrintAddr(p);
  PrintLn();

  return p;

#endif
}


void z7_AlignedFree(void *address)
{
#ifndef USE_posix_memalign
  if (address)
    MyFree(((void **)address)[-1]);
#else
  free(address);
#endif
}


static void *SzAlignedAlloc(ISzAllocPtr pp, size_t size)
{
  UNUSED_VAR(pp)
  return z7_AlignedAlloc(size);
}


static void SzAlignedFree(ISzAllocPtr pp, void *address)
{
  UNUSED_VAR(pp)
#ifndef USE_posix_memalign
  if (address)
    MyFree(((void **)address)[-1]);
#else
  free(address);
#endif
}

#ifndef _WIN32

#ifdef Z7_LARGE_PAGES

#if 0 // 1 for debug
  #include <stdio.h>
  #include <string.h>  // for strerror()
  #define PRF(x) x
#else
  #define PRF(x)
#endif

#ifdef USE_posix_memalign
  /* madvise():
     glibc <= 2.19 : _BSD_SOURCE
     glibc  > 2.19 : _DEFAULT_SOURCE
  */
  /* && (defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE)) */
#if 1 && !defined(Z7_NO_MADVISE) && \
  (defined(__linux__) || defined(__unix__) || defined(__APPLE__))
#include <sys/mman.h> // for madvise
// #pragma message("sys/mman.h")
#if (defined(MADV_HUGEPAGE) && defined(MADV_NOHUGEPAGE))
  #define Z7_USE_BIG_ALLOC_MADVISE
  // #pragma message("Z7_USE_BIG_ALLOC_MADVISE")
#endif
#endif
#endif // USE_posix_memalign

#ifdef Z7_USE_BIG_ALLOC_MADVISE
#define LARGE_PAGE_SIZE_DEFAULT (1 << 21)
#else
#define LARGE_PAGE_SIZE_DEFAULT 0
#endif

extern
size_t g_LargePageSize;
size_t g_LargePageSize = LARGE_PAGE_SIZE_DEFAULT;
extern
size_t g_LargePageThresholdMin;
size_t g_LargePageThresholdMin = LARGE_PAGE_SIZE_DEFAULT / 2;
extern
UInt32 g_LargePageFlags;
UInt32 g_LargePageFlags = 0;

void *BigAlloc(size_t size)
{
  if (size == 0)
    return NULL;
#ifdef USE_posix_memalign
  {
    const size_t pageSize = g_LargePageSize;
    void *buf = NULL; // on Linux (and other systems), posix_memalign() does not modify memptr on failure (POSIX.1-2008 TC2).
    PRF(printf("\nBigAlloc 0x%08x=%5uMB", (unsigned)(size), (unsigned)(size >> 20));)
    if (pageSize && size > g_LargePageThresholdMin)
    {
      int res;
      const size_t mask = pageSize - 1;
      /* we can allocate aligned size, so data at the end of buffer also will use huge page
         if (size2 for madvise() is not aligned for huge page size)
           { Last data block will use small pages. It reduces memory allocation,
             but last data block with small pages can work slower.
             It's useful, if we have very large HUGE_PAGE: 32MB or 512MB. }
      */
      size_t size2 = (size + mask) & ~mask;
      if (size2 < size || (size & mask) <= g_LargePageThresholdMin)
        size2 = size;
      res = posix_memalign(&buf, pageSize, size2);
      PRF(printf(" posix_memalign size=0x%08x=%5uMB align=%u",
          (unsigned)(size2), (unsigned)(size2 >> 20), (unsigned)pageSize);)
      PRF(printf(" buf=%p", (void *)buf);)
      if (res == 0)
      {
#ifdef Z7_USE_BIG_ALLOC_MADVISE
        if ((g_LargePageFlags & Z7_LARGE_PAGES_FLAG_NO_MADVISE) == 0)
        {
          // Advise the kernel to use huge pages for this memory range
          // MADV_HUGEPAGE / MADV_NOHUGEPAGE : since Linux 2.6.38
          // madvise() only operates on whole pages, therefore addr must be page-aligned (4KB/8KB/16KB/64KB).
          // The value of size is rounded up to a multiple of page size.
          PRF(printf(" madvise g_LargePageFlags=%x", (unsigned)g_LargePageFlags);)
          res = madvise(buf, size2, (g_LargePageFlags & Z7_LARGE_PAGES_FLAG_NO_HUGEPAGE) ? MADV_NOHUGEPAGE : MADV_HUGEPAGE);
          if (res)
          {
            PRF(printf("\nERROR res=%d, errno=%d=%s\n", res, (int)errno, strerror(errno));)
            if (g_LargePageFlags & Z7_LARGE_PAGES_FLAG_FAIL_STOP)
            {
              free(buf);
              return NULL;
            }
          }
        }
#endif // Z7_USE_BIG_ALLOC_MADVISE
        PRF(printf("\n");)
        return buf;
      }
      PRF(printf("\nERROR res=%d=%s\n", res, strerror(res));)
      if (g_LargePageFlags & Z7_LARGE_PAGES_FLAG_FAIL_STOP)
        return NULL;
      // (res == ENOMEM) "Out of memory" is possible, if pageSize is too big.
      // so we do second attempt with smaller alignment
    }
  }
#endif // !USE_posix_memalign
  PRF(printf(" z7_AlignedAlloc size=0x%08x=%5uMB\n", (unsigned)(size), (unsigned)(size >> 20));)
  return z7_AlignedAlloc(size);
}


void BigFree(void *address)
{
  z7_AlignedFree(address);
}
#endif // Z7_LARGE_PAGES
#endif // !_WIN32


#ifdef Z7_LARGE_PAGES
void z7_LargePage_Set(UInt32 flags, size_t pageSize, size_t threshold)
{
  g_LargePageFlags = flags;

#ifdef _WIN32
  if ((flags & Z7_LARGE_PAGES_FLAG_USE_HUGEPAGE) == 0)
  {
    g_LargePageSize = 0;
    g_LargePageThresholdMin = 0;
  }
  else
  {
    if ((flags & Z7_LARGE_PAGES_FLAG_DIRECT_PAGE_SIZE) == 0)
    {
#ifdef Z7_USE_DYN_GetLargePageMinimum
      Z7_DIAGNOSTIC_IGNORE_CAST_FUNCTION
typedef SIZE_T (WINAPI *Func_GetLargePageMinimum)(VOID);
      const
        Func_GetLargePageMinimum fn =
       (Func_GetLargePageMinimum) Z7_CAST_FUNC_C GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")),
            "GetLargePageMinimum");
      if (fn)
        pageSize = fn();
      else
        pageSize = 0;
#else
      pageSize = GetLargePageMinimum();
#endif
      if (pageSize & (pageSize - 1))
        pageSize = 0;
    }
    g_LargePageSize = pageSize;
    if ((flags & Z7_LARGE_PAGES_FLAG_DIRECT_THRESHOLD) == 0)
      threshold = pageSize / 2;
    g_LargePageThresholdMin = threshold;
  }

#else // !_WIN32

  if (flags & Z7_LARGE_PAGES_FLAG_NO_PAGECODE)
  {
    g_LargePageSize = 0;
    g_LargePageThresholdMin = 0;
  }
  else
  {
    if ((flags & Z7_LARGE_PAGES_FLAG_DIRECT_PAGE_SIZE) == 0)
      pageSize = LARGE_PAGE_SIZE_DEFAULT;
    g_LargePageSize = pageSize;
    if ((flags & Z7_LARGE_PAGES_FLAG_DIRECT_THRESHOLD) == 0)
      threshold = pageSize / 2;
    g_LargePageThresholdMin = threshold;
  }
  // PRF(printf("\ng_LargePageSize=%x g_LargePageThresholdMin = %x g_LargePageFlags = %x", (unsigned)g_LargePageSize, (unsigned)g_LargePageThresholdMin, (unsigned)g_LargePageFlags);)
#endif // !_WIN32
}
#endif // Z7_LARGE_PAGES

const ISzAlloc g_AlignedAlloc = { SzAlignedAlloc, SzAlignedFree };



/* we align ptr to support cases where CAlignOffsetAlloc::offset is not multiply of sizeof(void *) */
#ifndef Z7_ALLOC_NO_OFFSET_ALLOCATOR
#if 1
  #define MY_ALIGN_PTR_DOWN_1(p)  MY_ALIGN_PTR_DOWN(p, sizeof(void *))
  #define REAL_BLOCK_PTR_VAR(p)  ((void **)MY_ALIGN_PTR_DOWN_1(p))[-1]
#else
  // we can use this simplified code,
  // if (CAlignOffsetAlloc::offset == (k * sizeof(void *))
  #define REAL_BLOCK_PTR_VAR(p)  (((void **)(p))[-1])
#endif
#endif


#if 0
#ifndef Z7_ALLOC_NO_OFFSET_ALLOCATOR
#include <stdio.h>
static void PrintPtr(const char *s, const void *p)
{
  const Byte *p2 = (const Byte *)&p;
  unsigned i;
  printf("%s %p ", s, p);
  for (i = sizeof(p); i != 0;)
  {
    i--;
    printf("%02x", p2[i]);
  }
  printf("\n");
}
#endif
#endif


static void *AlignOffsetAlloc_Alloc(ISzAllocPtr pp, size_t size)
{
#if defined(Z7_ALLOC_NO_OFFSET_ALLOCATOR)
  UNUSED_VAR(pp)
  return z7_AlignedAlloc(size);
#else
  const CAlignOffsetAlloc *p = Z7_CONTAINER_FROM_VTBL_CONST(pp, CAlignOffsetAlloc, vt);
  void *adr;
  void *pAligned;
  size_t newSize;
  size_t extra;
  size_t alignSize = (size_t)1 << p->numAlignBits;

  if (alignSize < sizeof(void *))
    alignSize = sizeof(void *);
  
  if (p->offset >= alignSize)
    return NULL;

  /* also we can allocate additional dummy ALLOC_ALIGN_SIZE bytes after aligned
     block to prevent cache line sharing with another allocated blocks */
  extra = p->offset & (sizeof(void *) - 1);
  newSize = size + alignSize + extra + ADJUST_ALLOC_SIZE;
  if (newSize < size)
    return NULL;

  adr = ISzAlloc_Alloc(p->baseAlloc, newSize);
  
  if (!adr)
    return NULL;

  pAligned = (char *)MY_ALIGN_PTR_DOWN((char *)adr +
      alignSize - p->offset + extra + ADJUST_ALLOC_SIZE, alignSize) + p->offset;

#if 0
  printf("\nalignSize = %6x, offset=%6x, size=%8x \n", (unsigned)alignSize, (unsigned)p->offset, (unsigned)size);
  PrintPtr("base", adr);
  PrintPtr("alig", pAligned);
#endif

  PrintLn();
  Print("- Aligned: ");
  Print(" size="); PrintHex(size, 8);
  Print(" a_size="); PrintHex(newSize, 8);
  Print(" ptr="); PrintAddr(adr);
  Print(" a_ptr="); PrintAddr(pAligned);
  PrintLn();

  REAL_BLOCK_PTR_VAR(pAligned) = adr;

  return pAligned;
#endif
}


static void AlignOffsetAlloc_Free(ISzAllocPtr pp, void *address)
{
#if defined(Z7_ALLOC_NO_OFFSET_ALLOCATOR)
  UNUSED_VAR(pp)
  z7_AlignedFree(address);
#else
  if (address)
  {
    const CAlignOffsetAlloc *p = Z7_CONTAINER_FROM_VTBL_CONST(pp, CAlignOffsetAlloc, vt);
    PrintLn();
    Print("- Aligned Free: ");
    PrintLn();
    ISzAlloc_Free(p->baseAlloc, REAL_BLOCK_PTR_VAR(address));
  }
#endif
}


void AlignOffsetAlloc_CreateVTable(CAlignOffsetAlloc *p)
{
  p->vt.Alloc = AlignOffsetAlloc_Alloc;
  p->vt.Free = AlignOffsetAlloc_Free;
}
