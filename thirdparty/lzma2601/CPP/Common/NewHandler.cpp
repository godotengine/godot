// NewHandler.cpp
 
#include "StdAfx.h"

#include <stdlib.h>

#include "NewHandler.h"

// #define DEBUG_MEMORY_LEAK

#ifndef DEBUG_MEMORY_LEAK

#ifdef Z7_REDEFINE_OPERATOR_NEW

/*
void * my_new(size_t size)
{
  // void *p = ::HeapAlloc(::GetProcessHeap(), 0, size);
  if (size == 0)
    size = 1;
  void *p = ::malloc(size);
  if (!p)
    throw CNewException();
  return p;
}

void my_delete(void *p) throw()
{
  // if (!p) return; ::HeapFree(::GetProcessHeap(), 0, p);
  ::free(p);
}

void * my_Realloc(void *p, size_t newSize, size_t oldSize)
{
  void *newBuf = my_new(newSize);
  if (oldSize != 0)
    memcpy(newBuf, p, oldSize);
  my_delete(p);
  return newBuf;
}
*/

void *
#ifdef _MSC_VER
__cdecl
#endif
operator new(size_t size)
{
  /* by C++ specification:
       if (size == 0), operator new(size) returns non_NULL pointer.
     If (operator new(0) returns NULL), it's out of specification.
       but some calling code can work correctly even in this case too. */
  // if (size == 0) return NULL; // for debug only. don't use it

  /* malloc(0) returns non_NULL in main compilers, as we need here.
     But specification also allows malloc(0) to return NULL.
     So we change (size=0) to (size=1) here to get real non_NULL pointer */
  if (size == 0)
    size = 1;
  // void *p = ::HeapAlloc(::GetProcessHeap(), 0, size);
  // void *p = ::MyAlloc(size);  // note: MyAlloc(0) returns NULL
  void *p = ::malloc(size);
  if (!p)
    throw CNewException();
  return p;
}


#if defined(_MSC_VER) && _MSC_VER == 1600
// vs2010 has no throw() by default ?
#pragma warning(push)
#pragma warning(disable : 4986) // 'operator delete': exception specification does not match previous declaration
#endif

void
#ifdef _MSC_VER
__cdecl
#endif
operator delete(void *p) throw()
{
  // if (!p) return; ::HeapFree(::GetProcessHeap(), 0, p);
  // MyFree(p);
  ::free(p);
}

/* we define operator delete(void *p, size_t n) because
   vs2022 compiler uses delete(void *p, size_t n), and
   we want to mix files from different compilers:
     - old vc6 linker
     - old vc6 complier
     - new vs2022 complier
*/
void
#ifdef _MSC_VER
__cdecl
#endif
operator delete(void *p, size_t n) throw()
{
  UNUSED_VAR(n)
  ::free(p);
}

#if defined(_MSC_VER) && _MSC_VER == 1600
#pragma warning(pop)
#endif

/*
void *
#ifdef _MSC_VER
__cdecl
#endif
operator new[](size_t size)
{
  // void *p = ::HeapAlloc(::GetProcessHeap(), 0, size);
  if (size == 0)
    size = 1;
  void *p = ::malloc(size);
  if (!p)
    throw CNewException();
  return p;
}

void
#ifdef _MSC_VER
__cdecl
#endif
operator delete[](void *p) throw()
{
  // if (!p) return; ::HeapFree(::GetProcessHeap(), 0, p);
  ::free(p);
}
*/

#endif

#else

#include <stdio.h>

// #pragma init_seg(lib)
/*
const int kDebugSize = 1000000;
static void *a[kDebugSize];
static int g_index = 0;

class CC
{
public:
  CC()
  {
    for (int i = 0; i < kDebugSize; i++)
      a[i] = 0;
  }
  ~CC()
  {
    printf("\nDestructor: %d\n", numAllocs);
    for (int i = 0; i < kDebugSize; i++)
      if (a[i] != 0)
        return;
  }
} g_CC;
*/

#ifdef _WIN32
static bool wasInit = false;
static CRITICAL_SECTION cs;
#endif

static int numAllocs = 0;

void *
#ifdef _MSC_VER
__cdecl
#endif
operator new(size_t size)
{
 #ifdef _WIN32
  if (!wasInit)
  {
    InitializeCriticalSection(&cs);
    wasInit = true;
  }
  EnterCriticalSection(&cs);

  numAllocs++;
  int loc = numAllocs;
  void *p = HeapAlloc(GetProcessHeap(), 0, size);
  /*
  if (g_index < kDebugSize)
  {
    a[g_index] = p;
    g_index++;
  }
  */
  printf("Alloc %6d, size = %8u\n", loc, (unsigned)size);
  LeaveCriticalSection(&cs);
  if (!p)
    throw CNewException();
  return p;
 #else
  numAllocs++;
  int loc = numAllocs;
  if (size == 0)
    size = 1;
  void *p = malloc(size);
  /*
  if (g_index < kDebugSize)
  {
    a[g_index] = p;
    g_index++;
  }
  */
  printf("Alloc %6d, size = %8u\n", loc, (unsigned)size);
  if (!p)
    throw CNewException();
  return p;
 #endif
}

void
#ifdef _MSC_VER
__cdecl
#endif
operator delete(void *p) throw()
{
  if (!p)
    return;
 #ifdef _WIN32
  EnterCriticalSection(&cs);
  /*
  for (int i = 0; i < g_index; i++)
    if (a[i] == p)
      a[i] = 0;
  */
  HeapFree(GetProcessHeap(), 0, p);
  // if (numAllocs == 0) numAllocs = numAllocs; // ERROR
  numAllocs--;
  // if (numAllocs == 0) numAllocs = numAllocs; // OK: all objects were deleted
  printf("Free %d\n", numAllocs);
  LeaveCriticalSection(&cs);
 #else
  free(p);
  numAllocs--;
  printf("Free %d\n", numAllocs);
 #endif
}

void
#ifdef _MSC_VER
__cdecl
#endif
operator delete(void *p, size_t n) throw();
void
#ifdef _MSC_VER
__cdecl
#endif
operator delete(void *p, size_t n) throw()
{
  UNUSED_VAR(n)
  printf("delete_WITH_SIZE=%u, ptr = %p\n", (unsigned)n, p);
  operator delete(p);
}

/*
void *
#ifdef _MSC_VER
__cdecl
#endif
operator new[](size_t size)
{
  printf("operator_new[] : ");
  return operator new(size);
}

void
#ifdef _MSC_VER
__cdecl
#endif
operator delete(void *p, size_t sz) throw();

void
#ifdef _MSC_VER
__cdecl
#endif
operator delete(void *p, size_t sz) throw()
{
  if (!p)
    return;
  printf("operator_delete_size : size=%d  : ", (unsigned)sz);
  operator delete(p);
}

void
#ifdef _MSC_VER
__cdecl
#endif
operator delete[](void *p) throw()
{
  if (!p)
    return;
  printf("operator_delete[] : ");
  operator delete(p);
}

void
#ifdef _MSC_VER
__cdecl
#endif
operator delete[](void *p, size_t sz) throw();

void
#ifdef _MSC_VER
__cdecl
#endif
operator delete[](void *p, size_t sz) throw()
{
  if (!p)
    return;
  printf("operator_delete_size[] : size=%d  : ", (unsigned)sz);
  operator delete(p);
}
*/

#endif

/*
int MemErrorVC(size_t)
{
  throw CNewException();
  // return 1;
}
CNewHandlerSetter::CNewHandlerSetter()
{
  // MemErrorOldVCFunction = _set_new_handler(MemErrorVC);
}
CNewHandlerSetter::~CNewHandlerSetter()
{
  // _set_new_handler(MemErrorOldVCFunction);
}
*/
